import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gemm_result', ROOT/'scripts/check-gemm-result.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def valid_result():
    return dict(baseline_eligible=False, status='draft_validated', identity_verified=True,
        benchmark_git=dict(commit='0'*40, tracked_dirty=False, dirty=False, untracked_files=[]),
        device={'cuda_index': 0}, mode='bf16', shape=dict(m=8192,n=8192,k=8192),
        preflight={'status':'ready'}, accuracy={'passed':True},
        accuracy_checks={p:{'passed':True} for p in ('pre','post')}, precision_mode_suspect=False,
        compute_processes={p:dict(status='available',other_processes=dict(status='available',value=0)) for p in ('before','after')},
        summary={'tflops_from_median_latency':26.7},
        timed_measurement=dict(target_met=True,min_timed_seconds=2,gpu_span_seconds=2.1,selected_samples=100))


class GemmPolicyTests(unittest.TestCase):
    def test_invalid_git_state_is_never_valid(self):
        d=valid_result()
        self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','before','1'*40)[0],8)
        d['benchmark_git']['tracked_dirty']=True
        self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','before')[0],8)
        d.pop('benchmark_git')
        self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','before')[0],8)

    def test_full_plan_is_exactly_three_rounds_of_24(self):
        p=subprocess.run(['bash','scripts/run-before-gemm.sh','--full'],cwd=ROOT,capture_output=True,text=True,check=True)
        commands=[shlex.split(s) for s in p.stdout.splitlines()]
        expected=[(g,m,n,r) for r in (1,2,3) for g in (0,1)
                  for m in ('fp16','bf16','tf32','fp32') for n in (2048,4096,8192)]
        actual=[]
        for cmd in commands:
            def arg(key):return cmd[cmd.index(key)+1]
            repeat=int(Path(arg('--output')).stem.rsplit('-r',1)[1])
            actual.append((int(arg('--gpu')),arg('--mode'),int(arg('--m')),repeat))
            self.assertEqual(arg('--execution-kind'),'diagnostic')
            self.assertIn('--strict-preflight',cmd)
        self.assertEqual(actual,expected)
        self.assertEqual(len({tuple(x) for x in commands}),72)

    def test_before_guard_does_not_assume_fp16_or_after_rate(self):
        d=valid_result()
        self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','before')[0],0)
        d['summary']['tflops_from_median_latency']=51
        self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','before')[0],8)
        self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','after')[0],0)
        d['mode']='fp16'
        self.assertEqual(policy.classify(d,0,0,'fp16',8192,'full','before')[0],0)

    def test_duration_requires_correctness_and_consistent_exit(self):
        d=valid_result();d['timed_measurement'].update(target_met=False,gpu_span_seconds=1.9);d['status']='duration_target_not_met'
        self.assertEqual(policy.classify(d,7,0,'bf16',8192,'full','before')[0],7)
        self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','before')[0],8)
        for field,value in [('identity_verified',False),('precision_mode_suspect',True),('accuracy',{'passed':False})]:
            bad=copy.deepcopy(d);bad[field]=value
            self.assertEqual(policy.classify(bad,7,0,'bf16',8192,'full','before')[0],8)
        d['compute_processes']['after']['other_processes']['value']=1
        self.assertEqual(policy.classify(d,7,0,'bf16',8192,'full','before')[0],8)

    def test_timing_flag_cannot_override_actual_duration_or_sample_cap(self):
        for key,value in [('gpu_span_seconds',1.99),('gpu_span_seconds',float('nan')),('selected_samples',10001)]:
            d=valid_result();d['timed_measurement'][key]=value
            self.assertEqual(policy.classify(d,0,0,'bf16',8192,'full','before')[0],8)

    def run_mock_batch(self, scenario):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'scripts').mkdir();(root/'bin').mkdir()
            for file in ('run-before-common.sh','run-before-gemm.sh','check-gemm-result.py'):
                shutil.copy2(ROOT/'scripts'/file,root/'scripts'/file)
            (root/'smoke').mkdir();(root/'smoke/smoke-passed').touch()
            (root/'smoke/conditions.txt').write_text('benchmark=gemm state=before kind=smoke\n')
            (root/'note').write_text('CPU-only mock fixture; no GPU execution\n')
            fixture=valid_result();fixture.update(mode='fp16',shape=dict(m=2048,n=2048,k=2048))
            (root/'fixture.json').write_text(json.dumps(fixture))
            tool=root/'bin/mock-tool'
            tool.write_text('#!'+sys.executable+'\n'+r'''
import json,os,pathlib,sys,time
name=pathlib.Path(sys.argv[0]).name;args=sys.argv[1:]
if name=='nvidia-smi':
    if any('query-compute-apps' in a for a in args):pass
    elif '-i' in args:
        print('40' if args[args.index('-i')+1]=='0' or os.environ['SCENARIO']!='first_then_accuracy' else 'N/A')
    else:print('0, 40, 1, 0\n1, '+('65' if os.environ['SCENARIO']=='first_then_accuracy' else '40')+', 1, 0')
elif name=='git':
    if 'rev-parse' in args:print('0'*40)
    elif 'status' in args and os.environ['SCENARIO']=='dirty':print(' M benchmarks/core.py')
elif name=='fake-python':
    if '-c' in args:sys.exit(0)
    active=pathlib.Path('.fake-active')
    try:fd=os.open(active,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:sys.exit(99)
    os.close(fd);time.sleep(0.005)
    out=pathlib.Path(args[args.index('--output')+1]);repeat=int(out.stem.rsplit('-r',1)[1])
    d=json.load(open('fixture.json'));rc=0
    gpu=int(args[args.index('--gpu')+1]);mode=args[args.index('--mode')+1];size=int(args[args.index('--m')+1])
    d.update(device={'cuda_index':gpu},mode=mode,shape=dict(m=size,n=size,k=size))
    first_config=(gpu,mode,size)==(0,'fp16',2048)
    if os.environ['SCENARIO']!='all_valid' and first_config:
        rc=7;d['status']='duration_target_not_met';d['timed_measurement'].update(target_met=False,gpu_span_seconds=1.9)
    elif os.environ['SCENARIO']=='first_then_accuracy' and size==4096:
        rc=2;d['status']='accuracy_failed';d['accuracy']['passed']=False
    out.write_text(json.dumps(d));active.unlink();sys.exit(rc)
''')
            tool.chmod(0o755)
            for executable in ('nvidia-smi','git','dkms','modinfo','journalctl','fake-python','sleep'):
                (root/'bin'/executable).symlink_to(tool.name)
            env=dict(os.environ,PATH=str(root/'bin')+os.pathsep+os.environ['PATH'],
                PYTHON=str(root/'bin/fake-python'),OPERATOR_READY='1',STATE='before',
                STATE_EVIDENCE_FILE=str(root/'note'),SMOKE_DIR=str(root/'smoke'),SCENARIO=scenario)
            p=subprocess.run(['bash','scripts/run-before-gemm.sh','--full','--execute'],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            outputs=list((root/'results').glob('before-gemm-full-*'))
            if not outputs:
                return p.returncode,[],''
            output,=outputs
            commands=(output/'commands.sh').read_text().splitlines()
            return p.returncode,commands,(output/'batch-summary.txt').read_text()

    def test_second_shortfall_stops_at_same_configuration_in_round_two(self):
        rc,commands,totals=self.run_mock_batch('two_shortfalls')
        self.assertEqual(rc,7)
        self.assertEqual(len(commands),25)
        self.assertIn('gpu0-fp16-2048-r2.json',commands[-1])
        self.assertIn('valid=23\nfailed=2\nduration_failed=2',totals)

    def test_first_shortfall_continues_scheduled_repeat_then_accuracy_stops(self):
        rc,commands,totals=self.run_mock_batch('first_then_accuracy')
        self.assertEqual(rc,8)
        self.assertEqual(len(commands),2)
        self.assertIn('gpu0-fp16-2048-r1.json',commands[0])
        self.assertIn('gpu0-fp16-4096-r1.json',commands[1])
        self.assertIn('valid=0\nfailed=2\nduration_failed=1',totals)

    def test_all_72_mock_processes_are_serial_and_counted(self):
        rc,commands,totals=self.run_mock_batch('all_valid')
        self.assertEqual(rc,0)
        self.assertEqual(len(commands),72)
        self.assertIn('valid=72\nfailed=0\nduration_failed=0',totals)

    def test_dirty_tracked_tree_prevents_any_process(self):
        rc,commands,_=self.run_mock_batch('dirty')
        self.assertEqual(rc,8)
        self.assertEqual(commands,[])


if __name__=='__main__':
    unittest.main()
