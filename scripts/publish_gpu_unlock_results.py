"""Extract an explicit public schema, then summarize it using only the stdlib.

Private input locations are supplied at run time; they are never written to outputs.
Without input arguments, regenerate comparisons and report tables from public CSVs.
This program imports no measurement module and executes no GPU operation.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
from pathlib import Path

PHASES = ('before', 'after', 'reverted')
MODES = ('fp16', 'bf16', 'tf32', 'fp32')
MODELS = {
    'Qwen3.8-27B-Q4_K_M.gguf': ('27b-q4', 'Qwen3.8-27B', 'Q4_K_M'),
    'Qwen_Qwen3-8B-Q4_K_M.gguf': ('8b-q4', 'Qwen3-8B', 'Q4_K_M'),
    'Qwen_Qwen3-8B-bf16.gguf': ('8b-bf16', 'Qwen3-8B', 'BF16'),
}
COMMIT = '9bab18fab9d986b61f90287d4eadeadf9f66c41a'
METRICS = ('temperature_c', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz')
EVENTS = ('sw_power_cap', 'hw_slowdown', 'sw_thermal_slowdown', 'hw_thermal_slowdown')


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f, skipinitialspace=True))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load(path):
    return json.loads(path.read_text())


def normalized_command(line):
    args = shlex.split(line)
    for option in ('--output', '--requested-state'):
        if option in args:
            args[args.index(option)+1] = '<VALUE>'
    return args


def extract(sources, output):
    gemm, llama = [], []
    identities, commands, llama_commands, precisions = [], [], [], {}
    toolkit_records = []
    artifact_maps = []
    for source in sources:
        artifact_maps.append({Path(line.split(None, 1)[1].strip()).name: line.split(None, 1)[0]
                              for line in (source/'expected-sha256.txt').read_text().splitlines()})
    assert artifact_maps[0] == artifact_maps[1] == artifact_maps[2]
    for phase, source in zip(PHASES, sources):
        prefix = source if phase == 'before' else source/'full'
        dirs = list(prefix.glob(f'{phase}-gemm-full-*'))
        assert len(dirs) == 1, 'Expected one complete GEMM batch'
        batch = dirs[0]
        devices = read_csv(batch/'gpus-before.csv')
        identities.append({int(r['index']): r['uuid'] for r in devices})
        commands.append([normalized_command(s) for s in (batch/'commands.sh').read_text().splitlines()])
        order = read_csv_tsv(batch/'execution-order.tsv')
        assert len(order) == 72
        for sequence, entry in enumerate(order, 1):
            name = Path(entry['output']).name
            d = load(batch/name)
            gpu, mode, size = int(entry['gpu']), entry['mode'], int(entry['size'])
            assert d['status'] == 'draft_validated' and d['baseline_eligible'] is False
            assert d['identity_verified'] and d['accuracy']['passed']
            assert all(d['accuracy_checks'][p]['passed'] for p in ('pre', 'post'))
            assert d['environment']['guest_kernel'] == '6.8.0-147-generic'
            assert d['benchmark_git']['commit'] == COMMIT and d['benchmark_git']['tracked_dirty'] is False
            assert d['timed_measurement']['target_met'] and d['timed_measurement']['gpu_span_seconds'] >= 2
            assert d['timed_measurement']['duration_headroom_factor'] == 1.05
            assert d['preflight']['temperature']['temperature_c'] <= 50
            precisions.setdefault(mode, d['precision'])
            assert precisions[mode] == d['precision']
            timed = d['telemetry']['phases']['timed']
            row = dict(phase=phase, sequence=sequence, round=int(entry['round']), gpu=gpu,
                       precision=mode, matrix_size=size,
                       tflops=d['summary']['tflops_from_median_latency'],
                       accuracy_passed=True, baseline_eligible=False, measured_seconds=d['timed_measurement']['gpu_span_seconds'],
                       start_temperature_c=d['preflight']['temperature']['temperature_c'],
                       power_limit_w=d['device']['nvml']['power_limit_w']['value'])
            for metric in METRICS:
                summary = timed['summary'][metric]
                assert summary['status'] == 'available'
                for stat in ('mean', 'min', 'max'):
                    row[f'timed_{metric}_{stat}'] = summary[stat]
            row['event_samples'] = timed['clock_events']['available_samples']
            for event in EVENTS:
                row[event+'_samples'] = timed['clock_events']['observed_samples_by_reason'][event]
            gemm.append(row)
        process_dirs = list(prefix.glob(f'{phase}-llama-full-*'))
        assert len(process_dirs) == 21
        indexed = {}
        for directory in process_dirs:
            condition = (directory/'conditions.txt').read_text()
            match = re.search(r'round=(\d+) order=(\d+) test=(\w+)', condition)
            assert match and f'benchmark=llama state={phase} kind=full' in condition
            assert 'baseline_eligible=false' in condition
            round_no, test = int(match[1]), match[3]
            d, = load(directory/(test+'.json'))
            model = MODELS[Path(d['model_filename']).name]
            indexed[(model[0], round_no, test)] = (directory, d, model)
        phase_commands = []
        sequence = 0
        for model_key, tests in [('27b-q4', ('pp512', 'tg128', 'pp2048')),
                                 ('8b-q4', ('pp512', 'tg128')), ('8b-bf16', ('pp512', 'tg128'))]:
            for round_no in (1, 2, 3):
                for test in tests:
                    directory, d, model = indexed[(model_key, round_no, test)]
                    assert all(line.endswith(': OK') for line in (directory/'sha256-check.txt').read_text().splitlines())
                    for line in (directory/'sha256.txt').read_text().splitlines():
                        checksum, name = line.split(None, 1)
                        assert artifact_maps[0][Path(name.strip()).name] == checksum
                    sequence += 1
                    assert d['build_commit'] == '620d2225c' and d['build_number'] == 11669
                    assert len(d['samples_ts']) == 5 and all(math.isfinite(v) and v > 0 for v in d['samples_ts'])
                    phase_commands.append(shlex.split((directory/'commands.sh').read_text()))
                    condition = (directory/'conditions.txt').read_text()
                    for setting in ('CUDA_DEVICE_ORDER=PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES=0,1',
                                    'GGML_CUDA_ALLREDUCE=internal', 'LD_LIBRARY_PATH=/usr/local/cuda/lib64'):
                        assert setting in condition
                    starts = read_csv(directory/'before-gpu.csv')
                    assert len(starts) == 2 and max(int(r['temperature.gpu']) for r in starts) <= 50
                    row = dict(phase=phase, sequence=sequence, round=round_no, model=model[1],
                               quantization=model[2], benchmark=test,
                               benchmark_type='prefill' if test.startswith('pp') else 'token_generation',
                               mean_tok_s=d['avg_ts'], within_process_sd_tok_s=d['stddev_ts'], repeats=5, baseline_eligible=False)
                    for i, sample in enumerate(d['samples_ts'], 1):
                        row[f'repeat_{i}_tok_s'] = sample
                    telemetry = read_csv(directory/(test+'-telemetry.csv'))
                    clocks = read_csv(directory/'clocks-events.csv')
                    for gpu in (0, 1):
                        samples = [r for r in telemetry if int(r['index']) == gpu]
                        assert {r['uuid'] for r in samples} == {identities[-1][gpu]}
                        for field, metric in [('temperature.gpu', 'temperature_c'), ('power.draw [W]', 'power_w'),
                                              ('memory.used [MiB]', 'memory_used_mib')]:
                            values = [float(r[field].split()[0]) for r in samples]
                            row[f'gpu{gpu}_{metric}_mean'] = statistics.mean(values)
                            row[f'gpu{gpu}_{metric}_max'] = max(values)
                        cs = [r for r in clocks if int(r['index']) == gpu]
                        for field, metric in [('clocks.current.graphics [MHz]', 'graphics_clock_mhz'),
                                              ('clocks.current.memory [MHz]', 'memory_clock_mhz')]:
                            values = [float(r[field].split()[0]) for r in cs]
                            row[f'gpu{gpu}_{metric}_mean'] = statistics.mean(values)
                            row[f'gpu{gpu}_{metric}_max'] = max(values)
                        for event in EVENTS:
                            values = [r['clocks_event_reasons.'+event] for r in cs]
                            assert set(values) <= {'Active', 'Not Active'}
                            row[f'gpu{gpu}_{event}_samples'] = values.count('Active')
                        row[f'gpu{gpu}_clock_samples'] = len(cs)
                    llama.append(row)
        llama_commands.append(phase_commands)
        for cache in source.rglob('CMakeCache.txt'):
            text = cache.read_text()
            found = re.search(r'FIND_PACKAGE_MESSAGE_DETAILS_CUDAToolkit:INTERNAL=.*\[v([^()]+)\(\)\]', text)
            if found:
                toolkit_records.append(found[1])
    assert len(gemm) == 216 and len(llama) == 63
    assert identities[0] == identities[1] == identities[2]
    assert commands[0] == commands[1] == commands[2]
    assert llama_commands[0] == llama_commands[1] == llama_commands[2]
    assert set(toolkit_records) == {'12.8.61'}
    write_csv(output/'gpu-unlock-gemm-processes.csv', gemm)
    write_csv(output/'gpu-unlock-llama-processes.csv', llama)
    software = load(sources[1]/'python-software.json')
    package_names = ('torch', 'numpy', 'nvidia-ml-py', 'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12',
                     'nvidia-cuda-nvrtc-cu12', 'nvidia-cudnn-cu12', 'nvidia-nccl-cu12')
    environment = dict(kernel='6.8.0-147-generic', nvidia_driver='580.178.04', python=software['python'],
                       pytorch=software['pytorch'], pytorch_cuda_runtime=software['pytorch_cuda'],
                       driver_cuda_compatibility='13.0', llama_build=11669, llama_commit='620d2225c',
                       llama_build_toolkit='12.8.61', toolkit_evidence='Saved CMake package detection records',
                       toolkit_record_count=len(toolkit_records), host_compiler='GNU 13.3.0',
                       gemm_commit=COMMIT, identity_verified_across_phases=True, baseline_eligible=False,
                       artifact_sha256=artifact_maps[0],
                       precision_controls=precisions,
                       packages={name: software['packages'][name] for name in package_names})
    (output/'gpu-unlock-environment.json').write_text(json.dumps(environment, indent=2)+'\n')


def read_csv_tsv(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def summarize(data):
    for kind, key_fields, metric, count in [('gemm', ('gpu', 'precision', 'matrix_size'), 'tflops', 24),
                                            ('llama', ('model', 'quantization', 'benchmark', 'benchmark_type'), 'mean_tok_s', 7)]:
        groups = {}
        for row in read_csv(data/f'gpu-unlock-{kind}-processes.csv'):
            key = tuple(row[k] for k in key_fields)
            groups.setdefault(key, {p: [] for p in PHASES})[row['phase']].append(row)
        assert len(groups) == count
        output = []
        for key, phases in sorted(groups.items()):
            r = dict(zip(key_fields, key))
            r['unit'] = 'TFLOPS' if kind == 'gemm' else 'tok/s'
            r['baseline_eligible'] = False
            for phase in PHASES:
                processes = phases[phase]
                assert len(processes) == 3 and {p['round'] for p in processes} == {'1', '2', '3'}
                values = [float(p[metric]) for p in processes]
                r[phase+'_mean'] = statistics.mean(values)
                r[phase+'_sd'] = statistics.stdev(values)
                r[phase+'_process_count'] = len(values)
                if kind == 'gemm':
                    assert all(p['accuracy_passed'] == 'True' for p in processes)
                    r[phase+'_accuracy_pass_count'] = 3
                    r[phase+'_power_limit_w'] = float(processes[0]['power_limit_w'])
                    for m in METRICS:
                        r[phase+'_'+m+'_mean'] = statistics.mean(float(p['timed_'+m+'_mean']) for p in processes)
                        r[phase+'_'+m+'_max'] = max(float(p['timed_'+m+'_max']) for p in processes)
                    for event in EVENTS:
                        r[phase+'_'+event+'_sample_fraction'] = sum(int(p[event+'_samples']) for p in processes)/sum(int(p['event_samples']) for p in processes)
                else:
                    assert all(int(p['repeats']) == 5 for p in processes)
                    r['repeats_per_process'] = 5
            b, a, v = (r[p+'_mean'] for p in PHASES)
            r.update(after_minus_before=a-b, reverted_minus_before=v-b, reverted_minus_after=v-a,
                     after_vs_before_percent=100*(a/b-1), reverted_vs_before_percent=100*(v/b-1),
                     reverted_vs_after_percent=100*(v/a-1))
            output.append(r)
        write_csv(data/f'gpu-unlock-{kind}-comparison.csv', output)
    return read_csv(data/'gpu-unlock-gemm-comparison.csv'), read_csv(data/'gpu-unlock-llama-comparison.csv')


def report_tables(gemm, llama, report):
    for kind, rows in [('GEMM', gemm), ('LLAMA', llama)]:
        headers = ['GPU', 'Precision', 'Size'] if kind == 'GEMM' else ['Model', 'Quantization', 'Test']
        headers += ['Before mean ± SD', 'After mean ± SD', 'Reverted mean ± SD', 'After / Before', 'Reverted / Before']
        lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
        for r in rows:
            cells = [r['gpu'], r['precision'].upper() if r['precision'] != 'fp32' else 'FP32 IEEE', r['matrix_size']] if kind == 'GEMM' else [r['model'], r['quantization'], r['benchmark']]
            cells += [f"{float(r[p+'_mean']):.3f} ± {float(r[p+'_sd']):.3f}" for p in PHASES]
            cells += [f"{float(r[k]):+.2f}%" for k in ('after_vs_before_percent', 'reverted_vs_before_percent')]
            lines.append('| '+' | '.join(cells)+' |')
        start, end = f'<!-- {kind}_TABLE_START -->', f'<!-- {kind}_TABLE_END -->'
        if report.exists():
            text = report.read_text()
            assert start in text and end in text
            text = text.split(start)[0]+start+'\n\n'+'\n'.join(lines)+'\n\n'+end+text.split(end)[1]
            report.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for phase in PHASES:
        p.add_argument('--'+phase, type=Path, help='Private input folder; optional when public process CSVs already exist')
    p.add_argument('--data-dir', type=Path, default=Path('docs/data'))
    p.add_argument('--report', type=Path, default=Path('docs/gpu-unlock-benchmark.md'))
    args = p.parse_args()
    supplied = [getattr(args, phase) for phase in PHASES]
    assert all(supplied) or not any(supplied), 'Supply all three private inputs or none'
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if all(supplied):
        extract(supplied, args.data_dir)
    gemm, llama = summarize(args.data_dir)
    report_tables(gemm, llama, args.report)
    print('Validated 216 GEMM process records, 63 llama process records, 24 GEMM configurations and 7 llama conditions.')


if __name__ == '__main__':
    main()
