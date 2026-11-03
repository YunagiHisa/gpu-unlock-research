#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/run-before-common.sh"
BENCH=gemm
expected_commit=
initial_untracked=
if [[ "$EXECUTE" == 1 ]]; then
    expected_commit=$(git rev-parse HEAD)
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'Tracked files are dirty; measurement refused.' >&2; exit 8; }
    git ls-files --error-unmatch scripts/run-before-gemm.sh scripts/run-before-common.sh scripts/check-gemm-result.py benchmarks/gemm.py benchmarks/core.py benchmarks/metadata.py >/dev/null
    initial_untracked=$(git ls-files --others --exclude-standard)
fi
prepare
attempted=0
valid=0
duration_failed=0
planned=72
[[ "$KIND" != smoke ]] || planned=8
save_totals() {
    if [[ "$EXECUTE" == 1 ]]; then
        printf 'planned=%s\nattempted=%s\nvalid=%s\nfailed=%s\nduration_failed=%s\nunattempted=%s\n' "$planned" "$attempted" "$valid" "$((attempted-valid))" "$duration_failed" "$((planned-attempted))" > "$OUT/batch-summary.txt"
    fi
}
trap save_totals EXIT
if [[ "$EXECUTE" == 1 ]]; then
    "$PYTHON" -c 'import torch, pynvml; print(torch.__version__, torch.version.cuda)' > "$OUT/dependencies.txt" 2>&1 || { echo 'Missing GPU dependencies; see dependencies.txt' >&2; exit 5; }
    cp scripts/check-gemm-result.py "$OUT/"
    printf 'temperature_scope=measured_GPU_only; other_GPU_temperature_recorded\nexecution_kind=diagnostic strict_preflight=true\nduration_headroom_factor=1.05 max_samples=10000\norder=round_then_GPU_then_precision_then_size\nduration_shortfall_policy=first_per_GPU_mode_shape_continue_without_retry; second_stop_batch\nbefore_state_review=BF16_8192_TFLOPS_above_35_stops_without_asserting_unlock\n' >> "$OUT/conditions.txt"
    git status --short --untracked-files=all > "$OUT/git-state-at-start.txt"
    dkms status > "$OUT/dkms.txt"
    modinfo gpu_unlock > "$OUT/disk-module.txt"
    cat /sys/module/gpu_unlock/version > "$OUT/resident-module-version.txt"
    nvidia-smi --query-gpu=index,uuid,pci.bus_id,driver_version,temperature.gpu,memory.used,memory.free,utilization.gpu --format=csv > "$OUT/gpus-before.csv"
    OUT="$OUT" bash "$0" "--$KIND" > "$OUT/planned-commands.sh"
    python3 - "$OUT/planned-commands.sh" "$KIND" > "$OUT/plan-validation.txt" <<'PY'
import pathlib, shlex, sys
commands = [shlex.split(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
full = sys.argv[2] == 'full'
expected = [(g, mode, size, r) for r in ((1,2,3) if full else (1,))
            for g in (0,1) for mode in ('fp16','bf16','tf32','fp32')
            for size in ((2048,4096,8192) if full else (128,))]
observed, outputs = [], []
for cmd in commands:
    def arg(key): return cmd[cmd.index(key)+1]
    output = arg('--output'); outputs.append(output)
    repeat = int(pathlib.Path(output).stem.rsplit('-r',1)[1])
    size = int(arg('--m'))
    assert int(arg('--n')) == size == int(arg('--k'))
    assert arg('--execution-kind') == 'diagnostic' and '--strict-preflight' in cmd
    assert arg('--telemetry') == 'on' and arg('--max-start-temp') == '50' and arg('--cooldown-timeout-s') == '180'
    if full: assert arg('--min-timed-seconds') == '2'
    observed.append((int(arg('--gpu')), arg('--mode'), size, repeat))
assert observed == expected and len(set(outputs)) == len(outputs)
print('Validated %s unique sequential commands; %s configurations per round.' % (len(commands),24 if full else 8))
PY
    printf 'configuration\tclassification\treason\n' > "$OUT/result-checks.tsv"
    printf 'sequence\tround\tgpu\tmode\tsize\toutput\n' > "$OUT/execution-order.tsv"
fi
failed=0
declare -A duration_shortfalls=()
sizes=(2048 4096 8192); repeats=(1 2 3)
[[ "$KIND" != smoke ]] || { sizes=(128); repeats=(1); }
for repeat in "${repeats[@]}"; do
    for gpu in 0 1; do
        for mode in fp16 bf16 tf32 fp32; do
            for size in "${sizes[@]}"; do
                id="gpu${gpu}-${mode}-${size}-r${repeat}"
                output="${OUT:-results/PLAN}/$id.json"
                cmd=("$PYTHON" -m benchmarks.gemm --gpu "$gpu" --requested-state "$STATE" --execution-kind diagnostic --strict-preflight --mode "$mode" --m "$size" --n "$size" --k "$size" --telemetry on --max-start-temp 50 --cooldown-timeout-s 180 --output "$output")
                if [[ "$KIND" == smoke ]]; then
                    cmd+=(--warmup-seconds 0.01 --samples 3)
                else
                    cmd+=(--min-timed-seconds 2)
                fi
                rc=0
                MEASURE_GPU=$gpu
                if [[ "$EXECUTE" == 1 ]]; then
                    if [[ "$(git rev-parse HEAD)" != "$expected_commit" || -n "$(git status --porcelain --untracked-files=no)" || "$(git ls-files --others --exclude-standard)" != "$initial_untracked" ]]; then
                        printf '%s\t8\tGit state changed; measurement not started\n' "$id" >> "$OUT/status.tsv"
                        exit 8
                    fi
                    attempted=$((attempted+1))
                    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$attempted" "$repeat" "$gpu" "$mode" "$size" "$output" >> "$OUT/execution-order.tsv"
                fi
                run_one "$id" "${cmd[@]}" || rc=$?
                [[ "$EXECUTE" == 1 ]] || continue
                if [[ ! -e "$output" ]]; then
                    failure_json "$rc"
                fi
                decision=0
                reason=$(python3 scripts/check-gemm-result.py "$output" "$rc" "$gpu" "$mode" "$size" "$KIND" "$STATE" "$expected_commit") || decision=$?
                printf '%s\t%s\t%s\n' "$id" "$decision" "$reason" >> "$OUT/result-checks.tsv"
                if (( decision == 0 )); then
                    valid=$((valid+1))
                elif (( decision == 7 )); then
                    failed=1
                    duration_failed=$((duration_failed+1))
                    config="$gpu/$mode/$size"
                    count=$((${duration_shortfalls[$config]:-0}+1))
                    duration_shortfalls[$config]=$count
                    printf '%s: duration shortfall %s for this configuration; no retry\n' "$id" "$count"
                    if (( count >= 2 )); then
                        printf '%s\t7\tsecond duration shortfall for same GPU/mode/shape; batch stopped\n' "$id" >> "$OUT/status.tsv"
                        exit 7
                    fi
                else
                    printf '%s\t%s\t%s; batch stopped\n' "$id" "$decision" "$reason" >> "$OUT/status.tsv"
                    exit "$decision"
                fi
            done
        done
    done
done
if [[ "$EXECUTE" == 1 && "$KIND" == smoke && "$failed" == 0 ]]; then touch "$OUT/smoke-passed"; fi
exit "$failed"
