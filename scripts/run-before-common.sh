#!/usr/bin/env bash
# Small shared shell routines; sourced by the two benchmark launchers.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
STATE=${STATE:-before}
case "$STATE" in before|after|reverted) ;; *) echo 'Invalid STATE' >&2; exit 2;; esac
KIND=smoke
EXECUTE=0
for arg in "$@"; do
    case "$arg" in
        --smoke) KIND=smoke;;
        --full) KIND=full;;
        --execute) EXECUTE=1;;
        *) echo "Unknown argument: $arg" >&2; exit 2;;
    esac
done
PYTHON=${PYTHON:-python3}
child= sampler= monitor= current=
failure_json() {
    if [[ "${BENCH:-}" == gemm && -n "${OUT:-}" && -n "$current" && ! -e "$OUT/$current.json" ]]; then
        python3 - "$OUT/$current.json" "$1" <<'PY'
import json, subprocess, sys
def git(*args):
    p = subprocess.run(['git', *args], text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else None
tracked = git('status', '--porcelain', '--untracked-files=no')
status = git('status', '--porcelain', '--untracked-files=normal')
data = dict(status='launcher_failure', exit_code=int(sys.argv[2]), baseline_eligible=False,
            benchmark_git=dict(commit=git('rev-parse','HEAD'),
                tracked_dirty=bool(tracked) if tracked is not None else None,
                dirty=bool(status) if status is not None else None,
                untracked_files=(git('ls-files','--others','--exclude-standard') or '').splitlines()))
with open(sys.argv[1], 'x') as f:
    json.dump(data, f, indent=2); f.write('\n')
PY
    fi
}
interrupt() {
    trap '' INT TERM
    [[ -z "$monitor" ]] || kill -TERM "$monitor" 2>/dev/null || true
    [[ -z "$monitor" ]] || wait "$monitor" 2>/dev/null || true
    [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true
    [[ -z "$child" ]] || wait "$child" 2>/dev/null || true
    [[ -z "$sampler" ]] || kill -TERM "$sampler" 2>/dev/null || true
    [[ -z "$sampler" ]] || wait "$sampler" 2>/dev/null || true
    [[ -z "${OUT:-}" ]] || printf '%s\t130\tinterrupted\n' "$current" >> "$OUT/status.tsv"
    failure_json 130
    exit 130
}
trap interrupt INT TERM
command_line() { printf '%q ' "$@"; printf '\n'; }
prepare() {
    [[ "$EXECUTE" == 1 ]] || return 0
    [[ "${OPERATOR_READY:-}" == 1 ]] || { echo 'Operator must confirm service scheduling and GPU availability: OPERATOR_READY=1' >&2; exit 2; }
    [[ -s "${STATE_EVIDENCE_FILE:-}" ]] || { echo 'Set STATE_EVIDENCE_FILE to a reviewed, nonempty operator state/history note.' >&2; exit 2; }
    if [[ "$STATE" == before && -d /sys/module/gpu_unlock ]]; then
        local targets
        targets=$(cat /sys/module/gpu_unlock/parameters/targets) || exit 6
        if [[ -n "$targets" && "$targets" != '(null)' ]]; then
            echo 'Before candidate blocked: gpu_unlock has target GPUs configured.' >&2
            exit 6
        fi
        echo 'gpu_unlock loaded without targets: proceeding only as an inferred Before diagnostic candidate.'
    fi
    if [[ "$KIND" == full ]]; then
        [[ -f "${SMOKE_DIR:-}/smoke-passed" && -f "${SMOKE_DIR:-}/conditions.txt" ]] || { echo 'Full run requires SMOKE_DIR from a successful smoke run.' >&2; exit 2; }
        rg -qx "benchmark=$BENCH state=$STATE kind=smoke" "$SMOKE_DIR/conditions.txt" || exit 2
    fi
    mkdir -p results
    OUT=$(mktemp -d "$ROOT/results/${STATE}-${BENCH}-${KIND}-XXXXXXXX")
    exec 9>"$ROOT/results/benchmark.lock"
    flock -n 9 || { echo 'Another local benchmark launcher holds the lock.' >&2; exit 6; }
    printf 'benchmark=%s state=%s kind=%s\nCUDA_DEVICE_ORDER=PCI_BUS_ID\nmax_start_temp_c=50 cooldown_timeout_s=180\nbaseline_eligible=false\nverification_status=unconfirmed\ninitial_state_confidence=unknown\n' "$BENCH" "$STATE" "$KIND" > "$OUT/conditions.txt"
    cp -- "$STATE_EVIDENCE_FILE" "$OUT/operator-state.txt"
    git rev-parse HEAD > "$OUT/repository-head.txt"
    git diff -- benchmarks scripts > "$OUT/working.diff"
    cp -- scripts/run-before-*.sh "$OUT/"
    printf 'configuration\texit_code\tnote\n' > "$OUT/status.tsv"
    date -u +%FT%TZ > "$OUT/started-utc.txt"
    cat /proc/cmdline > "$OUT/kernel-cmdline.txt"
    cat /proc/modules > "$OUT/modules.txt"
    for parameter in /sys/module/gpu_unlock/parameters/*; do
        [[ -f "$parameter" ]] || continue
        printf '%s=' "${parameter##*/}" >> "$OUT/unlock-parameters.txt"
        cat "$parameter" >> "$OUT/unlock-parameters.txt"
    done
    journalctl -k -b --no-pager > "$OUT/kernel-journal.txt" 2> "$OUT/kernel-journal.err" || true
    echo "Results: $OUT"
}
no_compute() {
    local processes
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader) || return 6
    printf '%s\n' "$processes"
    [[ -z "$processes" ]]
}
idle() {
    local rows
    no_compute || return 6
    rows=$(nvidia-smi --query-gpu=index,temperature.gpu,memory.used,utilization.gpu --format=csv,noheader,nounits) || return 6
    printf '%s\n' "$rows"
    # Fail closed for graphics use, unreported work, or unknown fields, too.
    awk -F, 'NF!=4 {exit 1} {for(i=1;i<=4;i++) {gsub(/ /,"",$i); if($i !~ /^[0-9]+$/) exit 1} if($3>16 || $4>0) exit 1; n++} END {if(n!=2) exit 1}' <<< "$rows"
}
cool() {
    local start=$SECONDS temp gpu ready
    while :; do
        idle >> "$OUT/$current-preflight.log" 2>&1 || return 6
        ready=1
        for gpu in "$@"; do
            temp=$(nvidia-smi -i "$gpu" --query-gpu=temperature.gpu --format=csv,noheader,nounits) || return 6
            [[ "$temp" =~ ^[0-9]+$ ]] || return 6
            (( temp <= 50 )) || ready=0
        done
        (( ready )) && return 0
        (( SECONDS-start < 180 )) || return 6
        sleep 1
    done
}
run_one() {
    current=$1; shift
    command_line "$@"
    [[ "$EXECUTE" == 1 ]] || return 0
    command_line "$@" >> "$OUT/commands.sh"
    local -a cooling_gpus=(0 1)
    if [[ "${BENCH:-}" == gemm ]]; then
        [[ "${MEASURE_GPU:-}" =~ ^[01]$ ]] || return 2
        cooling_gpus=("$MEASURE_GPU")
    fi
    if ! cool "${cooling_gpus[@]}"; then
        printf '%s\t6\tGPU contention, unknown state, or cooldown timeout; stopped\n' "$current" >> "$OUT/status.tsv"
        return 6
    fi
    : > "$OUT/$current.stdout"
    : > "$OUT/$current.stderr"
    nvidia-smi --query-gpu=timestamp,index,uuid,pci.bus_id,temperature.gpu,power.draw,memory.used,memory.free,utilization.gpu --format=csv -lms 100 > "$OUT/$current-telemetry.csv" 2> "$OUT/$current-telemetry.err" &
    sampler=$!
    "$@" > "$OUT/$current.stdout" 2> "$OUT/$current.stderr" &
    child=$!
    # Check for competing CUDA processes while this independent process runs.
    # The sampler is read-only; no unrelated process is signalled.
    local measured_pid=$child
    (
        trap 'exit 0' INT TERM
        while kill -0 "$measured_pid" 2>/dev/null; do
            processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || processes=unknown
            if ! awk -v own="$measured_pid" 'NF {gsub(/ /, ""); if($0!=own) exit 1}' <<< "$processes"; then
                printf '%s\n' "$processes" > "$OUT/$current-contention.txt"
                kill -TERM "$measured_pid" 2>/dev/null || true
                exit 0
            fi
            sleep 1
        done
    ) &
    monitor=$!
    local rc=0
    wait "$child" || rc=$?
    child=
    kill -TERM "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    monitor=
    kill -TERM "$sampler" 2>/dev/null || true
    wait "$sampler" 2>/dev/null || true
    sampler=
    printf '%s\t%s\tcompleted\n' "$current" "$rc" >> "$OUT/status.tsv"
    if [[ -e "$OUT/$current-contention.txt" ]]; then
        printf '%s\t6\tcontention during measurement; data requires review; stopped\n' "$current" >> "$OUT/status.tsv"
        failure_json 6
        return 6
    fi
    if ! no_compute >> "$OUT/$current-postflight.log" 2>&1; then
        printf '%s\t6\tpostflight contention; data requires review; stopped\n' "$current" >> "$OUT/status.tsv"
        return 6
    fi
    return "$rc"
}
