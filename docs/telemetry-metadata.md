# GEMM diagnostics and telemetry design

## Inputs and preparation

Each invocation measures one CUDA device, precision mode, and M/N/K shape.
`--gpu`, `--mode {fp16,bf16,tf32,fp32}`, and
`--requested-state {before,after,reverted}` are required. Use
`CUDA_DEVICE_ORDER=PCI_BUS_ID` for reproducible enumeration.

| Option | Behavior |
|---|---|
| `--m`, `--n`, `--k` | Positive dimensions, each defaulting to 2048 |
| `--samples` | Fixed count, default 30; maximum 10,000 timing pairs |
| `--warmup-seconds` | Positive time-based warm-up, default 3 seconds |
| `--min-timed-seconds` | Optional duration target estimated from warm-up |
| `--seed` | Recorded deterministic input seed, default 2026 |
| `--telemetry {on,off}` | Default on; off disables worker polling but retains endpoints |
| `--telemetry-interval` | Positive target interval, default 0.1 seconds |
| `--max-start-temp`, `--cooldown-timeout-s` | Optional paired Celsius limit and nonnegative wait budget; no numeric defaults |
| `--execution-kind {formal,diagnostic}` | GEMM defaults to formal preparation; comparison defaults to diagnostic |
| `--experiment-metadata` | Allowlisted per-GPU operator metadata file |
| `--hmac-key-file` | Existing external pseudonym key |
| `--output` | Exclusive local JSON destination; otherwise stdout |

Formal preparation requires matching PCI and independent UUID evidence, a
passed configured temperature gate, and a known zero count of other compute
processes. Otherwise it saves a blocked diagnostic without GEMM. Diagnostic
mode explicitly records incomplete/contended conditions. A configured temperature
timeout blocks both modes. All results remain baseline-ineligible.

Cooldown passively polls NVML only while above the configured limit. Its result
records actual temperature, waiting time, checks, limit, budget, and outcome.
Process counts are read before cooldown, after waiting immediately before
measurement, and after measurement. Positive/unknown other-process counts add
explicit ineligibility reasons. No GPU setting or service is changed.

## Identity and external key management

The CUDA driver reads the device bound to the actual current context, including
visible-device remapping. Its PCI ID selects an NVML handle and is checked
against NVML PCI readback. CUDA/NVML UUIDs are independently compared in memory.
Both paths must match for verified identity. Contradictions stop before tensors
or GEMM with exit 4; missing evidence is explicit. Raw UUIDs, exception text,
PIDs, process names, and executable paths are not serialized.

Pseudonyms use HMAC-SHA256 over the UUID with prefix
`gpu-unlock-research/gpu-id/v1` followed by a null byte. The default key is
`$XDG_CONFIG_HOME/gpu-unlock-research/hmac.key`, falling back to
`~/.config/gpu-unlock-research/hmac.key`. It must be outside the repository,
32 binary bytes, an owned single-link mode-600 regular file inside an owned
mode-700 directory. Symlink key files/directories are rejected. No automatic
key creation, replacement, or unkeyed UUID fallback occurs.

Initialize explicitly once from the repository root. Existing files are refused
and no secret material is printed:

```bash
python - <<'PY'
import os
import secrets
import stat
from pathlib import Path
from benchmarks.identity import default_key_path

key = default_key_path()
if key.resolve().is_relative_to(Path.cwd().resolve()):
    raise SystemExit("Choose an external configuration directory")
os.umask(0o077)
key.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
info = key.parent.lstat()
if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
    raise SystemExit("Require an owned directory with mode 700")
descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as output:
    output.write(secrets.token_bytes(32))
print("Key initialized; back it up securely")
PY
```

Retain an exact encrypted offline/secret-store backup accessible only to the
experiment owner. Restore ownership and 600/700 permissions. Never paste a key
into command arguments, logs, JSON, tickets, or Git. Losing/replacing it prevents
matching old pseudonyms. Missing/unsafe keys still permit nonsecret diagnostics
with a JSON reason code and initialization guidance on stderr.

## Measurement and numerical validation

Use fixed-seed inputs and preallocated A/B/output tensors. FP16/BF16 intend
FP32 accumulation with rounded low-precision outputs. Disable FP16 accumulation
and FP16/BF16 reduced-precision reductions, read settings back, and fail on
unsupported/ineffective required controls. FP32 selects modern TF32/IEEE APIs
without legacy mixing. FP32 IEEE is the control mode in the research protocol.

For each sample, record a start event, enqueue `torch.mm(..., out=...)`, and
record stop on the same stream. Retain pairs and synchronize only the final
stop before reading elapsed times. Allocation, initialization, warm-up, FP64
checks, and telemetry summaries are outside per-operation intervals. Event
wrappers are preallocated; CUDA may initialize them lazily. These are queued GPU
operation times, not isolated synchronized calls or request latency.

FLOPs = `2*M*N*K`; TFLOPS = FLOPs / latency_seconds / 1e12. Retain raw timings,
median, IQR, and count. Optional duration uses warm-up CUDA span/operation count,
choosing at least --samples and at most 10,000 pairs. Record estimate, prediction,
count/cap, actual first-start/final-stop GPU span, wall span through final sync,
sum of per-operation times, and target outcome. No adaptive repeats/retries occur.

Check before warm-up and on the final timed output against FP64 products of the
quantized inputs, using `norm(C-Cref)/max(norm(Cref),1e-30)`. Full reference needs
both maximum dimension <= `--full-reference-max` (default 2048) and
`2*M*N*K <= --full-reference-max-flops` (default `2*2048^3`). Otherwise a
deterministic output tile of up to `--reference-tile` rows/columns (default 128)
retains the full K reduction. Tile seed is input seed+17. Method, checked shape,
reference FLOPs/budget, error, and limit are retained for each endpoint.

| Mode | Provisional normalized L2 upper limit |
|---|---:|
| FP16 | 1e-3 |
| BF16 | 5e-3 |
| TF32 | 1e-3 |
| FP32 IEEE | 2e-5 |

Equality passes; nonfinite errors fail and serialize as null. A failed pre-check
skips timing; a failed post-check preserves timings as diagnostics. TF32 error
below 2e-5 sets aggregate precision suspicion separately from numerical pass.

## Telemetry phases

Record monotonic read-start/read-completion offsets, source, SM/memory clocks
(MHz), power (W), temperature (Celsius), fan percentage, utilization percentage,
performance state, and raw clock-event masks. Missing values retain null and
availability/error categories, never fabricated zeros. Supported reason masks
are device metadata; raw masks retain unknown future bits.

| Phase | Included records |
|---|---|
| `idle_before` | Snapshot before warm-up; not a thermal-equilibrium assertion |
| `warmup` | Polls whose start and completion both lie inside warm-up |
| `timed` | Polls whose start and completion both lie inside queued timing |
| `post` | Snapshot after final timing sync, before post-validation |

Each phase records boundaries (seconds relative to sampler start), count,
status/reason, per-field available/missing counts and min/median/max/mean, and
reason-active observation counts. Empty phases/fields are unavailable with null
statistics. Cross-boundary reads remain raw but unassigned. Whole-collection
statistics are omitted because mixing idle/load phases is ambiguous and any
needed aggregate is reconstructible from the retained raw samples.

Violation counters retain before/after `violation_time_ns`, `reference_time_us`,
and `delta_ns`, covering collection including warm-up. Decreases invalidate a
delta as a reset. Clock-event getters fall back to legacy throttle-reason APIs
when absent. On exceptions, the worker is stopped/joined before NVML shutdown.

## Operator and environment metadata

Input is a strict allowlist requiring anonymous gpu_label, normalized pci_bus_id,
and cooling_state_id. Optional chronological cooling_history records date,
change, previous/current state IDs, operator_report provenance, and constrained
before/current numeric/boolean observations. Accepted changes are
thermal_paste_replaced/cooling_maintenance/unchanged_observation. Its final state
must match the current state ID. Real maintenance/measurement records belong in
experiment storage/discussion, not fabricated design examples.

Optional initial_state contains unlock_history, UTC documentation/attempt times,
power-cycle evidence, and residual-state suspicion. Never-attempted documented
history needs no extra cycle. Previously attempted GPUs need a completed host
shutdown, AC disconnection, and fresh power-on after their last attempt, with
UTC completion and documented method host_shutdown_ac_disconnect_power_on.
Residual-state suspicion gives low confidence; insufficient history is unknown.
Contradictions, non-UTC/future evidence timestamps, and invalid flags are rejected.

For Before, high confidence maps to not_applicable; low/unknown to unconfirmed.
After/reverted remain unconfirmed pending independent evidence review. The runner
never infers confirmed/failed from benchmark or module state. System evidence
records exact boot-parameter presence without value/full command line, module
presence, and accessible kernel marker counts. Denied logs are unavailable;
raw log lines are never output.

Host input accepts only Proxmox/host-kernel version strings and VFIO category
(dedicated_passthrough/shared/unknown). Read OS/CPU/kernel/Git metadata locally.
Device metadata includes driver/NVML/VBIOS, power limits, guest-visible PCIe and
persistence, and maximum supported SM/graphics/memory clocks, not all possible
clock combinations. Preferred BLAS is a getter. Only canonical CUDA_DEVICE_ORDER
is recorded from the environment. cuBLAS version is not collected, and other
environment variables are entirely omitted.

## Draft output layout: 0.3.0-draft

The version identifies a draft layout, not validated JSON Schema. Missing timing
data is absent when timing never ran. Output filenames are operator supplied,
exclusive, and never overwritten.

| Location | Canonical contents |
|---|---|
| Root | Version, UTC timestamp, status, execution/state labels, shape/mode/seed/precision, baseline flag/reason codes |
| device / uuid_verification | CUDA PCI mapping, NVML metadata, HMAC ID, independent UUID availability/match flags |
| environment / software | OS/CPU/kernel/Git/runtime metadata; one CUDA_DEVICE_ORDER record under allowed_environment |
| preflight | Readiness, reason codes, actual temperature conditions/outcome |
| compute_processes | before_cooldown, before, after total/other counts |
| unlock_state_evidence | Operator/system evidence and binding verification; labels stay at root |
| experiment_metadata | Cooling/host metadata; initial state stays only in unlock_state_evidence |
| accuracy / accuracy_checks | Aggregate numerical pass / complete separate pre/post checks; suspicion is aggregated at root |
| warmup / timed_measurement | Requested/actual warm-up and timing plan/spans/outcome |
| timings_ms / summary | Raw operation timings and GEMM latency/TFLOPS summary |
| telemetry | Interval, duration, total raw count, samples, phases, violation_counters |

This breaks 0.2.0-draft compatibility. Use phases.timed.summary and phase
boundaries/clock_events instead of telemetry.timed_summary/timed_coverage and
whole-collection summary/clock_events. The reason code is
admission_validator_not_implemented. Repeated explanation notes, unallowlisted
environment placeholders, duplicate state/process fields, duplicated accuracy
endpoint details, cuBLAS placeholders, and exhaustive clock lists are absent.
Older local JSON is never automatically migrated/rewritten. Comparison accepts
only the current draft; future Schema implementations must track version
boundaries rather than reinterpret historical output.

Exit codes: 0 numerical draft pass, 2 numerical failure, 3 precision suspicion,
4 identity contradiction, 5 dependency/runtime unavailable, 6 blocked preparation.
CLI errors use argparse behavior; collisions are I/O errors. Preserve failed
attempts and investigate after two configuration failures rather than retrying
until a passing sample is selected. No automated retry occurs.

## Independent-process ON/OFF comparison

Require GPU/state/mode and a new --output-dir. Comparison defaults to diagnostic
execution; paired temperature options remain optional. Use a fixed sample count
long enough for timed polls. Alternate pairs ON/OFF, OFF/ON, ON/OFF for the
default three runs per condition; pass identical state/configuration and
temperature conditions to every child.

--timeout-seconds is the startup/benchmark budget (default 300 seconds). Actual
child timeout adds the cooldown and warm-up budgets; increase the benchmark
budget for longer workloads. Failure/invalid evidence stops without retry and
preserves partial progress and raw child diagnostics locally.

A controlled comparison requires complete independent runs, matching settings,
verified identity, known zero other-process counts, identical passed temperature
gates, and timed ON observations. Missing temperature gates produce an otherwise
successful uncontrolled_diagnostic with comparison_valid=false and no ratio.
Conflicts/incomplete runs produce invalid. Exit 0 means a completed controlled
or uncontrolled diagnostic, not baseline admission; invalid sequences exit 5.

Per-condition median/IQR and sample-standard-deviation/mean CV retain valid
counts. CV is unavailable below two finite observations or at zero mean.
Endpoint observations are phase-separated; ON also has timed observations,
while OFF intentionally has no in-window polls. Results describe differences,
never a causal or equivalence verdict.

## Limitations

- All output remains baseline-ineligible. Formal Schema/validation/admission,
  automatic state/configuration filenames, baseline aggregation, CI, full
  independent baseline coverage and variability review remain incomplete.
  General baseline ordering and optional output hashes are unimplemented.
- Sampled FP64 cannot guarantee whole-output/intermediate-sample correctness;
  NaN/Inf outside the tile may be missed. Thresholds/TF32 floor are provisional
  heuristics, not universal error bounds. Settings/kernel names do not prove
  instructions; profiling evidence is separate. No output snapshot is allocated.
- Kernel counts lack individual sanitized entries, chronology, per-GPU linkage,
  and detailed unsuccessful-attempt evidence. Missing logs do not prove no
  attempt. Labels/module presence/performance do not prove physical state.
  Guest reboot need not reset passthrough GPUs; documented AC-disconnected host
  cycles increase operational confidence without proving internal state.
- Polling can perturb timing and miss brief events. Active sample counts are
  neither distinct events nor durations. Zero observations/counter delta is not
  absence proof. Violation counters use a deprecated API and may be unsupported.
  NVML caching/averaging may lag: read-time alignment does not guarantee sensor
  acquisition-time alignment. Fan percentage is not measured RPM.
- Endpoint zero process counts cannot exclude hidden/intervening contention;
  namespaces/MPS limit visibility and another process can arrive after preflight.
  Services are never stopped. Diagnostic overrides do not justify formal use.
- Start-temperature/wait conditions have no research defaults. Cooling history,
  temperature, power, clocks, and order can confound comparisons. A controlled
  start is not equilibrium or proof of sampler impact/absence of impact.
- PCI addresses/VM link fields are guest topology, not globally stable identity.
  Stable pseudonym matching requires the same protected key. Operator strings
  need publication review. Raw JSON/logs/keys remain local by default.

Getter semantics follow the [NVIDIA NVML reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).
