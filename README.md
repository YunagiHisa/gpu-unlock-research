# gpu-unlock-research

Reproducible experiments assessing [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) on dual NVIDIA GeForce RTX 3060 GPUs.

This draft implements single-device CUDA GEMM, optional NVML telemetry, GPU/environment metadata, and conservative unlock-state evidence. The complete baseline protocol from [Issue #1](https://github.com/YunagiHisa/gpu-unlock-research/issues/1) is not implemented yet. **Do not treat generated output as final before-unlock evidence.** Every result has `baseline_eligible: false`; numerical success does not certify a baseline.

## Running the draft

Activate a Python environment with CUDA-enabled PyTorch 2.11 and run:

```bash
python -m pip install -r requirements-gpu.txt
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -m benchmarks.gemm --gpu 0 --requested-state before --execution-kind diagnostic --mode fp16 --m 2048 --n 2048 --k 2048 --warmup-seconds 3 --samples 30
python -m benchmarks.gemm --gpu 1 --requested-state before --execution-kind diagnostic --mode bf16 --m 2048 --n 2048 --k 2048
python -m unittest discover -s tests -v
```

The draft currently prints JSON. Use `--output results/draft/sample.json` to save locally; output paths cannot be overwritten.

See [telemetry, cooling history, and state-evidence design](docs/telemetry-metadata.md) for the new CLI options and operator metadata. GPU 1's thermal paste was replaced on 2026-10-26; earlier smoke tests must not be reused as a formal baseline.

The [Issue #1 conformance review](docs/issue-1-conformance.md) records the
operator-supplied formal specification, implemented scope, and remaining gaps.
Historical verification reports are preserved in an [English PR comment](https://github.com/YunagiHisa/gpu-unlock-research/pull/3#issuecomment-6412810816).

## Explicit state and preparation gates

Both `--gpu` and `--requested-state {before,after,reverted}` are required.
There is no implicit Before measurement. The comparison runner requires and
passes the same state and temperature conditions to every child process.

`--execution-kind formal` (the default) prepares a run only when independent
PCI/UUID evidence matches, the configured start-temperature gate passes, and
the count of other compute processes is known to be zero. It does not confer
baseline eligibility. `--execution-kind diagnostic` explicitly permits
incomplete evidence or contention for investigation and records the reasons;
it must not bypass contention during the project's GPU smoke tests.

`--max-start-temp` and `--cooldown-timeout-s` have **no numeric defaults**.
Supply both when using the temperature gate. Issue #1 prescribes neither
value; an earlier 50 C proposal was not approved as a default. With neither
option, the gate is unverified and formal preparation saves a blocked diagnostic
without GEMM. Diagnostic mode records the missing gate. With both supplied,
NVML temperature is read and passively polled while above the operator's limit.
A timeout blocks GEMM in both modes; unavailable temperature blocks formal
preparation. Results record actual temperature, waiting time, checks, and outcome.
No clock, fan, power, or other GPU setting is changed during cooldown.
Blocked preparation exits with code 6. Other-process presence or unknown count
before/after adds an ineligibility reason. MPS, namespaces, and process arrival
between reads remain limitations; zero counts do not prove isolation.

## HMAC key setup

Stable GPU pseudonyms use HMAC-SHA256 over the NVML UUID with a versioned domain
prefix. Raw UUIDs and key contents are never included in results, logs, errors,
or Git. Operator labels such as `gpu-b` are separate from these stable IDs.

The default key is `$XDG_CONFIG_HOME/gpu-unlock-research/hmac.key`, or
`~/.config/gpu-unlock-research/hmac.key` when XDG_CONFIG_HOME is unset. Override
with `--hmac-key-file`. The key must be **outside this repository**, exactly
32 binary bytes, a regular file owned by the current user with mode **600**,
inside a directory owned by that user with mode **700**. Symlink files, symlink
key directories, and hardlinked key files are rejected. The benchmark never
creates keys or repairs permissions.

To explicitly initialize a key once, run the following manually from the
repository root. It refuses to overwrite a key, rejects unsafe directory
permissions, and prints no secret material. This is an operator setup step:

```bash
python - <<'PY'
import os
import secrets
import stat
from pathlib import Path

directory = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gpu-unlock-research"
if directory.resolve().is_relative_to(Path.cwd().resolve()):
    raise SystemExit("Choose a configuration directory outside the repository")
os.umask(0o077)
directory.mkdir(parents=True, mode=0o700, exist_ok=True)
info = directory.lstat()
if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
    raise SystemExit("Configuration directory must be owned by you with mode 700")
descriptor = os.open(directory / "hmac.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as output:
    output.write(secrets.token_bytes(32))
print("HMAC key initialized; back it up securely before collecting comparable results")
PY
```

Back up the exact binary key in an encrypted offline backup or encrypted secret
store, with access limited to the experiment owner. Keep backup key files at
600 in a 700 directory; restore those permissions and ownership before use.
Never paste the key into a ticket, shell command argument, JSON, logs, or Git.
Keep the same key across comparable experiments. Losing the key prevents
regenerating the same GPU pseudonyms; a replacement key changes all IDs and
breaks automatic matching to previous records. Public digests cannot recover
the lost key or prove which newly named GPU corresponds to an old record.

A missing or unusable key permits diagnostic measurement and JSON saving:
`device.pseudonym` is unavailable with a fixed, nonsecret reason and setup
guidance. The file path and exception text are not serialized. No unkeyed UUID
hash or automatically generated replacement is used.

## Duration and sampler diagnostics

`--samples` retains fixed-count behavior. Optional `--min-timed-seconds 1` uses
the warm-up CUDA span divided by its operation count to estimate the count for
that duration. The count is at least `--samples` and at most **10,000** (20,000
retained CUDA Events). A larger explicit `--samples` is rejected. The target is
an estimate: queue gaps, clock changes, and the cap may prevent reaching it.
There is no automatic remeasurement. `timed_measurement` records the estimate,
predicted duration, selected count, cap, actual first-start to final-stop CUDA
span, wall duration through final synchronization, and whether the target was
met. The sum of per-operation times is separate from the span. Formal sample
counts and minimum duration remain a PR #4 decision.

`telemetry.timed_coverage` counts only complete polling reads inside the queued
timed window, excluding endpoints and warm-up. Zero complete polls at 100 ms
is explicitly `missing`; endpoints are not substituted. `--telemetry off`
disables the worker while retaining matching endpoint snapshots and process
count reads outside timing, for diagnostic comparison.
`telemetry.timed_summary` contains only complete timed-window polls and marks
each field `unavailable` with null statistics when no valid samples exist.
`telemetry.summary` remains the separately labelled whole-collection summary.
Comparison statistics explicitly mark empty distributions unavailable; CV is
unavailable for fewer than two valid samples or a zero mean.

NVML compute-process **counts only** are recorded before and after measurement.
No PID, process name, or executable path is retained. `value` includes this
benchmark when visible; `other_processes` excludes the current process by a
transient PID comparison. PID namespaces and MPS may limit that classification.
Neither field proves exclusivity. Unavailable queries
have a reason, never a fabricated zero. Even zero at both endpoints cannot
prove absence of contention in between. Existing services are never stopped.

The comparison runner uses independent processes with ON/OFF, OFF/ON, ON/OFF
ordering for three runs per condition. It preserves raw results and compares
TFLOPS median/IQR, sample-standard-deviation/mean CV with sample count,
measurement duration, and temperature/power/SM clock by
endpoint phase. ON also has in-window environmental observations. OFF has none.
The runner stops on failure without retrying or changing measurement logic.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -m benchmarks.compare_telemetry --gpu 0 --requested-state before --execution-kind diagnostic --mode fp16 \
  --m 2048 --n 2048 --k 2048 --samples 1000 --warmup-seconds 1 \
  --runs 3 --output-dir results/diagnostics/telemetry-comparison
```

These runs never count as formal Before baseline. Examine thermal/power/clock
drift and contention before attributing performance differences to the sampler.
Report suspected effects; do not silently change the timing method.
See the [historical diagnostic report in the PR conversation](https://github.com/YunagiHisa/gpu-unlock-research/pull/3#issuecomment-6412810816).

## Measurement methodology

- Input dtypes: fp16, bf16, float32 (tf32 or IEEE matmul selected with the modern PyTorch API).
- FP16/BF16 inputs and outputs use their respective low-precision dtype. FP32 accumulation is intended; `allow_fp16_accumulation`, `allow_fp16_reduced_precision_reduction` and `allow_bf16_reduced_precision_reduction` are set to false and read back to disable FP16 accumulation and prohibit reduced-precision intermediate reductions. Missing support for `allow_fp16_accumulation`, or failure to read back false, raises an explicit error. The final output is still rounded to FP16/BF16. These PyTorch settings alone do not prove the actual accumulation path or GPU instructions executed; instruction-level claims require separate profiling evidence.
- Preallocate input and output buffers and retain one start/stop CUDA Event pair per sample. Record start, enqueue `torch.mm(..., out=...)`, and record stop for each operation on the same stream. Synchronize only the final stop event once, then read every pair's elapsed time and validate the final output. These are **per-operation GPU execution times during consecutive queued execution**, rather than end-to-end request latency or isolated synchronized calls. Host enqueue gaps and event overhead can still affect short operations; no gap-free execution is guaranteed. Allocation, FP64 validation, and warm-up are outside the timing intervals.
- Correctness uses normalized L2 difference relative to an FP64 product of the *quantized input values*, outside the timed region. Current thresholds are **provisional** and require review on the test GPUs. Full FP64 validation requires both a maximum dimension of 2048 and an operation count `2*M*N*K <= 2*2048^3` by default. Otherwise, accuracy uses a fixed-seed **128 × 128 sampled output tile** with the full K reduction instead of computing the complete FP64 output. A sampled check is not a whole-matrix guarantee: errors and NaN/Inf outside the sampled rows/columns may be missed. It is marked in JSON as `sampled_fp64`, with this limitation in `note`. Pre- and post-checks use the same deterministic sample. The sampling index seed is recorded. Configure `--full-reference-max`, `--full-reference-max-flops`, and `--reference-tile` if needed. Both full-reference limits must pass; the FLOP budget accounts for K even when the output is small. The sampled reference always retains the full K reduction, so the budget does not cap sampled-reference cost. JSON records `reference_flops` and `full_reference_max_flops` for each check.

- The provisional normalized L2 upper limits are `fp16=1e-3`, `bf16=5e-3`, `tf32=1e-3`, and `fp32=2e-5`. Equality passes. These screening criteria are based on current measurements, not scientific error bounds across arbitrary shapes, seeds, and inputs.
- TF32 error below `2e-5` sets `precision_mode_suspect` without failing numerical accuracy. The floor is a heuristic for possible IEEE execution, motivated by the much smaller IEEE error observed below; it does not prove which kernel ran. Either endpoint being suspect requires separate kernel verification before any formal baseline inclusion. No automatic resolution or kernel-verification workflow is implemented in this PR.
- TFLOPS = `2*M*N*K / latency_seconds / 1e12`.
- A single invocation benchmarks one configuration. Independent runs require independent processes.

## Provisional threshold evidence

On 2026-10-26 (Japan time), both reference RTX 3060 GPUs were smoke-tested independently with PyTorch 2.11.0+cu128, seed 2026, M=N=K=256, full FP64 reference, 0.05 s warm-up, and five queued timing samples. Pre- and post-timing normalized L2 errors were identical at the shown precision on both GPUs:

| Mode | GPU 0 pre / post | GPU 1 pre / post | Upper limit | Result |
|---|---|---|---|---|
| FP16 | 2.0712657e-4 / 2.0712657e-4 | 2.0712657e-4 / 2.0712657e-4 | 1e-3 | Pass |
| BF16 | 1.6633499e-3 / 1.6633499e-3 | 1.6633499e-3 / 1.6633499e-3 | 5e-3 | Pass |
| TF32 | 2.9521834e-4 / 2.9521834e-4 | 2.9521834e-4 / 2.9521834e-4 | 1e-3 | Pass; no floor suspicion |
| FP32 IEEE | 1.4892200e-7 / 1.4892200e-7 | 1.4892200e-7 / 1.4892200e-7 | 2e-5 | Pass |

The selected limits leave margin above observed rounding error while screening more strictly than the original draft thresholds. This single shape and seed do not establish an error distribution. Reassess the criteria on the required larger configurations before formal baseline collection. These are development smoke tests, not Before baseline measurements. Raw JSON remains local under ignored `results/`.

## Important limitations

The draft **does not yet** implement full result JSON Schema validation, formal baseline aggregation, kernel verification for suspicious precision modes, or official baseline admission/publication rules. These remain follow-up work under Issue #1. `schema_version: 0.2.0-draft` identifies the output layout, not a schema-validation guarantee. The sampler comparison has been checked on GPU 0; the full GPU/precision/configuration matrix remains untested for this implementation.

An installed gpu-unlock module is not proof of an active unlock. In Proxmox GPU passthrough, a VM reboot may not reset physical GPU state. For a GPU that has previously undergone an unlock attempt, the project's high-confidence reset protocol requires documented complete host shutdown with AC disconnected, then power-on. The procedure supports evidence gathering but cannot prove internal GPU state.

Never enable gpu-unlock merely to run this benchmark. Collect a verified baseline first.

## Failure policy

A proven CUDA/NVML PCI or UUID contradiction or contradictory operator PCI binding
stops before tensor allocation and GEMM, saves a sanitized diagnostic, and
exits **4**. NVML unavailable or missing CUDA PCI data permits diagnostic
execution only in explicit diagnostic mode with `identity_verified: false` and
a reason. Both PCI and independent context-device UUID matching are required
to set it true. UUID bytes are compared in memory and never serialized or logged.
A missing key independently makes the pseudonym unavailable.
Unavailable dependencies or runtime measurement errors save a sanitized
diagnostic and exit **5**. PR #3 always sets `baseline_eligible: false` with
`baseline_ineligibility_reasons`; final admission belongs to PR #4's validator.

Accuracy is checked before warm-up and again on the final timed output, outside CUDA timing. `accuracy_checks.pre` and `accuracy_checks.post` record the separate results; `post` is null when the pre-check fails and timing is skipped. The existing `accuracy` object retains its fields: its error and reference details describe the latest check, while `passed` requires both checks to pass. These endpoint checks do not validate every timing sample. `accuracy.passed` reports numerical accuracy only; it does not clear precision-mode suspicion.

When numerical checks pass but either TF32 endpoint is below the floor, status is `precision_mode_suspect`, timings are retained for investigation, and exit code is 3. Such results must be excluded from formal baseline counts until the suspicion is resolved by separate evidence review. `precision_mode_suspect` and `error_floor` are recorded in accuracy checks, and the top-level suspicion flag combines both endpoints.

Either check failing makes the entire measurement `accuracy_failed`, with exit code 2. Non-finite errors fail validation and are serialized as null. Diagnostic JSON is written to `--output`, or printed to stdout when no output path is supplied. Pre-check failures have no timings; post-check failures retain raw `timings_ms` and `summary` strictly for diagnostics. Failed measurements must never count toward successful baseline totals. Existing output files are rejected with `FileExistsError` and remain unchanged; this I/O error is distinct from accuracy failure exit code 2. Stop after two failures of the same configuration to investigate rather than selecting only passing samples. The runner does not automate retries.

## Optional output snapshot assessment

An exact all-element comparison between the initial and final outputs could detect changes outside the FP64 sample, but would not detect a consistently wrong output or prove correctness of intermediate samples. It would also require an additional M*N output snapshot: 128 MiB for FP16/BF16 or 256 MiB for FP32 at 8192 squared, excluding comparison workspace and allocator overhead. Exact mismatches can reflect backend nondeterminism and require investigation rather than automatic numerical failure. This option is deferred in the current core PR; no snapshot is allocated and no all-element equality claim is made.
