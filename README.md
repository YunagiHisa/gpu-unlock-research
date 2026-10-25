# gpu-unlock-research

Reproducible experiments assessing [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) on dual NVIDIA GeForce RTX 3060 GPUs.

This is the initial benchmark-core implementation. It implements a single-device CUDA GEMM runner and CPU-only tests. The complete baseline protocol from [Issue #1](https://github.com/YunagiHisa/gpu-unlock-research/issues/1) is not implemented yet. **Do not treat generated output as final before-unlock evidence.**

## Running the draft

Activate a Python environment with CUDA-enabled PyTorch 2.11 and run:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -m benchmarks.gemm --gpu 0 --mode fp16 --m 2048 --n 2048 --k 2048 --warmup-seconds 3 --samples 30
python -m benchmarks.gemm --gpu 1 --mode bf16 --m 2048 --n 2048 --k 2048
python -m unittest discover -s tests -v
```

The draft currently prints JSON. Use `--output results/draft/sample.json` to save locally; output paths cannot be overwritten.

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

The draft **does not yet** record NVML telemetry, robust PCI identity mapping, kernel/boot evidence, validated baseline state labels, full environment metadata, schema validation, multiple process aggregation, or official result publication rules. These are scheduled in later PRs under Issue #1.

An installed gpu-unlock module is not proof of an active unlock. In Proxmox GPU passthrough, a VM reboot may not reset physical GPU state. For a GPU that has previously undergone an unlock attempt, the project's high-confidence reset protocol requires documented complete host shutdown with AC disconnected, then power-on. The procedure supports evidence gathering but cannot prove internal GPU state.

Never enable gpu-unlock merely to run this benchmark. Collect a verified baseline first.

## Failure policy

Accuracy is checked before warm-up and again on the final timed output, outside CUDA timing. `accuracy_checks.pre` and `accuracy_checks.post` record the separate results; `post` is null when the pre-check fails and timing is skipped. The existing `accuracy` object retains its fields: its error and reference details describe the latest check, while `passed` requires both checks to pass. These endpoint checks do not validate every timing sample. `accuracy.passed` reports numerical accuracy only; it does not clear precision-mode suspicion.

When numerical checks pass but either TF32 endpoint is below the floor, status is `precision_mode_suspect`, timings are retained for investigation, and exit code is 3. Such results must be excluded from formal baseline counts until the suspicion is resolved by separate evidence review. `precision_mode_suspect` and `error_floor` are recorded in accuracy checks, and the top-level suspicion flag combines both endpoints.

Either check failing makes the entire measurement `accuracy_failed`, with exit code 2. Non-finite errors fail validation and are serialized as null. Diagnostic JSON is written to `--output`, or printed to stdout when no output path is supplied. Pre-check failures have no timings; post-check failures retain raw `timings_ms` and `summary` strictly for diagnostics. Failed measurements must never count toward successful baseline totals. Existing output files are rejected with `FileExistsError` and remain unchanged; this I/O error is distinct from accuracy failure exit code 2. Stop after two failures of the same configuration to investigate rather than selecting only passing samples. The runner does not automate retries.

## Optional output snapshot assessment

An exact all-element comparison between the initial and final outputs could detect changes outside the FP64 sample, but would not detect a consistently wrong output or prove correctness of intermediate samples. It would also require an additional M*N output snapshot: 128 MiB for FP16/BF16 or 256 MiB for FP32 at 8192 squared, excluding comparison workspace and allocator overhead. Exact mismatches can reflect backend nondeterminism and require investigation rather than automatic numerical failure. This option is deferred in the current core PR; no snapshot is allocated and no all-element equality claim is made.
