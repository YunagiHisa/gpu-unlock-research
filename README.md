# gpu-unlock-research

Reproducible experiments assessing [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) on dual NVIDIA GeForce RTX 3060 GPUs.

This is an **unmerged PR 1 draft**. It implements a single-device CUDA GEMM runner and CPU-only tests. The complete baseline protocol from [Issue #1](https://github.com/YunagiHisa/gpu-unlock-research/issues/1) is not implemented yet. **Do not treat generated output as final before-unlock evidence.**

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
- FP16/BF16 reduced-precision reduction flags set to false and recorded; high-level settings do not prove the GPU's actual instruction sequence.
- Preallocate input and output buffers; timed region contains `torch.mm(..., out=...)` and CUDA event recording/synchronization.
- Correctness uses normalized L2 difference relative to an FP64 product of the *quantized input values*, outside the timed region. Current thresholds are **provisional** and require review on the test GPUs. For matrix dimensions above 2048, accuracy uses a fixed-seed **128 × 128 sampled output tile** with the full K reduction instead of computing the complete FP64 output. A sampled check is not a whole-matrix guarantee and is marked in JSON as `sampled_fp64`. The sampling index seed is recorded. Configure `--full-reference-max` and `--reference-tile` if needed.

- The default normalized L2 limits (`fp16=0.02`, `bf16=0.08`, `tf32=0.02`, `fp32=0.0005`) are deliberately provisional screening thresholds, **not validated scientific error bounds**. TF32 accuracy depends on actual backend behavior; final limits must be justified against measured error distributions and a reference.
- TFLOPS = `2*M*N*K / latency_seconds / 1e12`.
- A single invocation benchmarks one configuration. Independent runs require independent processes.

## Important limitations

The draft **does not yet** record NVML telemetry, robust PCI identity mapping, kernel/boot evidence, validated baseline state labels, full environment metadata, schema validation, multiple process aggregation, or official result publication rules. These are scheduled in later PRs under Issue #1.

An installed gpu-unlock module is not proof of an active unlock. In Proxmox GPU passthrough, a VM reboot may not reset physical GPU state. For a GPU that has previously undergone an unlock attempt, the project's high-confidence reset protocol requires documented complete host shutdown with AC disconnected, then power-on. The procedure supports evidence gathering but cannot prove internal GPU state.

Never enable gpu-unlock merely to run this benchmark. Collect a verified baseline first.

## Failure policy

Validation failures are saved as diagnostic JSON with the same environment metadata, precision settings, and reference-check details, but no performance timings. Exit code 2 indicates an accuracy failure. Stop after two failures of the same configuration to investigate rather than selecting only passing samples. The runner does not automate retries.
