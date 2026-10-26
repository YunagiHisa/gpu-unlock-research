# gpu-unlock-research

Single-device CUDA GEMM diagnostics with read-only NVML telemetry, reproducible
precision controls, and GPU-state evidence. The [experiment requirements](https://github.com/YunagiHisa/gpu-unlock-research/issues/1)
define the broader baseline protocol. Current results are drafts and always
have `baseline_eligible: false`.

## Setup

Use Python 3.12 and CUDA-enabled PyTorch with the APIs required by the
[design](docs/telemetry-metadata.md). CPU tests use only the standard library.

```bash
python -m pip install -r requirements-gpu.txt
python -m unittest discover -s tests -v
```

## Minimal runs

Select an unused GPU and explicit state. Set `START_TEMP_C` and `WAIT_SECONDS`
to the intended diagnostic conditions; neither has a default. Never enable
gpu-unlock or stop an existing service merely to run diagnostics.

A short smoke test checks execution and accuracy; it may contain no timed NVML
polls:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -m benchmarks.gemm --gpu 0 --requested-state before \
  --execution-kind formal --mode fp16 --m 128 --n 128 --k 128 \
  --warmup-seconds 0.01 --samples 3 \
  --max-start-temp "$START_TEMP_C" --cooldown-timeout-s "$WAIT_SECONDS" \
  --output results/diagnostics/run.json
```

For NVML telemetry collection, use a longer duration-targeted diagnostic:

```bash
python -m benchmarks.gemm --gpu 0 --requested-state before \
  --execution-kind formal --mode fp16 --m 4096 --n 4096 --k 4096 \
  --min-timed-seconds 2 --telemetry on \
  --max-start-temp "$START_TEMP_C" --cooldown-timeout-s "$WAIT_SECONDS" \
  --output results/diagnostics/telemetry.json
```

`formal` enforces preparation gates, not baseline admission. Duration targets
are estimates; shortfalls return a failure while preserving evidence. Existing
output files are never overwritten. See the design for ON/OFF comparison,
external HMAC key setup, operator metadata, and the draft JSON layout.

## Limitations and references

Formal Schema/admission, baseline aggregation, detailed kernel evidence, and
full configuration validation remain incomplete. Labels, module presence,
numerical passes, and zero observed events/processes do not prove physical
state or isolation. Guest reboots may not reset passthrough GPUs. See the
[design limitations](docs/telemetry-metadata.md#limitations).

- [Current design and output](docs/telemetry-metadata.md)
- [Historical verification](https://github.com/YunagiHisa/gpu-unlock-research/pull/3#issuecomment-6412810816)
- [Cooling and precision research records](https://github.com/YunagiHisa/gpu-unlock-research/pull/3#issuecomment-6415089744)
