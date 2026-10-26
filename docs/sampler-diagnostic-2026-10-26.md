# Sampler comparison, 2026-10-26 (Asia/Tokyo)

Development diagnostics only; **zero formal Before baseline runs**.
GPU 0, FP16, M=N=K=2048, seed 2026, 1 s warm-up, 1000 timing samples,
100 ms NVML interval. PyTorch 2.11.0+cu128 and nvidia-ml-py 13.615.71.
Each run used a fresh process. Pair order: ON/OFF, OFF/ON, ON/OFF.
ON enables asynchronous polling; OFF retains only out-of-window snapshots.

| Condition | Independent runs | TFLOPS median | TFLOPS IQR | GPU span median (s) | GPU span IQR (s) |
|---|---:|---:|---:|---:|---:|
| ON | 3 | 24.2796 | 0.0707 | 0.710641 | 0.002016 |
| OFF | 3 | 24.4210 | 0.1783 | 0.707102 | 0.005039 |

ON's median was approximately 0.58% lower. This small experiment does not
establish a sampler effect or demonstrate equivalence. No automatic pass/fail
overhead threshold is defined in PR #3. The per-operation CUDA Events and final
synchronization method were retained without tuning based on these results.

Temperature rose across the sequence: ON endpoints were 37→47, 42→51,
44→52 C; OFF endpoints were 39→48, 41→49, 44→53 C. ON post-window power was
161.259–162.768 W and SM clock 1905–1912 MHz; OFF post-window power was
162.172–165.106 W and SM clock 1905–1927 MHz. Pre-window power was much lower
(approximately 17–40 W). These are endpoint observations, not matching
steady-state samples: thermal/order/power/clock drift can confound the measured
difference. OFF intentionally has no in-window temperature/power/clock samples.
Further review is required before attributing the difference to polling.

Every ON run had seven complete timed polling samples; OFF correctly recorded
zero with `polling_disabled`. CUDA/NVML PCI association was verified in all six
runs. Each process reported one compute context before and after, including
itself when visible. These counts cannot prove absence of intervening or hidden
contention. No existing service was stopped.

No HMAC key was present or generated. The stable pseudonym was explicitly
unavailable (`key_missing`); cross-run identity in this diagnostic is limited
to the consistent, verified PCI association during the sequence. No raw UUID,
key contents, PID, process name, or executable path was retained. Cooling and
unlock-state history were not established by this test. Formal admission is
deferred to PR #4, and every result has `baseline_eligible: false`.

Local raw results are ignored under
`results/diagnostics/pr3-telemetry-comparison-2026-10-26/`, with
`comparison.json` recording the order and aggregate statistics. No results were
published externally.

An additional GPU 0 diagnostic exercised `--min-timed-seconds 0.3` with a
0.2 s warm-up and `--samples 30`. The warm-up estimated 0.73024 ms per GEMM,
selecting 411 samples (predicted 0.300129 s). The actual GPU span was
0.289870 s, with two complete timed polls. `target_met` correctly remained
false and `requested_timed_duration_not_met` was recorded; no retry or timing
adjustment was performed. This illustrates why duration estimates and actual
spans are separate, and why PR #4 must establish formal duration requirements.
