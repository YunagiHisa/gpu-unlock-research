# Telemetry, metadata, and unlock-state evidence draft

This implements the telemetry/metadata portion of Issue #1 on main commit
`0eda774357b8b2dd1d188fa4ae1ca2c3f99f3d26`. It never changes NVIDIA settings,
executes gpu-unlock, or invokes sudo. CPU tests require only the standard library.
GPU dependencies are separate in `requirements-gpu.txt`; no NVML import occurs
when importing the benchmark modules.

## Collection boundaries and identity

After selecting the CUDA device, the runner reads the current CUDA context's
device and PCI bus ID through the CUDA driver. This respects CUDA visibility
remapping without assuming NVML ordinals match CUDA ordinals. NVML selects a
handle by normalized PCI ID and reads the PCI ID back to verify it. A missing
CUDA mapping or NVML failure produces explicit unavailable fields and never
falls back to an NVML index. Only a successful PCI match sets
`identity_verified: true`. A contradictory readback or operator PCI binding
stops before tensor allocation with exit code 4 and a sanitized diagnostic.
An unavailable mapping permits measurement with `identity_verified: false`
and a reason. If CUDA PCI is unavailable, operator evidence remains unbound
and cannot confer high confidence. PR #3 always keeps `baseline_eligible: false`
with reasons including pending PR #4 admission logic and missing evidence.

The sampler covers warm-up and queued timing together, excluding FP64 pre/post
validation. Initial/final samples and violation-counter reads occur outside CUDA
Event intervals. A worker thread polls at `--telemetry-interval` (default 0.1 s).
Each raw sample has a monotonic elapsed time and per-field status. The interval
is a target, not a real-time scheduling guarantee. Short runs may have only two
endpoint samples. CPU scheduling and NVML polling can perturb performance;
use the same collection settings in later comparisons. The worker is stopped
and joined before NVML shutdown, including when measurement raises an exception.

Recorded telemetry includes SM clock (MHz), board power (W), GPU temperature
(Celsius), fan percentage, and raw clock-event reason masks. Fan percentage is
NVML's fan speed percentage, not measured RPM. Modern clock-event APIs are used
with the legacy throttle-reason names as a compatibility fallback. Summary
statistics include available/missing counts, min, max, and mean. Raw masks retain
unknown bits; supported reason masks are in device metadata.
`summary` covers all collected samples including warm-up and endpoints.
`timed_summary` separately covers only complete timed-window polls. With zero
valid timed samples its fields have status `unavailable`, null min/max/mean,
and a reason. No endpoint value is substituted. The comparison runner also
marks empty distributions unavailable and does not compute CV with fewer than
two valid observations or with zero mean.

`observed_samples_by_reason` counts samples with a reason bit set, not distinct
events or time under a limit. No samples available yields null, not zero. Power
and thermal `violation_counters` retain before/after duration and reference
timestamps plus duration differences in nanoseconds. Counter decreases or a
reference-time decrease invalidate the delta. Unsupported counters stay null;
zero delta only means no increase in that available counter over this scope.
The compatibility API `nvmlDeviceGetViolationStatus` is deprecated by NVIDIA;
a future NVML field-value implementation will be needed when it is removed.
Neither sampled absence nor a zero counter delta proves that no brief limitation
occurred. See the [NVIDIA device-query reference](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).

Metadata includes name, VBIOS, configured/default power limit, supported memory
and graphics clocks, maximum SM clock, guest-visible PCIe generation/width,
persistence mode, NVIDIA driver, Python/PyTorch/CUDA runtime, guest kernel,
Git commit/dirty status, and CUDA device ordering. VM values describe the guest
view and may not describe the host link. cuBLAS version is explicitly unavailable
because there is no reliable public PyTorch query in this implementation.
Missing GPU dependencies permit a diagnostic benchmark result with unavailable
telemetry, which remains ineligible for a formal baseline.

## Operator input and cooling history

Use one local JSON file per measured GPU with `--experiment-metadata`. The input
is an allowlist, rejecting unknown fields at every supported object level.
Do not put UUIDs, serial numbers, hostnames, private paths, or identifying strings
in the anonymous labels. These labels are operator descriptions, separate from
the stable HMAC-SHA256 GPU pseudonym. See README for external key management.
PCI topology is retained as required by Issue #1; it is
not a globally unique GPU identifier. Raw results remain ignored under `results/`.
Publication still requires separate review.

For GPU 1, create a local file such as the following. **Replace the illustrative
PCI address with the actual CUDA GPU's address**; no address is inferred from
the index. The dated maintenance record uses the operator's local calendar date
(Asia/Tokyo). Initial-state evidence timestamps, when provided, must be UTC.

```json
{
  "gpu_label": "gpu-b",
  "pci_bus_id": "0000:02:00.0",
  "cooling_state_id": "repaste-2026-10-26",
  "cooling_history": [
    {
      "date": "2026-10-26",
      "change": "thermal_paste_replaced",
      "previous_state_id": "pre-repaste",
      "state_id": "repaste-2026-10-26",
      "source": "operator_report",
      "observations_before": {
        "max_temperature_c": 81,
        "sw_thermal_slowdown_observed": true
      },
      "observations": {
        "power_w": 167,
        "temperature_c": 71,
        "fan_percent": 53,
        "thermal_counter_increase_observed": false
      }
    }
  ],
  "initial_state": {"unlock_history": "unknown"},
  "host_environment": {"vfio_configuration": "unknown"}
}
```

These numbers are approximate operator observations, not new NVML measurements.
The operator reports no temperature-limit counter increase across multiple
post-maintenance benchmarks. The record does not supply their durations or
formal raw telemetry, and cannot establish baseline eligibility. GPU 0 was
reported at approximately 145 W, 62 C, and fan 36%; record that separately under
its own PCI ID and anonymous label using `change: unchanged_observation`.
Missing cooling metadata is stored as null, never inferred from the GPU index.
History is chronological; its last state ID must match `cooling_state_id`.
Other accepted change kinds are `cooling_maintenance` and
`unchanged_observation`. Before/after comparison must use matching physical GPU
and cooling-state records, or explicitly treat cooling changes as confounders.
Stable operator labels require operator-managed identity continuity; guest PCI
addresses alone cannot establish continuity across passthrough reconfiguration.

**No formal Before baseline has been collected.** The previous smoke tests,
including the provisional numerical-threshold tests in README, are development
evidence only. GPU 1's cooling state changed on 2026-10-26; do not relabel those
tests as a formal baseline. After this implementation and the schema/review work
are complete, collect a fresh baseline under documented cooling conditions.
Thermal maintenance does not establish unlock history or a qualifying power cycle.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -m benchmarks.gemm --gpu 1 --mode fp16 \
  --requested-state before --telemetry-interval 0.1 \
  --experiment-metadata results/local-gpu-b-metadata.json \
  --output results/before/draft-gpu-b.json
python -m unittest discover -s tests -v
```

Output paths remain explicit and exclusive: existing files are never overwritten.
Automatic timestamp/configuration filenames, a versioned JSON Schema, and formal
baseline admission remain in the next implementation scope.

## Unlock-state evidence

`requested_state` is an operator label (`before`, `after`, `reverted`), defaulting
to `before`. `verification_status` is `not_applicable` only for `before` with
`initial_state_confidence: high`; every other result is `unconfirmed`.
This runner never assigns `confirmed` or `failed`. Such decisions require a
separate evidence review, independent of numerical validation or telemetry.

Documented `unlock_history: never_attempted` with `documented_at_utc` qualifies
for high operational confidence. For `attempted`, high confidence additionally
requires `last_attempt_utc` and this `power_cycle` object:

```json
{
  "completed_at_utc": "2026-10-25T12:00:00Z",
  "method": "host_shutdown_ac_disconnect_power_on",
  "host_shutdown": true,
  "ac_disconnected": true,
  "fresh_power_on": true
}
```

The completion must follow the last attempt and precede or equal
`documented_at_utc`. Timestamps must be UTC, nonfuture, and per measured GPU.
`residual_state_suspected: true` forces low confidence. Insufficient records yield
unknown confidence. Contradictory histories and nonboolean flags are rejected.
Guest reboot, host restart, and undocumented shutdown do not qualify. This
records a completed operator procedure; the runner never performs it. These
records cannot prove the GPU's internal physical state.

Read-only system evidence records the presence of the exact
`gpu_unlock.targets` boot parameter (without its value or the full command line),
module presence, and sanitized kernel evidence. `dmesg` runs without elevation
with a timeout. Denied access is `unavailable`. Only counts of relevant lines and
success/failure text markers survive; raw lines, UUIDs, addresses, hostnames, and
paths are discarded. Marker counts preserve evidence of failure but are not
classified unlock outcomes, are not correlated to a particular GPU, and may
reflect previous attempts or ambiguous wording. Absence of logs does not
establish absence of attempts. Module presence and requested state do not prove
actual unlock state.

Marker counts alone are a partial implementation of Issue #1's sanitized
kernel-log evidence requirement; individual entries and per-GPU chronology
are not preserved. This limitation remains open in the conformance review.

Manual host fields accept only `proxmox_version`, `host_kernel_version` (version
strings), and `vfio_configuration` (`dedicated_passthrough`, `shared`, `unknown`).
They are operator reports, not host inspection. Full host command lines,
configuration dumps, and arbitrary notes are deliberately not accepted.
