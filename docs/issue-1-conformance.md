# PR #3 conformance review

Reviewed on 2026-10-26 (Asia/Tokyo), including the local P0/P1/P2 corrections.
Authoritative input: the operator-supplied formal Issue #1 snapshot.
SHA-256: `f75d8a0824c60464adad37388bde2a50fcb4d0eb16b105c5fadbabde62872fb9`.
Working branch: `feat/nvml-telemetry-metadata` (already present; no main edits).
This review does not amend Issue #1 or declare its acceptance criteria complete.

## Scoped implementation and approved additions

| Requirement | Implementation / disposition |
|---|---|
| Issue §1–2: single-GPU GEMM, precision readback, FP64 validation, CUDA Events | Existing core retained. Polling is asynchronous; telemetry summaries, count queries, and accuracy checks are outside per-operation CUDA Event intervals. ON/OFF diagnostics report possible indirect interference. |
| Issue §3: before-state table and per-GPU initial-state evidence | High → not_applicable for before; low/unknown → unconfirmed. No automatic confirmed/failed classification. AC-disconnected host power-cycle records are required after an unlock attempt; guest reboot is insufficient. |
| Issue §3: boot/module/log evidence | Boot parameter presence only, module presence, and log availability are recorded. Current log evidence is aggregate success/failure marker counts; see the unresolved limitation below. |
| Issue §4: NVML identity and asynchronous sampling | CUDA context PCI ID selects NVML handle; readback is checked. Clock, power, temperature, reasons, interval, raw samples, summaries, and unavailable states are recorded. Fan percentage and policy duration deltas extend this under operator instructions. |
| Approved identity extension | Contradiction stops before tensor allocation/GEMM, exit 4. Inability to verify permits diagnostics with identity_verified=false and a reason. Only successful matching sets true. |
| Approved HMAC extension | HMAC-SHA256 UUID pseudonym; external, owned 700 directory and 600 key file; no automatic key creation. Missing/unsafe keys preserve nonsecret diagnostics. README describes explicit setup, backup, and loss risks. |
| Approved duration extension | Optional min-timed-seconds uses the warm-up estimate; samples remains a lower bound with a 10,000 pair limit. Estimate, count, actual span, target shortfall, and absent timed polls are recorded. No adaptive retries. |
| Approved contention extension | Endpoint total compute-process counts and counts excluding the current process. PID is compared transiently, never retained; names/paths are not queried or saved. Zero is not proof of no contention; namespaces/MPS can limit visibility. |
| Issue §5 / approved sampler comparison | Independent ON/OFF processes with alternating pair order; fixed GPU/shape/mode/count. Median, IQR, sample count, and sample-standard-deviation/mean CV. Temperature/power/clock differences are reported. Failed child results survive; comparison stops on failure, without retries. |
| Issue §6: environment and privacy | GPU/driver/runtime/kernel/Git/guest PCIe/persistence metadata and constrained manual host fields. No raw UUID or key in results. cuBLAS version explicitly unavailable if not reliably queried. |
| Approved cooling history | Per-GPU pseudonymous operator label, PCI binding, cooling state/history, provenance. GPU 1 maintenance on 2026-10-26 and approximate observations are documented. Smoke/diagnostic tests never become formal Before measurements. |
| Approved baseline policy | baseline_eligible is always false with reasons, including the unimplemented PR #4 admission validator. Identity verification and numerical pass never grant formal admission. |

## P0/P1/P2 review corrections

- GPU and requested state are required in both CLI entry points and passed
  explicitly to every comparison child. There is no implicit Before label.
- Start-temperature limit and cooldown timeout have no numeric defaults. When
  configured both are required. The operator confirmed that the earlier 50 C
  proposal was not an approved default. Without configuration, the gate remains
  unverified; formal preparation saves a blocked diagnostic before GEMM.
  Explicit diagnostic mode may record incomplete evidence, but never grants
  formal baseline admission. Timeout blocks both modes. Waiting is passive.
- Current-context CUDA UUID and NVML UUID are compared independently of the
  retained PCI check, in memory only. PCI/UUID contradictions stop before GEMM
  with exit 4. Either missing UUID prevents verified identity. Context-device
  queries account for visible-device remapping; no raw UUID reaches output.
- Both before/after other-process counts are evaluated. Positive counts add
  other_compute_processes_present; unknown counts add
  other_compute_processes_unknown. Formal preparation blocks on either before
  GEMM. Counts are rechecked after cooldown. No service is stopped/restarted.
- idle_before/warmup/timed/post summaries report counts and min/median/max,
  with phase-specific sampled reason observations. Both read endpoints must lie
  inside warmup/timed boundaries. Missing phases and fields remain unavailable
  with null statistics. Zero observed reasons do not establish absence.
- OS name, CPU model, NVML version, preferred BLAS (getter only), memory clock,
  GPU utilization, and performance state are recorded or unavailable.
- The formal issue contains no environment-value allowlist beyond the explicitly
  documented CUDA_DEVICE_ORDER setting. Per the operator's clarification, only
  canonical CUDA_DEVICE_ORDER is saved. CUBLAS_WORKSPACE_CONFIG and
  TORCH_BLAS_PREFER_CUBLASLT are unavailable with no values recorded. This is
  pending specification detail, not permission to dump environment variables.
- Repository documentation is English. The two historical measurement reports
  were preserved in an English PR comment before local removal; private local
  paths and original raw artifacts were not posted. Conformance stays here.

## Revised-code validation

The original 56 CPU tests remain, updated for the required CLI arguments.
Additional regressions exercise UUID mismatch/missing sides, actual-context
identity under visible remapping, contention before/after, temperature timeout
and missing configuration, finite-sample CV, phase boundaries, privacy, and
exception cleanup.
All 68 CPU tests passed (the original 56 plus 12 regressions), and
`git diff --check` passed. A value-safe audit of all 16 present changed/new
files and the three new local GEMM JSON artifacts found no secret or private
identifier leakage. The two removed report files were already migrated to the
English PR comment. At that review checkpoint the index was empty and HEAD
was unchanged; the operator subsequently authorized publishing this revision.

At the revised actual-GEMM test time both GPUs had zero visible compute
processes before launch. Each ran FP16, 128 squared, seed 2026, 0.01 s warmup,
and three timing samples. Both returned draft_validated, passed accuracy and
independent PCI/UUID matching, and had zero other processes at preflight.
Temperatures were GPU 0: 35 to 36 C; GPU 1: 41 to 42 C. A 50 C start limit
and zero wait budget were explicit conditions for this tiny test only, not
defaults or formal admission criteria. The supervising read-only monitor would
stop only its own child on sustained temperature rises or competing activity;
neither occurred. The earlier report's busy GPU 1 remains historical and is not
retroactively claimed to have run GEMM. No service or system setting changed.
These checks do not validate prolonged loading, every precision/size, or a formal
baseline, and all diagnostic JSON remains local and ignored.
Visible-device remapping and unavailable-UUID branches were mock-tested; a
separate physical CUDA_VISIBLE_DEVICES-remapping experiment was not run.

## Findings and corrections in this review

The diagnostic comparison had omitted Issue §5's CV. Added
`cv_sample_std_over_mean = statistics.stdev(values) / mean(values)` and its
`samples` count. Fewer than two values or zero mean produces null, not a
fabricated CV. No acceptable CV threshold is invented.

Previously process counts included the benchmark. Kept that total and added
`other_processes` in each endpoint record so the operator can distinguish the
counts. Both counts come from one read-only NVML process-list query. The list
and individual PIDs are discarded before the result leaves the reader.
Historical diagnostic JSON predates these additions and is not rewritten or
retroactively claimed to contain the new fields.

The final verification additionally found that a missing timed polling window
needed its own explicitly unavailable statistics, separate from whole-session
statistics. Added `telemetry.timed_summary` without changing the CUDA Event
measurement method. Empty comparison distributions and CV with insufficient
valid observations now expose unavailable statuses and reasons. See the
[historical verification comment](https://github.com/YunagiHisa/gpu-unlock-research/pull/3#issuecomment-6412810816) for the read-only GPU
smokes, privacy findings, and retained acceptance gaps.

## Remaining gaps; original requirements retained

- Issue §3 asks for sanitized relevant kernel-log evidence and preservation of
  failed attempts. Aggregate marker counts preserve only limited failure signals;
  they lose individual entries, chronology, and association with a measured GPU.
  This is a partial implementation, not a claim of full compliance with that
  requirement. No detailed kernel-evidence format is substituted for the original
  requirement without review.
- Full JSON Schema and result validation, automatic UTC/PCI/configuration result
  filenames and state directories, baseline aggregation/admission, and CI remain
  in the approved subsequent PR scope. Existing explicit output files are
  exclusive and ignored under results/. Issue §7–8 remain open.
- General baseline configuration ordering and optional output hashes remain
  unimplemented. Explicit passive cooldown is now implemented, but its formal
  threshold/wait policy is not specified by Issue #1. The ON/OFF pair order is
  a diagnostic protocol, not a replacement for the full baseline protocol.
- The complete two-GPU/four-precision/three-size matrix, 72 schema-validated
  independent Before measurements, and variability review have not been
  completed. Prior smoke tests and sampler diagnostics count as zero formal
  baseline measurements.

Issue #1 explicitly permits split implementation PRs, but the deferred items
remain required before final implementation acceptance and issue closure.
No unlock activation, system setting change, service operation, or merge was
performed. Review validation initially kept all code changes local. The operator
subsequently authorized committing and pushing this reviewed revision to the
existing Draft PR; that authorization does not include merge, Issue #1 changes,
or formal Before baseline collection.
