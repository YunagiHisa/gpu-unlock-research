"""Measure one GPU / precision / matrix configuration per invocation.

Telemetry and evidence draft. Full JSON Schema validation and baseline
certification are deferred. Output is not a publishable baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.core import MAX_TIMING_SAMPLES, MODES, normalized_error, operation_count, select_samples, timing_summary, validate_dimensions
from benchmarks.evidence import load_experiment, state_evidence
from benchmarks.metadata import cuda_pci_bus_id, environment_metadata
from benchmarks.telemetry import NvmlSession, Sampler
from benchmarks.identity import IdentityMismatch

IDENTITY_MISMATCH_EXIT = 4

# Provisional screening criteria based on reference-GPU smoke measurements.
ERROR_LIMITS = {"fp16": 1e-3, "bf16": 5e-3, "tf32": 1e-3, "fp32": 2e-5}
TF32_ERROR_FLOOR = 2e-5


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index; resolve PCI address in output")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--m", type=positive_int, default=2048)
    parser.add_argument("--n", type=positive_int, default=2048)
    parser.add_argument("--k", type=positive_int, default=2048)
    parser.add_argument("--warmup-seconds", type=positive_float, default=3.0)
    parser.add_argument("--samples", type=positive_int, default=30)
    parser.add_argument("--min-timed-seconds", type=positive_float, help="optional duration target estimated from warm-up; not a guaranteed minimum")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-tile", type=positive_int, default=128, help="Max rows and cols in FP64 validation subset (large matrices)")
    parser.add_argument("--full-reference-max", type=positive_int, default=2048, help="Maximum dimension for full FP64 reference")
    parser.add_argument("--full-reference-max-flops", type=positive_int, default=2 * 2048**3,
                        help="Maximum 2*M*N*K for full FP64 reference; dimension limit also applies")
    parser.add_argument("--output", type=Path, help="local JSON path; omit to print to stdout")
    parser.add_argument("--telemetry-interval", type=positive_float, default=0.1, help="NVML polling interval in seconds")
    parser.add_argument("--telemetry", choices=("on", "off"), default="on", help="disable asynchronous polling for diagnostic overhead comparisons; keep endpoint snapshots")
    parser.add_argument("--hmac-key-file", type=Path, help="existing 32-byte key outside repository; path is never saved")
    parser.add_argument("--requested-state", choices=("before", "after", "reverted"), default="before")
    parser.add_argument("--experiment-metadata", type=Path, help="allowlisted per-GPU cooling and initial-state JSON")
    args = parser.parse_args(argv)
    if args.samples > MAX_TIMING_SAMPLES:
        parser.error(f"--samples exceeds the {MAX_TIMING_SAMPLES} CUDA Event pair budget")
    return args


def configure_mode(torch, mode: str):
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_fp16_accumulation"):
        raise RuntimeError("PyTorch must support matmul.allow_fp16_accumulation; cannot enforce FP32 accumulation")
    matmul.allow_fp16_accumulation = False
    if matmul.allow_fp16_accumulation:
        raise RuntimeError("FP16 accumulation unexpectedly enabled")
    # For precision experiments, prohibit reduced-precision intermediate
    # reductions where the installed PyTorch supports these flags.
    matmul.allow_fp16_reduced_precision_reduction = False
    matmul.allow_bf16_reduced_precision_reduction = False
    if not hasattr(matmul, "fp32_precision"):
        raise RuntimeError("PyTorch must support matmul.fp32_precision")
    matmul.fp32_precision = "tf32" if mode == "tf32" else "ieee"
    expected = "tf32" if mode == "tf32" else "ieee"
    if matmul.fp32_precision != expected:
        raise RuntimeError("requested FP32 matmul precision did not stick")
    if matmul.allow_fp16_reduced_precision_reduction or matmul.allow_bf16_reduced_precision_reduction:
        raise RuntimeError("reduced precision reduction unexpectedly enabled")
    return {
        "fp32_precision": matmul.fp32_precision,
        "allow_fp16_accumulation": matmul.allow_fp16_accumulation,
        "allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
    }


def use_full_reference(args):
    return (max(args.m, args.n, args.k) <= args.full_reference_max
            and operation_count(args.m, args.n, args.k) <= args.full_reference_max_flops)


def check_accuracy(torch, a, b, out, args):
    """Compare to exact-input FP64 product, full for small matrices, sampled for large.

    Sampling is deterministic and covers randomly selected output rows/columns.
    For sampled checks the reported error is NOT a whole-matrix error bound.
    """
    full = use_full_reference(args)
    if full:
        observed = out.double()
        reference = a.double() @ b.double()
        checked_shape = [args.m, args.n]
    else:
        # CPU generator ensures reproducible indices without altering GPU RNG state.
        gen = torch.Generator(device="cpu").manual_seed(args.seed + 17)
        nr = min(args.reference_tile, args.m)
        nc = min(args.reference_tile, args.n)
        rows = torch.randperm(args.m, generator=gen)[:nr].to(a.device)
        cols = torch.randperm(args.n, generator=gen)[:nc].to(a.device)
        observed = out.index_select(0, rows).index_select(1, cols).double()
        reference = a.index_select(0, rows).double() @ b.index_select(1, cols).double()
        checked_shape = [nr, nc]
    error = normalized_error(observed, reference)
    del observed, reference
    return error, {"method": "full_fp64" if full else "sampled_fp64", "checked_output_shape": checked_shape,
                   "reference_flops": operation_count(checked_shape[0], checked_shape[1], args.k),
                   "full_reference_max_flops": args.full_reference_max_flops,
                   "sampling_seed": args.seed + 17 if not full else None,
                   "note": None if full else "Sampled output check; errors and NaN/Inf outside the sampled rows/columns may be missed; not a whole-matrix accuracy guarantee"}


def accuracy_result(torch, a, b, out, args):
    error, details = check_accuracy(torch, a, b, out, args)
    threshold = ERROR_LIMITS[args.mode]
    return {"normalized_l2_error": error if math.isfinite(error) else None,
            "threshold": threshold,
            "precision_mode_suspect": args.mode == "tf32" and math.isfinite(error) and error < TF32_ERROR_FLOOR,
            "error_floor": TF32_ERROR_FLOOR if args.mode == "tf32" else None,
            "passed": math.isfinite(error) and error <= threshold, **details}


def measure_queued(torch, a, b, out, sample_count, measurement=None, sampler=None):
    if not 1 <= sample_count <= MAX_TIMING_SAMPLES:
        raise ValueError("sample count exceeds CUDA Event budget")
    # Retain all pairs until the final stop completes. Allocate wrappers before
    # enqueueing to reduce host-side gaps; CUDA may still initialize events lazily.
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(sample_count)]
    if sampler is not None:
        sampler.begin_timed()
    started = time.perf_counter()
    for start, stop in events:
        start.record()
        torch.mm(a, b, out=out)
        stop.record()
    events[-1][1].synchronize()
    wall_seconds = time.perf_counter() - started
    if sampler is not None:
        sampler.end_timed()
    if measurement is not None:
        measurement.update({"queued_wall_seconds": wall_seconds,
                            "gpu_span_seconds": events[0][0].elapsed_time(events[-1][1]) / 1000})
    return [start.elapsed_time(stop) for start, stop in events]


def run(args):
    # Validate operator input before importing GPU dependencies or allocating.
    experiment = load_experiment(args.experiment_metadata)
    import torch

    validate_dimensions(args.m, args.n, args.k)
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError("requested CUDA device unavailable")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    precision = configure_mode(torch, args.mode)
    dtype = torch.float16 if args.mode == "fp16" else torch.bfloat16 if args.mode == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    props = torch.cuda.get_device_properties(args.gpu)
    measured_pci = cuda_pci_bus_id(torch, args.gpu)
    evidence = state_evidence(args.requested_state, experiment, measured_pci)
    session = NvmlSession(measured_pci)
    try:
        result = run_measurement(args, torch, device, dtype, precision, props, measured_pci, evidence, experiment, session)
        result["compute_processes"]["after"] = session.compute_process_count()
        reasons = result["baseline_ineligibility_reasons"]
        if not result["identity_verified"]:
            reasons.append("physical_identity_unverified")
        if result["device"]["pseudonym"]["status"] != "available":
            reasons.append("stable_gpu_pseudonym_unavailable")
        if evidence["initial_state_confidence"] != "high":
            reasons.append("initial_state_evidence_insufficient")
        if experiment is None:
            reasons.append("cooling_history_not_documented")
        if not result.get("accuracy", {}).get("passed"):
            reasons.append("numerical_validation_failed")
        if result.get("precision_mode_suspect"):
            reasons.append("precision_mode_suspect")
        if result["telemetry"].get("timed_coverage", {}).get("sample_count", 0) == 0:
            reasons.append("timed_telemetry_missing")
        if session.status != "available":
            reasons.append("nvml_telemetry_unavailable")
        if any(result["compute_processes"][point]["status"] != "available" for point in ("before", "after")):
            reasons.append("compute_process_evidence_unavailable")
        if result.get("timed_measurement", {}).get("target_met") is False:
            reasons.append("requested_timed_duration_not_met")
        return result
    finally:
        session.close()


def run_measurement(args, torch, device, dtype, precision, props, measured_pci, evidence, experiment, session):
    result = {
        "schema_version": "0.2.0-draft",
        "status": "unverified_draft",
        "warning": "Telemetry/evidence draft; NOT a publishable baseline; full schema validation and independent-run review pending",
        "baseline_eligible": False,
        "baseline_ineligibility_reasons": ["pr4_admission_validator_not_implemented", "independent_run_review_pending"],
        "identity_verified": session.identity_verified,
        "identity_verification_reason": None if session.identity_verified else session.reason,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "device": {"cuda_index": args.gpu, "name": props.name, "pci_bus_id": measured_pci,
                   "pci_mapping_status": "available" if measured_pci else "unavailable", "nvml": session.metadata(),
                   "pseudonym": session.pseudonym(args.hmac_key_file)},
        "compute_processes": {"before": session.compute_process_count(), "after": None,
                              "note": "value counts all compute processes; other_processes excludes this process when identifiable; endpoint zero does not prove no contention; no PIDs retained; PID namespaces and MPS may limit visibility"},
        "software": {"python": sys.version.split()[0], "pytorch": torch.__version__, "pytorch_cuda": torch.version.cuda},
        "environment": environment_metadata(),
        **{key: evidence[key] for key in ("requested_state", "verification_status", "initial_state_confidence")},
        "unlock_state_evidence": evidence,
        "experiment_metadata": experiment,
        "telemetry": {"status": "not_collected", "reason": "pre_accuracy_check_not_passed",
                      "sampling_interval_seconds": args.telemetry_interval, "sample_count": 0},
        "precision": precision,
        "seed": args.seed,
        "precision_mode_suspect": False,
        "timing_method": "queued_cuda_events_per_operation_final_sync",
        "warmup": {"target_seconds": args.warmup_seconds, "operations": 0},
    }
    a = torch.randn((args.m, args.k), dtype=dtype, device=device)
    b = torch.randn((args.k, args.n), dtype=dtype, device=device)
    out = torch.empty((args.m, args.n), dtype=dtype, device=device)

    with torch.no_grad():
        torch.mm(a, b, out=out)
        torch.cuda.synchronize(device)
        pre = accuracy_result(torch, a, b, out, args)
        result["accuracy_checks"] = {"pre": pre, "post": None}
        result["accuracy"] = dict(pre)
        result["precision_mode_suspect"] = pre["precision_mode_suspect"]
        if not pre["passed"]:
            result["status"] = "accuracy_failed"
            result["failure_reason"] = "Numerical validation failed; timing skipped"
            return result

        torch.cuda.synchronize(device)
        with Sampler(session, args.telemetry_interval, enabled=args.telemetry == "on") as sampler:
            warmup_events = None
            if args.min_timed_seconds is not None:
                warmup_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                warmup_events[0].record()
            start_warmup = time.monotonic()
            warmups = 0
            while time.monotonic() - start_warmup < args.warmup_seconds:
                torch.mm(a, b, out=out)
                warmups += 1
                if warmups % 16 == 0:
                    torch.cuda.synchronize(device)
            torch.cuda.synchronize(device)
            estimated_ms = None
            if warmup_events is not None:
                warmup_events[1].record()
                warmup_events[1].synchronize()
                warmup_ms = warmup_events[0].elapsed_time(warmup_events[1])
                if warmups and math.isfinite(warmup_ms) and warmup_ms > 0:
                    estimated_ms = warmup_ms / warmups
            plan = select_samples(args.samples, args.min_timed_seconds, estimated_ms)
            measurement = {}
            samples = measure_queued(torch, a, b, out, plan["selected_samples"], measurement=measurement, sampler=sampler)
        result["telemetry"] = sampler.result()
        result["timed_measurement"] = {**plan, **measurement,
                                      "sum_operation_seconds": sum(samples) / 1000,
                                      "target_met": (measurement.get("gpu_span_seconds", 0) >= args.min_timed_seconds) if args.min_timed_seconds is not None else None,
                                      "estimate_note": "warm-up GPU span per operation includes queue gaps and synchronization; target is an estimate, not adaptive remeasurement"}
        # Validate the last timed output without running another GEMM into out.
        post = accuracy_result(torch, a, b, out, args)
        result["accuracy_checks"]["post"] = post
        result["accuracy"] = dict(post)
        result["accuracy"]["passed"] = pre["passed"] and post["passed"]
        result["precision_mode_suspect"] = pre["precision_mode_suspect"] or post["precision_mode_suspect"]
        result["accuracy"]["precision_mode_suspect"] = result["precision_mode_suspect"]
    result["status"] = "draft_validated"
    if not result["accuracy"]["passed"]:
        result["status"] = "accuracy_failed"
        result["failure_reason"] = "Post-timing numerical validation failed; timings retained for diagnostics only"
    elif result["precision_mode_suspect"]:
        result["status"] = "precision_mode_suspect"
        result["investigation_reason"] = "TF32 error below screening floor; verify kernel usage before baseline inclusion"
    result["warmup"]["operations"] = warmups
    result["timings_ms"] = samples
    result["summary"] = timing_summary(samples, args.m, args.n, args.k)
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run(args)
    except (IdentityMismatch, ImportError, RuntimeError) as exc:
        mismatch = isinstance(exc, IdentityMismatch)
        result = {"schema_version": "0.2.0-draft", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "identity_mismatch" if mismatch else "measurement_unavailable",
                  "identity_verified": False, "identity_verification_reason": "physical_gpu_identity_mismatch" if mismatch else "gpu_dependency_or_measurement_unavailable",
                  "baseline_eligible": False,
                  "baseline_ineligibility_reasons": ["pr4_admission_validator_not_implemented", "physical_gpu_identity_mismatch" if mismatch else "measurement_unavailable"],
                  "requested_state": args.requested_state, "verification_status": "unconfirmed", "initial_state_confidence": "unknown"}
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Unique path by default; never silently overwrite existing results.
        with args.output.open("x", encoding="utf-8") as f:
            f.write(serialized + "\n")
        print(f"Wrote {args.output}")
    else:
        print(serialized)
    if result["status"] == "precision_mode_suspect":
        return 3
    if result["status"] == "identity_mismatch":
        return IDENTITY_MISMATCH_EXIT
    if result["status"] == "measurement_unavailable":
        return 5
    return 0 if result["status"] == "draft_validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
