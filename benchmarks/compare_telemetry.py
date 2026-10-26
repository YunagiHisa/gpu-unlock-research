"""Diagnostic ON/OFF comparison using independent alternating child processes.

Never counts as Before baseline and never changes the GEMM measurement logic.
OFF disables the worker; both conditions retain read-only endpoint snapshots.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from benchmarks.core import MAX_TIMING_SAMPLES, MODES, percentile
from benchmarks.gemm import positive_float, positive_int


def execution_order(runs):
    return [condition for pair in range(runs) for condition in (("on", "off") if pair % 2 == 0 else ("off", "on"))]


def distribution(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    mean = statistics.mean(values) if values else None
    cv_available = len(values) >= 2 and bool(mean)
    return {"status": "available" if values else "unavailable",
            "reason": None if values else "no_valid_samples",
            "samples": len(values), "median": statistics.median(values) if values else None,
            "iqr": percentile(values, 75) - percentile(values, 25) if values else None,
            "cv_sample_std_over_mean": statistics.stdev(values) / mean if cv_available else None,
            "cv_status": "available" if cv_available else "unavailable",
            "cv_unavailability_reason": None if cv_available else ("fewer_than_two_valid_samples" if len(values) < 2 else "zero_mean")}


def comparison_summary(records):
    conditions = {}
    usable = [row for row in records if row["exit_code"] == 0 and row["result"].get("status") == "draft_validated"]
    for condition in ("on", "off"):
        results = [row["result"] for row in usable if row["condition"] == condition]
        endpoints = [sample for result in results for sample in result.get("telemetry", {}).get("samples", [])
                     if sample.get("source", "").startswith("endpoint_")]
        def environment(samples):
            return {field: distribution([s[field]["value"] for s in samples if s[field]["status"] == "available"])
                    for field in ("temperature_c", "power_w", "sm_clock_mhz")}
        timed_samples = []
        for result in results:
            telemetry = result.get("telemetry", {})
            coverage = telemetry.get("timed_coverage", {})
            start, end = coverage.get("start_seconds"), coverage.get("end_seconds")
            if start is not None and end is not None:
                timed_samples.extend(s for s in telemetry.get("samples", []) if s.get("source") == "poll"
                                     and start <= s["elapsed_seconds"] and s["completed_seconds"] <= end)
        conditions[condition] = {
            "valid_process_runs": len(results),
            "tflops": distribution([r.get("summary", {}).get("tflops_from_median_latency") for r in results]),
            "gpu_span_seconds": distribution([r.get("timed_measurement", {}).get("gpu_span_seconds") for r in results]),
            "queued_wall_seconds": distribution([r.get("timed_measurement", {}).get("queued_wall_seconds") for r in results]),
            "endpoint_environment": environment(endpoints),
            "endpoint_environment_by_phase": {phase: environment([s for s in endpoints if s["source"] == phase])
                                              for phase in ("endpoint_before", "endpoint_after")},
            "timed_environment": environment(timed_samples),
        }
    all_results = [row["result"] for row in records]
    identities = {(r.get("device", {}).get("pci_bus_id"), r.get("device", {}).get("pseudonym", {}).get("value")) for r in all_results}
    configs = {(r.get("mode"), tuple(sorted(r.get("shape", {}).items())), r.get("timed_measurement", {}).get("selected_samples"), r.get("seed")) for r in all_results}
    verified = bool(records) and all(r.get("identity_verified") is True for r in all_results) and len(identities) == 1
    a, b = conditions["on"]["tflops"]["median"], conditions["off"]["tflops"]["median"]
    return {"baseline_eligible": False, "purpose": "sampler_overhead_diagnostic_not_before_baseline",
            "identity_verified_across_runs": verified,
            "stable_pseudonym_available": bool(records) and all(r.get("device", {}).get("pseudonym", {}).get("value") for r in all_results),
            "configuration_consistent": len(configs) == 1,
            "conditions": conditions, "excluded_process_runs": len(records) - len(usable),
            "relative_median_tflops_on_vs_off": a / b - 1 if a is not None and b else None,
            "assessment": "manual_review_required_no_automatic_measurement_changes",
            "limitations": ["endpoint_zero_process_count_does_not_prove_absence_of_contention",
                           "without_hmac_pseudonym_cross_run_identity_is_pci_scoped_only",
                           "off_has_endpoint_environment_only_no_in_window_polling",
                           "temperature_power_clock_and_order_drift_can_confound_tflops_difference",
                           "no_formal_sampler_effect_threshold_in_pr3"]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", choices=MODES, required=True)
    for axis in ("m", "n", "k"):
        parser.add_argument("--" + axis, type=positive_int, default=2048)
    parser.add_argument("--samples", type=positive_int, default=100)
    parser.add_argument("--runs", type=positive_int, default=3, help="independent process runs per condition, at least 3")
    parser.add_argument("--warmup-seconds", type=positive_float, default=3)
    parser.add_argument("--telemetry-interval", type=positive_float, default=0.1)
    parser.add_argument("--timeout-seconds", type=positive_float, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experiment-metadata", type=Path)
    parser.add_argument("--hmac-key-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="new local directory; never overwritten")
    args = parser.parse_args(argv)
    if args.runs < 3 or args.samples > MAX_TIMING_SAMPLES:
        parser.error("require at least 3 process runs per condition and samples within CUDA Event budget")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for index, condition in enumerate(execution_order(args.runs)):
        filename = f"{index:02d}-{condition}.json"
        target = args.output_dir / filename
        command = [sys.executable, "-m", "benchmarks.gemm", "--telemetry", condition, "--output", str(target)]
        for name in ("gpu", "mode", "m", "n", "k", "samples", "warmup_seconds", "telemetry_interval", "seed", "experiment_metadata", "hmac_key_file"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        try:
            proc = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=args.timeout_seconds)
            code = proc.returncode
            result = json.loads(target.read_text()) if target.exists() else {"status": "child_failed_without_result"}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            code, result = 5, {"status": "child_unavailable_or_timeout"}
        if not target.exists():
            with target.open("x", encoding="utf-8") as output:
                json.dump(result, output, allow_nan=False, indent=2)
        records.append({"condition": condition, "file": filename, "exit_code": code, "result": result})
        # Preserve progress even if a subsequent process fails or is interrupted.
        report = {"execution_order": [r["condition"] for r in records], "summary": comparison_summary(records),
                  "runs": [{k: v for k, v in r.items() if k != "result"} for r in records]}
        temporary = args.output_dir / "comparison.json.tmp"
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        temporary.replace(args.output_dir / "comparison.json")
        if code != 0:
            break  # Do not select only successful retries or continue a GPU failure.
    summary = comparison_summary(records)
    print(json.dumps(summary, allow_nan=False, indent=2))
    return 0 if len(records) == args.runs * 2 and not summary["excluded_process_runs"] and summary["identity_verified_across_runs"] and summary["configuration_consistent"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
