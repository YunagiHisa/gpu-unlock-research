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
from benchmarks.gemm import SCHEMA_VERSION, positive_float, positive_int, preflight_arguments, validate_preflight_arguments
from benchmarks.preflight import contention_reasons


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


def configuration(result):
    temperature = result.get("preflight", {}).get("temperature", {})
    metadata = result.get("device", {}).get("nvml", {})
    return json.dumps({"schema_version": result.get("schema_version"),
                       "mode": result.get("mode"), "shape": result.get("shape"), "seed": result.get("seed"),
                       "requested_state": result.get("requested_state"), "execution_kind": result.get("execution_kind"),
                       "precision": result.get("precision"), "software": result.get("software"),
                       "experiment_metadata": result.get("experiment_metadata"),
                       "environment": result.get("environment"),
                       "device_settings": {key: metadata.get(key) for key in ("power_limit_w", "power_default_limit_w", "vbios_version", "driver_version", "nvml_version", "persistence_mode")},
                       "warmup_seconds": result.get("warmup", {}).get("target_seconds"),
                       "sampling_interval_seconds": result.get("telemetry", {}).get("sampling_interval_seconds"),
                       "selected_samples": result.get("timed_measurement", {}).get("selected_samples"),
                       "max_start_temp_c": temperature.get("max_start_temp_c"), "timeout_seconds": temperature.get("timeout_seconds")},
                      sort_keys=True)


def run_reasons(row):
    result = row["result"]
    reasons = []
    if result.get("schema_version") != SCHEMA_VERSION:
        reasons.append("unsupported_draft_version")
    if row["exit_code"] != 0 or result.get("status") != "draft_validated" or result.get("accuracy", {}).get("passed") is not True:
        reasons.append("measurement_failed")
    if result.get("timed_measurement", {}).get("target_met") is False:
        reasons.append("requested_timed_duration_not_met")
    if result.get("identity_verified") is not True:
        reasons.append("identity_unverified")
    for point in ("before_cooldown", "before", "after"):
        reasons.extend(contention_reasons(result.get("compute_processes", {}).get(point, {})))
    if result.get("preflight", {}).get("status") == "blocked":
        reasons.append("preflight_blocked")
    if result.get("telemetry", {}).get("polling_enabled") is not (row["condition"] == "on"):
        reasons.append("telemetry_condition_mismatch")
    if row["condition"] == "on" and result.get("telemetry", {}).get("phases", {}).get("timed", {}).get("sample_count", 0) < 1:
        reasons.append("timed_telemetry_missing")
    shape = result.get("shape", {})
    selected = result.get("timed_measurement", {}).get("selected_samples")
    if (result.get("mode") not in MODES or result.get("requested_state") not in ("before", "after", "reverted")
            or result.get("execution_kind") not in ("formal", "diagnostic") or type(result.get("seed")) is not int
            or any(type(shape.get(axis)) is not int or shape[axis] <= 0 for axis in ("m", "n", "k"))
            or type(selected) is not int or selected <= 0):
        reasons.append("configuration_incomplete")
    return list(dict.fromkeys(reasons))


def comparison_summary(records, expected_runs=3):
    conditions = {}
    usable = [row for row in records if not run_reasons(row)]
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
            coverage = telemetry.get("phases", {}).get("timed", {})
            start, end = coverage.get("start_seconds"), coverage.get("end_seconds")
            if start is not None and end is not None:
                timed_samples.extend(s for s in telemetry.get("samples", []) if s.get("source") == "poll"
                                     and start <= s["elapsed_seconds"] and s["completed_seconds"] <= end)
        conditions[condition] = {
            "valid_process_runs": len(results),
            "tflops": distribution([r.get("summary", {}).get("tflops_from_median_latency") for r in results]),
            "gpu_span_seconds": distribution([r.get("timed_measurement", {}).get("gpu_span_seconds") for r in results]),
            "queued_wall_seconds": distribution([r.get("timed_measurement", {}).get("queued_wall_seconds") for r in results]),
            "endpoint_environment_by_phase": {phase: environment([s for s in endpoints if s["source"] == phase])
                                              for phase in ("endpoint_before", "endpoint_after")},
            "timed_environment": environment(timed_samples),
        }
    all_results = [row["result"] for row in records]
    identities = {(r.get("device", {}).get("pci_bus_id"), r.get("device", {}).get("pseudonym", {}).get("value")) for r in all_results}
    configs = {configuration(result) for result in all_results}
    verified = bool(records) and all(r.get("identity_verified") is True for r in all_results) and len(identities) == 1
    a, b = conditions["on"]["tflops"]["median"], conditions["off"]["tflops"]["median"]
    reasons = list(dict.fromkeys(reason for row in records for reason in run_reasons(row)))
    if not verified:
        reasons.append("identity_inconsistent_or_unverified")
    if len(configs) != 1:
        reasons.append("configuration_mismatch")
    if any(conditions[condition]["valid_process_runs"] != expected_runs for condition in ("on", "off")):
        reasons.append("independent_runs_incomplete")
    controlled = bool(records) and all(r.get("preflight", {}).get("temperature", {}).get("passed") is True
                                      and r["preflight"]["temperature"].get("max_start_temp_c") is not None
                                      and r["preflight"]["temperature"].get("timeout_seconds") is not None for r in all_results)
    if not controlled:
        reasons.append("temperature_gate_uncontrolled")
    valid = not reasons
    status = "controlled_comparison" if valid else "uncontrolled_diagnostic" if reasons == ["temperature_gate_uncontrolled"] else "invalid"
    return {"baseline_eligible": False, "status": status, "comparison_valid": valid, "invalidity_reasons": reasons,
            "temperature_control": "controlled" if controlled else "uncontrolled",
            "identity_verified_across_runs": verified,
            "stable_pseudonym_available": bool(records) and all(r.get("device", {}).get("pseudonym", {}).get("value") for r in all_results),
            "configuration_consistent": len(configs) == 1,
            "conditions": conditions, "excluded_process_runs": len(records) - len(usable),
            "relative_median_tflops_on_vs_off": a / b - 1 if valid and a is not None and b else None}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    preflight_arguments(parser)
    parser.set_defaults(execution_kind="diagnostic")
    parser.add_argument("--mode", choices=MODES, required=True)
    for axis in ("m", "n", "k"):
        parser.add_argument("--" + axis, type=positive_int, default=2048)
    parser.add_argument("--samples", type=positive_int, default=100)
    parser.add_argument("--runs", type=positive_int, default=3, help="independent process runs per condition, at least 3")
    parser.add_argument("--warmup-seconds", type=positive_float, default=3)
    parser.add_argument("--telemetry-interval", type=positive_float, default=0.1)
    parser.add_argument("--timeout-seconds", type=positive_float, default=300, help="child benchmark/startup budget; cooldown and warmup budgets are added")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experiment-metadata", type=Path)
    parser.add_argument("--hmac-key-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="new local directory; never overwritten")
    args = parser.parse_args(argv)
    validate_preflight_arguments(parser, args)
    if args.runs < 3 or args.samples > MAX_TIMING_SAMPLES:
        parser.error("require at least 3 process runs per condition and samples within CUDA Event budget")
    return args


def child_timeout(args):
    return args.timeout_seconds + (args.cooldown_timeout_s or 0) + args.warmup_seconds


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for index, condition in enumerate(execution_order(args.runs)):
        filename = f"{index:02d}-{condition}.json"
        target = args.output_dir / filename
        command = [sys.executable, "-m", "benchmarks.gemm", "--telemetry", condition, "--output", str(target)]
        for name in ("gpu", "requested_state", "execution_kind", "max_start_temp", "cooldown_timeout_s", "mode", "m", "n", "k", "samples", "warmup_seconds", "telemetry_interval", "seed", "experiment_metadata", "hmac_key_file"):
            value = getattr(args, name)
            if value is not None:
                command.extend(["--" + name.replace("_", "-"), str(value)])
        try:
            proc = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=child_timeout(args))
            code = proc.returncode
            result = json.loads(target.read_text()) if target.exists() else {"status": "child_failed_without_result"}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            code, result = 5, {"status": "child_unavailable_or_timeout"}
        if not target.exists():
            with target.open("x", encoding="utf-8") as output:
                json.dump(result, output, allow_nan=False, indent=2)
        records.append({"condition": condition, "file": filename, "exit_code": code, "result": result})
        # Preserve progress even if a subsequent process fails or is interrupted.
        report = {"schema_version": SCHEMA_VERSION, "child_timeout_seconds": child_timeout(args),
                  "execution_order": [r["condition"] for r in records], "summary": comparison_summary(records, args.runs),
                  "runs": [{k: v for k, v in r.items() if k != "result"} for r in records]}
        temporary = args.output_dir / "comparison.json.tmp"
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2) + "\n")
        temporary.replace(args.output_dir / "comparison.json")
        if code != 0 or run_reasons(records[-1]):
            break  # Do not select only successful retries or continue a GPU failure.
    summary = comparison_summary(records, args.runs)
    print(json.dumps(summary, allow_nan=False, indent=2))
    return 0 if summary["status"] in ("controlled_comparison", "uncontrolled_diagnostic") else 5


if __name__ == "__main__":
    raise SystemExit(main())
