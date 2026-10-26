import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import compare_telemetry as compare


def result(tflops=20, condition="on"):
    empty_count = {"status": "available", "value": 1, "other_processes": {"status": "available", "value": 0}}
    return {"schema_version": compare.SCHEMA_VERSION, "status": "draft_validated", "identity_verified": True,
            "accuracy": {"passed": True}, "requested_state": "before", "execution_kind": "diagnostic",
            "preflight": {"status": "ready", "temperature": {"max_start_temp_c": 47, "timeout_seconds": 0, "passed": True}},
            "compute_processes": {point: dict(empty_count) for point in ("before_cooldown", "before", "after")},
            "telemetry": {"polling_enabled": condition == "on", "sampling_interval_seconds": 0.1,
                          "phases": {"timed": {"sample_count": 1 if condition == "on" else 0, "start_seconds": 1, "end_seconds": 2}},
                          "samples": [{"source": "poll", "elapsed_seconds": 1.1, "completed_seconds": 1.2,
                                       **{name: {"status": "available", "value": value} for name, value in
                                          (("temperature_c", 40), ("power_w", 100), ("sm_clock_mhz", 1000))}}] if condition == "on" else []},
            "device": {"pci_bus_id": "0000:01:00.0", "pseudonym": {"value": "synthetic-pseudonym"}},
            "mode": "fp16", "shape": {"m": 32, "n": 32, "k": 32}, "seed": 1,
            "summary": {"tflops_from_median_latency": tflops},
            "timed_measurement": {"selected_samples": 30, "gpu_span_seconds": 0.4, "queued_wall_seconds": 0.5}}


class TestComparison(unittest.TestCase):
    def test_cv_uses_sample_standard_deviation_and_records_sample_count(self):
        summary = compare.distribution([1, 2, 3])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["cv_sample_std_over_mean"], 0.5)
        for values in ([], [1], [0, 0]):
            stats = compare.distribution(values)
            self.assertIsNone(stats["cv_sample_std_over_mean"])
            self.assertEqual(stats["cv_status"], "unavailable")
        stats = compare.distribution([1, None, float("nan"), float("inf")])
        self.assertEqual(stats["samples"], 1)
        self.assertEqual(stats["cv_unavailability_reason"], "fewer_than_two_valid_samples")
        empty = compare.distribution([])
        self.assertEqual(empty["status"], "unavailable")
        self.assertIsNone(empty["median"])
        self.assertIsNone(empty["iqr"])
        summary = compare.comparison_summary([{"condition": "off", "exit_code": 0, "result": result(condition="off")}])
        for stats in summary["conditions"]["off"]["timed_environment"].values():
            self.assertEqual(stats["status"], "unavailable")
            self.assertEqual(stats["samples"], 0)

    def test_stats_identity_and_configuration_drift(self):
        records = [{"condition": condition, "exit_code": 0, "result": result(20 + i, condition)}
                   for i, condition in enumerate(compare.execution_order(3))]
        summary = compare.comparison_summary(records)
        self.assertEqual(summary["conditions"]["on"]["valid_process_runs"], 3)
        self.assertEqual(summary["conditions"]["off"]["tflops"]["median"], 22)
        self.assertTrue(summary["identity_verified_across_runs"])
        self.assertFalse(summary["baseline_eligible"])
        records[-1]["result"]["device"]["pseudonym"]["value"] = "other-gpu"
        records[-1]["result"]["timed_measurement"]["selected_samples"] = 31
        summary = compare.comparison_summary(records)
        self.assertFalse(summary["identity_verified_across_runs"])
        self.assertFalse(summary["configuration_consistent"])
        records[0]["exit_code"] = 2
        self.assertEqual(compare.comparison_summary(records)["excluded_process_runs"], 1)

    def test_runner_alternates_pair_order_and_uses_independent_processes(self):
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            target = Path(command[command.index("--output") + 1])
            target.write_text(json.dumps(result(condition=command[command.index("--telemetry") + 1])))
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            self.assertEqual(kwargs["timeout"], 303)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "runs"
            with patch.object(compare.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                code = compare.main(["--gpu", "0", "--requested-state", "before", "--execution-kind", "diagnostic", "--max-start-temp", "47", "--cooldown-timeout-s", "0", "--mode", "fp16", "--samples", "30", "--hmac-key-file", "/synthetic/private.key", "--output-dir", str(target)])
            report = json.loads((target / "comparison.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["execution_order"], ["on", "off", "off", "on", "on", "off"])
            self.assertEqual(len(commands), 6)
            self.assertTrue(all(command[command.index("--samples") + 1] == "30" for command in commands))
            for name, expected in (("--requested-state", "before"), ("--execution-kind", "diagnostic"),
                                   ("--max-start-temp", "47.0"), ("--cooldown-timeout-s", "0.0")):
                self.assertTrue(all(command[command.index(name) + 1] == expected for command in commands))
            self.assertNotIn("private.key", json.dumps(report))

    def test_failure_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "runs"
            with patch.object(compare.subprocess, "run", side_effect=subprocess.TimeoutExpired("private-command", 1)) as run, contextlib.redirect_stdout(io.StringIO()):
                code = compare.main(["--gpu", "0", "--requested-state", "before", "--execution-kind", "diagnostic", "--mode", "fp16", "--output-dir", str(target)])
            self.assertEqual(code, 5)
            run.assert_called_once()
            saved = (target / "00-on.json").read_text()
            self.assertNotIn("private-command", saved)
            self.assertTrue((target / "comparison.json").exists())

    def test_comparison_rejects_contention_unknown_identity_and_config_drift(self):
        import copy
        records = [{"condition": c, "exit_code": 0, "result": result(condition=c)} for c in compare.execution_order(3)]
        self.assertTrue(compare.comparison_summary(records)["comparison_valid"])
        for mutation in ("contention", "unknown_processes", "identity", "config", "missing_timed", "old_version", "power_limit"):
            rows = copy.deepcopy(records)
            last = rows[0]["result"]
            if mutation == "contention":
                last["compute_processes"]["after"]["other_processes"] = {"status": "available", "value": 1}
            elif mutation == "unknown_processes":
                last["compute_processes"]["before"] = {}
            elif mutation == "identity":
                last["identity_verified"] = False
            elif mutation == "missing_timed":
                last["telemetry"]["phases"]["timed"]["sample_count"] = 0
            elif mutation == "old_version":
                last["schema_version"] = "0.2.0-draft"
            elif mutation == "power_limit":
                last["device"]["nvml"] = {"power_limit_w": {"status": "available", "value": 120}}
            else:
                last["precision"] = {"fp32_precision": "unexpected"}
            summary = compare.comparison_summary(rows)
            self.assertFalse(summary["comparison_valid"], mutation)
            self.assertEqual(summary["status"], "invalid")
            self.assertIsNone(summary["relative_median_tflops_on_vs_off"])

    def test_uncontrolled_diagnostic_is_distinct_and_timeout_adds_wait_budgets(self):
        rows = [{"condition": c, "exit_code": 0, "result": result(condition=c)} for c in compare.execution_order(3)]
        for row in rows:
            row["result"]["preflight"] = {"status": "diagnostic_only", "temperature": {"passed": False}}
        summary = compare.comparison_summary(rows)
        self.assertEqual(summary["status"], "uncontrolled_diagnostic")
        self.assertFalse(summary["comparison_valid"])
        self.assertIsNone(summary["relative_median_tflops_on_vs_off"])
        args = compare.parse_args(["--gpu", "0", "--requested-state", "before", "--mode", "fp16", "--output-dir", "synthetic",
                                   "--max-start-temp", "50", "--cooldown-timeout-s", "120", "--timeout-seconds", "30", "--warmup-seconds", "2"])
        self.assertEqual(args.execution_kind, "diagnostic")
        self.assertEqual(compare.child_timeout(args), 152)
