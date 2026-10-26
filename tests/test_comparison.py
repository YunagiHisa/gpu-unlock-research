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


def result(tflops=20):
    return {"status": "draft_validated", "identity_verified": True,
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
        summary = compare.comparison_summary([{"condition": "off", "exit_code": 0, "result": result()}])
        for stats in summary["conditions"]["off"]["timed_environment"].values():
            self.assertEqual(stats["status"], "unavailable")
            self.assertEqual(stats["samples"], 0)

    def test_stats_identity_and_configuration_drift(self):
        records = [{"condition": condition, "exit_code": 0, "result": result(20 + i)}
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
            target.write_text(json.dumps(result()))
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "runs"
            with patch.object(compare.subprocess, "run", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                code = compare.main(["--mode", "fp16", "--samples", "30", "--hmac-key-file", "/synthetic/private.key", "--output-dir", str(target)])
            report = json.loads((target / "comparison.json").read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["execution_order"], ["on", "off", "off", "on", "on", "off"])
            self.assertEqual(len(commands), 6)
            self.assertTrue(all(command[command.index("--samples") + 1] == "30" for command in commands))
            self.assertNotIn("private.key", json.dumps(report))

    def test_failure_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "runs"
            with patch.object(compare.subprocess, "run", side_effect=subprocess.TimeoutExpired("private-command", 1)) as run, contextlib.redirect_stdout(io.StringIO()):
                code = compare.main(["--mode", "fp16", "--output-dir", str(target)])
            self.assertEqual(code, 5)
            run.assert_called_once()
            saved = (target / "00-on.json").read_text()
            self.assertNotIn("private-command", saved)
            self.assertTrue((target / "comparison.json").exists())
