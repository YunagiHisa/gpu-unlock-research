"""CPU-only tests of telemetry boundaries in the real GEMM orchestration."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from benchmarks import gemm


class TestIntegration(unittest.TestCase):
    def run_fake(self, errors, measurement_failure=False, min_seconds=None):
        calls = []
        torch = MagicMock()
        torch.__version__ = "test"
        torch.version.cuda = "test"
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(name="test")
        torch.cuda.Event.return_value.elapsed_time.return_value = 10.0
        session = Mock()
        session.status = "available"
        session.identity_verified = True
        session.pseudonym.return_value = {"status": "unavailable", "value": None, "reason": "key_missing"}
        session.compute_process_count.return_value = {"status": "available", "value": 1}
        session.metadata.return_value = {}
        def counters():
            calls.append("counter")
            return {name: {"value": {"violation_time_ns": 0, "reference_time_us": 1}, "status": "available"}
                    for name in ("power", "thermal")}
        def sample():
            calls.append("sample")
            return {name: {"value": 0, "status": "available"} for name in
                    ("power_w", "temperature_c", "fan_percent", "sm_clock_mhz", "clock_event_reasons")}
        session.counters.side_effect = counters
        session.sample.side_effect = sample
        session.close.side_effect = lambda: calls.append("close")
        errors = iter(errors)
        def accuracy(*args):
            calls.append("accuracy")
            error = next(errors)
            return {"passed": error < 0.001, "normalized_l2_error": error, "precision_mode_suspect": False}
        def measure(*args, **kwargs):
            calls.append("timing")
            kwargs["sampler"].begin_timed()
            if measurement_failure:
                raise RuntimeError("timing failure")
            kwargs["sampler"].end_timed()
            kwargs["measurement"].update({"gpu_span_seconds": args[4] / 1000, "queued_wall_seconds": args[4] / 1000 + 0.001})
            return [1] * args[4]
        args = gemm.parse_args(["--mode", "fp16", "--samples", "2", "--telemetry-interval", "1000"])
        args.min_timed_seconds = min_seconds
        with patch.dict("sys.modules", {"torch": torch}), \
             patch.object(gemm, "configure_mode", return_value={}), \
             patch.object(gemm, "cuda_pci_bus_id", return_value="0000:05:00.0"), \
             patch.object(gemm, "NvmlSession", return_value=session), \
             patch.object(gemm, "environment_metadata", return_value={}), \
             patch("benchmarks.evidence.kernel_evidence", return_value={}), \
             patch.object(gemm, "accuracy_result", side_effect=accuracy), \
             patch.object(gemm, "measure_queued", side_effect=measure), \
             patch.object(gemm.time, "monotonic", side_effect=[0, 0, 4] if min_seconds else [0, 4]):
            if measurement_failure:
                with self.assertRaisesRegex(RuntimeError, "timing failure"):
                    gemm.run(args)
                return calls
            result = gemm.run(args)
        return result, calls

    def test_collection_excludes_endpoint_accuracy_checks(self):
        result, calls = self.run_fake([0.0001, 0.0001])
        self.assertEqual(calls, ["accuracy", "counter", "sample", "timing", "sample", "counter", "accuracy", "close"])
        self.assertEqual(result["device"]["pci_bus_id"], "0000:05:00.0")
        self.assertEqual(result["telemetry"]["sample_count"], 2)
        self.assertFalse(result["baseline_eligible"])
        self.assertEqual(result["verification_status"], "unconfirmed")
        json.dumps(result, allow_nan=False)

    def test_pre_failure_skips_sampling_but_closes_nvml(self):
        result, calls = self.run_fake([0.5])
        self.assertEqual(calls, ["accuracy", "close"])
        self.assertEqual(result["telemetry"]["status"], "not_collected")
        self.assertEqual(result["telemetry"]["sample_count"], 0)

    def test_post_failure_retains_telemetry_and_timing(self):
        result, calls = self.run_fake([0.0001, 0.5])
        self.assertEqual(result["status"], "accuracy_failed")
        self.assertEqual(result["telemetry"]["sample_count"], 2)
        self.assertEqual(result["timings_ms"], [1, 1])
        self.assertEqual(calls[-1], "close")

    def test_measurement_exception_stops_sampler_before_nvml_shutdown(self):
        calls = self.run_fake([0.0001], measurement_failure=True)
        self.assertEqual(calls[-3:], ["sample", "counter", "close"])

    def test_invalid_interval_cli(self):
        for interval in ("0", "-1", "nan", "inf"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gemm.parse_args(["--mode", "fp16", "--telemetry-interval", interval])

    def test_warmup_estimate_and_actual_duration_are_distinct(self):
        result, _ = self.run_fake([0.0001, 0.0001], min_seconds=1)
        plan = result["timed_measurement"]
        self.assertEqual(plan["estimated_gemm_ms"], 10)
        self.assertEqual(plan["selected_samples"], 100)
        self.assertEqual(plan["predicted_timed_seconds"], 1)
        self.assertEqual(plan["gpu_span_seconds"], 0.1)
        self.assertFalse(plan["target_met"])
        self.assertIn("requested_timed_duration_not_met", result["baseline_ineligibility_reasons"])
