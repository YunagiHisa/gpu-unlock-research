import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks import gemm
from benchmarks.core import MAX_TIMING_SAMPLES, select_samples
from benchmarks.telemetry import NvmlSession, Sampler


class TestDuration(unittest.TestCase):
    def test_invalid_calibration_preserves_json_safe_evidence(self):
        for span, duration in ((float("nan"), 1), (float("inf"), 1), (0, 1), (-1, 1), (1e305, 1), (1e-323, 1),
                               (0.032, float("nan")), (0.032, 0), (0.032, float("inf"))):
            with self.subTest(span=span, duration=duration):
                def measure(*args, **kwargs):
                    kwargs["measurement"].update(gpu_span_seconds=span, queued_wall_seconds=0.04)
                    return [duration] * 32
                with patch.object(gemm, "measure_queued", side_effect=measure) as measured:
                    result = gemm.calibrate_queued(None, None, None, None)
                measured.assert_called_once()
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["reason"], "calibration_invalid_estimate")
                self.assertIsNone(result["estimated_gemm_ms"])
                self.assertEqual(result["queued_wall_seconds"], 0.04)
                json.dumps(result, allow_nan=False)

    def test_calibration_queues_once_and_uses_span_not_operation_sum(self):
        events, calls = [], []
        class Event:
            def __init__(self, **kwargs):
                self.index = len(events)
                events.append(self)
            def record(self):
                calls.append(("record", self.index))
            def synchronize(self):
                calls.append(("sync", self.index))
            def elapsed_time(self, other):
                return 64 if self.index == 0 and other.index == 63 else 1
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event), mm=Mock())
        result = gemm.calibrate_queued(torch, None, None, None)
        self.assertEqual(result["operations"], 32)
        self.assertEqual(torch.mm.call_count, 32)
        self.assertEqual(result["estimated_gemm_ms"], 2)
        self.assertEqual(result["sum_operation_seconds"], 0.032)
        self.assertEqual([c for c in calls if c[0] == "sync"], [("sync", 63)])
        self.assertNotIn("timings_ms", result)

    def test_calibration_samples_are_separate_and_require_complete_reads(self):
        session = NvmlSession(None)
        with Sampler(session, 0.1, enabled=False) as sampler:
            sampler.begin_calibration()
            sampler.end_calibration()
        sampler.calibration_start, sampler.calibration_end = 1, 2
        sampler.timed_start, sampler.timed_end = 3, 4
        template = sampler.samples[0]
        sampler.samples = [{**template, "source": "poll", "elapsed_seconds": start, "completed_seconds": end}
                           for start, end in ((1.1, 1.2), (1.9, 2.1), (2.9, 3.1), (3.1, 3.2))]
        phases = sampler.result()["phases"]
        self.assertEqual(phases["calibration"]["sample_count"], 1)
        self.assertEqual(phases["timed"]["sample_count"], 1)

    def test_sample_compatibility_estimation_and_event_budget(self):
        self.assertEqual(select_samples(30, None, None)["selected_samples"], 30)
        self.assertEqual(select_samples(30, 1, 10)["selected_samples"], 100)
        self.assertEqual(select_samples(300, 1, 10)["selected_samples"], 300)
        result = select_samples(30, 100, 0.001)
        self.assertEqual(result["selected_samples"], MAX_TIMING_SAMPLES)
        self.assertTrue(result["sample_cap_reached"])
        self.assertEqual(select_samples(30, 1, None)["selected_samples"], 30)
        with self.assertRaises(ValueError):
            select_samples(MAX_TIMING_SAMPLES + 1, None, None)

    def test_gpu_span_is_not_sum_of_operations_and_final_sync_is_preserved(self):
        calls = []
        events = []
        class Event:
            def __init__(self, **kwargs):
                self.index = len(events)
                events.append(self)
            def record(self):
                calls.append(("record", self.index))
            def synchronize(self):
                calls.append(("sync", self.index))
            def elapsed_time(self, other):
                return 30 if self.index == 0 and other.index == 3 else 10
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event), mm=Mock())
        measurement = {}
        samples = gemm.measure_queued(torch, None, None, None, 2, measurement=measurement)
        self.assertEqual(sum(samples), 20)
        self.assertEqual(measurement["gpu_span_seconds"], 0.03)
        self.assertEqual([c for c in calls if c[0] == "sync"], [("sync", 3)])
        with self.assertRaises(ValueError):
            gemm.measure_queued(torch, None, None, None, MAX_TIMING_SAMPLES + 1)
        self.assertEqual(len(events), 4)

    def test_no_timed_polls_is_missing_even_with_endpoints(self):
        session = NvmlSession(None)
        # Realistic nonzero endpoints must not substitute for absent timed polls.
        session.sample = Mock(return_value={name: {"value": value, "status": "available"} for name, value in
                                           (("temperature_c", 71), ("power_w", 167), ("fan_percent", 53),
                                            ("sm_clock_mhz", 1800), ("clock_event_reasons", 0))})
        with Sampler(session, 0.1) as sampler:
            sampler.begin_timed()
            sampler.end_timed()
        result = sampler.result()
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["phases"]["timed"]["sample_count"], 0)
        self.assertEqual(result["phases"]["timed"]["status"], "unavailable")
        self.assertEqual(result["phases"]["idle_before"]["summary"]["temperature_c"]["mean"], 71)
        for field in result["phases"]["timed"]["summary"].values():
            self.assertEqual(field["status"], "unavailable")
            self.assertEqual(field["available_samples"], 0)
            self.assertIsNone(field["mean"])
            self.assertIsNone(field["min"])
            self.assertIsNone(field["max"])
        with Sampler(session, 0.1, enabled=False) as sampler:
            self.assertIsNone(sampler.thread)
            sampler.begin_timed()
            sampler.end_timed()
        self.assertEqual(sampler.result()["phases"]["timed"]["reason"], "polling_disabled")
        self.assertEqual(sampler.result()["phases"]["timed"]["summary"]["power_w"]["status"], "unavailable")

    def test_process_counts_do_not_retain_process_objects(self):
        nvml = Mock()
        nvml.nvmlDeviceGetPciInfo.return_value = SimpleNamespace(busId="0000:01:00.0")
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [SimpleNamespace(pid=12345, name="private", path="/private")]
        session = NvmlSession("0000:01:00.0", nvml)
        self.assertEqual(session.compute_process_count()["value"], 1)
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        self.assertEqual(session.compute_process_count()["value"], 0)
        nvml.nvmlDeviceGetComputeRunningProcesses.side_effect = RuntimeError("private")
        failed = session.compute_process_count()
        self.assertEqual(failed["status"], "unavailable")
        self.assertIsNone(failed["value"])
        self.assertEqual(failed["reason"], "unavailable")
        self.assertIsNone(failed["other_processes"]["value"])
        session.close()

    def test_other_process_count_excludes_self_without_exporting_pids(self):
        import json
        from unittest.mock import patch
        nvml = Mock()
        nvml.nvmlDeviceGetPciInfo.return_value = SimpleNamespace(busId="0000:01:00.0")
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [SimpleNamespace(pid=12345), SimpleNamespace(pid=67890)]
        session = NvmlSession("0000:01:00.0", nvml)
        with patch("benchmarks.telemetry.os.getpid", return_value=12345):
            counts = session.compute_process_count()
        self.assertEqual(counts["value"], 2)
        self.assertEqual(counts["other_processes"], {"value": 1, "status": "available"})
        self.assertNotIn("12345", json.dumps(counts))
        self.assertNotIn("67890", json.dumps(counts))
        session.close()
