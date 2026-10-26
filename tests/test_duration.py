import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from benchmarks import gemm
from benchmarks.core import MAX_TIMING_SAMPLES, select_samples
from benchmarks.telemetry import NvmlSession, Sampler


class TestDuration(unittest.TestCase):
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
        self.assertEqual(result["timed_coverage"]["sample_count"], 0)
        self.assertEqual(result["timed_coverage"]["status"], "missing")
        self.assertEqual(result["summary"]["temperature_c"]["mean"], 71)
        for field in result["timed_summary"].values():
            self.assertEqual(field["status"], "unavailable")
            self.assertEqual(field["available_samples"], 0)
            self.assertIsNone(field["mean"])
            self.assertIsNone(field["min"])
            self.assertIsNone(field["max"])
        with Sampler(session, 0.1, enabled=False) as sampler:
            self.assertIsNone(sampler.thread)
            sampler.begin_timed()
            sampler.end_timed()
        self.assertEqual(sampler.result()["timed_coverage"]["reason"], "polling_disabled")
        self.assertEqual(sampler.result()["timed_summary"]["power_w"]["status"], "unavailable")

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
