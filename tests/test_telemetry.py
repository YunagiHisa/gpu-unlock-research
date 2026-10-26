"""NVML fakes exercise collection without importing PyTorch or NVIDIA libraries."""
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.telemetry import NvmlSession, Sampler, counter_deltas, read_value
from benchmarks.identity import IdentityMismatch


class NVMLError_NotSupported(Exception):
    pass


class NVMLError_NoPermission(Exception):
    pass


class FakeNvml:
    def __init__(self):
        self.calls = []
        self.shutdowns = 0
        self.counter = 100

    def nvmlInit(self):
        self.calls.append("init")

    def nvmlShutdown(self):
        self.shutdowns += 1

    def nvmlDeviceGetHandleByPciBusId(self, pci):
        self.calls.append(pci)
        return "pci-selected-handle"

    def nvmlDeviceGetPciInfo(self, handle):
        assert handle == "pci-selected-handle"
        return SimpleNamespace(busId=b"00000000:05:00.0")

    def nvmlDeviceGetClockInfo(self, handle, clock):
        return 1800

    def nvmlDeviceGetPowerUsage(self, handle):
        return 167000

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 71

    def nvmlDeviceGetFanSpeed(self, handle):
        return 53

    def nvmlDeviceGetCurrentClocksEventReasons(self, handle):
        return 0x24  # SW power cap and SW thermal slowdown.

    def nvmlDeviceGetViolationStatus(self, handle, policy):
        return SimpleNamespace(violationTime=self.counter + policy, referenceTime=self.counter * 10)


class TestTelemetry(unittest.TestCase):
    def session(self):
        nvml = FakeNvml()
        return NvmlSession("0000:05:00.0", nvml), nvml

    def test_pci_lookup_units_and_metadata_availability(self):
        session, nvml = self.session()
        self.assertEqual(nvml.calls, ["init", "0000:05:00.0"])
        sample = session.sample()
        self.assertEqual(sample["power_w"]["value"], 167)
        self.assertEqual(sample["temperature_c"]["value"], 71)
        self.assertEqual(sample["fan_percent"]["value"], 53)
        self.assertEqual(session.metadata()["vbios_version"]["status"], "api_unavailable")
        nvml.nvmlDeviceSetPowerManagementLimit = Mock()
        rejected = session.query("nvmlDeviceSetPowerManagementLimit", 170000)
        self.assertEqual(rejected["status"], "non_read_only_api_rejected")
        nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        session.close()
        session.close()
        self.assertEqual(nvml.shutdowns, 1)

    def test_pci_mismatch_never_samples_wrong_gpu(self):
        nvml = FakeNvml()
        with self.assertRaises(IdentityMismatch):
            NvmlSession("0000:06:00.0", nvml)
        self.assertEqual(nvml.shutdowns, 1)

    def test_missing_dependency_and_init_failure(self):
        with patch("benchmarks.telemetry.importlib.import_module", side_effect=ImportError("private path")):
            session = NvmlSession("0000:05:00.0")
        self.assertEqual(session.status, "unavailable")
        self.assertNotIn("private", json.dumps(session.sample()))
        nvml = FakeNvml()
        nvml.nvmlInit = Mock(side_effect=RuntimeError("private"))
        session = NvmlSession("0000:05:00.0", nvml)
        session.close()
        self.assertEqual(nvml.shutdowns, 0)

    def test_unknown_cuda_pci_does_not_initialize_nvml(self):
        nvml = Mock()
        session = NvmlSession(None, nvml)
        nvml.nvmlInit.assert_not_called()
        self.assertEqual(session.reason, "cuda_pci_unavailable")

    def test_unsupported_permission_denied_and_legacy_reasons(self):
        session, nvml = self.session()
        nvml.nvmlDeviceGetFanSpeed = Mock(side_effect=NVMLError_NotSupported("secret"))
        nvml.nvmlDeviceGetTemperature = Mock(side_effect=NVMLError_NoPermission("secret"))
        nvml.nvmlDeviceGetCurrentClocksEventReasons = Mock(side_effect=AttributeError)
        nvml.nvmlDeviceGetCurrentClocksThrottleReasons = Mock(return_value=0)
        sample = session.sample()
        self.assertEqual(sample["fan_percent"], {"value": None, "status": "unsupported"})
        self.assertEqual(sample["temperature_c"]["status"], "permission_denied")
        self.assertEqual(sample["clock_event_reasons"]["value"], 0)
        self.assertIsNone(read_value(lambda: float("nan"))["value"])
        session.close()

    def test_counter_delta_reset_and_unavailable(self):
        session, nvml = self.session()
        before = session.counters()
        nvml.counter += 25
        after = session.counters()
        self.assertEqual(counter_deltas(before, after)["thermal"]["value"], 25)
        self.assertEqual(counter_deltas(after, before)["thermal"]["status"], "counter_reset")
        nvml.nvmlDeviceGetViolationStatus = Mock(side_effect=NVMLError_NotSupported)
        self.assertIsNone(counter_deltas(before, session.counters())["power"]["value"])
        session.close()

    def test_sampler_async_observations_and_duration_counters(self):
        session, nvml = self.session()
        sampled = threading.Event()
        original = session.sample
        def sample():
            if threading.current_thread().name == "nvml-telemetry":
                sampled.set()
            return original()
        session.sample = sample
        with Sampler(session, 0.001) as sampler:
            self.assertTrue(sampled.wait(1), "worker did not sample asynchronously")
            nvml.counter += 10
        self.assertFalse(sampler.thread.is_alive())
        result = sampler.result()
        self.assertGreaterEqual(result["sample_count"], 3)
        self.assertEqual(result["clock_events"]["observed_samples_by_reason"]["sw_thermal_slowdown"], result["sample_count"])
        self.assertEqual(result["violation_counters"]["delta_ns"]["thermal"]["value"], 10)
        self.assertEqual(result["summary"]["fan_percent"]["mean"], 53)
        json.dumps(result, allow_nan=False)
        session.close()

    def test_cleanup_on_benchmark_exception(self):
        session, nvml = self.session()
        with self.assertRaisesRegex(RuntimeError, "benchmark"):
            try:
                with Sampler(session, 1) as sampler:
                    raise RuntimeError("benchmark failed")
            finally:
                session.close()
        self.assertFalse(sampler.thread.is_alive())
        self.assertEqual(nvml.shutdowns, 1)

    def test_unavailable_reasons_are_not_zero_event_counts(self):
        session = NvmlSession(None)
        with Sampler(session, 1) as sampler:
            pass
        result = sampler.result()
        self.assertIsNone(result["clock_events"]["observed_samples_by_reason"]["sw_thermal_slowdown"])
        self.assertIsNone(result["summary"]["fan_percent"]["mean"])
        self.assertEqual(result["summary"]["fan_percent"]["unavailable_samples"], 2)
        session.close()
