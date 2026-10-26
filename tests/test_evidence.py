"""CPU-only operator input, privacy, state rules, and identity tests."""
import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.evidence import initial_confidence, kernel_evidence, load_experiment, pci_id, state_evidence
from benchmarks.metadata import cuda_pci_bus_id, environment_metadata, git_metadata
from benchmarks.identity import IdentityMismatch


class TestEvidence(unittest.TestCase):
    def test_pci_normalization(self):
        self.assertEqual(pci_id(b"00000000:0A:00.0"), "0000:0a:00.0")
        for invalid in [None, 0, "GPU-secret", "01:00.0", "0000:01:00.8"]:
            with self.assertRaises(ValueError):
                pci_id(invalid)

    def test_before_rules_and_no_automatic_unlock_confirmation(self):
        for operator, confidence in [({}, "unknown"), ({"residual_state_suspected": True}, "low"),
                                     ({"unlock_history": "never_attempted", "documented_at_utc": "2026-01-01T00:00:00Z"}, "high")]:
            experiment = {"pci_bus_id": "0000:01:00.0", "initial_state": operator}
            for requested in ("before", "after", "reverted"):
                with patch("benchmarks.evidence.kernel_evidence", return_value={"module_present": True}):
                    result = state_evidence(requested, experiment, "0000:01:00.0")
                self.assertEqual(result["initial_state_confidence"], confidence)
                self.assertEqual(result["verification_status"], "not_applicable" if requested == "before" and confidence == "high" else "unconfirmed")

    def test_physical_cycle_must_follow_last_attempt(self):
        evidence = {"unlock_history": "attempted", "documented_at_utc": "2026-01-03T00:00:00Z",
                    "last_attempt_utc": "2026-01-01T00:00:00Z",
                    "power_cycle": {"completed_at_utc": "2026-01-02T00:00:00Z",
                                    "method": "host_shutdown_ac_disconnect_power_on",
                                    "host_shutdown": True, "ac_disconnected": True, "fresh_power_on": True}}
        self.assertEqual(initial_confidence(evidence), "high")
        for missing in ("host_shutdown", "ac_disconnected", "fresh_power_on", "method", "completed_at_utc"):
            changed = copy.deepcopy(evidence)
            del changed["power_cycle"][missing]
            self.assertEqual(initial_confidence(changed), "unknown")
        evidence["last_attempt_utc"] = "2026-01-02T12:00:00Z"
        self.assertEqual(initial_confidence(evidence), "unknown")

    def test_bad_timestamps_contradictions_and_string_booleans_rejected(self):
        for evidence in [
            {"unlock_history": "never_attempted", "documented_at_utc": "2026-01-01T00:00:00"},
            {"unlock_history": "never_attempted", "documented_at_utc": "2999-01-01T00:00:00Z"},
            {"unlock_history": "never_attempted", "last_attempt_utc": "2026-01-01T00:00:00Z"},
            {"residual_state_suspected": "false"},
            {"power_cycle": {"ac_disconnected": "true"}},
        ]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                initial_confidence(evidence)

    def test_metadata_cannot_attach_to_other_or_unknown_gpu(self):
        with self.assertRaises(IdentityMismatch):
            state_evidence("before", {"pci_bus_id": "0000:01:00.0"}, "0000:02:00.0")
        with patch("benchmarks.evidence.kernel_evidence", return_value={}):
            result = state_evidence("before", {"pci_bus_id": "0000:01:00.0"}, None)
        self.assertFalse(result["operator_gpu_binding_verified"])
        self.assertEqual(result["initial_state_confidence"], "unknown")

    def load(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps(data))
            return load_experiment(path)

    def test_cooling_history_and_private_fields(self):
        data = {"gpu_label": "gpu-b", "pci_bus_id": "00000000:02:00.0", "cooling_state_id": "repasted",
                "cooling_history": [{"date": "2026-10-26", "change": "thermal_paste_replaced", "state_id": "repasted",
                                     "source": "operator_report", "observations_before": {"max_temperature_c": 81, "sw_thermal_slowdown_observed": True},
                                     "observations": {"power_w": 167, "temperature_c": 71, "fan_percent": 53,
                                                      "thermal_counter_increase_observed": False}}]}
        self.assertEqual(self.load(data)["pci_bus_id"], "0000:02:00.0")
        for private in ("uuid", "serial", "hostname", "local_path"):
            changed = copy.deepcopy(data)
            changed[private] = "private-value"
            with self.assertRaises(ValueError):
                self.load(changed)
        for invalid in (float("nan"), -1, 101, "53", True):
            changed = copy.deepcopy(data)
            changed["cooling_history"][0]["observations"]["fan_percent"] = invalid
            with self.assertRaises(ValueError):
                self.load(changed)
        data["cooling_state_id"] = "old"
        with self.assertRaises(ValueError):
            self.load(data)

    def test_kernel_logs_are_sanitized_and_failures_retained(self):
        raw = "gpu_unlock GPU-private 0000:01:00.0 failed /home/private\ngpu-unlock success serial-private\nother secret"
        with patch("benchmarks.evidence.Path.read_text", return_value="root=private gpu_unlock.targets=GPU-private"), \
             patch("benchmarks.evidence.Path.exists", return_value=True), \
             patch("benchmarks.evidence.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=raw)):
            result = kernel_evidence()
        self.assertTrue(result["cmdline_gpu_unlock_targets_present"])
        self.assertEqual(result["kernel_log"]["failure_marker_count"], 1)
        self.assertEqual(result["kernel_log"]["success_marker_count"], 1)
        self.assertNotIn("private", json.dumps(result))

    def test_unavailable_logs_and_cmdline(self):
        for failure in (PermissionError("private path"), subprocess.TimeoutExpired("dmesg", 3)):
            with patch("benchmarks.evidence.Path.read_text", side_effect=PermissionError), \
                 patch("benchmarks.evidence.subprocess.run", side_effect=failure):
                result = kernel_evidence()
            self.assertIsNone(result["cmdline_gpu_unlock_targets_present"])
            self.assertEqual(result["kernel_log"]["status"], "unavailable")
        with patch("benchmarks.evidence.subprocess.run", return_value=SimpleNamespace(returncode=1)):
            self.assertEqual(kernel_evidence()["kernel_log"]["status"], "unavailable")

    def test_context_device_pci_not_cuda_ordinal(self):
        driver = Mock()
        def get_device(pointer):
            pointer._obj.value = 7
            return 0
        def get_bus(buffer, size, device):
            self.assertEqual(device, 7)
            buffer.value = b"00000000:05:00.0"
            return 0
        driver.cuCtxGetDevice.side_effect = get_device
        driver.cuDeviceGetPCIBusId.side_effect = get_bus
        torch = Mock()
        with patch("benchmarks.metadata.ctypes.CDLL", return_value=driver):
            self.assertEqual(cuda_pci_bus_id(torch, 0), "0000:05:00.0")
        torch.cuda.current_stream.assert_called_once_with(0)
        driver.cuCtxGetDevice.side_effect = None
        driver.cuCtxGetDevice.return_value = 1
        with patch("benchmarks.metadata.ctypes.CDLL", return_value=driver):
            self.assertIsNone(cuda_pci_bus_id(torch, 0))

    def test_environment_does_not_publish_paths_or_visibility_tokens(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-private", "CUDA_DEVICE_ORDER": "private"}), \
             patch("benchmarks.metadata.git_metadata", return_value={"commit": "abc", "dirty": True}):
            self.assertNotIn("private", json.dumps(environment_metadata()))
        with patch("benchmarks.metadata.subprocess.run", side_effect=OSError("/private")):
            self.assertEqual(git_metadata(), {"commit": None, "dirty": None})
