"""CPU-only regression tests for admission preparation and identity evidence."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from benchmarks import gemm, compare_telemetry
from benchmarks.identity import IdentityMismatch, compare_device_uuids
from benchmarks.metadata import cuda_uuid_bytes, environment_metadata
from benchmarks.preflight import cooldown, contention_reasons
from benchmarks.telemetry import NvmlSession, Sampler

BASE = ["--gpu", "0", "--requested-state", "before", "--mode", "fp16"]
UUID_A = "GPU-11111111-1111-1111-1111-111111111111"
UUID_B = "GPU-22222222-2222-2222-2222-222222222222"


class TestReviewRegressions(unittest.TestCase):
    def test_missing_gpu_or_state_is_error_in_both_tools(self):
        for parser in (gemm.parse_args, compare_telemetry.parse_args):
            tail = ["--output-dir", "synthetic"] if parser == compare_telemetry.parse_args else []
            for args in (["--mode", "fp16", "--gpu", "0"], ["--mode", "fp16", "--requested-state", "before"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser(args + tail)

    def test_temperature_configuration_has_no_defaults_and_requires_pair(self):
        args = gemm.parse_args(BASE)
        self.assertIsNone(args.max_start_temp)
        self.assertIsNone(args.cooldown_timeout_s)
        for argument in (["--max-start-temp", "50"], ["--cooldown-timeout-s", "0"],
                         ["--max-start-temp", "nan", "--cooldown-timeout-s", "1"],
                         ["--max-start-temp", "50", "--cooldown-timeout-s", "-1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                gemm.parse_args(BASE + argument)

    def test_cooldown_wait_pass_timeout_and_unknown(self):
        session = Mock()
        session.query.side_effect = [{"status": "available", "value": value} for value in (60, 49)]
        sleeper = Mock()
        result = cooldown(session, 50, 1, clock=Mock(side_effect=[0, 0, 0.1]), sleep=sleeper)
        self.assertTrue(result["passed"])
        self.assertEqual(result["temperature_c"], 49)
        self.assertEqual(result["wait_seconds"], 0.1)
        sleeper.assert_called_once()
        session.query.side_effect = None
        session.query.return_value = {"status": "available", "value": 60}
        result = cooldown(session, 50, 0, clock=lambda: 0)
        self.assertEqual(result["status"], "timeout")
        session.query.return_value = {"status": "unsupported", "value": None}
        self.assertEqual(cooldown(session, 50, 1)["reason"], "start_temperature_unavailable")
        session.query.return_value = {"status": "available", "value": 40}
        self.assertEqual(cooldown(session, None, None)["reason"], "temperature_condition_not_configured")

    def test_uuid_match_mismatch_and_each_missing_side(self):
        self.assertTrue(compare_device_uuids(lambda: UUID_A, lambda: UUID_A)["matched"])
        with self.assertRaises(IdentityMismatch) as caught:
            compare_device_uuids(lambda: UUID_A, lambda: UUID_B)
        self.assertNotIn(UUID_A, str(caught.exception))
        self.assertNotIn(UUID_B, str(caught.exception))
        for a, b in ((None, UUID_A), (UUID_A, None)):
            result = compare_device_uuids(lambda: a, lambda: b)
            self.assertFalse(result["matched"])
            self.assertEqual(result["status"], "unavailable")
            self.assertNotIn("11111111", json.dumps(result))

    def test_uuid_uses_current_context_device_under_visible_remapping(self):
        driver = Mock()
        def context(pointer):
            pointer._obj.value = 7
            return 0
        def uuid(pointer, device):
            self.assertEqual(device, 7)
            for index in range(16):
                pointer._obj[index] = 0x11
            return 0
        driver.cuCtxGetDevice.side_effect = context
        driver.cuDeviceGetUuid_v2.side_effect = uuid
        torch = Mock()
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "7"}), patch("benchmarks.metadata.ctypes.CDLL", return_value=driver):
            self.assertEqual(cuda_uuid_bytes(torch, 0), b"\x11" * 16)
        torch.cuda.current_stream.assert_called_once_with(0)
        driver.cuDeviceGetUuid_v2.side_effect = None
        driver.cuDeviceGetUuid_v2.return_value = 1
        with patch("benchmarks.metadata.ctypes.CDLL", return_value=driver):
            self.assertIsNone(cuda_uuid_bytes(torch, 0))

    def test_session_keeps_pci_and_uuid_evidence_independent(self):
        nvml = Mock()
        nvml.nvmlDeviceGetPciInfo.return_value = SimpleNamespace(busId="0000:01:00.0")
        nvml.nvmlDeviceGetUUID.return_value = UUID_A
        session = NvmlSession("0000:01:00.0", nvml)
        self.assertTrue(session.pci_verified)
        self.assertFalse(session.identity_verified)
        session.verify_cuda_uuid(lambda: UUID_A)
        self.assertTrue(session.identity_verified)
        session.verify_cuda_uuid(lambda: None)
        self.assertFalse(session.identity_verified)
        with self.assertRaises(IdentityMismatch):
            session.verify_cuda_uuid(lambda: UUID_B)
        session.close()
        nvml.nvmlShutdown.assert_called_once()

    def run_preflight(self, execution_kind="formal", other_count=0, uuid_match=True, temperature_status="passed", after_count=None):
        torch = MagicMock()
        torch.__version__ = "test"
        torch.version.cuda = "test"
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(name="synthetic")
        session = Mock()
        session.identity_verified = uuid_match
        session.reason = None if uuid_match else "uuid_evidence_incomplete"
        session.status = "available"
        session.uuid_verification = {"status": "available" if uuid_match else "unavailable", "matched": uuid_match}
        session.metadata.return_value = {}
        session.pseudonym.return_value = {"status": "unavailable", "value": None}
        def counts(value):
            return {"status": "available", "value": value + 1, "other_processes": {"status": "available", "value": value}} if value is not None else {"status": "unavailable", "value": None}
        session.compute_process_count.side_effect = [counts(other_count), counts(other_count), counts(after_count if after_count is not None else other_count)]
        temperature = {"status": temperature_status, "passed": temperature_status == "passed", "reason": None if temperature_status == "passed" else "cooldown_timeout" if temperature_status == "timeout" else "start_temperature_unavailable"}
        args = gemm.parse_args(BASE + ["--execution-kind", execution_kind])
        with patch.dict("sys.modules", {"torch": torch}), patch.object(gemm, "NvmlSession", return_value=session), \
             patch.object(gemm, "cuda_pci_bus_id", return_value="0000:01:00.0"), \
             patch.object(gemm, "configure_mode", return_value={}), patch.object(gemm, "environment_metadata", return_value={}), \
             patch("benchmarks.evidence.kernel_evidence", return_value={}), patch.object(gemm, "cooldown", return_value=temperature):
            if execution_kind == "diagnostic" and temperature_status != "timeout":
                # Exercise actual allocation/pre-check but stop at numerical failure.
                with patch.object(gemm, "accuracy_result", return_value={"passed": False, "precision_mode_suspect": False}):
                    result = gemm.run(args)
            else:
                result = gemm.run(args)
        return result, torch, session

    def test_formal_gates_block_before_gemm_and_close_session(self):
        for conditions in ({"other_count": 1}, {"other_count": None}, {"uuid_match": False},
                           {"temperature_status": "timeout"}, {"temperature_status": "unavailable"}):
            result, torch, session = self.run_preflight(**conditions)
            self.assertEqual(result["status"], "preflight_blocked")
            torch.randn.assert_not_called()
            torch.mm.assert_not_called()
            session.close.assert_called_once()
            self.assertFalse(result["baseline_eligible"])

    def test_diagnostic_is_distinct_and_before_after_contention_are_reasons(self):
        for count, after in ((1, 1), (0, 1)):
            result, torch, _ = self.run_preflight(execution_kind="diagnostic", other_count=count, after_count=after)
            torch.mm.assert_called_once()
            self.assertEqual(result["execution_kind"], "diagnostic")
            self.assertIn("other_compute_processes_present", result["baseline_ineligibility_reasons"])
        self.assertEqual(contention_reasons({}), ["other_compute_processes_unknown"])

    def test_phase_boundaries_exclude_reads_crossing_either_edge(self):
        session = NvmlSession(None)
        sampler = Sampler(session, 0.1)
        sample = {name: {"value": value, "status": "available"} for name, value in
                  (("temperature_c", 40), ("power_w", 12), ("sm_clock_mhz", 1000), ("fan_percent", 30), ("clock_event_reasons", 0x24))}
        sampler.samples = [{**sample, "source": source, "elapsed_seconds": a, "completed_seconds": b}
                           for source, a, b in (("endpoint_before", 0, 0.1), ("poll", 0.2, 0.3),
                                                ("poll", 0.9, 1.1), ("poll", 1, 1.1), ("poll", 1.9, 2.1),
                                                ("endpoint_after", 2.2, 2.3))]
        sampler.warmup_start, sampler.warmup_end = 0.1, 1
        sampler.timed_start, sampler.timed_end = 1, 2
        sampler.before = sampler.after = session.counters()
        sampler.duration = 2.3
        phases = sampler.result()["phases"]
        for phase in ("idle_before", "warmup", "timed", "post"):
            self.assertEqual(phases[phase]["sample_count"], 1)
            self.assertEqual(phases[phase]["summary"]["temperature_c"]["median"], 40)
            self.assertEqual(phases[phase]["clock_events"]["observed_samples_by_reason"]["sw_power_cap"], 1)
        self.assertIsNone(phases["timed"]["summary"]["memory_clock_mhz"]["median"])

    def test_metadata_environment_values_do_not_escape_allowlist(self):
        with patch.dict("os.environ", {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUBLAS_WORKSPACE_CONFIG": "/private/secret", "TORCH_BLAS_PREFER_CUBLASLT": "private-token"}), \
             patch("benchmarks.metadata.git_metadata", return_value={}):
            metadata = environment_metadata()
        self.assertNotIn("/private/secret", json.dumps(metadata))
        self.assertNotIn("private-token", json.dumps(metadata))
        self.assertEqual(metadata["allowed_environment"]["CUBLAS_WORKSPACE_CONFIG"]["status"], "unavailable")
        self.assertEqual(metadata["allowed_environment"]["CUBLAS_WORKSPACE_CONFIG"]["reason"], "allowlist_not_defined_in_issue_1")

    def test_sampler_endpoint_failure_is_sanitized_and_worker_stops(self):
        session = NvmlSession(None)
        session.sample = Mock(side_effect=RuntimeError("private-token"))
        with self.assertRaisesRegex(RuntimeError, "benchmark"):
            with Sampler(session, 0.1) as sampler:
                raise RuntimeError("benchmark")
        self.assertFalse(sampler.thread.is_alive())
        self.assertNotIn("private-token", json.dumps(sampler.result()))

    def test_uuid_contradiction_in_runner_prevents_allocation_and_closes(self):
        torch = MagicMock()
        torch.cuda.device_count.return_value = 1
        session = Mock()
        session.verify_cuda_uuid.side_effect = IdentityMismatch()
        with patch.dict("sys.modules", {"torch": torch}), patch.object(gemm, "configure_mode", return_value={}), \
             patch.object(gemm, "cuda_pci_bus_id", return_value="0000:01:00.0"), \
             patch.object(gemm, "state_evidence", return_value={}), patch.object(gemm, "NvmlSession", return_value=session):
            with self.assertRaises(IdentityMismatch):
                gemm.run(gemm.parse_args(BASE))
        torch.randn.assert_not_called()
        torch.mm.assert_not_called()
        session.close.assert_called_once()
