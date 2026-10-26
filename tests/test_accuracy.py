"""CPU-only orchestration and diagnostic-output regression tests."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmarks import gemm


class TestAccuracy(unittest.TestCase):
    def run_fake(self, errors, mode="fp16"):
        torch = MagicMock()
        torch.__version__ = "test"
        torch.version.cuda = None
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(name="CPU test double")
        torch.cuda.Event.return_value.elapsed_time.return_value = 1.0
        details = {"method": "full_fp64", "checked_output_shape": [2, 2],
                   "sampling_seed": None, "note": None}
        args = gemm.parse_args(["--gpu", "0", "--requested-state", "before", "--execution-kind", "diagnostic", "--mode", mode, "--m", "2", "--n", "2", "--k", "2", "--samples", "2"])
        with patch.dict("sys.modules", {"torch": torch}), \
             patch("benchmarks.identity._read_key", return_value=(None, "key_missing")), \
             patch.object(gemm, "cuda_pci_bus_id", return_value=None), \
             patch("benchmarks.evidence.kernel_evidence", return_value={}), \
             patch.object(gemm, "environment_metadata", return_value={}), \
             patch.object(gemm, "configure_mode", return_value={}), \
             patch.object(gemm, "check_accuracy", side_effect=[(e, details) for e in errors]) as check, \
             patch.object(gemm.time, "monotonic", side_effect=[0, 4]):
            result = gemm.run(args)
        self.assertEqual(check.call_count, len(errors))
        return result

    def save_result(self, result, expected_code):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "failure.json"
            with patch.object(gemm, "run", return_value=result), contextlib.redirect_stdout(io.StringIO()):
                code = gemm.main(["--gpu", "0", "--requested-state", "before", "--execution-kind", "diagnostic", "--mode", "fp16", "--output", str(target)])
            self.assertEqual(code, expected_code)
            saved = json.loads(target.read_text())
            self.assertEqual(saved, result)
            return saved

    def test_pre_failure_saved_without_timings(self):
        for error in [0.5, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(error=error):
                result = self.run_fake([error])
                self.assertEqual(result["status"], "accuracy_failed")
                self.assertFalse(result["accuracy"]["passed"])
                self.assertIsNone(result["accuracy_checks"]["post"])
                self.assertNotIn("timings_ms", result)
                self.save_result(result, 2)

    def test_post_failure_preserves_diagnostic_timings(self):
        for error in [0.5, float("nan"), float("inf"), -float("inf")]:
            with self.subTest(error=error):
                result = self.run_fake([0.0001, error])
                self.assertEqual(result["status"], "accuracy_failed")
                self.assertTrue(result["accuracy_checks"]["pre"]["passed"])
                self.assertFalse(result["accuracy_checks"]["post"]["passed"])
                self.assertFalse(result["accuracy"]["passed"])
                self.assertEqual(result["timings_ms"], [1.0, 1.0])
                self.assertIn("summary", result)
                self.save_result(result, 2)

    def test_both_checks_pass(self):
        result = self.run_fake([0.0001, 0.0002])
        self.assertEqual(result["status"], "draft_validated")
        self.assertTrue(result["accuracy"]["passed"])
        self.assertEqual(result["accuracy"]["normalized_l2_error"], 0.0002)
        self.assertFalse(result["identity_verified"])
        self.assertIsNotNone(result["identity_verification_reason"])
        self.assertIn("physical_identity_unverified", result["baseline_ineligibility_reasons"])
        self.assertIn("stable_gpu_pseudonym_unavailable", result["baseline_ineligibility_reasons"])
        self.save_result(result, 0)

    def test_tf32_suspect_preserves_numerical_pass_and_timings(self):
        for errors in [[1e-6, 3e-4], [3e-4, 1e-6], [1e-6, 1e-6]]:
            with self.subTest(errors=errors):
                result = self.run_fake(errors, mode="tf32")
                self.assertEqual(result["status"], "precision_mode_suspect")
                self.assertTrue(result["accuracy"]["passed"])
                self.assertTrue(result["precision_mode_suspect"])
                self.assertEqual(len(result["timings_ms"]), 2)
                self.save_result(result, 3)

    def test_tf32_numerical_failure_takes_priority(self):
        result = self.run_fake([1e-6, 0.5], mode="tf32")
        self.assertEqual(result["status"], "accuracy_failed")
        self.assertTrue(result["precision_mode_suspect"])
        self.save_result(result, 2)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing.json"
            target.write_text("original")
            with patch.object(gemm, "run", return_value={"status": "accuracy_failed"}):
                with self.assertRaises(FileExistsError):
                    gemm.main(["--gpu", "0", "--requested-state", "before", "--execution-kind", "diagnostic", "--mode", "fp16", "--output", str(target)])
            self.assertEqual(target.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
