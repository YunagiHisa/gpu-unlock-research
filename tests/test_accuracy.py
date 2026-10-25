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
    def run_fake(self, errors):
        torch = MagicMock()
        torch.__version__ = "test"
        torch.version.cuda = None
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(name="CPU test double")
        torch.cuda.Event.return_value.elapsed_time.return_value = 1.0
        details = {"method": "full_fp64", "checked_output_shape": [2, 2],
                   "sampling_seed": None, "note": None}
        args = gemm.parse_args(["--mode", "fp16", "--m", "2", "--n", "2", "--k", "2", "--samples", "2"])
        with patch.dict("sys.modules", {"torch": torch}), \
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
                code = gemm.main(["--mode", "fp16", "--output", str(target)])
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
                result = self.run_fake([0.001, error])
                self.assertEqual(result["status"], "accuracy_failed")
                self.assertTrue(result["accuracy_checks"]["pre"]["passed"])
                self.assertFalse(result["accuracy_checks"]["post"]["passed"])
                self.assertFalse(result["accuracy"]["passed"])
                self.assertEqual(result["timings_ms"], [1.0, 1.0])
                self.assertIn("summary", result)
                self.save_result(result, 2)

    def test_both_checks_pass(self):
        result = self.run_fake([0.001, 0.002])
        self.assertEqual(result["status"], "draft_validated")
        self.assertTrue(result["accuracy"]["passed"])
        self.assertEqual(result["accuracy"]["normalized_l2_error"], 0.002)
        self.save_result(result, 0)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "existing.json"
            target.write_text("original")
            with patch.object(gemm, "run", return_value={"status": "accuracy_failed"}):
                with self.assertRaises(FileExistsError):
                    gemm.main(["--mode", "fp16", "--output", str(target)])
            self.assertEqual(target.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
