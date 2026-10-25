import unittest
import contextlib
import io
from types import SimpleNamespace

from benchmarks.core import operation_count, percentile, tflops, timing_summary
from benchmarks.gemm import parse_args, configure_mode


class TestCore(unittest.TestCase):
    def test_flops(self):
        self.assertEqual(operation_count(2, 3, 4), 48)
        self.assertAlmostEqual(tflops(1000, 1000, 1000, 1), 2)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            operation_count(0, 2, 3)
        with self.assertRaises(ValueError):
            tflops(2, 3, 4, 0)

    def test_stats(self):
        result = timing_summary([1, 2, 3, 4], 2, 3, 4)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertAlmostEqual(result["iqr_ms"], 1.5)
        self.assertEqual(percentile([7], 50), 7)

    def test_cli(self):
        args = parse_args(["--mode", "bf16", "--m", "32", "--n", "64", "--k", "128"])
        self.assertEqual((args.m, args.n, args.k), (32, 64, 128))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--mode", "bf16", "--m", "0"])

    def test_precision_settings(self):
        class Matmul:
            fp32_precision = "ieee"
            allow_fp16_reduced_precision_reduction = True
            allow_bf16_reduced_precision_reduction = True
        fake = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=Matmul())))
        self.assertEqual(configure_mode(fake, "tf32")["fp32_precision"], "tf32")
        self.assertEqual(configure_mode(fake, "fp32")["fp32_precision"], "ieee")
        self.assertFalse(fake.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        self.assertFalse(fake.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


if __name__ == "__main__":
    unittest.main()
