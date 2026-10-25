"""CUDA-independent tests for precision controls and event orchestration."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import gemm


class TestMeasurement(unittest.TestCase):
    def test_error_limits_and_floor(self):
        self.assertEqual(gemm.ERROR_LIMITS, {"fp16": 1e-3, "bf16": 5e-3, "tf32": 1e-3, "fp32": 2e-5})
        for mode, limit in gemm.ERROR_LIMITS.items():
            for error, expected in [(limit, True), (limit * 1.01, False),
                                    (float("nan"), False), (float("inf"), False)]:
                with self.subTest(mode=mode, error=error), patch.object(gemm, "check_accuracy", return_value=(error, {})):
                    result = gemm.accuracy_result(None, None, None, None, SimpleNamespace(mode=mode))
                    self.assertEqual(result["passed"], expected)
        for error, suspect in [(0, True), (1.99e-5, True), (2e-5, False), (3e-4, False)]:
            with patch.object(gemm, "check_accuracy", return_value=(error, {})):
                result = gemm.accuracy_result(None, None, None, None, SimpleNamespace(mode="tf32"))
                self.assertTrue(result["passed"])
                self.assertEqual(result["precision_mode_suspect"], suspect)

    def test_fp16_accumulation_missing_or_rejected(self):
        missing = SimpleNamespace()

        class Rejected:
            @property
            def allow_fp16_accumulation(self):
                return True

            @allow_fp16_accumulation.setter
            def allow_fp16_accumulation(self, value):
                pass

        for matmul in [missing, Rejected()]:
            with self.subTest(matmul=type(matmul).__name__):
                torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)))
                with self.assertRaisesRegex(RuntimeError, "accumulation"):
                    gemm.configure_mode(torch, "fp16")

    def test_reference_budget_includes_k(self):
        args = gemm.parse_args(["--mode", "fp32", "--m", "64", "--n", "64", "--k", "64",
                                "--full-reference-max-flops", str(2 * 64**3)])
        self.assertTrue(gemm.use_full_reference(args))
        args.k = 65
        self.assertFalse(gemm.use_full_reference(args))
        args.k = 64
        args.full_reference_max = 32
        self.assertFalse(gemm.use_full_reference(args))

    def test_events_queue_then_single_sync_then_elapsed(self):
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

            def elapsed_time(self, stop):
                calls.append(("elapsed", self.index, stop.index))
                return 1.0

        torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event),
                                mm=lambda a, b, out: calls.append(("mm",)))
        self.assertEqual(gemm.measure_queued(torch, None, None, None, 2), [1.0, 1.0])
        self.assertEqual(calls, [("record", 0), ("mm",), ("record", 1),
                                 ("record", 2), ("mm",), ("record", 3),
                                 ("sync", 3), ("elapsed", 0, 1), ("elapsed", 2, 3)])


if __name__ == "__main__":
    unittest.main()
