"""CUDA-independent benchmark calculations and parameter validation."""
from __future__ import annotations

import math
import statistics
from typing import Sequence

MODES = ("fp16", "bf16", "tf32", "fp32")


def validate_dimensions(m: int, n: int, k: int) -> None:
    if any(type(v) is not int or v <= 0 for v in (m, n, k)):
        raise ValueError("matrix dimensions must be positive integers")


def operation_count(m: int, n: int, k: int) -> int:
    validate_dimensions(m, n, k)
    return 2 * m * n * k


def tflops(m: int, n: int, k: int, latency_ms: float) -> float:
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("latency_ms must be positive and finite")
    return operation_count(m, n, k) / (latency_ms * 1e9)


def percentile(values: Sequence[float], percent: float) -> float:
    if not values or not 0 <= percent <= 100:
        raise ValueError("nonempty values and percentile in [0, 100] required")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("all samples must be finite")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(latencies_ms: Sequence[float], m: int, n: int, k: int) -> dict:
    if not latencies_ms or any(not math.isfinite(v) or v <= 0 for v in latencies_ms):
        raise ValueError("timings must be nonempty, finite, and positive")
    median = statistics.median(latencies_ms)
    return {
        "samples": len(latencies_ms),
        "median_ms": median,
        "iqr_ms": percentile(latencies_ms, 75) - percentile(latencies_ms, 25),
        "tflops_from_median_latency": tflops(m, n, k, median),
    }


def normalized_error(actual, reference) -> float:
    """NumPy-like arrays supported; imported by benchmark only with torch tensors."""
    numerator = (actual - reference).norm().item()
    denominator = reference.norm().item()
    return numerator / max(denominator, 1e-30)
