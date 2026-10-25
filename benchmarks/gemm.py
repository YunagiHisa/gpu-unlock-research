"""Measure one GPU / precision / matrix configuration per invocation.

PR 1 only: JSON schema, telemetry, and hardware-state evidence are tracked in
later PRs. The output is an *unverified draft*, not a publishable baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.core import MODES, normalized_error, timing_summary, validate_dimensions

# Conservative starting thresholds. Review on the reference hardware before
# treating any measurement as a valid baseline.
ERROR_LIMITS = {"fp16": 0.02, "bf16": 0.08, "tf32": 0.02, "fp32": 0.0005}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index; resolve PCI address in output")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--m", type=positive_int, default=2048)
    parser.add_argument("--n", type=positive_int, default=2048)
    parser.add_argument("--k", type=positive_int, default=2048)
    parser.add_argument("--warmup-seconds", type=positive_float, default=3.0)
    parser.add_argument("--samples", type=positive_int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-tile", type=positive_int, default=128, help="Max rows and cols in FP64 validation subset (large matrices)")
    parser.add_argument("--full-reference-max", type=positive_int, default=2048, help="Maximum dimension for full FP64 reference")
    parser.add_argument("--output", type=Path, help="local JSON path; omit to print to stdout")
    return parser.parse_args(argv)


def configure_mode(torch, mode: str):
    matmul = torch.backends.cuda.matmul
    # For precision experiments, prohibit reduced-precision intermediate
    # reductions where the installed PyTorch supports these flags.
    matmul.allow_fp16_reduced_precision_reduction = False
    matmul.allow_bf16_reduced_precision_reduction = False
    if not hasattr(matmul, "fp32_precision"):
        raise RuntimeError("PyTorch must support matmul.fp32_precision")
    matmul.fp32_precision = "tf32" if mode == "tf32" else "ieee"
    expected = "tf32" if mode == "tf32" else "ieee"
    if matmul.fp32_precision != expected:
        raise RuntimeError("requested FP32 matmul precision did not stick")
    if matmul.allow_fp16_reduced_precision_reduction or matmul.allow_bf16_reduced_precision_reduction:
        raise RuntimeError("reduced precision reduction unexpectedly enabled")
    return {
        "fp32_precision": matmul.fp32_precision,
        "allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
    }


def check_accuracy(torch, a, b, out, args):
    """Compare to exact-input FP64 product, full for small matrices, sampled for large.

    Sampling is deterministic and covers randomly selected output rows/columns.
    For sampled checks the reported error is NOT a whole-matrix error bound.
    """
    full = max(args.m, args.n, args.k) <= args.full_reference_max
    if full:
        observed = out.double()
        reference = a.double() @ b.double()
        checked_shape = [args.m, args.n]
    else:
        # CPU generator ensures reproducible indices without altering GPU RNG state.
        gen = torch.Generator(device="cpu").manual_seed(args.seed + 17)
        nr = min(args.reference_tile, args.m)
        nc = min(args.reference_tile, args.n)
        rows = torch.randperm(args.m, generator=gen)[:nr].to(a.device)
        cols = torch.randperm(args.n, generator=gen)[:nc].to(a.device)
        observed = out.index_select(0, rows).index_select(1, cols).double()
        reference = a.index_select(0, rows).double() @ b.index_select(1, cols).double()
        checked_shape = [nr, nc]
    error = normalized_error(observed, reference)
    del observed, reference
    return error, {"method": "full_fp64" if full else "sampled_fp64", "checked_output_shape": checked_shape,
                   "sampling_seed": args.seed + 17 if not full else None,
                   "note": None if full else "Sampled output check; not a whole-matrix accuracy guarantee"}


def run(args):
    import torch

    validate_dimensions(args.m, args.n, args.k)
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        raise RuntimeError("requested CUDA device unavailable")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    precision = configure_mode(torch, args.mode)
    dtype = torch.float16 if args.mode == "fp16" else torch.bfloat16 if args.mode == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    props = torch.cuda.get_device_properties(args.gpu)
    result = {
        "status": "unverified_draft",
        "warning": "PR1 draft; NOT a publishable baseline (no NVML, state evidence or JSON Schema)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "device": {"cuda_index": args.gpu, "name": props.name, "pci_bus_id": None},
        "software": {"python": sys.version.split()[0], "pytorch": torch.__version__, "pytorch_cuda": torch.version.cuda},
        "precision": precision,
        "seed": args.seed,
        "warmup": {"target_seconds": args.warmup_seconds, "operations": 0},
    }
    a = torch.randn((args.m, args.k), dtype=dtype, device=device)
    b = torch.randn((args.k, args.n), dtype=dtype, device=device)
    out = torch.empty((args.m, args.n), dtype=dtype, device=device)

    with torch.no_grad():
        torch.mm(a, b, out=out)
        torch.cuda.synchronize(device)
        error, reference_details = check_accuracy(torch, a, b, out, args)
        threshold = ERROR_LIMITS[args.mode]
        passed = math.isfinite(error) and error <= threshold
        result["accuracy"] = {"normalized_l2_error": error if math.isfinite(error) else None,
                              "threshold": threshold, "passed": passed, **reference_details}
        if not passed:
            result["status"] = "accuracy_failed"
            result["failure_reason"] = "Numerical validation failed; timing skipped"
            return result

        torch.cuda.synchronize(device)
        start_warmup = time.monotonic()
        warmups = 0
        while time.monotonic() - start_warmup < args.warmup_seconds:
            torch.mm(a, b, out=out)
            warmups += 1
            if warmups % 16 == 0:
                torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(args.samples):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.mm(a, b, out=out)
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
    result["status"] = "draft_validated"
    result["warmup"]["operations"] = warmups
    result["timings_ms"] = samples
    result["summary"] = timing_summary(samples, args.m, args.n, args.k)
    return result


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Unique path by default; never silently overwrite existing results.
        with args.output.open("x", encoding="utf-8") as f:
            f.write(serialized + "\n")
        print(f"Wrote {args.output}")
    else:
        print(serialized)
    return 0 if result["status"] == "draft_validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
