"""Read-only environment metadata with no hostnames, UUIDs, or local paths."""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path

from benchmarks.evidence import pci_id


def cuda_pci_bus_id(torch, gpu):
    """Read the actual current CUDA context, including visible-device remapping.

    Initializing PyTorch's current stream ensures a context exists. Never use a
    CUDA ordinal as an NVML ordinal or parse CUDA_VISIBLE_DEVICES as an index.
    """
    try:
        torch.cuda.current_stream(gpu)
        driver = ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuCtxGetDevice.restype = ctypes.c_int
        if driver.cuCtxGetDevice(ctypes.byref(device)) != 0:
            return None
        bus = ctypes.create_string_buffer(32)
        driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
        if driver.cuDeviceGetPCIBusId(bus, len(bus), device.value) != 0:
            return None
        return pci_id(bus.value)
    except (OSError, AttributeError, ValueError, RuntimeError):
        return None


def git_metadata():
    root = Path(__file__).resolve().parent.parent
    def query(*args):
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=3)
        if proc.returncode:
            raise ValueError("git unavailable")
        return proc.stdout.strip()
    try:
        return {"commit": query("rev-parse", "HEAD"), "dirty": bool(query("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def environment_metadata():
    return {"guest_kernel": platform.release(), "benchmark_git": git_metadata(),
            "cuda_device_order": "PCI_BUS_ID" if os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" else "not_set_to_PCI_BUS_ID",
            "cublas_version": {"value": None, "status": "unavailable", "reason": "no_reliable_public_pytorch_query"}}
