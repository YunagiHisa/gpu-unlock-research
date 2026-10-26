"""Optional NVML reader. No setters, index mapping, or NVIDIA CLI commands."""
from __future__ import annotations

import importlib
import math
import os
import statistics
import threading
import time

from benchmarks.evidence import pci_id
from benchmarks.identity import IdentityMismatch, compare_device_uuids, gpu_pseudonym


def unavailable(reason):
    return {"value": None, "status": reason}


def read_value(call):
    try:
        value = call()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, float) and not math.isfinite(value):
            return unavailable("invalid_value")
        return {"value": value, "status": "available"}
    except Exception as exc:
        # Error text can contain device identifiers or local paths.
        name = type(exc).__name__
        status = {"NVMLError_NotSupported": "unsupported", "NVMLError_NoPermission": "permission_denied",
                  "AttributeError": "api_unavailable", "NVMLError_FunctionNotFound": "api_unavailable"}.get(name, "unavailable")
        return unavailable(status)


# NVML's stable public bit definitions; retain raw masks for unknown future bits.
REASONS = {"gpu_idle": 0x1, "applications_clocks": 0x2, "sw_power_cap": 0x4,
           "hw_slowdown": 0x8, "sync_boost": 0x10, "sw_thermal_slowdown": 0x20,
           "hw_thermal_slowdown": 0x40, "hw_power_brake_slowdown": 0x80,
           "display_clock_setting": 0x100}


class NvmlSession:
    def __init__(self, measured_pci, nvml=None):
        self.nvml = nvml
        self.handle = None
        self.initialized = False
        self.identity_verified = False
        self.pci_verified = False
        self.uuid_verification = {"status": "unavailable", "matched": False, "reason": "not_checked"}
        self.status = "unavailable"
        self.reason = "cuda_pci_unavailable"
        if measured_pci is None:
            return
        try:
            self.nvml = nvml if nvml is not None else importlib.import_module("pynvml")
            self.nvml.nvmlInit()
            self.initialized = True
            self.handle = self.nvml.nvmlDeviceGetHandleByPciBusId(pci_id(measured_pci))
            actual = pci_id(self.nvml.nvmlDeviceGetPciInfo(self.handle).busId)
            if actual != pci_id(measured_pci):
                raise IdentityMismatch()
            self.status, self.reason = "available", None
            self.pci_verified = True
            self.reason = "uuid_not_checked"
        except IdentityMismatch:
            self.close()
            raise
        except Exception:
            self.handle = None
            self.reason = "nvml_dependency_init_or_pci_mapping_failed"

    def verify_cuda_uuid(self, cuda_reader):
        def nvml_reader():
            if self.handle is None:
                raise ValueError
            return self.nvml.nvmlDeviceGetUUID(self.handle)
        self.uuid_verification = compare_device_uuids(cuda_reader, nvml_reader)
        self.identity_verified = self.pci_verified and self.uuid_verification["matched"]
        if self.identity_verified:
            self.reason = None
        else:
            self.reason = "uuid_evidence_incomplete" if self.pci_verified else self.reason
        return self.uuid_verification

    def pseudonym(self, key_path=None):
        def uuid_reader():
            if not self.identity_verified:
                raise ValueError
            return self.nvml.nvmlDeviceGetUUID(self.handle)
        return gpu_pseudonym(uuid_reader, key_path)

    def compute_process_count(self):
        if self.handle is None:
            return {"status": "unavailable", "value": None, "reason": self.reason,
                    "other_processes": {"status": "unavailable", "value": None, "reason": self.reason}}
        # Reduce immediately; PID/name/path objects must never reach results.
        def counts():
            processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            # PID is compared transiently and is never returned or serialized.
            own_pid = os.getpid()
            others = read_value(lambda: sum(process.pid != own_pid for process in processes))
            return {"total": len(processes), "others": others}
        field = read_value(counts)
        if field["status"] != "available":
            return {"status": "unavailable", "value": None, "reason": field["status"],
                    "other_processes": {"status": "unavailable", "value": None, "reason": field["status"]}}
        others = field["value"]["others"]
        if others["status"] != "available":
            others = {"status": "unavailable", "value": None, "reason": others["status"]}
        return {"status": "available", "value": field["value"]["total"], "other_processes": others}

    def close(self):
        if self.initialized:
            self.initialized = False
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                self.status, self.reason = "unavailable", "nvml_shutdown_failed"

    def query(self, function, *args, scale=1):
        if not function.startswith("nvmlDeviceGet"):
            return unavailable("non_read_only_api_rejected")
        if self.handle is None:
            return unavailable(self.reason)
        def get():
            value = getattr(self.nvml, function)(self.handle, *args)
            return value if scale == 1 else value / scale
        return read_value(get)

    def reasons(self, supported=False):
        stem = "Supported" if supported else "Current"
        current = self.query(f"nvmlDeviceGet{stem}ClocksEventReasons")
        if current["status"] == "api_unavailable":
            current = self.query(f"nvmlDeviceGet{stem}ClocksThrottleReasons")
        return current

    def metadata(self):
        fields = {"name": ("nvmlDeviceGetName",), "vbios_version": ("nvmlDeviceGetVbiosVersion",),
                  "power_limit_w": ("nvmlDeviceGetPowerManagementLimit",),
                  "power_default_limit_w": ("nvmlDeviceGetPowerManagementDefaultLimit",),
                  "persistence_mode": ("nvmlDeviceGetPersistenceMode",),
                  "pcie_generation": ("nvmlDeviceGetCurrPcieLinkGeneration",),
                  "pcie_width": ("nvmlDeviceGetCurrPcieLinkWidth",),
                  "pcie_max_generation": ("nvmlDeviceGetMaxPcieLinkGeneration",),
                  "pcie_max_width": ("nvmlDeviceGetMaxPcieLinkWidth",),
                  "max_memory_clock_mhz": ("nvmlDeviceGetMaxClockInfo", 2),
                  "max_graphics_clock_mhz": ("nvmlDeviceGetMaxClockInfo", 0),
                  "max_sm_clock_mhz": ("nvmlDeviceGetMaxClockInfo", 1)}
        result = {name: self.query(*args, scale=1000 if name.startswith("power_") else 1)
                  for name, args in fields.items()}
        result["supported_clock_event_reasons"] = self.reasons(supported=True)
        result["driver_version"] = (read_value(lambda: self.nvml.nvmlSystemGetDriverVersion())
                                    if self.initialized else unavailable(self.reason))
        result["nvml_version"] = (read_value(lambda: self.nvml.nvmlSystemGetNVMLVersion())
                                  if self.initialized else unavailable(self.reason))
        return result

    def sample(self):
        return {"sm_clock_mhz": self.query("nvmlDeviceGetClockInfo", 1),
                "power_w": self.query("nvmlDeviceGetPowerUsage", scale=1000),
                "temperature_c": self.query("nvmlDeviceGetTemperature", 0),
                "fan_percent": self.query("nvmlDeviceGetFanSpeed"),
                "memory_clock_mhz": self.query("nvmlDeviceGetClockInfo", 2),
                "gpu_utilization_percent": self.utilization(),
                "performance_state": self.query("nvmlDeviceGetPerformanceState"),
                "clock_event_reasons": self.reasons()}

    def utilization(self):
        if self.handle is None:
            return unavailable(self.reason)
        return read_value(lambda: int(self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu))

    def counters(self):
        result = {}
        for name, policy in (("power", 0), ("thermal", 1)):
            field = self.query("nvmlDeviceGetViolationStatus", policy)
            if field["status"] == "available":
                value = field["value"]
                field["value"] = {"violation_time_ns": int(value.violationTime), "reference_time_us": int(value.referenceTime)}
            result[name] = field
        return result


def counter_deltas(before, after):
    result = {}
    for name in ("power", "thermal"):
        a, b = before[name], after[name]
        if a["value"] is None or b["value"] is None:
            result[name] = unavailable("endpoint_unavailable")
        elif (b["value"]["reference_time_us"] < a["value"]["reference_time_us"]
              or b["value"]["violation_time_ns"] < a["value"]["violation_time_ns"]):
            result[name] = unavailable("counter_reset")
        else:
            result[name] = {"status": "available", "value": b["value"]["violation_time_ns"] - a["value"]["violation_time_ns"]}
    return result


def sample_summary(samples):
    summaries = {}
    for name in ("sm_clock_mhz", "power_w", "temperature_c", "fan_percent", "memory_clock_mhz", "gpu_utilization_percent", "performance_state"):
        values = [s[name]["value"] for s in samples if s.get(name, {}).get("status") == "available"]
        summaries[name] = {"status": "available" if values else "unavailable",
                           "reason": None if values else "no_available_field_samples",
                           "available_samples": len(values), "unavailable_samples": len(samples) - len(values),
                           "min": min(values) if values else None, "max": max(values) if values else None,
                           "median": statistics.median(values) if values else None,
                           "mean": statistics.mean(values) if values else None}
    return summaries


class Sampler:
    """Samples warm-up + queued measurement, excluding FP64 endpoint checks.

    The first sample and counter endpoints are synchronous outside CUDA events.
    Polling cannot guarantee detection of short clock events.
    """
    def __init__(self, session, interval, enabled=True):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("sampling interval must be positive and finite")
        self.session, self.interval = session, interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.before = self.after = None
        self.error = None
        self.enabled = enabled
        self.timed_start = self.timed_end = None
        self.warmup_start = self.warmup_end = None

    def _sample(self, source="poll"):
        start = time.perf_counter() - self.started
        try:
            values = self.session.sample()
        except Exception:
            self.error = "sampling_failed"
            values = {name: unavailable("sampling_failed") for name in
                      ("sm_clock_mhz", "power_w", "temperature_c", "fan_percent", "memory_clock_mhz",
                       "gpu_utilization_percent", "performance_state", "clock_event_reasons")}
        self.samples.append({"elapsed_seconds": start, "completed_seconds": time.perf_counter() - self.started,
                             "source": source, **values})

    def begin_timed(self):
        self.timed_start = time.perf_counter() - self.started

    def end_timed(self):
        self.timed_end = time.perf_counter() - self.started

    def begin_warmup(self):
        self.warmup_start = time.perf_counter() - self.started

    def end_warmup(self):
        self.warmup_end = time.perf_counter() - self.started

    def _worker(self):
        try:
            while not self.stop_event.wait(self.interval):
                self._sample()
        except Exception:
            self.error = "sampling_failed"

    def __enter__(self):
        self.started = time.perf_counter()
        self.before = self.session.counters()
        self._sample("endpoint_before")
        if self.enabled:
            self.thread = threading.Thread(target=self._worker, name="nvml-telemetry", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample("endpoint_after")
        self.after = self.session.counters()
        self.duration = time.perf_counter() - self.started

    def result(self):
        timed = [s for s in self.samples if self.timed_start is not None and self.timed_end is not None
                 and s["source"] == "poll" and self.timed_start <= s["elapsed_seconds"]
                 and s["completed_seconds"] <= self.timed_end]
        phases = {"idle_before": [s for s in self.samples if s["source"] == "endpoint_before"],
                  "warmup": [s for s in self.samples if s["source"] == "poll" and self.warmup_start is not None and self.warmup_end is not None
                             and self.warmup_start <= s["elapsed_seconds"] and s["completed_seconds"] <= self.warmup_end],
                  "timed": timed, "post": [s for s in self.samples if s["source"] == "endpoint_after"]}
        phase_statistics = {}
        for phase, samples in phases.items():
            phase_masks = [s["clock_event_reasons"]["value"] for s in samples if s["clock_event_reasons"]["status"] == "available"]
            start, end = ((self.warmup_start, self.warmup_end) if phase == "warmup" else (self.timed_start, self.timed_end)
                          if phase == "timed" else (samples[0]["elapsed_seconds"], samples[-1]["completed_seconds"]) if samples else (None, None))
            phase_statistics[phase] = {"start_seconds": start, "end_seconds": end,
                                       "sample_count": len(samples), "status": "available" if samples else "unavailable",
                                       "reason": None if samples else "no_complete_samples_in_phase" if self.enabled else "polling_disabled",
                                       "summary": sample_summary(samples),
                                       "clock_events": {"available_samples": len(phase_masks),
                                                        "status": "available" if phase_masks else "unavailable",
                                                        "observed_samples_by_reason": {name: sum(bool(mask & bit) for mask in phase_masks) if phase_masks else None for name, bit in REASONS.items()}}}
        return {"status": "partial" if self.error else self.session.status, "error": self.error,
                "polling_enabled": self.enabled,
                "phases": phase_statistics,
                "sampling_interval_seconds": self.interval, "duration_seconds": self.duration,
                "sample_count": len(self.samples), "samples": self.samples,
                "violation_counters": {"before": self.before, "after": self.after,
                                       "delta_ns": counter_deltas(self.before, self.after)}}
