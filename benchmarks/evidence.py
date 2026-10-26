"""Allowlisted operator metadata and conservative, read-only state evidence."""
from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def pci_id(value):
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", value):
        raise ValueError("invalid PCI bus ID")
    domain, bus, slot = value.lower().split(":")
    return f"{int(domain, 16):04x}:{bus}:{slot}"


def keys(value, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("unknown metadata fields; only documented public fields are accepted")


def label(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", value):
        raise ValueError("use a short anonymous lowercase label")
    return value


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("evidence timestamp must have UTC timezone")
    if parsed > datetime.now(timezone.utc):
        raise ValueError("evidence timestamp is in the future")
    return parsed


def load_experiment(path):
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    keys(data, {"gpu_label", "pci_bus_id", "cooling_state_id", "cooling_history", "initial_state", "host_environment"})
    for name in ("gpu_label", "cooling_state_id"):
        label(data[name])
    data["pci_bus_id"] = pci_id(data["pci_bus_id"])
    if not isinstance(data.get("cooling_history", []), list):
        raise ValueError("cooling_history must be a list")
    for event in data.get("cooling_history", []):
        keys(event, {"date", "change", "previous_state_id", "state_id", "source", "observations", "observations_before"})
        datetime.strptime(event["date"], "%Y-%m-%d")
        for name in ("state_id", "previous_state_id"):
            if name in event:
                label(event[name])
        if event["change"] not in ("thermal_paste_replaced", "unchanged_observation", "cooling_maintenance"):
            raise ValueError("unsupported cooling change")
        if event["source"] != "operator_report":
            raise ValueError("manual observations must be operator_report")
        observations = event.get("observations", {})
        _validate_observations(observations)
        _validate_observations(event.get("observations_before", {}))
    history = data.get("cooling_history", [])
    if history and history[-1]["state_id"] != data["cooling_state_id"]:
        raise ValueError("latest cooling state must match cooling_state_id")
    if [e["date"] for e in history] != sorted(e["date"] for e in history):
        raise ValueError("cooling history must be chronological")
    initial_confidence(data.get("initial_state", {}))
    host = data.get("host_environment", {})
    keys(host, {"proxmox_version", "host_kernel_version", "vfio_configuration"})
    for name, value in host.items():
        pattern = r"[0-9][0-9A-Za-z.+~-]{0,63}" if name != "vfio_configuration" else r"(dedicated_passthrough|shared|unknown)"
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError("host metadata must use version strings and documented VFIO categories")
    return data


def _validate_observations(observations):
    keys(observations, {"power_w", "temperature_c", "max_temperature_c", "fan_percent", "sw_thermal_slowdown_observed", "thermal_counter_increase_observed"})
    for name, value in observations.items():
        if name.endswith("observed"):
            if type(value) is not bool:
                raise ValueError("observation flags must be boolean")
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name == "fan_percent" and value > 100):
            raise ValueError("invalid numeric observation")


def initial_confidence(evidence):
    keys(evidence, {"unlock_history", "documented_at_utc", "last_attempt_utc", "power_cycle", "residual_state_suspected"})
    history = evidence.get("unlock_history", "unknown")
    if history not in ("never_attempted", "attempted", "unknown"):
        raise ValueError("invalid unlock history")
    documented = utc(evidence["documented_at_utc"]) if "documented_at_utc" in evidence else None
    attempt = utc(evidence["last_attempt_utc"]) if "last_attempt_utc" in evidence else None
    cycle = evidence.get("power_cycle", {})
    keys(cycle, {"completed_at_utc", "method", "host_shutdown", "ac_disconnected", "fresh_power_on"})
    for name in ("host_shutdown", "ac_disconnected", "fresh_power_on"):
        if name in cycle and type(cycle[name]) is not bool:
            raise ValueError("power-cycle flags must be boolean")
    if "residual_state_suspected" in evidence and type(evidence["residual_state_suspected"]) is not bool:
        raise ValueError("residual-state flag must be boolean")
    completed = utc(cycle["completed_at_utc"]) if "completed_at_utc" in cycle else None
    if cycle.get("method") not in (None, "host_shutdown_ac_disconnect_power_on"):
        raise ValueError("unsupported power-cycle method")
    if history == "never_attempted" and (attempt or cycle):
        raise ValueError("contradictory never-attempted history")
    if evidence.get("residual_state_suspected"):
        return "low"
    if history == "never_attempted" and documented:
        return "high"
    if (history == "attempted" and documented and attempt and completed and attempt < completed <= documented
            and cycle.get("method") == "host_shutdown_ac_disconnect_power_on"
            and all(cycle.get(k) is True for k in ("host_shutdown", "ac_disconnected", "fresh_power_on"))):
        return "high"
    return "unknown"


def kernel_evidence():
    # Never return raw text, command lines, PCI IDs, UUIDs, or exception messages.
    try:
        tokens = Path("/proc/cmdline").read_text().split()
        targets = any(t == "gpu_unlock.targets" or t.startswith("gpu_unlock.targets=") for t in tokens)
    except OSError:
        targets = None
    try:
        proc = subprocess.run(["dmesg", "--color=never"], capture_output=True, text=True, timeout=3)
        if proc.returncode:
            logs = {"status": "unavailable", "reason": "access_denied_or_command_failed"}
        else:
            relevant = [line for line in proc.stdout.splitlines() if re.search(r"gpu[_-]unlock", line, re.I)]
            logs = {"status": "available", "relevant_line_count": len(relevant),
                    "failure_marker_count": sum(bool(re.search(r"fail|error|unsuccess", line, re.I)) for line in relevant),
                    "success_marker_count": sum(bool(re.search(r"success", line, re.I)) and not bool(re.search(r"unsuccess", line, re.I)) for line in relevant),
                    "sanitization": "counts_only_no_raw_log_text"}
    except (OSError, subprocess.TimeoutExpired):
        logs = {"status": "unavailable", "reason": "command_unavailable_or_timeout"}
    return {"cmdline_gpu_unlock_targets_present": targets,
            "gpu_unlock_module_present": Path("/sys/module/gpu_unlock").exists(), "kernel_log": logs}


def state_evidence(requested_state, experiment, measured_pci):
    from benchmarks.identity import IdentityMismatch
    if experiment is not None and measured_pci is not None and experiment["pci_bus_id"] != measured_pci:
        raise IdentityMismatch()
    operator = experiment.get("initial_state", {}) if experiment else {}
    confidence = initial_confidence(operator) if measured_pci is not None else "unknown"
    return {"requested_state": requested_state, "initial_state_confidence": confidence,
            "verification_status": "not_applicable" if requested_state == "before" and confidence == "high" else "unconfirmed",
            "operator_evidence": operator, "system_evidence": kernel_evidence(),
            "operator_gpu_binding_verified": experiment is not None and measured_pci is not None,
            "note": "Operator records are operational evidence, not proof of internal GPU state; separate review required"}
