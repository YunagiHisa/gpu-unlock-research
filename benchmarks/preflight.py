"""Read-only admission preparation. No thresholds are supplied by this module."""
import time
import math


def cooldown(session, maximum, timeout, clock=time.perf_counter, sleep=time.sleep):
    started = clock()
    checks = 0
    while True:
        field = session.query("nvmlDeviceGetTemperature", 0)
        checks += 1
        temperature = field["value"] if field["status"] == "available" else None
        if type(temperature) not in (int, float) or not math.isfinite(temperature):
            temperature = None
        waited = clock() - started
        result = {"max_start_temp_c": maximum, "timeout_seconds": timeout, "temperature_c": temperature,
                  "wait_seconds": waited, "checks": checks, "status": "unavailable", "passed": False}
        if temperature is None:
            return {**result, "reason": "start_temperature_unavailable"}
        if maximum is None:
            return {**result, "reason": "temperature_condition_not_configured"}
        if temperature <= maximum:
            return {**result, "status": "passed", "passed": True, "reason": None}
        if waited >= timeout:
            return {**result, "status": "timeout", "reason": "cooldown_timeout"}
        sleep(min(0.1, timeout - waited))


def contention_reasons(counts):
    others = counts.get("other_processes", {})
    if others.get("status") != "available" or type(others.get("value")) is not int:
        return ["other_compute_processes_unknown"]
    return ["other_compute_processes_present"] if others["value"] > 0 else []
