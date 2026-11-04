"""Classify one diagnostic GEMM result; never change or admit a baseline."""
import json
import math
import re
import sys


def classify(d, rc, gpu, mode, size, kind, state, expected_commit=None):
    # A wrapper-detected safety failure takes priority over a completed JSON.
    if rc not in (0, 7):
        return 8, 'benchmark_or_safety_failure_exit_' + str(rc)
    if d.get('baseline_eligible') is not False:
        return 8, 'baseline_flag_missing_or_changed'
    git = d.get('benchmark_git', {})
    if not re.fullmatch('[0-9a-f]{40}', git.get('commit') or '') or git.get('tracked_dirty') is not False:
        return 8, 'git_identity_missing_or_tracked_tree_dirty'
    if expected_commit is not None and git['commit'] != expected_commit:
        return 8, 'git_commit_mismatch'
    if d.get('status') not in ('draft_validated', 'duration_target_not_met'):
        return 8, d.get('status', 'missing_status')
    if d.get('identity_verified') is not True:
        return 8, 'gpu_identity_unverified'
    if d.get('device', {}).get('cuda_index') != gpu:
        return 8, 'gpu_index_mismatch'
    if d.get('mode') != mode or d.get('shape') != dict(m=size, n=size, k=size):
        return 8, 'configuration_mismatch'
    if d.get('preflight', {}).get('status') != 'ready':
        return 8, 'preflight_not_ready'
    if d.get('accuracy', {}).get('passed') is not True or any(
        d.get('accuracy_checks', {}).get(phase, {}).get('passed') is not True
        for phase in ('pre', 'post')
    ):
        return 8, 'accuracy_validation_failed_or_missing'
    if d.get('precision_mode_suspect') is not False:
        return 8, 'precision_mode_suspect_or_unknown'
    for phase in ('before', 'after'):
        processes = d.get('compute_processes', {}).get(phase, {})
        others = processes.get('other_processes', {})
        if processes.get('status') != 'available' or others.get('status') != 'available' or others.get('value') != 0:
            return 8, 'compute_process_contention_or_unknown'
    rate = d.get('summary', {}).get('tflops_from_median_latency')
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        return 8, 'invalid_performance_value'
    if state == 'before' and mode == 'bf16' and size == 8192 and rate > 35:
        return 8, 'before_bf16_8192_above_35_requires_state_review'
    target = d.get('timed_measurement', {}).get('target_met')
    if kind == 'full':
        timing = d.get('timed_measurement', {})
        span = timing.get('gpu_span_seconds')
        samples = timing.get('selected_samples')
        if (timing.get('min_timed_seconds') != 2 or type(span) not in (int, float)
                or not math.isfinite(span) or span <= 0 or type(samples) is not int
                or not 1 <= samples <= 10000 or target is not (span >= 2)):
            return 8, 'invalid_or_inconsistent_timed_measurement'
        if target is False and rc == 7 and d['status'] == 'duration_target_not_met':
            return 7, 'duration_shortfall'
        if target is not True or rc != 0 or d['status'] != 'draft_validated':
            return 8, 'inconsistent_duration_status'
    elif rc != 0 or d['status'] != 'draft_validated':
        return 8, 'smoke_failure'
    return 0, 'valid_diagnostic'


if __name__ == '__main__':
    try:
        path, rc, gpu, mode, size, kind, state, expected_commit = sys.argv[1:]
        with open(path) as f:
            result = json.load(f)
        code, reason = classify(result, int(rc), int(gpu), mode, int(size), kind, state, expected_commit)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        code, reason = 8, 'unreadable_or_invalid_result'
    print(reason)
    raise SystemExit(code)
