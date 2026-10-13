"""Pure acceptance rules for the repaired STM32 measurement protocol."""
import math

PROTOCOL = 'rtc_wall_v3_stop_evidence_awake_tail'


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_measurement(metrics, *, latency_s, phase, clock_mhz, runs, period_ms,
                         dut_log, harness_log, require_harness):
    """Return instrumentation reasons separately from deployment infeasibility.

    The 2% timing agreement band detects gross timing mismatch, not meter accuracy.
    Two RTC ticks (62 us) bound endpoint quantization for cadence-window validity.
    Requested sleep is never interpreted as measured Stop residency.
    """
    reasons = []
    checks = {
        'phase_matches': metrics.get('phase') == phase,
        'clock_matches': metrics.get('clock_hz') == clock_mhz * 1_000_000,
        'runs_match': metrics.get('runs') == runs,
        'latency_positive_finite': positive(latency_s),
        'protocol_matches': f'measurement protocol output: {PROTOCOL}' in dut_log,
        'period_matches': f'cadence budget (ms): {round(period_ms)}' in dut_log,
        'rtc_window_positive_finite': positive(metrics.get('timer_per_window_s')),
        'rtc_lse': metrics.get('rtc_clock_source') == 'LSE' and metrics.get('rtc_clock_hz_nominal') == 32768,
    }
    rtc = metrics.get('timer_per_window_s')
    if require_harness:
        harness = metrics.get('harness_latency_s')
        checks.update(raw_harness_present=bool(harness_log),
                      energy_positive_finite=positive(metrics.get('energy_mj_per_inference')),
                      harness_window_positive_finite=positive(harness))
        checks['window_sources_agree'] = (positive(rtc) and positive(harness)
                                          and abs(harness * runs - rtc) / rtc <= .02)
    if phase == 'cadenced':
        slots, entries = metrics.get('slept_slots'), metrics.get('stop_entries')
        def valid_count(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 and int(value) == value and value <= max(0, runs - 1)
        checks['stop_counts_valid'] = valid_count(slots) and valid_count(entries)
        checks['stop_entries_match_slots'] = checks['stop_counts_valid'] and entries == slots
        misses = metrics.get('deadline_miss_count')
        checks['deadline_count_valid'] = isinstance(misses, (int, float)) and not isinstance(misses, bool) and math.isfinite(misses) and misses >= 0 and int(misses) == misses
    reasons = [name for name, passed in checks.items() if not passed]
    eligibility = []
    if positive(latency_s) and latency_s * 1000 > period_ms:
        eligibility.append('active_latency_exceeds_budget')
    if phase == 'cadenced':
        if checks['deadline_count_valid'] and metrics['deadline_miss_count'] > 0:
            eligibility.append('deadline_miss')
        if positive(rtc) and abs(rtc - runs * period_ms / 1000) > .000062:
            eligibility.append('cadence_window_outside_budget')
    return {'instrumentation_valid': not reasons, 'checks': checks,
            'instrumentation_reasons': reasons, 'deployment_eligible': not reasons and not eligibility,
            'infeasibility_reasons': eligibility, 'energy_unit': 'mJ_per_release',
            'energy_mj_per_release': metrics.get('energy_mj_per_inference'),
            'energy_mj_aggregate': metrics.get('energy_mj_per_inference', 0) * runs if positive(metrics.get('energy_mj_per_inference')) else None}
