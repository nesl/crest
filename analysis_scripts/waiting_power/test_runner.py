# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Hardware-free scheduling, acceptance, and waiting-power algebra tests."""
import pytest

from .runner import accept, schedule, shapes, solve, study_period_ms


def test_schedules_are_deterministic_and_end_in_ten_run_block():
    items = schedule(3, [80, 160], 2, 9, 'grouped')
    assert items == schedule(3, [80, 160], 2, 9, 'grouped')
    assert len(items) == 18
    assert shapes(2)[items[-1]['shape']][1] == 10


def test_waiting_power_solution_and_invalid_gap():
    # E=2mJ, C=.1s, T=.5s; busy=50mW, stop=5mW.
    assert solve(2, .1, 22, 1, 58, 10, .5) == pytest.approx((50, 5))
    with pytest.raises(ValueError):
        solve(2, .5, 22, 1, 58, 10, .5)
    with pytest.raises(SystemExit):
        study_period_ms({'period_ms': 0})
    assert study_period_ms({'period_ms': 250, 'period_s': .25}) == 250


def test_acceptance_fails_closed():
    assert accept({}, 0, 'cadenced', 1, 80, 'protocol', [])
