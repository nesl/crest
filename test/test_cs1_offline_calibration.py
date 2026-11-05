"""Calibration recovery on an independent synthetic probe."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path = Path(__file__).parents[1] / "analysis_scripts/cs1_offline/calibration.py"
spec = importlib.util.spec_from_file_location("calibration", path)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def test_two_period_fit_with_unfamiliar_clock_and_rounds():
    """Recover known powers on two rounds at a new clock and periods."""
    tables = []
    for period in [0.8, 0.2]:
        rows = []
        for rnd in [8, 12]:
            energy, duration, busy, stop, e0, correction = (
                7.0,
                0.03,
                90.0,
                20.0,
                2.0,
                -0.5,
            )
            y = (period - duration) * busy + e0
            w10 = 10 * energy + y + 9 * ((period - duration) * stop + correction)
            for shape, phase, runs, measured, entries in [
                ("continuous", "back_to_back", 10, energy, -1),
                ("w1", "cadenced", 1, energy + y, 0),
                ("w10", "cadenced", 10, w10 / 10, 9),
            ]:
                rows.append(
                    dict(
                        round=rnd,
                        clock_mhz=75,
                        shape=shape,
                        phase=phase,
                        runs=runs,
                        energy_mj_per_inference=measured,
                        status="accepted",
                        stop_entries=entries,
                        deadline_miss_count=0,
                        dwt_cycles_per_inference=duration * 75e6,
                        clock_hz=75e6,
                        harness_latency_s=duration,
                    )
                )
        tables.append(pd.DataFrame(rows))
    _, fitted = cal.calibrate(*tables, periods=(0.8, 0.2))
    np.testing.assert_allclose(
        fitted.loc[75, ["P_busy_mw_mean", "P_stop_mw_mean", "E0_mj_mean", "K_mj_mean"]],
        [90, 20, 2, -0.5],
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="unique"):
        cal.probe_windows(pd.concat([tables[0], tables[0].iloc[:1]]))
