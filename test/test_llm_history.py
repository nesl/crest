# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for bounded prompt evidence sourced directly from Optuna."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from optuna.study import StudyDirection
from optuna.trial import TrialState

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.optimizers.llm.history import build_recent_trial_history  # noqa: E402


def trial(number, *, params=None, attrs=None, state=TrialState.COMPLETE, value=None):
    """Build one compact FrozenTrial-like test record."""
    return SimpleNamespace(
        number=number,
        state=state,
        params=params or {},
        user_attrs=attrs or {},
        system_attrs={},
        value=value,
        values=None if value is None else [value],
    )


class RecentTrialHistoryTests(unittest.TestCase):
    """Verify bounded measured evidence and raw parameter preservation."""

    def test_history_respects_bound_and_includes_available_measurements(self) -> None:
        """Only the newest trials appear with compact measured evidence."""
        study = SimpleNamespace(
            directions=[StudyDirection.MAXIMIZE],
            trials=[
                trial(0, params={"dilations_index": 1}, value=0.1),
                trial(
                    1,
                    params={"dilations_index": 2, "quantization_mode": "int8_ptq"},
                    value=0.2,
                    attrs={
                        "feasibility_status": "feasible",
                        "latency_ms": 4.5,
                        "energy_mj_per_inference": 0.75,
                        "ram_bytes": 1024,
                        "flash_bytes": 2048,
                        "quantization_mode": "int8_ptq",
                        "cpu_clock_mhz_requested": 400,
                        "task_metrics": {"rmse_total": 0.33},
                    },
                ),
                trial(
                    2,
                    params={"dilations_index": 3},
                    state=TrialState.PRUNED,
                    attrs={"pruned": True, "prune_reason": "resource limit"},
                ),
            ],
        )

        history = build_recent_trial_history(study, window_size=2)

        self.assertEqual([record["number"] for record in history], [1, 2])
        measured = history[0]
        self.assertEqual(measured["params"]["dilations_index"], 2)
        self.assertEqual(measured["directions"], ["maximize"])
        self.assertEqual(measured["latency_ms"], 4.5)
        self.assertEqual(measured["energy_mj_per_inference"], 0.75)
        self.assertEqual(measured["ram_bytes"], 1024)
        self.assertEqual(measured["flash_bytes"], 2048)
        self.assertEqual(measured["cpu_clock_mhz_requested"], 400)
        self.assertEqual(measured["task_metrics"], {"rmse_total": 0.33})
        self.assertTrue(history[1]["pruned"])
        self.assertEqual(history[1]["prune_reason"], "resource limit")

    def test_missing_optional_metrics_and_unavailable_sentinels_are_omitted(self) -> None:
        """Sparse desktop/failure trials remain prompt-safe without fake evidence."""
        study = SimpleNamespace(
            directions=[StudyDirection.MINIMIZE],
            trials=[
                trial(
                    4,
                    params={"dilations_index": 7},
                    value=1.2,
                    attrs={"latency_ms": -1.0, "ram_bytes": -1},
                )
            ],
        )

        history = build_recent_trial_history(study, window_size=10)

        self.assertEqual(history[0]["params"], {"dilations_index": 7})
        self.assertNotIn("latency_ms", history[0])
        self.assertNotIn("ram_bytes", history[0])
        self.assertNotIn("task_metrics", history[0])
        self.assertEqual(build_recent_trial_history(study, window_size=0), ())


if __name__ == "__main__":
    unittest.main()
