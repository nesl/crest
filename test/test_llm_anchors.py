# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Best-trial prompt evidence stays bounded, feasible, and study-derived."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from optuna.study import StudyDirection
from optuna.trial import TrialState

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crest.optimizers.llm.history import (  # noqa: E402
    build_best_trial_anchors,
    build_recent_trial_history,
)


def trial(number, values, *, attrs=None, state=TrialState.COMPLETE):
    """Allow malformed historical values Optuna would reject on new writes."""
    return SimpleNamespace(
        number=number,
        state=state,
        params={"width": number + 2},
        user_attrs=attrs or {},
        system_attrs={},
        values=values,
        value=values[0] if values and len(values) == 1 else None,
    )


def study(trials, *directions):
    return SimpleNamespace(trials=trials, directions=directions)


class BestTrialAnchorTests(unittest.TestCase):
    def test_scalar_ranking_respects_direction_and_stable_ties(self):
        trials = [trial(4, [3.0]), trial(0, [9.0]), trial(2, [3.0]), trial(1, [5.0])]
        for direction, expected in [
            (StudyDirection.MINIMIZE, [2, 4, 1]),
            (StudyDirection.MAXIMIZE, [0, 1, 2]),
        ]:
            with self.subTest(direction=direction):
                anchors = build_best_trial_anchors(study(trials, direction), anchor_count=3)
                self.assertEqual([r["number"] for r in anchors], expected)
                self.assertEqual(anchors[0]["directions"], [direction.name.lower()])

    def test_only_successful_finite_complete_trials_can_anchor(self):
        records = [
            trial(0, [10.0]),
            trial(1, [11.0], attrs={"feasibility_status": "feasible"}),
            trial(2, [0.0], attrs={"feasibility_status": "infeasible"}),
            trial(3, [0.0], attrs={"feasible": False}),
            trial(4, [0.0], attrs={"feasibility_status": "not_evaluated",
                                       "feasible": True, "error_code": 1, "hil_error_code": 1}),
            trial(5, [0.0], attrs={"pruned": True}),
            trial(6, [0.0], attrs={"error_code_label": "HIL_MASTER_RAM_OVERFLOW"}),
            trial(7, [float("inf")]),
            trial(8, [float("nan")]),
            trial(9, [0.0], state=TrialState.PRUNED),
            trial(10, [0.0], state=TrialState.FAIL),
            trial(11, [0.0], state=TrialState.RUNNING),
            trial(12, [0.0], state=TrialState.WAITING),
            trial(13, None),
        ]
        anchors = build_best_trial_anchors(study(records, StudyDirection.MINIMIZE), anchor_count=20)
        self.assertEqual([r["number"] for r in anchors], [4, 0, 1])

    def test_enabled_feasibility_requires_explicit_feasible_evidence(self):
        records = [
            trial(0, [0.0]),
            trial(1, [1.0], attrs={"feasibility_status": "disabled"}),
            trial(2, [2.0], attrs={"feasibility_status": "feasible"}),
            trial(3, [0.0], attrs={"feasibility_status": "feasible", "pruned": True}),
        ]
        anchors = build_best_trial_anchors(
            study(records, StudyDirection.MINIMIZE), anchor_count=5, feasibility_enabled=True
        )
        self.assertEqual([r["number"] for r in anchors], [2])
        disabled_policy = build_best_trial_anchors(study(records, StudyDirection.MINIMIZE), anchor_count=5)
        self.assertEqual([r["number"] for r in disabled_policy], [0, 1, 2])

    def test_anchor_records_reuse_history_measurements_and_raw_params(self):
        measured = trial(0, [0.4], attrs={
            "feasibility_status": "feasible", "latency_ms": 4.5,
            "ram_bytes": 1024, "flash_bytes": -1,
            "quantization_mode": "int8_ptq", "task_metrics": {"rmse_total": 0.4},
        })
        source = study([measured], StudyDirection.MINIMIZE)
        anchors = build_best_trial_anchors(source, anchor_count=5)
        self.assertEqual(anchors, build_recent_trial_history(source, window_size=1))
        self.assertEqual(anchors[0]["params"], {"width": 2})
        self.assertEqual(anchors[0]["latency_ms"], 4.5)
        self.assertNotIn("flash_bytes", anchors[0])

    def test_bounded_pareto_subset_focuses_on_knee_across_scales_and_directions(self):
        front = [(0, 1000), (1, 600), (2, 350), (3, 200), (4, 150),
                 (5, 110), (6, 70), (8, 30), (10, 0)]
        for direction, sign in [(StudyDirection.MINIMIZE, 1), (StudyDirection.MAXIMIZE, -1)]:
            with self.subTest(direction=direction):
                records = [trial(i, [float(x), float(sign * y)]) for i, (x, y) in enumerate(front)]
                records += [
                    trial(20, [12.0, float(sign * 1200)]),
                    trial(21, [-1.0, float(sign * -100)], attrs={"pruned": True}),
                    trial(22, [float("nan"), 0.0]), trial(23, [1.0]),
                ]
                source = study(records, StudyDirection.MINIMIZE, direction)
                anchors = build_best_trial_anchors(source, anchor_count=3)
                self.assertEqual({r["number"] for r in anchors}, {2, 3, 4})
                self.assertEqual(
                    anchors,
                    build_best_trial_anchors(study(list(reversed(records)), *source.directions), anchor_count=3),
                )
                self.assertEqual(
                    [r["number"] for r in build_best_trial_anchors(source, anchor_count=1)], [3]
                )
                self.assertEqual(len(build_best_trial_anchors(source, anchor_count=20)), len(front))
                self.assertTrue(all(r["directions"] == ["minimize", direction.name.lower()] for r in anchors))

    def test_flat_front_and_three_objectives_use_deterministic_compromise(self):
        flat = study([trial(i, [float(i), float(10 - i)]) for i in range(11)],
                     StudyDirection.MINIMIZE, StudyDirection.MINIMIZE)
        self.assertEqual(
            {r["number"] for r in build_best_trial_anchors(flat, anchor_count=3)}, {4, 5, 6}
        )
        three = study([
            trial(0, [0.0, 5.0, 5.0]), trial(1, [3.0, 3.0, 3.0]),
            trial(2, [5.0, 0.0, 5.0]), trial(3, [5.0, 5.0, 0.0]),
        ], StudyDirection.MINIMIZE, StudyDirection.MINIMIZE, StudyDirection.MINIMIZE)
        self.assertEqual([r["number"] for r in build_best_trial_anchors(three, anchor_count=1)], [1])

    def test_pareto_dominance_respects_mixed_directions(self):
        source = study([
            trial(0, [1.0, 3.0]), trial(1, [2.0, 5.0]),
            trial(2, [2.0, 2.0]), trial(3, [3.0, 4.0]),
        ], StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE)
        anchors = build_best_trial_anchors(source, anchor_count=5)
        self.assertEqual({r["number"] for r in anchors}, {0, 1})

    def test_count_zero_disables_anchors_and_invalid_counts_fail(self):
        self.assertEqual(build_best_trial_anchors(study([], StudyDirection.MINIMIZE), anchor_count=0), ())
        for count in [-1, True, 1.5, "5", None]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_best_trial_anchors(study([], StudyDirection.MINIMIZE), anchor_count=count)


if __name__ == "__main__":
    unittest.main()
