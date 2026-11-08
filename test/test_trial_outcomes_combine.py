# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Independent final-trial counting checks."""

import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "combine_outcomes",
    Path(__file__).parents[1] / "analysis_scripts/appendix_trial_outcomes/combine.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_disjoint_counts_with_arbitrary_studies():
    rows = [
        dict(
            case_study="experiment-a",
            campaign="run",
            trial=0,
            outcome="measured_passed",
        ),
        dict(
            case_study="experiment-a",
            campaign="run",
            trial=1,
            outcome="not_deployed_or_incomplete",
        ),
        dict(
            case_study="experiment-b",
            campaign="run",
            trial=0,
            outcome="measured_violation",
        ),
    ]
    result = module.combine([rows[:2], rows[2:]])
    assert result[-1] == {
        "outcome": "Total",
        "experiment-a": 2,
        "experiment-b": 1,
        "Total": 3,
    }
    assert result[1]["Total"] == 1


@pytest.mark.parametrize("defect", ["duplicate", "outcome", "scope"])
def test_rejects_double_counts_or_wrong_population(defect):
    row = dict(case_study="a", campaign="r", trial=1, outcome="measured_passed")
    rows = [row]
    if defect == "duplicate":
        rows.append(row.copy())
    elif defect == "outcome":
        row["outcome"] = "unknown"
    else:
        row["scope"] = "replay"
    with pytest.raises(ValueError):
        module.combine([rows])


def test_rejects_host_only_rows_without_scope():
    row = dict(
        case_study="study",
        campaign="host",
        trial=0,
        outcome="measured_passed",
        host_only="True",
    )
    with pytest.raises(ValueError):
        module.combine([[row]])


def test_population_flags_without_manuscript_counts():
    rows = [
        dict(outcome="measured_passed", paired_eligible=True,
             paired_included=True, model_eligible=True),
        dict(outcome="measured_passed", paired_eligible="1",
             paired_included="0", model_eligible="false"),
    ]
    module.validate_population_flags(rows)


@pytest.mark.parametrize("changes", [
    {"paired_eligible": False},
    {"outcome": "measured_violation"},
    {"model_eligible": False},
])
def test_contradictory_population_flags_rejected(changes):
    row = dict(outcome="measured_passed", paired_eligible=True,
               paired_included=True, model_eligible=True)
    row.update(changes)
    with pytest.raises(ValueError):
        module.validate_population_flags([row])
