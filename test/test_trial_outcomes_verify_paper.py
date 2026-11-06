# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Submitted recipe checks distinguish campaign and paired populations."""

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).parents[1] / "analysis_scripts/trial_outcomes"
for name in ("combine", "verify_paper"):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
verify = module.verify


class PaperRecipeTests(unittest.TestCase):
    def rows(self):
        tables = []
        for study, counts in [
            ("CS1", (915, 564, 57)),
            ("CS2", (624, 0, 74)),
            ("CS3", (1200, 2, 85)),
        ]:
            rows = []
            for outcome, count in zip(
                ("measured_passed", "measured_violation", "not_deployed_or_incomplete"),
                counts,
            ):
                for _ in range(count):
                    n = len(rows)
                    rows.append(
                        dict(
                            case_study=study,
                            campaign="example",
                            trial=n,
                            outcome=outcome,
                            paired_eligible=n < 905,
                            paired_included=n < 756,
                            model_eligible=n < 756,
                        )
                    )
            tables.append(rows)
        return tables

    def test_recipe_and_distinct_populations(self):
        self.assertEqual(verify(self.rows())["population_excluded"], 149)

    def test_contradictory_model_membership_rejected(self):
        tables = self.rows()
        tables[0][800]["model_eligible"] = True
        with self.assertRaises(ValueError):
            verify(tables)

    def test_wrong_campaign_counts_rejected(self):
        tables = self.rows()
        tables[2].pop()
        with self.assertRaises(ValueError):
            verify(tables)


if __name__ == "__main__":
    unittest.main()
