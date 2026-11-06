# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Numerical regression tests for frozen transfer and policy outputs."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    """Load a portable module.

    Parameters
    ----------
    name : str
        Module filename stem.

    Returns
    -------
    module
        Loaded analysis implementation.
    """
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "analysis_scripts" / "cs1_offline" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SummaryTests(unittest.TestCase):
    """Verify independent small-sample interval arithmetic."""

    def test_transfer_interval(self):
        """Check transfer mean and Student t bounds.

        Returns
        -------
        None
            Asserts the known three-observation result.
        """
        result = load_module("paired_transfer").ci([1, 2, 3])
        self.assertAlmostEqual(result["mean"], 2)
        self.assertAlmostEqual(result["lo"], -0.484137711719546, places=12)
        self.assertAlmostEqual(result["hi"], 4.484137711719546, places=12)

    def test_generic_policy_dataset(self):
        """Analyze new candidate IDs and nonconsecutive rounds.

        Returns
        -------
        None
            Verifies the pair metadata and round IDs control calculation.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                dict(
                    round=r,
                    candidate=c,
                    shape="w10",
                    policy=p,
                    status="accepted",
                    runs=10,
                    power_metrics={"energy_mj_per_inference": value},
                )
                for r in [3, 7, 12]
                for c, offset in [("new_a", 0), ("new_b", 2)]
                for p, value in [("awake", r + offset), ("stop", r + offset + 1)]
            ]
            (root / "records.json").write_text(json.dumps(records))
            (root / "pairs.json").write_text(
                json.dumps([{"left": "new_a", "right": "new_b"}])
            )
            load_module("policy_replay").main(
                [
                    "--energy-records",
                    str(root / "records.json"),
                    "--pairs",
                    str(root / "pairs.json"),
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            result = json.loads((root / "out/paired_analysis.json").read_text())
            self.assertEqual(result["new_a_minus_new_b_awake"]["n"], 3)
            self.assertEqual(result["new_a_minus_new_b_awake"]["mean_mj"], -2)

    def test_holm(self):
        """Check Holm adjustment restores the original ordering.

        Returns
        -------
        None
            Asserts monotone step-down adjusted p-values.
        """
        actual = load_module("paired_transfer").holm_adjust([0.04, 0.001, 0.02])
        for observed, expected in zip(actual, [0.04, 0.003, 0.04]):
            self.assertAlmostEqual(observed, expected)

    def test_repeats_interval(self):
        """Check repeated-pair Student t arithmetic.

        Returns
        -------
        None
            Asserts known mean and interval bounds.
        """
        result = load_module("repeats").summarize([1, 2, 3])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean_mj"], 2)
        self.assertAlmostEqual(result["ci95_lo"], -0.484137711719546, places=12)

    def test_policy_interval(self):
        """Check policy interval and retained paired samples.

        Returns
        -------
        None
            Asserts the known three-observation result.
        """
        result = load_module("policy_replay").summary([1, 2, 3])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["differences_mj"], [1, 2, 3])
        self.assertAlmostEqual(result["mean_mj"], 2)
        self.assertAlmostEqual(result["ci95_lo"], -0.484137711719546, places=12)
        self.assertAlmostEqual(result["ci95_hi"], 4.484137711719546, places=12)


if __name__ == "__main__":
    unittest.main()
