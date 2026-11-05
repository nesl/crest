"""Legacy outcomes preserve terminal states and original eligibility evidence."""

import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "legacy", Path(__file__).parents[1] / "analysis_scripts/trial_outcomes/legacy.py"
)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


class LegacyTests(unittest.TestCase):
    def record(self, **attrs):
        return dict(
            trial=0,
            state="COMPLETE",
            attrs=dict(error_code_label="HIL_MASTER_SUCCESS", **attrs),
        )

    def test_original_missing_feasibility_convention(self):
        summary, rows = legacy.audit_campaign([self.record()], campaign="board")
        self.assertEqual(summary["measured_passed"], 1)
        self.assertEqual(summary["passed_feasibility_missing"], 1)
        self.assertIsNone(summary["instrumentation_failures"])
        self.assertFalse(rows[0]["feasibility_present"])

    def test_failures_are_disjoint(self):
        for label in (
            "HIL_MASTER_ARENA_EXHAUSTED",
            "HIL_MASTER_FLASH_OVERFLOW",
            "HIL_MASTER_FATAL",
        ):
            record = self.record(pruned=True, feasible=False)
            record["attrs"]["error_code_label"] = label
            self.assertEqual(legacy.classify(record)[0], "not_deployed_or_incomplete")
        self.assertEqual(
            legacy.classify(self.record(prune_reason="TFLite evaluation failed"))[0],
            "not_deployed_or_incomplete",
        )
        self.assertEqual(
            legacy.classify(self.record(feasible=False))[0], "measured_violation"
        )

    def test_state_and_duplicate_identity(self):
        record = self.record()
        record["state"] = "RUNNING"
        self.assertEqual(
            legacy.classify(record), ("not_deployed_or_incomplete", "state_RUNNING")
        )
        with self.assertRaises(ValueError):
            legacy.audit_campaign([record, record], campaign="board")

    def test_explicit_study_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.db"
            with sqlite3.connect(path) as db:
                db.executescript(
                    'CREATE TABLE studies(study_id INTEGER); INSERT INTO studies VALUES(7),(19); CREATE TABLE trials(trial_id INTEGER, number INTEGER, state TEXT, study_id INTEGER); INSERT INTO trials VALUES(1,0,"COMPLETE",7),(2,1,"FAIL",7),(3,0,"RUNNING",19); CREATE TABLE trial_user_attributes(trial_id INTEGER,key TEXT,value_json TEXT); CREATE TABLE trial_values(trial_id INTEGER,objective INTEGER,value REAL);'
                )
            with self.assertRaises(ValueError):
                legacy.read_sqlite(path)
            rows = legacy.read_sqlite(path, study_id=7, trial_min=1, trial_max=2)
            self.assertEqual([(r["trial"], r["state"]) for r in rows], [(1, "FAIL")])


if __name__ == "__main__":
    unittest.main()
