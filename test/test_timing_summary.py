"""Synthetic checks for additive timing and archive-safe output."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE_DIR = (
    Path(__file__).resolve().parents[1] / "analysis_scripts" / "timing_attribution"
)
sys.path.insert(0, str(MODULE_DIR))
try:
    from summarize_campaign import summarize
finally:
    sys.path.remove(str(MODULE_DIR))


class TimingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "archive"
        self.report = self.root / "analyzed"
        self.report.mkdir()
        self.rows = []
        self.requests = []

    def condition(self, condition, setup=False):
        run = self.source / "client" / "runs" / ("custom-" + condition)
        run.mkdir(parents=True)
        report = {
            "schema_version": 2,
            "timing_enabled": True,
            "status": "complete",
            "elapsed_s": 10,
            "counts": {"trained": 1},
            "run": {
                "board": "portenta_m7",
                "condition": condition,
                "run_id": condition,
                "manifest_sha256": "same",
                "seed": 4,
            },
            "trials": [
                {
                    "candidate_id": "candidate-a",
                    "trained": True,
                    "state": "COMPLETE",
                    "duration_s": 8,
                    "first_candidate": True,
                    "user_attrs": {},
                }
            ],
        }
        (run / "results.json").write_text(json.dumps(report))

        def span(span_id, stage, exclusive, inclusive, parent=None, **kw):
            self.rows.append(
                dict(
                    host="unfamiliar-client",
                    pid=1,
                    span_id=condition + span_id,
                    parent_id=condition + parent if parent else None,
                    stage=stage,
                    condition=condition,
                    exclusive_s=exclusive,
                    inclusive_s=inclusive,
                    orphan_parent=False,
                    **kw
                )
            )

        span("root", "process.total", 1, 10, role="client")
        span("trial", "trial.total", 2, 9, "root", candidate_id="candidate-a")
        span(
            "fit",
            "training.fit_including_validation_checkpoint",
            4,
            6,
            "trial",
            candidate_id="candidate-a",
        )
        span("validation", "training.validation", 2, 2, "fit")
        span("remaining", "proxy.setup" if setup else "loading.dataset", 1, 1, "trial")
        self.rows.append(
            dict(
                host="unfamiliar-server",
                pid=9,
                span_id=condition + "service",
                parent_id=None,
                stage="hil.service",
                condition=condition,
                exclusive_s=3,
                inclusive_s=3,
                orphan_parent=False,
            )
        )
        self.requests.append(
            dict(
                run_id=condition,
                server_matches=1,
                negative_residual=False,
                server_service_s=3,
                transport_queueing_residual_s=0.5,
            )
        )

    def run_summary(self, conditions):
        for name, records in [
            ("spans", self.rows),
            ("requests", self.requests),
            ("workers", []),
        ]:
            (self.report / (name + ".json")).write_text(json.dumps(records))
        return summarize(
            self.source,
            self.report,
            self.root / "out",
            run_prefix="custom-",
            conditions=conditions,
        )

    def test_dynamic_candidates_hosts_and_additive_union(self):
        self.condition("continuous")
        self.condition("flops", setup=True)
        result = self.run_summary(["continuous", "flops"])
        self.assertEqual(result["all_conditions"]["client_process_s"], 20)
        self.assertEqual(
            result["all_conditions"]["categories_s"]["model_and_proxy_setup"], 1
        )
        continuous = result["conditions"]["continuous"]
        self.assertEqual(sum(continuous["categories_s"].values()), 10)
        self.assertEqual(sum(continuous["fit_detail_s"].values()), 6)
        self.assertEqual(sum(continuous["hil_detail_s"].values()), 3)
        self.assertEqual(continuous["trained_candidates"], 1)
        self.assertEqual(result["hil_conditions_fit_percent"], 60)
        self.assertEqual(
            len((self.root / "out" / "candidate_times.csv").read_text().splitlines()), 3
        )

    def test_reject_output_inside_source(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            summarize(self.source, self.report, self.source / "report")

    def test_missing_client_evidence_is_explicit(self):
        self.condition("flops")
        self.rows = [r for r in self.rows if r.get("role") != "client"]
        with self.assertRaisesRegex(ValueError, "Expected one client"):
            self.run_summary(["flops"])


if __name__ == "__main__":
    unittest.main()
