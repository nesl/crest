# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic timing attribution checks without hardware or archived data."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis_scripts/timing_attribution/analyze.py"
)
SPEC = importlib.util.spec_from_file_location("timing_analyze", MODULE_PATH)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


def span(identifier, start, end, parent=None, stage="work", **fields):
    """Construct a synthetic span.

    Returns
    -------
    dict
        One record with timestamps expressed in seconds for readability.
    """
    return dict(
        run_id="run",
        host="host",
        pid=1,
        span_id=identifier,
        start_ns=int(start * 1e9),
        end_ns=int(end * 1e9),
        duration_ns=int((end - start) * 1e9),
        parent_id=parent,
        stage=stage,
        **fields
    )


class TimingTests(unittest.TestCase):
    """Exercise overlap, clock-domain, and cross-process accounting."""

    def test_nested_overlap(self):
        """Subtract immediate child coverage only.

        Returns
        -------
        None
        """
        rows = analyze.summarize(
            [
                span("root", 0, 10),
                span("a", 1, 6, "root"),
                span("b", 4, 9, "root"),
                span("c", 2, 3, "a"),
            ]
        )
        self.assertEqual([r["exclusive_s"] for r in rows], [2, 4, 5, 1])
        self.assertEqual(analyze.union_duration([(1, 2), (2, 3), (1, 3)]), 2)

    def test_invalid_records(self):
        """Reject duplicate, inconsistent, cyclic, and noncontained spans.

        Returns
        -------
        None
        """
        invalid = [
            [span("a", 0, 1)] * 2,
            [span("a", 2, 1)],
            [span("a", 0, 1), span("b", 0, 2, "a")],
            [span("a", 0, 1, "b"), span("b", 0, 1, "a")],
            [{**span("a", 0, 1), "duration_ns": 2}],
        ]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                analyze.summarize(records)

    def test_clock_and_run_scope(self):
        """Keep repeated IDs separate across runs and processes.

        Returns
        -------
        None
        """
        rows = [
            span("a", 0, 1),
            {**span("a", 10, 11), "pid": 2},
            {**span("a", 20, 21), "run_id": "other"},
            span("b", 0, 1, "absent"),
        ]
        self.assertTrue(analyze.summarize(rows)[-1]["orphan_parent"])

    def test_requests(self):
        """Link remote clocks and clip startup subtraction to service.

        Returns
        -------
        None
        """
        rows = [
            span("client", 0, 12, stage="transport.round_trip", request_id="q"),
            {
                **span(
                    "server", 100, 110, stage="hil.request_processing", request_id="q"
                ),
                "host": "remote",
            },
            {**span("service", 102, 110, "server", "hil.service"), "host": "remote"},
            {**span("boot", 100, 104, "server", "startup.load"), "host": "remote"},
            # Same ID on another process must not corrupt server ancestry.
            {**span("service", 0, 1), "pid": 99},
        ]
        analyze.summarize(rows)
        result = analyze.request_residuals(rows)[0]
        self.assertEqual(result["transport_queueing_residual_s"], 2)
        self.assertEqual(result["server_startup_s"], 4)
        self.assertEqual(result["server_service_excluding_startup_s"], 6)
        rows.append({**rows[1], "span_id": "duplicate_request"})
        self.assertIsNone(
            analyze.request_residuals(rows)[0]["transport_queueing_residual_s"]
        )

    def test_worker_links(self):
        """Union overlapping worker roots and withhold ambiguous evidence.

        Returns
        -------
        None
        """
        rows = [
            span("wait", 0, 12, stage="evaluation.worker_wait"),
            {
                **span(
                    "imports",
                    100,
                    104,
                    stage="worker.imports",
                    external_parent_id="wait",
                ),
                "pid": 2,
            },
            {
                **span(
                    "total", 103, 110, stage="worker.total", external_parent_id="wait"
                ),
                "pid": 2,
            },
        ]
        self.assertEqual(
            analyze.worker_residuals(rows)[0]["launch_exit_uninstrumented_residual_s"],
            2,
        )
        rows.append({**rows[-1], "span_id": "extra", "pid": 3})
        self.assertTrue(analyze.worker_residuals(rows)[0]["missing_worker_evidence"])

    def test_cli(self):
        """Write explicit derived outputs while preserving source files.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            content = json.dumps(span("trial", 0, 1, stage="trial.total")) + "\n"
            source.write_text(content)
            subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    str(source),
                    "--output",
                    str(root / "derived"),
                ],
                check=True,
            )
            self.assertEqual(source.read_text(), content)
            self.assertEqual(
                {p.name for p in (root / "derived").iterdir()},
                {
                    "spans.json",
                    "requests.json",
                    "workers.json",
                    "spans.csv",
                    "stages.csv",
                    "trials.csv",
                },
            )
            rejected = subprocess.run(
                [sys.executable, str(MODULE_PATH), str(source), "--output", str(root)],
                capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
