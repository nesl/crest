# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Build an additive client timeline and nested HIL/fit drilldowns from analyzed spans."""

import argparse
import csv
import json
import collections
import math
from pathlib import Path

try:
    from .compare_results import compare
except ImportError:
    from compare_results import compare
CONDITIONS = ("continuous", "cadenced", "flops", "memory")


def summarize(
    source,
    report_dir,
    output_dir,
    *,
    run_prefix="stm32-",
    conditions=CONDITIONS,
    client_host=None,
    server_host=None,
):
    """Summarize additive timing without modifying campaign evidence.

    Parameters
    ----------
    source : pathlib.Path
        Extracted campaign directory containing client/runs results.
    report_dir : pathlib.Path
        Analyzer output directory with spans, requests, and workers JSON.
    output_dir : pathlib.Path
        Destination for breakdown JSON and the two CSV tables.
    run_prefix : str, optional
        Prefix before condition names in client run directories.
    conditions : sequence of str, optional
        Conditions to include in the requested order.
    client_host, server_host : str, optional
        Host overrides; otherwise derive clients from process roles and servers
        from the host containing hil.service spans.

    Returns
    -------
    dict
        Timing breakdown, validation counts, and matched quality comparison.
    """
    source, report_dir, output_dir = map(Path, (source, report_dir, output_dir))
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be nonempty and unique")
    if (
        output_dir.resolve() == source.resolve()
        or source.resolve() in output_dir.resolve().parents
    ):
        raise ValueError(
            "Output directory must be outside the read-only source archive"
        )
    rows = json.loads((report_dir / "spans.json").read_text())
    index = {(r.get("run_id"), r["host"], r["pid"], r["span_id"]): r for r in rows}

    def lineage(r):
        """Return a span followed by its ancestors.

        Parameters
        ----------
        r : dict
            Span whose ancestry is needed.

        Returns
        -------
        list of dict
            Span and ancestors within the same run, host, and process.
        """
        result = [r]
        seen = {r["span_id"]}
        while result[-1].get("parent_id"):
            key = (r.get("run_id"), r["host"], r["pid"], result[-1]["parent_id"])
            if key not in index:
                raise ValueError(f"Missing parent span: {key}")
            parent = index[key]
            if parent["span_id"] in seen:
                raise ValueError("Cycle")
            seen.add(parent["span_id"])
            result.append(parent)
        return result

    reports = [
        json.loads((source / f"client/runs/{run_prefix}{c}/results.json").read_text())
        for c in conditions
    ]
    requests = json.loads((report_dir / "requests.json").read_text())
    workers = json.loads((report_dir / "workers.json").read_text())
    output = {}
    stage_totals = []
    cold = []
    for condition, report in zip(conditions, reports):
        parent = [
            r
            for r in rows
            if r["condition"] == condition
            and r["stage"] == "process.total"
            and r.get("role") == "client"
            and (client_host is None or r["host"] == client_host)
        ]
        if len(parent) != 1:
            raise ValueError(
                f"Expected one client process for {condition}, found {len(parent)}"
            )
        parent = parent[0]
        mainrows = [
            r
            for r in rows
            if r["condition"] == condition
            and (r["host"], r["pid"]) == (parent["host"], parent["pid"])
        ]
        category = collections.defaultdict(float)
        fit = collections.defaultdict(float)
        for r in mainrows:
            stages = {q["stage"] for q in lineage(r)}
            if "training.fit_including_validation_checkpoint" in stages:
                label = "training_fit"
                if r["stage"] == "training.callback":
                    fit[r["callback"]] += r["exclusive_s"]
                elif "training.validation" in stages:
                    fit["in_fit_validation"] += r["exclusive_s"]
                else:
                    fit["training_and_fit_overhead"] += r["exclusive_s"]
            elif "hil.request_total" in stages:
                label = "hil_request"
            elif "evaluation.total" in stages:
                label = "post_fit_evaluation"
            elif r["stage"] in ("loading.dataset", "loading.checkpoint"):
                label = "loading"
            elif r["stage"].startswith(("candidate.", "proxy.")):
                label = "model_and_proxy_setup"
            else:
                label = "other_startup_bookkeeping"
            category[label] += r["exclusive_s"]
        if not math.isclose(
            sum(category.values()), parent["inclusive_s"], abs_tol=1e-5
        ):
            raise ValueError(
                f"Client categories do not sum to process duration: {condition}"
            )
        if not math.isclose(sum(fit.values()), category["training_fit"], abs_tol=1e-5):
            raise ValueError(
                f"Fit detail does not sum to training duration: {condition}"
            )
        hil = collections.defaultdict(float)
        server_hosts = (
            {
                r["host"]
                for r in rows
                if r["condition"] == condition and r["stage"] == "hil.service"
            }
            if server_host is None
            else {server_host}
        )
        for r in rows:
            if r["condition"] != condition or r["host"] not in server_hosts:
                continue
            stages = {q["stage"] for q in lineage(r)}
            if "hil.service" not in stages:
                continue
            if any(x.startswith("startup.") for x in stages):
                label = "lazy_dataset_pipeline_startup"
            elif "backend.prepare" in stages:
                label = (
                    "model_export_and_backend_preparation"
                    if report["run"]["board"] == "portenta_m7"
                    else "export_stedgeai_analysis_codegen"
                )
            elif stages & {"backend.compile", "harness.build"}:
                label = "firmware_build_signing"
            elif stages & {
                "backend.program_images",
                "backend.program_weights",
                "harness.program",
                "toolchain.load_elf",
                "backend.upload",
                "toolchain.program_image",
            }:
                label = "programming_and_boot_loading"
            elif stages & {
                "hardware.runtime_session",
                "hardware.harness_done",
                "hardware.harness_ready",
                "hardware.serial_session",
                "hardware.harness_session",
                "backend.measure",
            }:
                label = "runtime_protocol_measurement_wait"
            else:
                label = "model_build_and_other_server_overhead"
            hil[label] += r["exclusive_s"]
        req = [r for r in requests if r["run_id"] == report["run"]["run_id"]]
        if req:
            if not all(
                r["server_matches"] == 1 and not r["negative_residual"] for r in req
            ):
                raise ValueError(
                    f"Unmatched server request or negative residual: {condition}"
                )
            if not math.isclose(
                sum(hil.values()), sum(r["server_service_s"] for r in req), abs_tol=1e-5
            ):
                raise ValueError(
                    f"HIL detail does not sum to server service duration: {condition}"
                )
        allhost = [
            r
            for r in rows
            if r["condition"] == condition and r["host"] == parent["host"]
        ]

        def incl(stage):
            """Sum inclusive client-host time for one stage.

            Parameters
            ----------
            stage : str
                Exact stage name.

            Returns
            -------
            float
                Inclusive seconds across matching spans.
            """
            return sum(r["inclusive_s"] for r in allhost if r["stage"] == stage)

        output[condition] = {
            "client_process_s": parent["inclusive_s"],
            "categories_s": dict(category),
            "fit_detail_s": dict(fit),
            "hil_detail_s": dict(hil),
            "rpc_residual_s": sum(r["transport_queueing_residual_s"] for r in req),
            "fit_percent": 100 * category["training_fit"] / parent["inclusive_s"],
            "host_tflite_loop_s": incl("evaluation.inference_loop"),
            "host_keras_evaluation_s": incl("evaluation.keras"),
            "host_export_s": incl("evaluation.export"),
            "worker_import_s": incl("worker.imports"),
            "trained_candidates": report["counts"]["trained"],
            "epochs": sum(r["stage"] == "training.epoch" for r in mainrows),
        }
        for t in report["trials"]:
            tr = [
                r
                for r in mainrows
                if r["stage"] == "trial.total"
                and r.get("candidate_id") == t["candidate_id"]
            ]
            if len(tr) != 1:
                raise ValueError(
                    f'Expected one trial span for {condition}/{t["candidate_id"]}, found {len(tr)}'
                )
            cold.append(
                {
                    "condition": condition,
                    "candidate_id": t["candidate_id"],
                    "first_candidate": t["first_candidate"],
                    "trial_s": tr[0]["inclusive_s"],
                    "fit_s": sum(
                        r["inclusive_s"]
                        for r in mainrows
                        if r["stage"] == "training.fit_including_validation_checkpoint"
                        and r.get("candidate_id") == t["candidate_id"]
                    ),
                }
            )
        stage_totals.append(
            {
                "condition": condition,
                "client_process_s": parent["inclusive_s"],
                **category,
                "fit_percent": output[condition]["fit_percent"],
            }
        )
    summary = {
        "conditions": output,
        "validation": {
            "span_count": len(rows),
            "orphan_parents": sum(r["orphan_parent"] for r in rows),
            "matched_requests": len(requests),
            "linked_workers": len(workers),
            "worker_issues": sum(
                r["missing_worker_evidence"] or r["negative_residual"] for r in workers
            ),
        },
        "quality": compare(reports, requests=requests),
    }
    totals = {
        key: sum(v["categories_s"].get(key, 0) for v in output.values())
        for key in dict.fromkeys(
            key for v in output.values() for key in v["categories_s"]
        )
    }
    total = sum(v["client_process_s"] for v in output.values())
    summary["all_conditions"] = {
        "client_process_s": total,
        "categories_s": totals,
        "fit_percent": 100 * totals["training_fit"] / total,
    }
    hil_conditions = [c for c in conditions if c in ("continuous", "cadenced")]
    summary["hil_conditions_fit_percent"] = (
        (
            100
            * sum(
                output[c]["categories_s"].get("training_fit", 0) for c in hil_conditions
            )
            / sum(output[c]["client_process_s"] for c in hil_conditions)
        )
        if hil_conditions
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "breakdown.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, data in [("breakdown.csv", stage_totals), ("candidate_times.csv", cold)]:
        with (output_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=(
                    list(dict.fromkeys(key for row in data for key in row))
                    if data
                    else [
                        "condition",
                        "candidate_id",
                        "first_candidate",
                        "trial_s",
                        "fit_s",
                    ]
                ),
            )
            w.writeheader()
            w.writerows(data)
    return summary


def main():
    """Parse the offline campaign summary CLI."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "archive", nargs="?", type=Path, help="Extracted read-only campaign directory"
    )
    p.add_argument("--source", type=Path, help="Extracted read-only campaign directory")
    p.add_argument(
        "--report-dir", required=True, type=Path, help="Analyzer output directory"
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Summary output directory outside source",
    )
    p.add_argument("--run-prefix", default="stm32-")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--client-host")
    p.add_argument("--server-host")
    a = p.parse_args()
    if (a.source is None) == (a.archive is None):
        p.error("Supply exactly one of --source or positional archive")
    summary = summarize(
        a.source or a.archive,
        a.report_dir,
        a.output_dir,
        run_prefix=a.run_prefix,
        conditions=a.conditions,
        client_host=a.client_host,
        server_host=a.server_host,
    )
    print(
        json.dumps({"conditions": len(summary["conditions"]), **summary["validation"]})
    )


if __name__ == "__main__":
    main()
