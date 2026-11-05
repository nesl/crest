"""Reconcile saved timing logs and prepare external plotting tables."""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

CONDITIONS = ("continuous", "cadenced", "flops", "memory")


def ancestors(row, index):
    """Follow parents within one host/process.

    Parameters
    ----------
    row : dict
        Starting span.
    index : dict
        Spans keyed by run, host, process and span ID.

    Yields
    ------
    dict
        Successive parent spans.
    """
    seen = set()
    while row.get("parent_id"):
        key = (row.get("run_id"), row["host"], row["pid"], row["parent_id"])
        if key in seen or key not in index:
            raise ValueError("Cyclic or missing timing parent")
        seen.add(key)
        row = index[key]
        yield row


def arena_search_split(root, spans, reports, prefix, mode):
    """Separate failed measurement sessions from build and programming costs.

    Parameters
    ----------
    root : pathlib.Path
        Read-only campaign archive.
    spans : list of dict
        Reconciled spans.
    reports : dict
        Condition-to-client-report mapping.
    prefix : str
        Run-directory prefix.
    mode : str
        ``arena-flags`` pairs console flags with base attempts; ``phase-attempts``
        verifies a backend with no arena sweep.

    Returns
    -------
    dict
        Per-condition failed measurement durations and attempt counts.
    """
    if mode not in {"arena-flags", "phase-attempts"}:
        raise ValueError("Unknown arena accounting mode")
    result = {}
    index = {(r.get("run_id"), r["host"], r["pid"], r["span_id"]): r for r in spans}
    for condition in ("continuous", "cadenced"):
        report = reports[condition]
        run_id = report["run"]["run_id"]
        rows = [
            r
            for r in spans
            if r.get("run_id") == run_id
            and any(a["stage"] == "hil.service" for a in ancestors(r, index))
        ]
        count = report["counts"]["trained"]
        if mode == "arena-flags":
            log = (
                root / f"server/runs/{prefix}{condition}/server.console.log"
            ).read_text()
            flags = [
                int(x)
                for x in re.findall(
                    r"HIL_controller attempt idx=\d+ arena=\d+ KiB err_flag=(\d+)", log
                )
            ]
            base = sorted(
                [
                    r
                    for r in rows
                    if r["stage"] == "backend.measure"
                    and not any(
                        a["stage"] == "hardware.cadenced_second_pass"
                        for a in ancestors(r, index)
                    )
                ],
                key=lambda r: r["start_ns"],
            )
            if len(base) != len(flags) or not flags or set(flags) - {0, 2}:
                raise ValueError("Cannot match arena attempts to success/timeout flags")
            if len({(r["host"], r["pid"]) for r in base}) != 1:
                raise ValueError(
                    "Console-order attribution requires one server process"
                )
            failed = [r for r, flag in zip(base, flags) if flag == 2]
            build_stages = {
                "harness.build",
                "harness.program",
                "backend.compile",
                "backend.upload",
                "backend.program_images",
                "backend.program_weights",
                "toolchain.program_image",
            }
            for failure in failed:
                if any(
                    r["stage"] in build_stages
                    and (r["host"], r["pid"]) == (failure["host"], failure["pid"])
                    and failure["start_ns"] <= r["start_ns"]
                    and r["end_ns"] <= failure["end_ns"]
                    for r in rows
                ):
                    raise ValueError(
                        "Failed measurement contains build/programming time"
                    )
            followups = [
                r
                for r in rows
                if r["stage"] == "backend.measure"
                and any(
                    a["stage"] == "hardware.cadenced_second_pass"
                    for a in ancestors(r, index)
                )
            ]
            result[condition] = dict(
                failed_probe_measurement_s=sum(r["inclusive_s"] for r in failed),
                failed_attempts=len(failed),
                base_attempts=len(base),
                successful_base_attempts=len(base) - len(failed),
                followup_attempts=len(followups),
                candidates=count,
                failed_spans=[
                    dict(
                        span_id=r["span_id"],
                        candidate_id=r.get("candidate_id"),
                        duration_s=r["inclusive_s"],
                    )
                    for r in failed
                ],
            )
        else:
            phases = [r for r in rows if r["stage"] == "backend.phase_attempt"]
            expected = count * (1 if condition == "continuous" else 2)
            if len(phases) != expected or any(
                r.get("status") == "error" for r in phases
            ):
                raise ValueError(
                    "Unexpected phase retries; zero timeouts cannot be assumed"
                )
            result[condition] = dict(
                failed_probe_measurement_s=0,
                failed_attempts=0,
                phase_attempts=len(phases),
                candidates=count,
                failed_spans=[],
            )
    return dict(
        definition="Failed arena-probe measurement/protocol sessions; excludes build/programming.",
        conditions=result,
    )


def prepare(source, output, prefix="", mode="phase-attempts"):
    """Run span and campaign analysis without modifying the source archive.

    Parameters
    ----------
    source, output : pathlib.Path
        Input archive and external output directory.
    prefix : str, optional
        Prefix before condition names under client/server ``runs/``.
    mode : str, optional
        Arena accounting method; see ``arena_search_split``.

    Returns
    -------
    None
        Writes reconciled tables and detailed local audit outputs.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source archive")
    scripts = Path(__file__).resolve().parent
    reports, logs = {}, []
    for condition in CONDITIONS:
        report = json.loads(
            (source / f"client/runs/{prefix}{condition}/results.json").read_text()
        )
        counts = report["counts"]
        if (
            report["status"] != "complete"
            or counts["trained"] <= 0
            or any(counts.get(k, 0) for k in ("failed", "remaining", "rejected"))
        ):
            raise ValueError(f"Incomplete or rejected candidate panel: {condition}")
        if counts["trained"] != counts["requested"]:
            raise ValueError(f"Not every requested candidate trained: {condition}")
        reports[condition] = report
        for host in ("client", "server"):
            logs.extend(
                sorted(
                    (source / f"{host}/runs/{prefix}{condition}/timings").glob(
                        "**/*.jsonl"
                    )
                )
            )
    if not logs:
        raise ValueError("No saved timing logs found")
    output.mkdir(parents=True, exist_ok=True)
    report_dir = output / "audit"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "analyze.py"),
            *map(str, logs),
            "--output",
            str(report_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "summarize_campaign.py"),
            str(source),
            "--run-prefix",
            prefix,
            "--report-dir",
            str(report_dir),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    breakdown = json.loads((output / "breakdown.json").read_text())
    validation = breakdown["validation"]
    if validation["orphan_parents"] or validation["worker_issues"]:
        raise ValueError(f"Invalid timing links: {validation}")
    spans = json.loads((report_dir / "spans.json").read_text())
    arena = arena_search_split(source, spans, reports, prefix, mode)
    (output / "arena_search.json").write_text(json.dumps(arena, indent=2) + "\n")
    proxy = []
    for condition, report in reports.items():
        clients = [
            r
            for r in spans
            if r.get("run_id") == report["run"]["run_id"]
            and r["stage"] == "process.total"
            and r.get("role") == "client"
        ]
        if len(clients) != 1:
            raise ValueError("Expected one client process per condition")
        client = clients[0]
        for stage in ("proxy.flops", "proxy.memory_estimate"):
            matches = [
                r
                for r in spans
                if (r["host"], r["pid"]) == (client["host"], client["pid"])
                and r.get("run_id") == report["run"]["run_id"]
                and r["stage"] == stage
            ]
            if len(matches) != report["counts"]["trained"] or len(
                {r.get("candidate_id") for r in matches}
            ) != len(matches):
                raise ValueError(f"Missing or duplicate {stage} measurements")
            proxy.extend(
                {k: r[k] for k in ("condition", "candidate_id", "stage", "inclusive_s")}
                for r in matches
            )
    with (output / "proxy_times.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["condition", "candidate_id", "stage", "inclusive_s"]
        )
        writer.writeheader()
        writer.writerows(proxy)
    print(
        f'Reconciled {sum(r["counts"]["trained"] for r in reports.values())} evaluations.'
    )


def main():
    """Parse input paths and the saved backend's accounting method.

    Returns
    -------
    None
        Executes the offline pipeline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-prefix", default="")
    parser.add_argument(
        "--arena-mode", choices=["arena-flags", "phase-attempts"], required=True
    )
    args = parser.parse_args()
    prepare(args.source, args.output_dir, args.run_prefix, args.arena_mode)


if __name__ == "__main__":
    main()
