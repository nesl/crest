"""Summarize nested profiling logs without summing across host clocks."""

from collections import defaultdict
import statistics
import argparse
import csv
import json
from pathlib import Path


def union_duration(intervals):
    """Measure the union of intervals on one clock.

    Parameters
    ----------
    intervals : iterable of tuple
        Inclusive start and exclusive end, in nanoseconds.

    Returns
    -------
    int
        Covered nanoseconds, counting overlaps once.
    """
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if hi < lo:
            raise ValueError("Negative interval")
        if end is None or lo >= end:
            total += hi - lo
        elif hi > end:
            total += hi - end
        end = max(end if end is not None else hi, hi)
    return total


def summarize(records):
    """Validate spans and subtract the union of immediate children.

    Parameters
    ----------
    records : list of dict
        Spans with local monotonic timestamps and parent identifiers.

    Returns
    -------
    list of dict
        Original fields plus inclusive/exclusive seconds and orphan flags.
    """
    indexed = {}
    for row in records:
        key = (row.get("run_id"), row["host"], row["pid"], row["span_id"])
        if key in indexed:
            raise ValueError(f"Duplicate span: {key}")
        if row["end_ns"] < row["start_ns"]:
            raise ValueError(f"Negative interval: {key}")
        if (
            row.get("duration_ns", row["end_ns"] - row["start_ns"])
            != row["end_ns"] - row["start_ns"]
        ):
            raise ValueError(f"Inconsistent duration: {key}")
        indexed[key] = row
    for key, row in indexed.items():
        visited = {key}
        parent = row.get("parent_id")
        while parent:
            parent_key = (row.get("run_id"), row["host"], row["pid"], parent)
            if parent_key in visited:
                raise ValueError("Cycle in span parents")
            visited.add(parent_key)
            ancestor = indexed.get(parent_key)
            parent = ancestor.get("parent_id") if ancestor else None
    children_by_parent = defaultdict(list)
    for row in records:
        if row.get("parent_id") == row["span_id"]:
            raise ValueError("A span cannot parent itself")
        children_by_parent[
            (row.get("run_id"), row["host"], row["pid"], row.get("parent_id"))
        ].append(row)
    result = []
    for row in records:
        children = children_by_parent[
            (row.get("run_id"), row["host"], row["pid"], row["span_id"])
        ]
        if any(
            c["start_ns"] < row["start_ns"] or c["end_ns"] > row["end_ns"]
            for c in children
        ):
            raise ValueError("Child interval outside parent")
        duration = row["end_ns"] - row["start_ns"]
        exclusive = duration - union_duration(
            [(c["start_ns"], c["end_ns"]) for c in children]
        )
        result.append(
            {
                **row,
                "inclusive_s": duration / 1e9,
                "exclusive_s": exclusive / 1e9,
                "orphan_parent": bool(
                    row.get("parent_id")
                    and (
                        row.get("run_id"),
                        row["host"],
                        row["pid"],
                        row.get("parent_id"),
                    )
                    not in indexed
                ),
            }
        )
    return result


def request_residuals(records):
    """Link requests across clocks using identifiers and durations.

    Parameters
    ----------
    records : list of dict
        Validated spans.

    Returns
    -------
    list of dict
        Request matches and residual seconds; ambiguous values are null.
    """
    by_id = {(r.get("run_id"), r["host"], r["pid"], r["span_id"]): r for r in records}
    servers = defaultdict(list)
    for row in records:
        if row["stage"] == "hil.request_processing" and row.get("request_id"):
            servers[(row.get("run_id"), row["request_id"])].append(row)
    result = []
    for row in records:
        if row["stage"] != "transport.round_trip":
            continue
        matches = (
            servers[(row.get("run_id"), row.get("request_id"))]
            if row.get("request_id")
            else []
        )
        residual = None
        if len(matches) == 1:
            residual = (row["duration_ns"] - matches[0]["duration_ns"]) / 1e9
        startup_s = None
        service_s = None
        if len(matches) == 1:
            root = matches[0]

            def descends(item):
                """Determine ancestry within the server's clock domain.

                Returns
                -------
                bool
                    Whether the item descends from the matched request.
                """
                visited = set()
                parent = item.get("parent_id")
                while parent and parent not in visited:
                    if parent == root["span_id"]:
                        return True
                    visited.add(parent)
                    parent = by_id.get(
                        (root.get("run_id"), root["host"], root["pid"], parent), {}
                    ).get("parent_id")
                return False

            descendants = [
                r
                for r in records
                if r.get("run_id") == root.get("run_id")
                and r["host"] == root["host"]
                and r["pid"] == root["pid"]
                and descends(r)
            ]
            service = [r for r in descendants if r["stage"] == "hil.service"]
            startup = [r for r in descendants if r["stage"].startswith("startup.")]
            startup_s = (
                union_duration([(r["start_ns"], r["end_ns"]) for r in startup]) / 1e9
            )
            service_startup_s = (
                union_duration(
                    [
                        (
                            max(r["start_ns"], service[0]["start_ns"]),
                            min(r["end_ns"], service[0]["end_ns"]),
                        )
                        for r in startup
                        if len(service) == 1
                        and r["end_ns"] > service[0]["start_ns"]
                        and r["start_ns"] < service[0]["end_ns"]
                    ]
                )
                / 1e9
                if len(service) == 1
                else None
            )
            service_s = service[0]["duration_ns"] / 1e9 if len(service) == 1 else None
        result.append(
            {
                "run_id": row.get("run_id"),
                "request_id": row.get("request_id"),
                "candidate_id": row.get("candidate_id"),
                "first_candidate": row.get("first_candidate"),
                "server_matches": len(matches),
                "transport_queueing_residual_s": residual,
                "server_service_s": service_s,
                "server_startup_s": startup_s,
                "server_service_excluding_startup_s": (
                    None if service_s is None else service_s - service_startup_s
                ),
                "negative_residual": residual is not None and residual < 0,
            }
        )
    return result


def worker_residuals(records):
    """Link worker roots to waits using durations across clocks.

    Parameters
    ----------
    records : list of dict
        Validated spans with optional external_parent_id links.

    Returns
    -------
    list of dict
        Worker evidence counts and launch/exit residual seconds.
    """
    roots = defaultdict(list)
    for row in records:
        if row.get("external_parent_id") and row.get("parent_id") is None:
            roots[(row.get("run_id"), row["external_parent_id"])].append(row)
    output = []
    for wait in records:
        if wait["stage"] != "evaluation.worker_wait":
            continue
        children = roots[(wait.get("run_id"), wait["span_id"])]
        residual = None
        # imports and main are sequential roots of one worker, not two workers.
        processes = {(r["host"], r["pid"]) for r in children}
        if (
            len(processes) == 1
            and len(children) == 2
            and {"worker.imports", "worker.total"} == {r["stage"] for r in children}
        ):
            residual = (
                wait["duration_ns"]
                - union_duration([(r["start_ns"], r["end_ns"]) for r in children])
            ) / 1e9
        output.append(
            {
                "run_id": wait.get("run_id"),
                "parent_span_id": wait["span_id"],
                "worker_process_count": len(processes),
                "worker_root_count": len(children),
                "launch_exit_uninstrumented_residual_s": residual,
                "missing_worker_evidence": residual is None,
                "negative_residual": residual is not None and residual < 0,
            }
        )
    return output


def stage_summary(rows):
    """Aggregate stages separately by run, board, condition, and host.

    Returns
    -------
    list of dict
        Descriptive span statistics; inclusive sums may contain nesting.
    """
    groups = defaultdict(list)
    keys = ("run_id", "board", "condition", "host", "stage")
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    output = []
    for key, group in groups.items():
        values = [r["inclusive_s"] for r in group]
        output.append(
            {
                **dict(zip(keys, key)),
                "span_count": len(group),
                "error_count": sum(r.get("status") == "error" for r in group),
                "inclusive_sum_s": sum(values),
                "exclusive_sum_s": sum(r["exclusive_s"] for r in group),
                "inclusive_mean_s": statistics.mean(values),
                "inclusive_median_s": statistics.median(values),
                "inclusive_min_s": min(values),
                "inclusive_max_s": max(values),
            }
        )
    return output


def write_csv(path, rows):
    """Write records to CSV, including an empty file for empty inputs.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    rows : list of dict
        Records sharing an output table.
    """
    path.write_text("")
    if rows:
        fields = sorted(set().union(*(r.keys() for r in rows)))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    """Read explicit JSONL inputs and write derived tables.

    Notes
    -----
    The output directory must differ from each source log directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output == path.resolve().parent for path in args.logs):
        parser.error("Output must not be a source log directory")
    records = [
        json.loads(line)
        for path in args.logs
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    rows = summarize(records)
    records = [{**r, "duration_ns": r["end_ns"] - r["start_ns"]} for r in records]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "spans.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "workers.json").write_text(
        json.dumps(worker_residuals(records), indent=2) + "\n"
    )
    (args.output / "requests.json").write_text(
        json.dumps(request_residuals(records), indent=2) + "\n"
    )
    write_csv(args.output / "spans.csv", rows)
    write_csv(args.output / "stages.csv", stage_summary(rows))
    trials = [r for r in rows if r["stage"] == "trial.total"]
    write_csv(args.output / "trials.csv", trials)


if __name__ == "__main__":
    main()
