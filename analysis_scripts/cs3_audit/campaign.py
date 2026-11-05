"""Audit archived campaign effort and saved response usage without running NAS.

The CLI accepts a JSON object with ``campaigns`` (a list of objects containing
``name``, ``database``, and optional ``response_root`` and ``ledger``). Optional
``mapping`` overrides attribute names and response usage dotted paths. All input
paths are explicit; response_root must identify the active optimizer archive,
excluding migration backups. API totals cover saved responses, including summaries,
not necessarily all attempted requests or the feasible endpoint's time window.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from collections import Counter

DEFAULT_MAPPING = {
    "feasible": "feasible",
    "feasibility_status": "feasibility_status",
    "proposal": "crest_proposal",
    "proposal_source": "proposal_source",
    "input_tokens": "usage.prompt_tokens",
    "output_tokens": "usage.completion_tokens",
    "latency_ms": "latency_ms",
}


def dotted(value, path):
    """Return a nested field, preserving absent values as None.

    Parameters
    ----------
    value : dict
        Source record.
    path : str
        Dot-separated field path.
    """
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def numeric(value):
    """Check whether a value is finite, numeric, and nonnegative.

    Parameters
    ----------
    value : object
        Observed accounting value; booleans are rejected.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def read_trials(database, mapping=None, study_id=None):
    """Read terminal trials and durations from an Optuna SQLite archive.

    Parameters
    ----------
    database : path-like
        Explicit SQLite input, opened read-only.
    mapping : dict, optional
        Attribute key overrides.
    study_id : int, optional
        Required when the database contains multiple studies.
    """
    keys = {**DEFAULT_MAPPING, **(mapping or {})}
    database = Path(database).resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as conn:
        studies = [r[0] for r in conn.execute("select study_id from studies")]
        if study_id is None:
            if len(studies) != 1:
                raise ValueError(
                    "Specify study_id for a database with multiple studies"
                )
            study_id = studies[0]
        if study_id not in studies:
            raise ValueError("Unknown study_id")
        rows = []
        for tid, number, state, start, end in conn.execute(
            "select trial_id,number,state,datetime_start,datetime_complete from trials where study_id=?",
            (study_id,),
        ):
            if state not in ("COMPLETE", "PRUNED", "FAIL"):
                continue
            attrs = {
                k: json.loads(v)
                for k, v in conn.execute(
                    "select key,value_json from trial_user_attributes where trial_id=?",
                    (tid,),
                )
            }
            score = conn.execute(
                "select value from trial_values where trial_id=? and objective=0",
                (tid,),
            ).fetchone()
            feasible = attrs.get(
                keys["feasible"], attrs.get(keys["feasibility_status"]) == "feasible"
            )
            valid = (
                state == "COMPLETE"
                and feasible is True
                and score is not None
                and score[0] is not None
                and math.isfinite(score[0])
            )
            duration = (
                (
                    datetime.fromisoformat(end) - datetime.fromisoformat(start)
                ).total_seconds()
                if start and end
                else None
            )
            if duration is not None and duration < 0:
                raise ValueError("Trial end precedes start")
            proposal = attrs.get(keys["proposal"]) or {}
            source = proposal.get(keys["proposal_source"], "unknown")
            params = {}
            for key, value, distribution in conn.execute(
                "select param_name,param_value,distribution_json from trial_params where trial_id=?",
                (tid,),
            ):
                dist = json.loads(distribution)
                params[key] = (
                    dist["attributes"]["choices"][int(value)]
                    if dist["name"] == "CategoricalDistribution"
                    else value
                )
            study_name = conn.execute(
                "select study_name from studies where study_id=?", (study_id,)
            ).fetchone()[0]
            rows.append(
                dict(
                    study_name=study_name,
                    hyperparameters_json=json.dumps(
                        attrs.get("hyperparameters"), sort_keys=True
                    ),
                    f1=attrs.get("metric__macro_f1"),
                    energy_mj=attrs.get("energy_mj_per_inference"),
                    score=score[0] if score else None,
                    latency_ms=attrs.get(
                        "latency_ms", attrs.get("latency_ms_per_inference")
                    ),
                    raw_params_json=json.dumps(params, sort_keys=True),
                    number=number,
                    state=state,
                    start=start,
                    end=end,
                    feasible=valid,
                    duration_seconds=duration,
                    proposal_source=source,
                )
            )
    if any(r["end"] is None for r in rows):
        raise ValueError(
            "Terminal trial lacks completion time; completion ordering is unknown"
        )
    return sorted(rows, key=lambda r: (datetime.fromisoformat(r["end"]), r["number"]))


def effort(rows, endpoint):
    """Summarize the terminal prefix through an arbitrary feasible endpoint.

    Parameters
    ----------
    rows : list of dict
        Terminal trials in completion order, as returned by ``read_trials``.
    endpoint : int
        Positive target count of feasible completions.
    """
    if endpoint < 1:
        raise ValueError("endpoint must be positive")
    prefix = []
    feasible = 0
    for row in rows:
        prefix.append(row)
        feasible += row["feasible"]
        if feasible >= endpoint:
            break
    return dict(
        feasible_endpoint=endpoint,
        endpoint_reached=feasible >= endpoint,
        feasible=feasible,
        terminal_attempts=len(prefix),
        complete_trial_hours=sum(
            r["duration_seconds"] or 0 for r in prefix if r["state"] == "COMPLETE"
        )
        / 3600,
        all_terminal_trial_hours=sum(r["duration_seconds"] or 0 for r in prefix) / 3600,
        missing_duration_count=sum(r["duration_seconds"] is None for r in prefix),
        proposal_sources=dict(Counter(r["proposal_source"] for r in prefix)),
        last_trial_number=prefix[-1]["number"] if prefix else None,
    )


def saved_usage(response_root, mapping=None):
    """Sum observed saved-response fields; report missingness without imputation.

    Parameters
    ----------
    response_root : path-like
        Active optimizer directory, excluding sibling migration backups.
    mapping : dict, optional
        Dotted response field paths overriding defaults.
    """
    keys = {**DEFAULT_MAPPING, **(mapping or {})}
    files = sorted(Path(response_root).rglob("*.response.json"))
    result = dict(saved_response_records=len(files), response_groups={})
    for field in ("input_tokens", "output_tokens", "latency_ms"):
        values = [dotted(json.loads(f.read_text()), keys[field]) for f in files]
        result[field + "_observed_sum"] = sum(v for v in values if numeric(v))
        result[field + "_missing_or_invalid_records"] = sum(
            not numeric(v) for v in values
        )
    result["response_groups"] = dict(
        Counter(str(f.parent.relative_to(response_root)) for f in files)
    )
    return result, files


def audit(config, endpoints):
    """Return effort, accounting, and content hashes for explicit campaign inputs.

    Parameters
    ----------
    config : dict
        Explicit campaign paths, metadata, and optional field mappings.
    endpoints : sequence of int
        Positive feasible-completion targets.
    """
    if not endpoints or any(e < 1 for e in endpoints):
        raise ValueError("Specify at least one positive endpoint")
    names = [item["name"] for item in config["campaigns"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Campaign names must be nonempty and unique")
    rows, accounting, sources, observations = [], [], [], []
    for item in config["campaigns"]:
        mapping = {**config.get("mapping", {}), **item.get("mapping", {})}
        trials = read_trials(item["database"], mapping, item.get("study_id"))
        rows.extend(dict(campaign=item["name"], **effort(trials, e)) for e in endpoints)
        n = 0
        for trial in trials:
            if trial["feasible"]:
                n += 1
                observations.append(
                    dict(
                        campaign=item["name"],
                        board=item.get("board", ""),
                        method=item.get("method", ""),
                        trial=trial["number"],
                        feasible=True,
                        **{
                            k: trial[k]
                            for k in (
                                "f1",
                                "energy_mj",
                                "score",
                                "latency_ms",
                                "raw_params_json",
                                "study_name",
                                "hyperparameters_json",
                            )
                        },
                    )
                )
            if n >= max(endpoints):
                break
        paths = [Path(item["database"])]
        if item.get("response_root"):
            usage, files = saved_usage(Path(item["response_root"]), mapping)
            accounting.append(dict(campaign=item["name"], **usage))
            paths.extend(files)
        if item.get("ledger"):
            ledger = json.loads(Path(item["ledger"]).read_text())
            if not accounting or accounting[-1]["campaign"] != item["name"]:
                accounting.append(dict(campaign=item["name"]))
            accounting[-1]["ledger_charged_usd"] = ledger.get("charged_usd")
            accounting[-1]["ledger_max_usd"] = ledger.get("max_usd")
            paths.append(Path(item["ledger"]))
        for index, path in enumerate(paths):
            sources.append(
                dict(
                    campaign=item["name"],
                    source_index=index,
                    name=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    return dict(
        effort=rows,
        feasible_observations=observations,
        saved_response_accounting=accounting,
        sources=sources,
        qualifications=[
            "Duration sums are trial evaluation proxies, not elapsed campaign wall time.",
            "Saved response usage includes Memory summaries when archived beneath response_root.",
            "Saved response accounting is not restricted to feasible endpoint; missing records are not imputed.",
            "Ledger charges may include retained reservations and are not invoice totals.",
        ],
    )


def main(argv=None):
    """Write JSON and CSV evidence from an explicit input configuration.

    Parameters
    ----------
    argv : sequence of str, optional
        CLI arguments; omitted to read process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--endpoint", required=True, type=int, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit(json.loads(args.config.read_text()), args.endpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "campaign_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if result["feasible_observations"]:
        with (args.output_dir / "feasible_observations.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=result["feasible_observations"][0]
            )
            writer.writeheader()
            writer.writerows(result["feasible_observations"])
    if result["effort"]:
        with (args.output_dir / "effort.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=result["effort"][0])
            writer.writeheader()
            writer.writerows(
                {
                    **r,
                    "proposal_sources": json.dumps(
                        r["proposal_sources"], sort_keys=True
                    ),
                }
                for r in result["effort"]
            )


if __name__ == "__main__":
    main()
