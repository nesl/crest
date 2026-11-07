#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Validate saved training trajectories and export conditional budget comparisons."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

from scipy.stats import spearmanr
import yaml

POLICIES = ("budget_only", "original_policy")


def read(path):
    """Read a saved JSON artifact.

    Parameters
    ----------
    path : pathlib.Path
        Input file.

    Returns
    -------
    dict or list
        Decoded contents.
    """
    return json.loads(path.read_text())


def sha256(path):
    """Hash a local artifact.

    Parameters
    ----------
    path : pathlib.Path
        Input file.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    """Reject invalid analysis inputs.

    Parameters
    ----------
    condition : bool
        Required invariant.
    message : str
        Error description.

    Raises
    ------
    ValueError
        If the invariant fails.
    """
    if not condition:
        raise ValueError(message)


def front_ids(points):
    """Return nondominated identifiers, retaining objective ties.

    Parameters
    ----------
    points : dict
        Candidate IDs mapped to minimized (error, energy) pairs.

    Returns
    -------
    list
        Front IDs sorted by energy, error and identifier.
    """
    return sorted(
        (
            k
            for k, v in points.items()
            if not any(
                all(a <= b for a, b in zip(w, v)) and any(a < b for a, b in zip(w, v))
                for j, w in points.items()
                if j != k
            )
        ),
        key=lambda k: (points[k][1], points[k][0], k),
    )


def reference_comparison(early, reference):
    """Compare ranks and reference-evaluated shortlist regret.

    Parameters
    ----------
    early, reference : dict
        Matched candidate identifiers mapped to (RMSE, historical energy).

    Returns
    -------
    dict
        Front memberships, tie-aware rank correlation, pair counts and cap regret.
    """
    require(
        set(early) == set(reference) and bool(early),
        "Unmatched or empty candidate panel",
    )
    ids = sorted(early)
    selected, final = front_ids(early), front_ids(reference)
    counts = Counter()
    for a, b in itertools.combinations(ids, 2):
        x, y = early[a][0] - early[b][0], reference[a][0] - reference[b][0]
        counts[
            "tied" if x == 0 or y == 0 else "concordant" if x * y > 0 else "discordant"
        ] += 1
    x, y = [early[k][0] for k in ids], [reference[k][0] for k in ids]
    rho = (
        float(spearmanr(x, y).statistic)
        if len(set(x)) > 1 and len(set(y)) > 1
        else None
    )
    regret = []
    for cap in sorted({v[1] for v in reference.values()}):
        best = min(v[0] for v in reference.values() if v[1] <= cap)
        kept = [reference[k][0] for k in selected if reference[k][1] <= cap]
        absolute = min(kept) - best if kept else None
        regret.append(
            dict(
                energy_cap_mj=cap,
                reference_rmse=best,
                selected_rmse=min(kept) if kept else None,
                absolute_regret=absolute,
                coverage_failure=not kept,
                relative_regret_pct=(
                    100 * absolute / best if absolute is not None and best > 0 else None
                ),
            )
        )
    pairs = math.comb(len(ids), 2)
    return dict(
        panel_size=len(ids),
        early_front=selected,
        reference_front=final,
        reference_evaluated_early_front=front_ids({k: reference[k] for k in selected}),
        entered=[k for k in final if k not in selected],
        left=[k for k in selected if k not in final],
        spearman_rho=rho,
        pairs=pairs,
        concordant=counts["concordant"],
        discordant=counts["discordant"],
        tied=counts["tied"],
        pairwise_agreement_pct=100 * counts["concordant"] / pairs if pairs else None,
        max_shortlist_regret_pct=max(
            (
                r["relative_regret_pct"]
                for r in regret
                if r["relative_regret_pct"] is not None
            ),
            default=None,
        ),
        regret=regret,
    )


def load_panels(paths):
    """Load complete matched panels, validating frozen identities and histories.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        Batch directories containing manifest/execution/config and candidate runs.

    Returns
    -------
    tuple
        Candidate metadata, checkpoint rows, histories, budgets and seeds.
    """
    candidates, records, histories = {}, {}, {}
    baseline = None
    payloads = set()
    for folder in map(Path, paths):
        m, e = read(folder / "manifest.json"), read(folder / "execution.json")
        require(
            sha256(folder / "manifest.json") == e["identity"]["manifest_sha256"],
            "Manifest hash mismatch",
        )
        require(
            sha256(folder / "source_config.yaml") == m["snapshot_sha256"],
            "Source snapshot hash mismatch",
        )
        require(
            sha256(folder / "effective_config.yaml") == e["effective_config_sha256"],
            "Effective config hash mismatch",
        )
        config = yaml.safe_load((folder / "effective_config.yaml").read_text())
        config.pop("outputs", None)
        require(e["test_loaded"] is False, "Held-out test set was loaded")
        budgets, seeds = m["budgets"], m["seeds"]
        require(
            budgets
            and len(set(budgets)) == len(budgets)
            and all(isinstance(b, int) and b > 0 for b in budgets),
            "Invalid budgets",
        )
        require(seeds and len(set(seeds)) == len(seeds), "Invalid seeds")
        identity = (
            sorted(budgets),
            sorted(seeds),
            m["source_csv_sha256"],
            m["snapshot_sha256"],
            config,
            {
                k: e[k]
                for k in (
                    "tensorflow_version",
                    "python",
                    "train_shape",
                    "validation_shape",
                )
            },
            {k: e["identity"][k] for k in ("dataset_dir", "split_hashes")},
        )
        require(
            baseline is None or identity == baseline, "Incompatible panel identities"
        )
        baseline = identity
        for c in m["candidates"]:
            cid = c["candidate_id"]
            require(cid not in candidates, "Duplicate candidate identifier")
            payload = c.get("replay_payload_key")
            require(
                payload is None or payload not in payloads,
                "Duplicate candidate payload",
            )
            if payload is not None:
                payloads.add(payload)
            candidates[cid] = c
            for seed in seeds:
                work = folder / cid / f"seed_{seed}"
                require(
                    (work / "complete.json").is_file()
                    and not (work / "failure.json").exists(),
                    "Incomplete trajectory",
                )
                h, rows, cp = (
                    read(work / name)
                    for name in ("history.json", "results.json", "checkpoints.json")
                )
                require(
                    [r["epoch"] for r in h] == list(range(1, max(budgets) + 1)),
                    "Incomplete epoch history",
                )
                times = [r["elapsed_s"] for r in h]
                require(
                    all(math.isfinite(t) and t >= 0 for t in times)
                    and all(a <= b for a, b in zip(times, times[1:])),
                    "Invalid cumulative training times",
                )
                require(
                    all(math.isfinite(r["val_loss"]) for r in h),
                    "Invalid validation loss",
                )
                expected = set(itertools.product(budgets, POLICIES))
                index = {(r["budget"], r["policy"]): r for r in rows}
                saved = {(r["budget"], r["policy"]): r for r in cp}
                require(
                    len(rows) == len(cp) == len(expected)
                    and set(index) == set(saved) == expected,
                    "Missing or duplicate checkpoint records",
                )
                for (b, p), r in index.items():
                    require(
                        r["status"] == "ok"
                        and r["candidate_id"] == cid
                        and r["seed"] == seed,
                        "Invalid checkpoint identity/status",
                    )
                    require(
                        r["energy_policy"] == "fixed_historical_measurement"
                        and r["energy_mj_per_inference"] == c["historical_energy_mj"]
                        and r["quantization_mode"] == c["quantization_mode"],
                        "Changed energy or quantization metadata",
                    )
                    require(
                        all(math.isfinite(v) and v >= 0 for v in r["metrics"].values()),
                        "Invalid evaluation metric",
                    )
                    metrics = r["metrics"]
                    if {"rmse_vel_x", "rmse_vel_y"} <= metrics.keys():
                        require(
                            math.isclose(
                                metrics["rmse_total"],
                                metrics["rmse_vel_x"] + metrics["rmse_vel_y"],
                                abs_tol=1e-12,
                            ),
                            "RMSE component mismatch",
                        )
                    require(
                        all(
                            r[k] == saved[b, p][k]
                            for k in (
                                "selected_epoch",
                                "sha256",
                                "checkpoint",
                                "virtual_stop_epoch",
                            )
                        ),
                        "Checkpoint metadata mismatch",
                    )
                    stop = h[b - 1].get("virtual_stop_epoch")
                    require(
                        stop == r["virtual_stop_epoch"],
                        "Virtual stopping metadata mismatch",
                    )
                    cutoff = (
                        min(b, stop)
                        if p == "original_policy" and stop is not None
                        else b
                    )
                    require(
                        r["selected_epoch"]
                        == min(h[:cutoff], key=lambda v: v["val_loss"])["epoch"],
                        "Checkpoint selection mismatch",
                    )
                    checkpoint = work / r["checkpoint"]
                    if checkpoint.exists():
                        require(
                            sha256(checkpoint) == r["sha256"],
                            "Checkpoint file hash mismatch",
                        )
                    r["_checkpoint_binary_verified"] = checkpoint.exists()
                    records[cid, seed, b, p] = r
                histories[cid, seed] = h
    require(bool(candidates), "Empty panel")
    return candidates, records, histories, sorted(budgets), sorted(seeds)


def analyze(
    paths, output_dir, reference_budget=300, reference_policy="budget_only", seeds=None
):
    """Export validated metrics and per-seed comparisons.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        Compatible saved batch directories.
    output_dir : pathlib.Path
        External result directory.
    reference_budget : int, optional
        Epoch budget used as the comparison reference.
    reference_policy : str, optional
        ``budget_only`` or ``original_policy``.
    seeds : sequence of int, optional
        Seeds to report; defaults to all available seeds.

    Returns
    -------
    dict
        Numerical summaries and validation coverage.
    """
    candidates, records, histories, budgets, available = load_panels(paths)
    seeds = available if seeds is None else seeds
    require(
        reference_budget in budgets and reference_policy in POLICIES,
        "Reference checkpoint unavailable",
    )
    require(
        bool(seeds) and len(set(seeds)) == len(seeds) and set(seeds) <= set(available),
        "Requested seeds unavailable or duplicated",
    )
    summary = dict(
        candidate_count=len(candidates),
        quantization_counts=dict(
            Counter(c["quantization_mode"] for c in candidates.values())
        ),
        budgets=budgets,
        reference_budget=reference_budget,
        reference_policy=reference_policy,
        seeds={},
        limitations="Selected panel; fixed historical energy; reference budget does not establish convergence. Prefix training times include checkpoint saving/logging and exclude subsequent evaluation.",
    )
    summary["validation"] = dict(
        checkpoint_metadata_records=len(records),
        checkpoint_binary_hashes_verified=sum(
            r["_checkpoint_binary_verified"] for r in records.values()
        ),
        missing_checkpoint_binaries=sum(
            not r["_checkpoint_binary_verified"] for r in records.values()
        ),
        note="Checkpoint/result metadata and selection validated for every record; actual binary hashes checked only for locally present checkpoint files. External source CSV and dataset split contents are not reopened.",
    )
    metric_rows, figure_rows = [], []
    for seed in seeds:

        def points(b, p):
            """Collect checkpoint objectives for the current seed.

            Parameters
            ----------
            b : int
                Epoch budget.
            p : str
                Checkpoint policy.

            Returns
            -------
            dict
                Candidate IDs mapped to (RMSE, energy).
            """
            return {
                cid: (
                    records[cid, seed, b, p]["metrics"]["rmse_total"],
                    c["historical_energy_mj"],
                )
                for cid, c in candidates.items()
            }

        reference = points(reference_budget, reference_policy)
        comparisons = []
        for b, p in itertools.product(budgets, POLICIES):
            result = reference_comparison(points(b, p), reference)
            result.update(budget=b, policy=p)
            comparisons.append(result)
            for cid in candidates:
                r = records[cid, seed, b, p]
                metric_rows.append(
                    dict(
                        candidate_id=cid,
                        seed=seed,
                        budget=b,
                        policy=p,
                        selected_epoch=r["selected_epoch"],
                        rmse_total=r["metrics"]["rmse_total"],
                        energy_mj_per_inference=r["energy_mj_per_inference"],
                        energy_policy=r["energy_policy"],
                        quantization_mode=r["quantization_mode"],
                    )
                )
                if p == "budget_only":
                    figure_rows.append(
                        dict(
                            candidate_id=cid,
                            budget=b,
                            rmse_total=r["metrics"]["rmse_total"],
                            energy_mj=r["energy_mj_per_inference"],
                            source="validated_saved_trajectory",
                            seed=seed,
                        )
                    )
        summary["seeds"][str(seed)] = dict(
            comparisons=comparisons,
            cumulative_training_hours={
                str(b): sum(
                    histories[cid, seed][b - 1]["elapsed_s"] for cid in candidates
                )
                / 3600
                for b in budgets
            },
            policies_identical={
                str(b): all(
                    records[cid, seed, b, "budget_only"]["sha256"]
                    == records[cid, seed, b, "original_policy"]["sha256"]
                    and points(b, "budget_only")[cid]
                    == points(b, "original_policy")[cid]
                    for cid in candidates
                )
                for b in budgets
            },
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    for name, rows in [("checkpoint_metrics.csv", metric_rows)]:
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for seed in seeds:
        name = (
            "figure10_training_budget.csv"
            if len(seeds) == 1
            else f"figure10_training_budget_seed_{seed}.csv"
        )
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "candidate_id",
                    "budget",
                    "rmse_total",
                    "energy_mj",
                    "source",
                ],
            )
            writer.writeheader()
            writer.writerows(
                {k: v for k, v in r.items() if k != "seed"}
                for r in figure_rows
                if r["seed"] == seed
            )
    return summary


def main(argv=None):
    """Run analysis from explicit input and output paths.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments; defaults to process arguments.

    Returns
    -------
    None
        Writes private analysis outputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-budget", type=int, default=300)
    parser.add_argument("--reference-policy", choices=POLICIES, default="budget_only")
    parser.add_argument(
        "--seeds",
        "--seed",
        type=int,
        nargs="+",
        help="Seeds to report (default: all available; submitted panel uses 17)",
    )
    args = parser.parse_args(argv)
    analyze(
        args.panels,
        args.output_dir,
        args.reference_budget,
        args.reference_policy,
        args.seeds,
    )


if __name__ == "__main__":
    main()
