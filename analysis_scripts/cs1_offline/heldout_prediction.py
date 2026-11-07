# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Offline CS1 architecture-separated prediction and fixed-bank selection validation."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = [
    "affine",
    "proportional",
    "calibrated",
    "measured_continuous",
    "scheduled_direct_affine",
]


def load(predictions, observations, archive_root=None):
    """Reconstruct inference graph identities from frozen source logs.

    Parameters
    ----------
    predictions, observations : path-like
        Explicit frozen input CSV paths.
    archive_root : path-like, optional
        Relocated source archive.

    Returns
    -------
    pandas.DataFrame
        Merged candidate manifest with conservative graph identities.
    """
    d = pd.read_csv(predictions)
    o = pd.read_csv(observations)
    required_predictions = {
        "pool",
        "trial",
        "latency_s",
        "continuous_mj",
        "scheduled_mj",
        "clock_mhz",
        "rmse",
        "study",
        "optuna_trial",
        "continuous_latency_ms",
        "continuous_hat_mj",
        "T_s",
        "C_s",
        "wait_mw",
        "offset_mj",
    }
    for label, frame, required in [
        ("predictions", d, required_predictions),
        ("observations", o, {"pool", "trial", "quantization", "source"}),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
        if (
            frame.empty
            or frame[["pool", "trial"]].isna().any().any()
            or frame.duplicated(["pool", "trial"]).any()
        ):
            raise ValueError(f"{label} must have nonempty, unique pool/trial keys")
    d = d.merge(
        o[["pool", "trial", "quantization", "source"]],
        on=["pool", "trial"],
        validate="one_to_one",
    )
    if d[["quantization", "source"]].isna().any().any() or len(d) != len(
        pd.read_csv(predictions)
    ):
        raise ValueError(
            "Every prediction requires exactly one observation with source and quantization"
        )
    keys = []
    for (pool, source), g in d.groupby(["pool", "source"], sort=False):
        src = pd.read_csv(resolve_source(source, archive_root))
        for r in g.itertuples():
            if (
                "family_hparams_json" in src.columns
                and "source_row_index" in src.columns
            ):
                s = src[src.source_row_index == r.trial]
                if len(s) != 1:
                    raise ValueError(
                        f"Expected one architecture source row for {pool}/{r.trial}; found {len(s)}"
                    )
                h = json.loads(s.family_hparams_json.iloc[0])
                for k in ["input_dim", "timesteps"]:
                    h[k] = s["source__hparam__" + k].iloc[0]
            else:
                s = src[
                    (abs(src.energy_mj_per_inference - r.continuous_mj) < 1e-05)
                    & (abs(src.cadenced_energy_mj_per_window - r.scheduled_mj) < 1e-05)
                ]
                if len(s) != 1:
                    raise ValueError(
                        f"Expected one architecture source row for {pool}/{r.trial}; found {len(s)}"
                    )
                h = {
                    k: s["hparam__" + k].iloc[0]
                    for k in [
                        "nb_filters",
                        "kernel_size",
                        "dilations",
                        "use_skip_connections",
                        "norm_flag",
                        "input_dim",
                        "timesteps",
                    ]
                }
                h["dilations"] = json.loads(h["dilations"])
            spec = {
                k: [int(v) for v in h[k]] if k == "dilations" else int(h[k])
                for k in [
                    "nb_filters",
                    "kernel_size",
                    "dilations",
                    "use_skip_connections",
                    "norm_flag",
                    "input_dim",
                    "timesteps",
                ]
            }
            arch = json.dumps(spec, sort_keys=True, separators=(",", ":"))
            keys.append(
                dict(
                    pool=pool,
                    trial=r.trial,
                    architecture=arch,
                    arch_id=hashlib.sha256(arch.encode()).hexdigest()[:20],
                )
            )
    d = d.merge(pd.DataFrame(keys), on=["pool", "trial"], validate="one_to_one")
    return d


def predict(train, test, groupcols, tag, repeat, fold):
    """Fit weighted energy curves and predict architecture-disjoint rows.

    Parameters
    ----------
    train, test : pandas.DataFrame
        Architecture-disjoint fitting and evaluation rows.
    groupcols : list of str
        Operating-point grouping columns.
    tag : str
        Evaluation protocol.
    repeat, fold : int or str
        Split identifiers.

    Returns
    -------
    tuple
        Prediction frame list and fitted coefficient records.
    """
    out = []
    co = []
    if set(train.arch_id) & set(test.arch_id):
        raise ValueError("Training and test architecture identities overlap")
    for key, g in test.groupby(groupcols):
        key = key if isinstance(key, tuple) else (key,)
        t = train.copy()
        for c, v in zip(groupcols, key):
            t = t[t[c] == v]
        if t.arch_id.nunique() < 3 or t.latency_s.nunique() < 2:
            continue
        w = 1 / t.groupby("arch_id").arch_id.transform("size").to_numpy()
        x = t.latency_s.to_numpy()
        y = t.continuous_mj.to_numpy()
        a, s = np.linalg.lstsq(
            np.column_stack([np.ones(len(t)), x]) * np.sqrt(w[:, None]),
            y * np.sqrt(w),
            rcond=None,
        )[0]
        p = np.sum(w * x * y) / np.sum(w * x * x)
        da, ds = np.linalg.lstsq(
            np.column_stack([np.ones(len(t)), x]) * np.sqrt(w[:, None]),
            t.scheduled_mj.to_numpy() * np.sqrt(w),
            rcond=None,
        )[0]
        co.append(
            dict(
                protocol=tag,
                repeat=repeat,
                fold=fold,
                group=str(key),
                n_train=len(t),
                architectures_train=t.arch_id.nunique(),
                a_mj=a,
                s_mw=s,
                proportional_mw=p,
                direct_scheduled_intercept_mj=da,
                direct_scheduled_slope_mw=ds,
            )
        )
        z = g.copy()
        z["protocol"] = tag
        z["repeat"] = repeat
        z["fold"] = fold
        z["affine_cont"] = a + s * z.latency_s
        z["proportional_cont"] = p * z.latency_s
        z["scheduled_direct_affine_cont"] = np.nan
        z["calibrated_cont"] = z.continuous_hat_mj
        z["measured_continuous_cont"] = z.continuous_mj
        for m in MODELS:
            z[m + "_scheduled"] = (
                z[m + "_cont"] + (z.T_s - z.C_s) * z.wait_mw + z.offset_mj
            )
        z["scheduled_direct_affine_scheduled"] = (
            da + ds * z.latency_s if tag == "same_pool_architecture_holdout" else np.nan
        )
        out.append(z)
    return (out, co)


def summarize(p):
    """Aggregate row-weighted prediction errors.

    Parameters
    ----------
    p : pandas.DataFrame
        Held-out predictions.

    Returns
    -------
    pandas.DataFrame
        RMSE, MAE, and bias by protocol, model, and target.
    """
    rows = []
    for protocol, g in p.groupby("protocol"):
        for m in MODELS:
            for target, col in [("continuous", "cont"), ("scheduled", "scheduled")]:
                if (
                    m in ["measured_continuous", "scheduled_direct_affine"]
                    and target == "continuous"
                ):
                    continue
                if g[m + "_" + col].isna().all():
                    continue
                e = g[m + "_" + col] - g[target + "_mj"]
                rows.append(
                    dict(
                        protocol=protocol,
                        model=m,
                        target=target,
                        n_predictions=len(g),
                        n_rows=g[["pool", "trial"]].drop_duplicates().shape[0],
                        rmse_mj=np.sqrt(np.mean(e**2)),
                        mae_mj=np.mean(abs(e)),
                        bias_mj=e.mean(),
                    )
                )
    return pd.DataFrame(rows)


def selections(p, quality_ceilings=(0.65, 0.75, 1.0)):
    """Select minimum predicted scheduled energy within quality ceilings.

    Parameters
    ----------
    p : pandas.DataFrame
        Held-out predictions with saved quality and energy.
    quality_ceilings : sequence of float, optional
        Inclusive quality-error ceilings.

    Returns
    -------
    pandas.DataFrame
        Selected rows and measured-reference regrets.
    """
    rows = []
    for (protocol, rep, fold, pool), g in p.groupby(
        ["protocol", "repeat", "fold", "pool"]
    ):
        for ceiling in quality_ceilings:
            e = g[g.rmse <= ceiling].copy()
            if len(e) < 2:
                continue
            for m in MODELS:
                if e[m + "_scheduled"].isna().all():
                    continue
                s = e.sort_values(
                    [m + "_scheduled", "rmse", "study", "optuna_trial"]
                ).iloc[0]
                rows.append(
                    dict(
                        protocol=protocol,
                        repeat=rep,
                        fold=fold,
                        pool=pool,
                        ceiling=ceiling,
                        model=m,
                        n_eligible=len(e),
                        trial=s.trial,
                        arch_id=s.arch_id,
                        clock_mhz=s.clock_mhz,
                        quantization=s.quantization,
                        quality_rmse=s.rmse,
                        continuous_latency_ms=s.continuous_latency_ms,
                        measured_mj=s.scheduled_mj,
                        reference_mj=e.scheduled_mj.min(),
                        regret_mj=s.scheduled_mj - e.scheduled_mj.min(),
                        regret_pct=100 * (s.scheduled_mj / e.scheduled_mj.min() - 1),
                    )
                )
    return pd.DataFrame(
        rows,
        columns=[
            "protocol",
            "repeat",
            "fold",
            "pool",
            "ceiling",
            "model",
            "n_eligible",
            "trial",
            "arch_id",
            "clock_mhz",
            "quantization",
            "quality_rmse",
            "continuous_latency_ms",
            "measured_mj",
            "reference_mj",
            "regret_mj",
            "regret_pct",
        ],
    )


def run(
    predictions,
    observations,
    output_dir,
    archive_root=None,
    repeats=10,
    folds=5,
    quality_ceilings=(0.65, 0.75, 1.0),
    seed=20261103,
    threshold_by_period=None,
):
    """Run deterministic architecture holdout and pool-transfer analyses.

    Parameters
    ----------
    predictions, observations : path-like
        Frozen input tables.
    output_dir : path-like
        Destination for generated tables.
    archive_root : path-like, optional
        Relocated source log root.
    repeats, folds : int, optional
        Deterministic architecture split counts.
    quality_ceilings : sequence of float, optional
        Inclusive selection quality ceilings.
    seed : int, optional
        Architecture bootstrap random seed.
    threshold_by_period : dict, optional
        Historical descriptive thresholds keyed by period in milliseconds.

    Returns
    -------
    None
        Writes analysis CSVs, provenance, and report.
    """
    if repeats < 1 or folds < 2:
        raise ValueError("repeats must be positive and folds must be at least two")
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    d = load(predictions, observations, archive_root)
    d.to_csv(OUT / "architecture_manifest.csv", index=False)
    pred = []
    coeff = []
    for rep in range(repeats):
        fold_ids = d.arch_id.map(
            lambda a: int(hashlib.sha256(f"{rep}:{a}".encode()).hexdigest()[:12], 16)
            % folds
        )
        for fold in range(folds):
            out, co = predict(
                d[fold_ids != fold],
                d[fold_ids == fold],
                ["pool", "clock_mhz", "quantization"],
                "same_pool_architecture_holdout",
                rep,
                fold,
            )
            pred += out
            coeff += co
    for pool, g in d.groupby("pool"):
        train = d[(d.pool != pool) & ~d.arch_id.isin(g.arch_id)]
        out, co = predict(
            train,
            g,
            ["clock_mhz", "quantization"],
            "leave_pool_out_architecture_disjoint",
            0,
            pool,
        )
        pred += out
        coeff += co
    if not pred:
        raise ValueError(
            "No supported evaluation groups: each fit needs three training architectures and two distinct latencies"
        )
    p = pd.concat(pred, ignore_index=True)
    p.to_csv(OUT / "heldout_predictions.csv", index=False)
    pd.DataFrame(coeff).to_csv(OUT / "fit_coefficients.csv", index=False)
    group_rows = []
    for key, g in p.groupby(["protocol", "pool", "clock_mhz", "quantization"]):
        for m in MODELS:
            if m == "scheduled_direct_affine":
                continue
            e = g[m + "_cont"] - g.continuous_mj
            group_rows.append(
                dict(
                    protocol=key[0],
                    pool=key[1],
                    clock_mhz=key[2],
                    quantization=key[3],
                    model=m,
                    n_predictions=len(g),
                    n_rows=g.trial.nunique(),
                    rmse_mj=np.sqrt(np.mean(e**2)),
                    nrmse_pct=100 * np.sqrt(np.mean(e**2)) / g.continuous_mj.mean(),
                )
            )
    gm = pd.DataFrame(group_rows)
    gm.to_csv(OUT / "continuous_group_metrics.csv", index=False)
    quant = []
    for q, g in p.groupby("quantization"):
        t = summarize(g)
        t["quantization"] = q
        quant.append(t)
    pd.concat(quant).to_csv(OUT / "metrics_by_quantization.csv", index=False)
    covered = p[["protocol", "pool", "trial"]].drop_duplicates()
    missing = []
    for protocol in p.protocol.unique():
        z = d.merge(
            covered[covered.protocol == protocol][["pool", "trial"]],
            on=["pool", "trial"],
            how="left",
            indicator=True,
        )
        z = z[z._merge == "left_only"].copy()
        z["protocol"] = protocol
        missing.append(z.drop(columns="_merge"))
    pd.concat(missing).to_csv(OUT / "unsupported_rows.csv", index=False)
    metrics = summarize(p)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    sel = selections(p, quality_ceilings)
    sel.to_csv(OUT / "selections.csv", index=False)
    ss = sel.groupby(["protocol", "model"]).regret_mj.agg(
        ["count", "mean", "median", "max"]
    )
    ss.to_csv(OUT / "selection_summary.csv")
    rng = np.random.default_rng(seed)
    ci = []
    for protocol, g in p.groupby("protocol"):
        for target, suffix in [("continuous", "cont"), ("scheduled", "scheduled")]:
            for other in ["proportional", "calibrated"] + (
                ["measured_continuous", "affine"]
                if target == "scheduled"
                and protocol == "same_pool_architecture_holdout"
                else []
            ):
                lead = (
                    "scheduled_direct_affine"
                    if other in ["measured_continuous", "affine"]
                    else "affine"
                )
                diff = (g[lead + "_" + suffix] - g[target + "_mj"]) ** 2 - (
                    g[other + "_" + suffix] - g[target + "_mj"]
                ) ** 2
                z = diff.groupby(g.arch_id).mean().to_numpy()
                boot = np.array(
                    [rng.choice(z, len(z), replace=True).mean() for _ in range(2000)]
                )
                ci.append(
                    dict(
                        protocol=protocol,
                        target=target,
                        contrast=lead + "_minus_" + other,
                        n_architectures=len(z),
                        mean_architecture_mse_difference_mj2=z.mean(),
                        lo=np.quantile(boot, 0.025),
                        hi=np.quantile(boot, 0.975),
                    )
                )
    pd.DataFrame(ci).to_csv(OUT / "cluster_bootstrap.csv", index=False)
    direct_coeff = pd.DataFrame(coeff).query(
        "protocol == 'same_pool_architecture_holdout'"
    )
    small = gm[
        (gm.protocol == "same_pool_architecture_holdout")
        & gm.model.isin(["affine", "proportional"])
    ]
    wide = sel.pivot(
        index=["protocol", "repeat", "fold", "pool", "ceiling"],
        columns="model",
        values="measured_mj",
    )
    rr = []
    for protocol, g in wide.groupby(level="protocol"):
        delta = g.affine - g.measured_continuous
        rr.append(
            dict(
                protocol=protocol,
                comparisons=len(g),
                affine_lower=int((delta < -1e-09).sum()),
                same_energy=int((abs(delta) <= 1e-09).sum()),
                measured_continuous_lower=int((delta > 1e-09).sum()),
                maximum_measured_continuous_advantage_mj=delta.max(),
            )
        )
        if threshold_by_period is not None and "period_ms" in d:
            periods = d.groupby("pool").period_ms.agg(
                lambda values: values.iloc[0] if values.nunique() == 1 else np.nan
            )
            threshold = pd.Series(
                [threshold_by_period.get(periods.get(ix[3]), np.nan) for ix in g.index],
                index=g.index,
            )
            if threshold.notna().all():
                rr[-1]["measured_continuous_advantage_above_historical_threshold"] = (
                    int((delta > threshold).sum())
                )
    comparison = pd.DataFrame(rr)
    comparison.to_csv(OUT / "selection_pair_comparison.csv", index=False)
    qt = pd.concat(quant)
    report = (
        f"# CS1 held-out prediction\n\n{repeats} deterministic {folds}-fold architecture holdouts; equal architecture fitting weights. "
        + "Pool transfer excludes every target architecture. Frozen calibration and saved measured energies are retained.\n\n"
        + metrics.round(5).to_markdown(index=False)
        + "\n\nSelection regrets use the same quality-qualified saved bank; repeated folds and quality ceilings are correlated. "
        + "Bootstrap intervals are conditional on this bank, not device/session transfer uncertainty.\n"
    )
    write_provenance(OUT, predictions, observations, d, archive_root)
    (OUT / "REPORT.md").write_text(report)
    print(metrics.to_string(index=False))
    print(ss.to_string())
    print("Unique graphs:", d.arch_id.nunique())
    print("Output:", OUT)


def resolve_source(recorded, archive_root=None):
    """Resolve a recorded architecture log, optionally under a relocated archive.

    Parameters
    ----------
    recorded : str
        Saved source path.
    archive_root : path-like, optional
        Root containing the saved tree; matching relative suffixes are accepted.

    Returns
    -------
    pathlib.Path
        Existing unambiguous source file.
    """
    path = Path(recorded)
    if archive_root is None:
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    root = Path(archive_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Archive root is not a directory: {root}")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for index in range(len(parts)):
        candidate = root.joinpath(*parts[index:])
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(path.name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one source for {recorded} under {root}; found {len(matches)}"
        )
    return matches[0]


def write_provenance(output_dir, predictions, observations, data, archive_root):
    """Write SHA-256 hashes for every consumed input.

    Parameters
    ----------
    output_dir : pathlib.Path
        Destination directory.
    predictions, observations : path-like
        Explicit input CSV files.
    data : pandas.DataFrame
        Manifest containing saved source paths.
    archive_root : path-like or None
        Optional source relocation root.

    Returns
    -------
    None
        Writes ``input_provenance.json``.
    """
    paths = [Path(predictions), Path(observations)] + [
        resolve_source(s, archive_root) for s in data.source.unique()
    ]
    items = [
        {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in dict.fromkeys(paths)
    ]
    (output_dir / "input_provenance.json").write_text(
        json.dumps(items, indent=2) + "\n"
    )


def main():
    """Parse explicit input/output paths and run the analysis.

    Returns
    -------
    None
        Writes analysis tables and a concise report.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--quality-ceilings", type=float, nargs="+", default=[0.65, 0.75, 1.0]
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261103,
        help="Architecture-bootstrap RNG seed; graph splits use deterministic repeat hashes",
    )
    parser.add_argument(
        "--thresholds-json",
        type=Path,
        help="Optional JSON object mapping period_ms to historical threshold in mJ",
    )
    args = parser.parse_args()
    thresholds = (
        None
        if args.thresholds_json is None
        else {
            float(k): float(v)
            for k, v in json.loads(args.thresholds_json.read_text()).items()
        }
    )
    run(
        args.predictions,
        args.observations,
        args.output_dir,
        args.archive_root,
        args.repeats,
        args.folds,
        args.quality_ceilings,
        args.seed,
        thresholds,
    )


if __name__ == "__main__":
    main()
