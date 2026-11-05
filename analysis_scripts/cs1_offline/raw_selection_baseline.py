"""Compare raw continuous-energy and latency rankings on saved held-out banks."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

MODELS = [
    "affine",
    "measured_continuous",
    "scheduled_direct_affine",
    "raw_continuous",
    "raw_latency",
]


def select(predictions, quality_ceilings=(0.65, 0.75, 1.0)):
    """Select candidates at the historical quality ceilings.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Saved held-out predictions, quality, and measured energy.
    quality_ceilings : sequence of float, optional
        Inclusive quality-error ceilings.

    Returns
    -------
    pandas.DataFrame
        Selected measured energy and regret for each ranking rule.
    """
    p = predictions.copy()
    p["raw_continuous_scheduled"] = p.continuous_mj
    p["raw_latency_scheduled"] = p.latency_s
    rows = []
    for (pr, rep, fold, pool), g in p.groupby(["protocol", "repeat", "fold", "pool"]):
        for c in quality_ceilings:
            e = g[g.rmse <= c]
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
                        pr=pr,
                        rep=rep,
                        fold=fold,
                        pool=pool,
                        c=c,
                        m=m,
                        sel=s.scheduled_mj,
                        regret=s.scheduled_mj - e.scheduled_mj.min(),
                    )
                )
    return pd.DataFrame(
        rows, columns=["pr", "rep", "fold", "pool", "c", "m", "sel", "regret"]
    )


def run(predictions, output_dir, quality_ceilings=(0.65, 0.75, 1.0)):
    """Write baseline selections, aggregate regrets, and paired comparisons.

    Parameters
    ----------
    predictions : path-like
        Explicit held-out prediction CSV path.
    output_dir : path-like
        Destination for generated analysis files.
    quality_ceilings : sequence of float, optional
        Inclusive quality-error ceilings.

    Returns
    -------
    None
        Writes tables, concise report, and input SHA-256 provenance.
    """
    predictions, output_dir = Path(predictions), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    r = select(pd.read_csv(predictions), quality_ceilings)
    r.to_csv(output_dir / "selections.csv", index=False)
    summary = r.groupby(["pr", "m"]).regret.agg(["count", "mean", "median", "max"])
    summary.to_csv(output_dir / "selection_summary.csv")
    w = r.pivot_table(
        index=["pr", "rep", "fold", "pool", "c"], columns="m", values="sel"
    )
    comparisons = []
    for pr, x in w.groupby(level=0):
        d = x.raw_continuous - x.measured_continuous
        comparisons.append(
            dict(
                protocol=pr,
                equal=int((d.abs() < 1e-9).sum()),
                measured_continuous_lower=int((d > 1e-9).sum()),
                raw_continuous_lower=int((d < -1e-9).sum()),
                mean_difference_mj=d.mean(),
            )
        )
    pd.DataFrame(comparisons).to_csv(output_dir / "paired_comparison.csv", index=False)
    r.groupby(["pr", "m", "pool"]).regret.mean().to_csv(
        output_dir / "regret_by_pool.csv"
    )
    (output_dir / "REPORT.md").write_text(
        "# CS1 raw selection baselines\n\n"
        f"Same saved held-out candidate banks and quality ceilings {tuple(quality_ceilings)} for all rankings. "
        "Regret is against the lowest saved scheduled energy in each eligible bank; folds and ceilings are correlated.\n\n"
        + summary.round(5).to_markdown()
        + "\n"
    )
    (output_dir / "input_provenance.json").write_text(
        json.dumps(
            [
                dict(
                    path=str(predictions.resolve()),
                    sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
                )
            ],
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string())
    print("Output:", output_dir)


def main():
    """Parse explicit file paths and run the comparison.

    Returns
    -------
    None
        Writes baseline results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quality-ceilings", type=float, nargs="+", default=[0.65, 0.75, 1.0]
    )
    args = parser.parse_args()
    run(args.predictions, args.output_dir, args.quality_ceilings)


if __name__ == "__main__":
    main()
