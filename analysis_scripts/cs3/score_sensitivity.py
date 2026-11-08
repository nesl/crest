#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Run score-sensitivity sweeps over CSV-derived NAS candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


DEFAULT_BUDGETS_MJ = (100.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1200.0)
DEFAULT_LAMBDAS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
DEFAULT_BASELINE_LAMBDA = 0.10
DEFAULT_BASELINE_BUDGET_MJ = 400.0
DEFAULT_STATUS_VALUES = ("complete", "completed", "success", "succeeded", "done", "ok")
DEFAULT_FEASIBLE_VALUES = ("true", "1", "yes", "y", "feasible", "complete", "completed", "ok")
OUTPUT_FIELDS = (
    "section",
    "run_label",
    "setting",
    "lambda",
    "energy_budget_mj",
    "trial_id",
    "candidate_id",
    "quality",
    "energy",
    "latency",
    "score",
    "matches_baseline",
    "architecture_fingerprint",
)


@dataclass(frozen=True)
class ColumnMapping:
    """Detected or configured CSV columns for one run.

    Parameters
    ----------
    csv_path:
        CSV file selected for the run.
    trial_id:
        Candidate identifier column, or ``None`` to use row indices.
    quality:
        Quality metric column used as macro-F1 in the preserved score formula.
    energy:
        Energy-per-inference column in millijoules.
    latency:
        Optional latency column.
    status:
        Optional status column.
    feasible:
        Optional feasibility column.
    source_score:
        Optional source score column. When detected or configured, rows must
        contain finite source scores to match the previous filtering behavior.
    architecture_columns:
        Columns used to build an architecture fingerprint.
    auto_detected:
        Whether the mapping came from heuristics rather than explicit columns.

    Attributes
    ----------
    csv_path : Path
        Path to the CSV artifact.
    trial_id : str | None
        Trial identifier column, when present.
    quality : str
        Quality value used for Pareto comparison.
    energy : str
        Energy column used for sensitivity scoring.
    latency : str | None
        Latency column used for optional feasibility filtering.
    status : str | None
        Status column used for optional run filtering.
    feasible : str | None
        Feasibility column used for optional run filtering.
    source_score : str | None
        Source score column used to mirror prior filtering behavior.
    architecture_columns : tuple[str, ...]
        Column names for architecture values.
    auto_detected : bool
        Whether the mapping came from heuristic column detection.
    """

    csv_path: Path
    trial_id: str | None
    quality: str
    energy: str
    latency: str | None
    status: str | None
    feasible: str | None
    source_score: str | None
    architecture_columns: tuple[str, ...]
    auto_detected: bool


@dataclass(frozen=True)
class RunInput:
    """One labeled input run.

    Parameters
    ----------
    label:
        Human-readable run label.
    path:
        Path to a CSV file or run directory containing top-level CSV logs.

    Attributes
    ----------
    label : str
        Display label used in reports and plots.
    path : Path
        CSV file or run directory containing log data.
    """

    label: str
    path: Path


@dataclass(frozen=True)
class ColumnOverrides:
    """Optional global column overrides.

    Parameters
    ----------
    quality:
        Explicit quality column.
    energy:
        Explicit energy column.
    trial_id:
        Explicit trial-id column.
    latency:
        Explicit latency column.
    status:
        Explicit status column.
    feasible:
        Explicit feasible column.
    source_score:
        Explicit source-score column.
    architecture_columns:
        Explicit architecture fingerprint columns.

    Attributes
    ----------
    quality : str | None
        Quality value used for Pareto comparison.
    energy : str | None
        Explicit energy column override.
    trial_id : str | None
        Explicit trial identifier column override.
    latency : str | None
        Explicit latency column override.
    status : str | None
        Explicit status column override.
    feasible : str | None
        Explicit feasibility column override.
    source_score : str | None
        Explicit source-score column override.
    architecture_columns : tuple[str, ...]
        Column names for architecture values.
    """

    quality: str | None = None
    energy: str | None = None
    trial_id: str | None = None
    latency: str | None = None
    status: str | None = None
    feasible: str | None = None
    source_score: str | None = None
    architecture_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensitivityConfig:
    """Configuration for one score-sensitivity run.

    Parameters
    ----------
    runs:
        Labeled CSV or run-directory inputs.
    output_dir:
        Directory where all generated artifacts are written.
    budgets_mj:
        Energy budgets for the budget sweep.
    lambdas:
        Penalty values for the lambda sweep.
    baseline_budget_mj:
        Baseline energy budget.
    baseline_lambda:
        Baseline energy penalty.
    status_values:
        Accepted status values when a status column is present.
    feasible_values:
        Accepted feasibility values when a feasible column is present.
    column_overrides:
        Optional global column overrides.
    command:
        Command line recorded in the manifest.

    Attributes
    ----------
    runs : tuple[RunInput, ...]
        Labeled CSV or run-directory inputs.
    output_dir : Path
        Directory where comparison artifacts are written.
    budgets_mj : tuple[float, ...]
        Energy budgets used for the budget sweep.
    lambdas : tuple[float, ...]
        Penalty values used for the lambda sweep.
    baseline_budget_mj : float
        Baseline energy budget in millijoules.
    baseline_lambda : float
        Baseline energy-penalty value.
    status_values : tuple[str, ...]
        Allowed values for status.
    feasible_values : tuple[str, ...]
        Allowed values for feasible.
    column_overrides : ColumnOverrides
        Global column overrides applied before auto-detection.
    command : tuple[str, ...]
        Command line recorded in the output manifest.
    """

    runs: tuple[RunInput, ...]
    output_dir: Path
    budgets_mj: tuple[float, ...] = DEFAULT_BUDGETS_MJ
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    baseline_budget_mj: float = DEFAULT_BASELINE_BUDGET_MJ
    baseline_lambda: float = DEFAULT_BASELINE_LAMBDA
    status_values: tuple[str, ...] = DEFAULT_STATUS_VALUES
    feasible_values: tuple[str, ...] = DEFAULT_FEASIBLE_VALUES
    column_overrides: ColumnOverrides = ColumnOverrides()
    command: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate scalar sweep settings.

        Raises
        ------
        ValueError
            If inputs, sweep values, or baseline values are invalid.
        """
        if not self.runs:
            raise ValueError("at least one --run LABEL=CSV_OR_RUN_DIR is required")
        labels = [run.label for run in self.runs]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"duplicate run label(s): {', '.join(duplicates)}")
        for budget in self.budgets_mj:
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("all --budget-mj values must be finite and positive")
        for lam in self.lambdas:
            if not math.isfinite(lam) or lam < 0:
                raise ValueError("all --lambda values must be finite and nonnegative")
        if not math.isfinite(self.baseline_budget_mj) or self.baseline_budget_mj <= 0:
            raise ValueError("--baseline-budget-mj must be finite and positive")
        if not math.isfinite(self.baseline_lambda) or self.baseline_lambda < 0:
            raise ValueError("--baseline-lambda must be finite and nonnegative")
        if not self.status_values:
            raise ValueError("--status-values must not be empty")
        if not self.feasible_values:
            raise ValueError("--feasible-values must not be empty")


def norm(name: str) -> str:
    """Normalize a column name for heuristic matching.

    Parameters
    ----------
    name:
        Raw column name.

    Returns
    -------
    str
        Lowercase alphanumeric/underscore column key.
    """
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def col_tokens(name: str) -> set[str]:
    """Split a normalized column name into nonempty tokens.

    Parameters
    ----------
    name:
        Raw column name.

    Returns
    -------
    set[str]
        Token set used by column scorers.
    """
    return {part for part in norm(name).split("_") if part}


def choose_col(columns: Sequence[str], scorer: Callable[[str], int]) -> str | None:
    """Choose the highest-scoring column under a heuristic scorer.

    Parameters
    ----------
    columns:
        Candidate column names in CSV order.
    scorer:
        Function returning a nonnegative score for a column.

    Returns
    -------
    str | None
        Selected column, or ``None`` when every score is zero.
    """
    ranked = sorted(
        ((scorer(col), idx, col) for idx, col in enumerate(columns)),
        key=lambda item: (-item[0], item[1]),
    )
    return ranked[0][2] if ranked and ranked[0][0] > 0 else None


def score_quality_col(col: str) -> int:
    """Score how likely a column is to contain macro-F1 quality.

    Parameters
    ----------
    col:
        Column name.

    Returns
    -------
    int
        Heuristic score. Zero means the column is not a quality candidate.
    """
    tokens = col_tokens(col)
    name = norm(col)
    score = 0
    if "macro" in tokens and ("f1" in tokens or "f_1" in name):
        score += 100
    else:
        return 0
    if "metric" in tokens:
        score += 20
    if "val" in tokens or "valid" in tokens or "validation" in tokens:
        score += 12
    if "user" in tokens and "attrs" in tokens:
        score += 8
    if "keras" in tokens:
        score -= 25
    if "per" in tokens and "class" in tokens:
        score -= 25
    return score


def score_energy_col(col: str) -> int:
    """Score how likely a column is to contain inference energy in mJ.

    Parameters
    ----------
    col:
        Column name.

    Returns
    -------
    int
        Heuristic score. Zero means the column is not an energy candidate.
    """
    tokens = col_tokens(col)
    if "energy" not in tokens:
        return 0
    score = 25
    if "mj" in tokens:
        score += 25
    if "inference" in tokens:
        score += 45
    if "per" in tokens:
        score += 8
    if "user" in tokens and "attrs" in tokens:
        score += 5
    if "trial" in tokens:
        score -= 35
    if "window" in tokens:
        score -= 35
    return score


def score_trial_col(col: str) -> int:
    """Score how likely a column is to contain trial identifiers.

    Parameters
    ----------
    col:
        Column name.

    Returns
    -------
    int
        Heuristic score. Zero means the column is not a trial-id candidate.
    """
    tokens = col_tokens(col)
    name = norm(col)
    if name in {"number", "trial", "trial_id", "trial_number", "trial_num"}:
        return 100
    if "trial" in tokens and ("id" in tokens or "number" in tokens or "num" in tokens):
        return 75
    return 0


def score_latency_col(col: str) -> int:
    """Score how likely a column is to contain latency in milliseconds.

    Parameters
    ----------
    col:
        Column name.

    Returns
    -------
    int
        Heuristic score. Zero means the column is not a latency candidate.
    """
    tokens = col_tokens(col)
    if "latency" not in tokens:
        return 0
    score = 20
    if "ms" in tokens:
        score += 35
    if norm(col) in {"latency_ms", "user_attrs_latency_ms"}:
        score += 80
    if "user" in tokens and "attrs" in tokens:
        score += 5
    if "budget" in tokens:
        score -= 90
    if "active" in tokens or "window" in tokens or "harness" in tokens:
        score -= 20
    return score


def score_status_col(col: str) -> int:
    """Score how likely a column is to contain trial status.

    Parameters
    ----------
    col:
        Column name.

    Returns
    -------
    int
        Heuristic score. Zero means the column is not a status candidate.
    """
    name = norm(col)
    tokens = col_tokens(col)
    if name == "state":
        return 100
    if "status" in tokens and "feasibility" not in tokens:
        return 60
    return 0


def score_feasible_col(col: str) -> int:
    """Score how likely a column is to contain feasibility flags.

    Parameters
    ----------
    col:
        Column name.

    Returns
    -------
    int
        Heuristic score. Zero means the column is not a feasibility candidate.
    """
    name = norm(col)
    tokens = col_tokens(col)
    if name in {"feasible", "user_attrs_feasible"}:
        return 100
    if "feasible" in tokens:
        return 80
    if "feasibility" in tokens and "status" in tokens:
        return 70
    return 0


def score_source_score_col(col: str) -> int:
    """Score how likely a column is to contain a source objective score.

    Parameters
    ----------
    col:
        Column name.

    Returns
    -------
    int
        Heuristic score. Zero means the column is not a source-score candidate.
    """
    name = norm(col)
    if name in {"value_score", "score", "objective_score"}:
        return 100
    if "score" in col_tokens(col):
        return 50
    return 0


def validate_columns(path: Path, columns: Sequence[str], required: Mapping[str, str | None]) -> None:
    """Validate that explicitly configured columns exist.

    Parameters
    ----------
    path:
        CSV path used in error messages.
    columns:
        Available CSV columns.
    required:
        Mapping from logical names to configured column names.

    Raises
    ------
    KeyError
        If a configured column is missing.
    """
    column_set = set(columns)
    missing = [
        f"{logical}={column!r}" for logical, column in required.items() if column and column not in column_set
    ]
    if missing:
        raise KeyError(f"{path}: missing configured column(s): {', '.join(missing)}")


def default_architecture_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Choose default architecture fingerprint columns.

    Parameters
    ----------
    columns:
        Available CSV columns.

    Returns
    -------
    tuple[str, ...]
        Architecture columns using the preserved ``params_`` priority, falling
        back to hparam columns only when no params columns are present.
    """
    params_cols = [col for col in columns if norm(col).startswith("params_")]
    hparam_cols = [
        col
        for col in columns
        if norm(col).startswith("user_attrs_hparam__") or norm(col).startswith("hparam__")
    ]
    return tuple(params_cols or hparam_cols)


def build_mapping(path: Path, df: pd.DataFrame, overrides: ColumnOverrides) -> ColumnMapping | None:
    """Build a mapping from explicit overrides plus auto-detected fallbacks.

    Parameters
    ----------
    path:
        CSV path.
    df:
        Loaded CSV frame.
    overrides:
        Global column overrides.

    Returns
    -------
    ColumnMapping | None
        Configured mapping, or ``None`` when required objective columns cannot
        be found.

    Raises
    ------
    KeyError
        If a configured column is missing.
    """
    columns = list(df.columns)
    validate_columns(
        path,
        columns,
        {
            "quality": overrides.quality,
            "energy": overrides.energy,
            "trial_id": overrides.trial_id,
            "latency": overrides.latency,
            "status": overrides.status,
            "feasible": overrides.feasible,
            "source_score": overrides.source_score,
        },
    )
    validate_columns(
        path,
        columns,
        {f"arch_col[{index}]": column for index, column in enumerate(overrides.architecture_columns)},
    )
    quality_col = overrides.quality or choose_col(columns, score_quality_col)
    energy_col = overrides.energy or choose_col(columns, score_energy_col)
    if quality_col is None or energy_col is None:
        return None
    # Optional columns can be overridden independently while objective columns
    # continue to use the original heuristics when not supplied.
    return ColumnMapping(
        csv_path=path,
        trial_id=overrides.trial_id or choose_col(columns, score_trial_col),
        quality=quality_col,
        energy=energy_col,
        latency=overrides.latency or choose_col(columns, score_latency_col),
        status=overrides.status or choose_col(columns, score_status_col),
        feasible=overrides.feasible or choose_col(columns, score_feasible_col),
        source_score=overrides.source_score or choose_col(columns, score_source_score_col),
        architecture_columns=overrides.architecture_columns or default_architecture_columns(columns),
        auto_detected=not any(
            (
                overrides.quality,
                overrides.energy,
                overrides.trial_id,
                overrides.latency,
                overrides.status,
                overrides.feasible,
                overrides.source_score,
                overrides.architecture_columns,
            )
        ),
    )


def score_log_candidate(
    path: Path,
    overrides: ColumnOverrides,
) -> tuple[int, pd.DataFrame, ColumnMapping | None]:
    """Score a CSV log candidate for directory discovery.

    Parameters
    ----------
    path:
        Candidate CSV path.
    overrides:
        Optional global column overrides.

    Returns
    -------
    tuple[int, pandas.DataFrame, ColumnMapping | None]
        Discovery score, loaded frame, and mapping when usable.
    """
    df = pd.read_csv(path)
    mapping = build_mapping(path, df, overrides)
    if mapping is None:
        return (0, df, None)
    completeness = int(df[mapping.quality].notna().sum()) + int(df[mapping.energy].notna().sum())
    richness = 0
    richness += 1000 if mapping.trial_id else 0
    richness += 500 if mapping.status else 0
    richness += 500 if mapping.feasible else 0
    richness += min(len(mapping.architecture_columns), 20) * 20
    return (completeness + richness, df, mapping)


def discover_csvs(path: Path) -> tuple[Path, ...]:
    """Discover top-level CSV files for a run input.

    Parameters
    ----------
    path:
        CSV file or run directory.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Deterministically ordered CSV paths.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the path is neither a file nor a directory, or no CSVs are present.
    """
    if not path.exists():
        raise FileNotFoundError(f"run input does not exist: {path}")
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise ValueError(f"run input is not a file or directory: {path}")
    csvs = tuple(
        sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.name.endswith(".csv")
            and not candidate.name.endswith(".lock")
        )
    )
    if not csvs:
        raise ValueError(f"no top-level CSV files found in {path}")
    return csvs


def load_run(run: RunInput, overrides: ColumnOverrides) -> tuple[pd.DataFrame, ColumnMapping, list[dict[str, Any]]]:
    """Load and map the best CSV for one run input.

    Parameters
    ----------
    run:
        Labeled run input.
    overrides:
        Optional global column overrides.

    Returns
    -------
    tuple[pandas.DataFrame, ColumnMapping, list[dict[str, Any]]]
        Loaded frame, selected column mapping, and discovery diagnostics.

    Raises
    ------
    RuntimeError
        If no usable CSV log is found.
    """
    diagnostics: list[dict[str, Any]] = []
    valid: list[tuple[int, pd.DataFrame, ColumnMapping]] = []
    for path in discover_csvs(run.path):
        score, df, mapping = score_log_candidate(path, overrides)
        diagnostics.append(
            {
                "csv_path": str(path),
                "discovery_score": score,
                "usable": mapping is not None,
            }
        )
        if mapping is not None:
            valid.append((score, df, mapping))
    if not valid:
        candidates = ", ".join(item["csv_path"] for item in diagnostics) or str(run.path)
        raise RuntimeError(f"{run.label}: no usable CSV log found among {candidates}")
    score, df, mapping = max(valid, key=lambda item: item[0])
    return df, mapping, diagnostics


def accepted_mask(series: pd.Series, accepted_values: Sequence[str]) -> pd.Series:
    """Build a case-insensitive mask for accepted string values.

    Parameters
    ----------
    series:
        Input column.
    accepted_values:
        Accepted values.

    Returns
    -------
    pandas.Series
        Boolean mask.
    """
    if pd.api.types.is_bool_dtype(series):
        bool_accepted = {value.strip().lower() for value in accepted_values} & {"true", "1", "yes", "y"}
        if bool_accepted:
            return series.fillna(False)
    accepted = {value.strip().lower() for value in accepted_values}
    text = series.astype(str).str.strip().str.lower()
    return text.isin(accepted)


def finite_numeric_mask(series: pd.Series) -> pd.Series:
    """Build a mask that accepts only finite numeric values.

    Parameters
    ----------
    series:
        Numeric input series.

    Returns
    -------
    pandas.Series
        Boolean mask that rejects missing, ``inf``, and ``-inf`` values.
    """
    return series.map(lambda value: bool(pd.notna(value) and math.isfinite(float(value))))


def clean_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalars to JSON-friendly Python scalars.

    Parameters
    ----------
    value:
        Scalar value.

    Returns
    -------
    Any
        Cleaned scalar, or ``None`` for missing values.
    """
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def compact_arch_fingerprint(row: pd.Series, arch_cols: Sequence[str]) -> str:
    """Build a compact JSON architecture fingerprint.

    Parameters
    ----------
    row:
        Candidate row.
    arch_cols:
        Architecture-defining columns.

    Returns
    -------
    str
        Deterministic compact JSON object.
    """
    values: dict[str, Any] = {}
    for col in arch_cols:
        key = col
        # Prefix stripping keeps fingerprints stable across Optuna export styles.
        for prefix in ("user_attrs_hparam__", "params_", "hparam__"):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        values[key] = clean_scalar(row[col])
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def prepare_candidates(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    status_values: Sequence[str],
    feasible_values: Sequence[str],
) -> pd.DataFrame:
    """Filter and normalize candidates for scoring.

    Parameters
    ----------
    df:
        Raw CSV frame.
    mapping:
        Column mapping.
    status_values:
        Accepted status values.
    feasible_values:
        Accepted feasibility values.

    Returns
    -------
    pandas.DataFrame
        Filtered frame with normalized helper columns.

    Raises
    ------
    ValueError
        If filtering removes every candidate.
    """
    work = df.copy()
    work["_quality"] = pd.to_numeric(work[mapping.quality], errors="coerce")
    work["_energy"] = pd.to_numeric(work[mapping.energy], errors="coerce")
    if mapping.latency:
        work["_latency"] = pd.to_numeric(work[mapping.latency], errors="coerce")
    else:
        work["_latency"] = math.nan
    if mapping.source_score:
        work["_source_score"] = pd.to_numeric(work[mapping.source_score], errors="coerce")
    else:
        work["_source_score"] = math.nan

    mask = finite_numeric_mask(work["_quality"]) & finite_numeric_mask(work["_energy"])
    if mapping.status:
        mask &= accepted_mask(work[mapping.status], status_values)
    if mapping.feasible:
        mask &= accepted_mask(work[mapping.feasible], feasible_values)
    if mapping.source_score:
        mask &= finite_numeric_mask(work["_source_score"])

    filtered = work.loc[mask].copy()
    if filtered.empty:
        raise ValueError(f"{mapping.csv_path}: no candidates remain after filtering")
    if mapping.trial_id:
        filtered["_trial_id"] = filtered[mapping.trial_id].map(clean_scalar).astype(str)
    else:
        filtered["_trial_id"] = filtered.index.astype(str)
    filtered["_candidate_id"] = filtered["_trial_id"]
    filtered["_arch_fingerprint"] = filtered.apply(
        lambda row: compact_arch_fingerprint(row, mapping.architecture_columns), axis=1
    )
    return filtered


def score_candidates(df: pd.DataFrame, lam: float, budget_mj: float) -> pd.DataFrame:
    """Apply the preserved score formula to candidates.

    Parameters
    ----------
    df:
        Prepared candidates.
    lam:
        Energy penalty lambda.
    budget_mj:
        Energy budget in millijoules.

    Returns
    -------
    pandas.DataFrame
        Scored candidates.
    """
    scored = df.copy()
    scored["_score"] = scored["_quality"] - lam * scored["_energy"] / budget_mj
    return scored


def select_candidate(df: pd.DataFrame, lam: float, budget_mj: float) -> pd.Series:
    """Select the best candidate under the preserved tie-break order.

    Parameters
    ----------
    df:
        Prepared candidates.
    lam:
        Energy penalty lambda.
    budget_mj:
        Energy budget in millijoules.

    Returns
    -------
    pandas.Series
        Selected candidate row.
    """
    scored = score_candidates(df, lam, budget_mj)
    scored = scored.sort_values(
        by=["_score", "_quality", "_energy", "_trial_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return scored.iloc[0]


def select_highest_quality(df: pd.DataFrame) -> pd.Series:
    """Select the highest-quality reference candidate.

    Parameters
    ----------
    df:
        Prepared candidates.

    Returns
    -------
    pandas.Series
        Candidate with max quality, min energy, then string trial-id tie-break.
    """
    selected = df.sort_values(
        by=["_quality", "_energy", "_trial_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0].copy()
    selected["_score"] = math.nan
    return selected


def select_lowest_energy(df: pd.DataFrame) -> pd.Series:
    """Select the lowest-energy reference candidate.

    Parameters
    ----------
    df:
        Prepared candidates.

    Returns
    -------
    pandas.Series
        Candidate with min energy, max quality, then string trial-id tie-break.
    """
    selected = df.sort_values(
        by=["_energy", "_quality", "_trial_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0].copy()
    selected["_score"] = math.nan
    return selected


def row_for_selection(
    run_label: str,
    section: str,
    setting: str,
    selected: pd.Series,
    baseline_id: str,
    lam: float | None = None,
    budget_mj: float | None = None,
) -> dict[str, Any]:
    """Project one selected candidate into the stable output schema.

    Parameters
    ----------
    run_label:
        Human-readable run label.
    section:
        Output section name.
    setting:
        Human-readable sweep setting.
    selected:
        Selected candidate row.
    baseline_id:
        Candidate ID selected by the baseline setting.
    lam:
        Lambda value for scored rows.
    budget_mj:
        Energy budget for scored rows.

    Returns
    -------
    dict[str, Any]
        Stable output row.
    """
    score = selected.get("_score", math.nan)
    return {
        "section": section,
        "run_label": run_label,
        "setting": setting,
        "lambda": lam,
        "energy_budget_mj": budget_mj,
        "trial_id": selected["_trial_id"],
        "candidate_id": selected["_candidate_id"],
        "quality": clean_scalar(selected["_quality"]),
        "energy": clean_scalar(selected["_energy"]),
        "latency": clean_scalar(selected["_latency"]),
        "score": clean_scalar(score),
        "matches_baseline": selected["_candidate_id"] == baseline_id,
        "architecture_fingerprint": selected["_arch_fingerprint"],
    }


def fmt_float(value: Any, digits: int = 4) -> str:
    """Format a float-like value for Markdown tables.

    Parameters
    ----------
    value:
        Value to format.
    digits:
        Decimal places.

    Returns
    -------
    str
        Formatted text or blank for missing values.
    """
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    """Render rows as a small Markdown table.

    Parameters
    ----------
    rows:
        Row dictionaries.
    columns:
        ``(label, key)`` table columns.

    Returns
    -------
    str
        Markdown table text.
    """
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt_float(value)
            elif isinstance(value, bool):
                value = "yes" if value else "no"
            elif value is None:
                value = ""
            cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body])


def mapping_dict(mapping: ColumnMapping) -> dict[str, Any]:
    """Convert a column mapping to a JSON-ready dictionary.

    Parameters
    ----------
    mapping:
        Column mapping.

    Returns
    -------
    dict[str, Any]
        JSON-ready mapping metadata.
    """
    return {
        "csv_path": str(mapping.csv_path),
        "trial_id": mapping.trial_id,
        "quality": mapping.quality,
        "energy": mapping.energy,
        "latency": mapping.latency,
        "status": mapping.status,
        "feasible": mapping.feasible,
        "source_score": mapping.source_score,
        "architecture_columns": list(mapping.architecture_columns),
        "auto_detected": mapping.auto_detected,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON file.

    Parameters
    ----------
    path:
        Output path.
    payload:
        JSON-compatible payload.

    Returns
    -------
    None
        The JSON file is written to disk.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(
    path: Path,
    config: SensitivityConfig,
    mappings: Mapping[str, ColumnMapping],
    feasible_counts: Mapping[str, int],
    baseline_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    budget_rows: Sequence[Mapping[str, Any]],
    lambda_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a human-readable Markdown summary.

    Parameters
    ----------
    path:
        Markdown output path.
    config:
        Sensitivity configuration.
    mappings:
        Run label to column mapping.
    feasible_counts:
        Run label to filtered candidate count.
    baseline_rows:
        Baseline selection rows.
    reference_rows:
        Reference selection rows.
    budget_rows:
        Budget sweep rows.
    lambda_rows:
        Lambda sweep rows.

    Returns
    -------
    None
        The Markdown file is written to disk.
    """
    lines = [
        "# Score Sensitivity Summary",
        "",
        "Score form: `quality - lambda * energy / energy_budget_mj`.",
        "",
        "## Detected Columns",
    ]
    for label, mapping in mappings.items():
        lines.extend(
            [
                f"- **{label}**: `{mapping.csv_path}`",
                f"  - trial id: `{mapping.trial_id}`",
                f"  - quality: `{mapping.quality}`",
                f"  - energy: `{mapping.energy}`",
                f"  - latency: `{mapping.latency}`",
                f"  - status: `{mapping.status}`",
                f"  - feasible: `{mapping.feasible}`",
                f"  - source score: `{mapping.source_score}`",
                f"  - architecture columns: `{', '.join(mapping.architecture_columns)}`",
            ]
        )
    lines.extend(["", "## Filtered Candidates"])
    for label, count in feasible_counts.items():
        lines.append(f"- **{label}**: {count}")

    selection_cols = [
        ("Run", "run_label"),
        ("Trial", "trial_id"),
        ("Quality", "quality"),
        ("Energy", "energy"),
        ("Latency", "latency"),
        ("Score", "score"),
    ]
    lines.extend(
        [
            "",
            "## Baseline Selection",
            "",
            f"Baseline uses `lambda = {config.baseline_lambda:g}` "
            f"and `energy_budget_mj = {config.baseline_budget_mj:g}`.",
            "",
            md_table(baseline_rows, selection_cols),
            "",
            "## Reference Candidates",
            "",
            md_table(
                reference_rows,
                [
                    ("Run", "run_label"),
                    ("Reference", "setting"),
                    ("Trial", "trial_id"),
                    ("Quality", "quality"),
                    ("Energy", "energy"),
                    ("Latency", "latency"),
                ],
            ),
        ]
    )

    sweep_cols = [
        ("Setting", "setting"),
        ("Trial", "trial_id"),
        ("Quality", "quality"),
        ("Energy", "energy"),
        ("Latency", "latency"),
        ("Score", "score"),
        ("Baseline?", "matches_baseline"),
    ]
    for label in mappings:
        lines.extend(
            [
                "",
                f"## Budget Sweep: {label}",
                "",
                md_table([row for row in budget_rows if row["run_label"] == label], sweep_cols),
            ]
        )
    for label in mappings:
        lines.extend(
            [
                "",
                f"## Lambda Sweep: {label}",
                "",
                md_table([row for row in lambda_rows if row["run_label"] == label], sweep_cols),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sensitivity(config: SensitivityConfig) -> dict[str, Any]:
    """Run score sensitivity and write output artifacts.

    Parameters
    ----------
    config:
        Sensitivity configuration.

    Returns
    -------
    dict[str, Any]
        Summary payload also written to ``summary.json``.

    Raises
    ------
    RuntimeError
        If existing validation or execution checks fail.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    mappings: dict[str, ColumnMapping] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {}
    candidates: dict[str, pd.DataFrame] = {}
    feasible_counts: dict[str, int] = {}

    for run in config.runs:
        df, mapping, run_diagnostics = load_run(run, config.column_overrides)
        mappings[run.label] = mapping
        diagnostics[run.label] = run_diagnostics
        filtered = prepare_candidates(df, mapping, config.status_values, config.feasible_values)
        candidates[run.label] = filtered
        feasible_counts[run.label] = len(filtered)

    baseline_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    baseline_ids: dict[str, str] = {}

    for label, df in candidates.items():
        baseline = select_candidate(df, config.baseline_lambda, config.baseline_budget_mj)
        baseline_ids[label] = baseline["_candidate_id"]
        baseline_row = row_for_selection(
            label,
            "baseline",
            f"lambda={config.baseline_lambda:g}, budget={config.baseline_budget_mj:g} mJ",
            baseline,
            baseline_ids[label],
            config.baseline_lambda,
            config.baseline_budget_mj,
        )
        baseline_rows.append(baseline_row)
        all_rows.append(baseline_row)

        for setting, selected in [
            ("highest quality", select_highest_quality(df)),
            ("lowest energy", select_lowest_energy(df)),
        ]:
            row = row_for_selection(label, "reference", setting, selected, baseline_ids[label])
            reference_rows.append(row)
            all_rows.append(row)

    for label, df in candidates.items():
        for budget in config.budgets_mj:
            selected = select_candidate(df, config.baseline_lambda, budget)
            row = row_for_selection(
                label,
                "budget_sweep",
                f"budget={budget:g} mJ",
                selected,
                baseline_ids[label],
                config.baseline_lambda,
                budget,
            )
            budget_rows.append(row)
            all_rows.append(row)
        for lam in config.lambdas:
            selected = select_candidate(df, lam, config.baseline_budget_mj)
            row = row_for_selection(
                label,
                "lambda_sweep",
                f"lambda={lam:g}",
                selected,
                baseline_ids[label],
                lam,
                config.baseline_budget_mj,
            )
            lambda_rows.append(row)
            all_rows.append(row)

    scored = [row["score"] for row in all_rows if row["section"] != "reference"]
    if not scored or not all(score is not None and math.isfinite(float(score)) for score in scored):
        raise RuntimeError("non-finite sensitivity score generated")

    selections_df = pd.DataFrame(all_rows, columns=OUTPUT_FIELDS)
    selections_df.to_csv(config.output_dir / "selections.csv", index=False)

    run_metadata = {
        label: {
            "mapping": mapping_dict(mappings[label]),
            "input_path": str(next(run.path for run in config.runs if run.label == label)),
            "candidate_count": feasible_counts[label],
            "baseline_trial_id": next(row["trial_id"] for row in baseline_rows if row["run_label"] == label),
            "baseline_candidate_id": baseline_ids[label],
            "csv_discovery": diagnostics[label],
        }
        for label in mappings
    }
    manifest = {
        "command": list(config.command),
        "formula": "quality - lambda * energy / energy_budget_mj",
        "baseline": {
            "lambda": config.baseline_lambda,
            "energy_budget_mj": config.baseline_budget_mj,
        },
        "budgets_mj": list(config.budgets_mj),
        "lambdas": list(config.lambdas),
        "status_values": list(config.status_values),
        "feasible_values": list(config.feasible_values),
        "output_fields": list(OUTPUT_FIELDS),
        "runs": run_metadata,
    }
    summary = {
        "formula": manifest["formula"],
        "baseline": manifest["baseline"],
        "runs": run_metadata,
        "baseline_selections": baseline_rows,
        "reference_selections": reference_rows,
        "budget_sweep": budget_rows,
        "lambda_sweep": lambda_rows,
        "selections": all_rows,
    }
    write_json(config.output_dir / "manifest.json", manifest)
    write_json(config.output_dir / "summary.json", summary)
    write_markdown(
        config.output_dir / "summary.md",
        config,
        mappings,
        feasible_counts,
        baseline_rows,
        reference_rows,
        budget_rows,
        lambda_rows,
    )
    return summary


def parse_run_spec(spec: str) -> RunInput:
    """Parse ``LABEL=CSV_OR_RUN_DIR`` CLI input.

    Parameters
    ----------
    spec:
        Raw ``--run`` argument.

    Returns
    -------
    RunInput
        Parsed run input.

    Raises
    ------
    argparse.ArgumentTypeError
        If the argument is malformed.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--run must use LABEL=CSV_OR_RUN_DIR")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("--run requires nonempty label and path")
    return RunInput(label=label, path=Path(raw_path))


def parse_float_list(values: Sequence[str] | None, default: Sequence[float], label: str) -> tuple[float, ...]:
    """Parse repeatable float CLI values.

    Parameters
    ----------
    values:
        Raw CLI values.
    default:
        Values used when CLI values are absent.
    label:
        Option label for error messages.

    Returns
    -------
    tuple[float, ...]
        Parsed float values.

    Raises
    ------
    ValueError
        If a value cannot be parsed as a float.
    """
    if values is None:
        return tuple(default)
    parsed: list[float] = []
    for raw in values:
        try:
            parsed.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"{label} value is not numeric: {raw!r}") from exc
    return tuple(parsed)


def parse_value_list(values: Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    """Parse comma- or space-separated accepted-value CLI input.

    Parameters
    ----------
    values:
        Raw values.
    default:
        Values used when CLI values are absent.

    Returns
    -------
    tuple[str, ...]
        Normalized nonempty accepted values.
    """
    if values is None:
        return tuple(default)
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(parsed)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Run score-sensitivity sweeps over CSV-derived NAS candidates.",
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        metavar="LABEL=CSV_OR_RUN_DIR",
        help="Labeled CSV file or run directory. Repeat for multiple runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated artifacts.")
    parser.add_argument("--quality-col", help="Quality metric column. Requires --energy-col.")
    parser.add_argument("--energy-col", help="Energy-per-inference column in mJ. Requires --quality-col.")
    parser.add_argument("--trial-id-col", help="Candidate trial-id column.")
    parser.add_argument("--latency-col", help="Latency column.")
    parser.add_argument("--status-col", help="Status column.")
    parser.add_argument("--feasible-col", help="Feasibility column.")
    parser.add_argument("--source-score-col", help="Source score column used as a finite-value filter.")
    parser.add_argument("--arch-col", action="append", default=[], help="Architecture fingerprint column.")
    parser.add_argument("--budget-mj", action="append", help="Budget-sweep energy budget in mJ. Repeatable.")
    parser.add_argument("--lambda", dest="lambdas", action="append", help="Lambda-sweep penalty. Repeatable.")
    parser.add_argument(
        "--baseline-budget-mj",
        type=float,
        default=DEFAULT_BASELINE_BUDGET_MJ,
        help="Baseline energy budget in mJ.",
    )
    parser.add_argument(
        "--baseline-lambda",
        type=float,
        default=DEFAULT_BASELINE_LAMBDA,
        help="Baseline lambda.",
    )
    parser.add_argument(
        "--status-values",
        nargs="+",
        help="Accepted status values; values may also be comma-separated.",
    )
    parser.add_argument(
        "--feasible-values",
        nargs="+",
        help="Accepted feasible values; values may also be comma-separated.",
    )
    return parser


def config_from_args(args: argparse.Namespace, argv: Sequence[str]) -> SensitivityConfig:
    """Convert parsed arguments into a validated configuration.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    argv:
        Original command arguments for manifest recording.

    Returns
    -------
    SensitivityConfig
        Validated configuration.
    """
    overrides = ColumnOverrides(
        quality=args.quality_col,
        energy=args.energy_col,
        trial_id=args.trial_id_col,
        latency=args.latency_col,
        status=args.status_col,
        feasible=args.feasible_col,
        source_score=args.source_score_col,
        architecture_columns=tuple(args.arch_col),
    )
    return SensitivityConfig(
        runs=tuple(args.run),
        output_dir=args.output_dir,
        budgets_mj=parse_float_list(args.budget_mj, DEFAULT_BUDGETS_MJ, "--budget-mj"),
        lambdas=parse_float_list(args.lambdas, DEFAULT_LAMBDAS, "--lambda"),
        baseline_budget_mj=args.baseline_budget_mj,
        baseline_lambda=args.baseline_lambda,
        status_values=parse_value_list(args.status_values, DEFAULT_STATUS_VALUES),
        feasible_values=parse_value_list(args.feasible_values, DEFAULT_FEASIBLE_VALUES),
        column_overrides=overrides,
        command=tuple(argv),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Parameters
    ----------
    argv:
        Command arguments excluding the program name, or ``None`` for
        ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_arg_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser.parse_args(raw_argv)
        config = config_from_args(args, raw_argv)
        summary = run_sensitivity(config)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")

    print("Baseline selections:")
    for row in summary["baseline_selections"]:
        print(
            f"{row['run_label']}: trial {row['trial_id']}, "
            f"quality={float(row['quality']):.6f}, energy={float(row['energy']):.6f}, "
            f"score={float(row['score']):.6f}"
        )
    print(f"Wrote {config.output_dir / 'manifest.json'}")
    print(f"Wrote {config.output_dir / 'summary.json'}")
    print(f"Wrote {config.output_dir / 'selections.csv'}")
    print(f"Wrote {config.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
