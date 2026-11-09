# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Read saved NAS/replay tables for the submitted CS2 proxy-front figure.

This support module contains no plotting entry point or Matplotlib dependency.
It preserves the original front selection, latency classification, and outcome
accounting used by the paper renderer. It performs no hardware operations.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SUCCESS_ERROR_CODE = 1


@dataclass(frozen=True)
class FrontPoint:
    """One valid point in measured-energy/RMSE objective space.

    Attributes
    ----------
    payload_key : str
        Replay payload key identifying the candidate.
    energy_mj : float
        Measured energy in millijoules.
    rmse : float
        Root-mean-square error for the candidate.
    latency_feasible : bool | None
        Whether the candidate met the latency constraint.
    row : Mapping[str, Any]
        Source row associated with the parsed record.
    """

    payload_key: str
    energy_mj: float
    rmse: float
    latency_feasible: bool | None
    row: Mapping[str, Any]


@dataclass(frozen=True)
class PairInput:
    """One labeled replay comparison input.

    Attributes
    ----------
    label : str
        Human-readable label used in plot legends.
    crest_run_dir : pathlib.Path or None
        Measured-energy NAS run directory containing ``log_NAS_*.csv``.
    replay_path : pathlib.Path or None
        Replay output directory or explicit ``replay_results.csv`` path.
    placeholder : bool
        Whether this input is an empty placeholder panel.

    """

    label: str
    crest_run_dir: Path | None
    replay_path: Path | None
    placeholder: bool = False


def parse_float(value: Any) -> float | None:
    """Parse a finite float from a CSV cell.

    Parameters
    ----------
    value : Any
        CSV cell value to convert with ``float``.

    Returns
    -------
    float | None
        Finite parsed value, or ``None`` for missing, empty, nonnumeric,
        or nonfinite values.
    """
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries.

    Parameters
    ----------
    path : Path
        Path to the file used by the helper.

    Returns
    -------
    list[dict[str, str]]
        Loaded CSV rows.

    Raises
    ------
    ValueError
        If the CSV has no header or no data rows.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def first_float(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    """Return the first finite float from a sequence of possible column names.

    Parameters
    ----------
    row : Mapping[str, Any]
        CSV row to inspect for candidate numeric columns.
    names : Sequence[str]
        Candidate column names, checked in order.

    Returns
    -------
    float | None
        First finite parsed value, or ``None`` when no candidate column parses.
    """
    for name in names:
        parsed = parse_float(row.get(name))
        if parsed is not None:
            return parsed
    return None


def find_log_csv(run_dir: Path) -> Path:
    """Resolve the NAS log CSV inside a measured-energy run directory.

    Parameters
    ----------
    run_dir : Path
        Directory used for run artifacts.

    Returns
    -------
    Path
        Lexicographically first ``log_NAS_*.csv`` match, or ``trials.csv``
        when no matching log exists.

    Raises
    ------
    FileNotFoundError
        If neither a matching NAS log nor ``trials.csv`` exists.
    """
    candidates = sorted(run_dir.glob("log_NAS_*.csv"))
    if candidates:
        return candidates[0]
    trials_csv = run_dir / "trials.csv"
    if trials_csv.is_file():
        return trials_csv
    raise FileNotFoundError(f"No log_NAS_*.csv or trials.csv found in {run_dir}")


def resolve_replay_results(replay_path: Path) -> Path:
    """Resolve a replay directory or explicit replay CSV path.

    Parameters
    ----------
    replay_path : Path
        Path to the replay used by the helper.

    Returns
    -------
    Path
        The supplied file path, or ``replay_results.csv`` inside the supplied
        directory.

    Raises
    ------
    FileNotFoundError
        If the supplied path is not a file and contains no ``replay_results.csv``.
    """
    if replay_path.is_file():
        return replay_path
    candidate = replay_path / "replay_results.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No replay_results.csv found at {replay_path}")


def latency_feasible_from_row(
    row: Mapping[str, Any],
    *,
    latency_columns: Sequence[str],
    budget_columns: Sequence[str],
) -> bool | None:
    """Derive latency feasibility from common latency and budget columns.

    Parameters
    ----------
    row : Mapping[str, Any]
        CSV row containing latency and budget fields.
    latency_columns : Sequence[str]
        Column names for latency values.
    budget_columns : Sequence[str]
        Column names for budget values.

    Returns
    -------
    bool | None
        ``True`` when latency is within budget, ``False`` when it exceeds
        budget, or ``None`` when feasibility cannot be derived.
    """
    latency = first_float(row, latency_columns)
    budget = first_float(row, budget_columns)
    if latency is None or budget is None or latency <= 0.0 or budget <= 0.0:
        return None
    return latency <= budget


def load_crest_points(csv_path: Path) -> list[FrontPoint]:
    """Load valid measured-energy NAS points in CSV order.

    A nonempty ``state`` (or ``status`` when ``state`` is absent) must be
    complete, completed, success, or succeeded, ignoring case and whitespace.
    A finite ``error_code`` must truncate to ``SUCCESS_ERROR_CODE``; missing
    or unparseable codes are allowed. Energy must be positive and finite;
    RMSE must be finite.

    Numeric aliases are tried in order until a finite value is found: energy
    uses ``values_energy_mj_per_inference``, then ``energy_mj_per_inference``;
    RMSE uses ``values_rmse_total``, ``metric__rmse_total``, then ``rmse_total``.
    A finite nonpositive energy excludes the row without trying later aliases.
    Latency and budget use their ``values_`` columns before the plain columns;
    missing or nonpositive values produce unknown feasibility. Payload keys use
    the first nonempty ``number`` or ``trial_id``, then the zero-based row index.

    Parameters
    ----------
    csv_path : Path
        Saved NAS CSV containing the status, objective, and optional timing
        columns described above.

    Returns
    -------
    list[FrontPoint]
        Valid NAS points with parsed objectives, timing classification, and
        their original source rows.
    """
    points: list[FrontPoint] = []
    for index, row in enumerate(read_csv_rows(csv_path)):
        state = str(row.get("state", row.get("status", ""))).strip().lower()
        if state and state not in {"complete", "completed", "success", "succeeded"}:
            continue
        error_code = parse_float(row.get("error_code"))
        if error_code is not None and int(error_code) != SUCCESS_ERROR_CODE:
            continue
        energy = first_float(
            row, ("values_energy_mj_per_inference", "energy_mj_per_inference")
        )
        rmse = first_float(
            row, ("values_rmse_total", "metric__rmse_total", "rmse_total")
        )
        if energy is None or rmse is None or energy <= 0.0:
            continue
        points.append(
            FrontPoint(
                payload_key=str(row.get("number") or row.get("trial_id") or index),
                energy_mj=energy,
                rmse=rmse,
                latency_feasible=latency_feasible_from_row(
                    row,
                    latency_columns=("values_latency_ms", "latency_ms"),
                    budget_columns=("values_latency_budget_ms", "latency_budget_ms"),
                ),
                row=row,
            )
        )
    return points


def load_replay_points(csv_path: Path) -> list[FrontPoint]:
    """Load valid replayed proxy points in measured target-board space.

    Rows retain CSV order. ``replay_status`` must be completed, ignoring case
    and whitespace, and finite ``target__error_code`` must truncate to
    ``SUCCESS_ERROR_CODE``. Energy must be positive and finite; RMSE must be
    finite. All other rows are excluded.

    Numeric aliases are tried in order until a finite value is found: energy
    uses ``target__energy_mj_per_inference`` then
    ``target__cadenced_energy_mj_per_window``; RMSE uses
    ``source__metric__rmse_total``, ``source__rmse_total``, then
    ``target__metric__rmse_total``. A finite nonpositive energy excludes the row
    without trying the later alias. Latency uses ``target__latency_ms`` then
    ``target__cadenced_active_inference_latency_ms``; budget uses
    ``target__latency_budget_ms`` then ``target__cadenced_latency_budget_ms``.
    Missing or nonpositive latency/budget values produce unknown feasibility.
    Payload keys use the first nonempty ``replay_payload_key``, ``payload_key``,
    or ``source_row_index``, then the zero-based row index.

    Parameters
    ----------
    csv_path : Path
        Saved replay CSV containing the status, target objectives, source
        RMSE, and optional timing columns described above.

    Returns
    -------
    list[FrontPoint]
        Valid replay points with parsed objectives, timing classification,
        and their original source rows.
    """
    points: list[FrontPoint] = []
    for index, row in enumerate(read_csv_rows(csv_path)):
        if str(row.get("replay_status", "")).strip().lower() != "completed":
            continue
        error_code = parse_float(row.get("target__error_code"))
        if error_code is None or int(error_code) != SUCCESS_ERROR_CODE:
            continue
        energy = first_float(
            row,
            (
                "target__energy_mj_per_inference",
                "target__cadenced_energy_mj_per_window",
            ),
        )
        rmse = first_float(
            row,
            (
                "source__metric__rmse_total",
                "source__rmse_total",
                "target__metric__rmse_total",
            ),
        )
        if energy is None or rmse is None or energy <= 0.0:
            continue
        points.append(
            FrontPoint(
                payload_key=str(
                    row.get("replay_payload_key")
                    or row.get("payload_key")
                    or row.get("source_row_index")
                    or index
                ),
                energy_mj=energy,
                rmse=rmse,
                latency_feasible=latency_feasible_from_row(
                    row,
                    latency_columns=(
                        "target__latency_ms",
                        "target__cadenced_active_inference_latency_ms",
                    ),
                    budget_columns=(
                        "target__latency_budget_ms",
                        "target__cadenced_latency_budget_ms",
                    ),
                ),
                row=row,
            )
        )
    return points


def filter_latency_feasible(
    points: Sequence[FrontPoint], feasible_only: bool
) -> list[FrontPoint]:
    """Optionally retain only latency-feasible points.

    Parameters
    ----------
    points : Sequence[FrontPoint]
        Candidate points to filter.
    feasible_only : bool
        Whether to discard points without confirmed latency feasibility.

    Returns
    -------
    list[FrontPoint]
        Filtered points, preserving the original ordering.
    """
    if not feasible_only:
        return list(points)
    return [point for point in points if point.latency_feasible is True]


def pareto_front(points: Sequence[FrontPoint]) -> list[FrontPoint]:
    """Return non-dominated points minimizing energy and RMSE.

    Parameters
    ----------
    points : Sequence[FrontPoint]
        Candidate points to rank by energy and RMSE.

    Returns
    -------
    list[FrontPoint]
        Non-dominated points sorted by energy and RMSE.
    """
    front: list[FrontPoint] = []
    for point in points:
        dominated = any(
            other is not point
            and other.energy_mj <= point.energy_mj
            and other.rmse <= point.rmse
            and (other.energy_mj < point.energy_mj or other.rmse < point.rmse)
            for other in points
        )
        if not dominated:
            front.append(point)
    return sorted(front, key=lambda item: (item.energy_mj, item.rmse))


def energy_regret_rows(
    *,
    crest_front: Sequence[FrontPoint],
    proxy_front: Sequence[FrontPoint],
) -> list[dict[str, float | str]]:
    """Match replayed points to CREST front points and compute energy regret.

    Parameters
    ----------
    crest_front : Sequence[FrontPoint]
        Reference CREST Pareto-front points.
    proxy_front : Sequence[FrontPoint]
        Replayed Pareto-front points matched against the CREST front.

    Returns
    -------
    list[dict[str, float | str]]
        Per-match rows containing energy regret and identifying metadata.
    """
    rows: list[dict[str, float | str]] = []
    if not crest_front:
        return rows
    for proxy in proxy_front:
        candidates = [point for point in crest_front if point.rmse <= proxy.rmse]
        fallback = False
        if not candidates:
            candidates = list(crest_front)
            fallback = True
        matched = min(
            candidates,
            key=lambda point: (point.energy_mj, abs(point.rmse - proxy.rmse)),
        )
        if matched.energy_mj <= 0.0:
            continue
        energy_delta = proxy.energy_mj - matched.energy_mj
        energy_ratio = proxy.energy_mj / matched.energy_mj
        rows.append(
            {
                "proxy_key": proxy.payload_key,
                "crest_key": matched.payload_key,
                "proxy_rmse": proxy.rmse,
                "crest_rmse": matched.rmse,
                "proxy_energy_mj": proxy.energy_mj,
                "crest_energy_mj": matched.energy_mj,
                "energy_delta_mj": energy_delta,
                "energy_ratio": energy_ratio,
                "energy_percent_increase": (energy_ratio - 1.0) * 100.0,
                "fallback_nearest_rmse": str(fallback).lower(),
            }
        )
    return rows


@dataclass(frozen=True)
class LoadedPair:
    """Loaded points and fronts for one comparison pair.

    Attributes
    ----------
    label : str
        Human-readable label used in plot legends.
    crest_points : list[FrontPoint]
        Valid measured-energy NAS points.
    replay_points : list[FrontPoint]
        Valid replayed proxy points.
    crest_front : list[FrontPoint]
        Pareto front from ``crest_points``.
    replay_front : list[FrontPoint]
        Pareto front from ``replay_points``.
    regret_rows : list[dict[str, Any]]
        Energy-regret rows comparing replay front against CREST front.
    crest_front_outcomes : CandidateOutcomeSummary
        Timing outcome counts for the measured-energy NAS front.
    replay_outcomes : CandidateOutcomeSummary
        Raw replay outcome counts for all selected source-front candidates.
    placeholder : bool
        Whether this pair is an empty placeholder panel.

    """

    label: str
    crest_points: list[FrontPoint]
    replay_points: list[FrontPoint]
    crest_front: list[FrontPoint]
    replay_front: list[FrontPoint]
    regret_rows: list[dict[str, Any]]
    crest_front_outcomes: "CandidateOutcomeSummary"
    replay_outcomes: "CandidateOutcomeSummary"
    placeholder: bool = False


@dataclass(frozen=True)
class CandidateOutcomeSummary:
    """Timing and HIL outcome counts for Pareto-front candidates.

    Attributes
    ----------
    total_candidates : int
        Number of candidates included in the outcome summary.
    timing_feasible : int
        Measurements with latency within the configured budget.
    timing_infeasible : int
        Measurements with latency above the configured budget.
    hil_failed : int
        HIL measurements that reported an error before producing valid metrics.
    replay_failed : int
        Replay rows that did not complete before target measurement interpretation.
    timing_unknown : int
        Successful measurements without a usable latency or budget.

    """

    total_candidates: int
    timing_feasible: int
    timing_infeasible: int
    hil_failed: int
    replay_failed: int
    timing_unknown: int


def load_pair(pair: PairInput, *, feasible_only: bool) -> LoadedPair:
    """Load one replay comparison pair.

    Parameters
    ----------
    pair : PairInput
        Pair to load.
    feasible_only : bool
        Whether to keep only latency-feasible points.

    Returns
    -------
    LoadedPair
        Loaded points, Pareto fronts, and regret rows.

    Raises
    ------
    ValueError
        If a non-placeholder pair lacks a CREST run or replay path.
        CSV files with no header or no data rows also raise ``ValueError``.
    """
    if pair.placeholder:
        empty_outcomes = CandidateOutcomeSummary(
            total_candidates=0,
            timing_feasible=0,
            timing_infeasible=0,
            hil_failed=0,
            replay_failed=0,
            timing_unknown=0,
        )
        return LoadedPair(
            label=pair.label,
            crest_points=[],
            replay_points=[],
            crest_front=[],
            replay_front=[],
            regret_rows=[],
            crest_front_outcomes=empty_outcomes,
            replay_outcomes=empty_outcomes,
            placeholder=True,
        )
    if pair.crest_run_dir is None or pair.replay_path is None:
        raise ValueError(
            f"Panel paths are required for non-placeholder label: {pair.label}"
        )
    crest_csv = find_log_csv(pair.crest_run_dir)
    replay_csv = resolve_replay_results(pair.replay_path)
    crest_points = filter_latency_feasible(load_crest_points(crest_csv), feasible_only)
    replay_points = filter_latency_feasible(
        load_replay_points(replay_csv), feasible_only
    )
    crest_front = pareto_front(crest_points)
    replay_front = pareto_front(replay_points)
    regret_rows = energy_regret_rows(crest_front=crest_front, proxy_front=replay_front)
    crest_front_outcomes = summarize_point_outcomes(crest_front)
    replay_outcomes = summarize_replay_outcomes(replay_csv)
    return LoadedPair(
        label=pair.label,
        crest_points=crest_points,
        replay_points=replay_points,
        crest_front=crest_front,
        replay_front=replay_front,
        regret_rows=regret_rows,
        crest_front_outcomes=crest_front_outcomes,
        replay_outcomes=replay_outcomes,
        placeholder=False,
    )


def summarize_point_outcomes(points: Sequence[FrontPoint]) -> CandidateOutcomeSummary:
    """Summarize timing outcomes for already-valid measured points.

    Parameters
    ----------
    points : Sequence[FrontPoint]
        Valid measured points to summarize.

    Returns
    -------
    CandidateOutcomeSummary
        Timing feasibility counts for the provided points.
    """
    timing_feasible = 0
    timing_infeasible = 0
    timing_unknown = 0
    for point in points:
        if point.latency_feasible is True:
            timing_feasible += 1
        elif point.latency_feasible is False:
            timing_infeasible += 1
        else:
            timing_unknown += 1
    return CandidateOutcomeSummary(
        total_candidates=len(points),
        timing_feasible=timing_feasible,
        timing_infeasible=timing_infeasible,
        hil_failed=0,
        replay_failed=0,
        timing_unknown=timing_unknown,
    )


def summarize_replay_outcomes(csv_path: Path) -> CandidateOutcomeSummary:
    """Summarize HIL replay outcomes for all scheduled replay candidates.

    Parameters
    ----------
    csv_path : pathlib.Path
        Replay results CSV.

    Returns
    -------
    CandidateOutcomeSummary
        Counts for successful timing-feasible measurements, timing violations,
        HIL failures, replay failures, and rows without latency information.
    """
    timing_feasible = 0
    timing_infeasible = 0
    hil_failed = 0
    replay_failed = 0
    timing_unknown = 0
    rows = read_csv_rows(csv_path)
    for row in rows:
        if row.get("replay_status") != "completed":
            replay_failed += 1
            continue
        error_code = parse_float(row.get("target__error_code"))
        if error_code is None or int(error_code) != SUCCESS_ERROR_CODE:
            hil_failed += 1
            continue
        latency = parse_float(row.get("target__latency_ms"))
        budget = parse_float(row.get("target__latency_budget_ms"))
        if latency is None or budget is None or latency <= 0.0 or budget <= 0.0:
            timing_unknown += 1
        elif latency <= budget:
            timing_feasible += 1
        else:
            timing_infeasible += 1
    return CandidateOutcomeSummary(
        total_candidates=len(rows),
        timing_feasible=timing_feasible,
        timing_infeasible=timing_infeasible,
        hil_failed=hil_failed,
        replay_failed=replay_failed,
        timing_unknown=timing_unknown,
    )
