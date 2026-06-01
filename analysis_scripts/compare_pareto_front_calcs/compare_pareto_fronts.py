#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Compare two CSV-derived Pareto fronts with explicit matching semantics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIRECTIONS = {"minimize", "maximize"}
MATCH_RULES = {"nearest-quality", "equal-or-better-quality"}
REDUCTION_DENOMINATORS = {"source", "target", "both"}
REDUCTION_DIRECTIONS = {"target-vs-source", "source-vs-target"}
FILTER_OPS = {"lt", "le", "eq", "ge", "gt", "ne"}
DEFAULT_STATUS_VALUES = ("COMPLETE", "completed", "success", "succeeded", "1", "true")


@dataclass(frozen=True)
class NumericFilter:
    """One numeric row filter.

    Parameters
    ----------
    column:
        CSV column to inspect.
    op:
        Comparison operation: ``lt``, ``le``, ``eq``, ``ge``, ``gt``, or ``ne``.
    value:
        Numeric threshold.

    Attributes
    ----------
    column : str
        Column inspected by the row filter.
    op : str
        Comparison operator used by the row filter.
    value : float
        Numeric threshold used by the row filter.
    """

    column: str
    op: str
    value: float

    def __post_init__(self) -> None:
        """Validate the numeric filter.

        Raises
        ------
        ValueError
            If the operation or threshold is invalid.
        """
        if self.op not in FILTER_OPS:
            raise ValueError(f"numeric filter op must be one of {sorted(FILTER_OPS)}")
        if not math.isfinite(self.value):
            raise ValueError("numeric filter value must be finite")


@dataclass(frozen=True)
class FrontCompareConfig:
    """Configuration for one Pareto-front comparison.

    Parameters
    ----------
    source_csv:
        CSV containing source rows.
    target_csv:
        CSV containing target rows.
    output_dir:
        Directory for reproducibility artifacts.
    source_quality_col:
        Source quality/objective column.
    source_cost_col:
        Source cost/energy column.
    target_quality_col:
        Target quality/objective column.
    target_cost_col:
        Target cost/energy column.
    quality_direction:
        Optimization direction for quality.
    cost_direction:
        Optimization direction for cost.
    match_rule:
        Rule used to pair each source-front point with a target-front point.
    reduction_denominator:
        Denominator side to emphasize in summaries.
    reduction_direction:
        Orientation for positive reduction values.
    source_label:
        Human-readable source label.
    target_label:
        Human-readable target label.
    source_id_col:
        Optional source row identifier column.
    target_id_col:
        Optional target row identifier column.
    source_status_col:
        Optional source status column.
    target_status_col:
        Optional target status column.
    source_status_values:
        Allowed source status values when ``source_status_col`` is set.
    target_status_values:
        Allowed target status values when ``target_status_col`` is set.
    source_filters:
        Additional numeric filters applied to source rows.
    target_filters:
        Additional numeric filters applied to target rows.
    allow_nonpositive_cost:
        Whether nonpositive minimized costs are allowed.
    sentinel_abs_threshold:
        Absolute-value threshold for sentinel objective values.

    Attributes
    ----------
    source_csv : Path
        CSV file containing source Pareto points.
    target_csv : Path
        CSV file containing target Pareto points.
    output_dir : Path
        Directory where comparison artifacts are written.
    source_quality_col : str
        Quality column name in the source CSV.
    source_cost_col : str
        Cost column name in the source CSV.
    target_quality_col : str
        Quality column name in the target CSV.
    target_cost_col : str
        Cost column name in the target CSV.
    quality_direction : str
        Optimization direction for quality values.
    cost_direction : str
        Optimization direction for cost values.
    match_rule : str
        Rule used to match source rows to target rows.
    reduction_denominator : str
        Denominator used when computing reduction fractions.
    reduction_direction : str
        Direction used when interpreting reduction values.
    source_label : str
        Display label for the source dataset.
    target_label : str
        Display label for the target dataset.
    source_id_col : str | None
        Identifier column name in the source CSV.
    target_id_col : str | None
        Identifier column name in the target CSV.
    source_status_col : str | None
        Status column name in the source CSV.
    target_status_col : str | None
        Status column name in the target CSV.
    source_status_values : tuple[str, ...]
        Allowed status values for source rows.
    target_status_values : tuple[str, ...]
        Allowed status values for target rows.
    source_filters : tuple[NumericFilter, ...]
        Filters applied to source rows before comparison.
    target_filters : tuple[NumericFilter, ...]
        Filters applied to target rows before comparison.
    allow_nonpositive_cost : bool
        Whether non-positive cost values are accepted during comparison.
    sentinel_abs_threshold : float
        Absolute threshold used to identify sentinel values.
    """

    source_csv: Path
    target_csv: Path
    output_dir: Path
    source_quality_col: str
    source_cost_col: str
    target_quality_col: str
    target_cost_col: str
    quality_direction: str = "minimize"
    cost_direction: str = "minimize"
    match_rule: str = "nearest-quality"
    reduction_denominator: str = "source"
    reduction_direction: str = "target-vs-source"
    source_label: str = "source"
    target_label: str = "target"
    source_id_col: str | None = None
    target_id_col: str | None = None
    source_status_col: str | None = None
    target_status_col: str | None = None
    source_status_values: tuple[str, ...] = DEFAULT_STATUS_VALUES
    target_status_values: tuple[str, ...] = DEFAULT_STATUS_VALUES
    source_filters: tuple[NumericFilter, ...] = ()
    target_filters: tuple[NumericFilter, ...] = ()
    allow_nonpositive_cost: bool = False
    sentinel_abs_threshold: float = 1.0e11

    def __post_init__(self) -> None:
        """Validate comparison configuration values.

        Raises
        ------
        ValueError
            If an enum-like option or threshold is invalid.
        """
        if self.quality_direction not in DIRECTIONS:
            raise ValueError(f"quality_direction must be one of {sorted(DIRECTIONS)}")
        if self.cost_direction not in DIRECTIONS:
            raise ValueError(f"cost_direction must be one of {sorted(DIRECTIONS)}")
        if self.match_rule not in MATCH_RULES:
            raise ValueError(f"match_rule must be one of {sorted(MATCH_RULES)}")
        if self.reduction_denominator not in REDUCTION_DENOMINATORS:
            raise ValueError(f"reduction_denominator must be one of {sorted(REDUCTION_DENOMINATORS)}")
        if self.reduction_direction not in REDUCTION_DIRECTIONS:
            raise ValueError(f"reduction_direction must be one of {sorted(REDUCTION_DIRECTIONS)}")
        if not math.isfinite(self.sentinel_abs_threshold) or self.sentinel_abs_threshold <= 0:
            raise ValueError("sentinel_abs_threshold must be finite and positive")


@dataclass(frozen=True)
class FrontPoint:
    """One valid CSV row projected into quality-cost space.

    Parameters
    ----------
    row_index:
        Zero-based row index in the source CSV body.
    point_id:
        Stable row identifier from the configured ID column or row index.
    quality:
        Parsed quality/objective value.
    cost:
        Parsed cost/energy value.
    label:
        Source or target label.

    Attributes
    ----------
    row_index : int
        Index of the source row in the input CSV.
    point_id : str
        Identifier for the Pareto point.
    quality : float
        Quality value used for Pareto comparison.
    cost : float
        Cost value used for Pareto comparison.
    label : str
        Display label used in reports and plots.
    """

    row_index: int
    point_id: str
    quality: float
    cost: float
    label: str


@dataclass(frozen=True)
class MatchRow:
    """One source-front to target-front match.

    Parameters
    ----------
    source:
        Source front point.
    target:
        Matched target front point.
    match_rule_applied:
        Matching rule actually used for this row.
    fallback_used:
        Whether an equal-or-better match fell back to nearest quality.
    reduction_source_fraction:
        Reduction fraction using source cost as denominator.
    reduction_target_fraction:
        Reduction fraction using target cost as denominator.
    oriented_cost_delta:
        Cost delta under the configured reduction orientation.

    Attributes
    ----------
    source : FrontPoint
        Source point in the matched pair.
    target : FrontPoint
        Target point in the matched pair.
    match_rule_applied : str
        Match rule used for this source-target pair.
    fallback_used : bool
        Whether fallback matching was used for the pair.
    reduction_source_fraction : float
        Reduction fraction computed from the source point.
    reduction_target_fraction : float
        Reduction fraction computed from the target point.
    oriented_cost_delta : float
        Cost delta after applying the configured optimization direction.
    """

    source: FrontPoint
    target: FrontPoint
    match_rule_applied: str
    fallback_used: bool
    reduction_source_fraction: float
    reduction_target_fraction: float
    oriented_cost_delta: float


def parse_float_cell(value: str, *, path: Path, row_index: int, column: str) -> float | None:
    """Parse a CSV cell as a finite float or blank invalid value.

    Parameters
    ----------
    value:
        Raw CSV cell.
    path:
        CSV path used in error messages.
    row_index:
        Zero-based body row index.
    column:
        Column name used in error messages.

    Returns
    -------
    float | None
        Parsed finite value, or ``None`` for blank/null/non-finite cells.

    Raises
    ------
    ValueError
        If a nonblank cell cannot be parsed as a float.
    """
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"{path}: row {row_index + 2} column {column!r} is not numeric: {text!r}") from exc
    if not math.isfinite(parsed):
        return None
    return parsed


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file and validate that it has a header.

    Parameters
    ----------
    path:
        CSV path to read.

    Returns
    -------
    tuple[list[str], list[dict[str, str]]]
        Header fields and row dictionaries.

    Raises
    ------
    ValueError
        If the CSV has no header.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def require_columns(path: Path, fields: Sequence[str], columns: Iterable[str | None]) -> None:
    """Validate that all requested columns exist.

    Parameters
    ----------
    path:
        CSV path used in error messages.
    fields:
        Available CSV fields.
    columns:
        Requested columns; ``None`` values are ignored.

    Raises
    ------
    KeyError
        If a requested column is absent.
    """
    available = set(fields)
    missing = [column for column in columns if column and column not in available]
    if missing:
        raise KeyError(f"{path} is missing required column(s): {', '.join(missing)}")


def status_allowed(row: Mapping[str, str], status_col: str | None, allowed_values: Sequence[str]) -> bool:
    """Return whether a row passes an optional status filter.

    Parameters
    ----------
    row:
        CSV row.
    status_col:
        Optional status column name.
    allowed_values:
        Allowed raw status values.

    Returns
    -------
    bool
        ``True`` when the row passes status filtering.
    """
    if status_col is None:
        return True
    return str(row.get(status_col, "")).strip() in set(allowed_values)


def apply_filter(value: float, numeric_filter: NumericFilter) -> bool:
    """Return whether a value passes one numeric filter.

    Parameters
    ----------
    value:
        Parsed numeric value.
    numeric_filter:
        Filter to apply.

    Returns
    -------
    bool
        ``True`` when the value satisfies the filter.
    """
    if numeric_filter.op == "lt":
        return value < numeric_filter.value
    if numeric_filter.op == "le":
        return value <= numeric_filter.value
    if numeric_filter.op == "eq":
        return value == numeric_filter.value
    if numeric_filter.op == "ge":
        return value >= numeric_filter.value
    if numeric_filter.op == "gt":
        return value > numeric_filter.value
    return value != numeric_filter.value


def filters_allowed(row: Mapping[str, str], filters: Sequence[NumericFilter], *, path: Path, row_index: int) -> bool:
    """Return whether a row passes all numeric filters.

    Parameters
    ----------
    row:
        CSV row.
    filters:
        Numeric filters to apply.
    path:
        CSV path used in error messages.
    row_index:
        Zero-based body row index.

    Returns
    -------
    bool
        ``True`` when every filter passes.
    """
    for numeric_filter in filters:
        value = parse_float_cell(row.get(numeric_filter.column, ""), path=path, row_index=row_index, column=numeric_filter.column)
        if value is None or not apply_filter(value, numeric_filter):
            return False
    return True


def valid_objectives(
    quality: float | None,
    cost: float | None,
    *,
    config: FrontCompareConfig,
) -> bool:
    """Return whether parsed objective values are usable for comparison.

    Parameters
    ----------
    quality:
        Parsed quality value.
    cost:
        Parsed cost value.
    config:
        Comparison config containing direction and sentinel rules.

    Returns
    -------
    bool
        ``True`` when both values are finite, non-sentinel objectives.
    """
    if quality is None or cost is None:
        return False
    if abs(quality) >= config.sentinel_abs_threshold or abs(cost) >= config.sentinel_abs_threshold:
        return False
    if config.cost_direction == "minimize" and not config.allow_nonpositive_cost and cost <= 0.0:
        return False
    return True


def load_points(
    *,
    path: Path,
    quality_col: str,
    cost_col: str,
    id_col: str | None,
    status_col: str | None,
    status_values: Sequence[str],
    filters: Sequence[NumericFilter],
    label: str,
    config: FrontCompareConfig,
) -> tuple[list[FrontPoint], dict[str, int]]:
    """Load valid comparison points from one CSV.

    Parameters
    ----------
    path:
        CSV path.
    quality_col:
        Quality/objective column.
    cost_col:
        Cost/energy column.
    id_col:
        Optional identifier column.
    status_col:
        Optional status filter column.
    status_values:
        Allowed status values.
    filters:
        Additional numeric row filters.
    label:
        Human-readable side label.
    config:
        Comparison configuration.

    Returns
    -------
    tuple[list[FrontPoint], dict[str, int]]
        Valid points and row-count metadata.
    """
    fields, rows = read_csv_rows(path)
    require_columns(path, fields, [quality_col, cost_col, id_col, status_col, *(numeric_filter.column for numeric_filter in filters)])
    valid: list[FrontPoint] = []
    status_filtered = 0
    numeric_filtered = 0
    objective_filtered = 0
    for row_index, row in enumerate(rows):
        if not status_allowed(row, status_col, status_values):
            status_filtered += 1
            continue
        if not filters_allowed(row, filters, path=path, row_index=row_index):
            numeric_filtered += 1
            continue
        quality = parse_float_cell(row.get(quality_col, ""), path=path, row_index=row_index, column=quality_col)
        cost = parse_float_cell(row.get(cost_col, ""), path=path, row_index=row_index, column=cost_col)
        if not valid_objectives(quality, cost, config=config):
            objective_filtered += 1
            continue
        point_id = str(row.get(id_col, "")).strip() if id_col else ""
        if not point_id:
            point_id = str(row_index)
        valid.append(
            FrontPoint(
                row_index=row_index,
                point_id=point_id,
                quality=float(quality),
                cost=float(cost),
                label=label,
            )
        )
    counts = {
        "input_rows": len(rows),
        "status_filtered_rows": status_filtered,
        "numeric_filtered_rows": numeric_filtered,
        "objective_filtered_rows": objective_filtered,
        "valid_rows": len(valid),
    }
    return valid, counts


def is_better(value: float, other: float, direction: str) -> bool:
    """Return whether ``value`` is strictly better than ``other``.

    Parameters
    ----------
    value:
        Candidate value.
    other:
        Reference value.
    direction:
        ``minimize`` or ``maximize``.

    Returns
    -------
    bool
        ``True`` when ``value`` is strictly better.
    """
    return value < other if direction == "minimize" else value > other


def is_no_worse(value: float, other: float, direction: str) -> bool:
    """Return whether ``value`` is no worse than ``other``.

    Parameters
    ----------
    value:
        Candidate value.
    other:
        Reference value.
    direction:
        ``minimize`` or ``maximize``.

    Returns
    -------
    bool
        ``True`` when ``value`` is no worse.
    """
    return value <= other if direction == "minimize" else value >= other


def cost_sort_value(point: FrontPoint, direction: str) -> float:
    """Return a sortable value where lower means better cost.

    Parameters
    ----------
    point:
        Front point.
    direction:
        Cost optimization direction.

    Returns
    -------
    float
        Sort key for deterministic best-cost ordering.
    """
    return point.cost if direction == "minimize" else -point.cost


def point_sort_key(point: FrontPoint, *, quality_direction: str, cost_direction: str) -> tuple[float, float, str, int]:
    """Return a deterministic output sort key for a point.

    Parameters
    ----------
    point:
        Front point.
    quality_direction:
        Quality optimization direction.
    cost_direction:
        Cost optimization direction.

    Returns
    -------
    tuple[float, float, str, int]
        Sort key ordered by quality, cost, ID, then row index.
    """
    quality_key = point.quality if quality_direction == "minimize" else -point.quality
    return (quality_key, cost_sort_value(point, cost_direction), point.point_id, point.row_index)


def dominates(candidate: FrontPoint, point: FrontPoint, *, config: FrontCompareConfig) -> bool:
    """Return whether one point dominates another.

    Parameters
    ----------
    candidate:
        Potential dominator.
    point:
        Potential dominated point.
    config:
        Comparison directions.

    Returns
    -------
    bool
        ``True`` when candidate is no worse in both objectives and strictly
        better in at least one.
    """
    no_worse_quality = is_no_worse(candidate.quality, point.quality, config.quality_direction)
    no_worse_cost = is_no_worse(candidate.cost, point.cost, config.cost_direction)
    strictly_better = is_better(candidate.quality, point.quality, config.quality_direction) or is_better(
        candidate.cost, point.cost, config.cost_direction
    )
    return no_worse_quality and no_worse_cost and strictly_better


def pareto_front(points: Sequence[FrontPoint], *, config: FrontCompareConfig) -> list[FrontPoint]:
    """Compute a deterministic two-objective Pareto front.

    Parameters
    ----------
    points:
        Valid candidate points.
    config:
        Comparison directions.

    Returns
    -------
    list[FrontPoint]
        Non-dominated points sorted by quality, cost, ID, and row index.
    """
    front: list[FrontPoint] = []
    for point in points:
        if not any(dominates(candidate, point, config=config) for candidate in points):
            front.append(point)
    return sorted(
        front,
        key=lambda point: point_sort_key(
            point,
            quality_direction=config.quality_direction,
            cost_direction=config.cost_direction,
        ),
    )


def count_dominated(points: Sequence[FrontPoint], candidates: Sequence[FrontPoint], *, config: FrontCompareConfig) -> int:
    """Count points dominated by any candidate.

    Parameters
    ----------
    points:
        Points being tested for domination.
    candidates:
        Potential dominators.
    config:
        Comparison directions.

    Returns
    -------
    int
        Number of dominated points.
    """
    return sum(1 for point in points if any(dominates(candidate, point, config=config) for candidate in candidates))


def nearest_quality_match(source: FrontPoint, targets: Sequence[FrontPoint], *, config: FrontCompareConfig) -> FrontPoint:
    """Match one source point to the nearest target quality.

    Parameters
    ----------
    source:
        Source-front point.
    targets:
        Target-front candidates.
    config:
        Comparison directions.

    Returns
    -------
    FrontPoint
        Matched target-front point.
    """
    return min(
        targets,
        key=lambda target: (
            abs(target.quality - source.quality),
            cost_sort_value(target, config.cost_direction),
            target.point_id,
            target.row_index,
        ),
    )


def equal_or_better_quality_match(
    source: FrontPoint,
    targets: Sequence[FrontPoint],
    *,
    config: FrontCompareConfig,
) -> tuple[FrontPoint, bool]:
    """Match to the best-cost target with equal-or-better quality.

    Parameters
    ----------
    source:
        Source-front point.
    targets:
        Target-front candidates.
    config:
        Comparison directions.

    Returns
    -------
    tuple[FrontPoint, bool]
        Matched target and whether nearest-quality fallback was used.
    """
    eligible = [
        target
        for target in targets
        if is_no_worse(target.quality, source.quality, config.quality_direction)
    ]
    if not eligible:
        return nearest_quality_match(source, targets, config=config), True
    return (
        min(
            eligible,
            key=lambda target: (
                cost_sort_value(target, config.cost_direction),
                abs(target.quality - source.quality),
                target.point_id,
                target.row_index,
            ),
        ),
        False,
    )


def oriented_delta(source: FrontPoint, target: FrontPoint, direction: str) -> float:
    """Return the positive-is-better cost delta under an orientation.

    Parameters
    ----------
    source:
        Source point.
    target:
        Target point.
    direction:
        ``target-vs-source`` or ``source-vs-target``.

    Returns
    -------
    float
        Oriented cost delta.
    """
    if direction == "target-vs-source":
        return source.cost - target.cost
    return target.cost - source.cost


def make_match(source: FrontPoint, target: FrontPoint, rule: str, fallback: bool, *, config: FrontCompareConfig) -> MatchRow:
    """Create a match row with denominator-specific reductions.

    Parameters
    ----------
    source:
        Source-front point.
    target:
        Target-front point.
    rule:
        Matching rule actually applied.
    fallback:
        Whether a nearest-quality fallback was used.
    config:
        Comparison configuration.

    Returns
    -------
    MatchRow
        Match with computed reductions.
    """
    delta = oriented_delta(source, target, config.reduction_direction)
    return MatchRow(
        source=source,
        target=target,
        match_rule_applied=rule,
        fallback_used=fallback,
        reduction_source_fraction=delta / source.cost,
        reduction_target_fraction=delta / target.cost,
        oriented_cost_delta=delta,
    )


def match_fronts(source_front: Sequence[FrontPoint], target_front: Sequence[FrontPoint], *, config: FrontCompareConfig) -> list[MatchRow]:
    """Match each source-front point to a target-front point.

    Parameters
    ----------
    source_front:
        Source Pareto front.
    target_front:
        Target Pareto front.
    config:
        Matching configuration.

    Returns
    -------
    list[MatchRow]
        Match rows ordered like the source front.

    Raises
    ------
    ValueError
        If either front is empty.
    """
    if not source_front:
        raise ValueError("source front is empty after filtering")
    if not target_front:
        raise ValueError("target front is empty after filtering")
    matches: list[MatchRow] = []
    for source in source_front:
        if config.match_rule == "nearest-quality":
            target = nearest_quality_match(source, target_front, config=config)
            matches.append(make_match(source, target, "nearest-quality", False, config=config))
            continue
        target, fallback = equal_or_better_quality_match(source, target_front, config=config)
        rule = "nearest-quality-fallback" if fallback else "equal-or-better-quality"
        matches.append(make_match(source, target, rule, fallback, config=config))
    return matches


def median(values: Iterable[float]) -> float | None:
    """Return the median of a finite sequence.

    Parameters
    ----------
    values:
        Values to summarize.

    Returns
    -------
    float | None
        Median, or ``None`` when no values are present.
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(statistics.median(finite))


def selected_reduction(match: MatchRow, denominator: str) -> float | None:
    """Return the reduction emphasized by the configured denominator.

    Parameters
    ----------
    match:
        Match row.
    denominator:
        ``source``, ``target``, or ``both``.

    Returns
    -------
    float | None
        Selected reduction fraction, or ``None`` for ``both``.
    """
    if denominator == "source":
        return match.reduction_source_fraction
    if denominator == "target":
        return match.reduction_target_fraction
    return None


def reduction_formulas(config: FrontCompareConfig) -> dict[str, str]:
    """Build human-readable reduction formulas.

    Parameters
    ----------
    config:
        Comparison configuration.

    Returns
    -------
    dict[str, str]
        Formula strings for manifest and summaries.
    """
    if config.reduction_direction == "target-vs-source":
        delta = "source_cost - target_cost"
        interpretation = "positive means target numeric cost is lower than source numeric cost"
    else:
        delta = "target_cost - source_cost"
        interpretation = "positive means source numeric cost is lower than target numeric cost"
    return {
        "oriented_cost_delta": delta,
        "source_denominator_reduction_fraction": f"({delta}) / source_cost",
        "target_denominator_reduction_fraction": f"({delta}) / target_cost",
        "interpretation": interpretation,
    }


def point_record(prefix: str, point: FrontPoint) -> dict[str, Any]:
    """Serialize a front point with a source/target prefix.

    Parameters
    ----------
    prefix:
        Field prefix, usually ``source`` or ``target``.
    point:
        Point to serialize.

    Returns
    -------
    dict[str, Any]
        Flat point record.
    """
    return {
        f"{prefix}_row_index": point.row_index,
        f"{prefix}_id": point.point_id,
        f"{prefix}_quality": point.quality,
        f"{prefix}_cost": point.cost,
        f"{prefix}_label": point.label,
    }


def match_record(match: MatchRow, *, config: FrontCompareConfig) -> dict[str, Any]:
    """Serialize one match row.

    Parameters
    ----------
    match:
        Match to serialize.
    config:
        Comparison configuration.

    Returns
    -------
    dict[str, Any]
        Flat match record.
    """
    selected = selected_reduction(match, config.reduction_denominator)
    record: dict[str, Any] = {}
    record.update(point_record("source", match.source))
    record.update(point_record("target", match.target))
    record.update(
        {
            "match_rule_requested": config.match_rule,
            "match_rule_applied": match.match_rule_applied,
            "fallback_used": match.fallback_used,
            "quality_delta_target_minus_source": match.target.quality - match.source.quality,
            "abs_quality_gap": abs(match.target.quality - match.source.quality),
            "cost_delta_source_minus_target": match.source.cost - match.target.cost,
            "oriented_cost_delta": match.oriented_cost_delta,
            "reduction_direction": config.reduction_direction,
            "selected_reduction_denominator": config.reduction_denominator,
            "selected_reduction_fraction": selected,
            "selected_reduction_percent": None if selected is None else selected * 100.0,
            "source_denominator_reduction_fraction": match.reduction_source_fraction,
            "source_denominator_reduction_percent": match.reduction_source_fraction * 100.0,
            "target_denominator_reduction_fraction": match.reduction_target_fraction,
            "target_denominator_reduction_percent": match.reduction_target_fraction * 100.0,
        }
    )
    return record


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write dictionaries to CSV with a stable header.

    Parameters
    ----------
    path:
        Output CSV path.
    rows:
        Records to write.
    fieldnames:
        Header order.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def front_fieldnames(prefix: str) -> list[str]:
    """Return output field names for one front CSV.

    Parameters
    ----------
    prefix:
        Field prefix, usually ``source`` or ``target``.

    Returns
    -------
    list[str]
        Front CSV field names.
    """
    return [f"{prefix}_row_index", f"{prefix}_id", f"{prefix}_quality", f"{prefix}_cost", f"{prefix}_label"]


def build_summary(
    *,
    config: FrontCompareConfig,
    source_counts: Mapping[str, int],
    target_counts: Mapping[str, int],
    source_front: Sequence[FrontPoint],
    target_front: Sequence[FrontPoint],
    matches: Sequence[MatchRow],
) -> dict[str, Any]:
    """Build machine-readable comparison summary.

    Parameters
    ----------
    config:
        Comparison configuration.
    source_counts:
        Source row-count metadata.
    target_counts:
        Target row-count metadata.
    source_front:
        Source Pareto front.
    target_front:
        Target Pareto front.
    matches:
        Match rows.

    Returns
    -------
    dict[str, Any]
        Summary metrics.
    """
    return {
        "source_label": config.source_label,
        "target_label": config.target_label,
        "quality_direction": config.quality_direction,
        "cost_direction": config.cost_direction,
        "match_rule": config.match_rule,
        "reduction_direction": config.reduction_direction,
        "reduction_denominator": config.reduction_denominator,
        "formulas": reduction_formulas(config),
        "source_counts": dict(source_counts),
        "target_counts": dict(target_counts),
        "source_front_points": len(source_front),
        "target_front_points": len(target_front),
        "source_front_dominated_by_target": count_dominated(source_front, target_front, config=config),
        "target_front_dominated_by_source": count_dominated(target_front, source_front, config=config),
        "match_count": len(matches),
        "fallback_count": sum(1 for match in matches if match.fallback_used),
        "median_abs_quality_gap": median(abs(match.target.quality - match.source.quality) for match in matches),
        "median_oriented_cost_delta": median(match.oriented_cost_delta for match in matches),
        "median_source_denominator_reduction_fraction": median(match.reduction_source_fraction for match in matches),
        "median_source_denominator_reduction_percent": median(match.reduction_source_fraction * 100.0 for match in matches),
        "median_target_denominator_reduction_fraction": median(match.reduction_target_fraction for match in matches),
        "median_target_denominator_reduction_percent": median(match.reduction_target_fraction * 100.0 for match in matches),
        "median_selected_reduction_fraction": median(
            value
            for value in (selected_reduction(match, config.reduction_denominator) for match in matches)
            if value is not None
        ),
        "median_selected_reduction_percent": median(
            value * 100.0
            for value in (selected_reduction(match, config.reduction_denominator) for match in matches)
            if value is not None
        ),
    }


def manifest_dict(config: FrontCompareConfig, argv: Sequence[str], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Build a manifest for reproducibility.

    Parameters
    ----------
    config:
        Comparison configuration.
    argv:
        CLI arguments excluding program name.
    summary:
        Computed summary.

    Returns
    -------
    dict[str, Any]
        Manifest JSON payload.
    """
    return {
        "tool": "compare_pareto_fronts.py",
        "argv": list(argv),
        "inputs": {
            "source_csv": str(config.source_csv),
            "target_csv": str(config.target_csv),
            "source_label": config.source_label,
            "target_label": config.target_label,
        },
        "columns": {
            "source_quality_col": config.source_quality_col,
            "source_cost_col": config.source_cost_col,
            "target_quality_col": config.target_quality_col,
            "target_cost_col": config.target_cost_col,
            "source_id_col": config.source_id_col,
            "target_id_col": config.target_id_col,
            "source_status_col": config.source_status_col,
            "target_status_col": config.target_status_col,
        },
        "rules": {
            "quality_direction": config.quality_direction,
            "cost_direction": config.cost_direction,
            "match_rule": config.match_rule,
            "reduction_denominator": config.reduction_denominator,
            "reduction_direction": config.reduction_direction,
            "source_status_values": list(config.source_status_values),
            "target_status_values": list(config.target_status_values),
            "source_filters": [numeric_filter.__dict__ for numeric_filter in config.source_filters],
            "target_filters": [numeric_filter.__dict__ for numeric_filter in config.target_filters],
            "allow_nonpositive_cost": config.allow_nonpositive_cost,
            "sentinel_abs_threshold": config.sentinel_abs_threshold,
        },
        "formulas": reduction_formulas(config),
        "counts": {
            "source_counts": summary["source_counts"],
            "target_counts": summary["target_counts"],
            "source_front_points": summary["source_front_points"],
            "target_front_points": summary["target_front_points"],
            "match_count": summary["match_count"],
        },
    }


def fmt_optional(value: Any, digits: int = 3) -> str:
    """Format an optional number for Markdown output.

    Parameters
    ----------
    value:
        Value to format.
    digits:
        Decimal places.

    Returns
    -------
    str
        Formatted number or ``n/a``.
    """
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Build a concise Markdown summary.

    Parameters
    ----------
    summary:
        Summary metrics.

    Returns
    -------
    str
        Markdown report.
    """
    formulas = summary["formulas"]
    lines = [
        "# Pareto Front Comparison",
        "",
        f"Source: `{summary['source_label']}`",
        f"Target: `{summary['target_label']}`",
        "",
        "## Rules",
        "",
        f"- Quality direction: `{summary['quality_direction']}`",
        f"- Cost direction: `{summary['cost_direction']}`",
        f"- Match rule: `{summary['match_rule']}`",
        f"- Reduction direction: `{summary['reduction_direction']}`",
        f"- Reduction denominator: `{summary['reduction_denominator']}`",
        f"- Oriented cost delta: `{formulas['oriented_cost_delta']}`",
        f"- Interpretation: {formulas['interpretation']}.",
        "",
        "## Counts",
        "",
        f"- Source valid/front rows: {summary['source_counts']['valid_rows']}/{summary['source_front_points']}",
        f"- Target valid/front rows: {summary['target_counts']['valid_rows']}/{summary['target_front_points']}",
        f"- Matches: {summary['match_count']}",
        f"- Fallback matches: {summary['fallback_count']}",
        f"- Source front dominated by target: {summary['source_front_dominated_by_target']}",
        f"- Target front dominated by source: {summary['target_front_dominated_by_source']}",
        "",
        "## Medians",
        "",
        f"- Absolute quality gap: {fmt_optional(summary['median_abs_quality_gap'])}",
        f"- Oriented cost delta: {fmt_optional(summary['median_oriented_cost_delta'])}",
        f"- Source-denominator reduction: {fmt_optional(summary['median_source_denominator_reduction_percent'])}%",
        f"- Target-denominator reduction: {fmt_optional(summary['median_target_denominator_reduction_percent'])}%",
        f"- Selected-denominator reduction: {fmt_optional(summary['median_selected_reduction_percent'])}%",
        "",
    ]
    return "\n".join(lines)


def run_comparison(config: FrontCompareConfig, argv: Sequence[str]) -> dict[str, Any]:
    """Run a front comparison and write all output artifacts.

    Parameters
    ----------
    config:
        Comparison configuration.
    argv:
        CLI arguments excluding program name for manifest provenance.

    Returns
    -------
    dict[str, Any]
        Summary metrics.
    """
    source_points, source_counts = load_points(
        path=config.source_csv,
        quality_col=config.source_quality_col,
        cost_col=config.source_cost_col,
        id_col=config.source_id_col,
        status_col=config.source_status_col,
        status_values=config.source_status_values,
        filters=config.source_filters,
        label=config.source_label,
        config=config,
    )
    target_points, target_counts = load_points(
        path=config.target_csv,
        quality_col=config.target_quality_col,
        cost_col=config.target_cost_col,
        id_col=config.target_id_col,
        status_col=config.target_status_col,
        status_values=config.target_status_values,
        filters=config.target_filters,
        label=config.target_label,
        config=config,
    )
    source_front = pareto_front(source_points, config=config)
    target_front = pareto_front(target_points, config=config)
    matches = match_fronts(source_front, target_front, config=config)
    summary = build_summary(
        config=config,
        source_counts=source_counts,
        target_counts=target_counts,
        source_front=source_front,
        target_front=target_front,
        matches=matches,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = [point_record("source", point) for point in source_front]
    target_rows = [point_record("target", point) for point in target_front]
    match_rows = [match_record(match, config=config) for match in matches]
    write_csv(config.output_dir / "source_front.csv", source_rows, front_fieldnames("source"))
    write_csv(config.output_dir / "target_front.csv", target_rows, front_fieldnames("target"))
    write_csv(config.output_dir / "matches.csv", match_rows, list(match_rows[0].keys()) if match_rows else [])
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest_dict(config, argv, summary), indent=2),
        encoding="utf-8",
    )
    (config.output_dir / "summary.md").write_text(build_summary_markdown(summary), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare two CSV-derived Pareto fronts. Example: python -B "
            "analysis_scripts/compare_pareto_front_calcs/compare_pareto_fronts.py "
            "--source-csv replay_results.csv --target-csv trials.csv "
            "--source-quality-col source__metric__rmse_total "
            "--source-cost-col target__energy_mj_per_inference "
            "--target-quality-col metric__rmse_total --target-cost-col energy_mj_per_inference "
            "--output-dir outputs/front_compare/example"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-csv", required=True, type=Path, help="Source CSV path.")
    parser.add_argument("--target-csv", required=True, type=Path, help="Target CSV path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for comparison artifacts.")
    parser.add_argument("--source-quality-col", required=True, help="Quality column in source CSV.")
    parser.add_argument("--source-cost-col", required=True, help="Cost/energy column in source CSV.")
    parser.add_argument("--target-quality-col", required=True, help="Quality column in target CSV.")
    parser.add_argument("--target-cost-col", required=True, help="Cost/energy column in target CSV.")
    parser.add_argument("--quality-direction", choices=sorted(DIRECTIONS), default="minimize")
    parser.add_argument("--cost-direction", choices=sorted(DIRECTIONS), default="minimize")
    parser.add_argument("--match-rule", choices=sorted(MATCH_RULES), default="nearest-quality")
    parser.add_argument("--reduction-denominator", choices=sorted(REDUCTION_DENOMINATORS), default="source")
    parser.add_argument("--reduction-direction", choices=sorted(REDUCTION_DIRECTIONS), default="target-vs-source")
    parser.add_argument("--source-label", default="source", help="Human-readable source label.")
    parser.add_argument("--target-label", default="target", help="Human-readable target label.")
    parser.add_argument("--source-id-col", default=None, help="Optional source row identifier column.")
    parser.add_argument("--target-id-col", default=None, help="Optional target row identifier column.")
    parser.add_argument("--source-status-col", default=None, help="Optional source status filter column.")
    parser.add_argument("--target-status-col", default=None, help="Optional target status filter column.")
    parser.add_argument(
        "--source-filter",
        action="append",
        nargs=3,
        metavar=("COLUMN", "OP", "VALUE"),
        default=[],
        help="Additional numeric source filter, repeatable. OP is one of lt, le, eq, ge, gt, ne.",
    )
    parser.add_argument(
        "--target-filter",
        action="append",
        nargs=3,
        metavar=("COLUMN", "OP", "VALUE"),
        default=[],
        help="Additional numeric target filter, repeatable. OP is one of lt, le, eq, ge, gt, ne.",
    )
    parser.add_argument(
        "--source-status-values",
        nargs="+",
        default=list(DEFAULT_STATUS_VALUES),
        help="Allowed source status values when --source-status-col is set.",
    )
    parser.add_argument(
        "--target-status-values",
        nargs="+",
        default=list(DEFAULT_STATUS_VALUES),
        help="Allowed target status values when --target-status-col is set.",
    )
    parser.add_argument(
        "--allow-nonpositive-cost",
        action="store_true",
        help="Allow zero or negative cost values instead of filtering them for minimized costs.",
    )
    parser.add_argument(
        "--sentinel-abs-threshold",
        type=float,
        default=1.0e11,
        help="Filter objective values with absolute value at or above this threshold.",
    )
    return parser


def parse_numeric_filters(raw_filters: Sequence[Sequence[str]]) -> tuple[NumericFilter, ...]:
    """Parse CLI numeric filter triples.

    Parameters
    ----------
    raw_filters:
        ``COLUMN OP VALUE`` triples from argparse.

    Returns
    -------
    tuple[NumericFilter, ...]
        Parsed numeric filters.

    Raises
    ------
    ValueError
        If a filter value is malformed.
    """
    filters: list[NumericFilter] = []
    for column, op, raw_value in raw_filters:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Filter value for {column!r} is not numeric: {raw_value!r}") from exc
        filters.append(NumericFilter(column=column, op=op, value=value))
    return tuple(filters)


def namespace_to_config(args: argparse.Namespace) -> FrontCompareConfig:
    """Convert parsed CLI arguments to comparison configuration.

    Parameters
    ----------
    args:
        Parsed CLI namespace.

    Returns
    -------
    FrontCompareConfig
        Comparison configuration.
    """
    return FrontCompareConfig(
        source_csv=args.source_csv.expanduser(),
        target_csv=args.target_csv.expanduser(),
        output_dir=args.output_dir.expanduser(),
        source_quality_col=args.source_quality_col,
        source_cost_col=args.source_cost_col,
        target_quality_col=args.target_quality_col,
        target_cost_col=args.target_cost_col,
        quality_direction=args.quality_direction,
        cost_direction=args.cost_direction,
        match_rule=args.match_rule,
        reduction_denominator=args.reduction_denominator,
        reduction_direction=args.reduction_direction,
        source_label=args.source_label,
        target_label=args.target_label,
        source_id_col=args.source_id_col,
        target_id_col=args.target_id_col,
        source_status_col=args.source_status_col,
        target_status_col=args.target_status_col,
        source_status_values=tuple(args.source_status_values),
        target_status_values=tuple(args.target_status_values),
        source_filters=parse_numeric_filters(args.source_filter),
        target_filters=parse_numeric_filters(args.target_filter),
        allow_nonpositive_cost=args.allow_nonpositive_cost,
        sentinel_abs_threshold=args.sentinel_abs_threshold,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line front comparison.

    Parameters
    ----------
    argv:
        Optional CLI arguments excluding program name.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_arg_parser()
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(parsed_argv)
    try:
        summary = run_comparison(namespace_to_config(args), parsed_argv)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote comparison artifacts: {args.output_dir}")
    print(f"Matches: {summary['match_count']}")
    print(f"Median selected reduction (%): {fmt_optional(summary['median_selected_reduction_percent'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
