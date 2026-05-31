"""Tests for the generic Pareto-front comparison analysis script."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "analysis_scripts" / "compare_pareto_front_calcs" / "compare_pareto_fronts.py"


def load_compare_module() -> Any:
    """Load the comparison script as an importable module.

    Returns
    -------
    Any
        Imported module object.

    Raises
    ------
    RuntimeError
        If existing validation or execution checks fail.
    """
    spec = importlib.util.spec_from_file_location("compare_pareto_fronts_for_tests", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_pareto_fronts_for_tests"] = module
    spec.loader.exec_module(module)
    return module


compare_pareto_fronts = load_compare_module()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to a CSV with a union header.

    Parameters
    ----------
    path:
        Output path.
    rows:
        Row mappings.

    Returns
    -------
    None
        The CSV is written to disk.
    """
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def base_config(tmp_path: Path, source_csv: Path, target_csv: Path, **overrides: Any) -> Any:
    """Build a comparison config for synthetic tests.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    source_csv:
        Source CSV path.
    target_csv:
        Target CSV path.
    **overrides:
        Config values to override.

    Returns
    -------
    Any
        ``FrontCompareConfig`` instance from the script under test.
    """
    kwargs = {
        "source_csv": source_csv,
        "target_csv": target_csv,
        "output_dir": tmp_path / "out",
        "source_quality_col": "quality",
        "source_cost_col": "cost",
        "target_quality_col": "quality",
        "target_cost_col": "cost",
        "source_id_col": "id",
        "target_id_col": "id",
    }
    kwargs.update(overrides)
    return compare_pareto_fronts.FrontCompareConfig(**kwargs)


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Parameters
    ----------
    path:
        JSON path.

    Returns
    -------
    Any
        Parsed JSON payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row dictionaries.

    Parameters
    ----------
    path:
        CSV path.

    Returns
    -------
    list[dict[str, str]]
        Parsed rows.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cli_help_lists_semantic_flags() -> None:
    """CLI help should expose matching, denominator, and orientation flags.

    Returns
    -------
    None
        The test passes when help exits successfully and includes expected flags.
    """
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--match-rule" in result.stdout
    assert "--reduction-denominator" in result.stdout
    assert "--reduction-direction" in result.stdout
    assert "--source-filter" in result.stdout
    assert "--source-quality-col" in result.stdout


def test_missing_column_and_malformed_numeric_errors(tmp_path: Path) -> None:
    """Missing columns and malformed numbers should fail clearly.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when expected exceptions are raised.
    """
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    write_csv(source, [{"id": "s1", "quality": "1.0", "cost": "10"}])
    write_csv(target, [{"id": "t1", "quality": "1.0", "cost": "8"}])

    with pytest.raises(KeyError, match="missing required column"):
        compare_pareto_fronts.run_comparison(
            base_config(tmp_path, source, target, source_quality_col="missing_quality"),
            ["test"],
        )

    write_csv(source, [{"id": "s1", "quality": "bad", "cost": "10"}])
    with pytest.raises(ValueError, match="not numeric"):
        compare_pareto_fronts.run_comparison(base_config(tmp_path, source, target), ["test"])


def test_filtering_status_sentinel_and_nonpositive_cost(tmp_path: Path) -> None:
    """Filtering should remove invalid rows before front construction.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when summary counts reflect filtered rows.
    """
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    write_csv(
        source,
        [
            {"id": "valid", "quality": "1.0", "cost": "10", "status": "completed", "latency": "100"},
            {"id": "bad_status", "quality": "1.0", "cost": "9", "status": "failed", "latency": "100"},
            {"id": "sentinel", "quality": "1e12", "cost": "8", "status": "completed", "latency": "100"},
            {"id": "nonpositive", "quality": "0.5", "cost": "0", "status": "completed", "latency": "100"},
            {"id": "numeric_filter", "quality": "0.9", "cost": "7", "status": "completed", "latency": "250"},
        ],
    )
    write_csv(target, [{"id": "t1", "quality": "1.0", "cost": "8", "status": "COMPLETE"}])

    summary = compare_pareto_fronts.run_comparison(
        base_config(
            tmp_path,
            source,
            target,
            source_status_col="status",
            source_status_values=("completed",),
            source_filters=(compare_pareto_fronts.NumericFilter("latency", "le", 200.0),),
            target_status_col="status",
            target_status_values=("COMPLETE",),
        ),
        ["test"],
    )

    assert summary["source_counts"]["valid_rows"] == 1
    assert summary["source_counts"]["status_filtered_rows"] == 1
    assert summary["source_counts"]["numeric_filtered_rows"] == 1
    assert summary["source_counts"]["objective_filtered_rows"] == 2


def test_pareto_recomputation_removes_dominated_rows_deterministically(tmp_path: Path) -> None:
    """Pareto recomputation should remove dominated rows and keep stable order.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when expected front IDs are written.
    """
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    write_csv(
        source,
        [
            {"id": "dominated", "quality": "1.0", "cost": "10"},
            {"id": "better_cost", "quality": "1.0", "cost": "9"},
            {"id": "tradeoff", "quality": "0.8", "cost": "12"},
            {"id": "tie_a", "quality": "0.7", "cost": "13"},
            {"id": "tie_b", "quality": "0.7", "cost": "13"},
        ],
    )
    write_csv(target, [{"id": "t1", "quality": "1.0", "cost": "8"}])

    compare_pareto_fronts.run_comparison(base_config(tmp_path, source, target), ["test"])
    rows = read_csv_rows(tmp_path / "out" / "source_front.csv")

    assert [row["source_id"] for row in rows] == ["tie_a", "tie_b", "tradeoff", "better_cost"]


def test_match_rules_can_produce_distinct_matches(tmp_path: Path) -> None:
    """Nearest-quality and equal-or-better-quality should be distinct rules.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when the two rules choose different target IDs.
    """
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    write_csv(source, [{"id": "s1", "quality": "1.0", "cost": "10"}])
    write_csv(
        target,
        [
            {"id": "nearest_but_worse_quality", "quality": "1.05", "cost": "5"},
            {"id": "eligible_better_quality", "quality": "0.8", "cost": "100"},
        ],
    )

    nearest_config = base_config(tmp_path, source, target, output_dir=tmp_path / "nearest")
    compare_pareto_fronts.run_comparison(nearest_config, ["nearest"])
    equal_config = base_config(
        tmp_path,
        source,
        target,
        output_dir=tmp_path / "equal",
        match_rule="equal-or-better-quality",
    )
    compare_pareto_fronts.run_comparison(equal_config, ["equal"])

    nearest_match = read_csv_rows(tmp_path / "nearest" / "matches.csv")[0]
    equal_match = read_csv_rows(tmp_path / "equal" / "matches.csv")[0]
    assert nearest_match["target_id"] == "nearest_but_worse_quality"
    assert equal_match["target_id"] == "eligible_better_quality"
    assert equal_match["fallback_used"] == "False"


def test_denominator_modes_change_selected_reduction(tmp_path: Path) -> None:
    """Source and target denominators should produce different percentages.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when summary reductions use the requested denominator.
    """
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    write_csv(source, [{"id": "s1", "quality": "1.0", "cost": "100"}])
    write_csv(target, [{"id": "t1", "quality": "1.0", "cost": "60"}])

    source_summary = compare_pareto_fronts.run_comparison(
        base_config(tmp_path, source, target, output_dir=tmp_path / "source_denom"),
        ["source"],
    )
    target_summary = compare_pareto_fronts.run_comparison(
        base_config(tmp_path, source, target, output_dir=tmp_path / "target_denom", reduction_denominator="target"),
        ["target"],
    )

    assert source_summary["median_selected_reduction_percent"] == pytest.approx(40.0)
    assert target_summary["median_selected_reduction_percent"] == pytest.approx(66.6666667)


def test_cs2_reverse_style_fixture_uses_source_vs_target_orientation(tmp_path: Path) -> None:
    """CS2 reverse-style reduction should support the 44.1 percent orientation.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when target-denominator source-vs-target reduction is
        positive and rounds to 44.1 percent.
    """
    source = tmp_path / "cadenced_replay.csv"
    target = tmp_path / "native_continuous.csv"
    write_csv(source, [{"id": "cad_replay", "quality": "1.0", "cost": "55.9"}])
    write_csv(target, [{"id": "native_b2b", "quality": "1.0", "cost": "100.0"}])

    summary = compare_pareto_fronts.run_comparison(
        base_config(
            tmp_path,
            source,
            target,
            reduction_direction="source-vs-target",
            reduction_denominator="target",
        ),
        ["cs2"],
    )

    assert summary["median_selected_reduction_percent"] == pytest.approx(44.1)


def test_outputs_include_manifest_summary_fronts_and_matches(tmp_path: Path) -> None:
    """End-to-end CLI run should write all reproducibility artifacts.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when files exist and include expected schema values.
    """
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    output_dir = tmp_path / "cli_out"
    write_csv(source, [{"id": "s1", "quality": "1.0", "cost": "100", "status": "completed"}])
    write_csv(target, [{"id": "t1", "quality": "1.0", "cost": "60", "status": "COMPLETE"}])

    exit_code = compare_pareto_fronts.main(
        [
            "--source-csv",
            str(source),
            "--target-csv",
            str(target),
            "--source-quality-col",
            "quality",
            "--source-cost-col",
            "cost",
            "--target-quality-col",
            "quality",
            "--target-cost-col",
            "cost",
            "--source-id-col",
            "id",
            "--target-id-col",
            "id",
            "--source-status-col",
            "status",
            "--source-status-values",
            "completed",
            "--target-status-col",
            "status",
            "--target-status-values",
            "COMPLETE",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    for filename in [
        "manifest.json",
        "source_front.csv",
        "target_front.csv",
        "matches.csv",
        "summary.json",
        "summary.md",
    ]:
        assert (output_dir / filename).exists()
    manifest = read_json(output_dir / "manifest.json")
    summary = read_json(output_dir / "summary.json")
    matches = read_csv_rows(output_dir / "matches.csv")
    assert manifest["formulas"]["oriented_cost_delta"] == "source_cost - target_cost"
    assert manifest["counts"]["match_count"] == 1
    assert summary["median_selected_reduction_percent"] == pytest.approx(40.0)
    assert matches[0]["source_id"] == "s1"
    assert matches[0]["target_id"] == "t1"
