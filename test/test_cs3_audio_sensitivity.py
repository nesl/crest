"""Tests for the generic CS3 score-sensitivity analysis script."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT_DIR / "analysis_scripts" / "cs3_audio_sensitivity" / "score_sensitivity.py"


def load_cs3_module() -> Any:
    """Load the score-sensitivity script as an importable module.

    Returns
    -------
    Any
        Imported module object.
    """

    spec = importlib.util.spec_from_file_location("cs3_score_sensitivity_for_tests", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cs3_score_sensitivity_for_tests"] = module
    spec.loader.exec_module(module)
    return module


cs3_score_sensitivity = load_cs3_module()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to a CSV with a union header.

    Parameters
    ----------
    path:
        Output CSV path.
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


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    Any
        Parsed JSON payload.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV headers and rows.

    Parameters
    ----------
    path:
        CSV file path.

    Returns
    -------
    tuple[list[str], list[dict[str, str]]]
        Header names and parsed row dictionaries.
    """

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def run_cli(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run the score-sensitivity CLI.

    Parameters
    ----------
    args:
        CLI arguments excluding the interpreter and script path.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed subprocess result.
    """

    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def nas_row(
    trial: str,
    quality: float,
    energy: float,
    *,
    latency: float = 1.0,
    state: str = "COMPLETE",
    feasible: str = "true",
    source_score: Any = 1.0,
    depth: str = "2",
) -> dict[str, Any]:
    """Build one synthetic NAS CSV row.

    Parameters
    ----------
    trial:
        Trial identifier.
    quality:
        Macro-F1-like quality value.
    energy:
        Energy per inference in millijoules.
    latency:
        Latency in milliseconds.
    state:
        Trial state.
    feasible:
        Feasibility flag.
    source_score:
        Source score cell.
    depth:
        Architecture parameter value.

    Returns
    -------
    dict[str, Any]
        Synthetic CSV row.
    """

    return {
        "number": trial,
        "user_attrs_metric_macro_f1": quality,
        "user_attrs_energy_mj_per_inference": energy,
        "user_attrs_latency_ms": latency,
        "state": state,
        "user_attrs_feasible": feasible,
        "value_score": source_score,
        "params_depth": depth,
    }


def test_cli_help_lists_sensitivity_flags() -> None:
    """CLI help should expose generic run, sweep, and column flags.

    Returns
    -------
    None
        The test passes when help exits successfully and includes expected flags.
    """

    result = run_cli(["--help"])

    assert result.returncode == 0
    assert "--run" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--quality-col" in result.stdout
    assert "--budget-mj" in result.stdout
    assert "--baseline-lambda" in result.stdout


def test_cli_rejects_malformed_run_and_duplicate_labels(tmp_path: Path) -> None:
    """Malformed run specs and duplicate labels should fail clearly.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when both invalid commands fail.
    """

    malformed = run_cli(["--run", "missing-equals", "--output-dir", str(tmp_path / "out")])
    duplicate = run_cli(
        [
            "--run",
            f"same={tmp_path / 'a.csv'}",
            "--run",
            f"same={tmp_path / 'b.csv'}",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert malformed.returncode == 2
    assert "--run must use LABEL=CSV_OR_RUN_DIR" in malformed.stderr
    assert duplicate.returncode == 2
    assert "duplicate run label" in duplicate.stderr


def test_missing_configured_column_fails_clearly(tmp_path: Path) -> None:
    """Explicit missing columns should produce a clear error.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when the CLI reports the missing configured column.
    """

    source = tmp_path / "source.csv"
    write_csv(source, [nas_row("1", 0.5, 10.0)])

    result = run_cli(
        [
            "--run",
            f"run={source}",
            "--quality-col",
            "missing_quality",
            "--energy-col",
            "user_attrs_energy_mj_per_inference",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert result.returncode == 2
    assert "missing configured column" in result.stderr
    assert "missing_quality" in result.stderr


def test_directory_discovery_is_top_level_and_uses_usable_csv(tmp_path: Path) -> None:
    """Directory discovery should ignore nested CSVs and unusable logs.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when the selected mapping points at the top-level usable CSV.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nested = run_dir / "nested"
    nested.mkdir()
    write_csv(run_dir / "bad.csv", [{"number": "1", "energy": 1.0}])
    write_csv(run_dir / "good.csv", [nas_row("7", 0.7, 10.0)])
    write_csv(nested / "better.csv", [nas_row("99", 0.99, 1.0)])
    (run_dir / "ignored.csv.lock").write_text("not,a,csv\n", encoding="utf-8")

    config = cs3_score_sensitivity.SensitivityConfig(
        runs=(cs3_score_sensitivity.RunInput("discovered", run_dir),),
        output_dir=tmp_path / "out",
    )
    summary = cs3_score_sensitivity.run_sensitivity(config)

    assert summary["runs"]["discovered"]["mapping"]["csv_path"].endswith("good.csv")
    assert summary["baseline_selections"][0]["trial_id"] == "7"


def test_filtering_and_source_score_behavior(tmp_path: Path) -> None:
    """Filtering should preserve status, feasible, finite objective, and source-score gates.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when only the valid candidate remains.
    """

    source = tmp_path / "source.csv"
    write_csv(
        source,
        [
            nas_row("valid", 0.6, 10.0),
            nas_row("failed", 0.9, 1.0, state="FAIL"),
            nas_row("infeasible", 0.9, 1.0, feasible="false"),
            nas_row("missing_source_score", 0.9, 1.0, source_score=""),
            nas_row("infinite_source_score", 0.9, 1.0, source_score="inf"),
            nas_row("missing_quality", "", 1.0),
            nas_row("infinite_quality", "inf", 1.0),
            nas_row("missing_energy", 0.9, ""),
            nas_row("infinite_energy", 0.9, "-inf"),
        ],
    )

    summary = cs3_score_sensitivity.run_sensitivity(
        cs3_score_sensitivity.SensitivityConfig(
            runs=(cs3_score_sensitivity.RunInput("filtered", source),),
            output_dir=tmp_path / "out",
        )
    )

    assert summary["runs"]["filtered"]["candidate_count"] == 1
    assert summary["baseline_selections"][0]["trial_id"] == "valid"


def test_output_schema_and_synthetic_golden_sweeps(tmp_path: Path) -> None:
    """Synthetic data should preserve the CS3 sensitivity outcome pattern.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when baseline selections and sweep changes match the
        preserved CS3 pattern.
    """

    portenta = tmp_path / "portenta.csv"
    stm32 = tmp_path / "stm32.csv"
    write_csv(
        portenta,
        [
            nas_row("49", 0.812, 702.0),
            nas_row("152", 0.786, 300.0),
            nas_row("225", 0.760, 152.0),
            nas_row("142", 0.700, 40.0),
            nas_row("5", 0.100, 23.0),
            nas_row("999", 0.990, 1.0, source_score=""),
        ],
    )
    write_csv(
        stm32,
        [
            nas_row("187", 0.7815, 228.0),
            nas_row("186", 0.7757, 49.0),
            nas_row("95", 0.7400, 10.0),
            nas_row("5", 0.2000, 4.0),
            nas_row("999", 0.9900, 1.0, source_score=""),
        ],
    )

    output_dir = tmp_path / "out"
    summary = cs3_score_sensitivity.run_sensitivity(
        cs3_score_sensitivity.SensitivityConfig(
            runs=(
                cs3_score_sensitivity.RunInput("Portenta M7", portenta),
                cs3_score_sensitivity.RunInput("STM32", stm32),
            ),
            output_dir=output_dir,
        )
    )
    manifest = read_json(output_dir / "manifest.json")
    fieldnames, _ = read_csv_rows(output_dir / "selections.csv")

    baseline = {row["run_label"]: row["trial_id"] for row in summary["baseline_selections"]}
    portenta_budgets = {
        row["energy_budget_mj"]: row["trial_id"]
        for row in summary["budget_sweep"]
        if row["run_label"] == "Portenta M7"
    }
    stm32_budgets = {
        row["energy_budget_mj"]: row["trial_id"]
        for row in summary["budget_sweep"]
        if row["run_label"] == "STM32"
    }
    portenta_lambdas = {
        row["lambda"]: row["trial_id"]
        for row in summary["lambda_sweep"]
        if row["run_label"] == "Portenta M7"
    }
    stm32_lambdas = {
        row["lambda"]: row["trial_id"]
        for row in summary["lambda_sweep"]
        if row["run_label"] == "STM32"
    }

    assert fieldnames == list(cs3_score_sensitivity.OUTPUT_FIELDS)
    assert manifest["formula"] == "quality - lambda * energy / energy_budget_mj"
    assert baseline == {"Portenta M7": "225", "STM32": "186"}
    assert portenta_budgets == {
        100.0: "142",
        200.0: "225",
        300.0: "225",
        400.0: "225",
        600.0: "152",
        800.0: "152",
        1200.0: "152",
    }
    assert stm32_budgets == {
        100.0: "95",
        200.0: "186",
        300.0: "186",
        400.0: "186",
        600.0: "186",
        800.0: "186",
        1200.0: "186",
    }
    assert portenta_lambdas == {
        0.0: "49",
        0.025: "49",
        0.05: "152",
        0.10: "225",
        0.15: "225",
        0.20: "225",
        0.30: "142",
    }
    assert stm32_lambdas[0.0] == "187"
    assert {trial for lam, trial in stm32_lambdas.items() if lam > 0.0} == {"186"}
