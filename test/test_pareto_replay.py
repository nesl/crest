"""Tests for first-class Pareto HIL replay support."""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pareto_hil_replay  # noqa: E402
from tinyodom import pareto_replay  # noqa: E402


def source_config() -> dict[str, Any]:
    """Build a minimal source NAS config for replay tests.

    Returns
    -------
    dict[str, Any]
        Config mapping with objectives and CSV log name.
    """

    return {
        "outputs": {"log_file_name": "log_NAS_test.csv"},
        "nas": {
            "score": {
                "params": {
                    "objectives": [
                        {"metric": "rmse_total", "direction": "minimize"},
                        {"metric": "flops", "direction": "minimize"},
                    ]
                }
            }
        },
    }


def base_row(**overrides: Any) -> dict[str, str]:
    """Build one valid synthetic NAS CSV row.

    Parameters
    ----------
    **overrides : Any
        Column values to override in the default row.

    Returns
    -------
    dict[str, str]
        String-valued CSV row mapping.
    """

    row: dict[str, Any] = {
        "trial_number": "0",
        "pruned": "False",
        "error_code": "1",
        "quantization_mode": "float",
        "metric__rmse_total": "0.10",
        "flops": "1000",
        "accuracy": "0.80",
        "hparam__batch_size": "1",
        "hparam__timesteps": "8",
        "hparam__input_dim": "6",
        "hparam__num_layers": "2",
        "hparam__kernel_sizes": "[3,5]",
        "cpu_clock_mhz_requested": "80",
        "objective_names_json": '["rmse_total", "flops"]',
        "objective_directions_json": '["minimize", "minimize"]',
    }
    row.update(overrides)
    return {key: "" if value is None else str(value) for key, value in row.items()}


def write_source_run(
    tmp_path: Path,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Write a synthetic source run directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    rows : Sequence[Mapping[str, Any]]
        Source CSV rows to write.
    config : Mapping[str, Any] | None, optional
        Optional source config override.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Source run directory, CSV path, and config path.
    """

    source_dir = tmp_path / "source_run"
    source_dir.mkdir()
    config_path = source_dir / "nas_config.yaml"
    config_path.write_text(yaml.safe_dump(dict(config or source_config())), encoding="utf-8")
    csv_path = source_dir / "log_NAS_test.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return source_dir, csv_path, config_path


def write_target_config(tmp_path: Path) -> Path:
    """Write a placeholder target HIL config.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    pathlib.Path
        Target config path.
    """

    target_config = tmp_path / "target.yaml"
    target_config.write_text("model:\n  family: odom_tcn\n", encoding="utf-8")
    return target_config


def make_args(
    *,
    source_run_dir: Path,
    target_config: Path,
    output_dir: Path,
    dry_run: bool,
    resume: bool = False,
    allow_gpu: bool = False,
    max_candidates: int | None = None,
    device_option_policy: str = "preserve-source",
    model_variant: str | None = None,
    checkpoint_path: str | None = None,
) -> pareto_replay.ReplayRunConfig:
    """Build a replay run config.

    Parameters
    ----------
    source_run_dir : pathlib.Path
        Source run directory.
    target_config : pathlib.Path
        Target config path.
    output_dir : pathlib.Path
        Replay output directory.
    dry_run : bool
        Whether to use dry-run mode.
    resume : bool, optional
        Whether to skip completed payloads.
    allow_gpu : bool, optional
        Whether to preserve ``CUDA_VISIBLE_DEVICES``.
    max_candidates : int | None, optional
        Optional candidate cap.
    device_option_policy : str, optional
        Device option replay policy.
    model_variant : str | None, optional
        Optional model variant.
    checkpoint_path : str | None, optional
        Optional checkpoint path.

    Returns
    -------
    tinyodom.pareto_replay.ReplayRunConfig
        Config compatible with ``pareto_replay.run_replay``.
    """

    return pareto_replay.ReplayRunConfig(
        source_run_dir=source_run_dir,
        target_run_dir=None,
        target_config=target_config,
        source_csv=None,
        source_config=None,
        objectives=None,
        output_dir=output_dir,
        max_candidates=max_candidates,
        dry_run=dry_run,
        resume=resume,
        allow_gpu=allow_gpu,
        device_option_policy=device_option_policy,
        model_variant=model_variant,
        checkpoint_path=checkpoint_path,
    )


def read_results(output_dir: Path) -> list[dict[str, str]]:
    """Read replay results from an output directory.

    Parameters
    ----------
    output_dir : pathlib.Path
        Replay output directory.

    Returns
    -------
    list[dict[str, str]]
        Parsed CSV rows.
    """

    with (output_dir / "replay_results.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_request_records(output_dir: Path) -> list[dict[str, Any]]:
    """Read replay request JSONL records.

    Parameters
    ----------
    output_dir : pathlib.Path
        Replay output directory.

    Returns
    -------
    list[dict[str, Any]]
        Parsed request records.
    """

    requests_path = output_dir / "replay_requests.jsonl"
    if not requests_path.exists():
        return []
    return [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines()]


def test_replay_run_config_rejects_invalid_direct_library_config(tmp_path: Path) -> None:
    """Check direct-library replay config validation.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when invalid configs raise ``ValueError``.
    """

    valid_kwargs = {
        "source_run_dir": tmp_path / "source",
        "target_run_dir": None,
        "target_config": tmp_path / "target.yaml",
        "source_csv": None,
        "source_config": None,
        "objectives": None,
        "output_dir": None,
        "max_candidates": None,
        "dry_run": True,
        "resume": False,
        "allow_gpu": False,
        "device_option_policy": "preserve-source",
        "model_variant": None,
        "checkpoint_path": None,
    }

    with pytest.raises(ValueError, match="positive integer"):
        pareto_replay.ReplayRunConfig(**{**valid_kwargs, "max_candidates": 0})
    with pytest.raises(ValueError, match="target_run_dir or target_config"):
        pareto_replay.ReplayRunConfig(**{**valid_kwargs, "target_config": None})
    with pytest.raises(ValueError, match="Unsupported device option policy"):
        pareto_replay.ReplayRunConfig(**{**valid_kwargs, "device_option_policy": "invalid"})


def test_namespace_to_replay_config_maps_all_cli_fields() -> None:
    """Check CLI namespace conversion to replay config.

    Returns
    -------
    None
        The test passes when every CLI field maps to the typed config.
    """

    parser = pareto_hil_replay.build_arg_parser()
    args = parser.parse_args(
        [
            "--source-run-dir",
            "models/source",
            "--target-config",
            "config.yaml",
            "--source-csv",
            "log.csv",
            "--source-config",
            "nas_config.yaml",
            "--objectives",
            "rmse_total:minimize,flops:minimize",
            "--output-dir",
            "models/replays/out",
            "--max-candidates",
            "3",
            "--dry-run",
            "--resume",
            "--allow-gpu",
            "--device-option-policy",
            "target-default",
            "--model-variant",
            "float_model",
            "--checkpoint-path",
            "/tmp/model.keras",
        ]
    )
    config = pareto_hil_replay.namespace_to_replay_config(args)
    assert config == pareto_replay.ReplayRunConfig(
        source_run_dir=Path("models/source"),
        target_run_dir=None,
        target_config=Path("config.yaml"),
        source_csv=Path("log.csv"),
        source_config=Path("nas_config.yaml"),
        objectives="rmse_total:minimize,flops:minimize",
        output_dir=Path("models/replays/out"),
        max_candidates=3,
        dry_run=True,
        resume=True,
        allow_gpu=True,
        device_option_policy="target-default",
        model_variant="float_model",
        checkpoint_path="/tmp/model.keras",
    )


def test_parser_requires_exactly_one_target_option() -> None:
    """Check parser validation for target selection.

    Returns
    -------
    None
        The test passes when exactly one target selector is required.
    """

    parser = pareto_hil_replay.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-run-dir", "models/source"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--source-run-dir",
                "models/source",
                "--target-run-dir",
                "models/target",
                "--target-config",
                "config.yaml",
            ]
        )
    args = parser.parse_args(["--source-run-dir", "models/source", "--target-run-dir", "models/target"])
    config = pareto_hil_replay.namespace_to_replay_config(args)
    assert config.target_run_dir == Path("models/target")
    assert config.target_config is None


def test_replay_library_has_no_cli_parser_ownership() -> None:
    """Check that CLI parser helpers live outside the replay library.

    Returns
    -------
    None
        The test passes when parser symbols are absent from the library module.
    """

    assert "argparse" not in pareto_replay.__dict__
    assert not hasattr(pareto_replay, "build_arg_parser")
    assert not hasattr(pareto_replay, "positive_int")
    assert not hasattr(pareto_replay, "main")


def test_parse_cell_value_handles_scalars_and_json() -> None:
    """Check parsing of scalar and JSON-like CSV cells.

    Returns
    -------
    None
        The test passes when parsed values match expected Python values.
    """

    assert pareto_replay.parse_cell_value("true") is True
    assert pareto_replay.parse_cell_value("7") == 7
    assert pareto_replay.parse_cell_value("7.5") == 7.5
    assert pareto_replay.parse_cell_value("[1, 2]") == [1, 2]
    assert pareto_replay.parse_cell_value("nan") is None
    assert pareto_replay.parse_cell_value("{not json}") == "{not json}"


def test_objective_override_and_log_inference_fallback() -> None:
    """Check objective override parsing and malformed-log fallback behavior.

    Returns
    -------
    None
        The test passes when objectives resolve from override and later rows.
    """

    columns = ["metric__rmse_total", "flops"]
    override = pareto_replay.parse_objective_override("rmse_total:min,flops:maximize", columns)
    assert override == (
        pareto_replay.ObjectiveSpec("metric__rmse_total", "minimize", "rmse_total"),
        pareto_replay.ObjectiveSpec("flops", "maximize", "flops"),
    )

    rows = [
        {"objective_names_json": "{bad", "objective_directions_json": '["minimize"]'},
        {
            "objective_names_json": '["rmse_total", "flops"]',
            "objective_directions_json": '["minimize", "minimize"]',
        },
    ]
    inferred = pareto_replay.resolve_objective_specs(config={}, rows=rows, columns=columns)
    assert [spec.column for spec in inferred] == ["metric__rmse_total", "flops"]


def test_read_csv_rows_rejects_empty_and_header_only_csv(tmp_path: Path) -> None:
    """Check CSV validation for missing headers and missing data rows.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when invalid CSV files raise clear errors.
    """

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no header"):
        pareto_replay.read_csv_rows(empty)

    header_only = tmp_path / "header_only.csv"
    header_only.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        pareto_replay.read_csv_rows(header_only)


def test_source_row_filtering_and_multi_direction_pareto() -> None:
    """Check filtering and mixed minimize/maximize Pareto dominance.

    Returns
    -------
    None
        The test passes when invalid rows are skipped and the front is stable.
    """

    objectives = (
        pareto_replay.ObjectiveSpec("metric__rmse_total", "minimize", "rmse_total"),
        pareto_replay.ObjectiveSpec("accuracy", "maximize", "accuracy"),
    )
    rows = [
        base_row(metric__rmse_total="0.10", accuracy="0.80"),
        base_row(metric__rmse_total="0.10", accuracy="0.70"),
        base_row(metric__rmse_total="0.20", accuracy="0.90"),
        base_row(metric__rmse_total="0.05", accuracy="0.95", pruned="True"),
    ]
    assert pareto_replay.pareto_indices(rows, objectives) == [0, 2]


def test_build_candidate_preserves_cpu_clock_and_runtime_payload() -> None:
    """Check replay payload reconstruction with source device options.

    Returns
    -------
    None
        The test passes when payload fields are split and typed correctly.
    """

    candidate = pareto_replay.build_replay_candidate(
        source_row_index=4,
        row=base_row(cpu_clock_mhz_requested="160.0"),
        objectives=(pareto_replay.ObjectiveSpec("metric__rmse_total", "minimize", "rmse_total"),),
        model_variant="float_model",
        checkpoint_path="/tmp/checkpoint.keras",
    )
    assert candidate.family_hparams == {"num_layers": 2, "kernel_sizes": [3, 5]}
    assert candidate.runtime_metadata == {"batch_size": 1, "timesteps": 8, "input_dim": 6, "flops": 1000}
    assert candidate.device_options_overrides == {"cpu_clock_mhz": 160}
    assert candidate.model_variant == "float_model"
    assert candidate.checkpoint_path == "/tmp/checkpoint.keras"


def test_build_candidate_rejects_missing_runtime_metadata() -> None:
    """Check strict selected-row validation for runtime metadata.

    Returns
    -------
    None
        The test passes when missing runtime fields fail before HIL.
    """

    row = base_row(**{"hparam__input_dim": None})
    with pytest.raises(ValueError, match="input_dim"):
        pareto_replay.build_replay_candidate(
            source_row_index=0,
            row=row,
            objectives=(pareto_replay.ObjectiveSpec("metric__rmse_total", "minimize", "rmse_total"),),
        )


def test_device_option_policies_preserve_omit_and_reject() -> None:
    """Check CPU-clock replay policies and malformed selected-row errors.

    Returns
    -------
    None
        The test passes when device option policies behave as documented.
    """

    objective = (pareto_replay.ObjectiveSpec("metric__rmse_total", "minimize", "rmse_total"),)
    preserved = pareto_replay.build_replay_candidate(
        source_row_index=0,
        row=base_row(cpu_clock_mhz_requested="80"),
        objectives=objective,
        device_option_policy="preserve-source",
    )
    omitted = pareto_replay.build_replay_candidate(
        source_row_index=0,
        row=base_row(cpu_clock_mhz_requested="80"),
        objectives=objective,
        device_option_policy="target-default",
    )
    assert preserved.device_options_overrides == {"cpu_clock_mhz": 80}
    assert omitted.device_options_overrides is None
    assert preserved.payload_key == omitted.payload_key
    assert preserved.replay_payload_key != omitted.replay_payload_key

    with pytest.raises(ValueError, match="malformed cpu_clock"):
        pareto_replay.build_replay_candidate(
            source_row_index=7,
            row=base_row(cpu_clock_mhz_requested="fast"),
            objectives=objective,
            device_option_policy="preserve-source",
        )
    with pytest.raises(ValueError, match="non-integer"):
        pareto_replay.build_replay_candidate(
            source_row_index=8,
            row=base_row(cpu_clock_mhz_requested="80.5"),
            objectives=objective,
            device_option_policy="preserve-source",
        )


def test_dedupe_preserves_order_and_completed_keys_include_dry_run(tmp_path: Path) -> None:
    """Check replay dedupe ordering and resume key loading.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when duplicate payloads and resume statuses are handled.
    """

    objective = (pareto_replay.ObjectiveSpec("metric__rmse_total", "minimize", "rmse_total"),)
    first = pareto_replay.build_replay_candidate(source_row_index=0, row=base_row(), objectives=objective)
    duplicate = pareto_replay.build_replay_candidate(source_row_index=1, row=base_row(), objectives=objective)
    different = pareto_replay.build_replay_candidate(
        source_row_index=2,
        row=base_row(**{"hparam__num_layers": "3"}),
        objectives=objective,
    )
    same_legacy_different_replay = pareto_replay.build_replay_candidate(
        source_row_index=3,
        row=base_row(),
        objectives=objective,
        device_option_policy="target-default",
    )
    assert pareto_replay.dedupe_candidates([first, duplicate, different]) == [first, different]

    results_path = tmp_path / "replay_results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["payload_key", "replay_payload_key", "replay_status"])
        writer.writeheader()
        writer.writerow({"payload_key": first.payload_key, "replay_payload_key": "", "replay_status": "completed"})
        writer.writerow({"payload_key": "legacy-b", "replay_payload_key": "", "replay_status": "dry_run"})
        writer.writerow({"payload_key": "legacy-c", "replay_payload_key": "", "replay_status": "runner_error"})
        writer.writerow(
            {
                "payload_key": same_legacy_different_replay.payload_key,
                "replay_payload_key": same_legacy_different_replay.replay_payload_key,
                "replay_status": "completed",
            }
        )
    completed = pareto_replay.load_completed_payload_keys(results_path)
    assert completed.legacy_payload_keys == frozenset({first.payload_key, "legacy-b"})
    assert completed.replay_payload_keys == frozenset({same_legacy_different_replay.replay_payload_key})
    assert completed.contains(first)
    assert completed.contains(same_legacy_different_replay)


def test_positive_int_rejects_non_positive_max_candidates() -> None:
    """Check CLI validation for non-positive candidate caps.

    Returns
    -------
    None
        The test passes when argparse rejects zero and negative values.
    """

    parser = pareto_hil_replay.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-run-dir", "x", "--target-config", "y", "--max-candidates", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-run-dir", "x", "--target-config", "y", "--max-candidates", "-1"])


def test_result_csv_header_expands_for_heterogeneous_metrics(tmp_path: Path) -> None:
    """Check result CSV header expansion across heterogeneous metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when later metric keys extend the CSV header.
    """

    objective = (pareto_replay.ObjectiveSpec("metric__rmse_total", "minimize", "rmse_total"),)
    candidate = pareto_replay.build_replay_candidate(source_row_index=0, row=base_row(), objectives=objective)
    writer = pareto_replay.ResultCsvWriter(tmp_path / "replay_results.csv")
    writer.append(
        pareto_replay.flatten_result_row(
            candidate=candidate,
            source_run_dir=tmp_path,
            target_config_path=tmp_path / "target.yaml",
            replay_status="completed",
            metrics={"a": 1},
        )
    )
    writer.append(
        pareto_replay.flatten_result_row(
            candidate=candidate,
            source_run_dir=tmp_path,
            target_config_path=tmp_path / "target.yaml",
            replay_status="completed",
            metrics={"b": 2},
        )
    )
    with (tmp_path / "replay_results.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert "target__a" in reader.fieldnames
    assert "target__b" in reader.fieldnames
    assert rows[0]["target__b"] == ""
    assert rows[1]["target__b"] == "2"


def test_dry_run_writes_manifest_requests_and_results_without_server(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Check dry-run artifacts and hardware-free server avoidance.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    caplog : pytest.LogCaptureFixture
        Pytest log capture fixture used to verify replay summary logging.

    Returns
    -------
    None
        The test passes when dry-run output and summary logs are complete
        without server creation.
    """

    source_dir, _, _ = write_source_run(tmp_path, [base_row()])
    target_config = write_target_config(tmp_path)
    output_dir = tmp_path / "replay"
    args = make_args(source_run_dir=source_dir, target_config=target_config, output_dir=output_dir, dry_run=True)

    def forbidden_factory(config_path: Path) -> object:
        """Fail if dry-run tries to instantiate a server.

        Parameters
        ----------
        config_path : pathlib.Path
            Target config path.

        Returns
        -------
        object
            This function never returns in the passing path.
        """

        raise AssertionError(f"Unexpected HIL server construction for {config_path}")

    with caplog.at_level(logging.INFO, logger="tinyodom.pareto_replay"):
        assert pareto_replay.run_replay(args, server_factory=forbidden_factory) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    requests = read_request_records(output_dir)
    results = read_results(output_dir)
    log_messages = [record.getMessage() for record in caplog.records if record.name == "tinyodom.pareto_replay"]
    assert "Source rows: 1" in log_messages
    assert "Selected Pareto rows: 1" in log_messages
    assert "Scheduled replay candidates: 1" in log_messages
    assert f"Wrote replay outputs: {output_dir}" in log_messages
    assert manifest["entrypoint"] == "src/pareto_hil_replay.py"
    assert manifest["scheduled_candidates"] == 1
    assert manifest["args"] == {
        "source_run_dir": str(source_dir),
        "target_run_dir": None,
        "target_config": str(target_config),
        "source_csv": None,
        "source_config": None,
        "objectives": None,
        "output_dir": str(output_dir),
        "max_candidates": None,
        "dry_run": True,
        "resume": False,
        "allow_gpu": False,
        "device_option_policy": "preserve-source",
        "model_variant": None,
        "checkpoint_path": None,
    }
    assert (output_dir / "replay_requests.jsonl").exists()
    assert (output_dir / "replay_results.csv").exists()
    assert results[0]["replay_status"] == "dry_run"
    assert requests[0] == {
        "ordinal": 1,
        "source_row_index": 0,
        "payload_key": requests[0]["payload_key"],
        "replay_payload_key": requests[0]["replay_payload_key"],
        "family_hparams": {"kernel_sizes": [3, 5], "num_layers": 2},
        "runtime_metadata": {"batch_size": 1, "flops": 1000, "input_dim": 6, "timesteps": 8},
        "quantization_mode": "float",
        "device_options_overrides": {"cpu_clock_mhz": 80},
        "model_variant": None,
        "checkpoint_path": None,
    }


def test_non_dry_run_invokes_metrics_callback_with_full_request(tmp_path: Path) -> None:
    """Check hardware-free non-dry-run metrics callback payloads.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when all reconstructed fields reach the callback.
    """

    source_dir, _, _ = write_source_run(tmp_path, [base_row(quantization_mode="int8_ptq")])
    target_config = write_target_config(tmp_path)
    output_dir = tmp_path / "replay"
    args = make_args(
        source_run_dir=source_dir,
        target_config=target_config,
        output_dir=output_dir,
        dry_run=False,
        model_variant="int8_model",
        checkpoint_path="/tmp/model.keras",
    )
    calls: list[dict[str, Any]] = []

    def metrics_callback(
        family_hparams: Mapping[str, Any],
        runtime_metadata: Mapping[str, Any],
        *,
        quantization_mode: str | None = None,
        device_options_overrides: Mapping[str, Any] | None = None,
        checkpoint_path: str | None = None,
        model_variant: str | None = None,
    ) -> Mapping[str, Any]:
        """Record a replay metrics request.

        Parameters
        ----------
        family_hparams : Mapping[str, Any]
            Reconstructed family hparams.
        runtime_metadata : Mapping[str, Any]
            Reconstructed runtime metadata.
        quantization_mode : str | None, optional
            Reconstructed quantization mode.
        device_options_overrides : Mapping[str, Any] | None, optional
            Optional device options.
        checkpoint_path : str | None, optional
            Optional checkpoint path.
        model_variant : str | None, optional
            Optional model variant.

        Returns
        -------
        Mapping[str, Any]
            Synthetic HIL metrics.
        """

        calls.append(
            {
                "family_hparams": dict(family_hparams),
                "runtime_metadata": dict(runtime_metadata),
                "quantization_mode": quantization_mode,
                "device_options_overrides": dict(device_options_overrides or {}),
                "checkpoint_path": checkpoint_path,
                "model_variant": model_variant,
            }
        )
        return {"error_code": 1, "energy_mj_per_inference": 2.5}

    assert pareto_replay.run_replay(args, metrics_callback=metrics_callback) == 0
    assert calls == [
        {
            "family_hparams": {"kernel_sizes": [3, 5], "num_layers": 2},
            "runtime_metadata": {"batch_size": 1, "timesteps": 8, "input_dim": 6, "flops": 1000},
            "quantization_mode": "int8_ptq",
            "device_options_overrides": {"cpu_clock_mhz": 80},
            "checkpoint_path": "/tmp/model.keras",
            "model_variant": "int8_model",
        }
    ]
    results = read_results(output_dir)
    assert results[0]["replay_status"] == "completed"
    assert results[0]["target__energy_mj_per_inference"] == "2.5"


def test_runner_errors_become_result_rows(tmp_path: Path) -> None:
    """Check that backend exceptions are recorded as runner-error rows.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when errors do not crash the replay loop.
    """

    source_dir, _, _ = write_source_run(tmp_path, [base_row()])
    target_config = write_target_config(tmp_path)
    output_dir = tmp_path / "replay"
    args = make_args(source_run_dir=source_dir, target_config=target_config, output_dir=output_dir, dry_run=False)

    def failing_callback(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Raise a synthetic metrics failure.

        Parameters
        ----------
        *args : Any
            Positional callback arguments.
        **kwargs : Any
            Keyword callback arguments.

        Returns
        -------
        Mapping[str, Any]
            This function never returns in the passing path.
        """

        raise RuntimeError("synthetic failure")

    assert pareto_replay.run_replay(args, metrics_callback=failing_callback) == 0
    results = read_results(output_dir)
    assert results[0]["replay_status"] == "runner_error"
    assert "synthetic failure" in results[0]["error_detail"]


def test_resume_skips_completed_and_dry_run_payloads(tmp_path: Path) -> None:
    """Check resume semantics for dry-run payload keys.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.

    Returns
    -------
    None
        The test passes when a dry-run output directory is skipped on resume.
    """

    source_dir, _, _ = write_source_run(tmp_path, [base_row()])
    target_config = write_target_config(tmp_path)
    output_dir = tmp_path / "replay"
    dry_args = make_args(source_run_dir=source_dir, target_config=target_config, output_dir=output_dir, dry_run=True)
    assert pareto_replay.run_replay(dry_args) == 0

    calls: list[str] = []

    def unexpected_callback(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Record an unexpected resume callback.

        Parameters
        ----------
        *args : Any
            Positional callback arguments.
        **kwargs : Any
            Keyword callback arguments.

        Returns
        -------
        Mapping[str, Any]
            Synthetic metrics mapping.
        """

        calls.append("called")
        return {"error_code": 1}

    resume_args = make_args(
        source_run_dir=source_dir,
        target_config=target_config,
        output_dir=output_dir,
        dry_run=False,
        resume=True,
    )
    assert pareto_replay.run_replay(resume_args, metrics_callback=unexpected_callback) == 0
    assert calls == []
    assert len(read_results(output_dir)) == 1


def test_gpu_environment_behavior_matches_allow_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check default CUDA clearing and allow-gpu preservation.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        The test passes when CUDA visibility follows CLI policy.
    """

    source_dir, _, _ = write_source_run(tmp_path, [base_row()])
    target_config = write_target_config(tmp_path)

    def metrics_callback(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        """Return successful synthetic metrics.

        Parameters
        ----------
        *args : Any
            Positional callback arguments.
        **kwargs : Any
            Keyword callback arguments.

        Returns
        -------
        Mapping[str, Any]
            Synthetic metrics mapping.
        """

        return {"error_code": 1}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    default_args = make_args(
        source_run_dir=source_dir,
        target_config=target_config,
        output_dir=tmp_path / "default_replay",
        dry_run=False,
        allow_gpu=False,
    )
    pareto_replay.run_replay(default_args, metrics_callback=metrics_callback)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    allow_args = make_args(
        source_run_dir=source_dir,
        target_config=target_config,
        output_dir=tmp_path / "allow_replay",
        dry_run=False,
        allow_gpu=True,
    )
    pareto_replay.run_replay(allow_args, metrics_callback=metrics_callback)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_cli_help_smoke() -> None:
    """Check that the top-level replay CLI imports and renders help.

    Returns
    -------
    None
        The test passes when the CLI exits successfully for ``--help``.
    """

    result = subprocess.run(
        [sys.executable, "src/pareto_hil_replay.py", "--help"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "exactly one target option" in result.stdout
    assert "required arguments:" in result.stdout
    for flag in [
        "--source-run-dir",
        "--target-run-dir",
        "--target-config",
        "--source-csv",
        "--source-config",
        "--objectives",
        "--output-dir",
        "--max-candidates",
        "--dry-run",
        "--resume",
        "--allow-gpu",
        "--device-option-policy",
        "--model-variant",
        "--checkpoint-path",
    ]:
        assert flag in result.stdout
    assert "Examples:" in result.stdout
    assert "Dry-run payload preflight:" in result.stdout
    assert "Hardware replay with resume:" in result.stdout
