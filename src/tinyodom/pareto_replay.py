"""Pareto-front HIL replay utilities for TinyODOM."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SUCCESS_ERROR_CODE = 1
PENALTY_ABS_THRESHOLD = 1.0e11
RUNTIME_METADATA_KEYS = {"flops", "batch_size", "timesteps", "input_dim"}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "models" / "replays"
DEVICE_OPTION_POLICY_CHOICES = ("preserve-source", "target-default")

MetricsCallback = Callable[..., Mapping[str, Any]]
ServerFactory = Callable[[Path], Any]


@dataclass(frozen=True)
class ObjectiveSpec:
    """Resolved objective column and direction used for Pareto selection.

    Parameters
    ----------
    column : str
        CSV column containing the objective values.
    direction : str
        Optimization direction, either ``"minimize"`` or ``"maximize"``.
    metric : str
        Config or log metric name that resolved to ``column``.
    """

    column: str
    direction: str
    metric: str


@dataclass(frozen=True)
class ReplayCandidate:
    """One source NAS candidate reconstructed for HIL replay.

    Parameters
    ----------
    source_row_index : int
        Zero-based CSV row index from the source run.
    source_row : dict[str, Any]
        Raw source CSV row.
    family_hparams : dict[str, Any]
        Model-family-owned hparams for ``HILServer.determine_metrics``.
    runtime_metadata : dict[str, Any]
        Runtime-owned metadata for ``HILServer.determine_metrics``.
    quantization_mode : str
        Candidate deployment quantization mode.
    device_options_overrides : dict[str, Any] | None
        Optional runner-owned device options reconstructed from the source row.
    model_variant : str | None
        Optional model variant override forwarded to HIL.
    checkpoint_path : str | None
        Optional checkpoint path forwarded to HIL.
    objective_values : dict[str, float]
        Selected objective values keyed by objective CSV column.
    payload_key : str
        Legacy stable JSON key preserved for artifact compatibility.
    replay_payload_key : str
        Runtime-option-aware stable JSON key used for new dedupe/resume checks.
    """

    source_row_index: int
    source_row: dict[str, Any]
    family_hparams: dict[str, Any]
    runtime_metadata: dict[str, Any]
    quantization_mode: str
    device_options_overrides: dict[str, Any] | None
    model_variant: str | None
    checkpoint_path: str | None
    objective_values: dict[str, float]
    payload_key: str
    replay_payload_key: str


@dataclass(frozen=True)
class CompletedReplayKeys:
    """Completed replay payload keys loaded from a results CSV.

    Parameters
    ----------
    replay_payload_keys : frozenset[str]
        New runtime-option-aware keys loaded from rows with
        ``replay_payload_key``.
    legacy_payload_keys : frozenset[str]
        Legacy keys loaded from older rows that only have ``payload_key``.
    """

    replay_payload_keys: frozenset[str]
    legacy_payload_keys: frozenset[str]

    def contains(self, candidate: ReplayCandidate) -> bool:
        """Return whether a candidate has already completed.

        Parameters
        ----------
        candidate : ReplayCandidate
            Candidate to check against loaded completion keys.

        Returns
        -------
        bool
            ``True`` when the candidate should be skipped for resume.
        """

        return (
            candidate.replay_payload_key in self.replay_payload_keys
            or candidate.payload_key in self.legacy_payload_keys
        )


@dataclass(frozen=True)
class ReplayRunConfig:
    """Configuration for one Pareto HIL replay run.

    Parameters
    ----------
    source_run_dir : pathlib.Path
        Source NAS run directory.
    target_run_dir : pathlib.Path | None
        Target run directory containing a target config.
    target_config : pathlib.Path | None
        Explicit target config path.
    source_csv : pathlib.Path | None
        Optional source CSV override.
    source_config : pathlib.Path | None
        Optional source config override.
    objectives : str | None
        Optional source objective override string.
    output_dir : pathlib.Path | None
        Optional replay output directory.
    max_candidates : int | None
        Optional positive cap after Pareto selection and dedupe.
    dry_run : bool
        Whether to write replay payloads without running HIL.
    resume : bool
        Whether to skip payloads already present in the result CSV.
    allow_gpu : bool
        Whether to leave ``CUDA_VISIBLE_DEVICES`` untouched for HIL execution.
    device_option_policy : str
        Device option replay policy.
    model_variant : str | None
        Optional model variant forwarded to HIL.
    checkpoint_path : str | None
        Optional checkpoint path forwarded to HIL.
    """

    source_run_dir: Path
    target_run_dir: Path | None
    target_config: Path | None
    source_csv: Path | None
    source_config: Path | None
    objectives: str | None
    output_dir: Path | None
    max_candidates: int | None
    dry_run: bool
    resume: bool
    allow_gpu: bool
    device_option_policy: str
    model_variant: str | None
    checkpoint_path: str | None

    def __post_init__(self) -> None:
        """Validate direct-library replay configuration.

        Returns
        -------
        None
            The method raises ``ValueError`` for invalid configurations.
        """

        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer when provided.")
        if self.target_run_dir is None and self.target_config is None:
            raise ValueError("Provide target_run_dir or target_config.")
        if self.device_option_policy not in DEVICE_OPTION_POLICY_CHOICES:
            raise ValueError(f"Unsupported device option policy: {self.device_option_policy!r}")


def normalize_direction(value: Any) -> str:
    """Normalize an objective direction string.

    Parameters
    ----------
    value : Any
        Direction-like value from config, CSV metadata, or CLI.

    Returns
    -------
    str
        ``"minimize"`` or ``"maximize"``.
    """

    text = str(value).strip().lower()
    if text in {"minimize", "min"}:
        return "minimize"
    if text in {"maximize", "max"}:
        return "maximize"
    raise ValueError(f"Unsupported objective direction: {value!r}")


def parse_bool(value: Any) -> bool:
    """Parse loose CSV/config boolean values.

    Parameters
    ----------
    value : Any
        Value to parse.

    Returns
    -------
    bool
        Parsed boolean value.
    """

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", "", "nan", "none", "null"}:
        return False
    return bool(value)


def parse_cell_value(value: Any) -> Any:
    """Parse one CSV cell into a JSON-compatible Python value.

    Parameters
    ----------
    value : Any
        Raw CSV cell value.

    Returns
    -------
    Any
        Parsed scalar/list value suitable for HIL request payloads.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "nan"}:
        return None
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    try:
        parsed_int = int(text)
    except ValueError:
        parsed_int = None
    if parsed_int is not None and str(parsed_int) == text:
        return parsed_int
    try:
        parsed_float = float(text)
    except ValueError:
        return text
    if math.isfinite(parsed_float):
        return parsed_float
    return None


def parse_finite_float(value: Any) -> float | None:
    """Parse a finite float from a loose CSV value.

    Parameters
    ----------
    value : Any
        Raw value to parse.

    Returns
    -------
    float | None
        Parsed finite float, or ``None``.
    """

    parsed = parse_cell_value(value)
    if isinstance(parsed, bool) or parsed is None:
        return None
    try:
        numeric = float(parsed)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def metric_column(columns: Sequence[str], metric: str) -> str | None:
    """Resolve a config/log metric name to a CSV column.

    Parameters
    ----------
    columns : Sequence[str]
        Available CSV columns.
    metric : str
        Config or log metric name.

    Returns
    -------
    str | None
        Matching column, or ``None`` when unavailable.
    """

    available = set(columns)
    candidates = [metric]
    if metric in {"rmse_total", "aggregate_rmse"}:
        candidates = ["metric__rmse_total", "rmse_total", "aggregate_rmse"]
    elif not metric.startswith("metric__"):
        candidates.append(f"metric__{metric}")
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def parse_objective_override(value: str | None, columns: Sequence[str]) -> tuple[ObjectiveSpec, ...] | None:
    """Parse a CLI objective override.

    Parameters
    ----------
    value : str | None
        Comma-separated ``metric_or_column:direction`` override.
    columns : Sequence[str]
        Available CSV columns used for metric-to-column resolution.

    Returns
    -------
    tuple[ObjectiveSpec, ...] | None
        Parsed objective specs, or ``None`` when no override was provided.
    """

    if value is None or not value.strip():
        return None
    specs: list[ObjectiveSpec] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Objective override must use column:direction syntax: {part!r}")
        metric, direction = part.split(":", 1)
        metric = metric.strip()
        column = metric_column(columns, metric)
        if column is None:
            raise ValueError(f"Objective column for {metric!r} was not found in the source CSV.")
        specs.append(
            ObjectiveSpec(
                column=column,
                direction=normalize_direction(direction),
                metric=metric,
            )
        )
    if not specs:
        raise ValueError("At least one objective override is required when --objectives is provided.")
    return tuple(specs)


def parse_json_list(value: Any) -> list[Any]:
    """Parse a JSON list field from a NAS log row.

    Parameters
    ----------
    value : Any
        Raw CSV cell value.

    Returns
    -------
    list[Any]
        Parsed JSON list, or an empty list.
    """

    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        return []
    return parsed


def resolve_objective_specs(
    *,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    override: str | None = None,
) -> tuple[ObjectiveSpec, ...]:
    """Resolve objective specs from CLI, config, or log metadata.

    Parameters
    ----------
    config : Mapping[str, Any]
        Parsed source run config.
    rows : Sequence[Mapping[str, Any]]
        Source CSV rows.
    columns : Sequence[str]
        Source CSV columns.
    override : str | None, optional
        Optional CLI override.

    Returns
    -------
    tuple[ObjectiveSpec, ...]
        Objective specs used for Pareto-front selection.
    """

    overridden = parse_objective_override(override, columns)
    if overridden is not None:
        return overridden

    nas = config.get("nas", {}) if isinstance(config.get("nas"), Mapping) else {}
    score = nas.get("score", {}) if isinstance(nas.get("score"), Mapping) else {}
    params = score.get("params", {}) if isinstance(score.get("params"), Mapping) else {}
    objectives = params.get("objectives", [])
    specs: list[ObjectiveSpec] = []
    if isinstance(objectives, list):
        for objective in objectives:
            if not isinstance(objective, Mapping):
                continue
            metric = str(objective.get("metric", "")).strip()
            if not metric:
                continue
            column = metric_column(columns, metric)
            if column is None:
                continue
            specs.append(
                ObjectiveSpec(
                    column=column,
                    direction=normalize_direction(objective.get("direction", "minimize")),
                    metric=metric,
                )
            )
    if specs:
        return tuple(specs)

    for row in rows:
        try:
            names = parse_json_list(row.get("objective_names_json"))
            directions = parse_json_list(row.get("objective_directions_json"))
        except json.JSONDecodeError:
            continue
        for metric, direction in zip(names, directions):
            column = metric_column(columns, str(metric))
            if column is None:
                continue
            specs.append(
                ObjectiveSpec(
                    column=column,
                    direction=normalize_direction(direction),
                    metric=str(metric),
                )
            )
        if specs:
            return tuple(specs)

    raise ValueError("Unable to infer source objective columns. Pass --objectives explicitly.")


def load_yaml_config(path: Path | None) -> dict[str, Any]:
    """Load a YAML config mapping.

    Parameters
    ----------
    path : pathlib.Path | None
        Config path to load.

    Returns
    -------
    dict[str, Any]
        Parsed config mapping.
    """

    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return loaded


def find_config_path(run_dir: Path) -> Path | None:
    """Find a NAS config in a run directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run directory to inspect.

    Returns
    -------
    pathlib.Path | None
        Resolved config path, or ``None``.
    """

    exact = run_dir / "nas_config.yaml"
    if exact.exists():
        return exact
    candidates = sorted(run_dir.glob("nas_config*.yaml"))
    return candidates[0] if candidates else None


def find_csv_path(run_dir: Path, config: Mapping[str, Any]) -> Path:
    """Find the NAS log CSV in a run directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Source run directory.
    config : Mapping[str, Any]
        Parsed source config.

    Returns
    -------
    pathlib.Path
        Source NAS log path.
    """

    outputs = config.get("outputs", {}) if isinstance(config.get("outputs"), Mapping) else {}
    log_name = outputs.get("log_file_name")
    if log_name:
        configured = run_dir / str(log_name)
        if configured.exists():
            return configured
    candidates = sorted(run_dir.glob("log_NAS_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No log_NAS_*.csv file found in {run_dir}")
    return candidates[0]


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read CSV rows as dictionaries.

    Parameters
    ----------
    path : pathlib.Path
        CSV file path.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Rows and fieldnames.
    """

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows, list(reader.fieldnames)


def is_source_row_valid(row: Mapping[str, Any], objectives: Sequence[ObjectiveSpec]) -> bool:
    """Return whether a source row is eligible for replay selection.

    Parameters
    ----------
    row : Mapping[str, Any]
        Source CSV row.
    objectives : Sequence[ObjectiveSpec]
        Objectives that must be finite and non-penalty.

    Returns
    -------
    bool
        ``True`` when the row can be considered for Pareto selection.
    """

    if parse_bool(row.get("pruned")):
        return False
    error_code = parse_finite_float(row.get("error_code"))
    if error_code is not None and int(error_code) != SUCCESS_ERROR_CODE:
        return False
    quantization_mode = str(row.get("quantization_mode", "")).strip().lower()
    if quantization_mode not in {"float", "int8_ptq"}:
        return False
    for objective in objectives:
        value = parse_finite_float(row.get(objective.column))
        if value is None:
            return False
        if abs(value) >= PENALTY_ABS_THRESHOLD:
            return False
        if value <= -1000.0:
            return False
    flops = parse_finite_float(row.get("flops"))
    if flops is None or flops <= 0.0:
        return False
    return True


def dominates(left: Sequence[float], right: Sequence[float], objectives: Sequence[ObjectiveSpec]) -> bool:
    """Return whether ``left`` dominates ``right`` under objective directions.

    Parameters
    ----------
    left : Sequence[float]
        Candidate objective values.
    right : Sequence[float]
        Candidate objective values to compare against.
    objectives : Sequence[ObjectiveSpec]
        Objective direction metadata.

    Returns
    -------
    bool
        ``True`` when ``left`` is no worse on all objectives and better on one.
    """

    no_worse = True
    strictly_better = False
    for left_value, right_value, objective in zip(left, right, objectives):
        if objective.direction == "minimize":
            if left_value > right_value:
                no_worse = False
                break
            if left_value < right_value:
                strictly_better = True
        else:
            if left_value < right_value:
                no_worse = False
                break
            if left_value > right_value:
                strictly_better = True
    return no_worse and strictly_better


def pareto_indices(
    rows: Sequence[Mapping[str, Any]],
    objectives: Sequence[ObjectiveSpec],
) -> list[int]:
    """Return CSV indices belonging to the valid source Pareto front.

    Parameters
    ----------
    rows : Sequence[Mapping[str, Any]]
        Source CSV rows.
    objectives : Sequence[ObjectiveSpec]
        Objective specs.

    Returns
    -------
    list[int]
        Zero-based indices of valid non-dominated rows.
    """

    valid_values: list[tuple[int, tuple[float, ...]]] = []
    for index, row in enumerate(rows):
        if not is_source_row_valid(row, objectives):
            continue
        values = tuple(float(parse_finite_float(row[objective.column])) for objective in objectives)
        valid_values.append((index, values))

    front: list[int] = []
    for index, values in valid_values:
        if any(
            other_index != index and dominates(other_values, values, objectives)
            for other_index, other_values in valid_values
        ):
            continue
        front.append(index)
    return front


def is_missing_device_option_value(value: Any) -> bool:
    """Return whether a source device option cell means no override.

    Parameters
    ----------
    value : Any
        Raw source CSV cell value.

    Returns
    -------
    bool
        ``True`` when the value is empty or a known sentinel.
    """

    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "-1", "-1.0"}


def resolve_device_options_overrides(
    *,
    row: Mapping[str, Any],
    policy: str,
    source_row_index: int,
) -> dict[str, Any] | None:
    """Resolve replay device option overrides from a source row.

    Parameters
    ----------
    row : Mapping[str, Any]
        Source CSV row.
    policy : str
        Device option policy, either ``"preserve-source"`` or
        ``"target-default"``.
    source_row_index : int
        Zero-based source row index used in validation messages.

    Returns
    -------
    dict[str, Any] | None
        Device option override payload, or ``None`` when omitted.
    """

    if policy == "target-default":
        return None
    if policy != "preserve-source":
        raise ValueError(f"Unsupported device option policy: {policy!r}")

    raw_cpu_clock = row.get("cpu_clock_mhz_requested")
    if is_missing_device_option_value(raw_cpu_clock):
        return None
    numeric = parse_finite_float(raw_cpu_clock)
    # CPU clock replay is runner-owned state, so malformed selected rows should
    # fail before any hardware call instead of silently changing the payload.
    if numeric is None:
        raise ValueError(
            f"Source row {source_row_index} has malformed cpu_clock_mhz_requested: {raw_cpu_clock!r}"
        )
    if numeric <= 0.0:
        return None
    if not float(numeric).is_integer():
        raise ValueError(
            f"Source row {source_row_index} has non-integer cpu_clock_mhz_requested: {raw_cpu_clock!r}"
        )
    return {"cpu_clock_mhz": int(numeric)}


def stable_payload_key(
    *,
    family_hparams: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
    quantization_mode: str,
    device_options_overrides: Mapping[str, Any] | None,
    model_variant: str | None,
    checkpoint_path: str | None,
) -> str:
    """Build the stable replay payload identity.

    Parameters
    ----------
    family_hparams : Mapping[str, Any]
        Model-family-owned hparams.
    runtime_metadata : Mapping[str, Any]
        Runtime-owned metadata.
    quantization_mode : str
        Candidate quantization mode.
    device_options_overrides : Mapping[str, Any] | None
        Optional device option overrides.
    model_variant : str | None
        Optional model variant override.
    checkpoint_path : str | None
        Optional checkpoint path override.

    Returns
    -------
    str
        Deterministic JSON payload key.
    """

    return json.dumps(
        {
            "family_hparams": dict(family_hparams),
            "runtime_metadata": dict(runtime_metadata),
            "quantization_mode": quantization_mode,
            "device_options_overrides": dict(device_options_overrides or {}),
            "model_variant": model_variant,
            "checkpoint_path": checkpoint_path,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def legacy_payload_key(
    *,
    family_hparams: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
    quantization_mode: str,
) -> str:
    """Build the legacy replay payload identity.

    Parameters
    ----------
    family_hparams : Mapping[str, Any]
        Model-family-owned hparams.
    runtime_metadata : Mapping[str, Any]
        Runtime-owned metadata.
    quantization_mode : str
        Candidate quantization mode.

    Returns
    -------
    str
        Deterministic JSON payload key matching pre-migration artifacts.
    """

    return json.dumps(
        {
            "family_hparams": dict(family_hparams),
            "runtime_metadata": dict(runtime_metadata),
            "quantization_mode": quantization_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def build_replay_candidate(
    *,
    source_row_index: int,
    row: Mapping[str, Any],
    objectives: Sequence[ObjectiveSpec],
    device_option_policy: str = "preserve-source",
    model_variant: str | None = None,
    checkpoint_path: str | None = None,
) -> ReplayCandidate:
    """Reconstruct one HIL replay candidate from a source CSV row.

    Parameters
    ----------
    source_row_index : int
        Zero-based source CSV row index.
    row : Mapping[str, Any]
        Source CSV row.
    objectives : Sequence[ObjectiveSpec]
        Objective specs used to copy objective values.
    device_option_policy : str, optional
        Policy for replaying logged device options.
    model_variant : str | None, optional
        Optional model variant forwarded to HIL.
    checkpoint_path : str | None, optional
        Optional checkpoint path forwarded to HIL.

    Returns
    -------
    ReplayCandidate
        Candidate payload ready for HIL replay.
    """

    family_hparams: dict[str, Any] = {}
    runtime_metadata: dict[str, Any] = {}
    for column, raw_value in row.items():
        if not column.startswith("hparam__"):
            continue
        key = column.removeprefix("hparam__")
        value = parse_cell_value(raw_value)
        if value is None:
            continue
        if key in RUNTIME_METADATA_KEYS:
            runtime_metadata[key] = value
        else:
            family_hparams[key] = value

    flops = parse_finite_float(row.get("flops"))
    if flops is None or flops <= 0.0:
        raise ValueError(f"Source row {source_row_index} does not contain a positive flops value.")
    runtime_metadata["flops"] = int(flops) if float(flops).is_integer() else float(flops)

    missing_runtime = sorted(RUNTIME_METADATA_KEYS - set(runtime_metadata))
    if missing_runtime:
        raise ValueError(
            f"Source row {source_row_index} is missing runtime metadata: {', '.join(missing_runtime)}"
        )
    if not family_hparams:
        raise ValueError(f"Source row {source_row_index} does not contain family hparams.")

    quantization_mode = str(row.get("quantization_mode", "")).strip().lower()
    objective_values = {
        objective.column: float(parse_finite_float(row[objective.column]))
        for objective in objectives
    }
    device_options_overrides = resolve_device_options_overrides(
        row=row,
        policy=device_option_policy,
        source_row_index=source_row_index,
    )
    payload_key = legacy_payload_key(
        family_hparams=family_hparams,
        runtime_metadata=runtime_metadata,
        quantization_mode=quantization_mode,
    )
    replay_payload_key = stable_payload_key(
        family_hparams=family_hparams,
        runtime_metadata=runtime_metadata,
        quantization_mode=quantization_mode,
        device_options_overrides=device_options_overrides,
        model_variant=model_variant,
        checkpoint_path=checkpoint_path,
    )
    return ReplayCandidate(
        source_row_index=source_row_index,
        source_row=dict(row),
        family_hparams=family_hparams,
        runtime_metadata=runtime_metadata,
        quantization_mode=quantization_mode,
        device_options_overrides=device_options_overrides,
        model_variant=model_variant,
        checkpoint_path=checkpoint_path,
        objective_values=objective_values,
        payload_key=payload_key,
        replay_payload_key=replay_payload_key,
    )


def dedupe_candidates(candidates: Iterable[ReplayCandidate]) -> list[ReplayCandidate]:
    """Deduplicate exact replay payloads while preserving order.

    Parameters
    ----------
    candidates : Iterable[ReplayCandidate]
        Candidate payloads.

    Returns
    -------
    list[ReplayCandidate]
        Deduplicated candidates.
    """

    seen: set[str] = set()
    deduped: list[ReplayCandidate] = []
    for candidate in candidates:
        if candidate.replay_payload_key in seen:
            continue
        seen.add(candidate.replay_payload_key)
        deduped.append(candidate)
    return deduped


def load_completed_payload_keys(results_path: Path) -> CompletedReplayKeys:
    """Load completed replay payload keys from an existing results CSV.

    Parameters
    ----------
    results_path : pathlib.Path
        Existing replay results path.

    Returns
    -------
    CompletedReplayKeys
        New-format and legacy completed payload keys.
    """

    if not results_path.exists():
        return CompletedReplayKeys(replay_payload_keys=frozenset(), legacy_payload_keys=frozenset())
    replay_payload_keys: set[str] = set()
    legacy_payload_keys: set[str] = set()
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("replay_status") not in {"completed", "dry_run"}:
                continue
            if row.get("replay_payload_key"):
                replay_payload_keys.add(row["replay_payload_key"])
            elif row.get("payload_key"):
                legacy_payload_keys.add(row["payload_key"])
    return CompletedReplayKeys(
        replay_payload_keys=frozenset(replay_payload_keys),
        legacy_payload_keys=frozenset(legacy_payload_keys),
    )


def default_output_dir(source_run_dir: Path, target_config_path: Path, timestamp: str) -> Path:
    """Build the default replay output directory path.

    Parameters
    ----------
    source_run_dir : pathlib.Path
        Source run directory.
    target_config_path : pathlib.Path
        Target config path.
    timestamp : str
        Timestamp suffix.

    Returns
    -------
    pathlib.Path
        Default replay artifact directory.
    """

    target_parent = target_config_path.parent
    target_name = target_parent.name if target_parent != REPO_ROOT else target_config_path.stem
    return DEFAULT_OUTPUT_ROOT / f"{source_run_dir.name}__on__{target_name}_{timestamp}"


def replay_config_to_manifest_args(config: ReplayRunConfig) -> dict[str, Any]:
    """Serialize replay config into CLI-shaped manifest arguments.

    Parameters
    ----------
    config : ReplayRunConfig
        Replay run configuration.

    Returns
    -------
    dict[str, Any]
        JSON-serializable mapping using current CLI argument names.
    """

    return {
        "source_run_dir": str(config.source_run_dir),
        "target_run_dir": str(config.target_run_dir) if config.target_run_dir else None,
        "target_config": str(config.target_config) if config.target_config else None,
        "source_csv": str(config.source_csv) if config.source_csv else None,
        "source_config": str(config.source_config) if config.source_config else None,
        "objectives": config.objectives,
        "output_dir": str(config.output_dir) if config.output_dir else None,
        "max_candidates": config.max_candidates,
        "dry_run": config.dry_run,
        "resume": config.resume,
        "allow_gpu": config.allow_gpu,
        "device_option_policy": config.device_option_policy,
        "model_variant": config.model_variant,
        "checkpoint_path": config.checkpoint_path,
    }


def write_manifest(
    *,
    output_dir: Path,
    source_run_dir: Path,
    source_csv_path: Path,
    source_config_path: Path | None,
    target_config_path: Path,
    objectives: Sequence[ObjectiveSpec],
    total_rows: int,
    selected_rows: int,
    replayed_candidates: int,
    config: ReplayRunConfig,
) -> None:
    """Write a JSON manifest describing the replay run.

    Parameters
    ----------
    output_dir : pathlib.Path
        Replay output directory.
    source_run_dir : pathlib.Path
        Source run directory.
    source_csv_path : pathlib.Path
        Source log CSV path.
    source_config_path : pathlib.Path | None
        Source config path.
    target_config_path : pathlib.Path
        Target HIL config path.
    objectives : Sequence[ObjectiveSpec]
        Objective specs.
    total_rows : int
        Total source CSV rows.
    selected_rows : int
        Number of Pareto rows before dedupe/max filtering.
    replayed_candidates : int
        Number of candidates scheduled for replay.
    config : ReplayRunConfig
        Replay run configuration.
    """

    manifest = {
        "source_run_dir": str(source_run_dir),
        "source_csv_path": str(source_csv_path),
        "source_config_path": str(source_config_path) if source_config_path else None,
        "target_config_path": str(target_config_path),
        "entrypoint": "src/pareto_hil_replay.py",
        "objectives": [
            {"metric": objective.metric, "column": objective.column, "direction": objective.direction}
            for objective in objectives
        ],
        "total_source_rows": total_rows,
        "selected_pareto_rows": selected_rows,
        "scheduled_candidates": replayed_candidates,
        "dry_run": config.dry_run,
        "args": replay_config_to_manifest_args(config),
        "timestamp_unix": time.time(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object to a JSONL file.

    Parameters
    ----------
    path : pathlib.Path
        Output JSONL path.
    record : Mapping[str, Any]
        JSON-compatible record.
    """

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def flatten_result_row(
    *,
    candidate: ReplayCandidate,
    source_run_dir: Path,
    target_config_path: Path,
    replay_status: str,
    metrics: Mapping[str, Any] | None,
    error_detail: str | None = None,
) -> dict[str, Any]:
    """Build one replay results CSV row.

    Parameters
    ----------
    candidate : ReplayCandidate
        Replayed candidate.
    source_run_dir : pathlib.Path
        Source run directory.
    target_config_path : pathlib.Path
        Target config path.
    replay_status : str
        Stable replay status label.
    metrics : Mapping[str, Any] | None
        HIL metrics, if any.
    error_detail : str | None, optional
        Runner-side error detail.

    Returns
    -------
    dict[str, Any]
        Flattened CSV row.
    """

    row: dict[str, Any] = {
        "source_run": source_run_dir.name,
        "source_row_index": candidate.source_row_index,
        "target_config": str(target_config_path),
        "replay_status": replay_status,
        "payload_key": candidate.payload_key,
        "replay_payload_key": candidate.replay_payload_key,
        "quantization_mode": candidate.quantization_mode,
        "device_options_overrides_json": json.dumps(candidate.device_options_overrides or {}, sort_keys=True),
        "model_variant": candidate.model_variant or "",
        "checkpoint_path": candidate.checkpoint_path or "",
        "family_hparams_json": json.dumps(candidate.family_hparams, sort_keys=True),
        "runtime_metadata_json": json.dumps(candidate.runtime_metadata, sort_keys=True),
        "source_objective_values_json": json.dumps(candidate.objective_values, sort_keys=True),
        "error_detail": error_detail or "",
        "timestamp_unix": time.time(),
    }
    for column, value in candidate.source_row.items():
        row[f"source__{column}"] = value
    if metrics:
        for key, value in metrics.items():
            row[f"target__{key}"] = value
    return row


class ResultCsvWriter:
    """Append replay result rows while preserving a stable header.

    Parameters
    ----------
    path : pathlib.Path
        Results CSV path.
    """

    def __init__(self, path: Path) -> None:
        """Initialize a replay results writer.

        Parameters
        ----------
        path : pathlib.Path
            Results CSV path.
        """

        self.path = path
        self.fieldnames: list[str] | None = None
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.fieldnames = list(reader.fieldnames or [])

    def append(self, row: Mapping[str, Any]) -> None:
        """Append one row to the replay results CSV.

        Parameters
        ----------
        row : Mapping[str, Any]
            Flattened row to append.
        """

        serialized = {key: self._serialize(value) for key, value in row.items()}
        if self.fieldnames is None:
            self.fieldnames = list(serialized.keys())
        missing = [key for key in serialized if key not in self.fieldnames]
        if missing:
            self.fieldnames.extend(missing)
            if self.path.exists():
                self._rewrite_with_extended_header()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({key: serialized.get(key, "") for key in self.fieldnames})

    def _rewrite_with_extended_header(self) -> None:
        """Rewrite an existing CSV after new metric columns appear.

        Returns
        -------
        None
            This method mutates the CSV file in place.
        """

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in self.fieldnames})

    @staticmethod
    def _serialize(value: Any) -> Any:
        """Serialize nested values for CSV output.

        Parameters
        ----------
        value : Any
            Raw row value.

        Returns
        -------
        Any
            CSV-compatible scalar value.
        """

        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        if value is None:
            return ""
        return value


def resolve_target_config(target_run_dir: Path | None, target_config: Path | None) -> Path:
    """Resolve the target HIL config path from CLI arguments.

    Parameters
    ----------
    target_run_dir : pathlib.Path | None
        Target run directory.
    target_config : pathlib.Path | None
        Explicit target config path.

    Returns
    -------
    pathlib.Path
        Resolved target config path.
    """

    if target_config:
        path = target_config.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target config does not exist: {path}")
        return path
    if not target_run_dir:
        raise ValueError("Provide --target-run-dir or --target-config.")
    run_dir = target_run_dir.expanduser().resolve()
    config_path = find_config_path(run_dir)
    if config_path is None:
        raise FileNotFoundError(f"No nas_config*.yaml file found in {run_dir}")
    return config_path.resolve()


def prepare_candidates(
    *,
    source_run_dir: Path,
    source_csv_path: Path | None,
    source_config_path: Path | None,
    objective_override: str | None,
    device_option_policy: str = "preserve-source",
    model_variant: str | None = None,
    checkpoint_path: str | None = None,
) -> tuple[list[ReplayCandidate], tuple[ObjectiveSpec, ...], Path, Path | None, int, int]:
    """Load source data and reconstruct deduplicated Pareto candidates.

    Parameters
    ----------
    source_run_dir : pathlib.Path
        Source NAS run directory.
    source_csv_path : pathlib.Path | None
        Optional source CSV override.
    source_config_path : pathlib.Path | None
        Optional source config override.
    objective_override : str | None
        Optional objective override string.
    device_option_policy : str, optional
        Policy for replaying logged device options.
    model_variant : str | None, optional
        Optional model variant forwarded to HIL.
    checkpoint_path : str | None, optional
        Optional checkpoint path forwarded to HIL.

    Returns
    -------
    tuple
        Candidates, objective specs, CSV path, config path, total row count,
        and selected Pareto row count before dedupe.
    """

    resolved_source_config_path = source_config_path or find_config_path(source_run_dir)
    source_config = load_yaml_config(resolved_source_config_path)
    resolved_source_csv_path = source_csv_path or find_csv_path(source_run_dir, source_config)
    rows, columns = read_csv_rows(resolved_source_csv_path)
    objectives = resolve_objective_specs(
        config=source_config,
        rows=rows,
        columns=columns,
        override=objective_override,
    )
    selected_indices = pareto_indices(rows, objectives)
    candidates = [
        build_replay_candidate(
            source_row_index=index,
            row=rows[index],
            objectives=objectives,
            device_option_policy=device_option_policy,
            model_variant=model_variant,
            checkpoint_path=checkpoint_path,
        )
        for index in selected_indices
    ]
    return (
        dedupe_candidates(candidates),
        objectives,
        resolved_source_csv_path.resolve(),
        resolved_source_config_path.resolve() if resolved_source_config_path else None,
        len(rows),
        len(selected_indices),
    )


def create_hil_server(config_path: Path) -> Any:
    """Create a HIL server lazily for hardware replay.

    Parameters
    ----------
    config_path : pathlib.Path
        Target config path.

    Returns
    -------
    Any
        ``HILServer`` instance.
    """

    from hil_server import HILServer

    return HILServer(config_path=config_path)


def determine_candidate_metrics(
    *,
    candidate: ReplayCandidate,
    server: Any | None,
    metrics_callback: MetricsCallback | None,
) -> Mapping[str, Any]:
    """Determine metrics for one replay candidate.

    Parameters
    ----------
    candidate : ReplayCandidate
        Candidate to replay.
    server : Any | None
        HIL server instance when no callback is provided.
    metrics_callback : MetricsCallback | None
        Optional injected metrics function matching ``determine_metrics``.

    Returns
    -------
    Mapping[str, Any]
        Metric mapping returned by the replay backend.
    """

    determine_metrics = metrics_callback if metrics_callback is not None else server.determine_metrics
    return determine_metrics(
        candidate.family_hparams,
        candidate.runtime_metadata,
        quantization_mode=candidate.quantization_mode,
        device_options_overrides=candidate.device_options_overrides,
        checkpoint_path=candidate.checkpoint_path,
        model_variant=candidate.model_variant,
    )


def request_record_for_candidate(candidate: ReplayCandidate, ordinal: int) -> dict[str, Any]:
    """Build one replay request JSONL record.

    Parameters
    ----------
    candidate : ReplayCandidate
        Candidate being scheduled.
    ordinal : int
        One-based scheduled candidate index.

    Returns
    -------
    dict[str, Any]
        JSON-compatible request record.
    """

    return {
        "ordinal": ordinal,
        "source_row_index": candidate.source_row_index,
        "payload_key": candidate.payload_key,
        "replay_payload_key": candidate.replay_payload_key,
        "family_hparams": candidate.family_hparams,
        "runtime_metadata": candidate.runtime_metadata,
        "quantization_mode": candidate.quantization_mode,
        "device_options_overrides": candidate.device_options_overrides,
        "model_variant": candidate.model_variant,
        "checkpoint_path": candidate.checkpoint_path,
    }


def run_replay(
    config: ReplayRunConfig,
    *,
    metrics_callback: MetricsCallback | None = None,
    server_factory: ServerFactory | None = None,
) -> int:
    """Run or dry-run a Pareto HIL replay.

    Parameters
    ----------
    config : ReplayRunConfig
        Replay run configuration.
    metrics_callback : MetricsCallback | None, optional
        Optional hardware-free callback matching ``determine_metrics``.
    server_factory : ServerFactory | None, optional
        Optional factory for constructing a HIL-compatible server.

    Returns
    -------
    int
        Process exit code.
    """

    source_run_dir = config.source_run_dir.expanduser().resolve()
    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source run directory does not exist: {source_run_dir}")
    source_csv_path = config.source_csv.expanduser().resolve() if config.source_csv else None
    source_config_path = config.source_config.expanduser().resolve() if config.source_config else None
    target_config_path = resolve_target_config(config.target_run_dir, config.target_config)

    candidates, objectives, resolved_csv_path, resolved_config_path, total_rows, selected_rows = prepare_candidates(
        source_run_dir=source_run_dir,
        source_csv_path=source_csv_path,
        source_config_path=source_config_path,
        objective_override=config.objectives,
        device_option_policy=config.device_option_policy,
        model_variant=config.model_variant,
        checkpoint_path=config.checkpoint_path,
    )
    if config.max_candidates is not None:
        candidates = candidates[: config.max_candidates]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir
        else default_output_dir(source_run_dir, target_config_path, timestamp)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "replay_results.csv"
    requests_path = output_dir / "replay_requests.jsonl"
    completed_payload_keys = (
        load_completed_payload_keys(results_path)
        if config.resume
        else CompletedReplayKeys(replay_payload_keys=frozenset(), legacy_payload_keys=frozenset())
    )
    candidates = [candidate for candidate in candidates if not completed_payload_keys.contains(candidate)]

    write_manifest(
        output_dir=output_dir,
        source_run_dir=source_run_dir,
        source_csv_path=resolved_csv_path,
        source_config_path=resolved_config_path,
        target_config_path=target_config_path,
        objectives=objectives,
        total_rows=total_rows,
        selected_rows=selected_rows,
        replayed_candidates=len(candidates),
        config=config,
    )

    writer = ResultCsvWriter(results_path)
    server = None
    if not config.dry_run and candidates:
        if not config.allow_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if metrics_callback is None:
            factory = server_factory or create_hil_server
            server = factory(target_config_path)

    for ordinal, candidate in enumerate(candidates, start=1):
        append_jsonl(requests_path, request_record_for_candidate(candidate, ordinal))
        if config.dry_run:
            row = flatten_result_row(
                candidate=candidate,
                source_run_dir=source_run_dir,
                target_config_path=target_config_path,
                replay_status="dry_run",
                metrics=None,
            )
            writer.append(row)
            continue
        try:
            metrics = determine_candidate_metrics(
                candidate=candidate,
                server=server,
                metrics_callback=metrics_callback,
            )
            row = flatten_result_row(
                candidate=candidate,
                source_run_dir=source_run_dir,
                target_config_path=target_config_path,
                replay_status="completed",
                metrics=metrics,
            )
        except Exception as exc:  # pragma: no cover - hardware failure boundary
            row = flatten_result_row(
                candidate=candidate,
                source_run_dir=source_run_dir,
                target_config_path=target_config_path,
                replay_status="runner_error",
                metrics=None,
                error_detail=str(exc),
            )
        writer.append(row)

    logger.info("Source rows: %s", total_rows)
    logger.info("Selected Pareto rows: %s", selected_rows)
    logger.info("Scheduled replay candidates: %s", len(candidates))
    logger.info("Wrote replay outputs: %s", output_dir)
    return 0
