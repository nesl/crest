"""Run Arduino HIL smoke checks for the cached-feature audio DS-CNN path."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hil_server import HILServer  # noqa: E402
from tinyodom.builtin_components import ensure_audio_components_registered  # noqa: E402
from tinyodom.devices import CandidatePrepareRequest, arduino_staged_sketch_path  # noqa: E402
from tinyodom.hil_runtime import build_collect_metrics_request, collect_metrics  # noqa: E402
from tinyodom.microcontrollers import (  # noqa: E402
    get_device as get_microcontroller_device,
    resolve_device_options,
)
from tinyodom.errors import HIL_ERROR_OK, HIL_MASTER_SUCCESS  # noqa: E402
from tinyodom.model import load_config  # noqa: E402
from tinyodom.pipeline_types import DataSplit  # noqa: E402
from tinyodom.runtime_bootstrap import BootstrappedPipeline, bootstrap_pipeline  # noqa: E402

DEFAULT_CONFIG = SRC_DIR / "config" / "nas_config_audio_portenta.yaml"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "audio_portenta_hil_smoke"
EXPECTED_AUDIO_INPUT_SHAPE = (201, 64)
EXPECTED_INT8_INPUT_BYTES = EXPECTED_AUDIO_INPUT_SHAPE[0] * EXPECTED_AUDIO_INPUT_SHAPE[1]
INPUT_SOURCE = "precomputed_log_mel_features"
PIPELINE_SCOPE = "classifier_inference_only"
FRONTEND_EXCLUDED_REASON = (
    "Microphone capture, buffering, FFT, mel filtering, and log-mel extraction "
    "are not implemented in firmware in Phase 8."
)
ARDUINO_AUDIO_DEVICES = ("PORTENTA_H7", "ARDUINO_NANO_33_BLE_SENSE")


@dataclass(frozen=True)
class AudioPortentaSmokePreflight:
    """Prepared shared state for one Arduino audio smoke run.

    Parameters
    ----------
    config : Any
        Loaded and in-memory-mutated TinyODOM config.
    pipeline : BootstrappedPipeline
        Bootstrapped audio dataset/task/model-family stack.
    hparams : dict[str, Any]
        Decoded audio DS-CNN hyperparameters.
    model : Any
        Keras model materialized for export.
    runtime_metadata : dict[str, Any]
        HIL request metadata for logical feature-shaped input.
    calibration_split : DataSplit
        Cached representative feature samples used by export preparation.
    request : CandidatePrepareRequest
        Backend preparation request for prepare-only mode.
    diagnostic_tflite_path : Path
        Float32 diagnostic TFLite path written by preflight.
    output_dir : Path
        Directory receiving summary and diagnostic artifacts.
    model_variant : str
        Resolved export variant.
    checkpoint_path : Path | None
        Optional checkpoint path for trained variants.
    """

    config: Any
    pipeline: BootstrappedPipeline
    hparams: dict[str, Any]
    model: Any
    runtime_metadata: dict[str, Any]
    calibration_split: DataSplit
    request: CandidatePrepareRequest
    diagnostic_tflite_path: Path
    output_dir: Path
    model_variant: str
    checkpoint_path: Path | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the Arduino audio smoke runner.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argument list. When omitted, argparse reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed CLI options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="Run hardware-free preflight only.")
    mode.add_argument("--prepare-only", action="store_true", help="Export, stage, and compile without upload.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the audio Portenta config YAML.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for smoke artifacts.")
    parser.add_argument("--model-variant", default=None, help="Optional export variant override.")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint path for trained variants.")
    parser.add_argument("--serial-port", default=None, help="Override device.serial_port.")
    parser.add_argument("--harness-serial-port", default=None, help="Override device.harness_serial_port.")
    parser.add_argument("--device-name", choices=ARDUINO_AUDIO_DEVICES, default=None, help="Override Arduino target.")
    parser.add_argument("--target-core", choices=("cm7", "cm4"), default=None, help="Override Portenta target core.")
    parser.add_argument("--split", choices=("50_50", "75_25", "100_0"), default=None, help="Override Portenta RAM split.")
    parser.add_argument("--measured-runs", type=int, default=None, help="Override measured inference runs.")
    return parser.parse_args(argv)


def _validate_positive_int(value: int | None, *, field_name: str) -> int | None:
    """Validate an optional positive integer.

    Parameters
    ----------
    value : int | None
        Candidate integer value.
    field_name : str
        Human-readable field name for errors.

    Returns
    -------
    int | None
        Validated value, or ``None`` when omitted.
    """

    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def validate_args(args: argparse.Namespace) -> None:
    """Validate cross-field CLI options.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    None
        Raises if options are inconsistent.
    """

    _validate_positive_int(args.measured_runs, field_name="--measured-runs")
    if args.model_variant is not None:
        variant = str(args.model_variant).strip().lower()
        if not variant:
            raise ValueError("--model-variant must be a non-empty string.")
        if variant.startswith("trained") and not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required when --model-variant starts with 'trained'.")
    if args.device_name == "ARDUINO_NANO_33_BLE_SENSE" and (args.target_core or args.split):
        raise ValueError("--target-core and --split apply only to PORTENTA_H7.")


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Read a value from mapping-like or attribute-style config objects.

    Parameters
    ----------
    container : Any
        Mapping or namespace-like object.
    key : str
        Field name to read.
    default : Any, optional
        Fallback when the field is absent.

    Returns
    -------
    Any
        Resolved value or ``default``.
    """

    getter = getattr(container, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(container, key, default)


def _json_safe(value: Any) -> Any:
    """Convert common Python/scientific objects into JSON-safe values.

    Parameters
    ----------
    value : Any
        Value to normalize.

    Returns
    -------
    Any
        JSON-serializable representation.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def status_from_metrics(metrics: Mapping[str, Any]) -> str:
    """Resolve smoke status from mixed HIL/master error semantics.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Metrics returned by compile-only or full-HIL collection.

    Returns
    -------
    str
        ``"ok"`` when the run completed successfully, otherwise ``"blocked"``.
    """

    error_label = metrics.get("error_label")
    try:
        error_code = int(metrics.get("error_code", HIL_ERROR_OK))
    except (TypeError, ValueError):
        return "blocked"
    if error_label in ("HIL_MASTER_SUCCESS", "HIL_ERROR_OK", None, "") and error_code in (
        HIL_ERROR_OK,
        HIL_MASTER_SUCCESS,
    ):
        return "ok"
    return "blocked"


def _summary_base(preflight: AudioPortentaSmokePreflight) -> dict[str, Any]:
    """Build the common JSON summary payload.

    Parameters
    ----------
    preflight : AudioPortentaSmokePreflight
        Prepared smoke run state.

    Returns
    -------
    dict[str, Any]
        JSON-safe common summary fields.
    """

    metadata = dict(preflight.pipeline.bundle.metadata)
    return {
        "config_path": str(Path(preflight.config.config_path) if hasattr(preflight.config, "config_path") else DEFAULT_CONFIG),
        "output_dir": str(preflight.output_dir),
        "device_name": str(preflight.config.device.name),
        "model_family": preflight.pipeline.selection["model_family_name"],
        "task_name": preflight.pipeline.selection["task_name"],
        "input_shape": list(preflight.pipeline.model_build_context.input_shape),
        "expected_int8_input_bytes": EXPECTED_INT8_INPUT_BYTES,
        "batch_period_ms": metadata.get("batch_period_ms"),
        "export_variant": preflight.model_variant,
        "checkpoint_path": None if preflight.checkpoint_path is None else str(preflight.checkpoint_path),
        "hparams": _json_safe(preflight.hparams),
        "runtime_metadata": _json_safe(preflight.runtime_metadata),
        "diagnostic_tflite_path": str(preflight.diagnostic_tflite_path),
        "candidate_root": None,
        "staged_sketch_path": None,
        "staged_window_size": None,
        "staged_num_channels": None,
        "staged_shape_ok": None,
        "metrics": None,
        "followups": [],
        "input_source": INPUT_SOURCE,
        "pipeline_scope": PIPELINE_SCOPE,
        "frontend_included": False,
        "frontend_excluded_reason": FRONTEND_EXCLUDED_REASON,
    }


def write_summary(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Write one smoke summary JSON file.

    Parameters
    ----------
    output_dir : Path
        Directory receiving the summary.
    payload : Mapping[str, Any]
        Summary payload.

    Returns
    -------
    Path
        Path to the written summary file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "audio_portenta_hil_smoke_summary.json"
    summary_path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def apply_output_overrides(config: Any, output_dir: Path) -> Path:
    """Apply script-local output paths to a loaded config.

    Parameters
    ----------
    config : Any
        Mutable loaded config.
    output_dir : Path
        Script output directory.

    Returns
    -------
    Path
        Diagnostic TFLite path.
    """

    preflight_dir = output_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    candidate_basename = Path(config.outputs.candidate_dir).name or "audio_dscnn"
    config.outputs.candidate_dir = str((output_dir / candidate_basename).resolve())
    config.outputs.tflite_model_path = str((preflight_dir / "audio_portenta_hil_smoke.tflite").resolve())
    return Path(config.outputs.tflite_model_path)


def apply_hardware_overrides(config: Any, args: argparse.Namespace, *, full_hil: bool) -> None:
    """Apply CLI hardware overrides to the mutable config object.

    Parameters
    ----------
    config : Any
        Mutable loaded config.
    args : argparse.Namespace
        Parsed CLI options.
    full_hil : bool
        Whether to force ``device.hil`` for full runtime collection.

    Returns
    -------
    None
        Mutates ``config`` in memory only.
    """

    config.device.hil = bool(full_hil)
    if args.device_name is not None:
        config.device.name = args.device_name
    if args.serial_port is not None:
        config.device.serial_port = args.serial_port
    if args.harness_serial_port is not None:
        config.device.harness_serial_port = args.harness_serial_port
    if args.measured_runs is not None:
        config.device.measured_inference_runs = int(args.measured_runs)
    if str(config.device.name).strip().upper() == "PORTENTA_H7":
        if _cfg_get(config.device, "portenta", None) is None:
            config.device.portenta = {}
        if args.target_core is not None:
            config.device.portenta.target_core = args.target_core
        if args.split is not None:
            config.device.portenta.split = args.split


def require_calibration_split(split: DataSplit | None) -> DataSplit:
    """Return a non-empty calibration split.

    Parameters
    ----------
    split : DataSplit | None
        Candidate calibration split.

    Returns
    -------
    DataSplit
        Non-empty calibration split.
    """

    if split is None or getattr(split.inputs, "shape", (0,))[0] <= 0:
        raise ValueError("Audio Portenta smoke requires cached calibration data. Run `make prepare-audio-dataset` first.")
    return split


def require_expected_audio_input_shape(input_shape: tuple[int, ...] | None) -> tuple[int, int]:
    """Validate the fixed audio feature input shape.

    Parameters
    ----------
    input_shape : tuple[int, ...] | None
        Logical model input shape from the bootstrapped pipeline.

    Returns
    -------
    tuple[int, int]
        Validated `(201, 64)` audio feature shape.
    """

    if tuple(input_shape or ()) != EXPECTED_AUDIO_INPUT_SHAPE:
        raise ValueError(
            "Audio Portenta smoke expects fixed log-mel input_shape "
            f"{EXPECTED_AUDIO_INPUT_SHAPE}; got {input_shape!r}."
        )
    return EXPECTED_AUDIO_INPUT_SHAPE


def resolve_model_variant(config: Any, model_config: Any, override: str | None) -> str:
    """Resolve the export model variant for the smoke run.

    Parameters
    ----------
    config : Any
        Loaded config.
    model_config : Any
        Model-family config from the bootstrapped pipeline.
    override : str | None
        Optional CLI override.

    Returns
    -------
    str
        Non-empty export variant.
    """

    if override is not None:
        return str(override).strip()
    params = _cfg_get(model_config, "params", None)
    variant = _cfg_get(params, "export_variant", None)
    if not isinstance(variant, str) or not variant.strip():
        variant = _cfg_get(config.model.params, "export_variant", None)
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("Expected model.params.export_variant to be a non-empty string.")
    return variant.strip()


def export_diagnostic_tflite(model: Any, output_path: Path) -> None:
    """Export a float32 diagnostic TFLite file.

    Parameters
    ----------
    model : Any
        Keras model to convert.
    output_path : Path
        Destination TFLite path.

    Returns
    -------
    None
        Writes converted bytes.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    output_path.write_bytes(converter.convert())


def build_runtime_metadata(preflight_model: Any, pipeline: BootstrappedPipeline, model_config: Any) -> dict[str, Any]:
    """Build HIL runtime metadata for the logical audio feature shape.

    Parameters
    ----------
    preflight_model : Any
        Keras model used for FLOP counting.
    pipeline : BootstrappedPipeline
        Bootstrapped audio pipeline.
    model_config : Any
        Model-family config.

    Returns
    -------
    dict[str, Any]
        Runtime metadata consumed by ``HILServer.determine_metrics``.
    """

    input_shape = pipeline.model_build_context.input_shape
    return {
        "timesteps": int(input_shape[0]),
        "input_dim": int(input_shape[1]),
        "batch_size": 1,
        "flops": int(
            pipeline.model_family.count_flops(
                preflight_model,
                pipeline.model_build_context,
                model_config,
            )
        ),
    }


def build_preflight(args: argparse.Namespace) -> AudioPortentaSmokePreflight:
    """Build all shared state needed by preflight, prepare-only, and full HIL.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    AudioPortentaSmokePreflight
        Prepared smoke run state.
    """

    validate_args(args)
    ensure_audio_components_registered()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(load_config(Path(args.config)))
    config.config_path = str(Path(args.config).expanduser().resolve())
    diagnostic_tflite_path = apply_output_overrides(config, output_dir)
    apply_hardware_overrides(config, args, full_hil=not args.preflight_only and not args.prepare_only)

    checkpoint_path = None if args.checkpoint_path is None else Path(args.checkpoint_path).expanduser().resolve()
    try:
        pipeline = bootstrap_pipeline(config, checkpoint_path=checkpoint_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}\nUrbanSound8K cache is missing. Run `make prepare-audio-dataset` first.") from exc

    model_config = pipeline.selection["model_config"]
    model_variant = resolve_model_variant(config, model_config, args.model_variant)
    require_expected_audio_input_shape(pipeline.model_build_context.input_shape)
    seed_trial = pipeline.model_family.default_seed_trial(pipeline.model_build_context, model_config)
    if seed_trial is None:
        raise ValueError("The active model family does not define a default seed trial.")
    hparams = pipeline.model_family.decode_trial_hparams(seed_trial, pipeline.model_build_context, model_config)
    model = pipeline.model_family.materialize_export_model(
        hparams,
        pipeline.model_build_context,
        model_config,
        model_variant=model_variant,
        checkpoint_path=checkpoint_path,
    )
    pipeline.task.validate_model_outputs(model, pipeline.target_spec)
    calibration_split = require_calibration_split(pipeline.bundle.calibration)
    runtime_metadata = build_runtime_metadata(model, pipeline, model_config)
    export_diagnostic_tflite(model, diagnostic_tflite_path)
    request = CandidatePrepareRequest(
        config=config,
        model=model,
        model_variant=model_variant,
        artifact_root=Path(config.outputs.candidate_dir),
        tflite_model_path=Path(config.outputs.tflite_model_path),
        calibration_split=calibration_split,
        input_shape=pipeline.model_build_context.input_shape,
        checkpoint_path=checkpoint_path,
    )
    return AudioPortentaSmokePreflight(
        config=config,
        pipeline=pipeline,
        hparams=hparams,
        model=model,
        runtime_metadata=runtime_metadata,
        calibration_split=calibration_split,
        request=request,
        diagnostic_tflite_path=diagnostic_tflite_path,
        output_dir=output_dir,
        model_variant=model_variant,
        checkpoint_path=checkpoint_path,
    )


def parse_staged_define(sketch_path: Path, define_name: str) -> int | None:
    """Parse one integer ``#define`` from a staged Arduino sketch.

    Parameters
    ----------
    sketch_path : Path
        Staged sketch file.
    define_name : str
        Macro name to parse.

    Returns
    -------
    int | None
        Parsed integer value, or ``None`` when absent.
    """

    if not sketch_path.is_file():
        return None
    pattern = re.compile(rf"(?m)^#define\s+{re.escape(define_name)}\s+\(?\s*(?P<value>\d+)\s*\)?")
    match = pattern.search(sketch_path.read_text(encoding="utf-8", errors="ignore"))
    if match is None:
        return None
    return int(match.group("value"))


def validate_staged_shape_contract(staged_sketch_path: Path) -> tuple[int | None, int | None, bool, list[str]]:
    """Validate staged Arduino shape macros against the audio feature shape.

    Parameters
    ----------
    staged_sketch_path : Path
        Path to the staged Arduino sketch.

    Returns
    -------
    tuple[int | None, int | None, bool, list[str]]
        Parsed window size, parsed channel count, contract status, and
        follow-up messages.
    """

    window_size = parse_staged_define(staged_sketch_path, "TINYODOM_WINDOW_SIZE")
    num_channels = parse_staged_define(staged_sketch_path, "TINYODOM_NUM_CHANNELS")
    ok = (window_size, num_channels) == EXPECTED_AUDIO_INPUT_SHAPE
    if ok:
        return window_size, num_channels, True, []
    return window_size, num_channels, False, [
        (
            "Staged Arduino sketch does not match audio feature shape; "
            f"expected window/channels {EXPECTED_AUDIO_INPUT_SHAPE}, observed "
            f"{window_size}/{num_channels}."
        )
    ]


def _ble_followups(config: Any, metrics: Mapping[str, Any], status: str) -> list[str]:
    """Build BLE-specific resource notes when BLE is requested.

    Parameters
    ----------
    config : Any
        Active smoke config.
    metrics : Mapping[str, Any]
        Metrics returned by compile or full-HIL collection.
    status : str
        Resolved smoke status.

    Returns
    -------
    list[str]
        Follow-up notes for BLE runs.
    """

    if str(config.device.name).strip().upper() != "ARDUINO_NANO_33_BLE_SENSE":
        return []
    if status == "ok":
        return []
    evidence = []
    for key in ("error_label", "backend_error_kind", "backend_error_detail", "ram_bytes", "flash_bytes", "arena_bytes"):
        value = metrics.get(key)
        if value not in (None, "", -1, -1.0):
            evidence.append(f"{key}={value}")
    detail = "; ".join(evidence) if evidence else "no detailed compiler evidence was returned"
    return [
        (
            "BLE audio is blocked in this run. Input tensor pressure is "
            f"{EXPECTED_INT8_INPUT_BYTES} int8 bytes before weights/arena; compile/runtime evidence: {detail}."
        )
    ]


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run the hardware-free Arduino audio preflight path.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    dict[str, Any]
        JSON-safe summary payload.
    """

    preflight = build_preflight(args)
    summary = _summary_base(preflight)
    summary["mode"] = "preflight"
    summary["status"] = "ok"
    summary["summary_path"] = str(write_summary(preflight.output_dir, summary))
    return summary


def run_prepare_only(args: argparse.Namespace) -> dict[str, Any]:
    """Run Arduino candidate preparation and compile without upload.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    dict[str, Any]
        JSON-safe summary payload.
    """

    preflight = build_preflight(args)
    device_options = resolve_device_options(str(preflight.config.device.name), preflight.config.device) or {}
    device = get_microcontroller_device(
        str(preflight.config.device.name),
        serial_port=getattr(preflight.config.device, "serial_port", None),
        device_options=device_options,
    )
    prepared_dir = Path(device.prepare_candidate(request=preflight.request))
    request_metrics_args = build_collect_metrics_request(
        config=preflight.config,
        runtime_metadata=preflight.runtime_metadata,
        latency_budget_ms=float(preflight.pipeline.bundle.metadata.get("batch_period_ms", 2000.0)),
        dirpath=prepared_dir,
        device_options=device_options,
        hil_enabled=False,
        energy_aware=False,
        window_size=EXPECTED_AUDIO_INPUT_SHAPE[0],
        input_dim=EXPECTED_AUDIO_INPUT_SHAPE[1],
    )
    metrics = collect_metrics(request_metrics_args)
    staged_sketch_path = arduino_staged_sketch_path(prepared_dir)
    window_size, num_channels, shape_ok, followups = validate_staged_shape_contract(staged_sketch_path)
    status = status_from_metrics(metrics)
    if status == "ok" and not shape_ok:
        status = "blocked"
    followups.extend(_ble_followups(preflight.config, metrics, status))
    summary = _summary_base(preflight)
    summary.update(
        {
            "mode": "prepare_only",
            "status": status,
            "candidate_root": str(prepared_dir),
            "staged_sketch_path": str(staged_sketch_path),
            "staged_window_size": window_size,
            "staged_num_channels": num_channels,
            "staged_shape_ok": shape_ok,
            "metrics": metrics,
            "followups": followups,
        }
    )
    summary["summary_path"] = str(write_summary(preflight.output_dir, summary))
    return summary


def run_full_hil(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full Arduino HIL smoke path through ``HILServer``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    dict[str, Any]
        JSON-safe summary payload.
    """

    preflight = build_preflight(args)
    server = HILServer(config=preflight.config)
    metrics = server.determine_metrics(
        preflight.hparams,
        preflight.runtime_metadata,
        checkpoint_path=preflight.checkpoint_path,
        model_variant=preflight.model_variant,
    )
    status = status_from_metrics(metrics)
    followups = _ble_followups(preflight.config, metrics, status)
    summary = _summary_base(preflight)
    summary.update({"mode": "full_hil", "status": status, "metrics": metrics, "followups": followups})
    summary["summary_path"] = str(write_summary(preflight.output_dir, summary))
    return summary


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch the selected Arduino audio smoke mode.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    dict[str, Any]
        JSON-safe summary payload.
    """

    if args.preflight_only:
        return run_preflight(args)
    if args.prepare_only:
        return run_prepare_only(args)
    return run_full_hil(args)


def main(argv: list[str] | None = None) -> int:
    """Run the Arduino audio smoke command-line entrypoint.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argument list for tests.

    Returns
    -------
    int
        Process exit code.
    """

    args = parse_args(argv)
    summary = run_smoke(args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if summary.get("status") == "blocked":
        return 2
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
