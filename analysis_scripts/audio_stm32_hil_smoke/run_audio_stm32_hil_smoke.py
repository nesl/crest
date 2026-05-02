"""Run STM32 HIL smoke checks for the cached-feature audio DS-CNN path."""

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
from tinyodom.devices import CandidatePrepareRequest  # noqa: E402
from tinyodom.microcontrollers import (  # noqa: E402
    get_device as get_microcontroller_device,
    resolve_device_options,
)
from tinyodom.microcontrollers.stm32_nucleo_n657x0 import (  # noqa: E402
    SUPPORTED_CPU_CLOCK_MHZ,
)
from tinyodom.errors import HIL_ERROR_OK, HIL_MASTER_SUCCESS  # noqa: E402
from tinyodom.model import load_config  # noqa: E402
from tinyodom.pipeline_types import DataSplit  # noqa: E402
from tinyodom.runtime_bootstrap import BootstrappedPipeline, bootstrap_pipeline  # noqa: E402

DEFAULT_CONFIG = SRC_DIR / "config" / "nas_config_audio_stm32.yaml"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "audio_stm32_hil_smoke"
EXPECTED_AUDIO_INPUT_SHAPE = (201, 64)
INPUT_SOURCE = "precomputed_log_mel_features"
PIPELINE_SCOPE = "classifier_inference_only"
FRONTEND_EXCLUDED_REASON = (
    "Microphone capture, buffering, FFT, mel filtering, and log-mel extraction "
    "are not implemented in firmware in Phase 6."
)


@dataclass(frozen=True)
class AudioSTM32SmokePreflight:
    """Prepared shared state for one audio STM32 smoke run.

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
    """Parse command-line arguments for the audio STM32 smoke runner.

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
    mode.add_argument("--prepare-only", action="store_true", help="Run STM32 candidate preparation only.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the audio STM32 config YAML.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for smoke artifacts.")
    parser.add_argument("--model-variant", default=None, help="Optional export variant override.")
    parser.add_argument("--checkpoint-path", default=None, help="Checkpoint path for trained variants.")
    parser.add_argument("--serial-port", default=None, help="Override device.serial_port.")
    parser.add_argument("--harness-serial-port", default=None, help="Override device.harness_serial_port.")
    parser.add_argument(
        "--energy-aware",
        action="store_true",
        help="Enable harness-assisted power metrics for this smoke run.",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("back_to_back", "cadenced"),
        default=None,
        help="Override device.runtime_mode.",
    )
    parser.add_argument("--measured-runs", type=int, default=None, help="Override measured inference runs.")
    parser.add_argument("--cpu-clock-mhz", type=int, default=None, help="Override STM32 CPU clock preset.")
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
    if args.cpu_clock_mhz is not None and int(args.cpu_clock_mhz) not in SUPPORTED_CPU_CLOCK_MHZ:
        allowed = ", ".join(str(value) for value in sorted(SUPPORTED_CPU_CLOCK_MHZ))
        raise ValueError(f"--cpu-clock-mhz must be one of: {allowed}.")
    if args.model_variant is not None:
        variant = str(args.model_variant).strip().lower()
        if not variant:
            raise ValueError("--model-variant must be a non-empty string.")
        if variant.startswith("trained") and not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required when --model-variant starts with 'trained'.")


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


def full_hil_status_from_metrics(metrics: Mapping[str, Any]) -> str:
    """Resolve full-HIL smoke status from mixed HIL/master error semantics.

    Parameters
    ----------
    metrics : Mapping[str, Any]
        Metrics returned by ``HILServer.determine_metrics``.

    Returns
    -------
    str
        ``"ok"`` when the run completed successfully, otherwise ``"blocked"``.
    """

    error_label = metrics.get("error_label")
    if error_label == "HIL_MASTER_SUCCESS":
        return "ok"
    try:
        error_code = int(metrics.get("error_code", HIL_ERROR_OK))
    except (TypeError, ValueError):
        return "blocked"
    # Unit tests and some direct helper paths use the immediate HIL namespace
    # where zero means OK. Full HIL success uses the master namespace where one
    # means success, so prefer the explicit label when present.
    if error_label in (None, "", "HIL_ERROR_OK") and error_code == HIL_ERROR_OK:
        return "ok"
    if error_label in (None, "") and error_code == HIL_MASTER_SUCCESS:
        return "ok"
    return "blocked"


def _summary_base(preflight: AudioSTM32SmokePreflight) -> dict[str, Any]:
    """Build the common JSON summary payload.

    Parameters
    ----------
    preflight : AudioSTM32SmokePreflight
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
        "model_family": preflight.pipeline.selection["model_family_name"],
        "task_name": preflight.pipeline.selection["task_name"],
        "input_shape": list(preflight.pipeline.model_build_context.input_shape),
        "batch_period_ms": metadata.get("batch_period_ms"),
        "export_variant": preflight.model_variant,
        "checkpoint_path": None if preflight.checkpoint_path is None else str(preflight.checkpoint_path),
        "hparams": _json_safe(preflight.hparams),
        "runtime_metadata": _json_safe(preflight.runtime_metadata),
        "diagnostic_tflite_path": str(preflight.diagnostic_tflite_path),
        "candidate_root": None,
        "prepared_network_input_bytes": None,
        "prepared_network_input_ok": None,
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
    summary_path = output_dir / "audio_stm32_hil_smoke_summary.json"
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
    config.outputs.candidate_dir = str((output_dir / "candidates").resolve())
    config.outputs.tflite_model_path = str((preflight_dir / "audio_stm32_hil_smoke.tflite").resolve())
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

    if full_hil:
        config.device.hil = True
    if args.serial_port is not None:
        config.device.serial_port = args.serial_port
    if args.harness_serial_port is not None:
        config.device.harness_serial_port = args.harness_serial_port
    if args.energy_aware:
        config.training.energy_aware = True
    if args.runtime_mode is not None:
        config.device.runtime_mode = args.runtime_mode
    if args.measured_runs is not None:
        config.device.measured_inference_runs = int(args.measured_runs)


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
        raise ValueError("Audio STM32 smoke requires cached calibration data. Run `make prepare-audio-dataset` first.")
    return split


def require_expected_audio_input_shape(input_shape: tuple[int, ...] | None) -> tuple[int, int]:
    """Validate the Phase 6 fixed audio feature input shape.

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
            "Audio STM32 smoke expects fixed log-mel input_shape "
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
    # This diagnostic export intentionally avoids representative quantization;
    # the STM32 backend owns the candidate-local quantized export path.
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


def build_preflight(args: argparse.Namespace) -> AudioSTM32SmokePreflight:
    """Build all shared state needed by preflight, prepare-only, and full HIL.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    AudioSTM32SmokePreflight
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
    return AudioSTM32SmokePreflight(
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


def parse_prepared_network_input_bytes(staged_root: Path) -> int | None:
    """Parse the generated ST Edge AI input byte-count macro.

    Parameters
    ----------
    staged_root : Path
        Staged STM32 workspace root returned by ``prepare_candidate``.

    Returns
    -------
    int | None
        Parsed ``AI_NETWORK_IN_1_SIZE_BYTES`` value, or ``None`` when absent.
    """

    pattern = re.compile(r"(?m)^#define\s+AI_NETWORK_IN_1_SIZE_BYTES\s+\(?\s*(?P<value>\d+)\s*\)?")
    for search_root in (staged_root / "Appli" / "Inc", staged_root / "Appli" / "Src"):
        if not search_root.is_dir():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path.suffix not in {".h", ".c"}:
                continue
            match = pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
            if match is not None:
                return int(match.group("value"))
    return None


def validate_prepared_input_contract(
    staged_root: Path,
    *,
    quantized_input: bool,
) -> tuple[int | None, bool, list[str]]:
    """Validate prepared STM32 input bytes against the logical audio shape.

    Parameters
    ----------
    staged_root : Path
        Staged STM32 workspace root.
    quantized_input : bool
        Whether STM32 candidate preparation exported an int8 quantized input.

    Returns
    -------
    tuple[int | None, bool, list[str]]
        Observed byte count, contract status, and follow-up messages.
    """

    observed = parse_prepared_network_input_bytes(staged_root)
    bytes_per_feature = 1 if quantized_input else 4
    expected = (
        int(EXPECTED_AUDIO_INPUT_SHAPE[0])
        * int(EXPECTED_AUDIO_INPUT_SHAPE[1])
        * bytes_per_feature
    )
    if observed == expected:
        return observed, True, []
    if observed is None:
        return observed, False, [
            "Phase 7: expose or parse generated STM32 network input bytes for audio compatibility checks."
        ]
    return observed, False, [
        (
            "Phase 7: STM32 generated network input byte count does not match "
            f"audio feature shape and dtype; expected {expected}, observed {observed}."
        )
    ]


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Run the hardware-free audio STM32 preflight path.

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
    """Run STM32 candidate preparation without upload/runtime collection.

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
    if args.cpu_clock_mhz is not None:
        device_options["cpu_clock_mhz"] = int(args.cpu_clock_mhz)
    device = get_microcontroller_device(
        str(preflight.config.device.name),
        serial_port=getattr(preflight.config.device, "serial_port", None),
        device_options=device_options,
    )
    prepared_dir = Path(device.prepare_candidate(request=preflight.request))
    observed, input_ok, followups = validate_prepared_input_contract(
        prepared_dir,
        quantized_input=bool(getattr(preflight.config.training, "quantization", False)),
    )
    summary = _summary_base(preflight)
    summary.update(
        {
            "mode": "prepare_only",
            "status": "ok" if input_ok else "blocked",
            "candidate_root": str(prepared_dir),
            "prepared_network_input_bytes": observed,
            "prepared_network_input_ok": input_ok,
            "followups": followups,
        }
    )
    summary["summary_path"] = str(write_summary(preflight.output_dir, summary))
    return summary


def run_full_hil(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full STM32 HIL smoke path through ``HILServer``.

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
    overrides: dict[str, Any] = {}
    if args.cpu_clock_mhz is not None:
        overrides["cpu_clock_mhz"] = int(args.cpu_clock_mhz)
    metrics = server.determine_metrics(
        preflight.hparams,
        preflight.runtime_metadata,
        device_options_overrides=overrides or None,
        checkpoint_path=preflight.checkpoint_path,
        model_variant=preflight.model_variant,
    )
    summary = _summary_base(preflight)
    summary.update({"mode": "full_hil", "status": full_hil_status_from_metrics(metrics), "metrics": metrics})
    summary["summary_path"] = str(write_summary(preflight.output_dir, summary))
    return summary


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch the selected audio STM32 smoke mode.

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
    """Run the audio STM32 smoke command-line entrypoint.

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
