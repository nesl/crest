"""Run a small hardware-free audio DS-CNN desktop smoke pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.model import load_config  # noqa: E402
from tinyodom.pipeline_types import DataSplit, DatasetBundle  # noqa: E402
from tinyodom.runtime_bootstrap import bootstrap_pipeline  # noqa: E402
from tinyodom.builtin_components import ensure_audio_components_registered  # noqa: E402


DEFAULT_CONFIG = SRC_DIR / "config" / "nas_config_audio_stm32.yaml"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "audio_desktop_smoke"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the audio desktop smoke script.

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
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the audio config YAML.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs for the smoke pass.")
    parser.add_argument("--max-train-examples", type=int, default=128, help="Maximum training examples.")
    parser.add_argument("--max-val-examples", type=int, default=64, help="Maximum validation examples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Keras batch size.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for smoke artifacts.")
    return parser.parse_args(argv)


def _validate_positive_int(value: int, *, field_name: str) -> int:
    """Validate a positive integer CLI value.

    Parameters
    ----------
    value : int
        Parsed CLI value.
    field_name : str
        Human-readable field name used in errors.

    Returns
    -------
    int
        Validated integer.

    Raises
    ------
    ValueError
        If ``value`` is not positive.
    """

    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _slice_optional(value: Any, limit: int, original_length: int) -> Any:
    """Slice an optional split payload when it aligns with split rows.

    Parameters
    ----------
    value : Any
        Candidate array-like payload.
    limit : int
        Maximum row count.
    original_length : int
        Original split row count.

    Returns
    -------
    Any
        Sliced value when row-aligned, otherwise the original value.
    """

    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and shape[0] == original_length:
        return value[:limit]
    if isinstance(value, list) and len(value) == original_length:
        return value[:limit]
    return value


def slice_split(split: DataSplit, max_examples: int) -> DataSplit:
    """Return a row-limited copy of one data split.

    Parameters
    ----------
    split : DataSplit
        Split to slice.
    max_examples : int
        Maximum row count to keep.

    Returns
    -------
    DataSplit
        New split with row-aligned payloads sliced.
    """

    row_count = int(split.inputs.shape[0])
    limit = min(_validate_positive_int(max_examples, field_name="max_examples"), row_count)
    metadata = {
        key: _slice_optional(value, limit, row_count)
        for key, value in split.metadata.items()
    }
    return DataSplit(
        inputs=split.inputs[:limit],
        targets=split.targets[:limit],
        sample_weights=_slice_optional(split.sample_weights, limit, row_count),
        metadata=metadata,
    )


def slice_bundle(
    bundle: DatasetBundle,
    *,
    max_train_examples: int,
    max_val_examples: int,
) -> DatasetBundle:
    """Return a dataset bundle with train/validation splits truncated.

    Parameters
    ----------
    bundle : DatasetBundle
        Loaded dataset bundle.
    max_train_examples : int
        Maximum training examples.
    max_val_examples : int
        Maximum validation examples.

    Returns
    -------
    DatasetBundle
        Bundle copy with sliced train/validation splits.
    """

    if bundle.val is None:
        raise ValueError("Audio desktop smoke requires a validation split.")
    return DatasetBundle(
        train=slice_split(bundle.train, max_train_examples),
        val=slice_split(bundle.val, max_val_examples),
        test=bundle.test,
        calibration=bundle.calibration,
        input_shape=bundle.input_shape,
        input_dtype=bundle.input_dtype,
        metadata=dict(bundle.metadata),
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the audio desktop smoke workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI options.

    Returns
    -------
    dict[str, Any]
        JSON-safe summary written to disk and returned to callers.
    """

    _validate_positive_int(args.epochs, field_name="--epochs")
    _validate_positive_int(args.max_train_examples, field_name="--max-train-examples")
    _validate_positive_int(args.max_val_examples, field_name="--max-val-examples")
    _validate_positive_int(args.batch_size, field_name="--batch-size")

    ensure_audio_components_registered()
    config = load_config(Path(args.config))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "audio_desktop_smoke.keras"

    try:
        loaded = bootstrap_pipeline(config, checkpoint_path=checkpoint_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nUrbanSound8K cache is missing. Run `make prepare-audio-dataset` first."
        ) from exc

    bundle = slice_bundle(
        loaded.bundle,
        max_train_examples=args.max_train_examples,
        max_val_examples=args.max_val_examples,
    )
    loaded = bootstrap_pipeline(
        config,
        dataset=loaded.dataset,
        bundle=bundle,
        checkpoint_path=checkpoint_path,
    )

    seed_trial = loaded.model_family.default_seed_trial(
        loaded.model_build_context,
        loaded.selection["model_config"],
    )
    if seed_trial is None:
        raise ValueError("The active model family does not define a default seed trial.")
    hparams = loaded.model_family.decode_trial_hparams(
        seed_trial,
        loaded.model_build_context,
        loaded.selection["model_config"],
    )
    model = loaded.model_family.build_model(
        hparams,
        loaded.model_build_context,
        loaded.selection["model_config"],
    )
    loaded.task.validate_model_outputs(model, loaded.target_spec)
    loaded.task.compile_model(model, loaded.selection["task_config"], loaded.target_spec)
    fit_plan = loaded.task.build_fit_plan(
        loaded.bundle,
        loaded.selection["task_config"],
        loaded.target_spec,
        mode="search",
        combine_train_val=False,
    )
    history = model.fit(
        **fit_plan.fit_kwargs,
        callbacks=fit_plan.callbacks,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )
    model.save(checkpoint_path)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Audio desktop smoke checkpoint was not written: {checkpoint_path}")
    evaluation = loaded.task.evaluate(
        model,
        loaded.bundle.val,
        loaded.selection["task_config"],
        loaded.target_spec,
    )
    history_payload = {
        key: [float(item) for item in values]
        for key, values in history.history.items()
    }
    history_path = output_dir / "audio_desktop_smoke_history.json"
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    payload = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "history": history_payload,
        "metrics": dict(evaluation.metrics),
        "artifacts": dict(evaluation.artifacts or {}),
        "hparams": dict(hparams),
    }
    metrics_path = output_dir / "audio_desktop_smoke_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["metrics_path"] = str(metrics_path)
    return payload


def main(argv: list[str] | None = None) -> None:
    """Run the audio desktop smoke CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argument list for tests.

    Returns
    -------
    None
        Prints the metrics path and metrics summary.
    """

    result = run_smoke(parse_args(argv))
    print(f"Audio desktop smoke metrics written to {result['metrics_path']}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
