#!/usr/bin/env python3
"""Export UrbanSound8K cached log-mel input profiles for HIL sketches.

Usage examples:
  python analysis_scripts/hil_noise_analysis/urbansound8k_input_profile.py --split train
  python analysis_scripts/hil_noise_analysis/urbansound8k_input_profile.py --split calibration --export-header sketches/analysis_sketches/urbansound8k_input_data.h
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


VALID_SPLITS = frozenset({"train", "val", "test", "calibration"})
DEFAULT_CONFIG_PATH = Path("src/config/nas_config_audio_stm32.yaml")


@dataclass(frozen=True)
class InputProfile:
    """UrbanSound8K input statistics and sampled real windows.

    Parameters
    ----------
    window_size : int
        Log-mel frame count per cached input window.
    n_channels : int
        Mel-bin count per cached input window.
    channel_means : np.ndarray
        Per-mel-bin means.
    channel_stds : np.ndarray
        Per-mel-bin standard deviations.
    channel_min : np.ndarray
        Per-mel-bin minima.
    channel_max : np.ndarray
        Per-mel-bin maxima.
    channel_is_binary : np.ndarray
        Per-mel-bin binary mask. Log-mel features should normally be false.
    real_windows : np.ndarray
        Sampled cached windows with shape
        ``(n_windows, window_size, n_channels)``.
    """

    window_size: int
    n_channels: int
    channel_means: np.ndarray
    channel_stds: np.ndarray
    channel_min: np.ndarray
    channel_max: np.ndarray
    channel_is_binary: np.ndarray
    real_windows: np.ndarray


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    """Read a field from a mapping-like object.

    Parameters
    ----------
    container : Any
        Mapping to inspect.
    key : str
        Field name to read.
    default : Any, optional
        Fallback when the field is absent.

    Returns
    -------
    Any
        Resolved value or ``default``.
    """

    if not isinstance(container, dict):
        return default
    return container.get(key, default)


def resolve_cache_dir(config_path: Path) -> Path:
    """Resolve ``dataset.params.cache_dir`` from a TinyODOM YAML config.

    Parameters
    ----------
    config_path : pathlib.Path
        Config YAML path.

    Returns
    -------
    pathlib.Path
        Cache directory path from the config.

    Raises
    ------
    ValueError
        If the config does not contain a non-empty cache directory.
    """

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = _cfg_get(config, "dataset", {})
    params = _cfg_get(dataset, "params", {})
    cache_dir = _cfg_get(params, "cache_dir", None)
    if not isinstance(cache_dir, str) or not cache_dir.strip():
        raise ValueError("Expected dataset.params.cache_dir in the UrbanSound8K config.")
    return Path(cache_dir).expanduser()


def load_cached_inputs(cache_dir: Path, split: str) -> np.ndarray:
    """Load cached UrbanSound8K log-mel inputs for one split.

    Parameters
    ----------
    cache_dir : pathlib.Path
        Directory containing split ``.npz`` cache files.
    split : str
        Split name: ``train``, ``val``, ``test``, or ``calibration``.

    Returns
    -------
    np.ndarray
        Float32 input tensor with shape ``(N, frames, mel_bins)``.

    Raises
    ------
    ValueError
        If the split name or cache payload is invalid.
    FileNotFoundError
        If the split cache file is absent.
    """

    normalized_split = str(split).strip().lower()
    if normalized_split not in VALID_SPLITS:
        allowed = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"Unknown UrbanSound8K split '{split}'. Expected one of: {allowed}.")
    split_path = Path(cache_dir) / f"{normalized_split}.npz"
    if not split_path.is_file():
        raise FileNotFoundError(f"UrbanSound8K cache split not found: {split_path}")
    with np.load(split_path, allow_pickle=False) as loaded:
        if "inputs" not in loaded.files:
            raise ValueError(f"UrbanSound8K cache split missing inputs array: {split_path}")
        inputs = loaded["inputs"].astype(np.float32, copy=False)
    if inputs.ndim != 3:
        raise ValueError("UrbanSound8K cached inputs must have shape (N, frames, mel_bins).")
    if inputs.shape[0] == 0:
        raise ValueError("UrbanSound8K cached inputs are empty.")
    if not np.all(np.isfinite(inputs)):
        raise ValueError("UrbanSound8K cached inputs must contain finite values.")
    return inputs


def build_profile(inputs: np.ndarray, *, real_window_count: int, seed: int) -> InputProfile:
    """Compute feature statistics and sample deterministic real windows.

    Parameters
    ----------
    inputs : np.ndarray
        Cached log-mel tensor with shape ``(N, frames, mel_bins)``.
    real_window_count : int
        Number of real cached windows to embed.
    seed : int
        Seed used for deterministic sampling.

    Returns
    -------
    InputProfile
        Computed statistics and sampled windows.

    Raises
    ------
    ValueError
        If the requested sample count is invalid.
    """

    if real_window_count <= 0:
        raise ValueError("real_window_count must be positive.")
    n_windows, window_size, n_channels = inputs.shape
    if real_window_count > n_windows:
        raise ValueError(
            f"real_window_count ({real_window_count}) exceeds available windows ({n_windows})."
        )
    flat = inputs.reshape(-1, n_channels)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_windows, size=real_window_count, replace=False)
    return InputProfile(
        window_size=window_size,
        n_channels=n_channels,
        channel_means=np.mean(flat, axis=0),
        channel_stds=np.std(flat, axis=0),
        channel_min=np.min(flat, axis=0),
        channel_max=np.max(flat, axis=0),
        channel_is_binary=np.all((flat == 0.0) | (flat == 1.0), axis=0),
        real_windows=inputs[indices].astype(np.float32, copy=False),
    )


def _format_array(values: Iterable[float], indent: int = 2, per_line: int = 8, fmt: str = "{:.6f}") -> str:
    """Format a flat array for inclusion in a C header.

    Parameters
    ----------
    values : Iterable[float]
        Values to format.
    indent : int, optional
        Spaces to indent each line.
    per_line : int, optional
        Number of values per line.
    fmt : str, optional
        Format string for each value.

    Returns
    -------
    str
        Formatted multi-line initializer contents.
    """

    values = list(values)
    lines: list[str] = []
    line = " " * indent
    for idx, value in enumerate(values):
        if idx > 0 and idx % per_line == 0:
            lines.append(line.rstrip())
            line = " " * indent
        line += fmt.format(value)
        if idx != len(values) - 1:
            line += ", "
    if line.strip():
        lines.append(line.rstrip())
    return "\n".join(lines)


def write_header(path: Path, profile: InputProfile) -> None:
    """Write an Arduino-compatible input profile header.

    Parameters
    ----------
    path : pathlib.Path
        Output header path.
    profile : InputProfile
        Statistics and real windows to serialize.

    Returns
    -------
    None
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    flat_windows = profile.real_windows.reshape(-1)
    header = [
        "#pragma once",
        "",
        "// Auto-generated by analysis_scripts/hil_noise_analysis/urbansound8k_input_profile.py",
        "// Contains cached log-mel statistics and representative windows for firmware input.",
        "",
        "#include <stdint.h>",
        "",
        f"static const int kInputWindowSize = {profile.window_size};",
        f"static const int kInputChannels = {profile.n_channels};",
        f"static const int kRealWindowCount = {profile.real_windows.shape[0]};",
        "",
        "static const float kChannelMeans[kInputChannels] = {",
        _format_array(profile.channel_means, fmt="{:.6f}"),
        "};",
        "",
        "static const float kChannelStds[kInputChannels] = {",
        _format_array(profile.channel_stds, fmt="{:.6f}"),
        "};",
        "",
        "static const float kChannelMin[kInputChannels] = {",
        _format_array(profile.channel_min, fmt="{:.6f}"),
        "};",
        "",
        "static const float kChannelMax[kInputChannels] = {",
        _format_array(profile.channel_max, fmt="{:.6f}"),
        "};",
        "",
        "static const uint8_t kChannelIsBinary[kInputChannels] = {",
        _format_array(profile.channel_is_binary.astype(np.uint8), fmt="{:d}"),
        "};",
        "",
        "static const float kRealWindows[",
        "    kRealWindowCount * kInputWindowSize * kInputChannels",
        "] = {",
        _format_array(flat_windows, fmt="{:.6f}"),
        "};",
        "",
    ]
    path.write_text("\n".join(header), encoding="utf-8")


def main() -> int:
    """Run the command-line profile exporter.

    Returns
    -------
    int
        Process exit status.
    """

    parser = argparse.ArgumentParser(
        description="Profile cached UrbanSound8K log-mel inputs for representative/real HIL sketches."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to audio config YAML.")
    parser.add_argument("--cache-dir", default=None, help="Override dataset.params.cache_dir.")
    parser.add_argument("--split", default="calibration", help="train, val, test, or calibration.")
    parser.add_argument(
        "--export-header",
        default=None,
        help="Write a C header with cached log-mel stats and real windows.",
    )
    parser.add_argument(
        "--real-window-count",
        type=int,
        default=10,
        help="Number of real windows to embed when exporting a header.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for sampled windows.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else resolve_cache_dir(Path(args.config))
    inputs = load_cached_inputs(cache_dir, args.split)
    profile = build_profile(inputs, real_window_count=args.real_window_count, seed=args.seed)

    print("UrbanSound8K cached log-mel input profiling")
    print(f"  split: {args.split}")
    print(f"  windows: {inputs.shape[0]}")
    print(f"  window_size: {profile.window_size}")
    print(f"  channels: {profile.n_channels}")
    print(f"  cache dir: {cache_dir}")
    print(
        "  value range: "
        f"{float(np.min(inputs)):.6f} to {float(np.max(inputs)):.6f}; "
        f"mean {float(np.mean(inputs)):.6f}; std {float(np.std(inputs)):.6f}"
    )

    if args.export_header:
        header_path = Path(args.export_header)
        write_header(header_path, profile)
        print(f"\nWrote header: {header_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
