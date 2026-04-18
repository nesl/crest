#!/usr/bin/env python3
"""Create STM32 CPU-clock sweep plots from an archived results folder.

This CLI reads one STM32 sweep archive that contains ``sweep_summary.csv`` and
writes the requested back-to-back and cadenced PNG plots into that same
results folder.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import statistics
from pathlib import Path
from typing import Any

SUMMARY_FILENAME = "sweep_summary.csv"
MASTER_SUCCESS_CODE = 1
BACK_TO_BACK_PHASE = "back_to_back"
CADENCED_PHASE = "cadenced"
EMBEDDED_MODE = "embedded"
EXTERNAL_FLASH_MODE = "external_flash"

BACK_TO_BACK_TIME_FILENAME = "back_to_back_inference_time_vs_cpu_frequency.png"
BACK_TO_BACK_ENERGY_FILENAME = "back_to_back_inference_energy_vs_cpu_frequency.png"
CADENCED_BOX_FILENAME = "cadenced_energy_per_inference_vs_cpu_frequency.png"
CADENCED_BOX_SPLIT_FILENAME = "cadenced_energy_per_inference_vs_cpu_frequency_by_mode.png"
CADENCED_BOX_FILTERED_FILENAME = "cadenced_energy_per_inference_vs_cpu_frequency_filtered.png"
CADENCED_BOX_FILTERED_SPLIT_FILENAME = "cadenced_energy_per_inference_vs_cpu_frequency_filtered_by_mode.png"
CADENCED_SCATTER_FILENAME = "cadenced_inference_time_vs_inference_energy.png"

MODE_LABELS = {
    EMBEDDED_MODE: "Embedded",
    EXTERNAL_FLASH_MODE: "External Flash",
}

MODE_COLORS = {
    EMBEDDED_MODE: "#4C78A8",
    EXTERNAL_FLASH_MODE: "#F58518",
}

CADENCED_COLOR_CYCLE = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
)
CADENCED_FILTERED_ENERGY_SIGMA_THRESHOLD = 4.5


def _safe_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` when the value is unusable.

    Parameters
    ----------
    value : Any
        Candidate numeric value parsed from the sweep summary CSV.

    Returns
    -------
    float | None
        Finite floating-point value when parsing succeeds, otherwise ``None``.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_int(value: Any) -> int | None:
    """Return an integer parsed from the input, or ``None``.

    Parameters
    ----------
    value : Any
        Candidate numeric value parsed from the sweep summary CSV.

    Returns
    -------
    int | None
        Parsed integer value when conversion succeeds, otherwise ``None``.
    """
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _is_successful_row(row: dict[str, str]) -> bool:
    """Return whether one summary row represents a successful HIL run.

    Parameters
    ----------
    row : dict[str, str]
        One row from ``sweep_summary.csv``.

    Returns
    -------
    bool
        ``True`` when the row reports ``return_code == 0`` and
        ``error_code == HIL_MASTER_SUCCESS``.
    """
    return _safe_int(row.get("return_code")) == 0 and _safe_int(row.get("error_code")) == MASTER_SUCCESS_CODE


def _load_summary_rows(results_dir: Path) -> list[dict[str, str]]:
    """Load ``sweep_summary.csv`` from one archived results folder.

    Parameters
    ----------
    results_dir : Path
        Path to one archived sweep results directory.

    Returns
    -------
    list[dict[str, str]]
        Parsed CSV rows in file order.

    Raises
    ------
    FileNotFoundError
        Raised when the results directory or summary CSV is missing.
    RuntimeError
        Raised when the summary CSV exists but has no data rows.
    """
    summary_path = results_dir / SUMMARY_FILENAME
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Expected summary CSV at: {summary_path}")
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in summary CSV: {summary_path}")
    return rows


def _collect_metric_series(
    rows: list[dict[str, str]],
    *,
    phase: str,
    metric: str,
    weight_storage_mode: str | None = None,
) -> dict[int, list[float]]:
    """Group successful metric values by requested CPU clock.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Parsed sweep summary rows.
    phase : str
        Sweep phase to retain, for example ``"back_to_back"`` or
        ``"cadenced"``.
    metric : str
        Summary CSV column to aggregate into box-plot samples.
    weight_storage_mode : str | None, optional
        Optional weight-storage filter such as ``"embedded"`` or
        ``"external_flash"``. When omitted, rows from all storage modes are
        considered.

    Returns
    -------
    dict[int, list[float]]
        Mapping from requested CPU frequency in MHz to successful metric
        values for that frequency.
    """
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row.get("phase") != phase:
            continue
        if weight_storage_mode is not None and row.get("weight_storage_mode") != weight_storage_mode:
            continue
        if not _is_successful_row(row):
            continue
        frequency = _safe_int(row.get("cpu_clock_mhz_requested"))
        value = _safe_float(row.get(metric))
        if frequency is None or value is None or value < 0.0:
            continue
        grouped.setdefault(frequency, []).append(value)
    return {freq: grouped[freq] for freq in sorted(grouped)}


def _iqr_bounds(values: list[float], *, whisker_scale: float = 2.0) -> tuple[float, float] | None:
    """Return IQR bounds for a list of values.

    Parameters
    ----------
    values : list[float]
        Numeric values to analyze.
    whisker_scale : float, optional
        Multiplier applied to the interquartile range when computing lower and
        upper Tukey-style bounds.

    Returns
    -------
    tuple[float, float] | None
        Lower and upper bounds using 1.5 * IQR, or ``None`` when too few
        samples are available.
    """
    if len(values) < 4:
        return None
    values = sorted(values)
    q1 = values[len(values) // 4]
    q3 = values[(3 * len(values)) // 4]
    iqr = q3 - q1
    return q1 - whisker_scale * iqr, q3 + whisker_scale * iqr


def _scaled_mad_bounds(
    values: list[float],
    *,
    sigma_threshold: float = CADENCED_FILTERED_ENERGY_SIGMA_THRESHOLD,
) -> tuple[float, float] | None:
    """Return robust median/MAD bounds for a list of values.

    Parameters
    ----------
    values : list[float]
        Numeric values to analyze.
    sigma_threshold : float, optional
        Scaled-MAD threshold to apply around the median.

    Returns
    -------
    tuple[float, float] | None
        Lower and upper bounds around the median, or ``None`` when too few
        samples are available or the group has zero spread.
    """
    if len(values) < 10:
        return None
    median = statistics.median(values)
    abs_deviations = [abs(value - median) for value in values]
    mad = statistics.median(abs_deviations)
    if mad <= 0.0:
        return None
    scaled_mad = 1.4826 * mad
    radius = sigma_threshold * scaled_mad
    return median - radius, median + radius


def _collect_cadenced_filtered_energy(
    rows: list[dict[str, str]],
) -> dict[str, dict[int, list[float]]]:
    """Collect cadenced inference energy excluding timing and energy outliers.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Parsed sweep summary rows.

    Returns
    -------
    dict[str, dict[int, list[float]]]
        Mapping from storage mode to frequency to filtered energy samples.
    """
    sleep_ratio_by: dict[str, dict[int, list[float]]] = {}
    energy_by: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if row.get("phase") != CADENCED_PHASE:
            continue
        mode = row.get("weight_storage_mode")
        if mode not in (EMBEDDED_MODE, EXTERNAL_FLASH_MODE):
            continue
        if not _is_successful_row(row):
            continue
        frequency = _safe_int(row.get("cpu_clock_mhz_requested"))
        energy = _safe_float(row.get("energy_mj_per_inference"))
        window_latency = _safe_float(row.get("window_latency_ms"))
        rtc_sleep = _safe_float(row.get("rtc_sleep_ms"))
        if frequency is None or energy is None or window_latency is None or rtc_sleep is None:
            continue
        if window_latency <= 0.0:
            continue
        ratio = rtc_sleep / window_latency
        sleep_ratio_by.setdefault(mode, {}).setdefault(frequency, []).append(ratio)
        energy_by.setdefault(mode, {}).setdefault(frequency, []).append((energy, ratio))

    filtered: dict[str, dict[int, list[float]]] = {}
    for mode, freq_map in energy_by.items():
        for frequency, pairs in freq_map.items():
            ratios = sleep_ratio_by.get(mode, {}).get(frequency, [])
            ratio_bounds = _iqr_bounds(ratios) if ratios else None
            if ratio_bounds is None:
                continue
            ratio_lo, ratio_hi = ratio_bounds
            timing_filtered = [
                energy for energy, ratio in pairs if ratio_lo <= ratio <= ratio_hi
            ]
            if not timing_filtered:
                continue
            energy_bounds = _scaled_mad_bounds(timing_filtered)
            for energy, ratio in pairs:
                if not (ratio_lo <= ratio <= ratio_hi):
                    continue
                if energy_bounds is not None:
                    energy_lo, energy_hi = energy_bounds
                    if not (energy_lo <= energy <= energy_hi):
                        continue
                filtered.setdefault(mode, {}).setdefault(frequency, []).append(energy)
    return filtered


def _collect_cadenced_scatter_points(
    rows: list[dict[str, str]],
) -> dict[str, dict[int, list[tuple[float, float]]]]:
    """Collect cadenced inference-time versus inference-energy points by mode and frequency.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Parsed sweep summary rows.

    Returns
    -------
    dict[str, dict[int, list[tuple[float, float]]]]
        Mapping from storage mode to CPU frequency to ``(inference_time_ms,
        inference_energy_mj)`` point pairs for successful cadenced runs.
    """
    grouped: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for row in rows:
        if row.get("phase") != CADENCED_PHASE:
            continue
        mode = row.get("weight_storage_mode")
        if mode not in (EMBEDDED_MODE, EXTERNAL_FLASH_MODE):
            continue
        if not _is_successful_row(row):
            continue
        frequency = _safe_int(row.get("cpu_clock_mhz_requested"))
        inference_time_ms = _safe_float(row.get("active_inference_latency_ms"))
        inference_energy_mj = _safe_float(row.get("energy_mj_per_inference"))
        if frequency is None or inference_time_ms is None or inference_energy_mj is None:
            continue
        if inference_time_ms < 0.0 or inference_energy_mj < 0.0:
            continue
        grouped.setdefault(mode, {}).setdefault(frequency, []).append((inference_time_ms, inference_energy_mj))
    return {mode: {freq: grouped[mode][freq] for freq in sorted(grouped[mode])} for mode in sorted(grouped)}


def _series_sample_count(series_by_mode: dict[str, dict[int, list[float]]], mode: str | None = None) -> int:
    """Return the total number of samples in one series mapping.

    Parameters
    ----------
    series_by_mode : dict[str, dict[int, list[float]]]
        Mapping from storage mode to frequency to metric samples.
    mode : str | None, optional
        Optional single mode to count. When ``None``, counts all modes.

    Returns
    -------
    int
        Total sample count.
    """
    modes = [mode] if mode is not None else list(series_by_mode)
    total = 0
    for selected_mode in modes:
        for values in series_by_mode.get(selected_mode, {}).values():
            total += len(values)
    return total


def _dropped_fraction_text(original_count: int, filtered_count: int) -> str:
    """Return a compact dropped-percentage label for filtered plots.

    Parameters
    ----------
    original_count : int
        Sample count before filtering.
    filtered_count : int
        Sample count after filtering.

    Returns
    -------
    str
        Human-readable label such as ``"10.0% dropped"``.
    """
    if original_count <= 0 or filtered_count >= original_count:
        return "0.0% dropped"
    dropped_fraction = ((original_count - filtered_count) / original_count) * 100.0
    return f"{dropped_fraction:.1f}% dropped"


def _import_plotting():
    """Import matplotlib in headless mode.

    Returns
    -------
    module
        Imported ``matplotlib.pyplot`` module with the ``Agg`` backend active.

    Raises
    ------
    RuntimeError
        Raised when matplotlib is unavailable in the current Python
        environment.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        current_python = Path(sys.executable).resolve()
        message_lines = [
            "matplotlib is required to generate the STM32 sweep plots.",
            f"Current interpreter: {current_python}",
        ]
        suggested_python = _find_python_with_matplotlib()
        if suggested_python is not None and suggested_python != current_python:
            message_lines.append(f"Suggested interpreter: {suggested_python}")
        raise RuntimeError(
            "\n".join(message_lines)
        ) from exc
    return plt


def _find_python_with_matplotlib() -> Path | None:
    """Return a likely Python interpreter that can import matplotlib.

    Returns
    -------
    Path | None
        Candidate Python interpreter path that successfully imports
        ``matplotlib``, or ``None`` when no known candidate succeeds.
    """
    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")
    candidates.append(Path.home() / "miniforge3" / "envs" / "tinyodomex" / "bin" / "python")

    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        probe = subprocess.run(
            [str(resolved), "-c", "import matplotlib"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return resolved
    return None


def _maybe_reexec_with_matplotlib(argv: list[str] | None = None) -> None:
    """Re-exec under a Python that has matplotlib when available.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argv override for testing. When ``None``, uses ``sys.argv``.
    """
    try:
        import matplotlib  # noqa: F401
    except Exception:
        suggested = _find_python_with_matplotlib()
        current = Path(sys.executable).resolve()
        if suggested is not None and suggested != current:
            args = argv if argv is not None else sys.argv
            print(
                "matplotlib is missing in the current interpreter. Re-running with:\n"
                f"  {suggested}",
                file=sys.stderr,
            )
            os.execv(str(suggested), [str(suggested), *args])


def _even_jitter(count: int, span: float) -> list[float]:
    """Return deterministic jitter offsets centered at zero.

    Parameters
    ----------
    count : int
        Number of offsets to generate.
    span : float
        Total width of the jitter span.

    Returns
    -------
    list[float]
        Offsets centered at zero with deterministic spacing.
    """
    if count <= 1:
        return [0.0] * max(count, 0)
    step = span / (count - 1)
    start = -span / 2.0
    return [start + idx * step for idx in range(count)]


def _write_grouped_pointplot(
    output_path: Path,
    frequencies: list[int],
    series_by_mode: dict[str, dict[int, list[float]]],
    *,
    title: str,
    ylabel: str,
) -> None:
    """Write grouped scatter points with mean and std error bars.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.
    frequencies : list[int]
        Ordered CPU frequencies shown on the x-axis.
    series_by_mode : dict[str, dict[int, list[float]]]
        Nested mapping from storage mode to frequency to metric samples.
    title : str
        Plot title.
    ylabel : str
        Y-axis label for the plotted metric.
    """
    plt = _import_plotting()
    fig, ax = plt.subplots(figsize=(10, 6))

    base_positions = list(range(len(frequencies)))
    offsets = {
        EMBEDDED_MODE: -0.18,
        EXTERNAL_FLASH_MODE: 0.18,
    }
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for mode in (EMBEDDED_MODE, EXTERNAL_FLASH_MODE):
        color = MODE_COLORS[mode]
        for idx, frequency in enumerate(frequencies):
            metric_values = series_by_mode.get(mode, {}).get(frequency)
            if not metric_values:
                continue
            base_x = base_positions[idx] + offsets[mode]
            jitters = _even_jitter(len(metric_values), span=0.18)
            x_values = [base_x + jitter for jitter in jitters]
            ax.scatter(x_values, metric_values, color=color, s=44, alpha=0.75)
            mean_value = statistics.fmean(metric_values)
            std_value = statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0
            err = ax.errorbar(
                [base_x],
                [mean_value],
                yerr=[std_value],
                color=color,
                fmt="o",
                markersize=7,
                capsize=4,
                linewidth=1.4,
                alpha=0.9,
                zorder=3,
            )
            if not legend_handles:
                legend_handles.append(err)
                legend_labels.append(MODE_LABELS[mode])
        if len(legend_handles) == 1 and MODE_LABELS[mode] not in legend_labels:
            legend_handles.append(
                ax.scatter([], [], color=color, s=44, alpha=0.75)
            )
            legend_labels.append(MODE_LABELS[mode])

    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(freq) for freq in frequencies])
    ax.set_xlim(-0.6, len(frequencies) - 0.4)
    ax.set_xlabel("CPU Frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_single_pointplot(
    output_path: Path,
    frequencies: list[int],
    series_by_frequency: dict[int, list[float]],
    *,
    title: str,
    ylabel: str,
    color: str | None,
    show_error_bars: bool = True,
) -> None:
    """Write single-series scatter points with mean and std error bars.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.
    frequencies : list[int]
        Ordered CPU frequencies shown on the x-axis.
    series_by_frequency : dict[int, list[float]]
        Mapping from frequency to metric samples.
    title : str
        Plot title.
    ylabel : str
        Y-axis label for the plotted metric.
    color : str | None
        Optional point color. When ``None``, matplotlib chooses the default.
    show_error_bars : bool, optional
        When True, overlay mean and standard deviation error bars.
    """
    plt = _import_plotting()
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, frequency in enumerate(frequencies):
        metric_values = series_by_frequency.get(frequency)
        if not metric_values:
            continue
        base_x = float(idx)
        jitters = _even_jitter(len(metric_values), span=0.24)
        x_values = [base_x + jitter for jitter in jitters]
        scatter_kwargs = {"s": 44, "alpha": 0.75}
        if color is not None:
            scatter_kwargs["color"] = color
        ax.scatter(x_values, metric_values, **scatter_kwargs)
        if show_error_bars:
            mean_value = statistics.fmean(metric_values)
            std_value = statistics.pstdev(metric_values) if len(metric_values) > 1 else 0.0
            error_kwargs = {
                "fmt": "o",
                "markersize": 7,
                "capsize": 4,
                "linewidth": 1.4,
                "alpha": 0.9,
                "zorder": 3,
            }
            if color is not None:
                error_kwargs["color"] = color
            ax.errorbar(
                [base_x],
                [mean_value],
                yerr=[std_value],
                **error_kwargs,
            )

    ax.set_xticks(list(range(len(frequencies))))
    ax.set_xticklabels([str(freq) for freq in frequencies])
    ax.set_xlim(-0.6, len(frequencies) - 0.4)
    ax.set_xlabel("CPU Frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_cadenced_scatter(
    output_path: Path,
    scatter_points: dict[str, dict[int, list[tuple[float, float]]]],
) -> None:
    """Write the cadenced inference-time versus inference-energy scatter plot.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.
    scatter_points : dict[str, dict[int, list[tuple[float, float]]]]
        Mapping from storage mode to frequency to ``(inference_time_ms,
        inference_energy_mj)`` point pairs.
    """
    plt = _import_plotting()
    fig, ax = plt.subplots(figsize=(10, 6))

    marker_by_mode = {EMBEDDED_MODE: "o", EXTERNAL_FLASH_MODE: "s"}
    for mode in (EMBEDDED_MODE, EXTERNAL_FLASH_MODE):
        mode_points = scatter_points.get(mode, {})
        for index, frequency in enumerate(sorted(mode_points)):
            points = mode_points[frequency]
            if not points:
                continue
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            color = CADENCED_COLOR_CYCLE[index % len(CADENCED_COLOR_CYCLE)]
            ax.scatter(
                x_values,
                y_values,
                label=f"{MODE_LABELS[mode]} {frequency} MHz",
                color=color,
                marker=marker_by_mode[mode],
                s=56,
                alpha=0.85,
            )

    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Inference Energy (mJ / inference)")
    ax.set_title("Cadenced Inference Time vs Inference Energy")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Mode / CPU Frequency", loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_cadenced_split_pointplot(
    output_path: Path,
    frequencies: list[int],
    series_by_mode: dict[str, dict[int, list[float]]],
    *,
    filtered: bool = False,
    filtered_label_by_mode: dict[str, str] | None = None,
) -> None:
    """Write vertical subplots for cadenced inference energy by storage mode.

    Parameters
    ----------
    output_path : Path
        Destination PNG path.
    frequencies : list[int]
        Ordered CPU frequencies shown on the x-axis.
    series_by_mode : dict[str, dict[int, list[float]]]
        Mapping from storage mode to frequency to metric samples.
    filtered : bool, optional
        When True, annotate subplot titles to indicate the data was filtered.
    filtered_label_by_mode : dict[str, str] | None, optional
        Optional per-mode subtitle label such as dropped-percentage text for
        filtered plots.
    """
    plt = _import_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    filtered_label_by_mode = filtered_label_by_mode or {}

    for axis, mode in zip(axes, (EMBEDDED_MODE, EXTERNAL_FLASH_MODE)):
        color = MODE_COLORS[mode]
        for idx, frequency in enumerate(frequencies):
            metric_values = series_by_mode.get(mode, {}).get(frequency)
            if not metric_values:
                continue
            base_x = float(idx)
            jitters = _even_jitter(len(metric_values), span=0.24)
            x_values = [base_x + jitter for jitter in jitters]
            axis.scatter(x_values, metric_values, color=color, s=44, alpha=0.75)
        axis.set_ylabel("Inference Energy (mJ / inference)")
        if filtered:
            label = filtered_label_by_mode.get(mode, "0.0% dropped")
            axis.set_title(f"Cadenced Inference Energy (Filtered, {label}) - {MODE_LABELS[mode]}")
        else:
            axis.set_title(f"Cadenced Inference Energy - {MODE_LABELS[mode]}")
        axis.grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[-1].set_xticks(list(range(len(frequencies))))
    axes[-1].set_xticklabels([str(freq) for freq in frequencies])
    axes[-1].set_xlabel("CPU Frequency (MHz)")
    axes[-1].set_xlim(-0.6, len(frequencies) - 0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_plots(results_dir: Path) -> list[Path]:
    """Generate the requested STM32 sweep plots into the results directory.

    Parameters
    ----------
    results_dir : Path
        Path to the archived sweep directory that contains
        ``sweep_summary.csv``.

    Returns
    -------
    list[Path]
        Generated PNG paths in write order.

    Raises
    ------
    RuntimeError
        Raised when the summary does not contain the successful rows needed to
        build one or more requested plots.
    """
    rows = _load_summary_rows(results_dir)

    back_to_back_time = {
        EMBEDDED_MODE: _collect_metric_series(
            rows,
            phase=BACK_TO_BACK_PHASE,
            metric="active_inference_latency_ms",
            weight_storage_mode=EMBEDDED_MODE,
        ),
        EXTERNAL_FLASH_MODE: _collect_metric_series(
            rows,
            phase=BACK_TO_BACK_PHASE,
            metric="active_inference_latency_ms",
            weight_storage_mode=EXTERNAL_FLASH_MODE,
        ),
    }
    back_to_back_energy = {
        EMBEDDED_MODE: _collect_metric_series(
            rows,
            phase=BACK_TO_BACK_PHASE,
            metric="energy_mj_per_inference",
            weight_storage_mode=EMBEDDED_MODE,
        ),
        EXTERNAL_FLASH_MODE: _collect_metric_series(
            rows,
            phase=BACK_TO_BACK_PHASE,
            metric="energy_mj_per_inference",
            weight_storage_mode=EXTERNAL_FLASH_MODE,
        ),
    }
    cadenced_energy = {
        EMBEDDED_MODE: _collect_metric_series(
            rows,
            phase=CADENCED_PHASE,
            metric="energy_mj_per_inference",
            weight_storage_mode=EMBEDDED_MODE,
        ),
        EXTERNAL_FLASH_MODE: _collect_metric_series(
            rows,
            phase=CADENCED_PHASE,
            metric="energy_mj_per_inference",
            weight_storage_mode=EXTERNAL_FLASH_MODE,
        ),
    }
    cadenced_energy_filtered = _collect_cadenced_filtered_energy(rows)
    cadenced_scatter = _collect_cadenced_scatter_points(rows)
    filtered_drop_label_by_mode = {
        mode: _dropped_fraction_text(
            _series_sample_count(cadenced_energy, mode),
            _series_sample_count(cadenced_energy_filtered, mode),
        )
        for mode in (EMBEDDED_MODE, EXTERNAL_FLASH_MODE)
    }
    filtered_drop_label_all = _dropped_fraction_text(
        _series_sample_count(cadenced_energy),
        _series_sample_count(cadenced_energy_filtered),
    )

    frequencies = sorted(
        set(back_to_back_time[EMBEDDED_MODE])
        | set(back_to_back_time[EXTERNAL_FLASH_MODE])
        | set(cadenced_energy[EMBEDDED_MODE])
        | set(cadenced_energy[EXTERNAL_FLASH_MODE])
        | set(cadenced_energy_filtered.get(EMBEDDED_MODE, {}))
        | set(cadenced_energy_filtered.get(EXTERNAL_FLASH_MODE, {}))
        | set(cadenced_scatter.get(EMBEDDED_MODE, {}))
        | set(cadenced_scatter.get(EXTERNAL_FLASH_MODE, {}))
    )
    if not frequencies:
        raise RuntimeError(f"No successful runs with plot-ready metrics found in {results_dir / SUMMARY_FILENAME}")
    has_back_to_back_time = bool(back_to_back_time[EMBEDDED_MODE] or back_to_back_time[EXTERNAL_FLASH_MODE])
    has_back_to_back_energy = bool(back_to_back_energy[EMBEDDED_MODE] or back_to_back_energy[EXTERNAL_FLASH_MODE])
    if not cadenced_energy[EMBEDDED_MODE] and not cadenced_energy[EXTERNAL_FLASH_MODE]:
        raise RuntimeError("No successful cadenced rows with inference-energy data were found.")
    if not cadenced_scatter:
        raise RuntimeError("No successful cadenced rows with inference-time/inference-energy data were found.")

    output_paths: list[Path] = []

    if has_back_to_back_time:
        back_to_back_time_path = results_dir / BACK_TO_BACK_TIME_FILENAME
        _write_grouped_pointplot(
            back_to_back_time_path,
            frequencies,
            back_to_back_time,
            title="Back-to-Back Inference Time by CPU Frequency",
            ylabel="Inference Time (ms)",
        )
        output_paths.append(back_to_back_time_path)
    if has_back_to_back_energy:
        back_to_back_energy_path = results_dir / BACK_TO_BACK_ENERGY_FILENAME
        _write_grouped_pointplot(
            back_to_back_energy_path,
            frequencies,
            back_to_back_energy,
            title="Back-to-Back Inference Energy by CPU Frequency",
            ylabel="Inference Energy (mJ / inference)",
        )
        output_paths.append(back_to_back_energy_path)

    cadenced_box_path = results_dir / CADENCED_BOX_FILENAME
    _write_grouped_pointplot(
        cadenced_box_path,
        frequencies,
        cadenced_energy,
        title="Cadenced Inference Energy by CPU Frequency",
        ylabel="Inference Energy (mJ / inference)",
    )
    output_paths.append(cadenced_box_path)

    cadenced_split_path = results_dir / CADENCED_BOX_SPLIT_FILENAME
    _write_cadenced_split_pointplot(cadenced_split_path, frequencies, cadenced_energy)
    output_paths.append(cadenced_split_path)

    if cadenced_energy_filtered:
        cadenced_filtered_path = results_dir / CADENCED_BOX_FILTERED_FILENAME
        _write_grouped_pointplot(
            cadenced_filtered_path,
            frequencies,
            cadenced_energy_filtered,
            title=f"Cadenced Inference Energy by CPU Frequency (Filtered, {filtered_drop_label_all})",
            ylabel="Inference Energy (mJ / inference)",
        )
        output_paths.append(cadenced_filtered_path)

        cadenced_filtered_split_path = results_dir / CADENCED_BOX_FILTERED_SPLIT_FILENAME
        _write_cadenced_split_pointplot(
            cadenced_filtered_split_path,
            frequencies,
            cadenced_energy_filtered,
            filtered=True,
            filtered_label_by_mode=filtered_drop_label_by_mode,
        )
        output_paths.append(cadenced_filtered_split_path)

    cadenced_scatter_path = results_dir / CADENCED_SCATTER_FILENAME
    _write_cadenced_scatter(cadenced_scatter_path, cadenced_scatter)
    output_paths.append(cadenced_scatter_path)
    return output_paths


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser for the plotting CLI.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create back-to-back and cadenced STM32 CPU-clock sweep plots from a "
            "results folder containing sweep_summary.csv."
        ),
        epilog=(
            "Example:\n"
            "  python analysis_scripts/stm32_example_project/plot_stm32_cpu_clock_sweep.py "
            "analysis_scripts/stm32_example_project/results/stm32_cpu_clock_sweep_20260411T054113Z"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help=(
            "Path to the archived sweep results folder that contains "
            "sweep_summary.csv. Example: "
            "analysis_scripts/stm32_example_project/results/"
            "stm32_cpu_clock_sweep_20260411T054113Z"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the plotting CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Optional argument vector for tests. When ``None``, arguments are read
        from ``sys.argv``.

    Returns
    -------
    int
        Process exit status code. Returns ``0`` after writing all requested
        plot files.
    """
    _maybe_reexec_with_matplotlib(argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_paths = generate_plots(args.results_dir.resolve())
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
