#!/usr/bin/env python3
"""Plot paired STM32 OXIOD back-to-back vs cadenced motivation data."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update(
    {
        "axes.edgecolor": "black",
        "axes.linewidth": 1.15,
        "axes.titlesize": 14.3,
        "axes.labelsize": 13.2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 13.5,
        "font.size": 11,
    }
)


DEFAULT_LOG = (
    "models/CREST_case1/models/OxIOD_STM32_CADENCED_case2_2_t3/"
    "log_NAS_OxIOD_STM32_CADENCED_case2_2_t3.csv"
)
DEFAULT_OUTDIR = "outputs/plots"
DEFAULT_DPI = 400


def as_float(row: dict[str, str], key: str) -> float:
    """Return a CSV field as float, or NaN when absent."""

    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else math.nan
    except ValueError:
        return math.nan


def r_squared(xs: list[float], ys: list[float]) -> float:
    """Return ordinary least-squares R^2 for a simple linear fit."""

    if len(xs) < 3:
        return math.nan
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    ssx = sum((x - x_mean) ** 2 for x in xs)
    ssy = sum((y - y_mean) ** 2 for y in ys)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    if ssx <= 0.0 or ssy <= 0.0:
        return math.nan
    return (sxy / math.sqrt(ssx * ssy)) ** 2


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return slope, intercept, and R^2 for a simple linear fit."""

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    ssx = sum((x - x_mean) ** 2 for x in xs)
    if ssx <= 0.0:
        return math.nan, math.nan, math.nan
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / ssx
    intercept = y_mean - slope * x_mean
    return slope, intercept, r_squared(xs, ys)


def load_paired_rows(
    log_path: Path,
    *,
    cpu_mhz: str,
    min_latency_ms: float,
    max_latency_ms: float,
) -> list[dict[str, Any]]:
    """Load rows with successful B2B and cadenced phases for one clock."""

    rows: list[dict[str, Any]] = []
    with log_path.open(newline="") as fp:
        for line_number, raw in enumerate(csv.DictReader(fp), start=2):
            b2b_latency = as_float(raw, "latency_ms")
            b2b_energy = as_float(raw, "energy_mj_per_inference")
            cadenced_latency = as_float(raw, "cadenced_active_inference_latency_ms")
            cadenced_energy = as_float(raw, "cadenced_energy_mj_per_window")
            if not cadenced_latency > 0:
                cadenced_latency = b2b_latency
            if not (
                raw.get("error_code") == "1"
                and raw.get("cadenced_error_code") == "0"
                and raw.get("cpu_clock_mhz_requested") == cpu_mhz
                and min_latency_ms <= b2b_latency <= max_latency_ms
                and b2b_energy > 0
                and min_latency_ms <= cadenced_latency <= max_latency_ms
                and cadenced_energy > 0
            ):
                continue
            rows.append(
                {
                    "row": str(line_number),
                    "b2b_latency_ms": b2b_latency,
                    "b2b_energy_mj_per_inference": b2b_energy,
                    "cadenced_active_latency_ms": cadenced_latency,
                    "cadenced_energy_mj_per_window": cadenced_energy,
                    "cpu_mhz": raw.get("cpu_clock_mhz_requested", ""),
                    "quantization": raw.get("quantization_mode", ""),
                    "nb_filters": raw.get("hparam__nb_filters", ""),
                    "kernel_size": raw.get("hparam__kernel_size", ""),
                    "dilations": raw.get("hparam__dilations", ""),
                }
            )
    return rows


def flip_score(lower_b2b: dict[str, Any], higher_b2b: dict[str, Any]) -> float:
    """Score a ranking flip by visible B2B separation and cadenced reversal."""

    b2b_gap = abs(
        higher_b2b["b2b_energy_mj_per_inference"]
        - lower_b2b["b2b_energy_mj_per_inference"]
    )
    cadenced_gap = abs(
        lower_b2b["cadenced_energy_mj_per_window"]
        - higher_b2b["cadenced_energy_mj_per_window"]
    )
    latency_gap = abs(
        higher_b2b["b2b_latency_ms"] - lower_b2b["b2b_latency_ms"]
    )
    return min(b2b_gap, 50.0) + 3.0 * cadenced_gap + 0.3 * min(latency_gap, 50.0)


def find_flip_pair(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a high-contrast pair that flips energy ranking."""

    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for first, second in itertools.combinations(rows, 2):
        first_b2b = first["b2b_energy_mj_per_inference"]
        second_b2b = second["b2b_energy_mj_per_inference"]
        first_cadenced = first["cadenced_energy_mj_per_window"]
        second_cadenced = second["cadenced_energy_mj_per_window"]
        if (first_b2b - second_b2b) * (first_cadenced - second_cadenced) >= 0:
            continue
        if first_b2b < second_b2b and first_cadenced > second_cadenced:
            lower_b2b, higher_b2b = first, second
        elif second_b2b < first_b2b and second_cadenced > first_cadenced:
            lower_b2b, higher_b2b = second, first
        else:
            continue
        candidates.append((flip_score(lower_b2b, higher_b2b), lower_b2b, higher_b2b))
    if not candidates:
        raise ValueError("No ranking flip found in the filtered paired rows.")
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, lower_b2b, higher_b2b = candidates[0]
    return lower_b2b, higher_b2b


def row_by_id(rows: list[dict[str, Any]], row_id: str) -> dict[str, Any]:
    """Return one loaded row by original CSV line number."""

    for row in rows:
        if row["row"] == row_id:
            return row
    raise ValueError(f"Requested row {row_id} was not present after filtering.")


def write_points_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the normalized paired points used by the plot."""

    fieldnames = [
        "row",
        "b2b_latency_ms",
        "b2b_energy_mj_per_inference",
        "cadenced_active_latency_ms",
        "cadenced_energy_mj_per_window",
        "cpu_mhz",
        "quantization",
        "nb_filters",
        "kernel_size",
        "dilations",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    y_label: str,
    highlights: dict[str, tuple[str, str]],
    add_inset: bool,
    inset_loc: str,
    inset_show_ticks: bool,
    flip_participants: set[str] | None,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    """Draw one scatter panel and optional zoom callout."""

    normal = [row for row in rows if row["row"] not in highlights]
    if flip_participants is None:
        ax.scatter(
            [row[x_key] for row in normal],
            [row[y_key] for row in normal],
            s=28,
            alpha=0.45,
            color="#4c78a8",
            label="_nolegend_",
        )
    else:
        nonflipping = [row for row in normal if row["row"] not in flip_participants]
        flipping = [row for row in normal if row["row"] in flip_participants]
        ax.scatter(
            [row[x_key] for row in nonflipping],
            [row[y_key] for row in nonflipping],
            s=28,
            alpha=0.45,
            color="#4c78a8",
            label="_nolegend_",
        )
        ax.scatter(
            [row[x_key] for row in flipping],
            [row[y_key] for row in flipping],
            s=30,
            alpha=0.58,
            color="#9467bd",
            label="Participates in a flip",
        )

    xs = [row[x_key] for row in rows]
    ys = [row[y_key] for row in rows]
    slope, intercept, r2 = linear_fit(xs, ys)
    lo, hi = min(xs), max(xs)
    ax.plot(
        [lo, hi],
        [slope * lo + intercept, slope * hi + intercept],
        color="#333333",
        linewidth=1.0,
        alpha=0.72,
        label="Linear fit",
    )

    highlighted_rows = []
    for row_id, (label, color) in highlights.items():
        row = row_by_id(rows, row_id)
        highlighted_rows.append(row)
        ax.scatter(
            [row[x_key]],
            [row[y_key]],
            s=92,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=f"Candidate {label}",
        )
        if label == "A":
            xytext = (0, 7)
            horizontal_alignment = "center"
            vertical_alignment = "bottom"
        elif title.startswith("Cadenced"):
            xytext = (0, -9)
            horizontal_alignment = "center"
            vertical_alignment = "top"
        else:
            xytext = (8, -3)
            horizontal_alignment = "left"
            vertical_alignment = "top"
        ax.annotate(
            label,
            xy=(row[x_key], row[y_key]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=color,
            ha=horizontal_alignment,
            va=vertical_alignment,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
        )

    if xlim[0] <= 200 <= xlim[1]:
        ax.axvline(200, color="#555555", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(f"{title}  R$^2$={r2:.2f}", pad=8)
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", color="#bfbfbf", alpha=0.42, linewidth=0.8)
    ax.grid(True, which="minor", color="#d9d9d9", alpha=0.35, linewidth=0.5)
    ax.minorticks_on()

    if not add_inset:
        return

    if inset_loc.startswith("lower"):
        inset = inset_axes(
            ax,
            width="36%",
            height="45%",
            loc=inset_loc,
            bbox_to_anchor=(0.14, 0.08, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
    else:
        inset = inset_axes(ax, width="36%", height="45%", loc=inset_loc, borderpad=1.0)
    if flip_participants is None:
        inset.scatter(
            [row[x_key] for row in normal],
            [row[y_key] for row in normal],
            s=18,
            alpha=0.25,
            color="#4c78a8",
        )
    else:
        nonflipping = [row for row in normal if row["row"] not in flip_participants]
        flipping = [row for row in normal if row["row"] in flip_participants]
        inset.scatter(
            [row[x_key] for row in nonflipping],
            [row[y_key] for row in nonflipping],
            s=18,
            alpha=0.25,
            color="#4c78a8",
        )
        inset.scatter(
            [row[x_key] for row in flipping],
            [row[y_key] for row in flipping],
            s=18,
            alpha=0.34,
            color="#9467bd",
        )
    inset.plot(
        [lo, hi],
        [slope * lo + intercept, slope * hi + intercept],
        color="#333333",
        linewidth=0.9,
        alpha=0.72,
    )
    for row_id, (label, color) in highlights.items():
        row = row_by_id(rows, row_id)
        inset.scatter(
            [row[x_key]],
            [row[y_key]],
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )
        inset.text(row[x_key] + 0.7, row[y_key] + 0.7, label, color=color, weight="bold")

    x_values = [row[x_key] for row in highlighted_rows]
    y_values = [row[y_key] for row in highlighted_rows]
    x_pad = max(5.0, (max(x_values) - min(x_values)) * 0.4)
    y_pad = max(5.0, (max(y_values) - min(y_values)) * 1.4)
    inset.set_xlim(max(xlim[0], min(x_values) - x_pad), min(xlim[1], max(x_values) + x_pad))
    inset.set_ylim(max(ylim[0], min(y_values) - y_pad), min(ylim[1], max(y_values) + y_pad))
    inset.grid(True, alpha=0.25, linewidth=0.35)
    if inset_show_ticks:
        inset.tick_params(labelsize=6, length=2)
    else:
        inset.tick_params(
            labelleft=False,
            labelbottom=False,
            left=False,
            bottom=False,
        )
    if inset_loc.startswith("lower"):
        mark_inset(ax, inset, loc1=1, loc2=3, fc="none", ec="#555555", lw=0.8, alpha=0.8)
    else:
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#555555", lw=0.8, alpha=0.8)


def count_flips(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Return number of ranking flips and candidate pairs."""

    pairs = 0
    flips = 0
    for first, second in itertools.combinations(rows, 2):
        b2b_delta = (
            first["b2b_energy_mj_per_inference"]
            - second["b2b_energy_mj_per_inference"]
        )
        cadenced_delta = (
            first["cadenced_energy_mj_per_window"]
            - second["cadenced_energy_mj_per_window"]
        )
        if b2b_delta == 0 or cadenced_delta == 0:
            continue
        pairs += 1
        if b2b_delta * cadenced_delta < 0:
            flips += 1
    return flips, pairs


def flip_participant_rows(rows: list[dict[str, Any]]) -> set[str]:
    """Return row IDs that participate in at least one ranking flip."""

    participants: set[str] = set()
    for first, second in itertools.combinations(rows, 2):
        b2b_delta = (
            first["b2b_energy_mj_per_inference"]
            - second["b2b_energy_mj_per_inference"]
        )
        cadenced_delta = (
            first["cadenced_energy_mj_per_window"]
            - second["cadenced_energy_mj_per_window"]
        )
        if b2b_delta == 0 or cadenced_delta == 0:
            continue
        if b2b_delta * cadenced_delta < 0:
            participants.add(first["row"])
            participants.add(second["row"])
    return participants


def build_plot(
    rows: list[dict[str, Any]],
    *,
    highlights: dict[str, tuple[str, str]],
    out_path: Path,
    dpi: int,
    add_inset: bool,
    color_flip_participants: bool,
    layout: str,
    xlim: tuple[float, float],
    b2b_ylim: tuple[float, float],
    cadenced_ylim: tuple[float, float],
) -> None:
    """Build B2B/cadenced panels."""

    flip_participants = flip_participant_rows(rows) if color_flip_participants else None
    if layout == "side-by-side":
        share_y = b2b_ylim == cadenced_ylim
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), dpi=dpi, sharex=True, sharey=share_y)
        b2b_ax, cadenced_ax = axes
    else:
        share_y = b2b_ylim == cadenced_ylim
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), dpi=dpi, sharex=True, sharey=share_y)
        b2b_ax, cadenced_ax = axes
    add_panel(
        b2b_ax,
        rows,
        title="Continuous",
        x_key="b2b_latency_ms",
        y_key="b2b_energy_mj_per_inference",
        y_label="Energy per inference (mJ)",
        highlights=highlights,
        add_inset=False,
        inset_loc="upper left",
        inset_show_ticks=False,
        flip_participants=flip_participants,
        xlim=xlim,
        ylim=b2b_ylim,
    )
    add_panel(
        cadenced_ax,
        rows,
        title="Cadenced",
        x_key="cadenced_active_latency_ms",
        y_key="cadenced_energy_mj_per_window",
        y_label="Energy per cadence window (mJ)",
        highlights=highlights,
        add_inset=add_inset and layout != "side-by-side",
        inset_loc="lower left",
        inset_show_ticks=True,
        flip_participants=flip_participants,
        xlim=xlim,
        ylim=cadenced_ylim,
    )
    if layout == "side-by-side":
        b2b_ax.set_xlabel("Latency / active latency (ms)")
        cadenced_ax.set_xlabel("Latency / active latency (ms)")
        cadenced_ax.set_ylabel("")
    else:
        cadenced_ax.set_xlabel("Latency / active latency (ms)")

    handles, labels = axes[0].get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    fig.legend(
        unique_handles,
        unique_labels,
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.8,
        columnspacing=1.2,
        markerscale=1.15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90 if layout == "side-by-side" else 0.935])
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path(DEFAULT_LOG))
    parser.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUTDIR))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--cpu-mhz", default="200")
    parser.add_argument("--min-latency-ms", type=float, default=0.0)
    parser.add_argument("--max-latency-ms", type=float, default=200.0)
    parser.add_argument("--candidate-a-row", default="243")
    parser.add_argument("--candidate-b-row", default="134")
    parser.add_argument("--auto-pair", action="store_true")
    parser.add_argument("--color-flip-participants", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("stacked", "side-by-side"),
        default="stacked",
    )
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=205.0)
    parser.add_argument("--b2b-y-min", type=float, default=0.0)
    parser.add_argument("--b2b-y-max", type=float, default=200.0)
    parser.add_argument("--cadenced-y-min", type=float, default=0.0)
    parser.add_argument("--cadenced-y-max", type=float, default=200.0)
    parser.add_argument("--no-inset", action="store_true")
    parser.add_argument(
        "--stem",
        default="oxiod_stm32_cadenced_motivation_200mhz_stacked_callout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = load_paired_rows(
        args.log,
        cpu_mhz=str(args.cpu_mhz),
        min_latency_ms=args.min_latency_ms,
        max_latency_ms=args.max_latency_ms,
    )
    if not rows:
        raise ValueError("No paired rows matched the requested filters.")

    if args.auto_pair:
        candidate_a, candidate_b = find_flip_pair(rows)
    else:
        candidate_a = row_by_id(rows, str(args.candidate_a_row))
        candidate_b = row_by_id(rows, str(args.candidate_b_row))
    highlights = {
        candidate_a["row"]: ("A", "#d62728"),
        candidate_b["row"]: ("B", "#ff7f0e"),
    }

    points_path = args.outdir / f"{args.stem}_points.csv"
    write_points_csv(rows, points_path)
    out_path = args.outdir / args.stem
    build_plot(
        rows,
        highlights=highlights,
        out_path=out_path,
        dpi=args.dpi,
        add_inset=not args.no_inset,
        color_flip_participants=args.color_flip_participants,
        layout=args.layout,
        xlim=(args.x_min, args.x_max),
        b2b_ylim=(args.b2b_y_min, args.b2b_y_max),
        cadenced_ylim=(args.cadenced_y_min, args.cadenced_y_max),
    )

    flips, pairs = count_flips(rows)
    print(f"wrote {out_path.with_suffix('.png')}")
    print(f"wrote {out_path.with_suffix('.pdf')}")
    print(f"wrote {points_path}")
    print(f"paired rows: {len(rows)}")
    print(f"ranking flips: {flips}/{pairs} ({flips / pairs:.1%})")
    participants = flip_participant_rows(rows)
    print(f"flip participants: {len(participants)}/{len(rows)}")
    for label, row in (("A", candidate_a), ("B", candidate_b)):
        print(
            f"{label} row {row['row']}: "
            f"B2B=({row['b2b_latency_ms']:.3f} ms, "
            f"{row['b2b_energy_mj_per_inference']:.3f} mJ), "
            f"cadenced=({row['cadenced_active_latency_ms']:.3f} ms, "
            f"{row['cadenced_energy_mj_per_window']:.3f} mJ), "
            f"filters={row['nb_filters']}, kernel={row['kernel_size']}, "
            f"dilations={row['dilations']}"
        )


if __name__ == "__main__":
    main()
