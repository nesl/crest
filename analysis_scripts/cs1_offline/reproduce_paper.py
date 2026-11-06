# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Run the submitted CS1 analyses from an external research archive."""

import argparse
from pathlib import Path
import subprocess
import sys


def reproduce(archive, output, source_root=None):
    """Run reusable analyses with the submitted paper's inputs and settings.

    Parameters
    ----------
    archive : pathlib.Path
        Research archive retaining the original relative data paths.
    output : pathlib.Path
        Destination for regenerated results.
    source_root : pathlib.Path, optional
        Relocated source logs referenced by the observation CSV.

    Returns
    -------
    None
        Writes results and a numerical verification report.
    """
    scripts = Path(__file__).resolve().parent
    base = archive / "analysis_results/cs2_sleep_ratio_model_20261027"
    bank = (
        archive
        / "analysis_scripts/paper_plots/cs2_revision_20261028/data/selection_regret"
    )
    observations = archive / "outputs/cs2_board_load_sensitivity_20261025"
    commands = [
        (
            "calibration",
            [
                "--probe-long",
                base / "probe_v2/raw_measurements.csv",
                "--probe-short",
                base / "probe_100ms_v1/raw_measurements.csv",
                "--candidates",
                bank / "joined_candidates.csv",
                "--latencies",
                bank / "continuous_latency.csv",
                "--fit-decimals",
                3,
                "--exclude-pool",
                "200ms_cadenced",
            ],
            "calibration",
        ),
        (
            "heldout_prediction",
            [
                "--predictions",
                output / "calibration/predictions.csv",
                "--observations",
                observations / "observations.csv",
            ]
            + (["--archive-root", source_root] if source_root else []),
            "prediction",
        ),
        (
            "raw_selection_baseline",
            ["--predictions", output / "prediction/heldout_predictions.csv"],
            "selection",
        ),
        (
            "paired_transfer",
            [
                "--energy-records",
                archive
                / "analysis_results/overnight_latency_20261103/energy_records.json",
                "--calibration",
                output / "calibration/calibration.csv",
                "--holm-cohort",
                "random_panel",
            ],
            "transfer",
        ),
        (
            "policy_replay",
            [
                "--energy-records",
                archive / "analysis_results/policy_replay_20261103/energy_records.json",
                "--pairs",
                scripts / "paper_config/policy_pairs.json",
            ],
            "policy",
        ),
        (
            "repeats",
            [
                "--repeated-observations",
                observations / "repeated_observations.csv",
                "--joint-fit",
                output / "calibration/calibration.csv",
                "--pairs",
                scripts / "paper_config/repeat_pairs.json",
            ],
            "repeats",
        ),
        (
            "scaling_periods",
            [
                "--probe-100",
                base / "probe_100ms_v1/raw_measurements.csv",
                "--probe-500",
                base / "probe_v2/raw_measurements.csv",
            ],
            "scaling",
        ),
        (
            "direct_session",
            ["--measurements", base / "final_slot_v1/raw_measurements.csv"],
            "direct_session",
        ),
    ]
    output.mkdir(parents=True, exist_ok=True)
    for name, arguments, destination in commands:
        command = [
            sys.executable,
            str(scripts / f"{name}.py"),
            *map(str, arguments),
            "--output-dir",
            str(output / destination),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        (output / f"{name}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {output / (name + ".log")}')
        print(f"{name}: done", flush=True)
    subprocess.run(
        [sys.executable, str(scripts / "verify_paper.py"), "--results", str(output)],
        check=True,
    )


def main():
    """Run the paper recipe with explicit archive and output directories.

    Returns
    -------
    None
        Executes the offline reproduction pipeline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.archive_root.resolve(), args.output_dir.resolve(), args.source_root)


if __name__ == "__main__":
    main()
