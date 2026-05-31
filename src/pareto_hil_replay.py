#!/usr/bin/env python3
"""Replay TinyODOM Pareto-front candidates through a target HIL config."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from tinyodom.pareto_replay import DEVICE_OPTION_POLICY_CHOICES, ReplayRunConfig, run_replay


def positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument.

    Parameters
    ----------
    value : str
        Raw CLI value.

    Returns
    -------
    int
        Parsed positive integer.

    Raises
    ------
    ArgumentTypeError
        If existing validation or execution checks fail.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replay source NAS Pareto-front candidates through a target HIL config. "
            "Required inputs are --source-run-dir and exactly one target option: "
            "--target-run-dir or --target-config."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Dry-run payload preflight:
    python src/pareto_hil_replay.py \\
      --source-run-dir models/OxIOD_FLOPS_PROXY_case1_1_t3 \\
      --target-run-dir models/OxIOD_BLE33_B2B_case1_2_t1 \\
      --dry-run \\
      --output-dir models/replays/flops_case1_1_t3_on_ble33_dry_run

  Hardware replay with resume:
    python src/pareto_hil_replay.py \\
      --source-run-dir models/OxIOD_FLOPS_PROXY_case1_1_t3 \\
      --target-run-dir models/OxIOD_BLE33_B2B_case1_2_t1 \\
      --output-dir models/replays/flops_case1_1_t3_on_ble33_hil \\
      --resume
""",
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument("--source-run-dir", required=True, help="NAS run directory to select Pareto candidates from.")
    target = required.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-run-dir",
        default=None,
        help="Target NAS run directory containing nas_config*.yaml. Use this or --target-config, not both.",
    )
    target.add_argument(
        "--target-config",
        default=None,
        help="Explicit target HIL YAML config path. Use this or --target-run-dir, not both.",
    )
    parser.add_argument("--source-csv", default=None, help="Explicit source NAS log CSV path.")
    parser.add_argument("--source-config", default=None, help="Explicit source NAS config YAML path.")
    parser.add_argument(
        "--objectives",
        default=None,
        help="Optional source objective override: metric_or_column:direction[,metric_or_column:direction].",
    )
    parser.add_argument("--output-dir", default=None, help="Replay output directory. Defaults under models/replays.")
    parser.add_argument(
        "--max-candidates",
        type=positive_int,
        default=None,
        help="Optional positive cap after Pareto selection/dedupe.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write replay payloads without running HIL.")
    parser.add_argument("--resume", action="store_true", help="Skip payloads already present in replay_results.csv.")
    parser.add_argument("--allow-gpu", action="store_true", help="Do not clear CUDA_VISIBLE_DEVICES before HIL import.")
    parser.add_argument(
        "--device-option-policy",
        choices=DEVICE_OPTION_POLICY_CHOICES,
        default="preserve-source",
        help="Replay logged source device options or let the target config choose defaults.",
    )
    parser.add_argument("--model-variant", default=None, help="Optional model variant forwarded to HIL.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional checkpoint path forwarded to HIL.")
    return parser


def optional_path(value: str | None) -> Path | None:
    """Convert an optional CLI path string to ``Path``.

    Parameters
    ----------
    value : str | None
        Raw CLI path value.

    Returns
    -------
    pathlib.Path | None
        Path object when provided, otherwise ``None``.
    """
    return Path(value) if value else None


def namespace_to_replay_config(args: argparse.Namespace) -> ReplayRunConfig:
    """Convert parsed CLI arguments to replay run config.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    ReplayRunConfig
        Typed replay configuration for the library runner.
    """
    return ReplayRunConfig(
        source_run_dir=Path(args.source_run_dir),
        target_run_dir=optional_path(args.target_run_dir),
        target_config=optional_path(args.target_config),
        source_csv=optional_path(args.source_csv),
        source_config=optional_path(args.source_config),
        objectives=args.objectives,
        output_dir=optional_path(args.output_dir),
        max_candidates=args.max_candidates,
        dry_run=args.dry_run,
        resume=args.resume,
        allow_gpu=args.allow_gpu,
        device_option_policy=args.device_option_policy,
        model_variant=args.model_variant,
        checkpoint_path=args.checkpoint_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Pareto HIL replay command.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command-line arguments, excluding the program name.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_replay(namespace_to_replay_config(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    raise SystemExit(main())
