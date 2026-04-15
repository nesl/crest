#!/usr/bin/env python3
"""
Run the host-only STM32N6 Phase 0 ST Edge AI probe.

This script keeps all generated artifacts outside the repo by default and
captures the exact command outputs needed to reproduce the current Phase 0
findings:

1. `stedgeai analyze` on the representative TinyODOM `.tflite`
2. `stedgeai generate` using the CM55-style binary/external-weights flow
3. `stedgeai generate` using the legacy non-binary flow that keeps weights in C

The binary flow is useful to prove the reference CM55 generation path. The
non-binary flow is useful because it is more compatible with a future
debug-load-only board smoke test.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "TinyOdomEx_OxIOD_PORTENTA_H7.tflite"
DEFAULT_OUTPUT_ROOT = Path("/tmp/tinyodom_stm32_phase0_probe")


def _default_stedgeai_root() -> Path | None:
    """Resolve the ST Edge AI install root from env or the standard Linux path."""

    env_value = os.environ.get("STEDGEAI_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = sorted(Path("/opt/ST/STEdgeAI").glob("*"))
    if not candidates:
        return None
    return candidates[-1].resolve()


def _default_memory_pool() -> Path:
    """Return the default STM32N6 memory-pool path derived from ST Edge AI."""

    stedgeai_root = _default_stedgeai_root()
    if stedgeai_root is None:
        return Path("/opt/ST/STEdgeAI/mypool_N6.json")
    return (
        stedgeai_root
        / "Projects"
        / "STM32N6570-DK"
        / "Applications"
        / "CM55_Validation"
        / "mypool_N6.json"
    )


DEFAULT_MEMORY_POOL = _default_memory_pool()


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the Phase 0 probe.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the model, memory-pool, and output-root
        options used by the host-only STM32N6 probe.
    """

    parser = argparse.ArgumentParser(
        description="Run the host-only ST Edge AI Phase 0 probe for STM32N6."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the representative TinyODOM .tflite model.",
    )
    parser.add_argument(
        "--memory-pool",
        type=Path,
        default=DEFAULT_MEMORY_POOL,
        help="CM55-style memory pool descriptor JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for probe logs and generated artifacts.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output root before running the probe.",
    )
    return parser


def _run_logged(argv: list[str], log_path: Path) -> None:
    """Run a subprocess and persist its stdout/stderr to a log file.

    Parameters
    ----------
    argv : list[str]
        Command and arguments to execute.
    log_path : pathlib.Path
        Destination path for the rendered command and captured output.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        Raised when the subprocess exits with a nonzero status code.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    rendered = " ".join(str(part) for part in argv)
    log_body = [
        f"$ {rendered}",
        "",
        proc.stdout.rstrip(),
    ]
    if proc.stderr.strip():
        log_body.extend(["", "[stderr]", proc.stderr.rstrip()])
    log_path.write_text("\n".join(log_body).rstrip() + "\n", encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {rendered}\n"
            f"See log: {log_path}"
        )


def _run_size(elf_path: Path, log_path: Path) -> None:
    """Capture the section-size summary for a generated ELF when available.

    Parameters
    ----------
    elf_path : pathlib.Path
        Path to the generated ELF emitted by the ST Edge AI workspace build.
    log_path : pathlib.Path
        Destination log path for the `arm-none-eabi-size` output.

    Returns
    -------
    None
    """

    if not elf_path.exists():
        return
    _run_logged(["arm-none-eabi-size", str(elf_path)], log_path)


def _print_summary(output_root: Path) -> None:
    """Print a compact summary of the probe artifact locations.

    Parameters
    ----------
    output_root : pathlib.Path
        Root directory containing the analyze and generate outputs.

    Returns
    -------
    None
    """

    print("Phase 0 host probe complete.")
    print(f"Artifacts: {output_root}")
    print(f"Analyze log: {output_root / 'analyze' / 'command.log'}")
    print(f"Binary generate log: {output_root / 'generate_binary' / 'command.log'}")
    print(f"Non-binary generate log: {output_root / 'generate_nobin' / 'command.log'}")


def main() -> int:
    """Run the full host-only STM32N6 Phase 0 probe sequence.

    Returns
    -------
    int
        Process exit code. Returns `0` on success.
    """

    parser = _build_parser()
    args = parser.parse_args()

    model_path = args.model.resolve()
    memory_pool = args.memory_pool.resolve()
    output_root = args.output_root.resolve()

    if not model_path.exists():
        parser.error(f"Model not found: {model_path}")
    if not memory_pool.exists():
        parser.error(f"Memory pool JSON not found: {memory_pool}")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    analyze_dir = output_root / "analyze"
    generate_binary_dir = output_root / "generate_binary"
    generate_nobin_dir = output_root / "generate_nobin"

    # Analyze proves model acceptance and captures the baseline memory report
    # without generating firmware-side assets.
    _run_logged(
        [
            "stedgeai",
            "analyze",
            "-m",
            str(model_path),
            "-t",
            "tflite",
            "--target",
            "stm32n6",
            "--quiet",
            "--workspace",
            str(analyze_dir / "ws"),
            "--output",
            str(analyze_dir / "out"),
        ],
        analyze_dir / "command.log",
    )

    # The binary flow mirrors the CM55 reference more closely, but it requires
    # the legacy C API in this ST Edge AI version when `--address` is used.
    _run_logged(
        [
            "stedgeai",
            "generate",
            "-m",
            str(model_path),
            "-t",
            "tflite",
            "--target",
            "stm32n6",
            "--c-api",
            "legacy",
            "--binary",
            "--address",
            "0x71000000",
            "--memory-pool",
            str(memory_pool),
            "--quiet",
            "--workspace",
            str(generate_binary_dir / "ws"),
            "--output",
            str(generate_binary_dir / "out"),
        ],
        generate_binary_dir / "command.log",
    )
    _run_size(
        generate_binary_dir / "ws" / "build_rt_network" / "network.elf",
        generate_binary_dir / "elf_size.log",
    )

    # The non-binary flow is important because it keeps weights in generated C
    # rather than forcing an external-weight placement scheme.
    _run_logged(
        [
            "stedgeai",
            "generate",
            "-m",
            str(model_path),
            "-t",
            "tflite",
            "--target",
            "stm32n6",
            "--c-api",
            "legacy",
            "--quiet",
            "--workspace",
            str(generate_nobin_dir / "ws"),
            "--output",
            str(generate_nobin_dir / "out"),
        ],
        generate_nobin_dir / "command.log",
    )
    _run_size(
        generate_nobin_dir / "ws" / "build_rt_network" / "network.elf",
        generate_nobin_dir / "elf_size.log",
    )

    _print_summary(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
