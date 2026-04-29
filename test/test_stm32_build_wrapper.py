"""Regression tests for the STM32 blink build/upload wrapper script."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_module(module_name: str, relative_path: str):
    """Load an analysis script by repository-relative path for wrapper tests.

    Parameters
    ----------
    module_name : str
        Synthetic module name to register in ``sys.modules``.
    relative_path : str
        Repository-relative path to the Python entrypoint under test.

    Returns
    -------
    module
        Imported module object loaded directly from the target file.
    """

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stm32_build_wrapper = _load_module(
    "stm32_build_wrapper",
    "analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py",
)


class Stm32BuildWrapperTests(unittest.TestCase):
    def test_run_gdb_load_raises_when_gdb_exits_early_with_error(self):
        # Early GDB failures should surface their stderr so build-wrapper callers can diagnose load problems.
        fake_proc = unittest.mock.Mock()
        fake_proc.communicate.return_value = ("target remote failed", "")
        fake_proc.returncode = 1

        with (
            patch.object(stm32_build_wrapper, "_get_elf_entry_point", return_value=0x1234),
            patch.object(stm32_build_wrapper.subprocess, "Popen", return_value=fake_proc),
        ):
            with self.assertRaises(stm32_build_wrapper.WorkflowError) as ctx:
                stm32_build_wrapper._run_gdb_load(
                    gdb=Path("/tmp/arm-none-eabi-gdb"),
                    elf_path=Path("/tmp/toy.elf"),
                    gdb_port=61234,
                    run_after_load=True,
                    verbose=False,
                )

        self.assertIn("GDB load/run failed", str(ctx.exception))
        self.assertIn("target remote failed", str(ctx.exception))
        fake_proc.kill.assert_not_called()

    def test_run_gdb_load_timeout_keeps_existing_success_path(self):
        # GDB-load timeouts should leave the existing success path untouched for non-timeout runs.
        fake_proc = unittest.mock.Mock()
        fake_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="gdb", timeout=stm32_build_wrapper.GDB_JUMP_TIMEOUT_S),
            ("", ""),
        ]

        with (
            patch.object(stm32_build_wrapper, "_get_elf_entry_point", return_value=0x1234),
            patch.object(stm32_build_wrapper.subprocess, "Popen", return_value=fake_proc),
        ):
            stm32_build_wrapper._run_gdb_load(
                gdb=Path("/tmp/arm-none-eabi-gdb"),
                elf_path=Path("/tmp/toy.elf"),
                gdb_port=61234,
                run_after_load=True,
                verbose=False,
            )

        fake_proc.kill.assert_called_once()
        self.assertEqual(fake_proc.communicate.call_count, 2)

    def test_run_gdb_load_timeout_raises_when_output_shows_failure(self):
        # Timeout handling should still preserve failure output when GDB reports an error before the process is killed.
        fake_proc = unittest.mock.Mock()
        fake_proc.communicate.side_effect = [
            subprocess.TimeoutExpired(
                cmd="gdb",
                timeout=stm32_build_wrapper.GDB_JUMP_TIMEOUT_S,
                output="remote communication error",
            ),
            ("target disconnected", ""),
        ]

        with (
            patch.object(stm32_build_wrapper, "_get_elf_entry_point", return_value=0x1234),
            patch.object(stm32_build_wrapper.subprocess, "Popen", return_value=fake_proc),
        ):
            with self.assertRaises(stm32_build_wrapper.WorkflowError) as ctx:
                stm32_build_wrapper._run_gdb_load(
                    gdb=Path("/tmp/arm-none-eabi-gdb"),
                    elf_path=Path("/tmp/toy.elf"),
                    gdb_port=61234,
                    run_after_load=True,
                    verbose=False,
                )

        self.assertIn("appears to have failed", str(ctx.exception))
        self.assertIn("remote communication error", str(ctx.exception))
        fake_proc.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
