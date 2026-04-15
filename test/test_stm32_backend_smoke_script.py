from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


stm32_backend_smoke = _load_module(
    "stm32_backend_smoke",
    "analysis_scripts/stm32_example_project/smoke_test_stm32_backend_phase1.py",
)


class Stm32BackendSmokeScriptTests(unittest.TestCase):
    def test_default_blink_path_compiles_then_uploads(self) -> None:
        fake_device = Mock()
        fake_device.compile.return_value = SimpleNamespace(
            success=True,
            log="compile ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("/tmp/blink/Debug"),
        )
        fake_device.upload.return_value = SimpleNamespace(success=True, log="upload ok")

        with (
            patch.object(sys, "argv", ["smoke_test_stm32_backend_phase1.py"]),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device) as get_device_mock,
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_SUCCESS)
        fake_device.compile.assert_called_once()
        fake_device.upload.assert_called_once()
        device_options = get_device_mock.call_args.kwargs["device_options"]
        self.assertEqual(
            device_options["project_root"],
            stm32_backend_smoke.DEFAULT_PROJECT_ROOT.resolve(),
        )

    def test_compile_only_skips_upload(self) -> None:
        fake_device = Mock()
        fake_device.compile.return_value = SimpleNamespace(
            success=True,
            log="compile ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("/tmp/blink/Debug"),
        )

        with (
            patch.object(sys, "argv", ["smoke_test_stm32_backend_phase1.py", "--compile-only"]),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device),
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_SUCCESS)
        fake_device.compile.assert_called_once()
        fake_device.upload.assert_not_called()

    def test_compile_failure_returns_exit_code_1(self) -> None:
        fake_device = Mock()
        fake_device.compile.return_value = SimpleNamespace(
            success=False,
            log="compile failed",
            flash_bytes=None,
            ram_bytes=None,
            overflow_kind=None,
            build_dir=Path("/tmp/blink/Debug"),
        )

        with (
            patch.object(sys, "argv", ["smoke_test_stm32_backend_phase1.py"]),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device),
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_COMPILE_FAILURE)
        fake_device.upload.assert_not_called()

    def test_upload_failure_returns_exit_code_2(self) -> None:
        fake_device = Mock()
        fake_device.compile.return_value = SimpleNamespace(
            success=True,
            log="compile ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("/tmp/blink/Debug"),
        )
        fake_device.upload.return_value = SimpleNamespace(success=False, log="upload failed")

        with (
            patch.object(sys, "argv", ["smoke_test_stm32_backend_phase1.py"]),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device),
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_UPLOAD_FAILURE)

    def test_serial_check_passes_when_monitor_matches_all_tokens(self) -> None:
        fake_device = Mock()
        fake_device.compile.return_value = SimpleNamespace(
            success=True,
            log="compile ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("/tmp/blink/Debug"),
        )
        fake_device.upload.return_value = SimpleNamespace(success=True, log="upload ok")
        fake_monitor = Mock()
        fake_monitor.wait_for_tokens.return_value = (True, [])

        with (
            patch.object(
                sys,
                "argv",
                [
                    "smoke_test_stm32_backend_phase1.py",
                    "--expect-token",
                    "BOOT_OK",
                    "--expect-token",
                    "APP_READY",
                ],
            ),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device),
            patch.object(stm32_backend_smoke, "_SerialTokenMonitor", return_value=fake_monitor),
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_SUCCESS)
        fake_monitor.wait_for_tokens.assert_called_once_with(
            ["BOOT_OK", "APP_READY"],
            timeout_s=stm32_backend_smoke.DEFAULT_SERIAL_TIMEOUT_S,
        )
        fake_monitor.close.assert_called_once()

    def test_serial_check_failure_returns_exit_code_3(self) -> None:
        fake_device = Mock()
        fake_device.compile.return_value = SimpleNamespace(
            success=True,
            log="compile ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("/tmp/blink/Debug"),
        )
        fake_device.upload.return_value = SimpleNamespace(success=True, log="upload ok")
        fake_monitor = Mock()
        fake_monitor.wait_for_tokens.return_value = (False, ["APP_READY"])

        with (
            patch.object(
                sys,
                "argv",
                ["smoke_test_stm32_backend_phase1.py", "--expect-token", "APP_READY"],
            ),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device),
            patch.object(stm32_backend_smoke, "_SerialTokenMonitor", return_value=fake_monitor),
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_SERIAL_FAILURE)
        fake_monitor.close.assert_called_once()

    def test_explicit_tool_overrides_are_forwarded_to_device_options(self) -> None:
        fake_device = Mock()
        fake_device.compile.return_value = SimpleNamespace(
            success=True,
            log="compile ok",
            flash_bytes=123,
            ram_bytes=456,
            overflow_kind=None,
            build_dir=Path("/tmp/blink/Debug"),
        )
        fake_device.upload.return_value = SimpleNamespace(success=True, log="upload ok")

        with (
            patch.object(
                sys,
                "argv",
                [
                    "smoke_test_stm32_backend_phase1.py",
                    "--gdbserver",
                    "/tmp/tools/ST-LINK_gdbserver",
                    "--gdb",
                    "/tmp/tools/arm-none-eabi-gdb",
                    "--cubeprog-bin",
                    "/tmp/tools/cubeprog/bin",
                    "--gdb-port",
                    "61235",
                    "--apid",
                    "2",
                    "--server-ready-timeout",
                    "20.0",
                ],
            ),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device) as get_device_mock,
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_SUCCESS)
        device_options = get_device_mock.call_args.kwargs["device_options"]
        self.assertEqual(device_options["gdbserver"], Path("/tmp/tools/ST-LINK_gdbserver"))
        self.assertEqual(device_options["gdb"], Path("/tmp/tools/arm-none-eabi-gdb"))
        self.assertEqual(device_options["cubeprog_bin"], Path("/tmp/tools/cubeprog/bin"))
        self.assertEqual(device_options["gdb_port"], 61235)
        self.assertEqual(device_options["apid"], 2)
        self.assertEqual(device_options["server_ready_timeout_s"], 20.0)

    def test_clean_triggers_helper_build_path(self) -> None:
        fake_device = Mock()
        fake_device.upload.return_value = SimpleNamespace(success=True, log="upload ok")
        fake_build_result = SimpleNamespace(
            log="clean build ok",
            debug_dir=Path("/tmp/blink/Debug"),
            elf_path=Path("/tmp/blink/Debug/app.elf"),
        )
        fake_size_result = SimpleNamespace(
            elf_flash_bytes=123,
            ram_bytes=456,
            raw_output="size output",
        )

        with (
            patch.object(
                sys,
                "argv",
                ["smoke_test_stm32_backend_phase1.py", "--clean", "--jobs", "7"],
            ),
            patch.object(stm32_backend_smoke, "get_device", return_value=fake_device),
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "validate_project_root",
                return_value=Path("/tmp/blink"),
            ),
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "build_project",
                return_value=fake_build_result,
            ) as build_project_mock,
            patch.object(
                stm32_backend_smoke.stm32_cube_clt,
                "parse_size_output",
                return_value=fake_size_result,
            ),
        ):
            exit_code = stm32_backend_smoke.main()

        self.assertEqual(exit_code, stm32_backend_smoke.EXIT_SUCCESS)
        build_project_mock.assert_called_once_with(
            project_root=Path("/tmp/blink"),
            jobs=7,
            clean=True,
        )
        fake_device.compile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
