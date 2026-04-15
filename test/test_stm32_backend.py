import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.errors import (  # noqa: E402
    HIL_ERROR_COMPILE,
    HIL_ERROR_FLASH_OVERFLOW,
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
)
from tinyodom.microcontrollers import get_device, list_device_specs, resolve_device_options  # noqa: E402
from tinyodom.microcontrollers import stm32_cube_clt  # noqa: E402
from tinyodom.microcontrollers.stm32_nucleo_n657x0 import (  # noqa: E402
    BOARD_NAME,
    STM32NucleoN657X0QDevice,
)


class STM32RegistryTests(unittest.TestCase):
    def test_registry_resolves_stm_device(self) -> None:
        """Ensure the STM device registry returns the Phase 1 backend.

        Returns
        -------
        None
        """
        device = get_device(
            "STM32_NUCLEO_N657X0_Q",
            device_options={"project_root": "/tmp/stm"},
        )
        self.assertIsInstance(device, STM32NucleoN657X0QDevice)
        self.assertEqual(device.spec.name, BOARD_NAME)
        self.assertEqual(device.spec.arena_sizes_kb, [-1])

    def test_list_device_specs_includes_stm_entry(self) -> None:
        """Ensure the public spec listing exposes the STM board metadata.

        Returns
        -------
        None
        """
        specs = list_device_specs()
        self.assertIn("STM32_NUCLEO_N657X0_Q", specs)
        self.assertEqual(specs["STM32_NUCLEO_N657X0_Q"]["arena_sizes"], [-1])


class STM32BackendBehaviorTests(unittest.TestCase):
    def test_resolve_device_options_defaults_partial_stm_numeric_block(self) -> None:
        """Ensure partial STM config blocks inherit numeric defaults cleanly.

        Returns
        -------
        None
        """
        resolved = resolve_device_options(
            "STM32_NUCLEO_N657X0_Q",
            type(
                "DeviceConfigDouble",
                (),
                {"stm32": type("STMConfigDouble", (), {"project_root": "/tmp/stm"})()},
            )(),
        )
        self.assertEqual(resolved["project_root"], Path("/tmp/stm").resolve())
        self.assertEqual(resolved["gdb_port"], stm32_cube_clt.DEFAULT_GDB_PORT)
        self.assertEqual(resolved["apid"], stm32_cube_clt.DEFAULT_APID)
        self.assertEqual(
            resolved["server_ready_timeout_s"],
            stm32_cube_clt.SERVER_READY_TIMEOUT_S,
        )

    def test_compile_success_parses_ram_and_flash(self) -> None:
        """Verify successful STM builds report parsed RAM and flash usage.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice(device_options={"project_root": "/tmp/stm"})
        with patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project"
        ) as build_mock, patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output"
        ) as size_mock:
            build_mock.return_value = stm32_cube_clt.BuildResult(
                log="build ok",
                debug_dir=Path("/tmp/stm/Debug"),
                elf_path=Path("/tmp/stm/Debug/app.elf"),
            )
            size_mock.return_value = stm32_cube_clt.SizeResult(
                elf_flash_bytes=1234,
                ram_bytes=5678,
                raw_output="size output",
            )
            result = device.compile(
                sketch_path=Path("/tmp/ignored"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.flash_bytes, 1234)
        self.assertEqual(result.ram_bytes, 5678)
        self.assertIsNone(result.overflow_kind)

    def test_compile_failure_classifies_flash_overflow(self) -> None:
        """Verify linker overflow text is normalized to flash overflow.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice(device_options={"project_root": "/tmp/stm"})
        with patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
            side_effect=stm32_cube_clt.WorkflowError("region `FLASH' overflowed by 100 bytes"),
        ):
            result = device.compile(
                sketch_path=Path("/tmp/ignored"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.overflow_kind, "flash")

    def test_evaluate_compile_only_reports_sentinel_arena_bytes(self) -> None:
        """Ensure Phase 1 STM evaluate normalizes arena bytes to ``-1``.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice(device_options={"project_root": "/tmp/stm"})
        with patch.object(
            device,
            "compile",
            return_value=type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": Path("/tmp/stm/Debug"),
                },
            )(),
        ):
            metrics = device.evaluate(
                dirpath=Path("/tmp/stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                run_hil=False,
            )

        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertEqual(metrics.arena_bytes, -1)
        self.assertEqual(metrics.flash_bytes, 2222)
        self.assertEqual(metrics.ram_bytes, 1111)

    def test_evaluate_never_uses_under_sized_or_latency_codes(self) -> None:
        """Ensure STM compile failures do not trigger Arduino retry semantics.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice(device_options={"project_root": "/tmp/stm"})
        compile_result = type(
            "CompileResultDouble",
            (),
            {
                "success": False,
                "log": "compile failed",
                "flash_bytes": None,
                "ram_bytes": None,
                "overflow_kind": None,
                "build_dir": Path("/tmp/stm/Debug"),
            },
        )()
        with patch.object(device, "compile", return_value=compile_result):
            metrics = device.evaluate(
                dirpath=Path("/tmp/stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                run_hil=False,
            )
        self.assertEqual(metrics.error_code, HIL_ERROR_COMPILE)

    def test_evaluate_maps_overflow_kinds(self) -> None:
        """Verify overflow classifications map to the shared HIL error codes.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice(device_options={"project_root": "/tmp/stm"})
        for overflow_kind, expected_error in (
            ("flash", HIL_ERROR_FLASH_OVERFLOW),
            ("ram", HIL_ERROR_RAM_OVERFLOW),
        ):
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": False,
                    "log": "overflow",
                    "flash_bytes": 100,
                    "ram_bytes": 200,
                    "overflow_kind": overflow_kind,
                    "build_dir": Path("/tmp/stm/Debug"),
                },
            )()
            with patch.object(device, "compile", return_value=compile_result):
                metrics = device.evaluate(
                    dirpath=Path("/tmp/stm"),
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    run_hil=False,
                )
            self.assertEqual(metrics.error_code, expected_error)


class STM32HelperTests(unittest.TestCase):
    def test_resolve_elf_path_accepts_non_blink_name(self) -> None:
        """Ensure ELF discovery is not hard-coded to the blink example name.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir)
            elf_path = debug_dir / "tinyodom_phase1.elf"
            elf_path.write_bytes(b"elf")
            resolved = stm32_cube_clt.resolve_elf_path(debug_dir)
        self.assertEqual(resolved, elf_path)

    def test_parse_size_output_raises_workflow_error_when_size_tool_is_missing(self) -> None:
        """Ensure missing ``arm-none-eabi-size`` is normalized to ``WorkflowError``.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_path = Path(tmpdir) / "tinyodom_phase1.elf"
            elf_path.write_bytes(b"elf")
            with patch(
                "tinyodom.microcontrollers.stm32_cube_clt.resolve_required_tool_path",
                side_effect=stm32_cube_clt.WorkflowError("arm-none-eabi-size was not provided."),
            ):
                with self.assertRaises(stm32_cube_clt.WorkflowError):
                    stm32_cube_clt.parse_size_output(elf_path)

    def test_classify_build_failure_treats_rom_overflow_as_flash(self) -> None:
        """Ensure STM ROM-region overflows map to flash overflow semantics.

        Returns
        -------
        None
        """
        classification = stm32_cube_clt.classify_build_failure(
            "ld: region `ROM' overflowed by 2048 bytes"
        )
        self.assertEqual(classification, "flash")

    def test_run_command_wraps_host_os_errors_in_workflow_error(self) -> None:
        """Ensure host-side command launch failures are normalized.

        Returns
        -------
        None
        """
        with patch(
            "tinyodom.microcontrollers.stm32_cube_clt.subprocess.run",
            side_effect=FileNotFoundError("make not found"),
        ):
            with self.assertRaises(stm32_cube_clt.WorkflowError):
                stm32_cube_clt._run_command(["make", "-C", "Debug", "all"])

    def test_parse_size_output_wraps_host_os_errors_in_workflow_error(self) -> None:
        """Ensure size-tool launch failures are normalized.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_path = Path(tmpdir) / "tinyodom_phase1.elf"
            elf_path.write_bytes(b"elf")
            with patch(
                "tinyodom.microcontrollers.stm32_cube_clt.resolve_required_tool_path",
                return_value=Path("/usr/bin/arm-none-eabi-size"),
            ), patch(
                "tinyodom.microcontrollers.stm32_cube_clt.subprocess.run",
                side_effect=OSError("exec format error"),
            ):
                with self.assertRaises(stm32_cube_clt.WorkflowError):
                    stm32_cube_clt.parse_size_output(elf_path)
