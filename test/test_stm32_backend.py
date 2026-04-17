import json
import os
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
    HIL_ERROR_LATENCY,
    HIL_ERROR_OK,
    HIL_ERROR_RAM_OVERFLOW,
    HIL_ERROR_UPLOAD,
)
from tinyodom.microcontrollers import get_device, list_device_specs, resolve_device_options  # noqa: E402
from tinyodom.microcontrollers import stm32_cube_clt  # noqa: E402
from tinyodom.microcontrollers import stm32_nucleo_n657x0 as stm32_n657_backend  # noqa: E402
from tinyodom.microcontrollers import stm32_runtime  # noqa: E402
from tinyodom.microcontrollers.stm32_nucleo_n657x0 import (  # noqa: E402
    BOARD_NAME,
    DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME,
    DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
    DEFAULT_TEMPLATE_ROOT,
    DEFAULT_WEIGHTS_FLASH_ADDRESS,
    DEFAULT_WEIGHTS_MEMORY_POOL,
    EXPECTED_GENERATED_OUTPUTS,
    STAGED_MANIFEST_NAME,
    STM32NucleoN657X0QDevice,
)


def _write_text(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories first.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path.
    text : str
        File contents to write.

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_project_tree(root: Path) -> Path:
    """Build a minimal STM32 FSBL tree for unit tests.

    Parameters
    ----------
    root : pathlib.Path
        Root directory to populate.

    Returns
    -------
    pathlib.Path
        The populated project root.
    """
    _write_text(root / "Debug" / "makefile", "# makefile\n")
    _write_text(root / "Src" / "main.c", "int main(void) { return 0; }\n")
    _write_text(root / "Inc" / "stm32n6xx_hal_conf.h", "#pragma once\n")
    _write_text(root / "Inc" / "stm32n6xx_nucleo_conf.h", "#pragma once\n")
    _write_text(root / "Inc" / "tcn_dut_runner.h", "#pragma once\n")
    _write_text(
        root / "Inc" / "network_data_params.h",
        "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (8192)\n",
    )
    _write_text(
        root / "STM32N657X0HXQ_AXISRAM2_fsbl.ld",
        "\n".join(
            [
                "_Min_Heap_Size = 0x2000;",
                "_Min_Stack_Size = 0x4000;",
                "",
            ]
        ),
    )
    return root


class _FakeSerialMonitor:
    """Context-manager double for STM direct-serial tests."""

    def __init__(self, port: str, baud: int, label: str) -> None:
        """Initialize the fake serial monitor.

        Parameters
        ----------
        port : str
            Serial device path associated with the monitor.
        baud : int
            Baud rate requested by the caller.
        label : str
            Human-readable channel label used in runtime logging.
        """
        self.port = port
        self.baud = baud
        self.label = label

    def __enter__(self):
        """Open the fake monitor context.

        Returns
        -------
        _FakeSerialMonitor
            The active fake serial monitor instance.
        """
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the fake monitor context.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type raised inside the context, if any.
        exc : BaseException | None
            Exception instance raised inside the context, if any.
        traceback : types.TracebackType | None
            Traceback associated with the exception, if one occurred.

        Returns
        -------
        None
            Always returns ``None`` so exceptions propagate normally.
        """
        del exc_type, exc, traceback

    def write_line(self, text: str) -> None:
        """Accept a serial log line emitted by the DUT.

        Parameters
        ----------
        text : str
            Serial log line that would have been written to the monitor.

        Returns
        -------
        None
            This test double stores no output and returns nothing.
        """
        del text


class _FakeHarnessSerial:
    """Context-manager double for open harness serial sessions."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the fake harness serial session.

        Parameters
        ----------
        *args
            Positional arguments passed to the real harness serial class.
        **kwargs
            Keyword arguments passed to the real harness serial class.
        """
        del args, kwargs

    def __enter__(self):
        """Open the fake harness serial session.

        Returns
        -------
        _FakeHarnessSerial
            The active fake harness serial session.
        """
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Close the fake harness serial session.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            Exception type raised inside the context, if any.
        exc : BaseException | None
            Exception instance raised inside the context, if any.
        traceback : types.TracebackType | None
            Traceback associated with the exception, if one occurred.

        Returns
        -------
        None
            Always returns ``None`` so exceptions propagate normally.
        """
        del exc_type, exc, traceback


class STM32RegistryTests(unittest.TestCase):
    """Validate STM32 registry wiring and default metadata."""

    def test_registry_resolves_stm_device_without_explicit_project_root(self) -> None:
        """Ensure registry construction uses the canonical STM template by default.

        Returns
        -------
        None
        """
        device = get_device("STM32_NUCLEO_N657X0_Q")
        self.assertIsInstance(device, STM32NucleoN657X0QDevice)
        self.assertEqual(device.spec.name, BOARD_NAME)
        self.assertEqual(device.spec.arena_sizes_kb, [-1])
        self.assertEqual(device.resolved_options.project_root, DEFAULT_TEMPLATE_ROOT.resolve())

    def test_list_device_specs_includes_stm_entry(self) -> None:
        """Ensure the public device spec listing includes the STM board.

        Returns
        -------
        None
        """
        specs = list_device_specs()
        self.assertIn("STM32_NUCLEO_N657X0_Q", specs)
        self.assertEqual(specs["STM32_NUCLEO_N657X0_Q"]["arena_sizes"], [-1])
        self.assertEqual(
            specs["STM32_NUCLEO_N657X0_Q"]["max_external_flash"],
            DEFAULT_MAX_EXTERNAL_FLASH_BYTES,
        )


class STM32BackendBehaviorTests(unittest.TestCase):
    """Validate STM Phase 2 option resolution, staging, and compile behavior."""

    def test_resolve_device_options_defaults_partial_stm_numeric_block(self) -> None:
        """Ensure partial STM config blocks inherit defaults from the backend.

        Returns
        -------
        None
        """
        resolved = resolve_device_options(
            "STM32_NUCLEO_N657X0_Q",
            type(
                "DeviceConfigDouble",
                (),
                {"stm32": type("STMConfigDouble", (), {"cpu_clock_mhz": 400})()},
            )(),
        )
        self.assertEqual(resolved["project_root"], DEFAULT_TEMPLATE_ROOT.resolve())
        self.assertEqual(resolved["gdb_port"], stm32_cube_clt.DEFAULT_GDB_PORT)
        self.assertEqual(resolved["apid"], stm32_cube_clt.DEFAULT_APID)
        self.assertEqual(
            resolved["server_ready_timeout_s"],
            stm32_cube_clt.SERVER_READY_TIMEOUT_S,
        )
        self.assertEqual(resolved["cpu_clock_mhz"], 400)
        self.assertEqual(resolved["weight_storage_mode"], "embedded")
        self.assertEqual(resolved["weights_flash_address"], DEFAULT_WEIGHTS_FLASH_ADDRESS)
        self.assertEqual(resolved["weights_memory_pool"], DEFAULT_WEIGHTS_MEMORY_POOL.resolve())
        self.assertIsNone(resolved["weights_external_loader"])
        self.assertEqual(resolved["max_external_flash_bytes"], DEFAULT_MAX_EXTERNAL_FLASH_BYTES)

    def test_resolve_device_options_accepts_template_root_override(self) -> None:
        """Ensure the new ``template_root`` override is accepted.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = _build_project_tree(Path(tmpdir) / "template")
            resolved = resolve_device_options(
                "STM32_NUCLEO_N657X0_Q",
                type(
                    "DeviceConfigDouble",
                    (),
                    {"stm32": type("STMConfigDouble", (), {"template_root": str(template_root)})()},
                )(),
            )
        self.assertEqual(resolved["project_root"], template_root.resolve())

    def test_resolve_device_options_accepts_legacy_project_root_alias(self) -> None:
        """Ensure the legacy ``project_root`` alias still resolves cleanly.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            template_root = _build_project_tree(Path(tmpdir) / "template")
            resolved = resolve_device_options(
                "STM32_NUCLEO_N657X0_Q",
                type(
                    "DeviceConfigDouble",
                    (),
                    {"stm32": type("STMConfigDouble", (), {"project_root": str(template_root)})()},
                )(),
            )
        self.assertEqual(resolved["project_root"], template_root.resolve())

    def test_compile_success_parses_ram_flash_and_arena_from_staged_path(self) -> None:
        """Ensure compile parses RAM, flash, arena, heap, and stack metrics.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_project_tree(Path(tmpdir) / "staged")
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})
            with patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project"
            ) as build_mock, patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output"
            ) as size_mock:
                build_mock.return_value = stm32_cube_clt.BuildResult(
                    log="build ok",
                    debug_dir=staged_root / "Debug",
                    elf_path=staged_root / "Debug" / "app.elf",
                )
                size_mock.return_value = stm32_cube_clt.SizeResult(
                    elf_flash_bytes=1234,
                    ram_bytes=5678,
                    raw_output="size output",
                )
                result = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

        self.assertTrue(result.success)
        build_mock.assert_called_once_with(
            project_root=staged_root.resolve(),
            jobs=os.cpu_count() or 1,
            clean=True,
        )
        self.assertEqual(result.flash_bytes, 1234)
        self.assertEqual(result.ram_bytes, 5678)
        self.assertEqual(result.arena_bytes, 8192)
        self.assertEqual(result.heap_bytes, 0x2000)
        self.assertEqual(result.stack_bytes, 0x4000)
        self.assertEqual(result.build_dir, staged_root / "Debug")

    def test_compile_failure_classifies_flash_overflow(self) -> None:
        """Ensure STM linker overflows are classified as flash failures.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_project_tree(Path(tmpdir) / "staged")
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})
            with patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project",
                side_effect=stm32_cube_clt.WorkflowError("region `FLASH' overflowed by 100 bytes"),
            ):
                result = device.compile(
                    sketch_path=staged_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                )

        self.assertFalse(result.success)
        self.assertEqual(result.overflow_kind, "flash")
        self.assertEqual(result.build_dir, staged_root / "Debug")

    def test_compile_fails_fast_when_stack_is_too_small(self) -> None:
        """Ensure undersized linker stack reservations fail before build.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_root = _build_project_tree(Path(tmpdir) / "staged")
            _write_text(
                staged_root / "STM32N657X0HXQ_AXISRAM2_fsbl.ld",
                "\n".join(
                    [
                        "_Min_Heap_Size = 0x2000;",
                        "_Min_Stack_Size = 0x0400;",
                        "",
                    ]
                ),
            )
            device = STM32NucleoN657X0QDevice(device_options={"project_root": str(staged_root)})
            result = device.compile(
                sketch_path=staged_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertFalse(result.success)
        self.assertIn("stack reservation is too small", result.log)

    def test_upload_requires_staged_build_dir(self) -> None:
        """Ensure upload does not silently fall back to the template Debug dir.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice()
        result = device.upload(
            sketch_path=Path("/tmp/ignored"),
            build_dir=None,
            serial_port="ttyACM0",
        )
        self.assertFalse(result.success)
        self.assertIn("requires a staged build directory", result.log)

    def test_evaluate_compile_only_reports_real_arena_bytes(self) -> None:
        """Ensure compile-only evaluate returns real parsed arena bytes.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice()
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
                    "arena_bytes": 4096,
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
        self.assertEqual(metrics.arena_bytes, 4096)
        self.assertEqual(metrics.flash_bytes, 2222)
        self.assertEqual(metrics.ram_bytes, 1111)

    def test_supports_runtime_measurement_is_enabled(self) -> None:
        """Ensure STM direct-serial runtime measurement is advertised.

        Returns
        -------
        None
            This test asserts that runtime measurement support is enabled.
        """
        device = STM32NucleoN657X0QDevice()
        self.assertTrue(device.supports_runtime_measurement())

    def test_supports_energy_measurement_is_enabled(self) -> None:
        """Ensure STM harness-assisted energy measurement is advertised.

        Returns
        -------
        None
            This test asserts that energy measurement support is enabled.
        """
        device = STM32NucleoN657X0QDevice()
        self.assertTrue(device.supports_energy_measurement())

    def test_evaluate_run_hil_success_uses_runtime_session(self) -> None:
        """Ensure HIL evaluation returns parsed runtime latency on success.

        Returns
        -------
        None
            This test asserts that successful runtime telemetry reaches the caller.
        """
        device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
        compile_result = type(
            "CompileResultDouble",
            (),
            {
                "success": True,
                "log": "ok",
                "flash_bytes": 2222,
                "ram_bytes": 1111,
                "overflow_kind": None,
                "build_dir": Path("/tmp/stm/Debug"),
                "arena_bytes": 4096,
            },
        )()
        telemetry = stm32_runtime.STM32RuntimeTelemetry(
            latency_s=0.0025,
            serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
            power_metrics={"clock_hz": 600000000.0, "sequence": 1.0},
        )

        with patch.object(device, "compile", return_value=compile_result), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
            return_value=Path("/tmp/stm/Debug/app.elf"),
        ), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
            _FakeSerialMonitor,
        ), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
            return_value="upload ok",
        ) as load_mock, patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
            return_value=telemetry,
        ) as runtime_mock:
            metrics = device.evaluate(
                dirpath=Path("/tmp/stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
                serial_timeout_s=7.5,
                dut_ready_timeout_s=3.5,
            )

        load_mock.assert_called_once()
        runtime_mock.assert_called_once()
        self.assertEqual(metrics.error_code, HIL_ERROR_OK)
        self.assertAlmostEqual(metrics.latency_s, 0.0025)
        self.assertEqual(metrics.power_metrics["clock_hz"], 600000000.0)
        self.assertEqual(metrics.power_metrics["sequence"], 1.0)
        self.assertEqual(metrics.power_metrics["weight_storage_mode"], "embedded")
        self.assertEqual(metrics.power_metrics["external_flash_bytes"], -1.0)

    def test_evaluate_run_hil_runtime_failure_maps_to_latency_error(self) -> None:
        """Ensure protocol failures become ``HIL_ERROR_LATENCY`` with backend detail.

        Returns
        -------
        None
            This test asserts that runtime protocol failures map to latency errors.
        """
        device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
        compile_result = type(
            "CompileResultDouble",
            (),
            {
                "success": True,
                "log": "ok",
                "flash_bytes": 2222,
                "ram_bytes": 1111,
                "overflow_kind": None,
                "build_dir": Path("/tmp/stm/Debug"),
                "arena_bytes": 4096,
            },
        )()
        protocol_error = stm32_runtime.STM32RuntimeProtocolError(
            kind="runtime_timeout",
            detail="Timed out waiting for DUT READY.",
            serial_log=["STM32_AI_INIT=OK"],
        )

        with patch.object(device, "compile", return_value=compile_result), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
            return_value=Path("/tmp/stm/Debug/app.elf"),
        ), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
            _FakeSerialMonitor,
        ), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
            return_value="upload ok",
        ), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
            side_effect=protocol_error,
        ):
            metrics = device.evaluate(
                dirpath=Path("/tmp/stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
            )

        self.assertEqual(metrics.error_code, HIL_ERROR_LATENCY)
        self.assertEqual(metrics.power_metrics["backend_error_kind"], "runtime_timeout")

    def test_evaluate_run_hil_upload_failure_maps_to_upload_error(self) -> None:
        """Ensure debug-load failures become ``HIL_ERROR_UPLOAD``.

        Returns
        -------
        None
            This test asserts that upload failures map to upload errors.
        """
        device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
        compile_result = type(
            "CompileResultDouble",
            (),
            {
                "success": True,
                "log": "ok",
                "flash_bytes": 2222,
                "ram_bytes": 1111,
                "overflow_kind": None,
                "build_dir": Path("/tmp/stm/Debug"),
                "arena_bytes": 4096,
            },
        )()

        with patch.object(device, "compile", return_value=compile_result), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
            return_value=Path("/tmp/stm/Debug/app.elf"),
        ), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
            _FakeSerialMonitor,
        ), patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
            side_effect=stm32_cube_clt.WorkflowError("ST-LINK failed"),
        ):
            metrics = device.evaluate(
                dirpath=Path("/tmp/stm"),
                arena_kb=-1,
                window_size=200,
                num_channels=6,
                serial_port="/dev/ttyACM0",
                run_hil=True,
            )

        self.assertEqual(metrics.error_code, HIL_ERROR_UPLOAD)
        self.assertEqual(metrics.power_metrics["backend_error_kind"], "upload")

    def test_evaluate_maps_overflow_kinds(self) -> None:
        """Ensure compile overflow classifications map to shared HIL codes.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice()
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
                    "arena_bytes": 4096,
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

    def test_prepare_candidate_copies_template_and_stages_generated_outputs(self) -> None:
        """Ensure candidate prep stages per-candidate files into the copied FSBL.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_project_tree(tmp_path / "template")
            _write_text(template_root / "Debug" / "Src" / "subdir.mk", "# src recipe\n")
            _write_text(template_root / "Debug" / "Startup" / "subdir.mk", "# startup recipe\n")
            _write_text(template_root / "Debug" / "stedgeai.mk", "# stedgeai\n")
            _write_text(template_root / "Debug" / "objects.mk", "# objects\n")
            _write_text(template_root / "Debug" / "sources.mk", "# sources\n")
            _write_text(template_root / "Debug" / "objects.list", "obj\n")
            _write_text(template_root / "Debug" / "stale.elf", "old elf\n")
            _write_text(template_root / "Debug" / "Src" / "stale.d", "old dep\n")
            _write_text(template_root / "Debug" / "Startup" / "stale.o", "old obj\n")
            outputs_dir = tmp_path / "outputs"
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "template_root": str(template_root),
                    "cpu_clock_mhz": 400,
                }
            )

            def _fake_generate(
                *,
                workspace_dir: Path,
                output_dir: Path,
                model_path: Path,
                **kwargs,
            ) -> None:
                """Create placeholder generated files for staged candidate tests.

                Parameters
                ----------
                workspace_dir : Path
                    ST Edge AI workspace directory used during generation.
                output_dir : Path
                    Destination directory for generated C artifacts.
                model_path : Path
                    TFLite model path supplied to the generator.
                **kwargs
                    Additional generation options ignored by this fake helper.

                Returns
                -------
                None
                    The helper writes placeholder generated outputs to disk.
                """
                del kwargs
                output_dir.mkdir(parents=True, exist_ok=True)
                workspace_dir.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(model_path.read_bytes())
                for name in EXPECTED_GENERATED_OUTPUTS:
                    target = output_dir / name
                    if name.endswith(".h"):
                        target.write_text(
                            "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (16384)\n",
                            encoding="utf-8",
                        )
                    else:
                        target.write_text("/* generated */\n", encoding="utf-8")

            fake_model = object()
            fake_training_data = type("TrainingData", (), {"inputs": [[[0.0] * 6] * 4]})()

            with patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._ensure_staging_tools"
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_analyze"
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_generate",
                side_effect=_fake_generate,
            ):
                def _write_tflite(*, output_name, **kwargs):
                    """Write a placeholder TFLite file for staging tests.

                    Parameters
                    ----------
                    output_name : str | os.PathLike[str]
                        Destination path where the TFLite model should be written.
                    **kwargs
                        Additional conversion arguments ignored by this fake helper.

                    Returns
                    -------
                    None
                        The helper writes a minimal TFLite payload to disk.
                    """
                    Path(output_name).write_bytes(b"tflite")

                with patch("tinyodom.hardware.convert_to_tflite_model", side_effect=_write_tflite):
                    staged_root = device.prepare_candidate(
                        config=type("Config", (), {"training": type("Training", (), {"quantization": True})()})(),
                        hyperparams=None,
                        model=fake_model,
                        outputs_dir=outputs_dir,
                        tflite_model_path=outputs_dir / "ignored.tflite",
                        training_data=fake_training_data,
                        model_variant="approx_trained",
                        checkpoint_path=None,
                    )

            self.assertTrue(staged_root.is_dir())
            self.assertEqual(staged_root.name, "FSBL")
            self.assertTrue((staged_root / "Src" / "network.c").is_file())
            self.assertTrue((staged_root / "Inc" / "network_data_params.h").is_file())
            self.assertFalse((staged_root / "Debug" / "stale.elf").exists())
            self.assertFalse((staged_root / "Debug" / "Src" / "stale.d").exists())
            self.assertFalse((staged_root / "Debug" / "Startup" / "stale.o").exists())
            self.assertTrue((staged_root / "Debug" / "Src" / "subdir.mk").is_file())
            self.assertTrue((staged_root / "Debug" / "Startup" / "subdir.mk").is_file())
            header_text = (staged_root / "Inc" / "tcn_dut_phase_config.h").read_text(encoding="utf-8")
            self.assertIn("TCN_DUT_SELECTED_PHASE TCN_DUT_PHASE_BACK_TO_BACK", header_text)
            self.assertIn("TCN_DUT_CPU_CLOCK_MHZ 400", header_text)
            candidate_root = staged_root.parent
            self.assertTrue((candidate_root / "model" / "tinyodom_candidate.tflite").is_file())
            self.assertTrue((candidate_root / "stedgeai_ws").is_dir())
            self.assertTrue((candidate_root / "stedgeai_out").is_dir())
            manifest_path = staged_root / STAGED_MANIFEST_NAME
            self.assertTrue(manifest_path.is_file())
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertIn('"weight_storage_mode": "embedded"', manifest_text)

    def test_prepare_candidate_runs_analyze_before_generate(self) -> None:
        """Ensure ST Edge AI analyze preflights compatibility before generate.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_project_tree(tmp_path / "template")
            outputs_dir = tmp_path / "outputs"
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "template_root": str(template_root),
                }
            )
            call_order: list[str] = []

            def _fake_analyze(*, workspace_dir: Path, output_dir: Path, model_path: Path) -> None:
                """Record the analyze step without touching disk.

                Parameters
                ----------
                workspace_dir : pathlib.Path
                    Ignored workspace path.
                output_dir : pathlib.Path
                    Ignored output path.
                model_path : pathlib.Path
                    Ignored model path.

                Returns
                -------
                None
                """
                del workspace_dir, output_dir, model_path
                call_order.append("analyze")

            def _fake_generate(
                *,
                workspace_dir: Path,
                output_dir: Path,
                model_path: Path,
                **kwargs,
            ) -> None:
                """Emit minimal generated outputs and record generation order.

                Parameters
                ----------
                workspace_dir : pathlib.Path
                    Ignored workspace path.
                output_dir : pathlib.Path
                    Destination directory populated with fake outputs.
                model_path : pathlib.Path
                    Ignored model path.
                **kwargs
                    Ignored code-generation keyword arguments.

                Returns
                -------
                None
                """
                del workspace_dir, model_path, kwargs
                call_order.append("generate")
                output_dir.mkdir(parents=True, exist_ok=True)
                for name in EXPECTED_GENERATED_OUTPUTS:
                    target = output_dir / name
                    if name.endswith(".h"):
                        target.write_text(
                            "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (8192)\n",
                            encoding="utf-8",
                        )
                    else:
                        target.write_text("/* generated */\n", encoding="utf-8")

            def _write_tflite(*, output_name, **kwargs):
                """Write a placeholder TFLite file for candidate staging.

                Parameters
                ----------
                output_name : str | pathlib.Path
                    Output file written by the test double.
                **kwargs
                    Ignored conversion keyword arguments.

                Returns
                -------
                None
                """
                del kwargs
                Path(output_name).write_bytes(b"tflite")

            with patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._ensure_staging_tools"
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_analyze",
                side_effect=_fake_analyze,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_generate",
                side_effect=_fake_generate,
            ), patch(
                "tinyodom.hardware.convert_to_tflite_model",
                side_effect=_write_tflite,
            ):
                staged_root = device.prepare_candidate(
                    config=type("Config", (), {"training": type("Training", (), {"quantization": True})()})(),
                    hyperparams=None,
                    model=object(),
                    outputs_dir=outputs_dir,
                    tflite_model_path=outputs_dir / "ignored.tflite",
                    training_data=type("TrainingData", (), {"inputs": [[[0.0] * 6] * 4]})(),
                    model_variant="approx_trained",
                    checkpoint_path=None,
                )
                self.assertTrue(staged_root.is_dir())

            self.assertEqual(call_order, ["analyze", "generate"])

    def test_cleanup_prepared_candidate_removes_staged_root(self) -> None:
        """Ensure staged STM32 candidate directories are deleted after use.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = _build_project_tree(tmp_path / "candidate" / "FSBL")
            _write_text(
                project_root / STAGED_MANIFEST_NAME,
                "{}\n",
            )
            device = STM32NucleoN657X0QDevice()

            device.cleanup_prepared_candidate(project_root)

            self.assertFalse(project_root.parent.exists())

    def test_real_template_parsers_match_checked_in_files(self) -> None:
        """Ensure parser helpers match the canonical checked-in STM template.

        Returns
        -------
        None
        """
        device = STM32NucleoN657X0QDevice()
        with patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.build_project"
        ) as build_mock, patch(
            "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.parse_size_output"
        ) as size_mock:
            build_mock.return_value = stm32_cube_clt.BuildResult(
                log="build ok",
                debug_dir=DEFAULT_TEMPLATE_ROOT / "Debug",
                elf_path=DEFAULT_TEMPLATE_ROOT / "Debug" / "tinyodom_tcn_stm32_fsbl.elf",
            )
            size_mock.return_value = stm32_cube_clt.SizeResult(
                elf_flash_bytes=1234,
                ram_bytes=5678,
                raw_output="size output",
            )
            result = device.compile(
                sketch_path=DEFAULT_TEMPLATE_ROOT,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertTrue(result.success, msg=result.log)
        build_mock.assert_called_once_with(
            project_root=DEFAULT_TEMPLATE_ROOT.resolve(),
            jobs=os.cpu_count() or 1,
            clean=True,
        )
        self.assertEqual(result.heap_bytes, 0x2000)
        self.assertEqual(result.stack_bytes, 0x4000)
        self.assertGreater(result.arena_bytes or -1, 0)


class STM32HelperTests(unittest.TestCase):
    """Validate lower-level STM32 helper behavior."""

    def test_resolve_weights_external_loader_autodiscovers_from_cubeprog_bin(self) -> None:
        """Ensure the backend finds the shipped Nucleo loader without a user path.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cubeprog_bin = tmp_path / "bin"
            loader_path = cubeprog_bin / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME
            _write_text(loader_path, "loader")

            resolved = stm32_n657_backend._resolve_weights_external_loader(cubeprog_bin, None)

        self.assertEqual(resolved, loader_path.resolve())

    def test_resolve_elf_path_accepts_non_blink_name(self) -> None:
        """Ensure ELF discovery remains name-agnostic.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir)
            elf_path = debug_dir / "tinyodom_phase2.elf"
            elf_path.write_bytes(b"elf")
            resolved = stm32_cube_clt.resolve_elf_path(debug_dir)
        self.assertEqual(resolved, elf_path)

    def test_parse_size_output_raises_workflow_error_when_size_tool_is_missing(self) -> None:
        """Ensure missing ``arm-none-eabi-size`` becomes ``WorkflowError``.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_path = Path(tmpdir) / "tinyodom_phase2.elf"
            elf_path.write_bytes(b"elf")
            with patch(
                "tinyodom.microcontrollers.stm32_cube_clt.resolve_required_tool_path",
                side_effect=stm32_cube_clt.WorkflowError("arm-none-eabi-size was not provided."),
            ):
                with self.assertRaises(stm32_cube_clt.WorkflowError):
                    stm32_cube_clt.parse_size_output(elf_path)

    def test_classify_build_failure_treats_rom_overflow_as_flash(self) -> None:
        """Ensure ROM linker overflows map to flash overflow semantics.

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

    def test_prepare_candidate_only_requires_staging_tools(self) -> None:
        """Ensure Phase 2 candidate prep does not demand upload/debug tools.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            template_root = _build_project_tree(tmp_path / "template")
            outputs_dir = tmp_path / "outputs"
            device = STM32NucleoN657X0QDevice(device_options={"template_root": str(template_root)})

            def _write_tflite(*, output_name, **kwargs):
                """Write a placeholder TFLite file for staging tests.

                Parameters
                ----------
                output_name : str | os.PathLike[str]
                    Destination path where the TFLite model should be written.
                **kwargs
                    Additional conversion arguments ignored by this fake helper.

                Returns
                -------
                None
                    The helper writes a minimal TFLite payload to disk.
                """
                Path(output_name).write_bytes(b"tflite")

            def _fake_generate(
                *,
                workspace_dir: Path,
                output_dir: Path,
                model_path: Path,
                **kwargs,
            ) -> None:
                """Create placeholder generated files for staged candidate tests.

                Parameters
                ----------
                workspace_dir : Path
                    ST Edge AI workspace directory used during generation.
                output_dir : Path
                    Destination directory for generated C artifacts.
                model_path : Path
                    TFLite model path supplied to the generator.
                **kwargs
                    Additional generation options ignored by this fake helper.

                Returns
                -------
                None
                    The helper writes placeholder generated outputs to disk.
                """
                del model_path, kwargs
                workspace_dir.mkdir(parents=True, exist_ok=True)
                output_dir.mkdir(parents=True, exist_ok=True)
                for name in EXPECTED_GENERATED_OUTPUTS:
                    target = output_dir / name
                    if name.endswith(".h"):
                        target.write_text(
                            "#define AI_NETWORK_DATA_ACTIVATIONS_SIZE (2048)\n",
                            encoding="utf-8",
                        )
                    else:
                        target.write_text("/* generated */\n", encoding="utf-8")

            with patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._ensure_staging_tools"
            ) as staging_tools_mock, patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_analyze"
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0._run_stedgeai_generate",
                side_effect=_fake_generate,
            ), patch(
                "tinyodom.hardware.convert_to_tflite_model",
                side_effect=_write_tflite,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_required_tool_path"
            ) as tool_mock:
                staged_root = device.prepare_candidate(
                    config=type("Config", (), {"training": type("Training", (), {"quantization": True})()})(),
                    hyperparams=None,
                    model=object(),
                    outputs_dir=outputs_dir,
                    tflite_model_path=outputs_dir / "ignored.tflite",
                    training_data=type("TrainingData", (), {"inputs": [[[0.0] * 6] * 4]})(),
                    model_variant="approx_trained",
                    checkpoint_path=None,
                )

                self.assertTrue(staged_root.is_dir())
                staging_tools_mock.assert_called_once_with()
                tool_mock.assert_not_called()

    def test_evaluate_combined_external_flash_and_harness_uses_canonical_order(self) -> None:
        """Ensure the combined STM path preserves the documented operation order.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Inc" / "tcn_dut_phase_config.h",
                "\n".join(
                    [
                        "#ifndef TCN_DUT_PHASE_CONFIG_H",
                        "#define TCN_DUT_PHASE_CONFIG_H",
                        "#define TCN_DUT_MEASURED_RUNS 5",
                        "#endif",
                        "",
                    ]
                ),
            )
            device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "Debug",
                    "arena_bytes": 4096,
                    "external_flash_bytes": 4096,
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 600000000.0,
                    "sequence": 1.0,
                    "runs": 5,
                    "phase": "back_to_back",
                    "timer_output_s": 0.015,
                    "timer_per_inference_s": 0.003,
                    "timer_per_window_s": 0.003,
                },
            )
            trace: list[str] = []

            class _TracingMonitor(_FakeSerialMonitor):
                def __init__(self, port: str, baud: int, label: str) -> None:
                    """Record DUT monitor creation for ordering assertions.

                    Parameters
                    ----------
                    port : str
                        DUT serial port.
                    baud : int
                        UART baud rate.
                    label : str
                        Human-readable monitor label.

                    Returns
                    -------
                    None
                    """
                    trace.append("open_dut")
                    super().__init__(port, baud, label)

            def _open_harness(*args, **kwargs):
                """Record harness-open ordering and return a fake serial session.

                Parameters
                ----------
                *args
                    Ignored serial constructor positional arguments.
                **kwargs
                    Ignored serial constructor keyword arguments.

                Returns
                -------
                _FakeHarnessSerial
                    Fake harness serial session.
                """
                del args, kwargs
                trace.append("open_harness")
                return _FakeHarnessSerial()

            def _prime_harness(**kwargs):
                """Return a ready harness session and record ordering.

                Parameters
                ----------
                **kwargs
                    Ignored harness priming keyword arguments.

                Returns
                -------
                object
                    Prime-result stand-in with ``harness_ready=True``.
                """
                del kwargs
                trace.append("prime_harness")
                return type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )()

            def _wait_done(**kwargs):
                """Return a completed harness result and record ordering.

                Parameters
                ----------
                **kwargs
                    Ignored harness wait keyword arguments.

                Returns
                -------
                object
                    Done-result stand-in with harness telemetry.
                """
                del kwargs
                trace.append("wait_done")
                return type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": True,
                        "runs_harness": 5,
                        "harness_log": [
                            "HARNESS READY",
                            "runs: 5",
                            "energy output: 1.25",
                            "avg power output: 2.5",
                            "avg current output: 0.5",
                            "bus voltage output: 5.0",
                            "idle power baseline: 0.1",
                            "harness timer output: 0.003",
                            "DONE",
                        ],
                        "error": None,
                    },
                )()

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "external_flash", "external_flash_bytes": 4096.0},
            ), patch.object(
                device,
                "_program_external_weights_if_needed",
                side_effect=lambda *_args, **_kwargs: trace.append("program_external") or {
                    "weight_storage_mode": "external_flash",
                    "external_flash_bytes": 4096.0,
                },
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
                return_value=project_root / "Debug" / "app.elf",
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ) as harness_mock, patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _TracingMonitor,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                side_effect=_open_harness,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                side_effect=_prime_harness,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                side_effect=_wait_done,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                side_effect=lambda **kwargs: trace.append("debug_load") or "upload ok",
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                side_effect=lambda *args, **kwargs: trace.append("runtime_session") or telemetry,
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

            self.assertEqual(
                trace,
                [
                    "program_external",
                    "open_dut",
                    "open_harness",
                    "prime_harness",
                    "debug_load",
                    "runtime_session",
                    "wait_done",
                ],
            )
            self.assertEqual(
                harness_mock.call_args.kwargs["build_defines"]["TINYODOM_INFERENCE_RUNS"],
                5,
            )
            self.assertEqual(metrics.error_code, HIL_ERROR_OK)
            self.assertEqual(metrics.external_flash_bytes, 4096)
            self.assertEqual(metrics.power_metrics["weight_storage_mode"], "external_flash")
            self.assertEqual(metrics.power_metrics["runs"], 5)
            self.assertEqual(metrics.power_metrics["timer_output_s"], 0.015)
            self.assertEqual(metrics.power_metrics["timer_per_inference_s"], 0.003)
            self.assertEqual(metrics.power_metrics["timer_per_window_s"], 0.003)

    def test_evaluate_harness_ready_timeout_maps_to_latency_error(self) -> None:
        """Ensure missing HARNESS READY becomes a stable latency failure.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Inc" / "tcn_dut_phase_config.h",
                "#define TCN_DUT_MEASURED_RUNS 1\n",
            )
            device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "Debug",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                },
            )()

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
                return_value=project_root / "Debug" / "app.elf",
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": False, "harness_log": [], "error": "timeout"},
                )(),
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

            self.assertEqual(metrics.error_code, HIL_ERROR_LATENCY)
            self.assertEqual(metrics.power_metrics["backend_error_kind"], "harness_ready_timeout")

    def test_evaluate_harness_done_timeout_maps_to_latency_error(self) -> None:
        """Ensure missing harness DONE becomes a stable latency failure.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = _build_project_tree(Path(tmpdir) / "stm")
            _write_text(
                project_root / "Inc" / "tcn_dut_phase_config.h",
                "#define TCN_DUT_MEASURED_RUNS 5\n",
            )
            device = STM32NucleoN657X0QDevice(serial_port="/dev/ttyACM0")
            compile_result = type(
                "CompileResultDouble",
                (),
                {
                    "success": True,
                    "log": "ok",
                    "flash_bytes": 2222,
                    "ram_bytes": 1111,
                    "overflow_kind": None,
                    "build_dir": project_root / "Debug",
                    "arena_bytes": 4096,
                    "external_flash_bytes": None,
                },
            )()
            telemetry = stm32_runtime.STM32RuntimeTelemetry(
                latency_s=0.003,
                serial_log=["STM32_AI_INIT=OK", "DUT READY", "STM32_AI_RUN=OK"],
                power_metrics={
                    "clock_hz": 600000000.0,
                    "sequence": 1.0,
                    "runs": 5,
                    "phase": "back_to_back",
                },
            )

            with patch.object(device, "compile", return_value=compile_result), patch.object(
                device,
                "_storage_power_metrics",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch.object(
                device,
                "_program_external_weights_if_needed",
                return_value={"weight_storage_mode": "embedded", "external_flash_bytes": -1.0},
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.resolve_elf_path",
                return_value=project_root / "Debug" / "app.elf",
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.arduino_base.ensure_harness_firmware"
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.SerialMonitor",
                _FakeSerialMonitor,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.serial.Serial",
                return_value=_FakeHarnessSerial(),
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.hil_protocol.prime_harness_session",
                return_value=type(
                    "PrimeResult",
                    (),
                    {"harness_ready": True, "harness_log": ["HARNESS READY"], "error": None},
                )(),
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.debug_load_elf",
                return_value="upload ok",
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_runtime.execute_runtime_session",
                return_value=telemetry,
            ), patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.hil_protocol.wait_for_harness_done",
                return_value=type(
                    "DoneResult",
                    (),
                    {
                        "harness_done": False,
                        "runs_harness": None,
                        "harness_log": ["HARNESS READY"],
                        "error": "timeout",
                    },
                )(),
            ):
                metrics = device.evaluate(
                    dirpath=project_root,
                    arena_kb=-1,
                    window_size=200,
                    num_channels=6,
                    serial_port="/dev/ttyACM0",
                    run_hil=True,
                    harness_serial_port="/dev/ttyACM1",
                )

            self.assertEqual(metrics.error_code, HIL_ERROR_LATENCY)
            self.assertEqual(metrics.power_metrics["backend_error_kind"], "harness_done_timeout")

    def test_compile_external_flash_overflow_maps_to_flash_failure(self) -> None:
        """Ensure oversized staged external weights fail as flash overflow.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = _build_project_tree(tmp_path / "FSBL")
            manifest_path = project_root / STAGED_MANIFEST_NAME
            manifest_path.write_text(
                "{\n"
                '  "weight_storage_mode": "external_flash",\n'
                '  "weights_blob_size": 4097,\n'
                '  "weights_blob_path": "/tmp/network_data.bin"\n'
                "}\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "template_root": str(project_root),
                    "weight_storage_mode": "external_flash",
                    "max_external_flash_bytes": 4096,
                }
            )

            result = device.compile(
                sketch_path=project_root,
                arena_kb=-1,
                window_size=200,
                num_channels=6,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.overflow_kind, "flash")
        self.assertEqual(result.external_flash_bytes, 4097)

    def test_program_external_weights_autodiscovers_loader_when_manifest_omits_it(self) -> None:
        """Ensure runtime external-flash programming can recover without a stored loader path.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            project_root = _build_project_tree(tmp_path / "FSBL")
            weights_blob = tmp_path / "network_data.bin"
            cubeprog_bin = tmp_path / "cubeprog" / "bin"
            loader_path = cubeprog_bin / "ExternalLoader" / DEFAULT_WEIGHTS_EXTERNAL_LOADER_NAME
            _write_text(weights_blob, "blob")
            _write_text(loader_path, "loader")
            (project_root / STAGED_MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "weight_storage_mode": "external_flash",
                        "weights_blob_path": str(weights_blob),
                        "weights_blob_size": weights_blob.stat().st_size,
                        "weights_flash_address": DEFAULT_WEIGHTS_FLASH_ADDRESS,
                        "weights_external_loader": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            device = STM32NucleoN657X0QDevice(
                device_options={
                    "template_root": str(project_root),
                    "weight_storage_mode": "external_flash",
                    "cubeprog_bin": str(cubeprog_bin),
                }
            )

            with patch(
                "tinyodom.microcontrollers.stm32_nucleo_n657x0.stm32_cube_clt.program_external_flash_blob",
                return_value="program ok",
            ) as program_mock:
                metrics = device._program_external_weights_if_needed(project_root)

        self.assertEqual(metrics["weight_storage_mode"], "external_flash")
        self.assertEqual(program_mock.call_args.kwargs["external_loader"], loader_path.resolve())

    def test_parse_size_output_wraps_host_os_errors_in_workflow_error(self) -> None:
        """Ensure host OS errors from ``arm-none-eabi-size`` become ``WorkflowError``.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            elf_path = Path(tmpdir) / "tinyodom_phase2.elf"
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
