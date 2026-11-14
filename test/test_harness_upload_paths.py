"""Regression tests for upload paths and retained harness diagnostics."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from crest.microcontrollers import arduino_base


@pytest.mark.parametrize("use_symlink", [False, True])
def test_upload_port_matches_discovery(tmp_path, monkeypatch, use_symlink):
    """Check uploader port normalization without opening a serial device.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing subprocess execution.
    use_symlink : bool
        Whether to pass a stable device alias.
    """
    target = tmp_path / "ttyACM0"
    target.touch()
    alias = tmp_path / "by-id"
    alias.symlink_to(target)
    port = str(alias if use_symlink else target)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    monkeypatch.setattr(arduino_base.subprocess, "run", run)
    arduino_base.upload_sketch(sketch_path=tmp_path, fqbn="arduino:mbed_nano:nano33ble",
                              build_dir=tmp_path, serial_port=port)
    command = run.call_args.args[0]
    assert command[command.index("-p") + 1] == str(target)


def test_harness_upload_error_retains_cli_output(tmp_path, monkeypatch):
    """Retain uploader diagnostics in the propagated harness error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary build directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing compile and upload operations.
    """
    monkeypatch.setattr(arduino_base, "compile_harness_sketch",
                        Mock(return_value=SimpleNamespace(success=True, build_dir=tmp_path, log="ok")))
    monkeypatch.setattr(arduino_base, "upload_harness_sketch",
                        Mock(return_value=SimpleNamespace(success=False, log="No device found on ttyACM0")))
    with pytest.raises(RuntimeError, match="No device found on ttyACM0"):
        arduino_base.ensure_harness_firmware(harness_serial_port="test", harness_auto_flash="always")
