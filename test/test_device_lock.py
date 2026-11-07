# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise device exclusion with independent Python processes."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis_scripts/experiment_support/device_lock.py"
SPEC = importlib.util.spec_from_file_location("device_lock_tests", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def child_attempt(directory, identity):
    """Return a fresh process's lock attempt result."""
    code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('device_lock', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    with module.device_lock(sys.argv[3], sys.argv[2]):
        print('acquired')
except module.DeviceBusyError:
    print('busy')
"""
    return subprocess.run(
        [sys.executable, "-c", code, str(MODULE_PATH), str(directory), identity],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()


def test_cross_process_exclusion_and_release(tmp_path):
    """Exclude matching identities and release after exceptional exit."""
    with pytest.raises(RuntimeError, match="test exit"):
        with MODULE.device_lock("board/one", tmp_path) as path:
            assert child_attempt(tmp_path, "board/one") == "busy"
            assert child_attempt(tmp_path, "board/two") == "acquired"
            inode = path.stat().st_ino
            raise RuntimeError("test exit")
    assert child_attempt(tmp_path, "board/one") == "acquired"
    assert path.stat().st_ino == inode


def test_environment_directory_and_stable_name(tmp_path, monkeypatch):
    """Honor shared configuration without embedding device paths."""
    monkeypatch.setenv("CREST_DEVICE_LOCK_DIR", str(tmp_path))
    with MODULE.device_lock("/dev/serial/device") as path:
        assert path.parent == tmp_path
        assert len(path.stem) == 64
        assert child_attempt(tmp_path, "/dev/serial/device") == "busy"


def test_abrupt_process_exit_releases_lock(tmp_path):
    """Let the operating system release a terminated owner's file lock."""
    code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('device_lock', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with module.device_lock('board', sys.argv[2]):
    print('ready', flush=True)
    sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code, str(MODULE_PATH), str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        assert child_attempt(tmp_path, "board") == "busy"
    finally:
        process.kill()
        process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()
    assert child_attempt(tmp_path, "board") == "acquired"


@pytest.mark.parametrize("identity", [None, "", "   "])
def test_empty_identity_rejected(identity, tmp_path):
    """Reject identities that cannot identify a shared device."""
    with pytest.raises(ValueError, match="device_id"):
        with MODULE.device_lock(identity, tmp_path):
            pass
