"""Pytest collection helpers for the TinyODOM test suite."""

import os
from pathlib import Path


ANALYSIS_SCRIPT_TESTS = {
    "test_analysis_scripts.py",
    "test_stedgeai_phase0_probe.py",
    "test_stm32_build_wrapper.py",
    "test_stm32_project_portability.py",
    "test_stm32_runner_wrappers.py",
    "test_stm32_template_ownership.py",
    "test_urbansound8k_input_profile.py",
}


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Keep non-default suites opt-in for the default `pytest test/` run."""
    path = Path(str(collection_path))
    if (
        os.environ.get("RUN_INTEGRATION_TESTS") != "1"
        and "integration" in path.parts
        and path.name.startswith("test_")
    ):
        return True
    return os.environ.get("RUN_ANALYSIS_SCRIPT_TESTS") != "1" and path.name in ANALYSIS_SCRIPT_TESTS
