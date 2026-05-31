"""Pytest collection helpers for the TinyODOM test suite."""

import os
from pathlib import Path


ANALYSIS_SCRIPT_TESTS = {
    "test_compare_pareto_fronts.py",
    "test_cs3_audio_sensitivity.py",
    "test_micro_workload_energy_probe.py",
}


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Keep non-default suites opt-in for the default `pytest test/` run.

    Parameters
    ----------
    collection_path : Path
        Path to the collection used by the helper.
    config : object
        Configuration object used by the helper.

    Returns
    -------
    bool
        Whether pytest should skip collecting the given path.
    """
    path = Path(str(collection_path))
    if (
        os.environ.get("RUN_INTEGRATION_TESTS") != "1"
        and "integration" in path.parts
        and path.name.startswith("test_")
    ):
        return True
    return os.environ.get("RUN_ANALYSIS_SCRIPT_TESTS") != "1" and path.name in ANALYSIS_SCRIPT_TESTS
