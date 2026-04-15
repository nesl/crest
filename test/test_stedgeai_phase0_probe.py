from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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


stedgeai_phase0_probe = _load_module(
    "stedgeai_phase0_probe",
    "analysis_scripts/stm32_example_project/run_stedgeai_phase0_probe.py",
)


class StEdgeAiPhase0ProbeTests(unittest.TestCase):
    def test_run_logged_writes_utf8_logs(self):
        proc = subprocess.CompletedProcess(["stedgeai"], 0, "caf\xe9", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "command.log"
            with patch.object(stedgeai_phase0_probe.subprocess, "run", return_value=proc):
                stedgeai_phase0_probe._run_logged(["stedgeai", "analyze"], log_path)

            self.assertIn("caf\xe9", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
