from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STM32_ROOT = REPO_ROOT / "analysis_scripts" / "stm32_example_project"


class Stm32ProjectPortabilityTests(unittest.TestCase):
    def test_toy_projects_no_longer_pin_stedgeai_version_in_makefiles(self):
        paths = [
            STM32_ROOT / "stm32_toy_ai_project" / "FSBL" / "Debug" / "objects.mk",
            STM32_ROOT / "stm32_toy_ai_project" / "FSBL" / "Debug" / "Src" / "subdir.mk",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / "FSBL" / "Debug" / "objects.mk",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / "FSBL" / "Debug" / "Src" / "subdir.mk",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/opt/ST/STEdgeAI/4.0", text, path)

    def test_toy_projects_include_repo_stedgeai_overrides(self):
        paths = [
            STM32_ROOT / "stm32_toy_ai_project" / "FSBL" / "Debug" / "stedgeai.mk",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / "FSBL" / "Debug" / "stedgeai.mk",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("STEDGEAI_ROOT ?=", text, path)
            self.assertIn("NetworkRuntime1200_CM55_GCC.a", text, path)
            self.assertIn("Middlewares/ST/AI/Inc", text, path)

    def test_cubeide_metadata_files_are_removed(self):
        paths = [
            STM32_ROOT / "stm32_blink_example_project" / ".project",
            STM32_ROOT / "stm32_blink_example_project" / "FSBL" / ".project",
            STM32_ROOT / "stm32_blink_example_project" / "FSBL" / ".cproject",
            STM32_ROOT / "stm32_blink_example_project" / "FSBL" / "stm32_blink_example_project_FSBL.launch",
            STM32_ROOT / "stm32_toy_ai_project" / ".project",
            STM32_ROOT / "stm32_toy_ai_project" / "FSBL" / ".project",
            STM32_ROOT / "stm32_toy_ai_project" / "FSBL" / ".cproject",
            STM32_ROOT / "stm32_toy_ai_project" / "FSBL" / "stm32_blink_example_project_FSBL.launch",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / ".project",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / "FSBL" / ".project",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / "FSBL" / ".cproject",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / "FSBL" / "stm32_blink_example_project_FSBL.launch",
        ]
        for path in paths:
            self.assertFalse(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
