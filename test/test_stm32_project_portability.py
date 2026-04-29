"""Tests that guard portability of the canonical STM32 project templates."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STM32_ROOT = REPO_ROOT / "analysis_scripts" / "stm32_example_project"
CANONICAL_ROOT = REPO_ROOT / "sketches" / "stm32" / "tinyodom_tcn_stm32_lrun"


class Stm32ProjectPortabilityTests(unittest.TestCase):
    def test_canonical_lrun_makefiles_avoid_tools_checkout_paths(self):
        # Verify that canonical LRUN makefiles avoid tools checkout paths.
        paths = [
            CANONICAL_ROOT / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "subdir.mk",
            CANONICAL_ROOT / "STM32CubeIDE" / "Boot" / "Debug" / "Src" / "subdir.mk",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("tools/stm32/STM32CubeN6", text, path)
            self.assertIn("../../../Drivers/STM32N6xx_HAL_Driver/Inc", text, path)

    def test_canonical_lrun_app_recipe_includes_repo_relative_secure_nsclib(self):
        # Verify that canonical LRUN app recipe includes repo relative secure nsclib.
        path = CANONICAL_ROOT / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "subdir.mk"
        text = path.read_text(encoding="utf-8")
        self.assertIn("../../../Secure_nsclib", text, path)
        self.assertIn("../../../Appli/Src/secure_nsc.c", text, path)

    def test_toy_projects_no_longer_pin_stedgeai_version_in_makefiles(self):
        # Verify that toy projects no longer pin stedgeai version in makefiles.
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
        # Verify that toy projects include repo stedgeai overrides.
        paths = [
            STM32_ROOT / "stm32_toy_ai_project" / "FSBL" / "Debug" / "stedgeai.mk",
            STM32_ROOT / "stm32_cadenced_toy_ai_project" / "FSBL" / "Debug" / "stedgeai.mk",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("STEDGEAI_ROOT ?=", text, path)
            self.assertIn("STEDGEAI_CANDIDATES :=", text, path)
            self.assertIn("/opt/ST/STEdgeAI/*/Middlewares/ST/AI/Inc", text, path)
            self.assertIn("NetworkRuntime1200_CM55_GCC.a", text, path)
            self.assertIn("Middlewares/ST/AI/Inc", text, path)

    def test_cubeide_metadata_files_are_removed_from_examples(self):
        # Verify that cubeide metadata files are removed from examples.
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
