"""Tests that guard portability of the canonical STM32 project templates."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STM32_ROOT = REPO_ROOT / "analysis_scripts" / "stm32_example_project"
CANONICAL_ROOT = REPO_ROOT / "sketches" / "stm32" / "tinyodom_stm32_lrun"


class Stm32ProjectPortabilityTests(unittest.TestCase):
    def test_canonical_lrun_makefiles_avoid_tools_checkout_paths(self):
        # Canonical LRUN makefiles should avoid hard-coded tools checkout paths so the project remains portable.
        paths = [
            CANONICAL_ROOT / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "subdir.mk",
            CANONICAL_ROOT / "STM32CubeIDE" / "Boot" / "Debug" / "Src" / "subdir.mk",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("tools/stm32/STM32CubeN6", text, path)
            self.assertIn("../../../Drivers/STM32N6xx_HAL_Driver/Inc", text, path)

    def test_canonical_lrun_app_recipe_includes_repo_relative_secure_nsclib(self):
        # The canonical LRUN app recipe should use the repo-relative secure NSC library path.
        path = CANONICAL_ROOT / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "subdir.mk"
        text = path.read_text(encoding="utf-8")
        self.assertIn("../../../Secure_nsclib", text, path)
        self.assertIn("../../../Appli/Src/secure_nsc.c", text, path)

    def test_canonical_lrun_app_recipe_uses_general_dut_runner_name(self) -> None:
        """Ensure the AppS recipe builds the generalized DUT runner source.

        Returns
        -------
        None
        """
        subdir_text = (
            CANONICAL_ROOT / "STM32CubeIDE" / "AppS" / "Debug" / "Src" / "subdir.mk"
        ).read_text(encoding="utf-8")
        objects_text = (
            CANONICAL_ROOT / "STM32CubeIDE" / "AppS" / "Debug" / "objects.list"
        ).read_text(encoding="utf-8")

        self.assertIn("../../../Appli/Src/tinyodom_dut_runner.c", subdir_text)
        self.assertIn("./Src/tinyodom_dut_runner.o", subdir_text)
        self.assertIn('"./Src/tinyodom_dut_runner.o"', objects_text)
        self.assertNotIn("tcn_" + "dut_runner", subdir_text)
        self.assertNotIn("tcn_" + "dut_runner", objects_text)

    def test_toy_projects_no_longer_pin_stedgeai_version_in_makefiles(self):
        # Example makefiles should stop pinning a specific ST Edge AI version so local toolchains remain portable.
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
        # Example projects should carry the repo's ST Edge AI override hooks.
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
        # Example projects should stay free of CubeIDE metadata so they remain portable and diff-friendly.
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
