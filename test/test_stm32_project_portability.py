"""Tests that guard portability of the canonical STM32 project templates."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "sketches" / "stm32" / "tinyodom_stm32_lrun"


class Stm32ProjectPortabilityTests(unittest.TestCase):
    """Tests covering STM32 project portability behavior."""

    def test_canonical_lrun_makefiles_avoid_tools_checkout_paths(self):
        # Canonical LRUN makefiles should avoid hard-coded tools checkout paths so the project remains portable.
        """Validate canonical lrun makefiles avoid tools checkout paths."""
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
        """Validate canonical lrun app recipe includes repo relative secure nsclib."""
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


if __name__ == "__main__":
    unittest.main()
