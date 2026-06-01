# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the checked-in STM32 LRUN template ownership manifest."""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANONICAL_ROOT = ROOT_DIR / "sketches" / "stm32" / "crest_stm32_lrun"
MANIFEST_PATH = CANONICAL_ROOT / "lrun_ownership_manifest.tsv"
ALLOWED_CATEGORIES = {
    "vendor_copy",
    "vendor_derived",
    "crest_owned",
    "generated",
    "build_recipe",
}


def _load_manifest_rows() -> list[tuple[str, str, str]]:
    """Parse the LRUN ownership manifest into normalized row tuples.

    Returns
    -------
    list[tuple[str, str, str]]
        ``(category, relative_path, source_path)`` rows. Entries that omit the
        optional source path are normalized to an empty string.

    Raises
    ------
    ValueError
        If existing validation or execution checks fail.
    """
    rows: list[tuple[str, str, str]] = []
    for raw_line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        # The manifest is a compact TSV format where source_path is only
        # present for vendor-copy rows.
        parts = raw_line.split("\t")
        if len(parts) == 2:
            category, relative_path = parts
            source_path = ""
        elif len(parts) == 3:
            category, relative_path, source_path = parts
        else:
            raise ValueError(f"Malformed ownership manifest row: {raw_line!r}")
        rows.append((category, relative_path, source_path))
    return rows


class STM32TemplateOwnershipTests(unittest.TestCase):
    """Tests covering STM32 template ownership behavior."""

    def test_tracked_text_files_do_not_use_legacy_lrun_dut_names(self) -> None:
        """Ensure tracked text files no longer use legacy LRUN DUT identifiers.

        Returns
        -------
        None
        """
        legacy_tokens = (
            "crest_" + "tcn_stm32",
            "crest_" + "tcn_stm32_lrun",
            "tcn_" + "dut",
            "TCN_" + "DUT",
            "Tcn" + "Dut",
        )
        scoped_prefixes = (
            "sketches/stm32/crest_stm32_lrun/",
            "src/config/",
            "src/crest/microcontrollers/stm32_nucleo_n657x0.py",
            "test/test_model.py",
            "test/test_stm32_",
        )
        scoped_files = {
            "README.md",
            "setup_stm32.sh",
            "sketches/README.md",
            "src/crest/microcontrollers/README.md",
            "src/crest/model.py",
        }
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )

        offenders: list[str] = []
        for relative_path in result.stdout.splitlines():
            if not (
                relative_path in scoped_files
                or any(relative_path.startswith(prefix) for prefix in scoped_prefixes)
            ):
                continue
            path = ROOT_DIR / relative_path
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in legacy_tokens:
                if token in relative_path or token in text:
                    offenders.append(f"{relative_path}: {token}")
                    break

        self.assertEqual(offenders, [])

    def test_setup_script_tracks_general_lrun_overlay_paths(self) -> None:
        """Ensure setup validation and cleanup use the generalized LRUN names.

        Returns
        -------
        None
        """
        script_text = (ROOT_DIR / "setup_stm32.sh").read_text(encoding="utf-8")

        self.assertIn("sketches/stm32/crest_stm32_lrun", script_text)
        self.assertIn('"Appli/Inc/crest_dut_runner.h"', script_text)
        self.assertIn('"Appli/Src/crest_dut_runner.c"', script_text)
        self.assertIn(
            'require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/crest_dut_runner.h"',
            script_text,
        )
        self.assertIn(
            'require_file "$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Src/crest_dut_runner.c"',
            script_text,
        )
        self.assertIn(
            '"$CANONICAL_LRUN_TEMPLATE_ROOT/Appli/Inc/crest_dut_phase_config.h"',
            script_text,
        )
        self.assertNotIn("crest_" + "tcn_stm32_lrun", script_text)
        self.assertNotIn("tcn_" + "dut_runner", script_text)
        self.assertNotIn("tcn_" + "dut_phase_config", script_text)

    def test_manifest_categories_and_paths_are_unique(self) -> None:
        # The LRUN ownership manifest should keep categories and paths unique.
        """Validate manifest categories and paths are unique."""
        rows = _load_manifest_rows()
        seen_paths: set[str] = set()

        self.assertTrue(rows)
        for category, relative_path, source_path in rows:
            self.assertIn(category, ALLOWED_CATEGORIES)
            self.assertNotIn(relative_path, seen_paths)
            seen_paths.add(relative_path)
            if category == "vendor_copy":
                self.assertTrue(source_path)
            else:
                self.assertEqual(source_path, "")

    def test_manifest_kept_files_exist_in_canonical_workspace(self) -> None:
        # Files marked as kept in the manifest should still exist in the canonical workspace.
        """Validate manifest kept files exist in canonical workspace."""
        for category, relative_path, _ in _load_manifest_rows():
            if category in {"vendor_derived", "crest_owned", "build_recipe"}:
                self.assertTrue((CANONICAL_ROOT / relative_path).is_file(), relative_path)

    def test_vendor_copy_sources_exist_when_cube_checkout_is_available(self) -> None:
        # Vendor copy sources should exist whenever the Cube checkout is available.
        """Validate vendor copy sources exist when cube checkout is available."""
        firmware_root = ROOT_DIR / "tools" / "stm32" / "STM32CubeN6"
        if not firmware_root.is_dir():
            self.skipTest("STM32CubeN6 checkout is not present under tools/stm32")

        for category, relative_path, source_path in _load_manifest_rows():
            if category == "vendor_copy":
                self.assertTrue((firmware_root / source_path).is_file(), relative_path)

    def test_gitignore_covers_setup_managed_vendor_copy_and_generated_paths(self) -> None:
        # The repo gitignore should keep covering setup-managed vendor copies and generated paths.
        """Validate gitignore covers setup managed vendor copy and generated paths."""
        for category, relative_path, _ in _load_manifest_rows():
            if category not in {"vendor_copy", "generated"}:
                continue
            result = subprocess.run(
                ["git", "check-ignore", "-q", str(CANONICAL_ROOT / relative_path)],
                cwd=ROOT_DIR,
                check=False,
            )
            self.assertEqual(result.returncode, 0, relative_path)

    def test_setup_script_references_lrun_manifest_only(self) -> None:
        # The setup script should only reference the LRUN manifest it is supposed to manage.
        """Validate setup script references lrun manifest only."""
        script_text = (ROOT_DIR / "setup_stm32.sh").read_text(encoding="utf-8")
        self.assertIn("lrun_ownership_manifest.tsv", script_text)
        self.assertIn("prune_materialized_vendor_copy_files", script_text)
        self.assertNotIn("fsbl_ownership_manifest.tsv", script_text)
        self.assertNotIn("assemble_canonical_template", script_text)

    def test_setup_script_preserves_checked_in_license_files(self) -> None:
        # These files are tracked overlay files, so rsync --delete must not remove them.
        """Validate setup script preserves checked in license files."""
        script_text = (ROOT_DIR / "setup_stm32.sh").read_text(encoding="utf-8")

        for license_path in (
            "LICENSE.md",
            "LICENSE.CMSIS.txt",
            "LICENSE.STM32N6xx_HAL_Driver.md",
            "LICENSE.STM32_ExtMem_Manager.md",
        ):
            self.assertIn(f'"{license_path}"', script_text)


if __name__ == "__main__":
    unittest.main()
