"""Tests for the checked-in STM32 LRUN template ownership manifest."""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANONICAL_ROOT = ROOT_DIR / "sketches" / "stm32" / "tinyodom_tcn_stm32_lrun"
MANIFEST_PATH = CANONICAL_ROOT / "lrun_ownership_manifest.tsv"
ALLOWED_CATEGORIES = {
    "vendor_copy",
    "vendor_derived",
    "tinyodom_owned",
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
    def test_manifest_categories_and_paths_are_unique(self) -> None:
        # Verify that manifest categories and paths are unique.
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
        # Verify that manifest kept files exist in canonical workspace.
        for category, relative_path, _ in _load_manifest_rows():
            if category in {"vendor_derived", "tinyodom_owned", "build_recipe"}:
                self.assertTrue((CANONICAL_ROOT / relative_path).is_file(), relative_path)

    def test_vendor_copy_sources_exist_when_cube_checkout_is_available(self) -> None:
        # Verify that vendor copy sources exist when cube checkout is available.
        firmware_root = ROOT_DIR / "tools" / "stm32" / "STM32CubeN6"
        if not firmware_root.is_dir():
            self.skipTest("STM32CubeN6 checkout is not present under tools/stm32")

        for category, relative_path, source_path in _load_manifest_rows():
            if category == "vendor_copy":
                self.assertTrue((firmware_root / source_path).is_file(), relative_path)

    def test_gitignore_covers_setup_managed_vendor_copy_and_generated_paths(self) -> None:
        # Verify that gitignore covers setup managed vendor copy and generated paths.
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
        # Verify that setup script references LRUN manifest only.
        script_text = (ROOT_DIR / "setup_stm32.sh").read_text(encoding="utf-8")
        self.assertIn("lrun_ownership_manifest.tsv", script_text)
        self.assertIn("prune_materialized_vendor_copy_files", script_text)
        self.assertNotIn("fsbl_ownership_manifest.tsv", script_text)
        self.assertNotIn("assemble_canonical_template", script_text)


if __name__ == "__main__":
    unittest.main()
