import subprocess
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANONICAL_ROOT = ROOT_DIR / "sketches" / "stm32" / "tinyodom_tcn_stm32" / "FSBL"
MANIFEST_PATH = ROOT_DIR / "sketches" / "stm32" / "tinyodom_tcn_stm32" / "fsbl_ownership_manifest.tsv"
ALLOWED_CATEGORIES = {
    "vendor_copy",
    "vendor_derived",
    "tinyodom_owned",
    "generated",
    "build_recipe",
}


def _load_manifest_rows() -> list[tuple[str, str, str]]:
    """Load the STM32 ownership manifest.

    Returns
    -------
    list[tuple[str, str, str]]
        Parsed ``(category, relative_path, source_path)`` rows. ``source_path``
        is an empty string for non-vendor entries.
    """
    rows: list[tuple[str, str, str]] = []
    for raw_line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        category, relative_path, source_path = raw_line.split("\t")
        rows.append((category, relative_path, source_path))
    return rows


class STM32TemplateOwnershipTests(unittest.TestCase):
    """Validate the canonical STM32 ownership manifest and gitignore policy."""

    def test_manifest_categories_and_paths_are_unique(self) -> None:
        """Ensure every manifest entry uses a known category and unique path.

        Returns
        -------
        None
        """
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
        self.assertTrue(
            any(category == "vendor_copy" and relative_path.startswith("Drivers/") for category, relative_path, _ in rows)
        )

    def test_manifest_kept_files_exist_in_canonical_template(self) -> None:
        """Ensure repo-owned canonical files exist where the manifest says.

        Returns
        -------
        None
        """
        for category, relative_path, _ in _load_manifest_rows():
            if category in {"vendor_derived", "tinyodom_owned", "build_recipe"}:
                self.assertTrue((CANONICAL_ROOT / relative_path).is_file(), relative_path)

    def test_vendor_copy_sources_exist_when_cube_checkout_is_available(self) -> None:
        """Ensure vendor-copy rows point at valid CubeN6 sources when present.

        Returns
        -------
        None
        """
        firmware_root = ROOT_DIR / "tools" / "stm32" / "STM32CubeN6"
        if not firmware_root.is_dir():
            self.skipTest("STM32CubeN6 checkout is not present under tools/stm32")

        for category, relative_path, source_path in _load_manifest_rows():
            if category == "vendor_copy":
                self.assertTrue((firmware_root / source_path).is_file(), relative_path)

    def test_gitignore_covers_vendor_copy_and_generated_paths(self) -> None:
        """Ensure ignored canonical files match the ownership plan.

        Returns
        -------
        None
        """
        for category, relative_path, _ in _load_manifest_rows():
            if category not in {"vendor_copy", "generated"}:
                continue

            result = subprocess.run(
                ["git", "check-ignore", "-q", str(CANONICAL_ROOT / relative_path)],
                cwd=ROOT_DIR,
                check=False,
            )
            self.assertEqual(result.returncode, 0, relative_path)

    def test_setup_script_references_manifest_and_canonical_template(self) -> None:
        """Ensure the STM setup script is wired to the ownership manifest.

        Returns
        -------
        None
        """
        script_text = (ROOT_DIR / "setup_stm32.sh").read_text(encoding="utf-8")
        self.assertIn("fsbl_ownership_manifest.tsv", script_text)
        self.assertIn("assemble_canonical_template", script_text)
        self.assertIn("validate_tracked_template_files", script_text)


if __name__ == "__main__":
    unittest.main()
