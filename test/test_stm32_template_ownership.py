import subprocess
import sys
import tempfile
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

    def test_manifest_loader_accepts_two_column_non_vendor_rows(self) -> None:
        """Ensure the manifest test helper matches the shell parser's tolerance.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.tsv"
            manifest_path.write_text(
                "# comment\n"
                "tinyodom_owned\tInc/example.h\n"
                "vendor_copy\tSrc/example.c\tDrivers/example.c\n",
                encoding="utf-8",
            )

            original_manifest_path = globals()["MANIFEST_PATH"]
            globals()["MANIFEST_PATH"] = manifest_path
            try:
                rows = _load_manifest_rows()
            finally:
                globals()["MANIFEST_PATH"] = original_manifest_path

        self.assertEqual(
            rows,
            [
                ("tinyodom_owned", "Inc/example.h", ""),
                ("vendor_copy", "Src/example.c", "Drivers/example.c"),
            ],
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

    def test_setup_script_prunes_stale_vendor_copy_files_between_runs(self) -> None:
        """Ensure manifest removals delete old materialized vendor-copy files.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            firmware_root = project_root / "tools" / "stm32" / "STM32CubeN6"
            canonical_root = project_root / "canonical"
            example_root = project_root / "example"
            manifest_path = project_root / "manifest.tsv"

            for relative_path in ("vendor/keep.c", "vendor/remove.c"):
                source_path = firmware_root / relative_path
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(f"// {relative_path}\n", encoding="utf-8")

            first_manifest = (
                "vendor_copy\tSrc/keep.c\tvendor/keep.c\n"
                "vendor_copy\tSrc/remove.c\tvendor/remove.c\n"
            )
            second_manifest = "vendor_copy\tSrc/keep.c\tvendor/keep.c\n"

            self._run_setup_materialization(
                project_root=project_root,
                firmware_root=firmware_root,
                canonical_root=canonical_root,
                example_root=example_root,
                manifest_path=manifest_path,
                manifest_text=first_manifest,
            )
            self.assertTrue((canonical_root / "Src" / "remove.c").is_file())
            self.assertTrue((example_root / "Src" / "remove.c").is_file())

            self._run_setup_materialization(
                project_root=project_root,
                firmware_root=firmware_root,
                canonical_root=canonical_root,
                example_root=example_root,
                manifest_path=manifest_path,
                manifest_text=second_manifest,
            )

            self.assertTrue((canonical_root / "Src" / "keep.c").is_file())
            self.assertFalse((canonical_root / "Src" / "remove.c").exists())
            self.assertTrue((example_root / "Src" / "keep.c").is_file())
            self.assertFalse((example_root / "Src" / "remove.c").exists())

    def test_setup_script_preserves_reclassified_vendor_copy_files(self) -> None:
        """Ensure reclassified vendor-copy paths are not deleted on refresh.

        Returns
        -------
        None
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            firmware_root = project_root / "tools" / "stm32" / "STM32CubeN6"
            canonical_root = project_root / "canonical"
            example_root = project_root / "example"
            manifest_path = project_root / "manifest.tsv"

            source_path = firmware_root / "vendor" / "keep.c"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text("// vendor/keep.c\n", encoding="utf-8")

            first_manifest = "vendor_copy\tSrc/keep.c\tvendor/keep.c\n"
            second_manifest = "vendor_derived\tSrc/keep.c\t\n"

            self._run_setup_materialization(
                project_root=project_root,
                firmware_root=firmware_root,
                canonical_root=canonical_root,
                example_root=example_root,
                manifest_path=manifest_path,
                manifest_text=first_manifest,
            )
            self.assertTrue((canonical_root / "Src" / "keep.c").is_file())
            self.assertTrue((example_root / "Src" / "keep.c").is_file())

            self._run_setup_materialization(
                project_root=project_root,
                firmware_root=firmware_root,
                canonical_root=canonical_root,
                example_root=example_root,
                manifest_path=manifest_path,
                manifest_text=second_manifest,
            )

            self.assertTrue((canonical_root / "Src" / "keep.c").is_file())
            self.assertTrue((example_root / "Src" / "keep.c").is_file())

    def _run_setup_materialization(
        self,
        *,
        project_root: Path,
        firmware_root: Path,
        canonical_root: Path,
        example_root: Path,
        manifest_path: Path,
        manifest_text: str,
    ) -> None:
        """Run the setup helpers against a temporary manifest and firmware tree.

        Returns
        -------
        None
        """
        manifest_path.write_text(manifest_text, encoding="utf-8")
        script_path = ROOT_DIR / "setup_stm32.sh"
        command = f"""
set -euo pipefail
source <(sed '$d' "{script_path}")
PROJECT_ROOT="{project_root}"
TOOLS_DIR="$PROJECT_ROOT/tools"
STM32_TOOLS_DIR="$TOOLS_DIR/stm32"
FIRMWARE_DIR="{firmware_root}"
CANONICAL_TEMPLATE_ROOT="{canonical_root}"
OWNERSHIP_MANIFEST="{manifest_path}"
EXTRA_TEMPLATE_ROOTS=("{example_root}")
MANIFEST_PATHS=()
declare -A CATEGORY_BY_PATH=()
declare -A SOURCE_BY_PATH=()
load_ownership_manifest
assemble_canonical_template
refresh_example_templates
"""
        subprocess.run(["bash", "-lc", command], check=True, cwd=ROOT_DIR)


if __name__ == "__main__":
    unittest.main()
