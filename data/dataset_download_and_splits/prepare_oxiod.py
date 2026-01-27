#!/usr/bin/env python3
"""
Unzip the OxIOD archive and restore the curated split text files.

Workflow:
1. Place the OxIOD download zipfile at --zip-path.
2. Keep the hand-authored split files that are git tracked under data/oxiod/<activity>.
3. Run `python data/dataset_download_and_splits/prepare_oxiod.py` after cloning; it wipes data/oxiod,
   extracts the archive, fixes folder names ("slow walking" -> "slow_walking"),
   and writes the tracked split text files back into each activity folder.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

SPLIT_FILES = ("Train.txt", "Valid.txt", "Test.txt", "Train_Valid.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        default="OxIOD.zip",
        help="Path to the OxIOD zip archive (default: %(default)s)",
    )
    parser.add_argument(
        "--dest-root",
        default="data/oxiod",
        help="Destination directory for the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--template-root",
        default="data/oxiod",
        help="Directory that currently holds the tracked split files "
        "(default: %(default)s)",
    )
    return parser.parse_args()


def capture_templates(template_root: Path) -> dict[Path, dict[str, str]]:
    """Read the tracked split files into memory before we blow away the folder."""
    templates: dict[Path, dict[str, str]] = {}
    for activity_dir in template_root.iterdir():
        if not activity_dir.is_dir():
            continue
        splits = {}
        for split_name in SPLIT_FILES:
            split_path = activity_dir / split_name
            if split_path.exists():
                splits[split_name] = split_path.read_text(encoding="utf-8")
        if splits:
            templates[activity_dir.relative_to(template_root)] = splits
    if not templates:
        raise RuntimeError(
            f"No split templates found under {template_root}. "
            "Ensure you cloned the repo with the curated .txt files."
        )
    return templates


def locate_dataset_root(tmp_path: Path) -> Path:
    """Find the folder inside the extracted archive that holds the activities."""
    candidates = [
        tmp_path / "data" / "oxiod",
        tmp_path / "Oxford Inertial Odometry Dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # fallback: find a folder that has the canonical activity directories
    activity_markers = {"handbag", "handheld", "pocket"}
    for folder in tmp_path.rglob("*"):
        if folder.is_dir() and activity_markers.issubset(
            {p.name for p in folder.iterdir() if p.is_dir()}
        ):
            return folder
    raise RuntimeError("Unable to locate the OxIOD root inside the archive.")


def extract_archive(zip_path: Path, dest_root: Path) -> None:
    """Extract only the oxiod portion of the archive into dest_root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_path)
        extracted_root = locate_dataset_root(tmp_path)
        if dest_root.exists():
            shutil.rmtree(dest_root)
        shutil.copytree(extracted_root, dest_root)


def normalize_folder_names(dest_root: Path) -> None:
    """Match the repo's folder naming (handle slow walking -> slow_walking)."""
    slow_original = dest_root / "slow walking"
    slow_target = dest_root / "slow_walking"
    if slow_original.exists():
        if slow_target.exists():
            shutil.rmtree(slow_target)
        slow_original.rename(slow_target)


def restore_templates(dest_root: Path, templates: dict[Path, dict[str, str]]) -> None:
    """Write the stored split files back into each activity folder."""
    for rel_path, splits in templates.items():
        folder = dest_root / rel_path
        folder.mkdir(parents=True, exist_ok=True)
        for split_name, content in splits.items():
            (folder / split_name).write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    dest_root = Path(args.dest_root)
    template_root = Path(args.template_root)

    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")
    if not template_root.exists():
        raise FileNotFoundError(f"Template directory not found: {template_root}")

    templates = capture_templates(template_root)
    extract_archive(zip_path, dest_root)
    normalize_folder_names(dest_root)
    restore_templates(dest_root, templates)
    print(f"OxIOD prepared in {dest_root}")


if __name__ == "__main__":
    main()
