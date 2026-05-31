"""Static logging hygiene checks for source files."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"


def _executable_print_calls(source_path: Path) -> list[tuple[int, int]]:
    """Return executable ``print`` call locations in one Python source file.

    Parameters
    ----------
    source_path : pathlib.Path
        Python source file to parse.

    Returns
    -------
    list[tuple[int, int]]
        One-based line and column locations for every executable ``print`` call.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    locations: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            locations.append((node.lineno, node.col_offset))
    return locations


def test_src_contains_no_executable_print_calls() -> None:
    """Source files should use logging instead of executable prints.

    Returns
    -------
    None
        The test passes when every Python file under ``src`` is free of
        executable ``print`` calls.
    """
    offenders: list[str] = []
    for source_path in sorted(SRC_DIR.rglob("*.py")):
        for line_number, column in _executable_print_calls(source_path):
            rel_path = source_path.relative_to(ROOT_DIR)
            offenders.append(f"{rel_path}:{line_number}:{column}")
    assert not offenders, "Executable print calls remain:\n" + "\n".join(offenders)
