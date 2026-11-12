# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Identity checks for ESP32 embedded model build artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


_HEX_BYTE = re.compile(r"0x([0-9a-fA-F]{2})")


def verify_generated_model_source(*, model_path: Path, source_path: Path) -> str:
    """Verify generated C array bytes exactly equal the staged TFLite bytes."""

    model_bytes = model_path.read_bytes()
    generated_bytes = bytes(int(value, 16) for value in _HEX_BYTE.findall(source_path.read_text()))
    if generated_bytes != model_bytes:
        raise ValueError("Generated ESP32 model source does not match the staged TFLite bytes.")
    return hashlib.sha256(model_bytes).hexdigest()


def resolve_built_application(build_dir: Path) -> Path:
    """Resolve the ESP-IDF application binary from project metadata."""

    description_path = build_dir / "project_description.json"
    description = json.loads(description_path.read_text(encoding="utf-8"))
    project_name = description.get("project_name")
    if not isinstance(project_name, str) or not project_name:
        raise ValueError("ESP-IDF project_description.json has no project_name.")
    app_path = build_dir / f"{project_name}.bin"
    if not app_path.is_file():
        raise ValueError(f"ESP-IDF application binary was not found: {app_path}")
    return app_path


def verify_application_contains_model(*, app_path: Path, model_path: Path) -> dict[str, object]:
    """Verify exact flatbuffer bytes occur contiguously in the app image."""

    app_bytes = app_path.read_bytes()
    model_bytes = model_path.read_bytes()
    occurrences = app_bytes.count(model_bytes)
    if occurrences != 1:
        raise ValueError(
            f"ESP-IDF application must contain the exact TFLite artifact once; found {occurrences}."
        )
    return {
        "application_path": str(app_path.resolve()),
        "application_sha256": hashlib.sha256(app_bytes).hexdigest(),
        "application_size_bytes": len(app_bytes),
        "embedded_model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "embedded_model_size_bytes": len(model_bytes),
        "embedded_model_occurrences": occurrences,
    }
