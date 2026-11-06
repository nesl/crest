#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Compare regenerated paper PNGs with an external submitted-asset directory.

Compare decoded RGBA pixels, not PNG metadata or compression. The command exits
nonzero if an asset is missing, dimensions differ, or any pixel differs. Neither
input directory is modified. This is an appearance check, not a scientific
validation of the data or its provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ASSETS = (
    "cs1_clock_reversal.png",
    "cs1_energy_ranks.png",
    "cs1_model_transfer.png",
    "cs2_proxy_replay_fronts.png",
    "cs3_proposal_comparison.png",
    "cs2_training_budget_ranking.png",
    "cs2_training_budget_fronts.png",
    "appendix_timing_analysis.png",
)


def compare_images(reference: Path, generated: Path) -> dict[str, Any]:
    """Compare image dimensions and all decoded color/alpha channels.

    Parameters
    ----------
    reference : pathlib.Path
        Submitted PNG asset, treated as immutable.
    generated : pathlib.Path
        Regenerated PNG asset.

    Returns
    -------
    dict
        Match status, dimensions, file hashes, and pixel difference metrics.
        Missing files or different dimensions are reported as mismatches.
    """
    result: dict[str, Any] = {"asset": reference.name, "match": False}
    missing = [str(p) for p in (reference, generated) if not p.is_file()]
    if missing:
        return {**result, "missing": missing}
    for label, path in (("reference", reference), ("generated", generated)):
        result[f"{label}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(reference) as image:
        expected = np.array(image.convert("RGBA"), dtype=np.int16)
    with Image.open(generated) as image:
        actual = np.array(image.convert("RGBA"), dtype=np.int16)
    result["reference_size"] = [expected.shape[1], expected.shape[0]]
    result["generated_size"] = [actual.shape[1], actual.shape[0]]
    if expected.shape != actual.shape:
        return result
    delta = np.abs(expected - actual)
    result.update(
        differing_pixels=int(np.any(delta != 0, axis=2).sum()),
        max_channel_difference=int(delta.max()),
        mean_absolute_channel_difference=float(delta.mean()),
        match=bool(np.all(delta == 0)),
    )
    return result


def main() -> int:
    """Check all eight assets and optionally save a JSON report.

    Returns
    -------
    int
        Zero only when every submitted asset matches pixel for pixel.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    comparisons = [
        compare_images(args.reference_dir / name, args.generated_dir / name)
        for name in ASSETS
    ]
    report = {
        "all_match": all(row["match"] for row in comparisons),
        "figures": comparisons,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload, end="")
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
