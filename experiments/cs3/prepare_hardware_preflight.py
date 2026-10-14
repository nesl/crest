#!/usr/bin/env python3
"""Generate isolated one-attempt CS3 HIL smoke configs; never execute a run."""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STUDY = "CS3_audio_hil_preflight_v1"


def main():
    """Write fresh role configs and the first baseline candidate.

    Returns
    -------
    None
        Writes preparation files only; no model or hardware code is imported.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/cs3/preflight_v1")
    parser.add_argument("--port", type=int, default=6003)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    destination = args.output_dir.resolve()
    if destination.exists():
        parser.error("output directory exists; preserve evidence before an intentional rerun")
    base = yaml.safe_load((ROOT / "experiments/cs3/nas_config_cs3_luna.yaml").read_text())
    warmup = json.loads((ROOT / "experiments/cs3/warmup_params.json").read_text())
    if not isinstance(warmup, list) or not warmup or not isinstance(warmup[0], dict):
        parser.error("baseline warm-up must contain parameter mappings")
    base["optimizer"] = {"type": "llm_generator", "llm": {
        "provider": "fake", "model": "cs3-hil-preflight-no-api",
        "responses": ['{"candidates": []}'], "batch_size": 1,
        "max_repair_attempts": 0, "random_seed": 0,
        "warmup_params_file": "warmup_first.json", "memory": {"enabled": False},
    }}
    base["network"].update(host="127.0.0.1", port=args.port)
    base["training"].update(nas_trials=1, max_total_trials=1, nas_epochs=3, model_epochs=3)
    base["device"]["stm32"]["weights_memory_pool"] = "src/config/stm32_nucleo_mypool.json"
    base["outputs"].update(artifact_stem=STUDY, log_file_name="log_NAS_" + STUDY + ".csv")
    destination.mkdir(parents=True)
    (destination / "warmup_first.json").write_text(json.dumps(warmup[:1], indent=2) + "\n")
    for role in ("client", "server"):
        base["outputs"]["models_dir"] = f"models/cs3_hil_preflight_v1/{role}"
        (destination / f"{role}.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    print(f"Prepared {destination}; study={STUDY}; port={args.port}. No run started.")


if __name__ == "__main__":
    main()
