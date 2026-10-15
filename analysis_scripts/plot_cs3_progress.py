#!/usr/bin/env python3
"""Render a read-only offline progress snapshot from an Optuna SQLite study."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from crest.progress import read_trials, write_progress

DEFAULT_STUDY = 'UrbanSound8K_DSCNN_STM32_AUDIO_case3_llm_t2'


def main():
    """Write PNG, PDF, and CSV progress artifacts without changing the database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-name', default=DEFAULT_STUDY)
    parser.add_argument('--study-dir', type=Path)
    parser.add_argument('--warmup-count', type=int, default=15)
    args = parser.parse_args()
    directory = args.study_dir or ROOT / 'models/cs3_luna_production_v2/client' / args.study_name
    rows = write_progress(read_trials(directory / 'optuna.db', args.study_name),
                          directory, args.study_name, args.warmup_count)
    print(f'Saved {len(rows)} finished attempts to {directory / "progress"}')


if __name__ == '__main__':
    main()
