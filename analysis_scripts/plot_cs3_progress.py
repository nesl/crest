#!/usr/bin/env python3
"""Render a read-only offline progress snapshot from an Optuna SQLite study."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from crest.progress import read_trials, write_progress

def main():
    """Write PNG, PDF, and CSV progress artifacts without changing the database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-name', required=True)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--warmup-count', type=int, default=0)
    args = parser.parse_args()
    if args.warmup_count < 0:
        parser.error('--warmup-count must be nonnegative')
    directory = args.study_dir
    rows = write_progress(read_trials(directory / 'optuna.db', args.study_name),
                          args.output_dir, args.study_name, args.warmup_count)
    print(f'Saved {len(rows)} finished attempts to {args.output_dir / "progress"}')


if __name__ == '__main__':
    main()
