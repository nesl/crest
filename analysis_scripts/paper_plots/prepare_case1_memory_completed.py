#!/usr/bin/env python3
# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Filter saved memory-proxy replays to the first completed source population.

The source CSV must retain original completed-row order and contain
objective_values_json (a JSON [RMSE, proxy] pair). Replay directories must contain
replay_results.csv with source_row_index and manifest.json. No measurements run.
"""
import argparse
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    """Write rows preserving their original field order.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV.
    rows : list of dict
        Nonempty source rows.
    """
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Filter saved source and replay rows into a separate output directory.

    Notes
    -----
    The source CSV must contain completed trials in their original completion
    order. The first ``--completed-count`` rows define the population; their
    zero-based positions must match replay ``source_row_index`` values. Both
    objective values are minimized when selecting its Pareto front. Outputs
    include the source prefix and one replay CSV/manifest pair per label.
    All input/output path collisions are rejected before any files are written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', type=Path, required=True)
    parser.add_argument('--replay', action='append', required=True,
                        help='LABEL=REPLAY_DIRECTORY; repeat for each board.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--completed-count', type=int, default=150)
    args = parser.parse_args()
    if args.completed_count < 1:
        parser.error('--completed-count must be positive')
    with args.source_csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < args.completed_count:
        raise ValueError('Source has fewer completed rows than requested')
    selected_source = rows[:args.completed_count]
    points = [(i, *json.loads(row['objective_values_json']))
              for i, row in enumerate(selected_source)]
    front = {x[0] for x in points if not any(
        y[1] <= x[1] and y[2] <= x[2] and (y[1] < x[1] or y[2] < x[2])
        for y in points)}
    # Validate all inputs before creating destination files.
    prepared = []
    input_paths = {args.source_csv.resolve()}
    output_paths = [(args.output_dir / 'memory_completed_source.csv').resolve()]
    labels = set()
    for value in args.replay:
        label, directory = value.split('=', 1)
        if Path(label).name != label or label in ('', '.', '..'):
            raise ValueError('Replay label must be a single directory name')
        if label in labels:
            raise ValueError('Replay labels must be unique')
        labels.add(label)
        directory = Path(directory)
        input_paths.update((directory / name).resolve()
                           for name in ('replay_results.csv', 'manifest.json'))
        destination = args.output_dir / label
        if destination.resolve() == directory.resolve():
            raise ValueError('Output must differ from source replay directory')
        with (directory / 'replay_results.csv').open(newline='') as stream:
            replay = list(csv.DictReader(stream))
        selected = [r for r in replay if int(r['source_row_index']) in front]
        if len(selected) != len(front) or {int(r['source_row_index']) for r in selected} != front:
            raise ValueError('Expected one saved replay per selected source-front row')
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest.update(total_source_rows=args.completed_count,
                        selected_pareto_rows=len(front), scheduled_candidates=len(front),
                        offline_population_filter='Saved replay rows filtered by completed source population; no new target evaluations')
        prepared.append((destination, selected, manifest))
        output_paths.extend((destination / name).resolve()
                            for name in ('replay_results.csv', 'manifest.json'))
    if len(set(output_paths)) != len(output_paths) or input_paths.intersection(output_paths):
        raise ValueError('Output files must be distinct from each other and all source files')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'memory_completed_source.csv', selected_source)
    for destination, selected, manifest in prepared:
        destination.mkdir(parents=True, exist_ok=True)
        write_csv(destination / 'replay_results.csv', selected)
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'completed_population': args.completed_count,
                      'front_source_row_indices': sorted(front)}, indent=2))


if __name__ == '__main__':
    main()
