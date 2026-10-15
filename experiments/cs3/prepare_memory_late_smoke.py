#!/usr/bin/env python3
"""Prepare a deterministic archived-baseline proxy for the late memory smoke test."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from optuna.distributions import json_to_distribution
from crest.optimizers.llm.history import build_terminal_trial_history
from crest.optimizers.llm.memory import summary_text


LABEL = 'Archived baseline proxy study/objective; deterministic seeded memory, not a historical LLM cumulative summary.'


def seed_findings(records):
    """Describe grouped observations using only the first 75 terminal records.

    Parameters
    ----------
    records : list of dict
        Exactly records numbered zero through 74.

    Returns
    -------
    list of dict
        Four factual findings with complete group-input audit references.
    """
    if [r['number'] for r in records] != list(range(75)):
        raise ValueError('Seed evidence must be exactly trials 0..74')
    findings = []
    for key in ('num_blocks', 'base_channels', 'stride_schedule', 'global_pool_type'):
        groups = defaultdict(list)
        for record in records:
            groups[json.dumps(record['params'][key])].append(record)
        descriptions = []
        for value, rows in sorted(groups.items()):
            scores = [r['value'] for r in rows if isinstance(r.get('value'), (int, float))]
            latencies = [r['latency_ms'] for r in rows if 'latency_ms' in r]
            feasible = sum(r.get('feasibility_status') == 'feasible' for r in rows)
            score_text = (f'score n={len(scores)}, mean={sum(scores)/len(scores):.5f}, '
                          f'range={min(scores):.5f}..{max(scores):.5f}') if scores else 'score n=0'
            latency_text = (f'latency n={len(latencies)}, range={min(latencies):.3f}..{max(latencies):.3f} ms') if latencies else 'latency n=0'
            descriptions.append(f'{value}: n={len(rows)}, feasible={feasible}; {score_text}; {latency_text}')
        findings.append(dict(
            id=f'seed-{key}',
            observation=f'Observed grouping by {key}: ' + ' | '.join(descriptions) + '.',
            conditions='Archived AudioDSCNN trials 0..74; decoded parameter groups.',
            evidence='All group inputs counted; scores include penalties; latency excludes absent measurements.',
            exceptions='Other architecture settings vary.',
            uncertainty='Observational, confounded comparisons; no causal or convergence claims. Deterministic seed, not an LLM summary.',
            trial_numbers=[r['number'] for r in records]))
    return findings


def prepare(source_dir, source_request, output):
    """Write a fresh scenario from read-only archived SQLite evidence.

    Parameters
    ----------
    source_dir : Path
        Archive containing optuna.db.
    source_request : Path
        Saved memory request supplying the experimental context.
    output : Path
        Fresh output JSON path; existing paths are rejected.

    Returns
    -------
    dict
        Scenario including exact evidence, seeded findings and source hashes.
    """
    source_dir, source_request, output = map(Path, (source_dir, source_request, output))
    if output.exists():
        raise FileExistsError(output)
    database = source_dir / 'optuna.db'
    trials = []
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        rows = connection.execute('SELECT trial_id,number,state FROM trials WHERE number BETWEEN 0 AND 104 ORDER BY number').fetchall()
        for trial_id, number, state in rows:
            params = {key: json_to_distribution(distribution).to_external_repr(value)
                      for key, value, distribution in connection.execute(
                          'SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?', (trial_id,))}
            attrs = {key: json.loads(value) for key, value in connection.execute(
                'SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?', (trial_id,))}
            values = [row[0] for row in connection.execute(
                'SELECT value FROM trial_values WHERE trial_id=? ORDER BY objective', (trial_id,))]
            trials.append(SimpleNamespace(number=number, state=state, params=params,
                                          user_attrs=attrs, values=values or None))
    records = list(build_terminal_trial_history(trials, directions=('maximize',)))
    if [record['number'] for record in records] != list(range(105)):
        raise ValueError('Archive must provide exactly terminal trials 0..104')
    request = json.loads(source_request.read_text())
    bodies = [json.loads(message['content']) for message in request['messages'] if message['role'] == 'user']
    if len(bodies) != 1 or not isinstance(bodies[0].get('context'), dict):
        raise ValueError('Expected exactly one saved user request body containing context')
    findings = seed_findings(records[:75])
    sizes = dict(structured=len(json.dumps(findings, ensure_ascii=False)), prose=len(summary_text(findings)))
    if not 4500 <= sizes['structured'] <= 5500 or max(sizes.values()) > 6000:
        raise ValueError(f'Seed memory outside required character bounds: {sizes}')
    context = dict(bodies[0]['context'])
    context['study_name'] += '_archived_baseline_memory_proxy'
    context['objective_summary'] = (LABEL + ' Records come from the archived baseline, not the stopped LLM run. '
                                    'Hardware context describes the proposed CS3 target; exact historical deployment may differ. '
                                    + context.get('objective_summary', ''))
    scenario = dict(context=context, records=records, initial_findings=findings,
                    covered_trial_numbers=list(range(75)), lag_trials=15,
                    expected_new_trial_numbers=list(range(75, 90)), expected_covered_trial_count=90,
                    provenance=dict(label=LABEL, source_dir=str(source_dir.resolve()),
                                    source_request=str(source_request.resolve()),
                                    database_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
                                    source_request_sha256=hashlib.sha256(source_request.read_bytes()).hexdigest(),
                                    seed_trial_numbers=list(range(75)), memory_characters=sizes))
    with output.open('x') as handle:
        json.dump(scenario, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return scenario


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--source-request', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source_dir, args.source_request, args.output)
