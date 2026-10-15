#!/usr/bin/env python3
"""Replay exact saved first-15 evidence; paid memory-only calls require --run-paid."""
import argparse
from dataclasses import dataclass, asdict, field
import hashlib
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import yaml
from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.memory import (MEMORY_PROMPT_VERSION, ExperimentalMemory, MemoryConfig,
                                         MemoryUpdate, digest, summary_text)
from crest.optimizers.llm.provider import FakeProvider, build_provider
from crest.optimizers.llm.spending import SpendTracker


@dataclass(frozen=True)
class SavedContext:
    """Memory-only adapter preserving the saved production prompt scope."""
    payload: dict
    recent_trials: tuple = ()
    pending_trials: tuple = ()
    knowledge_base: dict = field(default_factory=dict)

    @property
    def study_name(self):
        """Return the source study identifier.

        Returns
        -------
        str
            Archived study name used only for isolated prompt context.
        """
        return self.payload['study_name']

    def as_prompt_payload(self):
        """Return saved context without reconstructing or changing evidence.

        Returns
        -------
        dict
            Copy of the archived production prompt scope.
        """
        return dict(self.payload)


def run(output, source_request=None, paid=False, response=None, scenario=None):
    """Validate a bounded summary update and offline reload from saved evidence.

    Parameters
    ----------
    output : pathlib.Path
        Fresh evidence directory, never reused.
    source_request : pathlib.Path
        Saved memory request with exact real trials zero through fourteen.
    paid : bool, optional
        Enable at most two HTTP calls, one summary plus one repair.
    response : pathlib.Path, optional
        Saved normalized response or summary JSON for offline replay.
    scenario : pathlib.Path, optional
        Prepared late-stage archived-baseline scenario; exclusive with source_request.

    Returns
    -------
    dict
        Persisted evidence report, also written when validation fails.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    calls = []
    tracker = None
    report = dict(passed=False, mode='paid' if paid else 'offline', max_usd=0.05,
        max_http_requests=2, max_transport_retries=0, max_output_tokens=8192,
        candidate_api_requests=0, hardware_or_training=False,
        scope='Isolated first-15 summary test with lag_trials=0; does not test the 30-trial lag-15 schedule.')
    try:
        if (source_request is None) == (scenario is None):
            raise ValueError('Choose exactly one source request or prepared scenario.')
        source = Path(scenario or source_request)
        raw = source.read_bytes()
        report.update(source_request=str(source.resolve()),
                      source_sha256=hashlib.sha256(raw).hexdigest())
        (output / ('scenario.json' if scenario else 'source.request.json')).write_bytes(raw)
        saved = json.loads(raw)
        initial_findings, covered_numbers = [], []
        if scenario:
            records = saved['records']
            if [r['number'] for r in records] != list(range(105)):
                raise ValueError('Late-stage scenario requires terminal records 0 through 104.')
            if any(str(r['state']).upper() not in ('COMPLETE', 'FAIL', 'PRUNED') for r in records):
                raise ValueError('Late-stage scenario must contain terminal records only.')
            initial_findings = saved['initial_findings']
            covered_numbers = saved['covered_trial_numbers']
            if covered_numbers != list(range(75)) or saved['lag_trials'] != 15:
                raise ValueError('Expected 75 covered records and a 15-record lag.')
            if saved['expected_new_trial_numbers'] != list(range(75, 90)):
                raise ValueError('Expected summary evidence 75 through 89.')
            expected_covered, lag = 90, 15
            context = SavedContext(saved['context'], recent_trials=tuple(records[-15:]))
            report.update(scope='105-trial archived-baseline proxy; deterministic seeded memory, not an actual cumulative LLM run.',
                          provenance=saved['provenance'], new_trial_numbers=list(range(75, 90)),
                          held_out_trial_numbers=list(range(90, 105)))
        else:
            body = json.loads(next(m['content'] for m in saved['messages'] if m['role'] == 'user'))
            records = body['new_trials']
            if [r['number'] for r in records] != list(range(15)):
                raise ValueError('Expected exact saved real warmup records 0 through 14.')
            context = SavedContext(body['context'])
            expected_covered, lag = 15, 0
        settings = yaml.safe_load((Path(__file__).parent / 'nas_config_cs3_luna.yaml').read_text())['optimizer']['llm']
        settings.update(max_usd=0.05, max_transport_retries=0, max_output_tokens=8192,
                        max_repair_attempts=1, extra_headers={})
        settings = normalize_llm_config({'type': 'llm_generator', 'llm': settings})['llm']
        memory_cfg = MemoryConfig.from_config({**dict(settings.memory), 'lag_trials': lag})
        report['memory_config'] = asdict(memory_cfg)
        if paid and not os.environ.get(settings.api_key_env):
            raise RuntimeError('API key is absent; no requests sent.')
        tracker = SpendTracker(output, settings)
        provider = build_provider(settings, spend_tracker=tracker)
        transport = provider._urlopen
        if response:
            replay = json.loads(Path(response).read_text())
            replay = json.loads(replay['content']) if 'content' in replay else replay
            report['response_sha256'] = hashlib.sha256(Path(response).read_bytes()).hexdigest()
        else:
            replay = {'updates': [dict(id='offline-fixture', observation='Offline lifecycle fixture.',
                conditions='Saved warmup evidence supplied unchanged.', evidence='Fifteen records supplied.',
                exceptions='No scientific inference.', uncertainty='Lifecycle validation only.',
                trial_numbers=list(range(75, 90)) if scenario else list(range(15)))],
                'retire': [{'id': f['id'], 'reason': 'Offline lifecycle fixture only.'} for f in initial_findings]}
        if not paid:
            report['offline_response_kind'] = 'saved' if response else 'synthetic lifecycle fixture'

        def bounded_transport(request, **kwargs):
            """Enforce two attempts and substitute response bytes in offline mode.

            Parameters
            ----------
            request : object
                Serialized provider request.
            **kwargs : dict
                Transport options.

            Returns
            -------
            object
                Live response or offline response byte stream.
            """
            if len(calls) >= 2:
                raise RuntimeError('Two-call memory smoke limit exceeded.')
            calls.append(1)
            print(f'Memory request {len(calls)}/2 ({"LIVE" if paid else "OFFLINE"})', flush=True)
            if paid:
                return transport(request, **kwargs)
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(replay)},
                'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1000,
                'completion_tokens': 100, 'total_tokens': 1100}}).encode())

        provider._urlopen = bounded_transport
        with patch('crest.optimizers.llm.memory.build_terminal_trial_history', return_value=records):
            memory = ExperimentalMemory(output / 'memory', memory_cfg)
            if scenario:
                scope = context.as_prompt_payload()
                scope['memory_policy'] = {'prompt_version': MEMORY_PROMPT_VERSION, 'lag_trials': lag}
                memory.state = {'schema_version': 1, 'scope_hash': digest(scope), 'version': 5,
                                'covered': {str(r['number']): digest(r) for r in records[:75]},
                                'findings': initial_findings, 'summary_failure_count': 0,
                                'summary_failure_reason': None}
                memory._merge(MemoryUpdate(updates=[], retire=[]), set(covered_numbers))
                memory._save_state(memory.state)
                (output / 'seed_memory.json').write_text(json.dumps(memory.state, indent=2) + '\n')
            enriched = memory.enrich(None, context, provider)
            if enriched.knowledge_base['covered_trial_count'] != expected_covered or not enriched.knowledge_base['summary'].strip():
                raise RuntimeError('Expected accepted nonempty summary covering the eligible trials.')
            if not list((output / 'memory' / 'snapshots').glob('*.json')):
                raise RuntimeError('Accepted snapshot missing.')
            if scenario:
                if [r['number'] for r in enriched.recent_trials] != list(range(90, 105)):
                    raise RuntimeError('Newest 15 records were not retained as raw history.')
                for path in (output / 'memory' / 'requests').glob('*.request.json'):
                    request = json.loads(path.read_text())
                    if request['metadata']['new_trial_numbers'] != list(range(75, 90)):
                        raise RuntimeError('Summary request crossed the lag boundary.')
            no_calls = FakeProvider([])
            restored = ExperimentalMemory(output / 'memory', memory_cfg).enrich(None, context, no_calls)
            if no_calls.requests or restored.knowledge_base != enriched.knowledge_base:
                raise RuntimeError('Reload changed memory or requested a provider.')
        accounting = json.loads(tracker.path.read_text())
        if not accounting['reservations'] or any(r['status'] != 'settled' for r in accounting['reservations'].values()):
            raise RuntimeError('Missing usable settled usage accounting.')
        report.update(passed=True, covered_trial_count=expected_covered, reload_provider_requests=0,
                      stored_json_chars=len(json.dumps(memory.state['findings'], ensure_ascii=False)),
                      summary_prose_chars=len(summary_text(memory.state['findings'])),
                      finding_count=len(memory.state['findings']),
                      summary=enriched.knowledge_base['summary'])
        return report
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report.update(http_requests=len(calls) if paid else 0, transport_attempts=len(calls))
        if tracker is not None and tracker.path.exists():
            report['spending'] = json.loads(tracker.path.read_text())
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Report: {output / "report.json"}', flush=True)
        if 'spending' in report:
            print(f'Charged/reserved USD: {report["spending"]["charged_usd"]}', flush=True)
        if report['passed']:
            print(f'Memory characters: JSON={report["stored_json_chars"]}, prose={report["summary_prose_chars"]}; limit=6000 each.', flush=True)
            print(f'PASS: nonempty memory covers {report["covered_trial_count"]} saved trials; snapshot and offline reload verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-request', type=Path)
    source.add_argument('--scenario', type=Path)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--run-paid', action='store_true')
    args = parser.parse_args()
    if args.run_paid:
        if args.response:
            parser.error('--response is offline-only')
        run(args.output, args.source_request, True, scenario=args.scenario)
    else:
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-not-a-key'}), patch(
                'socket.socket.connect', side_effect=AssertionError('Offline network forbidden')):
            run(args.output, args.source_request, response=args.response, scenario=args.scenario)
