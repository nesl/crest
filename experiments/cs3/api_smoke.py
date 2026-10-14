#!/usr/bin/env python3
"""Two-purpose API-only smoke; offline transport check unless --run-paid is set."""
import argparse
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.history import build_recent_trial_history, build_best_trial_anchors
from crest.optimizers.llm.ledger import LLMLedger
from crest.optimizers.llm.memory import ExperimentalMemory, MemoryConfig
from crest.optimizers.llm.prompt_builder import PromptContext, build_candidate_request
from crest.optimizers.llm.provider import build_provider
from crest.optimizers.llm.schemas import CandidateBatch, validate_candidate_batch
from crest.optimizers.llm.spending import SpendTracker
from crest.search_space import SearchParam, SearchSpaceDescriptor


def run(output: Path, paid: bool) -> None:
    """Check memory and candidate responses without constructing or evaluating models.

    Parameters
    ----------
    output : pathlib.Path
        New directory for smoke-only prompts, responses, memory, and accounting.
    paid : bool
        Explicitly enable real API requests; false uses mocked HTTP responses.

    Raises
    ------
    RuntimeError
        Credentials, response validation, usage, or call limits fail.
    FileExistsError
        Output directory exists; prevents accidental repeated paid execution.
    """
    base = Path(__file__).resolve().parent
    cfg = yaml.safe_load((base / 'nas_config_cs3_luna.yaml').read_text())
    llm = cfg['optimizer']['llm']
    llm.update(max_usd=0.05, max_transport_retries=0, max_repair_attempts=0,
               timeout_s=90, extra_headers={})
    llm = normalize_llm_config({'type': 'llm_generator', 'llm': llm})['llm']
    if paid and not os.environ.get(llm.api_key_env):
        raise RuntimeError('OPENAI_API_KEY is not set; no requests sent.')
    output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((base / 'baseline_provenance.json').read_text())['descriptor_validation']['parameters']
    descriptor = SearchSpaceDescriptor([SearchParam(**{k: tuple(v) if isinstance(v, list) else v for k, v in p.items()}) for p in raw])
    warmup = json.loads((base / 'warmup_params.json').read_text())
    # Deliberately synthetic scores; do not read historical outcomes or evaluate models.
    trials = [SimpleNamespace(number=i, state='COMPLETE', params=params,
              values=[i / 100], user_attrs={'synthetic_smoke_only': True})
              for i, params in enumerate(warmup)]
    study = SimpleNamespace(study_name='cs3_api_smoke_synthetic', trials=trials, directions=('maximize',))
    context = PromptContext(study_name=study.study_name, model_family='audio_dscnn',
        descriptor=descriptor, objective_summary='API smoke ONLY: scores are synthetic arbitrary numbers, not measured model quality. Do not infer scientific findings. Propose five distinct new valid configurations.',
        attempted_trials=15, feasible_completed_trials=15, target_feasible_trials=200,
        max_total_attempts=300, batch_size=5,
        recent_trials=build_recent_trial_history(study, window_size=15),
        anchors=build_best_trial_anchors(study, anchor_count=5))
    tracker = SpendTracker(output, llm)
    provider = build_provider(llm, spend_tracker=tracker)
    ledger = LLMLedger(output)
    memory = ExperimentalMemory(output / 'memory', MemoryConfig.from_config(llm.memory))
    # Offline check still exercises production HTTP serialization/accounting.
    rng = random.Random(17)
    candidates = []
    while len(candidates) < 5:
        candidate = {p.name: rng.choice(p.choices) for p in descriptor.params}
        if candidate not in warmup and candidate not in candidates:
            candidates.append(candidate)
    offline_responses = iter([{'updates': [], 'retire': []}, {'candidates': candidates}])
    transport = provider._urlopen
    calls = []

    def bounded_transport(request, **kwargs):
        """Enforce the independent HTTP call count and substitute offline responses.

        Parameters
        ----------
        request : urllib.request.Request
            Production request, never logged with credential headers.
        **kwargs : dict
            Transport timeout settings.

        Returns
        -------
        object
            Context-managed HTTP response or offline byte stream.
        """
        if len(calls) >= 3:
            raise RuntimeError('Three-request smoke limit reached.')
        calls.append(1)
        print(f'Request {len(calls)}/3 ({"LIVE" if paid else "OFFLINE"})', flush=True)
        if paid:
            return transport(request, **kwargs)
        body = {'choices': [{'message': {'content': json.dumps(next(offline_responses))},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1000, 'completion_tokens': 100, 'total_tokens': 1100}}
        return io.BytesIO(json.dumps(body).encode())

    provider._urlopen = bounded_transport
    report = {'mode': 'paid' if paid else 'offline', 'passed': False,
              'model': llm.model, 'max_usd': 0.05, 'max_http_requests': 3,
              'synthetic_history': True, 'hardware_or_training': False}
    try:
        context = memory.enrich(study, context, provider)
        if context.knowledge_base['covered_trial_count'] != 15:
            raise RuntimeError('Memory smoke failed validation; stopping before candidate generation.')
        request = build_candidate_request(context, prompt_version=llm.prompt_version)
        request_id = ledger.next_request_id()
        ledger.write_request(request_id, request)
        response = provider.complete_json(request)
        ledger.write_response(request_id, response)
        if response.finish_reason != 'stop':
            raise RuntimeError(f'Candidate response incomplete: {response.finish_reason}')
        result = validate_candidate_batch(CandidateBatch.model_validate_json(response.content),
            descriptor, batch_size=5, completed_candidates=warmup)
        report['candidate_validation'] = asdict(result)
        if len(result.accepted) != 5 or result.rejected:
            raise RuntimeError('Expected five valid, distinct, new candidates; inspect report.json.')
        # A successful smoke also requires usable usage accounting from both purposes.
        state = json.loads(tracker.path.read_text())
        if any(r.get('status') != 'settled' for r in state['reservations'].values()):
            raise RuntimeError('Some requests lack valid usage accounting; inspect spend_state.json.')
        report['passed'] = True
    finally:
        report['http_requests'] = len(calls)
        report['spending'] = json.loads(tracker.path.read_text())
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Report: {output / "report.json"}', flush=True)
        print(f'Charged/reserved USD: {report["spending"]["charged_usd"]}', flush=True)
    print('PASS: memory update and five valid candidates; no model evaluated.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-paid', action='store_true')
    args = parser.parse_args()
    if args.run_paid:
        run(args.output, True)
    else:
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-not-a-key'}), \
                patch('socket.socket.connect', side_effect=AssertionError('Network forbidden in offline mode')):
            run(args.output, False)
