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
from addict import Dict
from crest.model_families.audio_dscnn import AudioDSCNNFamily
from crest.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext, TargetSpec, TaskMetricContract
from crest.semantic_context import build_semantic_context

from crest.optimizers.llm.config import normalize_llm_config
from crest.optimizers.llm.history import build_recent_trial_history, build_best_trial_anchors
from crest.optimizers.llm.ledger import LLMLedger
from crest.optimizers.llm.memory import ExperimentalMemory, MemoryConfig
from crest.optimizers.llm.prompt_builder import PromptContext, build_candidate_request
from crest.optimizers.llm.provider import FakeProvider, build_provider
from crest.optimizers.llm.schemas import CandidateBatch, validate_candidate_batch
from crest.optimizers.llm.spending import SpendTracker
from crest.search_space import build_search_space_descriptor


def run(output: Path, paid: bool, render_only: bool = False) -> None:
    """Check memory and candidate responses without constructing or evaluating models.

    Parameters
    ----------
    output : pathlib.Path
        New directory for smoke-only prompts, responses, memory, and accounting.
    paid : bool
        Explicitly enable real API requests; false uses mocked HTTP responses.

    render_only : bool, optional
        Save complete prompts with FakeProvider, without initializing transport.

    Raises
    ------
    RuntimeError
        Credentials, response validation, usage, or call limits fail.
    FileExistsError
        Output directory exists; prevents accidental repeated paid execution.
    """
    base = Path(__file__).resolve().parent
    if paid and render_only:
        raise ValueError('render_only and paid are mutually exclusive.')
    cfg = Dict(yaml.safe_load((base / 'nas_config_cs3_luna.yaml').read_text()))
    llm = cfg['optimizer']['llm']
    llm.update(max_usd=0.05, max_transport_retries=0, max_repair_attempts=0,
               timeout_s=90, extra_headers={})
    llm = normalize_llm_config({'type': 'llm_generator', 'llm': llm})['llm']
    if paid and not os.environ.get(llm.api_key_env):
        raise RuntimeError('OPENAI_API_KEY is not set; no requests sent.')
    output.mkdir(parents=True, exist_ok=False)
    provenance = json.loads((base / 'baseline_provenance.json').read_text())
    shape = tuple(provenance['descriptor_validation']['input_shape'])
    # Reconstructed offline proxy from archived cache config; never load datasets.
    metadata = dict(feature_kind='log_mel_power', sample_rate_hz=16000, mel_bins=64,
                    expected_frames=shape[0], window_ms=25, hop_ms=10)
    target = TargetSpec(task_type='classification', output_names=['class_logits'],
        output_shapes=[(10,)], metadata=dict(num_classes=10, label_encoding='class_index', from_logits=True))
    bundle = DatasetBundle(train=DataSplit(inputs=None, targets=None), input_shape=shape,
                           input_dtype='float32', metadata=metadata)
    model = ModelBuildContext(input_shape=shape, input_dtype='float32', target_spec=target,
                              dataset_metadata=metadata)
    descriptor = build_search_space_descriptor(AudioDSCNNFamily(), model, cfg.model, cfg,
                                                collect_compile_metrics=True)
    semantic = build_semantic_context(enabled=True, dataset_name=cfg.dataset.name,
        dataset_config=cfg.dataset, dataset_bundle=bundle, model_build_context=model,
        task_name=cfg.task.name, target_spec=target,
        metric_contract=TaskMetricContract(available_metric_names={'loss', 'accuracy', 'macro_f1'},
                                          primary_metric_names={'accuracy', 'macro_f1'}),
        score_config=cfg.nas.score, feasibility_config=cfg.nas.feasibility,
        model_family_name=cfg.model.family, descriptor=descriptor, device_config=cfg.device,
        training_config=cfg.training, collect_compile_metrics=True)
    warmup = json.loads((base / 'warmup_params.json').read_text())
    # Deliberately synthetic scores and repeated warmup fixtures reconstruct 30 trials.
    # Repetitions exercise memory cadence only; no historical outcomes or models are used.
    trials = [SimpleNamespace(number=i, state='COMPLETE', params=params,
              values=[i / 100], user_attrs={'synthetic_smoke_only': True})
              for i, params in enumerate(warmup * 2)]
    study = SimpleNamespace(study_name='cs3_api_smoke_synthetic', trials=trials, directions=('maximize',))
    context = PromptContext(study_name=study.study_name, model_family='audio_dscnn',
        descriptor=descriptor, objective_summary='API smoke ONLY: scores are synthetic arbitrary numbers, not measured model quality. Trial states are synthetic too. Dataset metadata is a reconstructed offline proxy; no dataset loaded. Do not infer scientific findings. Campaign score: ' + json.dumps(cfg.nas.score, sort_keys=True),
        attempted_trials=30, feasible_completed_trials=30, target_feasible_trials=200,
        max_total_attempts=300, batch_size=5, require_rationale=llm.require_rationale,
        recent_trials=build_recent_trial_history(study, window_size=15),
        anchors=build_best_trial_anchors(study, anchor_count=5), semantic_context=semantic)
    if render_only:
        provider = FakeProvider([{'updates': [], 'retire': []}, {'candidates': []}])
        memory = ExperimentalMemory(output / 'memory', MemoryConfig.from_config(llm.memory))
        context = memory.enrich(study, context, provider)
        provider.complete_json(build_candidate_request(context, prompt_version=llm.prompt_version))
        if len(provider.requests) != 2:
            raise RuntimeError('Expected one memory request and one candidate request.')
        ledger = LLMLedger(output)
        for request in provider.requests:
            ledger.write_request(ledger.next_request_id(), request)
        report = dict(mode='render_only', synthetic_history=True, http_requests=0,
            hardware_or_training=False, dataset_metadata='Reconstructed offline proxy; no dataset loaded.',
            memory_result='Fake empty accepted summary; live summaries may populate findings and change the candidate prompt.',
            requests=[dict(purpose=purpose, messages=request.messages(),
                           prompt_version=request.prompt_version, metadata=request.metadata)
                      for purpose, request in zip(('memory', 'candidate'), provider.requests)])
        (output / 'prompts.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Rendered complete prompts: {output / "prompts.json"}', flush=True)
        return
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
    candidate_envelopes = ([{'params': params, 'rationale':
        'Synthetic offline fixture: exercise a distinct architecture; no model quality claim.'}
        for params in candidates] if llm.require_rationale else candidates)
    offline_responses = iter([{'updates': [], 'retire': []}, {'candidates': candidate_envelopes}])
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
            descriptor, batch_size=5, completed_candidates=warmup,
            require_rationale=llm.require_rationale)
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run-paid', action='store_true')
    mode.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        run(args.output, False, render_only=True)
    elif args.run_paid:
        run(args.output, True)
    else:
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-not-a-key'}), \
                patch('socket.socket.connect', side_effect=AssertionError('Network forbidden in offline mode')):
            run(args.output, False)
