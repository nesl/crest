#!/usr/bin/env python3
"""Offline CS3 prompt sizing; historical records are proxies, not campaign inputs."""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
import yaml
from addict import Dict
from optuna.distributions import json_to_distribution
from crest.model_families.audio_dscnn import AudioDSCNNFamily
from crest.pipeline_types import DataSplit, DatasetBundle, ModelBuildContext, TargetSpec, TaskMetricContract
from crest.search_space import build_search_space_descriptor
from crest.semantic_context import build_semantic_context
from crest.optimizers.llm.history import build_recent_trial_history, build_best_trial_anchors, build_terminal_trial_history
from crest.optimizers.llm.memory import SUMMARY_SYSTEM, summary_text
from crest.optimizers.llm.prompt_builder import PromptContext, build_candidate_request
from crest.optimizers.llm.provider import LLMRequest


def measure(request):
    """Return heuristic tokens and the production conservative reservation.

    Parameters
    ----------
    request : LLMRequest
        Production-shaped text request.

    Returns
    -------
    dict
        Character count, characters/4 estimate and byte-based tracker guard.
    """
    chars = len(request.system_prompt) + len(request.user_prompt)
    size = len(json.dumps(request.messages(), separators=(',', ':'), allow_nan=False).encode())
    return dict(text_characters=chars, heuristic_input_tokens=math.ceil(chars/4), serialized_message_bytes=size, tracker_input_token_bound=size+1024)


def render(source, config):
    """Render real helper payloads using readonly archived trial proxies.

    Parameters
    ----------
    source : Path
        Local archived baseline directory.
    config : Path
        Proposed CS3 configuration.

    Returns
    -------
    dict
        Count samples and disclosed estimation limitations.
    """
    cfg = Dict(yaml.safe_load(config.read_text()))
    db = sqlite3.connect((source/'optuna.db').resolve().as_uri()+'?mode=ro', uri=True)
    trials=[]
    for tid,number,state in db.execute('SELECT trial_id,number,state FROM trials ORDER BY number'):
        params={k:json_to_distribution(d).to_external_repr(v) for k,v,d in db.execute('SELECT param_name,param_value,distribution_json FROM trial_params WHERE trial_id=?',(tid,))}
        attrs={k:json.loads(v) for k,v in db.execute('SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?',(tid,))}
        values=[r[0] for r in db.execute('SELECT value FROM trial_values WHERE trial_id=? ORDER BY objective',(tid,))]
        trials.append(SimpleNamespace(number=number,state=state,params=params,user_attrs=attrs,values=values or None))
    shape=tuple(trials[0].user_attrs['hyperparameters'][k] for k in ('timesteps','input_dim'))
    target=TargetSpec(task_type='classification',output_names=['class_logits'],output_shapes=[(10,)],metadata=dict(num_classes=10,label_encoding='class_index',from_logits=True))
    metadata=dict(feature_kind='log_mel_power',sample_rate_hz=16000,mel_bins=64,expected_frames=shape[0],window_ms=25,hop_ms=10)
    bundle=DatasetBundle(train=DataSplit(inputs=None,targets=None),input_shape=shape,input_dtype='float32',metadata=metadata)
    model=ModelBuildContext(input_shape=shape,input_dtype='float32',target_spec=target,dataset_metadata=metadata)
    descriptor=build_search_space_descriptor(AudioDSCNNFamily(),model,cfg.model,cfg,collect_compile_metrics=True)
    semantic=build_semantic_context(enabled=True,dataset_name=cfg.dataset.name,dataset_config=cfg.dataset,dataset_bundle=bundle,model_build_context=model,task_name=cfg.task.name,target_spec=target,metric_contract=TaskMetricContract(available_metric_names={'loss','accuracy','macro_f1'},primary_metric_names={'accuracy','macro_f1'}),score_config=cfg.nas.score,feasibility_config=cfg.nas.feasibility,model_family_name=cfg.model.family,descriptor=descriptor,device_config=cfg.device,training_config=cfg.training,collect_compile_metrics=True)
    findings=[dict(id=f'proxy-{i}',observation='SYNTHETIC SIZING ONLY. '+('Archived proxy requires validation. '*3),conditions='Archived AudioDSCNN campaign.',evidence='No new measured evidence.',exceptions='Confounded settings and failures.',uncertainty='Not scientific findings.',trial_numbers=[i]) for i in range(12)]
    while len(json.dumps(findings,ensure_ascii=False))<5999:
        findings[0]['observation']+='x'
    samples=[]
    for n in (15,30,100,190):
        history=trials[:n]
        recent=build_recent_trial_history(history,window_size=15,directions=('maximize',))
        anchors=build_best_trial_anchors(history,anchor_count=5,feasibility_enabled=True,directions=('maximize',))
        records=build_terminal_trial_history(history,directions=('maximize',))
        kb=dict(version=n//15,covered_trial_count=n,pending_trial_count=0,omitted_pending_trial_count=0,summary_status='active',summary_failure_count=0,summary_failure_reason=None,summary=summary_text(findings) if n>15 else '')
        context=PromptContext(study_name='UrbanSound8K_DSCNN_STM32_AUDIO_case3_llm_t1',model_family='audio_dscnn',descriptor=descriptor,objective_summary=json.dumps(cfg.nas.score,sort_keys=True,default=str),attempted_trials=n,feasible_completed_trials=sum(r.get('feasibility_status')=='feasible' for r in records),target_feasible_trials=200,max_total_attempts=300,batch_size=5,recent_trials=recent,anchors=anchors,knowledge_base=kb,semantic_context=semantic)
        scope={k:v for k,v in context.as_prompt_payload().items() if k not in {'trial_budget','recent_trials','pending_trials','anchors','knowledge_base','output_contract'}}
        body=dict(context=scope,previous_findings=findings if n>15 else [],previously_covered_trial_count=n-15,new_trials=list(records[-15:]),limits=dict(max_findings=12,max_chars=6000))
        summary=LLMRequest(SUMMARY_SYSTEM,json.dumps(body,sort_keys=True,separators=(',',':')),'memory-v1')
        samples.append(dict(attempted_trials=n,case='ordinary',recent_records=len(recent),anchor_records=len(anchors),generation=measure(build_candidate_request(context,prompt_version='v1')),summary=measure(summary)))
        if n==190:
            from dataclasses import replace
            # Match memory.enrich: cap newest pending records outside recent history.
            pending_outside_recent = records[-115:-15]
            visible_pending = pending_outside_recent[-cfg.optimizer.llm.memory.max_pending_trials:]
            omitted = len(pending_outside_recent) - len(visible_pending)
            backlog=replace(context,pending_trials=tuple(visible_pending),knowledge_base={**kb,
                'covered_trial_count':n-115,'summary_status':'suspended',
                'pending_trial_count':115,'omitted_pending_trial_count':omitted,
                'summary_failure_count':3})
            counts = measure(build_candidate_request(backlog,prompt_version='v1'))
            samples.append(dict(attempted_trials=n,case='suspended_memory_backlog',
                pending_records=len(visible_pending),omitted_pending_records=omitted,
                configured_input_allowance=cfg.optimizer.llm.max_input_tokens,
                fits_input_guard=counts['tracker_input_token_bound'] <= cfg.optimizer.llm.max_input_tokens,
                generation=counts))
    db.close()
    return dict(method='OFFLINE ceil(system+user characters/4), approximate not tokenizer counts',tokenizer='tiktoken absent in project environment; no downloads attempted',history='Read-only archived baseline proxy only; never injected into live campaign',memory='Synthetic near-6000-character structured findings; not measured scientific evidence',memory_structured_characters=len(json.dumps(findings,ensure_ascii=False)),memory_summary_characters=len(summary_text(findings)),baseline_trial_count=len(trials),input_shape=shape,samples=samples,limitations=['Summary schema mirrors ExperimentalMemory.enrich; no provider called.','15-trial empty-memory generation is a lower sizing case; actual first accepted summary may populate memory.','100/190 snapshots are not necessarily scheduled summary boundaries.','Semantic metadata reconstructed from archived log-mel config/shape.','Byte+1024 guard is reservation, not billable tokens; completion/reasoning outputs require assumptions.'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    # Reuse the existing pure arithmetic; warm-up is excluded from generation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analysis_scripts.llm_token_cost.estimate import project
    report = render(a.source_dir, a.config)
    cfg = yaml.safe_load(a.config.read_text())["optimizer"]["llm"]
    ordinary = [sample for sample in report["samples"] if sample["case"] == "ordinary"]
    report["configured_prices"] = {key: cfg[key] for key in (
        "input_usd_per_million_tokens", "output_usd_per_million_tokens", "pricing_source")}
    report["projections"] = []
    for attempts in (200, 207, 300):
        for output in (1024, 4096, 8192):
            parts = {}
            for kind, trials, batch in (("generation", attempts-15, 5), ("summary", (attempts-1)//15, 1)):
                parts[kind] = project(
                    [row[kind]["heuristic_input_tokens"] for row in ordinary],
                    trials=trials, batch_size=batch, output_tokens_per_call=output,
                    input_price=cfg["input_usd_per_million_tokens"],
                    output_price=cfg["output_usd_per_million_tokens"])
            costs = [sum(parts[kind]["scenarios"][i]["total_cost_usd"] for kind in parts) for i in range(3)]
            report["projections"].append(dict(total_attempts=attempts, assumed_output_tokens_per_call=output,
                generation_calls=parts["generation"]["expected_api_calls"], summary_calls=parts["summary"]["expected_api_calls"],
                combined_cost_usd=dict(zip(("sample_min", "sample_mean", "sample_max"), costs))))
    a.output.write_text(json.dumps(report, indent=2)+'\n')
