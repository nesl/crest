# Configurable evaluation-order implementation contract

This document refines Plan 4 into the concrete source delta used by the
evaluation-order branch. It does not redefine the transport contract.

## Current limitation and reuse

`NASModelClient.objective` currently performs HIL before training. Its TFLite
host evaluator combines export, subprocess inference, diagnostic-sidecar
writing, and task-owned prediction evaluation, so a training-first caller would
otherwise export twice. The selected checkpoint path is shared across trials,
and Pareto replay can rebuild from model/checkpoint inputs but cannot replay a
saved exact flatbuffer.

The implementation reuses the existing exporter, prediction subprocess, task
evaluation, `TrialOutcome`, feasibility/scoring, logging, infrastructure guard,
TCP probe, and transport-owned request API. It does not introduce a workflow
engine, optimizer changes, new task/model interfaces, or a second exporter.

## Exact helper delta

- `_evaluate_tflite_path(...) -> EvaluationResult` owns the existing
  path-based prediction, diagnostics sidecar, and
  `task.evaluate_predictions(...)` tail. `_evaluate_model_with_backend` remains
  the legacy wrapper: it exports once, then calls this helper. Training-first
  code exports once with `_export_tflite_for_evaluation`, evaluates that path,
  hashes those exact bytes, and later supplies the same path to transport.
- `_snapshot_trial_checkpoint(trial_number) -> Path` copies the selected
  checkpoint immediately after fitting to a unique artifact path before any
  later trial can overwrite it. Reload and provenance use the snapshot.
- `_sha256_file(path) -> str` computes local flatbuffer identity only. Wire
  framing, request identity, retries, staging, and server evidence remain owned
  by transport.
- `_fail_with_penalty` is lifted only enough to accept already available task
  metrics and artifact summary. Defaults preserve the hardware-first empty
  case. It never fabricates latency after compile/allocation/runtime failure.
- Training-first valid measurements run feasibility after training and retain
  real task/hardware objective values, signed constraints, checkpoint,
  flatbuffer hash, and artifact summary even when infeasible.
  `train_if_infeasible` continues to control only not-yet-run training in the
  hardware-first path.
- `require_replayable_source_order(source_config)` runs before replay output,
  server creation, or candidate execution. Missing historical `nas.hil_first`
  means `true`; explicit `false` with `training.train: true` raises a clear
  exact-flatbuffer replay limitation. Explicit row provenance for a trained
  artifact also rejects config overrides that would otherwise hide its source
  order. Existing hardware-first and no-training hardware-only replay remain
  unchanged.

## Transport integration boundary

After the reviewed transport implementation lands, training-first uses its
exact API only:

- `_ensure_hil_artifact_support(*, refresh: bool = False) -> dict`
- `_hil_request(payload, *, artifact_path: Path | None = None) -> dict`

Each `run_nas` and smoke invocation that will perform remote training-first HIL
does the existing TCP endpoint probe and a capability refresh before any
`model.fit`, including resumed invocations. Desktop-only runs remain probe-free.
The objective does not duplicate socket recovery, multipart framing, or
capability caching.

## Defaults, migration, and validation

`nas.hil_first` defaults to `true`, so existing configurations and historical
study identity preserve the current stage order. `false` with training and
remote compile/HIL selects train, checkpoint snapshot, one export, host
evaluation, exact-artifact HIL, then measured constraints/scoring. No-training
mode retains its supported hardware-only behavior and does not invent a
checkpoint. A backend without artifact support fails before training.

Focused tests cover the unchanged default order and failure behavior, one
export with preserved `EvaluationResult`, unique checkpoint/artifact identity,
valid measured infeasibility versus hardware/infrastructure failure, retained
quality/artifacts, three-strike integration, startup probe-before-fit,
desktop-only probe omission, and training-first replay rejection without output
side effects. No full training, NAS campaign, timing run, or hardware operation
is part of these tests.
