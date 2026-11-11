# Training batch and evaluation identity contract

## Limitation and reuse decision

The NAS objective and final-training runner previously hardcoded a batch size of
256. Task constructors do not receive runner schedule settings, and `FitPlan`
intentionally contains task-owned data wiring rather than epochs or batch size.
Passing a second editable task setting could diverge from the runner, while a
new constructor or `FitPlan` field would break or enlarge shared interfaces.

CREST therefore keeps one editable `training.batch_size` setting. The runner
derives a reserved task-config entry from it and uses the existing task methods
and fit kwargs. No `TaskABC`, `FitPlan`, task constructor, or model-family
signature changes are made.

## Exact delta

- `training.batch_size` is a non-boolean positive integer. Its default is 256.
- `nas.hil_first` is a boolean. Its default is `true`.
- `runtime_training_batch_size` is reserved. A user-provided value under
  `task.params` is rejected, even if it matches `training.batch_size`.
- `resolve_component_selection(config)` copies `task.params`, adds the derived
  reserved entry to that copy, and leaves the loaded source config unchanged.
  This keeps bootstrap and legacy selection consumers idempotent.
- Both production `build_fit_plan` callers use the same fit-kwargs helper.
  Array-backed inputs receive Keras `batch_size`; `tf.data.Dataset` inputs omit
  that ineffective argument because their task-owned pipeline is already
  batched using the derived config value.
- Runtime metadata records the resolved effective training batch size.
- Final training writes an adjacent `.training.json` provenance sidecar with
  the effective batch size, whether the fit input was already batched, the
  configured epoch cap, and the final-training mode. Training history and
  shared metric record shapes remain unchanged.
- The existing unconditional optimizer study signature adds
  `evaluation_policy: {hil_first, training_batch_size}`. The
  `build_optimizer_signature` helper accepts that payload as an optional
  argument so direct legacy callers remain source-compatible. Resume comparison
  treats a missing historical payload as `{hil_first: true,
  training_batch_size: 256}` without rewriting stored metadata. All other
  signature fields remain strict.

## Callers, defaults, and migration

The production fit-plan callers are `NASModelClient.objective` and
`NASModelClient._train_with_decoded_hparams`. Dataset/task/model bootstrap and
the HIL server consume the derived task-config copy but do not gain a training
API. Existing configuration files need no edits and resolve to `true`/256.
The FER HCI configuration selects `false`/64. A resumed historical study is
compatible only with the legacy defaults; changing either setting requires a
new study.

## Validation

Tests cover defaults and explicit values, invalid boolean/string/non-positive
batch sizes, invalid non-boolean order values, rejection of the reserved task
key, source-config immutability, derived task-config propagation, array and
prebatched-dataset fit kwargs in both training entry points, runtime metadata,
historical signature normalization, strict changed-policy rejection, and
feasibility-enabled and feasibility-disabled resumes.
