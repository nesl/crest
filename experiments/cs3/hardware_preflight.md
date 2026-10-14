# CS3 bounded HIL preflight — manual commands

Prepared, not executed. This is one fixed baseline candidate (warm-up index 0),
one attempted trial, at most three training epochs if deployment is feasible,
int8 PTQ, 2000 ms cadence and ten measured inference runs. The fake provider
cannot call a paid API; memory is disabled. The smoke CLI skips final training.
It may still compile, program both board and harness, measure, and train on GPU.
An infeasible or failed first candidate is inconclusive, not permission to add
trials. `export_variant: untrained` is retained from the archived CS3 baseline.

## Verified setup and prerequisites

Read-only SSH on October 14 confirmed both normal repositories at `20d578d`,
the Python executables below, blinky's HIL entry point and tracked memory-pool
file, and GPU audio cache `train.npz`, `val.npz`, `test.npz`, `calibration.npz`
and `fold_rotation/`. Blinky's audio cache directory exists; its contents were
not checked. Both hosts already listen on loopback port **6001**. Blinky 6003
was free; GPU 6003 and current board ownership still require the checks below.
Remote source has not been updated or validated for this LLM preflight.

**Port 6003 does not isolate hardware.** Before server startup, confirm the
existing 6001 experiment and all other board/harness workers are idle and have
been deliberately stopped by their owner. Do not kill an unidentified process.
The source paths/environments are verified; installed toolchain paths below
come from the prior validated CS2 manual runbook and are checked before use.
This is not evidence of a new hardware success or an API test.

The campaign memory-pool path was updated from the obsolete, absent
`analysis_scripts/stm32_example_project/nucleo_mypool.json` to tracked
`src/config/stm32_nucleo_mypool.json`, matching the current CS3 case-study config.
This resolves a repository path rename, not a newly tuned scientific setting.

## 1. Stage reviewed source on both hosts

Run after the reviewed `codex/llm-generator-mvp` commit has been pushed. On
blinky use `/home/joe/Projects/tinyodom-ex`; on m202-GPU use
`/home/joseph/Projects/tinyodom-ex`. In each checkout:

```bash
git status --short
test -z "$(git status --porcelain)" && \
git fetch origin codex/llm-generator-mvp && \
git switch codex/llm-generator-mvp && \
git merge --ff-only origin/codex/llm-generator-mvp
git rev-parse HEAD
```

Stop if the clean check or any Git step fails. Preserve/reconcile existing local
work before retrying; these commands do not reset or stash it. Record the same
reviewed HEAD on both hosts before proceeding. Existing ignored datasets and
vendor inputs must remain in place. No package installation is included.

## 2. Prepare isolated configs and check resources

On **blinky**:

```bash
cd /home/joe/Projects/tinyodom-ex
/home/joe/miniforge3/envs/tinyodomex/bin/python experiments/cs3/prepare_hardware_preflight.py
ss -ltnp 'sport = :6003'
pgrep -af 'hil_server|nas_model_client|STM32_Programmer|ST-LINK|openocd|gdb' || true
fuser /dev/ttyACM0 /dev/ttyACM1
test -d sketches/stm32/crest_stm32_lrun
test -f src/config/stm32_nucleo_mypool.json
test -x /home/joe/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin/arm-none-eabi-gcc
test -x /opt/ST/STEdgeAI/4.0/Utilities/linux/stedgeai
test ! -e models/cs3_hil_preflight_v1
```

On **m202-GPU**:

```bash
cd /home/joseph/Projects/tinyodom-ex
/home/joseph/miniforge3/envs/tinyodomex/bin/python experiments/cs3/prepare_hardware_preflight.py
ss -ltnp 'sport = :6003'
nvidia-smi
test ! -e models/cs3_hil_preflight_v1
```

Require port 6003 free on both hosts, an available selected GPU, and exclusive
board/harness ownership. A clean `fuser` result alone does not establish that an
idle HIL server will not take the device later. Stop on a failed file/tool check.
The helper refuses existing generated configs; preserve prior evidence before
choosing a deliberately new run identity. Compare generated `warmup_first.json`
and configs on both hosts; role YAMLs differ only in `outputs.models_dir`.

## 3. HIL server on blinky

```bash
cd /home/joe/Projects/tinyodom-ex
mkdir -p logs
tmux new -s cs3-preflight-server
```

Inside tmux:

```bash
export PYTHONHASHSEED=0
export PATH="/home/joe/st/stm32cubeclt_1.21.0/GNU-tools-for-STM32/bin:/opt/ST/STEdgeAI/4.0/Utilities/linux:$PATH"
set -o pipefail
/home/joe/miniforge3/envs/tinyodomex/bin/python -u src/hil_server.py \
  --config experiments/cs3/preflight_v1/server.yaml \
  2>&1 | tee logs/cs3_preflight_server_v1.log
```

Confirm startup without errors, then detach with Ctrl-b, d.

## 4. Reverse tunnel from blinky to m202-GPU

On blinky, `tmux new -s cs3-preflight-tunnel`, then:

```bash
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:6003:127.0.0.1:6003 m202-GPU
```

This is the documented blinky-to-GPU reverse direction using the existing alias
and jump host. Stop on any forwarding/authentication failure. Detach Ctrl-b, d.
On GPU, `ss -ltnp 'sport = :6003'` must now show the tunnel listener. Authentication
and forwarding for this new 6003 tunnel have not been executed here.

## 5. One-attempt smoke client on m202-GPU

On GPU, `tmux new -s cs3-preflight-client`, then:

```bash
cd /home/joseph/Projects/tinyodom-ex
mkdir -p logs
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export LD_LIBRARY_PATH=/home/joseph/miniforge3/envs/tinyodomex/lib
unset OPENAI_API_KEY OPENROUTER_API_KEY
set -o pipefail
test ! -e models/cs3_hil_preflight_v1/client/CS3_audio_hil_preflight_v1_STM32_NUCLEO_N657X0_Q && \
/home/joseph/miniforge3/envs/tinyodomex/bin/python -u src/nas_model_client.py \
  --config experiments/cs3/preflight_v1/client.yaml \
  --study-name CS3_audio_hil_preflight_v1 --smoke-test 1 \
  2>&1 | tee logs/cs3_preflight_client_v1.log
printf 'Smoke exit status: %s\n' "$?"
```

GPU 0 is conditional on `nvidia-smi` availability. `PYTHONHASHSEED` alone does
not fix TensorFlow/NumPy initialization; this is an operational check, not a
reproducibility or scientific comparison. Do not omit `--smoke-test 1`: the
alternate CLI branch starts the full workflow. Repeating this command appends
another attempt; do not rerun it as a status check.

## Acceptance and cleanup

Require exactly one attempted trial, COMPLETE with finite trained macro-F1 and
score, provenance `proposal_source=llm_warmup` / `warmup_index=0`, accepted raw
cadenced measurement linked to that request, valid instrumentation, and no
deadline/integrity failures. Inspect actual recorded measurement count/retries;
ten inference runs are not ten NAS trials. A zero exit with PRUNED/FAIL is
inconclusive. No generated provider candidate, paid request, final model
training, fold-rotation campaign, or full NAS campaign should occur.

Client DB is under
`models/cs3_hil_preflight_v1/client/CS3_audio_hil_preflight_v1_STM32_NUCLEO_N657X0_Q/optuna_smoke_test.db`.
Keep the complete study directory and LLM ledger. Server evidence is under
`models/cs3_hil_preflight_v1/server/hil_requests/` and `hil_measurements/`.
Keep both console logs and generated YAML/first-candidate JSON.

After the client has finished and no measurement remains active, attach to
`cs3-preflight-server` and `cs3-preflight-tunnel` and press Ctrl-c in each.
Stop only these owned sessions. A successful preflight is not approval to run
the 200-feasible/300-attempt production configuration.
