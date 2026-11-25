# Run a first study

Start with the OxIOD FLOPs example. It searches for odometry models with low validation error (`rmse_total`) and low compute cost (`flops`). The shipped YAML disables both hardware-in-the-loop (HIL) evaluation and compilation.

## Run three attempts

[Prepare OxIOD and activate the environment](setup.md), then run this command from the repository root.

```bash
python src/nas_model_client.py \
  --config src/config/nas_config_flops_rmse.yaml \
  --smoke-test 3 \
  --study-name oxiod_proxy_smoke
```

The CLI trains for up to three epochs per attempt and appends the configured device name to the supplied study name. Expect a database at `models/oxiod_proxy_smoke_STM32_NUCLEO_N657X0_Q/optuna_smoke_test.db` and a trial log in the same directory. Check the log for a completed trial with finite RMSE and FLOPs before increasing the budget. Give each independent smoke run a fresh name because reusing a name appends trials.

## Set a full search budget

Copy the example YAML before editing it. Review `training.nas_epochs` for the per-candidate training budget, `training.nas_trials` for the desired number of feasible completions, and `training.max_total_trials` for the total attempt cap. Invoke the same client command with your YAML and a new study name, omitting `--smoke-test`.

For a memory-traffic objective, start from the [OxIOD memory proxy example](https://github.com/nesl/crest/blob/main/src/config/nas_config_memory_proxy.yaml). To adapt an audio YAML for desktop use, set `device.hil: false`, `device.compile_when_hil_disabled: false`, and `training.energy_aware: false`. Use task metrics such as `macro_f1` and review score, pruning, and feasibility rules for dependencies on measured hardware metrics. [Configuration reference](https://github.com/nesl/crest/blob/main/src/config/README.md).

[Read the resulting study artifacts](results.md).
