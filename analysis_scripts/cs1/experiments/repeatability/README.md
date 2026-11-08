<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Paired repeatability experiments

Run paired STM32 measurements with a fixed randomized schedule and reusable compiled models.
This recipe initializes model weights.
It does not train models.

Run from the repository root.

```sh
python -m analysis_scripts.cs1.experiments.repeatability prepare --manifest /path/candidates.json --output /path/new-study --rounds 4 --seed 17
python -m analysis_scripts.cs1.experiments.repeatability build-artifacts --study /path/new-study --config /path/device-and-data.yaml
python -m analysis_scripts.cs1.experiments.repeatability run --study /path/new-study --dut-port /path/dut --harness-port /path/harness --board-ready
```

The caller supplies all candidate identities, architectures, clocks, input dimensions, cadence, run count, and initialization seed.
This small manifest uses the supported `odom_tcn` architecture schema.

```json
{
  "prospective_protocol": "rtc_wall_v4_dwt_active_stop_evidence_awake_tail",
  "timesteps": 16,
  "input_dim": 3,
  "period_ms": 250,
  "measured_runs": 4,
  "initialization_seed": 7,
  "candidates": [
    {"candidate_id": "a", "architecture": {"nb_filters": 8, "kernel_size": 3, "dropout_rate": 0.0, "use_skip_connections": true, "norm_flag": false, "dilations": [1, 2]}, "clock_mhz": 200},
    {"candidate_id": "b", "architecture": {"nb_filters": 8, "kernel_size": 3, "dropout_rate": 0.0, "use_skip_connections": true, "norm_flag": false, "dilations": [1, 2]}, "clock_mhz": 400}
  ],
  "pairs": [{"pair_id": "pair", "candidate_a": "a", "candidate_b": "b"}]
}
```

Architecture keys are defined by [OdomTCNFamily.validate_hparams](../../../../src/crest/model_families/odom_tcn.py).
Use a configuration selecting that family, such as the [STM32 OxIOD configuration](../../../../src/config/nas_config_stm32.yaml), adapted to your calibration data, dimensions, project and toolchain paths.
The [configuration guide](../../../../src/config/README.md) defines the accepted YAML settings.
Clocks and cadence must also suit your board and workload.
These values do not establish hardware feasibility.

Each candidate belongs to exactly one pair.
Each round measures both continuous and cadenced phases for every candidate.
AB/BA order is balanced across even round counts, with one randomized extra order for odd counts.
`--max-attempts` bounds retries across invocations.
`run --max-evaluations` bounds work per invocation.
Accepted evaluations are skipped on restart.
Exceptions and uncertain interruptions require inspection before resuming.
Evidence is retained.

`build-artifacts` initializes weights, quantizes models to int8, and compiles each measurement phase.
It requires the configured calibration dataset, CREST runtime, firmware template, ST Edge AI, and ARM/STM32 toolchain.
To use existing prepared bundles, run `bind-artifacts --artifact-root /path/bundles` before starting measurements.

Keep prepared bundles at their original paths and leave their inputs unchanged.
Rebuild after moving them or changing the model, configuration, or toolchain.

Run only one experiment at a time on a DUT/harness.
Harness firmware must already match the manifest's run count.
This runner disables automatic harness flashing.
Diagnostic profiling is outside this protocol.

Find compiled bundles under `artifacts/<candidate>/` and measurements, results, and logs under `evaluations/<round>/<candidate>/attempt-*/`.
This recipe has no analysis subcommand.
To use [CS1 repeatability analysis](../../README.md), first convert the measurements to its documented CSV inputs.
It does not read this study directory directly.

Safe checks

```sh
python -m analysis_scripts.cs1.experiments.repeatability --help
python -m pytest -q analysis_scripts/cs1/experiments/repeatability/test_runner.py
```
