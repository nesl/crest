<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# CS1 waiting-policy experiments

Compare prepared STM32 continuous, cadenced awake-tail, and cadenced STOP-tail deployments in randomized blocks.
Preparation and planning do not access hardware.
Supply separately built images with the appropriate `rtc_wall_v4_dwt_active_stop_evidence_awake_tail` or `rtc_wall_v4_dwt_active_stop_evidence_stop_tail` protocol.
This recipe does not build or modify firmware.

```sh
python -m analysis_scripts.cs1.experiments.waiting_policy.runner prepare --manifest /path/prepared.json --output /path/new-output
python -m analysis_scripts.cs1.experiments.waiting_policy.runner plan --study /path/new-output --rounds 5 --seed 42
python -m analysis_scripts.cs1.experiments.waiting_policy.runner run --study /path/new-output --rounds 5 --seed 42 --device-config /path/device.json --board-ready
```

The JSON manifest contains `schema_version: 1`, `variants: [...]` and `blocks: [["continuous", "awake", "stop"]]`.
Blocks partition variant names exactly once.
Each variant has `name`, `candidate`, `phase` (`back_to_back` or `cadenced`), `policy` (`awake` or `stop`), `runs`, `clock_mhz`, `period_ms`, native `device_options`, `window_size`, `num_channels`, `project`, `phase_header` (relative to project), `phase_header_sha256`, `app_elf_sha256`, and the native successful `compile_result` dictionary.
Project and compile-result paths resolve relative to the manifest.
Use 10 measured runs, 5000 us wake margin and minimum sleep for this protocol.
Continuous uses the awake image.
The phase header cadence is still checked.

For `compile_result`, preserve the successful [CompileResult](../../../../src/crest/devices.py) returned by the build, serialized as a dictionary with paths converted to strings.
See [repeatability preparation](../repeatability/experiment.py) for an example.
Use the [STM32 backend options](../../../../src/crest/microcontrollers/stm32_nucleo_n657x0.py) for `device_options`.

Keep a separate, complete prepared project for each condition at its original build location, including binaries, weights, storage metadata, and external loader.
Leave these projects unchanged between preparation and measurement.

A minimal device JSON uses native defaults for omitted pins/timeouts.

```json
{
  "device_id": "lab-device",
  "runtime_kwargs": {
    "serial_port": "/dev/ttyACM0",
    "harness_serial_port": "/dev/ttyACM1",
    "baud_rate": 115200
  }
}
```

Replace the ports and identity for your setup.
The [STM32 backend](../../../../src/crest/microcontrollers/stm32_nucleo_n657x0.py) documents optional wiring and timeout settings.

Install matching harness firmware before running.
Automatic harness flashing is disabled.
Run only one experiment at a time on a DUT/harness.

Restart with the original schedule and device settings to skip accepted measurements.
Failures preserve evidence and stop without automatic retries.
Inspect an interrupted or failed run before starting a new output directory.
Results are saved in `measurements/<id>/result.json`, with raw records and logs under `attempt-01/`.
This recipe has no analysis subcommand.
[CS1 policy analysis](../../README.md) requires its documented input format and does not read this study directory directly.
