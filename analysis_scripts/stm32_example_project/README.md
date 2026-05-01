# STM32 Example Project

This package contains the current STM32N6 example assets for the
`NUCLEO-N657X0-Q` board inside `tinyodom-ex`.

It started as a fresh STM32CubeIDE project and was then adapted from the
STM32CubeN6 Template FSBL project. The committed result is a repo-local,
CLI-oriented FSBL project that builds through the generated `FSBL/Debug`
makefiles and debug-loads through a Python wrapper without requiring
STM32CubeIDE. It also now includes the host-only ST Edge AI Phase 0 probe used
to validate the CPU-first TinyODOM path without touching the board.

The outer package has been renamed to `stm32_example_project` to reflect that
it now carries more than the original blink-only bring-up. The nested project
name remains `stm32_blink_example_project` for now so the generated makefiles
and debug-load workflow stay stable.

## Status Note

This directory is prototype/example tooling, not the production STM backend
surface.

- The production STM backend stages from `sketches/stm32/tinyodom_tcn_stm32_lrun`
  and is documented in `src/tinyodom/microcontrollers/README.md`.
- The flows here remain valid evidence and smoke-test helpers for bring-up,
  diagnostics, and exploratory measurement work.
- `cadenced`, archival clock sweeps, comparison runners, and other exploratory
  wrappers in this directory remain example-only unless they are explicitly
  promoted into the backend/config surface later.

## What To Keep Here

The useful long-lived contents of this directory are:

- repo-local STM32 project workspaces that are needed to reproduce builds and
  measurements:
  - `stm32_toy_ai_project/`
  - `stm32_cadenced_toy_ai_project/`
  - `stm32_lrun_toy_ai_project/`
- thin Python entrypoints that run a specific hardware-backed flow:
  - `run_*_hil.py`
  - `smoke_test_*.py`
  - `generate_and_stage_*.py`
  - sweep / comparison wrappers that call the main runners
- shared helpers that encode build / signing / staging behavior:
  - `stm32_phase2_candidate.py`
  - `stm32_lrun_common.py`
- plotting and summary scripts that consume saved sweep results

`stm32_phase2_candidate.py` now bootstraps the active dataset/task/model-family
pipeline before building the fixed `approx_trained` candidate. Downstream STM32
wrappers should consume its explicit `calibration_inputs`, `window_size`, and
`input_dim` fields instead of reaching through legacy `config.data.*` or raw
training-split aliases.

Things that should stay out of the committed source surface:

- `out/` run products
- ad hoc `results/` archives that only preserve past experiments
- one-off scratch notes and temporary debugging scripts
- generated ST Edge AI `network*` sources and weight blobs

For the LRUN track specifically, the durable value is:

- the `Appli` memory map and larger runtime RAM budget
- the unattended `dev_boot` measurement flow

The removed manual external-flash boot validation flow is intentionally not
part of the long-lived NAS path.

## Parallel LRUN Track

This directory now also carries a parallel LRUN example track under:

- `analysis_scripts/stm32_example_project/stm32_lrun_toy_ai_project/`

This LRUN path is separate from the existing FSBL-only analysis scripts. It is
meant to exercise the ST `Template_FSBL_LRUN` application layout without
changing the production backend in `src/tinyodom/microcontrollers`.

Important LRUN facts:

- Run LRUN commands from the `tinyodomex` conda environment.
- LRUN in this directory now supports only the unattended `dev_boot` flow.
- The point of the LRUN toy project is the `Appli` memory layout and RAM budget:
  about `2047K` linker-visible runtime RAM at `0x34000400`, not deployment-style
  external-flash boot validation.
- In LRUN `dev_boot`, the signed `Appli` image is programmed into external
  flash first, optional external weights are programmed separately, then the
  `FSBL` ELF is debug-loaded into RAM and copies the trusted app into AXISRAM.
- The old manual external-flash validation path was removed because it depended on
  manual BOOT strap changes and reset sequencing, which is not compatible with
  unattended NAS on the stock board.
- The repo-local LRUN project contains ST-derived files and ships copied
  license material:
  - `stm32_lrun_toy_ai_project/LICENSE.md`
  - `stm32_lrun_toy_ai_project/LICENSE.STM32N6xx_HAL_Driver.md`
  - `stm32_lrun_toy_ai_project/LICENSE.CMSIS.txt`

Main LRUN commands:

```bash
# Stub-only LRUN sanity pass, no harness required:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_lrun_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_lrun_toy_ai_project \
  --stub-only \
  --skip-harness

# Real LRUN run in development-boot mode:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_lrun_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_lrun_toy_ai_project

# LRUN smoke test: stub pass first, then real pass:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/smoke_test_stm32_lrun_toy_ai.py \
  --project-root analysis_scripts/stm32_example_project/stm32_lrun_toy_ai_project

# LRUN generation + staging only:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/generate_and_stage_stm32_lrun_toy_ai.py \
  --project-root analysis_scripts/stm32_example_project/stm32_lrun_toy_ai_project \
  --clean

# LRUN CPU sweep:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_lrun_cpu_clock_sweep.py \
  --project-root analysis_scripts/stm32_example_project/stm32_lrun_toy_ai_project

# LRUN back-to-back vs cadenced comparison:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_lrun_cadenced_comparison.py \
  --project-root analysis_scripts/stm32_example_project/stm32_lrun_toy_ai_project
```

## Current Status

- The project builds successfully from the committed `FSBL/Debug` makefiles.
- The board can be debug-loaded and run in development boot mode.
- The Python wrapper rebuilds the project, starts `ST-LINK_gdbserver`, loads the
  ELF with `arm-none-eabi-gdb`, and resumes execution.
- The ST Edge AI probe script can reproduce the current host-only Phase 0
  findings for STM32N6 CPU generation.
- After `make stm32-setup`, each committed FSBL carries the STM32CubeN6
  firmware subset it needs under a local `FSBL/Drivers/...` tree.

## Setup

The STM32 flow is split on purpose:

- `STM32CubeCLT` stays installed outside the repo
- STM32CubeN6 firmware is cloned into `tools/stm32/STM32CubeN6`

Before the first STM32 bootstrap run, ensure these `STM32CubeCLT` tools are on
your shell `PATH`:

- `ST-LINK_gdbserver`
- `arm-none-eabi-gdb`
- `STM32_Programmer_CLI`

The toy AI builds also need an ST Edge AI install. By default the committed
makefiles auto-discover the newest `/opt/ST/STEdgeAI/*` directory. If your
install lives elsewhere, export `STEDGEAI_ROOT=/path/to/STEdgeAI/<version>`
before building.

Then bootstrap the repo-local STM32 dependencies:

```bash
make stm32-setup
```

What `make stm32-setup` does:

- validates `ST-LINK_gdbserver`, `arm-none-eabi-gdb`, and `STM32_Programmer_CLI`
  on `PATH`
- clones or repairs `tools/stm32/STM32CubeN6`
- checks out the pinned firmware baseline `v1.3.0`
- refreshes the local `FSBL/Drivers/...` vendor subset used by the canonical
  template and the example STM32 projects

## Running The Wrapper

Default command from the repo root:

```bash
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py
```

Useful flags:

```bash
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --clean
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --no-run
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --verbose
```

Explicit path overrides still work:

```bash
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py \
  --gdbserver /path/to/ST-LINK_gdbserver \
  --gdb /path/to/arm-none-eabi-gdb \
  --cubeprog-bin /path/to/STM32CubeProgrammer/bin
```

Important: this is a development-memory debug load, not persistent flashing. It
does not sign binaries, program external flash, or change device security
state.

## Running The Phase 0 Smoke Test

The smoke test builds, loads, and runs the toy AI project end-to-end and
confirms the expected UART telemetry in one command:

```bash
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --clean
```

What the smoke test does:

- Opens `/dev/ttyACM0` at 115200 baud **before** triggering the load, so no
  early output is missed
- Optionally cleans and rebuilds the toy AI project
- Loads the toy AI ELF through the existing `ST-LINK_gdbserver` +
  `arm-none-eabi-gdb` RAM/debug-load flow
- Prints every received serial line to the terminal in real time
- Waits for `DUT READY`, then sends `START` to exercise the TinyODOM-style DUT
  handshake
- Waits up to 30 seconds for the legacy Phase 0 continuity tokens:
  - `STM32_AI_INIT=OK`
  - `STM32_AI_RUN=OK`
  - `STM32_AI_LATENCY_CYCLES=<value>`
- Exits 0 (PASS) if all tokens arrive, 1 (FAIL) otherwise

Useful flags:

```bash
# Use a different serial port or baud rate:
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --port /dev/ttyACM1 --baud 115200

# Extend the token-wait timeout (default 30 s):
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --serial-timeout 60
```

Phase 0 is considered complete when this script exits with status 0.

## Running The STM32 HIL Metrics Runner

The package also includes a standalone STM32 HIL runner that coordinates:

- the safe STM32 RAM/debug-load flow
- a fresh TinyODOM TFLite rebuild on each run using the perturbed `approx_trained` path
- ST Edge AI source generation + staging into the STM32 toy project
- selectable toy weight storage (`embedded` or `external_flash`, see below)
- the existing Arduino harness firmware on `D2`/`D3`
- the Nucleo GPIO mapping validated on hardware: `D2 -> PD0` trigger, `D3 -> PE9` active-low arm
- parsing of harness energy telemetry and DUT clock/cycle telemetry
- emission of a compact metrics JSON plus a diagnostic sidecar
- optional `back_to_back` versus `cadenced` DUT firmware modes through a generated
  `toy_ai_phase_config.h`

For current support expectations: `back_to_back` is the production-backed STM
mode today; `cadenced` in this directory remains prototype/example-only.

Run these commands from the `tinyodomex` conda environment. The cadence and
clock-preset paths below are validated against a connected STM32 board, not as
host-only dry runs.

Default command:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py
```

Useful flags:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --clean
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --config src/config/nas_config.yaml
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --dut-port /dev/ttyACM0 --harness-port /dev/ttyACM1
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --output analysis_scripts/stm32_example_project/last_metrics.json
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --latency-budget-ms 200.0
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --reuse-staged-model
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --weight-storage-mode external_flash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --weight-storage-mode external_flash --weights-flash-address 0x71000000
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --weight-storage-mode external_flash --weights-memory-pool analysis_scripts/stm32_example_project/nucleo_mypool.json
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --phase cadenced --latency-budget-ms 200 --measured-runs 100
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --phase cadenced --wake-margin-us 5000 --min-sleep-us 5000
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --cpu-clock-mhz 400
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --cpu-clock-mhz 800
```

`--cpu-clock-mhz` accepts fixed presets `200`, `300`, `400`, `600`, and `800`.
The default is `600`. Presets `200/300/400/600` keep the low-disturbance PLL1
tree, while `800` switches to the dedicated overdrive profile and is the
higher-risk option.

### Cadenced RTC Source

The cadenced STM32 project now uses `LSE` as its RTC source and keeps the RTC
calendar plus wake-up timer on one consistent `32.768 kHz` basis.

Why `LSE`:

- `LSI` is an internal RC oscillator, so its frequency tolerance is coarse and
  accumulated cadence drift shows up directly in Stop/wake timing.
- `LSE` is the crystal-backed `32.768 kHz` low-speed clock source, which makes
  cadenced wall-clock timing materially more stable and repeatable.
- The RTC source choice is independent of the `--cpu-clock-mhz` presets. CPU
  clock tuning still happens on the high-speed tree; the RTC change only affects
  the cadenced low-speed timing domain.
- The expected board-level energy difference between `LSI` and `LSE` is
  negligible in this workflow. The reason for choosing `LSE` is timing
  correctness, not power reduction.

This workflow is run from the `tinyodomex` conda environment. The STM32 board
and harness board are attached to the machine used for this flow, but live
validation of the `LSE` change is intentionally deferred when only a host-side
implementation pass is needed.

ST primary references:

- STM32N657 datasheet oscillator characteristics:
  <https://www.st.com/resource/en/datasheet/stm32n657i0.pdf>
- STM32N6 RTC bring-up example from ST:
  <https://community.st.com/t5/stm32-mcus/how-to-add-rtc-on-the-stm32n6/ta-p/823623>

Common commands:

```bash
# Standard single-attempt back-to-back run:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py

# Standard single-attempt cadenced run at 200 ms cadence:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --latency-budget-ms 200 \
  --measured-runs 100

# Cadenced run at the 400 MHz low-disturbance preset:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --cpu-clock-mhz 400

# Back-to-back run at the 800 MHz overdrive preset:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --cpu-clock-mhz 800

# Faster iteration when sources are already staged and the ELF is current:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --measured-runs 100 \
  --reuse-staged-model \
  --output /tmp/stm32_cadenced_metrics.json
```

Use `--measured-runs N` to change the number of measured inferences per DUT
attempt. For example, `--measured-runs 100` rebuilds and runs a 100-inference
window while keeping the default at `10` when the flag is omitted.

To inspect the full CLI with examples:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --help
```

To capture an external-flash run under a distinct output name:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --clean \
  --weight-storage-mode external_flash \
  --output analysis_scripts/stm32_example_project/stm32_toy_ai_metrics_ext_flash.json
```

### Weight Storage Modes

**`embedded` (default)**

ST Edge AI generates the model weights as C arrays that are compiled directly
into the ELF. When GDB debug-loads the ELF into RAM, weights travel with it.
The firmware accesses weights from RAM at runtime — no separate programming
step is needed and no external hardware is involved beyond the ST-LINK.

This is the faster iteration path. Use it when you want to validate inference
end-to-end without caring about where weights physically live on a shipped
device.

**`external_flash`**

ST Edge AI generates a separate raw binary blob instead of C arrays. Before
the GDB load, the runner programs that blob into the Nucleo's external NOR
flash (MX25UM51245G) at a fixed base address (default `0x71000000`) using
`STM32_Programmer_CLI` with the appropriate external loader (`.stldr`). The
firmware ELF is then debug-loaded into RAM as usual. At runtime the firmware
reads weights directly out of NOR flash rather than RAM.

This is the path that more closely matches a real deployment, where model
weights are typically too large for internal flash and live in external
storage. It adds a `STM32_Programmer_CLI` round-trip before each run and
requires the correct external loader to be present either on PATH or passed
explicitly via `--weights-external-loader`.

The difference shows up in the output JSON:

| Field | `embedded` | `external_flash` |
|---|---|---|
| `weights_flash_bytes` | `0` | size of the programmed blob |
| `flash_bytes` | ELF text + data only | ELF text + data + blob size |
| `weights_programmed` | `false` | `true` if programming succeeded |
| `weights_flash_address` | `null` | e.g. `"0x71000000"` |

Notes:

- By default, each run rebuilds the perturbed TinyODOM TFLite model, regenerates STM32 network sources, and then rebuilds the Cube project.
- In `external_flash` mode the runner reads `<stage_output_root>/staging_manifest.json`, programs the generated weights blob into Nucleo external flash, and only then starts the usual `ST-LINK_gdbserver` + `arm-none-eabi-gdb` RAM/debug-load.
- The default external weight address is intentionally `0x71000000` for the toy flow.

Main JSON fields:

- `ram_bytes`, `flash_bytes`, `elf_flash_bytes`, `weights_flash_bytes`, `arena_bytes`
- `weight_storage_mode`, `weights_flash_address`, `weights_programmed`
- `latency_ms`, `latency_budget_ms`, `harness_latency_ms`, `dwt_cycles_per_inference`
- `energy_mj_per_inference`, `avg_power_mw`, `avg_current_ma`
- `bus_voltage_v`, `idle_power_mw`
- `hil_enabled`, `error_code`, `error_label`

Size semantics:

- `elf_flash_bytes = ELF text + data`
- `weights_flash_bytes = 0` in `embedded` mode, otherwise the generated weights blob size
- `flash_bytes = elf_flash_bytes + weights_flash_bytes`

The runner also writes a sidecar diagnostics JSON next to the main output with:

- raw `arm-none-eabi-size` columns
- linker-reserved heap/stack
- parsed DUT clock/cycle telemetry
- DUT and harness line logs for debugging

Cadenced-mode fields now also include:

- `phase`
- `window_latency_ms`
- `active_inference_latency_ms`
- `rtc_sleep_ms`
- `deadline_miss_count`
- `wake_recovery_us_mean`
- `wake_overshoot_us_mean`
- `rtc_clock_source`
- `stop_mode_variant`

## Running The STM32 Cadenced Comparison Runner

The comparison runner wraps the single-run HIL flow and executes both:

- `back_to_back`
- `cadenced`

It writes:

- a JSON summary with metadata, attempts, aggregates, and per-attempt diagnostics
- a CSV file with one row per attempt plus comparison-friendly columns

Run this through `conda run -n tinyodomex ...` or from an already activated
`tinyodomex` shell. The wrapper is intended for live connected-board runs and
records both the requested preset (`cpu_clock_mhz_requested`) and measured DUT
telemetry (`clock_hz`) separately.

Default command:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py
```

Common commands:

```bash
# One back-to-back + cadenced comparison pass:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py

# Repeat each phase three times:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --repeats 3 \
  --latency-budget-ms 200

# Write outputs to explicit locations:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --output-json /tmp/stm32_compare.json \
  --output-csv /tmp/stm32_compare.csv

# Run the comparison against external-flash weights:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --weight-storage-mode external_flash

# Run the comparison at the 400 MHz low-disturbance preset:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --cpu-clock-mhz 400

# Run the comparison at the 800 MHz overdrive preset:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --cpu-clock-mhz 800
```

`--cpu-clock-mhz` accepts `200`, `300`, `400`, `600`, and `800`, with `600`
as the default baseline. Treat `800` as the explicit overdrive and higher-risk
preset.

To inspect the full CLI with examples:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py --help
```

Notes:

- The comparison runner delegates each attempt to `run_stm32_toy_ai_hil.py`.
- The generated phase header is rewritten before each phase build.
- `--wake-margin-us` and `--min-sleep-us` are exposed on both runners so cadence tuning stays scriptable.
- The copied cadenced project currently uses the RTC wake-up timer with the secure RTC IRQ path enabled.

## Running The STM32 Staging Helper

Use the staging helper when you want to regenerate the STM32 toy-AI source tree
without immediately building or running a HIL attempt:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/generate_and_stage_stm32_toy_ai.py --clean
```

Common commands:

```bash
# Rebuild the default perturbed model and restage the generated network files:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/generate_and_stage_stm32_toy_ai.py \
  --config src/config/nas_config.yaml

# Stage a prebuilt TFLite model instead of rebuilding one:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/generate_and_stage_stm32_toy_ai.py \
  --model /tmp/tinyodom_model.tflite

# Generate sources for external-flash weights and record the flash handoff in the manifest:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/generate_and_stage_stm32_toy_ai.py \
  --weight-storage-mode external_flash \
  --weights-flash-address 0x71000000
```

This script writes its transient outputs under `/tmp/tinyodom_stm32_toy_generate`
by default and refreshes `staging_manifest.json` for the downstream HIL runner.

## Running The STM32 CPU Clock Sweep

The archival sweep runner executes the STM32 HIL runner repeatedly across:

- requested CPU clock presets `200`, `300`, `400`, `600`, and `800`
- both phases (`back_to_back`, `cadenced`)
- both weight placement modes (`embedded`, `external_flash`)

It preserves every run's metrics JSON, diagnostics JSON, stdout/stderr logs,
and writes a summary CSV in a timestamped folder under
`analysis_scripts/stm32_example_project/results/`.

Default command:

```bash
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py
```

Common commands:

```bash
# Run the full archival matrix with an explicit fresh first build:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py \
  --clean-first

# Restrict the sweep to selected presets and phases:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py \
  --frequencies 400 600 800 \
  --phases cadenced \
  --repeats 3

# Stop immediately after the first failed child run:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py \
  --fail-fast
```

The default repeat count for this archival sweep is `5` per scenario and
frequency. Use `--results-root` to send the timestamped archive somewhere other
than `analysis_scripts/stm32_example_project/results/`.

## Plotting Archived Sweep Results

Use the plotting helper to turn one archived results folder into the current
comparison plots:

```bash
python analysis_scripts/stm32_example_project/plot_stm32_cpu_clock_sweep.py \
  analysis_scripts/stm32_example_project/results/stm32_cpu_clock_sweep_20260411T054113Z
```

The plotting script expects a results directory that already contains
`sweep_summary.csv` and prints the output plot paths it writes.

## Running The Phase 0 Probe

The package also contains a host-only ST Edge AI probe:

```bash
python analysis_scripts/stm32_example_project/run_stedgeai_phase0_probe.py --clean
```

What the probe does:

- runs `stedgeai analyze` on the representative TinyODOM `.tflite`
- runs a CM55-style binary `generate` pass with `--c-api legacy`
- runs a non-binary legacy `generate` pass
- captures logs and generated artifacts under `/tmp` by default

This probe is intentionally board-free. It does not use `ST-LINK_gdbserver`,
does not load firmware, and does not program flash.

## Project Layout

- `build_and_upload_stm32_blink.py`
  - CLI wrapper for build + debug-load of the blink project
- `smoke_test_stm32_toy_ai.py`
  - end-to-end smoke test: build, load, and confirm UART tokens from the toy AI firmware
- `run_stm32_toy_ai_hil.py`
  - full HIL runner: stages a fresh perturbed TinyODOM model, builds the toy AI project,
    optionally programs external weights, runs the Arduino harness, and emits metrics JSON
- `run_stm32_cadenced_comparison.py`
  - comparison wrapper that runs both `back_to_back` and `cadenced` phases and writes JSON + CSV summaries
- `run_stm32_cpu_clock_sweep.py`
  - archival sweep wrapper for repeated STM32 runs across clock presets, phases, and weight-storage modes
- `plot_stm32_cpu_clock_sweep.py`
  - plotting helper for an archived sweep folder that already contains `sweep_summary.csv`
- `run_stedgeai_phase0_probe.py`
  - host-only ST Edge AI probe for the STM32N6 CPU-first Phase 0 work
- `generate_and_stage_stm32_toy_ai.py`
  - toy STM32 source staging with `embedded` vs `external_flash` weight placement and a fixed-path staging manifest
- `stm32_blink_example_project/FSBL/`
  - original blink FSBL subproject kept in CLI-buildable form
- `stm32_blink_example_project/FSBL/Debug/`
  - committed generated make metadata required by the blink wrapper
- `stm32_toy_ai_project/FSBL/`
  - toy AI FSBL subproject; receives ST Edge AI generated sources and is built by the HIL runner
- `stm32_toy_ai_project/FSBL/Debug/`
  - committed generated make metadata plus `stedgeai.mk` path overrides required by the HIL runner
- `stm32_cadenced_toy_ai_project/FSBL/`
  - copied cadence-capable toy AI FSBL subproject used by the comparison runner
- `results/`
  - timestamped archived CPU-clock sweep folders containing metrics, logs, summaries, and plots

The example FSBL projects no longer compile directly from `tools/stm32`.
After `make stm32-setup`, they build from their local `FSBL/Drivers/...` trees.

## First Run Notes

- Use the board's standard development-boot strap setting for this debug flow
- Start with the debug-load flow, not the external-flash boot flow
