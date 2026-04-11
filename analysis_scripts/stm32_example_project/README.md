# STM32 Example Project

This package contains the current STM32N6 example assets for the
`NUCLEO-N657X0-Q` board inside `tinyodom-ex`.

It started as a fresh STM32CubeIDE project and was then adapted from the
STM32CubeN6 Template FSBL project. The committed result is a repo-local,
self-contained CubeIDE project that builds through the generated `FSBL/Debug`
makefiles and debug-loads through a Python wrapper. It also now includes the
host-only ST Edge AI Phase 0 probe used to validate the CPU-first TinyODOM
path without touching the board.

The outer package has been renamed to `stm32_example_project` to reflect that
it now carries more than the original blink-only bring-up. The nested
STM32CubeIDE project name remains `stm32_blink_example_project` for now so the
existing generated makefiles, launch metadata, and debug workflow stay stable.

## Current Status

- The project builds successfully from the committed `FSBL/Debug` makefiles.
- The board can be debug-loaded and run in development boot mode.
- The Python wrapper rebuilds the project, starts `ST-LINK_gdbserver`, loads the
  ELF with `arm-none-eabi-gdb`, and resumes execution.
- The ST Edge AI probe script can reproduce the current host-only Phase 0
  findings for STM32N6 CPU generation.
- The STM32 firmware headers are expected under `tools/stm32/STM32CubeN6`,
  pinned to `v1.3.0`.

## Setup

The STM32 flow is split on purpose:

- `STM32CubeCLT` stays installed outside the repo
- STM32CubeN6 firmware is cloned into `tools/stm32/STM32CubeN6`

Before the first STM32 bootstrap run, ensure these `STM32CubeCLT` tools are on
your shell `PATH`:

- `ST-LINK_gdbserver`
- `arm-none-eabi-gdb`
- `STM32_Programmer_CLI`

Then bootstrap the repo-local STM32 dependencies:

```bash
make stm32-setup
```

What `make stm32-setup` does:

- validates `ST-LINK_gdbserver`, `arm-none-eabi-gdb`, and `STM32_Programmer_CLI`
  on `PATH`
- clones or repairs `tools/stm32/STM32CubeN6`
- checks out the pinned firmware baseline `v1.3.0`

## Running The Wrapper

Default command from the repo root:

```bash
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py
```

Useful flags:

```bash
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --clean
python analysis_scripts/stm32_example_project/build_and_upload_stm32_blink.py --no-build
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
# Skip rebuild if the ELF is already current:
python analysis_scripts/stm32_example_project/smoke_test_stm32_toy_ai.py --no-build

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

Default command:

```bash
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py
```

Useful flags:

```bash
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --clean
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --config src/nas_config.yaml
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --dut-port /dev/ttyACM0 --harness-port /dev/ttyACM1
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --output analysis_scripts/stm32_example_project/last_metrics.json
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --latency-budget-ms 200.0
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --reuse-staged-model --no-build
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --weight-storage-mode external_flash
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --weight-storage-mode external_flash --weights-flash-address 0x71000000
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --weight-storage-mode external_flash --weights-memory-pool analysis_scripts/stm32_example_project/nucleo_mypool.json
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --phase cadenced --latency-budget-ms 200 --measured-runs 10
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --phase cadenced --wake-margin-us 5000 --min-sleep-us 5000
```

Common commands:

```bash
# Standard single-attempt back-to-back run:
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py

# Standard single-attempt cadenced run at 200 ms cadence:
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --latency-budget-ms 200 \
  --measured-runs 10

# Faster iteration when sources are already staged and the ELF is current:
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --reuse-staged-model \
  --no-build \
  --output /tmp/stm32_cadenced_metrics.json
```

To inspect the full CLI with examples:

```bash
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py --help
```

To capture an external-flash run under a distinct output name:

```bash
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
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
- `--no-build` is only valid together with `--reuse-staged-model`, because otherwise the staged sources would change underneath a stale ELF.
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

Default command:

```bash
python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py
```

Common commands:

```bash
# One back-to-back + cadenced comparison pass:
python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py

# Repeat each phase three times:
python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --repeats 3 \
  --latency-budget-ms 200

# Write outputs to explicit locations:
python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --output-json /tmp/stm32_compare.json \
  --output-csv /tmp/stm32_compare.csv

# Run the comparison against external-flash weights:
python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py \
  --weight-storage-mode external_flash
```

To inspect the full CLI with examples:

```bash
python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py --help
```

Notes:

- The comparison runner delegates each attempt to `run_stm32_toy_ai_hil.py`.
- The generated phase header is rewritten before each phase build.
- `--wake-margin-us` and `--min-sleep-us` are exposed on both runners so cadence tuning stays scriptable.
- The copied cadenced project currently uses the RTC wake-up timer with the secure RTC IRQ path enabled.

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
- `run_stedgeai_phase0_probe.py`
  - host-only ST Edge AI probe for the STM32N6 CPU-first Phase 0 work
- `generate_and_stage_stm32_toy_ai.py`
  - toy STM32 source staging with `embedded` vs `external_flash` weight placement and a fixed-path staging manifest
- `stm32_blink_example_project/FSBL/`
  - original blink STM32CubeIDE FSBL subproject
- `stm32_blink_example_project/FSBL/Debug/`
  - committed generated make metadata required by the blink wrapper
- `stm32_toy_ai_project/FSBL/`
  - toy AI STM32CubeIDE FSBL subproject; receives ST Edge AI generated sources and is built by the HIL runner
- `stm32_toy_ai_project/FSBL/Debug/`
  - committed generated make metadata required by the HIL runner
- `stm32_cadenced_toy_ai_project/FSBL/`
  - copied cadence-capable toy AI FSBL subproject used by the comparison runner

The FSBL project now expects the STM32CubeN6 firmware headers at:

```text
tools/stm32/STM32CubeN6
```

## First Run Notes

- Use development boot mode:
  - `BOOT1 = 2-3`
  - `BOOT0` does not matter for this debug flow
- Start with the debug-load flow, not the external-flash boot flow
