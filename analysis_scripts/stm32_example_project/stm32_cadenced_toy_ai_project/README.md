# STM32 Cadenced Toy AI Project

This directory contains the copied STM32 FSBL project used for the
`back_to_back` versus `cadenced` STM32 comparison flow.

The main entry points are the Python wrappers in the parent package:

- `analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py`
- `analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py`
- `analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py`
- `analysis_scripts/stm32_example_project/plot_stm32_cpu_clock_sweep.py`

## Notes

- The firmware phase is selected by the generated header:
  `FSBL/Inc/toy_ai_phase_config.h`
- The Python runners rewrite that header before each relevant build, including
  the fixed `TOY_AI_CPU_CLOCK_MHZ` preset.
- The project is meant to be built through the generated `FSBL/Debug`
  makefiles, typically via the Python wrappers rather than manually.
- The committed `FSBL/Debug/stedgeai.mk` file auto-discovers the newest
  `/opt/ST/STEdgeAI/*` install or honors `STEDGEAI_ROOT` if you need a
  non-default ST Edge AI location.
- The cadenced phase uses STM32 Stop mode plus the RTC wake-up timer and
  emits additional cadence telemetry for the host-side comparison scripts.
- Use the `tinyodomex` conda environment for the host runners. This flow is
  validated against a connected Nucleo board rather than as a host-only build.
- `--cpu-clock-mhz` accepts `200`, `300`, `400`, `600`, and `800`, with `600`
  as the default baseline. `800` is the dedicated overdrive and higher-risk
  preset.

## Cadenced RTC Source

The cadenced project uses `LSE` as its RTC source and keeps both RTC
subsecond timestamps and wake-up timer requests on one explicit `32.768 kHz`
timebase.

Why `LSE` is the chosen default:

- `LSI` is an internal RC source, so cadence windows accumulate drift and
  repeatability suffers.
- `LSE` is the crystal-backed `32.768 kHz` source, which is the correct clock
  domain when the goal is stable wall-clock cadence timing through Stop/wake.
- This change does not alter the CPU clock-preset behavior. `--cpu-clock-mhz`
  still controls the high-speed CPU tree only.
- The expected board-level energy impact of `LSE` versus `LSI` is negligible;
  the choice is about timing quality, not lowering power.

Host commands for this project should be run from the `tinyodomex` conda
environment. The STM32 board and harness board are attached to the machine used
for development, but live validation of this RTC-source change can be deferred
when only a firmware/docs implementation pass is needed.

ST primary references:

- STM32N657 datasheet oscillator characteristics:
  <https://www.st.com/resource/en/datasheet/stm32n657i0.pdf>
- STM32N6 RTC bring-up example from ST:
  <https://community.st.com/t5/stm32-mcus/how-to-add-rtc-on-the-stm32n6/ta-p/823623>

## Common Commands

From the repo root:

```bash
# Build and run one cadenced HIL attempt:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --latency-budget-ms 200 \
  --measured-runs 100

# Run the same flow at the 400 MHz low-disturbance preset:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --measured-runs 100 \
  --cpu-clock-mhz 400

# Run the same flow at the 800 MHz overdrive preset:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --measured-runs 100 \
  --cpu-clock-mhz 800

# Run the full back-to-back vs cadenced comparison:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py

# Run the archival CPU-clock sweep and preserve every child-run artifact:
conda run -n tinyodomex python analysis_scripts/stm32_example_project/run_stm32_cpu_clock_sweep.py \
  --clean-first

# Regenerate plots for one archived sweep folder:
python analysis_scripts/stm32_example_project/plot_stm32_cpu_clock_sweep.py \
  analysis_scripts/stm32_example_project/results/stm32_cpu_clock_sweep_20260411T054113Z

# Manual clean build only:
make -C analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL/Debug clean all -j1
```

Use `--measured-runs N` on the host runner to change the per-attempt
measurement window. For example, `--measured-runs 100` rebuilds the generated
phase-config header and runs a 100-slot cadenced attempt.
