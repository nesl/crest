# STM32 Cadenced Toy AI Project

This directory contains the copied STM32CubeIDE project used for the
`back_to_back` versus `cadenced` STM32 comparison flow.

The main entry points are the Python wrappers in the parent package:

- `analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py`
- `analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py`

## Notes

- The firmware phase is selected by the generated header:
  `FSBL/Inc/toy_ai_phase_config.h`
- The Python runners rewrite that header before each relevant build.
- The project is meant to be built through the generated `FSBL/Debug`
  makefiles, typically via the Python wrappers rather than manually.
- The cadenced phase uses STM32 Stop mode plus the RTC wake-up timer and
  emits additional cadence telemetry for the host-side comparison scripts.

## Common Commands

From the repo root:

```bash
# Build and run one cadenced HIL attempt:
python analysis_scripts/stm32_example_project/run_stm32_toy_ai_hil.py \
  --project-root analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL \
  --phase cadenced \
  --latency-budget-ms 200 \
  --measured-runs 10

# Run the full back-to-back vs cadenced comparison:
python analysis_scripts/stm32_example_project/run_stm32_cadenced_comparison.py

# Manual clean build only:
make -C analysis_scripts/stm32_example_project/stm32_cadenced_toy_ai_project/FSBL/Debug clean all -j1
```
