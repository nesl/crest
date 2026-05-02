# Sketches

This directory contains the repo-owned Arduino and STM32 sketch/workspace
sources used by TinyODOM-EX runtime, HIL, and analysis flows.

For board bring-up details, see
[`../src/tinyodom/microcontrollers/README.md`](../src/tinyodom/microcontrollers/README.md).
For config semantics, see
[`../src/config/README.md`](../src/config/README.md).

## Directory Map

- [`tinyodom_tcn_energy.ino`](tinyodom_tcn_energy.ino)
  Shared uniform-input Arduino runtime sketch used when
  `training.input_mode: uniform` and `training.energy_aware: true`.
- [`tinyodom_tcn_no_energy.ino`](tinyodom_tcn_no_energy.ino)
  Shared uniform-input Arduino runtime sketch used when
  `training.input_mode: uniform` and `training.energy_aware: false`.
- [`tinyodom_tcn_energy_cadenced.ino`](tinyodom_tcn_energy_cadenced.ino)
  Shared Arduino cadenced-pass sketch.
- [`analysis_sketches/`](analysis_sketches)
  Analysis-oriented sketch variants and data headers for non-uniform input
  modes.
- [`common/`](common)
  Shared headers copied into staged Arduino candidate directories.
- [`harness/`](harness)
  The energy/runtime harness sketch and its local support files.
- [`boot_m4_helper/`](boot_m4_helper)
  Portenta CM7 helper sketch used to bring up CM4 in the harness-only path.
- [`stm32/`](stm32)
  STM32 workspaces. The current committed production workspace is
  [`stm32/tinyodom_stm32_lrun/README.md`](stm32/tinyodom_stm32_lrun/README.md).

## Runtime Sketch Selection

The current Arduino runtime-selection path described in
[`../src/tinyodom/microcontrollers/README.md`](../src/tinyodom/microcontrollers/README.md)
uses this layout:

1. Uniform input mode:
   - [`tinyodom_tcn_energy.ino`](tinyodom_tcn_energy.ino)
   - [`tinyodom_tcn_no_energy.ino`](tinyodom_tcn_no_energy.ino)
2. Cadenced second pass:
   - [`tinyodom_tcn_energy_cadenced.ino`](tinyodom_tcn_energy_cadenced.ino)
3. Analysis input modes:
   - [`analysis_sketches/tinyodom_tcn_energy_representative.ino`](analysis_sketches/tinyodom_tcn_energy_representative.ino)
   - [`analysis_sketches/tinyodom_tcn_energy_real_data.ino`](analysis_sketches/tinyodom_tcn_energy_real_data.ino)
   - [`analysis_sketches/tinyodom_tcn_input_data.h`](analysis_sketches/tinyodom_tcn_input_data.h)
4. Shared copied headers:
   - [`common/tinyodom_hil_config.h`](common/tinyodom_hil_config.h)
   - [`common/tinyodom_power.h`](common/tinyodom_power.h)
   - [`common/tinyodom_clock_telemetry.h`](common/tinyodom_clock_telemetry.h)

## Harness And Helper Sketches

- [`harness/harness.ino`](harness/harness.ino)
  Used when runtime measurement is harness-assisted or harness-only.
- [`boot_m4_helper/boot_m4_helper.ino`](boot_m4_helper/boot_m4_helper.ino)
  Used by the current Portenta CM4 path to bring up CM4 before uploading the
  DUT sketch.

## STM32 Workspace

STM32 content is not organized as one Arduino sketch per board.

The current committed workspace is:

- [`stm32/tinyodom_stm32_lrun/README.md`](stm32/tinyodom_stm32_lrun/README.md)

See that README for the LRUN workspace internals, staged/generated file
layout, and STM32-specific workflow details.
