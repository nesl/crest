# Audio STM32 HIL Smoke

This folder contains the Phase 6 smoke runner for the UrbanSound8K
`audio_dscnn` path on the STM32 N657 backend.

The runner measures classifier inference over cached log-mel feature tensors.
It does not measure microphone capture, buffering, FFT, mel filtering, or
firmware-side feature extraction.

## Commands

Hardware-free preflight:

```bash
python analysis_scripts/audio_stm32_hil_smoke/run_audio_stm32_hil_smoke.py --preflight-only
```

ST Edge AI generation and STM32 candidate staging only:

```bash
python analysis_scripts/audio_stm32_hil_smoke/run_audio_stm32_hil_smoke.py --prepare-only
```

Full board run:

```bash
python analysis_scripts/audio_stm32_hil_smoke/run_audio_stm32_hil_smoke.py --serial-port /dev/ttyACM0
```

Useful overrides:

```bash
python analysis_scripts/audio_stm32_hil_smoke/run_audio_stm32_hil_smoke.py --runtime-mode cadenced --measured-runs 10 --cpu-clock-mhz 600
```

Harness-assisted power metrics:

```bash
python analysis_scripts/audio_stm32_hil_smoke/run_audio_stm32_hil_smoke.py --serial-port /dev/ttyACM0 --harness-serial-port /dev/ttyACM1 --runtime-mode cadenced --measured-runs 10 --cpu-clock-mhz 600 --energy-aware
```

## Outputs

By default the runner writes summaries and diagnostic artifacts under:

```text
models/audio_stm32_hil_smoke/
```

The diagnostic TFLite file from `--preflight-only` is float32 and exists only to
prove host-side export works. STM32 candidate preparation still creates its own
candidate-local TFLite artifact and applies the configured quantization path.
With the default audio STM32 config, that staged candidate uses int8 input, so
`AI_NETWORK_IN_1_SIZE_BYTES` is expected to be `201 * 64 = 12864`.

The STM32 LRUN template runs the generated ST Edge AI network and HIL timing
protocol for any staged model. Phase 6 still uses precomputed log-mel features;
microphone capture and on-device audio frontend processing remain out of scope
for this smoke script.
