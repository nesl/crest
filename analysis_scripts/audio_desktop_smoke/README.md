# Audio Desktop Smoke

This hardware-free smoke runner validates the UrbanSound8K `audio_dscnn`
training path on the host before moving into Arduino or STM32 HIL work. It
loads the configured cached log-mel tensors, slices the train/validation
splits to a small row count, trains the default seeded DS-CNN model briefly,
and writes checkpoint, history, and metrics artifacts.

## Command

```bash
python analysis_scripts/audio_desktop_smoke/run_audio_desktop_smoke.py
```

Useful overrides:

```bash
python analysis_scripts/audio_desktop_smoke/run_audio_desktop_smoke.py --epochs 2 --max-train-examples 256 --max-val-examples 128
python analysis_scripts/audio_desktop_smoke/run_audio_desktop_smoke.py --config src/config/nas_config_audio_portenta.yaml
```

The default config is `src/config/nas_config_audio_stm32.yaml`; the runner does
not construct a HIL server or touch board toolchains.

## Inputs And Outputs

Run this first if the UrbanSound8K cache is missing:

```bash
make prepare-audio-dataset
```

By default artifacts are written under:

```text
models/audio_desktop_smoke/
```

The runner writes:

- `audio_desktop_smoke.keras`
- `audio_desktop_smoke_history.json`
- `audio_desktop_smoke_metrics.json`
