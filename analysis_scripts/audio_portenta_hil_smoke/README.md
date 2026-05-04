# Audio Portenta HIL Smoke

This Phase 8 smoke runner validates the Arduino-backed audio DS-CNN path over
precomputed UrbanSound8K log-mel tensors. It does not add microphone capture,
buffering, FFT, mel filtering, or firmware-side frontend timing.

## Modes

```bash
python analysis_scripts/audio_portenta_hil_smoke/run_audio_portenta_hil_smoke.py --preflight-only
python analysis_scripts/audio_portenta_hil_smoke/run_audio_portenta_hil_smoke.py --prepare-only
python analysis_scripts/audio_portenta_hil_smoke/run_audio_portenta_hil_smoke.py
```

- `--preflight-only`: load config/cache, build the seeded untrained model, and
  write a diagnostic TFLite file.
- `--prepare-only`: export/stage the Arduino candidate and run compile-only
  metrics. This does not upload or measure hardware.
- default: run the full HIL path through `HILServer`.

Use BLE through the same path:

```bash
python analysis_scripts/audio_portenta_hil_smoke/run_audio_portenta_hil_smoke.py --device-name ARDUINO_NANO_33_BLE_SENSE --prepare-only
```

BLE may block because the int8 input tensor alone is `12,864` bytes before
weights and activation arena. The summary JSON records compile/runtime metrics
when that happens.
