# Clock Tick Latency

Run repeated perturbed-model HIL attempts on Portenta H7 and normalize
latency from `ms/inference` to `ticks/inference`.

This workflow is the main consumer of the shared DUT clock telemetry added to
the TinyODOM energy sketches:

- `clock hz output: ...`
- `dwt cycles per inference output: ...`

This script supports:
- `PORTENTA_H7` `cm7`
- `PORTENTA_H7` `cm4`

Outputs per run:
- attempt-level CSV
- JSON (`metadata`, `attempts`, `aggregates`)
- PNG scatter (`latency_ms` vs `ticks_per_inference`)

## Script

```bash
python analysis_scripts/clock_tick_latency/run_clock_tick_latency.py --help
```

## Usage

CM7:

```bash
python analysis_scripts/clock_tick_latency/run_clock_tick_latency.py \
  --device PORTENTA_H7 \
  --portenta-core cm7 \
  --repeats 5 \
  --input-mode uniform
```

CM4:

```bash
python analysis_scripts/clock_tick_latency/run_clock_tick_latency.py \
  --device PORTENTA_H7 \
  --portenta-core cm4 \
  --repeats 5 \
  --input-mode uniform
```

## Notes

- Default model variant is `approx_trained` (perturbed-path behavior).
- Tick normalization priority:
  1. `dwt_cycles_per_inference`
  2. `latency_s * clock_hz`
  3. fallback to `-1` with reason label
- Runtime clock source priority:
  1. DUT runtime telemetry (`runtime_reported`)
  2. Built-in fallback (`fallback_estimate`)
  3. unavailable (`-1`)

Built-in fallback clocks:

```text
cm7 = 400000000 Hz
cm4 = 240000000 Hz
```

Optional config overrides:

```yaml
device:
  portenta:
    clock_hz_cm7: 400000000
    clock_hz_cm4: 240000000
```

If runtime clock telemetry is unavailable, the script uses the built-in core
default unless you override it in config.

On cores that do not expose DWT telemetry in the active runtime path, expect the
script to fall back to `latency_s * clock_hz` or the configured core default.
