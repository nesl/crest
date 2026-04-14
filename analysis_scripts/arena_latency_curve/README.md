# Arena Latency Curve

Run fixed-arena HIL sweeps and produce latency/energy vs arena-size outputs for:
- `ARDUINO_NANO_33_BLE_SENSE`
- `PORTENTA_H7` (`cm7` or `cm4`)

Outputs per run:
- attempt-level CSV
- aggregated JSON
- dual-axis plot PNG (latency + energy vs arena KiB)

## Scripts

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve.py --help
```

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve_failure_probe.py --help
```

## Quick examples

BLE sweep:

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve.py \
  --device ARDUINO_NANO_33_BLE_SENSE \
  --arena-kb-list 10,25,32,64 \
  --repeats 1
```

Portenta CM7 sweep:

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve.py \
  --device PORTENTA_H7 \
  --portenta-core cm7 \
  --arena-kb-list 10,25,32,64 \
  --repeats 1
```

Portenta CM4 sweep:

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve.py \
  --device PORTENTA_H7 \
  --portenta-core cm4 \
  --arena-kb-list 10,25,32,64 \
  --repeats 1
```

Use device default arena candidates (omit `--arena-kb-list`):

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve.py \
  --device PORTENTA_H7 \
  --portenta-core cm7 \
  --repeats 3
```

Failure-focused probe that narrows the arena window and intentionally
under-fills the arena to make boundary behavior easier to inspect:

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve_failure_probe.py \
  --device PORTENTA_H7 \
  --portenta-core cm7 \
  --min-arena-kb 10 \
  --max-arena-kb 32 \
  --arena-underfill-kb 1 \
  --repeats 1
```

Probe a smaller or larger model footprint by forcing `nb_filters`:

```bash
python analysis_scripts/arena_latency_curve/run_arena_latency_curve_failure_probe.py \
  --device ARDUINO_NANO_33_BLE_SENSE \
  --nb-filters 12 \
  --arena-kb-list 10,12,14,16 \
  --repeats 1
```

## Notes

- `run_arena_latency_curve.py` defaults to `--repeats 3` and
  `--model-variant approx_trained`.
- `run_arena_latency_curve_failure_probe.py` adds `--min-arena-kb`,
  `--max-arena-kb`, `--arena-underfill-kb`, and `--nb-filters` for targeted
  failure reproduction.
- Both runners export model artifacts once per run, then execute fixed-arena
  attempts via `HIL_spec(...)` with an explicit arena index
  (no `HIL_controller` auto-search).
- Device defaults come from the board wrapper. For current Portenta H7 runs the
  default candidate set includes `25 KiB`, which is useful for the lower end of
  recent arena sweeps.
- For model variants whose name starts with `trained`, pass
  `--trained-checkpoint /path/to/checkpoint.keras`.
- Failures are recorded and the sweep continues.
- Arenas with zero successful runs are marked as failures in the PNG plot.
