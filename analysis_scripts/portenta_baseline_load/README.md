# Portenta Baseline Load Test

This package measures baseline timing/energy on Portenta H7 using two synthetic
workloads and the existing harness (INA228) telemetry path.

Workloads (10 iterations each):

1. `heavy`: each iteration busy-loops for ~200 ms.
2. `sleep`: each iteration calls `delay(200)`.

The runner executes both workloads on both cores:

- `cm7 + heavy`
- `cm7 + sleep`
- `cm4 + heavy`
- `cm4 + sleep`

## Why this test exists

This isolates platform-level behavior from model inference behavior. It helps
establish whether energy/timing deltas are dominated by:

- active CPU-heavy work, or
- mostly-idle/sleep-like runtime behavior.

## Wiring assumptions

- DUT and harness use shared pins:
  - trigger: D2
  - arm: D3 (active-low arming)
- Harness firmware uses the existing TinyODOM protocol (`PING/READY`, `DONE`,
  `runs:`, power telemetry keys, `harness timer output:`).

## Scripts and sketches

- `portenta_heavy_10x200ms.ino`
- `portenta_sleep_10x200ms.ino`
- `run_portenta_baseline_load.py`

## Usage

Default run (both cores, one repeat each workload):

```bash
python analysis_scripts/portenta_baseline_load/run_portenta_baseline_load.py
```

Example with explicit ports and repeat count:

```bash
python analysis_scripts/portenta_baseline_load/run_portenta_baseline_load.py \
  --dut-port /dev/ttyACM0 \
  --harness-port /dev/ttyACM1 \
  --repeats 3 \
  --cores cm7 cm4 \
  --output-json analysis_scripts/portenta_baseline_load/results/latest.json \
  --output-csv analysis_scripts/portenta_baseline_load/results/latest.csv
```

Skip harness reflashing (use currently flashed harness firmware):

```bash
python analysis_scripts/portenta_baseline_load/run_portenta_baseline_load.py \
  --skip-harness-flash
```

## Output schema

CSV has one row per attempt with key fields:

- `core`, `workload`, `repeat`, `runs`
- `latency_ms_per_iter`
- `energy_mj_per_iter`
- `avg_power_mw`, `avg_current_ma`, `bus_voltage_v`, `idle_power_mw`
- `error_code`, `error_label`

JSON includes:

- `metadata`
- `attempts`
- `aggregates` grouped by `core x workload` with mean/std/min/max

## Caveats

- This is a synthetic baseline test, not an inference benchmark.
- `delay(200)` may map to light sleep/idle behavior depending on platform
  sleep locks and active peripherals.
- CM4 telemetry is still sourced from harness path (expected for this repo).

