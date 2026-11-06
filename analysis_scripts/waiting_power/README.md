# Waiting-power experiments

This package builds and runs **new STM32 hardware experiments** and analyzes their recorded evidence. It does not supply archived results, datasets, models, binaries, serial identities, or assumed power values.

Run from the repository root, providing a candidate manifest with `candidates`, `timesteps`, and `input_dim` (candidate entries need `candidate_id`, `architecture`, and `clock_mhz`):

```sh
python -m analysis_scripts.waiting_power build --manifest /path/candidates.json --candidate candidate-name --config /path/device-and-data.yaml --output /path/new-study --clocks 80 160 --period-ms 250 --initialization-seed 7 --dry-run
```

Remove `--dry-run` to export initialized int8 PTQ weights once and compile continuous/ten-run/short-run variants for every clock. `--low-runs` selects the short cadenced window (1–9). The ten-run reference and firmware protocol are measurement-protocol constraints. The release period is frozen in study metadata and never inferred during measurement or analysis. `--firmware` selects the source used for protocol and provenance checks; the configured backend project must use the same protocol. Build requires the explicit configuration's calibration dataset, TensorFlow/NumPy and CREST dependencies, firmware, ST Edge AI, and STM32/ARM build tools. Missing dependencies are reported as failures; no substitutes are supplied.

```sh
python -m analysis_scripts.waiting_power run --study /path/new-study --rounds 4 --seed 17 --clocks 80 160 --device-id lab-device --dut-port /path/dut --harness-port /path/harness --board-ready
python -m analysis_scripts.waiting_power analyze --study /path/new-study --output /path/analysis
```

`run --dry-run` prints the schedule without opening devices. Real runs program app/weights and automatically update the harness run count. Grouped scheduling reduces harness reflashes; shuffled scheduling is also available. The grouped schedule ends with a ten-run block. After an interrupted run, check the harness configuration before another experiment.

All hardware runners using the same DUT/harness must use the **same `--device-id`** and lock root. `--lock-dir` or `CREST_DEVICE_LOCK_DIR` overrides the common per-user root. Accepted measurements are skipped on restart; schedule changes are refused. Exceptions and interrupted attempts require inspection. Retries remain bounded by `--max-attempts`.

The runner reuses compiled
variants, checks phase headers, clears native programming caches so app and
weights are programmed, and validates native acceptance flags, protocol, clock,
run count, STOP entries, and deadlines. CREST writes programming metadata into
the built workspace. Keep builds at their generated paths and rebuild after
changing their inputs.

Offline `analyze` reads recorded results only. It solves `W_k = k E + (T-C)[P_busy + (k-1)P_stop]`, with energies in mJ and times in seconds; output power is mW. The default active time uses DWT cycles with RTC fallback (`--latency-source rtc` selects RTC). It writes raw, per-round, and per-clock CSVs plus a JSON summary. Diagnostic profiling remains separate from these measurements.

```sh
python -m analysis_scripts.waiting_power --help
python -m pytest -q analysis_scripts/waiting_power/test_runner.py
```
