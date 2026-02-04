# Single HIL Run

This folder provides a minimal, single-pass HIL sanity check. It builds a
standard TinyODOM model, runs the HIL controller once, and prints the returned
metrics (latency/energy/RAM/flash/etc.).

## Usage

```bash
python analysis_scripts/hil_single_run/run_single_hil.py
```

Optional overrides:

```bash
python analysis_scripts/hil_single_run/run_single_hil.py --input-mode standard
python analysis_scripts/hil_single_run/run_single_hil.py --input-mode uniform
python analysis_scripts/hil_single_run/run_single_hil.py --output analysis_scripts/hil_single_run/last_run.json
```
