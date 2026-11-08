<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Appendix: candidate evaluation cost

Reconcile saved CREST client/server spans into additive host and HIL costs.
Python's standard library is sufficient. No training or hardware access occurs.

```sh
python analysis_scripts/appendix_evaluation_cost/prepare_inputs.py \
  --source /path/to/campaign --run-prefix experiment- \
  --arena-mode phase-attempts --output-dir /path/to/results
```

Inputs follow `client/runs/<prefix><condition>/results.json` and client/server
`timings/**/*.jsonl`, for `continuous`, `cadenced`, `flops`, and `memory`.
Use `arena-flags` for arena-search backends; it also reads saved server console
logs. Host names and candidate counts are inferred. Output must be outside the
source archive.

Outputs are `breakdown.json`, `proxy_times.csv`, `arena_search.json`, and detailed
audit files. For other log layouts, run `analyze.py`
and `summarize_campaign.py` separately; their `--help` lists input options.

Nested spans use exclusive time; remote/worker links compare durations without
subtracting clocks across hosts. Arena timeouts contain failed measurement and
protocol handling; compilation and programming remain separate.

For the two-board timing figure, prepare STM32 tables at the output root and Portenta
tables in `portenta_m7/`. Run
`paper_plots/plot_appendix_timing_analysis.py --data-dir /path/to/results` with
an external `--output-dir`. The analysis supports other panel sizes. Results describe these saved
campaigns, not repeated-run uncertainty or candidate-proposal overhead.
