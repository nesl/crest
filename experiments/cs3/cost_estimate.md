# CS3 offline API cost estimate

This is a planning estimate, not measured API usage or a billing guarantee. No
API/token-count requests, hardware evaluation, or training were performed.
Prices come from the prepared campaign YAML: $0.125/M input tokens (conservatively
including cache writes) and $0.50/M output tokens. They were not refreshed online
for this offline task. Output assumptions include hidden reasoning.

The 15 warm-up evaluations require no API calls. With five accepted candidates
per generation, summaries at 15/30/etc., and no unused summary after stopping:

| Total attempts | Generation calls | Memory calls | Total calls |
|---|---:|---:|---:|
| 200 | 37 | 13 | 50 |
| 207 (historical attempt count) | 39 | 13 | 52 |
| 300 (attempt cap) | 57 | 19 | 76 |

200 attempts assumes every trial is feasible; the actual target is 200 feasible
completions. Hardware failures and infeasibility can extend the run toward 300.
Partial accepted batches and repairs increase calls independently of hardware
feasibility. The table assumes regular batches and successful summaries; resumed
catch-up and irregular acceptance can change the memory schedule.

## Representative prompt estimate

The script uses the real AudioDSCNN descriptor, semantic/history helpers, and
candidate prompt builder with read-only historical trial records at attempts
15/30/100/190. Summary payloads mirror the runtime schema. These are size proxies,
not replayed outcomes for the live optimizer. Synthetic memory contains 5,999
structured characters (5,057 rendered summary characters). The first generation
sample has empty memory as a lower sizing case; later samples represent populated
memory. No tokenizer package/download or API was used: input estimates are
`ceil((system characters + user characters) / 4)`, not exact token counts.
Numeric JSON can tokenize differently; these ranges are sample ranges, not
confidence bounds. Configured full-limit costs below provide a separate stress
check. Semantic metadata was reconstructed from the archived config/shape.

| Proxy attempt | Generation input estimate | Memory input estimate |
|---|---:|---:|
| 15 | 6,425 | 5,408 |
| 30 | 7,645 | 6,816 |
| 100 | 7,733 | 6,912 |
| 190 | 7,746 | 6,921 |

Using the largest ordinary sampled input for each request type, with no repairs,
retries, or rejected candidates, the projected combined cost is:

| Total attempts | 1,024 output tokens/call | 4,096 output tokens/call | 8,192 output tokens/call |
|---|---:|---:|---:|
| 200 | $0.073 | $0.149 | $0.252 |
| 207 | $0.076 | $0.156 | $0.262 |
| 300 | $0.111 | $0.227 | $0.383 |

For 300 attempts with 4,096 output tokens/call, the sample range is $0.214–$0.227.
Adding 20% more calls of the same size gives about $0.273; doubling calls gives
$0.455. Repair prompts can be larger, and transport attempts with unknown usage
retain full reservations, so this simple multiplier is not a retry guarantee.
The 8,192-output column assumes every call uses the full output limit, but its
inputs remain heuristic estimates. Outputs include reasoning, not just JSON.

Ordinary byte-plus-1024 input guards range from 24,840 to 34,692, below the
configured 100,000 allowance. The original 100-visible-pending-record stress case
reached 128,062 and would stop locally. CS3 now uses the existing
`memory.max_pending_trials: 15` setting: the same backlog shows the newest 15
pending records outside the recent 15, reports 85 omitted records, and fits at
48,860 guard units (about 10,958 heuristic input tokens). All underlying trial
records and accepted memory remain saved. This also bounds each catch-up summary
chunk to 15 records; the regular summary cadence and ordinary cost table stay
unchanged. Exceptionally large individual records may still hit the input guard.

The regression uses 150 terminal trials and persisted summary suspension across
restarts: accepted memory and snapshots survive, no provider call is made while
suspended, and the newest pending records plus accurate omissions are retained.

Reproduce locally from the repository root (source files are not committed):

```sh
PYTHONPATH=src python experiments/cs3/estimate_cost_offline.py \
  --source-dir /path/to/archived/baseline \
  --config experiments/cs3/nas_config_cs3_luna.yaml \
  --output /tmp/cs3-cost-estimate.json
```

`cost_estimate.json` records samples and min/mean/max projections. Projection
arithmetic reuses `analysis_scripts/llm_token_cost/estimate.py`; its nine mocked,
offline tests pass. The archive is identified in `baseline_provenance.json`.

## Configured-limit stress cases

Each HTTP attempt reserves $0.016596, using the full 100,000 input allowance and
8,192 output ceiling. For 300 trials, 76 calls at those full limits total $1.261296.
One additional call for every generation and summary doubles that to $2.522592.
Six HTTP attempts per call (the configured initial attempt plus five retries)
would reserve $7.567776 before considering any content repairs, so the $5 cap
would stop the campaign. Missing usage retains the full reservation. An extreme
rejected-batch pattern that produces only one fallback candidate per round also
can exhaust the cap; no cost estimate guarantees completion.

Keep the $5 cap for now. It provides headroom for the ordinary planned call count,
but does not authorize spending or launch. Preserve accounting across restarts;
review actual input, billed output/reasoning, repair frequency, and memory usage
in a separately authorized bounded API smoke before trusting the typical-cost
estimate. Training/GPU time, hardware time, fees, and taxes are excluded.

## Measured API smoke (2026-10-14)

The user ran the bounded API-only smoke on this computer. Both requests passed;
no model training or hardware evaluation ran. The saved usage is in
`api_smoke_usage.json`; exact sent prompts are in `api_smoke_prompts.md`.
These two artifacts remain local to the preparation computer, outside this commit.

| Request | Input tokens | Billed output (including reasoning) | Configured accounted USD |
|---|---:|---:|---:|
| Memory | 2,237 | 35 (15 reasoning) | $0.000297125 |
| Five candidates | 2,714 | 460 (47 reasoning) | $0.000569250 |
| Total | 4,951 | 495 | $0.000866375 |

| Extrapolation | 200 total attempts | 300 total attempts |
|---|---:|---:|
| Same request sizes as smoke | $0.024925 | $0.038093 |
| Largest ordinary proxy inputs; measured smoke outputs | $0.055809 | $0.085070 |
| Proxy inputs, 4,096 output tokens/call, 20% extra calls | $0.179366 | $0.272731 |

The first row is optimistic: the smoke has synthetic outcomes, no campaign
semantic context, and only 35 output tokens for memory. Larger real summaries,
reasoning, repairs, and retries can change cost. Use roughly $0.10–$0.40 as an
ordinary planning range, retain the $5 local cap, and recalibrate separately for
memory and generation from real campaign usage. These are cost projections at
the configured rates, not measured campaign cost or a provider invoice.
