# Next-study prompt and Luna capability check

This test reads a saved study and uses the production full-history prompt,
Responses provider, allocation, decoded deduplication, and bounded refill path.
It never trains, constructs, uploads, or evaluates a model and never writes to
the input study. It does not launch or resume a campaign.

The new policy is opt-in through `history_mode: full`; existing configurations
retain their old prompt and behavior. The prepared
`nas_config_cs3_next_study.yaml` is not a deployed campaign configuration.
Complete the current study before any later hardware launch.

## Local offline preview and smoke

The following paths refer to the existing October 15 read-only audit snapshot.
Use a newly captured, consistent SQLite backup for fresher evidence. The prompt
context supplies the real campaign's MCU, task, objective, and search space.
`--trials` selects a chronological prefix; it does not invent later results.

```bash
cd /Users/jzales/.codex/worktrees/llm-token-estimator/fall-2026-nesl-crest
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR=/private/tmp/crest-pr-mpl
export CREST_NEXT_SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/crest-cs3-next-smoke.XXXXXX")"

/opt/anaconda3/envs/nesl-tinyodomex/bin/python \
  experiments/cs3/next_study_smoke.py \
  --history-db /private/tmp/cs3-oct15-audit/optuna.db \
  --prompt-context /private/tmp/cs3-oct15-audit/llm_optimizer/prompt_contexts.jsonl \
  --trials 100 --render-only \
  --output "$CREST_NEXT_SMOKE_ROOT/preview"

/opt/anaconda3/envs/nesl-tinyodomex/bin/python \
  experiments/cs3/next_study_smoke.py \
  --history-db /private/tmp/cs3-oct15-audit/optuna.db \
  --prompt-context /private/tmp/cs3-oct15-audit/llm_optimizer/prompt_contexts.jsonl \
  --trials 100 \
  --output "$CREST_NEXT_SMOKE_ROOT/offline"
```

No API key is needed for these commands. The default smoke substitutes mocked
HTTP responses behind the real Responses transport and accounting code. A pass
here verifies software behavior, not Luna's proposal quality.

## Optional paid capability test, run by the user

Only run this after reviewing the preview. It makes actual Luna requests using
the same production prompt and validation. It uses no hardware. Set the API key
in this terminal, then explicitly opt in:

```bash
# Bash: hidden prompt avoids putting the key in command history.
read -r -s -p "OpenAI API key: " OPENAI_API_KEY
printf '\n'
export OPENAI_API_KEY

/opt/anaconda3/envs/nesl-tinyodomex/bin/python \
  experiments/cs3/next_study_smoke.py \
  --history-db /private/tmp/cs3-oct15-audit/optuna.db \
  --prompt-context /private/tmp/cs3-oct15-audit/llm_optimizer/prompt_contexts.jsonl \
  --trials 100 --run-paid \
  --output "$CREST_NEXT_SMOKE_ROOT/paid"
```

Run the key-entry block in Bash (`bash --noprofile --norc` if necessary).
An existing exported key may be reused without entering it again. All output
directories must be new; an existing directory is refused to prevent accidental
repeated paid execution. Changing `--trials` to 15 or omitting it tests early or
latest saved evidence. Each separate invocation has its own cap.

Limits per invocation: **$0.05**, at most **four HTTP/content attempts**, no
transport retries, one shared malformed-response repair, two refills, and 8192
output tokens per request. Reservations are conservative and may stop the test
before four calls. Input uses the 120000 conservative byte-based
guard, not a tokenizer estimate. Prices are configured conservatively at
$0.125/M input and $0.50/M output; verify current pricing before paid execution.
Explicit no-breakpoint caching avoids speculative whole-prompt cache writes;
no cache savings are assumed.

## Optional separate cache probe

Add `--test-cache` to the render command to inspect the probe wire, or to the
paid command to make exactly two controlled requests. For example, reuse the
paths above with `--trials 100 --run-paid --test-cache` and a new output directory
such as `"$CREST_NEXT_SMOKE_ROOT/cache-paid"`. This mode replaces proposal
generation for that invocation. It uses the same $0.05 tracker, conservative
reservation, 120000 input guard, four-call ceiling, and zero transport retries.
No cache savings are assumed in reservations.

The probe follows the existing ranker replay `full_log` wire convention: a
fixed developer input block with an explicit breakpoint, stable context-derived
`prompt_cache_key`, and explicit 30-minute TTL. The two requests share that
identical production fixed context plus a diagnostic instruction to return the
small requested JSON object instead of candidates, and change only a short user
suffix. It does
not mark history windows or change the campaign's default no-breakpoint policy.
See the [official prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

`report.json` confirms caching only when the second real API response reports
positive `cached_tokens`. Zero reads are inconclusive, including prefixes below
the API's cache eligibility threshold. Offline `--test-cache` exercises the
provider and ledger but always reports `simulated_only`, never confirmation.
Inspect both raw responses and wire payloads in `requests/`.

## Review the result

- `system_prompt.txt` and `user_prompt.txt`: complete initial prompt.
- `wire_payload.json`: actual secret-free Responses request body.
- `requests/`: all subsequent requests, raw responses, and usage.
- `returned_candidates.jsonl`: accepted rationale, intent, assigned target,
  actual parent differences, and nearest-prior distances.
- `rejected_candidates.jsonl`: exact validation/duplicate feedback, when present.
- `report.json`: filled/partial/fallback outcome, allocation, coverage, calls,
  spending reservations, and cache counters.

A paid pass means Luna returned a full valid batch within these bounds. It does
not establish better search quality or hardware feasibility. Partial batches,
fallbacks, or exhausted caps are explicitly inconclusive. Inspect rationales
against Python-computed differences and the saved evidence, not just the pass
label. Retain artifacts before considering changes to the prompt or budget.

## Offline comparison metrics

`analysis_scripts/analyze_cs3_search.py --history-db SNAPSHOT.db --output NEW_DIR`
computes network-aggregated score trajectories, repeat variability, and resolved
cell coverage without API or hardware work. It groups all repeats, intentional
or accidental, using the same decoded identity and only evidence available at
each completion. Later repeats do not rewrite earlier curve points; the best
network mean can decrease while the historical best observation stays monotonic.

The CSV retains completed-only duration, all-attempt duration (including failures
and prunes), and elapsed wall time. Duration sums are proxies, not measurements of
GPU/hardware active time. Add `--ledger-dir SAVED_LLM_LEDGER` to report recorded
candidate and memory API latency separately; unresolved timings are flagged as
incomplete. The first coverage-complete prefix is recorded. Two networks per
resolved cell is block-count/stride breadth, not full search-space coverage; the
fixed phase allocation is a hypothesis, with no adaptive plateau mechanism.
Archive the input snapshot and output provenance when publishing comparisons.

Cache counters omitted by the API remain unknown. `cache_counter_completeness` and `cache_evidence_status` report whether read and write evidence is available; candidate-generation status is independent. Zero writes are confirmed only when every response supplies an explicit zero write counter. The back-to-back fixed-prefix probe does not establish reuse as history grows or retention across hardware-paced delays. Exact secret-free HTTP bodies are durably saved under `wire_requests/` before capped requests are sent, including requests rejected with HTTP 400.
