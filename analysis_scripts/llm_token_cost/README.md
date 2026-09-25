# LLM input-token and cost estimation

This standalone, standard-library-only utility calls OpenAI's
[`POST /v1/responses/input_tokens`](https://developers.openai.com/api/docs/guides/token-counting).
It does not generate candidates, train models, or touch hardware. No SDK or local
tokenizer dependency is required. Without `--count-api`, it only prepares JSON locally.

## Inputs

Pass one or more real CREST ledger files from
`models/<study>/llm_optimizer/requests/000001.request.json` using repeated
`--request` arguments. The script preserves the exact system/user message strings.
You can also supply a native Responses token-count request JSON with `input` and
optional counting fields; `--model` must match any model specified in that file.

The existing optimizer uses Chat Completions. Mapping its messages to Responses
counts the same text under a different API representation. Treat this as an
estimate for the existing optimizer, not an exact Chat Completions/OpenRouter bill.
The ledger conversion does not include transport-level JSON mode. For a Responses
experiment, include the intended `text.format` in a native request file.

A count from only the first empty-history prompt underestimates a run with populated
history. Include cold and representative full-history prompts (default runtime
window: ten trials), with the intended semantic context and batch size. The script
reports sample-minimum, equal-weight-mean, and sample-maximum scenarios; these are
not confidence bounds or a simulated history of the adaptive search.

If no ledger exists yet, obtain one from a fake-provider desktop smoke run, or
prepare a request using CREST's `build_candidate_request` and `LLMLedger` helpers.
That does not require a paid generation call. A hand-written prompt is accepted,
but its estimate only applies to that text. This utility does not load datasets or
automatically manufacture representative CREST histories.

## Usage

Run from the checkout root. Replace `MODEL_ID` with the exact OpenAI model ID;
OpenRouter's `openai/` prefix is not an OpenAI model ID.

```sh
python analysis_scripts/llm_token_cost/estimate.py \
  --request /path/to/000001.request.json \
  --request /path/to/000010.request.json \
  --model MODEL_ID --trials 150 --trials 250 --batch-size 5 \
  --report /tmp/crest-count-dry-run.json
```

Inspect the prepared report, then use the same command with `--count-api` after
setting `OPENAI_API_KEY` in the environment. Only the prepared payloads are sent to
OpenAI. Keys are never included in reports. Each input file makes one count call;
the script does not automatically retry or make projected generation calls.

Count once, then change assumptions without further API calls:

```sh
python analysis_scripts/llm_token_cost/estimate.py \
  --reuse-counts /tmp/crest-counts.json --model MODEL_ID \
  --trials 150 --trials 250 --batch-size 5 \
  --acceptance-rate 0.9 --extra-calls-per-round 0.1 \
  --repair-extra-input-tokens 100 --output-tokens-per-call 1000 \
  --report /tmp/crest-projection.json
```

Add `--input-usd-per-million` and `--output-usd-per-million` using current rates
for your intended model/provider, plus `--price-source` to record the URL and date.
Prices are deliberately not hardcoded. Without rates, costs remain null. Without an
output assumption, output usage and total cost remain null rather than implying zero.

`--output-tokens-per-call` means all billed output, including hidden reasoning
where applicable; counting visible candidate JSON cannot predict that quantity.
Use several values for sensitivity analysis, then calibrate against live usage.
Input counting also does not predict cache hits; the projection uses uncached rates.
It excludes fees/taxes and training/HIL costs.

## Projection

- `rounds = ceil(attempted_trials / (batch_size * acceptance_rate))`
- `expected_calls = rounds * (1 + extra_calls_per_round)`
- `input = expected_calls * sampled_prompt_tokens + rounds * extra_calls_per_round * repair_extra_input_tokens`
- `output = expected_calls * assumed_billed_output_tokens_per_call`
- `cost = (input * input_rate + output * output_rate) / 1,000,000`

Acceptance is the fraction enqueued after a generation round, not hardware
feasibility. Use attempted-trial budgets, not desired feasible completions. Defaults
assume every candidate is accepted and no repair calls. The final partial batch is
charged as a full batch, so this simplification can overestimate its size. Repair
responses use the same output-size assumption. Frequent failed batches with random
fallback should be modeled separately using observed candidates-per-round/call data.
For example, 150 attempts and five accepted candidates per round imply 30 generation
calls before repairs; 250 imply 50. These are planning assumptions, not measurements.

## Verification

```sh
python -m unittest discover -s analysis_scripts/llm_token_cost -p 'test_*.py' -v
```

Tests cover payload fidelity, endpoint selection, missing credentials, cost math,
partial batches, and invalid inputs. HTTP is mocked; these tests do not prove live
model/endpoint availability. Validate a real count with your account before relying
on the estimate. Reports contain prompt text; keep them with the experiment artifacts.
