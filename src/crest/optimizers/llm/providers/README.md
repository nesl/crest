# Completion providers

Providers translate a structured request into a completion response. They do not
validate candidate architectures, enqueue trials, or evaluate models. Those
responsibilities belong to the [LLM optimizer](../README.md) and shared runner.

| File | Responsibility |
| --- | --- |
| `provider.py` | `LLMProvider` protocol, `LLMRequest` and `LLMResponse` records, provider errors, fake/Chat Completions/Responses adapters, and factory selection. |
| `pricing.py`, `pricing.json` | Exact tariff lookup and bundled reviewed rates. |
| `__init__.py` | Package marker with no initialization side effects. |

## Choose a provider

Start with the [experiment quickstart](../README.md#run-an-experiment). The
provider-specific settings are:

| Provider | Endpoint and credentials |
| --- | --- |
| `openrouter` | Defaults to `https://openrouter.ai/api/v1`; uses `OPENROUTER_API_KEY` unless `api_key_env` specifies another variable. |
| `openai_compatible` | Set `base_url`, `model`, and `api_key_env` explicitly; the transport appends `/chat/completions`. |
| `openai_responses` | Set `base_url`, `model`, and `api_key_env` explicitly; the transport appends `/responses`. |
| `fake` | Supply an ordered `responses` list; no credentials or network access. See the [fixture examples](../EXAMPLES.md). |

`base_url` is the API root. `api_key_env` names an environment variable, not the
key itself. Keep actual credentials out of YAML, source files, and committed
shell scripts. The transport reads the variable at request time.

To enter a key without putting its value in shell history, use a hidden prompt
in zsh, then launch from the repository root in your CREST environment:

```zsh
read -rs 'OPENROUTER_API_KEY?API key: '; printf '\n'
export OPENROUTER_API_KEY
python src/nas_model_client.py --config /path/to/experiment.yaml --study-name llm-live
unset OPENROUTER_API_KEY
```

Trial budgets bound candidate evaluations. Provider spending is controlled
separately by `max_usd`, token limits, and configured token prices; see the
[configuration guide](../../../../config/README.md). Repair requests can make
additional provider calls.

## Exercise the completion interface

```python
from crest.optimizers.llm.providers.provider import FakeProvider, LLMRequest

provider = FakeProvider([{"candidates": [{"width": 4}]}])
response = provider.complete_json(LLMRequest(
    system_prompt="Return candidate parameters as JSON.",
    user_prompt="Propose a width.",
    prompt_version="v1",
))
```

This exercises only the completion boundary. The [generation example](../EXAMPLES.md)
also validates candidates against a search-space descriptor and writes provenance.

## Add an adapter

1. Implement `LLMProvider.complete_json` and return a normalized `LLMResponse`.
2. Add explicit factory selection and validation in `../config.py`.
3. Keep transport/authentication failures distinct from repairable response errors.
   Read credentials through environment variables; do not put secrets in config
   files or saved request provenance.
4. Test valid responses, malformed responses, and failure handling with mocked
   transport or fake responses before making live requests.

A provider is ready when configuration selects it correctly, focused tests cover
its success and failure paths, and it preserves the shared validation boundary.
Run the provider checks from the repository root:

```sh
python -m pytest -q test/test_llm_provider_prompt_ledger.py test/test_llm_optimizer_config.py
```

## Configure pricing and a spending cap

[`pricing.json`](pricing.json) contains reviewed USD input/output rates per
million tokens. Entries match **provider, base URL, model, and service tier**
exactly. An omitted tier selects `default`; an explicitly requested tier must
have its own entry. The bundled catalog includes standard OpenAI `gpt-6-luna` rates for Chat
Completions and Responses. Catalogs are dated snapshots, not automatically
refreshed price feeds. Other models, endpoints, and service tiers need a custom entry.

Add these settings to `optimizer.llm`, using your actual model and endpoint:

```yaml
provider: openai_responses
base_url: https://api.openai.com/v1
model: gpt-6-luna
api_key_env: OPENAI_API_KEY
max_usd: 5
max_input_tokens: 100000
max_output_tokens: 2000
# pricing_file: prices.json  # Optional custom catalog, relative to the YAML.
```

Export `OPENAI_API_KEY` in the terminal before launching your experiment. A cap
requires both token limits. A missing or ambiguous tariff stops initialization;
CREST never guesses prices for an unknown model or another endpoint.

For private endpoints or other models, set `pricing_file`. Relative paths resolve
beside the experiment YAML; direct Python configuration resolves them from the
working directory. The custom file replaces the bundled catalog and uses the
same schema. For example, these are **synthetic rates**, not a provider quote:

```json
{
  "schema_version": 1,
  "rates": [{
    "provider": "openai_compatible",
    "base_url": "https://provider.example/v1",
    "model": "my-model",
    "service_tier": "default",
    "currency": "USD",
    "input_usd_per_million_tokens": 1.0,
    "output_usd_per_million_tokens": 4.0,
    "source": "Your provider's tariff URL or contract reference",
    "verified_on": "2026-10-28"
  }]
}
```

For each capped study, `llm_optimizer/spend_state.json` saves the resolved rates
and the selected entry's source/date under `binding.pricing_snapshot`. Resuming
requires that tariff to match. Moving an unchanged file or editing unrelated
entries is allowed; changing the selected rates or provenance is rejected before
requests are made. Keep a copy of the original tariff to resume that study.

OpenAI's standard Luna tariff is $0.10 input / $0.50 output per million tokens,
with cache writes at $0.125. Above 272,000 input tokens, input/cache rates double
and output rates increase by 1.5x. The catalog records these rules and their source.

Flat accounting conservatively uses the highest input rate, including cache writes,
and selects the long-context rate when `max_input_tokens` exceeds the threshold.
Luna therefore reserves/settles at $0.125 / $0.50 for ceilings up to 272,000,
or $0.25 / $0.75 above it. The snapshot retains both the original tariff rules
and effective rates. Cached-input discounts are not deducted, so tracked cost
can exceed the invoice. These entries cover standard global processing only;
other tiers or regional premiums require a separately verified custom tariff.
Reservations bound cost using the token limits and settle against returned usage.
Unresolved attempts retain their reservation. The cap relies on correct rates
and the provider honoring the token limits.

Explicit inline rates and `pricing_source` remain supported for existing
configurations, but cannot be combined with a custom catalog. Without `max_usd`,
no tariff lookup or spending snapshot is created. To add a bundled tariff, verify
its exact endpoint/model/tier, record its source and date, and run
`python -m pytest -q test/test_llm_pricing.py test/test_llm_spending.py`.
