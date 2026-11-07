# Completion providers

Providers translate a structured request into a completion response. They do not
validate candidate architectures, enqueue trials, or evaluate models. Those
responsibilities belong to the [LLM optimizer](../README.md) and shared runner.

| File | Responsibility |
| --- | --- |
| `provider.py` | Public imports and `build_provider` factory selection. |
| `records.py` | `LLMProvider` protocol, request/response records, errors, and the offline `FakeProvider`. |
| `chat.py`, `responses.py` | Live adapters that build and interpret each API's payloads. |
| `transport.py` | Shared authentication, HTTP retries, and per-attempt spend accounting. |
| `utils.py` | JSON-mode defaults and pricing-catalog lookup for provider configuration. |
| `pricing.json` | Bundled conservative accounting rates. |
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

`json_response_mode` defaults to `true` for OpenRouter and Responses, and
`false` for generic Chat Completions endpoints. Set it explicitly to override
the default for your model.

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
2. Add factory selection in `provider.py` and validation in `../config.py`.
   Live adapters can reuse `HTTPProvider` from `transport.py` for authentication,
   retries, and accounting; each adapter builds and normalizes its own payloads.
3. Keep transport/authentication failures distinct from repairable response errors.
   Read credentials through environment variables; do not put secrets in config
   files or saved request provenance.
4. Test valid responses, malformed responses, and failure handling with mocked
   transport or fake responses before making live requests.

A provider is ready when configuration selects it correctly, focused tests cover
its success and failure paths, and it preserves the shared validation boundary.
Run the provider checks from the repository root:

```sh
python -m pytest -q test/test_llm_openai_provider.py test/test_llm_responses_provider.py test/test_llm_provider_defaults.py test/test_llm_provider_prompt_ledger.py
```

## Configure pricing and a spending cap

[`pricing.json`](pricing.json) contains conservative USD input/output accounting
rates per million tokens. These bound recorded spending and can exceed the
provider's actual bill. Entries match **provider, base URL, model, and service tier**
exactly. An omitted tier selects `default`; an explicitly requested tier must
have its own entry. The bundled catalog covers OpenAI `gpt-6-luna` through Chat
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

Each entry supplies explicit accounting rates. The bundled Luna
entry uses a conservative $0.25 input / $0.75 output per million tokens for every
request, covering standard global cache-write and long-context charges. Its
`accounting_notes` records these assumptions. Shorter or discounted requests
can cost less than the recorded amount; regional premiums and other service
tiers need their own verified entry.

`utils.py` provides JSON-mode defaults and loads, matches, and validates pricing entries. `spending.py` owns
cost arithmetic and saving that complete entry with the study. A custom catalog
must also supply explicit rates; cache or context multiplier rules are rejected.
Changing from a previous tariff changes the saved binding and requires a new
study, rather than silently repricing existing usage.

Reservations bound cost using the token limits and settle against returned usage.
Unresolved attempts retain their reservation. Charges and reservations are rounded
up to 12 decimal places in USD ($0.000000000001); totals add these amounts exactly.
For example, a $0.00000025 charge is preserved exactly. Rounding adds less than
$0.000000000001 per calculated charge, and the configured spending cap is never
rounded upward. The cap relies on correct rates and the provider honoring the
token limits.

Explicit inline rates and `pricing_source` remain supported for existing
configurations, but cannot be combined with a custom catalog. Without `max_usd`,
no tariff lookup or spending snapshot is created. To add a bundled tariff, verify
its exact endpoint/model/tier, record its source and date, and run
`python -m pytest -q test/test_llm_pricing.py test/test_llm_spending.py`.
