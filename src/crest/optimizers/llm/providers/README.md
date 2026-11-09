# Completion providers

Providers turn prompts into completion responses. The [LLM optimizer](../README.md)
validates candidates and passes them to the shared search runner.

| File | Responsibility |
| --- | --- |
| `provider.py` | Public imports and `build_provider` factory |
| `records.py` | Request/response records, `LLMProvider` protocol, errors, and `FakeProvider` |
| `chat.py`, `responses.py` | Chat Completions and Responses adapters |
| `transport.py` | Authentication, retries, and per-attempt spend accounting |
| `utils.py` | JSON-mode defaults and pricing lookup |
| `pricing.json` | Bundled conservative accounting rates |

## Choose a provider

| Provider | Configuration |
| --- | --- |
| `openrouter` | Defaults to `https://openrouter.ai/api/v1` and `OPENROUTER_API_KEY`; set `model`. |
| `openai_compatible` | Set `base_url`, `model`, and `api_key_env`; appends `/chat/completions`. |
| `openai_responses` | Set `base_url`, `model`, and `api_key_env`; appends `/responses`. |
| `fake` | Supply an ordered `responses` list; no credentials or network access. |

`base_url` is the API root. `json_response_mode` defaults to `true` for OpenRouter
and Responses, and `false` for generic Chat Completions endpoints.

## Configure pricing and a spending cap

Add this optimizer block to a complete [experiment configuration](../../../../config/README.md):

```yaml
optimizer:
  type: llm_generator
  llm:
    provider: openai_responses
    base_url: https://api.openai.com/v1
    model: gpt-6-luna
    api_key_env: OPENAI_API_KEY
    max_usd: 5
    max_input_tokens: 100000
    max_output_tokens: 2000
    # pricing_file: prices.json  # Optional custom catalog, relative to this YAML.
```

`api_key_env` names an environment variable; keep the key out of YAML and source
control. In zsh, enter it without adding its value to shell history, then run:

```zsh
read -rs 'OPENAI_API_KEY?API key: '; printf '\n'
export OPENAI_API_KEY
python src/nas_model_client.py --config /path/to/experiment.yaml --study-name llm-live
unset OPENAI_API_KEY
```

A spending cap requires both token limits and matching prices. The bundled
[`pricing.json`](pricing.json) covers OpenAI `gpt-6-luna`; its conservative rates
can exceed the actual bill. For another model, endpoint, or service tier, copy
that file and set `pricing_file`. A custom catalog replaces the bundled one;
entries must match provider, base URL, model, and tier (`default` when omitted).

Capped studies save their selected rates in `llm_optimizer/spend_state.json`;
resume requires the same rates and provenance. Failed or interrupted requests
retain their reservation because they may have been billed. Recorded charges
round upward to 12 decimal places in USD. Without `max_usd`, spending is not tracked.

## Try the interface offline

```python
from crest.optimizers.llm.providers.provider import FakeProvider, LLMRequest

provider = FakeProvider([{"candidates": [{"width": 4}]}])
response = provider.complete_json(LLMRequest(
    system_prompt="Return candidate parameters as JSON.",
    user_prompt="Propose a width.",
    prompt_version="v1",
))
```

For candidate validation and ledger output, see the [generation examples](../EXAMPLES.md).

## Add an adapter

1. Implement `LLMProvider.complete_json`, returning a normalized `LLMResponse`.
   Live adapters can reuse `HTTPProvider` in `transport.py` for authentication,
   retries, and accounting.
2. Add factory selection in `provider.py` and validation in `../config.py`.
3. Distinguish transport/authentication failures from repairable response errors.
   Keep credentials out of saved request and response evidence.
4. Test valid and malformed responses, retries, and spending limits with mocked
   transport. From the repository root, run:

   ```sh
   python -m pytest -q test/test_llm_openai_provider.py test/test_llm_responses_provider.py test/test_llm_provider_defaults.py test/test_llm_pricing.py test/test_llm_spending.py
   ```

An adapter is ready when configuration selects it, the checks pass, and generation
still validates candidates before returning them to the runner.
