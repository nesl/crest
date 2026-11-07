# Completion providers

Providers translate a structured request into a completion response. They do not
validate candidate architectures, enqueue trials, or evaluate models. Those
responsibilities belong to the [LLM optimizer](../README.md) and shared runner.

| File | Responsibility |
| --- | --- |
| `provider.py` | `LLMProvider` protocol, `LLMRequest` and `LLMResponse` records, provider errors, fake/Chat Completions/Responses adapters, and factory selection. |
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
