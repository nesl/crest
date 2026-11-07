# Completion providers

Providers translate a structured request into a completion response. They do not
validate candidate architectures, enqueue trials, or evaluate models. Those
responsibilities belong to the [LLM optimizer](../README.md) and shared runner.

| File | Responsibility |
| --- | --- |
| `provider.py` | `LLMProvider` protocol, `LLMRequest` and `LLMResponse` records, provider errors, `FakeProvider`, and factory selection. |
| `__init__.py` | Package marker with no initialization side effects. |

## Use a provider

Select `optimizer.type: llm_generator` and configure `optimizer.llm.provider` in
your experiment YAML. The supported adapter is `fake`: it consumes configured
responses in order without network access. The parent guide includes a runnable
generation example and campaign configuration. Direct callers can import:

```python
from crest.optimizers.llm.providers.provider import FakeProvider, LLMRequest

provider = FakeProvider([{"candidates": [{"width": 4}]}])
response = provider.complete_json(LLMRequest(
    system_prompt="Return candidate parameters as JSON.",
    user_prompt="Propose a width.",
    prompt_version="v1",
))
```

This exercises the completion boundary only; generation performs candidate
validation against the actual search-space descriptor.

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
