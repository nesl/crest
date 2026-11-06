<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# LLM candidate proposer

`LLMGeneratorOptimizer` in [`component.py`](component.py) implements the optimizer
contract for explicit raw-parameter proposals. Select it with
`optimizer.type: llm_generator`; start from the [configuration guide](../../../config/README.md).
Read the [optimizer system guide](../README.md) for the shared evaluation path,
validation boundaries, budgets, provenance, and restart rules.

| Modules | Responsibility |
| --- | --- |
| [`config.py`](config.py), [`provider.py`](provider.py) | Local settings validation and fake/OpenRouter/OpenAI-compatible/Responses transports |
| [`prompt_builder.py`](prompt_builder.py), [`objective_renderer.py`](objective_renderer.py) | Provider requests and objective descriptions |
| [`schemas.py`](schemas.py), [`enqueue.py`](enqueue.py) | Candidate schema/duplicate checks, repair feedback, and fallback candidate generation |
| [`history.py`](history.py), [`full_history.py`](full_history.py), [`phase_policy.py`](phase_policy.py) | Evidence selection, full-history representations, and proposal phase/allocation policy |
| [`memory.py`](memory.py), [`ledger.py`](ledger.py) | Derived memory and persisted request/response/proposal evidence |
| [`spending.py`](spending.py) | Provider usage/cost accounting and spending limits |
| [`rationale_migration.py`](rationale_migration.py) | Explicitly guarded migration of rationale-related study/ledger state |

Canonical class import: `from crest.optimizers.llm.component import LLMGeneratorOptimizer`.
The runner initializes this component with passive search context and supplies fresh
history and budget snapshots. Candidates still execute through the shared Optuna
objective; provider prose is not measurement evidence.

This layer uses Pydantic 2 and `addict`; provider HTTP transport uses the Python
standard library. Real providers need the configured credential environment variable
and network access. The fake provider supports offline fixtures without credentials
or hardware. Launch complete experiments through
[`nas_model_client.py`](../../../nas_model_client.py), rather than invoking these
modules as standalone search scripts.

For changes, preserve exact raw search-space validation, deterministic identity,
resume compatibility, and the separation between proposal provenance and sampled
trial parameters. The [`test`](../../../../test) suite contains focused `test_llm_*`
coverage for validation, providers, ledgers, memory, spending, warm-up, and runner
integration; use fake-provider cases before making paid provider calls.
