# LLM candidate generation

The `llm_generator` optimizer proposes validated raw candidates through the shared runner. Bounded recent history, best-trial anchors, experiment context, fixed warm-ups, rationales, repair, fallback, and request provenance are supported.

This layer uses the deterministic `fake` provider and configured responses for offline execution.

Persistent memory and full-history policies are introduced in the next policy layer; bounded execution uses `memory.enabled=false`.
