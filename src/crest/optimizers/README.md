# Optimizer contracts

`optuna` is the default optimizer. Shared search-space descriptors and immutable proposal/history records let registered components propose candidates through the same runner. The runner validates complete explicit batches, reserves trial budgets, and checks identity before resuming studies.

See `interfaces.py`, `pipeline_types.py`, and the optimizer runner/resume tests for the contract.
