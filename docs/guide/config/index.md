# Choose a configuration

Start with the closest shipped YAML. Its scoring policy determines which measurements the run needs. For a first run on a workstation, the FLOPs/RMSE example disables both board measurement and compile-only requests.

<!-- include: src/config/README.md#Example configs -->

## Find a setting

- [Workload and training](workload.md) covers component names, quantization and training budgets.
- [Target and schedule](target.md) covers HIL, compile-only requests and board-specific options.
- [Objectives and feasibility](search.md) explains how CREST ranks candidates and handles deadline misses.
- [Outputs and connections](outputs.md) covers artifact names, HIL endpoints and logging.

[Run your first study](../run/first-study.md)
