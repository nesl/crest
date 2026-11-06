<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Shared search utilities

These helpers support search execution and reporting across optimizers.

[`progress.py`](progress.py) reads a read-only SQLite snapshot of an Optuna study,
adapts in-memory trials, and writes score-progress CSV plus PNG/PDF charts.
The SQLite reader requires exactly one maximized objective. Incumbents use finite,
eligible COMPLETE scores; failed, pruned, and infeasible terminal attempts retain
their positions. Plotting imports Matplotlib lazily.

After installing CREST, an offline report can be generated from Python:

```python
from pathlib import Path
from crest.search.progress import read_trials, write_progress

trials = read_trials(Path("results/optuna.db"), "my-study")
write_progress(trials, Path("report"), "my-study")
```

Replace the database and exact study name with an existing study. The output goes
under `report/progress/`. The default requires affirmative feasibility evidence;
set `feasibility_enabled=False` only for studies configured without feasibility rules.

[`training_randomness.py`](training_randomness.py) validates `training.random_seed`
and `training.deterministic_ops`, then resets Python/NumPy/TensorFlow randomness
through TensorFlow before evaluation. The same configured seed is reused across
architectures and final fits. Deterministic operations require a compatible runtime;
a seed alone does not guarantee cross-platform reproducibility.

See [optimizer execution](../optimizers/README.md) and
[configuration](../../config/README.md) for integration. Changes should preserve
trial eligibility and time accounting; targeted coverage lives in
[`test_progress.py`](../../../test/test_progress.py) and
[`test_training_randomness.py`](../../../test/test_training_randomness.py).
