<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Shared search utilities

Search execution refreshes progress after every completed attempt and when a study resumes, under `<models_dir>/<study>/progress/`.
Failed, pruned, infeasible, and nonfinite attempts retain their attempt positions and time accounting, but cannot improve incumbents or hypervolume.

[`progress.py`](progress.py) writes scalar score-progress CSV and PNG/PDF charts against attempts and elapsed time.
Multiobjective reports use only hypervolume, with `hypervolume_progress.csv`, `hypervolume_history.png`/`.pdf`, `hypervolume_vs_time.png`/`.pdf`, and `hypervolume_reference.json`.
An `error.txt` in the progress directory indicates a reporting failure.
Existing plots may be stale even though search results remain available.

Hypervolume uses a fixed reference point in objective order and original units.
Configure `outputs.hypervolume_reference_point` for a meaningful comparison target.
The selected reference is persisted for resume.
Without a configured reference, the first eligible finite objective vector freezes a monitoring reference with a 10% margin in each objective's worse direction.
This fallback is for monitoring, not a substitute for a domain-chosen benchmark.
It does not move as results change.
Points outside it need not improve hypervolume.
Comparable studies must use the same reference, objective definitions, units, directions, and eligibility rules.
Only feasible, finite COMPLETE vectors contribute.
Pruned attempts cannot contribute partial results.

Generate a progress report from a saved study database.

```python
from pathlib import Path
from crest.search.progress import read_trials, write_progress

trials = read_trials(Path("results/optuna.db"), "my-study")
write_progress(trials, Path("report"), "my-study")
# For multiobjective reports, optionally pass reference_point=[...] in objective order.
```

Use an existing database and exact study name.
This writes to `report/progress/`.
The SQLite reader uses a read-only snapshot.
Feasibility defaults to requiring affirmative evidence.
Set `feasibility_enabled=False` only for a study configured without feasibility rules.
For in-memory multiobjective records, pass objective `directions` to `write_progress`.
SQLite records carry their directions.

[`training_randomness.py`](training_randomness.py) validates `training.random_seed` and `training.deterministic_ops`, then resets Python/NumPy/TensorFlow randomness before evaluation.
The same configured seed is reused across architectures and final fits.
Deterministic operations require a compatible runtime.
A seed alone does not guarantee cross-platform reproducibility.

See [optimizer execution](../optimizers/README.md) and [configuration](../../config/README.md).
Targeted coverage lives in [`test_progress.py`](../../../test/test_progress.py) and [`test_training_randomness.py`](../../../test/test_training_randomness.py).
