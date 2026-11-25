# Objectives and feasibility

Choose a scalar score when one ranking expresses the trade-off you care about. Use multiple objectives when you want to inspect a Pareto front. The shipped desktop OxIOD example minimizes validation error and model FLOPs separately.

```yaml
nas:
  score:
    type: multi-objective
    params:
      objectives:
        - metric: rmse_total
          direction: minimize
        - metric: flops
          direction: minimize
  prune:
    rules: []
```

## Decide what happens to an unsuitable candidate

**Pruning rules** can stop a trial before fitting. **Feasibility rules** expose deployment constraints to the sampler. A candidate that exceeds a deadline can be marked infeasible while keeping its result available for analysis. Each signed constraint is feasible at or below zero.

With `nas.feasibility.train_if_infeasible: false`, an infeasible candidate skips training and still consumes an attempt. When training is enabled for infeasible candidates, their objective values can be measured while the constraint violation stays recorded.

??? info "Scoring, pruning and feasibility fields"
<!-- config-section `nas` -->

[Inspect trial outputs](../run/results.md)
