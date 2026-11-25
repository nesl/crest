# How a trial works

A CREST study asks which models work well for a sensing task under a chosen deployment policy. Each trial binds a dataset, task, model family and target device. The same search loop can rank candidates by desktop proxies or by measurements from a board.

| Stage | What happens | Where to look |
| --- | --- | --- |
| Select | YAML names the dataset, task and model family. The runtime resolves them and checks the scoring policy against the metrics the task can produce. | [Configuration](../config/index.md) |
| Build | The family samples hyperparameters and builds a candidate. The dataset supplies inputs and the task defines the targets. | [Extension guides](../extend/index.md) |
| Measure | If the policy requires board measurements, the HIL server stages the candidate and returns metrics for the target and schedule. | [Measurement windows](measurement.md) |
| Evaluate | The policy checks pruning and feasibility. Eligible candidates train and receive task metrics. | [Scoring and constraints](../config/search.md) |
| Compare | The client records objectives, metrics and hyperparameters. Multi-objective studies can expose several trade-offs on a Pareto front. | [Read a result](../run/results.md) |

![CREST architecture showing the search client and hardware measurement path](../framework.png)

The diagram connects two jobs that can run on separate hosts. The training host owns search and evaluation. The device host runs the HIL server and needs the board toolchain, programming connection and measurement equipment required by the study.
