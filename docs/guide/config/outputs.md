# Outputs and connections

Set `outputs.artifact_stem` to distinguish a study's artifacts. CREST combines that stem with the device name to derive the model and checkpoint names. Use a fresh output location when changing a study's objectives or deployment assumptions.

| Setting | What to check |
| --- | --- |
| `outputs.models_dir` | Root for model artifacts |
| `outputs.candidate_dir` | Candidate output location |
| `outputs.log_file_name` | Name of the trial CSV |
| `network.host`, `network.port` | Endpoint the training host can reach for HIL |
| `logging.level` | Start with `INFO` to follow trial progress |

??? info "Artifact naming and derived paths"
<!-- config-section `outputs` -->

??? info "Socket settings"
<!-- config-section `network` -->

[Read a result](../run/results.md)
