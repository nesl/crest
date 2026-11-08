<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->
# Appendix: recorded trial outcomes

These read-only tools classify saved search trials into disjoint final outcomes.
They do not run training, contact hardware, or count measurement retries and
proxy replays as new search trials. Supply external artifacts and write outputs
outside the checkout.

- `periodic.py MANIFEST OUTPUT_DIR` audits periodic campaigns and separately
  records paired-measurement eligibility and configured population exclusions.
- `recorded.py --config CONFIG --output-dir OUTPUT_DIR` audits legacy SQLite
  databases or saved JSON exports using recorded operational eligibility.
- `scored.py --config CONFIG --output-dir OUTPUT_DIR` audits scored campaigns
  through an explicit completion endpoint.
- `combine.py --input CSV [CSV ...] --output-dir OUTPUT_DIR` combines native
  trial tables and rejects duplicate identities, host work, and replay scopes.
- `verify_paper.py --input CSV [CSV ...]` runs optional reference-count and population
  checks separately from the generic classifiers.

Each native `trial_outcomes.csv` contains `case_study`, `campaign`, `trial`, and
`outcome`, plus source evidence. Outcomes are `measured_passed`,
`measured_violation`, and `not_deployed_or_incomplete`. A stored COMPLETE state
alone does not establish measurement success. Missing evidence stays explicit;
historical success conventions do not imply modern validity checks ran.

Legacy configuration is a JSON list with `campaign`, `case_study`, and either
`db` or `export` plus `export_campaign`. Optional `study_id` selects an Optuna
study; databases with multiple studies require it. `trial_min` is inclusive and
`trial_max` exclusive. `host_only` campaigns appear in summary JSON and are
excluded from the native CSV. Saved exports may be JSON, gzip JSON, or
base64-encoded gzip JSON.

For periodic and scored configuration fields, see their module documentation.
Keep case-study labels consistent across input configurations. Periodic
`campaign_passed`, `paired_eligible`, `paired_included`, and `model_eligible`
flags describe distinct populations; the final campaign outcome is assigned
before paired-population filtering.
