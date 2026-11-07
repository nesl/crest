# LLM study maintenance tools

These commands operate on saved studies. They are separate from candidate
generation and are not needed to configure or run a new experiment. Normal
proposal generation lives in [`generation.py`](../generation.py); the optimizer
entry point is [`component.py`](../component.py).

| File | Purpose |
| --- | --- |
| `rationale_migration.py` | Migrate one stopped SQLite study to the required-rationale contract, with database/ledger backups and an audit record. |
| `__init__.py` | Package marker; importing it does not run maintenance. |

## Migrate a rationale contract

Use this only when deliberately upgrading an existing study's rationale contract.
It does not provide general optimizer-identity migration or bypass resume checks.

1. Stop the study's sole writer and resolve any RUNNING trials.
2. Derive the expected source and active signatures using the normal optimizer
   identity code and save each as JSON. The tool verifies the stored source
   identity and rejects unrelated identity changes.
3. Choose a new backup directory outside the ledger. The tool backs up the SQLite
   database and full ledger before recording the transition.
4. From the repository root in the CREST environment, run:

   ```sh
   PYTHONPATH=src python -m crest.optimizers.llm.tools.rationale_migration \
     --database /path/to/study.sqlite3 \
     --ledger /path/to/llm_optimizer \
     --study-name YOUR_STUDY_NAME \
     --before-signature /path/to/before.json \
     --active-signature /path/to/active.json \
     --backup /path/to/new-backup-directory \
     --confirm-stopped
   ```

Existing WAITING proposals retain their original evidence; the tool does not
invent explanations for older candidates. Check the backup and migration audit
record before resuming with the active configuration. Use `--help` to inspect the
command without changing a study.

## Add or change a tool

Keep maintenance commands explicit and independently invocable with `python -m`.
Document their inputs, preconditions, mutations, and recovery artifacts here.
They must not run as an import side effect or become part of ordinary generation.

A change is complete when its CLI help works, tests cover successful operation
and rejected preconditions, and backup/audit behavior remains verified. Run:

```sh
python -m pytest -q test/test_llm_rationale_migration.py
```
