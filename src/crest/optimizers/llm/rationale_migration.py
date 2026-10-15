"""Explicit offline migration of one stopped SQLite LLM study's rationale contract.

The caller must derive expected/active signatures with the normal runner identity
code. This helper never relaxes runner resume checks. Stop the sole campaign
writer and recover RUNNING trials first. Existing WAITING proposals are retained
and identified as legacy, without manufacturing retrospective explanations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import sqlite3

SIGNATURE = 'crest_optimizer_signature'
AUDIT = 'crest_rationale_contract_migration'


def validate_transition(before, after):
    """Permit precisely the legacy-to-required-rationale contract addition.

    Parameters
    ----------
    before, after : dict
        Signatures derived by normal runner identity code for old/new configs.

    Raises
    ------
    ValueError
        Source is not schema version 1 or any unrelated identity field changes.
    """
    if not isinstance(before, dict) or before.get('optimizer') != 'llm_generator' or before.get('version') != 1:
        raise ValueError('Expected a version 1 llm_generator signature')
    config = before.get('config')
    if (not isinstance(config, dict) or 'require_rationale' in config
            or type(config.get('candidate_schema_version')) is not int
            or config['candidate_schema_version'] != 1):
        raise ValueError('Source signature must have the legacy rationale contract')
    expected = {**before, 'config': {**config, 'require_rationale': True, 'candidate_schema_version': 2}}
    if json.dumps(after, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('Only require_rationale=true and candidate_schema_version=2 may change')


def migrate(database, ledger, study_name, before, after, backup, *, confirm_stopped=False):
    """Back up full campaign artifacts, then atomically update identity and audit.

    Parameters
    ----------
    database, ledger, backup : str or pathlib.Path
        Existing SQLite file, associated full ledger, and new backup directory.
    study_name : str
        Exact Optuna study name.
    before, after : dict
        Expected source and normal-runner-derived target signatures.
    confirm_stopped : bool
        Operator assertion that every campaign writer has stopped.

    Returns
    -------
    dict
        Audit record also atomically saved alongside the updated signature.

    Notes
    -----
    No training, API calls, trial state changes, memory changes, or spend changes
    occur. A stopped operator assertion and zero RUNNING records are mandatory;
    SQLite cannot detect an idle external campaign process, so the operator must
    ensure every writer is stopped. The backup path must not already exist.
    """
    validate_transition(before, after)
    database, ledger, backup = map(lambda p: Path(p).resolve(), (database, ledger, backup))
    if not confirm_stopped:
        raise ValueError('Explicit confirmation that all campaign writers stopped is required')
    if not database.is_file() or not ledger.is_dir():
        raise ValueError('Existing local SQLite database and full ledger directory are required')
    if backup.exists() or backup == ledger or ledger in backup.parents:
        raise ValueError('Backup must be a new directory outside the ledger')
    # mode=rw prevents silently creating a missing database.
    with sqlite3.connect(database.as_uri() + '?mode=rw', uri=True) as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT study_id FROM studies WHERE study_name=?', (study_name,)).fetchone()
        if row is None:
            raise ValueError('Study not found')
        study_id = row[0]
        row = conn.execute('SELECT value_json FROM study_user_attributes WHERE study_id=? AND key=?', (study_id, SIGNATURE)).fetchone()
        if row is None or json.dumps(json.loads(row[0]), sort_keys=True, allow_nan=False) != json.dumps(before, sort_keys=True, allow_nan=False):
            raise ValueError('Stored identity differs from expected source signature')
        states = conn.execute('SELECT number,state FROM trials WHERE study_id=?', (study_id,)).fetchall()
        if any(state == 'RUNNING' for _, state in states):
            raise ValueError('RUNNING trials must be resolved before offline migration')
        if conn.execute('SELECT 1 FROM study_user_attributes WHERE study_id=? AND key=?', (study_id, AUDIT)).fetchone():
            raise ValueError('Migration already recorded')
        backup.mkdir(parents=True)
        # Back up through another read connection while the write lock protects
        # the snapshot; copying the main file would lose committed WAL content.
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as source, sqlite3.connect(backup / 'study.sqlite3') as destination:
            source.backup(destination)
        shutil.copytree(ledger, backup / 'llm_optimizer', symlinks=True)
        event = {'schema_version': 1, 'study_name': study_name,
                 'migrated_at_utc': datetime.now(timezone.utc).isoformat(),
                 'existing_trial_numbers': sorted(n for n, _ in states),
                 'existing_terminal_trial_numbers': sorted(n for n, state in states if state in {'COMPLETE', 'PRUNED', 'FAIL'}),
                 'existing_trial_state_counts': {state: sum(s == state for _, s in states) for state in sorted({s for _, s in states})},
                 'before': before, 'after': after,
                 'legacy_waiting_trial_numbers': [n for n, state in states if state == 'WAITING'],
                 'legacy_rationale_status': 'not_recorded',
                 'legacy_queue_policy': 'grandfather_existing_proposals',
                 'backup_directory': str(backup)}
        (backup / 'migration.json').write_text(json.dumps(event, indent=2, sort_keys=True) + '\n')
        conn.execute('UPDATE study_user_attributes SET value_json=? WHERE study_id=? AND key=?', (json.dumps(after), study_id, SIGNATURE))
        conn.execute('INSERT INTO study_user_attributes (study_id,key,value_json) VALUES (?,?,?)', (study_id, AUDIT, json.dumps(event)))
        conn.commit()
        return event


def main():
    """Execute the explicit offline migration command.

    Returns
    -------
    None
        Prints the persisted migration audit record on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('database', 'ledger', 'study-name', 'before-signature', 'active-signature', 'backup'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--confirm-stopped', action='store_true')
    args = parser.parse_args()
    result = migrate(args.database, args.ledger, args.study_name,
                     json.loads(Path(args.before_signature).read_text()),
                     json.loads(Path(args.active_signature).read_text()), args.backup,
                     confirm_stopped=args.confirm_stopped)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Migration recorded: %s", json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
