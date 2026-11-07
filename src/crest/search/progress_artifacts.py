# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Shared eligibility and file-writing rules for progress snapshots."""
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def as_datetime(value):
    """Accept either an Optuna timestamp or its SQLite ISO representation."""
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def eligible_trial(trial, feasibility_enabled):
    """Require a completed, unrejected result under the study's feasibility policy."""
    attrs = trial['attrs']
    status = attrs.get('feasibility_status')
    eligible = (status == 'feasible' if feasibility_enabled else
                status in (None, '', 'not_evaluated', 'feasible'))
    return trial['state'] == 'COMPLETE' and eligible and not attrs.get('pruned', False)


@contextmanager
def atomic_output(path, mode):
    """Replace a complete artifact so readers never open a half-written plot."""
    temporary = None
    try:
        with NamedTemporaryFile(mode=mode, dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
