"""Regression checks for shared evidence conversion and ledger restart IDs."""
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from crest.optimizer_history import freeze_plain_data, to_plain
from crest.optimizers.llm.ledger import LLMLedger
from crest.optimizers.llm.provider import LLMRequest


def test_evidence_conversion_preserves_identity_and_sanitizes_metrics(tmp_path):
    """Keep immutable evidence strict while making nested ledger metrics JSON-safe."""
    @dataclass
    class Evidence:
        metrics: object

    original = Evidence(MappingProxyType({'path': tmp_path, 'value': np.float64('nan'),
                                          'count': np.int64(3), 'items': [1]}))
    frozen = freeze_plain_data(original)
    with pytest.raises(TypeError):
        frozen['metrics']['count'] = 4
    with pytest.raises(ValueError):
        json.dumps(to_plain(frozen), allow_nan=False)
    ledger = LLMLedger(tmp_path / 'ledger')
    ledger.record_event(original)
    saved = json.loads((ledger.root / 'optimizer_events.jsonl').read_text())
    assert saved == {'metrics': {'path': str(tmp_path), 'value': None, 'count': 3, 'items': [1]}}
    assert original.metrics['items'] == [1]


def test_request_numbers_scan_once_and_resume_after_gaps(tmp_path, monkeypatch):
    """Allocate unique IDs without repeated directory scans and recover persisted gaps."""
    ledger = LLMLedger(tmp_path)
    request = LLMRequest('system', 'user', 'v1', {})
    ledger.write_request(7, request)
    resumed = LLMLedger(tmp_path)
    def forbidden_scan(*args, **kwargs):
        """Reject rescanning after ledger initialization."""
        raise AssertionError('request directory scanned again')
    monkeypatch.setattr(Path, 'glob', forbidden_scan)
    assert resumed.next_request_id() == 8
    assert resumed.next_request_id() == 9
    resumed.write_request(12, request)
    assert resumed.next_request_id() == 13
