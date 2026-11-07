# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Import boundaries for shared search contracts and family decoding."""

import os
from pathlib import Path
import subprocess
import sys


def _checkout_env():
    """Make fresh interpreters resolve the same source checkout as pytest."""
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
    return env


def test_shared_search_and_decoder_imports_do_not_load_runtime_backends():
    """Offline contracts stay usable without loading an LLM or ML runtime."""
    subprocess.run(
        [sys.executable, '-c', '''
import sys
from crest.optimizers import context, history, search_space
from crest.model_families.audio_dscnn import decode, AUDIO_DSCNN_SEARCH_SEMANTICS
from crest.model_families.audio_dscnn.semantics import AUDIO_DSCNN_SEARCH_SEMANTICS as semantics
assert AUDIO_DSCNN_SEARCH_SEMANTICS is semantics
assert set(semantics) == set(decode.AUDIO_DSCNN_SEARCH_CHOICES)
assert "crest.model_families.audio_dscnn.family" not in sys.modules
from crest.search import progress, training_randomness
assert decode.AUDIO_DSCNN_SEARCH_CHOICES
for prefix in ('tensorflow', 'keras', 'torch', 'openai', 'anthropic',
               'crest.optimizers.llm'):
    assert not any(name == prefix or name.startswith(prefix + '.')
                   for name in sys.modules), prefix
'''],
        env=_checkout_env(),
        check=True,
        capture_output=True,
        text=True,
    )


def test_hil_protocol_import_does_not_load_device_runtime():
    """Serial protocol helpers do not initialize device backends or HIL runtime."""
    subprocess.run(
        [sys.executable, '-c', '''
import sys
from crest.hil import protocol
assert protocol.DUT_READY == 'DUT READY'
for prefix in ('crest.hil.runtime', 'crest.hardware', 'crest.microcontrollers',
               'tensorflow', 'keras'):
    assert not any(name == prefix or name.startswith(prefix + '.')
                   for name in sys.modules), prefix
'''],
        env=_checkout_env(),
        check=True,
        capture_output=True,
        text=True,
    )
