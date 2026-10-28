"""Import boundaries for shared search contracts and family decoding."""

import subprocess
import sys


def test_shared_search_and_decoder_imports_do_not_load_runtime_backends():
    """Offline contracts stay usable without loading an LLM or ML runtime."""
    subprocess.run(
        [sys.executable, '-c', '''
import sys
from crest.optimizers import context, history, search_space
from crest.model_families.audio_dscnn import decode
from crest.search import progress, training_randomness
assert decode.AUDIO_DSCNN_SEARCH_CHOICES
for prefix in ('tensorflow', 'keras', 'torch', 'openai', 'anthropic',
               'crest.optimizers.llm'):
    assert not any(name == prefix or name.startswith(prefix + '.')
                   for name in sys.modules), prefix
'''],
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
        check=True,
        capture_output=True,
        text=True,
    )
