# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Import boundaries for shared search contracts and family decoding."""

import subprocess
import sys


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
