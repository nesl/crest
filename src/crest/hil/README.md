<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# HIL request and serial protocol helpers

This package joins CREST's runtime configuration to the hardware controller and provides the DUT/harness serial handshake used by device backends.

- [`runtime.py`](runtime.py) builds measurement requests and normalizes collected metrics.
- [`protocol.py`](protocol.py) handles serial parsing and DUT/harness synchronization.

See the [configuration guide](../../config/README.md) to run [`hil_server.py`](../../hil_server.py), and the [device backend guide](../microcontrollers/README.md) for board setup.
