<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# HIL request and serial protocol helpers

This package joins CREST's runtime configuration to the hardware controller and
provides the DUT/harness serial handshake used by device backends.

- [`runtime.py`](runtime.py) defines `HarnessConfig` and `CollectMetricsRequest`,
  builds requests from configuration, invokes `crest.hardware.HIL_controller`, and
  normalizes resource, latency, power, error, and cadenced-runtime fields.
- [`protocol.py`](protocol.py) defines protocol tokens, parsed handshake results,
  serial-line parsing, and synchronization routines for DUT and harness sessions.

Canonical imports include `from crest.hil.runtime import collect_metrics` and
`from crest.hil.protocol import HandshakeResult`. These are library helpers;
start the service through [`hil_server.py`](../../hil_server.py) using the
[configuration guide](../../config/README.md). See the
[device backend guide](../microcontrollers/README.md) for toolchain and board setup.

Protocol code uses PySerial; runtime collection also depends on the controller and
selected backend. Compile-only and physical HIL requests follow different paths:
physical runs require a DUT serial port, and energy-aware runs require harness
configuration. Missing telemetry is normalized to explicit sentinel values; it
must not be interpreted as a measured zero.

When extending the protocol, preserve firmware token compatibility and timeout/error
semantics. Exercise parsing with serial fakes before board tests; server integration
coverage is in [`test_hil_server.py`](../../../test/test_hil_server.py).
