"""Readiness framing regressions with captured startup corruption."""
from unittest.mock import Mock

from crest.hil import protocol


def test_delimited_ready_survives_partial_boot_output():
    """Accept only the standalone token after a partial boot line."""
    serial = Mock()
    serial.readline.side_effect = [b"\x00xx\x00xxxx\n", b"DUT READY\n"]
    log = []
    assert protocol._wait_for_token(serial, "DUT READY", 1, log)
    assert serial.readline.call_count == 2
    assert log[-1] == "DUT READY"


def test_embedded_ready_is_not_accepted():
    """Keep strict matching when boot garbage contains the token."""
    serial = Mock()
    serial.readline.side_effect = [b"\x00xxDUT READY\n", b"DUT READY\n"]
    log = []
    assert protocol._wait_for_token(serial, "DUT READY", 1, log)
    assert serial.readline.call_count == 2
    assert log[-1] == "DUT READY"
