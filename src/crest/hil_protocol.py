"""Serial-line protocol helpers for DUT and harness HIL coordination.

This module centralizes the small protocol tokens, parsing helpers, and
session-level handshake routines used to synchronize the CREST DUT and
measurement harness over serial links.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import serial

logger = logging.getLogger(__name__)

DUT_READY = "DUT READY"
HARNESS_READY = "HARNESS READY"
CMD_PING = "PING"
CMD_START = "START"
RESP_ARMED = "ARMED"
RESP_DONE = "DONE"
WAIT_HEARTBEAT_S = 5.0

RUNS_RE = re.compile(r"^runs:\s*(?P<value>\d+)$", re.IGNORECASE)
TIMER_PREFIX = "timer output:"


@dataclass
class HandshakeResult:
    """Capture the outcome of a DUT and harness synchronization attempt.

    Attributes
    ----------
    dut_log : list[str]
        Lines observed from the DUT serial port.
    harness_log : list[str]
        Lines observed from the harness serial port.
    dut_timer_found : bool
        Whether a DUT timing line was detected.
    harness_done : bool
        Whether the harness reported ``DONE``.
    runs_dut : int | None
        Parsed DUT run count, if present.
    runs_harness : int | None
        Parsed harness run count, if present.
    error : str | None
        Human-readable protocol error, if the handshake failed.
    """

    dut_log: List[str]
    harness_log: List[str]
    dut_timer_found: bool
    harness_done: bool
    runs_dut: Optional[int]
    runs_harness: Optional[int]
    error: Optional[str] = None


@dataclass
class HarnessSessionResult:
    """Capture the state of a primed harness-only session.

    Attributes
    ----------
    harness_log : list[str]
        Lines observed from the harness serial port.
    harness_ready : bool
        Whether the harness emitted ``HARNESS READY``.
    harness_done : bool
        Whether the harness emitted ``DONE``.
    runs_harness : int | None
        Parsed harness run count, if present.
    error : str | None
        Human-readable protocol error, if one occurred.
    """

    harness_log: List[str]
    harness_ready: bool
    harness_done: bool
    runs_harness: Optional[int]
    error: Optional[str] = None


def _decode_line(raw: bytes) -> str:
    """Decode a raw serial line into UTF-8 text.

    Parameters
    ----------
    raw : bytes
        Raw bytes read from the serial port.

    Returns
    -------
    str
        Decoded and stripped line. Invalid UTF-8 bytes are dropped.
    """
    return raw.decode("utf-8", errors="ignore").strip()


def _flush_input(ser: serial.Serial) -> None:
    """Clear any buffered input from a serial port.

    Parameters
    ----------
    ser : serial.Serial
        Open serial connection to flush.
    """
    try:
        ser.reset_input_buffer()
    except AttributeError:
        while True:
            pending = 0
            in_waiting = getattr(ser, "in_waiting", None)
            if in_waiting is not None:
                pending = int(in_waiting)
            elif hasattr(ser, "inWaiting"):
                pending = int(ser.inWaiting())
            if pending <= 0:
                break
            ser.read(pending)


def _send_line(ser: serial.Serial, text: str) -> None:
    """Send a newline-terminated command over serial.

    Parameters
    ----------
    ser : serial.Serial
        Open serial connection.
    text : str
        Line payload to send (without newline).
    """
    payload = f"{text}\n".encode("utf-8")
    ser.write(payload)
    ser.flush()


def _wait_for_token(
    ser: serial.Serial,
    token: str,
    timeout_s: float,
    log: List[str],
    *,
    stage: Optional[str] = None,
) -> bool:
    """Wait for a line that exactly matches a token.

    Parameters
    ----------
    ser : serial.Serial
        Open serial connection to read from.
    token : str
        Expected line payload.
    timeout_s : float
        Maximum time to wait. Use <= 0 for no timeout.
    log : list[str]
        List to append all decoded lines.
    stage : str, optional
        Human-readable label used in progress logging.

    Returns
    -------
    bool
        True if the token was seen before timeout, else False.
    """
    stage_label = stage or f"token '{token}'"
    token_lower = token.lower()
    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    started_at = time.monotonic()
    next_heartbeat = started_at + WAIT_HEARTBEAT_S
    timeout_label = "infinite" if deadline is None else f"{timeout_s:.1f}s"
    logger.info("hil_protocol: waiting for %s (timeout=%s)", stage_label, timeout_label)

    while True:
        now = time.monotonic()
        if deadline is not None and now > deadline:
            logger.warning(
                "hil_protocol: timeout waiting for %s after %.1fs (lines=%d)",
                stage_label,
                now - started_at,
                len(log),
            )
            break
        raw = ser.readline()
        if not raw:
            now = time.monotonic()
            if now >= next_heartbeat:
                if deadline is None:
                    remaining = "infinite"
                else:
                    remaining = f"{max(0.0, deadline - now):.1f}s"
                logger.info(
                    "hil_protocol: still waiting for %s (elapsed=%.1fs, remaining=%s, lines=%d)",
                    stage_label,
                    now - started_at,
                    remaining,
                    len(log),
                )
                next_heartbeat = now + WAIT_HEARTBEAT_S
            continue
        line = _decode_line(raw)
        if not line:
            continue
        log.append(line)
        if line.lower() == token_lower:
            logger.info(
                "hil_protocol: received %s after %.1fs",
                stage_label,
                time.monotonic() - started_at,
            )
            return True
    return False


def _read_until_prefix(
    ser: serial.Serial,
    prefix: str,
    timeout_s: float,
    log: List[str],
    *,
    stage: Optional[str] = None,
) -> bool:
    """Read until a line begins with a prefix.

    Parameters
    ----------
    ser : serial.Serial
        Open serial connection to read from.
    prefix : str
        Line prefix to match (case-insensitive).
    timeout_s : float
        Maximum time to wait. Use <= 0 for no timeout.
    log : list[str]
        List to append all decoded lines.
    stage : str, optional
        Human-readable label used in progress logging.

    Returns
    -------
    bool
        True if a matching line is observed before timeout.
    """
    stage_label = stage or f"prefix '{prefix}'"
    prefix_lower = prefix.lower()
    deadline = None if timeout_s <= 0 else time.monotonic() + timeout_s
    started_at = time.monotonic()
    next_heartbeat = started_at + WAIT_HEARTBEAT_S
    timeout_label = "infinite" if deadline is None else f"{timeout_s:.1f}s"
    logger.info("hil_protocol: waiting for %s (timeout=%s)", stage_label, timeout_label)

    while True:
        now = time.monotonic()
        if deadline is not None and now > deadline:
            logger.warning(
                "hil_protocol: timeout waiting for %s after %.1fs (lines=%d)",
                stage_label,
                now - started_at,
                len(log),
            )
            break
        raw = ser.readline()
        if not raw:
            now = time.monotonic()
            if now >= next_heartbeat:
                if deadline is None:
                    remaining = "infinite"
                else:
                    remaining = f"{max(0.0, deadline - now):.1f}s"
                logger.info(
                    "hil_protocol: still waiting for %s (elapsed=%.1fs, remaining=%s, lines=%d)",
                    stage_label,
                    now - started_at,
                    remaining,
                    len(log),
                )
                next_heartbeat = now + WAIT_HEARTBEAT_S
            continue
        line = _decode_line(raw)
        if not line:
            continue
        log.append(line)
        if line.lower().startswith(prefix_lower):
            logger.info(
                "hil_protocol: received %s after %.1fs",
                stage_label,
                time.monotonic() - started_at,
            )
            return True
    return False


def _read_until_done(ser: serial.Serial, timeout_s: float, log: List[str]) -> bool:
    """Wait for the harness DONE token.

    Parameters
    ----------
    ser : serial.Serial
        Open serial connection.
    timeout_s : float
        Maximum time to wait. Use <= 0 for no timeout.
    log : list[str]
        List to append all decoded lines.

    Returns
    -------
    bool
        True if DONE was received before timeout.
    """
    return _wait_for_token(
        ser,
        RESP_DONE,
        timeout_s,
        log,
        stage="HARNESS DONE",
    )


def _parse_runs(lines: List[str]) -> Optional[int]:
    """Parse a `runs: <N>` line from a log.

    Parameters
    ----------
    lines : list[str]
        Serial log lines to scan.

    Returns
    -------
    int | None
        Parsed run count if present.
    """
    for line in lines:
        match = RUNS_RE.match(line.strip())
        if match:
            try:
                return int(match.group("value"))
            except (TypeError, ValueError):
                return None
    return None


def _total_harness_done_timeout(
    *,
    harness_active_timeout_s: float,
    harness_done_timeout_s: float,
) -> float:
    """Compute the combined timeout budget for waiting on harness completion.

    Parameters
    ----------
    harness_active_timeout_s : float
        Timeout budget for the active phase before ``DONE``.
    harness_done_timeout_s : float
        Additional timeout budget after activity has been observed.

    Returns
    -------
    float
        Total timeout in seconds. Returns ``0.0`` when either component is
        non-positive.
    """
    if harness_active_timeout_s <= 0 or harness_done_timeout_s <= 0:
        return 0.0
    return max(0.0, harness_active_timeout_s) + max(0.0, harness_done_timeout_s)


def prime_harness_session(
    *,
    harness: serial.Serial,
    harness_ready_timeout_s: float,
    harness_log: Optional[List[str]] = None,
    flush_input: bool = True,
) -> HarnessSessionResult:
    """Prime an already-open harness serial session.

    Parameters
    ----------
    harness : serial.Serial
        Open harness serial connection.
    harness_ready_timeout_s : float
        Timeout waiting for ``HARNESS READY`` after sending ``PING``.
    harness_log : list[str] | None, optional
        Existing log buffer to append to.
    flush_input : bool, optional
        When True, clear buffered input before sending ``PING``.

    Returns
    -------
    HarnessSessionResult
        Session status and captured harness log.
    """
    log = harness_log if harness_log is not None else []
    if flush_input:
        _flush_input(harness)
    logger.info("hil_protocol: sending PING to harness (open session)")
    _send_line(harness, CMD_PING)
    harness_ready = _wait_for_token(
        harness,
        HARNESS_READY,
        harness_ready_timeout_s,
        log,
        stage="HARNESS READY",
    )
    if not harness_ready:
        return HarnessSessionResult(
            harness_log=log,
            harness_ready=False,
            harness_done=False,
            runs_harness=None,
            error="harness_ready_timeout",
        )
    return HarnessSessionResult(
        harness_log=log,
        harness_ready=True,
        harness_done=False,
        runs_harness=None,
        error=None,
    )


def wait_for_harness_done(
    *,
    harness: serial.Serial,
    harness_active_timeout_s: float,
    harness_done_timeout_s: float,
    harness_log: Optional[List[str]] = None,
) -> HarnessSessionResult:
    """Read an open harness session until ``DONE`` is observed or timeout hits.

    Parameters
    ----------
    harness : serial.Serial
        Open harness serial connection.
    harness_active_timeout_s : float
        Maximum expected active measurement window.
    harness_done_timeout_s : float
        Additional timeout waiting for ``DONE`` after active phase.
    harness_log : list[str] | None, optional
        Existing log buffer to append to.

    Returns
    -------
    HarnessSessionResult
        Completion state and parsed run count.
    """
    log = harness_log if harness_log is not None else []
    total_harness_timeout = _total_harness_done_timeout(
        harness_active_timeout_s=harness_active_timeout_s,
        harness_done_timeout_s=harness_done_timeout_s,
    )
    timeout_label = "infinite" if total_harness_timeout <= 0 else f"{total_harness_timeout:.1f}s"
    logger.info("hil_protocol: waiting for HARNESS DONE (timeout=%s)", timeout_label)
    harness_done = _read_until_done(harness, total_harness_timeout, log)
    runs_harness = _parse_runs(log)
    return HarnessSessionResult(
        harness_log=log,
        harness_ready=True,
        harness_done=harness_done,
        runs_harness=runs_harness,
        error=None if harness_done else "harness_done_timeout",
    )


def run_dut_only(
    *,
    dut_port: str,
    baud_rate: int,
    dut_ready_timeout_s: float,
    dut_timer_timeout_s: float,
) -> List[str]:
    """Run a DUT-only handshake to capture timer output.

    This path uses best-effort startup behavior: if ``DUT READY`` is not
    observed within ``dut_ready_timeout_s``, START is still sent so recovery is
    possible when READY was missed in logs but the DUT is actually ready.

    Parameters
    ----------
    dut_port : str
        Serial port for the DUT.
    baud_rate : int
        Serial baud rate.
    dut_ready_timeout_s : float
        Timeout waiting for DUT READY.
    dut_timer_timeout_s : float
        Timeout waiting for timer output.

    Returns
    -------
    list[str]
        Captured DUT log lines.
    """
    dut_log: List[str] = []
    with serial.Serial(dut_port, baudrate=baud_rate, timeout=0.1) as dut:
        _flush_input(dut)
        ready_seen = _wait_for_token(
            dut,
            DUT_READY,
            dut_ready_timeout_s,
            dut_log,
            stage="DUT READY",
        )
        if not ready_seen:
            logger.warning(
                "hil_protocol: DUT READY not observed on %s; sending START in best-effort mode",
                dut_port,
            )
        logger.info("hil_protocol: sending START to DUT (%s)", dut_port)
        _send_line(dut, CMD_START)
        _read_until_prefix(
            dut,
            TIMER_PREFIX,
            dut_timer_timeout_s,
            dut_log,
            stage="DUT timer output",
        )
    return dut_log


def run_handshake(
    *,
    dut_port: str,
    harness_port: str,
    baud_rate: int,
    dut_ready_timeout_s: float,
    dut_timer_timeout_s: float,
    harness_ready_timeout_s: float,
    harness_active_timeout_s: float,
    harness_done_timeout_s: float,
) -> HandshakeResult:
    """Run the full DUT+harness handshake and collect logs.

    Parameters
    ----------
    dut_port : str
        Serial port for the DUT.
    harness_port : str
        Serial port for the harness.
    baud_rate : int
        Serial baud rate.
    dut_ready_timeout_s : float
        Timeout waiting for DUT READY.
    dut_timer_timeout_s : float
        Timeout waiting for DUT timer output.
    harness_ready_timeout_s : float
        Timeout waiting for HARNESS READY (<= 0 disables the timeout).
    harness_active_timeout_s : float
        Maximum time to wait for a measurement window (<= 0 disables the timeout).
    harness_done_timeout_s : float
        Timeout waiting for DONE after DUT timer output is observed
        (<= 0 disables the timeout). This timeout applies only to the final
        DONE read phase and does not include harness arming time before START.

    Returns
    -------
    HandshakeResult
        Logs, run counts, and any error marker.
    """
    dut_log: List[str] = []
    harness_log: List[str] = []

    with serial.Serial(harness_port, baudrate=baud_rate, timeout=0.1) as harness:
        _flush_input(harness)
        logger.info("hil_protocol: sending PING to harness (%s)", harness_port)
        _send_line(harness, CMD_PING)
        if not _wait_for_token(
            harness,
            HARNESS_READY,
            harness_ready_timeout_s,
            harness_log,
            stage="HARNESS READY",
        ):
            return HandshakeResult(
                dut_log=dut_log,
                harness_log=harness_log,
                dut_timer_found=False,
                harness_done=False,
                runs_dut=None,
                runs_harness=None,
                error="harness_ready_timeout",
            )

        with serial.Serial(dut_port, baudrate=baud_rate, timeout=0.1) as dut:
            _flush_input(dut)
            if not _wait_for_token(
                dut,
                DUT_READY,
                dut_ready_timeout_s,
                dut_log,
                stage="DUT READY",
            ):
                return HandshakeResult(
                    dut_log=dut_log,
                    harness_log=harness_log,
                    dut_timer_found=False,
                    harness_done=False,
                    runs_dut=None,
                    runs_harness=None,
                    error="dut_ready_timeout",
                )
            logger.info("hil_protocol: sending START to DUT (%s)", dut_port)
            _send_line(dut, CMD_START)
            dut_timer_found = _read_until_prefix(
                dut,
                TIMER_PREFIX,
                dut_timer_timeout_s,
                dut_log,
                stage="DUT timer output",
            )

        # DONE waiting starts only after DUT timer output is observed, so this
        # budget is intentionally scoped to harness completion after that point.
        total_harness_timeout = _total_harness_done_timeout(
            harness_active_timeout_s=harness_active_timeout_s,
            harness_done_timeout_s=harness_done_timeout_s,
        )
        timeout_label = (
            "infinite"
            if total_harness_timeout <= 0
            else f"{total_harness_timeout:.1f}s"
        )
        logger.info("hil_protocol: waiting for HARNESS DONE (timeout=%s)", timeout_label)
        harness_done = _read_until_done(harness, total_harness_timeout, harness_log)

    runs_dut = _parse_runs(dut_log)
    runs_harness = _parse_runs(harness_log)

    return HandshakeResult(
        dut_log=dut_log,
        harness_log=harness_log,
        dut_timer_found=dut_timer_found,
        harness_done=harness_done,
        runs_dut=runs_dut,
        runs_harness=runs_harness,
        error=None,
    )
