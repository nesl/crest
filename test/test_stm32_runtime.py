from __future__ import annotations

import queue
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinyodom.microcontrollers import stm32_runtime  # noqa: E402


class _FakeSerial:
    def __init__(self, port: str, baud: int, timeout: float) -> None:
        del timeout
        self.port = port
        self.baud = baud
        self.is_open = True
        self._lines: queue.Queue[bytes] = queue.Queue()
        self.writes: list[bytes] = []

    def readline(self) -> bytes:
        if not self.is_open:
            return b""
        try:
            return self._lines.get(timeout=0.05)
        except queue.Empty:
            return b""

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.is_open = False

    def push_line(self, text: str) -> None:
        self._lines.put(f"{text}\r\n".encode("utf-8"))


class _FakeMonitor:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.writes: list[str] = []

    def wait_for_match(self, predicate, timeout_s: float, stage: str, *, start_index: int = 0):
        del timeout_s, stage
        for index in range(start_index, len(self._lines)):
            if predicate(self._lines[index]):
                return self._lines[index], index + 1
        return None, len(self._lines)

    def write_line(self, text: str) -> None:
        self.writes.append(text)

    def snapshot_lines(self) -> list[str]:
        return list(self._lines)


class STM32RuntimeParserTests(unittest.TestCase):
    def test_parse_runtime_lines_accepts_required_back_to_back_grammar(self) -> None:
        lines = [
            "STM32_BOOT=START",
            "STM32_AI_INIT=OK",
            "DUT READY",
            "phase output: back_to_back",
            "runs: 10",
            "clock hz output: 600000000",
            "dwt cycles per inference output: 120000",
            "timer per inference output: 0.000200",
            "timer output: 0.000200",
            "inference seq output: 1",
            "STM32_AI_RUN=OK",
        ]

        telemetry = stm32_runtime.parse_stm32_runtime_lines(lines)

        self.assertAlmostEqual(telemetry.latency_s, 0.0002)
        self.assertEqual(telemetry.power_metrics["phase"], "back_to_back")
        self.assertEqual(telemetry.power_metrics["runs"], 10)
        self.assertAlmostEqual(telemetry.power_metrics["clock_hz"], 600000000.0)
        self.assertAlmostEqual(telemetry.power_metrics["dwt_cycles_per_inference"], 120000.0)

    def test_parse_runtime_lines_rejects_fail_tokens(self) -> None:
        with self.assertRaises(stm32_runtime.STM32RuntimeProtocolError) as context:
            stm32_runtime.parse_stm32_runtime_lines(
                [
                    "STM32_AI_INIT=OK",
                    "DUT READY",
                    "STM32_AI_RUN=FAIL batch=0",
                ]
            )

        self.assertEqual(context.exception.kind, "runtime_protocol")

    def test_parse_runtime_lines_requires_latency_token(self) -> None:
        with self.assertRaises(stm32_runtime.STM32RuntimeProtocolError) as context:
            stm32_runtime.parse_stm32_runtime_lines(
                [
                    "STM32_AI_INIT=OK",
                    "DUT READY",
                    "phase output: back_to_back",
                    "runs: 10",
                    "clock hz output: 600000000",
                    "STM32_AI_RUN=OK",
                ]
            )

        self.assertEqual(context.exception.kind, "runtime_latency")


class SerialMonitorTests(unittest.TestCase):
    def test_serial_monitor_replays_history_and_writes_lines(self) -> None:
        fake_serial = _FakeSerial("/dev/ttyACM0", 115200, 0.2)

        with patch("tinyodom.microcontrollers.stm32_runtime.serial.Serial", return_value=fake_serial):
            monitor = stm32_runtime.SerialMonitor("/dev/ttyACM0", 115200, "dut")
            try:
                fake_serial.push_line("STM32_AI_INIT=OK")
                time.sleep(0.05)
                line, cursor = monitor.wait_for_match(
                    lambda value: value == "STM32_AI_INIT=OK",
                    0.2,
                    "STM32_AI_INIT=OK",
                )
                self.assertEqual(line, "STM32_AI_INIT=OK")
                fake_serial.push_line("DUT READY")
                line, _ = monitor.wait_for_match(
                    lambda value: value == "DUT READY",
                    0.2,
                    "DUT READY",
                    start_index=cursor,
                )
                self.assertEqual(line, "DUT READY")
                monitor.write_line("START")
            finally:
                monitor.close()

        self.assertEqual(fake_serial.writes, [b"START\n"])


class STM32RuntimeSessionTests(unittest.TestCase):
    def test_execute_runtime_session_waits_for_init_then_ready_then_run(self) -> None:
        monitor = _FakeMonitor(
            [
                "STM32_BOOT=START",
                "STM32_AI_INIT=OK",
                "phase output: back_to_back",
                "DUT READY",
                "runs: 5",
                "clock hz output: 600000000",
                "timer output: 0.000500",
                "STM32_AI_RUN=OK",
            ]
        )

        telemetry = stm32_runtime.execute_runtime_session(
            monitor,
            boot_timeout_s=1.0,
            run_timeout_s=1.0,
        )

        self.assertEqual(monitor.writes, ["START"])
        self.assertAlmostEqual(telemetry.latency_s, 0.0005)

    def test_execute_runtime_session_times_out_before_init(self) -> None:
        monitor = _FakeMonitor(["STM32_BOOT=START"])

        with self.assertRaises(stm32_runtime.STM32RuntimeProtocolError) as context:
            stm32_runtime.execute_runtime_session(
                monitor,
                boot_timeout_s=0.1,
                run_timeout_s=0.1,
            )

        self.assertEqual(context.exception.kind, "runtime_timeout")


if __name__ == "__main__":
    unittest.main()
