#!/usr/bin/env python3
"""
Run a minimal two-board GPIO timing test.

This script flashes the toy DUT and harness sketches, then streams serial logs
from both boards to the terminal and to separate log files.
"""

from __future__ import annotations

import argparse
import subprocess
import threading
import time
from pathlib import Path

import serial


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal DUT/harness GPIO test.")
    parser.add_argument("--dut-port", default="/dev/ttyACM0", help="Serial port for the DUT.")
    parser.add_argument("--harness-port", default="/dev/ttyACM1", help="Serial port for the harness.")
    parser.add_argument(
        "--dut-fqbn",
        default="arduino:mbed_nano:nano33ble",
        help="Arduino FQBN for the DUT board.",
    )
    parser.add_argument(
        "--harness-fqbn",
        default="arduino:mbed_nano:nano33ble",
        help="Arduino FQBN for the harness board.",
    )
    parser.add_argument(
        "--skip-dut-flash",
        action="store_true",
        help="Skip flashing the DUT (useful if already flashed).",
    )
    parser.add_argument(
        "--skip-harness-flash",
        action="store_true",
        help="Skip flashing the harness (useful if already flashed).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of DUT flash/run cycles to perform.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=5.0,
        help="Seconds to sleep between DUT cycles.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    arduino_cli = "arduino-cli"
    arduino_cfg = repo_root / "tools" / "arduino-cli.yaml"
    dut_dir = repo_root / "analysis_scripts" / "toy_gpio_dut"
    harness_dir = repo_root / "analysis_scripts" / "toy_gpio_harness"
    log_dir = repo_root / "analysis_scripts" / "hil_single_run"
    log_dir.mkdir(parents=True, exist_ok=True)
    dut_log_path = log_dir / "dut_log.txt"
    harness_log_path = log_dir / "harness_log.txt"
    post_upload_sleep_s = 0.2
    harness_done_timeout_s = 5.0

    print(f"Writing DUT log to {dut_log_path}")
    print(f"Writing harness log to {harness_log_path}")

    if not args.skip_harness_flash:
        subprocess.run(
            [
                arduino_cli,
                "--config-file",
                str(arduino_cfg),
                "compile",
                "--fqbn",
                args.harness_fqbn,
                str(harness_dir),
            ],
            check=True,
        )
        time.sleep(post_upload_sleep_s)
        subprocess.run(
            [
                arduino_cli,
                "--config-file",
                str(arduino_cfg),
                "upload",
                "--fqbn",
                args.harness_fqbn,
                "--port",
                args.harness_port,
                str(harness_dir),
            ],
            check=True,
        )

    time.sleep(1.0)

    stop_event = threading.Event()

    def _monitor_serial(
        label: str,
        port: str,
        log_path: Path,
        done_event: threading.Event | None = None,
        done_token: str | None = None,
        stop_on_done: bool = False,
        local_stop: threading.Event | None = None,
    ) -> None:
        local_stop = local_stop or threading.Event()
        try:
            with serial.Serial(port, baudrate=115200, timeout=0.1) as ser, open(
                log_path, "a", buffering=1
            ) as log_file:
                ser.reset_input_buffer()
                while not stop_event.is_set() and not local_stop.is_set():
                    try:
                        raw = ser.readline()
                    except serial.SerialException:
                        break
                    if not raw:
                        continue
                    line = raw.decode(errors="ignore").strip()
                    if not line:
                        continue
                    print(f"{label}: {line}")
                    log_file.write(f"{line}\n")
                    if done_event and done_token and done_token in line:
                        done_event.set()
                        if stop_on_done:
                            local_stop.set()
        except serial.SerialException:
            return

    harness_done = threading.Event()
    harness_thread = threading.Thread(
        target=_monitor_serial,
        args=(
            "HARNESS",
            args.harness_port,
            harness_log_path,
            harness_done,
            "HARNESS: DONE",
            False,
        ),
        daemon=True,
    )
    harness_thread.start()

    dut_thread: threading.Thread | None = None
    dut_stop: threading.Event | None = None

    try:
        print("Monitoring serial output (Ctrl-C to stop)...")
        for cycle_idx in range(1, args.cycles + 1):
            if stop_event.is_set():
                break
            print(f"--- DUT cycle {cycle_idx}/{args.cycles} ---")
            with open(dut_log_path, "a", buffering=1) as log_file:
                log_file.write(f"--- DUT cycle {cycle_idx}/{args.cycles} ---\n")
            harness_done.clear()

            if dut_thread and dut_thread.is_alive():
                if dut_stop:
                    dut_stop.set()
                dut_thread.join(timeout=1.0)

            if not args.skip_dut_flash:
                subprocess.run(
                    [
                        arduino_cli,
                        "--config-file",
                        str(arduino_cfg),
                        "compile",
                        "--fqbn",
                        args.dut_fqbn,
                        str(dut_dir),
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        arduino_cli,
                        "--config-file",
                        str(arduino_cfg),
                        "upload",
                        "--fqbn",
                        args.dut_fqbn,
                        "--port",
                        args.dut_port,
                        str(dut_dir),
                    ],
                    check=True,
                )
                time.sleep(post_upload_sleep_s)

            time.sleep(1.0)

            dut_done = threading.Event()
            dut_stop = threading.Event()
            dut_thread = threading.Thread(
                target=_monitor_serial,
                args=("DUT", args.dut_port, dut_log_path, dut_done, "DUT: DONE", True, dut_stop),
                daemon=True,
            )
            dut_thread.start()
            while not dut_done.is_set() and not stop_event.is_set():
                time.sleep(0.2)
            if dut_stop:
                dut_stop.set()
            dut_thread.join(timeout=1.0)
            if not stop_event.is_set() and not harness_done.wait(harness_done_timeout_s):
                print("WARNING: Harness did not report DONE before timeout.")

            if cycle_idx < args.cycles:
                time.sleep(args.sleep_seconds)
    except KeyboardInterrupt:
        stop_event.set()
        harness_thread.join(timeout=1.0)
        print("Stopping...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
