#!/usr/bin/env python3
"""POSIX regression check for the persistent probe agent scheduler."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "sdn" / "udp_sla_probe.py"


def wait_for(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {path}")


def main() -> int:
    if not hasattr(os, "mkfifo"):
        print(
            json.dumps(
                {
                    "ok": True,
                    "skipped": True,
                    "reason": "persistent probe agent requires POSIX FIFO support",
                },
                indent=2,
            )
        )
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        fifo = directory / "agent.fifo"
        ready = directory / "agent.ready"
        origin = directory / "schedule.origin"
        output = directory / "sender.json"
        receiver_output = directory / "receiver.json"
        process = subprocess.Popen(
            [
                sys.executable,
                str(PROBE),
                "agent",
                "--command-fifo",
                str(fifo),
                "--ready-file",
                str(ready),
                "--sender-workers",
                "1",
                "--receiver-workers",
                "1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for(ready, 2.0)
            blocking_receiver = {
                "argv": [
                    "receiver",
                    "--group",
                    "127.0.0.1",
                    "--port",
                    "39104",
                    "--duration",
                    "1.2",
                    "--interface-ip",
                    "0.0.0.0",
                    "--delay-bound-ms",
                    "100",
                    "--delay-compliance-ratio",
                    "0.95",
                    "--packet-loss-bound",
                    "1.0",
                    "--grace-seconds",
                    "0",
                    "--output",
                    str(receiver_output),
                ]
            }
            with fifo.open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(blocking_receiver, separators=(",", ":")) + "\n"
                )
            payload = {
                "argv": [
                    "sender",
                    "--destination",
                    "127.0.0.1",
                    "--port",
                    "39103",
                    "--duration",
                    "0.5",
                    "--packets-per-second",
                    "20",
                    "--payload-bytes",
                    "128",
                    "--dscp",
                    "0",
                    "--ready-timeout",
                    "0.1",
                    "--schedule-origin-file",
                    str(origin),
                    "--stop-offset-ns",
                    "800000000",
                    "--minimum-duration",
                    "0.05",
                    "--output",
                    str(output),
                ],
                "schedule_origin_file": str(origin),
                "start_offset_ns": 300_000_000,
            }
            with fifo.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            origin.write_text(str(time.time_ns()), encoding="ascii")
            time.sleep(0.15)
            if output.exists():
                raise AssertionError("scheduled task ran before its start offset")
            wait_for(output, 2.0)
            if receiver_output.exists():
                raise AssertionError(
                    "sender waited behind a task in the receiver worker pool"
                )
            wait_for(receiver_output, 2.0)
            with fifo.open("w", encoding="utf-8") as handle:
                handle.write('{"operation":"shutdown"}\n')
            process.wait(timeout=5.0)
            result = json.loads(output.read_text(encoding="utf-8"))
            if (
                result.get("status") != "completed"
                or result.get("sent_packets", 0) <= 0
                or result.get("agent_queue_wait_ms", 1_000.0) > 100.0
            ):
                raise AssertionError(f"scheduled sender failed: {result}")
            print(
                json.dumps(
                    {
                        "ok": True,
                        "scheduled_agent": True,
                        "role_isolated_workers": True,
                        "sent_packets": result["sent_packets"],
                        "sender_queue_wait_ms": result["agent_queue_wait_ms"],
                    },
                    indent=2,
                )
            )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
