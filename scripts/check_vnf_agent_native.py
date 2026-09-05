#!/usr/bin/env python3
"""Linux regression check for native VNF control ACKs and UDP forwarding."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sdn" / "vnf_agent_native.c"


def wait_for(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise TimeoutError(path)


def send(fifo: Path, payload: dict) -> None:
    fd = os.open(fifo, os.O_WRONLY)
    try:
        os.write(fd, (json.dumps(payload, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(fd)


def receive_ack(listener: socket.socket, operation: str) -> dict:
    ack = json.loads(listener.recv(65536).decode("utf-8"))
    if not ack.get("accepted") or ack.get("operation") != operation:
        raise AssertionError(ack)
    return ack


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        executable = directory / "vnf-agent-native"
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", str(SOURCE), "-o", str(executable)],
            check=True,
        )
        fifo = directory / "agent.fifo"
        ready = directory / "agent.ready"
        process = subprocess.Popen(
            [
                str(executable),
                "--command-fifo", str(fifo),
                "--ready-file", str(ready),
                "--drain-timeout-ms", "200",
                "--drain-idle-ms", "5",
            ]
        )
        try:
            wait_for(ready)
            ack_path = directory / "acks.sock"
            stats = directory / "stats.json"
            drain = directory / "drain.json"
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as listener:
                listener.bind(str(ack_path))
                listener.settimeout(3.0)
                register = {
                    "operation": "register",
                    "request_id": 1,
                    "stage": 0,
                    "listen_port": 39401,
                    "next_host": "127.0.0.1",
                    "next_port": 39402,
                    "vnf_type": 0,
                    "ack_socket": str(ack_path),
                    "ack_token": "register-1",
                    "stats_output": str(stats),
                }
                send(fifo, register)
                register_ack = receive_ack(listener, "register")

                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as source, socket.socket(
                    socket.AF_INET, socket.SOCK_DGRAM
                ) as destination:
                    destination.bind(("127.0.0.1", 39402))
                    destination.settimeout(2.0)
                    expected_payloads = {
                        f"native-check-{index}".encode("ascii")
                        for index in range(128)
                    }
                    for payload in expected_payloads:
                        source.sendto(payload, ("127.0.0.1", 39401))
                    received_payloads = {
                        destination.recvfrom(65535)[0]
                        for _ in range(len(expected_payloads))
                    }
                if received_payloads != expected_payloads:
                    raise AssertionError(
                        f"forwarded payload mismatch: "
                        f"expected={len(expected_payloads)} "
                        f"received={len(received_payloads)}"
                    )

                send(
                    fifo,
                    {
                        "operation": "drain",
                        "request_id": 1,
                        "stage": 0,
                        "drain_ack": str(drain),
                        "ack_socket": str(ack_path),
                        "ack_token": "drain-1",
                    },
                )
                drain_control_ack = receive_ack(listener, "drain")
                wait_for(drain)
                send(
                    fifo,
                    {
                        "operation": "unregister",
                        "request_id": 1,
                        "stage": 0,
                        "ack_socket": str(ack_path),
                        "ack_token": "unregister-1",
                    },
                )
                unregister_ack = receive_ack(listener, "unregister")
                wait_for(stats)

                register.update(
                    request_id=2,
                    ack_token="register-2",
                    stats_output=str(directory / "stats-2.json"),
                )
                send(fifo, register)
                reused_ack = receive_ack(listener, "register")
            send(fifo, {"operation": "shutdown"})
            process.wait(timeout=5.0)
            report = {
                "ok": True,
                "register_ack": register_ack,
                "drain_ack": drain_control_ack,
                "unregister_ack": unregister_ack,
                "reused_port_ack": reused_ack,
                "stats": json.loads(stats.read_text(encoding="utf-8")),
            }
            print(json.dumps(report, indent=2))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
