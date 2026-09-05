#!/usr/bin/env python3
"""Validate stateful make-before-break migration for the native UDP VNF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sdn" / "vnf_agent_native.c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets-before", type=int, default=50)
    parser.add_argument("--packets-after", type=int, default=50)
    parser.add_argument("--interval-ms", type=float, default=2.0)
    return parser.parse_args()


def wait_for(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.005)
    raise TimeoutError(path)


def send_fifo(fifo: Path, payload: dict) -> None:
    fd = os.open(fifo, os.O_WRONLY)
    try:
        os.write(fd, (json.dumps(payload, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(fd)


def command(listener: socket.socket, fifo: Path, token: str, payload: dict) -> dict:
    request = dict(payload, ack_socket=listener.getsockname(), ack_token=token)
    started = time.perf_counter_ns()
    send_fifo(fifo, request)
    reply = json.loads(listener.recv(65536).decode("utf-8"))
    reply["control_latency_ms"] = (time.perf_counter_ns() - started) / 1e6
    if reply.get("ack_token") != token or not reply.get("accepted"):
        raise RuntimeError(reply)
    return reply


def start_agent(executable: Path, directory: Path, name: str):
    fifo = directory / f"{name}.fifo"
    ready = directory / f"{name}.ready"
    process = subprocess.Popen(
        [
            str(executable),
            "--command-fifo", str(fifo),
            "--ready-file", str(ready),
            "--drain-timeout-ms", "200",
            "--drain-idle-ms", "5",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    wait_for(ready)
    return process, fifo


def main() -> int:
    args = parse_args()
    if args.packets_before <= 0 or args.packets_after <= 0 or args.interval_ms < 0:
        raise ValueError("packet counts must be positive and interval must be non-negative")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        executable = directory / "vnf-agent-native"
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", str(SOURCE), "-o", str(executable)],
            check=True,
        )
        agents = [start_agent(executable, directory, name) for name in ("up", "old", "new")]
        try:
            ack_path = directory / "acks.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as listener, \
                    socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender, \
                    socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
                listener.bind(str(ack_path))
                listener.settimeout(3.0)
                receiver.bind(("127.0.0.1", 39504))
                receiver.settimeout(2.0)
                common = {"request_id": 1, "vnf_type": 0, "drop_every": 0}
                old_register = command(
                    listener,
                    agents[1][1],
                    "register-old",
                    dict(common, operation="register", stage=1, listen_port=39502,
                         next_host="127.0.0.1", next_port=39504, migration_epoch=0),
                )
                upstream_register = command(
                    listener,
                    agents[0][1],
                    "register-upstream",
                    dict(common, operation="register", stage=0, listen_port=39501,
                         next_host="127.0.0.1", next_port=39502, migration_epoch=0),
                )

                received = []
                receive_times = []

                def emit(start: int, count: int) -> None:
                    for sequence in range(start, start + count):
                        sender.sendto(struct.pack("!Q", sequence), ("127.0.0.1", 39501))
                        payload, _ = receiver.recvfrom(65535)
                        received.append(struct.unpack("!Q", payload)[0])
                        receive_times.append(time.perf_counter_ns())
                        if args.interval_ms:
                            time.sleep(args.interval_ms / 1000.0)

                emit(0, args.packets_before)
                old_snapshot = command(
                    listener,
                    agents[1][1],
                    "snapshot-old",
                    {"operation": "snapshot", "request_id": 1, "stage": 1},
                )
                state = old_snapshot["state"]
                target_register = command(
                    listener,
                    agents[2][1],
                    "register-target",
                    dict(
                        common,
                        operation="register",
                        stage=1,
                        listen_port=39503,
                        next_host="127.0.0.1",
                        next_port=39504,
                        restore_received=state["received"],
                        restore_forwarded=state["forwarded"],
                        restore_dropped=state["dropped"],
                        migration_epoch=1,
                    ),
                )
                switch = command(
                    listener,
                    agents[0][1],
                    "switch-upstream",
                    {"operation": "update_next", "request_id": 1, "stage": 0,
                     "next_host": "127.0.0.1", "next_port": 39503,
                     "migration_epoch": 1},
                )
                emit(args.packets_before, args.packets_after)
                target_snapshot = command(
                    listener,
                    agents[2][1],
                    "snapshot-target",
                    {"operation": "snapshot", "request_id": 1, "stage": 1},
                )
                old_drain = command(
                    listener,
                    agents[1][1],
                    "drain-old",
                    {"operation": "drain", "request_id": 1, "stage": 1},
                )
                old_unregister = command(
                    listener,
                    agents[1][1],
                    "unregister-old",
                    {"operation": "unregister", "request_id": 1, "stage": 1},
                )

            expected = list(range(args.packets_before + args.packets_after))
            if received != expected:
                raise AssertionError(f"sequence discontinuity: {received}")
            target_state = target_snapshot["state"]
            if target_state["received"] != len(expected):
                raise AssertionError((state, target_state))
            gaps_ms = [
                (right - left) / 1e6
                for left, right in zip(receive_times, receive_times[1:])
            ]
            report = {
                "ok": True,
                "state_schema": state["schema"],
                "packets_sent": len(expected),
                "packets_received": len(received),
                "packet_loss_rate": 0.0,
                "sequence_continuous": True,
                "precopy_state": state,
                "restored_final_state": target_state,
                "state_continuous": target_state["received"] == len(expected),
                "migration_epoch": target_state["migration_epoch"],
                "switch_control_ms": switch["control_latency_ms"],
                "migration_control_ms": sum(
                    row["control_latency_ms"]
                    for row in (old_snapshot, target_register, switch, old_drain, old_unregister)
                ),
                "max_packet_gap_ms": max(gaps_ms, default=0.0),
                "baseline_interval_ms": args.interval_ms,
                "service_interruption_ms": max(
                    0.0, max(gaps_ms, default=0.0) - args.interval_ms
                ),
                "control_acks": {
                    "old_register": old_register,
                    "upstream_register": upstream_register,
                    "target_register": target_register,
                    "switch": switch,
                    "drain": old_drain,
                    "unregister": old_unregister,
                },
            }
            print(json.dumps(report, indent=2, sort_keys=True))
        finally:
            for process, fifo in agents:
                if process.poll() is None:
                    try:
                        send_fifo(fifo, {"operation": "shutdown"})
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
