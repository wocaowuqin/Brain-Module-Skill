#!/usr/bin/env python3
"""POSIX regression check for the persistent per-DC VNF agent."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "sdn" / "vnf_agent.py"


def wait_for(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise TimeoutError(path)


def send_command(fifo: Path, payload: dict) -> None:
    with fifo.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def send_commands(fifo: Path, payloads: list[dict]) -> None:
    with fifo.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        fifo = directory / "agent.fifo"
        ready = directory / "agent.ready"
        process = subprocess.Popen(
            [
                sys.executable, str(AGENT),
                "--command-fifo", str(fifo),
                "--ready-file", str(ready),
                "--workers", "2",
                "--prebound-port-base", "39211",
                "--prebound-port-count", "2",
                "--drain-timeout-ms", "200",
                "--drain-idle-ms", "5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_for(ready)
            registrations = [
                (1, 39212, "127.0.0.1", 39213),
                (0, 39211, "127.0.0.1", 39212),
            ]
            for stage, listen_port, next_host, next_port in registrations:
                send_command(
                    fifo,
                    {
                        "operation": "register",
                        "request_id": 1,
                        "stage": stage,
                        "listen_port": listen_port,
                        "next_host": next_host,
                        "next_port": next_port,
                        "vnf_type": stage,
                        "dscp": 46,
                        "ready_file": str(directory / f"stage{stage}.ready"),
                        "stats_output": str(directory / f"stage{stage}.json"),
                    },
                )
                wait_for(directory / f"stage{stage}.ready")

            payloads = [f"packet-{index}".encode("ascii") for index in range(100)]
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender, socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM
            ) as receiver:
                receiver.bind(("127.0.0.1", 39213))
                receiver.setsockopt(socket.IPPROTO_IP, socket.IP_RECVTOS, 1)
                receiver.settimeout(2.0)
                for payload in payloads:
                    sender.sendto(payload, ("127.0.0.1", 39211))
                received = [receiver.recvmsg(65535, 64) for _ in payloads]
                observed = [row[0] for row in received]
            if observed != payloads:
                raise AssertionError("VNF agent changed packet order or payload")
            observed_tos = [
                int.from_bytes(data, byteorder="little")
                for _, ancillary, _, _ in received
                for level, kind, data in ancillary
                if level == socket.IPPROTO_IP and kind == socket.IP_TOS
            ]
            if observed_tos != [46 << 2] * len(payloads):
                raise AssertionError(f"VNF agent did not preserve DSCP: {observed_tos[:5]}")

            drain_rows = []
            for stage, *_ in reversed(registrations):
                drain_ack = directory / f"stage{stage}.drained.json"
                send_command(
                    fifo,
                    {
                        "operation": "drain",
                        "request_id": 1,
                        "stage": stage,
                        "drain_ack": str(drain_ack),
                    },
                )
                wait_for(drain_ack)
                drain_rows.append(
                    json.loads(drain_ack.read_text(encoding="utf-8"))
                )
                if drain_rows[-1].get("timed_out"):
                    raise AssertionError(f"VNF drain timed out: {drain_rows[-1]}")
                send_command(
                    fifo,
                    {"operation": "unregister", "request_id": 1, "stage": stage},
                )
                wait_for(directory / f"stage{stage}.json")

            reused_ready = directory / "reused.ready"
            reused_stats = directory / "reused.json"
            send_command(
                fifo,
                {
                    "operation": "register",
                    "request_id": 2,
                    "stage": 0,
                    "listen_port": 39211,
                    "next_host": "127.0.0.1",
                    "next_port": 39213,
                    "vnf_type": 0,
                    "ready_file": str(reused_ready),
                    "stats_output": str(reused_stats),
                },
            )
            wait_for(reused_ready)
            send_command(
                fifo,
                {"operation": "unregister", "request_id": 2, "stage": 0},
            )
            wait_for(reused_stats)

            # Exercise unregister while a readable event may already be queued.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as race_sender:
                for request_id in range(100, 120):
                    race_ready = directory / f"race-{request_id}.ready"
                    race_stats = directory / f"race-{request_id}.json"
                    send_command(
                        fifo,
                        {
                            "operation": "register",
                            "request_id": request_id,
                            "stage": 0,
                            "listen_port": 39214,
                            "next_host": "127.0.0.1",
                            "next_port": 39215,
                            "vnf_type": 0,
                            "ready_file": str(race_ready),
                            "stats_output": str(race_stats),
                        },
                    )
                    wait_for(race_ready)
                    for packet in range(250):
                        race_sender.sendto(
                            f"race-{request_id}-{packet}".encode("ascii"),
                            ("127.0.0.1", 39214),
                        )
                    send_command(
                        fifo,
                        {"operation": "unregister", "request_id": request_id, "stage": 0},
                    )
                    wait_for(race_stats)
                    if process.poll() is not None:
                        raise AssertionError("VNF agent exited during unregister race check")

            batch_ids = list(range(200, 208))
            ack_socket = directory / "registration-acks.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as ack_listener:
                ack_listener.bind(str(ack_socket))
                ack_listener.settimeout(2.0)
                send_commands(
                    fifo,
                    [
                        {
                            "operation": "register",
                            "request_id": request_id,
                            "stage": 0,
                            "listen_port": 39300 + request_id,
                            "next_host": "127.0.0.1",
                            "next_port": 39999,
                            "vnf_type": 0,
                            "ack_socket": str(ack_socket),
                            "ack_token": f"batch-{request_id}",
                            "stats_output": str(directory / f"batch-{request_id}.json"),
                        }
                        for request_id in batch_ids
                    ],
                )
                acknowledgements = [
                    json.loads(ack_listener.recv(65536).decode("utf-8"))
                    for _ in batch_ids
                ]
            if {row["request_id"] for row in acknowledgements} != set(batch_ids):
                raise AssertionError(acknowledgements)
            if not all(row.get("accepted") for row in acknowledgements):
                raise AssertionError(acknowledgements)
            if any((directory / f"batch-{request_id}.ready").exists() for request_id in batch_ids):
                raise AssertionError("ACK registration unexpectedly created ready files")
            unregister_ack_socket = directory / "unregister-acks.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as ack_listener:
                ack_listener.bind(str(unregister_ack_socket))
                ack_listener.settimeout(2.0)
                send_commands(
                    fifo,
                    [
                        {
                            "operation": "unregister",
                            "request_id": request_id,
                            "stage": 0,
                            "ack_socket": str(unregister_ack_socket),
                            "ack_token": f"unregister-{request_id}",
                        }
                        for request_id in batch_ids
                    ],
                )
                unregister_acks = [
                    json.loads(ack_listener.recv(65536).decode("utf-8"))
                    for _ in batch_ids
                ]
            if not all(
                row.get("accepted")
                and row.get("operation") == "unregister"
                and row.get("drain_wait_ms", 0.0) > 0.0
                for row in unregister_acks
            ):
                raise AssertionError(unregister_acks)
            for request_id in batch_ids:
                wait_for(directory / f"batch-{request_id}.json")

            send_command(fifo, {"operation": "shutdown"})
            process.wait(timeout=5.0)
            stats = [
                json.loads((directory / f"stage{stage}.json").read_text(encoding="utf-8"))
                for stage in (0, 1)
            ]
            if any(row["received"] != 100 or row["forwarded"] != 100 for row in stats):
                raise AssertionError(stats)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "packets": 100,
                        "workers": sorted({row["worker_id"] for row in stats}),
                        "registration_acks": acknowledgements,
                        "unregister_acks": unregister_acks,
                        "drains": drain_rows,
                        "stages": stats,
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
