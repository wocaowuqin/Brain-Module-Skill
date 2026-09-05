#!/usr/bin/env python3
"""Small containerized UDP service-function forwarder with a health endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import socket
import threading
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--next-host", required=True)
    parser.add_argument("--next-port", type=int, required=True)
    parser.add_argument("--health-port", type=int, default=0)
    parser.add_argument("--vnf-type", required=True)
    parser.add_argument("--processing-delay-ms", type=float, default=0.0)
    parser.add_argument("--cpu-work", type=int, default=1)
    parser.add_argument("--drop-every", type=int, default=0)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("--stats-output", default=None)
    return parser.parse_args()


def health_server(port: int, stop: threading.Event) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", int(port)))
        server.listen(8)
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                connection.sendall(b"ok\n")


def main() -> int:
    args = parse_args()
    if (
        not 1 <= args.listen_port <= 65535
        or not 1 <= args.next_port <= 65535
        or not 0 <= args.health_port <= 65535
        or args.processing_delay_ms < 0.0
        or args.cpu_work < 0
        or args.drop_every < 0
    ):
        raise ValueError("invalid VNF forwarder argument")
    stop = threading.Event()

    def request_stop(*_values) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    health = None
    if args.health_port:
        health = threading.Thread(
            target=health_server, args=(args.health_port, stop), daemon=True
        )
        health.start()
    received = forwarded = dropped = 0
    started = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver, socket.socket(
        socket.AF_INET, socket.SOCK_DGRAM
    ) as sender:
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        receiver.bind((args.listen_host, args.listen_port))
        if args.ready_file:
            ready_path = Path(args.ready_file)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text("ready\n", encoding="ascii")
        receiver.settimeout(0.2)
        while not stop.is_set():
            try:
                payload, _ = receiver.recvfrom(65535)
            except socket.timeout:
                continue
            received += 1
            if args.drop_every and received % args.drop_every == 0:
                dropped += 1
                continue
            value = payload
            for _ in range(args.cpu_work):
                value = hashlib.sha256(value).digest()
            if args.processing_delay_ms:
                time.sleep(args.processing_delay_ms / 1000.0)
            sender.sendto(payload, (args.next_host, args.next_port))
            forwarded += 1
    stop.set()
    if health is not None:
        health.join(timeout=1.0)
    stats = {
        "vnf_type": args.vnf_type,
        "received": received,
        "forwarded": forwarded,
        "dropped": dropped,
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
