#!/usr/bin/env python3
"""Small timestamped UDP source/receiver for the VNF migration benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import struct
import time


HEADER = struct.Struct("!QQ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sender = sub.add_parser("send")
    sender.add_argument("--host", required=True)
    sender.add_argument("--port", required=True, type=int)
    sender.add_argument("--duration", required=True, type=float)
    sender.add_argument("--pps", required=True, type=float)
    sender.add_argument("--payload-bytes", type=int, default=256)
    sender.add_argument("--output", required=True, type=Path)
    receiver = sub.add_parser("receive")
    receiver.add_argument("--group", required=True)
    receiver.add_argument("--port", required=True, type=int)
    receiver.add_argument("--duration", required=True, type=float)
    receiver.add_argument("--grace", type=float, default=0.5)
    receiver.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def send(args: argparse.Namespace) -> int:
    if args.duration <= 0 or args.pps <= 0 or args.payload_bytes < HEADER.size:
        raise ValueError("invalid sender timing or payload size")
    interval_ns = max(1, round(1e9 / args.pps))
    count = max(1, round(args.duration * args.pps))
    padding = bytes(args.payload_bytes - HEADER.size)
    started = time.monotonic_ns()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for sequence in range(count):
            deadline = started + sequence * interval_ns
            while True:
                remaining = deadline - time.monotonic_ns()
                if remaining <= 0:
                    break
                if remaining > 200_000:
                    time.sleep((remaining - 100_000) / 1e9)
            sent_ns = time.monotonic_ns()
            sock.sendto(HEADER.pack(sequence, sent_ns) + padding, (args.host, args.port))
    atomic_json(
        args.output,
        {"mode": "send", "sent": count, "pps": args.pps,
         "duration_seconds": (time.monotonic_ns() - started) / 1e9},
    )
    return 0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def receive(args: argparse.Namespace) -> int:
    if args.duration <= 0 or args.grace < 0:
        raise ValueError("invalid receiver timing")
    rows: list[tuple[int, int, int]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", args.port))
        membership = socket.inet_aton(args.group) + socket.inet_aton("0.0.0.0")
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sock.settimeout(0.1)
        deadline = time.monotonic() + args.duration + args.grace
        while time.monotonic() < deadline:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            if len(payload) < HEADER.size:
                continue
            sequence, sent_ns = HEADER.unpack_from(payload)
            rows.append((sequence, sent_ns, time.monotonic_ns()))
    sequences = [row[0] for row in rows]
    unique = sorted(set(sequences))
    expected = (max(unique) + 1) if unique else 0
    received_unique = len(unique)
    delays = [(received_ns - sent_ns) / 1e6 for _, sent_ns, received_ns in rows]
    arrivals = [row[2] for row in rows]
    gaps = [(right - left) / 1e6 for left, right in zip(arrivals, arrivals[1:])]
    atomic_json(
        args.output,
        {
            "mode": "receive",
            "received": len(rows),
            "received_unique": received_unique,
            "duplicates": len(rows) - received_unique,
            "expected_from_sequence": expected,
            "missing": max(0, expected - received_unique),
            "first_sequence": unique[0] if unique else None,
            "last_sequence": unique[-1] if unique else None,
            "mean_delay_ms": sum(delays) / len(delays) if delays else 0.0,
            "p95_delay_ms": percentile(delays, 0.95),
            "p99_delay_ms": percentile(delays, 0.99),
            "max_delay_ms": max(delays, default=0.0),
            "p95_gap_ms": percentile(gaps, 0.95),
            "max_gap_ms": max(gaps, default=0.0),
            "delay_samples_ms": delays,
            "sequence_numbers": sequences,
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    return send(args) if args.mode == "send" else receive(args)


if __name__ == "__main__":
    raise SystemExit(main())
