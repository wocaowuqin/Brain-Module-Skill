#!/usr/bin/env python3
"""UDP sender/receiver used to measure multicast SFT SLA inside Mininet."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import errno
import heapq
import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time
from typing import Any


HEADER = struct.Struct("!IIQ")
DEFAULT_RECEIVE_BUFFER_BYTES = 4 * 1024 * 1024
LATE_WAKEUP_THRESHOLD_NS = 50_000


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


def _load_clock_nanosleep():
    if os.name != "posix":
        return None
    try:
        function = ctypes.CDLL(None).clock_nanosleep
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_Timespec),
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function
    except (AttributeError, OSError):
        return None


_CLOCK_NANOSLEEP = _load_clock_nanosleep()


def sleep_until_monotonic_ns(target_ns: int) -> str:
    """Wait for an absolute monotonic timestamp without relative-sleep drift."""
    remaining_ns = int(target_ns) - time.monotonic_ns()
    if remaining_ns <= 0:
        return "already_due"
    if _CLOCK_NANOSLEEP is None:
        time.sleep(remaining_ns / 1_000_000_000.0)
        return "python_sleep_fallback"

    target = _Timespec(
        tv_sec=int(target_ns) // 1_000_000_000,
        tv_nsec=int(target_ns) % 1_000_000_000,
    )
    while True:
        result = int(_CLOCK_NANOSLEEP(1, 1, ctypes.byref(target), None))
        if result == 0:
            return "clock_nanosleep"
        if result != errno.EINTR:
            raise OSError(result, os.strerror(result))


def missing_sequence_ranges(
    sequences: set[int], expected: int, limit: int = 16
) -> tuple[int | None, int | None, list[list[int]], int]:
    """Return compact missing-sequence diagnostics without expanding JSON."""
    missing = [sequence for sequence in range(max(0, int(expected))) if sequence not in sequences]
    if not missing:
        return None, None, [], 0
    ranges: list[list[int]] = []
    start = previous = missing[0]
    for sequence in missing[1:]:
        if sequence == previous + 1:
            previous = sequence
            continue
        if len(ranges) < limit:
            ranges.append([start, previous])
        start = previous = sequence
    if len(ranges) < limit:
        ranges.append([start, previous])
    return missing[0], missing[-1], ranges, max(0, len(missing) - sum(end - start + 1 for start, end in ranges))


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(value * len(ordered)) - 1))
    return float(ordered[index])


def build_packet(sequence: int, total_packets: int, payload_bytes: int) -> bytes:
    # All Mininet namespaces share one kernel monotonic clock.  Using it for
    # one-way probe timestamps avoids wall-clock corrections appearing as
    # artificial delay spikes during a replay.
    header = HEADER.pack(int(sequence), int(total_packets), time.monotonic_ns())
    if payload_bytes < len(header):
        raise ValueError(f"payload must be at least {len(header)} bytes")
    return header + bytes(payload_bytes - len(header))


def send_probe(
    destination: str,
    port: int,
    duration: float,
    packets_per_second: float,
    payload_bytes: int,
    dscp: int,
    ttl: int = 64,
    stop_time_ns: int | None = None,
    max_catch_up_packets: int = 0,
) -> dict[str, Any]:
    if duration <= 0.0 or packets_per_second <= 0.0:
        raise ValueError("duration and packets-per-second must be positive")
    if not 0 <= dscp <= 63:
        raise ValueError("DSCP must be in [0, 63]")
    if max_catch_up_packets < 0:
        raise ValueError("max catch-up packets must be non-negative")
    target_packets = max(1, int(math.ceil(duration * packets_per_second)))
    interval_ns = max(1, int(round(1_000_000_000.0 / packets_per_second)))
    address = ipaddress.ip_address(destination)
    started_ns = time.monotonic_ns()
    sent = 0
    pacing_resets = 0
    skipped_slots = 0
    late_wakeups = 0
    maximum_lateness_ns = 0
    total_lateness_ns = 0
    pacing_lateness_us: list[float] = []
    catch_up_packets = 0
    catch_up_burst = 0
    maximum_catch_up_burst = 0
    wait_backend = "already_due"
    slot = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, int(dscp) << 2)
        if address.is_multicast:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, int(ttl))
        while slot < target_packets:
            if stop_time_ns is not None and time.time_ns() >= int(stop_time_ns):
                break
            target_ns = started_ns + slot * interval_ns
            now_ns = time.monotonic_ns()
            behind_slots = max(0, (now_ns - target_ns) // interval_ns)
            catch_up_budget = max(0, max_catch_up_packets - catch_up_burst)
            if behind_slots > catch_up_budget:
                skipped = behind_slots - catch_up_budget
                if catch_up_budget == 0 and max_catch_up_packets > 0:
                    # Force the next target into the future after a full burst,
                    # so the configured cap is a real on-wire burst bound.
                    skipped += 1
                skipped = min(skipped, target_packets - slot)
                slot += skipped
                skipped_slots += skipped
                pacing_resets += 1
                if slot >= target_packets:
                    break
                target_ns = started_ns + slot * interval_ns
                behind_slots = max(0, (time.monotonic_ns() - target_ns) // interval_ns)
                if catch_up_budget == 0:
                    catch_up_burst = 0
            if behind_slots > 0:
                catch_up_burst += 1
                catch_up_packets += 1
                maximum_catch_up_burst = max(
                    maximum_catch_up_burst, catch_up_burst
                )
            else:
                catch_up_burst = 0
            wait_backend = sleep_until_monotonic_ns(target_ns)
            woke_ns = time.monotonic_ns()
            lateness_ns = max(0, woke_ns - target_ns)
            if stop_time_ns is not None and time.time_ns() >= int(stop_time_ns):
                break
            sock.sendto(
                build_packet(sent, target_packets, payload_bytes),
                (destination, int(port)),
            )
            maximum_lateness_ns = max(maximum_lateness_ns, lateness_ns)
            total_lateness_ns += lateness_ns
            pacing_lateness_us.append(lateness_ns / 1000.0)
            if lateness_ns > LATE_WAKEUP_THRESHOLD_NS:
                late_wakeups += 1
            sent += 1
            slot += 1
    return {
        "mode": "sender",
        "destination": destination,
        "port": int(port),
        "dscp": int(dscp),
        "payload_bytes": int(payload_bytes),
        "packets_per_second": float(packets_per_second),
        "planned_packets": target_packets,
        "sent_packets": sent,
        "deadline_limited_packets": target_packets - sent,
        "pacing_resets": pacing_resets,
        "skipped_pacing_slots": skipped_slots,
        "max_catch_up_packets": int(max_catch_up_packets),
        "catch_up_packets": catch_up_packets,
        "max_catch_up_burst": maximum_catch_up_burst,
        "late_wakeups_over_50us": late_wakeups,
        "max_pacing_lateness_us": maximum_lateness_ns / 1000.0,
        "mean_pacing_lateness_us": total_lateness_ns / max(1, sent) / 1000.0,
        "p95_pacing_lateness_us": percentile(pacing_lateness_us, 0.95),
        "p99_pacing_lateness_us": percentile(pacing_lateness_us, 0.99),
        "pacing_wait_backend": wait_backend,
        "elapsed_seconds": (time.monotonic_ns() - started_ns) / 1_000_000_000.0,
    }


def send_probe_native(
    executable: str,
    destination: str,
    port: int,
    duration: float,
    packets_per_second: float,
    payload_bytes: int,
    dscp: int,
    ttl: int = 64,
    stop_time_ns: int | None = None,
    expected_file: str | Path | None = None,
    realtime_priority: int = 0,
) -> dict[str, Any]:
    if expected_file is None:
        raise ValueError("native sender requires an expected packet file")
    command = [
        executable,
        destination,
        str(int(port)),
        f"{duration:.9f}",
        f"{packets_per_second:.9f}",
        str(int(payload_bytes)),
        str(int(dscp)),
        str(int(ttl)),
        str(int(stop_time_ns or 0)),
        str(expected_file),
    ]
    def configure_scheduler() -> None:
        if realtime_priority > 0:
            os.sched_setscheduler(
                0,
                os.SCHED_RR,
                os.sched_param(int(realtime_priority)),
            )

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=configure_scheduler if realtime_priority > 0 else None,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"native sender returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError("native sender result must be a JSON object")
    result["realtime_priority"] = int(realtime_priority)
    return result


def wait_for_ready_files(
    ready_files: list[str | Path], timeout: float, poll_seconds: float = 0.005
) -> dict[str, Any]:
    paths = [Path(value) for value in ready_files]
    if not paths:
        return {"ready_files": 0, "ready_wait_seconds": 0.0}
    if timeout <= 0.0 or poll_seconds <= 0.0:
        raise ValueError("ready timeout and polling interval must be positive")

    started = time.monotonic()
    deadline = started + timeout
    missing = list(paths)
    while time.monotonic() < deadline:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            elapsed = time.monotonic() - started
            for path in paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            return {
                "ready_files": len(paths),
                "ready_wait_seconds": elapsed,
            }
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))
    missing = [path for path in paths if not path.is_file()]
    if not missing:
        elapsed = time.monotonic() - started
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return {"ready_files": len(paths), "ready_wait_seconds": elapsed}
    raise TimeoutError(
        "receiver readiness timed out; missing "
        + ", ".join(str(path) for path in missing)
    )


def send_receiver_ready_ack(
    ack_socket: str | None,
    ack_token: str | None,
    *,
    accepted: bool,
    request_id: int | None = None,
    destination_id: int | None = None,
    error: str | None = None,
) -> None:
    if not ack_socket or not ack_token:
        return
    payload = {
        "accepted": bool(accepted),
        "ack_token": str(ack_token),
        "operation": "receiver_ready",
        "ready_monotonic_ns": time.monotonic_ns(),
        "ready_time_ns": time.time_ns(),
    }
    if request_id is not None:
        payload["request_id"] = int(request_id)
    if destination_id is not None:
        payload["destination_id"] = int(destination_id)
    if error:
        payload["error"] = str(error)
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
        sender.sendto(encoded, str(ack_socket))


def schedule_origin_ns(path: str | Path) -> int:
    value = int(Path(path).read_text(encoding="ascii").strip())
    if value <= 0:
        raise ValueError("schedule origin must be a positive epoch timestamp")
    return value


def receive_probe(
    group: str,
    port: int,
    duration: float,
    interface_ip: str,
    delay_bound_ms: float,
    delay_compliance_ratio: float,
    jitter_bound_ms: float | None,
    packet_loss_bound: float,
    grace_seconds: float = 1.0,
    expected_packets_hint: int = 0,
    ready_file: str | Path | None = None,
    stop_time_ns: int | None = None,
    expected_file: str | Path | None = None,
    receive_buffer_bytes: int = DEFAULT_RECEIVE_BUFFER_BYTES,
    ready_ack_socket: str | None = None,
    ready_ack_token: str | None = None,
    ready_request_id: int | None = None,
    ready_destination_id: int | None = None,
) -> dict[str, Any]:
    if duration <= 0.0 or grace_seconds < 0.0:
        raise ValueError("duration must be positive and grace must be non-negative")
    if delay_bound_ms <= 0.0 or not 0.0 < delay_compliance_ratio <= 1.0:
        raise ValueError("invalid delay SLA")
    if not 0.0 <= packet_loss_bound <= 1.0:
        raise ValueError("packet loss bound must be in [0, 1]")
    if expected_packets_hint < 0:
        raise ValueError("expected packet hint must be non-negative")
    if receive_buffer_bytes <= 0:
        raise ValueError("receive buffer must be positive")

    address = ipaddress.ip_address(group)
    delays_ms: list[float] = []
    sequences = set()
    expected_packets = 0
    previous_transit = None
    jitter_ms = 0.0
    setup_started = time.monotonic()
    remaining_seconds = duration
    if stop_time_ns is not None:
        remaining_seconds = max(0.0, (int(stop_time_ns) - time.time_ns()) / 1e9)
    deadline = setup_started + remaining_seconds + grace_seconds
    ready_time_ns = None
    ready_setup_seconds = None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(receive_buffer_bytes))
        actual_receive_buffer_bytes = sock.getsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF
        )
        sock.bind(("", int(port)))
        if address.is_multicast:
            membership = socket.inet_aton(group) + socket.inet_aton(interface_ip)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        sock.settimeout(0.2)
        ready_setup_seconds = time.monotonic() - setup_started
        ready_time_ns = time.time_ns()
        if ready_file is not None:
            ready_path = Path(ready_file)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(str(ready_time_ns), encoding="ascii")
        send_receiver_ready_ack(
            ready_ack_socket,
            ready_ack_token,
            accepted=True,
            request_id=ready_request_id,
            destination_id=ready_destination_id,
        )
        while time.monotonic() < deadline:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            if len(payload) < HEADER.size:
                continue
            sequence, total_packets, sent_ns = HEADER.unpack_from(payload)
            if sequence in sequences:
                continue
            sequences.add(sequence)
            expected_packets = max(expected_packets, int(total_packets))
            transit_ms = max(
                0.0,
                (time.monotonic_ns() - int(sent_ns)) / 1_000_000.0,
            )
            delays_ms.append(transit_ms)
            if previous_transit is not None:
                jitter_ms += (abs(transit_ms - previous_transit) - jitter_ms) / 16.0
            previous_transit = transit_ms

    received = len(sequences)
    expected_from_file = 0
    if expected_file is not None:
        try:
            expected_from_file = int(Path(expected_file).read_text(encoding="ascii"))
        except (OSError, ValueError):
            expected_from_file = 0
    sender_failed = expected_from_file < 0
    expected_from_file = max(0, expected_from_file)
    if expected_from_file > 0:
        # The sender publishes the offered packet count before transmission.
        # Keeping that value unchanged makes sender pacing misses visible as
        # SLA loss instead of silently shrinking the receiver denominator.
        expected = max(expected_from_file, received)
    else:
        expected = max(expected_packets, received, int(expected_packets_hint))
    lost = max(0, expected - received)
    first_missing, last_missing, missing_ranges, omitted_missing = missing_sequence_ranges(
        sequences, expected
    )
    loss_rate = 1.0 if sender_failed else lost / max(1, expected)
    delay_violations = sum(value > delay_bound_ms for value in delays_ms)
    delay_violation_rate = delay_violations / max(1, received)
    required_delay_ratio = float(delay_compliance_ratio)
    result = {
        "mode": "receiver",
        "group": group,
        "port": int(port),
        "interface_ip": interface_ip,
        "expected_packets": expected,
        "received_packets": received,
        "lost_packets": lost,
        "packet_loss_rate": loss_rate,
        "first_missing_sequence": first_missing,
        "last_missing_sequence": last_missing,
        "missing_sequence_ranges": missing_ranges,
        "missing_sequences_omitted": omitted_missing,
        "out_of_range_sequences": sum(sequence >= expected for sequence in sequences),
        "mean_delay_ms": sum(delays_ms) / len(delays_ms) if delays_ms else None,
        "p50_delay_ms": percentile(delays_ms, 0.50),
        "p95_delay_ms": percentile(delays_ms, 0.95),
        "p99_delay_ms": percentile(delays_ms, 0.99),
        "max_delay_ms": max(delays_ms) if delays_ms else None,
        "jitter_ms": jitter_ms if delays_ms else None,
        "delay_bound_ms": float(delay_bound_ms),
        "delay_compliance_ratio": required_delay_ratio,
        "delay_violations": delay_violations,
        "delay_violation_rate": delay_violation_rate,
        "jitter_bound_ms": jitter_bound_ms,
        "packet_loss_bound": float(packet_loss_bound),
        "receiver_setup_seconds": ready_setup_seconds,
        "receive_buffer_bytes": int(actual_receive_buffer_bytes),
        "expected_packets_source": (
            "sender_failure"
            if sender_failed
            else "sender_file"
            if expected_from_file > 0
            else "packet_header"
            if expected_packets > 0
            else "hint"
            if expected_packets_hint > 0
            else "none"
        ),
        "measurement_status": "sender_failed" if sender_failed else "completed",
    }
    result["delay_sla_met"] = (
        received > 0 and delay_violation_rate <= 1.0 - required_delay_ratio + 1e-12
    )
    result["jitter_sla_met"] = (
        True
        if jitter_bound_ms is None
        else result["jitter_ms"] is not None
        and float(result["jitter_ms"]) <= float(jitter_bound_ms)
    )
    result["loss_sla_met"] = (
        not sender_failed and loss_rate <= packet_loss_bound + 1e-12
    )
    result["sla_met"] = bool(
        result["delay_sla_met"]
        and result["jitter_sla_met"]
        and result["loss_sla_met"]
    )
    return result


def receive_probe_native(
    executable: str,
    group: str,
    port: int,
    duration: float,
    interface_ip: str,
    delay_bound_ms: float,
    delay_compliance_ratio: float,
    jitter_bound_ms: float | None,
    packet_loss_bound: float,
    grace_seconds: float = 1.0,
    expected_packets_hint: int = 0,
    ready_file: str | Path | None = None,
    stop_time_ns: int | None = None,
    expected_file: str | Path | None = None,
    receive_buffer_bytes: int = DEFAULT_RECEIVE_BUFFER_BYTES,
    ready_ack_socket: str | None = None,
    ready_ack_token: str | None = None,
    ready_request_id: int | None = None,
    ready_destination_id: int | None = None,
) -> dict[str, Any]:
    command = [
        executable,
        str(group),
        str(int(port)),
        f"{float(duration):.9f}",
        str(interface_ip),
        f"{float(delay_bound_ms):.9f}",
        f"{float(delay_compliance_ratio):.9f}",
        (
            f"{float(jitter_bound_ms):.9f}"
            if jitter_bound_ms is not None
            else "-1"
        ),
        f"{float(packet_loss_bound):.9f}",
        f"{float(grace_seconds):.9f}",
        str(int(expected_packets_hint)),
        str(ready_file) if ready_file is not None else "-",
        str(int(stop_time_ns or 0)),
        str(expected_file) if expected_file is not None else "-",
        str(int(receive_buffer_bytes)),
        str(ready_ack_socket) if ready_ack_socket else "-",
        str(ready_ack_token) if ready_ack_token else "-",
        str(int(ready_request_id or 0)),
        str(int(ready_destination_id or 0)),
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"native receiver returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if not isinstance(result, dict):
        raise RuntimeError("native receiver result must be a JSON object")
    return result


def emit(
    result: dict[str, Any], output: str | None, *, print_output: bool = True
) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    if print_output:
        print(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure multicast UDP SLA in Mininet.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sender = subparsers.add_parser("sender")
    sender.add_argument("--destination", required=True)
    sender.add_argument("--port", type=int, required=True)
    sender.add_argument("--duration", type=float, required=True)
    sender.add_argument("--packets-per-second", type=float, default=100.0)
    sender.add_argument("--payload-bytes", type=int, default=256)
    sender.add_argument("--dscp", type=int, default=0)
    sender.add_argument("--ttl", type=int, default=64)
    sender.add_argument("--max-catch-up-packets", type=int, default=0)
    sender.add_argument(
        "--sender-backend", choices=("python", "native"), default="python"
    )
    sender.add_argument("--native-sender-path", default=None)
    sender.add_argument("--native-realtime-priority", type=int, default=0)
    sender.add_argument("--ready-file", action="append", default=[])
    sender.add_argument("--ready-timeout", type=float, default=1.0)
    sender.add_argument("--activation-file", default=None)
    sender.add_argument("--activation-timeout", type=float, default=0.0)
    sender.add_argument("--post-activation-delay", type=float, default=0.0)
    sender.add_argument("--stop-time-ns", type=int, default=None)
    sender.add_argument("--schedule-origin-file", default=None)
    sender.add_argument("--stop-offset-ns", type=int, default=None)
    sender.add_argument("--minimum-duration", type=float, default=0.0)
    sender.add_argument("--expected-file", default=None)
    sender.add_argument("--output", default=None)

    receiver = subparsers.add_parser("receiver")
    receiver.add_argument("--group", required=True)
    receiver.add_argument("--port", type=int, required=True)
    receiver.add_argument("--duration", type=float, required=True)
    receiver.add_argument("--interface-ip", default="0.0.0.0")
    receiver.add_argument("--delay-bound-ms", type=float, required=True)
    receiver.add_argument("--delay-compliance-ratio", type=float, default=0.99)
    receiver.add_argument("--jitter-bound-ms", type=float, default=None)
    receiver.add_argument("--packet-loss-bound", type=float, default=0.001)
    receiver.add_argument("--grace-seconds", type=float, default=1.0)
    receiver.add_argument("--expected-packets", type=int, default=0)
    receiver.add_argument("--ready-file", default=None)
    receiver.add_argument("--ready-ack-socket", default=None)
    receiver.add_argument("--ready-ack-token", default=None)
    receiver.add_argument("--ready-request-id", type=int, default=None)
    receiver.add_argument("--ready-destination-id", type=int, default=None)
    receiver.add_argument("--stop-time-ns", type=int, default=None)
    receiver.add_argument("--schedule-origin-file", default=None)
    receiver.add_argument("--stop-offset-ns", type=int, default=None)
    receiver.add_argument("--expected-file", default=None)
    receiver.add_argument(
        "--receiver-backend", choices=("python", "native"), default="python"
    )
    receiver.add_argument("--native-receiver-path", default=None)
    receiver.add_argument(
        "--receive-buffer-bytes",
        type=int,
        default=DEFAULT_RECEIVE_BUFFER_BYTES,
    )
    receiver.add_argument("--output", default=None)

    agent = subparsers.add_parser("agent")
    agent.add_argument("--command-fifo", required=True)
    agent.add_argument("--ready-file", default=None)
    agent.add_argument("--sender-workers", type=int, default=16)
    agent.add_argument("--receiver-workers", type=int, default=32)
    return parser.parse_args(argv)


def execute_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.stop_time_ns is None and args.stop_offset_ns is not None:
        if not args.schedule_origin_file:
            raise ValueError("stop offset requires a schedule origin file")
        args.stop_time_ns = (
            schedule_origin_ns(args.schedule_origin_file) + args.stop_offset_ns
        )
    if args.mode == "sender":
        ready_started = time.monotonic()
        probe_start_time_ns = time.time_ns()
        deadline_remaining_at_start_ms = (
            (int(args.stop_time_ns) - probe_start_time_ns) / 1_000_000.0
            if args.stop_time_ns is not None
            else None
        )
        deadline_remaining_after_ready_ms = None
        try:
            activation = {}
            if args.activation_file:
                activation_timeout = float(args.activation_timeout)
                if args.stop_time_ns is not None:
                    activation_timeout = min(
                        activation_timeout,
                        max(
                            0.0,
                            (args.stop_time_ns - time.time_ns()) / 1e9
                            - args.minimum_duration,
                        ),
                    )
                if activation_timeout <= 0.0:
                    raise TimeoutError(
                        "only 0.000000s remains before tree activation"
                    )
                activation = wait_for_ready_files(
                    [args.activation_file], activation_timeout
                )
                if args.post_activation_delay < 0.0:
                    raise ValueError("post-activation delay cannot be negative")
                if args.post_activation_delay:
                    time.sleep(args.post_activation_delay)
            readiness = wait_for_ready_files(args.ready_file, args.ready_timeout)
            effective_duration = float(args.duration)
            if args.stop_time_ns is not None:
                deadline_remaining_after_ready_ms = (
                    int(args.stop_time_ns) - time.time_ns()
                ) / 1_000_000.0
                effective_duration = min(
                    effective_duration,
                    max(0.0, deadline_remaining_after_ready_ms / 1000.0),
                )
            if effective_duration < args.minimum_duration:
                raise TimeoutError(
                    f"only {effective_duration:.6f}s remains after receiver readiness"
                )
            planned_packets = max(
                1, int(math.ceil(effective_duration * args.packets_per_second))
            )
            expected_path = Path(args.expected_file) if args.expected_file else None
            if expected_path is not None and args.sender_backend != "native":
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                expected_path.write_text(str(planned_packets), encoding="ascii")
            try:
                if args.sender_backend == "native":
                    if not args.native_sender_path:
                        raise ValueError(
                            "native sender backend requires --native-sender-path"
                        )
                    result = send_probe_native(
                        args.native_sender_path,
                        args.destination,
                        args.port,
                        effective_duration,
                        args.packets_per_second,
                        args.payload_bytes,
                        args.dscp,
                        args.ttl,
                        args.stop_time_ns,
                        args.expected_file,
                        args.native_realtime_priority,
                    )
                else:
                    result = send_probe(
                        args.destination,
                        args.port,
                        effective_duration,
                        args.packets_per_second,
                        args.payload_bytes,
                        args.dscp,
                        args.ttl,
                        args.stop_time_ns,
                        args.max_catch_up_packets,
                    )
            except Exception:
                if expected_path is not None:
                    expected_path.write_text("-1", encoding="ascii")
                raise
            result.update(readiness)
            if activation:
                result["activation_wait_seconds"] = activation[
                    "ready_wait_seconds"
                ]
            result["status"] = "completed"
        except TimeoutError as exc:
            if args.expected_file:
                expected_path = Path(args.expected_file)
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                expected_path.write_text("-1", encoding="ascii")
            status = (
                "startup_deadline_missed"
                if str(exc).startswith("only ")
                else "ready_timeout"
            )
            result = {
                "mode": "sender",
                "destination": args.destination,
                "port": int(args.port),
                "planned_packets": 0,
                "sent_packets": 0,
                "ready_files": len(args.ready_file),
                "ready_wait_seconds": time.monotonic() - ready_started,
                "status": status,
                "error": str(exc),
            }
        result["probe_start_time_ns"] = probe_start_time_ns
        result["stop_time_ns"] = (
            int(args.stop_time_ns) if args.stop_time_ns is not None else None
        )
        result["requested_duration_seconds"] = float(args.duration)
        result["minimum_duration_seconds"] = float(args.minimum_duration)
        result["deadline_remaining_at_start_ms"] = (
            deadline_remaining_at_start_ms
        )
        result["deadline_remaining_after_ready_ms"] = (
            deadline_remaining_after_ready_ms
        )
        return result
    if args.mode == "receiver":
        if args.receiver_backend == "native":
            if not args.native_receiver_path:
                raise ValueError(
                    "native receiver backend requires --native-receiver-path"
                )
            return receive_probe_native(
                args.native_receiver_path,
                args.group,
                args.port,
                args.duration,
                args.interface_ip,
                args.delay_bound_ms,
                args.delay_compliance_ratio,
                args.jitter_bound_ms,
                args.packet_loss_bound,
                args.grace_seconds,
                args.expected_packets,
                args.ready_file,
                args.stop_time_ns,
                args.expected_file,
                args.receive_buffer_bytes,
                args.ready_ack_socket,
                args.ready_ack_token,
                args.ready_request_id,
                args.ready_destination_id,
            )
        return receive_probe(
            args.group,
            args.port,
            args.duration,
            args.interface_ip,
            args.delay_bound_ms,
            args.delay_compliance_ratio,
            args.jitter_bound_ms,
            args.packet_loss_bound,
            args.grace_seconds,
            args.expected_packets,
            args.ready_file,
            args.stop_time_ns,
            args.expected_file,
            args.receive_buffer_bytes,
            args.ready_ack_socket,
            args.ready_ack_token,
            args.ready_request_id,
            args.ready_destination_id,
        )
    raise ValueError(f"unsupported probe task mode {args.mode!r}")


def serve_agent(
    command_fifo: str | Path,
    ready_file: str | Path | None,
    sender_workers: int = 16,
    receiver_workers: int = 32,
) -> None:
    if sender_workers <= 0 or receiver_workers <= 0:
        raise ValueError("agent worker counts must be positive")
    fifo_path = Path(command_fifo)
    fifo_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fifo_path.unlink()
    except FileNotFoundError:
        pass
    os.mkfifo(fifo_path)
    executors = {
        "sender": ThreadPoolExecutor(
            max_workers=sender_workers, thread_name_prefix="probe-sender"
        ),
        "receiver": ThreadPoolExecutor(
            max_workers=receiver_workers, thread_name_prefix="probe-receiver"
        ),
    }
    schedule_condition = threading.Condition()
    scheduled: list[tuple[int, int, str, list[str]]] = []
    schedule_sequence = 0
    schedule_stopping = False
    origins: dict[str, int] = {}

    def worker(argv: list[str], submitted_ns: int) -> None:
        started_ns = time.monotonic_ns()
        task_args = parse_args(argv)
        if task_args.mode == "agent":
            raise ValueError("nested agent task is not allowed")
        try:
            result = execute_probe(task_args)
        except Exception as exc:
            try:
                send_receiver_ready_ack(
                    getattr(task_args, "ready_ack_socket", None),
                    getattr(task_args, "ready_ack_token", None),
                    accepted=False,
                    request_id=getattr(task_args, "ready_request_id", None),
                    destination_id=getattr(
                        task_args, "ready_destination_id", None
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
            except OSError:
                pass
            result = {
                "mode": task_args.mode,
                "status": "probe_error",
                "error": f"{type(exc).__name__}: {exc}",
                "sla_met": False,
            }
        result["agent_queue_wait_ms"] = max(
            0.0, (started_ns - submitted_ns) / 1_000_000.0
        )
        emit(result, getattr(task_args, "output", None), print_output=False)

    def start_worker(argv: list[str]) -> None:
        mode = argv[0] if argv else ""
        executor = executors.get(mode)
        if executor is None:
            raise ValueError(f"unsupported agent task mode {mode!r}")
        executor.submit(worker, argv, time.monotonic_ns())

    def scheduler() -> None:
        nonlocal schedule_stopping
        while True:
            with schedule_condition:
                if schedule_stopping:
                    return
                if not scheduled:
                    schedule_condition.wait(timeout=0.1)
                    continue
                offset_ns, _, origin_file, argv = scheduled[0]
            origin_ns = origins.get(origin_file)
            if origin_ns is None:
                try:
                    origin_ns = schedule_origin_ns(origin_file)
                except (OSError, ValueError):
                    with schedule_condition:
                        schedule_condition.wait(timeout=0.01)
                    continue
                origins[origin_file] = origin_ns
            remaining = (origin_ns + offset_ns - time.time_ns()) / 1e9
            if remaining > 0.0:
                with schedule_condition:
                    schedule_condition.wait(timeout=min(remaining, 0.05))
                continue
            with schedule_condition:
                if not scheduled:
                    continue
                _, _, _, argv = heapq.heappop(scheduled)
            start_worker(argv)

    scheduler_thread = threading.Thread(target=scheduler, daemon=True)
    scheduler_thread.start()

    descriptor = os.open(fifo_path, os.O_RDWR)
    if ready_file is not None:
        ready_path = Path(ready_file)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.write_text("ready", encoding="ascii")
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("operation") == "shutdown":
                    break
                argv = payload.get("argv")
                if not isinstance(argv, list) or not all(
                    isinstance(value, str) for value in argv
                ):
                    raise ValueError("agent task argv must be a string list")
                argv = list(argv)
                ack_socket = payload.get("ack_socket")
                ack_token = payload.get("ack_token")
                if ack_socket is not None or ack_token is not None:
                    if not ack_socket or not ack_token or not argv or argv[0] != "receiver":
                        raise ValueError(
                            "ready ACK fields require a receiver task and a complete token"
                        )
                    argv.extend(
                        [
                            "--ready-ack-socket",
                            str(ack_socket),
                            "--ready-ack-token",
                            str(ack_token),
                        ]
                    )
                    if payload.get("request_id") is not None:
                        argv.extend(
                            ["--ready-request-id", str(int(payload["request_id"]))]
                        )
                    if payload.get("destination_id") is not None:
                        argv.extend(
                            [
                                "--ready-destination-id",
                                str(int(payload["destination_id"])),
                            ]
                        )
                origin_file = payload.get("schedule_origin_file")
                offset_ns = payload.get("start_offset_ns")
                if origin_file is None and offset_ns is None:
                    start_worker(argv)
                    continue
                if not isinstance(origin_file, str) or not isinstance(offset_ns, int):
                    raise ValueError(
                        "scheduled agent task requires origin file and integer offset"
                    )
                with schedule_condition:
                    schedule_sequence += 1
                    heapq.heappush(
                        scheduled,
                        (offset_ns, schedule_sequence, origin_file, argv),
                    )
                    schedule_condition.notify()
    finally:
        with schedule_condition:
            schedule_stopping = True
            schedule_condition.notify_all()
        scheduler_thread.join(timeout=2.0)
        for executor in executors.values():
            executor.shutdown(wait=True)
        try:
            fifo_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    args = parse_args()
    if args.mode == "agent":
        serve_agent(
            args.command_fifo,
            args.ready_file,
            args.sender_workers,
            args.receiver_workers,
        )
        return 0
    result = execute_probe(args)
    emit(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
