#!/usr/bin/env python3
"""Persistent sharded UDP VNF agent with request/stage accounting."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import selectors
import signal
import socket
import time
from typing import Any


DEFAULT_MAX_PACKETS_PER_SOCKET_EVENT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-fifo", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--initial-workers",
        type=int,
        default=0,
        help="initial worker count; zero starts all --workers for compatibility",
    )
    parser.add_argument(
        "--bindings-per-worker",
        type=int,
        default=0,
        help="spawn up to --workers when active bindings per worker reach this value",
    )
    parser.add_argument(
        "--prewarm-workers",
        action="store_true",
        help="pre-fork max workers and SIGSTOP idle workers for fast scale-out",
    )
    parser.add_argument("--drain-timeout-ms", type=float, default=100.0)
    parser.add_argument("--drain-idle-ms", type=float, default=5.0)
    parser.add_argument("--realtime-priority", type=int, default=0)
    parser.add_argument(
        "--max-packets-per-socket-event",
        type=int,
        default=DEFAULT_MAX_PACKETS_PER_SOCKET_EVENT,
        help="per-binding receive burst before yielding to another ready socket",
    )
    parser.add_argument(
        "--q0-max-packets-per-socket-event",
        type=int,
        default=0,
        help="Q0/DSCP-EF burst override; zero uses the general packet burst",
    )
    parser.add_argument("--prebound-port-base", type=int, default=30000)
    parser.add_argument(
        "--prebound-port-count",
        type=int,
        default=0,
        help="resident UDP receive endpoints shared by this DC; zero disables",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.initial_workers < 0 or args.initial_workers > args.workers:
        parser.error("--initial-workers must be between zero and --workers")
    if args.bindings_per_worker < 0:
        parser.error("--bindings-per-worker cannot be negative")
    if args.drain_timeout_ms <= 0.0 or args.drain_idle_ms <= 0.0:
        parser.error("drain timing values must be positive")
    if args.drain_idle_ms > args.drain_timeout_ms:
        parser.error("--drain-idle-ms cannot exceed --drain-timeout-ms")
    if not 0 <= args.realtime_priority <= 50:
        parser.error("--realtime-priority must be between 0 and 50")
    if args.max_packets_per_socket_event <= 0:
        parser.error("--max-packets-per-socket-event must be positive")
    if args.q0_max_packets_per_socket_event < 0:
        parser.error("--q0-max-packets-per-socket-event cannot be negative")
    if args.prebound_port_count < 0:
        parser.error("--prebound-port-count cannot be negative")
    if args.prebound_port_count and not (
        1024 <= args.prebound_port_base <= 65535
        and args.prebound_port_base + args.prebound_port_count <= 65536
    ):
        parser.error("prebound VNF port range must stay within 1024..65535")
    if args.prebound_port_count and int(args.initial_workers or args.workers) != args.workers:
        parser.error("prebound endpoints require all VNF workers to start resident")
    return args


def write_text(path: str | None, value: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, target)


def send_registration_ack(
    command: dict[str, Any],
    worker_id: int,
    *,
    accepted: bool,
    error: str | None = None,
) -> None:
    """Acknowledge a completed bind over local IPC instead of a shared file."""
    ack_socket = command.get("ack_socket")
    ack_token = command.get("ack_token")
    if not ack_socket or not ack_token:
        return
    payload = {
        "ack_token": str(ack_token),
        "operation": str(command.get("operation", "register")),
        "request_id": int(command["request_id"]),
        "stage": int(command["stage"]),
        "worker_id": int(worker_id),
        "accepted": bool(accepted),
        "ready_monotonic_ns": time.monotonic_ns(),
    }
    if error:
        payload["error"] = str(error)
    for key in (
        "drain_wait_ms",
        "drain_timed_out",
        "received",
        "forwarded",
        "dropped",
        "migration_epoch",
        "next_host",
        "next_port",
        "state",
    ):
        if key in command:
            payload[key] = command[key]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
        sender.sendto(encoded, str(ack_socket))


def binding_stats(binding: dict[str, Any], worker_id: int) -> dict[str, Any]:
    return {
        "request_id": binding["request_id"],
        "stage": binding["stage"],
        "vnf_type": str(binding["vnf_type"]),
        "received": binding["received"],
        "forwarded": binding["forwarded"],
        "dropped": binding["dropped"],
        "migration_epoch": binding["migration_epoch"],
        "elapsed_seconds": time.monotonic() - binding["started"],
        "runtime": "persistent_vnf_agent_sharded",
        "worker_id": worker_id,
        "drain_requested": binding.get("drain_requested_at") is not None,
        "drain_wait_ms": binding.get("drain_wait_ms", 0.0),
        "drain_timed_out": bool(binding.get("drain_timed_out", False)),
    }


def worker_main(
    worker_id: int,
    command_pipe: Connection,
    ready_event: Any,
    drain_timeout_seconds: float,
    drain_idle_seconds: float,
    realtime_priority: int,
    max_packets_per_socket_event: int,
    q0_max_packets_per_socket_event: int,
    prebound_ports: list[int],
) -> None:
    """Own one selector and one UDP sender for a stable binding shard."""
    stopped = False

    def request_stop(*_args: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if realtime_priority > 0:
        os.sched_setscheduler(
            0,
            os.SCHED_RR,
            os.sched_param(int(realtime_priority)),
        )
    selector = selectors.DefaultSelector()
    selector.register(command_pipe.fileno(), selectors.EVENT_READ, {"kind": "control"})
    bindings: dict[tuple[int, int], dict[str, Any]] = {}
    pooled_receivers: dict[int, socket.socket] = {}

    def create_receiver(port: int) -> socket.socket:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        receiver.setblocking(False)
        receiver.bind(("0.0.0.0", int(port)))
        return receiver

    for prebound_port in prebound_ports:
        pooled_receivers[int(prebound_port)] = create_receiver(int(prebound_port))

    def close_binding(binding: dict[str, Any]) -> None:
        key = (int(binding["request_id"]), int(binding["stage"]))
        bindings.pop(key, None)
        binding["closed"] = True
        try:
            selector.unregister(binding["socket"])
        except (KeyError, ValueError):
            pass
        receiver = binding["socket"]
        pooled_port = binding.get("pooled_port")
        if pooled_port is None:
            receiver.close()
        else:
            while True:
                try:
                    receiver.recvfrom(65535)
                except BlockingIOError:
                    break
            pooled_receivers[int(pooled_port)] = receiver
        binding["sender_socket"].close()
        write_text(
            binding.get("stats_output"),
            json.dumps(binding_stats(binding, worker_id), sort_keys=True),
        )

    def begin_drain(binding: dict[str, Any], command: dict[str, Any], close: bool) -> None:
        now = time.monotonic()
        binding["drain_requested_at"] = now
        binding["drain_last_packet_at"] = now
        binding["drain_deadline"] = now + max(
            0.001,
            float(command.get("drain_timeout_ms", drain_timeout_seconds * 1000.0))
            / 1000.0,
        )
        binding["drain_idle_seconds"] = max(
            0.0001,
            float(command.get("drain_idle_ms", drain_idle_seconds * 1000.0))
            / 1000.0,
        )
        binding["drain_ack"] = command.get("drain_ack")
        binding["close_after_drain"] = bool(close)
        binding["draining"] = True
        binding.pop("drain_completed_at", None)

    def finish_drain(binding: dict[str, Any], timed_out: bool) -> None:
        now = time.monotonic()
        binding["draining"] = False
        binding["drain_completed_at"] = now
        binding["drain_timed_out"] = bool(timed_out)
        binding["drain_wait_ms"] = (
            now - float(binding["drain_requested_at"])
        ) * 1000.0
        ack = {
            "request_id": binding["request_id"],
            "stage": binding["stage"],
            "worker_id": worker_id,
            "received": binding["received"],
            "forwarded": binding["forwarded"],
            "drain_wait_ms": binding["drain_wait_ms"],
            "timed_out": bool(timed_out),
        }
        write_text(binding.get("drain_ack"), json.dumps(ack, sort_keys=True))
        unregister_command = binding.get("unregister_command")
        if binding.get("close_after_drain", False):
            if isinstance(unregister_command, dict):
                unregister_command.update(
                    {
                        "drain_wait_ms": binding["drain_wait_ms"],
                        "drain_timed_out": bool(timed_out),
                        "received": binding["received"],
                        "forwarded": binding["forwarded"],
                    }
                )
            close_binding(binding)
            if isinstance(unregister_command, dict):
                send_registration_ack(
                    unregister_command, worker_id, accepted=True
                )

    def handle_command(command: dict[str, Any]) -> None:
        nonlocal stopped
        operation = command.get("operation")
        if operation == "shutdown":
            stopped = True
            return
        request_id = int(command["request_id"])
        stage = int(command["stage"])
        key = (request_id, stage)
        if operation == "register":
            receiver = None
            binding_sender = None
            pooled_port = None
            try:
                if key in bindings:
                    raise ValueError(f"duplicate VNF binding {key}")
                listen_port = int(command["listen_port"])
                pooled_port = listen_port if listen_port in prebound_ports else None
                if pooled_port is None:
                    receiver = create_receiver(listen_port)
                else:
                    receiver = pooled_receivers.pop(pooled_port, None)
                    if receiver is None:
                        raise ValueError(f"prebound VNF endpoint {pooled_port} is busy")
                binding_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                dscp = max(0, min(63, int(command.get("dscp", 0))))
                binding_sender.setsockopt(
                    socket.IPPROTO_IP, socket.IP_TOS, dscp << 2
                )
                binding = {
                    "kind": "binding",
                    "request_id": request_id,
                    "stage": stage,
                    "socket": receiver,
                    "pooled_port": pooled_port,
                    "sender_socket": binding_sender,
                    "next_host": str(command["next_host"]),
                    "next_port": int(command["next_port"]),
                    "vnf_type": str(command["vnf_type"]),
                    "dscp": dscp,
                    "cpu_work": max(0, int(command.get("cpu_work", 1))),
                    "drop_every": max(0, int(command.get("drop_every", 0))),
                    "processing_delay_us": max(
                        0, int(command.get("processing_delay_us", 0))
                    ),
                    "stats_output": command.get("stats_output"),
                    "received": max(0, int(command.get("restore_received", 0))),
                    "forwarded": max(0, int(command.get("restore_forwarded", 0))),
                    "dropped": max(0, int(command.get("restore_dropped", 0))),
                    "migration_epoch": max(
                        0, int(command.get("migration_epoch", 0))
                    ),
                    "closed": False,
                    "draining": False,
                    "started": time.monotonic(),
                }
                bindings[key] = binding
                selector.register(receiver, selectors.EVENT_READ, binding)
                write_text(command.get("ready_file"), "ready\n")
                send_registration_ack(command, worker_id, accepted=True)
            except Exception as exc:
                if key not in bindings:
                    if receiver is not None:
                        if pooled_port is None:
                            receiver.close()
                        else:
                            pooled_receivers[int(pooled_port)] = receiver
                    if binding_sender is not None:
                        binding_sender.close()
                try:
                    send_registration_ack(
                        command, worker_id, accepted=False, error=str(exc)
                    )
                finally:
                    raise
            return
        binding = bindings.get(key)
        if binding is None:
            if operation == "drain":
                write_text(
                    command.get("drain_ack"),
                    json.dumps(
                        {
                            "request_id": request_id,
                            "stage": stage,
                            "worker_id": worker_id,
                            "missing": True,
                        },
                        sort_keys=True,
                    ),
                )
            send_registration_ack(
                command, worker_id, accepted=False, error="missing_binding"
            )
            return
        if operation == "snapshot":
            state = {
                "schema": "udp_forwarder.v1",
                "received": int(binding["received"]),
                "forwarded": int(binding["forwarded"]),
                "dropped": int(binding["dropped"]),
                "migration_epoch": int(binding["migration_epoch"]),
                "next_host": str(binding["next_host"]),
                "next_port": int(binding["next_port"]),
            }
            command.update(
                {
                    "received": state["received"],
                    "forwarded": state["forwarded"],
                    "dropped": state["dropped"],
                    "migration_epoch": state["migration_epoch"],
                    "state": state,
                }
            )
            send_registration_ack(command, worker_id, accepted=True)
            return
        if operation == "restore":
            migration_epoch = int(command.get("migration_epoch", 0))
            if migration_epoch < int(binding["migration_epoch"]):
                send_registration_ack(
                    command,
                    worker_id,
                    accepted=False,
                    error="stale_migration_epoch",
                )
                return
            binding["received"] = max(
                int(binding["received"]), int(command.get("restore_received", 0))
            )
            binding["forwarded"] = max(
                int(binding["forwarded"]), int(command.get("restore_forwarded", 0))
            )
            binding["dropped"] = max(
                int(binding["dropped"]), int(command.get("restore_dropped", 0))
            )
            binding["migration_epoch"] = migration_epoch
            command.update(
                {
                    "received": binding["received"],
                    "forwarded": binding["forwarded"],
                    "dropped": binding["dropped"],
                    "migration_epoch": migration_epoch,
                }
            )
            send_registration_ack(command, worker_id, accepted=True)
            return
        if operation == "restore_delta":
            migration_epoch = int(command.get("migration_epoch", 0))
            if migration_epoch < int(binding["migration_epoch"]):
                send_registration_ack(
                    command,
                    worker_id,
                    accepted=False,
                    error="stale_migration_epoch",
                )
                return
            binding["received"] += max(0, int(command.get("delta_received", 0)))
            binding["forwarded"] += max(
                0, int(command.get("delta_forwarded", 0))
            )
            binding["dropped"] += max(0, int(command.get("delta_dropped", 0)))
            binding["migration_epoch"] = migration_epoch
            command.update(
                {
                    "received": binding["received"],
                    "forwarded": binding["forwarded"],
                    "dropped": binding["dropped"],
                    "migration_epoch": migration_epoch,
                }
            )
            send_registration_ack(command, worker_id, accepted=True)
            return
        if operation == "update_next":
            migration_epoch = int(
                command.get("migration_epoch", binding["migration_epoch"])
            )
            if migration_epoch < int(binding["migration_epoch"]):
                send_registration_ack(
                    command,
                    worker_id,
                    accepted=False,
                    error="stale_migration_epoch",
                )
                return
            next_host = str(command["next_host"])
            next_port = int(command["next_port"])
            if not 1 <= next_port <= 65535:
                raise ValueError("next_port must be in 1..65535")
            binding["next_host"] = next_host
            binding["next_port"] = next_port
            binding["migration_epoch"] = migration_epoch
            command.update(
                {
                    "next_host": next_host,
                    "next_port": next_port,
                    "migration_epoch": migration_epoch,
                }
            )
            send_registration_ack(command, worker_id, accepted=True)
            return
        if operation == "update_impairment":
            if "drop_every" in command:
                binding["drop_every"] = max(0, int(command["drop_every"]))
            if "processing_delay_us" in command:
                binding["processing_delay_us"] = max(
                    0, int(command["processing_delay_us"])
                )
            command.update(
                {
                    "drop_every": binding["drop_every"],
                    "processing_delay_us": binding["processing_delay_us"],
                    "migration_epoch": binding["migration_epoch"],
                }
            )
            send_registration_ack(command, worker_id, accepted=True)
            return
        if operation == "drain":
            begin_drain(binding, command, close=False)
            return
        if operation == "unregister":
            if binding.get("drain_completed_at") is not None:
                close_binding(binding)
                send_registration_ack(command, worker_id, accepted=True)
            else:
                binding["unregister_command"] = command
                begin_drain(binding, command, close=True)
            return
        raise ValueError(f"unsupported VNF agent operation {operation!r}")

    ready_event.set()
    try:
        while not stopped:
            drain_active = any(
                binding.get("draining", False) for binding in bindings.values()
            )
            selected_events = selector.select(timeout=0.002 if drain_active else 0.1)
            selected_events.sort(
                key=lambda row: (
                    row[0].data.get("kind") != "control",
                    -int(row[0].data.get("dscp", 0)),
                )
            )
            for selected, _ in selected_events:
                metadata = selected.data
                if metadata["kind"] == "control":
                    while command_pipe.poll():
                        command = command_pipe.recv()
                        handle_command(command)
                        if stopped:
                            break
                    continue

                binding = metadata
                if binding.get("closed", False):
                    continue
                packet_burst = (
                    q0_max_packets_per_socket_event
                    if int(binding.get("dscp", 0)) >= 46
                    and q0_max_packets_per_socket_event > 0
                    else max_packets_per_socket_event
                )
                for _ in range(packet_burst):
                    try:
                        payload, _ = binding["socket"].recvfrom(65535)
                    except BlockingIOError:
                        break
                    except OSError as exc:
                        if binding.get("closed", False) or exc.errno == errno.EBADF:
                            break
                        raise
                    binding["received"] += 1
                    if binding.get("draining", False):
                        binding["drain_last_packet_at"] = time.monotonic()
                    if (
                        binding["drop_every"]
                        and binding["received"] % binding["drop_every"] == 0
                    ):
                        binding["dropped"] += 1
                        continue
                    if binding["processing_delay_us"]:
                        time.sleep(binding["processing_delay_us"] / 1_000_000.0)
                    value = payload
                    for _ in range(binding["cpu_work"]):
                        value = hashlib.sha256(value).digest()
                    binding["sender_socket"].sendto(
                        payload, (binding["next_host"], binding["next_port"])
                    )
                    binding["forwarded"] += 1

            now = time.monotonic()
            for binding in list(bindings.values()):
                if not binding.get("draining", False):
                    continue
                timed_out = now >= float(binding["drain_deadline"])
                idle = (
                    now - float(binding["drain_last_packet_at"])
                    >= float(binding["drain_idle_seconds"])
                )
                if timed_out or idle:
                    finish_drain(binding, timed_out)
    finally:
        for binding in list(bindings.values()):
            close_binding(binding)
        for receiver in pooled_receivers.values():
            receiver.close()
        try:
            selector.unregister(command_pipe.fileno())
        except (KeyError, ValueError):
            pass
        command_pipe.close()
        selector.close()


def shard_index(request_id: int, stage: int, workers: int) -> int:
    return (int(request_id) * 31 + int(stage)) % int(workers)


def main() -> int:
    args = parse_args()
    fifo_path = Path(args.command_fifo)
    fifo_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fifo_path.unlink()
    except FileNotFoundError:
        pass
    os.mkfifo(fifo_path)
    fifo_fd = os.open(fifo_path, os.O_RDWR | os.O_NONBLOCK)
    selector = selectors.DefaultSelector()
    selector.register(fifo_fd, selectors.EVENT_READ)
    control_buffer = bytearray()
    stopped = False

    def request_stop(*_args: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    context = multiprocessing.get_context("fork")
    initial_workers = int(args.initial_workers or args.workers)
    dynamic_workers = bool(
        args.bindings_per_worker > 0 and initial_workers < args.workers
    )
    workers: dict[int, tuple[multiprocessing.Process, Connection]] = {}
    active_workers: set[int] = set()
    worker_bindings: dict[int, int] = {}
    binding_workers: dict[tuple[int, int], int] = {}
    next_worker_id = 0
    peak_workers = 0
    scale_out_events = 0
    scale_in_events = 0

    def spawn_worker() -> int:
        nonlocal next_worker_id
        worker_id = next_worker_id
        next_worker_id += 1
        receive_pipe, send_pipe = context.Pipe(duplex=False)
        ready_event = context.Event()
        process = context.Process(
            target=worker_main,
            args=(
                worker_id,
                receive_pipe,
                ready_event,
                args.drain_timeout_ms / 1000.0,
                args.drain_idle_ms / 1000.0,
                args.realtime_priority,
                args.max_packets_per_socket_event,
                args.q0_max_packets_per_socket_event,
                [
                    args.prebound_port_base + offset
                    for offset in range(args.prebound_port_count)
                    if offset % args.workers == worker_id
                ],
            ),
            name=f"vnf-worker-{worker_id}",
        )
        process.start()
        receive_pipe.close()
        if not ready_event.wait(timeout=5.0) or not process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            send_pipe.close()
            raise RuntimeError(f"VNF worker {worker_id} failed to start")
        workers[worker_id] = (process, send_pipe)
        worker_bindings[worker_id] = 0
        return worker_id

    def stop_worker(worker_id: int) -> None:
        process, pipe = workers.pop(worker_id)
        active_workers.discard(worker_id)
        worker_bindings.pop(worker_id, None)
        if process.is_alive():
            try:
                os.kill(process.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            try:
                pipe.send({"operation": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        pipe.close()

    resident_workers = args.workers if args.prewarm_workers and dynamic_workers else initial_workers
    for index in range(resident_workers):
        worker_id = spawn_worker()
        if index < initial_workers:
            active_workers.add(worker_id)
        else:
            os.kill(workers[worker_id][0].pid, signal.SIGSTOP)
    peak_workers = len(active_workers)
    def write_ready_state() -> None:
        write_text(
            args.ready_file,
            json.dumps({
                "ready": True,
                "workers": len(active_workers),
                "resident_workers": len(workers),
                "peak_workers": peak_workers,
                "max_workers": args.workers,
                "initial_workers": initial_workers,
                "bindings_per_worker": args.bindings_per_worker,
                "dynamic_workers": dynamic_workers,
                "prewarm_workers": bool(args.prewarm_workers),
                "max_packets_per_socket_event": args.max_packets_per_socket_event,
                "q0_max_packets_per_socket_event": args.q0_max_packets_per_socket_event,
                "prebound_port_base": args.prebound_port_base,
                "prebound_port_count": args.prebound_port_count,
                "active_bindings": len(binding_workers),
                "scale_out_events": scale_out_events,
                "scale_in_events": scale_in_events,
            }, sort_keys=True) + "\n",
        )

    write_ready_state()

    try:
        while not stopped:
            for process, _ in workers.values():
                if not process.is_alive():
                    raise RuntimeError(
                        f"VNF worker {process.name} exited with code {process.exitcode}"
                    )
            for _, _ in selector.select(timeout=0.05):
                while True:
                    try:
                        chunk = os.read(fifo_fd, 65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        break
                    control_buffer.extend(chunk)
                while b"\n" in control_buffer:
                    line, _, remainder = control_buffer.partition(b"\n")
                    control_buffer[:] = remainder
                    if not line.strip():
                        continue
                    command = json.loads(line.decode("utf-8"))
                    if command.get("operation") == "shutdown":
                        stopped = True
                        break
                    operation = str(command.get("operation"))
                    key = (int(command["request_id"]), int(command["stage"]))
                    if operation == "register":
                        if key in binding_workers:
                            raise ValueError(f"duplicate VNF binding {key}")
                        if (
                            dynamic_workers
                            and len(active_workers) < args.workers
                            and len(binding_workers)
                            >= len(active_workers) * args.bindings_per_worker
                        ):
                            dormant = sorted(set(workers) - active_workers)
                            if dormant:
                                resumed_worker = dormant[0]
                                os.kill(
                                    workers[resumed_worker][0].pid,
                                    signal.SIGCONT,
                                )
                            else:
                                resumed_worker = spawn_worker()
                            active_workers.add(resumed_worker)
                            peak_workers = max(peak_workers, len(active_workers))
                            scale_out_events += 1
                            write_ready_state()
                        listen_port = int(command["listen_port"])
                        if (
                            args.prebound_port_count
                            and args.prebound_port_base
                            <= listen_port
                            < args.prebound_port_base + args.prebound_port_count
                        ):
                            worker_id = (
                                listen_port - args.prebound_port_base
                            ) % args.workers
                            if worker_id not in active_workers:
                                raise RuntimeError(
                                    f"prebound port {listen_port} belongs to an inactive worker"
                                )
                        else:
                            worker_id = min(
                                active_workers,
                                key=lambda value: (worker_bindings[value], value),
                            )
                        binding_workers[key] = worker_id
                        worker_bindings[worker_id] += 1
                    else:
                        worker_id = binding_workers.get(key, -1)
                        if worker_id < 0:
                            if operation == "drain":
                                write_text(
                                    command.get("drain_ack"),
                                    json.dumps(
                                        {
                                            "request_id": key[0],
                                            "stage": key[1],
                                            "missing": True,
                                        },
                                        sort_keys=True,
                                    ),
                                )
                            send_registration_ack(
                                command,
                                -1,
                                accepted=False,
                                error="missing_binding",
                            )
                            continue
                    workers[worker_id][1].send(command)
                    if operation == "unregister":
                        binding_workers.pop(key, None)
                        worker_bindings[worker_id] -= 1
                        if (
                            dynamic_workers
                            and not args.prewarm_workers
                            and len(active_workers) > initial_workers
                        ):
                            idle_workers = [
                                value
                                for value, count in worker_bindings.items()
                                if count == 0 and value in active_workers
                            ]
                            if idle_workers:
                                idle_worker = max(idle_workers)
                                active_workers.remove(idle_worker)
                                stop_worker(idle_worker)
                                scale_in_events += 1
    finally:
        for worker_id in list(workers):
            stop_worker(worker_id)
        selector.unregister(fifo_fd)
        os.close(fifo_fd)
        selector.close()
        try:
            fifo_path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
