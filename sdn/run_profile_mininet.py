#!/usr/bin/env python3
"""Run a JSON profile while staggering large-topology switch handshakes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shlex
import socket
import socketserver
import threading
import time
from functools import partial

from mininet.cli import CLI
from mininet.link import Link, TCLink
from mininet.log import info, setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController

from real_topology import ProfileTopo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--switch-start-delay", type=float, default=0.2)
    parser.add_argument("--link-mode", choices=("basic", "tc"), default="tc")
    parser.add_argument(
        "--qdisc",
        choices=(
            "htb",
            "htb_fq",
            "htb_fq_codel",
            "htb_prio",
            "htb_prio_fq_codel",
            "tbf",
            "hfsc",
        ),
        default="htb",
    )
    parser.add_argument("--max-queue-size", type=int, default=1000)
    parser.add_argument("--no-hosts", action="store_true")
    parser.add_argument(
        "--command-port",
        type=int,
        default=0,
        help="localhost JSON command server port; zero starts the interactive CLI",
    )
    parser.add_argument(
        "--host-dpids",
        default=None,
        help="comma-separated DPID subset; omitted means every profile host",
    )
    return parser.parse_args()


class RuntimeCommandServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_snmp(text):
    sections = {}
    lines = [line.split() for line in text.splitlines() if line.strip()]
    for header, values in zip(lines, lines[1:]):
        if not header or not values or header[0] != values[0]:
            continue
        name = header[0].rstrip(":")
        try:
            sections[name] = {
                key: int(value) for key, value in zip(header[1:], values[1:])
            }
        except ValueError:
            continue
    return sections


def qdisc_summary(text):
    blocks = re.split(r"(?=^qdisc\s)", text, flags=re.MULTILINE)
    root = next(
        (
            block
            for block in blocks
            if block.startswith("qdisc ")
            and " root " in block.splitlines()[0]
        ),
        text,
    )

    def value(pattern):
        match = re.search(pattern, root)
        return int(match.group(1)) if match else 0

    return {
        "dropped": value(r"dropped\s+(\d+)"),
        "overlimits": value(r"overlimits\s+(\d+)"),
        "requeues": value(r"requeues\s+(\d+)"),
        "backlog_packets": value(r"backlog\s+\S+\s+(\d+)p"),
    }


def network_diagnostics(network):
    host_udp = {}
    for host in network.hosts:
        sections = parse_snmp(host.cmd("cat /proc/net/snmp"))
        host_udp[host.name] = sections.get("Udp", {})

    interfaces = {}
    interface_nodes = {
        intf.name: node
        for node in [*network.hosts, *network.switches]
        for intf in node.intfList()
        if intf.name and intf.name != "lo"
    }
    for name in sorted(interface_nodes):
        # host.cmd executes inside that host's network namespace.
        output = interface_nodes[name].cmd(
            f"tc -s qdisc show dev {name} 2>&1"
        )
        interfaces[name] = {
            **qdisc_summary(output),
            "raw": output[-4000:],
        }
    return {"host_udp": host_udp, "qdiscs": interfaces}


def runtime_handler(network, stop_event):
    hosts = {host.name: host for host in network.hosts}
    locks = {name: threading.Lock() for name in hosts}
    fifo_locks = defaultdict(threading.Lock)
    executor = ThreadPoolExecutor(max_workers=32)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while True:
                line = self.rfile.readline(16 * 1024 * 1024 + 1)
                if not line:
                    return
                try:
                    if len(line) > 16 * 1024 * 1024:
                        raise ValueError("runtime command request exceeds 16 MiB")
                    request = json.loads(line.decode("utf-8"))
                    operation = request.get("operation")
                    if operation == "health":
                        response = {"ok": True, "hosts": sorted(hosts)}
                    elif operation == "clock":
                        response = {"ok": True, "time_ns": time.time_ns()}
                    elif operation == "diagnostics":
                        response = {"ok": True, **network_diagnostics(network)}
                    elif operation == "commands":
                        commands = request.get("commands")
                        if not isinstance(commands, list) or not commands:
                            raise ValueError("commands must be a non-empty list")

                        def execute(item):
                            host_name = str(item["host"])
                            command = str(item["command"])
                            if host_name not in hosts:
                                raise ValueError(f"unknown Mininet host {host_name}")
                            with locks[host_name]:
                                output = hosts[host_name].cmd(command)
                            return {"host": host_name, "output": str(output)[-2000:]}

                        outputs = list(executor.map(execute, commands))
                        response = {"ok": True, "outputs": outputs}
                    elif operation in {"fifo_messages", "fifo_messages_wait_ack"}:
                        messages = request.get("messages")
                        if not isinstance(messages, list) or not messages:
                            raise ValueError("messages must be a non-empty list")
                        wait_for_acks = operation == "fifo_messages_wait_ack"
                        ack_listener = None
                        ack_path = None
                        expected_acks = {}
                        if wait_for_acks:
                            ack_timeout = float(request.get("ack_timeout", 5.0))
                            if ack_timeout <= 0.0:
                                raise ValueError("ack_timeout must be positive")
                            ack_path = (
                                f"/tmp/hra-{os.getpid():x}-{threading.get_ident():x}-"
                                f"{time.monotonic_ns():x}.sock"
                            )
                            ack_listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                            ack_listener.bind(ack_path)
                            decorated = []
                            for index, item in enumerate(messages):
                                item_copy = dict(item)
                                payload_copy = dict(item_copy.get("payload") or {})
                                token = f"{time.monotonic_ns():x}-{index:x}"
                                payload_copy["ack_socket"] = ack_path
                                payload_copy["ack_token"] = token
                                item_copy["payload"] = payload_copy
                                decorated.append(item_copy)
                                expected_acks[token] = {
                                    "request_id": payload_copy.get("request_id"),
                                    "stage": payload_copy.get("stage"),
                                }
                            messages = decorated

                        def write_fifo(item):
                            fifo = str(item["fifo"])
                            payload = item.get("payload")
                            if not isinstance(payload, dict):
                                raise ValueError("FIFO payload must be an object")
                            for raw_path in item.get("remove_paths", []):
                                try:
                                    os.unlink(str(raw_path))
                                except FileNotFoundError:
                                    pass
                            encoded = (
                                json.dumps(
                                    payload, separators=(",", ":"), sort_keys=True
                                )
                                + "\n"
                            ).encode("utf-8")
                            with fifo_locks[fifo]:
                                fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                                try:
                                    written = os.write(fd, encoded)
                                finally:
                                    os.close(fd)
                            if written != len(encoded):
                                raise RuntimeError(
                                    f"short FIFO write for {fifo}: {written}/{len(encoded)}"
                                )
                            return {"fifo": fifo, "bytes": written}

                        dispatch_started = time.monotonic()
                        try:
                            outputs = list(executor.map(write_fifo, messages))
                            dispatch_finished = time.monotonic()
                            acks = []
                            if wait_for_acks:
                                deadline = dispatch_started + ack_timeout
                                pending = dict(expected_acks)
                                while pending:
                                    remaining = deadline - time.monotonic()
                                    if remaining <= 0.0:
                                        missing = list(pending.values())
                                        raise TimeoutError(
                                            "FIFO ACK timed out for "
                                            + json.dumps(missing, sort_keys=True)
                                        )
                                    ack_listener.settimeout(remaining)
                                    try:
                                        raw, _ = ack_listener.recvfrom(65536)
                                    except socket.timeout as exc:
                                        missing = list(pending.values())
                                        raise TimeoutError(
                                            "FIFO ACK timed out for "
                                            + json.dumps(missing, sort_keys=True)
                                        ) from exc
                                    ack = json.loads(raw.decode("utf-8"))
                                    token = str(ack.get("ack_token", ""))
                                    if token not in expected_acks or token not in pending:
                                        continue
                                    pending.pop(token)
                                    acks.append(ack)
                                rejected = [
                                    ack for ack in acks if not bool(ack.get("accepted"))
                                ]
                                if rejected:
                                    raise RuntimeError(
                                        "VNF registration rejected: "
                                        + json.dumps(rejected, sort_keys=True)
                                    )
                            completed = time.monotonic()
                            response = {
                                "ok": True,
                                "outputs": outputs,
                                "acks": acks,
                                "dispatch_ms": (
                                    dispatch_finished - dispatch_started
                                )
                                * 1000.0,
                                "ack_wait_ms": (
                                    completed - dispatch_finished
                                )
                                * 1000.0,
                                "total_ms": (completed - dispatch_started) * 1000.0,
                            }
                        finally:
                            if ack_listener is not None:
                                ack_listener.close()
                            if ack_path:
                                try:
                                    os.unlink(ack_path)
                                except FileNotFoundError:
                                    pass
                    elif operation == "shutdown":
                        stop_event.set()
                        response = {"ok": True, "shutting_down": True}
                    else:
                        raise ValueError(f"unsupported operation {operation!r}")
                except Exception as exc:
                    response = {"ok": False, "error": str(exc)}
                self.wfile.write(
                    (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
                )
                self.wfile.flush()

    return Handler


def install_fq_codel(network, limit=150, target_ms=5, interval_ms=100):
    installed = []
    for switch in network.switches:
        for interface in switch.intfList():
            name = str(interface.name or "")
            if not name or name == "lo":
                continue
            quoted = shlex.quote(name)
            current = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            match = re.search(r"qdisc netem ([0-9a-f]+):", current)
            if match is None:
                continue
            parent = f"{match.group(1)}:1"
            output = switch.cmd(
                f"tc qdisc replace dev {quoted} parent {parent} handle 20: "
                f"fq_codel limit {int(limit)} target {int(target_ms)}ms "
                f"interval {int(interval_ms)}ms 2>&1"
            )
            if output.strip():
                raise RuntimeError(f"failed to configure fq_codel on {name}: {output}")
            verify = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            if "qdisc fq_codel 20:" not in verify:
                raise RuntimeError(f"fq_codel verification failed on {name}: {verify}")
            installed.append(name)
    if not installed:
        raise RuntimeError("fq_codel requested but no netem link qdiscs were found")
    return installed


def install_fq(network, limit=1000, quantum=1514):
    installed = []
    for switch in network.switches:
        for interface in switch.intfList():
            name = str(interface.name or "")
            if not name or name == "lo":
                continue
            quoted = shlex.quote(name)
            current = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            match = re.search(r"qdisc netem ([0-9a-f]+):", current)
            if match is None:
                continue
            parent = f"{match.group(1)}:1"
            output = switch.cmd(
                f"tc qdisc replace dev {quoted} parent {parent} handle 20: "
                f"fq limit {int(limit)} flow_limit {int(limit)} "
                f"quantum {int(quantum)} initial_quantum {int(quantum) * 10} 2>&1"
            )
            if output.strip():
                raise RuntimeError(f"failed to configure fq on {name}: {output}")
            verify = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            if "qdisc fq 20:" not in verify:
                raise RuntimeError(f"fq verification failed on {name}: {verify}")
            installed.append(name)
    if not installed:
        raise RuntimeError("fq requested but no netem link qdiscs were found")
    return installed


def install_dscp_prio(network, limit=200):
    """Install strict EF/AF41/best-effort bands below Mininet's netem qdisc."""
    installed = []
    band_limit = max(20, int(limit))
    for switch in network.switches:
        for interface in switch.intfList():
            name = str(interface.name or "")
            if not name or name == "lo":
                continue
            quoted = shlex.quote(name)
            current = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            match = re.search(r"qdisc netem ([0-9a-f]+):", current)
            if match is None:
                continue
            parent = f"{match.group(1)}:1"
            commands = [
                f"tc qdisc replace dev {quoted} parent {parent} handle 20: "
                "prio bands 3 priomap 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2",
                f"tc qdisc replace dev {quoted} parent 20:1 handle 21: "
                f"pfifo limit {band_limit}",
                f"tc qdisc replace dev {quoted} parent 20:2 handle 22: "
                f"pfifo limit {band_limit}",
                f"tc qdisc replace dev {quoted} parent 20:3 handle 23: "
                f"pfifo limit {band_limit}",
                f"tc filter replace dev {quoted} protocol ip parent 20: prio 1 "
                "u32 match ip tos 0xb8 0xfc flowid 20:1",
                f"tc filter replace dev {quoted} protocol ip parent 20: prio 2 "
                "u32 match ip tos 0x88 0xfc flowid 20:2",
            ]
            output = switch.cmd(" && ".join(commands) + " 2>&1")
            if output.strip():
                raise RuntimeError(f"failed to configure DSCP prio on {name}: {output}")
            verify = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            if "qdisc prio 20:" not in verify:
                raise RuntimeError(f"DSCP prio verification failed on {name}: {verify}")
            installed.append(name)
    if not installed:
        raise RuntimeError("DSCP prio requested but no netem link qdiscs were found")
    return installed


def install_dscp_prio_fq_codel(network, limit=1000, quantum=1514):
    """Prioritize SLA classes while bounding queueing within every class."""
    installed = []
    band_limit = max(20, int(limit))
    for switch in network.switches:
        for interface in switch.intfList():
            name = str(interface.name or "")
            if not name or name == "lo":
                continue
            quoted = shlex.quote(name)
            current = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            match = re.search(r"qdisc netem ([0-9a-f]+):", current)
            if match is None:
                continue
            parent = f"{match.group(1)}:1"
            commands = [
                f"tc qdisc replace dev {quoted} parent {parent} handle 20: "
                "prio bands 3 priomap 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2",
                *[
                    f"tc qdisc replace dev {quoted} parent 20:{band} "
                    f"handle 2{band}: fq_codel limit {band_limit} "
                    f"quantum {int(quantum)} target 5ms interval 100ms"
                    for band in range(1, 4)
                ],
                f"tc filter replace dev {quoted} protocol ip parent 20: prio 1 "
                "u32 match ip tos 0xb8 0xfc flowid 20:1",
                f"tc filter replace dev {quoted} protocol ip parent 20: prio 2 "
                "u32 match ip tos 0x88 0xfc flowid 20:2",
            ]
            output = switch.cmd(" && ".join(commands) + " 2>&1")
            if output.strip():
                raise RuntimeError(
                    f"failed to configure DSCP prio+fq_codel on {name}: {output}"
                )
            verify = switch.cmd(f"tc qdisc show dev {quoted} 2>&1")
            if "qdisc prio 20:" not in verify or verify.count("qdisc fq_codel") < 3:
                raise RuntimeError(
                    f"DSCP prio+fq_codel verification failed on {name}: {verify}"
                )
            installed.append(name)
    if not installed:
        raise RuntimeError("DSCP prio+fq_codel requested but no netem qdiscs were found")
    return installed


def main():
    args = parse_args()
    if args.switch_start_delay < 0:
        raise ValueError("switch_start_delay cannot be negative")
    if args.max_queue_size <= 0:
        raise ValueError("max_queue_size must be positive")
    setLogLevel("info")
    if args.no_hosts and args.host_dpids:
        raise ValueError("--no-hosts and --host-dpids cannot be combined")
    host_dpids = (
        [int(value) for value in args.host_dpids.split(",") if value.strip()]
        if args.host_dpids
        else None
    )
    topology = ProfileTopo(
        args.profile,
        include_hosts=not args.no_hosts,
        host_dpids=host_dpids,
        max_queue_size=args.max_queue_size,
        qdisc=args.qdisc,
    )
    link_class = TCLink if args.link_mode == "tc" else Link
    network = Mininet(
        topo=topology,
        build=False,
        controller=None,
        switch=partial(OVSSwitch, batch=False),
        link=link_class,
    )
    network.addController(
        "c0",
        controller=RemoteController,
        ip=args.controller_ip,
        port=args.controller_port,
    )
    try:
        network.build()
        for controller in network.controllers:
            controller.start()
        total = len(network.switches)
        for index, switch in enumerate(network.switches, 1):
            switch.start(network.controllers)
            if args.switch_start_delay:
                time.sleep(args.switch_start_delay)
            if index % 25 == 0 or index == total:
                info(f"*** Connected {index}/{total} switches\n")
        if args.qdisc == "htb_fq":
            interfaces = install_fq(
                network,
                limit=args.max_queue_size,
            )
            info(f"*** Added fq to {len(interfaces)} directed link interfaces\n")
        elif args.qdisc == "htb_fq_codel":
            interfaces = install_fq_codel(
                network,
                limit=args.max_queue_size,
            )
            info(f"*** Added fq_codel to {len(interfaces)} directed link interfaces\n")
        elif args.qdisc == "htb_prio":
            interfaces = install_dscp_prio(
                network,
                limit=args.max_queue_size,
            )
            info(f"*** Added DSCP prio to {len(interfaces)} directed link interfaces\n")
        elif args.qdisc == "htb_prio_fq_codel":
            interfaces = install_dscp_prio_fq_codel(
                network,
                limit=args.max_queue_size,
            )
            info(
                "*** Added DSCP prio+fq_codel to "
                f"{len(interfaces)} directed link interfaces\n"
            )
        if args.command_port:
            stop_event = threading.Event()
            server = RuntimeCommandServer(
                ("0.0.0.0", args.command_port),
                runtime_handler(network, stop_event),
            )
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            info(f"*** Runtime command server: 0.0.0.0:{args.command_port}\n")
            stop_event.wait()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5.0)
        else:
            CLI(network)
    finally:
        network.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
