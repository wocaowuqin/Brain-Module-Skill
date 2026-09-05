#!/usr/bin/env python3
"""Run no-migration, reactive, and predictive VNF migration on Mininet/Ryu."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sdn.ryu_client import RyuSFTClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--pps", type=float, default=250.0)
    parser.add_argument("--hotspot-at", type=float, default=2.5)
    parser.add_argument("--reactive-delay", type=float, default=0.5)
    parser.add_argument("--prediction-lead", type=float, default=0.2)
    parser.add_argument("--processing-delay-us", type=int, default=6000)
    parser.add_argument("--delay-sla-ms", type=float, default=10.0)
    parser.add_argument("--loss-sla", type=float, default=0.01)
    parser.add_argument("--command-port", type=int, default=8876)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "vnf_migration_benchmark"
    )
    return parser.parse_args()


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{resolved.as_posix().split(':', 1)[1].lstrip('/')}"


def runtime_request(host: str, port: int, payload: dict, timeout: float) -> dict:
    raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.sendall(raw)
        with connection.makefile("rb") as reader:
            response = reader.readline(16 * 1024 * 1024 + 1)
    result = json.loads(response.decode())
    if not result.get("ok"):
        raise RuntimeError(result.get("error"))
    return result


def wait_runtime(host: str, port: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        try:
            return runtime_request(host, port, {"operation": "health"}, 2.0)
        except Exception as exc:
            error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Mininet runtime unavailable: {error}")


def runtime_commands(host: str, port: int, commands: list[tuple[str, str]], timeout: float):
    return runtime_request(
        host,
        port,
        {"operation": "commands", "commands": [
            {"host": node, "command": command} for node, command in commands
        ]},
        timeout,
    )


def fifo_command(
    host: str, port: int, fifo: str, payload: dict, timeout: float
) -> dict:
    result = runtime_request(
        host,
        port,
        {"operation": "fifo_messages_wait_ack", "ack_timeout": timeout,
         "messages": [{"fifo": fifo, "payload": payload}]},
        timeout + 2.0,
    )
    if len(result.get("acks", [])) != 1:
        raise RuntimeError(f"invalid VNF ACK response: {result}")
    ack = result["acks"][0]
    ack["batch_total_ms"] = float(result["total_ms"])
    return ack


def wait_controller(client: RyuSFTClient, switches: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    error = None
    while time.monotonic() < deadline:
        try:
            status = client.status()
            if len(status.get("datapaths", [])) >= switches:
                return status
        except Exception as exc:
            error = exc
        time.sleep(0.2)
    raise RuntimeError(f"controller topology unavailable: {error}")


def plan(request_id: int, base_port: int, migrated: bool) -> dict[str, Any]:
    stage1_node = 4 if migrated else 3
    stage1_port = base_port + (3 if migrated else 1)
    return {
        "request_id": request_id,
        "segments": [
            {"stage": 0, "target_ip": "10.0.0.2", "udp_port": base_port,
             "path": [1, 2], "switch_outputs": {"1": [2], "2": [1]}},
            {"stage": 1, "target_ip": f"10.0.0.{stage1_node}",
             "udp_port": stage1_port, "path": [2, stage1_node],
             "switch_outputs": {"2": [4 if migrated else 3], str(stage1_node): [1]}},
            {"stage": 2, "target_ip": "10.0.0.5", "udp_port": base_port + 2,
             "path": [stage1_node, 5],
             "switch_outputs": {str(stage1_node): [3], "5": [1]}},
        ],
        "multicast": {
            "group_id": request_id,
            "root_dpid": 5,
            "dst_ip": f"239.250.0.{request_id - 100}",
            "udp_port": 15000 + request_id,
            "switch_outputs": {"5": [4], "6": [1]},
        },
    }


def register_payload(
    request_id: int,
    stage: int,
    listen_port: int,
    next_host: str,
    next_port: int,
    restore: dict | None = None,
    epoch: int = 0,
) -> dict:
    payload = {
        "operation": "register", "request_id": request_id, "stage": stage,
        "listen_port": listen_port, "next_host": next_host, "next_port": next_port,
        "vnf_type": stage, "dscp": 46, "migration_epoch": epoch,
    }
    if restore:
        payload.update(
            restore_received=int(restore["received"]),
            restore_forwarded=int(restore["forwarded"]),
            restore_dropped=int(restore["dropped"]),
        )
    return payload


def sleep_until(started: float, offset: float) -> None:
    remaining = started + offset - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def perform_migration(
    client: RyuSFTClient,
    runtime_host: str,
    command_port: int,
    fifos: dict[str, str],
    request_id: int,
    base_port: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    old_snapshot = fifo_command(
        runtime_host, command_port, fifos["h3"],
        {"operation": "snapshot", "request_id": request_id, "stage": 1}, timeout,
    )
    state = old_snapshot["state"]
    target_register = fifo_command(
        runtime_host, command_port, fifos["h4"],
        register_payload(request_id, 1, base_port + 3, "10.0.0.5", base_port + 2,
                         restore=state, epoch=1), timeout,
    )
    prepared = client.prepare_sfc_migration(plan(request_id, base_port, True))
    switched = fifo_command(
        runtime_host, command_port, fifos["h2"],
        {"operation": "update_next", "request_id": request_id, "stage": 0,
         "next_host": "10.0.0.4", "next_port": base_port + 3,
         "migration_epoch": 1}, timeout,
    )
    drain = fifo_command(
        runtime_host, command_port, fifos["h3"],
        {"operation": "drain", "request_id": request_id, "stage": 1,
         "drain_timeout_ms": 250.0, "drain_idle_ms": 8.0}, timeout,
    )
    final_snapshot = fifo_command(
        runtime_host, command_port, fifos["h3"],
        {"operation": "snapshot", "request_id": request_id, "stage": 1}, timeout,
    )
    final_state = final_snapshot["state"]
    delta = {
        key: max(0, int(final_state[key]) - int(state[key]))
        for key in ("received", "forwarded", "dropped")
    }
    merged = fifo_command(
        runtime_host, command_port, fifos["h4"],
        {"operation": "restore_delta", "request_id": request_id, "stage": 1,
         "delta_received": delta["received"],
         "delta_forwarded": delta["forwarded"],
         "delta_dropped": delta["dropped"], "migration_epoch": 1}, timeout,
    )
    unregistered = fifo_command(
        runtime_host, command_port, fifos["h3"],
        {"operation": "unregister", "request_id": request_id, "stage": 1}, timeout,
    )
    committed = client.commit_sfc_migration(
        request_id, prepared["migration_token"], drain_seconds=0.01
    )
    return {
        "success": True,
        "total_ms": (time.monotonic() - started) * 1000.0,
        "switch_ms": switched["batch_total_ms"],
        "precopy_state": state,
        "final_source_state": final_state,
        "state_delta": delta,
        "merged_target_state": merged.get("state", {
            key: merged.get(key) for key in ("received", "forwarded", "dropped")
        }),
        "prepare": prepared, "commit": committed, "drain": drain,
        "target_register": target_register, "source_unregister": unregistered,
    }


def run_policy(
    policy: str,
    index: int,
    args: argparse.Namespace,
    client: RyuSFTClient,
    runtime_host: str,
    fifos: dict[str, str],
    traffic_wsl: str,
    output_wsl: str,
) -> dict[str, Any]:
    request_id = 100 + index
    base_port = 22000 + index * 10
    initial = plan(request_id, base_port, False)
    group = initial["multicast"]["dst_ip"]
    receiver_port = initial["multicast"]["udp_port"]
    receiver_file = args.output_dir / f"{policy}_receiver.json"
    sender_file = args.output_dir / f"{policy}_sender.json"
    for host, stage, port, next_host, next_port in (
        ("h5", 2, base_port + 2, group, receiver_port),
        ("h3", 1, base_port + 1, "10.0.0.5", base_port + 2),
        ("h2", 0, base_port, "10.0.0.3", base_port + 1),
    ):
        fifo_command(
            runtime_host, args.command_port, fifos[host],
            register_payload(request_id, stage, port, next_host, next_port), args.timeout,
        )
    client.install_sfc(initial)

    receiver_command = (
        f"rm -f {shlex.quote(output_wsl + '/' + receiver_file.name)}; "
        f"nohup python3 {shlex.quote(traffic_wsl)} receive --group {shlex.quote(group)} "
        f"--port {receiver_port} --duration {args.duration} --grace 0.7 "
        f"--output {shlex.quote(output_wsl + '/' + receiver_file.name)} "
        ">/tmp/vnf-migration-receiver.log 2>&1 &"
    )
    sender_command = (
        f"rm -f {shlex.quote(output_wsl + '/' + sender_file.name)}; "
        f"nohup python3 {shlex.quote(traffic_wsl)} send --host 10.0.0.2 "
        f"--port {base_port} --duration {args.duration} --pps {args.pps} "
        f"--output {shlex.quote(output_wsl + '/' + sender_file.name)} "
        ">/tmp/vnf-migration-sender.log 2>&1 &"
    )
    runtime_commands(runtime_host, args.command_port, [("h6", receiver_command)], args.timeout)
    time.sleep(0.2)
    started = time.monotonic()
    runtime_commands(runtime_host, args.command_port, [("h1", sender_command)], args.timeout)
    migration = None
    if policy == "predictive-trace-oracle":
        sleep_until(started, max(0.0, args.hotspot_at - args.prediction_lead))
        migration = perform_migration(
            client, runtime_host, args.command_port, fifos, request_id, base_port, args.timeout
        )
        sleep_until(started, args.hotspot_at)
        # The hotspot still occurs on the old DC, but proactive migration has
        # already removed this service from that node.
    else:
        sleep_until(started, args.hotspot_at)
        fifo_command(
            runtime_host, args.command_port, fifos["h3"],
            {"operation": "update_impairment", "request_id": request_id, "stage": 1,
             "processing_delay_us": args.processing_delay_us}, args.timeout,
        )
        if policy == "threshold-greedy":
            sleep_until(started, args.hotspot_at + args.reactive_delay)
            migration = perform_migration(
                client, runtime_host, args.command_port, fifos, request_id, base_port, args.timeout
            )
    sleep_until(started, args.duration + 1.0)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and (
        not receiver_file.is_file() or not sender_file.is_file()
    ):
        time.sleep(0.05)
    if not receiver_file.is_file() or not sender_file.is_file():
        logs = runtime_commands(
            runtime_host, args.command_port,
            [("h6", "tail -100 /tmp/vnf-migration-receiver.log"),
             ("h1", "tail -100 /tmp/vnf-migration-sender.log")], args.timeout,
        )
        raise RuntimeError(f"traffic output missing: {logs}")
    receiver = json.loads(receiver_file.read_text(encoding="utf-8"))
    sender = json.loads(sender_file.read_text(encoding="utf-8"))
    sent = int(sender["sent"])
    received = int(receiver["received_unique"])
    loss_rate = max(0.0, (sent - received) / sent)
    strict_sla = (
        loss_rate <= args.loss_sla and float(receiver["p95_delay_ms"]) <= args.delay_sla_ms
    )
    hotspot_exposure = (
        args.duration - args.hotspot_at if policy == "no-migration"
        else args.reactive_delay if policy == "threshold-greedy" else 0.0
    )
    row = {
        "policy": policy,
        "sent_packets": sent,
        "received_packets": received,
        "packet_loss_rate": loss_rate,
        "mean_delay_ms": float(receiver["mean_delay_ms"]),
        "p95_delay_ms": float(receiver["p95_delay_ms"]),
        "p99_delay_ms": float(receiver["p99_delay_ms"]),
        "max_delay_ms": float(receiver["max_delay_ms"]),
        "max_packet_gap_ms": float(receiver["max_gap_ms"]),
        "migration_count": int(migration is not None),
        "migration_success": bool(migration and migration["success"]),
        "migration_ms": float(migration["total_ms"]) if migration else 0.0,
        "switch_ms": float(migration["switch_ms"]) if migration else 0.0,
        "hotspot_exposure_seconds": hotspot_exposure,
        "strict_sla_met": strict_sla,
        "delay_sla_ms": args.delay_sla_ms,
        "loss_sla": args.loss_sla,
        "migration": migration,
    }
    try:
        client.delete_sfc(request_id)
    finally:
        active_stage1_host = "h4" if migration else "h3"
        for host, stage in (("h2", 0), (active_stage1_host, 1), ("h5", 2)):
            try:
                fifo_command(
                    runtime_host, args.command_port, fifos[host],
                    {"operation": "unregister", "request_id": request_id, "stage": stage},
                    args.timeout,
                )
            except Exception:
                pass
    return row


def write_outputs(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary = {
        "experiment": "stateful_vnf_make_before_break_mininet_ryu",
        "predictive_policy_note": (
            "predictive-trace-oracle uses known hotspot time; it is an upper-bound "
            "trigger baseline, not a trained traffic predictor"
        ),
        "state_scope": (
            "udp_forwarder.v1 counters, periodic-drop phase, and migration epoch; "
            "not firewall/NAT application state"
        ),
        "parameters": vars(args) | {"output_dir": str(args.output_dir.resolve())},
        "results": rows,
    }
    (args.output_dir / "migration_results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = [key for key in rows[0] if key != "migration"]
    with (args.output_dir / "migration_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    try:
        import matplotlib.pyplot as plt
        labels = ["No migration", "Threshold", "Predictive"]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        axes[0].bar(labels, [row["p95_delay_ms"] for row in rows], color=["#c94c4c", "#d69e2e", "#2f855a"])
        axes[0].axhline(args.delay_sla_ms, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("P95 delay (ms)")
        axes[0].set_title("Tail latency")
        axes[1].bar(labels, [100 * row["packet_loss_rate"] for row in rows], color=["#c94c4c", "#d69e2e", "#2f855a"])
        axes[1].axhline(100 * args.loss_sla, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylabel("Packet loss (%)")
        axes[1].set_title("Service continuity")
        axes[2].bar(labels, [row["hotspot_exposure_seconds"] for row in rows], color=["#c94c4c", "#d69e2e", "#2f855a"])
        axes[2].set_ylabel("Hotspot exposure (s)")
        axes[2].set_title("Overload avoidance")
        for axis in axes:
            axis.tick_params(axis="x", rotation=18)
            axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / "migration_comparison.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass


def main() -> int:
    args = parse_args()
    if not 1.0 < args.hotspot_at < args.duration:
        raise ValueError("hotspot-at must be inside the experiment duration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile = ROOT / "sdn" / "topologies" / "vnf_migration_6.json"
    traffic = ROOT / "scripts" / "vnf_migration_traffic.py"
    native = ROOT / "sdn" / f".vnf-agent-migration-{os.getpid()}"
    mininet_log = (args.output_dir / "mininet.log").open("wb")
    wsl = ["wsl.exe", "-d", args.distro, "--"]
    mininet = None
    runtime_host = None
    try:
        subprocess.run(
            wsl + ["gcc", "-O3", "-std=c11", wsl_path(ROOT / "sdn" / "vnf_agent_native.c"),
                   "-o", wsl_path(native)], check=True,
        )
        for service in ("ryu-controller.service", "ryu-sft-controller.service", "ryu-sft-static-controller.service"):
            subprocess.run(wsl + ["sudo", "-n", "systemctl", "stop", service], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(wsl + ["sudo", "-n", "mn", "-c"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(
            wsl + ["sudo", "-n", "systemctl", "start", "ryu-sft-static-controller.service"],
            check=True,
        )
        ip_output = subprocess.run(
            wsl + ["hostname", "-I"], check=True, capture_output=True, text=True
        ).stdout.strip()
        runtime_host = ip_output.split()[0]
        client = RyuSFTClient(f"http://{runtime_host}:8080", timeout=args.timeout)
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                client.status()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        client.configure(topology_file=wsl_path(profile), stats_interval=0.25,
                         sfc_barrier_mode="staged")
        mininet = subprocess.Popen(
            wsl + [
                "sudo", "-n", "python3", wsl_path(ROOT / "sdn" / "run_profile_mininet.py"),
                "--profile", wsl_path(profile), "--controller-ip", "127.0.0.1",
                "--controller-port", "6653", "--switch-start-delay", "0.05",
                "--link-mode", "tc", "--qdisc", "htb_fq_codel",
                "--max-queue-size", "150", "--command-port", str(args.command_port),
            ],
            stdin=subprocess.DEVNULL, stdout=mininet_log, stderr=subprocess.STDOUT,
        )
        wait_runtime(runtime_host, args.command_port, args.timeout)
        wait_controller(client, 6, args.timeout)
        neighbor_commands = []
        for node in range(1, 7):
            peers = []
            for peer in range(1, 7):
                if peer == node:
                    continue
                mac = f"02:00:00:00:00:{peer:02x}"
                peers.append(
                    f"ip neigh replace 10.0.0.{peer} lladdr {mac} nud permanent dev h{node}-eth0"
                )
            neighbor_commands.append((f"h{node}", "; ".join(peers)))
        neighbor_commands.append(("h5", "ip route replace 239.0.0.0/8 dev h5-eth0"))
        neighbor_commands.append(("h6", "ip route replace 239.0.0.0/8 dev h6-eth0"))
        runtime_commands(runtime_host, args.command_port, neighbor_commands, args.timeout)

        fifos = {host: f"/tmp/vnf-migration-{host}.fifo" for host in ("h2", "h3", "h4", "h5")}
        start_commands = []
        for host, fifo in fifos.items():
            ready = f"/tmp/vnf-migration-{host}.ready"
            log = f"/tmp/vnf-migration-{host}.log"
            start_commands.append(
                (host, f"rm -f {fifo} {ready}; nohup {shlex.quote(wsl_path(native))} "
                       f"--command-fifo {fifo} --ready-file {ready} "
                       f"--drain-timeout-ms 250 --drain-idle-ms 8 >{log} 2>&1 &")
            )
        runtime_commands(runtime_host, args.command_port, start_commands, args.timeout)
        time.sleep(0.5)
        checks = runtime_commands(
            runtime_host, args.command_port,
            [(host, f"test -p {fifo} && echo ready") for host, fifo in fifos.items()], args.timeout,
        )
        if any("ready" not in row["output"] for row in checks["outputs"]):
            raise RuntimeError(f"VNF agents did not start: {checks}")

        rows = []
        for index, policy in enumerate(
            ("no-migration", "threshold-greedy", "predictive-trace-oracle"), start=1
        ):
            rows.append(
                run_policy(policy, index, args, client, runtime_host, fifos,
                           wsl_path(traffic), wsl_path(args.output_dir))
            )
            print(json.dumps(rows[-1], indent=2, sort_keys=True))
        write_outputs(args, rows)
        print(json.dumps({"ok": True, "output_dir": str(args.output_dir.resolve()),
                          "results": rows}, indent=2, sort_keys=True))
    finally:
        if runtime_host:
            try:
                runtime_request(runtime_host, args.command_port, {"operation": "shutdown"}, 3.0)
            except Exception:
                pass
        if mininet is not None:
            try:
                mininet.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                mininet.kill()
                mininet.wait()
        mininet_log.close()
        subprocess.run(wsl + ["sudo", "-n", "mn", "-c"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(wsl + ["rm", "-f", wsl_path(native)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
