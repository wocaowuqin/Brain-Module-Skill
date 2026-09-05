#!/usr/bin/env python3
"""Run a live Cogentco multicast tree cutover on Mininet/OVS/Ryu."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "sdn" / "topologies" / "cogentco_197.json"
SOURCE_DPID = 45
RECEIVER_DPIDS = (184, 20)
MULTICAST_IP = "239.19.7.1"
GROUP_ID = 91


def wsl_path(path: Path, distro: str) -> str:
    result = subprocess.run(
        ["wsl", "-d", distro, "--", "wslpath", "-a", str(path.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    converted = result.stdout.strip()
    if not converted:
        raise RuntimeError(f"wslpath returned no path for {path}")
    return converted


def request(url, method="GET", payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SFT REST {exc.code}: {body}") from exc


def wait_for_api(url, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return request(url)
        except OSError as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"SFT controller REST API was not ready: {last_error}")


def wait_for_switches(url, expected, timeout):
    deadline = time.monotonic() + timeout
    best = 0
    while time.monotonic() < deadline:
        try:
            status = request(url)
            current = len(status.get("datapaths", []))
            best = max(best, current)
            if current == expected:
                return status
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Cogentco switches did not connect: best={best}/{expected}")


def build_graph(profile):
    graph = nx.Graph()
    port_lookup = {}
    for edge in profile["edges"]:
        u = int(edge["u"])
        v = int(edge["v"])
        graph.add_edge(u, v)
        port_lookup.setdefault((u, v), int(edge["u_port"]))
        port_lookup.setdefault((v, u), int(edge["v_port"]))
    return graph, port_lookup


def edge_set(paths, targets):
    result = set()
    for target in targets:
        path = paths[int(target)]
        result.update(tuple(sorted(edge)) for edge in zip(path, path[1:]))
    return result


def select_trees(profile):
    graph, port_lookup = build_graph(profile)
    initial_paths = nx.single_source_shortest_path(graph, SOURCE_DPID)
    initial_edges = edge_set(initial_paths, RECEIVER_DPIDS)

    weighted = graph.copy()
    nx.set_edge_attributes(weighted, 1, "weight")
    for u, v in initial_edges:
        weighted[u][v]["weight"] = 25
    _, alternate_paths = nx.single_source_dijkstra(
        weighted,
        SOURCE_DPID,
        weight="weight",
    )
    alternate_edges = edge_set(alternate_paths, RECEIVER_DPIDS)
    if initial_edges == alternate_edges:
        raise RuntimeError("failed to construct a distinct alternate tree")

    host_ports = {
        int(node["dpid"]): int(node["host_port"])
        for node in profile["nodes"]
    }

    def outputs_from_paths(paths):
        children = {}
        parent = {}
        selected_paths = {}
        for receiver in RECEIVER_DPIDS:
            path = [int(node) for node in paths[int(receiver)]]
            selected_paths[str(receiver)] = path
            for source, destination in zip(path, path[1:]):
                previous = parent.setdefault(destination, source)
                if previous != source:
                    raise RuntimeError(
                        f"tree has multiple parents for {destination}: {previous}, {source}"
                    )
                children.setdefault(source, set()).add(destination)
        outputs = {}
        for dpid, child_nodes in children.items():
            outputs[str(dpid)] = sorted(
                port_lookup[(dpid, child)] for child in child_nodes
            )
        for receiver in RECEIVER_DPIDS:
            outputs.setdefault(str(receiver), []).append(host_ports[receiver])
            outputs[str(receiver)] = sorted(set(outputs[str(receiver)]))
        return outputs, selected_paths

    initial_outputs, initial_selected_paths = outputs_from_paths(initial_paths)
    alternate_outputs, alternate_selected_paths = outputs_from_paths(alternate_paths)
    return {
        "initial_outputs": initial_outputs,
        "alternate_outputs": alternate_outputs,
        "initial_paths": initial_selected_paths,
        "alternate_paths": alternate_selected_paths,
        "initial_edges": sorted(initial_edges),
        "alternate_edges": sorted(alternate_edges),
    }


def parse_receiver_log(text):
    matches = re.findall(r"(\d+)/(\d+) \(([\d.]+)%\)", text)
    if not matches:
        raise RuntimeError(f"iperf receiver log has no UDP summary: {text[-1000:]}")
    lost = int(matches[-1][0])
    total = int(matches[-1][1])
    loss_percent = float(matches[-1][2])
    if total <= 0:
        raise RuntimeError("iperf receiver got no UDP datagrams")
    return {"lost": lost, "received": total - lost, "total": total, "loss_percent": loss_percent}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--rest-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--switch-start-delay", type=float, default=0.2)
    parser.add_argument("--traffic-seconds", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "runs" / "mininet" / "cogentco_data_plane" / "result.json",
    )
    args = parser.parse_args()

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    trees = select_trees(profile)
    if set(trees["initial_edges"]) & set(trees["alternate_edges"]):
        raise RuntimeError("selected Cogentco trees are expected to be edge-disjoint")

    wsl = ["wsl", "-d", args.distro, "--"]
    repo = wsl_path(ROOT, args.distro)
    status_url = f"{args.rest_url}/sft/status"
    mininet = None

    try:
        for service in (
            "ryu-controller.service",
            "ryu-sft-controller.service",
        ):
            subprocess.run(
                wsl + ["sudo", "-n", "systemctl", "stop", service],
                check=True,
            )
        subprocess.run(
            wsl + ["sudo", "-n", "mn", "-c"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            wsl
            + [
                "sudo",
                "-n",
                "systemctl",
                "restart",
                "ryu-sft-static-controller.service",
            ],
            check=True,
        )
        wait_for_api(status_url, args.timeout)
        request(
            f"{args.rest_url}/sft/config",
            method="POST",
            payload={
                "topology_file": f"{repo}/sdn/topologies/cogentco_197.json",
                "default_capacity_bps": float(profile["default_bandwidth_mbps"]) * 1e6,
                "threshold": 0.95,
                "stats_interval": 5.0,
                "barrier_timeout": 10.0,
            },
        )

        mininet = subprocess.Popen(
            wsl
            + [
                "sudo",
                "-n",
                "python3",
                f"{repo}/sdn/run_profile_mininet.py",
                "--profile",
                f"{repo}/sdn/topologies/cogentco_197.json",
                "--controller-ip",
                "127.0.0.1",
                "--controller-port",
                "6653",
                "--switch-start-delay",
                str(args.switch_start_delay),
                "--link-mode",
                "tc",
                "--host-dpids",
                ",".join(str(dpid) for dpid in (SOURCE_DPID, *RECEIVER_DPIDS)),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status = wait_for_switches(status_url, int(profile["node_count"]), args.timeout)
        time.sleep(5.0)

        initial = request(
            f"{args.rest_url}/sft/group",
            method="POST",
            payload={
                "group_id": GROUP_ID,
                "dst_ip": MULTICAST_IP,
                "source_dpid": SOURCE_DPID,
                "switch_outputs": trees["initial_outputs"],
            },
        )
        time.sleep(1.0)

        commands = [
            f"h{SOURCE_DPID} ip route add 239.0.0.0/8 dev h{SOURCE_DPID}-eth0",
        ]
        for receiver in RECEIVER_DPIDS:
            commands.extend(
                [
                    f"h{receiver} ip route add 239.0.0.0/8 dev h{receiver}-eth0",
                    (
                        f"h{receiver} iperf -s -u -B {MULTICAST_IP} -p 5001 "
                        f"-t {args.traffic_seconds + 4} > /tmp/cogent-h{receiver}.log 2>&1 &"
                    ),
                ]
            )
        for command in commands:
            mininet.stdin.write(command + "\n")
        mininet.stdin.flush()
        time.sleep(1.0)

        mininet.stdin.write(
            f"h{SOURCE_DPID} iperf -c {MULTICAST_IP} -u -b 1M "
            f"-t {args.traffic_seconds} > /tmp/cogent-client.log 2>&1 &\n"
        )
        mininet.stdin.flush()
        time.sleep(3.0)

        reroute = request(
            f"{args.rest_url}/sft/reroute",
            method="POST",
            payload={
                "group_id": GROUP_ID,
                "dst_ip": MULTICAST_IP,
                "source_dpid": SOURCE_DPID,
                "drain_seconds": 0.5,
                "switch_outputs": trees["alternate_outputs"],
            },
        )
        time.sleep(max(1.0, float(args.traffic_seconds) - 2.0))

        receiver_results = {}
        receiver_logs = {}
        for receiver in RECEIVER_DPIDS:
            result = subprocess.run(
                wsl + ["cat", f"/tmp/cogent-h{receiver}.log"],
                check=False,
                capture_output=True,
                text=True,
            )
            receiver_logs[str(receiver)] = result.stdout[-2500:]
            receiver_results[str(receiver)] = parse_receiver_log(result.stdout)
        client_log = subprocess.run(
            wsl + ["cat", "/tmp/cogent-client.log"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout[-2500:]
        if any(result["lost"] != 0 for result in receiver_results.values()):
            raise RuntimeError(f"Cogentco reroute lost UDP datagrams: {receiver_results}")

        final_status = request(status_url)
        final_group = final_status.get("groups", {}).get(str(GROUP_ID), {})
        if final_group.get("switch_outputs") != trees["alternate_outputs"]:
            raise RuntimeError("controller final tree does not match the alternate tree")

        result_payload = {
                    "valid": True,
                    "topology": profile["name"],
                    "connected_switches": len(status["datapaths"]),
                    "source_dpid": SOURCE_DPID,
                    "receiver_dpids": list(RECEIVER_DPIDS),
                    "initial_paths": trees["initial_paths"],
                    "alternate_paths": trees["alternate_paths"],
                    "initial_tree_edges": len(trees["initial_edges"]),
                    "alternate_tree_edges": len(trees["alternate_edges"]),
                    "shared_tree_edges": 0,
                    "initial_group": initial,
                    "reroute": reroute,
                    "receivers": receiver_results,
                    "client_log": client_log,
                    "receiver_logs": receiver_logs,
                }
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        result_payload["output"] = str(output_path)
        print(json.dumps(result_payload, indent=2))
    finally:
        if mininet is not None and mininet.poll() is None and mininet.stdin:
            try:
                mininet.stdin.write("exit\n")
                mininet.stdin.flush()
                mininet.wait(timeout=60)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                mininet.kill()
                mininet.wait()
        subprocess.run(
            wsl + ["sudo", "-n", "mn", "-c"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            wsl
            + [
                "sudo",
                "-n",
                "systemctl",
                "stop",
                "ryu-sft-static-controller.service",
            ],
            check=False,
        )
        subprocess.run(
            wsl
            + ["sudo", "-n", "systemctl", "start", "ryu-controller.service"],
            check=False,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
