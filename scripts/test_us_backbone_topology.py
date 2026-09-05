#!/usr/bin/env python3
"""Validate an exported fixed-port topology against Ryu and OVS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "sdn" / "topologies" / "us_backbone_28.json"


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
    with urllib.request.urlopen(req, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_topology(url, nodes, directed_links, timeout):
    deadline = time.monotonic() + timeout
    best_nodes = 0
    best_links = 0
    while time.monotonic() < deadline:
        try:
            status = request(url)
            current_nodes = len(status.get("datapaths", []))
            current_links = len(status.get("links", []))
            best_nodes = max(best_nodes, current_nodes)
            best_links = max(best_links, current_links)
            if current_nodes == nodes and current_links == directed_links:
                return status
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(
        "SFT controller did not reach the complete configured topology: "
        f"best_switches={best_nodes}/{nodes}, best_directed_links={best_links}/{directed_links}"
    )


def configure_topology(url, payload, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            return request(url, method="POST", payload=payload)
        except OSError as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"SFT controller REST API was not ready: {last_error}")


def read_ovs_ports(wsl):
    result = subprocess.run(
        wsl
        + [
            "sudo",
            "-n",
            "ovs-vsctl",
            "--format=json",
            "--columns=name,ofport",
            "list",
            "Interface",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    table = json.loads(result.stdout)
    headings = table["headings"]
    name_index = headings.index("name")
    port_index = headings.index("ofport")
    ports = {}
    for row in table["data"]:
        name = row[name_index]
        ofport = row[port_index]
        if (
            isinstance(ofport, list)
            and len(ofport) == 2
            and ofport[0] == "set"
            and len(ofport[1]) == 1
        ):
            ofport = ofport[1][0]
        if isinstance(name, str) and isinstance(ofport, int):
            ports[name] = ofport
    return ports


def validate_ovs_ports(profile, ovs_ports, include_hosts=True):
    expected = {}
    if include_hosts:
        for node in profile["nodes"]:
            dpid = int(node["dpid"])
            host_port = int(node["host_port"])
            expected[f"s{dpid}-eth{host_port}"] = host_port
    for edge in profile["edges"]:
        for node_key, port_key in (("u", "u_port"), ("v", "v_port")):
            dpid = int(edge[node_key])
            port = int(edge[port_key])
            expected[f"s{dpid}-eth{port}"] = port
    mismatches = {
        interface: {"expected": port, "actual": ovs_ports.get(interface)}
        for interface, port in expected.items()
        if ovs_ports.get(interface) != port
    }
    if mismatches:
        raise RuntimeError(f"OVS port mapping mismatch: {mismatches}")
    return len(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--rest-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--topo", default="usbackbone")
    parser.add_argument("--switch-start-delay", type=float, default=0.0)
    parser.add_argument("--link-mode", choices=("basic", "tc"), default="tc")
    parser.add_argument("--no-hosts", action="store_true")
    parser.add_argument("--controller-service", default="ryu-sft-controller.service")
    parser.add_argument("--stats-interval", type=float, default=1.0)
    args = parser.parse_args()
    profile_path = args.profile.resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    wsl = ["wsl", "-d", args.distro, "--"]
    repo = wsl_path(ROOT, args.distro)

    allowed_services = {
        "ryu-sft-controller.service",
        "ryu-sft-static-controller.service",
    }
    if args.controller_service not in allowed_services:
        raise ValueError(f"unsupported controller service: {args.controller_service}")
    for service in ("ryu-controller.service", *sorted(allowed_services)):
        if service != args.controller_service:
            subprocess.run(
                wsl + ["sudo", "-n", "systemctl", "stop", service],
                check=True,
            )
    subprocess.run(wsl + ["sudo", "-n", "mn", "-c"], check=False)
    subprocess.run(
        wsl + ["sudo", "-n", "systemctl", "restart", args.controller_service],
        check=True,
    )
    configure_topology(
        f"{args.rest_url}/sft/config",
        {
            "topology_file": wsl_path(profile_path, args.distro),
            "default_capacity_bps": float(profile["default_bandwidth_mbps"]) * 1e6,
            "threshold": 0.95,
            "stats_interval": args.stats_interval,
        },
        args.timeout,
    )
    if args.switch_start_delay > 0:
        mininet_command = [
            "sudo",
            "-n",
            "python3",
            f"{repo}/sdn/run_profile_mininet.py",
            "--profile",
            f"{repo}/{profile_relative}",
            "--controller-ip",
            "127.0.0.1",
            "--controller-port",
            "6653",
            "--switch-start-delay",
            str(args.switch_start_delay),
            "--link-mode",
            args.link_mode,
        ]
        if args.no_hosts:
            mininet_command.append("--no-hosts")
    else:
        mininet_command = [
            "sudo",
            "-n",
            "mn",
            "--custom",
            f"{repo}/sdn/real_topology.py",
            "--topo",
            args.topo,
            "--link",
            args.link_mode,
            "--switch",
            "ovsk,protocols=OpenFlow13",
            "--controller",
            "remote,ip=127.0.0.1,port=6653",
        ]
    mn = subprocess.Popen(
        wsl + mininet_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        expected_links = int(profile["physical_link_count"]) * 2
        status = wait_for_topology(
            f"{args.rest_url}/sft/status",
            int(profile["node_count"]),
            expected_links,
            args.timeout,
        )
        topology = status.get("topology", {})
        if topology.get("mode") != "static_profile":
            raise RuntimeError(f"controller did not activate static topology: {topology}")
        loaded_profile = topology.get("profile") or {}
        if loaded_profile.get("name") != profile["name"]:
            raise RuntimeError(f"controller loaded the wrong topology: {loaded_profile}")
        expected = {
            (int(edge["u"]), int(edge["u_port"]), int(edge["v"]), int(edge["v_port"]))
            for edge in profile["edges"]
        }
        expected |= {(v, vp, u, up) for u, up, v, vp in list(expected)}
        observed = {
            (
                int(link["src_dpid"]),
                int(link["src_port"]),
                int(link["dst_dpid"]),
                int(link["dst_port"]),
            )
            for link in status["links"]
        }
        if observed != expected:
            raise RuntimeError(
                f"port mapping mismatch: missing={sorted(expected-observed)} "
                f"extra={sorted(observed-expected)}"
            )
        ovs_interface_count = validate_ovs_ports(
            profile,
            read_ovs_ports(wsl),
            include_hosts=not args.no_hosts,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "topology": profile["name"],
                    "switches": len(status["datapaths"]),
                    "physical_links": profile["physical_link_count"],
                    "directed_links": len(status["links"]),
                    "dc_nodes": len(profile["dc_nodes_1based"]),
                    "source_node_count": len(profile["source_nodes_1based"]),
                    "source_node_sample": profile["source_nodes_1based"][:20],
                    "bandwidth_mbps": profile["default_bandwidth_mbps"],
                    "delay_ms": profile["default_delay_ms"],
                    "delay_range_ms": profile.get("delay_range_ms"),
                    "port_mapping": "exact",
                    "ovs_interfaces_validated": ovs_interface_count,
                    "lldp_links_observed": topology.get("discovered_directed_links", 0),
                    "forwarding_topology_source": "static_profile",
                    "switch_start_delay_s": args.switch_start_delay,
                    "mininet_link_mode": args.link_mode,
                    "host_interfaces_included": not args.no_hosts,
                    "controller_service": args.controller_service,
                    "stats_interval_s": args.stats_interval,
                },
                indent=2,
            )
        )
    finally:
        if mn.poll() is None and mn.stdin:
            mn.stdin.write("exit\n")
            mn.stdin.flush()
        try:
            mn.wait(timeout=30)
        except subprocess.TimeoutExpired:
            mn.kill()
            mn.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
