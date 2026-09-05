#!/usr/bin/env python3
"""Test a multi-hop multicast tree and a live branch reroute."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def wait_for_topology(url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status = request(url)
            if len(status.get("datapaths", [])) >= 4 and len(status.get("links", [])) >= 4:
                return status
        except OSError:
            pass
        time.sleep(0.2)
    raise RuntimeError("diamond topology did not connect to the SFT controller")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--rest-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    wsl = ["wsl", "-d", args.distro, "--"]
    repo = wsl_path(ROOT, args.distro)
    subprocess.run(
        wsl + ["sudo", "-n", "systemctl", "stop", "ryu-controller.service"],
        check=True,
    )
    subprocess.run(
        wsl + ["sudo", "-n", "systemctl", "start", "ryu-sft-controller.service"],
        check=True,
    )
    subprocess.run(wsl + ["sudo", "-n", "mn", "-c"], check=False)

    mn = subprocess.Popen(
        wsl
        + [
            "sudo",
            "-n",
            "mn",
            "--custom",
            f"{repo}/sdn/diamond_topology.py",
            "--topo",
            "diamond",
            "--mac",
            "--switch",
            "ovsk,protocols=OpenFlow13",
            "--controller",
            "remote,ip=127.0.0.1,port=6653",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        status = wait_for_topology(f"{args.rest_url}/sft/status", args.timeout)
        print(json.dumps({"topology": status}, indent=2))

        initial = request(
            f"{args.rest_url}/sft/group",
            method="POST",
            payload={
                "group_id": 7,
                "dst_ip": "239.1.1.1",
                "switch_outputs": {"1": [1], "2": [2], "4": [3, 4]},
            },
        )
        print(json.dumps({"initial_tree": initial}, indent=2))

        # iperf2 supports multicast; it is installed alongside iperf3 solely
        # for this multicast data-plane check.
        for command in (
            "h1 route add -net 239.0.0.0 netmask 255.0.0.0 dev h1-eth0",
            "h2 route add -net 239.0.0.0 netmask 255.0.0.0 dev h2-eth0",
            "h3 route add -net 239.0.0.0 netmask 255.0.0.0 dev h3-eth0",
            "h2 iperf -s -u -B 239.1.1.1 -p 5001 -t 10 > /tmp/sft-h2.log 2>&1 &",
            "h3 iperf -s -u -B 239.1.1.1 -p 5001 -t 10 > /tmp/sft-h3.log 2>&1 &",
            "h1 iperf -c 239.1.1.1 -u -b 1M -t 8 > /tmp/sft-iperf.log 2>&1 &",
        ):
            mn.stdin.write(command + "\n")
        mn.stdin.flush()
        time.sleep(2.0)

        reroute = request(
            f"{args.rest_url}/sft/reroute",
            method="POST",
            payload={
                "group_id": 7,
                "dst_ip": "239.1.1.1",
                "switch_outputs": {"1": [2], "3": [2], "4": [3, 4]},
            },
        )
        print(json.dumps({"reroute": reroute}, indent=2))
        time.sleep(7.0)
        receiver_logs = {}
        for name in ("sft-h2.log", "sft-h3.log", "sft-iperf.log"):
            result = subprocess.run(
                wsl + ["cat", f"/tmp/{name}"],
                check=False,
                capture_output=True,
                text=True,
            )
            receiver_logs[name] = result.stdout[-4000:]
        print(json.dumps({"multicast_logs": receiver_logs}, indent=2))
        print(json.dumps({"final_status": request(f"{args.rest_url}/sft/status")}, indent=2))
    finally:
        if mn.stdin:
            mn.stdin.write("exit\n")
            mn.stdin.flush()
        try:
            mn.wait(timeout=25)
        except subprocess.TimeoutExpired:
            mn.kill()
            mn.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
