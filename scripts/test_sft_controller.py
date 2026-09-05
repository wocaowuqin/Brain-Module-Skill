#!/usr/bin/env python3
"""Exercise the SFT controller REST API against a live Mininet switch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def request(url, method="GET", payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_switch(url, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            status = request(url)
            if status.get("datapaths"):
                return status
        except OSError as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"no switch connected before timeout: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--rest-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    wsl = ["wsl", "-d", args.distro, "--"]
    subprocess.run(
        wsl
        + [
            "sudo",
            "-n",
            "systemctl",
            "stop",
            "ryu-controller.service",
        ],
        check=True,
    )
    subprocess.run(
        wsl
        + [
            "sudo",
            "-n",
            "systemctl",
            "start",
            "ryu-sft-controller.service",
        ],
        check=True,
    )

    subprocess.run(wsl + ["sudo", "-n", "mn", "-c"], check=False)
    mn = subprocess.Popen(
        wsl
        + [
            "sudo",
            "-n",
            "mn",
            "--topo",
            "single,3",
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
        status = wait_for_switch(f"{args.rest_url}/sft/status", args.timeout)
        print(json.dumps({"connected": status["datapaths"]}, indent=2))
        response = request(
            f"{args.rest_url}/sft/group",
            method="POST",
            payload={
                "group_id": 1,
                "dst_ip": "239.1.1.1",
                "switch_outputs": {"1": [2, 3]},
            },
        )
        print(json.dumps({"group_install": response}, indent=2))
        reroute = request(
            f"{args.rest_url}/sft/reroute",
            method="POST",
            payload={
                "group_id": 1,
                "dst_ip": "239.1.1.1",
                "switch_outputs": {"1": [2]},
            },
        )
        print(json.dumps({"reroute": reroute}, indent=2))
        after = request(f"{args.rest_url}/sft/status")
        print(json.dumps({"groups": after.get("groups", {})}, indent=2))
        deleted = request(
            f"{args.rest_url}/sft/group/1",
            method="DELETE",
        )
        print(json.dumps({"group_delete": deleted}, indent=2))
        final = request(f"{args.rest_url}/sft/status")
        print(json.dumps({"groups_after_delete": final.get("groups", {})}, indent=2))
        if final.get("groups"):
            raise RuntimeError(f"SFT groups remain after delete: {final['groups']}")
    finally:
        if mn.stdin:
            mn.stdin.write("exit\n")
            mn.stdin.flush()
        try:
            mn.wait(timeout=20)
        except subprocess.TimeoutExpired:
            mn.kill()
            mn.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
