#!/usr/bin/env python3
"""Start and verify the Ubuntu WSL2 SDN testbed from Windows or WSL."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence


DEFAULT_DISTRO = "Ubuntu-22.04"
DEFAULT_REST_URL = "http://127.0.0.1:8080/stats/switches"


def run(command: Sequence[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )


def in_wsl() -> bool:
    try:
        return "microsoft" in platform.release().lower()
    except OSError:
        return False


def wsl_command(distro: str, *command: str) -> list[str]:
    return ["wsl", "-d", distro, "--", *command]


def start_services(distro: str) -> None:
    stop_custom = ["sudo", "-n", "systemctl", "stop", "ryu-sft-controller.service"]
    command = [
        "sudo",
        "-n",
        "systemctl",
        "start",
        "openvswitch-switch.service",
        "ryu-controller.service",
    ]
    if os.name == "nt":
        run(wsl_command(distro, *stop_custom))
        run(wsl_command(distro, *command))
    elif in_wsl():
        run(stop_custom)
        run(command)
    else:
        raise RuntimeError("Run this checker from Windows or WSL2.")


def fetch_switches(rest_url: str) -> list[int]:
    with urllib.request.urlopen(rest_url, timeout=2.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Ryu response: {payload!r}")
    return [int(item) for item in payload]


def wait_for_ryu(rest_url: str, timeout: float) -> list[int]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return fetch_switches(rest_url)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Ryu REST did not become ready within {timeout:.1f}s: {last_error}")


def run_smoke_test(distro: str) -> None:
    script = Path(__file__).resolve().parents[1] / "sdn" / "smoke_test.sh"
    if os.name == "nt":
        windows_path = script.as_posix()
        if len(windows_path) < 3 or windows_path[1:3] != ":/":
            raise RuntimeError(f"Cannot map Windows path into WSL: {script}")
        linux_path = f"/mnt/{windows_path[0].lower()}/{windows_path[3:]}"
        run(wsl_command(distro, "bash", linux_path), capture=False)
    else:
        run(["bash", str(script)], capture=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", default=DEFAULT_DISTRO)
    parser.add_argument("--rest-url", default=DEFAULT_REST_URL)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--smoke", action="store_true", help="Run Mininet pingall and iperf3.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        start_services(args.distro)
        switches_before = wait_for_ryu(args.rest_url, args.timeout)
        if args.smoke:
            run_smoke_test(args.distro)
        switches_after = wait_for_ryu(args.rest_url, args.timeout)
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2))
        return 1

    result = {
        "ready": True,
        "distro": args.distro,
        "rest_url": args.rest_url,
        "switches_before": switches_before,
        "switches_after": switches_after,
        "smoke_test": bool(args.smoke),
        "note": "An empty switch list is normal when Mininet is not running.",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
