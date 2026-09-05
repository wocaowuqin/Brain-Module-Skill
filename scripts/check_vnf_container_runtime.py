#!/usr/bin/env python3
"""Check VNF container planning/rollback, with an optional real Docker chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdn.vnf_container_manager import (
    DockerCLIBackend,
    ResourceMapping,
    VNFContainerManager,
    VNFContainerSpec,
)


REQUEST = {
    "id": 101,
    "vnf": [0, 5, 7],
    "cpu_origin": [11, 14, 8],
    "memory_origin": [9, 8, 6],
}
PLACEMENT = {0: 1, 1: 3, 2: 7}


class FakeBackend:
    def __init__(self, fail_name: str | None = None) -> None:
        self.fail_name = fail_name
        self.started: list[str] = []
        self.removed: list[str] = []
        self.specs: dict[str, VNFContainerSpec] = {}

    def available(self) -> bool:
        return True

    def start(self, spec: VNFContainerSpec) -> str:
        self.started.append(spec.container_name)
        self.specs[spec.container_name] = spec
        return spec.container_name

    def wait_healthy(self, container_name: str, timeout: float) -> dict[str, Any]:
        del timeout
        if container_name == self.fail_name:
            raise RuntimeError("injected container health failure")
        return {"Name": container_name, "State": {"Running": True}}

    def remove(self, container_name: str) -> None:
        self.removed.append(container_name)

    def inspect(self, container_name: str) -> dict[str, Any]:
        return {"Name": container_name}


def build_manager(backend) -> VNFContainerManager:
    return VNFContainerManager(
        backend,
        resource_mapping=ResourceMapping(
            cpu_capacity_units=55.0,
            memory_capacity_units=45.0,
            dc_cpu_cores=2.0,
            dc_memory_mb=1024,
        ),
    )


def fake_checks() -> dict[str, Any]:
    backend = FakeBackend()
    manager = build_manager(backend)
    plan = manager.plan_chain(
        REQUEST,
        PLACEMENT,
        entry_port=24000,
        sink_port=24999,
    )
    deployment = manager.deploy(plan)
    expected_start = [spec.container_name for spec in reversed(plan.containers)]
    if backend.started != expected_start:
        raise AssertionError("containers were not started downstream-first")
    if set(manager.reservations) != {1, 3, 7}:
        raise AssertionError("per-DC resource reservations are incomplete")
    first = plan.containers[0]
    if first.cpu_cores != 0.4 or first.memory_mb != 205:
        raise AssertionError(f"unexpected 55/45 resource mapping: {first}")
    command = DockerCLIBackend.run_command(first)
    if "--cpus" not in command or "--memory" not in command or "--network" not in command:
        raise AssertionError("Docker command lacks resource or network limits")
    release = manager.release(REQUEST["id"])
    if not release["accepted"] or manager.reservations or manager.active:
        raise AssertionError("successful chain release leaked state")

    failing_backend = FakeBackend(fail_name=plan.containers[0].container_name)
    failing_manager = build_manager(failing_backend)
    try:
        failing_manager.deploy(plan)
    except RuntimeError as exc:
        if "injected" not in str(exc):
            raise
    else:
        raise AssertionError("injected container failure was not propagated")
    if failing_manager.reservations or failing_manager.active:
        raise AssertionError("failed deployment leaked reservations")
    if failing_backend.removed != list(reversed(failing_backend.started)):
        raise AssertionError("failed deployment did not roll back in reverse order")
    return {
        "ok": True,
        "plan": plan.to_dict(),
        "deployment": deployment,
        "release": release,
        "docker_command": ["docker", *command],
        "rollback": {
            "started": failing_backend.started,
            "removed": failing_backend.removed,
        },
    }


def wait_for_health(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"local VNF health endpoint {port} did not become ready")


def local_forwarder_smoke() -> dict[str, Any]:
    payload = b"hrl-local-vnf-forwarder-smoke"
    processes = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sink:
        sink.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sink.bind(("127.0.0.1", 25999))
        sink.settimeout(5.0)
        try:
            for stage in reversed(range(3)):
                listen_port = 25000 + stage
                next_port = 25999 if stage == 2 else listen_port + 1
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "sdn" / "vnf_forwarder.py"),
                        "--listen-port",
                        str(listen_port),
                        "--next-host",
                        "127.0.0.1",
                        "--next-port",
                        str(next_port),
                        "--health-port",
                        str(26000 + stage),
                        "--vnf-type",
                        ("firewall", "dpi_ids", "traffic_monitor")[stage],
                        "--cpu-work",
                        "2",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                processes.append(process)
                wait_for_health(26000 + stage)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(payload, ("127.0.0.1", 25000))
            observed, _ = sink.recvfrom(65535)
            if observed != payload:
                raise AssertionError("local VNF chain altered the payload")
        finally:
            for process in reversed(processes):
                process.terminate()
            for process in processes:
                try:
                    process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=2.0)
    return {"ok": True, "stages": 3, "payload_bytes": len(payload)}


def docker_smoke(build_image: bool) -> dict[str, Any]:
    backend = DockerCLIBackend()
    if not backend.available():
        raise RuntimeError("Docker runtime is unavailable; enable Docker Desktop WSL integration")
    if build_image:
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                "hrl-vnf-forwarder:latest",
                "-f",
                str(ROOT / "sdn" / "vnf_container" / "Dockerfile"),
                str(ROOT),
            ],
            check=True,
        )
    manager = build_manager(backend)
    plan = manager.plan_chain(REQUEST, PLACEMENT, entry_port=24000, sink_port=24999)
    payload = b"hrl-vnf-container-smoke"
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sink:
        sink.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sink.bind((plan.sink_host, plan.sink_port))
        sink.settimeout(5.0)
        deployment = manager.deploy(plan)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(payload, (plan.entry_host, plan.entry_port))
            observed, _ = sink.recvfrom(65535)
            if observed != payload:
                raise AssertionError("container chain altered the smoke payload")
            inspections = {
                spec.container_name: backend.inspect(spec.container_name)
                for spec in plan.containers
            }
        finally:
            release = manager.release(plan.request_id)
    return {
        "ok": True,
        "payload_bytes": len(payload),
        "deployment": deployment,
        "release": release,
        "containers": {
            name: {
                "running": value.get("State", {}).get("Running"),
                "health": value.get("State", {}).get("Health", {}).get("Status"),
                "nano_cpus": value.get("HostConfig", {}).get("NanoCpus"),
                "memory_bytes": value.get("HostConfig", {}).get("Memory"),
            }
            for name, value in inspections.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true", help="run a real three-container UDP chain")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = {
        "fake_backend": fake_checks(),
        "local_forwarder": local_forwarder_smoke(),
        "docker": docker_smoke(args.build_image) if args.docker else {"executed": False},
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
