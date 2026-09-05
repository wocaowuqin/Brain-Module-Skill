#!/usr/bin/env python3
"""Plan or execute one HRL placement as a resource-limited Docker VNF chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdn.vnf_container_manager import DockerCLIBackend, ResourceMapping, VNFContainerManager


def read_request(path: Path, request_id: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if int(value["id"]) == request_id:
                return value
    raise ValueError(f"request {request_id} is not present in {path}")


def read_placement(path: Path, request_id: int) -> dict[Any, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        matches = [row for row in value if int(row.get("request_id", -1)) == request_id]
        if len(matches) != 1:
            raise ValueError(f"placement file has {len(matches)} rows for request {request_id}")
        value = matches[0]
    if int(value.get("request_id", request_id)) != request_id:
        raise ValueError("placement request_id does not match --request-id")
    placement = value.get("placement_by_vnf", value.get("placement"))
    if not isinstance(placement, dict):
        raise ValueError("placement JSON must contain placement_by_vnf")
    return placement


def build_image(image: str) -> None:
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            image,
            "-f",
            str(ROOT / "sdn" / "vnf_container" / "Dockerfile"),
            str(ROOT),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--request-id", type=int, required=True)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--entry-port", type=int, default=24000)
    parser.add_argument("--sink-port", type=int, default=24999)
    parser.add_argument("--health-port-base", type=int, default=30000)
    parser.add_argument("--cpu-capacity-units", type=float, default=55.0)
    parser.add_argument("--memory-capacity-units", type=float, default=45.0)
    parser.add_argument("--dc-cpu-cores", type=float, default=2.0)
    parser.add_argument("--dc-memory-mb", type=int, default=1024)
    parser.add_argument("--image", default="hrl-vnf-forwarder:latest")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_path = Path(args.requests).resolve()
    placement_path = Path(args.placement).resolve()
    request = read_request(request_path, args.request_id)
    placement = read_placement(placement_path, args.request_id)
    backend = DockerCLIBackend()
    manager = VNFContainerManager(
        backend,
        resource_mapping=ResourceMapping(
            cpu_capacity_units=args.cpu_capacity_units,
            memory_capacity_units=args.memory_capacity_units,
            dc_cpu_cores=args.dc_cpu_cores,
            dc_memory_mb=args.dc_memory_mb,
        ),
        image=args.image,
    )
    plan = manager.plan_chain(
        request,
        placement,
        entry_port=args.entry_port,
        sink_port=args.sink_port,
        health_port_base=args.health_port_base,
    )
    report: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "request_file": str(request_path),
        "placement_file": str(placement_path),
        "resource_mapping": {
            "cpu_capacity_units": args.cpu_capacity_units,
            "memory_capacity_units": args.memory_capacity_units,
            "dc_cpu_cores": args.dc_cpu_cores,
            "dc_memory_mb": args.dc_memory_mb,
        },
        "plan": plan.to_dict(),
        "docker_commands": [
            ["docker", *DockerCLIBackend.run_command(spec)] for spec in reversed(plan.containers)
        ],
    }
    if args.execute:
        if args.build_image:
            build_image(args.image)
        payload = f"hrl-vnf-request-{args.request_id}".encode("ascii")
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
                    raise AssertionError("container VNF chain altered the probe payload")
                report["probe"] = {"ok": True, "bytes": len(observed)}
                report["deployment"] = deployment
            finally:
                report["release"] = manager.release(plan.request_id)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

