#!/usr/bin/env python3
"""Atomic Docker lifecycle and resource limits for containerized VNF chains."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "sdn" / "vnf_catalog.json"


@dataclass(frozen=True)
class ResourceMapping:
    cpu_capacity_units: float = 55.0
    memory_capacity_units: float = 45.0
    dc_cpu_cores: float = 2.0
    dc_memory_mb: int = 1024
    minimum_cpu_cores: float = 0.05
    minimum_memory_mb: int = 32

    def limits(self, cpu_units: float, memory_units: float) -> tuple[float, int]:
        cpu_units = float(cpu_units)
        memory_units = float(memory_units)
        if cpu_units <= 0.0 or memory_units <= 0.0:
            raise ValueError("VNF CPU and memory demands must be positive")
        if cpu_units > self.cpu_capacity_units or memory_units > self.memory_capacity_units:
            raise ValueError(
                f"VNF demand cpu={cpu_units}, memory={memory_units} exceeds "
                f"DC capacity {self.cpu_capacity_units}/{self.memory_capacity_units}"
            )
        cpu_cores = max(
            self.minimum_cpu_cores,
            cpu_units / self.cpu_capacity_units * self.dc_cpu_cores,
        )
        memory_mb = max(
            self.minimum_memory_mb,
            math.ceil(memory_units / self.memory_capacity_units * self.dc_memory_mb),
        )
        return round(cpu_cores, 4), int(memory_mb)


@dataclass(frozen=True)
class VNFContainerSpec:
    request_id: int
    stage: int
    vnf_type: int
    vnf_name: str
    dc_node: int
    cpu_units: float
    memory_units: float
    cpu_cores: float
    memory_mb: int
    listen_port: int
    next_host: str
    next_port: int
    health_port: int
    processing_delay_ms: float
    cpu_work: int
    image: str
    container_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VNFChainPlan:
    request_id: int
    entry_host: str
    entry_port: int
    sink_host: str
    sink_port: int
    containers: tuple[VNFContainerSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "entry_host": self.entry_host,
            "entry_port": self.entry_port,
            "sink_host": self.sink_host,
            "sink_port": self.sink_port,
            "containers": [value.to_dict() for value in self.containers],
        }


class ContainerBackend(Protocol):
    def available(self) -> bool: ...
    def start(self, spec: VNFContainerSpec) -> str: ...
    def wait_healthy(self, container_name: str, timeout: float) -> dict[str, Any]: ...
    def remove(self, container_name: str) -> None: ...
    def inspect(self, container_name: str) -> dict[str, Any]: ...


class DockerCLIBackend:
    def __init__(self, command: tuple[str, ...] = ("docker",)) -> None:
        self.command = tuple(command)

    def _run(self, values: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [*self.command, *values],
            text=True,
            capture_output=True,
            check=False,
        )
        if check and result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Docker command failed: {detail}")
        return result

    def available(self) -> bool:
        try:
            return self._run(["version", "--format", "{{.Server.Version}}"], check=False).returncode == 0
        except OSError:
            return False

    @staticmethod
    def run_command(spec: VNFContainerSpec) -> list[str]:
        return [
            "run",
            "-d",
            "--name",
            spec.container_name,
            "--network",
            "host",
            "--cpus",
            f"{spec.cpu_cores:g}",
            "--memory",
            f"{spec.memory_mb}m",
            "--pids-limit",
            "64",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--label",
            f"hrl.request_id={spec.request_id}",
            "--label",
            f"hrl.stage={spec.stage}",
            "--label",
            f"hrl.dc_node={spec.dc_node}",
            "--env",
            f"VNF_HEALTH_PORT={spec.health_port}",
            spec.image,
            "--listen-port",
            str(spec.listen_port),
            "--next-host",
            spec.next_host,
            "--next-port",
            str(spec.next_port),
            "--health-port",
            str(spec.health_port),
            "--vnf-type",
            spec.vnf_name,
            "--processing-delay-ms",
            f"{spec.processing_delay_ms:g}",
            "--cpu-work",
            str(spec.cpu_work),
        ]

    def start(self, spec: VNFContainerSpec) -> str:
        result = self._run(self.run_command(spec))
        return result.stdout.strip()

    def inspect(self, container_name: str) -> dict[str, Any]:
        result = self._run(["inspect", container_name])
        payload = json.loads(result.stdout)
        if not payload:
            raise RuntimeError(f"Docker returned no inspection data for {container_name}")
        return payload[0]

    def wait_healthy(self, container_name: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        last = None
        while time.monotonic() < deadline:
            last = self.inspect(container_name)
            state = last.get("State", {})
            health = state.get("Health", {}).get("Status")
            if state.get("Running") and health in {None, "healthy"}:
                return last
            if health == "unhealthy" or state.get("Status") in {"dead", "exited"}:
                break
            time.sleep(0.1)
        raise RuntimeError(f"container {container_name} did not become healthy: {last}")

    def remove(self, container_name: str) -> None:
        result = self._run(["rm", "-f", container_name], check=False)
        if result.returncode and "No such container" not in result.stderr:
            raise RuntimeError(result.stderr.strip())


class VNFContainerManager:
    def __init__(
        self,
        backend: ContainerBackend,
        *,
        resource_mapping: ResourceMapping | None = None,
        catalog_path: str | Path = DEFAULT_CATALOG,
        image: str = "hrl-vnf-forwarder:latest",
        health_timeout: float = 10.0,
    ) -> None:
        self.backend = backend
        self.mapping = resource_mapping or ResourceMapping()
        self.image = str(image)
        self.health_timeout = float(health_timeout)
        catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.vnf_types = catalog["types"]
        self.active: dict[int, VNFChainPlan] = {}
        self.reservations: dict[int, dict[str, float]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _demand(request: dict[str, Any], key: str, legacy_key: str) -> list[float]:
        values = request.get(key, request.get(legacy_key, []))
        return [float(value) for value in values]

    @staticmethod
    def _placement(placement_by_vnf: dict[Any, Any], stage: int) -> int:
        for key in (stage, str(stage)):
            if key in placement_by_vnf:
                value = placement_by_vnf[key]
                if isinstance(value, dict):
                    value = value.get("node", value.get("dc_node"))
                return int(value)
        raise ValueError(f"placement has no DC node for VNF stage {stage}")

    def plan_chain(
        self,
        request: dict[str, Any],
        placement_by_vnf: dict[Any, Any],
        *,
        entry_port: int,
        sink_port: int,
        entry_host: str = "127.0.0.1",
        sink_host: str = "127.0.0.1",
        health_port_base: int = 30000,
    ) -> VNFChainPlan:
        request_id = int(request["id"])
        chain = [int(value) for value in request.get("vnf", [])]
        cpu = self._demand(request, "cpu_origin", "vnf_cpu")
        memory = self._demand(request, "memory_origin", "vnf_mem")
        if not chain or len(cpu) != len(chain) or len(memory) != len(chain):
            raise ValueError("request VNF, CPU, and memory vectors must have equal non-zero length")
        if entry_port <= 0 or sink_port <= 0 or entry_port + len(chain) > 65535:
            raise ValueError("invalid VNF chain UDP port range")

        specs = []
        for stage, vnf_type in enumerate(chain):
            definition = self.vnf_types.get(str(vnf_type))
            if definition is None:
                raise ValueError(f"unknown VNF type {vnf_type}")
            dc_node = self._placement(placement_by_vnf, stage)
            cpu_cores, memory_mb = self.mapping.limits(cpu[stage], memory[stage])
            vnf_name = str(definition["name"])
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "-", vnf_name)
            listen_port = int(entry_port) + stage
            next_port = int(sink_port) if stage == len(chain) - 1 else listen_port + 1
            processing_delay_ms = float(definition.get("processing_delay_ms", 0.05))
            cpu_work = int(definition.get("cpu_work", max(1, round(float(definition["cpu_per_mbps"]) * 50))))
            specs.append(
                VNFContainerSpec(
                    request_id=request_id,
                    stage=stage,
                    vnf_type=vnf_type,
                    vnf_name=vnf_name,
                    dc_node=dc_node,
                    cpu_units=cpu[stage],
                    memory_units=memory[stage],
                    cpu_cores=cpu_cores,
                    memory_mb=memory_mb,
                    listen_port=listen_port,
                    next_host=sink_host,
                    next_port=next_port,
                    health_port=int(health_port_base) + request_id * 10 + stage,
                    processing_delay_ms=processing_delay_ms,
                    cpu_work=cpu_work,
                    image=self.image,
                    container_name=f"hrl-r{request_id}-s{stage}-{safe_name}",
                )
            )
        if any(spec.health_port > 65535 for spec in specs):
            raise ValueError("VNF health port range exceeds 65535")
        return VNFChainPlan(
            request_id=request_id,
            entry_host=entry_host,
            entry_port=int(entry_port),
            sink_host=sink_host,
            sink_port=int(sink_port),
            containers=tuple(specs),
        )

    def _reserve(self, plan: VNFChainPlan) -> None:
        requested: dict[int, dict[str, float]] = {}
        for spec in plan.containers:
            row = requested.setdefault(spec.dc_node, {"cpu": 0.0, "memory": 0.0})
            row["cpu"] += spec.cpu_units
            row["memory"] += spec.memory_units
        for node, demand in requested.items():
            current = self.reservations.get(node, {"cpu": 0.0, "memory": 0.0})
            if current["cpu"] + demand["cpu"] > self.mapping.cpu_capacity_units + 1e-9:
                raise RuntimeError(f"DC {node} has insufficient CPU")
            if current["memory"] + demand["memory"] > self.mapping.memory_capacity_units + 1e-9:
                raise RuntimeError(f"DC {node} has insufficient memory")
        for node, demand in requested.items():
            row = self.reservations.setdefault(node, {"cpu": 0.0, "memory": 0.0})
            row["cpu"] += demand["cpu"]
            row["memory"] += demand["memory"]

    def _unreserve(self, plan: VNFChainPlan) -> None:
        for spec in plan.containers:
            row = self.reservations.get(spec.dc_node)
            if row is None:
                continue
            row["cpu"] = max(0.0, row["cpu"] - spec.cpu_units)
            row["memory"] = max(0.0, row["memory"] - spec.memory_units)
            if row["cpu"] == 0.0 and row["memory"] == 0.0:
                self.reservations.pop(spec.dc_node, None)

    def deploy(self, plan: VNFChainPlan) -> dict[str, Any]:
        with self._lock:
            if plan.request_id in self.active:
                raise RuntimeError(f"request {plan.request_id} already has an active VNF chain")
            if not self.backend.available():
                raise RuntimeError("Docker runtime is unavailable")
            self._reserve(plan)
            started = []
            try:
                # Downstream-first startup mirrors the OpenFlow atomic cutover.
                for spec in reversed(plan.containers):
                    self.backend.start(spec)
                    started.append(spec.container_name)
                    self.backend.wait_healthy(spec.container_name, self.health_timeout)
            except Exception:
                for name in reversed(started):
                    try:
                        self.backend.remove(name)
                    except Exception:
                        pass
                self._unreserve(plan)
                raise
            self.active[plan.request_id] = plan
            return {
                "accepted": True,
                "request_id": plan.request_id,
                "entry": [plan.entry_host, plan.entry_port],
                "containers": list(reversed(started)),
                "reservations": {
                    node: dict(values) for node, values in self.reservations.items()
                },
            }

    def release(self, request_id: int) -> dict[str, Any]:
        with self._lock:
            plan = self.active.pop(int(request_id), None)
            if plan is None:
                return {"accepted": False, "request_id": int(request_id), "reason": "not_active"}
            errors = []
            for spec in plan.containers:
                try:
                    self.backend.remove(spec.container_name)
                except Exception as exc:
                    errors.append(str(exc))
            self._unreserve(plan)
            return {
                "accepted": not errors,
                "request_id": int(request_id),
                "removed": len(plan.containers) - len(errors),
                "errors": errors,
            }

    def status(self) -> dict[str, Any]:
        return {
            "docker_available": self.backend.available(),
            "active_requests": sorted(self.active),
            "reservations": self.reservations,
        }
