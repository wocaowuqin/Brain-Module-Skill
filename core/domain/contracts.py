"""Serializable domain objects for the staged architecture refactor.

These contracts are additive and intentionally do not import trainers, CLI
scripts, SDN clients or experiment directories.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    source: int | None = None
    destinations: tuple[int, ...] = ()
    vnf_sequence: tuple[int, ...] = ()
    bandwidth: float = 0.0
    arrival_time: float = 0.0
    duration: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ResourceSnapshot:
    version: str | int
    node_resources: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    link_resources: Mapping[tuple[int, int], Mapping[str, float]] = field(default_factory=dict)

@dataclass(frozen=True)
class ResourceFootprint:
    cpu: float = 0.0
    memory: float = 0.0
    bandwidth: float = 0.0
    nodes: tuple[int, ...] = ()
    edges: tuple[tuple[int, int], ...] = ()

@dataclass(frozen=True)
class CandidatePlan:
    request_id: int
    accepted: bool
    placement: Mapping[str, Any] = field(default_factory=dict)
    paths: tuple[tuple[int, ...], ...] = ()
    footprint: ResourceFootprint = field(default_factory=ResourceFootprint)
    reason: str = ""

@dataclass(frozen=True)
class JointAction:
    action_type: str
    target: Mapping[str, Any] = field(default_factory=dict)
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CommitResult:
    success: bool
    status: str
    ledger_version: str | int | None = None
    message: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)
