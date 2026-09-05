"""Shared accessors for legacy SDN trees and executable HRL SFC plans.

The project has two plan formats in active use.  Keeping the format handling
here prevents reporting tools from silently treating an SFC's multicast tail
as its complete network footprint.
"""

from __future__ import annotations

from typing import Any


def plan_format(plan: dict[str, Any]) -> str:
    """Return the supported plan format name, or raise for an unknown row."""
    if plan.get("version") == "hrl_sfc_plan_v1":
        return "hrl_sfc_plan_v1"
    if isinstance(plan.get("tree_edges"), list):
        return "sdn_tree_plan"
    raise ValueError(f"request {plan.get('request_id')} has an unsupported plan format")


def edge_pairs(path: list[Any]) -> set[tuple[int, int]]:
    """Convert a node path to directed physical edges."""
    return {
        (int(source), int(destination))
        for source, destination in zip(path, path[1:])
    }


def directed_physical_edges(plan: dict[str, Any]) -> set[tuple[int, int]]:
    """Return every directed physical edge used by a deployment plan.

    For an HRL SFC plan this includes all source-to-VNF segments and the
    multicast tree after the final VNF.  This is the correct scope for
    declared bandwidth-hop reporting.
    """
    fmt = plan_format(plan)
    if fmt == "sdn_tree_plan":
        return {tuple(map(int, edge)) for edge in plan["tree_edges"]}

    edges: set[tuple[int, int]] = set()
    for segment in plan.get("segments", []):
        edges.update(edge_pairs(list(segment.get("path", []))))
    multicast = plan.get("multicast", {})
    edges.update(tuple(map(int, edge)) for edge in multicast.get("tree_edges", []))
    return edges


def declared_compute(plan: dict[str, Any], request: dict[str, Any]) -> tuple[float, float]:
    """Return the plan's declared CPU and memory requirement.

    This intentionally represents per-request demand, not the authoritative
    pool consumption.  Reused VNF instances are accounted for only by the
    AllResourceManager ledger snapshot emitted during HRL export.
    """
    if plan_format(plan) == "hrl_sfc_plan_v1":
        placements = plan.get("placement_by_vnf", {})
        return (
            sum(float(value.get("cpu_units", 0.0)) for value in placements.values()),
            sum(float(value.get("memory_units", 0.0)) for value in placements.values()),
        )
    return (
        sum(float(value) for value in request.get("cpu_origin", [])),
        sum(float(value) for value in request.get("memory_origin", [])),
    )
