#!/usr/bin/env python3
"""Validate actionable_reroute_v2 structure, safety evidence, and coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate actionable_reroute_v2 scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/actionable_reroute_v2")
    parser.add_argument("--min-episodes", type=int, default=500)
    parser.add_argument("--min-positive-steps", type=int, default=30)
    parser.add_argument("--min-positive-rate", type=float, default=0.05)
    parser.add_argument("--max-positive-rate", type=float, default=0.30)
    parser.add_argument("--min-negative-decisions", type=int, default=100)
    parser.add_argument("--min-valid-base-rate", type=float, default=0.95)
    parser.add_argument(
        "--coverage-policy", choices=["controlled", "natural"], default="controlled",
        help="controlled enforces per-seed label balance; natural preserves rollout prevalence",
    )
    parser.add_argument("--min-split-positive-steps", type=int, default=30)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def edge_set(rows: Iterable[Dict[str, Any]]) -> set[tuple[int, int]]:
    return {(int(row["u"]), int(row["v"])) for row in rows}


def is_directed_tree(edges: set[tuple[int, int]], root: int) -> bool:
    if not edges or any(u == v for u, v in edges):
        return False
    nodes = {node for edge in edges for node in edge}
    if root not in nodes or len(edges) != len(nodes) - 1:
        return False
    indegree = {node: 0 for node in nodes}
    adjacency = {node: [] for node in nodes}
    for u, v in edges:
        indegree[v] += 1
        adjacency[u].append(v)
    if indegree[root] != 0 or any(indegree[node] != 1 for node in nodes if node != root):
        return False
    visited = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in visited:
            return False
        visited.add(node)
        stack.extend(adjacency[node])
    return visited == nodes


def append_error(errors: list[str], message: str, limit: int = 200) -> None:
    if len(errors) < limit:
        errors.append(message)


def validate_state(state: Dict[str, Any], label: str, errors: list[str]) -> None:
    if not state.get("state_id"):
        append_error(errors, f"{label}: missing state_id")
    vector = state.get("selector", {}).get("obs_vector", [])
    if not vector or any(not 0.0 <= float(value) <= 1.0 for value in vector):
        append_error(errors, f"{label}: selector obs_vector is missing or outside [0,1]")
    if not state.get("ledger_consistent", False):
        append_error(errors, f"{label}: state reports inconsistent ledger")
    for node in state.get("network", {}).get("nodes", []):
        for resource in ("cpu", "mem"):
            cap = float(node[f"{resource}_cap"])
            avail = float(node[f"{resource}_avail"])
            if cap <= 0.0 or avail < -1e-5 or avail > cap + 1e-5:
                append_error(errors, f"{label}: invalid {resource} resource at node {node['node']}")
    for link in state.get("network", {}).get("links", []):
        cap = float(link["cap"])
        avail = float(link["avail"])
        if cap <= 0.0 or avail < -1e-5 or avail > cap + 1e-5:
            append_error(errors, f"{label}: invalid bandwidth on edge {(link['u'], link['v'])}")
    for record in state.get("active_sfts", []):
        tree = edge_set(record.get("tree_edges", []))
        ledger_rows = record.get("edge_allocations", [])
        ledger = edge_set(ledger_rows)
        ledger_consistent = tree == ledger and len(ledger_rows) == len(ledger)
        rooted_tree_valid = is_directed_tree(tree, int(record["source"]))
        if not ledger_consistent:
            append_error(errors, f"{label}: tree/ledger mismatch for req {record.get('req_id')}")
        tree_nodes = {node for edge in tree for node in edge}
        critical = set(int(node) for node in record.get("connected_dests", []))
        critical.update(int(row["node"]) for row in record.get("placement_by_vnf", []))
        critical_connected = critical.issubset(tree_nodes)
        expected = {
            "rooted_tree_valid": rooted_tree_valid,
            "ledger_consistent": ledger_consistent,
            "critical_connected": critical_connected,
            "base_structure_valid": rooted_tree_valid and ledger_consistent and critical_connected,
        }
        if record.get("structure") != expected:
            append_error(errors, f"{label}: incorrect structure labels for req {record.get('req_id')}")


def validate_decision(
    decision: Dict[str, Any], states: Dict[str, Dict[str, Any]], label: str, errors: list[str]
) -> None:
    state_id = decision.get("state_id")
    if state_id not in states:
        append_error(errors, f"{label}: unknown state_id {state_id}")
        return
    available = bool(decision.get("labels", {}).get("action_available", False))
    valid_actions = [int(value) for value in decision.get("valid_actions", [])]
    action = decision.get("planner_action", {})
    vector = decision.get("reroute_obs_vector", [])
    if len(vector) != 10 or any(not 0.0 <= float(value) <= 1.0 for value in vector):
        append_error(errors, f"{label}: invalid reroute_obs_vector")
    if available:
        target = action.get("target") or {}
        if action.get("action_type") != "reroute_edge" or 1 not in valid_actions:
            append_error(errors, f"{label}: positive label does not expose reroute action")
        if action.get("status_code") != "ACTION_AVAILABLE":
            append_error(errors, f"{label}: positive action has wrong status code")
        if not decision.get("request", {}).get("structure", {}).get("base_structure_valid", False):
            append_error(errors, f"{label}: positive action uses an invalid base structure")
        if not target.get("candidate_id") or not target.get("state_fingerprint"):
            append_error(errors, f"{label}: positive action lacks stable identity")
        old_edge = tuple(target.get("old_edge", []))
        path = [int(node) for node in target.get("new_path", [])]
        if len(old_edge) != 2 or len(path) < 3:
            append_error(errors, f"{label}: malformed positive reroute")
        request_tree = edge_set(decision.get("request", {}).get("tree_edges", []))
        if old_edge not in request_tree:
            append_error(errors, f"{label}: replaced edge is absent from request tree")
        topology_edges = edge_set(states[state_id].get("network", {}).get("links", []))
        for index in range(max(0, len(path) - 1)):
            if (path[index], path[index + 1]) not in topology_edges:
                append_error(errors, f"{label}: path contains non-topology edge")
        evidence = decision.get("candidate_evidence", {})
        for margin in ("util_margin", "delay_margin_ms", "growth_margin"):
            if float(evidence.get(margin, -1.0)) < -1e-5:
                append_error(errors, f"{label}: negative safety margin {margin}")
    else:
        if valid_actions != [0]:
            append_error(errors, f"{label}: negative label exposes non-noop action")
        if decision.get("labels", {}).get("rejection_code") == "ACTION_AVAILABLE":
            append_error(errors, f"{label}: negative label uses positive rejection code")


def validate_scenario(
    spec_path: Path, args: argparse.Namespace, errors: list[str]
) -> Dict[str, Any]:
    folder = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    label = str(folder)
    if spec.get("dataset_version") != "actionable_reroute_v2":
        append_error(errors, f"{label}: wrong dataset_version")
    if spec.get("migration_enabled") is not False:
        append_error(errors, f"{label}: migration must be disabled")
    for name, expected in spec.get("files", {}).items():
        path = folder / name
        if not path.exists():
            append_error(errors, f"{label}: missing {name}")
        elif sha256(path) != expected:
            append_error(errors, f"{label}: checksum mismatch for {name}")
    try:
        states = read_jsonl(folder / "states.jsonl")
        decisions = read_jsonl(folder / "decisions.jsonl")
    except Exception as exc:
        append_error(errors, f"{label}: load failed: {exc}")
        return spec
    state_map = {}
    for index, state in enumerate(states, start=1):
        state_label = f"{label}/state#{index}"
        state_id = state.get("state_id")
        if state_id in state_map:
            append_error(errors, f"{state_label}: duplicate state_id")
        state_map[state_id] = state
        validate_state(state, state_label, errors)
    decision_ids = set()
    for index, decision in enumerate(decisions, start=1):
        decision_label = f"{label}/decision#{index}"
        decision_id = decision.get("decision_id")
        if not decision_id or decision_id in decision_ids:
            append_error(errors, f"{decision_label}: missing or duplicate decision_id")
        decision_ids.add(decision_id)
        validate_decision(decision, state_map, decision_label, errors)

    coverage = spec.get("coverage", {})
    positives = sum(bool(row.get("labels", {}).get("action_available")) for row in decisions)
    positive_steps = len({
        int(row["episode"]) for row in decisions
        if row.get("labels", {}).get("action_available")
    })
    negatives = len(decisions) - positives
    episodes = int(coverage.get("episodes", 0))
    computed_rate = positive_steps / max(1, episodes)
    expected_counts = {
        "states": len(states),
        "decision_points": len(decisions),
        "positive_steps": positive_steps,
        "positive_decisions": positives,
        "negative_decisions": negatives,
        "valid_base_decisions": sum(
            bool(row.get("request", {}).get("structure", {}).get("base_structure_valid"))
            for row in decisions
        ),
        "invalid_base_decisions": sum(
            not bool(row.get("request", {}).get("structure", {}).get("base_structure_valid"))
            for row in decisions
        ),
    }
    for key, value in expected_counts.items():
        if int(coverage.get(key, -1)) != value:
            append_error(errors, f"{label}: coverage mismatch for {key}")
    if not bool(coverage.get("ledger_consistent", False)):
        append_error(errors, f"{label}: coverage reports ledger failure")
    if episodes < args.min_episodes:
        append_error(errors, f"{label}: episodes {episodes} < {args.min_episodes}")
    if args.coverage_policy == "controlled":
        if positive_steps < args.min_positive_steps:
            append_error(errors, f"{label}: positive steps {positive_steps} < {args.min_positive_steps}")
        if computed_rate < args.min_positive_rate or computed_rate > args.max_positive_rate:
            append_error(
                errors,
                f"{label}: positive step rate {computed_rate:.4f} outside "
                f"[{args.min_positive_rate:.4f}, {args.max_positive_rate:.4f}]",
            )
    if negatives < args.min_negative_decisions:
        append_error(errors, f"{label}: negative decisions {negatives} < {args.min_negative_decisions}")
    valid_base_rate = expected_counts["valid_base_decisions"] / max(1, len(decisions))
    if valid_base_rate < args.min_valid_base_rate:
        append_error(
            errors,
            f"{label}: valid base rate {valid_base_rate:.4f} < {args.min_valid_base_rate:.4f}",
        )
    return spec


def main() -> int:
    args = parse_args()
    root = Path(args.data)
    if not root.is_absolute():
        root = ROOT / root
    specs = sorted(root.rglob("scenario_spec.json")) if root.exists() else []
    errors = []
    if not specs:
        errors.append(f"no scenario_spec.json files under {root}")
    scenarios = [validate_scenario(path, args, errors) for path in specs]
    split_seeds: Dict[str, set[int]] = {}
    source_hash_splits: Dict[str, set[str]] = {}
    split_coverage: Dict[str, Dict[str, float]] = {}
    planner_hashes = set()
    for spec in scenarios:
        split = str(spec.get("split"))
        split_seeds.setdefault(split, set()).add(int(spec.get("trace_seed", -1)))
        source_hash_splits.setdefault(split, set()).add(str(spec.get("source_request_sha256", "")))
        planner_hashes.add(str(spec.get("planner_config_hash", "")))
        coverage = spec.get("coverage", {})
        aggregate = split_coverage.setdefault(
            split, {"episodes": 0, "positive_steps": 0, "decision_points": 0, "scenarios": 0}
        )
        aggregate["episodes"] += int(coverage.get("episodes", 0))
        aggregate["positive_steps"] += int(coverage.get("positive_steps", 0))
        aggregate["decision_points"] += int(coverage.get("decision_points", 0))
        aggregate["scenarios"] += 1
    if len(planner_hashes) > 1:
        errors.append(f"planner configuration mismatch across scenarios: {len(planner_hashes)} hashes")
    if args.coverage_policy == "natural":
        for split, coverage in split_coverage.items():
            if coverage["positive_steps"] < args.min_split_positive_steps:
                errors.append(
                    f"split {split}: positive steps {int(coverage['positive_steps'])} "
                    f"< {args.min_split_positive_steps}"
                )
            coverage["positive_step_rate"] = (
                coverage["positive_steps"] / max(1, coverage["episodes"])
            )
    split_names = sorted(split_seeds)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = split_seeds[left] & split_seeds[right]
            if overlap:
                errors.append(f"trace seed leakage between {left} and {right}: {sorted(overlap)}")
            hash_overlap = source_hash_splits[left] & source_hash_splits[right]
            if hash_overlap:
                errors.append(f"request file leakage between {left} and {right}: {len(hash_overlap)} hashes")
    result = {
        "ok": not errors,
        "root": str(root),
        "scenarios": len(scenarios),
        "coverage_policy": args.coverage_policy,
        "split_seeds": {key: sorted(value) for key, value in split_seeds.items()},
        "split_coverage": split_coverage,
        "planner_config_hashes": sorted(planner_hashes),
        "errors": errors,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
