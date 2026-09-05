"""Paper-parameter dynamic SFC migration simulator.

The paper supplies resource ranges and training parameters but not a complete
machine-readable CERNET2 topology or the conversion from Telecom Italia
activity units to Mbps. Those two choices are recorded as assumptions in every
result rather than being presented as paper-defined values.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from .telecom_data import DATASET_DOI, load_activity_csv


PAPER_SFC_COUNTS = (10, 15, 20, 25, 30, 35)
POLICIES = ("no_migration", "reactive_mih", "predictive_heuristic", "brain_rule_marl")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260830
    node_count: int = 20
    edge_count: int = 22
    cpu_capacity_range: tuple[float, float] = (150.0, 300.0)
    memory_capacity_range: tuple[float, float] = (200.0, 400.0)
    bandwidth_capacity_range: tuple[float, float] = (100.0, 200.0)
    base_energy_range: tuple[float, float] = (90.0, 150.0)
    vnf_type_count: int = 9
    vnf_count_range: tuple[int, int] = (2, 9)
    lifetime_range: tuple[int, int] = (2, 10)
    cpu_coefficient_range: tuple[float, float] = (0.4, 0.6)
    memory_coefficient_range: tuple[float, float] = (0.4, 0.6)
    delay_bound_ms_range: tuple[float, float] = (30.0, 50.0)
    horizon: int = 10
    overload_penalty: float = 25.0
    objective_weight: float = 0.5
    discount_factor: float = 0.99
    trpo_kl_target: float = 0.01
    conjugate_gradient_steps: int = 15
    learning_rate_range: tuple[float, float] = (5.6e-4, 1e-3)
    learning_rate_decay: float = 0.96
    learning_rate_decay_steps: int = 200
    training_iterations: int = 2000
    traffic_mbps_range: tuple[float, float] = (2.0, 18.0)
    propagation_delay_ms_range: tuple[float, float] = (1.0, 3.0)
    overload_threshold: float = 0.90
    underload_threshold: float = 0.20
    max_migrations_per_slot: int = 4
    migration_cooldown_slots: int = 2
    minimum_utilization_relief: float = 0.02

    def metadata(self) -> dict[str, object]:
        return {
            "paper_parameters": {
                key: value
                for key, value in asdict(self).items()
                if key not in {
                    "seed",
                    "traffic_mbps_range",
                    "propagation_delay_ms_range",
                    "overload_threshold",
                    "underload_threshold",
                    "max_migrations_per_slot",
                    "migration_cooldown_slots",
                    "minimum_utilization_relief",
                }
            },
            "implementation_assumptions": {
                "traffic_mbps_range": self.traffic_mbps_range,
                "propagation_delay_ms_range": self.propagation_delay_ms_range,
                "overload_threshold": self.overload_threshold,
                "underload_threshold": self.underload_threshold,
                "max_migrations_per_slot": self.max_migrations_per_slot,
                "migration_cooldown_slots": self.migration_cooldown_slots,
                "minimum_utilization_relief": self.minimum_utilization_relief,
                "topology": "deterministic connected 20-node/22-link CERNET2-style graph; not digitized from the thesis figure",
                "traffic_mapping": "per-grid causal sliding quantile activity mapped linearly to Mbps",
                "forecast": "causal linear extrapolation from a strict pre-slot history window; brain_rule_marl blends it with the historical peak only",
            },
        }


@dataclass
class VNF:
    vnf_type: int
    node: int
    cpu_coefficient: float
    memory_coefficient: float
    state_factor: float
    last_migration_slot: int = -1_000_000


@dataclass
class SFC:
    sfc_id: int
    source: int
    destination: int
    profile: np.ndarray
    profile_offset: int
    lifetime: int
    age: int
    delay_bound_ms: float
    vnfs: list[VNF] = field(default_factory=list)

    def flow(self, slot: int) -> float:
        # A trace is finite.  Never wrap around to its tail: doing so exposes
        # an unrelated future prefix when a simulation starts near the end.
        # ``profile_offset`` is retained for serialized-plan compatibility,
        # but the runtime trace is on one global timestamp axis and therefore
        # deliberately ignores arbitrary phase offsets.
        index = slot
        if index < 0 or index >= len(self.profile):
            return 0.0
        return float(self.profile[index])


def _build_topology(config: ExperimentConfig) -> nx.Graph:
    rng = random.Random(config.seed)
    graph = nx.Graph()
    for node in range(config.node_count):
        graph.add_node(
            node,
            cpu_capacity=rng.uniform(*config.cpu_capacity_range),
            memory_capacity=rng.uniform(*config.memory_capacity_range),
            base_energy=rng.uniform(*config.base_energy_range),
        )
    edges = [(node, (node + 1) % config.node_count) for node in range(config.node_count)]
    edges.extend([(0, 10), (5, 15)])
    if len(edges) != config.edge_count:
        raise AssertionError("topology edge count does not match configuration")
    for u, v in edges:
        graph.add_edge(
            u,
            v,
            bandwidth_capacity=rng.uniform(*config.bandwidth_capacity_range),
            delay_ms=rng.uniform(*config.propagation_delay_ms_range),
        )
    return graph


def _causal_scale(
    values: Sequence[float], low: float, high: float, *, window: int = 96
) -> np.ndarray:
    """Scale each point using only values strictly before that point.

    The previous implementation used full-series quantiles, which let a
    later trace suffix change earlier observations.  A bounded sliding window
    keeps the transform causal and adapts to non-stationary traffic.
    """
    array = np.asarray(values, dtype=np.float64)
    scaled = np.empty(array.shape, dtype=np.float64)
    for index, value in enumerate(array):
        start = max(0, index - window)
        history = array[start:index]
        positive = history[history > 0.0]
        if positive.size == 0:
            # No prior calibration exists at the beginning of a trace.
            scaled[index] = low if value <= 0.0 else (low + high) / 2.0
            continue
        lower, upper = np.quantile(positive, [0.05, 0.95])
        if upper <= lower + 1e-12:
            scaled[index] = (low + high) / 2.0
            continue
        normalized = float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
        scaled[index] = low + normalized * (high - low)
    return scaled


def _profiles(activity_csv: Path, config: ExperimentConfig) -> tuple[list[int], list[np.ndarray]]:
    timestamps, raw = load_activity_csv(activity_csv)
    profiles = [
        _causal_scale(values, *config.traffic_mbps_range)
        for _, values in sorted(raw.items())
    ]
    if not profiles or len(timestamps) < 2:
        raise ValueError("activity CSV must contain at least one grid and two timestamps")
    return timestamps, profiles


def _new_sfc(
    sfc_id: int,
    profiles: Sequence[np.ndarray],
    graph: nx.Graph,
    config: ExperimentConfig,
    rng: random.Random,
) -> SFC:
    source = rng.randrange(config.node_count)
    destination = rng.randrange(config.node_count - 1)
    if destination >= source:
        destination += 1
    count = rng.randint(*config.vnf_count_range)
    # Start from a feasible energy-consolidated placement.  Five warm servers
    # create the energy/overload trade-off studied by the thesis while keeping
    # the initial route close to the source-destination shortest path.
    warm_servers = list(range(0, config.node_count, max(1, config.node_count // 5)))
    preferred = min(
        warm_servers,
        key=lambda node: nx.shortest_path_length(graph, source, node, weight="delay_ms")
        + nx.shortest_path_length(graph, node, destination, weight="delay_ms"),
    )
    vnfs = [
        VNF(
            vnf_type=rng.randrange(config.vnf_type_count),
            node=preferred,
            cpu_coefficient=rng.uniform(*config.cpu_coefficient_range),
            memory_coefficient=rng.uniform(*config.memory_coefficient_range),
            state_factor=rng.uniform(0.8, 1.2),
        )
        for index in range(count)
    ]
    profile = profiles[sfc_id % len(profiles)]
    return SFC(
        sfc_id=sfc_id,
        source=source,
        destination=destination,
        profile=profile,
        # Profiles share the dataset's global timestamp axis.  A random phase
        # would make a newly created SFC at slot t read profile[t + phase],
        # which is information from a future global slot.  New requests start
        # at the current trace slot instead.
        profile_offset=0,
        lifetime=rng.randint(*config.lifetime_range),
        age=0,
        delay_bound_ms=rng.uniform(*config.delay_bound_ms_range),
        vnfs=vnfs,
    )


def _history(profile: np.ndarray, index: int, window: int = 4) -> np.ndarray:
    """Return the strictly pre-index history, with no cyclic tail access."""
    start = max(0, index - window)
    return np.asarray(profile[start:index], dtype=np.float64)


def _forecast(profile: np.ndarray, index: int, horizon: int) -> float:
    """Causal trend forecast; all calibration points precede ``index``."""
    if index < 0 or index >= len(profile):
        # There is no exogenous observation once a finite trace is exhausted;
        # do not synthesize traffic by extrapolating the old tail.
        return 0.0
    history = _history(profile, index)
    if history.size == 0:
        # At t=0 no forecast can be learned; callers already know current flow.
        return 0.0
    if history.size == 1:
        return max(0.0, float(history[-1]))
    slope = float(np.polyfit(np.arange(history.size), history, 1)[0])
    extrapolated = float(history[-1] + max(0.0, slope) * horizon)
    # Bound only by historical observations.  Never inspect profile[index:]
    # (or the full trace) to cap a prediction.
    historical_peak = float(np.max(history))
    return min(historical_peak, max(0.0, extrapolated))


def _routes(graph: nx.Graph, sfc: SFC) -> list[list[int]]:
    waypoints = [sfc.source, *[vnf.node for vnf in sfc.vnfs], sfc.destination]
    return [
        nx.shortest_path(graph, source=u, target=v, weight="delay_ms")
        for u, v in zip(waypoints, waypoints[1:])
    ]


def _snapshot(
    graph: nx.Graph,
    sfcs: Sequence[SFC],
    slot: int,
) -> tuple[dict[int, float], dict[int, float], dict[tuple[int, int], float], dict[int, float]]:
    cpu = {node: 0.0 for node in graph.nodes}
    memory = {node: 0.0 for node in graph.nodes}
    bandwidth = {tuple(sorted(edge)): 0.0 for edge in graph.edges}
    delays: dict[int, float] = {}
    for sfc in sfcs:
        flow = sfc.flow(slot)
        for vnf in sfc.vnfs:
            cpu[vnf.node] += flow * vnf.cpu_coefficient
            memory[vnf.node] += flow * vnf.memory_coefficient
        paths = _routes(graph, sfc)
        used: set[tuple[int, int]] = set()
        delay = 0.0
        for path in paths:
            for u, v in zip(path, path[1:]):
                edge = tuple(sorted((u, v)))
                used.add(edge)
                delay += float(graph.edges[u, v]["delay_ms"])
        for edge in used:
            bandwidth[edge] += flow
        delays[sfc.sfc_id] = delay
    return cpu, memory, bandwidth, delays


def _node_utilization(graph: nx.Graph, cpu: dict[int, float], memory: dict[int, float]) -> dict[int, float]:
    return {
        node: max(
            cpu[node] / float(graph.nodes[node]["cpu_capacity"]),
            memory[node] / float(graph.nodes[node]["memory_capacity"]),
        )
        for node in graph.nodes
    }


def _candidate_score(
    graph: nx.Graph,
    sfc: SFC,
    vnf_index: int,
    target: int,
    flow: float,
    cpu: dict[int, float],
    memory: dict[int, float],
    config: ExperimentConfig,
) -> tuple[float, float, float] | None:
    vnf = sfc.vnfs[vnf_index]
    if target == vnf.node:
        return None
    projected_cpu = cpu[target] + flow * vnf.cpu_coefficient
    projected_memory = memory[target] + flow * vnf.memory_coefficient
    utilization = max(
        projected_cpu / float(graph.nodes[target]["cpu_capacity"]),
        projected_memory / float(graph.nodes[target]["memory_capacity"]),
    )
    if utilization > config.overload_threshold:
        return None
    source_after = max(
        (cpu[vnf.node] - flow * vnf.cpu_coefficient)
        / float(graph.nodes[vnf.node]["cpu_capacity"]),
        (memory[vnf.node] - flow * vnf.memory_coefficient)
        / float(graph.nodes[vnf.node]["memory_capacity"]),
    )
    before_peak = max(
        cpu[vnf.node] / float(graph.nodes[vnf.node]["cpu_capacity"]),
        memory[vnf.node] / float(graph.nodes[vnf.node]["memory_capacity"]),
        cpu[target] / float(graph.nodes[target]["cpu_capacity"]),
        memory[target] / float(graph.nodes[target]["memory_capacity"]),
    )
    relief = before_peak - max(source_after, utilization)
    if relief < config.minimum_utilization_relief:
        return None
    original = vnf.node
    vnf.node = target
    try:
        delay = sum(
            float(graph.edges[u, v]["delay_ms"])
            for path in _routes(graph, sfc)
            for u, v in zip(path, path[1:])
        )
    finally:
        vnf.node = original
    if delay > sfc.delay_bound_ms:
        return None
    distance = nx.shortest_path_length(graph, original, target, weight="delay_ms")
    migration_cost = flow * vnf.memory_coefficient * vnf.state_factor * float(distance)
    return (
        utilization + 0.002 * migration_cost + delay / sfc.delay_bound_ms - relief,
        migration_cost,
        relief,
    )


def _migrate(
    policy: str,
    graph: nx.Graph,
    sfcs: list[SFC],
    slot: int,
    config: ExperimentConfig,
) -> tuple[int, float]:
    if policy == "no_migration":
        return 0, 0.0
    migrations = 0
    migration_cost = 0.0
    for _ in range(config.max_migrations_per_slot):
        cpu, memory, _, _ = _snapshot(graph, sfcs, slot)
        current = _node_utilization(graph, cpu, memory)
        projected_cpu = dict(cpu)
        projected_memory = dict(memory)
        planning_flow = {sfc.sfc_id: sfc.flow(slot) for sfc in sfcs}
        if policy in {"predictive_heuristic", "brain_rule_marl"}:
            projected_cpu = {node: 0.0 for node in graph.nodes}
            projected_memory = {node: 0.0 for node in graph.nodes}
            for sfc in sfcs:
                predicted = _forecast(sfc.profile, slot, config.horizon)
                if policy == "brain_rule_marl":
                    # Brain/Migration/Reroute coordination remains causal:
                    # use the recent historical peak as a risk signal rather
                    # than peeking at future slots.
                    history = _history(sfc.profile, slot, config.horizon)
                    # With no pre-slot samples the causal prior is zero; the
                    # measured current flow remains available to the trigger,
                    # but is not smuggled into the historical forecast.
                    historical_peak = float(np.max(history)) if history.size else 0.0
                    predicted = 0.5 * predicted + 0.5 * historical_peak
                planning_flow[sfc.sfc_id] = max(sfc.flow(slot), predicted)
                for vnf in sfc.vnfs:
                    projected_cpu[vnf.node] += predicted * vnf.cpu_coefficient
                    projected_memory[vnf.node] += predicted * vnf.memory_coefficient
        projected = _node_utilization(graph, projected_cpu, projected_memory)
        trigger = current if policy == "reactive_mih" else projected
        overloaded = [node for node, value in trigger.items() if value > config.overload_threshold]
        if not overloaded:
            break
        source = max(overloaded, key=lambda node: (trigger[node], node))
        options: list[tuple[float, float, int, int, int, float]] = []
        for sfc_index, sfc in enumerate(sfcs):
            flow = planning_flow[sfc.sfc_id]
            for vnf_index, vnf in enumerate(sfc.vnfs):
                if vnf.node != source:
                    continue
                if slot - vnf.last_migration_slot < config.migration_cooldown_slots:
                    continue
                for target in graph.nodes:
                    scored = _candidate_score(
                        graph, sfc, vnf_index, target, flow, cpu, memory, config
                    )
                    if scored is None:
                        continue
                    score, cost, relief = scored
                    if policy == "brain_rule_marl":
                        score -= 0.02 * relief
                    options.append((score, -relief, sfc_index, vnf_index, target, cost))
        if not options:
            break
        _, _, sfc_index, vnf_index, target, cost = min(options)
        sfc = sfcs[sfc_index]
        sfc.vnfs[vnf_index].node = target
        sfc.vnfs[vnf_index].last_migration_slot = slot
        migrations += 1
        migration_cost += cost
    return migrations, migration_cost


def _run_one(
    profiles: Sequence[np.ndarray],
    timestamps: Sequence[int],
    sfc_count: int,
    policy: str,
    config: ExperimentConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng = random.Random(config.seed + sfc_count)
    graph = _build_topology(config)
    sfcs = [_new_sfc(index, profiles, graph, config, rng) for index in range(sfc_count)]
    next_sfc_id = sfc_count
    rows: list[dict[str, object]] = []
    for slot, timestamp in enumerate(timestamps):
        refreshed: list[SFC] = []
        for sfc in sfcs:
            sfc.age += 1
            if sfc.age >= sfc.lifetime:
                refreshed.append(_new_sfc(next_sfc_id, profiles, graph, config, rng))
                next_sfc_id += 1
            else:
                refreshed.append(sfc)
        sfcs = refreshed
        migrations, migration_cost = _migrate(policy, graph, sfcs, slot, config)
        cpu, memory, bandwidth, delays = _snapshot(graph, sfcs, slot)
        utilization = _node_utilization(graph, cpu, memory)
        overloaded_nodes = sum(value > 1.0 for value in utilization.values())
        overloaded_links = sum(
            bandwidth[tuple(sorted(edge))] > float(graph.edges[edge]["bandwidth_capacity"])
            for edge in graph.edges
        )
        active_nodes = [node for node in graph.nodes if cpu[node] > 1e-12 or memory[node] > 1e-12]
        energy = sum(
            float(graph.nodes[node]["base_energy"]) * (0.7 + 0.3 * min(1.5, utilization[node]))
            for node in active_nodes
        )
        sla_met = sum(delays[sfc.sfc_id] <= sfc.delay_bound_ms for sfc in sfcs)
        rows.append(
            {
                "slot": slot,
                "timestamp_ms": timestamp,
                "policy": policy,
                "sfc_count": sfc_count,
                "energy": energy,
                "migration_cost": migration_cost,
                "operating_cost_raw": config.objective_weight * energy
                + (1.0 - config.objective_weight) * migration_cost
                + config.overload_penalty * (overloaded_nodes + overloaded_links),
                "migrations": migrations,
                "overloaded_nodes": overloaded_nodes,
                "overloaded_links": overloaded_links,
                "active_nodes": len(active_nodes),
                "mean_node_utilization": fmean(utilization.values()),
                "max_node_utilization": max(utilization.values()),
                "sla_met": sla_met,
                "sla_total": len(sfcs),
            }
        )
    summary = {
        "policy": policy,
        "sfc_count": sfc_count,
        "slots": len(rows),
        "average_energy": fmean(float(row["energy"]) for row in rows),
        "average_migration_cost": fmean(float(row["migration_cost"]) for row in rows),
        "average_operating_cost_raw": fmean(float(row["operating_cost_raw"]) for row in rows),
        "average_overloaded_nodes": fmean(float(row["overloaded_nodes"]) for row in rows),
        "average_overloaded_links": fmean(float(row["overloaded_links"]) for row in rows),
        "average_migrations": fmean(float(row["migrations"]) for row in rows),
        "total_migrations": sum(int(row["migrations"]) for row in rows),
        "sla_satisfaction": sum(int(row["sla_met"]) for row in rows)
        / max(1, sum(int(row["sla_total"]) for row in rows)),
    }
    return summary, rows


def run_sweep(
    activity_csv: Path,
    output_dir: Path,
    *,
    config: ExperimentConfig | None = None,
    sfc_counts: Iterable[int] = PAPER_SFC_COUNTS,
    policies: Iterable[str] = POLICIES,
    max_slots: int | None = None,
) -> list[dict[str, object]]:
    config = config or ExperimentConfig()
    timestamps, profiles = _profiles(activity_csv, config)
    if max_slots is not None:
        timestamps = timestamps[:max_slots]
        profiles = [profile[:max_slots] for profile in profiles]
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for count in sfc_counts:
        for policy in policies:
            if policy not in POLICIES:
                raise ValueError(f"unknown policy: {policy}")
            summary, rows = _run_one(profiles, timestamps, int(count), policy, config)
            summaries.append(summary)
            all_rows.extend(rows)

    for count in {int(row["sfc_count"]) for row in summaries}:
        group = [row for row in summaries if int(row["sfc_count"]) == count]
        energy_scale = max(float(row["average_energy"]) for row in group) or 1.0
        migration_scale = max(float(row["average_migration_cost"]) for row in group) or 1.0
        for row in group:
            energy_normalized = float(row["average_energy"]) / energy_scale
            migration_normalized = float(row["average_migration_cost"]) / migration_scale
            row["average_operating_cost_normalized"] = (
                config.objective_weight * energy_normalized
                + (1.0 - config.objective_weight) * migration_normalized
                + config.overload_penalty
                * (
                    float(row["average_overloaded_nodes"])
                    + float(row["average_overloaded_links"])
                )
            )

    with (output_dir / "timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    spec = {
        "source": {"dataset_doi": DATASET_DOI, "activity_csv": str(activity_csv.resolve())},
        "config": asdict(config),
        **config.metadata(),
        "sfc_counts": list(sfc_counts),
        "policies": list(policies),
        "slots": len(timestamps),
        "result_semantics": {
            "reactive_mih": "local reimplementation inspired by the thesis baseline, not the authors' source code",
            "predictive_heuristic": "causal trend forecast heuristic; not the thesis MNTP-TTM neural predictor",
            "brain_rule_marl": "rule-coordinated Brain/Migration/Reroute roles; not a trained MARL checkpoint",
        },
    }
    (output_dir / "experiment_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summaries
