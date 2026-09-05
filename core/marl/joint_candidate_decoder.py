"""Bounded joint feasibility decoding for request-candidate batches.

The decoder is deliberately policy-agnostic.  It consumes per-request
candidate rankings produced by WQMIX (or a baseline), then returns a jointly
feasible action vector under one immutable resource snapshot.  Reject actions
are represented by ``None`` footprints, which gives the anytime search a
valid incumbent before it examines any deployment candidate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping, Optional, Sequence

from core.marl.batch_deployment_wqmix import (
    ResourceFootprint,
    ResourceSnapshot,
    VNFInstanceRequirement,
)


@dataclass(frozen=True)
class JointDecodeResult:
    actions: tuple[int, ...]
    accepted: int
    rejected: int
    objective_score: float
    elapsed_ms: float
    timed_out: bool
    candidates_examined: int
    greedy_budget_exhausted: bool
    repair_improvements: int
    repair_budget_exhausted: bool
    snapshot_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _CandidateContribution:
    """Precomputed sparse contribution for one candidate.

    The representation mirrors ``joint_footprints_feasible`` exactly.  In
    particular, a footprint carrying VNF instance requirements is charged via
    those requirements, while a footprint without them uses its direct
    CPU/memory maps.
    """

    cpu: Mapping[int, float]
    memory: Mapping[int, float]
    bandwidth: Mapping[tuple[int, int], float]
    instances: Mapping[tuple[int, int], VNFInstanceRequirement]


class _IncrementalAggregate:
    """Mutable aggregate for bounded candidate checks.

    ``test_replace`` is non-mutating and only touches resources referenced by
    the old/new candidates.  This removes the repeated O(batch * candidates)
    full scan from the hot path while retaining the shared-instance semantics.
    """

    def __init__(
        self,
        contributions: Sequence[Sequence[_CandidateContribution]],
        selected: Sequence[int],
        snapshot: ResourceSnapshot,
    ) -> None:
        self.contributions = contributions
        self.selected = list(selected)
        self.snapshot = snapshot
        self.cpu: Counter[int] = Counter()
        self.memory: Counter[int] = Counter()
        self.bandwidth: Counter[tuple[int, int]] = Counter()
        self.instance_requirements: dict[
            tuple[int, int], dict[int, VNFInstanceRequirement]
        ] = {}
        for agent, action in enumerate(self.selected):
            self._add(agent, action)

    def _add(self, agent: int, action: int) -> None:
        contribution = self.contributions[agent][action]
        self.cpu.update(contribution.cpu)
        self.memory.update(contribution.memory)
        self.bandwidth.update(contribution.bandwidth)
        for key, requirement in contribution.instances.items():
            self.instance_requirements.setdefault(key, {})[agent] = requirement

    def _remove(self, agent: int, action: int) -> None:
        contribution = self.contributions[agent][action]
        for key, amount in contribution.cpu.items():
            self.cpu[key] -= amount
            if self.cpu[key] <= 1e-12:
                del self.cpu[key]
        for key, amount in contribution.memory.items():
            self.memory[key] -= amount
            if self.memory[key] <= 1e-12:
                del self.memory[key]
        for key, amount in contribution.bandwidth.items():
            self.bandwidth[key] -= amount
            if self.bandwidth[key] <= 1e-12:
                del self.bandwidth[key]
        affected = contribution.instances.keys()
        for key in affected:
            agents = self.instance_requirements.get(key)
            if agents is not None:
                agents.pop(agent, None)
                if not agents:
                    del self.instance_requirements[key]

    def _max_requirement(
        self, key: tuple[int, int]
    ) -> Optional[VNFInstanceRequirement]:
        requirements = self.instance_requirements.get(key)
        if not requirements:
            return None
        return max(
            requirements.values(),
            key=lambda requirement: float(requirement.cpu) + float(requirement.memory),
        )

    def _totals_with_replacement(
        self, replacements: Mapping[int, int]
    ) -> tuple[Counter[int], Counter[int], Counter[tuple[int, int]], dict[tuple[int, int], dict[int, VNFInstanceRequirement]]]:
        """Build only affected aggregate state for a small replacement set."""
        cpu = self.cpu.copy()
        memory = self.memory.copy()
        bandwidth = self.bandwidth.copy()
        instance_requirements = {
            key: values.copy() for key, values in self.instance_requirements.items()
        }
        affected_instances: set[tuple[int, int]] = set()
        # ``self.cpu``/``self.memory`` contain only direct footprint amounts.
        # Add the currently selected shared-instance maxima once before
        # applying the small replacement delta below.
        for key, agents in self.instance_requirements.items():
            if key in self.snapshot.vnf_instances:
                continue
            requirement = max(
                agents.values(),
                key=lambda value: float(value.cpu) + float(value.memory),
            )
            cpu[int(key[0])] += float(requirement.cpu)
            memory[int(key[0])] += float(requirement.memory)
        # Remove old maxima once per affected key.  The replacement map is
        # then applied to the copied per-agent requirements below.
        affected_instances = set()
        for agent, action in replacements.items():
            old = self.contributions[agent][self.selected[agent]]
            new = self.contributions[agent][action]
            affected_instances.update(old.instances.keys())
            affected_instances.update(new.instances.keys())
        for key in affected_instances:
            if key in self.snapshot.vnf_instances:
                continue
            previous = max(
                self.instance_requirements.get(key, {}).values(),
                key=lambda value: float(value.cpu) + float(value.memory),
                default=None,
            )
            if previous is not None:
                cpu[int(key[0])] -= float(previous.cpu)
                memory[int(key[0])] -= float(previous.memory)
        for agent, action in replacements.items():
            old = self.contributions[agent][self.selected[agent]]
            new = self.contributions[agent][action]
            for key, amount in old.cpu.items():
                cpu[key] -= amount
            for key, amount in new.cpu.items():
                cpu[key] += amount
            for key, amount in old.memory.items():
                memory[key] -= amount
            for key, amount in new.memory.items():
                memory[key] += amount
            for key, amount in old.bandwidth.items():
                bandwidth[key] -= amount
            for key, amount in new.bandwidth.items():
                bandwidth[key] += amount
            for key in old.instances:
                agents = instance_requirements.get(key)
                if agents is not None:
                    agents.pop(agent, None)
                    if not agents:
                        instance_requirements.pop(key, None)
            for key, requirement in new.instances.items():
                instance_requirements.setdefault(key, {})[agent] = requirement
        for key in affected_instances:
            node = int(key[0])
            if key in self.snapshot.vnf_instances:
                continue
            requirement = max(
                instance_requirements.get(key, {}).values(),
                key=lambda value: float(value.cpu) + float(value.memory),
                default=None,
            )
            if requirement is not None:
                cpu[node] += float(requirement.cpu)
                memory[node] += float(requirement.memory)
        return cpu, memory, bandwidth, instance_requirements

    def _feasible_totals(
        self, cpu: Mapping[int, float], memory: Mapping[int, float],
        bandwidth: Mapping[tuple[int, int], float],
    ) -> tuple[bool, str]:
        if any(amount > float(self.snapshot.cpu_remaining.get(node, 0.0)) + 1e-9 for node, amount in cpu.items()):
            return False, "cpu"
        if any(amount > float(self.snapshot.memory_remaining.get(node, 0.0)) + 1e-9 for node, amount in memory.items()):
            return False, "memory"
        if any(amount > float(self.snapshot.bandwidth_remaining.get(edge, 0.0)) + 1e-9 for edge, amount in bandwidth.items()):
            return False, "bandwidth"
        return True, ""

    def test_replace(self, agent: int, action: int) -> tuple[bool, str]:
        cpu, memory, bandwidth, _ = self._totals_with_replacement({agent: action})
        return self._feasible_totals(cpu, memory, bandwidth)

    def test_replacements(self, replacements: Mapping[int, int]) -> tuple[bool, str]:
        cpu, memory, bandwidth, _ = self._totals_with_replacement(replacements)
        return self._feasible_totals(cpu, memory, bandwidth)

    def replace(self, agent: int, action: int) -> None:
        self._remove(agent, self.selected[agent])
        self.selected[agent] = action
        self._add(agent, action)


def joint_footprints_feasible(
    footprints: Sequence[Optional[ResourceFootprint]],
    snapshot: ResourceSnapshot,
) -> tuple[bool, str]:
    """Validate aggregate CPU, memory, and directed bandwidth reservations.

    VNF instances already present in the snapshot consume no additional node
    resources.  Within the proposed batch, a shared ``(node, vnf_type)``
    instance is charged once using the largest advertised requirement.
    """

    cpu: Counter[int] = Counter()
    memory: Counter[int] = Counter()
    bandwidth: Counter[tuple[int, int]] = Counter()
    new_instances: dict[tuple[int, int], Any] = {}

    for footprint in footprints:
        if footprint is None:
            continue
        bandwidth.update(footprint.bandwidth)
        if not footprint.vnf_instances:
            cpu.update(footprint.cpu)
            memory.update(footprint.memory)
            continue
        for requirement in footprint.vnf_instances:
            key = (int(requirement.node), int(requirement.vnf_type))
            if key in snapshot.vnf_instances:
                continue
            previous = new_instances.get(key)
            if previous is None or (
                float(requirement.cpu) + float(requirement.memory)
                > float(previous.cpu) + float(previous.memory)
            ):
                new_instances[key] = requirement

    for requirement in new_instances.values():
        cpu[int(requirement.node)] += float(requirement.cpu)
        memory[int(requirement.node)] += float(requirement.memory)

    if any(
        amount > float(snapshot.cpu_remaining.get(node, 0.0)) + 1e-9
        for node, amount in cpu.items()
    ):
        return False, "cpu"
    if any(
        amount > float(snapshot.memory_remaining.get(node, 0.0)) + 1e-9
        for node, amount in memory.items()
    ):
        return False, "memory"
    if any(
        amount > float(snapshot.bandwidth_remaining.get(edge, 0.0)) + 1e-9
        for edge, amount in bandwidth.items()
    ):
        return False, "bandwidth"
    return True, ""


def decode_joint_candidates(
    candidates: Sequence[Sequence[Optional[ResourceFootprint]]],
    snapshot: ResourceSnapshot,
    rankings: Sequence[Sequence[int]],
    *,
    reject_actions: Optional[Sequence[int]] = None,
    action_mask: Optional[Sequence[Sequence[bool]]] = None,
    scores: Optional[Sequence[Sequence[float]]] = None,
    priorities: Optional[Sequence[Any]] = None,
    top_r: int = 4,
    time_budget_ms: float = 2.0,
    max_greedy_evaluations: int = 48,
    max_repair_evaluations: int = 16,
    remaining_lifetimes_s: Optional[Sequence[float]] = None,
    migration_prepare_s: Optional[Sequence[float]] = None,
) -> JointDecodeResult:
    """Return an anytime, jointly feasible action vector.

    Acceptance count is lexicographically more important than Q-score.  The
    initial all-reject incumbent is always feasible.  The bounded repair pass
    may move an already accepted request to another non-reject candidate when
    that admits an additional request without violating the snapshot.
    """

    started_ns = time.perf_counter_ns()
    agent_count = len(candidates)
    if len(rankings) != agent_count:
        raise ValueError("candidates and rankings must align")
    if top_r <= 0:
        raise ValueError("top_r must be positive")
    if time_budget_ms < 0.0:
        raise ValueError("time_budget_ms must be non-negative")
    if max_greedy_evaluations <= 0 or max_repair_evaluations < 0:
        raise ValueError("invalid decoder evaluation limits")
    if action_mask is not None and len(action_mask) != agent_count:
        raise ValueError("candidates and action_mask must align")
    if scores is not None and len(scores) != agent_count:
        raise ValueError("candidates and scores must align")
    if priorities is not None and len(priorities) != agent_count:
        raise ValueError("candidates and priorities must align")
    if remaining_lifetimes_s is not None and len(remaining_lifetimes_s) != agent_count:
        raise ValueError("remaining_lifetimes_s must align with candidates")
    if migration_prepare_s is not None and len(migration_prepare_s) != agent_count:
        raise ValueError("migration_prepare_s must align with candidates")

    if reject_actions is None:
        resolved_rejects = []
        for request_candidates in candidates:
            try:
                resolved_rejects.append(next(
                    index
                    for index, footprint in enumerate(request_candidates)
                    if footprint is None
                ))
            except StopIteration as exc:
                raise ValueError("every request needs an explicit reject action") from exc
    else:
        if len(reject_actions) != agent_count:
            raise ValueError("candidates and reject_actions must align")
        resolved_rejects = [int(action) for action in reject_actions]

    for agent_index, (request_candidates, reject_action) in enumerate(
        zip(candidates, resolved_rejects)
    ):
        if not 0 <= reject_action < len(request_candidates):
            raise ValueError(f"invalid reject action for request agent {agent_index}")
        if request_candidates[reject_action] is not None:
            raise ValueError(f"reject action for request agent {agent_index} is not None")

    deadline_ns = started_ns + int(float(time_budget_ms) * 1_000_000.0)

    def expired() -> bool:
        return time.perf_counter_ns() >= deadline_ns

    valid_options: list[list[int]] = []
    rank_positions: list[dict[int, int]] = []
    for agent_index, (request_candidates, ranking, reject_action) in enumerate(
        zip(candidates, rankings, resolved_rejects)
    ):
        seen: set[int] = set()
        options: list[int] = []
        positions: dict[int, int] = {}
        for position, raw_action in enumerate(ranking):
            action = int(raw_action)
            if action in seen or not 0 <= action < len(request_candidates):
                continue
            seen.add(action)
            positions[action] = position
            if action == reject_action:
                continue
            if remaining_lifetimes_s is not None:
                prepare = float(migration_prepare_s[agent_index]) if migration_prepare_s is not None else 0.0
                if float(remaining_lifetimes_s[agent_index]) <= prepare + 1e-9:
                    continue
            if action_mask is not None and not bool(action_mask[agent_index][action]):
                continue
            if request_candidates[action] is None:
                continue
            options.append(action)
            if len(options) >= top_r:
                break
        valid_options.append(options)
        rank_positions.append(positions)

    def action_score(agent_index: int, action: int) -> float:
        if scores is not None and 0 <= action < len(scores[agent_index]):
            return float(scores[agent_index][action])
        position = rank_positions[agent_index].get(action, len(rankings[agent_index]))
        return float(-position)

    def make_contribution(
        footprint: Optional[ResourceFootprint],
    ) -> _CandidateContribution:
        if footprint is None:
            return _CandidateContribution({}, {}, {}, {})
        instances: dict[tuple[int, int], VNFInstanceRequirement] = {}
        for requirement in footprint.vnf_instances:
            key = (int(requirement.node), int(requirement.vnf_type))
            previous = instances.get(key)
            if previous is None or (
                float(requirement.cpu) + float(requirement.memory)
                > float(previous.cpu) + float(previous.memory)
            ):
                instances[key] = requirement
        # This intentionally mirrors joint_footprints_feasible: when instance
        # requirements exist, the footprint's direct CPU/memory maps are not
        # charged a second time.
        direct_cpu = {} if instances else dict(footprint.cpu)
        direct_memory = {} if instances else dict(footprint.memory)
        return _CandidateContribution(
            direct_cpu, direct_memory, dict(footprint.bandwidth), instances
        )

    contributions = [
        [make_contribution(footprint) for footprint in request_candidates]
        for request_candidates in candidates
    ]

    actions = list(resolved_rejects)
    candidates_examined = 0
    greedy_evaluations = 0
    greedy_budget_exhausted = False
    repair_improvements = 0
    repair_evaluations = 0
    repair_budget_exhausted = False
    timed_out = False
    aggregate = _IncrementalAggregate(contributions, actions, snapshot)

    order = sorted(
        range(agent_count),
        key=lambda index: (
            priorities[index] if priorities is not None else index,
            index,
        ),
    )
    for agent_index in order:
        if expired():
            timed_out = True
            break
        for action in valid_options[agent_index]:
            if expired():
                timed_out = True
                break
            if greedy_evaluations >= max_greedy_evaluations:
                greedy_budget_exhausted = True
                break
            candidates_examined += 1
            greedy_evaluations += 1
            feasible, _ = aggregate.test_replace(agent_index, action)
            if feasible:
                aggregate.replace(agent_index, action)
                actions[agent_index] = action
                break
        if greedy_budget_exhausted:
            break

    # Bounded one-request relocation: move one accepted request to another
    # non-reject candidate if doing so admits one currently rejected request.
    improved = True
    while (
        improved
        and not expired()
        and not greedy_budget_exhausted
        and repair_evaluations < max_repair_evaluations
    ):
        improved = False
        rejected_indices = [
            index
            for index, action in enumerate(actions)
            if action == resolved_rejects[index]
        ]
        accepted_indices = [
            index
            for index, action in enumerate(actions)
            if action != resolved_rejects[index]
        ]
        for rejected_index in rejected_indices:
            if expired():
                timed_out = True
                break
            for rejected_action in valid_options[rejected_index]:
                direct = list(actions)
                direct[rejected_index] = rejected_action
                candidates_examined += 1
                repair_evaluations += 1
                feasible, _ = aggregate.test_replace(rejected_index, rejected_action)
                if feasible:
                    aggregate.replace(rejected_index, rejected_action)
                    actions = direct
                    repair_improvements += 1
                    improved = True
                    break
                if repair_evaluations >= max_repair_evaluations:
                    repair_budget_exhausted = True
                    break
                for accepted_index in accepted_indices:
                    if expired():
                        timed_out = True
                        break
                    current_action = actions[accepted_index]
                    for alternate in valid_options[accepted_index]:
                        if alternate == current_action:
                            continue
                        if repair_evaluations >= max_repair_evaluations:
                            repair_budget_exhausted = True
                            break
                        proposal = list(direct)
                        proposal[accepted_index] = alternate
                        candidates_examined += 1
                        repair_evaluations += 1
                        feasible, _ = aggregate.test_replacements(
                            {rejected_index: rejected_action, accepted_index: alternate}
                        )
                        if feasible:
                            aggregate.replace(rejected_index, rejected_action)
                            aggregate.replace(accepted_index, alternate)
                            actions = proposal
                            repair_improvements += 1
                            improved = True
                            break
                    if improved or timed_out or repair_budget_exhausted:
                        break
                if improved or timed_out or repair_budget_exhausted:
                    break
            if improved or timed_out or repair_budget_exhausted:
                break

    if repair_evaluations >= max_repair_evaluations:
        repair_budget_exhausted = True

    if expired():
        timed_out = True
    feasible, reason = joint_footprints_feasible(
        [candidates[i][actions[i]] for i in range(agent_count)], snapshot
    )
    if not feasible:
        raise RuntimeError(f"joint decoder produced an infeasible incumbent: {reason}")
    accepted = sum(
        action != resolved_rejects[index] for index, action in enumerate(actions)
    )
    score = sum(
        action_score(index, action)
        for index, action in enumerate(actions)
        if action != resolved_rejects[index]
    )
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return JointDecodeResult(
        actions=tuple(actions),
        accepted=int(accepted),
        rejected=int(agent_count - accepted),
        objective_score=float(score),
        elapsed_ms=float(elapsed_ms),
        timed_out=bool(timed_out),
        candidates_examined=int(candidates_examined),
        greedy_budget_exhausted=bool(greedy_budget_exhausted),
        repair_improvements=int(repair_improvements),
        repair_budget_exhausted=bool(repair_budget_exhausted),
        snapshot_version=int(snapshot.version),
    )
