"""Causal, action-dependent SFT migration simulation with an injected ledger.

Arrivals, departures and offered-traffic samples are exogenous. Placements,
reservations and fluid queues are endogenous and never loaded from next-frame
snapshots. Gymnasium is the maintained Gym API (reset pair, step five-tuple).
This is a fluid simulation, not a packet-level or measured-SLA environment.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from core.marl.batch_deployment_wqmix import AtomicResourceLedger, ResourceFootprint
from core.marl.deployment_topk import CompletePlanCandidateGenerator, validate_complete_plan
from core.marl.migration_candidates import (
    MigrationCandidateGenerator, migration_batch_candidate_features,
    migration_global_state_features, migration_request_features,
    plan_edge_multiplicity,
)
from core.marl.migration_scheduler import MigrationTask


class DynamicMigrationEnv(gym.Env):
    """Own a supplied AtomicResourceLedger for a resettable simulation episode.

    ``requests`` uses runtime trace rows (id, arrival_time, leave_time, vnf,
    bw_origin, source_dpid, destination_dpids). A row may contain ``plan``;
    otherwise a fresh complete plan is generated at its actual arrival.
    ``initial_plans`` supplies plans for already committed active requests.
    ``traffic_trace`` rows are {timestamp, request_id, bandwidth_mbps}; a value
    becomes visible only at that timestamp and is held until the next sample.

    Reservations use the request's contracted bw_origin. Offered traffic is
    separate: excess offered load creates queues, never impossible reservations.
    Action 0 always means no migration. Actions 1..top_k select regenerated
    complete candidates for the corresponding active task. The existing
    candidate builder currently supports internal VNF stages only.
    """
    metadata = {"render_modes": []}

    def __init__(self, ledger: AtomicResourceLedger,
                 requests: Sequence[Mapping[str, Any]], profile: Mapping[str, Any], *,
                 initial_plans: Mapping[int, Mapping[str, Any]] | None = None,
                 traffic_trace: Sequence[Mapping[str, Any]] | None = None,
                 slot_seconds: float = 1.0, max_agents: int = 8, top_k: int = 8,
                 start_time: float = 0.0, max_steps: int | None = None,
                 control_plane_ms: float = 344.0, state_copy_mbps: float = 100.0):
        super().__init__()
        if not isinstance(ledger, AtomicResourceLedger):
            raise TypeError("ledger must be an AtomicResourceLedger instance")
        if not np.isfinite(slot_seconds) or slot_seconds <= 0 or max_agents < 1 or top_k < 1:
            raise ValueError("slot_seconds, max_agents and top_k must be positive")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.ledger = ledger
        self.profile = copy.deepcopy(dict(profile))
        self.requests = sorted(copy.deepcopy(list(requests)),
                               key=lambda r: (float(r['arrival_time']), int(r['id'])))
        self.request_by_id = {int(r['id']): r for r in self.requests}
        if len(self.request_by_id) != len(self.requests):
            raise ValueError("request ids must be unique")
        for r in self.requests:
            values = [float(r['arrival_time']), float(r['leave_time']), float(r['bw_origin'])]
            if not all(np.isfinite(values)) or values[1] <= values[0] or values[2] <= 0:
                raise ValueError("requests require finite arrival < leave and positive contracted bandwidth")
        self.traffic_trace = sorted(copy.deepcopy(list(traffic_trace or [])),
                                    key=lambda r: (float(r['timestamp']), int(r['request_id'])))
        for row in self.traffic_trace:
            if int(row['request_id']) not in self.request_by_id:
                raise ValueError("traffic sample has unknown request_id")
            if not np.isfinite(float(row['timestamp'])) or not np.isfinite(float(row['bandwidth_mbps'])) or float(row['bandwidth_mbps']) < 0:
                raise ValueError("traffic samples require finite timestamps and nonnegative bandwidth")
        self.start_time = float(start_time)
        if not np.isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        self.slot_seconds, self.max_agents, self.top_k = float(slot_seconds), int(max_agents), int(top_k)
        self.max_steps = max_steps
        self.initial_plans = {int(k): copy.deepcopy(dict(v)) for k, v in (initial_plans or {}).items()}
        if ledger.prepared_replacements:
            raise ValueError("cannot start an episode with in-flight ledger replacements")
        if set(ledger.allocations) != set(self.initial_plans):
            raise ValueError("initial_plans must cover exactly the injected ledger's active allocations")
        self._baseline_allocations = copy.deepcopy(ledger.allocations)
        for rid, plan in self.initial_plans.items():
            r = self.request_by_id.get(rid)
            if r is None or not float(r['arrival_time']) <= self.start_time < float(r['leave_time']):
                raise ValueError("initial services must be active at start_time and present in requests")
            validate_complete_plan(plan, r, self.profile)
            expected = ResourceFootprint.from_sfc_plan(plan, float(r['bw_origin']))
            actual = ledger.allocations[rid]
            if expected.bandwidth != actual.bandwidth or expected.vnf_instances != actual.vnf_instances:
                raise ValueError("initial plan and ledger footprint disagree")
        self.migration_generator = MigrationCandidateGenerator(
            self.profile, max_candidates=self.top_k,
            cpu_capacity=ledger.cpu_capacity, memory_capacity=ledger.memory_capacity,
            control_plane_ms=control_plane_ms, state_copy_mbps=state_copy_mbps)
        self.deployment_generator = CompletePlanCandidateGenerator(self.profile, max_candidates=1)
        self.nodes = sorted(ledger.cpu_capacity)
        self.edges = sorted(ledger.bandwidth_capacity)
        self.service_ids = sorted(self.request_by_id)
        self.max_services = max(1, len(self.service_ids))
        self.max_vnfs = max((len(r['vnf']) for r in self.requests), default=1)
        self.action_space = gym.spaces.MultiDiscrete([self.top_k + 1] * self.max_agents)
        box = lambda shape: gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'request_observations': box((self.max_agents, 14)),
            'candidate_features': box((self.max_agents, self.top_k + 1, 24)),
            'states': box((12,)),
            'action_mask': gym.spaces.MultiBinary((self.max_agents, self.top_k + 1)),
            'agent_mask': gym.spaces.MultiBinary(self.max_agents),
            'node_state': box((len(self.nodes), 3)),
            'link_state': box((len(self.edges), 3)),
            'uncertainty': box((len(self.nodes),)),
            'placements': gym.spaces.Box(0, max(self.nodes, default=1),
                                         (self.max_services, self.max_vnfs), dtype=np.int64),
            'service_mask': gym.spaces.MultiBinary(self.max_services),
            'time': box((1,)),
        })
        self._has_reset = False
        self._terminated = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.ledger.prepared_replacements:
            raise RuntimeError("cannot reset while a ledger replacement is in flight")
        allowed = set(self.initial_plans) | set(getattr(self, 'active_plans', {}))
        if not set(self.ledger.allocations) <= allowed:
            raise RuntimeError("ledger was modified by another owner; refusing destructive reset")
        for rid in list(self.ledger.allocations):
            self.ledger.release(rid)
        if self._baseline_allocations:
            ids = sorted(self._baseline_allocations)
            result = self.ledger.commit_exact(ids, [[self._baseline_allocations[i]] for i in ids],
                                              [0] * len(ids), expected_version=self.ledger.snapshot().version)
            if not result['committed']:
                raise RuntimeError(f"cannot restore initial ledger: {result}")
        self.active_plans = copy.deepcopy(self.initial_plans)
        self.current_traffic = {}
        self.node_queues = {}  # (request, VNF stage) -> outstanding CPU-work units
        self.link_queues = {}  # (request, u, v) -> outstanding Mbit at physical link
        self.now, self.steps = self.start_time, 0
        self._arrival_index = self._traffic_index = 0
        self._accepted = set(self.initial_plans)
        self._rejected = set()
        self._has_reset, self._terminated = True, False
        self._last_events = self._process_events(self.now)
        self._refresh_tasks()
        return self.get_state(), self._info()

    def _process_events(self, timestamp):
        events = {'arrivals': [], 'accepted': [], 'rejected': [], 'departures': [],
                  'traffic_updates': [], 'expired_queue_work': 0.0}
        # Departure first implements half-open [arrival, leave) lifetimes.
        for rid in list(self.active_plans):
            if float(self.request_by_id[rid]['leave_time']) <= timestamp + 1e-10:
                self.ledger.release(rid)
                self.active_plans.pop(rid)
                events['departures'].append(rid)
                for store in (self.node_queues, self.link_queues):
                    for key in [k for k in store if k[0] == rid]:
                        events['expired_queue_work'] += store.pop(key)
        while self._arrival_index < len(self.requests):
            request = self.requests[self._arrival_index]
            if float(request['arrival_time']) > timestamp + 1e-10:
                break
            self._arrival_index += 1
            rid = int(request['id'])
            if rid in self.initial_plans:
                continue
            events['arrivals'].append(rid)
            if float(request['leave_time']) <= timestamp + 1e-10:
                self._rejected.add(rid); events['rejected'].append(rid); continue
            snapshot = self.ledger.snapshot()
            plan = copy.deepcopy(request.get('plan'))
            if plan is None:
                candidate = self.deployment_generator.generate_first_feasible(request, snapshot)
                plan = candidate.plan if candidate is not None else None
            if plan is not None:
                validate_complete_plan(plan, request, self.profile)
                fp = ResourceFootprint.from_sfc_plan(plan, float(request['bw_origin']))
                result = self.ledger.commit_exact([rid], [[fp]], [0], expected_version=snapshot.version)
            else:
                result = {'committed': False}
            if result['committed']:
                self.active_plans[rid] = copy.deepcopy(plan)
                self._accepted.add(rid); events['accepted'].append(rid)
            else:
                self._rejected.add(rid); events['rejected'].append(rid)
        while self._traffic_index < len(self.traffic_trace):
            row = self.traffic_trace[self._traffic_index]
            if float(row['timestamp']) > timestamp + 1e-10:
                break
            self._traffic_index += 1
            rid = int(row['request_id'])
            self.current_traffic[rid] = float(row['bandwidth_mbps'])
            events['traffic_updates'].append(rid)
        return events

    def _offered(self, rid):
        return self.current_traffic.get(rid, float(self.request_by_id[rid]['bw_origin']))

    def _queue_totals(self):
        nodes = defaultdict(float); links = defaultdict(float)
        for (rid, stage), work in self.node_queues.items():
            plan = self.active_plans.get(rid)
            if plan is not None:
                nodes[int(plan['placement_by_vnf'][str(stage)]['dc_node'])] += work
        for (_, u, v), work in self.link_queues.items():
            links[(u, v)] += work
        return nodes, links

    def _advance_queues(self, dt):
        """Work-conserving proportional fluid service, with no future samples.

        VNF backlog follows the migrated stage. Old physical-link backlog drains
        at its old link; only newly offered traffic follows the new route.
        """
        node_work = defaultdict(dict); link_work = defaultdict(dict)
        for rid, plan in self.active_plans.items():
            rate = self._offered(rid)
            contracted = float(self.request_by_id[rid]['bw_origin'])
            for raw_stage, placement in plan['placement_by_vnf'].items():
                stage = int(raw_stage); key = (rid, stage); node = int(placement['dc_node'])
                cpu_per_mbit = float(placement.get('cpu_units', 0.0)) / contracted
                node_work[node][key] = self.node_queues.get(key, 0.0) + rate * cpu_per_mbit * dt
            for edge, count in plan_edge_multiplicity(plan).items():
                key = (rid, *edge)
                link_work[edge][key] = self.link_queues.get(key, 0.0) + rate * count * dt
        for key, work in self.link_queues.items():
            if key[0] in self.active_plans:
                link_work[(key[1], key[2])].setdefault(key, work)
        new_node, new_link = {}, {}
        for pools, capacities, target in (
            (node_work, self.ledger.cpu_capacity, new_node),
            (link_work, self.ledger.bandwidth_capacity, new_link),
        ):
            for resource, workloads in pools.items():
                total = sum(workloads.values())
                residual_ratio = max(0.0, 1.0 - capacities.get(resource, 0.0) * dt / total) if total else 0.0
                for key, work in workloads.items():
                    remaining = work * residual_ratio
                    if remaining > 1e-10: target[key] = remaining
        self.node_queues, self.link_queues = new_node, new_link

    def _refresh_tasks(self):
        self._snapshot = self.ledger.snapshot()
        snapshot = self._snapshot
        node_queues, _ = self._queue_totals()
        eligible = []
        for rid, plan in self.active_plans.items():
            placements = plan['placement_by_vnf']
            for stage in sorted(map(int, placements)):
                if stage <= 0 or str(stage + 1) not in placements:
                    continue
                p = placements[str(stage)]; node = int(p['dc_node'])
                util = max(1 - snapshot.cpu_remaining[node] / self.ledger.cpu_capacity[node],
                           1 - snapshot.memory_remaining[node] / self.ledger.memory_capacity[node])
                eligible.append((-node_queues[node], -util, rid, stage))
        self.tasks, self.candidates = [], []
        used_requests = set()
        for _, negative_util, rid, stage in sorted(eligible):
            if rid in used_requests or len(self.tasks) >= self.max_agents: continue
            used_requests.add(rid)
            request = self.request_by_id[rid]; plan = self.active_plans[rid]
            p = plan['placement_by_vnf'][str(stage)]; node = int(p['dc_node'])
            task = MigrationTask(
                task_id=f'{rid}:{stage}:{self.steps}', request_id=rid, stage=stage,
                vnf_type=int(p['vnf_type']), old_node=node,
                cpu=float(p.get('cpu_units', 0.0)), memory=float(p.get('memory_units', 0.0)),
                bandwidth_mbps=float(request['bw_origin']), state_size_mb=float(p.get('state_size_mb', 0.0)),
                current_utilization=-negative_util, predicted_utilization=-negative_util,
                utilization_relief=float(p.get('cpu_units', 0.0)) / self.ledger.cpu_capacity[node],
                sla_risk=0.0, remaining_lifetime_s=float(request['leave_time']) - self.now,
                estimated_migration_ms=self.migration_generator.control_plane_ms,
                priority=-negative_util, created_at=self.now)
            rows = self.migration_generator.generate(task, plan, snapshot,
                         delay_bound_ms=float(request.get('delay_bound_ms', np.inf)))
            self.tasks.append(task); self.candidates.append([None, *rows])
        self._mask = np.zeros((self.max_agents, self.top_k + 1), dtype=np.int8)
        self._mask[:, 0] = 1
        for i, rows in enumerate(self.candidates):
            for j, c in enumerate(rows[1:], 1): self._mask[i, j] = int(c.feasible)

    def get_state(self):
        if not self._has_reset: raise RuntimeError('call reset before get_state')
        snapshot = self.ledger.snapshot()
        request_obs = np.zeros((self.max_agents, 14), dtype=np.float32)
        candidates = np.zeros((self.max_agents, self.top_k + 1, 24), dtype=np.float32)
        agent_mask = np.zeros(self.max_agents, dtype=np.int8)
        features = migration_batch_candidate_features(self.candidates, self._snapshot) if self.candidates else []
        for i, (task, rows) in enumerate(zip(self.tasks, features)):
            agent_mask[i] = 1; request_obs[i] = migration_request_features(task)
            candidates[i, :len(rows)] = rows
        placements = np.zeros((self.max_services, self.max_vnfs), dtype=np.int64)
        service_mask = np.zeros(self.max_services, dtype=np.int8)
        for i, rid in enumerate(self.service_ids):
            if rid in self.active_plans:
                service_mask[i] = 1
                chain = self.active_plans[rid]['chain_nodes']
                placements[i, :len(chain)] = chain
        nq, lq = self._queue_totals(); offered = defaultdict(float)
        for rid, plan in self.active_plans.items():
            for edge, count in plan_edge_multiplicity(plan).items(): offered[edge] += count * self._offered(rid)
        # Causal uncertainty signal: standard deviation of the recent offered
        # load strictly before ``now`` for each currently used node.
        uncertainty = np.zeros(len(self.nodes), dtype=np.float32)
        history_by_node = {node: [] for node in self.nodes}
        for row in self.traffic_trace:
            timestamp = float(row['timestamp'])
            if timestamp >= self.now:
                break
            rid = int(row['request_id'])
            plan = self.active_plans.get(rid)
            if plan is None:
                continue
            for placement in plan['placement_by_vnf'].values():
                node = int(placement['dc_node'])
                if node in history_by_node:
                    history_by_node[node].append(float(row['bandwidth_mbps']))
        for index, node in enumerate(self.nodes):
            values = history_by_node[node][-16:]
            uncertainty[index] = float(np.std(values)) if len(values) >= 2 else 0.0
        return {
            'request_observations': request_obs, 'candidate_features': candidates,
            'states': np.asarray(migration_global_state_features(snapshot, self.tasks), dtype=np.float32),
            'agent_mask': agent_mask, 'action_mask': self._mask.copy(),
            'node_state': np.asarray([[self.ledger.cpu_used[n], self.ledger.memory_used[n], nq[n]] for n in self.nodes], dtype=np.float32).reshape(len(self.nodes), 3),
            'link_state': np.asarray([[self.ledger.bandwidth_used[e], offered[e], lq[e]] for e in self.edges], dtype=np.float32).reshape(len(self.edges), 3),
            'placements': placements, 'service_mask': service_mask,
            'time': np.asarray([self.now], dtype=np.float32),
            'uncertainty': uncertainty,
        }

    def _info(self):
        return {'time': self.now, 'ledger_version': self.ledger.snapshot().version,
                'active_request_ids': sorted(self.active_plans), 'accepted_count': len(self._accepted),
                'rejected_count': len(self._rejected), 'events': copy.deepcopy(self._last_events),
                'task_ids': [t.task_id for t in self.tasks], 'noop_action': 0,
                'transition_semantics': 'action_dependent_ledger_and_fluid_queues',
                'data_plane_sla_measured': False}

    def step(self, actions):
        if not self._has_reset or self._terminated: raise RuntimeError('reset required before step')
        values = np.asarray(actions)
        if values.shape != (self.max_agents,) or not np.issubdtype(values.dtype, np.integer) or not self.action_space.contains(values):
            raise ValueError('actions must be an integer vector matching action_space')
        before_queue = sum(self.node_queues.values()) + sum(self.link_queues.values())
        results = []
        stale = self.ledger.snapshot().version != self._snapshot.version
        for i, raw in enumerate(values):
            action = int(raw)
            if action == 0: continue
            if stale or not self._mask[i, action]:
                results.append({'applied': False, 'reason': 'version_mismatch' if stale else 'masked_action', 'agent': i}); continue
            task = self.tasks[i]; candidate = self.candidates[i][action]
            plan = copy.deepcopy(candidate.plan)
            validate_complete_plan(plan, self.request_by_id[task.request_id], self.profile)
            footprint = ResourceFootprint.from_sfc_plan(plan, task.bandwidth_mbps)
            result = self.ledger.apply_migration(task.request_id, footprint,
                                expected_version=self.ledger.snapshot().version)
            result = dict(result); result.update({'request_id': task.request_id, 'stage': task.stage, 'agent': i})
            if result.get('applied', result.get('replaced', False)):
                self.active_plans[task.request_id] = plan
                result.update({'old_node': task.old_node, 'target_node': candidate.target_node})
            results.append(result)
        target_time = self.now + self.slot_seconds
        merged = {'arrivals': [], 'accepted': [], 'rejected': [], 'departures': [], 'traffic_updates': [], 'expired_queue_work': 0.0}
        # Split a slot at all exogenous event times. No future payload is used
        # during the preceding interval, and short requests are not skipped.
        while self.now < target_time - 1e-10:
            times = [target_time]
            if self._arrival_index < len(self.requests): times.append(float(self.requests[self._arrival_index]['arrival_time']))
            if self._traffic_index < len(self.traffic_trace): times.append(float(self.traffic_trace[self._traffic_index]['timestamp']))
            times.extend(float(self.request_by_id[r]['leave_time']) for r in self.active_plans)
            next_time = min(t for t in times if t > self.now + 1e-10)
            self._advance_queues(next_time - self.now)
            self.now = next_time
            events = self._process_events(self.now)
            for key in merged:
                if isinstance(merged[key], list): merged[key].extend(events[key])
                else: merged[key] += events[key]
        self.steps += 1
        terminated = self._arrival_index >= len(self.requests) and not self.active_plans
        truncated = self.max_steps is not None and self.steps >= self.max_steps and not terminated
        self._terminated = terminated or truncated
        self._last_events = merged
        self._refresh_tasks()
        after_queue = sum(self.node_queues.values()) + sum(self.link_queues.values())
        # Phase-1 diagnostic reward only; economic/lifetime reward is stage 2.
        failures = sum(not r.get('applied', r.get('replaced', False)) for r in results)
        reward = float(before_queue - after_queue - failures)
        info = self._info(); info['migrations'] = results
        info['reward_semantics'] = 'diagnostic_queue_work_reduction_minus_failed_actions'
        return self.get_state(), reward, bool(terminated), bool(truncated), info

    def step_rankings(self, rankings, *, scores=None):
        """Convenience adapter; rankings are current action ids, including 0.

        Per-request migration commits are sequential, each atomically rechecked.
        This method does not claim an atomic multi-request joint transaction.
        """
        if len(rankings) != len(self.tasks):
            raise ValueError('one ranking is required per active task')
        actions = np.zeros(self.max_agents, dtype=np.int64)
        for i, ranking in enumerate(rankings):
            for raw in ranking:
                action = int(raw)
                if not 0 <= action <= self.top_k: raise ValueError('ranking action out of range')
                if self._mask[i, action]: actions[i] = action; break
        return self.step(actions)

    def close(self):
        """Release only this environment's active services, retaining audit state."""
        for rid in list(getattr(self, 'active_plans', {})):
            self.ledger.release(rid)
        if self._has_reset:
            self.active_plans.clear(); self.node_queues.clear(); self.link_queues.clear()
            self._terminated = True
