"""Persistent frozen-checkpoint HRL inference for runtime SFC requests."""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import random
import threading
import time
from types import SimpleNamespace
from types import MethodType
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import networkx as nx

from scripts.export_hrl_sfc_plans import (
    convert_plan,
    load_jsonl,
    load_legacy_evaluator,
    resource_ledger_snapshot,
    resolved,
    sha256,
    validate_request_alignment,
)
from core.marl.batch_deployment_wqmix import ResourceSnapshot
from core.marl.deployment_topk import CompletePlanCandidateGenerator


class _LegacyDecisionTimingProbe:
    """Collect granular timings when the loaded legacy coordinator lacks them."""

    def __init__(self, env: Any, agent: Any) -> None:
        self.env = env
        self.rows: dict[str, list[dict[str, Any]]] = {"high": [], "low": []}
        self.depth = {"high": 0, "low": 0}
        self._wrap(getattr(agent, "high_policy", None), "select_goal", "high", "policy")
        self._wrap(getattr(agent, "low_policy", None), "select_action", "low", "policy")

    def _wrap(self, owner: Any, method_name: str, level: str, source: str) -> None:
        if owner is None or not callable(getattr(owner, method_name, None)):
            return
        original = getattr(owner, method_name)
        probe = self

        def measured(_owner, *args, **kwargs):
            outermost = probe.depth[level] == 0
            started_ns = time.perf_counter_ns() if outermost else 0
            phase = getattr(probe.env, "current_phase", "unknown")
            probe.depth[level] += 1
            try:
                return original(*args, **kwargs)
            finally:
                probe.depth[level] -= 1
                if outermost:
                    probe.rows[level].append({
                        "duration_ms": (
                            time.perf_counter_ns() - started_ns
                        ) / 1_000_000.0,
                        "phase": str(phase or "unknown"),
                        "source": source,
                    })

        setattr(owner, method_name, MethodType(measured, owner))

    def reset(self) -> None:
        self.rows = {"high": [], "low": []}
        self.depth = {"high": 0, "low": 0}

    @staticmethod
    def _summary(rows: list[dict[str, Any]], phase: str | None = None) -> dict[str, Any]:
        values = np.asarray([
            float(row["duration_ms"])
            for row in rows
            if phase is None or row.get("phase") == phase
        ], dtype=np.float64)
        if values.size == 0:
            return {
                "count": 0, "total_ms": 0.0, "mean_ms": None,
                "p50_ms": None, "p95_ms": None, "max_ms": None,
            }
        return {
            "count": int(values.size),
            "total_ms": float(values.sum()),
            "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(values.max()),
        }

    def summary(self, request_total_ms: float) -> dict[str, Any]:
        high_rows, low_rows = self.rows["high"], self.rows["low"]
        high, low = self._summary(high_rows), self._summary(low_rows)
        decision_total = float(high["total_ms"] + low["total_ms"])
        return {
            "scope": "HRL algorithm only; excludes Mininet/Ryu/VNF execution",
            "source": "legacy_policy_call_probe",
            "high_level": high,
            "high_vnf_placement": self._summary(high_rows, "vnf_deployment"),
            "high_destination_connection": self._summary(
                high_rows, "destination_connection"
            ),
            "low_level": low,
            "low_vnf_routing": self._summary(low_rows, "vnf_deployment"),
            "low_destination_routing": self._summary(
                low_rows, "destination_connection"
            ),
            "decision_total_ms": decision_total,
            "request_total_ms": float(request_total_ms),
            "non_action_algorithm_ms": max(
                0.0, float(request_total_ms) - decision_total
            ),
        }


class OnlineLegacyHRLPlanner:
    """Load one legacy HRL checkpoint and infer one request at a time.

    The legacy environment owns the authoritative simulated resource ledger.
    Calls are serialized because each decision advances that shared lifecycle.
    Model weights remain frozen for the lifetime of the planner.
    """

    def __init__(
        self,
        *,
        legacy_root: str | Path,
        checkpoint: str | Path,
        data_path: str | Path,
        runtime_requests: str | Path,
        profile: str | Path,
        seed: int,
        max_steps: int = 600,
        bw_cap: float = 90.0,
        cpu_cap: float = 55.0,
        mem_cap: float = 45.0,
        stage_port_base: int = 20000,
        safe_dest_recovery: bool = False,
        planner_destinations: bool = False,
        reseed_per_request: bool = False,
        torch_threads: int = 1,
        quiet: bool = True,
        optimize_static_topology: bool = True,
        k_path_candidate_filter: bool = False,
        k_path_candidate_k: int = 4,
        macro_path_rollout: bool = False,
        skip_high_topk: bool = False,
        failure_step_budget: int | None = None,
        async_prefetch_workers: int = 0,
        fast_k_path_candidates: bool = False,
        completion_candidate_budget: int = 0,
        destination_beam_width: int = 64,
        collect_timing: bool = True,
    ) -> None:
        started = time.perf_counter()
        if int(torch_threads) > 0:
            torch.set_num_threads(int(torch_threads))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # PyTorch permits setting inter-op threads only before its
                # first parallel region; intra-op control still takes effect.
                pass
        self.legacy_root = resolved(Path(legacy_root))
        self.checkpoint = resolved(Path(checkpoint))
        self.data_path = resolved(Path(data_path))
        self.runtime_path = resolved(Path(runtime_requests))
        self.profile_path = resolved(Path(profile))
        for path in (
            self.legacy_root,
            self.checkpoint,
            self.data_path,
            self.runtime_path,
            self.profile_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.stage_port_base = int(stage_port_base)
        self.reseed_per_request = bool(reseed_per_request)
        self.torch_threads = int(torch_threads)
        self.profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        # A live policy can occasionally emit a disconnected multicast tree.
        # Keep a deterministic complete-plan repair path available so one bad
        # conversion cannot terminate the real-time deployment stream.
        self._complete_plan_repair = CompletePlanCandidateGenerator(
            self.profile, max_candidates=1, placement_beam=4,
            placement_chains=4, pool_limit=8,
        )
        self._complete_plan_repair.prewarm_paths(max_paths=4)
        runtime_rows = load_jsonl(self.runtime_path)
        self.original_by_id = {int(row["id"]): row for row in runtime_rows}
        self.request_order = [int(row["id"]) for row in runtime_rows]
        self.next_index = 0
        self._lock = threading.Lock()
        self.failure_step_budget = (
            int(failure_step_budget) if failure_step_budget and int(failure_step_budget) > 0 else None
        )
        self.async_prefetch_workers = max(0, int(async_prefetch_workers))
        self.fast_k_path_candidates = bool(fast_k_path_candidates)
        self.completion_candidate_budget = max(
            0, int(completion_candidate_budget)
        )
        self.destination_beam_width = max(1, int(destination_beam_width))
        self.collect_timing = bool(collect_timing)
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_futures: dict[int, Future] = {}
        self._prefetch_lock = threading.Lock()

        evaluator = load_legacy_evaluator(self.legacy_root)
        if quiet:
            for logger_name in (
                "core",
                "envs",
                "evaluate_checkpoint",
                "train_tahrl",
                "utils",
            ):
                logging.getLogger(logger_name).setLevel(logging.WARNING)
        eval_args = SimpleNamespace(
            config=None,
            topo="us_backbone",
            goal_strategy="adaptive",
            epsilon=0.0,
            bw_cap=float(bw_cap),
            cap_cpu=float(cpu_cap),
            cap_mem=float(mem_cap),
            ablation_variant="full",
            ablation_hop=False,
            ablation_reach=False,
            zero_candidate_feats=False,
            minimal_mlp_state=False,
            checkpoint=self.checkpoint,
            model_mode="eval",
            start_episode=0,
        )
        self.env, self.agent, self.coordinator = evaluator.build_eval_stack(
            eval_args, self.seed, self.data_path
        )
        validate_request_alignment(
            getattr(self.env, "all_requests", []), runtime_rows
        )
        self.env._k_path_candidate_filter = bool(k_path_candidate_filter)
        self.env._k_path_candidate_k = max(1, int(k_path_candidate_k))
        self.env._macro_path_rollout = bool(macro_path_rollout)
        self.env._online_skip_high_topk = bool(skip_high_topk)
        self.env._online_completion_candidate_budget = (
            self.completion_candidate_budget
        )
        self.env._online_destination_beam_width = self.destination_beam_width
        if self.env._k_path_candidate_filter:
            self._install_k_path_candidate_filter(self.env._k_path_candidate_k)
            if self.env._macro_path_rollout:
                self._install_macro_path_rollout()
        self._install_online_feature_cache()
        self._install_fast_resource_reads()
        if skip_high_topk:
            shared = getattr(self.coordinator, "shared", None)
            if shared is not None and hasattr(shared, "apply_high_topk_mask"):
                shared.apply_high_topk_mask = MethodType(
                    lambda _shared, high_mask, *args, **kwargs: high_mask,
                    shared,
                )
        self._install_inference_only_shortcuts()
        self._runtime_timing_stats: dict[str, dict[str, float | int]] = {}
        if self.collect_timing:
            self._install_runtime_timing_probe()
        self._native_decision_timing = hasattr(
            self.coordinator, "_decision_timing_summary"
        )
        self._decision_timing_probe = (
            None
            if self._native_decision_timing or not self.collect_timing
            else _LegacyDecisionTimingProbe(self.env, self.agent)
        )
        self.static_topology_cache_enabled = bool(optimize_static_topology)
        if self.static_topology_cache_enabled:
            self._install_static_topology_cache()
        self.agent.eval()
        self.env._safe_dest_recovery_enabled = bool(safe_dest_recovery)
        self.env._planner_destinations_enabled = bool(planner_destinations)
        self.initialization_ms = (time.perf_counter() - started) * 1000.0
        self.quiet = bool(quiet)
        if self.async_prefetch_workers:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=self.async_prefetch_workers,
                thread_name_prefix="hrl-prefetch",
            )

    def _install_online_feature_cache(self) -> None:
        """Cache expensive snapshot features with strict ledger invalidation."""
        manager = self.env.resource_mgr
        pool = manager.pool
        original_reach = manager.get_reach_feats
        planner = self
        self._bw_snapshot_version = 0
        self._resource_snapshot_version = 0
        self._reach_feature_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        self._high_score_cache: dict[tuple[Any, ...], list[int]] = {}
        self._online_feature_cache_stats = {"reach_hits": 0, "reach_misses": 0}

        def invalidate_after(method):
            def wrapped(_pool, *args, **kwargs):
                result = method(*args, **kwargs)
                # The wrapped operation is the only writer used by the
                # planner.  Comparing a full 28-node/90-link ledger before
                # and after every write costs more than the cache lookup it
                # protects.  Invalidate conservatively after every write;
                # failed allocations only cause a harmless cache miss.
                planner._bw_snapshot_version += 1
                planner._resource_snapshot_version += 1
                planner._reach_feature_cache.clear()
                planner._high_score_cache.clear()
                return result
            return wrapped

        def invalidate_resource_after(method):
            def wrapped(_pool, *args, **kwargs):
                result = method(*args, **kwargs)
                planner._resource_snapshot_version += 1
                planner._reach_feature_cache.clear()
                planner._high_score_cache.clear()
                return result
            return wrapped

        for method_name in (
            "allocate_bandwidth",
            "release_bandwidth",
            "reserve_bandwidth",
            "cancel_link_reservation",
            "reset",
        ):
            method = getattr(pool, method_name, None)
            if callable(method):
                setattr(
                    pool,
                    method_name,
                    MethodType(invalidate_after(method), pool),
                )

        for method_name in (
            "allocate_cpu",
            "release_cpu",
            "allocate_memory",
            "release_memory",
            "reserve_cpu",
            "cancel_reservation",
            "reserve_memory",
            "reset",
        ):
            method = getattr(pool, method_name, None)
            if callable(method) and method_name not in {
                "reset",
            }:
                setattr(
                    pool,
                    method_name,
                    MethodType(invalidate_resource_after(method), pool),
                )

        def cached_reach(_manager, remaining_dests):
            key = (
                int(planner._bw_snapshot_version),
                tuple(sorted(int(value) for value in remaining_dests)),
            )
            cached = planner._reach_feature_cache.get(key)
            if cached is not None:
                planner._online_feature_cache_stats["reach_hits"] += 1
                return cached
            planner._online_feature_cache_stats["reach_misses"] += 1
            value = original_reach(remaining_dests)
            planner._reach_feature_cache[key] = value
            return value

        manager.get_reach_feats = MethodType(cached_reach, manager)

    def _install_fast_resource_reads(self) -> None:
        """Use lock-free reads for the planner's serialized state snapshots.

        The legacy resource pool protects every scalar read with an RLock.
        Planning is serialized by ``self._lock`` and all mutations still use
        the original locked allocation/release methods, so these read-only
        accessors can read the backing arrays/dict directly.  This removes
        thousands of Python lock acquisitions from low-level state and mask
        construction without changing any resource or admission semantics.
        """
        pool = getattr(getattr(self.env, "resource_mgr", None), "pool", None)
        if pool is None or getattr(pool, "_online_fast_reads", False):
            return

        def fast_cpu(_pool, node):
            try:
                return max(0.0, float(_pool.cpu_avail[int(node)]))
            except (IndexError, KeyError, TypeError, ValueError):
                return 0.0

        def fast_memory(_pool, node):
            try:
                return max(0.0, float(_pool.mem_avail[int(node)]))
            except (IndexError, KeyError, TypeError, ValueError):
                return 0.0

        def fast_bandwidth(_pool, u, v):
            try:
                key_fn = getattr(_pool, "_bw_key", None)
                key = key_fn(int(u), int(v)) if callable(key_fn) else (int(u), int(v))
                return max(0.0, float(_pool.bw_avail.get(key, 0.0)))
            except (KeyError, TypeError, ValueError):
                return 0.0

        pool.get_available_cpu = MethodType(fast_cpu, pool)
        pool.get_available_memory = MethodType(fast_memory, pool)
        pool.get_available_bandwidth = MethodType(fast_bandwidth, pool)
        pool._online_fast_reads = True
        self.fast_resource_reads_enabled = True

    def _install_inference_only_shortcuts(self) -> None:
        """Remove values constructed by legacy APIs but ignored in eval mode."""
        controller = self.env.high_level_controller
        original_set_goal = controller.set_high_level_goal

        def set_goal_without_unused_graph(_controller, *args, **kwargs):
            original_graph_builder = _controller.get_high_level_state_graph
            _controller.get_high_level_state_graph = lambda: None
            try:
                original_set_goal(*args, **kwargs)
            finally:
                _controller.get_high_level_state_graph = original_graph_builder
            return None

        controller.set_high_level_goal = MethodType(
            set_goal_without_unused_graph, controller
        )

        low_controller = self.env.low_level_controller
        original_low_state = low_controller.get_state

        def low_state_without_embedded_mask(_controller, *args, **kwargs):
            original_mask = _controller.get_low_level_action_mask
            _controller.get_low_level_action_mask = lambda: np.ones(
                int(self.env.n), dtype=np.float32
            )
            try:
                return original_low_state(*args, **kwargs)
            finally:
                _controller.get_low_level_action_mask = original_mask

        low_controller.get_state = MethodType(
            low_state_without_embedded_mask, low_controller
        )

        encoder = getattr(self.coordinator.high_agent, "encoder", None)
        original_graph_emb = self.coordinator._get_high_graph_emb
        if encoder is None:
            return
        original_forward = encoder.forward
        planner = self
        self._last_high_encoder_key = None
        self._last_high_encoder_output = None

        def cached_forward(_encoder, x, *args, **kwargs):
            output = original_forward(x, *args, **kwargs)
            planner._last_high_encoder_key = (
                int(x.data_ptr()) if hasattr(x, "data_ptr") else id(x),
                tuple(x.shape) if hasattr(x, "shape") else None,
            )
            planner._last_high_encoder_output = output
            return output

        def graph_emb_from_cached_nodes(_coordinator, high_obs):
            cached = planner._last_high_encoder_output
            if cached is not None:
                return cached.mean(0, keepdim=True)
            return original_graph_emb(high_obs)

        encoder.forward = MethodType(cached_forward, encoder)
        self.coordinator._get_high_graph_emb = MethodType(
            graph_emb_from_cached_nodes, self.coordinator
        )

        original_cycle = self.coordinator.run_high_low_cycle

        def cycle_with_fresh_high_cache(_coordinator, *args, **kwargs):
            planner._last_high_encoder_key = None
            planner._last_high_encoder_output = None
            return original_cycle(*args, **kwargs)

        self.coordinator.run_high_low_cycle = MethodType(
            cycle_with_fresh_high_cache, self.coordinator
        )

    def _install_runtime_timing_probe(self) -> None:
        targets = (
            (self.env, "reset", "env_reset"),
            (self.env, "get_state", "env_get_state"),
            (self.env, "get_high_level_state_graph", "env_get_high_state"),
            (self.env, "get_low_level_action_mask", "env_low_mask"),
            (self.env, "get_high_level_action_mask", "env_high_mask"),
            (self.env, "step_low_level", "env_step_low"),
            (
                self.env.low_level_controller,
                "get_state",
                "low_controller_state",
            ),
            (
                self.env.low_level_controller,
                "get_low_level_candidates",
                "low_candidates",
            ),
            (self.coordinator, "_encode_low_state_for_policy", "encode_low_state"),
            (self.coordinator, "_get_high_graph_emb", "encode_high_state"),
            (self.coordinator, "run_high_low_cycle", "high_low_cycle"),
            (self.env.resource_mgr, "get_reach_feats", "reach_features"),
            (
                self.env.resource_mgr,
                "build_dynamic_edge_attr",
                "dynamic_edge_attr",
            ),
        )
        planner = self
        for owner, method_name, label in targets:
            original = getattr(owner, method_name, None)
            if not callable(original):
                continue

            def measured(_owner, *args, __original=original, __label=label, **kwargs):
                started_ns = time.perf_counter_ns()
                try:
                    return __original(*args, **kwargs)
                finally:
                    row = planner._runtime_timing_stats.setdefault(
                        __label, {"count": 0, "total_ms": 0.0}
                    )
                    row["count"] = int(row["count"]) + 1
                    row["total_ms"] = float(row["total_ms"]) + (
                        time.perf_counter_ns() - started_ns
                    ) / 1_000_000.0

            setattr(owner, method_name, MethodType(measured, owner))

    def _active_low_target(self) -> int | None:
        phase = getattr(self.env, "current_phase", None)
        if phase == "vnf_deployment":
            value = getattr(self.env, "current_deployment_target", None)
        elif phase == "destination_connection":
            value = getattr(self.env, "current_target_node", None)
        else:
            value = None
        return int(value) if value is not None else None

    def _install_k_path_candidate_filter(self, k: int) -> None:
        """Wrap the loaded legacy low controller with Top-K path pruning.

        The legacy evaluator is imported from ``legacy_root`` and may not
        contain the workspace helper implementation.  Installing the wrapper
        on the live controller guarantees that the online experiment actually
        exercises K-path candidate pruning while preserving the frozen low
        policy as the final action selector.
        """
        controller = getattr(self.env, "low_level_controller", None)
        original = getattr(controller, "get_low_level_candidates", None)
        if controller is None or not callable(original):
            raise RuntimeError("legacy low-level controller has no candidate interface")

        topology = np.asarray(self.env.resource_mgr.topo)
        graph = nx.Graph()
        rows, cols = np.where(topology > 0)
        graph.add_edges_from((int(u), int(v)) for u, v in zip(rows, cols) if u != v)
        cached_paths: dict[tuple[int, int], list[list[int]]] = {}
        nodes = sorted(int(node) for node in graph.nodes)
        from itertools import islice
        for source in nodes:
            for target in nodes:
                if source == target:
                    cached_paths[(source, target)] = [[source]]
                    continue
                try:
                    cached_paths[(source, target)] = [
                        [int(value) for value in path]
                        for path in islice(
                            nx.shortest_simple_paths(graph, source, target), k
                        )
                    ]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    cached_paths[(source, target)] = []

        planner = self
        self._k_path_cache = cached_paths
        self._macro_paths_by_first_hop: dict[int, list[int]] = {}
        self._last_low_mask = None
        self._last_low_mask_signature = None
        original_env_mask = getattr(self.env, "get_low_level_action_mask", None)
        original_controller_mask = getattr(controller, "get_low_level_action_mask", None)

        def low_mask_signature():
            tree = getattr(planner.env, "current_tree", None) or {}
            tree_data = tree.get("tree", {}) if isinstance(tree, dict) else {}
            connected = tree.get("connected_dests", set()) if isinstance(tree, dict) else set()
            return (
                str(getattr(planner.env, "current_phase", "")),
                getattr(planner.env, "current_node_location", None),
                planner._active_low_target(),
                int(getattr(planner.env, "next_vnf_idx", 0)),
                len(connected),
                sum(1 for flow in tree_data.values() if float(flow) > 0.0)
                if isinstance(tree_data, dict) else 0,
                len(getattr(planner.env, "current_path_trace", []) or []),
            )

        def remember_mask(value):
            planner._last_low_mask = np.asarray(value, dtype=np.float32).copy()
            planner._last_low_mask_signature = low_mask_signature()
            return value

        if callable(original_controller_mask):
            def cached_controller_mask(_controller, *args, **kwargs):
                mutate_env = kwargs.get("mutate_env", True)
                if (
                    mutate_env is False
                    and planner._last_low_mask is not None
                    and planner._last_low_mask_signature == low_mask_signature()
                ):
                    return planner._last_low_mask.copy()
                value = original_controller_mask(*args, **kwargs)
                return remember_mask(value)
            controller.get_low_level_action_mask = MethodType(
                cached_controller_mask, controller
            )

        if callable(original_env_mask):
            def remember_low_mask(_env, *args, **kwargs):
                value = original_env_mask(*args, **kwargs)
                return remember_mask(value)
            self.env.get_low_level_action_mask = MethodType(
                remember_low_mask, self.env
            )

        def fast_features(current, candidates, target, bw_req, tree_edges):
            nodes_on_tree = set(getattr(planner.env, "nodes_on_tree", set()))
            tabu_set = set(getattr(planner.env, "current_path_trace", []))
            request = getattr(planner.env, "current_request", None) or {}
            connected = planner.coordinator.shared.get_connected_dests_view()
            undone = [
                int(value) for value in request.get("dest", [])
                if int(value) not in connected
            ]
            max_hops = max(int(getattr(planner.env, "n", 1)), 1)
            def hop(first, second):
                if first is None or second is None:
                    return 0
                paths = cached_paths.get((int(first), int(second)), [])
                return len(paths[0]) - 1 if paths else 9999
            features = np.zeros((len(candidates), 6), dtype=np.float32)
            pool = planner.env.resource_mgr.pool
            for index, node in enumerate(candidates):
                d_current = hop(current, target) if target is not None else 0
                d_next = hop(node, target) if target is not None else 0
                if d_current < 9999 and d_next < 9999:
                    features[index, 0] = float(d_current - d_next) / max_hops
                edge = (int(current), int(node))
                reverse = (int(node), int(current))
                features[index, 1] = 1.0 if edge in tree_edges else 0.5 if reverse in tree_edges else 0.0
                try:
                    available = float(pool.get_available_bandwidth(*edge))
                    features[index, 2] = min(available / max(2.0 * bw_req, 1.0), 1.0)
                except Exception:
                    features[index, 2] = 0.0
                features[index, 3] = float(int(node) in nodes_on_tree)
                if undone:
                    hops = [hop(node, dest) for dest in undone]
                    valid = [value for value in hops if value < 9999]
                    features[index, 4] = (
                        sum(valid) / len(valid) / max_hops if valid else 1.0
                    )
                features[index, 5] = float(int(node) in tabu_set)
            return features

        def filtered_candidates(_controller, *args, **kwargs):
            macro_queue = getattr(planner, "_macro_queue", None)
            current_now = getattr(planner.env, "current_node_location", None)
            target_now = planner._active_low_target()
            expected_now = getattr(planner, "_macro_expected_current", None)
            macro_target = getattr(planner, "_macro_target", None)
            terminal_pending = bool(
                getattr(planner, "_macro_terminal_pending", False)
                and current_now is not None
                and target_now is not None
                and int(current_now) == int(target_now) == macro_target
            )
            queue_valid = bool(
                macro_queue
                and current_now is not None
                and int(current_now) == expected_now
                and target_now == macro_target
            )
            if queue_valid or terminal_pending:
                selected = int(macro_queue[0]) if queue_valid else int(current_now)
                result = {
                    "indices": [selected],
                    "features": None,
                    "mask": np.ones(1, dtype=np.float32),
                    "current_node": int(current_now),
                    "target_node": int(target_now),
                    "candidate_mode": "macro_suffix",
                    "candidate_k": int(k),
                }
                stats = getattr(planner.env, "_k_path_candidate_stats", None)
                if not isinstance(stats, dict):
                    stats = {
                        "calls": 0, "raw_candidates": 0,
                        "filtered_candidates": 0, "reduced_calls": 0,
                        "empty_fallback_calls": 0,
                    }
                    planner.env._k_path_candidate_stats = stats
                stats["calls"] += 1
                stats["raw_candidates"] += 1
                stats["filtered_candidates"] += 1
                return result

            # The coordinator has just computed the authoritative low-level
            # mask.  Reuse it and derive candidates from the prewarmed K-path
            # table instead of rebuilding the legacy NetworkX candidate graph.
            current_signature = (
                str(getattr(planner.env, "current_phase", "")),
                current_now,
                target_now,
                int(getattr(planner.env, "next_vnf_idx", 0)),
                len(
                    (getattr(planner.env, "current_tree", None) or {}).get(
                        "connected_dests", set()
                    )
                ),
                sum(
                    1
                    for flow in (
                        (getattr(planner.env, "current_tree", None) or {})
                    ).get("tree", {}).values()
                    if float(flow) > 0.0
                ),
                len(getattr(planner.env, "current_path_trace", []) or []),
            )
            mask = (
                planner._last_low_mask
                if planner._last_low_mask_signature == current_signature
                else None
            )
            if (
                self.fast_k_path_candidates
                and mask is not None
                and current_now is not None
                and target_now is not None
            ):
                indices = [int(value) for value in np.where(mask > 0)[0]]
                tree = (getattr(planner.env, "current_tree", None) or {}).get("tree", {})
                tree_edges = {
                    (int(edge[0]), int(edge[1]))
                    for edge, flow in tree.items() if float(flow) > 0.0
                }
                request = getattr(planner.env, "current_request", None) or {}
                bw_req = float(request.get("bw_origin", request.get("bw", 0.0)))
                feasible_by_first_hop: dict[int, list[int]] = {}
                for path in cached_paths.get((int(current_now), int(target_now)), []):
                    if len(path) <= 1:
                        continue
                    feasible = True
                    for u, v in zip(path, path[1:]):
                        if (int(u), int(v)) in tree_edges:
                            continue
                        if float(planner.env.resource_mgr.pool.get_available_bandwidth(int(u), int(v))) + 1e-9 < bw_req:
                            feasible = False
                            break
                    if feasible:
                        hop = int(path[1])
                        previous = feasible_by_first_hop.get(hop)
                        if previous is None or len(path) < len(previous):
                            feasible_by_first_hop[hop] = [int(value) for value in path]
                allowed = set(feasible_by_first_hop)
                allowed.update(
                    node for node in indices
                    if (int(current_now), int(node)) in tree_edges
                )
                selected = [node for node in indices if node in allowed]
                if selected:
                    planner._macro_paths_by_first_hop = {
                        node: feasible_by_first_hop[node]
                        for node in selected if node in feasible_by_first_hop
                    }
                    for node in selected:
                        if node in planner._macro_paths_by_first_hop:
                            continue
                        for suffix in cached_paths.get((node, int(target_now)), []):
                            if len(suffix) > 1:
                                planner._macro_paths_by_first_hop[node] = [
                                    int(current_now), *[int(value) for value in suffix]
                                ]
                                break
                    stats = getattr(planner.env, "_k_path_candidate_stats", None)
                    if not isinstance(stats, dict):
                        stats = {
                            "calls": 0, "raw_candidates": 0,
                            "filtered_candidates": 0, "reduced_calls": 0,
                            "empty_fallback_calls": 0,
                        }
                        planner.env._k_path_candidate_stats = stats
                    stats["calls"] += 1
                    stats["raw_candidates"] += len(indices)
                    stats["filtered_candidates"] += len(selected)
                    stats["reduced_calls"] += int(len(selected) < len(indices))
                    return {
                        "indices": selected,
                        "features": fast_features(
                            int(current_now), selected, int(target_now), bw_req, tree_edges
                        ),
                        "mask": np.ones(len(selected), dtype=np.float32),
                        "current_node": int(current_now),
                        "target_node": int(target_now),
                        "candidate_mode": "k_path_fast",
                        "candidate_k": int(k),
                    }

            result = original(*args, **kwargs)
            indices = [int(value) for value in result.get("indices", [])]
            raw_count = len(indices)
            current = result.get("current_node")
            target = result.get("target_node")
            selected_positions = list(range(raw_count))
            if raw_count and current is not None and target is not None and int(current) != int(target):
                tree = (getattr(planner.env, "current_tree", None) or {}).get("tree", {})
                tree_edges = {
                    (int(edge[0]), int(edge[1]))
                    for edge, flow in tree.items() if float(flow) > 0.0
                }
                request = getattr(planner.env, "current_request", None) or {}
                bw_req = float(request.get("bw_origin", request.get("bw", 0.0)))
                pool = planner.env.resource_mgr.pool
                path_hops: set[int] = set()
                feasible_by_first_hop: dict[int, list[int]] = {}
                for path in cached_paths.get((int(current), int(target)), []):
                    feasible = True
                    for u, v in zip(path, path[1:]):
                        if (u, v) in tree_edges:
                            continue
                        if float(pool.get_available_bandwidth(u, v)) + 1e-9 < bw_req:
                            feasible = False
                            break
                    if feasible and len(path) > 1:
                        first_hop = int(path[1])
                        path_hops.add(first_hop)
                        previous = feasible_by_first_hop.get(first_hop)
                        if previous is None or len(path) < len(previous):
                            feasible_by_first_hop[first_hop] = list(path)
                tree_hops = {
                    node for node in indices if (int(current), node) in tree_edges
                }
                allowed = path_hops | tree_hops
                positions = [pos for pos, node in enumerate(indices) if node in allowed]
                if positions:
                    selected_positions = positions
                    result = dict(result)
                    result["indices"] = [indices[pos] for pos in positions]
                    features = result.get("features")
                    if features is not None:
                        result["features"] = np.asarray(features)[positions]
                    result["mask"] = np.ones(len(positions), dtype=np.float32)
                    planner._macro_paths_by_first_hop = {
                        hop: feasible_by_first_hop[hop]
                        for hop in result["indices"]
                        if hop in feasible_by_first_hop
                    }
                    for hop in result["indices"]:
                        hop = int(hop)
                        if hop in planner._macro_paths_by_first_hop:
                            continue
                        for suffix in cached_paths.get((hop, int(target)), []):
                            if len(suffix) <= 1:
                                continue
                            feasible = True
                            for u, v in zip(suffix, suffix[1:]):
                                if (int(u), int(v)) in tree_edges:
                                    continue
                                if float(pool.get_available_bandwidth(int(u), int(v))) + 1e-9 < bw_req:
                                    feasible = False
                                    break
                            if feasible:
                                planner._macro_paths_by_first_hop[hop] = [
                                    int(current), *[int(node) for node in suffix]
                                ]
                                break
                else:
                    planner._macro_paths_by_first_hop = {}

            stats = getattr(planner.env, "_k_path_candidate_stats", None)
            if not isinstance(stats, dict):
                stats = {
                    "calls": 0, "raw_candidates": 0,
                    "filtered_candidates": 0, "reduced_calls": 0,
                    "empty_fallback_calls": 0,
                }
                planner.env._k_path_candidate_stats = stats
            filtered_count = len(result.get("indices", []))
            stats["calls"] += 1
            stats["raw_candidates"] += raw_count
            stats["filtered_candidates"] += filtered_count
            stats["reduced_calls"] += int(filtered_count < raw_count)
            stats["empty_fallback_calls"] += int(raw_count > 0 and not selected_positions)
            result["candidate_mode"] = "k_path_filtered"
            result["candidate_k"] = int(k)
            return result

        controller.get_low_level_candidates = MethodType(
            filtered_candidates, controller
        )
        self.k_path_pairs_prewarmed = len(cached_paths)

    def _install_macro_path_rollout(self) -> None:
        """Use one frozen low-policy decision for an entire candidate path.

        Every edge is still applied through the legacy low-level controller,
        preserving its bandwidth, cycle, tree, VNF, destination and rollback
        semantics.  Only redundant state construction and policy calls on the
        already-selected path suffix are skipped.
        """
        controller = getattr(self.env, "low_level_controller", None)
        low_policy = getattr(self.agent, "low_policy", None)
        if controller is None or low_policy is None:
            raise RuntimeError("macro path rollout requires low controller and policy")
        original_select = low_policy.select_action
        original_get_state = controller.get_state
        original_mask = controller.get_low_level_action_mask
        original_encode_low = self.coordinator._encode_low_state_for_policy
        planner = self
        self._macro_queue: list[int] = []
        self._macro_expected_current: int | None = None
        self._macro_target: int | None = None
        self._macro_cached_state = None
        self._macro_skip_state_once = False
        self._macro_terminal_pending = False
        self._macro_cached_encoding = None
        self._macro_rollout_stats = {
            "rl_path_decisions": 0,
            "macro_hops": 0,
            "skipped_state_builds": 0,
            "reused_low_encodings": 0,
            "rebuilt_low_encodings": 0,
            "invalidated_paths": 0,
        }

        def queue_is_valid() -> bool:
            current = getattr(planner.env, "current_node_location", None)
            target = planner._active_low_target()
            valid = bool(
                planner._macro_queue
                and current is not None
                and int(current) == planner._macro_expected_current
                and target == planner._macro_target
            )
            if planner._macro_queue and not valid:
                planner._macro_rollout_stats["invalidated_paths"] += 1
                planner._macro_queue = []
                planner._macro_expected_current = None
                planner._macro_target = None
                planner._macro_terminal_pending = False
            return valid

        def fast_get_state(_controller, *args, **kwargs):
            if planner._macro_skip_state_once and planner._macro_cached_state is not None:
                planner._macro_skip_state_once = False
                planner._macro_rollout_stats["skipped_state_builds"] += 1
                return planner._macro_cached_state
            state = original_get_state(*args, **kwargs)
            planner._macro_cached_state = state
            return state

        def macro_encode_low(_coordinator, low_state):
            current = getattr(planner.env, "current_node_location", None)
            target = planner._active_low_target()
            terminal_pending = bool(
                planner._macro_terminal_pending
                and current is not None
                and target is not None
                and int(current) == int(target) == planner._macro_target
            )
            if (queue_is_valid() or terminal_pending) and planner._macro_cached_encoding is not None:
                planner._macro_rollout_stats["reused_low_encodings"] += 1
                return planner._macro_cached_encoding
            encoded = original_encode_low(low_state)
            planner._macro_rollout_stats["rebuilt_low_encodings"] += 1
            planner._macro_cached_encoding = encoded
            return encoded

        def macro_mask(_controller, *args, **kwargs):
            current = getattr(planner.env, "current_node_location", None)
            target = planner._active_low_target()
            if (
                planner._macro_terminal_pending
                and current is not None and target is not None
                and int(current) == int(target) == planner._macro_target
            ):
                mask = np.zeros(int(planner.env.n), dtype=np.float32)
                mask[int(current)] = 1.0
                return mask
            if queue_is_valid():
                node = int(planner._macro_queue[0])
                mask = np.zeros(int(planner.env.n), dtype=np.float32)
                mask[node] = 1.0
                return mask
            return original_mask(*args, **kwargs)

        def macro_select(_policy, *args, **kwargs):
            current = getattr(planner.env, "current_node_location", None)
            target = planner._active_low_target()
            if (
                planner._macro_terminal_pending
                and current is not None and target is not None
                and int(current) == int(target) == planner._macro_target
            ):
                planner._macro_terminal_pending = False
                planner._macro_expected_current = None
                # The returned next state is not consumed after a successful
                # deploy/connect action during inference.
                planner._macro_skip_state_once = True
                device = getattr(_policy, "device", torch.device("cpu"))
                planner._macro_rollout_stats["macro_hops"] += 1
                return torch.tensor(int(current), dtype=torch.long, device=device), torch.zeros(
                    (1, 1), dtype=torch.float32, device=device
                )
            if queue_is_valid():
                action = int(planner._macro_queue.pop(0))
                planner._macro_expected_current = action
                # The selected path fixes every suffix action, including the
                # terminal stay, so none of these states needs re-encoding.
                planner._macro_skip_state_once = True
                planner._macro_terminal_pending = not planner._macro_queue
                planner._macro_rollout_stats["macro_hops"] += 1
                device = getattr(_policy, "device", torch.device("cpu"))
                return torch.tensor(action, dtype=torch.long, device=device), torch.zeros(
                    (1, 1), dtype=torch.float32, device=device
                )

            action, value = original_select(*args, **kwargs)
            chosen = int(action.item()) if hasattr(action, "item") else int(action)
            current = getattr(planner.env, "current_node_location", None)
            target = planner._active_low_target()
            path = planner._macro_paths_by_first_hop.get(chosen)
            planner._macro_rollout_stats["rl_path_decisions"] += 1
            if (
                path and len(path) > 1 and current is not None
                and int(path[0]) == int(current) and int(path[1]) == chosen
                and target is not None and int(path[-1]) == int(target)
            ):
                planner._macro_queue = [int(node) for node in path[2:]]
                planner._macro_expected_current = chosen
                planner._macro_target = int(target)
                planner._macro_skip_state_once = True
                planner._macro_terminal_pending = not planner._macro_queue
            else:
                planner._macro_queue = []
                planner._macro_expected_current = None
                planner._macro_target = None
                planner._macro_terminal_pending = False
            return action, value

        controller.get_state = MethodType(fast_get_state, controller)
        controller.get_low_level_action_mask = MethodType(macro_mask, controller)
        self.coordinator._encode_low_state_for_policy = MethodType(
            macro_encode_low, self.coordinator
        )
        low_policy.select_action = MethodType(macro_select, low_policy)

    def _install_static_topology_cache(self) -> None:
        resource_manager = self.env.resource_mgr
        topology = np.asarray(resource_manager.topo)
        node_count = int(topology.shape[0])
        neighbors = [
            np.flatnonzero(topology[node] > 0).astype(int).tolist()
            for node in range(node_count)
        ]

        def cached_neighbors(_manager, node: int):
            node = int(node)
            if node < 0 or node >= node_count:
                return []
            return neighbors[node]

        resource_manager.get_neighbors = MethodType(
            cached_neighbors, resource_manager
        )

        infinity = node_count + 1
        distances = np.full((node_count, node_count), infinity, dtype=np.int16)
        np.fill_diagonal(distances, 0)
        distances[topology > 0] = 1
        for intermediate in range(node_count):
            distances = np.minimum(
                distances,
                distances[:, intermediate, None]
                + distances[None, intermediate, :],
            )

        # ``build_high_level_candidates`` only asks for unweighted paths on
        # the immutable topology.  Precompute those paths once so candidate
        # feature construction does not invoke NetworkX BFS for every node
        # and connected destination on every high-level cycle.
        static_graph = nx.Graph()
        static_graph.add_nodes_from(range(node_count))
        rows, cols = np.where(topology > 0)
        static_graph.add_edges_from(
            (int(u), int(v)) for u, v in zip(rows, cols) if int(u) != int(v)
        )
        static_paths: dict[tuple[int, int], list[int]] = {}
        for source in range(node_count):
            try:
                paths = nx.single_source_shortest_path(static_graph, source)
                for target, path in paths.items():
                    static_paths[(int(source), int(target))] = [int(v) for v in path]
            except Exception:
                continue

        def cached_hop(_helper, first: int, second: int) -> int:
            first, second = int(first), int(second)
            if not (0 <= first < node_count and 0 <= second < node_count):
                return 9999
            value = int(distances[first, second])
            return value if value <= node_count else 9999

        def cached_valid_node(_helper, node: Any) -> bool:
            try:
                return 0 <= int(node) < node_count
            except (TypeError, ValueError):
                return False

        owners = [self.env, self.coordinator]
        owners.extend(vars(self.env).values())
        owners.extend(vars(self.coordinator).values())
        patched = set()
        for owner in owners:
            helper = getattr(owner, "shared", None)
            if helper is None or id(helper) in patched:
                continue
            if hasattr(helper, "get_hop_distance_lazy"):
                helper.get_hop_distance_lazy = MethodType(cached_hop, helper)
                helper.is_valid_node = MethodType(cached_valid_node, helper)
                original_graph = getattr(helper, "get_topology_graph", None)
                original_build = getattr(helper, "build_high_level_candidates", None)
                if callable(original_graph) and callable(original_build):
                    def tagged_graph(_helper, *args, __original=original_graph, **kwargs):
                        graph = __original(*args, **kwargs)
                        try:
                            graph.graph["_online_static_topology"] = True
                        except Exception:
                            pass
                        return graph

                    def cached_build(_helper, *args, __original=original_build, **kwargs):
                        original_shortest = nx.shortest_path

                        def static_shortest(graph, source, target, weight=None, *rest, **kw):
                            if weight is None and getattr(graph, "graph", {}).get(
                                "_online_static_topology", False
                            ):
                                path = static_paths.get((int(source), int(target)))
                                if path is not None:
                                    return list(path)
                            return original_shortest(graph, source, target, weight=weight, *rest, **kw)

                        nx.shortest_path = static_shortest
                        try:
                            return __original(*args, **kwargs)
                        finally:
                            nx.shortest_path = original_shortest

                    helper.get_topology_graph = MethodType(tagged_graph, helper)
                    helper.build_high_level_candidates = MethodType(cached_build, helper)
                    self.static_path_cache_patched = True
                patched.add(id(helper))
        self.static_topology_helpers_patched = len(patched)
        self._install_static_graph_reuse()
        self._install_fast_high_score()
        self._install_high_score_cache()

    def _install_fast_high_score(self) -> None:
        """Replace repeated NetworkX weighted shortest-path calls.

        High-level ranking asks for paths from the same source to several
        candidates and remaining destinations.  The legacy implementation
        rebuilds a NetworkX Dijkstra search for every pair.  This equivalent
        implementation builds the current directed weighted adjacency once
        and runs one small heap-based Dijkstra per distinct source.  It keeps
        the same bandwidth/tree edge admission, path cost, reuse counts and
        score equations; only the path-search implementation changes.
        """
        import heapq

        owners = [self.env, self.coordinator]
        owners.extend(vars(self.env).values())
        owners.extend(vars(self.coordinator).values())
        patched = set()
        planner = self

        for owner in owners:
            helper = getattr(owner, "shared", None)
            if helper is None or id(helper) in patched:
                continue
            original = getattr(helper, "score_high_candidates", None)
            if not callable(original):
                continue

            def fast_score(_helper, valid_indices, start_node, __original=original):
                try:
                    rm = getattr(planner.env, "resource_mgr", None)
                    if rm is None:
                        return [int(i) for i in valid_indices]
                    req = getattr(planner.env, "current_request", None) or {}
                    bw_need = float(req.get("bw_origin", 0.0))
                    phase = getattr(planner.env, "current_phase", "idle")
                    n_nodes = max(int(getattr(planner.env, "n", 1)), 1)
                    values = [int(i) for i in valid_indices]
                    all_dests = {int(d) for d in req.get("dest", [])}
                    done_dests = _helper.get_connected_dests_view()
                    undone = sorted(all_dests - {int(d) for d in done_dests})
                    remaining = len(undone)
                    tree_edges = _helper.get_positive_tree_edge_set()
                    pool = rm.pool

                    adjacency = [[] for _ in range(int(getattr(rm, "n", n_nodes)))]
                    for u in range(len(adjacency)):
                        for v in rm.get_neighbors(u):
                            u_i, v_i = int(u), int(v)
                            edge = (u_i, v_i)
                            avail = float(pool.get_available_bandwidth(u_i, v_i))
                            cap = float(getattr(pool, "bw_cap", {}).get(edge, 1.0))
                            if edge not in tree_edges and avail < bw_need:
                                continue
                            util = max(0.0, min(1.0, 1.0 - avail / max(cap, 1.0)))
                            if edge in tree_edges:
                                weight = 0.05 + 0.10 * util
                            elif (v_i, u_i) in tree_edges:
                                weight = 0.80 + 0.60 * util
                            else:
                                weight = 1.25 + 1.25 * util
                            adjacency[u_i].append((v_i, avail, float(weight), edge in tree_edges))

                    def all_stats(source):
                        source = int(source)
                        dist = {source: 0.0}
                        parent = {}
                        heap = [(0.0, source)]
                        while heap:
                            cost, node = heapq.heappop(heap)
                            if cost > dist.get(node, float("inf")) + 1e-12:
                                continue
                            for nxt, _avail, weight, _reused in adjacency[node] if 0 <= node < len(adjacency) else ():
                                new_cost = cost + weight
                                if new_cost < dist.get(nxt, float("inf")) - 1e-12:
                                    dist[nxt] = new_cost
                                    parent[nxt] = (node, _avail, weight, _reused)
                                    heapq.heappush(heap, (new_cost, nxt))

                        result = {}
                        for target in set(values) | set(undone):
                            target = int(target)
                            if target == source:
                                result[target] = {
                                    "reachable": True, "bottleneck": float("inf"),
                                    "hops": 0, "reused": 0, "new_edges": 0, "cost": 0.0,
                                }
                                continue
                            if target not in dist:
                                result[target] = {
                                    "reachable": False, "bottleneck": 0.0,
                                    "hops": 999, "reused": 0, "new_edges": 999,
                                    "cost": float("inf"),
                                }
                                continue
                            node = target
                            hops = 0
                            bottleneck = float("inf")
                            reused = 0
                            while node != source:
                                edge_info = parent.get(node)
                                if edge_info is None:
                                    break
                                prev, avail, _weight, edge_reused = edge_info
                                hops += 1
                                bottleneck = min(bottleneck, float(avail))
                                reused += int(bool(edge_reused))
                                node = prev
                            result[target] = {
                                "reachable": node == source,
                                "bottleneck": bottleneck if node == source else 0.0,
                                "hops": hops if node == source else 999,
                                "reused": reused if node == source else 0,
                                "new_edges": max(0, hops - reused) if node == source else 999,
                                "cost": float(dist[target]) if node == source else float("inf"),
                            }
                        return result

                    source_cache = {}
                    sources = {int(start_node), *values, *undone}
                    for source in sources:
                        source_cache[source] = all_stats(source)

                    def stats(src, dst):
                        if int(src) == int(dst):
                            return {
                                "reachable": True, "bottleneck": float("inf"),
                                "hops": 0, "reused": 0, "new_edges": 0, "cost": 0.0,
                            }
                        return source_cache.get(int(src), {}).get(int(dst), {
                            "reachable": False, "bottleneck": 0.0,
                            "hops": 999, "reused": 0, "new_edges": 999,
                            "cost": float("inf"),
                        })

                    scores = []
                    vnf_list = req.get("vnf", [])
                    vnf_idx = int(getattr(planner.env, "next_vnf_idx", 0))
                    cpu_list = req.get("cpu_origin", req.get("cpu", []))
                    mem_list = req.get("memory_origin", req.get("memory", []))
                    cpu_need = float(cpu_list[vnf_idx]) if vnf_idx < len(cpu_list) else 0.0
                    mem_need = float(mem_list[vnf_idx]) if vnf_idx < len(mem_list) else 0.0
                    for node in values:
                        item = stats(start_node, node)
                        if phase == "destination_connection":
                            ratio = item["bottleneck"] / max(1e-6, bw_need) if bw_need > 0 else 2.0
                            bw_bonus = 2.0 * min(ratio, 2.0)
                            hop_w = 4.0 if remaining <= 1 else 3.0 if remaining <= 2 else 2.0
                            score = (
                                bw_bonus - hop_w * item["hops"]
                                - 1.5 * item["new_edges"] - 0.8 * item["cost"]
                                + 1.2 * item["reused"]
                            )
                        else:
                            cpu_avail = pool.get_available_cpu(node)
                            mem_avail = pool.get_available_memory(node)
                            cpu_slack = (cpu_avail - cpu_need) / max(1.0, cpu_need if cpu_need > 0 else cpu_avail)
                            mem_slack = (mem_avail - mem_need) / max(1.0, mem_need if mem_need > 0 else mem_avail)
                            if not item["reachable"]:
                                score = -1e6 + 0.1 * cpu_slack + 0.1 * mem_slack
                                scores.append((score, node))
                                continue
                            dest_stats = [stats(node, d) for d in undone if int(d) != node]
                            reachable = [value for value in dest_stats if value["reachable"]]
                            if reachable:
                                avg_cost = sum(x["cost"] for x in reachable) / len(reachable)
                                avg_hops = sum(x["hops"] for x in reachable) / len(reachable)
                                avg_new = sum(x["new_edges"] for x in reachable) / len(reachable)
                                dest_reuse = sum(x["reused"] for x in reachable) / max(1.0, sum(x["hops"] for x in reachable))
                                dest_ratio = len(reachable) / max(1, len(dest_stats))
                            elif dest_stats:
                                avg_cost = avg_hops = avg_new = float(n_nodes)
                                dest_reuse = 0.0
                                dest_ratio = 0.0
                            else:
                                avg_cost = avg_hops = avg_new = dest_reuse = 0.0
                                dest_ratio = 1.0
                            future_w = 2.4 if vnf_list and vnf_idx >= len(vnf_list) - 1 else 1.5
                            score = (
                                0.9 * cpu_slack + 0.9 * mem_slack
                                - 2.4 * item["cost"] - 1.4 * item["new_edges"]
                                + 1.8 * item["reused"] - future_w * avg_cost
                                - 0.8 * avg_new - 0.25 * (item["hops"] + avg_hops) / max(1, n_nodes)
                                + 2.0 * dest_reuse + 3.0 * dest_ratio
                            )
                        scores.append((score, node))
                    scores.sort(key=lambda x: x[0], reverse=True)
                    return [node for _, node in scores]
                except Exception:
                    return __original(valid_indices, start_node)

            helper.score_high_candidates = MethodType(fast_score, helper)
            patched.add(id(helper))
        self.fast_high_score_patched = len(patched)

    def _install_high_score_cache(self) -> None:
        """Memoize repeated high-level route scoring within one live snapshot."""
        owners = [self.env, self.coordinator]
        owners.extend(vars(self.env).values())
        owners.extend(vars(self.coordinator).values())
        patched = set()
        self._high_score_cache_stats = {"hits": 0, "misses": 0}
        for owner in owners:
            helper = getattr(owner, "shared", None)
            if helper is None or id(helper) in patched:
                continue
            original = getattr(helper, "score_high_candidates", None)
            if not callable(original):
                continue
            planner = self

            def cached_score(
                _helper,
                valid_indices,
                start_node,
                __original=original,
            ):
                values = tuple(int(value) for value in valid_indices)
                request = getattr(planner.env, "current_request", None) or {}
                tree = getattr(planner.env, "current_tree", None) or {}
                tree_data = tree.get("tree", {}) if isinstance(tree, dict) else {}
                connected = tree.get("connected_dests", set()) if isinstance(tree, dict) else set()
                key = (
                    int(request.get("id", -1)) if isinstance(request, dict) else -1,
                    str(getattr(planner.env, "current_phase", "")),
                    int(getattr(planner.env, "next_vnf_idx", 0)),
                    int(start_node),
                    values,
                    len(connected),
                    sum(1 for value in tree_data.values() if float(value) > 0.0)
                    if isinstance(tree_data, dict) else 0,
                    int(getattr(planner, "_resource_snapshot_version", 0)),
                )
                cached = planner._high_score_cache.get(key)
                if cached is not None:
                    planner._high_score_cache_stats["hits"] += 1
                    return list(cached)
                planner._high_score_cache_stats["misses"] += 1
                result = list(__original(values, start_node))
                planner._high_score_cache[key] = result
                return list(result)

            helper.score_high_candidates = MethodType(cached_score, helper)
            patched.add(id(helper))
        self.high_score_cache_patched = len(patched)

    @staticmethod
    def _reuse_graph_edge_index(value: Any, cache: dict[str, torch.Tensor], key: str) -> Any:
        """Reuse the immutable topology edge index while keeping dynamic fields live.

        Legacy state builders recreate ``edge_index`` on every decision even
        though topology does not change online.  Only this immutable field is
        reused; edge attributes, tree edges, masks, and node features remain
        request/ledger dependent and are never cached.
        """
        state = value
        if isinstance(state, tuple):
            return tuple(
                OnlineLegacyHRLPlanner._reuse_graph_edge_index(item, cache, key)
                if hasattr(item, "edge_index") else item
                for item in state
            )
        if not hasattr(state, "edge_index"):
            return state
        edge_index = getattr(state, "edge_index", None)
        if edge_index is None:
            return state
        cached = cache.get(key)
        if cached is None or tuple(cached.shape) != tuple(edge_index.shape):
            cache[key] = edge_index
            cached = edge_index
        state.edge_index = cached
        return state

    def _install_static_graph_reuse(self) -> None:
        """Split immutable graph structure from dynamic state construction."""
        self._static_graph_edge_index: dict[str, torch.Tensor] = {}
        env = self.env
        high = getattr(env, "high_level_controller", None)
        low = getattr(env, "low_level_controller", None)
        patched = 0
        if high is not None and callable(getattr(high, "get_high_level_state_graph", None)):
            original_high = high.get_high_level_state_graph

            def cached_high(_controller, *args, **kwargs):
                value = original_high(*args, **kwargs)
                value = self._reuse_graph_edge_index(
                    value, self._static_graph_edge_index, "high"
                )
                return value

            high.get_high_level_state_graph = MethodType(cached_high, high)
            patched += 1
        if low is not None and callable(getattr(low, "get_state", None)):
            original_low = low.get_state

            def cached_low(_controller, *args, **kwargs):
                value = original_low(*args, **kwargs)
                value = self._reuse_graph_edge_index(
                    value, self._static_graph_edge_index, "low"
                )
                return value

            low.get_state = MethodType(cached_low, low)
            patched += 1
        self.static_graph_reuse_patched = patched

    def _set_request_seed(self, request_id: int) -> None:
        if not self.reseed_per_request:
            return
        episode_seed = (self.seed * 1_000_003 + int(request_id)) & 0x7FFFFFFF
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(episode_seed)

    def _request_step_budget(self, request: Mapping[str, Any]) -> int:
        """Return a bounded budget for hopeless requests without shrinking normal runs."""
        if self.failure_step_budget is None:
            return self.max_steps
        vnfs = len(request.get("vnf", []) or [])
        dests = len(request.get("dest", []) or [])
        # A successful request receives enough cycles for each placement and
        # destination subgoal; the configured ceiling mainly removes 600-step
        # tails caused by repeated no-progress retries.
        complexity_floor = 48 + 16 * vnfs + 16 * dests
        return min(self.max_steps, max(self.failure_step_budget, complexity_floor))

    def _plan_next_impl(self, runtime_request: dict[str, Any]) -> dict[str, Any]:
        request_id = int(runtime_request["id"])
        started = time.perf_counter()
        k_path_before = dict(getattr(self.env, "_k_path_candidate_stats", {}) or {})
        macro_before = dict(getattr(self, "_macro_rollout_stats", {}) or {})
        runtime_before = {
            key: dict(value) for key, value in self._runtime_timing_stats.items()
        }
        with self._lock:
            queue_started = time.perf_counter()
            if self.next_index >= len(self.request_order):
                raise RuntimeError("online HRL request stream is exhausted")
            expected_id = self.request_order[self.next_index]
            if request_id != expected_id:
                raise RuntimeError(
                    f"online HRL expected request {expected_id}, received {request_id}"
                )
            self._set_request_seed(request_id)
            if self._decision_timing_probe is not None:
                self._decision_timing_probe.reset()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                _, info = self.coordinator.run_episode(
                    training=False,
                    max_steps=self._request_step_budget(runtime_request),
                )
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            snapshot = info.get("req_snapshot") or {}
            inferred_id = int(snapshot.get("id", -1))
            if inferred_id != request_id:
                raise RuntimeError(
                    f"online HRL returned request {inferred_id}, expected {request_id}"
                )
            conversion_started = time.perf_counter()
            original = self.original_by_id[request_id]
            try:
                plan = convert_plan(
                    info,
                    original,
                    self.profile,
                    self.stage_port_base,
                )
            except (KeyError, TypeError, ValueError) as exc:
                # Preserve live operation when the frozen policy emits an
                # incomplete tree.  The repair generator is pure and creates
                # a complete, resource-feasible plan against a full snapshot;
                # the runtime executor still performs its own current-ledger
                # admission before installing anything.
                capacity = {
                    int(node["dpid"]): float(node.get("cpu_capacity", 55.0))
                    for node in self.profile.get("nodes", [])
                }
                memory = {
                    int(node["dpid"]): float(node.get("memory_capacity", 45.0))
                    for node in self.profile.get("nodes", [])
                }
                default_bw = float(self.profile.get("default_bandwidth_mbps", 90.0))
                bandwidth = {
                    (int(edge["u"]), int(edge["v"])): float(
                        edge.get("bandwidth_mbps", default_bw)
                    )
                    for edge in self.profile.get("edges", [])
                }
                bandwidth.update({(v, u): value for (u, v), value in list(bandwidth.items())})
                repair_snapshot = ResourceSnapshot(
                    version=0,
                    cpu_remaining=capacity,
                    memory_remaining=memory,
                    bandwidth_remaining=bandwidth,
                )
                repaired = self._complete_plan_repair.generate(
                    original, repair_snapshot
                )
                if not repaired:
                    raise RuntimeError(
                        f"HRL plan conversion failed and no complete repair exists: {exc}"
                    ) from exc
                candidate = repaired[0]
                plan = dict(candidate.plan)
                plan.setdefault("online_planning", {})["hrl_repair"] = {
                    "reason": f"{type(exc).__name__}: {exc}",
                    "source": "complete_plan_candidate_generator",
                }
            plan["resource_ledger"] = resource_ledger_snapshot(self.env, request_id)
            conversion_ms = (time.perf_counter() - conversion_started) * 1000.0
            self.next_index += 1

        algorithm_timing = info.get("algorithm_timing") or {}
        if not algorithm_timing and self._decision_timing_probe is not None:
            algorithm_timing = self._decision_timing_probe.summary(inference_ms)
        plan["online_planning"] = {
            "mode": "frozen_checkpoint_online_inference",
            "request_id": request_id,
            "stream_index": self.next_index - 1,
            "queue_wait_ms": (queue_started - started) * 1000.0,
            "inference_ms": inference_ms,
            "conversion_ms": conversion_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "steps": int(info.get("steps", 0)),
            "algorithm_timing": algorithm_timing,
        }
        k_path_after = dict(getattr(self.env, "_k_path_candidate_stats", {}) or {})
        if k_path_after:
            plan["online_planning"]["k_path_candidate_stats"] = {
                key: int(k_path_after.get(key, 0)) - int(k_path_before.get(key, 0))
                for key in k_path_after
            }
        macro_after = dict(getattr(self, "_macro_rollout_stats", {}) or {})
        if macro_after:
            plan["online_planning"]["macro_path_rollout_stats"] = {
                key: int(macro_after.get(key, 0)) - int(macro_before.get(key, 0))
                for key in macro_after
            }
        runtime_after = {
            key: dict(value) for key, value in self._runtime_timing_stats.items()
        }
        if runtime_after:
            plan["online_planning"]["runtime_timing_ms"] = {
                key: {
                    "count": int(value.get("count", 0)) - int(
                        runtime_before.get(key, {}).get("count", 0)
                    ),
                    "total_ms": float(value.get("total_ms", 0.0)) - float(
                        runtime_before.get(key, {}).get("total_ms", 0.0)
                    ),
                }
                for key, value in runtime_after.items()
            }
        plan["online_planning"]["optimization"] = {
            "macro_path_rollout": bool(getattr(self.env, "_macro_path_rollout", False)),
            "reach_cache": dict(getattr(self, "_online_feature_cache_stats", {})),
            "embedded_low_action_mask": False,
            "high_score_cache": dict(
                getattr(self, "_high_score_cache_stats", {})
            ),
            "static_graph_reuse": bool(
                getattr(self, "static_graph_reuse_patched", 0)
            ),
            "fast_resource_reads": bool(
                getattr(self, "fast_resource_reads_enabled", False)
            ),
            "fast_high_score": bool(
                getattr(self, "fast_high_score_patched", 0)
            ),
            "collect_timing": bool(self.collect_timing),
            "failure_step_budget": self.failure_step_budget,
            "completion_candidate_budget": self.completion_candidate_budget,
            "destination_beam_width": self.destination_beam_width,
            "request_step_budget": self._request_step_budget(runtime_request),
            "async_prefetch": bool(self._prefetch_executor is not None),
            "fast_k_path_candidates": self.fast_k_path_candidates,
        }
        return plan

    def plan_next(self, runtime_request: dict[str, Any]) -> dict[str, Any]:
        """Plan one request, consuming a matching background result when ready."""
        request_id = int(runtime_request["id"])
        future = None
        with self._prefetch_lock:
            future = self._prefetch_futures.pop(request_id, None)
        if future is not None:
            return future.result()
        return self._plan_next_impl(runtime_request)

    def plan_batch(self, requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch-facing HRL API used by the runtime scheduler.

        HRL remains stateful and therefore commits requests in arrival order;
        batching here provides one uniform scheduler contract and lets callers
        prefetch the following batch without changing the two-level policy.
        """
        return [self.plan_next(request) for request in requests]

    def release(self, request_id: int) -> bool:
        """Rollback a plan that never reached the deployment executor.

        ``run_episode`` commits a successful request into the shared legacy
        lifecycle ledger.  The runtime scheduler can subsequently reject that
        request because its wall-clock deadline expired while waiting in the
        deployment queue.  In that case the request must be removed from the
        lifecycle table and its CPU/MEM/BW footprint returned immediately;
        otherwise online HRL silently accumulates allocations for requests
        that were never deployed.
        """
        request_id = int(request_id)
        with self._lock:
            manager = getattr(self.env, "resource_mgr", None)
            if manager is None:
                return False
            request_manager = getattr(manager, "request_manager", None)
            released = False
            if request_manager is not None:
                active = getattr(request_manager, "active_requests", {})
                key = request_id if request_id in active else str(request_id)
                if key in active and hasattr(request_manager, "_release_request"):
                    request_manager._release_request(
                        key, float(getattr(self.env, "time_step", 0.0))
                    )
                    released = True
            if not released and hasattr(manager, "release_request_record"):
                released = bool(
                    manager.release_request_record(request_id, rollback=True)
                )
            if released:
                self._runtime_releases = int(getattr(self, "_runtime_releases", 0)) + 1
            return released

    def prefetch(self, requests: Sequence[dict[str, Any]]) -> int:
        """Schedule ordered requests on a background worker.

        The worker uses the same locked planner state, so resource/lifecycle
        mutations remain serialized.  It overlaps planning with data-plane
        execution when the caller submits the next batch after the current
        plan is handed off; stale or out-of-order requests are ignored.
        """
        if self._prefetch_executor is None:
            return 0
        submitted = 0
        with self._prefetch_lock:
            expected_index = self.next_index + len(self._prefetch_futures)
            for request in requests:
                request_id = int(request["id"])
                if request_id in self._prefetch_futures:
                    continue
                if expected_index >= len(self.request_order):
                    break
                if request_id != self.request_order[expected_index]:
                    break
                self._prefetch_futures[request_id] = self._prefetch_executor.submit(
                    self._plan_next_impl, request
                )
                expected_index += 1
                submitted += 1
        return submitted

    def close(self) -> None:
        executor = self._prefetch_executor
        self._prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": "frozen_checkpoint_online_inference",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256(self.checkpoint),
            "data_path": str(self.data_path),
            "runtime_requests": str(self.runtime_path),
            "profile": str(self.profile_path),
            "seed": self.seed,
            "max_steps": self.max_steps,
            "initialization_ms": self.initialization_ms,
            "planned_requests": self.next_index,
            "weights_updated_online": False,
            "torch_threads": self.torch_threads,
            "quiet": self.quiet,
            "static_topology_cache_enabled": self.static_topology_cache_enabled,
            "static_topology_helpers_patched": getattr(
                self, "static_topology_helpers_patched", 0
            ),
            "static_graph_reuse_patched": getattr(
                self, "static_graph_reuse_patched", 0
            ),
            "high_score_cache_patched": getattr(
                self, "high_score_cache_patched", 0
            ),
            "high_score_cache": dict(
                getattr(self, "_high_score_cache_stats", {})
            ),
            "failure_step_budget": self.failure_step_budget,
            "completion_candidate_budget": self.completion_candidate_budget,
            "destination_beam_width": self.destination_beam_width,
            "async_prefetch_workers": self.async_prefetch_workers,
            "fast_k_path_candidates": self.fast_k_path_candidates,
            "prefetch_pending": len(self._prefetch_futures),
            "runtime_releases": int(getattr(self, "_runtime_releases", 0)),
            "k_path_candidate_filter": bool(
                getattr(self.env, "_k_path_candidate_filter", False)
            ),
            "k_path_candidate_k": int(
                getattr(self.env, "_k_path_candidate_k", 0)
            ),
            "k_path_pairs_prewarmed": int(
                getattr(self, "k_path_pairs_prewarmed", 0)
            ),
            "macro_path_rollout": bool(
                getattr(self.env, "_macro_path_rollout", False)
            ),
        }
