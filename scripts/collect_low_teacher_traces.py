#!/usr/bin/env python3
"""Collect low-level HRL teacher decisions for student-policy distillation."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import types

from sdn.online_hrl_planner import OnlineLegacyHRLPlanner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--runtime-requests", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime_path = Path(args.runtime_requests)
    requests = [
        __import__("json").loads(line)
        for line in runtime_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: int(args.max_requests)]
    planner = OnlineLegacyHRLPlanner(
        legacy_root=args.legacy_root,
        checkpoint=args.checkpoint,
        data_path=args.data,
        runtime_requests=runtime_path,
        profile=args.profile,
        seed=args.seed,
        planner_destinations=True,
        torch_threads=1,
        quiet=True,
        k_path_candidate_filter=True,
        k_path_candidate_k=2,
        macro_path_rollout=True,
        failure_step_budget=240,
    )

    records = []
    low_policy = planner.agent.low_policy
    original_select = low_policy.select_action

    def capture(_policy, *call_args, **kwargs):
        action, value = original_select(*call_args, **kwargs)
        candidate_features = kwargs.get("candidate_local_feats")
        candidate_indices = kwargs.get("candidate_indices")
        if candidate_features is not None and candidate_indices:
            try:
                action_id = int(action.item()) if hasattr(action, "item") else int(action)
                indices = [int(value) for value in candidate_indices]
                if action_id in indices:
                    request = planner.env.current_request or {}
                    records.append({
                        "candidate_features": candidate_features.tolist(),
                        "candidate_indices": indices,
                        "teacher_action": action_id,
                        "current_node": int(kwargs.get("current_node_idx", -1)),
                        "target_node": planner._active_low_target(),
                        "phase": str(getattr(planner.env, "current_phase", "")),
                        "next_vnf_idx": int(getattr(planner.env, "next_vnf_idx", 0)),
                        "total_vnf": len(request.get("vnf", []) or []),
                        "bw_req": float(request.get("bw_origin", request.get("bw", 0.0))),
                        "node_count": int(getattr(planner.env, "n", 1)),
                    })
            except Exception:
                pass
        return action, value

    low_policy.select_action = types.MethodType(capture, low_policy)
    for request in requests:
        planner.plan_next(request)
    planner.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print({"requests": len(requests), "records": len(records), "output": str(output)})


if __name__ == "__main__":
    main()
