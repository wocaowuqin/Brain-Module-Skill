#!/usr/bin/env python3
"""Run three real requests through Brain -> Module -> Skill -> batched HRL."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration import BrainManagedDeploymentPlanner
from sdn.online_batched_hrl_planner import OnlineBatchedHRLPlanner
from sdn.online_hrl_planner import OnlineLegacyHRLPlanner


def main() -> int:
    legacy_root = Path.home() / "Desktop" / "hrl"
    request_dir = ROOT / "data" / "sdn_runtime_requests" / "seed_7071_lifetime50node_rate8"
    request_jsonl = request_dir / "requests.jsonl"
    request_pickle = request_dir / "requests.pkl"
    profile_path = ROOT / "sdn" / "topologies" / "us_backbone_28_bw90.json"
    checkpoint = (
        ROOT / "artifacts" / "runs" / "hrl" / "current_model_rate24_seed7071_100"
        / "noIL_rate24" / "final_model.pth"
    )
    output = (
        ROOT / "artifacts" / "runs" / "multiagent_online"
        / "brain_module_skill_real_3.json"
    )
    required = (legacy_root, request_jsonl, request_pickle, profile_path, checkpoint)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    requests = [
        json.loads(line)
        for line in request_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:3]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    loaded_hrl = OnlineLegacyHRLPlanner(
        legacy_root=legacy_root,
        checkpoint=checkpoint,
        data_path=request_pickle,
        runtime_requests=request_jsonl,
        profile=profile_path,
        seed=7071,
        max_steps=600,
        torch_threads=1,
        quiet=True,
        collect_timing=False,
    )
    base = OnlineBatchedHRLPlanner(
        loaded_hrl,
        profile,
        microbatch_ms=5.0,
        max_agents=32,
        top_k=4,
        worker_count=4,
    )
    planner = BrainManagedDeploymentPlanner(base)
    try:
        plans = planner.plan_batch(requests)
        metadata_before_release = planner.metadata()
        released = [
            int(plan["request_id"])
            for plan in plans
            if bool(plan.get("accepted")) and planner.release(int(plan["request_id"]))
        ]
        payload = {
            "valid": all(
                plan.get("orchestration", {}).get("architecture")
                == "brain_module_skill"
                for plan in plans
            ),
            "request_count": len(requests),
            "accepted_count": sum(bool(plan.get("accepted")) for plan in plans),
            "released_request_ids": released,
            "plans": plans,
            "metadata_before_release": metadata_before_release,
            "metadata_after_release": planner.metadata(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not payload["valid"]:
            raise RuntimeError("real plans did not traverse Brain/Module/Skill")
        print(json.dumps({
            "valid": payload["valid"],
            "request_count": payload["request_count"],
            "accepted_count": payload["accepted_count"],
            "released_request_ids": released,
            "output": str(output),
        }, ensure_ascii=False, indent=2))
    finally:
        planner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
