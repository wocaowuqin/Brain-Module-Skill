#!/usr/bin/env python3
"""Smoke-check the central brain, role permissions and safe action execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration import (
    AgentRole,
    EventType,
    MultiAgentSFTOrchestrator,
    OrchestrationEvent,
)
from scripts.check_reconfiguration_manager import build_resource_manager, deploy_active_sft


class FakeHRLPlanner:
    """Interface-compatible stand-in; the smoke test does not load a checkpoint."""

    def __init__(self) -> None:
        self.planned: list[int] = []
        self.released: list[int] = []

    def plan_next(self, request: dict) -> dict:
        request_id = int(request["id"])
        self.planned.append(request_id)
        return {
            "version": "hrl_sfc_plan_v1",
            "request_id": request_id,
            "accepted": True,
            "chain_nodes": [2, 4],
            "segments": [],
            "multicast": {"tree_edges": []},
        }

    def release(self, request_id: int) -> bool:
        self.released.append(int(request_id))
        return True


def run_check() -> dict:
    resource_manager = build_resource_manager()
    deploy_active_sft(resource_manager)
    hrl = FakeHRLPlanner()
    orchestrator = MultiAgentSFTOrchestrator(
        resource_manager,
        hrl_planner=hrl,
        node_util_threshold=0.50,
        link_util_threshold=0.80,
    )

    arrival = OrchestrationEvent(
        event_id="arrival-3001",
        event_type=EventType.REQUEST_ARRIVAL,
        timestamp=1.0,
        request={"id": 3001, "vnf": [0, 1], "dest": [3, 5]},
        deadline_ms=120.0,
    )
    deployment = orchestrator.handle(arrival, apply=True)

    review = OrchestrationEvent(
        event_id="review-1",
        event_type=EventType.PERIODIC_REVIEW,
        timestamp=2.0,
    )
    reconfiguration = orchestrator.handle(review, apply=True)

    permission_blocked = False
    try:
        orchestrator.registry.get("plan_tree_reroute", role=AgentRole.MIGRATION)
    except PermissionError:
        permission_blocked = True

    released = orchestrator.release(3001)
    snapshot = resource_manager.validate_request_sft_snapshot(2001)
    checks = {
        "hrl_received_arrival": hrl.planned == [3001],
        "deployment_role_only": [
            item.role.value for item in deployment.proposals
        ] == ["deployment"],
        "deployment_is_honest_plan_only": (
            deployment.success
            and not deployment.applied
            and deployment.status_code == "PLANNED_ONLY"
        ),
        "brain_requested_both_specialists": set(
            role.value for role in reconfiguration.command.assigned_roles
        ) == {"migration", "reroute"},
        "safe_single_action_applied": (
            reconfiguration.applied
            and reconfiguration.selected in {"migrate", "reroute"}
        ),
        "ledger_snapshot_valid": bool(snapshot.get("ok")),
        "cross_role_skill_blocked": permission_blocked,
        "hrl_release_forwarded": released and hrl.released == [3001],
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "metadata": orchestrator.metadata(),
        "deployment": deployment.to_dict(),
        "reconfiguration": reconfiguration.to_dict(),
        "snapshot": snapshot,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_check()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            "Multi-agent orchestration smoke check",
            "passed" if result["ok"] else "failed",
        )
        print(json.dumps(result["checks"], ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
