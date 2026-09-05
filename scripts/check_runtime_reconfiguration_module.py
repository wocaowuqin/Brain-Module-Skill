#!/usr/bin/env python3
"""Check Brain -> specialist -> runtime skill routing without mutating resources."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration import AgentRole, RuntimeReconfigurationModule


def main() -> int:
    module = RuntimeReconfigurationModule()
    migration = module.route_migration(
        timestamp=1.0,
        request_id=41,
        payload={
            "request_active": True,
            "policy": "migration_wqmix_joint",
            "stage": 1,
            "target_dc": 7,
            "estimated_gain": 0.6,
        },
        metrics={"node_hotspots": 1, "link_hotspots": 0},
        snapshot_version="ledger-9",
    )
    assert migration.status_code == "DISPATCHED"
    assert migration.command.assigned_roles == (AgentRole.MIGRATION,)
    assert migration.proposals[0].skill_name == "runtime_vnf_migration"
    assert migration.command.snapshot_version == "ledger-9"
    migration_outcome = module.record_outcome(
        migration,
        success=True,
        applied=True,
        attempted=True,
        reason="make-before-break committed",
    )

    reroute = module.route_reroute(
        timestamp=2.0,
        request_id=42,
        payload={
            "request_active": True,
            "policy": "strict_sla_reroute",
            "estimated_gain": 0.4,
        },
        metrics={"node_hotspots": 0, "link_hotspots": 1},
        sla_alert=True,
    )
    assert reroute.status_code == "DISPATCHED"
    assert reroute.command.assigned_roles == (AgentRole.REROUTE,)
    assert reroute.proposals[0].skill_name == "runtime_tree_reroute"
    reroute_outcome = module.record_outcome(
        reroute,
        success=False,
        applied=False,
        attempted=False,
        reason="strict gate rejected stale utilization",
    )

    inactive = module.route_migration(
        timestamp=3.0,
        request_id=43,
        payload={"request_active": False, "policy": "predictive_migration_wqmix"},
        metrics={"node_hotspots": 1},
    )
    assert inactive.status_code == "DISPATCH_REJECTED"
    metadata = module.metadata()
    assert metadata["architecture"] == "brain_module_skill"
    assert metadata["dispatch_count"] == 3
    assert metadata["outcome_counts"] == {
        "applied": 1,
        "skipped": 1,
        "dispatch_rejected": 1,
    }
    assert metadata["pending_commands"] == 0
    print(json.dumps({
        "valid": True,
        "migration_outcome": migration_outcome,
        "reroute_outcome": reroute_outcome,
        "metadata": metadata,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
