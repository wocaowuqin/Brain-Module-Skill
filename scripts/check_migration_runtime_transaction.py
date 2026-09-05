#!/usr/bin/env python3
"""Verify endpoint quarantine and ambiguous Ryu commit reconciliation helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sdn_runtime_requests import (
    VnfEndpointPool,
    ryu_status_matches_sfc_plan,
)


def endpoint_plan(request_id: int, dc_node: int) -> dict:
    return {
        "request_id": int(request_id),
        "placement_by_vnf": {
            "0": {"dc_node": int(dc_node), "listen_port": 0},
        },
        "segments": [{"stage": 0, "udp_port": 0}],
    }


def main() -> int:
    pool = VnfEndpointPool(port_base=30000, ports_per_dc=1)
    first, _ = pool.assign(1, endpoint_plan(1, 1))
    assert first is not None
    token, target_port, _ = pool.reserve_migration(1, 0, 2)
    assert token is not None and target_port == 30000
    assert pool.finish_migration(token, commit=True)
    committed_snapshot = pool.snapshot()
    assert committed_snapshot["active_endpoints"] == 2
    assert committed_snapshot["retired_migration_sources"] == 1

    blocked, blocked_detail = pool.assign(2, endpoint_plan(2, 1))
    assert blocked is None
    assert blocked_detail["reason"] == "vnf_endpoint_pool_exhausted"
    assert pool.release_migration_source(token)
    second, _ = pool.assign(2, endpoint_plan(2, 1))
    assert second is not None
    assert pool.release(2)
    assert pool.release(1)
    released_snapshot = pool.snapshot()
    assert released_snapshot["active_endpoints"] == 0
    assert released_snapshot["pending_migrations"] == 0
    assert released_snapshot["retired_migration_sources"] == 0

    abort_pool = VnfEndpointPool(port_base=31000, ports_per_dc=1)
    third, _ = abort_pool.assign(3, endpoint_plan(3, 1))
    assert third is not None
    abort_token, _, _ = abort_pool.reserve_migration(3, 0, 2)
    assert abort_pool.finish_migration(abort_token, commit=False)
    assert abort_pool.snapshot()["active_endpoints"] == 1
    assert abort_pool.release(3)

    plan = {
        "segments": [
            {
                "stage": 0,
                "target_ip": "10.0.0.2",
                "udp_port": 30001,
                "path": [1, 2],
                "switch_outputs": {"1": [2], "2": [1]},
            }
        ]
    }
    status = {
        "sfcs": {
            "9": {
                "segments": [
                    {
                        **plan["segments"][0],
                        "cookie": 123,
                    }
                ]
            }
        }
    }
    assert ryu_status_matches_sfc_plan(status, 9, plan)
    changed = json.loads(json.dumps(status))
    changed["sfcs"]["9"]["segments"][0]["path"] = [1, 3, 2]
    assert not ryu_status_matches_sfc_plan(changed, 9, plan)

    print(json.dumps({
        "valid": True,
        "source_endpoint_quarantined_until_unregister": True,
        "abort_releases_only_target_endpoint": True,
        "ryu_commit_status_reconciliation": True,
        "final_endpoint_pool": released_snapshot,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
