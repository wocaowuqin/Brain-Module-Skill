#!/usr/bin/env python3
"""Check that stateful HRL preparation is isolated from the central ledger."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.marl.orchestration.hrl_adapter import HRLPlannerAdapter, build_ledger_from_profile


class FakeStatefulHRL:
    def __init__(self) -> None:
        self.planned: list[int] = []
        self.released: list[int] = []

    def plan_next(self, request: dict) -> dict:
        self.planned.append(int(request["id"]))
        # The adapter does not trust this plan blindly; the pure generator
        # validates it again against the current profile and snapshot.
        return {"accepted": True, "request_id": int(request["id"]), "invalid": True}

    def release(self, request_id: int) -> bool:
        self.released.append(int(request_id))
        return True


def main() -> int:
    profile_path = ROOT / "sdn" / "topologies" / "us_backbone_28_bw90.json"
    request_path = (
        ROOT / "data" / "sdn_runtime_requests"
        / "seed_7071_lifetime50node_rate8" / "requests.jsonl"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8").splitlines()[0])
    planner = FakeStatefulHRL()
    adapter = HRLPlannerAdapter(planner, profile, max_candidates=4)
    ledger = build_ledger_from_profile(profile)
    before = ledger.snapshot().version
    report = adapter.prepare_batch([request])
    generation = adapter.candidate_factory(request, ledger.snapshot())
    adapter.forget_batch([request])
    checks = {
        "planner_called_in_order": planner.planned == [int(request["id"])],
        "accepted_plan_released": planner.released == [int(request["id"])],
        "candidate_contract_aligned": len(generation.footprints) == len(generation.payloads),
        "reject_is_final": generation.footprints[-1] is None,
        "central_ledger_untouched": ledger.snapshot().version == before,
        "report_counts": report.accepted == 1 and report.released == 1,
        "report_timing_present": report.elapsed_ms >= 0.0,
    }
    adapter.close()
    result = {"ok": all(checks.values()), "checks": checks, "report": report.to_dict()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
