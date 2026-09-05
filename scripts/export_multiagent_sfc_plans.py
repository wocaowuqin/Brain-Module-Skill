#!/usr/bin/env python3
"""Export committed multi-agent plans as runtime SFC-plan JSONL."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="multiagent_online result JSON")
    parser.add_argument("--output", required=True, help="runtime SFC plans JSONL")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    target = Path(args.output).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("requests")
    if not isinstance(rows, list):
        raise ValueError("input does not contain a requests list")
    exported: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        request_id = int(row["request_id"])
        if request_id in seen:
            raise ValueError(f"duplicate request_id {request_id}")
        seen.add(request_id)
        plan = row.get("plan")
        accepted = bool(row.get("accepted", False)) and isinstance(plan, dict)
        if not accepted:
            exported.append({
                "request_id": request_id,
                "accepted": False,
                "reason": "central ledger rejected candidate",
            })
            continue
        plan_copy = copy.deepcopy(plan)
        if int(plan_copy.get("request_id", request_id)) != request_id:
            raise ValueError(f"plan request_id mismatch for {request_id}")
        plan_copy["request_id"] = request_id
        plan_copy["accepted"] = True
        plan_copy.setdefault("source", "multiagent_online")
        exported.append(plan_copy)
    exported.sort(key=lambda row: int(row["request_id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    accepted = sum(bool(row.get("accepted", False)) for row in exported)
    print(json.dumps({
        "input": str(source),
        "output": str(target),
        "requests": len(exported),
        "accepted": accepted,
        "rejected": len(exported) - accepted,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
