#!/usr/bin/env python3
"""Explain the gap between complete SFC traversal and strict request SLA."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def analyze(
    result_path: Path, requests_path: Path, max_requests: int = 0
) -> dict[str, Any]:
    result = read_json(result_path)
    request_rows = read_jsonl(requests_path)
    if max_requests > 0:
        request_rows = request_rows[:max_requests]
    requests = {int(row["id"]): row for row in request_rows}
    sender_by_request = {
        int(row["request_id"]): row.get("result", {})
        for row in result.get("sender_results", [])
    }
    receivers_by_request: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in result.get("receiver_results", []):
        receivers_by_request[int(row["request_id"])].append(row.get("result", {}))
    vnfs_by_request: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in result.get("vnf_results", []):
        vnfs_by_request[int(row["request_id"])].append(row.get("result", {}))

    measured_ids = [int(row["id"]) for row in request_rows if int(row["id"]) in sender_by_request]
    reason_counts: Counter[str] = Counter()
    qos_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bandwidth_counts: dict[str, Counter[str]] = defaultdict(Counter)
    strict_ids: list[int] = []
    traversed_ids: list[int] = []

    for request_id in measured_ids:
        request = requests[request_id]
        sender = sender_by_request[request_id]
        receivers = receivers_by_request[request_id]
        vnfs = vnfs_by_request[request_id]
        expected_stages = len(request.get("vnf", []))
        started = int(sender.get("sent_packets", 0) or 0) > 0
        traversed = (
            started
            and len(vnfs) >= expected_stages
            and all(int(stage.get("forwarded", 0) or 0) > 0 for stage in vnfs)
        )
        strict = bool(receivers) and all(bool(row.get("sla_met")) for row in receivers)
        delay_ok = bool(receivers) and all(
            bool(row.get("delay_sla_met")) for row in receivers
        )
        jitter_ok = bool(receivers) and all(
            bool(row.get("jitter_sla_met")) for row in receivers
        )
        loss_ok = bool(receivers) and all(
            bool(row.get("loss_sla_met")) for row in receivers
        )

        if not started:
            reason = "not_started"
        elif not traversed:
            reason = "started_not_traversed"
        elif strict:
            reason = "strict_sla_met"
        else:
            failures = []
            if not delay_ok:
                failures.append("delay")
            if not jitter_ok:
                failures.append("jitter")
            if not loss_ok:
                failures.append("loss")
            reason = "+".join(failures) if failures else "other_sla"

        reason_counts[reason] += 1
        qos = str(request.get("qos_class", "unknown"))
        bandwidth = f"{float(request.get('bw_origin', 0.0)):g}Mbps"
        qos_counts[qos][reason] += 1
        bandwidth_counts[bandwidth][reason] += 1
        qos_counts[qos]["total"] += 1
        bandwidth_counts[bandwidth]["total"] += 1
        if traversed:
            traversed_ids.append(request_id)
            qos_counts[qos]["traversed"] += 1
            bandwidth_counts[bandwidth]["traversed"] += 1
        if strict:
            strict_ids.append(request_id)
            qos_counts[qos]["strict"] += 1
            bandwidth_counts[bandwidth]["strict"] += 1

    def breakdown(rows: dict[str, Counter[str]]) -> dict[str, Any]:
        return {
            key: {
                **dict(sorted(counts.items())),
                "strict_rate_total": rate(counts["strict"], counts["total"]),
                "strict_rate_given_traversal": rate(
                    counts["strict"], counts["traversed"]
                ),
            }
            for key, counts in sorted(rows.items())
        }

    total = len(measured_ids)
    traversed = len(traversed_ids)
    strict = len(strict_ids)
    return {
        "result": str(result_path),
        "requests": str(requests_path),
        "total_requests": total,
        "complete_sfc_traversal": traversed,
        "strict_sla": strict,
        "strict_rate_total": rate(strict, total),
        "strict_rate_given_traversal": rate(strict, traversed),
        "traversal_minus_strict": traversed - strict,
        "reasons": dict(sorted(reason_counts.items())),
        "by_qos": breakdown(qos_counts),
        "by_bandwidth": breakdown(bandwidth_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.result, args.requests, args.max_requests)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
