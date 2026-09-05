#!/usr/bin/env python3
"""Export an SDN runtime result into request, receiver, and sender CSV tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="runtime experiment result JSON")
    parser.add_argument("--requests", required=True, help="source request JSONL")
    parser.add_argument(
        "--output-dir",
        help="CSV directory; defaults to <result stem>_csv beside the result",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def ordered_fields(rows: list[dict[str, Any]], preferred: Iterable[str]) -> list[str]:
    available = {key for row in rows for key in row}
    fields = [key for key in preferred if key in available]
    fields.extend(sorted(available - set(fields)))
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: Iterable[str]) -> None:
    fields = ordered_fields(rows, preferred)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: csv_value(row.get(key)) for key in fields}
            for row in rows
        )


def request_record(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": int(request["id"]),
        **{key: value for key, value in request.items() if key != "id"},
    }


def prefixed(prefix: str, values: dict[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    return {f"{prefix}{key}": value for key, value in values.items()}


def mean(values: list[float]) -> float | str:
    return statistics.fmean(values) if values else ""


def maximum(values: list[float]) -> float | str:
    return max(values) if values else ""


def main() -> None:
    args = parse_args()
    result_path = resolve(args.result)
    request_path = resolve(args.requests)
    result = read_json(result_path)
    requests = read_jsonl(request_path)
    request_by_id = {int(row["id"]): row for row in requests}
    measured_request_ids = {
        int(row["request_id"])
        for key in ("sender_results", "receiver_results")
        for row in result.get(key, [])
    }
    measured_request_ids.update(
        int(row["request_id"])
        for row in result.get("events", [])
        if row.get("type") == "arrive"
    )
    missing_request_ids = sorted(measured_request_ids - set(request_by_id))
    if missing_request_ids:
        raise ValueError(
            f"result contains request IDs missing from the source trace: "
            f"{missing_request_ids[:10]}"
        )

    output_dir = (
        resolve(args.output_dir)
        if args.output_dir
        else result_path.parent / f"{result_path.stem}_csv"
    )
    stem = result_path.stem

    arrivals: dict[int, dict[str, Any]] = {}
    reroutes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in result.get("events", []):
        request_id = int(event["request_id"])
        if event.get("type") == "arrive":
            arrivals[request_id] = event
        elif event.get("type") == "reroute":
            reroutes[request_id].append(event)

    receivers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    receiver_rows: list[dict[str, Any]] = []
    for row in result.get("receiver_results", []):
        request_id = int(row["request_id"])
        request = request_by_id[request_id]
        probe = row.get("result")
        receivers[request_id].append(row)
        receiver_rows.append(
            {
                **request_record(request),
                "destination_dpid": row.get("destination_dpid"),
                "deployment_accepted": bool(
                    arrivals.get(request_id, {}).get("controller", {}).get("accepted")
                ),
                **prefixed("probe_", probe),
            }
        )

    senders: dict[int, dict[str, Any]] = {}
    sender_rows: list[dict[str, Any]] = []
    for row in result.get("sender_results", []):
        request_id = int(row["request_id"])
        request = request_by_id[request_id]
        probe = row.get("result")
        senders[request_id] = row
        sender_rows.append(
            {
                **request_record(request),
                "deployment_accepted": bool(
                    arrivals.get(request_id, {}).get("controller", {}).get("accepted")
                ),
                **prefixed("sender_", probe),
            }
        )

    request_rows: list[dict[str, Any]] = []
    for request_id in sorted(measured_request_ids):
        request = request_by_id[request_id]
        receiver_group = receivers.get(request_id, [])
        probes = [row["result"] for row in receiver_group if row.get("result")]
        expected_receivers = len(request.get("destination_dpids", request.get("dest", [])))
        complete = len(probes) == expected_receivers
        strict_accepted = complete and all(bool(row.get("sla_met")) for row in probes)
        delay_values = [float(row["mean_delay_ms"]) for row in probes if row.get("mean_delay_ms") is not None]
        p95_values = [float(row["p95_delay_ms"]) for row in probes if row.get("p95_delay_ms") is not None]
        max_delay_values = [float(row["max_delay_ms"]) for row in probes if row.get("max_delay_ms") is not None]
        jitter_values = [float(row["jitter_ms"]) for row in probes if row.get("jitter_ms") is not None]
        loss_values = [float(row["packet_loss_rate"]) for row in probes if row.get("packet_loss_rate") is not None]
        sender = senders.get(request_id, {}).get("result") or {}
        request_reroutes = reroutes.get(request_id, [])
        request_rows.append(
            {
                **request_record(request),
                "deployment_accepted": bool(
                    arrivals.get(request_id, {}).get("controller", {}).get("accepted")
                ),
                "expected_receivers": expected_receivers,
                "measured_receivers": len(probes),
                "strict_request_accepted": strict_accepted,
                "sla_met_receivers": sum(bool(row.get("sla_met")) for row in probes),
                "delay_sla_met_receivers": sum(bool(row.get("delay_sla_met")) for row in probes),
                "jitter_sla_met_receivers": sum(bool(row.get("jitter_sla_met")) for row in probes),
                "loss_sla_met_receivers": sum(bool(row.get("loss_sla_met")) for row in probes),
                "traffic_observed_receivers": sum(int(row.get("received_packets", 0)) > 0 for row in probes),
                "measurement_completed_receivers": sum(row.get("measurement_status") == "completed" for row in probes),
                "mean_receiver_delay_ms": mean(delay_values),
                "max_receiver_p95_delay_ms": maximum(p95_values),
                "max_receiver_delay_ms": maximum(max_delay_values),
                "mean_receiver_jitter_ms": mean(jitter_values),
                "max_receiver_jitter_ms": maximum(jitter_values),
                "mean_packet_loss_rate": mean(loss_values),
                "max_packet_loss_rate": maximum(loss_values),
                "sender_status": sender.get("status", ""),
                "sender_planned_packets": sender.get("planned_packets", ""),
                "sender_sent_packets": sender.get("sent_packets", ""),
                "sender_ready_wait_ms": (
                    float(sender["ready_wait_seconds"]) * 1000.0
                    if sender.get("ready_wait_seconds") is not None
                    else ""
                ),
                "sender_agent_queue_wait_ms": sender.get("agent_queue_wait_ms", ""),
                "reroutes_requested": len(request_reroutes),
                "reroutes_applied": sum(
                    bool(row.get("controller", {}).get("accepted"))
                    for row in request_reroutes
                ),
            }
        )

    request_preferred = [
        "request_id", "arrival_time", "leave_time", "lifetime", "source",
        "source_dpid", "dest", "destination_dpids", "qos_class", "priority",
        "traffic_profile", "bw_origin", "cpu_origin", "memory_origin", "vnf",
        "delay_bound_ms", "delay_compliance_ratio", "jitter_bound_ms",
        "packet_loss_bound", "deployment_accepted", "expected_receivers",
        "measured_receivers", "strict_request_accepted", "sla_met_receivers",
        "delay_sla_met_receivers", "jitter_sla_met_receivers",
        "loss_sla_met_receivers", "traffic_observed_receivers",
        "measurement_completed_receivers", "mean_receiver_delay_ms",
        "max_receiver_p95_delay_ms", "max_receiver_delay_ms",
        "mean_receiver_jitter_ms", "max_receiver_jitter_ms",
        "mean_packet_loss_rate", "max_packet_loss_rate", "sender_status",
        "sender_planned_packets", "sender_sent_packets", "sender_ready_wait_ms",
        "sender_agent_queue_wait_ms", "reroutes_requested", "reroutes_applied",
    ]
    receiver_preferred = [
        "request_id", "source_dpid", "destination_dpid", "qos_class", "priority",
        "bw_origin", "deployment_accepted", "probe_measurement_status",
        "probe_expected_packets", "probe_received_packets", "probe_lost_packets",
        "probe_packet_loss_rate", "probe_mean_delay_ms", "probe_p50_delay_ms",
        "probe_p95_delay_ms", "probe_p99_delay_ms", "probe_max_delay_ms",
        "probe_jitter_ms", "probe_delay_bound_ms", "probe_jitter_bound_ms",
        "probe_packet_loss_bound", "probe_delay_sla_met", "probe_jitter_sla_met",
        "probe_loss_sla_met", "probe_sla_met", "probe_agent_queue_wait_ms",
    ]
    sender_preferred = [
        "request_id", "source_dpid", "qos_class", "priority", "bw_origin",
        "deployment_accepted", "sender_status", "sender_destination", "sender_port",
        "sender_dscp", "sender_payload_bytes", "sender_packets_per_second",
        "sender_planned_packets", "sender_sent_packets", "sender_elapsed_seconds",
        "sender_ready_files", "sender_ready_wait_seconds",
        "sender_agent_queue_wait_ms",
    ]

    request_csv = output_dir / f"{stem}_requests.csv"
    receiver_csv = output_dir / f"{stem}_receivers.csv"
    sender_csv = output_dir / f"{stem}_senders.csv"
    write_csv(request_csv, request_rows, request_preferred)
    write_csv(receiver_csv, receiver_rows, receiver_preferred)
    write_csv(sender_csv, sender_rows, sender_preferred)

    summary = {
        "request_rows": len(request_rows),
        "receiver_rows": len(receiver_rows),
        "sender_rows": len(sender_rows),
        "strict_request_accepted": sum(bool(row["strict_request_accepted"]) for row in request_rows),
        "request_csv": str(request_csv),
        "receiver_csv": str(receiver_csv),
        "sender_csv": str(sender_csv),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
