#!/usr/bin/env python3
"""Deterministic checks for runtime request generation, trees, and SLA probes."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdn.runtime_request_generator import (
    generate_requests,
    load_json,
    request_events,
    write_trace,
)
from sdn.udp_sla_probe import receive_probe, send_probe, wait_for_ready_files
from scripts.run_sdn_runtime_requests import (
    RollingSetupEstimator,
    RyuBatchCommitter,
    StageCapacityGate,
    VnfEndpointPool,
    VnfControlBatcher,
    bandwidth_mbps_to_pps,
    probe_agent_worker_plan,
    load_sfc_candidate_plans,
    probe_stop_deadlines_ns,
    remote_deadline_ns,
    resolve_controller_rest_url,
    sfc_candidate_sla_score,
    shortest_tree_outputs,
    validate_sla_predictor_runtime_contract,
    vnf_agent_start_command,
)


def controller_rest_discovery_check() -> dict:
    discovered = resolve_controller_rest_url("auto", "172.27.10.20")
    explicit = resolve_controller_rest_url("http://127.0.0.1:8080/", "ignored")
    if discovered != "http://172.27.10.20:8080":
        raise AssertionError(f"invalid discovered REST URL: {discovered}")
    if explicit != "http://127.0.0.1:8080":
        raise AssertionError(f"explicit REST URL was changed: {explicit}")
    return {"discovered": discovered, "explicit": explicit}


def remote_deadline_check() -> dict:
    local_now_ns = 10_000_000_000
    remote_now_ns = 8_500_000_000
    fresh_offset_ns = remote_now_ns - local_now_ns
    budget_seconds = 1.25
    deadline_ns = remote_deadline_ns(
        local_now_ns, fresh_offset_ns, budget_seconds
    )
    if deadline_ns != remote_now_ns + 1_250_000_000:
        raise AssertionError("fresh remote clock mapping produced the wrong deadline")
    stale_deadline_ns = remote_deadline_ns(
        local_now_ns, fresh_offset_ns + 1_450_000_000, budget_seconds
    )
    sender_deadline_ns, receiver_deadline_ns = probe_stop_deadlines_ns(
        local_now_ns,
        fresh_offset_ns,
        remaining_lifetime_seconds=2.0,
        sender_stop_margin_seconds=0.1,
    )
    if receiver_deadline_ns - sender_deadline_ns != 100_000_000:
        raise AssertionError("receiver drain deadline is not 100 ms after sender")
    return {
        "budget_seconds": budget_seconds,
        "fresh_deadline_ns": deadline_ns,
        "sender_deadline_ns": sender_deadline_ns,
        "receiver_deadline_ns": receiver_deadline_ns,
        "drain_window_ms": (
            receiver_deadline_ns - sender_deadline_ns
        ) / 1_000_000.0,
        "stale_mapping_error_ms": (
            stale_deadline_ns - deadline_ns
        ) / 1_000_000.0,
    }


def sla_runtime_contract_check() -> dict:
    contract = {
        "label_semantics": "request_strict_sla_with_receiver_drain_v1",
        "required_receiver_drain_ms": 50.0,
        "probe_sender_backend": "native",
        "probe_receiver_backend": "native",
        "vnf_agent_backend": "native",
        "mininet_qdisc": "htb_fq_codel",
        "vnf_agent_packet_batch": 16,
        "vnf_agent_q0_packet_batch": 8,
        "vnf_agent_dscp_scheduling": False,
    }
    args = SimpleNamespace(
        probe_sender_backend="native",
        probe_receiver_backend="native",
        vnf_agent_backend="native",
        mininet_qdisc="htb_fq_codel",
        vnf_agent_packet_batch=16,
        vnf_agent_q0_packet_batch=8,
        vnf_agent_dscp_scheduling=False,
        sender_stop_margin_seconds=0.1,
    )
    validated = validate_sla_predictor_runtime_contract(contract, args)
    mismatched = SimpleNamespace(**vars(args))
    mismatched.mininet_qdisc = "htb_prio"
    try:
        validate_sla_predictor_runtime_contract(contract, mismatched)
    except ValueError:
        mismatch_rejected = True
    else:
        mismatch_rejected = False
    if not mismatch_rejected:
        raise AssertionError("SLA predictor accepted a mismatched runtime contract")

    native_command = vnf_agent_start_command(
        "/tmp/agent",
        "/tmp/agent.fifo",
        "/tmp/agent.ready",
        "/tmp/agent.log",
        "/tmp/agent.pid",
        1,
        0,
        0,
        False,
        100.0,
        5.0,
        0,
        backend="native",
        dscp_scheduling=True,
    )
    if "--dscp-scheduling" not in native_command:
        raise AssertionError("native VNF DSCP scheduler flag was not forwarded")
    return {
        "validated": bool(validated and validated["validated"]),
        "mismatch_rejected": mismatch_rejected,
        "native_scheduler_flag": True,
    }


def sfc_candidate_selection_check():
    candidate = {
        "request_id": 1,
        "accepted": True,
        "segments": [{"stage": 0, "path": [1, 2]}],
        "placement_by_vnf": {"0": {"dc_node": 2}},
        "multicast": {"switch_outputs": {"2": [1]}},
    }
    row = {
        "agents": [
            {
                "request_id": 1,
                "candidates": [
                    {
                        "candidate_id": "r1-c0",
                        "metrics": {
                            "estimated_delay_ms": 10.0,
                            "delay_bound_ms": 100.0,
                            "segment_hops": 1.0,
                            "tree_edges": 2.0,
                            "flowmod_estimate": 3.0,
                        },
                        "plan": candidate,
                    }
                ],
            }
        ]
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "candidates.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        loaded = load_sfc_candidate_plans(str(path))[1][0]
    metrics = loaded["topk_selection"].get("candidate_metrics")
    if not metrics or metrics.get("estimated_delay_ms") != 10.0:
        raise AssertionError(f"candidate metrics were not retained: {loaded}")
    low_load = sfc_candidate_sla_score(loaded, [0.2], 0)
    high_load = sfc_candidate_sla_score(loaded, [0.8], 0)
    if not low_load < high_load:
        raise AssertionError(
            f"SLA-aware score ignored projected congestion: {low_load}, {high_load}"
        )
    return {"low_load": low_load, "high_load": high_load}


def deployment_scheduler_check():
    estimator = RollingSetupEstimator(150.0, 8, 4, 1.2)
    for sample in (20.0, 30.0, 40.0):
        estimator.observe(sample)
    if estimator.estimate_ms() != 150.0:
        raise AssertionError("setup estimator left cold-start mode too early")
    estimator.observe(50.0)
    if not 50.0 < estimator.estimate_ms() < 60.0:
        raise AssertionError(f"invalid rolling setup estimate: {estimator.snapshot()}")

    gate = StageCapacityGate("check", 2)
    if not gate.try_acquire() or not gate.try_acquire() or gate.try_acquire():
        raise AssertionError(f"stage capacity limit failed: {gate.snapshot()}")
    gate.release()
    gate.release()
    if gate.snapshot() != {"limit": 2, "current": 0, "peak": 2, "rejections": 1}:
        raise AssertionError(f"stage capacity accounting failed: {gate.snapshot()}")

    endpoint_pool = VnfEndpointPool(30000, 2)
    plan = {
        "segments": [
            {"stage": 0, "udp_port": 20000},
            {"stage": 1, "udp_port": 20001},
        ],
        "placement_by_vnf": {
            "0": {"dc_node": 7, "listen_port": 20000},
            "1": {"dc_node": 7, "listen_port": 20001},
        },
    }
    pooled, detail = endpoint_pool.assign(1, plan)
    if pooled is None or detail["ports"] != [30000, 30001]:
        raise AssertionError(f"VNF endpoint assignment failed: {detail}")
    exhausted, detail = endpoint_pool.assign(2, plan)
    if exhausted is not None or detail.get("reason") != "vnf_endpoint_pool_exhausted":
        raise AssertionError(f"VNF endpoint exhaustion failed: {detail}")
    if not endpoint_pool.release(1):
        raise AssertionError("VNF endpoint release failed")
    reused, _ = endpoint_pool.assign(3, plan)
    if reused is None:
        raise AssertionError("released VNF endpoints were not reusable")
    endpoint_pool.release(3)
    pool_snapshot = endpoint_pool.snapshot()
    if pool_snapshot["active_by_dc"] or pool_snapshot["peak_active_by_dc"] != {"7": 2}:
        raise AssertionError(f"invalid per-DC endpoint accounting: {pool_snapshot}")
    if pool_snapshot["exhaustions_by_dc"] != {"7": 1}:
        raise AssertionError(f"invalid per-DC exhaustion accounting: {pool_snapshot}")

    class FakeRyuClient:
        def __init__(self):
            self.calls = []
            self.lock = threading.Lock()

        def install_sfc_batch(self, plans):
            with self.lock:
                self.calls.append([int(plan["request_id"]) for plan in plans])
            return [
                {"accepted": True, "request_id": int(plan["request_id"])}
                for plan in plans
            ]

    fake = FakeRyuClient()
    committer = RyuBatchCommitter(fake, window_ms=20.0, max_batch_size=4)
    start = threading.Barrier(5)
    results = []
    errors = []

    def submit(request_id):
        try:
            start.wait()
            results.append(committer.submit({"request_id": request_id}))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=(request_id,), daemon=True)
        for request_id in range(1, 5)
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2.0)
    committer.close()
    if errors or any(thread.is_alive() for thread in threads):
        raise AssertionError(f"Ryu batch committer failed: {errors}")
    if len(results) != 4 or sorted(sum(fake.calls, [])) != [1, 2, 3, 4]:
        raise AssertionError(
            f"Ryu batch committer lost requests: calls={fake.calls}, results={results}"
        )
    if not any(len(call) > 1 for call in fake.calls):
        raise AssertionError(f"Ryu batch committer did not aggregate: {fake.calls}")
    ryu_snapshot = committer.snapshot()
    if (
        ryu_snapshot["requests"] != 4
        or ryu_snapshot["request_batch_size"]["max"] <= 1.0
    ):
        raise AssertionError(f"invalid Ryu batch statistics: {ryu_snapshot}")

    vnf_calls = []
    vnf_calls_lock = threading.Lock()

    def fake_vnf_transport(port, messages, timeout):
        with vnf_calls_lock:
            vnf_calls.append(
                [(int(row["payload"]["request_id"]), int(row["payload"]["stage"]))
                 for row in messages]
            )
        return {
            "ok": True,
            "acks": [
                {
                    "request_id": int(row["payload"]["request_id"]),
                    "stage": int(row["payload"]["stage"]),
                    "accepted": True,
                }
                for row in messages
            ],
            "dispatch_ms": 0.1,
            "ack_wait_ms": 0.1,
            "total_ms": 0.2,
        }

    vnf_batcher = VnfControlBatcher(
        8765,
        1.0,
        window_ms=20.0,
        max_batch_size=4,
        name="check",
        transport=fake_vnf_transport,
    )
    vnf_start = threading.Barrier(5)
    vnf_results = []
    vnf_errors = []

    def submit_vnf(request_id):
        try:
            vnf_start.wait()
            vnf_results.append(
                vnf_batcher.submit(
                    [
                        {
                            "fifo": "/tmp/check.fifo",
                            "payload": {
                                "operation": "register",
                                "request_id": request_id,
                                "stage": stage,
                            },
                        }
                        for stage in (0, 1)
                    ]
                )
            )
        except Exception as exc:
            vnf_errors.append(exc)

    vnf_threads = [
        threading.Thread(target=submit_vnf, args=(request_id,), daemon=True)
        for request_id in range(1, 5)
    ]
    for thread in vnf_threads:
        thread.start()
    vnf_start.wait()
    for thread in vnf_threads:
        thread.join(timeout=2.0)
    vnf_batcher.close()
    if vnf_errors or any(thread.is_alive() for thread in vnf_threads):
        raise AssertionError(f"VNF control batcher failed: {vnf_errors}")
    if len(vnf_results) != 4 or sorted(
        request_id for call in vnf_calls for request_id, _ in call
    ) != [1, 1, 2, 2, 3, 3, 4, 4]:
        raise AssertionError(
            f"VNF control batcher lost requests: calls={vnf_calls}, "
            f"results={vnf_results}"
        )
    if not any(len(call) > 2 for call in vnf_calls):
        raise AssertionError(f"VNF control batcher did not aggregate: {vnf_calls}")
    return {
        "setup_estimator": estimator.snapshot(),
        "capacity_gate": gate.snapshot(),
        "endpoint_pool": pool_snapshot,
        "ryu_batch_sizes": [len(call) for call in fake.calls],
        "ryu_batch": ryu_snapshot,
        "vnf_control_batch": vnf_batcher.snapshot(),
    }


def check_tree_ports(profile, outputs):
    directed_ports = set()
    for edge in profile["edges"]:
        directed_ports.add((int(edge["u"]), int(edge["u_port"])))
        directed_ports.add((int(edge["v"]), int(edge["v_port"])))
    host_ports = {
        (int(node["dpid"]), int(node["host_port"])) for node in profile["nodes"]
    }
    for raw_dpid, ports in outputs.items():
        dpid = int(raw_dpid)
        for port in ports:
            if (dpid, int(port)) not in directed_ports | host_ports:
                raise AssertionError(f"tree uses unknown output {dpid}:{port}")


def probe_loopback_check():
    result = {}
    errors = []
    with tempfile.TemporaryDirectory() as temporary:
        ready_file = Path(temporary) / "receiver.ready"
        expected_file = Path(temporary) / "sender.expected"

        def receiver():
            try:
                result.update(
                    receive_probe(
                        "127.0.0.1",
                        39001,
                        0.8,
                        "0.0.0.0",
                        100.0,
                        0.95,
                        20.0,
                        0.05,
                        grace_seconds=0.2,
                        ready_file=ready_file,
                        expected_file=expected_file,
                    )
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=receiver, daemon=True)
        thread.start()
        readiness = wait_for_ready_files([ready_file], timeout=1.0)
        if ready_file.exists():
            raise AssertionError("receiver ready marker was not consumed")
        # Keep the smoke rate below the coarse Windows sleep quantum while
        # preserving 25 scheduled packets. Production load tests use the
        # native absolute-time sender instead of this Python fallback.
        sent = send_probe("127.0.0.1", 39001, 0.625, 40.0, 128, 0)
        expected_file.write_text(str(sent["planned_packets"]), encoding="ascii")
        thread.join(timeout=2.0)
    if thread.is_alive():
        raise AssertionError("UDP probe receiver did not finish")
    if errors:
        raise errors[0]
    if sent["sent_packets"] < 20 or result.get("received_packets", 0) < 20:
        raise AssertionError(f"UDP probe lost too many loopback packets: {result}")
    if result["packet_loss_rate"] > 0.05 or not result["sla_met"]:
        raise AssertionError(f"UDP loopback SLA check failed: {result}")
    if result["expected_packets_source"] != "sender_file":
        raise AssertionError(f"receiver did not use sender packet metadata: {result}")

    result.clear()
    errors.clear()
    with tempfile.TemporaryDirectory() as temporary:
        ready_file = Path(temporary) / "receiver.ready"
        expected_file = Path(temporary) / "sender.expected"

        def underdelivery_receiver():
            try:
                result.update(
                    receive_probe(
                        "127.0.0.1",
                        39003,
                        0.35,
                        "0.0.0.0",
                        100.0,
                        0.95,
                        20.0,
                        0.05,
                        grace_seconds=0.2,
                        ready_file=ready_file,
                        expected_file=expected_file,
                    )
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=underdelivery_receiver, daemon=True)
        thread.start()
        wait_for_ready_files([ready_file], timeout=1.0)
        expected_file.write_text("100", encoding="ascii")
        underdelivered = send_probe("127.0.0.1", 39003, 0.25, 100.0, 128, 0)
        thread.join(timeout=2.0)
    if thread.is_alive():
        raise AssertionError("underdelivery receiver did not finish")
    if errors:
        raise errors[0]
    if underdelivered["sent_packets"] >= 100 or result["expected_packets"] != 100:
        raise AssertionError(f"invalid underdelivery fixture: {result}")
    if result["loss_sla_met"] or result["sla_met"]:
        raise AssertionError(f"sender underdelivery was hidden from SLA: {result}")
    stop_time_ns = time.time_ns() + 30_000_000
    deadline_limited = send_probe(
        "127.0.0.1",
        39002,
        1.0,
        1000.0,
        128,
        0,
        stop_time_ns=stop_time_ns,
    )
    if (
        deadline_limited["sent_packets"] >= deadline_limited["planned_packets"]
        or deadline_limited["deadline_limited_packets"] <= 0
        or deadline_limited["elapsed_seconds"] > 0.2
    ):
        raise AssertionError(
            f"UDP sender ignored its absolute deadline: {deadline_limited}"
        )
    bounded_catch_up = send_probe(
        "127.0.0.1",
        39004,
        0.05,
        1000.0,
        128,
        0,
        max_catch_up_packets=4,
    )
    if bounded_catch_up["max_catch_up_burst"] > 4:
        raise AssertionError(
            f"UDP sender exceeded its catch-up burst cap: {bounded_catch_up}"
        )
    return {
        "readiness": readiness,
        "sender": sent,
        "deadline_limited_sender": deadline_limited,
        "bounded_catch_up_sender": bounded_catch_up,
        "receiver": result,
    }


def main() -> int:
    requested_mbps = 5.0
    payload_bytes = 1200
    overhead_bytes = 42
    requested_pps = bandwidth_mbps_to_pps(
        requested_mbps, payload_bytes, overhead_bytes
    )
    modeled_link_mbps = (
        requested_pps * (payload_bytes + overhead_bytes) * 8.0 / 1_000_000.0
    )
    if not math.isclose(modeled_link_mbps, requested_mbps, abs_tol=1e-9):
        raise AssertionError("bandwidth-to-PPS conversion exceeds the requested link rate")

    profile = load_json(ROOT / "sdn" / "topologies" / "us_backbone_28.json")
    qos = load_json(ROOT / "sdn" / "qos_profiles.json")
    vnfs = load_json(ROOT / "sdn" / "vnf_catalog.json")
    kwargs = {
        "seed": 7501,
        "duration": 8.0,
        "per_source_rate": 1.0,
        "destination_count": 5,
        "chain_length": 3,
    }
    first = generate_requests(profile, qos, vnfs, **kwargs)
    second = generate_requests(profile, qos, vnfs, **kwargs)
    if first != second or not first:
        raise AssertionError("request generation is empty or non-deterministic")

    worker_plan = probe_agent_worker_plan(first, profile)
    if (
        worker_plan["sender_workers"] < worker_plan["max_sender_concurrency"]
        or worker_plan["receiver_workers"] < worker_plan["max_receiver_concurrency"]
    ):
        raise AssertionError(f"worker plan underprovisions the trace: {worker_plan}")
    high_rate = generate_requests(
        profile,
        qos,
        vnfs,
        seed=kwargs["seed"],
        duration=8.0,
        per_source_rate=6.0,
        destination_count=5,
        chain_length=3,
    )
    high_rate_plan = probe_agent_worker_plan(high_rate, profile)
    if (
        high_rate_plan["sender_workers"] < worker_plan["sender_workers"]
        or high_rate_plan["receiver_workers"] < worker_plan["receiver_workers"]
    ):
        raise AssertionError("worker pools did not scale with the arrival rate")

    cogent_profile = load_json(ROOT / "sdn" / "topologies" / "cogentco_197.json")
    cogent_requests = generate_requests(
        cogent_profile,
        qos,
        vnfs,
        seed=7502,
        duration=2.0,
        per_source_rate=0.1,
        destination_count=5,
        chain_length=3,
    )
    cogent_worker_plan = probe_agent_worker_plan(cogent_requests, cogent_profile)
    if not cogent_worker_plan["sender_peaks_by_host"]:
        raise AssertionError("cross-topology worker plan has no source hosts")

    changed_qos = copy.deepcopy(qos)
    changed_qos["classes"]["Q0_REALTIME"]["weight"] = 0.0
    changed_qos["classes"]["Q1_INTERACTIVE"]["weight"] = 0.0
    changed_qos["classes"]["Q2_ELASTIC"]["weight"] = 1.0
    qos_variant = generate_requests(profile, changed_qos, vnfs, **kwargs)
    arrival_identity = [
        (row["arrival_time"], row["source_dpid"]) for row in first
    ]
    if arrival_identity != [
        (row["arrival_time"], row["source_dpid"]) for row in qos_variant
    ]:
        raise AssertionError("QoS changes altered the Poisson arrival stream")

    if len({row["multicast_ip"] for row in first}) != len(first):
        raise AssertionError("multicast addresses are not unique")
    if len({row["udp_port"] for row in first}) != len(first):
        raise AssertionError("UDP ports are not unique")
    if any(len(set(row["vnf"])) != len(row["vnf"]) for row in first):
        raise AssertionError("a generated chain repeats a VNF type")
    if any(not row["cpu_origin"] or not row["memory_origin"] for row in first):
        raise AssertionError("generated requests lack VNF resource demands")

    events = request_events(first)
    if len(events) != 2 * len(first):
        raise AssertionError("arrival/leave event count mismatch")
    if events != sorted(
        events, key=lambda item: (item["time"], 0 if item["type"] == "leave" else 1)
    ):
        raise AssertionError("request events are not chronologically sorted")

    tree_checks = []
    for request in first[:10]:
        outputs, paths = shortest_tree_outputs(
            profile, request["source_dpid"], request["destination_dpids"]
        )
        check_tree_ports(profile, outputs)
        for destination in request["destination_dpids"]:
            path = paths[str(destination)]
            if path[0] != request["source_dpid"] or path[-1] != destination:
                raise AssertionError(f"invalid source-rooted path {path}")
        tree_checks.append({"request_id": request["id"], "switches": len(outputs)})

    with tempfile.TemporaryDirectory() as temporary:
        trace = write_trace(
            Path(temporary),
            first,
            events,
            {"version": "check", "seed": kwargs["seed"]},
        )
        loaded = [
            json.loads(line)
            for line in (Path(temporary) / "requests.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        if loaded != first or len(trace["files"]) != 3:
            raise AssertionError("trace serialization round trip failed")

    probe = probe_loopback_check()
    controller_rest_discovery = controller_rest_discovery_check()
    remote_deadline = remote_deadline_check()
    sla_runtime_contract = sla_runtime_contract_check()
    deployment_scheduler = deployment_scheduler_check()
    candidate_selection = sfc_candidate_selection_check()
    result = {
        "ok": True,
        "bandwidth_rate_check": {
            "requested_mbps": requested_mbps,
            "packets_per_second": requested_pps,
            "modeled_link_mbps": modeled_link_mbps,
        },
        "requests": len(first),
        "events": len(events),
        "source_nodes": sorted({row["source_dpid"] for row in first}),
        "qos_classes": sorted({row["qos_class"] for row in first}),
        "tree_checks": tree_checks,
        "worker_plan": worker_plan,
        "high_rate_worker_plan": high_rate_plan,
        "cogent_worker_plan": {
            "requests": len(cogent_requests),
            "sender_workers": cogent_worker_plan["sender_workers"],
            "receiver_workers": cogent_worker_plan["receiver_workers"],
            "max_sender_concurrency": cogent_worker_plan[
                "max_sender_concurrency"
            ],
            "max_receiver_concurrency": cogent_worker_plan[
                "max_receiver_concurrency"
            ],
        },
        "probe": probe,
        "controller_rest_discovery": controller_rest_discovery,
        "remote_deadline": remote_deadline,
        "sla_runtime_contract": sla_runtime_contract,
        "deployment_scheduler": deployment_scheduler,
        "sfc_candidate_selection": candidate_selection,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
