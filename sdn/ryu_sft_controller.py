#!/usr/bin/env python3
"""Ryu controller for SFT-aware multicast forwarding and safe rerouting.

The controller is deliberately separated from the QMIX policy. A policy or
the existing Python reconfiguration manager sends a validated tree proposal
to the REST API; this app monitors the real OVS counters and applies the
proposal atomically at the OpenFlow level.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from ryu.app import simple_switch_13
from ryu.app.wsgi import ControllerBase, WSGIApplication, Response, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER,
    DEAD_DISPATCHER,
    MAIN_DISPATCHER,
    set_ev_cls,
)
from ryu.lib import hub
from ryu.lib.packet import ethernet, packet
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.topology.api import get_link, get_switch


APP_NAME = "sft_controller"
DEFAULT_CAPACITY_BPS = 100_000_000.0
DEFAULT_THRESHOLD = 0.95


def _json_response(payload, status=200):
    return Response(
        status=status,
        content_type="application/json",
        body=json.dumps(payload, sort_keys=True).encode("utf-8"),
    )


class SFTControllerRest(ControllerBase):
    """Small REST adapter used by PyCharm and the QMIX executor."""

    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data["sft_app"]

    @route(APP_NAME, "/sft/status", methods=["GET"])
    def status(self, req, **kwargs):
        return _json_response(self.app.status())

    @route(APP_NAME, "/sft/config", methods=["POST"])
    def config(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            return _json_response(self.app.configure(payload))
        except (ValueError, TypeError) as exc:
            return _json_response({"error": str(exc)}, status=400)

    @route(APP_NAME, "/sft/group", methods=["POST"])
    def group(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            return _json_response(self.app.apply_tree(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/group/{group_id}", methods=["DELETE"])
    def delete_group(self, req, group_id, **kwargs):
        del req, kwargs
        try:
            return _json_response(self.app.delete_tree(group_id))
        except (TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except KeyError as exc:
            return _json_response({"error": str(exc)}, status=404)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/reroute", methods=["POST"])
    def reroute(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            payload["reroute"] = True
            return _json_response(self.app.apply_tree(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/sfc", methods=["POST"])
    def sfc(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            return _json_response(self.app.apply_sfc(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/sfc/batch", methods=["POST"])
    def sfc_batch(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            return _json_response(self.app.apply_sfc_batch(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/sfc/migration/prepare", methods=["POST"])
    def prepare_sfc_migration(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            return _json_response(self.app.prepare_sfc_migration(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/sfc/migration/commit", methods=["POST"])
    def commit_sfc_migration(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            return _json_response(self.app.commit_sfc_migration(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/sfc/migration/abort", methods=["POST"])
    def abort_sfc_migration(self, req, **kwargs):
        try:
            payload = json.loads(req.body.decode("utf-8") or "{}")
            return _json_response(self.app.abort_sfc_migration(payload))
        except (KeyError, TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)

    @route(APP_NAME, "/sft/sfc/{request_id}", methods=["DELETE"])
    def delete_sfc(self, req, request_id, **kwargs):
        del req, kwargs
        try:
            return _json_response(self.app.delete_sfc(request_id))
        except (TypeError, ValueError) as exc:
            return _json_response({"error": str(exc)}, status=400)
        except KeyError as exc:
            return _json_response({"error": str(exc)}, status=404)
        except RuntimeError as exc:
            return _json_response({"error": str(exc)}, status=409)


class SFTController(simple_switch_13.SimpleSwitch13):
    """Learning switch plus SFT multicast groups and port utilization."""

    _CONTEXTS = {"wsgi": WSGIApplication}
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = APP_NAME
        self.wsgi = kwargs["wsgi"]
        self.wsgi.register(SFTControllerRest, {"sft_app": self})

        self.datapaths = {}
        self.links = {}
        self.discovered_links = {}
        self.static_links = {}
        self.topology_profile = None
        self.mac_to_port = defaultdict(dict)
        self.port_stats = defaultdict(dict)
        self.capacity_bps = {}
        self.default_capacity_bps = DEFAULT_CAPACITY_BPS
        self.link_util_threshold = DEFAULT_THRESHOLD
        self.stats_interval = 1.0
        self.barrier_timeout = 2.0
        self.sfc_barrier_mode = "staged"
        self.groups = {}
        self.sfcs = {}
        self.pending_sfc_migrations = {}
        self._version = defaultdict(int)
        self._barrier_waiters = {}
        self.monitor_thread = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        if self.topology_profile:
            # Static SFT experiments install every permitted forwarding path.
            # Dropping table misses prevents broadcast storms on cyclic WANs.
            instructions = []
        else:
            actions = [
                parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)
            ]
            instructions = [
                parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)
            ]
        datapath.send_msg(
            parser.OFPFlowMod(
                datapath=datapath,
                priority=0,
                match=match,
                instructions=instructions,
            )
        )

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if self.datapaths.get(datapath.id) is datapath:
                self.datapaths.pop(datapath.id, None)

    @set_ev_cls(event.EventSwitchEnter)
    @set_ev_cls(event.EventSwitchLeave)
    @set_ev_cls(event.EventLinkAdd)
    @set_ev_cls(event.EventLinkDelete)
    def topology_change_handler(self, ev):
        del ev
        self._refresh_topology()

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        now = time.monotonic()
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            port_no = int(stat.port_no)
            if port_no > ofproto_v1_3.OFPP_MAX:
                continue
            previous = self.port_stats[dpid].get(port_no)
            current = {
                "rx_bytes": int(stat.rx_bytes),
                "tx_bytes": int(stat.tx_bytes),
                "timestamp": now,
            }
            if previous:
                elapsed = max(now - previous["timestamp"], 1e-6)
                rx_delta = max(0, current["rx_bytes"] - previous["rx_bytes"])
                tx_delta = max(0, current["tx_bytes"] - previous["tx_bytes"])
                capacity = self._capacity(dpid, port_no)
                current["rx_bps"] = rx_delta * 8.0 / elapsed
                current["tx_bps"] = tx_delta * 8.0 / elapsed
                current["utilization"] = max(
                    current["rx_bps"], current["tx_bps"]
                ) / capacity
            self.port_stats[dpid][port_no] = current

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier_reply_handler(self, ev):
        key = (int(ev.msg.datapath.id), int(ev.msg.xid))
        waiter = self._barrier_waiters.pop(key, None)
        if waiter is not None:
            waiter.set()

    def _monitor(self):
        while True:
            for datapath in list(self.datapaths.values()):
                parser = datapath.ofproto_parser
                datapath.send_msg(
                    parser.OFPPortStatsRequest(
                        datapath, 0, datapath.ofproto.OFPP_ANY
                    )
                )
            hub.sleep(self.stats_interval)

    def _refresh_topology(self):
        try:
            switches = get_switch(self, None)
            links = get_link(self, None)
        except Exception:
            return
        self.discovered_links = {
            f"{link.src.dpid}:{link.src.port_no}->{link.dst.dpid}": {
                "src_dpid": int(link.src.dpid),
                "src_port": int(link.src.port_no),
                "dst_dpid": int(link.dst.dpid),
                "dst_port": int(link.dst.port_no),
            }
            for link in links
        }
        if not self.static_links:
            self.links = dict(self.discovered_links)
        for switch in switches:
            self.datapaths.setdefault(int(switch.dp.id), switch.dp)

    def _load_topology_profile(self, raw_path):
        if raw_path in (None, ""):
            self.topology_profile = None
            self.static_links = {}
            self.links = dict(self.discovered_links)
            return

        path = Path(str(raw_path)).expanduser().resolve()
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load topology profile {path}: {exc}")

        nodes = profile.get("nodes")
        edges = profile.get("edges")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("topology profile must contain a non-empty nodes list")
        if not isinstance(edges, list) or not edges:
            raise ValueError("topology profile must contain a non-empty edges list")

        dpids = {int(node["dpid"]) for node in nodes}
        if len(dpids) != len(nodes) or any(dpid <= 0 for dpid in dpids):
            raise ValueError("topology profile contains duplicate or invalid DPIDs")
        if int(profile.get("node_count", len(nodes))) != len(nodes):
            raise ValueError("topology node_count does not match nodes")

        static_links = {}
        capacities = {}
        for edge in edges:
            u = int(edge["u"])
            v = int(edge["v"])
            u_port = int(edge["u_port"])
            v_port = int(edge["v_port"])
            if u not in dpids or v not in dpids or u == v:
                raise ValueError(f"invalid topology edge {u}-{v}")
            if u_port <= 0 or v_port <= 0:
                raise ValueError(f"invalid topology ports for edge {u}-{v}")
            for src, src_port, dst, dst_port in (
                (u, u_port, v, v_port),
                (v, v_port, u, u_port),
            ):
                key = f"{src}:{src_port}->{dst}"
                if key in static_links:
                    raise ValueError(f"duplicate topology link {key}")
                static_links[key] = {
                    "src_dpid": src,
                    "src_port": src_port,
                    "dst_dpid": dst,
                    "dst_port": dst_port,
                }
            bandwidth_mbps = float(
                edge.get("bandwidth_mbps", profile.get("default_bandwidth_mbps", 0))
            )
            if bandwidth_mbps <= 0:
                raise ValueError(f"invalid bandwidth for edge {u}-{v}")
            capacities[f"{u}:{u_port}"] = bandwidth_mbps * 1e6
            capacities[f"{v}:{v_port}"] = bandwidth_mbps * 1e6

        expected_edges = int(profile.get("physical_link_count", len(edges)))
        if expected_edges != len(edges):
            raise ValueError("topology physical_link_count does not match edges")

        self.topology_profile = {
            "name": str(profile.get("name", path.stem)),
            "path": str(path),
            "node_count": len(nodes),
            "physical_link_count": len(edges),
            "dpids": sorted(dpids),
        }
        self.static_links = static_links
        self.links = dict(static_links)
        self.capacity_bps.update(capacities)
        if "default_bandwidth_mbps" in profile:
            self.default_capacity_bps = float(profile["default_bandwidth_mbps"]) * 1e6

    def _capacity(self, dpid, port_no):
        return float(
            self.capacity_bps.get(f"{int(dpid)}:{int(port_no)}", self.default_capacity_bps)
        )

    def _hotspots(self):
        result = []
        for dpid, ports in self.port_stats.items():
            for port_no, sample in ports.items():
                utilization = sample.get("utilization")
                if utilization is not None and utilization >= self.link_util_threshold:
                    result.append(
                        {
                            "dpid": int(dpid),
                            "port": int(port_no),
                            "utilization": round(float(utilization), 6),
                            "tx_bps": round(float(sample.get("tx_bps", 0.0)), 2),
                            "rx_bps": round(float(sample.get("rx_bps", 0.0)), 2),
                        }
                    )
        return sorted(result, key=lambda item: item["utilization"], reverse=True)

    def status(self):
        discovered_keys = set(self.discovered_links)
        static_keys = set(self.static_links)
        topology = {
            "mode": "static_profile" if self.topology_profile else "dynamic_lldp",
            "profile": self.topology_profile,
            "effective_directed_links": len(self.links),
            "discovered_directed_links": len(self.discovered_links),
            "observed_profile_links": len(static_keys & discovered_keys),
            "unobserved_profile_links": len(static_keys - discovered_keys),
            "unexpected_discovered_links": len(discovered_keys - static_keys)
            if static_keys
            else 0,
        }
        return {
            "datapaths": sorted(int(dpid) for dpid in self.datapaths),
            "links": [self.links[key] for key in sorted(self.links)],
            "discovered_links": [
                self.discovered_links[key] for key in sorted(self.discovered_links)
            ],
            "topology": topology,
            "hotspots": self._hotspots(),
            "threshold": self.link_util_threshold,
            "default_capacity_bps": self.default_capacity_bps,
            "barrier_timeout": self.barrier_timeout,
            "sfc_barrier_mode": self.sfc_barrier_mode,
            "port_stats": {
                str(dpid): {str(port): sample for port, sample in ports.items()}
                for dpid, ports in self.port_stats.items()
            },
            "groups": self.groups,
            "sfcs": self.sfcs,
            "pending_sfc_migrations": self.pending_sfc_migrations,
        }

    def configure(self, payload):
        if "topology_file" in payload:
            self._load_topology_profile(payload["topology_file"])
        if "threshold" in payload:
            threshold = float(payload["threshold"])
            if not 0.0 < threshold <= 1.0:
                raise ValueError("threshold must be in (0, 1]")
            self.link_util_threshold = threshold
        if "default_capacity_bps" in payload:
            capacity = float(payload["default_capacity_bps"])
            if capacity <= 0:
                raise ValueError("default_capacity_bps must be positive")
            self.default_capacity_bps = capacity
        for key, value in payload.get("capacity_bps", {}).items():
            if float(value) <= 0:
                raise ValueError(f"capacity must be positive for {key}")
            self.capacity_bps[str(key)] = float(value)
        if "stats_interval" in payload:
            interval = float(payload["stats_interval"])
            if interval <= 0:
                raise ValueError("stats_interval must be positive")
            self.stats_interval = interval
        if "barrier_timeout" in payload:
            timeout = float(payload["barrier_timeout"])
            if timeout <= 0:
                raise ValueError("barrier_timeout must be positive")
            self.barrier_timeout = timeout
        if "sfc_barrier_mode" in payload:
            mode = str(payload["sfc_barrier_mode"])
            if mode not in {"staged", "single"}:
                raise ValueError("sfc_barrier_mode must be 'staged' or 'single'")
            self.sfc_barrier_mode = mode
        return self.status()

    @staticmethod
    def _normalize_outputs(raw_outputs):
        outputs = {}
        for raw_dpid, raw_ports in raw_outputs.items():
            dpid = int(raw_dpid)
            ports = sorted({int(port) for port in raw_ports})
            if not ports:
                raise ValueError(f"no output ports for switch {dpid}")
            if any(port <= 0 or port > ofproto_v1_3.OFPP_MAX for port in ports):
                raise ValueError(f"invalid output port for switch {dpid}")
            outputs[str(dpid)] = ports
        if not outputs:
            raise ValueError("switch_outputs cannot be empty")
        return outputs

    def _group_mod(self, datapath, group_id, ports, command):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        buckets = [
            parser.OFPBucket(actions=[parser.OFPActionOutput(port)])
            for port in ports
        ]
        datapath.send_msg(
            parser.OFPGroupMod(
                datapath,
                command,
                ofp.OFPGT_ALL,
                int(group_id),
                buckets,
            )
        )

    def _flow_for_group(self, datapath, dst_ip, group_id, command):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=dst_ip)
        instructions = [
            parser.OFPInstructionActions(
                ofp.OFPIT_APPLY_ACTIONS,
                [parser.OFPActionGroup(int(group_id))],
            )
        ]
        datapath.send_msg(
            parser.OFPFlowMod(
                datapath=datapath,
                command=command,
                priority=200,
                match=match,
                instructions=instructions,
            )
        )

    def _delete_flow_for_group(self, datapath, dst_ip, group_id):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        datapath.send_msg(
            parser.OFPFlowMod(
                datapath=datapath,
                command=ofp.OFPFC_DELETE_STRICT,
                priority=200,
                match=parser.OFPMatch(eth_type=0x0800, ipv4_dst=str(dst_ip)),
                out_port=ofp.OFPP_ANY,
                out_group=int(group_id),
            )
        )

    @staticmethod
    def _segment_cookie(request_id, stage):
        request_id = int(request_id)
        stage = int(stage)
        if request_id <= 0 or request_id > 0xFFFFFFFF or stage < 0 or stage > 0xFFFF:
            raise ValueError("request_id/stage exceeds the SFC cookie range")
        return 0x5FC0000000000000 | (request_id << 16) | stage

    def _flow_for_segment(
        self, datapath, target_ip, udp_port, output_port, command, cookie
    ):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(
            eth_type=0x0800,
            ip_proto=17,
            ipv4_dst=str(target_ip),
            udp_dst=int(udp_port),
        )
        instructions = [
            parser.OFPInstructionActions(
                ofp.OFPIT_APPLY_ACTIONS,
                [parser.OFPActionOutput(int(output_port))],
            )
        ]
        datapath.send_msg(
            parser.OFPFlowMod(
                datapath=datapath,
                command=command,
                cookie=int(cookie),
                priority=300,
                match=match,
                instructions=instructions,
            )
        )

    def _delete_segment_flow(self, datapath, target_ip, udp_port, cookie):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        datapath.send_msg(
            parser.OFPFlowMod(
                datapath=datapath,
                command=ofp.OFPFC_DELETE_STRICT,
                cookie=int(cookie),
                cookie_mask=0xFFFFFFFFFFFFFFFF,
                priority=300,
                match=parser.OFPMatch(
                    eth_type=0x0800,
                    ip_proto=17,
                    ipv4_dst=str(target_ip),
                    udp_dst=int(udp_port),
                ),
                out_port=ofp.OFPP_ANY,
                out_group=ofp.OFPG_ANY,
            )
        )

    def _normalize_segments(self, request_id, raw_segments):
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("segments must be a non-empty list")
        segments = []
        seen_stages = set()
        for raw in raw_segments:
            stage = int(raw["stage"])
            if stage in seen_stages:
                raise ValueError(f"duplicate SFC stage {stage}")
            seen_stages.add(stage)
            target_ip = str(raw["target_ip"])
            udp_port = int(raw["udp_port"])
            if not 1 <= udp_port <= 65535:
                raise ValueError(f"invalid UDP port for SFC stage {stage}")
            path = [int(value) for value in raw["path"]]
            if not path or len(path) != len(set(path)):
                raise ValueError(f"stage {stage} path is empty or cyclic")
            outputs = self._normalize_outputs(raw["switch_outputs"])
            if set(outputs) != {str(dpid) for dpid in path}:
                raise ValueError(f"stage {stage} outputs do not match its path")
            if any(len(outputs[str(dpid)]) != 1 for dpid in path):
                raise ValueError(f"stage {stage} must have one output per switch")
            segments.append(
                {
                    "stage": stage,
                    "target_ip": target_ip,
                    "udp_port": udp_port,
                    "path": path,
                    "switch_outputs": outputs,
                    "cookie": self._segment_cookie(request_id, stage),
                }
            )
        segments.sort(key=lambda value: value["stage"])
        if [value["stage"] for value in segments] != list(range(len(segments))):
            raise ValueError("SFC stages must be contiguous and start at zero")
        return segments

    def _remove_segment_flows(self, segments):
        datapaths = []
        for segment in reversed(segments):
            for dpid in reversed(segment["path"]):
                datapath = self.datapaths.get(int(dpid))
                if datapath is None:
                    continue
                self._delete_segment_flow(
                    datapath,
                    segment["target_ip"],
                    segment["udp_port"],
                    segment["cookie"],
                )
                datapaths.append(datapath)
        self._barriers(datapaths)

    def apply_sfc(self, payload, defer_barriers=False):
        request_id = int(payload["request_id"])
        request_key = str(request_id)
        if request_key in self.sfcs:
            raise ValueError(f"SFC request {request_id} is already installed")
        segments = self._normalize_segments(request_id, payload["segments"])
        multicast = dict(payload["multicast"])
        group_id = int(multicast.get("group_id", request_id))
        root_dpid = int(multicast["root_dpid"])
        if segments[-1]["path"][-1] != root_dpid:
            raise ValueError("last SFC segment must terminate at multicast root")
        required = {
            int(dpid)
            for segment in segments
            for dpid in segment["path"]
        }
        required.update(int(dpid) for dpid in multicast["switch_outputs"])
        missing = sorted(required - set(self.datapaths))
        if missing:
            raise RuntimeError(f"switches are not connected: {missing}")

        try:
            single_barrier = defer_barriers or self.sfc_barrier_mode == "single"
            for segment in reversed(segments):
                for dpid in reversed(segment["path"]):
                    datapath = self.datapaths[dpid]
                    self._flow_for_segment(
                        datapath,
                        segment["target_ip"],
                        segment["udp_port"],
                        segment["switch_outputs"][str(dpid)][0],
                        datapath.ofproto.OFPFC_ADD,
                        segment["cookie"],
                    )
                if not single_barrier:
                    self._barriers(
                        [self.datapaths[dpid] for dpid in segment["path"]]
                    )
            tree = self.apply_tree(
                {
                    "group_id": group_id,
                    "dst_ip": multicast["dst_ip"],
                    "switch_outputs": multicast["switch_outputs"],
                    "source_dpid": root_dpid,
                },
                defer_barriers=single_barrier,
            )
            if single_barrier and not defer_barriers:
                # Every switch confirms all preceding group and flow updates on
                # its OpenFlow connection before the REST request completes.
                self._barriers([self.datapaths[dpid] for dpid in required])
        except Exception:
            self._remove_segment_flows(segments)
            if str(group_id) in self.groups:
                try:
                    self.delete_tree(group_id)
                except Exception:
                    pass
            raise

        self.sfcs[request_key] = {
            "request_id": request_id,
            "group_id": group_id,
            "segments": segments,
            "multicast": {
                "root_dpid": root_dpid,
                "dst_ip": str(multicast["dst_ip"]),
                "switch_outputs": tree["switch_outputs"],
            },
            "installed_at": time.time(),
        }
        return {
            "accepted": True,
            "request_id": request_id,
            "group_id": group_id,
            "segments_installed": len(segments),
            "segment_switch_rules": sum(len(value["path"]) for value in segments),
            "barrier_mode": self.sfc_barrier_mode,
            "multicast": tree,
        }

    def apply_sfc_batch(self, payload):
        requests = payload.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("SFC batch requires a non-empty requests list")
        request_ids = [int(item["request_id"]) for item in requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("SFC batch contains duplicate request IDs")

        installed = []
        results = []
        required = set()
        try:
            for item in requests:
                result = self.apply_sfc(item, defer_barriers=True)
                installed.append(int(item["request_id"]))
                results.append(result)
                current = self.sfcs[str(int(item["request_id"]))]
                for segment in current["segments"]:
                    required.update(int(dpid) for dpid in segment["path"])
                required.update(
                    int(dpid)
                    for dpid in current["multicast"]["switch_outputs"]
                )
            self._barriers([self.datapaths[dpid] for dpid in required])
        except Exception:
            for request_id in reversed(installed):
                if str(request_id) not in self.sfcs:
                    continue
                try:
                    self.delete_sfc(request_id)
                except Exception:
                    pass
            raise

        for result in results:
            result["barrier_mode"] = "batch_single"
            result["batch_size"] = len(results)
        return {
            "accepted": True,
            "batch_size": len(results),
            "switches_committed": len(required),
            "results": results,
        }

    def delete_sfc(self, request_id):
        request_key = str(int(request_id))
        current = self.sfcs.get(request_key)
        if current is None:
            raise KeyError(f"unknown SFC request {request_key}")
        tree_result = None
        tree_error = None
        try:
            tree_result = self.delete_tree(current["group_id"])
        except Exception as exc:
            tree_error = exc
        self._remove_segment_flows(current["segments"])
        self.sfcs.pop(request_key, None)
        if tree_error is not None:
            raise RuntimeError(f"SFC segment rules removed but tree deletion failed: {tree_error}")
        return {
            "accepted": True,
            "deleted": True,
            "request_id": int(request_key),
            "segments_deleted": len(current["segments"]),
            "multicast": tree_result,
        }

    @staticmethod
    def _segment_rule_map(segments):
        rules = {}
        for segment in segments:
            match = (str(segment["target_ip"]), int(segment["udp_port"]))
            for dpid in segment["path"]:
                key = (int(dpid), *match)
                output = int(segment["switch_outputs"][str(dpid)][0])
                if key in rules and rules[key]["output_port"] != output:
                    raise ValueError(f"conflicting SFC rule in one plan: {key}")
                rules[key] = {
                    "dpid": int(dpid),
                    "target_ip": match[0],
                    "udp_port": match[1],
                    "output_port": output,
                    "cookie": int(segment["cookie"]),
                }
        return rules

    def prepare_sfc_migration(self, payload):
        """Stage rules for a new VNF endpoint without touching the live path."""
        started = time.monotonic()
        request_id = int(payload["request_id"])
        request_key = str(request_id)
        if request_key in self.pending_sfc_migrations:
            raise ValueError(f"SFC request {request_id} already has a pending migration")
        current = self.sfcs.get(request_key)
        if current is None:
            raise KeyError(f"unknown SFC request {request_id}")
        segments = self._normalize_segments(request_id, payload["segments"])
        if len(segments) != len(current["segments"]):
            raise ValueError("online migration cannot change the SFC length")
        multicast = dict(payload.get("multicast") or current["multicast"])
        multicast.setdefault("group_id", current["group_id"])
        if int(multicast["group_id"]) != int(current["group_id"]):
            raise ValueError("online migration cannot change the logical group ID")
        normalized_multicast_outputs = self._normalize_outputs(
            multicast["switch_outputs"]
        )
        if (
            int(multicast["root_dpid"]) != int(current["multicast"]["root_dpid"])
            or str(multicast["dst_ip"]) != str(current["multicast"]["dst_ip"])
            or normalized_multicast_outputs
            != current["multicast"]["switch_outputs"]
        ):
            raise ValueError(
                "VNF migration must keep the multicast tree fixed; reroute it "
                "with the separate make-before-break tree API"
            )
        if segments[-1]["path"][-1] != int(multicast["root_dpid"]):
            raise ValueError("last migrated SFC segment must terminate at multicast root")

        old_rules = self._segment_rule_map(current["segments"])
        new_rules = self._segment_rule_map(segments)
        for key in set(old_rules) & set(new_rules):
            if old_rules[key]["output_port"] != new_rules[key]["output_port"]:
                raise ValueError(
                    "migration would overwrite a live overlapping rule at "
                    f"switch {key[0]} for {key[1]}:{key[2]}"
                )
        staged_keys = sorted(set(new_rules) - set(old_rules))
        required = {new_rules[key]["dpid"] for key in staged_keys}
        missing = sorted(required - set(self.datapaths))
        if missing:
            raise RuntimeError(f"switches are not connected: {missing}")
        installed = []
        try:
            for key in staged_keys:
                rule = new_rules[key]
                datapath = self.datapaths[rule["dpid"]]
                self._flow_for_segment(
                    datapath,
                    rule["target_ip"],
                    rule["udp_port"],
                    rule["output_port"],
                    datapath.ofproto.OFPFC_ADD,
                    rule["cookie"],
                )
                installed.append(key)
            self._barriers([self.datapaths[dpid] for dpid in required])
        except Exception:
            for key in reversed(installed):
                rule = new_rules[key]
                datapath = self.datapaths.get(rule["dpid"])
                if datapath is not None:
                    self._delete_segment_flow(
                        datapath, rule["target_ip"], rule["udp_port"], rule["cookie"]
                    )
            self._barriers(
                [self.datapaths[dpid] for dpid in required if dpid in self.datapaths]
            )
            raise
        token = str(payload.get("migration_token") or f"{request_id}-{time.time_ns()}")
        pending = {
            "request_id": request_id,
            "migration_token": token,
            "old": current,
            "segments": segments,
            "multicast": multicast,
            "staged_rule_keys": [list(key) for key in staged_keys],
            "prepared_at": time.time(),
            "prepare_ms": (time.monotonic() - started) * 1000.0,
        }
        self.pending_sfc_migrations[request_key] = pending
        return {
            "accepted": True,
            "prepared": True,
            "request_id": request_id,
            "migration_token": token,
            "staged_rules": len(staged_keys),
            "switches_prepared": len(required),
            "prepare_ms": pending["prepare_ms"],
        }

    def commit_sfc_migration(self, payload):
        """Commit after the VNF agent has switched its upstream endpoint."""
        started = time.monotonic()
        request_id = int(payload["request_id"])
        request_key = str(request_id)
        pending = self.pending_sfc_migrations.get(request_key)
        if pending is None:
            raise KeyError(f"no pending migration for SFC request {request_id}")
        if str(payload.get("migration_token")) != pending["migration_token"]:
            raise ValueError("migration token does not match the prepared operation")
        drain_seconds = float(payload.get("drain_seconds", 0.01))
        if not 0.0 <= drain_seconds <= 10.0:
            raise ValueError("drain_seconds must be in [0, 10]")
        old_rules = self._segment_rule_map(pending["old"]["segments"])
        new_rules = self._segment_rule_map(pending["segments"])
        old_only = sorted(set(old_rules) - set(new_rules))
        if drain_seconds:
            hub.sleep(drain_seconds)
        touched = set()
        for key in old_only:
            rule = old_rules[key]
            datapath = self.datapaths.get(rule["dpid"])
            if datapath is None:
                continue
            self._delete_segment_flow(
                datapath, rule["target_ip"], rule["udp_port"], rule["cookie"]
            )
            touched.add(rule["dpid"])
        self._barriers([self.datapaths[dpid] for dpid in touched])

        multicast = pending["multicast"]
        multicast_changed = False
        tree = self.groups[str(int(pending["old"]["group_id"]))]
        self.sfcs[request_key] = {
            "request_id": request_id,
            "group_id": int(pending["old"]["group_id"]),
            "segments": pending["segments"],
            "multicast": {
                "root_dpid": int(multicast["root_dpid"]),
                "dst_ip": str(multicast["dst_ip"]),
                "switch_outputs": self._normalize_outputs(multicast["switch_outputs"]),
            },
            "installed_at": pending["old"].get("installed_at", time.time()),
            "migrated_at": time.time(),
        }
        self.pending_sfc_migrations.pop(request_key, None)
        return {
            "accepted": True,
            "committed": True,
            "request_id": request_id,
            "migration_token": pending["migration_token"],
            "old_rules_removed": len(old_only),
            "multicast_changed": multicast_changed,
            "drain_seconds": drain_seconds,
            "prepare_ms": pending["prepare_ms"],
            "commit_ms": (time.monotonic() - started) * 1000.0,
            "multicast": tree,
        }

    def abort_sfc_migration(self, payload):
        request_id = int(payload["request_id"])
        request_key = str(request_id)
        pending = self.pending_sfc_migrations.get(request_key)
        if pending is None:
            raise KeyError(f"no pending migration for SFC request {request_id}")
        if str(payload.get("migration_token")) != pending["migration_token"]:
            raise ValueError("migration token does not match the prepared operation")
        old_rules = self._segment_rule_map(pending["old"]["segments"])
        new_rules = self._segment_rule_map(pending["segments"])
        staged = sorted(set(new_rules) - set(old_rules))
        touched = set()
        for key in staged:
            rule = new_rules[key]
            datapath = self.datapaths.get(rule["dpid"])
            if datapath is None:
                continue
            self._delete_segment_flow(
                datapath, rule["target_ip"], rule["udp_port"], rule["cookie"]
            )
            touched.add(rule["dpid"])
        self._barriers([self.datapaths[dpid] for dpid in touched])
        self.pending_sfc_migrations.pop(request_key, None)
        return {
            "accepted": True,
            "aborted": True,
            "request_id": request_id,
            "migration_token": pending["migration_token"],
            "staged_rules_removed": len(staged),
        }

    def _barrier(self, datapath, timeout=None):
        self._barriers([datapath], timeout=timeout)

    def _barriers(self, datapaths, timeout=None):
        timeout = self.barrier_timeout if timeout is None else float(timeout)
        unique = {int(datapath.id): datapath for datapath in datapaths}
        pending = []
        for dpid in sorted(unique):
            datapath = unique[dpid]
            request = datapath.ofproto_parser.OFPBarrierRequest(datapath)
            datapath.set_xid(request)
            key = (dpid, int(request.xid))
            waiter = hub.Event()
            self._barrier_waiters[key] = waiter
            pending.append((dpid, key, waiter))
            if not datapath.send_msg(request):
                for _, pending_key, _ in pending:
                    self._barrier_waiters.pop(pending_key, None)
                raise RuntimeError(f"failed to send barrier to switch {dpid}")

        deadline = time.monotonic() + timeout
        timed_out = []
        for dpid, key, waiter in pending:
            remaining = max(0.0, deadline - time.monotonic())
            completed = waiter.wait(timeout=remaining)
            self._barrier_waiters.pop(key, None)
            if completed is not True:
                timed_out.append(dpid)
        if timed_out:
            raise RuntimeError(f"barrier timeout on switches {timed_out}")

    def apply_tree(self, payload, defer_barriers=False):
        group_key = str(int(payload["group_id"]))
        dst_ip = str(payload["dst_ip"])
        outputs = self._normalize_outputs(payload["switch_outputs"])
        source_dpid = payload.get("source_dpid")
        source_key = str(int(source_dpid)) if source_dpid is not None else None
        if source_key is not None and source_key not in outputs:
            raise ValueError("source_dpid must be present in switch_outputs")
        drain_seconds = float(payload.get("drain_seconds", 0.0))
        if drain_seconds < 0.0 or drain_seconds > 10.0:
            raise ValueError("drain_seconds must be in [0, 10]")
        unknown = sorted(
            int(dpid) for dpid in outputs if int(dpid) not in self.datapaths
        )
        if unknown:
            raise RuntimeError(f"switches are not connected: {unknown}")

        previous = self.groups.get(group_key)
        if defer_barriers and previous is not None:
            raise ValueError("deferred barriers are only valid for a new tree")
        self._version[group_key] += 1
        new_group_id = int(group_key) * 10000 + self._version[group_key]
        old_group_id = previous["openflow_group_id"] if previous else None

        # Stage the new groups everywhere before changing any forwarding flow.
        for dpid, ports in outputs.items():
            self._group_mod(
                self.datapaths[int(dpid)],
                new_group_id,
                ports,
                self.datapaths[int(dpid)].ofproto.OFPGC_ADD,
            )
        if not defer_barriers:
            self._barriers([self.datapaths[int(dpid)] for dpid in outputs])

        old_outputs = previous["switch_outputs"] if previous else {}

        # Prepare every downstream flow before moving the ingress switch.
        downstream = sorted(set(outputs) - ({source_key} if source_key else set()))
        for dpid in downstream:
            datapath = self.datapaths[int(dpid)]
            command = (
                datapath.ofproto.OFPFC_ADD
                if previous is None or dpid not in old_outputs
                else datapath.ofproto.OFPFC_MODIFY_STRICT
            )
            self._flow_for_group(datapath, dst_ip, new_group_id, command)
        if not defer_barriers:
            self._barriers([self.datapaths[int(dpid)] for dpid in downstream])

        if source_key is not None:
            datapath = self.datapaths[int(source_key)]
            command = (
                datapath.ofproto.OFPFC_ADD
                if previous is None or source_key not in old_outputs
                else datapath.ofproto.OFPFC_MODIFY_STRICT
            )
            self._flow_for_group(datapath, dst_ip, new_group_id, command)
            if not defer_barriers:
                self._barrier(datapath)

        # Old-only switches remain intact until the ingress cutover is complete.
        if previous is not None:
            if drain_seconds:
                hub.sleep(drain_seconds)
            old_only_datapaths = []
            for dpid in sorted(set(old_outputs) - set(outputs)):
                datapath = self.datapaths.get(int(dpid))
                if datapath is None:
                    continue
                datapath.send_msg(
                    datapath.ofproto_parser.OFPFlowMod(
                        datapath=datapath,
                        command=datapath.ofproto.OFPFC_DELETE,
                        out_group=int(old_group_id),
                        out_port=datapath.ofproto.OFPP_ANY,
                    )
                )
                old_only_datapaths.append(datapath)
            self._barriers(old_only_datapaths)

        if old_group_id is not None:
            old_group_datapaths = []
            for dpid in old_outputs:
                datapath = self.datapaths.get(int(dpid))
                if datapath is not None:
                    self._group_mod(
                        datapath,
                        old_group_id,
                        [],
                        datapath.ofproto.OFPGC_DELETE,
                    )
                    old_group_datapaths.append(datapath)
            self._barriers(old_group_datapaths)

        self.groups[group_key] = {
            "dst_ip": dst_ip,
            "openflow_group_id": new_group_id,
            "switch_outputs": outputs,
            "reroute": bool(payload.get("reroute", False)),
            "source_dpid": int(source_key) if source_key is not None else None,
            "drain_seconds": drain_seconds,
            "updated_at": time.time(),
        }
        return {
            "accepted": True,
            "group_id": int(group_key),
            "openflow_group_id": new_group_id,
            "reroute": bool(payload.get("reroute", False)),
            "source_dpid": int(source_key) if source_key is not None else None,
            "drain_seconds": drain_seconds,
            "switch_outputs": outputs,
        }

    def delete_tree(self, group_id):
        group_key = str(int(group_id))
        current = self.groups.get(group_key)
        if current is None:
            raise KeyError(f"unknown SFT group {group_key}")

        outputs = current["switch_outputs"]
        missing = sorted(
            int(dpid) for dpid in outputs if int(dpid) not in self.datapaths
        )
        if missing:
            raise RuntimeError(f"switches are not connected: {missing}")

        openflow_group_id = int(current["openflow_group_id"])
        dst_ip = str(current["dst_ip"])
        for dpid in outputs:
            datapath = self.datapaths[int(dpid)]
            self._delete_flow_for_group(datapath, dst_ip, openflow_group_id)
        self._barriers([self.datapaths[int(dpid)] for dpid in outputs])

        for dpid in outputs:
            datapath = self.datapaths[int(dpid)]
            self._group_mod(
                datapath,
                openflow_group_id,
                [],
                datapath.ofproto.OFPGC_DELETE,
            )
        self._barriers([self.datapaths[int(dpid)] for dpid in outputs])

        self.groups.pop(group_key, None)
        return {
            "accepted": True,
            "deleted": True,
            "group_id": int(group_key),
            "openflow_group_id": openflow_group_id,
            "dst_ip": dst_ip,
            "switches": sorted(int(dpid) for dpid in outputs),
        }
