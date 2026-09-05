"""Mininet topology loader for exported fixed-port SDN profiles."""

from __future__ import annotations

import json
from pathlib import Path

from mininet.topo import Topo


try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # Mininet loads --custom files with exec(), where __file__ is absent.
    BASE_DIR = Path.cwd() / "sdn"
PROFILE_DIR = BASE_DIR / "topologies"


def deterministic_host_mac(dpid):
    value = int(dpid)
    if value <= 0 or value > 0xFFFFFF:
        raise ValueError(f"DPID {value} exceeds deterministic host MAC range")
    return f"02:00:00:{(value >> 16) & 0xff:02x}:{(value >> 8) & 0xff:02x}:{value & 0xff:02x}"


class ProfileTopo(Topo):
    def build(
        self,
        profile_name,
        include_hosts=True,
        host_dpids=None,
        max_queue_size=1000,
        qdisc="htb",
    ):
        profile_path = Path(str(profile_name))
        if not profile_path.is_absolute():
            profile_path = PROFILE_DIR / profile_path
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        selected_hosts = (
            {int(dpid) for dpid in host_dpids}
            if host_dpids is not None
            else None
        )
        max_queue_size = int(max_queue_size)
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if qdisc not in {
            "htb",
            "htb_fq",
            "htb_fq_codel",
            "htb_prio",
            "htb_prio_fq_codel",
            "tbf",
            "hfsc",
        }:
            raise ValueError(f"unsupported qdisc {qdisc!r}")
        qdisc_options = {
            "use_htb": qdisc in {
                "htb",
                "htb_fq",
                "htb_fq_codel",
                "htb_prio",
                "htb_prio_fq_codel",
            },
            "use_tbf": qdisc == "tbf",
            "use_hfsc": qdisc == "hfsc",
        }
        switches = {}
        for node in profile["nodes"]:
            node_id = int(node["dpid"])
            switch = self.addSwitch(
                node["switch"],
                dpid=f"{int(node['dpid']):016x}",
                protocols="OpenFlow13",
            )
            switches[node_id] = switch
            if include_hosts and (selected_hosts is None or node_id in selected_hosts):
                host = self.addHost(
                    node["host"],
                    ip=node["host_ip"],
                    mac=deterministic_host_mac(node_id),
                )
                self.addLink(
                    host,
                    switch,
                    port1=0,
                    port2=int(node["host_port"]),
                )

        for edge in profile["edges"]:
            self.addLink(
                switches[int(edge["u"])],
                switches[int(edge["v"])],
                port1=int(edge["u_port"]),
                port2=int(edge["v_port"]),
                bw=float(edge["bandwidth_mbps"]),
                delay=f"{float(edge['delay_ms']):g}ms",
                loss=0,
                max_queue_size=max_queue_size,
                **qdisc_options,
            )


topos = {
    "usbackbone": lambda: ProfileTopo("us_backbone_28.json"),
    "50node": lambda: ProfileTopo("50node_bw90.json"),
    "cogentco": lambda: ProfileTopo("cogentco_197.json"),
}
