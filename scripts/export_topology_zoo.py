#!/usr/bin/env python3
"""Export a Topology Zoo GraphML network as a stable Mininet profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.io


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URL = "https://topology-zoo.org/files/Cogentco.graphml"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml", type=Path, default=ROOT / "topo" / "Cogentco.graphml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sdn" / "topologies" / "cogentco_197.json",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=ROOT / "topo" / "Cogentco.mat",
    )
    parser.add_argument("--name", default="cogentco_197")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--bandwidth-mbps", type=float, default=80.0)
    parser.add_argument("--dc-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-delay-ms", type=float, default=0.1)
    return parser.parse_args()


def stable_node_key(node):
    value = str(node)
    return (0, int(value)) if value.isdigit() else (1, value)


def haversine_km(source, destination):
    required = ("Latitude", "Longitude")
    if any(key not in source or key not in destination for key in required):
        return None
    lat1 = math.radians(float(source["Latitude"]))
    lon1 = math.radians(float(source["Longitude"]))
    lat2 = math.radians(float(destination["Latitude"]))
    lon2 = math.radians(float(destination["Longitude"]))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(value)))


def select_dc_nodes(graph, dpid_by_node, fraction):
    if not 0.0 < fraction < 1.0:
        raise ValueError("dc_fraction must be in (0, 1)")
    simple = nx.Graph(graph)
    betweenness = nx.betweenness_centrality(simple, normalized=True)
    ranked = sorted(
        simple.nodes,
        key=lambda node: (
            -int(simple.degree(node)),
            -float(betweenness[node]),
            stable_node_key(node),
        ),
    )
    count = int(math.ceil(len(ranked) * fraction))
    return {dpid_by_node[node] for node in ranked[:count]}


def validate_graph(graph):
    if graph.is_directed():
        raise ValueError("Topology Zoo input must be undirected")
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Topology Zoo input is empty")
    if nx.number_connected_components(nx.Graph(graph)) != 1:
        raise ValueError("Topology Zoo input must be connected")
    if nx.number_of_selfloops(graph):
        raise ValueError("Topology Zoo input contains self-loops")


def main():
    args = parse_args()
    if args.bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive")
    graph = nx.read_graphml(args.graphml)
    validate_graph(graph)

    ordered_nodes = sorted(graph.nodes, key=stable_node_key)
    dpid_by_node = {node: index + 1 for index, node in enumerate(ordered_nodes)}
    dc_nodes = select_dc_nodes(graph, dpid_by_node, args.dc_fraction)
    source_nodes = sorted(set(dpid_by_node.values()) - dc_nodes)

    if graph.is_multigraph():
        raw_edges = list(graph.edges(keys=True, data=True))
    else:
        raw_edges = [(u, v, "0", data) for u, v, data in graph.edges(data=True)]
    raw_edges.sort(
        key=lambda item: (
            min(dpid_by_node[item[0]], dpid_by_node[item[1]]),
            max(dpid_by_node[item[0]], dpid_by_node[item[1]]),
            str(item[2]),
        )
    )
    known_distances = [
        distance
        for source, destination, _, _ in raw_edges
        for distance in [haversine_km(graph.nodes[source], graph.nodes[destination])]
        if distance is not None
    ]
    if not known_distances:
        raise ValueError("Topology Zoo input has no links with geographic coordinates")
    fallback_delay_ms = statistics.median(
        max(float(args.minimum_delay_ms), distance / 200.0)
        for distance in known_distances
    )

    next_port = {dpid: 2 for dpid in dpid_by_node.values()}
    edges = []
    delays = []
    for source, destination, edge_key, _ in raw_edges:
        u = dpid_by_node[source]
        v = dpid_by_node[destination]
        if u > v:
            source, destination = destination, source
            u, v = v, u
        u_port = next_port[u]
        v_port = next_port[v]
        next_port[u] += 1
        next_port[v] += 1
        distance_km = haversine_km(graph.nodes[source], graph.nodes[destination])
        delay_estimated = distance_km is None
        delay_ms = (
            fallback_delay_ms
            if delay_estimated
            else max(float(args.minimum_delay_ms), distance_km / 200.0)
        )
        delays.append(delay_ms)
        edges.append(
            {
                "u": u,
                "v": v,
                "u_port": u_port,
                "v_port": v_port,
                "topology_zoo_edge_key": str(edge_key),
                "distance_km": round(distance_km, 3) if distance_km is not None else None,
                "delay_estimated": delay_estimated,
                "bandwidth_mbps": float(args.bandwidth_mbps),
                "delay_ms": round(delay_ms, 4),
            }
        )

    nodes = []
    for original_id in ordered_nodes:
        dpid = dpid_by_node[original_id]
        attributes = graph.nodes[original_id]
        nodes.append(
            {
                "simulator_node_id": dpid - 1,
                "topology_zoo_node_id": str(original_id),
                "dpid": dpid,
                "switch": f"s{dpid}",
                "host": f"h{dpid}",
                "host_ip": f"10.0.0.{dpid}/24",
                "host_port": 1,
                "label": str(attributes.get("label", original_id)),
                "country": str(attributes.get("Country", "")),
                "latitude": float(attributes["Latitude"])
                if "Latitude" in attributes
                else None,
                "longitude": float(attributes["Longitude"])
                if "Longitude" in attributes
                else None,
                "is_dc": dpid in dc_nodes,
                "is_source": dpid in source_nodes,
            }
        )

    profile = {
        "schema_version": 1,
        "name": str(args.name),
        "source": {
            "file": str(args.graphml.relative_to(ROOT)).replace("\\", "/"),
            "url": str(args.source_url),
            "sha256": hashlib.sha256(args.graphml.read_bytes()).hexdigest(),
            "network": str(graph.graph.get("Network", "Cogent")),
            "network_date": str(graph.graph.get("NetworkDate", "2010_08")),
            "connectivity": "all GraphML edges, including parallel links",
            "capacity_provenance": "uniform controlled experiment parameter",
            "delay_provenance": (
                "great-circle endpoint distance / 200 km/ms; minimum delay floor; "
                "links with missing endpoint coordinates use the known-link median; "
                "modeled rather than measured"
            ),
        },
        "node_count": len(nodes),
        "physical_link_count": len(edges),
        "simple_link_count": nx.Graph(graph).number_of_edges(),
        "parallel_link_count": len(edges) - nx.Graph(graph).number_of_edges(),
        "nodes_missing_coordinates": sum(
            1 for node in nodes if node["latitude"] is None or node["longitude"] is None
        ),
        "links_with_estimated_delay": sum(
            1 for edge in edges if edge["delay_estimated"]
        ),
        "dc_nodes_1based": sorted(dc_nodes),
        "source_nodes_1based": source_nodes,
        "dc_selection": {
            "fraction": float(args.dc_fraction),
            "count": len(dc_nodes),
            "rule": "degree desc, betweenness centrality desc, original node id asc",
        },
        "default_bandwidth_mbps": float(args.bandwidth_mbps),
        "default_delay_ms": round(statistics.median(delays), 4),
        "delay_range_ms": [round(min(delays), 4), round(max(delays), 4)],
        "nodes": nodes,
        "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix = nx.to_numpy_array(
        nx.Graph(graph),
        nodelist=ordered_nodes,
        dtype=np.float32,
        weight=None,
    )
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(args.matrix_output, {"adjacency": matrix})
    print(
        json.dumps(
            {
                "output": str(args.output),
                "matrix_output": str(args.matrix_output),
                "nodes": len(nodes),
                "physical_links": len(edges),
                "simple_links": profile["simple_link_count"],
                "parallel_links": profile["parallel_link_count"],
                "dc_nodes": len(dc_nodes),
                "source_nodes": len(source_nodes),
                "delay_range_ms": profile["delay_range_ms"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
