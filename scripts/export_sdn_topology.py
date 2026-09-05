#!/usr/bin/env python3
"""Export a project MAT topology into a stable Mininet profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io


ROOT = Path(__file__).resolve().parents[1]
DC_NODES = [1, 3, 4, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28]


def extract_edges(mat_path: Path):
    data = scipy.io.loadmat(mat_path)
    if "adjacency" in data:
        adjacency = np.asarray(data["adjacency"])
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        node_count = int(adjacency.shape[0])
        edges = [
            (u + 1, v + 1)
            for u in range(node_count)
            for v in range(u + 1, node_count)
            if adjacency[u, v] > 0 or adjacency[v, u] > 0
        ]
        if not edges:
            raise RuntimeError("no physical edges found in adjacency")
        return node_count, edges, "adjacency matrix"

    if "Paths" not in data:
        raise KeyError("MAT topology must contain either 'Paths' or 'adjacency'")
    paths_matrix = data["Paths"]
    node_count = int(paths_matrix.shape[0])
    edges = set()
    for i in range(node_count):
        for j in range(node_count):
            cell = paths_matrix[i, j]
            if not hasattr(cell, "dtype") or not cell.dtype.names:
                continue
            if "paths" not in cell.dtype.names:
                continue
            paths = cell["paths"]
            if not isinstance(paths, np.ndarray):
                continue
            for row in np.atleast_2d(paths):
                nodes = [int(value) for value in np.ravel(row) if int(value) > 0]
                for u, v in zip(nodes, nodes[1:]):
                    if u != v:
                        edges.add(tuple(sorted((u, v))))
    if not edges:
        raise RuntimeError("no physical edges reconstructed from Paths")
    return node_count, sorted(edges), "all positive Paths entries"


def validate_connected(node_count, edges):
    adjacency = {node: set() for node in range(1, node_count + 1)}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    visited = set()
    stack = [1]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency[node] - visited)
    if len(visited) != node_count:
        raise RuntimeError(f"topology is disconnected: reached {len(visited)}/{node_count}")
    return adjacency


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=ROOT / "topo" / "US_Backbone_path.mat")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sdn" / "topologies" / "us_backbone_28.json",
    )
    parser.add_argument("--bandwidth-mbps", type=float, default=80.0)
    parser.add_argument("--delay-ms", type=float, default=2.0)
    parser.add_argument("--name", default="us_backbone_28")
    parser.add_argument("--expected-nodes", type=int, default=28)
    parser.add_argument(
        "--dc-nodes",
        type=int,
        nargs="+",
        default=DC_NODES,
        help="one-based nodes that host VNF/DC capacity",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.mat.is_absolute():
        args.mat = ROOT / args.mat
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.mat = args.mat.resolve()
    args.output = args.output.resolve()
    node_count, edge_pairs, connectivity_source = extract_edges(args.mat)
    if args.expected_nodes > 0 and node_count != args.expected_nodes:
        raise RuntimeError(f"expected {args.expected_nodes} nodes, found {node_count}")
    adjacency = validate_connected(node_count, edge_pairs)
    dc_nodes = set(args.dc_nodes)
    invalid_dc_nodes = sorted(dc_nodes - set(range(1, node_count + 1)))
    if invalid_dc_nodes:
        raise ValueError(f"DC nodes outside topology: {invalid_dc_nodes}")
    source_nodes = sorted(set(range(1, node_count + 1)) - dc_nodes)
    port_map = {
        node: {neighbor: index + 2 for index, neighbor in enumerate(sorted(adjacency[node]))}
        for node in adjacency
    }

    nodes = [
        {
            "simulator_node_id": node - 1,
            "matlab_node_id": node,
            "dpid": node,
            "switch": f"s{node}",
            "host": f"h{node}",
            "host_ip": f"10.0.0.{node}/24",
            "host_port": 1,
            "is_dc": node in dc_nodes,
            "is_source": node in source_nodes,
        }
        for node in range(1, node_count + 1)
    ]
    edges = [
        {
            "u": u,
            "v": v,
            "u_port": port_map[u][v],
            "v_port": port_map[v][u],
            "bandwidth_mbps": args.bandwidth_mbps,
            "delay_ms": args.delay_ms,
        }
        for u, v in edge_pairs
    ]
    profile = {
        "schema_version": 1,
        "name": args.name,
        "source": {
            "file": str(args.mat.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(args.mat.read_bytes()).hexdigest(),
            "connectivity": f"reconstructed from {connectivity_source}",
            "capacity_provenance": "uniform experiment parameter from configs/env.yaml",
            "delay_provenance": "uniform experiment parameter; MAT has no geographic latency",
        },
        "node_count": node_count,
        "physical_link_count": len(edges),
        "dc_nodes_1based": sorted(dc_nodes),
        "source_nodes_1based": source_nodes,
        "default_bandwidth_mbps": args.bandwidth_mbps,
        "default_delay_ms": args.delay_ms,
        "nodes": nodes,
        "edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "nodes": node_count,
                "physical_links": len(edges),
                "dc_nodes": len(dc_nodes),
                "source_nodes": source_nodes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
