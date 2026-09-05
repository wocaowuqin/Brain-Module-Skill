#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_path_db.py
==============
从 GML 拓扑文件生成 US_Backbone_path.mat 兼容的路径数据库

格式：Paths[N,N] 结构体，每个 cell 包含：
  paths:         [K, N] uint16  K条候选路径的节点序列（1-indexed，0=填充）
  pathshops:     [K, 1] uint8   每条路径的跳数（节点数）
  pathsdistance: [K, 1] uint8   每条路径的边数（= pathshops-1）
  pathindex:     [1, 1] uint8   候选路径数量（=K）
  link_ids:      [K, N] uint16  每条路径的链路ID序列（0=填充）

用法：
    python gen_path_db.py graph_attr_23node.txt path_db_23node.mat
    python gen_path_db.py                        # 批量处理 INPUT_FILES
"""

import re, os, sys
import numpy as np
import networkx as nx
import scipy.io

INPUT_FILES = [
    ('graph_attr_13node.txt', 'path_db_13node.mat'),
    ('graph_attr_23node.txt', 'path_db_23node.mat'),
    ('graph_attr_50node.txt', 'path_db_50node.mat'),
]
K = 5       # 候选路径数量（与 US Backbone 一致）


def parse_gml(path):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    node_ids = sorted(set(int(x) for x in re.findall(r'\bid (\d+)', content)))
    N = len(node_ids)
    p1 = re.compile(r'source\s+(\d+)\s+target\s+(\d+)\s+key\s+\d+\s+port\s+\d+\s+weight\s+\d+\s+bandwidth\s+"(\d+)kbps"', re.DOTALL)
    p2 = re.compile(r'source\s+(\d+)\s+target\s+(\d+)\s+key\s+\d+\s+bandwidth\s+"(\d+)kbps"\s+port\s+\d+\s+weight\s+\d+', re.DOTALL)
    edges = p1.findall(content) or p2.findall(content)
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for s, t, _ in edges:
        G.add_edge(int(s), int(t))
    return G, N


def get_k_shortest_paths(G, src, dst, k):
    """获取 K 条最短路径（使用 Yen's algorithm）"""
    try:
        gen = nx.shortest_simple_paths(G, src, dst)
        paths = []
        for p in gen:
            paths.append(p)
            if len(paths) >= k:
                break
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def build_path_db(gml_path, output_path, k=K):
    G, N = parse_gml(gml_path)
    print(f"\n处理：{os.path.basename(gml_path)}")
    print(f"  {N}节点，{G.number_of_edges()}条有向边")

    # 有向边→link_id（1-indexed，按(u,v)排序）
    edge_to_lid = {(u, v): i+1 for i, (u, v) in enumerate(sorted(G.edges()))}
    L = len(edge_to_lid)

    # dtype 与 US_Backbone_path.mat 完全一致
    cell_dtype = np.dtype([
        ('paths',         'O'),
        ('pathshops',     'O'),
        ('pathsdistance', 'O'),
        ('pathindex',     'O'),
        ('link_ids',      'O'),
    ])
    Paths = np.empty((N, N), dtype=cell_dtype)

    for i in range(N):
        for j in range(N):
            cell = np.zeros(1, dtype=cell_dtype)

            if i == j:
                # 对角线：全0
                cell['paths'][0]         = np.zeros((1, N), dtype=np.uint16)
                cell['pathshops'][0]     = np.zeros((1, 1), dtype=np.uint8)
                cell['pathsdistance'][0] = np.zeros((1, 1), dtype=np.uint8)
                cell['pathindex'][0]     = np.zeros((1, 1), dtype=np.uint8)
                cell['link_ids'][0]      = np.zeros((1, N), dtype=np.uint16)
                Paths[i, j] = cell[0]
                continue

            # 获取K条最短路径
            kpaths = get_k_shortest_paths(G, i, j, k)

            if not kpaths:
                # 不可达
                cell['paths'][0]         = np.zeros((k, N), dtype=np.uint16)
                cell['pathshops'][0]     = np.full((k, 1), 255, dtype=np.uint8)
                cell['pathsdistance'][0] = np.full((k, 1), 255, dtype=np.uint8)
                cell['pathindex'][0]     = np.array([[k]], dtype=np.uint8)
                cell['link_ids'][0]      = np.zeros((k, N), dtype=np.uint16)
                Paths[i, j] = cell[0]
                continue

            # 补足K条（不够的用最后一条填充）
            while len(kpaths) < k:
                kpaths.append(kpaths[-1])

            # 构建矩阵
            paths_mat = np.zeros((k, N), dtype=np.uint16)
            hops_mat  = np.zeros((k, 1), dtype=np.uint8)
            dist_mat  = np.zeros((k, 1), dtype=np.uint8)
            lids_mat  = np.zeros((k, N), dtype=np.uint16)

            for pi, path in enumerate(kpaths):
                hops = len(path)
                # 节点序列（1-indexed，0填充）
                for ci, node in enumerate(path):
                    paths_mat[pi, ci] = node + 1
                # 链路ID序列
                for ci in range(len(path)-1):
                    u, v = path[ci], path[ci+1]
                    lids_mat[pi, ci] = edge_to_lid.get((u, v), 0)

                hops_mat[pi, 0] = min(hops, 255)
                dist_mat[pi, 0] = min(hops-1, 255)

            cell['paths'][0]         = paths_mat
            cell['pathshops'][0]     = hops_mat
            cell['pathsdistance'][0] = dist_mat
            cell['pathindex'][0]     = np.array([[k]], dtype=np.uint8)
            cell['link_ids'][0]      = lids_mat
            Paths[i, j] = cell[0]

    scipy.io.savemat(output_path, {'Paths': Paths})
    print(f"  ✅ 已保存：{output_path}")
    print(f"  link_id范围：1~{L}，每对节点{k}条候选路径")

    # 简单验证
    check = scipy.io.loadmat(output_path)
    c = check['Paths'][0, 1]
    print(f"  验证[0,1]:")
    print(f"    paths=\n{c['paths'][:,:4]}")
    print(f"    pathshops={c['pathshops'][0].T}")
    print(f"    pathindex={c['pathindex'][0]}")


def main():
    if len(sys.argv) == 3:
        build_path_db(sys.argv[1], sys.argv[2])
        return
    for gml, out in INPUT_FILES:
        if not os.path.exists(gml):
            print(f"⚠️  跳过：{gml}")
            continue
        build_path_db(gml, out)
    print("\n全部完成！")
    print("使用时在 main.py 里把 'US_Backbone_path.mat' 改成对应的 path_db 文件名")


if __name__ == '__main__':
    main()