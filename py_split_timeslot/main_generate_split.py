import argparse
import scipy.io
import os
import pickle
import numpy as np
import random

# ── 内联采样函数（不依赖外部 sampling.py）──────────────────────────────
_FIXED_VNF_TABLE = [
    {'type': 1, 'cpu_need': 2.17571289793936,  'memory_need': 1.06847854592062},
    {'type': 2, 'cpu_need': 0.327471689918480, 'memory_need': 1.67212031605700},
    {'type': 3, 'cpu_need': 0.691000236990969, 'memory_need': 1.84641423103628},
    {'type': 4, 'cpu_need': 0.401957140875833, 'memory_need': 1.30930392240156},
    {'type': 5, 'cpu_need': 0.804261986990572, 'memory_need': 1.66085909342891},
    {'type': 6, 'cpu_need': 2.83555094177359,  'memory_need': 1.09494296372459},
    {'type': 7, 'cpu_need': 1.64532198696671,  'memory_need': 0.296237077563595},
    {'type': 8, 'cpu_need': 0.983307643923246, 'memory_need': 0.754735724747883},
]

def sample_bandwidth():
    """bw: 4-6占63%，7-8占37%"""
    return random.choices([4, 5, 6, 7, 8], weights=[20, 20, 20, 20, 20], k=1)[0]

def sample_cpu_needs(vnf_chain, bandwidth):
    """按VNF系数×bw计算CPU，clip到[1,24]"""
    return [max(1, min(24, int(round(bandwidth * _FIXED_VNF_TABLE[v-1]['cpu_need'])))) for v in vnf_chain]

def sample_memory_needs(vnf_chain, bandwidth):
    """按VNF系数×bw计算MEM，clip到[1,16]"""
    return [max(1, min(16, int(round(bandwidth * _FIXED_VNF_TABLE[v-1]['memory_need'])))) for v in vnf_chain]

def sample_lifetime():
    # 截断负指数分布[1,6]，均值≈2.55，对标MATLAB的exprnd截断分布
    while True:
        lt = random.expovariate(1/2.0)
        if 1.0 <= lt <= 6.0:
            return lt

def parse_args():
    parser = argparse.ArgumentParser(description="动态生成多播SFC请求数据 (TA-HRL v4)")
    parser.add_argument('--topo', type=str, default='23node',
                        choices=['us_backbone', '13node', '23node', '50node'])
    parser.add_argument('--num_requests', type=int, default=0,
                        help='0表示自动计算: rate × T × 非DC节点数')
    parser.add_argument('--sim_duration', type=int, default=400, help='仿真时间步T')
    parser.add_argument('--rate', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase', type=str, default='phase3',
                        choices=['phase1', 'phase2', 'phase3', 'test'])
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(current_dir)

    TOPO_CFG = {
        'us_backbone': {
            'file': os.path.join(PROJECT_ROOT, 'topo', 'US_Backbone_path.mat'),
            'dc_nodes': [1, 3, 4, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 22, 23, 25, 26, 27, 28]
        },
        '13node': {
            'file': os.path.join(PROJECT_ROOT, 'topo', '13node.mat'),
            'dc_nodes': [2, 6, 9, 10, 11, 12, 13]
        },
        '23node': {
            'file': os.path.join(PROJECT_ROOT, 'topo', '23node.mat'),
            'dc_nodes': [2, 3, 4, 7, 8, 9, 10, 12, 13, 14, 19, 22]
        },
        '50node': {
            'file': os.path.join(PROJECT_ROOT, 'topo', '50node.mat'),
            'dc_nodes': [4, 5, 6, 11, 12, 14, 17, 22,27,35,38, 44,  46, 49, 50]
        },
    }

    cfg = TOPO_CFG[args.topo]
    mat_path = cfg['file']
    dc_nodes = cfg['dc_nodes']

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"❌ 找不到拓扑文件: {mat_path}\n您的项目根目录被识别为: {PROJECT_ROOT}")

    mat_data = scipy.io.loadmat(mat_path)

    num_nodes = None
    if 'adjacency' in mat_data:
        num_nodes = mat_data['adjacency'].shape[0]
    else:
        for key, val in mat_data.items():
            if not key.startswith('__') and hasattr(val, 'shape') and len(val.shape) == 2 and val.shape[0] == val.shape[
                1]:
                num_nodes = val.shape[0]
                break

    if num_nodes is None:
        raise ValueError("❌ 无法从 mat 文件中解析出有效的拓扑矩阵 (无法确认节点数量)")

    # DC节点集合（用于source过滤）
    dc_nodes_set = set(dc_nodes)
    non_dc_count = num_nodes - len(dc_nodes_set)

    # 自动计算num_requests（对标MATLAB: 每个非DC源节点独立生成泊松序列）
    if args.num_requests == 0:
        num_requests = args.rate * args.sim_duration * non_dc_count
    else:
        num_requests = args.num_requests

    print(f"✅ 成功加载拓扑: {args.topo} (节点数: {num_nodes}, 非DC源节点数: {non_dc_count}) - 生成阶段: {args.phase}")
    print(f"📊 生成请求数: {num_requests} (rate={args.rate} × T={args.sim_duration} × 非DC节点={non_dc_count})")

    requests = []
    requests_by_slot = {}

    current_time = 0.0
    time_slot_duration = 0.1
    K_vnf = 8

    for req_id in range(1, num_requests + 1):
        inter_arrival = random.expovariate(args.rate)
        current_time += inter_arrival
        lifetime = sample_lifetime()  # [1,3)占70%，[3,6]占30%
        leave_time = current_time + lifetime

        arrive_slot = int(current_time / time_slot_duration)
        leave_slot = int(leave_time / time_slot_duration)
        duration_slots = max(1, leave_slot - arrive_slot)

        # source只从非DC节点里选（对标MATLAB）
        non_dc_nodes = [n for n in range(1, num_nodes + 1) if n not in dc_nodes_set]
        source = random.choice(non_dc_nodes)

        num_dests = 5  # 固定5个目的节点
        # dest从全部节点（除source外）里选（对标MATLAB node_important包含全部节点）
        dest_candidates = [n for n in range(1, num_nodes + 1) if n != source]
        dests = random.sample(dest_candidates, min(num_dests, len(dest_candidates)))

        num_vnfs = 3  # 固定3个VNF
        vnf_chain = [random.randint(1, K_vnf) for _ in range(num_vnfs)]

        bw_origin  = sample_bandwidth()
        cpu_origin = sample_cpu_needs(vnf_chain, bw_origin)
        mem_origin = sample_memory_needs(vnf_chain, bw_origin)

        req = {
            'id': req_id,
            'source': source,
            'dest': dests,
            'vnf': vnf_chain,
            'bw_origin': bw_origin,
            'cpu_origin': cpu_origin,
            'memory_origin': mem_origin,
            'arrival_time': current_time,
            'leave_time': leave_time,
            'lifetime': lifetime,
            'time_slot': arrive_slot,
            'leave_time_slot': leave_slot,
            'duration': duration_slots,
            'arrive_time_step': arrive_slot,
            'leave_time_step':  leave_slot
        }

        requests.append(req)

        if arrive_slot not in requests_by_slot:
            requests_by_slot[arrive_slot] = []
        requests_by_slot[arrive_slot].append(req)

    output_dir = os.path.join(PROJECT_ROOT, f'data/{args.topo}_rate{args.rate}')
    os.makedirs(output_dir, exist_ok=True)

    list_file = os.path.join(output_dir, f'{args.phase}_requests.pkl')
    slot_file = os.path.join(output_dir, f'{args.phase}_requests_by_slot.pkl')

    with open(list_file, 'wb') as f:
        pickle.dump(requests, f)

    with open(slot_file, 'wb') as f:
        pickle.dump(requests_by_slot, f)

    print(f"📁 列表格式保存至: {list_file}")
    print(f"📁 字典格式保存至: {slot_file}")


if __name__ == '__main__':
    main()