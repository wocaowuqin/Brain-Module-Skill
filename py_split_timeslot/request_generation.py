import random

import numpy as np

from config import (
    NODE_TRAFFIC_LIST,
    TIME_INTERVAL,
    TIME_SLOT_DELTA,
    NUM_DESTINATIONS,
    VNF_CHAIN_LENGTH,
    VNF_TYPES,
    MEAN_LIFETIME,
)

from arrivals import generate_poisson_arrivals

from sampling import (
    sample_lifetime,
    sample_bandwidth,
    sample_cpu_needs,
    sample_memory_needs,
)

from request_factory import generate_single_request
from grouping import group_requests_by_time_slot
from statistics_utils import print_statistics
from vnf_catalog import generate_vnfs_catalog


def generate_all_requests(num_intervals, lamda, seed=None, phase_name='Unknown'):
    """生成所有请求（时间槽版本）。"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    all_vnf = generate_vnfs_catalog(VNF_TYPES)
    T_duration = num_intervals * TIME_INTERVAL

    print('=' * 60)
    print(f'🚀 生成 {phase_name} 数据')
    print(f'   随机种子: {seed}')
    print(f'   时间间隔: {num_intervals} (总时长 {T_duration}s)')
    print(f'   到达率: {lamda:.3f} req/s')
    print(f'   时间槽大小: {TIME_SLOT_DELTA * 1000:.1f} ms')
    print(f'   预期时间槽数: ~{int(T_duration / TIME_SLOT_DELTA)}')
    print(f'   预期总请求数: ~{int(lamda * T_duration * len(NODE_TRAFFIC_LIST))}')
    print('=' * 60)

    all_requests = []

    for source_node in NODE_TRAFFIC_LIST:
        arrive_times = generate_poisson_arrivals(T_duration, lamda)
        candidate_dests = [n for n in NODE_TRAFFIC_LIST if n != source_node]

        for arrive_time in arrive_times:
            destinations = random.sample(candidate_dests, NUM_DESTINATIONS)
            vnf_chain = random.sample(range(1, VNF_TYPES + 1), VNF_CHAIN_LENGTH)
            bandwidth = sample_bandwidth()
            lifetime_seconds = sample_lifetime(MEAN_LIFETIME)
            cpu_needs = sample_cpu_needs(vnf_chain, bandwidth, all_vnf)
            mem_needs = sample_memory_needs(vnf_chain, bandwidth, all_vnf)
            req = generate_single_request(
                req_id=0,
                source=source_node,
                destinations=destinations,
                vnf_chain=vnf_chain,
                bandwidth=bandwidth,
                cpu_needs=cpu_needs,
                mem_needs=mem_needs,
                arrive_time=arrive_time,
                lifetime=lifetime_seconds,
                delta_t=TIME_SLOT_DELTA,
            )
            all_requests.append(req)

    all_requests.sort(key=lambda r: r['arrival_time'])
    for i, req in enumerate(all_requests, 1):
        req['id'] = i

    requests_by_slot = group_requests_by_time_slot(all_requests)
    print(f'✅ {phase_name} 生成完毕: 共 {len(all_requests)} 条请求')
    print_statistics(all_requests, requests_by_slot, phase_name, TIME_SLOT_DELTA)
    return all_requests, requests_by_slot
