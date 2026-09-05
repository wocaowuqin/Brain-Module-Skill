#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""从原 data_generator.py 拆出的类版本。"""

import random

from vnf_catalog import generate_vnfs_catalog
from arrivals import generate_poisson_arrivals
from grouping import group_requests_by_time_slot
from request_factory import generate_single_request


class DataGenerator:
    def __init__(self, config):
        self.num_nodes = config.get('num_nodes', 28)
        self.delta_t = config.get('time_slot_delta', 0.01)
        self.max_time_slots = config.get('max_time_slots', 10000)
        self.arrival_rate = config.get('arrival_rate', 56)
        self.vnf_type_num = config.get('vnf_type_num', 8)
        self.all_vnf = generate_vnfs_catalog(self.vnf_type_num)

        print('✅ DataGenerator 初始化完成')
        print(f'   时间槽大小: {self.delta_t * 1000:.1f} ms')
        print(f'   最大时间槽数: {self.max_time_slots}')
        print(f'   到达率: {self.arrival_rate} req/s')
        print(f'   VNF类型数: {self.vnf_type_num}')

    def _generate_node_requests(self, source, node_important, arrive_time_list):
        node_requests = []
        candidates = [n for n in node_important if n != source]

        max_bandwidth = 8
        min_bandwidth = 4
        multicast_num = 5

        for arrive_time in arrive_time_list:
            k = min(multicast_num, len(candidates))
            dest = random.sample(candidates, k)

            req = generate_single_request(
                req_id=0,
                source=source,
                destinations=dest,
                vnf_chain=[v['type'] for v in random.sample(self.all_vnf, 3)],
                bandwidth=random.randint(min_bandwidth, max_bandwidth),
                cpu_needs=[round(random.randint(4, 8) * v['cpu_need']) for v in random.sample(self.all_vnf, 3)],
                mem_needs=[round(random.randint(4, 8) * v['memory_need']) for v in random.sample(self.all_vnf, 3)],
                arrive_time=arrive_time,
                lifetime=1 + random.random() * 5,
                delta_t=self.delta_t,
            )
            node_requests.append(req)

        return node_requests

    def generate_all_requests(self, num_requests, node_important):
        print(f'\n🔄 开始生成 {num_requests} 个请求...')
        all_requests = []
        T = self.max_time_slots * self.delta_t
        num_sources = len(node_important)
        requests_per_source = num_requests // num_sources

        for source in node_important:
            arrive_time_list = generate_poisson_arrivals(T, self.arrival_rate / num_sources)
            arrive_time_list = arrive_time_list[:requests_per_source]
            node_requests = self._generate_node_requests(source, node_important, arrive_time_list)
            all_requests.extend(node_requests)

        all_requests = all_requests[:num_requests]
        all_requests.sort(key=lambda x: x['arrival_time'])
        for i, req in enumerate(all_requests):
            req['id'] = i

        requests_by_slot = group_requests_by_time_slot(all_requests)
        return all_requests, requests_by_slot


def quick_generate(num_requests=300, num_nodes=28, arrival_rate=56, delta_t=0.01):
    config = {
        'num_nodes': num_nodes,
        'time_slot_delta': delta_t,
        'max_time_slots': 10000,
        'arrival_rate': arrival_rate,
        'vnf_type_num': 8,
    }
    generator = DataGenerator(config)
    node_important = list(range(0, num_nodes, max(1, num_nodes // 6)))[:6]
    requests, requests_by_slot = generator.generate_all_requests(
        num_requests=num_requests,
        node_important=node_important,
    )
    return requests, requests_by_slot, generator.all_vnf
