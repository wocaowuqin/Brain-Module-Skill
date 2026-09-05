#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""统计输出"""

from collections import Counter


def print_statistics(requests, requests_by_slot, phase_name, time_slot_delta):
    """打印统计信息。"""
    print(f"\n{'=' * 60}")
    print(f"📊 {phase_name} 统计信息")
    print(f"{'=' * 60}")

    print(f"总请求数: {len(requests)}")
    print(f"时间槽数: {len(requests_by_slot)}")

    if requests_by_slot:
        min_slot = min(requests_by_slot.keys())
        max_slot = max(requests_by_slot.keys())
        print(f"时间槽范围: {min_slot} - {max_slot}")
        print(f"实际时间范围: {min_slot * time_slot_delta:.2f}s - {max_slot * time_slot_delta:.2f}s")

        slot_counts = [len(reqs) for reqs in requests_by_slot.values()]
        avg_per_slot = sum(slot_counts) / len(slot_counts)
        max_per_slot = max(slot_counts)

        print(f"平均每时间槽: {avg_per_slot:.2f} 个请求")
        print(f"最大每时间槽: {max_per_slot} 个请求")

        print("\n时间槽密度分布:")
        density = Counter(slot_counts)
        for count in sorted(density.keys())[:10]:
            print(f"  {count}个请求/槽: {density[count]} 个时间槽")

    durations = [req['duration'] for req in requests]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print("\n持续时间统计（时间槽）:")
        print(f"  平均: {avg_duration:.1f} ({avg_duration * time_slot_delta:.3f}s)")

    print(f"{'=' * 60}")
