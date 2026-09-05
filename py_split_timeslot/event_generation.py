#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""事件生成逻辑（从 generate_event.py 拆分）"""

import os
import pickle

from config import DATA_DIR


def process_single_file_timestep(input_filename, output_filename):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f'⚠️  跳过: 未找到 {input_path}')
        return

    print(f'🔄 正在处理（时间步版）: {input_filename} -> {output_filename} ...')

    with open(input_path, 'rb') as f:
        requests_list = pickle.load(f)

    if not requests_list:
        print('❌ 请求列表为空！')
        return

    max_leave_step = 0
    for req in requests_list:
        l_step = int(req['leave_time_step'])
        if l_step > max_leave_step:
            max_leave_step = l_step

    print(f'   ⏱️  最大时间步: {max_leave_step}')

    event_list = []
    for t in range(max_leave_step + 2):
        event_list.append({'time_step': t, 'arrive_event': [], 'leave_event': []})

    count_arrive = 0
    count_leave = 0
    for req in requests_list:
        req_id = req['id']
        t_arr = int(req['arrive_time_step'])
        t_leave = int(req['leave_time_step'])
        if 0 <= t_arr < len(event_list):
            event_list[t_arr]['arrive_event'].append(req_id)
            count_arrive += 1
        if 0 <= t_leave < len(event_list):
            event_list[t_leave]['leave_event'].append(req_id)
            count_leave += 1

    with open(output_path, 'wb') as f:
        pickle.dump(event_list, f)

    print(f'✅ 已生成（时间步版）: {output_path}')
    print(f'   统计: 到达 {count_arrive} 个, 离开 {count_leave} 个')
    print('-' * 50)


def process_single_file_timeslot(input_filename, output_filename):
    input_path = os.path.join(DATA_DIR, input_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    if not os.path.exists(input_path):
        print(f'⚠️  跳过: 未找到 {input_path}')
        return

    print(f'🔄 正在处理（时间槽版）: {input_filename} -> {output_filename} ...')

    with open(input_path, 'rb') as f:
        requests_list = pickle.load(f)

    if not requests_list:
        print('❌ 请求列表为空！')
        return

    max_leave_slot = 0
    for req in requests_list:
        if 'leave_time_slot' not in req:
            print(f"⚠️  请求 {req['id']} 缺少 'leave_time_slot' 字段，跳过")
            return
        l_slot = int(req['leave_time_slot'])
        if l_slot > max_leave_slot:
            max_leave_slot = l_slot

    print(f'   ⏱️  最大时间槽: {max_leave_slot}')

    event_list = []
    for slot in range(max_leave_slot + 2):
        event_list.append({'time_slot': slot, 'arrive_event': [], 'leave_event': []})

    count_arrive = 0
    count_leave = 0
    for req in requests_list:
        req_id = req['id']
        t_arr = int(req['time_slot'])
        t_leave = int(req['leave_time_slot'])
        if 0 <= t_arr < len(event_list):
            event_list[t_arr]['arrive_event'].append(req_id)
            count_arrive += 1
        if 0 <= t_leave < len(event_list):
            event_list[t_leave]['leave_event'].append(req_id)
            count_leave += 1

    with open(output_path, 'wb') as f:
        pickle.dump(event_list, f)

    print(f'✅ 已生成（时间槽版）: {output_path}')
    print(f'   统计: 到达 {count_arrive} 个, 离开 {count_leave} 个')
    print('-' * 50)


def generate_events():
    print('=' * 60)
    print('🚀 生成事件列表 (Event Generation)')
    print(f'📂 工作目录: {DATA_DIR}')
    print('=' * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    print('\n🔥 处理 Phase 1 数据...')
    process_single_file_timestep('phase1_requests.pkl', 'phase1_events.pkl')
    process_single_file_timeslot('phase1_requests.pkl', 'phase1_events_by_slot.pkl')

    print('\n🔥 处理 Phase 3 数据...')
    process_single_file_timestep('phase3_requests.pkl', 'phase3_events.pkl')
    process_single_file_timeslot('phase3_requests.pkl', 'phase3_events_by_slot.pkl')

    print('\n🎉 所有事件列表生成完毕！')
    print('\n📁 生成的事件文件:')
    for filename in sorted(os.listdir(DATA_DIR)):
        if 'event' in filename and filename.endswith('.pkl'):
            filepath = os.path.join(DATA_DIR, filename)
            size = os.path.getsize(filepath) / 1024
            print(f'   - {filename} ({size:.1f} KB)')


def verify_events(events_filename):
    filepath = os.path.join(DATA_DIR, events_filename)
    if not os.path.exists(filepath):
        print(f'⚠️  文件不存在: {filepath}')
        return

    print(f'\n🔍 验证事件列表: {events_filename}')
    with open(filepath, 'rb') as f:
        events = pickle.load(f)

    total_arrive = sum(len(e['arrive_event']) for e in events)
    total_leave = sum(len(e['leave_event']) for e in events)
    non_empty_slots = [
        e.get('time_slot', e.get('time_step'))
        for e in events
        if e['arrive_event'] or e['leave_event']
    ]

    if non_empty_slots:
        min_slot = min(non_empty_slots)
        max_slot = max(non_empty_slots)
    else:
        min_slot = max_slot = 0

    print(f'   总事件槽数: {len(events)}')
    print(f'   总到达事件: {total_arrive}')
    print(f'   总离开事件: {total_leave}')
    print(f'   有效槽范围: {min_slot} - {max_slot}')

    print('\n   前5个有事件的槽:')
    count = 0
    for e in events:
        slot_id = e.get('time_slot', e.get('time_step'))
        if e['arrive_event'] or e['leave_event']:
            print(f"      Slot {slot_id}: 到达={e['arrive_event']}, 离开={e['leave_event']}")
            count += 1
            if count >= 5:
                break
