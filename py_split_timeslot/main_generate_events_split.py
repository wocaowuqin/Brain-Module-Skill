import os
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="生成 SFC 请求的离散事件列表 (Arrive/Leave)")
    parser.add_argument('--topo', type=str, default='23node',
                        help='拓扑名称 (如 13node, 50node)')
    parser.add_argument('--rate', type=int, default=8,
                        help='到达率 (如 8, 28)')
    return parser.parse_args()


def process_events(input_file, output_list_file, output_slot_file):
    """读取 requests，生成按时间排序的 events 和按 slot 分组的 events"""
    if not os.path.exists(input_file):
        print(f"⚠️  跳过: 未找到 {input_file}")
        return False

    with open(input_file, 'rb') as f:
        requests = pickle.load(f)

    events = []
    events_by_slot = {}

    # 1. 将每个请求拆分为 "arrive" 和 "leave" 两个独立事件
    for req in requests:
        # 到达事件
        events.append({
            'time': req['arrival_time'],
            'time_slot': req['time_slot'],
            'type': 'arrive',
            'request': req
        })
        # 离开事件
        events.append({
            'time': req['leave_time'],
            'time_slot': req['leave_time_slot'],
            'type': 'leave',
            'request': req
        })

    # 2. 按绝对时间排序
    # (如果时间完全一样，优先处理离开 leave 事件释放资源，再处理到达 arrive)
    events.sort(key=lambda x: (x['time'], 0 if x['type'] == 'leave' else 1))

    # 3. 按 time_slot 分组
    for ev in events:
        slot = ev['time_slot']
        if slot not in events_by_slot:
            events_by_slot[slot] = []
        events_by_slot[slot].append(ev)

    # 4. 保存文件
    with open(output_list_file, 'wb') as f:
        pickle.dump(events, f)
    with open(output_slot_file, 'wb') as f:
        pickle.dump(events_by_slot, f)

    print(f"✅ 成功处理: 生成了 {len(events)} 个事件 (来自 {len(requests)} 个请求)")
    return True


def main():
    args = parse_args()

    # ==========================================
    # 获取项目根目录 (自动向上退一级)
    # ==========================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(current_dir)

    # ==========================================
    # 定位数据目录 (直接对齐前一个脚本的输出目录)
    # ==========================================
    data_dir = os.path.join(PROJECT_ROOT, 'data', f'{args.topo}_rate{args.rate}')

    print("=" * 60)
    print(f"🚀 开始生成离散事件列表 (Event Generation)")
    print(f"📂 目标数据目录: {data_dir}")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"❌ 严重错误: 数据目录不存在 {data_dir}")
        print(f"💡 请先运行 main_generate_split.py 生成 requests 数据！")
        return

    # 需要处理的阶段
    phases = ['phase1', 'phase3']
    success_count = 0

    for phase in phases:
        print(f"\n🔥 处理 {phase.upper()} 数据...")

        req_file = os.path.join(data_dir, f'{phase}_requests.pkl')
        ev_list_file = os.path.join(data_dir, f'{phase}_events.pkl')
        ev_slot_file = os.path.join(data_dir, f'{phase}_events_by_slot.pkl')

        if process_events(req_file, ev_list_file, ev_slot_file):
            success_count += 1
            print(f"📁 列表事件保存至: {ev_list_file}")
            print(f"📁 分槽事件保存至: {ev_slot_file}")

    print("\n" + "=" * 60)
    if success_count > 0:
        print("🎉 所有存在的事件列表生成完毕！")
    else:
        print("⚠️  没有任何事件被生成，请检查 requests.pkl 文件是否存在。")
    print("=" * 60)


if __name__ == '__main__':
    main()