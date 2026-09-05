# Ryu/Mininet 执行层

本目录负责把完整 SFC/SFT 计划部署到 WSL 中的 Ryu、Open vSwitch 和
Mininet，并使用常驻 VNF Agent 与 UDP probe 收集真实执行证据。项目总入口、
数据边界和恢复方法见根 [README](../README.md)、
[工作流手册](../docs/WORKFLOWS.md) 与
[架构说明](../docs/ARCHITECTURE.md)。

## 推荐入口

从项目根目录运行：

```powershell
$PY = 'C:\Users\11353\.conda\envs\sfc_ppo\python.exe'
& $PY -m sfc_project doctor --deep
& $PY -m sfc_project list --category mininet
```

所有 Mininet preset 默认仅显示命令。真实执行会改变 WSL 中的服务和网络状态，
必须同时提供两个开关：

```powershell
& $PY -m sfc_project run mininet-testbed-smoke `
  --execute --confirm-system-changes
```

当前预设：

| Preset | 用途 |
| --- | --- |
| `mininet-testbed-smoke` | 检查 WSL、Ryu、OVS、Mininet 和基本连通性 |
| `mininet-us-validate` | 验证 US-backbone 交换机和固定端口 profile |
| `mininet-cogentco-validate` | 构造并验证 Cogentco 交换机拓扑 |
| `mininet-sft-group` | 检查 SFT group 安装与替换 |
| `mininet-sft-multihop` | 检查多跳 SFT 和分支重路由 |
| `mininet-cogentco-data-plane` | 检查完整 Cogentco 多播切换和报文交付 |
| `mininet-sfc-smoke` | 部署并探测一条保留的 HRL SFC 计划 |

实际命令、输入和输出以 `configs/workflows.yaml` 为唯一准则。

## 组件

| 文件 | 责任 |
| --- | --- |
| `ryu_sft_controller.py` | REST API、SFC 分段、组播组、FlowMod、Barrier、迁移 prepare/commit/abort |
| `ryu_client.py` | Windows 执行器到 Ryu 的客户端协议 |
| `run_profile_mininet.py` | 从 JSON profile 创建 Mininet 网络 |
| `real_topology.py` | 固定端口的真实拓扑定义 |
| `online_hrl_planner.py` | 常驻冻结 HRL 在线规划器和决策计时 |
| `../core/marl/online_parallel_pipeline.py` | 并行完整候选生成、中央 WQMIX/排序接口、联合解码和共享账本原子提交 |
| `online_wqmix_planner.py` | 候选排序、联合解码和版本化提交 |
| `online_migration_wqmix_planner.py` | 迁移专用 WQMIX 规划器 |
| `migration_monitor.py` | 热点、趋势、冷却期和剩余生命周期筛选 |
| `vnf_agent.py` / `vnf_agent_native.c` | 常驻 VNF 控制与转发 |
| `udp_sla_probe.py` / `udp_sla_sender.c` / `udp_sla_receiver.c` | 接收端就绪、绝对时间发送和 SLA 测量 |
| `topologies/*.json` | 拓扑、端口、DC、容量和链路时延 profile |

真实执行主编排器位于 `scripts/run_sdn_runtime_requests.py`。它支持常驻 Agent、
部署/清理 worker、VNF 注册微批、Ryu commit 微批和并行部署流水线，但仍必须
保持每请求的准备、提交、探测和清理顺序。

## 计划模式边界

运行器将下列模式定义为互斥：

- 预生成单计划或候选计划；
- live HRL；
- live WQMIX。

因此当前 HRL + WQMIX + Ryu/Mininet 闭环采用阶段式流程：先导出 HRL 完整
计划并构造 Top-K，再让 WQMIX 在线选择候选，最后提交执行。不要把保留的
`artifacts/plans/hrl_seed7071_rate24/` 误称为当前 rate-24 checkpoint 的实时
输出；其真实来源已记录在 `artifacts/README.md`。

并行候选实验入口为
`scripts/benchmark_parallel_candidate_pipeline.py`。该入口只允许 worker
读取统一快照并返回完整计划候选，中央解码器在版本一致后调用精确事务提交；
worker 不得持有或修改资源账本。示例输出位于
`artifacts/runs/hrl/parallel_pipeline_rate8_*` 和
`artifacts/runs/hrl/parallel_pipeline_rate24_*`。这些结果属于共享账本规划层，
不包含 Ryu/Mininet 的 VNF ready、FlowMod、探针时延、抖动和丢包测量。

## 严格 SLA

请求级严格 SLA 至少要求：

1. 计划和资源提交成功；
2. VNF 与全部接收端 ready；
3. sender 达到规定 offered-load compliance；
4. 每个预期目的端都有完整 probe 结果；
5. 每个目的端的时延、抖动和丢包都满足该请求阈值；
6. SFC/VNF 遍历证据完整。

任何一个目的端失败，请求级严格 SLA 即失败。探针缺失是测量无效，不能从分母
中静默删除。纯仿真的 modeled SLA、风险预测概率和 Mininet UDP probe 的
measured SLA 必须分别报告。

## 输出

新的执行结果统一写入：

```text
artifacts/runs/mininet/<scenario>/<run_name>/
```

至少保留完整命令、请求和 checkpoint hash、拓扑 profile、per-request 结果、
receiver/sender 证据、聚合报告和失败原因。通过检查后才可人工提升到
`artifacts/reference_runs/`。

静态类审计写入：

```text
artifacts/reports/maintenance/unused_class_audit.json
artifacts/reports/maintenance/unused_class_audit.md
```

清理前的长版 SDN 操作记录仍可从
`C:\Users\11353\Desktop\hrl_marl_reconfig_starter_backup_20260826_1636_files`
恢复。
