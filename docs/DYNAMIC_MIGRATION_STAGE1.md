# 动态迁移环境：阶段一交付说明

本阶段把迁移仿真从静态 trace 回放改为**动作依赖的动态环境**。后续验证由用户执行；本次只完成代码接线与静态检查。

## 已完成

- 新增 [`envs/dynamic_migration_env.py`](../envs/dynamic_migration_env.py)，继承 `gymnasium.Env`。
- 环境构造时注入 `AtomicResourceLedger`。迁移动作成功后立即以完整新计划调用 `ledger.apply_migration(...)`，原子释放旧节点/链路并提交新占用；失败时账本保持不变。
- 请求到达、离开和 offered traffic 仍由 trace 在当前时间触发。已部署服务的位置、预留带宽、节点 CPU/内存和流体队列由当前动作逐步更新。
- 队列区分两类资源：合同带宽用于 reservation，实际 offered traffic 超出链路容量时形成 backlog；迁移后新流量走新路径，迁移前已产生的链路 backlog 留在原物理链路排空。
- `experiments/telecom_italia_migration/simulator.py` 已删除未来流量读取、全序列归一化和循环历史；预测只使用严格早于当前 `slot` 的滑动窗口。
- 旧 [`envs/migration_wqmix_env.py`](../envs/migration_wqmix_env.py) 已明确标记为 `MigrationReplayEnv` 静态回放。旧训练入口必须显式传 `--allow-static-replay`，避免把回放结果误称为动态环境结果。

## 新环境接口

```python
from envs.dynamic_migration_env import DynamicMigrationEnv

env = DynamicMigrationEnv(
    ledger, requests, profile,
    initial_plans={request_id: plan},
    traffic_trace=traffic_rows,
)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(actions)
```

动作空间为 `MultiDiscrete([top_k + 1] * max_agents)`：`0` 是 no-migration，`1..top_k` 选择当前步骤重新生成的候选计划。每个请求在一个 step 内顺序执行并逐次重新检查账本，因此不会使用旧快照强行覆盖并发提交。

观测包含原有迁移特征，以及 `node_state`、`link_state`、`placements`、`service_mask`、`action_mask` 和当前 `time`。奖励目前是用于调试环境因果性的临时量：队列工作量下降减去失败动作惩罚，不是第二阶段要求的生命周期经济效用。

## 输入约定

- `requests`：至少包含 `id`、`arrival_time`、`leave_time`、`bw_origin`、`vnf`、`source_dpid`、`destination_dpids`；可在请求行附带 `plan`。
- `initial_plans`：只用于 `start_time` 时已经存在的服务，必须与注入账本的 allocation 完全一致。
- `traffic_trace`：行格式为 `{timestamp, request_id, bandwidth_mbps}`，样本到达时间前不会进入观测或队列计算。
- `profile`：沿用现有拓扑格式（`nodes`、`edges`、`dc_nodes_1based`）。

当前候选生成器只产生**内部 VNF stage** 的迁移候选；首个/末个 VNF 暂不生成迁移任务，这是后续算法适配前的明确边界。

## 验证入口

新增 [`scripts/run_dynamic_migration.py`](../scripts/run_dynamic_migration.py)，可输出逐步 `transitions.jsonl` 和 `summary.json`：

```powershell
python scripts/run_dynamic_migration.py `
  --requests <requests.jsonl> `
  --profile <profile.json> `
  --plans <plans.jsonl> `
  --traffic <traffic.jsonl> `
  --policy noop --steps 20
```

`--policy first-feasible` 只用于诊断候选是否真正改变账本和队列，不代表已完成 WQMIX 训练。阶段二的经济奖励、阶段三的独立 no-migration 优势头与 gating、阶段四基线实验、阶段五快照冻结均未在本次实现。
