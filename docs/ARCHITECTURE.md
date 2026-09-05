# 项目架构

## 1. 设计目标

项目把在线多播 SFC 编排拆成三个职责边界：

1. HRL 负责为单个请求生成有质量的完整映射。
2. WQMIX 负责在同一微批的多个请求之间协调候选选择。
3. 原子账本与 Ryu/Mininet 执行器负责硬约束、并发一致性和真实数据面执行。

这种拆分的核心原因是：物理节点不是智能体。WQMIX 中一个临时智能体对应一个
待处理请求，其动作是 `K` 个完整候选之一或 reject。拓扑变大不会直接把
智能体数量扩展到物理节点数；在线智能体上限由微批大小控制。

## 2. 当前数据流

```text
请求 trace / 在线到达
        |
        v
双层 HRL（高层放置，低层逐跳/树扩展）
        |
        | export_hrl_sfc_plans.py 输出完整计划
        v
CompletePlanCandidateGenerator
  - 可接入 HRL baseline plan
  - 补充 balanced/compact/spread/residual 候选
  - 计算 CPU/MEM/有向 BW 足迹与候选特征
        |
        v
微批数据（同一不可变 ResourceSnapshot）
        |
        +--> Oracle/BC 离线监督
        |
        v
WQMIX/基线为每个请求排序候选
        |
        v
JointCandidateDecoder
  - Top-R 有界搜索
  - 聚合检查 CPU/MEM/有向 BW
  - 同批共享 VNF 实例只计一次
  - 始终保留 reject 可行解
        |
        v
commit_decoded_joint_actions
  - 校验账本版本
  - 一次提交完整联合动作
  - 版本过期则零写入并要求重解码
        |
        v
run_sdn_runtime_requests.py
  - VNF 注册/就绪
  - Ryu 批量或分阶段下发
  - UDP 发送与每个目的端探测
  - 到期清理与资源释放
```

### 重要实现边界

`scripts/run_sdn_runtime_requests.py` 明确禁止同时选择以下多个计划模式：

- 预生成单计划或候选计划；
- live `OnlineLegacyHRLPlanner`；
- live `OnlineWQMIXPlanner`。

因此，当前可复现的 HRL -> WQMIX -> Ryu/Mininet 流程是阶段式的：先用 HRL
导出计划，把它们构造成 Top-K 数据，再启动 WQMIX 在线选择。当前不是“请求到达
后先 live HRL、紧接着在同一个进程里 live WQMIX”的串联实现。

## 3. 双层 HRL

### 3.1 所有权

- `train_tahrl.py`：唯一训练入口。
- `core/hrl/`：agent、高低层 policy、目标嵌入、经验回放和训练更新。
- `core/gnn/`：节点/边特征、GAT、TreeTransformer 和可达性特征。
- `envs/sfc_env.py`：HRL 环境入口。
- `envs/modules/HRL_Coordinator.py`：高低层决策协调。
- `envs/modules/high_level_controller.py`：VNF 放置或子目标选择。
- `envs/modules/low_level_controller.py`：下一跳、路径和树扩展动作。
- `envs/modules/AllResourceManager.py`：HRL 环境内的资源和请求生命周期账本。
- `trainer/`：Phase 1 数据收集、Phase 2 模仿学习、Phase 3 RL。

`hrl_training/train_tahrl.py` 仅把旧调用转发到根入口。不得再在
`hrl_training/` 下维护第二套 `core/envs/trainer/configs/topo`。

### 3.2 在线推理

`sdn/online_hrl_planner.py` 的 `OnlineLegacyHRLPlanner` 常驻加载冻结 checkpoint，
提供 `plan_next`、`plan_batch`、`prefetch` 和 `release`。当前实现包含静态拓扑
复用、推理期资源读取优化、K 路径候选过滤、macro-path rollout 和高/低层
决策计时。这些是推理加速手段，不会把双层策略改成单层策略：高层仍选择放置
目标，低层仍对下一跳/路径动作作出策略选择。

缓存只加速不可变拓扑和路径结构；候选在提交前仍须按当前带宽与资源快照重验。

### 3.3 成功语义

一个请求只有在放置、所有 SFC 段、多播树和资源账本均完整后才是映射成功。
环境侧通过请求记录、SFT snapshot 和生命周期事件释放资源。不能把高层选出 DC、
低层连接了部分目的节点或生成了不完整计划算作成功。

## 4. Top-K 与 WQMIX

### 4.1 完整候选

`core/marl/deployment_topk.py` 中的 `CompletePlanCandidateGenerator` 生成并验证
完整计划。每个候选包含：

- VNF 到 DC 的放置；
- VNF 之间的有序路径段；
- 最后一阶段到全部目的节点的多播树；
- CPU、内存、每个有向链路带宽和 VNF 实例需求；
- 时延、压力、资源代价等候选指标。

同一个微批必须基于同一不可变资源快照生成 mask 和候选特征。reject 动作始终
有效。Top-K 数据中的 `source` 是审计元数据，不能作为模型识别 checkpoint 或
算法来源的输入特征。

### 4.2 学习层

`core/marl/batch_deployment_wqmix.py` 提供请求候选 Q 网络、QMIX mixer、Weighted
QMIX learner 和 `AtomicResourceLedger`。WQMIX 输出的是候选偏好，不是硬约束
最终裁决。

训练和在线执行共用：

- `core/marl/joint_candidate_decoder.py`：在固定 Top-R、时间和检查次数预算内，
  将各请求排序解成一个联合可行动作；
- `core/marl/joint_candidate_transaction.py`：按预期 snapshot version 精确提交；
- `core/marl/deployment_env.py`：用同一解码和提交语义计算动作后结果。

这避免了“训练假设同时动作、线上却串行回退”的主要语义差异。解码器超时仍有
reject 组成的可行 incumbent；超时或预算耗尽必须记录，不能静默宣称全局最优。

### 4.3 原子账本

`AtomicResourceLedger` 是部署 WQMIX 路径上的硬约束边界：

- CPU、内存和带宽不足时不提交；
- 带宽以 `(u,v)` 有向边分别记账，符合 Mininet 全双工链路语义；
- 相同 `(node, vnf_type)` 的活跃实例可复用，节点资源只在首次创建时计费；
- 同一请求的资源在生命周期结束或失败回滚时释放；
- 联合提交前检查 snapshot version，防止用过期快照覆盖并发变化。

HRL 环境的 `AllResourceManager` 和部署 MARL 的 `AtomicResourceLedger` 是两个
不同运行边界。跨边界计划必须通过明确的 footprint/plan 转换，不能默认两者状态
自动同步。

## 5. Ryu/Mininet 执行层

### 5.1 Windows 编排与 WSL 数据面

`scripts/run_sdn_runtime_requests.py` 是真实执行主入口。Windows 端组织 trace、
计划、并发 worker 和结果；WSL 中运行 Ryu、Open vSwitch、Mininet、VNF Agent
和 UDP probe。`sdn/topologies/*.json` 定义交换机、端口、DC、链路容量和时延。

主要组件：

- `sdn/ryu_sft_controller.py`：REST API、SFC 段、组播组、流表、Barrier、
  prepare/commit/abort 迁移。
- `sdn/ryu_client.py`：执行端与 Ryu 的客户端协议。
- `sdn/vnf_agent.py`、`sdn/vnf_agent_native.c`：常驻 VNF 控制与转发。
- `sdn/udp_sla_probe.py`、`sdn/udp_sla_sender.c`、`sdn/udp_sla_receiver.c`：
  接收端就绪、绝对时间发送和 SLA 测量。
- `sdn/run_profile_mininet.py`、`sdn/real_topology.py`：Mininet profile 启动。

### 5.2 部署顺序与并发

执行器支持常驻 Agent、VNF 注册并发上限、注册/注销微批、Ryu commit 微批、
部署 worker 和清理 worker。`--parallel-deployment-pipeline` 可以重叠邻居准备、
VNF 注册和流表提交，但不取消：

- 每请求的生命周期顺序；
- VNF ready 检查；
- 下游优先、入口最后的流表切换；
- 账本拒绝、失败回滚和到期清理；
- 完整探针作为有效测量的要求。

并发数增大不保证吞吐提高。单机 Mininet 中 Ryu、OVS、VNF 和探针共享 CPU，必须
用 setup latency、队列等待、过期前启动率和严格 SLA 一起判断。

## 6. SLA 的三层语义

### 6.1 模型化候选风险

候选生成器和纯仿真可用拓扑传播时延、VNF 处理时延、链路利用率及轻量队列模型
估计 SLA。它用于 mask、排序、Oracle 标签或仿真筛选，不是实际数据面测量。

### 6.2 监督风险预测器

`core/marl/sla_risk_predictor.py` 的 `runtime_sla_risk_mlp_v2` 使用 32 个提交前
特征，预测“发送端成功启动条件下，请求级严格 SLA 失败概率”。运行时实测的
delay、loss、部署耗时和 receiver ready 不能作为输入，避免标签泄漏。OOD 分数
过高时预测器不改变候选排序。默认用途是保守重排；概率硬拒绝门限默认关闭。

### 6.3 真实严格 SLA

Ryu/Mininet 执行中，一个请求的严格 SLA 至少要求：

- sender 成功启动并达到规定 offered-load compliance；
- 每个预期目的端都有完整结果；
- 每个目的端的时延、抖动和丢包都满足该请求 QoS 阈值；
- SFC/VNF 转发记录完整。

任何目的端失败都会导致请求级严格 SLA 失败。探针缺失使该次测量无效，不能从
分母中悄悄删除。跨物理主机测单向时延还需要 PTP 等时钟同步；单机 Mininet
共享主机时钟不代表多机环境已经同步。

## 7. VNF 迁移

迁移是初始部署后的独立决策，不等同于只改变多播路径的重路由。

```text
在线账本与活跃计划
    -> OnlineMigrationMonitor / EWMA 趋势筛选
    -> 只把过载、剩余生命周期、冷却期均通过的 VNF 变成任务
    -> 最多 max_agents 个临时迁移智能体
    -> 目标 DC Top-K + WQMIX/基线排序
    -> 共用有界联合解码器
    -> VNF 新端准备 + Ryu prepare/commit/abort
    -> drain 旧路径并释放旧实例
```

相关代码位于 `core/marl/migration_*`、`sdn/migration_monitor.py` 和
`sdn/online_migration_wqmix_planner.py`。部署 checkpoint 与迁移 checkpoint 类型
不同，在线迁移规划器会拒绝错误类型。预测驱动迁移当前要求在线部署 WQMIX 和
迁移专用 checkpoint；并发执行还受 `max_inflight` 限制。执行器通过
`migration_execution_waves()` 把同请求或同源 DC 的任务分到不同波次，波次内
才并发执行，避免共享实例和源端切换互相干扰。

当前保留迁移数据的 nominal train/validation/test 来自同一历史运行报告，并在
真实计划上注入合成热点。它们只适合 smoke 和算法接口对比。要形成论文结论，
必须采集独立 seed 的真实运行 trace，并使用动作会改变后续状态的迁移环境。

## 8. 纯仿真

`scripts/run_pure_sfc_simulation.py` 复用真实拓扑 profile、完整候选生成器和原子
账本，模拟到达、生命周期释放、资源预留、M/M/1/K 风格排队/丢包以及每时间槽
成本。它适合：

- 快速检查资源接受率和候选覆盖率；
- 做参数扫描与算法初筛；
- 验证生命周期是否返还资源；
- 在相同模型假设下比较候选选择方法。

它不能证明：Ryu 流表下发时延、VNF 注册耗时、主机调度抖动、真实 P95/P99
时延或真实丢包。最终 SLA 结论必须回到 Ryu/Mininet A/B 实验。

## 9. 代码所有权原则

- 新 HRL 逻辑只改根 `core/`、`envs/`、`trainer/` 和 `train_tahrl.py`。
- 新部署/迁移联合学习逻辑放 `core/marl/`。
- 真实执行协议放 `sdn/`，实验编排放 `scripts/`。
- 可复现实验输入放 `data/`；生成结果放 `artifacts/runs/`。
- 经过筛选且需要长期保留的权重/报告才进入 `artifacts/reference_runs/`。
- 不向 `outputs/` 增加新的正式实验依赖；该目录仅兼容旧脚本。

## 10. 分层多智能体编排

`core/marl/orchestration/` 在现有 HRL、迁移和重路由能力之上提供中央大脑、
结构化消息协议和角色级 Skill 权限。初始映射继续调用双层 HRL；中央大脑只决定
任务类型、目标请求和负责角色，专业 Agent 返回提案，资源修改仍集中在安全执行
边界。详细接口、文件归属和验证方式见
[`MULTIAGENT_ORCHESTRATION.md`](MULTIAGENT_ORCHESTRATION.md)。
