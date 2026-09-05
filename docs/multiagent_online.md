# 多智能体在线闭环

## 角色

- 中央大脑：`RuleBasedBrainAgent`，按事件把请求交给部署角色，后续可替换为可训练策略。
- 部署智能体：`HRLPlannerAdapter`，调用现有双层 HRL 生成完整映射计划。
- 批次联合调度：每个请求是一个批次智能体，候选由 `CompletePlanCandidateGenerator` 补齐，可选 `WQMIXCandidateRanker` 排序。
- 联合解码器：在有限候选中寻找可行组合。
- 原子账本：`AtomicResourceLedger`，统一检查 CPU、内存、双向带宽和 VNF 实例复用，并处理生命周期释放。
- 迁移/重路由：沿用 `MultiAgentSFTOrchestrator` 的专用角色和安全仲裁器；由 `NODE_OVERLOAD`、`LINK_OVERLOAD` 或 `SLA_ALERT` 事件触发。

## 批量 HRL 与 worker 池

`core/hrl/batched_policy.py` 提供 `BatchedHRLPolicy`。它接收已经由图编码器整理好的 padded tensor：高层一次输入 `[B, gnn_dim]`，低层一次输入 `[B, N, state_dim]` 和 `[B, K]` 候选邻居索引；输出 logits、动作和 mask 后的动作。它不调用 `reset`，不修改环境，也不写资源账本。

`core/marl/deployment_worker_pool.py` 提供 `StatelessHRLWorkerPool`。生产环境建议配置 3 个独立 worker、有限 `max_in_flight`，队列满时显式拒绝或降级，不能无限积压。每个 worker 的 `infer_batch` 必须是无状态函数；旧的 `OnlineLegacyHRLPlanner` 仍是有状态兼容基线，不能仅靠增加线程就宣称并行。

运行检查：

```powershell
C:\Users\11353\.conda\envs\sfc_ppo\python.exe scripts\check_batched_hrl.py
C:\Users\11353\.conda\envs\DL\python.exe scripts\check_batched_hrl_adapter.py
```

## 运行入口

使用 `scripts/run_multiagent_online.py`。下面是 1 条真实 HRL 请求的最小示例：

```powershell
C:\Users\11353\.conda\envs\DL\python.exe scripts\run_multiagent_online.py `
  --legacy-root C:\Users\11353\Desktop\hrl `
  --hrl-checkpoint artifacts\runs\hrl\current_model_rate24_seed7071_100\noIL_rate24\final_model.pth `
  --hrl-data data\sdn_runtime_requests\seed_7071_lifetime50node_rate8\requests.pkl `
  --requests data\sdn_runtime_requests\seed_7071_lifetime50node_rate8\requests.jsonl `
  --profile sdn\topologies\us_backbone_28_bw90.json `
  --max-requests 100 `
  --output artifacts\runs\multiagent_online\rate8_100.json
```

批量策略评分模式可用 `--hrl-mode batched-policy` 启用：

```powershell
C:\Users\11353\.conda\envs\DL\python.exe scripts\run_multiagent_online.py `
  --legacy-root C:\Users\11353\Desktop\hrl `
  --hrl-checkpoint <checkpoint> `
  --hrl-data <requests.pkl> `
  --requests <requests.jsonl> `
  --profile sdn\topologies\us_backbone_28_bw90.json `
  --hrl-mode batched-policy `
  --candidate-mode ultra `
  --max-requests 100 `
  --output artifacts\runs\multiagent_online\batched_policy_100.json
```

该模式加载同一份 HRL checkpoint 的图编码器、高层策略和低层策略，在一个
micro-batch 中批量编码请求状态并对完整候选计划排序。候选计划由纯函数式
`CompletePlanCandidateGenerator` 构造；策略评分不推进旧 HRL 环境，不分配资源。
因此它是“批量 HRL 策略评分 + 完整计划生成”的生产过渡模式，不能描述为已经把
旧 `run_episode` rollout 完全向量化。输出 `adapter.ranker` 中的
`rollout_vectorized=false` 用于防止实验记录误读。

加入已训练的部署 WQMIX 排序器时增加：

```powershell
--wqmix-checkpoint <path-to-wqmix-checkpoint> --wqmix-safety-guard
```

`--wqmix-safety-guard` 会比较 WQMIX 排序和目标函数排序的联合解码接受数，只有不劣于基线时才使用 WQMIX。

默认是因果在线模式：请求进入当前微批后才提交 HRL 后台任务，不读取尚未到达的请求。`--trace-lookahead-prefetch` 只用于已知轨迹的回放吞吐测试，不能用于在线 SLA 或接受率结论。

## 数据含义

输出目录固定为 `artifacts/runs/multiagent_online`（也可用 `--output` 指定文件）。结果包含：中央大脑命令、每批 HRL 准备报告、候选/联合解码动作、原子提交结果、批次时延、生命周期释放数和最终账本余额。

该入口测量的是“真实 HRL 推理 + 候选生成 + 中央联合调度 + 资源准入”。它不会伪造严格 SLA；严格 SLA 必须在 Ryu/Mininet 探针返回真实端到端时延、丢包和 SFC 遍历结果后单独统计。

## 交给 Ryu/Mininet 执行

先把审计结果转换为运行时计划：

```powershell
C:\Users\11353\.conda\envs\DL\python.exe scripts\export_multiagent_sfc_plans.py `
  --input artifacts\runs\multiagent_online\batched_policy_online_100_semantic_features.json `
  --output artifacts\runs\multiagent_online\batched_policy_online_100.sfc_plans.jsonl
```

转换后的 JSONL 可作为 `scripts/run_sdn_runtime_requests.py` 的 `--sfc-plans`
输入，并按该脚本的 Ryu/Mininet、VNF 启动和 UDP probe 参数执行。只有运行时返回
的探针统计才可用于严格 SLA；转换脚本不会把规划接受率复制成 SLA。

## 已验证的规划层结果

使用 `current_model_rate24_seed7071_100/noIL_rate24/final_model.pth`、US backbone
拓扑和 `seed_7071_lifetime50node_rate8` 请求流，在 `batched-policy`、`ultra`、
5 ms 微批配置下运行 100 条因果在线请求：接受率 100%，规划层耗时约 2.14 s，
吞吐约 46.8 req/s，批次总耗时均值 22.3 ms、P95 34.3 ms。结果文件为
`artifacts/runs/multiagent_online/batched_policy_online_100_semantic_features.json`。

该数字只说明规划与账本路径达到目标量级；它不包含 VNF 启动、接收端就绪、
Ryu FlowMod/Barrier 或真实探针，因此不能替代严格 SLA 结果。
