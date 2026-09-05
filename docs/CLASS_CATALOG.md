# 当前项目类目录
生成范围：排除 `legacy_hrl_runtime/`、`tests/`、`artifacts/` 和虚拟环境；共列出当前主代码类。
## 阅读方式
大脑负责调度模块；模块组织多个技能；技能调用预测、候选生成、资源校验、执行和验证能力。
## 大脑与编排
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `core/marl/online_parallel_pipeline.py` | 117 | `CandidateGeneration` | Pure result returned by one candidate worker. |
| `core/marl/online_parallel_pipeline.py` | 134 | `WQMIXCandidateRanker` | Central WQMIX ranker for :class:`CandidateGeneration` rows. |
| `core/marl/online_parallel_pipeline.py` | 251 | `ObjectiveCandidateRanker` | Deterministic objective-order baseline for safety gating. |
| `core/marl/online_parallel_pipeline.py` | 275 | `GuardedWQMIXRanker` | Use WQMIX only when its decoded acceptance is no worse than baseline. |
| `core/marl/online_parallel_pipeline.py` | 341 | `ParallelBatchResult` | Observable result of one central-ledger batch attempt. |
| `core/marl/online_parallel_pipeline.py` | 374 | `CentralSharedLedgerPipeline` | Run parallel candidate generation with one authoritative ledger. |
| `core/marl/orchestration/agents.py` | 21 | `BrainPolicyConfig` | 代码类 `BrainPolicyConfig`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/agents.py` | 26 | `RuleBasedBrainAgent` | Event-driven first policy with the same command surface as a learned brain. |
| `core/marl/orchestration/agents.py` | 99 | `SkillAgent` | A role specialist that may invoke only its explicitly assigned skills. |
| `core/marl/orchestration/deployment_module.py` | 18 | `DeploymentModule` | Route arrived requests through Brain, Deployment Agent and HRL Skill. |
| `core/marl/orchestration/deployment_module.py` | 118 | `BrainManagedDeploymentPlanner` | Compatibility wrapper preserving the runtime planner API. |
| `core/marl/orchestration/hrl_adapter.py` | 27 | `HRLPreparationReport` | Auditable result of the stateful HRL preparation pass. |
| `core/marl/orchestration/hrl_adapter.py` | 48 | `HRLPlannerAdapter` | Use a real HRL checkpoint as the first candidate for each request. |
| `core/marl/orchestration/orchestrator.py` | 30 | `MultiAgentSFTOrchestrator` | Coordinate specialists while keeping all resource mutation centralized. |
| `core/marl/orchestration/protocol.py` | 10 | `AgentRole` | 代码类 `AgentRole`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/protocol.py` | 17 | `EventType` | 代码类 `EventType`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/protocol.py` | 26 | `CommandType` | 代码类 `CommandType`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/protocol.py` | 36 | `OrchestrationEvent` | 代码类 `OrchestrationEvent`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/protocol.py` | 47 | `SystemObservation` | 代码类 `SystemObservation`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/protocol.py` | 57 | `BrainCommand` | 代码类 `BrainCommand`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/protocol.py` | 70 | `SkillProposal` | 代码类 `SkillProposal`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/protocol.py` | 86 | `ExecutionResult` | 代码类 `ExecutionResult`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/runtime_module.py` | 24 | `RuntimeReconfigurationModule` | Route live events while leaving mutation to the existing safe executor. |
| `core/marl/orchestration/skills.py` | 16 | `SkillContext` | 代码类 `SkillContext`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/skills.py` | 24 | `AgentSkill` | 代码类 `AgentSkill`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/skills.py` | 32 | `SkillRegistry` | Role-scoped registry; agents cannot invoke another role's skills. |
| `core/marl/orchestration/skills.py` | 63 | `HRLDeploymentSkill` | Use the project's two-level HRL planner for initial SFT mapping. |
| `core/marl/orchestration/skills.py` | 169 | `VNFMigrationPlanningSkill` | 代码类 `VNFMigrationPlanningSkill`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/skills.py` | 193 | `TreeReroutePlanningSkill` | 代码类 `TreeReroutePlanningSkill`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/orchestration/skills.py` | 222 | `RuntimeMigrationExecutionSkill` | Authorize the existing live make-before-break migration pipeline. |
| `core/marl/orchestration/skills.py` | 255 | `RuntimeTreeRerouteExecutionSkill` | Authorize the existing live gated tree-reroute pipeline. |
| `core/marl/role_coordinator.py` | 12 | `CoordinationResult` | 代码类 `CoordinationResult`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/role_coordinator.py` | 31 | `RoleCoordinator` | Resolve migration/reroute proposals from fixed role agents. |
## 动态环境与执行
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `core/marl/migration_scheduler.py` | 17 | `MigrationTask` | 代码类 `MigrationTask`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/migration_scheduler.py` | 44 | `EWMATrendForecaster` | Small online forecaster with an EWMA level and bounded linear trend. |
| `envs/dynamic_migration_env.py` | 27 | `DynamicMigrationEnv` | Own a supplied AtomicResourceLedger for a resettable simulation episode. |
| `envs/migration_wqmix_env.py` | 43 | `MigrationStepResult` | 代码类 `MigrationStepResult`，用于承载该文件中的相关数据或逻辑。 |
| `envs/migration_wqmix_env.py` | 50 | `MigrationReplayEnv` | Static trace replay for policy evaluation and counterfactual decoding. |
| `envs/migration_wqmix_env.py` | 173 | `MigrationWQMIXEnv` | Deprecated compatibility alias for the old static replay environment. |
| `envs/modules/AllResourceManager.py` | 63 | `DeployResult` | VNF 部署结果，替代旧的 bool 返回值. |
| `envs/modules/AllResourceManager.py` | 75 | `VNFBinding` | 请求级别：记录该请求绑定了哪个VNF实例. |
| `envs/modules/AllResourceManager.py` | 87 | `EdgeAllocation` | 请求级别：记录该请求占用了哪条边的带宽. |
| `envs/modules/AllResourceManager.py` | 96 | `RequestRecord` | 请求级状态（单一真相）. |
| `envs/modules/AllResourceManager.py` | 122 | `VNFInstanceRecord` | 实例级状态（单一真相）. |
| `envs/modules/AllResourceManager.py` | 135 | `SharedResourcePool` | 共享资源池 - 底层物理资源管理（原子操作）. |
| `envs/modules/AllResourceManager.py` | 401 | `RequestLifecycleManager` | 请求生命周期管理器 - 纯仿真时间版 - 完全依赖外部传入的 current_time（仿真时间） - 移除了 cleanup_interval 节流，每个时间步都会检查 - register_request 必须传入 arrival_time 和 lifetime - 所有释放操作都基于记录的资源量. |
| `envs/modules/AllResourceManager.py` | 560 | `RequestHandler` | 请求处理器 - 负责与请求相关的业务逻辑 包括部署、归档、状态查询等辅助方法. |
| `envs/modules/AllResourceManager.py` | 637 | `FusedResourceManager` | 融合版资源管理器 - 外观类 组合核心组件，提供统一接口，保持与原有代码兼容. |
| `envs/modules/controller_shared_helper.py` | 30 | `ControllerSharedHelper` | Shared env-bound helper used by controllers and coordinator. |
| `envs/modules/data_loader.py` | 10 | `DataLoader` | 数据加载器：最终修复版 V3 1. |
| `envs/modules/event_handler.py` | 10 | `EventHandler` | 代码类 `EventHandler`，用于承载该文件中的相关数据或逻辑。 |
| `envs/modules/high_level_controller.py` | 85 | `HighLevelController` | 楂樺眰浜や簰鎺у埗鍣?- HQDQN 瀵归綈浼樺寲鐗?. |
| `envs/modules/HRL_Coordinator.py` | 93 | `HRL_Coordinator` | 代码类 `HRL_Coordinator`，用于承载该文件中的相关数据或逻辑。 |
| `envs/modules/low_level_controller.py` | 113 | `LowLevelController` | 低层执行控制器 - TA-HRL v4 顶级架构优化版. |
| `envs/modules/reconfiguration_manager.py` | 35 | `Hotspot` | 代码类 `Hotspot`，用于承载该文件中的相关数据或逻辑。 |
| `envs/modules/reconfiguration_manager.py` | 44 | `SftRisk` | 代码类 `SftRisk`，用于承载该文件中的相关数据或逻辑。 |
| `envs/modules/reconfiguration_manager.py` | 55 | `DelayBreakdown` | 代码类 `DelayBreakdown`，用于承载该文件中的相关数据或逻辑。 |
| `envs/modules/reconfiguration_manager.py` | 77 | `ReconfigAction` | 代码类 `ReconfigAction`，用于承载该文件中的相关数据或逻辑。 |
| `envs/modules/reconfiguration_manager.py` | 92 | `ReconfigurationManager` | Low-disturbance SFT reconfiguration baseline manager. |
| `envs/modules/TimeSlotManager.py` | 24 | `TimeSlotManager` | 独立时间槽管理器 Parameters ---------- env : SFC_HIRL_Env 实例（用于读取 resource_mgr、online_mode 等） config : 环境配置字典. |
| `envs/modules/tools.py` | 25 | `SFCToolkit` | SFC 环境辅助工具箱. |
| `envs/sfc_env.py` | 101 | `SimpleTopologyManager` | 增强版简化拓扑管理器 补全 GNN 特征提取所需的度数和介数计算接口. |
| `envs/sfc_env.py` | 128 | `ExpertWrapper` | 包装 MSFCE_Solver，适配 BackupPolicy. |
| `envs/sfc_env.py` | 141 | `SimpleDataLoader` | 简化的数据加载器（兼容层） 当前主链使用 DataLoader(self. |
| `envs/sfc_env.py` | 182 | `SFC_HIRL_Env` | 代码类 `SFC_HIRL_Env`，用于承载该文件中的相关数据或逻辑。 |
| `envs/sft_role_marl_env.py` | 26 | `SFTRoleMARLEnv` | One-step role-MARL reconfiguration runner. |
| `sdn/diamond_topology.py` | 6 | `DiamondTopo` | 代码类 `DiamondTopo`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/migration_monitor.py` | 13 | `OnlineMigrationMonitor` | Return the minimum useful set of internal VNF stages to migrate. |
| `sdn/online_batched_hrl_planner.py` | 13 | `OnlineBatchedHRLPlanner` | Plan arrived requests without running the legacy environment rollout. |
| `sdn/online_hrl_planner.py` | 33 | `_LegacyDecisionTimingProbe` | Collect granular timings when the loaded legacy coordinator lacks them. |
| `sdn/online_hrl_planner.py` | 119 | `OnlineLegacyHRLPlanner` | Load one legacy HRL checkpoint and infer one request at a time. |
| `sdn/online_hrl_process.py` | 89 | `IsolatedOnlineLegacyHRLPlanner` | Synchronous RPC facade for one stateful planner subprocess. |
| `sdn/online_migration_wqmix_planner.py` | 30 | `OnlineMigrationWQMIXPlanner` | Select jointly feasible target DCs for at most ``max_agents`` VNFs. |
| `sdn/online_wqmix_planner.py` | 42 | `OnlineWQMIXPlanner` | Generate/select and atomically reserve complete plans after arrival. |
| `sdn/real_topology.py` | 26 | `ProfileTopo` | 代码类 `ProfileTopo`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/run_profile_mininet.py` | 64 | `RuntimeCommandServer` | 代码类 `RuntimeCommandServer`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/ryu_client.py` | 11 | `RyuSFTClient` | 代码类 `RyuSFTClient`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/ryu_sft_controller.py` | 47 | `SFTControllerRest` | Small REST adapter used by PyCharm and the QMIX executor. |
| `sdn/ryu_sft_controller.py` | 162 | `SFTController` | Learning switch plus SFT multicast groups and port utilization. |
| `sdn/udp_sla_probe.py` | 29 | `_Timespec` | 代码类 `_Timespec`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/vnf_container_manager.py` | 22 | `ResourceMapping` | 代码类 `ResourceMapping`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/vnf_container_manager.py` | 52 | `VNFContainerSpec` | 代码类 `VNFContainerSpec`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/vnf_container_manager.py` | 76 | `VNFChainPlan` | 代码类 `VNFChainPlan`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/vnf_container_manager.py` | 95 | `ContainerBackend` | 代码类 `ContainerBackend`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/vnf_container_manager.py` | 103 | `DockerCLIBackend` | 代码类 `DockerCLIBackend`，用于承载该文件中的相关数据或逻辑。 |
| `sdn/vnf_container_manager.py` | 203 | `VNFContainerManager` | 代码类 `VNFContainerManager`，用于承载该文件中的相关数据或逻辑。 |
## 迁移、资源账本与候选
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `core/marl/batch_deployment_wqmix.py` | 37 | `VNFInstanceRequirement` | 代码类 `VNFInstanceRequirement`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/batch_deployment_wqmix.py` | 45 | `ResourceFootprint` | Sparse resources required by one complete deployment candidate. |
| `core/marl/batch_deployment_wqmix.py` | 164 | `ResourceSnapshot` | 代码类 `ResourceSnapshot`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/batch_deployment_wqmix.py` | 340 | `BatchCandidateQNetwork` | Shared parameterized Q-network for variable request candidates. |
| `core/marl/batch_deployment_wqmix.py` | 404 | `MaskedQMixer` | Monotonic mixer supporting padded request agents. |
| `core/marl/batch_deployment_wqmix.py` | 439 | `WeightedQMIXLearner` | OW-QMIX-style learner for padded request micro-batches. |
| `core/marl/batch_deployment_wqmix.py` | 615 | `AtomicResourceLedger` | Versioned hard-resource ledger with ranked-candidate fallback. |
| `core/marl/deployment_topk.py` | 31 | `DeploymentCandidate` | 代码类 `DeploymentCandidate`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/deployment_topk.py` | 112 | `CompletePlanCandidateGenerator` | Generate diverse full SFC plans against one immutable snapshot. |
| `core/marl/joint_candidate_decoder.py` | 25 | `JointDecodeResult` | 代码类 `JointDecodeResult`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/joint_candidate_decoder.py` | 43 | `_CandidateContribution` | Precomputed sparse contribution for one candidate. |
| `core/marl/joint_candidate_decoder.py` | 58 | `_IncrementalAggregate` | Mutable aggregate for bounded candidate checks. |
| `core/marl/migration_baseline_learners.py` | 26 | `BehaviorCloningLearner` | Shared candidate policy trained from joint-oracle actions. |
| `core/marl/migration_baseline_learners.py` | 73 | `IndependentDQNLearner` | Fitted one-step Q baseline for ephemeral migration-task agents. |
| `core/marl/migration_baseline_learners.py` | 114 | `CentralizedValueNetwork` | 代码类 `CentralizedValueNetwork`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/migration_baseline_learners.py` | 129 | `MAPPOLearner` | Shared actor and centralized critic with clipped PPO updates. |
| `core/marl/migration_baselines.py` | 131 | `MigrationOracleResult` | 代码类 `MigrationOracleResult`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/migration_baselines.py` | 143 | `MigrationBatchMILPOracle` | Exact small-batch upper bound under the shared migration reward. |
| `core/marl/migration_benefit_predictor.py` | 45 | `MigrationBenefitNet` | Small CPU-friendly binary classifier for the online migration gate. |
| `core/marl/migration_benefit_predictor.py` | 71 | `MigrationBenefitEstimate` | 代码类 `MigrationBenefitEstimate`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/migration_benefit_predictor.py` | 156 | `MigrationBenefitPredictor` | Read-only predictor used to reject migrations without expected benefit. |
| `core/marl/migration_candidates.py` | 28 | `MigrationCandidate` | 代码类 `MigrationCandidate`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/migration_candidates.py` | 76 | `MigrationCandidateGenerator` | Rebuild adjacent SFC segments for each feasible target DC. |
| `core/marl/migration_dataset.py` | 43 | `MigrationFeatureNormalizer` | 代码类 `MigrationFeatureNormalizer`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/migration_dataset.py` | 108 | `MigrationTransitionDataset` | Padded consecutive batch transitions with oracle teacher actions. |
## 强化学习与神经网络
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `core/gnn/ablation_encoder.py` | 25 | `AblationEncoder` | 统一消融实验编码器 通过 variant 参数切换不同的消融配置，保持其他超参完全一致. |
| `core/gnn/feature_builder.py` | 12 | `GNNFeatureBuilder` | 代码类 `GNNFeatureBuilder`，用于承载该文件中的相关数据或逻辑。 |
| `core/gnn/gat_encoder.py` | 7 | `GATEncoder` | Plain GAT baseline for the MSFT-HIRL w/ GAT ablation. |
| `core/gnn/tree_transformer_encoder.py` | 9 | `TreeTransformerEncoder` | [v3 / 方向三] 在 v2(单拓扑流 + 目的集合注意力)基础上，新增 “放置↔路由耦合编码”：从同一次 GNN 输出分出放置语义 h_place 与路由语义 h_route 两个投影头，再用低秩双线性项刻画二者耦合，并可选输出辅助预测 (在某节点放 VNF 会引入多少增量路由代价)。 设计要点(对应讨论的六步)： - 只跑一遍 GNN，双头是投影而非独立消息传递流(规避树感知双流翻车史)。 - 耦合是候选集约束在“放置/路由联合可行性”上的延伸，不是第二创新。 - coupling_mode 三档对应 A/B/C 消融，且 B 与 C 参数自动对齐： 'off' = A：单流，等价 v2(无双头、无耦合) 'dual' = B：双头 h_place/h_route，但只做线性相加组合(无交叉项) 'full' = C：双头 + 低秩双线性耦合(有交叉项) ← 方向三完整版 - use_aux_head：辅助监督头，缓解耦合项只靠 RL 回报学不动的问题。. |
| `core/hrl/agent.py` | 25 | `HRLAgent` | Hierarchical RL Agent（重构版） 继承顺序决定MRO：Base最先提供__init__，其余Mixin只包含方法。 外部代码无需改动，接口完全向后兼容。. |
| `core/hrl/agent.py` | 37 | `GoalConditionedHRLAgent` | 向后兼容的旧类名. |
| `core/hrl/agent_action.py` | 24 | `HRLAgentAction` | 负责： - select_action: 对外统一接口 - _need_new_subgoal: 触发条件判断 - _select_subgoal: High-Level epsilon-greedy - _select_start_node / _get_default_start_node / _build_tree_mask - _select_low_action: Low-Level执行 - _get_local_embedding / _get_graph_embedding: 嵌入计算 - _generate_goal_embedding / _generate_and_encode_subgoal: 向后兼容. |
| `core/hrl/agent_base.py` | 18 | `HRLAgentBase` | 负责： - __init__: 所有网络/优化器/buffer/状态变量初始化 - train / eval / save / load - reset_network_parameters - register_encoder_to_optimizer. |
| `core/hrl/agent_memory.py` | 24 | `HRLAgentMemory` | 负责： - store_transition_high: 存储高层经验 - store_transition_low: 存储低层经验 + success_memory + 内在奖励 - store_transition: 向后兼容接口. |
| `core/hrl/agent_train.py` | 27 | `HRLAgentTrain` | 负责： - update / update_policies: 训练调度 - _update_high_level: High-Level Double DQN - _update_low_level: Low-Level Double DQN + success mix + Q监控 - _soft_update_target_networks / _hard_update_target_networks - _update_epsilon / update_epsilon - _log_training_stats. |
| `core/hrl/batched_policy.py` | 18 | `BatchedHighOutput` | 代码类 `BatchedHighOutput`，用于承载该文件中的相关数据或逻辑。 |
| `core/hrl/batched_policy.py` | 26 | `BatchedLowOutput` | 代码类 `BatchedLowOutput`，用于承载该文件中的相关数据或逻辑。 |
| `core/hrl/batched_policy.py` | 47 | `BatchedHRLPolicy` | Vectorized high/low policy heads for already encoded states. |
| `core/hrl/elite_buffer.py` | 9 | `EliteBuffer` | 按 episode 总奖励排序的精英经验池。 高奖励 episode 的 transition 被优先采样，引导策略复现成功轨迹。. |
| `core/hrl/goal_embedding.py` | 34 | `EnhancedRelativeGoalEmbedding` | 增强版相对目标嵌入 改进点： 1. |
| `core/hrl/goal_embedding.py` | 145 | `AdaptiveSubgoalEmbedding` | 自适应子目标嵌入 改进点： 1. |
| `core/hrl/goal_embedding.py` | 295 | `EnhancedOptionEmbedding` | 增强版 Option 嵌入 改进点： 1. |
| `core/hrl/goal_embedding.py` | 456 | `IterativeHybridGoalEmbedding` | 迭代优化的混合 Goal Embedding 改进点： 1. |
| `core/hrl/high_policy.py` | 32 | `HighLevelPolicy` | 代码类 `HighLevelPolicy`，用于承载该文件中的相关数据或逻辑。 |
| `core/hrl/low_policy.py` | 39 | `GoalConditionedLowLevelPolicy` | Goal-Conditioned Low-Level Policy (TA-HRL v4. |
| `core/hrl/low_policy.py` | 407 | `LowLevelPolicy` | 低层策略网络（向后兼容版） 底层已映射为 GoalConditionedLowLevelPolicy。. |
| `core/hrl/prioritized_buffer.py` | 9 | `PrioritizedReplayBuffer` | Prioritized Experience Replay (PER) 按TD-error优先采样，让模型更多学习难样本。. |
| `core/marl/qmix.py` | 14 | `QMixer` | Monotonic state-conditioned mixer from per-role Q values to Q_tot. |
| `core/marl/qmix.py` | 35 | `JointReplayBuffer` | 代码类 `JointReplayBuffer`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/qmix.py` | 70 | `QMIXLearner` | Joint learner that optimizes role Q-networks through a QMIX mixer. |
| `core/marl/trainable_role_agents.py` | 27 | `DQNRoleConfig` | 代码类 `DQNRoleConfig`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/trainable_role_agents.py` | 40 | `RoleReplayBuffer` | 代码类 `RoleReplayBuffer`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/trainable_role_agents.py` | 67 | `RoleQNetwork` | 代码类 `RoleQNetwork`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/trainable_role_agents.py` | 82 | `TrainableDQNRoleAgent` | Base class for small DQN role agents. |
| `core/marl/trainable_role_agents.py` | 169 | `TrainableSFTSelectionAgent` | DQN selector: action 0 is noop, action 1. |
| `core/marl/trainable_role_agents.py` | 246 | `TrainableVNFMigrationAgent` | DQN migration role: action 0 no-op, action 1 execute planned migration. |
| `core/marl/trainable_role_agents.py` | 298 | `TrainableTreeRerouteAgent` | DQN reroute role: action 0 no-op, action 1 execute planned reroute. |
## 数据、预测与评估
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `core/marl/batched_hrl_adapter.py` | 61 | `BatchedGraphState` | 代码类 `BatchedGraphState`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/batched_hrl_adapter.py` | 67 | `BatchedTopologyCollator` | Build a padded, request-conditioned graph batch from one snapshot. |
| `core/marl/batched_hrl_adapter.py` | 246 | `BatchedHRLPolicyRanker` | Rank complete plans with one batched high/low HRL policy pass. |
| `core/marl/batched_hrl_adapter.py` | 380 | `BatchedHRLCandidateAdapter` | Pure complete-plan generator plus batched HRL policy ranker. |
| `core/marl/deployment_dataset.py` | 24 | `FeatureNormalizer` | 代码类 `FeatureNormalizer`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/deployment_dataset.py` | 132 | `DeploymentOracleDataset` | Fixed-shape request-agent batches with Oracle action targets. |
| `core/marl/deployment_env.py` | 41 | `BatchDeploymentEnv` | Regenerate candidates after every policy action. |
| `core/marl/deployment_oracle.py` | 22 | `OracleSolution` | 代码类 `OracleSolution`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/deployment_oracle.py` | 86 | `DeploymentBatchOracle` | Solve one request batch as a binary resource-allocation MILP. |
| `core/marl/deployment_worker_pool.py` | 16 | `WorkerPoolStats` | 代码类 `WorkerPoolStats`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/deployment_worker_pool.py` | 24 | `StatelessHRLWorkerPool` | Run pure batch inference on independent workers with backpressure. |
| `core/marl/role_agents.py` | 16 | `RoleDecision` | 代码类 `RoleDecision`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/role_agents.py` | 37 | `SFTSelectionAgent` | Select one target SFT from the Top-K risky SFT list. |
| `core/marl/role_agents.py` | 72 | `VNFMigrationAgent` | Decide whether to propose VNF migration for the selected SFT. |
| `core/marl/role_agents.py` | 104 | `TreeRerouteAgent` | Decide whether to propose local tree-edge rerouting. |
| `core/marl/sla_risk_calibration.py` | 241 | `SlaRiskEstimate` | 代码类 `SlaRiskEstimate`，用于承载该文件中的相关数据或逻辑。 |
| `core/marl/sla_risk_calibration.py` | 255 | `EmpiricalSlaRiskCalibrator` | Read-only empirical risk lookup with hierarchical shrinkage. |
| `core/marl/sla_risk_predictor.py` | 64 | `RuntimeSlaRiskNet` | Small CPU-friendly MLP used for online candidate scoring. |
| `core/marl/sla_risk_predictor.py` | 214 | `RuntimeSlaRiskPredictor` | Read-only supervised probability predictor for online planning. |
| `experiments/telecom_italia_migration/simulator.py` | 31 | `ExperimentConfig` | 代码类 `ExperimentConfig`，用于承载该文件中的相关数据或逻辑。 |
| `experiments/telecom_italia_migration/simulator.py` | 95 | `VNF` | 代码类 `VNF`，用于承载该文件中的相关数据或逻辑。 |
| `experiments/telecom_italia_migration/simulator.py` | 105 | `SFC` | 代码类 `SFC`，用于承载该文件中的相关数据或逻辑。 |
| `experiments/telecom_italia_migration/telecom_data.py` | 25 | `SourceFile` | 代码类 `SourceFile`，用于承载该文件中的相关数据或逻辑。 |
| `trainer/phase1_collector.py` | 16 | `Phase1ExpertCollector` | Phase 1 专家数据收集器（时间槽版本 - 最终修复版） 修复问题： 1. |
| `trainer/phase2_il_trainer.py` | 20 | `EarlyStopping` | 代码类 `EarlyStopping`，用于承载该文件中的相关数据或逻辑。 |
| `trainer/phase2_il_trainer.py` | 45 | `ExpertDataset` | 支持两种格式： 1. |
| `trainer/phase2_il_trainer.py` | 139 | `Phase2ILTrainer` | 代码类 `Phase2ILTrainer`，用于承载该文件中的相关数据或逻辑。 |
| `trainer/phase3_rl_trainer.py` | 31 | `Phase3RLTrainer` | Phase 3: RL Trainer with HRL Coordinator (Clean Logs). |
| `trainer/phase3_rl_trainer_no_checkpoint.py` | 11 | `Phase3RLTrainerNoCheckpoint` | Drop-in Phase3 trainer that does not write checkpoint/model files. |
| `trainer/role_marl_trainer.py` | 18 | `RoleMARLTrainer` | Train role-DQN agents from online reconfiguration transitions. |
| `trainer/role_reconfig_eval.py` | 21 | `RoleReconfigEvaluator` | Run deployment episodes and periodically trigger role collaboration. |
| `trainer/training_analyzer.py` | 38 | `FailReason` | 代码类 `FailReason`，用于承载该文件中的相关数据或逻辑。 |
| `trainer/training_analyzer.py` | 52 | `EpisodeRecord` | 代码类 `EpisodeRecord`，用于承载该文件中的相关数据或逻辑。 |
| `trainer/training_analyzer.py` | 72 | `TrainingAnalyzer` | 代码类 `TrainingAnalyzer`，用于承载该文件中的相关数据或逻辑。 |
## 传统部署与求解器
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `core/baselines/feasible_paths.py` | 19 | `FeasiblePathOracle` | Cache up to ``k`` hop-shortest paths and select a BW-feasible one. |
| `core/expert/expert_msfce/core/solver.py` | 39 | `SolverConfig` | 集中式配置管理. |
| `core/expert/expert_msfce/core/solver.py` | 97 | `MetricsCollector` | 性能指标收集器. |
| `core/expert/expert_msfce/core/solver.py` | 164 | `CacheManager` | 缓存管理器. |
| `core/expert/expert_msfce/core/solver.py` | 211 | `LinkCache` | 链路查找缓存. |
| `core/expert/expert_msfce/core/solver.py` | 300 | `ResourceManager` | 完整的资源管理器 修复记录: 1. |
| `core/expert/expert_msfce/core/solver.py` | 813 | `VNFPlacementStrategy` | VNF放置策略基类. |
| `core/expert/expert_msfce/core/solver.py` | 851 | `OptimizedPlacementStrategy` | 高性能放置策略 - 带宽优化版 核心特性： 1. |
| `core/expert/expert_msfce/core/solver.py` | 1087 | `PathEngine` | 路径计算和查询引擎. |
| `core/expert/expert_msfce/core/solver.py` | 1287 | `_TreePathEngine` | 局部路径引擎，供 TreeBuilder 内部使用。 (与顶层 PathEngine 独立，基于拓扑矩阵做 BFS 和距离计算). |
| `core/expert/expert_msfce/core/solver.py` | 1403 | `TreeBuilder` | Tree Construction Algorithm 目标： 1. |
| `core/expert/expert_msfce/core/solver.py` | 1984 | `MSFCE_Solver` | MSFCE专家算法求解器（合并版）. |
## 工具与项目入口
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `scripts/audit_unused_classes.py` | 30 | `ClassDefinition` | 代码类 `ClassDefinition`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_brain_module_skill_deployment.py` | 17 | `FakeBatchPlanner` | 代码类 `FakeBatchPlanner`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_directed_edge_accounting.py` | 32 | `FakePool` | 代码类 `FakePool`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_directed_edge_accounting.py` | 42 | `FakeResourceManager` | 代码类 `FakeResourceManager`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_directed_edge_accounting.py` | 54 | `FakeEnv` | 代码类 `FakeEnv`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_adapter.py` | 17 | `FakeStatefulHRL` | 代码类 `FakeStatefulHRL`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_destination_anchor_retry.py` | 19 | `FakePool` | 代码类 `FakePool`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_destination_anchor_retry.py` | 27 | `FakeResourceManager` | 代码类 `FakeResourceManager`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_destination_anchor_retry.py` | 36 | `FakeShared` | 代码类 `FakeShared`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_order_masks.py` | 23 | `FakePool` | 代码类 `FakePool`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_order_masks.py` | 40 | `FakeResourceManager` | 代码类 `FakeResourceManager`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_order_masks.py` | 54 | `HopController` | 代码类 `HopController`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_planner_safety_guard.py` | 18 | `HopController` | 代码类 `HopController`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_hrl_planner_safety_guard.py` | 23 | `CompletionHigh` | 代码类 `CompletionHigh`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/check_multiagent_orchestration.py` | 25 | `FakeHRLPlanner` | Interface-compatible stand-in; the smoke test does not load a checkpoint. |
| `scripts/check_vnf_container_runtime.py` | 37 | `FakeBackend` | 代码类 `FakeBackend`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/eval_deployment_baselines.py` | 205 | `Metrics` | 代码类 `Metrics`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/eval_migration_baselines.py` | 104 | `PolicyMetrics` | 代码类 `PolicyMetrics`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/generate_sdn_policy_plans.py` | 36 | `ActiveTree` | 代码类 `ActiveTree`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/generate_sdn_policy_plans.py` | 44 | `Candidate` | 代码类 `Candidate`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/generate_sdn_policy_plans.py` | 88 | `DynamicTreePlanner` | 代码类 `DynamicTreePlanner`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/run_sdn_runtime_requests.py` | 55 | `_MigrationPolicyNoop` | Internal control flow for an intentional WQMIX no-migration action. |
| `scripts/run_sdn_runtime_requests.py` | 569 | `RollingSetupEstimator` | Thread-safe rolling setup-time estimate used for deadline admission. |
| `scripts/run_sdn_runtime_requests.py` | 635 | `StageCapacityGate` | Non-blocking capacity gate with auditable occupancy statistics. |
| `scripts/run_sdn_runtime_requests.py` | 671 | `VnfEndpointPool` | Allocate unique reusable VNF receive ports independently per DC. |
| `scripts/run_sdn_runtime_requests.py` | 891 | `RuntimeFifoAckTimeout` | 代码类 `RuntimeFifoAckTimeout`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/run_sdn_runtime_requests.py` | 895 | `ReceiverStartupTimeout` | 代码类 `ReceiverStartupTimeout`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/run_sdn_runtime_requests.py` | 899 | `RyuBatchCommitter` | Collect ready SFCs briefly and commit them with one controller barrier. |
| `scripts/run_sdn_runtime_requests.py` | 1021 | `VnfControlBatcher` | Coalesce cross-request VNF FIFO transactions into one Mininet RTT. |
| `scripts/train_role_marl_offline.py` | 35 | `OfflineSample` | 代码类 `OfflineSample`，用于承载该文件中的相关数据或逻辑。 |
| `scripts/train_sdn_strict_gate.py` | 21 | `StrictGateNet` | 代码类 `StrictGateNet`，用于承载该文件中的相关数据或逻辑。 |
| `sfc_project/__main__.py` | 31 | `WorkflowConfigError` | Raised when the workflow catalog is malformed. |
| `sfc_project/__main__.py` | 36 | `Preset` | 代码类 `Preset`，用于承载该文件中的相关数据或逻辑。 |
## 其他当前代码类
| 文件 | 行 | 类 | 作用简介 |
|---|---:|---|---|
| `core/reward/reward_critic.py` | 25 | `RewardCriticParams` | 奖励函数参数配置 - 修复版. |
| `core/reward/reward_critic.py` | 91 | `RewardCritic` | 修复版 RewardCritic - 平衡各阶段奖励，加强过程约束. |
| `py_split_timeslot/data_generator_class.py` | 14 | `DataGenerator` | 代码类 `DataGenerator`，用于承载该文件中的相关数据或逻辑。 |
