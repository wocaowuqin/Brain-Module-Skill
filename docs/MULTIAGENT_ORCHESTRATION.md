# 分层多智能体编排

## 目标与边界

该模块把项目现有能力组织为一个中央大脑和三个专业角色。它不复制 HRL、迁移、
重路由或资源账本实现。

```text
事件 -> BrainAgent -> DeploymentAgent -> hrl_sft_mapping
                  |-> MigrationAgent  -> plan_vnf_migration
                  `-> RerouteAgent    -> plan_tree_reroute
                                      -> 安全仲裁与既有执行器
```

- 初始映射只能由项目现有双层 HRL 完成。
- Skill 只读快照并生成提案，不直接修改资源。
- 专业 Agent 只能调用注册给自己角色的 Skill。
- 迁移和重路由在执行前继续使用状态指纹和现有验证逻辑。
- 尚未实现跨迁移与重路由的联合原子事务，因此同一 SFT 每轮只执行一个安全动作。
- 未配置真实部署执行器时，HRL 结果明确标为 `PLANNED_ONLY`。

## 文件所有权

- `core/marl/orchestration/protocol.py`：事件、命令、提案和结果消息。
- `core/marl/orchestration/skills.py`：现有能力的 Skill 适配器和权限注册表。
- `core/marl/orchestration/agents.py`：中央大脑和专业 Agent。
- `core/marl/orchestration/orchestrator.py`：观察、派发、仲裁和执行入口。
- `configs/multiagent_orchestration.yaml`：唯一配置文件。
- `scripts/check_multiagent_orchestration.py`：唯一冒烟验证入口。

## 当前策略与后续训练

第一版 BrainAgent 使用事件和热点规则，以先验证通信及执行语义。它与未来学习策略
使用相同的 `SystemObservation -> BrainCommand` 接口。下一阶段可将其替换为 PPO
或 DQN，而无需修改 HRL、专业 Skill 或安全执行器。

迁移和重路由 Agent 当前调用现有可行候选生成器。训练版应先用这些候选生成专家
标签做 BC，再用共同的 SLA、热点、迁移成本和执行失败奖励进行 CTDE 微调。硬约束
仍由 Action Mask、状态指纹和原子执行器保证。

## 验证

```powershell
python -m sfc_project run multiagent-orchestration-check --execute
```

该检查使用内存拓扑和接口兼容 HRL stub，不加载模型、不写实验输出，也不启动
Mininet/Ryu。真实 HRL 在线映射和真实数据面验证属于后续独立实验。
