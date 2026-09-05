# 阶段二：生命周期感知迁移奖励

已完成以下代码接线：

- `core/marl/migration_reward.py` 新增 `get_remaining_lifetime(request_id, requests, now)`。
- 迁移奖励改为“剩余寿命内的反事实 SLA 损失减少量 − 迁移开销”。候选可提供 `future_sla_loss_reduction`，否则使用 before/after 字段；候选生成器已提供基于风险与 delay ratio 的默认 proxy。
- 当剩余寿命不超过迁移收敛时间时，迁移动作直接给负奖励；no-migration 奖励为 0。
- `sdn/migration_monitor.py` 增加按 VNF 的迁移历史和窗口计数。窗口内迁移超过 2 次时，奖励乘以 0.5。

本阶段只完成奖励与监控接口，尚未运行训练或实验验证。
