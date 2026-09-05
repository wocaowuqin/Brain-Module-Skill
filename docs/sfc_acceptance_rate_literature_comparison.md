# SFC 严格接受率文献对照

## 结论先行

当前 seed=7304 高负载实验的总严格接受率约为 50%，不能直接与文献中的 85%--100% 结果横向比较。当前项目的判定同时要求：

1. 请求完成部署；
2. 流量经过全部 VNF 阶段；
3. 所有组播目的端均收到流量；
4. 发送完成率满足要求；
5. 每个目的端同时满足时延、抖动和丢包 SLA。

多数 SFC 仿真论文只把“资源和路径映射可行，且端到端 RTT/时延低于阈值”计为 accepted，不运行真实 OVS/Ryu 数据面，也通常不把启动失败和实际丢包纳入同一个请求级指标。因此当前结果反映的是更严格的端到端工程指标，而不是同一口径下的算法接受率。

## 代表性文献

### 1. DRL-FJM：联合 VNF 放置与路由

Wu, Y., Hu, H., and Zhang, Z. “DRL-Based Fast Joint Mapping Approach for SFC Deployment.” *Electronics*, 2025, 14(12), 2408. DOI: [10.3390/electronics14122408](https://doi.org/10.3390/electronics14122408).

实验条件：19 个节点、42 条链路、1 Gbps 链路、0.4 ms 传播时延、0.2 ms 单包传输时延；VNF 数量为 2--5；请求按泊松过程到达，平均到达率 `lambda=2`；流量需求为 40--50 Mbps；测试 2000 条请求。

结果：作者报告 DRL-FJM 的 SFC 映射成功率接近 100%，CRADR-Greedy 约 89%，CRADR-Random 约 85%。论文同时指出，随着平均到达率和流持续时间增加，接受率下降，这是 Little 定律导致活动 SFC 数量增加、资源耗尽的直接结果。

对当前项目的启示：

- 论文的核心不是单独优化 VNF 放置，而是先生成多个端到端“放置+路径”候选，再由 DRL 选择联合方案。
- 当前项目若只有一条固定 HRL 计划，遇到链路冲突时只能拒绝，无法复现 DRL-FJM 的候选切换能力。
- 应至少加入 Top-K 候选计划和原子提交失败后的次优候选回退。

### 2. Online SFC Deployment for Live-Streaming：拥塞条件下的接受率

Moreno, J. F. C. et al. “Online Service Function Chain Deployment for Live-Streaming in Virtualized Content Delivery Networks: A Deep Reinforcement Learning Approach.” *Future Internet*, 2021, 13(11), 278. DOI: [10.3390/fi13110278](https://doi.org/10.3390/fi13110278).

该文明确将 acceptance ratio 定义为 RTT 低于最大阈值的请求比例，并特别区分 RTT 与数据流总传播时延。实验使用真实视频访问日志，15 秒时间步，1 天测试轨迹；网络包含 41 个内容提供节点、16 个 hosting 节点和 4 个客户端集群，每个客户端集群每分钟约 `1e4` 个直播请求。

作者刻意设置了过载网络条件，并明确说明 AR 不可能达到 1；其算法在测试后半段维持了高于 0.5 的接受率。论文的关键改进是：

- 使用与 VNF 利用率相关的处理时间；
- 将 VNF 实例化时间计入请求启动时延；
- 使用 dense reward 和负奖励；
- 不在每次错误放置后简单回退，而是让智能体从错误动作中学习。

对当前项目的启示：

- 在真实 Mininet 中同时计算启动时延、实际丢包和所有接收端 SLA，比该文的 RTT 接受率更严格；
- “部署后才发现过载”会导致请求级 SLA 失败，应把资源可行性、生命周期和候选路径冲突放进动作掩码；
- 不能只用失败即拒绝的稀疏奖励训练 HRL/QMIX。

### 3. NFVdeep：自适应在线 SFC 部署

Xiao, Y. et al. “NFVdeep: Adaptive Online Service Function Chain Deployment with Deep Reinforcement Learning.” *Proceedings of the 10th ACM Multimedia Systems Conference*, 2019. DOI: [10.1145/3326285.3329056](https://doi.org/10.1145/3326285.3329056).

该工作以在线部署为目标，重点优化接受率和端到端响应时延。后续论文对 NFVdeep 的分析指出，它采用回溯机制：资源不足或时延超限时忽略请求，不给智能体奖励。这种稀疏反馈会降低策略发现长期可行部署的能力。

### 4. Deterministic latency/jitter-aware SFC

Yu, H. et al. “Deterministic Latency/Jitter-Aware Service Function Chaining over Beyond 5G Edge Fabric.” *IEEE Transactions on Network and Service Management*, 2022. IEEE document: [9714258](https://ieeexplore.ieee.org/document/9714258/).

该文同时考虑 SFC 生命周期、确定性时延和抖动，并提出初始部署算法 Det-SFCD 以及流量变化下的调整算法 Det-SFCA。论文报告调整机制在流量负载变化时改善接受率、收益和时延变化。

对当前项目的启示是：只在到达时做一次固定部署不能充分利用请求生命周期，应在运行中监控热点链路，并允许受门控的局部树重路由或 VNF 迁移。

### 5. Dynamic SFC Deployment and Readjustment

Liu, J. et al. “On Dynamic Service Function Chain Deployment and Readjustment.” *IEEE Transactions on Network and Service Management*, 2017. IEEE document: [7938396](https://ieeexplore.ieee.org/document/7938396/).

该文联合优化新请求部署和已部署 SFC 的重调整，使用 ILP 和 column generation 降低求解复杂度，同时权衡资源消耗和操作开销。其重点不是单次路径最短，而是通过 readjustment 在动态负载下提高服务收益和接受率。

### 6. Virne：统一 NFV-RA 基准与动作掩码证据

Wang, T. et al. “Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV.” arXiv:2507.19234, 2025. [论文](https://arxiv.org/abs/2507.19234)；[代码](https://github.com/GeminiLight/Virne)。

Virne 建议同时报告 request acceptance rate、long-term revenue-to-cost、long-term average revenue 和 average solving time，并增加 solvability、generalization 和 scalability 三类实用性评估。

其消融实验给出了对当前项目最直接的证据：

- PPO-ATT+ 在同时使用状态特征和拓扑特征时，接受率达到 `0.712`；
- 与相同网络但不使用动作掩码的配置相比，PPO-MLP+ 和 PPO-DualGAT+ 使用 action masking 后，接受率最高提升 `0.053`；
- 动作掩码用于屏蔽资源不可行动作，而不是等部署失败后再回退。

## 与当前实验的正确对照

| 项目 | 当前 Mininet/Ryu 实验 | 文献常见仿真 |
|---|---|---|
| 拓扑 | 28 节点、90 Mbps 物理链路；也测试更大拓扑 | 19 节点、1 Gbps 或抽象云/边缘网络 |
| 请求 | 组播，一条请求包含多个目的端和 3 个 VNF 阶段 | 多为单路径 SFC 或抽象请求 |
| 到达负载 | 高到达率，多个请求同时竞争链路和控制面 | `lambda=2` 等较低或按时隙仿真 |
| 接受判定 | 部署、全 VNF 遍历、所有目的端、发送完成率、时延、抖动、实际丢包全部满足 | 通常是映射可行且 RTT/时延达标 |
| 执行平台 | Mininet + OVS + Ryu + UDP 探针 | SFCSim/NetworkX 等离线仿真 |
| 失败来源 | 资源冲突、控制面排队、VNF ready、流表安装、实际丢包 | 主要是资源/路径约束不可行 |

因此当前 30 条高负载实验的 `15/30=50%` 应拆成两个指标：

- **总严格接受率**：`15/30=50%`，被提前拒绝的请求也计为失败；
- **已启动请求严格 SLA 率**：`15/15=100%`，说明带宽接纳控制保护了已经部署的请求。

两者都必须报告，否则只报 50% 会掩盖“启动后数据面已经稳定”的事实；只报 100% 又会掩盖接纳策略过于保守的问题。

## 应优先移植的三项机制

1. **Top-K 联合候选计划**：每条请求生成多组 VNF 放置+分段路径+组播树，按当前残余资源和链路冲突动态选择，而不是只有一条 HRL 计划。
2. **分层动作掩码**：高层屏蔽 CPU/内存/生命周期不可行的候选；低层屏蔽下一跳带宽不足、会造成环路或超过延迟预算的动作。Virne 的消融结果支持该方向。
3. **运行期重调整**：参考 Det-SFCA 和 Liu 等人的 readjustment，在剩余生命周期足够、预测收益为正且新旧路径带宽都可行时，做局部树重路由。

## 实验口径建议

下一轮应同时报告：

- 总请求严格接受率；
- 已启动请求严格 SLA 率；
- admission rejection rate；
- setup latency P50/P95；
- 实际丢包率和每接收端 SLA 达标率；
- 到达率和平均生命周期扫描曲线。

这样才能判断问题到底是“算法找不到可行计划”，还是“控制面/数据面执行太慢”。
