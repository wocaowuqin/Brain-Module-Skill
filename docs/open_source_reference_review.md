# 开源 SFC/NFV 实现对比

本文档记录对当前项目最相关的公开仓库的代码检查结果。目标是定位严格 SLA 的数据面瓶颈，并区分可以直接复用的代码、只能借鉴的机制和不适合接入当前 Mininet/Ryu 测试床的项目。

## 结论

当前严格 SLA 的主要风险在 Python VNF/探针调度造成的尾部排队，而不是 QMIX 前向推理。优先级建议如下：

1. 保留现有 Ryu、OVS、Mininet、Top-K/WQMIX 规划链路。
2. 参考 Containernet 的容器和 CPU 绑定方式，将 VNF 进程固定到独立 CPU 集合。
3. 参考 OpenNetVM 的 C/DPDK packet handler、批量收发和常驻 NF 模型，把当前 Python VNF 转发循环替换为常驻的原生 UDP 转发器；第一版不必迁移整个 DPDK/OVS 数据面。
4. 参考 NFVnice 的负载感知 CPU 权重和逐跳背压，在 VNF 队列超过阈值或预计等待时间超过剩余 SLA 时尽早丢弃/拒绝，避免缓冲区膨胀。
5. 参考 `abulanov/sfc_app` 的 VNF 自注册、服务目录和批量流表思想，但不要直接复制其线性单播假设。

## 仓库清单

| 仓库 | 许可证 | 关键代码 | 对当前项目的价值 | 主要限制 |
|---|---|---|---|---|
| [sdnfv/openNetVM](https://github.com/sdnfv/openNetVM) | 仓库 LICENSE 声明 BSD | `docs/NF_Dev.md`、`onvm/onvm_nflib`、`examples/NFD` | C/DPDK 常驻 NF、服务链转发、批量处理、NF 实例和线程扩展 | 依赖 DPDK/hugepage，不能直接作为 Mininet/OVS 的 Python 替换 |
| [nfvnice/NFVnice_Source](https://github.com/nfvnice/NFVnice_Source) | 仓库 LICENSE 声明 BSD | `onvm/onvm_mgr/onvm_nf.c`、`onvm/shared/onvm_ringbuf.c` | 基于 `comp_cost`、队列 `load` 和 `load*comp_cost` 动态调整 CPU 权重，并支持 NF 背压 | 代码较旧，依赖 OpenNetVM/DPDK；适合移植机制，不适合整仓库嵌入 |
| [NetSys/bess](https://github.com/NetSys/bess) | `COPYING` 声明 BSD | `bessctl/conf/perftest/chain.bess`、`bessctl/conf/port/latency.bess` | 原生 C++ 模块化数据面、批处理、队列和多核 pipeline | 需要重构 OVS/Ryu 数据面，接入成本最高 |
| [containernet/containernet](https://github.com/containernet/containernet) | Mininet 许可证 | `examples/docker_cpuset.py`、`examples/traffic_with_loss_example.py` | Docker VNF、`cpuset_cpus`、链路退化/丢包注入，最容易融入当前 Mininet 实验 | 容器本身不会自动消除 Python 转发开销 |
| [abulanov/sfc_app](https://github.com/abulanov/sfc_app) | MIT | `sfc_app.py`、`json_register.py`、`README.md` | Ryu SFC 流表、VNF 自注册、服务目录、REST 驱动规则安装 | 原型级线性单播链，未覆盖当前组播树和高并发部署 |
| [CN-UPB/NFVdeep](https://github.com/CN-UPB/NFVdeep) | MIT | `nfvdeep/environment/env.py`、`environment/sfc.py` | 在线 SFC 模拟、TTL/最大响应时延约束、回溯释放、PPO/FirstFit 对比 | 仅仿真；README 明确说明 DRL 结果未稳定复现，不能解决真实数据面时延 |
| [GeminiLight/drl-sfcp](https://github.com/GeminiLight/drl-sfcp) | Apache-2.0 | `solver/learning/a3c_gcn_seq2seq`、`base/environment.py` | 可参考 GCN+RL 的状态、请求到达和训练组织方式 | 离线仿真，已经并入 Virne |
| [GeminiLight/virne](https://github.com/GeminiLight/virne) | Apache-2.0 | `docs/source/evaluation/metrics.rst`、`solver`、`datasets/topology` | 提供 SFC/NFV-RA 数据集、RAC/LRC/LAR/AST 等统一指标和多种求解器 | 不负责 Ryu/Mininet 真实转发 |
| [ncl-teu/ncl_sfcsim](https://github.com/ncl-teu/ncl_sfcsim) | Apache-2.0 | `src/net/gripps/cloud/nfv` | 可参考 VNF 调度与生命周期模拟 | Java 仿真器，与当前 Python/Ryu 运行时不兼容 |

## 关键机制摘录

### OpenNetVM：常驻 NF + 批处理

`docs/NF_Dev.md` 定义了固定的 `packet_handler(struct rte_mbuf*, struct onvm_pkt_meta*)`，NF 完成处理后只设置 `NEXT`、`TONF`、`OUT` 或 `DROP` 动作，由管理器完成下一跳转发。高级 ring 接口还支持一次处理一批 packet，并支持多线程 NF 扩展。

当前项目可借鉴的最小子集：

- 一个进程长期绑定一个 VNF stage，而不是每个请求创建 Python 进程；
- 收包、处理、发包采用预分配缓冲区和批量循环；
- 每个 stage 只传递固定长度的二进制头，不在每个包上做 JSON 编解码；
- 把 `DROP` 作为明确的过载动作，并记录原因和队列深度。

### NFVnice：负载感知调度和背压

`onvm/onvm_mgr/onvm_nf.c` 中维护每个 NF 的 `comp_cost`、当前 `load`、服务速率 `svc_rate`，并按同一 CPU 上的总成本计算 `cpu_share`。动态模式使用 `comp_cost * load` 作为分配权重。背压逻辑从下游溢出点向上游标记 `throttle_this_upstream_nf`，避免上游继续把包推入已经拥塞的 stage。

这正对应当前项目的 Q0 尾延迟问题：不要等 UDP/OVS 队列积满后才统计丢包，而应在 VNF agent 入口根据队列深度和剩余 SLA 直接做 admission/drop。

### Containernet：容器化和 CPU 隔离

`examples/docker_cpuset.py` 展示了通过 `addDocker(..., cpuset_cpus="0,1")` 固定容器 CPU。当前项目可以将 Ryu、VNF agent、probe sender/receiver 分到互不重叠的 CPU 集合，并把 VNF 生命周期改为容器启动一次、请求只做绑定。

### Ryu `sfc_app`：注册和流表

`sfc_app.py` 使用 VNF 自注册消息维护服务目录，先安装 catching rule，再在首个数据包暴露入口端口后替换为 steering rule。这个流程可作为当前 Ryu 批量下发的参考，但其 `forward()` 逻辑是线性链，不能直接覆盖当前的分段路径和组播树。

### NFVdeep/Virne：训练与指标

NFVdeep 的环境包含 `ttl`、`bandwidth_demand`、`max_response_latency` 和 VNF 顺序放置，并在放置失败时回溯资源。Virne 将请求接受率定义为：

\[
RAC = \frac{\sum_t |\tilde{\mathcal I}(t)|}{\sum_t |\mathcal I(t)|}.
\]

这些代码适合检查离线数据集和奖励函数，不应被误认为真实 Mininet 数据面实现。

## 建议的接入路线

### 路线 A：最小风险，优先验证

在当前 `sdn/vnf_agent.py` 外增加一个常驻 C/C++ UDP fast path：

1. 启动时为每个 VNF stage 建立固定 socket；
2. 使用 `recvmmsg/sendmmsg` 或等价批量 API；
3. 使用 `SO_REUSEPORT` 将 stage 流量分到固定 worker；
4. 采用 CPU affinity，与 Ryu 和 probe CPU 集合隔离；
5. 保留现有 ready/drain/stop 控制协议，数据包路径不再经过 JSON；
6. 在入口维护有界队列，按 `queue_wait + predicted_processing + residual_path_delay` 判断是否继续接收；
7. 输出每个 stage 的 queue depth、drop reason、packet p99 delay。

### 路线 B：容器化验证

先用 Containernet 的 Docker host 运行这个原生 fast path，复用现在的 Ryu/OVS 规则和 Mininet 拓扑。这样可以单独比较：

`Python agent` vs `C fast path in Docker` vs `C fast path on host`。

### 路线 C：高性能重构

只有当路线 A/B 证明 Python 数据面确实是瓶颈后，再考虑 OpenNetVM 或 BESS。它们会改变网卡、内存、队列和流表接入方式，适合独立性能章节，不适合作为当前项目的快速修补。

## 建议的验证实验

保持同一拓扑、同一请求文件、同一随机种子和同一 Ryu 计划，至少比较 seed 7304：

| 版本 | 目的 |
|---|---|
| 当前 Python VNF agent | 基线 |
| Python agent + CPU 隔离/有界队列 | 区分调度串扰和队列膨胀 |
| C fast path + 现有 OVS/Ryu | 验证数据面尾延迟瓶颈 |
| C fast path + NFVnice 风格背压 | 验证过载时是否减少 Q0 的严格 SLA失败 |

必须同时报告：严格 SLA、每个 QoS 类别的 p99/p99.9 延迟、超过时延阈值的包比例、实际流量丢包率、deployment queue wait、VNF queue depth 和 drop reason。只看平均时延不能证明严格 SLA 改善。

## 许可证注意事项

`sfc_app`、NFVdeep、Virne 和 drl-sfcp 的仓库许可证分别为 MIT、MIT、Apache-2.0 和 Apache-2.0。OpenNetVM/NFVnice 的仓库 LICENSE 明确写 BSD，但 GitHub API 标记为 `NOASSERTION`；BESS 使用仓库中的 `COPYING`，Containernet 继承 Mininet 许可证。若复制代码进入论文项目，应保留原始版权和许可证文件，并优先采用“参考机制、重新实现接口”的方式。
