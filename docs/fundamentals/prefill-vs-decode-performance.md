# Prefill 与 Decode 为什么性能不同：从计算形状到执行开销

Prefill 和 Decode 使用同一份权重，设备收到的工作形状却差很多。Prefill 一次可以计算许多新 Token；普通 Decode 中，每条请求一轮通常只新增一个 Token，还需要访问此前积累的历史状态。

以一层矩阵乘为例，2048 个 Prompt Token 能共同复用一份权重，单条 Decode 请求当前却只有一行输入。进入 MoE 后，这些输入还会被路由到不同专家，单个专家可能只分到很少几行。要判断耗时来自计算、搬运还是反复提交任务，可以先把这两次形状变化算清，再追踪 DeepSeek-V4-Flash 的滑窗、压缩历史和实际执行开销。

刚接触推理性能时，可以先读前两节，理解一轮有多少新 Token、这些行如何复用权重。已经在分析日志和时间线，则可以从后面的 Attention 历史访问与 Profile 对照读起。术语可随时查[推理基础设施名词表](ai-infra-working-glossary.md)。

源码主要固定在 SGLang commit <code>b86a30afbae389eec8763f4bcf38f14904d2d00c</code>，具体案例采用 DeepSeek-V4-Flash-0731，核对日期为 2026-09-21。

为单独观察 Shape 对算术强度的影响，部分 GEMM 例子把数据统一按 BF16 字节数估算。这是思想实验：官方 DeepSeek-V4-Flash checkpoint 含 FP8 / FP4 等量化配置，Ascend 部署也可能走 W8A8 等路径。dtype 会改变系数；大 M 能让更多输入复用权重，小 M 更难分摊权重读取和 Launch 成本，这层关系仍然成立。

## 一、新 Token 数怎样改变计算形状

假设输入有 2048 个 Prompt Token，目标生成 512 个 Output Token，请求先后经过以下阶段：

~~~mermaid
flowchart TD
    A["Prompt · 2048 Tokens"]
    B["Prefill · 一次处理很多新 Token"]
    C["First Token"]
    D["Decode Step 1"]
    E["Decode Step 2"]
    F["..."]
    G["Decode Step 511"]

    A --> B --> C --> D --> E --> F --> G
~~~

Prefill 能一起计算已经给定的 Prompt Token。普通自回归 Decode 则需要先得到 Token(t) 的 Sampling 结果，才知道下一轮该输入什么。对单条请求，第 1、2、3 轮通常各有一枚新输入，后续仍依次推进。

这就是自回归依赖（Autoregressive Dependency）。把不同请求组成 Batch，能增加当前轮的输入行数，但每条请求未来的 Token 仍要等前面的结果。

这个区别也体现在 SGLang 的 Runtime 中。固定版本 <code>forward_batch_info.py</code> 把 <code>ForwardMode.EXTEND</code> 注释为通常意义上的 Prefill，把 <code>ForwardMode.DECODE</code> 解释为 Decode one token；<code>ForwardMode.MIXED</code> 则用于 Chunked Prefill 等包含 Extend / Decode 的混合批次。

Scheduler 准备 DP Attention batch 时，对两种阶段分别计数：

~~~python
if local_batch.forward_mode.is_decode():
    num_tokens = local_batch.batch_size()
else:
    num_tokens = local_batch.extend_num_tokens
~~~

按这里的普通 Decode 路径，64 条活跃请求约提供 64 行新 Token；Prefill / Extend 一轮则可能有 512、2048、8192 行，或者更多。矩阵乘的 M 要看当前新输入，不能直接拿总上下文长度代入。落到某个 Rank 后，分片和 Graph 填充还会继续改变执行尺寸。

暂且忽略分片与填充，Prefill 的 M 可以达到数千，普通 Decode 的 M 则近似等于当前活跃请求数。

同一层权重面对的输入行数已经不同。接下来权重复用以及专家负载的差别，都是从这里开始的。

Ascend 910C 的 DeepSeek-V4 Attention Backend 也按阶段准备 metadata。Extend 根据各 Request 的 <code>extend_seq_lens</code> 构造 Query prefix sum，普通 Decode 则构造 `actual_seq_lengths_q_pa = [0,1,2,...,B]`，将每条请求表示成一个 Query row。

因此，Scheduler 的 Token 计数、ForwardBatch 输入和 Attention 接收的 Query 形状，是同一差异在不同层的体现。

## 二、Dense GEMM：M 与权重复用

先看模型中常见的矩阵乘：

~~~text
Y = XW

X: [M,K]
W: [K,N]
Y: [M,N]
~~~

M 是这次输入的行数，也就是多少行 Token 表示一起使用权重；K 是输入特征维度，N 是输出特征维度。DeepSeek-V4-Flash 的几个 Attention 投影可以写成下面的形状。

| 投影 | 输入 | 权重的逻辑全局形状 |
| --- | --- | --- |
| `wq_a` | [M,4096] | [4096,1024] |
| `wkv` | [M,4096] | [4096,512] |
| `wq_b` | [M,1024] | [1024,32768] |

<code>wq_b</code> 实际每个 Rank 的 N 还受 <code>attn_tp_size</code> 影响。M 主要随当前处理阶段和负载变化，K/N 则需要结合模型结构与并行布局看。

现在只改变 M，并假设 <code>wq_a</code> 的 X、W、Y 全部使用 BF16，每元素 2 Byte。矩阵乘计算量约为 `FLOPs ≈ 2MKN`。

若只计输入、权重、输出各一次读写，数据量约为 `Bytes ≈ 2 × (MK + KN + MN)`，可得：

~~~text
AI
≈
2MKN
/
[2(MK + KN + MN)]
~~~

将 K=4096、N=1024 代入：

| M | 理想 Arithmetic Intensity |
| ---: | ---: |
| 1 | ≈ 1.0 FLOP/B |
| 8 | ≈ 7.9 FLOP/B |
| 64 | ≈ 59 FLOP/B |
| 512 | ≈ 315 FLOP/B |
| 2048 | ≈ 585 FLOP/B |

Arithmetic Intensity（算术强度，简称 AI）衡量每搬运一字节完成多少计算，单位为 FLOP/B。这张表是理想估算，未计入 Cache、Tiling、通信、量化 Scale 等因素，也不是 Ascend 910C 的实测值。M=1 时，整份权重只服务一行；M=2048 时，同一份权重可以服务更多行，计算量相对搬运量就更大。

当 M 远小于 K/N、权重读取占主要部分时，上式还可以近似为 `AI ≈ 2M / b_w`。

<code>b_w</code> 表示每个权重元素的字节数。BF16 下近似 AI≈M，INT8 下则近似 AI≈2M。

小 M 时，减少 Weight bytes 有机会缓解带宽压力，这也是量化适合 Decode 的一个原因。不过 W8A8 还会引入 Dynamic Quant、Scale、Tiling、Launch、Epilogue 等成本。最终 TPOT 能否下降，需要看节省的权重读取能否覆盖这些额外工作。

Continuous Batching 将多条请求当前的 Token 合并，增大 M，因而有机会提高 Dense GEMM 的算术强度。可以再用 Roofline 的转折点（Ridge Point）判断限制来自哪边，它等于设备有效计算能力与有效带宽之比：

~~~text
R_eff
=
effective FLOP/s
/
effective Byte/s
~~~

算术强度低于这个比值时，更容易先碰到带宽上限；高于它时，计算能力更可能先形成限制。在权重流量占主导的简化模型中，可进一步估算对应的 M：

~~~text
M_crossover
≈
R_eff × b_w / 2
~~~

实际转折点还受到 CANN Tiling、Weight Layout、Cache、Quantization、Kernel Occupancy、TP / DP 和通信影响，需要在目标配置上做 batch sweep 观察。

## 三、MoE：总 Token 怎样分散到 Expert

Dense Linear 可以共同处理当前一轮的输入行。MoE 路由后，每个 Expert GEMM 的 `M_e` 则是 Expert e 当前实际收到的 Token 行数。

计算平均 `M_e` 时，Token 总数和专家数量必须取自同一个 EP 通信域。

DeepSeek-V4-Flash 的公开配置为：

~~~text
256 routed experts
每个 Token 激活 6 个 routed experts
moe_intermediate_size = 2048
hidden_size = 4096
~~~

单个 routed expert 的 W13 逻辑形状近似为：

~~~text
[M_e,4096]
×
[4096,4096]
~~~

输出维度 4096 来自 Gate、Up 各 2048 的两个分支，之后 W2 的形状近似为：

~~~text
[M_e,2048]
×
[2048,4096]
~~~

每个 Token 激活 6 个 routed experts，因此 M 个 Token 约产生 6M 个 token-expert assignments。

先假定 Router 完全均匀，用平均值建立尺寸直觉；真实请求的路由分布还会有偏斜。

把 Prefill 的 M=2048 与 64 条活跃请求的普通 Decode 并排计算：

| 阶段 | M | Token–Expert assignments | 平均 M_e |
| --- | ---: | ---: | ---: |
| Prefill | 2048 | 2048×6=12288 | 12288/256=48 |
| Decode | 64 | 64×6=384 | 384/256=1.5 |

平均数可以是 1.5，实际单个专家收到的行数仍是整数：有些没有 Token，有些会收到多个。

仍以 BF16 估算 W13，得到下表。M_e=1.5 只是将平均负载代入公式比较量级，不代表 Kernel 会执行半行。

| Expert M_e | AI |
| ---: | ---: |
| 1 | ≈ 1.0 FLOP/B |
| 1.5 | ≈ 1.5 FLOP/B |
| 6 | ≈ 6.0 FLOP/B |
| 48 | ≈ 46.9 FLOP/B |

因此，从 Prefill 变为 Decode，不仅 Dense 的 M 从 2048 变为 64，平均 Expert M_e 还可能从约 48 降到约 1.5。

实际路由会进一步产生不均衡，例如：

~~~text
Expert 0: 0
Expert 1: 1
Expert 2: 7
Expert 3: 0
Expert 4: 13
...
~~~

有的专家没有工作，有的只做很小的 GEMM。GroupedMatmul 可以统一组织多组计算，却仍要处理各组过小或不均衡的尺寸。跨 Rank 时又增加 Dispatch / Combine 以及等待，所以即使总 Token 数相同，专家分布不同也会改变耗时。

分析 DeepSeek-V4 Decode 时，除了 global batch size，还应记录 Expert Token Histogram、mean / median / p90 / max M_e 和 empty expert ratio。EPLB 或 MoE runtime 的调整会改变这部分执行条件，最终可能影响 Decode 性能。

## 四、长上下文下的滑窗与压缩历史

入门时通常用普通 MHA 描述 Decode 的历史读取：

~~~text
Q_new: [1,d]

读取:
K_cache[0:L]
V_cache[0:L]
~~~

这个模型中，上下文 L 翻倍，历史 KV 读取量也大致翻倍。DeepSeek-V4 使用滑窗和压缩历史，需要分别计算各条路径，才能判断上下文增长带来什么成本。

DeepSeek-V4-Flash 的公开配置包含：

~~~text
sliding_window = 128
index_head_dim = 128
index_topk = 512

compress_ratios =
[0, 0, 4, 128, 4, 128, ...]
~~~

对应的 SGLang Ascend DSV4 路径会涉及 SWA、C4 compressed history、C128 compressed history、C4 Lightning Indexer，以及 Compressor state。

DeepSeek-V4 的 qk_nope_head_dim=448、qk_rope_head_dim=64，合起来一行宽度为 512。本文关注 Ascend 910C 的非 arch35 PA_ND 路径，其 KV row 使用 BF16，因此单行 logical payload 约为 `512 × 2 = 1024 Bytes`，即 1 KiB。

较新的 arch35 分支采用 packed FP8 KV 布局，后续字节数不能直接套过去。另外，这里估算的是 logical payload，不等于实际 HBM transaction；Cache、Prefetch、Page Layout、Tiling、量化格式及多 Head 复用都会影响真实搬运。

SWA 把历史窗口限制在约 128 rows，按 sliding_window=128 粗估，logical payload 约 128 KiB。这部分不会随上下文从 32K 增到 128K、1M 而无限线性增加。

C4 的 compress_ratio=4，长度 L 对应约 ceil(L/4) 个压缩位置。不过 NPU DSV4 backend 最终用 <code>c4_topk_indices</code> 选择历史，公开配置 index_topk=512，因此长上下文下最终 C4 Attention 消费的 compressed rows 上限约为 512。

只计最终 Attention 读取的逻辑 KV 数据，可估为：

~~~text
SWA:
128 × 1 KiB

+

C4 top-k:
512 × 1 KiB

≈
640 KiB / query / C4 layer
~~~

固定的候选上限只约束最终 Attention。候选先要由 Indexer 从压缩历史中找出来，上下文增长仍会增加这部分工作。

C4 Lightning Indexer 面对约 ceil(L/4) 个历史 key，公开配置 index_head_dim=128。在本文的 910C 路径中，Indexer storage 用 INT8 key 和 FP16 scale，每行原始 payload 约为 `128 Bytes INT8 key + 2 Bytes scale = 130 Bytes`。

整个逻辑 key 数据量约为 `ceil(L/4) × 130 Bytes`：

| Context L | C4 Index rows | Logical key payload |
| ---: | ---: | ---: |
| 8K | 2048 | ≈ 0.25 MiB |
| 32K | 8192 | ≈ 1.0 MiB |
| 128K | 32768 | ≈ 4.1 MiB |
| 1M | 262144 | ≈ 32.5 MiB |

这张表同样没有给出实际 HBM 流量或耗时，二者还依赖缓存和算子访问方式。它说明，最终 Sparse Attention 只用 Top-K 候选，选候选之前的 Indexer 历史规模却仍在增长。

因此看长上下文 Decode 的 Profile 时，应分别统计 Lightning Indexer 和最终 Sparse Attention。

C128 的处理又不同。compress_ratio=128，Ascend backend 设置 <code>cmp_sparse_indices=None</code>，读取完整 C128 compressed history。长度 L 对应约 ceil(L/128) 行。

加上 SWA 后，logical payload 近似为：

~~~text
128 KiB SWA
+
ceil(L/128) × 1 KiB
~~~

| Context | C128 Attention logical payload |
| ---: | ---: |
| 8K | ≈ 0.19 MiB |
| 32K | ≈ 0.38 MiB |
| 128K | ≈ 1.13 MiB |
| 1M | ≈ 8.13 MiB |

它仍随上下文增长，但每增加 128 个原始 Token 才增加约一行压缩历史。

此外，DSV4 Compressor 单独维护 FP32 state cache。源码中非 online state 的 last_dim 为 `2 × (1 + overlap) × head_dim`。

C4 使用 overlap=True，一行约 2048 个 FP32，即约 8 KiB；C128 使用 overlap=False，一行约 1024 个 FP32，即约 4 KiB。

每步实际访问多少 state rows，还要看 Compressor Plan、Ring State、当前位置和融合算子，可用该步 metadata 与 trace 核对。

将这些工作放到一起，可得到下面的成本分解。C4、C128 属于不同类型的层，需要分别统计；这里并不表示每一层依次执行所有分支，也没有展开通信与计算的重叠。

~~~text
T_decode_attention(L)

≈

T_SWA(<=128)
+
T_C4_Attention(top512)
+
T_C4_Indexer(L/4)
+
T_C128_Attention(L/128)
+
T_Compressor_State
+
metadata / page table / cache effects
~~~

历史状态的成本仍会随上下文增加，但具体由哪个子模块增长、增长多快，需要沿这些路径分别判断，不能继续套用标准 MHA 的比例。

## 五、跨 Token 依赖与重复开销

除了历史状态，Decode 还受跨 Token 依赖约束：下一轮输入需要等待当前轮确认。

第 t 个输出要经过 Forward、Logits 和 Sampling 才能得到 Token_t。接着将它送入 Embedding 与下一轮 Forward，才能采样 Token_t+1。这条依赖链决定了单请求未来的输出无法预先全部作为输入。

如果首个输出在 Prefill 末尾采样得到，生成 1000 个输出 Token 通常还要约 999 轮 Decode，停止条件等也会影响最终轮数。这些轮次前后相依，未来的 1000 个 Token 还不能像 Prompt 一样直接组成输入。

每轮除了 GEMM，还可能经过 Scheduler、ForwardBatch preparation、Graph replay / launch、Attention metadata、Collective、DeepEP、Sampling 和 State publication。每项即使只有几微秒或几十微秒，反复执行几百到几千轮，也可能成为 TPOT 的组成部分。

Prefill 同样需要 Launch、Runtime dispatch、metadata 和 collective，但一轮 2048-token Prefill 能把相关开销分摊给更多 Token。

因此还要比较每次 launch 能完成多少有效工作，不能只数启动了多少个 Kernel。

Graph Replay 减少重复提交，Kernel Fusion 减少算子启动及中间数据搬运，Host Overlap、Multi-Stream 尝试重叠独立的准备、计算和通信。它们受依赖和资源竞争约束，是否改善 TPOT，要看整轮关键路径有没有缩短。

Chunked Prefill 也在调节这笔开销。巨大的 Prefill 可能充分利用设备，却让正在流式生成的请求等待过久；拆成较短批次，能给 Decode 留出调度机会，同时需要权衡额外轮次。

## 六、常见优化分别作用在哪里

将前面的关系连起来，可以得到：

~~~mermaid
flowchart TD
    A["Same DeepSeek-V4 Weights"]

    A --> P["Prefill"]
    A --> D["Decode"]

    P --> P1["很多 Query Rows"]
    P1 --> P2["Dense M 大"]
    P1 --> P3["Expert M_e 相对更大"]
    P2 --> P4["Weight reuse ↑"]
    P3 --> P4
    P4 --> P5["更容易提高计算利用率"]

    D --> D1["每 Request 1 Query"]
    D1 --> D2["Dense M 小"]
    D1 --> D3["Expert M_e 更碎"]
    D1 --> D4["读取 SWA / C4 / C128 历史"]
    D4 --> D5["Indexer / compressed history 随 Context 增长"]
    D2 --> D6["固定 Launch 成本难摊薄"]
    D3 --> D6
    D --> D7["跨 Token 串行依赖"]

    P5 --> TTFT["TTFT / Input Throughput"]
    D5 --> TPOT["TPOT / Output Throughput"]
    D6 --> TPOT
    D7 --> TPOT
~~~

图中各环节对应的优化并不相同。KV Cache 保存已计算的历史状态；Paged Cache 管理不同长度请求的物理空间；Radix Cache 进一步复用请求间命中的前缀。它们主要改变状态的存储与复用。

Continuous Batching 合并多条请求当前的 Decode Token，增加 M；Chunked Prefill 则控制单轮 Prefill 工作量，让已有 Decode 请求获得调度机会。两者都影响组批，但优化方向和代价不同。

Quantization 减少权重等数据的字节数，收益还取决于 M、M_e、Tiling 和量化本身的成本。Speculative Decoding 用一次 Target 验证尝试推进多个候选，同时增加 Draft、Verify 与状态整理，需要按整个周期比较。

PD Disaggregation 让 Prefill、Decode 分别使用资源池和组批、并行参数，也带来 KV 传输及请求交接。该选哪条路径，要先辨明当前时间花在权重复用、历史访问、调度等待还是重复执行开销上。

## 七、用 Profile 判断实际瓶颈

上文推导了形状及可能的限制，没有给出目标配置的实测利用率。判断 Ascend 910C 上的实际瓶颈，需要把硬件指标、算子耗时与整轮执行对齐。

Ascend Profiling 可先查看目标产品、工具版本支持的流水指标：`mac_ratio` 是 Cube / Matrix 指令周期占比，`vec_ratio` 是 Vector 指令周期占比，`mte2_ratio` 是 MTE2 数据搬运相关周期占比。它们描述算子内部的工作，还需结合总耗时阅读。

官方定义的 `memory_bound` 比值如下：

~~~text
memory_bound
=
mte2_ratio
/
max(mac_ratio, vec_ratio)
~~~

在支持这一口径的采集中，<code>memory_bound &gt; 1</code> 表示相对主要计算流水，数据搬运形成了压力。产品和工具版本决定字段及流水名称，未提供时应记 N/A，不能按零解读。判断整轮 Decode 的关键路径，还需要其他证据；字段范围可查[官方指标与产品支持说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0069.html)。

可以按下面的对照验证形状推导，但本文没有执行这组 sweep。先固定 checkpoint、quantization、TP / DP / EP、CANN、Attention backend 和 MoE 通信配置，再对齐请求长度、缓存命中、请求数及统计区间。

Prefill 可将新输入行数 M 设为 128、512、1024、2048、4096、8192，比较各尺寸。

分别观察 Dense GEMM、GroupedMatmul、Attention、Lightning Indexer、Compressor、Collective 的 latency、mac_ratio、vec_ratio、mte2_ratio 和 memory_bound。

Decode 则先固定 Context，例如 8K、32K、128K，再将活跃请求数 B 设为 1、2、4、8、16、32、64、128。

记录 Dense M、Expert M_e histogram、Dense GEMM latency、GroupedMatmul latency、Attention latency、Indexer latency、DeepEP / HCCL、Host Gap、TPOT 及 output tok/s，才能对照算子变化与端到端收益。

这里至少要分清三种 crossover：Dense GEMM 的转折、以 Expert M_e distribution 为横轴的 MoE 转折，以及端到端 Decode 的转折。

Dense GEMM 即使已经更偏向 Compute-heavy，Attention、C4 Indexer、DeepEP 或 Host Gap 仍可能占据关键路径，限制整轮速度。

Batch 增加后，Dense 和 Expert GEMM 可能更充分地复用权重；Context 增长又会改变历史访问和索引工作。将 Batch × Context 放在一起观察，并保留整轮指标，才有机会找到目标配置真正有用的性能转折点。

### 源码与资料入口

SGLang Runtime：

- [ForwardMode / ForwardBatch](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/model_executor/forward_batch_info.py)
- [DP Attention batch token accounting](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/managers/scheduler_components/dp_attn.py)

DeepSeek-V4：

- [SGLang DeepSeek-V4 model](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/models/deepseek_v4.py)
- [DeepSeek-V4 MoE implementation base](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/models/deepseek_v2.py)
- [DeepSeek-V4-Flash-0731 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json)

Ascend 910C DSV4：

- [Ascend DSV4 Attention Backend](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)
- [Ascend DSV4 KV Pool](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)
- [DSV4 Compressor state](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/mem_cache/deepseek_v4_compress_state.py)
- [DSV4 Compressor](https://github.com/sgl-project/sglang/blob/b86a30afbae389eec8763f4bcf38f14904d2d00c/python/sglang/srt/layers/attention/dsv4/compressor.py)

Profiling：

- [Ascend AI Core utilization / memory_bound 指标](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/profiling/atlasprofiling_16_0069.html)
