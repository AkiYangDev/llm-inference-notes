# Prefill 与 Decode 为什么性能不同：从计算形状到执行开销

Prefill 和 Decode 使用同一份模型权重，却把不同形状的工作交给设备。Prefill 一次处理较多新 Token；普通 Decode 中，每条请求一轮通常只提供一个新 Token，同时还要读取已有历史。

例如同样一层矩阵乘，2048 个 Prompt Token 可以一起复用权重，单条 Decode 请求却只能先算当前这一个 Token。进入 MoE 后，这些 Token 又会分散给不同专家，每个专家实际收到的行数可能更少。理解这两次形状变化，才能判断时间更可能花在计算、数据搬运还是反复提交任务上。下面从矩阵形状推到 DeepSeek-V4-Flash 的滑窗与压缩历史，再讨论怎样用 Profile 核对。

本文主要固定到 SGLang commit <code>b86a30afbae389eec8763f4bcf38f14904d2d00c</code>，模型以 DeepSeek-V4-Flash-0731 为具体案例，核对日期为 2026-09-21。

为了把 Shape 对算术强度的影响单独讲清楚，部分 GEMM 示例会统一使用 BF16 bytes 做思想实验。官方 DeepSeek-V4-Flash checkpoint 实际包含 FP8 / FP4 等量化配置，不同 Ascend 部署也可能采用 W8A8 等路径；dtype 会改变数值系数，但不会改变“大 M 更容易复用 Weight、小 M 更难摊薄 Weight / Launch 成本”的基本结论。

## 一、新 Token 数怎样改变计算形状

假设用户输入 2048 个 Prompt Token，然后让模型生成 512 个 Output Token。

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

Prefill 可以把大量 Prompt Token 一起送进模型。普通自回归 Decode 按顺序推进：Token(t+1) 依赖 Token(t) 的 Sampling 结果。对单个 Request 来说，Decode 更接近：

~~~text
Step 1: 1 个新 Token
Step 2: 1 个新 Token
Step 3: 1 个新 Token
...
~~~

这种“先得到当前 Token，才能计算下一个 Token”的依赖，叫自回归依赖（Autoregressive Dependency）。把不同请求合成一个 Batch 可以增加同一轮的工作量，却不能直接提前知道每条请求未来的 Token。

SGLang 的 Runtime 直接保留了这个区别。当前 <code>forward_batch_info.py</code> 中，<code>ForwardMode.EXTEND</code> 的源码注释明确把它称作通常意义上的 Prefill，而 <code>ForwardMode.DECODE</code> 就是 Decode one token。现实系统还存在 <code>ForwardMode.MIXED</code>，用于 Chunked Prefill 等同时包含 Extend / Decode 的批次。

更关键的是 Scheduler 怎么数 Token。当前 DP Attention batch 准备逻辑里：

~~~python
if local_batch.forward_mode.is_decode():
    num_tokens = local_batch.batch_size()
else:
    num_tokens = local_batch.extend_num_tokens
~~~

因此，在这里讨论的普通 Decode 中，64 条活跃请求一轮约提供 64 行新 Token；一次 Prefill / Extend 则可能提供 512、2048、8192 行甚至更多。这里数的是当前批次的新 Token 行，不能把请求总上下文长度当作矩阵乘的 M；多卡分片、Graph 填充还可能改变某个 Rank 上实际执行的尺寸。

先忽略分片和填充，两种阶段的 M 可以概括为：

~~~text
Prefill:
M ≈ thousands

Decode:
M ≈ active_requests
~~~

同一份权重面对的输入行数，已经相差很多。后面讨论权重复用和专家负载时，都从这个区别出发。

Ascend 910C 的 DeepSeek-V4 Attention Backend 也把这个差异编码进 metadata。Extend 路径按每个 Request 的 <code>extend_seq_lens</code> 构造 Query prefix sum；普通 Decode 则直接构造：

~~~text
actual_seq_lengths_q_pa
=
[0, 1, 2, ..., B]
~~~

这相当于告诉后端：每个 Decode Request 当前只有一个 Query row。

从 Scheduler 组批、ForwardBatch 输入到 Attention Geometry，两阶段已经形成了不同的工作形状。

## 二、Dense GEMM：M 与权重复用

Transformer 里的大量计算最终都会落成矩阵乘：

~~~text
Y = XW

X: [M,K]
W: [K,N]
Y: [M,N]
~~~

M 表示参与这次矩阵乘的输入行数，可以理解为多少行 Token 表示共同使用一份权重。K 是输入特征维度，N 是输出特征维度。下面用 DeepSeek-V4-Flash 的 Attention 投影说明：

<code>wq_a</code>：

~~~text
[M,4096]
×
[4096,1024]
~~~

<code>wkv</code>：

~~~text
[M,4096]
×
[4096,512]
~~~

<code>wq_b</code> 的逻辑全局 shape：

~~~text
[M,1024]
×
[1024,32768]
~~~

实际每 Rank 的 N 还会受 <code>attn_tp_size</code> 影响。这里要区分两个维度：M 主要由当前 workload phase 决定，而 K/N 由模型结构和并行布局共同决定。

为了只观察 M 的影响，假设 <code>wq_a</code> 的 X / W / Y 都按 BF16 计算，每元素 2 Byte。计算量约为：`FLOPs ≈ 2MKN`

最低限度的数据流量近似：`Bytes ≈ 2 × (MK + KN + MN)`

~~~text
AI
≈
2MKN
/
[2(MK + KN + MN)]
~~~

代入 K=4096、N=1024：

| M | 理想 Arithmetic Intensity |
| ---: | ---: |
| 1 | ≈ 1.0 FLOP/B |
| 8 | ≈ 7.9 FLOP/B |
| 64 | ≈ 59 FLOP/B |
| 512 | ≈ 315 FLOP/B |
| 2048 | ≈ 585 FLOP/B |

Arithmetic Intensity（算术强度，简称 AI）表示每搬运一字节数据完成多少计算。表中是只计输入、权重、输出各一次读写的理想估算，单位 FLOP/B；它不是 Ascend 910C 实测值，也没有计入 Cache、Tiling、通信和量化 Scale 等影响。M=1 时，一份大权重只服务一行输入；M=2048 时，读来的权重可以服务更多行，所以每字节对应的计算更多。

当 M 远小于 K/N 时，Weight traffic 通常是主要项，于是可以近似：`AI ≈ 2M / b_w`

其中 <code>b_w</code> 是每个 Weight 元素的字节数。BF16 Weight 下大约有 AI≈M；INT8 Weight 下大约有 AI≈2M。

这也解释了为什么 Quantization 对 Decode 很有吸引力：小 M 下 Weight bandwidth 压力大，减小 Weight bytes 很可能有价值。但真实 W8A8 还要支付 Dynamic Quant、Scale、Tiling、Launch 和 Epilogue 等额外成本，这些开销共同决定最终 TPOT，实际收益取决于省下的权重读取成本能否覆盖量化开销。

Continuous Batching 把不同请求本轮的 Token 合起来，增加 M，也就有机会提高 Dense GEMM 的算术强度。用 Roofline 模型理解时，可以先看设备有效计算能力与有效带宽的比值，也叫转折点（Ridge Point）：

~~~text
R_eff
=
effective FLOP/s
/
effective Byte/s
~~~

当算术强度低于这个比值，带宽上限更容易先限制性能；高于它，计算能力更可能先成为限制。只在权重流量占主导的简单模型里，转折点对应的 M 近似为：

~~~text
M_crossover
≈
R_eff × b_w / 2
~~~

但真实 crossover 还取决于 CANN Tiling、Weight Layout、Cache、Quantization、Kernel Occupancy、TP / DP 与通信。在目标配置上做 batch sweep，可以观察这些因素共同作用下的转折点。

## 三、MoE：总 Token 怎样分散到 Expert

Dense Linear 的 M 还是这一轮的总 Token 数。MoE 决定 Expert GEMM Shape 的却是：

~~~text
M_e
=
Expert e 当前实际收到多少 Token
~~~

估算平均 `M_e` 时，分子取同一 EP 通信域的 Token 总量，分母也按该域的专家数量计算。

DeepSeek-V4-Flash 的公开配置是：

~~~text
256 routed experts
每个 Token 激活 6 个 routed experts
moe_intermediate_size = 2048
hidden_size = 4096
~~~

一个 routed expert 的逻辑 W13 可以近似看成：

~~~text
[M_e,4096]
×
[4096,4096]
~~~

这里 4096 的输出来自 Gate + Up 两个 2048 分支。W2 则近似：

~~~text
[M_e,2048]
×
[2048,4096]
~~~

每个 Token 被送到 6 个 routed experts，因此一轮总 token-expert assignments 约为 6M。

为了建立 Shape 直觉，先假设 Router 完全均匀。真实流量不会如此均匀，这个假设只用于算平均值。

Prefill 如果 M=2048：

~~~text
2048 × 6
=
12288 assignments

12288 / 256
=
48 rows / expert
~~~

即平均 M_e≈48。

Decode 如果有 64 个 active requests：

~~~text
64 × 6
=
384 assignments

384 / 256
=
1.5 rows / expert
~~~

即平均 M_e≈1.5。这里的 1.5 是跨专家的平均数，实际某个专家收到的行数仍是整数；有的可能没有 Token，有的会收到多个。

继续把 W13 按 BF16 估算，得到下表。M_e=1.5 这一行只是把平均负载代入公式，用于比较量级，不表示 Kernel 真能执行半行：

| Expert M_e | AI |
| ---: | ---: |
| 1 | ≈ 1.0 FLOP/B |
| 1.5 | ≈ 1.5 FLOP/B |
| 6 | ≈ 6.0 FLOP/B |
| 48 | ≈ 46.9 FLOP/B |

于是 DeepSeek-V4 的 Prefill / Decode 差异不只是 Dense M=2048 vs M=64，还可能变成 Expert M_e≈48 vs M_e≈1.5。

真实 Router 还会产生不均衡：

~~~text
Expert 0: 0
Expert 1: 1
Expert 2: 7
Expert 3: 0
Expert 4: 13
...
~~~

这意味着某些专家没有工作，另一些专家只做很小的 GEMM。GroupedMatmul 虽然能组织多组专家计算，仍要面对各组尺寸过小或不均衡的问题；跨 Rank 执行时，还要叠加 Dispatch / Combine 通信与等待。总 Token 数相同，不同的专家负载分布也可能产生不同耗时。

分析 DeepSeek-V4 Decode 时，将 global batch size 与各专家的负载分布一起记录，包括 Expert Token Histogram，以及 mean / median / p90 / max M_e 和 empty expert ratio。这也是 EPLB 与 MoE runtime 优化为什么会直接影响 Decode 性能。

## 四、长上下文下的滑窗与压缩历史

普通 MHA 的入门模型常常这样解释 Decode：

~~~text
Q_new: [1,d]

读取:
K_cache[0:L]
V_cache[0:L]
~~~

在这个入门模型里，上下文长度 L 翻倍，读取历史 KV 的工作量也近似翻倍。DeepSeek-V4 对历史状态使用了滑窗和压缩，不能直接沿用这个比例。下面分别算它需要访问的几类数据。

DeepSeek-V4-Flash 的公开配置包含：

~~~text
sliding_window = 128
index_head_dim = 128
index_topk = 512

compress_ratios =
[0, 0, 4, 128, 4, 128, ...]
~~~

当前 SGLang Ascend DSV4 路径因此会出现 SWA、C4 compressed history、C128 compressed history、C4 Lightning Indexer 和 Compressor state。

DeepSeek-V4 的 qk_nope_head_dim=448、qk_rope_head_dim=64，总 row width 为 512。在本文关注的 Ascend 910C 非-arch35 PA_ND 路径里，源码使用 BF16 KV row，因此一个 logical KV row 大约：

~~~text
512 × 2 Bytes
=
1024 Bytes
≈
1 KiB
~~~

较新的 arch35 分支会使用 packed FP8 KV 布局，因此下面的字节数只适用于本文关注的 910C 路径。更重要的是，下面计算的是 logical payload scale，而不是实际 HBM transaction。Cache 命中、Prefetch、Page Layout、Tiling、量化格式和多 Head 复用都会改变真实搬运量。

SWA 的历史读取是有界的。公开配置的 sliding_window=128，意味着原始 SWA branch 的历史窗口大约被限制在 128 rows，粗略 logical payload 约 128 KiB。它不会随着 32K、128K、1M Context 无限线性增长。

C4 则不同。compress_ratio=4，长度 L 的 Context 对应约 ceil(L/4) 个 compressed positions。当前 NPU DSV4 backend 的 C4 Attention 最终会使用 <code>c4_topk_indices</code>，公开配置 index_topk=512，所以长 Context 下最终 C4 Attention 消费的 compressed rows 上限约为 512。

如果只看最终 Attention 的 logical KV payload：

~~~text
SWA:
128 × 1 KiB

+

C4 top-k:
512 × 1 KiB

≈
640 KiB / query / C4 layer
~~~

这里估算的只是最终 Attention 选中后的数据量。上下文继续增长时，C4 路径还要先从更长的压缩历史中找出 Top-K，这部分工作由 Indexer 承担。不能因为最终只读取固定数量的候选，就认为整条 C4 路径成本不再增长。

C4 Lightning Indexer 面对的 key history 规模大约是 ceil(L/4)。公开配置 index_head_dim=128。在本文关注的 910C 路径里，Indexer storage 使用 INT8 key + FP16 scale，因此每个 compressed index row 的原始 payload 约为：

~~~text
128 Bytes INT8 key
+
2 Bytes scale
≈
130 Bytes
~~~

于是 logical key payload 规模大约：`ceil(L/4) × 130 Bytes`

| Context L | C4 Index rows | Logical key payload |
| ---: | ---: | ---: |
| 8K | 2048 | ≈ 0.25 MiB |
| 32K | 8192 | ≈ 1.0 MiB |
| 128K | 32768 | ≈ 4.1 MiB |
| 1M | 262144 | ≈ 32.5 MiB |

表中计算的是逻辑 key 数据量，实际 HBM 流量与耗时还取决于缓存及算子访问方式。可以看到，最终 C4 Sparse Attention 的候选数受 Top-K 限制，而选出候选之前，Indexer 面对的历史规模仍随 Context 增长。两部分要分开看。

因此，长 Context Decode 的 Profile 应分别记录 Lightning Indexer 和最终 Sparse Attention 的耗时。

C128 又是另一种 Shape。compress_ratio=128，当前 Ascend backend 对 C128 设置 <code>cmp_sparse_indices=None</code>，也就是读取完整的 C128 compressed history。长度 L 对应 ceil(L/128) 个 compressed rows。

logical payload 可以近似成：

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

它仍然随 Context 增长，但增长斜率已经从 1 row / original token 压缩成 1 row / 128 original tokens。

DSV4 Compressor 还有第三类数据：单独的 FP32 state cache。源码中非-online state 的 last_dim 为：`2 × (1 + overlap) × head_dim`

C4 的 overlap=True，因此一行大约是 2048 FP32，也就是约 8 KiB；C128 的 overlap=False，一行约 1024 FP32，也就是约 4 KiB。

每步实际访问的 state rows 随 Compressor Plan、Ring State、当前位置和融合算子而变，可以用当步 metadata 与 trace 对齐。

把整轮涉及的 Attention 工作归类，可以写成下面的成本分解。C4 与 C128 对应不同类型的层，应分别统计相应层的耗时；这个式子不是说每一层都依次执行所有分支，也没有把通信和计算重叠展开：

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

DeepSeek-V4 仍然会随着 Context 变长承受越来越重的历史状态成本，但增长发生在哪个子模块、以什么斜率增长，已经与标准 MHA 完全不同。

## 五、跨 Token 依赖与重复开销

即使暂时忽略 KV / compressed history，Decode 仍然有一个无法逃避的问题：未来 Token 尚不存在。

第 t 个输出 Token 的链路是：

~~~text
Forward
  ↓
Logits
  ↓
Sampling
  ↓
Token_t
~~~

只有 Token_t 确认以后，下一轮才能开始：

~~~text
Token_t
  ↓
Embedding
  ↓
Forward
  ↓
Token_t+1
~~~

因此普通生成要反复沿这条依赖推进。若首个输出 Token 由 Prefill 末尾采样得到，生成 1000 个输出 Token 通常还需要约 999 轮 Decode；具体轮数还会受停止条件等影响。这里的重点是这些轮次前后依赖，不能像 Prompt 那样直接把未来 1000 个 Token 一次送入模型。

每一步不只有 GEMM，还可能包含 Scheduler、ForwardBatch preparation、Graph replay / launch、Attention metadata、Collective、DeepEP、Sampling 和 State publication。即使每项成本只有几微秒或几十微秒，重复几百到几千次以后都可能进入 TPOT。

Prefill 更容易摊薄这些固定成本。一轮 2048-token Prefill 也需要 Launch、Runtime dispatch、metadata 和 collective，但这些成本可以被 2048 个 Token 共同承担。

所以 Prefill / Decode 差异不只有 Compute vs Memory，还有：

~~~text
large useful work / launch
vs
small useful work / launch
~~~

Graph Replay 可以减少重复提交，Kernel Fusion 可以减少算子启动与中间搬运，Host Overlap 和 Multi-Stream 则尝试把相互独立的准备、计算或通信重叠起来。它们针对的成本不同，也受依赖与资源竞争限制；只有缩短了整轮关键路径，才能改善 TPOT。

同样的逻辑也解释了 Chunked Prefill。一个巨大的 Prefill 往往硬件利用率很好，但如果它一次占设备太久，正在流式输出的 Decode Request 就只能等待。Chunked Prefill 将长输入分成较短的计算批次，给对尾延迟敏感的 Decode 留出调度机会。

## 六、常见优化分别作用在哪里

把前面的因果关系放到一起：

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

沿图中的不同环节，可以分清常见优化各自改动了什么。KV Cache 保存过去 Token 的状态，避免重复计算；Paged Cache 管理不同长度请求的物理缓存空间，Radix Cache 进一步复用请求间命中的前缀。它们处理的是状态存储与复用。

Continuous Batching 则把不同请求当前的 Decode Token 合起来，增加 M；Chunked Prefill 控制一轮 Prefill 的工作量，给正在流式输出的 Decode 留出调度机会。两者都改变组批方式，但前者主要增加同轮有效工作，后者还要权衡大批效率与请求等待。

Quantization 减少权重等数据的字节数，收益取决于 M、M_e、Tiling 和额外量化成本。Speculative Decoding 提前提出候选，让一次 Target 验证尝试推进多个 Token；它增加了 Draft、Verify 与状态整理的工作，需要比较整个周期的成本。

PD Disaggregation 把 Prefill 与 Decode 放到不同资源池，使两边可以分别选择组批与并行参数，同时引入 KV 传输和请求交接成本。选择哪种优化，要回到当前瓶颈：是权重复用不足、历史状态访问太重、调度等待，还是每轮固定开销过高。

## 七、用 Profile 判断实际瓶颈

前面推导的是工作量形状与可能的瓶颈，本文没有给出目标配置下的实测利用率。要判断 Ascend 910C 上究竟哪一项限制性能，需要把算子耗时和硬件指标放回同一轮执行。

Ascend Profiling 中，可以先查看目标产品和工具版本实际支持的流水指标：`mac_ratio` 表示 Cube / Matrix 指令周期占比，`vec_ratio` 表示 Vector 指令周期占比，`mte2_ratio` 描述 MTE2 数据搬运相关周期占比。它们解释的是算子内部的工作比例，需要与算子总耗时一起读。

官方还定义了下面的 `memory_bound` 比值：

~~~text
memory_bound
=
mte2_ratio
/
max(mac_ratio, vec_ratio)
~~~

在支持该字段的采集口径中，<code>memory_bound &gt; 1</code> 表示数据搬运相对主要计算流水形成了压力。字段是否提供、流水名称和支持范围会随产品与工具版本变化，缺失时应保留 N/A，不能当作零或直接推断没有瓶颈。它也不能单独代替整轮 Decode 的关键路径分析；可对照[官方指标与产品支持说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0069.html)。

如果要验证这些形状推导，可以按下面的方案设计对照；本文没有执行这组 sweep。先固定 checkpoint、quantization、TP / DP / EP、CANN、Attention backend 和 MoE 通信配置，同时对齐请求长度、缓存命中、请求数与统计区间。

Prefill sweep：

~~~text
new tokens M:

128
512
1024
2048
4096
8192
~~~

观察 Dense GEMM、GroupedMatmul、Attention、Lightning Indexer、Compressor、Collective 对应的 latency、mac_ratio、vec_ratio、mte2_ratio 和 memory_bound。

Decode sweep 则固定 Context，例如 8K、32K、128K，再 sweep active requests：

~~~text
B:

1
2
4
8
16
32
64
128
~~~

同时记录 Dense M、Expert M_e histogram、Dense GEMM latency、GroupedMatmul latency、Attention latency、Indexer latency、DeepEP / HCCL、Host Gap、TPOT 和 output tok/s。

“Decode crossover”至少有三种。第一种是 Dense GEMM crossover；第二种是 MoE crossover，此时横轴是 Expert M_e distribution；第三种才是 End-to-End Decode crossover。

即使 Dense GEMM 已经进入更偏 Compute-heavy 的区域，如果 Attention、C4 Indexer、DeepEP 或 Host Gap 仍然占据 Critical Path，这些阶段仍可能限制整轮 Decode 的速度。

随着 Batch 增大，Dense 和 Expert GEMM 可以提高权重复用，单个算子也可能跨过有效 Roofline 的转折点。整轮 Decode 是否因此更快，还取决于 Attention 历史读取、Indexer、通信与 Host Gap。用 Batch × Context 的组合实验观察这些组成部分，才能找到目标配置的性能转折点。

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
