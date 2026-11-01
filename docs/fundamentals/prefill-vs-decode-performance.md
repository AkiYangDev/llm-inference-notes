# Prefill 与 Decode 为什么性能不同：从计算形状到执行开销

同一份模型权重，2048 行 Prompt 可以一起使用，单条普通 Decode 请求这一轮却只有一行新输入。组到 64 条活跃请求，Dense 层有了 64 行；按后文均匀路由的算例，进入 MoE 分给专家后，平均每个 Expert 只剩约 1.5 行。

Prefill、Decode 的性能差异，先从这些形状变化开始。权重能被多少输入复用，专家实际分到多少行，历史又要读多少，决定了计算和数据搬运的比例。下面沿 DeepSeek-V4-Flash，把这些工作与每轮重复的执行开销算在一起。

前两节从新 Token 数、权重复用入手，适合刚开始读推理日志时对照。后面再展开 MoE、滑窗、压缩历史和 Profile。遇到不熟悉的术语，可以查[推理基础设施名词表](ai-infra-working-glossary.md)。

主要源码固定为 SGLang commit <code>b86a30afbae389eec8763f4bcf38f14904d2d00c</code>，模型案例为 DeepSeek-V4-Flash-0731，核对日期为 2026-09-21。

GEMM 算例先统一使用 BF16 字节数，以单独观察 Shape 对算术强度的影响。实际官方 checkpoint 含 FP8 / FP4，Ascend 也可能使用 W8A8；dtype 改变字节系数，大 M 分摊权重读取、小 M 难以分摊的关系仍在。

## 一、新 Token 数怎样改变计算形状

假设 Prompt 有 2048 个 Token，目标输出 512 个 Token：

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

Prompt 已经给定，Prefill 可以同时计算。普通自回归 Decode 要先采样出 Token(t)，才知道下一轮输入；单请求的第 1、2、3 轮通常各新增一枚，依次推进。

这条先后关系就是自回归依赖（Autoregressive Dependency）。Batch 能合并不同请求当前已知的输入，每条请求未来的 Token 仍要等待上一轮。

SGLang Runtime 也按此区分模式。固定版本 <code>forward_batch_info.py</code> 将 <code>ForwardMode.EXTEND</code> 注释为通常意义的 Prefill，<code>ForwardMode.DECODE</code> 为 Decode one token，<code>ForwardMode.MIXED</code> 则包含 Chunked Prefill 等 Extend / Decode 混合批次。

准备 DP Attention batch 时，Scheduler 分别计数：

~~~python
if local_batch.forward_mode.is_decode():
    num_tokens = local_batch.batch_size()
else:
    num_tokens = local_batch.extend_num_tokens
~~~

普通 Decode 的 64 条活跃请求约提供 64 行新输入；Prefill / Extend 一轮可能有 512、2048、8192 行或更多。GEMM 的 M 因而先取当前输入，再由各 Rank 的分片、Graph padding 形成实际执行尺寸。总上下文长度不能直接代入 M。

暂不计分片、padding，Prefill 的 M 可以达到数千，普通 Decode 的 M 近似为活跃请求数。

同一层权重，由此面对不同数量的输入行。

Ascend 910C DSV4 backend 也按阶段准备 metadata。Extend 使用每请求 <code>extend_seq_lens</code> 构造 Query prefix sum；普通 Decode 使用 `actual_seq_lengths_q_pa = [0,1,2,...,B]`，每请求对应一个 Query row。

Scheduler 计数、ForwardBatch 输入、Attention Query 因而表达同一轮工作在不同层的形状。

## 二、Dense GEMM：M 与权重复用

先取模型中常见的矩阵乘：

~~~text
Y = XW

X: [M,K]
W: [K,N]
Y: [M,N]
~~~

M 是共同使用权重的输入行数，K、N 分别是输入、输出特征维度。DeepSeek-V4-Flash 的几项 Attention 投影形状为：

| 投影 | 输入 | 权重的逻辑全局形状 |
| --- | --- | --- |
| `wq_a` | [M,4096] | [4096,1024] |
| `wkv` | [M,4096] | [4096,512] |
| `wq_b` | [M,1024] | [1024,32768] |

实际每 Rank 的 <code>wq_b</code> 输出 N 还受 <code>attn_tp_size</code> 影响。M 随本轮负载变化，K/N 则取决于结构、并行切分。

只改变 M，令 <code>wq_a</code> 的 X、W、Y 都为 BF16，每元素 2 Byte，计算量约为 `FLOPs ≈ 2MKN`。

输入、权重、输出各计一次读写，数据量为 `Bytes ≈ 2 × (MK + KN + MN)`：

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

Arithmetic Intensity（算术强度，AI）的单位为 FLOP/B，表示每搬一字节完成多少计算。M=1，整份权重只服务一行；M=2048，更多输入共用权重，计算量相对搬运量更大。表中未计 Cache、Tiling、通信、量化 Scale，是理想估算。

M 远小于 K/N、权重流量占主要部分时，可以近似为 `AI ≈ 2M / b_w`。

<code>b_w</code> 为每个权重元素字节数，因此 BF16 近似 AI≈M，INT8 近似 AI≈2M。

小 M 减少 Weight bytes，有机会缓解带宽压力，量化由此可能帮助 Decode。W8A8 同时增加 Dynamic Quant、Scale、Tiling、Launch、Epilogue 等工作，节省的读取要覆盖它们，TPOT 才会下降。

Continuous Batching 把多请求当前 Token 合并，增大 M，改善权重复用。Roofline 转折点（Ridge Point）则用有效计算能力除以有效带宽：

~~~text
R_eff
=
effective FLOP/s
/
effective Byte/s
~~~

算术强度低于该比值，更容易先碰到带宽上限；高于它，计算能力更可能成为限制。按权重流量主导的简化模型，可估算对应 M：

~~~text
M_crossover
≈
R_eff × b_w / 2
~~~

实际转折还受 CANN Tiling、Weight Layout、Cache、Quantization、Kernel Occupancy、TP / DP、通信影响，要在目标配置的 batch sweep 中观察。

## 三、MoE：总 Token 怎样分散到 Expert

Dense Linear 可以统一处理本轮输入。MoE 路由之后，Expert e 的 GEMM 只处理实际收到的 `M_e` 行。

估算平均 `M_e` 时，总 Token 数、专家数都要来自同一 EP 通信域。

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

Gate、Up 各输出 2048，合成 4096，之后 W2 的形状为：

~~~text
[M_e,2048]
×
[2048,4096]
~~~

每 Token 激活 6 个 routed experts，M 行约产生 6M 个 token-expert assignments。

先按均匀路由计算平均值：

Prefill M=2048 与普通 Decode 的 64 条活跃请求对应为：

| 阶段 | M | Token–Expert assignments | 平均 M_e |
| --- | ---: | ---: | ---: |
| Prefill | 2048 | 2048×6=12288 | 12288/256=48 |
| Decode | 64 | 64×6=384 | 384/256=1.5 |

平均值可以是 1.5，实际专家接收的行数仍为整数，有的为空，有的收到多行。

以 BF16 估算 W13，结果如下。M_e=1.5 只用于比较平均负载量级：

| Expert M_e | AI |
| ---: | ---: |
| 1 | ≈ 1.0 FLOP/B |
| 1.5 | ≈ 1.5 FLOP/B |
| 6 | ≈ 6.0 FLOP/B |
| 48 | ≈ 46.9 FLOP/B |

从 Prefill 到 Decode，Dense M 从 2048 变为 64；平均 Expert M_e 则由约 48 降为约 1.5。

真实路由还会偏斜，例如：

~~~text
Expert 0: 0
Expert 1: 1
Expert 2: 7
Expert 3: 0
Expert 4: 13
...
~~~

GroupedMatmul 把多组专家计算组织起来，却仍要执行过小或不均衡的尺寸。跨 Rank 再加入 Dispatch / Combine、等待，总 Token 相同也可能因专家分布不同而耗时不同。

因此 global batch size 之外，Expert Token Histogram、mean / median / p90 / max M_e、empty expert ratio 更能说明专家实际做多少工作。EPLB、MoE runtime 会改变这组执行条件。

## 四、长上下文下的滑窗与压缩历史

普通 MHA 的 Decode 历史读取常用下面的模型说明：

~~~text
Q_new: [1,d]

读取:
K_cache[0:L]
V_cache[0:L]
~~~

在该模型中，L 翻倍，历史 KV 读取约翻倍。V4 使用滑窗、压缩历史，各子路径的增长方式不同。

DeepSeek-V4-Flash 的公开配置包含：

~~~text
sliding_window = 128
index_head_dim = 128
index_topk = 512

compress_ratios =
[0, 0, 4, 128, 4, 128, ...]
~~~

SGLang Ascend DSV4 对应 SWA、C4 compressed history、C128 compressed history、C4 Lightning Indexer、Compressor state。

qk_nope_head_dim=448、qk_rope_head_dim=64，合计一行 512 维。本文关注 Ascend 910C 非 arch35 的 PA_ND 路径，KV row 使用 BF16，logical payload 约为 `512 × 2 = 1024 Bytes`，即 1 KiB。

较新的 arch35 使用 packed FP8 KV，不能沿用这组字节数。logical payload 也不等于 HBM transaction，实际搬运还受 Cache、Prefetch、Page Layout、Tiling、格式和多 Head 复用影响。

sliding_window=128，将 SWA 历史约束在约 128 rows，粗估为 128 KiB。上下文从 32K 到 128K、1M，这部分不会跟着无限线性增加。

C4 的 compress_ratio=4，压缩历史约有 ceil(L/4) 行。最终 Attention 再由 <code>c4_topk_indices</code> 选取，公开 index_topk=512，所以长上下文下消费的 compressed rows 上限约为 512。

只估最终 Attention 的逻辑 KV 读取：

~~~text
SWA:
128 × 1 KiB

+

C4 top-k:
512 × 1 KiB

≈
640 KiB / query / C4 layer
~~~

Indexer 还要从全部压缩历史选候选，这部分工作随上下文继续增长。

C4 Lightning Indexer 面对约 ceil(L/4) 个 key，公开 index_head_dim=128。在本文 910C 路径，storage 使用 INT8 key、FP16 scale，每行原始 payload 约为 `128 Bytes INT8 key + 2 Bytes scale = 130 Bytes`。

总逻辑 key 数据量为 `ceil(L/4) × 130 Bytes`：

| Context L | C4 Index rows | Logical key payload |
| ---: | ---: | ---: |
| 8K | 2048 | ≈ 0.25 MiB |
| 32K | 8192 | ≈ 1.0 MiB |
| 128K | 32768 | ≈ 4.1 MiB |
| 1M | 262144 | ≈ 32.5 MiB |

表中数据说明候选选择前的历史规模。实际 HBM 流量、耗时还由缓存和算子访问方式决定，最终 Top-K 有上限，不能据此推断 Indexer 工作也已封顶。

Profile 因而要把 Lightning Indexer 与最终 Sparse Attention 分开统计。

C128 的 compress_ratio=128，Ascend backend 设置 <code>cmp_sparse_indices=None</code>，读取约 ceil(L/128) 行的完整压缩历史。

加上 SWA，logical payload 近似为：

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

它随上下文增加，但每增加 128 个原始 Token，约增加一行压缩历史。

Compressor 另外维护 FP32 state cache，非 online state 的 last_dim 为 `2 × (1 + overlap) × head_dim`。

C4 取 overlap=True，每行约 2048 个 FP32、8 KiB；C128 取 overlap=False，每行约 1024 个 FP32、4 KiB。

每步访问多少 state rows，还由 Compressor Plan、Ring State、位置、融合算子决定，可以对照该步 metadata 和 trace。

将它们写成成本分解。C4、C128 对应不同层，图不表示每层依次执行全部分支，也未展开通信、计算的重叠：

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

上下文增长会增加历史相关工作，具体增长来自哪个模块，要按各路径判断。标准 MHA 的统一比例已经不足以描述这套结构。

## 五、跨 Token 依赖与重复开销

另一项成本来自每轮确认输入以后才能继续的依赖。

Forward、Logits、Sampling 得到 Token_t，再送进 Embedding、下轮 Forward，才会采到 Token_t+1。单条请求未来的输入因此还未全部给定。

如果 Prefill 末尾给出首个输出，生成 1000 枚通常还需约 999 轮 Decode，最终轮数也受停止条件影响。这些未来 Token 无法像已知 Prompt 那样直接成批输入。

每轮还可能经过 Scheduler、ForwardBatch preparation、Graph replay / launch、Attention metadata、Collective、DeepEP、Sampling、State publication。每项几微秒、几十微秒，反复执行几百至几千轮，也会进入 TPOT。

Prefill 同样要提交任务、准备 metadata、执行 collective，但一次 2048-token Prefill 能把相关开销分给更多 Token。

因此每次 launch 完成多少有效工作，比单看 Kernel 启动次数更能解释效率。

Graph Replay 减少重复提交；Fusion 减少启动和中间搬运；Host Overlap、Multi-Stream 则让独立准备、计算、通信并发。依赖、资源竞争仍在，收益最后体现为整轮关键路径是否缩短。

Chunked Prefill 也在调节每轮工作量。大 Prefill 可能充分利用设备，却让流式请求等待；拆短能给 Decode 留调度机会，同时增加批次轮数。

## 六、常见优化分别作用在哪里

前面的形状、历史和执行关系，可以对应到常见优化：

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

KV Cache 保存算过的历史，Paged Cache 管不同长度请求的物理空间，Radix Cache 进一步复用请求间前缀，三者改变状态存储、复用。

Continuous Batching 合并当前 Decode Token、增大 M；Chunked Prefill 控制单轮 Prefill 工作量，给 Decode 留机会。它们都改变组批，作用位置不同。

Quantization 减少字节数，成本取决于 M、M_e、Tiling 和量化本身。Speculative Decoding 让一次 Target 验证尝试多走几个位置，也要支付 Draft、Verify、状态整理的整轮成本。

PD Disaggregation 让 Prefill、Decode 分别配置资源池、组批、并行参数，同时增加 KV 传输和请求交接。选择优化之前，先确定目标时间花在权重、历史、调度还是重复执行上。

## 七、用 Profile 判断实际瓶颈

前面推导的是形状和可能限制。判断目标 Ascend 910C 配置的实际瓶颈，还需要将硬件指标、算子耗时、整轮时间对齐。

Ascend Profiling 的 `mac_ratio` 描述 Cube / Matrix 指令周期占比，`vec_ratio` 描述 Vector，`mte2_ratio` 描述 MTE2 数据搬运相关周期。字段是否可用，取决于产品、工具版本，含义也要与算子总耗时结合。

官方 `memory_bound` 定义为：

~~~text
memory_bound
=
mte2_ratio
/
max(mac_ratio, vec_ratio)
~~~

支持该口径时，<code>memory_bound &gt; 1</code> 表示数据搬运相对主要计算流水形成压力。产品、工具版本决定字段和流水名称，缺失记 N/A。它描述算子内部，整轮关键路径还要结合其他证据。字段范围见[官方指标与产品支持说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0069.html)。

下面是一组未执行的验证方案。先固定 checkpoint、quantization、TP / DP / EP、CANN、Attention backend、MoE 通信配置，并对齐长度、缓存、请求数、统计区间。

Prefill 比较 M=128、512、1024、2048、4096、8192。

各点观察 Dense GEMM、GroupedMatmul、Attention、Lightning Indexer、Compressor、Collective 的 latency、mac_ratio、vec_ratio、mte2_ratio、memory_bound。

Decode 固定 Context，例如 8K、32K、128K，再比较活跃 B=1、2、4、8、16、32、64、128。

记录 Dense M、Expert M_e histogram、Dense GEMM / GroupedMatmul / Attention / Indexer latency、DeepEP / HCCL、Host Gap、TPOT、output tok/s，把算子变化接到端到端产出。

这组对照会涉及三个不同转折：Dense GEMM，随 M_e distribution 变化的 MoE，以及整轮 Decode。

Dense 已偏向 Compute-heavy，Attention、C4 Indexer、DeepEP、Host Gap 仍可能主导整轮速度。

Batch 增长影响权重复用，Context 增长影响历史和索引。把 Batch × Context 放在一起，并保留整轮指标，才可找到对目标服务有用的转折点。

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
