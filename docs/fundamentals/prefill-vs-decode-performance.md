# 为什么 Prefill 和 Decode 明明跑的是同一个模型，性能却完全不同？

同一个 DeepSeek-V4，同一份模型权重，同样的 Attention、MoE、RMSNorm 和 Linear，甚至运行在同一组 Ascend 910C 上，Prefill 和 Decode 的性能表现却常常像两个完全不同的程序。

最常见的总结是：Prefill 更偏 Compute Bound，Decode 更偏 Memory Bound。这个说法可以作为第一层直觉，但不能当成定律。更准确的说法是：

> **模型结构没有变，但送给模型的 Workload Geometry 变了。**

一次 Forward 处理多少新 Token、Dense GEMM 的 M 有多大、MoE 每个 Expert 实际收到多少 Token、Weight 能被复用多少次、Attention 要读取多少历史状态、一次 Launch 与通信开销能被多少 Token 摊薄，以及下一轮能否提前开始，这些都会随着 Prefill / Decode 切换而变化。

~~~text
Same Model
≠
Same Workload
≠
Same Hardware Behavior
~~~

这也是为什么现代 LLM Serving 会自然长出 KV Cache、Continuous Batching、Paged / Radix Cache、Chunked Prefill、Graph Replay、Quantization、Speculative Decoding 和 PD Disaggregation：它们并不是一堆互不相关的优化，而是在分别修复 Prefill 与 Decode 两片完全不同的工作负载空间。

本文主要固定到 SGLang commit <code>b86a30afbae389eec8763f4bcf38f14904d2d00c</code>，模型以 DeepSeek-V4-Flash-0731 为具体案例，Review 日期为 2026-09-21。

为了把 Shape 对算术强度的影响单独讲清楚，部分 GEMM 示例会统一使用 BF16 bytes 做思想实验。官方 DeepSeek-V4-Flash checkpoint 实际包含 FP8 / FP4 等量化配置，不同 Ascend 部署也可能采用 W8A8 等路径；dtype 会改变数值系数，但不会改变“大 M 更容易复用 Weight、小 M 更难摊薄 Weight / Launch 成本”的基本结论。

---

## 一、第一性原因：Prefill 一次处理很多 Query，Decode 每个 Request 通常只有一个

假设用户输入 2048 个 Prompt Token，然后让模型生成 512 个 Output Token。

~~~mermaid
flowchart LR
    A["Prompt<br/>2048 Tokens"]
    B["Prefill<br/>一次处理很多新 Token"]
    C["First Token"]
    D["Decode Step 1"]
    E["Decode Step 2"]
    F["..."]
    G["Decode Step 511"]

    A --> B --> C --> D --> E --> F --> G
~~~

Prefill 可以把大量 Prompt Token 一起送进模型。Decode 却不能提前把未来 512 个 Token 一次算出来，因为 Token(t+1) 依赖 Token(t) 的 Sampling 结果。对单个 Request 来说，Decode 更接近：

~~~text
Step 1: 1 个新 Token
Step 2: 1 个新 Token
Step 3: 1 个新 Token
...
~~~

这就是 Autoregressive Dependency。

SGLang 的 Runtime 直接保留了这个区别。当前 <code>forward_batch_info.py</code> 中，<code>ForwardMode.EXTEND</code> 的源码注释明确把它称作通常意义上的 Prefill，而 <code>ForwardMode.DECODE</code> 就是 Decode one token。现实系统还存在 <code>ForwardMode.MIXED</code>，用于 Chunked Prefill 等同时包含 Extend / Decode 的批次。

更关键的是 Scheduler 怎么数 Token。当前 DP Attention batch 准备逻辑里：

~~~python
if local_batch.forward_mode.is_decode():
    num_tokens = local_batch.batch_size()
else:
    num_tokens = local_batch.extend_num_tokens
~~~

因此，如果有 64 个 active Decode Request，一轮大约只有 64 个新 Token row；而一次 Prefill / Extend 可能是 512、2048、8192 甚至更多 Token。

也就是说，即使模型完全没变：

~~~text
Prefill:
M ≈ thousands

Decode:
M ≈ active_requests
~~~

差别已经出现了。

Ascend 910C 的 DeepSeek-V4 Attention Backend 也把这个差异编码进 metadata。Extend 路径按每个 Request 的 <code>extend_seq_lens</code> 构造 Query prefix sum；普通 Decode 则直接构造：

~~~text
actual_seq_lengths_q_pa
=
[0, 1, 2, ..., B]
~~~

这相当于告诉后端：每个 Decode Request 当前只有一个 Query row。

所以 Prefill / Decode 的性能差异不是后来做 Profiling 时才出现的现象，而是从 Scheduler、ForwardBatch 到 Attention Geometry 一开始就已经不同。

---

## 二、把 DeepSeek-V4 的真实 Dense Shape 代进去，大 M / 小 M 的差异马上可见

Transformer 里的大量计算最终都会落成矩阵乘：

~~~text
Y = XW

X: [M,K]
W: [K,N]
Y: [M,N]
~~~

这里最重要的变量之一就是 M，因为它近似表示这一轮有多少 Token row 同时复用这份 Weight。

直接看 DeepSeek-V4-Flash 的真实 Attention projection。

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

为了只观察 M 的影响，假设 <code>wq_a</code> 的 X / W / Y 都按 BF16 计算，每元素 2 Byte。计算量约为：

~~~text
FLOPs ≈ 2MKN
~~~

最低限度的数据流量近似：

~~~text
Bytes ≈ 2 × (MK + KN + MN)
~~~

因此：

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

这些不是 Ascend 910C 实测值，而是只保留输入、Weight、输出最低数据量的 Roofline-style 思想实验。它没有计入 Cache、Tiling、通信、量化 Scale 等成本，但核心物理直觉已经很清楚：M=1 时，大量 Weight 基本只服务一个 Row；M=2048 时，同一份 Weight 同时服务 2048 个 Row。

当 M 远小于 K/N 时，Weight traffic 通常是主要项，于是可以近似：

~~~text
AI ≈ 2M / b_w
~~~

其中 <code>b_w</code> 是每个 Weight 元素的字节数。BF16 Weight 下大约有 AI≈M；INT8 Weight 下大约有 AI≈2M。

这也解释了为什么 Quantization 对 Decode 很有吸引力：小 M 下 Weight bandwidth 压力大，减小 Weight bytes 很可能有价值。但真实 W8A8 还要支付 Dynamic Quant、Scale、Tiling、Launch 和 Epilogue 等额外成本，因此 INT8 理论更有利，并不等于 Decode TPOT 必然同比提升。

Continuous Batching 的 Dense crossover 同样不能凭规格表猜。如果设备的有效计算/带宽 Ridge Point 是：

~~~text
R_eff
=
effective FLOP/s
/
effective Byte/s
~~~

在 Weight-dominated 的简单模型里：

~~~text
M_crossover
≈
R_eff × b_w / 2
~~~

但真实 crossover 还取决于 CANN Tiling、Weight Layout、Cache、Quantization、Kernel Occupancy、TP / DP 与通信。因此不能仅凭芯片规格宣布 Ascend 910C Decode 在 Batch=32 或 Batch=64 一定进入 Compute Bound，这个数字只能通过目标配置上的 batch sweep 得到。

---

## 三、DeepSeek-V4 的 MoE 会把 Prefill / Decode 的 Shape 差异再放大一层

Dense Linear 的 M 还是这一轮的总 Token 数。MoE 真正决定 Expert GEMM Shape 的却是：

~~~text
M_e
=
Expert e 当前实际收到多少 Token
~~~

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

即平均 M_e≈1.5。

如果继续把 W13 当 BF16 思想实验：

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

这会继续引入 empty experts、tiny Expert GEMM、GroupedMatmul 利用率下降、DeepEP dispatch / combine 开销和 load imbalance。

所以分析 DeepSeek-V4 Decode 时，只看 global batch size 远远不够。真正值得记录的是 Expert Token Histogram，以及 mean / median / p90 / max M_e 和 empty expert ratio。这也是 EPLB 与 MoE runtime 优化为什么会直接影响 Decode 性能。

---

## 四、长 Context Decode 不能再写成“每轮扫描完整 KV”：DSV4 已经是 SWA + C4 + C128

普通 MHA 的入门模型常常这样解释 Decode：

~~~text
Q_new: [1,d]

读取:
K_cache[0:L]
V_cache[0:L]
~~~

因此容易得到 Decode KV traffic∝L 的直觉。这对建立最初认识有用，但直接套到 DeepSeek-V4 会开始失真。

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

超过一定长度后，这部分不会继续按 L 增长。但成本没有消失，而是被移动到了 Indexer。

C4 Lightning Indexer 面对的 key history 规模大约是 ceil(L/4)。公开配置 index_head_dim=128。在本文关注的 910C 路径里，Indexer storage 使用 INT8 key + FP16 scale，因此每个 compressed index row 的原始 payload 约为：

~~~text
128 Bytes INT8 key
+
2 Bytes scale
≈
130 Bytes
~~~

于是 logical key payload 规模大约：

~~~text
ceil(L/4) × 130 Bytes
~~~

| Context L | C4 Index rows | Logical key payload |
| ---: | ---: | ---: |
| 8K | 2048 | ≈ 0.25 MiB |
| 32K | 8192 | ≈ 1.0 MiB |
| 128K | 32768 | ≈ 4.1 MiB |
| 1M | 262144 | ≈ 32.5 MiB |

这里绝不能理解成“1M Context 一定从 HBM 搬 32.5 MiB，然后除带宽就是 Kernel latency”。正确含义是：

> **C4 把最终 Sparse Attention 的历史消费限制在 Top-K，但为了找出这些 Top-K，Indexer 本身仍面对一个随 Context 增长的压缩历史。**

因此长 Context Decode 做 Profiling 时，不能只盯最终 Sparse Attention Kernel，还要把 Lightning Indexer 单独拿出来。

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

DSV4 Compressor 还有第三类数据：单独的 FP32 state cache。源码中非-online state 的 last_dim 为：

~~~text
2 × (1 + overlap) × head_dim
~~~

C4 的 overlap=True，因此一行大约是 2048 FP32，也就是约 8 KiB；C128 的 overlap=False，一行约 1024 FP32，也就是约 4 KiB。

但这里不能进一步声称“每一个 Decode Step 固定读取多少个 state rows”。真实访问量依赖 Compressor Plan、Ring State、当前位置和 fused operator，必须结合目标 trace / metadata 才能确认。

所以比“Decode 每轮读取完整 KV”更准确的 DSV4 长 Context 模型应该写成：

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

---

## 五、Decode 还有一个与 HBM 无关、但同样致命的问题：跨 Token 串行与固定成本反复支付

即使暂时忽略 KV / compressed history，Decode 仍然有一个 Prefill 无法逃避的问题：未来 Token 尚不存在。

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

只有 Token_t 真正确认以后，下一轮才能开始：

~~~text
Token_t
  ↓
Embedding
  ↓
Forward
  ↓
Token_t+1
~~~

所以生成 1000 个 Output Token，通常意味着大约 1000 次依次推进的 Decode iteration。

每一步不只有 GEMM，还可能包含 Scheduler、ForwardBatch preparation、Graph replay / launch、Attention metadata、Collective、DeepEP、Sampling 和 State publication。即使每项成本只有几微秒或几十微秒，重复几百到几千次以后都可能进入 TPOT。

Prefill 更容易摊薄这些固定成本。一轮 2048-token Prefill 也需要 Launch、Runtime dispatch、metadata 和 collective，但这些成本可以被 2048 个 Token 共同承担。

所以 Prefill / Decode 差异不只有 Compute vs Memory，还有：

~~~text
large useful work / launch
vs
small useful work / launch
~~~

这正是为什么 Graph Replay、Kernel Fusion、Host Overlap 和 Multi-Stream 对 Decode 尤其重要：它们很多时候不是在让一个大 GEMM 本身更快，而是在减少每个 Token 都要反复支付的固定开销。

同样的逻辑也解释了 Chunked Prefill。一个巨大的 Prefill 往往硬件利用率很好，但如果它一次占设备太久，正在流式输出的 Decode Request 就只能等待。Chunked Prefill 的本质不是“Prefill 算不动”，而是不要让高吞吐的大 Prefill 长时间阻塞对尾延迟敏感的 Decode。

---

## 六、理解这些 Shape 后，现代推理系统为什么长成这样就一目了然

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
    D5 --> D6
    D6 --> D7["跨 Token 串行"]

    P5 --> TTFT["TTFT / Input Throughput"]
    D7 --> TPOT["TPOT / Output Throughput"]
~~~

于是今天常见的推理优化就能从同一张图里推出。

KV Cache 用内存换掉过去 Token 的重复计算。

Continuous Batching 把不同 Request 当前的一个 Decode Token 拼起来，让 M≈active requests，不要长期停在 M=1。

Paged / Radix Cache 解决不同 Request、不同 Context Length 与不同生命周期下的 KV ownership。

Chunked Prefill 控制高吞吐大 Prefill 对低延迟 Decode 的阻塞。

Quantization 降低 Weight / Activation / Expert traffic，但它能否兑现仍然取决于 M、M_e、Tiling 与量化开销。

Speculative Decoding 更直接地攻击 Decode 的串行依赖：让一次昂贵 Target Step 尝试 Commit 多个 Token。

PD Disaggregation 则走到更彻底的一步：既然 Prefill 和 Decode 本来就是两类 Workload，就让不同资源池分别优化大 Token Batch 与稳定 TPOT / Continuous Batch，而不是强迫同一套并行与调度参数同时兼顾两边。

所以现代 AI Infra 不是一堆独立名词，而是在围绕同一个核心目标工作：

> **尽可能把 Prefill 与 Decode 各自的 Shape，重新组织到硬件更擅长的位置。**

---

## 七、Ascend 910C 到底是不是 Prefill 吃 Cube、Decode 吃 HBM？必须用 Profile 回答

这里必须把“常见经验”与“已验证事实”分开。

截至本文 Review，没有找到一份公开、可复现的 DeepSeek-V4 + Ascend 910C msprof trace，可以直接给出 Prefill / Decode 的 Cube、Vector、Memory 利用率百分比。因此不能写 Prefill Cube=85%、Decode HBM=90% 这种没有目标环境证据的数字。

但 Ascend 官方 Profiling 已经提供了验证这件事需要的指标。

AI Core profiling 中最值得看的包括：

~~~text
mac_ratio
~~~

Cube / Matrix instruction cycles 占比。

~~~text
vec_ratio
~~~

Vector instruction cycles 占比。

~~~text
mte2_ratio
~~~

Memory-to-AI-Core 数据搬运相关 cycles 占比。

以及官方定义的：

~~~text
memory_bound
=
mte2_ratio
/
max(mac_ratio, vec_ratio)
~~~

官方文档说明：当 <code>memory_bound &gt; 1</code> 时，数据搬运相对主要计算流水已经成为明显压力。

正确的 Prefill / Decode 实验应该固定同一个 checkpoint、quantization、TP / DP / EP、CANN、Attention backend 和 DeepEP 配置。

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

“Decode crossover”至少有三种。第一种是 Dense GEMM crossover；第二种是 MoE crossover，此时真正的横轴是 Expert M_e distribution；第三种才是 End-to-End Decode crossover。

即使 Dense GEMM 已经进入更偏 Compute-heavy 的区域，如果 Attention、C4 Indexer、DeepEP 或 Host Gap 仍然占据 Critical Path，整个 Decode Step 依然不能简单称为 Compute Bound。

所以本文不会写“Batch=32 或 64 就是 Ascend 910C crossover”。更严格的结论是：

> **随着 Continuous Batch 增大，Decode 的 Dense / Expert GEMM arithmetic intensity 会提高，并可能跨过目标硬件的有效 Roofline ridge point；但 DeepSeek-V4 整个 Decode Step 的 crossover 由 Dense、MoE、Compressed Attention、Indexer、通信和 Runtime 共同决定，只能在目标 Ascend 910C 配置上通过 Batch × Context sweep 实测。**

如果整篇只保留一个性能分析框架，可以记住：

~~~text
Phase
  ↓
Query Token Count
  ↓
Dense M
+
Expert M_e
  ↓
Weight Reuse
+
Kernel Occupancy
  ↓
Attention History Traffic
+
Compression / Indexer Traffic
  ↓
Launch / Communication Amortization
  ↓
真实硬件瓶颈
~~~

最终真正值得建立的直觉不是“Prefill 永远 Compute Bound、Decode 永远 Memory Bound”，而是：

> **Prefill 和 Decode 是同一个模型落在两片完全不同的 Workload Shape 空间里；现代推理系统的绝大多数优化，都是在重新组织这两片 Shape 空间。**

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
