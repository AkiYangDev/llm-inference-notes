# AI 推理基础设施工作名词表：SGLang、DeepSeek 与 Ascend 910C 从 Token 到 NPU Kernel

> 这不是一份按字母排序的缩写大全，而是一张面向推理工程工作的知识地图：看到一个陌生词，先判断它属于模型、Runtime、内存、分布式、投机解码，还是算子与 NPU Kernel。

刚开始接触大模型推理基础设施时，很容易遇到一种非常典型的困境：每一个中文字都认识，但一句话连起来就看不懂了。

例如下面这段启动参数：

~~~bash
--tp-size 16
--dp-size 8
--enable-dp-attention
--attention-backend dsv4
--moe-a2a-backend deepep
--deepep-mode auto
--quantization modelslim
--speculative-algorithm DSPARK
--page-size 128
--chunked-prefill-size 131072
~~~

第一次看到时，很容易产生一种错觉：是不是必须先把几十个概念全部背下来，才能开始看 SGLang、DeepSeek 和 Ascend 源码？

真正的问题并不是名词太多，而是这些名词通常被一个一个地学习，却没有被放进同一条执行链里。今天学 TP，明天学 KV Cache，后天学 Scheduler，再过几天学 DeepEP、CANN、Tiling。脑子里有很多点，却没有一张地图。

本文的目标就是建立这张地图。

> **源码与适用范围**：SGLang 源码入口固定到 <code>sgl-project/sglang @ 176dbcb85d3b7737564e4911947035cc0af65f0a</code>，核对日期为 2026-09-21。路径级入口用于帮助定位概念，不代表某个高层语义永远只由一个文件实现。SGLang 仍在快速演进，未来目录、类名和 backend 能力可能变化。

---

## 一、先建立全局地图：一次请求到底经过了什么

一个用户输入：

~~~text
请解释一下 DeepSeek 的 MoE 是怎么工作的？
~~~

从文本变成最终输出，在一个现代推理服务中大致会经过下面这些层：

~~~mermaid
flowchart TD
    U[用户 Prompt] --> T[Tokenizer / Token IDs]
    T --> R[Req / Scheduler]
    R --> SB[ScheduleBatch]
    SB --> FB[ForwardBatch]
    FB --> MR[ModelRunner]

    MR --> M[DeepSeek Model]
    M --> A[Attention]
    M --> E[MoE]
    A --> KV[KV Cache / Prefix Cache]
    E --> ROUTE[Router / Dispatch / Expert / Combine]

    A --> DIST[TP / DP Attention / CP]
    E --> DIST2[EP / DeepEP / All-to-All]
    DIST --> COMM[Process Group / Collective / HCCL]
    DIST2 --> COMM

    M --> L[Logits / Sampling]
    L --> O[Next Token]

    MR --> B[Attention / MoE / Quantization Backend]
    B --> C[CANN / Custom Operator]
    C --> TI[Tiling / Dispatch]
    TI --> K[NPU Kernel]
    K --> NPU[Ascend 910C]
~~~

这张图可以先建立七层心智模型：

| 层级 | 你会看到的词 | 核心问题 |
| --- | --- | --- |
| 模型语义 | Token、Embedding、Q/K/V、Attention、MLA、MoE、Logits | 模型数学上在算什么？ |
| 推理生命周期 | Prefill、Decode、KV Cache、Prefix Cache | 一条请求如何逐步生成？ |
| Runtime | Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner | 哪些请求在这一轮真正执行？ |
| 内存 | Page、KV Pool、Activation、Workspace、Buffer | 状态放在哪里，显存怎么分配？ |
| 分布式 | Rank、Group、TP、DP、DPA、EP、CP、Collective | 谁拥有哪部分数据，谁和谁通信？ |
| 投机解码 | Draft、Verify、Accept、DSpark | 怎样用更便宜的预测换取更少的 Target Decode？ |
| Device 执行 | Backend、CANN、Operator、Tiling、Kernel、Stream | Python 里的 Forward 最终怎样落到 910C？ |

以后碰到陌生词时，第一反应不要是“缩写是什么意思”，而是先问：

> **它属于上面哪一层？它的上游输入是什么，下游又会影响什么？**

这一步比背定义重要得多。

### 快速源码入口索引

下面这张表是本文最适合工作中反复回来查的一部分。第一次阅读可以先浏览，不要求一次看懂；以后日志、启动参数或源码里遇到某个词，可以直接从这里跳进去。

| 核心词 | SGLang 当前入口 | 进去先看什么 |
| --- | --- | --- |
| OpenAI Chat 请求 | [serving_chat.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/entrypoints/openai/serving_chat.py) | ChatCompletionRequest 如何转成内部生成请求 |
| Tokenizer / 输入管理 | [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/tokenizer_manager.py) | 文本、token IDs 和请求消息如何进入调度侧 |
| Req | [schedule_batch.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/schedule_batch.py) | 单请求跨多轮生成保存哪些状态 |
| Scheduler | [scheduler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/scheduler.py) | waiting/running 请求如何被选入下一轮 |
| ScheduleBatch | [schedule_batch.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/schedule_batch.py) | 调度侧如何组织本轮请求、长度和缓存位置 |
| ForwardBatch | [forward_batch_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/forward_batch_info.py) | ScheduleBatch 如何变成 Device Forward 所需 Tensor / Metadata |
| ModelRunner | [model_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/model_runner.py) | eager / graph、backend、model forward 如何汇合 |
| KV Cache / Page | [memory_pool.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/memory_pool.py) | ReqToTokenPool、KVCache、物理 token location |
| Prefix Cache / RadixCache | [radix_cache.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/radix_cache.py) | prefix match、radix tree 与 KV 复用 |
| DeepSeek-V4 Model | [deepseek_v4.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4.py) | Decoder Layer、Attention、MoE、LM Head 的模型边界 |
| Attention Backend 注册 | [attention_registry.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/attention/attention_registry.py) | dsv4 等 backend 如何按平台选择 |
| Ascend DSV4 Attention | [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py) | NPU 上 DSV4 attention 的 metadata、KV 与 dispatch |
| Logits | [logits_processor.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/logits_processor.py) | hidden states 如何变成 vocab logits |
| Sampling | [sampler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/sampler.py) | temperature / top-k / top-p 等如何参与 token 选择 |
| TP / DP / EP 宽度派生 | [runtime_context.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/runtime_context.py) | derive_attention_widths / derive_parallel_widths |
| Process Group / Rank | [parallel_state.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/parallel_state.py) | TP/EP/PP 等 group 如何初始化 |
| DP Attention | [dp_attention.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/dp_attention.py) | attn_dp / attn_tp rank 与数据布局 |
| EP MoE | [ep_moe/layer.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/ep_moe/layer.py) | dispatch → expert runner → combine 的主流程 |
| DeepEP | [deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) / [deepep_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py) | MoE Token 的跨 Rank Dispatch / Combine |
| Ascend MoE Runner | [moe_runner/ascend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/moe_runner/ascend.py) | 通信之后 Expert 计算如何落到 Ascend runner |
| DSpark Worker | [dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) | Draft / Verify / Commit 的 worker 级状态机 |
| DSpark Model | [deepseek_v4_dspark.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4_dspark.py) | DeepSeek-V4 的 DSpark stage / head 如何进入 forward |
| Speculative 元数据 | [spec_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/spec_info.py) | Draft / Verify 之间传递哪些候选与布局信息 |
| NPU Graph | [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) | Decode 图捕获 / replay 与 NPU graph runner |
| ModelSlim | [modelslim.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/quantization/modelslim/modelslim.py) | ModelSlim quantization scheme 怎样接入 Linear / MoE |
| PD 分离 | [disaggregation/prefill.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/prefill.py) / [Ascend transfer_engine.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/ascend/transfer_engine.py) | Prefill / Decode ownership 与 KV/state transfer |

这张表有两个使用原则。

第一，**不要从整个仓库目录开始翻**。带着一个问题跳到入口，再顺着对象和调用链往下追。

第二，**路径只是坐标，不是结论**。例如 Attention 不是只存在于一个 backend 文件里；真正完整的行为还要结合模型层、Runtime metadata、backend 和底层算子一起看。

---

## 二、模型语义层：Token、Tensor、Attention、MoE 到 Logits

大模型最上层的问题是：模型究竟在处理什么？

答案不是“文字”，而是 Tensor。

用户输入先经过 Tokenizer。文本被切分、映射为 Token ID，然后通过 Embedding 进入连续向量空间。假设某个模型的 hidden size 是 H，那么一个 Token 可以抽象成长度为 H 的向量；多 Token 共同形成 hidden states。

这里最重要的不是背某个 shape，而是形成三个源码阅读问题：

1. 这个 Tensor 的**语义**是什么？
2. 它的 **shape** 是什么？
3. 它的 **dtype** 是什么？

同样是二维 Tensor，它可能是 hidden states、weight、KV cache、temporary buffer；shape 相同不代表含义相同。

### Hidden States、Residual 与 Norm

**Hidden States** 是当前 Token 在模型某一层中的内部表示。Embedding 产生初始 hidden states，随后每个 Decoder Layer 对它继续变换。

Transformer 中还会反复看到 **Residual Connection** 和 **RMSNorm**。Residual 可以粗略理解成把 block 的新结果加回原表示；RMSNorm 则控制数值尺度。以后研究算子融合时，会进一步遇到 Add + RMSNorm、Residual + Norm 等融合路径。

### Attention、Q/K/V 与 Head

Attention 解决的是：当前 Token 应该从历史上下文的哪些位置读取信息。

经典形式里，输入 X 经过线性投影得到 Q、K、V：

~~~text
Q = XWq
K = XWk
V = XWv
~~~

随后通过 Q 与 K 的相关性决定怎样聚合 V。

Multi-Head Attention 的意义，是让不同投影子空间并行建模不同关联模式。需要避免一个常见误解：不能把某个 Head 稳定地等价为“语法 Head”“实体 Head”之类的人类可解释功能；这些最多只是帮助入门的例子。

在 DeepSeek 语境中还会遇到 **MLA**。它的重要工程意义之一，是改变 K/V 的表示与缓存方式，从而降低传统 Attention 下 KV Cache 的压力。理解 MLA 的重点不是先背全部公式，而是先建立这条因果关系：

~~~text
Attention 结构
   ↓
决定历史状态怎么保存
   ↓
决定 KV Cache Layout
   ↓
影响显存与分布式布局
~~~

对于 DeepSeek-V4，最终仍要回到当前模型实现与 dsv4 backend，因为 V4 还包含更加特化的 attention、压缩缓存和滑窗路径。对应源码起点是 [deepseek_v4.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4.py)，NPU backend 则从 [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py) 继续往下追。

### MLP、GEMM 与 MoE

Transformer 的另一大块是 MLP / FFN。Linear 层最终大量落成 GEMM，也就是通用矩阵乘法。因此你以后看到 BF16 GEMM、INT8 GEMM、Grouped GEMM、Expert GEMM，本质上都在讨论不同精度、不同布局和不同 batch 形态下的矩阵乘法。

MoE 把单一 FFN 进一步拆成多个 Expert：

~~~text
Token
  ↓
Router
  ↓
Top-K Experts
  ↓
Dispatch
  ↓
Expert Compute
  ↓
Combine
~~~

这条链一旦跨设备，就同时变成模型问题与通信问题。Router 决定 Token 去哪里，EP 决定哪些 Rank 持有哪些 Expert，DeepEP 等通信 backend 负责把 Token 送过去并把结果收回来，Expert Runner 再负责真正的 GEMM。

这也是为什么看 DeepSeek MoE 源码时，不能只盯着 Expert Linear 层：真正的性能瓶颈可能出现在 Router、Permutation、All-to-All、Dispatch/Combine 或 Expert GEMM 任意一段。

### Logits 与 Sampling

模型最后不是直接输出文字，而是输出 **Logits**：对词表中候选 Token 的原始分数。

随后 Sampling 决定真正选哪个 Token。常见词包括 Greedy、Temperature、Top-K、Top-P。

因此最简化的生成链是：

~~~text
Hidden States
    ↓
LM Head / Logits Processor
    ↓
Logits
    ↓
Sampler
    ↓
Next Token
~~~

SGLang 当前分别可以从 [logits_processor.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/logits_processor.py) 和 [sampler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/sampler.py) 进入。

这一层真正应该记住的不是几十个公式，而是：

> **模型结构决定 Tensor 语义；Tensor 语义决定后面的缓存、并行和 Kernel 需要处理什么数据。**

---

## 三、推理生命周期与 Runtime：Prefill、Decode、KV Cache、Scheduler

模型会算，并不代表已经理解“推理服务”。

真正的在线推理系统还要解决：同时来了几百个请求，谁先跑？一轮跑多少 Token？KV Cache 不够怎么办？Prefill 会不会把 Decode 堵死？某个请求结束后能不能立即把空位让给新请求？

这就是 Runtime 的世界。

### Prefill 与 Decode

一条请求通常经历两个不同阶段。

**Prefill** 处理输入 Prompt。一次可能计算很多 Token，矩阵规模更大，硬件计算单元通常更容易被利用。

**Decode** 在已有上下文基础上继续生成新 Token。每轮新增 Token 很少，却需要读取大量历史状态，因此更容易受到 KV Cache、Memory Bandwidth、小 GEMM、Kernel Launch 和通信延迟影响。

所以性能讨论至少要区分：

- TTFT：Time To First Token；
- TPOT：Time Per Output Token；
- Throughput：单位时间整体处理多少 Token。

“吞吐高”并不自动意味着“单请求体验好”。

### KV Cache 与 Page

如果生成第 N+1 个 Token 时把前 N 个 Token 的 K/V 全部重新计算一遍，Decode 会产生巨大重复开销。因此推理系统保存历史 Attention 状态，这就是 KV Cache。

上下文越长、并发越高，KV Cache 越大：

~~~text
Context Length ↑
      ↓
每个请求 KV ↑

Concurrency ↑
      ↓
同时存在的请求 KV ↑

最终：
HBM 压力 ↑
~~~

现代 Runtime 通常不会给每个 Request 预留一大块固定连续空间，而是把 KV Pool 按 Page / Block 管理，让逻辑 Sequence 和物理位置分离。

SGLang 当前 [memory_pool.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/memory_pool.py) 中可以直接看到 **ReqToTokenPool** 与 **KVCache** 这样的结构：前者把请求映射到 token location，后者管理真正的缓存存储抽象。

这类代码之所以不像 Transformer 论文，是因为你已经不在“模型数学层”，而是在“操作系统式的内存管理层”。

### Prefix Cache 与 RadixCache

如果两个请求共享长前缀，第二个请求没有必要把相同前缀重新 Prefill 一次。

SGLang 使用 Prefix Cache 复用已有 KV。当前源码中非常直接的入口是 [radix_cache.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/radix_cache.py) 的 **RadixCache**。

这里要区分两个容易混淆的表达：

- **RadixAttention** 更接近 SGLang 的整体 prefix reuse 设计语境；
- **RadixCache** 是当前 Runtime 中可以直接看到的 prefix-cache 数据结构。

所以工作中说“去看 Radix”时，首先要明确是在讨论高层机制，还是当前具体缓存实现。

### Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner

这五个词是阅读 SGLang 最值得优先搞清楚的一组。

**Req** 保存一条请求跨越多轮生成的状态，不只是 Prompt。它会携带长度、输出 Token、Sampling 参数、缓存状态、结束条件以及各种 Runtime metadata。

**Scheduler** 每一轮决定谁真正进入计算。它面对的是 Waiting Requests、Running Requests、Token Budget、KV Capacity、Prefill / Decode 状态以及 Speculative 状态。

**ScheduleBatch** 是调度侧对“这一轮跑什么”的表达。

**ForwardBatch** 则更接近模型执行侧：输入 Tensor、positions、seq_lens、cache location、attention metadata 等已经准备给 ModelRunner 消费。

当前源码非常适合用下面这条链来读：

~~~mermaid
flowchart LR
    A[Req] --> B[Scheduler]
    B --> C[ScheduleBatch]
    C --> D[ForwardBatch]
    D --> E[ModelRunner]
    E --> F[Model Forward]
    F --> G[Logits / Sampling]
    G --> H[更新 Req 状态]
    H --> B
~~~

对应入口：

- Req / ScheduleBatch：[schedule_batch.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/schedule_batch.py)
- Scheduler：[scheduler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/scheduler.py)
- ForwardBatch：[forward_batch_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/forward_batch_info.py)
- ModelRunner：[model_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/model_runner.py)

这一组概念的关键不是记住类名，而是理解**状态边界**：

> Req 是跨轮次生命周期；ScheduleBatch 是本轮调度视图；ForwardBatch 是本轮 Device Forward 视图。

这对后面读投机解码尤其重要，因为 Draft / Verify / Accept 会让“逻辑长度”“已分配 KV”“已正式提交 KV”出现不同时间点。

### Continuous Batching、Chunked Prefill、Graph 与 Overlap

在线推理不会等一个固定 Batch 的所有请求一起结束，而会不断移出完成请求、加入新请求，这就是 Continuous Batching。

长 Prompt 如果一次 Prefill 完成，可能占用大量预算并阻塞 Decode，因此 Runtime 可以把 Prefill 拆成多个 Chunk；这就是 Chunked Prefill。它是调度策略，不是模型结构。

Decode 期间大量小 Kernel 又会放大 Host Launch Overhead，因此还会出现 Graph Capture / Replay。Ascend 路径当前可以从 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) 进入。

最后是 **Overlap**：把原本串行的计算、通信、结果处理或下一轮准备重叠起来。Overlap 并不天然等于更快，因为两个任务可能竞争同一资源，也可能增加 Buffer 和同步复杂度。

所以 Runtime 优化的本质经常不是“一个算子更快”，而是：

> **让更多有用工作在正确的时间进入 Device，同时避免内存、通信和同步成为新的瓶颈。**

---

## 四、分布式推理：Rank、TP、DP Attention、EP 与 DeepEP

模型大到一张卡放不下，或者吞吐目标高到单卡不够，就必须进入分布式世界。

这一部分最容易学乱，因为 TP、DP、EP、CP、Group、Rank、AllReduce、DeepEP、HCCL 经常出现在同一句话里，但它们并不属于同一个层级。

### Rank、World Size 与 Process Group

先不要把 Rank 等价成“一张 NPU”。

更加准确的理解是：

> **Rank 是某个分布式进程在某个 Process Group 中的逻辑编号。**

一进程一卡时，Global Rank 和 Device 编号可能看起来一一对应，但概念上仍不同。

同一个进程还可能同时属于多个 Group，并在每个 Group 中拥有不同的局部 Rank。例如：

~~~text
Global Rank = 6

TP Group = [4, 5, 6, 7]
→ TP Rank = 2

DP Group = [2, 6, 10, 14]
→ DP Rank = 1
~~~

所以源码里看到 TP rank、DP rank、EP rank、Draft rank、Verify rank 时，不要把它们理解成五套硬件，而是理解为：

> **同一个进程在不同通信坐标系中的身份。**

当前 SGLang 的 group 初始化与 parallel state 可以从 [parallel_state.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/parallel_state.py) 进入。

### TP：Tensor Parallelism

TP 把同一个 Layer 内的大 Tensor / GEMM 拆到多个 Rank 共同完成。

例如 Y = XW，W 可以按某个维度切成多个 shard。不同 TP Rank 保存不同权重切片，完成局部计算后再通过 Collective 恢复下一步需要的数据语义。

因此 TP 的核心是：

> **同一个请求、同一个 Layer，需要多个 Rank 协作。**

也正因为如此，它天然会带来 AllReduce、AllGather、ReduceScatter 等通信。

### DP 与 DP Attention

经典 Data Parallelism 更接近：

> **模型 Replica 处理不同数据。**

而 SGLang 中的 **DP Attention / DPA** 不能简单理解成“普通 DP 的另一个名字”。

当前并行宽度派生集中在 [runtime_context.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/runtime_context.py)。其中 Attention 维度会根据配置派生出：

~~~text
attn_dp_size
attn_cp_size
attn_tp_size
~~~

在开启 DP Attention 时，dp_size 会进入 Attention 的数据并行布局；Attention TP 宽度则继续由总 TP world、DP 和 CP 共同决定。

因此类似：

~~~text
tp_size = 16
dp_size = 8
enable_dp_attention = true
~~~

不能直接翻译成“16 份 TP 再乘 8 份模型副本”。

真正应该继续问：

- Attention DP Group 怎么划？
- 每个 DP replica 内还有多少-way Attention TP？
- 每个 Rank 拥有哪些请求和 KV？
- MoE 阶段又怎样重新解释同一批 Rank？

DP Attention 的 per-process rank / metadata 入口可以继续看 [dp_attention.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/dp_attention.py)。

### EP：Expert Parallelism

MoE 中如果每张卡都保存全部 Expert，权重开销会非常大。EP 把不同 Expert 分布到不同 Rank：

~~~text
Rank 0 → Experts A/B
Rank 1 → Experts C/D
Rank 2 → Experts E/F
...
~~~

Router 选中某个远端 Expert 后，Token 必须先被发送到持有该 Expert 的 Rank，完成 Expert Compute 后结果再返回。

于是 MoE Forward 从模型视角的：

~~~text
Router → Expert
~~~

变成系统视角的：

~~~text
Router
  ↓
Top-K
  ↓
Dispatch
  ↓
Expert Runner
  ↓
Combine
~~~

SGLang 当前 EP 主流程可以从 [ep_moe/layer.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/ep_moe/layer.py) 开始。

### DeepEP：通信 Backend，不是 Expert GEMM

**DeepEP** 最容易被初学者误解成“一个 MoE 算子”。

更准确地说，它首先解决的是：

> **Expert Parallel 场景下 Token 的 Dispatch / Combine 通信。**

SGLang 当前有 [deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) 和 [deepep_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py) 等 dispatcher。

而 Expert 真正如何计算，则是另一层 Runner；Ascend 路径可以继续看 [moe_runner/ascend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/moe_runner/ascend.py)。

所以应该把两件事拆开：

~~~text
DeepEP
→ Token 怎么跨 Rank 到 Expert

Ascend MoE Runner
→ 到了本 Rank 后 Expert GEMM 怎么算
~~~

这一区分非常重要。很多性能问题不是“GEMM 慢”，而是 Dispatch / Combine 比 Expert Compute 更贵；反过来也可能通信已经隐藏得很好，瓶颈转移到 Expert GEMM。

### Collective：AllReduce、AllGather、ReduceScatter、All-to-All

这些是通信原语，不是并行策略。

- **AllReduce**：所有 Rank 贡献局部值，归约后所有 Rank 都得到结果；
- **AllGather**：每个 Rank 持有一部分，最终所有 Rank 得到完整集合；
- **ReduceScatter**：先归约，再把结果切分给不同 Rank；
- **All-to-All**：每个 Rank 都可能向其他 Rank 发送不同数据，MoE Dispatch 很典型。

从很多常见算法的逻辑上，可以把 AllReduce 理解成 ReduceScatter + AllGather 两阶段，但这是一种语义/算法模型，不意味着底层实现一定机械调用两个 API。

Ascend 环境中还会看到 **HCCL**。它属于 Collective Communication Library 层，是 AllReduce、AllGather、All-to-All 等通信真正落地的重要基础设施之一。

**RDMA** 则更下层，它描述远程内存访问/网络数据搬运能力，不是 TP 或 EP 本身。

所以可以这样分层：

~~~text
并行策略：
TP / DP / EP / CP

    ↓ 需要数据交换

通信原语：
AllReduce / AllGather / All-to-All

    ↓ 由通信栈执行

HCCL / 专用 MoE backend / 网络传输能力
~~~

### PP、CP、SP

**PP：Pipeline Parallelism** 沿模型深度切 Layer，不同 Stage 负责不同层。

**CP：Context Parallelism** 沿 Sequence / Context 维切分上下文。Attention 的 Query 可能仍需要看到跨 Rank 的历史 KV，因此 CP 不是简单“把 Token 数组切几段”。

**SP：Sequence Parallelism** 同样涉及 sequence 维，但经典实现往往服务于 TP 场景下 activation 的切分与内存优化，和 CP 的整体上下文分布式语义并不等价。

所以“CP 和 SP 都切 Sequence”只是一层表象，不能作为完整定义。

真正读源码时，要以当前框架对 Group、Tensor Layout 和 Collective 的定义为准。

---

## 五、投机解码：Draft、Verify、Accept 与 DSpark

普通自回归 Decode 的问题是：Target Model 每做一次昂贵 Forward，只向前推进很少的新 Token。

Speculative Decoding 的核心思想是：

> **先用更便宜的路径提出多个候选，再让 Target Model 一次验证。**

最简化状态机是：

~~~mermaid
flowchart LR
    A[Committed History] --> B[Draft]
    B --> C[Candidate Tokens]
    C --> D[Target Verify]
    D --> E{Accept?}
    E -->|接受前缀| F[Commit Token / KV State]
    E -->|拒绝后续| G[Rollback / Reclaim]
    F --> B
    G --> B
~~~

### Draft 是角色，不一定是独立小模型

这是非常重要的一点。

**Draft** 首先描述“谁负责提出候选”，并不必然意味着另起一套完整小模型。

不同实现可能是：

- 独立 Draft Model；
- MTP / NextN Head；
- EAGLE 风格 Draft；
- DSpark Head；
- N-gram 等非神经路径。

当前 DeepSeek-V4 文档里已经存在 checkpoint 自带 DSpark head 的部署方式；与此同时，Ascend 某些 CI / 模型配置也仍会显式传入 speculative draft model path。也就是说，不能把一个部署形态写成算法定义。

最稳妥的心智模型是：

> **Draft 是功能角色；“它是不是另一套权重、另一套模型实例、同一 checkpoint 的额外 head”，要看当前算法与 checkpoint。**

### Verify 与 Accept

Draft 只是假设未来。

Target Verify 才决定候选是否符合 Target Model 的分布/选择规则。

随后 **Accept** 确定最终哪些 Token 真正进入历史。

真正困难的是：Accept 不只是更新 token_ids，还要决定哪些状态正式成为历史，例如：

- Sequence Length；
- Position；
- KV Cache；
- Draft/Verify Metadata；
- 下一轮 Draft State。

因此投机解码代码经常围绕一个核心问题：

> **哪些状态只是 speculative future，哪些状态已经 committed？**

这也是理解 rollback、compact、commit、reclaim 等词的基础。

### Acceptance Rate 高不等于一定更快

投机解码收益取决于：

~~~text
有效接受的 Token 数
÷
Draft + Verify + KV + Scheduler + Communication 的总成本
~~~

所以高 Acceptance Rate 只是有利条件之一。

如果 Draft 很贵、Verify window 过大、KV 搬运增加、DP/TP 同步开销明显，投机解码可能并不比普通 Decode 更快。

### DSpark 在源码里从哪里看

当前入口建议按下面顺序：

1. [spec_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/spec_info.py)：先理解 speculative metadata 的数据结构；
2. [dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)：看 worker 如何组织 Draft / Target / Verify；
3. [deepseek_v4_dspark.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4_dspark.py)：看 V4 DSpark 模型结构怎样进入 forward。

到了多卡之后，问题还会继续升级：

- Draft 使用什么 TP/DP/EP Layout？
- Verify 使用什么 Group？
- Accept Decision 要向哪些 Rank 同步？
- KV ownership 在哪个坐标系里定义？
- 如果 Draft 与 Target 的并行布局不同，谁负责 bridge？

因此像 “TP != DP” 的支持问题，本质绝不只是删除一条 assert。

真正的完成标准应该是：

~~~text
允许新的配置
+
所有 Tensor Layout 正确
+
所有 Process Group 正确
+
Collective 语义正确
+
KV / Sequence State 正确
+
输出结果正确
~~~

这也是为什么分布式 Speculative Decoding 是典型的 AI Infra 问题：算法、Runtime、内存和通信同时耦合。

---

## 六、Ascend 910C 执行层：Backend、CANN、Tiling、Kernel 与量化

到这里，我们还主要在 Python / Runtime / Tensor 层。

真正把计算执行到 Ascend 910C，还需要继续往下追。

### Backend：同一高层语义的具体实现选择

SGLang 中经常看到：

- Attention Backend；
- MoE A2A Backend；
- MoE Runner Backend；
- Quantization Backend。

Backend 不应该简单理解成“另一种模型算法”。

它更像：

> **同一个高层语义，在特定硬件、模型结构、精度和运行模式下选择哪套实现。**

例如 Attention 是模型语义；而 <code>--attention-backend dsv4</code> 决定 DeepSeek-V4 Attention 在当前平台走哪套 Runtime / Operator 路径。

当前注册入口在 [attention_registry.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/attention/attention_registry.py)，Ascend DSV4 具体实现继续进入 [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)。

### 从 Operator 到 Kernel 不是严格一对一

为了建立直觉，可以画成：

~~~text
Model Forward
    ↓
Backend
    ↓
Operator
    ↓
Tiling / Dispatch
    ↓
NPU Kernel
    ↓
Ascend 910C
~~~

但一定要记住：

> **这是一张认知图，不代表一个 Operator 永远对应一个 Kernel。**

现实里可能出现：

- 一个高层 Operator 启动多个 Kernel；
- 多个高层操作被 Fuse；
- 同一个 Operator 因 Shape / DType / Mode 选择不同 Kernel；
- 某些路径还包含 Host 侧预处理、Workspace 和额外通信。

所以性能分析不能停在“调用了 Attention 算子”，还要继续问：

- 最后走了哪个 dispatch branch？
- Shape / dtype 是什么？
- Tiling 怎么选？
- Kernel 是 compute-bound 还是 memory-bound？
- 有没有额外的数据布局转换？

### CANN、Tiling 与 TilingKey

可以把 CANN 理解成 Ascend AI 软件栈中的关键基础设施集合，而不是单个 Python 包。

对于自定义/高性能算子，常见的一个思路是：

~~~text
Host
  ↓
读取 Shape / Attribute
  ↓
计算 Tiling
  ↓
生成 Kernel 所需参数
  ↓
Device Kernel
~~~

NPU 片上高速存储无法容纳任意大的 Tensor，所以大数据必须按 Tile 搬入、计算、写回。

Tiling 需要回答：

- 一块处理多少数据？
- 每个 Core 处理哪一段？
- 循环多少次？
- Local Buffer 怎么使用？
- 计算和搬运如何重叠？

**TilingKey** 则可以用于在不同 Shape、dtype、mode 下选择不同 Kernel 分支。

于是源码里看到某个算子名称时，不要默认“它永远是一条固定实现路径”。

### Stream、Graph 与异步执行

Host 发起 Kernel 后，任务通常进入 Device Stream 异步执行。Host 代码返回，并不意味着 Device 已经计算完成。

这就是为什么性能 Profiling 中经常需要区分：

- Host API 时间；
- Kernel 执行时间；
- Stream Synchronization；
- Compute / Communication overlap。

NPU Graph 则进一步减少重复 Launch / Python Host 调度成本。SGLang 当前 Ascend 入口可以从 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) 看起。

### BF16、W8A8 与 ModelSlim

**BF16** 是 16-bit 浮点格式。相比 FP32，它减少存储与内存传输开销；在硬件和 Kernel 支持良好的情况下，也可能获得更高吞吐。但“bit 更少”并不等价为“实际一定按比例更快”。

**W8A8** 最直观表示：

~~~text
Weight 8-bit
Activation 8-bit
~~~

但这远远不足以定义一套完整量化方案。

还必须继续问：

- Weight 是 per-tensor、per-channel 还是 per-group？
- Activation 是 static 还是 dynamic？
- 是 per-token 还是其他粒度？
- 对称还是非对称？
- Scale / zero point 怎么存？
- 哪些层排除量化？
- Linear 与 MoE 使用什么 Kernel？

因此“两个模型都叫 W8A8”不代表运行时格式和性能行为相同。

SGLang 当前 ModelSlim 入口在 [modelslim.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/quantization/modelslim/modelslim.py)，相关 scheme 继续位于同目录下。

这也是量化排障时最值得建立的习惯：

> 不要停在“模型是 W8A8”，要继续追到“这一个 Layer 的 Weight/Activation 具体是什么格式，最终被哪个 Kernel 消费”。

### PD Disaggregation

Prefill 和 Decode 的硬件特征不同，因此生产系统可以把它们拆到不同实例。

这叫 **PD Disaggregation**。

一旦分离，Prefill 侧产生的 KV / State 必须被 Decode 侧接管，于是问题从“调度”升级成：

- KV ownership；
- 远程传输；
- 目标侧预分配；
- handoff metadata；
- 网络与内存注册；
- 首轮 Decode 如何恢复执行状态。

SGLang 当前通用 Prefill 侧入口可以从 [disaggregation/prefill.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/prefill.py) 看，Ascend transfer 路径则有 [disaggregation/ascend/transfer_engine.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/ascend/transfer_engine.py)。

这很好地说明了 AI Infra 的一个共同规律：

> **一个优化只要改变“数据归谁所有”，就很快会变成分布式状态管理问题。**

---

## 七、怎样把这篇文章真正当成工作导航页

如果只是把上面的定义全部背下来，这篇文章仍然没有发挥最大价值。

更有效的使用方式，是把“名词 → 源码 → 专题文章”串成一条固定阅读路径。

### 从一次请求开始

如果你想知道“一次 Chat Completion 到底怎么跑到 NPU”：

1. 先读 [SGLang 推理执行链路解析](../sglang/sglang-ascend-request-lifecycle.md)；
2. 再读 [Scheduler 源码解析](../sglang/sglang-scheduler-source-analysis.md)；
3. 再读 [ModelRunner 源码解析](../sglang/sglang-modelrunner-source-analysis.md)。

这三篇解决的是：

~~~text
请求怎么进来
→ 怎么被调度
→ 怎么真正进入模型 Forward
~~~

### 如果你在看 Attention / KV Cache

继续读：

- [DeepSeek-V4 Attention 源码解析](../sglang/deepseek-v4-attention-source-analysis.md)

重点会从：

~~~text
ForwardBatch
→ Query
→ KV write
→ 历史 KV 读取
→ Ascend DSV4 Backend
~~~

继续向下。

### 如果你在看 MoE / TP / DP / EP

按这个顺序：

1. [DeepSeek-V4 MoE 推理源码解析](../sglang/deepseek-v4-moe-source-analysis.md)
2. [DeepSeek-V4 分布式推理源码解析](../sglang/deepseek-v4-distributed-parallel-source-analysis.md)

前者解决 Router、Dispatch、Expert、Combine。

后者重点解决：

~~~text
同一个 TP world
在 Attention 阶段怎样解释
在 MoE 阶段又怎样重新解释
~~~

也就是从“知道 TP/DP/EP 的定义”，走到真正能分析 parallel layout。

### 如果你在看 DSpark / 投机解码

推荐顺序：

1. [DeepSeek-V4 Speculative Decoding 总链路](../speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)
2. [DeepSeek-V4 DSpark Parallel Layout](../distributed/deepseek-v4-dspark-parallel-layout-analysis.md)
3. [SGLang Speculative Scheduler](../speculative-decoding/sglang-speculative-decoding-scheduler-source-analysis.md)
4. [Accept Decision](../speculative-decoding/deepseek-v4-accept-decision-source-analysis.md)
5. [KV Commit / Compact / Rollback](../speculative-decoding/deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md)
6. [多卡一致性](../distributed/deepseek-v4-speculative-distributed-consistency-source-analysis.md)
7. [Speculative 性能模型](../performance/deepseek-v4-speculative-performance-analysis.md)

如果需要进一步理解多 Token Draft 与 Verify Tree，再进入：

- [MTP / NextN](../speculative-decoding/deepseek-v4-mtp-speculative-decoding-source-analysis.md)
- [EAGLE Draft Tree](../speculative-decoding/deepseek-v4-eagle-draft-tree-source-analysis.md)
- [Tree Attention](../speculative-decoding/deepseek-v4-tree-attention-source-analysis.md)

### 如果你在看 PD 分离

继续读：

- [DeepSeek-V4 Prefill / Decode Disaggregation 源码解析](../sglang/deepseek-v4-pd-disaggregation-source-analysis.md)

重点从“Prefill 和 Decode 为什么不同”进入“KV / state ownership 怎样跨实例迁移”。

### 最终应该形成的因果链

真正值得长期记住的，不是几十个孤立缩写，而是下面这条因果链：

~~~mermaid
flowchart TD
    A[Model Architecture] --> B[Tensor / Attention / MoE]
    B --> C[KV / Weight / Activation Layout]
    C --> D[Memory Consumption]
    D --> E[Batch / Scheduler Policy]
    E --> F[TP / DP / EP / CP]
    F --> G[Collective / DeepEP / HCCL]
    G --> H[Runtime Backend]
    H --> I[Operator / Tiling]
    I --> J[Kernel]
    J --> K[Ascend 910C]
    K --> L[Latency / Throughput / Memory]
~~~

以后看到一个陌生词，可以用固定的五个问题去拆：

1. **它是什么语义？**
2. **它的数据 shape / dtype / ownership 是什么？**
3. **它处在哪个 Runtime 边界？**
4. **它需要哪些 Process Group 或通信？**
5. **最后由哪个 Backend / Operator / Kernel 消费？**

例如看到 **seq_lens**，不要只问“这个数组是什么”，而要继续问它处于 Req、ScheduleBatch 还是 ForwardBatch 的哪一层，以及它如何影响 position 与 KV。

看到 **attn_tp_group**，不要只问“TP 是什么”，而要问 Group 中包含哪些 Global Rank、当前 Attention DP/TP 坐标系是什么。

看到 **DeepEP**，先判断它是 Dispatch/Combine 通信层，而不是 Expert GEMM 本身。

看到 **TilingKey**，先判断你已经进入 Operator → Kernel Dispatch 层。

看到 **DSPARK**，先判断你进入了 Speculative Decoding 状态机，再继续追 Draft、Verify、Accept 和 Commit 边界。

如果这些问题逐渐形成条件反射，SGLang、DeepSeek 与 Ascend 910C 源码就不会再像几百个互不相干的缩写。

它们只是同一条推理执行链在不同层次上的名字。

---

## 一页速查

最后用一张最短的表收尾。

| 看到这个词 | 第一反应 |
| --- | --- |
| Token / Hidden States | 模型输入与中间表示 |
| Q / K / V / Attention | Token 怎样读取上下文 |
| MLA | Attention / KV 表示方式 |
| MoE / Router / Expert | Token 怎样选择稀疏 FFN |
| Logits / Sampler | 怎样决定下一个 Token |
| Prefill / Decode | 当前处于哪种推理阶段 |
| KV Cache / Page | 历史 Attention 状态放在哪里 |
| RadixCache | Prefix KV 怎样复用 |
| Req | 单请求跨轮次状态 |
| Scheduler | 下一轮谁能跑 |
| ScheduleBatch | 调度侧的本轮计算计划 |
| ForwardBatch | Device Forward 的执行视图 |
| ModelRunner | Runtime 到模型执行的关键边界 |
| Rank / Group | 当前进程在什么通信坐标系 |
| TP | 一个 Layer 的 Tensor/GEMM 怎样分片 |
| DP Attention | Attention 请求/KV 怎样按 DP 布局 |
| EP | Expert 怎样分布在不同 Rank |
| DeepEP | MoE Token 怎样 Dispatch / Combine |
| AllReduce / A2A | 需要执行什么 Collective |
| HCCL | Ascend Collective 通信基础库 |
| RDMA | 跨机器数据怎么高效搬 |
| Draft / Verify / Accept | 投机未来怎样收敛成正式历史 |
| DSpark | DeepSeek-V4 speculative 路径 |
| Backend | 当前语义选择哪套具体实现 |
| CANN / Operator | 上层 Forward 如何进入 Ascend 计算栈 |
| Tiling | 大 Tensor 如何切成 Device 可执行任务 |
| Kernel | 真正在 Device 上跑的计算程序 |
| BF16 / W8A8 | 数据精度与量化格式 |
| ModelSlim | Ascend 量化格式/方法如何接入 Runtime |
| PD Disaggregation | Prefill 与 Decode 的状态如何跨实例交接 |

如果只记一句话，可以记：

> **Model → Runtime → Memory → Distributed → Communication → Operator → Kernel → Hardware。**

之后所有新名词，都尝试把它放回这条链里。
