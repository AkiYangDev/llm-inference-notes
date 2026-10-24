# AI 推理基础设施名词表：从 Token 到 Ascend Kernel

读推理日志时，很容易在同一屏里遇到 Token、KV Cache、TP、Backend 和 Kernel。它们分别描述模型数据、缓存、分工和执行方式，放回一次请求里就容易分清。

这份名词表沿请求的执行顺序组织：文本怎样变成 Token，模型怎样生成下一个 Token，缓存与调度怎样支持持续生成，再到多卡通信和 Ascend 执行。查词可以直接用速查表；第一次接触这些概念，可以从“一次请求的全局地图”顺着读。源码链接固定到同一个版本，便于把定义与实际对象对上。

> **源码与适用范围**：SGLang 源码入口固定到 <code>sgl-project/sglang @ 176dbcb85d3b7737564e4911947035cc0af65f0a</code>，核对日期为 2026-09-21。路径用于定位相应模块；完整调用关系在各专题文章中展开。

## 速查：按当前问题找术语

点击术语可以跳到对应解释；同一组内的词条放在一起，便于比较它们的职责。

| 看到这个词 | 第一反应 |
| --- | --- |
| [Token / Hidden States](#hidden-statesresidual-norm) | 模型输入与中间表示 |
| [Q / K / V / Attention](#attentionqkv-head) | Token 怎样读取上下文 |
| [MLA](#attentionqkv-head) | Attention / KV 表示方式 |
| [MoE / Router / Expert](#mlpgemm-moe) | Token 怎样选择稀疏 FFN |
| [Logits / Sampler](#logits-sampling) | 怎样决定下一个 Token |
| [Prefill / Decode](#prefill-decode) | 当前处于哪种推理阶段 |
| [KV Cache / Page](#kv-cache-page) | 历史 Attention 状态放在哪里 |
| [RadixCache](#prefix-cache-radixcache) | Prefix KV 怎样复用 |
| [Req](#reqschedulerschedulebatchforwardbatchmodelrunner) | 单请求跨轮次状态 |
| [Scheduler](#reqschedulerschedulebatchforwardbatchmodelrunner) | 下一轮谁能跑 |
| [ScheduleBatch](#reqschedulerschedulebatchforwardbatchmodelrunner) | 调度侧的本轮计算计划 |
| [ForwardBatch](#reqschedulerschedulebatchforwardbatchmodelrunner) | Device Forward 的执行视图 |
| [ModelRunner](#reqschedulerschedulebatchforwardbatchmodelrunner) | Runtime 到模型执行的关键边界 |
| [Rank / Group](#rankworld-size-process-group) | 当前进程在什么通信坐标系 |
| [TP](#tptensor-parallelism) | 一个 Layer 的 Tensor/GEMM 怎样分片 |
| [DP Attention](#dp-dp-attention) | Attention 请求/KV 怎样按 DP 布局 |
| [EP](#epexpert-parallelism) | Expert 怎样分布在不同 Rank |
| [DeepEP](#deepep) | MoE Token 怎样 Dispatch / Combine |
| [AllReduce / A2A](#collectiveallreduceallgatherreducescatterall-to-all) | 需要执行什么 Collective |
| [HCCL](#collectiveallreduceallgatherreducescatterall-to-all) | Ascend Collective 通信基础库 |
| [RDMA](#collectiveallreduceallgatherreducescatterall-to-all) | 跨机器数据怎么高效搬 |
| [Draft / Verify / Accept](#verify-accept) | 投机未来怎样收敛成正式历史 |
| [DSpark](#dspark) | DeepSeek-V4 speculative 路径 |
| [Backend](#backend) | 当前语义选择哪套具体实现 |
| [CANN / Operator](#operator-kernel) | 上层 Forward 如何进入 Ascend 计算栈 |
| [Tiling](#canntiling-tilingkey) | 大 Tensor 如何切成 Device 可执行任务 |
| [Kernel](#operator-kernel) | 在 Device 上跑的计算程序 |
| [BF16 / W8A8](#bf16w8a8-modelslim) | 数据精度与量化格式 |
| [ModelSlim](#bf16w8a8-modelslim) | Ascend 量化格式/方法如何接入 Runtime |
| [PD Disaggregation](#pd-disaggregation) | Prefill 与 Decode 的状态如何跨实例交接 |

## 一、一次请求的全局地图

一个用户输入：`请解释一下 DeepSeek 的 MoE 是怎么工作的？`

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

图中的术语可以按职责分成七层。比如 KV Cache 容量不够，先看缓存与内存管理；日志里的通信组不符合预期，再看分布式布局：

| 层级 | 你会看到的词 | 核心问题 |
| --- | --- | --- |
| 模型语义 | Token、Embedding、Q/K/V、Attention、MLA、MoE、Logits | 模型数学上在算什么？ |
| 推理生命周期 | Prefill、Decode、KV Cache、Prefix Cache | 一条请求如何逐步生成？ |
| Runtime | Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner | 哪些请求在这一轮执行？ |
| 内存 | Page、KV Pool、Activation、Workspace、Buffer | 状态放在哪里，显存怎么分配？ |
| 分布式 | Rank、Group、TP、DP、DPA、EP、CP、Collective | 谁拥有哪部分数据，谁和谁通信？ |
| 投机解码 | Draft、Verify、Accept、DSpark | 怎样用更便宜的预测换取更少的 Target Decode？ |
| Device 执行 | Backend、CANN、Operator、Tiling、Kernel、Stream | Python 里的 Forward 最终怎样落到 910C？ |

遇到陌生术语，可以先确定它在系统中的位置：

> **它属于上面哪一层？它的上游输入是什么，下游又会影响什么？**

### 快速源码入口索引

日志、启动参数或源码里遇到相应术语时，可以从这张表进入对应模块。

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

使用这张表时，先带着一个具体问题进入文件，再沿对象和调用关系往下读。例如查 Attention 为什么读到这些历史位置，可以从模型准备的输入追到 Runtime 提供的 metadata，再看 backend 怎样访问缓存、调用算子。这里的 metadata 指长度、位置、缓存索引等辅助信息，它们告诉执行层怎样解释 Tensor。

## 二、模型：从 Token 与 Tensor 到 Logits

用户输入的是文本，模型内部处理的是 Tensor，也就是带有形状和数据类型的多维数组。Tokenizer 先把文本切分成 Token，并映射成整数 ID；一个 Token 可能对应一个字、一段词或其他文本片段，不能直接等同于一个汉字。Embedding 再把这些 ID 映射成向量。假设 hidden size 是 H，一个 Token 的表示就有 H 个分量，多个 Token 的表示共同组成 hidden states。

读到一个 Tensor 时，可以顺着三个问题确认它的含义：

1. 这个 Tensor 的**语义**是什么？
2. 它的 **shape** 是什么？
3. 它的 **dtype** 是什么？

同样是二维 Tensor，它可能保存 hidden states、权重，也可能只是临时缓冲区。例如形状为 `[T, H]` 的 hidden states 中，T 通常表示本轮参与计算的 Token 数，H 表示每个 Token 的特征维度；T 是否包含多个请求、是否经过填充，还要看当前执行路径。只看维度数量，无法确定数据的用途。

### Hidden States、Residual 与 Norm

**Hidden States** 是当前 Token 在模型某一层中的内部表示。Embedding 产生初始 hidden states，随后每个 Decoder Layer 对它继续变换。

Transformer 中还会反复看到 **Residual Connection** 和 **RMSNorm**。Residual 把当前模块算出的结果加回原表示，让信息沿层间继续传递；RMSNorm 根据均方根对数值做归一化，再施加可学习的缩放。日志里的 Add + RMSNorm、Residual + Norm 等融合名称，就是把相邻操作合到一条实现路径中，减少中间数据搬运或启动开销。具体融合了哪些步骤仍要看算子接口。

### Attention、Q/K/V 与 Head

Attention 解决的是：当前 Token 应该从历史上下文的哪些位置读取信息。

经典形式里，输入 X 经过线性投影得到 Q、K、V：

~~~text
Q = XWq
K = XWk
V = XWv
~~~

可以把 Q 理解为当前位置发出的查询，K 是各位置用于匹配的表示，V 是匹配后要读取的内容。Q 与可见位置的 K 计算相关分数，分数经过缩放、掩码和归一化后成为权重，再用这些权重聚合 V。对因果语言模型来说，可见位置受因果掩码约束，不能读取尚未生成的未来内容。

Multi-Head Attention 的意义，是让不同投影子空间并行建模不同关联模式。各 Head 的功能由训练形成；“语法 Head”“实体 Head”等说法适合作为局部观察或教学类比。

在 DeepSeek 语境中还会遇到 **MLA**。它的重要工程意义之一，是改变 K/V 的表示与缓存方式，从而降低传统 Attention 下 KV Cache 的压力。MLA 的表示变化沿下面这条关系影响缓存：

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

Transformer 的另一大块是 MLP / FFN，即前馈网络，通常包含线性变换和非线性激活。Linear 层的大量计算落在 GEMM，也就是通用矩阵乘法上。BF16 GEMM、INT8 GEMM 描述所用精度，Grouped GEMM 描述把多组矩阵乘法组织起来执行的方式，Expert GEMM 则说明这些计算来自 MoE 专家。看见这些名字时，继续核对矩阵尺寸与数据布局，才能判断实际计算量。

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

这条链一旦跨设备，就同时变成模型问题与通信问题。Router 决定 Token 去哪里，EP 决定哪些 Rank 持有哪些 Expert，DeepEP 等通信 backend 负责把 Token 送过去并把结果收回来，Expert Runner 再负责 GEMM。

MoE 性能分析也沿这条链展开，分别查看 Router、Permutation、Dispatch/Combine 和 Expert GEMM 的耗时。

### Logits 与 Sampling

模型最后输出 **Logits**：对词表中候选 Token 的原始分数。

随后 Sampling 决定选哪个 Token。Greedy 直接选分数最高的候选；Temperature 调整分布的尖锐程度；Top-K 限制候选数量；Top-P 保留累计概率达到阈值的一组候选。后几项会影响从哪些候选中抽样，实际顺序与组合要看采样实现。这里的 Top-K 筛选词表 Token；MoE Router 的 Top-K 筛选专家，两者操作的对象不同。

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

到这里，一轮模型计算已经从输入表示走到了下一个 Token。后面的缓存、并行和 Kernel 都围绕这些数据展开：哪些结果可以保留，哪些 Tensor 要切分，哪些操作交给设备执行。

## 三、运行时：生成阶段、缓存与调度

在线推理服务还要安排持续到来的请求：同时来了几百个请求，谁先跑？一轮跑多少 Token？KV Cache 不够怎么办？Prefill 会不会把 Decode 堵死？某个请求结束后能不能立即把空位让给新请求？

这些请求状态、内存预算和执行轮次的管理，合起来构成这里所说的 Runtime，也就是推理运行时。

### Prefill 与 Decode

一条请求通常经历两个不同阶段。

**Prefill** 处理输入 Prompt。一次可能计算很多 Token，矩阵规模更大，硬件计算单元通常更容易被利用。

**Decode** 在已有上下文基础上继续生成新 Token。每轮新增 Token 很少，却需要读取大量历史状态，因此更容易受到 KV Cache、Memory Bandwidth、小 GEMM、Kernel Launch 和通信延迟影响。

这些阶段对应不同的性能指标：

- **TTFT（Time To First Token）**：请求发出到收到首个 Token 的时间，客户端口径通常还包含排队、网络和输入处理。
- **TPOT（Time Per Output Token）**：首个 Token 之后，平均生成一个输出 Token 花费的时间。具体统计公式要看工具，并与相邻 Token 的间隔 ITL 区分。
- **Throughput（吞吐）**：单位时间内整体处理的请求或 Token 数。报告 Token/s 时要说明统计输入、输出还是两者之和。

例如提高并发可能增加总输出 Token/s，同时拉长每条请求的等待时间。吞吐与单请求延迟要一起看，不能互相代替。

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

SGLang 当前 [memory_pool.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/memory_pool.py) 中可以直接看到 **ReqToTokenPool** 与 **KVCache** 这样的结构：前者把请求映射到 token location，后者管理缓存存储抽象。

这一层负责请求状态的分配、寻址与回收，和后端工程中的内存管理职责相近。

### Prefix Cache 与 RadixCache

假设两个请求都以同一段长系统提示词开头，后面才是不同的问题。只要前缀对应的 KV 仍在缓存中、请求的缓存匹配条件也一致，后来的请求就可以复用命中的那段计算结果，只对未命中的部分继续 Prefill。相同文本并不保证一定命中，还要看实际 Token 序列、缓存命名空间和条目是否已被回收。

SGLang 使用 Prefix Cache 复用已有 KV。当前源码中非常直接的入口是 [radix_cache.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/radix_cache.py) 的 **RadixCache**。

这里要区分两个容易混淆的表达：

- **RadixAttention** 更接近 SGLang 的整体 prefix reuse 设计语境；
- **RadixCache** 是当前 Runtime 中可以直接看到的 prefix-cache 数据结构。

所以工作中说“去看 Radix”时，首先要明确是在讨论高层机制，还是当前具体缓存实现。

### Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner

这五个对象把一条请求从接收到设备执行串起来。

**Req** 保存一条请求跨越多轮生成的状态，不只是 Prompt。它会携带长度、输出 Token、Sampling 参数、缓存状态、结束条件以及各种 Runtime metadata。

**Scheduler** 每一轮决定谁进入计算。它面对的是 Waiting Requests、Running Requests、Token Budget、KV Capacity、Prefill / Decode 状态以及 Speculative 状态。

**ScheduleBatch** 表达调度侧“这一轮跑什么”的计划，例如选中了哪些请求、处理多少 Token、使用哪些缓存位置。

**ForwardBatch** 把这一轮计划转成模型执行所需的视图，包括输入 Tensor、positions（位置）、seq_lens（序列长度）、cache location（缓存位置）等信息。Attention metadata 的具体准备时机要继续看 backend 路径，不能仅凭对象名称判断。

**ModelRunner** 承接模型执行，协调模型、backend 以及 eager 或 graph 等执行方式。它让调度侧准备好的输入真正进入一次模型 Forward。

当前源码非常适合用下面这条链来读：

~~~mermaid
flowchart TD
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

这些对象按状态持续时间分工：

> Req 是跨轮次生命周期；ScheduleBatch 是本轮调度视图；ForwardBatch 是本轮 Device Forward 视图。

这对后面读投机解码尤其重要，因为 Draft / Verify / Accept 会让“逻辑长度”“已分配 KV”“已正式提交 KV”出现不同时间点。

### Continuous Batching、Chunked Prefill、Graph 与 Overlap

在线推理不会等一个固定 Batch 的所有请求一起结束，而会不断移出完成请求、加入新请求，这就是 Continuous Batching。

长 Prompt 如果一次 Prefill 完成，可能占用大量预算并阻塞 Decode，因此 Runtime 可以把 Prefill 拆成多个 Chunk；这就是 Chunked Prefill。它是调度策略，不是模型结构。

Decode 期间大量小 Kernel 又会放大 Host Launch Overhead，因此还会出现 Graph Capture / Replay。Ascend 路径当前可以从 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) 进入。

最后是 **Overlap**：把原本串行的计算、通信、结果处理或下一轮准备重叠起来。Overlap 并不天然等于更快，因为两个任务可能竞争同一资源，也可能增加 Buffer 和同步复杂度。

这几项优化各有对象：Continuous Batching 减少批次空位，Chunked Prefill 控制单轮工作量，Graph 减少重复提交开销，Overlap 则让相互独立的工作同时推进。分析效果时，先确认节省的是哪一段时间，再检查是否增加了其他等待。

### PD Disaggregation

**PD Disaggregation** 把 Prefill 与 Decode 放到不同服务实例，让两类负载可以分别调度和配置资源。分开之后，Prefill 侧算出的 KV 与请求状态必须交给 Decode 侧，后者才能接着生成。

这里的 **ownership** 指谁负责持有、更新和释放状态；**handoff metadata** 是交接时用于说明请求、长度、缓存位置等信息的元数据。实现中还要处理目标侧内存预分配、KV 传输、网络与内存注册，以及首轮 Decode 的恢复。拆开计算会引入交接成本，是否划算取决于节省的等待和资源竞争能否覆盖这些成本。

SGLang 当前通用 Prefill 侧入口可以从 [disaggregation/prefill.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/prefill.py) 看，Ascend transfer 路径则有 [disaggregation/ascend/transfer_engine.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/ascend/transfer_engine.py)。

读 PD 代码时，可以跟着一条请求的 KV 往下追：Prefill 侧什么时候写完，Decode 侧什么时候可读，哪一侧最终释放。这样比单看“已发送”或“已接收”的日志更容易确认交接是否完成。

## 四、分布式：Rank、TP、DP Attention 与 EP

模型大到一张卡放不下，或者吞吐目标高到单卡不够，就必须进入分布式世界。

读多卡代码时，常会在同一段里看到 TP、DP、EP、CP、Group、Rank、AllReduce、DeepEP、HCCL。可以先把它们按职责分开：并行方式决定怎样分工，Rank 和 Group 描述参与者，通信操作与后端负责交换数据。

### Rank、World Size 与 Process Group

**Process Group** 是一起参与通信的一组进程，**Rank** 是进程在相应组内的逻辑编号，**World Size** 是这个组的进程数。Global Rank 通常指默认全局组中的编号；TP Rank、EP Rank 等则属于相应子组。日志里出现 world size 时，也要先确认它指全局组还是某个子组。

一进程一卡时，Rank 和设备看起来一一对应，但两者的编号含义不同。尤其在多机部署中，机器内的设备编号会重复，全局 Rank 则用于区分不同进程。

同一个进程还可能同时属于多个 Group，并在每个 Group 中拥有不同的局部 Rank。例如：

~~~text
Global Rank = 6

TP Group = [4, 5, 6, 7]
→ TP Rank = 2

DP Group = [2, 6, 10, 14]
→ DP Rank = 1
~~~

上例中，全局编号 6 的进程在 TP 组里排第三位，在 DP 组里排第二位，因此局部 Rank 分别是 2 和 1。这是说明编号关系的例子，实际分组要以运行配置为准。TP、DP、EP，以及 Draft 和 Verify 阶段的 Rank，都需要连同所属组一起解释。

当前 SGLang 的 group 初始化与 parallel state 可以从 [parallel_state.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/parallel_state.py) 进入。

### TP：Tensor Parallelism

TP 把同一个 Layer 内的大 Tensor / GEMM 拆到多个 Rank 共同完成。

例如 Y = XW，W 可以按某个维度切成多个 shard。不同 TP Rank 保存不同权重切片，完成局部计算后再通过 Collective 恢复下一步需要的数据语义。

因此 TP 的核心是：

> **同一个请求、同一个 Layer，需要多个 Rank 协作。**

也正因为如此，它天然会带来 AllReduce、AllGather、ReduceScatter 等通信。

### DP 与 DP Attention

经典 **Data Parallelism** 让模型的多个副本（Replica）处理不同数据。推理服务里，可以用多个副本承接不同请求，增加服务容量。

SGLang 的 **DP Attention / DPA** 则在同一个模型并行世界内，为 Attention 划分请求与 KV。

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

展开该配置时，需要核对：

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

于是 MoE Forward 从模型视角的：`Router → Expert`

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

### DeepEP 与专家计算的边界

**DeepEP** 负责 Expert Parallel 中 Token 的 Dispatch / Combine 通信：先把数据送到相应专家所在的进程，专家算完后再合并返回。

SGLang 当前有 [deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) 和 [deepep_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py) 等 dispatcher。

而 Expert 如何计算，则是另一层 Runner；Ascend 路径可以继续看 [moe_runner/ascend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/moe_runner/ascend.py)。

通信与计算的边界可以记成：

~~~text
DeepEP
→ Token 怎么跨 Rank 到 Expert

Ascend MoE Runner
→ 到了本 Rank 后 Expert GEMM 怎么算
~~~

如果 Dispatch / Combine 占了较长时间，就继续检查通信、数据重排以及 Rank 间等待；如果通信与其他工作重叠，还要看其中多少落在关键路径上。单个通信区间长，并不代表把它缩短多少，整轮就一定能节省多少。

### Collective：AllReduce、AllGather、ReduceScatter、All-to-All

这些通信原语服务于 TP、DP、EP 等并行布局。

- **AllReduce**：所有 Rank 贡献局部值，归约后所有 Rank 都得到结果；
- **AllGather**：每个 Rank 持有一部分，最终所有 Rank 得到完整集合；
- **ReduceScatter**：先归约，再把结果切分给不同 Rank；
- **All-to-All**：每个 Rank 都可能向其他 Rank 发送不同数据，MoE Dispatch 很典型。

从很多常见算法的逻辑上，可以把 AllReduce 理解成 ReduceScatter + AllGather 两阶段，但这是一种语义/算法模型，不意味着底层实现一定机械调用两个 API。

Ascend 环境中还会看到 **HCCL**。它属于 Collective Communication Library 层，是 AllReduce、AllGather、All-to-All 等通信落地的重要基础设施之一。

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

区分 CP 与 SP 时，要同时看当前框架划分的 Group、每个 Rank 持有的 Tensor，以及计算中发生的通信。

## 五、投机解码：Draft、Verify 与 Accept

普通自回归 Decode 的问题是：Target Model 每做一次昂贵 Forward，只向前推进很少的新 Token。

Speculative Decoding 的核心思想是：

> **先用更便宜的路径提出多个候选，再让 Target Model 一次验证。**

最简化状态机是：

~~~mermaid
flowchart TD
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

**Draft** 描述的是提出候选的角色；实现这个角色时，可以使用独立小模型，也可以复用大模型中的预测模块。

不同实现可能是：

- 独立 Draft Model；
- MTP / NextN Head；
- EAGLE 风格 Draft；
- DSpark Head；
- N-gram 等非神经路径。

当前 DeepSeek-V4 文档里已经存在 checkpoint 自带 DSpark head 的部署方式；与此同时，Ascend 某些 CI / 模型配置也仍会显式传入 speculative draft model path。两种部署都服务于 Draft 提议，模型权重的组织方式各有不同。

最稳妥的心智模型是：

> **Draft 是功能角色；“它是不是另一套权重、另一套模型实例、同一 checkpoint 的额外 head”，要看当前算法与 checkpoint。**

### Verify 与 Accept

**Verify** 用 Target Model 计算候选位置所需的分数或概率，**Accept** 再按当前算法的接受规则决定哪些候选进入正式历史。Draft 产生的内容在这之前仍是待验证的未来，不能直接当作已输出序列使用。

Accept 同时推进输出序列与后续可见状态，包括：

- Sequence Length；
- Position；
- KV Cache；
- Draft/Verify Metadata；
- 下一轮 Draft State。

因此投机解码代码经常围绕一个核心问题：

> **哪些状态只是 speculative future，哪些状态已经 committed？**

沿这个边界就能理解几个常见动作：**commit** 把接受的 Token 和对应状态确认为正式历史；**compact** 把需要保留的状态整理到后续使用的布局；**rollback** 撤销未接受的临时状态；**reclaim** 回收不再使用的空间。它们的具体顺序与操作对象取决于算法和缓存实现。

### 接受长度怎样与周期成本一起决定收益

投机解码收益取决于：

~~~text
有效接受的 Token 数
÷
Draft + Verify + KV + Scheduler + Communication 的总成本
~~~

读日志时要区分接受长度与接受率。前者通常描述一个周期推进多少 Token，后者需要明确“接受的候选数 / 提出的候选数”等统计分子与分母；是否计入额外的 Target Token，也要看实现。二者都要和周期耗时一起看。

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

扩展 TP / DP 组合时，需要让下面几项一起对应新布局：

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

## 六、Ascend：Backend、CANN、Kernel 与量化

前面解释了模型要算什么、Runtime 怎样组织数据。接下来沿 Backend、Operator 和 Kernel，看看这些计算如何在 Ascend 910C 上执行。

### Backend：同一高层语义的具体实现选择

SGLang 中经常看到：

- Attention Backend；
- MoE A2A Backend；
- MoE Runner Backend；
- Quantization Backend。

Backend 将模型所需的计算映射到具体设备实现，选择算子、布局与执行方式。

同一类计算可能因为硬件、模型结构、精度和运行模式不同，使用不同 backend。对性能有影响的往往正是它选中的布局、算子和分派分支。

例如 Attention 是模型语义；而 <code>--attention-backend dsv4</code> 决定 DeepSeek-V4 Attention 在当前平台走哪套 Runtime / Operator 路径。

当前注册入口在 [attention_registry.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/attention/attention_registry.py)，Ascend DSV4 具体实现继续进入 [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)。

### Operator 怎样映射到 Kernel

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

算子到 Kernel 的映射随融合与分派变化：

- 一个高层 Operator 启动多个 Kernel；
- 多个高层操作被 Fuse；
- 同一个 Operator 因 Shape / DType / Mode 选择不同 Kernel；
- 某些路径还包含 Host 侧预处理、Workspace 和额外通信。

分析 Attention 耗时时，继续确认：

- 最后走了哪个 dispatch branch？
- Shape / dtype 是什么？
- Tiling 怎么选？
- Kernel 是 compute-bound 还是 memory-bound？
- 有没有额外的数据布局转换？

### CANN、Tiling 与 TilingKey

CANN 是 Ascend AI 软件栈中的基础设施集合，向上承接算子调用，向下组织设备执行。

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

只看这八个比特，还无法确定量化格式或计算路径，需要继续确认：

- Weight 是 per-tensor、per-channel 还是 per-group？
- Activation 是 static 还是 dynamic？
- 是 per-token 还是其他粒度？
- 对称还是非对称？
- Scale / zero point 怎么存？
- 哪些层排除量化？
- Linear 与 MoE 使用什么 Kernel？

比较两个 W8A8 模型时，需要将这些格式与执行选择逐项对齐。

SGLang 当前 ModelSlim 入口在 [modelslim.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/quantization/modelslim/modelslim.py)，相关 scheme 继续位于同目录下。

排查量化路径时，选定一个具体 Layer，把它的 Weight、Activation、Scale 格式与实际 Kernel 接口对上。这样才能判断格式是否兼容，以及时间花在量化、矩阵乘法还是额外的转换上。

## 七、按工作任务继续阅读

下面按常见工作任务列出专题入口。遇到某个对象不熟悉时，可以回到对应词条查定义，再接着追当前问题，不必先记住整张名词表。

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

可以从 ForwardBatch 提供的位置与长度信息入手，沿下面的顺序追踪，最后看 Ascend DSV4 Backend 怎样把这些信息传给算子：

~~~text
ForwardBatch
→ Query
→ KV write
→ 历史 KV 读取
→ Ascend DSV4 Backend
~~~

### 如果你在看 MoE / TP / DP / EP

按这个顺序：

1. [DeepSeek-V4 MoE 推理源码解析](../sglang/deepseek-v4-moe-source-analysis.md)
2. [DeepSeek-V4 分布式推理源码解析](../sglang/deepseek-v4-distributed-parallel-source-analysis.md)

前者沿 Router、Dispatch、Expert、Combine 解释一次 MoE 计算；后者继续看同一批 Rank 在两个阶段怎样分工、持有哪些行与权重：

~~~text
同一个 TP world
在 Attention 阶段怎样解释
在 MoE 阶段又怎样重新解释
~~~

把两篇连起来，才能把 TP、DP、EP 的定义落到具体的 Tensor 布局与通信组上。

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

### 把术语放回正在排查的问题

把这些词放回一次推理，可以沿下列关系连接起来：

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

查词之后，回到正在读的对象，按需要核对下面几个问题：

1. **它是什么语义？**
2. **它的数据 shape / dtype / ownership 是什么？**
3. **它处在哪个 Runtime 边界？**
4. **它需要哪些 Process Group 或通信？**
5. **最后由哪个 Backend / Operator / Kernel 消费？**

例如看到 **seq_lens**，不要只问“这个数组是什么”，而要继续问它处于 Req、ScheduleBatch 还是 ForwardBatch 的哪一层，以及它如何影响 position 与 KV。

看到 **attn_tp_group**，不要只问“TP 是什么”，而要问 Group 中包含哪些 Global Rank、当前 Attention DP/TP 坐标系是什么。

看到 **DeepEP**，沿 Dispatch/Combine 追踪 Token 交换，再到 Expert Runner 查看 GEMM。

看到 **TilingKey**，先判断你已经进入 Operator → Kernel Dispatch 层。

看到 **DSPARK**，先判断你进入了 Speculative Decoding 状态机，再继续追 Draft、Verify、Accept 和 Commit 边界。

例如排查一次 Decode 变慢，可以从请求的 seq_lens 确认上下文规模，再核对 Attention 的缓存读取和通信组，最后到设备时间线确认耗时。名词表的作用，是让这条追踪过程少一些概念上的停顿；具体行为仍以固定版本源码和当前运行证据为准。
