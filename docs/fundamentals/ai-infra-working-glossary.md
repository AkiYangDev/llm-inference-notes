# AI 推理基础设施名词表：从 Token 到 Ascend Kernel

同一屏推理日志里可能同时出现 Token、KV Cache、TP、Backend、Kernel。它们分别说明数据、历史状态、设备分工和执行实现，沿一次请求看就比较容易分清。

这份名词表先讲文本与模型，再讲缓存、调度、多卡通信，最后落到 Ascend 执行。查词可用速查表，初次阅读可从全局地图顺着走；源码入口固定在同一版本，便于继续定位实际对象。

> SGLang 入口固定到 <code>sgl-project/sglang @ 176dbcb85d3b7737564e4911947035cc0af65f0a</code>，核对日期为 2026-09-21。这里给出模块定位，完整调用关系在专题中展开。

## 速查：按当前问题找术语

速查链接跳到解释，同组术语并列放置，便于核对各自职责。

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

例如输入 `请解释一下 DeepSeek 的 MoE 是怎么工作的？`。

请求从文本到输出，主要经过：

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

图中可按七类职责查找问题：缓存容量先查内存，组成员不符先查并行布局。

| 层级 | 你会看到的词 | 核心问题 |
| --- | --- | --- |
| 模型语义 | Token、Embedding、Q/K/V、Attention、MLA、MoE、Logits | 模型数学上在算什么？ |
| 推理生命周期 | Prefill、Decode、KV Cache、Prefix Cache | 一条请求如何逐步生成？ |
| Runtime | Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner | 哪些请求在这一轮执行？ |
| 内存 | Page、KV Pool、Activation、Workspace、Buffer | 状态放在哪里，显存怎么分配？ |
| 分布式 | Rank、Group、TP、DP、DPA、EP、CP、Collective | 谁拥有哪部分数据，谁和谁通信？ |
| 投机解码 | Draft、Verify、Accept、DSpark | 怎样用更便宜的预测换取更少的 Target Decode？ |
| Device 执行 | Backend、CANN、Operator、Tiling、Kernel、Stream | Python 里的 Forward 最终怎样落到 910C？ |

遇到陌生对象，先找到它所在的层，再看输入来自哪里、输出交给谁。



### 快速源码入口索引

参数、日志或源码中的对象，可从这些模块进入。

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

例如查 Attention 的历史访问，可由模型输入追到 Runtime metadata，再到 backend 的缓存和算子。metadata 是长度、位置、缓存索引等辅助信息，用来说明 Tensor 该怎样被解释。

## 二、模型：从 Token 与 Tensor 到 Logits

模型处理的是有 shape、dtype 的多维数组。Tokenizer 将文本切成片段并映射为 ID，片段可以是字、词的一部分或其他内容，不保证与汉字一一对应。Embedding 再将 ID 映射为 H 维向量，各 Token 的表示组成 hidden states。

读 Tensor 时先核对：

1. 数据表示什么；
2. shape 各维对应什么；
3. dtype 使用哪种数值格式。

二维数组可能是 hidden、权重或工作 buffer。`[T,H]` 通常表示本轮 T 行、每行 H 个特征，但是否含多个请求、是否 padding，仍需查路径，维数本身不能决定语义。

### Hidden States、Residual 与 Norm

Hidden states 是 Token 在当前层的内部表示，初始由 Embedding 产生，再由各 Decoder Layer 变换。

Residual 将模块结果加回原表示；RMSNorm 按均方根归一化并施加可学习缩放。Add+RMSNorm 等融合可将相邻工作合并，减少中间搬运或启动，但实际范围要看接口。

### Attention、Q/K/V 与 Head

Attention 计算当前 Token 如何从可见上下文取信息。

经典形式用投影得到：

~~~text
Q = XWq
K = XWk
V = XWv
~~~

Q 发起查询，K 用于匹配，V 提供被聚合内容。Q/K 分数经缩放、mask、归一化后成为权重，再加权 V。因果模型限制可见位置，不能读取尚未生成的未来。

Multi-Head 在不同投影子空间建模关联。具体功能由训练形成，语法、实体等 Head 名称只能用于相应局部观察或类比。

MLA 改变 K/V 表示和缓存方式，是缓解 KV 压力的一条路径。其影响可沿下面关系理解：

~~~text
Attention 结构
   ↓
决定历史状态怎么保存
   ↓
决定 KV Cache Layout
   ↓
影响显存与分布式布局
~~~

V4 又有专用滑窗、压缩历史，因此具体读取与保存还需回到模型和 dsv4 backend。入口为 [deepseek_v4.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4.py) [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)。

### MLP、GEMM 与 MoE

MLP/FFN 是线性与非线性组成的前馈计算，大量 Linear 落为 GEMM。BF16、INT8 描述精度，Grouped 描述多组组织，Expert 说明来自专家模块；实际工作量仍需矩阵形状、布局。

MoE 按以下路径选择专家计算：

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

Router 选目标，EP 分布权重，DeepEP 等实现传送输入、合回贡献，Expert Runner 执行计算。跨设备后，需要把这些职责接起来看。

性能上也应分别定位 Router、重排、Dispatch/Combine 与 Expert GEMM。

### Logits 与 Sampling

Logits 是对词表候选的原始分数。

Sampler 决定输出 ID：Greedy 取最大分数，temperature 改分布刻度，top-k 限候选数，top-p 按累计概率保留集合。组合顺序以实现为准。这里筛词表，与 Router 选专家的 Top-K 不同。

生成关系为：

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

对应入口见 [logits_processor.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/logits_processor.py) [sampler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/sampler.py)。

模型到这里得到下一 ID，缓存、并行和设备执行则继续回答哪些结果保留、数据怎样分布、计算怎样提交。

## 三、运行时：生成阶段、缓存与调度

在线服务还要管理同时到达的请求：选择本轮工作、分配 Token 预算、处理缓存不足，并在请求完成后接纳新的任务。

这些持续状态与轮次管理构成推理 Runtime。

### Prefill 与 Decode

普通生成先后经过：

Prefill 处理已知 Prompt，一次可有较多新 Token，较大矩阵通常更容易利用计算单元。

Decode 逐轮接续输出，普通单请求新输入较少，还要访问历史，因此常受缓存、带宽、小 GEMM、启动或通信延迟影响。

读性能结果时分别看：

- TTFT：发请求到收到首 Token，客户端口径常含排队、网络、输入处理；
- TPOT：首 Token 后的平均逐 Token 时间，公式需看工具，并和 ITL 区分；
- Throughput：单位时间的请求或 Token 数，需注明输入、输出或合计。

提高并发可能同时增加总吞吐和请求等待，两种指标都要报告。

### KV Cache 与 Page

KV Cache 保存 Attention 历史状态，避免生成下一位置时重新计算此前全部 K/V。

上下文、并发增加通常会增加状态需求，具体增长取决于缓存结构：

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

Runtime 常将池按 Page/Block 管理，用映射连接逻辑序列与物理空间，不为每请求固定一块大连续区。

ReqToTokenPool 记录请求、位置关系，KVCache 管理存储抽象，可从 [memory_pool.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/memory_pool.py) 进入。

这一层负责分配、寻址、回收，与后端内存管理职责相近。

### Prefix Cache 与 RadixCache

例如两请求共享长系统前缀，后缀不同，若 KV 仍驻留且匹配条件成立，第二条可复用前缀、仅处理后缀。相同文本还不够，需要 Token 序列、命名空间和条目生命周期都符合。

SGLang 的具体前缀缓存结构 RadixCache 位于 [radix_cache.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/radix_cache.py)。

讨论 Radix 时应区分语境：

- RadixAttention 常用于整体 prefix reuse 设计的说明；
- RadixCache 是固定 Runtime 的具体前缀缓存数据结构。

查问题时先确认说的是机制，还是代码对象。

### Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner

五个对象衔接请求与设备执行。

Req 保存跨轮输入、输出、采样、长度、缓存、停止条件及 metadata。

Scheduler 根据等待、运行集合、Token 预算、KV 容量和各阶段状态，决定本轮执行谁。

ScheduleBatch 描述这轮选中的请求、输入区间及缓存安排。

ForwardBatch 提供模型输入 Tensor、positions、seq_lens、cache locations。Attention metadata 的准备时机还需看 backend，不能由对象名判断。

ModelRunner 将这些输入交到模型，并协调 backend、Eager 或 Graph 执行。

阅读时可沿下面的循环：

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

各对象入口为：

- Req / ScheduleBatch：[schedule_batch.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/schedule_batch.py)
- Scheduler：[scheduler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/scheduler.py)
- ForwardBatch：[forward_batch_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/forward_batch_info.py)
- ModelRunner：[model_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/model_runner.py)

其中 Req 跨轮存在，ScheduleBatch 是本轮计划，ForwardBatch 是一次执行视图。



投机和 Overlap 中，逻辑输出、KV 预留、正式提交在不同时刻更新，这种职责差别尤为重要。

### Continuous Batching、Chunked Prefill、Graph 与 Overlap

Continuous Batching 逐轮移出完成请求、加入新请求，无需等待一个固定 batch 全部结束。

Chunked Prefill 将长输入分段，控制单轮预算并给 Decode 调度机会，属于 Runtime 策略。

小 Kernel 还会放大 Host 提交成本，Graph Capture/Replay 尝试减少重复启动。NPU 入口见 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py)。

Overlap 安排独立计算、通信、结果处理或准备同时推进，也可能增加争用、buffer 和同步，收益需实测。

因此批次填充、工作分块、重复提交、独立工作重叠各有不同目标，比较时要看节省和新增的时间。

### PD Disaggregation

PD 分离将 Prefill、Decode 放进不同实例，分别调度资源，再将输入阶段完成的 KV、请求状态交给 D 接续。

ownership 说明谁持有、修改、释放；handoff metadata 传递请求身份、长度、位置。目标预分配、内存注册、网络传输和恢复都会增加成本，需要与隔离收益比较。

P 侧与 Ascend transfer 入口为 [disaggregation/prefill.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/prefill.py) [disaggregation/ascend/transfer_engine.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/ascend/transfer_engine.py)。

可跟踪同请求状态何时写完、何时可读、由谁释放，而不只依赖发送/接收日志。

## 四、分布式：Rank、TP、DP Attention 与 EP

容量或吞吐超过单设备能力时，需要将存储、计算分给多卡。

并行策略规定分工，Rank/Group 标识参与者，collective、backend 完成数据交换，先按这几层读。

### Rank、World Size 与 Process Group

Process Group 是通信集合，Rank 是集合内编号，World Size 是成员数。Global Rank 常指默认全局组，TP、EP Rank 则是各自子组位置，world size 也需查语境。

一进程一卡也不代表编号含义相同，多机设备编号可重复，全局 Rank 则区分进程。

一个进程可同时属于多个组：

~~~text
Global Rank = 6

TP Group = [4, 5, 6, 7]
→ TP Rank = 2

DP Group = [2, 6, 10, 14]
→ DP Rank = 1
~~~

全局 6 在 TP 组排第三、DP 组排第二，对应局部 2、1。这里只示意编号，真实成员需查配置，Draft、Verify 的编号也要带组名解释。

组初始化见 [parallel_state.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/parallel_state.py)。

### TP：Tensor Parallelism

TP 将同层 Tensor/GEMM 分给多 Rank。

例如 Y=XW，权重沿维度分成 shards，本地计算后再用 collective 形成下游所需结果。

这些 Rank 协作处理同一请求、同一层，也因此引入相应通信。





### DP 与 DP Attention

普通 DP 创建完整模型副本，让 Replica 分别服务不同请求，以提高系统的服务吞吐。

DPA 则在同一个模型并行 world 内划分 Attention 请求和 KV。

固定派生集中在 Runtime Context，产生以下宽度。[runtime_context.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/runtime_context.py)

~~~text
attn_dp_size
attn_cp_size
attn_tp_size
~~~

启用 DPA 后，Attention DP 来自相应配置，组内 TP 还需结合总宽与 CP 计算。

例如配置为：

~~~text
tp_size = 16
dp_size = 8
enable_dp_attention = true
~~~

需继续核对：

- Attention DP 的成员；
- 每请求组内的 Attention TP 宽度；
- 请求和 KV 的归属；
- MoE 如何重新组织同批 Rank。

相关 per-process metadata 见 [dp_attention.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/dp_attention.py)。

### EP：Expert Parallelism

EP 将不同专家权重分给不同 Rank，避免每卡复制所有专家：

~~~text
Rank 0 → Experts A/B
Rank 1 → Experts C/D
Rank 2 → Experts E/F
...
~~~

Token 选中远端专家后，输入先送到权重所在 Rank，计算完再返回贡献。

于是 Router→Expert 的模型过程在系统中展开为：



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

入口见 [ep_moe/layer.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/ep_moe/layer.py)。

### DeepEP 与专家计算的边界

DeepEP 承接专家并行的 Dispatch、Combine，而非专家矩阵乘本身。

Dispatcher 实现包括 [deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) [deepep_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py)。

Ascend 专家计算另由 Runner 承接，见 [moe_runner/ascend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/moe_runner/ascend.py)。

两层对应为：

~~~text
DeepEP
→ Token 怎么跨 Rank 到 Expert

Ascend MoE Runner
→ 到了本 Rank 后 Expert GEMM 怎么算
~~~

分发时长大还需区分重排、传输、跨 Rank 等待。部分若已被其他工作覆盖，缩短通信总时长未必等量缩短周期。

### Collective：AllReduce、AllGather、ReduceScatter、All-to-All

常见原语为：

- AllReduce 合并局部贡献，并将归约结果交给各 Rank；
- AllGather 收齐各 Rank 的分片；
- ReduceScatter 合并贡献后分别交付结果区间；
- All-to-All 按目的地交换各自的数据。

ReduceScatter+AllGather 可用来解释常见 AllReduce 算法，语义分解不代表底层必调用这两个 API。

HCCL 属于 Ascend collective library，承接相关分布式调用。

RDMA 描述跨机器远程内存和搬运能力，属于更下层传输，不能等同 TP、EP。

层次为：

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

PP 沿深度分层，各 stage 执行不同层。

CP 沿上下文分布数据，Attention 还可能访问其他 Rank 历史，不能只按切 Token 数组理解。

SP 也涉及 sequence，但经典形式常用于 TP activation 分片、节约内存，与 CP 的整段上下文协作不同。

具体定义需连同框架 Group、本地 Tensor 和交换方式确认。

## 五、投机解码：Draft、Verify 与 Accept

普通 Target Decode 每次完整前向推进的输出较少。

投机先由较轻路径提多个候选，再交 Target 验证。



过程为：

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

Draft 是提议角色，可以由独立模型或大模型附带模块实现。

例如：

- 独立 Draft Model；
- MTP / NextN Head；
- EAGLE 风格 Draft；
- DSpark Head；
- N-gram 等非神经路径。

V4 可使用 checkpoint 自带 DSpark head，部分 Ascend CI 也显式传 draft model path。两者都是提议路径，权重组织方式不同。

是否独立权重、独立实例或额外 head，需要按算法、checkpoint 确认。



### Verify 与 Accept

Verify 为候选位置计算 Target 结果，Accept 按算法规则选有效历史，候选在此之前尚未成为正式输出。

结果还会影响：

- Sequence Length；
- Position；
- KV Cache；
- Draft/Verify Metadata；
- 下一轮 Draft State。

需分清临时候选状态与已提交历史。



Commit 确认有效输入状态，Compact 转为下游布局，Rollback 阻止拒绝状态影响后续，Reclaim 回收空间。具体对象和顺序由算法、缓存决定。

### 接受长度怎样与周期成本一起决定收益

性能比较可先看：

~~~text
有效接受的 Token 数
÷
Draft + Verify + KV + Scheduler + Communication 的总成本
~~~

接受长度通常含周期输出推进，严格接受率则需明确实际接受/提出的候选计数，并说明 bonus。二者都要与周期耗时并列。

Draft、宽 Verify、KV 搬运或同步增加过多时，投机也可能比普通 Decode 慢。

### DSpark 在源码里从哪里看

源码可顺次进入：

1. [spec_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/spec_info.py)：先理解 speculative metadata 的数据结构；
2. [dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)：看 worker 如何组织 Draft / Target / Verify；
3. [deepseek_v4_dspark.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4_dspark.py)：看 V4 DSpark 模型结构怎样进入 forward。

多卡还需核对：

- Draft 的 TP/DP/EP 布局；
- Verify 使用的 Group；
- Accept 的同步参与者；
- KV 的坐标和所有权；
- 两种布局之间的转换。

扩展配置需同时满足：

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

这些依赖连接算法、Runtime、内存、通信，不能只放开参数。

## 六、Ascend：Backend、CANN、Kernel 与量化

最后由 Backend、算子和 Kernel 将计算交给 Ascend 910C。

### Backend：同一高层语义的具体实现选择

常见选择包括：

- Attention Backend；
- MoE A2A Backend；
- MoE Runner Backend；
- Quantization Backend。

Backend 将高层计算映射为具体布局、算子和执行分支。

设备、模型、精度、阶段不同，选择也可能不同，分析性能要找到实际进入的分支。

例如 Attention 是计算语义，dsv4 则决定平台对应的实现。

注册和 NPU 实现分别见 [attention_registry.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/attention/attention_registry.py) [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)。

### Operator 怎样映射到 Kernel

调用层次可画为：

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

映射并不一一对应：

- 一个 Operator 可启动多个 Kernel；
- 相邻高层操作可融合；
- shape、dtype、mode 可选择不同 Kernel；
- 路径还可能包含 Host 准备、workspace 或通信。

核对 Attention 耗时时应保留：

- 实际 dispatch 分支；
- shape、dtype；
- Tiling 选择；
- 计算与搬运限制；
- 额外布局转换。

### CANN、Tiling 与 TilingKey

CANN 向上承接算子调用，向下组织 Ascend 执行。

高性能或自定义算子常经过：

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

片上高速空间有限，大 Tensor 需分 Tile 搬入、计算、写回。

Tiling 决定：

- 每块数据量；
- 各 Core 的分工；
- 循环次数；
- Local Buffer 使用；
- 搬运和计算的安排。

TilingKey 用于选择相应形状、精度、模式的 Kernel 分支。

所以算子同名不保证内部执行相同。

### Stream、Graph 与异步执行

Host 提交后，设备通常按 Stream 异步执行，API 返回不等于计算完成。

计时需区分：

- Host API 时间；
- Kernel 执行时间；
- Stream Synchronization；
- Compute / Communication overlap。

NPU Graph 尝试减少重复 Launch、Host 调度，入口为 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py)。

### BF16、W8A8 与 ModelSlim

BF16 是 16-bit 浮点，比 FP32 占用更少字节，支持良好时可能提高吞吐，但实际收益不必与位宽成比例。

W8A8 表示权重和激活都使用 8-bit 表示，但这两个位宽还不足以描述完整量化方案。

还需继续确认：

- 权重的 per-tensor/channel/group 粒度；
- 激活静态或动态；
- 是否 per-token；
- 对称或非对称；
- scale、zero point 格式；
- 排除层；
- Dense、MoE Kernel。

比较同名 W8A8 前，需先对齐这些格式与路径。

ModelSlim 入口见 [modelslim.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/quantization/modelslim/modelslim.py)，各 Scheme 位于相应目录。

可以选一层，将权重、激活、scale 与算子实参逐项接上，再分辨耗时来自量化、矩阵乘或转换。

## 七、按工作任务继续阅读

下面按任务列出后续文章，查过定义即可返回当前问题，无需先记全表。

### 从一次请求开始

请求到设备可依次读：

1. 先读 [SGLang 请求生命周期](../sglang/sglang-ascend-request-lifecycle.md)；
2. 再读 [SGLang Scheduler](../sglang/sglang-scheduler-source-analysis.md)；
3. 再读 [SGLang ModelRunner](../sglang/sglang-modelrunner-source-analysis.md)。

这三篇分别跟踪请求接入、调度组批和模型执行，适合连着读。

### 如果你在看 Attention / KV Cache

缓存访问可读：

- [DeepSeek-V4 Attention 源码解析](../sglang/deepseek-v4-attention-source-analysis.md)

沿着位置和长度看 Query 怎样写入缓存、读取历史，再到 Ascend backend 核对实际算子参数。

### 如果你在看 MoE / TP / DP / EP

MoE 与并行可依次读：

1. [DeepSeek-V4 MoE：从路由到专家计算](../sglang/deepseek-v4-moe-source-analysis.md)
2. [DeepSeek-V4 分布式 Forward](../sglang/deepseek-v4-distributed-parallel-source-analysis.md)

第一篇跟路由到专家输出，第二篇看同一批 Rank 怎样在 Attention、MoE 两阶段转换数据布局和通信组。

### 如果你在看 DSpark / 投机解码

投机系列可依次读：

1. [DeepSeek-V4 Speculative Decoding 总链路](../speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)
2. [DeepSeek-V4 DSpark Parallel Layout](../distributed/deepseek-v4-dspark-parallel-layout-analysis.md)
3. [SGLang Speculative Scheduler](../speculative-decoding/sglang-speculative-decoding-scheduler-source-analysis.md)
4. [Accept Decision](../speculative-decoding/deepseek-v4-accept-decision-source-analysis.md)
5. [KV Commit / Compact / Rollback](../speculative-decoding/deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md)
6. [多卡一致性](../distributed/deepseek-v4-speculative-distributed-consistency-source-analysis.md)
7. [Speculative 性能模型](../performance/deepseek-v4-speculative-performance-analysis.md)

候选生成和树语义另见：

- [MTP / NextN](../speculative-decoding/deepseek-v4-mtp-speculative-decoding-source-analysis.md)
- [EAGLE Draft Tree](../speculative-decoding/deepseek-v4-eagle-draft-tree-source-analysis.md)
- [Tree Attention](../speculative-decoding/deepseek-v4-tree-attention-source-analysis.md)

### 如果你在看 PD 分离

PD 可读：

- [DeepSeek-V4 Prefill / Decode Disaggregation 源码解析](../sglang/deepseek-v4-pd-disaggregation-source-analysis.md)

沿阶段差异继续看 KV、state 如何跨实例交接。

### 把术语放回正在排查的问题

把术语放回数据流：

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

查词后，可核对：

1. 它表达的计算或状态；
2. shape、dtype、归属；
3. 所在 Runtime 边界；
4. 相关 Group、通信；
5. 消费它的 backend、算子、Kernel。

这些问题要落到具体对象上：

- 读 `seq_lens`，先确认对象和阶段，再看它怎样影响 positions、KV。
- 读 `attn_tp_group`，核对有序成员及当前 Attention 坐标。
- 读 DeepEP，先追分发、合并，再到 Runner 看专家计算。
- 读 TilingKey，继续检查对应的设备分支。
- 读 DSPARK，沿候选、Verify、Accept、Commit 追状态。

例如 Decode 变慢，先确认本请求的上下文，再查历史访问、Group，最后对齐设备时间线。这份表用于减少概念停顿，具体行为仍由固定源码与当前运行证据确认。
