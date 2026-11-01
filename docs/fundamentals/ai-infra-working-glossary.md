# AI 推理基础设施名词表：从 Token 到 Ascend Kernel

Token、KV Cache、TP、Backend、Kernel 经常出现在同一屏日志里。Token 是模型处理的数据，KV Cache 留住历史，TP 规定设备怎样分工，Backend 和 Kernel 决定具体怎样执行。这些词说的是同一条请求的不同部分。

正文按文本、模型、缓存、调度、通信、Ascend 执行排列。查单个词可以从速查表跳转，需要把几个词连起来时再看全局地图。源码入口固定在同一版本，便于找到对应对象。

> SGLang 入口固定到 <code>sgl-project/sglang @ 176dbcb85d3b7737564e4911947035cc0af65f0a</code>，核对日期为 2026-09-21。这里给出模块定位，完整调用关系在专题中展开。

## 速查：按当前问题找术语

同组链接放在一起，点击可到相应解释。

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

以输入 `请解释一下 DeepSeek 的 MoE 是怎么工作的？` 为例。

文本到回答，会经过这些处理。

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

图里的七类职责分别承担请求的一部分。缓存容量属于内存问题，组成员则属于并行布局。

| 层级 | 你会看到的词 | 核心问题 |
| --- | --- | --- |
| 模型语义 | Token、Embedding、Q/K/V、Attention、MLA、MoE、Logits | 模型数学上在算什么？ |
| 推理生命周期 | Prefill、Decode、KV Cache、Prefix Cache | 一条请求如何逐步生成？ |
| Runtime | Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner | 哪些请求在这一轮执行？ |
| 内存 | Page、KV Pool、Activation、Workspace、Buffer | 状态放在哪里，显存怎么分配？ |
| 分布式 | Rank、Group、TP、DP、DPA、EP、CP、Collective | 谁拥有哪部分数据，谁和谁通信？ |
| 投机解码 | Draft、Verify、Accept、DSpark | 怎样用更便宜的预测换取更少的 Target Decode？ |
| Device 执行 | Backend、CANN、Operator、Tiling、Kernel、Stream | Python 里的 Forward 最终怎样落到 910C？ |

一个对象所处的位置，决定了它接收什么、把结果交给谁。

### 快速源码入口索引

参数、日志里遇到的对象，可以在下列模块中找到。

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

例如 Attention 的历史访问，从模型输入经过 Runtime metadata，进入 backend 的缓存与算子。metadata 指长度、位置、缓存索引等辅助信息，它们给出了 Tensor 的解释方式。

## 二、模型：从 Token 与 Tensor 到 Logits

Tokenizer 将文本切成片段，再映射为整数 ID。片段可能是一个字、词的一部分或其他内容，与汉字不保证一一对应。Embedding 将 ID 换成 H 维向量，各 Token 的向量组成 hidden states。模型后面处理的 Tensor，就是带 shape 和 dtype 的多维数组。

一个 Tensor 的意义，要由内容、各维 shape 和 dtype 一起说明。

例如 `[T,H]` 常表示本轮 T 行、每行 H 个特征。它可能打包多个请求，也可能带 padding，具体由执行路径决定。同样是二维数组，也可能存权重或工作 buffer。

### Hidden States、Residual 与 Norm

Hidden states 是当前层里的 Token 表示，最初来自 Embedding，随后被各 Decoder Layer 变换。

Residual 将模块输出加回原表示。RMSNorm 按均方根归一化，再施加可学习缩放。Add+RMSNorm 等融合把相邻工作合到一起，可能省中间搬运或启动，具体覆盖范围随接口而变。

### Attention、Q/K/V 与 Head

Attention 从可见上下文中，为当前 Token 聚合信息。

经典形式先投影出下面三个量。

~~~text
Q = XWq
K = XWk
V = XWv
~~~

Q 提供查询，K 与它匹配，V 提供聚合内容。Q/K 分数经过缩放、mask、归一化，形成权重，再对 V 加权。因果模型只允许读取相应可见位置，未来内容被屏蔽。

Multi-Head 用不同投影子空间建模关联。head 的实际功能由训练形成，语法、实体等称呼只是局部观察或类比，不是每个 head 固定的职责。

MLA 改变 K/V 表示和缓存方式，以减轻 KV 压力，关系如下。

~~~text
Attention 结构
   ↓
决定历史状态怎么保存
   ↓
决定 KV Cache Layout
   ↓
影响显存与分布式布局
~~~

V4 另有专用滑窗与压缩历史，具体保存、读取方式在模型及 dsv4 backend 中实现。[deepseek_v4.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4.py) [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)

### MLP、GEMM 与 MoE

MLP/FFN 由线性与非线性计算组成，大量 Linear 由 GEMM 完成。名字中的 BF16、INT8 是精度，Grouped 是多组计算的组织方式，Expert 表示来自专家模块；工作量还要由矩阵形状和布局确定。

MoE 根据路由结果选择专家执行。

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

Router 选专家，EP 分配专家权重，DeepEP 等实现把输入送过去、把贡献合回来，Expert Runner 执行前馈。跨设备的 MoE 需要衔接这几部分。

因此 Router、重排、Dispatch/Combine 和 Expert GEMM 的耗时，需要分开定位。

### Logits 与 Sampling

Logits 是词表各候选的原始分数，还不是输出 Token ID。

Sampler 从分数决定 ID。Greedy 取最大值，temperature 调整分布刻度，top-k 限制候选数，top-p 按累计概率保留集合，具体组合顺序随实现而定。这里的 Top-K 筛词表，Router 的 Top-K 则选专家。

从输入到下一枚 ID 的关系如下。

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

词表分数处理与采样分别在 [logits_processor.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/logits_processor.py)、[sampler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/sampler.py)。

输出 ID 得到了，服务端还得保存状态、安排下一轮，并将结果交回请求。

## 三、运行时：生成阶段、缓存与调度

在线服务同时管理多条请求，选择本轮工作、划分 Token 预算、处理缓存不足，完成的请求退出后再接纳新任务。

推理 Runtime 管理的就是这些跨轮状态和执行安排。

### Prefill 与 Decode

普通自回归生成包含下面两个阶段。

Prefill 处理已知 Prompt，一次可计算较多新 Token，较大的矩阵通常更容易利用计算单元。

Decode 逐轮续接输出。普通单请求每轮新增输入少，还要读取历史，常见开销落在缓存、带宽、小 GEMM、启动或通信延迟上。

性能结果里，几个常见指标分别描述这些时间与产出。

- TTFT：发请求到收到首 Token，客户端口径常含排队、网络、输入处理；
- TPOT：首 Token 后的平均逐 Token 时间，公式需看工具，并和 ITL 区分；
- Throughput：单位时间的请求或 Token 数，需注明输入、输出或合计。

并发提高，总吞吐可能上升，每条请求的等待也可能变长，比较时两边都要保留。

### KV Cache 与 Page

KV Cache 保存 Attention 的历史状态，让生成下一位置时不必重算此前全部 K/V。

上下文变长、并发增加，通常都需要更多状态，实际增长由缓存结构决定。

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

Runtime 常按 Page/Block 管理缓存池，用映射连接逻辑序列和物理空间，请求不必各自绑定一块大连续内存。

ReqToTokenPool 记录请求与位置关系，KVCache 提供存储抽象，对应入口为 [memory_pool.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/memory_pool.py)。

这部分承担分配、寻址与回收，职责与后端内存管理相近。

### Prefix Cache 与 RadixCache

两条请求共享长系统前缀、后缀不同，前缀 KV 仍驻留且匹配条件满足时，第二条可以复用，只算后缀。Token 序列、命名空间和条目生命周期都得匹配，文本相同还不足够。

RadixCache 是 SGLang 具体的前缀缓存数据结构，入口在 [radix_cache.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/mem_cache/radix_cache.py)。

相近名字在不同语境里，可能指设计，也可能指一个代码对象。

- RadixAttention 常用于整体 prefix reuse 设计的说明；
- RadixCache 是固定 Runtime 的具体前缀缓存数据结构。

定位实现时，这两种含义需要对应到具体名称。

### Req、Scheduler、ScheduleBatch、ForwardBatch、ModelRunner

这五个对象把一条持续的请求，接到多轮设备执行。

Req 保存跨轮的输入、输出、采样、长度、缓存、停止条件和 metadata。

Scheduler 从等待与运行集合里选本轮请求，同时考虑 Token 预算、KV 容量和各阶段状态。

ScheduleBatch 是本轮计划，记录选中请求、输入区间和缓存安排。

ForwardBatch 将输入 Tensor、positions、seq_lens、cache locations 交给模型。seq_lens 要连同对象与阶段解释，它会影响位置和 KV 准备；Attention metadata 的具体准备时机由 backend 决定。

ModelRunner 把输入交到模型，协调 backend 与 Eager、Graph 等执行方式。

它们通过下面这条循环衔接。

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

对应源码入口如下。

- Req / ScheduleBatch：[schedule_batch.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/schedule_batch.py)
- Scheduler：[scheduler.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/managers/scheduler.py)
- ForwardBatch：[forward_batch_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/forward_batch_info.py)
- ModelRunner：[model_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/model_executor/model_runner.py)

Req 跨轮保留进度，ScheduleBatch、ForwardBatch 分别服务本轮计划和这一次执行。

投机与 Overlap 中，逻辑输出、KV 预留、正式提交会在不同时刻更新，各对象要维持这些状态的对应。

### Continuous Batching、Chunked Prefill、Graph 与 Overlap

Continuous Batching 让完成请求逐轮退出、新请求逐轮加入，不用等固定一批全部生成完。

Chunked Prefill 把长输入分段，控制单轮预算，给 Decode 留出调度机会，是 Runtime 的一种策略。

Kernel 很小时，Host 重复提交的成本更突出。Graph Capture/Replay 复用执行安排，尝试减少重复启动，NPU 入口为 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py)。

Overlap 让独立计算、通信、结果处理或准备重叠推进。它也可能增加争用、buffer 和同步，是否划算取决于实测。

批次填充、工作分块、减少重复提交、重叠独立任务，各自节省和新增的时间不同。

### PD Disaggregation

PD Disaggregation 把 Prefill、Decode 放在不同实例，分别调度。P 将输入阶段算好的 KV 和请求状态交给 D，后续生成从这些状态继续。

ownership 描述谁持有、修改、释放状态，handoff metadata 传请求身份、长度与位置。目标预分配、内存注册、传输、恢复都是新增成本，要与隔离收益一起评估。

P 侧与 Ascend transfer 分别见 [disaggregation/prefill.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/prefill.py)、[disaggregation/ascend/transfer_engine.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/disaggregation/ascend/transfer_engine.py)。

状态的写入完成、可读时刻和释放方，决定接续能否正确进行，发送、接收日志只是其中的部分记录。

## 四、分布式：Rank、TP、DP Attention 与 EP

模型放不下，或者服务吞吐超过单设备能力，存储和计算就需要分到多卡。

并行策略规定怎样分工，Rank/Group 标识谁参与，collective 和 backend 完成实际交换。

### Rank、World Size 与 Process Group

Process Group 是通信集合，Rank 是组内编号，World Size 是成员数量。Global Rank 通常指默认全局组，TP、EP Rank 则各有子组语境，world size 也随所指的组解释。

即使一进程一卡，设备编号与 Rank 也不等价。多机的本地设备号可以重复，全局 Rank 用来区分进程。

同一进程可以属于几个组。

~~~text
Global Rank = 6

TP Group = [4, 5, 6, 7]
→ TP Rank = 2

DP Group = [2, 6, 10, 14]
→ DP Rank = 1
~~~

例中全局 6 在 TP 组排第三、DP 组排第二，局部编号为 2、1。真实成员由配置确定，Draft、Verify 编号也需要带组名；`attn_tp_group` 还要看有序成员和当前 Attention 坐标。

这些组在 [parallel_state.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/parallel_state.py) 初始化。

### TP：Tensor Parallelism

TP 将同一层 Tensor、GEMM 分给多个 Rank 协作。

例如 Y=XW，W 沿某一维切成 shards，各 Rank 本地计算，再通过 collective 形成后续需要的结果。

这些 Rank 服务同一请求、同一层，分片计算也引入了相应通信。

### DP 与 DP Attention

普通 DP 建完整模型副本，各 Replica 接不同请求，提高整套服务吞吐。

DP Attention 则在同一个模型并行 world 内，划分 Attention 请求与 KV。

固定版本由 Runtime Context 派生下面这些宽度。[runtime_context.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/runtime_context.py)

~~~text
attn_dp_size
attn_cp_size
attn_tp_size
~~~

DPA 开启后，配置给出 Attention DP，组内 Attention TP 再结合总宽和 CP 计算。

例如下面的配置。

~~~text
tp_size = 16
dp_size = 8
enable_dp_attention = true
~~~

它描述的成员与分工，还包括这些关系。

这组参数要落实为 Attention DP 成员与请求、KV 的归属，组内 Attention TP 给出协作宽度。到了 MoE，同一批 Rank 再按专家布局重新组织。

对应的 per-process metadata 在 [dp_attention.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/dp_attention.py)。

### EP：Expert Parallelism

EP 把不同专家权重放到不同 Rank，每卡不必复制全部专家。

~~~text
Rank 0 → Experts A/B
Rank 1 → Experts C/D
Rank 2 → Experts E/F
...
~~~

Token 选中远端专家，就把输入送到权重所在 Rank，算完再取回贡献。

Router 到 Expert 的模型计算，因而展开成下面的设备协作。

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

专家并行层入口为 [ep_moe/layer.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/ep_moe/layer.py)。

### DeepEP 与专家计算的边界

DeepEP 承担专家并行的 Dispatch 和 Combine，专家矩阵乘另由计算实现执行。

Dispatcher 的具体实现有 [deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)、[deepep_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py)。

Ascend 专家计算的 Runner 在 [moe_runner/ascend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/moe_runner/ascend.py)。

分发与计算的对应关系如下。

~~~text
DeepEP
→ Token 怎么跨 Rank 到 Expert

Ascend MoE Runner
→ 到了本 Rank 后 Expert GEMM 怎么算
~~~

Dispatch 时间还包括重排、传输和跨 Rank 等待。通信若已被其他工作覆盖，缩短其总时长，未必等量缩短整个周期。

### Collective：AllReduce、AllGather、ReduceScatter、All-to-All

几种常见原语的操作对象不同。

- AllReduce 合并局部贡献，并将归约结果交给各 Rank；
- AllGather 收齐各 Rank 的分片；
- ReduceScatter 合并贡献后分别交付结果区间；
- All-to-All 按目的地交换各自的数据。

常见 AllReduce 算法可用 ReduceScatter+AllGather 解释，但这种语义分解不保证底层直接调用这两个 API。

HCCL 是 Ascend collective library，承接相应分布式调用。

RDMA 是更下层的远程内存与跨机搬运能力，TP、EP 则规定计算分工，处在不同层次。

这些层次的关系如下。

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

PP 沿模型深度分层，各 stage 执行不同层。

CP 沿上下文分布数据，Attention 可能需要访问其他 Rank 的历史，协作不止切开 Token 数组。

SP 也涉及 sequence，经典形式常用于 TP activation 分片以节省内存，与 CP 的上下文协作不同。

具体到框架，还要结合 Group、本地 Tensor 与交换方式确定含义。

## 五、投机解码：Draft、Verify 与 Accept

普通 Target Decode 每次完整前向，通常只推进较少输出。

投机先用较轻的路径提一组候选，再让 Target 验证。

候选到正式输出经过下面几步。

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

Draft 是提出候选的角色，可以来自独立模型，也可以来自大模型附带的模块。

常见提议路径如下。

- 独立 Draft Model；
- MTP / NextN Head；
- EAGLE 风格 Draft；
- DSpark Head；
- N-gram 等非神经路径。

V4 可以使用 Checkpoint 自带 DSpark head，部分 Ascend CI 也显式提供 draft model path。两种都承担提议，权重组织却不同。

独立权重、独立实例与额外 head 是不同安排，选择由算法和 Checkpoint 决定。

### Verify 与 Accept

Verify 计算候选位置的 Target 结果，Accept 按算法选有效历史。在此之前，候选还不是正式输出。

接受结果也会更新下面这些状态。

- Sequence Length；
- Position；
- KV Cache；
- Draft/Verify Metadata；
- 下一轮 Draft State。

候选写入的临时状态，还不等于已经提交的历史。

Commit 确认有效输入状态，Compact 整理成下游布局，Rollback 避免拒绝状态继续生效，Reclaim 回收空间。对象与执行顺序由算法和缓存实现决定。

### 接受长度怎样与周期成本一起决定收益

周期成本与输出推进的关系如下。

~~~text
有效接受的 Token 数
÷
Draft + Verify + KV + Scheduler + Communication 的总成本
~~~

接受长度通常反映周期输出推进，严格接受率则要有实际接受、提出的候选计数，并说明 bonus 的处理。推进多长和周期花多久，一起决定收益。

Draft、宽 Verify、KV 搬运或同步开销增加太多，投机也会慢过普通 Decode。

### DSpark 在源码里从哪里看

DSpark 的数据结构、执行组织和模型分别在下列入口。

1. [spec_info.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/spec_info.py) 定义 speculative metadata；
2. [dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) 组织 Draft / Target / Verify；
3. [deepseek_v4_dspark.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4_dspark.py) 实现 V4 DSpark 的模型 forward。

多卡执行还牵涉这些分工。

Draft 有自己的 TP/DP/EP 布局，Verify 使用相应 Group，Accept 需要确定同步参与者，KV 有坐标和所有权；两种布局之间还要完成转换。

支持一种配置，需要这些条件一起成立。

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

参数放行之后，算法、Runtime、内存与通信的状态仍必须衔接，DSPARK 从候选到 Verify、Accept、Commit 都有自己的更新时刻。

## 六、Ascend：Backend、CANN、Kernel 与量化

这些计算最后由 Backend、算子和 Kernel 接到 Ascend 910C。

### Backend：同一高层语义的具体实现选择

推理框架常见的 Backend 选择有下面几类。

- Attention Backend；
- MoE A2A Backend；
- MoE Runner Backend；
- Quantization Backend。

Backend 将高层计算落实为布局、算子与执行分支。

设备、模型、精度和阶段变了，实际选中的分支也可能变，性能分析需要落到运行的实现。

例如 Attention 描述计算语义，dsv4 给出相应平台的实现路径。

注册入口与 NPU 实现分别为 [attention_registry.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/attention/attention_registry.py)、[ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)。

### Operator 怎样映射到 Kernel

高层 Operator 到设备 Kernel，调用会经过这些层次。

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

两者并不一一对应。

一个 Operator 可以启动多个 Kernel，相邻高层操作也可以融合。shape、dtype、mode 会选择不同 Kernel，调用周围还可能有 Host 准备、workspace 或通信。

Attention 耗时取决于下面这些实际执行条件。

实际 dispatch 分支和 shape、dtype 决定输入，Tiling 选择具体执行安排，计算、搬运限制以及额外布局转换共同构成开销。

### CANN、Tiling 与 TilingKey

CANN 向上承接算子接口，向下组织 Ascend 执行。

高性能或自定义算子常经过下面这条路径。

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

片上高速空间有限，大 Tensor 通常分成 Tile，分批搬入、计算、写回。

Tiling 决定这些工作的安排。

每块多大、各 Core 做哪部分、循环几次、Local Buffer 怎么用，以及计算和搬运怎样衔接，都属于 Tiling。

TilingKey 用于选相应形状、精度、模式的 Kernel 分支，读到 key 还需要对应到具体设备实现。

同名算子因此也可能执行不同的内部路径。

### Stream、Graph 与异步执行

Host 提交任务后，设备通常在 Stream 上异步执行，API 返回时计算未必已经完成。

时间线里会分别出现这些区间。

- Host API 时间；
- Kernel 执行时间；
- Stream Synchronization；
- Compute / Communication overlap。

NPU Graph 复用安排，尝试减少重复 Launch 和 Host 调度，入口为 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py)。

### BF16、W8A8 与 ModelSlim

BF16 是 16-bit 浮点，相比 FP32 少占字节，硬件与实现支持良好时可能提高吞吐，但收益并不保证与位宽成比例。

W8A8 说明权重、激活都是 8-bit，完整量化方案还包含粒度、scale 和计算路径。

同样的位宽，下面这些条件仍可能不同。

- 权重的 per-tensor/channel/group 粒度；
- 激活静态或动态；
- 是否 per-token；
- 对称或非对称；
- scale、zero point 格式；
- 排除层；
- Dense、MoE Kernel。

因此比较 W8A8，需要对齐格式和实际路径。

ModelSlim 入口在 [modelslim.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/quantization/modelslim/modelslim.py)，各 Scheme 位于相应目录。

一层的权重、激活、scale 最后对应到算子实参，量化、矩阵乘和转换各自在这条路径上花时间。

## 七、按工作任务继续阅读

这些文章把相应术语放回实际请求，按正在看的问题选择即可。

### 从一次请求开始

请求到设备的过程，分在这三篇中。

1. [SGLang 请求生命周期](../sglang/sglang-ascend-request-lifecycle.md)；
2. [SGLang Scheduler](../sglang/sglang-scheduler-source-analysis.md)；
3. [SGLang ModelRunner](../sglang/sglang-modelrunner-source-analysis.md)。

它们分别跟踪接入、组批和模型执行。

### 如果你在看 Attention / KV Cache

缓存读取的具体实现见下文。

- [DeepSeek-V4 Attention 源码解析](../sglang/deepseek-v4-attention-source-analysis.md)

Query 怎样写缓存、沿长度和位置读取历史，最后传哪些参数给 Ascend backend，都在这篇里展开。

### 如果你在看 MoE / TP / DP / EP

路由与层间并行分别见这两篇。

1. [DeepSeek-V4 MoE：从路由到专家计算](../sglang/deepseek-v4-moe-source-analysis.md)
2. [DeepSeek-V4 分布式 Forward](../sglang/deepseek-v4-distributed-parallel-source-analysis.md)

前一篇从选专家走到输出，后一篇说明同一批 Rank 怎样切换 Attention、MoE 的数据布局与通信组。

### 如果你在看 DSpark / 投机解码

投机的主过程与状态管理，分别见这些专题。

1. [DeepSeek-V4 Speculative Decoding 总链路](../speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)
2. [DeepSeek-V4 DSpark Parallel Layout](../distributed/deepseek-v4-dspark-parallel-layout-analysis.md)
3. [SGLang Speculative Scheduler](../speculative-decoding/sglang-speculative-decoding-scheduler-source-analysis.md)
4. [Accept Decision](../speculative-decoding/deepseek-v4-accept-decision-source-analysis.md)
5. [KV Commit / Compact / Rollback](../speculative-decoding/deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md)
6. [多卡一致性](../distributed/deepseek-v4-speculative-distributed-consistency-source-analysis.md)
7. [Speculative 性能模型](../performance/deepseek-v4-speculative-performance-analysis.md)

候选生成和树结构还包括下面几篇。

- [MTP / NextN](../speculative-decoding/deepseek-v4-mtp-speculative-decoding-source-analysis.md)
- [EAGLE Draft Tree](../speculative-decoding/deepseek-v4-eagle-draft-tree-source-analysis.md)
- [Tree Attention](../speculative-decoding/deepseek-v4-tree-attention-source-analysis.md)

### 如果你在看 PD 分离

PD 的跨实例接续见下文。

- [DeepSeek-V4 Prefill / Decode Disaggregation 源码解析](../sglang/deepseek-v4-pd-disaggregation-source-analysis.md)

其中展开了 KV、state 如何从 Prefill 交到 Decode。

### 把术语放回正在排查的问题

把这些名词连到同一条数据流上，是下面的关系。

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

例如一次 Decode 变慢，当前请求的上下文、历史访问、参与 Group 和设备时间线，需要对应到同一轮工作。这里给出词义与入口，具体行为仍以固定源码和当前运行记录为准。
