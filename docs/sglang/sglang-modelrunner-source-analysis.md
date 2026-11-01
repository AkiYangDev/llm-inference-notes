# SGLang ModelRunner：九个输入 Token 怎样产生两行 Logits

一轮 Prefill 送入九个新 Token，模型层算出九行 hidden states，到了 LM Head 却只剩两行。这是两条请求的普通生成，没有请求输入 Logprob，各自只用末端位置预测下一枚 Token。

九行怎么打包、历史从哪里读取、末端两行怎样选出来，都要靠 Scheduler 带下来的执行信息。ModelRunner 接过计划，准备输入、选择执行方式，再运行模型与采样。Prefill 和普通 Decode 进入同一份模型代码，形状与执行元数据随阶段改变，组批过程见 [上一篇 Scheduler](sglang-scheduler-source-analysis.md)。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 本文固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线是普通文本生成、非投机、PP=1，沿容易观察的 eager execution 阅读；Decode Graph、Prefill Graph 只追到 ModelRunner 的分发处。模型使用公开实现 `DeepseekV4ForCausalLM`，Ascend 路径追到能够确认的 `DeepseekV4AscendAttnBackend` 接口，不能据此断定某个部署镜像的底层轨迹。`cuda_graph` 是框架接口命名，NPU 图执行仍需看对应平台实现。

</details>

## ForwardBatch 里的两套长度

沿用两条教学请求，用整数代替 Token ID。前缀长度只演示坐标关系，实际匹配粒度由 DeepSeek-V4 专用缓存实现决定。

| 请求 | 完整输入 | 已有可复用前缀 | 本轮新增输入 | 新增输入的逻辑位置 |
| --- | --- | --- | --- | --- |
| R1 | `[11,12,13,14,15,16]` | 无 | `[11,12,13,14,15,16]` | `[0,1,2,3,4,5]` |
| R2 | `[21,22,23,24,25,26]` | 前三个 Token | `[24,25,26]` | `[3,4,5]` |

两条请求的完整长度都是六，R1 要新算六行，R2 只算后三行。R2 的前三个位置已经有缓存状态，本轮输入里不用再放一遍。

普通生成的 Worker 调用 `ForwardBatch.init_new(batch, model_runner)`，把 Scheduler 计划整理成执行输入，再送给 `model_runner.forward(forward_batch)`。[`TpModelWorker.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/managers/tp_worker.py#L631-L749)

Token、模式、长度、位置、缓存映射和采样信息集中到 `ForwardBatch`。其中有些 Tensor 借用自 `ScheduleBatch`，有些是执行层补出的派生数据，没有将上游所有状态深拷贝一份。[源码：ForwardBatch 字段与构造](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L468-L605)

在这两条请求里，各字段取值如下。

```text
input_ids = [11,12,13,14,15,16,24,25,26]    # 9 行
positions = [ 0, 1, 2, 3, 4, 5, 3, 4, 5]    # 各自序列中的位置

extend_prefix_lens = [0,3]    # 已复用的前缀长度
extend_seq_lens    = [6,3]    # 本轮新增长度
seq_lens           = [6,6]    # 两条请求的当前总长度
extend_num_tokens  = 9
```

每条请求有自己的 positions。Torch fallback 分别构造 `arange(prefix_len, prefix_len + extend_len)`，再把区间打包；所以 R2 从位置三开始，不会接在 R1 的位置五后面继续数。[`compute_position_torch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L2032-L2046)

物理写入另有映射，`out_cache_loc` 指向新增缓存的槽位，`req_pool_indices` 定位映射池中的请求。DeepSeek-V4 NPU 还可通过 `out_cache_loc_dsv4` 传专用池的位置集合。后端结合这些映射和 positions，确定每行属于谁、读哪段历史、结果往哪里写。[源码：DSV4 ForwardBatch 字段](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L487-L505)

九个 Token 虽连续打包，长度、位置和 Attention 元数据仍保留各自边界，R2 读取的是本请求的历史。

<a id="modelrunner-graph-eager"></a>

## Graph 或 Eager，先走哪条分支

Graph Replay 复用捕获好的执行结构，Eager 沿当前调用运行前向。能否回放，要查 Runner 是否存在、模式与形状是否满足条件，普通 Decode 也可能走不到图。

`ModelRunner.forward()` 处理 tracing、专家计数等外围工作，再进入 `_forward_raw()`。下面省去无关分支，只展示 [`ModelRunner._forward_raw()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1849-L1947) 的分派顺序。

```text
检查 Decode Graph 是否可运行
  ├─ 可以：执行图并返回
  └─ 不可以：准备 live batch 的 padding、DP / MLP-sync 等状态
              ├─ Split Prefill：进入专门路径
              ├─ 满足 Prefill Graph 条件：执行对应图
              └─ 其余：交给 EagerRunner
```

Decode Graph 命中会提前返回。没有命中的 live batch 才经过 `_prepare_eager_forward_batch()`，随后还有 Split Prefill、Prefill Graph 和最终 Eager 等分支。这个名字带 eager 的准备函数，实际也在 Prefill Graph 之前。

源码用 `cuda_graph` 命名图接口，Ascend 路径同样沿用它，具体执行仍由 NPU 平台实现承接，下文统一称 Graph Replay。

Eager 的输入也要装载。默认 `EagerRunner.load_batch()` 先填入固定最大容量的静态缓冲，再取本轮大小的视图；设置 `SGLANG_EAGER_INPUT_NO_COPY` 才绕过这类复制。因此两种方式的比较还包括输入准备，单凭这里的调用关系不能判断快慢。[`EagerRunner.load_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L183-L211)

以下沿普通 Eager 进入模型。Graph Replay 复用捕获结构，不会每轮重走相同的 Python 调用序列。

## Prefill 与 Decode 共用模型入口

`EagerRunner.execute()` 根据 `forward_mode`，把普通 Decode 交给 `_execute_decode()`，Extend / Prefill 交给 `_execute_extend()`。Attention 元数据需要分别准备，Prefill 还可能带额外参数、涉及 Context Parallel。[`EagerRunner.execute()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L213-L229)

准备完成，两个普通生成分支调用同一个入口。

```python
# 省略其他 kwargs，只保留两条路径共同传入的参数。
model_runner.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
    **kwargs,
)
```

Prefill 可带多行新增输入，普通 Decode 通常每请求带一行，历史和位置由 `ForwardBatch`、后端元数据传入。同一份 DeepSeek 模型因而能接两种阶段的输入。[`EagerRunner._execute_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L244-L271) [`EagerRunner._execute_extend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L273-L378)

| 本轮执行信息 | 普通 Prefill / Extend | 普通非投机 Decode |
| --- | --- | --- |
| 每条请求新增 Token | 可以有多个 | 通常一个 |
| positions | 一段或多段区间 | 当前末尾位置 |
| 历史状态 | 可能已有前缀缓存 | 通常已经有多轮历史 |
| Attention 元数据 | 按 Prefill 查询组织 | 按 Decode 查询组织 |
| DeepSeek-V4 模型入口 | 同一个 `forward()` | 同一个 `forward()` |

`MIXED` 还有平台条件。在固定版本中，非 NPU 且无 CP 时可转成 Extend；NPU 保留给专用处理，CP 路径也保留 MIXED。

九行接着进入 DeepSeek-V4，`ForwardBatch` 继续随调用传下去。hidden states 本身没有记下各行的请求边界和历史位置，Attention 还得读取执行元数据。

<a id="token-deepseek-v4"></a>

## DeepSeek-V4 的九行隐状态

本例 PP=1，Embedding、Decoder Layers 和末端 LogitsProcessor 在同一条路径上，流水线并行还会增加 stage 交接。`DeepseekV4ForCausalLM.forward()` 先进入 `DeepseekV4Model.forward()`。[`DeepseekV4ForCausalLM.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5081-L5147)

九个 ID 经 Embedding 得到 `[9,H]`，再扩为 `[9,hc_mult,H]`，存多路 mHC 状态。`hc_mult` 是状态流数量，和 Attention head 数不同。层栈计算后，模型组合这些状态，完成末端归一化。[`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4630-L4840)

```text
input_ids [9]
    ↓ Embedding
初始表示 [9,H]
    ↓ 扩展为多路 mHC 状态
层间状态 [9,hc_mult,H]
    ↓ Decoder Layers：Attention、MoE 与 mHC 处理
    ↓ 最终组合与 Norm
末端表示 [9,H]
```

图跟踪行数，层内投影和布局还会变化。每层改变 hidden states 的表示，positions 与 `ForwardBatch` 继续保留请求、序列位置、缓存的对应关系。

例如 Query 进入 `DeepseekV4AscendAttnBackend.forward()` 后，后端读取准备好的长度、页表，按本层 `compress_ratio` 选择滑窗或压缩历史。写新增状态时，`store_cache()` 从当前映射定位 SWA buffer。[`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189) [源码：DSV4 cache write](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2191-L2219)

R2 送来三行，`seq_lens`、positions、请求索引与缓存位置仍让这三行访问自己已有的前三个位置。那些历史省去了重算，仍留在缓存里。

模型前向完成，九行输入得到九行 hidden states。用户的新 Token 还没采样，词表计算要先从这九行里选末端。历史读取细节见 [DeepSeek-V4 Attention](deepseek-v4-attention-source-analysis.md)。

<a id="hidden-states-logits"></a>

## LM Head 为什么只需要两行

普通生成给每条请求预测下一个位置。本轮打包 Tensor 从零编号，R1 的末端是行 5，R2 是行 8，采样需要的是这两处。

普通 Prefill 没有请求输入 Logprob 时，`LogitsProcessor` 先取末端 hidden states，再交给 LM Head。前面的模型层已经计算了九行，裁剪发生在词表分数计算之前。[`LogitsProcessor.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L495-L597)

把长度代入 [`LogitsProcessor._get_pruned_states()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L599-L665) 的末端选择逻辑。

```python
# 普通 Prefill、没有输入 logprob 的对应分支。
last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
pruned_states = hidden_states[last_index]
```

`extend_seq_lens=[6,3]` 累计得 `[6,9]`，减一得到 `[5,8]`。累计的是打包的新输入行数。R2 虽总长六，本轮只占三行，定位它在打包 Tensor 中的末端，就得用这三行。

```text
打包行号：  0  1  2  3  4  5 | 6  7  8
所属请求： R1 R1 R1 R1 R1 R1 | R2 R2 R2
选中行号：                5 |       8

模型末端 [9,H] → 选择末端行 [2,H] → LM Head → 逻辑 Logits [2,V]
```

V 是词表维度，`next_token_logits` 的逻辑接口为 `[#seq,vocab_size]`，两条请求对应两行待采样分数。多卡时各 Rank 是否持有完整词表，还取决于具体分布方式。[源码：LogitsProcessorOutput](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L201-L240)

请求输入 Logprob 时，需要为更多输入位置评分；Target Verify 等模式另有分支。这里九行裁成两行，对应的是普通生成的采样需求。

Worker 得到 `LogitsProcessorOutput`，再进入 `ModelRunner.sample()`。先施加 logits bias、grammar mask 等约束，再结合 `sampling_info` 调 Sampler，词表分数到这时才变成 Token ID。重叠调度还可能延后采样，所以 forward 返回、采样完成、CPU 取得结果会落在不同时刻。[`ModelRunner.sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1977-L2031)

R1、R2 各得到一枚输出，接着检查停止条件。首 Token 已经在 Prefill 末端采样出来，不用额外做一次普通 Decode 才能返回。

<a id="decode"></a>

## T1、T2 接回下一轮 Decode

假设两条请求继续，刚采样的结果记为 T1、T2，下一轮只计算它们，不重算前面的九行 Prompt。两条序列都已处理到长度六，这次各计算位置六，设备侧 `seq_lens` 推进到七。

| 字段或结果 | 本轮 Prefill | 下一轮普通 Decode |
| --- | --- | --- |
| 请求数 | 2 | 2 |
| 本轮输入行数 | R1 六行、R2 三行，共九行 | T1、T2 各一行，共两行 |
| positions | `[0,1,2,3,4,5,3,4,5]` | `[6,6]` |
| `seq_lens` | `[6,6]` | `[7,7]` |
| 模型末端表示 | `[9,H]` | `[2,H]` |
| 用于下一 Token 采样的行 | 每条请求选择最后一行 | 每条请求本来就只有一行 |
| 逻辑 next-token logits | `[2,V]` | `[2,V]` |

刚采样出 T1、T2，它们还没作为输入走过模型。等下一轮执行，才有自己的缓存状态，同时预测再下一枚 Token。因此输出进度比已计算的缓存进度先走一步。

九行 Prefill、两行 Logits、两行 Decode 输入，来自同一组请求在不同阶段的工作。新增行、逻辑位置、缓存槽位和采样行依次对应，请求索引则一直把它们连到各自的历史。

<a id="_1"></a>

### ModelRunner 源码入口

| 目标 | 固定版本入口 |
| --- | --- |
| `ScheduleBatch -> ForwardBatch` | [`TpModelWorker.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/managers/tp_worker.py#L631-L749) |
| `ForwardBatch` 构造 | [`ForwardBatch.init_new()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L847-L1073) |
| Prefill positions | [`compute_position_torch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L2032-L2046) |
| ModelRunner 入口 | [`ModelRunner.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1703-L1799) |
| Graph / Eager 分发 | [`ModelRunner._forward_raw()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1849-L1947) |
| Eager batch 装载 | [`EagerRunner.load_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L183-L211) |
| Eager 模式分发 | [`EagerRunner.execute()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L213-L229) |
| Eager Decode | [`EagerRunner._execute_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L244-L271) |
| Eager Prefill | [`EagerRunner._execute_extend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L273-L378) |
| DeepSeek-V4 主体 | [`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4630-L4840) |
| DeepSeek-V4 CausalLM 入口 | [`DeepseekV4ForCausalLM.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5081-L5147) |
| Logits 生成 | [`LogitsProcessor.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L495-L597) |
| Prefill hidden-state pruning | [`LogitsProcessor._get_pruned_states()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L599-L665) |
| Sampling | [`ModelRunner.sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1977-L2031) |
| DeepSeek-V4 Ascend Attention 边界 | [`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189) |
