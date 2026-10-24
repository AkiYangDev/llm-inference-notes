# SGLang ModelRunner：九个输入 Token 怎样产生两行 Logits

两条请求各有六个输入 Token，其中一条已经命中三个 Token 的前缀缓存。这一轮模型需要处理九个新位置，最后却只要为两条请求各选出一枚输出。九行输入怎样变成两行用于采样的 Logits，是读懂 ModelRunner 的一个入口。

在[上一篇 Scheduler](sglang-scheduler-source-analysis.md)里，调度器已经决定本轮计算哪些位置。这里接过这份计划，沿着输入准备、执行路径、模型前向和采样走下去。普通 Prefill 与 Decode 会进入同一个模型，区别体现在输入形状和执行元数据中。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线采用普通文本生成、非投机推理、PP=1，并以 eager execution 作为最容易观察的代表路径；Decode Graph、Prefill Graph 只解释到 ModelRunner 的分发边界。DeepSeek 案例使用公开源码中的 `DeepseekV4ForCausalLM`。Ascend 部分只追到可确认的 `DeepseekV4AscendAttnBackend` 接口，不把公开源码路径外推成具体部署镜像的底层运行轨迹。源码中的 `cuda_graph` 命名是框架内部接口名，本文不据此把 NPU 图执行等同于 CUDA。

</details>

## 一、ForwardBatch 描述本次执行

先把两条请求固定下来。以下是教学用的整数 Token 示例，用来观察新增行与序列位置；前缀命中长度不代表 DeepSeek-V4 专用缓存一定支持这样的匹配粒度。

| 请求 | 完整输入 | 已有可复用前缀 | 本轮新增输入 | 新增输入的逻辑位置 |
| --- | --- | --- | --- | --- |
| R1 | `[11,12,13,14,15,16]` | 无 | `[11,12,13,14,15,16]` | `[0,1,2,3,4,5]` |
| R2 | `[21,22,23,24,25,26]` | 前三个 Token | `[24,25,26]` | `[3,4,5]` |

两条请求的总长度都是六，本轮新增长度却分别是六和三。执行侧需要同时知道这两组长度：一组描述请求已经推进到哪里，另一组决定这次要送入模型多少行。R2 的前三个位置不在本轮输入里，它们的历史状态仍由缓存提供。

Scheduler 保存这些安排后，普通生成路径由 Worker 调用 `ForwardBatch.init_new(batch, model_runner)`，再把得到的对象交给 `model_runner.forward(forward_batch)`。[`TpModelWorker.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/managers/tp_worker.py#L631-L749)

`ForwardBatch` 把本轮 Token、模式、长度、位置、缓存映射和采样信息组织到一起。它是一份单次执行描述，其中部分 Tensor 直接借用 `ScheduleBatch` 的字段，并非把上游状态全部深拷贝一遍。构造时还会补出执行层需要的派生信息。[源码：ForwardBatch 字段与构造](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L468-L605)

在这个例子中，可以把关键输入放在同一处看：

```text
input_ids = [11,12,13,14,15,16,24,25,26]    # 9 行
positions = [ 0, 1, 2, 3, 4, 5, 3, 4, 5]    # 各自序列中的位置

extend_prefix_lens = [0,3]    # 已复用的前缀长度
extend_seq_lens    = [6,3]    # 本轮新增长度
seq_lens           = [6,6]    # 两条请求的当前总长度
extend_num_tokens  = 9
```

普通 Token 输入的 positions 可以由每条请求的前缀长度与新增长度生成。Torch fallback 为每条请求构造 `arange(prefix_len, prefix_len + extend_len)`，再将这些区间打包。所以上面第二段 positions 从三开始，而不是接在 R1 的五后面继续计数。[`compute_position_torch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L2032-L2046)

缓存位置使用另一套坐标。`out_cache_loc` 说明新增状态写入哪些物理槽位，`req_pool_indices` 标识请求在映射池中的位置。DeepSeek-V4 的 NPU 路径还可以携带 `out_cache_loc_dsv4`，传递专用缓存池的位置集合。它们与 positions 配合，保留每行属于谁、历史在哪里、当前状态写到哪里的关系。[源码：DSV4 ForwardBatch 字段](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L487-L505)

把九个 Token 连续存进一个 Tensor，只是打包方式。请求长度、位置和 Attention 元数据仍保留序列边界，R2 不会因为排在 R1 后面就读到 R1 的历史。下一步，ModelRunner 用这份执行描述选择运行方式。

## 二、ModelRunner 选择 Graph 或 Eager 路径

同样一份逻辑输入，可以通过图回放执行，也可以在这一轮直接运行模型代码。Graph 复用已经捕获的执行结构；Eager 沿当前调用路径完成前向。能否走图，需要检查 Runner 是否存在、模式与形状是否满足要求，不能只看到 batch 是 Decode 就认定一定回放。

`ModelRunner.forward()` 处理 tracing、专家计数等外围工作后，进入 `_forward_raw()`。下面是省略无关分支的阅读顺序，不是可直接替换源文件的实现：[`ModelRunner._forward_raw()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1849-L1947)

```text
检查 Decode Graph 是否可运行
  ├─ 可以：执行图并返回
  └─ 不可以：准备 live batch 的 padding、DP / MLP-sync 等状态
              ├─ Split Prefill：进入专门路径
              ├─ 满足 Prefill Graph 条件：执行对应图
              └─ 其余：交给 EagerRunner
```

这段顺序还有一个容易忽略的含义：`_prepare_eager_forward_batch()` 虽然名字里有 eager，却位于 Split Prefill、Prefill Graph 和 Eager 分支之前。Decode Graph 提前返回后，剩余 live batch 先完成这层准备，再继续判断执行分支。读调用关系时应看函数所在的位置，不能只按名字划分作用范围。

源码中的 `cuda_graph` 是框架沿用的接口命名。在 Ascend 上，它也用于选择 NPU 的图执行路径，这里统一称为 Graph Replay；具体底层行为仍要对应平台实现。

Eager 也有输入装载成本。默认的 `EagerRunner.load_batch()` 会把当前 batch 填入固定最大容量的静态缓冲区，再切出本轮大小的视图；设置 `SGLANG_EAGER_INPUT_NO_COPY` 才会绕过这一类复制。因此，比较 Graph 与 Eager 时，除了模型本身，还要知道输入在哪里准备、发生了哪些复制。这里说明源码中的准备流程，不据此给出性能收益。[`EagerRunner.load_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L183-L211)

后面沿普通 Eager 路径观察模型计算。Graph Replay 的逐轮 Python 调用方式不同，不能把这里每个调用都当成回放时必然重新执行的步骤。

## 三、Prefill 与 Decode 怎样进入同一个模型

`EagerRunner.execute()` 先读取 `forward_mode`，把普通 Decode 交给 `_execute_decode()`，把 Extend / Prefill 交给 `_execute_extend()`。分流发生在本轮执行准备上：Decode 和 Prefill 需要的 Attention 元数据不同，Prefill 还可能处理额外的参数与 Context Parallel。[`EagerRunner.execute()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L213-L229)

准备完成后，两条普通生成路径都调用同一个模型入口：

```python
# 省略其他 kwargs，只保留两条路径共同传入的参数。
model_runner.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
    **kwargs,
)
```

对应代码分别在 [`EagerRunner._execute_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L244-L271) 和 [`EagerRunner._execute_extend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L273-L378)。Prefill 一次可以送入多行新增输入，普通 Decode 通常每条请求只有一行；模型通过 `ForwardBatch` 和后端元数据取得各自的历史与位置。两种阶段不需要各维护一套独立的 DeepSeek 模型函数。

| 本轮执行信息 | 普通 Prefill / Extend | 普通非投机 Decode |
| --- | --- | --- |
| 每条请求新增 Token | 可以有多个 | 通常一个 |
| positions | 一段或多段区间 | 当前末尾位置 |
| 历史状态 | 可能已有前缀缓存 | 通常已经有多轮历史 |
| Attention 元数据 | 按 Prefill 查询组织 | 按 Decode 查询组织 |
| DeepSeek-V4 模型入口 | 同一个 `forward()` | 同一个 `forward()` |

`MIXED` 还需要单独看平台条件。固定版本中，非 NPU 且没有 CP 时，可以把 Mixed batch 按 Extend 形态执行；NPU 保留 MIXED 给专用处理，CP 路径也保留它。这些分支说明执行模式要与后端一起理解，不能把一处转换套到所有平台。

回到九行输入。接下来进入 DeepSeek-V4 模型层，`ForwardBatch` 仍会沿调用链传下去：只有一份 hidden states，Attention 还不知道这些行的请求边界和历史位置。

## 四、从打包的 Token 进入 DeepSeek-V4

本文采用 PP=1，便于在同一条路径里观察 Embedding、Decoder Layers 和末端的 LogitsProcessor；采用流水线并行时，中间还会涉及 stage 间交接。`DeepseekV4ForCausalLM.forward()` 先调用模型主体 `DeepseekV4Model.forward()`。[`DeepseekV4ForCausalLM.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5081-L5147)

Embedding 把整数 Token ID 映射为向量。九个输入对应 `[9,H]`，其中 H 是隐藏维度。DeepSeek-V4 随后把状态扩展为 `[9,hc_mult,H]`，保留多路 mHC 表示；`hc_mult` 表示状态流数量，与 Attention 的 head 数是两件事。各层计算后，模型再组合这些状态并执行末端归一化。[`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4630-L4840)

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

这是理解行数变化的示意，层内张量还有各自的投影与布局。在整个层栈里，positions 和 `ForwardBatch` 与 hidden states 一起传递。模型计算改变每行的表示，元数据则让后端知道每行对应的序列位置、请求和缓存。

以 `DeepseekV4AscendAttnBackend.forward()` 为例，后端接收 Query，并读取已准备的长度、页表等信息。它按当前层的 `compress_ratio` 选择滑窗或压缩历史路径；写入新增状态时，`store_cache()` 根据本轮缓存映射定位 SWA buffer。[`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189) [源码：DSV4 cache write](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2191-L2219)

所以前面准备的 `seq_lens`、positions、请求索引和缓存位置会影响实际计算。R2 本轮只有三个新增输入，但它们仍能读取该请求已有的前三个位置；这些历史并没有因为不出现在本轮 `input_ids` 中就消失。

Attention 内部的 Query、滑窗和压缩缓存怎样配合，可以继续看 [DeepSeek-V4 Attention](deepseek-v4-attention-source-analysis.md)。这里先跟住模型末端：九行输入完成前向后，得到九行 hidden states，此时还没有为用户选出 Token。

## 五、九行 hidden states 怎样得到两行 Logits

模型为九个新增位置完成了计算，普通生成却只需要每条请求下一位置的预测。按零起始索引，R1 最后一个输入对应打包 Tensor 的行号 5，R2 对应行号 8。

在没有请求输入 Logprob 的普通 Prefill 路径中，`LogitsProcessor` 先选择这些末端行，再把它们交给 LM Head。这个次序意味着前面的模型层仍处理了九行，只在准备词表分数时裁掉不需要采样的位置。[`LogitsProcessor.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L495-L597)

源码的选择方法可以直接代入例子：[`LogitsProcessor._get_pruned_states()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L599-L665)

```python
# 普通 Prefill、没有输入 logprob 的对应分支。
last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
pruned_states = hidden_states[last_index]
```

本轮 `extend_seq_lens=[6,3]`，累计后是 `[6,9]`，减一得到 `[5,8]`。注意这里累计的是本轮新增行数：R2 的完整序列长度虽然也是六，本轮却只有三行打包输入。若拿总长度去找末端，就会把逻辑序列位置与本轮 Tensor 行号混在一起。

```text
打包行号：  0  1  2  3  4  5 | 6  7  8
所属请求： R1 R1 R1 R1 R1 R1 | R2 R2 R2
选中行号：                5 |       8

模型末端 [9,H] → 选择末端行 [2,H] → LM Head → 逻辑 Logits [2,V]
```

V 表示词表维度。`next_token_logits` 的接口语义是 `[#seq,vocab_size]`；上面的逻辑形状说明两条请求各有一行待采样分数，不用于推断多卡上每个 Rank 都持有完整词表。[源码：LogitsProcessorOutput](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L201-L240)

这里有两个适用条件。请求输入 Logprob 时，还需要为其他输入位置评分，不能只保留两行；Target Verify 等模式也有各自的选择分支。因此，“九行变两行”描述的是本例普通生成所需的采样位置，并非所有模型输出都会按请求数裁剪。

Logits 是词表分数，仍不是 Token ID。普通生成的 Worker 取得 `LogitsProcessorOutput` 后，会进入 `ModelRunner.sample()`；采样前先处理 logits bias、grammar mask 等约束，再结合 `sampling_info` 交给 Sampler。重叠调度中还可能延后采样，不能把模型 forward 返回与 CPU 已拿到结果当作同一时刻。[`ModelRunner.sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1977-L2031)

至此，R1 和 R2 各得到一枚新 Token。首枚输出已经来自 Prefill 末端位置的预测，不需要再额外执行一次普通 Decode 才能得到它。接下来是否继续，交回请求状态与停止条件决定。

## 六、接上下一轮 Decode

假设两条请求都没有结束，刚采样出的 Token 分别记为 T1、T2。下一轮把它们作为新增模型输入，而不是重新提交先前的九行 Prompt。两条请求此前都已处理到长度六，现在处理各自的位置六，设备执行使用的 `seq_lens` 推进到七。

| 字段或结果 | 本轮 Prefill | 下一轮普通 Decode |
| --- | --- | --- |
| 请求数 | 2 | 2 |
| 本轮输入行数 | R1 六行、R2 三行，共九行 | T1、T2 各一行，共两行 |
| positions | `[0,1,2,3,4,5,3,4,5]` | `[6,6]` |
| `seq_lens` | `[6,6]` | `[7,7]` |
| 模型末端表示 | `[9,H]` | `[2,H]` |
| 用于下一 Token 采样的行 | 每条请求选择最后一行 | 每条请求本来就只有一行 |
| 逻辑 next-token logits | `[2,V]` | `[2,V]` |

T1、T2 在被采样出来时，还没有经过下一轮模型计算。等它们作为输入执行后，才形成自己的缓存状态，再分别预测后续 Token。要分清“已经有一枚输出”与“已经计算过这枚 Token 的 KV”，否则读请求长度和缓存进度时就容易错一位。

从这里继续读源码，可以固定住一条请求，依次核对本轮新增输入、逻辑位置、缓存写入映射和最终采样行。若这几项能接上，九行输入、两行 Logits 和下一轮两行 Decode 输入就属于同一个连续过程。后续深入 Attention 时，再沿同样的请求索引追历史访问。

### 源码阅读入口

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
