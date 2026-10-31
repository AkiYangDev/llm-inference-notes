# SGLang ModelRunner：九个输入 Token 怎样产生两行 Logits

两条请求的输入都是六个 Token，其中一条已命中三个 Token 的前缀。模型这一轮只需计算九个新位置，采样时却只要两行词表分数。跟着这九行输入走完 ModelRunner，可以看清请求长度、缓存位置和模型输出之间的关系。

Scheduler 已经选定本轮要处理的位置，ModelRunner 接下来把这份计划组织成输入，选择执行方式，再运行模型与采样。Prefill 和普通 Decode 都进入同一份模型代码，差别主要由输入形状和执行元数据传递。前面的组批过程见 [上一篇 Scheduler](sglang-scheduler-source-analysis.md)。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 本文固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线是普通文本生成、非投机、PP=1，沿容易观察的 eager execution 阅读；Decode Graph、Prefill Graph 只追到 ModelRunner 的分发处。模型使用公开实现 `DeepseekV4ForCausalLM`，Ascend 路径追到能够确认的 `DeepseekV4AscendAttnBackend` 接口，不能据此断定某个部署镜像的底层轨迹。`cuda_graph` 是框架接口命名，NPU 图执行仍需看对应平台实现。

</details>

## 一、ForwardBatch 描述本次执行

先用整数 Token 固定两条教学请求，观察新增行和逻辑位置。例中的前缀长度仅用于说明坐标，不代表 DeepSeek-V4 专用缓存必然支持这种匹配粒度。

| 请求 | 完整输入 | 已有可复用前缀 | 本轮新增输入 | 新增输入的逻辑位置 |
| --- | --- | --- | --- | --- |
| R1 | `[11,12,13,14,15,16]` | 无 | `[11,12,13,14,15,16]` | `[0,1,2,3,4,5]` |
| R2 | `[21,22,23,24,25,26]` | 前三个 Token | `[24,25,26]` | `[3,4,5]` |

两条请求的完整长度相同，新增长度却是六和三。执行侧需要知道当前序列有多长，也需要知道这次送入模型多少行。R2 的前三个位置由缓存提供历史状态，不再出现在本轮新输入中。

普通生成中，Worker 把 Scheduler 的计划交给 `ForwardBatch.init_new(batch, model_runner)`，构造执行输入，然后调用 `model_runner.forward(forward_batch)`。[`TpModelWorker.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/managers/tp_worker.py#L631-L749)

`ForwardBatch` 集中描述这一轮的 Token、模式、长度、位置、缓存映射和采样信息。它会借用 `ScheduleBatch` 的部分 Tensor，再补出执行层需要的派生数据，并非把全部上游状态深拷贝一遍。[源码：ForwardBatch 字段与构造](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L468-L605)

将本例的关键字段放在一起看：

```text
input_ids = [11,12,13,14,15,16,24,25,26]    # 9 行
positions = [ 0, 1, 2, 3, 4, 5, 3, 4, 5]    # 各自序列中的位置

extend_prefix_lens = [0,3]    # 已复用的前缀长度
extend_seq_lens    = [6,3]    # 本轮新增长度
seq_lens           = [6,6]    # 两条请求的当前总长度
extend_num_tokens  = 9
```

positions 按请求自己的位置生成。Torch fallback 为每条请求构造 `arange(prefix_len, prefix_len + extend_len)`，再打包这些区间。因此 R2 从位置三开始，不会接着 R1 的位置五继续递增。[`compute_position_torch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L2032-L2046)

物理缓存另有坐标：`out_cache_loc` 指向新增状态的写入槽位，`req_pool_indices` 指出请求在映射池的位置；DeepSeek-V4 NPU 路径还可用 `out_cache_loc_dsv4` 传递专用池的位置集合。结合 positions，后端才能知道每行属于哪条请求、去哪里找历史，以及当前结果写在哪里。[源码：DSV4 ForwardBatch 字段](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L487-L505)

九个 Token 连续存放只改变打包方式。序列边界仍由长度、位置和 Attention 元数据保留，R2 不会因此读到 R1 的历史。接下来，ModelRunner 先判断这份 batch 怎样执行。

## 二、ModelRunner 选择 Graph 或 Eager 路径

图回放复用已捕获的执行结构，Eager 则沿当前调用运行前向。同样的逻辑 batch 能否回放，需要检查 Runner 是否存在，以及模式、形状等条件；Decode 本身并不能保证命中图。

`ModelRunner.forward()` 处理 tracing、专家计数等外围工作后进入 `_forward_raw()`。省略无关分支后，其判断顺序如下，这段仅用于阅读调用关系：[`ModelRunner._forward_raw()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1849-L1947)

```text
检查 Decode Graph 是否可运行
  ├─ 可以：执行图并返回
  └─ 不可以：准备 live batch 的 padding、DP / MLP-sync 等状态
              ├─ Split Prefill：进入专门路径
              ├─ 满足 Prefill Graph 条件：执行对应图
              └─ 其余：交给 EagerRunner
```

注意 `_prepare_eager_forward_batch()` 的位置：它在 Split Prefill、Prefill Graph 和 Eager 分支之前。Decode Graph 若已命中会提前返回，其余 live batch 则先经过这层准备。因此不能因为函数名带 eager，就把它的作用范围只划给最终 Eager 分支。

源码沿用 `cuda_graph` 名称选择图路径，在 Ascend 上也用于 NPU 执行。下面称为 Graph Replay，具体行为仍以对应平台实现为准。

Eager 也需要装载输入。默认 `EagerRunner.load_batch()` 先将当前 batch 填进固定最大容量的静态缓冲，再取本轮大小的视图；只有设置 `SGLANG_EAGER_INPUT_NO_COPY` 才绕过这类复制。比较执行方式时还要看输入准备成本，这段源码本身并未证明某种方式更快。[`EagerRunner.load_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L183-L211)

后文沿普通 Eager 路径进入模型。Graph Replay 不会逐轮重复相同的 Python 调用过程，不能把这条阅读路径直接套到图回放。

## 三、Prefill 与 Decode 怎样进入同一个模型

`EagerRunner.execute()` 根据 `forward_mode` 分流：普通 Decode 进入 `_execute_decode()`，Extend / Prefill 进入 `_execute_extend()`。它们需要不同的 Attention 元数据，Prefill 还可能涉及额外参数和 Context Parallel，因此先分别准备。[`EagerRunner.execute()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L213-L229)

准备之后，两种普通生成路径调用同一个模型入口：

```python
# 省略其他 kwargs，只保留两条路径共同传入的参数。
model_runner.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
    **kwargs,
)
```

Prefill 可以一次送入多行新增输入，普通 Decode 通常每请求只送一行；历史状态与位置由 `ForwardBatch` 和后端元数据提供，无需为两个阶段分别实现一套 DeepSeek 模型。对应两条入口：[`EagerRunner._execute_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L244-L271) [`EagerRunner._execute_extend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/eager_runner.py#L273-L378)

| 本轮执行信息 | 普通 Prefill / Extend | 普通非投机 Decode |
| --- | --- | --- |
| 每条请求新增 Token | 可以有多个 | 通常一个 |
| positions | 一段或多段区间 | 当前末尾位置 |
| 历史状态 | 可能已有前缀缓存 | 通常已经有多轮历史 |
| Attention 元数据 | 按 Prefill 查询组织 | 按 Decode 查询组织 |
| DeepSeek-V4 模型入口 | 同一个 `forward()` | 同一个 `forward()` |

`MIXED` 的转换还依赖平台。固定版本在非 NPU 且无 CP 时可以按 Extend 执行；NPU 留给专用处理，CP 路径也保留 MIXED。阅读模式转换时要连同外层条件一起看。

本例的九行随后进入 DeepSeek-V4。`ForwardBatch` 仍需继续向下传递，因为仅凭 hidden states，Attention 无法恢复这些行的请求边界与历史位置。

## 四、从打包的 Token 进入 DeepSeek-V4

PP=1 让 Embedding、Decoder Layers 和末端 LogitsProcessor 都处于同一条路径；流水线并行还要考虑 stage 交接。这里 `DeepseekV4ForCausalLM.forward()` 先调用 `DeepseekV4Model.forward()`。[`DeepseekV4ForCausalLM.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5081-L5147)

Embedding 将九个 Token ID 映射为 `[9,H]`。随后 DeepSeek-V4 扩展为 `[9,hc_mult,H]`，保存多路 mHC 状态，`hc_mult` 表示状态流数量，与 Attention head 数不同。经过层栈后，模型组合这些状态，再完成末端归一化。[`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4630-L4840)

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

这个示意跟踪的是行数，层内还有投影及布局变化。hidden states 在各层改变表示，positions 和 `ForwardBatch` 则持续保留请求、序列位置及缓存的对应关系。

例如 `DeepseekV4AscendAttnBackend.forward()` 收到 Query 后，读取预先准备的长度和页表，并按本层 `compress_ratio` 选择滑窗或压缩历史。新增状态写入时，`store_cache()` 通过当前缓存映射定位 SWA buffer。[`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189) [源码：DSV4 cache write](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2191-L2219)

R2 虽只带三个新 Token，`seq_lens`、positions、请求索引与缓存位置仍让它们访问本请求已有的前三个位置。历史未放进这轮 `input_ids`，不等于历史不存在。

九行完成模型前向后，先得到九行 hidden states，还未为用户选出新 Token。接下来需要选择用于计算 Logits 的行；内部的历史访问可另见 [DeepSeek-V4 Attention](deepseek-v4-attention-source-analysis.md)。

## 五、九行 hidden states 怎样得到两行 Logits

普通生成要预测的是每条请求的下一个位置，无需为九个新输入分别采样。按打包 Tensor 的零起始行号，R1 末端在 5，R2 末端在 8。

普通 Prefill 且未请求输入 Logprob 时，`LogitsProcessor` 先取末端 hidden states，再送入 LM Head。前面的模型层已经算完九行，裁剪发生在准备词表分数之前。[`LogitsProcessor.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L495-L597)

把本例长度代入末端选择逻辑：[`LogitsProcessor._get_pruned_states()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L599-L665)

```python
# 普通 Prefill、没有输入 logprob 的对应分支。
last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
pruned_states = hidden_states[last_index]
```

`extend_seq_lens=[6,3]` 累计为 `[6,9]`，减一得到 `[5,8]`。这里累计的是打包的新输入行数；R2 总序列长六，本轮却只带三行，不能用总长度去定位打包 Tensor 的末端。

```text
打包行号：  0  1  2  3  4  5 | 6  7  8
所属请求： R1 R1 R1 R1 R1 R1 | R2 R2 R2
选中行号：                5 |       8

模型末端 [9,H] → 选择末端行 [2,H] → LM Head → 逻辑 Logits [2,V]
```

V 为词表维度。`next_token_logits` 的接口语义是 `[#seq,vocab_size]`，所以两条请求对应两行待采样分数。这个逻辑形状不能证明多卡中的每个 Rank 都持有完整词表。[源码：LogitsProcessorOutput](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L201-L240)

这种裁剪有适用条件。输入 Logprob 需要对更多输入位置评分，Target Verify 等模式也各有分支。“九行变两行”只对应本例普通生成所需的采样位置。

词表 Logits 还需经过采样才能得到 Token ID。Worker 取得 `LogitsProcessorOutput` 后进入 `ModelRunner.sample()`，先应用 logits bias、grammar mask 等约束，再结合 `sampling_info` 调用 Sampler。重叠调度还可能延后采样，所以 forward 返回、采样完成和 CPU 拿到结果并非同一时刻。[`ModelRunner.sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L1977-L2031)

R1、R2 至此各获得一枚输出，首 Token 已由 Prefill 末端预测产生，不必先额外跑一次普通 Decode。请求状态和停止条件接着决定是否继续。

## 六、接上下一轮 Decode

假设两条请求都继续，新采样的输出记为 T1、T2。下一轮只把它们作为新增输入，不重算原来的九行 Prompt。两条序列此前都处理到长度六，下一轮计算各自的位置六，设备侧 `seq_lens` 推进为七。

| 字段或结果 | 本轮 Prefill | 下一轮普通 Decode |
| --- | --- | --- |
| 请求数 | 2 | 2 |
| 本轮输入行数 | R1 六行、R2 三行，共九行 | T1、T2 各一行，共两行 |
| positions | `[0,1,2,3,4,5,3,4,5]` | `[6,6]` |
| `seq_lens` | `[6,6]` | `[7,7]` |
| 模型末端表示 | `[9,H]` | `[2,H]` |
| 用于下一 Token 采样的行 | 每条请求选择最后一行 | 每条请求本来就只有一行 |
| 逻辑 next-token logits | `[2,V]` | `[2,V]` |

T1、T2 刚被采样时，还未作为输入经过模型。下一轮执行后才形成它们自己的缓存状态，并预测再下一个 Token。读请求长度时要分清这两个时刻，避免把输出进度与缓存进度错开一位。

这样，九行 Prefill 输入、两行 Logits 和下一轮两行 Decode 输入就接在同一条请求链上。继续深入源码时，可以逐轮核对新增输入、逻辑位置、缓存写入映射及采样行，再沿请求索引追到 Attention 的历史访问。

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
