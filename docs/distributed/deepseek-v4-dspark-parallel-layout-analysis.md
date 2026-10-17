# DeepSeek-V4 DSpark 的并行限制与 Draft 布局

同样是 DeepSeek-V4，在 `TP32 / DP16` 下，Target 的 Attention 与 MoE 之间有显式的 Token 行布局转换；DSpark 的 MoE Draft 却会在启动时被拒绝，因为这时 `attn_tp_size=2`，不满足 Worker 要求的 `attn_tp_size=1`。

要理解这个差异，需要分别看 Draft 的并行上下文、各 Rank 的 Token 计数，以及投机决策同步。下面沿这三处源码展开：先定位限制，再说明现有布局如何衔接，最后列出验证更宽 Attention TP 所需的检查项。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。讨论 DeepSeek-V4 MoE draft + DSpark V2 + DP Attention。限制位于通用 DSpark worker 的配置检查中。NPU Draft 还涉及 ModelSlim、独立 vocab modules 和 shared-expert fusion 等平台分支，后文分别说明。

## 一、启动限制发生在哪里

对 DP Attention，前一篇得到：

~~~text
attn_dp_size = dp_size

attn_tp_size =
tp_size / attn_dp_size / attn_cp_size
~~~

~~~text
TP32
DP16
DP Attention = ON
CP1

=> attn_dp_size = 16
=> attn_tp_size = 2
~~~

而 DSpark guard 的三个条件恰好全部成立：

~~~text
enable_dp_attention = True
draft_is_moe = True
attn_tp_size > 1
~~~

所以 TP32 / DP16 在创建 DeepSeek-V4 draft worker 之前就被拒绝。

这条 guard 没有 `is_npu()` 判断，也没有 HCCL 判断。这项限制位于：

> **这项限制由 DSpark Worker 的配置检查实施，位于通用 MoE Draft Runtime 中。**

报错文本进一步给出一个重要线索：`attn_tp > 1 corrupts the MoE-under-DP all-reduce`

报错把线索指向 MoE-under-DP 的 all-reduce。定位还要向前走一步：对照各 Rank 的 Tensor 行归属与归约结果，找到第一次分歧。

## 二、Dense Draft 与 MoE Draft 的并行上下文

DSpark worker 初始化时先区分 Draft 类型：

~~~python
self._draft_is_moe = draft_is_deepseek_v4()

self._draft_dp_context_enabled = (
    get_parallel().enable_dp_attention
    and not self._draft_is_moe
)
~~~

然后 `_draft_context()` 只有在 `_draft_dp_context_enabled=True` 时才做：

~~~python
return draft_tp_context(
    get_parallel().attn_tp_group
)
~~~

`draft_tp_context()` 调用 `patch_tensor_parallel_group(tp_group)`。源码注释写的是：`Draft model doesn't use dp and has its own tp group.`

固定源码：[`draft_tp_context()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L706-L718)。

所以对于 **Dense Draft + DP Attention**：

~~~text
target tp world
      ↓
每个 attention-DP replica 内
      ↓
patch draft TP group = attn_tp_group
      ↓
Draft 在这个局部 group 内运行
~~~

但 DeepSeek-V4 MoE draft 刻意不走这一整个 forward context：

~~~text
_draft_is_moe = True
=> _draft_dp_context_enabled = False
=> DSparkWorkerV2._draft_context() = nullcontext()
~~~



### MoE Draft 为什么保留跨 DP 的参与关系

周边代码提供了非常强的设计信号：创建 `DraftBlockProposer` 时，MoE draft + DP Attention 会设置：

~~~python
dp_moe_sync = (
    self._draft_is_moe
    and get_parallel().enable_dp_attention
)
~~~

随后 Draft forward 会额外填充 DP/MoE 同步 metadata；空闲 Rank 也需要执行 `run_idle_participation()` 进入 draft model forward。

这些调用表明，MoE Draft 的执行保留了跨 DP / MoE Rank 的同步与 collective 参与关系。这也解释了它与 Dense Draft 的上下文选择为何不同：整个 Forward 要继续使用 MoE 所需的通信拓扑。

还有一个更细的点：`DraftBlockProposer._base_logits_context()` 在 `dp_moe_sync=True` 时，会短暂使用 `draft_tp_context(attn_tp_group)` 来计算 proposal-side base logits。`attn_tp_group` 在这两个阶段的用途不同：

~~~text
Draft model forward / MoE
→ 不整体 patch 到 attn_tp_group

部分 proposal/logit 计算
→ 可以临时进入 attn_tp_group context
~~~

这进一步说明 group ownership 是按阶段区分的。

## 三、每个 Rank 的 Token 计数怎样传递

这里需要区分 draft 模型参数和实际通信布局。

在 decode-family DP 同步中，`ScheduleBatch.global_num_tokens` 保存：

> **每个相关 Rank 的原始同步计数列表；在普通 decode-family round 中，这些原始值是 per-rank request counts。**

固定函数 [`spec_scale_global_num_tokens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_info.py#L451-L465) 的 docstring 直接写：

~~~text
Scale the raw per-rank sync values
(request counts on decode-family rounds)
into this forward's token units
using the spec input's uniform per-request widths.
~~~

也就是说，原始：

~~~text
batch.global_num_tokens
=
[rank0_request_count,
 rank1_request_count,
 rank2_request_count, ...]
~~~

DSpark Draft 再根据：`spec_info.num_tokens_per_req`

把它变成本次 Draft forward 的 token-row 单位。

例如测试里：

~~~text
raw per-rank request counts:
[1, 3, 0, 2]

num_tokens_per_req = 6

scaled token units:
[6, 18, 0, 12]
~~~

这个行为有固定单测：[`test_dspark_dp_tier.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/spec/dspark/test_dspark_dp_tier.py)。

### DSpark 为什么同时保存 raw 和 scaled 两份

`_fill_dp_moe_sync_metadata()` 会保留：

~~~python
forward_batch.original_global_num_tokens_cpu
~~~

作为 raw per-rank request counts；同时把缩放后的 draft-token counts 放进：

~~~text
global_num_tokens_cpu
global_num_tokens_gpu
~~~

固定源码：[`_fill_dp_moe_sync_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L475-L513)。

源码注释明确指出两者用途不同：

~~~text
raw per-rank request counts
→ graph bucket / admission semantics

scaled draft-token units
→ DP / MoE synchronization geometry
~~~

因此，读取 `global_num_tokens` 时需要同时确认计数单位：

> **它携带的是 per-rank 计数向量；对 speculative forward，会按每请求固定宽度缩放成每 Rank 的 token-row 数，用来让相关 DP/MoE runtime 对几何形状取得一致认识。**

### Idle Rank participation 也说明这些 metadata 属于 collective contract

当本地没有请求，但其他 DP Rank 仍在跑 MoE draft 时，`run_idle_participation()` 会构造 `ForwardMode.IDLE` 的空 batch，填入同一套同步 metadata，然后仍调用 draft model forward。

~~~text
本地 Token 数为 0
≠
这个 Rank 可以完全跳过本轮 distributed MoE path
~~~

collective participation 本身就是 correctness contract 的一部分。

## 四、SpecTpSync 同步哪些决策

`DSparkWorkerV2` 初始化：

~~~python
self._tp_sync = SpecTpSync(
    parallel.attn_tp_group
    if parallel.enable_dp_attention
    else parallel.tp_group
)
~~~

所以 DP Attention 开启后：

~~~text
SpecTpSync group
=
attn_tp_group
~~~

TP32 / DP16 / CP1 时，这就是每个 DP replica 内的 2-rank Attention TP group。

但必须非常清楚：

> **`SpecTpSync` 在所选 TP Group 内同步投机决策，DP / MoE 的模型通信另由对应路径完成。**

它的类注释是：`Broadcasts a speculative decision from rank 0 to its TP group.`

`sync()` 做的只是：

~~~python
self._tp_group.broadcast(values, src=0)
~~~

固定源码：[`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)。

同步 site 包括：

~~~text
DSPARK_DRAFT_GREEDY
DSPARK_DRAFT_SAMPLE
DSPARK_PLAN
DSPARK_ACCEPT_GREEDY
DSPARK_ACCEPT_SAMPLE
DSPARK_TARGET
...
~~~

默认环境变量：`SGLANG_SPEC_TP_SYNC = all`

见 [`environ.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375)。

所以默认情况下，代码中显式调用 `self._tp_sync.sync(site, tensor)` 的 speculative 决策会在 attn-TP group 内广播一致。

但它不会自动同步：

~~~text
全部 hidden_states
MoE router activations
EP dispatch buffers
DP global token geometry
KV cache pages
~~~

这些属于其他 runtime / communicator 的职责。

这形成两层不同的同步职责：

~~~text
Speculative decision agreement
→ SpecTpSync / attn_tp_group

Draft MoE distributed execution
→ DP / MoE / EP runtime topology
~~~

这两个 domain 同时存在，正是 `attn_tp_size>1` 时问题复杂的原因之一。

## 五、Planner 的 DP Tier Gather 条件

`attn_tp_size == 1` 出现在 DP-tier gather 的启用条件中：

真实代码里这个条件属于：

~~~python
self._dp_tier_gather_enabled = (...)
~~~

完整上下文还要求：

~~~text
RaggedVerifyMode.COMPACT
DP Attention enabled
attn_tp_size == 1
attn_cp_size == 1
require_mlp_tp_gather()
overlap schedule enabled
not skip_dp_mlp_sync
non-disaggregated mode
PP1
not scheduler skip all-gather
~~~

固定源码：[`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py)。

`attn_tp_size == 1` 与上面的其他条件一起，开启 compact ragged-verify 的 DP-tier token-count gather。它控制的是这一条优化分支。

当前 Worker 在 Planner 初始化前就拒绝 MoE Draft 的 `attn_tp_size>1`。因此，要验证更宽的 Attention TP，需要先处理 Worker 限制，再检查 Planner 的宽度选择、计数与 Graph Tier。

## 六、验证更宽 Attention TP 需要哪些证据

继续排查时，需要看 Draft 实际复用了哪些布局转换。

DeepSeek-V4 DSpark stage 直接继承 `DeepseekV4DecoderLayer`，其 `_run_ffn()` 明确调用：

~~~python
self._run_moe_ffn_dp_sync(
    x,
    forward_batch,
    input_ids=None,
    input_ids_global=None,
)
~~~

固定源码：[`DSparkV4Stage._run_ffn()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L701-L778)。

Draft 因此复用了这部分 Attention→MoE bridge，排查时继续检查调用前后的行布局与计数。

### 从限制定位到运行验证

前文已经确定三处边界：Worker 拒绝 MoE Draft 的 `attn_tp>1`；Draft 复用父类的 MoE bridge；投机决策在 Attention TP 子组内同步。还缺的是运行证据：哪一次布局转换或归约首先偏离了正确结果。

| 验证阶段 | 对照方式 | 要回答的问题 |
| --- | --- | --- |
| 可运行基线 | 保持 `attn_tp=1`，保存逐阶段输出与计数 | 正确结果和通信顺序是什么？ |
| Eager 布局实验 | 开发环境中检查更宽 Attention TP，逐边界对齐 | 首次数值或行归属分歧在哪里？ |
| 空闲 Rank 与批次变化 | 包含零请求 Rank、不均衡请求数 | 计数缩放和 collective 参与是否一致？ |
| Graph 与 Verify 模式 | 在正确 Eager 基础上分别验证 | padding、tier 选择或 metadata 是否改变语义？ |
| Ascend 目标环境 | 使用实际 ModelSlim、通信后端与部署拓扑 | 平台分支能否保持同样的正确性？ |

验证顺序是先建立正确基线，再逐边界比较更宽拓扑的输出。第一处差异会指明修复应落在 bridge、metadata 还是后续执行路径。

### 按依赖顺序推进验证

~~~text
TP32 / DP32 + DSpark
→ 当前正确 baseline

TP32 / DP16 + DSpark OFF
→ 验证 target path

TP32 / DP16 + DSpark ON
→ 临时绕过 guard
→ greedy + eager + tiny batch

逐阶段记录 ownership
→ 找到第一次 divergence

只修第一次 divergence
→ 再扩展 graph / multi-request / cross-node
~~~

按请求与轮次记录以下 Trace 字段：

~~~text
global_rank
attn_dp_rank
attn_tp_rank
moe_ep_rank

request identity
speculative step index
tensor row count
global_num_tokens_cpu

draft_block_ids
draft_tokens
verify_ids
correct_len
commit_lens
positions
cache locations
~~~

对每一个 layout boundary，都问同一句：

> **这行 Token 当前属于哪个请求、哪个 speculative step、哪个 Rank；下一个 collective 又认为它属于谁？**

这比一上来只抓 HCCL trace 更容易找到 ownership 第一次分叉的位置。

### Ascend NPU 上还要单独补一组验证

虽然 guard 不是 NPU-only，但 `DeepseekV4ForCausalLMDSpark` 在 NPU 上还有明确差异：`uses_own_vocab_modules = True`

并且 ModelSlim NPU weight loading 当前不支持把 shared experts 映射进 fused expert slots，因此会禁用相应 shared-expert fusion。

源码：[`DeepseekV4ForCausalLMDSpark`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L781-L815)。

因此最终验收至少要区分：

~~~text
通用 DSpark layout correctness
vs
Ascend NPU / ModelSlim / HCCL backend-specific correctness
~~~

CUDA Eager 结果可作为对照，Ascend 跨机配置则要覆盖上面的 ModelSlim、HCCL 与 Graph 路径。

## 七、区分模型布局、计数与决策

Target 与 Draft 复用模型层，并不意味着两者具有相同的运行约束。Draft 还要携带候选步数、Verify 长度和接受结果；每次行布局转换，都需要让这些 metadata 与 Token 行保持对应。

排查 `TP32 / DP16` 时，先确认当前阶段使用的 Group，再核对每 Rank 的原始请求数、缩放后的 Token 数和投机决策。这样才能将“配置被 guard 拒绝”推进到“具体哪处状态或计算需要修复”。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| DSpark `attn_tp>1` guard | [`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L120-L175) |
| Draft Worker 构造 | [`build_draft_tp_worker()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_worker_common.py) |
| Draft TP Context | [`draft_tp_context()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L706-L718) |
| Speculative TP decision sync | [`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py) |
| 默认 SpecTpSync 配置 | [`SGLANG_SPEC_TP_SYNC`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375) |
| Draft ForwardBatch 构造 | [`DraftBlockProposer._run_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L369-L473) |
| DP/MoE sync metadata | [`_fill_dp_moe_sync_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L475-L513) |
| Spec token-count scaling | [`spec_scale_global_num_tokens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_info.py#L451-L465) |
| DP-tier 单元测试 | [`test_dspark_dp_tier.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/spec/dspark/test_dspark_dp_tier.py) |
| Draft Stage / MoE bridge 复用 | [`DSparkV4Stage`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L609-L778) |
| Target Attention→MoE bridge | [`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915) |
| Verify Planner | [`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py) |
| Target Verify | [`TargetVerifyExecutor`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py) |
| NPU Draft model differences | [`DeepseekV4ForCausalLMDSpark`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L781-L815) |
