# DeepSeek-V4 DSpark 源码解析：Draft Model 为什么需要独立处理 TP/DP/EP Layout？

上一篇《[DeepSeek-V4 分布式推理源码解析：TP、EP、DP 三种并行如何共同驱动一次前向计算](deepseek-v4-distributed-parallel-source-analysis.md)》已经确认：普通 DeepSeek-V4 target model 在 Attention 与 MoE 之间存在显式的数据布局桥。对于 `attn_tp_size > 1 + A2A MoE`，post-attention hidden rows 会先按 Attention-TP rank 做 token-row split，MoE 完成后再 `attn_tp_all_gather()` 恢复下一层需要的布局。

但 DSpark 的 DeepSeek-V4 MoE draft 在更早的位置就拒绝了同一类拓扑：

~~~python
if (
    get_parallel().enable_dp_attention
    and self._draft_is_moe
    and ps.attn_tp_size > 1
):
    raise ValueError(
        "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
        "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
        "MoE-under-DP all-reduce."
    )
~~~

固定源码：[`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L120-L175)。

于是问题变成：

> **为什么 target forward 已经能在 Attention TP 与 MoE EP 之间转换 layout，而 DSpark draft 仍要求 `attn_tp == 1`？**

这篇文章不会把静态源码能看到的现象直接包装成“已经找到唯一根因”。全文会严格区分三层：

~~~text
源码事实
    ↓
从相邻代码能支持的设计推断
    ↓
必须通过真实 TP32/DP16 复现与 Trace 才能确认的根因
~~~

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。讨论 DeepSeek-V4 MoE draft + DSpark V2 + DP Attention。限制发生在通用 DSpark worker，而不是某个 NPU kernel，因此不能把它表述成“昇腾硬件不支持 TP≠DP”；但 NPU Draft 仍有 ModelSlim、独立 vocab modules、shared-expert fusion 等平台特有边界。

## 一、先把限制定位准：当前拒绝发生在 DSpark runtime，不在 NPU backend

对 DP Attention，前一篇得到：

~~~text
attn_dp_size = dp_size

attn_tp_size =
tp_size / attn_dp_size / attn_cp_size
~~~

所以：

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

所以 TP32 / DP16 在真正创建 DeepSeek-V4 draft worker 之前就被拒绝。

这条 guard 没有 `is_npu()` 判断，也没有 HCCL 判断。因此源码直接能证明的是：

> **当前限制属于 DSpark DeepSeek-V4 MoE draft 的 runtime correctness guard，而不是 Ascend NPU 硬件能力声明。**

报错文本进一步给出一个重要线索：

~~~text
attn_tp > 1 corrupts the MoE-under-DP all-reduce
~~~

但这仍然只是 guard 作者对已知 failure mode 的描述。仅凭这一句，还不能严谨地确定“第几个 all-reduce、哪一个 Tensor、哪一个 Rank 首先错误”。后面会单独区分已知事实和待验证根因。

## 二、MoE draft 为什么没有像 Dense draft 一样整体进入 `draft_tp_context`

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

`draft_tp_context()` 本质上调用 `patch_tensor_parallel_group(tp_group)`。源码注释写的是：

~~~text
Draft model doesn't use dp and has its own tp group.
~~~

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

这是**源码事实**。

### 为什么这么设计？源码没有一句注释直接给出唯一因果解释

周边代码提供了非常强的设计信号：创建 `DraftBlockProposer` 时，MoE draft + DP Attention 会设置：

~~~python
dp_moe_sync = (
    self._draft_is_moe
    and get_parallel().enable_dp_attention
)
~~~

随后 Draft forward 会额外填充 DP/MoE 同步 metadata；空闲 Rank 也需要执行 `run_idle_participation()` 进入 draft model forward。

所以可以合理推断：

> **MoE draft 不能简单像 Dense draft 那样把整个模型世界缩成一个 DP-local `attn_tp_group`，因为 Draft MoE path 还依赖跨 DP / MoE rank 的同步与 collective participation。**

但这句话属于**由代码结构支持的设计推断**，不是某条源码注释直接宣布的 root cause。发布时必须保留这个边界。

还有一个更细的点：`DraftBlockProposer._base_logits_context()` 在 `dp_moe_sync=True` 时，会短暂使用 `draft_tp_context(attn_tp_group)` 来计算 proposal-side base logits。也就是说，代码并不是“MoE draft 永远不用 attn_tp_group”，而是：

~~~text
Draft model forward / MoE
→ 不整体 patch 到 attn_tp_group

部分 proposal/logit 计算
→ 可以临时进入 attn_tp_group context
~~~

这进一步说明 group ownership 是按阶段区分的。

## 三、`global_num_tokens` 到底是什么：名字像总数，实际是“每 Rank 的计数列表”

这是初稿最需要收紧的地方之一。

`ScheduleBatch.global_num_tokens` 在 decode-family DP 同步语义里并不是一个 scalar“全局 Token 总数”。它是：

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

DSpark Draft 再根据：

~~~text
spec_info.num_tokens_per_req
~~~

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

因此文章不能再泛泛写成“`global_num_tokens` 表示整个 world 有多少 Draft Token”。更准确的是：

> **它携带的是 per-rank 计数向量；对 speculative forward，会按每请求固定宽度缩放成每 Rank 的 token-row 数，用来让相关 DP/MoE runtime 对几何形状取得一致认识。**

### Idle Rank participation 也说明这些 metadata 属于 collective contract

当本地没有请求，但其他 DP Rank 仍在跑 MoE draft 时，`run_idle_participation()` 会构造 `ForwardMode.IDLE` 的空 batch，填入同一套同步 metadata，然后仍调用 draft model forward。

这说明：

~~~text
本地 Token 数为 0
≠
这个 Rank 可以完全跳过本轮 distributed MoE path
~~~

collective participation 本身就是 correctness contract 的一部分。

## 四、`SpecTpSync` 的 group 边界：它同步 speculative 决策，不同步 MoE 数据

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

> **`SpecTpSync` 不是 DP/MoE synchronization mechanism。**

它的类注释是：

~~~text
Broadcasts a speculative decision from rank 0 to its TP group.
~~~

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

默认环境变量：

~~~text
SGLANG_SPEC_TP_SYNC = all
~~~

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

因此文章里更准确的两层 ownership 是：

~~~text
Speculative decision agreement
→ SpecTpSync / attn_tp_group

Draft MoE distributed execution
→ DP / MoE / EP runtime topology
~~~

这两个 domain 同时存在，正是 `attn_tp_size>1` 时问题复杂的原因之一。

## 五、Planner 里的 `attn_tp_size == 1` 不是全局 DSpark 限制，只约束一个 DP-tier gather 优化

初稿里如果只看到：

~~~python
and get_parallel().attn_tp_size == 1
~~~

很容易写成：

> DSpark Verify Planner 只支持 attn_tp=1。

这是错误的泛化。

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

所以源码事实是：

> **`attn_tp_size == 1` 是 compact ragged-verify 下 DP-tier token-count gather 这条优化路径的启用条件之一。**

它不是：

~~~text
整个 Planner 的全局合法性条件
~~~

也不能单独拿来证明：

~~~text
TP32/DP16 的根因就在 Verify Planner
~~~

当前 DSparkWorker guard 已经在 Planner 初始化之前阻止 MoE draft 的 `attn_tp_size>1`，所以 Planner 的其他路径是否能在放开 guard 后完整工作，还需要实测。

这个条件最多告诉我们：

> **至少部分 DP-specific verify scheduling / graph-tier logic 是在 pure DP-attention（attn_tp=1）的 topology 下设计和测试的。**

这是风险信号，不是已确认 root cause。

## 六、现在到底能确认什么，哪些仍只是 TP32/DP16 修复假设

严格 Review 后，最重要的结论反而是：

> **不能把当前问题简单归因为“Draft 缺少 target 的 token split / all-gather”。**

原因是 DeepSeek-V4 DSpark stage 直接继承 `DeepseekV4DecoderLayer`，其 `_run_ffn()` 明确调用：

~~~python
self._run_moe_ffn_dp_sync(
    x,
    forward_batch,
    input_ids=None,
    input_ids_global=None,
)
~~~

固定源码：[`DSparkV4Stage._run_ffn()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L701-L778)。

也就是说，target model 那套 Attention→MoE bridge 并不是完全与 Draft 隔离。

### 已确认事实

1. DeepSeek-V4 MoE Draft + DP Attention + `attn_tp_size>1` 当前在 `DSparkWorkerV2` 初始化阶段被显式拒绝。
2. guard 文本指出已知 failure mode 与 “MoE-under-DP all-reduce” 有关。
3. MoE draft 不整体 patch 到 `attn_tp_group` 的 Draft context；Dense Draft 会。
4. MoE draft 会携带 DP/MoE global token metadata，并要求 idle ranks 参与 Draft forward。
5. Draft Stage 本身复用 `_run_moe_ffn_dp_sync()`。
6. Speculative decisions 默认在 `attn_tp_group` 内通过 `SpecTpSync` 的显式 sites 保持一致。
7. Verify Planner 中存在只在 `attn_tp==1` 才开启的 compact-DP tier gather 优化，但它不是 Planner 的全局限制。

### 有源码支持，但仍属于设计推断

1. MoE draft 不整体进入 `draft_tp_context`，很可能是因为其执行需要保留 DP/MoE collective topology，而不是只在本 DP replica 内独立计算。
2. `attn_tp_size>1` 时，spec decision group、MoE collective group、token-row layout 之间存在额外 ownership 转换，因此比 Dense Draft 更容易产生 reduction / ordering mismatch。
3. 修复需要保证 hidden rows 之外的 draft metadata、verify metadata 和 collective geometry 一起转换，而不能只调整一个 Tensor。

### 当前不能靠静态源码直接确认

1. 第一个产生错误数值的具体 collective 是哪一个。
2. 是 routed expert partial output、shared expert output、还是后续 combine/reduction 首先发生错位。
3. target 的 `_run_moe_ffn_dp_sync()` 是否可以原样复用于 Draft，还是 Draft 前后还需要额外 split/gather。
4. compact verify、CUDA/NPU graph、idle rank、跨机 HCCL 中哪一个是第二个阻塞点。
5. 放开 guard 后 eager 模式能否先正确、graph 模式再失败。

这些都必须通过真实运行验证。

### 所以 TP32 / DP16 的正确研发顺序应该是

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

最值得记录的 Trace 字段：

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

虽然 guard 不是 NPU-only，但 `DeepseekV4ForCausalLMDSpark` 在 NPU 上还有明确差异：

~~~text
uses_own_vocab_modules = True
~~~

并且 ModelSlim NPU weight loading 当前不支持把 shared experts 映射进 fused expert slots，因此会禁用相应 shared-expert fusion。

源码：[`DeepseekV4ForCausalLMDSpark`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L781-L815)。

因此最终验收至少要区分：

~~~text
通用 DSpark layout correctness
vs
Ascend NPU / ModelSlim / HCCL backend-specific correctness
~~~

不能因为 CUDA eager 下通过，就直接宣布 Ascend 跨机 TP32/DP16 已完成。

## 结尾：Draft Model 为什么必须独立处理 Layout

普通 target forward 主要维护：

~~~text
Token ownership
Tensor-shard ownership
Expert ownership
~~~

DSpark 在这些基础上还增加：

~~~text
Draft-block ownership
Speculative-decision ownership
Verify-layout ownership
Accept-state ownership
Commit ownership
~~~

因此：

~~~text
Target 支持 TP≠DP
≠
Draft 自动支持 TP≠DP
~~~

真正需要建立的是一套 Speculative Layout Contract：

~~~text
Bonus / Draft rows
      ↓
Draft Attention / MoE
      ↓
Draft proposal
      ↓
Verify layout
      ↓
Target verify
      ↓
Accept state
      ↓
KV / hidden commit
~~~

每一次 layout 转换，都必须让 Token 本身和与它同行的 metadata 对同一个 ownership 达成一致。

这就是 TP32 / DP16 这个问题真正值得研究的地方：它不是单独一个 parallel flag 不兼容，而是 speculative decoding 状态机和 distributed inference 状态机第一次真正撞在一起。

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