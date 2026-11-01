# DeepSeek-V4 DSpark 的并行限制与 Draft 布局

`TP32 / DP16` 的 V4 Target 可以经过 Attention→MoE 的 Token 布局转换，DSpark MoE Draft 却在启动时被挡住。CP1 下派生的 `attn_tp_size=2`，超出了 Worker 要求的宽度 1。

限制发生在 Draft Worker，并没有以 NPU 或 HCCL 为条件。要追它背后的布局问题，需要继续看 Draft 用什么 Group、计数怎样从请求换算为候选行，以及哪些决定被同步。本文定位这些源码约束，尚未运行解除限制后的配置。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线为 V4 MoE Draft、DSpark V2、DP Attention。通用 Worker 检查之外，NPU 还有 ModelSlim、独立词表模块、shared-expert fusion 等平台分支。

</details>

## 一、启动限制发生在哪里

DPA 先将请求分给 Attention DP，剩余 TP 宽度再在组内协作。普通非弹性布局中，`attn_dp_size=dp_size`，`attn_tp_size=tp_size/attn_dp_size/attn_cp_size`。

TP32/DP16、CP1、DP Attention 开启，得到 attn_dp_size=16、attn_tp_size=2。每个 Attention-DP 内有两个 Rank，命中 `enable_dp_attention=True`、`draft_is_moe=True`、`attn_tp_size>1`。

检查在 V4 Draft Worker 创建前拒绝 TP32 / DP16。

条件里没有 `is_npu()`、HCCL，错误来源首先指向通用 Draft 布局。

报错说明为 `attn_tp > 1 corrupts the MoE-under-DP all-reduce`。

这句话指向归约风险，但没有指出哪一行最先出错。接下来要确认每个参与者拿到谁的 Token 行、数值是否为同一结果的局部贡献。删除 guard 只能允许代码继续执行。

## 二、Dense Draft 与 MoE Draft 的并行上下文

Worker 用 context 临时改变 TP Group 查询结果，同一模型层因而可以在不同组中运行。

~~~python
self._draft_is_moe = draft_is_deepseek_v4()

self._draft_dp_context_enabled = (
    get_parallel().enable_dp_attention
    and not self._draft_is_moe
)
~~~

`_draft_dp_context_enabled=True` 时，整体 `_draft_context()` 使用：

~~~python
return draft_tp_context(
    get_parallel().attn_tp_group
)
~~~

`draft_tp_context()` 内部调用 `patch_tensor_parallel_group(tp_group)`，注释为 `Draft model doesn't use dp and has its own tp group.`。

实现见 [`draft_tp_context()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L706-L718)。

Dense Draft + DPA 临时把 Draft TP Group 换成 `attn_tp_group`，各 Attention-DP 副本在自己的局部组执行。V4 MoE Draft 的 `_draft_is_moe=True` 使 `_draft_dp_context_enabled=False`，整体返回 `nullcontext()`，保留原组。

### MoE Draft 保留跨 DP 参与

构造 Proposer 时，MoE Draft + DPA 设置：

~~~python
dp_moe_sync = (
    self._draft_is_moe
    and get_parallel().enable_dp_attention
)
~~~

forward 随后填 DP/MoE 同步 metadata，idle Rank 也通过 `run_idle_participation()` 进入模型。

MoE Draft 需要这些跨 DP/MoE 参与者，只运行本地有请求的 Attention 子组，会改变这一分工。

局部 logits 另有 context：`dp_moe_sync=True` 时，`_base_logits_context()` 可用 `draft_tp_context(attn_tp_group)` 计算 proposal base logits。

因此要沿 context 的作用范围理解 Group：它包住整个 Draft，还是只包住 base logits，执行含义不同。

## 三、每个 Rank 的 Token 计数怎样传递

Group 之外，还需要确定计数单位。一条请求在 Draft 中可能占多行。

decode-family 同步的 `ScheduleBatch.global_num_tokens` 保存原始逐 Rank 计数，普通轮次按请求数使用。名字包含 tokens，具体单位却取决于生成阶段。

[`spec_scale_global_num_tokens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_info.py#L451-L465) 的 docstring 说明了换算：

~~~text
Scale the raw per-rank sync values
(request counts on decode-family rounds)
into this forward's token units
using the spec input's uniform per-request widths.
~~~

逐 Rank 的原始 `batch.global_num_tokens` 乘 `spec_info.num_tokens_per_req`，才成为当前 Draft 输入行数。

固定测试使用：

~~~text
raw per-rank request counts:
[1, 3, 0, 2]

num_tokens_per_req = 6

scaled token units:
[6, 18, 0, 12]
~~~

入口见 [`test_dspark_dp_tier.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/spec/dspark/test_dspark_dp_tier.py)。

### DSpark 保存 raw 与 scaled 计数

`_fill_dp_moe_sync_metadata()` 分别保留两份：

| 字段 | 单位 | 用途 |
| --- | --- | --- |
| `original_global_num_tokens_cpu` | 原始逐 Rank 请求数 | Graph bucket 与准入 |
| `global_num_tokens_cpu` / `global_num_tokens_gpu` | 缩放后的 Draft Token 行数 | DP/MoE 同步执行几何 |

实现见 [`_fill_dp_moe_sync_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L475-L512)。

raw `[1, 3, 0, 2]` 可用于请求级 Graph admission。此例每请求六行，执行几何就是 `[6, 18, 0, 12]`。3 数请求，18 数输入行；六行来自 num_tokens_per_req，不代表所有配置都提出六枚候选。

<a id="idle-rank-participation-metadata-collective-contract"></a>

### Idle Rank 参加同一轮 collective

本地没请求、其他 DP 仍执行 MoE Draft 时，idle 路径构造 `ForwardMode.IDLE` 空 batch，填同套 metadata，并调用 forward。

它没有本地输入，也可能仍是 collective 参与者。跳过整个 forward，会让其他 Rank 等待缺席者，空 forward 则维持对应轮次。

## 四、SpecTpSync 同步哪些决策

Worker 建立 `SpecTpSync`：

~~~python
self._tp_sync = SpecTpSync(
    parallel.attn_tp_group
    if parallel.enable_dp_attention
    else parallel.tp_group
)
~~~

DPA 下选用 `attn_tp_group`。

TP32/DP16/CP1 的概念拓扑中，这个组有两个 Rank。Worker 会先拒绝对应 MoE Draft 配置，这里只用它说明建组；DP/MoE 数据交换仍由模型路径处理。

类注释为 `Broadcasts a speculative decision from rank 0 to its TP group.`。

sync 操作为：

~~~python
self._tp_group.broadcast(values, src=0)
~~~

实现见 [`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)。

显式 site 包括：

~~~text
DSPARK_DRAFT_GREEDY
DSPARK_DRAFT_SAMPLE
DSPARK_PLAN
DSPARK_ACCEPT_GREEDY
DSPARK_ACCEPT_SAMPLE
DSPARK_TARGET
...
~~~

`SGLANG_SPEC_TP_SYNC = all` 为默认值。

定义见 [`environ.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375)。

多 Rank 因此默认在这些 site 使用源 Rank 的值。src=0 是组内编号，组宽为 1 时直接使用本地值。

hidden states、MoE router activation、EP dispatch buffer、DP 全局 Token 几何、KV pages 继续由各自 Runtime、communicator 管理。

SpecTpSync 对齐候选、接受等决定，DP/MoE/EP Runtime 执行分布式模型。查更宽 Attention TP 的问题，要同时核对决定是否一致、模型内部行分配和归约是否正确。

## 五、Planner 的 DP Tier Gather 条件

Planner 另用 `attn_tp_size==1` 控制 `_dp_tier_gather_enabled`，同时要求：

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

实现见 [`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py)。

这是 compact ragged 的 DP-tier Token 计数收集分支，用来协调本轮执行容量，发生位置与 Worker 启动检查不同。

MoE Draft 宽度不符会在 Planner 初始化前被拒绝。扩展宽度以后，还需要继续核 Planner 计数、Graph tier 对这套几何的处理。

## 六、验证更宽 Attention TP 需要哪些证据

Draft 本身已经复用了一部分布局转换。

`DSparkV4Stage` 继承 Decoder Layer，`_run_ffn()` 调用：

~~~python
self._run_moe_ffn_dp_sync(
    x,
    forward_batch,
    input_ids=None,
    input_ids_global=None,
)
~~~

实现见 [`DSparkV4Stage._run_ffn()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L701-L778)。

也就是说，Attention→MoE bridge 已经被调用。继续定位，需要在它前后对比每行归属、计数单位，而非先假设 Draft 缺少整个 bridge。

### 从限制定位到运行验证

已确认 Worker 的 guard、Draft 的 bridge、Attention 子组内的决策同步。还缺运行证据，确定哪次转换或归约首先偏离正确结果。

| 验证阶段 | 对照方式 | 要回答的问题 |
| --- | --- | --- |
| 可运行基线 | 保持 `attn_tp=1`，保存逐阶段输出与计数 | 正确结果和通信顺序是什么？ |
| Eager 布局实验 | 开发环境中检查更宽 Attention TP，逐边界对齐 | 首次数值或行归属分歧在哪里？ |
| 空闲 Rank 与批次变化 | 包含零请求 Rank、不均衡请求数 | 计数缩放和 collective 参与是否一致？ |
| Graph 与 Verify 模式 | 在正确 Eager 基础上分别验证 | padding、tier 选择或 metadata 是否改变语义？ |
| Ascend 目标环境 | 使用实际 ModelSlim、通信后端与部署拓扑 | 平台分支能否保持同样的正确性？ |

表中列的是未执行的验证方案。先建立受支持拓扑基线，再逐边界对比，才能把差异归到 bridge、metadata 或后续计算。

### 按依赖顺序推进验证

示意路线固定 CP1、PP1 及其他可比条件；TP32/DP32 也要满足其余模型、平台限制。开发隔离时绕过 guard，只用于观察错误从哪里开始。

~~~text
TP32 / DP32 + DSpark
→ 待建立并验证的 baseline

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

按请求、轮次记录：

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

逐边界标明行所属请求、speculative step、Rank，再对照下一次 collective 如何解释它。

两边 shape 即使都为四行，第二行若已经属于不同请求，归约仍然会错。身份与轮次能把这类布局差异从通信故障中分开。

### Ascend NPU 的平台分支验证

NPU 模型设置 `uses_own_vocab_modules = True`。

ModelSlim NPU 加载不能把 shared experts 映射进 fused slots，因此禁用对应 fusion。

实现见 [`DeepseekV4ForCausalLMDSpark`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L781-L815)。

通用布局对照之后，Ascend 910C 还要覆盖实际 ModelSlim、通信、Graph、跨机拓扑。CUDA Eager 结果只验证对应路径，平台分支仍需用相同的行与状态关系验收。

## 七、继续定位并行布局的起点

模型层可以复用，Draft 还额外带有候选步数、窗口、接受结果。这些数据需要一起经过每次布局转换。

对 TP32 / DP16，下一步是按阶段 Group 追 raw 请求数、scaled 输入行、决定结果，直到找出首个偏离的执行边界。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| DSpark `attn_tp>1` guard | [`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L120-L175) |
| Draft Worker 构造 | [`build_draft_tp_worker()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_worker_common.py) |
| Draft TP Context | [`draft_tp_context()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L706-L718) |
| Speculative TP decision sync | [`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py) |
| 默认 SpecTpSync 配置 | [`SGLANG_SPEC_TP_SYNC`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375) |
| Draft ForwardBatch 构造 | [`DraftBlockProposer._run_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L369-L473) |
| DP/MoE sync metadata | [`_fill_dp_moe_sync_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L475-L512) |
| Spec token-count scaling | [`spec_scale_global_num_tokens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_info.py#L451-L465) |
| DP-tier 单元测试 | [`test_dspark_dp_tier.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/spec/dspark/test_dspark_dp_tier.py) |
| Draft Stage / MoE bridge 复用 | [`DSparkV4Stage`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L609-L778) |
| Target Attention→MoE bridge | [`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915) |
| Verify Planner | [`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py) |
| Target Verify | [`TargetVerifyExecutor`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py) |
| NPU Draft model differences | [`DeepseekV4ForCausalLMDSpark`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L781-L815) |
