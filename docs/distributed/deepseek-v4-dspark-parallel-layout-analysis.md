# DeepSeek-V4 DSpark 的并行限制与 Draft 布局

DeepSeek-V4 Target 在 `TP32 / DP16` 下有 Attention→MoE 的 Token 布局转换，DSpark MoE Draft 却会在启动时被拒绝：派生的 `attn_tp_size=2` 不符合 Worker 的宽度 1 要求。

两条执行路径要分别判断。Draft 除了生成候选，还需让各 Rank 的请求数、候选行、接受结果保持对应。下面先定位检查，再追踪 Group、计数换算和决策同步，解释已确认的约束；本文没有完成解除限制后的运行验证。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线为 V4 MoE Draft、DSpark V2 和 DP Attention。检查位于通用 Worker；NPU 另有 ModelSlim、独立词表模块、shared-expert fusion 等分支，需要各自核对。

</details>

## 一、启动限制发生在哪里

DPA 将请求分到 Attention DP，再由组内剩余 TP 宽度协作。普通非弹性布局中，`attn_dp_size=dp_size`，`attn_tp_size=tp_size/attn_dp_size/attn_cp_size`。

TP32/DP16、CP1、DP Attention 开启时，得到 attn_dp_size=16、attn_tp_size=2，即每个 Attention-DP 内有两个 Rank 共同处理请求。这同时命中 `enable_dp_attention=True`、`draft_is_moe=True`、`attn_tp_size>1` 三个拒绝条件。

TP32 / DP16 因此在创建 V4 Draft Worker 前就被拒绝。

条件没有 `is_npu()` 或 HCCL 判断。即使发生在 Ascend 910C，也不能只靠这条错误归因于 NPU 通信库。

报错给出的线索为 `attn_tp > 1 corrupts the MoE-under-DP all-reduce`。

它指向 MoE-under-DP 的归约风险，但未定位首个错误行。仍需确认各参与者归约的行属于谁、数值是否为同一结果的局部贡献。删除检查只能放行执行，不会自动修好这些关系。

## 二、Dense Draft 与 MoE Draft 的并行上下文

先看 Worker 如何选 Draft context。context 可以临时改变代码查询到的 TP Group，因此相同模型层也可能在不同组中执行。

~~~python
self._draft_is_moe = draft_is_deepseek_v4()

self._draft_dp_context_enabled = (
    get_parallel().enable_dp_attention
    and not self._draft_is_moe
)
~~~

只有 `_draft_dp_context_enabled=True` 时，整体 `_draft_context()` 才使用：

~~~python
return draft_tp_context(
    get_parallel().attn_tp_group
)
~~~

`draft_tp_context()` 调用 `patch_tensor_parallel_group(tp_group)`，其注释为 `Draft model doesn't use dp and has its own tp group.`。

实现见 [`draft_tp_context()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L706-L718)。

Dense Draft + DPA 因而将 Draft TP Group 临时切到 `attn_tp_group`，让每个 Attention-DP 副本内的局部组运行 Draft。V4 MoE Draft 的 `_draft_is_moe=True`，所以 `_draft_dp_context_enabled=False`，整体 `_draft_context()` 返回 `nullcontext()`，不做同样的切换。

### MoE Draft 为什么保留跨 DP 的参与关系

构造 Proposer 时，MoE Draft 与 DPA 同时开启将设置：

~~~python
dp_moe_sync = (
    self._draft_is_moe
    and get_parallel().enable_dp_attention
)
~~~

forward 随后填充 DP/MoE 同步 metadata，空闲 Rank 也通过 `run_idle_participation()` 进入模型。

所以 MoE Draft 需保留跨 DP/MoE 的参与关系，不能只让本地有请求的 Attention 子组独立运行，这与 Dense Draft 不同。

还有局部例外：`_base_logits_context()` 在 `dp_moe_sync=True` 时，可暂时用 `draft_tp_context(attn_tp_group)` 计算 proposal base logits。

因此应看 context 包裹整个 Draft model 还是局部 logits。同名组参与不同阶段，不代表整体分工相同。

## 三、每个 Rank 的 Token 计数怎样传递

Group 确定后，还要对齐单位。每请求在 Draft 可占多行，请求数和输入行数不能混用。

decode-family 同步中的 `ScheduleBatch.global_num_tokens` 保存原始各 Rank 计数，普通轮次按请求数解释。字段名含 tokens，仍需追到其生成阶段。

换算函数的 docstring 为：[`spec_scale_global_num_tokens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_info.py#L451-L465)

~~~text
Scale the raw per-rank sync values
(request counts on decode-family rounds)
into this forward's token units
using the spec input's uniform per-request widths.
~~~

原始 `batch.global_num_tokens` 是逐 Rank 的请求计数列表，再乘 `spec_info.num_tokens_per_req`，换为当前 Draft Token 行数。



固定测试举例：

~~~text
raw per-rank request counts:
[1, 3, 0, 2]

num_tokens_per_req = 6

scaled token units:
[6, 18, 0, 12]
~~~

测试入口见 [`test_dspark_dp_tier.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/spec/dspark/test_dspark_dp_tier.py)。

### DSpark 为什么同时保存 raw 和 scaled 两份

`_fill_dp_moe_sync_metadata()` 同时保留两套单位：

| 字段 | 单位 | 用途 |
| --- | --- | --- |
| `original_global_num_tokens_cpu` | 原始逐 Rank 请求数 | Graph bucket 与准入 |
| `global_num_tokens_cpu` / `global_num_tokens_gpu` | 缩放后的 Draft Token 行数 | DP/MoE 同步执行几何 |

对应实现见 [`_fill_dp_moe_sync_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L475-L512)。

例如 raw `[1, 3, 0, 2]` 用于请求级 Graph 准入；每请求六行时，执行几何使用 `[6, 18, 0, 12]`。不能把 18 当请求数，也不能把 3 当实际输入行数。六行来自本例 num_tokens_per_req，不代表所有配置都提六枚候选。

### Idle Rank participation 也说明这些 metadata 属于 collective contract

本地没有请求而其他 DP 仍执行 MoE Draft 时，idle 路径构造 `ForwardMode.IDLE` 空 batch，填同套 metadata 并调用 forward。

零输入 Rank 仍可能属于本轮 collective。若其他参与者进入而它跳过，可能表现为等待，未必先出现数值错误。空 forward 用来维持一致轮次。

## 四、SpecTpSync 同步哪些决策

Worker 按以下规则建立 `SpecTpSync`：

~~~python
self._tp_sync = SpecTpSync(
    parallel.attn_tp_group
    if parallel.enable_dp_attention
    else parallel.tp_group
)
~~~

DPA 下同步组为 `attn_tp_group`。

在 TP32/DP16/CP1 的概念拓扑中，该组有两个 Rank，但 Worker 仍会拒绝这一 MoE Draft 配置。这里仅说明同步组选择，DP/MoE 数据交换由其执行路径另行负责。

类注释为 `Broadcasts a speculative decision from rank 0 to its TP group.`。

sync 内的操作为：

~~~python
self._tp_group.broadcast(values, src=0)
~~~

实现见 [`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)。

显式 site 包含：

~~~text
DSPARK_DRAFT_GREEDY
DSPARK_DRAFT_SAMPLE
DSPARK_PLAN
DSPARK_ACCEPT_GREEDY
DSPARK_ACCEPT_SAMPLE
DSPARK_TARGET
...
~~~

默认 `SGLANG_SPEC_TP_SYNC = all`。

见 [`environ.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375)。

所以多 Rank 默认对显式 site 采用源 Rank 值。src=0 是组内编号，未必是全局 Rank0；组宽 1 则无需跨 Rank 广播。

这套接口不会顺便同步全部 hidden states、MoE router activation、EP dispatch buffer、DP 全局 Token 几何或 KV pages；这些对象由其他 Runtime、communicator 管理。

SpecTpSync/Attention-TP Group 负责投机决定一致，DP/MoE/EP Runtime 负责 Draft 的分布式计算。对齐候选和接受结果，无法替代模型内部行分配与归约；定位更宽 Attention TP 的问题，要分别核对两者。

## 五、Planner 的 DP Tier Gather 条件

Planner 另有 `attn_tp_size==1` 条件，用于控制 `_dp_tier_gather_enabled`。它需要同时满足：

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

这组条件开启 compact ragged 的 DP-tier Token 计数收集，用来协调本轮执行容量。它属于 Planner 优化分支，与 Worker 启动检查不是同一位置。

Worker 在 Planner 初始化前已拒绝 MoE Draft 的更宽 Attention TP。扩展时还需继续核对 Planner 计数、宽度及 Graph tier，不能只改其中一处。

## 六、验证更宽 Attention TP 需要哪些证据

接着确认 Draft 实际复用了哪些布局转换。

`DSparkV4Stage` 继承 Decoder Layer，`_run_ffn()` 调用：

~~~python
self._run_moe_ffn_dp_sync(
    x,
    forward_batch,
    input_ids=None,
    input_ids_global=None,
)
~~~

对应实现见 [`DSparkV4Stage._run_ffn()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L701-L778)。

所以它已复用这部分 Attention→MoE bridge，剩余定位要在调用前后核对每行归属和计数单位。

### 从限制定位到运行验证

已知 Worker 拒绝宽度、Draft 复用 bridge、决策在 Attention 子组同步。尚缺的是运行证据：哪次转换或归约首先偏离正确结果。

| 验证阶段 | 对照方式 | 要回答的问题 |
| --- | --- | --- |
| 可运行基线 | 保持 `attn_tp=1`，保存逐阶段输出与计数 | 正确结果和通信顺序是什么？ |
| Eager 布局实验 | 开发环境中检查更宽 Attention TP，逐边界对齐 | 首次数值或行归属分歧在哪里？ |
| 空闲 Rank 与批次变化 | 包含零请求 Rank、不均衡请求数 | 计数缩放和 collective 参与是否一致？ |
| Graph 与 Verify 模式 | 在正确 Eager 基础上分别验证 | padding、tier 选择或 metadata 是否改变语义？ |
| Ascend 目标环境 | 使用实际 ModelSlim、通信后端与部署拓扑 | 平台分支能否保持同样的正确性？ |

表中是验证方案，尚未执行。需要先建立受支持拓扑的基线，再逐边界比较，才能分辨 bridge、metadata 或后续执行差异。

### 按依赖顺序推进验证

示意路线固定 CP1、PP1 和其他可比条件，TP32/DP32 也需核其余模型、平台条件。绕过 guard 仅用于隔离开发定位，不能视为部署方案。

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

逐请求、轮次记录：

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

在每个边界确认当前行的请求、speculative step、Rank，以及后一个 collective 怎样解释它。



例如两边 shape 都为四行，某边第二行却已属于另一请求，归约仍会错。记录身份和轮次才能把布局分歧与通信故障区分。

### Ascend NPU 上还要单独补一组验证

NPU 还有 `uses_own_vocab_modules = True` 的模型分支。

ModelSlim NPU 加载不支持把 shared experts 映射进 fused slots，因此禁用相应 fusion。

对应代码见 [`DeepseekV4ForCausalLMDSpark`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L781-L815)。

通用 DSpark 布局正确性与 Ascend NPU 平台正确性应分别验收。CUDA Eager 对照不能替代 Ascend 910C；目标平台还需覆盖实际 ModelSlim、通信、Graph 和跨机拓扑，确认平台分支维持同一行与状态关系。

## 七、区分模型布局、计数与决策

复用模型层不等于共用运行约束，Draft 还携带候选步数、窗口、接受结果，每次布局转换都要和 Token 行一致。

因此对 TP32 / DP16，先查阶段 Group，再对齐 raw 请求计数、scaled Token 行数和决策结果，才能从配置拒绝继续定位到具体执行边界。

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
