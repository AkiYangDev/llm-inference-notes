# DeepSeek-V4 DSpark 的并行限制与 Draft 布局

同样是 DeepSeek-V4，在 `TP32 / DP16` 下，Target 的 Attention 与 MoE 之间有显式的 Token 行布局转换；DSpark 的 MoE Draft 却会在启动时被拒绝，因为这时 `attn_tp_size=2`，不满足 Worker 要求的 `attn_tp_size=1`。

Target 与 Draft 的可用并行参数需要分别判断。Draft 除了计算候选，还要让各 Rank 对请求数、候选行数和接受结果保持对应。下面先找启动检查，再跟着同一批请求看 Draft 使用哪个通信组、计数怎样换算，以及哪些决定需要同步。本文定位的是固定版本的限制与执行约束，没有完成解除限制后的运行验证。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。讨论 DeepSeek-V4 MoE draft + DSpark V2 + DP Attention。限制位于通用 DSpark worker 的配置检查中。NPU Draft 还涉及 ModelSlim、独立 vocab modules 和 shared-expert fusion 等平台分支，后文分别说明。

</details>

## 一、启动限制发生在哪里

开启 DP Attention 后，`dp_size` 决定 Attention 侧怎样划分请求，剩余的 TP 宽度再由同一组请求内的 Rank 分担。沿用前一篇在普通、非弹性布局下的宽度关系：

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

这里 `attn_tp_size=2` 表示每个 Attention-DP 副本内，有两个 Rank 协同处理相应请求。把它代入 DSpark 的启动检查，三个条件同时成立：

~~~text
enable_dp_attention = True
draft_is_moe = True
attn_tp_size > 1
~~~

所以 TP32 / DP16 在创建 DeepSeek-V4 draft worker 之前就被拒绝。

检查发生在通用 DSpark Worker 中，条件没有 `is_npu()` 或 HCCL 判断。因此即使现场运行在 Ascend 910C，也不能仅凭这条报错把问题归到 NPU 通信库。

报错文本进一步给出一个重要线索：`attn_tp > 1 corrupts the MoE-under-DP all-reduce`

这段报错把排查方向指向 MoE-under-DP 的 all-reduce，但没有给出哪一行数据、哪一步归约先出错。要进一步定位，仍需对照各 Rank 送入归约的行分别属于哪个请求、它们是否是同一个结果的局部贡献。删除检查只能让程序继续进入该路径，不能证明这些关系已经正确。

## 二、Dense Draft 与 MoE Draft 的并行上下文

先看 DSpark 怎样选择 Draft 的执行上下文。这里的 context 会临时改变代码查询到的 TP Group，所以即使模型层函数相同，进入时看到的分组也可能不同。Worker 初始化先区分 Draft 类型：

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

DeepSeek-V4 MoE Draft 的条件不同：它不会给整个 Forward 套用上面的局部 TP Context：

~~~text
_draft_is_moe = True
=> _draft_dp_context_enabled = False
=> DSparkWorkerV2._draft_context() = nullcontext()
~~~



### MoE Draft 为什么保留跨 DP 的参与关系

继续看 `DraftBlockProposer` 的构造。MoE Draft 与 DP Attention 同时开启时，会设置：

~~~python
dp_moe_sync = (
    self._draft_is_moe
    and get_parallel().enable_dp_attention
)
~~~

随后 Draft forward 会额外填充 DP/MoE 同步 metadata；空闲 Rank 也需要执行 `run_idle_participation()` 进入 draft model forward。

这些调用把上下文选择与后续执行接上了：MoE Draft 保留跨 DP / MoE Rank 的计数同步和通信参与关系，不能只让本地有请求的 Attention 子组独自运行。与 Dense Draft 相比，它需要继续使用 MoE 所涉及的通信拓扑。

还有一个更细的点：`DraftBlockProposer._base_logits_context()` 在 `dp_moe_sync=True` 时，会短暂使用 `draft_tp_context(attn_tp_group)` 来计算 proposal-side base logits。`attn_tp_group` 在这两个阶段的用途不同：

~~~text
Draft model forward / MoE
→ 不整体 patch 到 attn_tp_group

部分 proposal/logit 计算
→ 可以临时进入 attn_tp_group context
~~~

因此读到 `attn_tp_group` 时，还要看它包住的是整个模型 Forward，还是其中一段 logits 计算。两个位置用了同一个组名，并不表示它们负责相同范围的执行。

## 三、每个 Rank 的 Token 计数怎样传递

选好通信组之后，还要让所有参与者用同一种单位理解本轮工作量。一次 Draft 可能给每条请求准备多行输入，因此“有几条请求”和“有几行 Token”是两个数。

在这里的 decode-family DP 同步中，`ScheduleBatch.global_num_tokens` 保存各相关 Rank 的原始同步计数列表，普通 decode-family 轮次中按每 Rank 的请求数计。名字里虽然有 tokens，读字段时仍要看产生它的阶段。

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

例如前面的 `[1, 3, 0, 2]`，Graph 准入需要知道四个参与者各有几条请求；本次 Draft Forward 若每条请求占六行，DP/MoE 布局则需要 `[6, 18, 0, 12]`。两份计数分别服务于请求级调度和 Token 行级执行，不能拿缩放后的 18 当作 18 条请求，也不能拿原始的 3 当作本次仅有三行输入。这个六行宽度来自例子中的 `num_tokens_per_req`，不应直接等同于所有配置的候选数。

### Idle Rank participation 也说明这些 metadata 属于 collective contract

当本地没有请求，但其他 DP Rank 仍在跑 MoE draft 时，`run_idle_participation()` 会构造 `ForwardMode.IDLE` 的空 batch，填入同一套同步 metadata，然后仍调用 draft model forward。

~~~text
本地 Token 数为 0
≠
这个 Rank 可以完全跳过本轮 distributed MoE path
~~~

零行参与者仍可能是同一个 Collective 的成员。其他 Rank 按计划进入通信时，它也必须按相同轮次参加；否则问题可能表现为等待，而非某个本地 Tensor 的数值错误。这里的空 Forward 就是为了保留这种参与关系。

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

按 TP32 / DP16 / CP1 的拓扑计算，这会是每个 DP 副本内由两个 Rank 组成的 Attention TP 组；前面的 Worker 检查仍会拒绝该 MoE Draft 配置，这里只说明组选择规则。`SpecTpSync` 在选定组内同步投机决策，模型中的 DP/MoE 数据通信仍由相应执行路径负责。

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

因此默认配置下，多 Rank 组中显式调用 `self._tp_sync.sync(site, tensor)` 的决策会取组内源 Rank 的值。这里的 `src=0` 指组内编号 0，不一定是全局 Rank0；单 Rank 组则无需执行这次跨 Rank 广播。

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

只对齐候选或接受结果，无法替代 Draft 模型对 Token 行的分配与归约。两部分都要核对，才能判断更宽 Attention TP 下是哪一个边界不成立。

## 五、Planner 的 DP Tier Gather 条件

Planner 中也能看到 `attn_tp_size == 1`，但它出现在另一处条件里：

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

这些条件一起决定是否开启 compact ragged-verify 的 DP-tier Token 计数收集。这里的 Tier 可以理解为本轮 Verify 选择的执行容量档位，计数会影响不同 Rank 如何选择与组织相应容量。这个条件控制的是一条 Planner 优化分支，不能把它与 Worker 的启动拒绝检查当作同一处限制。

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

Draft 因此复用了这部分 Attention→MoE bridge，也就是衔接两个阶段行布局的转换代码。排查时继续检查调用前后，每行属于哪个请求、计数按什么单位解释。

### 从限制定位到运行验证

前文已经确定三处边界：Worker 拒绝 MoE Draft 的 `attn_tp>1`；Draft 复用父类的 MoE bridge；投机决策在 Attention TP 子组内同步。还缺的是运行证据：哪一次布局转换或归约首先偏离了正确结果。

| 验证阶段 | 对照方式 | 要回答的问题 |
| --- | --- | --- |
| 可运行基线 | 保持 `attn_tp=1`，保存逐阶段输出与计数 | 正确结果和通信顺序是什么？ |
| Eager 布局实验 | 开发环境中检查更宽 Attention TP，逐边界对齐 | 首次数值或行归属分歧在哪里？ |
| 空闲 Rank 与批次变化 | 包含零请求 Rank、不均衡请求数 | 计数缩放和 collective 参与是否一致？ |
| Graph 与 Verify 模式 | 在正确 Eager 基础上分别验证 | padding、tier 选择或 metadata 是否改变语义？ |
| Ascend 目标环境 | 使用实际 ModelSlim、通信后端与部署拓扑 | 平台分支能否保持同样的正确性？ |

上表是后续验证方案，不是本文已经执行的实验结果。先在受支持的布局下建立目标模型的正确基线，再逐边界比较更宽拓扑的输出，才能判断差异出在 bridge、metadata 还是后续执行路径。

### 按依赖顺序推进验证

下面保持 CP1、PP1 与其余可比条件一致。TP32 / DP32 用于建立受支持宽度的基线，目标模型与平台是否满足其他条件仍需核对；绕过 guard 仅用于隔离开发环境中的定位：

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

例如两个 Rank 都有四行输入，shape 检查会通过；但如果其中一边的第二行已经对应另一个请求，后续归约仍然可能错误。把请求身份、轮次与行位置一起记录，才能把这种分歧与通信本身的故障区分开。

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

CUDA Eager 结果只能作为相应路径的对照，无法替代 Ascend 910C 的验收。目标部署仍要覆盖实际使用的 ModelSlim、通信后端、Graph 与跨机拓扑，确认平台分支没有改变前面建立的行与状态对应关系。

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
