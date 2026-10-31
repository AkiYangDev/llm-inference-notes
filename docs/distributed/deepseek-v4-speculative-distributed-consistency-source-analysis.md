# 多卡投机解码：Draft、Verify 与 Accept 怎样保持一致

多卡投机中，模型权重和 hidden 可以分片，同请求的协作 Rank 却需对后续输入作一致决定。一边接受两枚、另一边接受三枚，下一轮长度、缓存位置和通信形状就可能分叉。

最终的 KV 错位或等待，可能源于更早的候选、接受分歧。下面先区分允许不同的计算数据与必须对齐的执行状态，再沿 DSpark、EAGLE 看同步组及分支。这里给出源码约束和排查方式，没有认定某次现场故障。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线为 PP1 的 V4、DSpark V2、EAGLE/NextN、DPA、Ascend NPU。PP relay 另有跨 stage 状态，不在此展开。TP16/DP8 仅示范组划分，模型可运行条件仍需分别核对。

</details>

## 一、先区分计算分片与执行决策

TP 各 Rank 的本地 Tensor 可以不同。例如列并行由 Rank0 计算 `X @ W0`、Rank1 计算 `X @ W1`，权重分片不同，本地输出自然不同。MoE 也可让 EP0–EP3 分别持有 Expert0–Expert3，对应 activation 出现在不同专家持有者上，符合分工。

因此要分别检查计算数据与决定后续执行的状态，表中称为 Data Plane、Control Plane：

| 层次 | 可以不同吗 | 例子 |
| --- | --- | --- |
| Model Data Plane | 可以，甚至设计上就应该分片 | Q/K/V shard、expert activation、A2A buffer、TP partial output |
| Speculative Control Plane | 同一逻辑请求的协作 Rank 必须达成一致 | Draft token、verify length、accept path、bonus、new seq_len |

假设同请求在两个 Attention-TP Rank 验证，未触长度上限，接受 Draft 后还需加一枚 Target bonus。Rank0 的 correct_len=3，Rank1 却为 2，就会产生以下分歧：

| 状态 | Rank0 | Rank1 |
| --- | --- | --- |
| 已接受 Draft 数 | 3 | 2 |
| `commit_lens` | 4 | 3 |
| 新 `seq_len` | P+4 | P+3 |

P 为本轮之前的确认长度。差一枚将继续影响位置、缓存映射和 Draft 起点，即使局部 GEMM 仍能执行，两边也已不在处理同一历史。

固定 ROCm Greedy 注释正说明此风险：Accept 后广播 predict、accept_index、num_correct_drafts，避免接受数不同使 committed seq_lens 分叉，破坏下轮 collective。[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L800-L1029)





所以协作 Rank 需对接受数、推进长度、新 bonus 保持一致，Tensor 则继续按模型分片。

## 二、哪些 Rank 共同处理同一条请求

同步范围先由谁共同服务请求决定。DPA 的不同副本可处理不同请求，不应将整个 TP world 无条件当作一组控制状态。

Attention 将 TP block 展开为 DP × CP × TP。

对应派生为：[`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281)

~~~python
attn_dp_size = (
    dp_size
    if enable_dp_attention
    else 1
)

attn_tp_size = (
    tp_size
    // attn_dp_size
    // attn_cp_size
)
~~~





取一个教学拓扑：TP=16、DP=8、CP=1、DP Attention 开启、PP=1。于是 attn_dp_size=8，`attn_tp_size=16/8/1=2`。

坐标展开为：

~~~text
tp_rank
=
(attn_dp_rank * attn_cp_size + attn_cp_rank)
* attn_tp_size
+
attn_tp_rank
~~~

Attention TP 变化最快。

从 0 开始的 16-rank TP block 因此分成八个 Attention-DP 副本：

| 副本 | Attention-TP Rank 组 |
| ---: | --- |
| 0 | [0, 1] |
| 1 | [2, 3] |
| 2 | [4, 5] |
| 3 | [6, 7] |
| 4 | [8, 9] |
| 5 | [10, 11] |
| 6 | [12, 13] |
| 7 | [14, 15] |

`initialize_model_parallel()` 按以下连续区间建组：

~~~python
st = (
    tp_group_idx * tensor_model_parallel_size
    + cp_dp_combined_idx * attn_tp_size
)

en = st + attn_tp_size

ranks = list(range(st, en))
~~~



构造入口见 [Attention-TP group construction](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2742-L2774)。

Rank0/1 共同服务本组请求，要对齐候选与接受；Rank2/3 属于另一组，可能服务不同请求，因此 Rank0 与 Rank2 的 bonus Token 不必相同。

### 为什么 DSpark / EAGLE 都会选择 `attn_tp_group`

DSpark 选择同步组的代码为：[`DSparkWorkerV2.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L120-L230)

~~~python
self._tp_sync = SpecTpSync(
    parallel.attn_tp_group
    if parallel.enable_dp_attention
    else parallel.tp_group
)
~~~





EAGLE Sampling 同样选择：

~~~python
tp_group = (
    get_parallel().attn_tp_group
    if is_dp_attention_enabled()
    else get_parallel().tp_group
)
~~~

DPA 下使用 Attention TP 覆盖相同请求的协作者。TP16/DP8 只是拓扑示意，固定 V4 MoE Draft 的 Worker 仍要求 attn_tp_size=1，不能将示意当作启动验证。

## 三、DSpark 怎样同步决策和布局参数

DSpark 用 Site 标识同步调用，环境配置可按 Site 开关，目的是让组内在相应边界采用同一决定。

类定义为：

~~~python
class SpecTpSync:
    """Broadcasts a speculative decision from rank 0 to its TP group."""
~~~

sync 执行：

~~~python
def sync(self, site, values):
    if site in self._sites:
        self._tp_group.broadcast(
            values,
            src=0,
        )
    return values
~~~



实现见 [`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)。

默认 `SGLANG_SPEC_TP_SYNC=all`。

src=0 是组内编号，例如 `[2,3]` 从全局 Rank2 广播。只有 Site 开启且组宽大于一才实际广播，单 Rank 沿用本地结果。



配置入口见 [`SGLANG_SPEC_TP_SYNC`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375)。

### Draft Token 是显式同步点

Greedy 候选经过：

~~~python
tp_sync.sync(
    DSPARK_DRAFT_GREEDY,
    argmax_token,
)
~~~

Sampling 候选经过：

~~~python
tp_sync.sync(
    DSPARK_DRAFT_SAMPLE,
    sampled_token,
)
~~~

Multinomial 还有 `DSPARK_DRAFT_MULTINOMIAL`。



提议实现见 [`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200)。

若同请求的首候选在 Rank0 为 A、Rank1 为 B，后续链便已从不同输入起步。最终只统一接受长度，无法修正前面的 Verify，候选选择本身就需对齐。

### `verify_lens` 也是 control decision，但它直接决定后续 Layout

Planner 同样同步逐请求验证长度：

~~~python
verify_lens = ScheduleVerifyLensTopk.execute(...)

self._tp_sync.sync(
    DSPARK_PLAN,
    verify_lens,
)
~~~



入口见 [`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py#L575-L603)。

这里只广播显式传入的 verify_lens，位置、缓存目标和 buffer 仍由各 Rank 本地构造。对齐布局参数并非复制整份布局。

各 Rank 根据 `verify_lens` 确定性地规划 Verify 行、positions 和本地 KV 分配。请求顺序、Prefix 长度等输入也要对应，才能生成一致的逻辑布局；广播一个长度 Tensor 无法补救已错位的请求。

### Accept 同步的是最小决定量

Target 之后分别同步 `correct_len`、`bonus`、`cap_trim_lens`：

~~~python
self._tp_sync.sync(site, correct_len)
self._tp_sync.sync(site, bonus)
self._tp_sync.sync(site, cap_trim_lens)
~~~

各 Rank 再本地运行 `FinalizeAcceptLens` 和 `BuildOutTokens`，得到 `commit_lens`、`new_seq_lens` 与 `out_tokens`。



实现见 [`TargetVerifyExecutor.accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206)。

同步的是最小决定量。只要请求与其他输入对应，各 Rank 就能按本地规则派生相同的逻辑序列状态，无需广播所有中间 buffer。

### `SpecTpSync` 也不是“纯 decision broadcast 类”

初始化还通过 `available_memory_gb(...)` 协调资源。

DSPARK_MEM 开启时，通过指定 group 取得最小可用显存，协调 Draft Graph、folded sampling 等初始化选择。

例如：

~~~python
available_mem = self._tp_sync.available_memory_gb(
    DSPARK_MEM,
    ...,
    group=self._draft_graph_group,
)
~~~



对应入口见 [`DSparkWorkerV2.init_cuda_graphs()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)。

这避免各 Rank 按本地余量选出不兼容容量。DSPARK_MEM / DFLASH_MEM 属于资源协调，也不等于统一模型 Tensor、物理布局。

它不会自动复制 hidden states、Q/K/V shard、MoE router activation、EP dispatch buffer、KV page 内容、positions 或 `out_cache_loc`。这些对象由各自子系统管理，逻辑对应与物理分片也各有要求。

## 四、EAGLE NPU Greedy 的调用与一致性条件

EAGLE/NextN 没有统一沿用 DSpark Site，需逐分支查看 `eagle_sample()`。

### Greedy 分支如何进入

Greedy 分流判断为：

~~~python
return (
    is_all_greedy
    or is_cpu
    or is_xpu
    or (is_hip and not use_rejection_sampling)
)
~~~



实现见 [`_verify_uses_greedy()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L695-L710)。

NPU 满足 is_all_greedy 时，本地执行：

~~~python
target_predict = argmax(target_logits)

predict, accept_index, num_correct_drafts = (
    verify_tree_greedy_func(...)
)
~~~

之后仅在下列平台条件广播：

~~~python
if _is_hip:
    broadcast(...)
~~~



对应代码见 [`eagle_sample()` Greedy branch](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L810-L844)。

因此 CUDA/NPU Greedy 在此函数层没有额外的 rank0 Accept broadcast。这个观察仅限 `eagle_sample()`，不能据此推断整个 NPU 路径都无一致性保障。

### 但源码没有解释“NPU 为什么不需要”

ROCm 则有明确风险注释：

~~~text
per-rank draft tokens can differ
→ accepted drafts differ
→ committed seq_lens desync
→ next TP collective can deadlock
~~~

因此 HIP 显式广播。

固定 NPU Greedy 在该函数没有额外覆盖，仍需由上游候选、Target argmax 和遍历约束保证一致。是否成立应查看实际配置的逐 Rank 结果。

### Sampling path 则明确同步

非 Greedy 完成 Kernel 后广播：

~~~python
tp_group.broadcast(predict, src=0)
tp_group.broadcast(accept_index, src=0)
tp_group.broadcast(num_correct_drafts, src=0)
~~~

注释说明，不同设备可能因浮点非确定性得到略有差异的 target_probs，进而采到不同 Token，所以统一广播 rank0 的结果。



该分支见 [`eagle_sample()` Sampling branch](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L881-L1007)。

三个结果分别控制不同状态：

| 广播结果 | 决定什么 |
| --- | --- |
| `predict` | 本轮真正输出的 Token |
| `accept_index` | 树中接受哪条节点路径 |
| `num_correct_drafts` | 序列推进长度 |

例如两条同长路径 A→B→C、A→D→E，即使 accept_len 相同，待提交节点和 KV 仍不同，所以要同步路径、ID，不能只对齐数量。这是假设例子，不是现场故障记录。

## 五、决策同步与模型通信怎样配合

决策一致后，模型通信仍要正确分配数据。DPA+MoE 的布局转换不会由接受长度广播替代。

当 `attn_tp_size > 1` 且启用 MoE A2A backend 时，模型拆分 post-attention 行：

~~~python
hidden_states.tensor_split(
    attn_tp_size
)[attn_tp_rank]
~~~

各 Attention-TP Rank 只提交自己的 Token slice。

MoE 后通过 `attn_tp_all_gather` 恢复下一层行布局。





实现见 [`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)。

这些通信处理 Token 行归属、Tensor 分片、专家分发、部分输出合并，以及 Attention 与 MoE 的布局转换。SpecTpSync 则对齐下一枚 Draft、验证长度、接受结果和新 bonus。两边均需成立：行分配正确不保证接受一致，接受一致也不能证明专家布局正确。

### `verify_lens` 怎样影响实际布局

例如两个协作 Rank 都拿到 `verify_lens=4`。这只能说明窗口长度对齐，还需核同一请求、同一轮次、同一 Prefix 边界，才能确认四行分别对应相同的逻辑位置。

对应逻辑位置的本地物理地址可以不同。Rank0 用 slot120、Rank1 用 slot500，都可能正确；反过来，两边同为 slot120，也不能证明属于同请求。positions 的逻辑含义应对应，`out_cache_loc` 的本地地址不要求相等。

日志因而需要同时记录控制状态、逻辑归属和物理分片/布局。只比较 Tensor 数值或物理地址，都不足以定位问题。

## 六、一次分歧怎样影响后续执行

假设首处分歧发生在 Accept：Rank0 得到 correct_len=3、bonus=X，Rank1 得到 correct_len=2、bonus=Y。沿后续执行，可看到分歧怎样逐步扩大：

<a id="commit-length"></a>
<a id="sequence-length"></a>
<a id="position"></a>
<a id="kv-ownership"></a>
<a id="draft"></a>
<a id="collective"></a>

| 后续状态 | Rank0 | Rank1 | 影响 |
| --- | --- | --- | --- |
| Commit Length | 4 | 3 | 接受结果先产生长度差 |
| Sequence Length | P+4 | P+3 | 正式历史边界不同 |
| 下轮 Position | 从 P+4 开始 | 从 P+3 开始 | 位置与历史访问分叉 |
| KV ownership | 按自身边界分配或复用 | 按另一边界分配或复用 | 本地地址可合法，逻辑历史却不同 |
| Draft 输入 | X | Y | 后续候选从不同 bonus 起步 |

再次进入 Attention TP、MoE EP、DeepEP dispatch/combine 或 AllGather/ReduceScatter 时，各 Rank 对 Token 数、行身份、collective 顺序的理解可能已不同，最终表现为 shape mismatch、错误结果、超时或 collective hang。

超时 collective 可能只是先等不到对端的位置，应沿请求、轮次往前找首个分歧。但超时本身不能证明 Accept 出错，还需排除其他原因。

Verify 前也可能先分歧：若 Rank0 的 verify_len=4、Rank1 为 5，本轮执行形状便可能不同。等接受后再统一已太迟，DSPARK_PLAN 因此需要同步。

边界顺序应为：

~~~text
Decision
   ↓
Converge
   ↓
Materialize Layout
   ↓
Distributed Compute
   ↓
Decision
   ↓
Converge
   ↓
Commit
~~~

## 七、HCCL、ZBAL 与 DeepEP 的职责

Ascend 上还需区分上层广播、通用 backend、MoE 分发。当前 Group 调用可能由 HCCL 或 ZBAL 承接，DeepEP 负责另一层的数据交换。

### HCCL：默认 NPU distributed backend

平台映射为：

~~~python
"npu":
    "hccl"
    if not SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
    else "zbal"
~~~



实现见 [`_DEVICE_TO_DISTRIBUTED_BACKEND`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)。

普通未启用 ZBAL 的配置中，tp_group、attn_tp_group、pp_group 和 moe_* device process groups 默认使用 HCCL，`attn_tp_group.broadcast(...)` 也由这个 backend 承载。判断调用的实际承载时，仍应核当前 group backend。

### ZBAL：可以替换 NPU distributed backend，并同时参与内存 / DeepEP adaptor

`SGLANG_ZBAL_LOCAL_MEM_SIZE > 0` 时选择 zbal，NPU 初始化还调用 zbal_init。必须看实际配置，不能将 HCCL 默认值外推到所有 Ascend 部署。



- [`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)
- [`init_zbal()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/utils.py#L328-L415)

启用后，上层仍调用 `GroupCoordinator.broadcast()`，实际承载取决于当前 backend。查 SpecTpSync 调用点时应同时记录 Group 和后端，不能固定写成 HCCL broadcast。

### DeepEP：不是 HCCL/ZBAL 的同层替代品

DeepEP 则将 Token activation 送到 Router 选中的专家，再 Combine 返回贡献。

接口为：

~~~text
dispatch(hidden_states, topk_output)

combine(expert_outputs)
~~~



实现见 [`DeepEPDispatcher`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L972-L1126)。

Router 选定专家后，由它将 Token activation 送到对应 EP Rank，再将专家贡献 Combine 回来。

接受广播对齐的是后续决定，与专家交换的内容、组、时机都可不同，不能相互替代。

### ZBAL 与 DeepEP 的组合路径

ZBAL 还能为 DeepEP 提供适配器：

~~~python
_use_zbal = (
    is_npu
    and SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
)

if _use_zbal:
    from zbal.zbal.deepep_adaptor import Config
    from zbal.zbal_buffer import Buffer
else:
    from deep_ep import Buffer, Config
~~~



对应入口见 [`deepep.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L1-L60)。

二者可在同一执行路径共存：Dispatcher 保留 Dispatch/Combine 接口，底层 Buffer、Config 由 ZBAL adaptor 提供。

可按职责分为：

| 层 | 角色 | 典型对象 |
| --- | --- | --- |
| Speculative Control | 决定“下一步算什么” | Draft token、verify_len、accept_index、bonus |
| General Distributed Backend | 承载普通 process-group collective | HCCL，或启用后的 ZBAL backend path |
| MoE A2A Semantics | Router 后的 expert dispatch / combine | DeepEP |
| Model Layout Logic | 决定 Tensor 在哪些 group 间 split/gather | TP / DP Attention / EP runtime |

调用关系如下：

~~~mermaid
flowchart TD
    S[Speculative Control]
    S --> G[attn_tp_group / tp_group broadcast]

    G --> B{NPU distributed backend}
    B --> H[HCCL default]
    B --> Z[ZBAL when enabled]

    M[MoE Router] --> D[DeepEP dispatch/combine]
    Z -. can provide adaptor / Buffer .-> D
~~~

虚线只表示 Buffer/adaptor 来源，不表示决策广播要经过 DeepEP。

## 八、按执行边界检查一致性

最终需同时核投机决定、逻辑状态、通信调用。分片与地址可以各自不同，同请求的后续决定必须对应。

| 检查层 | 需要对应的内容 | 典型观察字段 |
| --- | --- | --- |
| 投机决策 | 候选、验证范围、接受路径与新 bonus | Draft token、`verify_lens`、`correct_len`、`accept_index`、`bonus` |
| 逻辑状态 | 请求身份、投机轮次、位置与序列进度 | `req_pool_idx`、round、positions、前后 `seq_len` |
| 通信调用 | 参与者、Group、调用顺序与张量形状 | global/TP/Attention/EP rank、group、shape |

DSpark 按显式 Site 同步，EAGLE 按具体分支处理。Sampling/UNO、HIP Greedy 有广播，固定 CUDA/NPU Greedy 在 eagle_sample 内没有额外 accept-level broadcast，仍需上游和执行约束保证一致。

以请求、轮次对齐 Draft→Verify→Accept→Commit，寻找第一处分歧。KV 错位或等待可能是状态分叉，也可能来自资源、通信问题，需要由逐边界对照判断。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| Speculative TP Sync | [`spec_tp_sync.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py) |
| 默认 Sync Site 配置 | [`SGLANG_SPEC_TP_SYNC`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375) |
| Attention DP/TP width | [`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281) |
| Attention rank layout | [`derive_attention_ranks()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L262-L281) |
| Attention-TP group construction | [`parallel_state.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2742-L2774) |
| DSpark sync group selection | [`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L120-L230) |
| DSpark memory coordination | [`DSparkWorkerV2.init_cuda_graphs()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) |
| DSpark Draft decision sync | [`dspark_draft.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200) |
| DSpark Planner sync | [`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py#L575-L603) |
| DSpark Accept sync | [`accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206) |
| EAGLE Greedy/Sampling split | [`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L731-L1029) |
| EAGLE Verify → seq state | [`run_eagle_verify()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L462-L675) |
| EAGLE Tree compaction | [`_finalize_accept_tree_path()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L407-L459) |
| DeepSeek-V4 Attention→MoE bridge | [`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915) |
| Draft TP context | [`draft_tp_context()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L706-L718) |
| Ascend backend selection | [`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93) |
| ZBAL initialization | [`init_zbal()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/utils.py#L328-L415) |
| DeepEP / ZBAL adaptor | [`deepep.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L1-L60) |
| DeepEP dispatch/combine | [`DeepEPDispatcher`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L972-L1126) |
