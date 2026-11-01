# 多卡投机解码：Draft、Verify 与 Accept 怎样保持一致

假设同一请求在 Rank0 接受三枚 Draft，Rank1 只接受两枚。两边局部 GEMM 都可能继续运行，下一轮却已使用不同长度、缓存位置和起始 Token。最后看到的 collective 等待，可能只是分歧一路传播后的结果。

多卡模型允许权重、hidden 分片，协作 Rank 对同一请求的后续决定却要对应。下面先找出哪些 Rank 共同服务请求，再沿 DSpark Site、EAGLE 分支追候选、验证长度、接受结果怎样交接。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线为 PP1 的 V4、DSpark V2、EAGLE/NextN、DPA、Ascend NPU。PP relay 的跨 stage 状态不在本篇展开；TP16/DP8 用于组划分示意，模型可运行条件另行核对。

</details>

## 一、先区分计算分片与执行决策

列并行中，Rank0 算 `X @ W0`，Rank1 算 `X @ W1`，权重分片不同，本地结果自然不同。MoE 也可以由 EP0–EP3 分别持有 Expert0–Expert3，让 activation 到达各自专家。这些差异属于模型分工。

而决定后续执行的状态，需要按请求对齐。下表用 Data Plane、Control Plane 区分两类对象：

| 层次 | 可以不同吗 | 例子 |
| --- | --- | --- |
| Model Data Plane | 可以，甚至设计上就应该分片 | Q/K/V shard、expert activation、A2A buffer、TP partial output |
| Speculative Control Plane | 同一逻辑请求的协作 Rank 必须达成一致 | Draft token、verify length、accept path、bonus、new seq_len |

设两个 Attention-TP Rank 共同验证一条请求，尚未触及长度上限，接受 Draft 后还要加一枚 Target bonus。Rank0 的 correct_len=3，Rank1 为 2：

| 状态 | Rank0 | Rank1 |
| --- | --- | --- |
| 已接受 Draft 数 | 3 | 2 |
| `commit_lens` | 4 | 3 |
| 新 `seq_len` | P+4 | P+3 |

P 表示本轮之前的确认长度。差一枚会继续改变位置、缓存映射、Draft 起点，两边逐渐偏离同一段历史。

ROCm Greedy 的注释描述了相同风险：Accept 后广播 predict、accept_index、num_correct_drafts，防止 committed seq_lens 不同，破坏下一轮 collective。[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L800-L1029)

因此接受数、推进长度、新 bonus 要在协作者间对应，模型 Tensor 则继续按分片计算。

## 二、哪些 Rank 共同处理同一条请求

同步组先取决于谁处理同一条请求。DPA 的不同副本可以服务不同请求，整个 TP world 因而不总是一组相同决定。

Attention 将 TP block 分解为 DP × CP × TP。

[`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281) 的派生关系为：

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

取教学拓扑 TP=16、DP=8、CP=1、PP=1，开启 DP Attention，得到 attn_dp_size=8、`attn_tp_size=16/8/1=2`。

坐标展开为：

~~~text
tp_rank
=
(attn_dp_rank * attn_cp_size + attn_cp_rank)
* attn_tp_size
+
attn_tp_rank
~~~

其中 Attention TP 变化最快。

从 0 开始的 16-rank TP block 分成八个 Attention-DP 副本：

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

`initialize_model_parallel()` 按连续区间建组：

~~~python
st = (
    tp_group_idx * tensor_model_parallel_size
    + cp_dp_combined_idx * attn_tp_size
)

en = st + attn_tp_size

ranks = list(range(st, en))
~~~

入口见 [Attention-TP group construction](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2742-L2774)。

Rank0/1 服务本组请求，候选、接受需要一致。Rank2/3 属于另一组，可以处理另一条请求，所以 Rank0、Rank2 的 bonus 可以不同。

### DSpark 与 EAGLE 的 attn_tp_group

[`DSparkWorkerV2.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L120-L230) 选择 DSpark 同步组：

~~~python
self._tp_sync = SpecTpSync(
    parallel.attn_tp_group
    if parallel.enable_dp_attention
    else parallel.tp_group
)
~~~

EAGLE Sampling 使用：

~~~python
tp_group = (
    get_parallel().attn_tp_group
    if is_dp_attention_enabled()
    else get_parallel().tp_group
)
~~~

DPA 以 Attention TP 覆盖相同请求的协作者。前面的 TP16/DP8 只示意建组，固定 V4 MoE Draft Worker 要求 attn_tp_size=1，部署还受这项限制。

## 三、DSpark 怎样同步决策和布局参数

DSpark 用 Site 标记同步边界，环境配置可按 Site 开关。

定义为：

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

`SGLANG_SPEC_TP_SYNC` 默认为 `all`。

src=0 使用组内编号：组 `[2,3]` 的源就是全局 Rank2。Site 已开启、组宽大于一时才真正广播，单 Rank 继续使用本地结果。

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

Multinomial 另有 `DSPARK_DRAFT_MULTINOMIAL`。

提议实现见 [`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200)。

同一请求若在 Rank0 首提 A、Rank1 首提 B，后面的 Verify 已有不同输入。事后统一接受长度也无法补回这些上下文，所以候选本身就是同步点。

<a id="verify_lens-control-decision-layout"></a>

### Planner 用 verify_lens 决定布局

Planner 同步逐请求验证长度：

~~~python
verify_lens = ScheduleVerifyLensTopk.execute(...)

self._tp_sync.sync(
    DSPARK_PLAN,
    verify_lens,
)
~~~

入口见 [`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py#L575-L603)。

这里显式广播 verify_lens。positions、缓存位置、buffer 仍由各 Rank 本地构造。

各 Rank 根据它确定 Verify 行数，再结合请求顺序、Prefix 长度规划位置和本地 KV。前面的请求、边界也要对应，长度 Tensor 才能派生出相同逻辑布局。

### Accept 同步的是最小决定量

Target 验证后同步 `correct_len`、`bonus`、`cap_trim_lens`：

~~~python
self._tp_sync.sync(site, correct_len)
self._tp_sync.sync(site, bonus)
self._tp_sync.sync(site, cap_trim_lens)
~~~

接着各自运行 `FinalizeAcceptLens`、`BuildOutTokens`，算出 `commit_lens`、`new_seq_lens`、`out_tokens`。

实现见 [`TargetVerifyExecutor.accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206)。

广播少量决定，再在本地派生序列状态，避免传递全部中间 buffer。这个过程依赖各 Rank 的请求和其他输入已经对应。

<a id="spectpsync-decision-broadcast"></a>

### SpecTpSync 还协调初始化资源

初始化使用 `available_memory_gb(...)` 协调可用资源。

DSPARK_MEM 开启时，经指定 group 取最小可用显存，影响 Draft Graph、folded sampling 等初始化选择。

例如：

~~~python
available_mem = self._tp_sync.available_memory_gb(
    DSPARK_MEM,
    ...,
    group=self._draft_graph_group,
)
~~~

入口见 [`DSparkWorkerV2.init_cuda_graphs()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)。

最小值让各 Rank 不会按各自余量选择不兼容容量。DSPARK_MEM / DFLASH_MEM 协调的是资源边界。

hidden states、Q/K/V shard、MoE router activation、EP dispatch buffer、KV page 内容、positions、`out_cache_loc` 仍由各自子系统管理；它们分别具有逻辑对应、物理分片的要求。

## 四、EAGLE NPU Greedy 的调用与一致性条件

EAGLE/NextN 在 `eagle_sample()` 内按分支处理，没有直接沿用 DSpark 的 Site 机制。

### Greedy 分支如何进入

Greedy 分流条件为：

~~~python
return (
    is_all_greedy
    or is_cpu
    or is_xpu
    or (is_hip and not use_rejection_sampling)
)
~~~

实现见 [`_verify_uses_greedy()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L695-L710)。

NPU 的 is_all_greedy 成立时，先本地执行：

~~~python
target_predict = argmax(target_logits)

predict, accept_index, num_correct_drafts = (
    verify_tree_greedy_func(...)
)
~~~

随后仅在这个平台条件下广播：

~~~python
if _is_hip:
    broadcast(...)
~~~

位置见 [`eagle_sample()` Greedy branch](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L810-L844)。

因此固定 CUDA/NPU Greedy 在 `eagle_sample()` 这一层没有额外的 rank0 Accept broadcast。这个范围之外的候选、模型执行仍有各自约束。

<a id="npu"></a>

### NPU Greedy 继续依赖上游输入与执行一致

ROCm 分支留下了风险注释：

~~~text
per-rank draft tokens can differ
→ accepted drafts differ
→ committed seq_lens desync
→ next TP collective can deadlock
~~~

HIP 因而在这里显式广播。

固定 NPU Greedy 没有同样覆盖，需要上游候选、Target argmax、遍历过程保持一致。验证实际配置时，要沿这些边界比较逐 Rank 结果。

<a id="sampling-path"></a>

### Sampling 显式同步结果

非 Greedy Kernel 完成后广播：

~~~python
tp_group.broadcast(predict, src=0)
tp_group.broadcast(accept_index, src=0)
tp_group.broadcast(num_correct_drafts, src=0)
~~~

注释说明，浮点非确定性可能使不同设备的 target_probs 略有差异，随后采到不同 Token，因此广播 rank0 结果。

分支见 [`eagle_sample()` Sampling branch](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L881-L1007)。

三类结果分别约束：

| 广播结果 | 决定什么 |
| --- | --- |
| `predict` | 本轮真正输出的 Token |
| `accept_index` | 树中接受哪条节点路径 |
| `num_correct_drafts` | 序列推进长度 |

两条路径 A→B→C、A→D→E 长度相同，提交的节点、KV 却不同。因此同步除了数量，还需要路径和 ID。

## 五、决策同步与模型通信怎样配合

接受决定对齐以后，模型还要正确分配数据。DPA+MoE 的行布局转换由另一条通信路径完成。

`attn_tp_size > 1` 且开启 MoE A2A backend 时，post-attention 行被拆分：

~~~python
hidden_states.tensor_split(
    attn_tp_size
)[attn_tp_rank]
~~~

各 Attention-TP Rank 提交自己的 Token slice。

MoE 后用 `attn_tp_all_gather` 恢复下一层行布局。

实现见 [`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)。

这些通信安排 Token 行归属、Tensor 分片、专家分发、部分结果合并。SpecTpSync 则对齐 Draft、验证长度、接受结果、bonus。前者提供正确数据，后者让后续执行使用同一决定。

### `verify_lens` 怎样影响实际布局

两个协作 Rank 都收到 `verify_lens=4`，还需要确定它们是同请求、同轮次、同 Prefix 边界，这四行才有共同的逻辑身份。

对应行的本地地址可以不同，Rank0 用 slot120、Rank1 用 slot500，都可能正确。相反，两边都用 slot120，也未必属于同请求。positions 的逻辑含义要对应，`out_cache_loc` 则保留本地地址。

日志要同时带有决定、逻辑归属、本地分片或布局，才能在地址不同的情况下追踪同一状态。

## 六、一次分歧怎样影响后续执行

继续使用开头的假设：Rank0 为 correct_len=3、bonus=X，Rank1 为 correct_len=2、bonus=Y。分歧向后传播为：

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

再进入 Attention TP、MoE EP、DeepEP dispatch/combine、AllGather/ReduceScatter 时，两边的 Token 数、行身份、通信次序可能已不同，继而出现 shape mismatch、错误结果、超时、collective hang。

超时的位置可能只是第一个等不到对端的调用。定位仍要按请求、轮次往前找分歧；Accept 是一个候选原因，资源、通信等其他原因也需由证据排查。

若更早在 Verify 前出现 Rank0 verify_len=4、Rank1 为 5，本轮执行形状就可能分叉。DSPARK_PLAN 因而要在执行前对齐预算，不能等接受结束再修正。

各边界按顺序接续：

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

Ascend 的上层广播、通用分布式 backend、MoE 分发各有调用层次。Group 可能使用 HCCL、ZBAL，DeepEP 则负责专家数据交换。

### HCCL：默认 NPU distributed backend

默认平台映射为：

~~~python
"npu":
    "hccl"
    if not SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
    else "zbal"
~~~

定义见 [`_DEVICE_TO_DISTRIBUTED_BACKEND`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)。

未启用 ZBAL 的普通配置中，tp_group、attn_tp_group、pp_group、moe_* device process groups 默认使用 HCCL，广播也由当前 group backend 承载。

<a id="zbal-npu-distributed-backend-deepep-adaptor"></a>

### ZBAL backend 与内存、DeepEP adaptor

`SGLANG_ZBAL_LOCAL_MEM_SIZE > 0` 选择 zbal，NPU 初始化还调用 zbal_init。具体部署使用哪个 backend，由实际配置决定。

- [`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)
- [`init_zbal()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/utils.py#L328-L415)

上层仍调用 `GroupCoordinator.broadcast()`，下层承载已经随 backend 改变。因此追 SpecTpSync 时，Group 和 backend 要一起记录。

<a id="deepep-hcclzbal"></a>

### DeepEP 承接专家 Dispatch 与 Combine

DeepEP 将 Token activation 发往 Router 选中的专家，并返回专家贡献。

接口为：

~~~text
dispatch(hidden_states, topk_output)

combine(expert_outputs)
~~~

实现见 [`DeepEPDispatcher`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L972-L1126)。

它跨相应 EP Rank 组织这些专家输入、输出。

接受广播则对齐后续决定，两类交换的内容、组、时机各不相同。

### ZBAL 与 DeepEP 的组合路径

ZBAL 还可提供 DeepEP 适配器：

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

入口见 [`deepep.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L1-L60)。

这条组合保留 Dispatcher 的 Dispatch/Combine 接口，底层 Buffer、Config 来自 ZBAL adaptor。

职责对应为：

| 层 | 角色 | 典型对象 |
| --- | --- | --- |
| Speculative Control | 决定“下一步算什么” | Draft token、verify_len、accept_index、bonus |
| General Distributed Backend | 承载普通 process-group collective | HCCL，或启用后的 ZBAL backend path |
| MoE A2A Semantics | Router 后的 expert dispatch / combine | DeepEP |
| Model Layout Logic | 决定 Tensor 在哪些 group 间 split/gather | TP / DP Attention / EP runtime |

调用关系为：

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

虚线表示 Buffer/adaptor 的来源，决策广播不因此经过 DeepEP。

## 八、按执行边界检查一致性

逐边界检查时，把投机决定、逻辑状态、通信调用放在一起：

| 检查层 | 需要对应的内容 | 典型观察字段 |
| --- | --- | --- |
| 投机决策 | 候选、验证范围、接受路径与新 bonus | Draft token、`verify_lens`、`correct_len`、`accept_index`、`bonus` |
| 逻辑状态 | 请求身份、投机轮次、位置与序列进度 | `req_pool_idx`、round、positions、前后 `seq_len` |
| 通信调用 | 参与者、Group、调用顺序与张量形状 | global/TP/Attention/EP rank、group、shape |

DSpark 的显式 Site 与 EAGLE 的具体分支，决定了在哪里比较或同步结果。Sampling/UNO、HIP Greedy 有广播；固定 CUDA/NPU Greedy 在 eagle_sample 内则依赖前述上游和执行约束。

沿同一请求的 Draft→Verify→Accept→Commit 找到首个不同值，再把它接到后面的 KV、collective 现象，才能判断问题从哪里开始。

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
