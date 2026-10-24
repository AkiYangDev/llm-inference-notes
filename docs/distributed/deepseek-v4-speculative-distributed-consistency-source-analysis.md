# 多卡投机解码：Draft、Verify 与 Accept 怎样保持一致

多张卡共同执行投机解码时，权重和 hidden states 可以各自分片，但参与同一请求的 Rank 必须对下一轮输入达成一致。例如，一边认为接受了两个 Draft，另一边认为接受了三个，下一轮的序列长度、缓存位置和通信形状就可能分叉。

这类问题往往最后才表现为 KV 错位或通信等待，真正的分歧却可能出现在更早的候选选择或接受决策中。下面先用两个 Rank 的例子说明哪些数据可以不同、哪些决定必须对应，再沿 DSpark 与 EAGLE 的具体分支，看同步发生在哪个组内。文中讨论的是源码中的约束与排查方法，没有据此认定某次现场故障的根因。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线讨论 PP1 下 DeepSeek-V4 + SGLang speculative decoding，重点覆盖 DSpark V2、EAGLE/NextN、DP Attention 与 Ascend NPU。PP speculative relay 有额外跨 stage 状态传递，不在本文展开。文中的 TP16/DP8 用于演示 Group 划分；实际部署配置见对应模型的支持条件。

</details>

## 一、先区分计算分片与执行决策

普通 Tensor Parallel 中，不同 Rank 的 Tensor 本来就可以不同。

例如一个 Column Parallel Linear：

~~~text
Rank 0:
X @ W0

Rank 1:
X @ W1
~~~

两个 Rank 持有不同的权重分片，算出的局部输出自然可以不同。MoE 中也是如此：Router 决定 Token 交给哪些 Expert，专家可能分别存放在不同 Rank。用一个简化分布表示：

~~~text
EP0 → Expert 0
EP1 → Expert 1
EP2 → Expert 2
EP3 → Expert 3
~~~

某一阶段的 expert-local activation 只存在于拥有对应 Expert 的 Rank，也没有任何问题。

因此判断多卡投机解码是否一致时，要先区分模型数据与执行决策。下面的 Data Plane 指实际参与计算的数据，Control Plane 指决定后续执行的状态：

| 层次 | 可以不同吗 | 例子 |
| --- | --- | --- |
| Model Data Plane | 可以，甚至设计上就应该分片 | Q/K/V shard、expert activation、A2A buffer、TP partial output |
| Speculative Control Plane | 同一逻辑请求的协作 Rank 必须达成一致 | Draft token、verify length、accept path、bonus、new seq_len |

假设同一个请求在两个 Attention-TP Rank 上做 Verify，`correct_len` 表示接受的 Draft 数。本例未触及长度上限，接受候选后还会额外提交目标模型补出的一枚 bonus Token：

~~~text
Rank 0:
correct_len = 3

Rank 1:
correct_len = 2
~~~

Rank 0 会认为：`commit_lens = 4`

Rank 1 却认为：`commit_lens = 3`

下一轮：

~~~text
Rank 0:
seq_len = P + 4

Rank 1:
seq_len = P + 3
~~~

这里 P 是本轮之前的已确认长度。接受结果相差一枚，提交后长度就相差一；接下来分配的 KV 逻辑位置、下一轮 position 和 Draft 起点都可能跟着错开。即使两边仍能完成某次局部 GEMM，也不再是在推进同一条逻辑序列。

EAGLE 源码里有一段非常直接的注释说明这种风险。ROCm Greedy 路径会在 Accept 后 broadcast `predict / accept_index / num_correct_drafts`，因为不同 Rank 如果接受不同数量的 Draft，就会让 committed `seq_lens` 不同，并可能使下一轮 TP collective 失去一致调用契约。

固定源码：

[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L800-L1029)

共同执行同一逻辑请求的 Rank，需要对改变后续执行的离散决定达成一致：接受多少候选、序列前进多少、下一轮使用哪枚 bonus。Tensor 则可以继续按 TP 或 EP 分片存放。

## 二、哪些 Rank 共同处理同一条请求

接受决定要在哪些 Rank 之间一致，取决于谁在共同处理这条请求。开启 DP Attention 后，不同 DP 副本可以服务不同请求，不能把整个全局 TP World 都当成同一个请求的协作组。

开启 DP Attention 后，SGLang 会把原始 TP block 重新解释成 Attention-DP × Attention-CP × Attention-TP。

固定源码：

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

固定入口：

[`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281)

假设一个纯教学拓扑：

~~~text
TP = 16
DP = 8
CP = 1
DP Attention = ON
PP = 1
~~~

那么：

~~~text
attn_dp_size = 8

attn_tp_size
=
16 / 8 / 1
=
2
~~~

Rank layout 又是：

~~~text
tp_rank
=
(attn_dp_rank * attn_cp_size + attn_cp_rank)
* attn_tp_size
+
attn_tp_rank
~~~

其中 `attn_tp_rank` 是最快变化维。

所以在一个从 rank 0 开始的 16-rank TP block 内，源码实际构造的 Attention-TP groups 是：

~~~text
attn-DP replica 0:
[0, 1]

attn-DP replica 1:
[2, 3]

attn-DP replica 2:
[4, 5]

attn-DP replica 3:
[6, 7]

attn-DP replica 4:
[8, 9]

attn-DP replica 5:
[10, 11]

attn-DP replica 6:
[12, 13]

attn-DP replica 7:
[14, 15]
~~~

这个连续分组方式对应固定源码中的构造逻辑：`initialize_model_parallel()` 在 `attn_tp_size < tp_size` 时，按以下区间创建组：

~~~python
st = (
    tp_group_idx * tensor_model_parallel_size
    + cp_dp_combined_idx * attn_tp_size
)

en = st + attn_tp_size

ranks = list(range(st, en))
~~~

固定源码：

[Attention-TP group construction](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2742-L2774)

因此 Rank0、Rank1 属于同一个 Attention-DP 副本内由两个 Rank 组成的 TP 组；共同执行同一条请求时，它们必须对该请求的候选与接受结果一致。Rank2、Rank3 属于另一个副本，可能正在服务另一批请求，无需满足：

~~~text
Rank0 bonus token
==
Rank2 bonus token
~~~

### 为什么 DSpark / EAGLE 都会选择 `attn_tp_group`

DSpark 初始化：

~~~python
self._tp_sync = SpecTpSync(
    parallel.attn_tp_group
    if parallel.enable_dp_attention
    else parallel.tp_group
)
~~~

固定源码：

[`DSparkWorkerV2.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L120-L230)

EAGLE Sampling Accept 也使用：

~~~python
tp_group = (
    get_parallel().attn_tp_group
    if is_dp_attention_enabled()
    else get_parallel().tp_group
)
~~~

这两处选择都在缩小同步范围：同步组覆盖共同执行同一份投机请求状态的 Rank，DP Attention 下使用相应的 `attn_tp_group`。上面的 TP16/DP8 只是组划分示例，不表示它已经满足 DeepSeek-V4 MoE Draft 的启动条件；固定版本的 DSpark Worker 仍要求该场景 `attn_tp_size=1`。

## 三、DSpark 怎样同步决策和布局参数

DSpark 把候选选择、Verify 计划和接受结果等同步位置列成明确的 Site。Site 是一处同步调用的标识，环境变量可以按标识决定是否启用。这里的 convergence 就是让同组 Rank 在这些位置采用同一个决定。

固定源码定义：

~~~python
class SpecTpSync:
    """Broadcasts a speculative decision from rank 0 to its TP group."""
~~~

`sync()` 只有一件事：

~~~python
def sync(self, site, values):
    if site in self._sites:
        self._tp_group.broadcast(
            values,
            src=0,
        )
    return values
~~~

固定源码：

[`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)

默认：`SGLANG_SPEC_TP_SYNC=all`

`src=0` 是组内编号。比如组为 `[2, 3]` 时，广播来源是全局 Rank2，不是全局 Rank0。同步实现只在 Site 开启且组内不止一个 Rank 时广播，单 Rank 组直接沿用本地值。

固定入口：

[`SGLANG_SPEC_TP_SYNC`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1362-L1375)

### Draft Token 是显式同步点

Greedy：

~~~python
tp_sync.sync(
    DSPARK_DRAFT_GREEDY,
    argmax_token,
)
~~~

Sampling：

~~~python
tp_sync.sync(
    DSPARK_DRAFT_SAMPLE,
    sampled_token,
)
~~~

Multinomial 也有：`DSPARK_DRAFT_MULTINOMIAL`

固定源码：

[`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200)

如果第一枚 candidate 就不同：

~~~text
Rank0 → A
Rank1 → B
~~~

这时后续 Draft 链已经从不同 Token 出发，Verify 看到的候选也不同。仅在最后统一接受长度，无法把前面的候选恢复成同一批；候选选择本身就需要一个明确的同步边界。

### `verify_lens` 也是 control decision，但它直接决定后续 Layout

Planner 会计算每个请求本轮要验证多少 Token：

~~~python
verify_lens = ScheduleVerifyLensTopk.execute(...)

self._tp_sync.sync(
    DSPARK_PLAN,
    verify_lens,
)
~~~

固定源码：

[`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py#L575-L603)

`verify_lens` 是每条请求本轮要验证的长度，它直接影响后续分配多少行、使用哪些位置。`SpecTpSync.sync()` 只广播显式传入的这个 Tensor，随后仍由各 Rank 根据本地输入构造 positions、缓存位置和模型缓冲区。同步了决定布局的参数，并不等于把所有布局数据也复制了一遍。

例如：

~~~text
verify_lens
      ↓
deterministic local planning
      ↓
verify row layout
positions
KV allocation
~~~

要得到对应的逻辑布局，各 Rank 不仅要拿到相同 `verify_lens`，还要对请求顺序、前缀长度等相关输入有一致理解，并执行对应的确定性规划。广播长度只能对齐其中一个输入，不能修复已经错位的请求行。

### Accept 同步的是最小决定量

Target Verify 后：

~~~text
correct_len
bonus
cap_trim_lens
~~~

会被同步：

~~~python
self._tp_sync.sync(site, correct_len)
self._tp_sync.sync(site, bonus)
self._tp_sync.sync(site, cap_trim_lens)
~~~

然后每张卡本地运行：

~~~text
FinalizeAcceptLens
BuildOutTokens
~~~

派生：

~~~text
commit_lens
new_seq_lens
out_tokens
~~~

固定源码：

[`TargetVerifyExecutor.accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206)

在这些决定与其余输入都对应的前提下，DSpark 可以让各 Rank 在本地派生后续状态：

~~~text
同步最小 canonical decision
        ↓
各 Rank 本地确定性派生
        ↓
得到相同逻辑 sequence state
~~~

### `SpecTpSync` 也不是“纯 decision broadcast 类”

初始化时，它还负责协调：

~~~python
available_memory_gb(...)
~~~

当 `DSPARK_MEM` site 开启时，它会通过传入 group 获取 group-wide 最小可用显存，用来决定 Draft CUDA Graph / folded sampling 等初始化策略。

DSpark 初始化 CUDA Graph 时：

~~~python
available_mem = self._tp_sync.available_memory_gb(
    DSPARK_MEM,
    ...,
    group=self._draft_graph_group,
)
~~~

固定源码：

[`DSparkWorkerV2.init_cuda_graphs()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

这里的目的，是避免同组 Rank 根据各自剩余显存选择不兼容的初始化容量。`SpecTpSync` 除了同步投机决定，还承担 `DSPARK_MEM / DFLASH_MEM` 这类资源协调；它并不负责统一模型 Tensor 的内容或物理布局。

它不会自动同步：

~~~text
hidden_states
Q/K/V shard
MoE router activations
EP dispatch buffers
KV page contents
positions
out_cache_loc
~~~

这些属于其他 subsystem。

## 四、EAGLE NPU Greedy 的调用与一致性条件

EAGLE / NextN 没有统一使用 DSpark 的 `SpecTpSyncSite` 体系，要沿具体分支检查。Target Verify 结束后，先进入下面的接受函数：

~~~python
eagle_sample(...)
~~~

### Greedy 分支如何进入

判断函数：

~~~python
return (
    is_all_greedy
    or is_cpu
    or is_xpu
    or (is_hip and not use_rejection_sampling)
)
~~~

固定源码：

[`_verify_uses_greedy()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L695-L710)

对 NPU，请求满足 `is_all_greedy=True` 时会进入 Greedy 分支，先在本地计算：

~~~python
target_predict = argmax(target_logits)

predict, accept_index, num_correct_drafts = (
    verify_tree_greedy_func(...)
)
~~~

随后源码只有：

~~~python
if _is_hip:
    broadcast(...)
~~~

固定源码：

[`eagle_sample()` Greedy branch](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L810-L844)

把这个条件放回本文固定 commit，可以确认：

~~~text
CUDA Greedy
NPU Greedy
~~~

在 `eagle_sample()` 这一层没有额外的 rank0 Accept broadcast。

这里确认的范围只到 `eagle_sample()` 这一层；不能从“没有这次广播”继续推出整条 NPU 路径没有其他一致性约束。

### 但源码没有解释“NPU 为什么不需要”

ROCm 分支有明确注释：

~~~text
per-rank draft tokens can differ
→ accepted drafts differ
→ committed seq_lens desync
→ next TP collective can deadlock
~~~

因此 HIP 要显式 broadcast。

当前 NPU Greedy 路径在 `eagle_sample()` 内没有额外的 accept-level rank0 override。进入该路径的 Rank 因而要在候选、Target argmax 与树遍历结果上保持一致，由上游输入和执行约束共同保障。这一检查应落在实际配置的逐 Rank 结果上。

### Sampling path 则明确同步

非 Greedy Sampling 路径在 kernel 完成后：

~~~python
tp_group.broadcast(predict, src=0)
tp_group.broadcast(accept_index, src=0)
tp_group.broadcast(num_correct_drafts, src=0)
~~~

源码注释直接解释：

~~~text
different GPUs may produce slightly
different target_probs due to
floating-point non-determinism

→ causing different sampled tokens
→ broadcast rank0 result
~~~

固定源码：

[`eagle_sample()` Sampling branch](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L881-L1007)

这里三项分别决定：

~~~text
predict
→ 本轮真正输出的 token


accept_index
→ Tree 中接受哪条物理 node path


num_correct_drafts
→ sequence length 前进多少
~~~

例如一棵树有两条长度同为三的分支，分别是 A→B→C 和 A→D→E。两边的 `accept_len` 即使都是三，保留的节点也不同，KV compact 时要搬的状态随之不同。因此树形候选除了长度，还要对齐接受的节点路径和实际输出 Token。这是解释语义的假设例子，不是本文观测到的故障。

## 五、决策同步与模型通信怎样配合

前面讨论的投机决策同步，要与模型自身的数据通信配合。DeepSeek-V4 在 DP Attention + MoE 场景中需要转换 Tensor 布局；即使候选和接受长度已经同步，这些数据转换仍要完成。

固定源码 `_run_moe_ffn_dp_sync()` 中，当：

~~~text
attn_tp_size > 1
+
MoE A2A backend enabled
~~~

Attention 后的 hidden rows 会：

~~~python
hidden_states.tensor_split(
    attn_tp_size
)[attn_tp_rank]
~~~

每个 Attention-TP Rank 只把自己的 token-row slice 送进 MoE。

MoE 结束后再：`attn_tp_all_gather`

恢复下一层需要的 row layout。

固定源码：

[`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)

这套通信解决：

~~~text
Token row ownership
Tensor shard ownership
Expert dispatch
Partial-output combine
Attention ↔ MoE layout conversion
~~~

而 `SpecTpSync` 解决：

~~~text
下一枚 Draft Token 是谁
这一轮 Verify 多长
接受多少 Token
新的 Bonus 是什么
~~~

~~~text
SpecTpSync 正确
≠
MoE layout 一定正确


MoE A2A 正确
≠
Accept state 一定一致
~~~

也就是说，要同时检查两个边界：模型通信是否把正确 Token 行交给正确计算，接受逻辑是否让协作 Rank 对后续序列作出对应决定。一边正确，不能代替另一边的检查。

### `verify_lens` 怎样影响实际布局

`SpecTpSync` 可以让两个 Rank 都得到：`verify_lens = 4`

但它不会自动确保：

~~~text
Rank0 position[4]
Rank1 position[4]

Rank0 cache_loc[4]
Rank1 cache_loc[4]
~~~

物理上相同。

不同分片与分配器可以使用各自的物理槽位。需要核对的是：这些槽位是否分别保存同一逻辑请求、同一轮次、同一序列位置所需的局部状态。比如 Rank0 的槽位 120 与 Rank1 的槽位 500 可以是各自正确的位置；反过来，两个 Rank 都使用编号 120，也不能证明它们对应同一请求。这是逻辑布局一致性（Logical Layout Consistency）与物理地址相等（Physical Address Equality）的区别。

所以排查多卡 speculative bug 时，应该同时记录：

~~~text
control state
+
logical ownership
+
physical shard/layout
~~~

而不是只比较 Tensor value。

## 六、一次分歧怎样影响后续执行

假设两个 Attention-TP Rank 在 Accept 时出现第一次 divergence：

~~~text
Rank0:
correct_len = 3
bonus = X


Rank1:
correct_len = 2
bonus = Y
~~~

### Commit Length 不同

~~~text
Rank0 commit_lens = 4
Rank1 commit_lens = 3
~~~

### Sequence Length 不同

~~~text
Rank0 new_seq_len = P + 4
Rank1 new_seq_len = P + 3
~~~

### Position 不同

下一轮：

~~~text
Rank0 starts at position P+4

Rank1 starts at position P+3
~~~

### KV ownership 不同

由于前面的已确认长度不同，ReqToToken 与页映射接下来可能针对不同逻辑位置分配或复用槽位。各自本地地址仍可能有效，但它们保存的已经不是同一段逻辑历史。

### Draft 输入不同

~~~text
Rank0 bonus = X
Rank1 bonus = Y
~~~

下一轮 candidate 自然继续分叉。

### 最后才表现为 Collective 问题

当模型再次进入：

~~~text
Attention TP
MoE EP
DeepEP dispatch/combine
AllGather / ReduceScatter
~~~

两个 Rank 可能已经对：

~~~text
token count
row identity
collective order
~~~

产生不同理解。

于是最终表现才可能是：

~~~text
shape mismatch
wrong result
timeout
collective hang
~~~

这条链说明，最后超时的 Collective 可能只是最先等待不到对端的地方。排查这种假设时，应按请求与轮次往前找第一处决策或逻辑布局分歧；超时本身仍不足以证明是 Accept 出错。

同样的风险也会出现在 Verify 之前，这就是 `DSPARK_PLAN` 要同步计划的原因。例如：

~~~text
Rank0 verify_len = 4
Rank1 verify_len = 5
~~~

本轮 Target Verify 的执行几何形状已经不同，再等 Accept 时修已经太晚。

正确模式是：

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

来到 Ascend 910C，还要把上层调用与底层通信后端分开。相同的 Group 广播接口，可能由 HCCL 或启用后的 ZBAL 承载；MoE 的 Dispatch / Combine 又属于另一层语义。先确认当前配置选中了哪条路径，才能把同步调用对应到正确的后端。

### HCCL：默认 NPU distributed backend

平台默认映射：

~~~python
"npu":
    "hccl"
    if not SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
    else "zbal"
~~~

固定源码：

[`_DEVICE_TO_DISTRIBUTED_BACKEND`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)

所以在未启用 ZBAL 的普通 Ascend 配置里：

~~~text
tp_group
attn_tp_group
pp_group
moe_* process groups
~~~

这类 PyTorch/SGLang device process groups 默认使用 HCCL backend。

因此此时：

~~~python
attn_tp_group.broadcast(...)
~~~

可以理解为一个标准 NPU distributed group broadcast，由 HCCL backend 承载。

但这只适用于：`当前 group backend = HCCL`

下面按启用条件分别说明。

### ZBAL：可以替换 NPU distributed backend，并同时参与内存 / DeepEP adaptor

固定平台映射中，`SGLANG_ZBAL_LOCAL_MEM_SIZE > 0` 时，NPU distributed backend 选择 `zbal`。相应 NPU 初始化代码还会调用 `zbal_init`，建立其内存与通信环境。这里的判断依据是源码中的配置分支，不能把默认 HCCL 路径当成所有 Ascend 部署的实际路径。

固定源码：

- [`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)
- [`init_zbal()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/utils.py#L328-L415)

启用 ZBAL 后，下面这句话就不再适用：

~~~text
SpecTpSync
→ 一定是 HCCL broadcast
~~~

`SpecTpSync` / EAGLE 调用的是 `GroupCoordinator.broadcast()`，底层由当前 NPU distributed backend 承载。默认配置下是 HCCL，启用对应 ZBAL 配置后则走 ZBAL backend path。读调用点时要同时核对 Group 与 backend。

### DeepEP：不是 HCCL/ZBAL 的同层替代品

DeepEP 负责 MoE Token 的 Dispatch / Combine：把 Router 选中的 Token 行送到专家所在 Rank，再收回并合并计算贡献。

`DeepEPDispatcher` 的接口就是：

~~~text
dispatch(hidden_states, topk_output)

combine(expert_outputs)
~~~

固定源码：

[`DeepEPDispatcher`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L972-L1127)

它解决的是：

~~~text
Router 已经决定 Expert
        ↓
Token activation 怎样送到对应 EP Rank
        ↓
Expert 输出怎样 combine 回来
~~~

而接受长度广播，是让同一请求的协作 Rank 采用同一个后续决定。两者发送的数据、参与组与调用时机都可能不同，不能用 DeepEP 的数据交换来替代 Accept 同步。

### ZBAL 与 DeepEP 的组合路径

固定 `deepep.py` 顶部：

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

固定源码：

[`deepep.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L1-L60)

~~~text
ZBAL
和
DeepEP
~~~

这两个名称可以出现在同一条执行路径中：启用 ZBAL 的 NPU 环境里，DeepEP dispatcher 继续提供 Dispatch / Combine 语义，底下使用的 Buffer / Config 则由 ZBAL adaptor 提供。

所以更准确的分层是：

| 层 | 角色 | 典型对象 |
| --- | --- | --- |
| Speculative Control | 决定“下一步算什么” | Draft token、verify_len、accept_index、bonus |
| General Distributed Backend | 承载普通 process-group collective | HCCL，或启用后的 ZBAL backend path |
| MoE A2A Semantics | Router 后的 expert dispatch / combine | DeepEP |
| Model Layout Logic | 决定 Tensor 在哪些 group 间 split/gather | TP / DP Attention / EP runtime |

对应到调用关系，这几层可以这样衔接：

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

图中虚线表示 ZBAL 可以向 DeepEP dispatcher 提供适配器与 Buffer，并不表示每次投机决策广播都要经过 DeepEP。

## 八、按执行边界检查一致性

多卡投机解码需要同时检查三类状态。共同处理同一请求的 Rank 要对决策达成一致，但它们持有的 Tensor 分片和物理 KV 地址可以不同。

| 检查层 | 需要对应的内容 | 典型观察字段 |
| --- | --- | --- |
| 投机决策 | 候选、验证范围、接受路径与新 bonus | Draft token、`verify_lens`、`correct_len`、`accept_index`、`bonus` |
| 逻辑状态 | 请求身份、投机轮次、位置与序列进度 | `req_pool_idx`、round、positions、前后 `seq_len` |
| 通信调用 | 参与者、Group、调用顺序与张量形状 | global/TP/Attention/EP rank、group、shape |

DSpark 在显式 `SpecTpSync` site 上同步相关决定；EAGLE 则要沿实际分支检查：Sampling/UNO 和 HIP Greedy 有显式广播，本文固定版本的 CUDA/NPU Greedy 在 `eagle_sample` 内没有额外的 accept-level broadcast。后者仍需上游输入、算子和分片约束共同保证一致性。

对齐日志时，使用请求与轮次作为共同索引，沿 Draft、Verify、Accept、Commit 比较各 Rank 的逻辑序列。第一次分歧比后续报错更接近问题源头。KV 错位或 collective timeout 可能是状态分叉的后果，也可能来自其他资源或通信故障，需要由这条对照链判断。

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
| DeepEP dispatch/combine | [`DeepEPDispatcher`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L972-L1127) |
