# DeepSeek-V4 Speculative Decoding 源码解析：多卡场景下 Draft、Verify 与 Accept 如何保持一致

前面的文章已经把 DeepSeek-V4 speculative decoding 拆到了 Draft、MTP/NextN、EAGLE Tree、Target Verify 和 KV Commit。

但真正进入多卡 Serving 后，还会多出一个比算法本身更底层的问题：

> **同一轮 speculative decoding 由多张卡共同执行时，哪些状态必须跨 Rank 保持一致，哪些 Tensor 又应该继续保持分片？**

这两个问题如果混在一起，很容易得到一个错误结论：

~~~text
“多卡一致”
=
所有 Rank 上的所有 Tensor 都一样
~~~

真实情况恰好相反。

Tensor Parallel、Expert Parallel 的意义，本来就是让不同 Rank 持有不同 shard；但 speculative decoding 又在这些分片 Tensor 之上增加了一套离散状态机：

~~~text
Draft token
Verify length
Tree path
Accept index
Accept length
Bonus token
New sequence length
~~~

如果共同执行同一逻辑请求的 Rank 对这些状态产生不同理解，下一轮就不再只是“数值略有误差”，而会变成：

~~~text
Token 不一致
    ↓
seq_len 不一致
    ↓
position / KV ownership 不一致
    ↓
collective shape 或调用次序不一致
    ↓
错误、超时，甚至 collective hang
~~~

本文沿两条 DeepSeek-V4 speculative path 分析这个问题：

~~~text
DSpark
→ SpecTpSync 显式 decision convergence

EAGLE / NextN
→ 按具体 Accept / Sampling 路径处理 Rank convergence
~~~

同时把它与：

~~~text
TP / DP / EP Tensor communication
HCCL / ZBAL distributed backend
DeepEP MoE dispatch / combine
~~~

严格分层。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线讨论 PP1 下 DeepSeek-V4 + SGLang speculative decoding，重点覆盖 DSpark V2、EAGLE/NextN、DP Attention 与 Ascend NPU。PP speculative relay 有额外跨 stage 状态传递，不在本文展开。文中的 TP16/DP8 是用于说明 group boundary 的纯拓扑例子，不代表当前 DeepSeek-V4 Ascend 官方推荐配置。

## 一、真正需要一致的不是所有 Tensor，而是“下一步算什么”

普通 Tensor Parallel 中，不同 Rank 的 Tensor 本来就可以不同。

例如一个 Column Parallel Linear：

~~~text
Rank 0:
X @ W0

Rank 1:
X @ W1
~~~

两个 Rank 持有的是不同权重 shard，产生不同 partial output 完全正常。

MoE 更明显。

假设 Token 被 Router 分给某个 Expert：

~~~text
EP0 → Expert 0
EP1 → Expert 1
EP2 → Expert 2
EP3 → Expert 3
~~~

某一阶段的 expert-local activation 只存在于拥有对应 Expert 的 Rank，也没有任何问题。

所以多卡 speculative correctness 应该拆成两层：

| 层次 | 可以不同吗 | 例子 |
| --- | --- | --- |
| Model Data Plane | 可以，甚至设计上就应该分片 | Q/K/V shard、expert activation、A2A buffer、TP partial output |
| Speculative Control Plane | 同一逻辑请求的协作 Rank 必须达成一致 | Draft token、verify length、accept path、bonus、new seq_len |

假设同一个请求在两个 Attention-TP Rank 上做 Verify：

~~~text
Rank 0:
correct_len = 3

Rank 1:
correct_len = 2
~~~

Rank 0 会认为：

~~~text
commit_lens = 4
~~~

Rank 1 却认为：

~~~text
commit_lens = 3
~~~

下一轮：

~~~text
Rank 0:
seq_len = P + 4

Rank 1:
seq_len = P + 3
~~~

Position、KV slot、Draft boundary 随即开始分叉。

EAGLE 源码里有一段非常直接的注释说明这种风险。ROCm Greedy 路径会在 Accept 后 broadcast `predict / accept_index / num_correct_drafts`，因为不同 Rank 如果接受不同数量的 Draft，就会让 committed `seq_lens` 不同，并可能使下一轮 TP collective 失去一致调用契约。

固定源码：

[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L800-L1029)

因此 speculative distributed consistency 的本质不是：

> “每张卡都算出完全相同的 Tensor。”

而是：

> **共同执行同一个逻辑 Request 的 Rank，必须对每一个会改变未来执行路径的离散 decision 达成一致。**

---

## 二、谁应该和谁一致：TP16 / DP8 下 `attn_tp_group` 到底包含哪些 Rank

“跨 Rank 一致”还必须回答第二个问题：

> **到底是哪几个 Rank？**

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

这不是推测。

`initialize_model_parallel()` 在 `attn_tp_size < tp_size` 时，直接按连续区间创建：

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

因此，在这个例子中：

~~~text
Rank 0 / Rank 1
~~~

属于同一个 Attention-DP replica 内的 2-rank TP group。

它们共同执行同一份 Attention-side logical request state 时，需要对 speculative control decision 保持一致。

而：

~~~text
Rank 2 / Rank 3
~~~

属于另一个 Attention-DP replica。

它可能正在服务另一批 Request，所以没有任何理由要求：

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

所以正确的抽象不是：

~~~text
所有 16 张卡 broadcast 一个 token
~~~

而是：

> **control sync 的 group 应覆盖共同执行同一份 speculative request state 的 Rank；DP Attention 下，这通常收缩到对应的 `attn_tp_group`。**

---

## 三、DSpark `SpecTpSync` 到底同步什么：它同步 decision，也同步“布局决定量”，但不会自动同步物化后的 Layout

DSpark 对 Rank convergence 的处理最显式。

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

默认：

~~~text
SGLANG_SPEC_TP_SYNC=all
~~~

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

Multinomial 也有：

~~~text
DSPARK_DRAFT_MULTINOMIAL
~~~

固定源码：

[`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200)

如果第一枚 candidate 就不同：

~~~text
Rank0 → A
Rank1 → B
~~~

后面的 Draft chain、Verify IDs、positions 和 Accept comparison 已经不再属于同一未来。

所以必须在最早的离散 decision boundary 收敛。

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

这说明“`SpecTpSync` 只同步 decision、不碰 layout”还需要再精确一点。

更准确的说法是：

> **`SpecTpSync.sync()` 只广播显式传进去的 Tensor；其中有些 Tensor（例如 `verify_lens`）本身就是 Layout 的决定量。它不会自动把之后物化出来的 positions、cache locations、hidden rows、MoE buffers 再同步一遍。**

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

只要各 Rank 拿到相同 `verify_lens`，后续本地 deterministic planner 可以重建相同逻辑 layout。

但这和：

~~~text
SpecTpSync 自动复制整个 layout buffer
~~~

完全不是一回事。

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

所以 DSpark 的模式可以总结成：

~~~text
同步最小 canonical decision
        ↓
各 Rank 本地确定性派生
        ↓
得到相同逻辑 sequence state
~~~

### `SpecTpSync` 也不是“纯 decision broadcast 类”

严格看源码，它还有一个特殊的初始化协调职责：

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

所以最严谨的总结是：

> **`SpecTpSync` 的主体职责是 speculative control convergence；此外还承担 `DSPARK_MEM / DFLASH_MEM` 这类跨 Rank 初始化资源协调。它不是模型 Tensor/layout 同步器。**

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

---

## 四、EAGLE NPU Greedy 为什么没有显式 Broadcast：能确认“没有”，不能从源码推导“为什么一定不需要”

EAGLE / NextN 没有 DSpark 那种统一的：

~~~text
SpecTpSyncSite
~~~

体系。

所以必须按具体路径看。

Target Verify 结束后，进入：

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

所以 NPU 上只要请求是：

~~~text
is_all_greedy = True
~~~

也会进入 Greedy branch。

Greedy 先本地计算：

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

因此在本文固定 commit：

~~~text
CUDA Greedy
NPU Greedy
~~~

在 `eagle_sample()` 这一层没有额外的 rank0 Accept broadcast。

这是源码事实。

### 但源码没有解释“NPU 为什么不需要”

ROCm 分支有明确注释：

~~~text
per-rank draft tokens can differ
→ accepted drafts differ
→ committed seq_lens desync
→ next TP collective can deadlock
~~~

因此 HIP 要显式 broadcast。

NPU Greedy 分支没有对应注释。

所以文章不能反向推导：

~~~text
“NPU 一定数值确定”
“NPU 所有 Rank 的 Draft 永远相同”
“Ascend 不需要 speculative sync”
~~~

这些结论源码都没有给出。

最严谨的表述是：

> **当前 NPU Greedy path 在 `eagle_sample()` 中没有 accept-level rank0 override，因此实现契约要求参与该 logical request 的 Rank 在进入/执行 Greedy Verify 时已经能得到一致的 candidate、Target argmax 与 tree traversal 结果。这个事实不等于源码证明了所有 NPU 配置都具有数学上的确定性。**

也就是说：

~~~text
没有显式 broadcast
≠
证明 broadcast 永远多余
~~~

它只说明当前固定版本的实现没有在这里增加第二层 canonicalization。

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

所以 Tree 场景里，仅同步一个：

~~~text
accept_len
~~~

还不够。

不同 Branch 可能有相同长度，但 KV compaction path 完全不同。

---

## 五、为什么 Speculative Control Sync 不能代替 TP / DP / EP Layout Communication

现在可以把最容易混淆的两套通信彻底拆开。

DeepSeek-V4 在 DP Attention + MoE 场景本身就需要复杂的 Tensor layout transition。

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

MoE 结束后再：

~~~text
attn_tp_all_gather
~~~

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

所以：

~~~text
SpecTpSync 正确
≠
MoE layout 一定正确


MoE A2A 正确
≠
Accept state 一定一致
~~~

两套 correctness domain 必须同时成立。

### 一个很容易误解的例子：`verify_lens`

`SpecTpSync` 可以让两个 Rank 都得到：

~~~text
verify_lens = 4
~~~

但它不会自动确保：

~~~text
Rank0 position[4]
Rank1 position[4]

Rank0 cache_loc[4]
Rank1 cache_loc[4]
~~~

物理上相同。

事实上，在不同 shard / allocator 设计下，它们甚至不应该物理相同。

真正要求的是：

> **这些局部 layout 都对应同一个逻辑 Request、同一个 speculative step、同一个已确认序列位置。**

这就是：

~~~text
Logical Layout Consistency
~~~

而不是：

~~~text
Physical Address Equality
~~~

所以排查多卡 speculative bug 时，应该同时记录：

~~~text
control state
+
logical ownership
+
physical shard/layout
~~~

而不是只比较 Tensor value。

---

## 六、Rank Divergence 是怎样从一个 Token 扩散成 Collective Hang 的

假设两个 Attention-TP Rank 在 Accept 时出现第一次 divergence：

~~~text
Rank0:
correct_len = 3
bonus = X


Rank1:
correct_len = 2
bonus = Y
~~~

### 1. Commit Length 不同

~~~text
Rank0 commit_lens = 4
Rank1 commit_lens = 3
~~~

### 2. Sequence Length 不同

~~~text
Rank0 new_seq_len = P + 4
Rank1 new_seq_len = P + 3
~~~

### 3. Position 不同

下一轮：

~~~text
Rank0 starts at position P+4

Rank1 starts at position P+3
~~~

### 4. KV ownership 不同

ReqToToken/page mapping 开始针对不同逻辑 position 分配或复用 slot。

### 5. Draft 输入不同

~~~text
Rank0 bonus = X
Rank1 bonus = Y
~~~

下一轮 candidate 自然继续分叉。

### 6. 最后才表现为 Collective 问题

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

所以真正应该找的是：

> **第一处 control / logical-layout divergence。**

而不是只盯着最后一个超时的通信算子。

DSpark 为什么连：

~~~text
DSPARK_PLAN
~~~

都要同步，也就很好理解。

如果：

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

---

## 七、Ascend 上 HCCL、ZBAL、DeepEP 到底分别属于哪一层

最后把这个问题落到 Ascend。

这里最需要避免一句过度简化的话：

> “Ascend 多卡通信就是 HCCL。”

固定版本已经不是这么简单。

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

但这只适用于：

~~~text
当前 group backend = HCCL
~~~

不能无条件推广到所有 Ascend 配置。

### ZBAL：可以替换 NPU distributed backend，并同时参与内存 / DeepEP adaptor

只要：

~~~text
SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
~~~

固定平台映射直接把：

~~~text
NPU distributed backend
~~~

切成：

~~~text
zbal
~~~

同时 NPU 初始化代码还会调用：

~~~text
zbal_init
~~~

建立对应的 ZBAL memory / communication environment。

固定源码：

- [`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)
- [`init_zbal()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/utils.py#L328-L415)

所以启用 ZBAL 后，文章不能继续说：

~~~text
SpecTpSync
→ 一定是 HCCL broadcast
~~~

更准确的是：

> **`SpecTpSync` / EAGLE 调用的是 `GroupCoordinator.broadcast()`；底层由当前 NPU distributed backend 承载。默认是 HCCL，启用 ZBAL 配置后则进入 ZBAL backend path。**

### DeepEP：不是 HCCL/ZBAL 的同层替代品

DeepEP 是：

~~~text
MoE token dispatch / combine
~~~

语义层。

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

这和：

~~~text
Rank0 把 accept_len broadcast 给 Rank1
~~~

不是一个问题。

### 更重要：ZBAL 和 DeepEP 还可以叠在一起

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

这说明：

~~~text
ZBAL
和
DeepEP
~~~

甚至不是互斥关系。

启用 ZBAL 的 NPU 环境里，DeepEP dispatcher 可以继续保持：

~~~text
DeepEP dispatch/combine semantics
~~~

但 Buffer / Config 由 ZBAL adaptor 提供。

所以更准确的分层是：

| 层 | 角色 | 典型对象 |
| --- | --- | --- |
| Speculative Control | 决定“下一步算什么” | Draft token、verify_len、accept_index、bonus |
| General Distributed Backend | 承载普通 process-group collective | HCCL，或启用后的 ZBAL backend path |
| MoE A2A Semantics | Router 后的 expert dispatch / combine | DeepEP |
| Model Layout Logic | 决定 Tensor 在哪些 group 间 split/gather | TP / DP Attention / EP runtime |

其中：

~~~text
ZBAL + DeepEP
~~~

可以组合。

因此不能画成：

~~~text
HCCL vs ZBAL vs DeepEP
三选一
~~~

而应该理解成：

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

这张图比“Ascend 都走 HCCL”更接近固定源码。

---

## 八、真正应该建立的是一份 Speculative Distributed Contract

把全文压缩以后，多卡 speculative correctness 可以拆成三份 contract。

### 1. Decision Contract

共同执行同一 logical request 的 control group 必须在关键边界一致：

~~~text
Draft token
Verify length / topology
Accept path
Accept length
Bonus token
~~~

DSpark 通过：

~~~text
SpecTpSync sites
~~~

显式收敛。

EAGLE 则按路径处理：

~~~text
Sampling / UNO
→ explicit broadcast

HIP Greedy
→ explicit broadcast

fixed-commit CUDA/NPU Greedy
→ eagle_sample 内无额外 accept-level broadcast
~~~

### 2. Logical Layout Contract

相同 decision 必须映射到相同逻辑状态：

~~~text
request identity
speculative step
logical position
accepted path
sequence length
~~~

但物理：

~~~text
KV address
Tensor shard
expert-local buffer
~~~

可以不同。

### 3. Collective Contract

各 communicator 上必须保证：

~~~text
参与 Rank
group identity
call order
Tensor shape
~~~

匹配。

例如：

~~~text
attn_tp_group
moe_ep_group
tp_group
DeepEP dispatcher group
~~~

都各有自己的 ownership。

最终一轮正确的 distributed speculative decoding 是：

~~~mermaid
flowchart TD
    R[Logical Request State]

    R --> D[Draft Decision]
    D --> S1[Control Convergence]

    S1 --> L1[Materialize Local Layout]
    L1 --> F1[Distributed Draft / Target Compute]

    F1 --> A[Accept Decision]
    A --> S2[Control Convergence]

    S2 --> L2[Commit Logical Sequence]
    L2 --> N[New bonus + new seq_len]
    N --> R2[Next Round]
~~~

所以排查问题时，最值得记录的不是只有通信 timeline，而是：

~~~text
global_rank
tp_rank
attn_dp_rank
attn_tp_rank
moe_ep_rank

request / req_pool_idx
speculative round

draft token / draft block

verify_lens
verify ids / tree topology

correct_len
accept_index
bonus
commit_lens

seq_len before / after

positions
cache locations
~~~

然后在每个阶段问一句：

> **这个 Rank 认为“当前已经由 Target 确认的逻辑序列”是什么？**

第一次出现两个协作 Rank 给出不同答案的位置，往往才是真正的 root-cause boundary。

后面的：

~~~text
KV 错位
DeepEP 异常
HCCL / ZBAL collective timeout
~~~

很多时候只是这次状态分叉继续向后传播的结果。

这也是 DeepSeek-V4 多卡 speculative decoding 比普通 TP 推理更难的根本原因：

~~~text
普通分布式推理
主要维护 Tensor ownership


Speculative distributed inference
还要额外维护 Future ownership
~~~

系统不仅要知道：

~~~text
这个 Tensor 属于哪个 Rank
~~~

还必须让相关 Rank 同时知道：

~~~text
这个 Token 是否已经被 Target 接受

它属于哪个 speculative step

下一轮从哪个 bonus token 开始

当前 logical sequence 到底已经前进了多少
~~~

最终可以把整套机制压成一句话：

> **每一次会改变未来执行路径的 speculative decision，都必须先在正确的 control group 内收敛；之后各 Rank 才能安全地在自己的 Tensor shard、Expert shard 和 KV layout 上继续分布式计算。**

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
