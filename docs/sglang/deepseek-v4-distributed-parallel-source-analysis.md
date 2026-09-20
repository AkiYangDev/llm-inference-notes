# DeepSeek-V4 分布式推理源码解析：TP、EP、DP 三种并行如何共同驱动一次前向计算

前几篇文章已经把一条 DeepSeek-V4 请求从 Scheduler、ModelRunner 一直追到 Attention、KV Cache 和 MoE。单卡视角下，一个 Decoder Layer 看起来仍然很简单：

~~~text
Attention
    ↓
MoE
    ↓
Next Layer
~~~

但一旦模型运行在多张 NPU 上，同一层马上变成一个数据布局不断切换的问题：谁拥有 Token，谁拥有 Tensor shard，谁拥有 Expert，什么时候 reduction，什么时候 token exchange，MoE 结束后结果又应该回到谁手里。

很多文章会把问题概括成：

~~~text
Attention 用 TP
MoE 用 EP
请求用 DP
~~~

方向没错，但当前 SGLang 更值得理解的是：

> **同一个 `tp_size` world，在 Attention 阶段按照 `attn_dp × attn_cp × attn_tp` 解释，在 MoE 阶段又按照 `moe_dp × moe_ep × moe_tp` 重新解释。**

所以这篇文章不把 TP、DP、EP 当成三个孤立名词，而是沿一次真实 Decoder Layer forward，追踪 hidden states 怎样从 Attention 坐标系进入 MoE 坐标系，再恢复回来。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线聚焦 DeepSeek-V4 target model 的 TP、DP Attention 与 MoE EP 组合。本文会单独指出 DSpark speculative path 的额外限制，避免把 target model 能表达的拓扑误写成 DSpark 已经支持的拓扑。Ascend 通信以当前 torch.distributed / HCCL 路径为基础说明，同时标注 ZBAL / FuseEP 等专用 backend 的例外。

## 一、先分清 `dp_size` 和 `attn_dp_size`：配置值不等于有效 Attention DP 宽度

当前 SGLang 的配置里有 `tp_size`、`dp_size`、`ep_size`，但真正进入模型执行后，还会派生出 `attn_dp_size`、`attn_tp_size`、`moe_dp_size`、`moe_ep_size`、`moe_tp_size`。

最关键的计算在 [`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281)：

~~~python
attn_dp_size = dp_size if enable_dp_attention else 1
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
~~~

因此必须先记住：

> **`dp_size=16` 本身并不意味着 `attn_dp_size=16`。只有 `enable_dp_attention=True` 时，`dp_size` 才成为 Attention 的有效 DP 宽度。**

例如：

~~~text
tp_size = 32
dp_size = 16
enable_dp_attention = False
attn_cp_size = 1

=> attn_dp_size = 1
=> attn_tp_size = 32
~~~

Attention 仍然是 32-way TP。

只有：

~~~text
tp_size = 32
dp_size = 16
enable_dp_attention = True
attn_cp_size = 1

=> attn_dp_size = 16
=> attn_tp_size = 2
~~~

后文 TP32 / DP16 的讨论全部建立在 **DP Attention 已开启** 这个前提上。

### SGLang 的 world 也不是简单 `TP × DP × EP`

`initialize_model_parallel()` 接受 tensor、attention-DP、attention-CP、expert、MoE-DP 等多个宽度。固定源码：[`initialize_model_parallel()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2515-L2585)。

忽略 PP 的普通主线里，底层 world 首先由 `tp_size` 定义，然后同一个 world 被解释成两套坐标系：

~~~text
Attention:
tp_size = attn_dp_size × attn_cp_size × attn_tp_size

MoE:
tp_size = moe_dp_size × moe_ep_size × moe_tp_size
~~~

MoE 派生公式在 [`derive_parallel_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L336-L377)：

~~~python
moe_tp_size = tp_size // moe_ep_size // moe_dp_size
~~~

因此更准确的总图是：

~~~mermaid
flowchart TD
    W[TP world<br/>tp_size ranks]
    W --> A[Attention coordinate system]
    A --> ADP[attn_dp_size]
    A --> ATP[attn_tp_size]
    A --> ACP[attn_cp_size]
    W --> M[MoE coordinate system]
    M --> MDP[moe_dp_size]
    M --> EP[moe_ep_size]
    M --> MTP[moe_tp_size]
~~~

### Rank 布局也不是猜出来的

源码明确使用 `(dp, cp, tp)` 布局，TP 是最快变化维：

~~~text
tp_rank =
(attn_dp_rank * attn_cp_size + attn_cp_rank)
* attn_tp_size
+ attn_tp_rank
~~~

见 [`derive_attention_ranks()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L262-L281)。

因此 TP32 / DP16 / CP1 / DP-Attention ON 时：

~~~text
DP0  -> ranks 0,1
DP1  -> ranks 2,3
DP2  -> ranks 4,5
...
DP15 -> ranks 30,31
~~~

每个 Attention DP replica 内部恰好是 2-way Attention TP。

## 二、Attention TP 到底切了什么：DeepSeek-V4 Query 侧按 `attn_tp_size` 分片

DeepSeek-V4 的 Query 主干包含 `wq_b`。当前构造：

~~~python
self.wq_b = ColumnParallelLinear(
    self.q_lora_rank,
    self.n_heads * self.head_dim,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
    ...
)
~~~

源码：[`wq_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L862-L870)。注意这里使用的是 `attn_tp_size`，不是外层 `tp_size`。

`ColumnParallelLinear` 沿权重输出维分片：

~~~text
Y = X A
A = [A0, A1, ..., Ap]
~~~

源码：[`ColumnParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L309-L405)。

如果 Query 有 64 heads、`attn_tp_size=4`，概念上每个 Rank 负责约 16 个 Query heads。

但 DeepSeek-V4 的共享 KV / latent 侧不能简单说成“和 Q 一样切”。当前 `wkv` 使用 `ReplicatedLinear`，就在 `wq_b` 前面。因此更准确的描述是：

~~~text
Query-side projection -> attention-TP sharded
shared KV / latent projection -> can remain replicated
~~~

Attention 输出侧的 `wo_b` 则是：

~~~python
self.wo_b = RowParallelLinear(
    self.n_groups * self.o_lora_rank,
    self.hidden_size,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
    ...
)
~~~

源码：[`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L904-L913)。

`RowParallelLinear` 先让各 Rank 计算 partial output，需要合并时调用 `tensor_model_parallel_all_reduce(output_parallel)`。固定源码：[`RowParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L1422-L1696)。

所以 Attention TP 的典型逻辑是：

~~~mermaid
flowchart LR
    X[hidden states] --> C[Column-parallel Query projection]
    C --> T0[attn TP rank 0]
    C --> T1[attn TP rank 1]
    C --> TN[other TP ranks]
    T0 --> A0[local attention]
    T1 --> A1[local attention]
    TN --> AN[local attention]
    A0 --> R[Row-parallel output]
    A1 --> R
    AN --> R
    R --> RED[TP reduction]
    RED --> Y[replicated post-attention hidden rows]
~~~

最后这一步“replicated post-attention hidden rows”正是下一节 Attention → MoE 数据布局转换的前提。

## 三、MoE 换成另一套 ownership：Expert Parallel group 与 Attention TP group 不是一回事

MoE 关心的不是 Query head shard，而是 Expert 权重归属。

SGLang 会单独构造 `_MOE_DP`、`_MOE_EP`、`_MOE_TP`。固定源码：[`MoE groups`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2770-L2895)。

例如：

~~~text
tp_size = 8
moe_dp_size = 2
moe_ep_size = 4

=> moe_tp_size = 8 / 2 / 4 = 1
~~~

同一组 8 Rank 在 MoE 阶段变成 2-way MoE DP × 4-way EP × 1-way MoE TP。

### Ascend 上还有一个很具体的 group 细节

源码直接写着：

~~~python
# NPU requires a standalone group for MOE expert parallelism
if moe_ep_size == tensor_model_parallel_size and not _is_npu:
    _MOE_EP = _TP
else:
    _MOE_EP = init_model_parallel_group(...)
~~~

固定入口：[`_MOE_EP`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2835-L2865)。

因此在 NPU 上，即使：

~~~text
moe_ep_size == tp_size
~~~

MoE EP 仍然使用 standalone process group。

> **same rank membership 不等于 same communicator object。**

这个细节对排查跨机 collective mismatch、通信超时和 communicator 初始化非常重要。

## 四、真正连接 Attention 与 MoE 的，是 `_run_moe_ffn_dp_sync()`

DeepSeek-V4 最值得看的并行桥接函数是：[`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)。

它的职责可以概括成：

> **把 Attention 阶段的数据布局转成 MoE 能消费的数据布局，再把 MoE 输出恢复成下一层 Attention 所需布局。**

当前代码有两条特别关键的分支。

### 路径 A：Attention DP > 1，但没有 A2A MoE backend

条件：

~~~python
_use_tp_moe_gather = (
    not _use_cp
    and get_parallel().attn_dp_size > 1
    and get_moe_a2a_backend().is_none()
)
~~~

此时 Attention Token 分散在多个 DP Rank，而 MoE 没有专用 A2A Expert dispatcher。

源码先：

~~~python
dp_gather_replicate(
    hidden_states,
    local_hidden_states,
    forward_batch,
)
~~~

把 DP-local Token rows gather 到 MoE 可消费的更大全局 buffer。

源码注释明确指出：Attention 已经在 attention TP 内完成 reduction，所以这些 hidden states 是 replicated 的，此处 gather 不能再把它们当 partial tensor 相加。

MoE 完成后，再使用 `reduce_scatterv`、`reduce_scatter` 或 `dp_scatter` 恢复本 Rank 的 Token slice。

逻辑链：

~~~text
Attention DP-local rows
        ↓
DP gather / replicate
        ↓
MoE
        ↓
scatter or reduce-scatter
        ↓
DP-local rows
~~~

### 路径 B：Attention TP > 1，并且使用 A2A MoE backend

条件：

~~~python
_use_tp_attn_a2a_scatter = (
    not _use_cp
    and get_parallel().attn_tp_size > 1
    and not get_moe_a2a_backend().is_none()
)
~~~

这是理解 TP32 / DP16 最关键的路径。

在进入 `self.mlp(...)` **之前**，源码执行：

~~~python
s = get_parallel().attn_tp_size
r = get_parallel().attn_tp_rank

_a2a_scatter_chunks = list(
    hidden_states.tensor_split(s)
)

hidden_states = (
    _a2a_scatter_chunks[r]
    .contiguous()
)
~~~

如果存在 `input_ids`，它们也按照同一个 `attn_tp_size` 做 token-row split。

这里切的不是 hidden dimension，而是 **Token rows**。

原因是 Attention 已经完成 TP reduction，同一个 attn-TP group 内 post-attention hidden rows 是 replicated 的。如果两个 Rank 都拿完整 Token rows 去做 EP dispatch，同一个 Token 会被 dispatch 多遍。

所以：

~~~text
Rank0: [A B C D]
Rank1: [A B C D]

      ↓ tensor_split(2)

Rank0: [A B]
Rank1: [C D]
~~~

然后这两份 Token rows 分别进入 MoE Router / EP dispatch。

### MoE 后的时序同样明确：先 MoE，再 attn-TP AllGather

`self.mlp(...)` 完成以后：

~~~python
gathered = [
    torch.empty_like(t)
    for t in _a2a_scatter_chunks
]

attn_tp_all_gather(
    gathered,
    hidden_states.contiguous(),
)

hidden_states = torch.cat(gathered)
~~~

所以精确时序是：

~~~text
Attention TP reduction 已完成
        ↓
token-row split by attn_tp_rank
        ↓
MoE Router / EP dispatch / expert compute / combine
        ↓
attn_tp_all_gather
        ↓
恢复下一层 Attention 所需的 replicated layout
~~~

~~~mermaid
flowchart TD
    A[Post-attention hidden rows<br/>replicated inside attn TP group]
    A --> S[Token-row split<br/>by attn_tp_rank]
    S --> M[MoE<br/>Router + EP + experts + combine]
    M --> G[attn_tp_all_gather]
    G --> N[Replicated rows<br/>for next layer]
~~~

这张图是理解“TP 与 EP 如何真正接上”的核心。

## 五、把 TP32 / DP16 真正代入源码：target path 能表达，但 DSpark 当前仍显式拒绝

现在设：

~~~text
tp_size = 32
dp_size = 16
enable_dp_attention = True
attn_cp_size = 1
~~~

于是：

~~~text
attn_dp_size = 16
attn_tp_size = 32 / 16 / 1 = 2
~~~

32 Rank 在 Attention 阶段是：

~~~text
16 个 Attention-DP replicas
每个 replica 由 2 个 Attention-TP ranks 组成
~~~

例如 DP replica 5 就是 Rank10 / Rank11。

### 使用当前 A2A MoE backend 时，resolved EP 会被强制成 TP width

固定版本 `_A2A_EP_SPANNING_BACKENDS` 包含 DeepEP、Mooncake、NIXL、Ascend FuseEP、MORI、PPLX、MegaMoE 等。对这些 backend，解析阶段执行：

~~~python
return {"ep_size": view.tp_size}
~~~

源码：[`_a2a_ep_size()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/overrides.py#L1630-L1673)。

因此在 TP32 + A2A backend 下：

~~~text
resolved moe_ep_size = 32
~~~

若 `moe_dp_size=1`：

~~~text
moe_tp_size = 32 / 32 / 1 = 1
~~~

同一组 32 Rank 的两套坐标系于是是：

~~~text
Attention:
16 DP × 2 TP

MoE:
32 EP × 1 MoE-TP
~~~

在 target model 主线里，前面看到的 `_run_moe_ffn_dp_sync()` 已经为这种 `attn_tp_size>1 + A2A` 准备了 token-row split / all-gather layout bridge。

### 但是固定 commit 的 DSpark MoE draft 明确要求 `attn_tp == 1`

同一个 commit 的 `DSparkWorkerV2` 有显式 guard：

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

固定源码：[`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L159-L175)。

而 TP32 / DP16 恰好得到：

~~~text
attn_tp_size = 2
~~~

所以这个 guard 会直接命中。

这把问题边界讲得非常清楚：

~~~text
Target model main forward:
已有 attn-TP → A2A-MoE 的 split/gather bridge

DSpark DeepSeek-V4 MoE draft:
当前仍要求 attn_tp == 1
也就是 DP size == TP size
~~~

因此从“显式拒绝 TP32 / DP16”推进到“请求输出正确”，真正工作绝不是删除 `raise ValueError`。

错误信息已经直接指出已知风险：

~~~text
attn_tp > 1 corrupts the MoE-under-DP all-reduce
~~~

至少需要重新验证：

~~~text
draft hidden-state ownership
draft MoE reduction group
token-row split / gather
draft-target token alignment
DP-local verify layout
graph / eager consistency
跨机 communicator participation
最终 logits / sampled token correctness
~~~

所以这个问题本质上是：

> **主模型已经有某种并行布局转换能力，不代表 speculative draft path 自动继承了同样的 ownership 语义。**

## 六、Ascend 上这些 collective 到底是不是 HCCL：默认是，但不能一刀切

固定源码里的默认设备映射：

~~~python
_DEVICE_TO_DISTRIBUTED_BACKEND = {
    "cuda": "nccl",
    ...
    "npu": "hccl" if not ZBAL_LOCAL_MEM else "zbal",
}
~~~

源码：[`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)。

因此普通 Ascend NPU torch.distributed 路径默认使用 HCCL。

SGLang 还会为 NPU 的默认 / MoE 相关 process group 创建：

~~~python
torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
~~~

并读取：

~~~text
DEEPEP_HCCL_BUFFSIZE
or
HCCL_BUFFSIZE
~~~

设置 HCCL buffer。固定源码：[`get_torch_distributed_pg_options()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L90-L105)。

所以可以说：

> **Ascend 上标准 TP / DP / EP process-group collectives 通常由 HCCL 承担。**

但不能写成：

> Ascend 上所有跨卡通信都一定表现为一个独立 HCCL collective。

原因有三点：

1. 配置 ZBAL local memory 后，默认 distributed backend 可以切成 `zbal`；
2. Ascend FuseEP 会把 dispatch、cross-rank exchange、Expert GEMM、combine 融进 `fused_deep_moe(...)`；
3. 即使底层最终使用 HCCL transport，Python / profiler 也未必出现独立的 `all_reduce`、`all_gather`、`all_to_all` 窗口。

因此本文统一使用两层语言：

~~~text
算法 / runtime 语义:
AllReduce / AllGather / All-to-All-like exchange

Ascend 标准 process-group 实现:
通常落到 HCCL

FuseEP / ZBAL 等专用路径:
可能融合或改写可观察边界
~~~

固定版本 Ascend DeepSeek-V4 教程也实际展示了 `--tp-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend deepep` 的部署路径。[官方示例](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx#L202-L225)

最后，可以把 TP、DP、EP 重新定义成三种 ownership。

### TP：Tensor-shard ownership

回答一个 Tensor / Weight Matrix 的 shard 归哪个 Rank。典型动作包括 Column Parallel、Row Parallel、AllReduce、AllGather、ReduceScatter。

### DP Attention：Token ownership

回答当前哪些请求 / Token rows 由哪个 Attention replica 负责。只有开启 DP Attention 时，`dp_size` 才会成为 `attn_dp_size`。

### EP：Expert ownership

回答 Router 选中的 Expert 权重位于哪个 Rank。典型动作是 expert dispatch、cross-rank token exchange、expert-local compute 和 combine。

于是一个 DeepSeek-V4 Decoder Layer 的分布式 forward 最终是：

~~~mermaid
flowchart TD
    F[ForwardBatch]
    F --> DP[Attention DP<br/>Token ownership]
    DP --> TP[Attention TP<br/>Tensor-shard ownership]
    TP --> ATT[DeepSeek-V4 Attention]
    ATT --> RED[TP reduction]
    RED --> B1[Attention → MoE layout bridge<br/>token-row split if needed]
    B1 --> R[Router / HashTopK]
    R --> EP[Expert Parallel<br/>Expert ownership]
    EP --> E[Expert-local compute]
    E --> C[EP combine]
    C --> B2[MoE → Attention bridge<br/>attn-TP all-gather if needed]
    B2 --> N[Next Decoder Layer]
~~~

真正困难的不是记住几个 collective 名字，而是知道：

> **某个 collective 为什么恰好出现在这个位置，它是在恢复哪一种 ownership，又准备切换到哪一种 ownership。**

TP32 / DP16 值得研究，也正因为它迫使系统同时正确处理：

~~~text
16-way Token ownership
2-way Attention Tensor ownership
32-way Expert ownership
~~~

而 DSpark speculative path 又在这三者之上增加了一套 draft / verify ownership。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| `dp_size → attn_dp_size / attn_tp_size` | [`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281) |
| Attention / MoE 派生宽度 | [`derive_parallel_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L336-L377) |
| Rank 坐标公式 | [`derive_attention_ranks()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L262-L281) |
| 并行组总初始化 | [`initialize_model_parallel()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2515-L2585) |
| Attention TP groups | [`parallel_state.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2690-L2770) |
| MoE DP / EP / TP groups | [`parallel_state.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2770-L2895) |
| DeepSeek-V4 Attention TP 权重 | [`wq_b / wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L854-L913) |
| Column Parallel | [`ColumnParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L309-L405) |
| Row Parallel / TP reduction | [`RowParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L1422-L1696) |
| Attention → MoE layout bridge | [`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915) |
| A2A backend 强制 resolved EP=TP | [`_a2a_ep_size()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/overrides.py#L1630-L1673) |
| DSpark `attn_tp>1` 显式限制 | [`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L159-L175) |
| NPU 默认 HCCL / ZBAL 映射 | [`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93) |
| HCCL process-group options | [`get_torch_distributed_pg_options()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L90-L105) |
| Ascend DeepSeek-V4 DP Attention 示例 | [`deepseek_v4_flash.mdx`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx#L202-L225) |