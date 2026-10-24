# SGLang Collective：从张量分片到 Attention 与 MoE 通信

两个 Rank 手里各有一个 Tensor，接下来该用 AllReduce 还是 AllGather，取决于它们之间的关系：如果是同一结果的局部贡献，就需要求和；如果是不同区间的分片，就需要拼接；如果 Token 要送到拥有目标 Expert 的 Rank，则需要改变数据归属。

例如两张卡各有 `[M,H]`，shape 相同并不代表可以直接相加：它们可能分别处理两条请求，也可能各自只算出了同一请求的部分结果。下面先用小张量分清四种 Collective，再进入 DeepSeek-V4 的 Attention 与 MoE，沿着数据在通信前后的含义判断该选哪一种操作。

<details markdown="1">
<summary>版本与源码入口</summary>

本文固定到以下 SGLang 源码：

~~~text
sgl-project/sglang
commit: 176dbcb85d3b7737564e4911947035cc0af65f0a
核对日期：2026-09-21
~~~

主要入口：

- [distributed/communication_op.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/communication_op.py)
- [layers/linear.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/linear.py)
- [layers/dp_attention.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/dp_attention.py)
- [models/deepseek_v4.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4.py)
- [layers/moe/utils.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/utils.py)
- [layers/moe/token_dispatcher/deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)
- [hardware_backend/npu/moe/fuseep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/moe/fuseep.py)

本文区分两层：Collective 的高层数据语义，以及 SGLang 当前版本选择的具体实现。同一通信语义可以由通用 API、专用通信库或融合算子完成。

</details>

## 一、四种 Collective 的输入与输出

假设四个 Rank 属于同一个通信组：`Group = [R0, R1, R2, R3]`。Collective 要由相应组内的参与者按约定共同调用，不能只看某一个 Rank 的本地代码。下面的归约示例统一使用 SUM，AllReduce 本身也支持其他归约操作。

### AllReduce：同一个逻辑结果的局部贡献需要求和

如果：

~~~text
R0: [1]
R1: [2]
R2: [3]
R3: [4]
~~~

SUM AllReduce 后：

~~~text
R0: [10]
R1: [10]
R2: [10]
R3: [10]
~~~

四个数是同一个逻辑结果的局部贡献，合起来才是 10。SUM AllReduce 同时完成求和与结果分发，让每个参与者都得到这个和；它不会把四个数保留成四个独立位置。

如果每个 Rank 通信前都是 [M,H]，AllReduce 后 shape 通常仍是 [M,H]。变化的是值从 local partial contribution 变成所有 Rank contribution 之和。

### AllGather：不同 Rank 手里是不同 shard，只需要拼完整

假设：

~~~text
R0: [A]
R1: [B]
R2: [C]
R3: [D]
~~~

AllGather 后：

~~~text
R0: [A B C D]
R1: [A B C D]
R2: [A B C D]
R3: [A B C D]
~~~

A、B、C、D 表示不同区间的数据，AllGather 把这些分片按组内顺序收齐，不改变它们各自的数值。例如 A 与 B 分别是输出特征的前半段和后半段时，需要把它们拼起来，不能逐元素求和。

### ReduceScatter：既要求和，但最终每个 Rank 只需要一块

假设每个 Rank 都有：

~~~text
R0: [A0 B0 C0 D0]
R1: [A1 B1 C1 D1]
R2: [A2 B2 C2 D2]
R3: [A3 B3 C3 D3]
~~~

先逻辑求和：

~~~text
A = A0+A1+A2+A3
B = B0+B1+B2+B3
C = C0+C1+C2+C3
D = D0+D1+D2+D3
~~~

ReduceScatter 直接得到：

~~~text
R0: [A]
R1: [B]
R2: [C]
R3: [D]
~~~

与 AllReduce 相比，ReduceScatter 同样先合并局部贡献，但每个 Rank 最终只保留自己负责的结果区间。下游若继续按分片计算，就不必先把完整结果复制到每个 Rank。

### All-to-All：将数据发送到各自的目标 Rank

如果每个 Rank 手里都有发往不同目标的数据：

~~~text
R0: [A0 B0 C0 D0]
R1: [A1 B1 C1 D1]
R2: [A2 B2 C2 D2]
R3: [A3 B3 C3 D3]
~~~

并规定 A 去 R0、B 去 R1、C 去 R2、D 去 R3，那么交换后：

~~~text
R0: [A0 A1 A2 A3]
R1: [B0 B1 B2 B3]
R2: [C0 C1 C2 C3]
R3: [D0 D1 D2 D3]
~~~

这里变化的是数据的目的地，各个 A/B/C/D 分块没有被相加。MoE 的 Router 选中专家后，就需要把 Token 行送到专家所在 Rank。上例用等长块说明这种交换关系，真实 MoE 各目的地的 Token 数可能不同，还需要计数和重排信息。

| Collective | 输入关系 | 输出关系 | 是否 Reduce | 所有 Rank 都拿完整结果 |
| --- | --- | --- | --- | --- |
| AllReduce | 同一逻辑 Tensor 的 partial contribution | 所有人拿 reduced 结果 | 是 | 是 |
| AllGather | 完整 Tensor 的不同 shard | 所有人拿全部 shard | 否 | 是 |
| ReduceScatter | 同一逻辑 Tensor 的 partial contribution | 每人拿 reduced 结果的一块 | 是 | 否 |
| All-to-All | 每人持有发往不同目标的数据 | 每人只收发给自己的数据 | 否 | 否 |

## 二、TP Linear 需要什么通信

先看 Y = XW。

假设：

~~~text
X.shape = [M,8]
W.shape = [8,8]
TP = 2
~~~

Column Parallel 沿输出维切 W：

~~~text
W0 [8,4]
W1 [8,4]
~~~

两个 Rank 分别得到：

~~~text
R0: Y0 = XW0 → [M,4]
R1: Y1 = XW1 → [M,4]
~~~

完整结果是 concat(Y0,Y1)，所以如果下游要求每个 Rank 都拿到完整 [M,8]，就需要 AllGather。

SGLang 当前 ColumnParallelLinear 在 gather_output=True 时会进入 tensor_model_parallel_all_gather。下游继续消费 shard 时，结果就保持分片；下游需要完整输出时，再启用 gather。

Row Parallel 则相反。把输入维切开：

~~~text
X = [X0 | X1]
Y = X0W0 + X1W1
~~~

两个 Rank 分别得到：

~~~text
R0: P0 = X0W0 → [M,H]
R1: P1 = X1W1 → [M,H]
~~~

P0/P1 的 shape 虽然相同，值却各自只包含一部分输入特征的贡献。完整结果是 P0+P1，因此需要归约。SGLang 当前 RowParallelLinear 在需要归约时进入 tensor_model_parallel_all_reduce；如果由后续融合或其他分支完成归约，这一步也可能延后。读到 `reduce_results=False` 时，要继续找谁接手了这份局部结果。

所以判断通信时，先把每个 Rank 的输出写成公式：结果是 `[Y0 | Y1]`，说明要拼不同位置；结果是 `P0 + P1`，说明要合并局部贡献。再看下游是否需要完整结果，决定归约后保留整份还是继续分片。

## 三、Attention 输出在哪个 Group 内归约

把这个判断用到 DeepSeek-V4：Attention 的输出投影 `wo_b` 产生 RowParallel 局部贡献，后面要把它们归约到完整输出。除了确认“要相加”，还要确认“与谁相加”。下面按普通、非弹性布局说明主线，后面的 DPA→MoE 部分以未进入 Prefill CP 专用分支为前提。

DeepSeek-V4 构造 wo_b 时使用 attn_tp_rank / attn_tp_size 进行权重切分，而 reduce_results 默认只有在：

~~~text
attn_tp_size == global tp_size
并且 attn_tp_size > 1
~~~

时才为 True。

### 情况一：ATTN_TP 等于整个 TP Group

如果：`attn_tp_size == tp_size > 1`

wo_b 是 RowParallelLinear，局部 GEMM 得到的是同一个 Attention output 的 partial contribution。此时 RowParallelLinear 内部直接使用 global TP Group AllReduce。

~~~text
wo_b rank 0 partial ─┐
wo_b rank 1 partial ─┤
...                   ├─ TP AllReduce → 完整 Attention output
wo_b rank N partial ─┘
~~~

### 情况二：DPA 把 TP world 切成多个 Attention shard

例如：

~~~text
TP = 16
DP = 8
ATTN_TP = 2
~~~

此时：`1 < attn_tp_size < global tp_size`

这时八个 Attention-DP 分组可能正在处理不同请求，每组内部只有两个 Rank 共同计算相应请求。若 `wo_b` 直接做 global TP AllReduce，就会把不同请求组的结果也逐元素相加。正确归约范围应限定在各自的 Attention TP 子组。

所以当前代码让 wo_b.reduce_results=False，只保留当前 ATTN_TP rank 的 partial output。随后才显式：

~~~python
attn_tp_all_reduce(o)
~~~

而 attn_tp_all_reduce 最终走的是 attn_tp_group，不是 global tp_group。

因此更准确的路径是：

~~~text
Attention kernel
    ↓
wo_a
    ↓
wo_b local RowParallel GEMM
    ↓
partial [M_local,H]
    ↓
ATTN_TP Group AllReduce
    ↓
完整 local Attention output
~~~

### 情况三：ATTN_TP=1

如果 TP=DP，例如 TP8/DP8：`attn_tp_size = 1`

每个 Attention-DP shard 只有一个 Rank，没有 partial contribution 需要跨 Rank 求和，因此无需 Attention TP AllReduce。

可以压成：

| Attention layout | wo_b 后怎样归约 |
| --- | --- |
| ATTN_TP=1 | 不需要 TP reduce |
| ATTN_TP=global TP>1 | RowParallelLinear 内部 global TP AllReduce |
| 1<ATTN_TP<global TP | wo_b 不归约，随后显式 ATTN_TP Group AllReduce |

当前 mHC AllReduce fusion 也没有打破这个边界：相关路径明确要求 global TP 等于 ATTN_TP 且等于 4，并要求 wo_b.reduce_results=True，因此它属于 full-TP reduction 的融合实现，不是 DPA subgroup 的第二次归约。

## 四、DP Attention 怎样恢复完整 Token 行

DeepSeek-V4 从 DPA Attention 进入没有 A2A backend 的 MoE 路径时，会调用 dp_gather_replicate。

这里的 replicate 说明输入已经是副本：Attention 在 ATTN_TP 组内归约完成后，同一 Attention-DP 分组的多个 Rank 拿到相同的完整本地行。例如某个元素已经是 7，两个副本再求和就会变成 14。因此收集不同 DP 分组的 Token 行之前，还要避免把同组副本重复计入。

当前 _dp_gather 会根据布局选择：

~~~text
AllGatherV
AllGather
zero-fill + AllReduce
~~~

这三条路径的底层操作不同，目标却相同：让 global buffer 中每个逻辑 Token 只出现一次，行顺序也与原来的请求归属对应。下面会用到两种填充方式：SUM_LEN 按各 DP 分组的行数总和组织全局缓冲区；MAX_LEN 先把各分组填充到相同长度，再收集等长分块。它们都会影响通信接口与结果切分方式。

### AllGatherV：ATTN_TP=1 时最直接

is_dp_gatherv_active 当前要求：

~~~text
SGLANG_DP_USE_GATHERV 开启
ATTN_TP=1
tp_size == attn_dp_size
SUM_LEN
非 elastic WORLD gather
~~~

因为 ATTN_TP=1，每个 DP shard 天然只有一个 contributor，所以可以直接按照每个 DP Rank 的真实 token count 做 variable-length gather：`[M0,H] + [M1,H] + ... → [M_global,H]`

### SUM_LEN + AllReduce：互不重叠的非零区间

这条路径虽然调用 AllReduce，高层目的仍是收集不同 DP 分组的行。之所以可以用求和实现拼接，在于各 Rank 的非零区间被安排成互不重叠。

每个 Rank 先把 global buffer 全部置零，然后只把自己 DP shard 的 local hidden 写到对应 global row 区间。

对于 dp_gather_replicate，如果同一 DP shard 内还有多个 ATTN_TP Rank，只有 attn_tp_rank=0 的 Rank 写真实值，其余 Rank 保持 0。

例如：

~~~text
DP0/ATP0: [A B 0 0 0 0]
DP0/ATP1: [0 0 0 0 0 0]

DP1/ATP0: [0 0 C D 0 0]
DP1/ATP1: [0 0 0 0 0 0]

DP2/ATP0: [0 0 0 0 E F]
DP2/ATP1: [0 0 0 0 0 0]
~~~

然后对 full TP Group 做 SUM AllReduce：`[A B C D E F]`

例如 A 所在的位置只有 DP0/ATP0 写真实值，其余 Rank 都写 0，求和后保留的仍是 A；B 到 F 同理。这里没有把不同 Token 相加，只是利用零缓冲区和唯一贡献者，把 DP 本地行放到全局视图的正确位置。

### MAX_LEN + AllGather：先在 ATTN_TP 内消重，再 full TP Gather

如果 ATTN_TP=1，可以直接 AllGather。

ATTN_TP>1 时还多一步去重。同一 Attention-DP 分组内有多份相同行，直接 full TP AllGather 会把每份副本都收进去，后面的行数与请求映射就会重复。

当前代码对 dp_gather_replicate 先把非 attn_tp_rank=0 的 local tensor 置零，再把 local tensor 按 Token 维切成 ATTN_TP 份，然后在 attn_tp_group 内做 ReduceScatter。

例如：

~~~text
ATP0: [A B C D]
ATP1: [A B C D]

先消重：
ATP0: [A B C D]
ATP1: [0 0 0 0]

ATTN_TP ReduceScatter 后：
ATP0: [A B]
ATP1: [C D]
~~~

这个 ReduceScatter 中，先前置零的副本不会增加数值；它的输出把原来的四行分成 `[A B]` 和 `[C D]` 两个互不重复的片段。再做 full TP AllGather，就能把不同 DP 分组的这些片段各收集一次。

所以三条实现的等价性来自一个共同原则：

| 路径 | 怎样保证每个 Token 只贡献一次 |
| --- | --- |
| AllGatherV | ATTN_TP=1，天然唯一 contributor |
| AllReduce | 只有 ATP0 写真实值，其余副本为 0 |
| AllGather | 先去掉 replicated duplicate，再用 ATTN_TP ReduceScatter 形成唯一 row shards |

普通非 elastic 主线主要围绕 TP Group；Elastic EP 场景下，部分 gather 使用 expanded WORLD Group。

## 五、ReduceScatter 怎样合并归约与分发

当前 no-A2A 路径有三种 combine 方式：

### Baseline：post-expert AllReduce + dp_scatter

基线是：

~~~text
DP gather
    ↓
MoE
    ↓
post_experts_all_reduce
    ↓
reduced global output
    ↓
dp_scatter
    ↓
local token rows
~~~

这条基线路径先让参与归约的 Rank 拿到完整全局输出，再从中取回本地请求对应的行。如果下游只需要本地行，就有机会把求和与分发合成一步。

### SUM_LEN 优化：ReduceScatterV 吸收 post-expert Reduce

ReduceScatterV 中的 V 表示分块长度可以变化，各 DP 分组可以按自己的真实 Token 数接收结果。DeepSeek-V4 在对应路径有效时可以设置 `_use_reduce_scatterv`，核心前提包括：

~~~text
DPA local→global MoE gather path
A2A backend = none
is_dp_gatherv_active() = true
dp_padding_mode = SUM_LEN
~~~

而 is_dp_gatherv_active 又要求 ATTN_TP=1、tp_size=attn_dp_size，并启用对应环境开关。

此时 decoder 发布 mlp_reduce_scatter=True。MoE 内部读取这个 flag 后会跳过原本的 post-expert AllReduce，最后由：

~~~python
tp_group.reduce_scatterv(...)
~~~

一次完成下面两件事，既合并专家的局部贡献，又只返回本地所需行：

~~~text
partial expert output SUM
+
按真实 per-DP token count 返回 local rows
~~~

### MAX_LEN 优化：等长 ReduceScatter

另一条路径由 SGLANG_DP_USE_REDUCE_SCATTER 控制，并要求：

~~~text
no-A2A DPA→MoE gather path
不是 reduce_scatterv
dp_padding_mode = MAX_LEN
tp_size == attn_dp_size
~~~

此时同样设置 mlp_reduce_scatter=True，让 MoE 跳过内部 post-expert Reduce，然后由 dp_reduce_scatter_tensor 完成 SUM + 等长 shard 返回。

### should_use_dp_reduce_scatterv 是另一条独立 skip 条件

当前 moe/utils.py 还有 should_use_dp_reduce_scatterv，主要条件包括：

~~~text
A2A backend = none
DPA 开启
attn_dp_size > 1
tp_size == attn_dp_size
moe_ep_size == attn_dp_size
并排除特定 FP4 allgather 路径
~~~

这是标准 Dispatcher 的 EP combine 路径上的独立判断，会直接参与 `should_skip_post_experts_all_reduce`。它与 Decoder 发布的 `mlp_reduce_scatter` 都可能使内部 AllReduce 被跳过，但触发位置和后续接手的组件不同，不能只凭名字相近就当作同一个开关。

所以源码中看起来相似的几个名字其实处在不同控制边界：

~~~text
mlp_reduce_scatter
should_use_dp_reduce_scatterv
should_skip_post_experts_all_reduce
~~~

沿输出继续追踪，可以把本文讨论的几种归约安排分开：

~~~text
MoE output
    ↓
当前条件判断

A. post-expert AllReduce + dp_scatter
B. skip post-expert reduce → ReduceScatterV
C. skip post-expert reduce → ReduceScatter
~~~

B/C 需要跳过原来的 post-expert AllReduce，由后续 ReduceScatter 承担求和；若两次都执行，就会把已经合并的副本再次求和。检查优化是否正确，既要找到新的归约位置，也要确认旧的归约确实被相应条件跳过。

## 六、Ascend DeepEP 与 FuseEP 的执行路径

MoE 的 All-to-All 把 Token 行从请求所在 Rank 送到专家所在 Rank，计算结束后再返回。这里的 owner 描述数据在当前阶段由谁持有和处理：

~~~text
Request/DP owner
    ↓ Router
Expert owner
    ↓ Expert Compute
Request/DP owner
~~~

当前 Ascend 路径至少要区分 DeepEP 和 ascend_fuseep。

### Ascend DeepEP：MOE_EP Group + Dispatcher 的 Dispatch/Combine

SGLang 创建 DeepEP dispatcher 时，在 NPU 分支选择：

~~~python
group = get_parallel().moe_ep_group.device_group
~~~

因此这条 Ascend DeepEP 路径把 MOE_EP device process group 交给 Buffer 构造通信资源。具体的数据传输仍由所选 Buffer 实现组织，不能仅凭组名推断底层只有某一种通信原语。

随后进入 DeepEPDispatcher / DeepEPBuffer。deepep.py 在 NPU 上还有两种 Buffer 来源：

~~~text
SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
→ ZBAL DeepEP adaptor / buffer

否则
→ deep_ep.Buffer
~~~

Normal Dispatch 主线可以概括为：

~~~text
Router TopK
    ↓
buffer.get_dispatch_layout(...)
    ↓
num_tokens_per_rank
num_tokens_per_rdma_rank
num_tokens_per_expert
is_token_in_rank
    ↓
buffer.dispatch(...)
    ↓
recv_x / recv_topk_ids / recv_topk_weights
    ↓
Local Expert Compute
    ↓
buffer.combine(...)
    ↓
恢复 source Token owner
~~~

SGLang 将 Routing Layout、可变 Token 数、异步事件和 Dispatch/Combine 交给 DeepEP 或 ZBAL Buffer，由它们组织这套 All-to-All 数据交换。

### Ascend FuseEP：同一个 MOE_EP Group，但绕过普通 Dispatcher

`ascend_fuseep` 从模型层直接进入融合执行路径。

create_moe_dispatcher 对它虽然返回 StandardDispatcher，但源码明确写明这个 dispatcher 不会执行；FusedMoE.forward 会直接：

~~~python
forward_fuseep(...)
~~~

进入 hardware_backend/npu/moe/fuseep.py。

FuseEP 同样通过：

~~~python
get_parallel().moe_ep_group.device_group
~~~

取得通信 Group，并创建低时延 DeepEPBuffer。

但后面不是：

~~~text
dispatch
→ expert runner
→ combine
~~~

三个独立 Python 边界，而是直接：

~~~python
buf.fused_deep_moe(
    hidden_states,
    topk_idx=...,
    topk_weights=...,
    gmm1_permuted_weight=...,
    gmm2_weight=...,
    # 其余参数省略
)
~~~

这个调用同时接收输入行、路由结果与专家权重。与普通 Dispatcher 路径相比，Python 调用点已经不再逐个暴露 Dispatch、Expert GEMM 和 Combine；这正是判断融合边界的直接线索。

所以从 SGLang Python 层能够确认的是：

~~~text
Router / TopK
    ↓
forward_fuseep
    ↓
fused_deep_moe(hidden, routing, expert weights, ...)
    ↓
final hidden
~~~

这个融合调用包含计算与数据交换，不能把整段耗时都记作纯通信，也不能仅凭一个 Python 调用就认定底层只启动一个 Kernel。

`fused_deep_moe` 内部的 Kernel 划分、传输与同步由对应 Runtime 组织。继续定位这段耗时，可以沿 DeepEP/ZBAL/Ascend Runtime 展开，并与 Profiler 事件对齐。

| 项目 | Ascend DeepEP | Ascend FuseEP |
| --- | --- | --- |
| Process Group | MOE_EP device group | MOE_EP device group |
| Dispatcher abstraction | DeepEPDispatcher | 绕过普通 dispatcher |
| Python 可见 Dispatch | get_dispatch_layout → buffer.dispatch | 不单独暴露 |
| Python 可见 Expert Compute | 独立 runner 阶段 | Expert weights 传给 fused backend |
| Python 可见 Combine | buffer.combine | 不单独暴露 |
| 核心 backend | DeepEP/ZBAL Buffer | fused_deep_moe |
| 高层语义 | Token owner ↔ Expert owner | Token owner ↔ Expert owner |

## 七、沿一层模型追踪数据归属

把上一篇 Rank/Group 和这一篇 Collective 连起来，可以得到一条稳定的源码阅读路径：

~~~mermaid
flowchart TD
    A["DPA local hidden [M_local,H]"]

    A --> B["Attention output projection wo_b"]
    B --> C{"ATTN_TP size"}

    C -->|"=1"| D["无需 TP Reduce"]
    C -->|"=global TP"| E["RowParallel 内部 TP AllReduce"]
    C -->|"1 < ATTN_TP < global TP"| F["显式 ATTN_TP Group AllReduce"]

    D --> G["完整 local Attention output"]
    E --> G
    F --> G

    G --> H{"MoE backend"}

    H -->|"no A2A"| I["DP Gather / GatherV / zero-fill AllReduce"]
    I --> J["global MoE view"]
    J --> K["Expert Compute"]
    K --> L{"Combine strategy"}
    L --> M["post-expert AllReduce + dp_scatter"]
    L --> N["ReduceScatter / ReduceScatterV"]

    H -->|"DeepEP"| O["按 Attention TP 拆分 Token 行"]
    O --> O2["DeepEP Dispatch"]
    O2 --> P["Local Expert Compute"]
    P --> Q["DeepEP Combine"]
    Q --> Q2["按需 AllGather 恢复 Attention 行"]

    H -->|"Ascend FuseEP"| R["fused_deep_moe"]
~~~

DeepEP 分支中的拆分和恢复适用于 `attn_tp_size>1` 的对应 bridge；宽度为 1 时不需要这组 Attention TP 行转换。图中的 no-A2A 与 FuseEP 是不同后端路径，应按实际配置选择。

读通信代码时，先写下通信前各 Rank 持有的 Tensor：它是不同位置的分片，还是同一结果的局部贡献？再确定下游需要完整结果还是分片结果，最后确认所用 Group 和 backend。这四项信息比单独看到 `all_reduce` 或 `dispatch` 的函数名更能说明数据在做什么。
