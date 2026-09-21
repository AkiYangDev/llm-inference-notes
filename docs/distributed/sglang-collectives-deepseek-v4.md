# SGLang 里的 AllReduce、AllGather、ReduceScatter、All-to-All 到底在搬什么？用 DeepSeek-V4 画清 Group、Tensor 和通信方向

上一篇 [SGLang 里的 Rank 和 Group 到底是什么？](sglang-rank-group-deepseek-v4.md) 解决了一个基础问题：Rank 表示进程在某套并行坐标系中的位置，Group 表示这次通信谁和谁是一组。

再往下一层，真正决定数据怎么流动的是 Collective。看到 all_reduce、all_gather_into_tensor、reduce_scatter_tensor、buffer.dispatch、buffer.combine 这些调用时，只记住“求和、收集、全互换”还不够。真正要问的是：

~~~text
通信前：
每个 Rank 手里是什么 Tensor？

这些 Tensor 之间是什么关系？
是不同 shard，还是同一个结果的 partial contribution？

通信后：
每个 Rank 应该拿完整结果，
还是只拿自己的 shard，
还是把 Token 的 owner 改成 Expert owner？
~~~

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

本文区分两层：Collective 的高层数据语义，以及 SGLang 当前版本选择的具体实现。高层语义相同，不代表底层一定调用同名的通用 API。

---

## 一、先只看 Tensor：四种 Collective 到底把数据变成什么

假设有四个 Rank：

~~~text
Group = [R0, R1, R2, R3]
~~~

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

它解决的是：每个 Rank 都算出了同一个逻辑 Tensor 的一部分贡献，先 Reduce，再让所有 Rank 都拿到完整结果。

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

这里没有数值求和。A、B、C、D 本来就是不同位置的数据。

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

因此 ReduceScatter 可以理解为 Reduce 加“结果继续保持 shard 状态”。

### All-to-All：改变的是 owner，不是把完整 Tensor 复制给所有人

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

这正适合 MoE：Router 决定 Token 的目标 Expert，而 Expert 分布在不同 Rank 上。

| Collective | 输入关系 | 输出关系 | 是否 Reduce | 所有 Rank 都拿完整结果 |
| --- | --- | --- | --- | --- |
| AllReduce | 同一逻辑 Tensor 的 partial contribution | 所有人拿 reduced 结果 | 是 | 是 |
| AllGather | 完整 Tensor 的不同 shard | 所有人拿全部 shard | 否 | 是 |
| ReduceScatter | 同一逻辑 Tensor 的 partial contribution | 每人拿 reduced 结果的一块 | 是 | 否 |
| All-to-All | 每人持有发往不同目标的数据 | 每人只收发给自己的数据 | 否 | 否 |

---

## 二、TP Linear 为什么天然出现 AllGather 和 AllReduce

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

SGLang 当前 ColumnParallelLinear 在 gather_output=True 时会进入 tensor_model_parallel_all_gather。真实模型中常常故意让结果继续保持 shard，所以不能把“Column Parallel 后一定 AllGather”当成定律。

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

P0/P1 都只是 partial sum，完整结果是 P0+P1，因此天然需要 Reduce。SGLang 当前 RowParallelLinear 在需要归约时会进入 tensor_model_parallel_all_reduce。

这给出一个比死记 API 更重要的判断方法：

> 不同 Rank 手里是不同位置的 shard，考虑 Gather；是同一个逻辑结果的 partial contribution，考虑 Reduce。

---

## 三、严格核 DeepSeek-V4 Attention：attn_tp_all_reduce 的真实边界

把 DeepSeek-V4 简化成“Attention 后做 AllReduce”不够精确。当前源码真正的归约边界在 Attention 输出投影 wo_b 的 RowParallel partial output。

DeepSeek-V4 构造 wo_b 时使用 attn_tp_rank / attn_tp_size 进行权重切分，而 reduce_results 默认只有在：

~~~text
attn_tp_size == global tp_size
并且 attn_tp_size > 1
~~~

时才为 True。

### 情况一：ATTN_TP 等于整个 TP Group

如果：

~~~text
attn_tp_size == tp_size > 1
~~~

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

此时：

~~~text
1 < attn_tp_size < global tp_size
~~~

如果 wo_b 直接做 global TP AllReduce，就会把不同 Attention-DP shard 的结果错误相加。

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

如果 TP=DP，例如 TP8/DP8：

~~~text
attn_tp_size = 1
~~~

每个 Attention-DP shard 只有一个 Rank，没有 partial contribution 需要跨 Rank 求和，因此无需 Attention TP AllReduce。

可以压成：

| Attention layout | wo_b 后怎样归约 |
| --- | --- |
| ATTN_TP=1 | 不需要 TP reduce |
| ATTN_TP=global TP>1 | RowParallelLinear 内部 global TP AllReduce |
| 1<ATTN_TP<global TP | wo_b 不归约，随后显式 ATTN_TP Group AllReduce |

当前 mHC AllReduce fusion 也没有打破这个边界：相关路径明确要求 global TP 等于 ATTN_TP 且等于 4，并要求 wo_b.reduce_results=True，因此它属于 full-TP reduction 的融合实现，不是 DPA subgroup 的第二次归约。

---

## 四、严格核 dp_gather_replicate：三种实现为什么语义等价

DeepSeek-V4 从 DPA Attention 进入没有 A2A backend 的 MoE 路径时，会调用 dp_gather_replicate。

这里的 replicate 很关键：Attention 已经在 ATTN_TP Group 中完成了归约，所以同一 Attention-DP shard 内多个 ATTN_TP Rank 上的 hidden 是复制值，而不是需要继续相加的 partial。

当前 _dp_gather 会根据布局选择：

~~~text
AllGatherV
AllGather
zero-fill + AllReduce
~~~

三条路径物理实现不同，但必须满足同一个不变量：

> global buffer 中每个逻辑 Token 只出现一次，而且 row order 与 DP token ownership 一致。

### AllGatherV：ATTN_TP=1 时最直接

is_dp_gatherv_active 当前要求：

~~~text
SGLANG_DP_USE_GATHERV 开启
ATTN_TP=1
tp_size == attn_dp_size
SUM_LEN
非 elastic WORLD gather
~~~

因为 ATTN_TP=1，每个 DP shard 天然只有一个 contributor，所以可以直接按照每个 DP Rank 的真实 token count 做 variable-length gather：

~~~text
[M0,H] + [M1,H] + ... → [M_global,H]
~~~

### SUM_LEN + AllReduce：互不重叠的非零区间

这条最反直觉。

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

然后对 full TP Group 做 SUM AllReduce：

~~~text
[A B C D E F]
~~~

因为每个逻辑 row 只有一个真实 contributor。

所以这里物理 primitive 是 AllReduce，但高层语义其实是 DP-local rows → global token view。

### MAX_LEN + AllGather：先在 ATTN_TP 内消重，再 full TP Gather

如果 ATTN_TP=1，可以直接 AllGather。

真正有意思的是 ATTN_TP>1。同一 Attention-DP shard 内多个 Rank 都有同一份 replicated local hidden，如果直接 full TP AllGather，就会重复收集。

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

最后 full TP Group AllGather，这样同一个 DP shard 的 local rows 只被拼接一次。

所以三条实现的等价性来自一个共同原则：

| 路径 | 怎样保证每个 Token 只贡献一次 |
| --- | --- |
| AllGatherV | ATTN_TP=1，天然唯一 contributor |
| AllReduce | 只有 ATP0 写真实值，其余副本为 0 |
| AllGather | 先去掉 replicated duplicate，再用 ATTN_TP ReduceScatter 形成唯一 row shards |

普通非 elastic 主线主要围绕 TP Group；Elastic EP 场景下部分 gather 可以切到 expanded WORLD Group，因此不能把 TP Group 写成所有配置下永远唯一的参与者。

---

## 五、严格核 ReduceScatter：什么时候真正替代 post-expert AllReduce

DPA → MoE 不等于“一定 Gather 后 ReduceScatter”。当前 no-A2A 路径至少有三种 combine 方式。

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

Reduce 和 Scatter 是两步。

### SUM_LEN 优化：ReduceScatterV 吸收 post-expert Reduce

当 variable-length gather 路径有效时，DeepSeek-V4 可以设置 _use_reduce_scatterv。核心前提包括：

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

一次完成：

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

它会直接参与 should_skip_post_experts_all_reduce。

所以源码中看起来相似的几个名字其实处在不同控制边界：

~~~text
mlp_reduce_scatter
should_use_dp_reduce_scatterv
should_skip_post_experts_all_reduce
~~~

更稳妥的心智模型是：

~~~text
MoE output
    ↓
当前条件判断

A. post-expert AllReduce + dp_scatter
B. skip post-expert reduce → ReduceScatterV
C. skip post-expert reduce → ReduceScatter
~~~

只有 B/C 才是真正“用 ReduceScatter 替代 post-expert AllReduce”。

---

## 六、严格核 Ascend 910C：DeepEP 和 FuseEP 都做 A2A，但数据面不是一条路径

MoE 的 All-to-All 本质是 owner migration：

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

所以 Ascend DeepEP 的 communicator 是 MOE_EP process group。

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

因此 DeepEP 是 All-to-All 语义，但当前 SGLang 并不是简单调用 torch.distributed.all_to_all。Routing layout、variable token count、async event、dispatch/combine 都交给 DeepEP 或 ZBAL Buffer 数据面。

### Ascend FuseEP：同一个 MOE_EP Group，但绕过普通 Dispatcher

ascend_fuseep 不能理解成“另一种 DeepEPDispatcher”。

create_moe_dispatcher 对它虽然返回 StandardDispatcher，但源码明确写明这个 dispatcher 不会真正执行；FusedMoE.forward 会直接：

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
    ...
)
~~~

注意 Expert 权重也被传入这个调用。

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

也就是说，FuseEP 把 A2A Dispatch、Expert Compute、Combine 的更多边界收进了一个 fused backend call。

至于 fused_deep_moe 内部最终拆成多少 Ascend Kernel、怎样组织 HCCL/RDMA/同步，仅凭 SGLang 这一层源码不能继续下结论，需要进入对应 DeepEP/ZBAL/Ascend runtime 或 profiler。

| 项目 | Ascend DeepEP | Ascend FuseEP |
| --- | --- | --- |
| Process Group | MOE_EP device group | MOE_EP device group |
| Dispatcher abstraction | DeepEPDispatcher | 绕过普通 dispatcher |
| Python 可见 Dispatch | get_dispatch_layout → buffer.dispatch | 不单独暴露 |
| Python 可见 Expert Compute | 独立 runner 阶段 | Expert weights 传给 fused backend |
| Python 可见 Combine | buffer.combine | 不单独暴露 |
| 核心 backend | DeepEP/ZBAL Buffer | fused_deep_moe |
| 高层语义 | Token owner ↔ Expert owner | Token owner ↔ Expert owner |

---

## 七、重新串起 DeepSeek-V4 一层：真正变化的是 Tensor ownership

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

    H -->|"DeepEP"| O["layout → dispatch"]
    O --> P["Local Expert Compute"]
    P --> Q["combine"]

    H -->|"Ascend FuseEP"| R["fused_deep_moe"]
~~~

真正的因果链是：

~~~text
Parallel Layout
    ↓
决定当前 Rank 拥有什么 Tensor
    ↓
下一阶段需要什么 ownership
    ↓
决定 Reduce / Gather / Scatter / A2A 语义
    ↓
SGLang 再根据 padding、shape、backend、硬件选具体实现
~~~

以后看到任何通信代码，可以固定问五个问题：

1. 哪个 Group 在通信？
2. 通信前每个 Rank 的 Tensor 语义和 shape 是什么？
3. 不同 Rank 的数据是 shard，还是 partial contribution？
4. 下一阶段希望每个 Rank 拥有什么？
5. 当前代码是通用 Collective API，还是专用 backend 的 fused data plane？

最后可以把整套方法压成一句话：

> **Group 决定和谁通信，Tensor relationship 决定要不要 Reduce，ownership transition 决定要 Gather、Scatter 还是 A2A，backend 决定这件事最终怎么执行。**

当这四层能同时看清，TP、DPA、MoE、DeepEP、FuseEP 就不再是几个互相独立的模块，而是同一份 hidden states 在不同并行阶段不断改变 owner 的过程。
