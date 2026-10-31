# SGLang Collective：从张量分片到 Attention 与 MoE 通信

两个 Rank 各持一个 Tensor，应该 AllReduce 还是 AllGather，先得看数据代表什么。同一结果的局部贡献需要求和，不同区间的 shard 需要拼接，送往远端专家的 Token 则需要重新分发。

即使两张卡都有 `[M,H]`，它们也可能属于不同请求，或只是同一输出的不同贡献。下面先用小张量分清四种 Collective，再进入 DeepSeek-V4 的 Attention、MoE，把通信前后的数值含义和数据归属接起来。

<details markdown="1">
<summary>版本与源码入口</summary>

以下阅读固定版本源码：

~~~text
sgl-project/sglang
commit: 176dbcb85d3b7737564e4911947035cc0af65f0a
核对日期：2026-09-21
~~~

相关入口包括：

- [distributed/communication_op.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/communication_op.py)
- [layers/linear.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/linear.py)
- [layers/dp_attention.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/dp_attention.py)
- [models/deepseek_v4.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4.py)
- [layers/moe/utils.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/utils.py)
- [layers/moe/token_dispatcher/deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)
- [hardware_backend/npu/moe/fuseep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/moe/fuseep.py)

需要分别看高层通信语义与具体实现。同一种数据交换，可能由通用 API、专用库或融合算子完成。

</details>

## 一、四种 Collective 的输入与输出

假设通信组为 `Group = [R0, R1, R2, R3]`。Collective 要由组内相应参与者共同按约定调用，仅看单个进程不够。归约示例统一使用 SUM，AllReduce 也可以使用其他归约操作。

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

四个数是同一逻辑结果的局部贡献，求和后为 10。SUM AllReduce 将这个和交给每位参与者，原来的四个位置不会单独保留。

所以输入各为 [M,H] 时，输出通常仍为 [M,H]，但数值已经从本地贡献变成组内贡献之和。

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

A、B、C、D 代表不同区间，AllGather 按组内顺序收齐，不将数值相加。例如输出特征的前后两半，需要拼成完整向量。

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

ReduceScatter 同样合并局部贡献，但每位参与者只拿自己负责的结果区间。若下一步仍按 shard 计算，就不必在各 Rank 保留完整输出。

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

All-to-All 改变数据目的地，不对 A/B/C/D 分块求和。MoE 要把 Token 交给目标专家所在 Rank，就有这种交换关系。例子使用等长块，实际专家接收数量可以不同，还需保留计数和重排信息。

| Collective | 输入关系 | 输出关系 | 是否 Reduce | 所有 Rank 都拿完整结果 |
| --- | --- | --- | --- | --- |
| AllReduce | 同一逻辑 Tensor 的 partial contribution | 所有人拿 reduced 结果 | 是 | 是 |
| AllGather | 完整 Tensor 的不同 shard | 所有人拿全部 shard | 否 | 是 |
| ReduceScatter | 同一逻辑 Tensor 的 partial contribution | 每人拿 reduced 结果的一块 | 是 | 否 |
| All-to-All | 每人持有发往不同目标的数据 | 每人只收发给自己的数据 | 否 | 否 |

## 二、TP Linear 需要什么通信

将这些关系放进矩阵乘 Y = XW。

假设：

~~~text
X.shape = [M,8]
W.shape = [8,8]
TP = 2
~~~

Column Parallel 沿 W 输出维分片：

~~~text
W0 [8,4]
W1 [8,4]
~~~

本地输出分别为：

~~~text
R0: Y0 = XW0 → [M,4]
R1: Y1 = XW1 → [M,4]
~~~

完整输出是 concat(Y0,Y1)。若下游要求各 Rank 都得到 [M,8]，需要收齐两份特征。

固定 `ColumnParallelLinear` 在 `gather_output=True` 时调用 `tensor_model_parallel_all_gather`。下游能消费 shard 时可继续分片，需要完整结果时再 gather。

Row Parallel 沿输入维分开：

~~~text
X = [X0 | X1]
Y = X0W0 + X1W1
~~~

两个 Rank 分别得到：

~~~text
R0: P0 = X0W0 → [M,H]
R1: P1 = X1W1 → [M,H]
~~~

P0、P1 虽同为 [M,H]，却只包含部分输入特征贡献，需按 P0+P1 归约。`RowParallelLinear` 在需要时调用 `tensor_model_parallel_all_reduce`，也可将归约留给后续融合或分支。看到 `reduce_results=False`，还要找谁继续合并。

可以先写出各 Rank 的输出关系：`[Y0 | Y1]` 表示拼不同位置，`P0 + P1` 表示求和；再看下游需要完整还是分片结果。

## 三、Attention 输出在哪个 Group 内归约

DeepSeek-V4 的 `wo_b` 产生 RowParallel 局部贡献，还要确认由哪个 Group 求和。下文采用普通非弹性主线，DPA→MoE 的讨论也不进入 Prefill CP 专用分支。

构造 `wo_b` 时，权重按 `attn_tp_rank`、`attn_tp_size` 切分，而默认 `reduce_results=True` 需满足：

~~~text
attn_tp_size == global tp_size
并且 attn_tp_size > 1
~~~



### 情况一：ATTN_TP 等于整个 TP Group

当 `attn_tp_size == tp_size > 1` 时：

各 Rank 共同形成同一 Attention 输出，`RowParallelLinear` 内部直接在全局 TP Group 归约。

~~~text
wo_b rank 0 partial ─┐
wo_b rank 1 partial ─┤
...                   ├─ TP AllReduce → 完整 Attention output
wo_b rank N partial ─┘
~~~

### 情况二：DPA 把 TP world 切成多个 Attention shard

DPA 的例子则是：

~~~text
TP = 16
DP = 8
ATTN_TP = 2
~~~

此时满足 `1 < attn_tp_size < global tp_size`。

八个 Attention-DP 组可能处理不同请求，每组只有两个 Rank 协作。全局 TP AllReduce 会把不同组的请求结果混加，因此归约必须限制在各自 Attention TP 子组内。

固定代码先设置 `wo_b.reduce_results=False`，留下局部贡献，再显式调用 `attn_tp_all_reduce(o)`。它使用 `attn_tp_group`，归约范围因此与当前 Attention 请求组一致。

对应执行路径为：

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

若 TP=DP，例如 TP8/DP8，得到 `attn_tp_size = 1`。

Attention-DP 组中只有一个 Rank，不存在待跨 Rank 合并的 Attention TP 贡献。

三种布局可对照为：

| Attention layout | wo_b 后怎样归约 |
| --- | --- |
| ATTN_TP=1 | 不需要 TP reduce |
| ATTN_TP=global TP>1 | RowParallelLinear 内部 global TP AllReduce |
| `1 < ATTN_TP < global TP` | wo_b 不归约，随后显式 ATTN_TP Group AllReduce |

mHC AllReduce fusion 同样受范围约束：要求 global TP 等于 ATTN_TP、宽度为 4，且 `wo_b.reduce_results=True`。它融合的是 full-TP 归约，不能当作 DPA 子组后的另一轮求和。

## 四、DP Attention 怎样恢复完整 Token 行

DeepSeek-V4 从 DPA Attention 进入 no-A2A MoE 时，会通过 `dp_gather_replicate` 准备输入。

replicate 指出同一 Attention-DP 组内已持有完整的重复结果。例如元素都已经为 7，再加一次便成 14。因此收集各 DP 行时，还要避免组内副本重复贡献。

`_dp_gather` 根据布局选择下列实现：

~~~text
AllGatherV
AllGather
zero-fill + AllReduce
~~~

它们共同要求全局 buffer 的每个逻辑 Token 只贡献一次，并保持正确行序。SUM_LEN 按各 DP 行数之和组织空间；MAX_LEN 先把每组补到相同长度再收集，这会影响接口和切分方式。

### AllGatherV：ATTN_TP=1 时最直接

`is_dp_gatherv_active` 在固定版本要求：

~~~text
SGLANG_DP_USE_GATHERV 开启
ATTN_TP=1
tp_size == attn_dp_size
SUM_LEN
非 elastic WORLD gather
~~~

ATTN_TP=1 时，每 DP shard 只有一份贡献，可以按真实行数变长收集：`[M0,H] + [M1,H] + ... → [M_global,H]`。

### SUM_LEN + AllReduce：互不重叠的非零区间

这条 SUM_LEN 路径虽然调用 AllReduce，目的仍是收集不同 DP 的行。它先安排互不重叠的非零区间，再借求和保留各段。

每个 Rank 先清零 global buffer，只在自己 DP 对应的行区间写入 local hidden。

`dp_gather_replicate` 遇到 ATTN_TP>1 时，仅让 `attn_tp_rank=0` 写真实值，其余副本保持零。

例如：

~~~text
DP0/ATP0: [A B 0 0 0 0]
DP0/ATP1: [0 0 0 0 0 0]

DP1/ATP0: [0 0 C D 0 0]
DP1/ATP1: [0 0 0 0 0 0]

DP2/ATP0: [0 0 0 0 E F]
DP2/ATP1: [0 0 0 0 0 0]
~~~

全 TP Group 再执行 SUM AllReduce，得到 `[A B C D E F]`。

例如 A 的位置只有 DP0/ATP0 非零，求和仍为 A。各段凭借唯一贡献者和零区间进入正确位置，没有让不同 Token 彼此相加。

### MAX_LEN + AllGather：先在 ATTN_TP 内消重，再 full TP Gather

MAX_LEN 在 ATTN_TP=1 时可以直接 AllGather。

宽度更大时，要先处理同组的重复行，否则全 TP AllGather 会把副本也收入，导致行数及请求映射重复。

固定 `dp_gather_replicate` 先将非 `attn_tp_rank=0` 的 local tensor 清零，再沿 Token 维分为 ATTN_TP 份，组内 ReduceScatter。

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

ReduceScatter 中另一份输入为零，因此数值不变，同时把四行分给两个 Rank，形成 `[A B]`、`[C D]`。后面的全 TP AllGather 就能各收一次不同片段。

这三条实现各自保证唯一贡献：

| 路径 | 怎样保证每个 Token 只贡献一次 |
| --- | --- |
| AllGatherV | ATTN_TP=1，天然唯一 contributor |
| AllReduce | 只有 ATP0 写真实值，其余副本为 0 |
| AllGather | 先去掉 replicated duplicate，再用 ATTN_TP ReduceScatter 形成唯一 row shards |

上面普通非 elastic 路径主要使用 TP Group；Elastic EP 某些 gather 会改用 expanded WORLD Group。

## 五、ReduceScatter 怎样合并归约与分发

no-A2A 的专家输出还可以按以下方式返回。

### Baseline：post-expert AllReduce + dp_scatter

基线路径先做：

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

各 Rank 先获得归约后的全局结果，再取回本地请求行。如果后续只需本地部分，可以尝试把归约与分发合并。

### SUM_LEN 优化：ReduceScatterV 吸收 post-expert Reduce

ReduceScatterV 允许各 DP 按真实 Token 数接收不同长度的分块。固定 DeepSeek-V4 在相应条件下设置 `_use_reduce_scatterv`：

~~~text
DPA local→global MoE gather path
A2A backend = none
is_dp_gatherv_active() = true
dp_padding_mode = SUM_LEN
~~~

其中 `is_dp_gatherv_active` 又要求 ATTN_TP=1、`tp_size=attn_dp_size` 并启用环境开关。

Decoder 发布 `mlp_reduce_scatter=True` 后，MoE 跳过内部 post-expert AllReduce，改由后面的 `tp_group.reduce_scatterv(...)` 接管。

它一次完成贡献求和与本地行返回：

~~~text
partial expert output SUM
+
按真实 per-DP token count 返回 local rows
~~~

### MAX_LEN 优化：等长 ReduceScatter

MAX_LEN 的等长路径由 `SGLANG_DP_USE_REDUCE_SCATTER` 控制，并要求：

~~~text
no-A2A DPA→MoE gather path
不是 reduce_scatterv
dp_padding_mode = MAX_LEN
tp_size == attn_dp_size
~~~

满足后也发布 `mlp_reduce_scatter=True`，跳过原内部归约，最后让 `dp_reduce_scatter_tensor` 执行 SUM 并返回等长 shard。

### should_use_dp_reduce_scatterv 是另一条独立 skip 条件

另外，`moe/utils.py` 的 `should_use_dp_reduce_scatterv` 还检查：

~~~text
A2A backend = none
DPA 开启
attn_dp_size > 1
tp_size == attn_dp_size
moe_ep_size == attn_dp_size
并排除特定 FP4 allgather 路径
~~~

这是标准 Dispatcher EP combine 的另一判断，会参与 `should_skip_post_experts_all_reduce`。它和 Decoder 的 `mlp_reduce_scatter` 都能使旧 AllReduce 跳过，但判断位置、后续接管组件不同。

因此，需要分别追踪这些控制点：

~~~text
mlp_reduce_scatter
should_use_dp_reduce_scatterv
should_skip_post_experts_all_reduce
~~~

沿输出往下看，可将归约安排写为：

~~~text
MoE output
    ↓
当前条件判断

A. post-expert AllReduce + dp_scatter
B. skip post-expert reduce → ReduceScatterV
C. skip post-expert reduce → ReduceScatter
~~~

B/C 把求和交给后续 ReduceScatter，因此必须跳过旧 post-expert AllReduce，否则同一贡献会重复求和。验证时既要找到新归约，也要确认旧位置在对应分支确实没有执行。

## 六、Ascend DeepEP 与 FuseEP 的执行路径

MoE All-to-All 将输入从请求所在 Rank 交给专家所在 Rank，计算后再恢复输出。这里 owner 随阶段变化：

~~~text
Request/DP owner
    ↓ Router
Expert owner
    ↓ Expert Compute
Request/DP owner
~~~

Ascend 上至少还要区分 DeepEP 与 ascend_fuseep 的实现。

### Ascend DeepEP：MOE_EP Group + Dispatcher 的 Dispatch/Combine

固定 NPU DeepEP dispatcher 从 `get_parallel().moe_ep_group.device_group` 取得 MOE_EP device process group，交给 Buffer 创建通信资源。底层传输仍由 Buffer 组织，不能仅凭组名推断具体通信原语。

随后进入 `DeepEPDispatcher` / `DeepEPBuffer`，NPU 上又有两种来源：

~~~text
SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
→ ZBAL DeepEP adaptor / buffer

否则
→ deep_ep.Buffer
~~~

Normal Dispatch 路径可沿下面的布局和调用追踪：

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

Routing Layout、各目标 Token 数、异步事件与 Dispatch/Combine 由 DeepEP 或 ZBAL Buffer 承接，构成实际的数据交换。

### Ascend FuseEP：同一个 MOE_EP Group，但绕过普通 Dispatcher

`ascend_fuseep` 则从模型层进入融合路径。

`create_moe_dispatcher` 虽返回 StandardDispatcher，但注释指出它不会执行；`FusedMoE.forward` 直接转到 `forward_fuseep(...)`，对应文件为 `hardware_backend/npu/moe/fuseep.py`。

FuseEP 同样通过 `get_parallel().moe_ep_group.device_group` 取得通信组，之后创建低时延 DeepEPBuffer。

普通路径分别执行 dispatch、expert runner、combine，而 FuseEP 将它们交给同一个入口：

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

输入行、路由结果、专家权重同时交给融合入口，Python 层不再逐个暴露 Dispatch、Expert GEMM、Combine，融合边界可以从这些参数看出来。

在 SGLang Python 层能追到的主线为：

~~~text
Router / TopK
    ↓
forward_fuseep
    ↓
fused_deep_moe(hidden, routing, expert weights, ...)
    ↓
final hidden
~~~

这一段同时包含计算和交换，不能将全部耗时计为纯通信，也不能因为上层只有一次调用就判断底层仅有一个 Kernel。

内部 Kernel、传输、同步由对应 Runtime 安排。继续定位时需沿 DeepEP/ZBAL/Ascend Runtime 追踪，并与 Profiler 事件对应。

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

将 Rank/Group 的坐标与本篇的数据关系连接，可以沿这一层确认通信范围和输出归属：

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

图中 DeepEP 的拆行与恢复对应 `attn_tp_size>1` 的 bridge，宽度为 1 时不需要。no-A2A、FuseEP 是另选的 backend 路径，应按实际配置进入。

读到一次通信，先记录各 Rank 手里的数据是 shard、局部贡献还是完整行，再确认下游需要什么形状，最后核对 Group 与 backend。这样才能判断调用是在求和、拼接还是重新分发，而不只是在时间线上数函数名。
