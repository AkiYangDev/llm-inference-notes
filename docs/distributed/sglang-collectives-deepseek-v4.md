# SGLang Collective：从张量分片到 Attention 与 MoE 通信

两个 Rank 都拿着 `[M,H]`，还看不出该求和还是拼接。它们可能算了同一输出的不同贡献，也可能各有不同请求的完整行。到了 MoE，输入还要送去持有目标专家的 Rank。

Collective 处理这些不同的数据关系。先看四种操作各自改变什么，再放进 DeepSeek-V4 的 Attention、MoE，通信范围和输出归属就能与具体调用对应。

<details markdown="1">
<summary>版本与源码入口</summary>

下文使用固定版本源码。

~~~text
sgl-project/sglang
commit: 176dbcb85d3b7737564e4911947035cc0af65f0a
核对日期：2026-09-21
~~~

相关模块入口如下。

- [distributed/communication_op.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/distributed/communication_op.py)
- [layers/linear.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/linear.py)
- [layers/dp_attention.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/dp_attention.py)
- [models/deepseek_v4.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/models/deepseek_v4.py)
- [layers/moe/utils.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/utils.py)
- [layers/moe/token_dispatcher/deepep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)
- [hardware_backend/npu/moe/fuseep.py](https://github.com/sgl-project/sglang/blob/176dbcb85d3b7737564e4911947035cc0af65f0a/python/sglang/srt/hardware_backend/npu/moe/fuseep.py)

同一种交换关系，可以由通用 API、专用库或融合算子实现。高层语义和底层调用因而分别解释。

</details>

## 一、四种 Collective 的输入与输出

假设组为 `Group = [R0, R1, R2, R3]`，参与者按协议共同调用。这里归约统一演示 SUM，AllReduce 本身也支持其他归约操作。

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

这四个数是同一逻辑结果的局部贡献，合起来为 10。SUM AllReduce 把和交给每个参与者，不再分别保留输入的四个位置。

输入各是 [M,H]，输出通常仍是 [M,H]，数值却变成了组内贡献之和。

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

A、B、C、D 表示不同区间，AllGather 按组内顺序收齐，数值不相加。比如一段输出特征的前后两半，需要拼起来才是完整向量。

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

ReduceScatter 也合并局部贡献，但每个 Rank 只拿自己负责的结果区间。下一步继续按 shard 计算，就不必处处保留完整输出。

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

All-to-All 把各块送到各自目标，不对 A/B/C/D 求和。MoE 将 Token 送到专家所在 Rank，正有这类交换关系。算例块等长，实际专家收多少可以不同，还需要计数与重排信息。

| Collective | 输入关系 | 输出关系 | 是否 Reduce | 所有 Rank 都拿完整结果 |
| --- | --- | --- | --- | --- |
| AllReduce | 同一逻辑 Tensor 的 partial contribution | 所有人拿 reduced 结果 | 是 | 是 |
| AllGather | 完整 Tensor 的不同 shard | 所有人拿全部 shard | 否 | 是 |
| ReduceScatter | 同一逻辑 Tensor 的 partial contribution | 每人拿 reduced 结果的一块 | 是 | 否 |
| All-to-All | 每人持有发往不同目标的数据 | 每人只收发给自己的数据 | 否 | 否 |

## 二、TP Linear 需要什么通信

矩阵乘 Y = XW 的两种切法，分别需要拼接和求和。

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

完整输出是 concat(Y0,Y1)。下游如果要求各 Rank 都持有 [M,8]，就要把两份特征收齐。

固定 `ColumnParallelLinear` 在 `gather_output=True` 时调用 `tensor_model_parallel_all_gather`。下游能接 shard，就继续分片；需要完整结果，再 gather。

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

P0、P1 虽都为 [M,H]，各只算了部分输入特征的贡献，完整结果是 P0+P1。`RowParallelLinear` 按需要调用 `tensor_model_parallel_all_reduce`，也可留给后续融合或分支归约。`reduce_results=False` 时，贡献仍待下游合并。

`[Y0 | Y1]` 拼不同位置，`P0 + P1` 加同一输出的贡献。下游要完整结果还是分片结果，再决定是否立即收齐。

## 三、Attention 输出在哪个 Group 内归约

DeepSeek-V4 的 `wo_b` 产生 RowParallel 局部贡献，归约还必须落在正确的 Group。下面按普通非弹性路径讨论，DPA→MoE 也不进入 Prefill CP 专用分支。

`wo_b` 的权重按 `attn_tp_rank`、`attn_tp_size` 切分，默认 `reduce_results=True` 有下面这些条件。

~~~text
attn_tp_size == global tp_size
并且 attn_tp_size > 1
~~~

### 情况一：ATTN_TP 等于整个 TP Group

当 `attn_tp_size == tp_size > 1` 时：

各 Rank 协作形成同一个 Attention 输出，`RowParallelLinear` 直接在完整 TP Group 中归约。

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

此时宽度是 `1 < attn_tp_size < global tp_size`。

八个 Attention-DP 组可能各处理不同请求，每组两个 Rank 协作。全局 TP AllReduce 会把不同请求混加，求和只能在各自 Attention TP 子组中完成。

所以固定代码令 `wo_b.reduce_results=False`，先留下局部贡献，再显式调用 `attn_tp_all_reduce(o)`，使用 `attn_tp_group` 完成子组归约。

这条分支的交接如下。

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

TP=DP，例如 TP8/DP8，便得到 `attn_tp_size = 1`。

每个 Attention-DP 组只有一个 Rank，已经没有待跨 Rank 合并的 Attention TP 贡献。

三种布局放在一起如下。

| Attention layout | wo_b 后怎样归约 |
| --- | --- |
| ATTN_TP=1 | 不需要 TP reduce |
| ATTN_TP=global TP>1 | RowParallelLinear 内部 global TP AllReduce |
| `1 < ATTN_TP < global TP` | wo_b 不归约，随后显式 ATTN_TP Group AllReduce |

mHC AllReduce fusion 也受范围限制，要求 global TP 等于 ATTN_TP、宽度为 4，且 `wo_b.reduce_results=True`。它融合 full-TP 归约，不是 DPA 子组归约后的另一次求和。

## 四、DP Attention 怎样恢复完整 Token 行

DPA Attention 接 no-A2A MoE 时，DeepSeek-V4 通过 `dp_gather_replicate` 准备输入。

replicate 表示同一 Attention-DP 组内已经有多份完整结果。元素都为 7，再加一次就成 14，因此收集各 DP 行前，还要让组内副本只贡献一次。

`_dp_gather` 根据布局选择下面的实现。

~~~text
AllGatherV
AllGather
zero-fill + AllReduce
~~~

这些实现都要保留正确行序，让每个逻辑 Token 在全局 buffer 中只贡献一次。SUM_LEN 按各 DP 的真实行数之和组织空间，MAX_LEN 则先补到共同长度，两种安排使用不同接口和切分方式。

### AllGatherV：ATTN_TP=1 时最直接

`is_dp_gatherv_active` 在固定版本要求：

~~~text
SGLANG_DP_USE_GATHERV 开启
ATTN_TP=1
tp_size == attn_dp_size
SUM_LEN
非 elastic WORLD gather
~~~

ATTN_TP=1，每个 DP shard 只有一份贡献，直接按真实行数变长收集即可，`[M0,H] + [M1,H] + ... → [M_global,H]`。

### SUM_LEN + AllReduce：互不重叠的非零区间

SUM_LEN 这一分支用 AllReduce 收集不同 DP 的行。先让非零区间互不重叠，SUM 就能在各位置留下唯一那份数据。

每个 Rank 清零 global buffer，只往本 DP 的行区间写 local hidden。

ATTN_TP>1 时，`dp_gather_replicate` 只让 `attn_tp_rank=0` 写真实值，其余副本维持零。

例如：

~~~text
DP0/ATP0: [A B 0 0 0 0]
DP0/ATP1: [0 0 0 0 0 0]

DP1/ATP0: [0 0 C D 0 0]
DP1/ATP1: [0 0 0 0 0 0]

DP2/ATP0: [0 0 0 0 E F]
DP2/ATP1: [0 0 0 0 0 0]
~~~

完整 TP Group 做 SUM AllReduce，得到 `[A B C D E F]`。

A 的位置只有 DP0/ATP0 非零，求和后仍是 A。不同 Token 没互相相加，各自靠唯一贡献者进入相应区间。

### MAX_LEN + AllGather：先在 ATTN_TP 内消重，再 full TP Gather

MAX_LEN 在 ATTN_TP=1 时直接 AllGather。

组内宽度更大，完整副本若原样参与全 TP AllGather，会把重复行也收进去，请求映射跟着重复。

固定 `dp_gather_replicate` 先将非 `attn_tp_rank=0` 的 local tensor 清零，再把 Token 维分成 ATTN_TP 份，做组内 ReduceScatter。

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

ReduceScatter 的另一份输入为零，求和不改数值，同时将四行分给两个 Rank，分别得到 `[A B]`、`[C D]`。后续完整 TP AllGather 各收一次，便没有重复。

三条实现保证唯一贡献的方式如下。

| 路径 | 怎样保证每个 Token 只贡献一次 |
| --- | --- |
| AllGatherV | ATTN_TP=1，天然唯一 contributor |
| AllReduce | 只有 ATP0 写真实值，其余副本为 0 |
| AllGather | 先去掉 replicated duplicate，再用 ATTN_TP ReduceScatter 形成唯一 row shards |

这里的普通非 elastic 路径主要用 TP Group，Elastic EP 的某些 gather 会换成 expanded WORLD Group。

## 五、ReduceScatter 怎样合并归约与分发

no-A2A MoE 结束以后，专家贡献还需要归约、返回本地请求行。

### Baseline：post-expert AllReduce + dp_scatter

Baseline 将这两件事分开完成。

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

各 Rank 先拿到归约后的全局结果，再切回本地请求行。下游只需本地部分时，就有机会合并归约与分发。

### SUM_LEN 优化：ReduceScatterV 吸收 post-expert Reduce

ReduceScatterV 支持按各 DP 的真实 Token 数返回不同长度分块。DeepSeek-V4 在下面的条件下设置 `_use_reduce_scatterv`。

~~~text
DPA local→global MoE gather path
A2A backend = none
is_dp_gatherv_active() = true
dp_padding_mode = SUM_LEN
~~~

其中 `is_dp_gatherv_active` 还要求 ATTN_TP=1、`tp_size=attn_dp_size`，并开启相应环境开关。

Decoder 发布 `mlp_reduce_scatter=True`，MoE 便跳过内部 post-expert AllReduce，由后续 `tp_group.reduce_scatterv(...)` 接管归约。

贡献求和和本地行返回，在这一次调用里完成。

~~~text
partial expert output SUM
+
按真实 per-DP token count 返回 local rows
~~~

### MAX_LEN 优化：等长 ReduceScatter

MAX_LEN 的等长分支由 `SGLANG_DP_USE_REDUCE_SCATTER` 控制，条件如下。

~~~text
no-A2A DPA→MoE gather path
不是 reduce_scatterv
dp_padding_mode = MAX_LEN
tp_size == attn_dp_size
~~~

满足后，同样发布 `mlp_reduce_scatter=True`、跳过旧内部归约，最后由 `dp_reduce_scatter_tensor` 做 SUM，返回等长 shard。

### should_use_dp_reduce_scatterv 是另一条独立 skip 条件

`moe/utils.py` 中另有 `should_use_dp_reduce_scatterv`，检查下面这些条件。

~~~text
A2A backend = none
DPA 开启
attn_dp_size > 1
tp_size == attn_dp_size
moe_ep_size == attn_dp_size
并排除特定 FP4 allgather 路径
~~~

它是标准 Dispatcher EP combine 的独立判断，参与 `should_skip_post_experts_all_reduce`。它和 Decoder 的 `mlp_reduce_scatter` 都能跳过旧 AllReduce，但判断位置和后面的接管组件不同。

这几个控制点各自对应的后续操作如下。

~~~text
mlp_reduce_scatter
should_use_dp_reduce_scatterv
should_skip_post_experts_all_reduce
~~~

从输出看，归约安排可以写成下面几条路径。

~~~text
MoE output
    ↓
当前条件判断

A. post-expert AllReduce + dp_scatter
B. skip post-expert reduce → ReduceScatterV
C. skip post-expert reduce → ReduceScatter
~~~

B/C 把求和交给后面的 ReduceScatter，所以旧 post-expert AllReduce 必须跳过，否则同一贡献会加两次。新归约与旧位置的 skip 要在同一分支里对应。

## 六、Ascend DeepEP 与 FuseEP 的执行路径

MoE All-to-All 把输入从请求所在 Rank 送往专家所在 Rank，算完再恢复输出，owner 随阶段变化。

~~~text
Request/DP owner
    ↓ Router
Expert owner
    ↓ Expert Compute
Request/DP owner
~~~

Ascend 的 DeepEP、ascend_fuseep 使用不同入口。

### Ascend DeepEP：MOE_EP Group + Dispatcher 的 Dispatch/Combine

固定 NPU DeepEP dispatcher 从 `get_parallel().moe_ep_group.device_group` 取 MOE_EP device process group，交给 Buffer 创建资源。传输由 Buffer 继续组织，Group 名字本身没有规定使用哪种通信原语。

进入 `DeepEPDispatcher` / `DeepEPBuffer` 后，NPU 有下面两种实现来源。

~~~text
SGLANG_ZBAL_LOCAL_MEM_SIZE > 0
→ ZBAL DeepEP adaptor / buffer

否则
→ deep_ep.Buffer
~~~

Normal Dispatch 的布局与调用如下。

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

DeepEP 或 ZBAL Buffer 处理 Routing Layout、各目标 Token 数、异步事件和 Dispatch/Combine，实际交换从这些参数与事件继续往下。

### Ascend FuseEP：同一个 MOE_EP Group，但绕过普通 Dispatcher

`ascend_fuseep` 从模型层直接走融合路径。

`create_moe_dispatcher` 返回的 StandardDispatcher 不实际执行，这点注释已有说明。`FusedMoE.forward` 转到 `hardware_backend/npu/moe/fuseep.py` 的 `forward_fuseep(...)`。

FuseEP 也取 `get_parallel().moe_ep_group.device_group`，随后创建低时延 DeepEPBuffer。

普通路径分开调 dispatch、expert runner、combine，FuseEP 把几件事交到同一入口。

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

输入行、路由结果和专家权重一起传入，Python 不再分开呈现 Dispatch、Expert GEMM、Combine，参数也显示了这个融合范围。

SGLang Python 层的调用主线如下。

~~~text
Router / TopK
    ↓
forward_fuseep
    ↓
fused_deep_moe(hidden, routing, expert weights, ...)
    ↓
final hidden
~~~

这个区间同时有计算和交换，不能全部计成纯通信；一次上层调用也可能启动多个 Kernel。

内部 Kernel、传输和同步由相应 Runtime 安排，需要与 DeepEP、ZBAL、Ascend Runtime 以及 Profiler 事件对应。

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

把 Rank/Group 坐标代入一层计算，数据归属会经历这些变化。

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

图里的 DeepEP 拆行与恢复对应 `attn_tp_size>1` 的 bridge，宽度为 1 时省去。no-A2A、FuseEP 则是另外选择的 backend 路径，分别按配置进入。

数据同为 `[M,H]`，局部贡献、分片和完整行需要的操作仍然不同。下游所需形状、当前 Group 与 backend 一起决定这次通信做什么，时间线上的函数名只是其中一部分信息。
