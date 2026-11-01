# DeepSeek-V4 分布式 Forward：Attention 与 MoE 怎样交换布局

Rank10、Rank11 做完 Attention，各有同样的四行 hidden states。进入 MoE 前，它们沿 Token 行一分为二，各拿两行；专家算完，再收回对方的两行。中间既有对相同输出的求和，也有对不同 Token 的拼接。

为什么要这样交换，要从两个阶段的布局算起。下面固定 `TP32 / DP16`、DP Attention 开启、CP1 的 Target 配置，跟着这一对 Rank 走过归约、拆行和恢复。这是源码布局例子，没有部署性能实测，DSpark Draft 的支持范围另行说明。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线讨论 DeepSeek-V4 target model 的 TP、DP Attention、MoE EP 组合，并单独说明 DSpark speculative path 的限制。Ascend 通信以固定 torch.distributed / HCCL 路径解释，同时保留 ZBAL、FuseEP 等专用 backend 的适用条件。

</details>

## 一、配置宽度怎样变成有效 Attention 布局

TP 描述张量怎样分片计算，DP Attention 分配请求与 Token 行，EP 分配专家权重。运行时还会从启动宽度派生 `attn_dp_size`、`attn_tp_size`，以及 MoE 的 `moe_dp_size`、`moe_ep_size`、`moe_tp_size`，各阶段按自己的分工使用。

`derive_attention_widths()` 检查 DP Attention 开关。开启时，`dp_size` 才成为有效 Attention DP；关闭则取一，再结合 CP 算 Attention TP。[`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281)

```text
attn_dp_size = dp_size（DP Attention 开启时），否则为 1
attn_tp_size = tp_size / attn_dp_size / attn_cp_size

TP32、DP16、CP1、DP Attention 开启：attn_tp_size = 2
TP32、DP16、CP1、DP Attention 关闭：attn_tp_size = 32
```

所以 `--dp-size 16` 不能离开 DPA 开关解释。本文一直开启 DPA，16 是 TP32 内的 Attention 分组数，不是在外层再复制十六套完整模型。

### 按并行模式计算 World Size

暂不计外层 Native DP 副本，只看普通、非弹性、PP=1 的 TP 域。三十二个 Rank 在 Attention 阶段分成十六组、每组两个协作，MoE 则重新按自己的 DP、EP、TP 划分。初始化接收这些阶段宽度，并不会无条件把它们全乘成进程总数。[`initialize_model_parallel()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2515-L2585)

MoE 满足 `moe_tp_size=tp_size/moe_ep_size/moe_dp_size`。同一批进程可写成 Attention 的 `32=16×1×2`，也可写成 `32=moe_dp_size×moe_ep_size×moe_tp_size`。加入 PP、Native DP 或弹性机制后，全局 WORLD 另按对应初始化过程计算。[`derive_parallel_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L336-L377)

### 从 Runtime Context 展开 Rank 坐标

Attention 的 `(dp,cp,tp)` 坐标里，TP 变化最快。CP1 时，`attn_tp_rank=tp_rank%2`、`attn_dp_rank=tp_rank//2`，Rank10、Rank11 因而同在 Attention DP 组五，组内 TP 编号分别为零、一。[`derive_attention_ranks()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L262-L281)

| 当前 TP 域内编号 | Attention DP 编号 | Attention TP 组内编号 |
| --- | --- | --- |
| 10 | 5 | 0 |
| 11 | 5 | 1 |

这两个 Rank 先一起算相同 Token 的 Attention，之后才各接走一段 Token 行去做 MoE。

## 二、Query 分片与 Attention 输出归约

Query 的 `wq_b` 用 `ColumnParallelLinear`，传入的是 `attn_tp_size`，沿输出维切 head。例如六十四 head、Attention TP4，每 Rank 概念上负责十六 head。这是独立的形状算例；TP32/DP16 主线的 Attention TP 仍为二。[`wq_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L862-L870) [`ColumnParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L309-L405)

KV 通过 `wkv` 的 `ReplicatedLinear` 准备。Query heads 分片，共同处理的请求历史仍由请求边界和 KV 表示确定，详见 [Attention 的缓存寻址](deepseek-v4-attention-source-analysis.md)。

`wo_b` 用 `RowParallelLinear`，每个 Rank 先算局部贡献，再按构造条件求和。同一 Token 的各份贡献由 AllReduce 加在一起。[`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L904-L913) [`RowParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L1422-L1696)

| Attention TP 布局 | 默认构造下的归约位置 |
| --- | --- |
| `attn_tp_size=1` | 不需要 Attention TP 归约 |
| `attn_tp_size=global_tp_size>1` | `wo_b` 内部执行 TP AllReduce |
| `1 < attn_tp_size < global_tp_size` | 关闭 `wo_b` 内部归约，随后显式调用 `attn_tp_all_reduce` |

主线走表中第三种条件。Rank10、Rank11 在自己的 Attention TP 子组归约，得到相同的 post-attention Token 行。后面能直接拆行，正因为这时已经拿到完整结果。[`reduce_results` 与 `attn_tp_all_reduce`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py)

## 三、MoE 的专家归属和通信组

MoE 建立 `_MOE_DP`、`_MOE_EP`、`_MOE_TP`，围绕专家权重组织协作，划分可以与 Attention 请求组不同。[`MoE groups`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2770-L2895)

另取 TP8、MoE DP2、EP4，算出 MoE TP 为一，八 Rank 域分成两组 MoE DP，每组四路 EP。这个小例子只解释组宽度，后面的 TP32 A2A 配置用自己的 EP 约束。

<a id="ascend-group"></a>

### Ascend 的 MoE EP group 与 TP 句柄

固定版本只有在 `moe_ep_size==tensor_model_parallel_size` 且非 NPU 时，才让 `_MOE_EP` 复用 `_TP`。NPU 会另建 MoE EP group，即使成员相同，也不等于使用同一个句柄。[`_MOE_EP`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2835-L2865)

成员列表回答谁参与，process-group handle 和 communicator 才能确定通信上下文。Ascend 上的 collective 顺序、初始化、超时，也需要对应到这个上下文。

## 四、Attention 与 MoE 之间的 Token 行转换

`_run_moe_ffn_dp_sync()` 接起 Attention 输出和 MoE，再恢复布局。下面两个分支都没有进入 Prefill CP，gather、scatter 的含义随分支而变。[`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)

### 路径 A：Attention DP > 1，但没有 A2A MoE backend

`attn_dp_size>1`、A2A backend 为 `none` 时，输入散在各 Attention DP 位置，`dp_gather_replicate()` 先组织更大的 Token buffer。Attention 已经归约完，收集时要避免将重复的完整结果再相加。

例如两 Rank 各有完整 `[2,3]`，再求和会重复成 `[4,6]`。若各自拿的是局部贡献 `[1,2]`、`[3,4]`，相加得到 `[4,6]` 才正确。向量形状一样，数值含义决定这一步能否用 SUM。

MoE 之后，由分支选择 `reduce_scatterv`、`reduce_scatter` 或 `dp_scatter`，返回本地 slice。reduce-scatter 如果已合并专家局部贡献，MoE 内的后置 AllReduce 就跳过，这份结果只加一次。padding、gatherv 与相关开关也参与选择。

源码用无 EP、专家中间维走 TP 分片的情形解释 reduce-scatter。专家输出还只是局部贡献，求和时同时分回各 Token slice，`mlp_reduce_scatter` 与后面的恢复分支共同决定归约在哪完成。

行布局经历本地行、收集、MoE、恢复本地行，数值是否需要相加仍取决于上述贡献关系。

### 路径 B：Attention TP > 1，并且使用 A2A MoE backend

TP32/DP16 主线的 Attention TP 为二，MoE A2A backend 不为 `none`，走另一个分支。调用 `self.mlp(...)` 前，`hidden_states.tensor_split(attn_tp_size)` 沿 Token 行切开，各 Rank 依组内编号取一段，Hidden 维仍完整。

假设归约后两 Rank 各有 A、B、C、D 四行，拆分如下。字母代替 Token 行，实际尺寸和 padding 随 batch 而变。

```text
Attention 归约后：
  Rank10: [A B C D]
  Rank11: [A B C D]

沿 Token 行切分后：
  Rank10: [A B]
  Rank11: [C D]
```

现在两 Rank 送去的是不同 Token，完整四行没有重复进入专家。调用若带 `input_ids` 或 `input_ids_global`，也会同步切分，保持 ID 与表示对应；没有传入，就不额外构造 ID。

<a id="moe-moe-attn-tp-allgather"></a>

### MoE 输出怎样回到 Attention TP 布局

MoE 返回，Rank10 有 A、B，Rank11 有 C、D 的输出。两者调用 `attn_tp_all_gather()`，沿 Token 维收集、拼接，重新各有完整四行。

```text
MoE 返回：
  Rank10: [y_A y_B]
  Rank11: [y_C y_D]

attn-TP AllGather 并拼接后：
  Rank10: [y_A y_B y_C y_D]
  Rank11: [y_A y_B y_C y_D]
```

这次 AllGather 拼接不同 Token，前面的 AllReduce 则加合同一 Token 的局部贡献。Expert Combine 已在 MoE 里面结束，这里只恢复 Attention 布局，不去全体 EP Rank 再做一次 Combine。

~~~mermaid
flowchart TD
    A[Post-attention hidden rows · replicated inside attn TP group]
    A --> S[Token-row split · by attn_tp_rank]
    S --> M[MoE · Router + EP + experts + combine]
    M --> G[attn_tp_all_gather]
    G --> N[Replicated rows · for next layer]
~~~

## 五、TP32 / DP16 示例与 DSpark 限制

代回主线，TP32、DPA 开启、DP16、CP1，得到十六组 Attention DP、组内 TP2，组五就是 Rank10、Rank11。选下面约束 EP 的 A2A backend，再取 MoE DP1，MoE 就是 EP32、MoE TP1。

### 使用当前 A2A MoE backend 时，resolved EP 会被强制成 TP width

固定 `_A2A_EP_SPANNING_BACKENDS` 包括 DeepEP、Mooncake、NIXL、Ascend FuseEP、MORI、PPLX、MegaMoE 等，参数后处理将相应 `ep_size` 设为 `tp_size`。完整集合与规则在 [`_a2a_ep_size()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/overrides.py#L1630-L1673)，这里说的是该版本的实现约束。

| 阶段 | 这三十二个 Rank 的分工 |
| --- | --- |
| Attention | DP16 × CP1 × Attention TP2 |
| MoE（上述 A2A 约束、MoE DP1） | MoE DP1 × EP32 × MoE TP1 |

Target 通过 Attention 输出归约、MoE 前拆行、MoE 后 Attention TP AllGather，衔接这两套分工。Draft 还走自己的执行路径，支持范围不会随 Target 自动扩展。

<a id="commit-dspark-moe-draft-attn_tp-1"></a>

### DSpark MoE Draft 的 `attn_tp == 1` 限制

同一提交的 `DSparkWorkerV2` 检查到 DPA 开启、Draft 为 MoE、`attn_tp_size>1` 就抛错。主线宽度二会触发这道检查，Target 已有的布局转换没有替 Draft 补上支持。[`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L159-L175)

错误文本是 `attn_tp > 1 corrupts the MoE-under-DP all-reduce`，指出了风险方向，还没有逐项定位中间状态。支持这类配置需要继续核对 Draft 并行组、Token 行归属、Rank 计数、接受决策同步和 Graph/Verify 布局。删掉检查只能放行启动，无法证明后续生成正确。

限制来自 Target、Draft 的执行路径差别，不能直接归因于 HCCL 不支持这个宽度，进一步分析见 [DSpark 并行限制与布局](../distributed/deepseek-v4-dspark-parallel-layout-analysis.md)。

## 六、Ascend 通信后端的分工

普通 NPU distributed backend 在固定版本映射为 HCCL，启用 ZBAL local memory 时可以改为 `zbal`。[`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)

SGLang 通过 `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options` 等配置进程组，读取默认及 MoE 的 HCCL buffer 选项。标准 collective 沿这些设置追踪，专用 MoE backend 还会进入自己的实现。[`get_torch_distributed_pg_options()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L90-L105)

Ascend FuseEP 的融合入口承接分发、交换、专家计算与合并，Profiler 未必分别列出 `all_reduce`、`all_gather`、`all_to_all`。Python 入口、进程组、backend 与设备事件需要对得上。固定官方示例使用 `--tp-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend deepep`，宽度又不同，不能当作 TP32/DP16 的实测证据。[官方示例](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx#L202-L225)

<a id="tptensor-shard-ownership"></a>
<a id="dp-attentiontoken-ownership"></a>
<a id="epexpert-ownership"></a>

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
