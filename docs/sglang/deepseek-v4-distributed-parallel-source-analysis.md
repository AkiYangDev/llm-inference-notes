# DeepSeek-V4 分布式 Forward：Attention 与 MoE 怎样交换布局

两个 Rank 完成 Attention 后，可能各持相同的四行 hidden states；进入 MoE 时，又各取两行计算；等专家输出回来，再收齐为四行。这些交换服务于不同的分工：Attention 协作计算相同 Token，MoE 则按专家布局处理分给自己的输入。

下面固定一个 `TP32 / DP16`、DP Attention 开启、CP1 的 Target 布局，先算 Attention 实际宽度，再跟踪归约、拆行和恢复。这是源码布局例子，没有对应的部署性能实测；DSpark Draft 能否使用相同配置，还得检查另一条路径。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线讨论 DeepSeek-V4 target model 的 TP、DP Attention、MoE EP 组合，并单独说明 DSpark speculative path 的限制。Ascend 通信以固定 torch.distributed / HCCL 路径解释，同时保留 ZBAL、FuseEP 等专用 backend 的适用条件。

</details>

## 一、配置宽度怎样变成有效 Attention 布局

启动参数的 TP、DP、EP 还不能直接描述所有阶段。运行时会分别派生 Attention 的 `attn_dp_size`、`attn_tp_size`，以及 MoE 的 `moe_dp_size`、`moe_ep_size`、`moe_tp_size`。

`derive_attention_widths()` 先判断 DP Attention 是否开启。开启才把 `dp_size` 作为有效 Attention DP，关闭时有效宽度为一，再结合 CP 算 Attention TP。[`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281)

```text
attn_dp_size = dp_size（DP Attention 开启时），否则为 1
attn_tp_size = tp_size / attn_dp_size / attn_cp_size

TP32、DP16、CP1、DP Attention 开启：attn_tp_size = 2
TP32、DP16、CP1、DP Attention 关闭：attn_tp_size = 32
```

所以 `--dp-size 16` 还要连同 DPA 开关阅读。后文始终启用 DPA，16 表示 TP32 内部的 Attention 分组，不能再在外层复制十六套完整模型。

### 按并行模式计算 World Size

先看普通、非弹性、PP=1 的 TP 域，并暂时排除外层 Native DP 副本。域内三十二个 Rank 在 Attention 中分成十六组，每组两个协作；MoE 再按自身 DP、EP、TP 划分。初始化接收各阶段宽度，不会把所有 size 无条件相乘为进程总数。[`initialize_model_parallel()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2515-L2585)

MoE 宽度满足 `moe_tp_size=tp_size/moe_ep_size/moe_dp_size`。本例可分别写成 `32=16×1×2`、`32=moe_dp_size×moe_ep_size×moe_tp_size`，两者描述同一批进程。加入 PP、Native DP 或弹性机制后，全局 WORLD 仍需回到对应初始化过程计算。[`derive_parallel_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L336-L377)

### 从 Runtime Context 展开 Rank 坐标

Attention 坐标按 `(dp,cp,tp)` 排列，TP 变化最快。CP1 时，`attn_tp_rank=tp_rank%2`、`attn_dp_rank=tp_rank//2`，因而当前 TP 域的 Rank10、Rank11 同在 DP 组五，组内 TP 编号分别为零和一。[`derive_attention_ranks()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L262-L281)

| 当前 TP 域内编号 | Attention DP 编号 | Attention TP 组内编号 |
| --- | --- | --- |
| 10 | 5 | 0 |
| 11 | 5 | 1 |

接下来跟踪这一对 Rank：Attention 中共同计算相同 Token，MoE 前再各自接走一段 Token 行。

## 二、Query 分片与 Attention 输出归约

`wq_b` 使用 `ColumnParallelLinear`，传入 `attn_tp_size` 而非外层 `tp_size`，沿输出维拆开 Query heads。单独举一个六十四 head、Attention TP4 的形状例子，每 Rank 概念上负责十六 head；本篇主线 TP32/DP16 的 Attention 宽度为二，两组数字不能混用。[`wq_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L862-L870) [`ColumnParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L309-L405)

KV 由 `wkv` 的 `ReplicatedLinear` 准备。Query heads 分片不等于各 Rank 持有互不相关的请求历史，还需区分 head 分工、请求边界与 KV 表示，详见 [Attention 的缓存寻址](deepseek-v4-attention-source-analysis.md)。

`wo_b` 使用 `RowParallelLinear`，先形成局部贡献，再按构造条件求和。这里需要合并同一输出的贡献，AllReduce 处理的是数值相加。[`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L904-L913) [`RowParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L1422-L1696)

| Attention TP 布局 | 默认构造下的归约位置 |
| --- | --- |
| `attn_tp_size=1` | 不需要 Attention TP 归约 |
| `attn_tp_size=global_tp_size>1` | `wo_b` 内部执行 TP AllReduce |
| `1 < attn_tp_size < global_tp_size` | 关闭 `wo_b` 内部归约，随后显式调用 `attn_tp_all_reduce` |

主线对应表中第三种条件。Rank10、Rank11 在自己的 Attention TP 子组归约后，各有相同的 post-attention Token 行，后面的拆行以此为前提。若跳过归约只看切片，就无法解释输入到底是完整结果还是局部贡献。[`reduce_results` 与 `attn_tp_all_reduce`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py)

## 三、MoE 的专家归属和通信组

MoE 创建 `_MOE_DP`、`_MOE_EP`、`_MOE_TP`，按专家权重归属组织协作，其组划分不必等于 Attention 请求组。[`MoE groups`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2770-L2895)

另举 TP8、MoE DP2、EP4 的小例子，可算出 MoE TP 为一：八 Rank 域分成两组 MoE DP，每组四路 EP。这只解释组宽度，不能将 EP4 套进后文 TP32 的 A2A 配置。

### Ascend 上还有一个很具体的 group 细节

组成员相同也不保证句柄相同。固定版本仅在 `moe_ep_size==tensor_model_parallel_size` 且非 NPU 时，让 `_MOE_EP` 直接复用 `_TP`；NPU 会另建 MoE EP group。[`_MOE_EP`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2835-L2865)

因此在 Ascend 上核对 collective 顺序、初始化或超时，还需看 process-group handle 与 communicator。成员列表能确定谁参与，不能证明两次调用使用同一通信上下文。

## 四、Attention 与 MoE 之间的 Token 行转换

`_run_moe_ffn_dp_sync()` 将 Attention 输出接入 MoE，再恢复布局。先看条件才能知道 gather、scatter 交换什么；下面两条路径都没有进入 Prefill CP。[`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)

### 路径 A：Attention DP > 1，但没有 A2A MoE backend

当 `attn_dp_size>1`、A2A backend 为 `none` 时，输入散在各 Attention DP 位置，先由 `dp_gather_replicate()` 组织更大的 Token buffer。Attention 已完成归约，收集这些行时不能再把重复的完整结果当局部贡献相加。

例如两个 Rank 都已有完整 `[2,3]`，再次求和就重复成 `[4,6]`；如果分别持局部贡献 `[1,2]`、`[3,4]`，求和为 `[4,6]` 才是预期。教学向量虽然形状相同，数值的含义却决定能否执行 SUM。

MoE 之后按分支使用 `reduce_scatterv`、`reduce_scatter` 或 `dp_scatter` 返回本地 slice。如果 reduce-scatter 同时完成专家局部贡献的求和，MoE 内部的后置 AllReduce 就要跳过，避免重复归约。padding、gatherv 及开关也参与选择。

源码以无 EP、专家中间维走 TP 分片的情形解释 reduce-scatter：专家输出仍是待合并贡献，可在求和时一并分回各 Token slice。读这条路径时应连同 `mlp_reduce_scatter` 和后续恢复分支一起检查，确认求和只做一次。

因此，“本地行 → 收集 → MoE → 恢复本地行”只是行布局过程，是否需要加法仍由数据含义决定。

### 路径 B：Attention TP > 1，并且使用 A2A MoE backend

TP32/DP16 主线走另一分支：Attention TP 为二，MoE A2A backend 不为 `none`。调用 `self.mlp(...)` 前，代码用 `hidden_states.tensor_split(attn_tp_size)` 沿 Token 行切开，各 Rank 按组内编号取一段，Hidden 维保持完整。

假设 Rank10、Rank11 在归约后各有 A、B、C、D 四行，则拆分如下。字母只代表 Token 行，真实尺寸与 padding 要看实际 batch。

```text
Attention 归约后：
  Rank10: [A B C D]
  Rank11: [A B C D]

沿 Token 行切分后：
  Rank10: [A B]
  Rank11: [C D]
```

两 Rank 现在提交不同 Token，避免把完整四行重复送往专家。如果本次调用还提供 `input_ids` 或 `input_ids_global`，也要同步切分以维持 ID 与表示的对应；未提供时则不构造额外 ID。

### MoE 后的时序同样明确：先 MoE，再 attn-TP AllGather

MoE 返回后，Rank10 持 A、B，Rank11 持 C、D 的输出。`attn_tp_all_gather()` 在这两个 Rank 间收齐并沿 Token 维拼接，使两边再次拥有完整四行。

```text
MoE 返回：
  Rank10: [y_A y_B]
  Rank11: [y_C y_D]

attn-TP AllGather 并拼接后：
  Rank10: [y_A y_B y_C y_D]
  Rank11: [y_A y_B y_C y_D]
```

此处 AllGather 拼接不同 Token 行，前面的 AllReduce 合并相同 Token 的局部贡献。Expert Combine 已在 MoE 内完成，现在只恢复 Attention 需要的排列，不会再到全部 EP Rank 重做 Combine。

~~~mermaid
flowchart TD
    A[Post-attention hidden rows · replicated inside attn TP group]
    A --> S[Token-row split · by attn_tp_rank]
    S --> M[MoE · Router + EP + experts + combine]
    M --> G[attn_tp_all_gather]
    G --> N[Replicated rows · for next layer]
~~~

## 五、TP32 / DP16 示例与 DSpark 限制

把主线代回配置：TP32、DPA 开启、DP16、CP1 得到十六组 Attention DP、组内 TP2，其中组五是 Rank10、Rank11。若选中下面约束 EP 的 A2A backend，并取 MoE DP1，则 MoE 使用 EP32、MoE TP1。

### 使用当前 A2A MoE backend 时，resolved EP 会被强制成 TP width

固定 `_A2A_EP_SPANNING_BACKENDS` 含 DeepEP、Mooncake、NIXL、Ascend FuseEP、MORI、PPLX、MegaMoE 等，参数后处理令对应 `ep_size=tp_size`。这是当前实现的约束，完整集合和版本需对照源码。[`_a2a_ep_size()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/overrides.py#L1630-L1673)

| 阶段 | 这三十二个 Rank 的分工 |
| --- | --- |
| Attention | DP16 × CP1 × Attention TP2 |
| MoE（上述 A2A 约束、MoE DP1） | MoE DP1 × EP32 × MoE TP1 |

Target 通过 Attention 输出归约、MoE 前拆行、专家计算后的 Attention TP AllGather 衔接这两种分工。这条路径说明 Target 如何处理布局，还不能证明 Draft 或任意 backend 组合也支持这些参数。

### 但是固定 commit 的 DSpark MoE draft 明确要求 `attn_tp == 1`

同一提交 `DSparkWorkerV2` 若发现 DPA 开启、Draft 为 MoE、`attn_tp_size>1`，会直接抛错。本例宽度二会触发该检查，Target 的布局桥接并未为 Draft 自动补上支持。[`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L159-L175)

错误文本 `attn_tp > 1 corrupts the MoE-under-DP all-reduce` 指出风险方向，但没有定位每个中间状态。扩展支持至少需核对 Draft 并行组、Token 行归属、各 Rank 计数、接受决策同步及 Graph/Verify 布局要求。删除检查只是放行启动，正确生成仍需证明。

这项限制属于 Target 与 Draft 的不同执行路径，不能直接解释为 HCCL 不支持某个宽度。更完整分析见 [DSpark 并行限制与布局](../distributed/deepseek-v4-dspark-parallel-layout-analysis.md)。

## 六、Ascend 通信后端的分工

明确交换目的后，再看实现。固定版本将普通 NPU distributed backend 映射为 HCCL，配置 ZBAL local memory 时可以改用 `zbal`。[`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)

SGLang 使用 `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options` 等设置进程组，并读取默认及 MoE 的 HCCL buffer 配置。标准 process-group collective 可沿这些选项追踪，专用 MoE backend 则还需继续查看自身入口。[`get_torch_distributed_pg_options()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L90-L105)

Ascend FuseEP 由融合入口承接分发、交换、专家计算和合并，Profiler 未必逐一呈现 `all_reduce`、`all_gather`、`all_to_all` 窗口。分析时要将 Python 入口、进程组、backend 和设备事件对应。固定官方示例中的 `--tp-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend deepep` 又是另一组宽度，不能用来证明 TP32/DP16 已实测。[官方示例](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx#L202-L225)

<a id="tptensor-shard-ownership"></a>
<a id="dp-attentiontoken-ownership"></a>
<a id="epexpert-ownership"></a>

核对这类通信时，TP 说明张量由谁分片计算，DP Attention 说明请求和 Token 行由哪组负责，EP 说明专家权重放在哪里。再追一层当前数据：它是分片、待相加的局部贡献，还是已经完整的结果？这会直接决定下一步应当求和，还是只需收集和拼接。

回到 Rank10、Rank11：先把同一 Token 的 Attention 贡献相加，再分开不同 Token 去做 MoE，最后收齐输出。读到 collective 时，既要看行如何排列，也要看其中数值是完整结果还是待相加的贡献，才能判断下一步为何需要这次交换。

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
