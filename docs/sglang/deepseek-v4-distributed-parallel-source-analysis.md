# DeepSeek-V4 分布式 Forward：Attention 与 MoE 怎样交换布局

Attention 算完后，两个 Rank 可能各自持有相同的四行 hidden states；进入 MoE 时，却只让第一个 Rank 处理其中两行，另一个处理其余两行。等专家计算完成，又把四行重新收齐。这几次变化看起来绕，原因在于同一批进程在 Attention 和 MoE 中承担不同分工。

下面沿 `TP32 / DP16`、开启 DP Attention、CP1 的 Target 模型布局解释这次交接。先算每条请求实际使用的 Attention TP 宽度，再看归约、Token 行拆分和结果恢复。它是固定源码的布局例子，不是一份实测部署或性能报告；DSpark Draft 是否支持同一配置，要单独检查。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线聚焦 DeepSeek-V4 target model 的 TP、DP Attention 与 MoE EP 组合。本文会单独指出 DSpark speculative path 的额外限制，避免把 target model 能表达的拓扑误写成 DSpark 已经支持的拓扑。Ascend 通信以当前 torch.distributed / HCCL 路径为基础说明，同时标注 ZBAL / FuseEP 等专用 backend 的例外。

</details>

## 一、配置宽度怎样变成有效 Attention 布局

启动参数里的 TP、DP、EP 不能直接当作模型每个阶段的分工。执行侧还会派生 `attn_dp_size`、`attn_tp_size`、`moe_dp_size`、`moe_ep_size` 和 `moe_tp_size`，分别描述 Attention 与 MoE 使用的布局。

`derive_attention_widths()` 先看是否开启 DP Attention。开启时，`dp_size` 才成为有效 Attention DP 宽度；关闭时，这个有效宽度是一。随后再结合 CP 计算 Attention TP：[`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L249-L281)

```text
attn_dp_size = dp_size（DP Attention 开启时），否则为 1
attn_tp_size = tp_size / attn_dp_size / attn_cp_size

TP32、DP16、CP1、DP Attention 开启：attn_tp_size = 2
TP32、DP16、CP1、DP Attention 关闭：attn_tp_size = 32
```

所以同样写着 `--dp-size 16`，开关不同就会产生不同的 Attention 布局。后文一直使用开启状态，不把这个参数解释成在 TP32 外再复制十六套完整模型。

### 按并行模式计算 World Size

先限定一个普通、非弹性、PP=1 的 TP 域，也不把外层 Native DP 副本混进来。这个域里的三十二个 Rank，在 Attention 阶段划成十六个 DP 组，每组两个 Rank 协作；在 MoE 阶段，还会重新按 MoE DP、EP 与 MoE TP 组织分工。初始化入口接收这些不同宽度，而不是将所有名字里的 size 逐项相乘来增加进程数。[`initialize_model_parallel()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2515-L2585)

MoE 侧的派生关系是 `moe_tp_size=tp_size/moe_ep_size/moe_dp_size`。因此，在本文限定的域内，可以分别对照 `32=16×1×2` 与 `32=moe_dp_size×moe_ep_size×moe_tp_size`。这说明同一批进程有两种布局，不代表所有运行模式的全局 WORLD 都能只用 TP 推算；PP、Native DP 或弹性加入等情形需要回到相应初始化过程。[`derive_parallel_widths()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L336-L377)

### 从 Runtime Context 展开 Rank 坐标

固定版本按 `(dp,cp,tp)` 排列 Attention 坐标，TP 是变化最快的维度。CP1 时可以直接写成 `attn_tp_rank=tp_rank%2`、`attn_dp_rank=tp_rank//2`。所以本文 TP 域内的 Rank10、Rank11 同属 Attention DP 组五，在组内分别是 TP Rank0、TP Rank1。[`derive_attention_ranks()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/runtime_context.py#L262-L281)

| 当前 TP 域内编号 | Attention DP 编号 | Attention TP 组内编号 |
| --- | --- | --- |
| 10 | 5 | 0 |
| 11 | 5 | 1 |

后面跟住这一对进程即可。它们在 Attention 中协作处理同一组 Token，在 MoE 交接时再把这些 Token 行分开。

## 二、Query 分片与 Attention 输出归约

DeepSeek-V4 的 `wq_b` 使用 `ColumnParallelLinear`，构造时传入的是 `attn_tp_size`，而不是外层 `tp_size`。它沿输出维分片，让当前 Attention TP 组内的 Rank 各自计算一部分 Query head。若用六十四个 head、Attention TP4 作独立的形状例子，每个 Rank 概念上负责十六个 head；本文 TP32/DP16 对应的有效宽度则是二，不要把两个例子混用。[`wq_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L862-L870) [`ColumnParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L309-L405)

共享 KV 一侧由 `wkv` 的 `ReplicatedLinear` 准备。多个 Rank 分担 Query head 的计算，并不意味着每个 Rank 都只保留互不相干的请求历史。对照[Attention 的缓存寻址](deepseek-v4-attention-source-analysis.md)时，还要继续区分 head 分片、请求边界与 KV 表示。

输出侧的 `wo_b` 使用 `RowParallelLinear`，每个 Rank 先得到局部贡献，再根据构造条件决定在哪里求和。这里的 AllReduce 解决“同一输出的不同贡献如何合成”，不是把来自不同请求的行接在一起。[`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L904-L913) [`RowParallelLinear`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/linear.py#L1422-L1696)

| Attention TP 布局 | 默认构造下的归约位置 |
| --- | --- |
| `attn_tp_size=1` | 不需要 Attention TP 归约 |
| `attn_tp_size=global_tp_size>1` | `wo_b` 内部执行 TP AllReduce |
| `1<attn_tp_size<global_tp_size` | 关闭 `wo_b` 内部归约，随后显式调用 `attn_tp_all_reduce` |

本文属于第三行：Rank10、Rank11 的局部 Attention 输出，在两个 Rank 组成的子组中归约。归约完成后，它们各自拿到相同的 post-attention Token 行。后面拆行正是以这个状态为前提，不能省略归约步骤，只看 MoE 前的一次切片。[`reduce_results` 与 `attn_tp_all_reduce`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py)

## 三、MoE 的专家归属和通信组

MoE 根据专家权重的位置组织计算，SGLang 为它建立 `_MOE_DP`、`_MOE_EP`、`_MOE_TP`。这套组描述的是 MoE 阶段的协作关系，不必与 Attention 的请求分组相同。[`MoE groups`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2770-L2895)

用另一组较小的数字说明宽度：TP8、MoE DP2、EP4 时，MoE TP 为一，同一个八 Rank 域被解释为两组 MoE DP，每组四路专家并行。这只是组划分例子，不能将其中的 EP4 填回后面 TP32 的 A2A 后端配置。

### Ascend 上还有一个很具体的 group 细节

成员相同，也未必是同一个通信组。固定版本只有在 `moe_ep_size==tensor_model_parallel_size` 且不是 NPU 时，才直接让 `_MOE_EP` 复用 `_TP`；NPU 会独立创建 MoE EP group。[`_MOE_EP`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L2835-L2865)

因此，在 Ascend 上看到两套 group 都列出相同的 Rank，仍要继续核对其 process-group handle 和 communicator。排查 collective 顺序、初始化或通信超时时，成员列表只能说明参与者，不能单独证明调用使用了同一套通信上下文。

## 四、Attention 与 MoE 之间的 Token 行转换

`_run_moe_ffn_dp_sync()` 负责把 Attention 的输出布局接到 MoE，再把结果恢复回来。先看源码的条件分支，确定本轮走哪条路径；名字里都带 gather 或 scatter，并不意味着交换的是同一种数据。下面两条路径都限定本轮没有走 Prefill CP。[`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)

### 路径 A：Attention DP > 1，但没有 A2A MoE backend

当 `attn_dp_size>1`、MoE A2A backend 为 `none` 时，输入 Token 分散在各 Attention DP 位置，MoE 前先通过 `dp_gather_replicate()` 组织更大的 Token buffer。源码特别说明，Attention 已经归约完成，这里要收集已有行，不能再把相同 Attention 结果当成局部贡献求和。

用一个向量例子区分这两种状态：若同一 Attention TP 组的两个 Rank 都已持有完整结果 `[2,3]`，再次求和会变成 `[4,6]`，数值就重复累计了。而两个 Rank 分别持有同一结果的局部贡献 `[1,2]`、`[3,4]` 时，求和得到 `[4,6]` 才是组合贡献。这里的整数只是教学示意，说明形状相同并不能决定能否做 SUM，还要看每份数值代表什么。

MoE 执行后，根据所选路径用 `reduce_scatterv`、`reduce_scatter` 或 `dp_scatter` 恢复本地 Token slice。它们不能只按接口名称互换：如果使用 reduce-scatter 承担专家局部结果的求和与分发，代码会相应跳过 MoE 内部的后置 AllReduce，避免同一份贡献被归约两次。具体选择还受 padding、gatherv 和配置开关限制。

源码注释以无 EP、专家中间维按 TP 分片的场景解释这组 reduce-scatter 路径：此时专家输出仍需合并局部贡献。reduce-scatter 可以把求和与返回各自 Token slice 组合在一次通信里，但前提是其他位置不再重复执行同一份求和。阅读此处的开关时，应连着检查 MoE 的 `mlp_reduce_scatter` 状态和调用后的恢复分支。

这条路径可以按“DP-local 行 → 收集 → MoE → 恢复本地行”来读，但是否还包含求和，要回到当前分支的张量含义判断。一次 Gather 不是天然的归约，一次 Scatter 也不能自动替代结果求和。

### 路径 B：Attention TP > 1，并且使用 A2A MoE backend

本文 TP32/DP16 的主线使用这条路径：有效 Attention TP 为二，且 MoE A2A backend 不为 `none`。进入 `self.mlp(...)` 前，源码用 `hidden_states.tensor_split(attn_tp_size)` 沿 Token 行拆分，每个 Rank 取自己组内编号对应的一段；每行 Hidden 维保持完整。

假设 Rank10、Rank11 归约后各有四行 A、B、C、D，教学布局如下。字母代表 Token 行，不代表专家编号或请求池编号；实际批次的行数与 padding 仍需对照运行状态。

```text
Attention 归约后：
  Rank10: [A B C D]
  Rank11: [A B C D]

沿 Token 行切分后：
  Rank10: [A B]
  Rank11: [C D]
```

两份输入现在覆盖不同 Token，各自进入 Router 和 EP 分发。如果两个 Rank 都带着完整四行继续发送，同一组 Token 就会被重复派发。若当前调用提供 `input_ids` 或 `input_ids_global`，代码也按同样规则切分它们，保持 ID 与 hidden states 的行对应；未提供 ID 的调用则不会凭空构造一份。

### MoE 后的时序同样明确：先 MoE，再 attn-TP AllGather

MoE 完成后，本例 Rank10 拿到 A、B 的输出，Rank11 拿到 C、D 的输出。`attn_tp_all_gather()` 在这对 Attention TP Rank 之间收齐各段，再沿 Token 维拼接，恢复两边都持有完整四行的布局。

```text
MoE 返回：
  Rank10: [y_A y_B]
  Rank11: [y_C y_D]

attn-TP AllGather 并拼接后：
  Rank10: [y_A y_B y_C y_D]
  Rank11: [y_A y_B y_C y_D]
```

这次 AllGather 恢复的是被拆开的 Token 行，与前面 Attention 输出的 AllReduce 求和不同；它也不是在所有 EP Rank 之间再次执行一遍 Expert Combine。专家贡献已经在 MoE 路径内合到对应 Token，当前交接再把这些 Token 的输出收回到 Attention 布局。

~~~mermaid
flowchart TD
    A[Post-attention hidden rows · replicated inside attn TP group]
    A --> S[Token-row split · by attn_tp_rank]
    S --> M[MoE · Router + EP + experts + combine]
    M --> G[attn_tp_all_gather]
    G --> N[Replicated rows · for next layer]
~~~

## 五、TP32 / DP16 示例与 DSpark 限制

把前面各项放回同一个例子：TP32、DP Attention 开启、DP16、CP1 得到十六个 Attention DP 组，每组 Attention TP2；组五就是 Rank10 与 Rank11。若使用下面这组会约束 EP 的 A2A 后端，并设 MoE DP1，MoE 侧将使用 EP32、MoE TP1。

### 使用当前 A2A MoE backend 时，resolved EP 会被强制成 TP width

固定版本 `_A2A_EP_SPANNING_BACKENDS` 包含 DeepEP、Mooncake、NIXL、Ascend FuseEP、MORI、PPLX、MegaMoE 等，参数后处理对集合内后端返回 `ep_size=tp_size`。这是运行时选择的约束，完整集合和适用版本见源码。[`_a2a_ep_size()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/overrides.py#L1630-L1673)

| 阶段 | 这三十二个 Rank 的分工 |
| --- | --- |
| Attention | DP16 × CP1 × Attention TP2 |
| MoE（上述 A2A 约束、MoE DP1） | MoE DP1 × EP32 × MoE TP1 |

Target 模型的布局交接正好需要处理这两种分工：Attention TP 内先完成输出归约，MoE 前拆 Token 行，专家计算后再用 Attention TP AllGather 恢复。这里解释的是源码如何组织 Target 的 Forward，不能据此推出所有 Draft 或后端组合也支持同样的参数。

### 但是固定 commit 的 DSpark MoE draft 明确要求 `attn_tp == 1`

同一提交的 `DSparkWorkerV2` 在开启 DP Attention、Draft 是 MoE、`attn_tp_size>1` 时直接抛错。本例算出的宽度是二，因此会命中检查，不会因为 Target 里存在布局桥接就自动获得 Draft 支持。[`DSparkWorkerV2`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L159-L175)

错误信息中的 `attn_tp > 1 corrupts the MoE-under-DP all-reduce` 给出了已知风险方向，但它不能代替对每个中间状态的定位。要扩展这条路径，至少要核对 Draft 使用的并行组、Token 行归属、每 Rank 的 Token 计数、接受决策同步，以及 Graph/Verify 对布局的要求。去掉启动检查，只能改变是否允许启动，不能证明生成已经正确。

更完整的 Draft 分析见 [DSpark 并行限制与布局](../distributed/deepseek-v4-dspark-parallel-layout-analysis.md)。这里保留 Target 与 Draft 两条 Forward 的边界，不把限制解释成 HCCL 本身不支持某个宽度。

## 六、Ascend 通信后端的分工

前面说明了每次交换为何存在，具体由谁承接，还需要看设备与后端配置。固定版本的设备映射将普通 NPU distributed backend 设为 HCCL；配置 ZBAL local memory 时可以改为 `zbal`。[`device_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/platforms/device_mixin.py#L86-L93)

SGLang 还会通过 `torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options` 等设置相关进程组，读取默认和 MoE 对应的 HCCL buffer 配置。看到标准 process-group collective 时，可以沿这些组与选项继续定位，但不能据此把所有专用 MoE 后端都解释成独立的 Python All-to-All 调用。[`get_torch_distributed_pg_options()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/parallel_state.py#L90-L105)

Ascend FuseEP 会把分发、跨 Rank 交换、专家计算和合并交给融合入口。即使涉及相同的逻辑职责，Profiler 也未必呈现一组一一对应的 `all_reduce`、`all_gather`、`all_to_all` 窗口。分析时间线时，要将 Python 入口、process group、后端和实际事件对应起来，不能只凭事件数量推断通信被省掉。固定版本的官方部署示例也给出了 `--tp-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend deepep`，它对应自己的宽度组合，不是本文 TP32/DP16 例子的运行验证。[官方示例](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx#L202-L225)

### TP：Tensor-shard ownership

追踪的是一份权重或 Tensor 的哪部分由谁计算。对照列并行、行并行与输出归约时，应说明交换前的数据是分片、局部贡献还是完整结果。

### DP Attention：Token ownership

追踪的是哪些请求和 Token 行由哪个 Attention 组处理。是否开启 DP Attention 决定有效宽度，后面的 Token 行拆分又决定 MoE 前各 Rank 实际提交哪些输入。

### EP：Expert ownership

追踪的是目标专家权重位于哪里，当前 Token 怎样过去、算完又怎样回来。这里应把专家分发与 Attention 布局恢复分别看清，不能因为都在搬数据就混成一次通信。

再回到 Rank10、Rank11 的四行例子：先归约相同 Token 的 Attention 贡献，再拆开 Token 行进入 MoE，最后收齐不同 Token 的输出。这三次动作之所以都需要，正是因为它们处理的数据关系不同。读到某个 collective 时，先确认当前行属于谁、数值表示什么，再看下一阶段需要怎样的排列。

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
