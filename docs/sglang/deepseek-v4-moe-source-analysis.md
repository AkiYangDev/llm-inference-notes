# DeepSeek-V4 MoE：Token 从路由到专家计算，再回到原位置

Attention 输出一行 hidden states 后，MoE 要先决定这行数据交给哪些专家，再把它送到持有专家权重的 Rank，最后把多个专家的结果合并回原 Token。

本文用四个 Rank、八个 Expert 的教学例子走过 Router、Dispatch、专家计算和 Combine，再对照 DeepSeek-V4 的实现。DeepEP 暴露这些阶段的独立接口，Ascend FuseEP 则会把其中多项工作融合到同一执行路径。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线分析 DeepSeek-V4 target model 的普通 MoE forward。具体 checkpoint 可以覆盖配置类默认值；文中的 4 Rank / 8 Expert 例子只用于解释通信，不代表 DeepSeek-V4 的真实规模。

## 一、DeepSeek-V4 复用的 MoE 实现

`DeepseekV4DecoderLayer.forward()` 在 Attention 和 post-attention mHC / norm 之后调用：

~~~python
hidden_states = self._run_moe_ffn_dp_sync(
    hidden_states,
    forward_batch,
    input_ids=input_ids,
    input_ids_global=input_ids_global,
)
~~~

源码见 [`DeepseekV4DecoderLayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3006-L3189) 和 [`_run_moe_ffn_dp_sync()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)。

DeepSeek-V4 没有单独重写一整套 `DeepseekV4MoE`，而是构造：

~~~python
self.mlp = deepseek_v2.DeepseekV2MoE(
    config=config,
    ...
    is_deepseek_v4=True,
)
~~~

固定入口：[DeepSeek-V4 Decoder Layer 构造](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2643-L2654)。

所以 `deepseek_v2.py` 在这里已经是 DeepSeek 系列共享 MoE runtime 的一部分。

### 先把真实默认配置和 toy example 分开

固定 commit 中，`DeepSeekV4Config` 类默认值是：

| 配置 | 默认值 |
| --- | ---: |
| `n_routed_experts` | 256 |
| `n_shared_experts` | 1 |
| `num_experts_per_tok` | 6 |
| `n_hash_layers` | 3 |
| `scoring_func` | `sqrtsoftplus` |

源码：[DeepSeekV4Config](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/configs/deepseek_v4.py)。这些是配置类默认值，checkpoint 可以覆盖。

后文使用的：

~~~text
TOY EXAMPLE ONLY
4 Ranks
8 routed experts
Top-K = 2
~~~

只用于画清通信拓扑，不是 DeepSeek-V4 的真实配置。

### Shared expert 不保证永远是一条独立分支

固定配置默认有 1 个 shared expert，但 runtime 先根据 construction-time fusion decision 决定：

~~~python
self.num_fused_shared_experts = (
    0 if is_shared_experts_fusion_disabled()
    else n_shared_experts
)
~~~

源码：[DeepseekV2MoE](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v2.py#L560-L850)。

因此模型语义是 routed experts + shared expert；runtime 可能把 shared expert 融合进 FusedMoE / TopK expert layout，也可能在 fusion disabled 时构造单独的 `shared_experts = DeepseekV2MLP(...)` 再合并。

DeepEP-family / MegaMoE 的一部分 fused-shared 路径使用 per-rank physical shared slots；Ascend FuseEP 不属于这个 per-rank helper。[源码 helper](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/utils.py#L629-L648)

## 二、Router、TopK 与 HashTopK

普通 learned-router 路径先通过 `MoEGate`：

~~~text
hidden_states [T,H]
      ↓ Router linear
router_logits [T,E]
      ↓ TopK
topk_ids / topk_weights [T,K]
~~~

Router 权重形状是 `[n_routed_experts, hidden_size]`。源码：[MoEGate](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v2.py#L458-L555)。

标准 TopK 输出是 `topk_weights / topk_ids / router_logits`。[StandardTopKOutput](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/topk.py#L313-L335)

DeepSeek-V4 还覆盖普通 TopK 的 grouped-routing 默认：构造时设置 `use_grouped_topk=False`，因此不能把 DeepSeek-V3 的 grouped top-k 规则直接照搬过来。[源码](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v2.py#L685-L727)

### 固定默认下，target model 前 3 层走 HashTopK

固定配置默认 `n_hash_layers = 3`，而 MoE 构造条件是：

~~~python
self.is_hash = (
    layer_id < n_hash_layers
    and not (is_deepseek_v4 and is_nextn)
)
~~~

因此在 checkpoint 未覆盖该字段时，target model 的 layer 0、1、2 会使用 `HashTopK`。

`HashTopK` 有一张 `tid2eid` token-id → expert-id 映射表：

~~~text
input_ids → tid2eid lookup → expert ids
router_logits → 对这些 expert 计算 routing weights
~~~

源码：[HashTopK](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/hash_topk.py)。

前 3 层的 HashTopK 与后续 learned TopK 一旦生成 `topk_ids / topk_weights`，后面的 Expert Parallel runtime 就重新汇合。

另外，默认 `num_experts_per_tok = 6` 描述的是 routed experts 数量；如果 shared-expert fusion 开启，内部 MoE layout 还可能额外包含 shared slot。看到内部宽度 7，不代表 routed Top-K 从 6 变成了 7。

## 三、从专家编号找到持有权重的 Rank

`FusedMoE` 记录 `moe_ep_size / moe_ep_rank`，并按 EP storage size 计算每 Rank 的 routed expert 数：

~~~python
self._num_local_routed = (
    self._num_global_routed
    // storage_ep_size
)
~~~

源码：[FusedMoE](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L327-L409)。

于是 Global Expert Set 被分片到多个 EP Rank。

用 toy topology：

~~~text
Rank0: E0 E1
Rank1: E2 E3
Rank2: E4 E5
Rank3: E6 E7
~~~

Token A 来自 Rank0，TopK 选择：

~~~text
E1 weight=0.7
E5 weight=0.3
~~~

则 `(A,E1)` 留在 Rank0，`(A,E5)` 必须发送到 Rank2。

一个 Token 选 K 个 routed experts，可以理解成 K 个 token-expert assignments。真实 backend 会做 packing / sorting / quantization / batched communication，不会逐 Token 发 Python 消息。

当 EP group 中每个 Rank 都可能向其他 Rank 的 Expert 发送 Token 时，形成典型的多源到多目的数据交换，所以通常称为 All-to-All。但这里的 All-to-All 是**通信语义**，不是对具体 backend kernel 名的承诺。

## 四、Dispatch、专家计算与 Combine

通用 `FusedMoE.forward_impl()` 很清楚：

~~~python
dispatch_output = self._dispatch_with_pre_quant(
    hidden_states, topk_output, pre_quant_input
)

combine_input = self.run_moe_core(
    dispatch_output=dispatch_output
)

final_hidden_states = self.dispatcher.combine(
    combine_input=combine_input
)
~~~

源码：[FusedMoE.forward_impl](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1562-L1599)。

Dispatch 负责按目的 Rank / Expert 重排 Token；MoE Core 在 expert-local layout 上做 Expert compute；Combine 把结果送回 Token owner 并恢复顺序与 routing weight。

继续前面的 Token A 示例：它以 0.7、0.3 的路由权重选择 E1、E5，两个专家分属 Rank0、Rank2。忽略 shared expert，教学表达式为 `y_A = 0.7 × E1(x_A) + 0.3 × E5(x_A)`。Combine 需要把这两份贡献合到 A 的输出行，而不是把两份结果当成两个新 Token 追加出去。具体 backend 可以提前或融合应用权重，这个式子描述最终语义。

Expert 的模型语义仍是：

~~~text
w1 / gate
w3 / up
    ↓
SwiGLU
    ↓
w2 / down
~~~

即 `W2(SiLU(W1x) ⊙ W3x)`。SGLang 将 gate/up 权重融合存为 `w13_weight`，down projection 使用 `w2_weight`。[权重映射](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1649-L1738)

Expert core 的统一入口是：

~~~python
return self.quant_method.apply(
    layer=self,
    dispatch_output=dispatch_output,
)
~~~

源码：[run_moe_core](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1642-L1647)。具体 grouped / fused GEMM kernel 取决于 quant method 与 MoE runner backend。

DeepEP-family 的 Python 边界更容易观察：

~~~python
dispatch_output = self.dispatcher.dispatch(...)
combine_input = self.run_moe_core(dispatch_output)
return self.dispatcher.combine(combine_input)
~~~

源码：[DeepEPMoE](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/ep_moe/layer.py#L252-L329)。

## 五、Ascend FuseEP 怎样融合执行阶段

选择 `--moe-a2a-backend ascend_fuseep` 时，`FusedMoE.forward()` 直接进入：

~~~python
return forward_fuseep(
    self, hidden_states, topk_output
)
~~~

源码：[FusedMoE.forward](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1521-L1560)。

`forward_fuseep()` 所在文件直接把自己描述成 Ascend FuseEP fused dispatch+GEMM+combine forward path，并最终调用：

~~~python
hidden_states, _ = buf.fused_deep_moe(
    hidden_states,
    topk_idx=topk_output.topk_ids,
    topk_weights=topk_output.topk_weights,
    gmm1_permuted_weight=layer.w13_weight,
    gmm2_weight=layer.w2_weight,
    ...
)
~~~

源码：[forward_fuseep](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/moe/fuseep.py)。

所以 Ascend Python 视角是：

`fused_deep_moe` 对上层暴露一次调用，内部承担多个阶段的职责：

| 内部职责 | 数据变化 |
| --- | --- |
| Dispatch 与跨 Rank 交换 | 将 Token 送到持有目标 Expert 的位置 |
| Expert 计算 | 经两次 GEMM 与中间激活得到专家输出 |
| Combine | 将贡献送回并合并到原 Token 行 |

这是语义拆解，Profiler 中未必出现对应数量的独立算子或 All-to-All 窗口。沿融合入口分析性能时，应使用该后端实际暴露的事件与指标。

## 六、EP 宽度与后端约束

固定 commit 的参数解析器定义了一组 `_A2A_EP_SPANNING_BACKENDS`：

~~~text
megamoe
deepep
deepep_v2
mooncake
nixl
ascend_fuseep
flashinfer
flashinfer_megamoe
mori
pplx
~~~

当使用这些 backend 时，解析阶段会把 `ep_size` 调整为 `tp_size`：

~~~python
if view.moe_a2a_backend in _A2A_EP_SPANNING_BACKENDS:
    return {"ep_size": view.tp_size}
~~~

源码：[A2A EP size override](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/overrides.py#L1630-L1673)。

所以更准确的结论是：

> **这是固定版本里一组 A2A backend 的 runtime / topology 约束，不是 Expert Parallel 概念本身要求 EP 必须等于 TP。**

不能写成 `SGLang 永远要求 EP = TP`。`none` backend 等路径使用不同的数据交换机制；官方 EP 文档也把 `none` 描述为 All-Reduce / All-Gather based dispatch。[官方 EP 文档](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/advanced_features/expert_parallelism.mdx)

这条限制背后其实是 Token ownership 与 Expert ownership 的映射问题：Attention 侧有 TP / attention TP / DP，MoE 侧有 EP。如果这些 group 不重合，进入 MoE 前后的 gather / scatter、Expert owner、combine destination 和 TP partial-output reduction 都会更复杂。

所以 `_run_moe_ffn_dp_sync()` 中才会同时出现 DP gather、reduce-scatter、A2A scatter、MoE 和 DP scatter / all-gather 等通信。[源码](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915)

最终，一次 Token 的 MoE 推理可以拆成三个不同问题：

~~~text
Router / TopK:
这个 Token 依赖哪些 Expert？

Expert Parallel:
这些 Expert 在哪些 Rank，Token 怎么过去、结果怎么回来？

MoE Core:
Expert-local 的矩阵乘怎样高效执行？
~~~

这三层分开以后，DeepSeek-V4 的 MoE runtime 就是一条清楚的数据链：

~~~text
Attention output
      ↓
Router / HashTopK
      ↓
topk_ids / topk_weights
      ↓
Expert ownership
      ↓
EP dispatch
      ↓
Expert-local compute
      ↓
combine
      ↓
MoE output
      ↓
Next Decoder Layer
~~~

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| DeepSeek-V4 创建 MoE | [Decoder Layer init](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2643-L2654) |
| Decoder Layer 进入 MoE | [Decoder Layer forward](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3006-L3189) |
| DP / TP / MoE 边界 | [_run_moe_ffn_dp_sync](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3748-L3915) |
| V4 默认 MoE / Hash 配置 | [DeepSeekV4Config](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/configs/deepseek_v4.py) |
| MoE 主类 / shared fusion | [DeepseekV2MoE](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v2.py#L560-L850) |
| Router / Gate | [MoEGate](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v2.py#L458-L555) |
| HashTopK | [hash_topk.py](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/hash_topk.py) |
| FusedMoE / Expert ownership | [FusedMoE](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L294-L572) |
| Dispatch → Core → Combine | [FusedMoE.forward_impl](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1562-L1599) |
| DeepEP EP 实现 | [DeepEPMoE](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/ep_moe/layer.py#L63-L365) |
| Ascend FuseEP | [fuseep.py](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/moe/fuseep.py) |
| shared-slot backend 判断 | [moe utils](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/utils.py#L629-L648) |
| A2A backend 强制 EP=TP | [overrides.py](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/overrides.py#L1630-L1673) |
