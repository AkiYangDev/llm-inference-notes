# DeepSeek-V4 MTP：NextN 怎样逐步生成候选

DeepSeek-V4 的 NextN 路径先通过 Draft Extend 产生第一枚候选，再用轻量 forward 逐步生成后续候选。它仍保留自回归依赖，只是把连续猜测交给较轻的 Draft，随后由 Target 一次验证。

下面以三个 Draft 候选为例，追踪 Target hidden、边界 Token、Draft Extend 和 NextN rollout 的衔接，算清一轮里有哪些输入、多少次 forward，以及哪些缓存要留给下一轮。它与 DSpark 的块式提议是不同的 Draft 实现。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线讨论 DeepSeek-V4 bundled MTP / NextN draft，在 SGLang 中由 EAGLE-family worker 驱动。它与 DSpark 是不同 Draft 实现。固定版本的 V4 EAGLE 配置限定 `topk=1`；通用 Runtime 的多分支机制见 Draft Tree 专题。

</details>

## 一、MTP、NextN 与 EAGLE 的职责

MTP、NextN 和 EAGLE 出现在同一条路径上，分别指向权重能力、模型实现和运行时算法：

| 名字 | 在本文中的层次 | 当前 DeepSeek-V4 路径里的含义 |
| --- | --- | --- |
| MTP | checkpoint / model capability | DeepSeek-V4 checkpoint 中携带的未来 Token 预测权重，源码可见 `mtp.*` 权重命名 |
| NextN | SGLang draft architecture | SGLang 把这组权重装进 `DeepseekV4ForCausalLMNextN`，当前只物化 1 个 NextN decoder layer |
| EAGLE | speculative runtime algorithm | SGLang 用 `EAGLEWorkerV2 / EagleDraftWorker` 驱动 Draft Extend、多步 Draft、Target Verify 与状态推进 |

因此，在当前 DeepSeek-V4 场景里可以写成：

~~~text
DeepSeek-V4 MTP weights
        ↓
DeepseekV4ForCausalLMNextN
        ↓
EAGLE runtime
        ↓
Draft candidate chain
        ↓
Target Verify
~~~

其中 MTP 描述训练目标，NextN 是 Draft 模型实现，EAGLE Runtime 负责候选生成与验证流程。

### 为什么 CLI 写 NEXTN，运行时最后却看到 EAGLE

SGLang 的参数解析会把普通 `NEXTN` alias 归一化到 `EAGLE`：

~~~python
if speculative_algorithm == "NEXTN"         or speculative_algorithm == "EAGLE":
    ...
    return "EAGLE"
~~~

固定源码：

[`_resolve_speculative_algorithm_alias()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L74-L111)

因此：`--speculative-algorithm NEXTN`

对这条常规路径来说是用户侧命名；进入 Runtime 后，`SpeculativeAlgorithm` 使用的是 `EAGLE`。

这里还有一个模型相关例外：Gemma4 assistant draft 可以把 NEXTN/EAGLE 继续解析成 `FROZEN_KV_MTP`。这恰好说明“NextN”和“EAGLE”本来也不是全局同义词。

### DeepSeek-V4 Draft architecture 为什么会变成 NextN

DeepSeek-V4 checkpoint 可能同时带不同 speculative head。当前 `ModelConfig` 会根据最终选择的算法决定 Draft architecture。

如果明确使用 DSpark：`DeepseekV4ForCausalLMDSpark`

而普通 EAGLE / NextN 路径则改写成：

~~~python
self.hf_config.architectures[0] = "DeepseekV4ForCausalLMNextN"
self.hf_config.num_nextn_predict_layers = 1
~~~

固定源码：

[`model_config.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/configs/model_config.py#L878-L888)

权重加载也明确识别 `mtp.*`，并把对应权重映射到 NextN architecture。当前实现还要求：

~~~python
assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
~~~

固定源码：

[`DeepseekV4ForCausalLM.load_weights()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5363-L5377)

所以这篇文章研究的是：

> **DeepSeek-V4 checkpoint 中的 MTP 权重，如何被 SGLang 物化成一个单层 NextN Draft Model，并由 EAGLE Runtime 反复调用，形成一条未来 Token 候选链。**

## 二、NextN 怎样组合 Token 与 Target hidden

NextN 的模型入口是：

[`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py)

核心类：

~~~python
DeepseekV4ModelNextN
DeepseekV4ForCausalLMNextN
~~~

它和“单独跑一个小 LLM 当 Draft Model”最大的区别，是 NextN 不需要只依赖自己的 Token 历史重新理解上下文。

它直接使用 Target Model 已经算好的 hidden representation。

一边是当前 candidate token：

~~~python
hidden_states = self.embed_tokens(input_ids)

e_proj_hidden_states, _ = self.e_proj(
    self.enorm(hidden_states)
)
~~~

另一边是 Target hidden：

~~~python
hc_flat = forward_batch.spec_info.hidden_states.view(
    n_tokens * self.hc_mult,
    d,
)

h_proj_out, _ = self.h_proj(
    self.hnorm(hc_flat)
)
~~~

最后融合：

~~~python
hidden_states = (
    e_proj_hidden_states[:, None, :]
    + h_proj_hidden_states
)
~~~

可以先把它看成：

~~~mermaid
flowchart TD
    T[Current candidate token]
    H[Target confirmed hidden]

    T --> E[Embedding]
    E --> EP[e_proj]

    H --> HP[h_proj]

    EP --> F[Feature fusion]
    HP --> F

    F --> D[1-layer DeepSeek-V4 NextN decoder]
    D --> HC[hc_head + norm]
    HC --> LM[LM Head]
    LM --> L[Next candidate logits]
~~~

NextN 使用 Target 提供的 Hidden Feature，与输入 Token 一起生成候选。

它更接近：

> **Target 已经把上下文压成高质量 feature，NextN 在这个 feature 上做更便宜的未来 rollout。**

### Target 到 Draft 的多路 Hidden Feature

DeepSeek-V4 有 mHC multi-stream hidden。

Target 最后一层结束以后，源码先保留：

~~~python
pre_hc_head = hidden_states.flatten(1)
~~~

如果内部 tensor 是：`[T, hc_mult, H]`

这里就变成：`[T, hc_mult × H]`

固定源码：

[`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4808-L4839)

随后 Target 的 LogitsProcessor 在 speculative hidden capture 时会同时看到普通 normalized hidden 和 `hidden_states_before_norm=pre_hc_head`。只要 before-norm hidden 存在，`_get_hidden_states_to_store()` 明确优先把它放进 `logits_output.hidden_states`：

~~~python
if hidden_states_to_store_before_norm is not None:
    hidden_states_to_store = hidden_states_to_store_before_norm
~~~

固定源码：

[`LogitsProcessor._get_hidden_states_to_store()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L785-L836)

所以对当前 DeepSeek-V4 EAGLE/NextN 路径，Target→Draft 边界传递的核心 feature shape 是：`[selected_token_rows, hc_mult × hidden_size]`

而不是普通的：`[selected_token_rows, hidden_size]`

NextN 收到后再：

~~~python
spec_info.hidden_states.view(
    n_tokens * hc_mult,
    hidden_size,
)
~~~

经过 `h_proj` 后恢复成：`[n_tokens, hc_mult, hidden_size]`

当前 Token Embedding 经 `e_proj` 是：`[n_tokens, hidden_size]`

再变成：`[n_tokens, 1, hidden_size]`

与 Target feature 广播相加。

如果用一个教学 shape：

~~~text
n_tokens = 2
hidden_size = 8
hc_mult = M
~~~

那么：

~~~text
Target captured hidden
[2, M*8]

        ↓ view

[2, M, 8]


Token embedding
[2, 8]

        ↓ e_proj + unsqueeze

[2, 1, 8]


广播相加

[2, M, 8]
~~~

上面的形状对应当前 DeepSeek-V4 NextN 实际接收的 Draft feature。

## 三、Draft Extend 产生第一枚候选

生成第一枚 Draft 候选前，先看这一轮从哪里起步。上一轮 Target Verify 或 Prefill 完成后，会得到一枚 **Target 已确认的 Token**。

在 EAGLE 数据结构里，它叫：`bonus_tokens`

`EagleDraftInput` 的源码注释也把它定义为：

~~~text
Per-req bonus token
(the "+1" target prediction at end of each accept chain)
~~~

固定源码：

[`EagleDraftInput`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_info.py#L143-L180)

它是：`Target confirmed boundary token`

不是：`Draft candidate 1`

### Target 状态先进入 Draft Extend

Prefill 后，EAGLE 会运行：`_draft_extend_for_prefill()`

Verify 后，则运行：`_draft_extend_for_decode()`

以 Decode 稳态为例，Draft Extend 输入里有：

~~~python
hidden_states=batch_result.logits_output.hidden_states
~~~

也就是刚刚 Target Verify 捕获的 mHC hidden。

同时，NextN 的 input token 来自已经接受的 Target token path；最终边界 token 会保留为下一轮的 `bonus_tokens`。

Draft Extend 执行一次：

~~~python
draft_logits_output = self.draft_runner.forward(
    forward_batch
).logits_output
~~~

然后从 logits 得到：

~~~text
ret_topk_p
ret_topk_index
ret_hidden_states
~~~

并写回下一轮：

~~~python
next_draft_input.topk_p = ret_topk_p
next_draft_input.topk_index = ret_topk_index
next_draft_input.hidden_states = ret_hidden_states
~~~

固定源码：

[`EagleDraftWorker._draft_extend_for_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L1127-L1276)

这里的：`topk_index`

才是下一轮 `draft_forward()` 开始时已经准备好的第一枚 Draft candidate。

因此准确时序是：

~~~mermaid
flowchart TD
    T[Target Verify]
    T --> B[Target bonus / confirmed token]
    T --> H[Target mHC hidden]

    B --> E[Draft Extend]
    H --> E

    E --> C1[Draft candidate 1]
    E --> DH[Draft hidden seed]

    C1 --> F[draft_forward]
    DH --> F
~~~

按输入与产出对应起来：

> **bonus 是 Target→Draft 的边界输入；candidate1 是 Draft Extend 对这个边界状态做一次 NextN forward 后得到的 proposal。**

### 为什么 Verify Window 里又会看到 bonus

后面构造 Target Verify tree 时，源码会再把 bonus prepend 到 Draft candidates：

~~~python
draft_tokens = torch.cat(
    (bonus_tokens.unsqueeze(1), draft_tokens),
    dim=1,
)
~~~

固定源码：

[`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py)

这一步是为了构造 Target Verify 的因果窗口，不意味着 bonus 是 Draft 生成的 Token。

## 四、算清候选数与 NextN Forward 次数

这一组数字如果不拆开，很容易把 MTP 的执行成本理解错。

假设：

~~~text
speculative_num_steps = 3
speculative_eagle_topk = 1
~~~

固定源码会要求 topk=1 时：

~~~python
speculative_num_draft_tokens
=
speculative_num_steps + 1
~~~

~~~text
num_steps = 3
num_draft_tokens = 4
~~~

固定源码：

[`_handle_eagle_family()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L1067-L1078)

其中 1 个位置留给旧 bonus，其余 3 个位置是 Draft 候选。

### `draft_forward()` 里只有 3 枚 Draft candidate

进入 `EagleDraftWorker.draft_forward()` 时：`topk_index`

已经由前一阶段 Draft Extend 准备好。

对 topk=1，它就是：`candidate1`

源码随后进入：

~~~python
for i in range(self.speculative_num_steps):
~~~

当：`i = 0`

先把已有 `topk_index` 记成 candidate1，然后运行一次 NextN：

~~~text
candidate1
   ↓
NextN forward
   ↓
candidate2
~~~

`i = 1`：

~~~text
candidate2
   ↓
NextN forward
   ↓
candidate3
~~~

`i = 2` 时先记录 candidate3，随后命中：

~~~python
if i == self.speculative_num_steps - 1:
    break
~~~

~~~text
draft_forward 输出的 Draft candidates = 3

draft_forward 内部真正调用
draft_runner.forward() = 2 次
~~~

固定源码：

[`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L778-L951)

但如果从“上一轮 Target 刚确认完状态”开始计算 candidate-generation 成本，还需要把前面的：`1 × Draft Extend forward`

算进去。

因此稳态下，从上一轮 Target Verify 结束，到下一轮 3 枚 candidate 全部准备完成，NextN 模型总共执行：

~~~text
1 × Draft Extend
+
2 × Draft Decode Forward
=
3 × NextN forward
~~~

只是它们跨在两个 Runtime 阶段里，不都写在 `draft_forward()` 里。

### 为什么 Target Verify 是 4 个 Token

`draft_forward()` 得到：`[draft1, draft2, draft3]`

构建 Verify Input 时，再 prepend：`bonus`

最终：

~~~text
Verify Window:

[bonus, draft1, draft2, draft3]
~~~

因此，在 topk=1 线性链下，窗口宽度是 `4`，对应 `speculative_num_draft_tokens = 4`。候选数量与窗口宽度相差一枚边界 Token，可以按下表区分：

| 概念 | `num_steps=3, topk=1` | 含义 |
| --- | ---: | --- |
| Target bonus token | 1 | 上一轮 Target 已确认的边界 Token |
| Draft candidates | 3 | NextN 提出的未来候选 |
| `draft_forward()` 内 NextN forward | 2 | candidate1 已由 Draft Extend 提前产生 |
| 从上一轮 Target 状态算起的 NextN forward 总数 | 3 | 1 次 Draft Extend + 2 次 Draft Decode |
| Target Verify window | 4 | `[bonus, d1, d2, d3]` |

`num_draft_tokens` 描述整个 Verify 窗口，计数时包含旧 bonus 所占的位置。

在 EAGLE 里，它更接近：

> **这一轮 Target Verify 需要处理的 speculative tree/window 宽度。**

## 五、NextN 的 SWA 缓存

DeepSeek-V4 Target 本身可能按层使用：

~~~text
SWA
C4
C128
~~~

但 NextN Draft Layer 在构造时显式写了：

~~~python
COMPRESS_RATIO_NEXTN_LAYER = 0
~~~

并传给：

~~~python
DeepseekV4DecoderLayer(
    ...,
    is_nextn=True,
    compress_ratio_override=0,
)
~~~

固定源码：

[`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py)

`compress_ratio=0` 选择 SWA 路径：

> **这个 NextN Layer 不使用 C4/C128 compressed extra-cache path，而使用 DSV4 的 SWA path。**

### Memory Pool 侧：ratio 0 没有 compressed pool，但 SWA pool 仍然存在

`DeepSeekV4TokenToKVPool` 始终给 stage 建立 SWA storage。

而 compressed pools 只从：

~~~text
4
128
~~~

这类实际压缩 ratio 中创建。

Layer mapping 对 ratio 0 会得到：

~~~text
compress_ratio = 0
compress_kv_pool = None
~~~

但这并不会影响它使用 SWA cache。

固定源码：

[`DeepSeekV4TokenToKVPool._init_compressed_layer_mapping()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1637-L1656)

### Ascend Backend 侧更直接：ratio 0 就进入 `_forward_swa()`

NPU DSV4 Attention 的主分支是：

~~~python
if compress_ratio == 0:
    return self._forward_swa(
        q, layer, forward_batch, attn_sink
    )
~~~

而 `_forward_swa()` 读取：

~~~python
ori_kv = pool.get_swa_buffer(layer.layer_id)
~~~

固定源码：

[`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2116)

所以对当前 Ascend DeepSeek-V4 NextN path：

~~~text
compress_ratio = 0
        ↓
SWA KV
        ↓
no C4 extra cache
no C128 extra cache
~~~

这是源码事实。

还有一个额外佐证：`DeepseekV4AscendAttnBackend` 如果发现自己运行在 draft worker，会把 Target checkpoint 中原本的 compression-ratio list 清空：

~~~python
if model_runner.is_draft_worker:
    self._dsv4_compress_ratios = type(
        hf.compress_ratios
    )()
~~~

因此 Draft backend 不会按 Target 的 C4/C128 层表去建立 compressed metadata。

### NextN 缓存配置的适用范围

可以写：

> **当前 DeepSeek-V4 NextN Draft Layer 的 attention/cache path 是 SWA-only。**

不要写：

> **DeepSeek-V4 开启 MTP 后就只使用 SWA。**

Target Model 仍然是 Target 自己的：`SWA / C4 / C128`

混合结构。

“SWA-only”只是在描述：`NextN Draft Layer`

## 六、Ascend 上的模型、Worker 与 Backend

到这里可以把理论和你关心的 Ascend 部署接起来。

固定版本的 DeepSeek-V4-Flash Ascend 官方文档给出的 speculative 配置就是：

~~~bash
--device npu
--attention-backend dsv4
--speculative-algorithm EAGLE
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
~~~

模型权重示例也是：`DeepSeek-V4-Flash-w8a8-mtp`

固定入口：

[DeepSeek-V4-Flash Ascend Tutorial](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx)

也就是说，Ascend 上官方生产示例走的是：

~~~text
MTP checkpoint
    ↓
EAGLE algorithm
    ↓
DeepseekV4ForCausalLMNextN
~~~

而不是 DSpark。

### Draft Model 层

在 EAGLE 模式下，DeepSeek-V4 draft architecture 是：`DeepseekV4ForCausalLMNextN`

内部只有当前受支持的：`1 × NextN DeepseekV4DecoderLayer`

ModelSlim 权重路径也有专门处理；`deepseek_v4_nextn.py` 在 ModelSlim 配置下把 decoder prefix 对到：`mtp.0`

所以 NPU W8A8 MTP checkpoint 并不是先转成 DSpark model 再运行。

### Draft Decode Attention 层

`DraftBackendFactory` 根据：

~~~text
attention backend = dsv4
device = npu
~~~

创建：`DeepseekV4AscendMultiStepDraftBackend`

固定源码：

[`DraftBackendFactory._create_dsv4_decode_backend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L426-L453)

这个 multi-step container 会为每个 speculative step 建立一个：`DeepseekV4AscendAttnBackend`

并维护各 step 自己的 KV location / metadata view。

固定源码：

[`DeepseekV4AscendMultiStepDraftBackend`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2354-L2512)

### Draft Extend Attention 层

Draft Extend 仍选择：`dsv4`

但这里不是 multi-step container，而是直接走注册表中的 NPU DSV4 backend：

~~~python
ATTENTION_BACKENDS["dsv4"](draft_model_runner)
~~~

而 `attention_registry.py` 在 NPU 上把 `dsv4` 映射成：`DeepseekV4AscendAttnBackend`

固定源码：

- [`DraftBackendFactory._create_dsv4_prefill_backend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L574-L601)
- [`attention_registry.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/attention_registry.py#L166-L192)

所以 Ascend 执行层级可以画成：

~~~mermaid
flowchart TD
    CKPT[DeepSeek-V4 MTP checkpoint]

    CKPT --> N[DeepseekV4ForCausalLMNextN]
    N --> E[EAGLEWorkerV2 / EagleDraftWorker]

    E --> X[Draft Extend]
    X --> DB1[DeepseekV4AscendAttnBackend]

    E --> M[Multi-step Draft Decode]
    M --> DB2[DeepseekV4AscendMultiStepDraftBackend]
    DB2 --> S0[Step 0: DeepseekV4AscendAttnBackend]
    DB2 --> S1[Step 1: DeepseekV4AscendAttnBackend]

    E --> V[Full Target Verify]
~~~

如果启用 NPU graph，EAGLE 还有对应的 NPU Draft / Draft-Extend graph runner；那是执行优化层，不改变上面的算法状态机。

这几层共同组成 Ascend 上的 MTP 执行路径：

> **MTP 权重被装进单层 NextN Draft Model，EAGLE Worker 用 Ascend DSV4 Draft Extend + Multi-Step Draft Backend 迭代构造候选，再交给完整 DeepSeek-V4 Target 做 Verify。**

## 七、少跑 Target 的收益与 Draft 成本

NextN 仍按候选顺序推进：Draft Extend 给出第一枚候选，后续 rollout 再生成下一枚。收益来自用较轻的 Draft 替代一部分昂贵的 Target Decode，并由 Target 对候选窗口统一验证。

在相同批次与资源条件下，设一轮实际推进 `A` 个输出 Token。普通 Decode 的对应成本约为 `A × C_target`；投机周期则包含 Draft Extend、rollout、Target Verify、Accept 与状态整理。后者总耗时更低时，才得到这一轮的生成加速。成本按实际执行的候选工作计算，产出按最终接受并提交的 Token 计算。

| 一轮中的工作 | 作用 |
| --- | --- |
| Draft Extend | 消费已确认状态，准备首个候选与后续 Draft 输入 |
| NextN rollout | 逐步产生剩余候选 |
| Target Verify | 为候选窗口计算预测与 hidden states |
| Accept 与提交 | 确定有效输出，整理已计算的输入状态，并保留下一轮 bonus |

接受率下降时，后半段 rollout 可能成为无效工作；窗口过宽时，Verify 与缓存整理也会变贵。读懂这条执行链之后，再结合 [投机解码性能分析](../performance/deepseek-v4-speculative-performance-analysis.md) 选择要测量的窗口和并发。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| NEXTN / EAGLE alias | [`_resolve_speculative_algorithm_alias()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L74-L111) |
| DeepSeek-V4 Draft architecture 选择 | [`model_config.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/configs/model_config.py#L878-L888) |
| DeepSeek-V4 NextN Model | [`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py) |
| MTP 权重映射 / NextN load | [`DeepseekV4ForCausalLM.load_weights()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5363-L5524) |
| Target mHC flatten | [`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4808-L4839) |
| before-norm hidden 捕获规则 | [`LogitsProcessor._get_hidden_states_to_store()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L785-L836) |
| EAGLE Draft State | [`EagleDraftInput`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_info.py#L143-L180) |
| Multi-step Draft Loop | [`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L778-L951) |
| Prefill → Draft Extend | [`_draft_extend_for_prefill()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L988-L1113) |
| Verify → Draft Extend | [`_draft_extend_for_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L1127-L1276) |
| bonus + draft → verify tree | [`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L154-L289) |
| Target Verify | [`run_eagle_verify()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L462-L675) |
| NextN ratio 0 / SWA path | [`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2116) |
| Ascend DSV4 Multi-Step Draft | [`DeepseekV4AscendMultiStepDraftBackend`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2354-L2512) |
| Draft Backend Factory | [`draft_utils.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L84-L170) |
| Ascend DSV4 backend registry | [`attention_registry.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/attention_registry.py#L166-L192) |
| DeepSeek-V4 Ascend MTP 配置 | [`deepseek_v4_flash.mdx`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx) |
