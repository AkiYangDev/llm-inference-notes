# DeepSeek-V4 MTP：NextN 怎样逐步生成候选

DeepSeek-V4 的 NextN 不会一次并行给出所有未来 token。Draft Extend 先准备第一枚候选与 hidden seed，后面的候选仍沿自回归依赖逐步生成；变化在于这些连续计算交给单层 Draft，再由完整 Target 一次验证候选窗口。

下面以三个 Draft 候选为例，追踪边界 token、Target hidden 与 NextN 自己的 hidden 怎样交接。重点是算清第一枚候选从哪里来、一次循环实际调用几次模型，以及输出已经确认但 KV 尚未写入时怎样继续。它与 DSpark 的块式提议是不同的 Draft 实现，不能用同一种候选生成成本来估算。

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

这三个名字可以同时出现在一条执行链中：

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

因此，在这条常规路径上指定 `--speculative-algorithm NEXTN`，进入 Runtime 后仍会看到 `SpeculativeAlgorithm.EAGLE`。这不是运行时偷偷换了一套 Draft 权重，而是参数别名解析与模型架构选择分属不同层。以下片段用于说明分支与数据关系，均省略了周围代码，不是可独立执行的完整程序。

这里还有一个模型相关例外：Gemma4 assistant draft 可以把 NEXTN/EAGLE 继续解析成 `FROZEN_KV_MTP`。这恰好说明“NextN”和“EAGLE”本来也不是全局同义词。

### DeepSeek-V4 Draft architecture 为什么会变成 NextN

DeepSeek-V4 checkpoint 可能同时带不同 speculative head。当前 `ModelConfig` 会根据最终选择的算法决定 Draft architecture。

明确选择 DSpark 时，Draft architecture 是 `DeepseekV4ForCausalLMDSpark`；EAGLE / NextN 路径则改成下面的 NextN 类，并设置预测层数：

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

固定版本只物化一个 NextN decoder layer，并不意味着只能提出一枚候选。EAGLE 可以反复调用同一层，每次消费上一枚候选与前一步特征，逐步形成更长的候选链。候选深度由 Runtime 的步数控制，不是每多猜一步就再加载一层 NextN。

## 二、NextN 怎样组合 Token 与 Target hidden

NextN 的模型入口是：

[`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py)

核心类：

~~~python
DeepseekV4ModelNextN
DeepseekV4ForCausalLMNextN
~~~

NextN 的输入除了 token，还带有上一步的多路 hidden feature。Draft Extend 从 Target 计算过的状态起步，后续 rollout 再沿 NextN 自己的输出特征递推；它不需要为了每枚候选重新执行完整 Target。先看模型怎样把这两种输入合起来。

一边是当前 candidate token：

~~~python
hidden_states = self.embed_tokens(input_ids)

e_proj_hidden_states, _ = self.e_proj(
    self.enorm(hidden_states)
)
~~~

另一边是 `spec_info.hidden_states` 携带的前一步特征。Draft Extend 时它来自 Target，rollout 中则会换成上一轮 NextN 的输出；模型本身按同一输入契约读取它：

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

图中 Target confirmed hidden 表示 Draft Extend 的起点。进入 `draft_forward()` 后，Worker 会在每次 NextN forward 前把当前 `hidden_states` 放进 `spec_info`，执行后再用 `logits_output.hidden_states` 更新它。因此第二、第三枚候选依赖前面 Draft 的 token 与特征，不是每一步都重新向 Target 索取一份已确认的未来 hidden。

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

因此，当前 DeepSeek-V4 Target→Draft 边界传递的 feature shape 是 `[selected_token_rows, hc_mult × hidden_size]`。它保留 mHC 的多路特征，不是普通的 `[selected_token_rows, hidden_size]`；读取方不能只因为行数相同，就按一个 hidden 向量解释整行。

NextN 收到后再：

~~~python
spec_info.hidden_states.view(
    n_tokens * hc_mult,
    hidden_size,
)
~~~

代码先把各路特征临时排成 `[n_tokens × hc_mult, hidden_size]` 做归一化与 `h_proj`，再恢复为 `[n_tokens, hc_mult, hidden_size]`。Token Embedding 经 `e_proj` 得到 `[n_tokens, hidden_size]`，插入一维后成为 `[n_tokens,1,hidden_size]`，向各路广播相加。这里是保留 stream 维度做融合，不是先把多路特征平均成一条。

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

这个教学例子把 hidden_size 缩到 8，M 代表 `hc_mult`，不是在声称模型实际维度为 8。示意图展示逻辑上的 `[2,M,8]`，实现做投影时会先临时展平为 `[2×M,8]`。NextN 的 decoder 输出也会保留 `pre_hc_head`，作为后续可捕获的多路特征；用于 LM Head 的则是聚合、归一化后的 hidden。

## 三、Draft Extend 产生第一枚候选

生成第一枚 Draft 候选前，先看这一轮从哪里起步。上一轮 Target Verify 或 Prefill 完成后，会得到一枚 **Target 已确认的 Token**。

在 EAGLE 数据结构里，这枚边界 token 叫 `bonus_tokens`。它是 Target 已经决定的输出，尚不等于它自己的 Target KV 已经算完。

`EagleDraftInput` 的源码注释也把它定义为：

~~~text
Per-req bonus token
(the "+1" target prediction at end of each accept chain)
~~~

固定源码：

[`EagleDraftInput`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_info.py#L143-L180)

bonus 属于已经确认的边界，candidate1 属于 NextN 接下来提出的未来。两者如果混为一谈，后面的候选数量和 Verify 窗口都会少算一位。

### Target 状态先进入 Draft Extend

Prefill 后通过 `_draft_extend_for_prefill()` 建立第一轮 Draft 状态；Verify 后则通过 `_draft_extend_for_decode()` 把新确认的历史送给 Draft。下面讨论后者，也就是稳态轮次之间的交接。

以 Decode 稳态为例，Draft Extend 输入里有：

~~~python
hidden_states=batch_result.logits_output.hidden_states
~~~

也就是刚刚 Target Verify 捕获的 mHC hidden。

NextN 的有效 input token 来自 Target 这轮产生的输出路径，hidden 则来自产生这些预测的 Target 输入行，两者相差一个位置。例如前一轮 Verify 输入 `R、A、C`、输出 `A、C、X`，Draft Extend 将有效 token `A、C、X` 与对应的 Target hidden `h(R)、h(A)、h(C)` 配对。最后一组用 `X` 的 Embedding 加 `h(C)`，为 `X` 后面的第一枚 Draft 候选提供起点。

这个例子只列有效路径；Decode Draft Extend 的实际 buffer 按固定窗口组织，还可能包含无效尾部。Worker 用每请求 block 起点加 `accept_lens-1` 的 `select_index` 选择最后有效位置，不能把数组末行直接当成每条请求的边界。

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

这里写回的 `topk_index` 才是下一轮 `draft_forward()` 开始时已经准备好的第一枚 Draft candidate；对应的 `ret_hidden_states` 则来自这次 NextN 执行，成为后续 rollout 的 seed。

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

所以从 Target 到 Draft 的交接不是仅传一个 token ID。边界 token、对齐的 Target feature、接受长度与 Draft 缓存共同决定 Extend 从哪里继续；它产出的 proposal 与 Draft feature 再一起进入后续循环。

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

Target 下一次验证候选时，仍要把旧 bonus 当作 root 输入。以 `X` 为边界，窗口从 `[X,d1,d2,d3]` 开始；Target 在这一轮才计算 `X` 自己的 KV，并检查后面的 Draft 候选。prepend 是在补齐因果窗口，不是把 `X` 重新归类为 Draft 生成的 token。

## 四、算清候选数与 NextN Forward 次数

下面按 Eager 控制流统计 NextN 执行次数。Graph capture/replay 仍需保留相同的数据依赖，但一次 replay 的提交次数要另计，不能与 Python 层的 forward 调用数混用。

假设：

~~~text
speculative_num_steps = 3
speculative_eagle_topk = 1
~~~

固定源码在 `topk=1` 且窗口宽度不一致时，会记录 warning，并将 `speculative_num_draft_tokens` 调整为：

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

进入 `EagleDraftWorker.draft_forward()` 时，`topk_index` 已由前一阶段 Draft Extend 准备好。对 `topk=1`，它只包含每条请求的 `candidate1`；循环开始前并不需要再运行一次模型来产生它。

源码随后进入：

~~~python
for i in range(self.speculative_num_steps):
~~~

`i=0` 时先记录已有 candidate1，再以它和当前 Draft hidden 为输入，运行一次 NextN，得到 candidate2：

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

只看这个循环会得到 2 次 NextN forward；若从上一轮 Target 刚确认完状态开始计算，还要计入前面产生 candidate1 的 `1 × Draft Extend forward`。后者通常按多行窗口执行，不能因为两种工作都记作一次 forward，就认为它们耗时相同。

因此稳态下，从上一轮 Target Verify 结束，到下一轮 3 枚 candidate 全部准备完成，NextN 模型总共执行：

~~~text
1 × Draft Extend
+
2 × Draft Decode Forward
=
3 × NextN forward
~~~

这个等式统计的是 Eager 路径上的模型调用次数，不是每 token 的算子数或设备耗时。三个调用跨在 Draft Extend 与 Draft Decode 两个阶段；Graph replay 可以改变提交方式，却不把这些先后依赖变成一次无依赖的块式提议。

### 为什么 Target Verify 是 4 个 Token

`draft_forward()` 得到 `[draft1,draft2,draft3]`，构建 Verify Input 时在前面补上旧 bonus，最终窗口为：

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

`num_draft_tokens` 在这里计数的是整个 Verify 窗口，包含旧 bonus。三个候选全部接受时，本轮最多输出三个 accepted drafts 加一个新 Target bonus；若第一个候选就不匹配，本轮仍可能只输出一个 Target token。因此窗口 4 行不是保证产出 4 枚 token，更不是接受长度的实测值。

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

这里的 `compress_ratio=0` 选择 SWA 路径，不为这个 NextN layer 使用 C4/C128 compressed extra-cache。它没有把 Attention 或 KV Cache 关掉；仍需要保存 NextN 继续 rollout 所用的 SWA 状态。

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

ratio 0 对应的 compressed pool 为 None，与 SWA storage 是否存在是两件事。排查 Draft 缓存时，如果只检查 compressed pool，就会误把“没有压缩池”当成“没有缓存”。

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

Backend 初始化还有一个对应分支：若当前 runner 是 draft worker，就清空从 Target 配置读取的 compression-ratio list，避免按 Target 的层表构造 Draft compressed metadata：

~~~python
if model_runner.is_draft_worker:
    self._dsv4_compress_ratios = type(
        hf.compress_ratios
    )()
~~~

因此 Draft backend 不会按 Target 的 C4/C128 层表去建立 compressed metadata。

### NextN 缓存配置的适用范围

SWA-only 只描述当前 NextN Draft layer。完整 Target 仍按自身层配置使用 `SWA / C4 / C128`，开启 MTP 并没有把 Target 的压缩历史路径改成 SWA。性能分析也应分别看 Draft 的 SWA 成本与 Target Verify 的混合缓存成本。

## 六、Ascend 上的模型、Worker 与 Backend

把模型输入、候选计数与缓存路径对齐后，再看 Ascend 910C 上由哪些执行对象承接这些阶段。

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

该固定教程的部署示例串起的是下面这条路径：

~~~text
MTP checkpoint
    ↓
EAGLE algorithm
    ↓
DeepseekV4ForCausalLMNextN
~~~

这里选择的是 MTP / NextN。Checkpoint 中存在其他 head，并不意味着这次示例同时执行了 DSpark。上面的 CLI 只是相关参数节选，尚不包含模型路径等完整启动条件。

### Draft Model 层

EAGLE 模式下，DeepSeek-V4 Draft architecture 是 `DeepseekV4ForCausalLMNextN`，内部物化一个 NextN `DeepseekV4DecoderLayer`。ModelSlim 配置会把 decoder 的权重 prefix 对到 `mtp.0`，这属于 MTP 权重加载路径，不是先转成 DSpark model 再运行。

### Draft Decode Attention 层

`DraftBackendFactory` 根据：

~~~text
attention backend = dsv4
device = npu
~~~

创建：`DeepseekV4AscendMultiStepDraftBackend`

固定源码：

[`DraftBackendFactory._create_dsv4_decode_backend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L426-L453)

这个 multi-step container 为不同 speculative step 组织 `DeepseekV4AscendAttnBackend`，维护各 step 的 KV location 和 metadata view。它是 Attention 执行组织，不是又加载多个 NextN decoder layer；模型层数与 step 后端对象数不能混算。

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

模型类负责用 MTP 权重计算，Worker 决定先 Extend 再 rollout 的执行顺序，Attention Backend 则承接各阶段的缓存与 metadata。区分这三层，才能解释为什么日志里既有 NextN、EAGLE，也有 Multi-Step Draft Backend，而它们并不是三套并行运行的 Draft 算法。

## 七、少跑 Target 的收益与 Draft 成本

NextN 仍按候选顺序推进：Draft Extend 给出第一枚候选，后续 rollout 再生成下一枚。收益来自用较轻的 Draft 替代一部分昂贵的 Target Decode，并由 Target 对候选窗口统一验证。

在相同批次与资源条件下，设一轮实际推进 `A` 个输出 Token。普通 Decode 的对应成本约为 `A × C_target`；投机周期则包含 Draft Extend、rollout、Target Verify、Accept 与状态整理。后者总耗时更低时，才得到这一轮的生成加速。成本按实际执行的候选工作计算，产出按最终接受并提交的 Token 计算。

| 一轮中的工作 | 作用 |
| --- | --- |
| Draft Extend | 消费已确认状态，准备首个候选与后续 Draft 输入 |
| NextN rollout | 逐步产生剩余候选 |
| Target Verify | 为候选窗口计算预测与 hidden states |
| Accept 与提交 | 确定有效输出，整理已计算的输入状态，并保留下一轮 bonus |

接受率下降时，后半段 rollout 可能成为无效工作；窗口过宽时，Verify 与缓存整理也会变贵。计数上，AL 若包含 bonus，就要与只统计 Draft 的严格接受率分开，并用同一测量窗口的实际提出、接受数计算后者，不能仅凭窗口宽度倒推。比较性能时继续对齐批次、资源、输入输出长度与缓存条件，同时观察 TPOT、E2E、输出吞吐和 Draft/Verify 耗时。读懂这条执行链之后，再结合 [投机解码性能分析](../performance/deepseek-v4-speculative-performance-analysis.md) 选择要测量的窗口和并发。

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
