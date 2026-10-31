# DeepSeek-V4 MTP：NextN 怎样逐步生成候选

NextN 仍按候选顺序生成：Draft Extend 先准备首枚候选和 hidden seed，后续候选依次消费前一步 Token 与特征。它用单层 Draft 承担连续预测，再让完整 Target 一次验证窗口。

下面取三个候选，跟踪边界 Token、Target hidden 和 NextN hidden 的交接，算清首枚候选来自哪里、一次循环实际调用几次模型。DSpark 使用另一种块式提议，二者的 Draft 成本不能用同一个调用次数估算。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线是 DeepSeek-V4 bundled MTP / NextN，由 EAGLE-family worker 驱动；DSpark 是不同 Draft 实现。固定 V4 EAGLE 配置限定 `topk=1`，通用多分支另见 Draft Tree 专题。

</details>

## 一、MTP、NextN 与 EAGLE 的职责

MTP、NextN、EAGLE 可以同时出现，但分别属于权重能力、模型实现和执行算法：

| 名字 | 在本文中的层次 | 当前 DeepSeek-V4 路径里的含义 |
| --- | --- | --- |
| MTP | checkpoint / model capability | DeepSeek-V4 checkpoint 中携带的未来 Token 预测权重，源码可见 `mtp.*` 权重命名 |
| NextN | SGLang draft architecture | SGLang 把这组权重装进 `DeepseekV4ForCausalLMNextN`，当前只物化 1 个 NextN decoder layer |
| EAGLE | speculative runtime algorithm | SGLang 用 `EAGLEWorkerV2 / EagleDraftWorker` 驱动 Draft Extend、多步 Draft、Target Verify 与状态推进 |

它们连接为：

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

因此，MTP 说明训练目标，NextN 提供 Draft 计算，EAGLE Runtime 安排生成、验证和状态推进。

### 为什么 CLI 写 NEXTN，运行时最后却看到 EAGLE

常规参数解析将 `NEXTN` alias 归一为 `EAGLE`，实现见 [`_resolve_speculative_algorithm_alias()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L74-L111)。

~~~python
if speculative_algorithm == "NEXTN"         or speculative_algorithm == "EAGLE":
    ...
    return "EAGLE"
~~~





所以指定 `--speculative-algorithm NEXTN` 后，可在 Runtime 看到 `SpeculativeAlgorithm.EAGLE`。这是参数别名与模型类处于不同层，并未因此替换 Draft 权重。下文代码均省略上下文，只用于解释分支和数据关系。

Gemma4 assistant draft 另可将 NEXTN/EAGLE 继续解析为 `FROZEN_KV_MTP`，因此两者并非全局可互换的术语。

### DeepSeek-V4 Draft architecture 为什么会变成 NextN

DeepSeek-V4 checkpoint 可带不同 speculative head，`ModelConfig` 按最终算法选择 Draft architecture。

DSpark 选择 `DeepseekV4ForCausalLMDSpark`；EAGLE / NextN 则设置下列类和预测层数。实现见 [`model_config.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/configs/model_config.py#L878-L888)。

~~~python
self.hf_config.architectures[0] = "DeepseekV4ForCausalLMNextN"
self.hf_config.num_nextn_predict_layers = 1
~~~





加载器识别 `mtp.*` 并映射到 NextN，固定实现还检查层数：[`DeepseekV4ForCausalLM.load_weights()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5363-L5377)

~~~python
assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
~~~





只有一个 NextN layer，也可以反复调用以生成多个候选。每次读取前一步 Token、特征，候选深度由 Runtime 步数控制，无需每步多加载一层。

## 二、NextN 怎样组合 Token 与 Target hidden

模型入口位于：

[`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py)

主要类为 `DeepseekV4ModelNextN` 和 `DeepseekV4ForCausalLMNextN`。

NextN 同时读取 Token 和前一步多路 hidden。Draft Extend 从 Target 状态起步，后续 rollout 递推 NextN 自己的特征，无需逐候选重新跑完整 Target。

Token 一侧先执行：

~~~python
hidden_states = self.embed_tokens(input_ids)

e_proj_hidden_states, _ = self.e_proj(
    self.enorm(hidden_states)
)
~~~

特征一侧从 `spec_info.hidden_states` 读取。它在 Extend 时来自 Target，在 rollout 时来自上一 NextN 输出，模型按相同输入契约处理：

~~~python
hc_flat = forward_batch.spec_info.hidden_states.view(
    n_tokens * self.hc_mult,
    d,
)

h_proj_out, _ = self.h_proj(
    self.hnorm(hc_flat)
)
~~~

再将两路融合：

~~~python
hidden_states = (
    e_proj_hidden_states[:, None, :]
    + h_proj_hidden_states
)
~~~

从 Extend 起点看，可画为：

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

图中 Target confirmed hidden 只表示 Extend 的起点。`draft_forward()` 每步先将当前特征放入 `spec_info`，forward 后再用 `logits_output.hidden_states` 更新，因此后续候选依赖 Draft 自身递推，不是向 Target 取得尚未确认的未来 hidden。

### Target 到 Draft 的多路 Hidden Feature

Target 的多路 mHC feature 也需完整保留。

最后一层结束时先保存：[`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4808-L4839)

~~~python
pre_hc_head = hidden_states.flatten(1)
~~~

原状态为 `[T, hc_mult, H]` 时，展平结果为 `[T, hc_mult × H]`。







Target LogitsProcessor 捕获特征时，同时取得 normalized hidden 与 `hidden_states_before_norm=pre_hc_head`。后者存在时，`_get_hidden_states_to_store()` 优先将它放进输出。[`LogitsProcessor._get_hidden_states_to_store()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L785-L836)

~~~python
if hidden_states_to_store_before_norm is not None:
    hidden_states_to_store = hidden_states_to_store_before_norm
~~~





所以 Target→Draft 特征是 `[selected_token_rows, hc_mult × hidden_size]`，保留多路信息，不能只因行数相同就按 `[selected_token_rows, hidden_size]` 解读。

NextN 读取后先做：

~~~python
spec_info.hidden_states.view(
    n_tokens * hc_mult,
    hidden_size,
)
~~~

各路暂排为 `[n_tokens × hc_mult, hidden_size]`，归一化并经 `h_proj` 后恢复多路形状。Embedding 经 `e_proj` 得到 `[n_tokens, hidden_size]`，扩为 `[n_tokens,1,hidden_size]` 向各路广播相加，没有先将多路平均。

用一个缩小维度的教学例子：

~~~text
n_tokens = 2
hidden_size = 8
hc_mult = M
~~~

可对照为：

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

这里 hidden_size=8 仅为演示，M 是 `hc_mult`。图画 `[2,M,8]` 逻辑关系，实际投影会暂排 `[2×M,8]`。NextN 输出继续保留 `pre_hc_head` 供捕获，LM Head 则消费聚合、归一化后的表示。

## 三、Draft Extend 产生第一枚候选

每轮从 Target Prefill 或 Verify 已确认的一枚输出起步。

EAGLE 称它为 `bonus_tokens`。ID 已确认，仍不代表该 Token 自身的 Target KV 已计算。

数据结构的注释定义为：[`EagleDraftInput`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_info.py#L143-L180)

~~~text
Per-req bonus token
(the "+1" target prediction at end of each accept chain)
~~~





bonus 是边界，candidate1 是边界之后的提议，混淆它们会将候选数或 Verify 窗口少算一位。

### Target 状态先进入 Draft Extend

Prefill 后由 `_draft_extend_for_prefill()` 初始化，Verify 后由 `_draft_extend_for_decode()` 接续确认历史，下面沿后者说明稳态交接。

Draft Extend 从 `batch_result.logits_output.hidden_states` 取得本轮 Target Verify 捕获的 mHC hidden。

有效 Token 来自 Target 输出路径，hidden 却来自产生这些预测的输入行，两者错一位。例如输入 `R、A、C` 预测出 `A、C、X`，配对便是 Token `A、C、X` 与 `h(R)、h(A)、h(C)`。最后用 X 的 Embedding 和 h(C) 形成继续预测的起点。

实际 buffer 还可能含无效尾部，Worker 按每请求 block 起点加 `accept_lens-1` 的 `select_index` 取最后有效位置，不能直接读数组末行。

Extend 跑一次 NextN，具体入口见 [`EagleDraftWorker._draft_extend_for_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L1127-L1276)。

~~~python
draft_logits_output = self.draft_runner.forward(
    forward_batch
).logits_output
~~~

输出的 `ret_topk_p`、`ret_topk_index`、`ret_hidden_states` 随后保存为下一轮输入：

~~~python
next_draft_input.topk_p = ret_topk_p
next_draft_input.topk_index = ret_topk_index
next_draft_input.hidden_states = ret_hidden_states
~~~





其中 `topk_index` 已包含下一轮 candidate1，`ret_hidden_states` 则是这次 NextN 形成的 rollout seed。

时序因此为：

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

这次交接同时依赖边界 Token、对齐的 Target feature、接受长度与 Draft 缓存。Extend 产生的候选与 Draft feature 再一起进入后续步骤。

### 为什么 Verify Window 里又会看到 bonus

构造 Target Verify 时，仍需在候选前加入旧 bonus，源码见 [`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py)。

~~~python
draft_tokens = torch.cat(
    (bonus_tokens.unsqueeze(1), draft_tokens),
    dim=1,
)
~~~





例如窗口 `[X,d1,d2,d3]` 中，Target 先计算 X 自身 KV，再检查候选。prepend 是补齐因果输入，不会将 X 改算成 Draft 候选。

## 四、算清候选数与 NextN Forward 次数

下面按 Eager 控制流计模型调用。Graph replay 可改变提交方式，但数据依赖不变，不能把 replay 次数与 Python forward 次数混算。

令参数为：

~~~text
speculative_num_steps = 3
speculative_eagle_topk = 1
~~~

固定 `topk=1` 下，若窗口宽度不符，会警告并调整为以下关系。实现见 [`_handle_eagle_family()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L1067-L1078)。

~~~python
speculative_num_draft_tokens
=
speculative_num_steps + 1
~~~

~~~text
num_steps = 3
num_draft_tokens = 4
~~~





一个位置是旧 bonus，另外三个才是 Draft 候选。

### `draft_forward()` 里只有 3 枚 Draft candidate

进入 `draft_forward()` 时，Extend 已准备 candidate1 的 `topk_index`；topk=1 时，每请求已有一个候选，无需再跑模型重复产生它。循环执行 `for i in range(self.speculative_num_steps)`，每次先保存当前候选，再判断是否还需继续 forward。[`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L778-L951)

把三个候选代入，顺序如下：

~~~text
Draft Extend：产生 candidate1 和 hidden seed
i=0：保存 candidate1 → NextN forward #1 → candidate2
i=1：保存 candidate2 → NextN forward #2 → candidate3
i=2：保存 candidate3 → 命中最后一步，break
~~~

最后一步对应 `if i == self.speculative_num_steps - 1: break`。因此，`draft_forward()` 输出三枚候选，内部只调用两次 `draft_runner.forward()`；从上轮 Target 刚确认后开始算，再加一次产生 candidate1 的 Draft Extend，总共三次 NextN forward。

Extend 常处理多行窗口，单步 rollout 则消费当前候选。二者都算一次模型调用，并不代表耗时相同。这里统计的是 Eager 控制流，Graph 可以改变提交组织，却不会让这些依赖变成相互独立的块式提议；也不能由调用数推算每 Token 的算子数或设备耗时。

### 为什么 Target Verify 是 4 个 Token

`draft_forward()` 返回三枚候选，Verify 构造再补旧 bonus：

~~~text
Verify Window:

[bonus, draft1, draft2, draft3]
~~~

topk=1 链式窗口因此为 `4`，即 `speculative_num_draft_tokens = 4`，相关计数如下：

| 概念 | `num_steps=3, topk=1` | 含义 |
| --- | ---: | --- |
| Target bonus token | 1 | 上一轮 Target 已确认的边界 Token |
| Draft candidates | 3 | NextN 提出的未来候选 |
| `draft_forward()` 内 NextN forward | 2 | candidate1 已由 Draft Extend 提前产生 |
| 从上一轮 Target 状态算起的 NextN forward 总数 | 3 | 1 次 Draft Extend + 2 次 Draft Decode |
| Target Verify window | 4 | `[bonus, d1, d2, d3]` |

这里 `num_draft_tokens` 指含 bonus 的整个 Verify 窗口。三枚全通过时最多输出三枚 Draft 加一个新 bonus，首枚就分歧则可能仅输出一枚 Target Token。窗口行数是配置，不是实测接受长度。

## 五、NextN 的 SWA 缓存

Target 各层可采用：

~~~text
SWA
C4
C128
~~~

NextN 构造则固定：

~~~python
COMPRESS_RATIO_NEXTN_LAYER = 0
~~~

并传入 Decoder Layer：

~~~python
DeepseekV4DecoderLayer(
    ...,
    is_nextn=True,
    compress_ratio_override=0,
)
~~~



对应实现见 [`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py)。

`compress_ratio=0` 选择 SWA，不使用该层 C4/C128 extra-cache，仍保留继续 rollout 所需 Attention、SWA KV。

### Memory Pool 侧：ratio 0 没有 compressed pool，但 SWA pool 仍然存在

`DeepSeekV4TokenToKVPool` 总会建立 stage 的 SWA storage，compressed pool 只为实际 ratio 4、128 创建。因此 ratio 0 映射为 `compress_kv_pool=None`，表示没有压缩池，SWA pool 仍然存在。只查 compressed pool，会把“无压缩缓存”误读为“无缓存”。[`DeepSeekV4TokenToKVPool._init_compressed_layer_mapping()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1637-L1656)

### Ascend Backend 侧更直接：ratio 0 就进入 `_forward_swa()`

Ascend 主分支相应选择：

~~~python
if compress_ratio == 0:
    return self._forward_swa(
        q, layer, forward_batch, attn_sink
    )
~~~

该函数通过 `ori_kv = pool.get_swa_buffer(layer.layer_id)` 读取当前层的 SWA。



实现见 [`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2116)。

因此当前 NextN 使用 ratio 0 的 SWA KV，不添加该层的 C4、C128 extra-cache。

Draft backend 初始化还会清除 Target 的 compression-ratio list：

~~~python
if model_runner.is_draft_worker:
    self._dsv4_compress_ratios = type(
        hf.compress_ratios
    )()
~~~

避免按 Target 层表建立 Draft compressed metadata。

### NextN 缓存配置的适用范围

SWA-only 仅针对当前 NextN layer，Target 仍按原层配置访问 `SWA / C4 / C128`。性能上应分别统计 Draft SWA 与 Target Verify 的混合缓存工作。

## 六、Ascend 上的模型、Worker 与 Backend

再看各阶段由哪些 Ascend 910C 执行对象承接。

固定官方教程的 speculative 参数为：

~~~bash
--device npu
--attention-backend dsv4
--speculative-algorithm EAGLE
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
~~~

对应权重示例为 `DeepSeek-V4-Flash-w8a8-mtp`。



教程入口见 [DeepSeek-V4-Flash Ascend Tutorial](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx)。

它连接的是：

~~~text
MTP checkpoint
    ↓
EAGLE algorithm
    ↓
DeepseekV4ForCausalLMNextN
~~~

存在其他 head 不意味着同时执行 DSpark。这里选中的是 MTP / NextN，CLI 也只是相关参数节选，尚缺模型路径等完整条件。

### Draft Model 层

EAGLE 模式物化一个 `DeepseekV4ForCausalLMNextN` 的 NextN `DeepseekV4DecoderLayer`，ModelSlim 将 decoder 权重 prefix 对到 `mtp.0`。这条加载路径不先转成 DSpark。

### Draft Decode Attention 层

Draft Decode 的工厂按下列条件选择：

~~~text
attention backend = dsv4
device = npu
~~~

得到 `DeepseekV4AscendMultiStepDraftBackend`，入口见 [`DraftBackendFactory._create_dsv4_decode_backend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L426-L453)。





这个 container 为各 speculative step 组织 Attention backend、KV locations 和 metadata view。它增加的是 step 执行对象，不是加载多个 NextN decoder layer。[`DeepseekV4AscendMultiStepDraftBackend`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2354-L2512)





### Draft Extend Attention 层

Draft Extend 同样选择 `dsv4`，但直接使用注册表：



~~~python
ATTENTION_BACKENDS["dsv4"](draft_model_runner)
~~~

NPU 注册项指向 `DeepseekV4AscendAttnBackend`，与 Decode 的 multi-step container 分别对应。



- [`DraftBackendFactory._create_dsv4_prefill_backend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L574-L601)
- [`attention_registry.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/attention_registry.py#L166-L192)

两条 Attention 入口与模型、Worker 的层级如下：

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

NPU Graph 还可使用专门 Draft / Draft-Extend runner，它们优化执行方式，不改变算法状态机。

因此 NextN 是 MTP 计算模型，Worker 安排 Extend、rollout，Backend 管理各阶段 Attention metadata 与缓存。日志里同时出现这些名字，不代表并行运行三套 Draft 算法。

## 七、少跑 Target 的收益与 Draft 成本

NextN 先由 Extend 给首候选，再顺序 rollout。它通过较轻 Draft 减少完整 Target 的逐步解码，再统一验证。

设同批次、资源下实际推进 `A` 枚输出，普通路径对应成本约为 `A × C_target`。投机周期要合计 Extend、rollout、Verify、Accept、状态整理；只有总耗时更低才加速。候选工作按实际执行计成本，产出按最终接受提交计。

| 一轮中的工作 | 作用 |
| --- | --- |
| Draft Extend | 消费已确认状态，准备首个候选与后续 Draft 输入 |
| NextN rollout | 逐步产生剩余候选 |
| Target Verify | 为候选窗口计算预测与 hidden states |
| Accept 与提交 | 确定有效输出，整理已计算的输入状态，并保留下一轮 bonus |

接受差时，后段 rollout 可能无效；窗口过宽，也会增加 Verify 和整理。AL 若含 bonus，需与仅数 Draft 的严格接受率分开，后者按同窗口实际提出、接受计数计算。性能比较还要对齐长度、缓存、批次、资源，并观察 TPOT、E2E、吞吐、Draft/Verify 耗时。后续测量可见 [投机解码性能分析](../performance/deepseek-v4-speculative-performance-analysis.md)。

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
