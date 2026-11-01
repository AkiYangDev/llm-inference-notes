# DeepSeek-V4 MTP：NextN 怎样逐步生成候选

topk=1 时，配置要生成三个候选，`draft_forward()` 却只调用两次 NextN forward。缺少的那次在上轮 Draft Extend：它已经准备好首枚候选，后面只需继续生成第二、第三枚。

NextN 用单层 Draft 连续预测，再让完整 Target 一次验证窗口。这里沿三个候选的交接追下去，看边界 Token、Target hidden、NextN hidden 怎样进入下一步。DSpark 一次算候选块，Draft 成本需要按另一条执行路径统计。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线为 DeepSeek-V4 bundled MTP / NextN，由 EAGLE-family worker 驱动；固定 V4 EAGLE 配置限定 `topk=1`，通用多分支另见 Draft Tree 专题。

</details>

## 一、MTP、NextN 与 EAGLE 的职责

MTP、NextN、EAGLE 出现在同一份配置中，分别描述训练能力、Draft 模型实现、运行算法：

| 名字 | 在本文中的层次 | 当前 DeepSeek-V4 路径里的含义 |
| --- | --- | --- |
| MTP | checkpoint / model capability | DeepSeek-V4 checkpoint 中携带的未来 Token 预测权重，源码可见 `mtp.*` 权重命名 |
| NextN | SGLang draft architecture | SGLang 把这组权重装进 `DeepseekV4ForCausalLMNextN`，当前只物化 1 个 NextN decoder layer |
| EAGLE | speculative runtime algorithm | SGLang 用 `EAGLEWorkerV2 / EagleDraftWorker` 驱动 Draft Extend、多步 Draft、Target Verify 与状态推进 |

调用关系是：

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

MTP 给出训练目标，NextN 执行 Draft 计算，EAGLE Runtime 安排候选生成、验证与状态推进。

<a id="cli-nextn-eagle"></a>

### NEXTN 参数别名与 EAGLE 运行算法

常规参数解析在 [`_resolve_speculative_algorithm_alias()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L74-L111) 中将 `NEXTN` 归一为 `EAGLE`。

~~~python
if speculative_algorithm == "NEXTN"         or speculative_algorithm == "EAGLE":
    ...
    return "EAGLE"
~~~

因此指定 `--speculative-algorithm NEXTN`，运行时会看到 `SpeculativeAlgorithm.EAGLE`，Draft 权重并没有被更换。文中代码省略上下文，用来展示分支和数据关系。

这个别名有适用范围。Gemma4 assistant draft 还可能继续解析到 `FROZEN_KV_MTP`，不能把 NEXTN、EAGLE 当成全局同义词。

<a id="deepseek-v4-draft-architecture-nextn"></a>

### ModelConfig 选择 NextN Draft architecture

DeepSeek-V4 checkpoint 可以包含不同 speculative head。`ModelConfig` 根据最终运行算法选择 Draft architecture。

DSpark 使用 `DeepseekV4ForCausalLMDSpark`；EAGLE / NextN 设置以下类和预测层数，见 [`model_config.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/configs/model_config.py#L878-L888)。

~~~python
self.hf_config.architectures[0] = "DeepseekV4ForCausalLMNextN"
self.hf_config.num_nextn_predict_layers = 1
~~~

[`DeepseekV4ForCausalLM.load_weights()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L5363-L5377) 将 `mtp.*` 映射到 NextN，并检查层数：

~~~python
assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
~~~

一个 NextN layer 可以反复调用。每次读取前一步 Token 和特征，Runtime 用调用步数控制候选深度，无需为每一步再加载一层。

## 二、NextN 怎样组合 Token 与 Target hidden

模型实现为：

[`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py)

主要类是 `DeepseekV4ModelNextN`、`DeepseekV4ForCausalLMNextN`。

Draft Extend 从 Target 状态起步，NextN 同时读取 Token 与前一步多路 hidden。后续 rollout 用 NextN 自己的输出继续递推，Target 不必逐候选重跑。

Token 一侧计算：

~~~python
hidden_states = self.embed_tokens(input_ids)

e_proj_hidden_states, _ = self.e_proj(
    self.enorm(hidden_states)
)
~~~

特征一侧读取 `spec_info.hidden_states`。Extend 时它来自 Target，rollout 时来自上一 NextN 输出：

~~~python
hc_flat = forward_batch.spec_info.hidden_states.view(
    n_tokens * self.hc_mult,
    d,
)

h_proj_out, _ = self.h_proj(
    self.hnorm(hc_flat)
)
~~~

两路融合为：

~~~python
hidden_states = (
    e_proj_hidden_states[:, None, :]
    + h_proj_hidden_states
)
~~~

从 Extend 的起点画出数据流：

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

图中的 Target confirmed hidden 只用于说明 Extend 起点。`draft_forward()` 每步把当前特征放进 `spec_info`，算完后用 `logits_output.hidden_states` 更新它。未来候选因而依赖 Draft 自身的递推特征。

### Target 到 Draft 的多路 Hidden Feature

Target 的 mHC feature 带有多路信息，传给 Draft 时仍需保留。

[`DeepseekV4Model.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L4808-L4839) 在最后一层之后保存：

~~~python
pre_hc_head = hidden_states.flatten(1)
~~~

原形状为 `[T, hc_mult, H]`，展平后是 `[T, hc_mult × H]`。

Target LogitsProcessor 捕获 normalized hidden，也取得 `hidden_states_before_norm=pre_hc_head`；后者存在时，`_get_hidden_states_to_store()` 优先把它放入输出。[`LogitsProcessor._get_hidden_states_to_store()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/logits_processor.py#L785-L836)

~~~python
if hidden_states_to_store_before_norm is not None:
    hidden_states_to_store = hidden_states_to_store_before_norm
~~~

所以传给 Draft 的特征宽度是 `hc_mult × hidden_size`，完整形状为 `[selected_token_rows, hc_mult × hidden_size]`。只看 Token 行数，容易漏掉多路宽度。

NextN 接收后重新排列：

~~~python
spec_info.hidden_states.view(
    n_tokens * hc_mult,
    hidden_size,
)
~~~

各路暂排成 `[n_tokens × hc_mult, hidden_size]`，归一化、经过 `h_proj` 后恢复多路形状。Embedding 经 `e_proj` 得到 `[n_tokens, hidden_size]`，扩成 `[n_tokens,1,hidden_size]`，广播加到各路上。多路在这里没有先被平均。

将维度缩小，数据形状为：

~~~text
n_tokens = 2
hidden_size = 8
hc_mult = M
~~~

两路计算可以对照：

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

例中 hidden_size=8，M 表示 `hc_mult`。图用 `[2,M,8]` 展示逻辑关系，实际投影暂排为 `[2×M,8]`。NextN 输出仍保留 `pre_hc_head` 供捕获，LM Head 使用聚合、归一化后的表示。

## 三、Draft Extend 产生第一枚候选

每轮都从 Target Prefill 或 Verify 确认的一枚输出开始。

EAGLE 将它称为 `bonus_tokens`。它的 ID 已经确认，自身的 Target KV 还要等下一次作为输入执行。

[`EagleDraftInput`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_info.py#L143-L180) 对它的定义是：

~~~text
Per-req bonus token
(the "+1" target prediction at end of each accept chain)
~~~

bonus 位于边界，candidate1 是其后的提议。后续统计候选数和 Verify 宽度时，要把这个边界位置单独计入。

### Target 状态先进入 Draft Extend

Prefill 由 `_draft_extend_for_prefill()` 初始化，Verify 后由 `_draft_extend_for_decode()` 接续确认历史。以下跟踪后者。

Draft Extend 从 `batch_result.logits_output.hidden_states` 取得这次 Target Verify 捕获的 mHC hidden。

输出 Token 与产生它的输入 hidden 相差一位。输入 `R、A、C` 若预测 `A、C、X`，配对就是 `A、C、X` 对 `h(R)、h(A)、h(C)`。最后用 X 的 Embedding、h(C) 形成继续预测的起点。

实际 buffer 可能还含无效尾部。Worker 用每请求 block 起点加 `accept_lens-1` 得到 `select_index`，据此选最后有效位置。

[`EagleDraftWorker._draft_extend_for_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L1127-L1276) 在 Extend 中运行一次 NextN：

~~~python
draft_logits_output = self.draft_runner.forward(
    forward_batch
).logits_output
~~~

得到 `ret_topk_p`、`ret_topk_index`、`ret_hidden_states`，保存到下一轮输入：

~~~python
next_draft_input.topk_p = ret_topk_p
next_draft_input.topk_index = ret_topk_index
next_draft_input.hidden_states = ret_hidden_states
~~~

`topk_index` 已经含有 candidate1；`ret_hidden_states` 是同次 NextN 产生的 rollout seed。

时序为：

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

边界 Token、对齐后的 Target feature、接受长度和 Draft 缓存共同完成这次 Extend。首候选与 Draft feature 随后一起进入 rollout。

<a id="verify-window-bonus"></a>

### Target Verify 窗口包含旧 bonus

[`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py) 构建 Target Verify 时，还会在候选前加入旧 bonus。

~~~python
draft_tokens = torch.cat(
    (bonus_tokens.unsqueeze(1), draft_tokens),
    dim=1,
)
~~~

窗口 `[X,d1,d2,d3]` 中，Target 需要计算 X 的 KV，并从这条因果链验证候选。因此 X 要占一行，候选数仍只有三枚。

## 四、算清候选数与 NextN Forward 次数

以下按 Eager 控制流统计模型调用。Graph replay 可以改变提交组织，统计 replay 与 Python forward 时要采用各自的计数。

令配置为：

~~~text
speculative_num_steps = 3
speculative_eagle_topk = 1
~~~

固定 `topk=1` 时，窗口宽度若不满足关系，[`_handle_eagle_family()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L1067-L1078) 会警告并调整：

~~~python
speculative_num_draft_tokens
=
speculative_num_steps + 1
~~~

~~~text
num_steps = 3
num_draft_tokens = 4
~~~

四个位置里，一位是旧 bonus，三位是 Draft 候选。

<a id="draft_forward-3-draft-candidate"></a>

### draft_forward 复用 Extend 的首候选

进入 `draft_forward()`，Extend 已提供 candidate1 的 `topk_index`。循环 `for i in range(self.speculative_num_steps)` 每次先保存当前候选，再判断要不要继续 forward。[`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L778-L951)

三枚候选依次处理为：

~~~text
Draft Extend：产生 candidate1 和 hidden seed
i=0：保存 candidate1 → NextN forward #1 → candidate2
i=1：保存 candidate2 → NextN forward #2 → candidate3
i=2：保存 candidate3 → 命中最后一步，break
~~~

最后一次循环触发 `if i == self.speculative_num_steps - 1: break`。因此 `draft_forward()` 返回三枚候选，只调用两次 `draft_runner.forward()`。从上轮 Target 刚确认的时刻算起，加上产生 candidate1 的 Draft Extend，一共三次 NextN forward。

Extend 常处理多行窗口，单步 rollout 只消费当前候选，二者调用一次的成本并不相同。Graph 也能改变提交方式，但仍须遵守顺序依赖；模型调用数本身无法给出算子数或设备耗时。

<a id="target-verify-4-token"></a>

### Target Verify 的四行输入

三个候选再加旧 bonus，组成：

~~~text
Verify Window:

[bonus, draft1, draft2, draft3]
~~~

topk=1 的 Verify 窗口是 `4`，即 `speculative_num_draft_tokens = 4`。几种计数的对应为：

| 概念 | `num_steps=3, topk=1` | 含义 |
| --- | ---: | --- |
| Target bonus token | 1 | 上一轮 Target 已确认的边界 Token |
| Draft candidates | 3 | NextN 提出的未来候选 |
| `draft_forward()` 内 NextN forward | 2 | candidate1 已由 Draft Extend 提前产生 |
| 从上一轮 Target 状态算起的 NextN forward 总数 | 3 | 1 次 Draft Extend + 2 次 Draft Decode |
| Target Verify window | 4 | `[bonus, d1, d2, d3]` |

这里 `num_draft_tokens` 包含 bonus。三枚候选全通过，最多输出三枚 Draft 加一个新 bonus；首枚即分歧，则可能只输出一枚 Target Token。配置窗口表示允许算多宽，实际推进仍由接受结果决定。

## 五、NextN 的 SWA 缓存

Target 各层可以选择：

~~~text
SWA
C4
C128
~~~

NextN 构造固定设置：

~~~python
COMPRESS_RATIO_NEXTN_LAYER = 0
~~~

再传给 Decoder Layer：

~~~python
DeepseekV4DecoderLayer(
    ...,
    is_nextn=True,
    compress_ratio_override=0,
)
~~~

实现位于 [`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py)。

`compress_ratio=0` 选择 SWA，不增加本层 C4/C128 extra-cache，继续 rollout 所需的 Attention、SWA KV 仍在。

<a id="memory-pool-ratio-0-compressed-pool-swa-pool"></a>

### Memory Pool 保留 SWA storage

`DeepSeekV4TokenToKVPool` 总会创建 stage 的 SWA storage，只有实际 ratio 4、128 才建立 compressed pool。ratio 0 对应 `compress_kv_pool=None`，因此查压缩池时看到 None，SWA pool 仍然存在。[`DeepSeekV4TokenToKVPool._init_compressed_layer_mapping()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1637-L1656)

<a id="ascend-backend-ratio-0-_forward_swa"></a>

### Ascend ratio 0 进入 _forward_swa

Ascend 主分支选择：

~~~python
if compress_ratio == 0:
    return self._forward_swa(
        q, layer, forward_batch, attn_sink
    )
~~~

函数通过 `ori_kv = pool.get_swa_buffer(layer.layer_id)` 取得本层 SWA。

调用见 [`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2116)。

当前 NextN 的缓存读写因而落在 SWA 路径。

Draft backend 初始化还会清除 Target 的 compression-ratio list：

~~~python
if model_runner.is_draft_worker:
    self._dsv4_compress_ratios = type(
        hf.compress_ratios
    )()
~~~

Draft metadata 因此不会沿用 Target 的压缩层表。

### NextN 缓存配置的适用范围

这个 SWA-only 配置只适用于当前 NextN layer。Target 仍按原层配置访问 `SWA / C4 / C128`，分析成本时需要分别统计 Draft SWA 与 Target Verify 的混合缓存工作。

## 六、Ascend 上的模型、Worker 与 Backend

在 Ascend 910C 上，模型、Worker、Attention backend 分别承接这些步骤。

固定官方教程给出的 speculative 参数为：

~~~bash
--device npu
--attention-backend dsv4
--speculative-algorithm EAGLE
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
~~~

权重示例是 `DeepSeek-V4-Flash-w8a8-mtp`。

参数出处为 [DeepSeek-V4-Flash Ascend Tutorial](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx)。

它们连接成：

~~~text
MTP checkpoint
    ↓
EAGLE algorithm
    ↓
DeepseekV4ForCausalLMNextN
~~~

这组选中 MTP / NextN。checkpoint 里即使存在其他 head，也不会因此同时执行 DSpark。上面的 CLI 只是 speculative 参数节选，还需要模型路径等启动条件。

### Draft Model 层

EAGLE 模式创建一个 `DeepseekV4ForCausalLMNextN` 的 NextN `DeepseekV4DecoderLayer`，ModelSlim 将 decoder 权重 prefix 对到 `mtp.0`，直接完成 NextN 加载。

### Draft Decode Attention 层

Draft Decode 工厂按这些条件选择 backend：

~~~text
attention backend = dsv4
device = npu
~~~

返回 `DeepseekV4AscendMultiStepDraftBackend`，入口见 [`DraftBackendFactory._create_dsv4_decode_backend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L426-L453)。

这个 container 为各 speculative step 组织 Attention backend、KV locations、metadata view。多个 step 执行对象复用同一 NextN decoder layer。[`DeepseekV4AscendMultiStepDraftBackend`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2354-L2512)

### Draft Extend Attention 层

Draft Extend 同样使用 `dsv4`，但直接查注册表：

~~~python
ATTENTION_BACKENDS["dsv4"](draft_model_runner)
~~~

NPU 注册项指向 `DeepseekV4AscendAttnBackend`；Decode 则由前面的 multi-step container 承接。

- [`DraftBackendFactory._create_dsv4_prefill_backend()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/draft_utils.py#L574-L601)
- [`attention_registry.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/attention_registry.py#L166-L192)

各层对象的对应为：

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

NPU Graph 还可使用专门的 Draft / Draft-Extend runner，以相应方式捕获、执行这几个阶段。

日志里的 NextN 是计算模型，Worker 安排 Extend、rollout，Backend 准备 Attention metadata、管理缓存。它们处于同一条执行链的不同层。

## 七、少跑 Target 的收益与 Draft 成本

少跑 Target 的收益，要与顺序执行的 NextN 成本放在一起看。

假设同一批次、资源下实际推进 `A` 枚输出，普通路径成本约为 `A × C_target`。投机周期则合计 Extend、rollout、Verify、Accept、状态整理。候选工作全按实际执行计入成本，产出只数最终接受提交的部分。

| 一轮中的工作 | 作用 |
| --- | --- |
| Draft Extend | 消费已确认状态，准备首个候选与后续 Draft 输入 |
| NextN rollout | 逐步产生剩余候选 |
| Target Verify | 为候选窗口计算预测与 hidden states |
| Accept 与提交 | 确定有效输出，整理已计算的输入状态，并保留下一轮 bonus |

接受较差，后段 rollout 就可能白算；窗口加宽，也会增加 Verify、整理开销。对齐长度、缓存、批次、资源后，可以结合 TPOT、E2E、吞吐和 Draft / Verify 耗时比较。AL 注明是否含 bonus，严格 Draft 接受率使用同窗口实际提出、接受数。后续见 [投机解码性能分析](../performance/deepseek-v4-speculative-performance-analysis.md)。

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
