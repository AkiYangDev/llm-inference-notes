# DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject 如何驱动一次投机推理

前面的 DSpark 文章重点分析了一个分布式边界：为什么 DeepSeek-V4 MoE draft 在 DP Attention 下仍然拒绝 `attn_tp_size > 1`。但那篇文章默认读者已经接受了一个更基础的事实：DSpark 一轮 Decode 并不是一次普通 `model.forward()`，而是一套 Draft、Verify、Accept、Commit 连续状态机。

这篇文章把并行问题暂时放到背景里，只追一轮完整的 speculative decoding：

~~~text
当前已确认状态
      ↓
Draft Proposal
      ↓
Verify Planning
      ↓
Target Verify
      ↓
Accept / Reject
      ↓
Commit accepted state
      ↓
下一轮 Draft
~~~

真正要回答的是：

> **Draft 为什么可以一次提出多枚候选 Token？Target 为什么可以一次验证一条候选链？错误候选为什么不会污染下一轮状态？**

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线使用 DSpark V2、DeepSeek-V4 target / draft、普通链式 speculative decoding。源码同时支持 greedy、sampling、static verify、compact/ragged verify、folded proposal / verify epilogue 等优化；正文先解释共同语义，再标注优化路径。本文不宣称 speculative decoding 在任意负载下必然提速：实际收益取决于 acceptance、Draft 成本、Target Verify 成本、batch、graph 与通信开销。

## 一、普通 Decode 为什么需要 speculative：减少的不是数学步骤，而是昂贵 Target Forward 的轮数

普通自回归 Decode 的控制流非常直接：

~~~text
Token t
   ↓
Target Model Forward
   ↓
sample Token t+1
   ↓
下一轮 Target Forward
~~~

如果最终要生成 100 枚 Token，通常需要把 Target decode loop 推进很多轮。每一轮虽然只新增少量 Query rows，却仍然要运行大模型的完整层栈、读取历史状态并完成必要通信。

Speculative decoding 的目标不是改变 Target Model 的分布，而是先用较便宜的 Draft 路径提出一串候选，再让 Target 在一次 verify forward 中检查这串候选。

概念上：

~~~text
普通 Decode:
Target → 1 token
Target → 1 token
Target → 1 token
Target → 1 token

Speculative:
Draft → candidate A B C D
             ↓
Target Verify(A B C D)
             ↓
一次提交多个正确 Token
~~~

如果 Draft 猜得足够准，那么一次 Target Verify 可以让序列前进多于 1 个 Token，从而减少后续 Target decode 轮数。反过来，如果 acceptance 很低、Draft 本身很贵或 Verify 开销很大，收益也会下降。

DSpark 当前总入口在 [`DSparkWorkerV2.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L555-L578)。Prefill / Extend 仍然先走 Target；真正的 speculative loop 从 `_forward_decode()` 开始。

一轮 Decode 的高层调用顺序可以直接从源码展开：

~~~mermaid
flowchart TD
    B[Running ScheduleBatch]
    B --> W[alloc_verify_window]
    W --> D[DraftBlockProposer.propose]
    D --> C[Confidence / Verify Planner]
    C --> V[Target Verify]
    V --> A[Accept / Finalize]
    A --> K[Commit accepted hidden / state]
    K --> N[make_next_draft_input]
    N --> R[GenerationBatchResult]
~~~

这个顺序对应 [`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L729-L979)。

## 二、Draft 阶段：不是连续跑 gamma 次小模型，而是先构造一个 Draft Block

DSpark 的 Draft 主入口是：

~~~python
proposal = self._proposer.propose(...)
~~~

对应 [`DraftBlockProposer.propose()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L242-L344)。

`DraftBlockProposer` 维护两个容易混淆的宽度：

~~~python
self.gamma = gamma
self.query_token_num = (
    self.gamma
    if sample_from_anchor
    else self.gamma + 1
)
~~~

所以：

- `gamma` 是最终要提出的 Draft Token 数；
- Draft model forward 实际输入的 query rows 数可能是 `gamma`，也可能是 `gamma + 1`，取决于 `sample_from_anchor`。

不要把两者机械等同。

### Draft Block 的第一列来自上一轮 bonus token

`_run_forward()` 先建立：

~~~python
draft_block_ids.shape = [bs, query_token_num]
~~~

buffer 初始填充 mask token，然后：

~~~python
draft_block_ids[:, 0].copy_(
    draft_input.bonus_tokens.view(-1)
)
~~~

源码：[`DraftBlockProposer._run_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L369-L473)。

可以把一条请求的 Draft 输入想成：

~~~text
[bonus, MASK, MASK, MASK, ...]
~~~

再配上 Verify Window 预先准备的：

~~~text
positions
cache locations
~~~

组成 Draft ForwardBatch。

当前代码甚至把 Draft ForwardBatch 的 `forward_mode` 设成：

~~~python
ForwardMode.TARGET_VERIFY
~~~

这说明这里不能用“普通 Decode batch”去理解它。它本质上是在一组预留 speculative positions 上做块状 forward。

### Draft Model Forward 先产生 hidden，再由 Markov Head / Sampler 形成 gamma 个候选

Draft forward：

~~~python
draft_out = self.draft_model_runner.forward(
    draft_forward_batch
)
~~~

得到 `raw_hidden`。非 folded proposal 路径随后：

~~~python
base_logits, confidence_tap = (
    self.draft_model.compute_base_logits(raw_hidden)
)

base_logits = base_logits.view(bs, gamma, -1)
~~~

再调用：

~~~python
sample_draft_block(
    base_logits=base_logits,
    anchor_tokens=draft_block_ids[:, 0],
    draft_hidden=fwd.draft_hidden_3d,
    markov_head=self.draft_model.markov_head,
    ...
)
~~~

`sample_draft_block()` 最终通过 Markov head 产生：

~~~text
draft_tokens       [B, gamma]
corrected_logits   可选
greedy_mask        [B]
temperatures       [B]
~~~

固定源码：[`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200)。

因此更准确的理解不是：

~~~text
Draft Model 连续 autoregressive forward gamma 次
~~~

而是：

> **DSpark 先构造一块 Draft query layout，Draft model 一次产生对应 hidden 表示，再由 proposal-side logits / Markov head 形成 gamma 个候选 Token。**

某些 graph 配置还可以把 proposal sampling 折叠进 captured path，此时 `proposal.folded=True`，但逻辑结果仍然是同一组 `draft_tokens`。

### Draft sampling 也必须在 TP Rank 之间一致

Greedy 路径：

~~~python
tp_sync.sync(
    SpecTpSyncSite.DSPARK_DRAFT_GREEDY,
    torch.argmax(step_logits, dim=-1),
)
~~~

Sampling 路径同样在显式 site 上同步。

因此 Draft proposal 不只是模型计算，还包含 speculative decision synchronization。这个边界在上一篇 TP/DP/EP Layout 文章中已经详细展开。

## 三、Verify Planner：不是所有请求都一定验证同样多的候选

Draft 得到候选以后，`DSparkWorkerV2` 会先解析 confidence：

~~~python
confidence = proposal.confidence
if confidence is None:
    confidence = self._verify_planner.compute_confidence_tensor(...)
~~~

然后：

~~~python
verify_token_budget = (
    self._verify_planner.resolve_verify_token_budget(...)
)

layout = self._verify_planner.schedule_layout(...)
~~~

所以 Target Verify 之前还有一层独立的：

~~~text
Verify Planning
~~~

### Static Verify：每请求使用固定 verify width

最容易理解的模式是固定链宽。假设：

~~~text
gamma = 4
verify_num_draft_tokens = 5
~~~

那么每个请求的 candidate chain 可以理解成：

~~~text
[anchor, draft1, draft2, draft3, draft4]
~~~

其中 anchor 来自当前 `draft_block_ids[:, :1]`。

实际 Verify IDs 在 `_forward_decode()` 里明确构造：

~~~python
verify_ids_2d = torch.cat(
    [draft_block_ids[:, :1], draft_tokens],
    dim=1,
).contiguous()
~~~

源码：[`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L818-L820)。

### Compact / Ragged Verify：不同请求可以验证不同长度

Planner 还支持 compact ragged layout。它可以根据 confidence、budget 等信息得到：

~~~text
R1 verify_len = 2
R2 verify_len = 5
R3 verify_len = 3
~~~

然后只把需要验证的 rows compact 到实际 Target Verify 输入中。

`schedule_layout()` 会生成 `RaggedVerifyLayout`，而 `BuildRaggedVerifyWindow` / `compact_verify_ids()` 把：

~~~text
anchor + selected draft prefix
~~~

重新打包成 compact verify rows。

Target forward 结束后，再通过 `ScatterCompactToStrided` 把 logits 和 hidden states 恢复回统一的 `[bs × stride]` 逻辑布局，方便后续 Accept。

因此：

> **Verify Planner 优化的是“这一轮 Target 需要真正验证多少 candidate rows”，不是重新改变 Draft Token 的语义。**

Planner 里的某些 DP-tier / graph-tier 优化有额外 topology 条件；这些限制应按具体路径理解，不能泛化为 DSpark 整体限制。

## 四、Target Verify：一次 forward 验证一条候选链，而不是逐 Token 再跑一遍 Target

进入 Verify 时，DSpark 会先分配一个 Verify Window：

~~~python
verify_window = alloc_verify_window(...)
~~~

里面包含候选位置和 staged cache locations。

非 compact 路径创建：

~~~python
DFlashVerifyInput(
    draft_token=verify_ids_2d.reshape(-1),
    positions=positions_2d.reshape(-1),
    draft_token_num=verify_w,
    capture_hidden_mode=CaptureHiddenMode.FULL,
)
~~~

随后构造 verify ForwardBatch，并调用：

~~~python
self.target_worker.forward_batch_generation(
    batch=None,
    forward_batch=verify_forward_batch,
    is_verify=True,
)
~~~

固定源码：[`TargetVerifyExecutor.run_non_compact()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L277-L352)。

关键点是：

> **Target 不再为 draft1、draft2、draft3、draft4 分别进入四次普通 Decode loop，而是用一个 Target Verify forward 对这条 candidate chain 计算对应 logits / hidden states。**

底层 Attention 仍然必须尊重链式因果关系；“一次 verify”不代表 candidate positions 彼此独立。

### Verify 输出为什么还要求 hidden states

普通生成最关心：

~~~text
logits → next token
~~~

DSpark Verify 则显式要求：

~~~text
logits
+
hidden_states
~~~

源码在 `commit_hidden()` 中甚至直接检查：

~~~python
if hidden is None:
    raise RuntimeError(
        "DSpark verify requires target hidden states, got None."
    )
~~~

原因在于：被 Target 确认的 hidden state 后面要注入 Draft 的持久状态，让下一轮 Draft 从 Target 已确认的上下文继续，而不是从错误 proposal 继续。

## 五、Accept / Reject：Greedy 是前缀匹配，Sampling 不能简化成“相等就接受”

Target Verify 得到 logits 后进入：

~~~python
accept = self._verify_executor.accept_and_finalize(...)
~~~

固定源码：[`accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206)。

这里必须把 greedy 和 sampling 分开。

### Greedy：检查 Draft 是否与 Target 的链式 argmax 前缀一致

Greedy 路径先把 target logits 做 argmax：

~~~python
target_predict = argmax(target_logits)
~~~

然后：

~~~text
Draft candidates:
A B C D

Target chain predictions:
A B X ...
~~~

则前两个 Draft Token 连续匹配，第三个开始不匹配：

~~~text
correct_len = 2
bonus = X
~~~

`correct_len` 表示连续接受的 Draft Token 数。

`bonus` 则是 Target 在第一个未接受位置上给出的 Target Token；如果全部 Draft 都被接受，bonus 是 Target 在已接受 Draft 链之后的下一枚预测。

源码：[`AcceptGreedy`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L560-L718)。

### Sampling：不是 token equality，而是 speculative sampling acceptance

如果请求不是纯 greedy，DSpark 会同时保留 Draft 侧的 `corrected_logits` / probabilities，并构造 Target probabilities。

随后通过 chain speculative sampling：

~~~python
chain_speculative_sampling_triton(...)
~~~

得到：

~~~text
correct_len
bonus
cap_trim_lens
~~~

因此 sampling 模式不能写成：

~~~text
Draft token == Target token
→ accept
~~~

更准确的是：

> **Sampling 路径使用 Draft 与 Target 两侧概率进行 speculative accept/reject，以保持目标采样分布；greedy 才可以直观理解成连续前缀匹配。**

固定源码：[`AcceptSampling`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)。

### `correct_len`、`bonus`、`commit_lens` 是三个不同概念

这是理解 DSpark 状态推进最关键的几个量。

`FinalizeAcceptLens` 明确：

~~~python
commit_lens = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
~~~

源码：[`finalize_accept_lens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L720-L780)。

为什么总是 `+1`？

因为一次 successful verify 不只是提交：

~~~text
accepted Draft prefix
~~~

还会提交：

~~~text
Target bonus token
~~~

例如：

~~~text
Draft:
A B C D

Target:
A B X ...

correct_len = 2
bonus       = X
commit_lens = 3
~~~

最终有效输出前缀是：

~~~text
A B X
~~~

`BuildOutTokens` 的实现也是先复制 Draft Tokens，再把 `bonus` scatter 到 `correct_len` 对应位置：

~~~python
out_tokens[:, :gamma].copy_(draft_tokens)
out_tokens.scatter_(
    1,
    correct_len[:, None],
    bonus[:, None],
)
~~~

只有前 `commit_lens` 个位置属于本轮真正推进的输出。

如果四个 Draft 全部接受：

~~~text
correct_len = 4
commit_lens = 5
~~~

一轮 Target Verify 可以让请求前进 5 个 Token：4 个 Draft + 1 个 Target bonus。

这正是 speculative decoding 能减少 Target decode 轮数的核心。

### Accept 结果也必须在 TP group 中达成一致

Eager accept 路径会对：

~~~text
correct_len
bonus
cap_trim_lens
~~~

调用 `SpecTpSync`。

否则不同 TP Rank 如果对接受长度意见不一致，后续：

~~~text
new_seq_lens
positions
cache state
bonus token
~~~

都会立即分叉。

## 六、Commit：真正保留的是接受前缀，Rejected candidate 不能成为下一轮 Draft 历史

Accept 完成后，DSpark 先得到：

~~~text
correct_len
bonus
commit_lens
new_seq_lens
out_tokens
~~~

然后才进入状态提交。

### Target Verify hidden 会按 `commit_lens` 注入 Draft 持久状态

非 folded commit 路径调用：

~~~python
self._verify_executor.commit_hidden(
    ...
    commit_lens=accept.commit_lens,
)
~~~

固定调用位置：[`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L919-L938)。

`TargetVerifyExecutor.commit_hidden()` 对普通 strided verify：

~~~python
self.kv_injector.inject_target_hidden(
    target_hidden=hidden.reshape(...),
    cache_loc=verify_window.verify_cache_loc,
    positions=verify_window.positions_2d.reshape(-1),
    commit_lens=commit_lens,
    ...
)
~~~

源码：[`commit_hidden()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L354-L396)。

这一步最准确的理解是：

> **把 Target Verify 已确认前缀对应的 Target hidden states 注入 Draft 模型的持久 KV / state，让下一轮 Draft 从 Target 已确认的状态继续。**

它不能简单描述成“把 Target 的所有 KV commit 进去”。Target Verify 本身已有自己的 staged verify cache；这里讨论的是 Target hidden → Draft persistent state 的注入。

`commit_lens` 起 gate 作用：未被接受的 candidate rows 不应该被注入 Draft 的下一轮有效状态。

某些 backend / graph 路径支持 folded commit：Accept 和 hidden injection 可以直接在 Verify epilogue 内完成；Python 控制流会缩短，但语义仍然是同一个 accepted-prefix commit。

### Mamba / KDA 类状态还有独立 commit

`_forward_decode()` 还会调用：

~~~python
self._commit_target_mamba_states_after_verify(...)
~~~

根据 `commit_lens` 选择最后一个被接受 verify step 对应的状态。

这提醒我们：speculative commit 不只是一份 KV Tensor 的生命周期问题。模型如果还有 recurrent / state-space 辅助状态，也必须和 accepted prefix 同步推进。

### 下一轮从新的 bonus token 和 new_seq_lens 开始

最后：

~~~python
next_draft_input = make_next_draft_input(
    bonus_tokens=accept.bonus,
    new_seq_lens=accept.new_seq_lens,
)
~~~

并返回：

~~~text
next_token_ids = accept.out_tokens
accept_lens    = accept.commit_lens
new_seq_lens  = accept.new_seq_lens
next_draft_input
~~~

所以下一轮 DSpark 不是从“上轮最后一个 Draft 猜测”开始，而是从 Target Verify 最终确认后的 bonus / sequence state 继续。

完整状态机可以画成：

~~~mermaid
flowchart TD
    S[Confirmed sequence state]
    S --> B[Bonus token]
    B --> D[Draft block / Draft forward]
    D --> P[gamma draft tokens]
    P --> L[Verify planner]
    L --> V[Target verify forward]
    V --> A[Accept / reject]
    A --> C[correct_len + bonus]
    C --> F[commit_lens = correct_len + 1]
    F --> H[Inject accepted target hidden into Draft state]
    H --> N[new_seq_lens + next bonus]
    N --> S2[Next speculative round]
~~~

到这里，就能重新回答标题里的三个动作。

**Draft** 做的不是让小模型永久接管生成，而是提出一条 candidate chain。

**Verify** 不是逐 Token 重跑普通 Target Decode，而是让 Target 用一次 verify forward 对候选链计算 logits / hidden states。

**Accept/Reject** 也不只是决定哪些 Token 输出给用户，它同时决定：

~~~text
本轮 sequence 前进多少
哪些 Target hidden 可以成为下一轮 Draft 状态
下一轮 bonus token 是谁
哪些 speculative rows 从有效状态中被排除
~~~

所以 speculative decoding 真正优化的不是一个 sampling 函数，而是把：

~~~text
Draft proposal
Target validation
State commit
~~~

组合成一个能够一次推进多 Token、同时保持 Target 语义正确的状态机。

这也解释了为什么上一篇 TP≠DP 问题会复杂：一旦 distributed layout 变化，需要保持一致的不只是 Draft hidden states，而是整套 Draft、Verify、Accept、Commit 状态。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| DSpark 总入口 | [`forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L555-L578) |
| 一轮 speculative decode 主链 | [`_forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L729-L979) |
| Draft proposer | [`DraftBlockProposer.propose()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L242-L344) |
| Draft ForwardBatch | [`_run_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L369-L473) |
| Draft block sampling | [`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200) |
| Verify Planner | [`DSparkVerifyPlanner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py) |
| Non-compact Target Verify | [`run_non_compact()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L277-L352) |
| Compact Target Verify | [`run_compact()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L436-L504) |
| Accept / Finalize | [`accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206) |
| Greedy Accept | [`AcceptGreedy`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L560-L718) |
| Sampling Accept | [`AcceptSampling`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py) |
| `commit_lens = correct_len + 1` | [`FinalizeAcceptLens`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L720-L780) |
| 输出 Token 构造 | [`BuildOutTokens`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py#L802-L911) |
| Target hidden → Draft state commit | [`commit_hidden()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L354-L396) |
| Speculative TP decision sync | [`SpecTpSync`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py) |
