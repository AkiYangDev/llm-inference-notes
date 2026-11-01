# DeepSeek-V4 DSpark：一轮 Draft、Verify、Accept 与 Commit

当前边界 Token 是 R，Draft 提出 A、B、C。Target 接受 A、B，随后给出 X。本轮输出是 A、B、X，能提交的输入状态却是 R、A、B：X 还没有自己的 KV，C 虽然算过，也不能留下来当正式历史。

DSpark 的一轮就围绕这个错位展开。Draft 一次提出候选块，Planner 安排验证长度，Target 带着旧 R 验证候选。Accept 找出分歧位置，Commit 再把有效输入的状态接回下一轮。先把普通候选链走通，树结构和并行布局另有专题。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。本文跟踪 DSpark V2、DeepSeek-V4 target / draft，涉及 greedy、sampling、static、compact/ragged，以及 folded proposal / verify epilogue。

</details>

## 一、一轮多枚输出怎样减少 Target 轮数

普通自回归中，Target 这一轮的采样结果，要到下一轮才作为输入：

~~~text
Token t
   ↓
Target Model Forward
   ↓
sample Token t+1
   ↓
下一轮 Target Forward
~~~

这条链生成 100 枚 Token，就要反复运行层栈、读取历史、完成通信。首枚若在 Prefill 产生，后面的 Decode 调用数会比总输出数少。每轮新 Query 很少，旧历史和权重仍需处理。

投机解码让较轻的 Draft 先提一段候选，再交 Target 统一验证。Greedy 接受与 Target 逐步预测相同的前缀；Sampling 还需要接受判定和分布校正，具体算法、配置决定输出分布是否保持。

对照两种生成过程：

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

候选准确，一次 Verify 就能推进多枚 Token，后续 Target 调用也随之减少。代价是验证窗口变宽，还多了 Draft、Accept、状态整理。接受长度说明一轮走多远，周期耗时和有效产出才说明这段路是否划算。

`forward_batch_generation()` 将 Prefill / Extend 交给 Target，后续 Decode 则进入 `_forward_decode()`。[`DSparkWorkerV2.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L555-L578)

Decode 的调用顺序为：

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

这几步都在 [`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L729-L979) 中串接。

## 二、Draft 怎样构造候选块

[`DraftBlockProposer.propose()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L242-L344) 返回本轮候选和提议信息，Worker 的调用为：

`proposal = self._proposer.propose(...)`。

`DraftBlockProposer` 同时处理候选数与 Query 行数：

~~~python
self.gamma = gamma
self.query_token_num = (
    self.gamma
    if sample_from_anchor
    else self.gamma + 1
)
~~~

`gamma` 数计划提出的候选 Token；`sample_from_anchor` 决定 Draft forward 输入 `gamma` 行还是 `gamma + 1` 行。

提 3 枚候选，Draft 可以算 3 行或 4 行，完整 Target Verify 则始终需要旧 anchor 加 3 枚候选，共 4 行。`sample_from_anchor=False` 跳过 Draft 的 anchor hidden，取后 gamma 行提议，Target 仍然要验证旧 anchor。

### Draft Block 的第一列来自上一轮 bonus token

候选块在 [`DraftBlockProposer._run_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L369-L473) 中准备。

`draft_block_ids.shape = [bs, query_token_num]`，一条请求占一行。

buffer 初始填 mask token，每轮把第一列改为当前 bonus：

~~~python
draft_block_ids[:, 0].copy_(
    draft_input.bonus_tokens.view(-1)
)
~~~

于是单条输入近似 `[bonus, MASK, MASK, MASK, ...]`。

Verify Window 同时提供 positions 和 cache locations，这些数据一起组成 Draft ForwardBatch。

这里设置了 `ForwardMode.TARGET_VERIFY`，以投机窗口的形态执行。实际调用对象仍是 Draft runner；Target 还没有验收候选。

<a id="draft-model-forward-hidden-markov-head-sampler-gamma"></a>

### Draft Block 的 hidden 与 Markov 采样

Draft model 先计算整块 hidden：

~~~python
draft_out = self.draft_model_runner.forward(
    draft_forward_batch
)
~~~

非 folded proposal 将 `raw_hidden` 整理为：

~~~python
base_logits, confidence_tap = (
    self.draft_model.compute_base_logits(raw_hidden)
)

base_logits = base_logits.view(bs, gamma, -1)
~~~

接着调用采样器：

~~~python
sample_draft_block(
    base_logits=base_logits,
    anchor_tokens=draft_block_ids[:, 0],
    draft_hidden=fwd.draft_hidden_3d,
    markov_head=self.draft_model.markov_head,
    ...
)
~~~

[`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200) 返回：

~~~text
draft_tokens       [B, gamma]
corrected_logits   可选
greedy_mask        [B]
temperatures       [B]
~~~

Draft 主 forward 一次算完整块，Markov head 再逐位置修正 logits、采样，并使用上一位置 Token 作为条件。这样省掉了逐候选重复完整 Draft forward 的开销，采样之间仍有先后依赖。NextN 则会逐步调用 Draft decoder。[Markov block sampling](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/dspark.py#L52-L83)

部分 Graph 把 proposal sampling 一起捕获，返回 `proposal.folded=True`。采样、Markov 修正仍在执行，只是 Python 后处理少了；后续照样要取得候选和提议分布。

<a id="draft-sampling-tp-rank"></a>

### TP Rank 之间的 Draft sampling 同步

Greedy 经过显式同步 site：

~~~python
tp_sync.sync(
    SpecTpSyncSite.DSPARK_DRAFT_GREEDY,
    torch.argmax(step_logits, dim=-1),
)
~~~

Sampling 也有相应 site。

同一请求的相关 TP Rank 必须提出同一串候选，后面的 Verify 才有共同输入。site 是否发起通信，还取决于同步开关和组宽。

## 三、Planner 决定本轮验证范围

Worker 从提议中解析 confidence：

~~~python
confidence = proposal.confidence
if confidence is None:
    confidence = self._verify_planner.compute_confidence_tensor(...)
~~~

然后决定预算、布局：

~~~python
verify_token_budget = (
    self._verify_planner.resolve_verify_token_budget(...)
)

layout = self._verify_planner.schedule_layout(...)
~~~

候选提出以后，Planner 才按模式决定验证多少、怎样排列，执行器据此组 Target 输入。Confidence 在这里用于估计收益，真正的接受结果要等 Target 验证完成。

### Static Verify：每请求使用固定 verify width

先看固定宽度。本节将候选数设为 4，演示配置计数：

~~~text
gamma = 4
verify_num_draft_tokens = 5
~~~

每条请求输入 `[anchor, draft1, draft2, draft3, draft4]`。

anchor 来自 `draft_block_ids[:, :1]`。

[`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L818-L820) 将它与候选拼成 Verify IDs：

~~~python
verify_ids_2d = torch.cat(
    [draft_block_ids[:, :1], draft_tokens],
    dim=1,
).contiguous()
~~~

### Compact / Ragged Verify：不同请求可以验证不同长度

compact ragged 可按置信度和预算，给每条请求不同长度：

~~~text
R1 verify_len = 2
R2 verify_len = 5
R3 verify_len = 3
~~~

长度包含 anchor，2、5、3 合计 10 行。按固定 stride 5 则有 15 行。紧凑打包减少逻辑 Query，Graph tier 却可能再 padding，因此 10/15 只比较行数。

`schedule_layout()` 生成 `RaggedVerifyLayout`，`BuildRaggedVerifyWindow` / `compact_verify_ids()` 随后打包各请求的 anchor 和候选前缀。

Target 算完，`ScatterCompactToStrided` 把 logits、hidden 整理回 `[bs × stride]` 逻辑布局，供 Accept 按原位置寻址。

恢复布局并未补算跳过的 Query，有效范围仍由 `verify_lens` 指定。

cap-accept 则保留完整 Target 窗口，仅限制接受长度，所以没有省掉 Verify 行。设备支持也有区别：本文固定的 Ascend 910C DSV4 路径仍受定宽 metadata 和 Graph 准入限制，不能直接套用通用 compact。

DP-tier / graph-tier 分支还带有拓扑条件；这些分支的计数、Graph tier 选择，需要沿各自路径追踪。

## 四、Target 一次验证候选链

`alloc_verify_window(...)` 早在 Draft 之前就已执行。它预留候选位置和 staged 缓存目标，Draft、Target Verify 共用这些投机位置和 cache locations。

非 compact 模式先准备：

~~~python
DFlashVerifyInput(
    draft_token=verify_ids_2d.reshape(-1),
    positions=positions_2d.reshape(-1),
    draft_token_num=verify_w,
    capture_hidden_mode=CaptureHiddenMode.FULL,
)
~~~

再构造 ForwardBatch，交给 Target：

~~~python
self.target_worker.forward_batch_generation(
    batch=None,
    forward_batch=verify_forward_batch,
    is_verify=True,
)
~~~

调用位于 [`TargetVerifyExecutor.run_non_compact()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L277-L352)。

gamma=4 时，anchor 和四枚候选共 5 行进入同一次 Target forward。每行依旧按链式因果关系读取 Prefix、祖先，但不必再为四枚候选分别跑普通 Decode。

回到 R、A、B、C：R 行预测 A，A 行预测 B，B 行预测 X。第三枚候选 C 与 X 不同，接受在这里停止。后面 C 行即使算出了 logits、hidden，也不能跨过分歧提交。

### Verify 输出为什么还要求 hidden states

接受判定要 logits，后续状态注入要 hidden，Verify 因此同时保留两者。

`commit_hidden()` 检查输出是否带有 hidden：

~~~python
if hidden is None:
    raise RuntimeError(
        "DSpark verify requires target hidden states, got None."
    )
~~~

接受 R、A、B 对应的输入行后，它们的 Target hidden 会更新 Draft。如果只把输出改成 A、B、X，却留下 C 对应的 Draft 状态，下轮就会接上错误历史。

## 五、Greedy 与 Sampling 怎样接受候选

接受与长度整理位于 [`accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206)。

Worker 通过 `accept = self._verify_executor.accept_and_finalize(...)` 取得结果。

Greedy 逐项匹配 Token，Sampling 使用提议和 Target 概率执行接受规则。两条路径最后都需要给出输出与长度，判定过程各不相同。

### Greedy：检查 Draft 是否与 Target 的链式 argmax 前缀一致

Greedy 取 `target_predict = argmax(target_logits)`，与候选逐步比较：

~~~text
Draft candidates:
A B C D

Target chain predictions:
A B X ...
~~~

前两枚匹配，第三枚分歧，结果为：

~~~text
correct_len = 2
bonus = X
~~~

`correct_len=2` 数接受的 Draft；`bonus=X` 是最后有效输入 B 的预测，用来接替 C。若 Draft 全部匹配，bonus 就取最后一行的预测。接受在哪里结束，bonus 就从哪里取得。

实现位于 [`AcceptGreedy`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L560-L718)。

<a id="sampling-token-equality-speculative-sampling-acceptance"></a>

### Sampling 使用提议概率与 Target 概率

Sampling 要保留 Draft 实际使用的提议分布。Markov 修正改变了 logits，因此接受时使用 `corrected_logits` / probabilities，修正前的 `base_logits` 不能代替它。

`chain_speculative_sampling_triton(...)` 用这些分布算出 `correct_len`、`bonus`、`cap_trim_lens`，再交给长度整理和状态提交。

Sampling 的分布判定和 Greedy 的前缀匹配，到这里收敛为相同的结果接口。

概率路径见 [`AcceptSampling`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)。

### `correct_len`、`bonus`、`commit_lens` 是三个不同概念

`correct_len` 是接受的 Draft 数，`bonus` 是 Token ID，`commit_lens` 是推进长度。后续的序列边界计算用长度，生成下轮 anchor 才用 Token ID。

`FinalizeAcceptLens` 按下面的关系整理：

~~~python
commit_lens = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
~~~

函数入口为 [`finalize_accept_lens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L720-L780)。

输出与 KV-ready frontier 推进同样多的位置，两组 Token 却相差一位：本例输出 `[A, B, X]`，算过输入 KV 的是 `[R, A, B]`。所以 `commit_lens=3` 时，X 的 KV 仍未形成。

这个 `+1` 对应 Target 给出的 fallback / bonus。Draft 一枚都没通过，root 也能预测一枚输出；上层还会按 EOS、stop、长度限制裁剪。

将本例代入：

~~~text
Draft:
A B C D

Target:
A B X ...

correct_len = 2
bonus       = X
commit_lens = 3
~~~

输出有效前缀为 `A B X`。

`BuildOutTokens` 复制候选，并在 `correct_len` 位置放入 bonus：

~~~python
out_tokens[:, :gamma].copy_(draft_tokens)
out_tokens.scatter_(
    1,
    correct_len[:, None],
    bonus[:, None],
)
~~~

其中前 `commit_lens` 个位置用于本轮输出。

如果四枚候选全部通过：

~~~text
correct_len = 4
commit_lens = 5
~~~

这一轮最多推进 5 个 Token：4 个 Draft 加 1 个 Target bonus。

这是窗口的推进上限。统计 AL 时要说明是否含 bonus；统计严格 Draft 接受率，则要有同窗口实际提出、接受的候选数，最大窗口宽度不能代替实测分母。

### Accept 结果也必须在 TP group 中达成一致

Eager 将 `correct_len`、`bonus`、`cap_trim_lens` 交给 `SpecTpSync` 的对应 site。相关 Rank 的接受数、末尾 ID、裁剪范围只要有一项不同，后续 `new_seq_lens`、positions、cache state 就可能分叉。

因此同一次接受同步携带的不只是长度，还有末尾 ID 与有效裁剪。

## 六、提交有效状态并开始下一轮

Accept 整理出 `correct_len`、`bonus`、`commit_lens`、`new_seq_lens`、`out_tokens`，Commit 再根据有效范围更新状态。

### Target Verify hidden 会按 `commit_lens` 注入 Draft 持久状态

非 folded commit 的调用为：

~~~python
self._verify_executor.commit_hidden(
    ...
    commit_lens=accept.commit_lens,
)
~~~

位置见 [`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L919-L938)。

普通 strided Verify 进入 `commit_hidden()`：

~~~python
self.kv_injector.inject_target_hidden(
    target_hidden=hidden.reshape(...),
    cache_loc=verify_window.verify_cache_loc,
    positions=verify_window.positions_2d.reshape(-1),
    commit_lens=commit_lens,
    ...
)
~~~

函数实现见 [`commit_hidden()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L354-L396)。

Draft 使用 Target hidden 更新自己的持久状态：先投影、归一化，再经 Draft 各 stage 的 KV 投影写缓存。`commit_lens` 指定提交哪些输入行，并没有原样复制 Target KV。[`write_target_hidden_kv()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L932-L968)

假设原缓存前缀为 10，新增 R、A、B 三个输入以后，KV-ready 边界到 13。X 成为下轮 anchor，等待计算自己的 Target KV；C 的预留空间即使还在，也不属于正式历史。

部分 Graph 能将 Accept、hidden injection 放进 Verify epilogue。固定路径同时要求 proposal folded、Greedy、没有生效 logits adjustment、无 Grammar，且本轮确实能够 replay。Draft folded 只是其中一项；折叠执行之后，接受边界和注入范围仍须与原过程一致。

<details markdown="1">
<summary>补充：通用 Worker 中的 recurrent state 提交</summary>

通用 `_forward_decode()` 还会调用 `_commit_target_mamba_states_after_verify(...)`，按 `commit_lens` 选择验证步，为带 recurrent / state-space 状态的模型提交状态。

</details>

### 下一轮从新的 bonus token 和 new_seq_lens 开始

最后更新下轮输入：

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

下一轮接收 X、新的 `new_seq_lens` 和已更新的 Draft 状态。Token、长度、状态三者要对应同一段历史，才不会把 X 接到 C 的投机分支上。

这一轮的状态关系如下：

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

在这张图里，`correct_len` 决定分歧位置，`commit_lens` 决定推进范围，选中的 Verify 行决定注入内容。三者共同把本轮提议接到下一轮。

### 补充源码入口

- [Verify window 分配与布局](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py#L802-L911)
- [Planner 决策与宽度选择](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py)
- [Target hidden 提交](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L436-L504)
- [跨 TP 决策同步](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)
