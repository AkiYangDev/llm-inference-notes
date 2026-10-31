# DeepSeek-V4 DSpark：一轮 Draft、Verify、Accept 与 Commit

当前边界 Token 是 R，Draft 提出 A、B、C。Target 连续接受 A、B，接着给出 X，于是本轮输出为 A、B、X，能够提交的输入状态却是 R、A、B。C 即使已经计算、写入过临时缓存，也不能进入下一轮的正式历史。

下面沿这条链走过 Draft、Verify Planning、Target Verify、Accept、Commit：先看 Draft 怎样一次计算候选块，再看旧 R 为什么仍要进入验证，以及哪些 Target hidden 最后写回 Draft。树式候选和并行布局留到专题展开。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线采用 DSpark V2、DeepSeek-V4 target / draft 和普通候选链，分别说明 greedy、sampling、static、compact/ragged 与 folded proposal / verify epilogue。性能仍需结合接受长度、Draft / Verify 周期、Batch、Graph 和通信判断。

</details>

## 一、一轮多枚输出怎样减少 Target 轮数

普通自回归先运行 Target，再把本次采样结果作为下一轮输入：

~~~text
Token t
   ↓
Target Model Forward
   ↓
sample Token t+1
   ↓
下一轮 Target Forward
~~~

生成 100 枚 Token 需要反复推进这条链，但首枚若来自 Prefill，Decode 调用数就不等于总输出数。每轮虽只有少量新 Query，仍需跑过层栈、读历史并完成相关通信。

投机路径先由较轻 Draft 提候选，再统一交 Target。Greedy 按 Target 的逐步预测匹配前缀，概率采样另需接受与分布校正。是否保持输出分布，要看具体算法和配置，不能单凭启用投机解码判断。

两种生成方式可以先对照为：

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

准确的候选能让一次 Verify 推进多于 1 个 Token，减少后续 Target 轮次。不过验证窗口有多行，又新增 Draft、Accept 和状态整理，单轮可能更贵。因此接受长度还需和同负载的耗时、有效产出一起看。

入口按模式分流：Prefill / Extend 先走 Target，`_forward_decode()` 承接后续投机循环。[`DSparkWorkerV2.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L555-L578)

Decode 内部顺序如下：

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

对应调用实现见 [`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L729-L979)。

## 二、Draft 怎样构造候选块

Draft 从以下调用开始，具体实现见 [`DraftBlockProposer.propose()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L242-L344)。

`proposal = self._proposer.propose(...)` 返回本轮候选及相关提议信息。



`DraftBlockProposer` 中先区分两个宽度：

~~~python
self.gamma = gamma
self.query_token_num = (
    self.gamma
    if sample_from_anchor
    else self.gamma + 1
)
~~~

- `gamma` 是计划提出的候选 Token 数；
- Draft forward 的 Query 行数由 `sample_from_anchor` 决定，可为 `gamma` 或 `gamma + 1`。

例如提 3 枚候选，Draft 可输入 3 或 4 行，完整 Target Verify 却仍需旧 anchor 加 3 枚候选，共 4 行。`sample_from_anchor=False` 让 Draft 跳过 anchor hidden、取后 gamma 行提议，不会取消 Target 对旧 anchor 的验证。

### Draft Block 的第一列来自上一轮 bonus token

`_run_forward()` 准备下面的块，入口见 [`DraftBlockProposer._run_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L369-L473)。

块的形状为 `draft_block_ids.shape = [bs, query_token_num]`。

buffer 先填 mask token，每轮更新第一列：

~~~python
draft_block_ids[:, 0].copy_(
    draft_input.bonus_tokens.view(-1)
)
~~~



一条请求的输入因此近似 `[bonus, MASK, MASK, MASK, ...]`。

再结合 Verify Window 预备的 positions、cache locations，共同构成 Draft ForwardBatch。

本次 Draft ForwardBatch 设置 `ForwardMode.TARGET_VERIFY`，描述预留投机位置的执行形态。此处调用的仍是 Draft runner，不能将这个模式名理解为已经完成 Target 验收。

### Draft Model Forward 先产生 hidden，再由 Markov Head / Sampler 形成 gamma 个候选

Draft model 先执行：

~~~python
draft_out = self.draft_model_runner.forward(
    draft_forward_batch
)
~~~

得到 `raw_hidden` 后，非 folded proposal 继续准备：

~~~python
base_logits, confidence_tap = (
    self.draft_model.compute_base_logits(raw_hidden)
)

base_logits = base_logits.view(bs, gamma, -1)
~~~

再交给：

~~~python
sample_draft_block(
    base_logits=base_logits,
    anchor_tokens=draft_block_ids[:, 0],
    draft_hidden=fwd.draft_hidden_3d,
    markov_head=self.draft_model.markov_head,
    ...
)
~~~

Markov head / sampler 生成以下结果，实现见 [`sample_draft_block()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L130-L200)。

~~~text
draft_tokens       [B, gamma]
corrected_logits   可选
greedy_mask        [B]
temperatures       [B]
~~~



Draft 主 forward 一次算整块 hidden，之后 Markov head 按位置修正 logits、采样，并以上一位置 Token 作为条件。省去的是逐候选重复完整 Draft forward，采样本身仍有顺序依赖。这与 NextN 逐步调用 Draft decoder 不同。[Markov block sampling](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/dspark.py#L52-L83)

部分 Graph 将 proposal sampling 折入 captured path，返回 `proposal.folded=True`。Python 后处理变少，采样和 Markov 修正仍会执行，后续也仍需候选及提议分布。

### Draft sampling 也必须在 TP Rank 之间一致

Greedy 在显式同步 site 处理：

~~~python
tp_sync.sync(
    SpecTpSyncSite.DSPARK_DRAFT_GREEDY,
    torch.argmax(step_logits, dim=-1),
)
~~~

Sampling 同样有对应 site。

候选需在同一请求的相关 TP Rank 保持一致，否则后面的 Verify 就不再处理同一序列。site 是否真的通信，还受同步开关与组宽影响。

## 三、Planner 决定本轮验证范围

提议完成后，Worker 先解析 confidence：

~~~python
confidence = proposal.confidence
if confidence is None:
    confidence = self._verify_planner.compute_confidence_tensor(...)
~~~

再决定预算与布局：

~~~python
verify_token_budget = (
    self._verify_planner.resolve_verify_token_budget(...)
)

layout = self._verify_planner.schedule_layout(...)
~~~

因此提了多少候选，不等于本轮一定验证多少。Planner 根据模式处理 Confidence、预算和布局，执行器据此组 Target 输入。Confidence 估计收益，尚不是 Target 接受结果。

### Static Verify：每请求使用固定 verify width

先看固定宽度。开头用 3 枚候选讲状态边界，下面另以 4 枚说明配置计数：

~~~text
gamma = 4
verify_num_draft_tokens = 5
~~~

每请求窗口为 `[anchor, draft1, draft2, draft3, draft4]`。

anchor 取自 `draft_block_ids[:, :1]`。

Worker 按下列方式构造 Verify IDs，源码见 [`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L818-L820)。

~~~python
verify_ids_2d = torch.cat(
    [draft_block_ids[:, :1], draft_tokens],
    dim=1,
).contiguous()
~~~



### Compact / Ragged Verify：不同请求可以验证不同长度

compact ragged 则可按置信度、预算给各请求不同窗口：

~~~text
R1 verify_len = 2
R2 verify_len = 5
R3 verify_len = 3
~~~

长度含 anchor，2、5、3 共 10 行；若都按 stride 5 执行则有 15 行。紧凑打包减少有效 Query 行，Graph tier 仍可能 padding，不能把 10/15 直接当作耗时比例。

`schedule_layout()` 形成 `RaggedVerifyLayout`，再经 `BuildRaggedVerifyWindow` / `compact_verify_ids()` 打包各请求的 anchor 与所选候选前缀。

Target 完成后，`ScatterCompactToStrided` 将 logits、hidden 恢复到 `[bs × stride]` 逻辑布局，供 Accept 寻址。

恢复形状不会补算跳过的 Query，`verify_lens` 仍决定哪些结果有效。

cap-accept 只限制接受长度，Target 依然执行固定窗口，不能据此宣称省了 Verify 行。通用 compact 支持也不代表全部设备可用：固定 Ascend 910C DSV4 仍受定宽 metadata、Graph 准入限制。

部分 DP-tier / graph-tier 分支还检查额外拓扑条件，应分别追踪计数和 Graph tier 的实际选择。

## 四、Target 一次验证候选链

`verify_window = alloc_verify_window(...)` 实际发生在 Draft 之前，保存候选位置和 staged 缓存目标。Draft 与 Target Verify 共用其中的投机位置和 cache locations。

非 compact 模式准备：

~~~python
DFlashVerifyInput(
    draft_token=verify_ids_2d.reshape(-1),
    positions=positions_2d.reshape(-1),
    draft_token_num=verify_w,
    capture_hidden_mode=CaptureHiddenMode.FULL,
)
~~~

构造 ForwardBatch 后交给 Target：

~~~python
self.target_worker.forward_batch_generation(
    batch=None,
    forward_batch=verify_forward_batch,
    is_verify=True,
)
~~~

执行入口见 [`TargetVerifyExecutor.run_non_compact()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L277-L352)。

gamma=4 时，Target 一次处理 anchor 加候选的 5 行，无需对四枚 Draft 分别再跑普通 Decode。各行仍依链式因果访问 Prefix 和祖先，统一 forward 并未消除上下文依赖。

在开头的例子里，R 行预测 A，A 行预测 B，B 行预测 X；候选 C 与 X 分歧，于是接受在此停止。C 行即使已产生 logits 和 hidden，也不能越过这处分歧提交。

### Verify 输出为什么还要求 hidden states

Logits 用于接受决策，hidden 则用于后续状态注入，所以 Verify 必须保留两者。

`commit_hidden()` 会检查：

~~~python
if hidden is None:
    raise RuntimeError(
        "DSpark verify requires target hidden states, got None."
    )
~~~

接受后，R、A、B 等有效输入行的 Target hidden 用于更新 Draft。若只把输出改成 A、B、X，却保留 C 对应的 Draft 状态，下轮历史仍会出错。

## 五、Greedy 与 Sampling 怎样接受候选

Verify 后进入接受与整理，实现见 [`accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206)。

Worker 调用 `accept = self._verify_executor.accept_and_finalize(...)` 取得本轮接受结果。



Greedy 比较连续 Token 匹配，Sampling 比较提议和 Target 概率并执行相应接受规则。它们共享输出、长度契约，判定方法不同。

### Greedy：检查 Draft 是否与 Target 的链式 argmax 前缀一致

Greedy 先取 `target_predict = argmax(target_logits)`，再与候选链逐步对照：

~~~text
Draft candidates:
A B C D

Target chain predictions:
A B X ...
~~~

前两枚匹配，第三枚分歧：

~~~text
correct_len = 2
bonus = X
~~~

`correct_len=2` 只数已接受 Draft，`bonus=X` 是最后有效输入 B 的预测，用来接替 C。若全部 Draft 匹配，bonus 则来自最后一行，因此它随接受终点移动，不能固定读窗口末行。

对应实现见 [`AcceptGreedy`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L560-L718)。

### Sampling：不是 token equality，而是 speculative sampling acceptance

Sampling 需用 Draft 实际提议分布与 Target probabilities。Markov 修正会改变 logits，所以应保留 `corrected_logits` / probabilities，不能以修正前 `base_logits` 代替。

`chain_speculative_sampling_triton(...)` 根据这些分布产生 `correct_len`、`bonus`、`cap_trim_lens`，交给后续长度整理与状态提交。

概率路径按 Draft / Target 概率接受或拒绝；Greedy 按预测匹配前缀，两者各自产生接受长度与新 bonus。

固定源码：[`AcceptSampling`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)。

### `correct_len`、`bonus`、`commit_lens` 是三个不同概念

`correct_len` 数 Draft，`bonus` 是 Token ID，`commit_lens` 表示推进长度。把 ID 当位置、或把接受数直接当提交数，都会让下轮缓存边界错位。

`FinalizeAcceptLens` 的关系为：

~~~python
commit_lens = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
~~~

固定函数见 [`finalize_accept_lens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L720-L780)。

同一个计数会推进输出和 KV-ready frontier，但两组 Token 相差一位。本例输出 `[A, B, X]`，已计算输入 KV 则是 `[R, A, B]`；`commit_lens=3` 不意味着 X 的 KV 已形成。

这个 `+1` 包含新的 Target bonus。即使未接受任何 Draft，root 预测仍可给一枚输出；EOS、stop、长度截断还需上层按各自规则处理。

代入可得：

~~~text
Draft:
A B C D

Target:
A B X ...

correct_len = 2
bonus       = X
commit_lens = 3
~~~

有效输出为 `A B X`。

`BuildOutTokens` 先复制候选，再在 `correct_len` 位置放入 bonus：

~~~python
out_tokens[:, :gamma].copy_(draft_tokens)
out_tokens.scatter_(
    1,
    correct_len[:, None],
    bonus[:, None],
)
~~~

只有前 `commit_lens` 个位置用于本轮输出推进。

四枚候选全部通过时：

~~~text
correct_len = 4
commit_lens = 5
~~~

最多可推进 5 个 Token，即 4 个 Draft 加 1 个 Target bonus。

这是全部接受时的上限。记录 AL 应说明是否含 bonus；严格 Draft 接受率需用同窗口的实际接受与提议计数，不能以最大窗口宽度冒充实测分母。

### Accept 结果也必须在 TP group 中达成一致

Eager 路径将 `correct_len`、`bonus`、`cap_trim_lens` 交给 `SpecTpSync` 的相应 site。相关 Rank 若接受数、末尾 ID 或裁剪范围不一致，后续 `new_seq_lens`、positions、cache state、bonus token 都可能分叉。

因此核对时既要对齐长度，也要对齐末尾 ID 和有效裁剪。

## 六、提交有效状态并开始下一轮

Accept 先确定 `correct_len`、`bonus`，整理出 `commit_lens`、`new_seq_lens` 和 `out_tokens`，随后才按有效范围提交状态。

### Target Verify hidden 会按 `commit_lens` 注入 Draft 持久状态

非 folded commit 调用：

~~~python
self._verify_executor.commit_hidden(
    ...
    commit_lens=accept.commit_lens,
)
~~~

调用处见 [`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L919-L938)。

普通 strided Verify 下，`commit_hidden()` 执行：

~~~python
self.kv_injector.inject_target_hidden(
    target_hidden=hidden.reshape(...),
    cache_loc=verify_window.verify_cache_loc,
    positions=verify_window.positions_2d.reshape(-1),
    commit_lens=commit_lens,
    ...
)
~~~

实现见 [`commit_hidden()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L354-L396)。

这一步由 Target hidden 更新 Draft 持久状态，并非原样复制 Target KV。V4 DSpark 先投影、归一化 hidden，再用 Draft 各 stage 的 KV 投影写缓存，`commit_lens` 限定可提交输入行。[`write_target_hidden_kv()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L932-L968)

若原已缓存前缀为 10，本例新增 R、A、B 三个输入，KV-ready 边界到 13。X 成为下轮 anchor，尚无自己的 Target KV；C 即使仍占预留空间，也不属于已提交历史。

部分 Graph 支持将 Accept、hidden injection 折入 Verify epilogue，但固定路径还要求 proposal folded、Greedy、无生效 logits adjustment、无 Grammar，并实际能 replay。不能只见 Draft folded 就认定 Sampling 或 Grammar 的接受提交也已折叠；有效边界和注入范围仍须相同。

<details markdown="1">
<summary>补充：通用 Worker 中的 recurrent state 提交</summary>

通用 `_forward_decode()` 另调用 `_commit_target_mamba_states_after_verify(...)`，按 `commit_lens` 选择验证步，服务于具有 recurrent / state-space 状态的模型。

</details>

### 下一轮从新的 bonus token 和 new_seq_lens 开始

最后准备下一轮输入：

~~~python
next_draft_input = make_next_draft_input(
    bonus_tokens=accept.bonus,
    new_seq_lens=accept.new_seq_lens,
)
~~~

随结果返回：

~~~text
next_token_ids = accept.out_tokens
accept_lens    = accept.commit_lens
new_seq_lens  = accept.new_seq_lens
next_draft_input
~~~

下一轮使用 X、`new_seq_lens` 和刚更新的 Draft 状态。只改 Token 不改状态，会把 X 接到错误候选历史；只改长度却保留 C，也会使 Target 从错误 anchor 开始。

完整关系可画成：

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

调试这条链时，将 `correct_len`、`commit_lens`、新长度和选中 Verify 行并排记录，便能核对提议、接受、注入是否对应同一段序列。

### 补充源码入口

- [Verify window 分配与布局](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py#L802-L911)
- [Planner 决策与宽度选择](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py)
- [Target hidden 提交](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L436-L504)
- [跨 TP 决策同步](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)
