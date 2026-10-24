# DeepSeek-V4 DSpark：一轮 Draft、Verify、Accept 与 Commit

假设当前已确认的边界 token 是 R，Draft 提出 A、B、C，Target 接受 A、B，却在下一位置预测 X。这一轮追加的输出是 A、B、X，实际可提交的输入状态却属于 R、A、B。候选 C 可能已经参与计算、写过临时缓存，但不能成为后续生成的正式历史。

DSpark 把这轮工作拆成 Draft、Verify Planning、Target Verify、Accept 和 Commit。下面沿这个例子看输入和状态怎样变化：Draft 的一次块计算与逐位置采样各做什么，Verify 为什么还需要旧 R，最后又把哪些 Target hidden 写回 Draft。并行布局和树式候选放在对应专题，这里先走通一条候选链。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线使用 DSpark V2、DeepSeek-V4 target / draft、普通链式 speculative decoding。源码同时支持 greedy、sampling、static verify、compact/ragged verify、folded proposal / verify epilogue 等优化；正文先解释共同语义，再标注优化路径。实际收益由接受长度、Draft / Verify 成本、Batch、Graph 与通信共同决定。

</details>

## 一、一轮多枚输出怎样减少 Target 轮数

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

如果最终要生成 100 枚 token，普通路径通常要反复推进 Target decode loop。计数时还要区分首枚输出是否由 Prefill 产生，不能把总输出数机械当成 Decode 调用数。每一轮虽然只新增少量 Query rows，却仍要执行完整层栈、读取历史状态，并完成配置所需的通信。

投机解码先用较轻的 Draft 路径提出候选，再由 Target 统一验证。Greedy 路径要求接受前缀与 Target 的逐步预测匹配；概率采样则还需要对应的接受与分布校正规则。减少 Target 轮数是执行目标，输出分布是否保持要依据实际算法与配置判断，不能仅凭“使用了投机解码”下结论。

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

Draft 猜得足够准时，一次 Verify 可以推进多于 1 个 token，从而少跑后续 Target Decode。但 Verify 处理的是多行窗口，还多出了 Draft、Accept 和状态整理，单轮并不保证比一次普通 Decode 便宜。接受长度增加后，整段生成是否加速，仍要对照相同负载的总耗时与有效产出。

DSpark 当前总入口在 [`DSparkWorkerV2.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py#L555-L578)。Prefill / Extend 仍然先走 Target； speculative loop 从 `_forward_decode()` 开始。

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

## 二、Draft 怎样构造候选块

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

- `gamma` 是最终要提出的 Draft Token 数；
- Draft model forward 实际输入的 query rows 数可能是 `gamma`，也可能是 `gamma + 1`，取决于 `sample_from_anchor`。

例如要提出 3 个候选，Draft query 宽度可以是 3 或 4，但完整 Target Verify 窗口仍需容纳旧 anchor 加 3 个候选，共 4 行。`sample_from_anchor=False` 时，Draft 输出还会跳过 anchor 对应的 hidden 行，选取后面的 gamma 行用于提议；它控制 Draft hidden 的取用方式，不是决定是否验证旧 anchor。

### Draft Block 的第一列来自上一轮 bonus token

`_run_forward()` 先建立：

~~~python
draft_block_ids.shape = [bs, query_token_num]
~~~

buffer 初始化时填充 mask token，每轮只更新第一列边界 token：

~~~python
draft_block_ids[:, 0].copy_(
    draft_input.bonus_tokens.view(-1)
)
~~~

源码：[`DraftBlockProposer._run_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L369-L473)。

可以把一条请求的 Draft 输入想成：`[bonus, MASK, MASK, MASK, ...]`

再配上 Verify Window 预先准备的：

~~~text
positions
cache locations
~~~

组成 Draft ForwardBatch。

Draft ForwardBatch 的 `forward_mode` 在这里设为：

~~~python
ForwardMode.TARGET_VERIFY
~~~

虽然 ForwardMode 的枚举名叫 TARGET_VERIFY，这次调用的仍是 Draft model runner。枚举值描述它要处理一块预留的投机位置，不表示 Draft 已经替代完整 Target 做了最终验收。

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

这段路径里，Draft model 的主 forward 一次产生整块 hidden，Markov head 再逐位置修正 logits 并产生候选。Head 会把前一步 token 作为下一步条件，因此提议并非完全没有顺序依赖；减少的是为每枚候选重复执行一次完整 Draft forward，不是把所有采样都变成相互独立的操作。这与 NextN 逐步调用 Draft decoder 的生成方式不同，逐位置的 Head 循环可对照固定版本的 [Markov block sampling](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/dspark.py#L52-L83)。

某些 Graph 配置可以把 proposal sampling 折叠进 captured path，此时 `proposal.folded=True`。Python 中少了一段显式后处理，不等于采样和 Markov 修正没有发生；后续仍消费同样的 `draft_tokens` 与需要的提议分布信息。

### Draft sampling 也必须在 TP Rank 之间一致

Greedy 路径：

~~~python
tp_sync.sync(
    SpecTpSyncSite.DSPARK_DRAFT_GREEDY,
    torch.argmax(step_logits, dim=-1),
)
~~~

Sampling 路径同样在显式 site 上同步。

这些调用把候选决策纳入 TP 同步策略，是否实际通信还受同步配置与组大小影响。同一请求的相关 rank 必须沿相同候选继续，否则各 rank 后面的 Verify 输入就不再表示同一条序列。这里要求的是决策一致，不是让每个 rank 独立生成一条未来。

## 三、Planner 决定本轮验证范围

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

候选数量与本轮实际验证范围由此分开。Planner 先根据所用模式解析 Confidence、预算与布局，执行器再据此构造 Target 输入。Confidence 只是收益估计；一个候选分数高，不表示已经被 Target 接受。

### Static Verify：每请求使用固定 verify width

先看固定链宽。开头用 3 枚候选解释输出与状态偏移，下面另用 4 枚候选说明配置中的窗口计数：

~~~text
gamma = 4
verify_num_draft_tokens = 5
~~~

那么每个请求的 candidate chain 可以理解成：`[anchor, draft1, draft2, draft3, draft4]`

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

这些长度包含每条请求的 anchor。例子中三个有效输入前缀分别有 2、5、3 行，共 10 行；若固定 stride 为 5，原本需要安排 15 行窗口。compact 将有效 Query rows 紧凑打包，但 Graph replay 仍可能按捕获的 tier 添加 padding，不能直接把 10/15 当成耗时比例。

`schedule_layout()` 生成 `RaggedVerifyLayout`，再由 `BuildRaggedVerifyWindow` / `compact_verify_ids()` 将各请求的 `anchor + selected draft prefix` 打包成紧凑的 Verify rows。

Target forward 结束后，再通过 `ScatterCompactToStrided` 把 logits 和 hidden states 恢复回统一的 `[bs × stride]` 逻辑布局，方便后续 Accept。

把结果 scatter 回 `[bs × stride]`，是为了让后面的 Accept 能按固定位置寻址；恢复了 Tensor 形状，不代表此前被跳过的 Query 行又执行了一遍。`verify_lens` 仍限定哪些结果有效。

还有 cap-accept 模式：它可以限制各请求的接受长度，但 Target 仍执行固定宽度窗口，不能据此声称节省了 Verify 行。通用 Runtime 存在 compact 路径，也不代表每个设备后端都已支持；本文固定版本的 Ascend 910C DSV4 还受固定宽度 metadata 与 Graph 准入限制，部署时需按实际后端核对。

Planner 里的某些 DP-tier / graph-tier 优化有额外 topology 条件；它们分别控制对应的计数与 Graph Tier 分支。

## 四、Target 一次验证候选链

Verify Window 实际在本轮开始、Draft proposal 之前就已分配。Draft 与后续 Target Verify 都会用到它准备的投机位置和 cache locations，调用入口是：

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

Target 用一个 Verify forward 处理 anchor 加候选组成的窗口。gamma=4 时，这是 5 行输入，不是对四枚 Draft 分别再调用四次普通 Decode。各行仍按链式因果关系读取 Prefix 和祖先；把输入并到一次 Forward，并没有取消候选之间的上下文依赖。

用开头的链看每行职责：输入 R 的预测用于检查 A，输入 A 的预测用于检查 B，输入 B 的预测给出 X。C 与 X 不同，接受在这里停止；C 行即使已经算出了 logits 和 hidden，也不能越过这个分歧继续提交。

### Verify 输出为什么还要求 hidden states

logits 用于接受决策，hidden 用于后续状态注入。DSpark Verify 需要同时保留：

~~~text
logits
+
hidden_states
~~~

`commit_hidden()` 会直接检查这一条件：

~~~python
if hidden is None:
    raise RuntimeError(
        "DSpark verify requires target hidden states, got None."
    )
~~~

接受决策确定后，选中输入行的 Target hidden 要用于更新 Draft 的持久状态。以 R、A、B 为接受输入路径，就使用这些行的 hidden；不能只把输出 token 改成 A、B、X，却让 Draft 缓存仍停在错误候选 C 的状态上。

## 五、Greedy 与 Sampling 怎样接受候选

Target Verify 得到 logits 后进入：

~~~python
accept = self._verify_executor.accept_and_finalize(...)
~~~

固定源码：[`accept_and_finalize()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L138-L206)。

这里先区分接受事件：Greedy 比较 token 是否连续匹配，Sampling 根据提议分布与 Target 分布接受或拒绝。两者可以共享长度与输出契约，但判定规则不同。

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

`correct_len` 只数连续接受的 Draft，本例是 2。`bonus=X` 来自最后一个有效输入 B 的预测，用来替代不匹配的候选 C；若所有 Draft 都匹配，则由最后一行预测产生链后的新 bonus。它的来源随接受终点移动，不是固定取窗口最后一行。

源码：[`AcceptGreedy`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L560-L718)。

### Sampling：不是 token equality，而是 speculative sampling acceptance

概率采样路径需要保留 Draft 侧的 `corrected_logits` / probabilities，再构造 Target probabilities。这里必须使用实际产生提议的概率信息：Markov 修正会改变 logits，不能把修正前的 `base_logits` 直接冒充候选的提议分布。

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

Sampling 路径使用 Draft 与 Target 概率完成 Accept/Reject；Greedy 路径则按 Target 预测寻找匹配前缀。两者分别沿对应规则确定接受长度与新 bonus。

固定源码：[`AcceptSampling`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)。

### `correct_len`、`bonus`、`commit_lens` 是三个不同概念

这几个量分属不同对象：`correct_len` 是接受的 Draft 数，`bonus` 是一枚 token ID，`commit_lens` 是本轮推进长度。若把 token ID 当成位置，或把接受数直接当成提交长度，下一轮位置和缓存边界都会出错。

`FinalizeAcceptLens` 明确：

~~~python
commit_lens = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
~~~

源码：[`finalize_accept_lens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py#L720-L780)。

这里的计数同时用于输出推进和 KV-ready frontier，但两者对应的 Token 相差一位。例如旧 anchor 为 R，接受 A、B，采样出 bonus X：本轮输出是 `[A, B, X]`，本轮已有的输入 KV 是 `[R, A, B]`。X 的 KV 要在下一次 forward 计算；`commit_lens=3` 并不表示 X 已有 KV。

这里的 `+1` 计入新的 Target bonus。即使零枚 Draft 被接受，root 的 Target 预测仍可提供一枚输出；上层 EOS、stop 或输出长度截断则需要按各自规则另行处理。

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

最终有效输出前缀是：`A B X`

`BuildOutTokens` 的实现也是先复制 Draft Tokens，再把 `bonus` scatter 到 `correct_len` 对应位置：

~~~python
out_tokens[:, :gamma].copy_(draft_tokens)
out_tokens.scatter_(
    1,
    correct_len[:, None],
    bonus[:, None],
)
~~~

只有前 `commit_lens` 个位置属于本轮推进的输出。

如果四个 Draft 全部接受：

~~~text
correct_len = 4
commit_lens = 5
~~~

一轮 Target Verify 可以让请求前进 5 个 Token：4 个 Draft + 1 个 Target bonus。

这是候选全部接受时的产出上限，不是本轮必然得到的输出数。记录 AL 时还要说明是否包括这枚 bonus；严格 Draft 接受率应按同一窗口的实际接受数与提议数计算，不能把最大窗口宽度直接当成实测分母。

### Accept 结果也必须在 TP group 中达成一致

Eager accept 路径将下面三项交给同步入口：

~~~text
correct_len
bonus
cap_trim_lens
~~~

它们由 `SpecTpSync` 的相应 site 处理。若相关 rank 对接受长度、bonus 或裁剪结果不一致，后续这些状态就会分叉：

~~~text
new_seq_lens
positions
cache state
bonus token
~~~

因此检查同步结果时，不能只确认长度相同，还要确认末尾 token 和有效裁剪范围一致。

## 六、提交有效状态并开始下一轮

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

这里更新的是 Draft 的持久状态，不能理解成把 Target KV 字节原样复制给 Draft。固定版本的 V4 DSpark 会先投影、归一化 Target hidden，再按 Draft 各 stage 的 KV 投影写入缓存；`commit_lens` 控制哪些输入位置可以提交，模型侧实现见 [`write_target_hidden_kv()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_dspark.py#L932-L968)。

若原有已缓存前缀长度是 10，接受 A、B 后 `commit_lens=3`，则新 KV-ready 边界为 13，对应新增输入 R、A、B。新输出 X 留作下一轮 anchor，尚没有自己的 Target KV；不匹配的 C 仍可能占着预留空间，但不属于这次正式提交的历史。

某些 backend / Graph 路径支持 folded commit，把 Accept 与 hidden injection 放进 Verify epilogue。固定版本还要求相应路径满足 proposal folded、Greedy、无生效的 logits adjustment、无 Grammar 等条件，且能实际 replay Graph；Sampling 或 Grammar 不能仅因为 Draft proposal 已 folded，就认定接受和提交也已折叠。Python 控制流缩短后，仍要检查相同的接受边界与注入范围。

<details markdown="1">
<summary>补充：通用 Worker 中的 recurrent state 提交</summary>

通用 `_forward_decode()` 还包含 `_commit_target_mamba_states_after_verify(...)`，按 `commit_lens` 选择对应验证步的状态。这条通用分支服务于具有 recurrent / state-space 状态的模型。

</details>

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

下一轮从新的 bonus X 与 `new_seq_lens` 继续，同时使用刚更新的 Draft 状态。若只更新 token、没有更新状态，Draft 会把 X 接在错误候选历史后面；若只更新长度、bonus 仍是 C，后面的 Target 窗口也会从错误输入起步。

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

一轮结束时，输出和持久状态已经分别推进，新的 bonus 留作下一轮输入。调试时可以把 `correct_len`、`commit_lens`、新序列长度与实际选中的 Verify 行并排记录，检查候选生成、接受和状态注入是否对应同一段序列。

### 补充源码入口

- [Verify window 分配与布局](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py#L802-L911)
- [Planner 决策与宽度选择](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_planner.py)
- [Target hidden 提交](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/dspark_components/dspark_verify.py#L436-L504)
- [跨 TP 决策同步](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_tp_sync.py)
