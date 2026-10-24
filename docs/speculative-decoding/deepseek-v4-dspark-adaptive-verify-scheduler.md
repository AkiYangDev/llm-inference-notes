# DSpark 自适应 Verify：怎样分配候选预算

同一 batch 中，有的请求后几枚 Draft 很可能被接受，有的请求可能很快遇到不匹配。如果 Verify 预算有限，把相同长度分给所有请求，未必能得到最多的有效输出。

DSpark 用 Confidence Head 估计候选前缀还能被接受的概率，再结合预计执行成本，决定每条请求的验证长度 `verify_lens`。这中间有两个容易混淆的地方：训练出的分数不一定直接等于当前服务的接受概率；少接受几枚候选，也不一定少执行几行 Target 计算。下面沿预算、校准、成本和执行布局逐层说明。本文固定版本的 Ascend 910C 配方仍使用 static，后面的 compact 适配与测量是建议步骤，不是已完成的性能结果。

<details markdown="1">
<summary>版本与源码入口</summary>

本文固定源码：

| 组件 | 版本 |
| --- | --- |
| SGLang | <code>a9871012acb768dc94a43a6542cc32626c7b7b0b</code> |
| DeepSeek DeepSpec | <code>005e03b81cec38b7da6399833d609ee89a2587f2</code> |
| 核对日期 | 2026-09-21 |

训练侧以 DeepSeek 公开的 DeepSpec DSpark 实现为例，运行侧分析 SGLang 怎样消费 V4 Checkpoint 的 Confidence Head。V4-Flash 的完整训练配方未公开，下面分别沿这两套实现解释标签与校准。

</details>

## 一、用存活概率分配 Verify 预算

假设一次 Planner 决策涉及 B 条请求，每条请求先保留一个 root 输入位置。即使所有 Draft 候选都不匹配，Target 也能从 root 的结果产生一个 fallback / bonus token。因此，在未考虑上层结束与截断的 Runtime 计数中，一轮的基础推进量是 `baseline progress = B`。这里 B 是该次决策看到的请求数，不是物理卡数，也不能直接当成所有 DP rank 请求数之和。

在这 B 行之外，每多验证一个 Draft 位置，就多一次接受它的机会。若它连同前面候选一起成为有效前缀的概率是 s，对期望输出推进的边际贡献就是 `+s`。这里问的是“接受路径能不能走到这个位置”，不是忽略前面分歧后单独判断这一枚 token 对不对。

把所有候选位置的 survival probability 从高到低排序：`s1 >= s2 >= s3 >= ...`

给出 b 个额外 Verify budget 时，Planner 使用：

~~~text
tau(b)
=
B + sum(i=1..b) s_i
~~~

SGLang 当前源码就是：

~~~python
candidates_sorted = torch.sort(candidates, descending=True).values
prefix_sum = torch.cumsum(candidates_sorted, dim=0)

tau_star = num_requests + torch.cat(
    [torch.zeros(1), prefix_sum]
)
~~~

例如两条请求的条件置信度分别是 `[0.9,0.8]` 和 `[0.6,0.5]`，累乘后的前缀存活概率是 `[0.9,0.72]` 与 `[0.6,0.3]`。若额外预算为 2，最高的两个值都来自第一条请求。在每条请求至少保留 root 的口径下，分配结果为 `verify_lens=[3,1]`，期望推进为 `2+0.9+0.72=3.62`。这是用于解释分配规则的教学算例。

预算也不能只按收益选得越大越好，因为新增 Verify 行会增加一轮的执行成本。Planner 接着将预计推进量换算成单位时间的产出。

一维 SPS 表：

~~~text
theta(b)
=
tau(b) * SPS(B+b)
~~~

Additive Cost Table：

~~~text
theta(b)
=
tau(b) / T(B, B+b)
~~~

最终：

~~~text
budget
=
argmax_b theta(b)
~~~

逐位置的阈值判断只能表达：

~~~text
Confidence > threshold
→ continue
~~~

这里的 Planner 则比较：

~~~text
Expected Extra Progress
        /
Predicted Step Cost
~~~

这个比值才是预算选择的目标。例如只比较两个可选预算：额外 2 行若预计耗时 1 ms，收益模型给出 `3.62/0.001=3620 token/s`；额外 3 行使期望推进增至 `4.22`，但若耗时变为 1.4 ms，预计产出反而约为 `3014 token/s`。这些时间是教学假设，说明更多候选可能增加单轮输出、却降低单位时间产出，不代表实测吞吐。

Planner 先决定这一轮总共多 Verify 多少位置，随后 <code>ScheduleVerifyLensTopk</code> 用当前的存活概率决定它们分给哪些请求。前面的简化推演采用每条请求保留一行的口径；实际分配还会应用 `min_verify_len`、`max_verify_len` 与 `survival_eps`，不能只从排序结果跳过这些长度约束。

同一 Request 的 survival 是 cumulative product：

~~~text
s1 = c1
s2 = c1*c2
s3 = c1*c2*c3
...
~~~

因为每个 c 都在 0 到 1 之间，所以同一 Request 内：`s1 >= s2 >= s3 >= ...`

前缀概率随位置不增；出现并列值时，实现优先选择较早的位置，再按请求编号排序，从而保留前缀结构。最终得到 `verify_lens = [l1, l2, ..., lB]`，再通过 <code>SpecTpSyncSite.DSPARK_PLAN</code> 的同步入口交给相关 TP Group。

## 二、训练标签与运行时校准的区别

前面的 s 要能用于预计产出，就需要先说明它估计的究竟是哪一种接受事件。Confidence Head 的训练标签与 Runtime 观察到的前缀匹配，不是同一个对象。

### DeepSeek 官方 DeepSpec 训练的是什么

DeepSpec 中 Confidence Head 叫 <code>AcceptRatePredictor</code>，结构仍然只是：

~~~text
Draft Hidden
+
optional Markov embedding
        ↓
Linear
        ↓
scalar confidence logit
~~~

Head 的输出是一个标量 logit；经过 sigmoid 可以转成 0 到 1 之间的分数。这个分数有什么含义，取决于训练时让它逼近的 target，而不是取决于 Head 叫不叫“AcceptRate”。

DeepSpec 先计算：

~~~python
draft_probs  = softmax(draft_logits)
target_probs = softmax(target_logits)

accept_rate =
    1 - 0.5 * abs(draft_probs - target_probs).sum(-1)
~~~

也就是：

~~~text
a_i
=
1 - TV(q_i, p_i)
~~~

然后 Confidence Head 使用：

~~~python
binary_cross_entropy_with_logits(
    confidence_pred,
    accept_rate
)
~~~

训练。这里 BCE 的 target 可以是 0 到 1 之间的小数：公开 DeepSpec 实现让每个位置的分数逼近 Draft 与 Target 分布的重叠程度，而不是用本次 token 最终接受或拒绝的 0/1 结果作标签。

在经典 rejection sampling 的前提下，候选确实从同一上下文的 `q` 采出，再按 `min(1,p/q)` 接受，平均接受概率就是 `sum_x min(p(x),q(x)) = 1-TV(q,p)`。这解释了该训练 target 的来源。但 Greedy 的事件是 Draft token 是否等于 Target argmax，不能把上面的分布关系直接套成 Greedy 匹配概率。

DeepSpec 还显式观测：

~~~python
confidence_prefix_probs =
    sigmoid(confidence_pred).cumprod(...)

confidence_prefix_targets =
    accept_rate.cumprod(...)
~~~

累乘的形式与 Runtime 使用的前缀 survival 相容；两者估计的是不是同一事件，还要继续看 Runtime 标签。这里仅借公开 DeepSpec 解释训练语义，不据此推定 V4-Flash 未公开的完整训练配方。

### 但 SGLang STS 校准的不是同一个 soft target

SGLang STS 数据收集路径做的是：

~~~python
target_predict =
    argmax(target_logits)

num_correct_drafts =
    compute_dflash_correct_drafts_and_bonus(
        candidates,
        target_predict,
    )
~~~

然后：

~~~python
prefix_mask =
    positions < num_correct_drafts
~~~

假设 Greedy Target 只匹配前三个 Draft：

~~~text
num_correct_drafts = 3

prefix_mask =
[1, 1, 1, 0, 0, ...]
~~~

这份 mask 表示 Greedy 接受前缀是否覆盖每个位置。前三个位置为 1，后面全为 0；即使第四个位置之后又偶然出现一个匹配 token，也不会越过首个分歧恢复接受。它与训练时每个位置的 soft accept-rate target 含义不同。

因此两边应画成：

~~~text
DeepSpec training
─────────────────
Draft / Target distributions
        ↓
a_i = 1 - TV(q_i,p_i)
        ↓
BCE soft target
        ↓
Confidence Head checkpoint


SGLang runtime STS
──────────────────
Confidence logits
        ↓
Target argmax 与 Draft 的实际 prefix match
        ↓
binary prefix_mask
        ↓
per-position Temperature
        ↓
calibrated cumulative survival
~~~

两侧标签分别对应分布重叠程度与 Greedy 前缀是否匹配。

### STS 更像运行时语义桥

STS 即这里按位置拟合的温度缩放。fitter 为每个位置选择一个正温度 `T_i`，修改分数的刻度，而不是改变候选 token 或重新训练整个 Draft 模型。

运行时：

~~~text
c_i = sigmoid(z_i / T_i)
S_k = product(i<=k) c_i
~~~

fitter 按位置依次搜索温度，保留前面已经拟合的 cumulative survival，再为当前位置选择 ECE 较低的结果。ECE（预期校准误差）比较一组相近预测概率与其实际事件频率，例如预测约 0.8 的前缀是否约有八成匹配；它不是任务准确率，也不直接衡量最终吞吐。

因此，STS 是把 Checkpoint Head 的分数刻度校准到当前采集负载下的 Greedy 前缀事件。校准数据分布或服务策略改变后，仍需检查这个映射是否成立，不能把一次拟合当成通用概率保证。

### Sampling 模式是一个重要边界

当前 STS 收集路径以 <code>argmax(target_logits)</code> 计算前缀匹配，recorder 保存这个 Greedy 标签。

服务切换到概率采样策略后，最终接受事件不再等同于这个 Greedy 标签。即使 STS 在 Greedy 采集集上校准良好，也不能直接推定它能预测 temperature、top-p、top-k 与 rejection sampling 路径的实际接受结果；需要按所用策略重新核对。

## 三、tau 怎样对应期望输出推进

Planner 的 tau 可以和实际 commit 计数对应起来。

DSpark Accept 最后进入 <code>FinalizeAcceptLens</code>：

~~~python
commit_lens =
    correct_len + 1
~~~

~~~text
correct_len
=
接受的 Draft Token 数

+1
=
Target fallback / bonus Token
~~~

所以单个 Request：

~~~text
commit_len
=
1 + accepted_drafts
~~~

全 batch：

~~~text
sum(commit_len)
=
B + sum(accepted_drafts)
~~~

每条请求都多计的这一枚，来自 Target 本轮提供的 fallback / bonus，这就是 Planner 从 B 开始的原因。它不表示所有新输出 token 都已有自己的 KV；输入与输出的位置偏移仍按 Accept 与 Commit 的规则处理。

### 概率事件对齐时的期望推导

对 Request r 定义：

~~~text
I(r,k)
=
accepted prefix 至少覆盖到第 k 个 Draft Token
~~~

则：

~~~text
accepted_drafts_r
=
sum_k I(r,k)
~~~

取期望：

~~~text
E[accepted_drafts_r]
=
sum_k P(I(r,k)=1)
~~~

如果：

~~~text
survival(r,k)
=
P(I(r,k)=1)
~~~

那么：

~~~text
E[commit_len_r]
=
1 + sum_k survival(r,k)
~~~

全 batch：

~~~text
E[sum(commit_lens)]
=
B + sum(survival)
~~~

这个推导用的是期望的线性性，不要求不同位置的接受事件彼此独立。前缀事件本来就相互依赖：接受第 k 个候选意味着前 k−1 个也被接受。真正需要的是每个 survival 都描述相应前缀事件，并且只对本轮允许验证的位置求和。

### Cap 本身也没有破坏这个推导

存在 per-request verify_len 时：

~~~python
ell_r = verify_len - 1

capped_correct_len =
    min(raw_correct_len, ell_r)
~~~

然后：

~~~text
commit_len =
capped_correct_len + 1
~~~

只要 survival 求和只覆盖被允许的 prefix positions，上面的期望关系仍然成立。

### 一般 Runtime 中的偏差来自六类条件

第一，训练 target、STS label 与最终服务 acceptance event 可能不同。

第二，Overlap Scheduler 下 global budget 可能使用 lagged confidence，而当前 confidence 只负责 allocation。因此：

~~~text
history survival
!=
current survival
~~~

时 budget 会偏。

第三，Request Pool slot 可能被复用，所以 Planner 用 <code>req_generation</code> 防止新 Request 继承旧 Confidence。generation 不匹配时回退到全 1 survival，避免沿用旧请求的低置信度。这是偏乐观的收益估计，并不表示新请求实测接受率为 100%。

第四，开启 graph-tier alignment 后，最终执行 budget 会向上填满已经支付的 graph tier；这时执行 budget 不再等于原始 argmax budget，而是刻意做 paid-padding reclamation。

第五，<code>survival_eps</code> 会把极小概率候选近似成 0。

第六，Sampling、Grammar、logits adjustment 和上层 stop / EOS 语义可能让最终用户可见 progress 与 Runtime 的 <code>commit_lens</code> 不完全相同。

tau 应读成提交长度的估计值。前面的条件说明，公式形式正确并不等于每轮预测都准确：标签是否对齐、历史分数是否适用于当前请求、预算是否又被 tier 对齐改写，都会影响它与实际提交量的差距。

## 四、SPS 成本模型与 Graph Tier

Confidence 估计多验证一个位置可能增加多少产出，SPS 表估计为此要付出多少时间。SPS 是 Steps Per Second，即每秒可执行多少个这样的 step；将期望 token 数乘以 SPS，才得到预算模型中的预计 token/s。

### 1D SpsCostTable

一维模型只有：

~~~text
M = total verify tokens
        ↓
Steps Per Second
~~~

Profiler 在 static sweep 中使用：

~~~text
batch_tokens =
num_running_reqs_per_rank
*
verify_num_draft_tokens

SPS =
1 / median(step_time)
~~~

Runtime 对 probe 做 floor lookup，即查找不超过当前 M 的最近采样点，而不是在两点间平滑插值。

用一组假设的测量点说明查表行为，M 的单位是 Verify token 行，SPS 的单位是 step/s：

| M | SPS |
|---:|---:|
| 32 | 100 |
| 64 | 95 |
| 96 | 68 |
| 128 | 66 |

1D 表可以原样保留 64 到 96 的 cliff。

当前 scheduler 单测还专门构造了 <code>_cliff_table()</code>，验证在明显的 SPS cliff 下，预算 argmax 与 brute-force scan 一致。

这项单测核对的是已有成本表到预算选择的计算：当输入表包含 cliff（成本突然增加、SPS 突然下降）时，Planner 能按它选预算。它没有证明某张实测表覆盖了服务负载，也没有证明后面的拟合能还原原始成本曲线。

### 为什么又需要 Additive SPS

同一个 M 可以来自不同 Request 数 B：

~~~text
B=8,  每个 Request 很长

B=32, 每个 Request 很短
~~~

即使总行数 M 相同，请求数 B 仍可能改变 Attention 历史读取、每请求管理开销与 DP/collective 的组织方式。1D 表只以 M 查找，无法区分这些来源。

所以 Additive model 写成：

~~~text
T(B,M)
=
bias
+
alpha(B)
+
theta(M)
~~~

它希望拆开：

~~~text
固定成本
+
Request-side 成本
+
Verify-token-side 成本
~~~

其中 bias、alpha 和 theta 都以时间为单位；这里的 `theta(M)` 是 token 侧成本项，与前面表示预算产出目标的 `theta(b)` 不是同一个量。把 B 和 M 分开能表达请求数效应，但也引入了可加近似，并非自动获得更准确的每一步预测。

### 三个关键假设

第一，它假设 B 与 M 的影响近似可加，不显式建模 B×M interaction。

第二，Profiler 拟合函数的默认参数是 `mbin_w = 64`，以 `round(M/64)*64` 将 M 归入 bin。同一个 bin 内的 M 共用 token 侧成本项，因此边界两侧的原始点可能先被合并。

第三，导出的 Additive table 在 Runtime 对 theta(M) 做线性插值。

这三点和 Graph tier 的真实阶跃并不天然一致。

真实 cost 可能是：

~~~text
M=63 → graph tier 64
M=64 → graph tier 64
M=65 → graph tier 96
~~~

而 Additive fit 更容易把它变成一条平滑曲线。

### Off-diagonal profiler 知道 graph tier，但模型没有显式 graph-tier 变量

Compact/off-diagonal profiling 会记录真实 graph tier，并检查 `graph_tier >= pinned M`，保证测量所用 tier 容得下指定行数。但拟合仍以 `(B, M, measured step time)` 为输入，没有把 graph tier 作为独立解释变量；tier 切换的成本变化只能由 `theta(M)` 间接吸收。

### 当前测试没有证明 Additive fit 能恢复 cliff

当前测试确认：

- 1D cliff table 的 scheduler argmax 正确；
- Additive Runtime interpolation 与 scalar reference 一致；
- Profiler fit 输出 residual、RMS、R²；
- self-check 验证 step_time 为正。

这些检查没有覆盖“真实 Graph tier 阶跃经过 64-token 分 bin 和最小二乘拟合后，是否仍能选到正确预算”这个问题。拟合有 residual、RMS、R²，只说明可以查看误差，不表示误差已经足够小。Additive SPS 在这里是能表达请求数效应的经验成本模型；tier 切换处是否可用，需要单独核对。

### 在 Graph Tier 两侧测量拟合误差

如果后续验证这个拟合，优先在每个 captured tier 两侧密集采样，而不是先大量增加随机点：

~~~text
tier-2
tier-1
tier
tier+1
tier+2
~~~

同时 sweep B，并检查：

~~~text
residual
R²
relative error
predicted optimal budget
actual throughput-optimal budget
~~~

如果误差集中在 tier boundary，更合理的升级是：

~~~text
显式加入 graph_tier
或
按 tier 分段拟合
或
使用 piecewise / lookup cost model
~~~

上面是针对边界误差的验证与模型改进建议，本文没有执行新的 profiling，也没有据此给出实际最优预算。

## 五、static、cap-accept 与 compact 的执行差异

Planner 最终给出：

~~~text
verify_lens =
[l1, l2, ..., lB]
~~~

这些长度描述允许各请求验证到哪里，但 Forward 是否真的减少行数，由后面的执行模式决定。需要把“接受范围”与“实际提交给模型的输入布局”分开。

### static

static 下不构建 Confidence Head，也没有 STS、SPS budget、per-request verify_lens。

每个 Request 固定 Verify：`gamma + 1`

当前 Ascend 910C DeepSeek-V4 官方注册 CI 就使用这个模式。

### cap-accept

cap-accept 可以产生不同 `verify_lens`，但 Worker 仍走 <code>run_non_compact()</code>，Target 输入保持固定的 `B * verify_num_draft_tokens` 行。到 Accept 阶段，才把 `correct_len` 裁到 `verify_len - 1`。

例如固定宽度 N=4、三条请求的 `verify_lens=[2,1,4]`，cap-accept 仍给 Target 12 行输入，只把各请求可接受的 Draft 数裁到 `[1,0,3]`。因此，仅看到有效提交长度变短，不能据此声称 Target Verify FLOPs 已减少。

### compact

compact 才会构造 <code>RaggedVerifyLayout</code>，压紧 Query rows。

例如：`verify_lens = [2,1,4]`

Target 只需要处理：

~~~text
A0 A1
B0
C0 C1 C2 C3
~~~

共 7 个逻辑 row，而不是前例固定宽度 N=4 时的 12 行。这里每条请求的第一行仍是 root，额外 Draft 行数为 `[1,0,3]`。compact 先减少的是有效输入行；如果后续使用更大的 Graph tier replay，实际执行还可能包含 padding，不能直接把 7/12 当作耗时比例。

### Graph tier alignment 是 paid-padding reclamation

假设真实需要 37 个 row，但 Graph 只能 replay 64-token tier。

不对齐时：`37 real + 27 padding`

开启 graph-tier alignment 后，可以继续按 survival 填真实候选，直到接近 64 real rows。

在该 tier 成本近似固定的模型下，原本为了 replay 支付的 27 行 padding 可以部分换成真实候选。这些候选是否增加有效提交仍取决于 survival；若可用候选不足，也不能强行填满。跨过 64 去选下一个 tier 则会改变预计成本，不能沿用“成本已经支付”的解释。

## 六、Ascend 910C 的固定宽度限制

当前 DeepSeek-V4-Flash W8A8 NPU Accuracy / Performance 注册测试都明确：

~~~bash
SGLANG_RAGGED_VERIFY_MODE=static
~~~

要从 static 走到 compact，还需要核对下面的 NPU 后端与输入布局。

### NPU DSV4 使用的是另一套 backend

在 NPU 上，<code>--attention-backend dsv4</code> 实际创建：`DeepseekV4AscendAttnBackend`

而不是 CUDA 侧的：`DeepseekV4AttnBackend`

CUDA DSV4 backend 明确声明：

~~~python
supports_ragged_verify_graph = True
~~~

NPU <code>DeepseekV4AscendAttnBackend</code> 当前没有 override 这个 capability。

Base class 默认：

~~~python
supports_ragged_verify_graph = False
~~~

因此该版本 Decode Graph Runner 的 ragged admission 不会放行 NPU DSV4 ragged graph replay。这个 capability 检查只说明 Graph 准入条件；能不能正确执行 Eager compact，还必须看输入 metadata，不能通过切到 Eager 就跳过后一个问题。

### 更硬的缺口：NPU Target Verify metadata 仍是固定宽度

当前 Ascend DSV4 backend 在 TARGET_VERIFY 下读取：

~~~text
n_draft =
spec_info.draft_token_num
~~~

然后构造：

~~~text
actual_seq_lengths_q =
[N, 2N, 3N, ..., B*N]
~~~

并设置：`max_seqlen_q = N`

这等价于假设每个 Request 的 Query width 都是同一个 N。

compact layout 例如：`verify_lens = [2,1,4]`

需要的是：

~~~text
cu_seqlens_q =
[0,2,3,7]
~~~

当前 NPU backend 的 TARGET_VERIFY metadata 没有消费：

~~~text
RaggedVerifyLayout.verify_lens
RaggedVerifyLayout.qo_indptr
~~~

这里的 `cu_seqlens_q` 含起始 0，表示每条请求在紧凑 Query 数组中的边界；相邻差分别是 2、1、4。当前 NPU Target Verify 的 metadata 按统一 `n_draft` 组织这些边界，单独修改 Graph capability flag，不会把它变成正确的 ragged 输入，也无法保证 Attention 或后续状态写入正确。

### C4 / C128 Compressor metadata 也依赖固定 width

NPU DSV4 Verify compressor 路径同样使用：`live_seq_len + n_draft`

Graph replay context 也维护固定：`tokens_per_bs`

并基于它构造 compressor position、seqused、sparse metadata 和 kernel metadata。

因此，NPU 的适配工作落在 Query 与 Compressor Metadata 上：将固定 `n_draft`、`tokens_per_bs` 改为各请求的 Verify 长度，再让这些长度与 Query Rows、位置、Sparse Metadata 和 Kernel 实参对应。

### 910C Compact 的适配工作

| 层 | 当前状态 / 缺口 |
|---|---|
| Confidence / STS / Planner | 通用 Runtime 已实现 |
| SPS Table | 通用实现存在，但需在 910C 重建 |
| RaggedVerifyLayout | 通用结构已存在 |
| NPU DSV4 Query metadata | **需消费 verify_lens / ragged cu_seqlens_q** |
| NPU C4/C128 metadata | **需支持 per-request ragged final length** |
| Ragged Graph capability | **Ascend backend 尚未 opt-in** |
| Token-keyed graph tiers | Generic runner 已有机制，需 NPU capture/replay 验证 |
| DP Attention tier agreement | Planner 有协议，需 NPU/HCCL 实测 |
| Idle DP rank participation | 需保证相同 global tier 下参加 collective |
| MoE / DeepEP | 需验证 ragged token count 下 dispatch/combine |
| Accept / Scatter / Commit | 通用逻辑存在，需 NPU 回归 |
| NPU CI | 当前公开 DSV4 DSpark 注册路径仍为 static |

### DP / DeepEP 是第二层复杂度

官方 910C DSV4 配置同时存在 TP、DP Attention、MoE EP、DeepEP 和 DSpark。

Compact 下不同 DP rank 的本地 Request 数和 <code>sum(verify_lens)</code> 都可能不同，相关 Rank 的 Graph Tier 选择还需要满足同一次 Collective 的形状与参与约定。

Planner 已经存在：

~~~text
local verify tier
        ↓
DP gather
        ↓
DP-global max token tier
~~~

以及 idle-rank participation 机制。

但把它带到 Ascend 910C，需要同时验证：

~~~text
ragged Attention geometry
+
NPU Graph
+
HCCL collective shape
+
DeepEP dispatch/combine
+
idle rank participation
+
C4/C128 state write
~~~

这些是输入、图执行和 collective 的共同约束。即使单 rank 的紧凑 Query 能执行，也不能据此宣称 DP Attention、空闲 rank 与 DeepEP 路径已经适配完成。

### 更合理的 bring-up 顺序

~~~text
1. 单 rank / eager
   补 NPU DSV4 ragged Query geometry

2. 单 rank / Graph
   验证 token-tier capture/replay

3. TP only
   验证 SpecTpSync 与 verify_lens 一致

4. DP Attention
   验证 DP-global tier 与 idle participation

5. MoE + DeepEP
   验证 ragged token count 的 collective

6. 完整 DSV4 W8A8
   static vs cap-accept vs compact
~~~

## 七、测量预算分配是否带来收益

### A. Confidence 事件语义

同时记录：

~~~text
training-style soft accept proxy:
1 - TV(q,p)

STS greedy prefix label

runtime actual correct_len
~~~

比较 per-position calibration、cumprod calibration、ECE、Brier 和 ranking quality。

如果服务使用 sampling，应按真实 sampling policy 重新评估。

### B. tau 对真实 committed progress 的偏差

每个 Step 记录：

~~~text
predicted_tau =
B + sum(selected_survival)

actual_progress =
sum(commit_lens)
~~~

按 batch size、lag、sampling mode、graph tier 分桶，观察：`E[actual_progress - predicted_tau]`

### C. Graph tier 两侧验证 SPS

每个 tier 密集 sweep：

~~~text
tier-2
tier-1
tier
tier+1
tier+2
~~~

同时改变 B，比较：

~~~text
1D lookup error
Additive fit error
argmax budget error
actual throughput-optimal budget
~~~

### D. Ascend 910C 先验证 backend correctness

第一批 compact NPU 测试应确认：

~~~text
input rows
=
sum(verify_lens)

cu_seqlens_q
=
prefix_sum(verify_lens)

C4/C128 write positions
与 ragged rows 一致

compact logits/hidden
能正确 scatter 回固定 stride

commit_lens/new_seq_lens
与 static baseline 保持 lossless
~~~

上述 correctness 项需要先通过，再比较 Target Verify device time、整轮耗时、TPOT 和输出吞吐。性能表同时保留 AL 与实测严格 Draft 接受率，并列出同一窗口的实际提出、接受计数；若没有候选分母，应标明缺失，不能把 tau 或 `(AL-1)/gamma` 改名为实测接受率。高预计产出、高 AL 和端到端提速需要分别用对应证据确认。

---

### 实现支持范围对照

| 组件 | 当前实现方式 |
| --- | --- |
| Confidence Head | 每个位置输出一个标量 |
| DeepSpec 训练标签 | `1-TV(q,p)`；V4 完整训练配方未公开 |
| STS 标签 | Greedy 前缀正确性的二值结果 |
| tau | 根据 Survival 估计提交长度 |
| 1D SPS | 直接保存吞吐曲线，可表达离散变化 |
| Additive SPS | 拟合成本组成，Graph Tier 切换处需检查误差 |
| CUDA DSV4 Ragged Verify | 有对应 Capability 与 Metadata 路径 |
| NPU DSV4 Verify | 当前 Metadata 使用固定 Draft 宽度，注册配置采用 static |

NPU Compact 的适配完成后，再用相同负载比较 Verify 时间、整轮耗时与有效产出。

把前面的数据流连起来：

~~~text
DeepSpec Training
soft conditional accept target: 1-TV(q,p)
            ↓
Confidence Head Checkpoint
            ↓
SGLang Runtime logits
            ↓
STS
calibrate toward greedy prefix event
            ↓
Survival estimate
            ↓
Expected-progress surrogate tau
            +
Empirical cost model T(B,M)
            ↓
argmax tau/T
            ↓
Global budget
            ↓
Current-survival Top-K allocation
            ↓
verify_lens
            ↓
cap-accept / compact
            ↓
actual commit_lens
            ↓
observed TPOT / Throughput
~~~

而在 Ascend 910C 上，当前链路实际仍停在：

~~~text
Confidence / Ragged Runtime infrastructure
          已存在
          ↓
NPU DSV4 Target Verify
仍是 fixed-width metadata
          ↓
官方注册路径
SGLANG_RAGGED_VERIFY_MODE=static
~~~

对本文固定版本的 Ascend 910C，适配起点是让 `DeepseekV4AscendAttnBackend` 正确消费 `RaggedVerifyLayout`，把 Query、Compressor 和 Graph metadata 与各请求的 `verify_lens` 对齐。完成这些输入与状态校验，再逐层验证 TP、DP Attention 和 DeepEP；预算公式与通用 Runtime 已存在，不能替代后端正确性和实际收益的验收。

## 源码阅读入口

### DeepSeek 官方训练侧

- [DeepSpec DSpark common / AcceptRatePredictor](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py)
- [DeepSpec DSpark loss](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py)
- [DeepSpec DSpark model example](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py)

### SGLang Confidence / STS

- [DSpark Confidence Head](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/models/dspark.py)
- [DeepSeek-V4 DSpark model](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/models/deepseek_v4_dspark.py)
- [STS recorder](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/speculative/dspark_components/dspark_sts.py)
- [STS fitter](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/benchmark/dspark_sts_fit.py)

### Planner / SPS / Accept

- [DSpark Planner](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/speculative/dspark_components/dspark_planner.py)
- [SPS Cost Table](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/speculative/dspark_components/dspark_sps.py)
- [SPS Profiler / Additive fit](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/benchmark/dspark_sps_profiler.py)
- [Verify length scheduling](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py)
- [Accept / Finalize kernels](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)

### Ragged Verify / Ascend 910C

- [RaggedVerifyLayout](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/speculative/ragged_verify.py)
- [DSpark Verify Executor](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/speculative/dspark_components/dspark_verify.py)
- [Decode Graph Runner](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)
- [Ascend DSV4 Attention Backend](https://github.com/sgl-project/sglang/blob/a9871012acb768dc94a43a6542cc32626c7b7b0b/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)
