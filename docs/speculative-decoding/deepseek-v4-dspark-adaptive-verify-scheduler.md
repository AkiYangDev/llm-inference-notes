# DSpark 自适应 Verify：怎样分配候选预算

同一 batch 中，有的请求后几枚 Draft 很可能被接受，有的请求可能很快遇到不匹配。如果 Verify 预算有限，把相同长度分给所有请求，未必能得到最多的有效输出。

DSpark 的自适应调度用置信度估计前缀存活概率，再结合执行成本分配 `verify_lens`。本文先用预算分配解释它在优化什么，再区分训练目标、运行时校准与成本模型。最后回到执行路径：`cap-accept` 只限制接受范围，进入 compact 才会缩减逻辑输入行数；本文固定版本的 Ascend 910C 配方仍使用 static。

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

假设当前有 B 个 Request。

每个 Request 即使一个 Draft Token 都没被接受，Target Verify 仍会提供一个正确的 fallback / bonus Token。因此一轮至少有：`baseline progress = B`

如果额外 Verify 一个 Draft position，而该 position 能成为 accepted prefix 的概率是 s，那么它对期望 progress 的边际贡献就是：`+s`

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

然后引入 Verify Step 成本。

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

所以 DSpark 自适应 Verify 不是：

~~~text
Confidence > threshold
→ continue
~~~

而是：

~~~text
Expected Extra Progress
        /
Predicted Step Cost
~~~

的全局优化。

Planner 先决定这一轮总共多 Verify 多少位置，随后 <code>ScheduleVerifyLensTopk</code> 再用当前 Confidence / Survival 决定这些位置分给哪些 Request。

同一 Request 的 survival 是 cumulative product：

~~~text
s1 = c1
s2 = c1*c2
s3 = c1*c2*c3
...
~~~

因为每个 c 都在 0 到 1 之间，所以同一 Request 内：`s1 >= s2 >= s3 >= ...`

前缀概率随位置不增；出现并列值时，实现优先选择较早的位置，再按请求编号排序，从而保留前缀结构。最终得到：`verify_lens = [l1, l2, ..., lB]`

并通过 <code>SpecTpSyncSite.DSPARK_PLAN</code> 在相关 TP Group 内同步。

## 二、训练标签与运行时校准的区别

这一区分决定了后面收益模型的含义。

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

关键是训练 target。

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

训练。

所以官方公开 DSpark 训练语义更准确地说是：

> **每个 position 预测一个 soft probabilistic accept rate。**

它不是 0/1 的“这个 Token 最后接受没接受”标签。

经典 rejection sampling 中，Draft distribution q 与 Target distribution p 的平均 proposal acceptance 与 <code>1-TV(q,p)</code> 直接相关。因此这可以作为该位置的条件接受率 proxy。

DeepSpec 还显式观测：

~~~python
confidence_prefix_probs =
    sigmoid(confidence_pred).cumprod(...)

confidence_prefix_targets =
    accept_rate.cumprod(...)
~~~

这与 SGLang Runtime 后面的 cumulative survival 是相容的。

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

这是一个 **greedy argmax exact-prefix correctness event**，而不是训练时的 soft accept-rate target。

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

STS fitter 对每个 position 选择 Temperature：`T_i`

运行时：

~~~text
c_i = sigmoid(z_i / T_i)
S_k = product(i<=k) c_i
~~~

然后用 ECE 让 cumulative survival 更贴近采集到的 binary prefix event。

所以 STS 更准确的作用是：

> **把训练为 probabilistic conditional accept-rate predictor 的 checkpoint head，校准成当前 calibration workload 下更可信的 cumulative greedy-prefix survival signal。**

校准在这里将模型分数映射到运行时观察的 Greedy 前缀事件。

### Sampling 模式是一个重要边界

当前 STS recorder 用的是 <code>argmax(target_logits)</code>。

因此它校准的是 Greedy exact-prefix event。

如果真实服务使用 temperature / top-p / top-k / rejection sampling，最终 Accept Decision 的随机变量不再等同于这个 greedy label。

STS 当前校准到 Greedy Target-argmax Prefix Correctness。服务切换到其他 Sampling Policy 后，接受事件也随之变化，对应概率需要按该 Policy 的实际结果重新检查。

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

这解释了 Planner 为什么从 B 开始。

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

当 Survival 描述的概率事件与实际接受事件一致时，tau 等于期望提交长度。

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

所以最稳妥的结论是：

> **tau 用于估计提交长度。Survival 与接受事件对齐，且状态时滞和二次调度修正足够小时，这项估计接近期望提交长度。**

## 四、SPS 成本模型与 Graph Tier

Confidence 解决收益，SPS 解决成本。

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

Runtime 对 probe 做 floor lookup。

因此如果真实数据是：

| M | SPS |
|---:|---:|
| 32 | 100 |
| 64 | 95 |
| 96 | 68 |
| 128 | 66 |

1D 表可以原样保留 64 到 96 的 cliff。

当前 scheduler 单测还专门构造了 <code>_cliff_table()</code>，验证在明显的 SPS cliff 下，预算 argmax 与 brute-force scan 一致。

这证明的是：

> **Planner 能正确消费一个 cliff-aware 1D cost table。**

### 为什么又需要 Additive SPS

同一个 M 可以来自不同 Request 数 B：

~~~text
B=8,  每个 Request 很长

B=32, 每个 Request 很短
~~~

Attention KV history、per-request bookkeeping、DP/collective geometry可能不同。

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

这比只看 M 更合理，但引入了新的近似。

### 三个关键假设

第一，它假设 B 与 M 的影响近似可加，不显式建模 B×M interaction。

第二，Profiler fitting 当前默认：`mbin_width = 64`

把 M 做 64-token 分 bin。

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

Compact/off-diagonal profiling 会记录真实 graph tier，还会检查：`graph_tier >= pinned M`

确保测量合法。

但最终 fit 输入仍然是：`(B, M, measured step time)`

而不是：`(B, M, graph_tier, step time)`

所以 Graph tier cliff 最终只能被 theta(M) 间接吸收。

### 当前测试没有证明 Additive fit 能恢复 cliff

当前测试确认：

- 1D cliff table 的 scheduler argmax 正确；
- Additive Runtime interpolation 与 scalar reference 一致；
- Profiler fit 输出 residual、RMS、R²；
- self-check 验证 step_time 为正。

但它没有证明：

> **带真实 Graph tier discontinuity 的 trace 经过 64-bin additive OLS 后，还能恢复正确的 cliff 和最优 budget。**

Additive SPS 在这里承担的角色是：

> **更能表达 Request-count effect 的经验 cost approximation，而不是 graph-tier-aware 精确模型。**

### 在 Graph Tier 两侧测量拟合误差

在每个 captured tier 两侧密集采样：

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

而不是单纯增加随机样本。

## 五、static、cap-accept 与 compact 的执行差异

Planner 最终给出：

~~~text
verify_lens =
[l1, l2, ..., lB]
~~~

但这不自动意味着 Target Forward 只算 <code>sum(verify_lens)</code>。

### static

static 下不构建 Confidence Head，也没有 STS、SPS budget、per-request verify_lens。

每个 Request 固定 Verify：`gamma + 1`

当前 Ascend 910C DeepSeek-V4 官方注册 CI 就使用这个模式。

### cap-accept

cap-accept 可以产生不同 verify_lens，但 Worker 仍走 <code>run_non_compact()</code>。

Target 输入仍是固定：`B * verify_num_draft_tokens`

只是 Accept 阶段把：`correct_len`

裁到：`verify_len - 1`

所以它主要改变 Accept Horizon，不直接节省 Target Verify FLOPs。

### compact

compact 才会构造 <code>RaggedVerifyLayout</code>，压紧 Query rows。

例如：`verify_lens = [2,1,4]`

Target 只需要处理：

~~~text
A0 A1
B0
C0 C1 C2 C3
~~~

共 7 个 row，而不是固定宽度的 3×N。

### Graph tier alignment 是 paid-padding reclamation

假设真实需要 37 个 row，但 Graph 只能 replay 64-token tier。

不对齐时：`37 real + 27 padding`

开启 graph-tier alignment 后，可以继续按 survival 填真实候选，直到接近 64 real rows。

因为 64-token Graph 的成本已经支付，这相当于把 padding 变成可能产生 accepted progress 的有效工作。

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

因此 Decode Graph Runner 的 ragged admission 会直接拒绝 NPU DSV4 ragged graph replay。

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

这里还存在输入布局约束：NPU DSV4 Target Verify 的 Query 组织方式仍要求 uniform-width。单独修改 Graph capability flag，无法改变这项要求。

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

所以 compact support 不是改一个环境变量就完成。

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

然后才比较 Target Verify device time、Step time、TPOT 和 throughput。

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

对 Ascend 910C，下一步先完成后端输入几何的适配：

> **先让 DeepseekV4AscendAttnBackend 正确消费 RaggedVerifyLayout，把 Query / Compressor / Graph metadata 从 fixed-width 改成 per-request verify_lens，再逐层恢复 TP / DP / DeepEP 能力。**

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
