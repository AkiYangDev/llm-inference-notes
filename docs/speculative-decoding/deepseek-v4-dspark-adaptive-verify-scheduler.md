# DSpark 自适应 Verify：怎样分配候选预算

同一个 batch 中，某些请求的后续候选仍有希望通过，另一些可能很快分歧。预算有限时，给所有请求同样宽的 Verify 窗口，未必能换得最多有效输出。

DSpark 先用 Confidence 估计前缀存活，再结合成本选择 `verify_lens`。训练分数是否对应当前接受事件、选出的长度是否真减少 Target 行数，需分别确认。下面从预算算例走到校准、查表和执行。固定 Ascend 910C 配方仍为 static，compact 适配和测量只列为方案。

<details markdown="1">
<summary>版本与源码入口</summary>

采用以下固定源码：

| 组件 | 版本 |
| --- | --- |
| SGLang | <code>a9871012acb768dc94a43a6542cc32626c7b7b0b</code> |
| DeepSeek DeepSpec | <code>005e03b81cec38b7da6399833d609ee89a2587f2</code> |
| 核对日期 | 2026-09-21 |

训练标签参考公开 DeepSpec DSpark，运行侧看 SGLang 如何使用 V4 Confidence Head。V4-Flash 完整训练配方尚未公开，不能将两套实现直接等同。

</details>

## 一、用存活概率分配 Verify 预算

设一次决策看见 B 条请求，先各留 root。即使没有候选通过，Target 仍可由 root 给出 fallback / bonus，因此未计上层结束截断时，基础推进为 `baseline progress = B`。B 是本次决策请求数，不是设备数，也不必然是全部 DP 请求之和。

若一个额外位置连同前面候选通过的概率为 s，验证它带来的期望推进就是 +s。这里看整段前缀能否走到该处，不能忽略更早分歧单独判断 Token。

将所有候选的 survival 降序排列：`s1 >= s2 >= s3 >= ...`。

额外预算 b 对应：

~~~text
tau(b)
=
B + sum(i=1..b) s_i
~~~

固定实现计算：

~~~python
candidates_sorted = torch.sort(candidates, descending=True).values
prefix_sum = torch.cumsum(candidates_sorted, dim=0)

tau_star = num_requests + torch.cat(
    [torch.zeros(1), prefix_sum]
)
~~~

例如两请求条件分数 `[0.9,0.8]`、`[0.6,0.5]`，累乘为 `[0.9,0.72]`、`[0.6,0.3]`。多给 2 行时，最高两项都来自首请求，得到 `verify_lens=[3,1]`，预期推进 `2+0.9+0.72=3.62`。这些值只用于教学。

继续加预算会增加 Verify 成本，因此还需将预期推进换算成单位时间产出。

成本模型有两种表达。一维表用总验证行数 `B+b` 查 steps/s，Additive 表则直接估计本轮耗时：

~~~text
1D:       theta(b) = tau(b) * SPS(B+b)
Additive: theta(b) = tau(b) / T(B, B+b)
budget = argmax_b theta(b)
~~~

这里比较的是预计推进与周期成本，而不只是用一个 Confidence 阈值决定是否继续。多验证一些位置，即使增加单轮产出，也可能降低单位时间收益。

假设额外 2 行耗时 1 ms，预计为 `3.62/0.001=3620 token/s`；再加一行使推进到 4.22，若成本涨到 1.4 ms，则约为 3014 token/s。这是时间假设，说明更多单轮产出也可能降低单位时间收益。

Planner 先选总预算，再由 `ScheduleVerifyLensTopk` 按当前 survival 分配。实际还需应用 `min_verify_len`、`max_verify_len` 和 `survival_eps`，不能只照简化排序跳过长度约束。

同请求前缀概率由条件分数累乘：

~~~text
s1 = c1
s2 = c1*c2
s3 = c1*c2*c3
...
~~~

每个 c 在 0 到 1，因而 `s1 >= s2 >= s3 >= ...`。

并列时先取较早位置，再按请求编号，保持前缀。最终长度经 `SpecTpSyncSite.DSPARK_PLAN` 交给相关 TP 组。

## 二、训练标签与运行时校准的区别

要使用这些概率估产出，先得确认它描述的接受事件。训练 soft target 与 Runtime 前缀匹配并不相同。

### DeepSeek 官方 DeepSpec 训练的是什么

公开 DeepSpec 的 `AcceptRatePredictor` 结构为：

~~~text
Draft Hidden
+
optional Markov embedding
        ↓
Linear
        ↓
scalar confidence logit
~~~

标量 logit 经 sigmoid 变为 0–1 分数，但含义取决于训练目标，不能只由 Head 名称判断。

训练先求：

~~~python
draft_probs  = softmax(draft_logits)
target_probs = softmax(target_logits)

accept_rate =
    1 - 0.5 * abs(draft_probs - target_probs).sum(-1)
~~~

即：

~~~text
a_i
=
1 - TV(q_i, p_i)
~~~

再优化：

~~~python
binary_cross_entropy_with_logits(
    confidence_pred,
    accept_rate
)
~~~

BCE 也可用小数标签。这里拟合的是各位置 Draft、Target 分布重叠，不是某次 Token 接受/拒绝的二值结果。

经典 rejection 若确实从同上下文 q 采样、按 min(1,p/q) 接受，平均接受为 `sum_x min(p(x),q(x)) = 1-TV(q,p)`。这是训练 target 的依据；Greedy 比较的却是 Token 与 Target argmax，不能直接沿用该概率等式。

DeepSpec 还观察：

~~~python
confidence_prefix_probs =
    sigmoid(confidence_pred).cumprod(...)

confidence_prefix_targets =
    accept_rate.cumprod(...)
~~~

这与 Runtime 的累乘形式相容，但事件是否相同还需核标签。公开训练示例不能证明 V4-Flash 未公开的训练细节。

### 但 SGLang STS 校准的不是同一个 soft target

SGLang STS 则先运行：

~~~python
target_predict =
    argmax(target_logits)

num_correct_drafts =
    compute_dflash_correct_drafts_and_bonus(
        candidates,
        target_predict,
    )
~~~

再形成：

~~~python
prefix_mask =
    positions < num_correct_drafts
~~~

若前三枚匹配：

~~~text
num_correct_drafts = 3

prefix_mask =
[1, 1, 1, 0, 0, ...]
~~~

它标记接受前缀是否覆盖各位置，首处分歧后即使偶然再次匹配，也维持 0。这和逐位置的 soft distribution overlap 是不同标签。

两侧可对照为：

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

前者度量分布重叠，后者记录 Greedy 前缀事件。

### STS 更像运行时语义桥

STS 按位置拟合正温度 `T_i`，调整分数刻度，不改候选，也不重训整个 Draft。

运行时采用：

~~~text
c_i = sigmoid(z_i / T_i)
S_k = product(i<=k) c_i
~~~

fitter 按位置搜索，保留此前 cumulative survival，再选当前 ECE 较低的温度。ECE 检查相近预测概率是否接近实际频率，例如 0.8 的前缀是否约八成匹配；它不是任务准确率或最终吞吐。

因此校准仅针对采集负载下的 Greedy 事件。数据、策略变化后，还需核对映射，不能把一次拟合当作通用概率保证。

### Sampling 模式是一个重要边界

固定 recorder 使用 `argmax(target_logits)` 得到 Greedy 前缀标签。

概率采样的最终接受另受 temperature、top-p、top-k、rejection 路径影响。Greedy 校准良好，不足以证明能预测这些策略，需按实际事件重新核对。

## 三、tau 怎样对应期望输出推进

tau 可以从实际提交计数理解。Accept 整理出 `commit_lens = correct_len + 1`：`correct_len` 只数接受的 Draft Token，另外一枚来自 Target fallback/bonus。对整个 batch，提交总量就是 `B + sum(accepted_drafts)`。

这一枚额外输出解释了基础 B，但不表示新输出已有自己的 KV，输入输出错位仍按提交规则处理。

### 概率事件对齐时的期望推导

为请求 r 的第 k 个候选定义指示量 `I(r,k)`：接受前缀至少覆盖到这里时取 1，否则取 0。接受的 Draft 数等于这些指示量之和，因此：

~~~text
accepted_drafts_r = sum_k I(r,k)
E[accepted_drafts_r] = sum_k P(I(r,k)=1)

若 survival(r,k) = P(I(r,k)=1)：
E[commit_len_r] = 1 + sum_k survival(r,k)
E[sum(commit_lens)] = B + sum(survival)
~~~

这里使用期望线性性，不要求接受事件独立。接受第 k 枚本就包含前 k−1 枚已通过这一条件；真正的前提是 survival 对应各前缀事件，而且只对本轮允许验证的位置求和。

### Cap 本身也没有破坏这个推导

设某请求允许的候选数为 `ell_r = verify_len - 1`，则：

~~~text
capped_correct_len = min(correct_len, ell_r)
commit_len = capped_correct_len + 1
~~~

把期望求和范围也限制在这个前缀内，关系仍然成立。

### 一般 Runtime 中的偏差来自六类条件

首先，训练、STS、最终策略可能定义不同事件。

其次，overlap 的总预算可用滞后 confidence，当前值只用于分配。两者不一致时：

~~~text
history survival
!=
current survival
~~~

总预算便可能偏离当前收益。

请求 slot 复用时，`req_generation` 防止沿用旧分数。generation 不匹配时回退全 1 survival，是乐观估计，不是新请求实测接受 100%。

graph-tier alignment 还会把执行预算扩到已付费 tier，不再严格等于原 argmax，这是主动利用 padding。

`survival_eps` 将极小概率近似为 0。

Sampling、Grammar、logits adjustment、上层 stop / EOS 也可能使用户可见产出不同于 Runtime commit_lens。

因此 tau 是提交推进的估计。标签、历史适用性、tier 对齐都影响误差，公式成立不保证每轮预测准确。

## 四、SPS 成本模型与 Graph Tier

Confidence 估计收益，SPS 估计周期成本。SPS 是 steps/s，乘预期每步 Token 数才得到预算模型的 token/s。

### 1D SpsCostTable

一维表只保存：

~~~text
M = total verify tokens
        ↓
Steps Per Second
~~~

static profiler 按以下方式采集：

~~~text
batch_tokens =
num_running_reqs_per_rank
*
verify_num_draft_tokens

SPS =
1 / median(step_time)
~~~

Runtime 做 floor lookup，取不超过当前 M 的最近采样点，不在两点间平滑插值。

以下是假设表，M 为 Verify 行数，SPS 为 step/s：

| M | SPS |
|---:|---:|
| 32 | 100 |
| 64 | 95 |
| 96 | 68 |
| 128 | 66 |

1D lookup 可以保留 64 到 96 的离散 cliff。

`_cliff_table()` 单测比较预算 argmax 与 brute-force scan。

它只证明给定 cliff 表后的计算一致，尚未证明某份实测表覆盖服务负载，也未证明后续拟合能复现原曲线。

### 为什么又需要 Additive SPS

同一 M 还可由不同 B 组成：

~~~text
B=8,  每个 Request 很长

B=32, 每个 Request 很短
~~~

请求数会改变历史读取、管理和 collective 组织，单轴 M 无法分辨。

Additive 因而写为：

~~~text
T(B,M)
=
bias
+
alpha(B)
+
theta(M)
~~~

尝试拆分：

~~~text
固定成本
+
Request-side 成本
+
Verify-token-side 成本
~~~

这些项都用时间单位，此处 theta(M) 是 Token 侧成本，和前面 theta(b) 产出目标不同。模型能表示 B 的影响，也引入了可加近似，未必自动更准。

### 三个关键假设

它首先假设 B、M 近似可加，不显式描述 B×M 交互。

拟合默认 `mbin_w = 64`，按 `round(M/64)*64` 分 bin，同 bin 共享 Token 成本，原始边界点可能已被合并。

Runtime 还对导出 theta(M) 线性插值。

这些选择不一定能保留真实 Graph tier 阶跃。

例如实际切换为：

~~~text
M=63 → graph tier 64
M=64 → graph tier 64
M=65 → graph tier 96
~~~

拟合却可能将其平滑。

### Off-diagonal profiler 知道 graph tier，但模型没有显式 graph-tier 变量

off-diagonal profiler 记录 tier、检查 `graph_tier >= pinned M`，但拟合输入仍是 B、M、耗时，没有独立 tier 变量，切换成本只能间接进入 theta(M)。

### 当前测试没有证明 Additive fit 能恢复 cliff

现有检查覆盖：

- 1D cliff 表的预算 argmax；
- Additive 插值与 scalar reference 一致；
- 拟合导出 residual、RMS、R²；
- self-check 的正 step_time。

它们未证明真实阶跃经过 64-token 分 bin、最小二乘后还能选到正确预算。误差可查看不代表已足够小，需在 tier 切换处另核。

### 在 Graph Tier 两侧测量拟合误差

后续验证可优先密集测 captured tier 两侧：

~~~text
tier-2
tier-1
tier
tier+1
tier+2
~~~

并改变 B，比较：

~~~text
residual
R²
relative error
predicted optimal budget
actual throughput-optimal budget
~~~

若误差集中在切换点，可考虑：

~~~text
显式加入 graph_tier
或
按 tier 分段拟合
或
使用 piecewise / lookup cost model
~~~

这些是验证、改模建议，本文没有新采 Profile，也未据此得到实际最优预算。

## 五、static、cap-accept 与 compact 的执行差异

最终调度得到：

~~~text
verify_lens =
[l1, l2, ..., lB]
~~~

长度先限定接受范围，是否减少 forward 输入，还需看执行模式。

### static

static 不创建 Confidence Head，也不走 STS、SPS 预算或逐请求长度选择。

每请求固定 `gamma + 1` 行。

固定 Ascend 910C 官方注册路径采用此模式。

### cap-accept

cap-accept 仍走 `run_non_compact()`、输入 `B * verify_num_draft_tokens` 行，只在接受时裁到 verify_len−1。

N=4、三请求 `[2,1,4]` 时，Target 仍处理 12 行，只将可接受 Draft 限为 `[1,0,3]`。提交变短不等于 Verify FLOPs 已减少。

### compact

compact 才构造 `RaggedVerifyLayout` 压紧 Query。

同样 `verify_lens = [2,1,4]`，输入为：



~~~text
A0 A1
B0
C0 C1 C2 C3
~~~

共 7 个逻辑行，各首行仍是 root，候选为 `[1,0,3]`。Graph 可能继续 padding，7/12 不能直接换算时间收益。

### Graph tier alignment 是 paid-padding reclamation

例如 37 个真实行落在 64-token tier。

原布局为 `37 real + 27 padding`。

对齐可按 survival 再填可用候选，接近 64 real。

在 tier 成本近似不变时，部分 padding 可换成真实探索，是否推进仍看接受。候选不足不能硬填，跨过 64 则进入新成本档，不能继续称为已经付费。

## 六、Ascend 910C 的固定宽度限制

固定 V4-Flash W8A8 NPU Accuracy / Performance 测试显式设置：

~~~bash
SGLANG_RAGGED_VERIFY_MODE=static
~~~

走向 compact 还需处理以下输入、后端条件。

### NPU DSV4 使用的是另一套 backend

NPU dsv4 创建 `DeepseekV4AscendAttnBackend`，与 CUDA 的 `DeepseekV4AttnBackend` 分开。CUDA 显式声明 `supports_ragged_verify_graph=True`，NPU 未覆盖该 capability，继承的默认值为 False。因此这个版本的 Graph admission 不放行 NPU DSV4 ragged replay。

关闭 Graph 也不能直接证明 eager 正确，还要继续检查输入的 Query 几何。

### NPU Target Verify metadata 仍使用固定宽度

Ascend TARGET_VERIFY 从 `spec_info.draft_token_num` 取得统一的 `n_draft`，为每请求组织相同宽度 N，并设 `max_seqlen_q=N`。它的累计边界与 ragged 所需边界不同：

~~~text
固定宽度：actual_seq_lengths_q = [N, 2N, 3N, ..., B*N]
ragged 示例：verify_lens = [2,1,4]
             cu_seqlens_q = [0,2,3,7]
~~~

后一组边界包含起始零，相邻差为 2、1、4。当前 metadata 未消费 `RaggedVerifyLayout.verify_lens`、`RaggedVerifyLayout.qo_indptr`，统一 n_draft 无法自动产生这份逐请求几何。只改 capability flag，不会让 Attention 与状态写入随之正确。

### C4 / C128 Compressor metadata 也依赖固定 width

NPU Compressor Verify 还用 `live_seq_len + n_draft`。

Graph context 保存固定 `tokens_per_bs`。

据此准备 position、seqused、sparse 和 Kernel metadata。

适配需把这些固定宽度变为逐请求长度，并让各输入行、位置、稀疏索引和算子参数一致。

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

官方配置还组合 TP、DPA、MoE EP、DeepEP、DSpark。

不同 DP 的请求数、`sum(verify_lens)` 可不同，相关 Rank 却仍需满足同一次 collective 的形状和参与约定。

Planner 已有：

~~~text
local verify tier
        ↓
DP gather
        ↓
DP-global max token tier
~~~

并安排 idle-rank 参与。

Ascend 910C 仍需一起验证：

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

单 Rank Query 正确，不足以证明 DPA、idle、DeepEP 全部接好。

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

可同时记录：

~~~text
training-style soft accept proxy:
1 - TV(q,p)

STS greedy prefix label

runtime actual correct_len
~~~

检查逐位置、累乘的校准，以及 ECE、Brier、排序质量。

Sampling 服务则需采用真实策略事件。

### B. tau 对真实 committed progress 的偏差

每步对照：

~~~text
predicted_tau =
B + sum(selected_survival)

actual_progress =
sum(commit_lens)
~~~

再按 B、lag、采样模式、tier 分桶，看 `E[actual_progress - predicted_tau]`。

### C. Graph tier 两侧验证 SPS

成本验证在 tier 两侧采点：

~~~text
tier-2
tier-1
tier
tier+1
tier+2
~~~

同时变化 B，比较：

~~~text
1D lookup error
Additive fit error
argmax budget error
actual throughput-optimal budget
~~~

### D. Ascend 910C 先验证 backend correctness

NPU compact 的首轮正确性检查包括：

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

先通过正确性，再比较 Verify 设备时间、整轮耗时、TPOT、吞吐。AL 与实测严格 Draft 接受率并列，附同窗口提出、接受数；分母缺失就标缺失，不能把 tau 或 `(AL-1)/gamma` 当实测接受率。预计推进、AL、端到端收益需各自证据。

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

适配完成后，再以相同负载比较 Verify、周期耗时和有效产出。

将训练示例与 Runtime 分开看：公开 DeepSpec 说明一种标签来源，并不能证明 V4 Head 采用同一训练配方。Runtime 仍需用实际事件校准，再比较预计推进、执行成本与观察到的收益。

~~~mermaid
flowchart TD
    subgraph REF["公开 DeepSpec 训练示例"]
        P["Draft / Target 分布"] --> Q["1-TV(q,p) 软标签"]
        Q --> H["拟合 Confidence logit"]
    end
    subgraph RUN["通用自适应 Runtime；V4 Head 的训练配方未公开"]
        C["已加载的 Confidence Head"] --> L["Runtime logits"]
        G["Greedy 前缀标签"] --> S["STS 温度校准"]
        L --> S --> V["Survival estimate"]
        V --> E["期望推进 tau"]
        COST["实测成本模型 T(B,M)"] --> PLAN["比较 tau/T，选择总预算"]
        E --> PLAN
        V --> ALLOC["按本轮 survival 分配"]
        PLAN --> ALLOC --> LEN["verify_lens"]
        LEN --> MODE["cap-accept / compact"]
        MODE --> COMMIT["实际 commit_lens"]
        COMMIT --> PERF["TPOT / throughput"]
    end
~~~

固定 Ascend 910C 的当前公开路径则为：

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

因此起点是让 Ascend Backend 正确读取 `RaggedVerifyLayout`，对齐 Query、Compressor、Graph metadata，再逐层验证 TP、DPA、DeepEP。通用预算算法已存在，但无法替代后端和实际收益的验收。

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
