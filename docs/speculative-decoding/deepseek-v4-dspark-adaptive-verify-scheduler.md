# DSpark 自适应 Verify：怎样分配候选预算

先用一个简化算例看预算分配：两条请求各有两个候选，条件分数分别是 `[0.9,0.8]`、`[0.6,0.5]`。额外只够验证两行，按前缀存活概率，两行都应该给第一条，得到 `verify_lens=[3,1]`。候选预算不必平均分。

但多验证一行要花时间，Confidence 分数也未必已经对应当前接受事件。DSpark 要同时估计前缀能走多远、这一轮花多久，再决定预算和逐请求长度。本文固定 Ascend 910C 配方仍为 static，compact 的适配、测量只讨论方案。

<details markdown="1">
<summary>版本与源码入口</summary>

以下分析采用固定源码。

| 组件 | 版本 |
| --- | --- |
| SGLang | <code>a9871012acb768dc94a43a6542cc32626c7b7b0b</code> |
| DeepSeek DeepSpec | <code>005e03b81cec38b7da6399833d609ee89a2587f2</code> |
| 核对日期 | 2026-09-21 |

训练标签参考公开 DeepSpec DSpark，运行侧看 SGLang 怎样使用 V4 Confidence Head。固定资料没有给出 V4-Flash 完整训练配方，两套实现不能直接等同。

</details>

## 一、用存活概率分配 Verify 预算

一次决策有 B 条请求，先给每条留 root。候选全未通过时，Target 仍可从 root 给 fallback / bonus，不计上层结束截断，基础推进就是 `baseline progress = B`。这里 B 数当前决策请求，不数设备，也未必等于全部 DP 请求之和。

再给一个位置，它连同此前候选一起通过的概率若为 s，期望推进增加 +s。第 k 枚能通过，包含了前面都没分歧的条件。

把候选 survival 降序排列，记为 `s1 >= s2 >= s3 >= ...`。

额外预算 b 对应下面的预期推进。

~~~text
tau(b)
=
B + sum(i=1..b) s_i
~~~

固定实现按下面的式子计算。

~~~python
candidates_sorted = torch.sort(candidates, descending=True).values
prefix_sum = torch.cumsum(candidates_sorted, dim=0)

tau_star = num_requests + torch.cat(
    [torch.zeros(1), prefix_sum]
)
~~~

开头两条教学请求的分数累乘，得到 `[0.9,0.72]`、`[0.6,0.3]`。给两行，最高两项都在首请求，`verify_lens=[3,1]`，预期推进为 `2+0.9+0.72=3.62`。

预算再增加，Verify 成本也跟着变，因此还要比较每单位时间能推进多少。

一维成本表按总验证行数 `B+b` 查 steps/s，Additive 表则估计整轮耗时。

~~~text
1D:       theta(b) = tau(b) * SPS(B+b)
Additive: theta(b) = tau(b) / T(B, B+b)
budget = argmax_b theta(b)
~~~

Confidence 提供预期收益，成本表把它换成时间上的产出。多验几个位置可以让一轮输出更多，却也可能降低每秒输出。

例如假设额外两行花 1 ms，预计 `3.62/0.001=3620 token/s`。再加一行，推进到 4.22，若成本变成 1.4 ms，就约为 3014 token/s。时间是假设值，这里只比较成本与推进的变化。

Planner 先选总预算，`ScheduleVerifyLensTopk` 再按当前 survival 分配，最后还要受 `min_verify_len`、`max_verify_len`、`survival_eps` 约束。简化排序没有省掉这些长度条件。

同请求的前缀概率，由条件分数累乘。

~~~text
s1 = c1
s2 = c1*c2
s3 = c1*c2*c3
...
~~~

每个 c 位于 0 到 1，累乘后保持 `s1 >= s2 >= s3 >= ...`。

并列先取较早位置，再按请求编号，保留前缀。最终长度通过 `SpecTpSyncSite.DSPARK_PLAN` 交给相应 TP 组。

## 二、训练标签与运行时校准的区别

用概率估计产出，还要知道它预测哪一种接受事件。训练的 soft target 与 Runtime 记录的前缀匹配并不相同。

### DeepSeek 官方 DeepSpec 训练的是什么

公开 DeepSpec 中，`AcceptRatePredictor` 的结构如下。

~~~text
Draft Hidden
+
optional Markov embedding
        ↓
Linear
        ↓
scalar confidence logit
~~~

sigmoid 将标量 logit 映成 0–1，分数的事件含义来自训练目标。

训练先计算下面的量。

~~~python
draft_probs  = softmax(draft_logits)
target_probs = softmax(target_logits)

accept_rate =
    1 - 0.5 * abs(draft_probs - target_probs).sum(-1)
~~~

其含义是下面的分布关系。

~~~text
a_i
=
1 - TV(q_i, p_i)
~~~

接着优化下面的目标。

~~~python
binary_cross_entropy_with_logits(
    confidence_pred,
    accept_rate
)
~~~

BCE 可以用小数标签。这里拟合各位置 Draft、Target 的分布重叠，并非一次真实 Token 接受或拒绝的二值记录。

经典 rejection 从同上下文 q 采样、按 min(1,p/q) 接受时，平均接受率为 `sum_x min(p(x),q(x)) = 1-TV(q,p)`，这是训练 target 的依据。Greedy 判断 Token 是否等于 Target argmax，用的是另一种事件。

DeepSpec 还记录了下面的观察。

~~~python
confidence_prefix_probs =
    sigmoid(confidence_pred).cumprod(...)

confidence_prefix_targets =
    accept_rate.cumprod(...)
~~~

这与 Runtime 累乘的形式相容，事件仍要通过标签对齐。公开示例说明这种训练目标，V4-Flash 的具体训练细节还需要自身配方佐证。

<a id="sglang-sts-soft-target"></a>

### STS 记录的是 Greedy 前缀

SGLang STS 先运行下面的步骤。

~~~python
target_predict =
    argmax(target_logits)

num_correct_drafts =
    compute_dflash_correct_drafts_and_bonus(
        candidates,
        target_predict,
    )
~~~

再形成下面的标签。

~~~python
prefix_mask =
    positions < num_correct_drafts
~~~

前三枚匹配时，例如下面这组结果。

~~~text
num_correct_drafts = 3

prefix_mask =
[1, 1, 1, 0, 0, ...]
~~~

标签表示接受前缀是否已经覆盖当前位置。首处分歧后，即使后面又碰巧匹配，也持续为 0，因此与逐位置 soft distribution overlap 不同。

两边预测、记录的对象如下。

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

一边度量分布重叠，另一边记录 Greedy 前缀是否通过。

### STS 用温度校准前缀存活概率

STS 按位置拟合正温度 `T_i`，调整分数刻度，候选不变，也不重训整个 Draft。

运行时使用下面的变换。

~~~text
c_i = sigmoid(z_i / T_i)
S_k = product(i<=k) c_i
~~~

fitter 逐位置搜索，保留此前的 cumulative survival，选择当前 ECE 较低的温度。预测 0.8 的前缀是否约八成匹配，是校准问题；ECE 不表示任务准确率，也不直接给出吞吐。

校准对应的是采集负载中的 Greedy 事件。换数据或策略，分数到实际事件的映射仍需要重新确认。

### Sampling 模式是一个重要边界

固定 recorder 用 `argmax(target_logits)` 生成 Greedy 前缀标签。

概率采样的接受还受 temperature、top-p、top-k 和 rejection 路径影响，Greedy 校准结果不能直接代表这些策略，需要对应实际事件。

## 三、tau 怎样对应期望输出推进

Accept 整理 `commit_lens = correct_len + 1`，`correct_len` 数接受的 Draft Token，另加一枚 Target fallback/bonus。整个 batch 的提交总量就是 `B + sum(accepted_drafts)`。tau 要估计的正是这种推进。

额外的一枚解释基础 B，但新输出还没作为输入计算，自身 KV 仍按提交规则接续。

### 概率事件对齐时的期望推导

给请求 r 的第 k 个候选定义 `I(r,k)`，接受前缀覆盖它就取 1，否则取 0。接受 Draft 数是这些指示量的和，期望如下。

~~~text
accepted_drafts_r = sum_k I(r,k)
E[accepted_drafts_r] = sum_k P(I(r,k)=1)

若 survival(r,k) = P(I(r,k)=1)：
E[commit_len_r] = 1 + sum_k survival(r,k)
E[sum(commit_lens)] = B + sum(survival)
~~~

这里用期望线性性，不要求事件独立。接受第 k 枚，本身包含前 k−1 枚通过的条件。survival 必须对应这些前缀事件，求和也只包含本轮允许验证的位置。

### Cap 本身也没有破坏这个推导

若某请求最多验证 `ell_r = verify_len - 1` 个候选，期望写成下面的范围。

~~~text
capped_correct_len = min(correct_len, ell_r)
commit_len = capped_correct_len + 1
~~~

把求和限定在这个前缀内，同一关系仍成立。

### Runtime 里哪些条件会改变估计

训练标签、STS 标签和最终策略，可能指向不同事件。

overlap 的总预算也可能用滞后 confidence，当前分数只参与分配，两者不一致时如下。

~~~text
history survival
!=
current survival
~~~

总预算就可能偏离这一轮的收益。

请求 slot 复用时，`req_generation` 防止把旧分数用给新请求。generation 不匹配就回退全 1 survival，这是乐观估计，并非新请求已经测出 100% 接受率。

graph-tier alignment 会进一步扩展到已付费 tier，不再严格保持原 argmax 预算，用来利用 padding。

`survival_eps` 把极小概率近似为 0。

Sampling、Grammar、logits adjustment、上层 stop / EOS，又可能让用户可见输出与 Runtime commit_lens 不一致。

tau 因而是提交推进的估计，标签、旧分数是否仍适用、tier 对齐都会影响逐轮误差。

## 四、SPS 成本模型与 Graph Tier

Confidence 估计推进，SPS 估计周期成本。SPS 单位为 steps/s，乘上预计每步 Token 数，才得到预算模型的 token/s。

### 1D SpsCostTable

一维表只保存下面的对应关系。

~~~text
M = total verify tokens
        ↓
Steps Per Second
~~~

static profiler 按下面的方式采集。

~~~text
batch_tokens =
num_running_reqs_per_rank
*
verify_num_draft_tokens

SPS =
1 / median(step_time)
~~~

Runtime 用 floor lookup，取不大于当前 M 的最近采样点，两点之间不平滑插值。

以下为教学表，M 是 Verify 行数，SPS 是 step/s。

| M | SPS |
|---:|---:|
| 32 | 100 |
| 64 | 95 |
| 96 | 68 |
| 128 | 66 |

1D lookup 因而能保留 64 到 96 的离散 cliff。

`_cliff_table()` 单测将预算 argmax 与 brute-force scan 对照。

它验证给定 cliff 表下的计算一致，服务负载是否被实测表覆盖、后续拟合能否恢复曲线，属于另外的验证。

### 为什么又需要 Additive SPS

同样的 M，可以由不同请求数 B 组成。

~~~text
B=8,  每个 Request 很长

B=32, 每个 Request 很短
~~~

B 会改变历史读取、请求管理和 collective 组织，只用 M 就分辨不出来。

Additive 把时间写成下面的分解。

~~~text
T(B,M)
=
bias
+
alpha(B)
+
theta(M)
~~~

它试着拆开这些成本。

~~~text
固定成本
+
Request-side 成本
+
Verify-token-side 成本
~~~

这些项都是时间，theta(M) 表示 Token 侧成本，和前面的产出目标 theta(b) 含义不同。它引入 B 的影响，同时采用可加近似，不保证因此更准。

### 三个关键假设

模型先假设 B、M 近似可加，没有显式的 B×M 交互项。

拟合默认 `mbin_w = 64`，按 `round(M/64)*64` 分 bin，同 bin 共用 Token 成本，原始边界点可能合并。

Runtime 再对导出的 theta(M) 线性插值。

因此真实 Graph tier 的阶跃，可能经过分 bin、拟合与插值后被平滑。

例如实际切换如下。

~~~text
M=63 → graph tier 64
M=64 → graph tier 64
M=65 → graph tier 96
~~~

拟合结果却可能更平。

### Off-diagonal profiler 知道 graph tier，但模型没有显式 graph-tier 变量

off-diagonal profiler 记录 tier，也检查 `graph_tier >= pinned M`。拟合输入仍只有 B、M、耗时，没有独立 tier 变量，切换成本只能间接进入 theta(M)。

<a id="additive-fit-cliff"></a>

### Additive 测试与 cliff 拟合验证

现有检查覆盖下面这些项目。

- 1D cliff 表的预算 argmax；
- Additive 插值与 scalar reference 一致；
- 拟合导出 residual、RMS、R²；
- self-check 的正 step_time。

真实阶跃经过 64-token 分 bin、最小二乘以后，是否仍选到正确预算，尚不是这些测试已经证明的事。residual 等误差可见，也不保证切换处误差足够小。

### 在 Graph Tier 两侧测量拟合误差

进一步验证可以在 captured tier 两侧密集采点。

~~~text
tier-2
tier-1
tier
tier+1
tier+2
~~~

再改变 B，比较下面这些结果。

~~~text
residual
R²
relative error
predicted optimal budget
actual throughput-optimal budget
~~~

误差若集中在切换点，可考虑下面的建模调整。

~~~text
显式加入 graph_tier
或
按 tier 分段拟合
或
使用 piecewise / lookup cost model
~~~

这是验证与改模建议，本文没有新采 Profile，也没有据此选出实际最优预算。

## 五、static、cap-accept 与 compact 的执行差异

调度最终生成下面这些长度。

~~~text
verify_lens =
[l1, l2, ..., lB]
~~~

长度限制接受范围，是否减少 forward 行数，则由执行模式决定。

### static

static 不建 Confidence Head，也不走 STS、SPS 预算和逐请求长度选择。

每请求固定输入 `gamma + 1` 行。

本文固定 Ascend 910C 官方注册路径使用 static。

### cap-accept

cap-accept 仍调用 `run_non_compact()`，输入 `B * verify_num_draft_tokens` 行，接受时才截到 verify_len−1。

N=4，三请求长度 `[2,1,4]`，Target 仍算 12 行，只把可接受 Draft 限为 `[1,0,3]`。接受少了，Verify FLOPs 并没有因此减少。

### compact

compact 则构造 `RaggedVerifyLayout`，压紧 Query。

还是 `verify_lens = [2,1,4]`，输入改成下面的布局。

~~~text
A0 A1
B0
C0 C1 C2 C3
~~~

共 7 个逻辑行，每请求首行仍是 root，候选数为 `[1,0,3]`。Graph 还可能 padding，所以 7/12 的行数比不能直接换成耗时比。

### Graph tier alignment 是 paid-padding reclamation

例如 37 个真实行落到 64-token tier。

原本是 `37 real + 27 padding`。

对齐可以按 survival 加入剩余候选，使真实行接近 64。

tier 成本近似不变，才可以把部分已付费 padding 换成真实探索，探索是否推进还看接受。候选不足不能硬填，超过 64 就进入新成本档。

<a id="_5"></a>

### DSpark 实现支持范围对照

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

适配完成，仍以相同负载比较 Verify、周期耗时与有效产出。

公开 DeepSpec 给出一种训练标签，Runtime 则用实际事件校准、估计推进和成本，两边的关系如下。

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

本篇固定 Ascend 910C 路径如下。

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

下一节展开 RaggedVerifyLayout 的 Query、Compressor 和 Graph metadata 条件，再看 TP、DPA、DeepEP。通用预算算法已有实现，后端仍需要让这些长度真正进入执行。

## 六、Ascend 910C 的固定宽度限制

固定 V4-Flash W8A8 NPU Accuracy / Performance 测试明确设置如下。

~~~bash
SGLANG_RAGGED_VERIFY_MODE=static
~~~

从 static 走向 compact，还会碰到下面的输入与后端条件。

### NPU DSV4 使用的是另一套 backend

NPU dsv4 创建 `DeepseekV4AscendAttnBackend`，与 CUDA `DeepseekV4AttnBackend` 分开。CUDA 显式设置 `supports_ragged_verify_graph=True`，NPU 没覆盖它，继承默认 False，因此此版本 Graph admission 不接受 NPU DSV4 ragged replay。

关掉 Graph，eager 的 Query 几何仍得正确，不能跳过这层检查。

### NPU Target Verify metadata 仍使用固定宽度

Ascend TARGET_VERIFY 从 `spec_info.draft_token_num` 取统一 `n_draft`，给每请求安排相同宽度 N，设置 `max_seqlen_q=N`，累计边界与 ragged 不同。

~~~text
固定宽度：actual_seq_lengths_q = [N, 2N, 3N, ..., B*N]
ragged 示例：verify_lens = [2,1,4]
             cu_seqlens_q = [0,2,3,7]
~~~

后一组含起始零，相邻差为 2、1、4。当前 metadata 不消费 `RaggedVerifyLayout.verify_lens`、`RaggedVerifyLayout.qo_indptr`，统一 n_draft 产生不了逐请求几何，改 capability flag 也不会自动改变 Attention 和状态写入。

### C4 / C128 Compressor metadata 也依赖固定 width

NPU Compressor Verify 同样使用 `live_seq_len + n_draft`。

Graph context 保存固定 `tokens_per_bs`。

position、seqused、sparse 与 Kernel metadata 都据此准备。

适配需要逐请求长度贯通输入行、位置、稀疏索引和算子参数，不能只变一个入口。

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

官方组合中还有 TP、DPA、MoE EP、DeepEP、DSpark。

不同 DP 的请求数和 `sum(verify_lens)` 可以不同，相关 Rank 仍要遵守同次 collective 的形状与参与约定。

Planner 已经提供下面的安排。

~~~text
local verify tier
        ↓
DP gather
        ↓
DP-global max token tier
~~~

idle-rank 也参与其中。

Ascend 910C 的完整接续还要一起验证下面这些项目。

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

单 Rank Query 正确，只解决局部问题，DPA、idle、DeepEP 的对应还需要验证。

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

<a id="_4"></a>

## 七、测量 DSpark 预算分配收益

### A. Confidence 事件语义

同时记录下列分数与事件。

~~~text
training-style soft accept proxy:
1 - TV(q,p)

STS greedy prefix label

runtime actual correct_len
~~~

分别看逐位置、累乘的校准，ECE、Brier 与排序质量。

Sampling 服务的标签则应取真实策略事件。

### B. tau 对真实 committed progress 的偏差

每步对照这些数值。

~~~text
predicted_tau =
B + sum(selected_survival)

actual_progress =
sum(commit_lens)
~~~

按 B、lag、采样模式、tier 分桶，再比较 `E[actual_progress - predicted_tau]`。

### C. Graph tier 两侧验证 SPS

在 tier 两侧采集成本点。

~~~text
tier-2
tier-1
tier
tier+1
tier+2
~~~

同时变化 B，对照下面这些结果。

~~~text
1D lookup error
Additive fit error
argmax budget error
actual throughput-optimal budget
~~~

### D. Ascend 910C 先验证 backend correctness

NPU compact 首轮正确性验证包括下面这些状态。

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

正确性通过以后，再比 Verify 设备时间、整轮耗时、TPOT 和吞吐。AL 与实测严格 Draft 接受率并列，并附同窗口提出、接受数；缺分母就标缺失，tau、`(AL-1)/gamma` 都不能代替实测接受率。预计推进、AL 和端到端收益，各用相应数据回答。

<a id="_6"></a>

## DSpark 源码阅读入口

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
