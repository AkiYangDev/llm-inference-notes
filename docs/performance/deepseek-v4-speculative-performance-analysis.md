# DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？从 Acceptance Rate、Draft Cost 到 KV Overhead

前面的几篇文章已经把 DeepSeek-V4 Speculative Decoding 的 correctness 链路拆到了 Draft、Verify、Tree Attention、Accept 与 KV Commit。接下来真正值得问的是：

> **为什么 Speculative Decoding 有时显著加速，有时却会让吞吐下降？**

最容易产生的误解是：

```text
一轮可以输出多个 Token
=
一定更快
```

真实系统优化的并不是“每轮提议多少 Token”，而是：

```text
单位 wall-clock 时间
真正提交多少有效 Token
```

SGLang 当前 DeepSeek-V4 cookbook 甚至明确区分三类 operating point：

```text
low-latency    → 更积极的 speculation
balanced       → 更短的 speculative window
high-throughput→ 直接关闭 MTP
```

原因很简单：系统进入饱和后，额外 Draft + Verify 的成本可能超过省掉的 Decode step。

> **源码基线**：本文基于 `sgl-project/sglang @ 5f017ffabb6ab8d214f6a4616ee8bd98a376034a`，核对日期为 2026-09-20。这个 commit 本身更新了 Ascend/NPU performance testing framework，并开始把 speculative acceptance 纳入性能 CI。

## Strict Review 结论先行

| Review 点 | 固定源码结论 |
| --- | --- |
| `accept_length / step_time` 是否只是低并发近似 | **不是只适用于低并发。** `bench_speculative.py` 在每个固定 `batch_size` 下都用它做配置排序。但它没有乘 batch size，更准确是“固定 operating point 下的 per-request progress proxy”，不能直接跨 batch size 当 aggregate throughput 比。 |
| runtime strict `accept_rate` 与 Ascend CI `accept_rate` | 两者**不是同一个指标**。runtime 是 `correct_drafts / proposed_drafts`，严格不含 bonus；Ascend CI 当前变量名 `accept_rate` 实际是 `accept_length / speculative_num_draft_tokens`，只是 performance baseline proxy。 |
| Block Size 增大后哪些成本可 amortize | 能 amortize 的主要是 fixed / latency-dominated 部分：launch、调度、部分同步延迟、权重读取与 GEMM 利用率。不会消失的是随真实 verify token 数增长的 Target compute、MoE routing/expert work、KV/state write 与通信字节。 |
| Ragged Verify 是否破坏固定 `N+1` 模型 | **会。** `static` 模式下每个请求确实 Verify `gamma+1`；`compact / cap-accept` 会给每个请求不同的 `verify_len ∈ [1, gamma+1]`。真实 token work 应改看 `sum(verify_lens)`，而 graph 还可能 pad 到更大的 `graph_num_tokens`。 |

---

## 一、第一性公式：Speculative Decoding 到底靠什么赚钱？

普通 autoregressive Decode 一步推进一个 Token。若 Target Decode step 的耗时为：

\[
T_{decode}
\]

那么单请求进度率近似：

\[
R_{base}\approx\frac{1}{T_{decode}}
\]

Speculative Decode 一轮平均推进 `A` 个输出 Token，整轮耗时 `T_spec`：

\[
R_{spec}\approx\frac{A}{T_{spec}}
\]

于是：

\[
Speedup\approx\frac{A\cdot T_{decode}}{T_{spec}}
\]

必要条件：

\[
\boxed{T_{spec}<A\cdot T_{decode}}
\]

这比“Acceptance Rate 高就会快”更接近第一性原理。

因为真正进入分子的不是 Draft Model 有多聪明，而是：

```text
这一轮最终推进多少有效输出 Token
```

而分母是：

```text
为了推进这些 Token，
整条 speculative critical path 花了多久
```

一轮 speculative iteration 至少可能包含：

```text
Draft
  ↓
Target Verify
  ↓
Sampling / Accept
  ↓
KV / State Commit
  ↓
Distributed Sync
```

真实系统还可能有 CUDA/NPU Graph、multi-stream、scheduler overlap 和 communication overlap。因此更正确的性能对象不是各 kernel duration 的机械求和，而是：

> **Speculative Round 的 critical path。**

---

## 二、`accept_length / step_time`：固定 operating point 的调参 proxy，而不是跨并发总吞吐公式

SGLang 自己的：

[`scripts/playground/bench_speculative.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/scripts/playground/bench_speculative.py)

直接计算：

```python
step_time = np.percentile(
    server_info["internal_states"][0]["step_time_dict"][str(batch_size)],
    20,
)

speed = 1 / step_time * acc_length
```

也就是：

\[
speed=\frac{accept\_length}{step\_time}
\]

但严格 Review 后必须讲清三个边界。

### 1. 它不是 `bs=1` 专用

脚本会遍历：

```text
batch_size
steps
topk
num_draft_tokens
```

所以 `accept_length / step_time` 不是低并发专属。它会在每个固定 batch size 下用于比较不同 speculative config。

### 2. 它不能直接跨 Batch Size 当服务器总吞吐

脚本没有乘 `batch_size`。

如果所有请求都稳定存活、每条请求每 step 平均推进 `A` 个 Token，那么 aggregate progress 更接近：

\[
R_{aggregate}\approx\frac{B\cdot A}{T_{step}}
\]

而脚本里的：

\[
\frac{A}{T_{step}}
\]

更像固定 operating point 下“单条 request 的平均推进速度 proxy”。

因此它非常适合：

```text
固定 B
比较 1-1-2 / 3-1-4 / 5-1-6
```

但不适合直接拿 `bs=1` 和 `bs=128` 的 speed 值比较“总吞吐”。

### 3. 它不是严格的端到端 TPOT

`step_time` 来自 server-side `step_time_dict`，而且脚本刻意使用 20th percentile，而不是 P50/Mean。

所以它更适合：

> **搜索 speculative 参数空间。**

真正决定线上服务仍要回到：

```text
TPOT
P99 ITL
Output Throughput
E2E Latency
HBM
Concurrency
```

SGLang 当前新增的：

[`dspark_sps_profiler.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/benchmark/dspark_sps_profiler.py)

进一步显式建模：

```text
batch_tokens
=
num_running_reqs_per_rank × verify_num_draft_tokens

steps_per_sec
=
1 / median(server-side step_time)
```

这已经说明：Step Time 必须和 request/token geometry 一起解释。

---

## 三、Acceptance Rate、Acceptance Length、CI Acceptance Proxy：三个名字很像，含义不同

当前 SGLang runtime 在：

[`tokenizer_manager.py::_calculate_spec_decoding_metrics()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/tokenizer_manager.py)

明确计算：

```python
num_proposed_drafts = spec_verify_ct * (
    speculative_num_draft_tokens - 1
)

spec_accept_rate =
    num_correct_drafts / num_proposed_drafts
```

源码注释：

```text
strict count, no bonus
```

因此：

\[
\boxed{
spec\_accept\_rate=
\frac{CorrectDrafts}{ProposedDrafts}
}
\]

### Accept Length 是另一件事

同一函数：

```python
spec_accept_length =
    completion_tokens / spec_verify_ct
```

源码注释：

```text
includes bonus token
```

因此：

\[
\boxed{
spec\_accept\_length
=
平均每次 Verify 推进的输出 Token 数
}
\]

例如：

```text
proposed drafts:
D1 D2 D3 D4 D5

accepted:
D1 D2 D3

bonus:
B
```

那么：

```text
strict accept rate = 3 / 5 = 0.60
accept length      = 4
```

### Ascend Performance CI 还有一个同名 `accept_rate`

最新 NPU performance framework 在：

[`test_npu_performance_utils.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/test/ascend/e2e/test_npu_performance_utils.py)

新增：

```python
accept_rate =
    float(metrics["accept_length"])
    / spec_num_draft_tokens
```

这和 runtime strict rate 不是一个公式。

它更接近：

\[
\boxed{
CIProxy=
\frac{AcceptLength}{ConfiguredVerifyWidth}
}
\]

当前一条 DeepSeek-V4 Flash NPU performance case 设置：

```text
--speculative-num-draft-tokens 7
--speculative-dspark-block-size 6
accept_rate baseline = 0.5
```

这里的 `7` 对应 DSpark：

```text
gamma = 6
verify window = 7
```

所以 CI baseline 的真实含义是：

```text
accept_length / 7 >= 0.5
```

即平均 Accept Length 至少约 3.5，而不是“strict draft acceptance ≥ 50%”。

以后建议统一区分：

```text
Strict Accept Rate
= correct_drafts / proposed_drafts

Accept Length
= outputs / verify step

CI Acceptance Proxy
= accept_length / configured verify width
```

---

## 四、Block Size 增大：到底摊薄了什么？

“Verify 多个 Token 能把成本摊薄”方向没错，但一定要继续问：

```text
摊薄的到底是什么？
```

### 可以被 amortize 的部分

随着一个 step 中 Token 数增加，下面这些固定或低斜率成本可能被摊薄：

```text
Kernel / Graph Launch
Scheduler / Runtime overhead
部分 collective latency
部分 barrier / sync latency
权重读取的重复固定成本
小 GEMM 的低利用率
```

特别是低 batch、q_len=1 时，大模型 Decode 往往很难把硬件吃满。更宽的 Verify 能把“每 step 付一次”的成本分摊到更多 candidate。

### 不会消失的 Token-Proportional Work

但这些工作仍会随真实 Verify Token 数增长：

```text
Target QKV / Attention query compute
Dense / MoE GEMM token work
MoE routing
Expert dispatch / combine bytes
KV write
C4 / C128 compressed state work
LM-head / logits work
Draft-side per-token compute
```

因此更准确的说法是：

```text
Total Cost ↑

但

Cost per Verified Token
可能 ↓
```

而不是“Block 越大，Verify 成本越接近固定”。

### SGLang 新 SPS Cost Model 已经把这件事写进代码

当前 DSpark planner 支持 additive cost model，近似：

\[
T(B,M)=bias+\alpha(B)+\theta(M)
\]

其中：

```text
B = request count
M = actual verify-token count
```

固定源码：

[`dspark_planner.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py)

这实际上明确承认：

> **相同 request 数量下，Verify Token 数仍然会改变 Step Time。**

### MoE / DeepEP 更是硬证据

DeepSeek-V4 cookbook 当前要求：

```text
max-running-requests
×
MTP_draft_tokens

≤
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
```

否则 steady-state 下可能打爆 DeepEP dispatch buffer。

所以增大 Block Size 后，MoE Token Work 和 DeepEP buffer / payload pressure 都会增加。更大的 payload 可能提高通信效率、摊薄 latency，但总 dispatch token work 并不会消失。

因此可以把 Step Time 抽象成：

\[
T_{step}(B,M)
=
T_{fixed}(B)
+
T_{token}(B,M)
\]

扩大 Block 后：

\[
\frac{T_{fixed}}{M}\downarrow
\]

但：

\[
T_{token}(B,M)\uparrow
\]

这就是为什么收益最终会出现拐点。

---

## 五、Static DSpark 才是固定 `gamma+1` Verify；Ragged Verify 已经改变了模型

DSpark runtime 仍然定义：

```text
gamma = draft block size
verify_num_draft_tokens = gamma + 1
```

在：

```text
SGLANG_RAGGED_VERIFY_MODE=static
```

下，可以近似认为：

\[
M_{static}=B(\gamma+1)
\]

但 SGLang 当前已经有三种 mode：

[`ragged_verify.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/ragged_verify.py)

```python
STATIC = "static"
CAP_ACCEPT = "cap-accept"
COMPACT = "compact"
```

非 static 模式下，DSpark planner 会使用 confidence head 的 per-position survival probability 和 global token budget，给每个请求安排不同 Verify Length。

固定实现：

[`dspark_schedule.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py)

最终：

```python
verify_lens =
    min_len + selected_extra

verify_lens =
    clamp(
        verify_lens,
        min=max(min_verify_len, 1),
        max=max_len,
    )
```

其中：

```text
max_len = gamma + 1
```

所以同一个 batch 可以变成：

```text
req0 verify_len = 6
req1 verify_len = 2
req2 verify_len = 1
req3 verify_len = 4
```

真实 scheduled verify tokens：

\[
\boxed{
M=\sum_r verify\_len_r
}
\]

不再是：

\[
B(\gamma+1)
\]

---

## 六、Ragged Verify 还必须区分“真实 Token Work”和“Graph Tier”

`RaggedVerifyLayout` 同时维护：

```text
verify_lens
total_verify_tokens
graph_num_tokens
```

并检查：

```text
total_verify_tokens
=
sum(verify_lens)

total_verify_tokens
<=
graph_num_tokens
```

因此：

```text
Logical Real Work:
total_verify_tokens

Graph Execution Tier:
graph_num_tokens
```

两者可能不相等。

例如：

```text
真实需要 Verify 37 tokens

但 CUDA/NPU Graph tier = 48
```

把真实 Verify Token 从 40 降到 37，不一定马上让 step time 下降，因为仍然落在同一个 graph bucket。

SGLang 甚至提供：

```text
--speculative-dspark-align-verify-tokens-to-graph-tier
```

其源码说明就是：

> 把真实 Verify Token 填到已经付费的 graph tier，把 padding slot 转成高 confidence candidate verification，目标是在几乎相同 step time 下获取更多有效验证。

所以 Ragged Verify 的优化目标不是：

```text
Verify Token 越少越好
```

而是：

> **在真实设备 cost curve 下，把有限 Verify Budget 分给最值得验证的 candidate。**

---

## 七、Static 与 Compact：相同 Verify Token 数，也可能不是相同步时

这一步非常重要。

`dspark_sps_profiler.py` 明确记录了一个 caveat：

```text
static:
同样 B 个 verify tokens
来自更少 request

compact:
同样 B 个 verify tokens
通常跨更多 request，
读取更多 KV history
```

所以 static profiling table 用来预测 compact 时：

```text
会略微高估 steps_per_sec
```

这说明性能模型甚至不能只写：

\[
T=f(M)
\]

而更接近：

\[
\boxed{
T=
f(
RequestCount,
VerifyTokenCount,
KVHistoryGeometry,
GraphTier,
ParallelTopology
)
}
\]

因为同样 128 个 Verify Token：

```text
16 requests × 8 tokens/request
```

和：

```text
64 requests × 2 tokens/request
```

会读取完全不同数量、不同长度的 prefix KV，也会产生不同 metadata 与并行布局。

---

## 八、DSpark Budget Planner 已经在优化“Expected Progress / Predicted Step Time”

这是整篇最重要的源码结论。

当前：

[`compute_verify_token_budget()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py)

会拿到：

```text
history_survival_probs
```

把每一个额外 candidate 的“仍然有机会被接受”的概率排序。

随后构造一个预期 useful progress：

```text
tau_star
```

再结合 SPS / additive cost table 预测 Step Time。

additive table 下：

```python
step_time = predicted_step_time(...)
theta = tau_star / step_time
idx = argmax(theta)
```

也就是说，当前 SGLang Ragged DSpark scheduler 实际就在求：

\[
\boxed{
\arg\max_M
\frac{
ExpectedUsefulProgress(M)
}{
PredictedStepTime(B,M)
}
}
\]

这和本文开头的第一性模型完全同构。

所以成熟的 speculative tuning 不是：

```text
maximize acceptance
```

甚至也不是：

```text
minimize verify work
```

而是：

> **让 Expected Useful Progress / Step Cost 最大。**

---

## 九、重新理解 Gamma：Static 中是实际宽度，Ragged 中更像最大搜索半径

Static 模式：

```text
gamma
≈
每个 request 实际的 speculative horizon
```

Ragged 模式：

```text
gamma
=
每个 request 最多允许向未来探索多远
```

实际这一轮 Verify 到哪里：

```text
confidence
+
global token budget
+
graph tier
```

共同决定。

因此以后压测不能只记录：

```text
gamma = 6
```

还需要同时记录：

```text
Configured Gamma
Average Verify Len
P50 / P99 Verify Len
Total Verify Tokens / Step
Graph Num Tokens / Step
Accept Length
Strict Accept Rate
Block Accept Length
```

否则你甚至不知道机器真正算了多少 speculative token。

---

## 十、为什么低并发更容易赚钱，高并发更容易到 Break-even？

低并发时：

```text
Request Parallelism 不够
```

普通 Decode 往往处于：

```text
小 GEMM
低硬件利用率
launch latency 占比高
communication latency 占比高
```

Speculative 把：

```text
B 小, Q=1
```

变成：

```text
B 小, Q>1
```

本质是在沿 Token 维人为制造并行度。

这时经常出现：

```text
Accept Length 增长幅度
>
Step Time 增长幅度
```

所以收益明显。

高并发时，Request Parallelism 已经把机器吃满：

```text
大 GEMM
大 MoE expert batch
高 HBM 利用率
通信 payload 已经够大
```

此时增加 speculative token 不再是填闲置算力，而是在和真实请求竞争已饱和资源。

这就是为什么当前 DeepSeek-V4 cookbook 明确：

```text
low-latency:
更积极 MTP

balanced:
更短 MTP

high-throughput:
MTP disabled
```

Speculative Decoding 更像：

```text
Latency / Low-to-Mid Concurrency Optimization
```

而不是所有 workload 都无条件提高 aggregate throughput。

---

## 十一、实际压测应该怎么做？

至少做四维 Sweep。

### Concurrency

```text
1
2
4
8
16
32
64
128
...
```

### Gamma / Block Size

```text
1
2
3
4
5
6
...
```

### Ragged Mode

```text
static
cap-accept
compact
```

### Workload Geometry

```text
short prompt / long decode
long prompt / short decode
long prompt / long decode
thinking / non-thinking
```

然后至少记录：

| 指标 | 含义 |
| --- | --- |
| `spec_accept_length` | 每次 Verify 真正推进多少输出 |
| strict `spec_accept_rate` | Draft 本身预测质量 |
| accept histogram | 均值背后的分布 |
| avg / P99 `verify_len` | Ragged 实际探索深度 |
| `total_verify_tokens` | 真正 scheduled candidate work |
| `graph_num_tokens` | graph tier / padding |
| step time | speculative critical path proxy |
| TPOT / P99 ITL | 用户侧延迟 |
| output throughput | 服务总吞吐 |
| HBM usage | Speculative 容量代价 |
| DeepEP dispatch pressure | MoE token volume |

最重要的是：

> **不要只看 Acceptance。**

一个配置完全可能：

```text
Accept Length ↑
```

但同时：

```text
Step Time ↑↑
```

最终更慢。

---

## 十二、最终性能模型：从固定窗口升级成 Runtime Cost Surface

初步可以写：

\[
R\approx\frac{AcceptLength}{StepTime}
\]

严格 Review 后，更完整的版本应该是：

\[
\boxed{
R_{request}
\approx
\frac{
E[UsefulTokens]
}{
T(B,M,H,G,P)
}
}
\]

其中：

```text
B = active request count
M = actual verify-token count
H = KV history geometry
G = graph tier / padding
P = TP / DP / EP / communication topology
```

Static：

\[
M=B(\gamma+1)
\]

Ragged：

\[
M=\sum_r verify\_len_r
\]

并且：

\[
M\le graph\_num\_tokens
\]

所以真正的优化目标是：

\[
\boxed{
\max_{\text{verify budget}}
\frac{
Expected Useful Token Progress
}{
Predicted Critical Path Cost
}
}
\]

这正是当前 DSpark Ragged Verify planner 开始在做的事情。

---

## 结语

现在再回答标题：

> **为什么 Speculative Decoding 不一定更快？**

因为它从来不是：

```text
免费得到多个 Token
```

而是：

```text
先花额外成本探索多个未来
        ↓
Target Verify
        ↓
只保留其中真正有用的部分
```

真正决定收益的不是 Acceptance Rate 最大，也不是 Verify Token 最少，而是：

\[
\boxed{
\frac{
Expected Useful Progress
}{
Actual Speculative Critical Path
}
}
\]

最大。

严格 Review 后，可以得到四条特别重要的工程结论：

1. **`accept_length / step_time` 不是低并发专属，但只能在固定 workload geometry 下作为配置 proxy；跨 batch size 比 aggregate throughput 必须把 request count 算进去。**
2. **runtime `spec_accept_rate` 与 Ascend CI 当前名为 `accept_rate` 的 proxy 不是同一个指标。**
3. **Block Size 增大后，被 amortize 的是 fixed / latency-dominated overhead；Target Compute、MoE Token Work、KV/State 与通信字节不会消失，只可能 per-token 更高效。**
4. **Ragged Verify 下，“每个请求固定 Verify N+1”已经不成立；`gamma+1` 是上界，真实 work 是 `sum(verify_lens)`，而真正 wall-clock 还受到 graph tier 与 KV history geometry 影响。**

所以未来真正成熟的 DSpark 调优会越来越接近：

```text
Candidate Survival Probability
        +
Device Cost Curve
        +
Current Batch Geometry
        ↓
Dynamic Verify Budget
```

这也是 Speculative Decoding 从“模型算法技巧”进入“推理调度问题”的真正分界线。

## 源码阅读入口

| 目标 | 固定版本源码 |
| --- | --- |
| `accept_length / step_time` 调参 proxy | [`scripts/playground/bench_speculative.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/scripts/playground/bench_speculative.py) |
| runtime strict Accept Rate / Accept Length | [`tokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/tokenizer_manager.py) |
| Scheduler speculative metrics | [`metrics_reporter.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler_components/metrics_reporter.py) |
| Ascend CI acceptance proxy | [`test_npu_performance_utils.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/test/ascend/e2e/test_npu_performance_utils.py) |
| DeepSeek-V4 NPU performance gate | [`test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/test/registered/npu/performance/deepseek_v4_flash/test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms.py) |
| Ragged Verify 数据结构 | [`ragged_verify.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/ragged_verify.py) |
| Verify Budget / SPS Cost Model | [`dspark_planner.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py) |
| Per-request Verify Length scheduling | [`dspark_schedule.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py) |
| Static / Compact SPS profiling caveat | [`dspark_sps_profiler.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/benchmark/dspark_sps_profiler.py) |
| DSpark / Ragged 配置字段 | [`fields/spec.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/arg_groups/fields/spec.py) |
| DeepSeek-V4 Speculative / DeepEP tuning | [`DeepSeek-V4.mdx`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) |
