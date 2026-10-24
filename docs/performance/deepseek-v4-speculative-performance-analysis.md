# 投机解码的性能收支：接受长度、Verify 成本与并发

投机解码用额外的 Draft 和 Verify 工作，换取更少的 Target 执行轮次。它是否划算，取决于每轮推进了多少有效输出，以及这一轮总共花了多久。

例如，同一批请求的平均接受长度增加了，但 Verify 行数、图填充或通信成本增长得更快，总吞吐仍可能下降。本文从固定 batch 的产出公式开始，依次解释接受指标、窗口大小、compact 执行和并发饱和，再把这些因素放回压测。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码基线**：本文基于 `sgl-project/sglang @ 5f017ffabb6ab8d214f6a4616ee8bd98a376034a`，核对日期为 2026-09-20。这个 commit 本身更新了 Ascend/NPU performance testing framework，并开始把 speculative acceptance 纳入性能 CI。

</details>

## 一、先计算每轮的有效产出

### 一轮成本与输出推进

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

分子取这一轮实际推进的有效输出 Token 数。

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

### 固定 batch 与总吞吐

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

这里需要区分三个边界。

#### 它不是 `bs=1` 专用

脚本会遍历：

```text
batch_size
steps
topk
num_draft_tokens
```

所以 `accept_length / step_time` 不是低并发专属。它会在每个固定 batch size 下用于比较不同 speculative config。

#### 单请求推进速度与整个 Batch 的吞吐

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

跨 Batch 比较总吞吐时，使用上面的 `B × A / T_step`，并保持相同的统计窗口。

#### 参数搜索指标与端到端服务指标

`step_time` 来自 server-side `step_time_dict`，而且脚本刻意使用 20th percentile，而不是 P50/Mean。

所以它更适合：

> **搜索 speculative 参数空间。**

决定线上服务仍要回到：

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

## 二、读懂三个接受指标

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

源码注释：`strict count, no bonus`

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

源码注释：`includes bonus token`

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

所以 CI baseline 的真实含义是：`accept_length / 7 >= 0.5`

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

## 三、窗口增大后，哪些成本会变化

### 固定成本与新增 Token 工作

“Verify 多个 Token 能把成本摊薄”方向没错，但一定要继续问：`摊薄的到底是什么？`

#### 可以被 amortize 的部分

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

#### 不会消失的 Token-Proportional Work

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

#### SGLang 新 SPS Cost Model 已经把这件事写进代码

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

#### MoE / DeepEP 更是硬证据

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

### 实际宽度与搜索上限

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

`gamma=6` 描述配置的候选上限。解释实际工作量时，同时记录：

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

这些量分别描述配置宽度、逻辑预算、实际执行与有效产出。

## 四、自适应 Verify 怎样改变执行成本

### static、cap-accept 与 compact

DSpark runtime 仍然定义：

```text
gamma = draft block size
verify_num_draft_tokens = gamma + 1
```

在：`SGLANG_RAGGED_VERIFY_MODE=static`

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

非 static 模式下，DSpark planner 会使用 confidence head 的 per-position survival probability 和 global token budget，为请求选择不同前缀长度。但 planner 选出的长度不等于 Target 实际执行的行数：`cap-accept` 保留固定宽度 forward，只在接受阶段应用长度上限。

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

其中：`max_len = gamma + 1`

所以同一个 batch 可以变成：

```text
req0 verify_len = 6
req1 verify_len = 2
req2 verify_len = 1
req3 verify_len = 4
```

三种模式应分开计数：

| 模式 / 执行路径 | Target 的逻辑输入行数 | `verify_lens` 的作用 |
| --- | --- | --- |
| `static` | `B*(gamma+1)` | 全窗口验证与接受 |
| `cap-accept` | `B*(gamma+1)` | 限制每个请求可接受的前缀，不缩减 forward |
| 实际进入 compact 路径 | `sum(verify_lens)` | 压紧输入与元数据；Graph 可能另有 padding |

Worker 用 `should_run_compact(layout)` 决定是否调用 compact forward；配置了调度长度，也可能仍走 non-compact 路径。核算节省的计算量时，要看实际执行路径。

固定实现：[`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)、[`dspark_verify.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_verify.py)。

下面沿启用了 compact 的后端分析 Graph Tier。Ascend NPU 的 static 路径按前文固定宽度计算。

### 逻辑行数与 Graph Tier

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

SGLang 甚至提供：`--speculative-dspark-align-verify-tokens-to-graph-tier`

其源码说明就是：

> 把真实 Verify Token 填到已经付费的 graph tier，把 padding slot 转成高 confidence candidate verification，目标是在几乎相同 step time 下获取更多有效验证。

所以 Ragged Verify 的优化目标不是：`Verify Token 越少越好`

而是：

> **在真实设备 cost curve 下，把有限 Verify Budget 分给最值得验证的 candidate。**

### 相同行数也可能有不同耗时

`dspark_sps_profiler.py` 记录了一个会影响成本比较的条件：

```text
static:
同样 B 个 verify tokens
来自更少 request

compact:
同样 B 个 verify tokens
通常跨更多 request，
读取更多 KV history
```

所以 static profiling table 用来预测 compact 时：`会略微高估 steps_per_sec`

这里需要在 Token 数之外加入请求与历史布局。仅以 Token 数表示的模型：

\[
T=f(M)
\]

可扩展为：

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

因为同样 128 个 Verify Token：`16 requests × 8 tokens/request`

和：`64 requests × 2 tokens/request`

会读取完全不同数量、不同长度的 prefix KV，也会产生不同 metadata 与并行布局。

### Planner 的收益成本估计

Planner 将收益估计与执行成本放在一起比较。当前：

[`compute_verify_token_budget()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py)

会拿到：`history_survival_probs`

把每一个额外 candidate 的“仍然有机会被接受”的概率排序。

随后构造一个预期 useful progress：`tau_star`

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

所以成熟的 speculative tuning 不是：`maximize acceptance`

甚至也不是：`minimize verify work`

而是：

> **让 Expected Useful Progress / Step Cost 最大。**

## 五、并发变化怎样改变盈亏平衡

低并发时：`Request Parallelism 不够`

普通 Decode 往往处于：

```text
小 GEMM
低硬件利用率
launch latency 占比高
communication latency 占比高
```

Speculative 把：`B 小, Q=1`

变成：`B 小, Q>1`

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

Speculative Decoding 更像：`Latency / Low-to-Mid Concurrency Optimization`

而不是所有 workload 都无条件提高 aggregate throughput。

## 六、设计可比较的压测

### 参数与记录字段

至少做四维 Sweep。

#### Concurrency

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

#### Gamma / Block Size

```text
1
2
3
4
5
6
...
```

#### Ragged Mode

```text
static
cap-accept
compact
```

#### Workload Geometry

```text
short prompt / long decode
long prompt / short decode
long prompt / long decode
thinking / non-thinking
```

然后至少记录：

| 指标 | 含义 |
| --- | --- |
| `spec_accept_length` | 每次 Verify 推进多少输出 |
| strict `spec_accept_rate` | Draft 本身预测质量 |
| accept histogram | 均值背后的分布 |
| avg / P99 `verify_len` | Ragged 实际探索深度 |
| `total_verify_tokens` | Planner 分配的逻辑预算，需与实际 forward 行数、Graph padding 分开记录 |
| `graph_num_tokens` | graph tier / padding |
| step time | speculative critical path proxy |
| TPOT / P99 ITL | 用户侧延迟 |
| output throughput | 服务总吞吐 |
| HBM usage | Speculative 容量代价 |
| DeepEP dispatch pressure | MoE token volume |

比较配置时保持以下条件一致：

> **不要只看 Acceptance。**

一个配置完全可能：`Accept Length ↑`

但同时：`Step Time ↑↑`

最终更慢。

### 汇总性能模型

初步可以写：

\[
R\approx\frac{AcceptLength}{StepTime}
\]

把这些因素补齐，得到：

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

所以优化目标是：

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

### 固定源码入口

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
