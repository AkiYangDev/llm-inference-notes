# 投机解码的性能收支：接受长度、Verify 成本与并发

一条 Flash NPU CI 的 `accept_rate >= 0.5`，要求的是平均 AL 约不低于 3.5。它把 AL 除以配置窗口 7，和严格 Draft 接受率不是一个指标。先辨清产出怎么算，才能判断投机这一轮多做的工作是否划算。

Draft、Verify、接受和状态整理增加成本，也可能减少 Target 轮次。下面从有效输出除以周期耗时开始，再把接受计数、窗口宽度、Graph 档位和并发放进同一笔账。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 5f017ffabb6ab8d214f6a4616ee8bd98a376034a`，核对日期为 2026-09-20。该提交更新 Ascend/NPU 性能测试框架，并开始将 speculative acceptance 纳入 CI，以下指标按这份快照解释。

</details>

## 一、先计算每轮的有效产出

### 一轮成本与输出推进

普通 Decode 每请求一步推进一枚输出。记单步耗时为：

\[
T_{decode}
\]

对应单请求推进率：

\[
R_{base}\approx\frac{1}{T_{decode}}
\]

投机一轮平均推进 A 枚有效输出，整轮耗时为 T_spec：

\[
R_{spec}\approx\frac{A}{T_{spec}}
\]

与普通路径比较：

\[
Speedup\approx\frac{A\cdot T_{decode}}{T_{spec}}
\]

因此加速条件为：

\[
\boxed{T_{spec}<A\cdot T_{decode}}
\]

接受率影响能推进多少，周期成本决定为这段推进花多久。

分子数有效输出，分母覆盖 Draft、Target Verify、Sampling/Accept、KV/State Commit 和分布式同步的完整关键路径。

Graph、多流、调度、通信重叠会改变最终时长。测量整轮关键路径，才不会把可重叠的 Kernel duration 重复相加。

### 固定 batch 与总吞吐

[`scripts/playground/bench_speculative.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/scripts/playground/bench_speculative.py) 在参数搜索中使用：

```python
step_time = np.percentile(
    server_info["internal_states"][0]["step_time_dict"][str(batch_size)],
    20,
)

speed = 1 / step_time * acc_length
```

搜索指标即：

\[
speed=\frac{accept\_length}{step\_time}
\]

这个 speed 要结合脚本的固定工作点理解。

<a id="bs1"></a>

#### 固定 bs 的配置比较

脚本遍历 `batch_size`、`steps`、`topk`、`num_draft_tokens`，在各自固定 batch 内比较，所以也能用于 bs 大于 1 的工作点。

#### 单请求推进速度与整个 Batch 的吞吐

这里没有乘 batch_size。

假设 B 条请求稳定存活，每请求每轮平均推进 A 枚，合计推进率为：

\[
R_{aggregate}\approx\frac{B\cdot A}{T_{step}}
\]

而脚本中的：

\[
\frac{A}{T_{step}}
\]

更接近这个工作点的单请求平均推进速度 proxy。

B 固定，可以用它比较 `1-1-2 / 3-1-4 / 5-1-6` 等配置。

跨 batch 比总吞吐，则要计入 B，并采用相同统计窗口。

#### 参数搜索指标与端到端服务指标

`step_time` 来自服务端字典，特意取第 20 百分位。

这个值服务于参数搜索。配置是否保留，还要看目标并发下的 TTFT、TPOT、P99 ITL、E2E、输出吞吐、失败数、HBM 占用。

[`dspark_sps_profiler.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/benchmark/dspark_sps_profiler.py) 进一步纳入这些维度：

```text
batch_tokens
=
num_running_reqs_per_rank × verify_num_draft_tokens

steps_per_sec
=
1 / median(server-side step_time)
```

请求数、Token 数、历史几何不同，同一个 Step Time 也可能来自不同的执行工作。

## 二、读懂三个接受指标

[`tokenizer_manager.py::_calculate_spec_decoding_metrics()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/tokenizer_manager.py) 用下面的计数计算 Runtime 严格接受率：

```python
num_proposed_drafts = spec_verify_ct * (
    speculative_num_draft_tokens - 1
)

spec_accept_rate =
    num_correct_drafts / num_proposed_drafts
```

注释 `strict count, no bonus` 说明它只数 Draft：

\[
\boxed{
spec\_accept\_rate=
\frac{CorrectDrafts}{ProposedDrafts}
}
\]

<a id="accept-length"></a>

### AL 统计包括 bonus 的推进长度

同一函数还计算：

```python
spec_accept_length =
    completion_tokens / spec_verify_ct
```

注释为 `includes bonus token`。

\[
\boxed{
spec\_accept\_length
=
平均每次 Verify 推进的输出 Token 数
}
\]

例如提出 D1–D5，接受 D1–D3，再输出 bonus B。严格接受率是 `3/5=0.60`，有效输出长度 AL 为 4。

<a id="ascend-performance-ci-accept_rate"></a>

### NPU CI 的 accept_rate 使用窗口比例

[`test_npu_performance_utils.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/test/ascend/e2e/test_npu_performance_utils.py) 使用另一个同名值：

```python
accept_rate =
    float(metrics["accept_length"])
    / spec_num_draft_tokens
```

它将 AL 除以配置窗口：

\[
\boxed{
CIProxy=
\frac{AcceptLength}{ConfiguredVerifyWidth}
}
\]

其中一条 Flash NPU case 为：

```text
--speculative-num-draft-tokens 7
--speculative-dspark-block-size 6
accept_rate baseline = 0.5
```

这组配置 gamma=6，Verify 窗口为 7。

gate 检查的是 `accept_length / 7 >= 0.5`。

也就是平均 AL 约不低于 3.5，无法据此得出严格 Draft 接受率超过 50%。

记录时可按如下名称区分：

| 指标 | 分子与分母 |
| --- | --- |
| Strict Accept Rate | `correct_drafts / proposed_drafts`，不含 bonus |
| Accept Length | `outputs / verify step`，含 bonus |
| CI Acceptance Proxy | `accept_length / configured verify width` |

遥测只有 AL，就保留 AL。严格接受率缺少实测提出、接受计数时标明缺失，窗口 proxy 继续按自己的定义报告。

## 三、窗口增大后，哪些成本会变化

### 固定成本与新增 Token 工作

窗口增大，部分成本被分摊，同时也增加新的 Token 计算。

<a id="amortize"></a>

#### Launch 等每轮固定成本

Kernel/Graph launch、调度与 Runtime、部分 collective 或 barrier 延迟、重复权重读取，可能被更多 Token 分摊。较小 GEMM 的工作量上升，也可能提高利用率。

低 batch、q_len=1 时，有效工作不足。Verify 扩宽有机会利用空闲资源，使一次提交处理更多候选。

<a id="token-proportional-work"></a>

#### 随 Token 行数增加的工作

QKV/Attention Query、Dense/MoE GEMM、路由、dispatch/combine 数据量、KV 写入、C4/C128 状态、LM head/logits，以及 Draft 计算，都随实际行数增加。每行分摊成本可能下降，总工作却在增加。

Verify 的成本要合计这两部分。

<a id="sglang-sps-cost-model"></a>

#### SPS Additive 模型拆开请求与 Token 成本

固定 Planner 使用：

\[
T(B,M)=bias+\alpha(B)+\theta(M)
\]

B 表示请求数，M 表示实际 Verify Token 数。

实现见 [`dspark_planner.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py)。

同样 B，增加 M 仍会改变估计耗时。

#### MoE / DeepEP 随窗口增加容量需求

DeepSeek-V4 cookbook 要求：

```text
max-running-requests
×
MTP_draft_tokens

≤
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
```

否则稳态 Dispatch 可能超出容量。

Block 增大，MoE 工作量、buffer、payload 压力跟着增加。消息变大也许提高传输效率，仍需承担总分发工作。

把固定与新增部分写成：

\[
T_{step}(B,M)
=
T_{fixed}(B)
+
T_{token}(B,M)
\]

窗口加宽时：

\[
\frac{T_{fixed}}{M}\downarrow
\]

同时：

\[
T_{token}(B,M)\uparrow
\]

两者共同决定窗口收益的转折点。

### 实际宽度与搜索上限

Static 中，gamma 近似对应每请求实际展开的 Draft 长度；Ragged 中，它规定探索上限，实际长度还取决于 confidence、全局预算、Graph tier。

因此记录 gamma=6 之外，还需平均及 P50/P99 Verify Len、每步 Total Verify Tokens、Graph Num Tokens，以及 AL、严格接受率、Block Accept Length。

配置、逻辑预算、物理执行、有效产出各有对应字段，成本变化才有来源。

## 四、自适应 Verify 怎样改变执行成本

### static、cap-accept 与 compact

Runtime 的 gamma 为 Draft block size，`verify_num_draft_tokens = gamma + 1`，旧 root 占额外一行。

`SGLANG_RAGGED_VERIFY_MODE=static` 近似为：

\[
M_{static}=B(\gamma+1)
\]

[`ragged_verify.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/ragged_verify.py) 定义了三种模式。

分别为 `static`、`cap-accept`、`compact`。

非 static 通过 per-position survival probability、全局预算选择请求前缀。cap-accept 仍执行固定 Target 窗口，只在接受时截断；compact 才可能减少 forward 行数。

长度调度见 [`dspark_schedule.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py)。

返回的窗口信息为：

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

其中 `max_len = gamma + 1`。

同一 batch 可以得到：

```text
req0 verify_len = 6
req1 verify_len = 2
req2 verify_len = 1
req3 verify_len = 4
```

对应计数为：

| 模式 / 执行路径 | Target 的逻辑输入行数 | `verify_lens` 的作用 |
| --- | --- | --- |
| `static` | `B*(gamma+1)` | 全窗口验证与接受 |
| `cap-accept` | `B*(gamma+1)` | 限制每个请求可接受的前缀，不缩减 forward |
| 实际进入 compact 路径 | `sum(verify_lens)` | 压紧输入与元数据；Graph 可能另有 padding |

是否压紧，最终由 `should_run_compact(layout)` 决定。verify_lens 改变以后仍可能运行 non-compact，节省行数要从实际 forward 统计。

执行路径见 [`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)、[`dspark_verify.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_verify.py)。

接下来的 Graph tier 分析针对已进入 compact 的 backend，Ascend NPU static 继续按定宽执行。

### 逻辑行数与 Graph Tier

`RaggedVerifyLayout` 保存各请求 `verify_lens`、逻辑总行数 `total_verify_tokens`、Graph 档位 `graph_num_tokens`：

```text
total_verify_tokens = sum(verify_lens)
total_verify_tokens <= graph_num_tokens
```

逻辑需要 37 行，物理 Graph 档位却可能为 48。

从 40 减到 37，只要还落在相同 bucket，耗时就未必下降。

`--speculative-dspark-align-verify-tokens-to-graph-tier` 会尝试利用 tier 内剩余空间。

它把部分 padding 换成高置信候选，希望在近似相同成本下增加有效推进。

预算因此需要沿设备真实曲线比较边际产出；行数最少不一定每秒输出最多。

### 相同行数也可能有不同耗时

SPS profiler 指出：同样 Verify 行数，static 可能来自较少请求，compact 则分布到更多请求，涉及更多份 KV history。

拿 static 表预测 compact，就可能高估 steps_per_sec。

仅按总 Token 数建模：

\[
T=f(M)
\]

还不足以区分这些工作，应继续考虑：

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

例如同样 128 行，可以是 `16 requests × 8 tokens/request`，也可以是 `64 requests × 2 tokens/request`。

后者涉及更多请求历史与 metadata，并行组织也可能不同。

### Planner 的收益成本估计

[`compute_verify_token_budget()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py) 将预期推进与预计成本共同用于预算求解。

它读取 `history_survival_probs`，按额外候选的前缀接受概率排序，得到预期推进 `tau_star`，再从 SPS / additive table 估计成本。

additive 路径为：

```python
step_time = predicted_step_time(...)
theta = tau_star / step_time
idx = argmax(theta)
```

相应优化目标为：

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

这是把产出/成本关系用于运行时预算选择。

估计选出预算后，实际收益仍由本轮有效推进、整轮耗时验证。

## 五、并发变化怎样改变盈亏平衡

低并发的普通 Decode 只有较小 GEMM，请求并行不足，launch、通信延迟可能占较大比重。

投机把 `B 小, Q=1` 扩为 `B 小, Q>1`，用同轮候选增加工作。

AL 的相对增幅超过 Step Time 的相对增幅，`AL / Step Time` 才提高。

高并发时，GEMM、MoE expert batch、通信 payload 已较大，HBM 等资源可能接近饱和。

候选和真实请求竞争同一资源，窗口加宽就可能降低整体吞吐。

固定 cookbook 给出的取舍为：

```text
low-latency:
更积极 MTP

balanced:
更短 MTP

high-throughput:
MTP disabled
```

因此延迟、低至中并发和服务总吞吐，需要在各自目标负载下比较。

## 六、设计可比较的压测

### 参数与记录字段

先选少量并发、窗口、执行模式、负载工作点，找到转折，再细化值得保留的区间：

<a id="concurrency"></a>
<a id="gamma-block-size"></a>
<a id="ragged-mode"></a>
<a id="workload-geometry"></a>

| 维度 | 可选对照 |
| --- | --- |
| 并发 | 1、2、4、8、16、32、64、128…… |
| Gamma / Block Size | 1、2、3、4、5、6…… |
| Ragged Mode | static、cap-accept、compact；以设备和后端实际支持范围为准 |
| 负载 | 短输入/长输出、长输入/短输出、长输入/长输出，以及 thinking/non-thinking |

每组保留这些字段：

| 指标 | 含义 |
| --- | --- |
| `spec_accept_length` | 每次 Verify 推进多少输出 |
| strict `spec_accept_rate` | Draft 本身预测质量 |
| accept histogram | 均值背后的分布 |
| avg / P99 `verify_len` | Ragged 实际探索深度 |
| `total_verify_tokens` | Planner 分配的逻辑预算，需与实际 forward 行数、Graph padding 分开记录 |
| `graph_num_tokens` | graph tier / padding |
| step time | speculative critical path proxy |
| TTFT / TPOT / P99 ITL / E2E | 用户侧首 Token、生成间隔与完整请求延迟 |
| 失败数 | 对照成功请求数与总请求数，保留错误原因 |
| output throughput | 服务总吞吐 |
| HBM usage | Speculative 容量代价 |
| DeepEP dispatch pressure | MoE token volume |

对照组采用相同模型、设备数量、输入输出长度、缓存口径、请求数、并发、到达速率、统计窗口。这样才可把产出、周期变化归因到本次配置。

如果 AL 上升，Step Time 的增幅更大，结果仍会变慢。

### 将测量字段带回性能模型

固定工作点先比较：

\[
R\approx\frac{AcceptLength}{StepTime}
\]

跨负载时，分母还依赖具体执行条件：

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

B 是活跃请求数，M 是实际 Verify Token 数，H 是 KV 历史几何，G 是 Graph tier/padding，P 是 TP/DP/EP 和通信拓扑。

Static 对应：

\[
M=B(\gamma+1)
\]

实际进入 compact 的 Ragged 对应：

\[
M=\sum_r verify\_len_r
\]

Graph 执行又可能有：

\[
M\le graph\_num\_tokens
\]

这些字段共同约束预算选择：

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

最后将同负载的有效推进、关键路径、服务指标并排，决定窗口与并发配置。

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
