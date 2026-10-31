# 投机解码的性能收支：接受长度、Verify 成本与并发

投机解码多做 Draft、Verify 和状态整理，希望换来更少 Target 轮次。是否值得，最终要比较每轮的有效输出和总耗时。

平均接受长度提高时，Verify 行数、Graph padding 或通信成本也可能涨得更快。下面从固定 batch 的产出关系出发，区分接受指标，再看窗口、自适应执行和并发怎样改变这笔账。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5f017ffabb6ab8d214f6a4616ee8bd98a376034a`，核对日期为 2026-09-20。该提交更新了 Ascend/NPU 性能测试框架，并开始将 speculative acceptance 纳入 CI。下文按这一快照解释指标和执行成本。

</details>

## 一、先计算每轮的有效产出

### 一轮成本与输出推进

普通 Decode 每请求一步推进一枚输出，令单步耗时为：

\[
T_{decode}
\]

单请求推进率近似为：

\[
R_{base}\approx\frac{1}{T_{decode}}
\]

投机一轮平均推进 A 个有效输出，总耗时为 T_spec，则：

\[
R_{spec}\approx\frac{A}{T_{spec}}
\]

对应加速比为：

\[
Speedup\approx\frac{A\cdot T_{decode}}{T_{spec}}
\]

要加速，需满足：

\[
\boxed{T_{spec}<A\cdot T_{decode}}
\]

这里比较的是实际产出和周期成本，接受率只解释其中一部分。

分子取本轮有效推进的输出，分母覆盖 Draft、Target Verify、Sampling/Accept、KV/State Commit 和分布式同步组成的完整关键路径。

Graph、多流、调度和通信重叠都会改变最终时长，因此应量整轮关键路径，不能将所有 Kernel duration 直接相加。



### 固定 batch 与总吞吐

固定调参脚本使用以下计算。入口见 [`scripts/playground/bench_speculative.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/scripts/playground/bench_speculative.py)。





```python
step_time = np.percentile(
    server_info["internal_states"][0]["step_time_dict"][str(batch_size)],
    20,
)

speed = 1 / step_time * acc_length
```

其搜索指标为：

\[
speed=\frac{accept\_length}{step\_time}
\]

使用时需先明确它的范围。

#### 它不是 `bs=1` 专用

脚本遍历 `batch_size`、`steps`、`topk`、`num_draft_tokens`，在每个固定 batch 内比较配置，所以这个比例并非只用于 bs=1。

#### 单请求推进速度与整个 Batch 的吞吐

脚本没有乘 batch_size。

若 B 条请求稳定存活，每请求每轮平均推进 A 枚，总推进率约为：

\[
R_{aggregate}\approx\frac{B\cdot A}{T_{step}}
\]

脚本中的：

\[
\frac{A}{T_{step}}
\]

则更接近这个固定工作点的单请求平均推进速度 proxy。

例如固定 B 后，可比较 `1-1-2 / 3-1-4 / 5-1-6` 这些配置。

跨 batch 比总吞吐时，还应纳入 B，并使用相同窗口。

#### 参数搜索指标与端到端服务指标

`step_time` 来自服务端字典，并刻意取第 20 百分位，而不是 P50 或均值。

它适合搜索参数，配置是否保留仍需结合目标并发下的 TTFT、TPOT、P99 ITL、E2E、输出吞吐、失败数和 HBM 占用判断。

SPS profiler 又显式加入这些维度，入口见 [`dspark_sps_profiler.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/benchmark/dspark_sps_profiler.py)。





```text
batch_tokens
=
num_running_reqs_per_rank × verify_num_draft_tokens

steps_per_sec
=
1 / median(server-side step_time)
```

因此同样的 Step Time，需连同请求数、Token 数和历史几何关系解释。

## 二、读懂三个接受指标

Runtime 的严格接受率按下列计数计算，函数见 [`tokenizer_manager.py::_calculate_spec_decoding_metrics()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/tokenizer_manager.py)。





```python
num_proposed_drafts = spec_verify_ct * (
    speculative_num_draft_tokens - 1
)

spec_accept_rate =
    num_correct_drafts / num_proposed_drafts
```

注释 `strict count, no bonus` 限定仅统计 Draft：

\[
\boxed{
spec\_accept\_rate=
\frac{CorrectDrafts}{ProposedDrafts}
}
\]

### Accept Length 是另一件事

同一函数对接受长度采用：

```python
spec_accept_length =
    completion_tokens / spec_verify_ct
```

其注释为 `includes bonus token`。

\[
\boxed{
spec\_accept\_length
=
平均每次 Verify 推进的输出 Token 数
}
\]

例如提议 D1–D5 五枚 Draft，接受 D1–D3，再输出 bonus B。严格接受率为 `3/5=0.60`，每轮有效输出长度 AL 则为 4。

### Ascend Performance CI 还有一个同名 `accept_rate`

固定 NPU CI 框架还有另一个同名 accept_rate。实现见 [`test_npu_performance_utils.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/test/ascend/e2e/test_npu_performance_utils.py)。





```python
accept_rate =
    float(metrics["accept_length"])
    / spec_num_draft_tokens
```

它使用 AL 与配置窗口的比值：



\[
\boxed{
CIProxy=
\frac{AcceptLength}{ConfiguredVerifyWidth}
}
\]

一条 Flash NPU case 配置为：

```text
--speculative-num-draft-tokens 7
--speculative-dspark-block-size 6
accept_rate baseline = 0.5
```

其中 gamma=6，Verify 窗口为 7。

所以 gate 实际要求 `accept_length / 7 >= 0.5`。

即平均 AL 约不低于 3.5，不能改读为严格 Draft 接受率超过 50%。

记录结果时应将三个名字写全：

| 指标 | 分子与分母 |
| --- | --- |
| Strict Accept Rate | `correct_drafts / proposed_drafts`，不含 bonus |
| Accept Length | `outputs / verify step`，含 bonus |
| CI Acceptance Proxy | `accept_length / configured verify width` |

若遥测只返回 AL，就只报告 AL；缺少严格接受率的分子、分母时，不能拿 proxy 补成实测值。

## 三、窗口增大后，哪些成本会变化

### 固定成本与新增 Token 工作

窗口变宽后，要分别看哪些成本可以分摊、哪些仍需新增执行。

#### 可以被 amortize 的部分

每 step 的 Kernel/Graph launch、调度与 Runtime 开销、部分 collective 或 barrier 延迟，以及权重重复读取等固定或增长较慢的成本，可能被更多 Token 分摊。小 GEMM 的利用率也可能随工作量增加而改善。

低 batch、q_len=1 时有效工作较少，Verify 扩宽有机会利用闲置资源，将一次提交的成本分给更多候选。

#### 不会消失的 Token-Proportional Work

Target 的 QKV/Attention Query、Dense/MoE GEMM、路由、dispatch/combine 数据量、KV 写入、C4/C128 压缩状态、LM head/logits，以及 Draft 侧的逐 Token 计算，仍随实际行数增加。总成本会上升，只是每个被验证 Token 分摊的成本可能下降。

不能因为固定成本被分摊，就将整个 Verify 看成近似不变。

#### SGLang 新 SPS Cost Model 已经把这件事写进代码

固定 Planner 的 additive cost model 也同时考虑请求和 Token 数：

\[
T(B,M)=bias+\alpha(B)+\theta(M)
\]

其中 B 是请求数，M 是实际 Verify Token 数。



实现见 [`dspark_planner.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py)。

同样请求数量下，增加 Verify Token 仍会改变成本。



#### MoE / DeepEP 更是硬证据

DeepSeek-V4 cookbook 相应要求：

```text
max-running-requests
×
MTP_draft_tokens

≤
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
```

否则稳态 Dispatch 可能超过容量。

增大 Block 同时增加 MoE 工作、buffer 和 payload 压力。较大消息可能更高效，但总分发工作不会消失。

可进一步抽象为：

\[
T_{step}(B,M)
=
T_{fixed}(B)
+
T_{token}(B,M)
\]

扩宽使：

\[
\frac{T_{fixed}}{M}\downarrow
\]

与此同时：

\[
T_{token}(B,M)\uparrow
\]

两种变化共同决定收益何时转折。

### 实际宽度与搜索上限

Static 下，gamma 近似对应每请求实际展开的 Draft 长度；Ragged 下，它是最多允许探索的上限，实际长度还受置信度、全局 Token 预算与 Graph tier 影响。

所以 gamma=6 不能单独说明做了多少工作。还需记录平均及 P50/P99 Verify Len、每步 Total Verify Tokens、Graph Num Tokens，以及 AL、严格接受率和 Block Accept Length。

分别保留配置、逻辑预算、物理执行与有效产出，才能解释差异。

## 四、自适应 Verify 怎样改变执行成本

### static、cap-accept 与 compact

Runtime 中 gamma 为 Draft block size，`verify_num_draft_tokens = gamma + 1`，额外一行来自旧 root。

`SGLANG_RAGGED_VERIFY_MODE=static` 时，可近似写为：



\[
M_{static}=B(\gamma+1)
\]

固定版本的三种模式如下。定义见 [`ragged_verify.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/ragged_verify.py)。



三种模式分别是 `static`、`cap-accept`、`compact`。

非 static 使用 per-position survival probability、全局预算选择请求前缀，但选定长度不一定减少 Target 行数，cap-accept 仍跑固定窗口，只在接受时设上限。



调度实现见 [`dspark_schedule.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/kernels/ops/speculative/dspark/dspark_schedule.py)。

输出的窗口信息包括：

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

`max_len = gamma + 1`。

同 batch 可安排为：

```text
req0 verify_len = 6
req1 verify_len = 2
req2 verify_len = 1
req3 verify_len = 4
```

统计时区分：

| 模式 / 执行路径 | Target 的逻辑输入行数 | `verify_lens` 的作用 |
| --- | --- | --- |
| `static` | `B*(gamma+1)` | 全窗口验证与接受 |
| `cap-accept` | `B*(gamma+1)` | 限制每个请求可接受的前缀，不缩减 forward |
| 实际进入 compact 路径 | `sum(verify_lens)` | 压紧输入与元数据；Graph 可能另有 padding |

`should_run_compact(layout)` 决定是否真正进入 compact。调了 verify_lens 仍可能走 non-compact，节省多少行要按实际路径计算。

对应 Worker 和 Verify 实现：[`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) [`dspark_verify.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_verify.py)

下面的 Graph tier 讨论限定已进入 compact 的后端，Ascend NPU static 仍按定宽执行。

### 逻辑行数与 Graph Tier

`RaggedVerifyLayout` 保存逐请求 `verify_lens`、逻辑总行数 `total_verify_tokens` 和 Graph 档位 `graph_num_tokens`，满足：

```text
total_verify_tokens = sum(verify_lens)
total_verify_tokens <= graph_num_tokens
```

前者是逻辑工作量，后者是物理执行容量。例如实际需要验证 37 行，Graph tier 仍可能为 48。

从 40 行减为 37 行，若仍落在同一 Graph bucket，耗时未必随之下降。

对齐开关 `--speculative-dspark-align-verify-tokens-to-graph-tier` 会尝试利用这个空间。

它将已付费 tier 内的 padding 换成高置信候选，目标是在近似相同成本下多做有效验证。



因此预算分配需沿真实设备曲线比较边际产出，不能只将行数最小化。





### 相同行数也可能有不同耗时

SPS profiler 还指出，同样数量的 Verify Token 在 static 中可来自较少请求，compact 中却往往跨更多请求，需要读取更多份 KV history。

用 static 表预测 compact，可能略高估 steps_per_sec。

仅按 Token 数估计的模型：

\[
T=f(M)
\]

还需扩为：

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

例如同样 128 行，可分为 `16 requests × 8 tokens/request`，也可为 `64 requests × 2 tokens/request`。



两者涉及不同数量的历史、metadata 和并行组织，所以等行数不保证等耗时。

### Planner 的收益成本估计

Planner 将预期收益与成本共同求解，入口见 [`compute_verify_token_budget()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_planner.py)。



它读取 `history_survival_probs`，按额外候选仍能被接受的概率排序，构造预期有效推进 `tau_star`，再结合 SPS / additive table 估计成本。

additive 路径为：

```python
step_time = predicted_step_time(...)
theta = tau_star / step_time
idx = argmax(theta)
```

相应目标可以表示为：

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

它将开头的产出/成本关系用于选择预算。

接受量、验证行数都只是中间量，最终需要比较预期推进与实际执行成本。







## 五、并发变化怎样改变盈亏平衡

低并发时请求并行不足，普通 Decode 的 GEMM 较小，硬件利用率较低，launch 和通信延迟可能占较大比重。

投机将 `B 小, Q=1` 改为 `B 小, Q>1`，增加同轮 Token 工作。





若新增窗口成本较低，AL 的相对增幅可能超过 Step Time 的相对增幅，此时 `AL / Step Time` 才会上升。

高并发时，GEMM、MoE expert batch 和通信 payload 已经较大，HBM 等资源可能接近饱和。

新增候选要与真实请求竞争同一资源，扩宽可能降低整体吞吐。

固定 cookbook 也给出相关取舍：

```text
low-latency:
更积极 MTP

balanced:
更短 MTP

high-throughput:
MTP disabled
```

投机常用于延迟和低至中并发，但是否改善总吞吐，仍应按目标负载实测。



## 六、设计可比较的压测

### 参数与记录字段

可围绕四个维度选择对照点，先用少量工作点定位转折，再对有价值的区间细化：

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

同时记录：

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

比较时先对齐模型、设备数量、输入输出长度、缓存口径、请求数、并发、到达速率和统计窗口，再判断产出与周期成本的变化。



例如 AL 增加，Step Time 却增加得更多，最终仍会变慢。





### 汇总性能模型

最初的 proxy 为：

\[
R\approx\frac{AcceptLength}{StepTime}
\]

加入负载和执行条件后：

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

这里 B 是活跃请求数，M 是实际 Verify Token 数，H 是 KV 历史几何关系，G 是 Graph tier/padding，P 是 TP/DP/EP 与通信拓扑。

Static 的行数为：

\[
M=B(\gamma+1)
\]

实际 compact 的 Ragged 行数为：

\[
M=\sum_r verify\_len_r
\]

Graph 还可能满足：

\[
M\le graph\_num\_tokens
\]

最终可按以下目标比较预算：

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

将相同负载的有效推进、关键路径和服务指标一起保留，才有依据决定窗口与并发，而不只按最高接受长度选择配置。

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
