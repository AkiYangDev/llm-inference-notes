# 在 Ascend 910C 上分析 DSpark：从阶段计时到 NPU 时间线

DSpark 开启后，吞吐变化只能说明整段服务结果。要决定接着优化哪里，还需知道 Draft、Verify 的计时范围，Accept、Commit 是否在图内，以及主机准备和通信等待怎样进入一轮耗时。

下面沿 Profiling 顺序建立阶段边界、选择稳定 Decode 窗口、对齐 NPU 时间线，再用对照实验检验判断。算子累计时长有助于查工作量，归因则需回到阻塞下一次输出的依赖。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 sgl-project/sglang @ ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e，核对日期为 2026-09-21。主线为 SGLang SRT 文本推理、V4 DSpark、Ascend 910C。以下解释均针对该版本，设备截图与毫秒数需目标环境实测，本文只提供源码分析及采集方法。

</details>

## 一、把一轮 DSpark 对应到计时字段

### 一轮执行的起止

Decode 主线可展开为：[DSparkWorkerV2](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

~~~mermaid
flowchart TD
    A[Committed history] --> B[Allocate verify window]
    B --> C[Draft proposal]
    C --> D[Confidence / verify budget]
    D --> E[Verify layout]
    E --> F[Target Verify]
    F --> G[Accept / Finalize]
    G --> H[Publish new_seq_lens]
    H --> I[Commit target hidden / state]
    I --> J[Build next draft input]
    J --> A
~~~

源码给 Draft、Target Verify 插入阶段边界：

~~~python
with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
    proposal = self._proposer.propose(...)

with self._observers.segment(InfoSegment.TARGET_VERIFY):
    target_verify = ...
~~~

因此可先分为：

~~~text
DSpark Step
├── Draft segment
├── Target Verify segment
└── Segment 外部的 Accept / Commit / 其他工作
~~~

逻辑上常写 `T_step = T_draft + T_verify + T_accept + T_commit`。

但这些是职责分类。异步多流可重叠，Event 测起止区间，Kernel duration 则累计各自工作，二者不能直接换算墙钟。

需要分别看：

~~~text
Kernel Duration Sum
        ≠
Stage Elapsed Time
        ≠
End-to-End Step Wall Time
~~~

### Event 时间覆盖什么

`STEP`、`DRAFT`、`TARGET_VERIFY` 通过设备 Event 记录起止。实现见 [dspark_observability.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_observability.py)，主要动作如下：

~~~python
start = torch.cuda.Event(enable_timing=True)
start.record()

...

end = torch.cuda.Event(enable_timing=True)
end.record()
~~~

读取记录时，才等待 End 并计算 elapsed：

~~~python
end.synchronize()
elapsed_ms = start.elapsed_time(end)
~~~

<a id="npu-transfer_to_npu"></a>

这里的 CUDA 风格命名经过 NPU 兼容映射。标准 SRT NPU 初始化先执行 `from torch_npu.contrib import transfer_to_npu`，再用 `torch.cuda.is_available = lambda: False` 覆盖 capability 判断，避免兼容层使它误报 CUDA 可用。[npu/utils.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/utils.py)

Ascend PyTorch 的兼容模块包含 `torch.cuda` 到 `torch_npu.npu` 的接口映射，因此这条路径中的 `torch.cuda.Event` 实际使用 NPU Event。[transfer_to_npu.py](https://github.com/Ascend/pytorch/blob/77037e26c8b39b4d5e1856bb23d601b67f7ee5a9/torch_npu/contrib/transfer_to_npu.py)

<a id="event-elapsed-kernel-duration"></a>

`record()` 将起止 Event 放进当前执行流，`draft_gpu_ms` 测的是 Draft 入口到出口在该流上的设备 elapsed，Target Verify 同理。side stream 若在 End 之前通过依赖汇合，其工作或等待就会影响这段时间；若工作跨过 End，它的完整 Kernel duration 则超出了这个 segment。

~~~text
Current Stream:

Draft Start Event
      │
      ├──────── Draft 主链工作 ─────────┐
      │                                 │
      │     Side / Comm Stream          │
      │     ───────────────             │
      │             │                   │
      │             └─ 若 End 前 join   │
      │                                 │
Draft End Event ◄───────────────────────┘
~~~

所以 Event elapsed 可以作为源码阶段的设备计时参照，却不等于 Python wall time、所有流 Busy Time 之和，也不等于全部 Draft / Verify Kernel duration 的累计。比较阶段时间与算子统计时，要先对齐各自覆盖的区间。

<a id="endsynchronize-draft-verify"></a>

还要区分记录和读取：segment 退出时只记录 End，`end.synchronize()` 在之后的 `_drain_pending()` 中执行。因此这段计时不会在 Draft→Verify 边界立即插入 Device Synchronize。详细观测仍有成本，常规压测与 Profile 结果应分别保留。

<a id="gpu"></a>

可以同时保留 `step_gpu_ms`、`draft_gpu_ms`、`target_verify_gpu_ms`，但不能预设后两项相加就是整步时间。固定 Ascend 路径的 Accept、Finalize、TP Sync、Commit 位于两个 segment 之外，这些尾部还需单独定位。

## 二、采集稳定的 Decode 窗口

### 先看 DSpark Observability

先用框架阶段记录找到稳定慢窗口，再针对它采设备 trace。



可用记录项为：

~~~text
core
step_cpu_time
step_gpu_time
draft_gpu_time
target_verify_gpu_time
reqs
~~~

在 Profile 环境可设置：

~~~bash
export SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time,reqs
~~~

DecodeStepRecord 包含以下字段。[DecodeStepRecord](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_observability.py)

~~~text
forward_ct
bs
mode
num_verify_tokens
verify_tokens_local
verify_tokens_dp_synced
verify_tokens_graph_key

step_cpu_ms
step_gpu_ms
draft_gpu_ms
target_verify_gpu_ms

request-level:
prefix_len
verify_len
acc_len
correct_drafts
commit_lens
...
~~~

阶段表可以先这样记录：

| Step | BS | Verify Tokens | Step GPU | Draft GPU | Verify GPU | Unattributed Step Time | Avg Commit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N | ... | ... | ... | ... | ... | ... | ... |
| N+1 | ... | ... | ... | ... | ... | ... | ... |

其中差值为：

~~~text
Unattributed Step Time
=
Step GPU
-
Draft GPU
-
Target Verify GPU
~~~

它帮助发现计时段外的变化，具体属于哪项仍需时间线。

当前 NPU Accept/Commit 在 Verify 外，这个差值尤其值得对齐。

#### step_cpu_ms 不是 NPU Event 时间

step_cpu_ms 来自主机 monotonic clock 的步间隔。

出现 `step_cpu_ms > step_gpu_ms`，并不能单独证明 Event 或 Profiler 有误。



两者之间还可能包含：

~~~text
Host preparation
Scheduler
CPU-side wait
D2H / synchronization
Observability overhead
~~~

### 采集 SGLang Ascend Trace

Ascend 采集入口见 [Ascend Performance Profiling Guide](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx)。



例如先设置：

~~~bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/dsv4_dspark_profile
~~~

让加载、通信初始化、Graph Capture、Warmup、Prefill 和初始 Decode 完成，再采稳定部分。

采集请求示例为：

~~~bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{
    "output_dir": "/tmp/dsv4_dspark_profile",
    "start_step": 3,
    "num_steps": 10,
    "activities": ["CPU", "GPU"],
    "detailed_annotations": true
  }'
~~~

CPU/GPU activity 名称同样由 torch_npu 兼容映射为对应 NPU activity。

#### 为什么 detailed_annotations 很重要

BS 只描述请求数，Verify 工作还随下列条件改变：



~~~text
verify width / verify_lens
prefix length
ragged layout
graph padding
N_Q
N_KV
~~~

所以相同 BS 的两步未必执行相同设备工作。

### 进入设备时间线

内部记录先帮助缩小范围：是 Draft、Verify 变慢，还是两个 segment 以外的时间增加？进入设备流后，先把同一轮工作对应起来，再追具体等待。

<a id="step-wall-window"></a>
<a id="forward_ct-bs-verify-tokens-dspark-record"></a>

选择稳定的 Step N、N+1、N+2，界定各步的墙钟区间，再以 `forward_ct`、BS 和 Verify Token 数对应 DSpark record。相同 BS 未必有相同验证宽度或历史长度，不能把不同负载的步骤直接比较。

<a id="target-verify-eager-tail"></a>

固定 NPU 路径的 Accept/Commit 没有 folded 进 Verify Graph。若未归属差值变大，先确定新增区间是否出现在 Verify Graph 结束之后，再定位 Accept、Sync、Commit 的具体工作。

<a id="replay-host-gap-graphupdate"></a>

Replay 前的 Host gap 则沿 batch、buffer、metadata、bucket 与 launch 查起。DSV4 已绕过 generic `graph.update`，下一节会展开这一分支差异。

<a id="deepep"></a>

通信需要同时标出发起、in-flight 和 hook / wait 三个位置。到消费点时数据若已到，暴露等待可能很少；仍在等待的部分才会延长关键路径。仅看通信任务总时长，无法还原本轮被它阻塞多久。

## 三、区分 Verify Graph 与图外工作

### Accept 与 Commit 的位置

共享实现提供 `DsparkVerifyEpilogue`，支持将接受、Finalize、输出与部分 Commit 注入 Verify 捕获。Ascend 是否使用它，取决于 Worker 创建条件。相关实现见 [dspark_verify.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_verify.py) 和 [dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)。

<a id="guard-is_cuda"></a>

~~~python
if (
    (self._verify_planner.is_compact_mode or static_epilogue_supported)
    and self._decode_graph_allowed
    and is_cuda()
):
    self._verify_epilogue = DsparkVerifyEpilogue(...)
~~~

这里要求真实的 `is_cuda()`，而 NPU 初始化已把 `torch.cuda.is_available` 设回 False。因此固定 Ascend 910C 路径不创建 `DsparkVerifyEpilogue`。`fold_eligible` 又先要求 `verify_epilogue is not None`，所以 `fold_eligible`、`folded_accept`、`folded_commit` 均为 False。

<a id="folded_accept-folded_commit-npu-false"></a>

Target Verify 后仍需执行下面这段 eager tail：

~~~text
Target Verify
      ↓
accept_and_finalize() eager path
      ↓
accept_draft_tokens
      ↓
SpecTpSync
      ↓
FinalizeAcceptLens
      ↓
BuildOutTokens
      ↓
commit_hidden()
~~~

Accept、Finalize、SpecTpSync、Commit 因而不在 `target_verify_gpu_ms` 内。可以先用 `step_gpu_ms - draft_gpu_ms - target_verify_gpu_ms` 观察 segment 之外的变化，但差值还可能包含规划和阶段间隙，多流也会影响加法关系。它只能标为未归属时间，不能全部算成 Commit。

继续在 trace 中对齐 Accept kernels、SpecTpSync、Finalize Accept Lens、BuildOutTokens、KV/hidden commit，以及其他 segment 外设备工作，才能把差值归到具体边界。

<a id="cuda-folded-path"></a>

CUDA folded path 只能用于代码对照。它还要求 proposal folded、Greedy、无额外 logits adjustment、无模拟接受、无 Grammar、Graph 可运行和合适模式；Commit 还需 `epilogue.folds_commit` 成立。这些条件不描述当前 Ascend 910C 的执行时间线。

### Replay 前的准备工作

NPU Runner 定义了 TARGET_VERIFY 的 `actual_seq_kvlen`，generic backend 也实现了 update，但这些接口存在，并不代表 DSV4 每次 replay 都调用它们。先看 [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) 的模型分派，再看实际进入的 backend。

<a id="generic-npu-graph-update"></a>

generic 路径提交 update 到线程后，马上等待 `update_future.result()`，完成后才 replay。对使用该分支的模型，update 仍是 Host 前置依赖，可能形成 pre-replay gap。[npu_cudagraph_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py)

~~~python
update_future = self._update_executor.submit(
    graph.update,
    cpu_update_input=cpu_update_input,
)
update_future.result()
graph.replay()
~~~

<a id="deepseek-v4"></a>

DSA/V4 被明确排除在这条路径之外：

~~~python
if not (
    is_deepseek_dsa(...)
    or is_deepseek_v4(...)
):
    output = self.backend.replay_with_input_update(...)
else:
    output = self.backend.replay(graph_key, forward_batch)
~~~

V4 直接调用 `backend.replay(...)`，不走 `replay_with_input_update(...)`。因此发现 DSV4 Host 间隙时，应追它自己的准备和等待，而不是先归给 generic `graph.update`。

<a id="dsv4-kv-query-metadata-backend"></a>

绕过 generic update 仍需刷新实时数据。DSV4 backend 根据模式和 live batch 更新 `actual_seq_lengths_q`、`actual_seq_lengths_q_pa`、`actual_seq_lengths_kv`、block tables 和 kernel metadata；Verify 还结合 `draft_token_num` 形成 Query 几何。[ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)

这条路径可概括为刷新 graph-bound buffers / DSV4 metadata，再调用 `backend.replay(graph_key, forward_batch)`，与 `NPUGraph.update(actual_seq_kvlen)` 后 replay 的流程不同。

<a id="dsv4-replay-host-gap"></a>

定位 Replay 前的空隙时，可按实际执行顺序检查：

1. `ScheduleBatch` / `ForwardBatch` 准备。
2. Graph bucket / shape key 选择。
3. `input_ids`、`positions`、`seq_lens`、`out_cache_loc` 的刷新。
4. DP 与 speculative metadata 准备。
5. DSV4 Attention metadata。
6. Replay launch。
7. allocator、Python 或隐含同步点。

找到具体阻塞后，再用时间线判断它是否限制了下一轮提交。

## 四、在时间线上寻找暴露的通信

### Launch、独立计算与 Wait

`async_finish`、`recv_hook` 可以推迟等待，但中间还要有独立工作，才能把通信覆盖起来。仅凭接口异步，不能判断设备时间缩短了多少。

<a id="ascend-zbal-dispatcher"></a>

NPU 配置 ZBAL local memory 时，Dispatcher 按下面的分支选择 Buffer。上层仍使用 `dispatch_a/b`、`combine_a/b` 状态机，替换的是底层通信资源。[deepep.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)

~~~python
if _use_zbal:
    from zbal.zbal.deepep_adaptor import Config
    from zbal.zbal_buffer import Buffer
else:
    from deep_ep import Buffer, Config
~~~

<a id="deepep-mode-auto-decode-target_verify-low_latency"></a>

`DeepEPMode.resolve()` 将 `--deepep-mode auto` 在 Decode / TARGET_VERIFY 中解析为 low_latency，在 Prefill / Extend 中解析为 normal。分析 DSpark Verify，应继续追 low-latency 的 recv-hook。[DeepEPMode.resolve()](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/utils.py)

~~~python
if is_extend_in_batch:
    return DeepEPMode.NORMAL
else:
    return DeepEPMode.LOW_LATENCY
~~~

<a id="normal-modeasync_finish-stream-dependency"></a>

Normal 模式的 `dispatch_a` 先捕获 previous_event，`dispatch_b` 再取得布局并 dispatch；`async_finish` 使用 `event.current_stream_wait()` 安排依赖，Combine 类似。通信可以先排进 comm stream，当前流到需要结果的位置再等待，但同批专家计算始终依赖收到的数据。

<a id="low-latency-moderecv_hook"></a>

Low-latency 设 `async_finish=False`、`return_recv_hook=True`。`dispatch_a` 发起通信并返回 hook，`dispatch_b` 到消费点调用 hook，确认 recv data 可用。Combine 也可将等待推迟到相应消费点：

~~~text
dispatch_a
   ↓
issue RDMA / communication
   ↓
return hook
   ↓
【理论上的 overlap window】
   ↓
dispatch_b
   ↓
hook()
   ↓
recv data 可消费
~~~

发起与消费之间若能安排独立计算，就有重叠机会。改变的是等待位置，不代表传输本身更快。

<a id="deepseek-v4-decode-target_verify-tbo"></a>

TBO 可以利用这种 A/B 边界，让一组 batch 在通信时切换到另一组计算。下面只是机制示意：

~~~text
Batch A dispatch_a
        ↓
Yield
        ↓
Batch B compute
        ↓
Batch A dispatch_b
~~~

固定 V4 策略只支持 EXTEND。源码注明 `DeepseekV4 TBO only supports prefill (EXTEND)`，也记载 Decode TBO 未实现及相关 regression、Graph capture 问题；这些是上游注释，不是本文新执行的实验。[operations_strategy.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/batch_overlap/operations_strategy.py)

<a id="side-stream-work"></a>
<a id="shared-expert-routed-expert-overlap"></a>
<a id="low-latency-combine-overlap_args"></a>

因此在 Decode / Verify 中，还要寻找其他实际触发的重叠：不依赖当前 recv 的 side-stream 计算、shared expert 与 routed expert 的并行，或已由上层配置的 low-latency Combine `overlap_args`。它们与 TBO、recv_hook 属于不同机制，需分别确认触发条件和汇合位置。

最后将 `T_comm_total` 与 `T_comm_exposed` 分开：关键路径受仍然暴露的等待限制，不是通信 duration 的机械总和。

### 把各阶段放回同一轮

将一轮工作放回时间线：

~~~text
Host / Scheduler
│
├─ prepare batch / verify window
│
├──────────────── Device current stream ────────────────────
│
│   [Draft Start Event]
│          │
│          ├── Draft Model / Proposal
│          │      └─ 可能存在 side / comm stream
│          │
│   [Draft End Event]
│
│   planner / layout / metadata preparation
│
│   [Target Verify Start Event]
│          │
│          ├── NPU Graph Replay or eager Target Forward
│          │      ├── DSV4 Attention
│          │      ├── MoE / DeepEP
│          │      └── Logits / Hidden
│          │
│   [Target Verify End Event]
│
│   Accept / Finalize          ← 当前 NPU 不 folded
│      ├── accept kernels
│      ├── SpecTpSync
│      ├── finalize accept lens
│      └── build output tokens
│
│   Commit                     ← 当前 NPU 不 folded
│      └── target hidden / KV injection
│
└─ publish / next draft metadata
~~~

其中 DSV4 replay 前的实际关系为 live metadata / bound buffers→replay。



可据此给 msprof / MindStudio 标阶段。

## 五、把耗时与有效输出放在一起

整轮成本可写为：

~~~text
T_step_wall
=
CriticalPath(
    Draft main/side streams,
    Target Verify main/side streams,
    Communication,
    Eager Accept,
    TP Sync,
    Eager Commit,
    Host gaps
)
~~~

再结合有效产出：

~~~text
Effective Progress Rate
=
Committed Tokens
/
Step Wall Time
~~~

单请求近似为：

~~~text
Effective TPOT
≈
Step Wall Time
/
Average Commit Length
~~~

AL 增加也可能伴随更多周期成本。

若扩大 Block 导致：

~~~text
Draft Cost ↑
Verify Query Tokens ↑
MoE Work ↑
DeepEP Exposed Comm ↑
Commit Cost ↑
~~~

超过有效提交增加，最终仍更慢。

## 六、设计对照实验并核对结果

### 控制实验变量

| 实验 | 控制变量 | 主要观察 |
| --- | --- | --- |
| Spec OFF vs DSpark ON | workload 完全相同 | DSpark 是否降低 TPOT / 提高 output throughput |
| NPU Graph ON vs OFF | 仅改 Graph | Device bubble、Host launch gap、Step Wall Time |
| DSpark Block Size 3 / 6 / 9 | 仅改 block | Draft GPU、Verify GPU、Avg Commit、Effective TPOT |
| DeepEP low_latency 对照 | 通信配置 | Total Comm、Exposed Comm、hook / wait 位置 |
| Low vs High Concurrency | 模型参数不变 | 瓶颈是否从 Host/Graph 转向 MoE/Communication |

各组保留：

~~~text
SGLang commit
model checkpoint
server args
TP / DP / EP layout
concurrency
input / output length
DSpark config
Graph config
DeepEP mode
Profiler config
~~~

对齐运行条件后，再比阶段与依赖变化。

### 按现象选择检查点

| Timeline 现象 | 先检查什么 | 一起对齐的量 |
| --- | --- | --- |
| `draft_gpu_ms` 变长 | Draft workload、模型与 side stream | Event 范围内的设备工作与等待 |
| Verify 时间变长 | Verify Token 数、Prefix 与 Graph Bucket | 实际 Forward 行数与历史长度 |
| Verify 后尾部变长 | Eager Accept、TP Sync、Commit | 各段开始和结束位置 |
| Replay 前 Host gap | metadata、buffer、Scheduler | DSV4 实际调用路径 |
| DeepEP 通信长 | recv hook 与 Wait | 总通信时长、等待时剩余工作 |
| 多流出现重叠 | 汇合点在等待谁 | 关键路径与整轮耗时 |
| Kernel 累计时长增加 | 并发工作是否增加 | 累计工作量与墙钟时间 |

可以按下面路径推进归因：

~~~mermaid
flowchart TD
    A[TTFT / TPOT / Throughput] --> B[DSpark step record]
    B --> C{哪段变慢}
    C --> D[Draft]
    C --> E[Target Verify]
    C --> F[阶段外时间差]

    D --> G[SGLang torch_npu trace]
    E --> G
    F --> G

    G --> H[NPU Graph / Eager boundary]
    H --> I[CANN / NPU streams]
    I --> J[DeepEP / ZBAL / HCCL]
    I --> K[Attention / MoE / KV kernels]

    J --> L[Critical Path]
    K --> L
    L --> M[Step Wall Time]
    M --> N[Committed Tokens]
    N --> O[Effective TPOT]
~~~

最后回到同轮记录：哪段阻塞下一次输出、调整后关键路径缩短多少、提交长度是否改变。阶段外 Eager、通信和 Host 准备都需落实到耗时与产出。

### 固定源码入口

| 问题 | 固定源码入口 |
| --- | --- |
| DSpark Decode 总状态机 | [dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) |
| Draft Proposal | [dspark_draft.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_draft.py) |
| Verify / Accept / Commit | [dspark_verify.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_verify.py) |
| Draft / Verify Event Timing | [dspark_observability.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_observability.py) |
| DSpark SPS Record | [dspark_sps_profiler.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/benchmark/dspark_sps_profiler.py) |
| NPU Compatibility Init | [npu/utils.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/utils.py) |
| torch.cuda → torch_npu compatibility | [Ascend PyTorch transfer_to_npu.py](https://github.com/Ascend/pytorch/blob/77037e26c8b39b4d5e1856bb23d601b67f7ee5a9/torch_npu/contrib/transfer_to_npu.py) |
| NPU Graph Runner | [npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) |
| NPUGraph Replay / Update | [npu_cudagraph_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py) |
| DSV4 Ascend Attention Metadata | [ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py) |
| DeepEP / ZBAL Dispatcher | [deepep.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) |
| DeepEP Mode AUTO | [moe/utils.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/utils.py) |
| DeepSeek-V4 TBO 能力边界 | [operations_strategy.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/batch_overlap/operations_strategy.py) |
| SGLang Ascend Profiling | [profiling.mdx](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx) |

### 延伸阅读

相关篇目可按以下顺序连接：

1. [DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject](../speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)
2. [DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？](deepseek-v4-speculative-performance-analysis.md)
3. [Ascend 910C 投机解码：Graph、多流与通信的执行依赖](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)

稳定窗口内，将 forward_ct、阶段时间、未归属差值、暴露通信、Replay 间隙、Commit Length、TPOT 并列，才能比较每轮实际成本。
