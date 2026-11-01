# 在 Ascend 910C 上分析 DSpark：从阶段计时到 NPU 时间线

`target_verify_gpu_ms` 没有覆盖 Ascend 上的一整轮 DSpark。本文固定版本里，Accept、Finalize、TP Sync 和 Commit 都在 Verify 计时段之外。Verify 看起来不慢，下一轮输出也可能还在等这些尾部工作。

阶段 Event、主机计时、算子累计时长，覆盖的范围又各不相同。把它们对应到同一轮稳定 Decode，才能找到挡住下一次输出的依赖。这篇沿源码解释这些边界，并给出采集与对照方法。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 sgl-project/sglang @ ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e，核对日期为 2026-09-21。主线为 SGLang SRT 文本推理、V4 DSpark、Ascend 910C。以下解释均针对该版本，设备截图与毫秒数需目标环境实测，本文只提供源码分析及采集方法。

</details>

## 一、把一轮 DSpark 对应到计时字段

### 一轮执行的起止

Decode 主线由 [DSparkWorkerV2](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) 组织，执行关系如下。

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

源码在 Draft、Target Verify 入口和出口插入阶段记录。

~~~python
with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
    proposal = self._proposer.propose(...)

with self._observers.segment(InfoSegment.TARGET_VERIFY):
    target_verify = ...
~~~

按职责可以分出下面几部分。

~~~text
DSpark Step
├── Draft segment
├── Target Verify segment
└── Segment 外部的 Accept / Commit / 其他工作
~~~

逻辑分解常写成 `T_step = T_draft + T_verify + T_accept + T_commit`。

执行时各部分未必串行。异步多流可以重叠，Event 量起止区间，Kernel duration 累计工作量，它们不能直接互换成墙钟时间。

这几种时间的覆盖范围如下。

~~~text
Kernel Duration Sum
        ≠
Stage Elapsed Time
        ≠
End-to-End Step Wall Time
~~~

### Event 时间覆盖什么

`STEP`、`DRAFT`、`TARGET_VERIFY` 使用设备 Event 记录起止，[dspark_observability.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_observability.py) 中的主要动作如下。

~~~python
start = torch.cuda.Event(enable_timing=True)
start.record()

...

end = torch.cuda.Event(enable_timing=True)
end.record()
~~~

之后读取记录，才等待 End、计算 elapsed。

~~~python
end.synchronize()
elapsed_ms = start.elapsed_time(end)
~~~

<a id="npu-transfer_to_npu"></a>

CUDA 风格名字经 NPU 兼容层映射。标准 SRT NPU 初始化先 `from torch_npu.contrib import transfer_to_npu`，再用 `torch.cuda.is_available = lambda: False` 恢复 capability 判断，防止兼容层使其误报 CUDA 可用。[npu/utils.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/utils.py)

Ascend PyTorch 的兼容模块把部分 `torch.cuda` 接口映射到 `torch_npu.npu`，因此这里的 `torch.cuda.Event` 实际落到 NPU Event。[transfer_to_npu.py](https://github.com/Ascend/pytorch/blob/77037e26c8b39b4d5e1856bb23d601b67f7ee5a9/torch_npu/contrib/transfer_to_npu.py)

<a id="event-elapsed-kernel-duration"></a>

`record()` 把起止 Event 放到当前执行流。`draft_gpu_ms` 量 Draft 入口到出口在该流上的设备 elapsed，Target Verify 同理。side stream 若在 End 前通过依赖汇合，它的工作或等待就会进入区间；工作若跨过 End，完整 Kernel duration 则超出这个 segment。

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

所以 Event 可以对应源码阶段的设备时间，却既不是 Python wall time，也不是各流 Busy Time 之和，更不是 Draft、Verify 全部 Kernel duration 的简单累计。阶段记录与算子统计，需要先对齐区间再比较。

<a id="endsynchronize-draft-verify"></a>

segment 退出只记 End，`end.synchronize()` 到后面的 `_drain_pending()` 才执行。计时没有在 Draft→Verify 边界立刻插入 Device Synchronize。详细观测仍有开销，常规压测和 Profile 结果分别保留。

<a id="gpu"></a>

`step_gpu_ms`、`draft_gpu_ms`、`target_verify_gpu_ms` 可以同时记录，后两项之和却不保证等于整步。固定 Ascend 的 Accept、Finalize、TP Sync、Commit 在这两个 segment 外，还需要在时间线上定位。

## 二、采集稳定的 Decode 窗口

### DSpark Observability 的记录

框架阶段记录用来缩小稳定的慢窗口，设备 trace 再解释其中发生了什么。

可用记录项如下。

~~~text
core
step_cpu_time
step_gpu_time
draft_gpu_time
target_verify_gpu_time
reqs
~~~

Profile 环境的设置示例如下。

~~~bash
export SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time,reqs
~~~

[DecodeStepRecord](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_observability.py) 包含下面这些字段。

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

同一稳定窗口里，把 forward_ct、阶段时间、未归属差值、暴露通信、Replay 间隙、Commit Length、TPOT 并列，可以对齐每轮成本。阶段表例如下面这样记录。

| Step | BS | Verify Tokens | Step GPU | Draft GPU | Verify GPU | Unattributed Step Time | Avg Commit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N | ... | ... | ... | ... | ... | ... | ... |
| N+1 | ... | ... | ... | ... | ... | ... | ... |

未归属差值按下面的式子计算。

~~~text
Unattributed Step Time
=
Step GPU
-
Draft GPU
-
Target Verify GPU
~~~

差值能提示两个计时段以外发生变化，具体工作还得在时间线上找到。

NPU 的 Accept、Commit 位于 Verify 之外，是这里的一部分排查对象。

#### step_cpu_ms 不是 NPU Event 时间

`step_cpu_ms` 用主机 monotonic clock 计算步间隔。

它可能大于 `step_gpu_ms`，单靠这个不等式，还不能判定 Event 或 Profiler 有误。

两种计时之间也可能包含下面这些工作。

~~~text
Host preparation
Scheduler
CPU-side wait
D2H / synchronization
Observability overhead
~~~

### 采集 SGLang Ascend Trace

SGLang Ascend 的采集入口见 [Ascend Performance Profiling Guide](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx)。

环境设置示例如下。

~~~bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/dsv4_dspark_profile
~~~

加载、通信初始化、Graph Capture、Warmup、Prefill 和初始 Decode 完成后，再取稳定部分。

采集请求示例如下。

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

CPU/GPU activity 名称也经 torch_npu 兼容层映射到对应 NPU activity。

#### detailed_annotations 还记录了什么

BS 只有请求数，Verify 的实际工作量还会随下面这些条件变化。

~~~text
verify width / verify_lens
prefix length
ragged layout
graph padding
N_Q
N_KV
~~~

因此两个相同 BS 的 step，也可能执行不同设备工作。

### 进入设备时间线

Draft、Verify 哪段变慢，或两个 segment 以外时间增加，可以先从内部记录看出来。设备时间线则把这些区间与同一轮工作对应，再显示具体等待。

<a id="step-wall-window"></a>
<a id="forward_ct-bs-verify-tokens-dspark-record"></a>

取稳定的 Step N、N+1、N+2，划定各步墙钟区间，用 `forward_ct`、BS、Verify Token 数匹配 DSpark record。验证宽度、历史长度也要对齐，相同 BS 不代表同样负载。

<a id="target-verify-eager-tail"></a>

固定 NPU 的 Accept、Commit 没有 folded 进 Verify Graph。未归属差值增加时，Verify Graph 结束后的区间可能出现 Accept、Sync、Commit 工作，需要逐项对应实际事件。

<a id="replay-host-gap-graphupdate"></a>

Replay 前的 Host gap 则关联 batch、buffer、metadata、bucket 和 launch。DSV4 绕过了 generic `graph.update`，它的准备路径与 generic 分支不同。

<a id="deepep"></a>

通信有发起、in-flight、hook / wait 几个时刻。数据在消费前到齐，暴露等待就可能很少；到了消费点仍未到，才继续阻塞下游。通信任务总时长不能直接代表这一轮被它阻塞多久。

## 三、区分 Verify Graph 与图外工作

### Accept 与 Commit 的位置

共享实现的 `DsparkVerifyEpilogue` 能把接受、Finalize、输出与部分 Commit 注入 Verify 捕获，Ascend 是否使用由 Worker 创建条件决定。[dspark_verify.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_verify.py) [dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

<a id="guard-is_cuda"></a>

~~~python
if (
    (self._verify_planner.is_compact_mode or static_epilogue_supported)
    and self._decode_graph_allowed
    and is_cuda()
):
    self._verify_epilogue = DsparkVerifyEpilogue(...)
~~~

这个条件要求真实 `is_cuda()`，NPU 初始化却已将 `torch.cuda.is_available` 设回 False，所以固定 Ascend 910C 路径不创建 `DsparkVerifyEpilogue`。`fold_eligible` 首先要求 `verify_epilogue is not None`，因此 `fold_eligible`、`folded_accept`、`folded_commit` 都为 False。

<a id="folded_accept-folded_commit-npu-false"></a>

Target Verify 以后仍有下面这段 eager tail。

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

Accept、Finalize、SpecTpSync、Commit 没计入 `target_verify_gpu_ms`。`step_gpu_ms - draft_gpu_ms - target_verify_gpu_ms` 还可能带规划、阶段间隙，多流也影响加法关系，因此先记作未归属时间，不能全部命名为 Commit。

trace 里需要逐个对应 Accept kernels、SpecTpSync、Finalize Accept Lens、BuildOutTokens、KV/hidden commit，以及其他 segment 外设备工作，差值才有了具体边界。

<a id="cuda-folded-path"></a>

CUDA folded path 还要求 proposal folded、Greedy、无额外 logits adjustment、无模拟接受、无 Grammar、Graph 可运行和合适模式，Commit 另须 `epilogue.folds_commit`。它可以用于代码对照，条件并不描述这里的 Ascend 910C 时间线。

### Replay 前的准备工作

NPU Runner 有 TARGET_VERIFY 的 `actual_seq_kvlen` 定义，generic backend 也实现 update，DSV4 却不一定经过这些接口。[npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) 中的模型分派决定实际进入哪个 backend。

<a id="generic-npu-graph-update"></a>

generic 路径将 update 提交到线程，随即等 `update_future.result()`，之后才 replay。采用该分支时，update 仍是 Host 前置依赖，可能形成 pre-replay gap。[npu_cudagraph_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py)

~~~python
update_future = self._update_executor.submit(
    graph.update,
    cpu_update_input=cpu_update_input,
)
update_future.result()
graph.replay()
~~~

<a id="deepseek-v4"></a>

DSA、V4 被排除在这个 generic 分支外。

~~~python
if not (
    is_deepseek_dsa(...)
    or is_deepseek_v4(...)
):
    output = self.backend.replay_with_input_update(...)
else:
    output = self.backend.replay(graph_key, forward_batch)
~~~

V4 直接调 `backend.replay(...)`，不走 `replay_with_input_update(...)`。因此它的 Host 间隙，要由自身准备和等待解释，不能先算到 generic `graph.update` 上。

<a id="dsv4-kv-query-metadata-backend"></a>

实时数据仍要更新。DSV4 backend 按模式和 live batch 刷新 `actual_seq_lengths_q`、`actual_seq_lengths_q_pa`、`actual_seq_lengths_kv`、block tables 与 kernel metadata，Verify 还用 `draft_token_num` 形成 Query 几何。[ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)

这里先刷新 graph-bound buffers / DSV4 metadata，再 `backend.replay(graph_key, forward_batch)`，与 `NPUGraph.update(actual_seq_kvlen)` 后 replay 是不同流程。

<a id="dsv4-replay-host-gap"></a>

Replay 之前还包括调度 batch、Graph bucket / shape key、输入缓冲和 metadata 的准备。

`ScheduleBatch` / `ForwardBatch`、`input_ids`、`positions`、`seq_lens`、`out_cache_loc`，以及 DP、speculative、DSV4 Attention metadata，都要对应到本轮。allocator、Python 和隐含同步也可能在 Replay launch 前形成间隙，实际阻塞位置由时间线确认。

## 四、在时间线上寻找暴露的通信

### Launch、独立计算与 Wait

`async_finish`、`recv_hook` 能把等待推后，间隔里有独立工作，通信才可能被覆盖。接口写着异步，还不足以说明节省了多少设备时间。

<a id="ascend-zbal-dispatcher"></a>

NPU 配置 ZBAL local memory，Dispatcher 按下面的分支选 Buffer。上层仍走 `dispatch_a/b`、`combine_a/b`，替换底层通信资源。[deepep.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)

~~~python
if _use_zbal:
    from zbal.zbal.deepep_adaptor import Config
    from zbal.zbal_buffer import Buffer
else:
    from deep_ep import Buffer, Config
~~~

<a id="deepep-mode-auto-decode-target_verify-low_latency"></a>

`--deepep-mode auto` 经 `DeepEPMode.resolve()`，在 Decode / TARGET_VERIFY 变成 low_latency，在 Prefill / Extend 变成 normal。DSpark Verify 因而要继续跟 low-latency recv-hook。[DeepEPMode.resolve()](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/utils.py)

~~~python
if is_extend_in_batch:
    return DeepEPMode.NORMAL
else:
    return DeepEPMode.LOW_LATENCY
~~~

<a id="normal-modeasync_finish-stream-dependency"></a>

Normal 模式的 `dispatch_a` 捕获 previous_event，`dispatch_b` 取布局、执行 dispatch。`async_finish` 用 `event.current_stream_wait()` 排依赖，Combine 类似。comm stream 可以先排进工作，当前流到消费结果时再等；同批专家计算仍依赖收齐输入。

<a id="low-latency-moderecv_hook"></a>

Low-latency 设置 `async_finish=False`、`return_recv_hook=True`。`dispatch_a` 发起通信、返回 hook，`dispatch_b` 到消费点调 hook，确认 recv data 可用。Combine 也可以把等待推迟到相应消费点。

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

发起到消费之间放得进独立计算，就有重叠机会。传输本身未必变快，变化的是等待落在哪里。

<a id="deepseek-v4-decode-target_verify-tbo"></a>

TBO 利用 A/B 边界，在一组 batch 通信时切到另一组计算。下面只画机制。

~~~text
Batch A dispatch_a
        ↓
Yield
        ↓
Batch B compute
        ↓
Batch A dispatch_b
~~~

固定 V4 策略只支持 EXTEND，注释写着 `DeepseekV4 TBO only supports prefill (EXTEND)`，也记录了 Decode TBO 未实现及相关 regression、Graph capture 问题。这些来自上游注释，本文没有重新执行实验。[operations_strategy.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/batch_overlap/operations_strategy.py)

<a id="side-stream-work"></a>
<a id="shared-expert-routed-expert-overlap"></a>
<a id="low-latency-combine-overlap_args"></a>

Decode / Verify 中还可能由其他路径重叠，例如不依赖当前 recv 的 side-stream 计算、shared 与 routed expert 并行，或上层配置的 low-latency Combine `overlap_args`。它们各有触发与汇合条件，不能与 TBO、recv_hook 混作一个开关。

`T_comm_total` 是通信总时长，`T_comm_exposed` 是仍暴露在关键路径上的等待，后者才直接挡住下游。

### 把各阶段放回同一轮

把这些工作放回同一轮，是下面的关系。

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

DSV4 在 Replay 前实际经过的是 live metadata / bound buffers→replay。

这些边界可以用于 msprof / MindStudio 的阶段标注。

## 五、把耗时与有效输出放在一起

整轮成本可写成下面的关系。

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

再除以有效产出。

~~~text
Effective Progress Rate
=
Committed Tokens
/
Step Wall Time
~~~

对单请求，可以作下面的近似。

~~~text
Effective TPOT
≈
Step Wall Time
/
Average Commit Length
~~~

AL 提高时，周期成本也可能一同增加。

例如扩大 Block，增加了下面这些工作。

~~~text
Draft Cost ↑
Verify Query Tokens ↑
MoE Work ↑
DeepEP Exposed Comm ↑
Commit Cost ↑
~~~

若周期耗时的增长快于有效提交长度的增长，单位输出成本仍会上升。

## 六、设计对照实验并核对结果

### 控制实验变量

| 实验 | 控制变量 | 主要观察 |
| --- | --- | --- |
| Spec OFF vs DSpark ON | workload 完全相同 | DSpark 是否降低 TPOT / 提高 output throughput |
| NPU Graph ON vs OFF | 仅改 Graph | Device bubble、Host launch gap、Step Wall Time |
| DSpark Block Size 3 / 6 / 9 | 仅改 block | Draft GPU、Verify GPU、Avg Commit、Effective TPOT |
| DeepEP low_latency 对照 | 通信配置 | Total Comm、Exposed Comm、hook / wait 位置 |
| Low vs High Concurrency | 模型参数不变 | 瓶颈是否从 Host/Graph 转向 MoE/Communication |

每组保留相同口径的这些数据。

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

运行条件对齐，阶段与执行依赖的变化才有可比性。

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

从现象到解释，可以沿下面的检查路径。

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

对照最后要回答的是，哪段挡住下一次输出，调整后关键路径缩短多少，有效提交长度是否改变。图外 Eager、通信与 Host 准备都要落到同轮的耗时和产出。

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

相关专题可连着阅读。

1. [DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject](../speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)
2. [DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？](deepseek-v4-speculative-performance-analysis.md)
3. [Ascend 910C 投机解码：Graph、多流与通信的执行依赖](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)
