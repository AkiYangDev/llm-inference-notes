# 在 Ascend 910C 上分析 DSpark：从阶段计时到 NPU 时间线

开启 DSpark 后，吞吐变化只是排查入口。要知道下一步该优化哪里，还需要把一轮执行拆开：Draft 和 Verify 的计时覆盖了什么，Accept 与 Commit 在哪里运行，通信和 Host 准备留下了多少未被覆盖的等待。

下面按一次 Profiling 的工作顺序展开：先建立阶段边界，再采集稳定 Decode 窗口，把框架记录与 NPU 时间线对齐，最后用对照实验检验判断。重点是把耗时归到具体依赖上，而不是只按 Kernel 总时长排序。

> **源码基线与适用范围**
>
> 本文基于 sgl-project/sglang @ ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e，核对日期为 2026-09-21。重点讨论 SGLang SRT 文本推理路径下的 DeepSeek-V4 + DSpark + Ascend 910C。本文所有“当前实现”结论都固定到该 commit；Profiler 截图与具体毫秒数必须在目标机器上实测，本文不会把源码推导伪装成硬件实测结果。

## 一、把一轮 DSpark 对应到计时字段

### 一轮执行的起止

当前 [DSparkWorkerV2](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) 的 Decode 主线可以抽象成：

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

源码里已经显式插入两个阶段计时边界：

~~~python
with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
    proposal = self._proposer.propose(...)

with self._observers.segment(InfoSegment.TARGET_VERIFY):
    target_verify = ...
~~~

因此 SGLang 自己已经给出了第一层性能模型：

~~~text
DSpark Step
├── Draft segment
├── Target Verify segment
└── Segment 外部的 Accept / Commit / 其他工作
~~~

最直觉的公式是：`T_step = T_draft + T_verify + T_accept + T_commit`

这四类描述逻辑阶段。设备异步执行时，多条 Stream 上的计算与通信可以重叠；Event elapsed time 测量起止点之间的设备时间，各 Stream 的 Kernel Duration 则分别累计工作时长。

要区分：

~~~text
Kernel Duration Sum
        ≠
Stage Elapsed Time
        ≠
End-to-End Step Wall Time
~~~

### Event 时间覆盖什么

当前 DSpark 阶段计时代码位于：

[dspark_observability.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_observability.py)

它定义了 STEP、DRAFT、TARGET_VERIFY 三个 segment，并用 Event 记录设备时间：

~~~python
start = torch.cuda.Event(enable_timing=True)
start.record()

...

end = torch.cuda.Event(enable_timing=True)
end.record()
~~~

读取时：

~~~python
end.synchronize()
elapsed_ms = start.elapsed_time(end)
~~~

问题是：这是 Ascend 910C，为什么仍然写 torch.cuda.Event？

#### NPU 文本路径会加载 transfer_to_npu

SGLang NPU 初始化在：

[npu/utils.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/utils.py)

标准 SRT 文本路径会加载：

~~~python
from torch_npu.contrib import transfer_to_npu
~~~

随后又显式执行：

~~~python
torch.cuda.is_available = lambda: False
~~~

避免兼容层把 CUDA capability 判断错误地变成 True。

Ascend PyTorch 的 [transfer_to_npu.py](https://github.com/Ascend/pytorch/blob/77037e26c8b39b4d5e1856bb23d601b67f7ee5a9/torch_npu/contrib/transfer_to_npu.py) 会把 CUDA 风格接口映射到 NPU 接口，其中包括 torch.cuda → torch_npu.npu 的兼容映射。

因此在本文讨论的标准 SGLang NPU 文本路径中，DSpark observability 写出的 torch.cuda.Event，运行时语义实际上是 NPU Event。

#### Event elapsed 不是所有 Kernel duration 的求和

两个 Event 都在调用 record() 时记录到当前执行流。

所以 draft_gpu_ms 更准确的含义是：

> **Draft segment 入口 Event 与出口 Event 在当前执行流上的设备 elapsed time。**

Target Verify 同理。

可以画成：

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

如果 side stream 工作在 Draft End Event 前通过 event / stream dependency 被重新 join，它会影响阶段 elapsed。

side stream 的工作若跨过 Draft End，完整 Kernel Duration 就跨越了 `draft_gpu_ms` 的计时范围。对齐时按 Event 起止点截取对应区间。

所以 draft_gpu_ms / target_verify_gpu_ms 不是：

1. Python wall time；
2. 所有 NPU Stream 的 Busy Time 总和；
3. 所有 Draft / Verify Kernel Duration 的加总。

更合适的称呼是：

> **源码阶段定义下的 current-stream device elapsed anchor。**

#### end.synchronize() 不在 Draft → Verify 边界立即发生

另一个必须核清的问题是 end.synchronize() 会不会强制 Draft 与 Verify 串行。

当前实现中，segment 结束只 record End Event。 end.synchronize() 在后续读取 record 的 _drain_pending() 过程中发生。

因此：

> **计时 Event 本身不会在 Draft → Verify 边界立即插入一次 Device Synchronize。**

不过开启详细 observability 仍然有测量成本，所以正式 benchmark 应区分“常规运行结果”和“详细 Profile 结果”。

#### 如何正确使用三个 GPU 时间

当前最有用的是同时记录：

~~~text
step_gpu_ms
draft_gpu_ms
target_verify_gpu_ms
~~~

不要默认：

~~~text
step_gpu_ms
=
draft_gpu_ms
+
target_verify_gpu_ms
~~~

因为当前 Ascend 还有一个明确位于两个 segment 之外的尾部：`Accept / Finalize / TP Sync / Commit`

这正是下一节的核心。

## 二、采集稳定的 Decode 窗口

### 先看 DSpark Observability

先用阶段记录找到稳定的慢窗口，再针对该窗口采集 msprof，定位会更集中。

先利用 SGLang 自带的 DSpark 分阶段记录把问题缩小。

当前可记录：

~~~text
core
step_cpu_time
step_gpu_time
draft_gpu_time
target_verify_gpu_time
reqs
~~~

Profile 实验环境可以启用：

~~~bash
export SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time,reqs
~~~

[DecodeStepRecord](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_observability.py) 会提供：

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

第一张性能表应该长这样：

| Step | BS | Verify Tokens | Step GPU | Draft GPU | Verify GPU | Unattributed Step Time | Avg Commit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N | ... | ... | ... | ... | ... | ... | ... |
| N+1 | ... | ... | ... | ... | ... | ... | ... |

其中：

~~~text
Unattributed Step Time
=
Step GPU
-
Draft GPU
-
Target Verify GPU
~~~

它用于发现阶段外时间的变化，具体归因再交给时间线。

在当前 Ascend 路径中尤其有价值，因为我们已经确认 Accept / Commit 位于 Target Verify segment 之外。

#### step_cpu_ms 不是 NPU Event 时间

step_cpu_ms 来自 Host monotonic clock 的 step 间隔。

因此：`step_cpu_ms > step_gpu_ms`

并不自动说明 NPU profiler 错了。

中间可能包含：

~~~text
Host preparation
Scheduler
CPU-side wait
D2H / synchronization
Observability overhead
~~~

### 采集 SGLang Ascend Trace

SGLang 当前 Ascend Profiling 入口：

[Ascend Performance Profiling Guide](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx)

先设置：

~~~bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/dsv4_dspark_profile
~~~

让服务完成模型加载、分布式初始化、Graph Capture、Warmup、Prefill 和最初几轮 Decode，再抓稳定窗口。

例如：

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

Ascend 环境仍使用 CPU / GPU 这个 Profiler Activity 表达，是因为 torch_npu compatibility patch 会把对应 CUDA profiler activity 映射到 NPU activity。

#### 为什么 detailed_annotations 很重要

Target Verify 的 workload 同时取决于 BS、Verify 宽度、历史长度与 Graph Padding。

Speculative Verify 还取决于：

~~~text
verify width / verify_lens
prefix length
ragged layout
graph padding
N_Q
N_KV
~~~

所以两个 BS 相同的 Verify Step，Device Work 可能完全不同。

### 进入设备时间线

SGLang internal record 回答：

~~~text
Draft 慢？
Verify 慢？
还是 segment 外尾部变大？
~~~

接下来才值得进入 CANN / NPU Timeline。

建议顺序如下。

#### 先定位 Step Wall Window

先找稳定的 Step N / N+1 / N+2，不要一上来做 Operator Ranking。

#### 用 forward_ct / BS / verify tokens 对齐 DSpark record

这样 Timeline 中的某一个异常 Step 才有 workload context。

#### 重点看 Target Verify 后面的 eager tail

当前 NPU Accept / Commit 不 folded，所以：

~~~text
Verify Graph End
        ↓
Accept / Sync / Commit tail
~~~

本身就是重要分析对象。

如果 Unattributed Step Time 很大，先在时间线上确认多出的时间是否确实位于这一段，再检查 Accept / Commit。

#### Replay 前有 Host gap时，不要先找 graph.update

DSV4 当前绕开 generic graph.update 分支。

先检查 batch、buffer、metadata、shape/bucket 和 replay launch。

#### DeepEP 看“等了多久”，不只看“通信跑了多久”

应该标：

~~~text
communication issue
        ↓
in-flight interval
        ↓
hook / wait
~~~

如果到 hook / wait 时通信已经基本完成，exposed comm 很小。

如果 hook 后还需要长时间等待，通信就真实暴露在 Critical Path 上。

## 三、区分 Verify Graph 与图外工作

### Accept 与 Commit 的位置

这一点决定了后处理应怎样计时。

跨平台 DSpark 源码里确实存在 DsparkVerifyEpilogue，它可以在支持路径上把 Accept、Finalize、Out Token，甚至某些 Commit 注入动作捕获进 Target Verify Graph。

入口在：

[dspark_verify.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_verify.py)

在 Ascend 910C 路径上，继续检查实例创建条件：

> **NPU 上它会不会被创建？**

#### 创建 guard 明确要求 is_cuda()

回到：

[dspark_worker_v2.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

当前逻辑是：

~~~python
if (
    (self._verify_planner.is_compact_mode or static_epilogue_supported)
    and self._decode_graph_allowed
    and is_cuda()
):
    self._verify_epilogue = DsparkVerifyEpilogue(...)
~~~

最后一个条件明确是 is_cuda()，不是 is_cuda_alike()，也不是 is_cuda() or is_npu()。

SGLang 的 is_cuda() 要求真实 CUDA availability；Ascend NPU 初始化又重新把 torch.cuda.is_available() 设为 False。

因此当前 commit 下：

> **Ascend 910C 不会构造 DsparkVerifyEpilogue。**

#### 于是 folded_accept / folded_commit 在 NPU 上都是 False

Decode 里 fold_eligible 的第一个条件就是：`verify_epilogue is not None`

NPU 上不满足，所以：

~~~text
fold_eligible = False
folded_accept = False
folded_commit = False
~~~

于是当前 Ascend 路径实际是：

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

这对 Profiling 的影响非常直接：

> **Accept / Finalize / SpecTpSync / Commit 不属于 target_verify_gpu_ms。**

因此可以构造一个很有用的诊断 proxy：

~~~text
Unattributed Step Time
≈
step_gpu_ms
-
draft_gpu_ms
-
target_verify_gpu_ms
~~~

这个差值包含 Draft、Verify 计时区间之外的工作：可能是前置规划、阶段间空隙，也可能是 Accept / Commit。多流重叠还会影响可加性，所以这个差值用于标记未归属时间，再由时间线确定工作发生在哪个阶段。

差值增大后，应在时间线上核对前置规划、阶段间空隙，以及以下后处理：

~~~text
Accept kernels
SpecTpSync
Finalize Accept Lens
BuildOutTokens
KV / hidden commit
其他 segment 外 device work
~~~

而不是继续把所有时间归到 Target Verify Forward。

#### CUDA folded path只能作为对照

在 CUDA 支持路径上，folded Accept 还要求 proposal folded、greedy sampling、没有额外 logits adjustments、没有 simulate_acc_len、没有 grammar、Graph 可运行，并且 verify mode 满足 compact / static 条件。

folded Commit 还需要 epilogue.folds_commit 成立。

这些条件对理解共享源码有价值，但在本文中必须明确标为：

> **跨平台对照，不是当前 910C Timeline 事实。**

### Replay 前的准备工作

当前：

[npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py)

确实定义了 TARGET_VERIFY 对应的 Host-side 属性：`actual_seq_kvlen`

而 generic NPU Graph backend 也确实支持 NPUGraph.update。

所以只看局部代码很容易得出：

> 每次 Verify Replay 前都执行 graph.update(actual_seq_kvlen)，可能产生 Host gap。

对 DeepSeek-V4 来说，这个结论是错的。

#### Generic NPU Graph 路径确实会同步等待 update

先看：

[npu_cudagraph_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py)

generic replay_with_input_update() 当前做：

~~~python
update_future = self._update_executor.submit(
    graph.update,
    cpu_update_input=cpu_update_input,
)
update_future.result()
graph.replay()
~~~

虽然 update 放进了专门 worker thread，但紧接着 update_future.result() 会等待它完成。

所以对使用这个分支的模型：

~~~text
Host:
submit graph.update
       ↓
wait update_future.result()
       ↓
graph.replay
~~~

NPUGraph.update 是 Replay 的真实 Host-side prerequisite，理论上可以形成可见的 pre-replay gap。

#### DeepSeek-V4 被明确排除

但 NPUGraphRunner.execute() 最后的实际分支是：

~~~python
if not (
    is_deepseek_dsa(...)
    or is_deepseek_v4(...)
):
    output = self.backend.replay_with_input_update(...)
else:
    output = self.backend.replay(graph_key, forward_batch)
~~~

所以 DeepSeek-V4 当前直接走：`backend.replay(...)`

而不是：`replay_with_input_update(...)`

因此对本文主角：

> **DeepSeek-V4 + DSpark + Ascend 910C**

当前路径跳过 `graph.update(actual_seq_kvlen)`；Replay 前的 Host gap 应继续检查下文的 metadata、buffer refresh 与 Scheduler。

#### DSV4 的动态 KV / Query metadata走自己的 backend

这并不意味着 DSV4 Graph 使用固定 capture-time 长度。

当前：

[ascend_dsv4_backend.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)

会根据 ForwardMode 和 live batch 构造 / 刷新：

~~~text
actual_seq_lengths_q
actual_seq_lengths_q_pa
actual_seq_lengths_kv
block tables
kernel metadata
~~~

Target Verify 还会结合 draft_token_num 构造 Query geometry。

所以 DeepSeek-V4 的思路更接近：

~~~text
refresh graph-bound buffers / DSV4 metadata
        ↓
backend.replay(graph_key, forward_batch)
~~~

而不是 generic：

~~~text
NPUGraph.update(actual_seq_kvlen)
        ↓
replay
~~~

#### 如果 DSV4 Replay 前有 Host gap，应该查什么

建议按顺序看：

~~~text
1. ScheduleBatch / ForwardBatch preparation
2. graph bucket / shape key selection
3. input_ids / positions / seq_lens / out_cache_loc refresh
4. DP / speculative metadata preparation
5. DSV4 attention metadata
6. replay launch
7. allocator / Python / hidden sync point
~~~

不要先问：`是不是 graph.update 太慢？`

更准确的文章结论应该是：

> **NPUGraph.update 是 generic NPU Target Verify 的潜在 Host 前置开销，但不是当前 DeepSeek-V4 DSV4 Target Verify 的 Replay 前置开销。**

## 四、在时间线上寻找暴露的通信

### Launch、独立计算与 Wait

通信 overlap 是最容易把“API 有 async”误写成“性能已经 overlap”的地方。

先说结论：

> **async_finish=True 或 return_recv_hook=True 只说明 Runtime 把“发起通信”和“等待结果可消费”解耦了。是否形成性能收益，取决于两者之间有没有独立的有用工作。**

#### Ascend + ZBAL 仍复用同一 Dispatcher 状态机

当前：

[deepep.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)

在 NPU 且开启 ZBAL local memory 时：

~~~python
if _use_zbal:
    from zbal.zbal.deepep_adaptor import Config
    from zbal.zbal_buffer import Buffer
else:
    from deep_ep import Buffer, Config
~~~

也就是说 SGLang 上层仍然使用 DeepEP Dispatcher 的 dispatch_a / dispatch_b / combine_a / combine_b 状态机，只是底层 Buffer 实现切换成 ZBAL adapter。

因此必须分清：

~~~text
SGLang dispatcher semantics
        ↓
DeepEP Buffer or ZBAL Buffer
        ↓
Ascend communication implementation
~~~

#### deepep-mode auto 在 Decode / TARGET_VERIFY 解析成 low_latency

当前：

[DeepEPMode.resolve()](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/utils.py)

非常明确：

~~~python
if is_extend_in_batch:
    return DeepEPMode.NORMAL
else:
    return DeepEPMode.LOW_LATENCY
~~~

因此常见配置：`--deepep-mode auto`

在 Decode / Target Verify 中走 low-latency，在 Prefill / Extend 中走 normal。

这意味着分析 DSpark Verify 时，更应该理解 low-latency recv-hook 语义。

#### Normal Mode：async_finish主要把等待变成 Stream dependency

Normal path 中，dispatch_a 首先捕获 previous_event；dispatch_b 才执行 get_dispatch_layout / buffer.dispatch，并在 async_finish 下调用：`event.current_stream_wait()`

Combine 也类似。

所以 async_finish=True 的真实意义更接近：

~~~text
communication queued on comm stream
        ↓
current stream only waits when result is needed
~~~

它不是：`Dispatch 与自己的 Expert GEMM 无依赖并发`

同一批 Token 的 Expert GEMM 必须消费 Dispatch 后的 recv data，这个数据依赖无法凭 async flag 消失。

#### Low-Latency Mode：recv_hook创造“发起 → 消费”的窗口

low-latency path 中当前 SGLang 传入 return_recv_hook=True。

于是 low_latency_dispatch 使用：

~~~text
async_finish = False
return_recv_hook = True
~~~

dispatch_a 可以先发起通信并返回 hook；dispatch_b 再调用 hook()，确保数据已经到达并可消费。

概念上：

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

Combine 同理。

所以 recv_hook 的核心不是“通信更快”，而是：

> **把同步点向消费数据的位置推迟。**

#### 但当前 DeepSeek-V4 Decode / TARGET_VERIFY 没有 TBO

Two-Batch Overlap 最适合利用这种 A/B 分段：

~~~text
Batch A dispatch_a
        ↓
Yield
        ↓
Batch B compute
        ↓
Batch A dispatch_b
~~~

然而：

[operations_strategy.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/batch_overlap/operations_strategy.py)

对 DeepseekV4DecoderLayer 明确写着：`DeepseekV4 TBO only supports prefill (EXTEND)`

注释还指出 Decode TBO 当前未实现，已有数据表明会 regression，需要进一步解决 graph capture。

当前 DeepSeek-V4 的 TBO 支持 EXTEND。DSpark 的 `DECODE / TARGET_VERIFY` 要从其他独立工作中寻找通信重叠：

##### 独立 Side Stream Work

通信在独立 Stream 飞行期间，如果其他 Stream 已经有与 recv data 无依赖的工作，可以形成真实 overlap。

##### Shared Expert / Routed Expert overlap

DeepSeek MoE 还有 shared expert side-stream 等独立优化。这可能覆盖部分通信，但它和 TBO / recv_hook 不是同一个机制。

##### Low-latency Combine overlap_args

Combine 路径支持额外 overlap stream 协调；只有上层实际配置并触发时才有意义。

因此最终一定要区分：`T_comm_total`

和：`T_comm_exposed`

影响 Step Wall Time 的是 exposed communication，而不是通信 Task 自己的累计 duration。

### 把各阶段放回同一轮

当前更合理的时间线是：

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

同时要记住：

> Target Verify Graph 对 DSV4 不是 graph.update(actual_seq_kvlen) → replay，而是 DSV4 live metadata / bound buffers → replay。

这张图才适合作为后面 msprof / MindStudio 的阶段标注模板。

## 五、把耗时与有效输出放在一起

结合阶段边界，得到下面的模型：

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

最终关注：

~~~text
Effective Progress Rate
=
Committed Tokens
/
Step Wall Time
~~~

单请求近似可以写成：

~~~text
Effective TPOT
≈
Step Wall Time
/
Average Commit Length
~~~

因此 Acceptance Length 上升并不保证 TPOT 下降。

如果更大的 Draft Block 同时带来：

~~~text
Draft Cost ↑
Verify Query Tokens ↑
MoE Work ↑
DeepEP Exposed Comm ↑
Commit Cost ↑
~~~

增长超过有效 Commit Token 的收益，系统仍然会变慢。

## 六、设计对照实验并核对结果

### 控制实验变量

| 实验 | 控制变量 | 主要观察 |
| --- | --- | --- |
| Spec OFF vs DSpark ON | workload 完全相同 | DSpark 是否降低 TPOT / 提高 output throughput |
| NPU Graph ON vs OFF | 仅改 Graph | Device bubble、Host launch gap、Step Wall Time |
| DSpark Block Size 3 / 6 / 9 | 仅改 block | Draft GPU、Verify GPU、Avg Commit、Effective TPOT |
| DeepEP low_latency 对照 | 通信配置 | Total Comm、Exposed Comm、hook / wait 位置 |
| Low vs High Concurrency | 模型参数不变 | 瓶颈是否从 Host/Graph 转向 MoE/Communication |

每组实验至少保留：

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

保持这些条件一致后，再比较 Trace 中的阶段差异。

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

最后把整套方法收束成：

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

归因完成后，用同一轮记录检查结论：哪段工作延迟了下一次有效输出，调整后关键路径缩短多少，有效提交长度是否变化。Draft / Verify 以外的 Eager 尾部、通信等待和 Replay 前 Host 准备，都应落实到这组时间与产出上。

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

建议按下面顺序阅读：

1. [DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject](../speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)
2. [DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？](deepseek-v4-speculative-performance-analysis.md)
3. [Ascend 910C 投机解码：Graph、多流与通信的执行依赖](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)
4. **本文：把源码机制映射到真实 Profiling 时间线。**

采集稳定窗口后，把 `forward_ct`、Draft / Verify 时间、阶段外时间、暴露的通信等待、Replay 前间隙、Commit Length 与 TPOT 放在同一张表中，即可比较改动前后每轮的成本与产出。
