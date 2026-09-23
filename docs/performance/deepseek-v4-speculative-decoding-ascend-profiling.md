# DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上怎么 Profile？从 SGLang Draft/Verify 到 NPU Kernel 时间线

如果只看 benchmark，Speculative Decoding 的性能问题似乎很简单：

~~~text
Spec OFF    → 100 tokens/s
DSpark ON   → 135 tokens/s

结论：快了 35%
~~~

但对推理基础设施工程师，这个结论远远不够。真正需要回答的是：这 35% 到底从哪里来？

一次 DSpark Decode Step 中，Draft 花了多少时间？Target Verify 花了多少？Accept Decision 和 KV / hidden-state Commit 是 Verify Graph 的一部分，还是 Verify 结束之后的独立工作？DeepEP 的通信是在 Critical Path 上完整暴露，还是被其他计算覆盖？NPU Graph Replay 前有没有 Host-side gap？如果有，这个 gap 来自 graph.update，还是来自 batch / metadata 准备？

这篇文章把问题从“整体快不快”继续下钻到：

~~~text
SGLang Python Stage
        ↓
ForwardMode / Runtime Boundary
        ↓
NPU Graph / Eager
        ↓
Stream / Communication
        ↓
CANN Operator / NPU Kernel
        ↓
Ascend 910C Timeline
~~~

目标不是写一份 msprof 参数说明书，而是建立一套可以反复使用的性能归因方法：

> **先用源码确定阶段边界，再用 Profiler 判断这些阶段在真实 Device Timeline 中怎样排列、重叠和等待。**

> **源码基线与适用范围**
>
> 本文基于 sgl-project/sglang @ ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e，核对日期为 2026-09-21。重点讨论 SGLang SRT 文本推理路径下的 DeepSeek-V4 + DSpark + Ascend 910C。本文所有“当前实现”结论都固定到该 commit；Profiler 截图与具体毫秒数必须在目标机器上实测，本文不会把源码推导伪装成硬件实测结果。

---

## Strict Review 结论先行

这篇文章最容易被写错的四个地方已经逐项核过。

| Review 点 | 当前源码结论 | 对 Profiling 的影响 |
| --- | --- | --- |
| draft_gpu_ms / target_verify_gpu_ms 在 Ascend 上到底测什么 | SGLang NPU 文本路径会加载 torch_npu.contrib.transfer_to_npu，因此 DSpark 里的 torch.cuda.Event 实际映射为 NPU Event。它测的是**当前执行流两个 Event 之间的设备 elapsed time**，不是全设备所有 Stream 的 Kernel Duration 求和。 | 可以用它做 Draft / Verify 阶段级时间锚点，但不能把它当“总 NPU Busy Time”。Side Stream 若未在结束 Event 前 join，可能不完整计入。 |
| folded Accept / Commit 是否会进入 Ascend NPU Graph | **当前不会。** DsparkVerifyEpilogue 的创建条件明确包含 is_cuda()；Ascend NPU 初始化又把 torch.cuda.is_available() 重新设为 False，所以 910C 上 epilogue 不创建，folded_accept=False、folded_commit=False。 | 在当前 910C DSpark 路径中，Accept / Finalize / TP Sync / Commit 应当视为 **Target Verify 之后的 eager Device Work**，而不是 Verify Graph 内部工作。 |
| TARGET_VERIFY Replay 前是否会执行 graph.update(actual_seq_kvlen) | **DeepSeek-V4 当前不会走 generic update 分支。** NPUGraphRunner.execute() 对 DeepSeek-V4 / DSA 明确走 backend.replay(graph_key, forward_batch)，绕过 replay_with_input_update()。 | 如果 DSV4 Verify Graph 前出现 Host gap，不应直接归因于 NPUGraph.update。应先查 ForwardBatch / buffer copy、attention metadata、bucket / layout preparation 和 replay launch。 |
| DeepEP async_finish / recv_hook 能隐藏什么 | deepep-mode=auto 在 Decode / Target Verify 会解析成 low-latency；low-latency 的 return_recv_hook 把“发起通信”和“等待数据真正可消费”拆开。但 DeepSeek-V4 当前 Decode / TARGET_VERIFY 的 TBO 尚未实现。 | API 提供的是**创造 overlap window 的机制**，不是“通信已经被隐藏”的证据。当前 DSpark Verify 不能用“V4 Decode TBO”解释通信隐藏；最终仍要看 Timeline 上 Dispatch / Expert Compute / Combine 是否实际重叠。 |

经过 Review 后，当前 Ascend 910C 的 DSpark 主线应该先画成：

~~~text
Draft
  ↓
Target Verify
  ↓
Eager Accept / Finalize / TP Sync
  ↓
Eager Commit
  ↓
Next Draft State
~~~

而不是把 Accept / Commit 默认塞进 Target Verify Graph。

---

## 一、先定义清楚：一次 DSpark Step 的源码边界

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

最直觉的公式是：

~~~text
T_step = T_draft + T_verify + T_accept + T_commit
~~~

但它只能作为逻辑分类，不能直接拿四类 Kernel Duration 机械相加。原因是 Device 异步执行、多 Stream 可能并发、Communication 可能在独立 Stream 飞行，而且 Event elapsed time 测的是阶段边界之间的设备 elapsed，而不是所有 Stream duration sum。

真正要区分：

~~~text
Kernel Duration Sum
        ≠
Stage Elapsed Time
        ≠
End-to-End Step Wall Time
~~~

---

## 二、Strict Review ①：draft_gpu_ms / target_verify_gpu_ms 在 Ascend 上的真实计时语义

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

### 2.1 NPU 文本路径会加载 transfer_to_npu

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

### 2.2 Event elapsed 不是所有 Kernel duration 的求和

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

如果某个 side stream 工作在 Draft End 之后仍独立飞行，那么不能把那条 stream 的完整 Kernel Duration 都算作 draft_gpu_ms。

所以 draft_gpu_ms / target_verify_gpu_ms 不是：

1. Python wall time；
2. 所有 NPU Stream 的 Busy Time 总和；
3. 所有 Draft / Verify Kernel Duration 的加总。

更合适的称呼是：

> **源码阶段定义下的 current-stream device elapsed anchor。**

### 2.3 end.synchronize() 不在 Draft → Verify 边界立即发生

另一个必须核清的问题是 end.synchronize() 会不会强制 Draft 与 Verify 串行。

当前实现中，segment 结束只 record End Event。真正的 end.synchronize() 在后续读取 record 的 _drain_pending() 过程中发生。

因此：

> **计时 Event 本身不会在 Draft → Verify 边界立即插入一次 Device Synchronize。**

不过开启详细 observability 仍然有测量成本，所以正式 benchmark 应区分“常规运行结果”和“详细 Profile 结果”。

### 2.4 如何正确使用三个 GPU 时间

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

因为当前 Ascend 还有一个明确位于两个 segment 之外的尾部：

~~~text
Accept / Finalize / TP Sync / Commit
~~~

这正是下一节的核心。

---

## 三、Strict Review ②：当前 910C 不会把 Accept / Commit folded 进 Target Verify Graph

这是本轮 Review 中最重要的修正。

跨平台 DSpark 源码里确实存在 DsparkVerifyEpilogue，它可以在支持路径上把 Accept、Finalize、Out Token，甚至某些 Commit 注入动作捕获进 Target Verify Graph。

入口在：

[dspark_verify.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/speculative/dspark_components/dspark_verify.py)

但文章讨论的是 Ascend 910C，因此不能停在“代码里存在这个类”。

真正要问：

> **NPU 上它会不会被创建？**

### 3.1 创建 guard 明确要求 is_cuda()

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

### 3.2 于是 folded_accept / folded_commit 在 NPU 上都是 False

Decode 里 fold_eligible 的第一个条件就是：

~~~text
verify_epilogue is not None
~~~

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
Post-Verify Residual
≈
step_gpu_ms
-
draft_gpu_ms
-
target_verify_gpu_ms
~~~

它不是严格的物理可加分解，因为仍可能有 multi-stream overlap，但在当前 NPU 路径里很适合发现 Verify 之后的尾部是否异常变大。

当 residual 突然增大时，优先看：

~~~text
Accept kernels
SpecTpSync
Finalize Accept Lens
BuildOutTokens
KV / hidden commit
其他 segment 外 device work
~~~

而不是继续把所有时间归到 Target Verify Forward。

### 3.3 CUDA folded path只能作为对照

在 CUDA 支持路径上，folded Accept 还要求 proposal folded、greedy sampling、没有额外 logits adjustments、没有 simulate_acc_len、没有 grammar、Graph 可运行，并且 verify mode 满足 compact / static 条件。

folded Commit 还需要 epilogue.folds_commit 成立。

这些条件对理解共享源码有价值，但在本文中必须明确标为：

> **跨平台对照，不是当前 910C Timeline 事实。**

---

## 四、Strict Review ③：DeepSeek-V4 TARGET_VERIFY 当前绕开 graph.update(actual_seq_kvlen)

当前：

[npu_graph_runner.py](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py)

确实定义了 TARGET_VERIFY 对应的 Host-side 属性：

~~~text
actual_seq_kvlen
~~~

而 generic NPU Graph backend 也确实支持 NPUGraph.update。

所以只看局部代码很容易得出：

> 每次 Verify Replay 前都执行 graph.update(actual_seq_kvlen)，可能产生 Host gap。

对 DeepSeek-V4 来说，这个结论是错的。

### 4.1 Generic NPU Graph 路径确实会同步等待 update

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

### 4.2 DeepSeek-V4 被明确排除

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

所以 DeepSeek-V4 当前直接走：

~~~text
backend.replay(...)
~~~

而不是：

~~~text
replay_with_input_update(...)
~~~

因此对本文主角：

> **DeepSeek-V4 + DSpark + Ascend 910C**

不能把 Replay 前的 Host gap 归因于 graph.update(actual_seq_kvlen)，因为当前路径根本没有执行它。

### 4.3 DSV4 的动态 KV / Query metadata走自己的 backend

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

### 4.4 如果 DSV4 Replay 前有 Host gap，应该查什么

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

不要先问：

~~~text
是不是 graph.update 太慢？
~~~

更准确的文章结论应该是：

> **NPUGraph.update 是 generic NPU Target Verify 的潜在 Host 前置开销，但不是当前 DeepSeek-V4 DSV4 Target Verify 的 Replay 前置开销。**

---

## 五、Strict Review ④：DeepEP async_finish / recv_hook 到底能隐藏哪一段通信

通信 overlap 是最容易把“API 有 async”误写成“性能已经 overlap”的地方。

先说结论：

> **async_finish=True 或 return_recv_hook=True 只说明 Runtime 把“发起通信”和“等待结果可消费”解耦了。是否形成性能收益，取决于两者之间有没有独立的有用工作。**

### 5.1 Ascend + ZBAL 仍复用同一 Dispatcher 状态机

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

### 5.2 deepep-mode auto 在 Decode / TARGET_VERIFY 解析成 low_latency

当前：

[DeepEPMode.resolve()](https://github.com/sgl-project/sglang/blob/ab03a8e7eb82d35907bfbeb645bf0af8b4ce290e/python/sglang/srt/layers/moe/utils.py)

非常明确：

~~~python
if is_extend_in_batch:
    return DeepEPMode.NORMAL
else:
    return DeepEPMode.LOW_LATENCY
~~~

因此常见配置：

~~~text
--deepep-mode auto
~~~

在 Decode / Target Verify 中走 low-latency，在 Prefill / Extend 中走 normal。

这意味着分析 DSpark Verify 时，更应该理解 low-latency recv-hook 语义。

### 5.3 Normal Mode：async_finish主要把等待变成 Stream dependency

Normal path 中，dispatch_a 首先捕获 previous_event；dispatch_b 才真正执行 get_dispatch_layout / buffer.dispatch，并在 async_finish 下调用：

~~~text
event.current_stream_wait()
~~~

Combine 也类似。

所以 async_finish=True 的真实意义更接近：

~~~text
communication queued on comm stream
        ↓
current stream only waits when result is needed
~~~

它不是：

~~~text
Dispatch 与自己的 Expert GEMM 无依赖并发
~~~

同一批 Token 的 Expert GEMM 必须消费 Dispatch 后的 recv data，这个数据依赖无法凭 async flag 消失。

### 5.4 Low-Latency Mode：recv_hook真正创造“发起 → 消费”的窗口

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

> **把同步点向真正消费数据的位置推迟。**

### 5.5 但当前 DeepSeek-V4 Decode / TARGET_VERIFY 没有 TBO

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

对 DeepseekV4DecoderLayer 明确写着：

~~~text
DeepseekV4 TBO only supports prefill (EXTEND)
~~~

注释还指出 Decode TBO 当前未实现，已有数据表明会 regression，需要进一步解决 graph capture。

因此对当前 DSpark：

~~~text
DECODE / TARGET_VERIFY
~~~

不能写成：

> recv_hook 已经借助 V4 TBO 把通信隐藏在另一个 micro-batch 后面。

这条实现不存在。

### 5.6 当前 910C 仍可能出现哪些真实 overlap

仍然有几类，但必须由 Timeline 证明。

#### 独立 Side Stream Work

通信在独立 Stream 飞行期间，如果其他 Stream 已经有与 recv data 无依赖的工作，可以形成真实 overlap。

#### Shared Expert / Routed Expert overlap

DeepSeek MoE 还有 shared expert side-stream 等独立优化。这可能覆盖部分通信，但它和 TBO / recv_hook 不是同一个机制。

#### Low-latency Combine overlap_args

Combine 路径支持额外 overlap stream 协调；只有上层实际配置并触发时才有意义。

因此最终一定要区分：

~~~text
T_comm_total
~~~

和：

~~~text
T_comm_exposed
~~~

真正影响 Step Wall Time 的是 exposed communication，而不是通信 Task 自己的累计 duration。

---

## 六、四点 Review 后，910C 的 DSpark Timeline 应该重新画成什么样

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

---

## 七、第一层采集：先用 DSpark 自带 Observability

真正做实验时，不建议第一步就录几十秒 msprof。

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

| Step | BS | Verify Tokens | Step GPU | Draft GPU | Verify GPU | Post-Verify Proxy | Avg Commit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N | ... | ... | ... | ... | ... | ... | ... |
| N+1 | ... | ... | ... | ... | ... | ... | ... |

其中：

~~~text
Post-Verify Proxy
=
Step GPU
-
Draft GPU
-
Target Verify GPU
~~~

它不是严格的可加物理分解，而是一个异常检测 proxy。

在当前 Ascend 路径中尤其有价值，因为我们已经确认 Accept / Commit 位于 Target Verify segment 之外。

### step_cpu_ms 不是 NPU Event 时间

step_cpu_ms 来自 Host monotonic clock 的 step 间隔。

因此：

~~~text
step_cpu_ms > step_gpu_ms
~~~

并不自动说明 NPU profiler 错了。

中间可能包含：

~~~text
Host preparation
Scheduler
CPU-side wait
D2H / synchronization
Observability overhead
~~~

---

## 八、第二层采集：SGLang Ascend Profiler 抓稳定 Decode Window

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

### 为什么 detailed_annotations 很重要

Target Verify 的 workload 不能只用 BS 表示。

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

---

## 九、第三层采集：msprof / MindStudio 看真正 Critical Path

SGLang internal record 回答：

~~~text
Draft 慢？
Verify 慢？
还是 segment 外尾部变大？
~~~

接下来才值得进入 CANN / NPU Timeline。

建议顺序如下。

### 9.1 先定位 Step Wall Window

先找稳定的 Step N / N+1 / N+2，不要一上来做 Operator Ranking。

### 9.2 用 forward_ct / BS / verify tokens 对齐 DSpark record

这样 Timeline 中的某一个异常 Step 才有 workload context。

### 9.3 重点看 Target Verify 后面的 eager tail

当前 NPU Accept / Commit 不 folded，所以：

~~~text
Verify Graph End
        ↓
Accept / Sync / Commit tail
~~~

本身就是重要分析对象。

如果 Post-Verify Proxy 很大，就优先从这里找。

### 9.4 Replay 前有 Host gap时，不要先找 graph.update

DSV4 当前绕开 generic graph.update 分支。

先检查 batch、buffer、metadata、shape/bucket 和 replay launch。

### 9.5 DeepEP 看“等了多久”，不只看“通信跑了多久”

真正应该标：

~~~text
communication issue
        ↓
in-flight interval
        ↓
hook / wait
~~~

如果到 hook / wait 时通信已经基本完成，exposed comm 很小。

如果 hook 后还需要长时间等待，通信就真实暴露在 Critical Path 上。

---

## 十、把性能模型从 Kernel Ranking 升级成 Effective Progress

经过严格 Review 后，更合适的模型是：

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

最终真正关注：

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

---

## 十一、建议的第一轮 A/B 实验

| 实验 | 控制变量 | 主要观察 |
| --- | --- | --- |
| Spec OFF vs DSpark ON | workload 完全相同 | DSpark 是否真正降低 TPOT / 提高 output throughput |
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

否则 Trace 之间无法严格比较。

---

## 十二、一张最终检查表

| Timeline 现象 | 第一检查点 | 不要过早下的结论 |
| --- | --- | --- |
| draft_gpu_ms 变长 | Draft workload / model / side stream | 不要直接等同“Draft Kernel Sum 变长” |
| target_verify_gpu_ms 变长 | verify tokens / prefix / graph bucket | 不要只看 BS |
| Verify 后尾巴变长 | eager Accept / TP Sync / Commit | 当前 NPU 不要说 folded epilogue |
| Replay 前 Host gap | metadata / buffer / scheduler | DSV4 不要直接怪 graph.update |
| DeepEP Communication 很长 | hook / wait 时还有多少未完成 | Total Comm 不等于 Exposed Comm |
| 多 Stream 有交叠 | Critical Path 是否真的缩短 | 有视觉 overlap 不代表有 speedup |
| Kernel Duration Sum 很大 | 是否存在并发 | 不要把 duration sum 当 Step Wall Time |

最后把整套方法收束成：

~~~mermaid
flowchart TD
    A[TTFT / TPOT / Throughput] --> B[DSpark step record]
    B --> C{哪段变慢}
    C --> D[Draft]
    C --> E[Target Verify]
    C --> F[Post-Verify tail]

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

真正成熟的 Profiling，不是找到“最慢 Kernel 是谁”，而是回答：

> **这个 Kernel、通信或 Host gap 是否真的位于本轮输出 Token 的 Critical Path 上？**

对于当前 DeepSeek-V4 + DSpark + Ascend 910C，还要进一步问：

> **Draft 和 Verify 之外的 Eager Accept / Commit 尾巴有多长？DeepEP low-latency 的通信有多少真实暴露在 hook / wait 点？NPU Graph Replay 前的 Host 时间究竟来自 metadata、buffer refresh，还是 scheduler？**

能稳定回答这几个问题后，Profiling 才真正从“看图”变成“性能工程”。

---

## 源码入口索引

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

## 与本仓库其他文章的关系

建议按下面顺序阅读：

1. [DeepSeek-V4 Speculative Decoding 源码解析：Draft → Verify → Accept/Reject](../speculative-decoding/deepseek-v4-speculative-decoding-source-analysis.md)
2. [DeepSeek-V4 Speculative Decoding 性能解析：为什么 Speculative Decoding 不一定更快？](deepseek-v4-speculative-performance-analysis.md)
3. [DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上的性能源码解析：NPU Graph、Multi-Stream 与 Communication Overlap](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)
4. **本文：把源码机制映射到真实 Profiling 时间线。**

下一步真正有价值的工作，是拿一组稳定的 910C Trace，把：

~~~text
forward_ct
Draft ms
Verify ms
Post-Verify residual
DeepEP exposed communication
NPU Graph pre-replay gap
Commit Length
TPOT
~~~

放进同一张表。

那时就可以从源码推导正式进入实测优化闭环。
