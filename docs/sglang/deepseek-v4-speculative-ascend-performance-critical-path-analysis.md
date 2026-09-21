# DeepSeek-V4 Speculative Decoding 在 Ascend 910C 上的性能源码解析：NPU Graph、Multi-Stream 与 Communication Overlap

前面的文章已经把 DeepSeek-V4 Speculative Decoding 的 correctness 链路拆到了 Draft、MTP / NextN、EAGLE Tree、Target Verify、Accept、KV Commit，以及多卡一致性。

继续往性能层深入，真正需要回答的已经不是：

> **一轮 Speculative Decoding 能接受几个 Token？**

而是：

> **一次 speculative iteration 在 Ascend 910C 上，哪些工作真正落在 wall-clock critical path 上？NPU Graph、Plan Stream、Attention Multi-Stream 和 DeepEP async communication 分别能够隐藏什么，又有哪些依赖无论怎么换 Stream 都绕不过去？**

最简单的性能模型常写成：

```text
T_spec
=
T_draft
+
T_verify
+
T_accept
+
T_commit
+
T_comm
```

但真实 Runtime 并不是单线程顺序执行。当前 SGLang + Ascend 路径同时存在 NPU Graph、Plan Stream、Attention Multi-Stream、DeepEP communication stream、DP / TP collectives 和 Graph bucket padding。

真正决定一轮延迟的，因此是依赖图里最长的一条路径：

```text
T_round = CriticalPath(G_round)
```

而不是所有 kernel duration 的机械求和。

> **源码基线**：本文固定基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-21。主线讨论 DeepSeek-V4 + EAGLE / NextN + Ascend 910C + `dsv4` backend。正文中的 timeline 是根据源码中的 Stream / Event / Wait / Graph dependency 推导出的执行关系，不是 910C Profiler 实测值，因此不会虚构任何毫秒数或加速百分比。

---

## 一、先把六个 Strict Review 结论钉死：性能优化首先是依赖关系问题

这篇文章最容易写错的地方，不是某个 Kernel 名字，而是把“异步”“Graph”“Multi-Stream”直接等同于“并行”。

固定源码核对后的结论如下。

| Review 点 | 固定源码结论 |
| --- | --- |
| Draft Graph 一次 replay 到底包含几步 NextN | Graph Capture 包住完整 `draft_forward()`；循环有 `num_steps` 个 candidate stage，但最后一轮在 model forward 前就 `break`，所以真正执行 NextN model forward 的次数是 **`num_steps - 1`**。candidate1 来自上一阶段 Draft Extend。 |
| Target Verify Graph 怎么 bucket | 普通 EAGLE **不是 Ragged Verify**。Target runner 的 `captured_req_width = speculative_num_draft_tokens`；先按 request batch size 选 / pad Graph bucket，再得到 `padded_tokens = bucket_bs × captured_req_width`。开启 DP Attention / MLP gather 时，bucket admission 使用各 DP rank 原始 request count 的最大值。 |
| Plan Stream 是否真的跨阶段 overlap | **不对称。** Verify Planning 显式 `plan_stream.wait_stream(fwd_stream)`，不能与 Draft device compute 重叠；Draft-Extend Planning 没有先 wait main stream，可与已经 enqueue 的 Target Verify / Accept tail 形成设备并发，随后 main stream 在 Draft Extend forward 前 join plan stream。 |
| NPU Multi-Stream 在哪些 ForwardMode 可达 | Target DeepSeek-V4 model 在环境变量开启时创建 alt streams，`DECODE` 和 `TARGET_VERIFY` 可以进入 NPU multi-stream；但 NextN model 构造唯一 decoder 时显式 `alt_streams=None`，所以 Draft `DECODE` 和 `DRAFT_EXTEND_V2` 实际都到不了这条 Attention multi-stream。 |
| DeepEP `async_finish / recv_hook` 隐藏什么 | 它们把 communication launch 与 completion wait 拆开，但 **routed Expert 仍必须等 Dispatch recv 完成**。Normal 模式 `dispatch_b()` 会 wait event；Low-Latency 模式可在 `dispatch_a` launch 和 `dispatch_b` recv-hook 之间插入独立工作，例如 SBO shared expert。真正隐藏的是被独立工作覆盖的 communication tail，不是整个 A2A。 |
| 为什么有 workload “关闭” Draft-Extend Graph | Runtime 正式 kill-switch 是 `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH`，注释明确说某些 capture memory pool / DeepEP full-dispatch workspace 可能“比 Graph 节省的更贵”。但固定版本 Ascend best-practice 页面写的是 `SGLANG_DISABLE_DRAFT_EXTEND_GRAPH=1`；仓库中没有找到这个旧名字的 alias / consumer，因此这是**文档与 runtime 名字漂移**，不能据此断言该配置在这个 commit 实际生效。 |

把六个结论放在一起，可以先得到一轮稳态 EAGLE 的骨架：

```text
上一轮 confirmed state
        │
        ▼
Draft Graph
(num_steps - 1 次 NextN forward)
        │
        ▼
Verify Planning
        │
        ▼
Target Verify Graph
        │
        ▼
Accept / Commit
        │
        ├──────── Draft-Extend Planning
        │           可与 main-stream tail overlap
        ▼
Draft Extend Forward / Graph
        │
        ▼
下一轮 candidate1
```

真正的优化问题因此变成：

> **每一个 Fork 最后在哪里 Join？哪条 branch 最长？哪些 communication / planning 已经被其他工作覆盖，哪些仍然暴露在 Critical Path 上？**

---

## 二、Draft NPU Graph 到底捕获什么：`num_steps=2` 不是两次 NextN Forward

DeepSeek-V4 的 EAGLE Draft 在 NPU 上有专门 Runner：

```text
EAGLEDraftNpuGraphRunner
```

固定源码：

[`eagle_draft_npu_graph_runner.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py)

它继承通用 `EAGLEDraftCudaGraphRunner`，但在 NPU 路径上底层真正使用的是 `torch.npu.NPUGraph`。历史类名里的 CUDA 不代表这里运行 CUDA。

真正值得看的，是 Graph Capture body。

`EAGLEDraftCudaGraphRunner.capture_one_shape()` 最终捕获：

```python
def run_once():
    ...
    ret = self.eagle_worker.draft_forward(
        forward_batch
    )
    ...
    return ret
```

也就是说，一次 Draft Graph Replay 包住的是整个 `draft_forward()`，而不是单独某一次 `draft_runner.forward()`。

固定源码：

[`EAGLEDraftCudaGraphRunner.capture_one_shape()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py)

### `draft_forward()` 中真正有几次 NextN Forward

主循环：

```python
for i in range(self.speculative_num_steps):
    ...
    select_top_k_tokens(...)

    if i == self.speculative_num_steps - 1:
        break

    logits_output = self.draft_runner.forward(
        forward_batch
    ).logits_output
```

固定入口：

[`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L778-L951)

因此严格关系是：

```text
NextN model forward count inside Draft Graph
=
speculative_num_steps - 1
```

例如官方常见：

```text
num_steps = 2
topk = 1
num_draft_tokens = 3
```

这一轮 Draft Graph 实际是：

```text
进入 Graph 前：
candidate1 已由上一轮 Draft Extend 准备好

Graph Replay：
candidate1
    ↓
1 × NextN forward
    ↓
candidate2
    ↓
organize draft result
```

不是两次 NextN forward。

如果 `num_steps=3`，则是：

```text
candidate1
    ↓
NextN forward #1
    ↓
candidate2
    ↓
NextN forward #2
    ↓
candidate3
```

所以 Graph Capture 的确覆盖“整个 multi-step Draft rollout”，但必须严格区分：

```text
candidate stage 数 = num_steps

真正的 NextN model forward 数 = num_steps - 1
```

第一枚 candidate 的成本属于上一轮 Draft Extend，而不是这一轮 Draft Graph。

### Draft Graph 的静态宽度是什么

Draft Decode 的 per-request width 统一由：

```python
resolve_num_tokens_per_req(
    phase="draft_decode"
)
```

解析为：

```text
speculative_eagle_topk
```

固定源码：

[`resolve_num_tokens_per_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L90-L119)

因此当前官方 `topk=1` 时，一次 NextN Draft forward 的输入宽度是 1 token / request，但 Graph body 内会执行 `num_steps-1` 次 NextN forward。

这正是 Graph 对轻量 Draft 特别有价值的原因：

```text
NextN compute 很轻
        ↓
固定 Python / launch / metadata overhead
占比容易变高
        ↓
Capture 整个小循环
比只优化某个单独 kernel 更有意义
```

### Graph/Eager 选择本身也可能是分布式一致性状态

`EAGLEDraftNpuGraphRunner.can_run_graph()` 在特定 DSA IndexShare + DP Attention 场景下，会把本地 `can_run_graph && seed_ready` 做成一个整数，然后对 model TP group：

```python
torch.distributed.all_reduce(
    decision,
    op=MIN,
    group=tp_group,
)
```

原因是 Draft forward 里仍包含 TP / EP collective。

如果同一个 distributed forward 中出现：

```text
Rank0 → Graph
Rank1 → Eager
```

collective 调用次序可能失去一致性。

因此：

> **Graph/Eager route 本身也可能属于 speculative control plane，必须在协作 Rank 间收敛。**

---

## 三、Target Verify Graph 的真实 Bucket：普通 EAGLE 按 Request Bucket Capture，每个 Request 固定 Verify Width

Draft 完成以后，Target Verify 会进入通用 `DecodeCudaGraphRunner / NPUGraphRunner`。

固定 NPU 入口：

[`NPUGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py)

Target Runner 初始化时：

```python
self.captured_req_width
=
model_runner.decode_num_tokens_per_req(
    num_draft_tokens=...
)
```

对于普通 EAGLE Target：

```text
captured_req_width
=
speculative_num_draft_tokens
```

固定来源：

- [`ModelRunner.decode_num_tokens_per_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/model_runner.py#L825-L846)
- [`resolve_num_tokens_per_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L90-L119)

因此如果 `num_draft_tokens=3`，每个请求 Target Verify 的 Graph 宽度固定是 3 rows。

### 当前 EAGLE 不是 Ragged Verify

这一点必须和 DSpark 区分。

`SpeculativeAlgorithm.supports_ragged_verify()` 在固定版本中：

```python
return self.is_dspark()
```

固定源码：

[`spec_info.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_info.py#L160-L164)

所以当前 EAGLE / NextN Target Verify 不是：

```text
Req0 verify 2 tokens
Req1 verify 3 tokens
Req2 verify 1 token
        ↓
按 sum(verify_lens) 选择 token bucket
```

而是每个 request 固定 Verify `num_draft_tokens` rows。

Accept Length 是 Verify **之后**才知道的，因此不能用本轮 acceptance 较短去减少当前这一次 EAGLE Target Verify 的 Graph Work。

### 普通 EAGLE Graph 首先按 Request Batch Size 选 Bucket

非 Ragged path 的 `load_batch()`：

```python
raw_num_token
=
raw_bs * captured_req_width

bs
=
_pad_to_bucket(
    raw_bs,
    capture_bs,
)

padded_num_tokens
=
bs * captured_req_width
```

固定源码：

[`DecodeCudaGraphRunner.load_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)

假设：

```text
raw_bs = 5
num_draft_tokens = 3

Graph request buckets:
1, 2, 4, 8
```

则：

```text
真实 Verify rows = 5 × 3 = 15

Graph bucket = 8 requests

Graph replay rows = 8 × 3 = 24
```

其中 9 rows 来自 bucket padding。

所以 Graph 的收益和代价同时存在：

```text
收益：
减少 launch / Python / control overhead

代价：
执行 padded request slots 对应的额外 token rows
```

### DP Attention 下 admission 甚至不是本 Rank 的 raw_bs

如果需要 MLP TP gather，`can_run_graph()` 会用：

```python
cuda_graph_bs
=
max(
    forward_batch.original_global_num_tokens_cpu
)
```

也就是 Attention-DP 各 Rank 原始 request count 的最大值。

随后相关 Rank 进入兼容这个 global-max geometry 的 Graph bucket。

因此对 DP Attention + Speculation，Graph padding 还承担一个分布式目的：

> **让协作 Rank 在同一 captured collective geometry 上 replay。**

分析 Target Verify Graph 时，不能只看当前 Rank 有几个 Request，还要看：

```text
DP group 最大 Request Count
×
固定 Verify Width
×
Capture Bucket
```

### `--cuda-graph-bs-decode` 在这里是 Request Bucket，不是 EAGLE Verify Token Bucket

固定 DeepSeek-V4 Ascend best-practice 中可以看到：

```text
--cuda-graph-bs-decode 1 2 4 8
```

或：

```text
1 2 4 8 10
```

在当前普通 EAGLE 路径里，这些 key 首先表示 captured request slots。

若 `num_draft_tokens=3`，Target Verify 对应的 token rows 才是：

```text
bucket_bs × 3
```

而 DSpark Ragged Verify 的 `graph_num_tokens tier` 是另一套机制，不能混进当前 EAGLE 性能模型。

---

## 四、Plan Stream 的真实作用并不对称：Verify Planning 不 overlap Draft，Draft-Extend Planning 才存在跨阶段设备并发

环境变量：

```text
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
```

会创建：

```python
plan_stream
=
torch.get_device_module(device).Stream()
```

固定源码：

[`get_plan_stream()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L1132-L1140)

但“有第二条 Stream”并不意味着任何两个阶段真的在设备上并行。

关键要看 `wait_stream` 在哪里。

### Verify Planning：不能与 Draft Device Compute 重叠

`run_eagle_verify()` 一开始拿到当前 compute stream：

```python
fwd_stream = current_stream()
```

随后：

```python
with plan_stream_ctx:
    plan_stream.wait_stream(
        fwd_stream
    )

    eagle_prepare_for_verify(...)
```

源码注释明确说明：Verify Prep 会读取 Draft 生成的 Tree metadata，因此不能在 Draft frontier 完成以前开始。

Plan 完成后，main stream 又：

```python
current_stream().wait_stream(
    plan_stream
)
```

才启动 Target Verify。

固定源码：

[`run_eagle_verify()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L462-L675)

Device dependency 实际是：

```text
Main Stream
Draft
████████████
            │
            ▼

Plan Stream
            Verify Plan
            ███████
                   │
                   ▼

Main Stream
                   Target Verify
                   ███████████████
```

因此不存在 Draft Compute 与 Verify Planning 的设备重叠。

Plan Stream 在这里的意义更接近：

- 把 planning device work 放到独立 stream；
- 减少不必要的 Host blocking；
- 明确 metadata ownership；
- 用 Stream dependency 而不是 Host 同步表达执行顺序。

但真实的数据依赖仍然位于 critical path。

### Draft-Extend Planning：这里反而真的可以跨阶段 overlap

Verify 完成后，`_draft_extend_for_decode()` 先构造 `EagleDraftExtendInput`、`select_index` 和 `next_token_ids`，然后直接：

```python
with self.plan_stream_ctx:
    forward_batch = prepare_for_draft_extend(...)
```

注意这里在 planning 之前**没有**：

```python
plan_stream.wait_stream(
    current_stream
)
```

固定源码：

[`_draft_extend_for_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py)

Python 从 `run_eagle_verify()` 返回，并不表示之前 enqueue 的 NPU kernels 已经执行完成。设备执行是异步的，因此 Plan Stream 上的 Draft-Extend Planning 可以与 Main Stream 上尚未完成的 Target Verify / Accept tail 形成并发，只要 Planning 本身不读取那些尚未 ready 的 Device value。

源码还专门把：

```python
next_token_ids
=
batch_result.next_token_ids.to(torch.int64)
```

放在进入 Plan Stream **之前**，并写明这是为了避免在 Plan Stream 内发生额外 cross-stream synchronization / data race。

随后真正进入 Draft Extend Forward 前：

```python
current_stream().wait_stream(
    self.plan_stream
)
```

因此更准确的 Timeline 是：

```text
Main Stream
Target Verify
████████████████
        Accept kernels
        ████████
                │

Plan Stream
        Draft-Extend Planning
        ███████████
                │

Main Stream
                wait(plan)
                    │
                    ▼
             Draft Extend Forward
             ███████████
```

所以不能笼统写：

```text
Plan Stream = Verify Planning 与 Draft 并行
```

更准确的是：

```text
Verify Prep:
Draft → Plan → Verify
严格依赖

Draft-Extend Prep:
Target Verify / Accept main-stream tail
        ∥
Draft-Extend Planning
        ↓
Join
        ↓
Draft Extend Compute
```

---

## 五、NPU Multi-Stream 的实际可达性：Target DECODE / TARGET_VERIFY 可用，NextN Draft 实际不可达

另一套容易和 Plan Stream 混淆的机制是：

```text
SGLANG_NPU_USE_MULTI_STREAM=1
```

它不是 speculative phase planner，而是深入 DeepSeek-V4 **单个模型 Layer 内部**的局部并行。

Target DeepSeek-V4 Model 初始化时，只要 NPU 环境变量开启，就创建一组 `alt_streams`，并把它传进每个 `DeepseekV4DecoderLayer`。

固定源码：

[`DeepseekV4Model.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py)

Attention Layer 收到 alt streams 后，NPU eligibility 是：

```python
_is_npu
and SGLANG_NPU_USE_MULTI_STREAM
and self.alt_streams is not None
and x.shape[0] <= self._multi_stream_bs_limit
and not forward_batch.forward_mode
    .is_extend_or_draft_extend_or_mixed()
```

固定源码：

[`DeepSeek-V4 Attention forward`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py)

这里有两个容易误读的地方。

### `TARGET_VERIFY` 没有被排除

`ForwardMode.is_extend()` 确实把 `TARGET_VERIFY` 看成 extend-family。

但 Multi-Stream 条件调用的不是 `is_extend()`，而是：

```text
is_extend_or_draft_extend_or_mixed()
```

这个 helper 的默认参数：

```text
include_draft_extend_v2 = False
```

所以默认只排除：

```text
EXTEND
MIXED
SPLIT_PREFILL
```

并不排除 `TARGET_VERIFY`。

固定源码：

[`ForwardMode`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L176-L275)

所以 Target Model 的：

```text
DECODE
TARGET_VERIFY
```

都可以满足 mode 这一层条件。

### `DRAFT_EXTEND_V2` 也没有被 helper 默认排除，但 NextN 自己没有 alt streams

更反直觉的是，`DRAFT_EXTEND_V2` 在默认参数下也不被这个 helper 排除。

如果只看 ForwardMode，会误以为 NextN Draft Extend 也能进入 NPU Multi-Stream。

但 DeepSeek-V4 NextN Model 构造唯一 decoder layer 时显式：

```python
self.decoder = DeepseekV4DecoderLayer(
    ...,
    alt_streams=None,
    ...
)
```

固定源码：

[`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py#L97-L105)

所以 NextN layer 内 `self.alt_streams=None`。

最终实际可达性是：

| Phase | ForwardMode | Model | Attention NPU Multi-Stream |
| --- | --- | --- | --- |
| 普通 Target Decode | `DECODE` | Target | **可达**，还需 env / token rows / layer 条件满足 |
| EAGLE Target Verify | `TARGET_VERIFY` | Target | **可达** |
| EAGLE Draft Decode | `DECODE` | NextN | **不可达**，NextN decoder 的 `alt_streams=None` |
| EAGLE Draft Extend | `DRAFT_EXTEND_V2` | NextN | **不可达**，同上 |
| 普通 Prefill | `EXTEND` | Target | 该 NPU Attention Multi-Stream path 被 mode 条件排除 |

### Target Multi-Stream 内到底并行什么

真正进入 `_forward_prepare_multi_stream_npu()` 后：

```text
stream_kv
→ KV projection / Norm / RoPE / KV cache store

stream_q
→ Q projection / RMSNorm / RoPE

current stream
→ Indexer / Compressor
```

最后：

```python
current_stream.wait_stream(stream_kv)
current_stream.wait_stream(stream_q)
```

才进入 downstream Attention。

固定源码：

[`_forward_prepare_multi_stream_npu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1537-L1635)

因此局部 Critical Path 理想模型更接近：

```text
T_prepare
≈
T_prefix
+
max(
  T_Q,
  T_KV,
  T_indexer/compressor
)
```

而不是三者相加。

但源码只能证明这些工作**被允许并发**。是否能在 910C 上完全重叠，还受 Cube / Vector 资源竞争、HBM 带宽、Kernel occupancy、CANN scheduling 和实际 token rows 影响，需要 Profiler Trace 验证。

另外固定源码中非 Blackwell 的 Multi-Stream row limit 是 64。这里判断的是当前 Layer 的 `x.shape[0]`，不是服务器的 `max_running_requests`。

---

## 六、DeepEP `async_finish / recv_hook` 到底能隐藏哪段通信：隐藏的是 Wait 前的独立工作，不是 Expert 的数据依赖

DeepSeek-V4 Target Verify 还会进入 MoE。

官方 Ascend recipe 使用：

```text
--moe-a2a-backend deepep
--deepep-mode auto
```

FusedMoE 创建 DeepEP dispatcher 时传入：

```python
async_finish=True
return_recv_hook=True
```

固定入口：

[`fused_moe_triton/layer.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/fused_moe_triton/layer.py)

但这两个参数并不意味着：

```text
Dispatch
∥
Routed Expert GEMM
```

天然并行。

### Normal DeepEP：通信异步 Launch，但 Expert 前仍必须 Wait

Normal dispatcher：

```python
previous_event = Buffer.capture()

buffer.dispatch(
    ...,
    async_finish=True,
    allocate_on_comm_stream=True,
)
```

通信可以在 Communication Stream 上启动。

但 `dispatch_b()` 随后：

```python
event.current_stream_wait()
```

然后才返回 Expert 输入。

固定源码：

[`_DeepEPDispatcherImplNormal`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L537-L735)

因此真实依赖仍然是：

```text
DeepEP Dispatch
        ↓
recv complete barrier
        ↓
Routed Expert GEMM
```

`async_finish=True` 本身不能把 Routed Expert 提前到 recv 完成之前。

Combine 也是同样结构：

```text
Combine async launch
        ↓
event.current_stream_wait()
        ↓
combined hidden consumer
```

所以 Normal mode 的 async API 主要提供：

> **把 communication 放到 communication stream 执行的能力；真正能隐藏多少，取决于 Wait 之前有没有别的 independent work。**

### Shared Expert 正好是一个独立 branch

DeepSeek MoE 在 Shared Expert 没有 fuse 到 routed path 时，可以把 Shared Expert 放到 alt stream：

```python
self.alt_stream.wait_stream(
    current_stream
)

with alt_stream:
    shared_output
    =
    _forward_shared_experts(...)
```

随后主路径继续：

```text
TopK
→ DeepEP Dispatch
→ Routed Experts
→ Combine
```

最终再等待 Shared Expert output。

固定源码：

[`DeepseekV2MoE.forward_deepep()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v2.py#L1418-L1665)

因此可能形成：

```text
Routed path
Router
  ↓
Dispatch ───────── Expert ───── Combine
██████████████████████████████████████

Shared side stream
Shared Expert
████████████████
```

这里 Shared Expert 可以覆盖 Routed Path 的一部分时间，其中也可能包含 Communication。

真正值得看的不是 DeepEP 总通信时间，而是：

```text
通信结束以后
还有多少 Tail
没有被 independent compute 覆盖
```

### Low-Latency DeepEP：`recv_hook` 把 Wait 推迟到 Consumer Boundary

Low-latency dispatch 调用：

```python
buffer.low_latency_dispatch(
    ...,
    async_finish=False,
    return_recv_hook=True,
)
```

得到：

```text
event
hook
```

外层 `DeepEPDispatcher.dispatch()` 的顺序是：

```text
dispatch_a()
    ↓
deepep_dispatch_hooks
    ↓
dispatch_b()
```

而 Low-Latency `dispatch_b()`：

```python
hook()
```

才真正等待 / 完成接收。

固定源码：

[`_DeepEPDispatcherImplLowLatency`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py)

这提供了一个真实 overlap slot：

```text
low_latency_dispatch launch
        ↓
[ dispatch hook 可执行 independent work ]
        ↓
recv_hook()
        ↓
Expert consumes received tokens
```

SBO 路径正是利用这个 slot，把 Shared Expert 等独立计算放进 Communication 等待窗口。

因此 `recv_hook` 的真正价值不是“不再等待”，而是：

> **把 Wait Point 从 Communication Launch 处推迟到真正消费 recv buffer 的边界，让中间可以插入与 recv 数据无关的 Compute。**

Combine 侧也是同样思想；有 `overlap_args` 时 Low-Latency combine 可以切到 overlap stream，最终仍在 Consumer Boundary Join。

### 应该看 Exposed Communication，不是 Total Communication

如果某段 Communication 时间是 `T_comm`，同时独立计算能覆盖 `T_independent`，理想情况下真正暴露在 Critical Path 上的是：

```text
T_exposed_comm
=
max(
  0,
  T_comm - T_independent
)
```

如果 async launch 后立刻 Wait：

```text
T_independent = 0
```

那么：

```text
T_exposed_comm ≈ T_comm
```

所以“async”并不自动带来性能收益。

---

## 七、官方 DeepSeek-V4 配置为什么有时“关闭 Draft-Extend Graph”：先区分设计动机和文档变量名漂移

固定版本 Ascend DeepSeek-V4 best-practice 中，部分 workload 可以看到：

```bash
export SGLANG_DISABLE_DRAFT_EXTEND_GRAPH=1
```

同时还开启：

```bash
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
```

并使用 EAGLE、NPU Graph 和 DeepEP。

第一眼很容易解释成：

> “这个 workload 实测发现 Draft Extend Graph 更慢，所以官方关闭。”

但 Strict Review 后不能直接这样写。

### Runtime 真正读取的是另一个变量名

固定 `environ.py` 注册的是：

```text
SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH
```

其源码注释：

> Kill-switch for draft-extend cuda graph. Escape hatch for setups where the capture's memory pool costs more than the graph saves, e.g. DeepEP MoE workspace captured at full dispatch capacity.

固定源码：

[`environ.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1371-L1375)

EAGLE Worker 创建 Draft-Extend Graph 时真正检查的也是：

```python
not envs
    .SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH
    .get()
```

固定入口：

[`EagleDraftWorker._capture_cuda_graphs()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py)

但固定版本 best-practice 页面写的是：

```text
SGLANG_DISABLE_DRAFT_EXTEND_GRAPH
```

少了 `_CUDA_`。

在这个 commit 的仓库里没有找到旧名字对应的 `EnvBoolWithAlias`、consumer 或 translation。

因此这条文档配置存在明显的：

```text
doc / runtime env-name drift
```

除非外部启动包装层另做转换——固定仓库中没有证据——否则不能仅根据这行文档断言 Draft-Extend Graph 在该 workload 中真的被关闭。

### 但“为什么需要这个 Kill Switch”有明确源码依据

正确 Runtime flag 的注释已经给出设计动机：

```text
Draft-Extend Graph Capture
        ↓
静态 memory pool / workspace
        ↓
某些 MoE / DeepEP workspace
按 full dispatch capacity 捕获
        ↓
常驻 footprint 变大
        ↓
Graph 节省的 launch latency
可能不值这份 memory cost
```

Draft Extend 本身又只是 NextN 的一段相对短 Forward。

所以它容易落入这样的 Trade-off：

```text
Graph benefit
=
saved launch/control overhead
-
capture memory cost
-
padding cost
-
replay preparation cost
```

当：

```text
Draft Extend compute 很短
+
DeepEP workspace capture 很大
```

时：

```text
Draft Decode Graph       ON
Target Verify Graph      ON
Draft Extend             Eager
```

完全可能是合理 operating point。

但要强调：

> 这是 **Runtime Kill-Switch 的设计动机**，不是对固定 best-practice 那个旧变量名已经生效的证明。

### Draft-Extend Graph 自己也有独立静态宽度

它的 Graph Runner `EAGLEDraftExtendCudaGraphRunner` 不是按 `num_steps + 1` 捕获。

源码明确：

```python
captured_req_width
=
resolve_num_tokens_per_req(
    phase="draft_extend"
)
```

也就是：

```text
speculative_num_draft_tokens
```

固定源码：

[`EAGLEDraftExtendCudaGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py)

这意味着即使最终 `accept_lens` 较短，Draft Extend Graph 的 Buffer / Capture Geometry 仍按照完整 speculative width 设计；真正 LM Head 只在 selected last-accepted rows 上做后续处理。

这也是为什么 Draft-Extend Graph 的 Memory Footprint 和 Launch Savings 需要单独评估，而不能因为“NextN 很轻”就默认 Graph 一定更赚。

---

## 把所有机制放回一轮 Ascend 910C Timeline

现在可以画出比“Draft + Verify + Accept”更接近真实 Runtime 的时间线：

```text
Time ─────────────────────────────────────────────────────────────→

MAIN COMPUTE
│
│ [ Draft NPU Graph ]
│   num_steps-1 × NextN forward
│ █████████████████
│                  │
│                  │ Draft result dependency
│                  ▼
│                       [ Target Verify NPU Graph ]
│                       ████████████████████████████
│                                             [Accept]
│                                               ███
│                                                   │
│                                                   │ wait plan
│                                                   ▼
│                                             [Draft Extend]
│                                             ███████████
│
├───────────────────────────────────────────────────────────────

PLAN STREAM
│
│                  [Verify Plan]
│                  ███████
│                  ↑ 必须等待 Draft
│
│                                      [Draft-Extend Plan]
│                                      ██████████
│                                      ↑ 可与 main-stream verify/accept tail overlap
│
├───────────────────────────────────────────────────────────────

TARGET ATTENTION LAYER
│
│ current:   [Indexer / Compressor────────────]
│ q-stream:  [Q projection / norm / RoPE─────]
│ kv-stream: [KV / norm / RoPE / cache──────]
│                                       │
│                                       └──── Join → Attention
│
├───────────────────────────────────────────────────────────────

TARGET MoE
│
│ routed path:
│ Router → Dispatch(comm) → wait → Experts → Combine(comm) → wait
│
│ shared side stream:
│          [ Shared Expert ─────────────────── ]
│
│ 真正可隐藏的是被 independent compute 覆盖的 communication tail
│
└───────────────────────────────────────────────────────────────
```

拿 Ascend Profiler 分析一轮 speculative decode 时，可以按这个顺序看：

| 观察点 | 对应源码问题 |
| --- | --- |
| Draft | Graph replay 还是 eager？`num_steps` 对应几次真正 NextN forward？ |
| Draft Graph | raw request count pad 到哪个 bucket？ |
| Verify seam | Draft 结束到 Verify start 中间，Verify Plan gap 多长？这段不能与 Draft overlap。 |
| Target Verify | 固定 Verify width 是多少？`bucket_bs × num_draft_tokens` 有多少 padding rows？ |
| Plan overlap | Draft-Extend Plan 是否真的与 Verify / Accept tail 同时出现在 NPU timeline？ |
| Attention | Q / KV / Indexer 三条 branch 哪条最长？Multi-Stream Join 最终等谁？ |
| MoE | DeepEP 总 duration 中，有多少被 Shared Expert / SBO 覆盖？真正 exposed tail 是多少？ |
| Rank control | 是否因为某 Rank Graph fallback 导致整个 TP group 一起 eager？ |
| Draft Extend | Graph 还是 eager？Capture memory / workspace 是否值得？ |

最终性能公式可以重新写成：

```text
Speedup
≈
AcceptedProgress × BaselineDecodeCost
────────────────────────────────────
CriticalPath(
  DraftGraph,
  VerifyPlan,
  TargetVerify,
  LayerMultiStream,
  MoECommunication,
  Accept,
  DraftExtendPlan,
  DraftExtend
)
```

Graph 优化的是固定 Launch / Control Overhead。

Plan Stream 优化的是可安全异步化的 Phase Planning，其中只有部分 seam 真正形成跨阶段设备 overlap。

NPU Multi-Stream 优化的是 **Target DeepSeek-V4 Layer** 内部的局部 Compute Fork；当前 NextN Draft 本身并没有接上这套 alt streams。

DeepEP async / recv-hook 优化的是 Wait Point，让 independent work 有机会覆盖 Communication Tail。

而下面这些真实依赖仍然无法消失：

```text
Draft tree ready
→ Verify topology / metadata

received Expert tokens ready
→ Routed Expert GEMM

Q / KV / Indexer ready
→ Attention

Accept decision ready
→ committed seq_len / next-round state

Draft-Extend plan ready
→ Draft Extend Forward
```

所以生产级 Speculative Decoding 性能优化，最终不是“把 Acceptance Rate 调高”这么简单。

它已经变成：

```text
Algorithm
    ↓
Work Geometry
    ↓
Graph Bucket
    ↓
Stream Dependency
    ↓
Communication Event
    ↓
Join Point
    ↓
Critical Path
```

这才是从源码走向 Ascend 910C Profiler 时最重要的分析框架。

### 源码阅读入口

| 目标 | 固定版本源码 |
| --- | --- |
| EAGLE Worker | [`eagle_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py) |
| Draft NPU Graph Runner | [`EAGLEDraftNpuGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py) |
| Draft Graph Capture Body | [`EAGLEDraftCudaGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py) |
| Draft Multi-Step Loop | [`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L778-L951) |
| Spec Phase Static Width | [`resolve_num_tokens_per_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L90-L119) |
| Target NPU Graph | [`NPUGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py) |
| Target Graph Fixed Width / Bucket | [`DecodeCudaGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py) |
| EAGLE / DSpark Ragged Verify 边界 | [`SpeculativeAlgorithm.supports_ragged_verify()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_info.py#L160-L164) |
| Verify Preparation | [`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L520-L622) |
| Verify + Plan Stream Dependency | [`run_eagle_verify()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L462-L675) |
| Plan Stream Creation | [`get_plan_stream()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L1132-L1140) |
| Draft-Extend Planning Overlap | [`_draft_extend_for_decode()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py) |
| Draft-Extend NPU Graph | [`EAGLEDraftExtendNpuGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/graph_runner/eagle_draft_extend_npu_graph_runner.py) |
| Draft-Extend Graph Buffer / Width | [`EAGLEDraftExtendCudaGraphRunner`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py) |
| Target NPU Multi-Stream Pool | [`DeepseekV4Model.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py) |
| NPU Attention Multi-Stream | [`_forward_prepare_multi_stream_npu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1537-L1635) |
| NextN Explicit `alt_streams=None` | [`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4_nextn.py#L97-L105) |
| ForwardMode Boundary | [`ForwardMode`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/model_executor/forward_batch_info.py#L176-L275) |
| DeepEP Normal Async Boundary | [`_DeepEPDispatcherImplNormal`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py#L537-L735) |
| DeepEP Low-Latency Recv Hook | [`_DeepEPDispatcherImplLowLatency`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/moe/token_dispatcher/deepep.py) |
| DeepSeek Shared-Expert Overlap | [`DeepseekV2MoE.forward_deepep()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v2.py#L1418-L1665) |
| Draft-Extend Graph Kill Switch | [`environ.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/environ.py#L1371-L1375) |
| Ascend DeepSeek-V4 Best Practice | [`deepseek_v4_flash.mdx`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash.mdx) |