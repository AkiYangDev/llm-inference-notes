# SGLang Speculative Decoding Scheduler 源码解析：ScheduleBatch 如何管理 Draft、Verify、Accept 状态机？

前面的几篇文章已经分别拆过 Draft、Target Verify、Tree Attention、Accept Decision、KV Commit / Compact / Rollback，以及多卡场景下 Draft / Verify / Accept 如何保持一致。

把这些模块拼起来以后，还剩下一个更底层的问题：

> **到底是谁在驱动这一整轮 Speculative Decoding？**

直觉上很容易把它理解成：

```text
Scheduler
  ↓
调度 Draft Batch
  ↓
调度 Verify Batch
  ↓
调度 Accept Batch
  ↓
调度下一轮
```

但当前 SGLang Spec V2 的实际设计并不是这样。

更准确的结构是：

```text
                    Scheduler

                ScheduleBatch(DECODE)
                        │
                        │ 一次 run_batch()
                        ▼
              Speculative Model Worker
                        │
          ┌─────────────┼─────────────┐
          │             │             │
        Draft        Verify         Accept
          │             │             │
          └─────────────┼─────────────┘
                        │
                  Draft / State Commit
                        │
                        ▼
              GenerationBatchResult
                 /                \
                /                  \
               ▼                    ▼
       CPU Result Settle       Next-round Relay
               │                    │
               ▼                    ▼
          Req / KV ledger      FutureMap / spec_info
```

也就是说：

> **Scheduler 通常调度的是一次 speculative decode iteration，而不是分别调度 Draft、Verify、Accept 三个全局 batch；真正的 Draft → Verify → Accept 微状态机在 Spec Worker 内部展开。**

这篇文章就沿着这个边界讲清楚：

```text
Req / ReqKvInfo
       ↓
ScheduleBatch
       ↓
ForwardBatch
       ↓
Spec Worker
       ↓
GenerationBatchResult
       ↓
FutureMap + BatchResultProcessor
       ↓
下一轮
```

> **源码基线**：本文基于 `sgl-project/sglang @ 5f017ffabb6ab8d214f6a4616ee8bd98a376034a`，核对日期为 2026-09-20。重点覆盖当前 Spec V2 Scheduler、EAGLE / EAGLE3、DSpark、DFLASH、UNO、NGRAM 及 overlap relay。
>
> 本文重点不是再次解释 speculative decoding 算法，而是解释 **Scheduler ownership、跨轮时钟、KV watermark、forward transaction 与 overlap relay**。

---

## 一、先说结论：SGLang 有两层状态机，而不是一个大状态机

理解 Speculative Scheduler 最容易犯的错误，是把 Scheduler 状态和 speculative algorithm 状态混成一层。

实际上至少有两层。

### 第一层：Scheduler Macro State

Scheduler 关心的是：

```text
这个 Request 现在能不能跑？

属于 Prefill 还是 Decode？

这一轮需要多少 KV capacity？

可以和哪些 Request 组 Batch？

有没有 finished / retracted？

下一轮还要不要留在 running_batch？
```

所以 Scheduler 眼里的 Request 生命周期更接近：

```text
WAITING
   │
   ▼
PREFILL / EXTEND
   │
   ▼
RUNNING / DECODE
   │
   ├───────────────┐
   │               │
   ▼               ▼
FINISHED         RETRACT
```

这里最核心的数据结构是：

```text
Req
ScheduleBatch
```

---

### 第二层：Spec Worker Micro State

当 Scheduler 已经决定：

```text
这一轮跑 DECODE
```

以后，Spec Worker 才进一步展开：

```text
Draft / Proposal
       ↓
Target Verify
       ↓
Accept
       ↓
Device-side state commit
       ↓
Draft Extend / next-state preparation
```

例如 EAGLE V2 的入口：

[`EAGLEWorkerV2.forward_batch_generation()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/eagle_worker_v2.py)

Decode 分支会在同一个 Scheduler iteration 内完成：

```text
draft()
  ↓
verify()
  ↓
eagle_sample()
  ↓
accepted-path commit
  ↓
_draft_extend_for_decode()
```

而 DSpark 则是：

```text
proposer.propose()
        ↓
resolve_verify_token_budget()
        ↓
schedule_layout()
        ↓
Target Verify
        ↓
accept_and_finalize()
        ↓
commit_hidden / state
```

对应：

[`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

所以同样一个：

```text
ScheduleBatch.forward_mode = DECODE
```

进入 Spec Worker 后，内部可能创建多个不同用途的 `ForwardBatch`：

```text
ScheduleBatch(DECODE)
        │
        ├─ Draft ForwardBatch
        ├─ TARGET_VERIFY ForwardBatch
        └─ DRAFT_EXTEND_V2 ForwardBatch
```

这就是全文最重要的第一层抽象：

> **`ScheduleBatch(DECODE)` 是 Scheduler 的宏观 transaction；Draft / Verify / Accept 是这个 transaction 内部的执行阶段。**

---

### 为什么 `model_worker` 是这个边界的关键

Scheduler 初始化 Worker 时：

```python
if self.spec_algorithm.is_none() or self.draft_worker is None:
    self.model_worker = self.tp_worker
else:
    self.model_worker = self.draft_worker
```

固定源码：

[`Scheduler.init_model_worker()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler.py)

因此普通生成：

```text
Scheduler
   ↓
TpModelWorker
   ↓
Target Model
```

Speculative Decoding：

```text
Scheduler
   ↓
Spec Worker
   ├─ Draft side
   └─ Target Worker
```

Scheduler 调用的仍然只是：

```python
self.model_worker.forward_batch_generation(batch)
```

但 Spec Worker 会把这一轮展开成完整的 speculative pipeline。

---

### `ForwardMode` 也说明了两层状态

当前源码中：

```text
EXTEND
DECODE
MIXED
IDLE

TARGET_VERIFY
DRAFT_EXTEND_V2
```

其中：

```text
DECODE
```

主要是 Scheduler 层的一轮 decode transaction。

而：

```text
TARGET_VERIFY
DRAFT_EXTEND_V2
```

则是 Worker / ModelRunner 真正执行某个内部 forward 时的设备执行模式。

因此不要把：

```text
ScheduleBatch.forward_mode == DECODE
```

理解成：

> “这一轮 GPU 只做了一次普通 Decode。”

在 Spec V2 中完全不是。

---

## 二、六个核心对象与三套时钟：Overlap 下它们穵竟允许错开多少？

真正理解 Scheduler，要先分清六个对象：

```text
Req
ReqKvInfo
ScheduleBatch
ForwardBatch
GenerationBatchResult
FutureMap
```

源码自己对前两层数据流有非常明确的说明：

```text
ScheduleBatch -> ForwardBatch

ScheduleBatch:
  managed by Scheduler
  high-level scheduling data

ForwardBatch:
  managed by ModelRunner
  low-level tensor data
```

固定源码：

[`schedule_batch.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/schedule_batch.py)

但 speculative + overlap 后，真正困难的是：

> **这些对象不再保证在任意瞬间都显示同一个 sequence length。**

这不是 bug，而是设计。

---

### `Req.seqlen`：用户逻辑序列时钟

定义非常直接：

```python
@property
def seqlen(self) -> int:
    return len(self.origin_input_ids) + len(self.output_ids)
```

所以它回答：

> **CPU Request 目前已经正式接受了多少逻辑 token？**

注意它依赖：

```text
req.output_ids
```

而 `output_ids` 是在 `BatchResultProcessor` 处理本轮结果时才增加的。

因此 overlap 下它天然是一个 **host-settled clock**。

---

### `kv_committed_len`：Host KV Ledger 的 committed boundary

`ReqKvInfo` 中：

```python
kv_committed_len: int = 0
kv_allocated_len: int = 0
```

源码注释：

```text
kv_committed_len:
KV content committed up to here,
<= kv_allocated_len
```

固定入口：

[`ReqKvInfo`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/schedule_batch.py)

它回答：

> **Host bookkeeping 认为当前正式 committed 的 KV boundary 在哪里？**

Spec V2 里一个极其重要的 ownership rule 是：

```text
Draft Worker
不得自己推进 kv_committed_len
```

仓库甚至有专门的 AST-level regression test：

[`test_decode_bookkeeping_ownership.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/test/registered/unit/spec/test_decode_bookkeeping_ownership.py)

其中明确规定：

```text
spec v2:
no pre-claim;
resolve commits the full accepted run uniformly.
```

`kv_committed_len` 的正常 Spec V2 owner 是：

```text
SchedulerBatchResultProcessor._resolve_spec_v2_tokens()
```

而不是 Draft Worker。

---

### `ScheduleBatch.seq_lens`：本轮 Device Forward Frontier

这一项最容易和前两个混淆。

Spec V2 的约定是：

```text
batch.seq_lens
=
当前这一轮 forward 能安全认为已经 KV-ready 的 prefix boundary
```

在 EAGLE / DSpark decode 中，Verify 完成后会生成：

```python
new_seq_lens = old_seq_lens + accept_lens
```

然后通过 `FutureMap.publish()` 让下一轮 device-side scheduling 提前看到这个新 frontier。

因此在 overlap 下：

```text
ScheduleBatch.seq_lens
```

可以已经是：

```text
Verify N 的新 frontier
```

而 CPU 上：

```text
Req.seqlen
Req.kv.kv_committed_len
```

还没有处理完 Verify N 的结果。

---

### 严格 Review 结论：允许错开“最多一个 Verify iteration”，不是“一个 Token”

这点源码里有非常直接的注释：

```text
kv_committed_len lags device seq_lens
by up to one verify under overlap;
the 2x window absorbs it.
```

固定源码：

[`mamba_lazy_spec_in_window()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/schedule_batch.py)

关键是：

> **one verify，不是 one token。**

如果一次 Verify 接受：

```text
accept_lens = 4
```

那么 overlap 窗口里完全可能出现：

```text
Device frontier:
ScheduleBatch.seq_lens = C + 4

Host ledger:
Req.kv.kv_committed_len = C
```

差距是：

```text
4 tokens
```

但仍然只是：

```text
1 个尚未 CPU settle 的 verify iteration
```

这和 `event_loop_overlap()` 的结构完全一致。

它最多保留一个尚未处理的上一批结果：

```text
run current batch
    ↓
append result
    ↓
process previous batch result
```

所以标准 overlap pipeline 是 **一轮 forward lookahead**。

---

### 为什么 reserve 要做 2x？

答案就在：

[`get_alloc_reserve_per_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6ee8bd98a376034a/python/sglang/srt