# SGLang 投机解码调度：请求进度、设备状态与跨轮交接

Scheduler 通常把一轮投机解码作为一次 DECODE 工作交给 Spec Worker；Draft、Verify、Accept 和提交在 Worker 内部展开。返回后，Scheduler 再更新请求状态，并安排下一轮。

开启 Overlap 后，设备已经确定的下一轮输入可能领先于 CPU 完成的结果处理。本文跟随同一条请求说明这种错开如何由序列长度、KV 预留、publish 和 FutureMap 管理，再把 Mixed Tail 的特殊交接放到进阶部分。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码基线**：本文基于 `sgl-project/sglang @ 5f017ffabb6ab8d214f6a4616ee8bd98a376034a`，核对日期为 2026-09-20。重点覆盖当前 Spec V2 Scheduler、EAGLE / EAGLE3、DSpark、DFLASH、UNO、NGRAM 及 overlap relay。
>
> 本篇关注 **Scheduler ownership、跨轮时钟、KV watermark、forward transaction 与 overlap relay**。

</details>

## 一、Scheduler 与 Spec Worker 的分工

Scheduler 决定请求何时执行、预留多少空间；Spec Worker 决定本轮怎样提出和验证候选。两者维护的状态范围不同，需要分别看。

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

当 Scheduler 已经决定：`这一轮跑 DECODE`

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

所以同样一个：`ScheduleBatch.forward_mode = DECODE`

进入 Spec Worker 后，内部可能创建多个不同用途的 `ForwardBatch`：

```text
ScheduleBatch(DECODE)
        │
        ├─ Draft ForwardBatch
        ├─ TARGET_VERIFY ForwardBatch
        └─ DRAFT_EXTEND_V2 ForwardBatch
```

这两层的职责分别是：

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

其中：`DECODE`

主要是 Scheduler 层的一轮 decode transaction。

而：

```text
TARGET_VERIFY
DRAFT_EXTEND_V2
```

则是 Worker / ModelRunner 执行某个内部 forward 时的设备执行模式。

因此不要把：`ScheduleBatch.forward_mode == DECODE`

理解成：

> “这一轮 GPU 只做了一次普通 Decode。”

在 Spec V2 中完全不是。

## 二、CPU、设备与下一轮状态怎样错开

下面先列出参与跨轮交接的六个对象：

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

启用投机解码与 Overlap 后，几项状态会错开推进：

> **这些对象不再保证在任意瞬间都显示同一个 sequence length。**

这些长度分别服务于 CPU 结果处理、设备执行和下一轮交接，更新时刻也不同。检查状态时，需要先确认读到的是哪个阶段的进度。

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

注意它依赖：`req.output_ids`

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

`kv_committed_len` 的正常 Spec V2 owner 是：`SchedulerBatchResultProcessor._resolve_spec_v2_tokens()`

而不是 Draft Worker。

---

### `ScheduleBatch.seq_lens`：本轮 Device Forward Frontier

前两个长度描述请求记录，这里的长度用于本轮设备执行。Spec V2 的约定是：

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

因此在 overlap 下：`ScheduleBatch.seq_lens`

可以已经是：`Verify N 的新 frontier`

而 CPU 上：

```text
Req.seqlen
Req.kv.kv_committed_len
```

还没有处理完 Verify N 的结果。

---

### 允许错开“最多一个 Verify iteration”，不是“一个 Token”

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

如果一次 Verify 接受：`accept_lens = 4`

那么 overlap 窗口里完全可能出现：

```text
Device frontier:
ScheduleBatch.seq_lens = C + 4

Host ledger:
Req.kv.kv_committed_len = C
```

差距是：`4 tokens`

但仍然只是：`1 个尚未 CPU settle 的 verify iteration`

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

[`get_alloc_reserve_per_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6ee8bd98a376034a/python/sglang/srt/mem_cache/allocation_sizing.py)

源码直接写：

```python
def get_alloc_reserve_per_decode() -> int:
    """
    The 2x is a double-buffer that absorbs
    the kv_committed_len lag in overlap mode.
    """
    return 2 * get_alloc_len_per_decode()
```

这是非常漂亮的一条设计关系。

因为 Scheduler N+1 轮做 memory planning 时：`Host kv_committed_len`

可能还只到：`C`

但 Device 实际已经通过 Verify N 前进到了：`C + accepted_N`

如果 reserve 只按一轮最大 speculative footprint：`1 × alloc_len`

就可能因为 host committed clock 落后一轮而低估下一轮 workspace。

所以当前设计使用：`2 × alloc_len`

吸收：

```text
one-verify host lag
+
current speculative working set
```

---

### `Req.seqlen` 与 `kv_committed_len` 为什么经常差 1？

这是和前一篇 root / bonus frontier 文章直接连接的地方。

在一个已经稳定进入 speculative decode 的 request 上，常见状态是：

```text
KV-ready committed prefix
|======================|

                       B
                       ▲
                 pending terminal bonus
```

其中 `B`：`已经存在于 req.output_ids`

但它自己的 KV 还要在下一轮作为 root 被 materialize。

所以一个典型 steady-state 关系是：

```text
Req.seqlen
≈
kv_committed_len + 1
```

这一格就是 pending bonus。

例如：

```text
kv_committed_len = P

Req logical tokens:
[0 ... P-1] + bonus_at_P

Req.seqlen = P + 1
```

下一轮 Target Verify 把 bonus 当 root 写入位置 `P`。

如果本轮又接受 4 个输出：`D1 D2 D3 B2`

Device  materialize 的 Verify rows 是：`old_bonus D1 D2 D3`

也是 4 个。

所以 CPU settle 时：

```python
req.kv.kv_committed_len += 4
req.output_ids.extend([D1, D2, D3, B2])
```

结果仍然是：

```text
Req.seqlen
=
kv_committed_len + 1
```

新的那 `+1` 就是 `B2`。

> 上式对应这里的活跃投机请求：已确认输出比已物化 KV 多一枚待输入的 bonus。进入 Transition、Mixed 或 Finished 状态后，再按各自处理阶段读取长度。

## 三、预留 KV 空间并保留提交边界

当 running batch 准备进入下一轮 decode 时：

```python
batch.prepare_for_decode()
```

普通 Decode 和 Speculative Decode 从这里分叉。

普通路径：

```text
alloc 1 token
   ↓
写 req_to_token
   ↓
kv_allocated_len += 1
kv_committed_len += 1
seq_lens += 1
```

Speculative 路径则直接交给：

```python
spec_prepare_for_decode(batch)
```

固定源码：

[`ScheduleBatch.prepare_for_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/schedule_batch.py)

---

### EAGLE 做的是 speculative reserve

`spec_prepare_for_decode()` 对 EAGLE 最终调用：

```python
eagle_prepare_for_decode(batch)
```

其中：

```python
reserve = get_alloc_reserve_per_decode()

cur_kv_lens, nxt_kv_lens, num_needed_tokens =
    page_aligned_decode_alloc_lens(
        batch.reqs,
        reserve=reserve,
        ...
    )
```

随后：

```python
alloc_for_spec_decode(...)
```

固定源码：

[`eagle_prepare_for_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/eagle_utils.py)

以及：

[`alloc_for_spec_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/mem_cache/allocation.py)

更新的是：

```python
req.kv.kv_allocated_len =
    max(req.kv.kv_allocated_len, nxt_kv_len)
```

没有：`kv_committed_len += speculative width`

```text
Prepare
=
Reserve Future Capacity
```

而不是：

```text
Prepare
=
Declare Future Correct
```

---

### 两个 watermark 的职责完全不同

`kv_allocated_len`

回答：

> **这一轮 speculative working set 最多可以写到哪里？**

而：`kv_committed_len`

回答：

> **Target 已经正式确认到哪里？**

因此：`allocation`

发生在 Draft / Verify 之前。

而：`host commit`

必须等 Accept 结果 settle 以后。

这就是 speculative memory bookkeeping 最核心的安全边界。

---

### 一个典型状态

```text
0
│
├──────── committed ────────┤──────── reserve ────────┤
│                           │                          │
                            C                          A
                            ▲                          ▲
                    kv_committed_len          kv_allocated_len
```

Draft / Verify 可以在：`[C, A)`

里临时写 future。

但只有 Accept 决定的那一段最终会成为下一轮 committed prefix。

## 四、on_publish 发布哪一部分进度

现在进入一轮运行。

Scheduler 主循环大致是：

```text
get_next_batch_to_run()
        ↓
prepare_for_decode()
        ↓
run_batch()
        ↓
process_batch_result()
```

开启 Spec V2 后：

```text
run_batch()
   ↓
self.model_worker.forward_batch_generation()
```

这里的 `model_worker` 已经是 speculative orchestrator。

---

### EAGLE：一次 Scheduler DECODE 包含 Draft → Verify → Accept → Draft Extend

Decode 分支：

```text
ScheduleBatch(DECODE)
        ↓
draft()
        ↓
EagleVerifyInput
        ↓
verify()
        ↓
eagle_sample()
        ↓
accept_lens / accept_index / new_seq_lens
        ↓
_draft_extend_for_decode()
        ↓
GenerationBatchResult
```

Verify 内部会临时把 forward 变成：`TARGET_VERIFY`

并创建 Target Model 用的 `ForwardBatch`。

```text
Scheduler Macro State:
DECODE

Worker Micro State:
DRAFT
→ TARGET_VERIFY
→ ACCEPT
→ DRAFT_EXTEND
```

Scheduler 管理请求与批次，Worker 在一次执行内推进 Draft、Verify 与 Accept；两者通过返回结果交接状态。

---

### DSpark：内部算法完全不同，但 Scheduler Contract 一样

DSpark `_forward_decode()`：

```text
Proposal
  ↓
Confidence
  ↓
Verify Budget
  ↓
schedule_layout()
  ↓
Target Verify
  ↓
accept_and_finalize()
  ↓
commit_hidden / Mamba state
```

最后仍然返回：

```python
GenerationBatchResult(
    next_token_ids=...,
    accept_lens=...,
    next_draft_input=...,
    new_seq_lens=...,
)
```

这就是 Scheduler abstraction 的价值。

Scheduler 不需要知道：`EAGLE Tree 是怎么长的`

也不需要知道：`DSpark Planner 怎么决定 verify token budget`

它只要求算法最后回答：

```text
本轮接受了什么？
下一轮 frontier 在哪？
下一轮 draft state 是什么？
```

---

### 所有 `on_publish(new_seq_lens)` 的语义边界是一致的，但物理位置不完全一致

下面对比各算法的 publish 位置。

#### EAGLE / Multi-layer EAGLE / Frozen-KV MTP

它们都是：

```text
Target Verify
    ↓
Accept / new_seq_lens known
    ↓
on_publish(new_seq_lens)
    ↓
Draft Extend
```

源码注释甚至直接写：

```text
Publish before draft_extend
so the fence is at verify-end.
```

因此 publish 边界是：

> **Verify / Accept 已经确定 next frontier。**

不是：

> Draft side 已经全部追平。

---

#### DSpark

DSpark 更能说明这个 distinction。

顺序是：

```text
accept_and_finalize()
        ↓
on_publish(accept.new_seq_lens)
        ↓
_commit_target_mamba_states_after_verify()
        ↓
commit_hidden()
```

也就是说：

> **publish 甚至发生在部分 target-side post-verify state commit 之前。**

固定源码：

[`DSparkWorkerV2._forward_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

`on_publish` 将下一轮所需的长度先交给调度链路，部分 Device Commit 则继续按原依赖执行。

---

#### DFLASH

DFLASH 又稍微不同：

```text
Verify / Accept
      ↓
必要时更新 target Mamba state
      ↓
on_publish(new_seq_lens)
      ↓
把 accepted target hidden materialize 到 draft KV
```

因此 publish 仍然位于：`next frontier 已经确定`

和：`全部后处理都做完`

之间。

---

#### NGRAM / UNO

NGRAM 在 accepted target KV move 完成后 publish，然后再更新 host-side ngram corpus。

UNO 在 acceptance 算出 `new_seq_lens` 后直接 publish。

它们虽然具体 placement 不同，但共同 contract 一样：

> **`on_publish(new_seq_lens)` 表示“下一轮 Scheduler 已经可以安全知道 sequence frontier 是多少”，而不是“当前 Worker 已经完全 return”。**

---

### 为什么这样仍然安全？

因为 publish 主要允许：`schedule_stream`

提前做下一轮 host / allocation planning。

它并不会让下一次 model forward 穿越当前 forward stream 中尚未完成的：

```text
Draft Extend
Mamba commit
hidden materialization
```

这些操作仍然按 forward stream 的程序顺序执行。

所以时序可以是：

```text
Forward Stream:
Verify N
   ↓
Accept N
   ↓
publish(frontier N)
   ↓
Draft/State Commit N
   ↓
Forward N+1


Schedule Stream:
             publish fence
                   ↓
          Prepare Schedule N+1
```

 overlap 的是：`CPU / schedule preparation`

和：`当前 Worker 剩余 device work`

而不是让两个依赖状态的 model forward 乱序。

---

### Prefill 的 publish 也遵守同一个思想

EAGLE / DFLASH / DSpark prefill 路径中：

```text
Target Prefill
      ↓
new_seq_lens = batch.seq_lens
      ↓
publish()
      ↓
Draft prefill / hidden injection
```

为什么这里没有 `+1`？

因为 Spec V2 convention 下：

```text
batch.seq_lens
=
当前已经 KV-ready 的 prompt boundary
```

Target Prefill 采样出来的 next token 只是：`下一轮 pending root / bonus`

它还不属于 KV-ready prefix。

所以下一轮 Verify 的 frontier 仍然就是：`batch.seq_lens`

这和 Decode 中 terminal bonus 的 frontier 语义完全一致。

## 五、隔离临时字段与保护共享对象

这里要区分字段替换和共享对象修改。

Spec Worker 在一轮内部会临时修改大量 `ScheduleBatch` 字段：

```text
forward_mode

input_ids

seq_lens
seq_lens_cpu
seq_lens_sum

spec_info

out_cache_loc

sampling_info
...
```

例如：

```text
DECODE
  ↓
TARGET_VERIFY
```

或者临时：`batch.spec_info = EagleVerifyInput`

如果这些 worker-internal mutation 泄漏回 Scheduler，下一轮就可能被错误地识别为：

```text
EXTEND
TARGET_VERIFY
```

甚至造成重复 merge / 错误 allocation。

---

### `_forward_isolation()` 恢复字段绑定，不深拷贝整个 Batch

`Scheduler._forward_isolation()` 当前实现：

```python
snapshot_v2_full = not batch.spec_algorithm.is_none()

sched_snapshot = {
    f.name: getattr(batch, f.name)
    for f in dataclasses.fields(batch)
}
```

退出时：

```python
for name, value in sched_snapshot.items():
    setattr(batch, name, value)
```

固定源码：

[`Scheduler._forward_isolation()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler.py)

这里保存的是每个字段原来指向的对象，退出时再用 `setattr` 恢复。它是一份浅快照：

| Worker 内的操作 | 退出 isolation 后 |
| --- | --- |
| `batch.seq_lens = new_tensor` | 字段重新指向原 Tensor |
| `batch.seq_lens.add_(1)` | 原 Tensor 的内容已改变，恢复引用不能撤销它 |
| 修改字段所引用的列表或嵌套对象 | 不会因 `setattr` 自动恢复内部内容 |

隔离区保存并恢复 dataclass 字段的绑定。共享 Tensor、列表和嵌套对象的内容则由各自的拷贝与同步方式保护，下文 `sampling_info.copy_for_forward()` 就负责其中一部分。

这条边界隔离的是临时字段替换。

---

### `sampling_info` 为什么还要额外换成 forward-only copy？

Isolation 进入前会：

```python
batch.sampling_info =
    sched_sampling_info.copy_for_forward()
```

原因是一个 Spec iteration 内部可能多次：`ForwardBatch.init_new()`

如果直接复用 Scheduler 的 sampling state，penalty 等状态可能被重复 accumulate。

因此：`Scheduler sampling state`

和：`forward-local sampling view`

也被隔离开了。

---

### Overlap 下还要保活两轮 tensor reference

`record_batch_in_overlap()` 会把：

```text
batch
+
attr_snapshot
```

保存在两槽 ring 中。

原因不是状态语义，而是 tensor lifetime。

Spec Worker 可能：

```text
rebind batch.input_ids
rebind batch.out_cache_loc
rebind batch.spec_info
```

旧 tensor Python 引用如果提前丢掉，Caching Allocator 可能在 forward stream 还没读完时复用那块显存。

所以：`semantic rollback`

和：`GPU tensor lifetime`

这里同时被 isolation 管住。

---

### 那么哪些字段最终必须显式 re-commit？

关键就在这里。

Isolation 的设计是：`先恢复快照中的字段绑定`

然后：`只重新提交属于 next iteration contract 的状态`

#### Overlap 路径

Worker 返回以后：

```python
batch.input_ids = None

batch.spec_info =
    batch_result.next_draft_input

batch.spec_info.future_indices =
    future_indices
```

但：`batch.seq_lens`

不会直接在这里赋成 `new_seq_lens`。

因为它已经通过：`FutureMap.publish(new_seq_lens)`

进入 device relay。

下一轮：

```python
future_map.resolve_seq_lens_cpu(batch)
```

再把 fresh frontier resolve 回来。

所以 overlap 的显式 commit 是：

```text
ScheduleBatch object:
next_draft_input / relay handle

FutureMap:
new_seq_lens
bonus/topk/hidden...
```

---

#### Non-overlap 路径

没有 FutureMap overlap relay，所以 isolation 结束后 Scheduler 必须显式：

```python
batch.spec_info = batch_result.next_draft_input

batch.seq_lens = batch_result.new_seq_lens

batch.seq_lens_cpu =
    batch_result.new_seq_lens.to("cpu")

batch.seq_lens_sum =
    int(batch.seq_lens_cpu.sum())

batch.input_ids = None
```

这就是同步路径中的 cross-iteration commit。

---

### 哪些字段必须保持 rollback？

最典型的有：`forward_mode`

Scheduler 外部仍然应该看到：`DECODE`

而不是 Worker 内部临时的：

```text
TARGET_VERIFY
DRAFT_EXTEND_V2
```

还有：

```text
verify-time input_ids
verify-time out_cache_loc
temporary EagleVerifyInput
temporary verify seq_lens
```

这些临时字段绑定随当前 Forward 结束而恢复，下一轮输入由返回结果重新接入。

---

### 字段隔离与显式提交的边界

Worker 执行时可以临时替换本轮 `forward_mode`、`out_cache_loc`、`spec_info` 或 `seq_lens` 的字段绑定；退出隔离区后，再从返回结果显式接入下一轮所需状态。

| 对象变化 | 隔离区退出后的处理 |
| --- | --- |
| `ScheduleBatch` 字段被重新绑定 | 恢复进入时保存的字段值或对象引用 |
| 原 Tensor 或嵌套对象被原地修改 | 浅快照不会自动撤销，需要调用方单独保护 |
| 下一轮 Draft 输入、长度与 bonus | 通过对应结果和交接路径显式发布或提交 |

因此，调试时既要检查字段最终指向谁，也要检查共享对象是否发生原地写入。只比较退出前后的字段绑定，无法覆盖第二类问题。

普通投机 Decode 的状态交接已经在前五节说明。Mixed Tail 还要把待处理的边界 Token 作为 EXTEND 输入重建，下面单独展开这条路径。

## 六、进阶：Mixed Tail 的 FutureMap 重建

<details markdown="1">
<summary>展开 Mixed Tail 的字段变化与 +1 来源</summary>

局部代码看起来的偏移，需要放回 mixed-tail 的执行语境。

`FutureMap.resolve_mixed_spec_tails()` 中有：

```python
fresh = self.new_seq_lens_buf[idx]

seq_lens = batch.seq_lens.clone()
seq_lens[-n:] = fresh + 1
batch.seq_lens = seq_lens
```

同时：

```python
out_cache_loc[-n:] =
    req_to_token[idx, fresh]
```

这里为什么使用 `new_seq_lens + 1`？

前面明明一直说：

```text
new_seq_lens
=
next KV-ready frontier
```

---

### 先看 Mixed Chunk 在做什么

当 chunked prefill 开启 mixed mode 时：

```text
new prefill requests
+
running decode requests
```

会被塞进同一个：`ForwardMode.MIXED`

batch。

running decode request 此时不是按 speculative Verify 的形态运行，而是被临时改写成：`1-token EXTEND tail`

`mix_with_running()` 明确写：`a tail's prefix is its row length - 1`

Spec 路径还写：`Spec rows sit at the committed base`

---

### 对 speculative request 来说，那个“1 token tail”是什么？

就是当前 pending bonus。

假设 FutureMap fresh frontier 是：`C`

代表：`[0, C)`

已经 KV-ready。

而：`bonus`

位于：`position C`

还没有 materialize KV。

Mixed EXTEND 要做的正是：

```text
prefix:
[0, C)

extend token:
bonus at C
```

所以 Extend 语义下的 full sequence length 必须是：`C + 1`

于是：

```python
seq_lens = fresh + 1
```

完全正确。

与此同时：

```python
out_cache_loc =
    req_to_token[idx, fresh]
```

就是 bonus 在位置 `C` 的写入 slot。

可以画成：

```text
FutureMap new_seq_lens = C

KV-ready:
|====================|
0                    C

                     B
                     ▲
               pending bonus


Mixed 1-token EXTEND:

prefix_len = C

input = B

write position = C

full seq_len = C + 1
```

这里将同一个 KV-ready Frontier 转换为单 Token EXTEND 写入后的长度，因此使用 `C+1`。

---

### 为什么 Result Processor 还要专门 `kv_committed_len += 1`？

因为这次 Mixed EXTEND 真的把 pending bonus 的 KV materialize 了。

`process_batch_result_prefill()` 有唯一的特殊 owner：

```python
if (
    not req.finished()
    and batch.decoding_reqs
    and req in batch.decoding_reqs
    and not batch.spec_algorithm.is_none()
):
    req.kv.kv_committed_len += 1
```

源码注释：

```text
A mixed spec tail committed its pending bonus token;
advance so the next spec prepare_for_decode
reserves from the right base.
```

固定源码：

[`SchedulerBatchResultProcessor.process_batch_result_prefill()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler_components/batch_result_processor.py)

这再次证明：`+1`

是 “pending bonus 被当前 EXTEND  commit” 的局部语义。

---

### 为什么不是所有 FutureMap resolve 都 `+1`？

普通 speculative decode 下一轮仍然要把 pending bonus 当：`Verify root`

它需要的 prefix frontier 就是：`C`

而不是：`C + 1`

如果你提前把：`ScheduleBatch.seq_lens = C + 1`

那等于宣称 bonus 的 KV 已经存在。

但实际上它还没 forward。

所以普通 next-spec iteration 必须：`seq_lens = new_seq_lens`

只有：`Mixed batch 把 bonus 直接变成一枚 EXTEND input`

时才需要：`new_seq_lens + 1`

---

### 这个 `+1` 也不是 DP Attention 的通用规则

这里还要区分 Mixed Tail 与普通 DP Attention 的处理。

当前 Scheduler 在：`spec + DP Attention`

下明确写：

```text
make sure prefill and decode batches
will not be mixed
```

在 `get_next_batch_to_run()` 中，Spec + DP-attention 会在 merge 前先协调 prefill/decode mode，避免把它们走成普通 mixed chunk。

固定源码：

[`Scheduler.get_next_batch_to_run()` 附近](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler.py)

DP Attention 还有另一条路径：

```python
maybe_convert_decode_to_extend()
```

当别的 DP rank 正在跑 extend、而当前 rank 是 decode 时，为了保持 mode-homogeneous、复用 extend graph，它会把整个 decode batch 临时转换成：`1-token EXTEND view`

对应：

[`SchedulerDPAttnAdapter.maybe_convert_decode_to_extend()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler_components/dp_attn.py)

这条路径调用：

```python
batch.convert_decode_to_extend()
```

它直接把当前 prepared `seq_lens` 解释成 full sequence length：

```text
prefix = seq_len - 1
extend_len = 1
```

它不是：`FutureMap.resolve_mixed_spec_tails()`

因此更准确的结论是：

> **`new_seq_lens + 1` 是“overlap mixed-chunk speculative tail late-binding”的局部转换规则，不是 FutureMap 的通用 seq_len 定义，也不是 DP Attention 的通用规则。**

---

### 为什么要 late-bind？

因为 mixed batch 在 schedule time 构造时：`CPU Req clock`

可能还落后于 GPU 最新 verify。

所以 `mix_with_running()` 只能先拿 host request state 做一个 provisional tail。

到了 forward entry：

```python
resolve_forward_inputs()
```

才有资格等待 `publish_ready` 并读取：`FutureMap.new_seq_lens_buf`

于是重新绑定：

```text
fresh prefix
fresh out_cache_loc
fresh seq_lens
```

这就是：`late binding`

存在的原因。

</details>

## 七、把跨轮状态交还 Scheduler

现在可以把全文压缩成一张图。

```mermaid
flowchart TD
    A[Req / ReqKvInfo] --> B[ScheduleBatch.prepare_for_decode]

    B --> C{Speculative?}

    C -->|No| D[alloc_for_decode: one token]
    C -->|Yes| E[spec_prepare_for_decode]

    E --> F[Reserve speculative KV capacity]
    F --> G[ScheduleBatch DECODE]

    G --> H[Spec Worker]

    H --> I[Draft / Proposal]
    I --> J[Target Verify]
    J --> K[Accept]

    K --> L[new_seq_lens]
    K --> M[accept_lens / accepted output]
    K --> N[next bonus / next draft state]

    L --> O[FutureMap.publish]
    N --> P[Draft / post-verify state commit]
    P --> Q[GenerationBatchResult]

    Q --> R[FutureMap.stash]
    Q --> S[BatchResultProcessor]

    S --> T[req.output_ids += retained accepted run]
    S --> U[req.kv.kv_committed_len += retained length]
    S --> V[finish / grammar / metrics]

    O --> W[Next scheduling iteration]
    R --> W

    W --> X[resolve_seq_lens]
    W --> Y[resolve_forward_inputs]

    X --> B
    Y --> B
```

这张图里存在三类“时间”。

### Past：CPU 已经 settle 的事实

```text
Req.output_ids

Req.kv.kv_committed_len
```

这是：`Committed Past`

---

### Present：这一轮 Scheduler Transaction

```text
ScheduleBatch

ForwardBatch

temporary TARGET_VERIFY / DRAFT_EXTEND states
```

这是：`Executing Present`

---

### Future：CPU 还没 settle，但下一轮 GPU 已经可以依赖的结果

```text
FutureMap.new_seq_lens

FutureMap.output_tokens_buf

next_draft_input

topk / hidden / confidence ...
```

这是：`Safe Future`

Spec V2 Scheduler 的核心价值，就是允许：

```text
Past
Present
Future
```

短暂不同步，但每一个 owner 都只能修改自己的那只时钟。

---

### 四个跨轮状态边界

**第一，Overlap 下 `ScheduleBatch.seq_lens` 可以领先 `Req.kv.kv_committed_len` 最多一个 Verify iteration，而不是一个 token。**

一次 Verify 可能接受多个 token，所以数值差可以大于 1。2x speculative reserve 正是为了吸收这一轮 host committed lag。

**第二，`on_publish(new_seq_lens)` 的统一语义是“next forward frontier 已经确定”，不是“所有 device-side post-verify commit 已结束”。**

不同算法 publish 后仍可能继续 Draft Extend、Mamba commit、hidden → draft KV materialization。

**第三，`_forward_isolation()` 对 Spec V2 保存并恢复全部 dataclass 字段的引用。**

字段替换在退出时恢复，需跨轮保留的新绑定通过 `GenerationBatchResult / FutureMap / explicit recommit` 交接。Tensor 原地修改和嵌套对象修改不在这份浅快照的撤销能力之内。

**第四，`new_seq_lens + 1` 只属于 mixed speculative tail 的 1-token EXTEND 语义。**

`new_seq_lens` 本身仍是 KV-ready base；`+1` 表示当前 mixed forward 正在把 base 上的 pending bonus 作为一枚 extend token materialize。它既不是普通 spec decode 规则，也不是 DP Attention 的统一规则。

---

### Debug Scheduler 时建议同时打印这些字段

Debug Spec V2 Scheduler 时，可以把下列字段按请求与轮次并排记录：

```text
rid

len(origin_input_ids)
len(output_ids)
Req.seqlen

req.kv.cache_protected_len
req.kv.kv_committed_len
req.kv.kv_allocated_len

batch.forward_iter
batch.forward_mode
batch.seq_lens
batch.seq_lens_cpu

type(batch.spec_info)
future_indices

result.accept_lens
result.new_seq_lens

FutureMap.new_seq_lens_buf[req_pool_idx]
FutureMap.output_tokens_buf[req_pool_idx]
```

然后先问四个问题：

```text
1. 现在看的是 Host clock 还是 Device clock？

2. 这个 field 的 owner 是 Scheduler、
   Worker、ResultProcessor 还是 FutureMap？

3. 这个值代表 committed boundary，
   allocated boundary，还是 pending bonus？

4. 当前是在普通 Decode、
   TARGET_VERIFY、
   Mixed 1-token Extend，
   还是 DP decode→extend view？
```

只要这四个问题先回答，绝大多数“为什么差 1 / 为什么多一轮 / 为什么 seq_len 不一样”的问题都会变得清楚。

## 源码阅读入口

固定版本：

```text
sgl-project/sglang
5f017ffabb6ab8d214f6a4616ee8bd98a376034a
```

推荐按下面顺序阅读：

| 目标 | 固定版本源码 |
| --- | --- |
| `Req / ReqKvInfo / ScheduleBatch` | [`schedule_batch.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/schedule_batch.py) |
| `ForwardMode / ForwardBatch.init_new()` | [`forward_batch_info.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/model_executor/forward_batch_info.py) |
| Scheduler Event Loop / `run_batch()` | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler.py) |
| Spec Worker dispatch | [`Scheduler.init_model_worker()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler.py) |
| `_forward_isolation()` | [`scheduler.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler.py) |
| Spec Decode Preparation | [`spec_utils.py::spec_prepare_for_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/spec_utils.py) |
| EAGLE reserve | [`eagle_utils.py::eagle_prepare_for_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/eagle_utils.py) |
| Spec KV sizing / double reserve | [`allocation_sizing.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/mem_cache/allocation_sizing.py) |
| Spec KV allocation | [`allocation.py::alloc_for_spec_decode()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/mem_cache/allocation.py) |
| EAGLE V2 micro-state machine | [`eagle_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/eagle_worker_v2.py) |
| Shared EAGLE Verify | [`eagle_worker_common.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/eagle_worker_common.py) |
| DSpark decode state machine | [`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) |
| DFLASH publish / draft-KV materialization | [`dflash_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/dflash_worker_v2.py) |
| UNO publish contract | [`uno_worker_v2.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/uno_worker_v2.py) |
| NGRAM publish contract | [`ngram_worker.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/speculative/ngram_worker.py) |
| `GenerationBatchResult` | [`managers/utils.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/utils.py) |
| Spec host commit | [`batch_result_processor.py::_resolve_spec_v2_tokens()`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler_components/batch_result_processor.py) |
| Mixed spec tail commit | [`batch_result_processor.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler_components/batch_result_processor.py) |
| `FutureMap.publish / stash / resolve` | [`overlap_utils.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/overlap_utils.py) |
| DP decode→extend conversion | [`dp_attn.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/python/sglang/srt/managers/scheduler_components/dp_attn.py) |
| Bookkeeping ownership regression test | [`test_decode_bookkeeping_ownership.py`](https://github.com/sgl-project/sglang/blob/5f017ffabb6ab8d214f6a4616ee8bd98a376034a/test/registered/unit/spec/test_decode_bookkeeping_ownership.py) |

阅读跨轮日志时，先标明每个长度属于 CPU 请求记录、设备执行进度，还是尚未由 Host 结算的下一轮结果。再沿 publish、resolve 和结果处理检查它何时被消费。这样既允许准备与执行重叠，也能判断某次长度差异是预期时差，还是漏提交、重复提交或共享对象被意外修改。
