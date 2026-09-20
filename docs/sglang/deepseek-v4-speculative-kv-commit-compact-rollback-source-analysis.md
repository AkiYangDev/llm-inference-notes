# DeepSeek-V4 Speculative Decoding 源码解析：Draft Tree 如何落到 KV Cache，Accept 后缓存如何 Commit、Compact 与 Rollback？

前面的文章已经把 DeepSeek-V4 speculative decoding 拆到了 Draft、MTP/NextN、EAGLE Tree、多卡一致性与 Target Verify。再往下一层，真正决定一轮 speculative decoding 能不能正确进入下一轮的，不再只是“接受了几个 Token”，而是：

> **Target Verify 期间产生的候选 KV，怎样从 speculative layout 收敛成正式 sequence state？Reject 的候选什么时候失去逻辑所有权，什么时候又真的归还物理显存？**

这件事看起来像一个 KV Cache 问题，实际上同时跨过了四层状态：

~~~text
token / accepted output
        ↓
tree / verify-row index
        ↓
logical sequence position
        ↓
physical KV slot
~~~

如果把这些索引空间混在一起，就很容易把 `accept_index` 当成 KV 地址、把 `accept_lens - 1` 理解成“bonus 不提交”，或者把 reject 想成“立刻 free 那几块 KV”。

本文把这条链一次追到底。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 6880a4795533640f41ebb3db9e4ae0af5a371a1f`，核对日期为 2026-09-20。主线覆盖 Spec V2 的 EAGLE accepted-path KV compaction、经典 rejection sampling、Target-only sampling，以及 Ascend DeepSeek-V4 的 C4/C128 KV/state 生命周期。当前 DeepSeek-V4 + DSpark 仍是 `topk=1` 的 linear chain；因此文中的 tree-specific `accept_index` compaction 主要解释通用 EAGLE Spec V2，DSpark 部分会单独标出。

## Review 结论先行

这次严格 Review 重点核了四个容易写错的点，结论如下。

| Review 点 | 固定源码结论 |
| --- | --- |
| `move_kv_cache()` 的 src/dst 方向 | API 是 `move_kv_cache(tgt_loc, src_loc)`，实际语义是 **`dst <- src`**。EAGLE commit 时 `accept_out_cache_loc` 是 source candidate KV，`tgt_cache_loc` 是 destination committed KV。 |
| `accept_index` 是否包含 bonus | **有效前缀长度等于 `accept_lens`，最后一个有效 index 承载 bonus。** 但 bonus 不是额外创建一个 tree node；它写入“最后一个 accepted verify row”的 `predict` 槽位。 |
| Spec V2 overshoot 什么时候真正 free | 正常 reject 后**不会每轮立即 free**。allocated-but-uncommitted tail 通常留作 reserve/reuse；真正回收发生在 request release/finish/abort、decode retract/preempt，以及 StreamingSession 的 trim/rewind 等生命周期点。 |
| Ascend DSV4 C4/C128 cleanup | 必须区分 **KV page 回收** 与 **compressor state 清理**。`clear_unaccepted_c128_draft_states()` 清的是 C128 state ring，不是释放 C128 KV pages；C4 state 跟随 SWA physical pages，不能套用同一个 request-scoped cleanup 模型。 |

这四个结论会贯穿全文。

---

## 一、先把四个地址空间分开：Token、Tree Row、Logical Position、Physical KV Slot

普通 Decode 很容易形成一个过度简化的心智模型：

~~~text
第 4096 个 token
        ↓
KV Cache 第 4096 行
~~~

真实 SGLang 并不是这样。

### 1. Logical Sequence Position

假设当前请求已经提交：

~~~text
seq_len = 4096
~~~

那么下一段正式 continuation 的逻辑位置是：

~~~text
4096, 4097, 4098, ...
~~~

这是 request 语义上的 sequence position。

### 2. `req_to_token`：Logical Position → Physical KV Slot

SGLang 用 request-to-token table 把逻辑位置映射到真实 KV pool slot：

~~~text
request #7

logical pos     physical KV slot
-----------     ----------------
4094        ->  18320
4095        ->  18321
4096        ->  42080
4097        ->  42081
4098        ->  42082
~~~

因此：

~~~python
req_to_token[req_pool_idx, logical_position]
~~~

得到的是物理 KV slot，而不是 token id。

这也是 Paged KV 能做到：

~~~text
logical sequence 连续
≠
physical memory 连续
~~~

的基础。

### 3. `out_cache_loc`：这次 Forward 的 K/V 应该写到哪里

Target Verify 一次会处理多个 speculative candidate。模型 Forward 产生新的 K/V 后，需要知道每一行应该写到哪个 physical slot。

这就是：

~~~python
batch.out_cache_loc
~~~

的职责。

EAGLE 在 `eagle_prepare_for_verify()` 中把：

~~~python
batch.input_ids = verify_input.draft_token

batch.out_cache_loc = assign_extend_cache_locs_uniform_func(
    req_pool_indices=batch.req_pool_indices,
    req_to_token=req_to_token_pool.req_to_token,
    start_offset=batch.seq_lens,
    batch_size=bs,
    draft_token_num=verify_input.draft_token_num,
    device=device,
)
~~~

绑定到 Target Verify。

所以 Verify Forward 并不是只生成 logits：

~~~text
Draft candidates
      │
      ▼
Target Model
      │
      ├── logits
      ├── hidden states
      └── candidate KV
               │
               ▼
         out_cache_loc
~~~

候选是否最终被接受，是 Forward **之后** 才决定的。

### 4. `accept_index`：不是 KV slot，而是 accepted verify-row path

当 `topk > 1` 时，candidate 是 tree：

~~~text
                 root
              /        \
             A          B
           /   \      /   \
          C     D    E     F
~~~

如果最终 accepted path 是：

~~~text
root → B → E
~~~

不能简单地“保留 candidate buffer 前 3 行”，因为 tree row 未必连续。

`accept_index` 解决的是：

~~~text
tree / verify-row space
        ↓
accepted path
~~~

而不是：

~~~text
physical KV address
~~~

真正的链路是：

~~~text
accept_index
    │
    ▼
verify row
    │
    ▼
out_cache_loc[row]
    │
    ▼
source physical KV slot
~~~

再通过 committed logical positions 找到 destination slot：

~~~text
logical committed continuation
        │
        ▼
req_to_token
        │
        ▼
destination physical KV slot
~~~

所以整篇最重要的第一条规则是：

> **不要把 `accept_index`、logical token position、`out_cache_loc` 和 physical KV slot 当成同一个索引空间。**

固定源码入口：

- [`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/eagle_utils.py)
- [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

---

## 二、Accept 的真实语义：`accept_index` 最后一格为什么是 bonus，但又不是一个新的 Tree Node

这是本次 Review 最值得讲清楚的地方。

很多文章会把 accepted output 画成：

~~~text
draft1
draft2
draft3
bonus
~~~

然后自然猜测：

~~~text
accept_index
=
[draft1_node, draft2_node, draft3_node, bonus_node]
~~~

这个说法不够准确。

### 1. Greedy Tree Verify 的真实写法

`verify_tree_greedy_kernel_triton()` 一开始会把 root 对应的 retrieve index 写到：

~~~text
accept_index[0]
~~~

然后每接受一个 draft child：

1. 把这个 draft token 写到“上一个 accepted row”的 `predicts`；
2. 把 child 的 tree/retrieve row index 追加进 `accept_index`；
3. 把这个 child row 设为新的 `last_accepted_global_idx`。

伪代码可以压成：

~~~python
accept_index[0] = root_row
last_row = root_row

for accepted_draft in path:
    predicts[last_row] = accepted_draft
    accept_index[next] = child_row
    last_row = child_row

predicts[last_row] = bonus
~~~

因此如果正确 draft 是：

~~~text
d1, d2, d3
~~~

最终不是：

~~~text
accept_index = [d1_node, d2_node, d3_node, bonus_node]
~~~

而更准确是：

~~~text
accept_index valid prefix
=
[root_row, row_after_d1, row_after_d2, row_after_d3]
~~~

与此同时：

~~~text
predict[accept_index[:4]]
=
[d1, d2, d3, bonus]
~~~

也就是说：

> **bonus 占据 accepted output 的最后一格，但它复用了“最后一个 accepted verify row”作为承载位置，并没有额外创造一个 bonus tree node。**

### 2. `accept_lens` 为什么等于 correct drafts + 1

`eagle_sample()` 返回前明确保持：

~~~python
num_correct_drafts
~~~

只统计 draft，然后返回：

~~~python
return predict, num_correct_drafts + 1, accept_index
~~~

所以：

~~~text
accept_lens
=
num_correct_drafts + 1
~~~

其中 `+1` 就是 terminal target/bonus token。

更重要的是，`fill_bonus_tokens` 的 kernel 明确使用：

~~~text
bonus_token_idx
=
accept_stride * pid
+
accept_len
-
1
~~~

也就是：

> **`accept_index` 的有效前缀长度就是 `accept_lens`，最后一个有效位置对应的 `predict` 值就是 bonus。**

### 3. Rejection Sampling 也保持相同契约

经典 rejection sampling 的 Triton kernel 同样：

~~~text
accept_index[0] = root_row
~~~

每接受一个 draft：

~~~text
append current row into accept_index
update last_accepted_global_idx
~~~

最后 residual / target sampling 得到 terminal token，再写到：

~~~python
Predicts[last_accepted_global_idx] = final_token
~~~

因此它与 Greedy 路径保持同一输出 contract：

~~~text
predict[accept_index[:accept_lens]]
=
accepted drafts + terminal token
~~~

### 4. Simulated accept 也遵循同一个布局

`generate_simulated_accept_index()` 会：

~~~python
sim_accept_index[:, :simulate_acc_len] = ...
num_correct_drafts.fill_(simulate_acc_len - 1)
~~~

real-draft-token 模式下，最后又显式：

~~~python
bonus_node_indices = sim_accept_index[:, simulate_acc_len - 1]
predict[bonus_node_indices] = target_predict[...]
~~~

所以 simulated path 没有另外定义一套语义。

### 5. NPU 分支的证据边界

非 Greedy NPU path 会从：

~~~text
sgl_kernel_npu.sample
~~~

导入 `tree_speculative_sampling_target_only` / `chain_speculative_sampling_triton`。

固定 SGLang 仓库中可以直接审计它的调用 contract 与 downstream invariants，但 `sgl_kernel_npu` 的具体 kernel 实现在这一仓库之外。

因此本文不做超出证据范围的表述，例如：

~~~text
“NPU 内核内部一定逐行执行了与 CUDA 完全相同的语句”
~~~

能确认的是：

- SGLang 给 NPU kernel 传入同样的 `predict / accept_index / accept_token_num` contract；
- 随后的 TP broadcast 同步这三项；
- downstream `fill_bonus_tokens` 与 KV compaction 仍按“有效 `accept_index` 前缀长度 = `accept_lens`”消费它们。

这足以确定 SGLang 层面对 NPU sampling output 的契约。

固定源码：

- [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/spec_tree.py)
- [`chain_speculative_sampling_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/reject_sampling.py)
- [`fill_bonus_tokens`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/eagle.py)
- [`generate_simulated_accept_index()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

---

## 三、KV Commit：`move_kv_cache(tgt, src)` 到底谁搬到谁，以及 `accept_lens - 1` 为什么没有丢掉 bonus

Target Verify 得到：

~~~python
predict, accept_lens, accept_index
~~~

之后，tree path finalize 会进入：

~~~python
move_accept_tokens_to_target_kvcache(
    batch,
    accept_index,
    accept_lens - 1,
    token_to_kv_pool_allocator,
)
~~~

这里有两个非常容易误读的点。

### 1. `accept_lens - 1` 不是“bonus 不进 KV”

被调函数的第三个参数叫：

~~~python
num_correct_drafts
~~~

而不是 `accept_lens`。

调用处做的是：

~~~text
accept_lens
    │
    │ - 1
    ▼
num_correct_drafts
~~~

函数内部创建 committed destination range 时又使用：

~~~python
batch.seq_lens + num_correct_drafts + 1
~~~

所以最终长度还是：

~~~text
num_correct_drafts + 1
=
accept_lens
~~~

例如：

~~~text
correct drafts = 3
bonus          = 1

accept_lens    = 4
~~~

调用处：

~~~text
accept_lens - 1 = 3
~~~

函数内部：

~~~text
3 + 1 = 4
~~~

四个 accepted output 对应的位置都会进入 committed continuation。

因此：

> **`accept_lens - 1` 只是把“accepted output 数”转换回“correct draft 数”，不是把 bonus 从 KV commit 中排除。**

### 2. `move_kv_cache()` 的参数方向已经从实现确认

`move_accept_tokens_to_target_kvcache()` 会先得到两组位置。

第一组是 destination：

~~~python
tgt_cache_loc
~~~

它来自正式 sequence continuation：

~~~text
[seq_len, seq_len + accept_lens)
~~~

对应的 `req_to_token` physical slots。

第二组是 source：

~~~python
accept_out_cache_loc
~~~

它来自：

~~~text
accept_index
    ↓
batch.out_cache_loc
    ↓
accepted candidate physical slots
~~~

最后调用：

~~~python
move_kv_cache(
    tgt_cache_loc,
    accept_out_cache_loc,
)
~~~

严格 Review `memory_pool.py` 后，可以确认 API 顺序就是：

~~~python
move_kv_cache(tgt_loc, src_loc)
~~~

实际赋值也是：

~~~text
KV[tgt] = KV[src]
~~~

例如 paged KV implementation 的核心操作是：

~~~python
kb[pages_t, ..., offs_t, :] = kb[pages_s, ..., offs_s, :]
vb[pages_t, ..., offs_t, :] = vb[pages_s, ..., offs_s, :]
~~~

Unified MLA 路径同样是：

~~~python
env[tgt_pages] = env[src_pages]
~~~

所以这条链应该明确画成：

~~~text
accept_index
    │
    ▼
accept_out_cache_loc
    │
    │ source
    ▼
candidate KV
    │
    │ move_kv_cache(tgt, src)
    ▼
tgt_cache_loc
    │
    ▼
committed KV
~~~

不是反过来。

固定源码：

- [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)
- [`KVCache.move_kv_cache()` implementations](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/memory_pool.py)

### 3. 为什么 Tree Path 还要 Compact `predict / hidden_states`

KV 搬完以后，tree node layout 仍然可能是：

~~~text
[root, A, B, C, D, E, F]
~~~

但 accepted path 是：

~~~text
[root, B, E]
~~~

下一轮 Draft Extend 希望消费的却是一个 linear chain。

所以 `_finalize_accept_tree_path()` 继续：

~~~python
predict = _compact_accept_to_front(...)

logits_output.hidden_states =
    _compact_accept_to_front(...)
~~~

`_compact_accept_to_front()` 会按 `accept_index` gather accepted rows，再放到每个 request block 的前面。

因此 Accept 后其实同时发生三件事：

~~~text
1. path selection
   tree rows → accepted rows

2. KV relocation
   candidate physical slots → committed physical slots

3. tensor compaction
   tree-node predict/hidden layout → contiguous chain layout
~~~

这三件事不能统称成一个“KV compact”。

### 4. 当前 DeepSeek-V4 + DSpark 为什么不走这套 Tree Compact

当前 `_handle_dspark()` 会把：

~~~text
speculative_num_steps = 1
speculative_eagle_topk = 1
~~~

而真正 candidate window 由 `gamma` 控制：

~~~text
verify_window = gamma + 1
~~~

`dspark_worker_v2.py` 也明确写着：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~

因此：

> **通用 EAGLE Spec V2 需要 `accept_index` 解决 tree → chain；当前 DeepSeek-V4 DSpark 本质上是 linear chain，accepted prefix 天然连续。**

不能把 EAGLE tree compaction 机械套到 DSpark 当前实现上。

---

## 四、Reject 不是立即 Free：Spec V2 Overshoot 的真实生命周期

如果只看算法，reject 好像意味着：

~~~text
candidate 被拒绝
    ↓
删掉对应 KV
~~~

但 Spec V2 的内存管理不是这么工作的。

### 1. `kv_committed_len` 与 `kv_allocated_len` 是两条不同的线

每个 request 至少维护：

~~~text
kv_committed_len
kv_allocated_len
~~~

可以画成：

~~~text
0                                              kv_allocated_len
|----------------------------------------------------|
|             committed              |   overshoot   |
|====================================|...............|
                                     ▲
                              kv_committed_len
~~~

`committed` 表示逻辑上已经由 request 接受的 KV 边界。

`allocated` 表示这个 request 当前持有到哪里的物理 capacity。

两者之间：

~~~text
[kv_committed_len, kv_allocated_len)
~~~

就是 speculative reserve / overshoot 的典型来源。

### 2. 正常下一轮 Decode 不会因为 reject 自动缩小 allocated tail

`page_aligned_decode_alloc_lens()` 的核心是：

~~~python
cur = r.kv.kv_allocated_len

nxt = max(
    cur,
    page_align(r.kv.kv_committed_len + reserve),
)
~~~

注意：

~~~text
nxt >= cur
~~~

所以正常 speculative iteration 的 allocation path 是“够了就复用，不够再扩”，而不是：

~~~text
每轮 Accept
→ 把 reject tail free
→ 下一轮再重新 alloc
~~~

另外：

~~~python
get_alloc_reserve_per_decode()
=
2 * get_alloc_len_per_decode()
~~~

源码注释明确说明这个 double buffer 用来吸收 overlap scheduling 下 `kv_committed_len` 的滞后。

因此正常 reject 后：

~~~text
candidate KV 被拒绝
        ↓
不进入 committed boundary
        ↓
physical slots 仍可能保持 allocated
        ↓
下一轮直接覆盖 / 复用
~~~

这不是 memory leak，而是 Spec V2 的 reserve 策略。

### 3. 那 Overshoot 到底什么时候真的 free？

严格看当前 lifecycle，至少有下面几类明确 free 点。

| 生命周期点 | 实际动作 |
| --- | --- |
| 正常 speculative reject | 通常**不立即 free**；只是不推进 committed ownership，tail 留作 reserve |
| Request finish / abort / normal release | `release_kv_cache()` 在 cache-owned/owned range 处理后，释放 `owned_kv_len → kv_allocated_len` 的 overallocated tail |
| Decode retract / preempt | `release_req()` 最终调用 `release_kv_cache(..., is_insert=False)`，设备侧 request KV（包括 overshoot）被释放；需要恢复的路径先做 backup/offload |
| StreamingSession finish overshoot | `_trim_overshoot()` 按 authoritative `finished_len` 释放超出下一轮真实输入的 tail |
| StreamingSession match/rewind | `_free_tail()` 在重新 extend 前释放旧 session row 中 `[prefix_len, kv_allocated_len)` 的 orphaned tail |
| 整个 session 释放 | session-owned row 最终整体回收 |

因此上一版如果只写：

> “Rejected slot 留着，最后 request finish 才 free”

也不够准确。

更准确的说法是：

> **普通 iteration 内 reject 不触发逐候选 free；真正 reclaim 由 request/session 生命周期的 ownership 转换触发。**

### 4. `page_size > 1` 时为什么还不能从任意 token 边界 free

Paged allocator 释放的是整页。

因此：

~~~text
committed boundary
~~~

如果落在一个 page 中间，不能直接把包含 live committed token 的整页释放。

`_release_overallocated_kv_indices()` 和 StreamingSession 的 `_free_kv_aligned()` 都会把 free start 向上对齐：

~~~text
ceil_align(boundary, page_size)
~~~

这意味着：

~~~text
boundary 到下一个 page 边界之间
~~~

即使逻辑上已经属于 overshoot，也可能暂时继续挂在 request/session 上，直到整页可以安全回收。

所以：

> **逻辑 Reject、逻辑 ownership 失效、物理 page free，是三个不同时间点。**

固定源码：

- [`allocation_sizing.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/allocation_sizing.py)
- [`release_kv_cache()` / `_release_overallocated_kv_indices()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/common.py)
- [`release_req()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/managers/schedule_batch.py)
- [`StreamingSession._trim_overshoot()` / `_free_tail()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/session/streaming_session.py)

---

## 五、Ascend DeepSeek-V4：C4/C128 不是“一份 KV”，Cleanup 至少要拆成三类

DeepSeek-V4 让“Reject 后清理什么”变得更复杂，因为它不只有 full KV。

Ascend DSV4 pool 至少要区分：

~~~text
Full KV
SWA KV
C4 compressed KV
C128 compressed KV

C4 compressor state
C128 compressor state
~~~

而它们的地址与 ownership 不是一套规则。

### 1. Ascend NPU 上 C4 state 与 C128 state 的 ownership 不同

NPU DSV4 memory pool 的源码注释明确写着：

~~~text
C4A / C4Li state
→ follows SWA physical pages

C128A state
→ follows req_pool_idx + absolute position
~~~

因此：

~~~text
C4 state
~~~

更像 page-owned / SWA-addressed state；

而：

~~~text
C128 state
~~~

是 request-scoped ring。

这已经决定了：

> **不能设计一个“按 req_pool_idx 把 C4/C128 全部清掉”的统一 rollback。**

### 2. `clear_unaccepted_c128_draft_states()` 清的是 state，不是 C128 KV pages

共享 DeepSeek-V4 pool 中存在：

~~~python
clear_unaccepted_c128_draft_states(...)
~~~

其注释直接给出原因：

~~~text
C128 compression can read rejected draft slots at a boundary;
C4 overwrites its draft slots before reading them.
~~~

kernel 对每个 rejected `draft_offset`：

~~~text
if draft_offset < accept_len:
    keep

else:
    slot = (seq_len + draft_offset) % ring_size
    clear that C128 state row
~~~

清理动作是：

~~~text
state first half  -> 0
state second half -> -inf
~~~

这属于：

~~~text
stale-state sanitation
~~~

目的不是把显存页归还 allocator，而是避免以后碰到 compression boundary 时读到 rejected draft 留下的 state。

因此必须把两个动作分开：

~~~text
C128 rejected state cleanup
≠
C128 KV page free
~~~

### 3. 为什么 C4 没有同样的 rejected-state clear

固定源码给出的理由很明确：

~~~text
C4 overwrites its draft slots before reading them.
~~~

所以它不需要照抄 C128 的逐 rejected-offset clear。

这不是说：

~~~text
“C4 永远不需要生命周期管理”
~~~

而是说：

> **在这个 specific rejected-draft stale-state hazard 上，C4 的 overwrite-before-read 语义已经避免了同类问题。**

另外，Ascend NPU 的 C4 state 跟随 SWA physical pages，本来也不属于 C128 那种 request-scoped ring ownership。

### 4. C128 KV page 真正怎样释放

Ascend NPU 的 C128 KV 有独立 allocator 和：

~~~text
req_to_c128_sidecar
~~~

用来记录 request 当前引用的 C128 physical pages。

`release_c128_pages()` 维护 page refcount：

~~~text
refcount -= 1
    ↓
refcount == 0
    ↓
真正归还 c128_attn_allocator
~~~

而 request row 释放时：

~~~text
DSV4NPUReqToTokenPool.free(req)
        ↓
_dsv4_free(req)
        ↓
DSV4NPUTokenToKVPoolAllocator.free(
    req=req,
    req_to_token_pool=...
)
        ↓
release C128 sidecar pages
zero sidecar row
clear request-scoped C128 state
~~~

还有一个容易忽略的路径：当 request 的 C128 prefix sidecar 被替换时，`replace_req_c128_prefix()` 会对变化掉的旧 page reference 做 release，并对新 page retain。

因此 C128 physical page 的 ownership 更接近：

~~~text
sidecar reference lifecycle
+
page refcount
~~~

而不是：

~~~text
accept_lens 直接决定 free 哪几页
~~~

### 5. `free(free_index)` 为什么不能顺便把 C128 全部 free

DSV4 allocator 自己把 free API 分成两种：

~~~text
free(free_index)
→ full + SWA only

free(req=..., req_to_token_pool=...)
→ 有 request identity
→ 才能处理 C128 sidecar / request-scoped state
~~~

原因也很直接：

> 只有一个 generic full-KV slot index 时，没有足够的 request identity 去安全决定独立 C128 page 的引用关系。

所以文章如果写成：

~~~text
tree_cache.free_kv_row(...)
→ Full/SWA/C4/C128 全部同步释放
~~~

会过度概括。

### 6. Ascend 还有一个重要例外：ONLINE_C128 根本不是当前 NPU 路径

NPU DSV4 pool 初始化时直接 assert：

~~~text
ONLINE_C128
+
ratio == 128
→ 不允许
~~~

注释说明：

~~~text
ONLINE_C128 is CUDA-only;
NPU fused compressor has no online mode.
~~~

因此 base implementation 里：

~~~python
if ONLINE_C128:
    return
~~~

这种 cleanup bypass 不能被解释成“Ascend 也可能这样”。

在本文固定版本里，Ascend NPU 不走 ONLINE_C128。

### 7. 最后一个必须保留的证据边界：EAGLE 有显式 C128 rejected-state cleanup，DSpark 不能直接照搬这个结论

当前仓库中：

~~~python
clear_unaccepted_c128_draft_states(...)
~~~

的显式 verify 调用点位于共享 EAGLE verify 流程：

~~~text
run_eagle_verify()
~~~

本次 Review 没有在 DSpark Worker / Verify 路径找到同名 cleanup 的直接调用。

因此不能写：

~~~text
“所有 DeepSeek-V4 speculative path
都会在 Accept 后调用 clear_unaccepted_c128_draft_states”
~~~

更安全、也更准确的是：

> **EAGLE shared verify path 明确对 rejected C128 compressor state 做 cleanup；DSpark 当前是另一套 chain verify / commit pipeline，必须按自己的 state-write/commit 逻辑审计，不能仅凭共享 pool 存在这个 helper 就推断它一定被调用。**

DSpark 的 `commit_lens` gate、Target hidden → Draft state injection 解决的是另一类“只提交 accepted state”的问题，也不能和 C128 compressor-state sanitation 混为一谈。

固定源码：

- [`deepseek_v4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
- [`c128_cleanup.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/attention/dsv4/c128_cleanup.py)
- [`dsv4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)
- [`dsv4_allocator.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py)
- [`dsv4_req_to_token_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py)

---

## 六、把 Commit、Compact、Rollback 重新画成一台状态机

到这里，可以把整轮 speculative KV 生命周期画成下面这样：

~~~mermaid
flowchart TD
    A[Committed Prefix] --> B[Reserve / Allocate Spec KV Capacity]
    B --> C[Draft Candidates]
    C --> D[Target Verify]
    D --> E[Candidate KV written]
    D --> F[Target logits]

    F --> G[Accept]
    G --> H[accept_index]
    G --> I[accept_lens / commit_lens]

    H --> J[Select accepted verify rows]
    J --> K[Resolve source KV slots]
    I --> L[Resolve committed logical positions]
    L --> M[Resolve target KV slots]

    K --> N["move_kv_cache(tgt, src)"]
    M --> N
    N --> O[Committed KV]

    H --> P[Compact predict / hidden]
    P --> Q[Next-round linear state]

    G --> R[Rejected candidates]
    R --> S[Do not advance committed ownership]
    S --> T[Keep as reusable overshoot]
    T --> U[Finish / retract / trim / rewind]
    U --> V[Page-safe physical reclaim]

    R --> W[DSV4 special state hygiene]
~~~

这里的 “Rollback” 已经可以重新定义。

它不是：

~~~text
restore an old snapshot
~~~

而是几类不同机制的组合：

~~~text
1. logical non-commit
2. accepted-path remap / relocation
3. tensor compaction
4. speculative reserve reuse
5. lifecycle-driven physical reclaim
6. backend-specific stale-state cleanup
~~~

所以更准确的词其实是：

> **state convergence**

一次 speculative iteration 结束后，所有子系统必须重新收敛到同一个 committed sequence boundary。

### Debug 时最值得检查的几个 invariant

以后排查 Spec V2 KV bug，可以先检查下面这组关系。

~~~text
accept_lens
=
num_correct_drafts + 1
~~~

~~~text
valid_accept_index_count
=
accept_lens
~~~

~~~text
predict[accept_index[:accept_lens]]
=
accepted drafts + terminal bonus
~~~

~~~text
move_kv_cache(tgt, src)
=
KV[tgt] <- KV[src]
~~~

~~~text
new_seq_lens
=
old_seq_lens + accept_lens
~~~

~~~text
kv_committed_len
<=
kv_allocated_len
~~~

以及：

~~~text
Reject
≠
immediate physical free
~~~

如果 DeepSeek-V4 / Ascend 还涉及 compressed state，再额外问三句：

~~~text
我现在讨论的是 KV page 还是 compressor state？

这个 state 是 SWA/page-owned 还是 request-scoped？

cleanup 是为了防 stale read，还是为了把 physical page 还给 allocator？
~~~

只要这三问没有先回答清楚，就不应该笼统地说“清 KV”。

---

## 七、源码阅读顺序与最终结论

如果想自己复现这次 Review，推荐按下面顺序读，不要从 DeepSeek-V4 backend 的几千行文件开始乱搜。

| 目标 | 固定版本源码 |
| --- | --- |
| Verify 输入与 candidate `out_cache_loc` | [`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/eagle_utils.py) |
| Greedy accepted path / bonus placement | [`spec_tree.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/spec_tree.py) |
| Rejection sampling accepted path / terminal token | [`reject_sampling.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/reject_sampling.py) |
| Bonus extraction contract | [`eagle.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/eagle.py) |
| Accepted KV relocation | [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py) |
| `move_kv_cache(tgt, src)` 实现 | [`memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/memory_pool.py) |
| Tree path tensor compaction | [`eagle_worker_common.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/eagle_worker_common.py) |
| Spec V2 reserve sizing | [`allocation_sizing.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/allocation_sizing.py) |
| Finish/abort overshoot reclaim | [`mem_cache/common.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/common.py) |
| Retract reclaim | [`schedule_batch.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/managers/schedule_batch.py) |
| Streaming overshoot trim | [`streaming_session.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/session/streaming_session.py) |
| Ascend C128 sidecar/refcount | [`dsv4_allocator.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py) |
| Ascend C4/C128 state ownership | [`dsv4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py) |
| Rejected C128 state sanitation | [`c128_cleanup.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/attention/dsv4/c128_cleanup.py) |

最终可以把整篇压成一句话：

> **Speculative Decoding 的 Accept 不是“挑几个 Token 输出”，而是把 verify-row space 中暂时存在的未来，重新收敛到一个线性的 committed sequence boundary；KV relocation、tree compaction、overshoot reclaim 与 DSV4 state cleanup，只是这次状态收敛在不同子系统中的具体表现。**

理解这一层以后，再看下一篇 Tree Attention 就会容易很多。

因为 Tree Attention 真正解决的，是 Commit 发生**之前**的另一个问题：

> **当未来还没有收敛成一条线时，一棵 speculative tree 中的每个 candidate，到底允许看到哪些 ancestor KV？**
