# 投机解码 KV 管理：接受路径怎样提交、整理与回收

Target 验证过一批候选后，候选 KV 已经写进缓存，但只有接受路径可以成为后续请求的历史。系统需要把这条路径对应的 Verify 行映射到正式序列位置，再决定多余的预留空间何时复用或释放。

本文用旧 root R、接受的 A/C 和新 bonus X 说明三个动作：Commit 确认有效历史，Compact 整理布局，Rollback 撤销候选对后续状态的影响。输出追加 A/C/X 时，已经物化的输入 KV 对应 R/A/C；这个错位关系贯穿后面的地址映射。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 6880a4795533640f41ebb3db9e4ae0af5a371a1f`，核对日期为 2026-09-20。主线覆盖 Spec V2 的 EAGLE accepted-path KV compaction、经典 rejection sampling、Target-only sampling，以及 Ascend DeepSeek-V4 的 C4/C128 KV/state 生命周期。当前 DeepSeek-V4 + DSpark 仍是 `topk=1` 的 linear chain；因此文中的 tree-specific `accept_index` compaction 主要解释通用 EAGLE Spec V2，DSpark 部分会单独标出。

## 一、Token、Verify 行、逻辑位置与物理槽位

普通 Decode 很容易形成一个过度简化的心智模型：

~~~text
第 4096 个 token
        ↓
KV Cache 第 4096 行
~~~

真实 SGLang 并不是这样。

### Logical Sequence Position

假设当前请求已经提交：`seq_len = 4096`

那么下一段正式 continuation 的逻辑位置是：`4096, 4097, 4098, ...`

这是 request 语义上的 sequence position。

### `req_to_token`：Logical Position → Physical KV Slot

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

### `out_cache_loc`：这次 Forward 的 K/V 应该写到哪里

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

### `accept_index`：不是 KV slot，而是 accepted verify-row path

当 `topk > 1` 时，candidate 是 tree：

~~~text
                 root
              /        \
             A          B
           /   \      /   \
          C     D    E     F
~~~

如果最终 accepted path 是：`root → B → E`

Tree 的接受行可能不连续，因此按 `accept_index` 收集选中的行。

`accept_index` 解决的是：

~~~text
tree / verify-row space
        ↓
accepted path
~~~

而不是：`physical KV address`

链路是：

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

索引在下面几层之间转换：

> **不要把 `accept_index`、logical token position、`out_cache_loc` 和 physical KV slot 当成同一个索引空间。**

固定源码入口：

- [`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/eagle_utils.py)
- [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

## 二、接受结果怎样索引输出

这里先固定接受索引与预测行的对应关系。

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

### Greedy Tree Verify 的真实写法

`verify_tree_greedy_kernel_triton()` 一开始会把 root 对应的 retrieve index 写到：`accept_index[0]`

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

因此如果正确 draft 是：`d1, d2, d3`

最终不是：`accept_index = [d1_node, d2_node, d3_node, bonus_node]`

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

### `accept_lens` 为什么等于 correct drafts + 1

`eagle_sample()` 返回前明确保持：

~~~python
num_correct_drafts
~~~

只统计 draft，然后返回：

~~~python
return predict, num_correct_drafts + 1, accept_index
~~~

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

### Rejection Sampling 也保持相同契约

经典 rejection sampling 的 Triton kernel 同样：`accept_index[0] = root_row`

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

### Simulated accept 也遵循同一个布局

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

### NPU Sampling 的输入与结果消费

非 Greedy NPU path 会从：`sgl_kernel_npu.sample`

导入 `tree_speculative_sampling_target_only` / `chain_speculative_sampling_triton`。

这条 NPU 分支的结果按以下方式交给后续模块：

- SGLang 给 NPU kernel 传入同样的 `predict / accept_index / accept_token_num` contract；
- 随后的 TP broadcast 同步这三项；
- downstream `fill_bonus_tokens` 与 KV compaction 仍按“有效 `accept_index` 前缀长度 = `accept_lens`”消费它们。

设备核位于 `sgl_kernel_npu`，SGLang 按上述输入输出关系衔接接受、广播与缓存整理。

固定源码：

- [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/spec_tree.py)
- [`chain_speculative_sampling_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/reject_sampling.py)
- [`fill_bonus_tokens`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/eagle.py)
- [`generate_simulated_accept_index()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

## 三、把接受路径映射到正式 KV

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

### 输出增加多少 Token，和哪些 Token 已经有 KV，是两件事

`move_accept_tokens_to_target_kvcache()` 的第三个参数是 `num_correct_drafts`，因此调用方传入 `accept_lens - 1`；函数内部再按 `num_correct_drafts + 1` 创建目标区间。这个 `+1` 对应本轮作为输入的旧 root，而本轮新采样出的 terminal bonus 要等下一次 forward 才有自己的 KV。

例如，旧 root 是 R，接受的 Draft 是 A、C，最后采样得到 X：

| 本轮 Verify 行 | 该行已计算的 KV 属于 | 该行选中的输出 |
| --- | --- | --- |
| root 行 | R | A |
| A 行 | A | C |
| C 行 | C | X（新 bonus） |

本轮输出追加 `[A, C, X]`，KV 则提交 `[R, A, C]`。两者长度都是 `accept_lens=3`，Token 身份相差一位。`accept_index` 选中的是 Verify 行；用它取 `predict` 得到输出，用它搬缓存得到这些行的输入 KV。

因此，`accept_lens - 1` 先转换为正确 Draft 数，函数再补上旧 Root 对应的输入 KV。新输出 X 留到下一轮 Forward 才生成 KV。

### `move_kv_cache()` 的参数方向已经从实现确认

`move_accept_tokens_to_target_kvcache()` 会先得到两组位置。

第一组是 destination：

~~~python
tgt_cache_loc
~~~

它来自正式 sequence continuation：`[seq_len, seq_len + accept_lens)`

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

`memory_pool.py` 的实现给出了明确的参数顺序：

~~~python
move_kv_cache(tgt_loc, src_loc)
~~~

实际赋值也是：`KV[tgt] = KV[src]`

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

### 为什么 Tree Path 还要 Compact `predict / hidden_states`

KV 搬完以后，tree node layout 仍然可能是：`[root, A, B, C, D, E, F]`

但 accepted path 是：`[root, B, E]`

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

下面分别沿这三种数据整理操作追踪。

### 当前 DeepSeek-V4 + DSpark 为什么不走这套 Tree Compact

当前 `_handle_dspark()` 会把：

~~~text
speculative_num_steps = 1
speculative_eagle_topk = 1
~~~

而 candidate window 由 `gamma` 控制：`verify_window = gamma + 1`

`dspark_worker_v2.py` 也明确写着：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~

因此：

> **通用 EAGLE Spec V2 需要 `accept_index` 解决 tree → chain；当前 DeepSeek-V4 DSpark 是 linear chain，accepted prefix 天然连续。**

DSpark 按连续前缀提交，EAGLE Tree 则额外按接受索引收集非连续行。

## 四、被拒绝候选与预留空间的生命周期

如果只看算法，reject 好像意味着：

~~~text
candidate 被拒绝
    ↓
删掉对应 KV
~~~

但 Spec V2 的内存管理不是这么工作的。

### `kv_committed_len` 与 `kv_allocated_len` 是两条不同的线

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

两者之间：`[kv_committed_len, kv_allocated_len)`

就是 speculative reserve / overshoot 的典型来源。

### 正常下一轮 Decode 不会因为 reject 自动缩小 allocated tail

`page_aligned_decode_alloc_lens()` 的核心是：

~~~python
cur = r.kv.kv_allocated_len

nxt = max(
    cur,
    page_align(r.kv.kv_committed_len + reserve),
)
~~~

注意：`nxt >= cur`

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

### Overshoot 的实际释放时机

当前生命周期在以下位置回收预留空间：

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

> **普通 iteration 内 reject 不触发逐候选 free； reclaim 由 request/session 生命周期的 ownership 转换触发。**

### 页分配下的回收边界

Paged allocator 释放的是整页。

因此：`committed boundary`

提交边界落在页中间时，包含活跃 Token 的整页继续保留，回收范围从可释放的整页开始。

`_release_overallocated_kv_indices()` 和 StreamingSession 的 `_free_kv_aligned()` 都会把 free start 向上对齐：`ceil_align(boundary, page_size)`

这意味着：`boundary 到下一个 page 边界之间`

即使逻辑上已经属于 overshoot，也可能暂时继续挂在 request/session 上，直到整页可以安全回收。

所以：

> **逻辑 Reject、逻辑 ownership 失效、物理 page free，是三个不同时间点。**

固定源码：

- [`allocation_sizing.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/allocation_sizing.py)
- [`release_kv_cache()` / `_release_overallocated_kv_indices()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/common.py)
- [`release_req()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/managers/schedule_batch.py)
- [`StreamingSession._trim_overshoot()` / `_free_tail()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/session/streaming_session.py)

## 五、Ascend 的 KV 页与压缩状态清理

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

### C4 state 的 ownership 还取决于 NPU 架构

固定版本的 `dsv4_memory_pool.py` 区分两条路径：

| 路径 | C4 compressor state 的寻址与注册 |
| --- | --- |
| pre-A5 / EXPLICIT | 跟随 SWA physical page/state index，纳入 SWA state buffers |
| A5 / CYCLE | 使用 `req_pool_idx` 对应的独立环形状态，通过 `get_c4_state_buf_infos()` 单独注册 |

C128 state 使用请求与绝对位置管理的 ring。讨论 rollback 时，既要区分 compressed KV 与 compressor state，也要确认架构分支。pre-A5 的 C4 按页绑定，A5/CYCLE 则按其请求 Ring 管理。

### `clear_unaccepted_c128_draft_states()` 清的是 state，不是 C128 KV pages

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

这属于：`stale-state sanitation`

目的不是把显存页归还 allocator，而是避免以后碰到 compression boundary 时读到 rejected draft 留下的 state。

因此必须把两个动作分开：

~~~text
C128 rejected state cleanup
≠
C128 KV page free
~~~

### 为什么 C4 没有同样的 rejected-state clear

固定源码给出的理由很明确：`C4 overwrites its draft slots before reading them.`

所以它不需要照抄 C128 的逐 rejected-offset clear。

这不是说：`“C4 永远不需要生命周期管理”`

而是说：

> **在这个 specific rejected-draft stale-state hazard 上，C4 的 overwrite-before-read 语义已经避免了同类问题。**

另外，pre-A5 的 C4 state 跟随 SWA physical pages；A5 的 CYCLE 路径采用独立请求状态。是否需要清理，要结合各自的寻址和覆盖顺序判断。

### C128 KV 页的引用计数与释放

Ascend NPU 的 C128 KV 有独立 allocator 和：`req_to_c128_sidecar`

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

而不是：`accept_lens 直接决定 free 哪几页`

### SWA 释放与 C128 引用计数的关系

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

### Ascend 还有一个重要例外：ONLINE_C128 根本不是当前 NPU 路径

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

这条 cleanup bypass 属于 CUDA 的 ONLINE_C128 路径；本文固定版本的 Ascend NPU 使用另一条路径。

### EAGLE Cleanup 与 DSpark 提交路径

`clear_unaccepted_c128_draft_states(...)` 的显式调用位于 EAGLE 共享 Verify 流程 `run_eagle_verify()`，清理未接受的 C128 Compressor State。

同版本 DSpark Worker / Verify 路径未找到同名直接调用。DSpark 通过 `commit_lens` 和 Target Hidden 注入控制提交范围；分析它的 Compressor State 时，沿自身写入与提交路径继续追踪。

固定源码：

- [`deepseek_v4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
- [`c128_cleanup.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/attention/dsv4/c128_cleanup.py)
- [`dsv4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)
- [`dsv4_allocator.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py)
- [`dsv4_req_to_token_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py)

## 六、串起 Commit、Compact 与 Rollback

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

它不是：`restore an old snapshot`

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

### Debug 时核对索引、长度与页归属

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

明确这三项后，就能定位要处理的是缓存行、预留页还是压缩器状态。

## 七、源码入口与计数对照

下面按输入构造、接受路径、物理搬移与回收顺序列出源码入口。

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

核对计数时，要分别记录有效输出长度、已物化输入 KV 的提交边界和预留空间上界。它们相互关联，却不总是指向同一枚 Token 或同一个物理位置。前面的 `R,A,C → A,C,X` 示例可以作为检查索引、搬移与回收的最小对照。
