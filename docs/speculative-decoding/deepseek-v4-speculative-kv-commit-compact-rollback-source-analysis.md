# 投机解码 KV 管理：接受路径怎样提交、整理与回收

Verify 可能已为所有候选写入 KV，请求却只能延续选中的历史。接受后要把正确输入行对应到正确物理槽位，拒绝的尾部则可能继续作为预留容量。显存未下降，尚不能说明回收出错。

下面用旧 root R、接受 A/C、新 bonus X，逐步检查行索引、地址与释放边界。Commit 确认有效历史，Compact 整理后续布局，Rollback 防止未接受状态被继续读取，不一定恢复整份快照或立刻 free。本轮输出 A/C/X，对应已计算输入 R/A/C，这个错位会一直延续到地址映射。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 6880a4795533640f41ebb3db9e4ae0af5a371a1f`，核对日期为 2026-09-20。分析 Spec V2 的 EAGLE compaction、classic rejection、Target-only sampling 及 Ascend V4 C4/C128 生命周期。固定 V4 DSpark 仍为 topk=1 链，树索引整理属于通用 EAGLE，相关差别单独说明。

</details>

## 一、Token、Verify 行、逻辑位置与物理槽位

第 4096 个 Token 并不必然写入 KV Cache 的第 4096 行。Token ID、逻辑位置、物理 slot 分属不同对象。下面地址均为教学数字，代码省略上下文，只用于推演调用关系。

### Logical Sequence Position

若有效输入 Prefix 长 `seq_len=4096`，下一个输入位于位置 4096，后面为 4097、4098。这个长度是边界，既非词表 ID，也非物理行号。

### `req_to_token`：Logical Position → Physical KV Slot

通过 `req_to_token[req_pool_idx, logical_position]` 查表，才能把序列位置转换成物理地址。以 request #7 为例：

| 逻辑位置 | 物理 KV slot |
| ---: | ---: |
| 4094 | 18320 |
| 4095 | 18321 |
| 4096 | 42080 |
| 4097 | 42081 |
| 4098 | 42082 |

位置 4096 在示例中落到 slot 42080，前一位置 4095 却在 18321，逻辑序列可以跨物理区间。

分页缓存通过这张映射表保持逻辑序列连续，无需物理内存也连续。



### `out_cache_loc`：这次 Forward 的 K/V 应该写到哪里

Verify 处理多行，`batch.out_cache_loc` 给出每行产生的 KV 应写入哪个物理 slot。



`eagle_prepare_for_verify()` 绑定输入和目标位置：

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



Target 为这些候选计算 logits、hidden states，同时把 candidate KV 写入 `out_cache_loc` 指定的位置。

先计算并写入，才能根据 Target 结果接受。物理上写过候选与逻辑上提交候选发生在不同阶段，后者才决定下一轮可用历史。

### `accept_index`：不是 KV slot，而是 accepted verify-row path

topk>1 的候选可以构成：

~~~text
                 root
              /        \
             A          B
           /   \      /   \
          C     D    E     F
~~~

若接受 `root → B → E`，这些行未必连续，需用 `accept_index` 收集。



`accept_index` 记录树内选中的 Verify 行，还需经过两次查表才能完成搬移：

| 搬移位置 | 怎样确定 |
| --- | --- |
| 源 slot | 由 `accept_index` 选出 Verify 行，再查 `out_cache_loc[row]` |
| 目标 slot | 由正式提交的逻辑位置查 `req_to_token` |

因此先由 accept_index 找 Verify 行，再用 out_cache_loc 找源 slot，最后按 req_to_token 找正式目标。核搬移时要保留这三步，不能因整数相同就认为对象相同。



- [`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/eagle_utils.py)
- [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

## 二、接受结果怎样索引输出

将输出写成 `[draft1,draft2,draft3,bonus]` 时，容易误以为接受索引也列出了这四个输出节点。实际上，新 bonus 来自某个已执行行的预测，本轮尚无以它为输入的行。索引必须从旧 root 开始，才能读取完整输出。

### Greedy Tree Verify 的真实写法

Greedy Kernel 先将 root 的 retrieve index 放入 `accept_index[0]`。

随后每接受一个 child：

1. 将候选 ID 写入上一有效行的 predict；
2. 将 child 行追加到 accept_index；
3. 把 child 设为新的 `last_accepted_global_idx`。

简化为：

~~~python
accept_index[0] = root_row
last_row = root_row

for accepted_draft in path:
    predicts[last_row] = accepted_draft
    accept_index[next] = child_row
    last_row = child_row

predicts[last_row] = bonus
~~~

若接受 d1、d2、d3，有效索引与预测的对应关系为：

| 有效索引顺序 | 被选中的输入行 | 该行预测 |
| ---: | --- | --- |
| 0 | root | d1 |
| 1 | d1 | d2 |
| 2 | d2 | d3 |
| 3 | d3 | bonus |

因此有效索引为 `[root_row,row_after_d1,row_after_d2,row_after_d3]`，读取 `predict[accept_index[:4]]` 才得到 `[d1,d2,d3,bonus]`。

末行 KV 属于最后有效输入，末行 predict 却是其预测的新 bonus。索引选择承载结果的 Verify 行，并不直接列出输出 Token 身份。

### `accept_lens` 为什么等于 correct drafts + 1

`eagle_sample()` 内部的 `num_correct_drafts` 只数接受的 Draft，返回时加上 terminal Target/bonus：

~~~python
return predict, num_correct_drafts + 1, accept_index
~~~

`fill_bonus_tokens` 再按每请求 block stride 取有效输出的末尾，索引为 `accept_stride * pid + accept_len - 1`。

有效索引数为 accept_lens，最后一行的 predict 是 bonus。buffer 还可能有填充，不能把分配宽度算成实际产出。

### Rejection Sampling 也保持相同契约

classic rejection Kernel 同样先设 `accept_index[0] = root_row`。每接受一个候选，就将当前行追加到 `accept_index`，并更新 `last_accepted_global_idx`；最后 residual / Target sampling 得到 terminal ID，写入 `Predicts[last_accepted_global_idx]`。

所以两条路径都通过 `predict[accept_index[:accept_lens]]` 取得“已接受 Draft + terminal Token”。采样方法不同，结果的行布局相同。

### Simulated accept 也遵循同一个布局

模拟接受索引也按相同方式填充：

~~~python
sim_accept_index[:, :simulate_acc_len] = ...
num_correct_drafts.fill_(simulate_acc_len - 1)
~~~

real-draft-token 模式最后补上：

~~~python
bonus_node_indices = sim_accept_index[:, simulate_acc_len - 1]
predict[bonus_node_indices] = target_predict[...]
~~~

布局可以用模拟值检查，但模拟长度不是实际模型接受结果，不能据此报告真实接受率或生成质量。

### NPU Sampling 的输入与结果消费

NPU 非 Greedy 从 `sgl_kernel_npu.sample` 导入 Tree/Chain Sampling Ops。



上层按以下契约交接：

- 传入 `predict / accept_index / accept_token_num`；
- 随后 TP broadcast 同步这些结果；
- bonus 提取与 KV compaction 只消费 accept_lens 指定的有效前缀。

这里确认 SGLang 的调用及消费方式，设备核本体位于外部包。



- [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/spec_tree.py)
- [`chain_speculative_sampling_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/reject_sampling.py)
- [`fill_bonus_tokens`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/eagle.py)
- [`generate_simulated_accept_index()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

## 三、把接受路径映射到正式 KV

Target 返回 `predict, accept_lens, accept_index` 后，树路径 finalize 继续调用：

~~~python
move_accept_tokens_to_target_kvcache(
    batch,
    accept_index,
    accept_lens - 1,
    token_to_kv_pool_allocator,
)
~~~

下面按输入行、源地址、目标地址核对，先看长度再看方向。

### 输出增加多少 Token，和哪些 Token 已经有 KV，是两件事

mover 的第三参数是 `num_correct_drafts`，所以传 `accept_lens-1`；内部再加旧 root，构成正式输入区间。新 terminal bonus 尚未执行，不属于本轮待搬 KV。

例如旧 R、接受 A/C、新 X：

| 本轮 Verify 行 | 该行已计算的 KV 属于 | 该行选中的输出 |
| --- | --- | --- |
| root 行 | R | A |
| A 行 | A | C |
| C 行 | C | X（新 bonus） |

输出 `[A, C, X]`，KV 提交 `[R, A, C]`，长度都为 `accept_lens=3`。同一索引读取 predict 和输入 KV，得到错位的两组 Token。

减 1 排除新 X，加 1 补旧 R，数量抵消但身份不同，不能因此要求搬移 X 的未生成缓存。

### `move_kv_cache()` 的参数方向已经从实现确认

mover 先构造 `tgt_cache_loc` 与 `accept_out_cache_loc`：前者取正式延续 `[seq_len, seq_len + accept_lens)` 对应的 `req_to_token` slots，后者通过 `accept_index` 从 `batch.out_cache_loc` 选出源 slots。随后调用：

~~~python
move_kv_cache(
    tgt_cache_loc,
    accept_out_cache_loc,
)
~~~

`move_kv_cache(tgt_loc, src_loc)` 对应赋值 `KV[tgt] = KV[src]`，目标参数在前。

paged KV 的操作为：

~~~python
kb[pages_t, ..., offs_t, :] = kb[pages_s, ..., offs_s, :]
vb[pages_t, ..., offs_t, :] = vb[pages_s, ..., offs_s, :]
~~~

Unified MLA 也按相同方向：

~~~python
env[tgt_pages] = env[src_pages]
~~~

继续教学例子。Verify 输入 `[R,A,B,C]` 写入 `[42080,42081,42082,42083]`，接受行 `[0,1,3]`：

| 接受的输入 | Verify 行 | 源 slot | 正式逻辑位置 | 目标 slot |
| --- | ---: | ---: | ---: | ---: |
| R | 0 | 42080 | 4096 | 42080 |
| A | 1 | 42081 | 4097 | 42081 |
| C | 3 | 42083 | 4098 | 42082 |

前两行源目标相同，最后一行仍需把 C 从 42083 复制到 42082，覆盖未接受的 B。



- [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)
- [`KVCache.move_kv_cache()` implementations](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/memory_pool.py)

### 为什么 Tree Path 还要 Compact `predict / hidden_states`

物理 KV 整理后，原树数组仍可能为 `[root, A, B, C, D, E, F]`。

若有效路径是 `[root, B, E]`，下一轮 Extend 却需连续链式数据。



因此还执行：

~~~python
predict = _compact_accept_to_front(...)

logits_output.hidden_states =
    _compact_accept_to_front(...)
~~~

`_compact_accept_to_front()` 依接受索引 gather，再置于每请求 block 前部。

这轮整理依次选择 accepted rows、将对应 KV 搬到正式 slots，再将 `predict/hidden` 收集到每请求 block 前部。它们处理的分别是树行索引、池地址与 Tensor 行布局。

KV relocation 改池地址，Tensor compaction 改 predict、hidden 行序。只搬对 KV 而 hidden 仍来自原分支，下轮 Extend 依然可能将正确 Token 配错特征。

### 当前 DeepSeek-V4 + DSpark 为什么不走这套 Tree Compact

DSpark 参数处理固定 `speculative_num_steps=1`、`speculative_eagle_topk=1`。

候选窗口由 gamma 决定，`verify_window = gamma + 1`。

其 Worker 对提交注明：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~

EAGLE 树需索引定位非连续路径，DSpark 链只需前缀长度。相同 accept length 名称不能证明两者用了同一种搬移。

后文分别按链边界和树索引理解提交。

## 四、被拒绝候选与预留空间的生命周期

拒绝候选后，Accept 先限制有效历史，allocator 可以之后再回收。判断尾部能否释放，还需看容量所有权、分页与下轮复用。

### `kv_committed_len` 与 `kv_allocated_len` 是两条不同的线

每请求分别用 `kv_committed_len` 记录有效输入边界、`kv_allocated_len` 记录持有容量：

~~~text
0                                              kv_allocated_len
|----------------------------------------------------|
|             committed              |   overshoot   |
|====================================|...............|
                                     ▲
                              kv_committed_len
~~~

committed 指有效输入边界，allocated 指持有容量。二者之间的 reserve / overshoot 可含旧候选，物理数据存在不赋予它历史身份，后续读取仍需受正式边界约束。

### 正常下一轮 Decode 不会因为 reject 自动缩小 allocated tail

正常迭代采用：

~~~python
cur = r.kv.kv_allocated_len

nxt = max(
    cur,
    page_align(r.kv.kv_committed_len + reserve),
)
~~~

`nxt >= cur` 让已有容量优先复用，不足才扩展，避免每轮拒绝后 free、下轮又 alloc。reserve 使用 `get_alloc_reserve_per_decode() = 2 * get_alloc_len_per_decode()`，以 double buffer 吸收 overlap 下 committed 长度的滞后。

候选虽然未进入 committed 历史，其 slots 仍可处于 allocated 范围，留待下一轮覆盖。

例如 allocated=4352、committed=4099、reserve=16、allocator page_size=128，`4099+16=4115` 向上对齐为 4224，再取 `max(4352,4224)`，容量仍为 4352。此例符合 reserve 策略；是否泄漏要看结束或转移所有权后能否回收，不能只看一次拒绝前后显存。

### Overshoot 的实际释放时机

预留空间在生命周期转换时处理：

| 生命周期点 | 实际动作 |
| --- | --- |
| 正常 speculative reject | 通常**不立即 free**；只是不推进 committed ownership，tail 留作 reserve |
| Request finish / abort / normal release | `release_kv_cache()` 在 cache-owned/owned range 处理后，释放 `owned_kv_len → kv_allocated_len` 的 overallocated tail |
| Decode retract / preempt | `release_req()` 最终调用 `release_kv_cache(..., is_insert=False)`，设备侧 request KV（包括 overshoot）被释放；需要恢复的路径先做 backup/offload |
| StreamingSession finish overshoot | `_trim_overshoot()` 按 authoritative `finished_len` 释放超出下一轮真实输入的 tail |
| StreamingSession match/rewind | `_free_tail()` 在重新 extend 前释放旧 session row 中 `[prefix_len, kv_allocated_len)` 的 orphaned tail |
| 整个 session 释放 | session-owned row 最终整体回收 |

普通 reject 通常只排除历史。结束释放也可能先将前缀交 cache 接管，session 还可继续持有，因此 request 不再持有某段，不等于它已归还 allocator。

### 页分配下的回收边界

分页释放要避开仍有活跃数据的部分页，否则从首个无效 slot free 可能误释放同页有效行。

相关实现先将起点对齐到 `ceil_align(boundary, page_size)`。

应取 allocator 真正 page_size，DCP 下可能大于 Attention kernel 页。4099、128 的安全起点为 4224；allocated=4352 时可释放 `[4224,4352)`。边界外 `[4099,4224)` 的 125 个位置仍与有效状态同页，不能在这一步单独释放。

所以排除候选、结束所有权、归还整页可以发生在不同时刻。



- [`allocation_sizing.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/allocation_sizing.py)
- [`release_kv_cache()` / `_release_overallocated_kv_indices()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/common.py)
- [`release_req()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/managers/schedule_batch.py)
- [`StreamingSession._trim_overshoot()` / `_free_tail()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/session/streaming_session.py)

## 五、Ascend 的 KV 页与压缩状态清理

V4 同时维护 Full、SWA、C4、C128 KV，以及 C4/C128 compressor state。

KV 供 Attention 读历史，compressor state 支持后续压缩，两者清理不会互相代替，寻址和所有权也不同。

### C4 state 的 ownership 还取决于 NPU 架构

固定 NPU pool 有以下架构分支：

| 路径 | C4 compressor state 的寻址与注册 |
| --- | --- |
| pre-A5 / EXPLICIT | 跟随 SWA physical page/state index，纳入 SWA state buffers |
| A5 / CYCLE | 使用 `req_pool_idx` 对应的独立环形状态，通过 `get_c4_state_buf_infos()` 单独注册 |

C128 state 按请求、绝对位置维护 ring。pre-A5 C4 随页，A5/CYCLE 则独立请求 ring，不能不分架构讨论同一种 rollback。

### `clear_unaccepted_c128_draft_states()` 清的是 state，不是 C128 KV pages

共享 V4 pool 的 `clear_unaccepted_c128_draft_states()` 处理被拒绝的压缩状态。源码注释解释了 C128 与 C4 的差别：

~~~text
C128 compression can read rejected draft slots at a boundary;
C4 overwrites its draft slots before reading them.
~~~

清理逻辑按 draft_offset 判断：

~~~text
if draft_offset < accept_len:
    keep

else:
    slot = (seq_len + draft_offset) % ring_size
    clear that C128 state row
~~~

无效 state 的前半部分写为 0，后半部分写为 `-inf`。

这是原地无效化，容量不变，用于避免后续压缩读旧候选。线性窗口 `accept_len=3` 保留 offset 0、1、2，清理从 3 开始；长度包括 root，不能换成仅数 Draft。

这一步没有释放 C128 KV page；它只是使被拒绝的 compressor state 无效。

### 为什么 C4 没有同样的 rejected-state clear

C4 的对应条件是 `C4 overwrites its draft slots before reading them.`，因此不用相同 rejected-offset clear。



这只说明当前 stale-state 风险由覆盖顺序处理，不代表 C4 页、请求 Ring 或其他生命周期无需管理。

清理选择仍需结合 pre-A5 的页绑定、A5/CYCLE 的请求状态判断。

### C128 KV 页的引用计数与释放

C128 KV 另有 allocator 和 `req_to_c128_sidecar`，记录请求引用的物理页。



`release_c128_pages()` 先将引用计数减一，仅在归零后归还 `c128_attn_allocator`。

请求 row 释放时还经过：

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

前缀 sidecar 替换也会在 `replace_req_c128_prefix()` 释放旧 page 引用、retain 新引用。

例如两个持有者使计数为 2，释放一个后为 1，最后归零才还页。替换需正确增减引用，不能仅用某请求的接受长度决定整页是否 free。

因此回收由 sidecar 引用的生命周期和 page refcount 决定，`accept_lens` 不能直接选出要释放的页。

### SWA 释放与 C128 引用计数的关系

free API 因而分为：

~~~text
free(free_index)
→ full + SWA only

free(req=..., req_to_token_pool=...)
→ 有 request identity
→ 才能处理 C128 sidecar / request-scoped state
~~~

只有 generic full slot 无法确定请求 sidecar 的哪条引用。带 req 的释放提供额外身份，两条 API 承接不同层信息。

若把 `tree_cache.free_kv_row(...)` 理解为 Full/SWA/C4/C128 全部同步释放，就会漏掉 sidecar 和请求 state 的条件。判断回收是否完整，要同时核调用参数、请求 row 及引用计数。

### Ascend 还有一个重要例外：ONLINE_C128 根本不是当前 NPU 路径

固定 NPU pool 初始化时拒绝 `ONLINE_C128` 与 `ratio == 128` 的组合，注释说明：

~~~text
ONLINE_C128 is CUDA-only;
NPU fused compressor has no online mode.
~~~

所以基类的这段绕过清理：

~~~python
if ONLINE_C128:
    return
~~~

属于 CUDA ONLINE_C128，不能套到本文 NPU 路径。

### EAGLE Cleanup 与 DSpark 提交路径

同名 C128 clear 显式位于 EAGLE 的 `run_eagle_verify()`。

同版本 DSpark Worker / Verify 未发现同名直接调用，它通过 commit_lens、hidden 注入控制范围。这既不足以证明不用清理，也不足以证明 bug；还需沿自己的写入、覆盖、提交确认是否可能读到未接受状态。



- [`deepseek_v4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
- [`c128_cleanup.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/attention/dsv4/c128_cleanup.py)
- [`dsv4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)
- [`dsv4_allocator.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py)
- [`dsv4_req_to_token_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py)

## 六、串起 Commit、Compact 与 Rollback

将这些操作放回一轮生命周期：

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

Rollback 可以依靠逻辑不提交、接受路径搬移、Tensor compaction、预留容量复用、生命周期回收及后端状态清理共同完成，无需逐字节恢复完整旧快照。

关键是下轮可见 Token、KV、hidden、compressor state 对应同一有效历史，持有容量则可更大。检查正确性应看后续读取，而非只看尾部是否清零或显存下降。

### Debug 时核对索引、长度与页归属

排查时先核对这些关系，并保留有效索引前缀，避免误读填充区：

| 要核对的量 | 关系或方向 |
| --- | --- |
| 输出长度 | `accept_lens = num_correct_drafts + 1` |
| 有效索引数 | `valid_accept_index_count = accept_lens` |
| 有效输出 | `predict[accept_index[:accept_lens]]` 为已接受 Draft 与 terminal bonus |
| 搬移方向 | `move_kv_cache(tgt, src)` 执行 `KV[tgt] = KV[src]` |
| 新序列边界 | `new_seq_lens = old_seq_lens + accept_lens` |
| 容量约束 | `kv_committed_len <= kv_allocated_len` |

拒绝不要求立即物理 free。涉及压缩状态时，还要查清对象是 KV page 还是 compressor state，所有权属于 SWA page 还是请求，以及当前操作是防止 stale read 还是归还物理页。

接受路径错查行索引，搬移错查源目标，容量未收回查所有权和页边界，压缩读旧值查覆盖、清理。它们需要不同证据，不能统称 KV 没释放。

## 七、源码入口与计数对照

源码按输入、接受、搬移、回收列在下面。

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

将有效输出长度、输入 KV 边界、预留上界分别记录，再用 `R,A,C → A,C,X` 对照行和地址，才能判断哪段状态需要提交、保留或回收。
