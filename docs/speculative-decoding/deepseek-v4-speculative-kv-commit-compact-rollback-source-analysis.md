# 投机解码 KV 管理：接受路径怎样提交、整理与回收

Verify 输入 `[R,A,B,C]` 已经写入四个 KV 槽，最后接受的却是 `R→A→C`。C 要从后面的槽搬到 B 的位置，输出则读取相同行的预测，得到 `A/C/X`。这次整理改变有效历史，持有的缓存容量却未必缩小。

Commit 确认历史，Compact 把选中状态排成下一轮需要的布局，Rollback 让未接受状态退出后续读取。它们可以通过边界、搬移、覆盖、清理共同完成，无需逐字节恢复快照。下面沿行索引、地址和释放边界，把这几步拆开。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 6880a4795533640f41ebb3db9e4ae0af5a371a1f`，核对日期为 2026-09-20。范围包括 Spec V2 EAGLE compaction、classic rejection、Target-only sampling、Ascend V4 C4/C128 生命周期。树索引整理属于通用 EAGLE；固定 V4 DSpark 使用 topk=1 链，提交方式单独说明。

</details>

## 一、Token、Verify 行、逻辑位置与物理槽位

第 4096 个位置，不一定写入 KV Cache 第 4096 行。Token ID、逻辑位置、物理 slot 通过不同数据表达。以下地址取教学数字，代码省略上下文，用来推演调用。

### Logical Sequence Position

有效输入 Prefix 的 `seq_len=4096`，下一个输入位于 4096，之后是 4097、4098。seq_len 在这里标记输入边界。

### `req_to_token`：Logical Position → Physical KV Slot

`req_to_token[req_pool_idx, logical_position]` 将逻辑位置映射到物理 slot。以 request #7 为例：

| 逻辑位置 | 物理 KV slot |
| ---: | ---: |
| 4094 | 18320 |
| 4095 | 18321 |
| 4096 | 42080 |
| 4097 | 42081 |
| 4098 | 42082 |

逻辑位置 4096 映射 slot 42080，前一个位置 4095 却在 18321。

映射表让逻辑历史连续，底下的物理区间可以分散。

<a id="out_cache_loc-forward-kv"></a>

### out_cache_loc 指定本次 KV 写入目标

Verify 处理多行，`batch.out_cache_loc` 逐行给出 KV 写入的物理 slot。

`eagle_prepare_for_verify()` 将输入和写入位置绑定：

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

Target 计算 logits、hidden states，也把这些行的 KV 写到 `out_cache_loc`。

候选先执行，Target 才能判断是否接受。因此物理上已写入的内容，比下一轮允许使用的正式历史更多。

<a id="accept_index-kv-slot-accepted-verify-row-path"></a>

### accept_index 选择 Verify 行

topk>1 的候选可以形成：

~~~text
                 root
              /        \
             A          B
           /   \      /   \
          C     D    E     F
~~~

接受 `root → B → E` 时，选中行未必连续，用 `accept_index` 收集。

选中行还要转换为源、目标地址：

| 搬移位置 | 怎样确定 |
| --- | --- |
| 源 slot | 由 `accept_index` 选出 Verify 行，再查 `out_cache_loc[row]` |
| 目标 slot | 由正式提交的逻辑位置查 `req_to_token` |

`accept_index` 找 Verify 行，`out_cache_loc` 找源 slot，`req_to_token` 找正式目标。三个整数分别处于行号、临时写入、正式映射的坐标中。

- [`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/eagle_utils.py)
- [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

## 二、接受结果怎样索引输出

新 bonus 来自最后有效输入行的预测，本轮还没有以它为输入的行。因此取得 `[draft1,draft2,draft3,bonus]`，索引必须从旧 root 开始。

<a id="greedy-tree-verify"></a>

### Greedy 从 root 行开始记录

Greedy Kernel 先把 root 的 retrieve index 放入 `accept_index[0]`。

每接受一个 child，就完成三次更新：

先将候选 ID 写入上一有效行的 predict，再把 child 行追加到 accept_index，最后更新 `last_accepted_global_idx`。

写成伪代码：

~~~python
accept_index[0] = root_row
last_row = root_row

for accepted_draft in path:
    predicts[last_row] = accepted_draft
    accept_index[next] = child_row
    last_row = child_row

predicts[last_row] = bonus
~~~

接受 d1、d2、d3 后，行索引与预测对应为：

| 有效索引顺序 | 被选中的输入行 | 该行预测 |
| ---: | --- | --- |
| 0 | root | d1 |
| 1 | d1 | d2 |
| 2 | d2 | d3 |
| 3 | d3 | bonus |

有效行是 `[root_row,row_after_d1,row_after_d2,row_after_d3]`，从 `predict[accept_index[:4]]` 读出 `[d1,d2,d3,bonus]`。

末行 KV 属于最后有效输入，末行 predict 属于下一个位置。索引选中的是承载预测的行。

<a id="accept_lens-correct-drafts-1"></a>

### accept_lens 包括末尾 Target 输出

`eagle_sample()` 内部数接受的 Draft，返回时再计 terminal Target/bonus：

~~~python
return predict, num_correct_drafts + 1, accept_index
~~~

`fill_bonus_tokens` 根据每请求 block stride，从 `accept_stride * pid + accept_len - 1` 提取末尾输出。

有效索引数为 accept_lens，其外仍可有 buffer 填充。

### Rejection Sampling 也保持相同契约

classic rejection 也先设 `accept_index[0] = root_row`，每通过一枚候选，就追加当前行、更新 `last_accepted_global_idx`。最后 residual / Target sampling 得到 terminal ID，写入 `Predicts[last_accepted_global_idx]`。

所以它仍通过 `predict[accept_index[:accept_lens]]` 返回已接受 Draft 加 terminal Token，与 Greedy 共享结果布局。

### Simulated accept 也遵循同一个布局

模拟接受也填入相同索引布局：

~~~python
sim_accept_index[:, :simulate_acc_len] = ...
num_correct_drafts.fill_(simulate_acc_len - 1)
~~~

real-draft-token 模式最后补入：

~~~python
bonus_node_indices = sim_accept_index[:, simulate_acc_len - 1]
predict[bonus_node_indices] = target_predict[...]
~~~

模拟值能检查下游布局是否正确，但不代表模型实际接受了这些候选，性能和质量报告仍需真实事件。

### NPU Sampling 的输入与结果消费

NPU 非 Greedy 从 `sgl_kernel_npu.sample` 导入 Tree/Chain Sampling Ops。

SGLang 上层的交接为：

传入 `predict / accept_index / accept_token_num`，随后 TP broadcast 同步结果；bonus 提取和 KV compaction 按 accept_lens 读取有效前缀。

这里核对的是调用与消费，设备核本体位于外部包。

- [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/spec_tree.py)
- [`chain_speculative_sampling_triton()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/reject_sampling.py)
- [`fill_bonus_tokens`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/speculative/eagle.py)
- [`generate_simulated_accept_index()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)

## 三、把接受路径映射到正式 KV

Target 返回 `predict, accept_lens, accept_index`，树路径 finalize 继续调用：

~~~python
move_accept_tokens_to_target_kvcache(
    batch,
    accept_index,
    accept_lens - 1,
    token_to_kv_pool_allocator,
)
~~~

下面先确定搬哪些输入，再沿源、目标地址看复制方向。

<a id="token-token-kv"></a>

### mover 接收的是已执行输入

mover 的第三参数为 `num_correct_drafts`，调用处传 `accept_lens-1`，内部再补旧 root。新 terminal bonus 还没有执行，不在待搬 KV 中。

代入 R、A/C、X：

| 本轮 Verify 行 | 该行已计算的 KV 属于 | 该行选中的输出 |
| --- | --- | --- |
| root 行 | R | A |
| A 行 | A | C |
| C 行 | C | X（新 bonus） |

输出 `[A, C, X]`，提交输入 KV `[R, A, C]`，长度同为 `accept_lens=3`。这两组 Token 来自同一组行的 predict 和输入状态。

减 1 去掉新 X，加 1 补旧 R，数量抵消，输入身份却已换回实际执行过的 Token。

### move_kv_cache 的目标参数在前

`tgt_cache_loc` 来自正式区间 `[seq_len, seq_len + accept_lens)` 的 `req_to_token`；`accept_out_cache_loc` 则经 `accept_index` 从 `batch.out_cache_loc` 取源。调用为：

~~~python
move_kv_cache(
    tgt_cache_loc,
    accept_out_cache_loc,
)
~~~

`move_kv_cache(tgt_loc, src_loc)` 执行 `KV[tgt] = KV[src]`。

paged KV 实现为：

~~~python
kb[pages_t, ..., offs_t, :] = kb[pages_s, ..., offs_s, :]
vb[pages_t, ..., offs_t, :] = vb[pages_s, ..., offs_s, :]
~~~

Unified MLA 使用相同方向：

~~~python
env[tgt_pages] = env[src_pages]
~~~

教学输入 `[R,A,B,C]` 写入 `[42080,42081,42082,42083]`，选中 `[0,1,3]`：

| 接受的输入 | Verify 行 | 源 slot | 正式逻辑位置 | 目标 slot |
| --- | ---: | ---: | ---: | ---: |
| R | 0 | 42080 | 4096 | 42080 |
| A | 1 | 42081 | 4097 | 42081 |
| C | 3 | 42083 | 4098 | 42082 |

前两行不用改位置，C 从 42083 复制到 42082，覆盖没有接受的 B。

- [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/speculative/spec_utils.py)
- [`KVCache.move_kv_cache()` implementations](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/memory_pool.py)

<a id="tree-path-compact-predict-hidden_states"></a>

### predict 与 hidden 随接受路径整理

搬完 KV，树数组仍可能排成 `[root, A, B, C, D, E, F]`。

有效路径若为 `[root, B, E]`，下轮 Extend 需要把它看作连续链：

因此继续执行：

~~~python
predict = _compact_accept_to_front(...)

logits_output.hidden_states =
    _compact_accept_to_front(...)
~~~

`_compact_accept_to_front()` 按接受索引 gather，放到每请求 block 前部。

整个过程先选树行，再搬池地址，最后整理 predict、hidden 的 Tensor 行。三步分别改变索引、存储地址、下游布局。

KV relocation 与 Tensor compaction 要选择同一路。只搬对 KV，hidden 却留在原分支，下轮 Extend 仍会把 Token 和特征配错。

<a id="deepseek-v4-dspark-tree-compact"></a>

### DSpark 通过链前缀确定提交范围

DSpark 参数固定 `speculative_num_steps=1`、`speculative_eagle_topk=1`。

候选宽度由 gamma 给出，`verify_window = gamma + 1`。

Worker 对提交的注释为：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~

EAGLE 树用行索引找非连续路径，DSpark 链则从前缀长度确定范围。两者接受长度相同，也可能使用不同的状态整理方式。

## 四、被拒绝候选与预留空间的生命周期

接受结果先排除无效历史，allocator 稍后才可能归还空间。哪些尾部可以释放，还取决于容量归谁持有、分页边界和下轮复用。

<a id="kv_committed_len-kv_allocated_len"></a>

### committed 与 allocated 分别记录历史和容量

每请求的两条边界为：

~~~text
0                                              kv_allocated_len
|----------------------------------------------------|
|             committed              |   overshoot   |
|====================================|...............|
                                     ▲
                              kv_committed_len
~~~

`kv_committed_len` 标出有效输入，`kv_allocated_len` 标出持有容量。中间 reserve / overshoot 可以保留旧候选的物理数据，下一轮读取仍由正式历史边界限制。

<a id="decode-reject-allocated-tail"></a>

### Decode 优先复用已分配尾部

正常迭代计算：

~~~python
cur = r.kv.kv_allocated_len

nxt = max(
    cur,
    page_align(r.kv.kv_committed_len + reserve),
)
~~~

`nxt >= cur` 让已有容量优先复用，不足再扩展，避免每轮 reject 后释放、下轮又分配。`get_alloc_reserve_per_decode() = 2 * get_alloc_len_per_decode()` 还用 double buffer 覆盖 overlap 中 Host committed 的滞后。

拒绝的候选因此可以留在 allocated 区间，等待下轮覆盖。

例如 allocated=4352、committed=4099、reserve=16、allocator page_size=128。`4099+16=4115` 向上对齐到 4224，再取 `max(4352,4224)`，持有容量仍为 4352。这里未缩小符合预留策略；泄漏要沿结束或转移所有权之后的回收判断。

### Overshoot 的实际释放时机

预留空间在这些生命周期转换中处理：

| 生命周期点 | 实际动作 |
| --- | --- |
| 正常 speculative reject | 通常**不立即 free**；只是不推进 committed ownership，tail 留作 reserve |
| Request finish / abort / normal release | `release_kv_cache()` 在 cache-owned/owned range 处理后，释放 `owned_kv_len → kv_allocated_len` 的 overallocated tail |
| Decode retract / preempt | `release_req()` 最终调用 `release_kv_cache(..., is_insert=False)`，设备侧 request KV（包括 overshoot）被释放；需要恢复的路径先做 backup/offload |
| StreamingSession finish overshoot | `_trim_overshoot()` 按 authoritative `finished_len` 释放超出下一轮真实输入的 tail |
| StreamingSession match/rewind | `_free_tail()` 在重新 extend 前释放旧 session row 中 `[prefix_len, kv_allocated_len)` 的 orphaned tail |
| 整个 session 释放 | session-owned row 最终整体回收 |

普通 reject 只排除历史。结束请求可能把前缀交给 cache，session 也可能继续持有；某请求放弃所有权以后，空间仍可能属于其他持有者。

### 页分配下的回收边界

分页释放要保留仍含有效数据的部分页。直接从首个无效 slot 释放，可能同时释放同页有效行。

相关实现先把起点对齐到 `ceil_align(boundary, page_size)`。

这里用 allocator 的真实 page_size，DCP 下它可能大于 Attention kernel 页。4099 按 128 对齐，安全起点为 4224；allocated=4352，可释放 `[4224,4352)`。而 `[4099,4224)` 的 125 个位置仍和有效状态同页，不能单独归还。

因此逻辑排除、所有权结束、整页归还，是三个可以错开的时刻。

- [`allocation_sizing.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/allocation_sizing.py)
- [`release_kv_cache()` / `_release_overallocated_kv_indices()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/common.py)
- [`release_req()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/managers/schedule_batch.py)
- [`StreamingSession._trim_overshoot()` / `_free_tail()`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/session/streaming_session.py)

## 五、Ascend 的 KV 页与压缩状态清理

V4 除了 Full、SWA、C4、C128 KV，还维护 C4/C128 compressor state。

KV 供 Attention 读历史，compressor state 供后续压缩。两者的寻址、所有权不同，清理一种不会自动处理另一种。

<a id="c4-state-ownership-npu"></a>

### C4 state 的架构分流

固定 NPU pool 根据架构选择：

| 路径 | C4 compressor state 的寻址与注册 |
| --- | --- |
| pre-A5 / EXPLICIT | 跟随 SWA physical page/state index，纳入 SWA state buffers |
| A5 / CYCLE | 使用 `req_pool_idx` 对应的独立环形状态，通过 `get_c4_state_buf_infos()` 单独注册 |

C128 state 按请求、绝对位置维护 ring。pre-A5 C4 随页，A5/CYCLE C4 则使用独立请求 ring，清理时首先要识别实际状态归属。

<a id="clear_unaccepted_c128_draft_states-state-c128-kv-pages"></a>

### clear_unaccepted_c128_draft_states 无效化压缩状态

共享 V4 pool 的 `clear_unaccepted_c128_draft_states()` 处理拒绝的 compressor state，注释区分了 C128、C4：

~~~text
C128 compression can read rejected draft slots at a boundary;
C4 overwrites its draft slots before reading them.
~~~

清理按 draft_offset 判断：

~~~text
if draft_offset < accept_len:
    keep

else:
    slot = (seq_len + draft_offset) % ring_size
    clear that C128 state row
~~~

无效 state 前半写 0，后半写 `-inf`。

这是原地无效化，用于防止后续压缩读取旧候选。线性窗口 `accept_len=3` 保留 offset 0、1、2，从 3 开始清理；计数包括 root。

state 仍占原容量，C128 KV page 也没有由这一步释放。

<a id="c4-rejected-state-clear"></a>

### C4 在读取前覆盖 draft slots

C4 的注释为 `C4 overwrites its draft slots before reading them.`，所以没有采用同一 rejected-offset clear。

当前 stale-state 风险由覆盖顺序处理，页、请求 Ring 的其他生命周期仍各自管理。

具体清理要继续区分 pre-A5 页绑定与 A5/CYCLE 请求状态。

### C128 KV 页的引用计数与释放

C128 KV 使用独立 allocator；`req_to_c128_sidecar` 保存请求引用的物理页。

`release_c128_pages()` 减少引用计数，归零才把页交还 `c128_attn_allocator`。

请求 row 释放还经过：

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

`replace_req_c128_prefix()` 替换前缀 sidecar 时，也会释放旧页引用、retain 新引用。

两个持有者使计数为 2，释放一个后为 1，最后归零才还页。前缀替换需要正确增减这些引用。

因此页回收跟随 sidecar 引用生命周期，单条请求的 `accept_lens` 无法确定哪些物理页已无人使用。

### SWA 释放与 C128 引用计数的关系

free API 也按信息范围分开：

~~~text
free(free_index)
→ full + SWA only

free(req=..., req_to_token_pool=...)
→ 有 request identity
→ 才能处理 C128 sidecar / request-scoped state
~~~

generic full slot 不包含请求 sidecar 引用身份，带 req 的释放才提供这一层信息。

`tree_cache.free_kv_row(...)` 因而不能概括为同时释放 Full/SWA/C4/C128。它的参数、请求 row、sidecar 引用和请求 state，需要沿对应调用继续追踪。

<a id="ascend-online_c128-npu"></a>

### ONLINE_C128 被固定 NPU pool 拒绝

NPU pool 初始化拒绝 `ONLINE_C128` 与 `ratio == 128` 的组合，注释为：

~~~text
ONLINE_C128 is CUDA-only;
NPU fused compressor has no online mode.
~~~

基类中这段跳过清理的逻辑：

~~~python
if ONLINE_C128:
    return
~~~

属于 CUDA ONLINE_C128，不适用于本篇固定 NPU 路径。

### EAGLE Cleanup 与 DSpark 提交路径

同名 C128 clear 显式出现在 EAGLE `run_eagle_verify()` 中。

同版本 DSpark Worker / Verify 没有找到同名直接调用；它用 commit_lens、hidden 注入限定提交范围。是否仍可能读到未接受 state，需要继续沿它自己的写入、覆盖、提交路径验证，仅凭没有同名调用还不能判断正确或错误。

- [`deepseek_v4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
- [`c128_cleanup.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/kernels/ops/attention/dsv4/c128_cleanup.py)
- [`dsv4_memory_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)
- [`dsv4_allocator.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_allocator.py)
- [`dsv4_req_to_token_pool.py`](https://github.com/sgl-project/sglang/blob/6880a4795533640f41ebb3db9e4ae0af5a371a1f/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_req_to_token_pool.py)

## 六、串起 Commit、Compact 与 Rollback

将地址整理、容量预留和状态清理放回一轮执行：

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

图中的 Rollback 由多种操作共同完成：逻辑不提交、接受路径搬移、Tensor compaction、预留区复用、生命周期回收、后端状态清理。

容量可以大于有效历史。判断是否回滚正确，要看下一轮能否只读到对应的 Token、KV、hidden、compressor state。

<a id="debug"></a>

### KV 的索引、地址与所有权排查

逐轮日志保留这些关系，并用有效长度排除填充：

| 要核对的量 | 关系或方向 |
| --- | --- |
| 输出长度 | `accept_lens = num_correct_drafts + 1` |
| 有效索引数 | `valid_accept_index_count = accept_lens` |
| 有效输出 | `predict[accept_index[:accept_lens]]` 为已接受 Draft 与 terminal bonus |
| 搬移方向 | `move_kv_cache(tgt, src)` 执行 `KV[tgt] = KV[src]` |
| 新序列边界 | `new_seq_lens = old_seq_lens + accept_lens` |
| 容量约束 | `kv_committed_len <= kv_allocated_len` |

压缩状态还需标出对象是 KV page 还是 compressor state，归 SWA page 还是请求持有，当前操作负责防 stale read 还是物理回收。

接受路径错误，查行索引；搬移错误，查源目标；容量没回收，查所有权、页边界；压缩读取旧值，查覆盖与清理。它们分别发生在状态生命周期的不同位置。

## 七、源码入口与计数对照

以下入口按输入、接受、搬移、回收排列。

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
