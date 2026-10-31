# 从一次 Decode 读懂 SGLang：Tensor、缓存与下一枚 Token

读 Decode 可以从一枚刚采样的 Token 开始：下一轮把它送进模型，为它计算并写入缓存，读取本请求历史，经过 MoE、LM Head 与采样，再得到下一枚输出。

下面用三条请求的小 batch 走一轮。首先分清采样和执行：知道一个 Token ID 时，它还没有作为输入经过模型；下轮执行才产生它在各层的 KV。后面的长度、位置与缓存映射，都沿这个先后关系理解。

主线为普通非投机 Target Decode，源码固定到 SGLang commit <code>0f6761b54facebb47f2068f87ecccd8f14da3a0e</code>，核对日期 2026-09-21。采用 DeepSeek-V4，只展开执行链所需的 mHC、DSV4 Attention、MoE 和 Vocab Parallel LM Head。

## 一、上一轮的输出怎样成为本轮输入

假设某请求在本轮开始前已有 L 个位置的 KV，对应逻辑位置 `0,1,2,...,L-1`。

上一轮采样出 <code>y_L</code>，它将成为当前 <code>input_ids</code>。此前只选出了整数 ID，位置 L 的 K/V 尚未计算。因此，当前轮先安排缓存槽位，模型执行时再写入内容。

普通 Decode 的这些动作依次为：

~~~mermaid
flowchart TD
    A["已有 KV · position 0 ... L-1"]
    B["上一轮采样 y_L · 本轮 input_id"]
    C["alloc_for_decode · 给 position L 分配 KV 槽"]
    D["req_to_token[req,L] · 写入新 slot 映射"]
    E["seq_lens: L → L+1"]
    F["ForwardBatch · seq_lens=L+1 · position=L"]
    G["模型计算 y_L 的 Q/K/V"]
    H["新 KV 写入 out_cache_loc"]
    I["Attention 读取可见历史"]
    J["Logits → Sampler"]
    K["产生 y_(L+1)"]

    A --> C
    B --> C
    C --> D --> E --> F --> G --> H --> I --> J --> K
~~~

固定 <code>ScheduleBatch.prepare_for_decode()</code> 先执行：

~~~python
self.out_cache_loc = alloc_for_decode(
    self,
    token_per_req=1,
)
~~~

之后再推进长度：

~~~python
self.seq_lens = self.seq_lens + 1
self.seq_lens_cpu = self.seq_lens_cpu + 1
~~~

<code>alloc_for_decode()</code> 仍使用加一前的 L，将物理 slot 写入：

~~~text
req_to_token[
    request_slot,
    logical_position=L
]
=
new_slot
~~~

allocation 只为位置 L 安排空间和映射，各层 Attention 真正处理当前 Token 时，才形成并写入状态。不能由槽位已分配推断 KV 已计算完成。

到 <code>ForwardBatch.init_new()</code> 时，<code>seq_lens</code> 已为 L+1，普通 Decode 默认按 `positions = clamp(seq_lens - 1, min=0)` 得到位置。

因此本轮仍是 `positions = L`。

字段可按时间顺序对照：

| 字段 | 普通 Decode 当前语义 |
| --- | --- |
| 当前 Forward 前已有 KV | positions <code>0 ... L-1</code> |
| <code>input_ids[i]</code> | 上一轮采样、这一轮送入模型的 <code>y_L</code> |
| allocation 使用的旧长度 | <code>L</code> |
| 新映射的 logical position | <code>L</code> |
| <code>ForwardBatch.seq_lens[i]</code> | <code>L+1</code> |
| <code>ForwardBatch.positions[i]</code> | <code>L</code> |
| <code>out_cache_loc[i]</code> | 当前 input token 的新 KV 写入目标 |
| 本轮 Sampler 输出 | 下一轮使用的 <code>y_(L+1)</code> |

例如已有 10 个位置，旧 L=10，新输入位于 position 10，ForwardBatch 长度则为 11。位置从 0 编号，长度统计位置总数；这里的普通非投机 <code>ForwardBatch.seq_lens</code> 供 Kernel 使用，已经包含当前输入。

Draft / Verify 的长度另有语义，不能直接套用这条普通 Decode 关系。

## 二、把三条请求组织为 ForwardBatch

调度器用 ScheduleBatch 描述这一轮选中了哪些请求，再由 ForwardBatch 把设备执行所需的 Tensor 交给 ModelRunner。字段包括 `forward_mode`、`batch_size`、`input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc`、`seq_lens_sum`、`positions` 和 `sampling_info`。

先跟踪本轮输入及其位置，也就是 `input_ids` 与 `positions`，再接上请求身份和缓存目标。

三条请求都继续时，各自贡献一个新 Token。忽略 Graph padding，两个 Tensor 的形状为：

~~~text
input_ids   [3]
positions   [3]
~~~

三行可以有不同历史。例如原有 KV 长度分别为 10、7、20，本轮字段为：

| 本轮 Batch 行 | 原有 KV 长度 | 本轮 position | ForwardBatch.seq_lens |
| ---: | ---: | ---: | ---: |
| 0 | 10 | 10 | 11 |
| 1 | 7 | 7 | 8 |
| 2 | 20 | 20 | 21 |

这是一个假设例子：ID、position 都只有三行，各行访问的历史却不同，因此还需 `req_pool_indices` 和 `seq_lens` 记录请求身份与长度。

batch 行号仅是本轮排列。<code>req_pool_indices[i]</code> 选择 ReqToTokenPool 中的请求记录行，再由该行查历史，自己并非 KV 地址；<code>seq_lens[i]</code> 则向 Attention、页表逻辑提供本轮可见长度。

新增状态的写入目标由 `out_cache_loc` 描述。

写入坐标在执行边界还可能转换。<code>ForwardBatch.init_new()</code> 会调用：

~~~python
model_runner.kv_index_translator.rebind_write_loc(ret)
~~~

固定 KVIndexTranslator 区分 virtual、physical、kernel-facing 三种 ID。它们在普通非 unified pool 中可以相同，unified / DCP 等布局则会将 <code>ForwardBatch.out_cache_loc</code> 重新绑定为 kernel-facing FULL-side ID，并把原值保留为 <code>out_cache_loc_virtual</code>。

因此要连同阶段理解 <code>out_cache_loc</code>：ScheduleBatch 取得 canonical allocation，ForwardBatch 又可能转为 Kernel 接口编号。比较两个整数前，需先确定 ID 空间。

从请求到模型的交接可以画为：

~~~mermaid
flowchart TD
    R["Req · 请求生命周期"]
    SB["ScheduleBatch · 调度状态"]
    FB["ForwardBatch · 一次 Forward 快照"]
    MR["ModelRunner"]
    M["DeepSeek-V4"]

    R --> SB --> FB --> MR --> M
~~~

到 <code>ForwardBatch</code> 后，执行集合已经选定，接着需要跟踪设备收到的输入行、长度和缓存目标。

## 三、Embedding 与 mHC 的形状变化

本轮普通 Decode 取 `T=B=3`，`input_ids` 形状为 `[3]`。执行 Embedding：

~~~python
hidden_states = self.embed_tokens(input_ids)
~~~

以 DeepSeek-V4-Flash 的 hidden_size=4096 为例，得到 `[3,4096]` 的 hidden states。

此例 V4 的 `hc_mult = 4`，表示扩展为 `[T,4,H]`。T=3、H=4096 时，每 Token 保存四路状态，4 并非请求数或 Attention Head 数。mHC 在子层前将它们混为二维 Working Hidden，子层后再混回多路状态。

这里同时存在两种形状：持久 mHC State 为 `[T,4,H]`，Attention/MoE 使用的临时 Working Hidden 为 `[T,H]`。

一层内对应以下变化：

~~~mermaid
flowchart TD
    P["持久 mHC State · [T,4,H]"]
    PRE["hc_pre / combine"]
    W1["Working Hidden · [T,H]"]
    ATT["Attention"]
    W2["Working Hidden · [T,H]"]
    MID["hc_post + hc_pre / combine"]
    W3["Working Hidden · [T,H]"]
    MOE["MoE"]
    W4["Working Hidden · [T,H]"]
    POST["hc_post"]
    P2["持久 mHC State · [T,4,H]"]

    P --> PRE --> W1 --> ATT --> W2 --> MID --> W3 --> MOE --> W4 --> POST --> P2
~~~

<code>hc_pre()</code> 将四路 persistent state 混为子层所需 Working Hidden，同时维护 mixing coefficient；<code>hc_post()</code> 将 <code>[T,H]</code> 子层输出混回 <code>[T,4,H]</code>。

因此 Attention、MoE 收到 `[T,H]`，并不意味着四路状态丢失，前后混合操作仍在维护它。先跟清这两种形状，再读 Sinkhorn、post、comb 的数学，较容易确认每项作用。

## 四、当前 Token 写入缓存并读取历史

进入 Attention 的 Working Hidden 为 `[T,4096]`。

它产生当前 Query 和新增 KV，Query 的概念路径为：

~~~text
x [T,4096]
   ↓ wq_a
q_lora [T,1024]
   ↓ norm / wq_b
q [T, local_heads, 512]
~~~

随后写入新状态、读取历史：写由 `out_cache_loc` 定位，读由请求记录及页表定位。逻辑位置与物理槽位通过映射连接。下方展开 DSV4 细节，也可以先跳到 MoE。

<details markdown="1">
<summary>深入：Ascend DSV4 的写入、页表和压缩状态</summary>

### 写路径：当前 Token 的新状态写到哪？

位置 L 已在 allocation 阶段取得 canonical full-side slot。

Ascend DSV4 的 SWA 写入继续将其转为 SWA location：

~~~python
swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
    forward_batch.out_cache_loc
    # Full-side 到 SWA 的位置映射
)
~~~

再更新对应层：

~~~python
pool.set_swa_buffer(
    layer_id=layer_id,
    loc=swa_loc,
    cache=swa_k,
)
~~~

DSV4 专用分配还可通过 <code>out_cache_loc_dsv4</code> 提供 `out_c4_loc` 与 `out_c128_loc`。若当前逻辑位置完成相应 compression ratio 的 block，便有新的 C4/C128 压缩目标。

因此当前 Token 的状态还可能分属以下位置：

~~~text
canonical full-side allocation
        │
        ├── SWA write destination
        ├── 可选 C4 compressed destination
        └── 可选 C128 compressed destination
~~~

### 读路径：req_pool_indices 首先只是 Request row

<code>req_pool_indices</code> 先选择 <code>ReqToTokenPool.req_to_token</code> 的请求行：



~~~text
req_to_token[
    request_slot,
    logical_position
]
=
canonical token-slot id
~~~

Base Ascend Backend 从该行读取历史 slots，再按 page size 转换为 page ID，构造 Block Table。

这里依次定位请求身份、逻辑 Token 位置和物理缓存页。

### SWA、C4、C128 的 ownership 不是同一种映射

Ascend 910C DSV4 的几条映射可展开为：

~~~mermaid
flowchart TD
    R["req_pool_indices · 选择 Request row"]
    RT["req_to_token[req, logical_pos]"]
    FULL["Canonical Full-side token slots"]

    SM["Full → SWA mapping"]
    SPT["swa_page_table"]
    SP["SWA physical pool"]

    C4M["按 C4 ratio / page 派生"]
    C4T["c4_page_table"]
    C4P["C4 physical pool"]

    SIDE["req_to_c128_sidecar · Request-scoped sidecar"]
    C128T["c128_page_table"]
    C128P["独立 C128 physical pool"]

    R --> RT --> FULL
    FULL --> SM --> SPT --> SP
    FULL --> C4M --> C4T --> C4P

    R --> SIDE --> C128T --> C128P
~~~

SWA 页表沿 full slot 做 Full→SWA 映射。

C4 同样从本请求 <code>req_to_token</code> 记录出发，在 ratio-4 的逻辑边界派生页。

C128 则使用独立记录：

~~~python
req_to_c128_sidecar[
    req_pool_indices,
    :n_groups
]
~~~

它通过 request-scoped sidecar 管理物理页，不能简单把 full Token slot 除以 128。

最终 Kernel 一起消费 Query、sequence lengths、page table 和对应的物理 pool。

沿请求、逻辑位置、物理池查明历史后，再对照 Query、长度、页表等实参，才能检查是否读错位置或用了错误长度。

</details>

## 五、MoE 怎样分发并合并 Token

Attention 输出经过 mHC 和 Norm，MoE 前又得到 `[T,4096]` Working Hidden。本例三行仍各自代表当前 Token，需要分别决定专家去向。

公开配置有 256 个 routed experts，每 Token 选择 6 个。先看 Router：

~~~python
router_logits = self.gate(hidden_states)

topk_output = self.topk(
    hidden_states=hidden_states,
    router_logits=router_logits,
    # 其余参数省略
)
~~~

Router 将 `[T,4096]` 的 hidden states 投影成 `[T,256]` 的 router logits，再由 TopK 为每个 Token 选 6 个 Expert，决定该行交给哪些专家权重计算。



若权重在远端 Rank，Dispatch / All-to-All 将这行表示送给 Expert owner。

~~~mermaid
flowchart TD
    T["Token hidden rows · Token order"]
    R["Router / TopK"]
    D["Dispatch / A2A"]
    E["Expert-local rows · M_e × H"]
    G["Expert GEMM"]
    C["Combine"]
    O["恢复 Token order · [T,H]"]

    T --> R --> D --> E --> G --> C --> O
~~~

数据在此期间从 Token/Batch 行顺序转为专家任务顺序，再由 Combine 恢复原 Token 顺序。跨 Rank 的输入由 DeepEP/A2A 发给专家，贡献计算完后返回对应 Token。

例如某个专家可一起收到三条请求中被路由来的行，前馈计算仍分别处理它们。Combine 要恢复每份贡献的 Token 对应关系。ownership 描述当前由谁持有、处理数据，本地重排后的行号不能代替请求身份。

## 六、从 hidden states 到下一枚 Token

层栈结束后得到的还是 hidden states，需要继续投影到词表。

常见逻辑示意为：

~~~text
[B,H]
  ↓ LM Head
[B,V]
~~~

但词表并行时，本地 LM Head MatMul 先得到词表 shard：

~~~text
[rows,H]
×
[H,V_shard]

→

[rows,V_shard]
~~~

固定 DeepSeek-V4 在 PP 最后阶段构造 <code>ParallelLMHead</code>。启用 <code>enable_dp_lm_head</code> 则使用 Attention TP Group，否则通常采用全局 TP Group。

### DP Attention，但 LM Head 仍沿全局 TP 分片

全局 TP 的 gather/scatter 路径需同时处理两件事：DPA 分布的请求行、TP 分布的词表列。LogitsProcessor 的语义顺序如下：

~~~mermaid
flowchart TD
    L["本 DP rank hidden rows · [B_local,H]"]
    DG["DP-Attention row gather · 形成 global rows"]
    LM["各 TP rank 本地 LM Head · [B_global,V_shard]"]
    VG["TP gather vocab shards · [B_global,V]"]
    DS["DP row scatter · 恢复本地 Request rows"]
    O["Sampler 输入 · [B_local,V]"]

    L --> DG --> LM --> VG --> DS --> O
~~~

对应源码依次执行：

~~~text
_gather_dp_attn_hidden_states()
        ↓
_compute_lm_head()
        ↓
TP vocab-shard gather
        ↓
_scatter_dp_attn_logits()
~~~

先收集要计算的请求行，再拼齐词表分片，最后把各 DP 的请求行分回本地。完整 Vocabulary 是词表通信后的结果，请求归属恢复则是另一项操作。

同一版本另有条件控制的 LM Head All-to-All：开关、行布局等满足时，用一次重分布替代词表 gather 和 row scatter。实际路径需核对 `_can_use_tp_lm_head_all_to_all()`，图中序列并非所有配置必经。

### 开启 DP LM Head

DP LM Head 则让 <code>ParallelLMHead</code> 使用 <code>attn_tp_group</code>。

各 DP 保留 local rows，只在自己的 Attention TP 组内拼词表：

~~~text
[B_local,H]
    ↓ local LM Head
[B_local,V_shard]
    ↓ attn-TP vocab gather
[B_local,V]
~~~

<code>attn_tp_size=1</code> 时，本地 LM Head 已有完整 Vocabulary Weight，不再需要 vocab gather。

两种布局最终都需为本地请求提供完整词表分数，差别是沿哪个组分片权重，以及中间是否重新收集、分发请求行。

普通生成交给 Sampler 的 `next_token_logits` 形状为 `[#local_sequences,vocab_size]`。

### Sampler 默认不再同步最终 Token ID

Greedy 拿到完整本地 logits 后可直接取最大值：

~~~python
batch_next_token_ids = torch.argmax(
    logits,
    -1,
)
~~~

其他采样使用 <code>SamplingBatchInfo</code> 的 temperature、top-k、top-p、min-p、grammar 等参数。

Sampler 的 sync group 在普通模式下为 `tp_group`，DP Attention 下为 `attn_tp_group`。

固定默认并不逐轮同步最终 Token ID，而依赖末端 collective、LM Head MatMul 和 Sampling Kernel 在相关 Rank 保持确定性，以减少这次通信。

若 `SYNC_TOKEN_IDS_ACROSS_TP=1`，或当前 Batch 使用 Grammar，才执行 `MIN all_reduce(next_token_ids)`。



这次同步针对同一批请求的最终 ID，使用 <code>tp_sync_group</code>；DPA 下对应 <code>attn_tp_group</code>。不同 DP 处理不同请求，不能把它们的 ID 直接放在全局 TP 中归约。

采样结果再经调度器成为下一轮输入：

~~~mermaid
flowchart TD
    H["hidden · [B_local,H]"]
    LM["Vocab-sharded LM Head"]
    G["Logits gather / row restore"]
    L["full logits · [B_local,V]"]
    S["Sampler"]
    N["next_token_ids · [B_local]"]
    R["Scheduler relay"]
    I["下一轮 input_ids"]

    H --> LM --> G --> L --> S --> N --> R --> I
~~~

下一轮又从 L → L+1 的长度变化开始。

## 七、按这个例子读源码

实际阅读时，可先沿同一枚 Token 核对 Decode 状态与 KV allocation：

~~~text
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/mem_cache/allocation.py
python/sglang/srt/model_executor/forward_batch_info.py
~~~

第一遍只跟踪 `input_ids`、`seq_lens`、`positions`、`req_pool_indices`、`out_cache_loc`，将旧 L 代入，确认“分配逻辑位置 L → 长度变为 L+1 → 当前 position=L”。接着从 ForwardBatch 进入 Graph/Eager、Model Forward、Logits 和 Sampler，把本轮输入接回下轮输入。

读到 `deepseek_v4.py` 时，重点核对表示和行归属怎样变化：

| 执行位置 | 追踪重点 |
| --- | --- |
| Embedding 与 mHC | Embedding 为 `[T,H]`，持久 mHC 为 `[T,4,H]`，Attention/MoE Working Hidden 回到 `[T,H]` |
| Attention | 同一请求的逻辑位置怎样查到物理/cache page；C128 另查独立 sidecar |
| MoE | gate、topk、dispatch、experts、combine 前后，一行属于哪个 Token，专家贡献怎样返回 |
| LM Head 与 Sampler | `[T,H]` 怎样变为本地 `[rows,V_shard]`，何时取得 `[B_local,V]`，再生成 `next_token_ids` |

每到一个边界，都记录 Token 数、Tensor shape、请求身份、逻辑位置，以及读取的历史和写入的新状态。遇到 gather/scatter，先判断变化发生在 Token 行归属还是 Tensor/词表分片，再继续追踪消费结果的位置。

把三条请求代回去：ID 已采样，KV 先分配后写入，专家任务会重排行，词表投影又需要恢复本地请求。最后确认输出回到同一请求的下一轮输入。沿这些具体状态继续读，比单独记住每个函数名更容易发现错位。

## 源码阅读入口

### Decode 状态与 KV Allocation

- [ScheduleBatch](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/managers/schedule_batch.py)
- [KV allocation](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/mem_cache/allocation.py)
- [ForwardBatch](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/model_executor/forward_batch_info.py)

### ModelRunner / DeepSeek-V4

- [ModelRunner](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/model_executor/model_runner.py)
- [DeepSeek-V4 model](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/models/deepseek_v4.py)
- [DeepSeek MoE](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/models/deepseek_v2.py)

### Ascend 910C KV / Attention

- [Ascend DSV4 Attention Backend](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py)
- [Ascend DSV4 memory pool](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)
- [KVIndexTranslator](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/mem_cache/kv_index_translator.py)

### Logits / Sampling

- [Vocab Parallel LM Head](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/layers/vocab_parallel_embedding.py)
- [LogitsProcessor](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/layers/logits_processor.py)
- [SamplingBatchInfo](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/sampling/sampling_batch_info.py)
- [Sampler](https://github.com/sgl-project/sglang/blob/0f6761b54facebb47f2068f87ecccd8f14da3a0e/python/sglang/srt/layers/sampler.py)
