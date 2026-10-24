# 从一次 Decode 读懂 SGLang：Tensor、缓存与下一枚 Token

读 SGLang 的 Decode 路径，可以先跟住一枚刚采样出来的 Token：下一轮它成为模型输入，写入自己的 KV，读取请求历史，再经过 MoE 和 LM Head，产生下一枚 Token。

下面用三条请求组成的小 Batch 走完一轮。先把一个容易混淆的时间点分清：Token 已经采样出来，只表示知道了它的 ID；下一次送进模型，才会算出它在各层的 KV。沿着这个区别，再看序列长度、输入 Tensor 和缓存映射，最后接到 MoE、LM Head 与下一次采样。

本文讨论普通、非投机 Target Decode，固定到 SGLang commit <code>0f6761b54facebb47f2068f87ecccd8f14da3a0e</code>，核对日期为 2026-09-21。模型使用 DeepSeek-V4，正文只保留理解执行链需要的 V4 特性：mHC、DSV4 Attention、MoE 和 Vocab Parallel LM Head。

## 一、上一轮的输出怎样成为本轮输入

先固定一轮 Decode 的输入与输出。假设某个 Request 在这一轮开始前已经有 L 个位置的历史 KV：

~~~text
logical position:
0, 1, 2, ..., L-1
~~~

上一轮已经采样出了新 Token <code>y_L</code>，它会成为这一轮 Decode 的 <code>input_ids</code>。采样只选出了一个整数 ID，没有让这个新 Token 经过模型；所以在当前 Forward 之前，位置 L 的 K/V 内容还没有算出来。接下来要先给这个位置安排缓存空间，再由模型写入本轮计算结果。

一轮普通 Decode 的时间线更接近：

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

当前 <code>ScheduleBatch.prepare_for_decode()</code> 先调用：

~~~python
self.out_cache_loc = alloc_for_decode(
    self,
    token_per_req=1,
)
~~~

随后才执行：

~~~python
self.seq_lens = self.seq_lens + 1
self.seq_lens_cpu = self.seq_lens_cpu + 1
~~~

而 <code>alloc_for_decode()</code> 使用的仍是加一前的旧长度 L，并把新分配的物理 slot 写到：

~~~text
req_to_token[
    request_slot,
    logical_position=L
]
=
new_slot
~~~

这里要区分“分配位置”和“写入内容”：allocation 先为逻辑位置 L 安排可写的槽位，并建立映射；当前 Token 真正经过各层 Attention 时，才写入对应层的新状态。看到分配完成，不代表这枚 Token 的 KV 已经计算完。

等到 <code>ForwardBatch.init_new()</code> 构造时，<code>seq_lens</code> 已经是 L+1。普通 Decode 的 Position 默认由：`positions = clamp(seq_lens - 1, min=0)`

得到，因此：`positions = L`

把几个字段放在一张表里最清楚：

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

例如本轮之前已有 10 个位置的 KV，旧长度 L=10；本轮输入放在位置 10，分配完成后传给 ForwardBatch 的长度是 11，position 仍是 10。长度统计有多少个位置，position 则从 0 开始编号。对这里的普通非投机 Decode，<code>ForwardBatch.seq_lens</code> 是给 Kernel 使用的序列长度，已经包含本轮输入位置。

这条关系不要机械套到 Draft / Verify 等投机模式，它们有自己的长度语义。

## 二、把三条请求组织为 ForwardBatch

准备好每条请求的长度和缓存位置后，调度侧还要把这些状态交给模型执行侧。阅读时先认清这个边界：

~~~text
ScheduleBatch
→
ForwardBatch
~~~

可以把它们理解成：

~~~text
ScheduleBatch
回答：
“这一轮谁运行？”

ForwardBatch
回答：
“这些 Request 已经决定运行，
模型 Forward 真正需要哪些 Tensor？”
~~~

当前 <code>ForwardBatch</code> 的核心字段直接包括：

~~~text
forward_mode
batch_size
input_ids
req_pool_indices
seq_lens
out_cache_loc
seq_lens_sum
positions
sampling_info
~~~

第一次读可以先按三个用途理解这些字段：本轮输入是什么，它属于哪条请求，以及新状态写到哪里。

第一组是 Token 本身：

~~~text
input_ids
positions
~~~

设三条请求本轮都继续生成，各提供一个输入 Token。不计 Graph 填充时，这两个 Tensor 的形状是：

~~~text
input_ids   [3]
positions   [3]
~~~

三条请求不需要有相同的历史长度。例如本轮开始前分别已有 10、7、20 个位置的 KV，按上一节的规则准备后：

| 本轮 Batch 行 | 原有 KV 长度 | 本轮 position | ForwardBatch.seq_lens |
| ---: | ---: | ---: | ---: |
| 0 | 10 | 10 | 11 |
| 1 | 7 | 7 | 8 |
| 2 | 20 | 20 | 21 |

这是说明字段关系的假设例子。`input_ids` 与 `positions` 都只有三行，但每行背后要读取的历史不同，因此还需要第二组字段描述 Request 与历史状态：

~~~text
req_pool_indices
seq_lens
~~~

Batch 的第 i 行只是本轮排列位置，请求可能在下一轮被重排。<code>req_pool_indices[i]</code> 用于找到这条请求在 ReqToTokenPool 中的记录行，再由那一行查历史缓存；它本身不是 KV 地址。<code>seq_lens[i]</code> 则告诉 Attention 和页表构造逻辑，这条请求本轮有多少个可见位置。

第三组是当前 Token 的新状态写入坐标：`out_cache_loc`

这里还有一个更底层的边界。<code>ForwardBatch.init_new()</code> 会调用：

~~~python
model_runner.kv_index_translator.rebind_write_loc(ret)
~~~

SGLang 当前 KVIndexTranslator 明确区分：

~~~text
virtual
physical
kernel-facing
~~~

三种 ID 空间。在普通非 unified pool 上三者可以重合；在 unified / DCP 等布局下，<code>ForwardBatch.out_cache_loc</code> 会被重新绑定为 kernel-facing FULL-side ID，原始值保存在 <code>out_cache_loc_virtual</code>。

因此，日志里同名的 <code>out_cache_loc</code> 要连同阶段一起看：ScheduleBatch 中来自分配器的统一分配编号（canonical allocation），进入 ForwardBatch 后还可能转换成 Kernel 接口需要的编号。比较两处整数是否相同之前，先确认它们属于哪个 ID 空间。

到这里，整个控制边界可以画成：

~~~mermaid
flowchart TD
    R["Req · 请求生命周期"]
    SB["ScheduleBatch · 调度状态"]
    FB["ForwardBatch · 一次 Forward 快照"]
    MR["ModelRunner"]
    M["DeepSeek-V4"]

    R --> SB --> FB --> MR --> M
~~~

从 <code>ForwardBatch</code> 往下，问题已经从“本轮选中了哪些请求”变成“设备拿到哪些 Tensor”。后面就跟着这三条请求的输入行、长度和缓存位置，看它们怎样进入 DeepSeek-V4。

## 三、Embedding 与 mHC 的形状变化

假设普通 Decode 当前：`T = B = 3`

那么：

~~~text
input_ids
[3]
~~~

经过：

~~~python
hidden_states = self.embed_tokens(input_ids)
~~~

这里采用 DeepSeek-V4-Flash 的 `hidden_size=4096` 作为具体示例，于是先得到：

~~~text
[3]
  ↓ Embedding
[3,4096]
~~~

当前 V4 的 `hc_mult = 4`，接下来会把表示扩展成 `[T,4,H]`。以 T=3、H=4096 为例，就是三条 Token 行各保存四路表示；这里的 4 不是四条请求，也不是 Attention Head 数。Attention 和 MoE 计算前，mHC 先把多路表示混合成二维 Working Hidden，子层算完后再把结果混回多路状态。

对第一次阅读，只需要区分两种 Tensor：

~~~text
Persistent mHC State
[T,4,H]
~~~

与：

~~~text
Working Hidden
[T,H]
~~~

一层里的 shape 主线是：

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

<code>hc_pre()</code> 把四路 persistent state 混成当前子层消费的二维 Working Hidden，并维护内部 mixing coefficient；<code>hc_post()</code> 再把子层产生的 <code>[T,H]</code> 输出混回 <code>[T,4,H]</code>。

所以读到 Attention 或 MoE 的 `[T,H]` 输入，并不意味着四路 mHC 状态被丢弃了。多路状态仍由子层前后的混合操作维护，子层只接收这次计算需要的二维表示。先把这个形状变化跟住，再看 Sinkhorn、post、comb 的具体数学，会更容易分清各项作用。

## 四、当前 Token 写入缓存并读取历史

现在 Working Hidden 是：`[T,4096]`

DeepSeek-V4 Attention 会从中构造 Query，并为当前 Token 产生新的 KV 状态。Q 路径可以先概念化成：

~~~text
x [T,4096]
   ↓ wq_a
q_lora [T,1024]
   ↓ norm / wq_b
q [T, local_heads, 512]
~~~

Query 准备好之后，Attention 既要保存本轮新状态，也要读取请求历史。写入从 `out_cache_loc` 找目标位置，读取则从请求记录和页表找历史位置。逻辑位置说明它是序列中的第几个 Token，物理槽位说明数据实际放在哪里；两者需要映射，不能混用。下面的折叠部分展开 DSV4 的具体映射，第一次阅读也可以先跳到 MoE。

<details markdown="1">
<summary>深入：Ascend DSV4 的写入、页表和压缩状态</summary>

### 写路径：当前 Token 的新状态写到哪？

前面已经知道，逻辑位置 L 在 allocation 阶段拥有一个新的 canonical full-side slot。

在当前 Ascend DSV4 Backend 中，SWA 写入会进一步把 Full-side location 转成 SWA location：

~~~python
swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
    forward_batch.out_cache_loc
    # Full-side 到 SWA 的位置映射
)
~~~

再写：

~~~python
pool.set_swa_buffer(
    layer_id=layer_id,
    loc=swa_loc,
    cache=swa_k,
)
~~~

DSV4-specific allocation 还可能在 <code>out_cache_loc_dsv4</code> 里携带：

~~~text
out_c4_loc
out_c128_loc
~~~

如果当前 logical position 正好完成对应 compression ratio 的 block，就会有新的 C4 / C128 compressed destination。

因此一个 Token 的状态可能同时涉及：

~~~text
canonical full-side allocation
        │
        ├── SWA write destination
        ├── 可选 C4 compressed destination
        └── 可选 C128 compressed destination
~~~

### 读路径：req_pool_indices 首先只是 Request row

<code>req_pool_indices</code> 不直接指向 KV Tensor。它先选择：`ReqToTokenPool.req_to_token`

的一行：

~~~text
req_to_token[
    request_slot,
    logical_position
]
=
canonical token-slot id
~~~

Base Ascend Backend 构造 Block Table 时，就是从对应 Request row 取出历史 slot，再按 page size 转成 page id。

所以三层语义应该分开：

~~~text
Request identity
    ↓
Logical token positions
    ↓
Physical cache pages
~~~

### SWA、C4、C128 的 ownership 不是同一种映射

当前 Ascend 910C DSV4 更适合画成：

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

SWA Page Table 可以沿 Full slot 做 Full→SWA 映射。

C4 Page Table 也从该 Request 的 <code>req_to_token</code> 记录行出发，在 ratio-4 对应的逻辑边界派生 C4 page。

但 C128 当前使用独立的：

~~~python
req_to_c128_sidecar[
    req_pool_indices,
    :n_groups
]
~~~

所以 C128 physical ownership 不是简单把 full token slot 除以 128，而有单独的 request-scoped sidecar page mapping。

最后 Attention Kernel 消费的是：

~~~text
Query
+
seq lengths
+
page table
+
对应 physical pool
~~~

沿这三步回查，就能回答“当前请求应该读取哪段历史”。再对照 Query、长度、页表和物理池传入算子的参数，才容易发现长度或映射是否用错。

</details>

## 五、MoE 怎样分发并合并 Token

Attention 的结果经过 mHC 混合与 Norm，进入 MoE 前又形成 `[T,4096]` 的 Working Hidden。仍沿用三条请求的例子，此时每行 Token 表示都要分别选择专家。

当前公开配置有 256 routed experts，每个 Token 选择 6 个 routed experts。第一步是 Router：

~~~python
router_logits = self.gate(hidden_states)

topk_output = self.topk(
    hidden_states=hidden_states,
    router_logits=router_logits,
    # 其余参数省略
)
~~~

~~~text
hidden_states
[T,4096]

    ↓ Router

router_logits
[T,256]

    ↓ TopK

每个 Token 选 6 个 Expert
~~~

Router 回答的是：

> 这行 hidden state 接下来应该交给哪些 Expert 权重计算？

如果选中的 Expert 不在当前 Rank，就必须把这行 hidden state 发给 Expert owner，于是出现 Dispatch / All-to-All。

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

Token 在这几步中改变数据归属：

~~~text
进入 MoE 前：
按 Token / Batch row 组织

Router 后：
按 Expert owner 组织

Expert 算完：
恢复 Token order
~~~

Token 与目标 Expert 位于不同 Rank 时，DeepEP / A2A 将输入送到 Expert，再把计算贡献送回。

例如三条 Token 行分别选中多个专家，Dispatch 后，某个 Rank 上可能聚集的是来自不同请求、但交给同一专家的行。专家算完后，Combine 要把这些计算贡献合回原 Token 行，才能让下一层继续对应正确的请求。这里的 ownership 描述数据由哪个 Rank 持有和处理，不能仅用当前 Tensor 的行号代替请求身份。

## 六、从 hidden states 到下一枚 Token

所有 Transformer Layer 结束后，模型拥有的仍然只是 hidden state。

教科书常画：

~~~text
[B,H]
  ↓ LM Head
[B,V]
~~~

但 SGLang 的 LM Head 可以做 Vocabulary Parallel，所以本地 MatMul 首先更像：

~~~text
[rows,H]
×
[H,V_shard]

→

[rows,V_shard]
~~~

当前 DeepSeek-V4 在 PP 最后一段构造 <code>ParallelLMHead</code>。如果开启 <code>enable_dp_lm_head</code>，它使用 Attention TP Group；否则通常使用全局 TP Group。

### DP Attention，但 LM Head 仍沿全局 TP 分片

先看全局 TP 分片下的 gather/scatter 路径。它需要协调两种维度：DP 划分的请求行，以及 TP 划分的词表列。对应的 LogitsProcessor 语义顺序是：

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

源码顺序就是：

~~~text
_gather_dp_attn_hidden_states()
        ↓
_compute_lm_head()
        ↓
TP vocab-shard gather
        ↓
_scatter_dp_attn_logits()
~~~

这条路径先让各 TP Rank 拿到需要计算的请求行，再拼齐词表分片，最后各 DP Rank 取回自己的请求行。完整 Vocabulary 维度是在词表分片通信后才出现的，恢复请求行归属则是另一步。

同一固定版本还保留了受条件控制的 LM Head All-to-All 分支：当开关和行布局等条件满足时，用一次重分布替代这里的词表 gather 与后续 row scatter。阅读实际调用时，应核对 `_can_use_tp_lm_head_all_to_all()`，不能把上图当成所有配置都必经的通信序列。

### 开启 DP LM Head

如果开启 DP LM Head，ParallelLMHead 改用 <code>attn_tp_group</code>。

每个 DP replica 保留自己的 local rows，只在它自己的 Attention TP Group 内拼 Vocabulary：

~~~text
[B_local,H]
    ↓ local LM Head
[B_local,V_shard]
    ↓ attn-TP vocab gather
[B_local,V]
~~~

如果 <code>attn_tp_size=1</code>，这一 DP replica 的 LM Head 本身就是完整 Vocabulary Weight，也不需要 vocab gather。

两种布局最终都要满足同一个输出要求：每条本地请求拿到完整词表的分数。区别在于 LM Head 权重沿哪个组切分，以及中间是否需要收集和重新分发请求行。

普通生成最终向 Sampler 暴露：

~~~text
next_token_logits
[#local_sequences, vocab_size]
~~~

### Sampler 默认不再同步最终 Token ID

Sampler 拿到完整的本地 Request logits 后，Greedy 可以直接：

~~~python
batch_next_token_ids = torch.argmax(
    logits,
    -1,
)
~~~

非 Greedy 则按 <code>SamplingBatchInfo</code> 中的 temperature、top-k、top-p、min-p、grammar 等参数采样。

Sampler 的 sync group 是：

~~~text
普通模式：
tp_group

DP Attention：
attn_tp_group
~~~

但为了性能，SGLang 默认不会每轮再同步最终 token id。它依赖最后的 collective、LM Head MatMul 和 Sampling Kernel 在相关 Rank 上保持确定性。

只有：`SYNC_TOKEN_IDS_ACROSS_TP=1`

或者当前 Batch 使用 Grammar 时，才执行：`MIN all_reduce(next_token_ids)`

这里同步的是同一批请求的最终 Token ID，Group 为 <code>tp_sync_group</code>。DP Attention 下它对应 <code>attn_tp_group</code>；不同 DP Rank 处理不同请求，不能把它们的 Token ID 放进全局 TP 组直接归约。

到这里，自回归循环闭环：

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

下一轮 Decode 再从第一章的 L → L+1 时间线开始。

## 七、按这个例子读源码

实际打开源码时，可以沿前面那枚 Token 分几次读完。先看 Decode 状态与 KV Allocation：

~~~text
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/mem_cache/allocation.py
python/sglang/srt/model_executor/forward_batch_info.py
~~~

只追：

~~~text
input_ids
seq_lens
positions
req_pool_indices
out_cache_loc
~~~

尤其亲自走一遍：

~~~text
old L
→ alloc logical position L
→ seq_lens=L+1
→ positions=L
~~~

长度与槽位对应上之后，再看 ModelRunner 怎样把这份输入交给执行路径：

~~~text
ForwardBatch
→ Graph / Eager execution
→ Model Forward
→ Logits
→ Sampler
~~~

进入 <code>deepseek_v4.py</code> 时，先跟住三种形状，避免把四路 mHC 状态和二维子层输入混在一起：

~~~text
Embedding:          [T,H]
Persistent mHC:     [T,4,H]
Attention / MoE:    [T,H]
~~~

回到 Ascend 910C 缓存路径时，逐步对应：

~~~text
哪个 Request？
    ↓
哪个 logical position？
    ↓
哪个 physical/cache page？
~~~

C128 则由独立 sidecar 管理。

MoE 则沿下面五步追踪 Token 行的去向：

~~~text
gate
topk
dispatch
experts
combine
~~~

最后把 Logits / Sampling 的输出接回下一轮输入：

~~~text
[T,H]
→ [rows,V_shard]
→ [B_local,V]
→ next_token_ids
~~~

这样从本轮输入出发，就能追到下一轮要使用的 Token ID。

沿调用链阅读时，可以用下面几项核对当前数据：

~~~text
当前处理多少 Token？
当前 Tensor shape 是什么？
它属于哪个 Request？
当前 logical position 是多少？
读哪个历史状态？
写哪个新状态？
这里发生的是 Token ownership 变化，
还是 Tensor/Vocab shard 变化？
什么时候拿到完整 Vocabulary？
Sample 出来的 Token 怎样成为下一轮 input？
~~~

遇到不确定的字段时，可以把三条请求的例子填进去：本轮输入 ID 已确定，新 KV 先分配再写入；MoE 中间按专家重排，Logits 阶段再拼齐词表并恢复本地请求行。最后核对 Sampler 的输出是否回到了同一条请求的下一轮输入。每次只追清这一段状态变化，就能继续沿调用链往下读。

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
