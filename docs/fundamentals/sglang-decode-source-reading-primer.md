# 第一次读 SGLang 源码，应该先看懂什么？用 DeepSeek-V4 一次 Decode 串起 Tensor、KV Cache、Attention、MoE 与 Sampling

第一次打开 SGLang 源码，很容易被类名淹没：Scheduler、Req、ScheduleBatch、ForwardBatch、ModelRunner、ReqToTokenPool、TokenToKVPool、LogitsProcessor、Sampler……如果按文件树从上到下读，很快就会失去方向。

更有效的办法不是先背类名，而是只跟踪一个普通 Decode Step：

> **上一轮刚采样出的 Token，这一轮怎样进入模型；它的 KV 写到哪里；Query 怎样找到自己的历史；Attention 和 MoE 怎样改变 Tensor；最后 hidden state 又怎样变成下一枚 Token？**

只要这条主线走通：

~~~text
sampled token
    ↓
ScheduleBatch
    ↓
ForwardBatch
    ↓
Embedding
    ↓
mHC working hidden
    ↓
Attention + KV Cache
    ↓
MoE
    ↓
LM Head / LogitsProcessor
    ↓
Sampler
    ↓
next_token_ids
    ↓
下一轮 Decode
~~~

Continuous Batching、Paged / Radix Cache、TP / DP / EP、DeepEP、NPU Graph、投机解码与 PD 分离，就会变成这条主链不同边界上的优化，而不是互不相关的术语。

本文讨论普通、非投机 Target Decode，固定到 SGLang commit <code>0f6761b54facebb47f2068f87ecccd8f14da3a0e</code>，Review 日期为 2026-09-21。模型使用 DeepSeek-V4，正文只保留理解执行链真正需要的 V4 特性：mHC、DSV4 Attention、MoE 和 Vocab Parallel LM Head。

---

## 一、先把一个 Decode Step 的时间语义钉死

这是第一次读 SGLang 最值得先弄清的地方。假设某个 Request 在这一轮开始前已经有 L 个位置的历史 KV：

~~~text
logical position:
0, 1, 2, ..., L-1
~~~

上一轮已经采样出了新 Token <code>y_L</code>。它会成为这一轮 Decode 的 <code>input_ids</code>，但在当前 Forward 执行之前，它还没有经过 DeepSeek-V4，因此它自己的 K/V 内容也还没有由模型计算出来。

一轮普通 Decode 的时间线更接近：

~~~mermaid
flowchart TD
    A["已有 KV<br/>position 0 ... L-1"]
    B["上一轮采样 y_L<br/>本轮 input_id"]
    C["alloc_for_decode<br/>给 position L 分配 KV 槽"]
    D["req_to_token[req,L]<br/>写入新 slot 映射"]
    E["seq_lens: L → L+1"]
    F["ForwardBatch<br/>seq_lens=L+1<br/>position=L"]
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

所以 allocation 阶段是在为当前 input token 所属的新逻辑位置 L 准备状态空间。

等到 <code>ForwardBatch.init_new()</code> 构造时，<code>seq_lens</code> 已经是 L+1。普通 Decode 的 Position 默认由：

~~~text
positions = clamp(seq_lens - 1, min=0)
~~~

得到，因此：

~~~text
positions = L
~~~

把几个字段放在一张表里最清楚：

| 字段 | 普通 Decode 当前语义 |
| --- | --- |
| 当前 Forward 前已有 KV | positions <code>0 ... L-1</code> |
| <code>input_ids[i]</code> | 上一轮采样、这一轮真正送入模型的 <code>y_L</code> |
| allocation 使用的旧长度 | <code>L</code> |
| 新映射的 logical position | <code>L</code> |
| <code>ForwardBatch.seq_lens[i]</code> | <code>L+1</code> |
| <code>ForwardBatch.positions[i]</code> | <code>L</code> |
| <code>out_cache_loc[i]</code> | 当前 input token 的新 KV 写入目标 |
| 本轮 Sampler 输出 | 下一轮使用的 <code>y_(L+1)</code> |

因此不要把 Decode 中的 <code>ForwardBatch.seq_lens</code> 简单理解成“进入这一轮之前已有多少历史 Token”。对于普通非投机 Decode，它是**当前 kernel-facing sequence length，已经包含当前 input token 的位置**。

这条关系不要机械套到 Draft / Verify 等投机模式，它们有自己的长度语义。

---

## 二、ScheduleBatch 到 ForwardBatch：先区分调度状态和模型输入快照

SGLang 源码自己给出了很好的边界：

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

第一次读不需要看完整 dataclass。先把字段分成三组。

第一组是 Token 本身：

~~~text
input_ids
positions
~~~

如果有三个 active Request，普通 Decode 通常大致是：

~~~text
input_ids   [3]
positions   [3]
~~~

第二组是 Request 与历史状态：

~~~text
req_pool_indices
seq_lens
~~~

<code>req_pool_indices[i]</code> 不是 KV 地址，而是第 i 个 Batch lane 属于 ReqToTokenPool 的哪条 Request 逻辑行。<code>seq_lens[i]</code> 则给后面的 Attention / page-table builder 提供当前 kernel-facing 长度。

第三组是当前 Token 的新状态写入坐标：

~~~text
out_cache_loc
~~~

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

所以更稳妥的心智模型是：

> <code>ScheduleBatch.out_cache_loc</code> 来自 allocator 的 canonical allocation；进入 <code>ForwardBatch</code> 后，真正给 Kernel 使用的 write loc 还可能经过 ID-space translation。

到这里，整个控制边界可以画成：

~~~mermaid
flowchart LR
    R["Req<br/>请求生命周期"]
    SB["ScheduleBatch<br/>调度状态"]
    FB["ForwardBatch<br/>一次 Forward 快照"]
    MR["ModelRunner"]
    M["DeepSeek-V4"]

    R --> SB --> FB --> MR --> M
~~~

从 <code>ForwardBatch</code> 开始，阅读方式也应该改变：少想 HTTP 和队列，多问 Tensor shape、状态 ownership 和物理地址。

---

## 三、Embedding 与 mHC：为什么 V4 的 hidden state 会在 [T,4,H] 和 [T,H] 之间变化？

假设普通 Decode 当前：

~~~text
T = B = 3
~~~

那么：

~~~text
input_ids
[3]
~~~

经过：

~~~python
hidden_states = self.embed_tokens(input_ids)
~~~

DeepSeek-V4 的 hidden size 是 4096，于是先得到：

~~~text
[3]
  ↓ Embedding
[3,4096]
~~~

随后当前 V4 会把它扩成：

~~~text
[T,4,H]
~~~

因为：

~~~text
hc_mult = 4
~~~

第一次看到这里，很容易误以为 Attention 和 MoE 都直接处理四路 hidden state。严格来说不是。

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
    P["持久 mHC State<br/>[T,4,H]"]
    PRE["hc_pre / combine"]
    W1["Working Hidden<br/>[T,H]"]
    ATT["Attention"]
    W2["Working Hidden<br/>[T,H]"]
    MID["hc_post + hc_pre / combine"]
    W3["Working Hidden<br/>[T,H]"]
    MOE["MoE"]
    W4["Working Hidden<br/>[T,H]"]
    POST["hc_post"]
    P2["持久 mHC State<br/>[T,4,H]"]

    P --> PRE --> W1 --> ATT --> W2 --> MID --> W3 --> MOE --> W4 --> POST --> P2
~~~

<code>hc_pre()</code> 把四路 persistent state 混成当前子层真正消费的二维 Working Hidden，并维护内部 mixing coefficient；<code>hc_post()</code> 再把子层产生的 <code>[T,H]</code> 输出混回 <code>[T,4,H]</code>。

所以第一次读 V4 时只需要记住：

> **mHC 的四路通道是跨子层保存和混合的状态；Attention 与 MoE 的主要计算入口仍然是普通 Token-row Tensor [T,H]。**

Sinkhorn、post、comb 的具体数学可以等主链读通以后再补。

---

## 四、Attention 与 KV Cache：关键是 Request 的逻辑位置怎样落到 910C 的物理 Pool

现在 Working Hidden 是：

~~~text
[T,4096]
~~~

DeepSeek-V4 Attention 会从中构造 Query，并为当前 Token 产生新的 KV 状态。Q 路径可以先概念化成：

~~~text
x [T,4096]
   ↓ wq_a
q_lora [T,1024]
   ↓ norm / wq_b
q [T, local_heads, 512]
~~~

真正容易迷路的是 Cache ownership。

先把“写路径”和“读路径”分开。

### 写路径：当前 Token 的新状态写到哪？

前面已经知道，逻辑位置 L 在 allocation 阶段拥有一个新的 canonical full-side slot。

在当前 Ascend DSV4 Backend 中，SWA 写入会进一步把 Full-side location 转成 SWA location：

~~~python
swa_loc =
    token_to_kv_pool.translate_loc_from_full_to_swa(
        forward_batch.out_cache_loc
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

<code>req_pool_indices</code> 不直接指向 KV Tensor。它先选择：

~~~text
ReqToTokenPool.req_to_token
~~~

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
    R["req_pool_indices<br/>选择 Request row"]
    RT["req_to_token[req, logical_pos]"]
    FULL["Canonical Full-side token slots"]

    SM["Full → SWA mapping"]
    SPT["swa_page_table"]
    SP["SWA physical pool"]

    C4M["按 C4 ratio / page 派生"]
    C4T["c4_page_table"]
    C4P["C4 physical pool"]

    SIDE["req_to_c128_sidecar<br/>Request-scoped sidecar"]
    C128T["c128_page_table"]
    C128P["独立 C128 physical pool"]

    R --> RT --> FULL
    FULL --> SM --> SPT --> SP
    FULL --> C4M --> C4T --> C4P

    R --> SIDE --> C128T --> C128P
~~~

SWA Page Table 可以沿 Full slot 做 Full→SWA 映射。

C4 Page Table 也以该 Request 的 <code>req_to_token</code> 行为来源，在 ratio-4 对应的逻辑边界派生 C4 page。

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

这也是为什么第一次读 Attention 源码，应该先搞懂 Request → Logical Position → Physical Page，而不是一上来记 CANN Operator 的所有参数。

---

## 五、MoE：为什么 Attention 后突然出现 Router、A2A 和 Expert Parallel？

Attention 完成以后，经过 mHC / Norm 边界，Working Hidden 又是：

~~~text
[T,4096]
~~~

然后进入 V4 的 MoE。

当前公开配置有 256 routed experts，每个 Token 选择 6 个 routed experts。第一步是 Router：

~~~python
router_logits = self.gate(hidden_states)

topk_output = self.topk(
    hidden_states=hidden_states,
    router_logits=router_logits,
    ...
)
~~~

因此：

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
flowchart LR
    T["Token hidden rows<br/>Token order"]
    R["Router / TopK"]
    D["Dispatch / A2A"]
    E["Expert-local rows<br/>M_e × H"]
    G["Expert GEMM"]
    C["Combine"]
    O["恢复 Token order<br/>[T,H]"]

    T --> R --> D --> E --> G --> C --> O
~~~

最值得理解的是 ownership 变化：

~~~text
进入 MoE 前：
按 Token / Batch row 组织

Router 后：
按 Expert owner 组织

Expert 算完：
恢复 Token order
~~~

所以 DeepEP / A2A 不是额外附加的“分布式复杂度”，而是 Token ownership 与 Expert ownership 不一致的直接结果。

第一次读到这里，先理解 Dispatch 是 Token→Expert ownership 转换、Combine 是 Expert→Token ownership 恢复，就足够继续往下走。

---

## 六、LM Head 与 Sampling：完整 [B,V] Logits 到底在什么时候存在？

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

这时当前 LogitsProcessor 的语义顺序是：

~~~mermaid
flowchart TD
    L["本 DP rank hidden rows<br/>[B_local,H]"]
    DG["DP-Attention row gather<br/>形成 global rows"]
    LM["各 TP rank 本地 LM Head<br/>[B_global,V_shard]"]
    VG["TP gather vocab shards<br/>[B_global,V]"]
    DS["DP row scatter<br/>恢复本地 Request rows"]
    O["Sampler 输入<br/>[B_local,V]"]

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

所以完整 Vocabulary 维度是在 vocab-shard collective 完成后才出现的；随后 DP row ownership 还要恢复。

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

因此最准确的入门结论不是“LM Head 输出 [B,V]”，而是：

> **LM Head 先按照 Vocabulary Parallel layout 产生 local vocab shard；LogitsProcessor 再恢复 Sampler 所需的完整 Vocabulary 维度和正确 Request row ownership。**

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

只有：

~~~text
SYNC_TOKEN_IDS_ACROSS_TP=1
~~~

或者当前 Batch 使用 Grammar 时，才执行：

~~~text
MIN all_reduce(next_token_ids)
~~~

Group 是 <code>tp_sync_group</code>。因此 DP Attention 下这里是 <code>attn_tp_group</code>，不是整个全局 TP World。

到这里，自回归循环闭环：

~~~mermaid
flowchart LR
    H["hidden<br/>[B_local,H]"]
    LM["Vocab-sharded LM Head"]
    G["Logits gather / row restore"]
    L["full logits<br/>[B_local,V]"]
    S["Sampler"]
    N["next_token_ids<br/>[B_local]"]
    R["Scheduler relay"]
    I["下一轮 input_ids"]

    H --> LM --> G --> L --> S --> N --> R --> I
~~~

下一轮 Decode 再从第一章的 L → L+1 时间线开始。

---

## 七、第一次真正读 SGLang，按这条路线就够了

第一站读 Decode 状态与 KV Allocation：

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

第二站读 ModelRunner，只理解：

~~~text
ForwardBatch
→ Graph / Eager execution
→ Model Forward
→ Logits
→ Sampler
~~~

第三站进入 <code>deepseek_v4.py</code>，先认清：

~~~text
Embedding:          [T,H]
Persistent mHC:     [T,4,H]
Attention / MoE:    [T,H]
~~~

第四站读 Ascend 910C KV ownership，始终只问：

~~~text
哪个 Request？
    ↓
哪个 logical position？
    ↓
哪个 physical/cache page？
~~~

并记住 C128 是独立 sidecar ownership。

第五站读 MoE，只先找：

~~~text
gate
topk
dispatch
experts
combine
~~~

第六站读 Logits / Sampling，把：

~~~text
[T,H]
→ [rows,V_shard]
→ [B_local,V]
→ next_token_ids
~~~

走通。

到这里，一次普通 SGLang Decode 已经完整串起来。

真正值得形成的源码阅读习惯只有几个问题：

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

第一次读 SGLang 最应该学会的，不是“这个类是干什么的”，而是：

> **这个 Token 现在处在哪个时间点、哪个 Tensor、哪个 ownership 空间；下一步为什么必须发生这次状态或布局转换？**

当开始这样读源码时，SGLang 就会从一大片类名，变成一条可以持续追踪的数据流。

---

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
