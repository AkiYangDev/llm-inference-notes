# 从一次 Decode 读懂 SGLang：Tensor、缓存与下一枚 Token

上一轮采样出一枚 Token，只是确定了整数 ID，它还没有作为输入经过模型。这一轮才会计算它的各层状态，再预测下一枚输出。

下面用三条请求走一次普通 Decode。把输入 ID 接到逻辑位置、缓存目标，再经过 Attention、MoE、LM Head、Sampler，最后回到同一请求。源码中的长度加一、行重排和分组通信，都可以在这次执行中找到对应。

主线为普通非投机 Target Decode，采用 DeepSeek-V4，源码固定为 SGLang commit <code>0f6761b54facebb47f2068f87ecccd8f14da3a0e</code>，核对日期为 2026-09-21。mHC、DSV4 Attention、MoE、Vocab Parallel LM Head 只展开这条链需要的部分。

## 一、上一轮的输出怎样成为本轮输入

设本轮开始前已缓存 L 个位置，逻辑历史为 `0,1,2,...,L-1`。

上轮采样的 <code>y_L</code> 成为当前 <code>input_ids</code>。位置 L 尚无 K/V，先由本轮分配槽位，再由各层执行写入。

普通 Decode 的顺序为：

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

固定 <code>ScheduleBatch.prepare_for_decode()</code> 先运行：

~~~python
self.out_cache_loc = alloc_for_decode(
    self,
    token_per_req=1,
)
~~~

然后推进长度：

~~~python
self.seq_lens = self.seq_lens + 1
self.seq_lens_cpu = self.seq_lens_cpu + 1
~~~

<code>alloc_for_decode()</code> 使用加一之前的 L，把分配的 slot 填到：

~~~text
req_to_token[
    request_slot,
    logical_position=L
]
=
new_slot
~~~

allocation 完成空间、映射准备。Attention 真正处理该 Token 时，才会向各层缓存写入状态。

到 <code>ForwardBatch.init_new()</code>，<code>seq_lens</code> 已是 L+1。普通 Decode 默认使用 `positions = clamp(seq_lens - 1, min=0)`。

所以当前 Token 的 `positions = L`。

字段按先后顺序为：

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

已有 10 个位置，旧 L=10，新输入 position=10，ForwardBatch 长度为 11。位置从 0 开始编号，长度统计总位置数；普通非投机 <code>ForwardBatch.seq_lens</code> 已包含当前输入，供 Kernel 使用。

Draft / Verify 另外约定长度含义，本文先沿普通 Decode 继续。

## 二、把三条请求组织为 ForwardBatch

ScheduleBatch 描述本轮选中的请求，ForwardBatch 再把执行 Tensor 交给 ModelRunner：`forward_mode`、`batch_size`、`input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc`、`seq_lens_sum`、`positions`、`sampling_info`。

先把 `input_ids`、`positions` 按行配对。

三条继续中的请求各贡献一枚新 Token。不计 Graph padding，两份 Tensor 为：

~~~text
input_ids   [3]
positions   [3]
~~~

假设原 KV 长度分别为 10、7、20：

| 本轮 Batch 行 | 原有 KV 长度 | 本轮 position | ForwardBatch.seq_lens |
| ---: | ---: | ---: | ---: |
| 0 | 10 | 10 | 11 |
| 1 | 7 | 7 | 8 |
| 2 | 20 | 20 | 21 |

ID、position 各只有三行，每行读取的历史长度却不同。因此还需要 `req_pool_indices`、`seq_lens`。

本轮 batch 行号随组批变化。<code>req_pool_indices[i]</code> 先选 ReqToTokenPool 的请求记录行，由该行查历史；<code>seq_lens[i]</code> 提供本轮可见长度。请求记录行本身还不是 KV 地址。

`out_cache_loc` 再给出新增状态写入的位置。

到执行边界，位置编号还可能再转换，<code>ForwardBatch.init_new()</code> 调用：

~~~python
model_runner.kv_index_translator.rebind_write_loc(ret)
~~~

固定 KVIndexTranslator 区分 virtual、physical、kernel-facing ID。普通非 unified pool 中它们可以相同；unified / DCP 等布局会将 <code>ForwardBatch.out_cache_loc</code> 重绑为 kernel-facing FULL-side ID，原值留在 <code>out_cache_loc_virtual</code>。

因此 ScheduleBatch 中的 canonical allocation，到了 ForwardBatch 可能已换成 Kernel 接口编号。字段同名，读取时仍要识别它所处的 ID 空间。

从请求到模型的交接为：

~~~mermaid
flowchart TD
    R["Req · 请求生命周期"]
    SB["ScheduleBatch · 调度状态"]
    FB["ForwardBatch · 一次 Forward 快照"]
    MR["ModelRunner"]
    M["DeepSeek-V4"]

    R --> SB --> FB --> MR --> M
~~~

进入 <code>ForwardBatch</code> 时，本轮请求已经选定。接下来跟踪这些行怎样被模型消费。

## 三、Embedding 与 mHC 的形状变化

本轮 `T=B=3`，`input_ids` 为 `[3]`。Embedding 执行：

~~~python
hidden_states = self.embed_tokens(input_ids)
~~~

DeepSeek-V4-Flash 的 hidden_size=4096，因此输出 hidden states 为 `[3,4096]`。

本例 `hc_mult = 4`，再扩为 `[T,4,H]`。T=3、H=4096，每 Token 保留四路状态。mHC 在子层前把它们混成二维 Working Hidden，子层后再混回多路。

持久 mHC State 因而为 `[T,4,H]`，Attention/MoE 的临时 Working Hidden 为 `[T,H]`。

一层内形状变化为：

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

<code>hc_pre()</code> 混合四路状态、生成 Working Hidden，同时维护 mixing coefficient；<code>hc_post()</code> 再把 <code>[T,H]</code> 子层输出混回 <code>[T,4,H]</code>。

Attention、MoE 使用二维 Tensor 时，多路状态仍由前后的混合操作维护。读到 Sinkhorn、post、comb，可以先将它们放回这组输入、输出形状。

## 四、当前 Token 写入缓存并读取历史

Attention 接收 `[T,4096]` Working Hidden。

由它生成当前 Query、新增 KV。Query 的概念路径为：

~~~text
x [T,4096]
   ↓ wq_a
q_lora [T,1024]
   ↓ norm / wq_b
q [T, local_heads, 512]
~~~

新增 KV 经 `out_cache_loc` 写入；历史经请求记录、页表读取，逻辑位置由映射接到物理 slot。DSV4 的具体映射放在折叠段中，暂时略过也能继续接到 MoE。

<details markdown="1">
<summary>深入：Ascend DSV4 的写入、页表和压缩状态</summary>

### 写路径：当前 Token 的新状态写到哪？

位置 L 的 canonical full-side slot 已在 allocation 阶段分配。

Ascend DSV4 写 SWA 前，继续转成 SWA location：

~~~python
swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(
    forward_batch.out_cache_loc
    # Full-side 到 SWA 的位置映射
)
~~~

然后更新对应层：

~~~python
pool.set_swa_buffer(
    layer_id=layer_id,
    loc=swa_loc,
    cache=swa_k,
)
~~~

DSV4 专用分配还可以通过 <code>out_cache_loc_dsv4</code> 提供 `out_c4_loc`、`out_c128_loc`。当前位置完成相应 compression ratio 的 block 时，才形成新的压缩目标。

所以新增状态可能分属：

~~~text
canonical full-side allocation
        │
        ├── SWA write destination
        ├── 可选 C4 compressed destination
        └── 可选 C128 compressed destination
~~~

### 读路径：req_pool_indices 首先只是 Request row

历史读取先用 <code>req_pool_indices</code> 选择 <code>ReqToTokenPool.req_to_token</code> 的请求行：

~~~text
req_to_token[
    request_slot,
    logical_position
]
=
canonical token-slot id
~~~

Base Ascend Backend 从记录行取历史 slots，按 page size 转成 page ID，建立 Block Table。

这次查找依次经过请求、逻辑 Token 位置、物理缓存页。

### SWA、C4、C128 的 ownership 不是同一种映射

Ascend 910C DSV4 的映射为：

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

SWA 页表通过 Full→SWA 映射转换 full slot。

C4 也从本请求 <code>req_to_token</code> 开始，在 ratio-4 逻辑边界派生页。

C128 使用独立记录：

~~~python
req_to_c128_sidecar[
    req_pool_indices,
    :n_groups
]
~~~

物理页由 request-scoped sidecar 管理，不能直接拿 full Token slot 除以 128。

Kernel 最后共同消费 Query、sequence lengths、page table、物理 pool。

沿这一层层映射查回当前行，再对照实参，才能确认读取的历史和长度属于该请求。

</details>

## 五、MoE 怎样分发并合并 Token

Attention 输出经过 mHC、Norm，MoE 前又得到 `[T,4096]` Working Hidden。三行仍各自代表本轮 Token，Router 再为它们选择专家。

公开配置有 256 个 routed experts，每 Token 选 6 个：

~~~python
router_logits = self.gate(hidden_states)

topk_output = self.topk(
    hidden_states=hidden_states,
    router_logits=router_logits,
    # 其余参数省略
)
~~~

Router 将 `[T,4096]` 投影成 `[T,256]` logits，TopK 为每行挑选 6 份专家权重。

专家在远端 Rank 时，Dispatch / All-to-All 把 Token 表示发给 owner。

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

执行顺序从 Token/Batch 行变为专家任务，贡献算完后，由 Combine 恢复原 Token 顺序。DeepEP/A2A 承担跨 Rank 的发送、返回。

一个专家可以同时收到来自三条请求的行，前馈仍逐行计算。Combine 需要知道每份贡献属于哪个 Token；重排后的本地行号，无法单独标识原请求。ownership 描述的则是这一步由谁持有、处理数据。

## 六、从 hidden states 到下一枚 Token

层栈输出仍是 hidden states，接着要投影到词表。

逻辑上可以画为：

~~~text
[B,H]
  ↓ LM Head
[B,V]
~~~

词表并行的本地 LM Head MatMul，首先只算一个 vocabulary shard：

~~~text
[rows,H]
×
[H,V_shard]

→

[rows,V_shard]
~~~

固定 DeepSeek-V4 在 PP 最后阶段创建 <code>ParallelLMHead</code>。`enable_dp_lm_head` 开启时使用 Attention TP Group，否则通常沿全局 TP Group。

### DP Attention，但 LM Head 仍沿全局 TP 分片

全局 TP 路径同时面对 DPA 分布的请求行和 TP 分片的词表列。LogitsProcessor 分别处理：

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

对应调用为：

~~~text
_gather_dp_attn_hidden_states()
        ↓
_compute_lm_head()
        ↓
TP vocab-shard gather
        ↓
_scatter_dp_attn_logits()
~~~

先 gather 请求行，再拼齐 vocabulary shards，最后 scatter 请求行回各 DP。恢复完整词表、恢复请求归属，分别作用于列和行。

同一版本还可按条件使用 LM Head All-to-All，一次重分布替代词表 gather 与 row scatter。实际是否走这条路径，由 `_can_use_tp_lm_head_all_to_all()` 检查开关、行布局等条件。

### 开启 DP LM Head

DP LM Head 把 <code>ParallelLMHead</code> 切到 <code>attn_tp_group</code>。

各 DP 保留 local rows，在自己的 Attention TP 组拼齐词表：

~~~text
[B_local,H]
    ↓ local LM Head
[B_local,V_shard]
    ↓ attn-TP vocab gather
[B_local,V]
~~~

<code>attn_tp_size=1</code>，本地已持有完整 Vocabulary Weight，就省掉 vocab gather。

两条路径都为本地请求提供完整词表分数，区别在权重分给哪个组，以及中间是否重新 gather/scatter 请求行。

交给普通生成 Sampler 的 `next_token_logits` 因而为 `[#local_sequences,vocab_size]`。

### Sampler 默认不再同步最终 Token ID

Greedy 从完整本地 logits 取最大值：

~~~python
batch_next_token_ids = torch.argmax(
    logits,
    -1,
)
~~~

其他采样再使用 <code>SamplingBatchInfo</code> 中的 temperature、top-k、top-p、min-p、grammar。

普通模式的 Sampler sync group 为 `tp_group`，DP Attention 下则为 `attn_tp_group`。

固定默认不逐轮同步最终 Token ID，而依赖相关 Rank 的末端 collective、LM Head MatMul、Sampling Kernel 保持确定性，减少一次通信。

`SYNC_TOKEN_IDS_ACROSS_TP=1` 或当前 Batch 使用 Grammar 时，执行 `MIN all_reduce(next_token_ids)`。

它在 <code>tp_sync_group</code> 内对齐同批请求的 ID。DPA 使用 <code>attn_tp_group</code>，不同 DP 的请求因此各自在对应组处理。

采样结果再交给调度器：

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

新输出成为下一轮输入，继续从 L → L+1 推进。

## 七、按这个例子读源码

可以沿这条路径进入 Decode 状态、KV allocation：

~~~text
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/mem_cache/allocation.py
python/sglang/srt/model_executor/forward_batch_info.py
~~~

第一遍将旧 L 代入 `input_ids`、`seq_lens`、`positions`、`req_pool_indices`、`out_cache_loc`，得到“分配位置 L → 长度 L+1 → 当前 position=L”。再从 ForwardBatch 进入 Graph/Eager、Model Forward、Logits、Sampler，接回下一轮。

`deepseek_v4.py` 中的表示、行归属变化为：

| 执行位置 | 追踪重点 |
| --- | --- |
| Embedding 与 mHC | Embedding 为 `[T,H]`，持久 mHC 为 `[T,4,H]`，Attention/MoE Working Hidden 回到 `[T,H]` |
| Attention | 同一请求的逻辑位置怎样查到物理/cache page；C128 另查独立 sidecar |
| MoE | gate、topk、dispatch、experts、combine 前后，一行属于哪个 Token，专家贡献怎样返回 |
| LM Head 与 Sampler | `[T,H]` 怎样变为本地 `[rows,V_shard]`，何时取得 `[B_local,V]`，再生成 `next_token_ids` |

逐边界记录 Token 数、shape、请求身份、位置、历史读取、新状态写入。gather/scatter 则继续区分是在重新分配 Token 行，还是在拼 Tensor/词表分片。

三条请求从采样 ID 出发，先安排再写 KV；MoE 按专家重排行；词表投影后恢复本地请求。把最后的输出接回原请求，下轮仍能从相同身份继续，这一轮的数据关系就连起来了。

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
