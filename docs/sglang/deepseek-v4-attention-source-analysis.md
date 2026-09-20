# DeepSeek-V4 Attention 源码解析：一个 ForwardBatch 如何变成 Query，并通过 KV Cache 找回历史上下文

上一篇《[SGLang ModelRunner 源码解析：ForwardBatch 如何驱动 DeepSeek 完成一次 Prefill / Decode](sglang-modelrunner-source-analysis.md)》把执行链追到了 `DeepseekV4DecoderLayer`。从这里继续往下一层，就是推理引擎里最核心的一段：当前 token 怎样形成 Query，又怎样准确找到属于自己这条请求的历史缓存。

如果只用 Transformer 教科书公式回答，大概会写成：

~~~text
Q = XWq
K = XWk
V = XWv
Attention(Q, K, V)
~~~

这足以解释基本数学，却不足以解释 DeepSeek-V4 在 SGLang 里的真实执行。服务端已经把多条请求打包成一个 token-major Tensor；历史状态分散在缓存池的不同物理页；DeepSeek-V4 还同时存在滑窗历史与压缩历史。真正需要追的是：

> `ForwardBatch` 怎样把 packed token 重新绑定到各自请求的逻辑位置和物理缓存，再让当前 Query 只读取属于自己的历史？

这篇文章沿着一个普通自回归 batch，把 `ForwardBatch → MQALayer → Query / cache write → page table → SWA / C4 / C128 → Attention output` 连起来。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线使用公开源码中的 DeepSeek-V4、普通文本生成、非投机推理、PP=1，并聚焦 Ascend NPU eager 的普通单流职责路径。图回放、多流、Context Parallel、unified FP8 KV 和 DeepSeek-V4.1 的 `compress_ratio=1/2` 会改变部分中间表示或时序，不在本文代表路径中展开。本文追到公开 NPU Attention operator 入口与缓存映射，不把这些代码外推成某个私有部署镜像的底层运行轨迹。

## 一、ForwardBatch 不会直接“变成 Query”，它先把 packed token 放回各自请求的坐标系

`DeepseekV4DecoderLayer.forward()` 在完成 mHC / norm 准备后，会把当前层的 hidden states、positions 和整份 `ForwardBatch` 一起交给 Attention：

~~~python
hidden_states = self.self_attn(
    x=hidden_states,
    positions=positions,
    forward_batch=forward_batch,
    x_quant=x_quant,
)
~~~

固定版本入口见 [`DeepseekV4DecoderLayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3006-L3189)。

这里的 `self.self_attn` 是 `MQALayer`。[构造入口](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2693-L2710)

继续用前两篇里的两个请求。假设 Prefill 已经为它们各生成第一枚 token：

~~~text
R1 -> T1
R2 -> T2
~~~

下一轮普通 Decode：

~~~text
input_ids = [T1, T2]
seq_lens = [7, 7]
positions = [6, 6]
~~~

经过 Embedding、mHC 和输入层归一化后，某一层 Attention 收到：

~~~text
x.shape = [2, H]
~~~

其中 `x[0]` 属于 R1，`x[1]` 属于 R2。问题在于，单看 `x` 自己并没有“请求身份”。真正把它们区分开的，是继续同行的 `ForwardBatch`：

| 信息 | 回答的问题 |
| --- | --- |
| `positions` | 当前 token 在自己的序列中处于哪个逻辑位置？ |
| `seq_lens` | 这条请求到本轮为止有多长？ |
| `req_pool_indices` | 这条请求对应 request-token 映射池中的哪一行？ |
| `out_cache_loc` / DSV4 cache loc | 本轮新缓存应该写到哪个物理位置？ |
| Attention metadata | 当前 Query 可以读取哪些历史页、每条序列边界在哪里？ |

因此标题里的“ForwardBatch 变成 Query”更准确的理解是：

> hidden states 负责产生 Query；ForwardBatch 负责告诉 Query 它是谁、它在哪里，以及它的过去被存在哪里。

这里还需要先避开一个标准 MHA 的惯性理解。DeepSeek-V4 的 `MqaAttentionBase` 明确要求：

~~~python
assert config.num_key_value_heads == 1
~~~

并构造：

~~~python
self.attn_mqa = RadixAttention(
    self.n_local_heads,
    self.head_dim,
    self.softmax_scale,
    num_kv_heads=1,
    ...
)
~~~

源码见 [`MqaAttentionBase`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L742-L945) 和 [`MQALayer.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1041-L1193)。

所以不能把这里画成传统 MHA 的：

~~~text
Q0 -> K0 / V0
Q1 -> K1 / V1
Q2 -> K2 / V2
...
~~~

当前实现拥有多个 Query heads，但共享一套 KV / latent-cache 侧表示。后文仍会沿用源码里的变量名 `kv`、`k`、`v`，但不会把它们机械等价为传统 Transformer 中“两套独立 K Cache 和 V Cache”。

## 二、当前 hidden states 如何生成 Query，并把 position 注入这一轮计算

`MQALayer` 的 Query 主干不是一次单独的 `XWq`，而是低秩两阶段投影。逻辑上：

~~~text
x
[T, H]
   |
   | wq_a
   v
q_lora
[T, Rq]
   |
   | q_norm
   v
normalized q_lora
[T, Rq]
   |
   | wq_b
   v
Q
[T, Nlocal * D]
   |
   | view
   v
[T, Nlocal, D]
~~~

固定版本的辅助函数可以在 [`_compute_q_a()` / `_compute_q_b()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1291-L1347) 找到。实际配置可以把第一阶段 Query 和 KV 投影融合成 `wqkv_a`，因此这张图表示的是逻辑数据流，不是所有配置都必须逐个执行三个独立 Python kernel。

核心维度来自：

~~~python
self.qk_rope_head_dim = config.qk_rope_head_dim
self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
~~~

固定版本 DSV4 Attention 测试矩阵使用：

~~~text
qk_nope_head_dim = 448
qk_rope_head_dim = 64
head_dim          = 512
~~~

见 [DSV4 Attention Capability Matrix](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/attention/unittests/dsv4/README.md)。

因此一个 Query head 可以理解为：

~~~text
512 dims
├── 448 NoPE dims
└── 64  RoPE dims
~~~

普通 NPU 单流路径在 `MQALayer._forward_prepare()` 中形成 Q 后，还会对当前 head 表示进行归一化，并把位置编码作用到对应维度。固定源码见 [`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095)。

当前两个 Decode token：

~~~text
positions = [6, 6]
~~~

并不冲突。position 是“每条请求内部的逻辑位置”，不是 packed batch 的全局行号：

~~~text
Q[0] -> R1 position 6
Q[1] -> R2 position 6
~~~

所以同一个 batch 中可以同时存在多个 position 6。

~~~mermaid
flowchart TD
    X[hidden states<br/>T x H]
    X --> QA[wq_a / fused qkv_a]
    QA --> QL[q_lora<br/>T x Rq]
    QL --> QN[q_norm]
    QN --> QB[wq_b]
    QB --> Q[Query<br/>T x Nlocal x D]

    P[ForwardBatch.positions] --> R[RoPE / position transform]
    Q --> R
    R --> QR[position-aware Query]
~~~

到这里已经有了当前 Query，但历史还没有出现。历史不是从这些 hidden states 中重新计算出来的，而是在此前 Prefill / Decode 轮次中逐步写进长期缓存池。

## 三、普通 NPU 路径会先写当前缓存，再执行主 Attention；`k=v` 只是接口约定

同一个当前层输入 `x` 还会产生本轮新的共享缓存表示。未融合时，基础投影是：

~~~python
kv, _ = self.wkv(x)
~~~

`wkv` 的输出宽度是 `head_dim`，因此概念 shape 是：

~~~text
x
[T, H]
   |
   | wkv
   v
new cache representation
[T, D]
~~~

普通 NPU 单流分支会继续对它做 `kv_norm` 和 RoPE，然后调用：

~~~python
attn_backend.store_cache(
    layer_id=self.layer_id,
    swa_k=kv_for_cache,
    forward_batch=forward_batch,
)
~~~

也就是说，在本文代表路径里：

> **当前 token / 当前 Prefill chunk 的 SWA 缓存先被写入 pool，随后才调用主 Attention backend。**

源码就在 [`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095)。

这点对理解 Prefill 很重要。物理上先把当前 chunk 的缓存写进 pool，并不意味着一个 Query 可以看到“未来 token”。真正的逻辑可见范围仍由 Attention metadata、序列边界和算子 mask 决定。**cache write timing 与 causal visibility 是两件不同的事。**

在 SWA 写入之后，如果这一层存在 C4 indexer / compressor，相关压缩状态也会在 `_forward_prepare()` 返回之前完成准备；随后 `MQALayer.forward()` 才进入主 Attention。

当前普通 NPU 路径写完缓存后会把局部变量 `kv` 设为 `None`。接下来的代码：

~~~python
attn_k = kv if kv is not None else q
~~~

然后调用：

~~~python
attn_backend.forward(
    q=attn_q,
    k=attn_k,
    v=attn_k,
    ...
    save_kv_cache=False,
)
~~~

固定入口见 [`MQALayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2285-L2375)。

这正是最容易被“看代码看错”的地方。

源码确实把同一个对象同时传给 `k` 和 `v`。通用 DSV4 backend 甚至有：

~~~python
assert k is v, "DeepseekV4 shares k and v"
~~~

见 [`DeepseekV4AttnBackend._forward_attention()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L3715-L3750)。

但这**不能**被翻译成：

> “DeepSeek-V4 的数学 K 向量和 V 向量数值完全相等。”

尤其在本文普通 NPU 路径里，真实缓存已经提前写入 pool，`save_kv_cache=False`，此时传进去的 `attn_k=q` 只是为了满足统一 backend 接口；主 Attention 真正读取的是 cache pool，而不会把这个 sentinel 当成历史 K/V 数据。

因此更准确的表述是：

> **DSV4 Attention backend 的接口把 K/V 形参别名到同一个共享缓存对象；在部分路径中这个参数甚至只是 sentinel。它反映的是 backend / cache 的接口设计，不应被当作标准 Attention 数学中的 “K = V” 结论。**

这个区分也解释了为什么理解 DSV4 时应该追“cache representation + backend contract”，而不是只盯变量名 `k`、`v`。

~~~mermaid
flowchart TD
    X[current hidden states] --> Q[build Query]
    X --> C[build current cache representation]
    C --> W[write SWA cache]

    W --> P[prepare optional C4/C128 state]
    P --> A[Attention backend]
    Q --> A

    H[historical cache pool] --> A

    W -. physical write timing .-> N[logical visibility still comes from metadata and masks]
~~~

## 四、Query 找回历史的真正桥梁，是请求索引、req_to_token 和 page table

写入缓存解决的是“当前 token 的状态放在哪里”。读取历史还需要反方向回答：

> 当前 Query 属于哪个请求？这个请求之前的 token 被放在缓存池的哪些物理页？

这条映射从 `ForwardBatch.req_pool_indices` 开始。

Ascend backend 在 forward 前构造 metadata 时，会读取 request-token pool：

~~~python
self.forward_metadata.block_tables = (
    self.req_to_token_pool.req_to_token[
        forward_batch.req_pool_indices,
        :seq_lens_max
    ][:, :: self.page_size]
    // self.page_size
)
~~~

固定源码见 [`AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L479-L540)。

可以把两层 pool 的职责分开：

~~~text
ReqToTokenPool
回答：
“请求 R1 的逻辑 token 0、1、2... 映射到哪些 cache slot？”

TokenToKVPool / DSV4 pools
回答：
“这些 slot 上真正保存的 Attention cache bytes / tensors 是什么？”
~~~

因此历史读取链是：

~~~text
req_pool_index
      ↓
req_to_token[row, logical_position]
      ↓
physical token/cache locations
      ↓
block / page tables
      ↓
SWA / compressed cache pools
~~~

假设教学示例中：

~~~text
R1 req_pool_index = 3
R2 req_pool_index = 8
~~~

那么即使当前：

~~~text
input_ids = [T1, T2]
~~~

两个 token 在同一个 packed Tensor 中，backend 仍会分别沿：

~~~text
req_to_token[3, ...]
req_to_token[8, ...]
~~~

构造两套历史映射。

所以：

~~~text
Q0 -> R1 page table -> R1 cache pages
Q1 -> R2 page table -> R2 cache pages
~~~

而不会变成：

~~~text
Q0 / Q1 -> 同一整块历史
~~~

这就是 packed batching 不会导致请求上下文串线的关键原因。

### Prefill 和 Decode 的 metadata 到底差在哪里

`DeepseekV4AscendAttnBackend.init_forward_metadata()` 会进一步构造 DSV4 所需的 Query / KV 长度 metadata。[固定源码](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1837-L1930)

继续用前文普通、无 speculative / CP / padding 的两个请求。

Prefill 时：

~~~text
extend_seq_lens = [6, 3]
seq_lens        = [6, 6]
~~~

源码首先计算累计 Query 结束位置：

~~~text
actual_seq_lengths_q
=
[6, 9]
~~~

再在前面补 0：

~~~text
actual_seq_lengths_q_pa
=
[0, 6, 9]
~~~

而 KV 有效长度是：

~~~text
actual_seq_lengths_kv
=
[6, 6]
~~~

三者的语义不要混淆：

~~~text
actual_seq_lengths_q      = 每条请求 Query 的累计结束位置
actual_seq_lengths_q_pa   = 带 leading 0 的 Query 边界
actual_seq_lengths_kv     = 每条请求当前可用 KV 长度
~~~

因此：

~~~text
Query rows 0..5 -> R1
Query rows 6..8 -> R2
~~~

到了下一轮普通 Decode：

~~~text
batch_size = 2
seq_lens   = [7, 7]
~~~

源码直接构造：

~~~text
actual_seq_lengths_q
=
[1, 2]

actual_seq_lengths_q_pa
=
[0, 1, 2]

actual_seq_lengths_kv
=
[7, 7]
~~~

于是同一个 backend 面对的是：

| | Prefill | Decode |
| --- | --- | --- |
| packed Query rows | 9 | 2 |
| 每请求新 Query 数 | R1=6, R2=3 | R1=1, R2=1 |
| `actual_seq_lengths_q_pa` | `[0,6,9]` | `[0,1,2]` |
| `actual_seq_lengths_kv` | `[6,6]` | `[7,7]` |

这张表比“Prefill 处理 prompt、Decode 处理一个 token”更接近真实算子视角：Prefill 有较多 Query rows；Decode 的 Query 很少，但历史 KV 长度会持续增长。

~~~mermaid
flowchart TD
    F[ForwardBatch] --> R[req_pool_indices]
    F --> L[seq_lens]

    R --> RT[ReqToTokenPool]
    L --> RT
    RT --> B[Block / Page Tables]
    B --> C[Physical Cache Pools]

    F --> M[Query / KV length metadata]

    Q[current Query] --> A[DSV4 Attention Backend]
    B --> A
    C --> A
    M --> A

    A --> O[Attention Output]
~~~

## 五、SWA / C4 / C128 不是三路同时开启，而是每个 Attention Layer 按 compress_ratio 选择自己的历史结构

初稿最容易造成误解的地方，是把 SWA、C4、C128 画成了 Query 同时扇出到三路。

真实源码不是这样。

每个 `MQALayer` 都有自己的：

~~~python
self.compress_ratio = config.compress_ratios[layer_id]
~~~

对于本文 DeepSeek-V4 + Ascend backend 主线，主 Attention 接口接受：

~~~text
compress_ratio = 0 / 4 / 128
~~~

源码见 [`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2075)。

因此应该画成“每层选择一条结构”：

~~~mermaid
flowchart TD
    L[One MQALayer] --> R{compress_ratio}

    R -->|0| S0[SWA only]

    R -->|4| S4[SWA plus C4 compressed history]
    S4 --> I[C4 Indexer selects Top-K compressed positions]

    R -->|128| S128[SWA plus C128 compressed history]
~~~

也就是说：

~~~text
Layer A: ratio 0
-> SWA only

Layer B: ratio 4
-> SWA + C4

Layer C: ratio 128
-> SWA + C128
~~~

不是：

~~~text
每一层都同时做 SWA + C4 + C128
~~~

### ratio = 0：只读取 SWA

当 `compress_ratio == 0`，Ascend backend 进入 `_forward_swa()`。

它从 pool 获取当前层的 SWA buffer，并把：

~~~text
actual_seq_lengths_q_pa
actual_seq_lengths_kv
swa_page_table
kernel_metadata
~~~

一起交给 NPU sparse attention operator。[源码：`_forward_swa()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2077-L2116)

固定版本 backend 在模型没有显式 sliding-window 配置时使用 128 作为 DSV4 fallback window。它可以被理解成“近期高分辨率历史”，但这个 128 是当前实现的 fallback，不应泛化成所有 DeepSeek-V4 配置的永恒常数。

### ratio = 4：SWA + C4，且 C4 有两套不同职责的压缩状态

当 `compress_ratio == 4`，`MQALayer.__init__()` 会创建：

~~~text
core Compressor
+
C4Indexer
~~~

而 `C4Indexer` 内部还拥有**自己的 Compressor**。[源码：MQALayer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1115-L1173) [源码：C4Indexer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/dsv4/indexer.py#L1125-L1240)

所以 C4 更准确的准备链是：

~~~text
当前 hidden states
      |
      +--> indexer compressor
      |       |
      |       v
      |   indexer key/history
      |       |
      |       v
      |   C4 Indexer scoring
      |       |
      |       v
      |   c4_topk_indices
      |
      +--> core compressor
              |
              v
         C4 Attention compressed history
~~~

普通 NPU 单流路径中，这些工作都在主 Attention 调用之前完成。Indexer 最终把选择结果写入：

~~~python
self.forward_metadata.c4_topk_indices
~~~

见 [`forward_c4_indexer()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L875-L901)。

主 C4 Attention 随后既读取 SWA 历史，也读取 core compressor 产生的 C4 压缩历史；其中压缩侧通过：

~~~python
cmp_sparse_indices = c4_topk_indices
~~~

只访问 Indexer 选出的 compressed positions。[源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

因此“C4 Indexer”不是主 Attention cache 本身，它更像是：

> 给 C4 compressed history 生成稀疏读取索引的侧路检索器。

### ratio = 128：SWA + C128，但没有 C4 Top-K 索引

`compress_ratio == 128` 同样会创建 core Compressor，却不会创建 `C4Indexer`。

主 Attention 仍然同时拿：

~~~text
SWA history
+
C128 compressed history
~~~

但源码把：

~~~python
cmp_sparse_indices = None
~~~

因此没有 C4 那条 Top-K 稀疏索引路径。[源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

所以三类层的区别可以压缩成：

| 层类型 | 近期历史 | 压缩历史 | 额外 Indexer |
| --- | --- | --- | --- |
| ratio 0 | SWA | 无 | 无 |
| ratio 4 | SWA | C4 | C4 Top-K Indexer |
| ratio 128 | SWA | C128 | 无 C4 Top-K |

这也是为什么 DeepSeek-V4 的缓存不能直接套标准 MHA KV Cache 公式。真实运行时状态还包括 SWA pool、compressed pool、compressor state、C4 indexer state、page table、压缩布局与不同 dtype。

## 六、把一次 Decode 完整串起来：Query 怎样只读取 R1 自己的过去

最后把所有环节重新落回 R1 / R2。

当前 Decode：

~~~text
input_ids = [T1, T2]
positions = [6, 6]
seq_lens  = [7, 7]
~~~

某层收到：

~~~text
x.shape = [2, H]
~~~

第一步，生成 Query：

~~~text
x
[2,H]
  |
  | q low-rank projections
  v
q_lora
[2,Rq]
  |
  v
Q
[2,Nlocal,D]
  |
  | positions=[6,6]
  v
position-aware Q
~~~

第二步，生成当前缓存表示并先写入 SWA pool：

~~~text
x
  |
  | wkv / fused path
  v
current cache representation
  |
  | norm + RoPE
  v
SWA cache write
~~~

如果当前层是 ratio 4，还会在主 Attention 前完成：

~~~text
indexer compressor
      ↓
C4 Indexer Top-K

core compressor
      ↓
C4 compressed cache
~~~

如果是 ratio 128，则只需要相应的 core compressor 路径。

第三步，backend 根据：

~~~text
req_pool_indices
      ↓
req_to_token
      ↓
block / page tables
~~~

分别得到 R1 和 R2 的历史。

于是即使：

~~~text
Q.shape = [2, Nlocal, D]
~~~

两个 Query 在同一个 Tensor 中，它们的历史仍然是：

~~~text
Q[0] -> R1 page tables -> R1 cache
Q[1] -> R2 page tables -> R2 cache
~~~

第四步，根据当前层的 `compress_ratio` 选择历史结构：

~~~text
ratio 0   -> R1/R2 SWA
ratio 4   -> R1/R2 SWA + selected C4 history
ratio 128 -> R1/R2 SWA + C128 history
~~~

第五步，Attention backend 返回当前 token 对应的输出。随后 `MQALayer.forward()` 还会执行 DSV4 的 inverse-RoPE / `wo_a` / `wo_b` 等输出投影，最终重新回到 token-major hidden states，再交还 Decoder Layer。[源码：MQALayer 输出路径](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2360-L2585)

整条链最终是：

~~~mermaid
flowchart TD
    X[Layer hidden states<br/>T x H]

    X --> Q[Build current Query]
    X --> K[Build current cache representation]

    P[positions] --> Q
    P --> K

    K --> W[Write current SWA cache]

    F[ForwardBatch<br/>req_pool_indices / seq_lens] --> RT[ReqToTokenPool]
    RT --> PT[Block / Page Tables]

    W --> CP[Physical cache pools]
    PT --> CP

    R{this layer compress_ratio}
    R -->|0| S0[Read SWA]
    R -->|4| S4[Read SWA plus Top-K C4]
    R -->|128| S128[Read SWA plus C128]

    CP --> S0
    CP --> S4
    CP --> S128

    Q --> S0
    Q --> S4
    Q --> S128

    S0 --> O[Attention output]
    S4 --> O
    S128 --> O
~~~

图里三个分支是**不同层的备选路径**，不是同一个层同时执行三次 Attention。

这也回答了开头的问题。

`ForwardBatch` 本身不会变成 Query。它真正做的是把 packed token 放回各自请求的时空坐标系：

- hidden states 负责产生当前 Query；
- `positions` 决定当前 Query / cache 的逻辑位置；
- `req_pool_indices` 找到这条请求的映射行；
- `req_to_token` 把逻辑 token 位置翻译成物理 cache slot；
- page table 让 Attention kernel 定位历史页；
- `seq_lens` 和 Query/KV metadata 决定有效边界；
- cache write location 决定本轮新状态写到哪里；
- `compress_ratio` 决定这一层读取 SWA-only、SWA+C4，还是 SWA+C128。

Prefix Cache / Radix Cache 则位于更上一层：它负责判断哪些已有前缀缓存可以被另一个请求复用，并把可复用的 cache indices 交给后续请求状态。到了本篇这一层，Attention backend 并不重新“搜索 Radix Tree”；它消费的是 Scheduler / cache system 已经落实到 request-token mapping 和 page table 上的结果。

因此，一次 Decode 真正依赖的不是一句抽象的“读取 KV Cache”，而是一整条确定的数据关系：

~~~text
Req identity
    ↓
logical token positions
    ↓
physical cache mapping
    ↓
layer-specific history structure
    ↓
Attention operator
~~~

这条链才是在线推理系统能够让大量请求共享同一个 batch、却仍然各自记住自己过去的根本原因。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| Decoder Layer 调用 Attention | [`DeepseekV4DecoderLayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3006-L3189) |
| DSV4 Attention 基础参数 | [`MqaAttentionBase`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L742-L945) |
| `MQALayer` 初始化 | [`MQALayer.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1041-L1193) |
| Query 投影 | [`_compute_q_a()` / `_compute_q_b()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1291-L1347) |
| NPU Query / cache / compressor 准备 | [`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095) |
| Attention 主调用与 backend contract | [`MQALayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2097-L2585) |
| `k is v` backend 接口约束 | [`DeepseekV4AttnBackend._forward_attention()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L3715-L3750) |
| Req → Block Table | [`AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L479-L540) |
| DSV4 NPU Query/KV metadata | [`DeepseekV4AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1837-L2045) |
| DSV4 NPU Attention 分流 | [`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189) |
| SWA cache 写入 | [`store_cache()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2191-L2219) |
| C4 Indexer | [`C4Indexer`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/dsv4/indexer.py#L1125-L1240) |
| NPU C4 Top-K | [`forward_c4_indexer()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L875-L901) |
| C4 / C128 Compressor | [`Compressor`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/dsv4/compressor.py#L333-L492) |
| DSV4 Attention 测试与能力边界 | [Capability Matrix](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/attention/unittests/dsv4/README.md) |
