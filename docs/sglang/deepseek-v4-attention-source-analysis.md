# DeepSeek-V4 Attention：Query、缓存写入与历史寻址

一个 batch 可以把多条请求的 Token 放进同一个 Tensor，Attention 却必须让每一行读取自己的历史。这里有两条相互配合的数据流：hidden states 生成 Query，请求索引、位置和页表决定 Query 能访问哪些缓存。

下面沿 DeepSeek-V4 的普通 Decode 路径追踪这两条线，先生成 Query 并写入当前状态，再找到请求历史，最后按当前层的配置组合滑窗与压缩信息。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线使用公开源码中的 DeepSeek-V4、普通文本生成、非投机推理、PP=1，并聚焦 Ascend NPU eager 的普通单流职责路径。图回放、多流、Context Parallel、unified FP8 KV 和 DeepSeek-V4.1 的 `compress_ratio=1/2` 会改变部分中间表示或时序，不在本文代表路径中展开。本文追到公开 NPU Attention operator 入口与缓存映射，不把这些代码外推成某个私有部署镜像的底层运行轨迹。

</details>

## 一、为打包的 Token 保留请求坐标

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

经过 Embedding、mHC 和输入层归一化后，某一层 Attention 收到：`x.shape = [2, H]`

其中 `x[0]` 属于 R1，`x[1]` 属于 R2。问题在于，单看 `x` 自己并没有“请求身份”。把它们区分开的，是继续同行的 `ForwardBatch`：

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

当前实现有多个 Query Heads，共享一套 KV / latent-cache 表示。后文沿用源码中的 `kv`、`k`、`v` 命名，并在读取缓存时说明它们对应的具体对象。

## 二、从 hidden states 生成 Query

本节形状描述模型的逻辑 head 维。后端按算子要求补齐 Head 维后，设备实参可能比模型的逻辑 Head 数更大，输出再切回有效部分。

`MQALayer` 的 Query 主干采用低秩两阶段投影：

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

固定版本的辅助函数可以在 [`_compute_q_a()` / `_compute_q_b()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1291-L1347) 找到。实际配置可以把第一阶段 Query 和 KV 投影融合成 `wqkv_a`，因此图中按逻辑步骤画出数据流，融合后的执行入口可合并相邻投影。

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

当前两个 Decode token：`positions = [6, 6]`

并不冲突。position 是“每条请求内部的逻辑位置”，不是 packed batch 的全局行号：

~~~text
Q[0] -> R1 position 6
Q[1] -> R2 position 6
~~~

所以同一个 batch 中可以同时存在多个 position 6。

~~~mermaid
flowchart TD
    X[hidden states · T x H]
    X --> QA[wq_a / fused qkv_a]
    QA --> QL[q_lora · T x Rq]
    QL --> QN[q_norm]
    QN --> QB[wq_b]
    QB --> Q[Query · T x Nlocal x D]

    P[ForwardBatch.positions] --> R[RoPE / position transform]
    Q --> R
    R --> QR[position-aware Query]
~~~

当前 Query 已经准备好，接下来从长期缓存池读取历史。这些状态由此前 Prefill / Decode 逐步写入。

## 三、写入当前 Token 的缓存状态

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

Prefill 可以先将整个 Chunk 的缓存写入 Pool，再由 Attention Metadata、序列边界和 Mask 限定每个 Query 的可见范围。存储已经就绪，因果访问仍按各 Query 的位置控制。

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

这里需要同时看参数对象与后端读取位置。

源码确实把同一个对象同时传给 `k` 和 `v`。通用 DSV4 backend 甚至有：

~~~python
assert k is v, "DeepseekV4 shares k and v"
~~~

见 [`DeepseekV4AttnBackend._forward_attention()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L3715-L3750)。

普通 NPU 路径已提前把缓存写入 Pool，并设置 `save_kv_cache=False`。此时 `attn_k=q` 用作统一接口的 sentinel，主 Attention 从 Cache Pool 读取历史。

因此，这里的 `k is v` 描述接口对象共享。理解其数据含义时，继续追踪后端访问的缓存表示和 Pool。

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

## 四、用请求映射和页表找到历史

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

那么即使当前：`input_ids = [T1, T2]`

两个 token 在同一个 packed Tensor 中，backend 仍会分别沿：

~~~text
req_to_token[3, ...]
req_to_token[8, ...]
~~~

构造两套历史映射。

~~~text
Q0 -> R1 page table -> R1 cache pages
Q1 -> R2 page table -> R2 cache pages
~~~

而不会变成：`Q0 / Q1 -> 同一整块历史`

这就是 packed batching 不会导致请求上下文串线的关键原因。

### Prefill 与 Decode 的 metadata 差异

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

## 五、按层选择 SWA、C4 与 C128

每层按压缩配置选择 SWA、SWA+C4 或 SWA+C128，后端再组织对应的访问。

每个 `MQALayer` 都有自己的：

~~~python
self.compress_ratio = config.compress_ratios[layer_id]
~~~

对于本文 DeepSeek-V4 + Ascend backend 主线，主 Attention 接口接受：`compress_ratio = 0 / 4 / 128`

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

~~~text
Layer A: ratio 0
-> SWA only

Layer B: ratio 4
-> SWA + C4

Layer C: ratio 128
-> SWA + C128
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

固定版本 backend 在模型没有显式 sliding-window 配置时使用 128 作为 DSV4 fallback window。这一窗口保留近期高分辨率历史；模型显式提供 sliding-window 配置时，使用相应配置值。

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

C4 Indexer 为主 Attention 选择要读取的压缩历史位置：

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

计算 DeepSeek-V4 的缓存容量时，分别核算 SWA Pool、Compressed Pool、Compressor State、C4 Indexer State 和页表，再结合各自布局与 dtype 求和。

## 六、把请求坐标与 Attention 输出连起来

回到 R1、R2 的普通 Decode：`input_ids=[T1,T2]`，`positions=[6,6]`，`seq_lens=[7,7]`。两行 hidden states 可以在同一 Tensor 中计算，读取历史时则分别使用各自的请求映射。

| 当前对象 | 本轮发生的变化 | 保持请求隔离的依据 |
| --- | --- | --- |
| hidden states | 投影成当前 Query 与缓存表示 | Token 行仍与请求对应 |
| 当前缓存 | 写入各自的 SWA 槽位，并更新该层需要的压缩状态 | 本轮写入位置 |
| 历史寻址 | 查到 SWA 及当前层需要的压缩条目 | 请求映射、页表和有效长度 |
| Attention 输出 | 经过输出投影，回到 token-major hidden states | 每行输出对应原输入 Token |

`compress_ratio=0/4/128` 分别选择 SWA、SWA 加选中的 C4 历史、SWA 加 C128 历史。它们是不同层的配置选择。[Attention 后的 inverse-RoPE、`wo_a`、`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2360-L2585) 将结果送回 Decoder Layer，继续下一子层。

Prefix Cache 的匹配发生在更上层。到 Attention 执行时，可复用前缀已经落实为请求映射与页表，后端消费这些结果，不在每次 Query 计算时重新搜索 Radix Tree。

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
