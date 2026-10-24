# DeepSeek-V4 Attention：Query、缓存写入与历史寻址

上一轮模型为 R1、R2 各生成一枚 Token，下一轮把它们打包成 `[T1,T2]` 送入设备。两枚 Token 在各自序列中都位于位置六，Attention 却要分别读取两套历史。理解这一步，需要同时追踪计算出来的 Query，以及让 Query 找到自己历史的请求映射。

这里接着 [ModelRunner 的九行输入例子](sglang-modelrunner-source-analysis.md)，观察 DeepSeek-V4 在普通 Decode 中的一层 Attention：先准备当前查询与缓存，再按请求定位历史，最后由该层的压缩配置决定读什么。文中的短序列用于解释坐标，不代表专用前缀缓存可以按任意 Token 数匹配。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线使用公开源码中的 DeepSeek-V4、普通文本生成、非投机推理、PP=1，并聚焦 Ascend NPU eager 的普通单流职责路径。图回放、多流、Context Parallel、unified FP8 KV 和 DeepSeek-V4.1 的 `compress_ratio=1/2` 会改变部分中间表示或时序，不在本文代表路径中展开。本文追到公开 NPU Attention operator 入口与缓存映射，不把这些代码外推成某个私有部署镜像的底层运行轨迹。

</details>

## 一、为打包的 Token 保留请求坐标

Prefill 已经为两条请求处理完各自六个位置，采样结果为 T1、T2。它们成为下一轮的模型输入，这轮执行使用 `positions=[6,6]`、`seq_lens=[7,7]`。序列位置从零计数，因此处理位置六后总长度是七。

经过 Embedding、mHC 和当前子层的归一化，Attention 收到 `x.shape=[2,H]`，其中 H 是隐藏维度。第一行属于 R1，第二行属于 R2。张量形状只说明有两行数据，不能单独表达哪行该读哪份历史，所以 Decoder Layer 还会传入 positions 与 `ForwardBatch`。固定版本的 `self.self_attn` 是 `MQALayer`。[`DeepseekV4DecoderLayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3006-L3189) [构造入口](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2693-L2710)

| 随输入传递的信息 | 参与解决的问题 |
| --- | --- |
| `positions` | 当前 Token 在自己的序列中位于哪里 |
| `seq_lens` | 每条请求到本轮推进到多长 |
| `req_pool_indices` | 到请求映射池中找哪一行 |
| `out_cache_loc` 与 DSV4 专用位置 | 新缓存写入哪些槽位 |
| Attention metadata | 每条请求的查询边界、历史页表和有效范围 |

后面会出现三种坐标：打包 Tensor 的行号、请求内部的逻辑位置、设备缓存的物理位置。R1 的 T1 位于打包行零，R2 的 T2 位于打包行一，但它们的逻辑位置都可以是六，物理缓存落点又由分配器决定。把三者分开，才能看清为什么打包没有让请求上下文混到一起。

DeepSeek-V4 的缓存形式还要对应模型实现。`MqaAttentionBase` 要求 `num_key_value_heads==1`，构造 Attention 参数时也使用 `num_kv_heads=1`。因此，这条路径有多个 Query head，却共享 KV 表示；不能按普通多头 Attention 为每个 Query head 各画一份独立 K/V 缓存。[`MqaAttentionBase`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L742-L945) [`MQALayer.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1041-L1193)

## 二、从 hidden states 生成 Query

Query 可以理解为当前输入用于访问历史的查询表示。`MQALayer` 先经过较窄的投影与归一化，再展开到各个本地 Query head；沿逻辑步骤看，形状如下：

```text
x [T,H]
   → wq_a → q_lora [T,Rq]
   → q_norm
   → wq_b → [T,Nlocal × D]
   → 整理为 Q [T,Nlocal,D]
```

T 是本轮新增 Token 行数，本例为二；Rq 是中间投影维度，Nlocal 是当前 Rank 的有效 Query head 数，D 是单个 head 的维度。实际配置可以把前段 Query 与 KV 投影融合为 `wqkv_a`，上图表达数据关系，不要求运行时一定出现相同数量的独立算子。[`_compute_q_a()` / `_compute_q_b()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1291-L1347)

普通 NPU 单流准备路径形成 Q 后，还会对 head 表示做归一化，并将 RoPE 作用到对应的位置编码分量。固定版本的测试矩阵以 448 维非旋转部分、64 维旋转部分组成 512 维 head；这组数字来自对应测试配置，不能据此替代所有 Checkpoint 的参数。[`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095) [DSV4 Attention Capability Matrix](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/attention/unittests/dsv4/README.md)

两行 Q 使用的 positions 都是六，并不会冲突。RoPE 使用的是各条序列内部的位置，R1 和 R2 的历史边界另由请求映射决定。后端还可能按内核要求补齐 head 维，再把输出切回有效部分；这里的 Nlocal 描述逻辑 head 数，不能直接当作所有设备调用的实参宽度。

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

到这里，当前查询已经形成。下一步要准备它能读取的状态，其中也包含本轮新增 Token 的缓存表示。

## 三、写入当前 Token 的缓存状态

同一份 x 还会经过 `wkv` 产生共享缓存表示。未融合时，`wkv` 的输出宽度是 `head_dim`，因此本例的概念形状为 `[2,D]`；NPU 路径继续执行 `kv_norm` 与 RoPE，再通过 `store_cache()` 写入对应层的 SWA 缓存。SWA 指滑窗 Attention，保留近期内容的细粒度表示。[`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095)

在本文限定的普通单流路径中，顺序是：

```text
当前层输入 x
  → 生成 Q 与共享缓存表示
  → 写入本轮 SWA 槽位
  → 按当前层需要准备 indexer / compressor 状态
  → 调用主 Attention 后端读取缓存
```

这也适用于理解 Prefill：整个 chunk 的缓存可以先写入池，再由查询边界、有效长度与因果约束限制各行能读哪些位置。“存进去了”和“当前查询能够访问”属于不同条件，不能用写入顺序代替可见范围。

读 `MQALayer.forward()` 时会遇到一段看起来容易矛盾的代码：准备路径写完缓存后，将局部 `kv` 设为 `None`，后续却仍向统一接口传入 k、v。关键是看后端从哪里取数据。[`MQALayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2097-L2585)

```python
# 摘录接口交接关系，省略其他参数与平台分支。
attn_k = kv if kv is not None else q
# 随后向后端传入 k=attn_k、v=attn_k。
```

这条普通 NPU 路径已经提前写入缓存，并传入 `save_kv_cache=False`。此时 q 是接口占位对象，主 Attention 从 Cache Pool 读取历史，不是把当前 Query 当成整段历史 K/V。通用 DSV4 后端的 `k is v` 检查描述的是参数对象共享；它也不能单独说明底层缓存怎样存储。[`DeepseekV4AttnBackend._forward_attention()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L3715-L3750)

因此，分析这一步应同时确认两处：谁负责写缓存，主 Attention 又从哪个池读取。只沿局部变量名 k、v 追踪，容易错过真正的数据来源。下面继续沿请求索引找到这些历史位置。

## 四、用请求映射和页表找到历史

`req_pool_indices` 把当前 batch 中的请求指向 request-token 映射池。元数据准备阶段再从对应请求行取得 Token 到缓存槽位的映射，构造 Attention 使用的页表。映射池保存关系，KV Pool 保存实际表示，两者共同把逻辑位置接到物理存储。[`AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L479-L540)

例如，请求池索引可以是 `req_pool_indices=[5,9]`。那么打包行零的 Q 沿请求行五定位 R1 历史，打包行一的 Q 沿请求行九定位 R2 历史。五和九是教学用索引，与当前 batch 的零、一行号无须相同，也不是物理页地址。

```text
Q[0] → R1 → 请求映射行 5 → R1 的历史页 → R1 的缓存表示
Q[1] → R2 → 请求映射行 9 → R2 的历史页 → R2 的缓存表示
```

即使两条请求同在一个打包 Tensor，后端仍按各自映射与有效范围访问历史。相邻存储只改变输入组织方式，不会取消请求边界。对于 DeepSeek-V4，后续还要结合滑窗及对应压缩缓存的页表，不能把基础请求映射当成全部寻址过程。

### Prefill 与 Decode 的 metadata 差异

除了找到历史，后端还需要知道每条请求在本轮占多少 Query 行。这个数量与请求当前总长度不同：此前 Prefill 中 R1 新增六行，R2 新增三行，两条请求的总长度却都是六。`DeepseekV4AscendAttnBackend.init_forward_metadata()` 为 Query 与 KV 分别准备长度信息。[`DeepseekV4AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1837-L2045)

普通 Prefill 先对本轮新增长度 `[6,3]` 累计，得到 Query 结束位置 `[6,9]`，再在前面补零得到 `[0,6,9]`。这组边界把九行 Q 划成两段。下一轮普通 Decode 每请求只有一行，边界就变成 `[0,1,2]`。

| 后端元数据 | 普通 Prefill | 下一轮普通 Decode |
| --- | --- | --- |
| 本轮打包 Query 行数 | 9 | 2 |
| 每请求新 Query 数 | R1=6，R2=3 | R1=1，R2=1 |
| `actual_seq_lengths_q` | `[6,9]` | `[1,2]` |
| `actual_seq_lengths_q_pa` | `[0,6,9]` | `[0,1,2]` |
| `actual_seq_lengths_kv` | `[6,6]` | `[7,7]` |

表中 Query 行数回答“这轮算多少”，KV 有效长度回答“各请求已经推进到哪里”，缓存池容量回答“设备上还能放多少”。它们不能互换。这里采用普通、无投机、无 CP 和额外 padding 的示例；Target Verify 等模式的查询组织另有分支，不直接套用每请求一行。

## 五、按层选择 SWA、C4 与 C128

有了请求边界和历史位置，还要确定当前层使用哪种历史表示。本文固定版本的 Ascend DSV4 主 Attention 接口接受 `compress_ratio=0/4/128`：每层按自己的配置选择一类路径，而不是每层都把三类依次执行一遍。[`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189)

### ratio = 0：只读取 SWA

比例为零时进入 `_forward_swa()`，从池中取当前层 SWA buffer，连同查询边界、KV 有效长度、滑窗页表及窗口参数交给 NPU Attention 算子。近期历史保留在这个滑窗路径中。固定版本在模型未显式提供 sliding-window 配置时使用 128 作为 DSV4 fallback window；模型提供配置时以相应值为准。[源码：`_forward_swa()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2077-L2116)

### ratio = 4：SWA + C4，且 C4 有两套不同职责的压缩状态

C4 层既有主 Attention 使用的 core Compressor，也有 `C4Indexer`，后者内部还包含自己的 Compressor。主缓存压缩与索引器压缩服务于不同的数据用途，不能看到两个 Compressor 就把它们算成同一份状态。[源码：MQALayer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1115-L1173) [源码：C4Indexer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/dsv4/indexer.py#L1125-L1240)

core Compressor 产生 C4 压缩历史，Indexer 则为当前查询选择要访问的压缩位置。普通单流路径在主 Attention 之前准备这些状态，索引结果写入 `forward_metadata.c4_topk_indices`；后端将其作为 `cmp_sparse_indices`，限制压缩侧的读取位置。SWA 历史仍同时参与 Attention。[`forward_c4_indexer()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L875-L901) [源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

这层的 TopK 选择的是压缩历史位置，与 MoE Router 选择专家是两套不同机制。把它放回本轮请求中看：当前 Q 已经确定，Indexer 帮助定位它需要读取的部分压缩历史，主 Attention 再消费这些表示。

### ratio = 128：SWA + C128，但没有 C4 Top-K 索引

C128 层同样构造 core Compressor，保留压缩历史，但不创建 C4Indexer。后端同时传入 SWA 与 C128 缓存，将 `cmp_sparse_indices` 设为 `None`，因此没有 C4 的 TopK 位置筛选步骤。读取仍受请求、长度和相应算子规则约束，不是访问其他请求的压缩缓存。[源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

| 当前层类型 | 近期历史 | 压缩历史 | 额外索引器 |
| --- | --- | --- | --- |
| ratio 0 | SWA | 无 | 无 |
| ratio 4 | SWA | C4 | C4 TopK |
| ratio 128 | SWA | C128 | 无 C4 TopK |

压缩比例描述历史表示方式，不表示每生成一枚 Token 就必定增加一条完整压缩记录。计算缓存占用时，还要分别核算 SWA、压缩池、Compressor State、C4 Indexer State 和页表，结合实际布局、dtype 与分配情况求和。仅用序列长度乘上一份普通多头 KV 大小，无法覆盖这里的状态。

## 六、把请求坐标与 Attention 输出连起来

回到本轮的 T1、T2：它们形成两行当前查询，分别在各自请求的逻辑位置六写入新增状态，再通过请求映射、有效长度与当前层页表读取历史。比例为零的层走 SWA；C4 层还读取 Indexer 选中的压缩历史；C128 层读取相应压缩表示。

Attention 结果随后经过 inverse-RoPE 及 `wo_a`、`wo_b` 等输出处理，回到 Token 对应的 hidden states，再交给后续子层。输出行的对应关系仍与输入一致：R1 的结果回到 R1，R2 的结果回到 R2。[Attention 后的 inverse-RoPE、`wo_a`、`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2360-L2585)

Prefix Cache 的匹配发生在更上层，到当前 Attention 执行时，复用结果已经体现在请求映射、页表和相关状态里。这里不需要每算一次 Query 就重新搜索 Radix Tree。排查历史访问问题时，可以沿本轮写入位置、请求映射、Query 边界和层压缩类型检查，逐步确认读到的是哪条请求的哪种历史。

下一步看 [MoE 的路由与专家计算](deepseek-v4-moe-source-analysis.md)：Attention 仍按请求历史组织访问，MoE 则根据专家选择把当前 Token 送往不同计算位置，再将贡献合回原行。

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
