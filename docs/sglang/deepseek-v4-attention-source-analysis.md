# DeepSeek-V4 Attention：Query、缓存写入与历史寻址

ModelRunner 上一轮为 R1、R2 各采样了一枚输出，下一轮将 `[T1,T2]` 打包送入设备。两枚 Token 在自己的序列中都位于位置六，却必须访问不同的历史。沿着 Query 与请求映射一起读，才能知道 Attention 怎样保持这两条请求的边界。

下面继续前文的两条请求，跟踪普通 Decode 的一层 Attention：先形成当前查询并写缓存，再通过请求映射定位历史，最后看本层压缩配置决定读取什么。短序列只用于解释坐标，不代表专用前缀缓存支持任意 Token 数的匹配。前例见 [ModelRunner 的九行输入例子](sglang-modelrunner-source-analysis.md)。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线使用公开 DeepSeek-V4 实现、普通文本生成、非投机、PP=1，沿 Ascend NPU eager 的普通单流路径阅读。图回放、多流、Context Parallel、unified FP8 KV 和 DeepSeek-V4.1 的 `compress_ratio=1/2` 可能改变中间表示或时序，不在这条代表路径内。下文追到公开 NPU Attention operator 入口和缓存映射，不据此推断私有镜像的底层执行轨迹。

</details>

## 一、为打包的 Token 保留请求坐标

Prefill 已处理两条请求各自的六个位置，得到 T1、T2。下一轮把它们作为输入，使用 `positions=[6,6]`、`seq_lens=[7,7]`：位置从零开始，所以算完位置六后，每条序列总长为七。

经过 Embedding、mHC 和当前子层归一化，Attention 收到 `x.shape=[2,H]`，H 是隐藏维度。第一个输入行属于 R1，第二个属于 R2；仅凭两行向量无法知道该读谁的历史，因此 Decoder Layer 还传入 positions 与 `ForwardBatch`。固定源码中的 `self.self_attn` 实例是 `MQALayer`。[`DeepseekV4DecoderLayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3006-L3189) [构造入口](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2693-L2710)

| 随输入传递的信息 | 参与解决的问题 |
| --- | --- |
| `positions` | 当前 Token 在自己的序列中位于哪里 |
| `seq_lens` | 每条请求到本轮推进到多长 |
| `req_pool_indices` | 到请求映射池中找哪一行 |
| `out_cache_loc` 与 DSV4 专用位置 | 新缓存写入哪些槽位 |
| Attention metadata | 每条请求的查询边界、历史页表和有效范围 |

下面要区分打包行号、序列逻辑位置和缓存物理位置。T1、T2 在打包 Tensor 中分别是行零、行一，序列位置却都为六；缓存最终写到哪里则由分配器决定。三套坐标通过映射关联，并不要求取值相同。

这条路径还要求 `num_key_value_heads==1`，构造 Attention 时使用 `num_kv_heads=1`。因此多个 Query head 共享 KV 表示，不能按普通多头 Attention 给每个 Query head 单独画一份 K/V 缓存。[`MqaAttentionBase`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L742-L945) [`MQALayer.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1041-L1193)

## 二、从 hidden states 生成 Query

Query 是当前输入访问历史所用的表示。`MQALayer` 先做较窄的投影和归一化，再展开到本 Rank 的 Query heads，逻辑过程如下：

```text
x [T,H]
   → wq_a → q_lora [T,Rq]
   → q_norm
   → wq_b → [T,Nlocal × D]
   → 整理为 Q [T,Nlocal,D]
```

T 表示本轮新输入行数，本例为二；Rq 是中间投影维度，Nlocal 是本 Rank 的有效 Query head 数，D 是单个 head 的维度。实际配置可能用 `wqkv_a` 融合前段 Query、KV 投影，所以示意中的逻辑步骤未必各自对应一个独立算子。[`_compute_q_a()` / `_compute_q_b()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1291-L1347)

普通 NPU 单流准备得到 Q 后，还对 head 表示归一化，并在相应分量上应用 RoPE。固定版本测试矩阵采用 448 维非旋转部分和 64 维旋转部分，合成 512 维 head；这是该测试配置，其他 Checkpoint 仍需看自己的参数。[`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095) [DSV4 Attention Capability Matrix](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/attention/unittests/dsv4/README.md)

两行都使用位置六不影响请求边界：RoPE 描述各自序列的位置，请求映射决定历史属于谁。此外，后端可能按内核要求补齐 head 维，再将输出切回有效范围，因此逻辑 Nlocal 不一定等于设备调用的实参宽度。

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

当前 Q 准备好后，再看供它读取的缓存，其中也包括本轮新增 Token 的状态。

## 三、写入当前 Token 的缓存状态

同一个 x 经 `wkv` 形成共享缓存表示。未融合时输出宽度为 `head_dim`，本例概念形状是 `[2,D]`；NPU 路径接着做 `kv_norm`、RoPE，再由 `store_cache()` 写入对应层的 SWA 缓存。SWA 即滑窗 Attention，用来保存近期内容的细粒度表示。[`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095)

本文普通单流路径按以下顺序处理：

```text
当前层输入 x
  → 生成 Q 与共享缓存表示
  → 写入本轮 SWA 槽位
  → 按当前层需要准备 indexer / compressor 状态
  → 调用主 Attention 后端读取缓存
```

Prefill 也可以先写入整个 chunk，再用 Query 边界、有效长度和因果约束限制可访问的位置。某个状态已经在池里，并不表示当前行就能读取它，不能用写入顺序代替因果可见范围。

继续读 `MQALayer.forward()`，会发现准备阶段写完后将局部 `kv` 设成 `None`，后面却仍给统一接口传 k、v。要解释这段代码，需要追到后端实际取数据的位置。[`MQALayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2097-L2585)

```python
# 摘录接口交接关系，省略其他参数与平台分支。
attn_k = kv if kv is not None else q
# 随后向后端传入 k=attn_k、v=attn_k。
```

普通 NPU 路径已提前写缓存，并设置 `save_kv_cache=False`。这里传入的 q 起接口占位作用，主 Attention 真正读取历史的位置是 Cache Pool。通用 DSV4 后端的 `k is v` 只约束参数对象共享，不能据此判断底层缓存布局，更不能把 Query 当成整段历史 K/V。[`DeepseekV4AttnBackend._forward_attention()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L3715-L3750)

因此，需要把缓存写入者和主 Attention 的读取池接起来看。局部变量 k、v 的名字还不足以说明数据来源，下面继续沿请求索引追踪物理历史。

## 四、用请求映射和页表找到历史

`req_pool_indices` 先指向 request-token 映射池中的请求行，metadata 准备阶段据此取得 Token 到缓存槽位的映射，构造页表。映射池保存位置关系，KV Pool 保存实际表示，两者共同完成逻辑到物理的寻址。[`AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L479-L540)

例如取教学索引 `req_pool_indices=[5,9]`，打包行零的 Q 沿请求行五查 R1，打包行一沿请求行九查 R2。请求池索引五、九既不同于 batch 行号零、一，也不是物理页地址。

```text
Q[0] → R1 → 请求映射行 5 → R1 的历史页 → R1 的缓存表示
Q[1] → R2 → 请求映射行 9 → R2 的历史页 → R2 的缓存表示
```

后端始终按各请求的映射和有效范围读取，因此打包相邻不会取消边界。DeepSeek-V4 还要继续使用滑窗、压缩缓存各自的页表，基础 request-token 映射只完成其中一部分。

### Prefill 与 Decode 的 metadata 差异

metadata 还要划出每条请求当前占哪些 Query 行。前面的 Prefill 中，R1 新增六行、R2 新增三行，两条总长度却都是六，所以 Query 长度和 KV 长度需分别准备。[`DeepseekV4AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1837-L2045)

Prefill 将新增行数 `[6,3]` 累计为 `[6,9]`，再前置零得到 `[0,6,9]`，用这组边界划开九行 Q。普通 Decode 每请求一行，边界就变为 `[0,1,2]`。

| 后端元数据 | 普通 Prefill | 下一轮普通 Decode |
| --- | --- | --- |
| 本轮打包 Query 行数 | 9 | 2 |
| 每请求新 Query 数 | R1=6，R2=3 | R1=1，R2=1 |
| `actual_seq_lengths_q` | `[6,9]` | `[1,2]` |
| `actual_seq_lengths_q_pa` | `[0,6,9]` | `[0,1,2]` |
| `actual_seq_lengths_kv` | `[6,6]` | `[7,7]` |

Query 行数描述本轮要计算多少，KV 有效长度描述请求已推进到哪里，缓存池容量则描述还能存放多少状态。表中没有投机、CP 或额外 padding；Target Verify 等模式有自己的 Query 组织，不能继续套每请求一行。

## 五、按层选择 SWA、C4 与 C128

位置确定后，再看这一层用哪种历史。固定 Ascend DSV4 主接口支持 `compress_ratio=0/4/128`，每层按配置选择其中一条路径，并非逐层把三种都执行一遍。[`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189)

### ratio = 0：只读取 SWA

ratio 为零时进入 `_forward_swa()`，取本层 SWA buffer，携带 Query 边界、KV 有效长度、滑窗页表和窗口参数调用 NPU Attention。若模型未明确给 sliding-window，固定实现采用 DSV4 fallback window 128；明确提供时则使用模型配置。[源码：`_forward_swa()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2077-L2116)

### ratio = 4：SWA + C4，且 C4 有两套不同职责的压缩状态

C4 层同时有主 Attention 的 core Compressor 和 `C4Indexer` 内部的 Compressor。前者准备主压缩缓存，后者服务索引器，两份状态用途不同，不能只因名称相同就合算成一份。[源码：MQALayer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1115-L1173) [源码：C4Indexer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/dsv4/indexer.py#L1125-L1240)

core Compressor 生成 C4 压缩历史，Indexer 为当前 Q 选择压缩位置。单流路径在主 Attention 前准备好它们，并把结果写到 `forward_metadata.c4_topk_indices`。后端将结果作为 `cmp_sparse_indices` 限定压缩侧读取，同时仍读取 SWA 历史。[`forward_c4_indexer()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L875-L901) [源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

这里的 TopK 选择历史位置，MoE Router 的 TopK 则选择专家。对本轮 Q 来说，Indexer 先确定压缩历史的读取位置，主 Attention 再消费相应表示。

### ratio = 128：SWA + C128，但没有 C4 Top-K 索引

C128 也有 core Compressor，但不创建 C4Indexer。后端携带 SWA、C128 缓存，将 `cmp_sparse_indices=None`，因此不经过 C4 的 TopK 筛选。读取仍受本请求、有效长度和算子规则限制。[源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

| 当前层类型 | 近期历史 | 压缩历史 | 额外索引器 |
| --- | --- | --- | --- |
| ratio 0 | SWA | 无 | 无 |
| ratio 4 | SWA | C4 | C4 TopK |
| ratio 128 | SWA | C128 | 无 C4 TopK |

压缩比例描述历史表示，不意味着每产生一个输出 Token 就增加一条完整压缩记录。容量核算还需分别检查 SWA、压缩池、Compressor State、C4 Indexer State、页表的布局、dtype 和实际分配，普通多头 KV 的单一公式无法覆盖这些状态。

## 六、把请求坐标与 Attention 输出连起来

回看 T1、T2，两行分别形成 Query、在各自位置六写入新增状态，再通过请求映射、长度和本层页表读取历史。ratio 0 读 SWA，C4 额外读取 Indexer 选中的压缩位置，C128 读取对应压缩表示。

Attention 结果经 inverse-RoPE、`wo_a`、`wo_b` 等输出处理，回到各 Token 的 hidden states，再进入后续子层。输出仍保持原行的请求对应关系。[Attention 后的 inverse-RoPE、`wo_a`、`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2360-L2585)

Prefix Cache 匹配已在更上层完成，Attention 使用的是复用后准备好的映射、页表和状态，不需要每个 Query 重新搜索 Radix Tree。排查历史访问时，可以按本轮写入位置、请求映射、Query 边界和层压缩类型，逐项确认读到了哪份历史。

下一篇转到 MoE：它将当前 Token 按专家选择重新分发，再把贡献合回原行。具体路径见 [MoE 的路由与专家计算](deepseek-v4-moe-source-analysis.md)。

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
