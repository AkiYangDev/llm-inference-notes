# DeepSeek-V4 Attention：Query、缓存写入与历史寻址

T1、T2 在各自序列里的位置都是六，一起打包成 `[T1,T2]` 送入设备。Attention 读历史时，T1 要找到 R1 的缓存，T2 要找到 R2 的缓存。相同的位置、相邻的输入行，靠什么分开？

这两枚 Token 来自上一轮 ModelRunner 的采样，前例见 [ModelRunner 的九行输入例子](sglang-modelrunner-source-analysis.md)。这里跟到普通 Decode 的一层 Attention，看 Query 与请求映射怎样一起确定读取范围。短序列只演示坐标，不代表专用前缀缓存能按任意 Token 数匹配。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线使用公开 DeepSeek-V4 实现、普通文本生成、非投机、PP=1，沿 Ascend NPU eager 的普通单流路径阅读。图回放、多流、Context Parallel、unified FP8 KV 和 DeepSeek-V4.1 的 `compress_ratio=1/2` 可能改变中间表示或时序，不在这条代表路径内。下文追到公开 NPU Attention operator 入口和缓存映射，不据此推断私有镜像的底层执行轨迹。

</details>

## 一、为打包的 Token 保留请求坐标

Prefill 已处理完两条请求各自的六个位置。把 T1、T2 作为新输入时，`positions=[6,6]`，`seq_lens=[7,7]`，从零开始的位置六算完，各序列总长就是七。

经过 Embedding、mHC 和当前子层归一化，Attention 收到 `x.shape=[2,H]`，H 为隐藏维度。行零属于 R1，行一属于 R2，Decoder Layer 还把 positions、`ForwardBatch` 一并传入，提供这两行各自的历史关系。固定源码的 `self.self_attn` 实例是 `MQALayer`。[`DeepseekV4DecoderLayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L3006-L3189) [构造入口](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2693-L2710)

| 随输入传递的信息 | 参与解决的问题 |
| --- | --- |
| `positions` | 当前 Token 在自己的序列中位于哪里 |
| `seq_lens` | 每条请求到本轮推进到多长 |
| `req_pool_indices` | 到请求映射池中找哪一行 |
| `out_cache_loc` 与 DSV4 专用位置 | 新缓存写入哪些槽位 |
| Attention metadata | 每条请求的查询边界、历史页表和有效范围 |

T1、T2 的打包行号分别为零、一，序列内位置却都是六。最终缓存槽位另由分配器决定，三种坐标通过映射相连，数值不必相同。

这条路径要求 `num_key_value_heads==1`，构造 Attention 也设置 `num_kv_heads=1`。多个 Query head 共享 KV 表示，每个 Query head 不再各有一份普通多头 K/V 缓存。[`MqaAttentionBase`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L742-L945) [`MQALayer.__init__()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1041-L1193)

## 二、从 hidden states 生成 Query

`MQALayer` 把当前输入投影成访问历史所用的 Query，先做较窄投影和归一化，再展开到本 Rank 的 Query heads。

```text
x [T,H]
   → wq_a → q_lora [T,Rq]
   → q_norm
   → wq_b → [T,Nlocal × D]
   → 整理为 Q [T,Nlocal,D]
```

T 是本轮新输入行数，这里为二，Rq 是中间投影维度，Nlocal 为本 Rank 有效 Query head 数，D 为单 head 维度。配置也可能用 `wqkv_a` 融合 Query、KV 的前段投影，因此这些逻辑步骤未必各对应一个独立算子。[`_compute_q_a()` / `_compute_q_b()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1291-L1347)

普通 NPU 单流得到 Q 后，还对 head 表示归一化，在相应分量上应用 RoPE。固定版本测试矩阵使用 448 维非旋转部分、64 维旋转部分，合成 512 维 head；其他 Checkpoint 的宽度各看自身参数。[`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095) [DSV4 Attention Capability Matrix](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/attention/unittests/dsv4/README.md)

两行用同一个位置六做 RoPE，各自的历史归属仍由请求映射决定。后端还可能按内核要求补齐 head 维，输出再切回有效范围，因此设备实际调用的 head 宽度未必就是逻辑 Nlocal。

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

## 三、写入当前 Token 的缓存状态

同一份 x 还经过 `wkv`，形成共享缓存表示。未融合时输出宽度为 `head_dim`，本例概念形状为 `[2,D]`。NPU 接着做 `kv_norm`、RoPE，调用 `store_cache()` 写入这一层 SWA，保存近期内容的细粒度表示。[`MQALayer._forward_prepare()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1750-L2095)

普通单流的顺序如下。

```text
当前层输入 x
  → 生成 Q 与共享缓存表示
  → 写入本轮 SWA 槽位
  → 按当前层需要准备 indexer / compressor 状态
  → 调用主 Attention 后端读取缓存
```

Prefill 可以先把整个 chunk 写入池，再用 Query 边界、有效长度和因果约束控制读取。后面的状态虽然已写好，当前行也只能访问因果范围内的部分。

准备阶段写完缓存，`MQALayer.forward()` 会把局部 `kv` 设为 `None`，后面仍给统一接口传 k、v。这里的参数和实际历史存储，需要沿后端接着看。[`MQALayer.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2097-L2585)

```python
# 摘录接口交接关系，省略其他参数与平台分支。
attn_k = kv if kv is not None else q
# 随后向后端传入 k=attn_k、v=attn_k。
```

普通 NPU 已经提前写缓存，设置 `save_kv_cache=False`，传入的 q 只作接口占位。主 Attention 从 Cache Pool 读历史。通用 DSV4 后端的 `k is v` 约束参数对象共享，单靠它无法确定缓存布局，也不能把占位 Query 理解成整段历史 K/V。[`DeepseekV4AttnBackend._forward_attention()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/deepseek_v4_backend.py#L3715-L3750)

## 四、用请求映射和页表找到历史

metadata 准备时，先用 `req_pool_indices` 找 request-token 映射池中的请求行，取得 Token 到槽位的映射，再构造页表。映射池存位置关系，KV Pool 存实际表示，两部分一起完成寻址。[`AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py#L479-L540)

假设教学索引为 `req_pool_indices=[5,9]`，输入行零沿请求行五查 R1，输入行一沿请求行九查 R2。五和九是请求池索引，不是 batch 行号，也不是物理页地址。

```text
Q[0] → R1 → 请求映射行 5 → R1 的历史页 → R1 的缓存表示
Q[1] → R2 → 请求映射行 9 → R2 的历史页 → R2 的缓存表示
```

每个 Query 沿本请求的映射和有效范围读取。DeepSeek-V4 还分别用滑窗、压缩缓存的页表，基础 request-token 映射只完成其中一部分。Prefix Cache 匹配已经在更上层做完，Attention 使用复用后准备好的映射、页表和状态，不会为每个 Query 重新搜索 Radix Tree。

### Prefill 与 Decode 的 metadata 差异

metadata 还划分当前 Query 行。Prefill 例中 R1 新增六行，R2 新增三行，两条总长都是六，因此 Query 长度与 KV 长度分开准备。[`DeepseekV4AscendAttnBackend.init_forward_metadata()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L1837-L2045)

新增行数 `[6,3]` 累计成 `[6,9]`，前置零得到 `[0,6,9]`，划开九行 Q；普通 Decode 每请求一行，边界改为 `[0,1,2]`。

| 后端元数据 | 普通 Prefill | 下一轮普通 Decode |
| --- | --- | --- |
| 本轮打包 Query 行数 | 9 | 2 |
| 每请求新 Query 数 | R1=6，R2=3 | R1=1，R2=1 |
| `actual_seq_lengths_q` | `[6,9]` | `[1,2]` |
| `actual_seq_lengths_q_pa` | `[0,6,9]` | `[0,1,2]` |
| `actual_seq_lengths_kv` | `[6,6]` | `[7,7]` |

Query 行数是本轮计算量，KV 有效长度是请求进度，池容量是可分配空间。表中不含投机、CP 或额外 padding，Target Verify 等模式会另外组织 Query，未必每请求一行。

## 五、按层选择 SWA、C4 与 C128

Ascend DSV4 主接口在固定版本支持 `compress_ratio=0/4/128`。每一层按自身配置选一条历史读取路径，三种方式不会在同一层依次全部执行。[`DeepseekV4AscendAttnBackend.forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2047-L2189)

### ratio = 0：只读取 SWA

ratio 为零，后端进入 `_forward_swa()`，取本层 SWA buffer，带着 Query 边界、KV 有效长度、滑窗页表与窗口参数调用 NPU Attention。模型明确提供 sliding-window，就用模型配置；没有提供时，这份实现用 DSV4 fallback window 128。[源码：`_forward_swa()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2077-L2116)

### ratio = 4：SWA + C4，且 C4 有两套不同职责的压缩状态

C4 层有两个 Compressor。core Compressor 准备主 Attention 的压缩缓存，`C4Indexer` 内部的 Compressor 为索引器服务，两套状态分开保存、各有用途。[源码：MQALayer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L1115-L1173) [源码：C4Indexer](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/dsv4/indexer.py#L1125-L1240)

core Compressor 生成 C4 历史，Indexer 为当前 Q 选压缩位置。单流路径在主 Attention 之前准备这些内容，把索引结果放到 `forward_metadata.c4_topk_indices`。后端以 `cmp_sparse_indices` 限制压缩侧读取，同时仍读 SWA。[`forward_c4_indexer()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L875-L901) [源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

这里的 TopK 选历史位置，MoE Router 的 TopK 选专家。Indexer 决定本轮 Q 要读哪些压缩位置，主 Attention 消费对应表示。

### ratio = 128：SWA + C128，但没有 C4 Top-K 索引

C128 也有 core Compressor，但不创建 C4Indexer。后端同时带 SWA、C128 缓存，将 `cmp_sparse_indices=None`，省去 C4 TopK 筛选，读取范围仍受当前请求、有效长度和算子规则约束。[源码：`_forward_compressed()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2118-L2189)

| 当前层类型 | 近期历史 | 压缩历史 | 额外索引器 |
| --- | --- | --- | --- |
| ratio 0 | SWA | 无 | 无 |
| ratio 4 | SWA | C4 | C4 TopK |
| ratio 128 | SWA | C128 | 无 C4 TopK |

压缩比例描述历史表示，输出一个 Token 不保证就增加一条完整压缩记录。算容量时，SWA、压缩池、Compressor State、C4 Indexer State 和页表的布局、dtype、实际分配都要各自计入，普通多头 KV 的单一公式覆盖不了这些状态。

<a id="attention"></a>

## 六、Attention 的输出回到各自的 Token

读取、计算之后，Attention 结果经过 inverse-RoPE、`wo_a`、`wo_b` 等输出处理，回到各 Token 的 hidden states，再进入后续子层。T1、T2 的输出仍对应原来的请求行。[Attention 后的 inverse-RoPE、`wo_a`、`wo_b`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/models/deepseek_v4.py#L2360-L2585)

MoE 会接着把这些 Token 按专家选择重新分发，再将贡献合回原行，具体过程见 [MoE 的路由与专家计算](deepseek-v4-moe-source-analysis.md)。

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
