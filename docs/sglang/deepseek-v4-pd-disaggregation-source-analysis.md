# DeepSeek-V4 PD 分离：首 Token、KV 状态与 Decode 接管

PD 分离把输入处理和持续生成交给两组服务。P 端完成 Prompt 计算后，D 端需要接收足够的缓存与辅助状态，才能继续生成，而不必重算整段输入。

交接里最容易错一位的是首 Token：P 已经采样出它，但它还没有经过下一次模型计算，因此没有自己的 KV。下面从这枚 handoff Token 出发，追踪目标资源预分配、状态传输、完成确认和第一次 Decode，同时说明拆分带来的独立调度能力与传输成本。

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线讨论 DeepSeek-V4 + SGLang PD Disaggregation，并重点说明 Ascend NPU 路径。PD 的工程价值主要出现在 P/D 负载不对称明显、Prefill 会扰动 Decode SLO、流量足以分别形成 P/D resource pool，且网络能够承担状态迁移成本的场景。

## 一、拆开 Prefill 与 Decode 的动机和成本

Prefill 和 Decode 都运行同一个 Transformer，但一次 forward 看到的工作量并不一样。

假设一条请求有 8192 个 Prompt Token。Prefill 要一次处理大量新 token rows：

~~~text
Prompt = 8192 tokens

hidden_states
≈ [8192, H]
~~~

进入 Linear、Attention、MoE 后，能够形成较大的矩阵计算和较大的 token workload。

而到了普通 Decode，一条请求每轮只新增一枚 Token。即使 Continuous Batching 同时放进 32 条请求，也更接近：

~~~text
32 requests
×
1 new token

hidden_states
≈ [32, H]
~~~

与此同时，每个 Query 还需要访问已经存在的历史 Attention state。

因此更准确的工程抽象是：

| 阶段 | 主要工作形态 | Serving 更关心什么 |
| --- | --- | --- |
| Prefill | 一次处理大量新 Token，矩阵计算规模大，Prompt 长度差异明显 | TTFT、Prefill throughput |
| Decode | 大量连续小步迭代，反复读取已有上下文状态 | TPOT、Decode throughput、稳定尾延迟 |

两阶段都使用计算、显存和通信资源，资源占比与时延目标各有侧重。

固定版本的 SGLang PD 文档也把 Unified Scheduling 的问题直接归纳为两类：**Prefill interruption** 和 **DP Attention imbalance**。

源码/文档入口：

[PD Disaggregation](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/advanced_features/pd_disaggregation.mdx)

想象一台 Unified Worker 正稳定服务 Decode：`Decode → Decode → Decode → Decode → ...`

此时插入一条很长的 Prefill：`Decode → Decode → 64K Prefill → Decode → Decode`

大 Prompt 会与正在运行的 Decode 共享计算资源、内存带宽、通信和 Scheduler budget。对用户来说，最直接的表现就是 TPOT 抖动。

DeepSeek 的 DP Attention + MoE 场景还会把这种差异放大。如果不同 DP shard 同时处理完全不同的 token workload：

~~~text
DP0 → long Prefill
DP1 → Decode
DP2 → Decode
DP3 → Decode
~~~

后面的同步与 MoE execution 也要面对不平衡的工作量。

PD Disaggregation 的第一层价值于是出现了：

~~~mermaid
flowchart TD
    R[Requests] --> P[Prefill Pool]
    P -->|state handoff| D[Decode Pool]
    D --> O[Generated Tokens]
~~~

Prefill 和 Decode 从同一个统一调度域中拆出来之后，可以分别选择更适合自己的 batch、并行、graph、并发上限和机器数量。

拆分之后，下一步要完成请求状态交接：

> **Prefill 已经算过的上下文，怎样交给另一台 Decode Worker，而不重新算一遍 Prompt？**

## 二、首 Token 与待传输的请求状态

第一枚输出，也就是 handoff token，已经由 Prefill Worker 采样产生。

Prefill batch 完成后，`process_batch_result_disagg_prefill()` 会拿到：

~~~python
result.next_token_ids
~~~

并执行：

~~~python
req.output_ids.append(next_token_id)
~~~

之后才进入最终 KV / state 发送。

固定源码：

[`process_batch_result_disagg_prefill()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L723-L962)

这枚 token 随 PD metadata 一起交给 Decode。MetadataBuffers 的注释甚至直接写明：`We transfer the metadata of first output token to decode`

并在 `set_buf()` 中写入：

~~~python
self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
~~~

固定源码：

[`MetadataBuffers`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L288-L571)

### 但 handoff token 的 KV 此时还不存在

这枚 Token 正好连接 Prefill 输出与 Decode 输入。

对 fresh request，Decode 侧预分配 Prompt 状态时使用：

~~~python
return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
~~~

源码旁边明确解释：`the last output token's KV has not been written yet`

固定源码：

[`DecodePreallocQueue._pre_alloc_fill_len()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L808-L829)

因此，对正常 fresh PD handoff，可以把状态理解成：

~~~text
Prefill 已完成：

Prompt tokens:
t0 t1 t2 ... tn

Prompt KV/state:
✓  ✓  ✓      ✓

Prefill sampled:
handoff_token = h

h 的 KV:
尚未计算
~~~

交给 Decode 的是：

~~~text
Prompt 已计算的持久状态
+
handoff token ID
+
必要 auxiliary metadata
~~~

其中不包含 `h` 自己的 KV；这部分留到下一次模型计算。

Decode 收到 `h` 后，下一次 Decode forward 会把这枚 handoff token 作为当前输入，计算它自己的 KV/state，同时预测下一枚 Token。

所以 ownership 可以精确写成：

~~~text
Prefill Worker
负责：
Prompt Forward
Prompt persistent state
handoff token 的采样


Decode Worker
接手：
handoff token ID
并在第一轮 Decode 中
物化 handoff token 对应的 KV/state
~~~

Decode 复用已经算好的 Prompt 状态，从 handoff token 的 Forward 继续生成。

## 三、Bootstrap 与目标资源预分配

理解 PD 最容易犯的第二个错误，是把生命周期画成绝对串行：

~~~text
Decode 完成全部 prealloc
        ↓
Prefill 才允许开始计算
~~~

发送需要目标地址就绪。计算则有两条路径：普通 Bootstrap 按就绪状态推进，**optimistic prefill** 允许在目标资源尚未完全准备好时提前计算。

### 两边先各自建立 Sender / Receiver

Prefill 请求首先进入：`PrefillBootstrapQueue`

并创建 `KVSender`：

~~~python
req.disagg_kv_sender = kv_sender_class(
    mgr=self.kv_manager,
    bootstrap_addr=...,
    bootstrap_room=req.bootstrap_room,
    ...
)
~~~

随后：

~~~python
req.pending_bootstrap = True
~~~

源码：

[`PrefillBootstrapQueue.create_sender()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L349-L418)

Decode 侧则先创建 `KVReceiver`，把请求放进 `DecodePreallocQueue`：

~~~python
kv_receiver = kv_receiver_class(
    mgr=self.kv_manager,
    bootstrap_addr=_bootstrap_addr(req),
    bootstrap_room=req.bootstrap_room,
)
~~~

源码：

[`DecodePreallocQueue.add()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L673-L785)

之后 Decode 根据本地 allocator：

~~~text
allocate req_pool row
allocate destination KV/state slots
allocate metadata slot
~~~

再通过 `send_metadata(...)` 把目标页索引等信息发布给 Prefill。

这一步发生在 `pop_preallocated()` 内，目标地址发布完成之后，请求才交给 DecodeTransferQueue。

### Prefill 正常路径等到 WaitingForInput 再 finalize bootstrap

Prefill 会 poll sender。

当状态进入：`KVPoll.WaitingForInput`

说明 Decode 侧已经具备可用 destination metadata。此时：

~~~python
decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
...
req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
req.pending_bootstrap = False
~~~

源码：

[`finalize_bootstrap()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L386-L406)

Bootstrap 在两端各自分配的物理地址之间建立映射：

~~~text
Prefill source pages
        ↓ mapping
Decode destination pages
~~~

例如：

~~~text
Prefill                    Decode

physical page 100  ─────► physical page 501
physical page 101  ─────► physical page 803
physical page 102  ─────► physical page 804
~~~

两边 allocator 独立，所以物理 page id 本来就不需要相同。

### Optimistic Prefill 打破了“prealloc 完成才能计算”的绝对顺序

`pop_bootstrapped()` 还有另一条路径：

~~~python
elif poll == KVPoll.Bootstrapping:
    if req.prefill_attempt_count < optimistic_prefill_attempts:
        ...
        bootstrapped_reqs.append(req)
~~~

也就是说，在配置允许时，Prefill 可以在 bootstrap 尚未完成时先进入计算。

如果模型已经 Prefill 完，但 Decode 的 destination 仍未准备好，请求会停在 Prefill Inflight Queue：

~~~python
if req.pending_bootstrap:
    # Parked: prefill finished before bootstrap completed.
~~~

等 sender 最终进入 `WaitingForInput`，再 `finalize_bootstrap()` 并发送最后一段 KV。

源码：

[`process_disagg_prefill_inflight_queue()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L965-L1035)

这两条路径的差别在于计算何时开始，状态传输仍依赖目标资源就绪：

| 路径 | Prefill 何时进入计算 | 传输前要满足什么 |
| --- | --- | --- |
| 普通 Bootstrap | 按正常就绪状态进入 | Sender 完成 Bootstrap，取得目标布局 |
| Optimistic Prefill | 在允许的尝试次数内提前计算 | 若目标尚未就绪，停留 Inflight 等待完成 Bootstrap |

无论选择哪条路径，源页与目标页的对应关系都必须先建立，才能把请求状态交给 Decode。

## 四、传输完成后，Decode 怎样接管

Prefill 最终 chunk 会先把 handoff metadata 写入 MetadataBuffers，再构造需要发送的 state indices：

~~~python
if last_chunk:
    self.disagg_metadata_buffers.set_buf(req)
    ...
~~~

随后根据当前请求需要传输的页调用：

~~~python
req.disagg_kv_sender.send(
    page_indices,
    send_state_indices,
    num_kv_tokens=...
)
~~~

固定源码：

[`_send_kv_chunk()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L1302-L1495)

Prefill 请求然后停留在 Inflight Queue，持续 poll transfer 状态；只有所有参与 Rank 对 terminal state 达成一致，才释放本地 ownership。

Decode 侧则在：`DecodeTransferQueue`

轮询 Receiver。成功之后：

~~~python
self._commit_transfer_to_req(decode_req)
~~~

会恢复：

~~~text
handoff output token
cached-token accounting
logprob metadata
sampling metadata
speculative hidden/top-k state（如启用）
bootstrap identity
...
~~~

源码中对 handoff token 的 ownership 再次给出了直接注释：`The handoff token is generated on the prefill worker`

固定源码：

[`DecodeTransferQueue._commit_transfer_to_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2215-L2350)

### PREBUILT 重建请求元数据与输出状态

Transfer 成功的请求进入 Decode waiting queue 后，会构造一个 `ScheduleBatch`，然后：

~~~python
new_batch.prepare_for_prebuilt()
...
new_batch.process_prebuilt(self.future_map)
~~~

源码：

[`_get_new_prebuilt_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2822-L2900)

`prepare_for_prebuilt()` 的注释非常明确：`PREBUILT never enters a model forward.`

它做的是恢复：

~~~text
req_pool_indices
seq_lens
prefix_lens
extend_lens
out_cache_loc
sampling_info
...
~~~

但不会把整段 transferred Prompt 再 flatten 到 GPU 去跑一次模型。

固定源码：

[`ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py)

然后 `process_prebuilt()` 取：

~~~python
req.output_ids[-1]
~~~

也就是刚从 Prefill 交接过来的 handoff token。

非 speculative 模式下，它把这枚 token 放进 `FutureMap` relay：

~~~python
future_map.stash(
    self.req_pool_indices,
    RelayPayload(bonus_tokens=last_tokens_tensor),
)
~~~

下一个 Decode forward 再从 relay 重建 `input_ids`。

PREBUILT 在本地重建“Prefill 已由远端完成”的请求元数据与输出状态，随后 Decode 从 handoff token 继续。

整个 Decode 接管过程可以画成：

~~~mermaid
flowchart TD
    A[DecodePreallocQueue]
    A --> B[Destination KV/state allocated]
    B --> C[DecodeTransferQueue]
    C --> D{Transfer complete?}
    D -->|No| C
    D -->|Yes| E[Commit handoff metadata]
    E --> F[Decode Waiting Queue]
    F --> G[PREBUILT]
    G --> H[Restore seq / cache / sampling metadata]
    H --> I[Relay handoff token]
    I --> J[Merge into Running Batch]
    J --> K[First real Decode forward]
~~~

这里被跳过的是：

> **Prompt 的模型 Prefill Forward。**

没有被跳过的是：

> **handoff token 后续作为新输入所需要的第一轮 Decode。**

## 五、DeepSeek-V4 的 KV 与辅助状态

DeepSeek-V4 在 Ascend 上使用 `DSV4NPUTokenToKVPool`，分别注册 SWA、C4、C128 和 Compression / Indexer State。下面按注册入口与页、请求两种归属方式展开。

### C4 主数据走 main PD buffer

`DSV4NPUTokenToKVPool.get_contiguous_buf_infos()` 明确返回：

~~~text
C4 KV
+
C4 indexer K
+
C4 index scale
~~~

源码注释：`Main PD buffers addressed by the full KV page id.`

固定源码：

[`DSV4NPUTokenToKVPool.get_contiguous_buf_infos()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)

因此在当前 NPU DSV4 path 里，**C4 的主 KV / indexer 数据属于 main KV transfer component**。

### SWA KV 是独立 state component

`setup_state_kv_args()` 对 `BaseSWAKVPool` 注册：

~~~python
StateType.SWA
~~~

而 NPU DSV4 的 `get_state_buf_infos()` 始终先加入 SWA KV buffer。

SWA 因此通过独立 State Buffer 传输。

### C4 compression state 在 A3 和 A5 上的 wire layout 不一样

这点与 Ascend 代际直接相关。

在 **pre-A5 / explicit-location** 路径（包括当前 A3 语义），`get_state_buf_infos()` 除 SWA KV 外，还会把：

~~~text
C4 attention compression state
+
C4 indexer compression state
~~~

一起注册到 `StateType.SWA` component，因为它们共享 SWA page/state index 语义。

而在 **A5 / CYCLE cache mode**，C4 state 改成 request-local ring ownership，于是代码将它从 `StateType.SWA` 中拿出来，单独注册：

~~~python
AscendStateType.DSV4_C4_STATE
~~~

固定源码：

- [`setup_state_kv_args()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L1313-L1612)
- [`DSV4NPUTokenToKVPool.get_state_buf_infos()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)
- [`AscendStateType`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/conn.py#L24-L34)

因此，C4 State 的传输布局要区分 A3 与 A5：A3 按对应 Pool 布局传递，A5 使用独立注册的 `DSV4_C4_STATE`。

### C128 KV 和 C128 request state 也是两件事

NPU DSV4 另外把 C128 KV buffer 注册为：

~~~python
AscendStateType.DSV4_C128
~~~

来源：

~~~python
token_to_kv_pool.get_c128_kv_buf_infos()
~~~

而 request-scoped compression state 则由：

~~~python
get_request_state_buf_infos()
~~~

注册成：

~~~python
StateType.DSV4_REQUEST_STATE
~~~

后者的固定实现用于 request-scoped compressed state；当前实现文档明确包含 C128 raw-token ring / online row，并在相应配置下包含其他 request-scoped pending state。

~~~text
C128 KV
≠
C128 compressor/request state
~~~

这两类数据有不同的 buffer、index 和 transfer ownership。

### Speculative Decode 还会继续扩大 wire contract

如果 target 和 draft 都是 DeepSeek-V4 pool，`setup_state_kv_args()` 还会校验 Draft NextN 层必须是 SWA-only，并把 Draft SWA buffers 作为独立 positional component 注册。

而 Prefill metadata 还可能包含：

~~~text
topk_p
topk_index
hidden_states
DSA top-k indices
~~~

这与上一篇 speculative decoding 的状态机直接连接起来。

所以 DeepSeek-V4 PD 更准确的抽象是：`Model execution state migration`

而不是简单：`KV memcpy`

当前 Ascend DSV4 的注册对象对应如下：

| 组件 | 当前 PD wire 角色 |
| --- | --- |
| C4 KV + indexer K/scale | main KV buffers |
| SWA KV | `StateType.SWA` |
| C4 attention/indexer compression state，A3/pre-A5 | 随 `StateType.SWA` |
| C4 compression state，A5/CYCLE | `AscendStateType.DSV4_C4_STATE` |
| C128 KV | `AscendStateType.DSV4_C128` |
| request-scoped C128/compressed state | `StateType.DSV4_REQUEST_STATE` |
| DSpark/NextN Draft SWA（适用配置） | 独立 Draft SWA state component |

页布局、请求状态与辅助缓冲按这些注册项分别交接。

## 六、Ascend 传输组件怎样分工

SGLang 把 PD backend 统一抽象成：

~~~python
class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"
~~~

然后 `get_kv_class()` 按 backend 解析：

~~~text
KVManager
KVSender
KVReceiver
BootstrapServer
~~~

固定源码：

[`TransferBackend / get_kv_class()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L579-L698)

Ascend path 最容易被误解的地方，是类继承关系里出现了很多 `Mooncake` 名字：

~~~python
class AscendKVManager(MooncakeKVManager):
    ...

class AscendKVSender(MooncakeKVSender):
    pass

class AscendKVReceiver(MooncakeKVReceiver):
    pass

class AscendKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
~~~

这不意味着：

> Ascend PD 的底层数据面还是 Mooncake engine。

更准确的理解是：**Ascend 复用了 Mooncake backend 已经实现好的 PD control / sender / receiver protocol scaffolding。**

初始化数据传输 engine 的地方在：

~~~python
AscendKVManager.init_engine()
~~~

它创建：

~~~python
AscendTransferEngine(...)
~~~

而 `AscendTransferEngine` 虽然继承 `MooncakeTransferEngine` 以复用统一的 register / transfer wrapper API，但构造时把底层：

~~~python
self.engine
~~~

替换成：

~~~python
memfabric_hybrid.TransferEngine()
~~~

并用：

~~~text
ASCEND_MF_STORE_URL
role = Prefill / Decode
NPU id
transfer protocol
~~~

初始化。

固定源码：

- [`AscendKVManager`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/conn.py)
- [`AscendTransferEngine`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/transfer_engine.py#L26-L111)
- [`MooncakeTransferEngine`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py)

因此架构可以拆成：

~~~mermaid
flowchart TD
    S[Prefill/Decode Scheduler]
    S --> K[AscendKVSender / Receiver]
    K --> M[AscendKVManager]
    M --> A[AscendTransferEngine]
    A --> F[memfabric_hybrid.TransferEngine]
    F --> X[SDMA or device_rdma]
~~~

默认协议：`sdma`

也可以通过：`ASCEND_MF_TRANSFER_PROTOCOL=device_rdma`

选择 device RDMA。

当使用 `device_rdma` 时，源码会在 MemFabric 初始化前先做一次 NPU world `all_gather`，注释说明这是为了**提前初始化 HCCL，避免 HCCL 与 RDMA 初始化冲突**。

这次 `all_gather` 用于初始化协调，PD Payload 随后由传输引擎按 SDMA 或 device RDMA 协议发送。

另外，固定版本的 Ascend support matrix 还明确要求 NPU PD 手工指定：

~~~bash
--disaggregation-transfer-backend ascend
~~~

因为全局默认 backend 仍是 `mooncake`，而文档标明该默认值在 NPU 上不支持。

固定入口：

[Ascend Support Features](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/reference/support_features.mdx)

## 七、PD 的独立调度能力与适用负载

PD 将两类计算放到独立资源池，让它们各自选择调度与执行策略。

PD 改变的是：

> **Prefill 和 Decode 不再必须共享同一个资源池、同一个 batch 形态和同一套性能参数。**

固定版本的 DeepSeek-V4-Flash Ascend Best Practice 就展示了这种差异。

官方 1P1D 配置中，Prefill 一侧强调：

~~~text
small prefill concurrency
large prefill token budget
prefill-oriented scheduling
example disables graph
~~~

Decode 一侧则可以设置：

~~~text
much larger max_running_requests
decode graph buckets
speculative decoding
decode-oriented DeepEP tuning
~~~

两边同时使用：`--disaggregation-transfer-backend ascend`

固定入口：

[DeepSeek-V4-Flash Ascend Best Practice](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash.mdx)

这些 Benchmark 配置展示了独立设计 P、D 的方式：根据各自负载选择并发、分块、Graph 与并行策略，再在目标 SLO 下测量。

最终的生产权衡更接近：

~~~text
PD 收益
────────────────────
Prefill / Decode 独立扩缩
不同 batch policy
不同 graph policy
不同并行与并发策略
资源隔离
降低长 Prefill 对 Decode TPOT 的干扰


PD 成本
────────────────────
state transfer latency
network bandwidth
Decode destination preallocation
bootstrap / metadata coordination
failure / timeout handling
跨 Rank 一致性
更多 observability 与运维复杂度
~~~

是否采用 PD，可以从负载与传输代价一起评估。请求规模较小、P/D 干扰不明显时，统一服务更简单；长输入持续挤占 Decode 资源，且 TTFT、TPOT 有不同目标时，独立资源池才可能带来足够收益。

例如，假设一次交接需要传 2 GiB 状态，有效链路带宽为 40 GiB/s，仅数据传输的理想下限就是 50 ms。这里的容量与带宽均为教学设定；排队、协议、预分配和计算依赖还会增加交接耗时。部署时应使用实际传输字节数与有效带宽代入，再与隔离 Prefill 干扰所节省的时间比较。

将资源准备与计算合在一起，一次请求的交接流程如下：

~~~mermaid
flowchart TD
    R[Request] --> P1[Prefill Sender / Bootstrap]
    R --> D1[Decode Receiver / Prealloc]

    P1 --> P2[Prefill Forward]
    D1 --> D2[Publish destination layout]

    P2 --> T[Generate handoff token]
    D2 --> X[Transfer can start]
    T --> X

    X --> S[Prompt KV + model state + handoff metadata]
    S --> C[Decode transfer commit]

    C --> B[PREBUILT metadata reconstruction]
    B --> H[Relay handoff token]
    H --> D[First real Decode forward]
~~~

交接完成后，Decode 接管已物化的 Prompt 状态，并用 handoff token 继续生成。DeepSeek-V4 的交接内容还包括对应路径下的 SWA、压缩历史和辅助状态。资源隔离能否改善服务质量，需要把这些状态的迁移成本一起计入。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| PD 官方说明 | [`pd_disaggregation.mdx`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/advanced_features/pd_disaggregation.mdx) |
| P/D Mode 与 Transfer Backend | [`disaggregation/utils.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py) |
| Prefill Bootstrap / Sender | [`PrefillBootstrapQueue`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L145-L545) |
| Prefill 结果与 handoff token | [`process_batch_result_disagg_prefill()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L723-L962) |
| Prefill KV/state send | [`_send_kv_chunk()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L1302-L1495) |
| Decode Receiver / Preallocation | [`DecodePreallocQueue`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py) |
| Transfer commit | [`_commit_transfer_to_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2215-L2350) |
| PREBUILT metadata reconstruction | [`decode_schedule_batch_mixin.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py) |
| Decode PD Scheduler | [`SchedulerDisaggregationDecodeMixin`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2625-L2900) |
| DeepSeek-V4 generic KV/state registration | [`setup_state_kv_args()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L1313-L1612) |
| Ascend DSV4 Pool | [`DSV4NPUTokenToKVPool`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py) |
| Ascend DSV4 state payload | [`dsv4_common_hooks.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_common_hooks.py) |
| Ascend KV protocol classes | [`ascend/conn.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/conn.py) |
| Ascend MemFabric engine | [`AscendTransferEngine`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/transfer_engine.py) |
| DeepSeek-V4 Ascend Tutorial | [`deepseek_v4_flash.mdx`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx) |
| DeepSeek-V4 Ascend Best Practice | [`best-practices/deepseek_v4_flash.mdx`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash.mdx) |
