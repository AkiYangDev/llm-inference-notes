# DeepSeek-V4 PD 分离：首 Token、KV 状态与 Decode 接管

PD 分离把输入处理和持续生成交给两组服务。P 端完成 Prompt 计算后，D 端需要接收足够的缓存与辅助状态，才能继续生成，而不必重算整段输入。

交接里最容易错一位的是首 Token：P 已经知道要输出哪个 Token ID，但这枚新 Token 还没经过模型，所以没有自己的 KV。D 接收的是已算好的 Prompt 状态，要从这枚 handoff Token 继续向前。沿着它追踪资源预分配、状态传输和第一次 Decode，就能分清“传输完成”“请求已接管”和“新一轮模型计算”各自发生了什么。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线讨论 DeepSeek-V4 + SGLang PD Disaggregation，并重点说明 Ascend NPU 路径。PD 的工程价值主要出现在 P/D 负载不对称明显、Prefill 会扰动 Decode SLO、流量足以分别形成 P/D resource pool，且网络能够承担状态迁移成本的场景。

</details>

## 一、拆开 Prefill 与 Decode 的动机和成本

Prefill 和 Decode 都运行同一个 Transformer，但一次 forward 看到的工作量并不一样。

假设一条请求有 8192 个 Prompt Token。Prefill 要一次处理大量新 token rows：

~~~text
Prompt = 8192 tokens

hidden_states
≈ [8192, H]
~~~

这些新 Token 行一起进入 Linear、Attention、MoE，通常比单条 Decode 请求形成更大的矩阵工作量。这里的 8192 行是说明阶段差异的例子；实际还会受 Chunked Prefill、缓存命中与分片方式影响。

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

分成两个资源池后，P 可以按长输入组织较大的计算批次，D 则按持续输出的时延目标安排并发和 Graph。两边也能分别选择并行度与机器数量。不过这些调度选择要真正形成收益，还取决于交接的状态量、目标端等待和传输时间。

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

这枚 token 随 PD metadata 一起交给 Decode。MetadataBuffers 的注释写明：`We transfer the metadata of first output token to decode`

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

例如一条新请求有 N 个 Prompt Token，P 已完成位置 `0...N-1` 的状态计算，并采样出首个输出 h。此时交接的状态边界可以画成：

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

这里 N 表示 Prompt Token 数，与图中最后一个下标 n 的关系是 `N=n+1`。h 将进入位置 N，但采样只产生了 ID，没有计算位置 N 的 KV。因此交接时不能把 h 的缓存也当作已完成状态传给 D。

D 收到 h 后，第一轮真正的 Decode Forward 以 h 为输入：读取位置 `0...N-1` 的已有历史，计算并写入位置 N 的新状态，再预测下一枚 Token。这里说的是正常 fresh handoff；源码的 rebootstrap 分支有单独长度规则，不能把上面的减一公式直接套到所有恢复场景。

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

发送端需要知道数据要写到 D 的哪些位置，才能传输。这个依赖限制的是发送时机，不必然限制 P 的计算时机。下面这种把两者绑定的绝对串行关系并不适用于所有配置：

~~~text
Decode 完成全部 prealloc
        ↓
Prefill 才允许开始计算
~~~

Bootstrap 在这里负责让两端交换交接所需的信息，包括请求标识和目标位置。普通路径按就绪状态推进；配置允许 optimistic prefill 时，P 可以在 D 尚未完全准备好前先算 Prompt，再等待发送条件满足。

### 两边先各自建立 Sender / Receiver

Prefill 请求首先进入：`PrefillBootstrapQueue`

并创建 `KVSender`：

~~~python
req.disagg_kv_sender = kv_sender_class(
    mgr=self.kv_manager,
    bootstrap_addr=...,
    bootstrap_room=req.bootstrap_room,
    # 其余参数省略
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

这一步发生在 `pop_preallocated()` 内。可以把它理解为 D 先准备一份“接收位置表”，再把这份表交给 P；地址发布完成后，请求转入 DecodeTransferQueue 等待数据到达。此时有了目标空间，还不代表里面已经写入有效状态。

### Prefill 正常路径等到 WaitingForInput 再 finalize bootstrap

Prefill 会 poll sender。

当状态进入：`KVPoll.WaitingForInput`

说明发送端已经取得可用的目标元数据。P 还要确保本地 metadata buffer 槽位可用，`finalize_bootstrap()` 成功后才会完成 Sender 初始化并清除 `pending_bootstrap`。核心操作包括：

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

上面的页编号是示意值。两端 allocator 独立，源页 100 对应目标页 501 完全可以正确；需要保持的是它们代表同一请求的同一段逻辑状态。只比较两边 page id 是否相等，无法验证交接正确。

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

Optimistic Prefill 提前的是计算，不是绕过目标资源的准备。若 P 先算完，它还要持有源状态并等在 Inflight Queue，不能在 D 尚无可用目标位置时直接完成交接。两条路径的区别是：

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

Prefill 请求随后停留在 Inflight Queue，持续轮询传输状态。发送调用返回不等于 D 已经接管；相关 Rank 还要对终态取得一致认识，成功路径才释放本地缓存的持有关系，失败则走对应清理流程。

Decode 侧由 `DecodeTransferQueue` 轮询 Receiver，确认成功后才执行：

~~~python
self._commit_transfer_to_req(decode_req)
~~~

这一步把接收缓冲区中的元数据写回 Decode 端的请求对象，包括：

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

PREBUILT 因而是一段调度侧接续：让本地请求对象认识到 Prompt 已由远端算完，并把 handoff Token 放入后续输入的 relay。它不执行模型，也不生成另一枚 Token；下一次真正的 Decode Forward 才接着计算。

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

因此观察日志时，要把 PREBUILT 与模型 Forward 分开。D 省掉了整段 Prompt 的重复 Prefill，但仍需用 h 做第一轮 Decode；若把 PREBUILT 记作这轮计算已经完成，就会把输入 Token 和 KV 长度错开一位。

## 五、DeepSeek-V4 的 KV 与辅助状态

DeepSeek-V4 在 Ascend 上使用 `DSV4NPUTokenToKVPool`。交接内容既有供 Attention 读取的历史数据，也有供后续压缩与索引计算继续使用的辅助状态。下面按注册入口展开：有的以页为索引，有的以请求为索引，传输时要各自匹配。

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

C128 KV 保存已经形成的压缩历史，request state 则维护后续压缩计算要继续使用的状态。传完前者，不代表后者也已到齐；两者使用不同的 Buffer 与索引，需要按各自注册项交接。

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

因此检查 DeepSeek-V4 的 PD 传输时，需要沿着模型继续执行所需的状态逐项对应。仅确认某块 KV Buffer 已复制，不足以说明 D 已具备后续计算的全部输入。当前 Ascend DSV4 的注册对象对应如下：

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

Ascend 路径继承了多个带有 `Mooncake` 名字的类。阅读时要继续追踪它覆盖了哪些方法，不能仅凭类名判断传输实现：

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

这里继承的是 Mooncake backend 的控制与 Sender/Receiver 协议框架。要知道请求的数据实际由谁传输，还得继续看 Manager 创建了哪个 Engine，不能用父类名字代替执行路径。

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

默认协议为 `sdma`，设置 `ASCEND_MF_TRANSFER_PROTOCOL=device_rdma` 可以请求使用 device RDMA。固定源码还会检查已安装 MemFabric 提供的协议类型；不支持所选类型时记录警告并回退到 SDMA。因此配置了变量之后，仍要核对实际初始化日志与运行环境支持。

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

前面的状态交接，使 P、D 能在不同资源池继续同一条请求。独立资源池的用途，是让两种工作负载分别选择 Batch、Graph、并行度和并发上限；它并不自动保证所有请求都更快。

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

这些官方 Benchmark 配置用于说明两端如何分别设计，不能把其中的单个参数当作通用最优值。复用配置时，还要对齐输入输出长度、请求压力、缓存与网络条件，再看 TTFT、TPOT 和吞吐是否满足目标 SLO。本文没有执行这些 Benchmark。

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

例如，假设一次交接需要传 2 GiB 状态，有效链路带宽为 40 GiB/s，仅按 `2 / 40 = 0.05 s` 计算，数据传输的理想下限就是 50 ms。这里的容量与带宽均为教学设定。端到端交接还要考虑排队、预分配、Bootstrap 和计算依赖，其中有些阶段可以重叠；不能把所有单段耗时直接相加当成实测延迟。应使用实际状态字节数与有效带宽估算，再对照时间线确认这些工作是否落在请求关键路径上。

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
