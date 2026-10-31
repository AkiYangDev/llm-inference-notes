# DeepSeek-V4 PD 分离：首 Token、KV 状态与 Decode 接管

PD 分离让 P 处理输入、D 持续生成。P 算完 Prompt 后，还要将后续执行所需的缓存和辅助状态交给 D，才能省去输入重算。

首 Token 是这次交接中容易错一位的地方：P 已经采样出 ID，却还没把它当作输入计算，因此没有它自己的 KV。D 取得 Prompt 状态后，从这枚 handoff Token 继续。下面沿着它看预分配、状态迁移和第一轮 Decode，分别确定数据到达、调度接管及模型执行的时刻。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线为 DeepSeek-V4 的 SGLang PD Disaggregation，重点展开 Ascend NPU。是否值得拆分，取决于 P/D 负载差异、Prefill 对 Decode SLO 的干扰、分别形成资源池的流量，以及状态传输成本。

</details>

## 一、拆开 Prefill 与 Decode 的动机和成本

两阶段使用同一个 Transformer，一次 forward 的工作量却不同。

例如一条 8192 Token 的 Prompt，会让 Prefill 处理大量新行：

~~~text
Prompt = 8192 tokens

hidden_states
≈ [8192, H]
~~~

这些行共同进入 Linear、Attention、MoE，比单请求普通 Decode 更容易形成大矩阵。8192 只是阶段差异示例，实际行数还受缓存、分块和并行布局影响。

普通 Decode 每请求一轮只新增一枚 Token，32 条请求一起执行时，形状更接近：

~~~text
32 requests
×
1 new token

hidden_states
≈ [32, H]
~~~

每个 Query 同时还需访问已存在的历史状态。

因此两阶段有不同的工作形态和服务目标：

| 阶段 | 主要工作形态 | Serving 更关心什么 |
| --- | --- | --- |
| Prefill | 一次处理大量新 Token，矩阵计算规模大，Prompt 长度差异明显 | TTFT、Prefill throughput |
| Decode | 大量连续小步迭代，反复读取已有上下文状态 | TPOT、Decode throughput、稳定尾延迟 |

计算、内存、通信两边都会用到，但占比与时延要求不同。

固定 PD 文档将统一调度中的问题概括为 Prefill interruption 和 DP Attention imbalance，见 [PD Disaggregation](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/advanced_features/pd_disaggregation.mdx)。

假设 Unified Worker 正在连续服务 Decode：`Decode → Decode → Decode → Decode → ...`

此时插入长输入：`Decode → Decode → 64K Prefill → Decode → Decode`

长 Prompt 与 Decode 竞争计算、带宽、通信及调度预算，可能表现为持续输出间隔变长、TPOT 抖动。

DPA + MoE 还会遇到各 shard 工作量不同的情形：

~~~text
DP0 → long Prefill
DP1 → Decode
DP2 → Decode
DP3 → Decode
~~~

后续同步和 MoE 执行也要处理这组不平衡。

拆分后，两种工作可以进入各自资源池：

~~~mermaid
flowchart TD
    R[Requests] --> P[Prefill Pool]
    P -->|state handoff| D[Decode Pool]
    D --> O[Generated Tokens]
~~~

P 可围绕长输入组织大批次，D 围绕输出延迟安排并发和 Graph，两端分别选择并行度与规模。收益是否兑现，还得扣除状态交接、目标端等待和传输带来的成本。

因此拆开之后，首先要解决如何把已经算过的上下文交给 D。

## 二、首 Token 与待传输的请求状态

首枚输出已经由 P 采样，这枚 ID 就是 handoff token。Prefill 完成后，`process_batch_result_disagg_prefill()` 从 `result.next_token_ids` 取得结果，用 `req.output_ids.append(next_token_id)` 追加到请求，再进入最终 KV/state 发送。[`process_batch_result_disagg_prefill()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L723-L962)

这枚 ID 还要经 PD metadata 交给 D。`MetadataBuffers.set_buf()` 把 `req.output_ids[0]` 写入 `self.output_ids[req.metadata_buffer_index][0]`，让接收端拿到首枚输出。[`MetadataBuffers`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L288-L571)

### 但 handoff token 的 KV 此时还不存在

这枚新 Token 在 P 是输出，到 D 则成为下一次模型输入。

fresh request 在 D 预分配 Prompt 状态时，长度采用：

~~~python
return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
~~~

源码注释 `the last output token's KV has not been written yet` 说明减一的原因。[`DecodePreallocQueue._pre_alloc_fill_len()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L808-L829)

设 Prompt 有 N 个 Token，P 已算好位置 `0...N-1`，并采样出 h：

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

实际交接包括 Prompt 已计算的持久状态、handoff Token ID，以及必要的辅助 metadata。

图中最后下标 n 对应 `N=n+1`。h 将位于 N，但目前只有 ID，因此待传状态还不包含 h 的 KV。

D 的第一轮真实 Decode 用 h 为输入，读取 `0...N-1` 历史，计算并写入位置 N，再预测下一枚。这里限定正常 fresh handoff，rebootstrap 恢复分支的长度规则需要另看。

P 到此完成 Prompt Forward、保存持久状态并采样 h；D 接过 h 的 ID，在自己的第一轮 Decode 中形成 h 对应的 KV/state，无需重算 Prompt。

## 三、Bootstrap 与目标资源预分配

发送需要先知道 D 的目标位置，但这个要求针对传输，不一定要求 P 也等到此时才计算。不能将全部配置都画成下面的严格串行关系：

~~~text
Decode 完成全部 prealloc
        ↓
Prefill 才允许开始计算
~~~

Bootstrap 交换请求身份和目标位置等信息。普通路径按就绪状态推进；允许 optimistic prefill 时，P 可以先计算，再等目标信息满足发送条件。

### 两边先各自建立 Sender / Receiver

P 的请求先进入 `PrefillBootstrapQueue` 并创建 `KVSender`：[`PrefillBootstrapQueue.create_sender()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L349-L418)

~~~python
req.disagg_kv_sender = kv_sender_class(
    mgr=self.kv_manager,
    bootstrap_addr=...,
    bootstrap_room=req.bootstrap_room,
    # 其余参数省略
)
~~~

标记 Bootstrap 仍待完成：

~~~python
req.pending_bootstrap = True
~~~

D 创建 `KVReceiver` 后将请求放入 `DecodePreallocQueue`。[`DecodePreallocQueue.add()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L673-L785)

~~~python
kv_receiver = kv_receiver_class(
    mgr=self.kv_manager,
    bootstrap_addr=_bootstrap_addr(req),
    bootstrap_room=req.bootstrap_room,
)
~~~

再由本地 allocator 分配：

~~~text
allocate req_pool row
allocate destination KV/state slots
allocate metadata slot
~~~

D 随后用 `send_metadata(...)` 发布目标页索引等信息。

这些工作在 `pop_preallocated()` 内完成，可理解为先准备接收位置表并告知 P，再进入 DecodeTransferQueue 等待数据。目标空间已分配，尚不代表其中状态有效。

### Prefill 正常路径等到 WaitingForInput 再 finalize bootstrap

P 持续轮询 Sender，当返回 `KVPoll.WaitingForInput`，说明已取得可用目标 metadata。

还需确认本地 metadata buffer 有槽位，`finalize_bootstrap()` 成功后才初始化 Sender、清除 `pending_bootstrap`：[`finalize_bootstrap()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L386-L406)

~~~python
decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
...
req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
req.pending_bootstrap = False
~~~

两端独立分配的物理页通过下面的关系对应：

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

这些页号只是示意。源 100 对应目标 501 可以完全正确，必须相同的是请求及其逻辑状态区间。比较两边 page ID 是否相等不能验证交接。

### Optimistic Prefill 打破了“prealloc 完成才能计算”的绝对顺序

`pop_bootstrapped()` 还允许下面的提前执行分支：

~~~python
elif poll == KVPoll.Bootstrapping:
    if req.prefill_attempt_count < optimistic_prefill_attempts:
        ...
        bootstrapped_reqs.append(req)
~~~

在配置允许的尝试范围内，P 可以尚未完成 Bootstrap 就先算输入。

如果已算完而 D 仍无目标，请求停在 Prefill Inflight Queue：

~~~python
if req.pending_bootstrap:
    # Parked: prefill finished before bootstrap completed.
~~~

等 Sender 进入 `WaitingForInput`，再完成 Bootstrap、发送最后一段 KV。[`process_disagg_prefill_inflight_queue()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L965-L1035)

提前的是计算。P 仍需保存源状态并等目标可用，不会跳过资源准备直接交接。两种时序可对照为：

| 路径 | Prefill 何时进入计算 | 传输前要满足什么 |
| --- | --- | --- |
| 普通 Bootstrap | 按正常就绪状态进入 | Sender 完成 Bootstrap，取得目标布局 |
| Optimistic Prefill | 在允许的尝试次数内提前计算 | 若目标尚未就绪，停留 Inflight 等待完成 Bootstrap |

无论走哪条路径，实际发送都要求源页、目标页先建立对应。

## 四、传输完成后，Decode 怎样接管

Prefill 的最后一个 chunk 将 handoff metadata 放入 MetadataBuffers，再准备 state indices。相关发送入口见 [`_send_kv_chunk()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L1302-L1495)。

~~~python
if last_chunk:
    self.disagg_metadata_buffers.set_buf(req)
    ...
~~~

按本请求待传页调用：

~~~python
req.disagg_kv_sender.send(
    page_indices,
    send_state_indices,
    num_kv_tokens=...
)
~~~

发送之后，P 继续在 Inflight Queue 轮询。调用返回与 D 完成接管之间还有终态确认，相关 Rank 对成功取得一致认识后才释放本地缓存持有，失败则按对应路径清理。

D 的 `DecodeTransferQueue` 确认 Receiver 成功后，调用 `self._commit_transfer_to_req(decode_req)`，将收到的 metadata 写回本地 `Req`。其中包括 handoff output Token、缓存命中记账、logprob、采样参数、bootstrap 身份，以及启用投机时的 hidden/top-k 状态。该函数的注释也说明 handoff Token 由 P 生成。[`DecodeTransferQueue._commit_transfer_to_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2215-L2350)

### PREBUILT 重建请求元数据与输出状态

请求经传输提交进入 D 的 waiting queue，再组成 `ScheduleBatch` 进行 PREBUILT。入口见 [`_get_new_prebuilt_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2822-L2900)。

~~~python
new_batch.prepare_for_prebuilt()
...
new_batch.process_prebuilt(self.future_map)
~~~

`prepare_for_prebuilt()` 在调度侧恢复 `req_pool_indices`、`seq_lens`、`prefix_lens`、`extend_lens`、`out_cache_loc`、`sampling_info` 等状态。它明确注明 `PREBUILT never enters a model forward.`，不会将收到的整个 Prompt 再展开成模型输入重算。[`ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py)

`process_prebuilt()` 随后读取 `req.output_ids[-1]`，即 P 生成并交接过来的最后一枚输出。

普通非投机模式将它写入 `FutureMap` relay：

~~~python
future_map.stash(
    self.req_pool_indices,
    RelayPayload(bonus_tokens=last_tokens_tensor),
)
~~~

下次真实 Decode 再从 relay 取得 `input_ids`。

所以 PREBUILT 只恢复本地请求、衔接已完成的 Prompt 状态，并安排 handoff Token 的后续输入。它不调用模型，也不额外生成 Token。

接管时序如下：

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

日志里应区分 PREBUILT 与真实 Forward。D 已省去 Prompt Prefill，仍需用 h 执行第一轮 Decode；把 PREBUILT 当成 h 已算完，会使输入和缓存进度错一位。

## 五、DeepSeek-V4 的 KV 与辅助状态

Ascend 的 `DSV4NPUTokenToKVPool` 同时提供 Attention 历史和继续压缩、索引所需的辅助状态。它们分别按页或按请求寻址，传输也需按注册项逐一匹配。

### C4 主数据走 main PD buffer

`DSV4NPUTokenToKVPool.get_contiguous_buf_infos()` 提供 C4 KV、C4 indexer K 和 C4 index scale，按完整 KV page ID 寻址。这些主数据经 main KV transfer component 交接。[`DSV4NPUTokenToKVPool.get_contiguous_buf_infos()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)

### SWA KV 是独立 state component

`setup_state_kv_args()` 为 `BaseSWAKVPool` 注册 `StateType.SWA`。NPU DSV4 的 `get_state_buf_infos()` 总是先加入 SWA KV buffer，由这份独立 State Buffer 传输。

<a id="c4-compression-state-a3-a5-wire-layout"></a>

### C4 压缩状态在 A3 与 A5 上怎样传输

pre-A5 / explicit-location 路径包括本文的 A3 语义：在 SWA KV 之后，还加入 C4 Attention 与 C4 Indexer 两份 compression state。它们共享 SWA page/state index 语义，一起注册到 `StateType.SWA`。

A5 / CYCLE cache 改用 request-local ring ownership，将 C4 state 从 SWA component 分出，单独注册为 `AscendStateType.DSV4_C4_STATE`。因此核对收发项时，要先确认当前使用的是哪种布局。相关注册和类型入口为 [`setup_state_kv_args()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L1313-L1612)、[`DSV4NPUTokenToKVPool.get_state_buf_infos()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)、[`AscendStateType`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/conn.py#L24-L34)。

### C128 KV 和 C128 request state 也是两件事

C128 KV 来自 `token_to_kv_pool.get_c128_kv_buf_infos()`，单独注册为 `AscendStateType.DSV4_C128`，保存已形成的压缩历史。继续执行压缩还需要 request-scoped state，由 `get_request_state_buf_infos()` 提供，注册为 `StateType.DSV4_REQUEST_STATE`。

固定实现文档中的 request state 包含 C128 raw-token ring / online row，也可按配置包含其他 request-scoped pending state。它与 C128 KV 各有 Buffer、索引，传到已形成的压缩历史，并不保证后续计算所需的中间状态也已到齐。

<a id="speculative-decode-wire-contract"></a>

### 投机生成还需交接 Draft 状态

若 Target、Draft 都使用 DeepSeek-V4 pool，`setup_state_kv_args()` 还检查 Draft NextN 层必须 SWA-only，并将 Draft SWA buffers 注册为独立 positional component。Prefill metadata 也可能携带 `topk_p`、`topk_index`、`hidden_states` 和 DSA top-k indices，供后续投机生成继续使用。

因此不能仅凭一块 KV Buffer 复制成功就判断所有后续输入齐备。当前 Ascend DSV4 的注册关系如下：

| 组件 | 当前 PD wire 角色 |
| --- | --- |
| C4 KV + indexer K/scale | main KV buffers |
| SWA KV | `StateType.SWA` |
| C4 attention/indexer compression state，A3/pre-A5 | 随 `StateType.SWA` |
| C4 compression state，A5/CYCLE | `AscendStateType.DSV4_C4_STATE` |
| C128 KV | `AscendStateType.DSV4_C128` |
| request-scoped C128/compressed state | `StateType.DSV4_REQUEST_STATE` |
| DSpark/NextN Draft SWA（适用配置） | 独立 Draft SWA state component |

页状态、请求状态及辅助缓冲均需与对应注册项一起接续。

## 六、Ascend 传输组件怎样分工

SGLang 将 PD backend 定义为：

~~~python
class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"
~~~

`get_kv_class()` 按所选 backend 解析 `KVManager`、`KVSender`、`KVReceiver` 和 `BootstrapServer`，分别承接管理、收发和 Bootstrap。[`TransferBackend / get_kv_class()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L579-L698)

Ascend 复用了一些名带 Mooncake 的类，具体传输还需看重写方法：

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

继承提供的是控制和 Sender/Receiver 协议框架，实际数据引擎由 Manager 决定，不能仅凭父类名判断。

`AscendKVManager.init_engine()` 创建 `AscendTransferEngine(...)`。后者沿用 `MooncakeTransferEngine` 的 register / transfer wrapper API，但构造时将 `self.engine` 设为 `memfabric_hybrid.TransferEngine()`。初始化还传入 `ASCEND_MF_STORE_URL`、Prefill / Decode 角色、NPU ID 和传输协议，实际数据移动由这一层承接。相关入口：

- [`AscendKVManager`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/conn.py)
- [`AscendTransferEngine`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/transfer_engine.py#L26-L111)
- [`MooncakeTransferEngine`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py)

因此从调度到实际数据传输，可沿以下组件追踪：

~~~mermaid
flowchart TD
    S[Prefill/Decode Scheduler]
    S --> K[AscendKVSender / Receiver]
    K --> M[AscendKVManager]
    M --> A[AscendTransferEngine]
    A --> F[memfabric_hybrid.TransferEngine]
    F --> X[SDMA or device_rdma]
~~~

默认协议是 `sdma`。设置 `ASCEND_MF_TRANSFER_PROTOCOL=device_rdma` 后，还要检查已安装 MemFabric 支持的协议类型，不支持时固定实现会警告并回退 SDMA，所以最终使用值需看初始化日志。

选用 `device_rdma` 时，代码在 MemFabric 初始化前先做 NPU world `all_gather`，用于提前初始化 HCCL，避免与 RDMA 初始化冲突。

这次 collective 负责初始化协调，后续 PD Payload 仍由传输引擎按实际选定协议发送。

固定 Ascend support matrix 要求 NPU PD 显式指定 `--disaggregation-transfer-backend ascend`。全局默认仍为 `mooncake`，文档将这个默认项标为 NPU 不支持，因此配置时需要补上该参数。[Ascend Support Features](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/reference/support_features.mdx)

## 七、PD 的独立调度能力与适用负载

交接完成后，同一请求在两个资源池继续。P、D 可以分别选择 Batch、Graph、并行度和并发限制，但独立部署本身不能保证所有请求更快。

固定 DeepSeek-V4-Flash Ascend Best Practice 给出了两端分别设计的例子。

官方 1P1D 的 P 配置偏向：

~~~text
small prefill concurrency
large prefill token budget
prefill-oriented scheduling
example disables graph
~~~

D 则可以采用：

~~~text
much larger max_running_requests
decode graph buckets
speculative decoding
decode-oriented DeepEP tuning
~~~

两端均指定 `--disaggregation-transfer-backend ascend`。

固定入口：

[DeepSeek-V4-Flash Ascend Best Practice](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash.mdx)

这些配置用于说明设计差异，单个参数不代表通用最优。复用前需对齐输入输出、请求压力、缓存和网络，再看 TTFT、TPOT、吞吐是否达标；本文没有执行这些 Benchmark。

生产中还需同时衡量下面的收益与成本：

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

流量小或 P/D 干扰有限时，统一服务更简单。长输入持续干扰 Decode，且两端时延目标不同时，才更有机会用隔离收益覆盖迁移成本。

假设交接 2 GiB、有效带宽 40 GiB/s，只算数据移动的理想下限为 `2 / 40 = 0.05 s`，即 50 ms。这是教学设定。请求还受排队、预分配、Bootstrap、计算依赖影响，部分阶段可重叠，不能简单累加为实测时延。需用真实字节数、有效带宽估算，再看它们是否处在请求关键路径上。

把两端资源准备和计算放在一起：

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

D 最终接收已物化的 Prompt 状态，以 handoff Token 开始自己的计算。DeepSeek-V4 还需按路径迁移 SWA、压缩历史和辅助状态；评估资源隔离时，这些交接工作都属于成本。

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
