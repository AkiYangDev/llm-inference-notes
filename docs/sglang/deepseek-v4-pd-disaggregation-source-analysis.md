# DeepSeek-V4 PD 分离：首 Token、KV 状态与 Decode 接管

P 算完 Prompt，已经采样出首枚输出 h；D 接手后，第一轮真正计算的输入就是 h。此时 h 的 ID 已经存在，它自己的 KV 却还没有生成，这正是 PD 交接容易错开一个位置的地方。

要让 D 从这里继续，P 得交出 Prompt 已有的缓存、辅助状态和 h 的 ID。接收端还要预分配位置、确认传输、恢复调度状态，数据到达和开始模型计算是前后几件事。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。主线为 DeepSeek-V4 的 SGLang PD Disaggregation，重点展开 Ascend NPU。是否值得拆分，取决于 P/D 负载差异、Prefill 对 Decode SLO 的干扰、分别形成资源池的流量，以及状态传输成本。

</details>

## 一、拆开 Prefill 与 Decode 的动机和成本

Prefill 和 Decode 使用同一份 Transformer，一次 forward 的输入行数却差很多。

以 8192 Token Prompt 为例，Prefill 处理的新增行大致如下。

~~~text
Prompt = 8192 tokens

hidden_states
≈ [8192, H]
~~~

这些行一起进入 Linear、Attention、MoE，比单请求普通 Decode 更容易组成大矩阵。8192 只演示阶段差别，实际新增行数还随缓存、分块与并行布局变化。

普通 Decode 每条请求一轮加一枚 Token，32 条一起执行时，输入更接近下面的形状。

~~~text
32 requests
×
1 new token

hidden_states
≈ [32, H]
~~~

每个 Query 仍要读取此前积累的历史。

两种工作形态对应的服务目标也不同。

| 阶段 | 主要工作形态 | Serving 更关心什么 |
| --- | --- | --- |
| Prefill | 一次处理大量新 Token，矩阵计算规模大，Prompt 长度差异明显 | TTFT、Prefill throughput |
| Decode | 大量连续小步迭代，反复读取已有上下文状态 | TPOT、Decode throughput、稳定尾延迟 |

两边都用计算、内存和通信，占比与时延要求却各不相同。

固定版本的 PD 文档把统一调度中的问题概括为 Prefill interruption、DP Attention imbalance。[PD Disaggregation](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/advanced_features/pd_disaggregation.mdx)

假设 Unified Worker 正连续做 Decode，节奏是 `Decode → Decode → Decode → Decode → ...`。

长输入插入后，变成 `Decode → Decode → 64K Prefill → Decode → Decode`。

长 Prompt 竞争计算、带宽、通信和调度预算，持续输出可能被拉开，表现为 TPOT 抖动。

DPA + MoE 还会遇到 shard 之间工作量不一致。

~~~text
DP0 → long Prefill
DP1 → Decode
DP2 → Decode
DP3 → Decode
~~~

后面的同步和 MoE 执行也要接着处理这种不平衡。

拆开后，两种工作进入各自资源池。

~~~mermaid
flowchart TD
    R[Requests] --> P[Prefill Pool]
    P -->|state handoff| D[Decode Pool]
    D --> O[Generated Tokens]
~~~

P 围绕长输入组织大批次，D 围绕输出延迟选择并发与 Graph，两端分别定并行度和规模。代价是状态交接、目标端等待与传输，隔离收益需要覆盖这些开销。

### PD 两端分别配置，收益还要扣掉交接

固定 DeepSeek-V4-Flash Ascend Best Practice 中，两端就采用了不同设计。

官方 1P1D 的 P 端偏向下面这些安排。

~~~text
small prefill concurrency
large prefill token budget
prefill-oriented scheduling
example disables graph
~~~

D 端采用的安排则不同。

~~~text
much larger max_running_requests
decode graph buckets
speculative decoding
decode-oriented DeepEP tuning
~~~

两端都指定 `--disaggregation-transfer-backend ascend`。

这些安排来自固定版本的 [DeepSeek-V4-Flash Ascend Best Practice](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash.mdx)。

这些例子解释设计差异，没有给出通用最优参数，本文也没有执行相应 Benchmark。复用时仍要对齐输入输出、请求压力、缓存和网络，看 TTFT、TPOT、吞吐能否达标。

独立调度带来的收益与额外成本，一起列在下面。

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

流量少、P/D 干扰小，统一服务更简单。长输入持续干扰 Decode、两端时延目标又不同，隔离才更有机会覆盖迁移开销。

例如交接 2 GiB、有效带宽 40 GiB/s，数据移动的理想下限是 `2 / 40 = 0.05 s`，即 50 ms。这只是教学算例。排队、预分配、Bootstrap 和计算依赖还会参与请求时延，部分阶段又能重叠，不能直接累加成实测值。实际评估要用真实字节数与有效带宽，并确认传输是否落在关键路径。

计算和资源准备汇到同一请求上，关系如下。

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

## 二、首 Token 与待传输的请求状态

Prefill 完成，`process_batch_result_disagg_prefill()` 从 `result.next_token_ids` 取首枚输出，执行 `req.output_ids.append(next_token_id)`，再进入最终 KV/state 发送。这个 ID 就是 handoff token。[`process_batch_result_disagg_prefill()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L723-L962)

`MetadataBuffers.set_buf()` 将 `req.output_ids[0]` 写进 `self.output_ids[req.metadata_buffer_index][0]`，经 PD metadata 把首枚 ID 交到 D。[`MetadataBuffers`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L288-L571)

### handoff Token 已采样，自身 KV 还没生成

它在 P 是输出，到 D 才作为下一次模型输入。

D 为 fresh request 预分配 Prompt 状态，使用下面的长度。

~~~python
return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
~~~

[`DecodePreallocQueue._pre_alloc_fill_len()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L808-L829) 的注释 `the last output token's KV has not been written yet`，解释了为什么要减一。

Prompt 长 N，P 已经算过 `0...N-1`，并采样出 h。

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

交接内容包含已计算的 Prompt 持久状态、handoff ID 和必要的辅助 metadata。

图的最后下标 n 对应 `N=n+1`，h 位于 N，当前只有 ID，待传 KV 还没包含这个位置。

D 的第一轮真实 Decode 消费 h，读取 `0...N-1`，计算、写入位置 N，再预测下一枚。这里沿正常 fresh handoff，rebootstrap 恢复另有长度规则。

Prompt 不用重算，h 的 KV/state 在 D 首轮 Decode 才形成。

## 三、Bootstrap 与目标资源预分配

传输得先知道 D 的目标位置，P 的计算却不总要等它。下面这种完全串行的顺序，不能概括所有配置。

~~~text
Decode 完成全部 prealloc
        ↓
Prefill 才允许开始计算
~~~

Bootstrap 交换请求身份、目标位置等信息。普通路径随就绪状态推进，optimistic prefill 则可以让 P 先算，等发送条件齐了再传。

### 两边先各自建立 Sender / Receiver

P 请求进入 `PrefillBootstrapQueue`，创建 `KVSender`。[`PrefillBootstrapQueue.create_sender()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L349-L418)

~~~python
req.disagg_kv_sender = kv_sender_class(
    mgr=self.kv_manager,
    bootstrap_addr=...,
    bootstrap_room=req.bootstrap_room,
    # 其余参数省略
)
~~~

请求上的标记仍是 Bootstrap 待完成。

~~~python
req.pending_bootstrap = True
~~~

D 创建 `KVReceiver`，把请求放入 `DecodePreallocQueue`。[`DecodePreallocQueue.add()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L673-L785)

~~~python
kv_receiver = kv_receiver_class(
    mgr=self.kv_manager,
    bootstrap_addr=_bootstrap_addr(req),
    bootstrap_room=req.bootstrap_room,
)
~~~

本地 allocator 接着分配这些位置。

~~~text
allocate req_pool row
allocate destination KV/state slots
allocate metadata slot
~~~

随后 D 调用 `send_metadata(...)`，发布目标页索引等信息。

`pop_preallocated()` 做完这些工作，请求进入 DecodeTransferQueue 等待数据。接收位置虽然有了，里面的状态还得等传输完成才有效。

### Prefill 正常路径等到 WaitingForInput 再 finalize bootstrap

P 轮询 Sender，得到 `KVPoll.WaitingForInput` 时，已经拿到可用的目标 metadata。

本地 metadata buffer 也得有槽位。`finalize_bootstrap()` 成功后初始化 Sender、清掉 `pending_bootstrap`。[`finalize_bootstrap()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L386-L406)

~~~python
decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
...
req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
req.pending_bootstrap = False
~~~

P、D 各自分配物理页，通过下面的映射对应。

~~~text
Prefill source pages
        ↓ mapping
Decode destination pages
~~~

例如这组教学页号。

~~~text
Prefill                    Decode

physical page 100  ─────► physical page 501
physical page 101  ─────► physical page 803
physical page 102  ─────► physical page 804
~~~

源 100 传到目标 501 没有问题，两边要对齐的是请求及逻辑状态区间，物理 page ID 可以不同。

<a id="optimistic-prefill-prealloc"></a>

### Optimistic Prefill 先算，再等目标位置

`pop_bootstrapped()` 保留了提前执行的分支。

~~~python
elif poll == KVPoll.Bootstrapping:
    if req.prefill_attempt_count < optimistic_prefill_attempts:
        ...
        bootstrapped_reqs.append(req)
~~~

在配置允许的尝试次数内，Bootstrap 没完成，P 也可以先计算输入。

算完后 D 还没给出目标，请求就停在 Prefill Inflight Queue。

~~~python
if req.pending_bootstrap:
    # Parked: prefill finished before bootstrap completed.
~~~

Sender 到 `WaitingForInput`，再完成 Bootstrap、发送最后一段 KV。[`process_disagg_prefill_inflight_queue()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L965-L1035)

计算提前了，源状态仍要保留到目标可用，资源准备没有省略。两条路径的发送条件如下。

| 路径 | Prefill 何时进入计算 | 传输前要满足什么 |
| --- | --- | --- |
| 普通 Bootstrap | 按正常就绪状态进入 | Sender 完成 Bootstrap，取得目标布局 |
| Optimistic Prefill | 在允许的尝试次数内提前计算 | 若目标尚未就绪，停留 Inflight 等待完成 Bootstrap |

实际发送以前，源页、目标页的对应都必须建立好。

## 四、传输完成后，Decode 怎样接管

Prefill 最后一个 chunk 将 handoff metadata 写入 MetadataBuffers，再准备 state indices。发送入口是 [`_send_kv_chunk()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/prefill.py#L1302-L1495)。

~~~python
if last_chunk:
    self.disagg_metadata_buffers.set_buf(req)
    ...
~~~

接着按当前请求的待传页发送。

~~~python
req.disagg_kv_sender.send(
    page_indices,
    send_state_indices,
    num_kv_tokens=...
)
~~~

P 发完继续在 Inflight Queue 轮询，相关 Rank 对成功终态达成一致，再释放本地缓存持有；失败按对应路径清理。发送调用返回，还不等于 D 已经接管请求。

D 的 `DecodeTransferQueue` 确认 Receiver 成功，调用 `self._commit_transfer_to_req(decode_req)`，把 metadata 提交到本地 `Req`。其中有 handoff output Token、缓存命中记账、logprob、采样参数、bootstrap 身份，以及启用投机时的 hidden/top-k 状态。函数注释也注明这枚 Token 由 P 生成。[`DecodeTransferQueue._commit_transfer_to_req()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2215-L2350)

### PREBUILT 重建请求元数据与输出状态

传输提交后的请求进入 D waiting queue，再组成 `ScheduleBatch`，执行 PREBUILT。[`_get_new_prebuilt_batch()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode.py#L2822-L2900)

~~~python
new_batch.prepare_for_prebuilt()
...
new_batch.process_prebuilt(self.future_map)
~~~

`prepare_for_prebuilt()` 恢复 `req_pool_indices`、`seq_lens`、`prefix_lens`、`extend_lens`、`out_cache_loc`、`sampling_info` 等调度状态。注释明确写着 `PREBUILT never enters a model forward.`，收到的 Prompt 不会在这里再送模型重算。[`ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py)

`process_prebuilt()` 再读取 `req.output_ids[-1]`，取出 P 交过来的最后一枚输出。

普通非投机路径把它存入 `FutureMap` relay。

~~~python
future_map.stash(
    self.req_pool_indices,
    RelayPayload(bonus_tokens=last_tokens_tensor),
)
~~~

下一次真实 Decode 从 relay 取得 `input_ids`。

PREBUILT 接起本地请求和已完成的 Prompt 状态，安排 handoff Token 的后续输入，本身不调用模型，也不生成新 Token。

接管过程对应下面的时序。

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

h 要等真实 Forward 才被计算。PREBUILT 已经完成，只说明请求元数据和输入交接好了，这时 h 的缓存仍没生成。

## 五、DeepSeek-V4 的 KV 与辅助状态

`DSV4NPUTokenToKVPool` 既保存 Attention 历史，也保存继续压缩、索引所需的辅助状态。它们有的按页、有的按请求寻址，传输时按各自注册项匹配。

### C4 主数据走 main PD buffer

`DSV4NPUTokenToKVPool.get_contiguous_buf_infos()` 返回 C4 KV、C4 indexer K 和 C4 index scale，按完整 KV page ID 寻址，经 main KV transfer component 交接。[`DSV4NPUTokenToKVPool.get_contiguous_buf_infos()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)

### SWA KV 是独立 state component

`setup_state_kv_args()` 为 `BaseSWAKVPool` 注册 `StateType.SWA`。NPU DSV4 的 `get_state_buf_infos()` 总先加入 SWA KV buffer，由这份独立 State Buffer 传输。

<a id="c4-compression-state-a3-a5-wire-layout"></a>

### C4 压缩状态在 A3 与 A5 上怎样传输

pre-A5 / explicit-location 路径包含这里的 A3 语义。SWA KV 后还跟着 C4 Attention、C4 Indexer 两份 compression state，共用 SWA page/state index 语义，一起注册到 `StateType.SWA`。

A5 / CYCLE cache 使用 request-local ring ownership，把 C4 state 从 SWA component 分出来，单列 `AscendStateType.DSV4_C4_STATE`。收发项目因此随布局变化，注册与类型入口见 [`setup_state_kv_args()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L1313-L1612)、[`DSV4NPUTokenToKVPool.get_state_buf_infos()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py)、[`AscendStateType`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/conn.py#L24-L34)。

### C128 KV 和 C128 request state 也是两件事

`token_to_kv_pool.get_c128_kv_buf_infos()` 提供 C128 KV，注册为 `AscendStateType.DSV4_C128`，保存已形成的压缩历史。压缩后续还要接着算，request-scoped state 另由 `get_request_state_buf_infos()` 提供，注册成 `StateType.DSV4_REQUEST_STATE`。

固定文档中的 request state 包括 C128 raw-token ring / online row，也可随配置包含其他 request-scoped pending state。它与 C128 KV 使用不同 Buffer、索引，接收端除了已有压缩历史，还需要这些中间状态才能继续计算。

<a id="speculative-decode-wire-contract"></a>

### 投机生成还需交接 Draft 状态

Target、Draft 都用 DeepSeek-V4 pool 时，`setup_state_kv_args()` 还检查 Draft NextN 层必须 SWA-only，将 Draft SWA buffers 注册为独立 positional component。Prefill metadata 也可能带 `topk_p`、`topk_index`、`hidden_states` 与 DSA top-k indices，供后续投机生成使用。

把这些状态放到同一张注册表里，能看出一块 KV Buffer 到齐还不够。

| 组件 | 当前 PD wire 角色 |
| --- | --- |
| C4 KV + indexer K/scale | main KV buffers |
| SWA KV | `StateType.SWA` |
| C4 attention/indexer compression state，A3/pre-A5 | 随 `StateType.SWA` |
| C4 compression state，A5/CYCLE | `AscendStateType.DSV4_C4_STATE` |
| C128 KV | `AscendStateType.DSV4_C128` |
| request-scoped C128/compressed state | `StateType.DSV4_REQUEST_STATE` |
| DSpark/NextN Draft SWA（适用配置） | 独立 Draft SWA state component |

后续计算沿着各自的页状态、请求状态和辅助缓冲继续。

## 六、Ascend 传输组件怎样分工

SGLang 的 PD backend 枚举如下。

~~~python
class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    MORI = "mori"
    NIXL = "nixl"
    ASCEND = "ascend"
    FAKE = "fake"
~~~

`get_kv_class()` 根据 backend 解析 `KVManager`、`KVSender`、`KVReceiver`、`BootstrapServer`，承担管理、收发与 Bootstrap。[`TransferBackend / get_kv_class()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/utils.py#L579-L698)

Ascend 复用了几个带 Mooncake 名字的类。

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

继承保留控制与 Sender/Receiver 协议框架，数据引擎则由 Manager 选择。

`AscendKVManager.init_engine()` 创建 `AscendTransferEngine(...)`。它沿用 `MooncakeTransferEngine` 的 register / transfer wrapper API，构造时实际把 `self.engine` 设为 `memfabric_hybrid.TransferEngine()`。初始化还传入 `ASCEND_MF_STORE_URL`、P/D 角色、NPU ID 与协议，数据由这一层移动。对应实现如下。

- [`AscendKVManager`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/conn.py)
- [`AscendTransferEngine`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/disaggregation/ascend/transfer_engine.py#L26-L111)
- [`MooncakeTransferEngine`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py)

从请求调度往下，实际经过的传输组件是这些。

~~~mermaid
flowchart TD
    S[Prefill/Decode Scheduler]
    S --> K[AscendKVSender / Receiver]
    K --> M[AscendKVManager]
    M --> A[AscendTransferEngine]
    A --> F[memfabric_hybrid.TransferEngine]
    F --> X[SDMA or device_rdma]
~~~

协议默认 `sdma`。设 `ASCEND_MF_TRANSFER_PROTOCOL=device_rdma` 后，实现会检查已安装 MemFabric 支持的协议类型，不支持便告警、回退 SDMA，最终值要看初始化日志。

使用 `device_rdma` 时，MemFabric 初始化以前先在 NPU world 做 `all_gather`，提前初始化 HCCL，避免与 RDMA 初始化冲突。

这次 collective 做初始化协调，PD Payload 随后仍经传输引擎，用实际选定的协议发送。

固定 Ascend support matrix 要求 NPU PD 显式指定 `--disaggregation-transfer-backend ascend`。全局默认仍是 `mooncake`，文档将该默认标为 NPU 不支持，Ascend 配置需要覆盖它。[Ascend Support Features](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/reference/support_features.mdx)

D 接收的是 P 已经算好的 Prompt 状态，以 handoff Token 开始自己的计算。DeepSeek-V4 的 SWA、压缩历史和辅助状态都要按路径接上，这些工作也一起计入 PD 的交接成本。

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
