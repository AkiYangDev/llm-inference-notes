# 日志里的 TP16 为什么应是 TP8：SGLang PR #39871

配置 `tp_size=16、dp_size=2`，启用 DP Attention，且 `attn_cp_size=1` 时，Attention TP 宽度是 8。SGLang 曾在相关日志中直接打印原始 `tp_size`，把它误报成 16；实际运行时的宽度派生并没有算错。

[PR #39871](https://github.com/sgl-project/sglang/pull/39871) 让这条日志改用已有的派生结果。这个小补丁适合观察配置系统里常见的问题：输入参数经过 Runtime 解释后，下游日志是否仍在使用未经解释的原值。

本文固定到 `sgl-project/sglang@803f0c93d20104cf19ff6a95f7ef580b9fe449a2`。如果想完整理解 WORLD、Rank、Group、DPA 与 MoE EP，可继续阅读[《SGLang 里的 Rank 和 Group 到底是什么？》](../distributed/sglang-rank-group-deepseek-v4.md)；本文只保留读懂这个 PR 所需的最小背景。

## 一、日志里的宽度与实际运行值

PR 给出的原始日志是：

```text
DSA with TP mode is active,
dp_size=2,
tp_size=16,
attn_tp_size=16,
attention weights will be sharded across 16 ranks.
```

真正的问题是最后两处 `16`。在这个配置下，Attention 实际 TP width 应该是 8。

如果只是普通业务日志写错一个数字，可能影响不大；但分布式推理的日志经常承担另一种职责：**它是工程师理解当前拓扑的入口。**

可以把这次 Bug 的 Impact Surface 写得很具体：

| 维度 | #39871 的情况 |
| --- | --- |
| Trigger | DSA、`dp_size < tp_size`、未走 prefill CP 分支，并进入 CUDA / ROCm model-specific adjustment |
| Symptom | warning 把 `attn_tp_size` 打成 `cfg.tp_size` |
| Runtime behavior | 不受影响，真实并行宽度没有被这个日志修改 |
| User-visible impact | Debug / Profiling 时看到错误的 Attention sharding width |
| Blast radius | 主要影响观察与排障认知，不是模型数值正确性 |
| Persistence | 每次进入该启动路径都可能打印同样的错误拓扑信息 |

为什么这会误导排障？

假设你看到：`AttnTP = 16`

很自然会开始问：

```text
是不是 16-way TP communication 太重？
是不是 16 Rank collective 带宽不够？
是不是 Attention weight 真被切成了 16 份？
```

但真实 topology 如果是：`AttnTP = 8`

那么整个问题模型从第一步就错了。

所以这个 PR 的影响可以浓缩成：

```text
Runtime Truth       = 8
Operational Truth   = 16
```

这里的 Correctness 不是模型数学 correctness，而是 **Observability Correctness**：人看到的系统必须和机器实际运行的系统一致。

## 二、从 TP16 / DP2 推导 Attention TP8

理解这个 PR，只需要先看一个函数：

[`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/803f0c93d20104cf19ff6a95f7ef580b9fe449a2/python/sglang/srt/runtime_context.py#L138-L148)

固定 merge commit 下的实现是：

```python
def derive_attention_widths(
    *, tp_size: int, attn_cp_size: int, dp_size: int, enable_dp_attention: bool
) -> tuple:
    attn_dp_size = dp_size if enable_dp_attention else 1
    return attn_dp_size, tp_size // attn_dp_size // attn_cp_size
```

先决定 Attention 是否真的使用 DP：

```text
enable_dp_attention = true  → attn_dp_size = dp_size
enable_dp_attention = false → attn_dp_size = 1
```

然后计算 Attention 自己的 TP width：

```text
attn_tp_size
=
tp_size / attn_dp_size / attn_cp_size
```

因此在合法配置里，可以把关系记成：

```text
tp_size
=
attn_dp_size
× attn_cp_size
× attn_tp_size
```

代入 PR 的例子：

```text
tp_size = 16
dp_size = 2
attn_cp_size = 1
enable_dp_attention = true

attn_dp_size = 2
attn_tp_size = 16 / 2 / 1 = 8
```

概念上可以画成：

```mermaid
flowchart TB
    W["TP world = 16 ranks"]
    W --> D0["Attention DP 0"]
    W --> D1["Attention DP 1"]
    D0 --> T0["Attention TP group · 8 ranks"]
    D1 --> T1["Attention TP group · 8 ranks"]
```

所以：`tp_size = 16`

和：`attn_tp_size = 8`

完全可以同时成立。

这也是为什么看到：

```python
attn_tp_size == 1
```

不能翻译成“整个模型单卡运行”。它只表示 **Attention 这套并行坐标中的 TP width 为 1**。

本文不继续把 DP/CP/Rank/Group 全部展开，因为那会和已有分布式专题重复。对这个 PR 来说，最重要的只有一句：

> **原始配置里的 `tp_size`，不是所有子模块最终运行时都必须直接采用的 TP width。**

## 三、派生规则与错误日志的来源

**在 #39871 之前，Attention TP 的派生规则是怎样组织的？**

这里能找到一个很有价值的前置设计背景。

在 #39871 之前约两周，SGLang 合入了 [PR #38113](https://github.com/sgl-project/sglang/pull/38113)。那次配置系统整理明确强调：`attn_tp_size` 等 parallel quotients 是由 configured leaves 派生的值，相关 arithmetic 应该有统一来源，而不是让各个消费者各写一份。

PR #38113 不是本文能够证明的“Bug introducing PR”。我没有找到可靠证据说明错误 warning 就是由它引入；因此更准确的 History Chain 是：

```text
#38113
Parallel quotient 语义进一步集中
attn_tp_size 明确属于 derived topology
        │
        ▼
Runtime 已有统一 width derivation
        │
        ▼
某个旧 Logging consumer
仍然直接打印 cfg.tp_size
        │
        ▼
#39871
让 Logging 回到同一个 derivation source
```

也就是说，#39871 很像一种常见的重构尾部 Bug：

> **核心语义已经集中，但还有一个边缘消费者没有迁移到新的 Single Source of Truth。**

固定 merge commit 中，Runtime helper 自己的注释已经把这个意图说得很清楚：

```text
Split out because the rank computation in
dp_attention.compute_dp_attention_world_info
needs the same two numbers and must not carry
a second copy of the arithmetic.
```

Runtime 路径因此已经是：

```text
cfg.tp_size
cfg.dp_size
cfg.attn_cp_size
cfg.enable_dp_attention
        │
        ▼
derive_attention_widths()
        │
        ├── attn_dp_size
        └── attn_tp_size = 8
```

旧 Logging 却旁路了它：

```mermaid
flowchart LR
    subgraph B[Before]
        direction TB
        BC["Config · tp=16 / dp=2 / cp=1 / DPA=on"]
        BD["derive_attention_widths()"]
        BR["Runtime topology · AttnTP = 8 ✓"]
        BL["Logging reads cfg.tp_size"]
        BW["Log · AttnTP = 16 ✗"]
        BC --> BD --> BR
        BC --> BL --> BW
    end

    subgraph A[After]
        direction TB
        AC["Config"]
        AD["derive_attention_widths()"]
        AR["Runtime · AttnTP = 8 ✓"]
        AL["Logging reuses derived width"]
        AW["Log · AttnTP = 8 ✓"]
        AC --> AD
        AD --> AR
        AD --> AL --> AW
    end
```

因此这个 PR 的 Root Cause 不是“有人不会算 `16 / 2`”，而是：

> **Logging 直接消费了 configured leaf，而没有消费已经存在的 runtime-derived semantic value。**

进一步可以写出核心 invariant：

> **Invariant：所有描述 Attention TP width 的消费者，都必须与 Rank / Group 构建共享同一套 width derivation；不能让原始配置值和模块派生值各自形成一套事实。**

这个 invariant 比 `16 → 8` 这个具体数字稳定得多。

## 四、补丁怎样复用统一计算入口

修复后的代码位于 [`model_hook.py`](https://github.com/sgl-project/sglang/blob/803f0c93d20104cf19ff6a95f7ef580b9fe449a2/python/sglang/srt/arg_groups/model_hook.py#L285-L300)。

如果按设计意图拆，只有三个动作。

**第一，复用现有的权威派生函数。**

```python
_, attn_tp_size = derive_attention_widths(
    tp_size=cfg.tp_size,
    attn_cp_size=cfg.attn_cp_size,
    dp_size=cfg.dp_size,
    enable_dp_attention=cfg.enable_dp_attention,
)
```

**第二，把 Logging 从原始配置值切到 derived value。**

旧逻辑等价于：

```python
attn_tp_size = cfg.tp_size
```

新逻辑变成：

```python
attn_tp_size = derive_attention_widths(...)[1]
```

**第三，保持 Runtime 行为完全不变。**

Process Group、Attention Kernel、Collective、Weight Sharding 都没有被这个 PR 修改。

### 为什么不直接写 `cfg.tp_size // cfg.dp_size`？

因为那只是“把当前样例算对”，却重新复制了一份系统规则。

真实公式还涉及：

```text
enable_dp_attention
attn_cp_size
```

如果不同消费者自己写：

```text
A.py → tp / dp
B.py → tp / dp / cp
C.py → DPA 开了才除
D.py → 直接打印 tp
```

迟早会再次漂移。

而 `derive_attention_widths()` 本身存在的理由，就是让 Rank 计算和其他消费者**不要持有第二份 arithmetic**。

所以这个修复真正重要的不是：`16 / 2 = 8`

而是：

```text
One semantic rule
      ↓
One derivation home
      ↓
Many consumers reuse it
```

这就是 Single Source of Truth 在 Runtime 代码里的具体样子。

## 五、检查日志变化与行为范围

这次改动属于 log-only；下面分别核对日志变化、测试覆盖和行为范围。

先做 Evidence Ladder：

| 证据类型 | 能支持的结论 |
| --- | --- |
| **Source-confirmed** | merge commit 中 Logging 调用 `derive_attention_widths()`；helper 的公式和共享 arithmetic 目的可直接从源码确认 |
| **PR-confirmed** | PR 作者明确说明错误发生在 partial DP Attention warning，且是 log-only、no behavioral impact |
| **Measured** | 没有 accuracy benchmark，也没有 performance benchmark；PR 明确认为二者不适用于这次 log-only 改动 |
| **Inference / Transfer** | 其他日志项也应检查配置值与有效运行值是否混用 |

再看 Validation Surface：

| Surface | 状态 | 说明 |
| --- | --- | --- |
| Source diff | covered | 1 个文件、`+8/-2`，修改边界非常清楚 |
| Lint | reported-pass | 对应 PR head 的 Lint workflow 成功 |
| Dedicated topology runtime test | not shown | PR 没有新增针对该 warning 的专门 runtime test |
| Accuracy | not-run | PR 明确标为不需要 |
| Performance / Profiling | not-run | PR 明确标为不需要 |
| Broad PR workflows | mixed / red | 多个 PR Test workflow 在该 head 上记录为 failure，因此不能把“整套 CI green”当成本 PR 的验证证据 |
| Ascend runtime | out of scope | 这段代码被 `not is_npu and not is_xpu` 条件排除 |

这里有一个很重要的 Maintainer 习惯：

> **CI 存在，不等于我们可以笼统写“CI 已验证”。要看目标测试是否真的执行、它验证的又是哪一层 contract。**

对于 #39871，真正能强证实的是：

```text
旧日志读取了错误的 source
新日志复用了正确 derivation
Runtime 行为没有被 diff 修改
```

而不是：

```text
所有 DSA / 所有平台 / 所有 topology
都经过了专门 runtime regression test
```

### 补丁的行为范围

它没有改变：

- Attention Kernel；
- Process Group 创建；
- Weight Sharding；
- Collective Communication；
- 模型精度；
- Runtime 性能；
- Ascend NPU / XPU 的这条 DSA 分支。

固定源码里的外层条件明确是：

```python
if not get_platform().is_npu and not get_platform().is_xpu:
    # CUDA or ROCm GPU
```

因此最准确的表述是：

> **#39871 修复了 CUDA / ROCm DSA model-specific adjustment 中 partial DP Attention 的日志语义；它没有修复或改变 Ascend 910C 的 Runtime Parallelism。**

## 六、配置派生值的使用原则

原始配置和有效运行值承担不同职责。`tp_size` 用来表达用户的配置意图；日志描述 Attention 的实际宽度时，应使用同一套派生规则得到的 `attn_tp_size`。把算术公式再复制一遍，即使眼下能得到 8，也会多出一个需要跟随 DP、CP 和启用条件维护的位置。

因此，这个补丁的价值是让日志与 Runtime 共用含义一致的结果。复核时只需把链路走完：配置进入派生函数，Runtime 消费派生状态，日志读取同一语义，再用目标配置检查输出。

### 固定源码与延伸阅读

- [SGLang PR #39871](https://github.com/sgl-project/sglang/pull/39871)
- [PR #38113：parallel quotients / derived config 的前置设计背景](https://github.com/sgl-project/sglang/pull/38113)
- [`model_hook.py`：修复后的 warning](https://github.com/sgl-project/sglang/blob/803f0c93d20104cf19ff6a95f7ef580b9fe449a2/python/sglang/srt/arg_groups/model_hook.py#L285-L300)
- [`runtime_context.py`：`derive_attention_widths()`](https://github.com/sgl-project/sglang/blob/803f0c93d20104cf19ff6a95f7ef580b9fe449a2/python/sglang/srt/runtime_context.py#L138-L148)
- [`runtime_context.py`：parallel width derivation](https://github.com/sgl-project/sglang/blob/803f0c93d20104cf19ff6a95f7ef580b9fe449a2/python/sglang/srt/runtime_context.py#L151-L205)
- [《SGLang 里的 Rank 和 Group 到底是什么？》](../distributed/sglang-rank-group-deepseek-v4.md)
