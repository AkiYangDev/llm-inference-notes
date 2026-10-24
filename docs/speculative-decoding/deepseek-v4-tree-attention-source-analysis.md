# EAGLE Tree Attention：候选布局、位置与可见性

一棵候选树可以按 `[root, A, B, C, D]` 排进 Tensor，每行仍保留原来的父子关系。假设 C 是 A 的子节点、D 是 B 的子节点，C 应该读取 root 和 A，却不该读取另一条分支上的 B。

Tree Attention 要让一次 Forward 中每行的结果，都等价于沿它自己的祖先路径执行，而不是沿 Tensor 的物理顺序执行。下面用同一棵小树对齐行号、位置和可见性，再看后端怎样用 Mask、页表或 Cascade 实现这些关系。当前 DeepSeek-V4 的 EAGLE 配置仍限定 `topk=1`，多分支部分讨论通用 Runtime，不能直接当成 V4 已支持的部署方式。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码基线**：本文基于 `sgl-project/sglang @ 791c7850d0960fd768102f71e7d999b036bb75ba`，核对日期为 2026-09-20。重点覆盖 EAGLE Spec V2 的 tree builder、VerifyMask、Triton / FlashAttention target-verify，以及 DeepSeek-V4 DSV4 的当前 capability gate。
>
> **DeepSeek-V4 当前边界**：固定版本中，DeepSeek-V4 的 EAGLE 配置仍要求 `speculative_eagle_topk == 1`；CUDA/HIP DSV4 Attention backend 还会独立 assert `topk in [0, 1]`。因此本文的 multi-branch Tree Attention 是对 **SGLang 通用 EAGLE Tree Verify 机制，以及 DeepSeek-V4 若未来放开 topk>1 所必须满足的 correctness contract** 的源码解析，而不是声称当前 DeepSeek-V4 生产路径已经在跑多分支 Tree Attention。

</details>

## 一、把候选树放进 Verify Tensor

### 祖先关系与可见历史

先看最简单的例子。

假设 prefix 已经确定：`P0 P1 P2 ... Pn`

Draft Model 提出：

~~~text
                 root
               /      \
              A        B
              |        |
              C        D
~~~

概念上，可以把每条路径分别交给 Target，作为理解语义的参照：

~~~text
Target(prefix + root + A + C)

Target(prefix + root + B + D)
~~~

只要各路径的输入、位置与可见性正确，分开计算可以得到各自的 Target 结果，但会重复处理共享上下文。这里希望把候选合进一次 Target Forward，并保持同样的路径语义；是否更快还取决于实际行数、后端与状态整理成本。

~~~text
一次 Target Forward

物理输入：
[root, A, B, C, D]

逻辑语义：
root → A → C

以及

root → B → D
~~~

物理布局只决定行放在哪里，祖先关系才决定这一行能读哪些候选。不同分支共享已确认 Prefix 和 root，同深度的 A、B 则是互斥未来，不能因为 B 排在 A 后面就把 A 当成历史。下面先讨论非滑窗的标准可见性；SWA 层还要叠加各 Query 的窗口约束，后文单独说明。

#### 普通 causal mask 为什么会错

如果 Verify row 顺序是：

~~~text
row 0 = root
row 1 = A
row 2 = B
row 3 = C
row 4 = D
~~~

普通 lower-triangular causal mask 会得到：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 1 1 0 0
  3       1 1 1 1 0
  4       1 1 1 1 1
~~~

于是：

~~~text
B 看见 A
C 看见 B
D 看见 A/B/C
~~~

这些都是 sibling / cross-branch leakage。

正确的 candidate visibility 应该是：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 0 1 0 0
  3       1 1 0 1 0
  4       1 0 1 0 1
~~~

Tree Attention 的核心规则可以写成：

M(i,j)=1

当且仅当：

~~~text
j 属于 committed prefix

或者

j == i

或者

j 是 i 的 ancestor
~~~

否则：

M(i,j)=0

因此 Tree Attention 与普通 causal attention 的根本区别是：

~~~text
普通 causal：
“物理上排在我前面”
        ↓
“属于我的历史”

Tree Attention：
“逻辑上是我的 ancestor”
        ↓
“属于我的历史”
~~~

#### 为什么这直接决定 Accept correctness

Target Verify 最终希望满足：

~~~text
L(v)
=
Target(prefix + path(root → v))
~~~

也就是说，node `v` 对应的 logits 必须来自它自己的 root-to-v history。

如果 Attention 阶段已经让 sibling 泄漏进来：

~~~text
L(v)
=
Target(prefix + physical rows before v)
~~~

那么后面的：

~~~text
retrieve_next_token
retrieve_next_sibling
verify_tree_greedy
~~~

无法修复这些已经算错上下文的 logits。Accept 只能在给定预测上选择路径，不会重新执行一次正确 Attention。因此先核对行的可见历史，再检查遍历链接，不能只因为接受索引“看上去走对了”就认为整轮正确。

### 候选池到 Verify 行

生成 Mask 前，必须先确定候选池与 Verify 行之间的映射：

~~~text
每一个 Verify row
到底是哪一个 candidate node
~~~

当前 EAGLE V2 的 Draft 主入口在：

[`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_v2.py)

多步 Draft 会不断维护：

~~~text
score_list
token_list
parents_list
~~~

后续 candidate score 不是单步概率，而是累计 path score。

在：

[`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/spec_utils.py)

中：

~~~python
expand_scores =
    scores.unsqueeze(2)
    * topk_p.view(-1, topk, topk)
~~~

例如：

~~~text
root
├─ A 0.60
│  ├─ C 0.70  → path score 0.42
│  └─ E 0.20  → path score 0.12
│
└─ B 0.40
   ├─ D 0.80  → path score 0.32
   └─ F 0.10  → path score 0.04
~~~

最终并不是把整个 expansion tree 都交给 Target。

`organize_draft_results()` 会：

~~~text
1. flatten score pool
2. top-k 选出 num_draft_token - 1 个 candidate
3. sort selected indices
4. gather 真正进入 Verify 的 draft tokens
5. 保留 parent topology
~~~

固定源码：

[`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

这里最需要区分两个 index space：

~~~text
top_scores_index / selected_index
        │
        └─ Draft expansion candidate-pool space

retrieve_index
        │
        └─ Compact Target-Verify row space
~~~

可以画成：

~~~text
Draft Expansion Pool

#0 #1 #2 #3 #4 #5 #6 ...
        │
        │ top_scores_index
        ▼
Selected Candidate Set

#0 #1 #3 #5 ...
        │
        │ build_tree_kernel
        ▼
Compact Verify Rows

row0 row1 row2 row3 ...
~~~

Tree Attention 消费的是最终 Verify 行。这里的 compact 指从候选池裁剪、排好后的树节点数组，不等同于 DSpark 自适应 ragged Verify 的执行模式。多请求 batch 中，`retrieve_index` 还会带上该请求的全局 Verify 行偏移，不能一律按每请求从 0 开始的 node id 使用。

#### Verify root 不是普通 draft candidate

进入：

[`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

后第一步就是：

~~~python
draft_tokens = torch.cat(
    (
        bonus_tokens.unsqueeze(1),
        draft_tokens,
    ),
    dim=1,
).flatten()
~~~

Target Verify 实际输入是 `[上一轮 bonus/root, selected draft candidates...]`。root 虽然不是这轮新提出的候选，却必须作为输入计算自己的状态，并提供检查第一层候选的预测。它占据窗口第一行，也影响后面的 position 与 KV 边界。

## 二、位置编号与跨轮 KV 边界

### 同深度节点共享时间坐标

Mask 决定可见对象，position 决定模型把这一行放在序列的哪个时间位置。两者来自同一棵树，但不能互相替代。

考虑：

~~~text
root
├─ A
│  └─ C
└─ B
   └─ D
~~~

假设当前 KV-ready prefix 长度是：`P`

正确 Position 必须是：

~~~text
root → P

A    → P + 1
B    → P + 1

C    → P + 2
D    → P + 2
~~~

若直接使用物理行号，会得到下面这组与树深度不符的位置：

~~~text
root → P
A    → P+1
B    → P+2
C    → P+3
D    → P+4
~~~

A、B 同在 root 后一层，没有先后发生的关系，所以都使用 P+1；C、D 再向后一层，都使用 P+2。row 编号用于区分物理行，position 用于表示逻辑深度。用同一个整数同时承担这两种含义，就会把分支排成一条假历史。

~~~text
same tree depth
=
same logical position
~~~

#### 源码确实按 depth 算 Position

Tree kernel 在 root 位置写：

~~~python
positions[root] = seq_len
~~~

其它 node 则沿 parent relation 向上回溯，统计 depth：

~~~text
position
=
seq_len + tree_depth
~~~

固定 Triton 实现：

[`spec_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py)

源码注释给出的例子就是：

~~~text
depth:
[0, 1, 1, 2]

prefix length:
7

positions:
[7, 8, 8, 9]
~~~

测试源码还给出一组更长树的预期值，可以用来对照这些索引的含义。

[`test_build_eagle_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/test/registered/spec/utils/test_build_eagle_tree.py)

其中第一条 request 的预期 position 是：

~~~text
positions =
[5, 6, 6, 7, 7, 8, 8, 9]
~~~

配合：

~~~text
retrieve_next_token =
[1, 3, 4, 5, 6, 7, -1, -1]

retrieve_next_sibling =
[-1, 2, -1, -1, -1, -1, -1, -1]
~~~

对应的树是：

~~~text
row0 pos5
├─ row1 pos6
│  └─ row3 pos7
│     └─ row5 pos8
│        └─ row7 pos9
└─ row2 pos6
   └─ row4 pos7
      └─ row6 pos8
~~~

#### Mask 正确而 Position 错，仍然会错

假设 B 的 mask 已经保证它只看：`prefix + root + B`

但误把 B 的 position 编成：`P + 2`

而不是：`P + 1`

对于 RoPE 模型，B 的位置变化会改变它与可见历史之间的相对旋转关系，从而可能改变 Attention score 与 logits。并非每次数值变化都会翻转最终 argmax，但 Mask 正确本身已不足以保证结果等价。下图表示这条错误传播路径：

~~~text
Position 错
  ↓
RoPE phase 错
  ↓
QK score 错
  ↓
hidden state 错
  ↓
Target logits 错
  ↓
Accept 可能错
~~~

~~~text
Tree Mask
回答：
“我能看谁？”

Tree Position
回答：
“我处在哪个未来时间步？”
~~~

两者缺一不可。

### 旧 root 与新 bonus

这决定了后面 mask 与位置映射必须同时满足的约束。

直觉上容易认为：

~~~text
上一轮 bonus 已经输出
        ↓
它一定已经在 KV Cache 里
~~~

但 EAGLE 的流水线不是这样。

#### `batch.seq_lens` 在这里更接近 KV-ready boundary

`prepare_for_draft()` 对 paged tree layout 有一段非常直接的源码注释：

~~~text
base is batch.seq_lens
(== KV-ready committed prefix at draft time;
the bonus is the tree root written by verify,
not part of [0:seq_lens])
~~~

固定源码：

[`prepare_for_draft()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_common.py)

这里 `batch.seq_lens=P` 表示 `[0,P)` 已经是 KV-ready prefix。上一轮产生的 terminal bonus R 已经成为逻辑输出，但还没有作为 Target 输入执行，它自己的 KV 不在这段 Prefix 内。下一轮以 R 为 root，正好使用 `position(R)=P`；若提前把 R 算进 Prefix，再从 P+1 开始写，就会跳过它本应计算的位置。

同时 `eagle_prepare_for_verify()` 从：

~~~python
start_offset=batch.seq_lens
~~~

开始给 Verify rows 分配写入位置。

固定源码：

[`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

~~~text
上一轮 terminal bonus
        │
        │ 已输出 / 已成为逻辑 frontier
        ▼
下一轮 root
        │
        │ position = old seq_len
        ▼
本轮 Target Verify
        │
        ▼
此时才真正生成 root KV
~~~

#### 为什么 `accept_lens` 同时能推进 KV-ready boundary

假设请求继续生成，当前 KV-ready prefix 是 `[0,P)`，上一轮 bonus 是 R，本轮 Draft 在它后面提出下面三枚候选。这里先不计上层 EOS、stop 或输出长度截断：

~~~text
D1
D2
D3
~~~

Target Verify 的**输入 rows**是：

~~~text
row0 = R
row1 = D1
row2 = D2
row3 = D3
~~~

对应 Position：

~~~text
P
P+1
P+2
P+3
~~~

而每个 row 的 logits 分别预测：

~~~text
row0(R)  → predicts D1
row1(D1) → predicts D2
row2(D2) → predicts D3
row3(D3) → predicts new bonus B
~~~

如果 D1/D2/D3 都被接受，那么：

~~~text
num_correct_drafts = 3
accept_lens = 4
~~~

用户侧本轮 accepted outputs 是：`D1 D2 D3 B`

但是本轮算出、可以进入 KV-ready prefix 的 Verify input rows 是：`R D1 D2 D3`

同样正好 4 个。

~~~python
new_seq_lens =
    batch.seq_lens + accept_lens
~~~

数值上完全正确：

~~~text
new KV-ready boundary
=
P + 4
~~~

边界推进后，新增的输入 KV 覆盖 R、D1、D2、D3；新输出 B 尚无自己的 KV，位于 `position(B)=P+4`，等待下一轮作为 root 执行。加 4 同时适用于输出推进与输入 KV 边界，不代表两边是同一组 token。

因此每轮都存在这样一个 frontier：

~~~text
KV-ready prefix
|=======================|

                        B
                        ▲
                  已输出的 terminal bonus
                  但 KV 尚未 materialize
~~~

下一轮：

~~~text
|=======================| B
                          │
                          ▼
                    Verify root
                          │
                          ▼
                    materialize KV
~~~

bonus 和 root 是同一枚 token 在相邻轮次的两种角色。记录日志时应同时标轮次与输入/输出侧，避免把本轮旧 root 和本轮新 bonus 混在一起。

#### 将位置编号与候选行对应起来

候选池裁剪后：

~~~text
accept_lens
=
num_correct_drafts + 1
~~~

而 accepted KV relocation 同样处理 `accept_lens` 个 Verify rows。

现在可以更准确地理解这两个同样的数字。

它们对应的是两组**错开一格**的 token：

~~~text
本轮 accepted outputs：
D1 D2 D3 B

本轮 committed Verify/KV rows：
R  D1 D2 D3
~~~

两边长度都等于：`4`

但 token identity 向前错了一格。

这个一位偏移连接了本轮输入 KV 与本轮预测输出，后续 Compact 继续沿用同一对应关系。

## 三、FULL_MASK 与 QLEN_ONLY 怎样表达可见性

位置与跨轮边界确定后，再看后端怎样表示候选的可见性。以下 FULL_MASK 和 QLEN_ONLY 描述存储布局，不是两套不同的祖先规则。

当前 SGLang 定义：

~~~python
class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2
~~~

固定源码：

[`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

### FULL_MASK：显式包含 Prefix + Tree Block

对于一个 request，FULL_MASK 的逻辑形状近似：

~~~text
[num_verify_tokens,
 seq_len + num_verify_tokens]
~~~

例如：

~~~text
                 Prefix KV                  Candidate KV
        ┌────────────────────────┬─────────────────────────┐
        │ P0 P1 ... P{L-1}      │ 0 1 2 3 4 ...          │
┌───────┼────────────────────────┼─────────────────────────┤
│ Q0    │ 1  1 ... 1            │ tree ancestry           │
│ Q1    │ 1  1 ... 1            │ tree ancestry           │
│ Q2    │ 1  1 ... 1            │ tree ancestry           │
└───────┴────────────────────────┴─────────────────────────┘
~~~

~~~text
Prefix:
所有 candidate 都可见

Candidate block:
只允许 self + ancestors
~~~

FULL_MASK 的 size 大致是：`B × N × (L + N)`

这里 B 是请求数、N 是 Verify 窗口行数、L 是 Prefix 长度。以教学配置 B=64、N=64、L=32768，按每个 bool 元素 1 Byte 估算，FULL_MASK 有 134479872 个元素，约 128.25 MiB；QLEN_ONLY 的 `64×64²` 只有 262144 个元素，即 256 KiB。这只是 Mask 存储量，不是 KV 容量或 Attention 耗时的测量。

`verify_mask.py` 直接写道：`FULL_MASK reaches 100s of MB at long context`

### QLEN_ONLY：只物化 Candidate × Candidate Tree Block

QLEN_ONLY 只保留：`[N, N]`

例如：

~~~text
        0 1 2 3 4

0       1 0 0 0 0
1       1 1 0 0 0
2       1 0 1 0 0
3       1 1 0 1 0
4       1 0 1 0 1
~~~

Prefix 不在这块 Tensor 中显式展开，后端仍必须正确读取该行应见的历史。小 buffer 省掉的是重复的 Prefix mask 表示，不是把 Prefix Attention 本身关掉。

因此它把 mask storage 从：`O(B × N × (L + N))`

降成：`O(B × N²)`

### 各后端如何消费 FULL 与 QLEN

当前还有：

[`VerifyMask`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/verify_mask.py)

它除了：

~~~text
buffer
mode
max_bs
~~~

还带：

~~~python
is_read
~~~

`is_read` 区分后端是否读取这块 buffer。Tree Builder 还会生成 position 和 retrieve links，因此统一调用链可能保留一个写入 buffer，即使目标后端并不从它提取可见性。检查到 buffer 已分配、甚至写过值，并不能证明 Attention 实际按它执行。

Graph 使用的 `VerifyMask` 通常按 `max_bs` 和 `max_context_len` 预分配；估算这块静态显存时，应代入分配上限，而不是只代入当前某条请求的实际 seq_len。

例如当前 CUDA `DeepseekV4AttnBackend`：`Verify metadata never extracts the mask.`

然后：

~~~python
maybe_create_verify_mask(
    ...,
    is_read=False,
)
~~~

固定源码：

[`DeepseekV4AttnBackend.init_cuda_graph_state()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py)

因为 `is_read=False`，`maybe_create_verify_mask()` 会选择更小的：`QLEN_ONLY`

当前 DSV4 配置使用 `topk=1`，其 Verify Metadata 不读取这块 Mask。其他支持 EAGLE Tree 的 Backend，可以用 FULL_MASK、QLEN_ONLY、页表改写或 Retrieve Topology 表达祖先可见性。非 SWA 场景下，完整可见 Prefix 与当前节点的祖先后缀共同构成它的 Attention 历史。

### Triton 是最直观的 Direct-Mask 实现

Triton Target Verify 直接：

~~~python
custom_mask = spec_info.custom_mask
~~~

随后把：

~~~text
custom_mask
mask_indptr
qo_indptr
kv_indptr
kv_indices
~~~

交给 `extend_attention_fwd()`。

固定源码：

[`triton_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/triton_backend.py)

它最接近：

~~~text
Tree
 ↓
FULL_MASK
 ↓
Attention Kernel
~~~

### 线性 / Hybrid backend 甚至可能传 Tree Topology 而不是 Bool Mask

例如 Ascend hybrid linear attention 对 `topk > 1` 的 target verify 会携带：

~~~text
retrieve_next_token
retrieve_next_sibling
retrieve_parent_token
~~~

这种路径可以直接消费 topology metadata，具体调用见固定版本的 [Ascend hybrid linear backend](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py)。它说明通用 Runtime 可以用不同表示承接树关系，不表示 DeepSeek-V4 的模型与 DSV4 后端限制已被解除。

~~~text
Tree Attention semantics
≠
必须存在一张 bool Tree Mask
~~~

## 四、FlashAttention Cascade 怎样合并结果

这里还要看具体后端的消费方式。

当前 FlashAttention backend 在：

~~~python
forward_batch.forward_mode.is_target_verify()
and self.topk > 1
and not is_swa_layer
~~~

时设置：

~~~python
use_cascade_attn = True
~~~

源码注释直接写：`We do cascade attention for Target Verify with topk > 1`

固定源码：

[`flashattention_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/flashattention_backend.py)

这条路径通过两段 Attention 与结果合并实现树语义，而不要求设备核直接理解一棵树。两段分别读取所有 Query 共享的已确认 Prefix，以及各 Query 自己的合法祖先后缀。

### 第一部分：所有 Query 共享的 Common Prefix

对于 topk>1，第一份 metadata：`target_verify_metadata_topk_normal`

只让 query attend：`committed prefix`

这里所有 tree node 的 prefix context 都完全相同，所以不需要为每个 branch 重复存整份 mask。

### 第二部分：每个 Query 自己的 Ancestor Suffix

源码从 `custom_mask` 中抽出 candidate block：

~~~python
mask =
    spec_info.custom_mask[
        mask_extraction_indices
    ].view(
        -1,
        speculative_num_draft_tokens,
    )
~~~

然后把：`mask == True`

的 candidate physical slots 排到每个 query 自己的 suffix page table 前面。

源码中的注释例子非常直观。

原始 candidate slots：`[8, 9, 10]`

Tree mask：

~~~text
query0 [1, 0, 0]
query1 [1, 1, 0]
query2 [1, 0, 1]
~~~

最后可以变成：

~~~text
query0:
[8]
len = 1

query1:
[8, 9]
len = 2

query2:
[8, 10]
len = 2
~~~

后端先筛出每行合法的 suffix slot，再用有效长度告诉普通 Attention kernel 读到哪里。query2 的列表是 `[8,10]`，不能偷换成连续范围 `[8,9,10]`；同样，页表后面即使仍有其他候选地址，也不能超过本行长度读取。

所以这里的工程转换是：

~~~text
Tree Topology
      ↓
Tree Mask
      ↓
Per-query Ancestor KV List
      ↓
普通 FlashAttention
~~~

### 用 LSE 合并 Prefix 与 Suffix 的 Attention 结果

第一段 Attention 得到：`(O_prefix, LSE_prefix)`

第二段得到：`(O_tree, LSE_tree)`

最终用：

~~~python
merge_state_v2_wrapper(...)
~~~

合并。

两段输出各自已经做过 softmax 归一化，直接相加或取平均都不对。令两段未归一化权重总和为 `Z_prefix`、`Z_tree`，LSE 保存各自的 log-sum-exp，合并时使用 `O=(Z_prefix×O_prefix+Z_tree×O_tree)/(Z_prefix+Z_tree)`。这要求两段使用相同 Query 与一致的 score 定义，且 KV 集合按这里的 Prefix/Suffix 划分，不能把重叠部分重复计入。

用一个标量教学例子：若 Z 分别是 3 和 1，归一化输出分别是 2 和 10，则合并结果为 `(3×2+1×10)/4=4`，不是平均后的 6。LSE 的作用是保留两段各自占多少归一化质量，使普通 Attention 的分段执行能够还原共同的 softmax。

最终效果等价于：

~~~text
Attention(
    query,
    committed prefix
    ∪
    allowed ancestor suffix
)
~~~

FlashAttention 在这条路径上通过 Prefix/Suffix Cascade 与 Softmax State Merge，实现候选树的祖先可见性。

### SWA 是一个重要例外

源码还明确：`We don't use cascade attention for Sliding Window Attention`

原因包括：

~~~text
不同 query 需要不同 window size，
而 FA3 cascade interface 不能传一组不同 window sizes。
~~~

所以 SWA 层不走这条 common-prefix cascade，而使用展开后的 spec metadata。

两种路径分别是：

~~~text
FlashAttention 非-SWA topk>1 Target Verify
→ cascade Tree Attention

FlashAttention SWA topk>1
→ 另一套 expanded metadata / page-table 路径
~~~

这也解释了前文的可见性范围：非 SWA 层可以共享完整 Prefix；SWA 层还要按 Query position 限制窗口，不能照搬“每行看见相同全部 Prefix”的两段拆法。同一个 Backend 按层类型选择不同 metadata，不能看到一个 Tree Mask 就推定所有层走同一 kernel。

## 五、DeepSeek-V4 的模型与后端限制

再看 DeepSeek-V4 在固定版本中的支持范围。

当前模型级 hook：

[`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/deepseek_v4_hook.py)

明确：

~~~python
if cfg.speculative_algorithm == "EAGLE":
    assert cfg.speculative_eagle_topk == 1
~~~

模型级检查与后端准入是独立约束。即使只看运行时是否存在通用 Tree Builder，也不能据此推定 V4 的 Attention、压缩状态和提交路径都能处理多分支。

### 模型配置层的 topk=1 约束

这个 guard 在 runtime 进入 EAGLE Tree Verify 前就拒绝。

它限定的是这条模型配置对外开放的能力：当前只接受 Chain，而不是等某个 kernel 执行失败后才局部回退。

### 第二层：CUDA/HIP DSV4 Attention 本身也明确拒绝 Tree

CUDA：

[`DeepseekV4AttnBackend`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py)

初始化直接：

~~~python
self.topk = get_spec().speculative_eagle_topk or 0

assert self.topk in [0, 1], (
    "MTP Topk > 1 not supported for DeepSeek V4"
)
~~~

HIP radix backend 也有同样限制。

两个后端都在初始化阶段拒绝多分支，说明 DSV4 Attention 本身也是固定版本中可确认的阻塞点。仅处理模型参数检查，不会让下游自动具备树可见性。

### 多分支支持还依赖哪些组件

DeepSeek-V4 的 Target Attention 不只是普通 full KV。

还涉及：

~~~text
SWA
C4
C128
DSA / Indexer
Compressor state
paged KV ownership
speculative KV commit
~~~

如果未来允许：`topk > 1`

每个 candidate node 都必须保证：

~~~text
Candidate identity
        │
        ├─ position
        ├─ SWA visible ancestors
        ├─ compressed-KV visible ancestors
        ├─ indexer candidate history
        └─ post-accept commit mapping
~~~

都属于同一条 root-to-node path。

DeepSeek-V4 的多分支能力必须把这些状态一起对齐。Attention backend 是已确认的限制之一；压缩 KV、Indexer 和 compressor state 是否沿同一祖先路径生成、接受后怎样整理，也都需要各自的实现与验证，不能用通用 Tree Builder 的存在替代。

### DSpark 的 `topk=1` 又是另一层问题

DSpark 在：

[`speculative_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/speculative_hook.py)

会把：

~~~text
speculative_num_steps = 1
speculative_eagle_topk = 1
~~~

固定下来。

而 DSpark Worker 当前也明确依赖：`Chain layout only`

这属于 DSpark 算法/runtime layout 自己的 chain contract。

DSpark 的 Chain layout 是自身提议与提交实现的契约，V4 EAGLE 的限制则来自模型配置与后端能力。两者在参数上都出现 topk=1，扩展时仍要分别分析，不能认为解除其中一项就同时解决另一条路径。

## 六、检查 Tree Verify 的五项一致性

排查 Tree Verify 时，可以按下面五项依次核对。它们把同一棵候选树连接到输入行、位置、Attention、Accept 和缓存；后端选择哪种表示，不改变这些对应关系。

### Candidate Identity

对于 Verify row `i`：`draft_token[i]`

必须就是这个 row 所代表的 candidate node。

### Position Identity

对于 node `v`：

~~~text
position(v)
=
KVReadyPrefixLen + depth(v)
~~~

Sibling：

~~~text
same depth
→ same position
~~~

### Visibility Identity

对于 node `v`：

~~~text
visible(v)
=
KV-ready committed prefix
∪
ancestors(v)
∪
{v}
~~~

这个公式先描述非 SWA 层：KV-ready Prefix 不包含尚未执行的本轮 root，root 作为本行自身或祖先节点加入；SWA 层还需按相应 position 与窗口规则限制可见范围。不能把“已输出的 bonus”直接当成已缓存历史列。

### Traversal Identity

~~~text
retrieve_next_token
retrieve_next_sibling
retrieve_index
~~~

必须描述和 Attention Visibility 相同的 Tree。

否则可能出现：

~~~text
Attention 按 Tree A 算 logits

Accept 按 Tree B 遍历
~~~

这类错误可能不报异常，却已经用错误上下文完成了接受决策。

### KV / Row Identity

同一个 Verify row `i` 的：

~~~text
draft_token[i]
positions[i]
tree visibility[i]
retrieve_index[i]
out_cache_loc[i]
~~~

必须共同指向同一个 candidate identity。

可以把整个 contract 画成：

~~~mermaid
flowchart TD
    A[Draft Expansion Pool] --> B[Candidate Selection]
    B --> C[Compact Verify Rows]

    C --> D[Token Identity]
    C --> E[Parent Topology]

    E --> F[Tree Position]
    E --> G[Attention Visibility]
    E --> H[Retrieve Links]

    D --> I[EagleVerifyInput]
    F --> I
    G --> I
    H --> I

    I --> J[Target Verify Attention]

    J --> K{Backend Representation}
    K --> L[Direct Custom Mask]
    K --> M[Per-query KV Page Table]
    K --> N[Prefix/Suffix Cascade]
    K --> O[Topology Metadata]

    L --> P[Target Logits per Verify Row]
    M --> P
    N --> P
    O --> P

    P --> Q[Tree Traversal / Accept]
    H --> Q

    Q --> R[accept_index / accept_lens]
    R --> S[KV Commit / Compact]
    S --> T[New KV-ready Boundary]
    T --> U[Terminal Bonus = Next Round Root]
~~~

图中五类状态分别在对应阶段保持一致：

~~~text
Candidate Tree
=
Position Tree
=
Visibility Tree
=
Traversal Tree
=
KV Identity
~~~

调试时可以选中一个 Verify row，沿这五类信息追到底：它的 token 属于哪个节点、position 是否等于 Prefix 长度加深度、Attention 是否只见合法历史、Accept 链接是否指向同一父子关系，以及最终 KV 写到了哪里。这样才能区分错误来自输入重排、可见性表示还是后续提交。

### 固定源码入口

| 目标 | 固定版本源码 |
| --- | --- |
| EAGLE Draft 多步候选生成 | [`eagle_worker_v2.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_v2.py) |
| Candidate path-score / parent 选择 | [`spec_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/spec_utils.py) |
| Candidate compact / TreeMaskMode | [`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py) |
| Verify Input 构造 | [`eagle_worker_common.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_common.py) |
| Tree Mask / Position / Retrieve kernel | [`spec_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py) |
| EagleVerifyInput | [`eagle_info.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_info.py) |
| VerifyMask / FULL vs QLEN allocation | [`verify_mask.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/verify_mask.py) |
| Triton Direct Custom Mask | [`triton_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/triton_backend.py) |
| FlashAttention Tree page-table / Cascade | [`flashattention_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/flashattention_backend.py) |
| Tree Builder Golden Test | [`test_build_eagle_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/test/registered/spec/utils/test_build_eagle_tree.py) |
| DeepSeek-V4 model-level EAGLE guard | [`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/deepseek_v4_hook.py) |
| DeepSeek-V4 CUDA DSV4 backend guard | [`deepseek_v4_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py) |
| DSpark chain-only resolution | [`speculative_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/speculative_hook.py) |
