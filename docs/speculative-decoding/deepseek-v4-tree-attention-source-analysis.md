# EAGLE Tree Attention：候选布局、位置与可见性

把树放成 `[root, A, B, C, D]`，再套普通下三角，C 就会看到不属于自己的 B。假设 C 在 A 分支、D 在 B 分支，它们各自只能沿祖先读取上下文，数组中谁排在前面并不能决定历史。

Tree Attention 要使每行结果等价于单独执行自己的祖先路径。以下先对齐行号、position、可见性，再看 Mask、页表、Cascade 怎样表达它们。固定 DeepSeek-V4 EAGLE 仍为 `topk=1`，本文多分支部分解释的是通用机制。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 791c7850d0960fd768102f71e7d999b036bb75ba`，核对日期为 2026-09-20。范围包括 EAGLE Spec V2 Tree Builder、VerifyMask、Triton / FlashAttention Verify，以及 DSV4 capability gate。V4 模型要求 `speculative_eagle_topk == 1`，CUDA/HIP DSV4 还独立要求 `topk in [0, 1]`，具体分流见第五节。

</details>

## 一、把候选树放进 Verify Tensor

### 祖先关系与可见历史

先设已确认 Prefix 和候选树：

Prefix 为 `P0 P1 P2 ... Pn`。

Draft 提出：

~~~text
                 root
               /      \
              A        B
              |        |
              C        D
~~~

分别运行两条路径，可以作为结果的参照：

~~~text
Target(prefix + root + A + C)

Target(prefix + root + B + D)
~~~

单独运行能保持各自上下文，却会重复处理共享部分。合成一次 forward，需要在减少重复的同时保留相同路径语义。能省多少，还取决于行数、backend 和后续状态整理。

~~~text
一次 Target Forward

物理输入：
[root, A, B, C, D]

逻辑语义：
root → A → C

以及

root → B → D
~~~

对非滑窗层，分支共享确认 Prefix、root，同深度节点对应互斥未来。B 即使排在 A 后面，也不把 A 当作历史。SWA 还要按 Query 限制窗口，第四节讨论这一区别。

<a id="causal-mask"></a>

#### causal 下三角混入其他分支

输入行序为：

~~~text
row 0 = root
row 1 = A
row 2 = B
row 3 = C
row 4 = D
~~~

普通下三角形成：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 1 1 0 0
  3       1 1 1 1 0
  4       1 1 1 1 1
~~~

因此出现以下读取：

~~~text
B 看见 A
C 看见 B
D 看见 A/B/C
~~~

其中混入了 sibling 或另一分支。

合法的候选可见性应为：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 0 1 0 0
  3       1 1 0 1 0
  4       1 0 1 0 1
~~~

非滑窗层的 M(i,j)=1，取决于以下条件：

~~~text
j 属于 committed prefix

或者

j == i

或者

j 是 i 的 ancestor
~~~

其他位置取 M(i,j)=0。

物理行序与逻辑祖先由此分开：

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

<a id="accept-correctness"></a>

#### Accept 使用的是这些上下文产生的 logits

节点 v 要基于自己的 root-to-v 历史预测：

~~~text
正确：L(v) = Target(prefix + path(root → v))
错误：L(v) = Target(prefix + physical_rows_through_v)
~~~

后一式按物理行序读取到 v，可能包含其他分支。这样的 logits 已来自错误上下文，`retrieve_next_token`、`retrieve_next_sibling`、`verify_tree_greedy` 即使遍历正确，也只是在错误预测中选择，无法重新计算 Attention。

### 候选池到 Verify 行

每个 Verify row 的候选身份，先由搜索、筛选确定：

~~~text
每一个 Verify row
到底是哪一个 candidate node
~~~

Draft 多步生成入口为：

[`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_v2.py)

每步保存：

~~~text
score_list
token_list
parents_list
~~~

后续评分经 [`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/spec_utils.py) 沿路径累计。

计算为：

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

搜索池中的节点再经过裁剪，部分进入 Target。

`organize_draft_results()` 整理为：

~~~text
1. flatten score pool
2. top-k 选出 num_draft_token - 1 个 candidate
3. sort selected indices
4. gather 真正进入 Verify 的 draft tokens
5. 保留 parent topology
~~~

实现为 [`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)。

搜索池与 Verify 使用各自的索引空间：

~~~text
top_scores_index / selected_index
        │
        └─ Draft expansion candidate-pool space

retrieve_index
        │
        └─ Compact Target-Verify row space
~~~

候选选择、Tree Builder 将两者连接：

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

这里 compact 指从搜索池筛出的 Verify 树节点数组。DSpark ragged 是另一种执行模式。多请求的 `retrieve_index` 还带有全局行偏移，后续寻址需要保留请求位置。

<a id="verify-root-draft-candidate"></a>

#### Verify 首行放旧 bonus

[`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py) 先完成这次拼接：

~~~python
draft_tokens = torch.cat(
    (
        bonus_tokens.unsqueeze(1),
        draft_tokens,
    ),
    dim=1,
).flatten()
~~~

窗口成为 `[上一轮 bonus/root, selected draft candidates...]`。root 占首行，计算自己的状态，并给出首层候选的验证预测，因此也参与 position、KV 边界安排。

## 二、位置编号与跨轮 KV 边界

### 同深度节点共享时间坐标

Mask 表示可以读取谁，position 表示节点在路径上的时间位置。两者都来自候选树。

还是这棵树：

~~~text
root
├─ A
│  └─ C
└─ B
   └─ D
~~~

设 KV-ready Prefix 长度为 `P`。

按深度分配位置：

~~~text
root → P

A    → P + 1
B    → P + 1

C    → P + 2
D    → P + 2
~~~

若直接用连续 row 编号，就会变成：

~~~text
root → P
A    → P+1
B    → P+2
C    → P+3
D    → P+4
~~~

A、B 都紧跟 root，position 同为 P+1；C、D 同为 P+2。连续行号区分存储位置，逻辑深度则保留不同分支处于同一时间步的关系。

~~~text
same tree depth
=
same logical position
~~~

<a id="depth-position"></a>

#### Tree Builder 按 depth 写 position

root 的位置设置为：

~~~python
positions[root] = seq_len
~~~

其余节点沿 parent 追到深度，再计算：

~~~text
position
=
seq_len + tree_depth
~~~

实现见 [`spec_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py)。

源码注释给出的例子为：

~~~text
depth:
[0, 1, 1, 2]

prefix length:
7

positions:
[7, 8, 8, 9]
~~~

[`test_build_eagle_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/test/registered/spec/utils/test_build_eagle_tree.py) 还包含更长树的 positions 断言。

其中首请求为：

~~~text
positions =
[5, 6, 6, 7, 7, 8, 8, 9]
~~~

结合父节点关系：

~~~text
retrieve_next_token =
[1, 3, 4, 5, 6, 7, -1, -1]

retrieve_next_sibling =
[-1, 2, -1, -1, -1, -1, -1, -1]
~~~

可以恢复成：

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

<a id="mask-position"></a>

#### position 错误怎样传到预测

即便 B 只读 `prefix + root + B`，position 也可能设置错误。

正确位置为 `P + 1`，若误设成 `P + 2`：

RoPE 相对旋转随之变化，Attention score、hidden、logits 也会受到影响。它未必每次翻转 argmax，但已经不再保证与原路径等价。传播过程为：

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

因此 Mask 保证可见对象正确，position 还要保证它们的位置关系正确。

### 旧 root 与新 bonus

树的起始位置来自跨轮 KV 边界。旧 terminal bonus R 已经输出，但尚未作为 Target 输入执行，所以此时还不属于 KV Prefix。

<a id="batchseq_lens-kv-ready-boundary"></a>

paged-tree 的 `prepare_for_draft()` 将 `batch.seq_lens` 解释为此时 KV-ready 的前缀边界，bonus 将作为 Verify root 写入。若 `batch.seq_lens=P`，已有 KV 覆盖 `[0,P)`，R 应取 `position(R)=P`；改从 P+1 写会跳过这个位置。[`prepare_for_draft()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_common.py)

Verify 分配也从 `start_offset=batch.seq_lens` 开始，为这轮输入安排写入位置。[`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

<a id="accept_lens-kv-ready-boundary"></a>

#### 输入状态与输出预测相差一位

假设 R 后提出 D1、D2、D3，全部接受，且本轮未被 EOS、stop、长度限制截断。每行输入和预测对应为：

| Verify 行 | 当前输入 | position | 该行预测 |
| --- | --- | --- | --- |
| row0 | R | P | D1 |
| row1 | D1 | P+1 | D2 |
| row2 | D2 | P+2 | D3 |
| row3 | D3 | P+3 | 新 bonus B |

`num_correct_drafts=3`，因此 `accept_lens=num_correct_drafts+1=4`。输出 `[D1,D2,D3,B]`，可提交的输入 KV 为 `[R,D1,D2,D3]`。两边都推进 4 个位置，所以 `new_seq_lens=batch.seq_lens+accept_lens=P+4`。

新增 KV 覆盖到 D3；B 在 P+4，留到下一轮做 root。KV relocation 搬移的是本轮已执行输入，Compact 也要让它们和各自产生的预测保持配对。日志带上轮次、输入/输出侧，才能区分同一 ID 的 bonus、root 角色。

## 三、FULL_MASK 与 QLEN_ONLY 怎样表达可见性

FULL_MASK 与 QLEN_ONLY 以不同布局表达同一祖先关系。

枚举定义为：

~~~python
class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2
~~~

位置见 [`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)。

### FULL_MASK：显式包含 Prefix + Tree Block

单请求的 FULL_MASK 近似为：

~~~text
[num_verify_tokens,
 seq_len + num_verify_tokens]
~~~

左边放历史，右边放候选：

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

总大小约为 `B × N × (L + N)`。

B 为请求数，N 为窗口行数，L 为 Prefix 长度。以 B=64、N=64、L=32768、bool 每元素 1 Byte 估算，共 134479872 元素，约 128.25 MiB；QLEN_ONLY 的 `64×64²` 只有 262144 元素，即 256 KiB。这比较 Mask 存储，不含 KV，也没有测量耗时。

`verify_mask.py` 的注释同样指出 `FULL_MASK reaches 100s of MB at long context`。

### QLEN_ONLY：只物化 Candidate × Candidate Tree Block

QLEN_ONLY 只保存 `[N, N]` 候选块。

对应本例：

~~~text
        0 1 2 3 4

0       1 0 0 0 0
1       1 1 0 0 0
2       1 0 1 0 0
3       1 1 0 1 0
4       1 0 1 0 1
~~~

后端仍然读取合法 Prefix，减少的是历史 mask 的重复表示。

Mask 存储复杂度因而从 `O(B × N × (L + N))` 降至 `O(B × N²)`。

### 各后端如何消费 FULL 与 QLEN

[`VerifyMask`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/verify_mask.py) 进一步约定 backend 怎样使用 buffer。

除了 buffer、mode、max_bs，它还保存 `is_read`。

`is_read` 表示 Attention 是否读取这块 Mask。统一 Builder 即使只需要提供 positions、retrieve links，也可能继续写固定 buffer，所以写入 Mask 与消费 Mask 是两件事。

Graph buffer 常按 `max_bs`、`max_context_len` 预分配。估算静态空间时要使用这些上限，而非当前请求长度。

例如 [`DeepseekV4AttnBackend.init_cuda_graph_state()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py) 的注释为 `Verify metadata never extracts the mask.`。

相应设置为：

~~~python
maybe_create_verify_mask(
    ...,
    is_read=False,
)
~~~

创建逻辑在 `is_read=False` 时选择较小的 `QLEN_ONLY`。

当前 DSV4 topk=1 metadata 不读这块 Mask。其他树 backend 可以通过 FULL、QLEN、页表、topology 表达祖先关系；非 SWA 的可见范围仍是合法 Prefix 加分支后缀。

<a id="triton-direct-mask"></a>

### Triton 直接消费 custom_mask

Triton Target Verify 从 `spec_info.custom_mask` 取得 Mask，将它与 `mask_indptr`、`qo_indptr`、`kv_indptr`、`kv_indices` 一起交给 `extend_attention_fwd()`。[`triton_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/triton_backend.py)

这条读取链为：

~~~text
Tree
 ↓
FULL_MASK
 ↓
Attention Kernel
~~~

<a id="hybrid-backend-tree-topology-bool-mask"></a>

### Linear / Hybrid backend 传递 topology

有些线性 / hybrid backend 使用 topology，例如 Ascend hybrid linear 的 topk>1 Verify：

~~~text
retrieve_next_token
retrieve_next_sibling
retrieve_parent_token
~~~

这种表示属于对应 backend。它并没有解除 V4 DSV4 的模型限制。[Ascend hybrid linear backend](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py)

~~~text
Tree Attention semantics
≠
必须存在一张 bool Tree Mask
~~~

## 四、FlashAttention Cascade 怎样合并结果

FlashAttention 则把历史、候选后缀拆开计算，再合并结果。

满足这些条件时：

~~~python
forward_batch.forward_mode.is_target_verify()
and self.topk > 1
and not is_swa_layer
~~~

代码选择：

~~~python
use_cascade_attn = True
~~~

注释称其为 `We do cascade attention for Target Verify with topk > 1`。[`flashattention_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/flashattention_backend.py)

共享 Prefix 和各行合法祖先后缀分成两段 Attention，设备核因此无需直接理解整棵树。

### 第一部分：所有 Query 共享的 Common Prefix

`target_verify_metadata_topk_normal` 安排各 Query 读取共同 committed Prefix。

这部分历史由所有节点共享，无需为每条分支重复保存完整 mask。

### 第二部分：每个 Query 自己的 Ancestor Suffix

候选部分从 `custom_mask` 取出：

~~~python
mask =
    spec_info.custom_mask[
        mask_extraction_indices
    ].view(
        -1,
        speculative_num_draft_tokens,
    )
~~~

将 `mask == True` 对应的物理 slots 放在各 Query 后缀页表前部。

源码用这组数据说明：

candidate slots 为 `[8, 9, 10]`，mask 为：

~~~text
query0 [1, 0, 0]
query1 [1, 1, 0]
query2 [1, 0, 1]
~~~

整理后：

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

query2 的合法 slots 是 `[8,10]`。筛选以后，Kernel 仍需受有效长度限制；页表尾部即使保留了地址，也不能读入无效的 9。

转换关系为：

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

两段分别返回 `(O_prefix, LSE_prefix)`、`(O_tree, LSE_tree)`。

随后调用：

~~~python
merge_state_v2_wrapper(...)
~~~

每段输出都已单独 softmax。设未归一化总权重为 `Z_prefix`、`Z_tree`，合并应为 `O=(Z_prefix×O_prefix+Z_tree×O_tree)/(Z_prefix+Z_tree)`，LSE 保存的 log-sum-exp 正用于恢复这组权重。两段还要采用相同 Query、score 定义，并避免重复计入重叠 KV。

例如两段 Z 分别为 3、1，输出为 2、10，合并结果是 `(3×2+1×10)/4=4`。直接平均会得到 6，丢掉了两段不同的归一化权重。

使用 Softmax State Merge 后，等价于：

~~~text
Attention(
    query,
    committed prefix
    ∪
    allowed ancestor suffix
)
~~~

合法 Prefix、Suffix 分别计算，再恢复完整 softmax 的结果。

<a id="swa"></a>

### SWA 按各 Query 的窗口取历史

SWA 的注释为 `We don't use cascade attention for Sliding Window Attention`。

原因在于：

~~~text
不同 query 需要不同 window size，
而 FA3 cascade interface 不能传一组不同 window sizes。
~~~

因此 SWA 使用 expanded spec metadata，绕开这条共同 Prefix Cascade。

两类层分别选择：

~~~text
FlashAttention 非-SWA topk>1 Target Verify
→ cascade Tree Attention

FlashAttention SWA topk>1
→ 另一套 expanded metadata / page-table 路径
~~~

非 SWA 可以共享全部 Prefix，SWA 的合法历史还随 Query position 和窗口变化。历史范围不同，无法套用相同的共同 Prefix 划分。

## 五、DeepSeek-V4 的模型与后端限制

固定 V4 的多分支准入，在模型与 backend 各有一层检查。

模型 hook 先判断：[`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/deepseek_v4_hook.py)

~~~python
if cfg.speculative_algorithm == "EAGLE":
    assert cfg.speculative_eagle_topk == 1
~~~

模型检查发生在进入 Tree Verify 之前，通用 Builder 后面还要接 V4 Attention、压缩状态和提交路径。

### 模型配置层的 topk=1 约束

它限定模型配置为 Chain。

多分支会在 kernel 执行前被拒绝。

<a id="cudahip-dsv4-attention-tree"></a>

### CUDA/HIP DSV4 另有限制

CUDA DSV4 在初始化时再次检查：[`DeepseekV4AttnBackend`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py)

~~~python
self.topk = get_spec().speculative_eagle_topk or 0

assert self.topk in [0, 1], (
    "MTP Topk > 1 not supported for DeepSeek V4"
)
~~~

HIP radix backend 也有相同限制。

只放开模型参数，仍会在 backend 初始化遇到限制，树可见性也没有因此完成适配。

### 多分支支持还依赖哪些组件

V4 的历史状态还包含：

~~~text
SWA
C4
C128
DSA / Indexer
Compressor state
paged KV ownership
speculative KV commit
~~~

扩展 `topk > 1` 时，这些状态都要对齐：

~~~text
Candidate identity
        │
        ├─ position
        ├─ SWA visible ancestors
        ├─ compressed-KV visible ancestors
        ├─ indexer candidate history
        └─ post-accept commit mapping
~~~

对应同一条 root-to-node 路径。

除了已确认的 backend 限制，压缩历史、Indexer、Compressor、接受后整理也要实现并验证各自的分支语义。

### DSpark 的 `topk=1` 又是另一层问题

DSpark 在参数处理中单独固定这些值，见 [`speculative_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/speculative_hook.py)。

~~~text
speculative_num_steps = 1
speculative_eagle_topk = 1
~~~

DSpark Worker 还要求 `Chain layout only`，由自己的提议、提交契约限定布局。

因此 DSpark 链布局、V4 EAGLE 模型约束、DSV4 backend 限制虽然都表现为 topk=1，却来自不同执行层。

<a id="tree-verify"></a>

## 六、沿同一 Verify row 核对数据

选定一个节点，可以把输入、位置、Attention、Accept、KV 放在同一张表中：

<a id="candidate-identity"></a>
<a id="position-identity"></a>
<a id="visibility-identity"></a>
<a id="traversal-identity"></a>
<a id="kv-row-identity"></a>

| 检查项 | 同一节点应满足的关系 |
| --- | --- |
| 候选身份 | Verify row i 的 `draft_token[i]` 就是该行代表的节点 |
| 位置 | `position(v)=KVReadyPrefixLen+depth(v)`；同深度 sibling 使用相同 position |
| 可见性 | 非 SWA 层读取 KV-ready Prefix、`ancestors(v)` 与 `{v}`；SWA 再叠加位置与窗口限制 |
| 接受遍历 | `retrieve_next_token`、`retrieve_next_sibling`、`retrieve_index` 描述与 Attention 相同的树 |
| KV / row 对应 | `draft_token[i]`、`positions[i]`、`tree visibility[i]`、`retrieve_index[i]`、`out_cache_loc[i]` 都指向同一候选 |

表中的 Prefix 尚未包含未执行的 root，它应作为当前节点或祖先加入。Attention 使用一棵树、Accept 遍历另一棵树时，可能全程没有报错，却在错误上下文中确认了候选。

数据关系为：

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

各 backend 的表示不同，这些对象仍要指向同一个候选及其 root-to-node 路径。

逐项对齐这行的 Token、Prefix 加深度的位置、合法祖先、遍历链接、写入目标，才能将输入重排错误、历史可见性错误和提交错误分开定位。

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
