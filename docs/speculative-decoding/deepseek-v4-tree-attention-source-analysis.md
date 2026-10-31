# EAGLE Tree Attention：候选布局、位置与可见性

将树按 `[root, A, B, C, D]` 放进 Tensor 后，物理顺序并不会替代父子关系。若 C 属于 A、D 属于 B，那么 C 可以读 root、A，却不能将 B 当成自己的历史。

Tree Attention 要让每行等价于沿自己的祖先路径执行。下面先将小树的行号、position、可见性对应，再看 Mask、页表、Cascade 如何实现。固定 DeepSeek-V4 EAGLE 仍限定 `topk=1`，通用多分支机制并不等于 V4 已支持相同部署。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 791c7850d0960fd768102f71e7d999b036bb75ba`，核对日期为 2026-09-20，覆盖 EAGLE Spec V2 Tree Builder、VerifyMask、Triton / FlashAttention Verify 和 DSV4 capability gate。
>
> 固定 V4 模型要求 `speculative_eagle_topk == 1`，CUDA/HIP DSV4 也独立要求 `topk in [0, 1]`。多分支部分用于解释通用 EAGLE 语义，以及未来扩展 V4 时需满足的数据关系，不能视为当前生产路径已经使用。

</details>

## 一、把候选树放进 Verify Tensor

### 祖先关系与可见历史

先固定一份历史和一棵候选树。

已确认 Prefix 为 `P0 P1 P2 ... Pn`。

Draft 提出：

~~~text
                 root
               /      \
              A        B
              |        |
              C        D
~~~

可以先分别执行两条路径，作为语义参照：

~~~text
Target(prefix + root + A + C)

Target(prefix + root + B + D)
~~~

分别运行能得到各自上下文的预测，但会重复共享部分。合成一次 forward 的目标是在保留相同路径语义的同时减少重复，实际收益还取决于行数、backend 和状态整理。

~~~text
一次 Target Forward

物理输入：
[root, A, B, C, D]

逻辑语义：
root → A → C

以及

root → B → D
~~~

非滑窗场景下，各分支共享已确认 Prefix、root，同深度节点却互斥。B 排在 A 后面，不表示 A 属于 B 的历史。SWA 还要限制窗口，后文另看。

#### 普通 causal mask 为什么会错

按以下行序输入：

~~~text
row 0 = root
row 1 = A
row 2 = B
row 3 = C
row 4 = D
~~~

普通下三角会给出：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 1 1 0 0
  3       1 1 1 1 0
  4       1 1 1 1 1
~~~

从而错误地产生：

~~~text
B 看见 A
C 看见 B
D 看见 A/B/C
~~~

这些读取混入了 sibling 或另一分支。

真正需要的候选可见性为：

~~~text
          K
          0 1 2 3 4

Q 0       1 0 0 0 0
  1       1 1 0 0 0
  2       1 0 1 0 0
  3       1 1 0 1 0
  4       1 0 1 0 1
~~~

对非滑窗层，可写为 M(i,j)=1 当且仅当满足以下之一：





~~~text
j 属于 committed prefix

或者

j == i

或者

j 是 i 的 ancestor
~~~

其余为 M(i,j)=0。



因此物理行序和逻辑祖先关系需分别看：

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

节点 v 的 logits 必须来自自己的 root-to-v 历史，而不是按物理行序收下此前所有节点：

~~~text
正确：L(v) = Target(prefix + path(root → v))
错误：L(v) = Target(prefix + physical_rows_through_v)
~~~

后一式表示普通 causal mask 让 v 读取物理行序中直到自身的内容，其中可能混入 sibling 或其他分支。若上下文在 Attention 中已经算错，后续 `retrieve_next_token`、`retrieve_next_sibling`、`verify_tree_greedy` 即使遍历正确，也只能从这些错误 logits 中选路径，无法重算历史。因此要先核可见性，再查接受链接。

### 候选池到 Verify 行

生成 Mask 前，先确定每个 Verify row 代表哪个候选：

~~~text
每一个 Verify row
到底是哪一个 candidate node
~~~

Draft 多步入口为：

[`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_v2.py)

过程持续保存：

~~~text
score_list
token_list
parents_list
~~~

后续评分沿路径累计，具体计算见 [`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/spec_utils.py)。





其计算为：

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

这些展开节点还会裁剪，只有部分进入 Target。

`organize_draft_results()` 执行下列整理：

~~~text
1. flatten score pool
2. top-k 选出 num_draft_token - 1 个 candidate
3. sort selected indices
4. gather 真正进入 Verify 的 draft tokens
5. 保留 parent topology
~~~



实现见 [`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)。

因此有两套索引空间：

~~~text
top_scores_index / selected_index
        │
        └─ Draft expansion candidate-pool space

retrieve_index
        │
        └─ Compact Target-Verify row space
~~~

它们通过候选选择、Tree Builder 连接：

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

这里 compact 表示从搜索池选出的 Verify 树节点数组，和 DSpark ragged 执行并非同一模式。多请求时 `retrieve_index` 还包含请求的全局行偏移，不能全部按每请求从 0 编号使用。

#### Verify root 不是普通 draft candidate

构建窗口先执行以下拼接，入口见 [`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)。





~~~python
draft_tokens = torch.cat(
    (
        bonus_tokens.unsqueeze(1),
        draft_tokens,
    ),
    dim=1,
).flatten()
~~~

窗口为 `[上一轮 bonus/root, selected draft candidates...]`。root 占第一行，虽然不是新候选，仍需计算自身状态并提供检查首层候选的预测，因此也影响 position、KV 边界。

## 二、位置编号与跨轮 KV 边界

### 同深度节点共享时间坐标

Mask 限制能看哪些对象，position 表示该行在序列中的时间位置，二者都需要从同一棵树得到。

仍看：

~~~text
root
├─ A
│  └─ C
└─ B
   └─ D
~~~

设 KV-ready Prefix 长度为 `P`。

按树深度分配为：

~~~text
root → P

A    → P + 1
B    → P + 1

C    → P + 2
D    → P + 2
~~~

若误用连续 row 编号，则变成：

~~~text
root → P
A    → P+1
B    → P+2
C    → P+3
D    → P+4
~~~

A、B 都在 root 后一步，所以都是 P+1；C、D 都为 P+2。Tensor 行区分物理位置，position 表示逻辑深度，混用会把互斥分支排成假历史。

~~~text
same tree depth
=
same logical position
~~~

#### 源码确实按 depth 算 Position

kernel 为 root 写入：

~~~python
positions[root] = seq_len
~~~

其他节点沿 parent 追溯深度，得到：

~~~text
position
=
seq_len + tree_depth
~~~



固定实现见 [`spec_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py)。

注释中的例子为：

~~~text
depth:
[0, 1, 1, 2]

prefix length:
7

positions:
[7, 8, 8, 9]
~~~

更长树的测试也给出预期 positions，见 [`test_build_eagle_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/test/registered/spec/utils/test_build_eagle_tree.py)。



其中首请求为：

~~~text
positions =
[5, 6, 6, 7, 7, 8, 8, 9]
~~~

结合：

~~~text
retrieve_next_token =
[1, 3, 4, 5, 6, 7, -1, -1]

retrieve_next_sibling =
[-1, 2, -1, -1, -1, -1, -1, -1]
~~~

可恢复为：

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

即使 B 已只看 `prefix + root + B`，仍可能错在 position。

例如误设 `P + 2`，正确应是 `P + 1`。



RoPE 位置改变会影响与历史的相对旋转，进一步改变 Attention score、hidden 和 logits。不一定每次都翻转 argmax，但已无法由正确 Mask 单独保证等价。错误传播可画为：

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

因此可见性与位置需同时对齐。

### 旧 root 与新 bonus

跨轮边界决定这组位置从哪里开始。旧 terminal bonus R 虽然已经输出，却还没有作为 Target 输入执行，不能提前算进 KV Prefix。

<a id="batchseq_lens-kv-ready-boundary"></a>

paged-tree 的 `prepare_for_draft()` 将 `batch.seq_lens` 解释为此时 KV-ready 的前缀边界，bonus 将作为 Verify root 写入。若 `batch.seq_lens=P`，已有 KV 覆盖 `[0,P)`，R 应取 `position(R)=P`；改从 P+1 写会跳过这个位置。[`prepare_for_draft()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_common.py)

Verify 分配同样以 `start_offset=batch.seq_lens` 开始，为本轮输入安排写入目标。[`eagle_prepare_for_verify()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

<a id="accept_lens-kv-ready-boundary"></a>

#### 将位置编号与候选行对应起来

假设请求尚未被 EOS、stop 或长度截断结束，R 后提出 D1、D2、D3，并且全部接受。每行的输入与下一位置预测如下：

| Verify 行 | 当前输入 | position | 该行预测 |
| --- | --- | --- | --- |
| row0 | R | P | D1 |
| row1 | D1 | P+1 | D2 |
| row2 | D2 | P+2 | D3 |
| row3 | D3 | P+3 | 新 bonus B |

此时 `num_correct_drafts=3`，`accept_lens=num_correct_drafts+1=4`。本轮新输出是 `[D1,D2,D3,B]`，可提交输入 KV 却是 `[R,D1,D2,D3]`。二者长度都为 4，所以 `new_seq_lens=batch.seq_lens+accept_lens=P+4`，数量相同并不代表 Token 身份相同。

新增 KV 覆盖到 D3，新 B 位于 P+4，还要等下一轮作为 root 执行。bonus、root 是同一 ID 在相邻轮次的不同角色，日志必须带上轮次及输入/输出侧。KV relocation 处理 `accept_lens` 行时，搬的是这些已执行输入；后续 Compact 也必须保持它们与输出预测的错位对应。

## 三、FULL_MASK 与 QLEN_ONLY 怎样表达可见性

下面两种 Mask 模式只改变表示布局，都需表达相同祖先规则。

固定枚举为：

~~~python
class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2
~~~



定义见 [`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)。

### FULL_MASK：显式包含 Prefix + Tree Block

单请求 FULL_MASK 近似为：

~~~text
[num_verify_tokens,
 seq_len + num_verify_tokens]
~~~

包括左侧历史、右侧候选：

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

B 为请求数、N 为窗口行数、L 为 Prefix 长度。教学设定 B=64、N=64、L=32768，bool 每元素 1 Byte 时，共 134479872 元素、约 128.25 MiB；QLEN_ONLY 的 `64×64²` 为 262144 元素、256 KiB。这是 Mask 存储估算，并非 KV 或耗时实测。

`verify_mask.py` 也注明 `FULL_MASK reaches 100s of MB at long context`。

### QLEN_ONLY：只物化 Candidate × Candidate Tree Block

QLEN_ONLY 仅存 `[N, N]` 候选块。

本例为：

~~~text
        0 1 2 3 4

0       1 0 0 0 0
1       1 1 0 0 0
2       1 0 1 0 0
3       1 1 0 1 0
4       1 0 1 0 1
~~~

Prefix 仍要由后端正确读取，省掉的是重复历史 mask 表示，不是历史 Attention 工作。

因此存储复杂度由 `O(B × N × (L + N))` 降为 `O(B × N²)`。



### 各后端如何消费 FULL 与 QLEN

后端还通过 VerifyMask 描述如何使用 buffer。[`VerifyMask`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/verify_mask.py)



这份对象除 buffer、mode、max_bs 外，还保存 `is_read`。

`is_read` 表示是否实际读取这块 Mask。统一 Builder 还需产生 positions、retrieve links，因而即使后端不读，也可能保留写入 buffer。分配或写入过，并不足以证明 Attention 据此读取历史。

Graph buffer 常按 `max_bs`、`max_context_len` 预分配，算静态空间要用分配上限，不能只代当前请求长度。

例如 CUDA `DeepseekV4AttnBackend` 注明 `Verify metadata never extracts the mask.`。[`DeepseekV4AttnBackend.init_cuda_graph_state()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py)

并设置：

~~~python
maybe_create_verify_mask(
    ...,
    is_read=False,
)
~~~





`is_read=False` 时，创建逻辑选择较小的 `QLEN_ONLY`。

当前 DSV4 topk=1 的 metadata 不读这块 Mask。其他树 backend 可通过 FULL、QLEN、页表或 topology 表达祖先关系；非 SWA 历史仍由可见 Prefix 加合法后缀组成。

### Triton 是最直观的 Direct-Mask 实现

Triton Target Verify 从 `spec_info.custom_mask` 取得 `custom_mask`，再将它和 `mask_indptr`、`qo_indptr`、`kv_indptr`、`kv_indices` 交给 `extend_attention_fwd()`。实现见 [`triton_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/triton_backend.py)。

这条路径可直接概括为：

~~~text
Tree
 ↓
FULL_MASK
 ↓
Attention Kernel
~~~

### 线性 / Hybrid backend 甚至可能传 Tree Topology 而不是 Bool Mask

有些线性 / hybrid backend 则传 topology，例如 Ascend hybrid linear 的 topk>1 Verify：

~~~text
retrieve_next_token
retrieve_next_sibling
retrieve_parent_token
~~~

这只说明通用 Runtime 有不同的树表示，不能用其他 backend 的支持解除 V4 DSV4 模型限制。[Ascend hybrid linear backend](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py)

~~~text
Tree Attention semantics
≠
必须存在一张 bool Tree Mask
~~~

## 四、FlashAttention Cascade 怎样合并结果

FlashAttention 又提供了另一种实现。

满足以下条件：

~~~python
forward_batch.forward_mode.is_target_verify()
and self.topk > 1
and not is_swa_layer
~~~

会选择：

~~~python
use_cascade_attn = True
~~~

相应注释为 `We do cascade attention for Target Verify with topk > 1`。[`flashattention_backend.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/flashattention_backend.py)





它把共同 Prefix 和每行合法祖先后缀拆成两段 Attention，再合并结果，不要求设备核直接理解整棵树。

### 第一部分：所有 Query 共享的 Common Prefix

`target_verify_metadata_topk_normal` 只安排查询读取共同 committed Prefix。



所有节点共享这部分历史，因而无需为每分支重复保存完整 mask。

### 第二部分：每个 Query 自己的 Ancestor Suffix

候选后缀则先从 `custom_mask` 取出：

~~~python
mask =
    spec_info.custom_mask[
        mask_extraction_indices
    ].view(
        -1,
        speculative_num_draft_tokens,
    )
~~~

将 `mask == True` 的物理 slots 排在该 Query 后缀页表前部。



源码举例为：

candidate slots `[8, 9, 10]`，对应 mask：



~~~text
query0 [1, 0, 0]
query1 [1, 1, 0]
query2 [1, 0, 1]
~~~

整理后为：

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

先筛合法 slots，再用有效长度约束 Kernel。query2 只可读 `[8,10]`，不能换成 `[8,9,10]`，页表尾部即使还有地址也不能越界消费。

转换过程为：

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

第一段返回 `(O_prefix, LSE_prefix)`，第二段返回 `(O_tree, LSE_tree)`。



通过下列入口合并：

~~~python
merge_state_v2_wrapper(...)
~~~



两段已各自 softmax，不能直接相加或平均。令未归一化总权重为 `Z_prefix`、`Z_tree`，则 `O=(Z_prefix×O_prefix+Z_tree×O_tree)/(Z_prefix+Z_tree)`。LSE 保存各段 log-sum-exp。两段需使用相同 Query、score 定义，并避免重复计算重叠 KV。

例如 Z 为 3、1，输出为 2、10，合并得到 `(3×2+1×10)/4=4`，并非平均值 6。LSE 保留各段归一化质量，才能恢复共同 softmax。

因此等价于：

~~~text
Attention(
    query,
    committed prefix
    ∪
    allowed ancestor suffix
)
~~~

合法 Prefix/Suffix 分开执行，再通过 Softmax State Merge 恢复完整结果。

### SWA 是一个重要例外

SWA 的注释则明确 `We don't use cascade attention for Sliding Window Attention`。

原因是：

~~~text
不同 query 需要不同 window size，
而 FA3 cascade interface 不能传一组不同 window sizes。
~~~

因此 SWA 改用 expanded spec metadata，不走这条共同 Prefix Cascade。

两类层分别处理为：

~~~text
FlashAttention 非-SWA topk>1 Target Verify
→ cascade Tree Attention

FlashAttention SWA topk>1
→ 另一套 expanded metadata / page-table 路径
~~~

非 SWA 可共享全部 Prefix，SWA 还受当前 Query position、窗口约束。不能用同一个共同 Prefix 划分替代不同层的合法可见范围。

## 五、DeepSeek-V4 的模型与后端限制

再回到固定 V4 能力。

模型 hook 直接判断：[`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/deepseek_v4_hook.py)





~~~python
if cfg.speculative_algorithm == "EAGLE":
    assert cfg.speculative_eagle_topk == 1
~~~

模型检查独立于后端准入。通用 Tree Builder 存在，不能证明 V4 Attention、压缩状态及提交都能承接。

### 模型配置层的 topk=1 约束

这项检查发生在进入 EAGLE Tree Verify 前。

它限定可使用的模型配置为 Chain，尚未执行 kernel 就会拒绝多分支。

### 第二层：CUDA/HIP DSV4 Attention 本身也明确拒绝 Tree

CUDA DSV4 后端也独立检查：[`DeepseekV4AttnBackend`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/layers/attention/deepseek_v4_backend.py)





~~~python
self.topk = get_spec().speculative_eagle_topk or 0

assert self.topk in [0, 1], (
    "MTP Topk > 1 not supported for DeepSeek V4"
)
~~~

HIP radix backend 同样限制。

所以仅放开模型参数，还会在 backend 初始化遇到限制，下游不会自动取得树可见性。

### 多分支支持还依赖哪些组件

V4 历史除了普通 KV，还包含：



~~~text
SWA
C4
C128
DSA / Indexer
Compressor state
paged KV ownership
speculative KV commit
~~~

将来扩展 `topk > 1` 时，需要使下列状态一致：



~~~text
Candidate identity
        │
        ├─ position
        ├─ SWA visible ancestors
        ├─ compressed-KV visible ancestors
        ├─ indexer candidate history
        └─ post-accept commit mapping
~~~

全部对应同一 root-to-node 路径。

已确认的 backend 限制只是其中一处；压缩历史、Indexer、Compressor 和接受后整理，也要有各自实现及验证，不能由通用 Builder 推断完成。

### DSpark 的 `topk=1` 又是另一层问题

DSpark 参数处理则另行固定以下值，见 [`speculative_hook.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/arg_groups/speculative_hook.py)。





~~~text
speculative_num_steps = 1
speculative_eagle_topk = 1
~~~



其 Worker 还要求 `Chain layout only`，属于 DSpark 自己的提议和提交契约。



因此 DSpark 链布局与 V4 EAGLE 模型、backend 限制应分别处理，即使都呈现 topk=1，也不是同一处开关。

## 六、检查 Tree Verify 的五项一致性

排查时可以按下面五项，确认同一节点贯穿输入、位置、Attention、Accept、KV。具体表示不同，也需满足这些关系。

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

这里的 Prefix 不含未执行的 root，它要作为当前节点或祖先加入。已输出 bonus 不能提前当作缓存 Prefix。若 Attention 按 Tree A 计算、Accept 却按 Tree B 遍历，可能没有报错，却已经在错误上下文完成接受。

整体对应关系如下：

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

这些结构的表示可以不同，但必须描述同一候选及其 root-to-node 路径。

选择一个 Verify row，核对 Token、Prefix 加深度的位置、可见祖先、遍历链接和最终写入目标。这样才能区分错误出在输入重排、历史可见性还是接受后的提交。

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
