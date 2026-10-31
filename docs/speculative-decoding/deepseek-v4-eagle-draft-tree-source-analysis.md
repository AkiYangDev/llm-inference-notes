# EAGLE 候选树：从候选池到 Verify 与接受路径

链式 Draft 把候选预算用在一条路径上，一处分歧就会使后续候选无法继续接受。通用 EAGLE 可以保留分支，把预算分给几条可能的未来，再由 Target 统一验证。

分支合在一次 forward 中，仍必须保持各自上下文：同深度节点可用同一个 position，却不能读到彼此的未来。下面沿一棵小树看搜索、索引、Mask 和提交。固定 DeepSeek-V4 仍要求 EAGLE `topk=1`，多分支用于解释通用能力，末节再核对模型、后端限制。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。正文分析 EAGLE Runtime 的 `topk>1`，以 DeepSeek-V4 / NextN 为背景。固定官方 Ascend DSV4 使用 `--attention-backend dsv4`、`page_size=128`、`speculative_eagle_topk=1`；V4 模型 hook 要求 EAGLE topk=1，另有 dsv4+page_size>1+topk>1 guard。通用树机制不能直接视为当前 V4 Ascend 已可运行的多分支配置。

</details>

## 一、在有限预算内保留候选路径

例如 NextN 提议 `A → B → C`，Target 接受 A 后却预测 X，那么 B、C 都不能提交。后面的 Token 即使偶然匹配，也无法越过这处分歧。

`topk>1` 允许保留同一上下文的多个候选。取 `topk=2`，假设 Extend 首次给出：

~~~text
A   p=0.60
B   p=0.40
~~~

`select_top_k_tokens()` 将两枚候选作为下一次 NextN 输入，并把父 hidden 复制到两个 branch。它们分别以 A、B 继续，但复制 hidden 尚未处理分支 KV 的分配、写入。关键片段及实现如下。[`_select_top_k_tokens_first()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L312-L330)

~~~python
input_ids = topk_index.flatten()

hidden_states = hidden_states.repeat_interleave(
    topk,
    dim=0,
)
~~~





假设第二层各得到：

~~~text
A 后：
C  0.70
D  0.30

B 后：
E  0.80
F  0.20
~~~

本轮为每个 frontier 提候选，再乘父路径 score。实现见 [`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L332-L366)。

~~~python
expand_scores = (
    scores.unsqueeze(2)
    * topk_p.view(-1, topk, topk)
)
~~~

由 `scores=[0.60, 0.40]` 得到：

~~~text
A → C : 0.60 × 0.70 = 0.42
A → D : 0.60 × 0.30 = 0.18
B → E : 0.40 × 0.80 = 0.32
B → F : 0.40 × 0.20 = 0.08
~~~

在 `topk × topk` 个 child paths 中，仅取 top-k 继续展开：

~~~python
topk_cs_p, topk_cs_index = fast_topk(
    expand_scores.flatten(start_dim=1),
    topk,
    dim=-1,
)
~~~

新的 frontier 为：

~~~text
A → C : 0.42
B → E : 0.32
~~~

下一轮模型只继续这两个节点，但本轮四个 child 的分数、Token 都进入候选池，最终裁剪还可使用它们。因此不再展开，并不等于不能参加 Verify。





在这条非 rejection-sampling 的 Tree 路径中，score 是 Draft 沿 frontier 递推的累计路径概率，不是 Target probability，也不是另一个 Confidence Head：

~~~text
score(child)
=
score(parent)
×
draft_probability(child | parent)
~~~

例如 B→E 的最后一跳 0.80 高于 A→C 的 0.70，乘父概率后却是 0.32 低于 0.42。累计 score 分配搜索与 Verify 预算，最后一跳排序不同，二者都不能代替 Target 接受。

每步 rollout 保存：

~~~python
score_list.append(tree_info[0])
token_list.append(tree_info[1])
parents_list.append(tree_info[2])
~~~

结束后再由 `organize_draft_results()` 全局裁剪：

~~~python
top_scores = torch.topk(
    score_list,
    num_draft_token - 1,
    dim=-1,
)
~~~

窗口先给旧 bonus/root 留 1 行，剩下 `num_draft_tokens - 1` 行才是候选。旧 bonus 已是输出，这轮仍需作为 Target 输入。

层内 top-k 限制继续计算的 frontier，全局 top-k 限制进入 Verify 的节点数。因此不会完整展开 k 叉树，topk 也不等于最终窗口节点数。

## 二、候选池索引怎样还原父节点关系

源码并未直接保留一张可视树图，需要从不同索引恢复父子关系。

候选整理返回：

~~~text
parent_list
top_scores_index
draft_tokens
~~~

`top_scores_index` 进入 Tree Builder 后称 `selected_index`，值仍是原始展平候选池的位置，不是压缩后 node ID。它先用于取 Token，再用于追溯父节点。

### 搜索候选池是怎样展平的

搜索池按 rollout 层次展平。第一层 A、B 放在 index 0、1；topk=2 时，下一层各 frontier 再提两个 child，AC、AD、BE、BF 放在 index 2、3、4、5。AC 只代表末端 C，父路径由 metadata 保存，并不是一个拼接 Token。

`organize_draft_results()` 展平 scores 后，先选再按索引排序：[`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L110-L139)

~~~python
top_scores_index = torch.topk(...).indices
top_scores_index = torch.sort(
    top_scores_index
).values
~~~

用同一索引取出 Token：

~~~python
draft_tokens = torch.gather(
    ss_token_list,
    index=top_scores_index,
    dim=1,
)
~~~





总窗口 4 行时，root 外只能留 3 个节点。按分数选出 A=0.60、AC=0.42、B=0.40，再按原池索引排为 `selected_index=[0,1,2]`，得到 `[A,B,C]`，不是 `[A,C,B]`。这只是存放顺序调整，没有重新评分。

### 那 `parent_list` 又是什么

还需将原池节点的父候选转为最终树 parent。例如知道候选来自 index 17，还不能直接得出它在 Verify 中的父 node。

第一层构造：

~~~python
[-1, 0, 1, ..., topk-1]
~~~

后续记录选中 frontier 的来源：

~~~python
topk_cs_index
+
(topk_sq * (i - 1) + topk)
~~~

最终汇成 `parent_list`。

它映射候选组与原池父候选，最终 node parent 仍需在 `selected_index` 中再次查找。

CPU reference 显式展开了这个转换。[`build_tree_kernel_efficient_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)

对一个选中节点先计算：

~~~python
parent_tb_idx = (
    selected_index[...] / topk
)
~~~

若 `parent_tb_idx == 0`，说明位于第一层，父亲就是 root/bonus。



否则读取：

~~~python
parent_token_idx = parent_list[parent_tb_idx]
~~~

得到原池 parent，再在最终选中索引中定位：

~~~python
find_parent_node(
    selected_index,
    parent_token_idx,
)
~~~





将本例选中的节点放在一张表里，原池位置与最终 node ID 就能对应起来：

| Verify 节点 | 原候选池位置 | 最终 node ID | 最终父节点 |
| --- | ---: | ---: | --- |
| root / previous bonus | 不来自候选池 | 0 | 无 |
| A | 0 | 1 | root，node 0 |
| B | 1 | 2 | root，node 0 |
| C（原池中的 AC） | 2 | 3 | A，node 1 |

`selected_index=[0,1,2]` 确定保留谁，`parent_list` 再帮助 Tree Builder 从候选组追到原池父节点，最后重建这棵 Verify 树：

~~~mermaid
flowchart TD
    R["node 0：root / bonus"] --> A["node 1：A；原池 index 0"]
    R --> B["node 2：B；原池 index 1"]
    A --> C["node 3：C；原池 index 2"]
~~~

固定测试给出下面的手工例子，见 [`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)。

~~~text
topk = 2
steps = 2
draft_token_num = 4
~~~

输入指定：

~~~python
parent_list   = [[-1, 0, 1]]
selected_index = [[0, 1, 2]]
~~~

得到：

~~~text
node 1、2 是 root children
node 3 是 node 1 的 child
~~~





以 C 为例：原池 index 2 除以 topk，`2 // topk = 1`，查得 `parent_list[1]=0`，指向 A。A 在 selected_index 中排第 0，加 root 后为 node 1；C 排第 2，加 root 后为 node 3，因此 parent 是 node 1。CPU C++ 的 `/ topk` 为整数除法，对应 Python `//`。

所以 `selected_index` 识别原池节点，`parent_list` 找回原池父候选，`retrieve_*` 才描述最终可遍历树，三者不能混用。

## 三、把树关系转成 Verify Mask

树还要转换成 Target Attention 能使用的布局。

`build_tree_kernel_efficient(...)` 根据这些关系生成 `tree_mask`、positions、`retrieve_index`、`retrieve_next_token`、`retrieve_next_sibling`，将树交给 Target Verify 与接受遍历。具体入口见 [`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L154-L289)。

继续用固定测试的小树：

~~~text
seq_len = 5

       root / bonus
        /       \
       A         B
       |
       C
~~~

positions 为：

~~~text
root → 5
A    → 6
B    → 6
C    → 7
~~~

已有 5 个历史位置，root 用 position 5。A、B 是不同 Tensor rows，却代表同一位置的互斥未来，因此都为 6；C 在 A 后为 7。position 不能替代 row 或 branch 身份。

若直接按普通下三角处理 `[root, A, B, C]`：

B 会读到 A，C 也读到 B，Target 输入已混入互斥未来。最后只选一路输出，无法修正前面算错的上下文。

真正需要的祖先关系为：

~~~text
root:
root

A:
root + A

B:
root + B

C:
root + A + C
~~~

QLEN-only 测试结果如下。[`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)

~~~text
          root A B C

root       1   0 0 0
A          1   1 0 0
B          1   0 1 0
C          1   1 0 1
~~~





### QLEN_ONLY：只显式保存 speculative query block

`QLEN_ONLY` 的 buffer shape 是 `[bs × draft_token_num × draft_token_num]`。

它只显式描述树内的 4×4 ancestor mask，不画 committed Prefix 列。历史仍须由后端正确提供，不能将没有历史列理解成不读历史。

### FULL_MASK：把 Prefix 可见性和 Tree ancestor block 放在同一个布局里

`FULL_MASK` 则为每请求安排：

~~~text
draft_token_num
×
(seq_len + draft_token_num)
~~~

左侧 seq_len 列是所有节点可见的已提交历史，右侧是祖先约束。本例 5 个历史、4 个树节点，完整布局有 `4×(5+4)=36` 元素；QLEN_ONLY 只保存右侧 `4×4=16` 元素。

关系如下：

~~~text
Committed Prefix      | Verify Tree
──────────────────────┼────────────────
t0 t1 t2 t3 t4        | root A B C

root  ✓ ✓ ✓ ✓ ✓       | 1 0 0 0
A     ✓ ✓ ✓ ✓ ✓       | 1 1 0 0
B     ✓ ✓ ✓ ✓ ✓       | 1 0 1 0
C     ✓ ✓ ✓ ✓ ✓       | 1 1 0 1
~~~

CPU reference 也区分：[`spec.cpp / build_tree_kernel_efficient_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)

~~~text
QLEN_ONLY:
只包含 qlen × qlen

FULL_MASK:
包含 seq_len prefix +
draft_token_num × draft_token_num tree block
~~~





### 不要把它简单写成“CPU 用 QLEN，GPU/NPU 用 FULL”

平台默认选择为：

~~~python
CPU      → QLEN_ONLY
非 CPU   → FULL_MASK
~~~

Runtime 还会进一步读取 backend 的 VerifyMask。

`build_eagle_verify_input()` 先查询 `target_attn_backend.verify_mask`。backend 若提供 `VerifyMask`，就用它的 `mode` 覆盖 Worker 默认值。`is_read=False` 时还可选择 QLEN_ONLY，让 Tree Builder 写固定 buffer，而后端不必读取完整 Prefix mask。[`build_eagle_verify_input()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L317-L404) [`verify_mask.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/verify_mask.py)

两种模式都需表达相同可见性，差别在历史列如何表示、后端如何读取。因此要核实际 `VerifyMask.mode`、`is_read`，不能只按 CPU/GPU/NPU 默认值分类。

## 四、Target 预测怎样确定接受路径

构造 Verify Input 时，将旧 bonus 放到 Draft nodes 前面：

~~~python
draft_tokens = torch.cat(
    (
        bonus_tokens.unsqueeze(1),
        draft_tokens,
    ),
    dim=1,
)
~~~

Target 以 `TARGET_VERIFY` 一次处理 `[root/previous bonus, draft nodes...]`。物理上数组连续，逻辑分支仍由 Mask 和 position 保持。

各行按自己的上下文得到预测：

~~~text
Target(root | committed prefix)

Target(A | prefix + root)

Target(B | prefix + root)

Target(C | prefix + root + A)
...
~~~

每行表达的是“输入该节点之后”的预测。例如 A 行读取 Prefix、root、自身 A，预测 A 后一个 Token。Greedy 随后进入：

~~~python
verify_tree_greedy_func(...)
~~~

Draft score 到这里已完成候选筛选，真正接受则逐节点服从 Target 预测，不会重新按整条路径概率挑最高分叶子。

### Tree 被转换成 first-child / next-sibling

Tree Builder 提供：

~~~text
retrieve_index
retrieve_next_token
retrieve_next_sibling
~~~

对下面的树：

~~~text
       root
      /    \
     A      B
     |
     C
~~~

固定测试生成：

~~~python
retrieve_index
=
[0, 1, 2, 3]

retrieve_next_token
=
[1, 3, -1, -1]

retrieve_next_sibling
=
[-1, 2, -1, -1]
~~~

即：

~~~text
root.first_child = A
A.next_sibling  = B
A.first_child   = C
~~~



测试见 [`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)。

### Greedy kernel 从 root 开始，用 Target prediction 匹配 child

CPU、Triton reference 从 `last_accept=root` 出发。先用 `cur = retrieve_next_token[last_accept]` 取得第一个 child，再读父行的 `target_tok = target_predict[last_accept]`，寻找 Token 与预测相符的 child。

若当前 child 不匹配，就令 `cur = retrieve_next_sibling[cur]`，继续检查兄弟节点；匹配后，将 `last_accept` 更新为这个 child，再进入下一层。若所有 sibling 都不匹配，就停止接受，保留当前父行的 Target 预测作为末尾输出，不跳到别的分支挑一个高分节点。实现见 [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/spec_tree.py)、[`verify_tree_greedy_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)。

两类决策因此分工为：

~~~text
Draft cumulative score
        ↓
决定哪些 branches 值得进入 Verify Tree


Target prediction
        ↓
决定最终接受哪一个 child
        ↓
形成唯一 accepted path
~~~

## 五、旧 root 与新 bonus 的位置

旧 bonus R 本轮作为 root，已经输出过，不再次返回用户；作为输入执行后才有自己的 Target KV。新的 X 则要等下一轮才有自身缓存。

例如树为：

~~~text
root = R

       R
      / \
     A   B
     |
     C
~~~

Target 预测为：

~~~text
at R → A
at A → C
at C → X
~~~

Kernel 在 R 行匹配 A，写 `predict[row(R)]=A`；在 A 行匹配 C，写 `predict[row(A)]=C`。到 C 无可匹配 child，则保存其本身的 Target 预测：

~~~python
predicts[last_accept_index]
=
target_predict[last_accept_index]
~~~

得到 `predict[row(C)] = X`。



实现见 [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/spec_tree.py#L176-L281)。

本轮输出为 `[A,C,X]`：A、C 是通过验证的 Draft Token，X 是最后有效输入 C 的 Target 预测。旧 root R 已在上一轮输出，本轮只作为输入，不再次返回用户。

| Token | 是否属于本轮新输出 | 本轮是否已形成自身输入 KV |
| --- | --- | --- |
| 旧 root R | 否，上一轮已输出 | 是，本轮作为 root 执行 |
| A、C | 是，接受的 Draft | 是，Verify 已执行这些输入行 |
| 新 bonus X | 是，Target 给出的末尾输出 | 否，尚未作为模型输入执行 |

`eagle_sample()` 的 `num_correct_drafts` 只计 A、C 两个节点，返回时加 1 纳入 X。源码见 [`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L731-L1029)。

~~~python
return (
    predict,
    num_correct_drafts + 1,
    accept_index,
)
~~~

所以两枚 Draft 对应有效输出长度 3；即使零候选匹配，root 行仍可提供一枚 Target 输出。

~~~text
accept_lens
=
accepted Draft token count
+
1 trailing Target token
~~~





但整理 KV 时，还要回到本轮真正执行过的输入行。

## 六、接受路径对应哪些输入 KV

Verify 产生的 KV 先按整棵树存放。

例如：

~~~text
Verify input slots:

slot0 → root R
slot1 → A
slot2 → B
slot3 → C
~~~

接受输入为 `R → A → C`，物理 slot 为 0、1、3，中间夹着 B。后续按链消费，所以 topk>1 时 `run_eagle_verify()` 还调用：

~~~python
_finalize_accept_tree_path(...)
~~~

先整理 KV，具体实现见 [`_finalize_accept_tree_path()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L407-L459)。

~~~python
move_accept_tokens_to_target_kvcache(
    batch,
    accept_index,
    accept_lens - 1,
    token_to_kv_pool_allocator,
)
~~~





`accept_lens-1` 将输出长度还原为接受的 Draft 数；mover 再补 root 的输入行。扣掉新 bonus、加回旧 root 只是数量相同，二者并没有相同 KV 状态。

mover 的 `num_correct_drafts` 只数正确 Draft，不含新 trailing bonus。调用方从输出长度减去新 bonus，mover 再补回旧 root 的输入行，计数可写为：

~~~text
accept_lens = accepted_drafts + 1 个新 trailing bonus
传给 mover：num_correct_drafts = accept_lens - 1
实际提交 KV 行数：num_correct_drafts + 1 个旧 root
~~~

这里加减的数量相同，Token 身份却不同：提交的是 previous bonus/root 与接受的 Draft input nodes。实现见 [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L728-L790)。

X 只有预测结果，没有本轮输入 KV，不能参与搬移。应按选中 Verify 行找源缓存，不能直接照新输出 ID 列表查位置。

代回输入 `[R,A,B,C]`：`accept_lens=3`，减一得到两枚接受的 Draft；实际整理的输入行是 R、A、C，共 `2+1=3` 行。

`accept_index=[0,1,3]` 读输入数组得到 `[R,A,C]`，读相同行的 predict 数组则得到 `[A,C,X]`。同一索引连接输入状态与输出预测，Token 因而错一位。

### Compaction 不只整理 KV

`_finalize_accept_tree_path()` 还按接受索引，将 predict 与 Target hidden_states 整理到请求 block 前部。

下游只认连续有效前缀。若 hidden 存在，也必须与 predict 使用相同索引，否则即便 KV、Token 正确，下一轮仍可能配错分支特征。

布局因此经历：

~~~text
候选阶段
────────────────
Tree layout

R
├─ A
│  └─ C
└─ B


Target 接受
────────────────
R → A → C


提交阶段
────────────────
把 R/A/C 对应的
KV / predict / hidden
整理成 chain-compatible layout
~~~

B 的 KV 可暂留 overshoot slots，之后释放或复用，但不属于 committed sequence。固定形状 gather 也可能读到无效填充值，下游必须受 `accept_lens` 限制。临时探索用树，继续生成只沿选中的链。

## 七、DeepSeek-V4 的模型与后端限制

最后核对 V4 模型条件：固定 hook 直接要求 EAGLE `speculative_eagle_topk == 1`。因此前文多分支属于通用 Runtime，当前 V4 配置只开放 Chain。[`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/deepseek_v4_hook.py)

官方 Ascend 配置节选为：

~~~bash
--device npu
--attention-backend dsv4
--page-size 128

--speculative-algorithm EAGLE
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
~~~

topk=1 每位置只有一个后继，实际是链式候选。节选仍缺模型路径、地址等完整启动条件。



教程见 [DeepSeek-V4-Flash Ascend Tutorial](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx)。

另外还有 paged-tree backend guard，源码见 [`_handle_eagle_family()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L1067-L1095)。

~~~python
_PAGE_TREE_SPEC_BACKENDS = (
    "flashinfer",
    "fa3",
    "triton",
)

if (
    speculative_eagle_topk > 1
    and page_size > 1
    and attention_backend
        not in _PAGE_TREE_SPEC_BACKENDS
):
    raise ValueError(...)
~~~





~~~text
dsv4
+
page_size = 128
+
topk > 1
~~~

这个组合会在启动时拒绝。

### 这个 Guard 不是在说“NPU 不支持 Tree”

NPU 已有 `torch.ops.npu.build_tree_kernel_efficient` 和 `sgl_kernel_npu.sample.verify_tree_greedy`，V4 Ascend 也有 multi-step Draft backend。它们分别提供部分组件：

~~~text
Tree Builder
Greedy Tree Verify
Multi-step Draft
~~~

这些入口尚不能证明已拼成可运行的 V4 多分支路径。paged-tree 还需处理共享历史与分支未来的分页归属，Tree Build、Accept 可用并不足以解决分配。

源码称相关机制为 `two-pass cascade draft-decode`。

它包括：

~~~text
shared prefix pass
+
per-branch expand pass
+
prefix-tail duplication
~~~

以一个未填满的 Prefix 尾页看这项需求。

假设尾页为：

~~~text
Prefix tail page:

[P P P _ _ _ _ _]
~~~

A、B 两路都要继续写入。

page-size>1 时，两路的首个分支页都需已有 Prefix tail：

~~~text
branch A page:
[P P P A1 A2 ...]

branch B page:
[P P P B1 B2 ...]
~~~

共写同页会覆盖未来，只给空页又会丢掉已有 tail。因此先给各 branch 准备正确首个页、复制 partial-tail，再各自写 A1/A2 或 B1/B2。8 格仅为教学图，V4 配置仍是 page_size=128。

`duplicate_prefix_tail_to_draft_branches(...)` 将 Prefix partial-tail page 复制到各 branch 的首个页空位，用于 `page_size>1` 且 `topk>1` 的路径。[`duplicate_prefix_tail_to_draft_branches()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L59-L104)

固定 allowlist 只有 `flashinfer`、`fa3`、`triton`，这些 backend 才进入对应的 paged-tree draft-decode。dsv4 不在其中，因此固定版本拒绝该组合。

### 将模型约束与页布局一起检查

即使改为 page_size=1 消除这一 guard，V4 模型 hook 的 topk=1 仍然生效。两层条件都需满足，不能只改页面大小就宣称树可用。

扩展时还需连接模型约束、branch KV、Prefix Tail 复制、Attention metadata 和 Compact。通用 NPU Kernel 是其中的组件，完整支持还需覆盖整条模型路径。

从搜索到提交，score 分配候选预算，索引与 Mask 保持路径上下文，接受后再将状态整理回唯一链。每一步都要与物理缓存布局对应。

### 源码阅读入口

| 目标 | 固定版本入口 |
| --- | --- |
| EAGLE Draft 主循环 | [`EagleDraftWorker.draft_forward()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_v2.py#L778-L951) |
| 第一层 / 后续 Top-K frontier | [`select_top_k_tokens()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L312-L381) |
| 累计路径 score | [`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L332-L366) |
| 全局 Verify Budget 裁剪 | [`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L110-L139) |
| Tree Builder | [`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L154-L289) |
| parent / selected index reference | [`build_tree_kernel_efficient_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp) |
| TopK=2 手工 Tree/Mask 测试 | [`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562) |
| FULL_MASK / QLEN_ONLY buffer contract | [`verify_mask.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/verify_mask.py) |
| Verify Input 组装 | [`build_eagle_verify_input()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L317-L404) |
| Target Verify | [`run_eagle_verify()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L462-L675) |
| Greedy / Sampling dispatch | [`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L731-L1029) |
| Greedy Tree traversal | [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/spec_tree.py#L176-L281) |
| CPU Greedy / Tree reference | [`spec.cpp`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp) |
| Accepted path compaction | [`_finalize_accept_tree_path()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L407-L459) |
| Target KV mover | [`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L728-L790) |
| Page-tree Prefix Tail Duplication | [`duplicate_prefix_tail_to_draft_branches()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L59-L104) |
| `topk>1 + page_size>1` guard | [`_handle_eagle_family()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L1067-L1095) |
| Ascend DSV4 Multi-Step Draft | [`DeepseekV4AscendMultiStepDraftBackend`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L2354-L2512) |
| Ascend 官方 DeepSeek-V4 Spec 配置 | [`deepseek_v4_flash.mdx`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx) |
