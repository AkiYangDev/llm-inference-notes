# EAGLE 候选树：从候选池到 Verify 与接受路径

Draft 把预算全用在 `A → B → C`，Target 接受 A 后预测 X，B、C 的计算就用不上。通用 EAGLE 可以多留几个分支，让 Target 从同一批候选中找出能够继续的路径。

多条路径一起 forward，Attention 仍须为每个节点保留自己的上下文。这里用一棵小树跟踪候选池、父节点、Mask 和接受后的缓存整理。多分支是通用 Runtime 的能力；本文固定 DeepSeek-V4 配置要求 EAGLE `topk=1`，最后一节说明限制来自哪里。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。正文分析通用 EAGLE `topk>1`。官方 Ascend DSV4 配置使用 `--attention-backend dsv4`、`page_size=128`、`speculative_eagle_topk=1`；V4 模型 hook 与 paged-tree backend guard 的具体条件见末节。

</details>

## 一、在有限预算内保留候选路径

链式候选在首次分歧处停止。A 后面若预测 X，B、C 即使之后偶然匹配，也已经属于另一段上下文。

`topk>1` 在同一上下文保留多个提议。例如 `topk=2`，Extend 首先给出：

~~~text
A   p=0.60
B   p=0.40
~~~

`select_top_k_tokens()` 将 A、B 作为下一次 NextN 输入，并为两路复制父 hidden。下一步分别在 A、B 后继续预测。分支的 KV 分配、写入还要另行处理，复制 hidden 只完成了特征准备。[`_select_top_k_tokens_first()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L312-L330)

~~~python
input_ids = topk_index.flatten()

hidden_states = hidden_states.repeat_interleave(
    topk,
    dim=0,
)
~~~

假设两路接着得到：

~~~text
A 后：
C  0.70
D  0.30

B 后：
E  0.80
F  0.20
~~~

新候选概率乘上父路径 score，得到累计分数。[`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L332-L366)

~~~python
expand_scores = (
    scores.unsqueeze(2)
    * topk_p.view(-1, topk, topk)
)
~~~

父分数 `scores=[0.60, 0.40]` 对应：

~~~text
A → C : 0.60 × 0.70 = 0.42
A → D : 0.60 × 0.30 = 0.18
B → E : 0.40 × 0.80 = 0.32
B → F : 0.40 × 0.20 = 0.08
~~~

下一步只继续展开这 `topk × topk` 条 child paths 中的 top-k：

~~~python
topk_cs_p, topk_cs_index = fast_topk(
    expand_scores.flatten(start_dim=1),
    topk,
    dim=-1,
)
~~~

新的 frontier 是：

~~~text
A → C : 0.42
B → E : 0.32
~~~

模型下一次只继续这两个节点。本轮四个 child 的分数、Token 却都已进入候选池，最终裁剪仍可把没有继续展开的节点送去 Verify。

这条非 rejection-sampling Tree 路径使用 Draft 累计路径概率：

~~~text
score(child)
=
score(parent)
×
draft_probability(child | parent)
~~~

B→E 的最后一跳概率 0.80，虽然高于 A→C 的 0.70，乘上父概率后却是 0.32 低于 0.42。累计 score 用来分配搜索和 Verify 预算，候选最终能否接受，仍由 Target 判断。

rollout 每步保存：

~~~python
score_list.append(tree_info[0])
token_list.append(tree_info[1])
parents_list.append(tree_info[2])
~~~

`organize_draft_results()` 再从整个池子中裁剪：

~~~python
top_scores = torch.topk(
    score_list,
    num_draft_token - 1,
    dim=-1,
)
~~~

总窗口先给旧 bonus/root 留 1 行，候选占 `num_draft_tokens - 1` 行。旧 bonus 早已输出，这次作为输入才能形成自己的 Target KV。

层内 top-k 控制继续展开的 frontier，全局 top-k 控制 Verify 节点数。因此搜索没有完整展开 k 叉树，topk 与最终窗口宽度也各有用途。

## 二、候选池索引怎样还原父节点关系

搜索中不断展平、筛选，最终 Verify 的节点号已经不同于候选池位置。父子关系要靠保留下来的索引重建。

候选整理返回：

~~~text
parent_list
top_scores_index
draft_tokens
~~~

`top_scores_index` 进入 Tree Builder 后改称 `selected_index`。它保存原始展平候选池的位置，用来取 Token、追父节点；最终压缩后的 node ID 还要再转换一次。

<a id="_3"></a>

### rollout 候选池的展平顺序

第一层 A、B 位于 index 0、1。topk=2 时，下一层每个 frontier 再给两个 child，AC、AD、BE、BF 依次放在 index 2、3、4、5。AC 表示末端 C，前面的 A 由父节点 metadata 表达。

[`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L110-L139) 展平 scores，选出高分节点后，再按原池索引排序：

~~~python
top_scores_index = torch.topk(...).indices
top_scores_index = torch.sort(
    top_scores_index
).values
~~~

相同索引接着取出 Token：

~~~python
draft_tokens = torch.gather(
    ss_token_list,
    index=top_scores_index,
    dim=1,
)
~~~

总窗口 4 行，扣除 root 只能留 3 个节点。分数选中 A=0.60、AC=0.42、B=0.40，再按原索引排成 `selected_index=[0,1,2]`，存放为 `[A,B,C]`。评分没有改变，只把选中节点放回池中的先后顺序。

### parent_list 保存原池父候选

知道某个节点来自原池 index 17，还不能直接得到它在 Verify 树中的父 node ID。

第一层先构造：

~~~python
[-1, 0, 1, ..., topk-1]
~~~

后续层记录选中 frontier 来自哪里：

~~~python
topk_cs_index
+
(topk_sq * (i - 1) + topk)
~~~

这些信息汇成 `parent_list`。

它从候选组映射到原池父候选，再由 `selected_index` 将父候选换成最终 node ID。

CPU reference 展示了这次转换。[`build_tree_kernel_efficient_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)

对选中节点先计算：

~~~python
parent_tb_idx = (
    selected_index[...] / topk
)
~~~

`parent_tb_idx == 0` 表示第一层，父节点为 root/bonus。

其他层读取：

~~~python
parent_token_idx = parent_list[parent_tb_idx]
~~~

取得原池 parent，再在最终选中节点中定位：

~~~python
find_parent_node(
    selected_index,
    parent_token_idx,
)
~~~

本例的原池位置、最终 node ID 对应如下：

| Verify 节点 | 原候选池位置 | 最终 node ID | 最终父节点 |
| --- | ---: | ---: | --- |
| root / previous bonus | 不来自候选池 | 0 | 无 |
| A | 0 | 1 | root，node 0 |
| B | 1 | 2 | root，node 0 |
| C（原池中的 AC） | 2 | 3 | A，node 1 |

`selected_index=[0,1,2]` 确定保留 A、B、C，`parent_list` 将 C 追到原池 A，Tree Builder 由此恢复 Verify 树：

~~~mermaid
flowchart TD
    R["node 0：root / bonus"] --> A["node 1：A；原池 index 0"]
    R --> B["node 2：B；原池 index 1"]
    A --> C["node 3：C；原池 index 2"]
~~~

[`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562) 使用同一棵手工小树。

~~~text
topk = 2
steps = 2
draft_token_num = 4
~~~

测试输入为：

~~~python
parent_list   = [[-1, 0, 1]]
selected_index = [[0, 1, 2]]
~~~

输出为：

~~~text
node 1、2 是 root children
node 3 是 node 1 的 child
~~~

以 C 为例，原池 index 2 先算 `2 // topk = 1`，查得 `parent_list[1]=0`，指向 A。A 在 selected_index 中排第 0，加 root 后为 node 1；C 排第 2，加 root 后为 node 3，因此 C 的 parent 是 node 1。CPU C++ 的 `/ topk` 是整数除法，对应 Python `//`。

这几组索引依次完成三件事：`selected_index` 找原池节点，`parent_list` 找原池父候选，`retrieve_*` 描述最后可遍历的树。

## 三、把树关系转成 Verify Mask

父子关系重建以后，还要变成 Target Attention 的输入布局。

[`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L154-L289) 生成 `tree_mask`、positions、`retrieve_index`、`retrieve_next_token`、`retrieve_next_sibling`。Mask、位置用于 Verify，retrieve 数据用于之后遍历接受路径。

继续看这棵测试树：

~~~text
seq_len = 5

       root / bonus
        /       \
       A         B
       |
       C
~~~

它的 positions 是：

~~~text
root → 5
A    → 6
B    → 6
C    → 7
~~~

历史已有 5 个位置，root 的 position 为 5。A、B 对应同一位置的两种未来，都取 6，但各占一行；C 在 A 后，position 为 7。行号区分存储，position 表示路径上的位置。

若把 `[root, A, B, C]` 当普通下三角：

B 会读到 A，C 也会读到 B，互斥的未来已经混进 Attention。最后只输出一路，也无法补救前面用错的上下文。

树中实际需要的可见性为：

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

QLEN-only 测试给出的结果为：[`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)

~~~text
          root A B C

root       1   0 0 0
A          1   1 0 0
B          1   0 1 0
C          1   1 0 1
~~~

### QLEN_ONLY：只显式保存 speculative query block

`QLEN_ONLY` 使用 `[bs × draft_token_num × draft_token_num]` 的 buffer。

这里显式存的是树内 4×4 ancestor mask，committed Prefix 由后端另行提供。没有画出历史列，节点仍然需要读取历史。

### FULL_MASK：把 Prefix 可见性和 Tree ancestor block 放在同一个布局里

`FULL_MASK` 为每请求安排：

~~~text
draft_token_num
×
(seq_len + draft_token_num)
~~~

左侧 seq_len 列表示所有节点可见的已提交历史，右侧限制树内祖先。本例有 5 个历史位置、4 个树节点，完整布局为 `4×(5+4)=36` 个元素；QLEN_ONLY 只保存右侧 `4×4=16` 个。

两种表示的关系为：

~~~text
Committed Prefix      | Verify Tree
──────────────────────┼────────────────
t0 t1 t2 t3 t4        | root A B C

root  ✓ ✓ ✓ ✓ ✓       | 1 0 0 0
A     ✓ ✓ ✓ ✓ ✓       | 1 1 0 0
B     ✓ ✓ ✓ ✓ ✓       | 1 0 1 0
C     ✓ ✓ ✓ ✓ ✓       | 1 1 0 1
~~~

CPU reference 也分别处理两种模式。[`spec.cpp / build_tree_kernel_efficient_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)

~~~text
QLEN_ONLY:
只包含 qlen × qlen

FULL_MASK:
包含 seq_len prefix +
draft_token_num × draft_token_num tree block
~~~

<a id="cpu-qlengpunpu-full"></a>

### VerifyMask 覆盖平台默认模式

平台先给出默认选择：

~~~python
CPU      → QLEN_ONLY
非 CPU   → FULL_MASK
~~~

实际 Runtime 还会读取 Attention backend 的 VerifyMask。

`build_eagle_verify_input()` 查询 `target_attn_backend.verify_mask`，若 backend 提供 `VerifyMask`，就用其 `mode` 覆盖默认值。`is_read=False` 时还可以选择 QLEN_ONLY，让 Tree Builder 写固定 buffer，后端不用读取完整 Prefix mask。[`build_eagle_verify_input()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L317-L404) [`verify_mask.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/verify_mask.py)

两种模式表达相同的可见性，区别是历史列怎样表示、后端怎样消费。因此定位这条路径时，实际生效的是 `VerifyMask.mode` 与 `is_read`。

## 四、Target 预测怎样确定接受路径

Verify Input 把旧 bonus 放在 Draft nodes 前：

~~~python
draft_tokens = torch.cat(
    (
        bonus_tokens.unsqueeze(1),
        draft_tokens,
    ),
    dim=1,
)
~~~

Target 用 `TARGET_VERIFY` 一次处理 `[root/previous bonus, draft nodes...]`，连续数组中的分支仍靠 Mask、position 保持独立。

每行根据自身上下文产生预测：

~~~text
Target(root | committed prefix)

Target(A | prefix + root)

Target(B | prefix + root)

Target(C | prefix + root + A)
...
~~~

例如 A 行读 Prefix、root、自身 A，预测 A 后面应出现的 Token。Greedy 再进入：

~~~python
verify_tree_greedy_func(...)
~~~

Draft score 到此完成筛选。接受过程逐节点匹配 Target 预测，高分叶子没有额外优先权。

### Tree 被转换成 first-child / next-sibling

Tree Builder 给出：

~~~text
retrieve_index
retrieve_next_token
retrieve_next_sibling
~~~

对于这棵树：

~~~text
       root
      /    \
     A      B
     |
     C
~~~

固定测试返回：

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

这些数组表示：

~~~text
root.first_child = A
A.next_sibling  = B
A.first_child   = C
~~~

具体断言见 [`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)。

### Greedy kernel 从 root 开始，用 Target prediction 匹配 child

CPU、Triton reference 从 `last_accept=root` 开始。`cur = retrieve_next_token[last_accept]` 取得第一个 child，`target_tok = target_predict[last_accept]` 取得父行预测，然后比较 child 的 Token。

当前 child 不匹配，就沿 `retrieve_next_sibling[cur]` 找兄弟；匹配后把 `last_accept` 改为该 child，继续下一层。所有兄弟都不匹配时停止，末尾输出取当前父行的 Target 预测。实现为 [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/spec_tree.py)、[`verify_tree_greedy_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)。

Draft 筛选、Target 接受的分工为：

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

旧 bonus R 已输出过，本轮只作为 root 输入，执行后形成自己的 Target KV。新 bonus X 这轮只有预测，自己的 KV 留待下一轮计算。

例如候选树：

~~~text
root = R

       R
      / \
     A   B
     |
     C
~~~

Target 给出：

~~~text
at R → A
at A → C
at C → X
~~~

Kernel 在 R 行匹配 A，写 `predict[row(R)]=A`；在 A 行匹配 C，写 `predict[row(A)]=C`。C 没有可匹配的 child，便保留自己的 Target 预测：

~~~python
predicts[last_accept_index]
=
target_predict[last_accept_index]
~~~

即 `predict[row(C)] = X`。

写入过程见 [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/spec_tree.py#L176-L281)。

返回的有效输出为 `[A,C,X]`，A、C 是接受的 Draft，X 来自最后有效输入 C 的预测。旧 R 留在输入一侧。

| Token | 是否属于本轮新输出 | 本轮是否已形成自身输入 KV |
| --- | --- | --- |
| 旧 root R | 否，上一轮已输出 | 是，本轮作为 root 执行 |
| A、C | 是，接受的 Draft | 是，Verify 已执行这些输入行 |
| 新 bonus X | 是，Target 给出的末尾输出 | 否，尚未作为模型输入执行 |

`eagle_sample()` 的 `num_correct_drafts` 计 A、C 两枚，返回时加 1，把 X 纳入有效输出长度。[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L731-L1029)

~~~python
return (
    predict,
    num_correct_drafts + 1,
    accept_index,
)
~~~

因此两枚 Draft 使本轮推进 3 个位置。即使没有候选匹配，root 行也能给出一枚 Target 输出。

~~~text
accept_lens
=
accepted Draft token count
+
1 trailing Target token
~~~

缓存整理则要取这些预测对应的输入行。

## 六、接受路径对应哪些输入 KV

Verify 按整棵树写入临时 KV。

本例的存放顺序为：

~~~text
Verify input slots:

slot0 → root R
slot1 → A
slot2 → B
slot3 → C
~~~

有效输入链 `R → A → C` 占物理 slot 0、1、3，B 夹在中间。为了让后续按连续链消费，topk>1 时 `run_eagle_verify()` 还调用：

~~~python
_finalize_accept_tree_path(...)
~~~

[`_finalize_accept_tree_path()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L407-L459) 先整理 KV。

~~~python
move_accept_tokens_to_target_kvcache(
    batch,
    accept_index,
    accept_lens - 1,
    token_to_kv_pool_allocator,
)
~~~

`accept_lens-1` 还原接受的 Draft 数，mover 再补 root 输入行。这里减去新 bonus、加回旧 root，数量相同，Token 身份却发生了变化。

mover 的 `num_correct_drafts` 因而不含新 trailing bonus。数量关系为：

~~~text
accept_lens = accepted_drafts + 1 个新 trailing bonus
传给 mover：num_correct_drafts = accept_lens - 1
实际提交 KV 行数：num_correct_drafts + 1 个旧 root
~~~

提交内容是 previous bonus/root 与接受的 Draft input nodes。[`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L728-L790)

X 本轮只有预测，尚无输入 KV。缓存搬移要从选中的 Verify 行取源，输出 ID 列表无法直接当作 KV 位置。

输入 `[R,A,B,C]` 中，`accept_lens=3` 减一是两枚接受的 Draft，实际搬移 R、A、C，共 `2+1=3` 行。

同一个 `accept_index=[0,1,3]`，读输入数组得到 `[R,A,C]`，读 predict 数组得到 `[A,C,X]`。索引把每行输入状态和它产生的输出预测联系起来。

### Compaction 不只整理 KV

`_finalize_accept_tree_path()` 还会按相同接受索引，将 predict、Target hidden_states 移到请求 block 前部。

下游按连续有效前缀读取，hidden 与 predict 必须来自同一路。KV、Token 即使正确，分支 hidden 配错仍会影响下轮 Draft。

布局变化为：

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

B 的 KV 可以暂留 overshoot slots，之后释放或复用。固定形状 gather 也可能读到无效填充值，所以有效范围仍受 `accept_lens` 限制。到下一轮继续生成时，正式历史只剩选中链。

## 七、DeepSeek-V4 的模型与后端限制

固定 V4 hook 直接要求 EAGLE `speculative_eagle_topk == 1`，当前模型配置只开放 Chain。前面的多分支过程属于通用 Runtime。[`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/deepseek_v4_hook.py)

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

topk=1 每位置只有一个后继。这里只列相关参数，完整启动还需要模型路径、地址等条件。

出处为 [DeepSeek-V4-Flash Ascend Tutorial](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx)。

此外，[`_handle_eagle_family()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L1067-L1095) 还有 paged-tree backend guard。

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

命中这个组合，启动会拒绝配置。

<a id="guard-npu-tree"></a>

### NPU 组件与 paged-tree 分配路径

NPU 已有 `torch.ops.npu.build_tree_kernel_efficient`、`sgl_kernel_npu.sample.verify_tree_greedy`，V4 Ascend 也有 multi-step Draft backend。这些入口分别承担：

~~~text
Tree Builder
Greedy Tree Verify
Multi-step Draft
~~~

完整 V4 多分支还需要把这些组件接起来。paged-tree 尤其要安排共享历史、分支未来的分页归属，Tree Build 与 Accept 并不负责整套页分配。

源码把相应机制称为 `two-pass cascade draft-decode`。

过程包括：

~~~text
shared prefix pass
+
per-branch expand pass
+
prefix-tail duplication
~~~

未填满的 Prefix 尾页，可以说明分支分配为什么还需复制。

假设尾页为：

~~~text
Prefix tail page:

[P P P _ _ _ _ _]
~~~

A、B 两路接着分别写入自己的未来。

page-size>1，两路的首个分支页都要带上原 Prefix tail：

~~~text
branch A page:
[P P P A1 A2 ...]

branch B page:
[P P P B1 B2 ...]
~~~

共用同一页，未来写入会相互覆盖；直接给空页，又会丢掉尾部历史。因此各 branch 先取得首个页、复制 partial-tail，再写 A1/A2 或 B1/B2。图用 8 格演示，V4 配置的 page_size=128。

`duplicate_prefix_tail_to_draft_branches(...)` 在 `page_size>1`、`topk>1` 路径中，把 Prefix partial-tail page 复制到各 branch 首个页的空位。[`duplicate_prefix_tail_to_draft_branches()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L59-L104)

固定 allowlist 只含 `flashinfer`、`fa3`、`triton`。dsv4 不在这条 paged-tree draft-decode 支持路径中，因此该版本拒绝前述组合。

<a id="_4"></a>

### paged-tree 的模型约束与页布局

改成 page_size=1，只会消除这一 backend guard，V4 模型 hook 的 topk=1 仍然有效。

扩展 V4 多分支，需要同时处理模型约束、branch KV、Prefix Tail 复制、Attention metadata 和 Compact，才能贯通从提议到提交的模型路径。

<a id="_5"></a>

### EAGLE 源码阅读入口

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
