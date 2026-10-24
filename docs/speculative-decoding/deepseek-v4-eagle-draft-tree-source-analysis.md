# EAGLE 候选树：从候选池到 Verify 与接受路径

链式 Draft 把预算放在同一条候选路径上，中途预测错误后，后续候选便无法继续接受。通用 EAGLE Runtime 还可以保留多个分支，把有限的候选预算分给几条较有希望的路径，再交给 Target 一次验证。

本文沿一棵小树解释累计分数、候选池和父节点索引，说明它怎样变成可执行的 Verify 输入。DeepSeek-V4 在本文固定版本中仍要求 EAGLE `topk=1`；多分支部分讨论的是通用 Runtime 机制，最后单独核对 V4 的配置与后端限制。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。正文分析 SGLang EAGLE Runtime 中真实存在的 `topk>1` Draft Tree 机制，并以 DeepSeek-V4 / NextN 为模型背景。当前 DeepSeek-V4 + Ascend 官方部署使用 `--attention-backend dsv4`、`page_size=128`、`speculative_eagle_topk=1`；同一版本的 DeepSeek-V4 模型配置 hook 对 EAGLE 直接要求 `topk=1`；另有 `dsv4 + page_size>1 + topk>1` 的后端 guard。因此本文讲的是 DeepSeek-V4 所使用的 EAGLE Runtime 的 Tree 能力，以及它为什么在当前 Ascend DSV4 官方配置中退化成 Chain，而不是宣称当前 Ascend DSV4 已经开放多分支 Tree 生产配置。

</details>

## 一、在有限预算内保留候选路径

先从 `topk=1` 开始。

假设 NextN 依次提出：`A → B → C`

如果 Target 实际希望：`A → X`

那么 B 之后的 rollout 都无法进入最终已接受上下文。

`topk>1` 的直觉是：不要在每个位置只押一个答案。

假设：`topk = 2`

第一步 Draft Extend 得到：

~~~text
A   p=0.60
B   p=0.40
~~~

`select_top_k_tokens()` 的第一步直接把这两个 candidate 作为下一次 NextN 的输入，同时把父 hidden state复制到两个 branch：

~~~python
input_ids = topk_index.flatten()

hidden_states = hidden_states.repeat_interleave(
    topk,
    dim=0,
)
~~~

固定源码：

[`_select_top_k_tokens_first()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L312-L330)

到了第二层，假设两个 branch 分别给出：

~~~text
A 后：
C  0.70
D  0.30

B 后：
E  0.80
F  0.20
~~~

这时源码不是分别从 A、B 各保留两个 child，而是先算**累计路径 score**：

~~~python
expand_scores = (
    scores.unsqueeze(2)
    * topk_p.view(-1, topk, topk)
)
~~~

如果第一层 `scores=[0.60, 0.40]`，那么第二层得到：

~~~text
A → C : 0.60 × 0.70 = 0.42
A → D : 0.60 × 0.30 = 0.18
B → E : 0.40 × 0.80 = 0.32
B → F : 0.40 × 0.20 = 0.08
~~~

然后源码在这 `topk × topk` 个 child path 中再选回 top-k：

~~~python
topk_cs_p, topk_cs_index = fast_topk(
    expand_scores.flatten(start_dim=1),
    topk,
    dim=-1,
)
~~~

于是下一轮继续展开的 frontier 是：

~~~text
A → C : 0.42
B → E : 0.32
~~~

而不是四条全部继续向下。

固定源码：

[`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L332-L366)

这里的 score 要精确理解。

它不是 Target probability，也不是一个额外训练出来的 confidence head；在当前非 rejection-sampling 的 EAGLE Tree 路径里，它就是沿 Draft frontier 递推得到的**候选路径累计概率质量**：

~~~text
score(child)
=
score(parent)
×
draft_probability(child | parent)
~~~

因此它的用途是：

> **给 Draft 搜索空间排序，决定哪些未来更值得继续展开、哪些节点更值得占用有限的 Target Verify Budget。**

它不决定最终哪条路径被 Target 接受。

Draft rollout 每一步都记录三组数据：

~~~python
score_list.append(tree_info[0])
token_list.append(tree_info[1])
parents_list.append(tree_info[2])
~~~

等多步 rollout 结束后，`organize_draft_results()` 还会做第二次全局裁剪：

~~~python
top_scores = torch.topk(
    score_list,
    num_draft_token - 1,
    dim=-1,
)
~~~

为什么是 `num_draft_token - 1`？

因为 Target Verify Window 的第一个位置留给上一轮已经由 Target 确认的 `bonus token`，留给 Draft nodes 的容量只有：`num_draft_tokens - 1`

因此 EAGLE 有两层 budget：

~~~text
层内：
k × k child
    ↓
只保留 top-k frontier 继续展开


全局：
所有 rollout 过程中出现的候选
    ↓
只保留 num_draft_tokens - 1
进入最终 Verify Tree
~~~

这不是完整 k 叉树，更接近一个固定 frontier width、固定 verify-token budget 的未来搜索。

## 二、候选池索引怎样还原父节点关系

树图直接画出了父子关系，源码却要通过多组索引还原它。接下来分别看每组索引对应哪个候选池。

`organize_draft_results()` 最后返回：

~~~text
parent_list
top_scores_index
draft_tokens
~~~

这里的 `top_scores_index` 在传给 Tree Builder 后叫：`selected_index`

必须先明确：

> **`selected_index` 不是最终 Verify Tree 内部的 node id。它是 Draft rollout 过程中“展平候选搜索池”的索引。**

### 搜索候选池是怎样展平的

第一层 top-k：

~~~text
A
B
~~~

可以理解成搜索池：

~~~text
index 0 → A
index 1 → B
~~~

第二层时，每个 frontier parent 又产生 k 个 child。

对于 `topk=2`：

~~~text
A → C
A → D
B → E
B → F
~~~

会继续排到候选搜索池后面。

所以：`selected_index`

回答的是：

> **最终保留下来的这个 Draft node，最初位于 rollout 候选池的哪个位置？**

`organize_draft_results()` 正是先把所有 score flatten，再：

~~~python
top_scores_index = torch.topk(...).indices
top_scores_index = torch.sort(
    top_scores_index
).values
~~~

然后拿这组 index 去 gather token：

~~~python
draft_tokens = torch.gather(
    ss_token_list,
    index=top_scores_index,
    dim=1,
)
~~~

固定源码：

[`organize_draft_results()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L110-L139)

所以最终 `draft_tokens` 已经被压成一个小数组，但 `selected_index` 仍然保留每个节点来自原搜索池的身份。

### 那 `parent_list` 又是什么

只知道“这个 node 原来是 search-pool index 17”仍然不够。

Tree Builder还要知道：`index 17 的 parent 是谁？`

`_select_top_k_tokens_first()` 第一层构造：

~~~python
[-1, 0, 1, ..., topk-1]
~~~

后续层则记录被选中 frontier 的来源：

~~~python
topk_cs_index
+
(topk_sq * (i - 1) + topk)
~~~

这些值最终组成 `parent_list`。

它不是“最终 Tree 的父节点编号数组”，而是一个**从候选组 / parent-table 位置回到 rollout 搜索池 parent candidate 的映射表**。

CPU reference kernel 把这个关系写得很清楚。

对于最终某个 selected node：

~~~python
parent_tb_idx = (
    selected_index[...] / topk
)
~~~

如果：`parent_tb_idx == 0`

说明它属于第一层，父节点就是 Verify Tree 的 root/bonus。

否则先：

~~~python
parent_token_idx = parent_list[parent_tb_idx]
~~~

得到父候选在搜索池中的 index，再到最终 `selected_index` 里寻找它：

~~~python
find_parent_node(
    selected_index,
    parent_token_idx,
)
~~~

固定实现：

[`build_tree_kernel_efficient_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)

因此两套 index 的关系可以画成：

~~~text
Draft rollout candidate pool
───────────────────────────
0  A
1  B
2  AC
3  AD
4  BE
5  BF
...


        global score pruning
                ↓


selected_index
───────────────────────────
[0, 1, 2]

含义：
最终留下 search-pool 中
A、B、AC


parent_list
───────────────────────────
帮助 Tree Builder 从
AC 的候选组
追溯到它的 parent A


        rebuild
                ↓


Final Verify Tree
───────────────────────────
       root
      /    \
     A      B
     |
     C
~~~

一个很好的固定测试是：

~~~text
topk = 2
steps = 2
draft_token_num = 4
~~~

测试手工给出：

~~~python
parent_list   = [[-1, 0, 1]]
selected_index = [[0, 1, 2]]
~~~

最终构造出：

~~~text
node 1、2 是 root children
node 3 是 node 1 的 child
~~~

固定测试：

[`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)

三类索引的职责分别是：

> **`selected_index` 保存“最终节点来自原搜索池哪里”，`parent_list` 保存“搜索池中的 parent 关系怎样追溯”；`retrieve_*` 才是 Tree Build 完成以后面向 Verify traversal 的最终树结构。**

## 三、把树关系转成 Verify Mask

最终 Tree 结构确定以后，需要把它转换成一次 Target Forward 可以执行的 Attention Layout。

入口：

~~~python
build_tree_kernel_efficient(...)
~~~

它同时生成：

~~~text
tree_mask
positions
retrieve_index
retrieve_next_token
retrieve_next_sibling
~~~

固定源码：

[`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L154-L289)

仍然使用固定测试中的小树：

~~~text
seq_len = 5

       root / bonus
        /       \
       A         B
       |
       C
~~~

测试得到 positions：

~~~text
root → 5
A    → 6
B    → 6
C    → 7
~~~

A 和 B 虽然是不同 Tensor row，却代表同一语言模型位置上的两种互斥未来，所以共享 position 6。

如果简单用普通 lower-triangular causal mask，把节点 flatten 为：`[root, A, B, C]`

那么 B 会错误地看到 A，C 也可能错误地看到 B。

Tree Mask 必须表达的逻辑是：

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

固定测试给出的 QLEN-only mask 正是：

~~~text
          root A B C

root       1   0 0 0
A          1   1 0 0
B          1   0 1 0
C          1   1 0 1
~~~

来源：

[`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)

### QLEN_ONLY：只显式保存 speculative query block

`QLEN_ONLY` 的 shape 是：`[bs × draft_token_num × draft_token_num]`

它只描述：

> **Verify Tree 内部的节点之间谁能看谁。**

也就是上面那个 4×4 ancestor mask。

Committed Prefix 并不作为列出现在这个 buffer 里。

### FULL_MASK：把 Prefix 可见性和 Tree ancestor block 放在同一个布局里

`FULL_MASK` 的总长度对应每个请求：

~~~text
draft_token_num
×
(seq_len + draft_token_num)
~~~

左侧：`seq_len`

列表示已经提交的历史 Prefix。

这些列对所有 Tree node 都应该可见。

右侧：`draft_token_num`

列才是刚才那块 ancestor-only Tree Mask。

概念上：

~~~text
Committed Prefix      | Verify Tree
──────────────────────┼────────────────
t0 t1 t2 t3 t4        | root A B C

root  ✓ ✓ ✓ ✓ ✓       | 1 0 0 0
A     ✓ ✓ ✓ ✓ ✓       | 1 1 0 0
B     ✓ ✓ ✓ ✓ ✓       | 1 0 1 0
C     ✓ ✓ ✓ ✓ ✓       | 1 1 0 1
~~~

CPU reference 也明确说明：

~~~text
QLEN_ONLY:
只包含 qlen × qlen

FULL_MASK:
包含 seq_len prefix +
draft_token_num × draft_token_num tree block
~~~

固定源码：

[`spec.cpp / build_tree_kernel_efficient_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)

### 不要把它简单写成“CPU 用 QLEN，GPU/NPU 用 FULL”

`default_tree_mask_mode()` 的默认值确实是：

~~~python
CPU      → QLEN_ONLY
非 CPU   → FULL_MASK
~~~

但 Runtime 还有第二层选择。

`build_eagle_verify_input()` 会查询 Target attention backend：

~~~python
verify_mask = target_attn_backend.verify_mask
~~~

如果 backend 自己提供 VerifyMask：

~~~python
mask_mode = verify_mask.mode
~~~

会覆盖 worker 的默认模式。

`VerifyMask` 甚至可以在 `is_read=False` 时主动选择 QLEN_ONLY，只提供一个供 Tree Builder 写入、但 backend 不必读取完整 Prefix mask 的固定 buffer。

固定源码：

- [`build_eagle_verify_input()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L317-L404)
- [`verify_mask.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/layers/attention/verify_mask.py)

因此两种模式边界应该写成：

> **它们表达相同的 Tree ancestor 语义；区别主要在于 Mask buffer 是否显式携带 committed-prefix columns，以及具体 attention backend 怎样消费或重建 Prefix 部分。**

这比按硬件平台硬分更准确。

## 四、Target 预测怎样确定接受路径

Tree 构建完成后，`build_eagle_verify_input()` 先把上一轮 Target 确认的 `bonus token` prepend 到 Draft nodes：

~~~python
draft_tokens = torch.cat(
    (
        bonus_tokens.unsqueeze(1),
        draft_tokens,
    ),
    dim=1,
)
~~~

所以 Target Verify 看到：`[root/previous bonus, draft nodes...]`

随后：`ForwardMode = TARGET_VERIFY`

完整 Target Model 对所有 Tree rows 一次 Forward。

由于每个 row 的 Tree Mask 不同，Target 得到的是：

~~~text
Target(root | committed prefix)

Target(A | prefix + root)

Target(B | prefix + root)

Target(C | prefix + root + A)
...
~~~

然后 Greedy 模式进入：

~~~python
verify_tree_greedy_func(...)
~~~

Draft 的累计分数用于筛选候选。进入 Greedy Verify 后，接受路径由 Target 在各节点的预测逐步确定，并不按整条 root-to-leaf 路径的概率重新排名。

### Tree 被转换成 first-child / next-sibling

Tree Builder 输出：

~~~text
retrieve_index
retrieve_next_token
retrieve_next_sibling
~~~

对于：

~~~text
       root
      /    \
     A      B
     |
     C
~~~

固定测试得到：

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

也就是：

~~~text
root.first_child = A
A.next_sibling  = B
A.first_child   = C
~~~

固定测试：

[`test_build_tree_topk2_hand_case()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/test/registered/cpu/test_spec_kernels.py#L524-L562)

### Greedy kernel 从 root 开始，用 Target prediction 匹配 child

CPU 与 Triton reference 的算法语义一致。

初始：`last_accept = root`

下一层先进入：

~~~python
cur = retrieve_next_token[last_accept]
~~~

也就是 root 的第一个 child。

然后读取：

~~~python
target_tok = target_predict[last_accept]
~~~

如果 child token：`== Target 在 parent row 上预测的 token`

则接受。

如果不匹配：

~~~python
cur = retrieve_next_sibling[cur]
~~~

沿 sibling 链继续找。

找到匹配 child 后：`last_accept = child`

再进入它的下一层。

如果整个 sibling list 都没有 Target 想要的 Token：`停止 Draft-path acceptance`

固定源码：

- [`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/spec_tree.py)
- [`verify_tree_greedy_cpu()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/aot/csrc/cpu/spec.cpp)

所以 Draft score 和 Target decision 的职责完全不同：

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

EAGLE 里“bonus”这个词很容易因为时序不同而混乱。

这一轮 Target Verify 开始前，输入 Tree 的 root 是：`上一轮 Target 已确认的 bonus token`

它已经是当前序列下一次模型执行的输入，但它的 Target KV 要在本轮 Verify forward 中计算。

假设 Verify Tree：

~~~text
root = R

       R
      / \
     A   B
     |
     C
~~~

Target 的 Greedy prediction：

~~~text
at R → A
at A → C
at C → X
~~~

Kernel 首先发现：`A == Target(R)`

于是接受 Draft A，并写：`predict[row(R)] = A`

下一层：`C == Target(A)`

于是接受 Draft C：`predict[row(A)] = C`

到 C 后没有匹配 child，Kernel 最后执行：

~~~python
predicts[last_accept_index]
=
target_predict[last_accept_index]
~~~

于是：`predict[row(C)] = X`

固定源码：

[`verify_tree_greedy_kernel_triton()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/kernels/ops/speculative/spec_tree.py#L176-L281)

因此本轮返回的生成结果是：

~~~text
A
C
X
~~~

其中：

~~~text
A、C
=
被 Target 验证通过的 Draft tokens

X
=
Target 在最后一个 accepted input node 上直接预测出来的
new trailing bonus
~~~

这时必须区分两种 bonus：

~~~text
Verify 输入的 root R
=
previous-round bonus
=
本轮作为模型输入，KV 在本轮被物化


本轮最后得到的 X
=
new trailing bonus
=
只是 Target logits 的输出
=
尚未作为模型输入运行，因此还没有 KV
~~~

`eagle_sample()` 内部的 `num_correct_drafts` 始终只数接受了多少 Draft nodes。

返回前才：

~~~python
return (
    predict,
    num_correct_drafts + 1,
    accept_index,
)
~~~

这个：`+1`

表示输出 token 数还包括新的 Target trailing bonus。

~~~text
accept_lens
=
accepted Draft token count
+
1 trailing Target token
~~~

固定源码：

[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_utils.py#L731-L1029)

这和下一节的 KV compaction 直接相关。

## 六、接受路径对应哪些输入 KV

多分支 Verify 结束后，物理 KV 仍然按照整棵 Verify Tree 的 layout 存放。

例如：

~~~text
Verify input slots:

slot0 → root R
slot1 → A
slot2 → B
slot3 → C
~~~

而 Target 接受的是：`R → A → C`

后续 Runtime 希望已提交的输入历史重新成为连续 Chain。

于是 `run_eagle_verify()` 在：`topk > 1`

时调用：

~~~python
_finalize_accept_tree_path(...)
~~~

里面第一步：

~~~python
move_accept_tokens_to_target_kvcache(
    batch,
    accept_index,
    accept_lens - 1,
    token_to_kv_pool_allocator,
)
~~~

固定源码：

[`_finalize_accept_tree_path()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L407-L459)

为什么传的是：`accept_lens - 1`

这里先将接受长度转换为正确 Draft 数，再由函数补上旧 Root 对应的输入行。

`move_accept_tokens_to_target_kvcache()` 的第三个参数名字就是：`num_correct_drafts`

函数契约明确：

~~~text
num_correct_drafts:
正确 Draft 数量，不含 trailing bonus
~~~

但函数内部为每个请求准备的 committed KV 长度是：

~~~python
num_correct_drafts + 1
~~~

因为本轮 Verify forward 实际已经物化 KV 的输入节点包括：

~~~text
previous bonus/root
+
accepted Draft input nodes
~~~

所以调用链是：

~~~text
accept_lens
=
accepted_drafts + new_trailing_bonus


accept_lens - 1
=
accepted_drafts
        ↓
传给 KV mover


KV mover 内部 +1
=
root/previous-bonus KV
+
accepted Draft-node KV
~~~

固定源码：

[`move_accept_tokens_to_target_kvcache()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L728-L790)

**不会被搬 KV**的是：`new trailing bonus`

因为它只是这轮 Target logits 的预测结果，还没作为输入执行，自然不存在本轮可搬的 KV row。

如果前面例子：

~~~text
input tree:
R, A, B, C

accepted drafts:
A, C

new trailing bonus:
X
~~~

那么：

~~~text
accept_lens = 3
accept_lens - 1 = 2 accepted drafts
~~~

KV mover 实际整理的输入历史是：

~~~text
R
A
C
~~~

共：`2 + 1 = 3`

个 KV rows。

而 X 要等下一轮成为模型输入时才写 KV。

### Compaction 不只整理 KV

`_finalize_accept_tree_path()` 随后还会把：

~~~text
predict
Target hidden_states
~~~

按 `accept_index` gather 到每个请求 block 的前部。

源码注释写得非常明确：

> downstream chain-layout code 假设 accepted path 位于 contiguous front。

因此 Tree speculative decoding 会经历：

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

Rejected branch B 的 KV 仍可能暂时留在 overshoot slots，但不再属于 committed sequence，后续会被释放或复用。

所以 Tree 的生命周期是暂时的。

模型最终历史永远重新收敛成一条：`autoregressive chain`

## 七、DeepSeek-V4 的模型与后端限制

最后回到 DeepSeek-V4 + Ascend。首先要检查模型级限制，而不只是后端 allowlist。同一固定版本的 [`deepseek_v4_hook.py`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/deepseek_v4_hook.py) 在 EAGLE 配置下直接断言 `speculative_eagle_topk == 1`。因此，前面的多分支树解释的是通用 EAGLE Runtime；当前 DeepSeek-V4 配置只开放 Chain。

固定版本官方配置：

~~~bash
--device npu
--attention-backend dsv4
--page-size 128

--speculative-algorithm EAGLE
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
~~~

所以官方实际使用的是：`topk = 1`

即 Chain。

固定入口：

[DeepSeek-V4-Flash Ascend Tutorial](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash.mdx)

而 `_handle_eagle_family()` 有一个非常具体的 guard：

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

固定源码：

[`_handle_eagle_family()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/arg_groups/speculative_hook.py#L1067-L1095)

~~~text
dsv4
+
page_size = 128
+
topk > 1
~~~

会在启动阶段直接被拒绝。

### 这个 Guard 不是在说“NPU 不支持 Tree”

NPU 路径已经有：`torch.ops.npu.build_tree_kernel_efficient`

用于构造 Tree topology / mask，也有：`sgl_kernel_npu.sample.verify_tree_greedy`

用于 Greedy tree traversal。

DeepSeek-V4 Ascend 还拥有：`DeepseekV4AscendMultiStepDraftBackend`

~~~text
Tree Builder
Greedy Tree Verify
Multi-step Draft
~~~

这些组件并不是完全不存在。

Guard 限制的是更具体的一层：

> **当 Draft Tree 有多个 branch 且 KV allocator 使用 page_size>1 时，Draft Decode 如何让不同 branch 正确共享 Prefix，同时又在自己的 paged KV 区域写入未来 Token。**

源码注释称这一能力为：`two-pass cascade draft-decode`

需要：

~~~text
shared prefix pass
+
per-branch expand pass
+
prefix-tail duplication
~~~

原因可以用一个 page 示意理解。

假设 Prefix 最后一个 page 只填了一部分：

~~~text
Prefix tail page:

[P P P _ _ _ _ _]
~~~

现在 A、B 两个 branch 都要从这个 Prefix 继续写。

对于 page-size > 1 的 paged cache，两个 branch 各自的第一页都必须拥有正确的 Prefix tail：

~~~text
branch A page:
[P P P A1 A2 ...]

branch B page:
[P P P B1 B2 ...]
~~~

所以需要把 Prefix partial-tail KV 复制到每个 branch 的第一页空洞中。

SGLang 已有专门 helper：

~~~python
duplicate_prefix_tail_to_draft_branches(...)
~~~

源码注释：

~~~text
Copy the prefix partial-tail page
into each branch's first-page holes
(page>1 + topk>1)
~~~

固定源码：

[`duplicate_prefix_tail_to_draft_branches()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/eagle_worker_common.py#L59-L104)

当前 guard 只允许：

~~~text
flashinfer
fa3
triton
~~~

走这套 paged-tree draft-decode 逻辑。

`dsv4` 没在 allowlist 里，因此固定版本不允许该组合。

### 将模型约束与页布局一起检查

设置 `page_size=1` 后，paged-tree guard 的触发条件消失，DeepSeek-V4 模型 Hook 的 `topk=1` 约束仍然生效。因此，该固定版本的 V4 EAGLE 配置使用 Chain。

扩展多分支能力时，需要同时处理模型约束、分支 KV 归属、Prefix Tail 复制、Attention Metadata 与后续 Compact。通用 NPU Tree Build / Greedy Verify Kernel 已提供部分组件，完整模型路径再将这些组件衔接起来。

候选分数决定哪些路径进入 Verify，树索引与 Mask 保持父子关系和可见性，接受结果最后收敛为一条输出序列。这些关系与物理缓存布局一起，构成多分支执行的完整数据流。

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
