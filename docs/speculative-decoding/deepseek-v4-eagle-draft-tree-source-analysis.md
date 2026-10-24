# EAGLE 候选树：从候选池到 Verify 与接受路径

链式 Draft 把预算放在同一条候选路径上，中途预测错误后，后续候选便无法继续接受。通用 EAGLE Runtime 还可以保留多个分支，把有限的候选预算分给几条较有希望的路径，再交给 Target 一次验证。

把几条候选一起送进 Target，困难在于保持每条分支的上下文：同一层的两个节点可以共享位置，却不能互相读到对方的未来。本文沿候选搜索、索引重建、Attention Mask 和接受后的缓存整理走完一棵小树。DeepSeek-V4 在固定版本中仍要求 EAGLE `topk=1`；多分支部分用于解释通用 Runtime，最后再核对 V4 的模型和后端限制。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 5c69e32abe013fa1b913022682a3104c79105f37`，核对日期为 2026-09-20。正文分析 SGLang EAGLE Runtime 中真实存在的 `topk>1` Draft Tree 机制，并以 DeepSeek-V4 / NextN 为模型背景。当前 DeepSeek-V4 + Ascend 官方部署使用 `--attention-backend dsv4`、`page_size=128`、`speculative_eagle_topk=1`；同一版本的 DeepSeek-V4 模型配置 hook 对 EAGLE 直接要求 `topk=1`；另有 `dsv4 + page_size>1 + topk>1` 的后端 guard。因此本文讲的是 DeepSeek-V4 所使用的 EAGLE Runtime 的 Tree 能力，以及它为什么在当前 Ascend DSV4 官方配置中退化成 Chain，而不是宣称当前 Ascend DSV4 已经开放多分支 Tree 生产配置。

</details>

## 一、在有限预算内保留候选路径

先看链式候选：NextN 依次提出 `A → B → C`，但 Target 在接受 `A` 后预测的是 `X`。分歧从 `B` 开始，`B` 和后面的 `C` 都不能进入接受前缀；哪怕某个更后面的 token 恰好匹配，也不能越过分歧继续接受。

`topk>1` 保留的是同一上下文下的多个候选。下面用 `topk=2` 的教学例子追踪它们，第一步 Draft Extend 得到：

~~~text
A   p=0.60
B   p=0.40
~~~

`select_top_k_tokens()` 第一步把这两个 candidate 作为下一次 NextN 的输入，并将父 hidden state 沿 batch 行复制到两个 branch。它们起点相同，接下来会分别以 `A`、`B` 为输入继续计算；这一步张量复制本身还没有解决分支 KV 的分配与写入。以下是省略周围逻辑的源码片段：

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

第二层先为两个 branch 各生成候选，再乘上父路径分数，得到累计路径 score：

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

这里保留的是继续展开的两个 frontier 节点，四条 child path 不会全部进入下一轮模型计算。但本轮四个 child 的分数与 token 仍记入候选池，最终 Verify 的全局裁剪还会使用它们。因此，“没有继续展开”与“不能进入最终 Verify”是两个不同决定。

固定源码：

[`_select_top_k_tokens_later()`](https://github.com/sgl-project/sglang/blob/5c69e32abe013fa1b913022682a3104c79105f37/python/sglang/srt/speculative/spec_utils.py#L332-L366)

这里的 score 来自 Draft 自己的概率。在当前非 rejection-sampling 的 EAGLE Tree 路径中，它沿 frontier 递推，表示候选路径的累计概率质量；它不是 Target probability，也不是额外 Confidence Head 的预测：

~~~text
score(child)
=
score(parent)
×
draft_probability(child | parent)
~~~

累计分数用于分配搜索和 Verify 预算。例子中，`B → E` 的条件概率 0.80 高于 `A → C` 的 0.70，但乘上父路径概率后，`A → C` 的 0.42 仍高于 `B → E` 的 0.32。只比较最后一跳会得到不同的排序；无论哪种排序，都不能代替后面 Target 的接受决定。

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

这里减去的 1 留给 Verify Window 的 root，也就是上一轮已经由 Target 确认的 bonus token。该 token 已经成为输出，但本轮仍要把它作为模型输入；总窗口为 `num_draft_tokens` 行时，Draft nodes 只能占 `num_draft_tokens - 1` 行。

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

这两次裁剪限制的是不同阶段：层内 top-k 控制下一轮继续算哪些节点，全局 top-k 控制最终送给 Target 的节点数。它不会展开一棵完整 k 叉树，`topk` 也不能直接当成最终 Verify 节点总数。

## 二、候选池索引怎样还原父节点关系

树图直接画出了父子关系，源码却要通过多组索引还原它。接下来分别看每组索引对应哪个候选池。

`organize_draft_results()` 最后返回：

~~~text
parent_list
top_scores_index
draft_tokens
~~~

这里的 `top_scores_index` 传给 Tree Builder 后叫 `selected_index`。它保存的是节点在原始展平候选池中的位置，不是压缩后 Verify Tree 的 node id。后面先按它找回 token，再用它追溯 parent。

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

会继续排到候选池后面：`A、B` 占索引 0、1，第二层 `AC、AD、BE、BF` 占索引 2、3、4、5。这里 `AC` 表示路径末端的候选 `C`，不是一个拼接后的 token；它的父路径通过另一组 metadata 保存。

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

以总 Verify 窗口 4 行为例，root 之外只能保留 3 个 Draft 节点。前面的分数中，最高三个是 `A=0.60、AC=0.42、B=0.40`。源码选中它们以后又按原池索引排序，得到 `selected_index=[0,1,2]`，gather 出的 `draft_tokens` 因而是 `[A,B,C]`，不是按分数顺序排列的 `[A,C,B]`。排序改变了存放顺序，没有重新打分。

### 那 `parent_list` 又是什么

只知道某个节点来自候选池的 index 17 仍然不够，Tree Builder 还要找到它的 parent。对一个最终被保留的节点，需要从原候选池身份，转换到最终 Verify Tree 中的父节点编号。

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

`parent_list` 保存候选组与原池父候选之间的映射，不是最终 Tree 的父节点编号数组。最终父节点编号要等全局裁剪后，在 `selected_index` 中再查一次才能得到。

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

把测试中的 `C` 走一遍：它来自候选池索引 2，`2 // topk = 1`，于是查 `parent_list[1]=0`，找到原池中的父候选 `A`。原池索引 0 在 `selected_index` 中排第 0，加上 root 占据的一行，`A` 就是最终 node 1；`C` 排第 2，加上 root 后是 node 3，因此 node 3 的 parent 是 node 1。这里 `/ topk` 在 CPU C++ 中是整数除法，Python 推演应按 `//` 理解。

三组索引由此衔接：`selected_index` 保留原池身份，`parent_list` 找回原池 parent，`retrieve_*` 才记录重建后可遍历的最终树。不能拿前一组索引直接去索引后一阶段的数组。

## 三、把树关系转成 Verify Mask

最终 Tree 结构确定以后，需要把它转换成一次 Target Forward 可以执行的 Attention Layout。

下面保留调用与索引的关键片段，用于对照源码，不是可独立执行的完整 Python 程序。入口为：

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

这里 `seq_len=5` 表示已有 5 个历史位置，按零起始编号，root 使用 position 5。`A、B` 是两个不同 Tensor row，但位于同一深度，代表同一个序列位置的两种互斥未来，因此都使用 position 6；`C` 是 `A` 的 child，使用 position 7。position 编号不能代替 row 编号或分支身份。

如果简单用普通 lower-triangular causal mask，把节点 flatten 为：`[root, A, B, C]`

普通下三角 mask 会让后面的 row 看到前面所有 row，于是 `B` 能读到互斥分支 `A`，`C` 也会混入 `B`。Target 得到的已不再是原候选路径的上下文，这种错误不是最后只选一条输出就能修正的。

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

它只描述 Verify Tree 内部的可见性，也就是上面的 4×4 ancestor mask。Committed Prefix 不作为列出现在这个 buffer 中；后端仍需要让每行读到正确历史，只是 Prefix 的表示由后端另行处理，不能把“buffer 里没画历史列”理解成“Attention 不读历史”。

### FULL_MASK：把 Prefix 可见性和 Tree ancestor block 放在同一个布局里

`FULL_MASK` 的总长度对应每个请求：

~~~text
draft_token_num
×
(seq_len + draft_token_num)
~~~

左侧 `seq_len` 列表示已经提交的历史 Prefix，对每个 Tree node 可见；右侧 `draft_token_num` 列表示本轮 Tree，按 ancestor 关系控制可见性。对例子里的 5 行历史和 4 个 Tree node，这是每请求 `4×(5+4)=36` 个 mask 元素；QLEN_ONLY 则只显式保存右侧的 `4×4=16` 个元素。

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

两种模式要表达同样的分支可见性，主要区别在于 buffer 是否显式携带历史列，以及后端怎样处理 Prefix。检查实际路径时，应看后端选出的 `VerifyMask.mode` 和 `is_read`，而不只看 CPU、GPU 或 NPU 的平台默认值。

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

Target Verify 因而看到 `[root/previous bonus, draft nodes...]`，并在 `ForwardMode=TARGET_VERIFY` 下对这些输入行执行一次完整 Target Forward。树节点在物理上排成一段数组，Mask 和 position 让它们在逻辑上仍属于各自的路径。

由于每个 row 的 Tree Mask 不同，Target 得到的是：

~~~text
Target(root | committed prefix)

Target(A | prefix + root)

Target(B | prefix + root)

Target(C | prefix + root + A)
...
~~~

这里每一行写的是“以该节点作为当前输入后得到的 Target 预测”。例如 `A` 行能读到 Prefix、root 和自身 `A`，预测的是 `A` 后面的 token；它不是重新计算是否接受输入 `A` 的单 token 概率。随后 Greedy 模式进入：

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

CPU 与 Triton reference 都从 root 开始，以父行的 Target 预测匹配下一层 child。初始 `last_accept=root`，先进入它的第一个 child：

~~~python
cur = retrieve_next_token[last_accept]
~~~

也就是 root 的第一个 child。

然后读取：

~~~python
target_tok = target_predict[last_accept]
~~~

child token 等于这个 `target_tok` 就接受；不匹配则查下一个 sibling：

~~~python
cur = retrieve_next_sibling[cur]
~~~

找到匹配 child 后，更新 `last_accept=child`，再从该节点的 first child 继续。若整条 sibling 链都没有匹配项，就结束 Draft-path acceptance；当前父行的 Target 预测仍可作为末尾 token 返回。这个过程沿实际上下文逐层推进，不会从另一条已分叉路径跳回一个高分叶子。

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

同一个 token 会在相邻两轮扮演不同角色。上一轮末尾产生的 bonus `R`，本轮成为 Tree root；它已经是正式输出，本轮不重复向用户输出，但要作为模型输入计算自己的 Target KV。本轮末尾新产生的 `X` 则要等下一轮才有自己的 KV。

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

Kernel 先在 `R` 行发现预测 `A`，匹配 child `A`，写入 `predict[row(R)]=A`；再在 `A` 行发现预测 `C`，匹配 child `C`，写入 `predict[row(A)]=C`。到 `C` 后没有可匹配的 child，Kernel 将它自己的 Target 预测作为末尾输出：

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

`eagle_sample()` 内部的 `num_correct_drafts` 只数接受的 Draft nodes，本例是 `A、C` 两个。返回时才将长度加 1，计入新 bonus `X`：

~~~python
return (
    predict,
    num_correct_drafts + 1,
    accept_index,
)
~~~

所以接受 2 个 Draft 时，返回的有效输出长度是 3；即使没有候选匹配，也会有 root 行产生的一枚 Target token。

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

本例选中的输入路径是 `R → A → C`，对应的 slot 是 `0、1、3`，中间夹着未选中的 `B`。后续模块按连续 Chain 消费有效状态，因此 `run_eagle_verify()` 在 `topk>1` 时调用下面的整理函数：

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

这里传 `accept_lens-1`，是先把输出长度还原成正确 Draft 数。`move_accept_tokens_to_target_kvcache()` 的第三个参数名为 `num_correct_drafts`，内部再补上 root 对应的输入行。前者扣掉新输出 bonus，后者补回旧输入 root；数值上减 1 再加 1，不表示两个 token 拥有相同的 KV 状态。

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

新 trailing bonus 不参与这次 KV 搬移：它只是本轮 logits 的预测结果，没有作为本轮输入执行，因此没有对应的 KV row。搬移依据仍是所选 Verify 输入行，不能直接拿新输出 token 列表去找缓存位置。

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

共 `2+1=3` 个 KV rows。`accept_index` 的有效前缀在本例是 `[0,1,3]`，读输入行得到 `[R,A,C]`，读同一行的 `predict` 则得到 `[A,C,X]`。同一组索引服务于两种数组，这正是输入状态与输出 token 相差一位的原因。

### Compaction 不只整理 KV

`_finalize_accept_tree_path()` 随后还会把：

~~~text
predict
Target hidden_states
~~~

按 `accept_index` gather 到每个请求 block 的前部。

下游 chain-layout 代码假设接受路径已经位于每请求 block 的连续前部。因此，KV 搬对还不够：若 `hidden_states` 存在，它也要和预测结果一起按相同行索引 gather，否则下一轮可能拿着正确 token 配上另一分支的 hidden state。

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

未接受分支 `B` 的 KV 仍可能暂时留在 overshoot slots，但不属于 committed sequence，后续会释放或复用。buffer 的分配宽度也不等于有效接受长度：`accept_index` 中的无效填充值可能参与形状固定的 gather，下游只能消费 `accept_lens` 指定的有效前缀。树用于临时探索，单条请求继续生成时只沿所选的 autoregressive chain 使用状态。

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

这里 `topk=1`，每个位置只有一个后继候选，因此实际使用的是 Chain。上面是固定教程中的相关参数节选，不是包含模型路径、服务地址等参数的完整启动命令。

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

NPU 路径已经有 `torch.ops.npu.build_tree_kernel_efficient` 用于构造 topology 与 mask，也有 `sgl_kernel_npu.sample.verify_tree_greedy` 用于 Greedy tree traversal。DeepSeek-V4 Ascend 还具有 `DeepseekV4AscendMultiStepDraftBackend`，提供多步 Draft 所需的后端组织。

~~~text
Tree Builder
Greedy Tree Verify
Multi-step Draft
~~~

这些入口说明 NPU 侧存在相应组件，不代表它们已经拼成可用于当前 V4 配置的多分支路径。paged-tree guard 检查的是另一层约束：多个 branch 共享历史 Prefix，同时分别写入未来 token 时，分页 KV 的归属和 Prefix tail 必须正确。即使 Tree Builder 和 Greedy Accept 可用，也不能跳过这个分配问题。

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

如果两个 branch 继续写同一个物理尾页，未来位置会互相覆盖；若只给它们空白新页，又会丢失该页中已有的 Prefix tail。因此这里需要为每个 branch 准备正确的第一页，并把原 Prefix partial-tail 的 KV 放到对应位置，再分别追加 `A1、A2` 或 `B1、B2`。这个 8 格页是教学示意，不是修改 V4 的 `page_size=128` 配置。

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
