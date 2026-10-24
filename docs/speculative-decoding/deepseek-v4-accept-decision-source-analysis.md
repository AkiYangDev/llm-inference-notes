# 投机解码 Accept：从 Verify 行到最终输出路径

Target Verify 为候选位置算出 logits 后，系统还要决定哪些候选可以成为正式输出。Greedy 路径沿 Target 的预测寻找匹配前缀；Sampling 路径则按各自的概率与接受规则推进。

本文用输入 R、A、C 对应输出 A、C、X 的小例子追踪接受结果：`accept_index` 保存的是 Verify 行索引，输出中的新 bonus X 与这些行已经计算出的 KV 相差一位。先讲清这个映射，再展开树遍历、不同 Sampling 算法和 DSpark 的链式接受。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> **源码与适用范围**：本文基于 `sgl-project/sglang @ 791c7850d0960fd768102f71e7d999b036bb75ba`，核对日期为 2026-09-20。EAGLE 部分分析 SGLang Runtime 的 tree accept；DSpark 部分分析当前 DeepSeek-V4 的 chain accept。GPU JIT kernel 的内部逻辑可以直接从 SGLang 仓库核对；NPU 路径在 SGLang 中调用 `sgl_kernel_npu` 的同名算子，本文只把当前 SGLang 调用契约视为已确认，不把外部 NPU kernel 的未展开实现写成已经逐行验证的事实。

</details>

## 一、Target 怎样确认候选前缀

先固定一个贯穿全文的小例子。上一轮 Target 已经产生了一个 bonus token `R`，这一轮把 `R` 作为 Verify Tree 的 root。Draft 保留了下面几种未来：

~~~text
                 R
              /     \
             A       B
           /   \
          C     D
~~~

假设 Target Verify 的 greedy prediction 是：

~~~text
Target(R) = A
Target(A) = C
Target(C) = X
~~~

最终这一轮对外输出的是：`A → C → X`

其中 `A、C` 是 Draft 提出且被 Target 接受的候选，`X` 是 Target 在最后一个 accepted input node `C` 上直接产生的新 token，也就是这一轮的 terminal bonus。

这里首先要和 Draft 阶段的 path score 分开。EAGLE 在 rollout 时会用累计 Draft probability 给候选未来排序，从有限 token budget 中决定“哪些 node 值得进入 Target Verify”；进入 Verify 后，**Target 的 prediction / probability、Tree topology 与当前 sampling policy** 共同决定接受路径。

因此可以把两层职责写得很清楚：

| 阶段 | 决定什么 | 主要依据 |
| --- | --- | --- |
| Draft search | 哪些未来值得花 Verify budget | Draft score / candidate pruning |
| Accept decision | 哪些未来真的成为历史 | Target result + topology + sampling rule |

候选排序分配验证预算，最终接受则按 Target 结果与当前采样规则决定。

## 二、接受前先把父节点关系变成可遍历结构

父节点关系在 Tree Build 阶段就转换为可遍历结构，Accept Kernel 随后按这个结构从 Root 寻找匹配 Child。

Draft rollout 结束后，`parent_list + selected_index` 先交给 Tree Builder。Builder 把不方便设备端逐层搜索的 parent metadata 转换成三组直接可遍历的数据：

~~~text
retrieve_index
retrieve_next_token
retrieve_next_sibling
~~~

它们可以近似理解为：

~~~text
retrieve_index[node]
    = 这个 Tree node 对应哪个 flattened Verify row

retrieve_next_token[parent]
    = parent 的 first child

retrieve_next_sibling[child]
    = 当前 child 的 next sibling
~~~

对应关系是：

~~~mermaid
flowchart TD
    A["parent_list + selected_index"] --> B["Tree Builder"]
    B --> C["retrieve_index"]
    B --> D["retrieve_next_token"]
    B --> E["retrieve_next_sibling"]
    C --> F["Target Verify / Accept"]
    D --> F
    E --> F
    F --> G["accept_index"]
~~~

固定源码入口：

- [`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)
- [`sgl_build_tree_kernel_efficient_triton()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py)
- [`build_tree_efficient` / `VerifyTreeGreedy`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh)

为什么要先变成 first-child / next-sibling？因为 Accept 的自然方向是：

~~~text
root
  ↓ Target(root) 决定下一步
child
  ↓ Target(child) 决定下一步
child
  ↓
...
~~~

而不是先知道叶子是谁，再向上找 parent。换句话说，所谓 Parent Tree Recovery 更准确地说是：

> **先把 Draft search 的 parent metadata 恢复成面向 Verify / Accept 的 forward-traversal topology，然后 Target 从 root 开始一层层选择真实路径。**

`parent_list` 与 `selected_index` 用于重建树，`accept_index` 再从 Verify 行中选出接受路径。这三组索引按各自的输入、输出空间衔接。

## 三、Greedy Accept 的行索引与输出

Greedy 路径最适合把这个索引语义看透。`eagle_sample()` 先对 Target logits 做 argmax，然后进入 `verify_tree_greedy_func()`：

[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

[`VerifyTreeGreedy`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh)

核心逻辑可以压缩成：

~~~python
current = root
accept_index[0] = row(root)

while depth remains:
    child = first_child(current)

    while child exists:
        if candidate(child) == target_predict(current):
            predict[row(current)] = candidate(child)
            accept_index.append(row(child))
            current = child
            break
        child = next_sibling(child)

    if no child matched:
        break

predict[row(current)] = target_predict(current)  # terminal Target token
~~~

仍然使用开头的小树。假设 flattened Verify rows 是：

~~~text
row 0 → R
row 1 → A
row 2 → B
row 3 → C
row 4 → D
~~~

Target 依次希望 `A、C、X`。Kernel 的结果不是：`accept_index = [1, 3]`

而是：`accept_index = [0, 1, 3]`

同时：

~~~text
predict[row 0] = A
predict[row 1] = C
predict[row 3] = X
~~~

~~~python
predict[accept_index] == [A, C, X]
~~~

这说明 `accept_index` 最准确的心智模型不是“被接受的 Token 下标”，而是：

> **accepted-output-producing Verify row indices。**

也就是“最终线性输出由哪些 Verify input row 的 prediction 产生”。

这一点会直接解释 `accept_lens` 的 `+1`。`VerifyTreeGreedy` 内部的 `accept_token_num` 只统计匹配成功的 Draft nodes；`eagle_sample()` 返回前再执行语义上的：`accept_lens = num_correct_drafts + 1`

例子里：

~~~text
num_correct_drafts = 2    # A, C
accept_lens        = 3    # A, C, X
~~~

固定源码在 [`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py) 中明确保留了“`num_correct_drafts` 在函数内部始终只表示 drafts-only，返回值才 +1”的注释。

### 最容易写错的一格：输出 `[A, C, X]`，本轮新物化 KV 的却是 `[R, A, C]`

将本轮输入与输出并排，可以看到：

~~~text
Verify input row      prediction / output      本轮是否已有 KV
────────────────────────────────────────────────────────
R                     A                        R 的 KV 已物化
A                     C                        A 的 KV 已物化
C                     X                        C 的 KV 已物化

最终对外输出：         A, C, X
本轮可提交的 Verify KV：R, A, C
下一轮 root：          X
~~~

也就是说，terminal bonus `X` 是本轮 Target logits 的输出，但还没有作为模型输入跑过，因此 **本轮并不存在 `X` 的 KV**。它会在下一轮成为 root，再被送入模型并物化 KV。

这也是为什么同一个 `accept_index=[row(R), row(A), row(C)]` 可以同时服务两个看似“错一位”的世界：

~~~text
用于 output gather：
predict[accept_index] → [A, C, X]

用于 Verify-row / KV gather：
rows[accept_index]    → [R, A, C]
~~~

上一篇 KV 文章中的 accepted-path relocation 正是依赖这个映射，而不是去寻找一个并不存在的 `row(X)`。

`run_eagle_verify()` 随后做：

~~~python
accept_tokens = predict[accept_index]
~~~

再由 `fill_bonus_tokens()` 取每个 request 的 `accept_len - 1` 位置作为下一轮 bonus：

[`fill_bonus_tokens()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/eagle.py)

所以 Greedy Tree Accept 的闭环可以写成：

~~~mermaid
flowchart TD
    A["Target prediction at root"] --> B["scan first child / siblings"]
    B --> C["match child"]
    C --> D["append child row to accept_index"]
    D --> E["Target prediction at accepted child"]
    E --> B
    B -->|no matching child| F["write terminal Target prediction"]
    F --> G["predict[accept_index]"]
    G --> H["accepted drafts + bonus"]
~~~

## 四、区分两类 Sampling 接受规则

Greedy 只需要问“candidate 是否等于 Target argmax”。一旦开启 temperature、top-k 或 top-p，Accept 就必须处理概率分布。

这里当前源码存在两套容易被混写的路径。

### Classic Rejection Sampling 的 Draft 概率 `q`

经典 speculative sampling 设 Draft proposal distribution 为 `q(x)`，Target distribution 为 `p(x)`。Draft 先从 `q` 提出 token `x`，接受概率为：

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

实现中不必真的计算除法。采样 `u ~ U(0,1)`，判断：

$$
u\,q(x)<p(x)
$$

当前 SGLang chain kernel 就是这个语义：

[`reject_sampling.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/reject_sampling.py)

如果 candidate 被拒绝，最终 token 不直接重新从 `p` 采，而从 residual distribution：

$$
r(y) \propto \max(p(y)-q(y),0)
$$

采样。这样才能把 Draft 多分配的概率质量纠正回 Target 分布。

例如：

~~~text
          A      B      C
Draft q   0.6    0.3    0.1
Target p  0.3    0.4    0.3
~~~

Draft 提出 `A` 时，接受概率只有 `0.3 / 0.6 = 0.5`。如果拒绝，residual 为：

~~~text
max(p-q, 0)

A: 0
B: 0.1
C: 0.2
~~~

归一化后再采，所以刚刚被判定“Draft 给得过多”的 `A` 不会从 residual branch 又被原样抽回来。

这也是为什么开启 classic rejection sampling 后，SGLang 会检查 `draft_probs` 是否存在、词表维度是否与 Target 一致：没有 proposal `q`，就无法执行 `p/q` 语义。

### Target-only Tree Sampling：沿 sibling 累加 Target mass，但不是 classic `p/q` rejection

EAGLE tree 还有另一套 sampler：

[`TreeSpeculativeSamplingTargetOnly`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh)

它同样沿 `first child → next sibling` 遍历，但核心 accept condition 使用的是当前 siblings 的 Target probability mass 与 threshold：

~~~text
prob_acc += target_prob(candidate)

accept if:
coin < prob_acc / threshold_acc
OR
target_prob(candidate) >= threshold_single
~~~

这不是 `u × q(x) < p(x)`。

还有一个很容易被变量名骗到的细节。非 classic-rejection 分支进入这个 sampler 前，`eagle_sample()` 会传入：

~~~python
draft_probs = torch.zeros_like(target_probs)
~~~

在 Target-only kernel 内，一个 sibling 没被接受时，会把对应 candidate 的 Target probability 写到这个 buffer；最后再对：`relu(target_probs - draft_probs)`

做 terminal sampling。此时这个名为 `draft_probs` 的 buffer **并不是 Draft proposal distribution `q`**，更接近“记录已经被排除 candidate probability mass 的工作缓冲区”。源码旁边也保留了 `FIXME: leverage draft probs`。

所以阅读当前实现时最好直接把四类路径分开：

| Accept path | Topology | 核心判定 | 是否使用 Draft `q` |
| --- | --- | --- | --- |
| EAGLE Greedy | Tree | candidate == Target argmax | 否 |
| Target-only Tree Sampling | Tree | Target mass + threshold | 否，当前不是 classic `p/q` |
| Classic Rejection Sampling | Chain | `u × q(x) < p(x)` | 是 |
| DSpark Greedy | Chain | 最长 Target-match prefix | 否 |

这几条路径使用不同的判定规则。Classic Rejection Sampling 的分布校正来自 `p/q`；Target-only Tree Sampling 则要沿其 Target Mass 与 Threshold 规则分析。

Ascend NPU 分支从 `sgl_kernel_npu.sample` 导入 Tree / Chain Sampling Op。本文沿 SGLang 的分流说明它们接收什么、返回什么以及上层怎样使用结果；设备核的算法细节位于该外部实现中。

## 五、DSpark 链式接受的 correct_len

回到当前 DeepSeek-V4 DSpark。这里和 EAGLE `topk>1` tree 最大的区别不是“算法名字不同”，而是 **layout 本身是 chain**。

`DSparkWorkerV2` 在 state commit 处直接写明：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~

固定源码：

[`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

Chain 的好处是路径天然由 prefix length 唯一确定。假设：

~~~text
Draft:   A → B → C
Target:  A → B → X
~~~

那么只需要：

~~~text
correct_len = 2
bonus       = X
~~~

就已经知道最终有效输出是：`A, B, X`

不需要再保存 `root → A → B` 的 tree mapping。

DSpark Greedy 通过 `compute_dflash_correct_drafts_and_bonus()` 得到 `correct_len` 与 `bonus`；Sampling 则复用 chain rejection sampler。入口集中在：

[`dspark_accept.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)

随后 `TargetVerifyExecutor.accept_and_finalize()` 会先在 TP group 内同步 Accept 结果，再执行：

~~~python
commit_lens = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
~~~

固定源码：

[`dspark_verify.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_verify.py)

这里的 `+1` 与 EAGLE 的 `accept_lens = num_correct_drafts + 1` 是同一个高层语义：

~~~text
accepted Draft tokens
        +
one terminal Target token
~~~

这些变量分别描述不同阶段，映射时按其计数单位转换。

DSpark 还会用 `BuildOutTokens` 构造固定宽度输出 buffer：先放 Draft tokens，再把 `bonus` scatter 到 `correct_len` 位置。消费时由 `commit_lens` 告诉下游哪些位置有效：

[`dspark_verify_window.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py)

因此可以把 EAGLE Tree 与 DSpark Chain 对齐成：

~~~text
EAGLE Tree
Tree topology
   ↓
Target-guided traversal
   ↓
accept_index
   ↓
accept_lens = correct drafts + 1


DSpark Chain
Linear topology
   ↓
prefix verification
   ↓
correct_len
   ↓
commit_lens = correct_len + 1
~~~

两者最终都在做同一件事：**给系统确定新的 committed sequence boundary。**

## 六、接受结果怎样交给输出与缓存

到这里，Accept Decision 可以重新理解成一个系统状态收敛问题。

在 Target Verify 之前，系统允许同时存在：

~~~text
多个 candidate
多个 Verify rows
多个 branch-local KV / hidden state
尚未确定的未来
~~~

在 Accept 之后，这些状态必须共同收敛到一条线：

~~~mermaid
flowchart TD
    A["Draft future"] --> B["Target Verify"]
    B --> C["Accept Decision"]
    C --> D["accept_index / correct_len"]
    D --> E["Output: accepted drafts + bonus"]
    D --> F["Committed seq length"]
    D --> G["KV / hidden / recurrent state selection"]
    E --> H["Next iteration"]
    F --> H
    G --> H
~~~

Debug 时，可用以下关系检查接受结果：

对于 EAGLE Tree：

~~~text
accept_lens
= num_correct_drafts + 1

valid accept_index entries
= accept_lens

predict[accept_index]
= accepted Draft outputs + terminal Target bonus
~~~

但要同时记住那一格 input/output shift：

~~~text
accept_index 对应的 Verify input rows
= previous bonus/root + accepted Draft input nodes

predict[accept_index]
= accepted Draft output tokens + new terminal bonus
~~~

对于当前 DSpark Chain：

~~~text
commit_lens
= correct_len + 1

new_seq_lens
= prefix_lens + commit_lens
~~~

如果这几组关系对不上，后面的 KV、hidden state、Mamba/recurrent state 或下一轮 Draft 都可能从错误边界继续。

### 源码阅读顺序

| 目标 | 固定版本源码 |
| --- | --- |
| EAGLE Accept 总入口 | [`eagle_sample()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py) |
| Tree topology 构造 | [`spec_tree.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py) |
| Greedy Tree traversal | [`eagle.cuh`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh) |
| Target-only Tree Sampling | [`sampling.cuh`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh) |
| Classic Rejection Sampling | [`reject_sampling.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/reject_sampling.py) |
| `accept_index → accept_tokens → bonus` | [`eagle_worker_common.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_worker_common.py) |
| Bonus extraction | [`eagle.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/eagle.py) |
| DSpark Greedy / Sampling Accept | [`dspark_accept.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py) |
| DSpark Accept finalize | [`dspark_verify.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_verify.py) |
| DSpark output window | [`dspark_verify_window.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py) |

`accept_index` 或 `correct_len` 把接受结果交给后续模块。输出侧读取对应预测，缓存侧选择已经计算过的输入状态，两者保持同一条路径和正确的长度关系。接下来可以沿 [KV Commit](deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md) 看这个决定如何落到物理缓存。
