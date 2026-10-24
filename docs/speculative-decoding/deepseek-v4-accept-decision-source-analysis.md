# 投机解码 Accept：从 Verify 行到最终输出路径

Target Verify 可以一次算完多条候选分支，但一次请求最终只能沿一条历史继续。Accept 的工作就是从这些结果中选出有效路径，并把同一个决定交给输出、缓存和下一轮 Draft。Greedy 按 Target 的预测寻找匹配前缀；Sampling 则按对应的概率规则接受候选。

最容易看错的是输入与输出相差一位：输入 `R、A、C` 的三行预测，可能产生输出 `A、C、X`。`accept_index` 记录这三条 Verify 输入行，不能把它直接当作输出 token 的下标。下面用这个教学例子走完树遍历，再看概率采样和 DSpark 链式接受如何交接同样的状态边界。

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

忽略请求结束与输出截断等后续处理，这轮得到的新输出是 `A → C → X`。其中 `A、C` 是 Draft 提出、被 Target 接受的候选；`X` 来自输入节点 `C` 的 Target 预测，称为本轮的 terminal bonus。`R` 已经在上一轮成为正式输出，这一轮用它作输入，不会再向用户输出一次。

Draft 的 path score 决定哪些候选能进入这轮验证，作用到这里就结束了。EAGLE rollout 用累计 Draft probability 给候选排序，把有限的 Verify token budget 分给较有希望的节点；进入 Verify 后，接受路径由 Target 结果、树的连接关系和当前采样规则共同决定。Draft 给 `B` 的分数即使很高，只要 Greedy Target 在 `R` 上选择 `A`，也不会沿 `B` 继续。

两层职责可以这样对照：

| 阶段 | 决定什么 | 主要依据 |
| --- | --- | --- |
| Draft search | 哪些未来值得花 Verify budget | Draft score / candidate pruning |
| Accept decision | 哪些未来真的成为历史 | Target result + topology + sampling rule |

因此，Draft score 高只表示值得验证，不表示已经通过验证。后面的索引结构负责把 Target 的选择落到实际候选行上。

## 二、接受前先把父节点关系变成可遍历结构

父节点关系在 Tree Build 阶段就转换为可遍历结构，Accept Kernel 随后按这个结构从 Root 寻找匹配 Child。

Draft rollout 结束后，`parent_list + selected_index` 先交给 Tree Builder。Builder 把不方便设备端逐层搜索的 parent metadata 转换成三组直接可遍历的数据：

~~~text
retrieve_index
retrieve_next_token
retrieve_next_sibling
~~~

先把 Tree node 的局部编号与整个 batch 中的 Verify 行号分开。这几组数组分别回答“这一节点存在哪一行”和“下一步能走到谁”：

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

在小树的 `R` 上，Accept 先检查 child `A`；不匹配才沿 sibling 找 `B`。一旦匹配 `A`，就转去检查 `A` 的 children `C、D`。它不需要先猜最终叶子，再沿父指针倒查。

Parent Tree Recovery 在这里指把 Draft search 的 parent metadata 转成这种向前遍历结构。`parent_list` 与 `selected_index` 是重建树的输入，`retrieve_*` 是遍历时使用的结构，最后的 `accept_index` 才记录选中的 Verify 行。读代码时把这三个阶段分开，可以避免把“候选在 Draft 数组里的位置”当成“接受后在输出里的位置”。

## 三、Greedy Accept 的行索引与输出

Greedy 路径最适合把这个索引语义看透。`eagle_sample()` 先对 Target logits 做 argmax，然后进入 `verify_tree_greedy_func()`：

[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

[`VerifyTreeGreedy`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh)

下面是省略 batch 维度、容量限制和无效值填充的伪代码，用于说明遍历顺序；其中 `row(...)` 表示节点对应的 Verify 行：

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

Kernel 先记录 root 行 `0`，再根据 Target 在 `R` 上的预测 `A`，把匹配的 `A` 行 `1` 接入路径；在 `A` 上预测 `C`，再接入行 `3`。因此，有效的 `accept_index` 是 `[0, 1, 3]`，不是只记录 Draft 候选的 `[1, 3]`。这三行保存的预测分别是：

~~~text
predict[row 0] = A
predict[row 1] = C
predict[row 3] = X
~~~

~~~python
predict[accept_index] == [A, C, X]
~~~

`accept_index` 可以读成“哪些 Verify 输入行产生了本轮的有效输出”。它同时包含 root 和被接受的候选行；root 虽然不是本轮新接受的 Draft token，却提供了第一个输出 `A` 的预测。多请求 batch 中这些值还要指向对应请求的 flattened 行，例子里的 `0、1、3` 只针对这一棵树。

这也解释了 `accept_lens` 的 `+1`。`VerifyTreeGreedy` 内部的 `accept_token_num` 只统计匹配成功的 Draft nodes；`eagle_sample()` 返回的长度则包括末尾 Target token，即 `accept_lens = num_correct_drafts + 1`。前者数被接受的候选，后者数这一轮产生的有效输出，不能混作同一种计数。

例子里：

~~~text
num_correct_drafts = 2    # A, C
accept_lens        = 3    # A, C, X
~~~

固定源码在 [`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py) 中明确保留了“`num_correct_drafts` 在函数内部始终只表示 drafts-only，返回值才 +1”的注释。

如果 root 的预测没有匹配任何 child，`num_correct_drafts=0`，有效的 `accept_index` 仍包含 root，输出长度为 1，输出就是 root 的 Target 预测。反过来，即使一路候选全部匹配，最后一个输入行仍会产生一个 terminal token。因此这个 `+1` 来自输入行提供的预测，与是否在中途遇到拒绝无关。

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

这里的 KV 指相应 token 位置作为模型输入时计算出的缓存。`X` 目前只是从 `C` 这一行 logits 得到的 token ID，还没有作为输入执行，因此这一轮没有 `X` 的 KV。请求继续生成时，它在下一轮成为 root，才会计算自己的 KV。与此同时，Verify 可能已经算出了 `B、D` 分支的临时状态；这些行没有进入接受路径，不能当作正式历史继续使用。

同一个 `accept_index=[row(R), row(A), row(C)]` 可以用于两次 gather，但所读取的数组不同：读预测数组得到新输出，读输入行或 KV 数组得到已经执行过的输入状态。

~~~text
用于 output gather：
predict[accept_index] → [A, C, X]

用于 Verify-row / KV gather：
rows[accept_index]    → [R, A, C]
~~~

accepted-path relocation 搬移的是已经计算好的 `R、A、C` 状态，不需要寻找尚不存在的 `row(X)`。如果把输出 token 列表直接用于缓存搬移，就会漏掉 root，并错误地要求提交 `X` 的缓存。

`run_eagle_verify()` 随后做：

~~~python
accept_tokens = predict[accept_index]
~~~

`accept_index` 的 buffer 还可能带有无效填充，只有 `accept_lens` 指定的前缀有效。`fill_bonus_tokens()` 从每个 request 的有效前缀末尾，也就是 `accept_len - 1` 位置，取出下一轮 bonus：

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

前面的 Greedy 路径比较 candidate 与 Target argmax。进入概率采样分支后，`eagle_sample()` 先按 temperature 计算 Target softmax，再按需要做 top-k、top-p 过滤和重新归一化。下面的 `p` 指实际送进 Accept 的 Target 分布，不能直接拿原始 logits 代替。

固定版本中有两套容易混淆的概率接受规则。具体是否进入这些分支还受平台与策略分流影响，不能只看到请求带了采样参数，就认定一定执行了某个 sampler。

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

采样。这个校正与前面的接受概率是一套规则：Draft 在某个 token 上提出得过多，就在拒绝分支中把这部分质量扣掉。分布保持的推导要求 `q` 确实是提出该候选时使用的概率分布，且与 `p` 对齐到同一上下文、同一词表；形状相同本身还不够。

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

归一化后，拒绝分支在 `B、C` 上的概率分别是 `1/3、2/3`，`A` 的概率为 0。只看第一个候选位置，接受分支给 `A、B、C` 的概率质量分别为 `0.3、0.3、0.1`，剩余 `0.3` 的拒绝质量再补给 `B、C` 各 `0.1、0.2`，合起来正好回到 Target 的 `0.3、0.4、0.3`。这是算法教学算例，不是一次模型测量。

开启 classic rejection sampling 后，SGLang 会检查 `draft_probs` 是否存在、词表维度是否与 Target 一致，因为没有 proposal `q` 就无法执行这套校正。若整条候选链都被接受，末尾 bonus 则直接从最后位置的 Target 分布 `p` 采样，不再扣除一个并不存在的下一步 Draft proposal。

### Target-only Tree Sampling：沿 sibling 累加 Target mass，但不是 classic `p/q` rejection

EAGLE tree 还有另一套 sampler：

[`TreeSpeculativeSamplingTargetOnly`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh)

它同样沿 `first child → next sibling` 遍历，但核心 accept condition 使用的是当前 siblings 的 Target probability mass 与 threshold：

~~~text
prob_acc += target_prob(candidate)

accept if target_prob(candidate) > 0 AND (
    coin < prob_acc / threshold_acc
    OR target_prob(candidate) >= threshold_single
)
~~~

这里还有 `target_prob(candidate) > 0` 的保护条件，避免阈值设置让零概率候选被接受。每接受一个 child，kernel 会切到该 child 的 Target 分布，清空累计量并换用对应的随机数，再扫描下一层 siblings。它累加的是当前层候选的 Target mass，没有使用 classic 规则中的 proposal `q`。

还有一个很容易被变量名骗到的细节。非 classic-rejection 分支进入这个 sampler 前，`eagle_sample()` 会传入：

~~~python
draft_probs = torch.zeros_like(target_probs)
~~~

在 Target-only kernel 内，一个 sibling 没被接受时，会把它的 Target probability 写进对应行的这个 buffer。结束遍历后，terminal sampling 使用 `relu(target_probs - draft_probs)`，排除终止位置上已经被拒绝的候选质量。这里的 `draft_probs` 是工作缓冲区，并不是提出候选时的 Draft distribution `q`；源码旁的 `FIXME: leverage draft probs` 也提示了这个差别。不能仅凭两种实现都做一次减法，就把它们认作同一套 rejection sampling。

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

本文固定版本的 DeepSeek-V4 DSpark 使用 chain layout：每个位置只有一个后继候选。一棵树可能有多条同样长的路径，长度无法区分它们；一条链的前缀则只用长度就能唯一确定。这个布局差别决定了 DSpark 为什么可以使用 `correct_len`，而不必再携带树路径索引。

`DSparkWorkerV2` 在 state commit 处直接写明：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~

固定源码：

[`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)

用一条候选链看具体计数。假设：

~~~text
Draft:   A → B → C
Target:  A → B → X
~~~

那么只需要：

~~~text
correct_len = 2
bonus       = X
~~~

就能确定有效输出 `A, B, X`。第三个 Draft 候选 `C` 与 Target 的 `X` 不符，接受在这里停止；即使链上更后面某个 token 恰好匹配，也不能跳过这个分歧继续接受。缓存侧保留的输入路径仍然是 `root → A → B`，与新输出保持前面解释过的一位偏移。

DSpark Greedy 通过 `compute_dflash_correct_drafts_and_bonus()` 得到 `correct_len` 与 `bonus`；Sampling 则复用 chain rejection sampler。入口集中在：

[`dspark_accept.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)

`TargetVerifyExecutor.accept_and_finalize()` 的 Eager 路径通过 `_tp_sync` 同步 `correct_len`、`bonus` 与 `cap_trim_lens`，再交给 `FinalizeAcceptLens` 计算下面两个长度；若 Accept 已折叠进 Verify graph，则直接读取该路径准备好的结果。同步是否执行由对应的策略控制，长度计算不等于无条件插入一次通信。

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

例如已缓存前缀长度为 10，`correct_len=2`，则本轮提交 3 个输入位置，`new_seq_lens=13`；新输出也是 3 个 token，但最后的 `X` 要到下一轮才有自己的缓存。`commit_lens` 因而不能解释成“本轮新输出的所有 token 都已经有 KV”。

DSpark 用 `BuildOutTokens` 构造固定宽度输出 buffer：先放 Draft tokens，再把 `bonus` scatter 到 `correct_len` 位置。在 `correct_len=2` 的例子中，零起始位置 2 原本放 `C`，现在被 `X` 覆盖；只有前 `commit_lens=3` 格有效，其余格即使有旧候选也不能作为正式输出。若实际 Verify 窗口受限，接受长度还会先按可验证范围裁剪，不能仅从 buffer 宽度推算有效输出：

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

两者都确定了本轮可提交的序列边界。Tree 还需要告诉后续模块具体选中了哪些行，Chain 则可以由长度还原这些行。

## 六、接受结果怎样交给输出与缓存

接受决策完成后，输出与缓存必须使用同一条路径。Verify 阶段可以暂时保留多种未来：

~~~text
多个 candidate
多个 Verify rows
多个 branch-local KV / hidden state
尚未确定的未来
~~~

Accept 选出路径与长度后，输出侧读取这些行的预测，缓存侧提交这些行已经计算出的状态，下一轮则从末尾 bonus 继续。三者按下面的关系衔接：

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

排查时先核对接受路径和有效长度，再检查状态搬移。若输出 `[A,C,X]` 正确，但缓存仍混入 `B` 分支，问题已经发生在 Accept 结果的消费阶段；若 `accept_index` 自身就走到了错误 child，则应向前检查拓扑和判定。KV、hidden state 或具体模型使用的 recurrent state 都必须与所选路径对齐，不应仅凭最终输出长度正确就结束检查。

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
