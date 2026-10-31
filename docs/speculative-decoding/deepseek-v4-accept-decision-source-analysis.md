# 投机解码 Accept：从 Verify 行到最终输出路径

Verify 可以处理多条候选分支，请求下一轮却只能延续一条历史。Accept 选出有效路径，再让输出、缓存、Draft 使用相同决定。Greedy 寻找 Target 预测匹配的前缀，Sampling 则应用相应概率规则。

例如输入行 `R、A、C` 预测出 `A、C、X`，`accept_index` 标识的是三条 Verify 输入行。读预测数组得到新输出，读输入状态则得到已计算的历史。下面沿这组对应关系看树遍历、采样和 DSpark 链式接受。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

> 源码固定到 `sgl-project/sglang @ 791c7850d0960fd768102f71e7d999b036bb75ba`，核对日期为 2026-09-20。EAGLE 分析通用 tree accept，DSpark 分析 V4 chain accept。GPU JIT 内部可从仓库核对；NPU 调用外部 `sgl_kernel_npu` 算子，本文只确认 SGLang 的输入输出契约，不将外部实现当作已逐行验证。

</details>

## 一、Target 怎样确认候选前缀

先用一棵树贯穿例子。旧 bonus `R` 已经由上一轮 Target 确认，这轮成为 root：

~~~text
                 R
              /     \
             A       B
           /   \
          C     D
~~~

令 Greedy Target 结果为：

~~~text
Target(R) = A
Target(A) = C
Target(C) = X
~~~

暂不计结束、截断，新输出为 `A → C → X`：A、C 是通过验证的 Draft，X 是 C 行预测的 terminal bonus。R 本轮仅作为输入，不重复返回用户。

Draft path score 用于挑选值得消耗 Verify 预算的候选。到 Accept 时，决定路径的是 Target、连接关系与采样规则。B 即便 Draft 评分更高，Greedy Target 在 R 预测 A，也不会改走 B。

两者可按决策对象区分：

| 阶段 | 决定什么 | 主要依据 |
| --- | --- | --- |
| Draft search | 哪些未来值得花 Verify budget | Draft score / candidate pruning |
| Accept decision | 哪些未来真的成为历史 | Target result + topology + sampling rule |

因此分数表示优先验证程度，实际是否接受还需由后面的行与拓扑落地。

## 二、接受前先把父节点关系变成可遍历结构

Tree Build 先把父节点信息改成可遍历结构，Accept 再由 root 向 child 推进。

`parent_list + selected_index` 输入 Builder，得到三组 metadata。先分清树内 node ID 与 batch 的 flattened Verify 行：

| metadata | 定位什么 |
| --- | --- |
| `retrieve_index[node]` | 当前 node 对应哪一条 flattened Verify row |
| `retrieve_next_token[parent]` | parent 的 first child |
| `retrieve_next_sibling[child]` | 当前 child 的 next sibling |

数据交接为：

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



- [`build_tree_kernel_efficient()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)
- [`sgl_build_tree_kernel_efficient_triton()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/spec_tree.py)
- [`build_tree_efficient` / `VerifyTreeGreedy`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh)

first-child / next-sibling 适合沿当前预测向前寻找：

~~~text
root
  ↓ Target(root) 决定下一步
child
  ↓ Target(child) 决定下一步
child
  ↓
...
~~~

在 R 先查 A，不匹配才查 sibling B。匹配 A 后继续找 C、D，无需预先选择叶子再倒查祖先。

因此，parent metadata 和 selected_index 属于树重建输入，`retrieve_*` 属于遍历结构，`accept_index` 才是最终有效 Verify 行，不能把三阶段的位置混用。

## 三、Greedy Accept 的行索引与输出

`eagle_sample()` 的 Greedy 分支先 argmax Target logits，再调用 `verify_tree_greedy_func()`。入口与 Kernel 见 [`eagle_sample()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py) [`VerifyTreeGreedy`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh)。





下面省略 batch 维、容量和无效填充，`row(...)` 代表对应 Verify 行，仅作遍历伪代码：

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

令物理行序为：

~~~text
row 0 → R
row 1 → A
row 2 → B
row 3 → C
row 4 → D
~~~

先记录 root 行 0，按 R 的预测 A 接入行 1，再按 A 的预测 C 接入行 3。有效索引为 `[0, 1, 3]`，不是仅含候选的 `[1, 3]`。对应预测为：

~~~text
predict[row 0] = A
predict[row 1] = C
predict[row 3] = X
~~~

~~~python
predict[accept_index] == [A, C, X]
~~~

这组索引表示哪些输入行提供有效预测，包含 root 与接受节点。多请求时还要加对应请求的 flattened 偏移，0、1、3 仅用于本例。

`VerifyTreeGreedy.accept_token_num` 只计匹配 Draft，`eagle_sample()` 返回 `accept_lens = num_correct_drafts + 1`，将末尾 Target 输出纳入。候选接受数和本轮有效输出数因此不同。

本例为：

~~~text
num_correct_drafts = 2    # A, C
accept_lens        = 3    # A, C, X
~~~

源码注释也将内部 drafts-only 与返回时 +1 明确分开。[`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

若没有 child 匹配，`num_correct_drafts=0`，仍由 root 预测提供长度 1 的输出；全部候选通过时，末行也会预测 terminal Token。这个 +1 不依赖是否发生拒绝。

### 最容易写错的一格：输出 `[A, C, X]`，本轮新物化 KV 的却是 `[R, A, C]`

将输入与预测并排，就能看到同一行的两种用途：

| Verify 输入 | 该行预测 / 新输出 | 本轮已形成的自身 KV |
| --- | --- | --- |
| R | A | R 的 KV |
| A | C | A 的 KV |
| C | X | C 的 KV |

所以新输出是 `[A,C,X]`，可提交的 Verify KV 是 `[R,A,C]`，下一轮 root 才是 X。

X 仍只有预测 ID，尚未作为输入，没有本轮 KV；请求继续时它才在下一轮成为 root。B、D 即使已有临时状态，未被选中也不能当作正式历史。

因此同一 `accept_index=[row(R), row(A), row(C)]` 用于不同数组，可得到：

~~~text
用于 output gather：
predict[accept_index] → [A, C, X]

用于 Verify-row / KV gather：
rows[accept_index]    → [R, A, C]
~~~

relocation 处理 R、A、C，无需寻找不存在的 row(X)。照输出列表搬移会漏掉旧 root，并要求一份尚未产生的 KV。

随后 `run_eagle_verify()` 读取 `accept_tokens = predict[accept_index]`。

buffer 可带填充值，只有 `accept_lens` 前缀有效。`fill_bonus_tokens()` 在每请求有效末尾 `accept_len - 1` 取下轮 bonus。[`fill_bonus_tokens()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/eagle.py)



Greedy 接受流程为：

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

Sampling 中的 p 指处理后真正送入 Accept 的 Target 分布。`eagle_sample()` 先按 temperature softmax，再按需 top-k、top-p 和归一化，不能把原 logits 直接当成 p。

固定实现有两类概率接受，平台和策略还会决定实际分流。带了采样参数，并不能证明一定调用某个 sampler。

### Classic Rejection Sampling 的 Draft 概率 `q`

经典规则令 Draft 提议分布为 q(x)、Target 为 p(x)，从 q 提出 x 后，接受概率为：

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

实现可抽 `u ~ U(0,1)` 并直接比较：

$$
u\,q(x)<p(x)
$$

chain Kernel 对应这一规则，见 [`reject_sampling.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/reject_sampling.py)。



拒绝时从残差分布采样：

$$
r(y) \propto \max(p(y)-q(y),0)
$$

这与接受概率共同完成质量校正，扣除 Draft 已经过量提议的部分。推导要求 q 确实用于生成候选，并与 p 对应相同上下文、词表，不能只检查形状。

用一个教学分布代入，可同时列出接受分支贡献和拒绝残差：

| Token | Draft q | Target p | 接受分支质量 `min(p,q)` | 正残差 `max(p-q,0)` |
| --- | ---: | ---: | ---: | ---: |
| A | 0.6 | 0.3 | 0.3 | 0 |
| B | 0.3 | 0.4 | 0.3 | 0.1 |
| C | 0.1 | 0.3 | 0.1 | 0.2 |

提议 A 时，接受概率为 `0.3/0.6=0.5`。拒绝后从最后一列的归一化分布补采样，A 不再分到残差质量。

归一化后 B、C 为 `1/3、2/3`，A 为 0。首位置接受分支贡献 0.3、0.3、0.1，剩余拒绝质量 0.3 再给 B、C 补 0.1、0.2，恰回到 Target 的 0.3、0.4、0.3。这是算法算例，不是模型实测。

classic 分支会检查 `draft_probs` 及词表维度，缺少真正 q 就无法校正。若全部接受，链后 bonus 直接从最后 Target p 采样，无需扣除并不存在的下一步提议。

### Target-only Tree Sampling：沿 sibling 累加 Target mass，但不是 classic `p/q` rejection

另一条 Tree sampler 入口为：

[`TreeSpeculativeSamplingTargetOnly`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh)

它也沿 first child / sibling 查找，但以本层 Target mass 和阈值判断：

~~~text
prob_acc += target_prob(candidate)

accept if target_prob(candidate) > 0 AND (
    coin < prob_acc / threshold_acc
    OR target_prob(candidate) >= threshold_single
)
~~~

`target_prob(candidate) > 0` 防止零概率候选因阈值而通过。接受 child 后，切到其 Target 分布、重置累计量和随机数，再找下一层。这条规则没有使用 classic q。

还需留意 `draft_probs` 在这里是不同含义的变量。非 classic 分支先创建 `draft_probs = torch.zeros_like(target_probs)`。

拒绝 sibling 后，Kernel 将其 Target probability 写进该 buffer，终止时用 `relu(target_probs - draft_probs)` 排除已拒绝质量。因此它是工作缓冲，不是 Draft 的真实提议 q；`FIXME: leverage draft probs` 也指出这个差别，不能仅凭相减形式就认作经典 rejection。

这几条路径应分别阅读：

| Accept path | Topology | 核心判定 | 是否使用 Draft `q` |
| --- | --- | --- | --- |
| EAGLE Greedy | Tree | candidate == Target argmax | 否 |
| Target-only Tree Sampling | Tree | Target mass + threshold | 否，当前不是 classic `p/q` |
| Classic Rejection Sampling | Chain | `u × q(x) < p(x)` | 是 |
| DSpark Greedy | Chain | 最长 Target-match prefix | 否 |

classic 的分布校正来自 p/q 规则，Target-only 则需按 Target mass、threshold 分析其行为。

NPU 从 `sgl_kernel_npu.sample` 导入 Tree / Chain Ops。这里确认的是 SGLang 如何分流和消费结果，设备内部算法仍属于外部实现。

## 五、DSpark 链式接受的 correct_len

固定 V4 DSpark 使用链：树的相同长度不能标识唯一分支，链的前缀却可用 `correct_len` 唯一确定，因而不必再携带树路径索引。

state commit 对此注明：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~



实现见 [`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)。

代入候选链：

~~~text
Draft:   A → B → C
Target:  A → B → X
~~~

接受结果为：

~~~text
correct_len = 2
bonus       = X
~~~

新输出 `A, B, X`，C 处分歧后不能跳过继续接受。缓存保留 root→A→B 的输入路径，与输出错一位。

Greedy 使用 `compute_dflash_correct_drafts_and_bonus()`，Sampling 复用 chain rejection，入口见 [`dspark_accept.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)。



Eager `accept_and_finalize()` 将 `correct_len`、`bonus`、`cap_trim_lens` 交 `_tp_sync`，再整理长度；若已 folded，直接读对应结果。同步是否通信由策略决定，不能把整理长度算成必然新增 collective。

~~~python
commit_lens = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
~~~



对应实现见 [`dspark_verify.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_verify.py)。

这里的 +1 与 EAGLE 输出计数含义相同：已接受 Draft Token 之外，再有一枚 terminal Target Token。

Prefix 为 10、`correct_len=2` 时，本轮提交 3 个输入位置到 `new_seq_lens=13`，输出也是 3 枚。X 仍等下一轮才有 KV，不能把 commit_lens 理解成所有新输出都已缓存。

`BuildOutTokens` 先复制候选，再将 bonus 放入 correct_len 位置。本例索引 2 从 C 换为 X，仅前 `commit_lens=3` 有效。若 Verify 受限，还要按可验证范围裁剪，buffer 宽度不能代替有效输出长度。[`dspark_verify_window.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py)



两种布局对应为：

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

二者都给出提交边界，Tree 还需指定行，Chain 则可从长度还原。

## 六、接受结果怎样交给输出与缓存

Accept 的结果要同时约束输出与缓存，验证期间暂存的多个未来如下：

~~~text
多个 candidate
多个 Verify rows
多个 branch-local KV / hidden state
尚未确定的未来
~~~

选定后，输出读各行预测，缓存读同一行的已算状态，下轮以末尾 bonus 继续：

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

检查结果时，可将路径与长度放在一起：

| 布局 | 有效长度 | 怎样定位有效预测 |
| --- | --- | --- |
| EAGLE Tree | `accept_lens=num_correct_drafts+1` | 有效 `accept_index` 项数为 `accept_lens`，按这些 Verify 行读取 `predict[accept_index]` |
| DSpark Chain | `commit_lens=correct_len+1` | 由前缀长度确定有效输出，`new_seq_lens=prefix_lens+commit_lens` |

两者的已执行输入都包含旧 root 与接受的 Draft input nodes，新输出则包含接受的 Draft 与新 terminal bonus。EAGLE 的同一组行索引分别用于输入状态和输出预测，不能把输出 ID 列表直接当作 KV 搬移列表。

先核接受路径和长度，再看消费阶段。若输出 A/C/X 对了却残留 B 的正式状态，应查搬移；若 accept_index 本身走错 child，则向前查拓扑、判定。KV、hidden、recurrent state 都需属于同一路径，长度正确还不够。

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

下一步沿 [KV Commit](deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md) 看接受决定怎样落实到物理状态：输出取预测，缓存取已执行输入，二者共同保持路径和长度关系。
