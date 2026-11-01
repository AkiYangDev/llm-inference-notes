# 投机解码 Accept：从 Verify 行到最终输出路径

Verify 输入 `R、A、C`，逐行预测 `A、C、X`。接受以后返回用户的是 A、C、X，缓存留下的却是 R、A、C。两份结果都来自同一组 Verify 行，只是读了不同数组。

Accept 将候选收束到一条有效历史。Greedy 沿 Target 预测匹配候选，Sampling 按对应概率规则判断；随后输出、缓存、Draft 都要使用这次决定。下面先沿一棵小树找出有效行，再看 Sampling 和 DSpark 链怎样给出结果。

<details markdown="1">
<summary>源码版本与阅读范围</summary>

源码固定为 `sgl-project/sglang @ 791c7850d0960fd768102f71e7d999b036bb75ba`，核对日期为 2026-09-20。EAGLE 部分跟踪通用 tree accept，DSpark 部分跟踪 V4 chain accept。GPU JIT 内部可从仓库核对；NPU 使用外部 `sgl_kernel_npu`，本文分析 SGLang 的输入、输出和分流，未逐行核验外部算子。

</details>

## 一、Target 怎样确认候选前缀

旧 bonus `R` 是上一轮 Target 已确认的输出，这轮放在树的 root：

~~~text
                 R
              /     \
             A       B
           /   \
          C     D
~~~

假设 Greedy Target 预测为：

~~~text
Target(R) = A
Target(A) = C
Target(C) = X
~~~

暂不计结束和截断，本轮输出 `A → C → X`。A、C 是接受的 Draft，X 是 C 行的 terminal bonus。R 只进入本轮输入。

Draft path score 在 Verify 之前筛选候选。到了 Accept，Greedy 根据 Target 和树连接选择路径：R 行预测 A，就沿 A 继续，即使 B 的 Draft 分数更高。

两类决策处理的对象为：

| 阶段 | 决定什么 | 主要依据 |
| --- | --- | --- |
| Draft search | 哪些未来值得花 Verify budget | Draft score / candidate pruning |
| Accept decision | 哪些未来真的成为历史 | Target result + topology + sampling rule |

分数分配验证机会，Target 判定把这些机会转成实际输出。

## 二、接受前先把父节点关系变成可遍历结构

Tree Builder 将父节点数据转换为可遍历结构，Accept 才能从 root 逐层寻找 child。

`parent_list + selected_index` 输入 Builder，输出三组 metadata。其中树内 node ID 与 batch 展平 Verify 行之间的关系为：

| metadata | 定位什么 |
| --- | --- |
| `retrieve_index[node]` | 当前 node 对应哪一条 flattened Verify row |
| `retrieve_next_token[parent]` | parent 的 first child |
| `retrieve_next_sibling[child]` | 当前 child 的 next sibling |

交接过程为：

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

first-child / next-sibling 沿当前节点寻找后继：

~~~text
root
  ↓ Target(root) 决定下一步
child
  ↓ Target(child) 决定下一步
child
  ↓
...
~~~

R 先检查 A，不匹配再查兄弟 B；匹配 A 就进入 C、D。整个过程从 root 向前走，无需先选叶子再倒查路径。

数据也随阶段变化：parent metadata、selected_index 重建树，`retrieve_*` 表达遍历链接，最后的 `accept_index` 才指向有效 Verify 行。

## 三、Greedy Accept 的行索引与输出

`eagle_sample()` 的 Greedy 分支先对 Target logits 取 argmax，再调用 `verify_tree_greedy_func()`。[`eagle_sample()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py) [`VerifyTreeGreedy`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh)

省略 batch 维、容量和无效填充，用 `row(...)` 表示 Verify 行，遍历可以写为：

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

设物理行序为：

~~~text
row 0 → R
row 1 → A
row 2 → B
row 3 → C
row 4 → D
~~~

先记 root 行 0，R 预测 A，接入行 1；A 预测 C，接入行 3。得到 `[0, 1, 3]`，包含 root。读取这些行的预测：

~~~text
predict[row 0] = A
predict[row 1] = C
predict[row 3] = X
~~~

~~~python
predict[accept_index] == [A, C, X]
~~~

索引连接 root、接受节点与它们各自产生的预测。多请求还带有相应 flattened 偏移，0、1、3 是这条请求的例子。

`VerifyTreeGreedy.accept_token_num` 数匹配的 Draft，`eagle_sample()` 返回时用 `accept_lens = num_correct_drafts + 1` 加入末尾 Target 输出。

本例得到：

~~~text
num_correct_drafts = 2    # A, C
accept_lens        = 3    # A, C, X
~~~

源码注释也区分了内部 drafts-only 计数与返回时的 +1。[`eagle_utils.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/eagle_utils.py)

没有 child 匹配，root 预测仍能给出长度 1 的输出；候选全部通过，末行也会产生 terminal Token。所以发生拒绝与否，都需要末尾这一枚。

<a id="a-c-x-kv-r-a-c"></a>

### accept_index 同时连接输入 KV 与输出预测

将它们按行并排：

| Verify 输入 | 该行预测 / 新输出 | 本轮已形成的自身 KV |
| --- | --- | --- |
| R | A | R 的 KV |
| A | C | A 的 KV |
| C | X | C 的 KV |

输出 `[A,C,X]`，已执行输入 KV 为 `[R,A,C]`，下一轮从 X 开始。

X 本轮只有 ID，等待下轮作为输入执行。B、D 即使写入了临时状态，也没有进入这条正式历史。

`accept_index=[row(R), row(A), row(C)]` 读取不同数组时，分别得到：

~~~text
用于 output gather：
predict[accept_index] → [A, C, X]

用于 Verify-row / KV gather：
rows[accept_index]    → [R, A, C]
~~~

relocation 因此搬移 R、A、C。若照新输出列表取缓存，就会漏 R，并试图读取不存在的 row(X)。

`run_eagle_verify()` 随后用 `accept_tokens = predict[accept_index]` 取得输出。

buffer 还可能有填充值，只有 `accept_lens` 前缀有效。`fill_bonus_tokens()` 从每请求的 `accept_len - 1` 位置取下一轮 bonus。[`fill_bonus_tokens()`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/eagle.py)

Greedy 的结果交接为：

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

Sampling 使用的是实际送入 Accept 的 Target 分布 p。`eagle_sample()` 先按 temperature softmax，再按需做 top-k、top-p、归一化，处理后的概率才参与判定。

固定实现包含两类概率接受。实际分流还由平台、策略决定，需要沿运行分支识别使用哪一个 sampler。

### Classic Rejection Sampling 的 Draft 概率 `q`

经典规则从 Draft 分布 q(x) 提出 x，Target 分布为 p(x)，接受概率为：

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

实现抽取 `u ~ U(0,1)`，比较：

$$
u\,q(x)<p(x)
$$

chain Kernel 使用这条规则。[`reject_sampling.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/reject_sampling.py)

发生拒绝，再从残差分布采样：

$$
r(y) \propto \max(p(y)-q(y),0)
$$

接受与残差共同扣除 Draft 已过量提议的概率质量。这里的 q 必须就是生成候选的分布，且与 p 对应相同上下文、词表。

取一个教学分布，同时列出接受贡献和拒绝残差：

| Token | Draft q | Target p | 接受分支质量 `min(p,q)` | 正残差 `max(p-q,0)` |
| --- | ---: | ---: | ---: | ---: |
| A | 0.6 | 0.3 | 0.3 | 0 |
| B | 0.3 | 0.4 | 0.3 | 0.1 |
| C | 0.1 | 0.3 | 0.1 | 0.2 |

提出 A 时，接受概率为 `0.3/0.6=0.5`。拒绝后按最后一列归一化补采样，A 已没有残差质量。

残差归一化后，B、C 概率为 `1/3、2/3`，A 为 0。首位置接受分支贡献 0.3、0.3、0.1，拒绝质量 0.3 又给 B、C 补入 0.1、0.2，正好恢复 Target 的 0.3、0.4、0.3。这是算法算例。

classic 检查 `draft_probs`、词表维度，使用真实 q 才能完成校正。候选若全部接受，末尾 bonus 直接从最后 Target p 采样；后面没有提议，无需再扣 q。

<a id="target-only-tree-sampling-sibling-target-mass-classic-pq-rejection"></a>

### Target-only Tree Sampling 累加 Target mass

Tree sampler 的另一条入口为：

[`TreeSpeculativeSamplingTargetOnly`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/jit/include/sgl_kernel/speculative/sampling.cuh)

它也沿 first child / sibling 遍历，以本层 Target mass 和阈值判断：

~~~text
prob_acc += target_prob(candidate)

accept if target_prob(candidate) > 0 AND (
    coin < prob_acc / threshold_acc
    OR target_prob(candidate) >= threshold_single
)
~~~

`target_prob(candidate) > 0` 排除零概率候选。接受 child 后切到该节点的 Target 分布，重置累计量、随机数，再进入下一层。这里没有使用 classic q。

这条路径的 `draft_probs` 名称容易令人误解：非 classic 分支先用 `torch.zeros_like(target_probs)` 创建它。

拒绝 sibling 时，Kernel 把其 Target probability 写进这个 buffer；停止后用 `relu(target_probs - draft_probs)` 排除已拒绝质量。它记录的是工作状态，而非 Draft 真实提议 q。源码的 `FIXME: leverage draft probs` 也保留了这个区别。

各路径对应为：

| Accept path | Topology | 核心判定 | 是否使用 Draft `q` |
| --- | --- | --- | --- |
| EAGLE Greedy | Tree | candidate == Target argmax | 否 |
| Target-only Tree Sampling | Tree | Target mass + threshold | 否，当前不是 classic `p/q` |
| Classic Rejection Sampling | Chain | `u × q(x) < p(x)` | 是 |
| DSpark Greedy | Chain | 最长 Target-match prefix | 否 |

classic 按 p/q 与残差校正分布，Target-only 按 Target mass、threshold 推进，具体行为各自沿规则分析。

NPU 从 `sgl_kernel_npu.sample` 导入 Tree / Chain Ops。本文可确认 SGLang 选择哪个入口、怎样消费返回值，设备内部实现仍在外部包。

## 五、DSpark 链式接受的 correct_len

树中两条分支可能一样长，仅有长度无法区分选中了谁。固定 V4 DSpark 使用链，`correct_len` 就能唯一确定接受前缀，无需再带树路径索引。

state commit 的注释为：

~~~text
Chain layout only:
step index = commit_lens - 1

A tree (topk > 1) layout
would need the accept-index mapping
~~~

实现位于 [`dspark_worker_v2.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)。

代入这条候选链：

~~~text
Draft:   A → B → C
Target:  A → B → X
~~~

返回结果为：

~~~text
correct_len = 2
bonus       = X
~~~

输出 `A, B, X`，在 C 处分歧。输入状态留下 root→A→B，两边仍错一位。

Greedy 调用 `compute_dflash_correct_drafts_and_bonus()`，Sampling 复用 chain rejection。[`dspark_accept.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_accept.py)

Eager `accept_and_finalize()` 将 `correct_len`、`bonus`、`cap_trim_lens` 交 `_tp_sync`，再整理长度；folded 路径直接读对应结果。同步是否产生通信取决于策略，长度整理本身不等于新增 collective。

~~~python
commit_lens = correct_len + 1
new_seq_lens = prefix_lens + commit_lens
~~~

实现见 [`dspark_verify.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/srt/speculative/dspark_components/dspark_verify.py)。

与 EAGLE 一样，+1 在接受的 Draft 外再计入一枚 terminal Target Token。

原 Prefix 为 10，`correct_len=2`，提交 3 个输入位置后 `new_seq_lens=13`，输出也有 3 枚。末尾 X 的 KV 留待下一轮形成。

`BuildOutTokens` 复制候选，再把 bonus 写到 correct_len 位置。本例将索引 2 的 C 换成 X，仅前 `commit_lens=3` 有效。Verify 范围受限时还要裁剪，有效长度由接受结果给出。[`dspark_verify_window.py`](https://github.com/sgl-project/sglang/blob/791c7850d0960fd768102f71e7d999b036bb75ba/python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py)

两种布局的结果为：

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

Tree 用长度加行索引确定历史，Chain 则从长度直接还原前缀。

## 六、接受结果怎样交给输出与缓存

验证时保存了多个可能的未来：

~~~text
多个 candidate
多个 Verify rows
多个 branch-local KV / hidden state
尚未确定的未来
~~~

接受以后，输出读取有效行的预测，缓存读取相同行的输入状态，下轮接末尾 bonus：

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

路径与长度的对应为：

| 布局 | 有效长度 | 怎样定位有效预测 |
| --- | --- | --- |
| EAGLE Tree | `accept_lens=num_correct_drafts+1` | 有效 `accept_index` 项数为 `accept_lens`，按这些 Verify 行读取 `predict[accept_index]` |
| DSpark Chain | `commit_lens=correct_len+1` | 由前缀长度确定有效输出，`new_seq_lens=prefix_lens+commit_lens` |

这次决定接下来还要落实到 KV、hidden、recurrent state。只对上输出长度，并不能保证所有状态都来自同一条分支。

如果输出 A/C/X 正确，正式缓存却保留 B，可以向后追搬移；如果 accept_index 已选错 child，就先向前查拓扑和判定。两类问题发生在接受决定的不同侧。

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

下一篇沿 [KV Commit](deepseek-v4-speculative-kv-commit-compact-rollback-source-analysis.md) 跟踪这些状态的写入、整理、释放过程。
