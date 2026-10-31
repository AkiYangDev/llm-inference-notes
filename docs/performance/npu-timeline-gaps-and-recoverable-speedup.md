# NPU 时间线里有 19% 空隙，为什么不能直接得到 19% 加速？

一次 DeepSeek-V4-Flash、Ascend 910C Decode trace 的 rank0 窗口约为 2.085 秒。其中有 391.37ms 没有被已记录的 device 任务区间覆盖，占 18.77%；rank1 的未覆盖比例为 20.68%。这看起来是值得调查的空隙，但还需要查明其中实际发生了什么。

继续追踪后，部分空隙与 host collective 重叠，部分靠近动态元数据准备。其中两次小张量拷贝的相关 host 时间，全窗口累计只有约 3ms。它们提供了具体线索，却无法解释或回收全部 19%。

下面复用 2026 年 10 月 20 日冻结环境中的 rank0/1 分析与源码，分别说明区间计算、等待归属和收益判断，没有新采 trace 或运行模型。全部数字及证据范围见[摘录](../../assets/npu-timeline-gaps-and-recoverable-speedup/evidence.json)。

## 1. 先把“19%”的分母和覆盖范围写出来

输入是两个 rank 的 `kernel_details.csv` 和对应 host trace。将每个 device 任务的开始时间与 Duration 转成区间，再合并同 rank 所有已记录 stream。窗口减去这个并集，才得到本文所说的空隙。

记观察窗口为 W、记录任务区间为 I：

```text
recorded_busy = union(I_1, I_2, ...)
recorded_gap  = W - recorded_busy
gap_ratio     = length(recorded_gap) / length(W)
```

按毫秒计算，结果如下：

| 统计量 | rank0 | rank1 |
| --- | ---: | ---: |
| 窗口 wall | 2085.2914 | 2085.7130 |
| 已记录任务区间并集 | 1693.9219 | 1654.3460 |
| 并集未覆盖时间 | 391.3695 | 431.3671 |
| 未覆盖比例 | 18.7681% | 20.6820% |
| 单次最大空隙 | 9.4046 | 9.4402 |
| kernel Duration 算术和 | 2319.7540 | 2286.0664 |
| Duration + Wait 算术和 | 2734.9002 | 2738.0109 |

这个数描述该 rank 所有已记录任务未覆盖的时间。记录不仅有 AI Core，也有 Vector、AI CPU 和通信路径任务；只取 AI Core 会漏掉活动。记录覆盖的区间也未必都在高效计算，轮询和等待任务同样可能占据资源。

因此 18.77% 无法直接作为硬件利用率或全机空闲率。若导出遗漏 Event/Notify 等任务，空白中仍可能有未表达的活动，需要在已知采集范围内解释。

## 2. 同一段时间，为什么会有四种“耗时”

rank0 的 Duration 算术和约 2319.75ms，大于窗口 2085.29ms，是因为 stream 重叠时，同一毫秒会重复计入。区间并集只计算一次，得到约 1693.92ms。

下图是教学示意，时间和形状均为构造值，并非实测 trace 截图：

![跨 stream 并集与跨 rank 空隙交集的教学示意](../../assets/npu-timeline-gaps-and-recoverable-speedup/interval-scope-example.svg)

上半图中，stream A 任务共 7ms、B 共 4ms，算术和为 11ms；区间合并后覆盖 9ms，整个窗口长 10ms。累加任务时间、记录覆盖时间与墙时分别使用不同口径。

`Duration + Wait` 还包含每任务等待。Wait 不能直接接成 Duration 后的一段独占执行：任务等待时其他 stream 可以运行，多个任务也可能等待同一依赖。把等待都相加，会再次计算已经经过的时间。

所以分析同时保存 wall、busy union、Duration sum 和 Duration+Wait sum。最后一项适合定位记录了大量等待的任务，无法给出整机空闲时长。若只有 `op_summary` 累计数，没有逐次时间戳，也无法重建区间并集。

计算时先用 Decimal 从大绝对微秒时间戳减去共同原点，再处理局部区间，避免低精度相减。原分析因此纠正了早期约 0.42ms 累计差别。这只是解析精度修正，没有产生新的性能收益。

## 3. 空隙后面的算子，未必制造了这段空隙

rank0 最大空隙约 9.405ms，相邻任务为：

| 位置 | 原始任务名 | stream | Task ID | 执行时长（μs） | 记录 Wait（μs） |
| --- | --- | ---: | ---: | ---: | ---: |
| 空隙前 | `assign_req_to_token_pool` | 27 | 3426 | 2.72 | 230.15 |
| 空隙后 | `aclnnIndex_IndexCheck_IndexCheck` | 30 | 12336 | 2.96 | 9404.53 |

IndexCheck 实际只执行 2.96μs，约 9.4ms 在 Wait 字段中。若按 Duration+Wait 排序后直接重写 IndexCheck，就会将等待的承接位置误作等待来源。

对应 host 区间内，`gloo:all_gather` 与该空隙重叠约 85.4%。Task ID、stream、时间和 `connection_id` 还能关联到前后 `Node@launch`，比按名称猜测更有依据。它支持优先检查跨 rank host collective，却未确定哪个 peer 迟到，也未测得网络带宽。

Fill→Clamp 是另一类较短例子：rank0 Fill 执行 1.44μs、Clamp 执行 3.00μs，两者间约 540.86μs，位于 stream43，Task ID 为 21924、21925。host 有 `aten::to`、`_to_copy`、`copy_`，值得追查输入准备，但这半毫秒并非 Clamp 执行时间。

前后任务区间是直接记录，提交事件关联提供更明确锚点，真正归因还需执行依赖。宽 Scheduler 标记覆盖空隙时，也可能只是外层包装；本次计算排除了这种宽标记，避免仅凭同时出现认定根因。

## 4. 从动态端点追到两次拷贝，收益预算反而缩小了

按空隙前后任务的 Model ID 分类，dynamic→dynamic 占未覆盖时间的 88.28%、89.41%，因此动态输入准备值得优先检查。但 dynamic 标签并不等于所有任务都在 Graph 外，无具体 Model ID 的导出任务也可能穿插于 replay。

两端均为 Model47 的空隙，最大仅 4.22μs、5.34μs，未出现 100μs 以上实例。这仅描述该端点类别，不能排除整个 Graph47 replay 内的更大空隙；后者需要实际 replay 起止及完整嵌套记录。

两个 rank 各有 28 次 Fill→Clamp 结构：Clamp 处理 FLOAT `[bs]`，之后比较同一 bs，之前有两次 int64 `[16]` 传输准备。它与保存的 DSpark 路径对应：

```text
DraftBlockProposer._run_forward
  → _fill_dp_moe_sync_metadata
  → stage_sampling_params
  → draft model forward / graph
```

`_fill_dp_moe_sync_metadata` 分别构造全局 token 数、logprob token 数的 host 向量并复制到设备，随后 sampling 参数准备对温度 clamp 并生成 greedy mask。只读复核时，两个 helper 与安装源 SHA 匹配。形状、顺序和源码支持这项关联，但采集没开 Python stack，不能写成已获取完整调用栈。

具体到这类重复结构，相关成本为：

| 指标 | rank0 | rank1 |
| --- | ---: | ---: |
| 重复 Fill→Clamp 空隙累计（ms） | 11.4883 | 12.3239 |
| 占各自整个采样窗口 | 0.5509% | 0.5909% |
| 两向量对应 host `_to_copy` 时长累计（ms） | 3.3663 | 3.1080 |
| 占各自整个采样窗口 | 0.1614% | 0.1490% |

391ms 总空隙、11ms Fill→Clamp 空隙和 3ms host 拷贝相关时间覆盖不同对象。合包两次传输只能减少其中部分准备与提交，不能将整个空隙计作潜在收益。host `_to_copy` 自身也可能包含等待，未必全是 DMA。

两个计数向量也不能任意删除或跨批缓存：全局与 logprob 列表未必相同，序列、接受结果和 token 数持续变化，异步使用还要求正确存活及覆盖顺序。考虑到已识别的成本很小，本次将合包降为低优先级，没有为它重启服务或报告 NPU 加速。

## 5. 一个 rank 的空白，为什么不是整机可消除的时间

若每 rank 的记录空隙为 G，全组同时未覆盖的时间是各 G 交集。相加或求平均都得不到它。下半图的反例中，两个 rank 各有 20% 空白，但位置错开，全组共同空白为零。

求交集前还需证明时钟对齐、rank 属于同一范围、窗口覆盖同一阶段。仅凭时间戳接近就拼接设备 CSV 并不充分。本次只完成相邻 rank0/1 的完整材料分析，未取得所有参与 rank 的公共时钟与覆盖证明，所以未给出全组空闲率。

全组共同空白仍可能来自必要 host 决策或同步，消除它要修改真实依赖。共同空白为零，也不排除关键路径有等待：其他 rank 可能只在执行不影响完成时间的旁路工作。因此区间交集只判断同时覆盖，关键路径才解释完成时间由谁决定。

服务吞吐还受 PD 流水线影响。若 P 供给不足，缩短 D 的局部等待不一定增加完成请求。P 交付、D 组批和客户端在途共同决定产出，[尾块合批实验](decode-fast-throughput-limited-by-prefill.md)就曾得到 TPOT 略升而系统吞吐增加的结果。

即使理想情况下能删除固定工作 T 中的 `0.19T`，保持工作量和输出不变，吞吐倍率也应算 `1/0.81≈1.2346`，约提高 23.46%，并非直接提高 19%。这个算式只是教学假设；本次没有测定可删关键路径时间，不能用作收益预测或上界。

## 6. 旧 trace 与采集扰动，还会再改变一次判断

trace 对应 P/D 软件为 `0.0.0.dev18433+gce0a8fea5.d20261013`，并含当时适配。IndexCache 关闭，D 使用草稿 3/验证宽度 4、Graph `[1,2,4,8,12,16]`，未包括后来档位 6、G5/Compact 等变化。因此 18.77% 不能直接标到后续最优配置上。

采集为 CPU+NPU、Level1、PipeUtilization，`record_shapes=True`、`with_stack=False`，有形状但无完整 Python 栈。[PyTorch Profiler 文档](https://docs.pytorch.org/docs/2.14/profiler.html)说明形状和栈记录有额外开销，[Ascend 服务采集说明](https://github.com/Ascend/msserviceprofiler/blob/60b77faa071b16e54b76dc27d470206d63820980/docs/en/msserviceprofiler_serving_tuning_instruct.md#L81-L90)也区分层级成本。这能说明存在扰动可能，不能量化本次扰动。

本次未做同条件的采集开关配对，无法认定 19% 全由 profiler 造成，也无法保证它等同未采集服务。约 2.085 秒窗口与 28 次相关标记，更不能代表完整 AISBench native 的吞吐、AL 和逐轮 Draft/Verify 成本。

原计划曾写 160 请求，实际独立核对为成功 640。文章以实际产物身份为准，并未将 profiling 运行当作正式性能得分。确认 trace 属于哪轮、何种负载，应早于进一步解释百分比。

## 7. 怎样把空隙变成一个值得做的实验

下一步可以先查具体问题：重复空隙是否由两次向量准备引起，哪些输入确实不变，以及究竟是 host 晚提交还是设备等待 peer。不同答案对应不同改动，无须先给整套服务贴 host-bound 标签。

这次已有形状、执行顺序、提交锚点和 helper 源码，尚缺部分动态路径的唯一来源、跨 rank 对应以及未采集时的阶段成本。局部空隙已确认，但这些缺口仍会影响改动位置与优先级。

验证合包候选时，先看相关 host 调用数和准备时间是否按预期减少，再用相同数据、缓存、采样、并发测未采集的 TTFT、TPOT、ITL、E2E 与 whole/native 吞吐。同步、元数据和输出语义必须保持，不能用省略必要工作获得更好看的时间线。

附带[纯 CPU 区间脚本](../../examples/npu-gap-accounting/gap_accounting.py)和[教学 CSV](../../examples/npu-gap-accounting/example.csv)，可从仓库根目录复算口径：

```bash
python examples/npu-gap-accounting/gap_accounting.py examples/npu-gap-accounting/example.csv
```

示例输出 wall10ms、Duration sum11ms、busy union9ms、gap1ms。脚本包含所有记录，支持显式窗口与边界裁剪，不只筛某 stream。但它只处理单 CSV 区间，没有解决跨 rank 时钟、漏采、关键路径或因果，因此 `gap_ratio` 仍不能表示可回收容量。

这次分析最终缩小了候选范围。两次拷贝有较具体的解释，却因相关成本很小而降级。保留这些记录，下次便能从已确认的局部关系继续，而无须为一个未经测定的 19% 收益先投入重启和压测。

## 配套数据与延伸阅读

- [rank0/1 四种时间统计 CSV](../../assets/npu-timeline-gaps-and-recoverable-speedup/rank-clocks.csv)、[脱敏证据 JSON](../../assets/npu-timeline-gaps-and-recoverable-speedup/evidence.json)：原数值、任务锚点、采集配置、源码 SHA 与未观测项。
- [在 Ascend 910C 上分析 DSpark](deepseek-v4-speculative-decoding-ascend-profiling.md)：阶段、Graph 与 NPU 时间线的对应方法。
- [Graph、多流与通信的执行依赖](deepseek-v4-speculative-ascend-performance-critical-path-analysis.md)：进一步理解关键路径与重叠。
- [合理优化没有吞吐收益的负例](ascend-910c-inference-tuning-negative-results-and-stop-rules.md)：如何用实验改变优先级。

[返回性能分析](README.md)
