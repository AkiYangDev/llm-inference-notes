# Ascend 950DT 单机 PD 调优：128K/1K、90% Cache 场景下的配置与实测

这次优化针对 Ascend 950DT 单机上的 SGLang PD 分离部署：输入约 128K token，输出约 1K token，通过共享长前缀实现约 90% 的缓存命中。目标是缩短首字等待、提高整轮输出吞吐。即使九成输入能够复用，每条请求仍有一万多个 token 需要新计算；P 端处理完输入后，还要完成向 D 端的状态交接。

调整主要分为两部分。P 端关闭 HiCache、多流和 WAR barrier，保留设备上的前缀缓存；D 端启用 DSPARK，再配套扩大 DeepEP dispatch 容量、细化图捕获档位。原始脚本已有的量化、KV 缓存精度和内存分配配置继续沿用。

调整后，一轮 192 请求、并发上限 48、到达速率 2 req/s 的测试完成了全部请求，平均输出吞吐为 1782.61 tok/s，平均 TPOT 为 3.55 ms，平均 TTFT 为 12.04 s。生成阶段已经较快，首 token 前的时间仍占平均请求时长的约 77%。这轮结果也给出了后续调优的重点：提高请求供给，并拆开检查输入处理、排队和 PD 交接的耗时。

下文按 P、D 两端说明配置变化，再结合各轮压测分析结果。模型统一记为 DeepSeek 系列 MoE 模型；内部版本、地址和账号目录已脱敏，参数片段对应本次使用的 SGLang Ascend 适配环境。

## 一、一条 128K 请求，命中缓存之后还剩什么

先看部署。P 和 D 位于同一台机器，分别运行服务实例，由 Router 协调请求。P 处理输入、准备缓存状态，D 接收所需状态并继续生成。两端拥有各自的调度过程，单机部署依然要完成 P→D 的交接。

[![单机 PD 的逻辑分工：Router 协调 P 和 D，设备前缀缓存保留，HiCache 主机缓存路径关闭](../../assets/ascend-950dt-single-node-pd-optimization/pd-cache-paths.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/pd-cache-paths.svg)

*图 1：逻辑分工示意。连线区分请求协调、设备缓存复用和 P→D 状态交接，不表示所有步骤严格串行；图中不指定物理卡号。 [查看大图](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/pd-cache-paths.svg)*

TTFT 描述从发出请求到收到第一个 token 的时间；TPOT 描述首 token 之后，平均每个输出 token 花了多久。本文的输出吞吐则用整轮生成 token 总数除以压测时长，覆盖了 P、D 和交接过程。三个指标分别回答“多久开始”“开始后多快”和“整套服务总共完成多少工作”。

| 场景要素 | 本次设置 |
| --- | --- |
| 硬件 | Ascend 950DT，单机 |
| 服务框架 | SGLang，Ascend 后端 |
| 部署方式 | Prefill/Decode 分离 |
| 模型类型 | DeepSeek 系列 MoE |
| 输入 / 输出长度 | 约 128K / 1K token |
| 输入复用 | 共享长前缀，目标缓存命中率约 90% |
| 主要指标 | TTFT、TPOT、端到端平均输出吞吐 |

90% Cache 看起来已经很高，但输入足够长，剩下的工作仍然不少。以 131072 token 为例，10% 未命中就是约 13107 token。几十条请求陆续进入 P，这些后缀需要排队、组批、计算，并生成相应的缓存状态。

而且，新 token 的 Attention 还要结合已有上下文。缓存命中省去了可复用前缀的重复计算，并没有把一条 128K 上下文请求变成一条完全独立的 13K 请求。后续 D 也要在已有长上下文的基础上继续生成。

高命中场景还有一个容易被百分比掩盖的变化：

| 输入长度固定为 131072 token | 90% 命中 | 95% 命中 |
| --- | ---: | ---: |
| 需要新计算的 token | 约 13107 | 约 6554 |
| 未命中工作量的相对变化 | 基准 | 减少约一半 |

因此，命中率从 90% 到 95%，对 P 的影响可能很大。比较不同机器或不同配置时，平均每请求未命中 token 数比单独一个命中百分比更能说明输入计算的工作量。

TTFT 则是客户端视角的时间。入口处理、分词、排队、输入计算，以及首 token 返回路径上实际经历的交接和调度，都可能影响它。第一枚 token 在哪里产生、何时送回，取决于具体实现，所以不能把整个 TTFT 都算成 P 的设备计算时间。

例如，D 的生成速度提高后，如果 P 仍在排队处理输入，D 就可能缺少可执行的请求。此时 TPOT 降低，整轮输出吞吐却不一定同比增长。

## 二、P 端关闭 HiCache，保留设备前缀复用

原始 P 脚本开启了层次化缓存。除了设备 KV，还配置了主机缓存后端，并使用 write-through 写入策略：

```bash
--enable-hierarchical-cache \
--hicache-io-backend kernel_ascend \
--hicache-ratio 1.5 \
--hicache-write-policy write_through
```

HiCache 解决的是缓存容量和复用范围的问题。设备放不下的历史前缀，如果能留在更大的缓存层里，后续请求就有机会通过搬运恢复状态，省去重复计算。[官方文档](https://docs.sglang.ai/advanced_features/hicache_best_practices.html)将主机容量、I/O 后端和写入策略分别配置，正是因为容量收益与搬运成本需要一起考虑。

这次负载的特点，是反复使用一个共享长前缀。主要复用已经发生在设备上，额外的主机缓存层没有表现出足够的收益。关闭 HiCache 后，现场性能有提升，后面的实验便保留了这个设置。

关闭 HiCache 后，设备前缀缓存和 `--enable-cache-report` 继续保留。前者复用已有前缀，后者记录命中情况，均不依赖开启主机缓存层。

后续一轮的统计是：

```text
Total prompt tokens:  24,972,134
Total cached tokens:  22,788,096
Device cached:        22,788,096
Host cached:                   0
Cache hit rate:            91.3%
```

关闭以后 Host 命中为 0 是正常结果；更有用的信息是，设备缓存仍然完成了 91.3% 的输入复用。这个负载没有因为移除主机缓存层而丢掉主要的缓存收益。

从成本上理解，少一层缓存意味着少了一部分主机内存占用、数据搬运和管理工作。这与观察到的改善相符。这里也有一个统计上的细节：Host 命中统计描述的是请求复用了多少主机缓存，不是写入主机缓存花了多少时间。因此，即使开启 HiCache 时某轮 Host 命中为 0，也不能据此认定那一轮没有相关开销。

这个选择与负载有关。固定共享前缀能够留在设备上时，主机缓存的额外容量较难发挥作用；如果大量不同前缀轮换、设备缓存频繁淘汰，恢复历史前缀带来的收益就可能更大。本次保留 HiCache 关闭，适用的是前一种情况。

还有一条路径不能混淆：HiCache 关闭以后，PD 的状态交接仍然存在。图 1 中，关闭的是设备到主机缓存的那条支路；P→D 的必要传输继续保留。因此，看到 HiCache 被删除，也不应顺手删除 PD 传输依赖的运行环境。

## 三、P 端关闭多流和 WAR barrier，把执行方式一起调整

P 端第二组变化，是关闭多流和 WAR barrier：

| 配置 | 原始状态 | 调整后状态 |
| --- | --- | --- |
| `SGLANG_NPU_USE_MULTI_STREAM` | 开启 | 关闭 |
| `SGLANG_ENABLE_WAR_BARRIER` | 开启 | 关闭 |
| `SGLANG_FORCE_COARSE_WAR_BARRIER` | 开启 | 关闭 |

多流允许不同工作在条件满足时重叠执行。比如某一部分计算尚未结束，另一部分准备工作已经可以推进，就有机会缩短总时间。但重叠需要满足依赖关系，还取决于设备是否有足够资源承接并行工作。

如果两项工作都在争用相同资源，多个流未必带来更短的关键路径；流之间还需要通过事件或其他同步机制协调。看到“开启多流”，只能说明执行组织方式改变了，性能好坏仍要看具体负载。

WAR 指 write-after-read，也就是写入必须等待已有读取完成的依赖。相关 barrier 用于协调这种顺序。本次在 P 端关闭多流，同时关闭了这两项 WAR barrier 配置，作为一组执行方式调整保留下来。

这组调整同时改变了流的组织方式和同步行为，现有结果没有拆分每个开关的独立收益。进一步分析时，需要结合执行时间线看计算与通信的重叠、流间等待，以及完成一批输入的总耗时。

本次 P 的 `chunked-prefill-size` 仍为 `65536`。原始脚本已有的 DeepEP 长序列配置也继续保留。这两部分没有作为本轮新增改动。

## 四、D 端启用 DSPARK，一次验证推进多个 token

原始 D 脚本已经设置了 Spec V2 和 overlap plan stream 相关环境变量，但没有通过启动参数指定投机算法。新版本明确加入了 DSPARK：

```bash
--speculative-algorithm DSPARK \
--speculative-draft-attention-backend ascend \
--speculative-num-draft-tokens 7 \
--speculative-dspark-block-size 6
```

普通自回归生成通常每轮为每条活跃请求推进一个新 token。投机解码会先准备多个候选，再由目标模型验证。通过验证的部分可以一起提交，下一轮从新的边界继续。

[![投机解码概念示意：候选先进入验证，通过的前缀提交，拒绝之后的后缀不提交](../../assets/ascend-950dt-single-node-pd-optimization/speculative-cycle.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/speculative-cycle.svg)

*图 2：为解释接受边界，示意画出 6 个候选、其中前 4 个通过。数字不是本次运行的接受长度；修正或额外 token 的处理没有展开。 [查看大图](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/speculative-cycle.svg)*

投机的收益，可以用“一个周期花了多少时间，最终提交了多少 token”来理解：

```text
单个有效输出 token 的平均成本
≈ 一个 draft/verify 周期的总耗时 ÷ 该周期提交的有效 token 数
```

假设一个周期耗时 20 毫秒，提交 5 个有效 token，摊下来是每 token 4 毫秒。另一个配置虽然能提交 6 个，但周期增加到了 30 毫秒，摊下来反而是 5 毫秒。这只是说明取舍的算例：接受更多 token 有价值，也要看为此多花了多少工作。

这里有三类不同的数字：`draft tokens` 和 `block size` 是配置值；接受率、平均接受长度反映实际验证情况；TPOT 和输出吞吐描述最终运行效果。配置成 7，并不意味着每轮固定提交 7 个有效 token。

DSPARK 也改变了 D 内部一次执行所处理的 token 规模。普通 decode 的一条请求常对应当前步一个 token，投机验证可能一次处理多个位置。送进 Attention、MoE 和通信环节的工作量发生变化，后面的容量和图配置也必须跟上。

因此，启用 DSPARK 后还要检查验证 batch 的通信容量和图覆盖，避免新增的工作量触发容量不足，或产生过多填充计算。关于周期耗时与有效产出的关系，可以继续对照本站的[投机解码性能分析](deepseek-v4-speculative-performance-analysis.md)。

## 五、给投机执行补齐容量，再减少图中的填充工作

打开 DSPARK 后，D 端需要处理更大的 token batch。调试中遇到过图捕获阶段的容量断言：

```text
num_max_dispatch_tokens_per_rank >= x.size(0)
```

它检查的对象是传入张量的 token 行数。这个数与当前执行阶段、请求 batch、每请求验证 token 数和 padding 有关。用客户端并发直接代替它，往往会低估实际需要的容量。

原始公共环境中的 dispatch 上限为 35，P 分支随后覆盖为 128；原始 D 继承的是 35。D 端中间试验用过 128，最终调整到 512，解决了这次图捕获中的容量不足。

如果只是帮助理解，可以假设某个 rank 有 8 条请求、每条一次处理 7 个验证位置，那么在没有其他布局变化的简化条件下，就是 56 行工作，而不是 `8 + 7`。实际断言仍然以出错阶段的 `x.size(0)` 为准。

D 端的主要配套改动如下：

| 配置 | 原始 D | 调整后 D | 本次用途 |
| --- | --- | --- | --- |
| Dispatch token 上限 | 35 | 512 | 容纳投机执行的 token batch |
| `DEEPEP_HCCL_BUFFSIZE` | 400 | 2600 | 扩大通信缓冲配置 |
| 图捕获 batch 档位 | 1、2、4、16 | 1、2、3、4、6、8、10、12、14、16 | 增加常见小 batch 的覆盖 |
| `max-running-requests` | 64 | 196 | 扩大请求容量 |
| `chunked-prefill-size` | 16384 | -1 | 关闭 D 端 chunked prefill |

图配置从 `--cuda-graph-bs` 改为本次分支使用的 `--cuda-graph-bs-decode`。名称保留了框架 CLI 的写法，实际运行硬件仍是 Ascend NPU。

图捕获通常预先准备若干形状，以便后续重复执行。实际 batch 与预先捕获的档位不完全一致时，某些执行路径会 padding 到合适档位。原来 4 到 16 之间跨度较大，加入 6、8、10、12、14 后，有机会少算一些填充位置。

[![图捕获 padding 示意：实际 batch 为5时，旧档位填充到16，新档位可填充到6](../../assets/ascend-950dt-single-node-pd-optimization/graph-padding.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/graph-padding.svg)

*图 3：假设 runner 选择不小于实际 batch 的最近档位。示例只比较填充请求槽位，不代表时间按同样比例缩短。 [查看大图](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/graph-padding.svg)*

图中的 5 条请求只是一个局部 batch 例子。客户端 C48 描述整条服务链路的在途上限，某个 DP worker、某个 runner 实际执行的 batch 还受到分配与执行阶段影响。把两者混成一个数，很容易出现“客户端48，为什么只捕获到16”的误解。

更多档位也会增加捕获和图资源成本，因此目标是覆盖经常出现的 batch。原始 D 已经使用了图执行，本次调整的是覆盖档位和适用入口。

通信侧还加入了原先在 P 端使用的配置：

```bash
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
```

这些参数记录为当前组合中的通信配置。它们是否影响某一轮 D，要看是否进入对应的 normal 长序列路径。长上下文需要 D 持续访问历史状态，但不会让它每一步都把全部 128K 历史 token 重新送进 MoE。MoE 的工作量要从当前步或当前 verify batch 计算。

请求上限从 64 扩大到 196，为后续更高负载预留了容量。本文完整测试的客户端上限为 48、峰值为 39，尚未触及原来的 64，因此这轮还不能体现提高上限的效果。D 端同时关闭 chunked prefill，与其承担生成阶段的分工相配合。

最后还整理了日志和运行路径。P、D 日志统一保存，保留时间和有效配置，方便把一轮压测与两端运行状态对齐。代码路径切换要记录实际 import 位置和版本，模型目录变化要确认权重是否一致。`expandable_segments:True` 在原始公共环境里已有，这次只是继续使用。

## 六、192 请求的完整压测：吞吐、时延与实际工作量

服务逐步稳定以后，压测方式成了另一个影响结果的变量。前面有些测试只有几十条请求，发完以后很快就进入收尾。这样的平均吞吐对起步和结束阶段很敏感，看起来会比运行中间的输出速度低不少。

后续把总请求数增加到 192。为方便区分，下面用 N 表示请求总数、C 表示客户端并发上限、rate 表示请求到达速率。N192/C48/rate2 的意思是，总共发送 192 条，最多 48 条同时在途，按平均每秒 2 条请求的配置发出。

本轮生成共享前缀数据的关键参数是：

```bash
--gsp-num-groups 1 \
--gsp-prompts-per-group 192 \
--gsp-system-prompt-len 117965 \
--gsp-question-len 9216 \
--gsp-output-len 1024 \
--gsp-range-ratio 1 \
--max-concurrency 48 \
--request-rate 2 \
--warmup-requests 16 \
--seed 100
```

这是该轮参数摘录。128K/1K、90% Cache 是测试场景的简称，输入构造长度与最终 tokenizer 统计存在差异，复盘时使用实际 token 数。日志中的 `flush_cache=False` 表示本次没有通过 benchmark 的该开关清理缓存；是否在外部清理、预热做了什么，应另外记录。

本轮完整结果如下：

| 指标 | 结果 |
| --- | ---: |
| 成功请求 | 192 / 192 |
| Benchmark duration | 110.29 s |
| 总输入 token | 24,972,134 |
| 总缓存 token | 22,788,096 |
| 缓存命中率 | 91.3% |
| 总输出 token | 196,608 |
| 平均输出吞吐 | 1782.61 tok/s |
| 平均 TTFT | 12,037.73 ms |
| P95 TTFT | 14,698.48 ms |
| 平均 TPOT | 3.55 ms |
| 平均 E2E | 15,669.95 ms |
| 平均在途并发 | 27.28 |
| 峰值并发 | 39 |

图 4 只使用这一轮数据。把请求从发出到结束的平均时间拆开，首 token 前的占比一眼就能看出来。

[![本轮平均请求时延15.67秒，其中TTFT为12.04秒，首token之后为3.63秒](../../assets/ascend-950dt-single-node-pd-optimization/latency-budget.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/latency-budget.svg)

*图 4：按同一批请求的平均 TTFT 和平均 E2E 绘制。右侧为两者之差；颜色没有进一步分解分词、P 计算、排队或 KV 交接时间。 [查看大图](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/latency-budget.svg)*

平均每请求的输入与命中情况，也可以从这轮统计直接算出来：

```text
平均输入     = 24,972,134 / 192 ≈ 130,063 token
平均命中     = 22,788,096 / 192 = 118,688 token
平均未命中   = 2,184,038 / 192  ≈ 11,375 token
```

这一轮平均每请求有约 11.4K token 未命中。后续比较另一台机器或另一组配置时，需要先对齐这部分工作量，再比较 TTFT。

N192 这轮还说明，C48 并没有持续用满。请求以 2 req/s 的配置到达，峰值只有 39。每条输出 1024 token，对应的长期输出供给量级是 2048 tok/s；本轮平均 1782.61 tok/s，约为这个量级的 87%。有限测试中的到达波动、起步和收尾都会影响比例，所以这个数用于判断供给规模，不是设备利用率。

日志还给出了 5191 tok/s 的峰值输出吞吐。峰值可以高于长期供给量级，因为前面到达的请求可能在某个窗口集中输出。整轮平均 1782.61 tok/s 包含起步、持续运行和收尾，更适合用来比较整套服务的表现。

下一轮计划固定 N192/C48，将 rate 从 2 提高到 4，观察增加请求供给后，平均输出吞吐能否继续增长，以及 TTFT 和排队是否明显增加。这可以帮助判断当前吞吐受请求到达速率限制的程度。

缓存准备也需要固定。同一个 seed 反复运行，可能复用前一轮留下的请求后缀；预热也可能覆盖正式请求的一部分内容。客户端读取的生成数据 `.pkl` 文件只负责复用输入数据，服务端 KV 是否命中则取决于缓存状态。比较时应保留相同输入、清理流程和预热条件，再核对实际未命中 token 数。

## 七、整体结果与下一步：缩短首 token 前的时间

把整个过程放在一起看，早期与后续的一组代表性记录如下：

| 指标 | 早期记录 | 后续代表性记录 |
| --- | ---: | ---: |
| 客户端并发上限 | 160 | 48 |
| 缓存命中率 | 88.6% | 92.1% |
| 平均 TTFT | 50.8 s | 7.5 s |
| 平均 TPOT | 17.3 ms | 4.45 ms |
| 平均输出吞吐 | 约 750 tok/s | 1626 tok/s |

这组记录保留了调优过程的整体变化：首字等待明显缩短，生成速度和平均输出吞吐也改善了。它同时包含服务参数、并发和缓存状态的变化，因此适合描述最终组合的效果，而不适合给某个开关单独分配百分比收益。

例如，相同输入量下，命中率从 88.6% 到 92.1%，未命中比例由 11.4% 降到 7.9%，需要新计算的 token 就减少了约 30.7%。并发上限从 160 降到 48，也会改变排队情况。这些变化都会影响 TTFT。

上表是调优前后的两组历史记录；上一节的 12.04 s TTFT、3.55 ms TPOT 和 1782.61 tok/s 吞吐则来自后续 N192 测试，三者属于同一轮结果。

当前 N192 测试中，首 token 前占平均请求时长的约 77%。继续优化时，需要检查这一段的主要耗时：P 等待队列是否持续增长，输入计算是否占满设备，P 完成之后是否等待 D 接入，以及分词和 Router 是否还有明显排队。

如果 P 持续有队列而 D 的生成请求较少，应优先改善 P 的供给；如果 P 完成后迟迟没有进入生成，应重点检查交接和 D 的接收条件；如果 D 的 KV 容量已经紧张，则要先处理容量与接入。需要用两端的队列状态和时间线确定具体原因。

选择下一组配置时，也可以接受一定的 TPOT 增加，换取更短的整体请求时间。用一个假设算例说明：输出 1024 token，TTFT 为 12 s、TPOT 为 3.55 ms 时，E2E 约为 15.63 s；若 TTFT 降到 7 s、TPOT 升到 5 ms，E2E 仍可降到约 12.12 s。最终是否值得采用，还要同时看平均吞吐和尾部时延。

本轮保留 P 端关闭 HiCache、多流和 WAR barrier，D 端启用 DSPARK、扩大 dispatch 容量并细化图档位的组合。下一轮先保持这些配置不变，增加请求到达速率，并对齐两端日志：既观察吞吐能否提高，也定位新增等待发生在哪个阶段。

延伸阅读：[SGLang PD 分离说明](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)、[HiCache 配置](https://docs.sglang.ai/advanced_features/hicache_best_practices.html)、[本站 PD 交接源码解析](../sglang/deepseek-v4-pd-disaggregation-source-analysis.md)。公开文档用于理解机制与配置类别，具体参数以运行版本为准。
