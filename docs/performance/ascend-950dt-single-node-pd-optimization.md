# Ascend 950DT 单机 PD 调优：128K/1K、90% Cache 场景下的配置与实测

这次在一台 Ascend 950DT 上调单机 PD：输入约 128K Token、输出约 1K Token，九成左右的输入复用缓存。P 服务计算 Prefill，D 服务接管后续生成。

改动集中在两端：P 关闭 HiCache、多流和 WAR barrier；D 启用 DSPARK，再补足通信容量和图捕获档位。代表性记录的输出吞吐由约 750 tok/s 提高到 1626 tok/s，但两组的并发和缓存命中率也发生了变化。下面先列出条件，再沿缓存与 Decode 执行路径解释配置，最后分析一轮单独留存的完整压测。

## 一、负载条件与结果口径

P、D 虽然在同一台机器上，仍是各自运行的服务实例。Router 协调请求，P 计算输入并准备缓存状态，D 收到所需状态后继续生成；两端的调度和状态传输都属于这条请求路径。

模型为 DeepSeek 系列 MoE，启动脚本基于本次使用的 SGLang Ascend 适配环境。文中保留参数写法，省略内部版本标识和部署地址等信息。

先看两条代表性记录，后文的完整压测会单独列出：

| 指标 | 早期记录 | 调优后记录 |
| --- | ---: | ---: |
| 客户端并发上限 | 160 | 48 |
| 实际缓存命中率 | 88.6% | 92.1% |
| 平均 TTFT | 50.8 s | 7.5 s |
| 平均 TPOT | 17.3 ms | 4.45 ms |
| 平均输出吞吐 | 约 750 tok/s | 1626 tok/s |

TTFT 统计首 Token 前的等待，TPOT 统计后续 Token 的平均生成时间，输出吞吐则按整轮完成的输出和耗时计算。这些指标覆盖的时间范围不同。

表里除了配置，并发也从 160 降到 48，命中率从 88.6% 升到 92.1%。仅非命中比例就从 11.4% 降为 7.9%，减少约 30.7%。因此，750 到 1626 tok/s 是整组条件变化后的结果，不能逐项分摊到某个开关。

两端配置的主要变化如下：

| 调整位置 | 主要变化 | 调整目的 |
| --- | --- | --- |
| P 端 | 关闭 HiCache、多流和 WAR barrier | 调整缓存层级与执行方式 |
| D 端算法 | 启用 DSPARK，draft tokens 设为 7，block size 设为 6 | 让一次验证推进多个有效输出 token |
| D 端运行配置 | 扩大 dispatch 容量、增加图档位、调整通信缓冲 | 容纳更大的验证 batch，减少填充工作 |

两端还沿用了 `--quantization fp8`、`--kv-cache-dtype fp8_e4m3` 和 `expandable_segments:True`。这些配置属于本轮运行背景。

## 二、P 端：关闭 HiCache，保留设备前缀复用

P 端先尝试关闭 HiCache。观测到性能改善后保留了这一配置，设备侧的前缀缓存仍然使用。

原配置包含以下主机缓存选项：

```bash
# 原始 P 配置；本次删除了这些 HiCache 参数
--enable-hierarchical-cache \
--hicache-io-backend kernel_ascend \
--hicache-ratio 1.5 \
--hicache-write-policy write_through
```

HiCache 的作用是扩大可复用缓存的容量。历史前缀被移出设备后，如果主机上仍保留着相应状态，后续请求就有机会将其恢复，避免重新计算。容量收益伴随着主机内存占用、缓存写入和搬运成本，具体取舍与前缀的复用方式有关。[HiCache 官方文档](https://docs.sglang.ai/advanced_features/hicache_best_practices.html)分别介绍了容量、I/O 后端与写入策略。

本轮请求反复使用同一个共享前缀，设备缓存足够容纳它。这个条件下，关闭主机缓存可能减少额外维护和搬运，但已有记录没有把各段成本单独测出。换成许多不同前缀、设备缓存容纳不下的负载，还需重新比较容量收益。

[![单机 PD 缓存路径：设备前缀复用和 P 到 D 的状态传输保留，主机缓存支路关闭](../../assets/ascend-950dt-single-node-pd-optimization/pd-cache-paths.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/pd-cache-paths.svg)

图中关闭的是主机缓存支路，设备前缀复用以及 P 到 D 的状态传输仍然保留。

后面的完整压测记录到 91.3% 命中，命中部分全部来自 Device，Host 为 0。`--enable-cache-report` 继续保留，两端原有的 PD 环境配置也没有随 HiCache 一起关闭。

128K 输入即使命中九成，仍有约 13K Token 需要处理，还要在计算中访问历史状态。高命中率并没有消除 P 端的工作。

P 端同时关闭了多流与 WAR barrier：

| 配置 | 原始状态 | 调整后 |
| --- | --- | --- |
| `SGLANG_NPU_USE_MULTI_STREAM` | 开启 | 关闭 |
| `SGLANG_ENABLE_WAR_BARRIER` | 开启 | 关闭 |
| `SGLANG_FORCE_COARSE_WAR_BARRIER` | 开启 | 关闭 |

多流试图重叠计算与通信，也要处理流间依赖。WAR 是 write-after-read，即写入要等已有读取完成。本轮把多流和相关 barrier 成组关闭，未分别隔离收益，因此结果只能支持这组配置的选择。

P 端 `chunked-prefill-size` 仍为 `65536`，DeepEP 的长序列配置也未改变，避免把更多变量混入这轮比较。

## 三、D 端：用 DSPARK 增加每轮有效输出

D 端原来已有 SpecV2 和 overlap 相关环境配置，但没有设置本次使用的投机算法参数。接下来补上 DSPARK 配置：

```bash
--speculative-algorithm DSPARK \
--speculative-draft-attention-backend ascend \
--speculative-num-draft-tokens 7 \
--speculative-dspark-block-size 6
```

普通自回归迭代通常推进一个输出位置；DSPARK 先生成候选，再由 Target 验证，一轮可以提交多个 Token。能否缩短每 Token 耗时，要同时看这轮多做了多少计算和最终提交多少输出。

[![投机解码概念示意：六个候选进入验证，前四个通过并提交，后面的候选不提交](../../assets/ascend-950dt-single-node-pd-optimization/speculative-cycle.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/speculative-cycle.svg)

图中只是一个候选验证的例子：通过的部分被提交，未通过的后续候选不进入已确认输出。

对单条请求，可以把它经历的一轮 Draft/Verify 耗时，除以该周期为它提交的有效 Token 数，估算每个输出的平均时间成本。

例如，20 ms 提交 5 个 Token，摊到每 Token 为 4 ms；如果为了提交 6 个 Token 把周期拉长到 30 ms，反而变成 5 ms。batch 中的总工作量也要纳入同一周期比较。

本次 `draft tokens=7`、`block size=6` 是执行配置，实际每轮接受多少，需要看运行统计。TPOT 和整轮输出吞吐则反映这部分收益最终是否兑现。关于接受长度与周期成本的关系，可参见[投机解码性能分析](deepseek-v4-speculative-performance-analysis.md)。

这还要求通信和缓冲区能够容纳 Verify 实际展开的行数。客户端请求数不能直接代替这些行数。

## 四、通信容量与图捕获配置

### Dispatch 容量为什么要扩大

运行中触发了下面的容量断言：`num_max_dispatch_tokens_per_rank >= x.size(0)`。

断言比较的是当前 Rank 的通信容量和输入实际行数，行数可能已经包含投机展开及填充。

举例说，8 条请求若各展开 7 行，就有 `8 × 7 = 56` 行。这个示例用于解释容量关系，不能拿来直接代替本次实际执行形状。

`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` 原值为 35。调整过程中 P 使用 128，D 从 35 改到中间值 128，最终设为 512，以满足本次固定图捕获的要求。最终值来自实际执行约束，不能由上面的教学示例推出。

通信容量的变化汇总如下：

| 配置 | 原始 D | 调整后 D |
| --- | --- | --- |
| Dispatch token 上限 | 35 | 512 |
| `DEEPEP_HCCL_BUFFSIZE` | 400 | 2600 |
| 图捕获 batch 档位 | 1、2、4、16 | 1、2、3、4、6、8、10、12、14、16 |
| `max-running-requests` | 64 | 196 |
| `chunked-prefill-size` | 16384 | -1 |

### 为什么补充中间图档位

D 原先已经使用图捕获，这次把参数入口从 `--cuda-graph-bs` 改为 `--cuda-graph-bs-decode`。参数保留 CUDA 命名，但此处运行在 NPU 上。

原有档位从 4 直接跳到 16。增加 6、8、10、12、14 后，中间规模的 batch 可以选更接近的档位，减少为了复用图而补入的行。

[![图填充示例：实际 batch 为五条请求，原档位补齐到十六，新档位补齐到六](../../assets/ascend-950dt-single-node-pd-optimization/graph-padding.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/graph-padding.svg)

图中实际有 5 条请求：原档位要补到 16，新增档位只需补到 6，填充的请求槽位由 11 降为 1。这里数的是请求槽位；每请求若在 Verify 中展开多行，实际填充的 Token 行数还需乘对应宽度。

这提供了减少填充计算的机会，但耗时不会必然按填充行数同比下降。更多档位也要付出捕获时间和内存成本。此外，客户端 C48 只限制在途请求，不能直接用来选择每个 Worker 的图档位。

### 其余容量与通信配置

通信 buffer 从 400 增加到 2600；长序列配置继续保留，D 侧补上所需配置：

```bash
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=8192
export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
```

这些配置是否进入执行取决于实际通信分支。128K 是历史上下文长度，不表示 MoE 每一步都新增 128K 行，容量仍应按本轮实际处理的输入解释。

`max-running-requests` 从 64 提高到 196，但报告中的峰值为 39，没有触及原来的 64，因而这轮数据不能证明放宽上限带来了收益。D 的 `chunked-prefill-size=-1` 则关闭了其分块 Prefill。

日志也统一了保存方式：原先 P 使用 tee，D 未同样留存，调整后保存两端完整初始化日志。代码和权重目录发生切换时，还需一起记录身份信息，才能知道结果究竟对应哪份环境。

## 五、完整压测的吞吐和时延

以下是一轮 N192、C48、请求速率 2 的完整测试。它和开头平均 TTFT 7.5 s 的代表记录是不同运行，数值需要分别看。

N 决定整轮请求数量，C 限制最多同时在途多少条，速率决定新请求进入的节奏。只提高 C 并不保证服务持续收到足够负载。

测试使用共享前缀、目标输出 1,024，预热 16 条，seed 为 100，具体参数如下：

<details markdown="1">
<summary>本轮压测参数摘录</summary>

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

命令记录中的 `flush_cache=False` 表示客户端没有启用相应清理选项；外部准备步骤仍需保持一致，才能比较各轮缓存状态。

</details>

192 条请求全部成功，平均 TTFT 为 12.04 s。这个值对应下面的完整运行，和开头的 7.5 s 分别归档。

| 指标 | N192 / C48 / rate2 |
| --- | ---: |
| 成功请求 | 192 / 192 |
| 测试时长 | 110.29 s |
| 实际缓存命中率 | 91.3% |
| 平均输出吞吐 | 1782.61 tok/s |
| 平均 TTFT | 12,037.73 ms |
| P95 TTFT | 14,698.48 ms |
| 平均 TPOT | 3.55 ms |
| 平均 E2E | 15,669.95 ms |
| 平均在途并发 / 报告峰值并发 | 27.28 / 39 |

### 输入究竟还有多少新工作

本轮实际输入 24,972,134 Token，缓存 22,788,096 Token，命中部分全部来自 Device，Host 为 0。除以 192 条请求后，平均输入与缓存量如下：

| 平均每请求 | token 数 |
| --- | ---: |
| 输入 | 约 130,063 |
| 缓存命中 | 118,688 |
| 未命中 | 约 11,375 |

平均每条仍有约 11.4K 非命中 Token。对 128K 负载而言，命中从 90% 升到 95% 会让非命中部分减半，所以比较性能时需要对齐实测命中量。

重复使用相同 seed 时，独立后缀也可能进入缓存。`.pkl` 保存输入数据，不代表恢复同样的缓存状态；数据、清理顺序和预热方式应一起固定。

### 首 token 之前占了多少时间

平均 E2E 为 15.67 s，减去 TTFT 12.04 s，首 Token 后还需约 3.63 s。两部分约占 77% 和 23%，首 Token 前的时间已经更值得追踪。

[![平均请求时长分布：首 token 前十二点零四秒，后续生成三点六三秒](../../assets/ascend-950dt-single-node-pd-optimization/latency-budget.svg)](https://raw.githubusercontent.com/AkiYangDev/llm-inference-notes/main/assets/ascend-950dt-single-node-pd-optimization/latency-budget.svg)

图按平均请求耗时划分这两部分，尚未继续拆开首 Token 前的各段。

要进一步区分 TTFT 中的排队、P 端计算、状态交接和返回开销，需要对应日志或时间线，并确认本分支在什么时刻返回首 Token。

### 到达速率如何影响平均吞吐

本轮共输出 196,608 Token，整轮平均吞吐为 1782.61 tok/s。5191 的峰值来自局部时间窗口，不能直接当成持续容量。

请求速率为 2，每条输出 1,024 Token，持续供给对应的输出量约为 `2 × 1024 = 2048 tok/s`。

实测均值约为这一供给量的 87%，这个比例并非设备利用率。平均在途数为 27.28，报告峰值为 39，尚未达到 C48；峰值本身也受客户端时间桶统计方式影响。

因此，请求到达速率可能已经限制了这轮吞吐。下一步可以提高到达速率，同时观察队列是否持续增长，才能判断服务还有多少余量。

## 六、配置选择与后续对照

本轮最终保留了适用于该单机 PD、共享前缀负载的配置：关闭 HiCache 后仍有约九成设备缓存命中，D 使用 DSPARK，并补足通信容量和图档位。

下一组对照可以保持 N192、C48，把请求速率由 2 调到 4，再比较平均与尾部 TTFT、P/D 队列以及状态交接时间。这只是后续验证方案，本轮没有执行。

选择配置时也可以按完整请求耗时判断：`E2E ≈ TTFT + (输出长度 - 1) × TPOT`。例如输出 1,024，TTFT 为 12 s、TPOT 为 3.55 ms 时，约需 15.63 s；另一组 TTFT 为 7 s、TPOT 为 5 ms，约需 12.12 s。后者即使逐 Token 较慢，完整请求仍可能先完成。

现有完整记录中，首 Token 后生成约 3.63 s。继续优化前先提高负载供给、拆解首 Token 前的等待，可以更清楚地判断下一步该改 P、D 还是调度。

延伸阅读：[SGLang PD 分离](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)、[HiCache 配置](https://docs.sglang.ai/advanced_features/hicache_best_practices.html)、[PD 交接源码解析](../sglang/deepseek-v4-pd-disaggregation-source-analysis.md)。
