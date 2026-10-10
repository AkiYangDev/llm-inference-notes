# 32k/1k 共享前缀与并发测试

配套文章：[从缓存校准到并发取舍：DeepSeek-V4-Pro 在 Ascend 910C 上的一次 32k/1k 调优记录](../../docs/performance/deepseek-v4-pro-32k-prefix-cache-concurrency-tuning.md)。

这个脚本保留了文章中的客户端参数，向已经运行的 SGLang 服务发送请求，不启动或重配服务。先确认目标分支支持 `python3 -m sglang.benchmark.serving --help` 中对应的选项。示例是根据原测试命令整理的可复用版本，未在新的 NPU 环境上重跑。

在能访问服务、模型 Tokenizer 和对应 SGLang Python 包的客户端环境执行。服务必须处于可清缓存的空闲状态。将以下地址和路径换为自己的环境值，再选一个并发运行：

```bash
export BENCH_HOST=127.0.0.1
export BENCH_PORT=6677
export BENCH_MODEL=/path/to/DeepSeek-V4-Pro-W4A8
export BENCH_SERVED_MODEL=dsv4

bash examples/deepseek-v4-pro-32k-tuning/benchmark.sh 24 96
```

同一份数据的另外两个并发点分别为：

```bash
bash examples/deepseek-v4-pro-32k-tuning/benchmark.sh 36 96
bash examples/deepseek-v4-pro-32k-tuning/benchmark.sh 72 96
```

每次脚本都会先调用 `/flush_cache?timeout=60`，再让 benchmark 执行 8 条预热。正式测试不传 `--flush-cache`；在文章核对的公开快照中，该选项会在预热后清缓存。客户端启用 `SGLANG_IS_IN_CI` 时也可能触发预热后清理，本示例对已识别的真值直接报错，避免悄悄改变缓存负载。

`31900/868` 只保留当轮的构造方式，命中率不会由脚本强制固定到 91.8%。目标总输入为 32,768，原日志重新分词后的实际平均输入约 33.5k。更换模型 Tokenizer、路由、分支或缓存配置后，都需要重新核对输入长度与实测命中率。

日志、JSONL、客户端环境信息保存在结果目录，默认是当前工作目录下的 `bench_out`，可通过 `BENCH_OUT_DIR` 指定。运行时建议指定仓库外的目录，避免将包含真实环境信息的原始日志提交到公开仓库。环境记录包括实际导入路径、包版本、能够取得的客户端 commit、benchmark 的帮助信息和服务模型列表。远端服务的完整 commit、镜像摘要及启动参数需要另外归档；本脚本不能从客户端替服务端证明这些字段。

[check_result.py](check_result.py) 检查正式请求数、总生成量、逐请求输出长度、非空错误，并打印平均输入、缓存命中和主要性能字段。它检查结果结构与输出完整性，不验证模型回答质量。原始记录 CSV 在 [measurements.csv](../../assets/deepseek-v4-pro-32k-tuning/measurements.csv)，空字段表示没有保留，不代表零。

文章中的服务基线为 Ascend 定制分支上的 TP32/DP8、DP Attention、DSpark 与 DeepEP。示例不提供可跨版本使用的服务启动命令，迁移时需要先确认目标分支的并行与投机推理兼容性。

重绘文章中的图表需要 Matplotlib；在仓库根目录执行：

```bash
python3 examples/deepseek-v4-pro-32k-tuning/plot_tradeoff.py
```

它读取已经留存的 CSV，仅绘制 N96 的三个实测点，不插值、不添加误差区间。
