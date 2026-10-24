# AISBench 指标离线算例

对应[《AISBench 的 whole、native、TPOT 和 ITL，到底在量什么？》](../../docs/performance/aisbench-whole-native-tpot-itl.md)。

在仓库根目录运行，Python 3.10+，只需标准库：

~~~bash
python examples/aisbench-metrics/explain_metrics.py
~~~

三个小例子分别演示：

- 输出 13 token、两个 60 ms 事件间隔，TPOT 为 10 ms。
- ITL 按事件等权与先按请求平均得到不同结果。
- C=2 时，原生选择区间 [3.1, 7.1] 纳入 C/D/E 的完整输出，E 延伸到 12 s；吞吐分母为 4 s，客户端在途比值以 C 封顶。

这些时间点全部是教学构造。脚本不访问网络、不读取真实请求、不调用模型，不是 AISBench 的替代计算器，也没有包含真实压测结果。真实统计使用[原始数值的脱敏投影](../../assets/aisbench-whole-native-tpot-itl/evidence.json)。

原生行为对应 AISBench [e00d1f7 的 StablePerfMetricCalculator](https://github.com/AISBench/benchmark/blob/e00d1f758310c45580576af291a3126919776c89/ais_bench/benchmark/calculators/stable_perf_metric_calculator.py)。教学实现保留该版本按 time 稳定排序、第二次 start 到达 C、闭区间选请求及 Concurrency 封顶规则；没有覆盖错误请求、数据库、单位转换和全部汇总分支。

