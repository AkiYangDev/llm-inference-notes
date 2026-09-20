# 用户任务

请根据下面给出的全部材料，写一篇 1200～1800 个中文字左右的完整教程《一个请求怎样分两轮完成 Prefill，再进入 Decode》。面向后端工程师，用自然段、必要的图或表解释状态变化。可以使用少量代码标识符，不要按函数逐个开章节。只分析这里的教学实现，不要联网，不把它改写成真实 SGLang 或 DeepSeek 源码，不需要运行代码。

## 材料范围

这是虚构的 Python 调度教学模型，快照 toy-r1。列表 items 始终代表同一请求，只有一条请求且各轮依次完成。processed 是已经作为模型输入处理的逻辑位置数量。输出代号不是汉字。没有设备实现、真实 KV Tensor 或网络返回代码。

```python
from dataclasses import dataclass, field

@dataclass
class Request:
    prompt: list[str]
    processed: int = 0
    output: list[str] = field(default_factory=list)
    done: bool = False

def plan(r, budget):
    if r.done:
        return None
    if r.processed < len(r.prompt):
        end = min(r.processed + budget, len(r.prompt))
        return ("prefill", r.prompt[r.processed:end], end)
    return ("decode", [r.output[-1]], r.processed + 1)

def finish(r, batch, sample):
    mode, items, end = batch
    r.processed = end
    if mode == "prefill" and end < len(r.prompt):
        return
    r.output.append(sample)
    r.done = len(r.output) >= 2

r = Request(["A", "B", "C", "D", "E"])
# 每轮：b = plan(r, budget=3)，计算 b 的输入后调用 finish。
# 测试驱动在第 1、2、3 次 finish 分别传入 "unused"、"y1"、"y2"。
# 这些值只是模拟采样结果，不是模型预测。
```

另一份设计草稿 toy-r2 提议让每个 Prefill chunk 都对外产生输出，但尚未改动上面 toy-r1 的代码。本文解释 toy-r1。
