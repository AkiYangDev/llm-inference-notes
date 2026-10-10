"""Check a serving benchmark result without substituting missing fields."""
import json
import math
import sys
from collections import Counter
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: check_result.py RESULT.jsonl NUM_PROMPTS OUTPUT_LEN")
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("结果文件为空")
    result = rows[-1]
    expected_requests, expected_output_len = map(int, sys.argv[2:])
    if expected_requests < 1 or expected_output_len < 1:
        raise SystemExit("请求数和输出长度必须大于 0")

    failures = []
    for key in ("input_lens", "output_lens", "errors"):
        if not isinstance(result.get(key), list) or len(result[key]) != expected_requests:
            failures.append(f"{key} 必须包含 {expected_requests} 条记录")
    lengths = result.get("output_lens")
    if isinstance(lengths, list):
        if not all(type(n) is int for n in lengths):
            failures.append("output_lens 必须是整数列表")
        else:
            print("输出长度分布：", dict(Counter(lengths)))
            print("输出总量：", sum(lengths))
            if any(n != expected_output_len for n in lengths):
                failures.append(f"存在输出长度不为 {expected_output_len} 的请求")
    errors = result.get("errors")
    if isinstance(errors, list) and any(errors):
        failures.append(f"非空错误数：{sum(bool(e) for e in errors)}")
    if result.get("completed") != expected_requests:
        failures.append("成功请求数与预期不一致")
    if result.get("total_output_tokens") != expected_requests * expected_output_len:
        failures.append("总生成 Token 数与预期不一致")
    if failures:
        raise SystemExit("输出完整性检查失败：\n" + "\n".join(failures))

    inputs = result["input_lens"]
    if any(type(n) is not int or n < 1 for n in inputs):
        raise SystemExit("input_lens 包含无效输入长度")
    print(f"平均输入 Token：{sum(inputs) / expected_requests:.2f}")
    for key in ("duration", "mean_ttft_ms", "mean_tpot_ms", "mean_e2e_latency_ms", "output_throughput"):
        value = result.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"缺少或无效性能字段：{key}")
        print(f"{key}: {value:.2f}")
    cached = result.get("cached_tokens")
    if not isinstance(cached, list) or len(cached) != expected_requests:
        raise SystemExit("缺少逐请求 cached_tokens，请确认启用了 --cache-report 和 --output-details")
    if any(type(c) is not int or c < 0 or c > n for c, n in zip(cached, inputs)):
        raise SystemExit("cached_tokens 包含无效缓存计数")
    print(f"按 Token 汇总的缓存命中率：{sum(cached) / sum(inputs) * 100:.2f}%")
    print("输出完整性检查通过")


if __name__ == "__main__":
    main()
