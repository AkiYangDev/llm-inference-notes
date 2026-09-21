# Validation Surface

“有测试”不等于“这个结论被完整验证”。Validation Surface 用矩阵描述：测试覆盖了哪些路径、平台和 workload，又留下哪些空白。

## 基本矩阵

| 维度 | 可能取值 |
| --- | --- |
| Test level | unit / component / integration / e2e |
| Semantics | accuracy / correctness / lifecycle / failure path |
| Platform | CPU / CUDA / ROCm / Ascend / XPU |
| Workload | prefill / decode / spec / PD / multimodal / router |
| Shape | small / boundary / long-context / large-batch / zero-size |
| Concurrency | single request / batched / high concurrency / multi-rank |
| Performance | mechanism / microbench / workload / serving |

## 状态词

- **covered**：有对应测试且证据表明实际执行。
- **reported-pass**：PR/CI 报告通过，但你没有独立复现。
- **blocked**：计划测试存在，但环境或依赖阻断。
- **not-run**：明确没有运行。
- **unknown**：材料无法判断，不能当成 covered。

## Failure-path 优先

修复异步、capacity、fallback、zero-work、OOM、timeout 的 PR，不能只看正常请求测试。优先检查导致原 Bug 的触发条件是否被回归测试直接覆盖。

## CI Gate 陷阱

CI workflow green 不代表目标测试实际执行。检查是否因 path filter、label、platform gate、skip condition、cache dependency 等原因跳过。

## 性能验证的四层

1. **Mechanism**：源码证明少一次 launch/copy/sync、缩小临时 Tensor、减少复杂度；这是机制事实，不是 timing。
2. **Microbenchmark**：kernel / helper / isolated loop timing，只证明局部。
3. **Workload**：model step、prefill/decode iteration、具体模型请求；开始包含上下游摊销。
4. **Serving**：TTFT、TPOT、throughput、request latency、并发；最接近用户体验。

不得从 Level 1 或 2 直接写成 Level 4 收益。不同层的 benchmark 可以同时成立而倍数完全不同。
