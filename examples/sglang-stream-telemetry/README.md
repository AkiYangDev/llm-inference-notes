# SGLang流式统计：保留metadata，沿用AISBench解析与计时

配合[文章](../../docs/performance/sglang-stream-acceptance-cache-timing.md)使用。本目录提供教学版有界observer和离线回放，不是新的SSE客户端。

适用AISBench源码提交为`e00d1f758310c45580576af291a3126919776c89`，两份API源码的SHA256写在`replay.py`中。SGLang侧需已确认支持本次字段与语义；不能只靠客户端开关推定服务会返回。

## 文件

- `sglang_telemetry.py`：调用父类解析后保留metadata，不重写transport或clock。
- `telemetry.py`：保留白名单字段、最后usage快照，限制单次投机metadata为8KiB。多条final metadata、n>1列表和异常数值分别记录；不更改模型成功状态。
- `replay.py`：从固定源码提取原`iter_lines`、`stream_infer`和父类解析方法，在假响应/确定性时钟下比较文本、reasoning、token数与时间点。

## 离线回放

在已检出固定AISBench源码的环境中执行，路径换成自己的仓库。只需要Python标准库，不要求安装完整AISBench运行依赖：

```bash
python replay.py --aisbench-root /absolute/path/to/aisbench
```

源码SHA必须匹配，Git检出应保留原LF换行。该脚本没有HTTP、模型、tokenizer、NPU或事件循环；FakeResponse的async操作在CPU直接完成。适配类的registry装饰器在fixture中移除，因此这不是实际MODELS构建或writer测试。人工时刻、输入与输出计数只是教学数据，不是测量结果。

## 接到前文的独立配置

先使用[入门示例](../aisbench-sglang/README.md)确认普通流式请求可用，再做这一项变化。以下是在Linux Bash中的配置步骤，本次写作未对当前模型服务执行：

```bash
cd /absolute/path/to/notes/examples/sglang-stream-telemetry
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cp ../aisbench-sglang/perf_config.py perf_telemetry.py
cat >> perf_telemetry.py <<'PY'

custom_imports = dict(imports=['sglang_telemetry'], allow_failed_imports=False)
models[0]['type'] = 'sglang_telemetry.SGLangTelemetryChat'
models[0]['stream'] = True
models[0]['generation_kwargs'].update(return_spec_tokens_details=True, n=1)
PY
```

沿用入门示例的`HOST_IP`、`HOST_PORT`、`SERVED_MODEL`、`TOKENIZER_PATH`和`AIS_DATASET`，确保数据文件的绝对路径仍正确。把同一`PYTHONPATH`传给实际worker和后处理进程。然后仅发送一条诊断，输出到新的目录：

```bash
OUT="$PWD/outputs/telemetry-$(date +%Y%m%d_%H%M%S)"
ais_bench "$PWD/perf_telemetry.py" --mode perf \
  --summarizer default_perf --num-prompts 1 --num-warmups 0 \
  --work-dir "$OUT" --debug
```

父类在stream模式下请求`stream_options.include_usage=True`。该固定版本把`generation_kwargs`合并到最终JSON根部；这里没有把开关再包进一个`extra_body`字段。SGLang仍需在实际PD链路返回`sglext.spec_tokens_details`，缓存报告也需由服务端支持和启用。

检查最终details里的`extra_details_data`：`spec_capture_status`应为`RETAINED_UNVALIDATED`，`metadata_events`应为1；还需检查原始整数计数和直方图。`prefix_cache_usage.status`区分实际cached token、未观测与异常数值。解析成功不等于字段已被服务发出。

## 数值口径

`summarize_counts`默认只在存在有效verify计数和completion快照时计算output AL。严格接受率默认`NOT_OBSERVED`；`proposal_denominator_verified=True`是来源前提，仅在已证明返回的proposed对应实际候选分母时使用，不是看到数字就打开的功能开关。

metadata-only SSE继续留在原时间线。相同事件序列的回放可以验证observer没有增加/删除计时调用，但不能证明运行开销为零，也不能把metadata ON成绩与旧OFF成绩当成纯数据对照。样例未处理n>1列表、连续多份spec累计快照或跨请求复用同一个Output。

历史实际writer链和完成后的计数摘录见文章附件；本次新增验证只包含上述离线回放。没有新HTTP、服务重启或性能实验。
