# SGLang × AISBench 最小服务化性能示例

配合[教程](../../docs/performance/aisbench-sglang-first-request-to-report.md)使用。

适用源码为 AISBench/benchmark `e00d1f758310c45580576af291a3126919776c89`，不是所有版本的统一参数表。SGLang 已在运行，支持 `/v1/models`、流式 `/v1/chat/completions` 和 usage。PD 场景连接统一入口。客户端需有对应 tokenizer；示例不启动服务、不清缓存、不修改安装目录。

## 安装客户端

在 Linux Bash 中执行；已具备匹配客户端时直接复用，避免升级活动环境。以下安装写入新的虚拟环境，Python 3.11 是示例解释器：

```bash
mkdir -p "$HOME/aisbench-sglang-lab"
cd "$HOME/aisbench-sglang-lab"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
git clone https://github.com/AISBench/benchmark.git aisbench
git -C aisbench checkout e00d1f758310c45580576af291a3126919776c89
python -m pip install -e ./aisbench
git clone https://github.com/AkiYangDev/llm-inference-notes.git notes
cd notes/examples/aisbench-sglang
```

仓库依赖包含 torch 等库，客户端依赖安装与模型服务器的运行环境分别管理。安装命令是读者操作步骤，本次写作没有重装完整 AISBench。

## 配置现有服务

以下地址仅用于本机示例；根据实际服务修改。`SERVED_MODEL` 必须改成 `/v1/models` 回包中的 id，`TOKENIZER_PATH` 改成客户端可读目录：

```bash
export HOST_IP=127.0.0.1
export HOST_PORT=30000
export SERVED_MODEL=deepseek-v4-flash
export TOKENIZER_PATH=/models/DeepSeek-V4-Flash
export API_KEY=""
export AIS_DATASET="$PWD/data.jsonl"

curl --fail --silent --show-error \
  -H "Authorization: Bearer ${API_KEY}" \
  "http://${HOST_IP}:${HOST_PORT}/v1/models"
```

`perf_config.py` 使用完整点分字符串注册路径；实际 registry 构建发生在 AISBench 中。这里使用 `VLLMCustomAPIChat(stream=True)`，类名不要求服务端必须是 vLLM。4 条数据只用于接通和检查报告，未宣称固定 token 输入长度、缓存比例或稳态。`answer` 留空，因为本例运行性能模式，不做问答准确率评测。

固定源码的发送循环为 `range(self.retry)`，配置 `retry=1` 表示总共尝试一次；这里不能按“额外重试次数”填 0。

## 先检查配置，再发送一条和四条

```bash
python validate_config.py
NUM_REQUESTS=1 CONCURRENCY=1 REQUEST_RATE=1 bash run.sh
NUM_REQUESTS=4 CONCURRENCY=2 REQUEST_RATE=1 bash run.sh
```

校验脚本只做 JSONL 结构、真实 MMEngine Config 读取、dump 和新进程 reload，不发送 HTTP，不加载 tokenizer，不证明实际 AISBench registry 或服务兼容。1 条请求用于检查这些尚未覆盖的路径。

每轮默认独立 `outputs/<外层时间戳>/`，AISBench 再生成其内层时间戳目录。看控制台末尾实际打印的结果目录；同一秒连续启动时请显式提供不同 `OUT`。shell 使用 `pipefail`，避免 `tee` 掩盖 AISBench 非零退出。

## 正式数据与并发

准备并核验自己的实体数据后，显式指定文件和请求数，例如：

```bash
export AIS_DATASET=/data/my-validated-640.jsonl
export OUTPUT_TOKENS=1024
NUM_REQUESTS=640 CONCURRENCY=90 REQUEST_RATE=6 bash run.sh
```

`/data/my-validated-640.jsonl` 是需要替换的文件占位符，应至少有 640 条已核验数据。请求数不会生成实体数据，输出上限不会保证实际 completion 长度。正式比较还应记录缓存、服务配置、物理卡数和采样参数。

## 复用结果计算稳定区间

先把下面两个路径改成本次控制台打印的实际工作根和内层时间戳。请保留整个结果目录中的 details JSONL、数据库、配置及日志，后处理会用到原始时序：

```bash
OUT_ROOT=/absolute/path/to/this-run-work-dir
RUN_STAMP=20261024_120000
cp -a "$OUT_ROOT/$RUN_STAMP/performances" \
      "$OUT_ROOT/$RUN_STAMP/performances-whole"
ais_bench "$PWD/perf_config.py" \
  --mode perf_viz --reuse "$RUN_STAMP" \
  --summarizer stable_stage --work-dir "$OUT_ROOT" --debug
```

上述 `perf_viz` 在固定源码中只包含 PerfViz；重算前保存 whole 文件，因为新汇总可能写回相同的性能结果路径。保持原来的环境变量和模型、数据 abbr。未找到稳定阶段时保留失败诊断与 whole，不把重新发送请求作为默认补救。

本次示例已经做配置序列化/新进程读取、源码符号与 CLI 参数核对；没有对当前服务执行上述 HTTP、AISBench 小测或正式测量。正文实测表来自此前封存记录。
