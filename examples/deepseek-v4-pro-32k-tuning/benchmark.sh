#!/usr/bin/env bash
set -euo pipefail

: "${BENCH_HOST:?Set BENCH_HOST to the running service host}"
: "${BENCH_MODEL:?Set BENCH_MODEL to a locally accessible model/tokenizer directory}"
BENCH_PORT="${BENCH_PORT:-6677}"
BENCH_SERVED_MODEL="${BENCH_SERVED_MODEL:-dsv4}"
BENCH_C="${1:-24}"
BENCH_N="${2:-96}"

if [[ ! "$BENCH_C" =~ ^[1-9][0-9]*$ || ! "$BENCH_N" =~ ^[1-9][0-9]*$ ]] || (( BENCH_N <= BENCH_C )); then
  echo "Use positive integers with NUM_PROMPTS > CONCURRENCY" >&2
  exit 2
fi
BENCH_CI_FLAG="${SGLANG_IS_IN_CI:-}"
case "${BENCH_CI_FLAG,,}" in
  1|true|yes|on)
    echo "SGLANG_IS_IN_CI may flush the cache after warmup. Use a client environment with that flag disabled." >&2
    exit 2
    ;;
esac

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
BENCH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_RESULT_DIR="${BENCH_OUT_DIR:-$PWD/bench_out}"
mkdir -p -- "$BENCH_RESULT_DIR"
BENCH_STEM="$BENCH_RESULT_DIR/v4pro_32k_1k_prefix31900_question868_c${BENCH_C}_n${BENCH_N}_warm8_seed42_$(date +%Y%m%d_%H%M%S)_$$"
BENCH_URL="http://${BENCH_HOST}:${BENCH_PORT}"
echo "结果：${BENCH_STEM}.jsonl"
echo "日志：${BENCH_STEM}.log"

python3 -m sglang.benchmark.serving --help > "${BENCH_STEM}.help.txt"
python3 - <<'PY' > "${BENCH_STEM}.client.json"
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
import sglang

info = {"python": platform.python_version(), "sglang_import_path": sglang.__file__}
try:
    info["sglang_version"] = importlib.metadata.version("sglang")
except importlib.metadata.PackageNotFoundError:
    info["sglang_version"] = None
try:
    info["client_source_commit"] = subprocess.check_output(
        ["git", "-C", str(Path(sglang.__file__).resolve().parent), "rev-parse", "HEAD"],
        text=True, stderr=subprocess.DEVNULL,
    ).strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    info["client_source_commit"] = None
print(json.dumps(info, ensure_ascii=False, indent=2))
PY

curl --noproxy '*' -fsS --connect-timeout 10 --max-time 30 \
  "${BENCH_URL}/v1/models" > "${BENCH_STEM}.models.json"
curl --noproxy '*' -fsS --connect-timeout 10 --max-time 90 \
  -X POST "${BENCH_URL}/flush_cache?timeout=60" | tee "${BENCH_STEM}.flush.txt"

python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host "$BENCH_HOST" --port "$BENCH_PORT" \
  --model "$BENCH_MODEL" \
  --served-model-name "$BENCH_SERVED_MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group "$BENCH_N" \
  --gsp-system-prompt-len 31900 \
  --gsp-question-len 868 \
  --gsp-output-len 1024 \
  --gsp-range-ratio 1.0 \
  --gsp-num-turns 1 \
  --num-prompts "$BENCH_N" \
  --max-concurrency "$BENCH_C" \
  --request-rate inf \
  --seed 42 \
  --warmup-requests 8 \
  --cache-report \
  --output-details \
  --output-file "${BENCH_STEM}.jsonl" \
  2>&1 | tee "${BENCH_STEM}.log"

python3 "$BENCH_SCRIPT_DIR/check_result.py" "${BENCH_STEM}.jsonl" "$BENCH_N" 1024 \
  | tee "${BENCH_STEM}.check.txt"
