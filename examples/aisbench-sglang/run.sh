#!/usr/bin/env bash
# Run against an already-started OpenAI-compatible SGLang service.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${TOKENIZER_PATH:?Export the local tokenizer directory first}"
: "${SERVED_MODEL:?Export the model id returned by /v1/models first}"
export AIS_DATASET="${AIS_DATASET:-$here/data.jsonl}"
OUT="${OUT:-$here/outputs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
ais_bench "$here/perf_config.py" \
  --mode perf --summarizer default_perf \
  --num-prompts "${NUM_REQUESTS:-1}" --num-warmups 0 \
  --work-dir "$OUT" --debug 2>&1 | tee "$OUT/console.log"
