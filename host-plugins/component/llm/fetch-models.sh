#!/usr/bin/env bash
# Download Qwen3-0.6B (GGUF, Q8_0) and its tokenizer into models/qwen3-0.6b/,
# the directory example/.wash/config.yaml mounts into the plugin.
#
# Override MODEL_URL / TOKENIZER_URL to fetch a different Qwen3 GGUF; keep the
# file names in the binding config (`model-path`, `tokenizer-path`) in step.
set -euo pipefail

DIR="${1:-models/qwen3-0.6b}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf}"
TOKENIZER_URL="${TOKENIZER_URL:-https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json}"

mkdir -p "$DIR"
fetch() {
  local url="$1" out="$2"
  if [ -s "$out" ]; then
    echo "already have $out"
    return
  fi
  echo "fetching $url"
  curl --fail --location --progress-bar --output "$out.part" "$url"
  mv "$out.part" "$out"
}
fetch "$MODEL_URL" "$DIR/$(basename "${MODEL_URL%%\?*}")"
fetch "$TOKENIZER_URL" "$DIR/tokenizer.json"
echo "models ready in $DIR"
