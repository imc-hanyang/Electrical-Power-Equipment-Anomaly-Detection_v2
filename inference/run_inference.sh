#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 INPUT_IMAGE_DIR OUTPUT_DIR [DEVICE] [LOSS=1|2]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="$1"
OUTPUT_DIR="$2"
DEVICE="${3:-cuda:0}"
LOSS="${4:-2}"
PYTHON="${PYTHON:-python}"

if [[ "$LOSS" != "1" && "$LOSS" != "2" ]]; then
  echo "LOSS must be 1 or 2" >&2
  exit 2
fi

cd "$ROOT"
"$PYTHON" -m inference.infer_soft_voting \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --loss "$LOSS" \
  --copy-images
