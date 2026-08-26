#!/usr/bin/env bash
# wire_inference.sh — KEPCO OPGW 전선 이상탐지 (5-Fold CLAdapter 소프트보팅)
#
# 최소 실행 (기본값만):
#   bash wire_inference.sh
#     → ./dataset 추론 → ./predictions
#       dataset/Anomaly + dataset/Normal 이 있으면 Precision/Recall/F1/AUROC 자동 계산
#
# 옵션 override (선택):
#   bash wire_inference.sh --input-dir /path/to/images --output-dir /path/to/out
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" run_wire.py "$@"
