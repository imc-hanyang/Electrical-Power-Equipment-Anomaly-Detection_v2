#!/usr/bin/env bash
# wire_inference.sh — KEPCO OPGW 전선 이상탐지 (5-Fold CLAdapter 소프트보팅)
#
# 한 명령 실행 (clone 후 이것만):
#   bash wire_inference.sh
#     ① requirements 설치(없을 때만) → ② 체크포인트 다운로드(없을 때만)
#     → ③ 검증셋(val_infer) 다운로드(입력 없을 때만) → ④ 추론
#
# 자기 데이터로 돌리려면 dataset/Anomaly, dataset/Normal 에 이미지를 넣으면
# 자동으로 그걸 사용합니다.  옵션:
#   bash wire_inference.sh --input-dir /path/to/images --output-dir /path/to/out
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"

# ── ① 의존성 (없을 때만 설치) ─────────────────────────────
if ! "$PYTHON" -c "import timm, torch, open_clip, sklearn, gdown" >/dev/null 2>&1; then
  echo "[SETUP] requirements 설치 중... (torch 등 대용량 — 수 분 소요, 진행바 표시)"
  "$PYTHON" -m pip install -r requirements.txt
  "$PYTHON" -m pip install gdown
fi

# ── ② 체크포인트 (없을 때만 다운로드) ─────────────────────
if [ ! -f checkpoints/oof_loss2_k5/fold_0.pth ]; then
  echo "[SETUP] 체크포인트 다운로드 중..."
  "$PYTHON" download_checkpoints.py --target loss2
fi

# ── ③ 입력셋 (dataset/ 도 val_infer/ 도 없으면 검증셋 다운로드) ──
if [ ! -d dataset ] && [ ! -d val_infer ]; then
  echo "[SETUP] 검증셋(val_infer · 230) 다운로드 중..."
  "$PYTHON" download_checkpoints.py --target val_infer
fi

# ── ④ 추론 ────────────────────────────────────────────────
exec "$PYTHON" run_wire.py "$@"
