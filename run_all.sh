#!/usr/bin/env bash
# run_all.sh — 전선 이상탐지 원-파일 실행
#   이 파일 하나만 있으면 됩니다:  bash run_all.sh
#   → 저장소 clone → (활성 python에) 의존성 설치(없을 때만)
#     → 체크포인트/검증셋 다운로드 → 추론(성능표 출력)
#
#   학습:  bash run_all.sh train     # 학습데이터 자동 다운로드 → Group 5-Fold 학습
#          (옵션 환경변수: LOSS_MULTIPLIER=1|2, GPU=0, OUTPUT_DIR=...)
#
# 전제조건: git · python3(+pip) · 인터넷 · (권장) NVIDIA GPU
#   ※ 격리가 필요하면 먼저 'conda activate <env>' 또는 venv 활성화 후 실행하세요.
# 자기 데이터로 돌리려면, clone된 폴더 안 dataset/Anomaly, dataset/Normal 에
# 이미지를 넣고 다시 실행하면 그걸 사용합니다.
set -e

REPO="https://github.com/imc-hanyang/Electrical-Power-Equipment-Anomaly-Detection_v2.git"
DIR="Electrical-Power-Equipment-Anomaly-Detection_v2"

# ── 1) 저장소 ─────────────────────────────────────────────
if [ ! -d "$DIR" ]; then
  echo "[1/2] 저장소 clone 중..."
  git clone "$REPO"
else
  echo "[1/2] 저장소 이미 있음 → 최신화(git pull)"
  git -C "$DIR" pull --ff-only || true
fi
cd "$DIR"

# ── Drive 링크 ID (비공개 · 이 파일에만 존재 · git 미공개) ──
export LOSS2_ID="15mRS3nH_mZrNPBKbKMXEV50X0m2vnNEB"
export VAL_INFER_ID="1MKNm4oXV9RCeHY9KQBMJ6ffukixxobqw"
export TRAIN_DATA_ID="1P6g0tyvCmUQ6TJTaE_ueTJMJTIFqQcnf"

# ── 2) 실행 ───────────────────────────────────────────────
if [ "${1:-}" = "train" ]; then
  shift
  echo "[2/2] 학습: 의존성 설치(없을 때만) · 학습데이터 다운로드 · Group 5-Fold 학습"
  PYTHON="${PYTHON:-python}"
  "$PYTHON" -c "import torch, timm" 2>/dev/null || "$PYTHON" -m pip install -r requirements.txt
  "$PYTHON" -c "import gdown" 2>/dev/null || "$PYTHON" -m pip install gdown
  PYTHON="$PYTHON" bash oof/train_group5fold.sh "$@"
else
  echo "[2/2] 실행: 설치(없을 때만) · 다운로드 · 추론"
  bash wire_inference.sh "$@"
fi
