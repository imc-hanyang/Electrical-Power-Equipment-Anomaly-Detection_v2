#!/usr/bin/env bash
# run_all.sh — 전선 이상탐지 원-파일 실행 (가상환경 자동 격리)
#   이 파일 하나만 있으면 됩니다:  bash run_all.sh
#   → 저장소 clone → .venv 생성·활성화 → 의존성 설치
#     → 체크포인트/검증셋 다운로드 → 추론(성능표 출력)
#
# 전제조건: git · python3 (venv 포함) · 인터넷 · (권장) NVIDIA GPU
# 자기 데이터로 돌리려면, clone된 폴더 안 dataset/Anomaly, dataset/Normal 에
# 이미지를 넣고 다시 실행하면 그걸 사용합니다.
set -e

REPO="https://github.com/imc-hanyang/Electrical-Power-Equipment-Anomaly-Detection_v2.git"
DIR="Electrical-Power-Equipment-Anomaly-Detection_v2"

# ── 1) 저장소 ─────────────────────────────────────────────
if [ ! -d "$DIR" ]; then
  echo "[1/3] 저장소 clone 중..."
  git clone "$REPO"
else
  echo "[1/3] 저장소 이미 있음 → 최신화(git pull)"
  git -C "$DIR" pull --ff-only || true
fi
cd "$DIR"

# ── 2) 격리된 가상환경 (.venv) ────────────────────────────
echo "[2/3] 가상환경(.venv) 준비..."
if [ ! -d .venv ]; then
  python3 -m venv .venv || {
    echo "[ERROR] python3 venv 모듈 필요. 예: 'sudo apt install -y python3-venv' 후 재실행"
    exit 1
  }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip -q

# ── 3) 실행 (설치·다운로드·추론) ──────────────────────────
echo "[3/3] 실행: 설치 · 다운로드 · 추론"
PYTHON=python bash wire_inference.sh "$@"
