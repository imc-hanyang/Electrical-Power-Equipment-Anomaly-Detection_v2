#!/usr/bin/env bash
# run_all.sh — 전선 이상탐지 원-파일 실행
#   이 파일 하나만 있으면 됩니다:  bash run_all.sh
#   → 저장소 clone → 의존성 설치 → 체크포인트/검증셋 다운로드 → 추론(성능표 출력)
#
# 자기 데이터로 돌리려면, clone된 폴더 안 dataset/Anomaly, dataset/Normal 에
# 이미지를 넣고 다시 실행하면 그걸 사용합니다.
set -e

REPO="https://github.com/imc-hanyang/Electrical-Power-Equipment-Anomaly-Detection_v2.git"
DIR="Electrical-Power-Equipment-Anomaly-Detection_v2"

if [ ! -d "$DIR" ]; then
  echo "[1/2] 저장소 clone 중..."
  git clone "$REPO"
else
  echo "[1/2] 저장소 이미 있음 → 최신화(git pull)"
  git -C "$DIR" pull --ff-only || true
fi

cd "$DIR"
echo "[2/2] 실행: 설치 · 다운로드 · 추론"
bash wire_inference.sh "$@"
