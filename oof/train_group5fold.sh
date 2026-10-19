#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# DATA_ROOT 미지정 시 기본값 data_train/ — 없으면 Drive(TRAIN_DATA_ID)에서 자동 다운로드
DATA_ROOT="${DATA_ROOT:-$ROOT/data_train}"
MANIFEST_DIR="${MANIFEST_DIR:-$ROOT/oof/reference_manifests}"
LOSS_MULTIPLIER="${LOSS_MULTIPLIER:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/loss${LOSS_MULTIPLIER}_group5fold}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

if [[ "$LOSS_MULTIPLIER" != "1" && "$LOSS_MULTIPLIER" != "2" ]]; then
  echo "LOSS_MULTIPLIER must be 1 or 2" >&2
  exit 2
fi

# ── 학습데이터 준비 (manifest 경로가 DATA_ROOT 기준으로 존재하는지 확인) ──
if [[ ! -d "$DATA_ROOT/wire/normal" || ! -d "$DATA_ROOT/wire/anomaly" ]]; then
  if [[ "$DATA_ROOT" == "$ROOT/data_train" ]]; then
    echo "[SETUP] 학습데이터 없음 → 다운로드 (train_data.tar · 1.3GB)"
    "$PYTHON" "$ROOT/download_checkpoints.py" --target train_data
  fi
fi
if [[ ! -d "$DATA_ROOT/wire/normal" || ! -d "$DATA_ROOT/wire/anomaly" ]]; then
  echo "[ERROR] $DATA_ROOT/wire/{normal,anomaly} 가 없습니다." >&2
  echo "        run_all.sh train 으로 실행하거나(TRAIN_DATA_ID 포함), DATA_ROOT 를 직접 지정하세요." >&2
  exit 1
fi
MISSING="$("$PYTHON" - "$MANIFEST_DIR/oof_master.csv" "$DATA_ROOT" <<'PY'
import csv, os, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
miss = [r["image_path"] for r in rows if not os.path.isfile(os.path.join(sys.argv[2], r["image_path"]))]
print(len(miss))
for m in miss[:5]:
    print("  " + m, file=sys.stderr)
PY
)"
if [[ "$MISSING" != "0" ]]; then
  echo "[ERROR] manifest 이미지 중 ${MISSING}장이 $DATA_ROOT 에 없습니다 (위 예시 참고)." >&2
  exit 1
fi
echo "[OK] 학습데이터 확인: $DATA_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$OUTPUT_DIR"

for FOLD in 0 1 2 3 4; do
  MANIFEST="$MANIFEST_DIR/fold_${FOLD}.csv"
  STAGE1="$OUTPUT_DIR/fold_${FOLD}/stage1"
  STAGE2="$OUTPUT_DIR/fold_${FOLD}/stage2"
  STAGE1_CKPT="$STAGE1/vit_base_patch16_clip_224.laion2b_best.pth"
  STAGE2_CKPT="$STAGE2/vit_base_patch16_clip_224.laion2b_best.pth"
  mkdir -p "$STAGE1" "$STAGE2"

  if [[ ! -f "$STAGE1_CKPT" ]]; then
    (
      cd "$ROOT/src"
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT="$((46100 + FOLD * 10))" \
        "$PYTHON" train.py \
          --model-mode vit --finetune-mode cla --image-size 224 \
          --csv-dir "$MANIFEST" --config-name config_clip_vit --data-root "$DATA_ROOT" \
          --gpu_id 0 --batch-size 16 --num-workers 4 \
          --init-lr 1e-4 --weight_decay 1e-4 --optimizer AdamW \
          --epochs 40 --warmup_epochs 2 --selection-metric ap \
          --normal-loss-multiplier 1.0 --anomaly-loss-multiplier "$LOSS_MULTIPLIER" \
          --backbone-name vit_base_patch16_clip_224.laion2b \
          --backbone-out-dim 768 --backbone-num-patch 196 --norm clip \
          --pooling-mode mean --output-dir "$STAGE1"
    )
  fi

  if [[ ! -f "$STAGE2_CKPT" ]]; then
    (
      cd "$ROOT/src"
      RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT="$((46101 + FOLD * 10))" \
        "$PYTHON" train.py \
          --model-mode vit --finetune-mode cla --image-size 224 \
          --csv-dir "$MANIFEST" --config-name config_clip_vit --data-root "$DATA_ROOT" \
          --gpu_id 0 --batch-size 16 --num-workers 4 \
          --init-lr 1e-4 --weight_decay 1e-4 --optimizer AdamW \
          --epochs 40 --warmup_epochs 2 --selection-metric ap \
          --normal-loss-multiplier 1.0 --anomaly-loss-multiplier "$LOSS_MULTIPLIER" \
          --backbone-name vit_base_patch16_clip_224.laion2b \
          --backbone-out-dim 768 --backbone-num-patch 196 --norm clip \
          --pooling-mode mean --finetune-ckpt "$STAGE1_CKPT" --output-dir "$STAGE2"
    )
  fi
done

echo "Loss x${LOSS_MULTIPLIER} Group 5-Fold training complete: $OUTPUT_DIR"
