#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing paths in the manifests}"
MANIFEST_DIR="${MANIFEST_DIR:-$ROOT/oof/reference_manifests}"
LOSS_MULTIPLIER="${LOSS_MULTIPLIER:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/loss${LOSS_MULTIPLIER}_group5fold}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

if [[ "$LOSS_MULTIPLIER" != "1" && "$LOSS_MULTIPLIER" != "2" ]]; then
  echo "LOSS_MULTIPLIER must be 1 or 2" >&2
  exit 2
fi

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
