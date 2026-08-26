#!/usr/bin/env python3
"""
KEPCO OPGW 전선 이상탐지 — 5-Fold CLAdapter 소프트보팅 추론 (zero-arg)

사용:
    python run_wire.py                     # dataset/ 추론 → predictions/  (기본값만으로 동작)
    python run_wire.py --input-dir <path>  # 필요 시 옵션 override

입력 구조 (dataset/):
    dataset/
    ├── Anomaly/   # 이상 이미지  ← Anomaly/ + Normal/ 가 있으면 성능(P/R/F1/AUROC) 자동 계산
    └── Normal/    # 정상 이미지
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference.model_runtime import ImagePathDataset, build_model, scan_images, sha256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="전선 이상탐지 5-Fold 소프트보팅 추론 (기본값만으로 실행 가능)")
    p.add_argument("--input-dir", type=Path, default=ROOT / "dataset",
                   help="테스트 이미지 폴더 (기본: ./dataset)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "predictions",
                   help="결과 저장 폴더 (기본: ./predictions)")
    p.add_argument("--loss", type=int, choices=(1, 2), default=2,
                   help="체크포인트 세트 (기본: 2 = 권장 모델)")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="미지정 시 operating_rule.json 의 고정 임계값 사용")
    p.add_argument("--device", default=None, help="미지정 시 GPU 있으면 cuda, 없으면 cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def resolve_device(arg: str | None) -> str:
    if arg:
        return arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def gt_from_path(path_str: str):
    s = path_str.replace("\\", "/").lower()
    if "/anomaly/" in s or s.endswith("/anomaly"):
        return 1
    if "/normal/" in s or s.endswith("/normal"):
        return 0
    return None


def main() -> None:
    a = parse_args()
    ckpt_dir = a.checkpoint_dir or (ROOT / f"checkpoints/oof_loss{a.loss}_k5")
    rule_path = ckpt_dir / "operating_rule.json"
    if not rule_path.is_file():
        sys.exit(f"[ERROR] operating_rule.json 없음: {rule_path}\n"
                 f"  → python download_checkpoints.py 로 체크포인트를 먼저 받으세요.")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    threshold = float(a.threshold if a.threshold is not None else rule["threshold"])
    folds = int(rule.get("folds", 5))

    if not a.input_dir.is_dir():
        sys.exit(f"[ERROR] 입력 폴더 없음: {a.input_dir}\n"
                 f"  → dataset/Anomaly, dataset/Normal 에 이미지를 넣으세요.")

    # 체크포인트 존재 + 무결성
    ckpts = [ckpt_dir / f"fold_{k}.pth" for k in range(folds)]
    missing = [str(c) for c in ckpts if not c.is_file()]
    if missing:
        sys.exit(f"[ERROR] 체크포인트 없음: {missing}\n"
                 f"  → python download_checkpoints.py")
    for k, c in enumerate(ckpts):
        exp = rule.get("checkpoint_sha256", {}).get(str(k))
        if exp and sha256(c) != exp:
            sys.exit(f"[ERROR] 체크포인트 SHA256 불일치: {c}")

    paths = scan_images(a.input_dir)
    device = torch.device(resolve_device(a.device))

    print("=" * 66)
    print("  KEPCO OPGW 전선 이상탐지 — 5-Fold 소프트보팅")
    print(f"  input     : {a.input_dir}  ({len(paths)}장)")
    print(f"  device    : {device}   threshold : {threshold:.6f}  (loss×{a.loss})")
    print("=" * 66)

    loader = DataLoader(ImagePathDataset(paths), batch_size=a.batch_size,
                        num_workers=a.num_workers, shuffle=False, pin_memory=True)

    cols = []
    for k, c in enumerate(ckpts):
        model = build_model(c, device)
        probs = []
        with torch.inference_mode():
            for imgs, _ in loader:
                logits = model(imgs.to(device, non_blocking=True))
                probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        cols.append(np.concatenate(probs).astype(np.float64))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  fold {k} 추론 완료")

    score_mat = np.column_stack(cols)
    mean_scores = score_mat.mean(axis=1)
    preds = (mean_scores >= threshold).astype(int)

    # 저장
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with open(a.output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path"] + [f"fold_{k}_score" for k in range(folds)]
                   + ["soft_voting_score", "threshold", "prediction", "prediction_name"])
        for i, p in enumerate(paths):
            w.writerow([str(p)] + [f"{score_mat[i, k]:.6f}" for k in range(folds)]
                       + [f"{mean_scores[i]:.6f}", threshold, int(preds[i]),
                          "Anomaly" if preds[i] == 1 else "Normal"])

    summary = {
        "model": "ViT-B/16 + CLAdapter mean-pool · 5-Fold soft-voting",
        "loss_multiplier": a.loss,
        "threshold": threshold,
        "images": len(paths),
        "anomaly_predictions": int((preds == 1).sum()),
        "normal_predictions": int((preds == 0).sum()),
    }

    # GT (Anomaly/Normal 폴더) 있으면 성능 자동 계산
    gts = [gt_from_path(str(p)) for p in paths]
    has_gt = len(gts) > 0 and all(g is not None for g in gts)
    if has_gt:
        y = np.array(gts, dtype=int)
        TP = int(((preds == 1) & (y == 1)).sum())
        FP = int(((preds == 1) & (y == 0)).sum())
        FN = int(((preds == 0) & (y == 1)).sum())
        TN = int(((preds == 0) & (y == 0)).sum())
        P = TP / (TP + FP) * 100 if TP + FP else 0.0
        R = TP / (TP + FN) * 100 if TP + FN else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        try:
            from sklearn.metrics import roc_auc_score
            AUROC = roc_auc_score(y, mean_scores) * 100
        except Exception:
            AUROC = float("nan")
        summary.update({"TP": TP, "FN": FN, "FP": FP, "TN": TN,
                        "precision": round(P, 2), "recall": round(R, 2),
                        "f1": round(F1, 2), "auroc": round(AUROC, 2)})
        print("\n" + "=" * 66)
        print("  전선 이상탐지 성능 (5-Fold 소프트보팅)")
        print("-" * 66)
        print(f"  이미지 {len(paths)}장   정상 {TN + FP} · 이상 {TP + FN}")
        print(f"  TP={TP}   FN={FN}   FP={FP}   TN={TN}")
        print(f"  Precision={P:.2f}%   Recall={R:.2f}%   F1={F1:.2f}%   AUROC={AUROC:.2f}%")
        print("=" * 66)
    else:
        print(f"\n  추론 완료: 이상 {int((preds == 1).sum())} / 정상 {int((preds == 0).sum())}")
        print("  (GT 없음 — dataset/Anomaly + dataset/Normal 구조면 성능 자동 계산)")

    (a.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  결과 저장: {a.output_dir}\\predictions.csv , summary.json\n")


if __name__ == "__main__":
    main()
