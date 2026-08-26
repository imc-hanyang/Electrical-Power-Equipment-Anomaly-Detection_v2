#!/usr/bin/env python3
"""Select one common threshold from the vertically pooled OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from inference.model_runtime import sha256


def wilson_bounds(successes: np.ndarray, total: int, z: float = 1.96):
    successes = np.asarray(successes, dtype=np.float64)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
    ) / denominator
    return np.maximum(0.0, center - half), np.minimum(1.0, center + half)


def candidates(labels: np.ndarray, scores: np.ndarray, prior: float) -> pd.DataFrame:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    cumulative_fp = np.cumsum(sorted_labels == 0)
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    tp = cumulative_tp[ends].astype(np.int64)
    fp = cumulative_fp[ends].astype(np.int64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    tpr = tp / positives
    fpr = fp / negatives
    precision = tpr * prior / (tpr * prior + fpr * (1.0 - prior) + 1e-15)
    f1 = 2 * precision * tpr / (precision + tpr + 1e-15)
    tpr_lower, _ = wilson_bounds(tp, positives)
    _, fpr_upper = wilson_bounds(fp, negatives)
    precision_lower = tpr_lower * prior / (
        tpr_lower * prior + fpr_upper * (1.0 - prior) + 1e-15
    )
    return pd.DataFrame(
        {
            "threshold": sorted_scores[ends],
            "tp": tp,
            "fp": fp,
            "recall": tpr,
            "fpr": fpr,
            "precision_at_target_prior": precision,
            "f1_at_target_prior": f1,
            "precision_lower_95": precision_lower,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-prior", type=float, default=0.5)
    parser.add_argument("--precision-floor", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(args.predictions_dir / f"fold_{fold}.csv") for fold in range(args.folds)]
    pooled = pd.concat(frames, ignore_index=True)
    required = {"image_path", "true_label", "prob_anomaly"}
    if not required.issubset(pooled.columns):
        raise RuntimeError(f"OOF prediction columns missing: {sorted(required - set(pooled.columns))}")
    if pooled["image_path"].duplicated().any():
        raise RuntimeError("OOF rows must form one unique cover, not an average")
    labels = pooled["true_label"].to_numpy(dtype=np.int64)
    scores = pooled["prob_anomaly"].to_numpy(dtype=np.float64)
    points = candidates(labels, scores, args.target_prior)
    feasible = points[points["precision_lower_95"] >= args.precision_floor].copy()
    if feasible.empty:
        raise RuntimeError("No threshold satisfies the requested Precision lower bound")
    selected = feasible.sort_values(
        ["recall", "f1_at_target_prior", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]

    rule = {
        "status": "FROZEN_FROM_OOF_BEFORE_TARGET",
        "method": "uniform mean of five anomaly probabilities",
        "threshold": float(selected["threshold"]),
        "threshold_policy": "maximize OOF recall subject to Precision 95% Wilson lower bound >= floor",
        "target_prior": args.target_prior,
        "precision_floor": args.precision_floor,
        "oof_rows": len(pooled),
        "oof_normal": int((labels == 0).sum()),
        "oof_anomaly": int((labels == 1).sum()),
        "oof_ap": float(average_precision_score(labels, scores)),
        "oof_auroc": float(roc_auc_score(labels, scores)),
        "oof_threshold_precision": float(selected["precision_at_target_prior"]),
        "oof_threshold_precision_lower_95": float(selected["precision_lower_95"]),
        "oof_threshold_recall": float(selected["recall"]),
        "checkpoint_sha256": {
            str(fold): sha256(args.checkpoint_dir / f"fold_{fold}.pth")
            for fold in range(args.folds)
        },
        "methodological_note": (
            "Each source-Train image contributes one prediction from the model that did not train on its outer fold. "
            "Those rows are concatenated, not averaged. The frozen threshold is then applied to the five-model mean "
            "on new images as a practical cross-fitted approximation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rule, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rule, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

