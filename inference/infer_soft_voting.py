#!/usr/bin/env python3
"""Run five CLAdapter models and apply the frozen OOF operating threshold."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.model_runtime import (
    REPOSITORY_ROOT,
    ImagePathDataset,
    build_model,
    scan_images,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group 5-Fold CLAdapter soft-voting inference"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--loss",
        type=int,
        choices=(1, 2),
        default=2,
        help="Checkpoint set to use. Default: 2",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--rule", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output-dir/Normal and output-dir/Anomaly",
    )
    return parser.parse_args()


def load_rule(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_paths(directory: Path, folds: int = 5) -> list[Path]:
    paths = [directory / f"fold_{fold}.pth" for fold in range(folds)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    return paths


def predict_one_model(
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
    fold: int,
) -> np.ndarray:
    model = build_model(checkpoint, device)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc=f"Fold {fold}", leave=False):
            logits = model(images.to(device, non_blocking=True))
            batches.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(batches).astype(np.float64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def unique_destination(directory: Path, source: Path) -> Path:
    candidate = directory / source.name
    if not candidate.exists():
        return candidate
    return directory / f"{source.stem}_{sha256(source)[:8]}{source.suffix.lower()}"


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir or (
        REPOSITORY_ROOT / f"checkpoints/oof_loss{args.loss}_k5"
    )
    rule_path = args.rule or checkpoint_dir / "operating_rule.json"
    rule = load_rule(rule_path)
    threshold = float(args.threshold if args.threshold is not None else rule["threshold"])
    paths = scan_images(args.input_dir)
    dataset = ImagePathDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    device = torch.device(args.device)
    checkpoints = checkpoint_paths(checkpoint_dir)

    expected_hashes = rule.get("checkpoint_sha256", {})
    for fold, checkpoint in enumerate(checkpoints):
        expected = expected_hashes.get(str(fold))
        if expected and sha256(checkpoint) != expected:
            raise RuntimeError(f"Checkpoint SHA256 mismatch: {checkpoint}")

    score_matrix = np.column_stack(
        [predict_one_model(checkpoint, loader, device, fold) for fold, checkpoint in enumerate(checkpoints)]
    )
    if score_matrix.shape != (len(paths), 5):
        raise RuntimeError(f"Unexpected score shape: {score_matrix.shape}")
    mean_scores = score_matrix.mean(axis=1)
    decisions = (mean_scores >= threshold).astype(np.int64)

    output = pd.DataFrame({"image_path": [str(path) for path in paths]})
    for fold in range(5):
        output[f"fold_{fold}_anomaly_score"] = score_matrix[:, fold]
    output["soft_voting_anomaly_score"] = mean_scores
    output["threshold"] = threshold
    output["prediction"] = decisions
    output["prediction_name"] = np.where(decisions == 1, "Anomaly", "Normal")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    output.to_csv(prediction_path, index=False)

    if args.copy_images:
        for class_name in ("Normal", "Anomaly"):
            (args.output_dir / class_name).mkdir(parents=True, exist_ok=True)
        for path, class_name in zip(paths, output["prediction_name"]):
            destination = unique_destination(args.output_dir / class_name, path)
            shutil.copy2(path, destination)

    summary = {
        "status": "PASS",
        "input_dir": str(args.input_dir.resolve()),
        "rows": len(output),
        "normal_predictions": int(np.sum(decisions == 0)),
        "anomaly_predictions": int(np.sum(decisions == 1)),
        "loss_multiplier": args.loss,
        "threshold": threshold,
        "threshold_source": str(rule_path.resolve()) if args.threshold is None else "CLI override",
        "ensemble": "uniform mean of five anomaly probabilities",
        "prediction_csv": str(prediction_path.resolve()),
        "checkpoint_sha256": {str(fold): sha256(path) for fold, path in enumerate(checkpoints)},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
