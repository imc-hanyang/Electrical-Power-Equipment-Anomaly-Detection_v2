#!/usr/bin/env python3
"""Predict each outer fold using only the model that did not train on that fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.model_runtime import ImagePathDataset, build_model, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def resolve_paths(frame: pd.DataFrame, data_root: Path) -> list[Path]:
    paths = []
    for value in frame["image_path"].astype(str):
        path = Path(value)
        paths.append((path if path.is_absolute() else data_root / path).resolve())
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images, first 10: {missing[:10]}")
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    report = {"status": "PASS", "folds": args.folds, "fold_details": []}
    all_paths: list[str] = []

    for fold in range(args.folds):
        manifest_path = args.manifest_dir / f"fold_{fold}.csv"
        checkpoint = args.checkpoint_dir / f"fold_{fold}.pth"
        manifest = pd.read_csv(manifest_path)
        outer = manifest[manifest["split"] == "oof"].copy().reset_index(drop=True)
        fit_groups = set(manifest.loc[manifest["split"].isin(["train", "val"]), "group"].astype(str))
        if fit_groups & set(outer["group"].astype(str)):
            raise RuntimeError(f"Fold {fold}: OOF group overlaps Train/Validation")

        image_paths = resolve_paths(outer, args.data_root)
        loader = DataLoader(
            ImagePathDataset(image_paths),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        model = build_model(checkpoint, device)
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for images, _ in tqdm(loader, desc=f"OOF Fold {fold}"):
                logits = model(images.to(device, non_blocking=True))
                batches.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        scores = np.concatenate(batches).astype(np.float64)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(scores) != len(outer):
            raise RuntimeError(f"Fold {fold}: prediction count mismatch")

        output = pd.DataFrame(
            {
                "image_path": outer["image_path"].astype(str),
                "group": outer["group"].astype(str),
                "true_label": outer["label"].astype(int),
                "outer_fold": fold,
                "prob_anomaly": scores,
            }
        )
        output_path = args.output_dir / f"fold_{fold}.csv"
        output.to_csv(output_path, index=False)
        all_paths.extend(output["image_path"].tolist())
        report["fold_details"].append(
            {
                "fold": fold,
                "rows": len(output),
                "normal": int((output["true_label"] == 0).sum()),
                "anomaly": int((output["true_label"] == 1).sum()),
                "checkpoint_sha256": sha256(checkpoint),
                "prediction_csv": str(output_path.resolve()),
                "prediction_sha256": sha256(output_path),
            }
        )

    if len(all_paths) != len(set(all_paths)):
        raise RuntimeError("Pooled OOF predictions contain duplicate image paths")
    report["oof_cover"] = {"rows": len(all_paths), "exact_unique_cover": True}
    (args.output_dir / "prediction_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

