#!/usr/bin/env python3
"""Build leakage-free outer OOF and inner Validation manifests by source group."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {"image_path", "label", "group"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split(frame: pd.DataFrame, folds: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(frame, frame["label"], frame["group"]))


def counts(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "normal": int((frame["label"] == 0).sum()),
        "anomaly": int((frame["label"] == 1).sum()),
        "groups": int(frame["group"].nunique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--outer-seed", type=int, default=1721)
    parser.add_argument("--inner-seed-base", type=int, default=9100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source_csv)
    missing = REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    if "split" in source.columns:
        source = source[source["split"] == "train"].copy()
    source = source.reset_index(drop=True)
    source["label"] = source["label"].astype(int)
    if source["image_path"].duplicated().any():
        raise RuntimeError("Duplicate image_path in source Train")
    if set(source["label"].unique()) != {0, 1}:
        raise RuntimeError("Source Train must contain labels 0 and 1")

    # Preserve the exact column layout used for the supplied checkpoints.
    source["source_split"] = "train"
    source["outer_fold"] = -1
    for fold, (_, outer_indices) in enumerate(split(source, args.folds, args.outer_seed)):
        source.loc[outer_indices, "outer_fold"] = fold
    if source.groupby("group")["outer_fold"].nunique().max() != 1:
        raise RuntimeError("A source group spans more than one outer fold")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    master_path = args.output_dir / "oof_master.csv"
    source.to_csv(master_path, index=False)
    covered: list[str] = []
    report = {
        "status": "PASS",
        "source_csv": str(args.source_csv.resolve()),
        "source_sha256": sha256(args.source_csv),
        "folds": args.folds,
        "outer_seed": args.outer_seed,
        "inner_seed_base": args.inner_seed_base,
        "source_train": counts(source),
        "fold_details": [],
    }

    for fold in range(args.folds):
        outer = source[source["outer_fold"] == fold].copy()
        development = source[source["outer_fold"] != fold].copy().reset_index(drop=True)
        inner_train, inner_val = split(
            development, args.folds, args.inner_seed_base + fold
        )[0]
        inner = development.copy()
        inner["split"] = "train"
        inner.loc[inner_val, "split"] = "val"
        outer["split"] = "oof"
        manifest = pd.concat([inner, outer], ignore_index=True)
        manifest["evaluation_policy"] = manifest["split"].map(
            {
                "train": "fit",
                "val": "inner_checkpoint_selection",
                "oof": "outer_evaluation_only",
            }
        )

        role_groups = {
            role: set(manifest.loc[manifest["split"] == role, "group"])
            for role in ("train", "val", "oof")
        }
        overlap = {
            "train_val": len(role_groups["train"] & role_groups["val"]),
            "train_oof": len(role_groups["train"] & role_groups["oof"]),
            "val_oof": len(role_groups["val"] & role_groups["oof"]),
        }
        if any(overlap.values()):
            raise RuntimeError(f"Fold {fold} group leakage: {overlap}")
        manifest_path = args.output_dir / f"fold_{fold}.csv"
        manifest.to_csv(manifest_path, index=False)
        covered.extend(outer["image_path"].astype(str))
        report["fold_details"].append(
            {
                "fold": fold,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256(manifest_path),
                "roles": {
                    role: counts(manifest[manifest["split"] == role])
                    for role in ("train", "val", "oof")
                },
                "group_overlap": overlap,
            }
        )

    if len(covered) != len(source) or len(set(covered)) != len(source):
        raise RuntimeError("Outer folds are not an exact unique OOF cover")
    report["oof_cover"] = {"rows": len(covered), "exact_unique_cover": True}
    report["master_sha256"] = sha256(master_path)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
