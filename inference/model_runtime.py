#!/usr/bin/env python3
"""Shared model loading and image preprocessing for OOF inference."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from build_model import CLAdapter_CLIP_ViT  # noqa: E402
from config_clip_vit import get_config  # noqa: E402


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload.get("state_dict", payload)


def build_model(checkpoint: Path, device: torch.device) -> CLAdapter_CLIP_ViT:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = get_config()
    config.defrost()
    config.MODEL.num_classes = 2
    config.MODEL.m_mode = "vit"
    config.MODEL.f_mode = "cla"
    config.MODEL.img_size = 224
    config.MODEL.pooling_mode = "mean"
    config.MODEL.backbone.model_name = "vit_base_patch16_clip_224.laion2b"
    config.MODEL.backbone.out_dim = 768
    config.MODEL.backbone.num_patch = 196
    config.MODEL.backbone.pretrained = False
    config.MODEL.finetune = "checkpoint"
    config.freeze()
    model = CLAdapter_CLIP_ViT(config)
    model.load_state_dict(load_state(checkpoint), strict=True)
    return model.to(device).eval()


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path]):
        if not paths:
            raise ValueError("No input images were found")
        self.paths = paths
        self.transform = evaluation_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path)


def scan_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No supported images under {root}")
    return paths

