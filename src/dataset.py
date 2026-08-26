import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFilter
import os
import pandas as pd
import numpy as np
import random
import cv2
from torchvision import transforms
from torchvision.transforms import functional as TF


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CLAdapterDataset(Dataset):
    def __init__(self, *args):
        # Original public train.py calls:
        # CLAdapterDataset(is_malignant, df, val_fold, test_fold, mode, img_size, root)
        if len(args) == 8:
            _, df, val_fold, test_fold, mode, img_size, root, norm_name = args
        elif len(args) == 7:
            _, df, val_fold, test_fold, mode, img_size, root = args
            norm_name = "clip"
        elif len(args) == 3:
            mode, img_size, root = args
            df = pd.read_csv("./data/Industry/defect_supervised/yoke-suspension/anno/train.csv")
            val_fold = 0
            test_fold = 1
            norm_name = "clip"
        else:
            raise TypeError("CLAdapterDataset expects 8/7 official-train args or 3 legacy args.")

        self.mode = mode
        self.img_size = img_size
        self.root = root
        self.norm_name = norm_name
        self.df = self.select_split(df, mode, val_fold, test_fold).reset_index(drop=True)
        self.images = [path if os.path.isabs(path) else os.path.join(root, path) for path in self.df['image_path'].tolist()]
        self.labels = [int(label) for label in self.df['label'].tolist()]
        self.transforms = self.get_transforms()

    def select_split(self, df, mode, val_fold, test_fold):
        if 'split' in df.columns:
            split_name = 'val' if mode == 'valid' else mode
            return df[df['split'] == split_name].copy()
        if 'fold' in df.columns:
            if mode == 'train':
                return df[(df['fold'] != int(val_fold)) & (df['fold'] != int(test_fold))].copy()
            if mode == 'valid':
                return df[df['fold'] == int(val_fold)].copy()
            if mode == 'test':
                return df[df['fold'] == int(test_fold)].copy()
        if mode == 'train':
            return df.copy()
        return df.iloc[0:0].copy()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        image = self.transforms(image)
        label = torch.as_tensor(self.labels[index], dtype=torch.long)
        return image, label

    def get_transforms(self,):
        if self.norm_name == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = CLIP_MEAN, CLIP_STD
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.RandomAffine(
                    degrees=180,
                    translate=(0.05, 0.05),
                    scale=(0.70, 1.30),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transform


class SyntheticMaskCLAdapterDataset(CLAdapterDataset):
    """Training dataset with deterministic wire-ROI synthetic defects.

    Real images and labels are never overwritten.  Synthetic defects are only
    generated for normal training samples.  The exact generated pixel mask is
    reduced to the ViT 14x14 patch grid and returned as auxiliary supervision.
    """

    def __init__(
        self,
        *args,
        synthetic_probability=0.5,
        synthetic_seed=3107,
        patch_size=16,
        roi_top=0.28,
        roi_bottom=0.72,
    ):
        super().__init__(*args)
        if self.mode != "train":
            raise ValueError("SyntheticMaskCLAdapterDataset is train-only")
        if not 0.0 < synthetic_probability <= 1.0:
            raise ValueError("synthetic_probability must be in (0, 1]")
        if self.img_size % patch_size != 0:
            raise ValueError("image size must be divisible by patch size")
        self.synthetic_probability = float(synthetic_probability)
        self.synthetic_seed = int(synthetic_seed)
        self.patch_size = int(patch_size)
        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng(self, index):
        return random.Random(self.synthetic_seed + self.epoch * 1_000_003 + int(index))

    def _augment_clean(self, image, rng):
        image = TF.resize(
            image,
            [self.img_size, self.img_size],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        if rng.random() < 0.5:
            image = TF.hflip(image)
        if rng.random() < 0.5:
            image = TF.vflip(image)
        image = TF.adjust_brightness(image, rng.uniform(0.90, 1.10))
        image = TF.adjust_contrast(image, rng.uniform(0.90, 1.10))
        angle = rng.uniform(-180.0, 180.0)
        max_shift = int(round(self.img_size * 0.05))
        translate = [rng.randint(-max_shift, max_shift), rng.randint(-max_shift, max_shift)]
        image = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=rng.uniform(0.70, 1.30),
            shear=[0.0, 0.0],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        return image

    def _sample_wire_candidate(self, image, rng):
        gray_image = image.convert("L")
        gray = np.asarray(gray_image, dtype=np.float32)
        local_mean = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=5.0)), dtype=np.float32)
        dark_ridge = local_mean - gray
        first_row = int(round(self.roi_top * self.img_size))
        last_row = int(round(self.roi_bottom * self.img_size))
        roi = dark_ridge[first_row:last_row]
        cutoff = float(np.percentile(roi, 85.0))
        candidates = np.argwhere(roi >= cutoff)
        if len(candidates) == 0:
            return self.img_size // 2, self.img_size // 2
        y, x = candidates[rng.randrange(len(candidates))]
        return int(y + first_row), int(x)

    def _synthetic_defect(self, image, rng):
        height = width = self.img_size
        center_y, center_x = self._sample_wire_candidate(image, rng)
        long_side = rng.randint(14, 44)
        short_side = rng.randint(5, 13)
        theta = np.deg2rad(rng.uniform(0.0, 180.0))
        along = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
        across = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float32)
        center = np.asarray([center_x, center_y], dtype=np.float32)
        corners = []
        for along_sign, across_sign in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            point = (
                center
                + along_sign * 0.5 * long_side * along
                + across_sign * 0.5 * short_side * across
            )
            point[0] = np.clip(point[0], 0, width - 1)
            point[1] = np.clip(point[1], 0, height - 1)
            corners.append((float(point[0]), float(point[1])))

        mask_image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask_image).polygon(corners, fill=255)
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.2)))
        alpha = np.asarray(mask_image, dtype=np.float32) / 255.0
        alpha *= rng.uniform(0.65, 0.95)

        base = np.asarray(image, dtype=np.float32)
        shift_y = rng.choice((-1, 1)) * rng.randint(10, 36)
        shift_x = rng.choice((-1, 1)) * rng.randint(10, 48)
        source = np.roll(base, shift=(shift_y, shift_x), axis=(0, 1))
        source *= rng.uniform(0.55, 1.55)
        channel_gain = np.asarray(
            [rng.uniform(0.80, 1.20) for _ in range(3)], dtype=np.float32
        )
        source *= channel_gain.reshape(1, 1, 3)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0.0, rng.uniform(2.0, 10.0), size=source.shape
        )
        source = np.clip(source + noise, 0.0, 255.0)
        mixed = base * (1.0 - alpha[..., None]) + source * alpha[..., None]
        binary_mask = alpha >= 0.10
        return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8)), binary_mask

    def _to_normalized_tensor(self, image):
        if self.norm_name == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            mean, std = CLIP_MEAN, CLIP_STD
        return TF.normalize(TF.to_tensor(image), mean, std)

    def _patch_targets(self, binary_mask):
        mask = torch.from_numpy(binary_mask.astype(np.float32)).view(
            1, 1, self.img_size, self.img_size
        )
        coverage = F.avg_pool2d(mask, kernel_size=self.patch_size, stride=self.patch_size)
        return (coverage.flatten() >= 0.02).to(torch.float32)

    def __getitem__(self, index):
        rng = self._rng(index)
        image = Image.open(self.images[index]).convert("RGB")
        clean_image = self._augment_clean(image, rng)
        label = int(self.labels[index])
        use_synthetic = label == 0 and rng.random() < self.synthetic_probability
        if use_synthetic:
            synthetic_image, binary_mask = self._synthetic_defect(clean_image, rng)
            patch_targets = self._patch_targets(binary_mask)
        else:
            synthetic_image = clean_image
            grid = self.img_size // self.patch_size
            patch_targets = torch.zeros(grid * grid, dtype=torch.float32)
        return (
            self._to_normalized_tensor(clean_image),
            torch.as_tensor(label, dtype=torch.long),
            self._to_normalized_tensor(synthetic_image),
            patch_targets,
            torch.as_tensor(use_synthetic, dtype=torch.bool),
        )


class PoissonPerlinSyntheticMaskCLAdapterDataset(SyntheticMaskCLAdapterDataset):
    """NSA/DRAEM-inspired anomalies constrained to the estimated wire ROI.

    The irregular mask is Perlin-like low-frequency noise intersected with a
    dilated dark-ridge wire estimate.  Its appearance comes from a different
    normal image in the current outer-fold training partition and is inserted
    with Poisson mixed cloning.  A deterministic alpha blend is used only when
    OpenCV cannot solve a particular Poisson system.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_indices = [
            index for index, label in enumerate(self.labels) if int(label) == 0
        ]
        if len(self.normal_indices) < 2:
            raise ValueError("Poisson/Perlin synthesis needs at least two normal images")

    def _source_normal(self, target_index, rng):
        source_index = self.normal_indices[rng.randrange(len(self.normal_indices))]
        if source_index == target_index:
            position = self.normal_indices.index(source_index)
            source_index = self.normal_indices[(position + 1) % len(self.normal_indices)]
        source = Image.open(self.images[source_index]).convert("RGB")
        source = TF.resize(
            source,
            [self.img_size, self.img_size],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        if rng.random() < 0.5:
            source = TF.hflip(source)
        if rng.random() < 0.35:
            source = TF.vflip(source)
        return source

    def _wire_support(self, image):
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0)
        ridge = smooth - gray
        first_row = int(round(self.roi_top * self.img_size))
        last_row = int(round(self.roi_bottom * self.img_size))
        support = np.zeros_like(gray, dtype=np.uint8)
        roi = ridge[first_row:last_row]
        threshold = float(np.percentile(roi, 85.0))
        support[first_row:last_row] = (roi >= threshold).astype(np.uint8)
        yy, xx = np.ogrid[: self.img_size, : self.img_size]
        central_ellipse = (
            ((xx - self.img_size / 2) / (self.img_size * 0.43)) ** 2
            + ((yy - self.img_size / 2) / (self.img_size * 0.34)) ** 2
        ) <= 1.0
        support[~central_ellipse] = 0
        support = cv2.morphologyEx(
            support,
            cv2.MORPH_CLOSE,
            np.ones((3, 9), dtype=np.uint8),
        )
        support = cv2.dilate(support, np.ones((7, 7), dtype=np.uint8), iterations=1)
        return support.astype(bool), ridge

    def _wire_angle(self, ridge, center_y, center_x):
        radius = max(18, self.img_size // 8)
        y0, y1 = max(0, center_y - radius), min(self.img_size, center_y + radius + 1)
        x0, x1 = max(0, center_x - radius), min(self.img_size, center_x + radius + 1)
        local = ridge[y0:y1, x0:x1]
        cutoff = float(np.percentile(local, 80.0))
        points = np.argwhere(local >= cutoff)
        if len(points) < 8:
            return 0.0
        xy = points[:, ::-1].astype(np.float32)
        xy -= xy.mean(axis=0, keepdims=True)
        covariance = xy.T @ xy
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        return float(np.degrees(np.arctan2(direction[1], direction[0])))

    def _defect_size(self, rng):
        draw = rng.random()
        if draw < 0.55:  # Deliberately include 1-2 patch subtle defects.
            return rng.randint(8, 28), rng.randint(3, 12)
        if draw < 0.90:
            return rng.randint(20, 48), rng.randint(6, 20)
        return rng.randint(40, 72), rng.randint(10, 28)

    def _perlin_local_mask(self, width, height, rng):
        grid_h = rng.choice((2, 3, 4, 5))
        grid_w = rng.choice((2, 3, 4, 5))
        generator = np.random.default_rng(rng.randrange(2**32))
        coarse = generator.random((grid_h, grid_w), dtype=np.float32)
        noise = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=rng.uniform(0.4, 1.4))
        threshold = float(np.quantile(noise, rng.uniform(0.48, 0.68)))
        irregular = noise >= threshold
        yy, xx = np.ogrid[:height, :width]
        ellipse = (
            ((xx - (width - 1) / 2) / max(width / 2, 1)) ** 2
            + ((yy - (height - 1) / 2) / max(height / 2, 1)) ** 2
        ) <= 1.0
        return (irregular & ellipse).astype(np.uint8) * 255

    def _placed_mask(self, image, rng):
        wire_support, ridge = self._wire_support(image)
        yy, xx = np.ogrid[: self.img_size, : self.img_size]
        distance_squared = (xx - self.img_size / 2) ** 2 + (yy - self.img_size / 2) ** 2
        center_weight = np.exp(-distance_squared / (2.0 * (self.img_size * 0.22) ** 2))
        candidate_values = np.maximum(ridge, 0.0) * center_weight
        candidate_values[~wire_support] = 0.0
        positive = candidate_values > 0
        if not positive.any():
            center_y = center_x = self.img_size // 2
        else:
            cutoff = float(np.percentile(candidate_values[positive], 88.0))
            candidates = np.argwhere(candidate_values >= cutoff)
            center_y, center_x = candidates[rng.randrange(len(candidates))]
            center_y, center_x = int(center_y), int(center_x)

        long_side, short_side = self._defect_size(rng)
        local = self._perlin_local_mask(long_side, short_side, rng)
        canvas = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        y0 = max(0, center_y - short_side // 2)
        x0 = max(0, center_x - long_side // 2)
        y1 = min(self.img_size, y0 + short_side)
        x1 = min(self.img_size, x0 + long_side)
        canvas[y0:y1, x0:x1] = local[: y1 - y0, : x1 - x0]

        angle = self._wire_angle(ridge, center_y, center_x) + rng.uniform(-18.0, 18.0)
        matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        canvas = cv2.warpAffine(
            canvas,
            matrix,
            (self.img_size, self.img_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        constrained = (canvas > 0) & wire_support
        if int(constrained.sum()) < 12:
            fallback = np.zeros_like(canvas)
            cv2.ellipse(
                fallback,
                (center_x, center_y),
                (max(3, long_side // 2), max(2, short_side // 2)),
                angle,
                0,
                360,
                255,
                -1,
            )
            constrained = (fallback > 0) & wire_support
        if int(constrained.sum()) < 12:
            yy, xx = np.ogrid[: self.img_size, : self.img_size]
            local_radius = max(6, min(long_side, short_side) // 2 + 4)
            local_disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= local_radius**2
            constrained = wire_support & local_disk
        if int(constrained.sum()) < 12:
            support_points = np.argwhere(wire_support)
            if len(support_points) == 0:
                raise RuntimeError("Automatic wire ROI estimation produced no pixels")
            distances = (
                (support_points[:, 0] - center_y) ** 2
                + (support_points[:, 1] - center_x) ** 2
            )
            take = support_points[np.argsort(distances)[: min(32, len(support_points))]]
            constrained = np.zeros_like(wire_support, dtype=bool)
            constrained[take[:, 0], take[:, 1]] = True
        return constrained, (center_x, center_y)

    def _alter_source(self, source, rng):
        array = np.asarray(source, dtype=np.float32)
        mean = array.mean(axis=(0, 1), keepdims=True)
        array = (array - mean) * rng.uniform(0.55, 1.55) + mean
        array += rng.uniform(-45.0, 45.0)
        gains = np.asarray(
            [rng.uniform(0.75, 1.25) for _ in range(3)], dtype=np.float32
        ).reshape(1, 1, 3)
        array *= gains
        generator = np.random.default_rng(rng.randrange(2**32))
        array += generator.normal(0.0, rng.uniform(0.0, 9.0), array.shape)
        array = np.clip(array, 0, 255).astype(np.uint8)
        blur = rng.choice((0, 0, 3, 5))
        if blur:
            array = cv2.GaussianBlur(array, (blur, blur), sigmaX=0)
        return array

    def _synthetic_defect_from_other_normal(self, image, source, rng):
        binary_mask, center = self._placed_mask(image, rng)
        target_array = np.asarray(image, dtype=np.uint8)
        source_array = self._alter_source(source, rng)
        mask = binary_mask.astype(np.uint8) * 255
        try:
            # RGB/BGR channel order is immaterial because both operands use the
            # same order and the result is converted back without color mixing.
            mixed = cv2.seamlessClone(
                source_array,
                target_array,
                mask,
                center,
                cv2.MIXED_CLONE,
            )
        except cv2.error:
            mixed = target_array.copy()

        # On low-texture wire crops, a pure Poisson solution can collapse to
        # the target even when a non-empty mask is supplied.  Retain its smooth
        # boundary while injecting a controlled fraction of source residual so
        # that the pixel target and the patch label cannot contradict each
        # other.
        alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.9)
        strength = rng.uniform(0.30, 0.62)
        hybrid_alpha = np.clip(alpha * strength, 0.0, 1.0)[..., None]
        mixed = (
            mixed.astype(np.float32) * (1.0 - hybrid_alpha)
            + source_array.astype(np.float32) * hybrid_alpha
        )

        masked_difference = np.abs(mixed - target_array.astype(np.float32))[binary_mask]
        if masked_difference.size == 0 or float(masked_difference.mean()) < 5.0:
            signed_offset = rng.choice((-1.0, 1.0)) * rng.uniform(18.0, 46.0)
            emphasized = np.clip(source_array.astype(np.float32) + signed_offset, 0, 255)
            rescue_alpha = np.clip(alpha * rng.uniform(0.48, 0.78), 0.0, 1.0)[..., None]
            mixed = (
                target_array.astype(np.float32) * (1.0 - rescue_alpha)
                + emphasized * rescue_alpha
            )
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed), binary_mask

    def __getitem__(self, index):
        rng = self._rng(index)
        image = Image.open(self.images[index]).convert("RGB")
        clean_image = self._augment_clean(image, rng)
        label = int(self.labels[index])
        use_synthetic = label == 0 and rng.random() < self.synthetic_probability
        if use_synthetic:
            source = self._source_normal(index, rng)
            synthetic_image, binary_mask = self._synthetic_defect_from_other_normal(
                clean_image, source, rng
            )
            patch_targets = self._patch_targets(binary_mask)
        else:
            synthetic_image = clean_image
            grid = self.img_size // self.patch_size
            patch_targets = torch.zeros(grid * grid, dtype=torch.float32)
        return (
            self._to_normalized_tensor(clean_image),
            torch.as_tensor(label, dtype=torch.long),
            self._to_normalized_tensor(synthetic_image),
            patch_targets,
            torch.as_tensor(use_synthetic, dtype=torch.bool),
        )
