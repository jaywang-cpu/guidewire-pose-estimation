"""
Dataset module for Guidewire Pose Estimation.
Handles data loading, preprocessing, augmentation, and train/val/test splitting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import cv2
from config import DataConfig


class GuidewireDataset(Dataset):
    """
    PyTorch Dataset for fluoroscopic guidewire images.

    Each sample returns:
        - image: (1, H, W) float32 tensor, normalized to [0, 1]
        - targets: dict with:
            - positions: (num_wires, 2) normalized tip positions (x, y) in [0, 1]
            - directions: (num_wires, 2) direction as (sin θ, cos θ)
            - angles: (num_wires,) angle in radians
            - positions_px: (num_wires, 2) original pixel positions (for evaluation)
    """

    def __init__(
        self,
        images: np.ndarray,
        positions: np.ndarray,
        directions: np.ndarray,
        config: DataConfig,
        augment: bool = False,
    ):
        self.config = config
        self.augment = augment
        self.input_size = config.input_size
        self.orig_size = config.image_size

        self.images = images.astype(np.float32)
        self.positions = positions.astype(np.float64)
        self.directions = directions.astype(np.float64)

        # Sort wires by x-coordinate so output ordering is deterministic.
        # This eliminates the need for Hungarian matching during training.
        for i in range(len(self.positions)):
            if self.positions[i, 0, 0] > self.positions[i, 1, 0]:
                self.positions[i] = self.positions[i, [1, 0]]
                self.directions[i] = self.directions[i, [1, 0]]

        # Compute angles from direction vectors (directions are stored as (sin θ, cos θ))
        self.angles = np.arctan2(
            self.directions[:, :, 0], self.directions[:, :, 1]
        )  # (N, 2) — arctan2(sin θ, cos θ) = θ

        # Percentile-based normalization (robust to outliers in X-ray)
        self.img_p1 = float(np.percentile(self.images, 1))
        self.img_p99 = float(np.percentile(self.images, 99))

    def __len__(self) -> int:
        return len(self.images)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Percentile-based normalization to [0, 1], then clip."""
        img = (img - self.img_p1) / (self.img_p99 - self.img_p1 + 1e-8)
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to input_size x input_size."""
        return cv2.resize(
            img, (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR
        )

    def _apply_augmentation(
        self, img: np.ndarray, positions: np.ndarray, angles: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation to image and labels jointly."""
        h, w = img.shape[:2]
        cfg = self.config

        # Random horizontal flip
        if np.random.rand() < cfg.aug_hflip:
            img = np.fliplr(img).copy()
            positions[:, 0] = (w - 1) - positions[:, 0]
            angles = np.pi - angles

        # Random vertical flip
        if np.random.rand() < cfg.aug_vflip:
            img = np.flipud(img).copy()
            positions[:, 1] = (h - 1) - positions[:, 1]
            angles = -angles

        # Random rotation
        if cfg.aug_rotation_degrees > 0:
            angle_deg = np.random.uniform(
                -cfg.aug_rotation_degrees, cfg.aug_rotation_degrees
            )
            angle_rad = np.deg2rad(angle_deg)
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Transform positions with the same affine matrix
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            for i in range(positions.shape[0]):
                px, py = positions[i, 0] - cx, positions[i, 1] - cy
                positions[i, 0] = cos_a * px + sin_a * py + cx
                positions[i, 1] = -sin_a * px + cos_a * py + cy

            # Direction angles rotate by the same amount
            angles = angles - angle_rad

        # Random intensity scaling
        scale = np.random.uniform(*cfg.aug_intensity_scale)
        img = np.clip(img * scale, 0, 1)

        # Random Gaussian noise
        if cfg.aug_gaussian_noise_std > 0:
            noise = np.random.normal(0, cfg.aug_gaussian_noise_std, img.shape)
            img = np.clip(img + noise, 0, 1).astype(np.float32)

        # Wrap angles to [-pi, pi]
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        return img, positions, angles

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img = self.images[idx].copy()
        img = self._normalize_image(img)
        img = self._resize_image(img)

        # Scale positions to resized coordinates
        scale = self.input_size / self.orig_size
        positions = self.positions[idx].copy()  # (2, 2) in original pixels
        positions_px_orig = positions.copy()    # keep original for eval
        positions = positions * scale           # to resized pixel space
        angles = self.angles[idx].copy()        # (2,)

        # Augmentation (on resized image + scaled positions)
        if self.augment and self.config.augment_train:
            img, positions, angles = self._apply_augmentation(
                img, positions, angles
            )
            # Re-sort by x after augmentation (flips can swap wire order)
            if positions[0, 0] > positions[1, 0]:
                positions = positions[[1, 0]]
                angles = angles[[1, 0]]

        # Normalize positions to [0, 1]
        positions_normalized = positions / self.input_size

        # Direction as (sin θ, cos θ) — always use sincos for stability
        direction_target = np.stack(
            [np.sin(angles), np.cos(angles)], axis=-1
        )  # (2, 2)

        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        targets = {
            "positions": torch.from_numpy(positions_normalized).float(),
            "directions": torch.from_numpy(direction_target).float(),
            "angles": torch.from_numpy(angles).float(),
            "positions_px": torch.from_numpy(positions_px_orig).float(),
        }

        return img_tensor, targets


def load_data(config: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw dataset from .npz file."""
    data = np.load(config.data_path, allow_pickle=True)
    return data["images"], data["positions"], data["directions"]


def split_data(
    n_samples: int, config: DataConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data indices into train/val/test sets.

    NOTE: Consecutive images may come from the same cadaveric specimen/anatomical
    site. To reduce potential data leakage, we split by contiguous blocks rather
    than purely random shuffling. This approximates a per-subject split when
    exact subject IDs are unavailable.
    """
    rng = np.random.RandomState(config.split_seed)

    # Group consecutive images into blocks of ~10 to approximate per-acquisition splits
    block_size = 10
    n_blocks = (n_samples + block_size - 1) // block_size
    block_indices = rng.permutation(n_blocks)

    # Assign blocks to splits
    n_train_blocks = int(n_blocks * config.train_ratio)
    n_val_blocks = int(n_blocks * config.val_ratio)

    train_blocks = block_indices[:n_train_blocks]
    val_blocks = block_indices[n_train_blocks:n_train_blocks + n_val_blocks]
    test_blocks = block_indices[n_train_blocks + n_val_blocks:]

    def blocks_to_indices(blocks):
        indices = []
        for b in blocks:
            start = b * block_size
            end = min(start + block_size, n_samples)
            indices.extend(range(start, end))
        return np.array(sorted(indices))

    return blocks_to_indices(train_blocks), blocks_to_indices(val_blocks), blocks_to_indices(test_blocks)


def create_dataloaders(
    config: DataConfig, train_config=None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create train, validation, and test DataLoaders."""
    images, positions, directions = load_data(config)
    train_idx, val_idx, test_idx = split_data(len(images), config)

    train_dataset = GuidewireDataset(
        images[train_idx], positions[train_idx], directions[train_idx],
        config, augment=True,
    )
    val_dataset = GuidewireDataset(
        images[val_idx], positions[val_idx], directions[val_idx],
        config, augment=False,
    )
    test_dataset = GuidewireDataset(
        images[test_idx], positions[test_idx], directions[test_idx],
        config, augment=False,
    )

    # Use TRAINING set normalization stats for val/test (prevent data leakage)
    val_dataset.img_p1 = train_dataset.img_p1
    val_dataset.img_p99 = train_dataset.img_p99
    test_dataset.img_p1 = train_dataset.img_p1
    test_dataset.img_p99 = train_dataset.img_p99

    batch_size = train_config.batch_size if train_config else 8

    # MPS on macOS does not support multiprocessing well — use 0 workers
    import platform
    if platform.system() == "Darwin":
        num_workers = 0
        pin_memory = False
    else:
        num_workers = train_config.num_workers if train_config else 4
        pin_memory = train_config.pin_memory if train_config else True

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    info = {
        "n_total": len(images),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "img_p1": train_dataset.img_p1,
        "img_p99": train_dataset.img_p99,
    }

    return train_loader, val_loader, test_loader, info
