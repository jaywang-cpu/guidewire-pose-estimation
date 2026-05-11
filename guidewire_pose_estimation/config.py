"""
Configuration for Guidewire Pose Estimation Project.
EN.580.627 Deep Learning for Medical Imaging - Final Project #1
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    data_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "raw", "GuidewireDataset.npz"
    )
    image_size: int = 976          # original image size
    input_size: int = 384          # resized input for the model
    num_guidewires: int = 2        # each image contains exactly 2 guidewires

    # Train / Val / Test split ratios
    train_ratio: float = 0.60
    val_ratio: float = 0.25
    test_ratio: float = 0.15
    split_seed: int = 42          # for reproducibility

    # Normalization
    normalize_positions: bool = True   # normalize positions to [0, 1] relative to image size
    direction_repr: str = "sincos"     # "sincos" for (sin θ, cos θ) or "angle" for raw θ in radians

    # Augmentation (geometric aug disabled — causes label ordering issues
    # and hurts more than helps on this small dataset. Use mixup instead.)
    augment_train: bool = False
    aug_hflip: float = 0.5
    aug_vflip: float = 0.5
    aug_rotation_degrees: float = 15.0
    aug_scale_range: Tuple[float, float] = (0.9, 1.1)
    aug_translate: Tuple[float, float] = (0.05, 0.05)
    aug_intensity_scale: Tuple[float, float] = (0.9, 1.1)
    aug_gaussian_noise_std: float = 0.02


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "resnet18"              # resnet18, resnet34, resnet50, efficientnet_b0
    pretrained: bool = True
    in_channels: int = 1                    # grayscale fluoroscopy
    num_guidewires: int = 2

    # Output: per guidewire -> (x, y, sin_theta, cos_theta)
    outputs_per_wire: int = 4
    total_outputs: int = 8                  # num_guidewires * outputs_per_wire

    # Heatmap head (for heatmap-based approach)
    use_heatmap_head: bool = False
    heatmap_size: int = 128                 # output heatmap resolution (higher = more precise)
    heatmap_sigma: float = 3.0              # Gaussian sigma for target heatmaps

    dropout: float = 0.3
    freeze_backbone_epochs: int = 0         # no freezing — small dataset benefits from full tuning


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimization
    batch_size: int = 8
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    optimizer: str = "adamw"                # adam, adamw, sgd
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    scheduler: str = "cosine"               # cosine, step, plateau
    scheduler_patience: int = 10            # for ReduceLROnPlateau
    scheduler_step_size: int = 30           # for StepLR
    scheduler_gamma: float = 0.1
    warmup_epochs: int = 0

    # Differential LR: backbone gets learning_rate * backbone_lr_scale
    backbone_lr_scale: float = 0.1

    # Loss weights
    position_loss_weight: float = 5.0
    direction_loss_weight: float = 5.0
    position_loss_fn: str = "mse"           # mse, l1, smooth_l1
    direction_loss_fn: str = "cosine"       # mse, cosine

    # Training
    early_stopping_patience: int = 30
    gradient_clip_norm: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpointing
    checkpoint_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints"
    )
    save_best_only: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    results_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    figures_dir: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"
    )
    # Confidence interval
    ci_level: float = 0.95
    ci_n_bootstrap: int = 1000

    # Visualization
    num_qualitative_samples: int = 10
    arrow_length: float = 50.0              # arrow length for direction visualization


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_name: str = "guidewire_pose_resnet18"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    device: str = "auto"  # auto, cuda, mps, cpu


def get_config(experiment: str = "baseline") -> ExperimentConfig:
    """Get predefined experiment configurations."""
    if experiment == "baseline":
        return ExperimentConfig(
            experiment_name="baseline_resnet18",
            model=ModelConfig(backbone="resnet18", pretrained=True),
        )
    elif experiment == "resnet34":
        return ExperimentConfig(
            experiment_name="ablation_resnet34",
            model=ModelConfig(backbone="resnet34", pretrained=True),
        )
    elif experiment == "resnet50":
        return ExperimentConfig(
            experiment_name="ablation_resnet50",
            model=ModelConfig(backbone="resnet50", pretrained=True),
        )
    elif experiment == "heatmap":
        return ExperimentConfig(
            experiment_name="heatmap_resnet18",
            model=ModelConfig(
                backbone="resnet18", pretrained=True,
                use_heatmap_head=True,
            ),
        )
    elif experiment == "from_scratch":
        return ExperimentConfig(
            experiment_name="ablation_no_pretrain",
            model=ModelConfig(backbone="resnet18", pretrained=False),
        )
    elif experiment == "no_augmentation":
        return ExperimentConfig(
            experiment_name="ablation_no_aug",
            data=DataConfig(augment_train=False),
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
