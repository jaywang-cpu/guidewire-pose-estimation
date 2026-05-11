"""
Training pipeline for Guidewire Pose Estimation.
Includes training loop with real-metric tracking, validation, checkpointing,
early stopping, and logging.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from config import ExperimentConfig
from model import build_model, GuidewireLoss


def get_device(config: ExperimentConfig) -> torch.device:
    """Select computation device."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(config.device)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """
    Build optimizer with differential learning rates:
    - Pretrained backbone gets backbone_lr_scale * learning_rate (default 0.1x)
    - New heads get the full learning_rate
    """
    tc = config.train
    backbone_lr = tc.learning_rate * tc.backbone_lr_scale

    # Separate backbone vs head parameters
    # Backbone: "backbone.*" for regression model, "stem.*"/"layer1-4.*" for heatmap model
    backbone_params = []
    head_params = []
    backbone_prefixes = ("backbone", "encoder", "stem", "layer1", "layer2", "layer3", "layer4")
    for name, param in model.named_parameters():
        if name.startswith(backbone_prefixes):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": tc.learning_rate},
    ]

    if tc.optimizer == "adamw":
        return AdamW(param_groups, weight_decay=tc.weight_decay)
    elif tc.optimizer == "adam":
        return Adam(param_groups, weight_decay=tc.weight_decay)
    elif tc.optimizer == "sgd":
        return SGD(param_groups, momentum=0.9, weight_decay=tc.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {tc.optimizer}")


def build_scheduler(optimizer, config: ExperimentConfig):
    tc = config.train
    if tc.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=tc.num_epochs - tc.warmup_epochs, eta_min=1e-7)
    elif tc.scheduler == "step":
        return StepLR(optimizer, step_size=tc.scheduler_step_size, gamma=tc.scheduler_gamma)
    elif tc.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=tc.scheduler_patience, factor=tc.scheduler_gamma)
    else:
        raise ValueError(f"Unknown scheduler: {tc.scheduler}")


class WarmupWrapper:
    """Linear warmup wrapper for any scheduler. Respects per-group initial LRs."""

    def __init__(self, optimizer, scheduler, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        # Store each param group's initial LR so warmup scales them proportionally
        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int, val_loss: Optional[float] = None):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
            for pg, init_lr in zip(self.optimizer.param_groups, self.initial_lrs):
                pg["lr"] = init_lr * scale
        else:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _compute_real_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    orig_size: int = 976,
) -> Dict[str, float]:
    """
    Compute interpretable metrics (Euclidean distance in px, angular error in deg)
    from a single batch. Used during training for monitoring.
    Applies Hungarian matching to align predictions with ground truth.
    """
    pred_pos = predictions["positions"].detach().cpu().numpy() * orig_size  # (B, 2, 2)
    true_pos = targets["positions_px"].cpu().numpy()                        # (B, 2, 2)

    pred_dir = predictions["directions"].detach().cpu().numpy()
    pred_angles = np.arctan2(pred_dir[:, :, 0], pred_dir[:, :, 1])
    true_angles = targets["angles"].cpu().numpy()

    # Hungarian matching: reorder predictions per sample
    for i in range(pred_pos.shape[0]):
        cost_id = (np.sum((pred_pos[i, 0] - true_pos[i, 0]) ** 2) +
                   np.sum((pred_pos[i, 1] - true_pos[i, 1]) ** 2))
        cost_sw = (np.sum((pred_pos[i, 0] - true_pos[i, 1]) ** 2) +
                   np.sum((pred_pos[i, 1] - true_pos[i, 0]) ** 2))
        if cost_sw < cost_id:
            pred_pos[i] = pred_pos[i, [1, 0]]
            pred_angles[i] = pred_angles[i, [1, 0]]

    euc = np.sqrt(np.sum((pred_pos - true_pos) ** 2, axis=-1)).mean()

    diff = pred_angles - true_angles
    ang_err = np.abs(np.arctan2(np.sin(diff), np.cos(diff))) * 180 / np.pi
    ang = ang_err.mean()

    return {"euc_px": float(euc), "ang_deg": float(ang)}


class Trainer:
    """Full training pipeline with real-metric tracking."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = GuidewireLoss(
            position_weight=config.train.position_loss_weight,
            direction_weight=config.train.direction_loss_weight,
            position_loss_fn=config.train.position_loss_fn,
            direction_loss_fn=config.train.direction_loss_fn,
            heatmap_sigma=config.model.heatmap_sigma,
            heatmap_size=config.model.heatmap_size,
        )

        self.optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(self.optimizer, config)
        self.scheduler = WarmupWrapper(
            self.optimizer, scheduler,
            config.train.warmup_epochs, config.train.learning_rate
        )

        self.early_stopping = EarlyStopping(config.train.early_stopping_patience)

        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        os.makedirs(config.train.checkpoint_dir, exist_ok=True)

    def _mixup_batch(self, images, targets_gpu):
        """Apply mixup: interpolate images and targets within the batch."""
        alpha = self.config.train.mixup_alpha
        lam = float(np.random.beta(alpha, alpha))
        B = images.size(0)
        idx = torch.randperm(B, device=images.device)

        images = lam * images + (1 - lam) * images[idx]
        mixed_targets = {}
        for k, v in targets_gpu.items():
            if k == "directions":
                # Interpolate then renormalize to unit vector
                mixed = lam * v + (1 - lam) * v[idx]
                mixed_targets[k] = torch.nn.functional.normalize(mixed, dim=-1)
            else:
                mixed_targets[k] = lam * v + (1 - lam) * v[idx]
        return images, mixed_targets

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets_gpu = {k: v.to(self.device) for k, v in targets.items()}

            # Apply mixup if enabled
            if self.config.train.use_mixup:
                images, targets_gpu = self._mixup_batch(images, targets_gpu)

            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets_gpu)

            loss_dict["loss"].backward()

            if self.config.train.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.gradient_clip_norm
                )

            self.optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
            n_batches += 1

        return {k: v / n_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def _validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Returns (loss_dict, real_metrics_dict)."""
        self.model.eval()
        epoch_losses = defaultdict(float)
        all_euc = []
        all_ang = []
        n_batches = 0

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets_gpu = {k: v.to(self.device) for k, v in targets.items()}

            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets_gpu)

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()

            # Real metrics in interpretable units
            m = _compute_real_metrics(predictions, targets, self.config.data.image_size)
            all_euc.append(m["euc_px"])
            all_ang.append(m["ang_deg"])
            n_batches += 1

        losses = {k: v / n_batches for k, v in epoch_losses.items()}
        metrics = {
            "val_euc_px": float(np.mean(all_euc)),
            "val_ang_deg": float(np.mean(all_ang)),
        }
        return losses, metrics

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": {
                "experiment_name": self.config.experiment_name,
                "backbone": self.config.model.backbone,
            },
        }
        if is_best:
            path = os.path.join(
                self.config.train.checkpoint_dir,
                f"{self.config.experiment_name}_best.pth"
            )
            torch.save(checkpoint, path)

    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """Run the full training loop."""
        tc = self.config.train

        if verbose:
            print(f"Training {self.config.experiment_name} on {self.device}")
            print(f"  Model: {self.config.model.backbone}, "
                  f"Epochs: {tc.num_epochs}, BS: {tc.batch_size}, LR: {tc.learning_rate}")
            print(f"  Train batches: {len(self.train_loader)}, "
                  f"Val batches: {len(self.val_loader)}")
            print("-" * 80)

        for epoch in range(tc.num_epochs):
            t0 = time.time()

            # Freeze/unfreeze backbone for transfer learning
            if epoch == 0 and self.config.model.freeze_backbone_epochs > 0:
                self.model.freeze_backbone()
                if verbose:
                    print(f"  [Backbone frozen for first {self.config.model.freeze_backbone_epochs} epochs]")
            elif epoch == self.config.model.freeze_backbone_epochs:
                self.model.unfreeze_backbone()
                if verbose:
                    print(f"  [Backbone unfrozen at epoch {epoch}]")

            train_losses = self._train_one_epoch(epoch)
            val_losses, val_metrics = self._validate()

            self.scheduler.step(epoch, val_losses["loss"])

            # Log everything
            for k, v in train_losses.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_losses.items():
                self.history[f"val_{k}"].append(v)
            for k, v in val_metrics.items():
                self.history[k].append(v)
            self.history["lr"].append(self.scheduler.get_lr())

            is_best = val_losses["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["loss"]
                self.best_epoch = epoch
            self._save_checkpoint(epoch, val_losses["loss"], is_best)

            dt = time.time() - t0
            if verbose and (epoch % 5 == 0 or epoch == tc.num_epochs - 1 or is_best):
                star = " *" if is_best else ""
                swap_info = ""
                if "train_swap_rate" in self.history:
                    swap_info = f" swap={train_losses.get('swap_rate', 0):.0%}"
                print(
                    f"  Epoch {epoch:3d}/{tc.num_epochs} | "
                    f"Loss: {train_losses['loss']:.4f}/{val_losses['loss']:.4f} | "
                    f"Pos: {val_metrics['val_euc_px']:.1f}px | "
                    f"Ang: {val_metrics['val_ang_deg']:.1f}\u00b0 | "
                    f"LR: {self.scheduler.get_lr():.2e} | "
                    f"{dt:.1f}s{swap_info}{star}"
                )

            if self.early_stopping.step(val_losses["loss"]):
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}")
                break

        # Save history
        history_path = os.path.join(
            self.config.train.checkpoint_dir,
            f"{self.config.experiment_name}_history.json"
        )
        with open(history_path, "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in self.history.items()}, f)

        if verbose:
            print(f"\n  Training complete. Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
            best_idx = self.best_epoch
            if best_idx < len(self.history.get("val_euc_px", [])):
                print(f"  Best val metrics: {self.history['val_euc_px'][best_idx]:.1f} px, "
                      f"{self.history['val_ang_deg'][best_idx]:.1f}\u00b0")

        return dict(self.history)

    def load_best(self):
        path = os.path.join(
            self.config.train.checkpoint_dir,
            f"{self.config.experiment_name}_best.pth"
        )
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded best model from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")
        return self.model


def train_experiment(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[nn.Module, Dict]:
    """Convenience function to train a single experiment end-to-end."""
    device = get_device(config)
    set_seed(config.seed)

    model = build_model(config.model)
    trainer = Trainer(config, model, train_loader, val_loader, device)
    history = trainer.train()
    best_model = trainer.load_best()

    return best_model, history
