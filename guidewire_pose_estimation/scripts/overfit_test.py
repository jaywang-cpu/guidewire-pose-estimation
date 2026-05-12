"""
Overfit test: train on 10 images, no augmentation, high LR.
Goal: loss → ~0, position error < 5px.
If this fails, there's a bug in model/loss/data pipeline.
"""

import os
import sys

# Resolve the package directory (parent of this scripts/ folder)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "modeling"))

import torch
import numpy as np
from config import get_config, DataConfig, TrainConfig, ModelConfig, ExperimentConfig
from dataset import load_data, GuidewireDataset
from torch.utils.data import DataLoader
from model import build_model, GuidewireLoss
from modeling.train import get_device, set_seed, _compute_real_metrics

set_seed(42)

# Load data, take only 10 samples
config = get_config("baseline")
images, positions, directions = load_data(config.data)
images = images[:10]
positions = positions[:10]
directions = directions[:10]

# Dataset: no augmentation
ds = GuidewireDataset(images, positions, directions, config.data, augment=False)
loader = DataLoader(ds, batch_size=10, shuffle=False)

# Model
device = get_device(config)
model = build_model(config.model).to(device)
model.unfreeze_backbone()  # no freezing

# High LR, no weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = GuidewireLoss(
    position_weight=10.0,  # boost position signal
    direction_weight=1.0,
    position_loss_fn="smooth_l1",
    direction_loss_fn="cosine",
)

print(f"Device: {device}")
print(f"Overfit test: 10 images, 500 epochs, lr=1e-3")
print("-" * 60)

for epoch in range(500):
    model.train()
    for imgs, targets in loader:
        imgs = imgs.to(device)
        tgt = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        preds = model(imgs)
        loss_dict = criterion(preds, tgt)
        loss_dict["loss"].backward()
        optimizer.step()

    if epoch % 25 == 0 or epoch == 499:
        model.eval()
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(device)
                tgt = {k: v.to(device) for k, v in targets.items()}
                preds = model(imgs)
                loss_dict = criterion(preds, tgt)
                metrics = _compute_real_metrics(preds, targets, config.data.image_size)

        print(
            f"Epoch {epoch:3d} | loss={loss_dict['loss'].item():.4f} "
            f"pos_loss={loss_dict['pos_loss'].item():.5f} "
            f"dir_loss={loss_dict['dir_loss'].item():.4f} "
            f"| euc={metrics['euc_px']:.1f}px ang={metrics['ang_deg']:.1f}°"
        )

print("-" * 60)
if metrics['euc_px'] < 5 and metrics['ang_deg'] < 5:
    print("PASS: Model can memorize 10 samples. Code is correct.")
    print("Next step: tune hyperparameters for full training.")
elif metrics['euc_px'] < 20:
    print("PARTIAL: Position learning works but slowly. May need more epochs or higher LR.")
else:
    print("FAIL: Model cannot memorize 10 samples. Bug in model/loss/data pipeline.")
