"""
Run the complete Guidewire Pose Estimation pipeline:
  1. Data exploration & figure generation
  2. Train baseline (ResNet-18, pretrained, with augmentation)
  3. Train ablation experiments (ResNet-34, ResNet-50, no pretrain, no aug)
  4. Train heatmap model (U-Net decoder with skip connections)
  5. Evaluate all on test set (with test-time augmentation)
  6. Generate comparison figures & statistical tests
  7. Save all results
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script mode
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

import torch

# Project imports
from config import get_config
from dataset import load_data, split_data, create_dataloaders, GuidewireDataset
from model import build_model, GuidewireLoss
from modeling.train import train_experiment, get_device, set_seed
from modeling.evaluate import (
    run_inference, compute_all_metrics, print_metrics,
    plot_training_curves, plot_lr_curve, plot_error_distributions,
    plot_qualitative_results, plot_worst_cases, plot_best_cases,
    plot_scatter_errors, compare_experiments, compare_significance,
    save_metrics, euclidean_distance, angular_error,
)

config = get_config("baseline")
FIGURES_DIR = config.eval.figures_dir
RESULTS_DIR = config.eval.results_dir
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device_config = get_config("baseline")
device = get_device(device_config)
print(f"Device: {device}")
print(f"Output dirs: figures={FIGURES_DIR}, results={RESULTS_DIR}")

# ======================================================================
# STEP 1: Data Exploration Figures
# ======================================================================
print("\n" + "=" * 70)
print("STEP 1: Data Exploration")
print("=" * 70)

images, positions, directions = load_data(config.data)
print(f"Loaded {len(images)} images, shape={images.shape}, dtype={images.dtype}")

# --- Figure 1: Sample images with annotations ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
sample_indices = [0, 50, 100, 150, 200, 250, 290, 313]

for ax, idx in zip(axes.flatten(), sample_indices):
    img = images[idx].astype(np.float32)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
    ax.imshow(img, cmap="gray")

    for w in range(2):
        px, py = positions[idx, w]
        dx, dy = directions[idx, w]
        norm = np.sqrt(dx ** 2 + dy ** 2)
        dx_n, dy_n = dx / norm * 50, dy / norm * 50

        color = "lime" if w == 0 else "cyan"
        ax.plot(px, py, "o", color=color, markersize=6)
        ax.arrow(px, py, dx_n, dy_n, head_width=8, head_length=5,
                 fc=color, ec=color, linewidth=2)

    ax.set_title(f"Image {idx}", fontsize=10)
    ax.axis("off")

plt.suptitle("Sample Images with Guidewire Annotations (Green=Wire1, Cyan=Wire2)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "data_samples.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: data_samples.png")

# --- Figure 2: Annotation distributions ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for w in range(2):
    axes[0, 0].scatter(positions[:, w, 0], positions[:, w, 1],
                       alpha=0.5, s=10, label=f"Wire {w + 1}")
axes[0, 0].set_xlabel("X (pixels)")
axes[0, 0].set_ylabel("Y (pixels)")
axes[0, 0].set_title("Tip Positions")
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 976)
axes[0, 0].set_ylim(976, 0)
axes[0, 0].set_aspect("equal")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(positions[:, 0, 0], bins=30, alpha=0.6, label="Wire 1")
axes[0, 1].hist(positions[:, 1, 0], bins=30, alpha=0.6, label="Wire 2")
axes[0, 1].set_xlabel("X position (pixels)")
axes[0, 1].set_title("X Position Distribution")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(positions[:, 0, 1], bins=30, alpha=0.6, label="Wire 1")
axes[0, 2].hist(positions[:, 1, 1], bins=30, alpha=0.6, label="Wire 2")
axes[0, 2].set_xlabel("Y position (pixels)")
axes[0, 2].set_title("Y Position Distribution")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

angles_deg = np.arctan2(directions[:, :, 0], directions[:, :, 1]) * 180 / np.pi
axes[1, 0].hist(angles_deg[:, 0], bins=30, alpha=0.6, label="Wire 1")
axes[1, 0].hist(angles_deg[:, 1], bins=30, alpha=0.6, label="Wire 2")
axes[1, 0].set_xlabel("Angle (degrees)")
axes[1, 0].set_title("Orientation Distribution")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

norms = np.linalg.norm(directions, axis=2)
axes[1, 1].hist(norms[:, 0], bins=30, alpha=0.6, label="Wire 1")
axes[1, 1].hist(norms[:, 1], bins=30, alpha=0.6, label="Wire 2")
axes[1, 1].set_xlabel("Direction Vector Norm")
axes[1, 1].set_title("Direction Vector Magnitude")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

tip_distances = np.sqrt(np.sum((positions[:, 0] - positions[:, 1]) ** 2, axis=1))
axes[1, 2].hist(tip_distances, bins=30, alpha=0.7, color="purple")
axes[1, 2].set_xlabel("Distance (pixels)")
axes[1, 2].set_title(f"Inter-tip Distance (mean={tip_distances.mean():.1f} px)")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "data_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: data_analysis.png")

# --- Figure 3: Image intensity distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
flat_sample = images[::10].flatten()  # subsample for speed
ax.hist(flat_sample, bins=200, color="#2196F3", alpha=0.7, log=True)
ax.axvline(np.percentile(flat_sample, 1), color="red", linestyle="--", label="p1")
ax.axvline(np.percentile(flat_sample, 99), color="red", linestyle="--", label="p99")
ax.set_xlabel("Pixel Intensity (uint16)")
ax.set_ylabel("Count (log)")
ax.set_title("Image Intensity Distribution (log scale) — p1/p99 normalization bounds shown")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "intensity_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: intensity_distribution.png")

# --- Figure 4: Augmentation examples ---
train_loader, val_loader, test_loader, data_info = create_dataloaders(config.data, config.train)
train_idx = data_info["train_indices"]

ds_orig = GuidewireDataset(
    images[train_idx], positions[train_idx], directions[train_idx],
    config.data, augment=False,
)
ds_aug = GuidewireDataset(
    images[train_idx], positions[train_idx], directions[train_idx],
    config.data, augment=True,
)
# Use training set normalization for both
ds_orig.img_p1 = data_info["img_p1"]
ds_orig.img_p99 = data_info["img_p99"]
ds_aug.img_p1 = data_info["img_p1"]
ds_aug.img_p99 = data_info["img_p99"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Same Image: Top=Original, Bottom=Augmented", fontsize=14)
for i in range(4):
    img_orig, tgt_orig = ds_orig[i]
    img_aug, tgt_aug = ds_aug[i]

    axes[0, i].imshow(img_orig[0].numpy(), cmap="gray")
    # draw positions
    for w in range(2):
        px = tgt_orig["positions"][w, 0].item() * config.data.input_size
        py = tgt_orig["positions"][w, 1].item() * config.data.input_size
        axes[0, i].plot(px, py, "o", color="lime", markersize=5)
    axes[0, i].set_title(f"Original {i}")
    axes[0, i].axis("off")

    axes[1, i].imshow(img_aug[0].numpy(), cmap="gray")
    for w in range(2):
        px = tgt_aug["positions"][w, 0].item() * config.data.input_size
        py = tgt_aug["positions"][w, 1].item() * config.data.input_size
        axes[1, i].plot(px, py, "o", color="lime", markersize=5)
    axes[1, i].set_title(f"Augmented {i}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "augmentation_examples.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: augmentation_examples.png")


# ======================================================================
# STEP 2: Train Baseline (ResNet-18)
# ======================================================================
print("\n" + "=" * 70)
print("STEP 2: Training Baseline (ResNet-18)")
print("=" * 70)

config_baseline = get_config("baseline")
set_seed(config_baseline.seed)
train_loader, val_loader, test_loader, data_info = create_dataloaders(
    config_baseline.data, config_baseline.train
)

baseline_model, baseline_history = train_experiment(
    config_baseline, train_loader, val_loader
)

# Evaluate baseline on test (with TTA)
baseline_results = run_inference(
    baseline_model, test_loader, device,
    config_baseline.data.input_size, config_baseline.data.image_size,
    use_tta=False,
)
baseline_metrics = compute_all_metrics(baseline_results, config_baseline.eval)
print_metrics(baseline_metrics, "Baseline ResNet-18")
save_metrics(baseline_metrics, os.path.join(RESULTS_DIR, "baseline_metrics.json"))

# Training curves
plot_training_curves(
    baseline_history,
    save_path=os.path.join(FIGURES_DIR, "baseline_training_curves.png"),
)
plt.close("all")
plot_lr_curve(
    baseline_history,
    save_path=os.path.join(FIGURES_DIR, "baseline_lr_curve.png"),
)
plt.close("all")
print("  Saved: baseline_training_curves.png, baseline_lr_curve.png")

# Error distributions
plot_error_distributions(
    baseline_results,
    save_path=os.path.join(FIGURES_DIR, "baseline_error_dist.png"),
)
plt.close("all")
print("  Saved: baseline_error_dist.png")

# Qualitative
np.random.seed(42)
plot_qualitative_results(
    baseline_results, n_samples=9, arrow_length=50.0,
    save_path=os.path.join(FIGURES_DIR, "baseline_qualitative.png"),
)
plt.close("all")
print("  Saved: baseline_qualitative.png")

# Best / worst
plot_best_cases(
    baseline_results, n_best=6,
    save_path=os.path.join(FIGURES_DIR, "baseline_best_cases.png"),
)
plt.close("all")
plot_worst_cases(
    baseline_results, n_worst=6,
    save_path=os.path.join(FIGURES_DIR, "baseline_worst_cases.png"),
)
plt.close("all")
print("  Saved: baseline_best_cases.png, baseline_worst_cases.png")

# Scatter
plot_scatter_errors(
    baseline_results,
    save_path=os.path.join(FIGURES_DIR, "baseline_scatter.png"),
)
plt.close("all")
print("  Saved: baseline_scatter.png")


# ======================================================================
# STEP 3: Ablation Studies
# ======================================================================
print("\n" + "=" * 70)
print("STEP 3: Ablation Studies")
print("=" * 70)

all_metrics = {"ResNet-18 (Baseline)": baseline_metrics}
all_histories = {"ResNet-18 (Baseline)": baseline_history}

ablation_configs = [
    ("resnet34", "ResNet-34"),
    ("resnet50", "ResNet-50"),
    ("from_scratch", "No Pre-training"),
    ("no_augmentation", "No Augmentation"),
    ("heatmap", "Heatmap Model"),
]

for config_name, display_name in ablation_configs:
    print(f"\n--- Training: {display_name} ---")
    cfg = get_config(config_name)
    set_seed(cfg.seed)
    tl, vl, tel, _ = create_dataloaders(cfg.data, cfg.train)
    model_ab, history_ab = train_experiment(cfg, tl, vl)

    results_ab = run_inference(
        model_ab, tel, device, cfg.data.input_size, cfg.data.image_size,
        use_tta=False,
    )
    metrics_ab = compute_all_metrics(results_ab, cfg.eval)
    print_metrics(metrics_ab, display_name)
    save_metrics(metrics_ab, os.path.join(RESULTS_DIR, f"{cfg.experiment_name}_metrics.json"))

    all_metrics[display_name] = metrics_ab
    all_histories[display_name] = history_ab

    # Save training curves per ablation
    plot_training_curves(
        history_ab,
        save_path=os.path.join(FIGURES_DIR, f"{cfg.experiment_name}_training_curves.png"),
    )
    plt.close("all")


# ======================================================================
# STEP 4: Comparison Figures & Statistical Tests
# ======================================================================
print("\n" + "=" * 70)
print("STEP 4: Experiment Comparison & Statistical Tests")
print("=" * 70)

# Bar chart comparison
compare_experiments(
    all_metrics,
    save_path=os.path.join(FIGURES_DIR, "experiment_comparison.png"),
)
plt.close("all")
print("  Saved: experiment_comparison.png")

# Summary table
print(f"\n{'Experiment':<25} {'Pos Error (px)':<25} {'Ang Error (deg)':<25}")
print("-" * 75)
for name, m in all_metrics.items():
    agg = m["aggregate"]
    print(
        f"{name:<25} {agg['euclidean_mean']:>6.2f} "
        f"[{agg['euclidean_ci'][0]:.2f}, {agg['euclidean_ci'][1]:.2f}]  "
        f"{agg['angular_mean']:>10.2f} "
        f"[{agg['angular_ci'][0]:.2f}, {agg['angular_ci'][1]:.2f}]"
    )

# Statistical significance
print(f"\n{'=' * 60}")
print("Statistical Significance vs Baseline (Wilcoxon signed-rank)")
print("=" * 60)
significance_results = {}
for name, m in all_metrics.items():
    if name == "ResNet-18 (Baseline)":
        continue
    sig = compare_significance(baseline_metrics, m, name)
    significance_results[name] = sig
    print()


# ======================================================================
# STEP 5: Overfitting Check
# ======================================================================
print("\n" + "=" * 70)
print("STEP 5: Overfitting Check (Train / Val / Test)")
print("=" * 70)

print(f"{'Split':<15} {'Pos Error (px)':<25} {'Ang Error (deg)':<25}")
print("-" * 65)
for name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
    res = run_inference(
        baseline_model, loader, device,
        config_baseline.data.input_size, config_baseline.data.image_size,
        use_tta=False,
    )
    m = compute_all_metrics(res, config_baseline.eval)
    agg = m["aggregate"]
    print(
        f"  {name:<13} {agg['euclidean_mean']:>6.2f} "
        f"[{agg['euclidean_ci'][0]:.2f}, {agg['euclidean_ci'][1]:.2f}]  "
        f"{agg['angular_mean']:>10.2f} "
        f"[{agg['angular_ci'][0]:.2f}, {agg['angular_ci'][1]:.2f}]"
    )


# ======================================================================
# STEP 6: Save Final Summary
# ======================================================================
print("\n" + "=" * 70)
print("STEP 6: Saving Final Summary")
print("=" * 70)

summary = {
    "experiment": "Guidewire Pose Estimation - Final Results",
    "dataset": {
        "total_images": data_info["n_total"],
        "train": data_info["n_train"],
        "val": data_info["n_val"],
        "test": data_info["n_test"],
        "split_method": "block-based (block_size=10) to reduce data leakage",
        "normalization": f"percentile p1={data_info['img_p1']:.1f}, p99={data_info['img_p99']:.1f}",
    },
    "results": {
        name: {k: v for k, v in m["aggregate"].items()} for name, m in all_metrics.items()
    },
}

with open(os.path.join(RESULTS_DIR, "final_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Figures saved to: {FIGURES_DIR}")
print(f"Checkpoints saved to: {config.train.checkpoint_dir}")

# List all generated files
print("\n--- Generated Files ---")
for d in [FIGURES_DIR, RESULTS_DIR, config.train.checkpoint_dir]:
    files = sorted(os.listdir(d))
    if files:
        print(f"\n  {d}/")
        for f in files:
            size = os.path.getsize(os.path.join(d, f))
            print(f"    {f} ({size / 1024:.1f} KB)")

print("\n" + "=" * 70)
print("COMPLETE! All figures, models, and results have been generated.")
print("=" * 70)
