"""
Generate cumulative-distribution-function (CDF) plots of per-sample
tip-localization error and per-sample angular error, with one line per
experiment. Complements the bar chart in figures/experiment_comparison.png
by showing the full error distribution rather than just the mean.

Saves figures/error_cdf.png and a per-sample-errors JSON for downstream use.
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# Resolve the package directory (parent of this scripts/ folder)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "modeling"))
from config import get_config
from dataset import create_dataloaders
from model import build_model
from train import get_device, set_seed
from evaluate import run_inference, euclidean_distance, angular_error


EXPERIMENTS = [
    # (display name, config name, checkpoint filename)
    ("ResNet-34", "resnet34", "ablation_resnet34_best.pth"),
    ("Heatmap (ResNet-18)", "heatmap", "heatmap_resnet18_best.pth"),
    ("ResNet-50", "resnet50", "ablation_resnet50_best.pth"),
    ("ResNet-18 (baseline)", "baseline", "baseline_resnet18_best.pth"),
    ("No pre-training", "from_scratch", "ablation_no_pretrain_best.pth"),
]

# Add with-augmentation if its checkpoint has finished training
WITH_AUG = ("ResNet-18 + augmentation", "with_augmentation", "ablation_with_aug_best.pth")


def per_sample_errors(model, loader, device, cfg):
    """Run inference and return (euc, ang) per-sample arrays, flattened over wires."""
    results = run_inference(
        model, loader, device,
        cfg.data.input_size, cfg.data.image_size, use_tta=False,
    )
    euc = euclidean_distance(results["pred_positions_px"], results["true_positions_px"])
    ang = angular_error(results["pred_angles"], results["true_angles"])
    return euc.flatten(), ang.flatten()


def main():
    device = None
    out_dir = os.path.join(PKG_DIR, "figures")
    results_dir = os.path.join(PKG_DIR, "results")
    ckpt_dir = os.path.join(PKG_DIR, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    experiments = list(EXPERIMENTS)
    if os.path.exists(os.path.join(ckpt_dir, WITH_AUG[2])):
        experiments.append(WITH_AUG)

    all_errors = {}
    for display_name, cfg_name, ckpt_name in experiments:
        cfg = get_config(cfg_name)
        set_seed(cfg.seed)
        if device is None:
            device = get_device(cfg)
            print(f"Device: {device}")

        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  [skip] {display_name}: checkpoint not found at {ckpt_path}")
            continue

        _, _, test_loader, _ = create_dataloaders(cfg.data, cfg.train)
        model = build_model(cfg.model).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        model.load_state_dict(state)
        model.eval()

        euc, ang = per_sample_errors(model, test_loader, device, cfg)
        all_errors[display_name] = {"euclidean": euc.tolist(), "angular": ang.tolist()}
        print(f"  {display_name}: N={len(euc)}, median pos={np.median(euc):.1f} px, median ang={np.median(ang):.1f}°")

    # Save raw per-sample errors for downstream use
    with open(os.path.join(results_dir, "per_sample_errors.json"), "w") as f:
        json.dump(all_errors, f, indent=2)

    # Plot CDFs side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_errors)))
    for color, (name, errs) in zip(colors, all_errors.items()):
        for ax_idx, key, label, unit in [
            (0, "euclidean", "Tip-localization error", "px"),
            (1, "angular",   "Angular error",          "°"),
        ]:
            vals = np.sort(np.array(errs[key]))
            ys = np.arange(1, len(vals) + 1) / len(vals)
            axes[ax_idx].plot(vals, ys, label=name, color=color, lw=2)

    for ax, title, xlabel in [
        (axes[0], "CDF of per-sample tip-localization error", "Euclidean error (pixels)"),
        (axes[1], "CDF of per-sample angular error", "Angular error (degrees)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fraction of wire predictions")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, 1)

    # Cap the x-axis at p99 for readability (long-tail will still be in the data)
    all_euc = np.concatenate([np.array(e["euclidean"]) for e in all_errors.values()])
    all_ang = np.concatenate([np.array(e["angular"]) for e in all_errors.values()])
    axes[0].set_xlim(0, float(np.percentile(all_euc, 99)))
    axes[1].set_xlim(0, float(np.percentile(all_ang, 99)))

    plt.tight_layout()
    out_path = os.path.join(out_dir, "error_cdf.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved CDF figure → {out_path}")


if __name__ == "__main__":
    main()
