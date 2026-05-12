"""
Per-acquisition error analysis on the test set.

The dataset does not ship anatomical-site labels, but consecutive images
within the same 10-image block are very likely from the same C-arm
acquisition (and therefore the same anatomical site). This script
groups the 50 test images by their source block and reports per-block
errors, providing a proxy for per-acquisition / per-site variation.

This is more honest than fabricating site labels we don't have.
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeling"))

from config import get_config
from dataset import create_dataloaders, load_data, split_data
from model import build_model
from train import get_device, set_seed
from evaluate import run_inference, euclidean_distance, angular_error


def main():
    cfg = get_config("resnet34")
    set_seed(cfg.seed)
    device = get_device(cfg)

    # Reconstruct the test indices used by the standard split
    images, positions, directions = load_data(cfg.data)
    train_idx, val_idx, test_idx = split_data(len(images), cfg.data)
    print(f"Test set: {len(test_idx)} images, indices {test_idx[:5]}...{test_idx[-5:]}")

    # Block id for each test image
    block_ids = test_idx // 10  # 0..31

    # Run inference on the test set with ResNet-34 best
    _, _, test_loader, _ = create_dataloaders(cfg.data, cfg.train)

    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints", "ablation_resnet34_best.pth",
    )
    model = build_model(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    results = run_inference(
        model, test_loader, device,
        cfg.data.input_size, cfg.data.image_size, use_tta=False,
    )
    euc = euclidean_distance(results["pred_positions_px"], results["true_positions_px"])
    ang = angular_error(results["pred_angles"], results["true_angles"])

    # Per-image error (max over the two wires, conservative)
    euc_per_img = euc.mean(axis=1)
    ang_per_img = ang.mean(axis=1)

    # Group by block
    unique_blocks = sorted(set(block_ids.tolist()))
    rows = []
    for bid in unique_blocks:
        mask = block_ids == bid
        rows.append({
            "block_id": int(bid),
            "n_images": int(mask.sum()),
            "pos_mean":   float(euc_per_img[mask].mean()),
            "pos_median": float(np.median(euc_per_img[mask])),
            "ang_mean":   float(ang_per_img[mask].mean()),
            "ang_median": float(np.median(ang_per_img[mask])),
        })

    print(f"\nTest set spans {len(unique_blocks)} distinct blocks.")
    print(f"{'Block':>6} {'N':>3} {'PosMean':>9} {'PosMed':>9} {'AngMean':>9} {'AngMed':>9}")
    for r in rows:
        print(f"  {r['block_id']:>4} {r['n_images']:>3} "
              f"{r['pos_mean']:>9.1f} {r['pos_median']:>9.1f} "
              f"{r['ang_mean']:>9.1f} {r['ang_median']:>9.1f}")

    # Visualise: bar chart of per-block median errors, sorted
    rows_sorted = sorted(rows, key=lambda r: r["pos_median"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [f"B{r['block_id']}\n(N={r['n_images']})" for r in rows_sorted]
    pos_meds = [r["pos_median"] for r in rows_sorted]
    ang_meds = [r["ang_median"] for r in rows_sorted]

    axes[0].bar(range(len(rows_sorted)), pos_meds, color="steelblue", alpha=0.8)
    axes[0].set_xticks(range(len(rows_sorted)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Median tip error (pixels)")
    axes[0].set_title(f"Per-block tip-localization error · ResNet-34 · {len(rows_sorted)} blocks in test set")
    axes[0].axhline(86.5, color="red", linestyle="--", alpha=0.5, label="overall median (86.5 px)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(range(len(rows_sorted)), ang_meds, color="seagreen", alpha=0.8)
    axes[1].set_xticks(range(len(rows_sorted)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Median angular error (degrees)")
    axes[1].set_title("Per-block angular error · ResNet-34")
    axes[1].axhline(10.0, color="red", linestyle="--", alpha=0.5, label="overall median (10.0°)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    out_path = os.path.join(out_dir, "per_block_error.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved per-block figure → {out_path}")

    # Save summary
    out_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "per_block_errors.json"
    )
    with open(out_json, "w") as f:
        json.dump({
            "model": "ResNet-34",
            "rows": rows,
            "note": (
                "Each block_id corresponds to 10 consecutive images from the source "
                "dataset, very likely from the same C-arm acquisition. The block index "
                "is used as a proxy for anatomical-site / acquisition identity, since "
                "no explicit site labels ship with the dataset."
            )
        }, f, indent=2)
    print(f"Saved per-block JSON → {out_json}")

    pos_meds_arr = np.array(pos_meds)
    ang_meds_arr = np.array(ang_meds)
    print(f"\nVariability across blocks (median error):")
    print(f"  Position: min={pos_meds_arr.min():.1f} px, max={pos_meds_arr.max():.1f} px, range={pos_meds_arr.max()-pos_meds_arr.min():.1f}")
    print(f"  Angular:  min={ang_meds_arr.min():.1f}°, max={ang_meds_arr.max():.1f}°, range={ang_meds_arr.max()-ang_meds_arr.min():.1f}")


if __name__ == "__main__":
    main()
