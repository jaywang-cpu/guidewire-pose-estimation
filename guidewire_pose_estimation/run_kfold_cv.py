"""
5-fold cross-validation for the best model (ResNet-34).

Splits the 314 images into k = 5 contiguous-block folds. For each fold,
that fold is the test set, the next fold (mod k) is the validation set,
and the remaining three folds are the training set. Trains ResNet-34
with the standard configuration and saves per-fold metrics. Aggregates
across folds at the end.

Rationale: the single-split test set has only 50 images, leading to wide
bootstrap confidence intervals. K-fold lets us evaluate on the full 314-image
dataset (each image appears in exactly one test fold), substantially
tightening the estimate while preserving the block-based leakage-avoidance.
"""
import os
import sys
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modeling"))

from config import get_config
from dataset import GuidewireDataset, load_data
from train import train_experiment, get_device, set_seed
from evaluate import run_inference, compute_all_metrics


def kfold_block_indices(n_samples: int, k: int, fold_idx: int, block_size: int = 10, seed: int = 42):
    """
    Block-based k-fold split. Same blocks as the single split (block_size=10) but
    partitioned into k roughly equal groups. Returns (train_idx, val_idx, test_idx).
    """
    rng = np.random.RandomState(seed)
    n_blocks = (n_samples + block_size - 1) // block_size
    block_order = rng.permutation(n_blocks)

    fold_blocks = np.array_split(block_order, k)
    test_blocks = fold_blocks[fold_idx]
    val_blocks = fold_blocks[(fold_idx + 1) % k]
    train_blocks = np.concatenate(
        [fold_blocks[i] for i in range(k) if i != fold_idx and i != (fold_idx + 1) % k]
    )

    def blocks_to_indices(blocks):
        out = []
        for b in blocks:
            start = int(b) * block_size
            end = min(start + block_size, n_samples)
            out.extend(range(start, end))
        return np.array(sorted(out))

    return blocks_to_indices(train_blocks), blocks_to_indices(val_blocks), blocks_to_indices(test_blocks)


def make_loaders(cfg, images, positions, directions, train_idx, val_idx, test_idx):
    ds_train = GuidewireDataset(
        images[train_idx], positions[train_idx], directions[train_idx],
        cfg.data, augment=cfg.data.augment_train,
    )
    ds_val = GuidewireDataset(
        images[val_idx], positions[val_idx], directions[val_idx],
        cfg.data, augment=False,
    )
    ds_test = GuidewireDataset(
        images[test_idx], positions[test_idx], directions[test_idx],
        cfg.data, augment=False,
    )
    common = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    return (
        DataLoader(ds_train, shuffle=True, **common),
        DataLoader(ds_val, shuffle=False, **common),
        DataLoader(ds_test, shuffle=False, **common),
    )


def main():
    K = 5
    cfg = get_config("resnet34")
    cfg.train.checkpoint_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "checkpoints", "kfold"
    )
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    device = get_device(cfg)
    images, positions, directions = load_data(cfg.data)
    n = len(images)
    print(f"5-fold CV on ResNet-34 over {n} images. Device: {device}")
    print(f"Checkpoints → {cfg.train.checkpoint_dir}")

    fold_metrics = []
    per_fold_test_idx = []

    t_global = time.time()
    for fold in range(K):
        t_fold = time.time()
        train_idx, val_idx, test_idx = kfold_block_indices(n, K, fold)
        print(f"\n=== Fold {fold+1}/{K}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} ===")

        cfg.experiment_name = f"kfold_resnet34_fold{fold}"
        set_seed(cfg.seed)

        train_loader, val_loader, test_loader = make_loaders(
            cfg, images, positions, directions, train_idx, val_idx, test_idx
        )
        model, history = train_experiment(cfg, train_loader, val_loader)

        results = run_inference(
            model, test_loader, device,
            cfg.data.input_size, cfg.data.image_size, use_tta=False,
        )
        metrics = compute_all_metrics(results, cfg.eval)
        agg = metrics["aggregate"]
        agg["n_test_images"] = int(len(test_idx))
        fold_metrics.append(agg)
        per_fold_test_idx.append(test_idx.tolist())

        print(f"  Fold {fold}: pos mean={agg['euclidean_mean']:.2f} px, "
              f"ang mean={agg['angular_mean']:.2f}°  (took {time.time()-t_fold:.0f} s)")

    # Aggregate across folds — mean and std of per-fold means
    pos_means = np.array([m["euclidean_mean"] for m in fold_metrics])
    pos_medians = np.array([m["euclidean_median"] for m in fold_metrics])
    ang_means = np.array([m["angular_mean"] for m in fold_metrics])
    ang_medians = np.array([m["angular_median"] for m in fold_metrics])

    summary = {
        "model": "ResNet-34",
        "k": K,
        "split_method": f"block-based block_size=10, k={K} folds",
        "per_fold_metrics": fold_metrics,
        "per_fold_test_indices": per_fold_test_idx,
        "aggregate_across_folds": {
            "pos_mean_of_means": float(pos_means.mean()),
            "pos_std_of_means":  float(pos_means.std(ddof=1)),
            "pos_min_of_means":  float(pos_means.min()),
            "pos_max_of_means":  float(pos_means.max()),
            "pos_median_of_medians": float(np.median(pos_medians)),
            "ang_mean_of_means": float(ang_means.mean()),
            "ang_std_of_means":  float(ang_means.std(ddof=1)),
            "ang_median_of_medians": float(np.median(ang_medians)),
        },
        "total_runtime_sec": time.time() - t_global,
    }

    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "kfold_summary.json"
    )
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved k-fold summary → {out}")
    print(f"Across {K} folds: pos = {pos_means.mean():.1f} ± {pos_means.std(ddof=1):.1f} px, "
          f"ang = {ang_means.mean():.2f} ± {ang_means.std(ddof=1):.2f}°")
    print(f"Total runtime: {(time.time()-t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
