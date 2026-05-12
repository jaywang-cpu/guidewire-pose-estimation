"""
Evaluation module for Guidewire Pose Estimation.

Includes:
  - Quantitative metrics: Euclidean distance, angular error, with 95% bootstrap CI
  - Statistical significance tests (Wilcoxon signed-rank)
  - Qualitative visualization: overlay predictions on images
  - Training curve plots with real-metric tracking
  - Error distribution analysis
  - Failure case identification and categorization
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from config import ExperimentConfig, EvalConfig


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def euclidean_distance(pred_pos: np.ndarray, true_pos: np.ndarray) -> np.ndarray:
    """
    Euclidean distance between predicted and true positions.
    Input: (N, 2, 2)  Output: (N, 2) distances per guidewire.
    """
    return np.sqrt(np.sum((pred_pos - true_pos) ** 2, axis=-1))


def angular_error(pred_angles: np.ndarray, true_angles: np.ndarray) -> np.ndarray:
    """
    Absolute angular error in degrees, wrapped correctly.
    Input: (N, 2) radians  Output: (N, 2) degrees.
    """
    diff = pred_angles - true_angles
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(diff) * 180.0 / np.pi


def bootstrap_confidence_interval(
    values: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap CI. Returns: (mean, ci_lower, ci_upper)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([
        np.mean(values[rng.randint(0, n, size=n)]) for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci_level
    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test comparing two sets of per-sample errors.
    Returns statistic, p-value, and effect size (rank-biserial r).
    """
    stat, p_value = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
    n = len(errors_a)
    # Rank-biserial correlation as effect size
    r = 1 - (2 * stat) / (n * (n + 1) / 2)
    return {"statistic": float(stat), "p_value": float(p_value), "effect_size_r": float(r)}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _match_predictions(
    pred_pos: np.ndarray,
    pred_angles: np.ndarray,
    true_pos: np.ndarray,
    true_angles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Hungarian matching to align predictions with ground truth.
    For K=2 guidewires, compare both assignments and pick the lower-cost one.

    This fixes the critical bug where evaluation metrics were computed without
    matching, causing inflated errors when the model predicts wires in swapped order.

    Returns matched (pred_pos, pred_angles) reordered to align with ground truth.
    """
    N = pred_pos.shape[0]
    matched_pred_pos = pred_pos.copy()
    matched_pred_angles = pred_angles.copy()

    for i in range(N):
        # Cost of identity assignment: pred[0]->gt[0], pred[1]->gt[1]
        cost_id = (np.sum((pred_pos[i, 0] - true_pos[i, 0]) ** 2) +
                   np.sum((pred_pos[i, 1] - true_pos[i, 1]) ** 2))
        # Cost of swapped assignment: pred[0]->gt[1], pred[1]->gt[0]
        cost_sw = (np.sum((pred_pos[i, 0] - true_pos[i, 1]) ** 2) +
                   np.sum((pred_pos[i, 1] - true_pos[i, 0]) ** 2))
        if cost_sw < cost_id:
            matched_pred_pos[i] = pred_pos[i, [1, 0]]
            matched_pred_angles[i] = pred_angles[i, [1, 0]]

    return matched_pred_pos, matched_pred_angles


def _tta_forward(
    model: nn.Module, images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Test-time augmentation: average predictions over the original and
    horizontally flipped image. Flip position x-coords and direction sin-component
    back before averaging.
    """
    # Original prediction
    pred_orig = model(images)

    # Horizontal flip prediction
    images_flip = torch.flip(images, dims=[-1])
    pred_flip = model(images_flip)

    # Un-flip position: x -> 1 - x (positions are in [0, 1])
    pos_flip = pred_flip["positions"].clone()
    pos_flip[:, :, 0] = 1.0 - pos_flip[:, :, 0]

    # Un-flip direction: angle mirrors across y-axis -> sin(pi - a) = sin(a), cos(pi - a) = -cos(a)
    # Since direction = (sin theta, cos theta), after h-flip: sin stays, cos negates
    dir_flip = pred_flip["directions"].clone()
    dir_flip[:, :, 1] = -dir_flip[:, :, 1]

    # Average
    avg_pos = (pred_orig["positions"] + pos_flip) / 2.0
    avg_dir = F.normalize(pred_orig["directions"] + dir_flip, p=2, dim=-1)

    result = {"positions": avg_pos, "directions": avg_dir}
    if "heatmaps" in pred_orig:
        result["heatmaps"] = pred_orig["heatmaps"]
    return result


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_size: int = 512,
    orig_size: int = 976,
    use_tta: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run model inference on a data loader.
    Returns dict with pred/true positions (original px), pred/true angles, images.
    Predictions are reordered via Hungarian matching to align with ground truth.
    """
    model.eval()
    all_pred_pos, all_true_pos = [], []
    all_pred_angles, all_true_angles = [], []
    all_images = []

    for images, targets in loader:
        images_gpu = images.to(device)

        if use_tta:
            predictions = _tta_forward(model, images_gpu)
        else:
            predictions = model(images_gpu)

        pred_pos = predictions["positions"].cpu().numpy() * orig_size
        pred_dir = predictions["directions"].cpu().numpy()
        pred_angles = np.arctan2(pred_dir[:, :, 0], pred_dir[:, :, 1])

        true_pos_px = targets["positions_px"].numpy()
        true_angles = targets["angles"].numpy()

        # Hungarian matching: reorder predictions to best align with ground truth
        pred_pos, pred_angles = _match_predictions(
            pred_pos, pred_angles, true_pos_px, true_angles
        )

        all_pred_pos.append(pred_pos)
        all_true_pos.append(true_pos_px)
        all_pred_angles.append(pred_angles)
        all_true_angles.append(true_angles)
        all_images.append(images[:, 0].numpy())

    return {
        "pred_positions_px": np.concatenate(all_pred_pos),
        "true_positions_px": np.concatenate(all_true_pos),
        "pred_angles": np.concatenate(all_pred_angles),
        "true_angles": np.concatenate(all_true_angles),
        "images": np.concatenate(all_images),
    }


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: Dict[str, np.ndarray],
    config: EvalConfig,
) -> Dict:
    """Compute all evaluation metrics with confidence intervals."""
    euc_dist = euclidean_distance(
        results["pred_positions_px"], results["true_positions_px"]
    )
    ang_err = angular_error(results["pred_angles"], results["true_angles"])

    euc_flat = euc_dist.flatten()
    ang_flat = ang_err.flatten()

    metrics = {}
    for wire_idx in range(2):
        w = f"wire{wire_idx}"
        euc_w, ang_w = euc_dist[:, wire_idx], ang_err[:, wire_idx]

        mean_euc, ci_lo_euc, ci_hi_euc = bootstrap_confidence_interval(
            euc_w, config.ci_n_bootstrap, config.ci_level
        )
        mean_ang, ci_lo_ang, ci_hi_ang = bootstrap_confidence_interval(
            ang_w, config.ci_n_bootstrap, config.ci_level
        )

        metrics[w] = {
            "euclidean_mean": mean_euc,
            "euclidean_ci": [ci_lo_euc, ci_hi_euc],
            "euclidean_median": float(np.median(euc_w)),
            "euclidean_std": float(np.std(euc_w)),
            "euclidean_p90": float(np.percentile(euc_w, 90)),
            "angular_mean": mean_ang,
            "angular_ci": [ci_lo_ang, ci_hi_ang],
            "angular_median": float(np.median(ang_w)),
            "angular_std": float(np.std(ang_w)),
            "angular_p90": float(np.percentile(ang_w, 90)),
        }

    mean_euc, ci_lo_euc, ci_hi_euc = bootstrap_confidence_interval(
        euc_flat, config.ci_n_bootstrap, config.ci_level
    )
    mean_ang, ci_lo_ang, ci_hi_ang = bootstrap_confidence_interval(
        ang_flat, config.ci_n_bootstrap, config.ci_level
    )

    metrics["aggregate"] = {
        "euclidean_mean": mean_euc,
        "euclidean_ci": [ci_lo_euc, ci_hi_euc],
        "euclidean_median": float(np.median(euc_flat)),
        "euclidean_std": float(np.std(euc_flat)),
        "euclidean_p90": float(np.percentile(euc_flat, 90)),
        "angular_mean": mean_ang,
        "angular_ci": [ci_lo_ang, ci_hi_ang],
        "angular_median": float(np.median(ang_flat)),
        "angular_std": float(np.std(ang_flat)),
        "angular_p90": float(np.percentile(ang_flat, 90)),
        "n_samples": len(euc_dist),
    }

    # Store raw errors for significance testing
    metrics["_raw_euc"] = euc_flat
    metrics["_raw_ang"] = ang_flat

    return metrics


def print_metrics(metrics: Dict, experiment_name: str = ""):
    print(f"\n{'='*70}")
    if experiment_name:
        print(f"  Results: {experiment_name}")
    print(f"{'='*70}")

    agg = metrics["aggregate"]
    print(f"\n  Aggregate (N={agg['n_samples']} images, {agg['n_samples']*2} guidewires):")
    print(f"    Position Error (px):  {agg['euclidean_mean']:.2f} "
          f"[{agg['euclidean_ci'][0]:.2f}, {agg['euclidean_ci'][1]:.2f}] "
          f"(median={agg['euclidean_median']:.2f}, p90={agg['euclidean_p90']:.2f})")
    print(f"    Angular Error (deg):  {agg['angular_mean']:.2f} "
          f"[{agg['angular_ci'][0]:.2f}, {agg['angular_ci'][1]:.2f}] "
          f"(median={agg['angular_median']:.2f}, p90={agg['angular_p90']:.2f})")

    for wire_idx in range(2):
        w = metrics[f"wire{wire_idx}"]
        print(f"\n  Guidewire {wire_idx + 1}:")
        print(f"    Position Error (px):  {w['euclidean_mean']:.2f} "
              f"[{w['euclidean_ci'][0]:.2f}, {w['euclidean_ci'][1]:.2f}]")
        print(f"    Angular Error (deg):  {w['angular_mean']:.2f} "
              f"[{w['angular_ci'][0]:.2f}, {w['angular_ci'][1]:.2f}]")

    print(f"{'='*70}\n")


def compare_significance(
    baseline_metrics: Dict, other_metrics: Dict, other_name: str
) -> Dict:
    """
    Run Wilcoxon signed-rank test comparing baseline vs another experiment.
    Returns dict with test results for position and angular errors.
    """
    n = min(len(baseline_metrics["_raw_euc"]), len(other_metrics["_raw_euc"]))

    euc_test = wilcoxon_test(
        baseline_metrics["_raw_euc"][:n], other_metrics["_raw_euc"][:n]
    )
    ang_test = wilcoxon_test(
        baseline_metrics["_raw_ang"][:n], other_metrics["_raw_ang"][:n]
    )

    sig_euc = "YES" if euc_test["p_value"] < 0.05 else "no"
    sig_ang = "YES" if ang_test["p_value"] < 0.05 else "no"

    print(f"  vs {other_name}:")
    print(f"    Position: p={euc_test['p_value']:.4f} (sig={sig_euc}), effect r={euc_test['effect_size_r']:.3f}")
    print(f"    Angular:  p={ang_test['p_value']:.4f} (sig={sig_ang}), effect r={ang_test['effect_size_r']:.3f}")

    return {"position": euc_test, "angular": ang_test}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """Plot training/validation loss curves AND real validation metrics."""
    has_real_metrics = "val_euc_px" in history

    n_cols = 4 if has_real_metrics else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Total loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Position loss
    axes[1].plot(epochs, history["train_pos_loss"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_pos_loss"], label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Position Loss")
    axes[1].set_title("Position Loss (Smooth L1)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Direction loss
    axes[2].plot(epochs, history["train_dir_loss"], label="Train", linewidth=2)
    axes[2].plot(epochs, history["val_dir_loss"], label="Val", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Direction Loss")
    axes[2].set_title("Direction Loss (Cosine)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Real validation metrics
    if has_real_metrics:
        ax2 = axes[3]
        color1, color2 = "#2196F3", "#FF9800"
        ax2.plot(epochs, history["val_euc_px"], color=color1, linewidth=2, label="Pos (px)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Position Error (px)", color=color1)
        ax2.tick_params(axis="y", labelcolor=color1)

        ax2b = ax2.twinx()
        ax2b.plot(epochs, history["val_ang_deg"], color=color2, linewidth=2, label="Ang (\u00b0)")
        ax2b.set_ylabel("Angular Error (\u00b0)", color=color2)
        ax2b.tick_params(axis="y", labelcolor=color2)

        ax2.set_title("Validation Metrics (real units)")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_lr_curve(history: Dict[str, List[float]], save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["lr"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_error_distributions(
    results: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
):
    euc_dist = euclidean_distance(
        results["pred_positions_px"], results["true_positions_px"]
    ).flatten()
    ang_err = angular_error(
        results["pred_angles"], results["true_angles"]
    ).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(euc_dist, bins=30, edgecolor="black", alpha=0.7, color="#2196F3")
    axes[0].axvline(np.mean(euc_dist), color="red", linestyle="--",
                    label=f"Mean={np.mean(euc_dist):.1f} px")
    axes[0].axvline(np.median(euc_dist), color="green", linestyle="--",
                    label=f"Median={np.median(euc_dist):.1f} px")
    axes[0].set_xlabel("Euclidean Distance (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Position Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(ang_err, bins=30, edgecolor="black", alpha=0.7, color="#FF9800")
    axes[1].axvline(np.mean(ang_err), color="red", linestyle="--",
                    label=f"Mean={np.mean(ang_err):.1f}\u00b0")
    axes[1].axvline(np.median(ang_err), color="green", linestyle="--",
                    label=f"Median={np.median(ang_err):.1f}\u00b0")
    axes[1].set_xlabel("Angular Error (degrees)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Angular Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_qualitative_results(
    results: Dict[str, np.ndarray],
    indices: Optional[List[int]] = None,
    n_samples: int = 6,
    arrow_length: float = 50.0,
    save_path: Optional[str] = None,
):
    """Overlay predicted and true guidewire poses. Green=GT, Red=Pred."""
    N = len(results["images"])
    if indices is None:
        indices = np.random.choice(N, min(n_samples, N), replace=False)
        indices = sorted(indices)

    n = len(indices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, sample_idx in enumerate(indices):
        ax = axes[ax_idx]
        ax.imshow(results["images"][sample_idx], cmap="gray")

        for wire in range(2):
            tx, ty = results["true_positions_px"][sample_idx, wire]
            ta = results["true_angles"][sample_idx, wire]
            ax.plot(tx, ty, "go", markersize=8, label="GT" if wire == 0 else "")
            ax.arrow(tx, ty, arrow_length * np.cos(ta), arrow_length * np.sin(ta),
                     head_width=8, head_length=5, fc="green", ec="green", linewidth=2)

            px, py = results["pred_positions_px"][sample_idx, wire]
            pa = results["pred_angles"][sample_idx, wire]
            ax.plot(px, py, "rs", markersize=8, label="Pred" if wire == 0 else "")
            ax.arrow(px, py, arrow_length * np.cos(pa), arrow_length * np.sin(pa),
                     head_width=8, head_length=5, fc="red", ec="red", linewidth=2)

        euc = euclidean_distance(
            results["pred_positions_px"][sample_idx:sample_idx+1],
            results["true_positions_px"][sample_idx:sample_idx+1],
        )
        ang = angular_error(
            results["pred_angles"][sample_idx:sample_idx+1],
            results["true_angles"][sample_idx:sample_idx+1],
        )
        ax.set_title(
            f"Sample {sample_idx}\n"
            f"Pos: {euc[0,0]:.1f}, {euc[0,1]:.1f} px | "
            f"Ang: {ang[0,0]:.1f}\u00b0, {ang[0,1]:.1f}\u00b0",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.axis("off")

    for ax_idx in range(len(indices), len(axes)):
        axes[ax_idx].axis("off")

    plt.suptitle("Qualitative Results (Green=GT, Red=Pred)", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_worst_cases(results, n_worst=6, arrow_length=50.0, save_path=None):
    euc_dist = euclidean_distance(
        results["pred_positions_px"], results["true_positions_px"]
    )
    max_err = euc_dist.max(axis=1)
    worst_indices = np.argsort(max_err)[-n_worst:][::-1]

    print("Worst cases (highest position error):")
    for i, idx in enumerate(worst_indices):
        print(f"  #{i+1}: Sample {idx}, max err = {max_err[idx]:.1f} px")

    plot_qualitative_results(
        results, indices=list(worst_indices),
        arrow_length=arrow_length, save_path=save_path,
    )


def plot_best_cases(results, n_best=6, arrow_length=50.0, save_path=None):
    euc_dist = euclidean_distance(
        results["pred_positions_px"], results["true_positions_px"]
    )
    max_err = euc_dist.max(axis=1)
    best_indices = np.argsort(max_err)[:n_best]

    print("Best cases (lowest position error):")
    for i, idx in enumerate(best_indices):
        print(f"  #{i+1}: Sample {idx}, max err = {max_err[idx]:.1f} px")

    plot_qualitative_results(
        results, indices=list(best_indices),
        arrow_length=arrow_length, save_path=save_path,
    )


def plot_scatter_errors(results, save_path=None):
    euc_dist = euclidean_distance(
        results["pred_positions_px"], results["true_positions_px"]
    ).flatten()
    ang_err = angular_error(
        results["pred_angles"], results["true_angles"]
    ).flatten()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(euc_dist, ang_err, alpha=0.5, s=30, edgecolors="none")
    ax.set_xlabel("Position Error (pixels)")
    ax.set_ylabel("Angular Error (degrees)")
    ax.set_title("Position Error vs Angular Error")
    ax.grid(True, alpha=0.3)

    r, p = stats.pearsonr(euc_dist, ang_err)
    ax.text(0.05, 0.95, f"Pearson r={r:.3f}, p={p:.2e}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_experiments(all_metrics, save_path=None):
    """Bar chart comparing multiple experiments with 95% CI error bars."""
    names = list(all_metrics.keys())
    euc_means = [all_metrics[n]["aggregate"]["euclidean_mean"] for n in names]
    euc_cis = [
        (all_metrics[n]["aggregate"]["euclidean_mean"] - all_metrics[n]["aggregate"]["euclidean_ci"][0],
         all_metrics[n]["aggregate"]["euclidean_ci"][1] - all_metrics[n]["aggregate"]["euclidean_mean"])
        for n in names
    ]
    ang_means = [all_metrics[n]["aggregate"]["angular_mean"] for n in names]
    ang_cis = [
        (all_metrics[n]["aggregate"]["angular_mean"] - all_metrics[n]["aggregate"]["angular_ci"][0],
         all_metrics[n]["aggregate"]["angular_ci"][1] - all_metrics[n]["aggregate"]["angular_mean"])
        for n in names
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(names))
    axes[0].bar(x, euc_means, yerr=np.array(euc_cis).T, capsize=5,
                color="#2196F3", alpha=0.8, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("Euclidean Distance (px)")
    axes[0].set_title("Position Error Comparison (95% CI)")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, ang_means, yerr=np.array(ang_cis).T, capsize=5,
                color="#FF9800", alpha=0.8, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("Angular Error (degrees)")
    axes[1].set_title("Angular Error Comparison (95% CI)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


_METRIC_FIELDS = [
    "experiment", "scope", "n_samples",
    "euc_mean", "euc_ci_lo", "euc_ci_hi", "euc_median", "euc_std", "euc_p90",
    "ang_mean", "ang_ci_lo", "ang_ci_hi", "ang_median", "ang_std", "ang_p90",
]


def _metrics_to_rows(metrics: Dict, experiment: str) -> list:
    """Flatten compute_all_metrics output to 3 rows (wire0, wire1, aggregate)."""
    import csv  # noqa: F401
    rows = []
    for scope in ("wire0", "wire1", "aggregate"):
        if scope not in metrics:
            continue
        v = metrics[scope]
        ci_e = v.get("euclidean_ci") or [None, None]
        ci_a = v.get("angular_ci") or [None, None]
        rows.append({
            "experiment":  experiment,
            "scope":       scope,
            "n_samples":   v.get("n_samples", ""),
            "euc_mean":    round(float(v["euclidean_mean"]), 4),
            "euc_ci_lo":   round(float(ci_e[0]), 4),
            "euc_ci_hi":   round(float(ci_e[1]), 4),
            "euc_median":  round(float(v["euclidean_median"]), 4),
            "euc_std":     round(float(v["euclidean_std"]), 4),
            "euc_p90":     round(float(v["euclidean_p90"]), 4),
            "ang_mean":    round(float(v["angular_mean"]), 4),
            "ang_ci_lo":   round(float(ci_a[0]), 4),
            "ang_ci_hi":   round(float(ci_a[1]), 4),
            "ang_median":  round(float(v["angular_median"]), 4),
            "ang_std":     round(float(v["angular_std"]), 4),
            "ang_p90":     round(float(v["angular_p90"]), 4),
        })
    return rows


def save_metrics(metrics: Dict, path: str, experiment: str = None):
    """Save per-experiment metrics to CSV (3 rows: wire0, wire1, aggregate).

    `path` may carry either a .csv or .json extension; the CSV file is always
    what gets written (the .json extension is silently coerced to .csv for
    backwards-compat with older callers).
    """
    import csv
    if path.endswith(".json"):
        path = path[:-5] + ".csv"
    if experiment is None:
        # Recover experiment name from the filename, e.g. ablation_resnet34_metrics.csv
        experiment = os.path.splitext(os.path.basename(path))[0].replace("_metrics", "")
    rows = _metrics_to_rows(metrics, experiment)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METRIC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def generate_full_report(
    model, loader, device, config, history=None, use_tta=True
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Generate complete evaluation: metrics + all plots."""
    os.makedirs(config.eval.figures_dir, exist_ok=True)
    os.makedirs(config.eval.results_dir, exist_ok=True)

    results = run_inference(
        model, loader, device,
        input_size=config.data.input_size,
        orig_size=config.data.image_size,
        use_tta=use_tta,
    )

    metrics = compute_all_metrics(results, config.eval)
    print_metrics(metrics, config.experiment_name)

    save_metrics(metrics, os.path.join(
        config.eval.results_dir, f"{config.experiment_name}_metrics.csv"
    ), experiment=config.experiment_name)

    prefix = os.path.join(config.eval.figures_dir, config.experiment_name)

    if history:
        plot_training_curves(history, save_path=f"{prefix}_training_curves.png")
        plot_lr_curve(history, save_path=f"{prefix}_lr_curve.png")

    plot_error_distributions(results, save_path=f"{prefix}_error_dist.png")
    plot_scatter_errors(results, save_path=f"{prefix}_scatter.png")
    plot_qualitative_results(
        results, n_samples=config.eval.num_qualitative_samples,
        arrow_length=config.eval.arrow_length,
        save_path=f"{prefix}_qualitative.png",
    )
    plot_best_cases(results, n_best=6, save_path=f"{prefix}_best_cases.png")
    plot_worst_cases(results, n_worst=6, save_path=f"{prefix}_worst_cases.png")

    return metrics, results
