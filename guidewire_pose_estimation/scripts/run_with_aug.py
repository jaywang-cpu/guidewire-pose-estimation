"""
Run the with-augmentation ablation:
  - ResNet-18, augment_train=True, post-aug x-resort active.
  - Trains, evaluates on the same test set, saves metrics + history.
This is the "real" no-aug ablation comparison (the previous `no_augmentation`
config was identical to the baseline because the final baseline already had
augment_train=False).
"""
import os
import sys
import json

# Resolve the package directory (parent of this scripts/ folder)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "modeling"))
from config import get_config
from dataset import create_dataloaders
from train import train_experiment, get_device, set_seed
from evaluate import run_inference, compute_all_metrics, print_metrics


def main():
    cfg = get_config("with_augmentation")
    set_seed(cfg.seed)
    print(f"Experiment: {cfg.experiment_name}")
    print(f"augment_train: {cfg.data.augment_train}")
    print(f"backbone: {cfg.model.backbone}, pretrained: {cfg.model.pretrained}")

    train_loader, val_loader, test_loader, info = create_dataloaders(
        cfg.data, cfg.train
    )
    print(f"Split: train={info['n_train']}, val={info['n_val']}, test={info['n_test']}")

    device = get_device(cfg)
    print(f"Device: {device}")

    model, history = train_experiment(cfg, train_loader, val_loader)

    results = run_inference(
        model, test_loader, device,
        cfg.data.input_size, cfg.data.image_size,
        use_tta=False,
    )
    metrics = compute_all_metrics(results, cfg.eval)
    print_metrics(metrics, "ResNet-18 + augmentation")

    os.makedirs(cfg.eval.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.eval.results_dir, "ablation_with_aug_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved metrics → {out_path}")


if __name__ == "__main__":
    main()
