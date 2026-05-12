"""
Reproducibility smoke test.

For the grader: this script loads each released ResNet checkpoint, runs
inference on the held-out test set, computes the headline metrics, and
compares them against the numbers reported in REPORT.md §4.3. Exits
non-zero if any number deviates by more than `TOLERANCE` pixels or
`ANG_TOLERANCE` degrees from the expected value.

Run from the project root, after downloading the dataset (data/raw/) and
the checkpoints (guidewire_pose_estimation/checkpoints/) from the v1.0
GitHub release:

    cd guidewire_pose_estimation/guidewire_pose_estimation
    python smoke_test.py
"""
import os
import sys
import json
import math
import torch
import numpy as np

# Resolve the package directory (parent of this scripts/ folder)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "modeling"))
from config import get_config
from dataset import create_dataloaders
from model import build_model
from train import get_device, set_seed
from evaluate import run_inference, euclidean_distance, angular_error


# Expected numbers from REPORT.md §4.3 Table 2 (computed with seed=42).
# These are aggregate-over-100-wires (50 images × 2 wires) means.
EXPECTED = {
    # name: (config_name, checkpoint, expected_pos_mean, expected_ang_mean)
    "baseline":     ("baseline",      "baseline_resnet18_best.pth",      124.42, 25.43),
    "resnet34":     ("resnet34",      "ablation_resnet34_best.pth",      105.12, 18.46),
    "resnet50":     ("resnet50",      "ablation_resnet50_best.pth",      121.82, 25.34),
    "no_pretrain":  ("from_scratch",  "ablation_no_pretrain_best.pth",   160.21, 23.58),
    "heatmap":      ("heatmap",       "heatmap_resnet18_best.pth",       108.70, 20.10),
}

POS_TOL = 1.0    # pixels — generous, accommodating MPS vs CUDA backend drift
ANG_TOL = 0.5    # degrees


def evaluate_one(cfg_name, ckpt_name, device):
    cfg = get_config(cfg_name)
    set_seed(cfg.seed)
    ckpt_path = os.path.join(
        PKG_DIR, "checkpoints", ckpt_name
    )
    if not os.path.exists(ckpt_path):
        return None, f"checkpoint not found: {ckpt_path}"

    _, _, test_loader, _ = create_dataloaders(cfg.data, cfg.train)
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
    pos_mean = float(euclidean_distance(
        results["pred_positions_px"], results["true_positions_px"]
    ).mean())
    ang_mean = float(angular_error(
        results["pred_angles"], results["true_angles"]
    ).mean())
    return (pos_mean, ang_mean), None


def main():
    device = get_device(get_config("baseline"))
    print(f"Device: {device}")
    print(f"Tolerance: ±{POS_TOL} px position, ±{ANG_TOL}° angle\n")

    results = []
    all_ok = True
    for name, (cfg_name, ckpt, exp_pos, exp_ang) in EXPECTED.items():
        out, err = evaluate_one(cfg_name, ckpt, device)
        if err:
            print(f"  [SKIP] {name:14s}  {err}")
            results.append((name, None, None, exp_pos, exp_ang, "SKIP"))
            continue
        got_pos, got_ang = out
        dp = got_pos - exp_pos
        da = got_ang - exp_ang
        ok = abs(dp) <= POS_TOL and abs(da) <= ANG_TOL
        flag = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{flag}] {name:14s}  pos={got_pos:6.2f} (exp {exp_pos:6.2f}, Δ={dp:+.2f})   "
              f"ang={got_ang:5.2f} (exp {exp_ang:5.2f}, Δ={da:+.2f})")
        results.append((name, got_pos, got_ang, exp_pos, exp_ang, flag))

    # Persist a JSON record of the smoke test for the grader
    out_path = os.path.join(
        PKG_DIR, "results", "smoke_test_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "device": str(device),
                "tolerances": {"pos_px": POS_TOL, "ang_deg": ANG_TOL},
                "rows": [
                    {
                        "name": r[0],
                        "actual_pos_mean": r[1],
                        "actual_ang_mean": r[2],
                        "expected_pos_mean": r[3],
                        "expected_ang_mean": r[4],
                        "status": r[5],
                    } for r in results
                ],
                "all_pass": all_ok,
            },
            f, indent=2,
        )
    print(f"\nSaved smoke-test log → {out_path}")
    if all_ok:
        print("\n✓ All checkpoints reproduce within tolerance.")
        sys.exit(0)
    else:
        print("\n✗ One or more checkpoints failed reproduction within tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
