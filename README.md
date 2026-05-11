# Fluoroscopic Guidewire Localization and Pose Regression

EN.580.627 Deep Learning for Medical Imaging — Final Project (Option #1)

Estimate the 2D tip position (x, y) and the angular orientation θ of each guidewire in a fluoroscopic X-ray image. Two wires per image, 314 images total. Full write-up: [`reports/REPORT.md`](reports/REPORT.md).

---

## Repository layout

```
.
├── README.md                       # this file
├── reports/
│   └── REPORT.md                   # final written report (Intro / Materials / Methods / Results / Discussion)
├── TRAINING_NOTES.md               # debugging journal — failed attempts and how they were diagnosed
├── notebooks/
│   └── main.ipynb                  # end-to-end notebook
├── data/raw/
│   └── GuidewireDataset.npz        # 314 images + tip / direction annotations
├── guidewire_pose_estimation/      # source package
│   ├── config.py                   # all hyperparameters (single source of truth)
│   ├── dataset.py                  # loading, percentile normalization, block split, x-sort
│   ├── model.py                    # GuidewireRegressionModel, HeatmapGuidewireModel, GuidewireLoss
│   ├── overfit_test.py             # 10-image overfit sanity check
│   ├── modeling/
│   │   ├── train.py                # training loop with early stopping
│   │   ├── evaluate.py             # bootstrap CI + Wilcoxon + visualisations
│   │   └── predict.py
│   ├── run_all.py                  # reproduces every experiment in one shot
│   ├── checkpoints/                # trained weights (one .pth + one _history.json per experiment)
│   ├── results/                    # final_summary.json with all metrics
│   └── figures/                    # PNG figures used in the report
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## Getting the large artifacts (dataset + trained weights)

The git repository contains source code, figures, JSON metrics, and the written report. The dataset (571 MB) and the six trained model checkpoints (~150–330 MB each) are too large for the repo itself and are attached to the **v1.0 release**:

**Download page:** <https://github.com/jaywang-cpu/guidewire-pose-estimation/releases/tag/v1.0>

After downloading, place the files like this so the loader and the notebook find them:

```
data/raw/GuidewireDataset.npz
data/raw/LICENSE.txt
guidewire_pose_estimation/checkpoints/*.pth
```

Or, in one shell command (requires the GitHub CLI `gh`):

```bash
gh release download v1.0 --repo jaywang-cpu/guidewire-pose-estimation \
  -p "GuidewireDataset.npz" -p "LICENSE.txt" -D data/raw/

gh release download v1.0 --repo jaywang-cpu/guidewire-pose-estimation \
  -p "*.pth" -D guidewire_pose_estimation/checkpoints/
```

## How to run

```bash
conda activate biomedical
pip install -r requirements.txt

cd guidewire_pose_estimation/guidewire_pose_estimation
python run_all.py                  # reproduces every experiment, ~ a few hours on a GPU
# or, interactively:
jupyter notebook ../../notebooks/main.ipynb
```

Random seed is fixed (`seed = 42`) so the train/val/test split and the training trajectories are deterministic.

## Final results (test set, N = 50 images, 100 wire predictions)

| Experiment | Pos mean (px) | Pos median | Ang mean (°) | Ang median |
|---|:-:|:-:|:-:|:-:|
| **ResNet-34** (best) | **105.1** | **86.5** | **18.5** | **10.0** |
| Heatmap (ResNet-18) | 108.7 | 98.0 | 20.1 | 16.2 |
| ResNet-50 | 121.8 | 113.2 | 25.3 | 19.3 |
| ResNet-18 (baseline) | 124.4 | 90.9 | 25.4 | 15.2 |
| No pre-training | 160.2 | 122.5 | 23.6 | 22.7 |

Errors are reported in pixels on the original 976×976 image. 95% bootstrap confidence intervals and Wilcoxon signed-rank tests are in `reports/REPORT.md` §4.

## Short summary of the approach

- **Architecture**: ResNet encoder, then a hybrid head — **spatial conv + soft-argmax** for tip position (preserves "where") and **GAP + FC + Tanh + L2-normalize** for direction (predicts a unit (sin θ, cos θ)). The asymmetric head was the single biggest improvement over a plain GAP+FC regression baseline.
- **Two-wire ordering**: wires are sorted by tip x-coordinate at dataset construction so the network has a deterministic output ordering. No Hungarian matching at training time; matching is used at evaluation only.
- **Regularization**: Mixup (α = 0.4), dropout 0.3 in the heads, weight decay 1e-3, early stopping on validation loss. Geometric augmentation was tested and dropped — it interacted badly with the x-sort.
- **Loss**: `5 · MSE(position) + 5 · (1 − cos sim)(direction)`. The 5:5 weighting is required — under 5:1 the angular metric stays near the random-guess regime.
- **Training**: AdamW, differential LR (heads 1e-3, backbone 1e-4), cosine annealing, no warmup, no freeze. Max 200 epochs with patience-30 early stopping.

The "before vs after" effect of these decisions on test-set numbers is in `TRAINING_NOTES.md` and is summarised in `reports/REPORT.md` §3.7 and §5.1.

## Files to reproduce the reported numbers

| File | What it contains |
|---|---|
| `guidewire_pose_estimation/results/final_summary.json` | All metrics + 95% bootstrap CIs |
| `guidewire_pose_estimation/checkpoints/*.pth` | Trained model weights, one per experiment |
| `guidewire_pose_estimation/checkpoints/*_history.json` | Per-epoch train/val loss + metric curves |
| `guidewire_pose_estimation/figures/*.png` | Every figure cited in the report |

`ablation_resnet34_best.pth` is the best overall model and is the one referenced for the headline numbers in the report.
