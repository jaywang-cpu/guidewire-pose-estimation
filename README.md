# Fluoroscopic Guidewire Localization and Pose Regression

EN.580.627 Deep Learning for Medical Imaging — Final Project (Option #1)

Estimate the 2D tip position (x, y) and the angular orientation θ of each guidewire in a fluoroscopic X-ray image. Two wires per image, 314 images total. Full write-up: [`reports/REPORT.md`](reports/REPORT.md).

---

## Repository layout

```
.
├── README.md                              # this file — start here
├── TRAINING_NOTES.md                      # engineering journal (what was tried and why)
├── Makefile                               # `make create_environment` + `make requirements`
├── pyproject.toml                         # Python 3.10 package metadata
├── requirements.txt                       # minimal runtime deps
│
├── reports/
│   ├── REPORT.md                          # written report (Markdown, source of truth)
│   ├── REPORT.pdf                         # written report (PDF, Overleaf-compiled, Times New Roman)
│   ├── overleaf_bundle.zip                # full LaTeX project (drop into Overleaf "Upload Project")
│   ├── latex_header.tex / report.css      # build inputs (kept for transparency)
│   └── overleaf_bundle/                   # unzipped LaTeX sources (gitignored)
│
├── notebooks/
│   └── main.ipynb                         # interactive end-to-end notebook
│
├── docs/
│   └── DLMI Final Project.pdf             # the original assignment specification
│
├── data/
│   └── raw/GuidewireDataset.npz           # 314 images + annotations (download from Release)
│
└── guidewire_pose_estimation/             # the importable package
    ├── __init__.py
    ├── config.py                          # all hyperparameters (single source of truth)
    ├── dataset.py                         # loading + percentile-norm + block-split + x-sort
    ├── model.py                           # GuidewireRegressionModel, HeatmapGuidewireModel, GuidewireLoss
    ├── modeling/
    │   ├── train.py                       # training loop, optimizer, schedule, early stopping
    │   └── evaluate.py                    # inference, bootstrap CI, Wilcoxon, plotting
    ├── scripts/                           # all entry-point scripts (run from repo root)
    │   ├── smoke_test.py                  # ★ one-command reproducibility check
    │   ├── run_all.py                     # end-to-end training of every experiment
    │   ├── run_with_aug.py                # the "with augmentation" ablation
    │   ├── run_kfold_cv.py                # 5-fold CV scaffold (see REPORT §V.G)
    │   ├── overfit_test.py                # 10-image sanity overfit
    │   ├── plot_cdfs.py                   # per-sample error CDF figure
    │   ├── plot_per_block.py              # per-acquisition error breakdown
    │   ├── compute_stats_table.py         # pairwise Wilcoxon + rank-biserial
    │   └── peek_data.py                   # quick dataset inspection
    ├── checkpoints/                       # trained weights (*.pth gitignored; download from Release)
    │   └── *_history.json                 # per-epoch training curves (committed)
    ├── results/                           # all metric JSONs (final_summary, per_block, Wilcoxon...)
    └── figures/                           # every PNG figure cited in the report
```

### Quick orientation for the grader

| What you want | Where it is |
|---|---|
| The headline numbers in one place | [`reports/REPORT.pdf`](reports/REPORT.pdf) §V.B (Table II) |
| Verify those numbers reproduce | `python guidewire_pose_estimation/scripts/smoke_test.py` |
| All hyperparameter settings | `guidewire_pose_estimation/config.py` |
| Hybrid-head model architecture | `guidewire_pose_estimation/model.py` |
| Training loop | `guidewire_pose_estimation/modeling/train.py` |
| Bootstrap + Wilcoxon + plotting | `guidewire_pose_estimation/modeling/evaluate.py` |
| What I tried that did not work | `TRAINING_NOTES.md` |
| Raw per-experiment metrics | `guidewire_pose_estimation/results/final_summary.json` |
| Pairwise Wilcoxon p-values + r_rb | `guidewire_pose_estimation/results/wilcoxon_table.json` |

## Where to find each deliverable

| Deliverable | Location |
|---|---|
| Final written report | [`reports/REPORT.md`](reports/REPORT.md) — Markdown source, in-repo |
| Final report (PDF) | [`reports/REPORT.pdf`](reports/REPORT.pdf) — generated from the markdown, in-repo |
| Overleaf-ready LaTeX bundle | [`reports/overleaf_bundle.zip`](reports/overleaf_bundle.zip) — `main.tex` + all figures, drag-and-drop into Overleaf |
| Source code | `guidewire_pose_estimation/` |
| Interactive notebook | [`notebooks/main.ipynb`](notebooks/main.ipynb) |
| Reproducibility smoke test | `guidewire_pose_estimation/smoke_test.py` (run after downloading dataset + checkpoints) |
| Dataset (571 MB) | v1.0 release attachment, see below |
| Trained model checkpoints (~150–330 MB each) | v1.0 release attachments, see below |

## Getting the large artifacts (dataset + trained weights)

The dataset (571 MB) and the trained model checkpoints (~150–330 MB each) are too large for the repo itself and are attached to the **v1.0 release**:

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

### 1. Create a Python 3.10 environment

Pick whichever path you prefer. Both produce an isolated environment with the project's pinned dependencies.

**Option A — conda (recommended, matches the Makefile):**
```bash
make create_environment            # conda create --name guidewire_pose_estimation python=3.10
conda activate guidewire_pose_estimation
make requirements                  # pip install -r requirements.txt
```

**Option B — plain `venv`:**
```bash
python3.10 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): PyTorch ≥ 2.0, torchvision, numpy, opencv-python, matplotlib, scipy. GPU is optional — the code automatically falls back from CUDA → MPS (Apple Silicon) → CPU.

### 2. Get the dataset and the trained weights

The dataset (571 MB) and the model checkpoints (~150–330 MB each) live in the v1.0 GitHub release (see the previous section). After downloading, place them as:
```
data/raw/GuidewireDataset.npz
guidewire_pose_estimation/checkpoints/*.pth
```

### 3. Run the pipeline

All entry-point scripts live under `guidewire_pose_estimation/scripts/`. Run them from the repository root:

```bash
# (a) one-command reproducibility check against the released checkpoints
#     (~1 min on MPS; expects checkpoints + dataset already downloaded)
python guidewire_pose_estimation/scripts/smoke_test.py

# (b) end-to-end reproduction of every experiment (a few hours on a GPU)
python guidewire_pose_estimation/scripts/run_all.py

# (c) re-generate the per-sample CDF and per-block analysis figures
python guidewire_pose_estimation/scripts/plot_cdfs.py
python guidewire_pose_estimation/scripts/plot_per_block.py

# (d) re-compute the full pairwise Wilcoxon table (REPORT Table 5)
python guidewire_pose_estimation/scripts/compute_stats_table.py

# (e) interactive notebook walkthrough
jupyter notebook notebooks/main.ipynb
```

The random seed is fixed (`seed = 42`) throughout, so the train/val/test split and the training trajectories are deterministic. `smoke_test.py` reports `Δ = 0.00` for every released checkpoint against the headline numbers in the report.

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
