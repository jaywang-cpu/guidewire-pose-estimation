"""
Compute the full statistical-comparison table that the IEEE paper
references: pairwise Wilcoxon signed-rank p-values and rank-biserial effect
sizes for every (experiment vs. baseline) pair, separately for position and
angular error.
"""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon

# Resolve the package directory (parent of this scripts/ folder)
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, os.path.join(PKG_DIR, "modeling"))
PATH = os.path.join(PKG_DIR, "results", "per_sample_errors.json")
with open(PATH) as f:
    raw = json.load(f)

# Reorder so baseline is first
order = [
    "ResNet-18 (baseline)",
    "ResNet-34",
    "Heatmap (ResNet-18)",
    "ResNet-50",
    "No pre-training",
    "ResNet-18 + augmentation",
]
data = {n: raw[n] for n in order}
baseline = data["ResNet-18 (baseline)"]

print(f"{'Comparison':<32} {'Metric':<6} {'Base mean':<10} {'Cand mean':<10} "
      f"{'Δ':<8} {'W':<10} {'p':<10} {'r_rb':<8} {'sig':<4}")
print("-" * 100)

rows = []
for name in order[1:]:
    cand = data[name]
    for metric in ("euclidean", "angular"):
        b = np.array(baseline[metric])
        c = np.array(cand[metric])
        n = len(b)
        # Wilcoxon signed-rank, two-sided
        stat, p = wilcoxon(c, b, zero_method="wilcox", alternative="two-sided")
        # Rank-biserial correlation (effect size): r = 1 - (2W / (n*(n+1)/2))
        # But the more interpretable form: r_rb = (W+ - W-) / (W+ + W-)
        diffs = c - b
        diffs_nonzero = diffs[diffs != 0]
        abs_ranks = np.argsort(np.argsort(np.abs(diffs_nonzero))) + 1
        W_plus = abs_ranks[diffs_nonzero > 0].sum()
        W_minus = abs_ranks[diffs_nonzero < 0].sum()
        r_rb = (W_plus - W_minus) / (W_plus + W_minus) if (W_plus + W_minus) > 0 else 0.0

        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"{name:<32} {metric:<6} {b.mean():<10.2f} {c.mean():<10.2f} "
              f"{c.mean()-b.mean():<+8.2f} {stat:<10.1f} {p:<10.2e} {r_rb:<+8.3f} {sig:<4}")
        rows.append({
            "comparison": name + " vs baseline",
            "metric": metric,
            "n": n,
            "baseline_mean": float(b.mean()),
            "candidate_mean": float(c.mean()),
            "delta": float(c.mean() - b.mean()),
            "wilcoxon_W": float(stat),
            "p_value": float(p),
            "rank_biserial_r": float(r_rb),
            "significance": sig,
        })

# Bonferroni-corrected alpha for 5 comparisons × 2 metrics = 10 tests
alpha_bonf = 0.05 / 10
print(f"\nBonferroni-corrected α (k=10): {alpha_bonf:.4f}")

# Save
out_path = os.path.join(
    PKG_DIR, "results", "wilcoxon_table.json"
)
with open(out_path, "w") as f:
    json.dump({"alpha_uncorrected": 0.05,
               "alpha_bonferroni_k10": alpha_bonf,
               "rows": rows}, f, indent=2)
print(f"Saved → {out_path}")
