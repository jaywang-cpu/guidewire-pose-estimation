# Results directory

Every CSV here is a final or intermediate output of the experiments described in `reports/REPORT.md`.

| File | What it is | Generator |
|---|---|---|
| `metrics_per_experiment.csv` | Master table: per-wire and aggregate metrics (mean, 95% bootstrap CI, median, std, p90) for every experiment and metric. **Open this first.** | `scripts/run_all.py`, `scripts/run_with_aug.py` |
| `final_summary.csv` | Headline aggregate-only view (one row per experiment) — same data as the `aggregate` rows of `metrics_per_experiment.csv` extracted for convenience. | `scripts/run_all.py` |
| `per_sample_errors.csv` | Long-format: every per-prediction error (100 predictions per experiment) for the Euclidean tip and angular metrics. Source for the CDF figure. | `scripts/plot_cdfs.py` |
| `per_block_errors.csv` | ResNet-34 median test errors broken down by source block (proxy for anatomical acquisition). See REPORT §4.5. | `scripts/plot_per_block.py` |
| `wilcoxon_table.csv` | Paired Wilcoxon signed-rank statistics for every ablation vs. baseline, separately for position and angular error, with rank-biserial effect size. See REPORT §V.C. | `scripts/compute_stats_table.py` |
| `smoke_test_results.csv` | Reproducibility check: actual vs. expected headline metrics for each released checkpoint. PASS / FAIL / SKIP. | `scripts/smoke_test.py` |

## How to regenerate

After downloading the dataset and `.pth` checkpoints from the v1.0 release:

```bash
# from the repository root:
python guidewire_pose_estimation/scripts/smoke_test.py        # smoke_test_results.csv
python guidewire_pose_estimation/scripts/plot_cdfs.py         # per_sample_errors.csv (and the CDF figure)
python guidewire_pose_estimation/scripts/plot_per_block.py    # per_block_errors.csv
python guidewire_pose_estimation/scripts/compute_stats_table.py   # wilcoxon_table.csv
python guidewire_pose_estimation/scripts/run_all.py           # metrics_per_experiment.csv + final_summary.csv (slow)
```

## Notes on the units

- `euc_*` / `pos_*` columns are pixels on the original 976 × 976 image.
- `ang_*` columns are degrees in [0°, 180°].
- `*_ci_lo` / `*_ci_hi` columns are the 2.5th / 97.5th percentiles of a 1000-resample bootstrap on the corresponding scalar.
- `p_value` is from a two-sided paired Wilcoxon signed-rank test.
- `rank_biserial_r` is the rank-biserial effect size in [−1, +1]; negative favors the candidate over the baseline.
