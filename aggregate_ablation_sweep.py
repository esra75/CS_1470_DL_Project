"""
aggregate_ablation_sweep.py

Reads all 35 prediction CSVs from the ablation sweep (7 H5s × 5 seeds) and
produces three comparison tables:

  1. Per-H5 aggregate (mean ± std AUROC across seeds)
  2. Paired-vs-baseline: for each non-baseline H5, the per-seed AUROC
     difference vs baseline. Controls for seed noise.
  3. Critical scientific comparisons:
       a. baseline vs random_order (does spatial structure help?)
       b. no_chr9 vs random_drop_938 (is chr 9 special, or just gene loss?)
       c. no_chr9 vs no_chr3_chr10 (transgene chromosome vs other chromosomes)

OOF AUROC is the primary metric

Outputs:
  analysis/ablation_sweep/ablation_summary.csv
  analysis/ablation_sweep/ablation_paired_vs_baseline.csv
  analysis/ablation_sweep/ablation_critical_comparisons.csv
  + a printed report
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
SWEEP_DIR = PROJECT_ROOT / "analysis" / "ablation_sweep"

SEEDS = [42, 1, 7, 123, 2026]

# Labels must match run_ablation_sweep.sh
H5_LABELS = [
    "baseline",
    "random_order",
    "no_chr9",
    "random_drop_938_seedA",
    "random_drop_938_seedB",
    "no_chr3_chr10",
    "autosomes_only",
]

# Critical scientific comparisons - Format: (label_A, label_B, hypothesis)
CRITICAL_COMPARISONS = [
    ("baseline", "random_order",
     "If baseline > random_order: chromosome ordering carries signal."),
    ("no_chr9", "random_drop_938_seedA",
     "If no_chr9 < random_drop_938_seedA: chr 9 is special (not just gene loss)."),
    ("no_chr9", "random_drop_938_seedB",
     "Replicate of above with second random-drop seed."),
    ("no_chr9", "no_chr3_chr10",
     "If no_chr9 < no_chr3_chr10: chr 9 hurts more than other chromosomes."),
    ("baseline", "no_chr9",
     "Magnitude of the chr 9 ablation effect."),
    ("baseline", "autosomes_only",
     "Magnitude of the sex-chromosome ablation effect (smaller expected)."),
]


def score_one_run(predictions_csv: Path) -> dict:
    df = pd.read_csv(predictions_csv)
    y_binary = (df["disease_score"] > 0).astype(int).values
    p  = df["pred_disease"].values
    y_cont = df["disease_score"].values
    return {
        "n_samples": len(df),
        "auroc": float(roc_auc_score(y_binary, p)),
        "accuracy": float(accuracy_score(y_binary, (p >= 0.5).astype(int))),
        "mse_cont": float(np.mean((p - y_cont) ** 2)),
        "ad_pred": float(p[y_binary == 1].mean()),
        "wt_pred": float(p[y_binary == 0].mean()),
    }


def main():
    if not SWEEP_DIR.exists():
        sys.exit(f"[ERROR] {SWEEP_DIR} not found. Run run_ablation_sweep.sh first")

    # Collect all runs
    rows = []
    missing = []
    for label in H5_LABELS:
        for seed in SEEDS:
            run_dir = SWEEP_DIR / f"{label}_seed{seed}"
            preds_csv = run_dir / "logo_cv_predictions.csv"
            if not preds_csv.exists():
                missing.append(f"{label}_seed{seed}")
                continue
            metrics = score_one_run(preds_csv)
            rows.append({"h5": label, "seed": seed, **metrics})

    if missing:
        print(f"[WARN] {len(missing)} runs missing:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()

    if not rows:
        sys.exit("[ERROR] no prediction files found")

    df = pd.DataFrame(rows)

    # Per-seed table
    print("=" * 88)
    print("Per-seed AUROC table")
    print("=" * 88)
    pivot = df.pivot(index="h5", columns="seed", values="auroc")
    pivot = pivot.reindex(H5_LABELS)
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    # Aggregate (mean ± std)
    print("=" * 88)
    print("Aggregate across seeds (mean ± std)")
    print("=" * 88)
    agg = (df.groupby("h5")
             .agg(auroc_mean=("auroc", "mean"),
                  auroc_std =("auroc", "std"),
                  auroc_min =("auroc", "min"),
                  auroc_max =("auroc", "max"),
                  acc_mean  =("accuracy", "mean"),
                  ad_pred_mean=("ad_pred", "mean"),
                  wt_pred_mean=("wt_pred", "mean"),
                  n_seeds   =("seed", "count"))
             .reindex(H5_LABELS)
             .reset_index())

    print(f"{'h5':<26} {'AUROC':>7} {'± std':>7} {'min':>6} {'max':>6} "
          f"{'acc':>6} {'AD':>5} {'WT':>5} {'n':>3}")
    print("─" * 88)
    for _, r in agg.iterrows():
        print(f"{r['h5']:<26} {r['auroc_mean']:>7.4f} {r['auroc_std']:>7.4f} "
              f"{r['auroc_min']:>6.4f} {r['auroc_max']:>6.4f} {r['acc_mean']:>6.3f} "
              f"{r['ad_pred_mean']:>5.2f} {r['wt_pred_mean']:>5.2f} "
              f"{int(r['n_seeds']):>3}")
    print()

    # Paired vs baseline
    print("=" * 88)
    print("Paired-vs-baseline AUROC differences (per seed)")
    print("=" * 88)
    print("Each (ablation - baseline) computed at matched seed → seed-noise cancels.")
    print()
    wide = df.pivot(index="seed", columns="h5", values="auroc")
    paired_rows = []
    if "baseline" not in wide.columns:
        print("[WARN] baseline runs missing — paired comparison skipped")
    else:
        baseline_col = wide["baseline"]
        for label in H5_LABELS:
            if label == "baseline" or label not in wide.columns:
                continue
            diff = wide[label] - baseline_col
            paired_rows.append({
                "h5":              label,
                "n_paired_seeds": int(diff.notna().sum()),
                "mean_diff": float(diff.mean()),
                "std_diff": float(diff.std()),
                "min_diff": float(diff.min()),
                "max_diff": float(diff.max()),
                "decisive": bool(abs(diff.mean()) > diff.std()),
            })
        paired_df = pd.DataFrame(paired_rows)
        print(f"{'h5':<26} {'mean Δ':>9} {'std Δ':>8} {'min Δ':>8} {'max Δ':>8} "
              f"{'decisive':>9}")
        print("─" * 88)
        for _, r in paired_df.iterrows():
            print(f"{r['h5']:<26} {r['mean_diff']:>+9.4f} {r['std_diff']:>8.4f} "
                  f"{r['min_diff']:>+8.4f} {r['max_diff']:>+8.4f} "
                  f"{'yes' if r['decisive'] else 'no':>9}")
        print()
        print("'decisive' means |mean Δ| > std Δ across seeds. Negative mean Δ = ablation hurt.")
    print()

    # scientific comparisons
    print("=" * 88)
    print("Critical scientific comparisons")
    print("=" * 88)
    crit_rows = []
    for label_a, label_b, hypothesis in CRITICAL_COMPARISONS:
        if label_a not in wide.columns or label_b not in wide.columns:
            print(f"[skip] {label_a} vs {label_b}: missing runs")
            continue
        diff = wide[label_a] - wide[label_b]
        d_mean = float(diff.mean())
        d_std  = float(diff.std())
        decisive = abs(d_mean) > d_std
        crit_rows.append({
            "comparison": f"{label_a} - {label_b}",
            "hypothesis": hypothesis,
            "n_paired_seeds": int(diff.notna().sum()),
            "mean_diff": d_mean,
            "std_diff": d_std,
            "decisive": decisive,
            f"auroc_{label_a}_mean": float(wide[label_a].mean()),
            f"auroc_{label_b}_mean": float(wide[label_b].mean()),
        })
        print(f"\n  {label_a:<26} − {label_b:<26}")
        print(f" {hypothesis}")
        print(f" Mean diff: {d_mean:+.4f}  (std {d_std:.4f}, "
              f"{'decisive' if decisive else 'within seed noise'})")
        for seed in wide.index:
            a = wide.loc[seed, label_a]
            b = wide.loc[seed, label_b]
            print(f"      seed {int(seed):>5}: {a:.4f} − {b:.4f} = {a-b:+.4f}")
    print()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SWEEP_DIR / "ablation_full_per_seed.csv", index=False)
    agg.to_csv(SWEEP_DIR / "ablation_summary.csv", index=False)
    if 'paired_df' in locals():
        paired_df.to_csv(SWEEP_DIR / "ablation_paired_vs_baseline.csv", index=False)
    if crit_rows:
        pd.DataFrame(crit_rows).to_csv(
            SWEEP_DIR / "ablation_critical_comparisons.csv", index=False)

    print(f"Written:")
    print(f" {SWEEP_DIR / 'ablation_full_per_seed.csv'}")
    print(f" {SWEEP_DIR / 'ablation_summary.csv'}")
    print(f" {SWEEP_DIR / 'ablation_paired_vs_baseline.csv'}")
    print(f" {SWEEP_DIR / 'ablation_critical_comparisons.csv'}")


if __name__ == "__main__":
    main()
