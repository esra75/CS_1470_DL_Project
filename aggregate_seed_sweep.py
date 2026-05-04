"""
aggregate_seed_sweep.py

Reads the 10 prediction CSVs produced by run_seed_sweep.sh (5 seeds x 2 models)
and produces a clean comparison table:

  - Per-seed OOF AUROC for each model
  - Mean ± std across seeds
  - Paired (per-seed) comparison so seed-noise is controlled
  - Writes the full table to seed_sweep_summary.csv for the record

"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
SWEEP_DIR    = PROJECT_ROOT / "analysis" / "seed_sweep"

SEEDS  = [42, 1, 7, 123, 2026]
MODELS = [
    ("legacy_hinge", "Legacy MouseClockCNN (hinge-MAE)"),
    ("legacy_bce", "MouseClockCNN_BCE (BCE on binary)"),
]


def score_one_run(predictions_csv: Path) -> dict:
    """Compute OOF AUROC, accuracy, MSE, AD/WT pred means from a saved
    logo_cv_predictions.csv."""
    df = pd.read_csv(predictions_csv)
    y_binary = (df["disease_score"] > 0).astype(int).values
    p        = df["pred_disease"].values
    y_cont   = df["disease_score"].values

    return {
        "n_samples": len(df),
        "auroc": float(roc_auc_score(y_binary, p)),
        "accuracy": float(accuracy_score(y_binary, (p >= 0.5).astype(int))),
        "mse_cont": float(np.mean((p - y_cont) ** 2)),
        "ad_pred": float(p[y_binary == 1].mean()),
        "wt_pred": float(p[y_binary == 0].mean()),
        "ad_wt_gap": float(p[y_binary == 1].mean() - p[y_binary == 0].mean()),
    }


def main():
    if not SWEEP_DIR.exists():
        sys.exit(f"[ERROR] {SWEEP_DIR} not found. Run run_seed_sweep.sh first")

    rows = []
    for model_key, model_label in MODELS:
        for seed in SEEDS:
            run_dir = SWEEP_DIR / f"{model_key}_seed{seed}"
            preds_csv = run_dir / "logo_cv_predictions.csv"
            if not preds_csv.exists():
                print(f"[WARN] missing: {preds_csv}")
                continue
            metrics = score_one_run(preds_csv)
            rows.append({
                "model": model_key,
                "seed": seed,
                **metrics,
            })

    if not rows:
        sys.exit("ERROR! no prediction files found")

    df = pd.DataFrame(rows)

    # Per-seed table
    print("\n" + "="*80)
    print("Per-seed results")
    print("="*80)
    for model_key, model_label in MODELS:
        sub = df[df["model"] == model_key].sort_values("seed")
        if sub.empty:
            print(f"\n{model_label}: NO RUNS FOUND")
            continue
        print(f"\n{model_label}")
        print(f"  {'seed':>6}  {'AUROC':>7}  {'acc':>6}  {'AD_pred':>7}  {'WT_pred':>7}  {'gap':>6}")
        for _, r in sub.iterrows():
            print(f"  {int(r['seed']):>6}  {r['auroc']:>7.4f}  {r['accuracy']:>6.3f}  "
                  f"{r['ad_pred']:>7.3f}  {r['wt_pred']:>7.3f}  {r['ad_wt_gap']:>6.3f}")

    # Aggregate (mean ± std) 
    print("\n" + "="*80)
    print("Aggregate across seeds (mean ± std)")
    print("="*80)
    agg = (df.groupby("model")
             .agg(auroc_mean=("auroc", "mean"),
                  auroc_std =("auroc", "std"),
                  auroc_min =("auroc", "min"),
                  auroc_max =("auroc", "max"),
                  acc_mean  =("accuracy", "mean"),
                  acc_std   =("accuracy", "std"),
                  ad_pred   =("ad_pred", "mean"),
                  wt_pred   =("wt_pred", "mean"),
                  n_seeds   =("seed", "count"))
             .reset_index())

    print(f"\n  {'model':<14}  {'AUROC mean':>11}  {'± std':>7}  {'min':>6}  {'max':>6}  "
          f"{'acc':>6}  {'n':>3}")
    for _, r in agg.iterrows():
        print(f"  {r['model']:<14}  {r['auroc_mean']:>11.4f}  {r['auroc_std']:>7.4f}  "
              f"{r['auroc_min']:>6.4f}  {r['auroc_max']:>6.4f}  {r['acc_mean']:>6.3f}  "
              f"{int(r['n_seeds']):>3}")

    # Paired comparison (controls for seed noise)
    if all(m in df["model"].values for m, _ in MODELS):
        wide = df.pivot(index="seed", columns="model", values="auroc")
        wide = wide.dropna()
        if len(wide) >= 2:
            diff = wide["legacy_bce"] - wide["legacy_hinge"]
            print(f"\n  Paired (per-seed) AUROC difference: BCE − hinge-MAE")
            for seed in wide.index:
                d = wide.loc[seed, "legacy_bce"] - wide.loc[seed, "legacy_hinge"]
                print(f"    seed {int(seed):>5}: {d:+.4f}  "
                      f"(BCE={wide.loc[seed, 'legacy_bce']:.4f}, "
                      f"hinge={wide.loc[seed, 'legacy_hinge']:.4f})")
            print(f"  Mean paired diff: {diff.mean():+.4f}  "
                  f"(std: {diff.std():.4f})")
            if diff.mean() > 0:
                winner = "legacy_bce"
            else:
                winner = "legacy_hinge"
            print(f"\n  Per-seed winner: {winner}")
            print(f"  Decisive? abs(mean diff) {'>' if abs(diff.mean()) > diff.std() else '<='} std → "
                  f"{'yes' if abs(diff.mean()) > diff.std() else 'within seed noise'}")

    # Save full table
    out_csv = SWEEP_DIR / "seed_sweep_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nFull table written to: {out_csv}")


if __name__ == "__main__":
    main()
