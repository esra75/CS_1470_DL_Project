"""
evaluate_LOGO.py - Leave-One-Group-Out Cross-Validation for the Mouse AD Clock


Usage (from mouse_clock/ project root):
  python evaluate_LOGO.py # CNN on ComBat 55k genes
  python evaluate_LOGO.py --model mlp --n-hvgs # MLP on HVG data
  python evaluate_LOGO.py --no-combat # CNN on no-ComBat data
  python evaluate_LOGO.py --epochs 50 # quick diagnostic run
"""

import argparse
import os
import sys
import time
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DATASET_TO_ID # {"3xTgAD":0, "5xFAD":1, ...}
from model.cnn_clock import MouseClockCNN  # forward(x, model_id)  x=(B,1,G)
from model.mlp_clock import MouseClockMLP # forward(x, model_id)  x=(B,G)

def parse_args():
    p = argparse.ArgumentParser(description="LOGO-CV for Mouse AD Clock")
    p.add_argument("--model", choices=["cnn", "mlp"], default="cnn",
                   help="Architecture to evaluate (default: cnn)")
    p.add_argument("--no-combat", action="store_true",
                   help="Use non-ComBat-corrected data (nc_combined_*)")
    p.add_argument("--n-hvgs", action="store_true",
                   help="Use HVG-filtered data (hvg_combined_*)")
    # Training hyper-parameters — match train.py / train_HVG.py defaults exactly
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--wd",         type=float, default=1e-4)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lambda-age", dest="lambda_age", type=float, default=0.2,
                   help="Weight for auxiliary age-regression loss")
    p.add_argument("--delta",      type=float, default=0.0,
                   help="Hinge margin for AD samples (0 = plain MAE for AD)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out-dir",    type=str,   default="analysis/logo_cv")
    return p.parse_args()


# load data 
def load_data(args):
    """
    Return (X, meta) where:
      X : (N, G) float32 numpy array  (log1p z-scored, ± ComBat)
      meta : pd.DataFrame with columns:
             sample_id, dataset, genotype, genotype_norm, sex,
             age_months, disease_score, strat_key
    """
    if args.n_hvgs:
        expr_path = PROJECT_ROOT / "data" / "hvg_combined_expression.h5"
        # preprocess_HVG.py writes hvg_combined_metadata.csv as a copy of
        # combined_metadata.csv — fall back if it's missing
        meta_path = PROJECT_ROOT / "data" / "hvg_combined_metadata.csv"
        if not meta_path.exists():
            meta_path = PROJECT_ROOT / "data" / "combined_metadata.csv"
        if not expr_path.exists():
            sys.exit(
                f " ERROR! HVG expression file not found: {expr_path}\n"
                "Run: python preprocessing/preprocess_HVG.py first"
            )
    else:
        prefix = "nc_" if args.no_combat else ""
        expr_path = PROJECT_ROOT / "data" / f"{prefix}combined_expression.h5"
        meta_path = PROJECT_ROOT / "data" / f"{prefix}combined_metadata.csv"

    if not expr_path.exists():
        sys.exit(f" ERROR! Expression file not found: {expr_path}")
    if not meta_path.exists():
        sys.exit(f"ERROR! Metadata file not found:   {meta_path}")

    print(f"Loading expression : {expr_path}")
    # Key is "X" — confirmed in preprocess.py and preprocess_HVG.py
    with h5py.File(expr_path, "r") as f:
        X = f["X"][:].astype(np.float32)
        sample_ids_h5 = np.array(f["sample_ids"]).astype(str)

    print(f"Loading metadata : {meta_path}")
    meta_raw = pd.read_csv(meta_path)


    if "sample_id" in meta_raw.columns:
        missing = set(sample_ids_h5) - set(meta_raw["sample_id"])
        if missing:
            sys.exit(f"[ERROR] {len(missing)} h5 sample_id(s) missing from metadata: "
                     f"{list(missing)[:3]}")
        meta = (meta_raw.set_index("sample_id")
                        .loc[sample_ids_h5]
                        .reset_index())
    else:
        print(" WARNING! No sample_id column — assuming rows match h5 order")
        meta = meta_raw.reset_index(drop=True)

    assert len(X) == len(meta), \
        f"Row mismatch: expression has {len(X)} rows, metadata has {len(meta)}"

    # genotype_norm should exist (written by preprocess.py); rebuild if missing
    if "genotype_norm" not in meta.columns:
        print("WARNING! 'genotype_norm' not found — inferring from 'genotype'")
        wt_tokens = {"wt", "wildtype", "bl6", "3xtgadwt"}
        meta["genotype_norm"] = meta["genotype"].apply(
            lambda g: "WT" if any(t in str(g).lower() for t in wt_tokens) else "AD"
        )

    print(f"\nDataset: {len(X)} samples × {X.shape[1]} genes")
    print(meta.groupby(["dataset", "genotype_norm", "age_months", "sex"])
              .size().rename("n").reset_index().to_string(index=False))
    return X, meta


def build_group_labels(meta: pd.DataFrame) -> np.ndarray:
    """
    Integer group ID per sample from (dataset x genotype_norm x age_months x sex)
    Mirrors the strat_key logic in preprocess.py / GroupShuffleSplit in dataset.py
    """
    key = (meta["dataset"].astype(str) + "|" +
           meta["genotype_norm"].astype(str) + "|" +
           meta["age_months"].astype(str) + "|" +
           meta["sex"].str.lower().astype(str))
    codes, uniques = pd.factorize(key)
    print(f"\nTotal groups: {len(uniques)}")
    for gid, label in enumerate(uniques):
        print(f" group {gid:2d}: {label}  (n={(codes == gid).sum()})")
    return codes



def make_model(args, n_genes: int, device: torch.device) -> nn.Module:
    n_models = len(DATASET_TO_ID)
    if args.model == "cnn":
        model = MouseClockCNN(
            n_genes = n_genes,
            n_models = n_models,
            lambda_age = args.lambda_age,
            # sex_embed_dim uses the default (8) — matches train.py
        )
    else:
        model = MouseClockMLP(
            n_genes    = n_genes,
            n_models   = n_models,
            lambda_age = args.lambda_age,
        )
    return model.to(device)



# One-fold training
def train_one_fold(
    X_tr: np.ndarray,
    meta_tr: pd.DataFrame,
    args,
    n_genes: int,
    device: torch.device,
) -> nn.Module:
    model     = make_model(args, n_genes, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)


    disease_t = torch.tensor(meta_tr["disease_score"].values, dtype=torch.float32)
    age_t = torch.tensor(meta_tr["age_months"].values,    dtype=torch.float32)
    model_t = torch.tensor(
        meta_tr["dataset"].map(DATASET_TO_ID).fillna(0).values.astype(np.int64),
        dtype=torch.long)
    sex_t = torch.tensor(
        meta_tr["sex"].str.lower().map({"male": 0, "female": 1}).fillna(0).values.astype(np.int64),
        dtype=torch.long)
    X_t = torch.tensor(X_tr, dtype=torch.float32)   # (N, G)

    is_cnn = (args.model == "cnn")
    N = len(X_t)

    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, args.batch):
            idx = perm[start : start + args.batch]

            xb = X_t[idx].to(device)
            db = disease_t[idx].to(device)
            ab = age_t[idx].to(device)
            mb = model_t[idx].to(device)
            sb = sex_t[idx].to(device)
            x_in = xb.unsqueeze(1) if is_cnn else xb

            optimizer.zero_grad()
            d_pred, a_pred = model(x_in, mb, sb) # model.forward(x, model_id, sex_id)
            genotype_norm = (db > 0).long()
            total, _, _ = model.compute_loss(
                d_pred, a_pred, db, ab,
                genotype_norm=genotype_norm,
                delta=args.delta,
            )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total.item()
            n_batches  += 1

        scheduler.step(epoch_loss / n_batches)

    return model


# One-fold evaluation
@torch.no_grad()
def eval_fold(
    model: nn.Module,
    X_te: np.ndarray,
    meta_te: pd.DataFrame,
    args,
    device: torch.device,
):
    model.eval()

    X_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    mb  = torch.tensor(
        meta_te["dataset"].map(DATASET_TO_ID).fillna(0).values.astype(np.int64),
        dtype=torch.long).to(device)
    sb  = torch.tensor(
        meta_te["sex"].str.lower().map({"male": 0, "female": 1}).fillna(0).values.astype(np.int64),
        dtype=torch.long).to(device)

    x_in = X_t.unsqueeze(1) if args.model == "cnn" else X_t
    d_pred, _ = model(x_in, mb, sb)

    preds = np.atleast_1d(d_pred.squeeze().cpu().numpy())
    trues = np.atleast_1d(meta_te["disease_score"].values.astype(np.float32))

    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    r, pval = (pearsonr(trues, preds) if len(trues) > 2
               else (float("nan"), float("nan")))

    return preds, trues, mse, mae, float(r), float(pval)



def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device : {device}")
    print(f"Model : {args.model.upper()}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, meta = load_data(args)
    n_genes = X.shape[1]
    groups = build_group_labels(meta)

    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(X, groups=groups)
    print(f"\nRunning {n_folds}-fold LOGO-CV …\n")

    all_results = []
    all_preds = []

    for fold_idx, (train_idx, test_idx) in enumerate(
            logo.split(X, groups=groups), start=1):

        X_tr = X[train_idx]
        X_te = X[test_idx]
        meta_tr = meta.iloc[train_idx].reset_index(drop=True)
        meta_te = meta.iloc[test_idx].reset_index(drop=True)

        g_label = (f"{meta_te['dataset'].iloc[0]}|"
                   f"{meta_te['genotype_norm'].iloc[0]}|"
                   f"{meta_te['age_months'].iloc[0]}m|"
                   f"{meta_te['sex'].iloc[0].lower()}")

        print(f"Fold {fold_idx:2d}/{n_folds}  hold-out: {g_label:48s}  "
              f"(n={len(test_idx)})  training on {len(train_idx)} samples")

        t0    = time.time()
        model = train_one_fold(X_tr, meta_tr, args, n_genes, device)
        preds, trues, mse, mae, r, pval = eval_fold(
            model, X_te, meta_te, args, device)
        elapsed = time.time() - t0

        naive_mse = float(np.mean(trues ** 2))   # predict-zero baseline
        print(f" MSE={mse:.4f}  naive={naive_mse:.4f}  "
              f"MAE={mae:.4f}  r={r:.3f}  ({elapsed:.0f}s)")

        all_results.append({
            "fold": fold_idx,
            "group": g_label,
            "dataset": meta_te["dataset"].iloc[0],
            "genotype_norm": meta_te["genotype_norm"].iloc[0],
            "age_months": meta_te["age_months"].iloc[0],
            "sex": meta_te["sex"].iloc[0].lower(),
            "n_test": len(test_idx),
            "mse":  mse,
            "naive_mse": naive_mse,
            "beats_baseline":int(mse < naive_mse),
            "mae": mae,
            "pearson_r": r,
            "pearson_p": pval,
            "train_time_s":  elapsed,
        })

        fold_pred_df = meta_te[
            ["dataset", "genotype_norm", "age_months", "sex", "disease_score"]
        ].copy()
        fold_pred_df["sample_id"] = (meta_te["sample_id"].values
                                         if "sample_id" in meta_te.columns
                                         else np.arange(len(meta_te)))
        fold_pred_df["pred_disease"] = preds
        fold_pred_df["fold"]         = fold_idx
        all_preds.append(fold_pred_df)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


    results_df = pd.DataFrame(all_results)
    preds_df   = pd.concat(all_preds, ignore_index=True)

    oof_mse      = float(np.mean((preds_df["pred_disease"] -
                                   preds_df["disease_score"]) ** 2))
    oof_mae      = float(np.mean(np.abs(preds_df["pred_disease"] - preds_df["disease_score"])))
    oof_r, oof_p = pearsonr(preds_df["disease_score"], preds_df["pred_disease"])
    naive_overall= float(np.mean(preds_df["disease_score"] ** 2))
    n_beats      = int(results_df["beats_baseline"].sum())

    summary_lines = [
        f"LOGO-CV Summary ({n_folds} folds)",
        f"Model: {args.model.upper()}",
        f"Genes: {n_genes}",
        f"Epochs/fold: {args.epochs}",
        f"Lambda-age:  {args.lambda_age}",
        f"Delta (hinge): {args.delta}",
        "",
        f"Overall OOF MSE: {oof_mse:.4f}  "
        f"(predict-zero naive: {naive_overall:.4f})",
        f"Overall OOF MAE: {oof_mae:.4f}",
        f"Overall OOF r: {oof_r:.3f}  (p={oof_p:.3e})",
        f"Folds beating naive baseline: {n_beats}/{n_folds}",
        "",
        f"{'Fold':>4} {'Group':<50}  {'MSE':>7}  {'Naive':>7}  "
        f"{'Beats?':>6}  {'r':>6}",
        "-" * 90,
    ]
    for _, row in results_df.iterrows():
        beat = "check" if row.beats_baseline else "X"
        summary_lines.append(
            f"{int(row.fold):4d}  {row.group:<50}  "
            f"{row.mse:7.4f}  {row.naive_mse:7.4f}  {beat:>6}  "
            f"{row.pearson_r:6.3f}")

    summary_lines += ["", "Mean MSE by dataset:"]
    for ds, grp in results_df.groupby("dataset"):
        summary_lines.append(
            f" {ds}: {grp['mse'].mean():.4f} ± {grp['mse'].std():.4f}  "
            f"(naive: {grp['naive_mse'].mean():.4f})")

    summary_lines += ["", "Mean MSE by genotype:"]
    for gt, grp in results_df.groupby("genotype_norm"):
        summary_lines.append(
            f"  {gt}: {grp['mse'].mean():.4f} ± {grp['mse'].std():.4f}  "
            f"(naive: {grp['naive_mse'].mean():.4f})")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    results_df.to_csv(out_dir / "logo_cv_results.csv", index=False)
    preds_df.to_csv(out_dir / "logo_cv_predictions.csv", index=False)
    (out_dir / "logo_cv_summary.txt").write_text(summary_text)
    print(f"\nSaved to {out_dir}/")

    _plot_mse_by_group(results_df, out_dir)
    _plot_scatter(preds_df, oof_r, oof_mse, out_dir)
    _plot_mse_by_age(results_df, out_dir)

    print("Done!")

# plotting 
DS_COLORS = {"5xFAD": "#E74C3C", "3xTgAD": "#3498DB"}


def _plot_mse_by_group(results_df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    x       = np.arange(len(results_df))
    colors  = results_df["dataset"].map(DS_COLORS).fillna("grey")

    ax.bar(x, results_df["mse"], color=colors,
           edgecolor="white", linewidth=0.5)
    ax.plot(x, results_df["naive_mse"], marker="x", color="black",
            linewidth=0, markersize=6, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(results_df["group"], rotation=90, fontsize=7)
    ax.set_ylabel("Test MSE")
    ax.set_title("LOGO-CV: Per-Group Test MSE vs Predict-Zero Baseline (x)")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=DS_COLORS["5xFAD"],  label="5xFAD"),
        Patch(facecolor=DS_COLORS["3xTgAD"], label="3xTgAD"),
        plt.Line2D([0],[0], color="black", marker="x",
                   linewidth=0, markersize=7, label="Predict-zero baseline"),
    ], loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / "logo_cv_mse_by_group.png", dpi=150)
    plt.close()


def _plot_scatter(preds_df: pd.DataFrame, r: float, mse: float, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors  = preds_df["dataset"].map(DS_COLORS).fillna("grey")
    ax.scatter(preds_df["disease_score"], preds_df["pred_disease"],
               c=colors, alpha=0.65, s=35, edgecolors="none")
    lo = min(preds_df["disease_score"].min(), preds_df["pred_disease"].min()) - 0.05
    hi = max(preds_df["disease_score"].max(), preds_df["pred_disease"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True disease score")
    ax.set_ylabel("Predicted disease score")
    ax.set_title(f"LOGO-CV OOF predictions\nr={r:.3f}  MSE={mse:.4f}")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=DS_COLORS["5xFAD"],  label="5xFAD"),
        Patch(facecolor=DS_COLORS["3xTgAD"], label="3xTgAD"),
    ])
    plt.tight_layout()
    plt.savefig(out_dir / "logo_cv_scatter.png", dpi=150)
    plt.close()


def _plot_mse_by_age(results_df: pd.DataFrame, out_dir: Path):
    datasets = sorted(results_df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets),
                              figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub   = results_df[results_df["dataset"] == ds]
        color = DS_COLORS.get(ds, "grey")
        for gt, grp in sub.groupby("genotype_norm"):
            grp = grp.sort_values("age_months")
            ls  = "-" if gt == "AD" else "--"
            ax.plot(grp["age_months"], grp["mse"],
                    marker="o", linestyle=ls, color=color, label=gt, alpha=0.85)
            ax.plot(grp["age_months"], grp["naive_mse"],
                    marker="x", linestyle=ls, color="black", alpha=0.35, linewidth=1)
        ax.set_title(ds)
        ax.set_xlabel("Age (months)")
        ax.set_ylabel("Test MSE")
        ax.legend(fontsize=8, title="Genotype")

    fig.suptitle("LOGO-CV MSE by Age and Genotype\n(faint x = predict-zero baseline)",
                 y=1.03)
    plt.tight_layout()
    plt.savefig(out_dir / "logo_cv_mse_by_age.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
