"""
evaluate_LOGO_legacy_loop_bce.py

Legacy LOGO training loop with MouseClockCNN_BCE (legacy architecture +
BCE loss). Tests whether the loss change alone is what kills the legacy
0.96 result


Run:
python evaluate_LOGO_legacy_loop_bce.py --epochs 150 \\
      --out-dir analysis/logo_legacy_loop_bce
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneGroupOut

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import DATASET_TO_ID
from model.cnn_clock_bce import MouseClockCNN_BCE


def parse_args():
    p = argparse.ArgumentParser(
        description="Legacy LOGO loop with MouseClockCNN_BCE (legacy arch + BCE loss)"
    )
    p.add_argument("--no-combat", action="store_true",
                   help="Use nc_combined_* instead of combined_* (ignored if --h5 given)")
    p.add_argument("--h5", type=str, default=None,
                   help="Explicit H5 path (e.g. data/combined_no_chr9.h5). "
                        "Overrides --no-combat. For ablation runs.")
    p.add_argument("--meta", type=str, default=None,
                   help="Explicit metadata CSV. Defaults to data/combined_metadata.csv "
                        "when --h5 is given (most gene-axis ablations share the baseline "
                        "metadata). Pass explicitly for sample-axis ablations like "
                        "combined_WT_only.h5.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lambda-age", dest="lambda_age", type=float, default=0.2)
    p.add_argument("--delta", type=float, default=0.0,
                   help="Ignored — BCE loss has no hinge margin")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str,
                   default="analysis/logo_legacy_loop_bce")
    return p.parse_args()


def load_data(args):
    if args.h5 is not None:
        expr_path = Path(args.h5)
        if not expr_path.is_absolute():
            expr_path = PROJECT_ROOT / expr_path
        if args.meta is not None:
            meta_path = Path(args.meta)
            if not meta_path.is_absolute():
                meta_path = PROJECT_ROOT / meta_path
        else:
            # Default: most gene-axis ablations share the baseline metadata
            # Sample-axis ablations (e.g. combined_WT_only.h5) need --meta
            # explicitly — passing this default would silently mismatch rows
            meta_path = PROJECT_ROOT / "data" / "combined_metadata.csv"
    else:
        prefix    = "nc_" if args.no_combat else ""
        expr_path = PROJECT_ROOT / "data" / f"{prefix}combined_expression.h5"
        meta_path = PROJECT_ROOT / "data" / f"{prefix}combined_metadata.csv"

    if not expr_path.exists():
        sys.exit(f"ERROR {expr_path} not found")
    if not meta_path.exists():
        sys.exit(f"ERROR {meta_path} not found")

    print(f"Loading expression : {expr_path}")
    with h5py.File(expr_path, "r") as f:
        X             = f["X"][:].astype(np.float32)
        sample_ids_h5 = np.array(f["sample_ids"]).astype(str)

    print(f"Loading metadata   : {meta_path}")
    meta_raw = pd.read_csv(meta_path)
    if "sample_id" in meta_raw.columns:
        # common silent failure when --h5 and --meta don't match
        h5_set  = set(sample_ids_h5)
        csv_set = set(meta_raw["sample_id"])
        missing = h5_set - csv_set
        if missing:
            sys.exit(
                f"ERROR! H5 has {len(missing)} sample_ids not in metadata CSV"
                f"First few: {sorted(missing)[:3]}. "
                f"If using a sample-axis ablation H5 (e.g. combined_WT_only.h5), "
                f"pass --meta data/combined_WT_only_metadata.csv"
            )
        meta = (meta_raw.set_index("sample_id")
                        .loc[sample_ids_h5]
                        .reset_index())
    else:
        if len(meta_raw) != len(X):
            sys.exit(
                f"ERROR! CSV has no 'sample_id' column and row count "
                f"({len(meta_raw)}) doesn't match H5 ({len(X)})."
            )
        meta = meta_raw.reset_index(drop=True)

    if "genotype_norm" not in meta.columns:
        wt_tokens = {"wt", "wildtype", "bl6", "3xtgadwt"}
        meta["genotype_norm"] = meta["genotype"].apply(
            lambda g: "WT" if any(t in str(g).lower() for t in wt_tokens) else "AD"
        )

    print(f"\nDataset: {len(X)} samples × {X.shape[1]} genes")
    return X, meta


def build_group_labels(meta: pd.DataFrame) -> np.ndarray:
    key = (meta["dataset"].astype(str) + "|" +
           meta["genotype_norm"].astype(str) + "|" +
           meta["age_months"].astype(str) + "|" +
           meta["sex"].str.lower().astype(str))
    codes, uniques = pd.factorize(key)
    print(f"\nTotal groups: {len(uniques)}")
    return codes


def make_model(args, n_genes: int, device: torch.device) -> nn.Module:
    model = MouseClockCNN_BCE(
        n_genes = n_genes,
        n_models = len(DATASET_TO_ID),
        lambda_age = args.lambda_age,
    )
    return model.to(device)


def train_one_fold(X_tr, meta_tr, args, n_genes, device):
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
    X_t  = torch.tensor(X_tr, dtype=torch.float32)

    N = len(X_t)
    model.train()
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, N, args.batch):
            idx = perm[start : start + args.batch]
            xb = X_t[idx].to(device)
            db = disease_t[idx].to(device)
            ab = age_t[idx].to(device)
            mb = model_t[idx].to(device)
            sb = sex_t[idx].to(device)

            x_in = xb.unsqueeze(1)
            optimizer.zero_grad()
            d_pred, a_pred = model(x_in, mb, sb)

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


@torch.no_grad()
def eval_fold(model, X_te, meta_te, args, device):
    model.eval()
    X_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    mb  = torch.tensor(
        meta_te["dataset"].map(DATASET_TO_ID).fillna(0).values.astype(np.int64),
        dtype=torch.long).to(device)
    sb  = torch.tensor(
        meta_te["sex"].str.lower().map({"male": 0, "female": 1}).fillna(0).values.astype(np.int64),
        dtype=torch.long).to(device)
    x_in = X_t.unsqueeze(1)
    d_pred, _ = model(x_in, mb, sb)

    preds = np.atleast_1d(d_pred.squeeze().cpu().numpy())
    trues = np.atleast_1d(meta_te["disease_score"].values.astype(np.float32))
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    if len(trues) > 2 and len(np.unique(trues)) > 1:
        r, pval = pearsonr(trues, preds)
    else:
        r, pval = float("nan"), float("nan")
    return preds, trues, mse, mae, float(r), float(pval)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    print(f"Model : MouseClockCNN_BCE (legacy arch + BCE loss)")
    print(f"Epochs: {args.epochs}  lr={args.lr}  batch={args.batch}  "
          f"lambda_age={args.lambda_age}")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, meta  = load_data(args)
    n_genes  = X.shape[1]
    groups   = build_group_labels(meta)

    probe = make_model(args, n_genes, device)
    total = sum(p.numel() for p in probe.parameters())
    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable")
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logo    = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(X, groups=groups)
    print(f"\nRunning {n_folds}-fold LOGO-CV (legacy loop, BCE loss) …\n")

    all_results = []
    all_preds   = []
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
        preds, trues, mse, mae, r, pval = eval_fold(model, X_te, meta_te, args, device)
        elapsed = time.time() - t0

        naive_mse = float(np.mean(trues ** 2))
        print(f" MSE={mse:.4f}  naive={naive_mse:.4f}  "
              f"MAE={mae:.4f}  r={r:.3f}  ({elapsed:.0f}s)")

        all_results.append({
            "fold": fold_idx, "group": g_label,
            "dataset": meta_te["dataset"].iloc[0],
            "genotype_norm": meta_te["genotype_norm"].iloc[0],
            "age_months": meta_te["age_months"].iloc[0],
            "sex": meta_te["sex"].iloc[0].lower(),
            "n_test": len(test_idx),
            "mse": mse, "naive_mse": naive_mse,
            "beats_baseline": int(mse < naive_mse),
            "mae": mae, "pearson_r": r, "pearson_p": pval,
            "train_time_s": elapsed,
        })

        fold_pred_df = meta_te[
            ["dataset", "genotype_norm", "age_months", "sex", "disease_score"]
        ].copy()
        fold_pred_df["sample_id"] = (meta_te["sample_id"].values
                                      if "sample_id" in meta_te.columns
                                      else np.arange(len(meta_te)))
        fold_pred_df["pred_disease"] = preds
        fold_pred_df["fold"] = fold_idx
        all_preds.append(fold_pred_df)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    preds_df   = pd.concat(all_preds, ignore_index=True)

    oof_mse = float(np.mean((preds_df["pred_disease"] -
                              preds_df["disease_score"]) ** 2))
    oof_mae = float(np.mean(np.abs(preds_df["pred_disease"] -
                                    preds_df["disease_score"])))
    oof_r, oof_p = pearsonr(preds_df["disease_score"], preds_df["pred_disease"])
    n_beats = int(results_df["beats_baseline"].sum())

    print("\n" + "="*60)
    print("LOGO-CV Summary (legacy loop, MouseClockCNN_BCE)")
    print("="*60)
    print(f"Genes: {n_genes}")
    print(f"Epochs/fold: {args.epochs}")
    print(f"Overall OOF MSE: {oof_mse:.4f}")
    print(f"Overall OOF MAE: {oof_mae:.4f}")
    print(f"Overall OOF r: {oof_r:.3f}  (p={oof_p:.3e})")
    print(f"Folds beating naive: {n_beats}/{n_folds}")

    results_df.to_csv(out_dir / "logo_cv_results.csv", index=False)
    preds_df.to_csv(out_dir / "logo_cv_predictions.csv", index=False)

    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
        y = (preds_df["disease_score"] > 0).astype(int)
        p = preds_df["pred_disease"]
        auroc = roc_auc_score(y, p)
        acc   = accuracy_score(y, (p >= 0.5).astype(int))
        print(f"\nOOF AUROC:    {auroc:.4f}")
        print(f"OOF accuracy: {acc:.4f}")
        print(f"AD pred mean: {p[y==1].mean():.3f}")
        print(f"WT pred mean: {p[y==0].mean():.3f}")
    except Exception as e:
        print(f"\n[WARN] AUROC failed: {e}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
