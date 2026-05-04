"""
train.py
========

Train ONE v31-family model from a fixed train/val/test split

Used for:
  - Producing a single trained checkpoint for downstream interpretation
    (SHAP, saliency, GradCAM, attention inspection) which require ONE model
    to explain
  - Quick diagnostic runs


Usage
-----
python train.py --arch hybrid  # defaults, 17518-gene CNN
python train.py --arch mlp --epochs 100 # MLP baseline
python train.py --arch hybrid --no-combat # use nc_combined_*.h5

Outputs
-------
checkpoints/best_model_<arch>.pt # best val-loss checkpoint
analysis/training/<arch>/loss_curves.png
analysis/training/<arch>/training_log.csv
analysis/training/<arch>/test_predictions.csv
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from data.dataset import MouseClockDataset, DATASET_TO_ID
from model.training import (
    setup_determinism,
    train_one_model,
    TrainingConfig,
    _run_epoch,  # for test-set eval
)
import eval_hybrid_common as ehc


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def parse_args():
    p = argparse.ArgumentParser(
        description="Train one v31 model for SHAP / interpretation."
    )
    p.add_argument(
        "--arch",
        choices=ehc.ARCH_CHOICES,
        default="hybrid",
        help="Architecture shorthand. See eval_hybrid_common.ARCH_TO_MODEL_FILE",
    )
    p.add_argument(
        "--hybrid-model",
        default="model/cnn_clock_hybrid_v31.py",
        help="Path to a v31-compatible model file (overrides --arch when set)",
    )
    p.add_argument(
        "--coord-map",
        default="data/chrom_coord_map_with_symbols.csv",
        help="Coordinate map (use NONE to disable; ignored by non-hybrid arches)",
    )
    p.add_argument(
        "--gene-order",
        default="data/chrom_gene_order.txt",
        help="Gene-order file (use NONE to disable; ignored by non-hybrid arches)",
    )

    # Data
    p.add_argument("--h5", default="data/combined_expression.h5")
    p.add_argument("--meta", default="data/combined_metadata.csv")
    p.add_argument(
        "--no-combat", action="store_true",
        help="Use nc_combined_* files instead of ComBat-corrected",
    )

    # Training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lambda-age", dest="lambda_age", type=float, default=0.2)
    p.add_argument(
        "--delta", type=float, default=0.0,
        help="Retained for API compat; unused under symmetric MAE",
    )
    p.add_argument(
        "--noise-std", type=float, default=0.02,
        help="Gaussian noise augmentation on training x",
    )
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="Strict cuDNN determinism")

    # Embedding ablation
    p.add_argument("--use-dataset-embedding", action="store_true",
                   help="Enable explicit dataset/model embedding")
    p.add_argument("--disable-dataset-embedding", action="store_true",
                   help="Runtime zero-out ablation")

    # Output
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument(
        "--out-dir", default=None,
        help="Output dir. Defaults to analysis/training/<arch>/",
    )

    # Smoke testing
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick stratified-subset run; caps epochs and disables checkpointing",
    )
    p.add_argument("--smoke-n", type=int, default=48)

    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f"analysis/training/{args.arch}"
    return args


# Plotting

def plot_loss_curves(log_df, out_path, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=13)
    for ax, col, title in zip(
        axes,
        ["total", "disease", "age"],
        ["Total loss", "Disease score loss", "Age loss (normalised)"],
    ):
        ax.plot(log_df["epoch"], log_df[f"train_{col}"], label="train")
        ax.plot(log_df["epoch"], log_df[f"val_{col}"], label="val")
        best_epoch = log_df.loc[log_df["val_total"].idxmin(), "epoch"]
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                   label=f"best (ep {best_epoch})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss curves -> {out_path}")


def main():
    args = parse_args()
    setup_determinism(args.seed, deterministic=args.deterministic)

    # Resolve data paths
    if args.no_combat:
        h5 = "data/nc_combined_expression.h5"
        meta = "data/nc_combined_metadata.csv"
        print("Using no-ComBat data (nc_combined_*)")
    else:
        h5 = args.h5
        meta = args.meta

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"Arch:   {args.arch}")

    # Stratified train/val/test split via dataset's splitter
    full_ds = MouseClockDataset(h5, meta, noise_std=0.0)
    train_idx, val_idx, test_idx = full_ds.get_splits(seed=args.seed)
    n_genes = full_ds.get_gene_count()

    # Smoke-test subsetting
    if args.smoke_test:
        meta_df = full_ds.meta.iloc[train_idx]
        if "strat_key" in meta_df.columns:
            groups = meta_df["strat_key"].values
        else:
            groups = (meta_df["dataset"].astype(str) + "_" +
                      meta_df.get("genotype_norm",
                                  meta_df.get("genotype", "")).astype(str)).values
        rng = np.random.default_rng(args.seed)
        unique_groups = np.unique(groups)
        n_per_group = max(1, args.smoke_n // len(unique_groups))
        keep_local = []
        for g in unique_groups:
            idx_local = np.where(groups == g)[0]
            chosen = rng.choice(idx_local,
                                size=min(len(idx_local), n_per_group),
                                replace=False)
            keep_local.extend(chosen.tolist())
        keep_local = np.array(sorted(set(keep_local)))[:args.smoke_n]
        train_idx = train_idx[keep_local]
        print(f"\n[SMOKE TEST] subset n={len(train_idx)}\n")
        args.epochs = min(args.epochs, 3)

    # Build model via the shared factory (same as eval scripts)
    factory_args = ehc.factory_args_from_namespace(
        args,
        use=bool(args.use_dataset_embedding),
        disable=bool(args.disable_dataset_embedding),
    )
    model, hybrid_path = ehc.make_model(
        factory_args, n_genes, device, PROJECT_ROOT,
        n_models=len(DATASET_TO_ID),
    )
    print(f"Model file: {hybrid_path.name}")

    if hasattr(model, "count_parameters"):
        total_params, trainable_params = model.count_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # Training config
    cfg = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        patience=args.patience,
        lambda_age=args.lambda_age,
        delta=args.delta,
        verbose=True,
        track_log=True,
    )

    print()
    model, train_log, best_val_loss = train_one_model(
        model, h5, meta, train_idx, val_idx, cfg, device,
    )

    # Save log + curves
    log_df = pd.DataFrame(train_log)
    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_df.to_csv(log_path, index=False)

    curve_path = os.path.join(args.out_dir, "loss_curves.png")
    plot_loss_curves(log_df, curve_path, title_suffix=args.arch.upper())

    # Save checkpoint
    ckpt_name = f"best_model_{args.arch}.pt"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    if not args.smoke_test:
        torch.save({
            "model_state": model.state_dict(),
            "val_loss": best_val_loss,
            "args": vars(args),
            "arch": args.arch,
            "n_genes": n_genes,
            "n_models": len(DATASET_TO_ID),
        }, ckpt_path)
        print(f"\nBest val loss: {best_val_loss:.4f}  (checkpoint: {ckpt_path})")
    else:
        print(f"\n[SMOKE TEST] best val loss: {best_val_loss:.4f}  (no checkpoint)")
    print(f"Training log: {log_path}")

    # Test set evaluation
    test_ds = MouseClockDataset(h5, meta, indices=test_idx, noise_std=0.0)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_metrics = _run_epoch(
        model, test_loader, device, optimizer=None, delta=args.delta,
    )
    print(f" test total loss: {test_metrics['total']:.4f}")
    print(f" test disease loss: {test_metrics['disease']:.4f}")
    print(f" test age loss: {test_metrics['age']:.4f}")

    id_to_name = {v: k for k, v in DATASET_TO_ID.items()}
    if hasattr(model, "embedding_distance_matrix"):
        emb = model.embedding_distance_matrix(id_to_name)
        if len(emb["names"]) > 0:
            print(f"\nModel embedding cosine similarity (trained):")
            print(f" Models: {emb['names']}")
            print(f" {emb['cosine_similarity'].round(3)}")

    # Per-sample test predictions
    model.eval()
    all_preds, all_true, all_age, all_model = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, disease, age, model_id, sex_id = batch
            x = x.to(device)
            model_id = model_id.to(device)
            sex_id = sex_id.to(device)
            d_pred, _ = model(x, model_id, sex_id)
            all_preds.append(d_pred.cpu())
            all_true.append(disease)
            all_age.append(age)
            all_model.append(model_id.cpu())

    preds = torch.cat(all_preds).numpy()
    true = torch.cat(all_true).numpy()
    ages = torch.cat(all_age).numpy()
    mids = torch.cat(all_model).numpy()

    pred_df = pd.DataFrame({
        "disease_true": true,
        "disease_pred": preds,
        "age_months": ages,
        "dataset": [id_to_name.get(m, str(m)) for m in mids],
    })
    pred_path = os.path.join(args.out_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\nTest predictions saved -> {pred_path}")

    if true.std() > 0 and preds.std() > 0:
        corr = np.corrcoef(true, preds)[0, 1]
        mse = np.mean((true - preds) ** 2)
        print(f"  Pearson r: {corr:.3f}")
        print(f"  MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
