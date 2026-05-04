"""
train_HVG.py

Train ONE model on the HVG-filtered (~2,000 gene) data

Usage
-----
python train_HVG.py --arch mlp  # MLP on HVGs
python train_HVG.py --arch hybrid # Hybrid on HVGs (with chrom-aware features OFF by default)
python train_HVG.py --arch hybrid \\
    --coord-map data/chrom_coord_map_with_symbols.csv \\
    --gene-order data/chrom_gene_order.txt  # opt back into chromosome features

Outputs
-------
analysis/training_HVG/<arch>/{loss_curves.png, training_log.csv, test_predictions.csv}
checkpoints/hvg_<arch>_best.pt
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
    _run_epoch,
)
import eval_hybrid_common as ehc


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Train one v31 model on HVG-filtered data")
    # Architecture dispatch
    p.add_argument(
        "--arch",
        choices=ehc.ARCH_CHOICES,
        default="mlp",
        help="Architecture shorthand (default: mlp for HVG comparison)",
    )
    p.add_argument(
        "--hybrid-model",
        default="model/cnn_clock_hybrid_v31.py",
        help="Path to a v31-compatible model file (overrides --arch when set)",
    )

    # HVG defaults to NONE for chromosome metadata. HVG selection breaks
    # contiguous chromosomal order (the top 2000 variance genes are scattered
    # across chromosomes), so the chrom embedding's inductive bias doesn't
    # apply. Pass explicit paths to opt back in
    p.add_argument("--coord-map", default="NONE")
    p.add_argument("--gene-order", default="NONE")

    # Data — HVG paths by default
    p.add_argument("--h5", default="data/hvg_combined_expression.h5")
    p.add_argument("--meta", default="data/hvg_combined_metadata.csv")

    # Training — HVG-specific defaults
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--lr", type=float, default=5e-4,
        help="Lower default than full-gene CNN due to smaller feature space",
    )
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lambda-age", dest="lambda_age", type=float, default=0.2)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument(
        "--noise-std", type=float, default=0.05,
        help="Slightly higher than full-gene since the input is denser",
    )
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    # Embedding ablation
    p.add_argument("--use-dataset-embedding", action="store_true")
    p.add_argument("--disable-dataset-embedding", action="store_true")

    # Output
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument(
        "--out-dir", default=None,
        help="Output dir. Defaults to analysis/training_HVG/<arch>/",
    )

    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f"analysis/training_HVG/{args.arch}"
    return args


def plot_loss_curves(log_df, out_path, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if title_suffix:
        fig.suptitle(f"{title_suffix} on HVGs", fontsize=13)
    for ax, col, title in zip(
        axes,
        ["total", "disease", "age"],
        ["Total loss", "Disease score loss", "Age loss (normalised)"],
    ):
        ax.plot(log_df["epoch"], log_df[f"train_{col}"], label="train")
        ax.plot(log_df["epoch"], log_df[f"val_{col}"], label="val")
        best_ep = log_df.loc[log_df["val_total"].idxmin(), "epoch"]
        ax.axvline(best_ep, color="red", linestyle="--",
                   alpha=0.5, label=f"best (ep {best_ep})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curves -> {out_path}")


def main():
    args = parse_args()
    setup_determinism(args.seed, deterministic=args.deterministic)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device()
    print(f"Device:    {device}")
    print(f"Arch:      {args.arch} (HVG-filtered input)")
    print(f"Outputs -> {args.out_dir}")

    # Stratified split via dataset's splitter
    full_ds = MouseClockDataset(args.h5, args.meta, noise_std=0.0)
    train_idx, val_idx, test_idx = full_ds.get_splits(seed=args.seed)
    n_genes = full_ds.get_gene_count()
    print(f"Input genes: {n_genes}")

    # Build model via shared factory
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
        trainable_params = total_params
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # Train via pipeline
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
        model, args.h5, args.meta, train_idx, val_idx, cfg, device,
    )

    # Save log + curves
    log_df = pd.DataFrame(train_log)
    log_df.to_csv(os.path.join(args.out_dir, "training_log.csv"), index=False)
    plot_loss_curves(
        log_df,
        os.path.join(args.out_dir, "loss_curves.png"),
        title_suffix=args.arch.upper(),
    )
    print(f"\nBest val loss: {best_val_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, f"hvg_{args.arch}_best.pt")
    torch.save({
        "model_state": model.state_dict(),
        "val_loss": best_val_loss,
        "args": vars(args),
        "arch": args.arch,
        "n_genes": n_genes,
        "n_models": len(DATASET_TO_ID),
    }, ckpt_path)
    print(f"Checkpoint -> {ckpt_path}")

    # Test eval
    test_ds = MouseClockDataset(args.h5, args.meta, indices=test_idx, noise_std=0.0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_m = _run_epoch(model, test_loader, device, optimizer=None, delta=args.delta)
    print(f" test total: {test_m['total']:.4f}")
    print(f" test disease: {test_m['disease']:.4f}")
    print(f" test age: {test_m['age']:.4f}")

    # Per-sample predictions
    id_to_name = {v: k for k, v in DATASET_TO_ID.items()}
    model.eval()
    all_preds, all_true, all_age, all_mid = [], [], [], []
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
            all_mid.append(model_id.cpu())

    preds = torch.cat(all_preds).numpy()
    true = torch.cat(all_true).numpy()
    ages = torch.cat(all_age).numpy()
    mids = torch.cat(all_mid).numpy()

    pred_df = pd.DataFrame({
        "disease_true": true,
        "disease_pred": preds,
        "age_months": ages,
        "dataset": [id_to_name.get(m, str(m)) for m in mids],
    })
    pred_df.to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    if true.std() > 0 and preds.std() > 0:
        corr = np.corrcoef(true, preds)[0, 1]
        mse = np.mean((true - preds) ** 2)
        print(f"\n  Pearson r:  {corr:.3f}")
        print(f" Test MSE: {mse:.4f}")

    # Embedding distances
    if hasattr(model, "embedding_distance_matrix"):
        emb = model.embedding_distance_matrix(id_to_name)
        if len(emb["names"]) > 0:
            print(f"\nEmbedding cosine similarity ({args.arch.upper()}, trained):")
            print(f"  {emb['cosine_similarity'].round(3)}")


if __name__ == "__main__":
    main()
