"""
model/training.py

Training procedure for MouseClockCNN/MouseClockMLP

Used by:
  - train.py --> single train/val/test split (one trained checkpoint)
  - evaluate_LOGO.py --> one model per LOGO fold
  - scripts/train_ensemble.py --> multi-seed ensembles

This module contains ONE training function so that both train.py and
evaluate_LOGO.py produce models trained identically. (Previously they had
divergent loops, which caused the LOGO OOF r to drop ~0.14 vs the train.py
ensemble result)

Features (all callers get them):
  - WeightedRandomSampler on inverse-frequency strat_key weights
  - Gaussian noise augmentation on training x (configurable)
  - Per-epoch validation
  - Early stopping with patience
  - ReduceLROnPlateau driven by val loss (NOT train loss)
  - Best-val-loss checkpoint kept in memory
  - Gradient clipping max_norm=1.0
  - genotype_norm = (disease_score > 0).long() to activate symmetric MAE
  - Determinism setup helper

"""

import copy
import os
import random
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data.dataset import MouseClockDataset, DATASET_TO_ID, SEX_TO_ID


# Determinism setup

def setup_determinism(seed: int, deterministic: bool = False):
    """
    Single canonical determinism setup. Called once at script start (optionally per-fold)

    Parameters
    ----------
    seed : RNG seed for python, numpy, torch (CPU and CUDA)
    deterministic : if True, also enable strict cudnn determinism. This makes CUDA 
    runs bit-reproducible at some performance cost
    On CPU, runs are deterministic without this flag
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# Training config
@dataclass
class TrainingConfig:
    epochs: int   = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    noise_std: float = 0.02 # Gaussian augmentation on train x
    patience: int = 25 # early stopping; set 0 to disable
    lambda_age: float = 0.2
    delta: float = 0.0 # retained for API compat; unused
    grad_clip: float = 1.0
    lr_factor: float = 0.5
    lr_patience: int  = 10
    verbose: bool  = True
    track_log: bool  = False # if True, return per-epoch metrics


# DataLoader builders for arbitrary index split
def _build_loaders_from_indices(
    h5_path: str,
    meta_path: str,
    train_idx: np.ndarray,
    val_idx:   np.ndarray,
    cfg: TrainingConfig,
):
    """
    Build train & val DataLoaders given explicit row indices

    The training loader uses WeightedRandomSampler with inverse-frequency
    weights based on strat_key. This is the same logic build_loaders() uses
    in train.py — but exposed at the index level so evaluate_LOGO.py can
    use it for arbitrary LOGO fold splits
    """
    train_ds = MouseClockDataset(h5_path, meta_path,
                                 indices=train_idx,
                                 noise_std=cfg.noise_std)
    val_ds   = MouseClockDataset(h5_path, meta_path,
                                 indices=val_idx,
                                 noise_std=0.0)

    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    # drop_last=True on training loader: prevents BatchNorm crash when the
    # final batch has size 1 (BN needs >1 sample per channel in train mode).
    # Loses up to (batch_size - 1) samples per epoch in the worst case, but
    # the WeightedRandomSampler will resample them next epoch anyway
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


# One epoch

def _run_epoch(model, loader, device, optimizer=None,
               delta: float = 0.0, grad_clip: float = 1.0):
    """
    Run one train or eval epoch

    Returns dict of mean losses: {'total', 'disease', 'age'}
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_sum = disease_sum = age_sum = 0.0
    n_batches = 0

    # Detect MLP by class name so can squeeze the channel dim from the
    # dataset's (B, 1, G) output. CNN models expect (B, 1, G); MLP models
    # expect (B, G). Need to check the class hierarchy via __class__.__name__
    # so this works for MouseClockMLP (and any future subclass) without
    # requiring a circular import here
    is_mlp = "MLP" in type(model).__name__

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            x, disease, age, model_id, sex_id = batch
            x = x.to(device)
            disease = disease.to(device)
            age = age.to(device)
            model_id = model_id.to(device)
            sex_id = sex_id.to(device)

            if is_mlp:
                x = x.squeeze(1) # (B, 1, G) --> (B, G)

            d_pred, a_pred = model(x, model_id, sex_id)
            # genotype_norm = AD if disease_score > 0 else WT
            # this activates the symmetric-MAE branching in compute_loss
            genotype_norm = (disease > 0).long()

            total, d_loss, a_loss = model.compute_loss(
                d_pred, a_pred, disease, age,
                genotype_norm=genotype_norm, delta=delta)

            if is_train:
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            total_sum += total.item()
            disease_sum += d_loss.item()
            age_sum += a_loss.item()
            n_batches += 1

    return {
        "total": total_sum / n_batches,
        "disease": disease_sum / n_batches,
        "age": age_sum / n_batches,
    }


# Main entry point
def train_one_model(
    model: nn.Module,
    h5_path: str,
    meta_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainingConfig,
    device: torch.device,
):
    """
    Train ONE model, with the canonical procedure 
    Returns the best-val-loss weights restored into the model

    This is the function evaluate_LOGO.py and train.py both call — single
    source of truth for "how does the mouse AD clock train?"

    Parameters
    ----------
    model : freshly-initialised MouseClockCNN or MouseClockMLP
    h5_path : path to the expression h5
    meta_path : path to the metadata CSV
    train_idx : row indices to use for training
    val_idx : row indices to use for validation (early stopping signal)
    cfg : TrainingConfig
    device : torch device

    Returns
    -------
    model : with best-val-loss weights restored
    train_log : list of dicts, one per epoch, with train+val losses (or None if cfg.track_log is False)
    best_val : float, best validation total loss seen
    """
    train_loader, val_loader = _build_loaders_from_indices(
        h5_path, meta_path, train_idx, val_idx, cfg)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)

    best_val_loss = float("inf")
    best_state_dict = None
    patience_count = 0
    train_log = [] if cfg.track_log else None

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer=optimizer, delta=cfg.delta, grad_clip=cfg.grad_clip)
        val_metrics   = _run_epoch(model, val_loader, device, optimizer=None, delta=cfg.delta)

        # NOTE: scheduler is driven by VAL loss, not train loss
        scheduler.step(val_metrics["total"])

        if cfg.track_log:
            train_log.append({
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "train_disease": train_metrics["disease"],
                "train_age": train_metrics["age"],
                "val_total": val_metrics["total"],
                "val_disease": val_metrics["disease"],
                "val_age": val_metrics["age"],
                "lr": optimizer.param_groups[0]["lr"],
            })

        improved = val_metrics["total"] < best_val_loss
        if improved:
            best_val_loss   = val_metrics["total"]
            # deep-copy state dict so subsequent in-place updates don't corrupt it
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_count  = 0
        else:
            patience_count += 1

        if cfg.verbose and (epoch % 10 == 0 or epoch == 1):
            marker = " good " if improved else ""
            print(f" epoch {epoch:3d}/{cfg.epochs} "
                  f"train={train_metrics['total']:.4f} "
                  f"val={val_metrics['total']:.4f}{marker}")

        if cfg.patience > 0 and patience_count >= cfg.patience:
            if cfg.verbose:
                print(f" early stopping at epoch {epoch} "
                      f"(no val improvement for {cfg.patience} epochs)")
            break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, train_log, best_val_loss


# Convenience: build train/val split from train_idx for LOGO callers
def split_train_for_validation(
    train_idx: np.ndarray,
    meta: pd.DataFrame,
    val_frac: float = 0.15,
    seed: int = 42,
):
    """
    Helper for evaluate_LOGO.py: given the LOGO outer-fold's train_idx, split
    it further into inner train + validation, holding out groups (strat_key)
    so that no group leaks across the inner split

    This gives the training loop a validation signal for early stopping
    without ever touching the held-out LOGO test fold

    Parameters
    ----------
    train_idx : row indices for the LOGO outer fold's train set
    meta : full metadata DataFrame
    val_frac : fraction of GROUPS (not samples) held out for val
    seed : random seed

    Returns
    -------
    inner_train_idx, val_idx : numpy int arrays, both relative to the original metadata, NOT relative to train_idx
    """
    from sklearn.model_selection import GroupShuffleSplit

    sub_meta = meta.iloc[train_idx].reset_index(drop=True)

    if "strat_key" in sub_meta.columns:
        groups = sub_meta["strat_key"].values
    else:
        # fallback if strat_key wasn't preserved by upstream
        groups = (sub_meta["dataset"].astype(str) + "|"
                  + sub_meta["genotype_norm"].astype(str) + "|"
                  + sub_meta["age_months"].astype(str) + "|"
                  + sub_meta["sex"].str.lower().astype(str)).values

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    inner_local, val_local = next(gss.split(np.arange(len(sub_meta)), groups=groups))

    inner_train_idx = train_idx[inner_local]
    val_idx         = train_idx[val_local]

    return inner_train_idx, val_idx
