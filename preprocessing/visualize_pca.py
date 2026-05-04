#!/usr/bin/env python3
"""
preprocessing/visualize_pca.py

Generates PCA diagnostic plots for preprocessed expression data, and 
detects both ComBat-corrected and no-ComBat versions and saves plots to analysis/pca/

Usage
-----
    python preprocessing/visualize_pca.py # both versions!
    python preprocessing/visualize_pca.py --combat-only
    python preprocessing/visualize_pca.py --no-combat-only

Output
------
    analysis/pca/pca_combat.png
    analysis/pca/pca_no_combat.png
"""

import argparse
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# NOTE 
# Both variants live in data/ with the following naming convention
# preprocess.py --no-combat writes nc_combined_*
# preprocess.py (default)  writes combined_*

VERSIONS = {
    "combat": {
        "h5": "data/combined_expression.h5",
        "meta": "data/combined_metadata.csv",
        "out": "analysis/pca/pca_combat.png",
        "tag": "ComBat-corrected",
    },
    "no_combat": {
        "h5": "data/nc_combined_expression.h5",
        "meta": "data/nc_combined_metadata.csv",
        "out": "analysis/pca/pca_no_combat.png",
        "tag": "No batch correction (baseline)",
    },
}

# PCA graph coloring
AGE_PALETTE = {4: "#3B0064", 8: "#1D6FA5", 12: "#2DC5A2", 18: "#F5E642"}

def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_ * 100
    return pcs, var_explained


def make_pca_figure(meta, var_explained, tag):
    """
    3-panel PCA: dataset x age x genotype
    Returns: matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(tag, fontsize=13, y=1.02)

    pc1_label = f"PC1 ({var_explained[0]:.1f}%)"
    pc2_label = f"PC2 ({var_explained[1]:.1f}%)"

    # Panel 1: dataset 
    dataset_palette = {"5xFAD": "#4C9BE8", "3xTgAD": "#F5A54A"}
    sns.scatterplot(
        data=meta, x="PC1", y="PC2",
        hue="dataset", palette=dataset_palette,
        ax=axes[0], alpha=0.75, s=45, edgecolors="white", linewidths=0.3,
    )
    axes[0].set_title("Color by dataset", fontsize=11)
    axes[0].set_xlabel(pc1_label); axes[0].set_ylabel(pc2_label)

    # Panel 2: age 
    ages = sorted(meta["age_months"].unique())
    age_pal = {a: AGE_PALETTE.get(a, "#888888") for a in ages}
    sns.scatterplot(
        data=meta, x="PC1", y="PC2",
        hue="age_months", palette=age_pal,
        hue_order=ages,
        ax=axes[1], alpha=0.75, s=45, edgecolors="white", linewidths=0.3,
    )
    axes[1].set_title("Color by age (months)", fontsize=11)
    axes[1].set_xlabel(pc1_label); axes[1].set_ylabel(pc2_label)

    # Panel 3: genotype
    geno_palette = {"AD": "#4C9BE8", "WT": "#F5A54A"}
    sns.scatterplot(
        data=meta, x="PC1", y="PC2",
        hue="genotype_norm", palette=geno_palette,
        ax=axes[2], alpha=0.75, s=45, edgecolors="white", linewidths=0.3,
    )
    axes[2].set_title("Color by genotype", fontsize=11)
    axes[2].set_xlabel(pc1_label); axes[2].set_ylabel(pc2_label)

    plt.tight_layout()
    return fig


def process_version(key, cfg):
    h5_path = cfg["h5"]
    meta_path = cfg["meta"]
    out_path = cfg["out"]
    tag = cfg["tag"]

    if not os.path.exists(h5_path):
        print(f" SKIP! {h5_path} not found — run preprocess.py first")
        return
    if not os.path.exists(meta_path):
        print(f" SKIP! {meta_path} not found — run preprocess.py first")
        return

    print(f"\nProcessing: {tag}")
    print(f" h5 : {h5_path}")
    print(f" meta : {meta_path}")

    with h5py.File(h5_path, "r") as f:
        X = f["X"][:]

    meta = pd.read_csv(meta_path)
    print(f" Loaded: {X.shape[0]} samples × {X.shape[1]} genes")

    pcs, var_explained = run_pca(X)
    meta["PC1"] = pcs[:, 0]
    meta["PC2"] = pcs[:, 1]

    print(f" PC1: {var_explained[0]:.1f}%  PC2: {var_explained[1]:.1f}%")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = make_pca_figure(meta, var_explained, tag)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--combat-only",    action="store_true",
                       help="Only plot the ComBat-corrected version")
    group.add_argument("--no-combat-only", action="store_true",
                       help="Only plot the no-correction baseline")
    args = parser.parse_args()

    if args.combat_only:
        to_run = ["combat"]
    elif args.no_combat_only:
        to_run = ["no_combat"]
    else:
        to_run = list(VERSIONS.keys())

    for key in to_run:
        process_version(key, VERSIONS[key])

    print("\nDone.")


if __name__ == "__main__":
    main()
