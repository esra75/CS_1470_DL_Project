#!/usr/bin/env python3
"""
preprocessing/preprocess_HVG.py

Selects top N highly variable genes (HVGs) from the QC'd, ComBat-corrected
expression matrix, using variance computed on the no-ComBat matrix to avoid
selecting genes whose variance is driven by residual batch structure

Pipeline Workflow
1. Load combined_expression.h5 (ComBat) and nc_combined_expression.h5 (no ComBat)
2. **Strict cross-file consistency check** — same samples, same genes, same QC
3. Compute per-gene variance on the no-ComBat matrix
4. Select top N HVGs by variance
5. Apply HVG mask to ComBat-corrected matrix
6. Write hvg_combined_expression.h5 with provenance attributes
7. Write provenance JSON

Outputs
data/hvg_combined_expression.h5
data/hvg_combined_metadata.csv
data/hvg_gene_list.txt
data/hvg_combined_expression.provenance.json
data/qc/hvg_selection_<timestamp>.csv (HVG ranks + variance + chromosome)

Usage
python preprocessing/preprocess_HVG.py # default 2000 HVGs
python preprocessing/preprocess_HVG.py --n-hvgs 5000
python preprocessing/preprocess_HVG.py --plot
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "preprocessing") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))

from _qc_lib import write_provenance, file_metadata, hash_array

PREPROCESS_HVG_VERSION = "v3_2026-04-27" #updated 


def load_h5(path: Path):
    """Read X, gene_names, sample_ids, attrs from an h5"""
    with h5py.File(path, "r") as f:
        return {
            "X": f["X"][:],
            "gene_names": np.array(f["gene_names"]).astype(str),
            "sample_ids": np.array(f["sample_ids"]).astype(str),
            "attrs": dict(f.attrs),
        }


def assert_consistent(combat_data, nc_data, combat_path, nc_path):
    """
    Strict consistency check!! Catches the failure mode of files getting out of
    sync (e.g., one preprocessed with --keep-mt and the other without)
    """
    print(f"\n{'='*60}")
    print("Cross-file consistency check")
    print(f"{'='*60}")

    # make sure same number of samples
    if len(combat_data["sample_ids"]) != len(nc_data["sample_ids"]):
        sys.exit(
            f" FATAL!! Sample count differs:\n"
            f"  {combat_path}: {len(combat_data['sample_ids'])}\n"
            f"  {nc_path}:     {len(nc_data['sample_ids'])}\n"
            f"Re-run preprocess.py for both before HVG selection"
        )

    # Same sample IDs in same order
    if not np.array_equal(combat_data["sample_ids"], nc_data["sample_ids"]):
        sys.exit(
            f"FATAL Sample IDs differ between files. They must be in identical order.\n"
            f"Re-run preprocess.py for both"
        )
    print(f" {len(combat_data['sample_ids'])} sample IDs match in identical order")

    # Same gene names in same order
    if not np.array_equal(combat_data["gene_names"], nc_data["gene_names"]):
        sys.exit(
            f"FATAL! Gene names differ between {combat_path} and {nc_path}.\n"
            f" combat n_genes: {len(combat_data['gene_names'])}\n"
            f" nc n_genes: {len(nc_data['gene_names'])}\n"
            f"Re-run preprocess.py for both with the same QC settings"
        )
    print(f" {len(combat_data['gene_names'])} gene names match in identical order")

    # Critical attributes MUST match
    critical_attrs = ["mt_removed", "n_genes_after_qc", "low_expr_threshold_tpm",
                      "low_expr_pct", "filter_hemoglobin", "filter_ribosomal",
                      "filter_pseudogenes", "preprocess_version"]
    mismatches = []
    for attr in critical_attrs:
        c_val = combat_data["attrs"].get(attr, "MISSING")
        n_val = nc_data["attrs"].get(attr, "MISSING")
        # Convert numpy scalars
        if hasattr(c_val, "item"): c_val = c_val.item()
        if hasattr(n_val, "item"): n_val = n_val.item()
        # Decode bytes
        if isinstance(c_val, bytes): c_val = c_val.decode()
        if isinstance(n_val, bytes): n_val = n_val.decode()
        if c_val != n_val:
            mismatches.append(f" {attr}: combat={c_val!r}, nc={n_val!r}")

    if mismatches:
        sys.exit("FATAL! Critical attributes differ between files:\n" +
                 "\n".join(mismatches) +
                 "\nRe-run preprocess.py for both with identical args")
    print(f" Critical QC attributes match: {critical_attrs}")

    # Check & verify combat status: combat=True for the combat file, combat=False for nc
    combat_attr = combat_data["attrs"].get("combat", None)
    nc_combat_attr = nc_data["attrs"].get("combat", None)
    if hasattr(combat_attr, "item"): combat_attr = combat_attr.item()
    if hasattr(nc_combat_attr, "item"): nc_combat_attr = nc_combat_attr.item()
    if not combat_attr:
        sys.exit(f" FATAL! {combat_path} has combat=False. Expected the ComBat-corrected file here")
    if nc_combat_attr:
        sys.exit(f" FATAL! {nc_path} has combat=True. Expected the no-ComBat file here")
    print(f" ComBat status: combat={combat_attr}, nc={nc_combat_attr}")

    print(" All consistency checks passed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hvgs", type=int, default=2000)
    parser.add_argument("--combat-h5", default="data/combined_expression.h5")
    parser.add_argument("--nc-h5", default="data/nc_combined_expression.h5")
    parser.add_argument("--meta", default="data/combined_metadata.csv")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--coords", default="data/chrom_coord_map.csv",
                        help="Optional: chrom coord map for HVG annotation")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    combat_h5 = PROJECT_ROOT / args.combat_h5
    nc_h5 = PROJECT_ROOT / args.nc_h5


    print(f" preprocess_HVG.py {PREPROCESS_HVG_VERSION}")
    print(f" Timestamp: {timestamp}")
    print(f" combat h5: {combat_h5}")
    print(f" nc h5: {nc_h5}")
    print(f" n_hvgs: {args.n_hvgs}")

    # 1. Load
    if not combat_h5.exists():
        sys.exit(f"FATAL! {combat_h5} not found. Run preprocess.py first")
    if not nc_h5.exists():
        sys.exit(f"FATAL! {nc_h5} not found. Run preprocess.py --no-combat first")

    combat_data = load_h5(combat_h5)
    nc_data = load_h5(nc_h5)

    # 2. Consistency check 
    assert_consistent(combat_data, nc_data, combat_h5, nc_h5)

    X_combat   = combat_data["X"]
    X_nc       = nc_data["X"]
    gene_names = combat_data["gene_names"]
    sample_ids = combat_data["sample_ids"]

    n_samples, n_genes = X_combat.shape
    print(f"\n Working with: {n_samples} samples x {n_genes} genes")

    if args.n_hvgs > n_genes:
        sys.exit(f"FATAL! --n-hvgs={args.n_hvgs} > available genes ({n_genes})")

    # 3. Compute variance on no-ComBat matrix
    print(f"\n{'='*60}")
    print("Selecting HVGs by variance on no-ComBat matrix")
    print(f"{'='*60}")
    gene_var = np.var(X_nc, axis=0)
    print(f" Gene variance range: {gene_var.min():.4f} – {gene_var.max():.4f}")
    print(f" Mean variance: {gene_var.mean():.4f} median: {np.median(gene_var):.4f}")
    n_zero_var = int((gene_var == 0).sum())
    if n_zero_var > 0:
        print(f" WARNING! {n_zero_var} genes still have zero variance - should have been "
              "filtered by mandatory QC. Continuing anyway")

    # 4. Rank and select
    ranked_idx = np.argsort(gene_var)[::-1]
    hvg_idx = ranked_idx[:args.n_hvgs]
    hvg_idx_sorted = np.sort(hvg_idx) # preserve original gene order in HVG subset

    hvg_genes = gene_names[hvg_idx_sorted]
    print(f"\n Selected top {args.n_hvgs} HVGs by variance")
    print(f" Variance threshold at rank {args.n_hvgs}: "
          f"{gene_var[ranked_idx[args.n_hvgs - 1]]:.4f}")

    # 5. Apply mask to ComBat matrix
    X_hvg = X_combat[:, hvg_idx_sorted]
    print(f" X_hvg shape: {X_hvg.shape}")
    print(f" X_hvg mean: {X_hvg.mean():.4f} std: {X_hvg.std():.4f}")

    # 6. Optional HVG annotation table
    coords_path = PROJECT_ROOT / args.coords
    hvg_csv = qc_dir / f"hvg_selection_{timestamp}.csv"
    hvg_table = pd.DataFrame({
        "gene_id": gene_names[hvg_idx_sorted],
        "variance_no_combat": gene_var[hvg_idx_sorted],
        "rank_by_variance": [int(np.where(ranked_idx == i)[0][0]) + 1 for i in hvg_idx_sorted],
    })
    if coords_path.exists():
        coord_df = pd.read_csv(coords_path)
        if "ensembl_gene_id" in coord_df.columns:
            coord_df = coord_df.rename(columns={"ensembl_gene_id": "gene_id"})
        hvg_table = hvg_table.merge(
            coord_df[["gene_id", "chromosome_name", "start_position"]],
            on="gene_id", how="left"
        )
        print(f"\n HVG chromosomal distribution (top 5):")
        print("  " + str(hvg_table["chromosome_name"].value_counts().head().to_dict()))
    else:
        print(f"  [INFO] {coords_path} not found - HVG annotation will lack chromosomes")
    hvg_table = hvg_table.sort_values("rank_by_variance")
    hvg_table.to_csv(hvg_csv, index=False)
    print(f" HVG ranks saved: {hvg_csv}")

    # 7. Write outputs
    out_h5 = out_dir / "hvg_combined_expression.h5"
    out_meta  = out_dir / "hvg_combined_metadata.csv"
    out_genes = out_dir / "hvg_gene_list.txt"

    print(f"\n{'='*60}")
    print("Writing outputs:")
    print(f"{'='*60}")
    print(f" {out_h5}")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("X", data=X_hvg, dtype="float32",
                         compression="gzip", compression_opts=4)
        f.create_dataset("gene_names", data=hvg_genes.astype("S"),
                         compression="gzip")
        f.create_dataset("sample_ids", data=sample_ids.astype("S"),
                         compression="gzip")

        # Inherit attributes from the source combat h5
        for k, v in combat_data["attrs"].items():
            f.attrs[k] = v

        # HVG-specific attributes
        f.attrs["n_hvgs"] = args.n_hvgs
        f.attrs["hvg_filtered"] = True
        f.attrs["source_combat"] = str(combat_h5)
        f.attrs["source_nc"] = str(nc_h5)
        f.attrs["preprocess_hvg_version"] = PREPROCESS_HVG_VERSION
        f.attrs["X_sha256"] = hash_array(X_hvg)

    print(f" {out_meta}")
    meta = pd.read_csv(PROJECT_ROOT / args.meta)
    meta.to_csv(out_meta, index=False)

    print(f" {out_genes}")
    with open(out_genes, "w") as f:
        for g in hvg_genes:
            f.write(g + "\n")

    # 8. Variance plot (optional)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_dir = PROJECT_ROOT / "analysis" / "pca"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "hvg_variance.png"

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(np.sort(gene_var)[::-1], linewidth=0.8)
            ax.axvline(args.n_hvgs, color="red", linestyle="--",
                       label=f"HVG cutoff (n={args.n_hvgs})")
            ax.set_xlabel("Gene rank (by variance)")
            ax.set_ylabel("Variance")
            ax.set_title("Gene variance across samples (log1p z-scored)")
            ax.set_yscale("log")
            ax.legend()
            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f" Variance plot saved: {plot_path}")
        except Exception as e:
            print(f" WARNING! Plot generation failed: {e}")

    # 9. Provenance
    prov_path = out_dir / "hvg_combined_expression.provenance.json"
    write_provenance(
        prov_path,
        script_name=__file__,
        inputs={
            "combat_h5": file_metadata(combat_h5, include_hash=False),
            "nc_h5": file_metadata(nc_h5, include_hash=False),
            "metadata": file_metadata(PROJECT_ROOT / args.meta),
        },
        args=vars(args),
        outputs={
            "h5": file_metadata(out_h5),
            "meta": file_metadata(out_meta),
            "gene_list": file_metadata(out_genes),
            "hvg_table": file_metadata(hvg_csv),
        },
        extras={
            "preprocess_hvg_version": PREPROCESS_HVG_VERSION,
            "n_hvgs": args.n_hvgs,
            "n_input_genes": int(n_genes),
            "n_samples": int(n_samples),
            "variance_threshold": float(gene_var[ranked_idx[args.n_hvgs - 1]]),
            "X_sha256": hash_array(X_hvg),
            "x_min": float(X_hvg.min()),
            "x_max": float(X_hvg.max()),
            "x_mean": float(X_hvg.mean()),
            "x_std": float(X_hvg.std()),
        },
    )
    print(f"  {prov_path}")

    size_mb = os.path.getsize(out_h5) / 1e6
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f" HVG h5: {out_h5}  ({size_mb:.1f} MB)")
    print(f" HVG metadata: {out_meta}")
    print(f" HVG gene list: {out_genes}")
    print(f" Provenance: {prov_path}")
    print(f"\n  h5 attributes:")
    with h5py.File(out_h5, "r") as f:
        for k, v in dict(f.attrs).items():
            if k == "X_sha256":
                v = str(v)[:16] + "..."
            print(f"    {k:30s} = {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
