#!/usr/bin/env python3
"""
preprocessing/make_chrom_ablation_h5.py

Generate gene-axis ablation H5 files from a canonical baseline H5 by
filtering or permuting genes. Every output file:
- Has a freshly-computed X_sha256 attribute (NOT inherited from parent)
- Has all canonical preprocessing attrs (combat, mt_removed, dash_ps_filter,
etc.) preserved from parent
- Has new attrs documenting the ablation (mode, source, timestamp, seed).
- Has a matching .provenance.json sidecar

Modes:
Gene-axis filter modes (drop genes from the gene axis):
- no_sex                 drop chrX + chrY genes
- no_chr2               drop chr2 genes
- no_chr9               drop chr9 genes (the full chromosome — 938 genes)
- no_chr3_chr10         drop chr3 + chr10 genes (control for chr 9 size)
- no_thy1               drop ONLY Thy1 (single gene; ENSMUSG00000032011)
- no_thy1_neighborhood  drop Thy1 ± 25 chr9 neighbors (52 genes total)
- autosomes_only        keep chr1..chr19 only (drop X, Y, MT, _unmapped)
- autosomes_no_chr2     autosomes_only + drop chr2

Random-drop modes (drop a random gene set): 
random_drop_NNNN - drop a random subset of size NNNN; reproducible by seed

Permutation modes (same gene set, reordered axis):
random_order - shuffle gene-axis order with seed (deterministic)

All output files preserve sample_ids in original order. Only the gene
axis changes (or, for random_order, only its ordering)

Usage
python preprocessing/make_chrom_ablation_h5.py \\
      --mode no_chr2 \\
      --input data/combined_expression.h5 \\
      --output data/combined_no_chr2.h5 \\
      --coord-map data/chrom_coord_map.csv

python preprocessing/make_chrom_ablation_h5.py \\
      --mode no_chr9 \\
      --input data/combined_expression.h5 \\
      --output data/combined_no_chr9.h5

python preprocessing/make_chrom_ablation_h5.py \\
      --mode no_thy1 \\
      --input data/combined_expression.h5 \\
      --output data/combined_no_thy1.h5

python preprocessing/make_chrom_ablation_h5.py \\
      --mode no_thy1_neighborhood \\
      --input data/combined_expression.h5 \\
      --output data/combined_no_thy1_neighborhood.h5 \\
      --neighborhood-window 25

python preprocessing/make_chrom_ablation_h5.py \\
      --mode random_drop \\
      --input data/combined_expression.h5 \\
      --output data/combined_random_drop_1356_seedA.h5 \\
      --n-drop 1356 --seed 1

python preprocessing/make_chrom_ablation_h5.py \\
      --mode random_order \\
      --input data/combined_expression.h5 \\
      --output data/combined_random_order.h5 \\
      --seed 42
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SCRIPT_VERSION = "v2_2026-04-30"

# Thy1 Ensembl gene ID — the dominant feature in the disease head's SHAP
# This gene is the readout reporter for the 5xFAD transgene cassette
THY1_ENSEMBL_ID = "ENSMUSG00000032011"

VALID_GENE_FILTER_MODES = [
    "no_sex",
    "no_chr2",
    "no_chr9",
    "no_chr3_chr10",
    "no_thy1",
    "no_thy1_neighborhood",
    "autosomes_only",
    "autosomes_no_chr2",
]
VALID_RANDOM_MODES = ["random_drop", "random_order"]
VALID_MODES = VALID_GENE_FILTER_MODES + VALID_RANDOM_MODES


# Hashing
def hash_array(X: np.ndarray) -> str:
    """SHA256 of X bytes."""
    return hashlib.sha256(X.tobytes()).hexdigest()


# I/O helpers
def load_baseline(h5_path: Path):
    """Load X, gene_names, sample_ids, attrs from baseline h5."""
    with h5py.File(h5_path, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        gene_ids = np.array(f["gene_names"]).astype(str)
        sample_ids = np.array(f["sample_ids"]).astype(str)
        attrs = {k: f.attrs[k] for k in f.attrs}
    # strip version suffix on gene IDs (e.g. ENSMUSG00000032011.5 -> ENSMUSG00000032011)
    gene_ids = np.array([g.split(".")[0] for g in gene_ids])
    return X, gene_ids, sample_ids, attrs


def load_chrom_map(coord_map_path: Path):
    """Build {ensembl_id -> chromosome_name} and {ensembl_id -> position} dicts."""
    cm = pd.read_csv(coord_map_path)
    cm["ensembl_gene_id"] = cm["ensembl_gene_id"].astype(str).str.split(".").str[0]
    chrom = dict(zip(cm["ensembl_gene_id"], cm["chromosome_name"].astype(str)))
    pos_col = ("start_position" if "start_position" in cm.columns else
               "position"       if "position"       in cm.columns else
               None)
    pos = (dict(zip(cm["ensembl_gene_id"], cm[pos_col].astype(int)))
           if pos_col else None)
    return chrom, pos


# Gene-axis filter masks
def build_chrom_filter_mask(gene_ids, chrom_map, mode):
    """
    Build keep mask for a chromosome-based filter mode! 
    Returns (keep_bool_array, chroms_per_gene)
    """
    chroms = np.array([chrom_map.get(g, "_unmapped") for g in gene_ids])
    chroms_norm = np.array([c.lstrip("chr").upper() for c in chroms])

    if mode == "no_sex":
        keep = ~np.isin(chroms_norm, ["X", "Y"])
    elif mode == "no_chr2":
        keep = chroms_norm != "2"
    elif mode == "no_chr9":
        keep = chroms_norm != "9"
    elif mode == "no_chr3_chr10":
        keep = ~np.isin(chroms_norm, ["3", "10"])
    elif mode == "autosomes_only":
        autosome_set = {str(i) for i in range(1, 20)}
        keep = np.isin(chroms_norm, list(autosome_set))
    elif mode == "autosomes_no_chr2":
        autosome_set = {str(i) for i in range(1, 20)} - {"2"}
        keep = np.isin(chroms_norm, list(autosome_set))
    else:
        raise ValueError(f"Unknown chrom-filter mode: {mode}")

    return keep, chroms


def build_thy1_only_mask(gene_ids):
    """Drop ONLY Thy1, but keep everything else."""
    keep = gene_ids != THY1_ENSEMBL_ID
    n_dropped = int((~keep).sum())
    if n_dropped == 0:
        sys.exit(f"[ERROR] Thy1 ({THY1_ENSEMBL_ID}) not found in gene_names")
    if n_dropped > 1:
        sys.exit(f"[ERROR] Found {n_dropped} matches for Thy1 — gene IDs not unique?")
    return keep


def build_thy1_neighborhood_mask(gene_ids, chrom_map, pos_map, window):
    """
    Drop Thy1 + the N nearest chr9 genes by chromosomal position on either side
    `window` = N (default 25), so total dropped = 2*N + 1 = 51 (typically)
    """
    if pos_map is None:
        sys.exit("[ERROR] no_thy1_neighborhood requires a coord map with "
                 "'start_position' or 'position' column")
    if THY1_ENSEMBL_ID not in pos_map:
        sys.exit(f"[ERROR] Thy1 ({THY1_ENSEMBL_ID}) not in coord map")

    thy1_pos = pos_map[THY1_ENSEMBL_ID]
    thy1_chrom = chrom_map[THY1_ENSEMBL_ID].lstrip("chr").upper()

    # Find chr9 genes in the input (note: chr9 = 9 in normalized notation)
    chr9_ids = []
    for g in gene_ids:
        c = chrom_map.get(g, "_unmapped").lstrip("chr").upper()
        if c == thy1_chrom and g in pos_map:
            chr9_ids.append((g, pos_map[g]))

    # Sort chr9 genes by position
    chr9_sorted = sorted(chr9_ids, key=lambda t: t[1])

    # Find Thy1's index in the sorted list, then take ±window neighbors
    thy1_idx = next((i for i, (g, _) in enumerate(chr9_sorted)
                     if g == THY1_ENSEMBL_ID), None)
    if thy1_idx is None:
        sys.exit("[ERROR] Thy1 not found in chr9 input genes")
    lo = max(0, thy1_idx - window)
    hi = min(len(chr9_sorted), thy1_idx + window + 1)
    drop_set = {g for g, _ in chr9_sorted[lo:hi]}
    keep = np.array([g not in drop_set for g in gene_ids])
    return keep, drop_set


def build_random_drop_mask(n_genes, n_drop, seed):
    """Drop a random subset of size n_drop (deterministic by seed)."""
    if n_drop <= 0 or n_drop >= n_genes:
        sys.exit(f"[ERROR] n_drop={n_drop} must be in (0, {n_genes})")
    rng = np.random.default_rng(seed)
    drop_idx = rng.choice(n_genes, size=n_drop, replace=False)
    keep = np.ones(n_genes, dtype=bool)
    keep[drop_idx] = False
    return keep


def build_random_permutation(n_genes, seed):
    """Return a deterministic permutation of [0, n_genes)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_genes)


# Output writer
def write_output(out_path: Path,
                 X_out: np.ndarray,
                 gene_ids_out: np.ndarray,
                 sample_ids: np.ndarray,
                 baseline_attrs: dict,
                 mode: str,
                 source_path: Path,
                 cli_args: dict,
                 seed: int = None,
                 perm_indices: np.ndarray = None,
                 extra_attrs: dict = None):
    """Write the ablation h5 with all provenance + a fresh hash"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(out_path, backup)
            print(f"  [backup] {backup}")
        out_path.unlink()

    # compute fresh hash of the OUTPUT X (not the parent's!!)
    new_hash = hash_array(X_out)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X", data=X_out, dtype="float32")
        max_g = max(len(g) for g in gene_ids_out)
        f.create_dataset("gene_names", data=gene_ids_out.astype(f"S{max_g}"))
        max_s = max(len(s) for s in sample_ids)
        f.create_dataset("sample_ids",
                         data=sample_ids.astype(f"S{max_s}"))

        # Preserve baseline attrs except X_sha256 (refreshed below)
        for k, v in baseline_attrs.items():
            if k == "X_sha256":
                continue
            f.attrs[k] = v

        # Refresh hash to match new X
        f.attrs["X_sha256"] = new_hash
        f.attrs["ablation_mode"] = mode
        f.attrs["ablation_source"] = str(source_path)
        f.attrs["ablation_script_version"] = SCRIPT_VERSION
        f.attrs["ablation_timestamp"] = datetime.now(timezone.utc).isoformat()

        if seed is not None:
            f.attrs["ablation_seed"] = int(seed)
        if perm_indices is not None:
            f.create_dataset("gene_permutation_indices",
                             data=perm_indices.astype(np.int64))
            f.attrs["has_gene_permutation"] = True
        else:
            f.attrs["has_gene_permutation"] = False

        if extra_attrs:
            for k, v in extra_attrs.items():
                f.attrs[k] = v

    print(f"  Wrote {out_path}")
    print(f"    shape  : {X_out.shape}")
    print(f"    X_sha256: {new_hash[:16]}...")

    # write out provenance JSON sidecar
    prov_path = out_path.with_suffix("").with_suffix(".provenance.json")
    record = {
        "generator": "make_chrom_ablation_h5.py",
        "generator_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ablation_mode": mode,
        "ablation_source": str(source_path),
        "command": " ".join(sys.argv),
        "cli_args": cli_args,
        "extras": {
            "X_sha256": new_hash,
            "shape": list(X_out.shape),
            "n_samples": int(X_out.shape[0]),
            "n_genes": int(X_out.shape[1]),
            "ablation_seed": int(seed) if seed is not None else None,
            "has_gene_permutation": perm_indices is not None,
        },
    }
    if extra_attrs:
        record["extras"].update({k: v for k, v in extra_attrs.items()
                                  if not isinstance(v, (np.ndarray, list, set))})
    prov_path.write_text(json.dumps(record, indent=2, default=str))
    print(f"    provenance: {prov_path}")


# Main
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=VALID_MODES,
                   help="Ablation mode")
    p.add_argument("--input", required=True,
                   help="Baseline h5 to ablate from "
                        "(typically data/combined_expression.h5)")
    p.add_argument("--output", required=True,
                   help="Output h5 path")
    p.add_argument("--coord-map", default="data/chrom_coord_map.csv",
                   help="CSV with ensembl_gene_id, chromosome_name, "
                        "(start_position required for no_thy1_neighborhood)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for random_order / random_drop modes")
    p.add_argument("--n-drop", type=int, default=None,
                   help="Number of genes to drop (random_drop mode)")
    p.add_argument("--neighborhood-window", type=int, default=25,
                   help="±N chr9 genes around Thy1 to drop "
                        "(no_thy1_neighborhood mode; default 25 → 51 total)")
    args = p.parse_args()

    print(f"  make_chrom_ablation_h5.py  ({SCRIPT_VERSION})")
    print(f"  Mode:      {args.mode}")
    print(f"  Input:     {args.input}")
    print(f"  Output:    {args.output}")
    if args.mode in VALID_RANDOM_MODES:
        print(f"  Seed:      {args.seed}")
    if args.mode == "random_drop":
        print(f"  N drop:    {args.n_drop}")
    if args.mode == "no_thy1_neighborhood":
        print(f"  Window:    ±{args.neighborhood_window} chr9 neighbors of Thy1")
    print()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        sys.exit(f"  [ERROR] input not found: {in_path}")

    print("Loading baseline h5 ...")
    X, gene_ids, sample_ids, baseline_attrs = load_baseline(in_path)
    print(f"  Shape: {X.shape}")
    print(f"  Attrs: {len(baseline_attrs)}")
    print()

    # Mode dispatch
    extra_attrs = None
    perm_indices = None

    if args.mode == "random_order":
        # permute gene axis, but keep all genes
        perm_indices = build_random_permutation(X.shape[1], args.seed)
        X_out = X[:, perm_indices]
        gene_ids_out = gene_ids[perm_indices]
        print(f"  Permuted gene axis with seed={args.seed}")
        print(f"  Output shape: {X_out.shape} (no genes dropped)")
        write_output(
            out_path, X_out, gene_ids_out, sample_ids,
            baseline_attrs, args.mode, in_path, vars(args),
            seed=args.seed, perm_indices=perm_indices,
        )
        return

    if args.mode == "random_drop":
        if args.n_drop is None:
            sys.exit("[ERROR] --n-drop is required for random_drop mode")
        keep_mask = build_random_drop_mask(X.shape[1], args.n_drop, args.seed)
        n_keep = int(keep_mask.sum())
        n_drop_actual = int((~keep_mask).sum())
        print(f"  Genes kept:    {n_keep}")
        print(f"  Genes dropped: {n_drop_actual}  (random, seed={args.seed})")
        X_out = X[:, keep_mask]
        gene_ids_out = gene_ids[keep_mask]
        write_output(
            out_path, X_out, gene_ids_out, sample_ids,
            baseline_attrs, args.mode, in_path, vars(args),
            seed=args.seed,
            extra_attrs={"n_genes_dropped": n_drop_actual},
        )
        return

    # all remaining modes need the chrom map
    coord_map_path = Path(args.coord_map)
    if not coord_map_path.exists():
        sys.exit(f"[ERROR] coord map not found: {coord_map_path}")
    chrom_map, pos_map = load_chrom_map(coord_map_path)
    print(f"  Coord map: {len(chrom_map)} entries")

    if args.mode == "no_thy1":
        keep_mask = build_thy1_only_mask(gene_ids)
    elif args.mode == "no_thy1_neighborhood":
        keep_mask, drop_set = build_thy1_neighborhood_mask(
            gene_ids, chrom_map, pos_map,
            window=args.neighborhood_window)
        extra_attrs = {
            "neighborhood_window": args.neighborhood_window,
            "n_genes_in_neighborhood": len(drop_set),
        }
    else:
        keep_mask, chroms = build_chrom_filter_mask(gene_ids, chrom_map, args.mode)

    n_keep = int(keep_mask.sum())
    n_drop = int((~keep_mask).sum())
    print(f" Genes kept: {n_keep}")
    print(f" Genes dropped: {n_drop}")
    print()

    # Show breakdown of dropped genes by chromosome (for diagnostic)
    if args.mode != "no_thy1": # 1-gene case, no need
        chroms_arr = np.array([chrom_map.get(g, "_unmapped") for g in gene_ids])
        dropped_chroms = chroms_arr[~keep_mask]
        if len(dropped_chroms) > 0:
            unique, counts = np.unique(dropped_chroms, return_counts=True)
            print("  Dropped by chromosome (top entries):")
            for c, n in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
                label = c if (c.startswith("chr") or c == "_unmapped") else f"chr{c}"
                print(f"    {label}: {n}")
            print()

    if n_keep == 0:
        sys.exit(f"[ERROR] mode '{args.mode}' would drop all genes")

    X_out = X[:, keep_mask]
    gene_ids_out = gene_ids[keep_mask]

    write_output(
        out_path, X_out, gene_ids_out, sample_ids,
        baseline_attrs, args.mode, in_path, vars(args),
        seed=None, extra_attrs=extra_attrs,
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
