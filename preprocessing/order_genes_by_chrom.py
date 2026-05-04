#!/usr/bin/env python3
"""
preprocessing/order_genes_by_chrom.py

One-time preprocessing step: reorders the genes in the combined expression h5
files by chromosomal position (chromosome, start_position) so that 1D CNN
convolutions operate over genomically meaningful neighborhoods rather than
an arbitrary gene ordering

Biological rationale
--------------------
Co-expressed genes are frequently co-located: TADs (Topologically Associating
Domains), polycistronic clusters, cytokine gene families, and lncRNA/target
pairs are all physically clustered on chromosomes. A 1D CNN scanning a
chromosomally-ordered expression vector can detect these co-expression blocks
as spatial "features", analogous to the way a CNN detects edges in an image.
Without ordering, convolutional filters observe random noise.

This approach is adopted from TimeFlies (Drosophila aging clock - Larschan & Singh Labs
@ Brown), which discovered the X-chromosome roX1/roX2 dosage-compensation signature
specifically because chromosomal ordering made the regional signal detectable.

What this script does:
1. Queries Ensembl BioMart for genomic coordinates of all Ensembl gene ID 
in the h5 files (mouse GRCm38/mm10)
2. Sorts genes: chromosomes 1–19, X, Y, MT (ummapped contigs go last)
3. Saves the ordered gene list to data/chrom_gene_order.txt for reproducibility
4. Rewrites combined_expression.h5 and nc_combined_expression.h5 with columns
reordered to match chromosomal order. Existing files are backed up to .bak.
5. Regenerates hvg_combined_expression.h5 to match (HVG selection is reapplied
from the reordered no-combat matrix).

UPDATE 2026-04-26
- reorder_h5() and regenerate_hvg() now PRESERVE existing h5 attributes
  (mt_removed, combat, combat_requested, etc.) when rewriting. Previously
  these attributes were silently dropped because h5py.File(..., "w") truncates
  the file. They were re-added with only "chrom_ordered=True", which made it
  look like the MT filter never ran.
- regenerate_hvg() now refuses to run if .attrs["mt_removed"] disagrees
  between the combat and no-combat h5 files (defensive — would indicate the
  files were generated separately).

Dependencies
------------
pip install pybiomart

Usage
-----
# From mouse_clock/ project root:
python preprocessing/order_genes_by_chrom.py

# Dry-run (query and save order file only, do not rewrite h5):
python preprocessing/order_genes_by_chrom.py --dry-run

# If there is a saved mapping from a previous run, skip the BioMart query:
python preprocessing/order_genes_by_chrom.py --order-file data/chrom_gene_order.txt

Outputs
-------
data/chrom_gene_order.txt -- one Ensembl ID per line, in chrom order
data/combined_expression.h5  -- reordered in-place (original backed up)
data/nc_combined_expression.h5 -- reordered in-place (original backed up)
data/hvg_combined_expression.h5 -- regenerated (HVG mask reapplied)
data/chrom_coord_map.csv -- full coordinate table for inspection
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


#Chromosome sort key
CHR_ORDER = {str(i): i for i in range(1, 20)} # 1–19
CHR_ORDER.update({"X": 20, "Y": 21, "MT": 22}) # sex chromosomes and mitochondrial

def chr_sort_key(chrom: str) -> tuple:
    """Return (int_rank, str) so unknown contigs sort last alphabetically."""
    return (CHR_ORDER.get(str(chrom), 99), str(chrom))


# Helpers functions
def strip_ensembl_version(gene_id: str) -> str:
    """
    Convert ENSMUSG00000000001.15 -> ENSMUSG00000000001
    Leaves IDs without version suffix unchanged.
    """
    return str(gene_id).split(".")[0]


def choose_gene_filter(dataset) -> str:
    """
    Pick a usable BioMart filter for Ensembl gene IDs
    """
    available_filters = set(dataset.filters.keys())

    candidates = [
        "ensembl_gene_id",
        "link_ensembl_gene_id",
        "gene_id",
    ]

    for filt in candidates:
        if filt in available_filters:
            return filt

    preview = sorted(available_filters)[:80]
    raise RuntimeError(
        "Could not find a usable Ensembl gene ID filter in BioMart.\n"
        f"Tried: {candidates}\n"
        f"Available filters (first 80): {preview}"
    )


# BioMart query
def query_biomart(gene_ids: list) -> pd.DataFrame:
    """
    Query Ensembl BioMart (GRCm38/mm10) for chromosomal coordinates

    Returns a DataFrame with cols:
        ensembl_gene_id, chromosome_name, start_position
    """
    try:
        from pybiomart import Server
    except ImportError:
        sys.exit(
            "[ERROR] pybiomart not installed.\n"
            "Install with: pip install pybiomart\n"
            "Then re-run this script"
        )

    BIOMART_HOST = "dec2021.archive.ensembl.org"

    print(f"Connecting to Ensembl BioMart archive ({BIOMART_HOST}) ...")
    print(" Downloading full mouse gene table — no filter argument.")
    print("(~65k rows, typically 60–180 seconds)\n")

    server = Server(host=BIOMART_HOST)
    dataset = server["ENSEMBL_MART_ENSEMBL"]["mmusculus_gene_ensembl"]

    required_attributes = ["ensembl_gene_id", "chromosome_name", "start_position"]
    available_attributes = set(dataset.attributes.keys())
    missing_attrs = [a for a in required_attributes if a not in available_attributes]
    if missing_attrs:
        raise RuntimeError(
            f"BioMart dataset is missing attributes: {missing_attrs}\n"
            f"Available (first 80): {sorted(available_attributes)[:80]}"
        )

    result = dataset.query(attributes=required_attributes)
    print(f" Downloaded {len(result)} rows from BioMart.")

    # strip Ensembl version suffixes from the h5 gene IDs for matching
    original_gene_ids = [str(g) for g in gene_ids]
    stripped_gene_ids = [strip_ensembl_version(g) for g in original_gene_ids]
    stripped_to_originals = {}
    for orig, stripped in zip(original_gene_ids, stripped_gene_ids):
        stripped_to_originals.setdefault(stripped, []).append(orig)

    result.columns = ["ensembl_gene_id", "chromosome_name", "start_position"]
    result["start_position"] = (
        pd.to_numeric(result["start_position"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    print(f" BioMart returned {len(result)} rows for {len(original_gene_ids)} query genes")
    # Some genes return multiple rows (alternative assemblies / patches)
    # Keep the entry on the canonical chromosome (lowest CHR_ORDER rank)
    result["_chr_rank"] = result["chromosome_name"].apply(
        lambda c: CHR_ORDER.get(str(c), 99)
    )
    result = (
        result.sort_values(["ensembl_gene_id", "_chr_rank", "start_position"])
        .drop_duplicates(subset="ensembl_gene_id", keep="first")
        .drop(columns="_chr_rank")
        .reset_index(drop=True)
    )

    print(f"  After deduplication: {len(result)} unique genes with coordinates.")
    rows = []
    found_original_ids = set()

    for _, row in result.iterrows():
        stripped_id = row["ensembl_gene_id"]
        originals = stripped_to_originals.get(stripped_id, [])
        for original_id in originals:
            rows.append({
                "ensembl_gene_id": original_id,
                "chromosome_name": row["chromosome_name"],
                "start_position": row["start_position"],
            })
            found_original_ids.add(original_id)

    mapped_result = pd.DataFrame(rows)

    # fill in genes not returned by BioMart
    missing = [g for g in original_gene_ids if g not in found_original_ids]
    if missing:
        print(f" {len(missing)} genes not found in BioMart — appended as '_unmapped'.")
        missing_df = pd.DataFrame({
            "ensembl_gene_id": missing,
            "chromosome_name": "_unmapped",
            "start_position": 0,
        })
        mapped_result = pd.concat([mapped_result, missing_df], ignore_index=True)

    mapped_result["_chr_rank"] = mapped_result["chromosome_name"].apply(
        lambda c: CHR_ORDER.get(str(c), 99)
    )
    mapped_result = (
        mapped_result.sort_values(["ensembl_gene_id", "_chr_rank", "start_position"])
        .drop_duplicates(subset="ensembl_gene_id", keep="first")
        .drop(columns="_chr_rank")
        .reset_index(drop=True)
    )

    print(
        f"  Final mapped coordinate table: {len(mapped_result)} rows "
        f"for {len(original_gene_ids)} original genes."
    )

    return mapped_result


# Sorting
def sort_genes(coord_df: pd.DataFrame) -> list:
    """
    Return a list of Ensembl IDs sorted by (chr_sort_key, start_position)
    """
    coord_df = coord_df.copy()
    coord_df["_chr_rank"] = coord_df["chromosome_name"].apply(
        lambda c: CHR_ORDER.get(str(c), 99)
    )
    coord_df = coord_df.sort_values(
        ["_chr_rank", "chromosome_name", "start_position"],
        ascending=True
    ).drop(columns="_chr_rank")

    ordered = coord_df["ensembl_gene_id"].tolist()

    chrs_seen = coord_df["chromosome_name"].unique()
    print(f"\nChromosomes represented: {sorted(chrs_seen, key=chr_sort_key)}")
    for chrom in sorted(chrs_seen, key=chr_sort_key)[:5]:
        n = (coord_df["chromosome_name"] == chrom).sum()
        print(f"  chr{chrom}: {n} genes")
    print("  ...")

    return ordered


# H5 rewrite (updated to preserve attributes)
def reorder_h5(h5_path: str, gene_names_current: np.ndarray,
               new_order_indices: np.ndarray) -> None:
    """
    Rewrite h5_path in-place with columns reordered by new_order_indices
    Creates a .bak backup first

    UPDATE: now PRESERVES existing h5 attributes (mt_removed, combat, etc.)
    by reading them from the source file and re-writing them in the output
    """
    bak_path = h5_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(h5_path, bak_path)
        print(f" Backup: {bak_path}")
    else:
        print(f" Backup already exists: {bak_path} (skipping copy)")

    print(f" Reordering {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        X = f["X"][:] # (N, G)
        sample_ids = np.array(f["sample_ids"])
        # UPDATE FIX: capture existing attributes for preservation
        existing_attrs = {k: v for k, v in f.attrs.items()}

    X_reordered = X[:, new_order_indices] # (N, G) now reordered
    genes_reordered = gene_names_current[new_order_indices]

    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "X",
            data=X_reordered,
            dtype="float32",
            compression="gzip",
            compression_opts=4
        )
        f.create_dataset(
            "gene_names",
            data=genes_reordered.astype("S"),
            compression="gzip"
        )
        f.create_dataset(
            "sample_ids",
            data=sample_ids,
            compression="gzip"
        )
        # FIX: re-write all preserved attributes
        for k, v in existing_attrs.items():
            f.attrs[k] = v
        # FIX: add chrom_ordered AFTER preserving existing attrs
        f.attrs["chrom_ordered"] = True
        # POST-RUN FIX: refresh X_sha256 because the data was reordered
        # without this, the stored hash is from before reordering, which
        # makes sanity_check_h5.py flag a false-positive integrity failure
        new_hash_for_provenance = None
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _qc_lib import hash_array
            new_hash_for_provenance = hash_array(X_reordered)
            f.attrs["X_sha256"] = new_hash_for_provenance
        except Exception as _e:
            print(f" WARNING could not refresh X_sha256: {_e}")

    # POST-RUN FIX: also refresh the *.provenance.json sidecar so
    # sanity_check_h5.py's provenance_hash check passes after reorder
    # UPDATE: append a 'post_reorder' record rather than overwriting the original
    # write, so the full lineage is preserved
    if new_hash_for_provenance is not None:
        try:
            import json
            from datetime import datetime, timezone
            prov_path = Path(h5_path).with_suffix("").with_suffix(".provenance.json")
            # above gives e.g. "combined_expression.provenance.json" because
            # str(Path("data/combined_expression.h5").with_suffix("")) -->
            # "data/combined_expression". Then with_suffix(".provenance.json") gives
            # "data/combined_expression.provenance.json"
            if prov_path.exists():
                with open(prov_path, "r") as pf:
                    record = json.load(pf)
                # Append a post-reorder block AND update the top-level X_sha256
                # in extras so the sanity check finds matching value
                if "extras" not in record:
                    record["extras"] = {}
                record["extras"]["X_sha256_pre_reorder"] = record["extras"].get("X_sha256")
                record["extras"]["X_sha256"] = new_hash_for_provenance
                record["extras"]["reordered"] = {
                    "by_script": __file__,
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "new_X_sha256": new_hash_for_provenance,
                    "previous_X_sha256": record["extras"].get("X_sha256_pre_reorder"),
                }
                with open(prov_path, "w") as pf:
                    json.dump(record, pf, indent=2, default=str)
                print(f"  Provenance JSON refreshed: {prov_path}")
            else:
                print(f"  [INFO] no provenance JSON at {prov_path} — skipping refresh")
        except Exception as _e:
            print(f"  [WARN] could not refresh provenance JSON: {_e}")

    print(f" Done. Shape: {X_reordered.shape}")
    print(f" Attributes preserved: {sorted(existing_attrs.keys())}")
    print(f" Plus added: chrom_ordered=True")


# HVG regeneration (FIXED & UPDATED to preserve attributes + verify consistency)
def regenerate_hvg(nc_h5: str, combat_h5: str, hvg_h5: str,
                   n_hvgs: int = 2000) -> None:
    """
    Reapply HVG selection from the reordered no-combat matrix and write
    hvg_combined_expression.h5 with the same key layout as before

    UPDATE: preserves attributes from the source combat h5, plus verifies
    that mt_removed and combat-status are consistent between the nc and
    combat h5 files (catches the case where they got out of sync)
    """
    print(f"\nRegenerating {hvg_h5} (n_hvgs={n_hvgs}) ...")

    with h5py.File(nc_h5, "r") as f:
        X_nc = f["X"][:]
        gene_names = np.array(f["gene_names"]).astype(str)
        sample_ids = np.array(f["sample_ids"])
        nc_attrs = {k: v for k, v in f.attrs.items()}

    with h5py.File(combat_h5, "r") as f:
        X_combat = f["X"][:]
        combat_attrs = {k: v for k, v in f.attrs.items()}

    # FIX: consistency check between nc and combat h5
    if nc_attrs.get("mt_removed") != combat_attrs.get("mt_removed"):
        sys.exit(
            f"FATAL! mt_removed differs between {nc_h5} "
            f"({nc_attrs.get('mt_removed')}) and {combat_h5} "
            f"({combat_attrs.get('mt_removed')}). "
            "Files must have been preprocessed independently "
            "Re-run preprocess.py for both before reordering"
        )

    # Select HVGs by variance on the no-combat matrix
    gene_var = np.var(X_nc, axis=0)
    ranked_idx = np.argsort(gene_var)[::-1]
    hvg_idx = ranked_idx[:n_hvgs]

    # Keep chromosomal order within the HVG subset by sorting hvg_idx
    hvg_idx_sorted = np.sort(hvg_idx)

    X_hvg = X_combat[:, hvg_idx_sorted]
    hvg_genes = gene_names[hvg_idx_sorted]

    print(f" HVG variance threshold: {gene_var[ranked_idx[n_hvgs - 1]]:.4f}")
    print(f" X_hvg shape: {X_hvg.shape}")

    bak_path = hvg_h5 + ".bak"
    if os.path.exists(hvg_h5) and not os.path.exists(bak_path):
        shutil.copy2(hvg_h5, bak_path)
        print(f"  Backup: {bak_path}")

    with h5py.File(hvg_h5, "w") as f:
        f.create_dataset(
            "X",
            data=X_hvg,
            dtype="float32",
            compression="gzip",
            compression_opts=4
        )
        f.create_dataset(
            "gene_names",
            data=hvg_genes.astype("S"),
            compression="gzip"
        )
        f.create_dataset(
            "sample_ids",
            data=sample_ids,
            compression="gzip"
        )
        # UPDATE/FIX: inherit attrs from the source combat h5
        for k, v in combat_attrs.items():
            f.attrs[k] = v
        # then add HVG-specific attrs
        f.attrs["n_hvgs"]        = n_hvgs
        f.attrs["chrom_ordered"] = True
        f.attrs["hvg_filtered"]  = True
        f.attrs["source_nc"]     = str(nc_h5)
        f.attrs["source_combat"] = str(combat_h5)

    gene_list_path = os.path.join(os.path.dirname(hvg_h5), "hvg_gene_list.txt")
    with open(gene_list_path, "w") as f:
        for g in hvg_genes:
            f.write(g + "\n")
    print(f" Gene list: {gene_list_path} ({len(hvg_genes)} genes)")
    print(f" Attributes inherited from combat h5: {sorted(combat_attrs.keys())}")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Reorder genes in expression h5 files by chromosomal position"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the h5 and metadata files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query BioMart and save order file only; do NOT rewrite h5"
    )
    parser.add_argument(
        "--order-file",
        default=None,
        help="Use a pre-existing chrom_gene_order.txt instead of querying BioMart"
    )
    parser.add_argument(
        "--n-hvgs",
        type=int,
        default=2000,
        help="Number of HVGs to retain when regenerating HVG h5 (default: 2000)"
    )
    parser.add_argument(
        "--skip-hvg-regen",
        action="store_true",
        help="Skip HVG h5 regeneration (use preprocess_HVG.py separately)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    combat_h5 = data_dir / "combined_expression.h5"
    nc_h5 = data_dir / "nc_combined_expression.h5"
    hvg_h5 = data_dir / "hvg_combined_expression.h5"
    order_txt = data_dir / "chrom_gene_order.txt"
    coord_csv = data_dir / "chrom_coord_map.csv"

    for path in [combat_h5, nc_h5]:
        if not path.exists():
            sys.exit(
                f" ERROR Required file not found: {path}\n"
                "Run preprocessing/preprocess.py first (both with and without --no-combat)"
            )

    # 1. Get current gene list from h5 
    print(f"Reading gene list from {combat_h5} ...")
    with h5py.File(combat_h5, "r") as f:
        gene_names_current = np.array(f["gene_names"]).astype(str)
        attrs_before = dict(f.attrs)
    n_genes = len(gene_names_current)
    print(f" {n_genes} genes in current h5")
    print(f" Existing attrs: {sorted(attrs_before.keys())}")
    print(f" mt_removed: {attrs_before.get('mt_removed', 'NOT SET')}")
    print(f" combat: {attrs_before.get('combat', 'NOT SET')}")

    # 2. Get chromosomal order
    if args.order_file:
        print(f"\nLoading pre-computed gene order from {args.order_file} ...")
        with open(args.order_file) as fh:
            ordered_genes = [line.strip() for line in fh if line.strip()]
        print(f"  Loaded {len(ordered_genes)} genes")
        coord_df = None
    else:
        print(f"\nQuerying BioMart for {n_genes} genes ...")
        coord_df = query_biomart(gene_names_current.tolist())
        ordered_genes = sort_genes(coord_df)

        if coord_df is not None:
            coord_df.to_csv(coord_csv, index=False)
            print(f"\nCoordinate map saved: {coord_csv}")

    # Save ordered gene list (always)
    with open(order_txt, "w") as fh:
        for g in ordered_genes:
            fh.write(g + "\n")
    print(f"Gene order saved: {order_txt} ({len(ordered_genes)} genes)")

    if args.dry_run:
        print("\n[dry-run] Stopping here — h5 files not modified.")
        return

    # 3. Validate: ordered_genes must be a permutation of current genes
    current_set = set(gene_names_current)
    ordered_set = set(ordered_genes)
    extra = ordered_set - current_set
    missing = current_set - ordered_set

    if extra:
        print(f"WARNING {len(extra)} genes in order file not in h5 — will be ignored")
        ordered_genes = [g for g in ordered_genes if g in current_set]

    if missing:
        print(f"WARNING {len(missing)} genes in h5 not in order file — appended at end")
        ordered_genes = ordered_genes + [
            g for g in gene_names_current if g not in ordered_set
        ]

    # build index array: new_order_indices[i] = original column index for new position i
    gene_to_idx = {g: i for i, g in enumerate(gene_names_current)}
    new_order_indices = np.array([gene_to_idx[g] for g in ordered_genes], dtype=np.int32)

    assert len(new_order_indices) == n_genes, "Index array length mismatch"
    assert len(set(new_order_indices)) == n_genes, "Duplicate indices in order — quitting"

    # 4. Rewrite combat and no-combat h5 files
    print("\nRewriting h5 files with chromosomal gene order ...")
    gene_names_arr = gene_names_current

    reorder_h5(str(combat_h5), gene_names_arr, new_order_indices)
    reorder_h5(str(nc_h5), gene_names_arr, new_order_indices)

    # 5. Regenerate HVG h5
    if args.skip_hvg_regen:
        print("\n[--skip-hvg-regen] Skipping HVG regeneration "
              "Run preprocessing/preprocess_HVG.py manually")
    elif hvg_h5.exists():
        regenerate_hvg(str(nc_h5), str(combat_h5), str(hvg_h5), n_hvgs=args.n_hvgs)
    else:
        print(f"\n[INFO] {hvg_h5} not found — skipping HVG regeneration")
        print(" Run: python preprocessing/preprocess_HVG.py to regenerate")

    # 6. Verify outputs 
    # UPDATE/FIX: explicitly verify that critical attrs survived the rewrite
    print("\nVerifying preserved attributes...")
    for path in [combat_h5, nc_h5]:
        with h5py.File(path, "r") as f:
            attrs = dict(f.attrs)
        print(f"  {path.name}:")
        for k in ("mt_removed", "combat", "combat_requested",
                  "env_combat_available", "chrom_ordered",
                  "preprocess_version"):
            print(f" {k} = {attrs.get(k, 'NOT SET')}")

    # 7. Summary
    print(f"""
Done. Summary
=============
  Gene order file : {order_txt}
  Coordinate map : {coord_csv if coord_df is not None else '(used pre-computed order)'}
  combat h5 : {combat_h5}  (backup: {combat_h5}.bak)
  nc h5 : {nc_h5}  (backup: {nc_h5}.bak)
  hvg h5 : {hvg_h5}  (backup: {hvg_h5}.bak if existed)

Next steps
----------
  1. Rerun LOGO-CV: python evaluate_LOGO.py
  2. Compare OOF r vs the pre-ordering baseline

To revert to the previous ordering:
  cp data/combined_expression.h5.bak data/combined_expression.h5
  cp data/nc_combined_expression.h5.bak data/nc_combined_expression.h5
  cp data/hvg_combined_expression.h5.bak data/hvg_combined_expression.h5
""")


if __name__ == "__main__":
    main()
