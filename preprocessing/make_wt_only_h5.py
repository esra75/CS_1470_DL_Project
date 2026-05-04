#!/usr/bin/env python3
"""
preprocessing/make_wt_only_h5.py

Sample-axis filter: extract the WT (wild-type) subset of the
expression matrix, preserving the full gene set

This is conceptually distinct from the gene-axis ablations done by
make_chrom_ablation_h5.py. WT-only is a sample-cohort filter: keeps
all preprocessed genes but drop the AD samples, leaving 90 WT samples
(BL6 controls + 3xTgADWT controls)

The output H5 is fully consistent with the parent: same gene_names,
same gene-axis order, same QC flags. Only sample_ids and X rows differ

Outputs:
- data/combined_WT_only.h5
- data/combined_WT_only_metadata.csv
- data/combined_WT_only.provenance.json

Usage
  python preprocessing/make_wt_only_h5.py \\
      --input data/combined_expression.h5 \\
      --metadata data/combined_metadata.csv \\
      --output data/combined_WT_only.h5
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

SCRIPT_VERSION = "v1_2026-04-30" #final 

# Tokens used by the project's preprocessing to identify WT samples in the
# 'genotype' column. Matches the inference rule in evaluate_LOGO.py and
# train.py — keeps WT_only definition consistent across the codebase
WT_TOKENS = {"wt", "wildtype", "bl6", "3xtgadwt"}


def hash_array(X: np.ndarray) -> str:
    return hashlib.sha256(X.tobytes()).hexdigest()

def is_wt(genotype_str: str) -> bool:
    s = str(genotype_str).lower()
    return any(t in s for t in WT_TOKENS)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Baseline H5 (data/combined_expression.h5)")
    p.add_argument("--metadata", required=True,
                   help="Baseline metadata CSV (data/combined_metadata.csv)")
    p.add_argument("--output", required=True,
                   help="Output H5 path (data/combined_WT_only.h5)")
    p.add_argument("--metadata-output", default=None,
                   help="Output metadata CSV "
                        "(default: same dir as --output, "
                        "filename: combined_WT_only_metadata.csv)")
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    meta_in  = Path(args.metadata)

    if not in_path.exists():
        sys.exit(f"[ERROR] input not found: {in_path}")
    if not meta_in.exists():
        sys.exit(f"[ERROR] metadata not found: {meta_in}")

    if args.metadata_output:
        meta_out = Path(args.metadata_output)
    else:
        meta_out = out_path.parent / "combined_WT_only_metadata.csv"


    print(f"  make_wt_only_h5.py  ({SCRIPT_VERSION})")
    print(f" Input H5: {in_path}")
    print(f" Input meta: {meta_in}")
    print(f" Output H5: {out_path}")
    print(f" Output meta: {meta_out}")
    print()

    # Load baseline H5 & metadata
    with h5py.File(in_path, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        genes = np.array(f["gene_names"]).astype(str)
        sample_ids = np.array(f["sample_ids"]).astype(str)
        baseline_attrs = {k: f.attrs[k] for k in f.attrs}

    print(f"Loaded baseline: {X.shape}")
    print(f"  Sample IDs[0]:  {sample_ids[0]}")
    print(f"  Sample IDs[-1]: {sample_ids[-1]}")

    meta_raw = pd.read_csv(meta_in)
    if "sample_id" not in meta_raw.columns:
        sys.exit(f"ERROR metadata CSV must have a 'sample_id' column")

    # Reorder metadata to match H5 sample order
    meta = meta_raw.set_index("sample_id").loc[sample_ids].reset_index()

    if "genotype" not in meta.columns:
        sys.exit(f"ERROR metadata must have a 'genotype' column "
                 "for WT identification")

    # Identify WT samples
    is_wt_arr = meta["genotype"].apply(is_wt).values
    n_wt = int(is_wt_arr.sum())
    n_total = len(meta)
    n_ad = n_total - n_wt

    print()
    print(f" Total samples: {n_total}")
    print(f" WT (kept): {n_wt}")
    print(f" AD (dropped): {n_ad}")
    print()

    if n_wt == 0:
        sys.exit("ERROR No WT samples found — check WT_TOKENS or genotype column")

    # Show breakdown by dataset for diagnostic
    print(" Breakdown of WT samples by dataset x sex x age:")
    wt_meta = meta[is_wt_arr].copy()
    if all(c in wt_meta.columns for c in ("dataset", "sex", "age_months")):
        for (ds, sex, age), grp in wt_meta.groupby(["dataset", "sex", "age_months"]):
            print(f"    {ds:<10} {sex:<6} {age:>4}mo  n={len(grp)}")
    print()

    # Build WT-only arrays
    X_wt = X[is_wt_arr, :].copy()
    sample_ids_wt = sample_ids[is_wt_arr]
    meta_wt = meta[is_wt_arr].reset_index(drop=True)

    print(f"  Output X shape: {X_wt.shape}")

    # Write output H5
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        if not backup.exists():
            import shutil
            shutil.copy2(out_path, backup)
            print(f"  [backup] {backup}")
        out_path.unlink()

    new_hash = hash_array(X_wt)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("X", data=X_wt, dtype="float32")
        max_g = max(len(g) for g in genes)
        f.create_dataset("gene_names", data=genes.astype(f"S{max_g}"))
        max_s = max(len(s) for s in sample_ids_wt)
        f.create_dataset("sample_ids",
                         data=sample_ids_wt.astype(f"S{max_s}"))

        # Preserve baseline attrs except X_sha256 (refreshed)
        for k, v in baseline_attrs.items():
            if k == "X_sha256":
                continue
            f.attrs[k] = v

        # Refresh hash
        f.attrs["X_sha256"] = new_hash
        f.attrs["ablation_mode"] = "WT_only"
        f.attrs["ablation_axis"] = "samples"
        f.attrs["ablation_source"] = str(in_path)
        f.attrs["ablation_script_version"] = SCRIPT_VERSION
        f.attrs["ablation_timestamp"] = datetime.now(timezone.utc).isoformat()
        f.attrs["wt_tokens"] = ",".join(sorted(WT_TOKENS))
        f.attrs["n_samples_input"] = n_total
        f.attrs["n_samples_kept"] = n_wt
        f.attrs["n_samples_dropped"] = n_ad

    print(f" Wrote {out_path}")
    print(f" shape: {X_wt.shape}")
    print(f" X_sha256: {new_hash[:16]}...")

    # Write output metadata
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_wt.to_csv(meta_out, index=False)
    print(f"  Wrote {meta_out}")

    # Provenance JSON 
    # to document 
    prov_path = out_path.with_suffix("").with_suffix(".provenance.json")
    record = {
        "generator": "make_wt_only_h5.py",
        "generator_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ablation_mode": "WT_only",
        "ablation_axis": "samples",
        "ablation_source": str(in_path),
        "command": " ".join(sys.argv),
        "extras": {
            "X_sha256": new_hash,
            "shape": list(X_wt.shape),
            "n_samples_input": n_total,
            "n_samples_kept": n_wt,
            "n_samples_dropped": n_ad,
            "wt_tokens": sorted(WT_TOKENS),
            "metadata_output": str(meta_out),
        },
    }
    prov_path.write_text(json.dumps(record, indent=2, default=str))
    print(f" Wrote {prov_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
