#!/usr/bin/env python3
"""
preprocessing/preprocess.py

Combines 5xFAD AD mouse model (Forner 2021) and 3xTg-AD (Javonillo 2022) 
AD mouse model hippocampal expression data into a single HDF5 file 
fed into MLP and CNN models.

This is latest, final version (2026-04-27): adds mandatory gene QC, sample QC report,
group-size warnings, and provenance JSON tracking. See README at top of
_qc_lib.py 

Pipeline Steps
1. Load each dataset's TPM matrix + metadata
2. Filter 5xFAD to hippocampus only (tissue == 'hippocampus')
3. Find shared gene set (intersection)
4. Gene QC in fixed order:
   a. Remove ERCC spike-ins
   b. Remove zero-variance genes
   c. Remove low-expression genes (< 1 TPM in <10% of samples)
   d. Remove mitochondrial genes (unless --keep-mt)
5. Optional Sensitivity filters (off by default):
   --filter-hemoglobin, --filter-ribosomal, --filter-pseudogenes
6. Sample QC report (never silent removal)
7. Group-size warnings (n<3 strong, n<5 mild) 
8. Perform log1p-transform TPM
9. Per-sample z-score (zero-mean, unit-variance across genes)
10. Optionally apply ComBat batch correction across datasets
11. Write outputs:
    data/combined_expression.h5
    data/combined_metadata.csv
    data/qc/preprocess_log_<TIMESTAMP>.txt
    data/qc/sample_qc_combat.csv  (or _no_combat.csv)
    data/combined_expression.provenance.json

Usage Flags
-----
python preprocessing/preprocess.py # combat + default QC
python preprocessing/preprocess.py --no-combat # no batch correction
python preprocessing/preprocess.py --keep-mt # retain MT genes
python preprocessing/preprocess.py --filter-ribosomal # extra sensitivity filter
python preprocessing/preprocess.py --drop-flagged-samples # apply sample-QC drops
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy import stats

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "preprocessing") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))

from _qc_lib import (
    apply_mandatory_gene_qc,
    apply_sensitivity_gene_qc,
    compute_sample_qc_report,
    compute_group_size_warnings,
    write_provenance,
    file_metadata,
    hash_array,
)


# Data Paths
DATA_5XFAD = {
    "expr": PROJECT_ROOT / "data" / "5xFAD_Forner2021" / "GEO_expression" /
            "GSE168137_expressionList.txt.gz",
    "meta": PROJECT_ROOT / "data" / "5xFAD_Forner2021" / "metadata" /
            "sample_metadata_final.csv",
}
DATA_3XTG = {
    "expr": PROJECT_ROOT / "data" / "3xTgAD_Javonillo2022" /
            "expressionList.csv.gz",
    "meta": PROJECT_ROOT / "data" / "3xTgAD_Javonillo2022" /
            "sample_metadata.csv",
}

OUT_DIR = PROJECT_ROOT / "data"
QC_DIR  = OUT_DIR / "qc"

PREPROCESS_VERSION = "v3_2026-04-27"


# Disease score scheme
# NOTE: WT animals are anchored at 0.0. AD animals are scored as their age in months divided by 18
# ,the maximum timepoint in this study. So, a 4-month AD animal scores 0.22, an 8-month animal scores 0.44, 
# and an 18-month animal at the terminal stage scores 1.0. 
# Anchoring WT at 0.0 is a deliberate biological choice, not a convenience. 
# It nets out normal healthy aging entirely, forcing the model to learn only the transcriptomic signature
# that is specific to the Alzheimer's pathological trajectory; not simply the passage of time in any aging animal. 

MAX_AGE_5XFAD = 18.0
MAX_AGE_3XTG  = 18.0

WT_GENOTYPES_5XFAD = {"WT", "BL6", "wt", "wildtype"}
WT_GENOTYPES_3XTG  = {"3xTgADWT", "WT", "wt", "wildtype"}


def norm_genotype(gt, wt_set):
    if any(w.lower() in str(gt).lower() for w in wt_set):
        return "WT"
    return "AD"


def compute_disease_score(row, wt_set, max_age):
    gt = norm_genotype(row["genotype"], wt_set)
    if gt == "WT":
        return 0.0
    age = float(row["age_months"])
    return min(age / max_age, 1.0)


# Loaders
def load_dataset(expr_path, meta_path, dataset_name,
                 tissue_filter=None, wt_set=None, max_age=18.0):
    """Load one dataset, filter by tissue, ensure disease_score exists."""
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name}")
    print(f"  expr : {expr_path}")
    print(f"  meta : {meta_path}")

    sep = "\t" if str(expr_path).endswith(".txt.gz") else ","
    tpm = pd.read_csv(expr_path, index_col=0, sep=sep, compression="gzip")
    print(f"  TPM shape (raw): {tpm.shape}  (genes × samples)")

    tpm.index = tpm.index.str.split(".").str[0]

    meta = pd.read_csv(meta_path, index_col=0)
    print(f"  Meta shape (raw): {meta.shape}")

    meta.columns = [c.strip().lower().replace(" ", "_") for c in meta.columns]

    if tissue_filter is not None and "tissue" in meta.columns:
        before = len(meta)
        meta = meta[meta["tissue"].str.lower() == tissue_filter.lower()].copy()
        print(f"  Tissue filter '{tissue_filter}': {before} → {len(meta)} samples")

    if "sample_id" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "sample_id"})

    sample_ids = meta["sample_id"].tolist()
    available = [s for s in sample_ids if s in tpm.columns]
    missing   = [s for s in sample_ids if s not in tpm.columns]
    if missing:
        print(f"  WARNING: {len(missing)} sample(s) in meta not in expr — dropping")
        meta = meta[meta["sample_id"].isin(available)].copy()
        sample_ids = available

    tpm = tpm[sample_ids]
    assert list(tpm.columns) == list(meta["sample_id"]), \
        "Column/meta order mismatch after alignment!"
    print(f"  Aligned: {tpm.shape[1]} samples × {tpm.shape[0]} genes")

    required = ["sample_id", "genotype", "sex", "age_months"]
    for col in required:
        if col not in meta.columns:
            raise ValueError(f"Missing required column '{col}' in {dataset_name} metadata")

    if wt_set is not None:
        meta["genotype_norm"] = meta["genotype"].apply(
            lambda g: norm_genotype(g, wt_set))
    else:
        meta["genotype_norm"] = meta["genotype"]

    if "disease_score" not in meta.columns or meta["disease_score"].isna().any():
        print("  Recomputing disease_score ...")
        meta["disease_score"] = meta.apply(
            lambda r: compute_disease_score(r, wt_set or set(), max_age), axis=1)

    meta["dataset"] = dataset_name
    if "tissue" not in meta.columns:
        meta["tissue"] = "hippocampus"

    meta["strat_key"] = (
        meta["dataset"] + "_" +
        meta["genotype_norm"] + "_" +
        meta["sex"].str.lower() + "_" +
        meta["age_months"].astype(str)
    )

    print(f"  Disease score range: {meta['disease_score'].min():.3f}–"
          f"{meta['disease_score'].max():.3f}")
    print(f"  Genotypes: {sorted(meta['genotype_norm'].unique())}")
    print(f"  Ages (months): {sorted(meta['age_months'].unique())}")

    return tpm, meta



# Normalization helpers
def log1p_zscore(tpm_df):
    log_tpm = np.log1p(tpm_df.values.astype(np.float32))
    log_tpm_T = log_tpm.T
    zscored = stats.zscore(log_tpm_T, axis=1, nan_policy="omit")
    zscored = np.nan_to_num(zscored, nan=0.0).astype(np.float32)
    return zscored


def check_combat_available() -> bool:
    try:
        from combat.pycombat import pycombat
        return True
    except ImportError:
        return False


def combat_correction(X, batch_labels):
    if not check_combat_available():
        print("\n  [ComBat] pycombat not importable in this environment")
        return X, False
    from combat.pycombat import pycombat
    df = pd.DataFrame(X.T)
    batch_series = pd.Series(batch_labels)
    corrected = pycombat(df, batch_series)
    return corrected.values.T.astype(np.float32), True


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-combat", action="store_true",
                        help="Skip ComBat batch correction")
    parser.add_argument("--keep-mt",  action="store_true",
                        help="Retain mitochondrial genes (default: remove)")
    parser.add_argument("--filter-hemoglobin", action="store_true",
                        help="(sensitivity) remove hemoglobin gene cluster")
    parser.add_argument("--filter-ribosomal", action="store_true",
                        help="(sensitivity) remove Rpl*/Rps* ribosomal genes")
    parser.add_argument("--filter-pseudogenes", action="store_true",
                        help="(sensitivity) remove Gm##### predicted/pseudogenes")
    parser.add_argument("--drop-flagged-samples", action="store_true",
                        help="(opt-in) drop samples flagged by sample QC report. "
                             "DEFAULT: report only, never drop.")
    parser.add_argument("--low-expr-threshold", type=float, default=1.0,
                        help="TPM threshold for low-expression filter (default 1.0)")
    parser.add_argument("--low-expr-pct", type=float, default=10.0,
                        help="Min %% of samples with expr ≥ threshold (default 10)")
    parser.add_argument("--allow-missing-combat", action="store_true",
                        help="(unsafe) proceed if ComBat requested but pycombat absent")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pre-flight ComBat check
    env_combat_available = check_combat_available()
    if (not args.no_combat) and (not env_combat_available):
        msg = (
            "\n FATAL!! ComBat correction requested (default) but pycombat is "
            "not importable in this environment.\n"
            "Likely cause: running in (base) conda env instead of (aging_cnn).\n"
            "Fix: conda activate aging_cnn (or pass --no-combat to skip ComBat)"
        )
        if args.allow_missing_combat:
            print(msg)
            print("[--allow-missing-combat] continuing. Output WILL NOT be ComBat-corrected")
        else:
            sys.exit(msg)

    print(f" preprocess.py {PREPROCESS_VERSION}")
    print(f" Timestamp: {timestamp}")
    print(f" Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}")
    print(f" pycombat available: {env_combat_available}")
    print(f" Args: {vars(args)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets
    tpm_5x, meta_5x = load_dataset(
        expr_path=DATA_5XFAD["expr"], meta_path=DATA_5XFAD["meta"],
        dataset_name="5xFAD", tissue_filter="hippocampus",
        wt_set=WT_GENOTYPES_5XFAD, max_age=MAX_AGE_5XFAD,
    )
    tpm_3x, meta_3x = load_dataset(
        expr_path=DATA_3XTG["expr"], meta_path=DATA_3XTG["meta"],
        dataset_name="3xTgAD", tissue_filter=None,
        wt_set=WT_GENOTYPES_3XTG, max_age=MAX_AGE_3XTG,
    )

    # 2. Gene intersection
    genes_5x = set(tpm_5x.index)
    genes_3x = set(tpm_3x.index)
    shared = sorted(genes_5x & genes_3x)
    print(f"\n{'='*60}")
    print(f"Gene set overlap")
    print(f"  5xFAD:  {len(genes_5x)}")
    print(f"  3xTgAD: {len(genes_3x)}")
    print(f"  Shared: {len(shared)}")

    if len(shared) < 10000:
        print("\n  WARNING: very few shared genes — check Ensembl IDs")

    tpm_5x = tpm_5x.loc[shared]
    tpm_3x = tpm_3x.loc[shared]

    # 3. Concatenate
    tpm_all  = pd.concat([tpm_5x, tpm_3x], axis=1)
    meta_all = pd.concat([meta_5x, meta_3x], ignore_index=True)
    assert tpm_all.shape[1] == len(meta_all), "Sample count mismatch!"
    print(f"\nCombined (pre-QC): {tpm_all.shape[1]} samples × {tpm_all.shape[0]} genes")

    n_genes_input = tpm_all.shape[0]
    n_samples = tpm_all.shape[1]

    # 4. QC
    print(f"\n{'='*60}")
    print("MANDATORY gene QC (filters in fixed order)")
    print(f"{'='*60}")
    tpm_all, qc_log_mandatory = apply_mandatory_gene_qc(
        tpm_all,
        keep_mt=args.keep_mt,
        min_samples_expressed_pct=args.low_expr_pct,
        expression_threshold=args.low_expr_threshold,
        verbose=True,
    )
    print(f"\n After mandatory QC: {tpm_all.shape[0]} genes "
          f"({n_genes_input - tpm_all.shape[0]} removed)")

    # 5. OPTIONAL sensitivity filters
    print(f"\n{'='*60}")
    print("OPTIONAL sensitivity filters")
    print(f"{'='*60}")
    tpm_all, qc_log_sensitivity = apply_sensitivity_gene_qc(
        tpm_all,
        filter_hemoglobin=args.filter_hemoglobin,
        filter_ribosomal=args.filter_ribosomal,
        filter_pseudogenes=args.filter_pseudogenes,
        gene_symbol_map=None, # symbol map not yet available — see build_coord_map_with_symbols.py
        verbose=True,
    )
    if args.filter_ribosomal or args.filter_pseudogenes:
        print(f" Note: ribosomal/pseudogene filters require symbols ")
        print(f" These filters are no-ops in preprocess.py — apply them post-symbols")

    n_genes_post_qc = tpm_all.shape[0]
    print(f"\n Final gene count after all QC: {n_genes_post_qc}")

    gene_names = np.array(tpm_all.index.tolist())

    # 6. SAMPLE QC report
    print(f"\n{'='*60}")
    print("Sample QC report (no automatic dropping)")
    print(f"{'='*60}")
    sample_qc = compute_sample_qc_report(
        tpm_all, meta_all,
        expression_threshold=args.low_expr_threshold,
        verbose=True,
    )

    # Save the per-sample QC table!
    qc_csv_name = f"sample_qc_{'no_combat' if args.no_combat else 'combat'}.csv"
    qc_csv = QC_DIR / qc_csv_name
    sample_qc["per_sample"].to_csv(qc_csv)
    print(f"\n  Sample QC table saved: {qc_csv}")

    # Optional: drop flagged samples
    # in final run kept, but optional here 
    if args.drop_flagged_samples:
        flagged = sample_qc["flags"]["any_flag"]
        if flagged:
            print(f"\n  --drop-flagged-samples: dropping {len(flagged)} samples")
            for sid in flagged:
                print(f"    DROP: {sid}")
            keep_mask = ~meta_all["sample_id"].isin(flagged)
            meta_all = meta_all[keep_mask].reset_index(drop=True)
            tpm_all = tpm_all.loc[:, meta_all["sample_id"].tolist()]
            print(f"  After drops: {len(meta_all)} samples remain")
        else:
            print(f"\n  --drop-flagged-samples: 0 flagged, nothing to drop")
    else:
        if sample_qc["flags"]["any_flag"]:
            print(f"\n  [REVIEW] {len(sample_qc['flags']['any_flag'])} sample(s) flagged "
                  "Pass --drop-flagged-samples to drop, or review manually before training")

    # 7. GROUP-size warnings
    print(f"\n{'='*60}")
    print("Group-size warnings (mandatory)")
    print(f"{'='*60}")
    group_qc = compute_group_size_warnings(meta_all, verbose=True)

    if group_qc["n_strong_warnings"] > 0:
        print(f"\nWARN! {group_qc['n_strong_warnings']} group(s) with n<3 "
              "These will produce noisy / undefined LOGO-CV folds")

    # 8. log1p & per-sample z-score
    print(f"\n{'='*60}")
    print("Normalization: log1p --> per-sample z-score")
    print(f"{'='*60}")
    X = log1p_zscore(tpm_all)
    print(f" X shape: {X.shape}  dtype: {X.dtype}")
    print(f" X mean: {X.mean():.4f}  std: {X.std():.4f}")

    # 9. Optional ComBat
    # included in final, but optional if wanted 
    combat_actually_ran = False
    if not args.no_combat:
        print(f"\nApplying ComBat batch correction (dataset as batch) ...")
        batch_labels = meta_all["dataset"].tolist()
        X, combat_actually_ran = combat_correction(X, batch_labels)
        if combat_actually_ran:
            print(f"Post-ComBat X mean: {X.mean():.4f}  std: {X.std():.4f}")
        else:
            print(f" WARNING! ComBat skipped at runtime. X is NOT batch-corrected")
    else:
        print(f"\nSkipping ComBat (--no-combat flag)")

    # 10. Write outputs
    prefix = "nc_" if args.no_combat else ""
    OUT_H5   = OUT_DIR / f"{prefix}combined_expression.h5"
    OUT_META = OUT_DIR / f"{prefix}combined_metadata.csv"

    print(f"\n{'='*60}")
    print(f"Writing outputs:")
    print(f"{'='*60}")
    print(f"  {OUT_H5}")
    with h5py.File(OUT_H5, "w") as f:
        f.create_dataset("X", data=X, dtype="float32",
                         compression="gzip", compression_opts=4)
        f.create_dataset("gene_names", data=gene_names.astype("S"),
                         compression="gzip")
        f.create_dataset("sample_ids",
                         data=meta_all["sample_id"].values.astype("S"),
                         compression="gzip")
        # attributes
        f.attrs["mt_removed"] = (not args.keep_mt)
        f.attrs["combat"] = combat_actually_ran
        f.attrs["combat_requested"] = (not args.no_combat)
        f.attrs["env_combat_available"] = env_combat_available
        f.attrs["preprocess_version"] = PREPROCESS_VERSION
        f.attrs["gene_qc_applied"] = True
        f.attrs["n_genes_input"] = n_genes_input
        f.attrs["n_genes_after_qc"] = n_genes_post_qc
        f.attrs["low_expr_threshold_tpm"] = args.low_expr_threshold
        f.attrs["low_expr_pct"] = args.low_expr_pct
        f.attrs["filter_hemoglobin"] = args.filter_hemoglobin
        f.attrs["filter_ribosomal"] = args.filter_ribosomal
        f.attrs["filter_pseudogenes"] = args.filter_pseudogenes
        f.attrs["drop_flagged_samples"] = args.drop_flagged_samples
        f.attrs["timestamp"] = timestamp
        f.attrs["X_sha256"] = hash_array(X)

    print(f" {OUT_META}")
    keep_cols = ["sample_id", "dataset", "genotype", "genotype_norm",
                 "sex", "age_months", "disease_score", "tissue", "strat_key"]
    out_cols = [c for c in keep_cols if c in meta_all.columns]
    meta_all[out_cols].to_csv(OUT_META, index=False)

    # 11. QC log
    log_path = QC_DIR / f"preprocess_log_{prefix or 'combat_'}{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write(f"Preprocess log — {PREPROCESS_VERSION}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'unknown')}\n")
        f.write(f"Args: {vars(args)}\n\n")
        f.write(f"Inputs:\n")
        f.write(f"  5xFAD: {DATA_5XFAD['expr']}\n")
        f.write(f"  3xTg:  {DATA_3XTG['expr']}\n\n")
        f.write(f"Pipeline summary:\n")
        f.write(f"  Genes input (intersection): {n_genes_input}\n")
        f.write(f"  Genes after mandatory QC: {qc_log_mandatory['n_genes_after_mandatory']}\n")
        f.write(f"  Genes after sensitivity filters: {qc_log_sensitivity['n_genes_after_sensitivity']}\n")
        f.write(f"  Final gene count: {n_genes_post_qc}\n\n")
        f.write(f"Sample count: {n_samples}\n")
        if args.drop_flagged_samples:
            f.write(f"Samples dropped by --drop-flagged-samples: "
                    f"{n_samples - len(meta_all)}\n")
        f.write(f"Final sample count: {len(meta_all)}\n\n")
        f.write(f"Mandatory QC details:\n")
        for filt in qc_log_mandatory["filters_applied"]:
            f.write(f"  {filt['filter']}: {filt['n_removed']} genes removed\n")
        f.write(f"\n{sample_qc['summary']}\n\n")
        f.write(f"\n{group_qc['summary']}\n")
        f.write(f"\nOutputs:\n")
        f.write(f"  {OUT_H5}\n")
        f.write(f"  {OUT_META}\n")
        f.write(f"  {qc_csv}\n")
    print(f"  {log_path}")

    # 12. Provenance JSON
    prov_path = OUT_DIR / f"{prefix}combined_expression.provenance.json"
    write_provenance(
        prov_path,
        script_name=__file__,
        inputs={
            "5xFAD_expression": file_metadata(DATA_5XFAD["expr"]),
            "5xFAD_metadata": file_metadata(DATA_5XFAD["meta"]),
            "3xTgAD_expression":file_metadata(DATA_3XTG["expr"]),
            "3xTgAD_metadata": file_metadata(DATA_3XTG["meta"]),
        },
        args=vars(args),
        outputs={
            "h5": file_metadata(OUT_H5),
            "meta": file_metadata(OUT_META),
            "qc_log": file_metadata(log_path),
            "sample_qc_csv": file_metadata(qc_csv),
        },
        extras={
            "preprocess_version": PREPROCESS_VERSION,
            "n_genes_input": n_genes_input,
            "n_genes_after_mandatory_qc": qc_log_mandatory["n_genes_after_mandatory"],
            "n_genes_final": n_genes_post_qc,
            "n_samples": len(meta_all),
            "n_samples_input": n_samples,
            "mandatory_qc_filters": qc_log_mandatory["filters_applied"],
            "sensitivity_filters": qc_log_sensitivity["filters_applied"],
            "sample_qc_flags": {k: len(v) for k, v in sample_qc["flags"].items()},
            "group_qc_warnings": group_qc["warnings"],
            "combat_actually_ran": combat_actually_ran,
            "X_sha256": hash_array(X),
            "x_min": float(X.min()),
            "x_max": float(X.max()),
            "x_mean": float(X.mean()),
            "x_std": float(X.std()),
        },
    )
    print(f" {prov_path}")

    # Final summary - writeout 
    size_mb = os.path.getsize(OUT_H5) / 1e6
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f" H5: {OUT_H5}  ({size_mb:.1f} MB)")
    print(f" Meta: {OUT_META}")
    print(f" QC log: {log_path}")
    print(f" Sample QC: {qc_csv}")
    print(f" Provenance: {prov_path}")
    print(f"\n  h5 attributes:")
    with h5py.File(OUT_H5, "r") as f:
        for k, v in dict(f.attrs).items():
            if k == "X_sha256":
                v = str(v)[:16] + "..."
            print(f"    {k:30s} = {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
