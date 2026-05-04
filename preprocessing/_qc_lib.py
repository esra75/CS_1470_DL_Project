"""
preprocessing/_qc_lib.py


Shared QC utilities used by preprocess.py, preprocess_HVG.py, and
sanity_check_h5.py. Single source of truth for:

  - Mitochondrial gene IDs (mouse Ensembl GRCm38)
  - ERCC spike-in detection
  - Hemoglobin / ribosomal / pseudogene filter sets (mouse)
  - Sample QC computations (library size, gene detection, sex, PCA)
  - Group-size warning tiers
  - Provenance JSON writing

This module doesn't modifies files !! It computes, and returns dicts/arrays/text
for the calling script to write or print
"""

from __future__ import annotations
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


# Constants: mouse-specific gene families

# 37 mitochondrial (MT) gene Ensembl IDs (GRCm38)
MT_ENSEMBL_IDS = frozenset([
    "ENSMUSG00000064336", "ENSMUSG00000064337", "ENSMUSG00000064338",
    "ENSMUSG00000064339", "ENSMUSG00000064340", "ENSMUSG00000064341",
    "ENSMUSG00000064342", "ENSMUSG00000064343", "ENSMUSG00000064344",
    "ENSMUSG00000064345", "ENSMUSG00000064346", "ENSMUSG00000064347",
    "ENSMUSG00000064348", "ENSMUSG00000064349", "ENSMUSG00000064350",
    "ENSMUSG00000064351", "ENSMUSG00000064352", "ENSMUSG00000064353",
    "ENSMUSG00000064354", "ENSMUSG00000064355", "ENSMUSG00000064356",
    "ENSMUSG00000064357", "ENSMUSG00000064358", "ENSMUSG00000064359",
    "ENSMUSG00000064360", "ENSMUSG00000064361", "ENSMUSG00000064363",
    "ENSMUSG00000064364", "ENSMUSG00000064365", "ENSMUSG00000064366",
    "ENSMUSG00000064367", "ENSMUSG00000064368", "ENSMUSG00000064369",
    "ENSMUSG00000064370", "ENSMUSG00000064371", "ENSMUSG00000064372",
    "ENSMUSG00000065947",
])

# mouse hemoglobin Ensembl IDs (Hba/Hbb cluster — chr11, chr7)
# inc. adult, embryonic, & beta-like variants
HEMOGLOBIN_ENSEMBL_IDS = frozenset([
    "ENSMUSG00000069919", # Hba-a1
    "ENSMUSG00000069917", # Hba-a2
    "ENSMUSG00000073940", # Hbb-b1 (Hbb-bt in some annotations! double chekc)
    "ENSMUSG00000052305", # Hbb-bs
    "ENSMUSG00000052217", # Hbb-bh1
    "ENSMUSG00000093674", # Hbb-bh2
    "ENSMUSG00000059070", # Hbb-y
    "ENSMUSG00000078676", # Hbb-b2 / Hbb-bt
])


def is_ercc_id(gene_id: str) -> bool:
    """ERCC spike-ins start with 'ERCC-' (e.g., ERCC-00002)."""
    return str(gene_id).startswith("ERCC-")


def is_pseudogene_symbol(symbol: Optional[str]) -> bool:
    """
    Mouse predicted/pseudogene loci often have 'Gm' prefix followed by digits
    (e.g., Gm12345). These are uncharacterised loci with limited functional
    annotation. Aggressive filter — only used when --filter-pseudogenes is set.
    """
    if symbol is None or pd.isna(symbol):
        return False
    s = str(symbol)
    if not s.startswith("Gm"):
        return False
    # MUST be Gm followed by digits only
    rest = s[2:]
    return rest.isdigit() and len(rest) >= 3


def is_ribosomal_symbol(symbol: Optional[str]) -> bool:
    """
    Mouse ribosomal protein genes use Rpl* (large subunit) or Rps* (small
    subunit) symbols. ~80 genes total. Optional filter — useful when
    ribosomal-gene variance is dominating PCA / clustering, but they are
    real biology in some contexts
    """
    if symbol is None or pd.isna(symbol):
        return False
    s = str(symbol)
    return (s.startswith("Rpl") or s.startswith("Rps")) and len(s) > 3



# Gene QC — mandatory and optional filters
def apply_mandatory_gene_qc(
    tpm_matrix: pd.DataFrame,
    *,
    keep_mt: bool = False,
    min_samples_expressed_pct: float = 10.0,
    expression_threshold: float = 1.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply mandatory gene-level QC filters in fixed order:
      1. Remove ERCC spike-ins
      2. Remove zero-variance genes
      3. Remove low-expression genes (< threshold TPM in <pct% of samples)
      4. Remove MT genes (unless keep_mt=True)

    each filter logs N removed and example IDs

    PARAMETERS
    tpm_matrix : DataFrame, genes x samples (rows = Ensembl IDs)
    keep_mt : if True, MT step is skipped
    min_samples_expressed_pct : threshold (default 10.0 = 10%)
    expression_threshold : TPM cutoff (default 1.0)

    RETURNS
    filtered_tpm : DataFrame after all filters applied
    qc_log : dict summary of what was removed
    """
    n_samples = tpm_matrix.shape[1]
    log = {
        "n_genes_input": int(tpm_matrix.shape[0]),
        "n_samples": int(n_samples),
        "filters_applied": [],
        "filters_skipped": [],
        "expression_threshold": float(expression_threshold),
        "min_samples_expressed_pct": float(min_samples_expressed_pct),
    }
    current = tpm_matrix.copy()

    # 1. ERCC spike ins
    ercc_mask = current.index.to_series().apply(is_ercc_id)
    n_ercc = int(ercc_mask.sum())
    if n_ercc > 0:
        if verbose:
            print(f" Mandatory: removing {n_ercc} ERCC spike-in(s)")
            print(f" Examples: {current.index[ercc_mask].tolist()[:5]}")
        current = current.loc[~ercc_mask]
        log["filters_applied"].append({
            "filter": "ercc",
            "n_removed": n_ercc,
            "examples": current.index[ercc_mask].tolist()[:5] if False else [],
        })
    else:
        if verbose:
            print(f" Mandatory: 0 ERCC spike-ins present (skipped)")
        log["filters_skipped"].append("ercc (none present)")

    # 2. Zero-variance
    # variance across samples per gene
    gene_var = current.var(axis=1)
    zero_var_mask = (gene_var == 0)
    n_zero_var = int(zero_var_mask.sum())
    if verbose:
        print(f"Mandatory: removing {n_zero_var} zero-variance genes")
    log["filters_applied"].append({
        "filter": "zero_variance",
        "n_removed": n_zero_var,
    })
    current = current.loc[~zero_var_mask]

    # 3. Low-expression
    # number of samples in which each gene exceeds threshold
    n_samples_expressed = (current >= expression_threshold).sum(axis=1)
    min_samples = int(np.ceil(n_samples * min_samples_expressed_pct / 100.0))
    low_expr_mask = (n_samples_expressed < min_samples)
    n_low_expr = int(low_expr_mask.sum())
    if verbose:
        print(f"  Mandatory: removing {n_low_expr} low-expression genes "
              f"(< {expression_threshold} TPM in < {min_samples} of {n_samples} samples)")
    log["filters_applied"].append({
        "filter": "low_expression",
        "n_removed": n_low_expr,
        "expression_threshold_tpm": float(expression_threshold),
        "min_samples": min_samples,
    })
    current = current.loc[~low_expr_mask]

    # 4. MT
    if not keep_mt:
        mt_mask = current.index.to_series().isin(MT_ENSEMBL_IDS)
        n_mt = int(mt_mask.sum())
        if verbose:
            print(f"Mandatory: removing {n_mt} mitochondrial genes")
        current = current.loc[~mt_mask]
        log["filters_applied"].append({
            "filter": "mt",
            "n_removed": n_mt,
        })
    else:
        log["filters_skipped"].append("mt (--keep-mt set)")
        if verbose:
            print(f"  Mandatory MT: SKIPPED (--keep-mt)")

    log["n_genes_after_mandatory"] = int(current.shape[0])
    return current, log


def apply_sensitivity_gene_qc(
    tpm_matrix: pd.DataFrame,
    *,
    filter_hemoglobin: bool = False,
    filter_ribosomal: bool = False,
    filter_pseudogenes: bool = False,
    gene_symbol_map: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply optional sensitivity filters. None applied by default!! 

    PARAMETERS
    tpm_matrix : DataFrame, genes x samples
    filter_hemoglobin / filter_ribosomal / filter_pseudogenes : booleans
    gene_symbol_map : dict mapping ensembl_id -> symbol (needed for ribosomal/pseudogene)

    RETURNS
    filtered_tpm, qc_log
    """
    log = {
        "filters_applied": [],
        "filters_skipped": [],
    }
    current = tpm_matrix

    if filter_hemoglobin:
        mask = current.index.to_series().isin(HEMOGLOBIN_ENSEMBL_IDS)
        n = int(mask.sum())
        if verbose:
            print(f" Sensitivity: removing {n} hemoglobin gene(s)")
        current = current.loc[~mask]
        log["filters_applied"].append({"filter": "hemoglobin", "n_removed": n})
    else:
        log["filters_skipped"].append("hemoglobin (not requested)")

    if filter_ribosomal:
        if gene_symbol_map is None:
            print(f"  [WARN] --filter-ribosomal requires gene_symbol_map; skipping")
            log["filters_skipped"].append("ribosomal (no symbol map)")
        else:
            symbols = current.index.map(gene_symbol_map.get)
            mask = pd.Series(symbols, index=current.index).apply(is_ribosomal_symbol)
            n = int(mask.sum())
            if verbose:
                print(f" Sensitivity: removing {n} ribosomal gene(s) (Rpl*/Rps*)")
            current = current.loc[~mask]
            log["filters_applied"].append({"filter": "ribosomal", "n_removed": n})
    else:
        log["filters_skipped"].append("ribosomal (not requested)")

    if filter_pseudogenes:
        if gene_symbol_map is None:
            print(f"  [WARN] --filter-pseudogenes requires gene_symbol_map; skipping")
            log["filters_skipped"].append("pseudogenes (no symbol map)")
        else:
            symbols = current.index.map(gene_symbol_map.get)
            mask = pd.Series(symbols, index=current.index).apply(is_pseudogene_symbol)
            n = int(mask.sum())
            if verbose:
                print(f"Sensitivity: removing {n} predicted/pseudogene(s) (Gm prefix)")
            current = current.loc[~mask]
            log["filters_applied"].append({"filter": "pseudogenes", "n_removed": n})
    else:
        log["filters_skipped"].append("pseudogenes (not requested)")

    log["n_genes_after_sensitivity"] = int(current.shape[0])
    return current, log



# Sample QC — report only, never silent removal
def compute_sample_qc_report(
    tpm_matrix: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    expression_threshold: float = 1.0,
    library_size_n_sd: float = 3.0,
    detection_n_sd: float = 3.0,
    pca_n_sd: float = 3.0,
    verbose: bool = True,
) -> Dict:
    """
    Compute per-sample QC metrics and flag outliers. Never modifies input

    Returns dict with:
      - per_sample : DF indexed by sample_id with all QC metrics
      - flags : dict of {flag_name: list_of_sample_ids}
      - summary : text summary
    """
    samples = list(tpm_matrix.columns)
    n_samples = len(samples)

    # library size = sum of log1p(TPM) per sample
    log_tpm = np.log1p(tpm_matrix.values.astype(np.float32)) # genes × samples
    lib_sizes = log_tpm.sum(axis=0) # per sample

    # number of genes detected (TPM >= threshold)
    n_detected = (tpm_matrix.values >= expression_threshold).sum(axis=0)

    # Now build per-sample DF
    per_sample = pd.DataFrame({
        "sample_id": samples,
        "library_size_log1p_sum": lib_sizes,
        "n_genes_detected": n_detected,
    }).set_index("sample_id")

    # Library size outliers (z-score)
    lib_z = (lib_sizes - lib_sizes.mean()) / lib_sizes.std() if lib_sizes.std() > 0 else np.zeros_like(lib_sizes)
    per_sample["library_size_z"] = lib_z
    lib_outlier_mask = np.abs(lib_z) > library_size_n_sd
    per_sample["library_size_outlier"] = lib_outlier_mask

    # detection-count outliers
    det_z = (n_detected - n_detected.mean()) / n_detected.std() if n_detected.std() > 0 else np.zeros_like(n_detected, dtype=float)
    per_sample["n_detected_z"] = det_z
    det_outlier_mask = np.abs(det_z) > detection_n_sd
    per_sample["n_detected_outlier"] = det_outlier_mask

    # PCA on log1p data, flag samples > N SD on PC1 or PC2
    try:
        from sklearn.decomposition import PCA
        # z-score per sample first to mimic downstream input
        from scipy import stats
        zscored = stats.zscore(log_tpm.T, axis=1, nan_policy="omit")
        zscored = np.nan_to_num(zscored, nan=0.0)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(zscored)
        per_sample["PC1"] = pcs[:, 0]
        per_sample["PC2"] = pcs[:, 1]
        pc1_z = (pcs[:, 0] - pcs[:, 0].mean()) / pcs[:, 0].std() if pcs[:, 0].std() > 0 else np.zeros(n_samples)
        pc2_z = (pcs[:, 1] - pcs[:, 1].mean()) / pcs[:, 1].std() if pcs[:, 1].std() > 0 else np.zeros(n_samples)
        per_sample["PC1_z"] = pc1_z
        per_sample["PC2_z"] = pc2_z
        pca_outlier_mask = (np.abs(pc1_z) > pca_n_sd) | (np.abs(pc2_z) > pca_n_sd)
        per_sample["pca_outlier"] = pca_outlier_mask
    except Exception as e:
        print(f"  [WARN] PCA computation failed: {e}")
        per_sample["PC1"] = np.nan
        per_sample["PC2"] = np.nan
        per_sample["pca_outlier"] = False
        pca_outlier_mask = np.zeros(n_samples, dtype=bool)

    # Sex-label consistency check via Xist expression !!! 
    # Xist (ENSMUSG00000086503) is highly expressed in females, silenced in males
    # POST-RUN FIX: Replace binary sex_mismatch with graded sex_qc_flag
    # Low Xist in a labeled female is informational, not a hard mismatch:
    # Xist can vary biologically (individual variation, library quirks)
    # Hard contradictions (e.g., Y-gene-expressing "female") would warrant
    # sex_mismatch=True, but Xist alone is not sufficient
    # metadata confirmation - samples correctly labeled by sex 
    xist_id = "ENSMUSG00000086503"
    if xist_id in tpm_matrix.index:
        xist_expr = tpm_matrix.loc[xist_id].values
        per_sample["Xist_TPM"] = xist_expr

        # get sex from metadata
        meta_indexed = meta.set_index("sample_id") if "sample_id" in meta.columns else meta
        sample_to_sex = {}
        for s in samples:
            if s in meta_indexed.index:
                sex_val = (str(meta_indexed.loc[s, "sex"]).lower()
                           if "sex" in meta_indexed.columns else "unknown")
                sample_to_sex[s] = sex_val

        # categorize each sample
        flag_categories = []
        for s, expr in zip(samples, xist_expr):
            sex = sample_to_sex.get(s, "unknown")
            if sex == "female":
                if expr < 1.0:
                    flag_categories.append("low_xist_in_female")
                else:
                    flag_categories.append("consistent")
            elif sex == "male":
                if expr > 5.0:
                    flag_categories.append("high_xist_in_male")
                else:
                    flag_categories.append("consistent")
            else:
                flag_categories.append("unknown")

        per_sample["sex_label"] = [sample_to_sex.get(s, "unknown") for s in samples]
        per_sample["sex_qc_flag"] = flag_categories
        # Keep sex_mismatch boolean for backward compatibility, but use it
        # ONLY for hard contradictions (high Xist in male)/ Low Xist in female
        # alone is informational and does NOT trigger sex_mismatch
        per_sample["sex_mismatch"] = (per_sample["sex_qc_flag"] == "high_xist_in_male")
    else:
        if verbose:
            print(f" [WARN] Xist ({xist_id}) not in expression matrix — skipping sex consistency check")
        per_sample["Xist_TPM"] = np.nan
        per_sample["sex_label"] = "unknown"
        per_sample["sex_qc_flag"] = "xist_not_available"
        per_sample["sex_mismatch"] = False

    # Within-group outliers: PC1/PC2 z-score within each group
    meta_indexed = meta.set_index("sample_id") if "sample_id" in meta.columns else meta
    if "strat_key" in meta_indexed.columns and "PC1" in per_sample.columns:
        per_sample["strat_key"] = [meta_indexed.loc[s, "strat_key"] if s in meta_indexed.index else "unknown"
                                     for s in samples]
        within_group_outliers = []
        for grp_name, grp_idx in per_sample.groupby("strat_key").groups.items():
            grp = per_sample.loc[grp_idx]
            if len(grp) < 3:
                continue   # too small to compute z-scores meaningfully :/ 
            for col in ("PC1", "PC2"):
                if col not in grp.columns:
                    continue
                vals = grp[col].values
                if np.std(vals) == 0:
                    continue
                z = (vals - vals.mean()) / vals.std()
                outliers_in_grp = grp.index[np.abs(z) > 2.5].tolist()
                within_group_outliers.extend(outliers_in_grp)
        per_sample["within_group_outlier"] = per_sample.index.isin(within_group_outliers)
    else:
        per_sample["within_group_outlier"] = False

    # Aggregate flags. POST-RUN FIX: any_flag triggers on hard outliers only
    # low_xist_in_female is informational and does NOT contribute to any_flag
    # high_xist_in_male is the only sex-related condition that contributes
    per_sample["any_flag"] = (
        per_sample["library_size_outlier"]
        | per_sample["n_detected_outlier"]
        | per_sample["pca_outlier"]
        | per_sample["sex_mismatch"]
        | per_sample["within_group_outlier"]
    )

    flags = {
        "library_size_outlier": per_sample.index[per_sample["library_size_outlier"]].tolist(),
        "n_detected_outlier": per_sample.index[per_sample["n_detected_outlier"]].tolist(),
        "pca_outlier": per_sample.index[per_sample["pca_outlier"]].tolist(),
        "sex_mismatch": per_sample.index[per_sample["sex_mismatch"]].tolist(),
        "low_xist_in_female": per_sample.index[per_sample["sex_qc_flag"] == "low_xist_in_female"].tolist() if "sex_qc_flag" in per_sample.columns else [],
        "high_xist_in_male": per_sample.index[per_sample["sex_qc_flag"] == "high_xist_in_male"].tolist() if "sex_qc_flag" in per_sample.columns else [],
        "within_group_outlier": per_sample.index[per_sample["within_group_outlier"]].tolist(),
        "any_flag": per_sample.index[per_sample["any_flag"]].tolist(),
    }

    summary_lines = [
        "Sample QC summary",
        "=" * 60,
        f"  Total samples: {n_samples}",
        f"  Library-size outliers (|z|>{library_size_n_sd}): {len(flags['library_size_outlier'])}",
        f"  Gene-detection outliers (|z|>{detection_n_sd}): {len(flags['n_detected_outlier'])}",
        f"  PCA outliers (|z|>{pca_n_sd} on PC1/PC2): {len(flags['pca_outlier'])}",
        f"  Sex hard mismatches (Y in female): {len(flags['sex_mismatch'])}",
        f"  Sex info: low Xist in female: {len(flags.get('low_xist_in_female', []))}",
        f"  Sex info: high Xist in male:  {len(flags.get('high_xist_in_male', []))}",
        f"  Within-group outliers: {len(flags['within_group_outlier'])}",
        f"  ─────────────────────────────────────────", # breakup 
        f"  Samples flagged in ≥1 hard category:   {len(flags['any_flag'])}",
        f"  (low_xist_in_female is informational — not in any_flag)",
    ]
    # show samples flagged for HARD outlier conditions
    if flags["any_flag"]:
        summary_lines.append("\n HARD-flagged samples (review before training):")
        for sid in sorted(flags["any_flag"])[:20]:
            row = per_sample.loc[sid]
            tags = []
            if row["library_size_outlier"]: tags.append(f"libsize z={row['library_size_z']:+.2f}")
            if row["n_detected_outlier"]: tags.append(f"ndet z={row['n_detected_z']:+.2f}")
            if row.get("pca_outlier", False): tags.append(f"PCA z=({row.get('PC1_z',0):+.1f},{row.get('PC2_z',0):+.1f})")
            if row.get("sex_mismatch", False): tags.append(f"sex_HARD={row.get('sex_label')} Xist={row.get('Xist_TPM',np.nan):.1f}")
            if row.get("within_group_outlier", False): tags.append("wg_out")
            summary_lines.append(f"    {sid}: {'; '.join(tags)}")
        if len(flags["any_flag"]) > 20:
            summary_lines.append(f" ... and {len(flags['any_flag']) - 20} more (see CSV)")

    # Also show informational sex-flag samples (NOT in any_flag, just info)
    info_sex = flags.get("low_xist_in_female", []) + flags.get("high_xist_in_male", [])
    info_sex = [s for s in info_sex if s not in flags["any_flag"]]
    if info_sex:
        summary_lines.append("\n INFORMATIONAL sex-QC flags (not blocking; review if cohort is sex-balanced):")
        for sid in sorted(info_sex)[:20]:
            row = per_sample.loc[sid]
            flag = row.get("sex_qc_flag", "")
            xist = row.get("Xist_TPM", np.nan)
            sex_lbl = row.get("sex_label", "?")
            summary_lines.append(f"    {sid}: {flag}  sex={sex_lbl}  Xist={xist:.2f} TPM")

    summary = "\n".join(summary_lines)
    if verbose:
        print(summary)

    return {
        "per_sample": per_sample,
        "flags": flags,
        "summary": summary,
        "thresholds": {
            "library_size_n_sd": library_size_n_sd,
            "detection_n_sd": detection_n_sd,
            "pca_n_sd": pca_n_sd,
            "expression_threshold": expression_threshold,
        },
    }



# Group-size warnings
def compute_group_size_warnings(meta: pd.DataFrame, *,
                                 strict_threshold: int = 3,
                                 mild_threshold: int = 5,
                                 verbose: bool = True) -> Dict:
    """
    group-level QC: warn about (dataset x genotype x age x sex) groups that
    are too small for stable cross-validation

    Tiers:
      n < strict_threshold (default 3) --> STRONG warning, recommend exclusion
      n < mild_threshold (default 5) --> MILD warning, note
      n >= mild_threshold --> silent

    Returns dict with:
      - group_sizes : DataFrame (strat_key, n_samples)
      - warnings : list of dicts (level, group, n)
      - summary : human-readable text
    """
    if "strat_key" not in meta.columns:
        return {"group_sizes": pd.DataFrame(), "warnings": [],
                "summary": "[group QC skipped: no strat_key column in metadata]"}

    sizes = meta.groupby("strat_key").size().reset_index(name="n_samples")
    sizes = sizes.sort_values("n_samples")

    warnings = []
    for _, row in sizes.iterrows():
        if row["n_samples"] < strict_threshold:
            warnings.append({
                "level": "STRONG",
                "group": row["strat_key"],
                "n": int(row["n_samples"]),
                "message": "n<3, recommend exclusion or merging",
            })
        elif row["n_samples"] < mild_threshold:
            warnings.append({
                "level": "MILD",
                "group": row["strat_key"],
                "n": int(row["n_samples"]),
                "message": "n<5, fold metrics will be noisy",
            })

    summary_lines = [
        "Group-size summary",
        "=" * 60,
        f" Total groups: {len(sizes)}",
        f" Total samples: {int(sizes['n_samples'].sum())}",
        f" Smallest group: n={int(sizes['n_samples'].min())}",
        f" Largest group:  n={int(sizes['n_samples'].max())}",
        f" Median group:   n={int(sizes['n_samples'].median())}",
        "",
    ]
    if warnings:
        summary_lines.append(f"  Warnings ({len(warnings)} groups flagged):")
        for w in warnings:
            tag = "[STRONG]" if w["level"] == "STRONG" else "[mild]  "
            summary_lines.append(f" {tag} {w['group']:<45s} n={w['n']}  {w['message']}")
    else:
        summary_lines.append(" no group-size warnings (all groups n≥{0})".format(mild_threshold))

    summary_lines.append("")
    summary_lines.append(" Full group sizes:")
    for _, row in sizes.iterrows():
        marker = ""
        if row["n_samples"] < strict_threshold: marker = " ⚠⚠"
        elif row["n_samples"] < mild_threshold: marker = " ⚠"
        summary_lines.append(f"    {row['strat_key']:<45s} n={int(row['n_samples'])}{marker}")

    summary = "\n".join(summary_lines)
    if verbose:
        print(summary)

    return {
        "group_sizes": sizes,
        "warnings": warnings,
        "summary": summary,
        "n_strong_warnings": sum(1 for w in warnings if w["level"] == "STRONG"),
        "n_mild_warnings": sum(1 for w in warnings if w["level"] == "MILD"),
    }


# Provenance JSON
def hash_array(arr: np.ndarray) -> str:
    """SHA256 hex digest of array bytes — for detecting silent overwrites"""
    h = hashlib.sha256()
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def write_provenance(
    output_path: Path,
    *,
    script_name: str,
    inputs: Dict[str, dict],
    args: dict,
    outputs: Dict[str, dict],
    extras: Optional[Dict] = None,
):
    """
    Write a JSON provenance file next to a primary output (e.g.
    combined_expression.h5 --> combined_expression.provenance.json)

    Parameters
    ----------
    output_path : path to the provenance JSON to write
    script_name : __file__ of the calling script
    inputs : dict of {name: {path, size_bytes, mtime, sha256(optional)}}
    args : dict of CLI arguments / parameters
    outputs : dict of {name: {path, size_bytes, sha256(optional)}}
    extras : any additional info (gene counts, sample counts, etc.)
    """
    record = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "script": script_name,
        "args": args,
        "inputs": inputs,
        "outputs": outputs,
        "extras": extras or {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, default=str)


def file_metadata(path: Path, include_hash: bool = False) -> dict:
    """Build a {path, size, mtime, sha256?} dict for provenance."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    rec = {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if include_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        rec["sha256"] = h.hexdigest()
    return rec
