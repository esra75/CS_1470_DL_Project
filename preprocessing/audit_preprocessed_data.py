#!/usr/bin/env python3
"""
audit_preprocessed_data.py

Read-only audit of preprocessed H5 files in a data directory. So, for each H5,
reports:
  - shape and dtype
  - sample/gene counts
  - whether SHA256 in attrs matches the actual X bytes (corruption check)
  - all attrs (preprocessing config, ablation mode, etc)
  - matching .provenance.json (parsed and summarized) if present

Usage
-----
    python audit_preprocessed_data.py /path/to/data_dir
    python audit_preprocessed_data.py /path/to/data_dir --verify-hash

The --verify-hash flag triggers a SHA256 recompute on the X array — slower
(~5-30s for our matrix sizes) but tells u definitively if any H5 has been
silently corrupted since preprocessing

Without --verify-hash, just reads attrs and reports what each file claims
to contain, plus does a row/column count sanity check
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np


def hash_x(X: np.ndarray) -> str:
    return hashlib.sha256(X.tobytes()).hexdigest()

def audit_h5(path: Path, verify_hash: bool):
    name = path.name
    print("=" * 76)
    print(f"FILE: {name}")
    print("=" * 76)

    try:
        with h5py.File(path, "r") as f:
            # datasets present
            keys = list(f.keys())
            print(f"  Datasets: {keys}")

            X = f["X"][:]
            print(f"  X shape:   {X.shape}  dtype={X.dtype}")
            print(f"  X stats:   min={X.min():.4f}  max={X.max():.4f}  "
                  f"mean={X.mean():.4f}  std={X.std():.4f}")

            if "gene_names" in keys:
                genes = np.array(f["gene_names"]).astype(str)
                print(f"  Genes:     {len(genes)}  e.g. {genes[0]} ... {genes[-1]}")

            if "sample_ids" in keys:
                sids = np.array(f["sample_ids"]).astype(str)
                print(f"  Samples:   {len(sids)}  e.g. {sids[0]} ... {sids[-1]}")

            # attributes (preprocessing config, hashes, etc)
            attrs = {k: f.attrs[k] for k in f.attrs}
            if attrs:
                print(f"  Attrs:")
                for k, v in attrs.items():
                    if k == "X_sha256":
                        print(f"    {k:<28} {str(v)[:16]}...")
                    else:
                        print(f"    {k:<28} {v}")
            else:
                print(f"  Attrs:     (none)")

            # verify hash 
            stored_hash = str(attrs.get("X_sha256", ""))
            if not stored_hash or stored_hash == "MISSING!":
                print(f"No X_sha256 in attrs — preprocessing version "
                      f"predates hash-stamping or this file was hand-crafted")
            elif verify_hash:
                actual_hash = hash_x(X)
                if actual_hash == stored_hash:
                    print(f" SHA256 verified - X matches stored hash")
                else:
                    print(f" SHA256 MISMATCH!!:")
                    print(f" stored:  {stored_hash[:16]}...")
                    print(f" actual:  {actual_hash[:16]}...")
                    return False
            else:
                print(f" [hash check skipped — pass --verify-hash to recompute]")

    except Exception as e:
        print(f" Error reading file: {e}")
        return False

    # provenance JSON sidecar
    prov_path = path.with_suffix("").with_suffix(".provenance.json")
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
            print(f"  Provenance: {prov_path.name}")
            for k in ["generator", "generator_version", "generated_at",
                      "ablation_mode", "ablation_source",
                      "preprocess_version"]:
                if k in prov:
                    print(f"    {k:<22} {prov[k]}")
            if "extras" in prov:
                for k, v in prov["extras"].items():
                    if k == "X_sha256":
                        print(f" extras.{k:<14}   {str(v)[:16]}...")
                    else:
                        print(f" extras.{k:<14}   {v}")
        except Exception as e:
            print(f" [WARN] could not parse {prov_path.name}: {e}")
    else:
        print(f" Provenance: (no .provenance.json)")

    print()
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", help="directory containing H5 files")
    p.add_argument("--verify-hash", action="store_true",
                   help="recompute SHA256 of X and compare to stored hash")
    p.add_argument("--pattern", default="*.h5",
                   help="Filename glob (default: *.h5)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"ERROR - Directory not found: {data_dir}")

    h5_files = sorted(data_dir.glob(args.pattern))
    # skip backups
    h5_files = [f for f in h5_files if not f.name.endswith(".bak")]

    if not h5_files:
        sys.exit(f"[ERROR] No H5 files matching '{args.pattern}' in {data_dir}")

    print(f"Auditing {len(h5_files)} H5 files in {data_dir}")
    if args.verify_hash:
        print("(--verify-hash: recomputing SHA256)")
    print()

    ok = []
    issues = []
    for path in h5_files:
        result = audit_h5(path, args.verify_hash)
        if result is False:
            issues.append(path.name)
        else:
            ok.append(path.name)

    # Summary table
    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"Files audited: {len(h5_files)}")
    print(f"OK: {len(ok)}")
    if issues:
        print(f" Issues: {len(issues)}")
        for f in issues:
            print(f" - {f}")
    else:
        print(f" Issues: none")

    # Cross-file consistency table
    print()
    print("quick cross-file consistency check:")
    print(f" {'file':<55} {'samples':>7} {'genes':>6}  ablation_mode")
    print(f" {'-'*55} {'-'*7:>7} {'-'*6:>6}  {'-'*15}")
    for path in h5_files:
        try:
            with h5py.File(path, "r") as f:
                ns = f["X"].shape[0]
                ng = f["X"].shape[1]
                ab = str(f.attrs.get("ablation_mode", ""))
            print(f"  {path.name:<55} {ns:>7} {ng:>6}  {ab}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
