#!/usr/bin/env python3
"""
compare_connectivity_with_cortex.py

Reads Allen connectivity CSVs (projection_density_*_to_*.csv) and your CORTEX
inter-area weights from EITHER CSV or Excel (.xlsx/.xls), computes a Spearman
rank correlation, and saves a labeled scatter.

Usage:
  python compare_connectivity_with_cortex.py --allen-dir allen_connectivity_outputs --cortex cortex_weights.xlsx
  # or (if CSV) --cortex cortex_weights.csv

Optional:
  --metric projection_density|projection_energy
  --out connectivity_cortex_vs_allen.png
  --force-encoding utf-16   (only for CSV)
  --force-sep ;             (only for CSV)
"""

import argparse, os, glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def read_csv_smart(path, forced_encoding=None, forced_sep=None, what="file"):
    tried = []
    encs = [forced_encoding] if forced_encoding else ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'cp1252', 'latin1']
    seps = [forced_sep] if forced_sep else [None, ',', ';', '\t', '|']
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine='python')
                print(f"[read_csv] {what}: OK with encoding={enc or 'auto'} sep={repr(sep)}")
                return df
            except Exception as e:
                tried.append(f"encoding={enc or 'auto'} sep={repr(sep)} -> {e.__class__.__name__}")
                last_err = e
    raise RuntimeError(f"Failed to read {what} at {path}. Tried: " + " | ".join(tried)) from last_err

def read_weights_any(path, force_encoding=None, force_sep=None):
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(path)  # requires openpyxl for .xlsx
            print(f"[read_excel] weights: OK ({path})")
            return df
        except ImportError as e:
            print("ERROR: Reading .xlsx needs 'openpyxl'. Run: pip install openpyxl", file=sys.stderr)
            raise
    else:
        return read_csv_smart(path, forced_encoding=force_encoding, forced_sep=force_sep, what="cortex_weights")

def rank_avg_ties(x):
    return pd.Series(x).rank(method='average').to_numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allen-dir", required=True, help="Folder with projection_density_<SRC>_to_<TGT>.csv files")
    p.add_argument("--cortex", required=True, help="CORTEX weights file (.xlsx/.xls/.csv) with columns: source,target,weight")
    p.add_argument("--metric", default="projection_density", choices=["projection_density", "projection_energy"])
    p.add_argument("--out", default="connectivity_cortex_vs_allen.png")
    p.add_argument("--force-encoding", default=None, help="Force a file encoding (CSV only)")
    p.add_argument("--force-sep", default=None, help="Force a delimiter (CSV only)")
    args = p.parse_args()

    # 1) Load CORTEX weights (Excel or CSV)
    cw = read_weights_any(args.cortex, force_encoding=args.force_encoding, force_sep=args.force_sep)
    # Normalize columns
    colmap = {c.lower(): c for c in cw.columns}
    required = {"source","target","weight"}
    if not required.issubset(set(colmap.keys())):
        raise ValueError(f"CORTEX weights file must have columns: source,target,weight (found: {list(cw.columns)})")
    cw = cw.rename(columns={colmap["source"]:"source", colmap["target"]:"target", colmap["weight"]:"weight"})
    cw["key"] = cw["source"].astype(str).str.upper() + "→" + cw["target"].astype(str).str.upper()
    cw = cw[["key","weight"]].copy()

    # 2) Read all Allen per-experiment CSVs and aggregate per pair
    rows = []
    pattern = os.path.join(args.allen_dir, f"{args.metric}_*_to_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No Allen CSVs matched: {pattern}")
    for path in files:
        name = os.path.basename(path)
        try:
            stem = name[:-4]
            _, tail = stem.split(f"{args.metric}_", 1)
            src, tgt = tail.split("_to_")
            df = read_csv_smart(path, what=name)
            col = args.metric if args.metric in df.columns else None
            if col is None:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    continue
                col = num_cols[-1]
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            rows.append({"key": f"{src.upper()}→{tgt.upper()}",
                         "allen_mean": float(vals.mean()),
                         "allen_sum":  float(vals.sum())})
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue

    adf = pd.DataFrame(rows)
    if adf.empty:
        print("No Allen pairs parsed. Check filenames and metric.")
        return

    # 3) Join
    merged = pd.merge(cw, adf, on="key", how="inner")
    if merged.empty:
        print("No overlapping pairs between CORTEX weights and Allen CSVs.")
        return

    # 4) Spearman (average ranks for ties)
    r_w = rank_avg_ties(merged["weight"].to_numpy())
    r_a = rank_avg_ties(merged["allen_mean"].to_numpy())
    r_w -= r_w.mean(); r_a -= r_a.mean()
    denom = np.sqrt((r_w**2).sum() * (r_a**2).sum())
    rho = float((r_w * r_a).sum() / denom) if denom > 0 else float("nan")

    # 5) Plot
    x = np.log10(np.maximum(merged["allen_mean"].to_numpy(), 1e-12))
    y = merged["weight"].to_numpy()
    plt.figure(figsize=(7,5))
    plt.scatter(x, y)
    for _, row in merged.iterrows():
        plt.annotate(row["key"], (np.log10(max(row["allen_mean"], 1e-12)), row["weight"]),
                     fontsize=8, alpha=0.6)
    plt.xlabel(f"log10(Allen {args.metric} mean)")
    plt.ylabel("CORTEX weight (normalized)")
    plt.title(f"CORTEX vs Allen ({args.metric}) — Spearman rho={rho:.2f}, n={len(merged)})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=170)
    print(f"\nSaved: {args.out}")
    print("\nMerged preview:\n", merged.sort_values("weight", ascending=False).head(30))
    print(f"\nSpearman rho = {rho:.3f}  (using mean per pair)")

if __name__ == "__main__":
    main()
