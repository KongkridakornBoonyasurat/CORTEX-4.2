#!/usr/bin/env python3
"""
allen_connectivity_check.py

Compute a simple Allen Institute Mouse Connectivity metric from a SOURCE area
to a TARGET area, then save per-experiment values to CSV. Use this to sanity-check
and later correlate against your CORTEX inter-area weights.

Example:
  python allen_connectivity_check.py --source VISp --target MOp --verbose --max-exps 5
  python allen_connectivity_check.py --source VISp --target ALM --metric projection_energy --hemi left --verbose

Notes:
- Uses MouseConnectivityCache.get_structure_unionizes() under the hood.
- Some AllenSDK builds lack certain kwargs (e.g., include_descendants or singular/plural IDs).
  This script handles those differences with try/except.
- First run may download large caches; use --verbose to see progress.
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
except Exception as e:
    print("ERROR: allensdk not available. Install with: pip install allensdk", file=sys.stderr)
    raise

OUTDIR = Path("allen_connectivity_outputs")


def lookup_structure_id_by_acronym(structure_tree, acronym: str) -> int:
    alias = {
        "ALM": "MOs",     # anterolateral motor → MOs
        "PMd": "MOs",     # common alias mapping
        "PMv": "MOs",
        "V1":  "VISp",    # primary visual
        "S1":  "SSp",     # primary somatosensory
        "M1":  "MOp",     # primary motor
        "M2":  "MOs",     # secondary motor
    }
    ac = alias.get(acronym, acronym)
    recs = structure_tree.get_structures_by_acronym([ac])
    if not recs:
        raise ValueError(f"Unknown Allen acronym: {acronym} (tried alias {ac})")
    return int(recs[0]['id'])


def get_unionizes_for_experiment(
    mcc: MouseConnectivityCache,
    structure_tree,
    experiment_id: int,
    target_id: int,
    hemi_ids: list[int],
    verbose: bool = False
):
    """
    Robust wrapper around get_structure_unionizes. Some SDKs require experiment_ids (plural)
    and may not support include_descendants; we fallback to manual descendant expansion.
    """
    struct_ids = [int(target_id)]

    # Try with include_descendants (newer SDKs + plural experiment_ids)
    try:
        if verbose:
            print(f"    -> unionizes(experiment_ids=[{experiment_id}], include_descendants=True)")
        df_u = mcc.get_structure_unionizes(
            experiment_ids=[int(experiment_id)],
            is_injection=False,
            structure_ids=struct_ids,
            hemisphere_ids=hemi_ids,
            include_descendants=True
        )
        return df_u
    except TypeError:
        # Fallback: manual descendant expansion (older SDKs)
        if verbose:
            print("    -> include_descendants not supported; expanding descendants manually")
        desc = structure_tree.descendant_ids([int(target_id)])[0]  # list of ints
        struct_ids = [int(target_id)] + [int(x) for x in desc]
        # Some SDKs still want experiment_ids (plural)
        df_u = mcc.get_structure_unionizes(
            experiment_ids=[int(experiment_id)],
            is_injection=False,
            structure_ids=struct_ids,
            hemisphere_ids=hemi_ids
        )
        return df_u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="VISp",
                        help="Allen acronym of source area, e.g., VISp")
    parser.add_argument("--target", type=str, default="MOp",
                        help="Allen acronym of target area, e.g., MOp")
    parser.add_argument("--resolution", type=int, default=100,
                        help="MouseConnectivityCache voxel size (µm)")
    parser.add_argument("--max-exps", type=int, default=10,
                        help="Cap number of source injection experiments to process")
    parser.add_argument("--hemi", type=str, default="both",
                        choices=["both", "left", "right"],
                        help="Hemisphere(s) for unionizes aggregation")
    parser.add_argument("--metric", type=str, default="projection_density",
                        choices=["projection_density", "projection_energy"],
                        help="Which unionizes metric to aggregate")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress")
    args = parser.parse_args()

    # Map hemisphere flag to Allen hemisphere IDs
    # Using [1,2] for 'both' is broadly compatible; some builds dislike [3].
    hemi_map = {"both": [1, 2], "left": [1], "right": [2]}
    hemi_ids = hemi_map[args.hemi]

    if args.verbose:
        print(f"[1/5] Initializing MouseConnectivityCache(resolution={args.resolution} µm)")
    mcc = MouseConnectivityCache(resolution=args.resolution)

    if args.verbose:
        print("[2/5] Loading structure tree…")
    st = mcc.get_structure_tree()

    # Resolve acronyms to IDs
    src_id = lookup_structure_id_by_acronym(st, args.source)
    tgt_id = lookup_structure_id_by_acronym(st, args.target)
    if args.verbose:
        print(f"[3/5] Source={args.source} (id={src_id}), Target={args.target} (id={tgt_id}), "
              f"Hemi={args.hemi}, Metric={args.metric}")

    if args.verbose:
        print("[4/5] Querying injection experiments for the source…")
    exps = mcc.get_experiments(dataframe=True, injection_structure_ids=[int(src_id)])
    if exps is None or len(exps) == 0:
        print(f"No injection experiments found with source={args.source}.")
        sys.exit(1)

    total = len(exps)
    exps = exps.head(args.max_exps)
    if args.verbose:
        print(f"Found {total} experiments. Limiting to {len(exps)}.")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    rows = []
    vals = []

    if args.verbose and len(exps) > 0:
        print("[5/5] Downloading unionizes per experiment…")

    for idx, eid in enumerate(exps["id"].values, start=1):
        eid_int = int(eid)
        if args.verbose:
            print(f"  ({idx}/{len(exps)}) experiment {eid_int}:")

        try:
            df_u = get_unionizes_for_experiment(
                mcc=mcc,
                structure_tree=st,
                experiment_id=eid_int,
                target_id=tgt_id,
                hemi_ids=hemi_ids,
                verbose=args.verbose
            )
        except Exception as e:
            if args.verbose:
                print(f"    -> FAILED: {e}")
            continue

        if df_u is None or len(df_u) == 0:
            if args.verbose:
                print("    -> no unionizes rows")
            continue

        # Aggregate chosen metric across the target (and descendants/hemispheres)
        if args.metric not in df_u.columns:
            if args.verbose:
                print(f"    -> metric '{args.metric}' not in unionizes columns; available: {list(df_u.columns)}")
            continue

        val = float(df_u[args.metric].sum())
        vals.append(val)
        rows.append({"experiment_id": eid_int, args.metric: val})
        if args.verbose:
            print(f"    -> {args.metric} sum = {val:.6f}")

    # Summary + CSV
    mean_v = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    med_v = float(np.median(vals)) if len(vals) > 0 else float("nan")
    std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    print(f"\nSummary {args.source} -> {args.target}  (n={len(vals)} of {len(exps)} used)")
    print(f"  mean {args.metric}:   {mean_v:.6f}")
    print(f"  median {args.metric}: {med_v:.6f}")
    print(f"  std {args.metric}:    {std_v:.6f}")

    out_csv = OUTDIR / f"{args.metric}_{args.source}_to_{args.target}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved per-experiment CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
