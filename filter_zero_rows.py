#!/usr/bin/env python3
"""
filter_zero_heavy_rows.py

Remove rows from a CSV if they contain at least N zero values (default: 5)
within the selected sensor columns.

Default columns: R1..R17, T, H, CO2, H2S, CH2O
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_FEATURES = [*(f"R{i}" for i in range(1, 18)), "T", "H", "CO2", "H2S", "CH2O"]

def main():
    ap = argparse.ArgumentParser(description="Drop rows having >= N zeros in selected columns.")
    ap.add_argument("input", help="Path to input CSV")
    ap.add_argument("-o", "--output", help="Path to output CSV (default: input basename + _filtered.csv)")
    ap.add_argument("-t", "--threshold", type=int, default=5,
                    help="Drop any row with >= this many zeros (default: 5)")
    ap.add_argument("--columns", nargs="*", default=None,
                    help="Explicit list of columns to check (default: auto: only the 22 sensor features present)")
    ap.add_argument("--numeric-only", action="store_true",
                    help="If set, ignore --columns and count zeros across ALL numeric columns")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not write a file; just print how many rows would be removed.")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    df = pd.read_csv(inp)

    # Choose which columns to inspect for zeros
    if args.numeric_only:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    elif args.columns:
        cols = [c for c in args.columns if c in df.columns]
        missing = [c for c in args.columns if c not in df.columns]
        if missing:
            print(f"Warning: these requested columns are missing and will be ignored: {missing}")
    else:
        # default: only the 22 sensor features that are actually present
        cols = [c for c in DEFAULT_FEATURES if c in df.columns]

    if not cols:
        raise SystemExit("No columns selected for zero-counting. "
                         "Use --columns R1 R2 ... or --numeric-only to pick columns.")

    # Coerce the chosen columns to numeric (strings like "0.0" -> 0.0), keep others untouched
    num = df[cols].apply(pd.to_numeric, errors="coerce")

    # Count zeros per row across the selected columns
    # (pandas eq == elementwise equality; sum axis=1 counts True values across columns)
    zeros_per_row = num.eq(0.0).sum(axis=1)  # :contentReference[oaicite:1]{index=1}

    # Keep rows with strictly fewer than threshold zeros
    keep_mask = zeros_per_row < args.threshold
    removed = int((~keep_mask).sum())
    kept = int(keep_mask.sum())

    print("=== Zero-heavy row filter ===")
    print(f"Input file          : {inp}")
    print(f"Rows (total)        : {len(df):,}")
    print(f"Columns checked     : {len(cols)} -> {cols}")
    print(f"Threshold (>= zeros): {args.threshold}")
    print(f"Rows removed        : {removed:,}")
    print(f"Rows kept           : {kept:,}")

    if args.dry_run:
        return

    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_filtered.csv")
    df_filtered = df.loc[keep_mask].copy()
    df_filtered.to_csv(out, index=False)
    print(f"Saved filtered CSV to: {out.resolve()}")

if __name__ == "__main__":
    main()
