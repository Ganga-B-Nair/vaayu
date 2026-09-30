"""Profile whatever air-quality CSVs are sitting in data/raw/.

Schema-agnostic on purpose: it reports what IS there rather than assuming.
Run this first, before any analysis is designed.

    python scripts/profile_raw.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pd.set_option("display.width", 200)

files = sorted(RAW.glob("*.csv"))
if not files:
    raise SystemExit(f"No CSVs found in {RAW}. Put the Kaggle files there first.")

print(f"Found {len(files)} file(s) in {RAW}\n")

for f in files:
    print("=" * 74)
    print(f"FILE: {f.name}   ({f.stat().st_size / 1e6:.1f} MB)")
    print("=" * 74)

    # read a chunk first so a huge file doesn't stall the profile
    head = pd.read_csv(f, nrows=5)
    print(f"columns ({len(head.columns)}): {list(head.columns)}\n")

    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        print(f"  !! could not read fully: {e}\n")
        continue

    print(f"rows: {len(df):,}")

    # date-ish columns
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time")):
            d = pd.to_datetime(df[c], errors="coerce")
            ok = d.notna().sum()
            if ok:
                span = (d.max() - d.min()).days
                print(f"  {c}: {d.min()} -> {d.max()}  ({span} days, "
                      f"{ok:,} parseable, {len(df)-ok:,} not)")

    print("\n--- missingness & range ---")
    rep = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(),
    })
    num = df.select_dtypes(include=[np.number])
    if len(num.columns):
        rep["min"] = num.min().round(2)
        rep["median"] = num.median().round(2)
        rep["max"] = num.max().round(2)
    print(rep.sort_values("pct_missing", ascending=False).to_string())

    # categorical distributions worth seeing
    for c in df.columns:
        if df[c].dtype == object and 1 < df[c].nunique() <= 30:
            print(f"\n--- {c} ---")
            print(df[c].value_counts(dropna=False).to_string())

    # per-entity coverage: which cities/stations are usable?
    key = next((c for c in ("City", "StationId", "station", "city")
                if c in df.columns), None)
    tgt = next((c for c in ("PM2.5", "AQI", "pm25") if c in df.columns), None)
    if key and tgt:
        cov = (df.groupby(key)[tgt]
               .agg(n="size", present=lambda s: s.notna().sum())
               .assign(coverage=lambda x: (x["present"] / x["n"]).round(3))
               .sort_values("coverage", ascending=False))
        print(f"\n--- {tgt} coverage by {key} (top 10 / bottom 10) ---")
        print(cov.head(10).to_string())
        print("...")
        print(cov.tail(10).to_string())
        print(f"\n  entities with >=80% coverage: "
              f"{(cov['coverage'] >= 0.8).sum()} of {len(cov)}")
    print()