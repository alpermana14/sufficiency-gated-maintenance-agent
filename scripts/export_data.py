"""Export the conveyor table from MySQL to data/conveyor_export.csv.

Run this ONCE (with backend/.env configured) so every experiment is reproducible
from a frozen CSV instead of a live, still-growing database table.

Also prints the per-channel missing-value percentages of the RAW table —
needed verbatim for the revised preprocessing section (reviewer R4-5).

Usage:  python scripts/export_data.py
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
sys.path.insert(0, BACKEND_DIR)

import mysql.connector  # noqa: E402
from ml_engine import DB_CONFIG, TABLE_NAME, TARGETS, load_conveyor_data  # noqa: E402


def main() -> None:
    out_dir = os.path.join(REPO_ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. RAW missing-value statistics (before any imputation) ---
    print("[INFO] Reading raw table for missing-value statistics...")
    conn = mysql.connector.connect(**DB_CONFIG)
    raw = pd.read_sql(f"SELECT * FROM {TABLE_NAME} WHERE conveyor_id > 1079", conn)
    conn.close()
    print(f"[INFO] Raw rows: {len(raw)}")
    print("\nPer-channel missing-value rate (RAW table) — cite in preprocessing section:")
    for col in TARGETS:
        if col in raw.columns:
            vals = pd.to_numeric(raw[col], errors="coerce")
            pct = 100.0 * vals.isna().mean()
            print(f"  {col:12s}: {pct:6.2f}% missing")

    # --- 2. Cleaned/resampled export (same pipeline the paper describes) ---
    print("\n[INFO] Running the standard cleaning pipeline (load_conveyor_data)...")
    df = load_conveyor_data()
    if df is None or df.empty:
        raise RuntimeError("load_conveyor_data returned no data — check backend/.env")

    out_csv = os.path.join(out_dir, "conveyor_export.csv")
    df.reset_index().rename(columns={"index": "datetime"}).to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} ({len(df)} rows, {df.index[0]} .. {df.index[-1]})")


if __name__ == "__main__":
    main()
