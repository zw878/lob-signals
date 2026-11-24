"""End-to-end demo: build signals, run predictive tests, print results.

Usage
-----
    python scripts/run_demo.py            # uses data/demo_lob.csv if present
    python scripts/run_demo.py --rebuild  # regenerates the synthetic sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import decile_table, long_short, panel_regression  # noqa: E402
from lob_signals import (  # noqa: E402
    add_forward_return,
    add_signals,
    predictive_correlation,
)

from make_demo_data import make_demo_data  # noqa: E402

DATA_PATH = ROOT / "data" / "demo_lob.csv"
RESULTS_DIR = ROOT / "results"


def load_data(rebuild: bool, days: int) -> pd.DataFrame:
    if rebuild or not DATA_PATH.exists():
        df = make_demo_data(n_days=days)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"[data] generated synthetic sample: {len(df):,} rows -> {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"[data] loaded {len(df):,} rows from {DATA_PATH}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="regenerate data")
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=1, help="forward horizon, minutes")
    args = parser.parse_args()

    df = add_signals(load_data(args.rebuild, args.days))
    df = add_forward_return(df, horizon=args.horizon)

    print("\n[predictability] correlation with next-minute mid return")
    for signal in ("oib_l1", "depth_imb", "microprice_dev_bps"):
        print(f"  {signal:<20s} {predictive_correlation(df, signal): .4f}")

    reg = panel_regression(df)
    print("\n[regression] ret_next_bps ~ oib_l1, symbol FE, SE clustered by date")
    print(f"  coef = {reg['coef_bps']: .3f} bps per unit imbalance, "
          f"t = {reg['t_stat']: .2f}, N = {reg['n_obs']:,}, "
          f"clusters = {reg['n_clusters']}")

    table = decile_table(df)
    print("\n[decile sort] mean next-minute return (bps) by OIB decile")
    print(table.to_string(float_format=lambda v: f"{v:,.2f}"))

    spread, summary = long_short(df)
    print("\n[long-short] top minus bottom decile, per date")
    for key, value in summary.items():
        print(f"  {key:<18s} {value:,.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "decile_table.csv")
    pd.DataFrame([reg]).to_csv(RESULTS_DIR / "regression.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "long_short.csv", index=False)
    spread.rename("spread_bps").to_csv(RESULTS_DIR / "spread_by_date.csv")
    print(f"\n[saved] {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
