"""Generate a synthetic limit-order book for pipeline testing.

The real study uses licensed level-2 data that cannot be redistributed. This
generator produces a *synthetic* book with realistic features -- persistent
order imbalance, a bid-ask spread, and a weak predictive link between
imbalance and the next-minute return -- so that the code can be run end to
end without proprietary data.

The synthetic results are illustrative only and must not be read as empirical
evidence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_PRICE = {"AMZN": 180.0, "GOOG": 190.0, "INTC": 45.0, "MSFT": 420.0}
LEVELS = 5


def make_demo_data(
    n_days: int = 20,
    n_minutes: int = 240,
    symbols: tuple[str, ...] = ("AMZN", "GOOG", "INTC", "MSFT"),
    seed: int = 7,
    impact: float = 1.5e-5,
    noise: float = 1.0e-4,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-03-02")
    records: list[dict] = []

    for symbol in symbols:
        price = REFERENCE_PRICE[symbol]
        for day in range(n_days):
            date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
            imbalance_prev = 0.0
            for minute in range(n_minutes):
                imbalance = 0.7 * imbalance_prev + rng.normal(0.0, 0.35)
                imbalance = float(np.clip(imbalance, -0.9, 0.9))

                # The price impact of an imbalance arrives one minute later, so
                # the imbalance observed at t is informative about the move from
                # t to t+1 -- this is what makes the timing signal useful.
                price *= 1.0 + impact * imbalance_prev + rng.normal(0.0, noise)
                half_spread = max(0.01, round(price * 1e-4, 2))
                bid_1 = round(price - half_spread, 2)
                ask_1 = round(price + half_spread, 2)
                if ask_1 <= bid_1:
                    ask_1 = round(bid_1 + 0.01, 2)

                base_size = int(rng.integers(200, 800))
                bid_size_1 = max(10, int(base_size * (1.0 + 0.8 * imbalance)))
                ask_size_1 = max(10, int(base_size * (1.0 - 0.8 * imbalance)))

                row: dict = {
                    "symbol": symbol,
                    "date": date,
                    "minute": minute,
                    "bid_px_1": bid_1,
                    "bid_sz_1": bid_size_1,
                    "ask_px_1": ask_1,
                    "ask_sz_1": ask_size_1,
                }
                for level in range(2, LEVELS + 1):
                    step = half_spread * 2 * (level - 1)
                    row[f"bid_px_{level}"] = round(bid_1 - step, 2)
                    row[f"ask_px_{level}"] = round(ask_1 + step, 2)
                    row[f"bid_sz_{level}"] = int(rng.integers(100, 1200))
                    row[f"ask_sz_{level}"] = int(rng.integers(100, 1200))
                records.append(row)
                imbalance_prev = imbalance

    return pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--minutes", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("data/demo_lob.csv"))
    args = parser.parse_args()

    df = make_demo_data(n_days=args.days, n_minutes=args.minutes, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
