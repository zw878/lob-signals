"""Unit tests for signal construction. Run: python -m unittest discover -s tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_signals import add_forward_return, add_signals  # noqa: E402


def sample_book(bid_1: float, ask_1: float, bid_sz_1: int, ask_sz_1: int) -> pd.DataFrame:
    row = {
        "symbol": "TEST",
        "date": "2026-03-02",
        "minute": 0,
        "bid_px_1": bid_1,
        "bid_sz_1": bid_sz_1,
        "ask_px_1": ask_1,
        "ask_sz_1": ask_sz_1,
    }
    for level in range(2, 6):
        row[f"bid_px_{level}"] = bid_1 - 0.01 * (level - 1)
        row[f"ask_px_{level}"] = ask_1 + 0.01 * (level - 1)
        row[f"bid_sz_{level}"] = 100
        row[f"ask_sz_{level}"] = 100
    return pd.DataFrame([row])


class SignalConstructionTest(unittest.TestCase):
    def test_mid_and_spread(self):
        out = add_signals(sample_book(99.99, 100.01, 300, 100))
        self.assertAlmostEqual(out.loc[0, "mid"], 100.0)
        self.assertAlmostEqual(out.loc[0, "spread_bps"], 2.0)

    def test_order_imbalance(self):
        out = add_signals(sample_book(99.99, 100.01, 300, 100))
        self.assertAlmostEqual(out.loc[0, "oib_l1"], 0.5)

    def test_microprice(self):
        out = add_signals(sample_book(99.99, 100.01, 300, 100))
        expected = (100.01 * 300 + 99.99 * 100) / 400
        self.assertAlmostEqual(out.loc[0, "microprice"], expected)
        self.assertAlmostEqual(out.loc[0, "microprice_dev_bps"], 0.5)

    def test_forward_return_is_within_symbol(self):
        df = pd.concat(
            [
                sample_book(99.99, 100.01, 100, 100),
                sample_book(100.99, 101.01, 100, 100),
            ],
            ignore_index=True,
        )
        out = add_forward_return(df, horizon=1)
        self.assertAlmostEqual(out.loc[0, "ret_next_bps"], 100.0, places=6)
        self.assertTrue(pd.isna(out.loc[1, "ret_next_bps"]))


if __name__ == "__main__":
    unittest.main()
