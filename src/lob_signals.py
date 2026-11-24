"""Signal construction for limit-order-book (LOB) snapshots.

The module expects a tidy DataFrame with one row per timestamp and symbol,
and level-1..N bid/ask prices and sizes stored as ``bid_px_i`` / ``bid_sz_i``
and ``ask_px_i`` / ``ask_sz_i``.

All signals are computed without look-ahead: every value at time ``t`` uses
only information observed at or before ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LEVELS = 5


def _level_columns(levels: int, side: str, field: str) -> list[str]:
    return [f"{side}_{field}_{i}" for i in range(1, levels + 1)]


def add_signals(df: pd.DataFrame, levels: int = DEFAULT_LEVELS) -> pd.DataFrame:
    """Return a copy of ``df`` with standard microstructure signals.

    Added columns
    -------------
    mid                 midpoint, (bid_px_1 + ask_px_1) / 2
    spread_bps          quoted spread in basis points of the midpoint
    oib_l1              level-1 order imbalance in [-1, 1]
    depth_imb           order imbalance aggregated over ``levels``
    microprice          size-weighted midpoint (Stoikov, 2018)
    microprice_dev_bps  microprice deviation from mid, in basis points
    """
    out = df.copy()
    bid_px = _level_columns(levels, "bid", "px")
    ask_px = _level_columns(levels, "ask", "px")
    bid_sz = _level_columns(levels, "bid", "sz")
    ask_sz = _level_columns(levels, "ask", "sz")

    out["mid"] = (out["bid_px_1"] + out["ask_px_1"]) / 2.0
    out["spread_bps"] = (out["ask_px_1"] - out["bid_px_1"]) / out["mid"] * 1e4

    denom_l1 = out["bid_sz_1"] + out["ask_sz_1"]
    out["oib_l1"] = np.where(
        denom_l1 > 0, (out["bid_sz_1"] - out["ask_sz_1"]) / denom_l1, 0.0
    )

    total_bid = out[bid_sz].sum(axis=1)
    total_ask = out[ask_sz].sum(axis=1)
    denom_all = total_bid + total_ask
    out["depth_imb"] = np.where(
        denom_all > 0, (total_bid - total_ask) / denom_all, 0.0
    )

    out["microprice"] = np.where(
        denom_l1 > 0,
        (out["ask_px_1"] * out["bid_sz_1"] + out["bid_px_1"] * out["ask_sz_1"])
        / denom_l1,
        out["mid"],
    )
    out["microprice_dev_bps"] = (out["microprice"] - out["mid"]) / out["mid"] * 1e4
    return out


def add_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Attach the forward mid-price return over ``horizon`` minutes, in bps."""
    out = df.copy()
    if "mid" not in out.columns:
        out["mid"] = (out["bid_px_1"] + out["ask_px_1"]) / 2.0
    forward = out.groupby("symbol", sort=False)["mid"].shift(-horizon)
    out["ret_next_bps"] = (forward / out["mid"] - 1.0) * 1e4
    return out


def next_minute_return(df: pd.DataFrame) -> pd.Series:
    """Forward one-period mid-price return, grouped by symbol."""
    return df.groupby("symbol", sort=False)["mid"].shift(-1) / df["mid"] - 1.0


def predictive_correlation(df: pd.DataFrame, signal: str = "oib_l1") -> float:
    """Pearson correlation between a signal and the next-minute return."""
    y = next_minute_return(df)
    mask = y.notna() & df[signal].notna()
    return float(np.corrcoef(df.loc[mask, signal], y.loc[mask])[0, 1])
