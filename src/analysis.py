"""Empirical tests: do order-book signals predict short-horizon returns?

Three pieces, all standard in empirical asset pricing:

1. ``panel_regression`` -- next-period return on a signal, with symbol fixed
   effects and standard errors clustered by date.
2. ``decile_table``     -- average next-period return by signal decile.
3. ``long_short``       -- top-minus-bottom decile spread, by date, with a
   t-statistic and an annualised Sharpe ratio.

All returns are expressed in basis points. Every test uses the *next* period's
return, so there is no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """OLS with one-way cluster-robust covariance."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)

    meat = np.zeros((X.shape[1], X.shape[1]))
    for group in np.unique(groups):
        mask = groups == group
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)

    cov = xtx_inv @ meat @ xtx_inv
    return beta, cov


def panel_regression(
    df: pd.DataFrame,
    y_col: str = "ret_next_bps",
    x_col: str = "oib_l1",
    cluster: str = "date",
) -> dict:
    """Regress the next-period return on a signal with symbol fixed effects."""
    sub = df.dropna(subset=[y_col, x_col])
    y = sub[y_col].to_numpy(dtype=float)
    x = sub[x_col].to_numpy(dtype=float)
    dummies = pd.get_dummies(sub["symbol"], drop_first=True).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(sub)), x, dummies])

    beta, cov = _ols_cluster(y, X, sub[cluster].to_numpy())
    se = np.sqrt(np.diag(cov))
    return {
        "coef_bps": float(beta[1]),
        "se_bps": float(se[1]),
        "t_stat": float(beta[1] / se[1]),
        "n_obs": int(len(sub)),
        "n_clusters": int(sub[cluster].nunique()),
    }


def decile_table(
    df: pd.DataFrame,
    signal: str = "oib_l1",
    y_col: str = "ret_next_bps",
    q: int = 10,
) -> pd.DataFrame:
    """Average next-period return by signal decile (decile 10 = highest)."""
    sub = df.dropna(subset=[y_col, signal]).copy()
    sub["bucket"] = pd.qcut(sub[signal], q, labels=False, duplicates="drop") + 1
    table = sub.groupby("bucket")[y_col].agg(["mean", "count"])
    table.columns = ["mean_bps", "n_obs"]
    return table


def long_short(
    df: pd.DataFrame,
    signal: str = "oib_l1",
    y_col: str = "ret_next_bps",
    q: int = 10,
) -> tuple[pd.Series, dict]:
    """Top-minus-bottom decile spread by date.

    Returns the per-date spread series and a summary dictionary. All figures are
    gross: they ignore the bid-ask spread and any market impact, and are not an
    investable return series.
    """
    sub = df.dropna(subset=[y_col, signal]).copy()
    sub["bucket"] = pd.qcut(sub[signal], q, labels=False, duplicates="drop") + 1
    by_date = sub.groupby(["date", "bucket"])[y_col].mean().unstack()
    if by_date.shape[1] < q:
        raise ValueError("not enough deciles to form a long-short spread")

    spread = (by_date[q] - by_date[1]).dropna()
    n = len(spread)
    mean = float(spread.mean())
    std = float(spread.std(ddof=1))
    t_stat = mean / (std / np.sqrt(n)) if n > 1 and std > 0 else float("nan")

    summary = {
        "n_dates": n,
        "mean_bps": mean,
        "std_bps": std,
        "t_stat": t_stat,
        "win_rate": float((spread > 0).mean()),
    }
    return spread, summary
