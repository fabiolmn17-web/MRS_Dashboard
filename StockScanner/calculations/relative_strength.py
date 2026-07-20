"""
relative_strength.py — Relative strength vs SPY
=================================================
Matches the EXACT formula used in the sector RS table in app.py.

RS at period n = (stock_now / stock_n_ago) - (SPY_now / SPY_n_ago)
  = stock_return(n) - SPY_return(n)   [as multipliers, not percentage points]

Composite = sc(rs_1y) + sc(rs_6m) + sc(rs_3m)
  where sc(v) = +0.5 if v > 0, -0.5 if v < 0, 0.0 if v == 0 or NaN
  Range: -1.5 to +1.5 in 0.5 steps
"""

from typing import Optional
import numpy as np
import pandas as pd


_PERIOD_BARS = {
    '1y': 252,
    '6m': 126,
    '3m': 63,
}


def _sc(v: float) -> float:
    """Sign-based composite scorer — same as sector table."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return 0.5 if v > 0 else (-0.5 if v < 0 else 0.0)


def _composite_label(score: float) -> str:
    """Same labels as sector table _sector_composite_label()."""
    if score >= 1.5:  return 'STRUCTURAL LEADER'
    if score >= 1.0:  return 'STRONG OUTPERFORMER'
    if score >= 0.5:  return 'OUTPERFORMING'
    if score <= -1.5: return 'STRUCTURAL LAGGARD'
    if score <= -1.0: return 'STRONG UNDERPERFORMER'
    if score <= -0.5: return 'UNDERPERFORMING'
    return 'NEUTRAL'


class RelativeStrengthCalculator:
    """Compute RS vs SPY for a single stock."""

    def compute(
        self,
        stock_df: pd.DataFrame,
        spy_df: pd.DataFrame,
    ) -> dict:
        """
        Compute RS metrics using sector table formula.

        Args:
            stock_df: Stock price history (OHLCV), DatetimeIndex
            spy_df:   SPY price history (OHLCV), DatetimeIndex

        Returns dict with:
            rs_1y, rs_6m, rs_3m (raw differentials)
            rs_composite (-1.5 to +1.5)
            rs_label (str)
            rs_ok (bool, composite > 0)
        """
        result = {
            'rs_1y': np.nan,
            'rs_6m': np.nan,
            'rs_3m': np.nan,
            'rs_composite': 0.0,
            'rs_label': 'NEUTRAL',
            'rs_ok': False,
        }

        if stock_df is None or spy_df is None:
            return result

        stock_close = stock_df['Close'].dropna()
        spy_close   = spy_df['Close'].dropna()

        if stock_close.empty or spy_close.empty:
            return result

        stock_now = float(stock_close.iloc[-1])
        spy_now   = float(spy_close.iloc[-1])

        def rs(n: int) -> Optional[float]:
            """RS vs SPY at n bars — matches sector table rs() exactly."""
            if len(stock_close) <= n or len(spy_close) <= n:
                return np.nan
            stock_ago = float(stock_close.iloc[-n])
            spy_ago   = float(spy_close.iloc[-n])
            if stock_ago == 0 or spy_ago == 0:
                return np.nan
            return (stock_now / stock_ago) - (spy_now / spy_ago)

        rs_1y = rs(_PERIOD_BARS['1y'])
        rs_6m = rs(_PERIOD_BARS['6m'])
        rs_3m = rs(_PERIOD_BARS['3m'])

        composite = _sc(rs_1y) + _sc(rs_6m) + _sc(rs_3m)

        result['rs_1y']       = rs_1y
        result['rs_6m']       = rs_6m
        result['rs_3m']       = rs_3m
        result['rs_composite'] = composite
        result['rs_label']    = _composite_label(composite)
        result['rs_ok']       = composite > 0.0  # outperforming in at least 2 of 3 periods

        return result
