"""
patterns.py — Actionability / chart pattern detection
======================================================
Detects:
1. ATR Compression  — tight base / low volatility consolidation
2. Local Swing High — most recent pivot high (N bars each side)
3. Near Pivot       — price within 5% below (or just above) the swing high

These are NOT filters — they annotate every passing stock to help the user
identify which stocks are in a low-risk entry zone.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd


class PatternDetector:
    """Detect actionable chart patterns from price history."""

    # ATR compression parameters
    ATR_PERIOD        = 14     # Standard ATR period
    ATR_LOOKBACK      = 50     # Compare current ATR vs last N bars
    ATR_COMPRESSED_PCTL = 0.35 # Below 35th percentile = compressed

    # Swing high parameters
    SWING_BARS        = 10     # Bars on each side to qualify as local max
    SWING_LOOKBACK    = 60     # Only look back this many bars for swing highs

    # Near-pivot zone (relative to swing high)
    PIVOT_BELOW_MAX   = -0.05  # Must be within 5% below
    PIVOT_ABOVE_MAX   =  0.03  # Allow up to 3% above (early breakout still valid)

    def compute(self, df: pd.DataFrame) -> dict:
        """
        Compute actionability metrics for a stock.

        Args:
            df: OHLCV DataFrame, DatetimeIndex, sorted oldest-to-newest

        Returns dict with:
            atr_current:      float — current 14-day ATR
            atr_pct:          float — ATR as % of price (normalized)
            atr_pct_rank:     float — percentile 0-1 vs last 50 bars (lower = tighter)
            atr_compressed:   bool  — True if in bottom 35th percentile
            swing_high:       float or None — most recent local pivot high
            swing_high_idx:   int   — bars ago the swing high occurred
            near_pivot:       bool  — price within pivot zone
            pivot_gap_pct:    float — (price / swing_high - 1) × 100
        """
        result = {
            'atr_current':    np.nan,
            'atr_pct':        np.nan,
            'atr_pct_rank':   np.nan,
            'atr_compressed': False,
            'swing_high':     None,
            'swing_high_idx': None,
            'near_pivot':     False,
            'pivot_gap_pct':  np.nan,
        }

        if df is None or df.empty:
            return result

        required = {'High', 'Low', 'Close'}
        if not required.issubset(df.columns):
            return result

        close = df['Close'].dropna()
        high  = df['High'].dropna()
        low   = df['Low'].dropna()

        if len(close) < max(self.ATR_PERIOD + self.ATR_LOOKBACK, self.SWING_BARS * 2 + 1):
            return result

        price = float(close.iloc[-1])

        # ── ATR Compression ───────────────────────────────────────────────────────
        atr_series = self._compute_atr(high, low, close, self.ATR_PERIOD)
        if atr_series is not None and len(atr_series) >= self.ATR_LOOKBACK:
            current_atr  = float(atr_series.iloc[-1])
            atr_pct      = current_atr / price  # normalized
            window       = atr_series.iloc[-self.ATR_LOOKBACK:]
            atr_pct_series = window / close.iloc[-self.ATR_LOOKBACK:]
            pct_rank     = float((atr_pct_series < atr_pct).mean())  # 0=tightest, 1=widest

            result['atr_current']    = current_atr
            result['atr_pct']        = atr_pct
            result['atr_pct_rank']   = pct_rank
            result['atr_compressed'] = pct_rank <= self.ATR_COMPRESSED_PCTL

        # ── Local Swing High ─────────────────────────────────────────────────────
        swing_high, swing_idx = self._find_swing_high(high, close)
        if swing_high is not None:
            pivot_gap = (price / swing_high) - 1.0
            near = self.PIVOT_BELOW_MAX <= pivot_gap <= self.PIVOT_ABOVE_MAX

            result['swing_high']     = swing_high
            result['swing_high_idx'] = swing_idx
            result['near_pivot']     = near
            result['pivot_gap_pct']  = pivot_gap * 100.0

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> Optional[pd.Series]:
        """
        True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        ATR = rolling mean of True Range over `period` bars.
        """
        try:
            h = high.values
            l = low.values
            c = close.values

            tr = np.maximum(
                h[1:] - l[1:],
                np.maximum(
                    np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1]),
                )
            )
            tr_series = pd.Series(tr, index=close.index[1:])
            atr = tr_series.rolling(window=period, min_periods=period).mean()
            return atr.dropna()
        except Exception:
            return None

    def _find_swing_high(
        self,
        high: pd.Series,
        close: pd.Series,
    ) -> Tuple[Optional[float], Optional[int]]:
        """
        Find the most recent local pivot high within the last SWING_LOOKBACK bars.

        A bar i is a local max if high[i] is the highest in the window
        [i - SWING_BARS, i + SWING_BARS].

        We exclude the rightmost SWING_BARS bars (can't confirm pivot without
        right-side bars yet).
        """
        n = self.SWING_BARS
        lookback = self.SWING_LOOKBACK

        h = high.values
        total = len(h)

        if total < n * 2 + 1:
            return None, None

        # Only look back `lookback` bars, excluding the last n bars
        search_start = max(0, total - lookback)
        search_end   = total - n  # need n bars on the right to confirm

        best_val   = -np.inf
        best_idx   = None  # index in h[] array

        for i in range(search_start, search_end):
            window = h[max(0, i-n) : i+n+1]
            if h[i] == np.max(window):
                if h[i] > best_val:
                    best_val = h[i]
                    best_idx = i

        if best_idx is None:
            return None, None

        bars_ago = total - 1 - best_idx
        return float(best_val), bars_ago
