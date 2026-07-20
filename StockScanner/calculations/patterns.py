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

from typing import Optional
import numpy as np
import pandas as pd


class PatternDetector:
    """Detect actionable chart patterns from price history."""

    # ATR compression parameters
    ATR_PERIOD          = 14     # Standard ATR period
    ATR_LOOKBACK        = 50     # Compare current ATR vs last N bars
    ATR_COMPRESSED_PCTL = 0.35   # Below 35th percentile = compressed

    def compute(self, df: pd.DataFrame) -> dict:
        """
        Compute ATR compression from price history.

        Args:
            df: OHLCV DataFrame, DatetimeIndex, sorted oldest-to-newest

        Returns dict with:
            atr_current:      float — current 14-day ATR
            atr_pct:          float — ATR as % of price (normalized)
            atr_pct_rank:     float — percentile 0-1 vs last 50 bars (lower = tighter)
            atr_compressed:   bool  — True if in bottom 35th percentile
        """
        result = {
            'atr_current':    np.nan,
            'atr_pct':        np.nan,
            'atr_pct_rank':   np.nan,
            'atr_compressed': False,
        }

        if df is None or df.empty:
            return result

        required = {'High', 'Low', 'Close'}
        if not required.issubset(df.columns):
            return result

        close = df['Close'].dropna()
        high  = df['High'].dropna()
        low   = df['Low'].dropna()

        if len(close) < self.ATR_PERIOD + self.ATR_LOOKBACK:
            return result

        price = float(close.iloc[-1])

        # ── ATR Compression (percentile rank) ────────────────────────────────────
        atr_series = self._compute_atr(high, low, close, self.ATR_PERIOD)
        if atr_series is not None and len(atr_series) >= self.ATR_LOOKBACK:
            current_atr    = float(atr_series.iloc[-1])
            atr_pct        = current_atr / price
            window_atr     = atr_series.iloc[-self.ATR_LOOKBACK:]
            window_close   = close.iloc[-self.ATR_LOOKBACK:]
            atr_pct_series = window_atr.values / window_close.values
            pct_rank       = float((atr_pct_series < atr_pct).mean())

            result['atr_current']    = current_atr
            result['atr_pct']        = atr_pct
            result['atr_pct_rank']   = pct_rank
            result['atr_compressed'] = pct_rank <= self.ATR_COMPRESSED_PCTL

            # ── BB inside KC (Keltner Squeeze) ───────────────────────────────────
            # Bollinger Bands: 20-period SMA ± 2σ
            # Keltner Channel: 20-period EMA ± 1 × ATR(14)
            # Squeeze = BB upper < KC upper  AND  BB lower > KC lower
            # (BB is entirely inside KC → volatility squeeze, potential breakout)
            result['bb_kc_squeeze'] = self._compute_bb_kc_squeeze(close, atr_series)

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

    def _compute_bb_kc_squeeze(
        self,
        close: pd.Series,
        atr_series: pd.Series,
        bb_period: int = 20,
    ) -> bool:
        """
        Detect Keltner Squeeze: Bollinger Bands (±2σ) inside Keltner Channel (±1 ATR).

        When BB is entirely inside KC, volatility is compressed and a directional
        breakout is likely. Uses ±1 ATR (tighter than standard 1.5 ATR) per user spec.

        Returns True if squeeze is active at the last bar.
        """
        try:
            if len(close) < bb_period + 1:
                return False

            # Align ATR to close index
            atr_aligned = atr_series.reindex(close.index).ffill()

            # Bollinger Bands
            bb_sma   = close.rolling(bb_period).mean()
            bb_std   = close.rolling(bb_period).std(ddof=1)
            bb_upper = bb_sma + 2.0 * bb_std
            bb_lower = bb_sma - 2.0 * bb_std

            # Keltner Channel (EMA centre, ±1 ATR)
            kc_ema   = close.ewm(span=bb_period, adjust=False).mean()
            kc_upper = kc_ema + atr_aligned
            kc_lower = kc_ema - atr_aligned

            # Squeeze at last bar
            squeeze = (
                float(bb_upper.iloc[-1]) < float(kc_upper.iloc[-1])
                and float(bb_lower.iloc[-1]) > float(kc_lower.iloc[-1])
            )
            return bool(squeeze)
        except Exception:
            return False
