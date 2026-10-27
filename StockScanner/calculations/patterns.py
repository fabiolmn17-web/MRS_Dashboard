"""
patterns.py — Actionability / chart pattern detection
======================================================
Detects:
1. ATR Compression  — tight base / low volatility consolidation
2. Local Swing High — most recent pivot high (N bars each side)
3. Near Pivot       — price within 5% below (or just above) the swing high
4. Squeeze          — Beardy Squeeze Pro levels (BB 2.0 inside KC 1.0 / 1.5 / 2.0) and momentum

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

        # ── Squeeze (Beardy Squeeze Pro): needs only 2 x 20 bars, so before the ATR length check
        result.update(self.compute_squeeze(df))

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

    # ── Squeeze (Beardy Squeeze Pro) ─────────────────────────────────────────
    SQZ_LENGTH = 20
    SQZ_BB_MULT = 2.0
    SQZ_KC = (1.0, 1.5, 2.0)          # High / Mid / Low compression (Keltner multipliers)

    def compute_squeeze(self, df: pd.DataFrame) -> dict:
        """Beardy Squeeze Pro (TradingView, Beardy_Fred), same definitions as the Pine script:
          BB  = SMA(close, 20) +/- 2.0 x stdev(close, 20)          (population stdev, as ta.stdev)
          KC  = SMA(close, 20) +/- k x SMA(true range, 20)        k = 1.0 / 1.5 / 2.0
          Both channels share the SMA basis, so "BB inside KC" reduces to 2 x stdev <= k x SMA(TR).
          Level: High (inside KC 1.0, orange dots), Mid (inside 1.5, red), Low (inside 2.0, black),
                 Off (outside 2.0, green = no squeeze). ('Off', not 'None': pandas reads 'None' as NaN.)
          Fired = no squeeze on the last bar after a squeeze on the bar before (Pine alert "Squeeze Fired").
          Momentum = linreg(close - avg(avg(highest high 20, lowest low 20), SMA close 20), 20, 0);
          colour as in the script: aqua (>0 rising), blue (>0 falling), red (<0 falling), yellow (<0 rising).
        """
        out = {'sqz_ratio': np.nan, 'sqz_level': '', 'bb_kc_squeeze': False, 'sqz_started': False,
               'sqz_fired': False, 'sqz_bars': 0, 'sqz_mom': np.nan, 'sqz_mom_color': ''}
        try:
            n = self.SQZ_LENGTH
            d = df[['High', 'Low', 'Close']].dropna()
            if len(d) < 2 * n + 2:
                return out
            c, h, l = d['Close'], d['High'], d['Low']
            basis = c.rolling(n).mean()
            dev = self.SQZ_BB_MULT * c.rolling(n).std(ddof=0)
            tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
            tr.iloc[0] = np.nan                       # ta.tr is na on the first bar
            dev_kc = tr.rolling(n).mean()
            ratio = dev / dev_kc                      # <= 1.0 High, <= 1.5 Mid, <= 2.0 Low
            hi_k, mid_k, low_k = self.SQZ_KC

            def level(x):
                if pd.isna(x):
                    return ''
                return 'High' if x <= hi_k else 'Mid' if x <= mid_k else 'Low' if x <= low_k else 'Off'

            on = ratio <= low_k                       # any compression = not green
            r_now, r_prev = ratio.iloc[-1], ratio.iloc[-2]
            on_now, on_prev = bool(on.iloc[-1]), bool(on.iloc[-2])
            bars = 0
            for v in on.iloc[::-1]:
                if not v:
                    break
                bars += 1
            src = c - ((h.rolling(n).max() + l.rolling(n).min()) / 2 + basis) / 2
            x = np.arange(n)

            def _lr(y):                              # ta.linreg(src, n, 0): fitted value at the last bar
                b, a = np.polyfit(x, y, 1)
                return a + b * (n - 1)

            mom = src.rolling(n).apply(_lr, raw=True)
            m_now, m_prev = mom.iloc[-1], mom.iloc[-2]
            m_prev = 0.0 if pd.isna(m_prev) else m_prev   # nz(mom[1])
            if pd.isna(m_now):
                col = ''
            elif m_now > 0:
                col = 'aqua' if m_now > m_prev else 'blue'
            else:
                col = 'red' if m_now < m_prev else 'yellow'
            out.update({'sqz_ratio': float(r_now) if not pd.isna(r_now) else np.nan,
                        'sqz_level': level(r_now), 'bb_kc_squeeze': on_now,
                        'sqz_started': on_now and not on_prev, 'sqz_fired': (not on_now) and on_prev,
                        'sqz_bars': int(bars), 'sqz_mom': float(m_now) if not pd.isna(m_now) else np.nan,
                        'sqz_mom_color': col})
        except Exception:
            pass
        return out
