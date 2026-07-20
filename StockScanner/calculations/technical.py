"""
technical.py — Technical filter calculations
=============================================
Computes SMA50/200, ATH distance, volume filter.
"""

from typing import Optional
import numpy as np
import pandas as pd


class TechnicalCalculator:
    """Compute technical filter metrics from price history."""

    def compute(
        self,
        df: pd.DataFrame,
        min_avg_volume: int = 500_000,
        volume_lookback: int = 50,
    ) -> dict:
        """
        Compute technical indicators and pass/fail flags.

        Args:
            df: OHLCV DataFrame (DatetimeIndex, columns: Open/High/Low/Close/Volume)
            min_avg_volume: Minimum 50-day average volume to pass liquidity filter
            volume_lookback: Days for average volume calculation

        Returns dict with:
            price, sma50, sma200, price_above_sma50, sma50_above_sma200,
            ath, pct_from_ath, within_20pct_ath, avg_volume_50d, volume_ok,
            passes_technical, fail_reason
        """
        result = {
            'price': np.nan,
            'sma50': np.nan,
            'sma200': np.nan,
            'price_above_sma50': False,
            'sma50_above_sma200': False,
            'ath': np.nan,
            'high_52w': np.nan,
            'pct_from_ath': np.nan,
            'pct_from_52w_high': np.nan,
            'within_20pct_ath': False,
            'avg_volume_50d': np.nan,
            'volume_ok': False,
            'passes_technical': False,
            'fail_reason': None,
        }

        if df is None or df.empty or 'Close' not in df.columns:
            result['fail_reason'] = 'no_data'
            return result

        close = df['Close'].dropna()
        if len(close) < 50:
            result['fail_reason'] = 'insufficient_history'
            return result

        price = float(close.iloc[-1])
        result['price'] = price

        # SMAs
        sma50 = float(close.iloc[-50:].mean()) if len(close) >= 50 else np.nan
        sma200 = float(close.iloc[-200:].mean()) if len(close) >= 200 else np.nan
        result['sma50'] = sma50
        result['sma200'] = sma200

        # Trend checks — need at least 200 bars for golden cross
        if np.isnan(sma50):
            result['fail_reason'] = 'need_50d_history'
            return result

        price_above_sma50 = price > sma50
        result['price_above_sma50'] = price_above_sma50

        if not price_above_sma50:
            result['fail_reason'] = 'price_below_sma50'
            return result

        if not np.isnan(sma200):
            sma50_above_sma200 = sma50 > sma200
            result['sma50_above_sma200'] = sma50_above_sma200
            if not sma50_above_sma200:
                result['fail_reason'] = 'sma50_below_sma200'
                return result
        # If we don't have 200 bars, skip the golden cross check (early-stage stock)

        # ATH (all available history) + 52-week high
        ath      = float(close.max())
        high_52w = float(close.iloc[-252:].max()) if len(close) >= 252 else ath

        pct_from_ath      = (price / ath)      - 1.0   # ≤ 0, negative = below ATH
        pct_from_52w_high = (price / high_52w) - 1.0   # ≤ 0, negative = below 52W high

        result['ath']               = ath
        result['high_52w']          = high_52w
        result['pct_from_ath']      = pct_from_ath
        result['pct_from_52w_high'] = pct_from_52w_high
        within_20 = pct_from_ath >= -0.20  # within 20% of ATH
        result['within_20pct_ath'] = within_20

        if not within_20:
            result['fail_reason'] = f'too_far_from_ath_{pct_from_ath*100:.1f}pct'
            return result

        # Volume filter
        if 'Volume' in df.columns:
            vol = df['Volume'].dropna()
            avg_vol = float(vol.iloc[-volume_lookback:].mean()) if len(vol) >= volume_lookback else float(vol.mean())
            result['avg_volume_50d'] = avg_vol
            vol_ok = avg_vol >= min_avg_volume
            result['volume_ok'] = vol_ok
            if not vol_ok:
                result['fail_reason'] = f'low_volume_{avg_vol:,.0f}'
                return result

        result['passes_technical'] = True
        return result
