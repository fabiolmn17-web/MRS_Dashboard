"""
fundamentals.py — CAN SLIM fundamental calculations
=====================================================
Data source: ticker.info from yfinance.

This matches TradingView's request.financial() data — both pull from
Yahoo Finance's pre-computed fundamental database, not raw SEC filings.
This is far more reliable than parsing quarterly_income_stmt, which
returns empty or malformed data for a large fraction of stocks.

Key metrics (mirrors the Pine Script CAN SLIM Fundamentals indicator):
  C  - Current quarter EPS YoY growth  (earningsQuarterlyGrowth)
  A  - Annual EPS 3yr CAGR             (derived from trailingEps / earningsGrowth)
  S  - Quarterly revenue YoY growth    (revenueQuarterlyGrowth)
  Quality extras: ROE, profit margin, D/E, current ratio (all from info)

Thresholds (aligned with Pine indicator):
  EPS growth  PASS ≥ 25%   WATCH ≥ 15%
  Rev growth  PASS ≥ 20%   WATCH ≥ 10%
  (scanner strict gate: EPS ≥ 25% AND Rev ≥ 25%)
  (scanner relaxed gate: EPS ≥ 20% AND Rev ≥ 15%)
"""

from typing import Any, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FundamentalCalculator:
    """Compute CAN SLIM fundamental metrics from yfinance ticker.info."""

    # ── Strict thresholds ─────────────────────────────────────────────────────
    STRICT_EPS_MIN = 0.25   # 25% quarterly EPS YoY
    STRICT_REV_MIN = 0.25   # 25% quarterly revenue YoY

    # ── Relaxed thresholds ────────────────────────────────────────────────────
    RELAXED_EPS_MIN = 0.20  # 20% quarterly EPS YoY
    RELAXED_REV_MIN = 0.15  # 15% quarterly revenue YoY

    def compute(
        self,
        info: Dict[str, Any],
        quarterly_income=None,   # kept for API compatibility, no longer primary
    ) -> dict:
        """
        Compute fundamental metrics from ticker.info.

        Primary fields used (all from Yahoo Finance's processed database):
          earningsQuarterlyGrowth  — most-recent-quarter EPS YoY as decimal
          revenueQuarterlyGrowth   — most-recent-quarter revenue YoY as decimal
          returnOnEquity           — trailing ROE as decimal
          profitMargins            — trailing net profit margin as decimal
          debtToEquity             — trailing D/E (yfinance returns as %)
          currentRatio             — most recent current ratio
          trailingEps              — TTM EPS (positive = profitable TTM)
          earningsGrowth           — TTM earnings growth YoY (used for annual proxy)

        Falls back to quarterly_income_stmt for eps_prev_qtr_yoy only when
        earningsQuarterlyGrowth is available (prior-quarter acceleration check).

        Returns dict with all metrics + pass flags.
        """
        result = self._empty_result()

        # ── Primary: earningsQuarterlyGrowth (C in CAN SLIM) ─────────────────
        eqg = info.get('earningsQuarterlyGrowth')
        if eqg is not None:
            try:
                result['eps_qtr_yoy'] = float(eqg)
            except (TypeError, ValueError):
                pass

        # ── Primary: revenueQuarterlyGrowth (S in CAN SLIM) ──────────────────
        rqg = info.get('revenueQuarterlyGrowth')
        if rqg is not None:
            try:
                result['rev_qtr_yoy'] = float(rqg)
            except (TypeError, ValueError):
                pass

        # ── TTM profitability check ───────────────────────────────────────────
        trailing_eps = info.get('trailingEps')
        if trailing_eps is not None:
            try:
                result['ttm_positive'] = float(trailing_eps) > 0
            except (TypeError, ValueError):
                pass

        # ── Quality metrics (display + optional gate) ─────────────────────────
        for key, field in [
            ('roe',            'returnOnEquity'),
            ('profit_margin',  'profitMargins'),
            ('debt_to_equity', 'debtToEquity'),
            ('current_ratio',  'currentRatio'),
        ]:
            val = info.get(field)
            if val is not None:
                try:
                    result[key] = float(val)
                except (TypeError, ValueError):
                    pass

        # yfinance returns debtToEquity as a percentage (e.g. 45.6 = 0.456 ratio)
        # normalise to a ratio to match Pine's DEBT_TO_EQUITY field
        if result['debt_to_equity'] is not None:
            result['debt_to_equity'] = result['debt_to_equity'] / 100.0

        # ── Annual EPS proxy (A in CAN SLIM) ─────────────────────────────────
        # earningsGrowth = TTM earnings YoY, used as annual proxy when 3yr CAGR
        # is unavailable. Not a strict gate, display only.
        eg = info.get('earningsGrowth')
        if eg is not None:
            try:
                result['eps_annual_proxy'] = float(eg)
            except (TypeError, ValueError):
                pass

        # ── Prior-quarter EPS (for acceleration display) ──────────────────────
        # Attempt from quarterly_income_stmt as a secondary source.
        # Not used as a gate, only for the acceleration indicator.
        if quarterly_income is not None and not getattr(quarterly_income, 'empty', True):
            try:
                result['eps_prev_qtr_yoy'] = self._prev_qtr_yoy(info, quarterly_income)
            except Exception:
                pass

        # ── Data quality assessment ───────────────────────────────────────────
        result['data_quality'] = self._assess_quality(result)

        # ── Pass / fail gates ─────────────────────────────────────────────────
        result['passes_strict']  = self._check_strict(result)
        result['passes_relaxed'] = self._check_relaxed(result)

        return result

    def fetch_and_compute(self, ticker_symbol: str) -> dict:
        """Full round-trip: fetch from yfinance + compute. Used in batch scan."""
        try:
            import yfinance as yf
            t    = yf.Ticker(ticker_symbol)
            info = t.info or {}
            # Attempt quarterly income for prev-qtr acceleration (optional)
            income = None
            try:
                income = t.quarterly_income_stmt
            except Exception:
                pass
            return self.compute(info, income)
        except Exception as e:
            logger.warning(f'{ticker_symbol}: fundamental fetch error: {e}')
            return self._empty_result(f'fetch_error: {e}')

    # ── Private helpers ────────────────────────────────────────────────────────

    def _prev_qtr_yoy(self, info: dict, income) -> Optional[float]:
        """
        Compute prior-quarter (Q-1) EPS YoY growth from quarterly_income_stmt.
        Used only for the acceleration display metric, not as a gate.
        Returns None if data unavailable.
        """
        import numpy as np
        import pandas as pd

        eps_row = self._find_row(income, ['Basic EPS', 'Diluted EPS'])
        if eps_row is None:
            # Fallback: net income / shares
            ni_row = self._find_row(income, ['Net Income', 'Net Income Common Stockholders'])
            if ni_row is None or len(income.columns) < 6:
                return None
            shares = (info.get('sharesOutstanding') or
                      info.get('impliedSharesOutstanding'))
            if not shares or shares <= 0:
                return None
            eps_row = ni_row / shares

        if eps_row is None or len(eps_row.dropna()) < 6:
            return None

        vals = eps_row.dropna()
        # columns newest-first: [0]=Q0, [1]=Q1, [4]=Q0_ya, [5]=Q1_ya
        q1   = float(vals.iloc[1])
        q1ya = float(vals.iloc[5])
        if q1ya == 0 or np.isnan(q1ya):
            return None
        return (q1 - q1ya) / abs(q1ya)

    @staticmethod
    def _find_row(df, candidates):
        for name in candidates:
            if name in df.index:
                row = df.loc[name]
                if row.notna().any():
                    return row
        return None

    def _assess_quality(self, r: dict) -> str:
        has_eps = not self._is_nan(r.get('eps_qtr_yoy'))
        has_rev = not self._is_nan(r.get('rev_qtr_yoy'))
        has_roe = r.get('roe') is not None
        if has_eps and has_rev and has_roe:
            return 'full'
        if has_eps or has_rev:
            return 'partial'
        return 'minimal'

    def _check_strict(self, r: dict) -> bool:
        """Strict: 25% EPS + 25% Rev + TTM profitable."""
        if not r.get('ttm_positive', False):
            return False
        eps = r.get('eps_qtr_yoy')
        if self._is_nan(eps) or eps < self.STRICT_EPS_MIN:
            return False
        rev = r.get('rev_qtr_yoy')
        if self._is_nan(rev) or rev < self.STRICT_REV_MIN:
            return False
        return True

    def _check_relaxed(self, r: dict) -> bool:
        """Relaxed: 20% EPS + 15% Rev + TTM profitable."""
        if not r.get('ttm_positive', False):
            return False
        eps = r.get('eps_qtr_yoy')
        if self._is_nan(eps) or eps < self.RELAXED_EPS_MIN:
            return False
        rev = r.get('rev_qtr_yoy')
        if self._is_nan(rev) or rev < self.RELAXED_REV_MIN:
            return False
        return True

    @staticmethod
    def _is_nan(v) -> bool:
        if v is None:
            return True
        try:
            return np.isnan(float(v))
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _empty_result(fail_reason: str = None) -> dict:
        return {
            'eps_qtr_yoy':      None,
            'eps_prev_qtr_yoy': None,
            'eps_annual_proxy': None,
            'rev_qtr_yoy':      None,
            'roe':              None,
            'profit_margin':    None,
            'debt_to_equity':   None,
            'current_ratio':    None,
            'ttm_positive':     None,
            'passes_strict':    False,
            'passes_relaxed':   False,
            'data_quality':     'minimal',
            'notes':            [fail_reason] if fail_reason else [],
        }
