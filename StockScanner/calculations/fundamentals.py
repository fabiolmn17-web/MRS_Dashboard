"""
fundamentals.py — CAN SLIM fundamental calculations (Yahoo Finance)
===================================================================
Extracts EPS growth, revenue growth, and ROE from yfinance data.

Key metrics:
  C - Current quarter EPS YoY growth
  A - Prior quarter EPS YoY growth (for acceleration check)
  S - Revenue growth (proxy for Supply/Demand + quality)
  ROE - Return on equity (quality filter)

Data source: ticker.quarterly_income_stmt + ticker.info
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalCalculator:
    """Extract and score CAN SLIM fundamental metrics from yfinance."""

    # Strict CAN SLIM thresholds
    STRICT_EPS_MIN      = 0.25   # 25% YoY quarterly EPS growth
    STRICT_REV_MIN      = 0.25   # 25% YoY quarterly revenue growth
    STRICT_ROE_MIN      = 0.17   # 17% minimum ROE
    STRICT_REQUIRE_BOTH_QTRS = True  # Both Q0 and Q1 must pass

    # Relaxed thresholds
    RELAXED_EPS_MIN     = 0.20   # 20% YoY EPS growth
    RELAXED_REV_MIN     = 0.15   # 15% YoY revenue growth
    RELAXED_ROE_MIN     = 0.15   # 15% minimum ROE
    RELAXED_REQUIRE_BOTH_QTRS = False  # Only most recent quarter required

    def fetch_and_compute(self, ticker_symbol: str) -> dict:
        """
        Fetch fundamental data from yfinance and compute all metrics.

        This does the full round-trip: fetch + compute.
        For batch scanning, call this per stock.

        Returns dict with all fundamental metrics + pass flags.
        """
        try:
            import yfinance as yf
            t = yf.Ticker(ticker_symbol)
            info = t.info or {}
            income = None
            try:
                income = t.quarterly_income_stmt
            except Exception:
                pass

            return self.compute(info, income)
        except Exception as e:
            logger.warning(f'{ticker_symbol}: fundamental fetch error: {e}')
            return self._empty_result(f'fetch_error: {e}')

    def compute(
        self,
        info: Dict[str, Any],
        quarterly_income: Optional[pd.DataFrame],
    ) -> dict:
        """
        Compute fundamental metrics from already-fetched data.

        Args:
            info: ticker.info dict
            quarterly_income: ticker.quarterly_income_stmt DataFrame
                Columns = quarter end dates (newest first)
                Index = line items (e.g. 'Total Revenue', 'Net Income', 'Basic EPS')

        Returns dict with:
            eps_qtr_yoy:       float — most recent quarter EPS YoY growth
            eps_prev_qtr_yoy:  float — previous quarter EPS YoY growth
            rev_qtr_yoy:       float — most recent quarter revenue YoY growth
            roe:               float — from ticker.info
            ttm_positive:      bool  — TTM net income > 0
            passes_strict:     bool
            passes_relaxed:    bool
            data_quality:      str   — 'full', 'partial', 'minimal'
            notes:             list  — explanation of any failures
        """
        result = self._empty_result()

        # ── ROE ──────────────────────────────────────────────────────────────────
        roe = info.get('returnOnEquity')
        if roe is not None:
            try:
                result['roe'] = float(roe)
            except (TypeError, ValueError):
                pass

        # ── TTM profitability ─────────────────────────────────────────────────────
        # Use trailingEps as a proxy: positive trailing EPS = profitable TTM
        trailing_eps = info.get('trailingEps')
        if trailing_eps is not None:
            try:
                result['ttm_positive'] = float(trailing_eps) > 0
            except (TypeError, ValueError):
                pass

        # ── Quarterly EPS & Revenue ───────────────────────────────────────────────
        if quarterly_income is not None and not quarterly_income.empty:
            eps_data = self._extract_quarterly_eps(info, quarterly_income)
            rev_data = self._extract_quarterly_revenue(quarterly_income)

            result['eps_qtr_yoy']      = eps_data.get('q0_yoy')
            result['eps_prev_qtr_yoy'] = eps_data.get('q1_yoy')
            result['rev_qtr_yoy']      = rev_data.get('q0_yoy')
            result['data_quality']     = self._assess_quality(result)
        else:
            # Fallback: use trailing EPS vs forward EPS as rough proxy
            result['notes'].append('no_quarterly_stmt')
            result['data_quality'] = 'minimal'

        # ── Filter checks ─────────────────────────────────────────────────────────
        result['passes_strict']  = self._check_strict(result)
        result['passes_relaxed'] = self._check_relaxed(result)

        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_quarterly_eps(
        self,
        info: Dict,
        income: pd.DataFrame,
    ) -> dict:
        """
        Extract Q0 and Q1 YoY EPS growth from quarterly income statement.

        Strategy:
        1. Try 'Basic EPS' or 'Diluted EPS' rows if available
        2. Fall back to Net Income / shares outstanding
        """
        # Try explicit EPS rows first
        eps_row = self._find_row(income, ['Basic EPS', 'Diluted EPS'])
        if eps_row is not None and len(income.columns) >= 5:
            return self._yoy_growth_series(eps_row)

        # Fall back to Net Income / shares
        ni_row = self._find_row(income, ['Net Income', 'Net Income Common Stockholders'])
        if ni_row is None or len(income.columns) < 5:
            return {}

        shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
        if shares and shares > 0:
            eps_series = ni_row / shares
            return self._yoy_growth_series(eps_series)

        # Last resort: use net income itself (scaled for growth rate purposes)
        return self._yoy_growth_series(ni_row)

    def _extract_quarterly_revenue(self, income: pd.DataFrame) -> dict:
        """Extract Q0 YoY revenue growth."""
        rev_row = self._find_row(income, ['Total Revenue', 'Revenue', 'Net Sales'])
        if rev_row is None or len(income.columns) < 5:
            return {}
        return self._yoy_growth_series(rev_row)

    def _find_row(
        self,
        df: pd.DataFrame,
        candidates: List[str],
    ) -> Optional[pd.Series]:
        """Find the first matching row in a DataFrame by index name."""
        for name in candidates:
            if name in df.index:
                row = df.loc[name]
                if row.notna().any():
                    return row
        return None

    def _yoy_growth_series(self, series: pd.Series) -> dict:
        """
        Compute YoY growth for Q0 (most recent) and Q1 (prior).

        Quarterly income stmt columns are newest-first:
          col[0] = Q0 (most recent quarter)
          col[1] = Q1 (1 quarter ago)
          col[2] = Q2
          col[3] = Q3
          col[4] = Q4 (same quarter last year = YoY for Q0)
          col[5] = Q5 (same quarter last year = YoY for Q1)
        """
        vals = series.dropna()
        if len(vals) < 5:
            return {}

        cols = list(vals.index)
        q0, q4 = float(vals.iloc[0]), float(vals.iloc[4])
        q1_yoy = np.nan
        if len(vals) >= 6:
            q1, q5 = float(vals.iloc[1]), float(vals.iloc[5])
            if q5 != 0 and not np.isnan(q5):
                q1_yoy = (q1 - q5) / abs(q5)

        q0_yoy = np.nan
        if q4 != 0 and not np.isnan(q4):
            q0_yoy = (q0 - q4) / abs(q4)

        return {'q0_yoy': q0_yoy, 'q1_yoy': q1_yoy}

    def _assess_quality(self, result: dict) -> str:
        has_eps = result['eps_qtr_yoy'] is not None and not np.isnan(result['eps_qtr_yoy'] or np.nan)
        has_rev = result['rev_qtr_yoy'] is not None and not np.isnan(result['rev_qtr_yoy'] or np.nan)
        has_roe = result['roe'] is not None
        if has_eps and has_rev and has_roe:
            return 'full'
        if has_eps or has_rev:
            return 'partial'
        return 'minimal'

    def _check_strict(self, r: dict) -> bool:
        """Strict CAN SLIM: 25% EPS + 25% Rev + 17% ROE, both quarters pass."""
        eps = r.get('eps_qtr_yoy')
        prev_eps = r.get('eps_prev_qtr_yoy')
        rev = r.get('rev_qtr_yoy')
        roe = r.get('roe')
        ttm_ok = r.get('ttm_positive', False)

        if not ttm_ok:
            return False
        if self._is_nan(eps) or eps < self.STRICT_EPS_MIN:
            return False
        if self.STRICT_REQUIRE_BOTH_QTRS and (self._is_nan(prev_eps) or prev_eps < self.STRICT_EPS_MIN):
            return False
        if self._is_nan(rev) or rev < self.STRICT_REV_MIN:
            return False
        if roe is None or roe < self.STRICT_ROE_MIN:
            return False
        return True

    def _check_relaxed(self, r: dict) -> bool:
        """Relaxed: 20% EPS + 15% Rev + 15% ROE, only most recent quarter."""
        eps = r.get('eps_qtr_yoy')
        rev = r.get('rev_qtr_yoy')
        roe = r.get('roe')
        ttm_ok = r.get('ttm_positive', False)

        if not ttm_ok:
            return False
        if self._is_nan(eps) or eps < self.RELAXED_EPS_MIN:
            return False
        if self._is_nan(rev) or rev < self.RELAXED_REV_MIN:
            return False
        if roe is None or roe < self.RELAXED_ROE_MIN:
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
            'eps_qtr_yoy':      np.nan,
            'eps_prev_qtr_yoy': np.nan,
            'rev_qtr_yoy':      np.nan,
            'roe':              None,
            'ttm_positive':     None,
            'passes_strict':    False,
            'passes_relaxed':   False,
            'data_quality':     'minimal',
            'notes':            [fail_reason] if fail_reason else [],
        }
