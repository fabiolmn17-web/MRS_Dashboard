"""
fundamentals.py — CAN SLIM fundamental calculations
=====================================================
3-tier data cascade (most reliable → least reliable):

  Tier 1 — ticker.info pre-computed fields
            earningsQuarterlyGrowth, revenueQuarterlyGrowth
            (same Yahoo Finance database as TradingView request.financial)

  Tier 2 — quarterly_income_stmt parsing
            Basic EPS / Net Income rows, Revenue rows
            (raw financial statements — broad coverage, sometimes stale)

  Tier 3 — proxy / estimation
            earningsGrowth (TTM proxy for annual check)
            trailingEps sign for TTM profitability

Missing-data policy:
  - If both EPS tiers fail → eps_qtr_yoy = None → stock fails on EPS gate
  - If TTM profitability unknown → do NOT gate on it (avoid false rejections)
  - Revenue gate applied only when data is available

Thresholds aligned with Pine Script CAN SLIM Fundamentals indicator:
  EPS growth   STRICT ≥ 25%   RELAXED ≥ 20%
  Rev growth   STRICT ≥ 25%   RELAXED ≥ 15%
"""

from typing import Any, Dict, List, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FundamentalCalculator:

    STRICT_EPS_MIN  = 0.25
    STRICT_REV_MIN  = 0.25
    RELAXED_EPS_MIN = 0.20
    RELAXED_REV_MIN = 0.15

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(
        self,
        info: Dict[str, Any],
        quarterly_income=None,
    ) -> dict:
        result = self._empty_result()

        # ── TTM profitability (Tier 1: trailingEps) ───────────────────────────
        # When unknown, we do NOT gate on it (missing ≠ unprofitable).
        trailing_eps = info.get('trailingEps')
        if trailing_eps is not None:
            try:
                result['ttm_positive'] = float(trailing_eps) > 0
            except (TypeError, ValueError):
                pass
        # Fallback: positive trailing PE implies positive earnings
        if result['ttm_positive'] is None:
            pe = info.get('trailingPE')
            if pe is not None:
                try:
                    result['ttm_positive'] = float(pe) > 0
                except (TypeError, ValueError):
                    pass
        # Fallback: netIncomeToCommon
        if result['ttm_positive'] is None:
            ni = info.get('netIncomeToCommon')
            if ni is not None:
                try:
                    result['ttm_positive'] = float(ni) > 0
                except (TypeError, ValueError):
                    pass

        # ── EPS growth: Tier 1 — ticker.info ─────────────────────────────────
        eqg = info.get('earningsQuarterlyGrowth')
        if eqg is not None:
            try:
                result['eps_qtr_yoy'] = float(eqg)
            except (TypeError, ValueError):
                pass

        # ── EPS growth: Tier 2 — quarterly_income_stmt ───────────────────────
        if result['eps_qtr_yoy'] is None and quarterly_income is not None:
            try:
                if not getattr(quarterly_income, 'empty', True):
                    eps_data = self._extract_quarterly_eps(info, quarterly_income)
                    result['eps_qtr_yoy']      = eps_data.get('q0_yoy')
                    result['eps_prev_qtr_yoy'] = eps_data.get('q1_yoy')
            except Exception:
                pass

        # ── EPS acceleration (prev quarter, Tier 2 only when Tier 1 used) ────
        if result['eps_prev_qtr_yoy'] is None and quarterly_income is not None:
            try:
                if not getattr(quarterly_income, 'empty', True):
                    eps_data = self._extract_quarterly_eps(info, quarterly_income)
                    result['eps_prev_qtr_yoy'] = eps_data.get('q1_yoy')
            except Exception:
                pass

        # ── Revenue growth: Tier 1 — ticker.info ─────────────────────────────
        rqg = info.get('revenueQuarterlyGrowth')
        if rqg is not None:
            try:
                result['rev_qtr_yoy'] = float(rqg)
            except (TypeError, ValueError):
                pass

        # ── Revenue growth: Tier 2 — quarterly_income_stmt ───────────────────
        if result['rev_qtr_yoy'] is None and quarterly_income is not None:
            try:
                if not getattr(quarterly_income, 'empty', True):
                    rev_data = self._extract_quarterly_revenue(quarterly_income)
                    result['rev_qtr_yoy'] = rev_data.get('q0_yoy')
            except Exception:
                pass

        # ── Quality metrics (Tier 1 only — all in info) ───────────────────────
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

        # yfinance returns debtToEquity as percentage (45.6 = 0.456)
        if result['debt_to_equity'] is not None:
            result['debt_to_equity'] = result['debt_to_equity'] / 100.0

        # ── Annual EPS proxy (Tier 1) ─────────────────────────────────────────
        eg = info.get('earningsGrowth')
        if eg is not None:
            try:
                result['eps_annual_proxy'] = float(eg)
            except (TypeError, ValueError):
                pass

        # ── Data quality + pass flags ─────────────────────────────────────────
        result['data_quality']   = self._assess_quality(result)
        result['passes_strict']  = self._check_strict(result)
        result['passes_relaxed'] = self._check_relaxed(result)

        return result

    def fetch_and_compute(self, ticker_symbol: str) -> dict:
        try:
            import yfinance as yf
            t      = yf.Ticker(ticker_symbol)
            info   = t.info or {}
            income = None
            try:
                income = t.quarterly_income_stmt
            except Exception:
                pass
            return self.compute(info, income)
        except Exception as e:
            logger.warning(f'{ticker_symbol}: fundamental fetch error: {e}')
            return self._empty_result(f'fetch_error: {e}')

    # ── Income statement parsers (Tier 2) ────────────────────────────────────

    def _extract_quarterly_eps(self, info: dict, income: pd.DataFrame) -> dict:
        eps_row = self._find_row(income, ['Basic EPS', 'Diluted EPS'])
        if eps_row is not None and len(income.columns) >= 5:
            return self._yoy_growth(eps_row)
        # Fallback: net income / shares
        ni_row = self._find_row(income, ['Net Income', 'Net Income Common Stockholders'])
        if ni_row is None or len(income.columns) < 5:
            return {}
        shares = (info.get('sharesOutstanding') or
                  info.get('impliedSharesOutstanding'))
        if shares and shares > 0:
            return self._yoy_growth(ni_row / shares)
        return self._yoy_growth(ni_row)

    def _extract_quarterly_revenue(self, income: pd.DataFrame) -> dict:
        rev_row = self._find_row(income, ['Total Revenue', 'Revenue', 'Net Sales'])
        if rev_row is None or len(income.columns) < 5:
            return {}
        return self._yoy_growth(rev_row)

    @staticmethod
    def _find_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        for name in candidates:
            if name in df.index:
                row = df.loc[name]
                if row.notna().any():
                    return row
        return None

    @staticmethod
    def _yoy_growth(series: pd.Series) -> dict:
        """
        Compute YoY growth for Q0 and Q1.
        Quarterly stmt columns are newest-first:
          col[0]=Q0, col[1]=Q1, col[4]=Q0_ya, col[5]=Q1_ya
        """
        vals = series.dropna()
        if len(vals) < 5:
            return {}
        q0, q4 = float(vals.iloc[0]), float(vals.iloc[4])
        q0_yoy = (q0 - q4) / abs(q4) if q4 != 0 and not np.isnan(q4) else np.nan
        q1_yoy = np.nan
        if len(vals) >= 6:
            q1, q5 = float(vals.iloc[1]), float(vals.iloc[5])
            if q5 != 0 and not np.isnan(q5):
                q1_yoy = (q1 - q5) / abs(q5)
        return {'q0_yoy': q0_yoy if not np.isnan(q0_yoy) else None,
                'q1_yoy': q1_yoy if not np.isnan(q1_yoy) else None}

    # ── Scoring ───────────────────────────────────────────────────────────────

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
        """
        STRICT: EPS ≥ 25% AND Rev ≥ 25%.
        TTM gate: only applied when ttm_positive is explicitly False.
        Missing TTM data → do not reject.
        """
        if r.get('ttm_positive') is False:   # explicitly unprofitable
            return False
        eps = r.get('eps_qtr_yoy')
        if self._is_nan(eps) or eps < self.STRICT_EPS_MIN:
            return False
        rev = r.get('rev_qtr_yoy')
        if self._is_nan(rev) or rev < self.STRICT_REV_MIN:
            return False
        return True

    def _check_relaxed(self, r: dict) -> bool:
        """
        RELAXED: EPS ≥ 20% AND Rev ≥ 15%.
        TTM gate: only applied when ttm_positive is explicitly False.
        """
        if r.get('ttm_positive') is False:   # explicitly unprofitable
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
