"""
scanner.py — CAN SLIM Stock Scanner
=====================================
Orchestrates a full S&P 500 scan:

Pipeline:
  1. Fetch S&P 500 universe (Wikipedia)
  2. Batch download 1Y price history for all tickers + SPY (single yf.download call)
  3. Fast pass per stock: technicals + RS (no per-stock API calls)
  4. Slow pass for candidates only: fetch fundamentals (per-stock yfinance call)
  5. Compute actionability (patterns) for all fundamentally-passing stocks
  6. Save results to output/scan_results.csv

Runtime: ~3-5 min for S&P 500 (dominated by fundamental fetch step)
"""

import logging
import time
import warnings
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from .calculations import (
    TechnicalCalculator,
    RelativeStrengthCalculator,
    FundamentalCalculator,
    PatternDetector,
)
from .data.universe import UniverseManager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path(__file__).parent / 'output'
RESULTS_CSV = OUTPUT_DIR / 'scan_results.csv'

# Price history period (calendar days) — covers 252+ trading days for 1Y RS
PRICE_DAYS    = 420   # ~14 months calendar
BATCH_SIZE    = 100   # tickers per yf.download call
RATE_DELAY    = 0.4   # seconds between fundamental fetch calls


def run_scan(
    universe_source: str = 'sp500',
    mrs_score: Optional[float] = None,
    mrs_state: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full CAN SLIM scan.

    Args:
        universe_source: 'sp500', 'sp500_nasdaq100', 'custom', etc.
        mrs_score:  Current MRS score to stamp on results (context only, not a gate)
        mrs_state:  Current MRS state string (e.g. 'RISK-OFF')
        verbose:    Print progress

    Returns:
        DataFrame with all scanned stocks (including rejects with reason column).
        Also saves to RESULTS_CSV.
    """
    t_start = time.time()
    log = print if verbose else logger.info

    log('═' * 60)
    log('CAN SLIM Scanner — Starting')
    log(f'  Universe: {universe_source}')
    log(f'  MRS context: {mrs_score} / {mrs_state}')
    log('═' * 60)

    # ── Step 1: Universe ──────────────────────────────────────────────────────
    log('\n[1/5] Loading S&P 500 universe...')
    um = UniverseManager()
    tickers = um.load_universe(source=universe_source)
    log(f'  {len(tickers)} tickers loaded')

    if not tickers:
        log('  ERROR: No tickers loaded — aborting')
        return pd.DataFrame()

    # ── Step 2: Batch price download ──────────────────────────────────────────
    log('\n[2/5] Downloading price history...')
    all_tickers = tickers + ['SPY']
    price_data  = _batch_download(all_tickers, PRICE_DAYS, BATCH_SIZE, log)

    spy_df = price_data.get('SPY')
    if spy_df is None or spy_df.empty:
        log('  ERROR: SPY data missing — aborting')
        return pd.DataFrame()
    log(f'  SPY: {len(spy_df)} bars ({spy_df.index[-1].date()} last)')
    log(f'  Price data fetched for {len(price_data)-1}/{len(tickers)} stocks')

    # ── Step 3: Fast pass — technicals + RS ──────────────────────────────────
    log('\n[3/5] Technical + RS fast filter...')
    tech_calc = TechnicalCalculator()
    rs_calc   = RelativeStrengthCalculator()

    tech_pass    = []
    all_tech_rows = []

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        tech = tech_calc.compute(df)
        rs   = rs_calc.compute(df, spy_df)

        row = {
            'ticker':            ticker,
            'price':             tech['price'],
            'sma50':             tech['sma50'],
            'sma200':            tech['sma200'],
            'high_52w':          tech['high_52w'],
            'pct_from_ath':      tech['pct_from_ath'],
            'pct_from_52w_high': tech['pct_from_52w_high'],
            'avg_volume_50d':    tech['avg_volume_50d'],
            'rs_1y':             rs['rs_1y'],
            'rs_6m':             rs['rs_6m'],
            'rs_3m':             rs['rs_3m'],
            'rs_composite':      rs['rs_composite'],
            'rs_label':          rs['rs_label'],
            'passes_technical':  tech['passes_technical'],
            'passes_rs':         rs['rs_ok'],
            'tech_fail_reason':  tech['fail_reason'],
        }
        all_tech_rows.append(row)

        if tech['passes_technical'] and rs['rs_ok']:
            tech_pass.append(ticker)

    log(f'  {len(tech_pass)}/{len(all_tech_rows)} pass tech + RS filter')

    if not tech_pass:
        log('  No candidates after tech filter — saving empty results')
        return _save_empty(mrs_score, mrs_state)

    # ── Step 4: Fundamental fetch (candidates only) ───────────────────────────
    log(f'\n[4/5] Fetching fundamentals for {len(tech_pass)} candidates...')
    fund_calc    = FundamentalCalculator()
    fund_results = {}

    import yfinance as yf
    for i, ticker in enumerate(tech_pass):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            income = None
            try:
                income = t.quarterly_income_stmt
            except Exception:
                pass
            fund_results[ticker] = fund_calc.compute(info, income)
            # Attach name/sector from info
            fund_results[ticker]['name']     = info.get('longName') or info.get('shortName', ticker)
            fund_results[ticker]['sector']   = info.get('sector', '')
            fund_results[ticker]['industry'] = info.get('industry', '')
        except Exception as e:
            fund_results[ticker] = FundamentalCalculator._empty_result(f'error:{e}')
            fund_results[ticker]['name']     = ticker
            fund_results[ticker]['sector']   = ''
            fund_results[ticker]['industry'] = ''

        if verbose and (i + 1) % 20 == 0:
            log(f'  ... {i+1}/{len(tech_pass)} fundamentals fetched')
        time.sleep(RATE_DELAY)

    # ── Step 5: Patterns (actionability) ─────────────────────────────────────
    log('\n[5/5] Computing actionability (ATR compression + pivot)...')
    pattern_calc = PatternDetector()

    all_rows = []
    for row in all_tech_rows:
        ticker = row['ticker']
        fund   = fund_results.get(ticker, {})
        pat    = {}

        if row['passes_technical'] and row['passes_rs']:
            df = price_data.get(ticker)
            if df is not None:
                pat = pattern_calc.compute(df)

        # Determine overall pass mode
        passes_strict  = fund.get('passes_strict',  False)
        passes_relaxed = fund.get('passes_relaxed', False)
        if passes_strict:
            pass_mode = 'STRICT'
        elif passes_relaxed:
            pass_mode = 'RELAXED'
        else:
            pass_mode = None

        all_rows.append({
            # Identity
            'ticker':             ticker,
            'name':               fund.get('name', ticker),
            'sector':             fund.get('sector', ''),
            'industry':           fund.get('industry', ''),
            # Price
            'close':              row['price'],
            'sma50':              row['sma50'],
            'sma200':             row['sma200'],
            'high_52w':           row.get('high_52w'),
            'pct_from_ath':       row['pct_from_ath'],
            'pct_from_52w_high':  row.get('pct_from_52w_high'),
            'avg_volume_50d':     row['avg_volume_50d'],
            # RS
            'rs_1y':              row['rs_1y'],
            'rs_6m':              row['rs_6m'],
            'rs_3m':              row['rs_3m'],
            'rs_composite':       row['rs_composite'],
            'rs_label':           row['rs_label'],
            # Fundamentals
            'eps_qtr_yoy':        fund.get('eps_qtr_yoy'),
            'eps_prev_qtr_yoy':   fund.get('eps_prev_qtr_yoy'),
            'rev_qtr_yoy':        fund.get('rev_qtr_yoy'),
            'roe':                fund.get('roe'),
            'ttm_positive':       fund.get('ttm_positive'),
            'data_quality':       fund.get('data_quality', ''),
            # Patterns
            'atr_pct':            pat.get('atr_pct'),
            'atr_pct_rank':       pat.get('atr_pct_rank'),
            'atr_compressed':     pat.get('atr_compressed', False),
            'bb_kc_squeeze':      pat.get('bb_kc_squeeze', False),
            # Filters
            'passes_technical':   row['passes_technical'],
            'passes_rs':          row['passes_rs'],
            'passes_strict':      passes_strict,
            'passes_relaxed':     passes_relaxed,
            'pass_mode':          pass_mode,
            'tech_fail_reason':   row['tech_fail_reason'],
            # MRS context
            'mrs_score':          mrs_score,
            'mrs_state':          mrs_state,
            'scan_date':          date.today().isoformat(),
        })

    results_df = pd.DataFrame(all_rows)

    # Sort: CAN SLIM candidates first, then by RS composite desc
    sort_mask = results_df['pass_mode'].notna()
    passing   = results_df[sort_mask].sort_values('rs_composite', ascending=False)
    rejected  = results_df[~sort_mask]
    final_df  = pd.concat([passing, rejected], ignore_index=True)

    # Summary
    n_strict  = (final_df['pass_mode'] == 'STRICT').sum()
    n_relaxed = (final_df['pass_mode'] == 'RELAXED').sum()
    elapsed   = time.time() - t_start
    log(f'\n  ✓ Strict candidates:  {n_strict}')
    log(f'  ✓ Relaxed candidates: {n_relaxed}')
    log(f'  Total scanned: {len(final_df)} | Elapsed: {elapsed:.0f}s')

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(RESULTS_CSV, index=False)
    log(f'\n  Results saved → {RESULTS_CSV}')
    log('═' * 60)

    return final_df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _batch_download(
    tickers: List[str],
    days: int,
    batch_size: int,
    log,
) -> Dict[str, pd.DataFrame]:
    """
    Batch-download price history using yf.download.
    Splits into batches to avoid timeout / memory issues.
    """
    import yfinance as yf
    from datetime import timedelta

    end   = datetime.now()
    start = end - timedelta(days=int(days * 1.5))
    start_str = start.strftime('%Y-%m-%d')
    end_str   = end.strftime('%Y-%m-%d')

    results = {}
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        log(f'  Batch {batch_num}/{n_batches}: {len(batch)} tickers...')

        try:
            raw = yf.download(
                batch,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by='ticker',
            )

            if raw.empty:
                log(f'  Batch {batch_num}: empty result')
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    elif isinstance(raw.columns, pd.MultiIndex):
                        if ticker in raw.columns.get_level_values(0):
                            df = raw[ticker].copy()
                        else:
                            continue
                    else:
                        df = raw.copy()

                    if df.empty or 'Close' not in df.columns:
                        continue

                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                    df = df[cols].dropna(subset=['Close'])

                    if not df.empty:
                        results[ticker] = df

                except Exception as e:
                    logger.debug(f'  {ticker}: extraction error: {e}')

        except Exception as e:
            log(f'  Batch {batch_num} download error: {e}')
            # Fall back to individual downloads for failed batch
            for ticker in batch:
                try:
                    import yfinance as yf
                    t = yf.Ticker(ticker)
                    df = t.history(start=start_str, end=end_str, auto_adjust=True)
                    if df.empty:
                        continue
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                    df = df[cols].dropna(subset=['Close'])
                    if not df.empty:
                        results[ticker] = df
                except Exception:
                    pass
            time.sleep(2)

    return results


def _save_empty(mrs_score, mrs_state) -> pd.DataFrame:
    df = pd.DataFrame(columns=['ticker', 'pass_mode', 'mrs_score', 'mrs_state', 'scan_date'])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    return df


def load_results() -> Optional[pd.DataFrame]:
    """Load the most recent scan results CSV. Returns None if not found."""
    if not RESULTS_CSV.exists():
        return None
    try:
        df = pd.read_csv(RESULTS_CSV)
        return df
    except Exception as e:
        logger.warning(f'Failed to load scan results: {e}')
        return None
