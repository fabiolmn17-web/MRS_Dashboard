"""
scanner.py — CAN SLIM Stock Scanner
=====================================
Orchestrates the nightly scan (3 Nov 2026: US liquid universe, ~1,500-2,000 names, see data/universe.py):

Pipeline:
  1. Universe: US common stocks / ADRs, market cap >= $1B, price >= $10 (saved list as fallback; S&P 500 last);
     then 50-day average dollar volume >= $20M from the price download
  2. Batch download 1Y price history for all tickers + SPY (single yf.download call)
  3. Fast pass per stock: technicals + RS; "L" = RS rating >= 80 (IBD-style percentile across the universe)
  4. Slow pass for candidates only: fetch fundamentals (per-stock yfinance call)
  5. Compute actionability (patterns) for all fundamentally-passing stocks
  6. Save results to output/scan_results.csv

Momentum & trend scan (added 26 Oct 2026, momentum.py): every scanned stock also gets the held RS
label, trend (close > SMA200, SMA50 > SMA200), Ext-Mom extension / momentum zones and 5-day pullback,
and `mom_list` = held RS score >= +1.0 and both trend conditions. Momentum-list names get name /
sector / industry and the next earnings date even when they are not CAN SLIM candidates. Everything
goes into the same scan_results.csv (the GitHub workflow commits only that file).

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
from .calculations.relative_strength import weighted_performance, rs_ratings
from .data.universe import UniverseManager
from . import momentum
from . import bases

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path(__file__).parent / 'output'
RESULTS_CSV = OUTPUT_DIR / 'scan_results.csv'

# Price history period (calendar days) — covers 252+ trading days for 1Y RS
PRICE_DAYS    = 500   # x1.5 in _batch_download = ~750 calendar days (~2 years): cups up to 65 weeks + their prior uptrend
BATCH_SIZE    = 100   # tickers per yf.download call
RATE_DELAY    = 0.4   # seconds between fundamental fetch calls
MIN_DOLLAR_VOLUME = 20e6   # universe: 50-day average dollar volume (close x volume), 3 Nov 2026
RS_MIN_RATING     = 80     # "L": RS rating (percentile 1-99 across the scanned universe) >= 80, 3 Nov 2026


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
    log('\n[1/5] Loading the universe...')
    um = UniverseManager()
    tickers = um.load_universe(source=universe_source)
    universe_note = getattr(um, 'source_note', '') or universe_source
    us_meta = getattr(um, 'us_meta', {}) or {}
    log(f'  {len(tickers)} tickers loaded ({universe_note})')

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

    # Liquidity: drop names whose 50-day average dollar volume is under MIN_DOLLAR_VOLUME (3 Nov 2026)
    dollar_vol = {}
    for t_ in tickers:
        df_ = price_data.get(t_)
        if df_ is None or df_.empty or 'Volume' not in df_.columns:
            continue
        dv_ = (df_['Close'] * df_['Volume']).dropna().iloc[-50:]
        dollar_vol[t_] = float(dv_.mean()) if len(dv_) else np.nan
    n_before = len(tickers)
    tickers = [t_ for t_ in tickers if dollar_vol.get(t_, 0) >= MIN_DOLLAR_VOLUME]
    log(f'  {len(tickers)} pass the liquidity floor (50-day dollar volume >= ${MIN_DOLLAR_VOLUME / 1e6:.0f}M; '
        f'{n_before - len(tickers)} dropped)')

    # ── Step 3: Fast pass — technicals + RS ──────────────────────────────────
    log('\n[3/5] Technical + RS fast filter...')
    tech_calc = TechnicalCalculator()
    rs_calc   = RelativeStrengthCalculator()

    tech_pass    = []
    all_tech_rows = []
    mom_by       = {}
    perf_by      = {}

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        tech = tech_calc.compute(df)
        rs   = rs_calc.compute(df, spy_df)
        mom_by[ticker] = momentum.momentum_fields(df, spy_df)
        perf_by[ticker] = weighted_performance(df['Close'])

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
            'rs_1m':             rs.get('rs_1m'),
            'rs_composite':      rs['rs_composite'],
            'rs_label':          rs['rs_label'],
            'passes_technical':  tech['passes_technical'],
            'passes_rs':         False,                 # set below from the RS rating
            'tech_fail_reason':  tech['fail_reason'],
            'dollar_volume_50d': dollar_vol.get(ticker),
        }
        all_tech_rows.append(row)

    # "L": RS rating across the whole scanned universe (until 3 Nov 2026: RS composite > 0 = beats SPY in 2 of 3 periods)
    ratings = rs_ratings(perf_by)
    for row in all_tech_rows:
        row['rs_rating'] = ratings.get(row['ticker'], np.nan)
        row['passes_rs'] = bool(np.isfinite(row['rs_rating']) and row['rs_rating'] >= RS_MIN_RATING)
        if row['passes_technical'] and row['passes_rs']:
            tech_pass.append(row['ticker'])

    log(f'  {sum(r["passes_technical"] for r in all_tech_rows)} pass the technical filter; '
        f'{len(tech_pass)}/{len(all_tech_rows)} also have RS rating >= {RS_MIN_RATING}')

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

    # ── Step 4b: Momentum list — name / sector / industry and next earnings date ─
    mom_names = [t for t, m in mom_by.items() if m.get('mom_list')]
    log(f'\n[4b] Momentum & trend list: {len(mom_names)} names (held RS >= +1, above SMA200, SMA50 > SMA200)')
    meta = {}
    gics = momentum.sp500_meta()          # sector / industry for every stock (one request)
    log(f'  sector metadata for {len(gics)} S&P 500 names')
    # also the stocks held by the industry ETFs (top 10 each): the dashboard filters on "in a leader industry"
    try:
        _hold = pd.read_csv(OUTPUT_DIR / 'industry_holdings.csv')
        held = set(_hold['ticker'].astype(str)) & set(t for t in mom_by)
    except Exception:
        held = set()
    for i, ticker in enumerate(sorted(set(mom_names) | set(tech_pass) | held)):
        m = {'next_earnings': momentum.next_earnings(ticker)}
        if ticker not in fund_results and ticker not in gics:
            um_ = us_meta.get(ticker, {})
            try:
                info = yf.Ticker(ticker).info or {}
                m.update({'name': info.get('longName') or info.get('shortName') or um_.get('name', ticker),
                          'sector': info.get('sector') or um_.get('sector', ''),
                          'industry': info.get('industry') or um_.get('industry', '')})
            except Exception:
                m.update({'name': um_.get('name', ticker), 'sector': um_.get('sector', ''),
                          'industry': um_.get('industry', '')})
        meta[ticker] = m
        time.sleep(RATE_DELAY)
        if verbose and (i + 1) % 25 == 0:
            log(f'  ... {i+1} names done')

    # ── Step 5: Patterns (actionability) ─────────────────────────────────────
    log('\n[5/5] Actionability (ATR compression, squeeze, 21-day / 52-week breakouts, bases) for every scanned stock...')
    pattern_calc = PatternDetector()

    all_rows = []
    for row in all_tech_rows:
        ticker = row['ticker']
        fund   = fund_results.get(ticker, {})
        pat    = {}

        mom = mom_by.get(ticker, {})
        mt  = meta.get(ticker, {})
        # every scanned stock (cheap, price data only): the squeeze and ATR filters work on any list
        df = price_data.get(ticker)
        bse = {}
        if df is not None:
            pat = pattern_calc.compute(df)
            try:                                  # 21-day / 52-week breakouts and bases (bases.py, 3 Nov 2026)
                bse = bases.detect(df, row.get('rs_rating'))
            except Exception as e:
                logger.debug(f'{ticker}: bases error {e}')

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
            'name':               (fund.get('name') or mt.get('name') or gics.get(ticker, {}).get('name')
                                   or us_meta.get(ticker, {}).get('name', ticker)),
            'sector':             (fund.get('sector') or mt.get('sector') or gics.get(ticker, {}).get('sector')
                                   or us_meta.get(ticker, {}).get('sector', '')),
            'industry':           (fund.get('industry') or mt.get('industry') or gics.get(ticker, {}).get('industry')
                                   or us_meta.get(ticker, {}).get('industry', '')),
            # Price
            'close':              row['price'],
            'sma50':              row['sma50'],
            'sma200':             row['sma200'],
            'high_52w':           row.get('high_52w'),
            'pct_from_ath':       row['pct_from_ath'],
            'pct_from_52w_high':  row.get('pct_from_52w_high'),
            'avg_volume_50d':     row['avg_volume_50d'],
            'dollar_volume_50d':  row.get('dollar_volume_50d'),
            # RS
            'rs_1y':              row['rs_1y'],
            'rs_6m':              row['rs_6m'],
            'rs_3m':              row['rs_3m'],
            'rs_1m':              row.get('rs_1m'),
            'rs_composite':       row['rs_composite'],
            'rs_label':           row['rs_label'],
            'rs_rating':          row.get('rs_rating'),
            # Fundamentals
            'eps_qtr_yoy':        fund.get('eps_qtr_yoy'),
            'eps_prev_qtr_yoy':   fund.get('eps_prev_qtr_yoy'),
            'eps_annual_proxy':   fund.get('eps_annual_proxy'),
            'eps_source':         fund.get('eps_source'),        # 'stmt'|'yahoo'
            'eps_accelerating':   fund.get('eps_accelerating'),  # True|False|None
            'eps_yago':           fund.get('eps_yago'),          # EPS of the same quarter a year ago
            'eps_low_base':       fund.get('eps_low_base'),      # True = growth from <= 0 or > 300%: not counted
            'rev_qtr_yoy':        fund.get('rev_qtr_yoy'),
            'rev_source':         fund.get('rev_source'),        # 'stmt'|'yahoo'
            'roe':                fund.get('roe'),
            'profit_margin':      fund.get('profit_margin'),
            'debt_to_equity':     fund.get('debt_to_equity'),
            'current_ratio':      fund.get('current_ratio'),
            'ttm_positive':       fund.get('ttm_positive'),
            'data_quality':       fund.get('data_quality', ''),
            # Patterns
            'atr_pct':            pat.get('atr_pct'),
            'atr_pct_rank':       pat.get('atr_pct_rank'),
            'atr_compressed':     pat.get('atr_compressed', False),
            'bb_kc_squeeze':      pat.get('bb_kc_squeeze', False),   # any squeeze (Beardy: not green)
            'sqz_level':          pat.get('sqz_level', ''),          # High / Mid / Low / Off
            'sqz_ratio':          pat.get('sqz_ratio'),              # 2 x stdev / SMA(TR); <=1 / 1.5 / 2
            'sqz_bars':           pat.get('sqz_bars', 0),            # consecutive bars in a squeeze
            'sqz_started':        pat.get('sqz_started', False),
            'sqz_fired':          pat.get('sqz_fired', False),
            'sqz_mom':            pat.get('sqz_mom'),
            'sqz_mom_color':      pat.get('sqz_mom_color', ''),      # aqua / blue / red / yellow
            # Filters
            'passes_technical':   row['passes_technical'],
            'passes_rs':          row['passes_rs'],
            'passes_strict':      passes_strict,
            'passes_relaxed':     passes_relaxed,
            'pass_mode':          pass_mode,
            'tech_fail_reason':   row['tech_fail_reason'],
            # Breakouts and bases (bases.py)
            **{k: bse.get(k) for k in bases.FIELDS},
            # Momentum & trend (momentum.py)
            **{k: mom.get(k) for k in momentum.FIELDS},
            'next_earnings':      mt.get('next_earnings', ''),
            # MRS context
            'mrs_score':          mrs_score,
            'mrs_state':          mrs_state,
            'scan_date':          date.today().isoformat(),
            'universe':           universe_note,
        })

    results_df = pd.DataFrame(all_rows)

    # Sort: CAN SLIM candidates first, then by RS rating desc
    sort_mask = results_df['pass_mode'].notna()
    passing   = results_df[sort_mask].sort_values('rs_rating', ascending=False)
    rejected  = results_df[~sort_mask]
    final_df  = pd.concat([passing, rejected], ignore_index=True)

    # Summary
    n_strict  = (final_df['pass_mode'] == 'STRICT').sum()
    n_relaxed = (final_df['pass_mode'] == 'RELAXED').sum()
    elapsed   = time.time() - t_start
    log(f'\n  ✓ Strict candidates:  {n_strict}')
    log(f'  ✓ Relaxed candidates: {n_relaxed}')
    if 'base_type' in final_df:
        log(f"  Bases: {final_df['base_type'].value_counts().to_dict()} · "
            f"states {final_df['base_state'].value_counts().to_dict()}")
        log(f"  Breakouts today: 21-day high {int(final_df['bo21'].fillna(False).astype(bool).sum())}, "
            f"52-week high {int(final_df['bo52'].fillna(False).astype(bool).sum())}")
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

            # ── Staleness check ───────────────────────────────────────────────
            # yf.download() can silently return stale/cached data for a SUBSET
            # of tickers in a batch without raising an exception — the same
            # failure mode found in the sector table. Detect tickers whose last
            # bar lags the batch's own most-recent date and re-fetch them
            # individually via yf.Ticker().history() (same fix pattern used
            # elsewhere in this codebase for cloud-infra yfinance issues).
            if results:
                batch_tickers = [t for t in batch if t in results]
                if batch_tickers:
                    max_date = max(results[t].index.max() for t in batch_tickers)
                    stale = [t for t in batch_tickers
                             if results[t].index.max() < max_date - pd.Timedelta(days=2)]
                    if stale:
                        log(f'  Batch {batch_num}: {len(stale)} stale ticker(s), re-fetching individually...')
                        for ticker in stale:
                            try:
                                t = yf.Ticker(ticker)
                                df = t.history(start=start_str, end=end_str, auto_adjust=True)
                                if df.empty:
                                    continue
                                df.index = pd.to_datetime(df.index).tz_localize(None)
                                cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                                df = df[cols].dropna(subset=['Close'])
                                if not df.empty and df.index.max() > results[ticker].index.max():
                                    results[ticker] = df
                            except Exception as e:
                                logger.debug(f'  {ticker}: stale re-fetch error: {e}')

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
