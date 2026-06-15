"""
auto_fetch.py — Automatic data fetching for the three previously-manual inputs
===============================================================================
Replaces the daily MRS_Inputs_v4.xlsx entry with fully automated sources:

  Zero Gamma  →  InsiderFinance /gamma-exposure/SPX  (static HTML scrape)
  ADL Level   →  yfinance ^NYAD  (daily net advances, cumulated from last known)
  B20%        →  yfinance SPX500 constituents  (% above their own 20-day MA)

All three return NaN on failure — the pipeline scores them as neutral, which
is conservative and safe (neutral = 0 contribution, not a false signal).
"""

import re
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import date, timedelta

warnings.filterwarnings('ignore')

IF_URL = 'https://www.insiderfinance.io/gamma-exposure/SPX'

# ── Zero Gamma ─────────────────────────────────────────────────────────────────
def fetch_zero_gamma() -> float:
    """
    Scrape SPX zero-gamma level from InsiderFinance.
    The value appears in the static HTML article section as:
        **Zero-Gamma Level:** $7411.00
    No headless browser required.
    Returns the level as a float, or np.nan on failure.
    """
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            )
        }
        r = requests.get(IF_URL, headers=headers, timeout=20)
        r.raise_for_status()
        # Pattern matches: Zero-Gamma Level:** $7411.00  or  Zero Gamma\n\n$7411.00
        patterns = [
            r'Zero[- ]Gamma Level\*\*:\s*\$?([\d,]+(?:\.\d+)?)',   # article
            r'Zero Gamma\s*\n+\s*\$?([\d,]+(?:\.\d+)?)',           # header block
            r'"zeroGamma"\s*:\s*([\d.]+)',                          # JSON embed
            r'zero.gamma[^$\d]{0,30}\$\s*([\d,]+(?:\.\d+)?)',      # generic
        ]
        for pat in patterns:
            m = re.search(pat, r.text, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(',', ''))
                if 3000 < val < 15000:   # sanity check — SPX plausible range
                    print(f'  Zero Gamma fetched: {val}')
                    return val
        print('  [WARN] Zero Gamma: pattern not found in InsiderFinance page')
    except Exception as e:
        print(f'  [WARN] Zero Gamma fetch failed: {e}')
    return np.nan


# ── ADL Level ──────────────────────────────────────────────────────────────────
def fetch_adl(last_adl: float, last_adl_date: pd.Timestamp) -> dict:
    """
    Extend the existing NYSE ADL series using yfinance ^NYAD.
    ^NYAD = daily NYSE net advances (advances - declines).
    Cumulative sum starting from last_adl produces the ADL level.

    Returns a dict of  pd.Timestamp -> adl_level  for all NEW dates
    (i.e., after last_adl_date).
    """
    result = {}
    try:
        fetch_start = (last_adl_date - timedelta(days=5)).strftime('%Y-%m-%d')
        raw = yf.download('^NYAD', start=fetch_start, progress=False, auto_adjust=True)
        if raw.empty:
            print('  [WARN] ADL: ^NYAD returned no data')
            return result

        # Handle multi-index vs flat columns
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw['Close']['^NYAD']
        else:
            s = raw['Close']

        s.index = pd.to_datetime(s.index).normalize()
        s = s.dropna().sort_index()

        # Only keep dates AFTER last_adl_date
        s_new = s[s.index > last_adl_date]
        if s_new.empty:
            print('  ADL: no new dates beyond history — unchanged')
            return result

        # Cumulative ADL starting from last_adl
        running = last_adl
        for dt, net_adv in s_new.items():
            running += float(net_adv)
            result[dt] = round(running, 0)

        print(f'  ADL fetched: {len(result)} new date(s), latest={list(result.values())[-1]:,.0f}')
    except Exception as e:
        print(f'  [WARN] ADL fetch failed: {e}')
    return result


# ── B20% — % SPX500 stocks above their 20-day MA ──────────────────────────────
def fetch_b20() -> float:
    """
    Compute % of S&P 500 stocks closing above their own 20-day moving average.
    Equivalent to the CBOE S5TW index (the same source used in TradingView).

    Downloads the current S&P 500 list from Wikipedia, then downloads 45 days
    of close prices via yfinance (batch call, single request).

    Returns percentage as a float (e.g. 63.4), or np.nan on failure.
    """
    try:
        # Step 1: get current S&P 500 constituent list
        print('  B20: fetching S&P 500 constituent list...')
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            attrs={'id': 'constituents'}
        )
        tickers = tables[0]['Symbol'].tolist()
        # yfinance uses '-' not '.' for BRK.B, BF.B etc.
        tickers = [t.replace('.', '-') for t in tickers]

        # Step 2: batch download 45 days of close prices
        print(f'  B20: downloading {len(tickers)} tickers (45d)...')
        prices = yf.download(
            tickers,
            period='45d',
            auto_adjust=True,
            progress=False,
            threads=True
        )
        if isinstance(prices.columns, pd.MultiIndex):
            closes = prices['Close']
        else:
            closes = prices

        closes = closes.dropna(axis=1, how='all')   # drop tickers with no data

        if closes.shape[0] < 21:
            print('  [WARN] B20: insufficient price history')
            return np.nan

        # Step 3: compute % above 20-day MA on the latest available date
        ma20   = closes.rolling(20, min_periods=20).mean()
        latest = closes.iloc[-1]
        ma_lat = ma20.iloc[-1]
        valid  = ma_lat.notna() & latest.notna()
        above  = (latest[valid] > ma_lat[valid]).sum()
        pct    = round(float(above) / valid.sum() * 100, 2)
        print(f'  B20: {pct:.1f}% ({above}/{valid.sum()} stocks above 20-day MA)')
        return pct

    except Exception as e:
        print(f'  [WARN] B20 fetch failed: {e}')
        return np.nan


# ── Build inp_map for a single "today" row ─────────────────────────────────────
def build_inp_map(hist: pd.DataFrame) -> dict:
    """
    Build the inp_map dict consumed by pipeline.update_history().
    Auto-fetches all three previously-manual inputs for today's date.

    Only adds the TODAY entry — the pipeline's carry-forward logic handles
    all prior dates from the existing history.
    """
    today_ts = pd.Timestamp(date.today())
    print('\n=== Auto-fetching manual inputs ===')

    # ── Zero Gamma ─────────────────────────────────────────────────────────
    zg = fetch_zero_gamma()

    # ── ADL — extend from last known value ─────────────────────────────────
    adl_series = hist['adl_level'].dropna()
    if adl_series.empty:
        print('  [WARN] ADL: no historical ADL data — scoring as neutral')
        adl_today = np.nan
    else:
        last_adl       = float(adl_series.iloc[-1])
        last_adl_date  = hist.loc[adl_series.index[-1], 'date']
        adl_new        = fetch_adl(last_adl, pd.Timestamp(last_adl_date))
        # Get the most recent value (today or last trading day)
        if adl_new:
            adl_today = list(adl_new.values())[-1]
        else:
            adl_today = last_adl  # unchanged — carry forward existing value
            print(f'  ADL: using carry-forward value {adl_today:,.0f}')

    # ── B20% ────────────────────────────────────────────────────────────────
    b20_today = fetch_b20()

    inp_map = {
        today_ts: {
            'adl_level':  adl_today,
            'b20_pct':    b20_today,
            'zero_gamma': zg,
            'pc_ratio':   np.nan,   # always auto-fetched from CBOE
            'skew':       np.nan,   # always auto-fetched from yfinance
        }
    }

    print(f'\n  Inputs for {today_ts.date()}:')
    print(f'    ADL        = {adl_today:,.0f}' if not np.isnan(adl_today) else '    ADL        = NaN (neutral)')
    print(f'    B20%       = {b20_today:.1f}%' if not np.isnan(b20_today) else '    B20%       = NaN (neutral)')
    print(f'    Zero Gamma = {zg:.2f}' if not np.isnan(zg) else '    Zero Gamma = NaN (neutral)')

    return inp_map
