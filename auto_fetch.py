"""
auto_fetch.py — Automatic data fetching for the three previously-manual inputs
===============================================================================
Replaces the daily MRS_Inputs_v4.xlsx entry with fully automated sources:
  Zero Gamma  →  InsiderFinance /gamma-exposure/SPX  (static HTML scrape)
  ADL Level   →  yfinance ^NYAD  (daily net advances, cumulated from last known)
  B20%        →  TradingView S5TW via tvdatafeed
  PC Ratio    →  TradingView PC/USI via tvdatafeed
All return NaN on failure — the pipeline scores them as neutral, which
is conservative and safe (neutral = 0 contribution, not a false signal).
"""
import re
import warnings
import numpy as np
import os
import pandas as pd
import yfinance as yf
import requests
from datetime import date, timedelta
warnings.filterwarnings('ignore')

# ── Cloud-safe browser session (bypasses 403 on Yahoo Finance / CBOE) ──────────
_BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://finance.yahoo.com/',
}
_CLOUD_SESSION = requests.Session()
_CLOUD_SESSION.headers.update(_BROWSER_HEADERS)

IF_URL = 'https://www.insiderfinance.io/gamma-exposure/SPX'

# ── Zero Gamma ─────────────────────────────────────────────────────────────────
def fetch_zero_gamma() -> float:
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
        patterns = [
            r'Zero[- ]Gamma Level\*\*:\s*\$?([\d,]+(?:\.\d+)?)',
            r'Zero Gamma\s*\n+\s*\$?([\d,]+(?:\.\d+)?)',
            r'"zeroGamma"\s*:\s*([\d.]+)',
            r'zero.gamma[^$\d]{0,30}\$\s*([\d,]+(?:\.\d+)?)',
        ]
        for pat in patterns:
            m = re.search(pat, r.text, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(',', ''))
                if 3000 < val < 15000:
                    print(f'  Zero Gamma fetched: {val}')
                    return val
        print('  [WARN] Zero Gamma: pattern not found in InsiderFinance page')
    except Exception as e:
        print(f'  [WARN] Zero Gamma fetch failed: {e}')
    return np.nan

# ── ADL Level ──────────────────────────────────────────────────────────────────
def fetch_adl(last_adl: float, last_adl_date: pd.Timestamp) -> dict:
    _ADL_TICKERS = ['^NYAD', '^ADD', 'NYAD', 'ADD']
    result = {}
    try:
        fetch_start = (last_adl_date - timedelta(days=5)).strftime('%Y-%m-%d')
        raw = pd.DataFrame()
        used_ticker = None
        for adl_ticker in _ADL_TICKERS:
            try:
                raw = yf.download(
                    adl_ticker, start=fetch_start, progress=False,
                    auto_adjust=True, session=_CLOUD_SESSION
                )
                if not raw.empty:
                    used_ticker = adl_ticker
                    print(f'  ADL: using ticker {adl_ticker}')
                    break
            except Exception:
                continue
        if raw.empty:
            print('  [WARN] ADL: all tickers returned no data')
            return result
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw['Close'][used_ticker]
        else:
            s = raw['Close']
        s.index = pd.to_datetime(s.index).normalize()
        s = s.dropna().sort_index()
        s_new = s[s.index > last_adl_date]
        if s_new.empty:
            print('  ADL: no new dates beyond history — unchanged')
            return result
        running = last_adl
        for dt, net_adv in s_new.items():
            running += float(net_adv)
            result[dt] = round(running, 0)
        print(f'  ADL fetched: {len(result)} new date(s), latest={list(result.values())[-1]:,.0f}')
    except Exception as e:
        print(f'  [WARN] ADL fetch failed: {e}')
    return result

# ── B20% — S5TW from TradingView ──────────────────────────────────────────────
def fetch_b20(tv_user: str = '', tv_pass: str = '') -> float:
    try:
        from tvdatafeed import TvDatafeed, Interval
        print('  B20: fetching S5TW from TradingView...')
        tv = TvDatafeed(username=tv_user, password=tv_pass) if tv_user else TvDatafeed()
        data = tv.get_hist('S5TW', 'CBOE', interval=Interval.in_daily, n_bars=5)
        if data is None or data.empty:
            print('  [WARN] B20: S5TW returned no data')
            return np.nan
        val = float(data['close'].dropna().iloc[-1])
        print(f'  B20 (S5TW): {val:.2f}%')
        return val
    except Exception as e:
        print(f'  [WARN] B20 fetch failed: {e}')
        return np.nan

# ── PC Ratio — PUT/CALL from TradingView ──────────────────────────────────────
def fetch_pc_tv(tv_user: str = '', tv_pass: str = '') -> float:
    try:
        from tvdatafeed import TvDatafeed, Interval
        print('  PC: fetching PUT/CALL from TradingView...')
        tv = TvDatafeed(username=tv_user, password=tv_pass) if tv_user else TvDatafeed()
        data = tv.get_hist('PC', 'USI', interval=Interval.in_daily, n_bars=5)
        if data is None or data.empty:
            print('  [WARN] PC: returned no data')
            return np.nan
        val = float(data['close'].dropna().iloc[-1])
        print(f'  PC Ratio (TV): {val:.3f}')
        return val
    except Exception as e:
        print(f'  [WARN] PC fetch failed: {e}')
        return np.nan

# ── SKEW — CBOE SKEW Index ─────────────────────────────────────────────────────
def fetch_skew() -> float:
    try:
        raw = yf.download('^SKEW', period='5d', auto_adjust=True, progress=False)
        if raw.empty:
            print('  [WARN] SKEW: ^SKEW returned no data')
            return np.nan
        if isinstance(raw.columns, pd.MultiIndex):
            s = raw['Close']['^SKEW']
        else:
            s = raw['Close']
        val = float(s.dropna().iloc[-1])
        print(f'  SKEW fetched: {val:.2f}')
        return val
    except Exception as e:
        print(f'  [WARN] SKEW fetch failed: {e}')
        return np.nan

# ── Build inp_map for a single "today" row ─────────────────────────────────────
def build_inp_map(hist: pd.DataFrame) -> dict:
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
        last_adl      = float(adl_series.iloc[-1])
        last_adl_date = hist.loc[adl_series.index[-1], 'date']
        adl_new       = fetch_adl(last_adl, pd.Timestamp(last_adl_date))
        if adl_new:
            adl_today = list(adl_new.values())[-1]
        else:
            adl_today = last_adl
            print(f'  ADL: using carry-forward value {adl_today:,.0f}')

    # ── TradingView credentials from environment ────────────────────────────
    tv_user = os.environ.get('TV_USERNAME', '')
    tv_pass = os.environ.get('TV_PASSWORD', '')

    # ── B20% via TradingView S5TW ───────────────────────────────────────────
    b20_today = fetch_b20(tv_user=tv_user, tv_pass=tv_pass)

    # ── PC Ratio via TradingView ────────────────────────────────────────────
    pc_today = fetch_pc_tv(tv_user=tv_user, tv_pass=tv_pass)

    # ── SKEW — fallback if yfinance returns NaN for ^SKEW today ────────────
    skew_today = fetch_skew()

    inp_map = {
        today_ts: {
            'adl_level':  adl_today,
            'b20_pct':    b20_today,
            'zero_gamma': zg,
            'pc_ratio':   pc_today,
            'skew':       skew_today,
        }
    }

    print(f'\n  Inputs for {today_ts.date()}:')
    print(f'    ADL        = {adl_today:,.0f}' if not np.isnan(adl_today) else '    ADL        = NaN (neutral)')
    print(f'    B20%       = {b20_today:.1f}%' if not np.isnan(b20_today) else '    B20%       = NaN (neutral)')
    print(f'    Zero Gamma = {zg:.2f}' if not np.isnan(zg) else '    Zero Gamma = NaN (neutral)')
    print(f'    PC Ratio   = {pc_today:.3f}' if not np.isnan(pc_today) else '    PC Ratio   = NaN (neutral)')
    print(f'    SKEW       = {skew_today:.2f}' if not np.isnan(skew_today) else '    SKEW       = NaN (pipeline fallback)')

    return inp_map
