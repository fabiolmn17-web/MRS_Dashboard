"""
auto_fetch.py — Automatic data fetching for the three previously-manual inputs
"""
import re
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import date, timedelta
warnings.filterwarnings('ignore')

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

# ── B20% — computed from individual yfinance downloads (100-stock sample) ──────
# S&P 500 representative sample — 10 stocks per sector, avoids bulk endpoint
_SP500_SAMPLE = [
    # Tech
    'AAPL','MSFT','NVDA','AVGO','ORCL','CRM','AMD','QCOM','TXN','AMAT',
    # Financials
    'JPM','BAC','WFC','GS','MS','BLK','SCHW','AXP','PGR','CB',
    # Healthcare
    'LLY','UNH','JNJ','ABBV','MRK','TMO','ABT','DHR','SYK','ISRG',
    # Consumer Discretionary
    'AMZN','TSLA','HD','MCD','NKE','BKNG','LOW','TJX','SBUX','GM',
    # Industrials
    'CAT','RTX','HON','UPS','BA','GE','LMT','DE','ETN','EMR',
    # Communication
    'META','GOOGL','GOOG','NFLX','DIS','T','VZ','CMCSA','EA','TTWO',
    # Consumer Staples
    'PG','KO','PEP','COST','WMT','PM','MO','CL','GIS','KHC',
    # Energy
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY','HAL',
    # Materials
    'LIN','APD','ECL','SHW','FCX','NEM','ALB','PPG','VMC','MLM',
    # Real Estate + Utilities
    'PLD','AMT','EQIX','NEE','SO','DUK','AEP','D','EXC','SRE',
]

def fetch_b20() -> float:
    try:
        print(f'  B20: computing from {len(_SP500_SAMPLE)}-stock S&P 500 sample...')
        above = 0
        valid = 0
        for ticker in _SP500_SAMPLE:
            try:
                hist = yf.Ticker(ticker).history(period='45d', auto_adjust=True)
                if hist.empty or len(hist) < 21:
                    continue
                close = hist['Close'].dropna()
                if len(close) < 21:
                    continue
                ma20 = close.rolling(20).mean()
                if close.iloc[-1] > ma20.iloc[-1]:
                    above += 1
                valid += 1
            except Exception:
                continue
        if valid < 20:
            print(f'  [WARN] B20: only {valid} valid stocks — insufficient')
            return np.nan
        pct = round(above / valid * 100, 2)
        print(f'  B20: {pct:.1f}% ({above}/{valid} stocks above 20-day MA)')
        return pct
    except Exception as e:
        print(f'  [WARN] B20 fetch failed: {e}')
        return np.nan

# ── PC Ratio — CBOE tickers + SPY options proxy fallback ──────────────────────
def fetch_pc_ratio() -> float:
    # Try 1: Yahoo Finance CBOE tickers
    for ticker in ['^CPCE', '^CPC', '^CPCI']:
        try:
            print(f'  PC: trying {ticker} from Yahoo Finance...')
            raw = yf.download(ticker, period='5d', auto_adjust=True, progress=False)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                s = raw['Close'][ticker]
            else:
                s = raw['Close']
            s = s.dropna()
            if s.empty:
                continue
            val = float(s.iloc[-1])
            print(f'  PC Ratio ({ticker}): {val:.3f}')
            return val
        except Exception as e:
            print(f'  [WARN] {ticker} failed: {e}')
            continue

    # Try 2: Compute from SPY options chain via yfinance
    try:
        print('  PC: computing from SPY options chain...')
        spy = yf.Ticker('SPY')
        expiries = spy.options
        if not expiries:
            raise ValueError('No expiries available')
        total_puts  = 0.0
        total_calls = 0.0
        for exp in expiries[:2]:   # nearest 2 expiries for volume
            chain = spy.option_chain(exp)
            total_puts  += chain.puts['volume'].fillna(0).sum()
            total_calls += chain.calls['volume'].fillna(0).sum()
        if total_calls < 100:
            raise ValueError(f'Insufficient call volume: {total_calls}')
        val = round(total_puts / total_calls, 3)
        print(f'  PC Ratio (SPY options proxy): {val:.3f}')
        return val
    except Exception as e:
        print(f'  [WARN] PC options proxy failed: {e}')

    print('  [WARN] PC Ratio: all sources failed — carrying forward from history')
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

    zg = fetch_zero_gamma()

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

    b20_today  = fetch_b20()
    pc_today   = fetch_pc_ratio()
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
