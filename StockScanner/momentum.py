"""
momentum.py — Momentum & trend fields for the nightly scan (one stock at a time, price data only)
==================================================================================================
Used by scanner.py. Same definitions as the dashboard and the Epistruct studies:

  RS label (rs_labels.py): signs of RS 1Y / 6M / 3M vs SPY, held for 3 sessions; RS 1M shown, not scored.
  Momentum list:  held score >= +1.0  AND  close > SMA200  AND  SMA50 > SMA200.

  Ext-Mom zones (Epistruct_ExtMom v4.2, ATR units, SMA 50):
    extension = (close - SMA50) / ATR(14, Wilder)
      Deep below <= -3 | Below <= -1 | Just below <= 0 | Near <= +0.5 | Above <= +5 | Far above > +5
      ("Above" was "Stretched" and "Far above" was "Extreme" until 28 Oct 2026: in the stretched-label study the
       left tail did not grow with extension in stocks, so the zone is a trend state, not a reversal warning)
    momentum  = % change of SMA50 over 5 bars
      Weak <= -0.5 | Falling <= 0 | Building <= +0.5 (Fading if it was above +0.5 in the last 11 bars) | Persistent > +0.5
    5-day return: Dip <= -2% | Pullback <= -3% | Sharp pullback <= -4%

  The odds behind these zones were measured on SPY and QQQ (index ETFs), not on single stocks: on a stock
  they describe the state, not tested probabilities.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import rs_labels  # noqa: E402

logger = logging.getLogger(__name__)

EXT_ZONES = ['Deep below', 'Below', 'Just below', 'Near', 'Above', 'Far above']
ZONE_ALIAS = {'Stretched': 'Above', 'Extreme': 'Far above'}   # old labels in older CSVs
MOM_ZONES = ['Weak', 'Falling', 'Building', 'Persistent', 'Fading']
PB_BINS = ['None', 'Dip', 'Pullback', 'Sharp pullback']

FIELDS = ['held_score', 'held_label', 'held_trend', 'days_in_label', 'pending_label', 'pending_days', 'tag_1m',
          'above_sma200', 'sma50_above_sma200', 'ext_atr', 'ext_zone', 'sma50_slope5', 'mom_zone',
          'ret_5d', 'pullback', 'pct_52w', 'mom_list']


def _atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c = df['Close']
    pc = c.shift(1)
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - pc).abs(), (df['Low'] - pc).abs()], axis=1).max(axis=1)
    # RMA seeded with the simple mean of the first n true ranges (as ta.atr in Pine)
    tr = tr.dropna()
    out = pd.Series(np.nan, index=tr.index)
    if len(tr) < n:
        return out
    a = tr.iloc[:n].mean()
    vals = [np.nan] * (n - 1) + [a]
    for x in tr.iloc[n:]:
        a = (a * (n - 1) + x) / n
        vals.append(a)
    return pd.Series(vals, index=tr.index)


def ext_zone(x: float) -> str:
    if x is None or np.isnan(x):
        return ''
    return 'Deep below' if x <= -3 else 'Below' if x <= -1 else 'Just below' if x <= 0 else 'Near' if x <= 0.5 else 'Above' if x <= 5 else 'Far above'


def mom_zone(slope: float, slope_peak: float) -> str:
    if slope is None or np.isnan(slope):
        return ''
    if slope <= -0.5:
        return 'Weak'
    if slope <= 0:
        return 'Falling'
    if slope <= 0.5:
        return 'Fading' if (not np.isnan(slope_peak) and slope_peak > 0.5) else 'Building'
    return 'Persistent'


def pullback_bin(r5: float) -> str:
    if r5 is None or np.isnan(r5):
        return ''
    return 'Sharp pullback' if r5 <= -4 else 'Pullback' if r5 <= -3 else 'Dip' if r5 <= -2 else 'None'


def momentum_fields(df: Optional[pd.DataFrame], spy_df: Optional[pd.DataFrame]) -> dict:
    """All momentum / trend / Ext-Mom fields for one stock. NaN / '' where history is too short."""
    out = {k: np.nan for k in FIELDS}
    out.update({'held_label': '', 'held_trend': '', 'pending_label': '', 'tag_1m': '', 'ext_zone': '',
                'mom_zone': '', 'pullback': '', 'mom_list': False, 'above_sma200': False, 'sma50_above_sma200': False})
    if df is None or spy_df is None or df.empty or 'Close' not in df.columns:
        return out
    d = df[['High', 'Low', 'Close']].dropna()
    c = d['Close']
    if len(c) < 60:
        return out
    try:
        li = rs_labels.label_info(c, spy_df['Close'])
        if li is not None:
            out.update({'held_score': li['Score'], 'held_label': li['Label'], 'held_trend': li['Trend'],
                        'days_in_label': li['DaysInLabel'], 'pending_label': li['PendingLabel'],
                        'pending_days': li['PendingDays'], 'tag_1m': li['Tag1M']})
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        atr = _atr_wilder(d, 14)
        p = float(c.iloc[-1])
        s50 = float(sma50.iloc[-1])
        s200 = float(sma200.iloc[-1]) if len(c) >= 200 else np.nan
        a = float(atr.iloc[-1]) if len(atr) else np.nan
        slope = (sma50 / sma50.shift(5) - 1) * 100
        ext = (p - s50) / a if a and a > 0 else np.nan
        sl = float(slope.iloc[-1])
        pk = float(slope.iloc[-11:].max())
        r5 = (p / float(c.iloc[-6]) - 1) * 100 if len(c) > 5 else np.nan
        above200 = (not np.isnan(s200)) and p > s200
        golden = (not np.isnan(s200)) and s50 > s200
        out.update({
            'above_sma200': bool(above200), 'sma50_above_sma200': bool(golden),
            'ext_atr': ext, 'ext_zone': ext_zone(ext), 'sma50_slope5': sl, 'mom_zone': mom_zone(sl, pk),
            'ret_5d': r5, 'pullback': pullback_bin(r5),
            'pct_52w': p / float(c.iloc[-252:].max()) - 1,
        })
        hs = out['held_score']
        out['mom_list'] = bool((not pd.isna(hs)) and hs >= 1.0 and above200 and golden)
    except Exception as e:  # never break the scan on one ticker
        logger.warning(f'momentum fields error: {e}')
    return out


def next_earnings(ticker: str) -> str:
    """Next earnings date (ISO) from yfinance, '' if unknown. Network call: use only for short lists."""
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar
        dates = None
        if isinstance(cal, dict):
            dates = cal.get('Earnings Date')
        elif cal is not None and hasattr(cal, 'loc') and 'Earnings Date' in getattr(cal, 'index', []):
            dates = list(cal.loc['Earnings Date'].values)
        if dates:
            ds = sorted(pd.to_datetime(pd.Series(list(dates))).dt.date)
            today = pd.Timestamp.today().date()
            fut = [x for x in ds if x >= today]
            # only a date still ahead; a past date means the next one is not announced yet
            return fut[0].isoformat() if fut else ''
    except Exception as e:
        logger.debug(f'{ticker}: earnings date error: {e}')
    return ''


# GICS sector names -> the yfinance names the dashboard maps (YF_TO_SECTOR in app.py)
GICS_TO_YF = {
    'Information Technology': 'Technology', 'Health Care': 'Healthcare', 'Financials': 'Financial Services',
    'Consumer Discretionary': 'Consumer Cyclical', 'Communication Services': 'Communication Services',
    'Industrials': 'Industrials', 'Consumer Staples': 'Consumer Defensive', 'Energy': 'Energy',
    'Utilities': 'Utilities', 'Real Estate': 'Real Estate', 'Materials': 'Basic Materials',
}


def sp500_meta() -> dict:
    """ticker -> {name, sector (yfinance naming), industry (GICS sub-industry)} for the S&P 500, one request.
    Fills the sector of every scanned stock, not only the CAN SLIM candidates. {} on failure."""
    try:
        import requests
        from io import StringIO
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        r = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        sym = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
        out = {}
        for _, row in df.iterrows():
            t = str(row[sym]).replace('.', '-').strip()
            out[t] = {'name': str(row.get('Security', t)),
                      'sector': GICS_TO_YF.get(str(row.get('GICS Sector', '')), str(row.get('GICS Sector', ''))),
                      'industry': str(row.get('GICS Sub-Industry', ''))}
        return out
    except Exception as e:
        logger.warning(f'S&P 500 metadata fetch failed: {e}')
        return {}
