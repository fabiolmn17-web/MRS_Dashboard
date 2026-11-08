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

  7 Nov 2026 (scanner v3, long + short scans):
  Short list (mirror of the momentum list): held score <= -1.0 AND close < SMA200 AND SMA50 < SMA200.
  Pullback (long):  5-day return <= -2% AND close within -1 to +0.5 ATR of a RISING SMA (20-day: its 5-bar slope > 0;
                    50-day: its 5-bar slope > 0, momentum Building / Persistent / Fading). 50-day first if both.
  Rally (short):    5-day return >= +2% AND close within -0.5 to +1 ATR of a FALLING SMA (5-bar slope < 0). 50 first.
  Descriptive definitions (the indicator's pullback colour, mirrored for shorts), not tested signals.
  ADR% = 100 x (mean of high / low over the last 20 sessions - 1), as TradingView's ADR%; avg_vol50 = mean volume of
  the last 50 sessions. The dashboard scans keep only stocks with avg_vol50 >= 500K and ADR% >= 3 (7 Nov 2026).
  sma200_slope21 = % change of the 200-day SMA over 21 sessions.
  7 Nov 2026 (c): Deep dip / Deep rip scans (dashboard): ext200_atr = (close - SMA200) / ATR(14);
  off_lo252 = % of the close above the lowest close of the last 252 sessions (52-week low).
  8 Nov 2026: Deep dip / Deep rip are TRIGGERS (Deep_Dip study rules), not proximity (deep_dip_fields below).
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
          'ret_5d', 'pullback', 'pct_52w', 'mom_list',
          'ext20_atr', 'sma20_slope5', 'short_list', 'pb_setup', 'rally_setup', 'adr_pct', 'avg_vol50',
          'sma200_slope21', 'ext200_atr', 'off_lo252', 'ema21_above', 'ema10_above21']
# 8 Nov 2026: Deep dip (dd_) / Deep rip (dr_) triggers on the 200 and 250 SMA (TRADING DESK/Studies/Deep_Dip)
DD_KEYS = ['trig', 'k', 'sma', 'days', 'entry', 'stop', 'risk_pct', 'rvol', 'slope21', 'watch', 'watch_atr']
DD_FIELDS = [f'{p}_{k}' for p in ('dd', 'dr') for k in DD_KEYS]
FIELDS += DD_FIELDS


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



def deep_dip_fields(df: pd.DataFrame, side: str = 'long', n_above: int = 20, max_below: int = 5, max_days: int = 3) -> dict:
    """8 Nov 2026 (Deep_Dip study, v1-v3). Long ('dd_'): the 20 closes before were all above the 200 or 250 SMA, then
      UNDERCUT  = low below the SMA, close back above (prior close above), or
      RECLAIM   = 1-5 closes below the SMA, then a close back above (8 Nov 2026: 5, was 3 in the study; wider net,
                  Fabio screens by hand).
    A trigger in the last max_days sessions stays listed while every close since is still above that SMA and the
    stop (trigger-day low - 1 ATR(10)) has not been hit. If both SMAs trigger on the same bar: '200+250'.
    entry = trigger close; risk_pct = (entry - stop) / entry; rvol = trigger-day volume / mean of the 10 before;
    slope21 = % change of the trigger SMA over 21 sessions (the 200 when both).
    WATCH (no live trigger): the last 20 closes all above the SMA and the close within 1 ATR above it ('Near 200' /
    'Near 250' / 'Near 200+250'); watch_atr = (close - SMA) / ATR of the nearer one.
    Short ('dr_'): the exact mirror (rip up into the SMA from below). Descriptive; see the study for the evidence."""
    pre = 'dd' if side == 'long' else 'dr'
    out = {f'{pre}_{k}': (np.nan if k not in ('trig', 'sma', 'watch') else '') for k in DD_KEYS}
    s = 1 if side == 'long' else -1
    try:
        d = df[['High', 'Low', 'Close']].dropna()
        if len(d) < 250 + n_above + max_below + max_days + 2:
            return out
        h, l, c = d['High'].values, d['Low'].values, d['Close'].values
        vol = df['Volume'].reindex(d.index).values if 'Volume' in df.columns else np.full(len(c), np.nan)
        a10 = _atr_wilder(d, 10).reindex(d.index).values
        n = len(c)
        smas = {L: d['Close'].rolling(L).mean().values for L in (200, 250)}
        best = None                                   # (days_ago, [L...], type, k)
        for L, sma in smas.items():
            x = s * (c - sma)
            for dago in range(0, max_days + 1):
                i = n - 1 - dago
                if not np.isfinite(sma[i]) or x[i] <= 0:
                    continue
                typ, k = None, 0
                if x[i-1] > 0:
                    pierce = (l[i] < sma[i]) if s > 0 else (h[i] > sma[i])
                    if pierce and (x[i-n_above:i] > 0).all():
                        typ = 'Undercut' if s > 0 else 'Overcut'
                else:
                    while k <= max_below and x[i-1-k] <= 0:
                        k += 1
                    if 1 <= k <= max_below and (x[i-k-n_above:i-k] > 0).all():
                        typ = 'Reclaim' if s > 0 else 'Lost reclaim'
                    else:
                        k = 0
                if typ is None:
                    continue
                stop = (l[i] - a10[i]) if s > 0 else (h[i] + a10[i])
                later = slice(i + 1, n)
                if (x[later] <= 0).any() or ((l[later] <= stop).any() if s > 0 else (h[later] >= stop).any()):
                    continue                          # closed back through the SMA, or stopped out since
                if best is None or dago < best[0]:
                    best = (dago, [L], typ, k)
                elif dago == best[0] and L not in best[1]:
                    best[1].append(L)
                break                                 # most recent trigger on this SMA
        if best is not None:
            dago, Ls, typ, k = best
            i = n - 1 - dago
            stop = (l[i] - a10[i]) if s > 0 else (h[i] + a10[i])
            Lr = 200 if 200 in Ls else 250
            sm = smas[Lr]
            v10 = np.nanmean(vol[i-10:i]) if i >= 10 else np.nan
            out.update({f'{pre}_trig': typ, f'{pre}_k': k, f'{pre}_sma': '+'.join(str(L) for L in sorted(Ls)),
                        f'{pre}_days': dago, f'{pre}_entry': float(c[i]), f'{pre}_stop': float(stop),
                        f'{pre}_risk_pct': float(s * (c[i] - stop) / c[i] * 100),
                        f'{pre}_rvol': float(vol[i] / v10) if v10 and np.isfinite(v10) and v10 > 0 else np.nan,
                        f'{pre}_slope21': float((sm[-1] / sm[-22] - 1) * 100)})
        else:
            near = []
            for L, sma in smas.items():
                x = s * (c - sma)
                if np.isfinite(sma[-1]) and (x[-n_above:] > 0).all() and x[-1] <= a10[-1]:
                    near.append((x[-1] / a10[-1], L))
            if near:
                near.sort()
                Ls = sorted(L for _, L in near)
                sm = smas[near[0][1]]
                out.update({f'{pre}_watch': 'Near ' + '+'.join(str(L) for L in Ls), f'{pre}_watch_atr': float(near[0][0]),
                            f'{pre}_slope21': float((sm[-1] / sm[-22] - 1) * 100)})
    except Exception as e:  # never break the scan on one ticker
        logger.warning(f'deep dip fields error: {e}')
    return out

def momentum_fields(df: Optional[pd.DataFrame], spy_df: Optional[pd.DataFrame]) -> dict:
    """All momentum / trend / Ext-Mom fields for one stock. NaN / '' where history is too short."""
    out = {k: np.nan for k in FIELDS}
    out.update({'held_label': '', 'held_trend': '', 'pending_label': '', 'tag_1m': '', 'ext_zone': '',
                'mom_zone': '', 'pullback': '', 'mom_list': False, 'above_sma200': False, 'sma50_above_sma200': False,
                'short_list': False, 'pb_setup': '', 'rally_setup': ''})
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
        # 7 Nov 2026: short list, 20-day SMA, pullback / rally setups
        below200 = (not np.isnan(s200)) and p < s200
        death = (not np.isnan(s200)) and s50 < s200
        out['short_list'] = bool((not pd.isna(hs)) and hs <= -1.0 and below200 and death)
        sma20 = c.rolling(20).mean()
        s20 = float(sma20.iloc[-1])
        sl20 = float((sma20.iloc[-1] / sma20.iloc[-6] - 1) * 100) if len(c) > 25 else np.nan
        e20 = (p - s20) / a if a and a > 0 else np.nan
        out.update({'ext20_atr': e20, 'sma20_slope5': sl20})
        out['sma200_slope21'] = (float((sma200.iloc[-1] / sma200.iloc[-22] - 1) * 100)
                                 if len(c) >= 221 and sma200.iloc[-22] > 0 else np.nan)
        out['ext200_atr'] = (p - s200) / a if (not np.isnan(s200)) and a and a > 0 else np.nan
        out['off_lo252'] = (p / float(c.iloc[-252:].min()) - 1) * 100
        e21 = c.ewm(span=21, adjust=False).mean(); e10 = c.ewm(span=10, adjust=False).mean()
        out['ema21_above'] = bool(p > float(e21.iloc[-1]))
        out['ema10_above21'] = bool(float(e10.iloc[-1]) > float(e21.iloc[-1]))
        out.update(deep_dip_fields(df, 'long')); out.update(deep_dip_fields(df, 'short'))
        hl = (d['High'] / d['Low']).iloc[-20:]
        out['adr_pct'] = float((hl.mean() - 1) * 100) if len(hl) == 20 else np.nan
        if 'Volume' in df.columns:
            vv = df['Volume'].dropna().iloc[-50:]
            out['avg_vol50'] = float(vv.mean()) if len(vv) >= 20 else np.nan
        ok = lambda v: v is not None and not np.isnan(v)
        if ok(r5) and r5 <= -2:
            if ok(ext) and -1 <= ext <= 0.5 and ok(sl) and sl > 0:
                out['pb_setup'] = 'Pullback to 50-day'
            elif ok(e20) and -1 <= e20 <= 0.5 and ok(sl20) and sl20 > 0:
                out['pb_setup'] = 'Pullback to 20-day'
        if ok(r5) and r5 >= 2:
            if ok(ext) and -0.5 <= ext <= 1 and ok(sl) and sl < 0:
                out['rally_setup'] = 'Rally to 50-day'
            elif ok(e20) and -0.5 <= e20 <= 1 and ok(sl20) and sl20 < 0:
                out['rally_setup'] = 'Rally to 20-day'
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
