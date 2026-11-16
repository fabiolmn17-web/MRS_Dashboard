"""
vol_core.py — SPY vol surface snapshot for the MRS (8.4.0, 16 Nov 2026)
=======================================================================
The term-structure / event maths are ported unchanged from Gamma Live panels.py v5.9 (vs_term, vs_event), so the MRS
panel reads the same as Gamma Live. Two producers write the same snapshot format:
  Gamma Live  (primary)  gamma_live/mrs_push.py builds it from the IBKR close chain and sends it to the repo
                         (workflow vol_snapshot.yml -> vol_snapshot.py -> vol_gl.json)
  Yahoo       (backup)   update_data.py builds it nightly from the yfinance chain -> vol_yahoo.json
The dashboard (vol_view.py) uses Gamma Live when it has the session, else Yahoo, and says which.
Snapshot = {date, src, spot, at, term: [...], smiles: {expiry: [[strike, iv], ...]}, event: {...}}. Descriptive.
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RATE = 0.043
HUMP_X, HUMP_DTE = 1.3, 10
SMILE_DTE, SMILE_BAND, TERM_DTE = 35, 0.07, 60
KEEP_DAYS = 15
BASE = Path(__file__).parent
FILES = {'gl': BASE / 'vol_gl.json', 'yahoo': BASE / 'vol_yahoo.json'}


def fin(x):
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S, K, T, iv, right, r=RATE):
    if not all(fin(v) for v in (S, K, T, iv)) or S <= 0 or K <= 0 or T <= 0 or iv <= 0:
        return np.nan
    sd = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / sd
    d2 = d1 - sd
    if right == 'C':
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _holidays():
    try:
        from data_check import _CAL
        return [d.strftime('%Y-%m-%d') for d in _CAL.holidays(start='2024-01-01', end='2030-12-31')]
    except Exception:
        return []


HOLIDAYS = _holidays()


def trading_days(expiry: str, day: str, left_today: float = 0.0) -> float:
    d0 = np.datetime64(day) + np.timedelta64(1, 'D')
    e1 = np.datetime64(pd.Timestamp(expiry).date()) + np.timedelta64(1, 'D')
    full = int(np.busday_count(d0, e1, holidays=HOLIDAYS)) if e1 > d0 else 0
    return left_today + full


def _at(x, y, x0):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 2 or x0 < x.min() or x0 > x.max():
        return np.nan
    return float(np.interp(x0, x, y))


def term(w: pd.DataFrame, spot: float, day: str) -> pd.DataFrame:
    """Per expiry (after the close, nothing left of today): ATM IV, ATM straddle, 5%-below put IV and skew, trading days,
    forward vol on the trading-day clock (panels.vs_term, left_today = 0)."""
    rows = []
    for e, g in w.groupby('expiry'):
        g = g.sort_values('strike')
        t = float(g['t'].dropna().median()) if g['t'].notna().any() else np.nan
        atm = _at(g['strike'].values, g['iv'].values, spot)
        strad = _at(g['strike'].values, (g['c_mid'] + g['p_mid']).values, spot)
        p95 = _at(g['strike'].values, g['iv'].values, spot * 0.95)
        rows.append(dict(expiry=str(e), dte=int(g['dte'].iloc[0]), t=t, atm_iv=atm, straddle=strad, iv_95=p95,
                         skew_95=p95 - atm if fin(p95) and fin(atm) else np.nan))
    tm = pd.DataFrame(rows)
    if not len(tm):
        return tm
    tm = tm[tm['dte'] >= 1].sort_values('t').reset_index(drop=True)
    tm['td'] = [max(trading_days(e, day), 1 / 390) for e in tm['expiry']]
    var = tm['atm_iv'] ** 2 * tm['t']
    tm['var_td'] = var / tm['td']
    fv = np.sqrt(np.clip(var.diff() / tm['td'].diff(), 0, None) * 252)
    fv[tm['td'].diff() <= 0.05] = np.nan
    if len(fv):
        fv.iloc[0] = np.sqrt(tm['var_td'].iloc[0] * 252)
    tm['fwd_iv'] = fv
    return tm


def event(tm: pd.DataFrame, spot: float) -> dict:
    out = dict(hump=None, base_iv=None, base_exp=None, next=None)
    ok = tm[tm['atm_iv'].notna() & tm['t'].notna()]
    if len(ok) < 3:
        return out
    base = ok.iloc[(ok['dte'] - 30).abs().argsort()[:1]].iloc[0]
    out['base_iv'], out['base_exp'] = float(base['atm_iv']), str(base['expiry'])
    ref = ok.loc[ok['dte'] >= 7, 'fwd_iv'].median()
    if fin(ref) and ref > 0:
        cand = ok[(ok['dte'] <= HUMP_DTE) & (ok['dte'] >= 1) & (ok['fwd_iv'] >= HUMP_X * ref)]
        if len(cand):
            h = cand.loc[cand['fwd_iv'].idxmax()]
            out['hump'] = dict(expiry=str(h['expiry']), dte=int(h['dte']), fwd_iv=float(h['fwd_iv']), ref_fwd=float(ref))
    n = ok.iloc[0]
    if fin(n['straddle']):
        sd = math.sqrt(float(base['var_td']) * float(n['td']))
        iv_eq = sd / math.sqrt(float(n['t']))
        norm = bs(spot, spot, float(n['t']), iv_eq, 'C') + bs(spot, spot, float(n['t']), iv_eq, 'P')
        prem = float(n['straddle']) - norm if fin(norm) else np.nan
        out['next'] = dict(expiry=str(n['expiry']), dte=int(n['dte']), straddle=float(n['straddle']),
                           move_pct=float(n['straddle']) / spot * 100, normal=norm, premium=prem,
                           prem_share=prem / float(n['straddle']) * 100 if fin(prem) else None, atm_iv=float(n['atm_iv']))
    return out


def snapshot(w: pd.DataFrame, spot: float, day: str, src: str) -> dict:
    """Compact, JSON-ready snapshot from a per-strike frame (expiry, strike, iv, c_mid, p_mid, t, dte)."""
    tm = term(w, spot, day)
    ev = event(tm, spot) if len(tm) else {}
    keep = ['expiry', 'dte', 'atm_iv', 'straddle', 'skew_95', 'fwd_iv', 'td', 't']
    tl = [{k: (round(float(r[k]), 5) if k not in ('expiry', 'dte') and fin(r[k]) else (r[k] if k in ('expiry',) else
           (int(r[k]) if k == 'dte' else None))) for k in keep} for _, r in tm[tm['dte'] <= TERM_DTE].iterrows()]
    sm = {}
    near = w[(w['dte'] >= 1) & (w['dte'] <= SMILE_DTE) & w['iv'].notna() & w['strike'].between(spot * (1 - SMILE_BAND), spot * (1 + SMILE_BAND))]
    for e, g in near.groupby('expiry'):
        sm[str(e)] = [[round(float(k), 2), round(float(v), 4)] for k, v in zip(g['strike'], g['iv'])]
    def clean(d):
        if isinstance(d, dict):
            return {k: clean(v) for k, v in d.items()}
        if isinstance(d, (float, np.floating)):
            return round(float(d), 5) if np.isfinite(d) else None
        return d
    return dict(date=str(day), src=src, spot=round(float(spot), 4), at=datetime.utcnow().strftime('%Y-%m-%dT%H:%MZ'),
                term=tl, smiles=sm, event=clean(ev))


def store(snap: dict, src: str, path: Path = None) -> Path:
    path = path or FILES[src]
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            data = {}
    data[snap['date']] = snap
    data = {k: data[k] for k in sorted(data)[-KEEP_DAYS:]}
    path.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
    return path


def load(src: str) -> dict:
    p = FILES[src]
    try:
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    except Exception:
        return {}


def yahoo_frame(max_dte: int = 45):
    """Yahoo backup: per-strike OTM IV / mids for SPY expiries up to max_dte (yfinance, after the close)."""
    import yfinance as yf
    t = yf.Ticker('SPY')
    h = t.history(period='5d', auto_adjust=False)
    spot = float(h['Close'].iloc[-1]); day = pd.Timestamp(h.index[-1]).tz_localize(None).normalize()
    rows = []
    for e in t.options:
        dte = (pd.Timestamp(e) - day).days
        if dte < 1 or dte > max_dte:
            continue
        ch = t.option_chain(e)
        c = ch.calls.set_index('strike'); p = ch.puts.set_index('strike')
        idx = c.index.union(p.index)
        for k in idx:
            cb, ca = (float(c.at[k, 'bid']), float(c.at[k, 'ask'])) if k in c.index else (np.nan, np.nan)
            pb, pa = (float(p.at[k, 'bid']), float(p.at[k, 'ask'])) if k in p.index else (np.nan, np.nan)
            civ = float(c.at[k, 'impliedVolatility']) if k in c.index else np.nan
            piv = float(p.at[k, 'impliedVolatility']) if k in p.index else np.nan
            otm_put = k < spot
            iv, bid = (piv, pb) if otm_put else (civ, cb)
            rows.append(dict(expiry=e.replace('-', ''), strike=float(k), dte=dte, t=dte / 365.0,
                             c_mid=(cb + ca) / 2 if ca > 0 else np.nan, p_mid=(pb + pa) / 2 if pa > 0 else np.nan,
                             iv=iv if (fin(iv) and 0.02 < iv < 3 and fin(bid) and bid > 0) else np.nan))
    return pd.DataFrame(rows), spot, day.strftime('%Y-%m-%d')
