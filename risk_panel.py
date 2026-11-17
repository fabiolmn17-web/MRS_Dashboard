"""
risk_panel.py — "Risk today" panel (MRS v5, 30 Oct 2026)
==========================================================
Desk lines built on studies/mrs_risk_control (steps 1-4) and studies/model_portfolio. Nothing here changes the score.

  Pedal           REBOUND WINDOW (key 'PRESS'): SPY > 1 ATR below its 50-day SMA (upside beyond the VIX level: Tested;
                  step 5: mechanical support signals did not improve the path and gave up upside; no high-beta excess)
                  EASE OFF: RISK-OFF and SPY below its close of a year ago (fast drops 1 in 3; Tested vs other days)
                  NORMAL: everything else
  Today's cell    20-session distribution of SPY since May 2008 for today's state
                  (band group x 12-month trend x position vs the 50-day). Descriptive.
  Effective bets  trailing 63-session average correlation of the industry ETFs -> 5 / (1 + 4 rho).
                  Trailing tracks the next 20 sessions (Spearman 0.53, both halves). Descriptive.
  Insurance cost  3-month SPY put, 10% OTM, Black-Scholes on VIX3M with the study's skew surcharge (k = 1.5).
                  Cost per month and its percentile since 2008; state sentence from step 3. Descriptive.
  Vol guide       invested share for a 10% volatility target on 21-session realised SPY vol, cap 60%. Descriptive.

v8.2.0 (15 Nov 2026): the panel is rebuilt on the MRS v8 state and shown open under the v8 card. The MRS band, the
pedal, the insurance tile and the volatility guide are no longer shown (the band is retired; the pedal's names clashed
with the v8 stance; the hedge is on the card; the vol-target guide is not part of the v8 framework). Tiles now:
  SPY vs its 50-day  ATR gauge and sell-off episode; in the dip zone the tested wide-both-ways numbers (Tested)
  This state         20-session distribution for today's SPY phase x VIX width (v8 cell). 8.6.0: S&P 500 since 1990
                     (risk_long.csv, built offline by build_risk_long.py); falls back to SPY since 2008. Descriptive.
  Clock              per-trade quality by sessions since the last DIP ZONE day (random stock entries). 8.6.0: 2001-26,
                     split by the 200-day filter (MRS v8.5). Descriptive.
  Effective bets     unchanged (Tested)
The forward-log columns (pedal, cell, put3m_cost, vt_invested) are still written by the nightly update.

Labels and texts follow the MRS evidence audit v2 (8 Nov 2026, studies/_framework/audit/mrs_risk_control): REBOUND
WINDOW and EASE OFF Tested; effective bets Tested; vol guide Provisional; today's cell and insurance cost Descriptive.

All functions take plain DataFrames / Series so the nightly update (forward log) and the app share one implementation.
"""
import html as _html
import math

import numpy as np
import pandas as pd

TEAL, TEAL_M, ORANGE, ORANGE_M = '#85cac5', '#649995', '#f4a062', '#b9805a'
COPPER, INK, MUTED, DIM, LINE = '#e3b48e', '#f3ece2', '#a89f95', '#6e675f', '#2a2724'
BANDG = {'RISK-ON': 'Calm', 'MILD RISK-ON': 'Calm', 'NEUTRAL': 'Middle', 'MILD RISK-OFF': 'Middle', 'RISK-OFF': 'RISK-OFF'}
EXT_TXT = {'below': 'more than 1 ATR below the 50-day', 'near': 'near the 50-day', 'above': 'more than 1 ATR above the 50-day'}
CORR_TERCILES = (0.458, 0.578)                      # trailing-63 tercile edges (2008-2026)
CORR_STRESS_P90 = {'low': 0.59, 'mid': 0.67, 'high': 0.80}   # forward-20 P90 of correlation by trailing tercile
# trailing-63 correlation, 5% steps of its 2008-2026 distribution (for a percentile read-out)
CORR_Q = [0.143, 0.288, 0.338, 0.367, 0.394, 0.417, 0.443, 0.458, 0.476, 0.492, 0.516, 0.535, 0.549, 0.569, 0.598,
          0.615, 0.638, 0.667, 0.699, 0.784, 0.841]
START = '2008-05-12'
MIN_CELL_DAYS = 150
VIX_MEDIAN = 17.56                                  # MRS v8 width: expanding median of the VIX since 1990 (13 Nov 2026)
PH_TXT = {0: 'DIP ZONE', 1: 'ticket A open', 2: 'ticket B open', 3: 'no ticket'}
# 8.6.0 (17 Nov 2026): random uptrend stock entries, runner exit, net of costs, survivorship-free, 2001-26 pooled
# (2001-08 + 2009-26), by sessions since SPY's last DIP ZONE day and the 200-day filter. (bucket, mean R, P(>= 3R)).
# Filter ON: same sign in both eras in every bucket. Filter OFF: mixed; 60+ has 2009-26 data only (shown as no data).
# Source: Studies/.../oos_2000_2008 + clock_filter.py (17 Nov). Replaces the C1 2009-26 in-sample values.
C1_CLOCK = [('IN DIP', 0.30, 0.095), ('EARLY', 0.17, 0.085), ('MID', 0.10, 0.077), ('LATE', -0.06, 0.064)]
C1_CLOCK_OFF = [('IN DIP', -0.08, 0.059), ('EARLY', 0.12, 0.072), ('MID', -0.17, 0.053), ('LATE', None, None)]
import os as _os
_LONG_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'risk_long.csv')


def long_hist():
    """8.6.0: S&P 500 since 1990 (date, ph, vb, f20, mae20), or None when the file is missing."""
    try:
        x = pd.read_csv(_LONG_PATH)
        return x if len(x) > 1000 else None
    except Exception:
        return None
C1_LBL = {'IN DIP': 'In the dip', 'EARLY': '1–20 after', 'MID': '21–60', 'LATE': '60+'}


def _e(s):
    return _html.escape(str(s), quote=True)


def _fin(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ── state and outcomes ────────────────────────────────────────────────────────────
def spy_states(spy: pd.DataFrame, mrs: pd.Series, label_fn) -> pd.DataFrame:
    """spy: daily OHLC (columns open/high/low/close, any case), mrs: MRS score by date, label_fn: score -> band name.
    Returns one row per session from START with the state and the 20-session outcomes (NaN for the last 20)."""
    s = spy.rename(columns=str.lower)[['high', 'low', 'close']].dropna().sort_index()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize() if getattr(s.index, 'tz', None) else pd.to_datetime(s.index).normalize()
    c, h, l = s['close'], s['high'], s['low']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    # 13 Nov 2026 (audit, Fabio): Wilder ATR(14), the same as the scanner, the allocation tickets and the Ext·Mom
    # indicator (was a simple 14-day mean). Re-check on SPY 2008-26: E < -1 on 948 days vs 940; 5%+ up in 20 sessions
    # 28% vs 29%, fast drop of 7% 27% vs 27%; 44 of 4,746 days change side of -1 ATR. The pedal text stays.
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    sma = c.rolling(50).mean()
    m = mrs.copy(); m.index = pd.to_datetime(m.index).normalize()
    sc = m.reindex(s.index).ffill()
    d = pd.DataFrame(index=s.index)
    d['close'], d['score'] = c, sc
    d['band'] = sc.map(lambda v: label_fn(v) if _fin(v) else np.nan)
    d['bandg'] = d['band'].map(BANDG)
    d['r12'] = c / c.shift(251) - 1
    d['trend'] = np.where(d['r12'] >= 0, 'up', 'down')
    d['ext_atr'] = (c - sma) / atr
    d['ext'] = np.select([d['ext_atr'] < -1, d['ext_atr'] > 1], ['below', 'above'], 'near')
    d.loc[d['ext_atr'].isna(), 'ext'] = np.nan
    cv, lv = c.values, l.values
    n = len(cv); f20 = np.full(n, np.nan); mae = np.full(n, np.nan)
    for i in range(n - 20):
        f20[i] = cv[i + 20] / cv[i] - 1
        mae[i] = lv[i + 1:i + 21].min() / cv[i] - 1
    d['f20'], d['mae20'] = f20, mae
    return d[d.index >= START]


def cell_stats(d: pd.DataFrame, bandg: str, trend: str, ext: str) -> dict:
    """20-session stats for today's cell (falls back to the band group if the cell has < MIN_CELL_DAYS days)."""
    x = d.dropna(subset=['f20'])
    cell = x[(x['bandg'] == bandg) & (x['trend'] == trend) & (x['ext'] == ext)]
    scope = 'cell'
    if len(cell) < MIN_CELL_DAYS:
        cell, scope = x[x['bandg'] == bandg], 'band'
    def st(g):
        return dict(n=len(g), p10=g['f20'].quantile(.1), p50=g['f20'].median(), p90=g['f20'].quantile(.9),
                    up5=(g['f20'] > .05).mean(), dn5=(g['f20'] < -.05).mean(), fast=(g['mae20'] <= -.07).mean())
    return dict(scope=scope, cell=st(cell), all=st(x))


def pedal(bandg: str, trend: str, ext: str) -> str:
    if bandg == 'RISK-OFF' and trend == 'down':
        return 'EASE OFF'
    if ext == 'below':
        return 'PRESS'
    return 'NORMAL'


def industry_corr(closes: pd.DataFrame, window: int = 63):
    """Average pairwise correlation of daily returns over the last `window` sessions -> (rho, n_eff for 5, n_etfs)."""
    if closes is None or len(closes) < window + 1:
        return np.nan, np.nan, 0
    r = closes.sort_index().pct_change().iloc[-window:]
    r = r.dropna(axis=1, thresh=int(0.9 * window))
    if r.shape[1] < 6:
        return np.nan, np.nan, r.shape[1]
    cm = r.corr().values; k = cm.shape[0]
    rho = (np.nansum(cm) - k) / (k * (k - 1))
    return float(rho), float(5 / (1 + 4 * max(rho, 0))), k


def corr_pctile(rho: float) -> float:
    return float(np.interp(rho, CORR_Q, np.linspace(0, 1, len(CORR_Q))))


def corr_tercile(rho: float) -> str:
    return 'low' if rho < CORR_TERCILES[0] else ('mid' if rho < CORR_TERCILES[1] else 'high')


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def put_cost_per_month(vix3m: float, otm: float = 0.10, k: float = 1.5, days: int = 63) -> float:
    """3-month put premium / spot, per month of cover (Black-Scholes, r = q = 0, IV = VIX3M x skew surcharge)."""
    if not _fin(vix3m) or vix3m <= 0:
        return np.nan
    K = 1 - otm; T = days / 252
    iv = vix3m / 100 * (1 + k * math.log(1 / K))
    d1 = (math.log(1 / K) + 0.5 * iv * iv * T) / (iv * math.sqrt(T)); d2 = d1 - iv * math.sqrt(T)
    prem = K * _ncdf(-d2) - _ncdf(-d1)
    return prem / (days / 21)


def put_cost_series(vix3m: pd.Series) -> pd.Series:
    return vix3m.dropna().map(put_cost_per_month)


def vol_guide(close: pd.Series, target: float = 0.10, cap: float = 0.60):
    rv = close.sort_index().pct_change().iloc[-21:].std() * math.sqrt(252)
    return float(rv), float(min(cap, target / rv)) if rv > 0 else cap


def sell_off_episode(ext_atr: pd.Series):
    """Sell-off episode as in step 5: starts on a day > 1 ATR below the 50-day with none in the previous 20 sessions,
    runs until 20 sessions after its last such day. -> (day number of today in the episode or None, start date)."""
    x = (ext_atr.dropna() < -1).values
    idx = ext_atr.dropna().index
    n = len(x)
    recent = [j for j in range(max(0, n - 21), n) if x[j]]
    if not recent:
        return None, None
    j = recent[-1]
    while True:   # walk back to the episode start
        prev = [k for k in range(max(0, j - 20), j) if x[k]]
        if not prev:
            break
        j = prev[0]
    return n - j, idx[j]


def vol_band(v):
    if not _fin(v):
        return np.nan
    return 'CALM' if v < VIX_MEDIAN else ('WIDE' if v < 1.5 * VIX_MEDIAN else 'STRESSED')


def phase_series(spy: pd.DataFrame):
    """Per-session SPY phase (0 DIP ZONE, 1 A open, 2 B open, 3 no ticket) from the allocation replay, as
    chart_view.spy_phase. None without an open column or on error."""
    try:
        import allocation as _al
        x = spy.rename(columns=str.lower)
        if not {'open', 'high', 'low', 'close'} <= set(x.columns) or len(x) < 60:
            return None
        dd = _al.indicators(x[['open', 'high', 'low', 'close']])
        n = len(dd)
        def _open(trades):
            m = np.zeros(n, bool)
            for t in trades:
                m[t['i']:(t['j'] if t['j'] is not None else n)] = True
            return m
        a = _open(_al.replay(dd, _al.rebound_entries(dd), dd['sma50'].values))
        b = _open(_al.replay(dd, _al.breakout_entries(dd), dd['ema21'].values))
        E = dd['E'].values
        ph = np.where(~np.isfinite(E), 3, np.where(E <= -1, 0, np.where(a, 1, np.where(b, 2, 3))))
        return pd.Series(ph, index=pd.to_datetime(dd.index).normalize())
    except Exception:
        return None


def clock_of(ph: pd.Series):
    """(bucket, sessions since the last DIP ZONE day) for the last session of ph."""
    if ph is None or not len(ph):
        return None, None
    p = np.asarray(ph.values, int)
    z = np.flatnonzero(p == 0)
    if not len(z):
        return 'LATE', None
    k = len(p) - 1 - int(z[-1])
    return ('IN DIP' if k == 0 else 'EARLY' if k <= 20 else 'MID' if k <= 60 else 'LATE'), k


def v8_cell(d: pd.DataFrame, ph_hist: pd.Series, vix_hist: pd.Series, code_now: int, vix_now: float, use_long: bool = True) -> dict:
    """20-session stats for today's v8 cell (SPY phase x VIX width); falls back to the phase alone when the cell has fewer
    than MIN_CELL_DAYS days. 8.6.0: S&P 500 since 1990 (risk_long.csv); without it, SPY since 2008 as before."""
    vb_now = vol_band(vix_now)
    L = long_hist() if use_long else None
    if L is not None:
        x, src = L.copy(), 'S&P 500 since 1990'
    else:
        x, src = d.dropna(subset=['f20']).copy(), 'SPY since 2008'
        x['ph'] = ph_hist.reindex(x.index).ffill() if ph_hist is not None else np.nan
        x['vb'] = vix_hist.reindex(x.index).ffill().map(vol_band) if vix_hist is not None else np.nan
    cell = x[(x['ph'] == code_now) & (x['vb'] == vb_now)]
    scope = 'cell'
    if len(cell) < MIN_CELL_DAYS:
        cell, scope = x[x['ph'] == code_now], 'phase'
    def st(g):
        return dict(n=len(g), p10=g['f20'].quantile(.1), p50=g['f20'].median(), p90=g['f20'].quantile(.9),
                    up5=(g['f20'] > .05).mean(), dn5=(g['f20'] < -.05).mean(), fast=(g['mae20'] <= -.07).mean())
    return dict(scope=scope, code=code_now, vb=vb_now, vix=vix_now, cell=st(cell), all=st(x), dz=st(x[x['ph'] == 0]), src=src,
                f_cell=cell['f20'].round(5).tolist(), f_all=x['f20'].round(5).tolist())   # 8.4.0: for the distribution picture


def snapshot(d: pd.DataFrame, rho: float, neff: float, vix: float, vix3m: float, cost_hist: pd.Series) -> dict:
    """Everything the panel and the forward log need, for the last session in d."""
    ep_day, ep_start = sell_off_episode(d['ext_atr'])
    last = d.dropna(subset=['bandg', 'ext']).iloc[-1]
    bg, tr, ex = last['bandg'], last['trend'], last['ext']
    cs = cell_stats(d, bg, tr, ex)
    cost = put_cost_per_month(vix3m)
    pct = float((cost_hist <= cost).mean()) if _fin(cost) and cost_hist is not None and len(cost_hist) else np.nan
    rv, inv = vol_guide(d['close'])
    return dict(date=d.index[-1], band=last['band'], bandg=bg, trend=tr, ext=ex, ext_atr=float(last['ext_atr']),
                r12=float(last['r12']) if _fin(last.get('r12')) else np.nan,
                pedal=pedal(bg, tr, ex), ep_day=ep_day, ep_start=ep_start, cell=cs, rho=rho, neff=neff, inverted=bool(_fin(vix) and _fin(vix3m) and vix > vix3m),
                vix=vix, vix3m=vix3m, cost=cost, cost_pct=pct, rv=rv, invested=inv,
                ext_hist=[float(x) for x in d['ext_atr'].iloc[-60:]])   # 8.4.1: sparkline


def log_row(snap: dict) -> dict:
    """Columns for the forward log in mrs_history.csv."""
    return {'pedal': snap['pedal'], 'cell': f"{snap['bandg']}|{snap['trend']}|{snap['ext']}",
            'ind_corr63': round(snap['rho'], 4) if _fin(snap['rho']) else np.nan,
            'n_eff5': round(snap['neff'], 3) if _fin(snap['neff']) else np.nan,
            'put3m_cost': round(snap['cost'], 5) if _fin(snap['cost']) else np.nan,
            'put3m_cost_pct': round(snap['cost_pct'], 3) if _fin(snap['cost_pct']) else np.nan,
            'vt_invested': round(snap['invested'], 3) if _fin(snap['invested']) else np.nan}


# ── visuals (inline SVG / HTML, same look as the decision strip) ─────────────────
CSS = (
    '<style>'
    '.rp{background:#121212;border:1px solid #2a2724;border-left:3px solid #e3b48e;border-radius:6px;padding:12px 16px 10px 16px;margin:2px 0 14px 0;}'
    '.rp .k{font-size:0.70rem;letter-spacing:0.12em;text-transform:uppercase;color:#e3b48e;font-weight:700;margin-bottom:10px;}'
    '.rp .grid{display:grid;grid-template-columns:1fr 1.4fr 1fr 1fr;gap:12px;}'
    '.rp .tile{background:#0d0d0d;border:1px solid #24211e;border-radius:6px;padding:12px 14px;display:flex;flex-direction:column;gap:9px;}'
    '.rp .lab{font-size:0.68rem;letter-spacing:0.13em;text-transform:uppercase;color:#a89f95;display:flex;justify-content:space-between;align-items:center;gap:6px;}'
    '.rp .big{font-size:1.9rem;font-weight:800;line-height:1.0;}'
    '.rp .act{font-size:0.86rem;color:#f3ece2;line-height:1.5;}'
    '.rp .sub{font-size:0.76rem;color:#a89f95;line-height:1.5;}'
    '.rp .num{font-family:"IBM Plex Mono",monospace;}'
    '.rp .ev{display:inline-block;border:1px solid;border-radius:3px;padding:0 5px;font-size:0.60rem;font-weight:700;letter-spacing:0.04em;white-space:nowrap;line-height:1.5;}'
    '.rp .seg{display:flex;gap:4px;}'
    '.rp .seg div{flex:1;height:14px;border-radius:3px;background:#24211e;}'
    '.rp .trk{position:relative;height:16px;background:#1c1a18;border-radius:3px;flex:1;}'
    '.rp .row{display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#c9c1b7;}'
    '.rp .vg{margin-top:10px;padding-top:8px;border-top:1px solid #1f1d1b;}'
    '@media (max-width:1100px){.rp .grid{grid-template-columns:1fr 1fr;}}'
    '@media (max-width:640px){.rp .grid{grid-template-columns:1fr;}}'
    '</style>')


_EV_TIP = {'Tested': 'Formally tested (clustered bootstrap, p ≤ 0.005 or pre-registered, consistent across subperiods)',
           'Provisional': 'Large enough to matter and consistent across checks, not yet statistically certain: act at '
                          'reduced size and track it',
           'Descriptive': 'Historical regularity: context, not a rule',
           'Watch': 'Not tested: an early warning worth watching'}


def _ev(kind):
    import ui                       # 8.4.0: one evidence tag across the MRS
    return f'<span title="{_EV_TIP.get(kind, "")}">{ui.tag(kind)}</span>'
    c = {'Tested': COPPER, 'Provisional': COPPER, 'Descriptive': MUTED, 'Watch': '#f8d6b4'}[kind]
    bg = {'Tested': 'rgba(227,180,142,0.14)', 'Provisional': 'rgba(227,180,142,0.06)'}.get(kind, 'rgba(0,0,0,0)')
    bs = 'dashed' if kind in ('Provisional', 'Watch') else 'solid'
    return (f'<span class="ev" title="{_EV_TIP[kind]}" style="color:{c};border-color:{c};background:{bg};'
            f'border-style:{bs};">{kind.upper()}</span>')


def _ord(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _pct(v, dec=1, sign=True):
    if not _fin(v):
        return '—'
    v = float(v) * 100
    return f'{"+" if sign and v > 0 else ""}{v:.{dec}f}%'


def _x(v, lo, hi):
    return (min(max(float(v), lo), hi) - lo) / (hi - lo) * 100


def _range_svg(cell: dict, allx: dict, lo=-0.12, hi=0.12) -> str:
    """Bad case (P10) to good case (P90) over the next 20 sessions: all days vs today's state, median tick."""
    def row(lbl, st, col, txtcol):
        a, b, m, z = _x(st['p10'], lo, hi), _x(st['p90'], lo, hi), _x(st['p50'], lo, hi), _x(0, lo, hi)
        return (f'<div class="row"><span style="width:70px;color:{txtcol};">{lbl}</span><div class="trk">'
                f'<div style="position:absolute;left:{z:.1f}%;top:-3px;bottom:-3px;width:1px;background:#5b544d;"></div>'
                f'<div style="position:absolute;left:{a:.1f}%;width:{max(b - a, 1):.1f}%;top:0;bottom:0;background:{col};border-radius:3px;"></div>'
                f'<div title="median" style="position:absolute;left:{m:.1f}%;top:-4px;bottom:-4px;width:3px;margin-left:-1px;background:{INK};border-radius:1px;"></div>'
                f'</div><span class="num" style="width:104px;text-align:right;"><span style="color:{ORANGE};">{_pct(st["p10"])}</span>'
                f' / <span style="color:{TEAL};">{_pct(st["p90"])}</span></span></div>')
    ticks = ''.join(f'<span style="position:absolute;left:{_x(t, lo, hi):.1f}%;transform:translateX(-50%);">{int(t * 100):+d}%</span>'
                    for t in (-0.10, -0.05, 0.0, 0.05, 0.10))
    return (row("Today's state", cell, COPPER, INK) + row('All days', allx, '#5b5148', MUTED)
            + f'<div class="row"><span style="width:70px;"></span><div style="position:relative;flex:1;height:14px;font-size:0.66rem;color:{DIM};">{ticks}</div>'
            f'<span style="width:104px;"></span></div>'
            f'<div class="sub">Bar = bad case (P10) to good case (P90), white tick = median.</div>')


def _split_bar(up: float, dn: float, up_all: float, dn_all: float) -> str:
    """Odds of a ≥5% move up vs down within 20 sessions (bars scaled to 40%), white marker = all days."""
    def row(lbl, v, va, col):
        wv = min(v / 0.4, 1) * 100; wa = min(va / 0.4, 1) * 100
        return (f'<div class="row"><span style="width:70px;">{lbl}</span><div class="trk">'
                f'<div style="position:absolute;left:0;top:0;bottom:0;width:{wv:.0f}%;background:{col};border-radius:3px;"></div>'
                f'<div title="all days {va * 100:.0f}%" style="position:absolute;left:{wa:.0f}%;top:-4px;bottom:-4px;width:3px;margin-left:-1px;background:{INK};"></div></div>'
                f'<span class="num" style="width:104px;text-align:right;color:{col};font-weight:700;">{v * 100:.0f}%'
                f' <span style="color:{DIM};font-weight:400;">vs {va * 100:.0f}%</span></span></div>')
    return (row('≥5% up', up, up_all, TEAL) + row('≥5% down', dn, dn_all, ORANGE)
            + '<div class="sub">Chance of a 5% move within 20 sessions; white marker = all days.</div>')


def _dots(neff: float, total: int = 5) -> str:
    """Five position dots; the filled share = independent bets."""
    out = ''
    for i in range(total):
        f = min(max(neff - i, 0), 1)
        out += (f'<svg width="30" height="30" viewBox="0 0 30 30" style="margin-right:4px;">'
                f'<circle cx="15" cy="15" r="12" fill="none" stroke="{TEAL_M}" stroke-width="2"/>'
                f'<clipPath id="rpc{i}"><rect x="0" y="0" width="{30 * f:.1f}" height="30"/></clipPath>'
                f'<circle cx="15" cy="15" r="12" fill="{TEAL}" clip-path="url(#rpc{i})"/></svg>')
    return f'<div style="display:flex;align-items:center;">{out}</div>'


def _meter(v: float, lo: float, hi: float, marks: dict, grad: str, w=None) -> str:
    """Gradient meter (HTML) with a marker at v and labelled ticks."""
    g = (f'<div style="position:relative;height:14px;border-radius:4px;background:linear-gradient(90deg,{grad});margin-top:8px;">')
    for lbl, z in marks.items():
        g += f'<div style="position:absolute;left:{_x(z, lo, hi):.1f}%;top:-2px;bottom:-2px;width:1px;background:rgba(243,236,226,0.45);"></div>'
    if _fin(v):
        xv = _x(v, lo, hi)
        g += (f'<div style="position:absolute;left:{xv:.1f}%;top:-9px;transform:translateX(-50%);width:0;height:0;'
              f'border-left:7px solid transparent;border-right:7px solid transparent;border-top:9px solid {INK};"></div>'
              f'<div style="position:absolute;left:{xv:.1f}%;top:-2px;bottom:-2px;width:3px;margin-left:-1px;background:{INK};"></div>')
    g += '</div><div style="position:relative;height:14px;font-size:0.66rem;color:#8a8279;">'
    g += ''.join(f'<span style="position:absolute;left:{_x(z, lo, hi):.1f}%;transform:translateX(-50%);top:2px;">{_e(l)}</span>'
                 for l, z in marks.items())
    return g + '</div>'


GRAD = f'{TEAL} 0%,#5b5148 50%,{ORANGE} 100%'


def _atr_gauge(ext_atr: float, lo=-3.0, hi=3.0) -> str:
    """SPY vs its 50-day in ATR; the rebound-window zone (below -1 ATR) shaded."""
    z = _x(-1, lo, hi)
    g = (f'<div style="position:relative;height:14px;border-radius:4px;background:#1c1a18;margin-top:8px;">'
         f'<div style="position:absolute;left:0;width:{z:.1f}%;top:0;bottom:0;background:rgba(133,202,197,0.30);border-radius:4px 0 0 4px;"></div>'
         f'<div style="position:absolute;left:{_x(0, lo, hi):.1f}%;top:-2px;bottom:-2px;width:1px;background:#6e675f;"></div>')
    if _fin(ext_atr):
        xv = _x(ext_atr, lo, hi)
        g += (f'<div style="position:absolute;left:{xv:.1f}%;top:-9px;transform:translateX(-50%);width:0;height:0;'
              f'border-left:7px solid transparent;border-right:7px solid transparent;border-top:9px solid {INK};"></div>'
              f'<div style="position:absolute;left:{xv:.1f}%;top:-2px;bottom:-2px;width:3px;margin-left:-1px;background:{INK};"></div>')
    g += '</div><div style="position:relative;height:14px;font-size:0.66rem;color:#8a8279;">'
    g += (f'<span style="position:absolute;left:{z / 2:.1f}%;transform:translateX(-50%);top:2px;color:{TEAL};">dip zone</span>'
          f'<span style="position:absolute;left:{_x(0, lo, hi):.1f}%;transform:translateX(-50%);top:2px;">50-day</span>'
          f'<span style="position:absolute;right:0;top:2px;">+3 ATR</span>')
    return g + '</div>'


PEDAL_NAME = {'PRESS': 'DIP ZONE', 'NORMAL': 'NORMAL', 'EASE OFF': 'EASE OFF'}   # 'PRESS' kept as the log key
PEDAL_TXT = {
    'PRESS': ('Wide both ways, not more exposure.',
              '<b>Long setups at a level that holds, at normal planned risk. No extra exposure without a setup; '
              'every stop stands.</b><div class="sub" style="margin-top:5px;">Wide both ways, more than the VIX implies: SPY rose '
              '5%+ within 20 sessions on 31% of these days (17% at the same VIX level) but also dropped 7%+ on 27% (14%). '
              'Average return about the same. Waiting for a mechanical support signal (back above −1 ATR, a higher low, VIX '
              'falling) gave up about half of the upside (Provisional).</div>',
              'Tested'),
    'EASE OFF': ('Worst balance of upside vs downside.',
                 'Fast drops of 7% or more in 1 of 3 cases (all days 1 in 8), more than the VIX alone implies; industries '
                 'move together (5 positions ≈ 1.5 bets). <b>Fewer positions, partials sooner.</b> Puts have roughly broken '
                 'even here (10% OTM 1.2×, ATM 0.9× their cost), but on too few crashes to rely on.', 'Tested'),
    'NORMAL': ('Normal process.',
               '<b>Planned setups, normal planned risk, standard rules.</b>', None),   # + the tail sentence (panel_html)
}
PEDAL_COL = {'PRESS': TEAL, 'NORMAL': '#cfc6bb', 'EASE OFF': ORANGE}


def insurance_sentence(bandg: str, trend: str, inverted: bool) -> str:
    if bandg == 'RISK-OFF' and trend == 'down':
        s = ('<b>Puts have roughly broken even here</b> (10% OTM 1.2×, ATM 0.9× their cost at expiry), but on too few '
             'crashes to rely on.')
    elif bandg == 'RISK-OFF':
        s = ('<b>Protection is expensive for what it has paid.</b> In this state puts returned 5% (10% OTM) to 23% (ATM) '
             'of their cost, vs about a third elsewhere (most episodes ended in a rebound). One crash from this state would '
             'change that. Avoid new protection unless a specific position needs it.')
    else:
        s = ('<b>Protection is cheapest here</b> (about 0.3% of SPY a month). It has not paid better here than elsewhere: '
             'buy it as crash insurance within a ≤1% NLV budget, 3 months out, roll after 1–2 months.')
    if inverted:
        s += (' <span style="color:#f8d6b4;">VIX curve inverted: VXX has paid only in this state (+10% over 21 sessions '
              'on average, though the median month still lost 2%). Outside it, it has lost about 5% a month.</span>')
    return s


def normal_tails(snap: dict) -> str:
    """13 Nov 2026: one factual sentence for NORMAL comparing today's state with all days (next 20 sessions), so the
    pedal line and the Today's state tile read the same way."""
    try:
        c, a = snap['cell']['cell'], snap['cell']['all']
        r = max(c['fast'] / a['fast'], c['dn5'] / a['dn5'])
        w = ('Downside wider than on all days' if r >= 1.3 else 'Downside a little wider than on all days' if r >= 1.1
             else 'Downside narrower than on all days' if r <= 0.77 else 'Downside in line with all days')
        return (f'{w} (fast drop of 7%: {c["fast"] * 100:.0f}% vs {a["fast"] * 100:.0f}%; 5% down: '
                f'{c["dn5"] * 100:.0f}% vs {a["dn5"] * 100:.0f}%).')
    except Exception:
        return ''


def _clock_bars(bucket: str) -> str:
    """C1: mean R and P(>= 3R) of a random stock entry by sessions since SPY's last DIP ZONE day; today's bucket lit."""
    out = ''
    top = max(r for _, r, _ in C1_CLOCK)
    for b, r, p3 in C1_CLOCK:
        on = b == bucket
        col = (TEAL if r >= 0.15 else (INK if r >= 0 else ORANGE)) if on else '#3a3632'
        w = max(abs(r) / top, 0.04) * 100
        out += (f'<div class="row"><span style="width:78px;color:{INK if on else DIM};font-weight:{700 if on else 400};">'
                f'{C1_LBL[b]}</span><div class="trk" style="height:12px;">'
                f'<div style="position:absolute;left:0;top:0;bottom:0;width:{w:.0f}%;background:{col};border-radius:3px;"></div></div>'
                f'<span class="num" style="width:104px;text-align:right;color:{col if on else DIM};font-weight:{700 if on else 400};">'
                f'{r:+.2f}R · {p3 * 100:.0f}%</span></div>')
    return out + ('<div class="sub">Mean R of a random stock entry and its chance of reaching 3R, by sessions since '
                  'SPY\'s last DIP ZONE day (C1, 2009–26). Per trade, not size: adding size in the dip window deepened the '
                  'account drawdown (v8 sizing).</div>')


def panel_html(snap: dict) -> str:
    """v8.2.0: Risk today on the MRS v8 state (no MRS band, no pedal)."""
    v8 = snap.get('v8') or {}
    ep = snap.get('ep_day')
    ext = snap.get('ext_atr')
    h = CSS + ('<div class="rp"><div class="k">Risk today · what the next 20 sessions looked like in this state, '
               'and how many bets the book really holds</div><div class="grid">')
    # 1. SPY vs its 50-day (+ the tested dip-zone numbers when below -1 ATR)
    inz = _fin(ext) and ext <= -1
    ecol = TEAL if inz else INK
    ep_txt = (f'Sell-off episode <b style="color:{INK};">day {ep}</b> (started {pd.Timestamp(snap["ep_start"]):%d %b}). '
              if ep else '')
    dz = ('<div class="act"><b>Dip zone: wide both ways.</b> SPY rose 5%+ within 20 sessions on 31% of these days (17% '
          'at the same VIX) and dropped 7%+ on 27% (14%). Long setups at a level that holds; every stop stands.</div>'
          if inz else
          f'<div class="sub">The dip zone starts at −1 ATR (the rebound window for ticket A).</div>')
    h += (f'<div class="tile"><div class="lab">SPY vs its 50-day {_ev("Tested") if inz else ""}</div>'
          f'<div style="display:flex;align-items:baseline;gap:8px;"><span class="big num" style="color:{ecol};">'
          f'{ext:+.2f}</span><span class="sub">ATR(14)</span></div>{_atr_gauge(ext)}'
          f'<div class="sub">{ep_txt}</div>{dz}</div>')
    # 2. This state (v8 cell)
    if v8.get('cell'):
        c, a = v8['cell'], v8['all']
        vb = v8.get('vb')
        vcol = TEAL if vb == 'CALM' else ORANGE
        state_txt = (f'<b>SPY {PH_TXT.get(v8["code"], "?")}</b> · VIX <b style="color:{vcol};">'
                     f'{(v8["vix"] if _fin(v8["vix"]) else float("nan")):.1f} {_e(vb)}</b>')
        scope = ('' if v8['scope'] == 'cell' else
                 ' <span style="color:#f8d6b4;">(phase only: too few days at this VIX width)</span>')
        fast_c = ORANGE if c['fast'] > a['fast'] * 1.3 else (TEAL if c['fast'] < a['fast'] * 0.7 else INK)
        h += (f'<div class="tile"><div class="lab">This state · SPY next 20 sessions, since 2008 {_ev("Descriptive")}</div>'
              f'<div class="act">{state_txt}{scope}</div>'
              f'{_range_svg(c, a)}{_split_bar(c["up5"], c["dn5"], a["up5"], a["dn5"])}'
              f'<div class="act">Fast drop of 7% or more: <b class="num" style="color:{fast_c};">{c["fast"] * 100:.0f}%</b> '
              f'<span class="sub">of cases (all days {a["fast"] * 100:.0f}%) · {c["n"]:,} days</span></div></div>')
    else:
        h += '<div class="tile"><div class="lab">This state</div><div class="sub">SPY phase not available right now.</div></div>'
    # 3. Clock (C1)
    b = v8.get('bucket')
    if b:
        ns = v8.get('n_since')
        btxt = 'in the dip zone today' if b == 'IN DIP' else (f'{ns} sessions since the last DIP ZONE day' if ns is not None else 'no dip in the history')
        bcol = TEAL if b in ('IN DIP', 'EARLY') else (INK if b == 'MID' else COPPER)
        h += (f'<div class="tile"><div class="lab">Clock · quality of a new trade {_ev("Tested")}</div>'
              f'<div style="display:flex;align-items:baseline;gap:8px;"><span class="big" style="color:{bcol};">{_e(b)}</span>'
              f'<span class="sub">{_e(btxt)}</span></div>{_clock_bars(b)}</div>')
    else:
        h += '<div class="tile"><div class="lab">Clock</div><div class="sub">SPY phase not available right now.</div></div>'
    # 4. Effective bets (unchanged, Tested)
    if _fin(snap['rho']):
        terc = corr_tercile(snap['rho']); stress = CORR_STRESS_P90[terc]; neff_s = 5 / (1 + 4 * stress)
        flag = snap['trend'] == 'down' or snap['inverted']
        nc = TEAL if snap['neff'] >= 2.3 else (ORANGE if snap['neff'] < 1.6 else INK)
        cp = corr_pctile(snap['rho'])
        h += (f'<div class="tile"><div class="lab">Effective bets {_ev("Tested")}</div>'
              f'<div style="display:flex;align-items:baseline;gap:8px;"><span class="big num" style="color:{nc};">{snap["neff"]:.1f}</span>'
              f'<span class="sub">independent bets in 5 positions</span></div>{_dots(snap["neff"])}'
              f'<div class="sub">How much industries move together (63-day): <b class="num" style="color:{INK};">{snap["rho"]:.2f}</b>, '
              f'{_ord(int(round(cp * 100)))} percentile since 2008.</div>'
              f'{_meter(snap["rho"], 0.1, 0.9, {"P10": CORR_Q[2], "median": CORR_Q[10], "P90": CORR_Q[18]}, GRAD)}'
              f'<div class="sub">In a stress month this has dropped to about <b style="color:{INK};">{neff_s:.1f} bets</b>.</div>'
              f'<div class="act"><b>Count names that move together as one position; no more than half of the open '
              f'risk in one sector.</b>'
              + (' <span style="color:#f4a062;"><b>Correlation runs highest when SPY is below its level of a year ago or '
                 'the VIX curve is inverted: treat the book as about 1.5 bets.</b></span>' if flag else '')
              + '</div></div>')
    else:
        h += '<div class="tile"><div class="lab">Effective bets</div><div class="sub">Industry prices not available.</div></div>'
    h += '</div></div>'
    return h.replace('\n', ' ')


# ── data (shared by the app and the nightly forward log) ─────────────────────────
def _norm(s: pd.Series) -> pd.Series:
    s = s.copy()
    idx = pd.to_datetime(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return s[~s.index.duplicated(keep='last')]


def fetch_inputs(etfs, workers: int = 8):
    """SPY daily OHLC (max), ^VIX3M closes (max) and ~6 months of industry-ETF closes, one yf.Ticker call each."""
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    spy = yf.Ticker('SPY').history(period='max', auto_adjust=True)
    spy = pd.DataFrame({c.lower(): _norm(spy[c]) for c in ('Open', 'High', 'Low', 'Close')}) if spy is not None and len(spy) else None
    try:
        v3 = yf.Ticker('^VIX3M').history(period='max')
        v3 = _norm(v3['Close']) if v3 is not None and len(v3) else None
    except Exception:
        v3 = None

    def one(t):
        try:
            h = yf.Ticker(t).history(period='6mo', auto_adjust=True)
            return t, (_norm(h['Close']) if h is not None and len(h) else None)
        except Exception:
            return t, None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(one, list(dict.fromkeys(etfs))))
    closes = pd.DataFrame({t: s for t, s in res.items() if s is not None}).sort_index()
    return spy, v3, closes


def build_snapshot(hist: pd.DataFrame, spy: pd.DataFrame, v3: pd.Series, closes: pd.DataFrame, label_fn,
                   phase: pd.Series = None) -> dict:
    """hist = mrs_history (date, mrs_score, vix). Returns the snapshot for the last SPY session.
    phase (v8.2.0): today's SPY phase series as the MRS card has it (chart_view.spy_phase); the history of the phase for
    the 20-session statistics is replayed here on the full SPY history."""
    h = hist.dropna(subset=['mrs_score']).sort_values('date')
    mrs = pd.Series(h['mrs_score'].astype(float).values, index=pd.to_datetime(h['date']))
    d = spy_states(spy, mrs, label_fn)
    rho, neff, _ = industry_corr(closes.iloc[-100:] if closes is not None else None)
    vix = float(pd.to_numeric(h['vix'], errors='coerce').dropna().iloc[-1]) if 'vix' in h else np.nan
    v3s = v3.dropna() if v3 is not None else pd.Series(dtype=float)
    vix3m = float(v3s.iloc[-1]) if len(v3s) else np.nan
    snap = snapshot(d, rho, neff, vix, vix3m, put_cost_series(v3s[v3s.index >= '2008-01-01']) if len(v3s) else None)
    try:   # v8.2.0: the v8 cell and the clock
        ph_hist = phase_series(spy)
        ph_now = phase if (phase is not None and len(phase)) else ph_hist
        if ph_now is not None and len(ph_now):
            hv = hist.dropna(subset=['vix']).sort_values('date')
            vix_hist = pd.Series(pd.to_numeric(hv['vix'], errors='coerce').values, index=pd.to_datetime(hv['date']).dt.normalize())
            vix_hist = vix_hist[~vix_hist.index.duplicated(keep='last')]
            v8 = v8_cell(d, ph_hist if ph_hist is not None else ph_now, vix_hist, int(ph_now.iloc[-1]), vix)
            v8['bucket'], v8['n_since'] = clock_of(ph_now)
            try:   # 8.6.0: the 200-day filter picks the clock table (ON: close > SMA200 and SMA200 up over 20 sessions)
                _c = spy.rename(columns=str.lower)['close'].astype(float).dropna()
                _s = _c.rolling(200).mean()
                v8['f200'] = bool(_c.iloc[-1] > _s.iloc[-1] and _s.iloc[-1] > _s.iloc[-21])
            except Exception:
                v8['f200'] = None
            snap['v8'] = v8
    except Exception as _e8:
        print(f'[risk panel v8] {_e8}')
    return snap



# ══ 8.4.0 (16 Nov 2026): Risk today, unified components; "This state" as the distribution itself + the odds bars ══
def _kde(x, grid, bw=0.008):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return np.zeros_like(grid)
    return np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(1) / (len(x) * bw * math.sqrt(2 * math.pi))


def density_svg(f_cell, f_all, W=520, H=170, src='since 2008') -> str:
    """SPY next-20-session returns: this state (orange) vs all days (grey), with this state's P10 / P90."""
    grid = np.linspace(-0.16, 0.14, 240)
    a, c = _kde(f_all, grid), _kde(f_cell, grid)
    top = max(a.max(), c.max()) * 1.08 or 1
    L, R, Tp, B = 6, 6, 16, 22
    X = lambda v: L + (v - grid[0]) / (grid[-1] - grid[0]) * (W - L - R)
    Y = lambda d: Tp + (1 - d / top) * (H - Tp - B)
    def path(d):
        pts = ' '.join(f'{X(g):.1f},{Y(v):.1f}' for g, v in zip(grid, d))
        return f'M{X(grid[0]):.1f},{Y(0):.1f} L{pts} L{X(grid[-1]):.1f},{Y(0):.1f} Z'
    q10, q90 = (float(np.nanquantile(f_cell, q)) for q in (0.1, 0.9)) if len(f_cell) else (np.nan, np.nan)
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="none" style="display:block">']
    for t in (-0.10, -0.05, 0.0, 0.05, 0.10):
        g.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{Tp}" y2="{H - B}" stroke="{"#3a3632" if t == 0 else "#24211e"}"/>'
                 f'<text x="{X(t):.1f}" y="{H - 6}" text-anchor="middle" fill="#6e675f" font-size="10" font-family="IBM Plex Mono, monospace">{t * 100:+.0f}%</text>')
    g.append(f'<path d="{path(a)}" fill="rgba(155,149,143,0.16)" stroke="#9b958f" stroke-width="1.3"/>')
    g.append(f'<path d="{path(c)}" fill="rgba(244,160,98,0.22)" stroke="#f4a062" stroke-width="2"/>')
    for q, lbl in ((q10, 'P10'), (q90, 'P90')):
        if np.isfinite(q):
            g.append(f'<line x1="{X(q):.1f}" x2="{X(q):.1f}" y1="{Tp}" y2="{H - B}" stroke="#f4a062" stroke-dasharray="2,3"/>'
                     f'<text x="{X(q):.1f}" y="{Tp - 4}" text-anchor="middle" fill="#f4a062" font-size="10.5" font-family="IBM Plex Mono, monospace">{lbl} {q * 100:+.1f}%</text>')
    g.append('</svg>')
    key = ('<div class="atn-sm" style="font-size:11px"><span style="color:#f4a062">━</span> this state · '
           f'<span style="color:#9b958f">━</span> all days, {src}</div>')
    return ''.join(g) + key


def ext_spark_svg(xs, W=420, H=110) -> str:
    """8.4.1: SPY distance from its 50-day in ATR(14), last 60 sessions, with the DIP ZONE (<= -1) shaded."""
    x = np.asarray([v for v in (xs or []) if _fin(v)], float)
    if len(x) < 5:
        return ''
    lo, hi = min(x.min(), -1.6) - 0.2, max(x.max(), 1.0) + 0.2
    L, R, Tp, B = 4, 30, 6, 14
    X = lambda i: L + i / (len(x) - 1) * (W - L - R)
    Y = lambda v: Tp + (hi - v) / (hi - lo) * (H - Tp - B)
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block">',
         f'<rect x="{L}" y="{Y(-1):.1f}" width="{W - L - R}" height="{H - B - Y(-1):.1f}" fill="rgba(133,202,197,0.16)"/>']
    for v, lab in ((0, '50d'), (-1, '−1')):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{"#3a3632" if v == 0 else "#649995"}" '
                 f'stroke-dasharray="{"" if v == 0 else "3,3"}"/><text x="{W - R + 4}" y="{Y(v) + 3.5:.1f}" fill="#6e675f" '
                 f'font-size="10" font-family="IBM Plex Mono, monospace">{lab}</text>')
    pts = ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(x))
    g.append(f'<polyline points="{pts}" fill="none" stroke="#f3ece2" stroke-width="1.6"/>')
    for i, v in enumerate(x):
        if v <= -1:
            g.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2" fill="#85cac5"/>')
    g.append(f'<circle cx="{X(len(x) - 1):.1f}" cy="{Y(x[-1]):.1f}" r="3.5" fill="#f3ece2"/>'
             f'<text x="{L}" y="{H - 2}" fill="#6e675f" font-size="10" font-family="IBM Plex Mono, monospace">60 sessions</text></svg>')
    return ''.join(g)


def _odds(lbl, v, va, col):
    import ui
    return (f'<div style="display:flex;flex-direction:column;font-size:12px;color:#a89f95">'
            f'<span>{lbl}</span><div>'
            + ui.bar([(0, min(v / 0.4, 1) * 100, col)], min(va / 0.4, 1) * 100, margin='8px 0')
            + f'</div><span style="font-family:IBM Plex Mono,monospace;color:{col};font-size:12.5px">{v * 100:.0f}% '
            f'<span style="color:#6e675f">vs {va * 100:.0f}%</span></span></div>')


def panel_v9_html(snap: dict) -> str:
    import ui
    v8 = snap.get('v8') or {}
    ext, ep = snap.get('ext_atr'), snap.get('ep_day')
    inz = _fin(ext) and ext <= -1
    lo_, hi_ = -3.0, 3.0
    zx = (-1 - lo_) / (hi_ - lo_) * 100
    t1 = (f'<div class="atn-tile"><div class="atn-tl">SPY vs its 50-day {ui.tag("Tested") if inz else ""}</div>'
          f'<div class="atn-val" style="color:{TEAL if inz else INK}">{ext:+.2f} <span class="atn-vs">ATR(14)</span></div>'
          + ui.bar([(0, zx, 'rgba(133,202,197,0.35)')], (min(max(ext, lo_), hi_) - lo_) / (hi_ - lo_) * 100 if _fin(ext) else None,
                   [(zx / 2, 'dip zone'), (50, '50-day'), (94, '+3')])
          + (f'<div class="atn-sm">sell-off day <b style="color:#f3ece2">{ep}</b> · started {pd.Timestamp(snap["ep_start"]):%d %b}</div>' if ep else '')
          + ((f'<div class="atn-sm" style="margin-top:6px">Dip zone, {v8["src"]}: wide both ways. After 20 sessions '
              f'ended 5%+ up {v8["dz"]["up5"] * 100:.0f}% vs {v8["all"]["up5"] * 100:.0f}%; fast 7% drop '
              f'{v8["dz"]["fast"] * 100:.0f}% vs {v8["all"]["fast"] * 100:.0f}%.</div>') if inz and v8.get('dz') and v8['dz']['n']
             else '<div class="atn-sm" style="margin-top:6px">Dip zone: wide both ways.</div>' if inz
             else '<div class="atn-sm" style="margin-top:6px">The dip zone starts at −1 ATR.</div>')
          + ext_spark_svg(snap.get('ext_hist')) + '</div>')
    if v8.get('cell'):
        c, a = v8['cell'], v8['all']
        vb = v8.get('vb'); vcol = TEAL if vb == 'CALM' else ORANGE
        scope = '' if v8['scope'] == 'cell' else ' <span style="color:#f8d6b4">(phase only)</span>'
        t2 = (f'<div class="atn-tile"><div class="atn-tl">This state · next 20 sessions, {_e(v8.get("src", "SPY since 2008"))} {ui.tag("Descriptive")}</div>'
              f'<div class="atn-sm" style="margin-bottom:2px"><b style="color:#f3ece2">SPY {PH_TXT.get(v8["code"], "?")}</b> · VIX '
              f'<b style="color:{vcol}">{(v8["vix"] if _fin(v8["vix"]) else float("nan")):.1f} {_e(vb)}</b>{scope} · {c["n"]:,} days</div>'
              f'{density_svg(v8.get("f_cell", []), v8.get("f_all", []), W=620, H=170, src=v8.get("src", "since 2008"))}'
              f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px 14px;margin-top:6px">'
              + _odds('ended ≥5% up', c['up5'], a['up5'], TEAL) + _odds('ended ≥5% down', c['dn5'], a['dn5'], ORANGE)
              + _odds('7% fast drop', c['fast'], a['fast'], ORANGE)
              + '</div><div class="atn-sm" style="font-size:11px;margin-top:2px">bars: share of days in this state · ended = where it '
              'closed 20 sessions later; fast drop = at any point in the 20 sessions · white pin = all days</div></div>')
    else:
        t2 = '<div class="atn-tile"><div class="atn-tl">This state</div><div class="atn-sm">SPY phase not available right now.</div></div>'
    b = v8.get('bucket')
    if b:
        rows = ''
        f_on = v8.get('f200') is not False
        for bb, r, p3 in (C1_CLOCK if f_on else C1_CLOCK_OFF):
            on = bb == b
            if r is None:
                rows += (f'<div style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:{INK if on else DIM}">'
                         f'<span style="width:64px;font-weight:{700 if on else 400}">{C1_LBL[bb]}</span><div style="flex:1"></div>'
                         f'<span style="width:96px;text-align:right;color:{DIM}">no data</span></div>')
                continue
            col = (TEAL if r >= 0.15 else (INK if r >= 0 else ORANGE)) if on else '#3a3632'
            rows += (f'<div style="display:flex;align-items:center;gap:8px;font-size:12.5px;color:{INK if on else DIM}">'
                     f'<span style="width:64px;font-weight:{700 if on else 400}">{C1_LBL[bb]}</span><div style="flex:1">'
                     + ui.bar([(0, min(max(abs(r), 0.02) / 0.30, 1) * 100, col)], margin='7px 0', height=12)
                     + f'</div><span style="width:96px;text-align:right;font-family:IBM Plex Mono,monospace;color:{col if on else DIM}">'
                     f'{r:+.2f}R · {p3 * 100:.0f}%</span></div>')
        t3 = (f'<div class="atn-tile"><div class="atn-tl">Clock · quality of a new trade {ui.tag("Descriptive · 2001–26")}</div>'
              f'<div class="atn-val" style="color:{TEAL if b in ("IN DIP", "EARLY") else (INK if b == "MID" else COPPER)}">{_e(b)}'
              f'{"" if not v8.get("n_since") else " " + str(v8["n_since"])}</div>'
              f'<div style="display:flex;flex-direction:column;justify-content:space-evenly;flex:1;min-height:150px">{rows}</div>'
              + ('<div class="atn-sm" style="font-size:11px;margin-top:4px">mean R · P(≥3R) of a random stock entry, 2001–26, 200-day filter '
                 + ('ON (same sign in 2001–08 and 2009–26)' if f_on else 'OFF: the dip is not an edge') + '. Per trade, not size.</div></div>'))
    else:
        t3 = '<div class="atn-tile"><div class="atn-tl">Clock</div><div class="atn-sm">not available right now.</div></div>'
    if _fin(snap.get('rho')):
        terc = corr_tercile(snap['rho']); neff_s = 5 / (1 + 4 * CORR_STRESS_P90[terc])
        nc = TEAL if snap['neff'] >= 2.3 else (ORANGE if snap['neff'] < 1.6 else INK)
        flag = snap['trend'] == 'down' or snap['inverted']
        t4 = (f'<div class="atn-tile"><div class="atn-tl">Effective bets {ui.tag("Tested")}</div>'
              f'<div class="atn-val" style="color:{nc}">{snap["neff"]:.1f} <span class="atn-vs">of 5 positions</span></div>'
              f'<div style="margin:6px 0">{_dots(snap["neff"])}</div>'
              f'<div class="atn-sm">industries move together: <b style="font-family:IBM Plex Mono,monospace;color:#f3ece2">{snap["rho"]:.2f}</b> '
              f'({_ord(int(round(corr_pctile(snap["rho"]) * 100)))} pct) · stress month ≈ {neff_s:.1f}</div>'
              f'<div class="atn-sm" style="color:#f3ece2;margin-top:4px">Max half of the open risk in one sector.'
              + (' <b style="color:#f4a062">Treat the book as about 1.5 bets today.</b>' if flag else '') + '</div></div>')
    else:
        t4 = '<div class="atn-tile"><div class="atn-tl">Effective bets</div><div class="atn-sm">Industry prices not available.</div></div>'
    return ('<div style="background:#121212;border:1px solid #2a2724;border-radius:6px;padding:14px 16px 12px 16px;margin:2px 0 12px 0">'
            + ui.kicker('Risk today', 'what the next 20 sessions looked like in this state · how many bets the book holds')
            + '<div style="display:grid;grid-template-columns:1.05fr 2.1fr 1.15fr;gap:10px;align-items:stretch">'
            + t2.replace('<div class="atn-tile">', '<div class="atn-tile" style="grid-column:2;grid-row:1 / span 2">', 1)
            + t1.replace('<div class="atn-tile">', '<div class="atn-tile" style="grid-column:1;grid-row:1">', 1)
            + t4.replace('<div class="atn-tile">', '<div class="atn-tile" style="grid-column:1;grid-row:2">', 1)
            + t3.replace('<div class="atn-tile">', '<div class="atn-tile" style="grid-column:3;grid-row:1 / span 2;display:flex;flex-direction:column">', 1)
            + '</div></div>').replace('\n', ' ')
