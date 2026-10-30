"""
risk_panel.py — "Risk today" panel (MRS v5, 30 Oct 2026)
==========================================================
Desk lines built on studies/mrs_risk_control (steps 1-4) and studies/model_portfolio. Nothing here changes the score.

  Pedal           PRESS: SPY > 1 ATR below its 50-day SMA (upside beyond the VIX level: Tested)
                  EASE OFF: RISK-OFF and SPY below its close of a year ago (fast drops 1 in 3; Tested vs other days)
                  NORMAL: everything else
  Today's cell    20-session distribution of SPY since May 2008 for today's state
                  (band group x 12-month trend x position vs the 50-day). Descriptive.
  Effective bets  trailing 63-session average correlation of the industry ETFs -> 5 / (1 + 4 rho).
                  Trailing tracks the next 20 sessions (Spearman 0.53, both halves). Descriptive.
  Insurance cost  3-month SPY put, 10% OTM, Black-Scholes on VIX3M with the study's skew surcharge (k = 1.5).
                  Cost per month and its percentile since 2008; state sentence from step 3. Descriptive.
  Vol guide       invested share for a 10% volatility target on 21-session realised SPY vol, cap 60%. Descriptive.

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
    atr = tr.rolling(14).mean()
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


def snapshot(d: pd.DataFrame, rho: float, neff: float, vix: float, vix3m: float, cost_hist: pd.Series) -> dict:
    """Everything the panel and the forward log need, for the last session in d."""
    last = d.dropna(subset=['bandg', 'ext']).iloc[-1]
    bg, tr, ex = last['bandg'], last['trend'], last['ext']
    cs = cell_stats(d, bg, tr, ex)
    cost = put_cost_per_month(vix3m)
    pct = float((cost_hist <= cost).mean()) if _fin(cost) and cost_hist is not None and len(cost_hist) else np.nan
    rv, inv = vol_guide(d['close'])
    return dict(date=d.index[-1], band=last['band'], bandg=bg, trend=tr, ext=ex, ext_atr=float(last['ext_atr']),
                r12=float(last['r12']) if _fin(last.get('r12')) else np.nan,
                pedal=pedal(bg, tr, ex), cell=cs, rho=rho, neff=neff, inverted=bool(_fin(vix) and _fin(vix3m) and vix > vix3m),
                vix=vix, vix3m=vix3m, cost=cost, cost_pct=pct, rv=rv, invested=inv)


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


def _ev(kind):
    c = {'Tested': COPPER, 'Descriptive': MUTED, 'Watch': '#f8d6b4'}[kind]
    bg = 'rgba(227,180,142,0.14)' if kind == 'Tested' else 'rgba(0,0,0,0)'
    return f'<span class="ev" style="color:{c};border-color:{c};background:{bg};">{kind.upper()}</span>'


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
    """SPY vs its 50-day in ATR; the PRESS zone (below -1 ATR) shaded."""
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
    g += (f'<span style="position:absolute;left:{z / 2:.1f}%;transform:translateX(-50%);top:2px;color:{TEAL};">PRESS zone</span>'
          f'<span style="position:absolute;left:{_x(0, lo, hi):.1f}%;transform:translateX(-50%);top:2px;">50-day</span>'
          f'<span style="position:absolute;right:0;top:2px;">+3 ATR</span>')
    return g + '</div>'


PEDAL_TXT = {
    'PRESS': ('More upside than the VIX implies.',
              'After sell-offs like this, SPY gained 5% or more within 20 sessions 38–41% of the time, vs about 20% '
              'expected from the VIX level. <b>Take your A-setups at full planned risk.</b> Fast drops of 7% or more still '
              'came in 28% of cases: <b>every stop stands.</b>', 'Tested'),
    'EASE OFF': ('Worst balance of upside vs downside.',
                 'Fast drops of 7% or more in 1 of 3 cases; industries move together (5 positions ≈ 1.5 bets). '
                 '<b>Fewer positions, partials sooner.</b> Protection has roughly paid for itself here.', 'Tested'),
    'NORMAL': ('No edge from the state in either direction.',
               'Normal process: <b>A-setups, 1% risk, your rules.</b>', None),
}
PEDAL_COL = {'PRESS': TEAL, 'NORMAL': '#cfc6bb', 'EASE OFF': ORANGE}


def insurance_sentence(bandg: str, trend: str, inverted: bool) -> str:
    if bandg == 'RISK-OFF' and trend == 'down':
        s = '<b>Protection has roughly paid for itself here</b> (0.9–1.2× its cost at expiry).'
    elif bandg == 'RISK-OFF':
        s = ('<b>Protection is expensive for what it has paid.</b> In this state puts returned only 5–23% of their cost '
             '(most episodes ended in a rebound). Avoid new protection unless a specific position needs it.')
    else:
        s = ('<b>Protection is cheap.</b> If you want insurance, this is when to buy it: 3 months out, within your ≤1% NLV '
             'hedge budget, roll after 1–2 months.')
    if inverted:
        s += ' <span style="color:#f8d6b4;">VIX curve inverted: VXX has paid only in this state (+10% a month on average).</span>'
    return s


def panel_html(snap: dict) -> str:
    p = snap['pedal']; col = PEDAL_COL[p]
    head, body, evk = PEDAL_TXT[p]
    segs = ''.join(f'<div style="background:{PEDAL_COL[k] if k == p else "#24211e"};"></div>' for k in ('EASE OFF', 'NORMAL', 'PRESS'))
    seg_lbl = ''.join(f'<span style="flex:1;text-align:center;color:{PEDAL_COL[k] if k == p else DIM};font-weight:{700 if k == p else 400};">{k}</span>'
                      for k in ('EASE OFF', 'NORMAL', 'PRESS'))
    h = CSS + '<div class="rp"><div class="k">Risk today · how much risk the book can carry</div><div class="grid">'
    # 1. Pedal
    r12 = snap.get('r12')
    ease_txt = (f'EASE OFF needs RISK-OFF <b>and</b> SPY below its level of a year ago '
                f'(now {_pct(r12)} over 12 months).' if _fin(r12) else 'EASE OFF needs RISK-OFF and SPY below its level of a year ago.')
    h += (f'<div class="tile"><div class="lab">Pedal {_ev(evk) if evk else ""}</div>'
          f'<div class="big" style="color:{col};">{p}</div>'
          f'<div class="seg">{segs}</div><div style="display:flex;font-size:0.66rem;letter-spacing:0.05em;margin-top:-4px;">{seg_lbl}</div>'
          f'<div class="act"><b>{head}</b> {body}</div>'
          f'<div class="sub" style="margin-top:2px;">SPY vs its 50-day: <b class="num" style="color:{INK};">{snap["ext_atr"]:+.2f} ATR</b>. '
          f'PRESS below −1 ATR.</div>{_atr_gauge(snap["ext_atr"])}'
          f'<div class="sub">{ease_txt}</div></div>')
    # 2. Today's state
    cs = snap['cell']; c, a = cs['cell'], cs['all']
    state_txt = (f'<b>{_e(snap["band"])}</b> · SPY 12-month {snap["trend"]} · {EXT_TXT[snap["ext"]]}')
    scope = '' if cs['scope'] == 'cell' else ' <span style="color:#f8d6b4;">(band only: too few days in the exact state)</span>'
    fast_c = ORANGE if c['fast'] > a['fast'] * 1.3 else (TEAL if c['fast'] < a['fast'] * 0.7 else INK)
    h += (f'<div class="tile"><div class="lab">Today\'s state · next 20 sessions, since 2008 {_ev("Descriptive")}</div>'
          f'<div class="act">{state_txt}{scope}</div>'
          f'{_range_svg(c, a)}{_split_bar(c["up5"], c["dn5"], a["up5"], a["dn5"])}'
          f'<div class="act">Fast drop of 7% or more: <b class="num" style="color:{fast_c};">{c["fast"] * 100:.0f}%</b> '
          f'<span class="sub">of cases (all days {a["fast"] * 100:.0f}%) · {c["n"]:,} days</span></div></div>')
    # 3. Effective bets
    if _fin(snap['rho']):
        terc = corr_tercile(snap['rho']); stress = CORR_STRESS_P90[terc]; neff_s = 5 / (1 + 4 * stress)
        flag = snap['trend'] == 'down' or snap['inverted']
        nc = TEAL if snap['neff'] >= 2.3 else (ORANGE if snap['neff'] < 1.6 else INK)
        cp = corr_pctile(snap['rho'])
        h += (f'<div class="tile"><div class="lab">Effective bets {_ev("Descriptive")}</div>'
              f'<div style="display:flex;align-items:baseline;gap:8px;"><span class="big num" style="color:{nc};">{snap["neff"]:.1f}</span>'
              f'<span class="sub">independent bets in 5 positions</span></div>{_dots(snap["neff"])}'
              f'<div class="sub">How much industries move together (63-day): <b class="num" style="color:{INK};">{snap["rho"]:.2f}</b>, '
              f'{_ord(int(round(cp * 100)))} percentile since 2008.</div>'
              f'{_meter(snap["rho"], 0.1, 0.9, {"P10": CORR_Q[2], "median": CORR_Q[10], "P90": CORR_Q[18]}, GRAD)}'
              f'<div class="sub">In a stress month this has dropped to about <b style="color:{INK};">{neff_s:.1f} bets</b>.</div>'
              f'<div class="act"><b>Count names that move together as one position; no more than half of your open '
              f'risk in one sector.</b>'
              + (' <span style="color:#f4a062;"><b>Correlation runs highest in this state: treat the book as about 1.5 bets.</b></span>' if flag else '')
              + '</div></div>')
    else:
        h += f'<div class="tile"><div class="lab">Effective bets</div><div class="sub">Industry prices not available.</div></div>'
    # 4. Insurance cost
    if _fin(snap['cost']):
        pc = snap['cost_pct']
        h += (f'<div class="tile"><div class="lab">Insurance cost {_ev("Descriptive")}</div>'
              f'<div style="display:flex;align-items:baseline;gap:8px;"><span class="big num" style="color:{INK};">{snap["cost"] * 100:.2f}%</span>'
              f'<span class="sub">of SPY per month</span></div>'
              f'<div class="sub">3-month SPY put, 10% out of the money · VIX3M {snap["vix3m"]:.1f} · '
              f'{"—" if not _fin(pc) else _ord(int(round(pc * 100))) + " percentile since 2008"}</div>'
              f'{_meter(pc, 0, 1, {"cheap": 0.2, "median": 0.5, "dear": 0.8}, GRAD)}'
              f'<div class="act">{insurance_sentence(snap["bandg"], snap["trend"], snap["inverted"])}</div></div>')
    else:
        h += '<div class="tile"><div class="lab">Insurance cost</div><div class="sub">VIX3M not available.</div></div>'
    h += '</div>'
    # 5. Vol guide (collapsed by default)
    inv = snap['invested']
    if _fin(inv) and _fin(snap['rv']):
        h += (f'<details class="vg"><summary class="lab" style="display:list-item;cursor:pointer;">Book volatility guide '
              f'· {inv * 100:.0f}% invested {_ev("Descriptive")}</summary>'
              f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:8px;">'
              f'<div class="trk" style="flex:0 0 260px;">'
              f'<div style="position:absolute;left:0;top:0;bottom:0;width:{inv * 100:.0f}%;background:{TEAL_M};border-radius:3px;"></div>'
              f'<div title="cap 60%" style="position:absolute;left:60%;top:-4px;bottom:-4px;width:3px;background:{COPPER};"></div></div>'
              f'<span class="sub">SPY realized volatility (21 days) <b class="num" style="color:{INK};">{snap["rv"] * 100:.1f}%</b>. '
              f'A 10% volatility target would hold <b class="num" style="color:{INK};">{inv * 100:.0f}% invested</b> (cap 60%, copper line). '
              f'Reviewed weekly. In the 2008–2026 replay, weekly volatility scaling had a 10–13 pp smaller max drawdown at a '
              f'similar or better Sharpe than holding the same average exposure.</span></div></details>')
    h += '</div>'
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
    spy = pd.DataFrame({c.lower(): _norm(spy[c]) for c in ('High', 'Low', 'Close')}) if spy is not None and len(spy) else None
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


def build_snapshot(hist: pd.DataFrame, spy: pd.DataFrame, v3: pd.Series, closes: pd.DataFrame, label_fn) -> dict:
    """hist = mrs_history (date, mrs_score, vix). Returns the snapshot for the last SPY session."""
    h = hist.dropna(subset=['mrs_score']).sort_values('date')
    mrs = pd.Series(h['mrs_score'].astype(float).values, index=pd.to_datetime(h['date']))
    d = spy_states(spy, mrs, label_fn)
    rho, neff, _ = industry_corr(closes.iloc[-100:] if closes is not None else None)
    vix = float(pd.to_numeric(h['vix'], errors='coerce').dropna().iloc[-1]) if 'vix' in h else np.nan
    v3s = v3.dropna() if v3 is not None else pd.Series(dtype=float)
    vix3m = float(v3s.iloc[-1]) if len(v3s) else np.nan
    return snapshot(d, rho, neff, vix, vix3m, put_cost_series(v3s[v3s.index >= '2008-01-01']) if len(v3s) else None)
