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
EXT_TXT = {'below': 'more than 1 ATR below the 50-day', 'near': 'near the 50-day (within 1 ATR)', 'above': 'more than 1 ATR above the 50-day'}
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
    d['trend'] = np.where(c / c.shift(251) - 1 >= 0, 'up', 'down')
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
    '.rp{background:#121212;border:1px solid #2a2724;border-left:3px solid #e3b48e;border-radius:6px;padding:12px 16px 10px 16px;margin:2px 0 12px 0;}'
    '.rp .k{font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#e3b48e;font-weight:700;margin-bottom:8px;}'
    '.rp .grid{display:grid;grid-template-columns:1.05fr 1.35fr 1fr 1fr;gap:12px;}'
    '.rp .tile{background:#0d0d0d;border:1px solid #1f1d1b;border-radius:5px;padding:10px 12px;display:flex;flex-direction:column;gap:6px;}'
    '.rp .lab{font-size:0.64rem;letter-spacing:0.14em;text-transform:uppercase;color:#a89f95;display:flex;justify-content:space-between;align-items:center;gap:6px;}'
    '.rp .big{font-size:1.5rem;font-weight:800;line-height:1.05;}'
    '.rp .act{font-size:0.80rem;color:#f3ece2;line-height:1.45;}'
    '.rp .sub{font-size:0.72rem;color:#a89f95;line-height:1.45;}'
    '.rp .num{font-family:"IBM Plex Mono",monospace;}'
    '.rp .ev{display:inline-block;border:1px solid;border-radius:3px;padding:0 5px;font-size:0.60rem;font-weight:700;letter-spacing:0.04em;white-space:nowrap;line-height:1.5;}'
    '.rp .seg{display:flex;gap:3px;margin:2px 0 2px 0;}'
    '.rp .seg div{flex:1;height:7px;border-radius:2px;background:#1f1d1b;}'
    '.rp .vg{margin-top:8px;padding-top:8px;border-top:1px solid #1f1d1b;}'
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


def _range_svg(cell: dict, allx: dict, lo=-0.12, hi=0.12, w=300) -> str:
    """Two horizontal P10-P90 bars on a -12%..+12% axis: all days (grey) and today's cell (colour), median ticks."""
    def x(v):
        return 8 + (min(max(v, lo), hi) - lo) / (hi - lo) * (w - 16)
    def bar(st, y, col, op):
        return (f'<rect x="{x(st["p10"]):.1f}" y="{y}" width="{max(x(st["p90"]) - x(st["p10"]), 2):.1f}" height="9" rx="2" fill="{col}" opacity="{op}"/>'
                f'<rect x="{x(st["p50"]) - 1:.1f}" y="{y - 2}" width="2" height="13" fill="{INK}"/>')
    g = ''
    for t in (-0.10, -0.05, 0.0, 0.05, 0.10):
        g += (f'<line x1="{x(t):.1f}" y1="6" x2="{x(t):.1f}" y2="44" stroke="{LINE if t else "#4a4540"}" stroke-width="1"/>'
              f'<text x="{x(t):.1f}" y="56" fill="{DIM}" font-size="9" text-anchor="middle">{int(t * 100):+d}%</text>')
    g += bar(allx, 10, '#5b5148', 0.9) + bar(cell, 28, COPPER, 0.85)
    g += (f'<text x="{x(cell["p10"]) - 3:.1f}" y="36" fill="{ORANGE}" font-size="9.5" text-anchor="end">{_pct(cell["p10"])}</text>'
          f'<text x="{x(cell["p90"]) + 3:.1f}" y="36" fill="{TEAL}" font-size="9.5">{_pct(cell["p90"])}</text>')
    return (f'<div class="act">Bad case <b class="num" style="color:{ORANGE};">{_pct(cell["p10"])}</b>, good case '
            f'<b class="num" style="color:{TEAL};">{_pct(cell["p90"])}</b> <span class="sub">(all days {_pct(allx["p10"])} / '
            f'{_pct(allx["p90"])})</span></div>'
            f'<svg viewBox="0 0 {w} 60" width="100%" style="display:block;max-height:80px;">{g}</svg>'
            f'<div class="sub"><span style="color:#8a7f73;">■</span> all days &nbsp; <span style="color:{COPPER};">■</span> today\'s state'
            f' &nbsp;·&nbsp; bars = bad case (P10) to good case (P90), tick = median</div>')


def _split_bar(up: float, dn: float, up_all: float, dn_all: float) -> str:
    """Odds of a ≥5% move up vs down within 20 sessions, as two bars scaled to 40%."""
    def row(lbl, v, va, col):
        wv = min(v / 0.4, 1) * 100; wa = min(va / 0.4, 1) * 100
        return (f'<div style="display:flex;align-items:center;gap:6px;font-size:0.72rem;color:{MUTED};">'
                f'<span style="width:62px;">{lbl}</span>'
                f'<div style="flex:1;position:relative;height:9px;background:#1a1918;border-radius:2px;">'
                f'<div style="position:absolute;left:0;top:0;bottom:0;width:{wv:.0f}%;background:{col};border-radius:2px;"></div>'
                f'<div title="all days" style="position:absolute;left:{wa:.0f}%;top:-2px;bottom:-2px;width:2px;background:{INK};opacity:.7;"></div></div>'
                f'<span class="num" style="width:34px;text-align:right;color:{col};">{v * 100:.0f}%</span></div>')
    return row('≥5% up', up, up_all, TEAL) + row('≥5% down', dn, dn_all, ORANGE)


def _dots(neff: float, total: int = 5) -> str:
    """Five position dots; the filled share = independent bets."""
    out = ''
    for i in range(total):
        f = min(max(neff - i, 0), 1)
        out += (f'<svg width="20" height="20" viewBox="0 0 20 20" style="margin-right:3px;">'
                f'<circle cx="10" cy="10" r="8" fill="none" stroke="{TEAL_M}" stroke-width="1.5"/>'
                f'<clipPath id="c{i}"><rect x="0" y="0" width="{20 * f:.1f}" height="20"/></clipPath>'
                f'<circle cx="10" cy="10" r="8" fill="{TEAL}" clip-path="url(#c{i})"/></svg>')
    return f'<div style="display:flex;align-items:center;">{out}</div>'


def _meter(v: float, lo: float, hi: float, marks: dict, grad: str, w=200) -> str:
    """Horizontal gradient meter with a marker at v and labelled ticks."""
    def x(z):
        return 4 + (min(max(z, lo), hi) - lo) / (hi - lo) * (w - 8)
    g = f'<rect x="4" y="8" width="{w - 8}" height="8" rx="3" fill="url(#gr{id(marks)})"/>'
    g += f'<defs><linearGradient id="gr{id(marks)}" x1="0" x2="1" y1="0" y2="0">{grad}</linearGradient></defs>'
    for lbl, z in marks.items():
        g += (f'<line x1="{x(z):.1f}" y1="5" x2="{x(z):.1f}" y2="19" stroke="{MUTED}" stroke-width="1" stroke-dasharray="2,2"/>'
              f'<text x="{x(z):.1f}" y="30" fill="{DIM}" font-size="8.5" text-anchor="middle">{_e(lbl)}</text>')
    if _fin(v):
        g += (f'<path d="M{x(v) - 5:.1f},1 L{x(v) + 5:.1f},1 L{x(v):.1f},8 Z" fill="{INK}"/>'
              f'<rect x="{x(v) - 1:.1f}" y="7" width="2" height="11" fill="{INK}"/>')
    return f'<svg viewBox="0 0 {w} 34" width="100%" height="34" style="display:block;">{g}</svg>'


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
    segs = ''.join(f'<div style="background:{PEDAL_COL[k] if k == p else "#1f1d1b"};"></div>' for k in ('EASE OFF', 'NORMAL', 'PRESS'))
    seg_lbl = ''.join(f'<span style="flex:1;text-align:center;color:{PEDAL_COL[k] if k == p else DIM};">{k}</span>'
                      for k in ('EASE OFF', 'NORMAL', 'PRESS'))
    h = CSS + '<div class="rp"><div class="k">Risk today · how much risk the book can carry</div><div class="grid">'
    # 1. Pedal
    h += (f'<div class="tile"><div class="lab">Pedal {_ev(evk) if evk else ""}</div>'
          f'<div class="big" style="color:{col};">{p}</div>'
          f'<div class="seg">{segs}</div><div style="display:flex;font-size:0.58rem;letter-spacing:0.06em;">{seg_lbl}</div>'
          f'<div class="act"><b>{head}</b> {body}</div></div>')
    # 2. Today's cell
    cs = snap['cell']; c, a = cs['cell'], cs['all']
    state_txt = (f'{_e(snap["band"])} · SPY 12-month {snap["trend"]} · {EXT_TXT[snap["ext"]]} '
                 f'({snap["ext_atr"]:+.2f} ATR)')
    scope = '' if cs['scope'] == 'cell' else ' <span style="color:#f8d6b4;">(band only: too few days in the exact cell)</span>'
    fast_c = ORANGE if c['fast'] > a['fast'] * 1.3 else (TEAL if c['fast'] < a['fast'] * 0.7 else INK)
    h += (f'<div class="tile"><div class="lab">Today\'s state · next 20 sessions since 2008 {_ev("Descriptive")}</div>'
          f'<div class="act">{state_txt}{scope}</div>'
          f'{_range_svg(c, a)}{_split_bar(c["up5"], c["dn5"], a["up5"], a["dn5"])}'
          f'<div class="sub">Fast drop of 7% or more: <b class="num" style="color:{fast_c};">{c["fast"] * 100:.0f}%</b> of cases '
          f'(all days {a["fast"] * 100:.0f}%) · {c["n"]:,} days</div></div>')
    # 3. Effective bets
    if _fin(snap['rho']):
        terc = corr_tercile(snap['rho']); stress = CORR_STRESS_P90[terc]; neff_s = 5 / (1 + 4 * stress)
        flag = snap['trend'] == 'down' or snap['inverted']
        nc = TEAL if snap['neff'] >= 2.3 else (ORANGE if snap['neff'] < 1.6 else INK)
        cp = corr_pctile(snap['rho'])
        grad = f'<stop offset="0" stop-color="{TEAL}"/><stop offset="0.5" stop-color="#5b5148"/><stop offset="1" stop-color="{ORANGE}"/>'
        h += (f'<div class="tile"><div class="lab">Effective bets {_ev("Descriptive")}</div>'
              f'<div style="display:flex;align-items:baseline;gap:8px;"><span class="big num" style="color:{nc};">{snap["neff"]:.1f}</span>'
              f'<span class="sub">independent bets in 5 positions</span></div>{_dots(snap["neff"])}'
              f'{_meter(snap["rho"], 0.1, 0.9, {"P10": CORR_Q[2], "median": CORR_Q[10], "P90": CORR_Q[18]}, grad)}'
              f'<div class="sub">Industries move together at <b class="num" style="color:{INK};">{snap["rho"]:.2f}</b> '
              f'(63-day average correlation, {_ord(int(round(cp * 100)))} percentile since 2008). '
              f'In a stress month this has dropped to about <b>{neff_s:.1f} bets</b>.</div>'
              f'<div class="act"><b>Count names that move together as one position, and keep no more than half of your open '
              f'risk in one sector.</b>'
              + (' <span style="color:#f4a062;"><b>Correlation runs highest in this state: treat the book as about 1.5 bets.</b></span>' if flag else '')
              + '</div></div>')
    else:
        h += f'<div class="tile"><div class="lab">Effective bets</div><div class="sub">Industry prices not available.</div></div>'
    # 4. Insurance cost
    if _fin(snap['cost']):
        grad = f'<stop offset="0" stop-color="{TEAL}"/><stop offset="0.5" stop-color="#5b5148"/><stop offset="1" stop-color="{ORANGE}"/>'
        pc = snap['cost_pct']
        h += (f'<div class="tile"><div class="lab">Insurance cost {_ev("Descriptive")}</div>'
              f'<div style="display:flex;align-items:baseline;gap:8px;"><span class="big num" style="color:{INK};">{snap["cost"] * 100:.2f}%</span>'
              f'<span class="sub">of SPY per month · 3-month put, 10% OTM</span></div>'
              f'{_meter(pc, 0, 1, {"cheap": 0.2, "median": 0.5, "dear": 0.8}, grad)}'
              f'<div class="sub">{"—" if not _fin(pc) else _ord(int(round(pc * 100))) + " percentile since 2008"} · VIX3M {snap["vix3m"]:.1f}</div>'
              f'<div class="act">{insurance_sentence(snap["bandg"], snap["trend"], snap["inverted"])}</div></div>')
    else:
        h += '<div class="tile"><div class="lab">Insurance cost</div><div class="sub">VIX3M not available.</div></div>'
    h += '</div>'
    # 5. Vol guide (collapsed by default)
    inv = snap['invested']
    if _fin(inv) and _fin(snap['rv']):
        h += (f'<details class="vg"><summary class="lab" style="display:list-item;cursor:pointer;">Book volatility guide '
              f'· {inv * 100:.0f}% invested {_ev("Descriptive")}</summary>'
              f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:6px;">'
              f'<div style="flex:0 0 220px;height:9px;background:#1a1918;border-radius:2px;position:relative;">'
              f'<div style="position:absolute;left:0;top:0;bottom:0;width:{inv * 100:.0f}%;background:{TEAL_M};border-radius:2px;"></div>'
              f'<div title="cap 60%" style="position:absolute;left:60%;top:-3px;bottom:-3px;width:2px;background:{COPPER};"></div></div>'
              f'<span class="sub">SPY realized volatility (21 days) <b class="num" style="color:{INK};">{snap["rv"] * 100:.1f}%</b>. '
              f'A 10% volatility target would hold <b class="num" style="color:{INK};">{inv * 100:.0f}% invested</b> (cap 60%). '
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
