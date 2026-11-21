"""
joint_view.py — Joint view card on the MRS (8.10.0, 21 Nov 2026)
==================================================================
Two variables at once, in two market states, with today's point on top (the PCRM-style joint plot):
  1. The 200-day filter: dip depth E (ATR below the 50-day) vs the worst further fall within 60 sessions, S&P 500 DIP ZONE
     days since 1963, filter ON vs OFF. The split itself is tested (DF1, RB1, RB2); the cell around today is descriptive.
  2. Fear vs price: SPY daily return vs the change in VIX (points), 1990-2026, by VIX width at the prior close.
History: joint_long.csv (offline, build_joint_long.py). Today: SPY OHLC for E, the MRS history for SPY and VIX closes.
Descriptive: it shows where today sits in the past, not what comes next.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import ui

T = ui.T
PATH = Path(__file__).with_name('joint_long.csv')
STATE_COL = {'ON': T['teal'], 'OFF': T['orange'], 'CALM': T['teal'], 'WIDE': T['copper'], 'STRESSED': T['orange']}
STATE_NAME = {'ON': 'Filter ON · above a rising 200-day', 'OFF': 'Filter OFF',
              'CALM': 'VIX CALM', 'WIDE': 'VIX WIDE', 'STRESSED': 'VIX STRESSED'}
F_RANGE = dict(x=(-5.5, -1.0), y=(-30.0, 2.0))
V_RANGE = dict(x=(-5.0, 5.0), y=(-8.0, 12.0))


def _e(s):
    return ui.e(s)


def _fin(x):
    try:
        return x is not None and np.isfinite(float(x))
    except Exception:
        return False


def load(path=PATH):
    """The offline history, or None."""
    try:
        d = pd.read_csv(path, parse_dates=['date'])
        d['vb'] = d['vb'].fillna('')
        return d
    except Exception:
        return None


def today_inputs(spy_ohlc=None, hist=None, f200=None, vol_band=None) -> dict:
    """E (SPY vs its 50-day in Wilder ATR14), filter ON/OFF, SPY return %, VIX change and the VIX width at the prior close."""
    out = dict(date=None, E=None, f=None, ret=None, dvix=None, vb=None, vix=None)
    try:
        s = spy_ohlc.rename(columns=str.lower)[['high', 'low', 'close']].astype(float).dropna()
        c, h, l = s['close'], s['high'], s['low']
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        e = (c - c.rolling(50).mean()) / atr
        if _fin(e.iloc[-1]):
            out['E'], out['date'] = float(e.iloc[-1]), pd.Timestamp(s.index[-1])
    except Exception:
        pass
    st_ = f200.get('state') if isinstance(f200, dict) else f200
    if st_:
        out['f'] = 'ON' if st_ == 'F-UP' else 'OFF'
    try:
        hh = hist[['date', 'spy', 'vix']].dropna().sort_values('date')
        if len(hh) >= 2:
            a, b = hh.iloc[-2], hh.iloc[-1]
            out['ret'] = float(b['spy'] / a['spy'] - 1) * 100
            out['dvix'] = float(b['vix'] - a['vix'])
            out['vix'] = float(b['vix'])
            out['vb'] = vol_band(float(a['vix'])) if vol_band else None
            out['date'] = out['date'] or pd.Timestamp(b['date'])
            out['vdate'] = pd.Timestamp(b['date'])
    except Exception:
        pass
    return out


# ── statistics ───────────────────────────────────────────────────────────────────────────────────────────────
def _fall_stats(x: pd.Series) -> dict:
    x = x.dropna()
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), med=float(x.median()), p10=float(x.quantile(.1)), p10pct=float((x <= -10).mean()))


def filter_stats(L: pd.DataFrame, E_now=None, f_now=None, k=120) -> dict:
    """Further-fall stats by filter state; plus the k DIP ZONE days nearest today's E in today's state."""
    d = L[L['fall60'].notna()]
    out = {f: _fall_stats(d[d['F'] == f]['fall60']) for f in ('ON', 'OFF')}
    out['in_dip'] = bool(_fin(E_now) and E_now <= -1)
    if out['in_dip'] and f_now in ('ON', 'OFF'):
        x = d[d['F'] == f_now].assign(dist=lambda z: (z['E'] - E_now).abs()).nsmallest(k, 'dist')
        out['near'] = dict(_fall_stats(x['fall60']), lo=float(x['E'].min()), hi=float(x['E'].max()), f=f_now)
    return out


def vix_stats(L: pd.DataFrame, ret_now=None, dvix_now=None, vb_now=None) -> dict:
    """Slope of VIX on SPY in today's width; where today's VIX change sits among days with a similar SPY move."""
    d = L[(L['vb'] != '') & L['dvix'].notna() & L['ret'].notna()]
    out = {}
    for vb in ('CALM', 'WIDE', 'STRESSED'):
        x = d[d['vb'] == vb]
        if len(x) < 30:
            continue
        sl, ic = np.polyfit(x['ret'], x['dvix'], 1)
        dn = x[x['ret'] <= -1]
        out[vb] = dict(n=int(len(x)), slope=float(sl), icpt=float(ic), corr=float(x['ret'].corr(x['dvix'])),
                       sd=float(x['dvix'].std()), p_down_down=float((dn['dvix'] < 0).mean()) if len(dn) else np.nan, n_dn=int(len(dn)))
    if vb_now in out and _fin(ret_now) and _fin(dvix_now):
        x = d[d['vb'] == vb_now]
        for w in (0.25, 0.5, 1.0):
            sim = x[(x['ret'] - ret_now).abs() <= w]
            if len(sim) >= 40:
                break
        if len(sim) >= 10:
            pct = float((sim['dvix'] < dvix_now).mean())
            out['today'] = dict(n=int(len(sim)), w=w, med=float(sim['dvix'].median()), p10=float(sim['dvix'].quantile(.1)),
                                p90=float(sim['dvix'].quantile(.9)), pct=pct,
                                label=('more fear than usual for this move' if pct >= 0.8 else
                                       'less fear than usual for this move' if pct <= 0.2 else
                                       'a normal fear reaction for this move'))
    return out


# ── figures ─────────────────────────────────────────────────────────────────────────────────────────────────
def _rgba(hex_, a):
    h = hex_.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})'


def _scale(col):
    return [[0, _rgba(col, 0)], [0.06, _rgba(col, 0.10)], [0.25, _rgba(col, 0.28)], [0.5, _rgba(col, 0.5)],
            [0.75, _rgba(col, 0.75)], [1, col]]


def _grid(titles):
    from plotly.subplots import make_subplots
    return make_subplots(rows=2, cols=4, row_heights=[0.16, 0.84], column_widths=[0.42, 0.07, 0.42, 0.07],
                         horizontal_spacing=0.012, vertical_spacing=0.02,
                         subplot_titles=(titles[0], '', titles[1], '', '', '', '', ''))


def _smooth(a, sigma):
    """Gaussian smoothing along every axis of a (numpy only; no scipy on the server)."""
    r = int(3 * sigma) or 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    for ax in range(a.ndim):
        a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode='constant'), k, mode='valid'), ax, a)
    return a


def density(x, y, xr, yr, n=90, sigma=3.2):
    """Smoothed 2-D density on an n x n grid, scaled to max 1 -> (xc, yc, z[y, x]). Points outside the window are left out."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x >= xr[0]) & (x <= xr[1]) & (y >= yr[0]) & (y <= yr[1])
    H, xe, ye = np.histogram2d(x[m], y[m], bins=n, range=[list(xr), list(yr)])
    z = _smooth(H, sigma)
    z = z / z.max() if z.max() > 0 else z
    return (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, z.T


def _marg(v, rng, n=90, sigma=3.2):
    v = np.asarray(v, float); v = v[(v >= rng[0]) & (v <= rng[1])]
    h, e = np.histogram(v, bins=n, range=list(rng))
    h = _smooth(h.astype(float), sigma)
    return (e[:-1] + e[1:]) / 2, h / h.max() if h.max() > 0 else h


def _panel(fig, x, y, col, c0, xr, yr):
    import plotly.graph_objects as go
    xc, yc, z = density(x, y, xr, yr)
    fig.add_trace(go.Contour(x=xc, y=yc, z=z, colorscale=_scale(col), showscale=False, hoverinfo='skip',
                             contours=dict(coloring='fill', showlines=False, start=0.07, end=1.0, size=0.12),
                             line=dict(width=0)), 2, c0)
    mx, hx = _marg(x, xr); my, hy = _marg(y, yr)
    fig.add_trace(go.Scatter(x=mx, y=hx, mode='lines', fill='tozeroy', line=dict(color=col, width=1.2),
                             fillcolor=_rgba(col, 0.25), hoverinfo='skip'), 1, c0)
    fig.add_trace(go.Scatter(x=hy, y=my, mode='lines', fill='tozerox', line=dict(color=col, width=1.2),
                             fillcolor=_rgba(col, 0.25), hoverinfo='skip'), 2, c0 + 1)


def _style(fig, height, xr, yr, xt, yt):
    import plotly.io as pio
    fig.update_layout(template='plotly_dark+atn' if 'atn' in pio.templates else 'plotly_dark')
    fig.update_layout(height=height, margin=dict(l=46, r=8, t=30, b=42), bargap=0.05, showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    for a in fig.layout.annotations:
        if a.text and a.yref == 'paper' and a.font.size in (None, 16):
            a.font = dict(size=12, color=T['muted'], family=ui.DISPLAY)
    for c0 in (1, 3):
        fig.update_xaxes(range=list(xr), title_text=xt, title_font=dict(size=11, color=T['muted']), row=2, col=c0,
                         zeroline=False, gridcolor=T['line'])
        fig.update_yaxes(range=list(yr), row=2, col=c0, zeroline=False, gridcolor=T['line'])
        fig.update_xaxes(visible=False, range=list(xr), row=1, col=c0); fig.update_yaxes(visible=False, range=[0, 1.05], row=1, col=c0)
        fig.update_xaxes(visible=False, range=[0, 1.05], row=2, col=c0 + 1); fig.update_yaxes(visible=False, range=list(yr), row=2, col=c0 + 1)
    fig.update_yaxes(title_text=yt, title_font=dict(size=11, color=T['muted']), row=2, col=1)


def _note(fig, x, y, text, col, c0, ax=-30, ay=-26):
    fig.add_annotation(x=x, y=y, text=text, showarrow=True, arrowhead=0, arrowcolor=col, ax=ax, ay=ay,
                       font=dict(size=11, color=col, family=ui.MONO), bgcolor='rgba(10,10,10,0.75)',
                       xref=f'x{5 + c0 - 1}', yref=f'y{5 + c0 - 1}')


def figure_filter(L: pd.DataFrame, today: dict = None, stats: dict = None):
    import plotly.graph_objects as go
    d = L[L['fall60'].notna()]
    fig = _grid([STATE_NAME['ON'], STATE_NAME['OFF']])
    for f, c0 in (('ON', 1), ('OFF', 3)):
        x = d[d['F'] == f]
        _panel(fig, x['E'].values, x['fall60'].values, STATE_COL[f], c0, F_RANGE['x'], F_RANGE['y'])
    _style(fig, 360, F_RANGE['x'], F_RANGE['y'], 'dip depth E · ATR below the 50-day', 'worst further fall · 60 sessions (%)')
    for c0 in (1, 3):
        fig.update_xaxes(dtick=1, row=2, col=c0)
    today, stats = today or {}, stats or {}
    if stats.get('in_dip') and today.get('f') in ('ON', 'OFF'):
        c0 = 1 if today['f'] == 'ON' else 3
        e = float(np.clip(today['E'], F_RANGE['x'][0] + 0.05, F_RANGE['x'][1]))
        fig.add_vline(x=e, line=dict(color=T['ink'], width=1.2, dash='dot'), row=2, col=c0)
        nr = stats.get('near') or {}
        if nr.get('n'):
            fig.add_trace(go.Scatter(x=[e, e], y=[nr['p10'], nr['med']], mode='lines+markers',
                                     line=dict(color=T['ink'], width=4), marker=dict(size=8, color=[T['orange'], T['ink']]),
                                     hoverinfo='skip'), 2, c0)
            _note(fig, e, nr['med'], f'today E {today["E"]:.1f} · median {nr["med"]:.1f}%', T['ink'], c0, ax=60, ay=-22)
            _note(fig, e, nr['p10'], f'worst 10% {nr["p10"]:.1f}%', T['orange'], c0, ax=60, ay=18)
    return fig


def figure_vix(L: pd.DataFrame, today: dict = None, stats: dict = None):
    import plotly.graph_objects as go
    d = L[(L['vb'] != '') & L['dvix'].notna()]
    today, stats = today or {}, stats or {}
    vb = today.get('vb') if today.get('vb') in ('CALM', 'WIDE', 'STRESSED') else 'STRESSED'
    left, right = ('CALM', vb) if vb != 'CALM' else ('CALM', 'STRESSED')
    titles = [STATE_NAME[left] + (' · today' if vb == 'CALM' else ' · for comparison'),
              STATE_NAME[right] + (' · today' if vb == right else ' · for comparison')]
    fig = _grid(titles)
    for s_, c0 in ((left, 1), (right, 3)):
        x = d[d['vb'] == s_]
        _panel(fig, x['ret'].values, x['dvix'].values, STATE_COL[s_], c0, V_RANGE['x'], V_RANGE['y'])
    _style(fig, 360, V_RANGE['x'], V_RANGE['y'], 'SPY daily return (%)', 'VIX change (points)')
    c0 = 1 if vb == 'CALM' else 3
    st_ = stats.get(vb) or {}
    if st_:
        xs = np.array(V_RANGE['x'])
        fig.add_trace(go.Scatter(x=xs, y=st_['icpt'] + st_['slope'] * xs, mode='lines', hoverinfo='skip',
                                 line=dict(color=T['grey'], width=1, dash='dash')), 2, c0)
    if _fin(today.get('ret')) and _fin(today.get('dvix')):
        x = float(np.clip(today['ret'], *V_RANGE['x'])); y = float(np.clip(today['dvix'], *V_RANGE['y']))
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', hoverinfo='skip',
                                 marker=dict(size=13, color='rgba(0,0,0,0)', line=dict(color=T['ink'], width=2.5))), 2, c0)
        _note(fig, x, y, f'today {today["ret"]:+.1f}% · VIX {today["dvix"]:+.1f}', T['ink'], c0, ax=70 if x < 0 else -70, ay=-26)
    return fig


# ── words ───────────────────────────────────────────────────────────────────────────────────────────────────
def _p(v, dec=1):
    return f'{v:.{dec}f}%'.replace('-', '\u2212')


def _m(text):
    """Typographic minus for numbers in the reading."""
    import re
    return re.sub(r'(?<![\w])-(?=\d)', '\u2212', text)


def filter_lines(today: dict, stats: dict) -> list:
    on, off = stats.get('ON', {}), stats.get('OFF', {})
    L_ = []
    if _fin(today.get('E')):
        where = 'in the DIP ZONE' if stats.get('in_dip') else 'not in the DIP ZONE (E above −1)'
        L_.append(f'<b>Today:</b> SPY {today["E"]:.1f} ATR from its 50-day, {where}; filter <b>{_e(today.get("f") or "n/a")}</b>.')
    if on.get('n') and off.get('n'):
        L_.append(f'<b>What it shows:</b> on DIP ZONE days since 1963 the filter-OFF cloud sits lower and runs deeper. '
                  f'Worst further fall within 60 sessions: median {_p(off["med"])} OFF vs {_p(on["med"])} ON; '
                  f'a further 10%+ on {off["p10pct"] * 100:.0f}% of OFF days vs {on["p10pct"] * 100:.0f}% ON.')
        L_.append('Each cloud barely tilts: how deep the dip is says little; the 200-day state sets the left tail.')
    nr = stats.get('near')
    if nr and nr.get('n'):
        L_.append(f'<b>Days like today</b> (filter {nr["f"]}, E {nr["lo"]:.1f} to {nr["hi"]:.1f}, {nr["n"]} days): '
                  f'median further fall {_p(nr["med"])}, worst 10% {_p(nr["p10"])}, 10%+ on {nr["p10pct"] * 100:.0f}%.')
    elif not stats.get('in_dip'):
        L_.append('Today is outside the DIP ZONE, so no point is drawn.')
    return [_m(x) for x in L_]


def vix_lines(today: dict, stats: dict) -> list:
    L_ = []
    vb = today.get('vb')
    if _fin(today.get('ret')) and _fin(today.get('dvix')):
        L_.append(f'<b>Today:</b> SPY {today["ret"]:+.1f}%, VIX {today["dvix"]:+.1f} pts to {today["vix"]:.1f}; '
                  f'VIX width at the prior close <b>{_e(vb or "n/a")}</b>.')
    c, s = stats.get('CALM'), stats.get(vb) if vb in stats else None
    if c and stats.get('STRESSED'):
        stx = stats['STRESSED']
        L_.append(f'<b>What it shows:</b> both clouds tilt down to the right: VIX rises when SPY falls (about '
                  f'{-c["slope"]:.1f} pts per 1% fall when CALM, {-stx["slope"]:.1f} when STRESSED). '
                  f'The STRESSED cloud is the same shape, about {stx["sd"] / c["sd"]:.0f}× wider.')
    t = stats.get('today')
    if t:
        L_.append(f'<b>Today vs similar days</b> ({vb}, SPY within ±{t["w"]:.2g}% of today, {t["n"]} days): VIX changed '
                  f'{t["med"]:+.1f} pts at the median (10–90%: {t["p10"]:+.1f} to {t["p90"]:+.1f}). '
                  f'Today is at the {t["pct"] * 100:.0f}th percentile: <b>{t["label"]}</b>.')
    if s and _fin(s.get('p_down_down')):
        L_.append(f'Rare case to watch, lower left: SPY down 1%+ with VIX down. {s["p_down_down"] * 100:.0f}% of such days '
                  f'when {vb} ({s["n_dn"]} days): protection already paid for.')
    return [_m(x) for x in L_]


def lines_html(title: str, lines: list, tag_html: str = '') -> str:
    return (f'<div class="atn-tile"><div class="atn-tl">{_e(title)}</div>'
            + ''.join(f'<div class="atn-sm" style="font-size:13px;margin:0 0 9px 0;color:{T["cream"]}">{l}</div>' for l in lines)
            + (f'<div style="margin-top:6px">{tag_html}</div>' if tag_html else '') + '</div>')


def kicker_html(today: dict) -> str:
    d = today.get('date')
    dt = f' · {pd.Timestamp(d):%a %d %b}' if d is not None else ''
    return (f'<div class="atn-k" style="display:flex;justify-content:space-between;align-items:center;gap:8px;">'
            f'<span>Joint view <b>· two variables at once, today on top{dt}</b></span>'
            f'{ui.tag("Descriptive · history, not a forecast")}</div>')


HOW_TO_MD = """
**How to read a joint plot.** Each panel is one market state; each point in the shaded cloud is one past day.
- **Shading = how often:** dark = where most days landed, each lighter ring = rarer, the faint edge = the tails.
- **Edge bars = each variable alone** (top: across, right: up). They cannot show how the two move together; the cloud does.
- **Tilt = correlation:** up-right positive, down-right negative, round none. **Height / width = how much each swings.**
- **Compare the panels:** same axes, different state. Look for a change in tilt, in size, and in the faint outer band.
- **Today's point** shows where today sits in that history. It does not say what happens next.

**200-day filter:** S&P 500 DIP ZONE days since 1963 (consecutive days overlap). The ON / OFF split is tested (DF1, robust in
RB1, out of sample on 12 ETFs in RB2); the band around today is descriptive.
**Fear vs price:** S&P 500 daily returns and VIX changes since 1990, by VIX width at the prior close. Descriptive.
"""
