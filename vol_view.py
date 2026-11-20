"""
vol_view.py — Vol surface panel on the MRS (8.4.0, 16 Nov 2026)
================================================================
The Gamma Live smile and term structure in the MRS style, with a computed morning read. Reads the close snapshots
written by vol_core (vol_gl.json from Gamma Live, vol_yahoo.json as the backup); for each date Gamma Live wins.
Shows the latest session against the one before. Descriptive: it says what options price, not what SPY will do.
8.9.0 (20 Nov 2026): the charts carry their own reading: spot, the next-expiry move band, the 30-day 25-delta put / call
with RR25 and the -5% put skew, the front-vs-30d slope, the VIX line and the event hump (stage 1: snapshot data only).
"""
import html as _html
import math

import numpy as np
import pandas as pd

import ui
import vol_core

T = ui.T
SRC_NAME = {'gl': 'Gamma Live close snapshot', 'yahoo': 'Yahoo backup (Gamma Live not received)'}


def _e(s):
    return _html.escape(str(s), quote=True)


def merged(gl: dict = None, yh: dict = None) -> dict:
    """{date: snapshot}, Gamma Live preferred per date."""
    gl = vol_core.load('gl') if gl is None else gl
    yh = vol_core.load('yahoo') if yh is None else yh
    out = dict(yh or {})
    out.update(gl or {})
    return {k: out[k] for k in sorted(out)}


def latest(snaps: dict):
    """(today, previous) snapshots, or (None, None)."""
    ks = [k for k in sorted(snaps) if snaps[k].get('term')]
    if not ks:
        return None, None
    return snaps[ks[-1]], (snaps[ks[-2]] if len(ks) > 1 else None)


def term_df(snap: dict) -> pd.DataFrame:
    t = pd.DataFrame((snap or {}).get('term') or [])
    if not len(t):
        return t
    for c in ('atm_iv', 'straddle', 'skew_95', 'fwd_iv', 'td', 't'):
        t[c] = pd.to_numeric(t.get(c), errors='coerce')
    return t.sort_values('dte').reset_index(drop=True)


def picks(snap: dict):
    """The smile's expiries: next, the first Friday after it, the one nearest 30 days -> [(expiry, label)]."""
    t = term_df(snap)
    sm = (snap or {}).get('smiles') or {}
    t = t[t['atm_iv'].notna() & t['expiry'].isin(sm.keys())] if len(t) else t
    out, seen = [], set()
    def add(e, lab):
        if e is not None and e not in seen:
            out.append((e, lab)); seen.add(e)
    if not len(t):
        return out
    e1 = t['expiry'].iloc[0]
    add(e1, 'next')
    fri = t[(pd.to_datetime(t['expiry'], format='%Y%m%d').dt.weekday == 4) & (t['expiry'] > e1)]
    add(fri['expiry'].iloc[0] if len(fri) else None, 'Friday')
    add(t.iloc[(t['dte'] - 30).abs().argsort()[:1]]['expiry'].iloc[0], '~30d')
    return out


def _md(e):
    return f'{e[4:6]}/{e[6:]}'


def _near30(t: pd.DataFrame):
    ok = t[t['atm_iv'].notna()]
    return ok.iloc[(ok['dte'] - 30).abs().argsort()[:1]].iloc[0] if len(ok) else None



def delta(S, K, t, iv, right, r=None):
    """Black-Scholes delta (same rate as vol_core). NaN when an input is missing."""
    r = vol_core.RATE if r is None else r
    if not all(vol_core.fin(v) for v in (S, K, t, iv)) or S <= 0 or K <= 0 or t <= 0 or iv <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    n = vol_core._ncdf(d1)
    return n if right == 'C' else n - 1.0


def rr25(snap: dict, expiry: str):
    """25-delta put and call on one smile (OTM side each, interpolated in delta) -> dict or None.
    rr = call IV - put IV in vol points (more negative = puts dearer)."""
    sm = ((snap or {}).get('smiles') or {}).get(expiry)
    t = term_df(snap)
    row = t[t['expiry'] == expiry] if len(t) else t
    if not sm or not len(row) or not vol_core.fin(row['t'].iloc[0]):
        return None
    S, tt = float(snap['spot']), float(row['t'].iloc[0])
    pts = np.array(sm, dtype=float)
    out = {}
    for right, side, target in (('P', pts[:, 0] <= S, -0.25), ('C', pts[:, 0] >= S, 0.25)):
        p = pts[side]
        if len(p) < 2:
            return None
        d = np.array([delta(S, k, tt, iv, right) for k, iv in p])
        ok = np.isfinite(d)
        p, d = p[ok], d[ok]
        o = np.argsort(d)
        if not len(d) or target < d.min() or target > d.max():
            return None
        out['k_' + right] = float(np.interp(target, d[o], p[o, 0]))
        out['iv_' + right] = float(np.interp(target, d[o], p[o, 1]))
    out['rr'] = (out['iv_C'] - out['iv_P']) * 100
    return out


def figure(today: dict, prev: dict = None, vix=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    cols = [T['orange'], T['teal'], T['ink']]
    spot = float(today['spot'])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.08,
                        subplot_titles=('Smile · OTM IV by strike · solid today, dashed the session before',
                                        'Term structure · ATM IV by days to expiry'))
    for (e, lab), col in zip(picks(today), cols):
        pts = np.array(today['smiles'][e], dtype=float)
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1] * 100, mode='lines', line=dict(color=col, width=2),
                                 name=f'{lab} {_md(e)}', hovertemplate='%{x:.0f}: %{y:.1f}%<extra>' + f'{lab} {_md(e)}</extra>'), 1, 1)
        pv = ((prev or {}).get('smiles') or {}).get(e)
        if pv:
            pv = np.array(pv, dtype=float)
            fig.add_trace(go.Scatter(x=pv[:, 0], y=pv[:, 1] * 100, mode='lines', line=dict(color=col, width=1, dash='dash'),
                                     opacity=.7, showlegend=False, hoverinfo='skip'), 1, 1)
    fig.add_vline(x=spot, line=dict(color=T['strong'], width=1, dash='dot'), row=1, col=1)
    for snap, nm, sty in ((today, 'ATM IV today', dict(color=T['ink'], width=2)),
                          (prev, 'session before', dict(color=T['grey'], width=1.2, dash='dash'))):
        t = term_df(snap) if snap else pd.DataFrame()
        if not len(t):
            continue
        t = t[t['atm_iv'].notna() & (t['dte'] <= 45)]
        fig.add_trace(go.Scatter(x=t['dte'], y=t['atm_iv'] * 100, mode='lines+markers', line=sty, marker=dict(size=5),
                                 name=nm, customdata=t['expiry'].map(_md),
                                 hovertemplate='%{customdata} · %{x}d: %{y:.1f}%<extra></extra>'), 1, 2)
    _annotate(fig, today, vix)
    import plotly.io as pio
    fig.update_layout(template='plotly_dark+atn' if 'atn' in pio.templates else 'plotly_dark')
    fig.update_layout(height=330, margin=dict(l=40, r=10, t=34, b=40),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True,
                      legend=dict(orientation='h', x=0, y=-0.2, font=dict(size=11, color=T['muted'])))
    for a in fig.layout.annotations:
        if a.text and a.text.startswith(('Smile', 'Term')):
            a.font = dict(size=12, color=T['muted'], family=ui.DISPLAY)
    fig.update_xaxes(title_text='strike', row=1, col=1, title_font=dict(size=11, color=T['muted']))
    fig.update_xaxes(title_text='calendar days to expiry', row=1, col=2, title_font=dict(size=11, color=T['muted']))
    fig.update_yaxes(ticksuffix='%')
    return fig


def _annotate(fig, today: dict, vix=None):
    """Stage-1 reading on the charts (snapshot data only; no Gamma Live levels)."""
    import plotly.graph_objects as go
    spot = float(today['spot'])
    small = dict(size=11, color=T['muted'])
    ev = today.get('event') or {}
    n = ev.get('next') or {}
    # spot label
    fig.add_annotation(x=spot, y=0.99, xref='x', yref='y domain', text=f'spot {spot:.2f}', showarrow=False,
                       xanchor='left', yanchor='top', xshift=4, font=dict(size=11, color=T['ink']), row=1, col=1)
    # next-expiry move band (the straddle)
    if vol_core.fin(n.get('straddle')):
        st_ = float(n['straddle'])
        fig.add_vrect(x0=spot - st_, x1=spot + st_, fillcolor=T['copper'], opacity=0.08, line_width=0, row=1, col=1)
        fig.add_annotation(x=spot + st_, y=0.0, xref='x', yref='y domain', showarrow=False, xanchor='left', yanchor='bottom',
                           text=f"next {_md(n['expiry'])} move ±{float(n['move_pct']):.1f}%", font=dict(size=11, color=T['copper']),
                           row=1, col=1)
    # 30-day 25-delta put / call and RR25, -5% put skew
    t = term_df(today)
    a30 = _near30(t) if len(t) else None
    pk = [e for e, lab in picks(today) if lab == '~30d']
    r = rr25(today, pk[0]) if pk else None
    if r:
        fig.add_trace(go.Scatter(x=[r['k_P'], r['k_C']], y=[r['iv_P'] * 100, r['iv_C'] * 100], mode='markers',
                                 marker=dict(size=8, color=T['ink'], line=dict(color=T['ground'], width=1)),
                                 showlegend=False, hoverinfo='skip'), 1, 1)
        for k, iv, lab, ax in ((r['k_P'], r['iv_P'], 'put', -10), (r['k_C'], r['iv_C'], 'call', 10)):
            fig.add_annotation(x=k, y=iv * 100, text=f'25Δ {lab} {iv * 100:.1f}%', showarrow=True, arrowcolor=T['strong'],
                               ax=ax, ay=-28, font=small, bgcolor='rgba(10,10,10,.8)', borderpad=2, row=1, col=1)
    box = []
    if r:
        box.append(f"30d RR25 <b>{r['rr']:+.1f}</b> pts")
    if a30 is not None and vol_core.fin(a30['skew_95']):
        box.append(f"−5% put <b>{float(a30['skew_95']) * 100:+.1f}</b> pts over ATM")
    if box:
        fig.add_annotation(x=0.99, y=0.98, xref='x domain', yref='y domain', xanchor='right', yanchor='top', showarrow=False,
                           align='right', text='<br>'.join(box), font=dict(size=11, color=T['cream']),
                           bgcolor='rgba(22,21,20,.85)', bordercolor=T['line2'], borderwidth=1, borderpad=4, row=1, col=1)
    # term structure: next and 30d points, the slope, the VIX line, the hump
    ok = t[t['atm_iv'].notna() & (t['dte'] <= 45)] if len(t) else t
    if len(ok) and a30 is not None:
        f0 = ok.iloc[0]
        slope = (float(f0['atm_iv']) - float(a30['atm_iv'])) * 100
        word = 'inverted: stress in the front' if slope > 1 else ('flat' if slope > -1 else 'upward: calm front')
        fig.add_annotation(x=float(f0['dte']), y=float(f0['atm_iv']) * 100, text=f"next {float(f0['atm_iv']) * 100:.1f}%",
                           showarrow=True, arrowcolor=T['strong'], ax=34, ay=-6, font=small, xanchor='left', row=1, col=2)
        fig.add_annotation(x=float(a30['dte']), y=float(a30['atm_iv']) * 100, text=f"30d {float(a30['atm_iv']) * 100:.1f}%",
                           showarrow=True, arrowcolor=T['strong'], ax=0, ay=-26, font=small, row=1, col=2)
        fig.add_annotation(x=0.98, y=0.98, xref='x2 domain', yref='y2 domain', xanchor='right', yanchor='top', showarrow=False,
                           align='right', text=f"front − 30d <b>{slope:+.1f}</b> pts<br>{word}",
                           font=dict(size=11, color=T['orange'] if slope > 1 else T['cream']),
                           bgcolor='rgba(22,21,20,.85)', bordercolor=T['line2'], borderwidth=1, borderpad=4)
    if vix is not None and vol_core.fin(vix):
        fig.add_hline(y=float(vix), line=dict(color=T['copper'], width=1, dash='dot'), row=1, col=2)
        fig.add_annotation(x=0.0, y=float(vix), xref='x2 domain', yref='y2', xanchor='left', yanchor='bottom', showarrow=False, xshift=6,
                           text=f'VIX {float(vix):.1f}', font=dict(size=11, color=T['copper']))
    hump = ev.get('hump')
    if hump and len(t):
        y = t.loc[t['expiry'] == hump['expiry'], 'atm_iv']
        if len(y):
            txt = f"event {_md(hump['expiry'])}"
            if vol_core.fin(hump.get('fwd_iv')) and vol_core.fin(hump.get('ref_fwd')):
                txt += f": fwd {hump['fwd_iv'] * 100:.0f}% vs {hump['ref_fwd'] * 100:.0f}%"
            fig.add_annotation(x=hump['dte'], y=float(y.iloc[0]) * 100, text=txt, showarrow=True, arrowcolor=T['copper'],
                               ay=-30, font=dict(size=11, color=T['copper']), row=1, col=2)


def read_lines(today: dict, prev: dict = None, vix=None) -> list:
    """The morning read: next expiry and its event premium, term slope, 30-day skew, and the read with the card."""
    t, tp = term_df(today), term_df(prev) if prev else pd.DataFrame()
    ev = today.get('event') or {}
    n, hump = ev.get('next'), ev.get('hump')
    a30 = _near30(t) if len(t) else None
    a30p = _near30(tp) if len(tp) else None
    def m(s):
        return '<span style="font-family:' + ui.MONO.replace('"', "'") + '">' + s + '</span>'
    out = []
    if n and n.get('straddle') is not None:
        prem, share = n.get('premium'), n.get('prem_share')
        pc = T['orange'] if prem is not None and prem > 0 else T['teal']
        line = (f"<b>Next expiry {_md(n['expiry'])}</b>: straddle {m('$%.2f' % n['straddle'])} "
                f"≈ {m('±%.1f%%' % n['move_pct'])}")
        if prem is not None and share is not None:
            line += f"; event premium <b style=\"color:{pc}\">{m('%+.2f' % prem)} ({share:+.0f}%)</b> over the 30-day pace"
        out.append(line)
    if n and a30 is not None and n.get('atm_iv') is not None:
        slope = (n['atm_iv'] - float(a30['atm_iv'])) * 100          # 8.9.0: front − 30d, as the chart and the vol log
        word = 'inverted: stress priced in the front' if slope > 1 else ('flat' if slope > -1 else 'upward: calm front')
        out.append(f"<b>Term structure</b>: ATM {m('%.1f%%' % (n['atm_iv'] * 100))} (next) → "
                   f"{m('%.1f%%' % (a30['atm_iv'] * 100))} (30d), front − 30d {m('%+.1f pts' % slope)}: {word}")
    if a30 is not None and pd.notna(a30['skew_95']):
        y = ya = ''
        if a30p is not None and pd.notna(a30p['skew_95']):
            y = f" (before {m('%+.1f' % (a30p['skew_95'] * 100))})"
        if a30p is not None and pd.notna(a30p['atm_iv']):
            ya = f" vs {m('%.1f%%' % (a30p['atm_iv'] * 100))}"
        out.append(f"<b>30-day put skew (−5%)</b>: {m('%+.1f pts' % (a30['skew_95'] * 100))}{y} · "
                   f"30d ATM {m('%.1f%%' % (a30['atm_iv'] * 100))}{ya}")
    if a30 is not None:
        if hump:
            hump_txt = (f"Hump at {_md(hump['expiry'])}: forward vol {hump['fwd_iv'] * 100:.0f}% vs "
                        f"{hump['ref_fwd'] * 100:.0f}% normal: an event is priced.")
        else:
            hump_txt = 'No hump in the term structure: no single event is priced in the next 10 days.'
        if vix is not None and vix == vix:
            gap = vix - float(a30['atm_iv']) * 100
            wing = ('much of the width sits in the downside wing (put skew)' if gap >= 3 else
                    'the wings add little: the width is in at-the-money vol')
            out.append(f"<b>Read with the card</b>: VIX {vix:.1f} vs 30-day SPY ATM {a30['atm_iv'] * 100:.1f}% "
                       f"(gap {gap:+.1f} pts): {wing}. {hump_txt}")
        else:
            out.append(f'<b>Read</b>: {hump_txt}')
    return out


def kicker_html(today: dict, prev: dict = None) -> str:
    d = pd.Timestamp(today['date'])
    pv = f' vs {pd.Timestamp(prev["date"]):%a %d %b}' if prev else ''
    src = SRC_NAME.get(today.get('src'), today.get('src', ''))
    return (f'<div class="atn-k" style="display:flex;justify-content:space-between;align-items:center;gap:8px;">'
            f'<span>Vol surface <b>· SPY {float(today["spot"]):.2f} · {_e(src)} · {d:%a %d %b}{pv}</b></span>'
            f'{ui.tag("Descriptive · what options price")}</div>')


def read_html(today: dict, prev: dict = None, vix=None) -> str:
    return ('<div class="atn-tile"><div class="atn-tl">Morning read</div>'
            + ''.join(f'<div class="atn-sm" style="font-size:13px;margin:0 0 9px 0;color:{T["cream"]}">{l}</div>'
                      for l in read_lines(today, prev, vix)) + '</div>')


def last_date(snaps: dict):
    ks = [k for k in sorted(snaps) if snaps[k].get('term')]
    return pd.Timestamp(ks[-1]) if ks else None
