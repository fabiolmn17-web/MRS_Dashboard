"""
rotation_view.py — Sector rotation at a glance (top banner)
============================================================
A quadrant map of the 11 SPDR sectors: x = RS 6M, y = RS 3M vs SPY (the same
additive RS as the sector table). The four corners are the sector table's own
trend states:
    Leading   (6M > 0, 3M > 0)      Fading    (6M > 0, 3M < 0)
    Improving (6M < 0, 3M > 0)      Lagging   (6M < 0, 3M < 0)
A short tail shows where each sector was 5 and 10 sessions ago, so the direction
of rotation is visible. Next to it, a text summary names the sectors in each
corner and the strongest / weakest industries (industry ETFs, RS 3M, nightly).

Descriptive only: no forecast is implied by a sector's quadrant.
"""
import html as _html
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SHORT = {
    'Technology': 'Tech', 'Comm Services': 'Comm', 'Consumer Disc': 'Discretionary',
    'Consumer Staples': 'Staples', 'Energy': 'Energy', 'Financials': 'Financials',
    'Health Care': 'Health Care', 'Industrials': 'Industrials', 'Materials': 'Materials',
    'Real Estate': 'Real Estate', 'Utilities': 'Utilities',
}

# quadrant -> (label, colour). Same semantic colours as the sector table's trend cells.
QUAD = {
    'LEADING':   ('Leading',   '#22c55e'),
    'IMPROVING': ('Improving', '#86efac'),
    'FADING':    ('Fading',    '#f97316'),
    'LAGGING':   ('Lagging',   '#f87171'),
}
INK, MUTED, GRID = '#e5e7eb', '#9ca3af', '#2b3340'


def quadrant(rs6, rs3):
    if rs6 is None or rs3 is None or np.isnan(rs6) or np.isnan(rs3):
        return None
    if rs6 >= 0:
        return 'LEADING' if rs3 >= 0 else 'FADING'
    return 'IMPROVING' if rs3 >= 0 else 'LAGGING'


def _rs_at(s: pd.Series, spy: pd.Series, n: int, lag: int):
    """RS vs SPY over n bars, measured as of `lag` bars ago (same formula as the sector table)."""
    # Base bar = iloc[-n] at lag 0, exactly as rs() in app.build_sector_table, so the map
    # and the sector table always agree (the earlier version used iloc[-n-1], one bar older).
    if len(s) <= n + lag or len(spy) <= n + lag:
        return np.nan
    p_now, p_ago = float(s.iloc[-1 - lag]), float(s.iloc[-n - lag])
    q_now, q_ago = float(spy.iloc[-1 - lag]), float(spy.iloc[-n - lag])
    if p_ago == 0 or q_ago == 0:
        return np.nan
    return p_now / p_ago - q_now / q_ago


def rotation_points(closes: pd.DataFrame, sector_map: dict, lags=(10, 5, 0)) -> pd.DataFrame:
    """One row per sector and lag: Sector, ETF, lag, rs6, rs3 (as fractions)."""
    if closes is None or 'SPY' not in closes.columns:
        return pd.DataFrame()
    spy = closes['SPY'].dropna()
    rows = []
    for etf, sector in sector_map.items():
        if etf not in closes.columns:
            continue
        s = closes[etf].dropna()
        for lag in lags:
            rows.append({'Sector': sector, 'ETF': etf, 'lag': lag,
                         'rs6': _rs_at(s, spy, 126, lag), 'rs3': _rs_at(s, spy, 63, lag)})
    return pd.DataFrame(rows)


def _place_labels(pts, xr, yr, w_px=560, h_px=300):
    """Greedy label placement in pixel space. Returns {sector: (ax, ay, xanchor)}."""
    sx = w_px / (xr[1] - xr[0])
    sy = h_px / (yr[1] - yr[0])
    cands = [(0, -16, 'center'), (0, 16, 'center'), (12, 0, 'left'), (-12, 0, 'right'),
             (12, -14, 'left'), (-12, -14, 'right'), (12, 14, 'left'), (-12, 14, 'right'),
             (0, -30, 'center'), (0, 30, 'center')]
    # keep clear of the four corner quadrant names
    placed = [(0, 0, 80, 16), (w_px - 80, 0, w_px, 16), (0, h_px - 16, 80, h_px), (w_px - 80, h_px - 16, w_px, h_px)]
    out = {}
    dots = [((r.rs6 * 100 - xr[0]) * sx, (yr[1] - r.rs3 * 100) * sy) for r in pts.itertuples()]
    for r, (px, py) in sorted(zip(pts.itertuples(), dots), key=lambda t: t[1][1]):
        lab = SHORT.get(r.Sector, r.Sector)
        w, h = 6.4 * len(lab) + 4, 14
        best = None
        for ax, ay, anc in cands:
            cx = px + ax
            x0 = cx - w / 2 if anc == 'center' else (cx if anc == 'left' else cx - w)
            y0 = py + ay - h / 2
            box = (x0, y0, x0 + w, y0 + h)
            inside = box[0] >= 0 and box[2] <= w_px and box[1] >= 0 and box[3] <= h_px
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed)
            hit_dot = any(box[0] - 4 < dx < box[2] + 4 and box[1] - 4 < dy < box[3] + 4 for dx, dy in dots)
            score = (hit, hit_dot, not inside)
            if best is None or score < best[0]:
                best = (score, (ax, ay, anc), box)
            if score == (False, False, False):
                break
        placed.append(best[2])
        out[r.Sector] = best[1]
    return out


def rotation_figure(points: pd.DataFrame, height: int = 300) -> go.Figure:
    now = points[points['lag'] == 0].dropna(subset=['rs6', 'rs3'])
    fig = go.Figure()
    if now.empty:
        return fig
    allx = points['rs6'].dropna() * 100
    ally = points['rs3'].dropna() * 100
    mx = max(abs(allx).max(), 1.0) * 1.18
    my = max(abs(ally).max(), 1.0) * 1.35
    xr, yr = (-mx, mx), (-my, my)

    # quadrant shading + corner names
    for (x0, x1, y0, y1, key, xa, ya, xan, yan) in [
        (0, mx, 0, my, 'LEADING', mx, my, 'right', 'top'),
        (-mx, 0, 0, my, 'IMPROVING', -mx, my, 'left', 'top'),
        (0, mx, -my, 0, 'FADING', mx, -my, 'right', 'bottom'),
        (-mx, 0, -my, 0, 'LAGGING', -mx, -my, 'left', 'bottom'),
    ]:
        lbl, col = QUAD[key]
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1, line_width=0,
                      fillcolor=col, opacity=0.06, layer='below')
        fig.add_annotation(x=xa, y=ya, text=lbl.upper(), showarrow=False, xanchor=xan, yanchor=yan,
                           font=dict(size=10, color=col), opacity=0.85)
    fig.add_hline(y=0, line_color='#4b5563', line_width=1)
    fig.add_vline(x=0, line_color='#4b5563', line_width=1)

    # tails (10 and 5 sessions ago -> now)
    for sector, g in points.dropna(subset=['rs6', 'rs3']).groupby('Sector'):
        g = g.sort_values('lag', ascending=False)
        q = quadrant(*now.loc[now.Sector == sector, ['rs6', 'rs3']].iloc[0]) if sector in set(now.Sector) else None
        col = QUAD[q][1] if q else MUTED
        fig.add_trace(go.Scatter(x=g['rs6'] * 100, y=g['rs3'] * 100, mode='lines+markers',
                                 line=dict(color=col, width=1.5), opacity=0.45,
                                 marker=dict(size=[4, 5, 0][:len(g)], color=col),
                                 hoverinfo='skip', showlegend=False))

    # current dots
    cols = [QUAD[quadrant(a, b)][1] for a, b in zip(now.rs6, now.rs3)]
    hover = [f"<b>{s}</b> ({e})<br>RS 6M {a*100:+.1f}%  ·  RS 3M {b*100:+.1f}%<br>{QUAD[quadrant(a, b)][0]}"
             for s, e, a, b in zip(now.Sector, now.ETF, now.rs6, now.rs3)]
    fig.add_trace(go.Scatter(x=now.rs6 * 100, y=now.rs3 * 100, mode='markers',
                             marker=dict(size=11, color=cols, line=dict(color='#252538', width=2)),
                             hovertext=hover, hoverinfo='text', showlegend=False))
    # direct labels, placed to avoid collisions
    pos = _place_labels(now, xr, yr, w_px=600, h_px=height - 44)
    for r in now.itertuples():
        ax, ay, anc = pos[r.Sector]
        fig.add_annotation(x=r.rs6 * 100, y=r.rs3 * 100, text=SHORT.get(r.Sector, r.Sector),
                           showarrow=False, xshift=ax, yshift=-ay, xanchor=anc,
                           font=dict(size=11, color=INK))

    fig.update_layout(
        height=height, margin=dict(l=44, r=10, t=10, b=34),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=MUTED, size=10),
        xaxis=dict(range=xr, title=dict(text='RS 6M vs SPY (%)', font=dict(size=10)),
                   gridcolor=GRID, zeroline=False, ticksuffix='%', tickfont=dict(size=9)),
        yaxis=dict(range=yr, title=dict(text='RS 3M vs SPY (%)', font=dict(size=10)),
                   gridcolor=GRID, zeroline=False, ticksuffix='%', tickfont=dict(size=9)),
        hoverlabel=dict(bgcolor='#161b22', bordercolor='#374151', font=dict(color=INK, size=11)),
    )
    return fig


def _e(s):
    return _html.escape(str(s), quote=True)


def rotation_summary_html(points: pd.DataFrame, ind_df, ind_as_of=None, n_ind: int = 3) -> str:
    now = points[points['lag'] == 0].dropna(subset=['rs6', 'rs3']).copy()
    now['q'] = [quadrant(a, b) for a, b in zip(now.rs6, now.rs3)]
    then = points[points['lag'] == 10].set_index('Sector')
    h = '<div style="font-size:0.80rem;line-height:1.65;color:#d1d5db;">'
    for key in ['LEADING', 'IMPROVING', 'FADING', 'LAGGING']:
        lbl, col = QUAD[key]
        grp = now[now.q == key].sort_values('rs3', ascending=False)
        names = []
        for r in grp.itertuples():
            moved = ''
            if r.Sector in then.index:
                q0 = quadrant(then.loc[r.Sector, 'rs6'], then.loc[r.Sector, 'rs3'])
                if q0 and q0 != key:
                    moved = f' <span style="color:#6b7280;">(from {QUAD[q0][0].lower()})</span>'
            names.append(_e(SHORT.get(r.Sector, r.Sector)) + moved)
        h += (f'<div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
              f'background:{col};margin-right:7px;"></span><b style="color:{col};">{lbl}</b>'
              f'<span style="color:#6b7280;"> &nbsp;·&nbsp; </span>{", ".join(names) if names else "—"}</div>')
    if ind_df is not None and len(ind_df) and 'RS 3M' in ind_df.columns:
        d = ind_df.dropna(subset=['RS 3M'])
        top = d.nlargest(n_ind, 'RS 3M')
        bot = d.nsmallest(n_ind, 'RS 3M')
        fmt = lambda t: ', '.join(f'{_e(r[0])} <span style="color:#9ca3af;">{r[1] * 100:+.0f}%</span>'
                                  for r in t[['Industry', 'RS 3M']].itertuples(index=False))
        h += ('<div style="margin-top:10px;border-top:1px solid #374151;padding-top:8px;">'
              f'<div><b style="color:#22c55e;">Strongest industries</b> <span style="color:#6b7280;">(RS 3M)</span>: {fmt(top)}</div>'
              f'<div><b style="color:#f87171;">Weakest industries</b>: {fmt(bot)}</div></div>')
    h += ('<div style="font-size:0.70rem;color:#6b7280;margin-top:8px;">Map: RS vs SPY, tails show the position 5 and 10 '
          'sessions ago. Quadrants match the sector table trend. Descriptive, not a forecast.'
          + (f' Industries as of {_e(ind_as_of)}.' if ind_as_of else '') + '</div>')
    h += '</div>'
    return h
