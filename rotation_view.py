"""
rotation_view.py — Sector rotation at a glance (top banner)
============================================================
A quadrant map of the 11 SPDR sectors: x = RS 6M, y = RS 3M vs SPY (the same
additive RS as the sector table). The four corners are the sector table's own
trend states:
    Leading   (6M > 0, 3M > 0)      Fading    (6M > 0, 3M < 0)
    Improving (6M < 0, 3M > 0)      Lagging   (6M < 0, 3M < 0)
A short tail shows where each sector was 5 and 10 sessions ago, so the direction
of rotation is visible. A direction tag (↑ strengthening / ↓ weakening) is added when
RS 3M has moved by more than DIR_THRESH over the last 5 sessions: the quadrant is where
a sector IS, the tag is where it is GOING (e.g. 'Leading ↓' = still leading, losing ground). Next to it, a text summary names the sectors in each
corner and the strongest / weakest industries (industry ETFs, RS 3M, nightly).

Two horizons (HORIZONS):
    'standard'  x = RS 6M, y = RS 3M   (matches the sector table's 6M -> 3M trend)
    'short'     x = RS 3M, y = RS 1M   (the same map one step faster: one month vs three)
RS 1M = 21 sessions, same additive formula. In the short map the direction tag follows RS 1M.

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
DIR_THRESH = 0.01      # RS 3M change over 5 sessions (1 pt) needed for a direction tag
DIR = {'UP': ('↑', 'strengthening', '#22c55e'), 'DOWN': ('↓', 'weakening', '#f97316')}

# x / y columns of rotation_points() and their labels for each map horizon
HORIZONS = {
    'standard': {'x': 'rs6', 'y': 'rs3', 'xl': 'RS 6M', 'yl': 'RS 3M', 'ind': 'RS 3M',
                 'name': '6M vs 3M', 'note': 'Quadrants match the sector table trend (6M to 3M)'},
    'short':    {'x': 'rs3', 'y': 'rs1', 'xl': 'RS 3M', 'yl': 'RS 1M', 'ind': 'RS 1M',
                 'name': '3M vs 1M', 'note': 'Short-term view: the same quadrants one step faster (3M to 1M)'},
}


def quadrant(rs6, rs3):
    if rs6 is None or rs3 is None or np.isnan(rs6) or np.isnan(rs3):
        return None
    if rs6 >= 0:
        return 'LEADING' if rs3 >= 0 else 'FADING'
    return 'IMPROVING' if rs3 >= 0 else 'LAGGING'


def direction(rs3_now, rs3_5ago):
    """'UP' / 'DOWN' when RS 3M moved by more than DIR_THRESH over the last 5 sessions, else None."""
    if rs3_now is None or rs3_5ago is None or np.isnan(rs3_now) or np.isnan(rs3_5ago):
        return None
    d = rs3_now - rs3_5ago
    return 'UP' if d > DIR_THRESH else ('DOWN' if d < -DIR_THRESH else None)


def _with_dir(points: pd.DataFrame, now: pd.DataFrame, yc: str = 'rs3') -> pd.DataFrame:
    """Add d5 (change of the map's y measure over 5 sessions) and dir to the lag-0 rows."""
    ago = points[points['lag'] == 5].set_index('Sector')[yc]
    now = now.copy()
    now['d5'] = [r - ago.get(sct, np.nan) for sct, r in zip(now.Sector, now[yc])]
    now['dir'] = [direction(r, ago.get(sct, np.nan)) or '' for sct, r in zip(now.Sector, now[yc])]
    return now


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
    """One row per sector and lag: Sector, ETF, lag, rs6, rs3, rs1 (RS 6M / 3M / 1M as fractions)."""
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
                         'rs6': _rs_at(s, spy, 126, lag), 'rs3': _rs_at(s, spy, 63, lag),
                         'rs1': _rs_at(s, spy, 21, lag)})
    return pd.DataFrame(rows)


def _place_labels(pts, xr, yr, w_px=560, h_px=300, xc='rs6', yc='rs3'):
    """Greedy label placement in pixel space. Returns {sector: (ax, ay, xanchor)}."""
    sx = w_px / (xr[1] - xr[0])
    sy = h_px / (yr[1] - yr[0])
    cands = [(0, -16, 'center'), (0, 16, 'center'), (12, 0, 'left'), (-12, 0, 'right'),
             (12, -14, 'left'), (-12, -14, 'right'), (12, 14, 'left'), (-12, 14, 'right'),
             (0, -30, 'center'), (0, 30, 'center')]
    # keep clear of the four corner quadrant names
    placed = [(0, 0, 80, 16), (w_px - 80, 0, w_px, 16), (0, h_px - 16, 80, h_px), (w_px - 80, h_px - 16, w_px, h_px)]
    out = {}
    dots = [((getattr(r, xc) * 100 - xr[0]) * sx, (yr[1] - getattr(r, yc) * 100) * sy) for r in pts.itertuples()]
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


def rotation_figure(points: pd.DataFrame, height: int = 300, horizon: str = 'standard') -> go.Figure:
    H = HORIZONS.get(horizon, HORIZONS['standard'])
    xc, yc, xl, yl = H['x'], H['y'], H['xl'], H['yl']
    now = points[points['lag'] == 0].dropna(subset=[xc, yc])
    fig = go.Figure()
    if not now.empty:
        now = _with_dir(points, now, yc)
    if now.empty:
        return fig
    allx = points[xc].dropna() * 100
    ally = points[yc].dropna() * 100
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
    for sector, g in points.dropna(subset=[xc, yc]).groupby('Sector'):
        g = g.sort_values('lag', ascending=False)
        q = quadrant(*now.loc[now.Sector == sector, [xc, yc]].iloc[0]) if sector in set(now.Sector) else None
        col = QUAD[q][1] if q else MUTED
        fig.add_trace(go.Scatter(x=g[xc] * 100, y=g[yc] * 100, mode='lines+markers',
                                 line=dict(color=col, width=1.5), opacity=0.45,
                                 marker=dict(size=[4, 5, 0][:len(g)], color=col),
                                 hoverinfo='skip', showlegend=False))

    # current dots (hover shows all three RS horizons)
    cols = [QUAD[quadrant(a, b)][1] for a, b in zip(now[xc], now[yc])]
    _p = lambda v: '—' if v != v else f'{v * 100:+.1f}%'
    hover = [f"<b>{s}</b> ({e})<br>RS 6M {_p(r6)}  ·  RS 3M {_p(r3)}  ·  RS 1M {_p(r1)}<br>{QUAD[quadrant(a, b)][0]}"
             + (f" {DIR[dr][0]} {DIR[dr][1]}" if dr else '')
             + (f"<br>{yl} over 5 sessions: {d5*100:+.1f} pts" if d5 == d5 else '')
             for s, e, a, b, r6, r3, r1, dr, d5 in zip(now.Sector, now.ETF, now[xc], now[yc], now.rs6, now.rs3,
                                                        now['rs1'] if 'rs1' in now else [np.nan] * len(now),
                                                        now.dir, now.d5)]
    fig.add_trace(go.Scatter(x=now[xc] * 100, y=now[yc] * 100, mode='markers',
                             marker=dict(size=11, color=cols, line=dict(color='#252538', width=2)),
                             hovertext=hover, hoverinfo='text', showlegend=False))
    # direct labels, placed to avoid collisions
    pos = _place_labels(now, xr, yr, w_px=600, h_px=height - 44, xc=xc, yc=yc)
    for r in now.itertuples():
        ax, ay, anc = pos[r.Sector]
        lab = SHORT.get(r.Sector, r.Sector)
        if r.dir:
            lab += f' <span style="color:{DIR[r.dir][2]}">{DIR[r.dir][0]}</span>'
        fig.add_annotation(x=getattr(r, xc) * 100, y=getattr(r, yc) * 100, text=lab,
                           showarrow=False, xshift=ax, yshift=-ay, xanchor=anc,
                           font=dict(size=11, color=INK))

    fig.update_layout(
        height=height, margin=dict(l=44, r=10, t=10, b=34),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=MUTED, size=10),
        xaxis=dict(range=xr, title=dict(text=f'{xl} vs SPY (%)', font=dict(size=10)),
                   gridcolor=GRID, zeroline=False, ticksuffix='%', tickfont=dict(size=9)),
        yaxis=dict(range=yr, title=dict(text=f'{yl} vs SPY (%)', font=dict(size=10)),
                   gridcolor=GRID, zeroline=False, ticksuffix='%', tickfont=dict(size=9)),
        hoverlabel=dict(bgcolor='#161b22', bordercolor='#374151', font=dict(color=INK, size=11)),
    )
    return fig


def _e(s):
    return _html.escape(str(s), quote=True)


def rotation_summary_html(points: pd.DataFrame, ind_df, ind_as_of=None, n_ind: int = 3,
                          horizon: str = 'standard') -> str:
    H = HORIZONS.get(horizon, HORIZONS['standard'])
    xc, yc, yl = H['x'], H['y'], H['yl']
    now = points[points['lag'] == 0].dropna(subset=[xc, yc]).copy()
    now['q'] = [quadrant(a, b) for a, b in zip(now[xc], now[yc])]
    now = _with_dir(points, now, yc)
    then = points[points['lag'] == 10].set_index('Sector')
    h = '<div style="font-size:0.80rem;line-height:1.65;color:#d1d5db;">'
    for key in ['LEADING', 'IMPROVING', 'FADING', 'LAGGING']:
        lbl, col = QUAD[key]
        grp = now[now.q == key].sort_values(yc, ascending=False)
        names = []
        for r in grp.itertuples():
            moved = ''
            if r.Sector in then.index:
                q0 = quadrant(then.loc[r.Sector, xc], then.loc[r.Sector, yc])
                if q0 and q0 != key:
                    moved = f' <span style="color:#6b7280;">(from {QUAD[q0][0].lower()})</span>'
            tag = ''
            if r.dir:
                arrow, word, dcol = DIR[r.dir]
                tag = f' <span style="color:{dcol};" title="{word}: {yl} {r.d5 * 100:+.1f} pts over 5 sessions">{arrow}</span>'
            names.append(_e(SHORT.get(r.Sector, r.Sector)) + tag + moved)
        h += (f'<div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
              f'background:{col};margin-right:7px;"></span><b style="color:{col};">{lbl}</b>'
              f'<span style="color:#6b7280;"> &nbsp;·&nbsp; </span>{", ".join(names) if names else "—"}</div>')
    # sectors moving against their corner: still above the y line but falling, or below it but rising
    turn = now[((now.q.isin(['LEADING', 'IMPROVING'])) & (now.dir == 'DOWN')) |
               ((now.q.isin(['FADING', 'LAGGING'])) & (now.dir == 'UP'))]
    turn = turn.reindex(turn['d5'].abs().sort_values(ascending=False).index).head(4)    # the 4 largest moves
    if len(turn):
        items = [f'{_e(SHORT.get(r.Sector, r.Sector))} <span style="color:#9ca3af;">{QUAD[r.q][0].lower()}</span> '
                 f'<span style="color:{DIR[r.dir][2]};">{DIR[r.dir][0]} {DIR[r.dir][1]}</span> '
                 f'<span style="color:#6b7280;">({yl} {r.d5 * 100:+.1f} pts in 5 sessions)</span>' for r in turn.itertuples()]
        h += f'<div style="margin-top:6px;"><b style="color:#e5e7eb;">Turning</b><span style="color:#6b7280;"> &nbsp;·&nbsp; </span>{", ".join(items)}</div>'
    icol = H['ind'] if (ind_df is not None and H['ind'] in ind_df.columns) else 'RS 3M'
    if ind_df is not None and len(ind_df) and icol in ind_df.columns and ind_df[icol].notna().any():
        d = ind_df.dropna(subset=[icol])
        top = d.nlargest(n_ind, icol)
        bot = d.nsmallest(n_ind, icol)
        fmt = lambda t: ', '.join(f'{_e(r[0])} <span style="color:#9ca3af;">{r[1] * 100:+.0f}%</span>'
                                  for r in t[['Industry', icol]].itertuples(index=False))
        h += ('<div style="margin-top:10px;border-top:1px solid #374151;padding-top:8px;">'
              f'<div><b style="color:#22c55e;">Strongest industries</b> <span style="color:#6b7280;">({icol})</span>: {fmt(top)}</div>'
              f'<div><b style="color:#f87171;">Weakest industries</b>: {fmt(bot)}</div></div>')
    h += (f'<div style="font-size:0.70rem;color:#6b7280;margin-top:8px;">Map ({H["name"]}): RS vs SPY, tails show the position '
          f'5 and 10 sessions ago. {H["note"]}; ↑ / ↓ = {yl} up / down by more than '
          f'{DIR_THRESH * 100:.0f} pt over the last 5 sessions. RS 1M = 21 sessions. Descriptive, not a forecast.'
          + (f' Industries as of {_e(ind_as_of)}.' if ind_as_of else '') + '</div>')
    h += '</div>'
    return h
