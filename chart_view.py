"""
chart_view.py — daily candle chart for the scanner's side panel
================================================================
One ticker at a time, fetched on demand (yfinance, per-ticker history like the rest of the app) and cached.
Shows the last ~12 months of daily candles with:
  EMA 10 (thin) and EMA 21, SMA 200 and SMA 250 dashed, as Fabio's TradingView ribbon (8 Nov 2026; the SMA 50 line
  was replaced). Computed on the full 2-year history, so the long lines start at the left edge
  52-week high  = highest high of the last 252 sessions (including today)
  21-day high   = highest high of the 21 sessions BEFORE today (the pivot a breakout has to clear; 20 until 3 Nov 2026)
  Volume pane   (4 Nov 2026) daily volume bars (teal up / orange down day) and the 50-day average, under the price
  Base overlay  (3 Nov 2026) the current base from StockScanner/bases.py on this chart's own (unadjusted) prices:
                shaded base, the cup's left lip / low / right lip, the handle, and the pivot line
v6.5 (14 Nov 2026): drawn like the TradingView "Ext·RS" indicator v7.5 (see figure()): candles in the SPY phase colour,
  SMA 50 / 200 / 250, SMA 50 +/- 2..5 ATR lines, Deep dip / chase / R1 markers. The EMA 10 and the 52W / 21D high lines
  above are replaced (EMA 21 and the highs are in the legend).
Descriptive chart only: same data the scanner uses, no signals.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

UP, DOWN = '#85cac5', '#f4a062'
SMA50_COL, SMA200_COL = '#e8dccb', '#b9805a'
HI52_COL, HI20_COL = '#e3b48e', '#9b958f'
BASE_COL, PIVOT_COL = '#85cac5', '#f3ece2'
BARS = 252
HI_N = 21                                   # short-term high: the 21 sessions before today


def fetch(ticker: str):
    """10 years of daily OHLC for one ticker, or None (the chart shows the last 12 months; the Ext·Mom table uses
    the whole history for its odds). Wrap in st.cache_data in the app."""
    import yfinance as yf
    try:
        h = yf.Ticker(ticker).history(period='10y', auto_adjust=False)
        if h is None or h.empty or 'Close' not in h.columns:
            return None
        h = h[[c_ for c_ in ('Open', 'High', 'Low', 'Close', 'Volume') if c_ in h.columns]].dropna(subset=['Close'])
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return h
    except Exception:
        return None


def levels(h: pd.DataFrame) -> dict:
    """Key levels from the full history (last bar = latest session)."""
    out = {'hi52': np.nan, 'hi20': np.nan, 'hi21': np.nan, 'sma50': np.nan, 'sma200': np.nan, 'close': np.nan,
           'chg': np.nan}
    if h is None or len(h) < 2:
        return out
    c = h['Close']
    out['close'] = float(c.iloc[-1])
    out['chg'] = float(c.iloc[-1] / c.iloc[-2] - 1)
    out['hi52'] = float(h['High'].iloc[-252:].max())
    if len(h) > HI_N:
        out['hi21'] = float(h['High'].iloc[-HI_N - 1:-1].max())
        out['hi20'] = out['hi21']               # old key, kept for callers that still read it
    if len(c) >= 50:
        out['sma50'] = float(c.rolling(50).mean().iloc[-1])
    if len(c) >= 200:
        out['sma200'] = float(c.rolling(200).mean().iloc[-1])
    return out


def base_info(h: pd.DataFrame) -> dict:
    """The current base / breakout state on this history (bases.detect), {} when unavailable."""
    try:
        from StockScanner import bases
        return bases.detect(h)
    except Exception:
        return {}


def _add_base(fig, v, b):
    """Draw the base from bases.detect output b on the visible bars v (category x axis = date strings)."""
    if not b or not b.get('base_type'):
        return
    xs = list(v.index.strftime('%Y-%m-%d'))
    first, last = xs[0], xs[-1]
    clip = lambda d: None if d is None else (first if d < first else d if d in xs else
                                             next((x for x in xs if x >= d), last))
    x0, x1 = clip(b.get('base_start')), clip(b.get('base_end'))
    pv, typ = b['base_pivot'], b['base_type']
    lo = b.get('base_low')
    if x0 and x1 and lo is not None:
        fig.add_shape(type='rect', xref='x', yref='y', x0=x0, x1=x1, y0=lo, y1=pv, layer='below', line_width=0,
                      fillcolor='rgba(133,202,197,0.08)')
    if typ in ('Cup w/ handle', 'Cup'):
        # v4 (9 Nov 2026): a handle found as a flat base carries the cup low separately (cup_low); a cup with no handle
        # yet is drawn from the left lip through the low to today's close
        cl, cld = (b.get('cup_low'), b.get('cup_low_date')) if b.get('cup_low_date') else (lo, b.get('base_low_date'))
        rgt = (b.get('base_right_date'), pv - 0.10) if typ == 'Cup w/ handle' else (last, float(v['Close'].iloc[-1]))
        pts = [(b.get('base_left_date'), b.get('base_left')), (cld, cl), rgt]
        pts = [(clip(d), y) for d, y in pts if d and y is not None and d >= first]
        if len(pts) >= 2:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode='lines+markers',
                                     line=dict(color=BASE_COL, width=1.4, shape='spline', smoothing=1.1, dash='dot'),
                                     marker=dict(size=6, color=BASE_COL), name='Cup', hoverinfo='skip',
                                     showlegend=False))
        hs = clip(b.get('handle_start'))
        if hs and b.get('handle_low') is not None:
            fig.add_shape(type='rect', xref='x', yref='y', x0=hs, x1=x1, y0=b['handle_low'], y1=pv, layer='below',
                          line=dict(color=BASE_COL, width=1, dash='dot'), fillcolor='rgba(133,202,197,0.10)')
    if typ == 'Double bottom' and b.get('db_l1_date'):       # v3 (8 Nov 2026): the W, left high -> L1 -> middle -> L2
        pts = [(b.get('base_left_date'), b.get('base_left')), (b.get('db_l1_date'), b.get('db_l1')),
               (b.get('db_mid_date'), b.get('db_mid')), (b.get('base_low_date'), lo)]
        pts = [(clip(d), y) for d, y in pts if d and y is not None and d >= first]
        if len(pts) >= 2:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode='lines+markers',
                                     line=dict(color=BASE_COL, width=1.3, dash='dot'), marker=dict(size=6, color=BASE_COL),
                                     name='Double bottom', hoverinfo='skip', showlegend=False))
        if b.get('db_sk3_level') is not None and x0:            # shakeout+3 level (10% above the shakeout low)
            fig.add_shape(type='line', xref='x', yref='y', x0=clip(b.get('base_low_date')) or x0, x1=last,
                          y0=b['db_sk3_level'], y1=b['db_sk3_level'], line=dict(color=BASE_COL, width=1, dash='dot'))
    if typ == 'VCP' and b.get('vcp_points'):                 # contractions: swing high -> low, numbered
        for j, seg in enumerate(str(b['vcp_points']).split(';')):
            try:
                d1, y1, d2, y2 = seg.split(',')
                x1_, x2_ = clip(d1), clip(d2)
                if d1 < first:
                    continue
                fig.add_trace(go.Scatter(x=[x1_, x2_], y=[float(y1), float(y2)], mode='lines+markers',
                                         line=dict(color=BASE_COL, width=1.3, dash='dot'), marker=dict(size=5, color=BASE_COL),
                                         hoverinfo='skip', showlegend=False))
                fig.add_annotation(x=x2_, y=float(y2), xref='x', yref='y', text=f'{j + 1}', showarrow=False, yshift=-10,
                                   font=dict(size=10, color=BASE_COL))
            except ValueError:
                continue
    state = b.get('base_state') or ''
    lbl = f"{typ} · pivot {pv:,.2f} · {state}"
    if x0:
        fig.add_shape(type='line', xref='x', yref='y', x0=x0, x1=last, y0=pv, y1=pv,
                      line=dict(color=PIVOT_COL, width=1.3, dash='dash'))
        fig.add_annotation(x=x0, y=pv, xref='x', yref='y', text=lbl, showarrow=False, xanchor='left', yanchor='bottom',
                           font=dict(size=10, color=PIVOT_COL), bgcolor='rgba(18,18,18,0.75)')


def _add_pivots(fig, h, v):
    """Confirmed 10/10 pivot highs / lows (same rule as the TradingView 'Pivot Points High Low' script), as small
    price labels on the visible bars."""
    try:
        from StockScanner import bases
    except Exception:
        return
    hh, ll = h['High'].to_numpy(float), h['Low'].to_numpy(float)
    e = len(hh) - 1
    off = len(h) - len(v)
    xs = list(v.index.strftime('%Y-%m-%d'))
    for kind, idx, arr, pos in (('H', bases.pivot_highs(hh, e), hh, 'top'), ('L', bases.pivot_lows(ll, e), ll, 'bottom')):
        idx = [i for i in idx if i >= off]
        if not idx:
            continue
        fig.add_trace(go.Scatter(x=[xs[i - off] for i in idx], y=[arr[i] for i in idx], mode='text',
                                 text=[f'{arr[i]:,.2f}' for i in idx], textposition=f'{pos} center',
                                 textfont=dict(size=9, color='#c9c1b7'), hoverinfo='skip', showlegend=False))


# ── v6.5 (14 Nov 2026): the chart drawn like the TradingView "Ext·RS" indicator v7.5 (Fabio's settings) ──────────────
PH_COL = {0: '#85cac5', 1: '#649995', 2: '#e8dccb', 3: '#8a847c'}       # Dip zone / Recovery / Trend / Out
PH_NAME = {0: 'Dip zone', 1: 'Recovery', 2: 'Trend', 3: 'Out'}
S50_COL, S200_COL, E21_COL = 'rgba(201,193,183,0.65)', '#b9805a', '#9b958f'
ATR_UP_COL, ATR_DN_COL = '#b9805a', '#649995'
ATR_UP, ATR_DN, ATR_N = (2, 3, 4, 5), (2, 3, 4, 5), 20  # SMA 50 + / - k ATR(14), drawn over the last 20 sessions
MK_COL = dict(dd='#76ff03', dw='#9e9e9e', ch='#ff5252', os='#ffeb3b')
FUTURE = 9                                              # empty sessions at the right edge, for the ATR labels


def _etfs() -> set:
    out = {'SPY', 'QQQ', 'IWM', 'DIA', 'RSP', 'MDY', 'GLD', 'SLV', 'TLT', 'IEF', 'HYG', 'XBI', 'XOP', 'GDX', 'GDXJ', 'SMH',
           'IGV', 'KRE', 'KBE', 'ITB', 'XHB', 'XRT', 'XME', 'IYT', 'JETS', 'ARKK', 'VXX', 'UVXY', 'USO', 'UNG', 'FXI', 'EEM',
           'EFA', 'EWZ', 'EWJ'}
    try:
        from StockScanner import industry as _ind
        out |= set(_ind.SECTOR_ETF.values())
        out |= {t for lst in _ind.INDUSTRY_MAP.values() for t, _ in lst}
    except Exception:
        pass
    return out


def spy_phase(spy):
    """Per-session SPY phase from the allocation replay (allocation.py, the same rules as the Today card and the
    TradingView indicator): 0 = DIP ZONE (close at least 1 ATR(14) below the 50-day), 1 = RECOVERY (rebound ticket open),
    2 = TREND (21-day breakout ticket open), 3 = OUT (no ticket). pd.Series by date, or None without data."""
    if spy is None or len(spy) < 60:
        return None
    try:
        import allocation as _al
        d = _al.indicators(spy)
        n = len(d)
        tA = _al.replay(d, _al.rebound_entries(d), d['sma50'].values)
        tB = _al.replay(d, _al.breakout_entries(d), d['ema21'].values)

        def _open(trades):                 # open from the entry close up to the session before the exit
            m = np.zeros(n, bool)
            for t in trades:
                m[t['i']:(t['j'] if t['j'] is not None else n)] = True
            return m
        a, b = _open(tA), _open(tB)
        E = d['E'].values
        ph = np.where(~np.isfinite(E), 3, np.where(E <= -1, 0, np.where(a, 1, np.where(b, 2, 3))))
        return pd.Series(ph, index=pd.to_datetime(d.index).normalize())
    except Exception:
        return None


def _ma_trig(c, lo, sma, n_above: int = 20, max_below: int = 5):
    """The Deep dip rule on one moving average, per session (scanner and Pine f_ddtrig): the 20 closes before all above
    it, then an undercut (low below, close back above) or a reclaim (1-5 closes below, then a close back above)."""
    n = len(c)
    x = c - sma
    pos = np.where(np.isfinite(x), x > 0, False)
    out = np.zeros(n, bool)
    for i in range(n_above + max_below + 2, n):
        if not pos[i]:
            continue
        if pos[i - 1]:
            out[i] = bool(lo[i] < sma[i] and pos[i - n_above:i].all())
        else:
            k = 0
            while k <= max_below and np.isfinite(x[i - 1 - k]) and x[i - 1 - k] <= 0:
                k += 1
            out[i] = bool(1 <= k <= max_below and pos[i - k - n_above:i - k].all())
    return out


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    dl = c.diff()
    up = dl.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-dl.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def figure(h: pd.DataFrame, ticker: str, height: int = 520, base: dict = None, spy: pd.DataFrame = None) -> go.Figure:
    """Daily chart drawn like the TradingView Ext·RS indicator (v7.5, Fabio's settings, 14 Nov 2026):
      candles in the SPY phase colour (Dip zone / Recovery / Trend / Out), hollow on up days, filled on down days;
      SMA 50 / 200 / 250 (250 dashed); EMA 21, 52W high and 21D high in the legend (click to show);
      dotted SMA 50 +2..+5 and -2..-5 ATR(14) lines over the last 20 sessions, labelled at the right edge;
      markers: Deep dip trigger (grey when the 2 ATR stop is wider than 10%), breakout while SPY is in DIP ZONE /
      RECOVERY (stocks only), 50-day reclaim after RSI(14) < 35 (R1);
      10/10 swing price labels; the base overlay (MRS only) and the volume pane as before.
    spy: SPY daily OHLC for the phase (None = all candles in the Out colour). base: bases.detect(h) output to draw
    (computed here when None; pass {} to draw none)."""
    h = h.copy()
    c = h['Close']
    s50 = c.rolling(50).mean()
    s200 = c.rolling(200).mean()
    s250 = c.rolling(250).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    pc = c.shift()
    tr = pd.concat([h['High'] - h['Low'], (h['High'] - pc).abs(), (h['Low'] - pc).abs()], axis=1).max(axis=1)
    a14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    v = h.iloc[-BARS:]
    off = len(h) - len(v)
    x = list(v.index.strftime('%Y-%m-%d'))          # category axis: no gaps on weekends and holidays
    xf = [' ' * (i + 1) for i in range(FUTURE)]       # empty sessions at the right edge
    lv = levels(h)
    fig = go.Figure()

    # candles in the SPY phase colour
    ph_all = spy_phase(spy)
    if ph_all is not None:
        ph = ph_all.reindex(v.index).fillna(3).astype(int).to_numpy()
    else:
        ph = np.full(len(v), 3)
    for p in (0, 1, 2, 3):
        m = ph == p
        if not m.any():
            continue
        col = PH_COL[p]
        w = lambda s_: s_.where(m)
        fig.add_trace(go.Candlestick(
            x=x, open=w(v['Open']), high=w(v['High']), low=w(v['Low']), close=w(v['Close']),
            name=f'SPY {PH_NAME[p].lower()}', showlegend=False,
            increasing=dict(line=dict(color=col, width=1), fillcolor='rgba(0,0,0,0)'),
            decreasing=dict(line=dict(color=col, width=1), fillcolor=col)))
    fig.add_trace(go.Scatter(x=xf, y=[float(c.iloc[-1])] * FUTURE, mode='markers', marker=dict(opacity=0),
                             hoverinfo='skip', showlegend=False))

    # moving averages (EMA 21 and the highs: click in the legend)
    fig.add_trace(go.Scatter(x=x, y=s50.iloc[-BARS:], name='SMA 50', mode='lines',
                             line=dict(color=S50_COL, width=1), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=s200.iloc[-BARS:], name='SMA 200', mode='lines',
                             line=dict(color=S200_COL, width=1), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=s250.iloc[-BARS:], name='SMA 250', mode='lines',
                             line=dict(color=S200_COL, width=1, dash='dash'), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=e21.iloc[-BARS:], name='EMA 21 (exit trail)', mode='lines', visible='legendonly',
                             line=dict(color=E21_COL, width=1), hoverinfo='skip'))
    for y, col, dash, lbl in ((lv['hi52'], HI52_COL, 'dash', '52W high'), (lv['hi21'], HI20_COL, 'dot', '21D high')):
        if np.isfinite(y):
            fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[y, y], name=f'{lbl} {y:,.2f}', mode='lines',
                                     visible='legendonly', line=dict(color=col, width=1, dash=dash), hoverinfo='skip'))

    # ATR(14) lines from the SMA 50, last ATR_N sessions, labels at the right edge (one legend entry toggles all)
    xa = x[-ATR_N:]
    first_atr = True
    lab_x, lab_y, lab_t, lab_c = [], [], [], []
    for k in [-q for q in ATR_DN][::-1] + list(ATR_UP):
        yk = (s50 + k * a14).iloc[-ATR_N:]
        if not np.isfinite(yk.iloc[-1]):
            continue
        col = ATR_UP_COL if k > 0 else ATR_DN_COL
        fig.add_trace(go.Scatter(x=xa, y=yk, mode='lines', name='SMA 50 ± ATR lines', legendgroup='atr',
                                 showlegend=first_atr, line=dict(color=col, width=1, dash='dot'), hoverinfo='skip'))
        first_atr = False
        lab_x.append(xf[1]); lab_y.append(float(yk.iloc[-1]))
        lab_t.append(f'{"+" if k > 0 else "−"}{abs(k)} ATR'); lab_c.append(col)
    if lab_x:
        fig.add_trace(go.Scatter(x=lab_x, y=lab_y, mode='text', text=lab_t, textposition='middle right',
                                 textfont=dict(size=9, color=lab_c), legendgroup='atr', showlegend=False,
                                 hoverinfo='skip', cliponaxis=False))

    # markers (same rules as the scanner and the TradingView indicator)
    cc, ll, hh = c.to_numpy(float), h['Low'].to_numpy(float), h['High'].to_numpy(float)
    dd = _ma_trig(cc, ll, s200.to_numpy(float)) | _ma_trig(cc, ll, s250.to_numpy(float))
    wide = (2 * a14 / c).to_numpy(float) > 0.10
    os50 = _ma_trig(cc, ll, s50.to_numpy(float)) & (_rsi(c).rolling(10).min().shift(1) < 35).to_numpy()
    bo = (c > h['High'].rolling(21).max().shift(1)).to_numpy()
    bo_new = bo & ~np.r_[False, bo[:-1]]
    ph_full = np.r_[np.full(off, 3), ph]
    chase = bo_new & np.isin(ph_full, (0, 1)) & (str(ticker).upper() not in _etfs())
    av = a14.to_numpy(float)
    vis = np.arange(len(h)) >= off
    for key, mask, sym, ypos, name, tip in (
            ('dd', dd & ~wide & vis, 'triangle-up', ll - 0.6 * av, 'Deep dip trigger',
             'Deep dip trigger (undercut / reclaim of the 200 / 250 SMA after 20 closes above)'),
            ('dw', dd & wide & vis, 'triangle-up', ll - 0.6 * av, 'Deep dip, stop wider than 10%',
             'Deep dip, but the 2 ATR(14) stop is more than 10% away: about 0R in S1'),
            ('ch', chase & vis, 'circle', hh + 0.6 * av, 'Breakout in dip zone / recovery',
             'First close above the 21-day high while SPY is in DIP ZONE / RECOVERY: lagged a random entry (S2)'),
            ('os', os50 & vis, 'x', ll - 1.3 * av, '50-day reclaim after RSI < 35',
             '50-day reclaim after RSI(14) < 35 in the 10 sessions before: −0.26R (R1, Provisional avoid)')):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        fig.add_trace(go.Scatter(x=[x[i - off] for i in idx], y=ypos[idx], mode='markers', name=name,
                                 marker=dict(symbol=sym, size=8 if sym != 'circle' else 6, color=MK_COL[key],
                                             line=dict(width=0)),
                                 hovertemplate=tip + '<extra></extra>'))
    _add_base(fig, v, base_info(h) if base is None else base)
    _add_pivots(fig, h, v)

    # SPY phase key (top left)
    if ph_all is not None:
        today = PH_NAME[int(ph[-1])]
        key = ' '.join(f'<span style="color:{PH_COL[p]}">■ {PH_NAME[p]}</span>' for p in (0, 1, 2, 3))
        ktxt = f'SPY phase today: <b style="color:{PH_COL[int(ph[-1])]}">{today.upper()}</b>   {key}'
    else:
        ktxt = '<span style="color:#9b958f">SPY phase not available right now (candles in the Out colour)</span>'
    fig.add_annotation(xref='paper', yref='paper', x=0.0, y=1.0, xanchor='left', yanchor='top', showarrow=False,
                       text=ktxt, font=dict(size=10, color='#c9c1b7'), bgcolor='rgba(18,18,18,0.75)')

    # volume pane (4 Nov 2026): bars coloured by the candle, 50-day average line; its own axis under the price
    has_vol = 'Volume' in v.columns and pd.to_numeric(v['Volume'], errors='coerce').fillna(0).gt(0).any()
    if has_vol:
        vol = pd.to_numeric(v['Volume'], errors='coerce').fillna(0)
        va50 = pd.to_numeric(h['Volume'], errors='coerce').rolling(50).mean().iloc[-BARS:]
        up = (v['Close'] >= v['Open']).to_numpy()
        fig.add_trace(go.Bar(x=x, y=vol, yaxis='y2', name='Volume', showlegend=False,
                             marker=dict(color=np.where(up, 'rgba(133,202,197,0.45)', 'rgba(244,160,98,0.45)'), line_width=0),
                             hovertemplate='Volume %{y:,.0f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=x, y=va50, yaxis='y2', name='Vol 50d avg', mode='lines', showlegend=False,
                                 line=dict(color='#e8dccb', width=1, dash='dot'),
                                 hovertemplate='50-day avg %{y:,.0f}<extra></extra>'))
    # month ticks on the category axis: first session of each month
    first = ~v.index.to_period('M').duplicated()
    xs = pd.Index(x)
    tv = list(xs[first])[1:]
    tt = [d.strftime('%b %Y') if d.month == 1 else d.strftime('%b') for d in v.index[first]][1:]
    fig.update_layout(
        template='plotly_dark+atn', height=height, margin=dict(l=8, r=56, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(type='category', rangeslider=dict(visible=False), showgrid=False, tickangle=0,
                   tickmode='array', tickvals=tv, ticktext=tt, tickfont=dict(size=10)),
        yaxis=dict(side='right', tickfont=dict(size=10), gridcolor='#1f1d1b', domain=[0.22, 1.0] if has_vol else [0, 1]),
        bargap=0.15,
        legend=dict(orientation='h', x=0, y=1.0, yanchor='bottom', font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        hovermode='x', spikedistance=-1, dragmode='zoom')
    # crosshair (4 Nov 2026): dotted vertical and horizontal lines that follow the cursor, date label on the x axis
    _spk = dict(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=0.5, spikedash='dot',
                spikecolor='rgba(168,159,149,0.55)')
    fig.update_xaxes(**_spk)
    fig.update_yaxes(**_spk)
    if has_vol:
        fig.update_layout(yaxis2=dict(side='right', domain=[0, 0.17], showgrid=False, nticks=3, tickformat='.2s', showspikes=False,
                                      tickfont=dict(size=9, color='#9b958f'), rangemode='tozero', anchor='x'),
                          xaxis=dict(anchor='y2'))                 # month labels under the volume pane
    return fig



# ── Chart that keeps your zoom when you switch tickers (same behaviour as Gamma Live) ──────────────────────────────
# Streamlit is always given the chart's FIRST figure (so the chart is never removed and redrawn), and each new ticker's
# figure is pushed into it in the browser with Plotly.react. Your time window (x range) is remembered as "bars from the
# right edge", so it carries over to the next ticker; the price axis is re-fitted to the candles inside that window.
# Box-drag = zoom (both axes), mouse wheel = zoom, double-click = full 12 months again.
CHART_JS = """<script>
(function () {
  const P = window.parent, D = P.document, KEY = %s, VER = %s, FIG = %s, HI = %s, LO = %s;
  const SK = 'mrs_zoom_' + KEY;
  P.__mrsFigs = P.__mrsFigs || {};
  P.__mrsFigs[KEY] = {ver: VER, fig: FIG, hi: HI, lo: LO};
  function find() {
    for (const g of D.querySelectorAll('.js-plotly-plot')) {
      if (g.layout && typeof g.layout.meta === 'string' && g.layout.meta.split('|')[0] === KEY) return g;
    }
    return null;
  }
  function load() { try { return JSON.parse(P.sessionStorage.getItem(SK) || 'null'); } catch (e) { return null; } }
  function fit(cur, r) {                 // price range of the candles inside x range r (category index units)
    const n = cur.hi.length, a = Math.max(0, Math.ceil(r[0])), b = Math.min(n - 1, Math.floor(r[1]));
    let lo = Infinity, hi = -Infinity;
    for (let i = a; i <= b; i++) { if (cur.lo[i] < lo) lo = cur.lo[i]; if (cur.hi[i] > hi) hi = cur.hi[i]; }
    if (!isFinite(lo) || !isFinite(hi)) return null;
    const pad = (hi - lo) * 0.06 || hi * 0.01;
    return [lo - pad, hi + pad];
  }
  function apply(g) {
    const cur = P.__mrsFigs[KEY];
    if (!cur || !P.Plotly || !g._fullLayout) return;
    hook(g);
    if (g.layout.meta === KEY + '|' + cur.ver) return;
    const lay = Object.assign({}, cur.fig.layout,
      {width: g._fullLayout.width, height: g._fullLayout.height, meta: KEY + '|' + cur.ver});
    const z = load(), n = cur.hi.length;
    if (z && n > 1) {                    // saved window = bars counted from the right edge
      const r = [n - 1 - z.a, n - 1 - z.b];
      lay.xaxis = Object.assign({}, lay.xaxis || {}, {range: r, autorange: false});
      const y = fit(cur, r);
      if (y) lay.yaxis = Object.assign({}, lay.yaxis || {}, {range: y, autorange: false});
    }
    g.__applying = true;
    P.Plotly.react(g, cur.fig.data, lay).then(function () { g.__applying = false; }, function () { g.__applying = false; });
  }
  function hook(g) {
    if (!g.__mrsHook) {
      g.__mrsHook = true;
      g.on('plotly_afterplot', function () { setTimeout(function () { apply(g); }, 0); });
      g.on('plotly_relayout', function (ev) {
        if (g.__applying || !ev) return;
        if (ev['xaxis.autorange']) { try { P.sessionStorage.removeItem(SK); } catch (e) {} return; }
        const xr = ev['xaxis.range'] || (ev['xaxis.range[0]'] !== undefined ? [ev['xaxis.range[0]'], ev['xaxis.range[1]']] : null);
        if (!xr) return;
        const c2 = P.__mrsFigs[KEY], n2 = c2.hi.length, r = [+xr[0], +xr[1]];
        try { P.sessionStorage.setItem(SK, JSON.stringify({a: n2 - 1 - r[0], b: n2 - 1 - r[1]})); } catch (e) {}
        const yGiven = ev['yaxis.range'] || ev['yaxis.range[0]'] !== undefined;
        if (!yGiven) {                   // panned / zoomed in time only: re-fit the price axis to the visible candles
          const y = fit(c2, r);
          if (y) { g.__applying = true; P.Plotly.relayout(g, {'yaxis.range': y, 'yaxis.autorange': false})
                     .then(function () { g.__applying = false; }, function () { g.__applying = false; }); }
        }
      });
    }
  }
  // keep checking: Streamlit can redraw the chart element (first load, resize), and the new element needs the hooks too
  setInterval(function () { const g = find(); if (g && g._fullLayout) apply(g); }, 300);
})();
</script>"""


def show(st, components, fig, key: str, config: dict):
    """st.plotly_chart that keeps the zoom across tickers and refreshes in place (see CHART_JS)."""
    import json
    import time
    ver = str(time.time_ns())
    k = f"{key}_{int(fig.layout.height or 0)}"
    fig.update_layout(meta=f'{k}|{ver}')
    first = st.session_state.setdefault('_mrs_first_fig', {})
    if k not in first:
        first[k] = fig
    hi_a = lo_a = None
    for tr_ in fig.data:                       # v6.5: one candle trace per SPY phase; combine them bar by bar
        if tr_.type == 'candlestick':
            a_ = np.array(tr_.high, dtype=float)
            b_ = np.array(tr_.low, dtype=float)
            hi_a = a_ if hi_a is None else np.fmax(hi_a, a_)
            lo_a = b_ if lo_a is None else np.fmin(lo_a, b_)
    hi = [float(v) for v in pd.Series(hi_a).ffill().bfill()]
    lo = [float(v) for v in pd.Series(lo_a).ffill().bfill()]
    st.plotly_chart(first[k], use_container_width=True, key=k, config=config, theme=None)
    components.html(CHART_JS % (json.dumps(k), json.dumps(ver), fig.to_json(), json.dumps(hi), json.dumps(lo)), height=0)
