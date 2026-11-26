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
v8.3.0 (15 Nov 2026): ATR(14) zones from the SMA 50 as faint shading (no lines), the 21D high visible over the last
  60 sessions with a label, as in the TradingView Ext·RS v8.0 indicator.
v8 (15 Nov 2026): candles teal up / orange down again (no SPY phase colours); EMA 21 light teal, SMA 50 bright
  grey, SMA 200 orange, SMA 250 dashed orange. The rest of v6.5 stays (ATR lines, markers).
v6.5 (14 Nov 2026): drawn like the TradingView "Ext·RS" indicator v7.5 (see figure()): candles in the SPY phase colour,
  SMA 50 / 200 / 250, SMA 50 +/- 2..5 ATR lines, Deep dip / chase / R1 markers. The EMA 10 and the 52W / 21D high lines
  above are replaced (EMA 21 and the highs are in the legend).
Descriptive chart only: same data the scanner uses, no signals.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

UP, DOWN = '#85cac5', '#f4a062'
HOLLOW = 'rgba(0,0,0,0)'


def hollow_classes(o, c, prev) -> dict:
    """8.13.0 (26 Nov 2026): TradingView hollow candles. Colour = close vs the prior close (teal up, orange down);
    body = hollow when the close is at or above the open, filled when below. Returns four boolean masks."""
    o, c, prev = (np.asarray(a, dtype=float) for a in (o, c, prev))
    prev = np.where(np.isfinite(prev), prev, o)
    up = c >= prev
    hol = c >= o
    return {'hollow_up': hol & up, 'filled_up': ~hol & up, 'hollow_down': hol & ~up, 'filled_down': ~hol & ~up}
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
S50_COL, S200_COL, E21_COL = '#e6e6e6', '#f4a062', '#a6e3de'   # v8 (15 Nov 2026): SMA 50 bright grey, SMA 200/250 orange, EMA 21 light teal
ATR_UP_COL, ATR_DN_COL = '#b9805a', '#649995'
ATR_UP, ATR_DN, ATR_N = (2, 3, 4, 5), (2, 3, 4, 5), 20  # SMA 50 + / - k ATR(14), drawn over the last 20 sessions
MK_COL = dict(dd='#76ff03', dw='#9e9e9e', ch='#ff5252', os='#ffeb3b')
FUTURE = 9                                              # empty sessions at the right edge, for the ATR labels
ZONE_W = 3                                              # 8.13.1: width of the ATR strip, in empty sessions
ZN_DN, ZN_UP = '100,153,149', '185,128,90'                # v8.3.0 zone fill colours (rgb of #649995 / #b9805a)
H21_N = 60                                              # v8.3.0: 21D high line over the last 60 sessions


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

    # 8.13.1 (26 Nov 2026, Fabio): ATR(14) zones from the SMA 50 as a short strip in the empty sessions right of the last
    #    candle (as in the TradingView Ext·RS indicator): +2..+5 above, -1..-5 below, darker further out; levels of the
    #    last session. Labels sit right of the strip (below). Shapes, drawn under everything.
    _zs = []
    for lo_k, hi_k, a in ((2, 3, 0.14), (3, 5, 0.24), (-2, -1, 0.14), (-3, -2, 0.19), (-5, -3, 0.26)):
        ylo = float((s50 + lo_k * a14).iloc[-1]); yhi = float((s50 + hi_k * a14).iloc[-1])
        if np.isfinite(ylo) and np.isfinite(yhi):
            _zs.append(dict(type='rect', xref='x', yref='y', x0=xf[0], x1=xf[ZONE_W], y0=ylo, y1=yhi, layer='below',
                            line=dict(width=0), fillcolor=f'rgba(168,159,149,{a})'))
    for k_out in ((2, 5), (-1, -5)):                   # thin outline around the upper and lower block
        ya_ = float((s50 + k_out[0] * a14).iloc[-1]); yb_ = float((s50 + k_out[1] * a14).iloc[-1])
        if np.isfinite(ya_) and np.isfinite(yb_):
            _zs.append(dict(type='rect', xref='x', yref='y', x0=xf[0], x1=xf[ZONE_W], y0=ya_, y1=yb_, layer='below',
                            line=dict(width=0.6, color='rgba(168,159,149,0.35)'), fillcolor='rgba(0,0,0,0)'))
    for z_ in _zs:
        fig.add_shape(**z_)

    # candles: teal up / orange down (v8, 15 Nov 2026: the SPY phase colours are gone; the phase lives on the MRS card)
    ph_all = spy_phase(spy)                     # still used for the breakout-in-dip-zone marker
    if ph_all is not None:
        ph = ph_all.reindex(v.index).fillna(3).astype(int).to_numpy()
    else:
        ph = np.full(len(v), 3)
    # 8.13.0 (26 Nov 2026, Fabio): TradingView hollow candles, one trace per class (colour by the prior close,
    #    hollow body when close >= open); show() already combines several candle traces bar by bar
    for k_, m_ in hollow_classes(v['Open'], v['Close'], pc.iloc[-BARS:]).items():
        if not m_.any():
            continue
        col_ = UP if k_.endswith('up') else DOWN
        sty_ = dict(line=dict(color=col_, width=1), fillcolor=HOLLOW if k_.startswith('hollow') else col_)
        xs_ = [x[i] for i in np.flatnonzero(m_)]      # 8.13.2: only this class's bars (NaN bars drew broken paths)
        fig.add_trace(go.Candlestick(
            x=xs_, open=v['Open'][m_], high=v['High'][m_], low=v['Low'][m_],
            close=v['Close'][m_], name=ticker, showlegend=False, increasing=sty_, decreasing=sty_))
    fig.add_trace(go.Scatter(x=xf, y=[float(c.iloc[-1])] * FUTURE, mode='markers', marker=dict(opacity=0),
                             hoverinfo='skip', showlegend=False))

    # moving averages (v8, 15 Nov 2026: Fabio's original set): EMA 21 light teal, SMA 50 bright grey,
    # SMA 200 orange, SMA 250 dashed orange
    fig.add_trace(go.Scatter(x=x, y=e21.iloc[-BARS:], name='EMA 21', mode='lines',
                             line=dict(color=E21_COL, width=1.3), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=s50.iloc[-BARS:], name='SMA 50', mode='lines',
                             line=dict(color=S50_COL, width=1.3), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=s200.iloc[-BARS:], name='SMA 200', mode='lines',
                             line=dict(color=S200_COL, width=1.5), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=s250.iloc[-BARS:], name='SMA 250', mode='lines',
                             line=dict(color=S200_COL, width=1.2, dash='dash'), hoverinfo='skip'))
    if np.isfinite(lv['hi52']):
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[lv['hi52']] * 2, name=f'52W hi {lv["hi52"]:,.2f}', mode='lines',
                                 visible='legendonly', line=dict(color=HI52_COL, width=1, dash='dash'), hoverinfo='skip'))
    if np.isfinite(lv['hi21']):     # v8.3.0: visible, dotted, last H21_N sessions (the breakout level), as in TradingView
        xh = x[-H21_N:]
        fig.add_trace(go.Scatter(x=xh, y=[lv['hi21']] * len(xh), name=f'21D hi {lv["hi21"]:,.2f}', mode='lines',
                                 line=dict(color=HI20_COL, width=1, dash='dot'), hoverinfo='skip'))

    # v8.3.0: zone labels at the right edge (the dotted ATR lines are gone) and the 21D high label
    lab_x, lab_y, lab_t, lab_c = [], [], [], []
    for k in (-1, -2, -3, -5, 2, 3, 5):
        yk = float((s50 + k * a14).iloc[-1])
        if np.isfinite(yk):
            lab_x.append(xf[ZONE_W]); lab_y.append(yk); lab_t.append(f'{"+" if k > 0 else "−"}{abs(k)} ATR')
            lab_c.append(ATR_UP_COL if k > 0 else ATR_DN_COL)
    if np.isfinite(lv['hi21']):
        lab_x.append(xf[ZONE_W]); lab_y.append(float(lv['hi21'])); lab_t.append(f'21D {lv["hi21"]:,.2f}'); lab_c.append(HI20_COL)
    if lab_x:
        fig.add_trace(go.Scatter(x=lab_x, y=lab_y, mode='text', text=lab_t, textposition='middle right',
                                 textfont=dict(size=9, color=lab_c), showlegend=False,     # 8.14.1: no legend group (it stacked the legend)
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
    # 8.14.1: short legend names (full text in the hover) so the legend stays on one or two rows
    for key, mask, sym, ypos, name, tip in (
            ('dd', dd & ~wide & vis, 'triangle-up', ll - 0.6 * av, 'Deep dip',
             'Deep dip trigger (undercut / reclaim of the 200 / 250 SMA after 20 closes above)'),
            ('dw', dd & wide & vis, 'triangle-up', ll - 0.6 * av, 'Deep dip, stop > 10%',
             'Deep dip, but the 2 ATR(14) stop is more than 10% away: about 0R in S1'),
            ('ch', chase & vis, 'circle', hh + 0.6 * av, 'Dip-zone breakout',
             'First close above the 21-day high while SPY is in DIP ZONE / RECOVERY: lagged a random entry (S2)'),
            ('os', os50 & vis, 'x', ll - 1.3 * av, '50d reclaim, RSI < 35',
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

    # (v8, 15 Nov 2026: the SPY phase key is removed with the phase colours)

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
    # 8.13.1 (26 Nov 2026, Fabio): price line at the last close across the chart and the price on the axis, coloured by
    #    the last candle (vs the prior close), as in TradingView
    _lc, _pc = float(c.iloc[-1]), float(pc.iloc[-1]) if np.isfinite(pc.iloc[-1]) else float(h['Open'].iloc[-1])
    _col = UP if _lc >= _pc else DOWN
    fig.add_shape(type='line', xref='paper', yref='y', x0=0, x1=1, y0=_lc, y1=_lc, layer='below',
                  line=dict(color=_col, width=0.8, dash='dot'))
    fig.add_annotation(xref='paper', yref='y', x=1, y=_lc, text=f'{_lc:,.2f}', showarrow=False, xanchor='left',
                       yanchor='middle', xshift=2, bgcolor=_col, borderpad=2, font=dict(size=10, color='#0a0a0a'))

    # month ticks on the category axis: first session of each month
    first = ~v.index.to_period('M').duplicated()
    xs = pd.Index(x)
    tv = list(xs[first])[1:]
    tt = [d.strftime('%b %Y') if d.month == 1 else d.strftime('%b') for d in v.index[first]][1:]
    fig.update_layout(
        template='plotly_dark+atn', height=height, margin=dict(l=8, r=56, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(type='category', categoryorder='array', categoryarray=x + xf,   # 8.13.2: order fixed (class traces)
                   rangeslider=dict(visible=False), showgrid=False, tickangle=0,
                   tickmode='array', tickvals=tv, ticktext=tt, tickfont=dict(size=10)),
        yaxis=dict(side='right', tickfont=dict(size=10), gridcolor='#1f1d1b', domain=[0.22, 1.0] if has_vol else [0, 1]),
        bargap=0.15,
        legend=dict(orientation='h', x=0, y=1.0, yanchor='bottom', font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                    traceorder='normal'),   # 8.14.1: one row; a legend group puts every entry on its own row
        hovermode='x', spikedistance=-1, dragmode='zoom')
    # crosshair (4 Nov 2026): dotted vertical and horizontal lines that follow the cursor, date label on the x axis
    # 8.13.1: Plotly draws its spike lines with a 2 px contrasting outline, so they can't be made thin. The crosshair
    #    (hairlines + the price on the price axis) is drawn by CHART_JS; Plotly keeps only the date label (hovermode x).
    _spk = dict(showspikes=False)
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
  const ME = {};                         // 8.13.2: this script run; each Streamlit rerun replaces the iframe
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
  function cross(g) {                    // 8.13.1 / 8.13.2: TradingView crosshair: hairlines + the price on the price axis
    if (g.__mrsCrossOwner === ME) return;  // 8.13.2: re-attach after every rerun (the old iframe's handlers die with it)
    if (g.__mrsCrossOff) { try { g.__mrsCrossOff(); } catch (e) {} }
    for (const old of Array.from(g.querySelectorAll('.mrs-x'))) { try { old.remove(); } catch (e) {} }
    g.__mrsCrossOwner = ME;
    const mk = function (css) { const d = D.createElement('div'); d.className = 'mrs-x';
      d.style.cssText = 'position:absolute;display:none;pointer-events:none;z-index:1000;' + css; return d; };
    const V = mk('width:0;border-left:1px solid rgba(200,192,183,0.55);');
    const H = mk('height:0;border-top:1px solid rgba(200,192,183,0.55);');
    const L = mk('font:11px IBM Plex Mono,monospace;background:#3a3631;color:#f3ece2;padding:1px 5px;border-radius:2px;' +
                 'white-space:nowrap;');
    const hide = function () { V.style.display = H.style.display = L.style.display = 'none'; };
    const move = function (e) {
      const fl = g._fullLayout;
      if (!fl || !fl.yaxis || !fl.xaxis || !fl._size) return;
      if (L.parentNode !== g) {
        if (P.getComputedStyle(g).position === 'static') g.style.position = 'relative';
        g.appendChild(V); g.appendChild(H); g.appendChild(L);
      }
      const ya = fl.yaxis, xa = fl.xaxis, sz = fl._size, r = g.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      if (px < xa._offset || px > xa._offset + xa._length || py < sz.t || py > sz.t + sz.h) { hide(); return; }
      V.style.left = Math.round(px) + 'px'; V.style.top = sz.t + 'px'; V.style.height = sz.h + 'px'; V.style.display = 'block';
      if (py < ya._offset || py > ya._offset + ya._length) { H.style.display = L.style.display = 'none'; return; }
      H.style.left = xa._offset + 'px'; H.style.width = xa._length + 'px'; H.style.top = Math.round(py) + 'px';
      H.style.display = 'block';
      const v = ya.p2l(py - ya._offset);
      if (!isFinite(v)) { L.style.display = 'none'; return; }
      L.textContent = v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
      L.style.left = (xa._offset + xa._length + 2) + 'px';
      L.style.top = (py - 8) + 'px';
      L.style.display = 'block';
    };
    g.addEventListener('mousemove', move, true);
    g.addEventListener('mouseleave', hide);
    g.__mrsCrossOff = function () { g.removeEventListener('mousemove', move, true); g.removeEventListener('mouseleave', hide); };
  }
  function lines(g) {                    // 8.13.2 horizontal lines (Alt + click); 8.14.0 trend lines (Shift + click twice)
    if (g.__mrsLinesOwner === ME) return;
    if (g.__mrsLinesOff) { try { g.__mrsLinesOff(); } catch (e) {} }
    for (const old of Array.from(g.querySelectorAll('.mrs-hl'))) { try { old.remove(); } catch (e) {} }
    g.__mrsLinesOwner = ME;
    const COL = '#e3b48e', NS = 'http://www.w3.org/2000/svg';
    const ld = function (k) { try { return JSON.parse(P.localStorage.getItem(k) || '{}') || {}; } catch (e) { return {}; } };
    const sv = function (k, o) { try { P.localStorage.setItem(k, JSON.stringify(o)); } catch (e) {} };
    const tick = function () { const d = (g.data || []).find(function (t) { return t.type === 'candlestick'; });
                               return d ? String(d.name) : null; };
    const fmt = function (v) { return v.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}); };
    const box = D.createElement('div');
    box.className = 'mrs-hl';
    box.style.cssText = 'position:absolute;left:0;top:0;width:0;height:0;pointer-events:none;z-index:900;';
    const svg = D.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'mrs-hl');
    svg.style.cssText = 'position:absolute;overflow:hidden;pointer-events:none;z-index:899;';
    const rub = D.createElementNS(NS, 'line');
    rub.setAttribute('stroke', COL); rub.setAttribute('stroke-width', '1'); rub.setAttribute('stroke-dasharray', '4,3');
    rub.style.display = 'none';
    let pend = null;
    const ax = function () { const fl = g._fullLayout; return fl && fl.xaxis && fl.yaxis ? [fl.xaxis, fl.yaxis] : null; };
    const cats = function (xa) { return xa._categories || []; };
    const lastReal = function (cs) { for (let i = cs.length - 1; i >= 0; i--) { if (String(cs[i]).trim() !== '') return i; } return -1; };
    // a point = {d: date of the session (or of the last candle), n: sessions after it (empty space at the right), p: price}
    const toPt = function (xa, ya, px, py) {
      const cs = cats(xa), lr = lastReal(cs);
      let i = Math.round(xa.p2l(px - xa._offset));
      i = Math.max(0, Math.min(cs.length - 1, i));
      const p = Math.round(ya.p2l(py - ya._offset) * 100) / 100;
      return i <= lr ? {d: String(cs[i]), n: 0, p: p} : {d: String(cs[lr]), n: i - lr, p: p};
    };
    const toPx = function (xa, ya, pt) {
      const i = cats(xa).indexOf(pt.d);
      if (i < 0) return null;
      return [xa.l2p(i + (pt.n || 0)), ya.l2p(pt.p)];
    };
    const inPane = function (xa, ya, px, py) {
      return px >= xa._offset && px <= xa._offset + xa._length && py >= ya._offset && py <= ya._offset + ya._length;
    };
    const attach = function () {
      if (box.parentNode !== g) {
        if (P.getComputedStyle(g).position === 'static') g.style.position = 'relative';
        g.appendChild(svg); g.appendChild(box); svg.appendChild(rub);
      }
    };
    const tag = function (left, top, text, title, onclick) {
      const lb = D.createElement('div');
      lb.textContent = text; lb.title = title;
      lb.style.cssText = 'position:absolute;left:' + left + 'px;top:' + top + 'px;font:11px IBM Plex Mono,monospace;' +
                         'background:' + COL + ';color:#0a0a0a;padding:1px 4px;border-radius:2px;cursor:pointer;' +
                         'pointer-events:auto;white-space:nowrap;';
      lb.addEventListener('click', function (ev) { ev.stopPropagation(); onclick(); });
      box.appendChild(lb);
    };
    const draw = function () {
      const a = ax(); if (!a) return;
      const xa = a[0], ya = a[1], t = tick();
      attach();
      const hl = ld('mrs_hlines')[t] || [], tl = ld('mrs_tlines')[t] || [];
      const sig = t + '|' + JSON.stringify(hl) + '|' + JSON.stringify(tl) + '|' + (ya.range || []).join(',') + '|' +
                  (xa.range || []).join(',') + '|' + xa._offset + '|' + xa._length + '|' + ya._offset + '|' + ya._length;
      if (box.__sig === sig) return;
      box.__sig = sig;
      box.innerHTML = '';
      for (const el of Array.from(svg.querySelectorAll('.mrs-tl'))) el.remove();
      svg.style.left = xa._offset + 'px'; svg.style.top = ya._offset + 'px';
      svg.setAttribute('width', xa._length); svg.setAttribute('height', ya._length);
      hl.forEach(function (p) {                     // horizontal lines: dashed, price tag on the axis
        const y = ya.l2p(p);
        if (!isFinite(y) || y < 0 || y > ya._length) return;
        const ln = D.createElement('div');
        ln.style.cssText = 'position:absolute;left:' + xa._offset + 'px;width:' + xa._length + 'px;top:' +
                           Math.round(ya._offset + y) + 'px;height:0;border-top:1px dashed ' + COL + ';pointer-events:none;';
        box.appendChild(ln);
        tag(xa._offset + xa._length + 2, Math.round(ya._offset + y) - 8, fmt(p) + '  ×', 'Click to remove this line', function () {
          const o = ld('mrs_hlines'); o[t] = (o[t] || []).filter(function (q) { return q !== p; });
          if (!o[t].length) delete o[t];
          sv('mrs_hlines', o); box.__sig = null; draw();
        });
      });
      tl.forEach(function (ln_, j) {                // trend lines: solid segment between the two points, × at the end
        const A = toPx(xa, ya, ln_.a), B = toPx(xa, ya, ln_.b);
        if (!A || !B) return;
        const e = D.createElementNS(NS, 'line');
        e.setAttribute('class', 'mrs-tl');
        e.setAttribute('x1', A[0]); e.setAttribute('y1', A[1]); e.setAttribute('x2', B[0]); e.setAttribute('y2', B[1]);
        e.setAttribute('stroke', COL); e.setAttribute('stroke-width', '1.3');
        svg.insertBefore(e, rub);
        const end = [B, A].find(function (q) { return q[0] >= 0 && q[0] <= xa._length && q[1] >= 0 && q[1] <= ya._length; });
        if (end) {
          tag(xa._offset + end[0] + 4, ya._offset + end[1] - 8, '×', 'Click to remove this trend line', function () {
            const o = ld('mrs_tlines'); o[t] = (o[t] || []).filter(function (_, k) { return k !== j; });
            if (!o[t].length) delete o[t];
            sv('mrs_tlines', o); box.__sig = null; draw();
          });
        }
      });
    };
    const click = function (e) {
      if (!e.altKey && !e.shiftKey) return;
      const a = ax(); if (!a) return;
      const xa = a[0], ya = a[1], t = tick(), r = g.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      if (!t || !inPane(xa, ya, px, py)) return;
      e.preventDefault(); e.stopPropagation();
      if (e.altKey) {                               // horizontal line
        const v = Math.round(ya.p2l(py - ya._offset) * 100) / 100;
        if (!isFinite(v)) return;
        const o = ld('mrs_hlines'); o[t] = (o[t] || []).concat([v]); sv('mrs_hlines', o);
      } else {                                      // trend line: first click = start, second click = end
        const pt = toPt(xa, ya, px, py);
        if (!isFinite(pt.p)) return;
        if (!pend || pend.t !== t) { pend = {t: t, a: pt}; return; }
        const o = ld('mrs_tlines'); o[t] = (o[t] || []).concat([{a: pend.a, b: pt}]); sv('mrs_tlines', o);
        pend = null; rub.style.display = 'none';
      }
      box.__sig = null; draw();
    };
    const move = function (e) {                     // rubber band while a trend line is being drawn
      if (!pend) return;
      const a = ax(); if (!a) return;
      const xa = a[0], ya = a[1], A = toPx(xa, ya, pend.a);
      if (!A || tick() !== pend.t) { rub.style.display = 'none'; return; }
      const r = g.getBoundingClientRect();
      rub.setAttribute('x1', A[0]); rub.setAttribute('y1', A[1]);
      rub.setAttribute('x2', e.clientX - r.left - xa._offset); rub.setAttribute('y2', e.clientY - r.top - ya._offset);
      rub.style.display = 'inline';
    };
    const esc = function (e) { if (e.key === 'Escape' && pend) { pend = null; rub.style.display = 'none'; } };
    g.addEventListener('click', click, true);
    g.addEventListener('mousemove', move, true);
    D.addEventListener('keydown', esc);
    const iv = setInterval(draw, 250);       // follows zoom, pan, resize and ticker changes
    g.__mrsLinesOff = function () { g.removeEventListener('click', click, true); g.removeEventListener('mousemove', move, true);
                                    D.removeEventListener('keydown', esc); clearInterval(iv); };
  }
  function hook(g) {
    cross(g);
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
  setInterval(function () { const g = find(); if (!g) return; cross(g); lines(g); if (g._fullLayout) apply(g); }, 300);   // 8.13.2: crosshair and lines even without window.Plotly
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
    # 8.13.2: the candle classes are separate traces with their own bars; put each bar's high / low back in its slot on the
    #    category axis (the zoom fit in CHART_JS indexes by category position)
    cats = list(fig.layout.xaxis.categoryarray or [])
    pos = {c_: i for i, c_ in enumerate(cats)}
    hi_a = np.full(len(cats), np.nan); lo_a = np.full(len(cats), np.nan)
    for tr_ in fig.data:
        if tr_.type == 'candlestick':
            for xv, a_, b_ in zip(tr_.x, tr_.high, tr_.low):
                i = pos.get(xv)
                if i is not None:
                    hi_a[i] = np.fmax(hi_a[i], float(a_)); lo_a[i] = np.fmin(lo_a[i], float(b_))
    n_c = int(np.flatnonzero(np.isfinite(hi_a)).max()) + 1 if np.isfinite(hi_a).any() else 0
    hi_a, lo_a = hi_a[:n_c], lo_a[:n_c]                 # the empty sessions at the right edge are not candles
    hi = [float(v) for v in pd.Series(hi_a).ffill().bfill()]
    lo = [float(v) for v in pd.Series(lo_a).ffill().bfill()]
    st.plotly_chart(first[k], use_container_width=True, key=k, config=config, theme=None)
    components.html(CHART_JS % (json.dumps(k), json.dumps(ver), fig.to_json(), json.dumps(hi), json.dumps(lo)), height=0)
    st.markdown('<div style="font-size:0.68rem;color:#6e675f;margin:-6px 0 4px 0;">Drawing: Alt (Option) + click = '
                'horizontal line · Shift + click twice = trend line (Esc cancels) · click a tag or × to remove · '
                'kept per ticker in this browser</div>', unsafe_allow_html=True)        # 8.13.2 / 8.14.0
