"""
chart_view.py — daily candle chart for the scanner's side panel
================================================================
One ticker at a time, fetched on demand (yfinance, per-ticker history like the rest of the app) and cached.
Shows the last ~12 months of daily candles with:
  SMA 50 and SMA 200 (computed on the full 2-year history, so the 200-day line starts at the left edge)
  52-week high  = highest high of the last 252 sessions (including today)
  21-day high   = highest high of the 21 sessions BEFORE today (the pivot a breakout has to clear; 20 until 3 Nov 2026)
  Volume pane   (4 Nov 2026) daily volume bars (teal up / orange down day) and the 50-day average, under the price
  Base overlay  (3 Nov 2026) the current base from StockScanner/bases.py on this chart's own (unadjusted) prices:
                shaded base, the cup's left lip / low / right lip, the handle, and the pivot line
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
    if typ == 'Cup w/ handle':
        pts = [(b.get('base_left_date'), b.get('base_left')), (b.get('base_low_date'), lo),
               (b.get('base_right_date'), pv - 0.10)]
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
                                 textfont=dict(size=9, color='#9b958f'), hoverinfo='skip', showlegend=False))


def figure(h: pd.DataFrame, ticker: str, height: int = 520, base: dict = None) -> go.Figure:
    """base: bases.detect(h) output to draw (computed here when None; pass {} to draw none)."""
    c = h['Close']
    s50 = c.rolling(50).mean()
    s200 = c.rolling(200).mean()
    v = h.iloc[-BARS:]
    x = v.index.strftime('%Y-%m-%d')          # category axis: no gaps on weekends and holidays
    lv = levels(h)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=v['Open'], high=v['High'], low=v['Low'], close=v['Close'], name=ticker,
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
        showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=s50.iloc[-BARS:], name='SMA 50', mode='lines',
                             line=dict(color=SMA50_COL, width=1.3), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x, y=s200.iloc[-BARS:], name='SMA 200', mode='lines',
                             line=dict(color=SMA200_COL, width=1.5), hoverinfo='skip'))
    hi52, hi20 = lv['hi52'], lv['hi20']
    same = (not np.isnan(hi52)) and (not np.isnan(hi20)) and abs(hi52 / hi20 - 1) < 0.003
    lines = [(hi52, HI52_COL, 'dash', '52W & 21D high' if same else '52W high', 'top left')]
    if not same:
        lines.append((hi20, HI20_COL, 'dot', '21D high', 'bottom left'))
    for y, col, dash, lbl, pos in lines:
        if not np.isnan(y):
            fig.add_hline(y=y, line=dict(color=col, width=1, dash=dash),
                          annotation=dict(text=f'{lbl} {y:,.2f}', font=dict(size=10, color=col), bgcolor='rgba(18,18,18,0.75)'),
                          annotation_position=pos)
    _add_base(fig, v, base_info(h) if base is None else base)
    _add_pivots(fig, h, v)
    # volume pane (4 Nov 2026): bars coloured by the candle, 50-day average line; its own axis under the price
    has_vol = 'Volume' in v.columns and pd.to_numeric(v['Volume'], errors='coerce').fillna(0).gt(0).any()
    if has_vol:
        vol = pd.to_numeric(v['Volume'], errors='coerce').fillna(0)
        va50 = pd.to_numeric(h['Volume'], errors='coerce').rolling(50).mean().iloc[-BARS:]
        up = (v['Close'] >= v['Open']).to_numpy()
        fig.add_trace(go.Bar(x=x, y=vol, yaxis='y2', name='Volume', showlegend=False,
                             marker=dict(color=np.where(up, 'rgba(133,202,197,0.55)', 'rgba(244,160,98,0.55)'), line_width=0),
                             hovertemplate='Volume %{y:,.0f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=x, y=va50, yaxis='y2', name='Vol 50d avg', mode='lines', showlegend=False,
                                 line=dict(color='#e8dccb', width=1, dash='dot'),
                                 hovertemplate='50-day avg %{y:,.0f}<extra></extra>'))
    # month ticks on the category axis: first session of each month
    first = ~v.index.to_period('M').duplicated()
    tv = list(x[first])[1:]
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
    _spk = dict(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikedash='dot',
                spikecolor='#9b958f')
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
    c = fig.data[0]
    hi = [float(v) for v in c.high]
    lo = [float(v) for v in c.low]
    st.plotly_chart(first[k], use_container_width=True, key=k, config=config, theme=None)
    components.html(CHART_JS % (json.dumps(k), json.dumps(ver), fig.to_json(), json.dumps(hi), json.dumps(lo)), height=0)
