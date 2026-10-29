"""
chart_view.py — daily candle chart for the scanner's side panel
================================================================
One ticker at a time, fetched on demand (yfinance, per-ticker history like the rest of the app) and cached.
Shows the last ~12 months of daily candles with:
  SMA 50 and SMA 200 (computed on the full 2-year history, so the 200-day line starts at the left edge)
  52-week high  = highest high of the last 252 sessions (including today)
  20-day high   = highest high of the 20 sessions BEFORE today (the pivot a breakout has to clear)
Descriptive chart only: same data the scanner uses, no signals.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

UP, DOWN = '#85cac5', '#f4a062'
SMA50_COL, SMA200_COL = '#e8dccb', '#b9805a'
HI52_COL, HI20_COL = '#e3b48e', '#9b958f'
BARS = 252


def fetch(ticker: str):
    """10 years of daily OHLC for one ticker, or None (the chart shows the last 12 months; the Ext·Mom table uses
    the whole history for its odds). Wrap in st.cache_data in the app."""
    import yfinance as yf
    try:
        h = yf.Ticker(ticker).history(period='10y', auto_adjust=False)
        if h is None or h.empty or 'Close' not in h.columns:
            return None
        h = h[['Open', 'High', 'Low', 'Close']].dropna()
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return h
    except Exception:
        return None


def levels(h: pd.DataFrame) -> dict:
    """Key levels from the full history (last bar = latest session)."""
    out = {'hi52': np.nan, 'hi20': np.nan, 'sma50': np.nan, 'sma200': np.nan, 'close': np.nan, 'chg': np.nan}
    if h is None or len(h) < 2:
        return out
    c = h['Close']
    out['close'] = float(c.iloc[-1])
    out['chg'] = float(c.iloc[-1] / c.iloc[-2] - 1)
    out['hi52'] = float(h['High'].iloc[-252:].max())
    if len(h) > 20:
        out['hi20'] = float(h['High'].iloc[-21:-1].max())
    if len(c) >= 50:
        out['sma50'] = float(c.rolling(50).mean().iloc[-1])
    if len(c) >= 200:
        out['sma200'] = float(c.rolling(200).mean().iloc[-1])
    return out


def figure(h: pd.DataFrame, ticker: str, height: int = 520) -> go.Figure:
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
    lines = [(hi52, HI52_COL, 'dash', '52W & 20D high' if same else '52W high', 'top left')]
    if not same:
        lines.append((hi20, HI20_COL, 'dot', '20D high', 'bottom left'))
    for y, col, dash, lbl, pos in lines:
        if not np.isnan(y):
            fig.add_hline(y=y, line=dict(color=col, width=1, dash=dash),
                          annotation=dict(text=f'{lbl} {y:,.2f}', font=dict(size=10, color=col), bgcolor='rgba(18,18,18,0.75)'),
                          annotation_position=pos)
    # month ticks on the category axis: first session of each month
    first = ~v.index.to_period('M').duplicated()
    tv = list(x[first])[1:]
    tt = [d.strftime('%b %Y') if d.month == 1 else d.strftime('%b') for d in v.index[first]][1:]
    fig.update_layout(
        template='plotly_dark+atn', height=height, margin=dict(l=8, r=56, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(type='category', rangeslider=dict(visible=False), showgrid=False, tickangle=0,
                   tickmode='array', tickvals=tv, ticktext=tt, tickfont=dict(size=10)),
        yaxis=dict(side='right', tickfont=dict(size=10), gridcolor='#1f1d1b'),
        legend=dict(orientation='h', x=0, y=1.0, yanchor='bottom', font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified', dragmode='zoom')
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
