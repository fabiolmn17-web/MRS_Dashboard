"""
allocation.py — "Allocation rule" card (MRS v6.2, 10 Nov 2026)
================================================================
Two independent entry tickets on SPY (and QQQ, with a caution), from the studies of 9-10 Nov 2026
(TRADING DESK/Studies/Rebound_Depth, Breakout_21d, Entry_Exit_Combo):

  Ticket A · Rebound   buy at the close the FIRST time a sell-off closes at least 1 ATR(14) below the 50-day SMA
                       (the sell-off resets once the close is back above the 50-day). Stop: entry close - 2 ATR(10).
                       Exit: 2 consecutive closes below the 50-day SMA, counted only after a first close above it.
  Ticket B · Breakout  buy at the close above the highest high of the prior 21 sessions. Same stop.
                       Exit: 2 consecutive closes below the 21-day EMA, counted only after a first close above it.
  Position             100% in the ETF while at least one ticket is open (never more than 100%); cash otherwise.

Prices: daily, split-adjusted but NOT dividend-adjusted (as on TradingView). ATRs are Wilder averages.
Evidence: in-sample backtests only (SPY 1995-2026, QQQ 1999-2026), descriptive. Forward tracking from 10 Nov 2026.
The replay below is the same logic as the study code; test_allocation.py checks it on a synthetic series.
"""
import html as _html
import math

import numpy as np
import pandas as pd

TEAL, TEAL_M, ORANGE, COPPER, INK, MUTED, DIM = '#85cac5', '#649995', '#f4a062', '#e3b48e', '#f3ece2', '#a89f95', '#6e675f'
GO_LIVE = pd.Timestamp('2026-11-10')
TICKERS = ('SPY', 'QQQ')
# in-sample reference numbers (Entry_Exit_Combo, 10 Nov 2026): CAGR, max DD, Sharpe vs buy-and-hold
REF = {'SPY': dict(period='1995-2026', cagr=10.2, dd=-31, sh=0.68, bh_cagr=9.2, bh_dd=-56, bh_sh=0.43, expo=68),
       'QQQ': dict(period='1999-2026', cagr=10.4, dd=-63, sh=0.55, bh_cagr=9.7, bh_dd=-83, bh_sh=0.40, expo=67)}
CAUTION = {'QQQ': 'Rebound ticket failed in 2000-02 on QQQ (bought every dip of an 80% bear market). Track only; '
                  'a filter is still to be tested.'}


def _e(s):
    return _html.escape(str(s), quote=True)


def _fin(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ── indicators and replay ──────────────────────────────────────────────────────────
def indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    d = ohlc.rename(columns=str.lower)[['open', 'high', 'low', 'close']].astype(float).dropna().sort_index().copy()
    pc = d['close'].shift()
    tr = pd.concat([d['high'] - d['low'], (d['high'] - pc).abs(), (d['low'] - pc).abs()], axis=1).max(axis=1)
    d['atr14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d['atr10'] = tr.ewm(alpha=1 / 10, adjust=False).mean()
    d['sma50'] = d['close'].rolling(50).mean()
    d['ema21'] = d['close'].ewm(span=21, adjust=False).mean()
    d['E'] = (d['close'] - d['sma50']) / d['atr14']
    d['hh21'] = d['high'].rolling(21).max().shift(1)
    return d


def rebound_entries(d: pd.DataFrame) -> list:
    """First close of each sell-off episode with E <= -1 (episode ends at the first close with E >= 0)."""
    E = d['E'].values; out = []; inep = False
    for i, e in enumerate(E):
        if not np.isfinite(e):
            continue
        if not inep and e <= -1:
            inep = True; out.append(i)
        elif inep and e >= 0:
            inep = False
    return out


def breakout_entries(d: pd.DataFrame) -> list:
    c, hh = d['close'].values, d['hh21'].values
    return [i for i in range(len(c)) if np.isfinite(hh[i]) and c[i] > hh[i]]


def _run_trade(d: pd.DataFrame, i: int, ma: np.ndarray):
    """Returns (exit index or None if still open, exit price, stop, armed, closes_below) for an entry at close i."""
    o, l, c, a10 = d['open'].values, d['low'].values, d['close'].values, d['atr10'].values
    stop = c[i] - 2 * a10[i]; armed = bool(c[i] > ma[i]); run = 0
    for j in range(i + 1, len(c)):
        if l[j] <= stop:
            return j, min(o[j], stop), stop, armed, run
        below = c[j] < ma[j]
        if armed and below:
            run += 1
            if run >= 2:
                return j, c[j], stop, armed, run
        else:
            run = 0
        if not below:
            armed = True
    return None, np.nan, stop, armed, run


def replay(d: pd.DataFrame, entries: list, ma: np.ndarray) -> list:
    """One position at a time; a new entry is allowed from the close of the exit day on (as in the study)."""
    trades, busy = [], -1
    for i in entries:
        if i < busy:
            continue
        j, px, stop, armed, run = _run_trade(d, i, ma)
        trades.append(dict(i=i, j=j, entry=d['close'].values[i], exit=px, stop=stop, armed=armed, run=run))
        if j is None:
            break
        busy = j
    return trades


def daily_returns(d: pd.DataFrame, tA: list, tB: list) -> pd.Series:
    """Rule return per session: ETF return while at least one ticket is open (stop-day fill at the stop), else 0."""
    c = d['close'].values; n = len(c)
    x = [np.zeros(n), np.zeros(n)]; r = [np.zeros(n), np.zeros(n)]
    for k, tr in enumerate((tA, tB)):
        for t in tr:
            j = t['j'] if t['j'] is not None else n - 1
            for m in range(t['i'] + 1, j + 1):
                px = t['exit'] if (t['j'] is not None and m == j) else c[m]
                x[k][m] = 1.0; r[k][m] = px / c[m - 1] - 1
    tot = x[0] + x[1]
    out = np.where(tot > 0, (r[0] + r[1]) / np.where(tot > 0, tot, 1), 0.0)
    return pd.Series(out, index=d.index)


def state(ohlc: pd.DataFrame, ticker: str) -> dict:
    d = indicators(ohlc)
    tA = replay(d, rebound_entries(d), d['sma50'].values)
    tB = replay(d, breakout_entries(d), d['ema21'].values)
    c = d['close'].values; n = len(c); last = d.iloc[-1]

    def ticket(trades, ma_name):
        t = trades[-1] if trades else None
        if t is None or t['j'] is not None:
            if t is None:
                return dict(open=False, last_exit=None)
            return dict(open=False, last_exit=d.index[t['j']], last_entry=d.index[t['i']],
                        last_ret=float(t['exit'] / t['entry'] - 1), last_why='stop' if t['exit'] <= t['stop'] + 1e-9 else 'MA exit')
        ma_now = float(d[ma_name].iloc[-1])
        return dict(open=True, entry_date=d.index[t['i']], entry=float(t['entry']), stop=float(t['stop']),
                    stop_dist=float(c[-1] / t['stop'] - 1), armed=bool(t['armed']), run=int(t['run']), ma=ma_now,
                    pnl=float(c[-1] / t['entry'] - 1), days=n - 1 - t['i'])

    A, B = ticket(tA, 'sma50'), ticket(tB, 'ema21')
    # sell-off status for the rebound trigger: current episode, its start, and whether its entry was taken
    E = d['E'].values; inep = False; ep_start = None
    for i, e in enumerate(E):
        if not np.isfinite(e):
            continue
        if not inep and e <= -1:
            inep, ep_start = True, i
        elif inep and e >= 0:
            inep, ep_start = False, None
    ep_taken = bool(inep and any(t['i'] == ep_start for t in tA))
    used = inep
    rb_level = float(last['sma50'] - last['atr14'])            # close at or below this (approx: SMA and ATR move tomorrow)
    bo_level = float(d['high'].iloc[-21:].max())                # close above this triggers tomorrow
    r = daily_returns(d, tA, tB)
    live = r[r.index > GO_LIVE]; bh = d['close'].pct_change()[d.index > GO_LIVE]
    return dict(ticker=ticker, date=d.index[-1], close=float(c[-1]), E=float(last['E']), sma50=float(last['sma50']),
                ema21=float(last['ema21']), atr14=float(last['atr14']), A=A, B=B, position=A['open'] or B['open'],
                in_selloff=inep, rb_used=used, rb_level=rb_level, bo_level=bo_level,
                ep_start=(d.index[ep_start] if ep_start is not None else None), ep_taken=ep_taken,
                live_rule=float((1 + live).prod() - 1) if len(live) else np.nan,
                live_bh=float((1 + bh).prod() - 1) if len(bh) else np.nan, live_days=len(live),
                hist=d.iloc[-130:][['close', 'sma50', 'ema21']], expo_series=_exposure(d, tA, tB).iloc[-130:],
                n_trades_A=len(tA), n_trades_B=len(tB))


def _exposure(d, tA, tB):
    n = len(d); x = np.zeros(n)
    for tr in (tA, tB):
        for t in tr:
            j = t['j'] if t['j'] is not None else n - 1
            x[t['i'] + 1:j + 1] = 1
    return pd.Series(x, index=d.index)


def log_row(states: dict) -> dict:
    """Forward-log columns for mrs_history.csv (one set per ticker)."""
    row = {}
    for tk, s in states.items():
        k = tk.lower()
        tickets = '+'.join(n for n, t in (('A', s['A']), ('B', s['B'])) if t['open']) or '-'
        row[f'alloc_{k}_pos'] = 'IN' if s['position'] else 'OUT'
        row[f'alloc_{k}_tickets'] = tickets
        row[f'alloc_{k}_stopA'] = round(s['A']['stop'], 2) if s['A']['open'] else np.nan
        row[f'alloc_{k}_stopB'] = round(s['B']['stop'], 2) if s['B']['open'] else np.nan
    return row


# ── entry guide for stock setups (exploratory, 10 Nov 2026) ─────────────────────
GUIDE = {
    True: ('IN', TEAL, 'Continuation setups (bases, pivots, breakouts): normal size. Deep dip: normal size.'),
    False: ('OUT', ORANGE, 'Continuation setups (bases, pivots, breakouts): no new entries, or quarter size. '
                           'Deep dip: allowed. Open risk: at most half the ladder cap.'),
}
GUIDE_EVIDENCE = ('Stock trades 2008-2026 by SPY rule state on the entry day, mean R per trade: base breakout +0.06 IN / '
                  '-0.14 OUT; mid-base pivot +0.10 / -0.06; Deep dip +0.15 / +0.10. Exploratory, in-sample; logged forward.')


def entry_guide(states: dict):
    """(spy_in, label, colour, text) or None. Uses the SPY state only."""
    s = states.get('SPY')
    if not s:
        return None
    lab, col, txt = GUIDE[bool(s['position'])]
    return bool(s['position']), lab, col, txt


def guide_html(states: dict, compact: bool = False) -> str:
    g = entry_guide(states)
    if not g:
        return ''
    _, lab, col, txt = g
    return (f'<div title="{_e(GUIDE_EVIDENCE)}" style="font-size:0.78rem;color:#f3ece2;line-height:1.35;'
            f'border-left:3px solid {col};padding:3px 8px;margin:{"0 0 6px 0" if compact else "0 0 8px 0"};background:#0d0d0d;">'
            f'<b style="color:{col};">Stock entries · SPY rule {lab}</b> — {txt} '
            f'<span style="color:#6e675f;">exploratory</span></div>')


# ── card (compact: one line per ticker) ───────────────────────────────────────────
CSS = ('<style>.al{background:#121212;border:1px solid #2a2724;border-left:3px solid #85cac5;border-radius:6px;'
       'padding:8px 12px 6px 12px;margin:2px 0 12px 0;}'
       '.al .k{font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;color:#85cac5;font-weight:700;margin-bottom:6px;}'
       '.al .row{display:grid;grid-template-columns:96px 1fr 1fr 230px;gap:10px;align-items:center;'
       'border-top:1px solid #24211e;padding:5px 0;}'
       '.al .pill{font-size:1.05rem;font-weight:800;}'
       '.al .t{font-size:0.66rem;letter-spacing:0.10em;text-transform:uppercase;color:#a89f95;}'
       '.al .tx{font-size:0.78rem;color:#f3ece2;line-height:1.3;}'
       '.al .sub{font-size:0.72rem;color:#a89f95;line-height:1.3;}'
       '.al .num{font-variant-numeric:tabular-nums;}'
       '@media (max-width:900px){.al .row{grid-template-columns:1fr;}}</style>')


def _spark(s: dict, w=230, h=34) -> str:
    hst = s['hist'].dropna(); ex = s['expo_series'].reindex(hst.index).fillna(0).values
    if len(hst) < 5:
        return ''
    lo = float(hst.min().min()); hi = float(hst.max().max()); span = (hi - lo) or 1
    X = lambda i: 1 + i * (w - 2) / (len(hst) - 1)
    Y = lambda v: 2 + (hi - v) * (h - 4) / span
    bw = (w - 2) / len(hst)
    bars = ''.join(f'<rect x="{X(i) - bw / 2:.1f}" y="0" width="{bw + 0.3:.1f}" height="{h}" fill="#132322"/>'
                   for i in range(len(hst)) if ex[i] > 0)

    def line(col, color, sw, dash=''):
        pts = ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(hst[col].values))
        da = f' stroke-dasharray="{dash}"' if dash else ''
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"{da}/>'
    marks = ''.join(f'<line x1="1" x2="{w - 1}" y1="{Y(t["stop"]):.1f}" y2="{Y(t["stop"]):.1f}" stroke="{col}" '
                    f'stroke-dasharray="3,3" stroke-width="1"/>'
                    for t, col in ((s['A'], ORANGE), (s['B'], '#7fa7d9')) if t['open'] and lo <= t['stop'] <= hi)
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="6 months">'
            f'<title>6 months: close (white), 50-day (copper), 21 EMA (dotted); shaded = in the rule; dashed = stops</title>'
            f'{bars}{line("sma50", COPPER, 1.1)}{line("ema21", MUTED, 0.9, "2,2")}{line("close", INK, 1.3)}{marks}</svg>')


def _open_txt(t: dict, ma_lbl: str) -> str:
    pc = TEAL if t['pnl'] >= 0 else ORANGE
    arm = (f'exit {t["run"]}/2 closes below the {ma_lbl} ({t["ma"]:.2f})' if t['armed'] else
           f'exit not armed until a close above the {ma_lbl} ({t["ma"]:.2f})')
    return (f'since {pd.Timestamp(t["entry_date"]):%d %b} at <b class="num">{t["entry"]:.2f}</b> '
            f'<b class="num" style="color:{pc};">{t["pnl"] * 100:+.1f}%</b> · stop <b class="num">{t["stop"]:.2f}</b> '
            f'({t["stop_dist"] * 100:.1f}% away) · {arm}')


def _last_txt(t: dict) -> str:
    if t.get('last_exit') is None:
        return ''
    rc = TEAL if t['last_ret'] >= 0 else ORANGE
    return (f' <span class="sub">Last: {pd.Timestamp(t["last_entry"]):%d %b} → {pd.Timestamp(t["last_exit"]):%d %b} '
            f'<b class="num" style="color:{rc};">{t["last_ret"] * 100:+.1f}%</b> ({t["last_why"]}).</span>')


def _rebound_txt(s: dict) -> str:
    t = s['A']
    if t['open']:
        return _open_txt(t, '50-day')
    if s['in_selloff'] and s['ep_start'] is not None:
        st = pd.Timestamp(s['ep_start'])
        if s['ep_taken']:
            why = f'this sell-off (from {st:%d %b}) already gave its entry'
        else:
            why = (f'this sell-off started {st:%d %b} while the previous rebound trade was still open, so it gives no entry '
                   f'(one per sell-off)')
        return (f'Waiting: {why}. Resets on a close above the 50-day (<b class="num">{s["sma50"]:.2f}</b>), '
                f'then a new close 1 ATR below it.' + _last_txt(t))
    return (f'Buys on a close at or below about <b class="num">{s["rb_level"]:.2f}</b> (50-day − 1 ATR; now '
            f'{s["E"]:+.2f} ATR).' + _last_txt(t))


def _breakout_txt(s: dict) -> str:
    t = s['B']
    if t['open']:
        return _open_txt(t, '21 EMA')
    return f'Buys on a close above <b class="num">{s["bo_level"]:.2f}</b> (21-session high).' + _last_txt(t)


def card_html(states: dict) -> str:
    h = (CSS + '<div class="al"><div class="k">Allocation rule · SPY / QQQ · in-sample, forward tracking from 10 Nov 2026</div>'
         + guide_html(states, compact=True))
    for tk, s in states.items():
        pos_c = TEAL if s['position'] else DIM
        ref = REF.get(tk, {})
        tip = (f'Backtest {ref["period"]} (in-sample): {ref["cagr"]:.1f}%/yr, max DD {ref["dd"]}%, Sharpe {ref["sh"]:.2f}, '
               f'{ref["expo"]}% invested; buy-and-hold {ref["bh_cagr"]:.1f}%, {ref["bh_dd"]}%, {ref["bh_sh"]:.2f}.' if ref else '')
        if tk in CAUTION:
            tip += ' ' + CAUTION[tk]
        live = (f'since 10 Nov {s["live_rule"] * 100:+.1f}% vs {s["live_bh"] * 100:+.1f}%' if s['live_days'] else 'tracking from 10 Nov')
        warn = ' <span style="color:#f8d6b4;" title="' + _e(CAUTION[tk]) + '">⚠</span>' if tk in CAUTION else ''
        h += (f'<div class="row"><div title="{_e(tip)}"><span class="t">{tk}</span>{warn}<br>'
              f'<span class="pill" style="color:{pos_c};">{"IN" if s["position"] else "OUT"}</span>'
              f'<div class="sub">{live}</div></div>'
              f'<div class="tx"><span class="t">A · Rebound</span> {_rebound_txt(s)}</div>'
              f'<div class="tx"><span class="t">B · Breakout</span> {_breakout_txt(s)}</div>'
              f'<div>{_spark(s)}</div></div>')
    h += ('<div class="sub" style="border-top:1px solid #24211e;padding-top:4px;">After the close: entries and MA exits at the close, '
          'stops as resting orders. 100% of the allocated capital while a ticket is open; a stop-out costs about 2 ATR(10) '
          '(3–5% on SPY). Hover a ticker for the backtest.</div></div>')
    return h.replace('\n', ' ')


# ── data ──────────────────────────────────────────────────────────────────────────
def fetch(tickers=TICKERS) -> dict:
    """Daily OHLC, split-adjusted, not dividend-adjusted (auto_adjust=False), full history."""
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period='max', auto_adjust=False)
            if h is None or not len(h):
                continue
            idx = pd.to_datetime(h.index)
            h.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
            out[t] = h[['Open', 'High', 'Low', 'Close']].rename(columns=str.lower)
        except Exception as e:
            print(f'[allocation] {t}: {e}')
    return out


def build(data: dict) -> dict:
    return {t: state(df, t) for t, df in data.items() if df is not None and len(df) > 120}
