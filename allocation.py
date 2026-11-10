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
    # sell-off status for the rebound trigger
    E = d['E'].values; inep = False; used = False
    for i, e in enumerate(E):
        if not np.isfinite(e):
            continue
        if not inep and e <= -1:
            inep, used = True, True
        elif inep and e >= 0:
            inep, used = False, False
    rb_level = float(last['sma50'] - last['atr14'])            # close at or below this (approx: SMA and ATR move tomorrow)
    bo_level = float(d['high'].iloc[-21:].max())                # close above this triggers tomorrow
    r = daily_returns(d, tA, tB)
    live = r[r.index > GO_LIVE]; bh = d['close'].pct_change()[d.index > GO_LIVE]
    return dict(ticker=ticker, date=d.index[-1], close=float(c[-1]), E=float(last['E']), sma50=float(last['sma50']),
                ema21=float(last['ema21']), atr14=float(last['atr14']), A=A, B=B, position=A['open'] or B['open'],
                in_selloff=inep, rb_used=used, rb_level=rb_level, bo_level=bo_level,
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


# ── card (same look as the Risk today panel) ───────────────────────────────────────
CSS = ('<style>.al{background:#121212;border:1px solid #2a2724;border-left:3px solid #85cac5;border-radius:6px;'
       'padding:12px 16px 10px 16px;margin:2px 0 14px 0;}'
       '.al .k{font-size:0.70rem;letter-spacing:0.12em;text-transform:uppercase;color:#85cac5;font-weight:700;margin-bottom:10px;}'
       '.al .row{display:grid;grid-template-columns:230px 1fr 1fr 340px;gap:12px;margin-bottom:10px;}'
       '.al .tile{background:#0d0d0d;border:1px solid #24211e;border-radius:6px;padding:10px 12px;display:flex;flex-direction:column;gap:6px;}'
       '.al .lab{font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;color:#a89f95;}'
       '.al .big{font-size:1.6rem;font-weight:800;line-height:1.0;}'
       '.al .sub{font-size:0.78rem;color:#a89f95;line-height:1.35;}'
       '.al .act{font-size:0.82rem;color:#f3ece2;line-height:1.35;}'
       '.al .num{font-variant-numeric:tabular-nums;}'
       '.al .chip{display:inline-block;font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;padding:1px 6px;border-radius:3px;'
       'border:1px solid #6e675f;color:#a89f95;margin-left:6px;}'
       '@media (max-width:900px){.al .row{grid-template-columns:1fr;}}</style>')


def _spark(s: dict, w=320, h=86) -> str:
    hst = s['hist'].dropna(); ex = s['expo_series'].reindex(hst.index).fillna(0).values
    if len(hst) < 5:
        return ''
    lo = float(hst.min().min()); hi = float(hst.max().max()); span = (hi - lo) or 1
    X = lambda i: 2 + i * (w - 4) / (len(hst) - 1)
    Y = lambda v: 4 + (hi - v) * (h - 8) / span
    bars = ''.join(f'<rect x="{X(i) - 0.5 * (w - 4) / len(hst):.1f}" y="0" width="{(w - 4) / len(hst) + 0.3:.1f}" height="{h}" fill="#132322"/>'
                   for i in range(len(hst)) if ex[i] > 0)
    def line(col, color, sw, dash=''):
        pts = ' '.join(f'{X(i):.1f},{Y(v):.1f}' for i, v in enumerate(hst[col].values))
        da = f' stroke-dasharray="{dash}"' if dash else ''
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"{da}/>'
    marks = ''
    for t, col in ((s['A'], ORANGE), (s['B'], '#7fa7d9')):
        if t['open'] and lo <= t['stop'] <= hi:
            marks += f'<line x1="2" x2="{w - 2}" y1="{Y(t["stop"]):.1f}" y2="{Y(t["stop"]):.1f}" stroke="{col}" stroke-dasharray="3,3" stroke-width="1"/>'
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="last 6 months">{bars}'
            f'{line("sma50", COPPER, 1.2)}{line("ema21", MUTED, 1, "2,2")}{line("close", INK, 1.5)}{marks}</svg>'
            f'<div class="sub">6 months · close (white), 50-day (copper), 21 EMA (dotted), shaded = in the rule · dashed = stops</div>')


def _ticket_html(name: str, t: dict, ma_lbl: str, trig: str) -> str:
    if t['open']:
        arm = (f'Exit armed: <b class="num">{t["run"]}/2</b> closes below the {ma_lbl} ({t["ma"]:.2f})' if t['armed'] else
               f'Exit not armed until a close above the {ma_lbl} ({t["ma"]:.2f}); only the stop protects')
        pc = TEAL if t['pnl'] >= 0 else ORANGE
        return (f'<div class="tile"><div class="lab">{name} · <span style="color:{TEAL};">open</span></div>'
                f'<div class="act">Since {pd.Timestamp(t["entry_date"]):%d %b %Y} at <b class="num">{t["entry"]:.2f}</b> · '
                f'<b class="num" style="color:{pc};">{t["pnl"] * 100:+.1f}%</b> · {t["days"]} sessions</div>'
                f'<div class="act">Stop <b class="num">{t["stop"]:.2f}</b> <span class="sub">({t["stop_dist"] * 100:.1f}% away)</span></div>'
                f'<div class="sub">{arm}.</div></div>')
    le = t.get('last_exit')
    last = ''
    if le is not None:
        rc = TEAL if t['last_ret'] >= 0 else ORANGE
        last = (f'<div class="sub">Last trade {pd.Timestamp(t["last_entry"]):%d %b} → {pd.Timestamp(le):%d %b %Y}: '
                f'<b class="num" style="color:{rc};">{t["last_ret"] * 100:+.1f}%</b> ({t["last_why"]}).</div>')
    return (f'<div class="tile"><div class="lab">{name} · <span style="color:{DIM};">closed</span></div>'
            f'<div class="act">{trig}</div>{last}</div>')


def card_html(states: dict) -> str:
    h = CSS + '<div class="al"><div class="k">Allocation rule · rebound + breakout tickets (in-sample, forward tracking from 10 Nov 2026)</div>'
    for tk, s in states.items():
        pos_c = TEAL if s['position'] else DIM
        n_open = int(s['A']['open']) + int(s['B']['open'])
        ref = REF.get(tk, {})
        live = (f'Since 10 Nov: rule <b class="num">{s["live_rule"] * 100:+.1f}%</b> vs {tk} <b class="num">{s["live_bh"] * 100:+.1f}%</b>'
                if s['live_days'] else 'Forward tracking starts after 10 Nov 2026.')
        caution = (f'<div class="sub" style="color:#f8d6b4;">{_e(CAUTION[tk])}</div>' if tk in CAUTION else '')
        if s['rb_used'] and s['in_selloff']:
            rb_trig = (f'This sell-off already gave its rebound entry. Resets on a close above the 50-day '
                       f'(<b class="num">{s["sma50"]:.2f}</b>).')
        else:
            rb_trig = (f'Buys on a close at or below about <b class="num">{s["rb_level"]:.2f}</b> '
                       f'(50-day − 1 ATR; now {s["E"]:+.2f} ATR).')
        bo_trig = f'Buys on a close above <b class="num">{s["bo_level"]:.2f}</b> (21-session high).'
        h += (f'<div class="row"><div class="tile"><div class="lab">{tk}</div>'
              f'<div class="big" style="color:{pos_c};">{"IN" if s["position"] else "OUT"}</div>'
              f'<div class="sub">{"100% invested" if s["position"] else "in cash"} · {n_open} ticket{"s" if n_open != 1 else ""} open</div>'
              f'<div class="sub">{live}</div>'
              + (f'<div class="sub">Backtest {ref["period"]}: {ref["cagr"]:.1f}%/yr, max DD {ref["dd"]}%, Sharpe {ref["sh"]:.2f} '
                 f'vs buy-and-hold {ref["bh_cagr"]:.1f}%, {ref["bh_dd"]}%, {ref["bh_sh"]:.2f} · {ref["expo"]}% invested '
                 f'<span class="chip">in-sample</span></div>' if ref else '')
              + f'{caution}</div>'
              f'{_ticket_html("A · Rebound", s["A"], "50-day", rb_trig)}'
              f'{_ticket_html("B · Breakout", s["B"], "21 EMA", bo_trig)}'
              f'<div class="tile">{_spark(s)}</div></div>')
    h += ('<div class="sub">Check after the close. Entries and MA exits are at the close; stops are resting orders. '
          'A stop-out costs about 2 ATR(10) of the allocated capital (3–5% on SPY): size it as a core allocation, not as a swing trade.</div></div>')
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
