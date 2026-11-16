"""
exposure_guide.py — MRS v8 card (8.0.0, 15 Nov 2026)
====================================================
One visual card on top of the page; the words are in a fold under it (explainer_md(), closed by default).
Framework: Studies/Index_Edge_Program_2026-11/MRS_v7 (C1 clock, v8 sizing, L3 chain check), anchor 00_GOAL_anchor.md.

States (all ex ante, at the close)
  Clock       sessions since SPY's last DIP ZONE day (E = (close - SMA50) / ATR14 <= -1):
              IN DIP / EARLY (1-20) / MID (21-60) / LATE (> 60). C1: passed on stocks by episode (in sample).
  SPY phase   allocation.py replay: DIP ZONE / A (rebound) open / B (breakout) open / NONE.
  VIX width   VIX against its expanding median since 1990 (17.56 on 13 Nov 2026): CALM < m <= WIDE < 1.5 m <= STRESSED.
              Width, not direction (V1, V4).
200-day       8.5.0 (16 Nov 2026, DF1): F-UP = SPY close > SMA200 and SMA200 above its value 20 sessions ago.
              SPX 1963-2026: dips below a falling 200-day fell another 10% within 60 sessions in 45% of episodes,
              vs 12% above a rising one (both eras, same vol mix).
Stance        set by the SPY phase and, in the DIP ZONE, the 200-day filter (v8 sizing 15 Nov; OOS 2001-08 + DF1 16 Nov):
              PRESS = ticket A or B open: 0.75% a trade, swing open risk <= 4%;
              NORMAL = DIP ZONE with SPY above a rising 200-day: 0.75% a trade, <= 3%;
              EASE = no ticket, or DIP ZONE without F-UP: 0.375% a trade, <= 1.5%, swing gross <= 60%.
              (V71F: beat the v8 budget in 2001-08 and 2009-26.) Pressing in the dip raised the drawdown in both eras.
              The clock stays on the card as per-trade context; it held only above a rising 200-day (2001-08 OOS).
Where to look In the DIP ZONE, beaten-down names (bottom 40% by 6-month return) beat prior leaders (RS1-b, tested
              2001-08, same direction 2009-26). After the zone, no RS edge. No chart pattern beat a random entry (S2).
Budget        SPY tickets one unit each at 1.5% NLV risk (Fabio, 15 Nov 2026); account gross <= 100%, core first; no margin.
Exit          +2R -> break-even, then trail the 21 EMA or structure; no profit cap.
Hedge         only the ~60-session ~5% OTM SPY put, bought in the 5 sessions after a trend start (ticket B opens) or a
              trend end (B -> no ticket); never roll short-dated puts. Price: the daily put log (put_log.py).
Not on the card (decided 15 Nov 2026): composite MRS score, short book, FTD as a rule, RS tilt, margin, rolling puts,
deep dips as a place to look (16 Nov: no better than a random entry, 2001-26).
"""
import html as _html
import numpy as np
import pandas as pd

TEAL, TEAL_D, ORANGE, COPPER, RED = '#85cac5', '#649995', '#f4a062', '#e3b48e', '#f4a062'   # 8.4.0: ATN tokens only
INK, MUTED, DIM, LINE, BG = '#f3ece2', '#a89f95', '#6e675f', '#2a2724', '#121212'
VIX_MEDIAN = 17.56            # expanding median of the VIX since 1990-01-02, as of 2026-11-13
CORE_RISK = 0.015             # SPY ticket unit risk (Fabio, 15 Nov 2026)
BUDGET = {'PRESS': (0.0075, 0.04, 1.00), 'NORMAL': (0.0075, 0.03, 1.00), 'EASE': (0.00375, 0.015, 0.60)}  # risk a trade,
#          swing open-risk cap, swing gross cap (fractions of NLV). 8.5.0 rule V71F (NORMAL only in the DIP ZONE with
#          SPY above a rising 200-day), 500 paths: 2009-26 CAGR 9.6%, max DD -22.7%; 2001-08 CAGR -1.1%, max DD -31.9%
#          (v8 before: 9.1% / -25.9% and -2.9% / -39.3%).
F200_SLOPE = 20               # sessions over which the 200-day's slope is read (DF1)
SWING_CAP = {k: v[1] for k, v in BUDGET.items()}
SWING_RISK = BUDGET['NORMAL'][0]
STANCE_COL = {'PRESS': TEAL, 'NORMAL': INK, 'EASE': COPPER}
PH_NAME = {0: 'DIP', 1: 'A', 2: 'B', 3: 'NONE'}
HEDGE_WINDOW = 5              # sessions after a trend start / end in which the 60-session put is the rule


def _fin(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _e(s):
    return _html.escape(str(s), quote=True)


# ── states ────────────────────────────────────────────────────────────────────────────────────────────────────
def vol_band(vix):
    if vix is None or vix != vix:
        return None
    if vix < VIX_MEDIAN:
        return 'CALM'
    return 'WIDE' if vix < 1.5 * VIX_MEDIAN else 'STRESSED'


def clock(ph: pd.Series):
    """(bucket, sessions since the last DIP ZONE day) from the per-session phase series (0 = DIP ZONE)."""
    if ph is None or not len(ph):
        return None, None
    p = np.asarray(ph.values, int)
    dips = np.flatnonzero(p == 0)
    if not len(dips):
        return 'LATE', None
    n = len(p) - 1 - dips[-1]
    if n == 0:
        return 'IN DIP', 0
    return ('EARLY' if n <= 20 else ('MID' if n <= 60 else 'LATE')), int(n)


def filter200(ohlc: pd.DataFrame, upto: int = None):
    """8.5.0 (DF1): SPY against its 200-day at the last close (or at row `upto`, exclusive end).
    Returns dict(state='F-UP'|'F-MIX'|'F-DN', close, sma, slope) or None when there are fewer than 220 closes."""
    try:
        s = ohlc.rename(columns=str.lower)['close'].astype(float).dropna()
    except Exception:
        return None
    if upto is not None:
        s = s.iloc[:upto]
    if len(s) < 200 + F200_SLOPE:
        return None
    sma = s.rolling(200).mean()
    c, m, slope = float(s.iloc[-1]), float(sma.iloc[-1]), float(sma.iloc[-1] - sma.iloc[-1 - F200_SLOPE])
    st_ = 'F-UP' if (c > m and slope > 0) else ('F-DN' if (c < m and slope < 0) else 'F-MIX')
    return dict(state=st_, close=c, sma=m, slope=slope)


def _fs(f200):
    """The filter state string from a filter200() dict or a plain state string."""
    return f200.get('state') if isinstance(f200, dict) else f200


def stance(code, f200=None):
    """Budget stance from the SPY phase code (0 DIP ZONE, 1 A open, 2 B open, 3 no ticket) and, in the DIP ZONE, the
    200-day filter (8.5.0): NORMAL only when SPY is above a rising 200-day; otherwise (or unknown) EASE."""
    if code is None:
        return None
    if code in (1, 2):
        return 'PRESS'
    if code == 0:
        return 'NORMAL' if _fs(f200) == 'F-UP' else 'EASE'
    return 'EASE'


def hedge_window(ph: pd.Series):
    """('trend start' | 'trend end', sessions ago) if a trend started / ended in the last HEDGE_WINDOW sessions."""
    if ph is None or len(ph) < 2:
        return None
    p = np.asarray(ph.values, int)
    for k in range(0, HEDGE_WINDOW):
        i = len(p) - 1 - k
        if i < 1:
            break
        if p[i] == 2 and p[i - 1] != 2:
            return 'trend start', k
        if p[i] == 3 and p[i - 1] == 2:
            return 'trend end', k
    return None


def last_ftd(spy: pd.DataFrame, dd=0.05):
    """Last follow-through day (C2 definition: close >= 5% below the prior 252-session high; rally attempt day 4+;
    close >= +1.25% on higher volume). Commentary only (C2: waiting for it did not help). Returns a Timestamp or None."""
    try:
        s = spy.rename(columns=str.lower)
        c, v = s['close'].to_numpy(float), s['volume'].to_numpy(float)
    except Exception:
        return None
    hi = pd.Series(c).rolling(252, min_periods=60).max().shift(1).to_numpy()
    last, in_pb, low, day, done = None, False, np.inf, 0, False
    for i in range(1, len(c)):
        if not in_pb:
            if np.isfinite(hi[i]) and c[i] <= (1 - dd) * hi[i]:
                in_pb, low, day, done = True, c[i], 0, False
            continue
        if c[i] < low:
            low, day, done = c[i], 0, False
            continue
        if day == 0 and c[i] > c[i - 1]:
            day = 1
        elif day > 0:
            day += 1
        if day >= 4 and not done and c[i] / c[i - 1] - 1 >= 0.0125 and v[i] > v[i - 1]:
            last, done = s.index[i], True
        if np.isfinite(hi[i]) and c[i] >= hi[i]:
            in_pb = False
    return last


def unit_size(entry, stop, risk=CORE_RISK):
    if not entry or not stop or entry <= stop:
        return None
    return min(1.0, risk / ((entry - stop) / entry))


# ── drawing helpers (plain HTML / CSS: renders in st.markdown) ─────────────────────────────────────────────────
CSS = ('<style>'
       '.v8{background:#121212;border:1px solid #2a2724;border-radius:8px;padding:12px 14px 10px 14px;margin:2px 0 6px 0;}'
       '.v8 .top{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:10px;}'
       '.v8 .badge{font-size:1.9rem;font-weight:900;letter-spacing:0.08em;padding:2px 16px;border-radius:6px;color:#0a0a0a;line-height:1.25;}'
       '.v8 .k{font-size:0.62rem;letter-spacing:0.13em;text-transform:uppercase;color:#a89f95;font-weight:700;}'
       '.v8 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px;}'
       '.v8 .g2{grid-template-columns:1.25fr 1fr;}'
       '.v8 .tile{background:#171615;border:1px solid #24211e;border-radius:6px;padding:8px 10px;min-width:0;}'
       '.v8 .val{font-size:1.15rem;font-weight:800;color:#f3ece2;line-height:1.2;}'
       '.v8 .sm{font-size:0.72rem;color:#a89f95;line-height:1.3;}'
       '.v8 .tag{float:right;font-size:0.55rem;font-weight:700;letter-spacing:0.05em;padding:0 4px;border-radius:3px;border:1px solid #3a3632;color:#6e675f;}'
       '.v8 .bar{position:relative;height:12px;border-radius:3px;overflow:hidden;display:flex;margin:6px 0 3px 0;}'
       '.v8 .seg{height:100%;}'
       '.v8 .pin{position:absolute;top:-3px;width:3px;height:18px;background:#f3ece2;border-radius:1px;box-shadow:0 0 0 1px #121212;}'
       '.v8 .lbls{display:flex;font-size:0.58rem;color:#6e675f;letter-spacing:0.04em;}'
       '.v8 .pill{display:inline-block;min-width:34px;text-align:center;font-size:0.72rem;font-weight:800;padding:3px 6px;margin:6px 4px 3px 0;border-radius:4px;border:1px solid #3a3632;color:#6e675f;}'
       '.v8 .dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:-1px;}'
       '.v8 .lad{display:flex;align-items:center;gap:4px;margin:6px 0 2px 0;flex-wrap:wrap;}'
       '.v8 .st{font-size:0.7rem;font-weight:800;padding:3px 7px;border-radius:4px;background:#24211e;color:#f3ece2;white-space:nowrap;}'
       '.v8 .ar{color:#6e675f;font-size:0.8rem;}'
       '.v8 .ft{font-size:0.66rem;color:#6e675f;margin-top:2px;}'
       '@media (max-width:900px){.v8 .grid,.v8 .g2{grid-template-columns:1fr;}}'
       '</style>')


def _tag(t):
    import ui                       # 8.4.0: one evidence tag across the MRS
    return ui.tag(t)


def _bar(segs, pin_frac, labels, align='left'):
    """segs: [(width %, colour)], pin at pin_frac (0..1), labels: [(width %, text)]."""
    h = '<div class="bar">' + ''.join(f'<div class="seg" style="width:{w}%;background:{c};"></div>' for w, c in segs)
    if pin_frac is not None:
        h += f'<div class="pin" style="left:calc({min(max(pin_frac, 0), 1) * 100:.1f}% - 1px);"></div>'
    h += '</div><div class="lbls">' + ''.join(f'<span style="width:{w}%;text-align:{align};">{_e(t)}</span>' for w, t in labels) + '</div>'
    return h


def _clock_tile(bucket, n):
    # IN DIP 12% | EARLY 1-20 28% | MID 21-60 35% | LATE >60 25% (LATE drawn up to 120 sessions)
    if bucket is None:
        return f'<div class="tile"><div class="k">Clock</div><div class="sm">not available</div></div>'
    if bucket == 'IN DIP':
        f = 0.06
    elif bucket == 'EARLY':
        f = 0.12 + 0.28 * n / 20
    elif bucket == 'MID':
        f = 0.40 + 0.35 * (n - 20) / 40
    else:
        f = 0.75 + 0.25 * min((n or 120) - 60, 60) / 60
    col = TEAL if bucket in ('IN DIP', 'EARLY') else (INK if bucket == 'MID' else COPPER)
    segs = [(12, TEAL), (28, TEAL_D), (35, '#3a3632'), (25, '#5a4636')]
    lab = [(12, 'DIP'), (28, '1–20'), (35, '21–60'), (25, '60+')]
    nn = '' if n in (None, 0) else f' · {n}'
    return (f'<div class="tile">{_tag("Per trade · above a rising 200-day")}<div class="k">Clock · sessions since the dip</div>'
            f'<div class="val" style="color:{col};">{_e(bucket)}{nn}</div>{_bar(segs, f, lab)}</div>')


def _phase_tile(code):
    pills = ''
    for p in (0, 1, 2, 3):
        on = code == p
        bg = {0: TEAL, 1: TEAL_D, 2: INK, 3: '#8a847c'}[p]
        sty = f'background:{bg};color:#0a0a0a;border-color:{bg};' if on else ''
        pills += f'<span class="pill" style="{sty}">{PH_NAME[p]}</span>'
    return (f'<div class="tile">{_tag("Tested")}<div class="k">SPY phase · ticket</div>{pills}'
            f'<div class="lbls"><span>dip zone · A rebound · B breakout · none</span></div></div>')


def _vix_tile(vix, vb):
    if vb is None:
        return '<div class="tile"><div class="k">VIX width</div><div class="sm">not available</div></div>'
    lo, hi = 10.0, 40.0
    m, m15 = VIX_MEDIAN, 1.5 * VIX_MEDIAN
    w1, w2 = (m - lo) / (hi - lo) * 100, (m15 - m) / (hi - lo) * 100
    segs = [(w1, TEAL_D), (w2, '#b9805a'), (100 - w1 - w2, '#f4a062')]
    lab = [(w1, f'calm < {m:.1f}'), (w2, 'wide'), (100 - w1 - w2, f'stressed > {m15:.1f}')]
    col = TEAL if vb == 'CALM' else (ORANGE if vb == 'WIDE' else RED)
    return (f'<div class="tile">{_tag("Tested")}<div class="k">VIX · width of moves</div>'
            f'<div class="val" style="color:{col};">{vix:.1f} · {_e(vb)}</div>{_bar(segs, (vix - lo) / (hi - lo), lab)}</div>')


def _budget_tile(st_):
    cap = SWING_CAP[st_]
    top = max(SWING_CAP.values())
    segs = []
    order = ['EASE', 'NORMAL', 'PRESS']
    prev = 0.0
    for s_ in order:
        w = (SWING_CAP[s_] - prev) / top * 100
        on = SWING_CAP[s_] <= cap + 1e-12
        segs.append((w, STANCE_COL[s_] if on else '#2a2724'))
        prev = SWING_CAP[s_]
    lab = [((SWING_CAP['EASE']) / top * 100, f'{SWING_CAP["EASE"] * 100:.1f}%'),
           ((SWING_CAP['NORMAL'] - SWING_CAP['EASE']) / top * 100, f'{SWING_CAP["NORMAL"] * 100:.1f}%'),
           ((SWING_CAP['PRESS'] - SWING_CAP['NORMAL']) / top * 100, f'{SWING_CAP["PRESS"] * 100:.1f}%')]
    rt, _, g = BUDGET[st_]
    n = int(round(cap / rt))
    return (f'<div class="tile">{_tag("Tested · v8 sizing")}<div class="k">Swing open risk</div>'
            f'<div class="val">{cap * 100:.1f}% <span class="sm">· {rt * 100:.3g}% a trade · ≈ {n} at full risk'
            + (f' · swing gross ≤ {g * 100:.0f}%' if g < 1 else '') + '</span></div>'
            f'{_bar(segs, None, lab, "right")}</div>')


def _ticket_rows(s):
    """(rows html, footer html) for the two SPY tickets."""
    s = s or {}
    rows = ''
    tot = 0.0
    for nm, lbl in (('A', 'A rebound'), ('B', 'B breakout')):
        t = s.get(nm, {}) or {}
        if t.get('open'):
            u = unit_size(t.get('entry'), t.get('stop'))
            tot += u or 0
            rows += (f'<div class="sm"><span class="dot" style="background:{TEAL};"></span><b style="color:{INK};">{lbl}</b>'
                     f' · stop {t.get("stop", float("nan")):.2f}' + (f' · unit ≈ {u * 100:.0f}% NLV' if u else '') + '</div>')
        else:
            lvl = s.get('rb_level') if nm == 'A' else s.get('bo_level')
            lock = nm == 'A' and s.get('in_selloff') and s.get('rb_used')
            hint = ' · locked until a close above the 50-day' if lock else (
                (f' · buys {"≤" if nm == "A" else ">"} {lvl:.2f}') if lvl else '')
            rows += (f'<div class="sm"><span class="dot" style="border:1px solid {DIM};"></span>{lbl}{_e(hint)}</div>')
    ft = (f'<div class="ft">{CORE_RISK * 100:.1f}% risk each · core ≈ {min(tot, 1) * 100:.0f}% NLV · gross ≤ 100%, core first</div>' if tot else
          f'<div class="ft">{CORE_RISK * 100:.1f}% risk each · gross ≤ 100%, core first · no margin</div>')
    return rows, ft


def _tickets_tile(s):
    rows, ft = _ticket_rows(s)
    return (f'<div class="tile">{_tag("Tested · drawdown control")}<div class="k">SPY tickets · {CORE_RISK * 100:.1f}% risk each</div>'
            f'<div style="margin-top:5px;">{rows}</div>{ft}</div>')





def _exit_tile():
    lad = (f'<span class="st">stop</span><span class="ar">→</span><span class="st">+2R: break-even</span>'
           f'<span class="ar">→</span><span class="st" style="background:{TEAL_D};">trail 21 EMA</span>'
           f'<span class="ar">→</span><span class="st" style="background:{TEAL};color:#0a0a0a;">no cap</span>')
    return f'<div class="tile">{_tag("Tested")}<div class="k">Exit</div><div class="lad">{lad}</div></div>'


LOOK_DIP = ('▲ beaten-down names', '▼ prior leaders')          # 8.5.0: RS1-b, DIP ZONE only (tested 2001-08)
LOOK_ELSE = ('no RS edge', 'setups frame the trade · the state carries it')   # 8.5.1; S2 2001-26, RS1-b''


def _look(bucket):
    """(chips, tag) for Where to look: the RS line only inside the DIP ZONE."""
    if bucket == 'IN DIP':
        return LOOK_DIP, 'Tested · 2001–08'
    return LOOK_ELSE, 'Tested (null)'


def _look_tile(st_, bucket):
    (a, b), tag = _look(bucket)
    sty = f'background:{TEAL};color:#0a0a0a;' if bucket == 'IN DIP' else ''
    body = (f'<div class="lad"><span class="st" style="{sty}">{_e(a)}</span>'
            f'<span class="st" style="color:{MUTED};">{_e(b)}</span></div>')
    return f'<div class="tile">{_tag(tag)}<div class="k">Where to look</div>{body}</div>'


def _hedge_tile(win, put):
    px = ''
    if put:
        ratio = put.get('model_ratio')
        px = (f'<div class="ft">60-session {put["strike"]:.0f} put ≈ {put["cost_pct"] * 100:.1f}% of SPY'
              + (f' · IV {put["iv"] * 100:.1f}%' if put.get('iv') == put.get('iv') and put.get('iv') else '')
              + f' · {put["date"]:%d %b}{" (seed)" if "seed" in put.get("src", "") else ""}</div>')
    if win:
        what, k = win
        body = (f'<div class="lad"><span class="st" style="background:{TEAL};color:#0a0a0a;">HEDGE WINDOW · {_e(what)}'
                f'{"" if k == 0 else f" · {k}d ago"}</span></div>'
                f'<div class="sm">~60-session put, ~5% below SPY</div>')
    else:
        body = (f'<div class="lad"><span class="st" style="color:{MUTED};">no hedge now</span>'
                f'<span class="st" style="color:{MUTED};">never roll puts</span></div>')
    return f'<div class="tile">{_tag("Modelled + real quotes")}<div class="k">Hedge</div>{body}{px}</div>'


# ── public ────────────────────────────────────────────────────────────────────────────────────────────────────
def card_html(spy_state: dict, ph: pd.Series, vix, put: dict = None, spy_ohlcv: pd.DataFrame = None,
              date_txt: str = '') -> str:
    code = int(ph.iloc[-1]) if ph is not None and len(ph) else None
    bucket, n = clock(ph)
    st_ = stance(code, filter200(spy_ohlcv) if spy_ohlcv is not None else None)
    vb = vol_band(vix)
    h = CSS + '<div class="v8"><div class="top">'
    if st_ is None:
        return h + ('<span class="k">MRS v8</span></div><div class="sm">SPY history not available right now: '
                    'no stance.</div></div>')
    h += (f'<span class="badge" style="background:{STANCE_COL[st_]};">{st_}</span>'
          f'<span class="k">MRS v8{(" · " + _e(date_txt)) if date_txt else ""}</span></div>')
    h += '<div class="grid">' + _clock_tile(bucket, n) + _phase_tile(code) + _vix_tile(vix, vb) + '</div>'
    h += '<div class="grid g2">' + _budget_tile(st_) + _tickets_tile(spy_state) + '</div>'
    h += '<div class="grid">' + _exit_tile() + _look_tile(st_, bucket) + _hedge_tile(hedge_window(ph), put) + '</div>'
    f = last_ftd(spy_ohlcv) if spy_ohlcv is not None else None
    foot = [f'sessions since the dip: {n}' if n is not None else 'no DIP ZONE in the history']
    if f is not None:
        foot.append(f'last follow-through day {pd.Timestamp(f):%d %b %Y} (commentary)')
    h += f'<div class="ft">{" · ".join(_e(x) for x in foot)}</div>'
    return (h + '</div>').replace('\n', ' ')


def explainer_md() -> str:
    """The words behind each tile (shown in a fold, closed by default)."""
    c = SWING_CAP
    return f"""
**The picture.** The chart shows SPY's last 160 sessions as candles. The background shows the phase:
- teal: DIP ZONE;
- muted teal: ticket A (rebound) open;
- copper: ticket B (breakout) open;
- no shading: no ticket.

The dotted copper line is today, labelled with the clock. The three rings, from left to right:
- **Clock:** sessions since the last DIP ZONE close; the ring is full at 60.
- **VIX:** the ring fills from 10 to 40; teal when CALM, orange when WIDE or STRESSED.
- **Open risk:** today's swing cap against the 4% maximum.

Under the rings, the **200-day filter** line shows SPY against its 200-day and whether a DIP ZONE keeps the normal budget.

**Stance (the big badge).** How much risk the swing book may carry. It is set by the SPY phase and, in the DIP ZONE, by the 200-day filter:
- **PRESS:** ticket A or B is open. 0.75% a trade, open risk up to 4%.
- **NORMAL:** SPY is in the DIP ZONE **and above a rising 200-day**. 0.75% a trade, up to 3%.
- **EASE:** no ticket is open, or SPY is in the DIP ZONE without a rising 200-day. 0.375% a trade, up to 1.5%, swing gross up to 60%.

Account simulation (500 paths, random entries, survivorship-free stocks), this budget against the previous v8 budget:
- 2009–2026: CAGR 9.6% vs 9.1%, max drawdown −22.7% vs −25.9%;
- 2001–2008 (out of sample, two bear markets): CAGR −1.1% vs −2.9%, max drawdown −31.9% vs −39.3%.

Raising the budget in the dip and in the 20 sessions after it did the opposite in both periods: drawdowns 5 to 15 points deeper. Dips can keep falling.

**200-day filter.** ON when SPY closes above its 200-day and the 200-day is higher than {F200_SLOPE} sessions ago. On the S&P 500 since 1963 (259 dip episodes):
- dips above a rising 200-day: 12% fell another 10% within 60 sessions; median +4.0% after 60 sessions;
- dips below a falling 200-day: 45% fell another 10%; median 0.0%.

The gap held in 1963–94 and 1995–2026, and at the same volatility. It tells which dips tend to keep falling, not which way the next move goes: the biggest rebounds also start below a falling 200-day.

**Clock.** Sessions since SPY's last close in the DIP ZONE, which is a close at least 1 ATR(14) below the 50-day.
- DIP: in the zone today.
- 1–20: early.
- 21–60: mid.
- 60+: late.

The clock tells you about the quality of each trade, not how big to be. **With SPY above a rising 200-day**, stock entries in the dip and the 20 sessions after it were the best and entries 60+ sessions into a trend the worst, in 2001–08 and in 2009–26 alike (about +0.15R early vs −0.06R late). **Below a falling 200-day the dip is not an edge:** in 2001–08 entries there lost about −0.3R. The ring turns orange in that case.

**SPY phase.** Which SPY ticket is open: DIP zone, A (rebound), B (breakout) or none. It sets the stance.

**VIX.** How wide the moves are, not which way they go. The VIX is compared with its long-run median since 1990 ({VIX_MEDIAN:.2f}).
- CALM: below the median.
- WIDE: up to 1.5× the median.
- STRESSED: above that.

A high VIX widens both tails. Because stops are set in ATR, size shrinks on its own; it is not a reason to be out.

**Swing open risk.** The total risk of open swing trades, measured to their current stops. A trade at break-even uses none. The bar shows the three levels: EASE {c['EASE'] * 100:.1f}%, NORMAL {c['NORMAL'] * 100:.1f}%, PRESS {c['PRESS'] * 100:.1f}%. Raise the levels only when the forward log shows your trades beating a random entry.

**SPY tickets.** Each ticket is one unit sized to risk {CORE_RISK * 100:.1f}% of NLV at its own stop. Account gross stays ≤ 100%, core first, no margin.
These are trades with crash protection, not the engine that beats SPY: out of sample they cut drawdowns but did not beat buy-and-hold.

**Exit.** Initial stop by structure or about 1.5–2 ATR(14). Past +2R, move the stop to break-even, then trail the 21 EMA or structure. No profit cap.
The best 1% of trades made more than the whole book (111% of net P&L).

**Where to look.** In the DIP ZONE itself, beaten-down names (the bottom 40% by 6-month return) beat the prior 6-month leaders: tested on 2001–08 (−0.31R for leaders), same direction in 2009–26. Once SPY has left the zone, there is no RS edge either way.
Use the setups (deep dip, 21 EMA or 50-day pullback, breakout, base) to find clean structures and to frame entry, stop and target. Their return came from the market state: in 2001–2026 none beat a random uptrend entry in the same state. So the state decides whether to trade and how big; the setup decides where the entry, stop and target go. Your own selection is what the forward log measures.

**Hedge.** The only hedge rule kept is a ~60-session put about 5% below SPY, bought in the {HEDGE_WINDOW} sessions after:

- a trend start (ticket B opens); or
- a trend end (ticket B closes and no ticket is open).

The supporting evidence:

- Real quotes (13 Nov 2026) matched the model for this put: model ÷ market 1.02.
- Short-dated and 10% out-of-the-money puts cost far more than the model assumed, so they are out.
- Never roll puts month after month: −0.60 to −0.82 return on premium at real prices.

The price on the tile comes from the daily put log.

**Footer.** The last follow-through day is commentary only: in C2, waiting for one did not help.

*Not on this card on purpose: the composite MRS score, the short book, RS ranking, margin, deep dips as a place to look.*
"""


# ══ 8.4.0 (16 Nov 2026): card layout B (Fabio's pick): header, SPY candles with the phase as bands + rings, A's detail row ══
def header_html(spy_state, ph: pd.Series, vix, date_txt: str = '', f200=None) -> str:
    import ui
    code = int(ph.iloc[-1]) if ph is not None and len(ph) else None
    st_ = stance(code, f200)
    if st_ is None:
        return ui.CSS() + '<div class="atn-k">MRS v8 <b>· SPY history not available right now: no stance</b></div>'
    rt, cap, g = BUDGET[st_]
    sub = f'{date_txt} · swing {rt * 100:.3g}% a trade · open risk ≤ {cap * 100:.1f}%' + (f' · swing gross ≤ {g * 100:.0f}%' if g < 1 else '')
    if code == 0 and st_ == 'EASE':
        sub += ' · DIP ZONE without a rising 200-day: small budget'
    return (f'<div style="display:flex;align-items:center;gap:14px;margin:2px 0 6px 0;">'
            f'<span style="font-family:Barlow,sans-serif;font-weight:800;font-size:30px;letter-spacing:.08em;padding:2px 18px;'
            f'border-radius:5px;color:#0a0a0a;background:{STANCE_COL[st_]};">{st_}</span>'
            f'<div class="atn-k" style="margin:0">MRS v8 <b>· {_e(sub)}</b></div></div>')


PH_BAND = {0: ('rgba(133,202,197,0.40)', 'DIP ZONE'), 1: ('rgba(100,153,149,0.24)', 'ticket A open'),
           2: ('rgba(227,180,142,0.16)', 'ticket B open')}


def rings_html(ph: pd.Series, vix, f200=None) -> str:
    """8.4.1: three compact rings (clock, VIX width, open-risk cap). 8.5.0: the clock reads with the 200-day filter."""
    code = int(ph.iloc[-1]) if ph is not None and len(ph) else None
    st_ = stance(code, f200)
    bucket, n = clock(ph)
    vb = vol_band(vix)
    def ring(frac, col, big, cap, sub):
        return (f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">'
                f'<div class="atn-ring sm" style="background:conic-gradient({col} {max(min(frac, 1), 0.02) * 360:.0f}deg,#24211e 0)"><div>{_e(big)}</div></div>'
                f'<div class="atn-tl" style="margin:0">{_e(cap)}</div><div class="atn-sm" style="font-size:11px;color:#6e675f;margin-top:-4px">{_e(sub)}</div></div>')
    if bucket is None or st_ is None:
        return '<div class="atn-sm">State not available right now.</div>'
    up = _fs(f200) == 'F-UP'
    ccol = (TEAL if up else ORANGE) if bucket in ('IN DIP', 'EARLY') else (INK if bucket == 'MID' else COPPER)
    cfrac = 1.0 if bucket == 'IN DIP' else (n or 0) / 60
    csub = ('early = best entries' if up else 'early ≠ edge: 200-day not up') if bucket in ('IN DIP', 'EARLY') else (
        'late = weakest entries' if bucket == 'LATE' else 'sessions since the dip')
    vcol = TEAL if vb == 'CALM' else ORANGE
    cap = SWING_CAP[st_]
    return ('<div style="display:flex;justify-content:space-around;margin:2px 0 0 0;">'
            + ring(cfrac, ccol, 'DIP' if bucket == 'IN DIP' else str(n), f'Clock · {bucket.lower()}', csub)
            + ring(((vix or 10) - 10) / 30, vcol, f'{vix:.1f}' if vix else '–', f'VIX · {(vb or "n/a").lower()}', f'median {VIX_MEDIAN:.1f}')
            + ring(cap / max(SWING_CAP.values()), STANCE_COL[st_], f'{cap * 100:.1f}%', 'Open risk', f'of a {max(SWING_CAP.values()) * 100:.0f}% max')
            + '</div>')


def filter_html(f200) -> str:
    """8.5.0: one line with SPY against its 200-day (DF1) and what it switches."""
    import ui
    if not isinstance(f200, dict):
        return (f'<div class="atn-sm" style="margin:10px 0 0 0">200-day filter not available (needs 220 closes): '
                f'a DIP ZONE gets the small budget.</div>')
    up = f200['state'] == 'F-UP'
    col = TEAL if up else ORANGE
    word = {'F-UP': 'above a rising 200-day', 'F-DN': 'below a falling 200-day', 'F-MIX': '200-day not rising with SPY above it'
            if f200['close'] > f200['sma'] else 'below a still-rising 200-day'}[f200['state']]
    eff = ('a DIP ZONE keeps the normal budget (3%) and the early-dip window is the best place to enter' if up else
           'a DIP ZONE gets the small budget (1.5%): dips here have kept falling')
    return (f'<div class="atn-tl" style="margin:12px 0 2px 0">200-day filter {ui.tag("Tested · SPX 1963–2026")}</div>'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:12px">'
            f'<span class="atn-chip" style="background:{col};color:#0a0a0a;font-weight:700">{"ON" if up else "OFF"} · SPY {_e(word)}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;color:#a89f95">{f200["close"]:,.2f} vs {f200["sma"]:,.2f} '
            f'({(f200["close"] / f200["sma"] - 1) * 100:+.1f}%) · slope {f200["slope"]:+.1f} / {F200_SLOPE}d</span></div>'
            f'<div class="atn-sm" style="font-size:11px;color:#6e675f;margin-top:3px">{_e(eff)}</div>')


def budget_html(ph: pd.Series, f200=None) -> str:
    """8.4.1: the three stance budgets as bars (cap on a 0-4% scale), today's highlighted."""
    import ui
    code = int(ph.iloc[-1]) if ph is not None and len(ph) else None
    now = stance(code, f200)
    rows = ''
    for st_, when in (('PRESS', 'A / B open'), ('NORMAL', 'DIP ZONE · 200-day up'), ('EASE', 'no ticket · other dips')):
        rt, cap, g = BUDGET[st_]
        on = st_ == now
        col = STANCE_COL[st_] if on else '#3a3632'
        rows += (f'<div style="display:grid;grid-template-columns:200px 1fr 196px;align-items:center;gap:10px;'
                 f'font-size:12px;color:{INK if on else DIM}"><span><b style="font-weight:{700 if on else 500}">{st_}</b> '
                 f'<span style="color:{MUTED if on else DIM};font-size:11px">{when}</span></span><div>'
                 + ui.bar([(0, cap / 0.04 * 100, col)], margin='5px 0', height=8)
                 + f'</div><span style="font-family:IBM Plex Mono,monospace;text-align:right">{rt * 100:.3g}% · ≤ {cap * 100:.1f}%'
                 + (f' · gross {g * 100:.0f}%' if g < 1 else '') + '</span></div>')
    return (f'<div class="atn-tl" style="margin:10px 0 2px 0">Swing budget by stance <span style="font-size:10px;'
            f'letter-spacing:.06em;color:#6e675f;text-transform:none">a trade · open-risk cap</span></div>' + rows)


def levels_svg(s: dict, W: int = 880, H: int = 104, f200=None) -> str:
    """8.4.1: where SPY is against the levels that change the stance (allocation state: dip line, 50-day, B trigger,
    open-ticket stops), as one price axis."""
    if not s or not _fin(s.get('close')):
        return ''
    c = float(s['close'])
    lv = []
    A, B = s.get('A') or {}, s.get('B') or {}
    if _fin(s.get('rb_level')):
        lv.append((float(s['rb_level']), 'DIP ZONE', stance(0, f200), TEAL))
    if _fin(s.get('sma50')):
        if A.get('open'):
            lv.append((float(s['sma50']), '50-day', 'A exit line', INK))
        elif s.get('in_selloff'):
            lv.append((float(s['sma50']), 'A unlocks', 'above 50-day', TEAL_D))
        else:
            lv.append((float(s['sma50']), '50-day', '', DIM))
    if A.get('open') and _fin(A.get('stop')):
        lv.append((float(A['stop']), 'A stop', '', ORANGE))
    if B.get('open'):
        if _fin(B.get('stop')):
            lv.append((float(B['stop']), 'B stop', '', ORANGE))
    elif _fin(s.get('bo_level')):
        lv.append((float(s['bo_level']), 'B buys', 'PRESS', COPPER))
    if not lv:
        return ''
    xs = [v for v, *_ in lv] + [c]
    lo, hi = min(xs), max(xs)
    pad = max((hi - lo) * 0.10, c * 0.004)
    lo, hi = lo - pad, hi + pad
    L, R = 8, 8
    X = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    ya = 58
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block">',
         f'<rect x="{L}" y="{ya - 5}" width="{W - L - R}" height="10" rx="3" fill="#1f1d1b"/>']
    dip = next((v for v, n, *_ in lv if n == 'DIP ZONE'), None)
    if dip is not None:
        g.append(f'<rect x="{L}" y="{ya - 5}" width="{max(X(dip) - L, 0):.1f}" height="10" rx="3" fill="rgba(133,202,197,0.38)"/>')
    bo = next((v for v, n, *_ in lv if n == 'B buys'), None)
    if bo is not None:
        g.append(f'<rect x="{X(bo):.1f}" y="{ya - 5}" width="{max(W - R - X(bo), 0):.1f}" height="10" rx="3" fill="rgba(227,180,142,0.32)"/>')
    for k, (v, name, sub, col) in enumerate(sorted(lv)):
        x = X(v); up = k % 2 == 0
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ya - 9}" y2="{ya + 9}" stroke="{col}" stroke-width="2"/>')
        ty = 18 if up else 96
        anchor = 'start' if x < 60 else ('end' if x > W - 60 else 'middle')
        g.append(f'<text x="{x:.1f}" y="{ty}" text-anchor="{anchor}" fill="{col}" font-size="11" font-family="Barlow, sans-serif" '
                 f'font-weight="600">{_e(name)}{(" → " + _e(sub)) if sub else ""}</text>'
                 f'<text x="{x:.1f}" y="{ty + (13 if up else -13)}" text-anchor="{anchor}" fill="#a89f95" font-size="10.5" '
                 f'font-family="IBM Plex Mono, monospace">{v:,.2f} ({(v / c - 1) * 100:+.1f}%)</text>')
    xc = X(c)
    g.append(f'<polygon points="{xc - 6:.1f},{ya - 15} {xc + 6:.1f},{ya - 15} {xc:.1f},{ya - 7}" fill="{INK}"/>'
             f'<line x1="{xc:.1f}" x2="{xc:.1f}" y1="{ya - 8}" y2="{ya + 8}" stroke="{INK}" stroke-width="3"/>')
    g.append('</svg>')
    return (f'<div class="atn-tl" style="margin:12px 0 0 0">Distance to the next stance <span style="font-size:10px;'
            f'letter-spacing:.06em;color:#6e675f;text-transform:none">SPY {c:,.2f} · closes decide</span></div>' + ''.join(g))


def detail_html(spy_state: dict, ph: pd.Series, put: dict = None, spy_ohlcv: pd.DataFrame = None, f200=None) -> str:
    """A's detail row: phase + tickets · exit + where to look · hedge (one tile each), then the footer."""
    import ui
    code = int(ph.iloc[-1]) if ph is not None and len(ph) else None
    bucket, n = clock(ph)
    pills = ''
    for p in (0, 1, 2, 3):
        on = code == p
        bg = {0: TEAL, 1: TEAL_D, 2: INK, 3: '#9b958f'}[p]
        sty = f'background:{bg};color:#0a0a0a;border-color:{bg};' if on else ''
        pills += f'<span class="pill" style="{sty}">{PH_NAME[p]}</span>'
    rows_, ft_ = _ticket_rows(spy_state)
    tk = rows_ + ft_
    t1 = (f'<div class="atn-tile"><div class="atn-tl">SPY phase · tickets {ui.tag("Tested · drawdown control")}</div>{pills}'
          f'<div style="margin-top:6px">{tk}</div></div>')
    (la, lb), ltag = _look(bucket)
    lsty = f'background:{TEAL};color:#0a0a0a;' if bucket == 'IN DIP' else ''
    look = (f'<span class="atn-chip" style="{lsty}">{_e(la)}</span><span class="atn-chip" style="color:{MUTED};">{_e(lb)}</span>'
            + ('<div class="atn-sm" style="font-size:11px;color:#6e675f">bottom 40% by 6-month return · not prior leaders</div>'
               if bucket == 'IN DIP' else '<div class="atn-sm" style="font-size:11px;color:#6e675f">use setups for entry, stop and target; '
               'their return came from the state</div>'))
    t2 = (f'<div class="atn-tile"><div class="atn-tl">Exit {ui.tag("Tested")}</div>'
          f'<span class="atn-chip">stop</span> → <span class="atn-chip">+2R break-even</span> → '
          f'<span class="atn-chip" style="background:{TEAL_D};">trail 21 EMA</span> → <span class="atn-chip" style="background:{TEAL};color:#0a0a0a;">no cap</span>'
          f'<div class="atn-tl" style="margin-top:10px">Where to look {ui.tag(ltag)}</div>{look}</div>')
    win = hedge_window(ph)
    px = ''
    if put:
        px = (f'<div class="atn-sm" style="margin-top:6px">60-session {put["strike"]:.0f} put ≈ <b style="font-family:IBM Plex Mono,monospace;color:#f3ece2">'
              f'{put["cost_pct"] * 100:.1f}%</b> of SPY' + (f' · IV {put["iv"] * 100:.1f}%' if put.get('iv') == put.get('iv') and put.get('iv') else '')
              + f' · {put["date"]:%d %b}{" (seed)" if "seed" in put.get("src", "") else ""}</div>')
    if win:
        what, k = win
        hb = (f'<span class="atn-chip" style="background:{TEAL};color:#0a0a0a;">HEDGE WINDOW · {_e(what)}{"" if k == 0 else f" · {k}d ago"}</span>'
              f'<div class="atn-sm">~60-session put, ~5% below SPY</div>')
    else:
        hb = f'<span class="atn-chip" style="color:{MUTED};">no hedge now</span><span class="atn-chip" style="color:{MUTED};">never roll puts</span>'
    t3 = f'<div class="atn-tile"><div class="atn-tl">Hedge {ui.tag("Modelled + real quotes")}</div>{hb}{px}</div>'
    f = last_ftd(spy_ohlcv) if spy_ohlcv is not None else None
    foot = [f'sessions since the dip: {n}' if n is not None else 'no DIP ZONE in the history']
    if f is not None:
        foot.append(f'last follow-through day {pd.Timestamp(f):%d %b %Y} (commentary)')
    return ('<div class="atn-row" style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:4px;">' + t1 + t2 + t3 + '</div>'
            + f'<div class="atn-sm" style="font-size:11px;color:#6e675f;margin-top:6px">{_e(" · ".join(foot))}</div>').replace('\n', ' ')


def ribbon_svg(spy_ohlcv: pd.DataFrame, ph: pd.Series, n: int = 120, W: int = 640, H: int = 270) -> str:
    """Static SVG: SPY daily candles for the last n sessions with the SPY phase as background bands (one HTML block, so
    the card keeps one shell; no Streamlit layout inside the card)."""
    if spy_ohlcv is None or ph is None or len(spy_ohlcv) < 20:
        return '<div class="atn-sm">SPY history not available right now.</div>'
    s = spy_ohlcv.rename(columns=str.title).iloc[-n:].copy()
    s.index = pd.to_datetime(s.index).normalize()
    p = ph.reindex(s.index).ffill().fillna(3).astype(int).values
    o, h, l, c = (s[k].astype(float).values for k in ('Open', 'High', 'Low', 'Close'))
    L, R, Tp, B = 6, 52, 22, 20
    pw, phh = W - L - R, H - Tp - B
    lo, hi = float(np.nanmin(l)), float(np.nanmax(h)); pad = (hi - lo) * 0.06; lo -= pad; hi += pad
    X = lambda i: L + (i + 0.5) * pw / len(s)
    Y = lambda v: Tp + (hi - v) / (hi - lo) * phh
    bw = max(pw / len(s) * 0.62, 1.2)
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="none" style="display:block">']
    band = {0: 'rgba(133,202,197,0.30)', 1: 'rgba(100,153,149,0.20)', 2: 'rgba(227,180,142,0.13)'}
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[j + 1] == p[i]:
            j += 1
        if p[i] in band:
            x0 = L + i * pw / len(s); x1 = L + (j + 1) * pw / len(s)
            g.append(f'<rect x="{x0:.1f}" y="{Tp}" width="{x1 - x0:.1f}" height="{phh}" fill="{band[p[i]]}"/>')
        i = j + 1
    for v in np.linspace(lo + pad, hi - pad, 4):                       # price grid + right labels
        g.append(f'<line x1="{L}" x2="{L + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#2a2724" stroke-width="1"/>'
                 f'<text x="{L + pw + 6}" y="{Y(v) + 3.5:.1f}" fill="#a89f95" font-size="10" font-family="IBM Plex Mono, monospace">{v:,.0f}</text>')
    last_m = None
    for k, d in enumerate(s.index):                                     # month labels
        if d.month != last_m:
            last_m = d.month
            if k > 2:
                g.append(f'<text x="{X(k):.1f}" y="{H - 5}" fill="#6e675f" font-size="10" font-family="IBM Plex Mono, monospace">{d:%b}</text>')
    for k in range(len(s)):
        up = c[k] >= o[k]; col = TEAL if up else ORANGE
        g.append(f'<line x1="{X(k):.1f}" x2="{X(k):.1f}" y1="{Y(h[k]):.1f}" y2="{Y(l[k]):.1f}" stroke="{col}" stroke-width="1"/>')
        y0, y1 = sorted((Y(o[k]), Y(c[k])))
        g.append(f'<rect x="{X(k) - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(y1 - y0, 0.8):.1f}" fill="{col}"/>')
    bucket, nn = clock(ph)
    lab = f'today · {bucket}{"" if not nn else " " + str(nn)}'
    xl = X(len(s) - 1)
    g.append(f'<line x1="{xl:.1f}" x2="{xl:.1f}" y1="{Tp - 4}" y2="{Tp + phh}" stroke="{COPPER}" stroke-width="1" stroke-dasharray="2,3"/>'
             f'<text x="{xl - 6:.1f}" y="{Tp - 8}" text-anchor="end" fill="{COPPER}" font-size="11" font-family="IBM Plex Mono, monospace">{_e(lab)}</text>')
    g.append('</svg>')
    key = ''.join(f'<span style="margin-right:14px"><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                  f'background:{c_};margin-right:5px;vertical-align:-1px"></span>{nm}</span>'
                  for c_, nm in (('rgba(133,202,197,0.85)', 'DIP ZONE'), ('rgba(100,153,149,0.8)', 'ticket A open'),
                                 ('rgba(227,180,142,0.7)', 'ticket B open'), ('#24211e', 'no ticket')))
    return f'<div class="atn-sm" style="font-size:11px;margin-bottom:2px">{key}</div>' + ''.join(g)


def card_v9_html(spy_state, ph: pd.Series, vix, put: dict = None, spy_ohlcv: pd.DataFrame = None, date_txt: str = '') -> str:
    """8.4.0 card (layout B + A's detail row) as one HTML block."""
    import ui
    if ph is None or not len(ph):
        return ui.CSS() + '<div class="atn-k">MRS v8 <b>· SPY history not available right now: no stance</b></div>'
    f200 = filter200(spy_ohlcv) if spy_ohlcv is not None else None
    return ('<div style="background:#121212;border:1px solid #2a2724;border-radius:6px;padding:14px 16px 10px 16px;margin:2px 0 6px 0">'
            + header_html(spy_state, ph, vix, date_txt, f200)
            + '<div style="display:grid;grid-template-columns:1fr 1.15fr;gap:10px;align-items:stretch">'
            + f'<div class="atn-tile"><div class="atn-tl">SPY · last 120 sessions with its phase {ui.tag("Tested")}</div>{ribbon_svg(spy_ohlcv, ph)}</div>'
            + f'<div class="atn-tile"><div class="atn-tl">Clock · width · budget {ui.tag("Tested · v8 sizing")}</div>{rings_html(ph, vix, f200)}'
            + filter_html(f200) + levels_svg(spy_state, f200=f200) + budget_html(ph, f200) + '</div></div>'
            + detail_html(spy_state, ph, put, spy_ohlcv, f200) + '</div>').replace('\n', ' ')
