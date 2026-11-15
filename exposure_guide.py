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
Stance        set by the SPY phase (v8 sizing run, 15 Nov 2026: the v7.1a phase budget beat every clock budget at the
              account level; pressing in the dip / early window raised the drawdown 4.6-9 points):
              PRESS = ticket A or B open: 0.75% a trade, swing open risk <= 4%;
              NORMAL = DIP ZONE: 0.75% a trade, <= 3%;
              EASE = no ticket: 0.375% a trade, <= 1.5%, swing gross <= 60%.
              The clock stays on the card as the per-trade context (C1: early entries beat late ones, in sample).
Budget        SPY tickets one unit each at 1.5% NLV risk (Fabio, 15 Nov 2026); account gross <= 100%, core first; no margin.
Exit          +2R -> break-even, then trail the 21 EMA or structure; no profit cap.
Hedge         only the ~60-session ~5% OTM SPY put, bought in the 5 sessions after a trend start (ticket B opens) or a
              trend end (B -> no ticket); never roll short-dated puts. Price: the daily put log (put_log.py).
Not on the card (decided 15 Nov 2026): composite MRS score, short book, FTD as a rule, RS tilt, margin, rolling puts.
"""
import html as _html
import numpy as np
import pandas as pd

TEAL, TEAL_D, ORANGE, COPPER, RED = '#85cac5', '#4f8f8a', '#f4a062', '#e3b48e', '#e07a5f'
INK, MUTED, DIM, LINE, BG = '#f3ece2', '#a89f95', '#6e675f', '#2a2724', '#121212'
VIX_MEDIAN = 17.56            # expanding median of the VIX since 1990-01-02, as of 2026-11-13
CORE_RISK = 0.015             # SPY ticket unit risk (Fabio, 15 Nov 2026)
BUDGET = {'PRESS': (0.0075, 0.04, 1.00), 'NORMAL': (0.0075, 0.03, 1.00), 'EASE': (0.00375, 0.015, 0.60)}  # risk a trade,
#          swing open-risk cap, swing gross cap (fractions of NLV). v8 sizing run (15 Nov 2026), 500 paths, 2009-26:
#          CAGR 9.1%, max DD -25.9%, beat a flat 3% cap on return / drawdown in 79% of paths.
SWING_CAP = {k: v[1] for k, v in BUDGET.items()}
SWING_RISK = BUDGET['NORMAL'][0]
STANCE_COL = {'PRESS': TEAL, 'NORMAL': INK, 'EASE': COPPER}
PH_NAME = {0: 'DIP', 1: 'A', 2: 'B', 3: 'NONE'}
HEDGE_WINDOW = 5              # sessions after a trend start / end in which the 60-session put is the rule


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


def stance(code):
    """Budget stance from the SPY phase code (0 DIP ZONE, 1 A open, 2 B open, 3 no ticket)."""
    if code is None:
        return None
    return 'PRESS' if code in (1, 2) else ('NORMAL' if code == 0 else 'EASE')


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
    return f'<span class="tag">{_e(t)}</span>'


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
    return (f'<div class="tile">{_tag("Per trade · in sample")}<div class="k">Clock · sessions since the dip</div>'
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
    segs = [(w1, TEAL_D), (w2, '#8a5a35'), (100 - w1 - w2, '#7a3a2e')]
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


def _tickets_tile(s):
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
    return (f'<div class="tile">{_tag("Tested · drawdown control")}<div class="k">SPY tickets · {CORE_RISK * 100:.1f}% risk each</div>'
            f'<div style="margin-top:5px;">{rows}</div>'
            + (f'<div class="ft">core ≈ {min(tot, 1) * 100:.0f}% NLV · gross ≤ 100%, core first</div>' if tot else
               '<div class="ft">gross ≤ 100%, core first · no margin</div>') + '</div>')


def _exit_tile():
    lad = (f'<span class="st">stop</span><span class="ar">→</span><span class="st">+2R: break-even</span>'
           f'<span class="ar">→</span><span class="st" style="background:{TEAL_D};">trail 21 EMA</span>'
           f'<span class="ar">→</span><span class="st" style="background:{TEAL};color:#0a0a0a;">no cap</span>')
    return f'<div class="tile">{_tag("Tested")}<div class="k">Exit</div><div class="lad">{lad}</div></div>'


def _look_tile(st_, bucket):
    if bucket in ('IN DIP', 'EARLY'):
        body = (f'<div class="lad"><span class="st" style="background:{TEAL};color:#0a0a0a;">▲ laggards · dips</span>'
                f'<span class="st" style="color:{MUTED};">▼ prior leaders</span></div>')
        tag = 'Tested'
    else:
        body = (f'<div class="lad"><span class="st">setups you trust</span>'
                f'<span class="st" style="color:{MUTED};">no leader edge</span></div>')
        tag = 'Tested (null)'
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
    st_ = stance(code)
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
**Stance (the big badge).** How much risk the swing book may carry. It is set by the SPY phase:
- **PRESS:** ticket A or B is open. 0.75% a trade, open risk up to 4%.
- **NORMAL:** SPY is in the DIP ZONE. 0.75% a trade, up to 3%.
- **EASE:** no ticket is open. 0.375% a trade, up to 1.5%, swing gross up to 60%.

In the v8 sizing run (500 paths, 2009–2026, random entries), this budget beat every alternative. Against a flat 3% cap it gave:
- CAGR 9.1% vs 7.5%;
- max drawdown −25.9% vs −28.8%;
- better return per unit of drawdown in 79% of paths.

Raising the budget in the dip and in the 20 sessions after it (the clock) did the opposite: drawdowns 4.6 to 9 points deeper. Dips can keep falling.

**Clock.** Sessions since SPY's last close in the DIP ZONE, which is a close at least 1 ATR(14) below the 50-day.
- DIP: in the zone today.
- 1–20: early.
- 21–60: mid.
- 60+: late.

The clock tells you about the quality of each trade, not how big to be. In C1, stock entries in the dip and the 20 sessions after it beat entries late in an old trend: +0.24R a trade and +2 points of P(≥ 3R). Late in an old trend, entries lost money on average, so be choosy there. Two limits: in sample, and it holds for stocks, not for the index.

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

**Where to look.** In the dip and early after it, prior laggards and deep dips beat the prior 6-month leaders (RS1).
Later in a trend, leaders showed no extra right tail. Your own selection is what the forward log measures.

**Hedge.** The only hedge rule kept is a ~60-session put about 5% below SPY, bought in the {HEDGE_WINDOW} sessions after:

- a trend start (ticket B opens); or
- a trend end (ticket B closes and no ticket is open).

The supporting evidence:

- Real quotes (13 Nov 2026) matched the model for this put: model ÷ market 1.02.
- Short-dated and 10% out-of-the-money puts cost far more than the model assumed, so they are out.
- Never roll puts month after month: −0.60 to −0.82 return on premium at real prices.

The price on the tile comes from the daily put log.

**Footer.** The last follow-through day is commentary only: in C2, waiting for one did not help.

*Not on this card on purpose: the composite MRS score, the short book, RS ranking, margin.*
"""
