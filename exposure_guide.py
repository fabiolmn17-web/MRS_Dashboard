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
              8.7.1: m is recomputed at every run (vix_1990.csv + every later close in mrs_history), set_vix_median().
              Width, not direction (V1, V4).
200-day       8.5.0 (16 Nov 2026, DF1): F-UP = SPY close > SMA200 and SMA200 above its value 20 sessions ago.
              SPX 1963-2026: dips below a falling 200-day fell another 10% within 60 sessions in 45% of episodes,
              vs 12% above a rising one (both eras, same vol mix).
Stance        set by the SPY phase and, in the DIP ZONE, the 200-day filter (v8 sizing 15 Nov; OOS 2001-08 + DF1 16 Nov; DS1 21 Nov: selection-adjusted):
              PRESS = ticket A or B open: 0.75% a trade, swing open risk <= 4%;
              NORMAL = DIP ZONE with SPY above a rising 200-day: 0.75% a trade, <= 3%;
              EASE = no ticket, or DIP ZONE without F-UP: 0.375% a trade, <= 1.5%, swing gross <= 60%.
              (V71F: beat the v8 budget in 2001-08 and 2009-26.) Pressing in the dip raised the drawdown in both eras.
              The clock stays on the card as per-trade context; it held only above a rising 200-day (2001-08 OOS).
Where to look In the DIP ZONE, beaten-down names (bottom 40% by 6-month return) beat prior leaders (RS1-b, tested
              2001-08, same direction 2009-26). After the zone, no RS edge. No chart pattern beat a random entry (S2).
Budget        SPY tickets one unit each at 1.5% NLV risk (Fabio, 15 Nov 2026); account gross <= 100%, core first; no margin.
Exit          +2R -> break-even, then trail the 21 EMA or structure; no profit cap.
Entry timing  8.11.0 (WD1, frozen and run 20 Nov 2026): act at the signal close or the next open, except after a weak
              Friday (the week's last session moves in the bottom 10% of all daily moves in ATR): long entries signalled
              at that close wait for Monday's open. SPY 1995-2026: Monday gap <= -0.5 ATR 18% vs 8% after other Fridays.
Hedge         8.11.0: no standing hedge rule. The ~60-session ~5% OTM put was dropped after CA1 (22 Nov 2026: on the core
              account it cost return in 2009-26 and deepened the drawdown in both eras). put_log.py keeps logging quotes.
Not on the card (decided 15 Nov 2026): composite MRS score, short book, FTD as a rule, RS tilt, margin, rolling puts, a standing put hedge (22 Nov),
deep dips as a place to look (16 Nov: no better than a random entry, 2001-26).
"""
import html as _html
from pathlib import Path
import numpy as np
import pandas as pd

TEAL, TEAL_D, ORANGE, COPPER, RED = '#85cac5', '#649995', '#f4a062', '#e3b48e', '#f4a062'   # 8.4.0: ATN tokens only
INK, MUTED, DIM, LINE, BG = '#f3ece2', '#a89f95', '#6e675f', '#2a2724', '#121212'
VIX_MEDIAN_FIXED = 17.56      # fallback: expanding median of the VIX since 1990, as of 2026-11-13
VIX_MEDIAN = VIX_MEDIAN_FIXED # 8.7.1: recomputed at every app run by set_vix_median(hist)
VIX_MEDIAN_ASOF = '2026-11-13'
VIX_BASE_PATH = Path(__file__).parent / 'vix_1990.csv'   # VIX closes 1990-01-03 .. 2026-11-13 (TradingView)
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
HEDGE_WINDOW = 5              # sessions after a trend start / end (the 60-session put rule; off the card since 8.11.0, CA1)
WEAK_FRI_PCT = 0.10           # 8.11.0 WD1: a move in the bottom 10% of all daily moves (in ATR) = weak
WEAK_FRI_FALLBACK = -0.89     # SPY 1993-2026: 10th percentile of (close - prior close) / prior ATR14 (Wilder), for short histories


def _fin(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _e(s):
    return _html.escape(str(s), quote=True)


# ── VIX median (8.7.1) ──────────────────────────────────────────────────────────────────────────────────────────
def vix_median(hist=None, base_path=None):
    """Expanding median of the VIX close since 1990, up to the last available close: the bundled vix_1990.csv plus
    every later close in mrs_history (nightly). Returns (median, as-of 'YYYY-MM-DD', n closes) or None."""
    base = pd.read_csv(base_path or VIX_BASE_PATH, parse_dates=['date']).set_index('date')['vix'].dropna()
    s = base
    if hist is not None and 'vix' in hist and 'date' in hist:
        h = pd.Series(pd.to_numeric(hist['vix'], errors='coerce').values, index=pd.to_datetime(hist['date'])).dropna()
        h = h[(h.index > base.index[-1]) & (h > 0)]
        h = h[~h.index.duplicated(keep='last')].sort_index()
        s = pd.concat([base, h])
    if len(s) < 1000:
        return None
    return float(s.median()), s.index[-1].strftime('%Y-%m-%d'), int(len(s))


def set_vix_median(hist=None):
    """Set VIX_MEDIAN for this run; keeps the fixed value if the data cannot be read."""
    global VIX_MEDIAN, VIX_MEDIAN_ASOF
    try:
        r = vix_median(hist)
    except Exception as e:
        print(f'[vix median] {e}')
        r = None
    if r:
        VIX_MEDIAN, VIX_MEDIAN_ASOF = r[0], r[1]
    return VIX_MEDIAN


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
    # 8.7.0: the close needed next session for ON. Tomorrow's 200-day = (sum of the last 199 closes + x) / 200; ON needs x above
    # it (x > sum199 / 199) and that 200-day above the one 20 sessions before tomorrow (x > 200 * sma[-20] - sum199).
    s199 = float(s.iloc[-199:].sum())
    need = max(s199 / 199, 200 * float(sma.iloc[-F200_SLOPE]) - s199)
    return dict(state=st_, close=c, sma=m, slope=slope, need=need)


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


# ── 8.8.0 (19 Nov 2026): plain-language risk label, from the same inputs as stance() ─────────────────────────────
#    The SPY phase gives the short-term side, the 200-day filter the long-term backdrop. A translation of the v8
#    states into market language (Descriptive); the stance alone sets the size. The VIX band sits beside it as width.
RISK_COL = {'RISK-ON': TEAL, 'RISK-OFF': ORANGE, 'NEUTRAL': MUTED}
RISK_LINE = {
    ('RISK-ON', 'TREND'): 'Breakout ticket open: the trend is running. Stock entries 60+ sessions into a trend did worst.',
    ('RISK-ON', 'RECOVERY'): 'Rebound ticket open after a dip: it often still feels like risk-off.',
    ('NEUTRAL', 'BEAR RALLY'): 'A ticket is open below a falling 200-day: the tickets trade it, but the long-term trend is down.',
    ('RISK-OFF', 'BUYABLE DIP'): 'Dip above a rising 200-day: 36% of such dips fell another 3% within 20 sessions (any day 39%); '
                                 'stock entries here were the best (+0.30R, both eras). Normal size.',
    ('RISK-OFF', 'LINE LOST'): 'Dip below a still-rising 200-day: 47% fell another 3% within 20 sessions (any day 39%); '
                               'the upside was about the same. Not a bear call.',
    ('RISK-OFF', 'LINE NOT RISING'): 'Dip above a 200-day that is no longer rising: only 4 such dips since 1963; pooled with '
                                     'the line-lost dips, 44% fell another 3% within 20 sessions (any day 39%).',
    ('RISK-OFF', 'MIXED 200-DAY'): 'Dip with the 200-day lost or not rising: 44% fell another 3% within 20 sessions (any day 39%).',
    ('RISK-OFF', 'DOWNTREND'): 'Below a falling 200-day: in dips there, 67% fell another 3% within 20 sessions and 45% '
                               'another 10% within 60 (any day 39% / 14%).',
    ('RISK-OFF', 'DIP'): 'DIP ZONE; the 200-day filter is not available right now.',
    ('NEUTRAL', 'NO TICKET'): 'No trend ticket and no dip: the small budget.',
}


def risk_label(code, f200=None):
    """Label for the SPY phase code (0 DIP ZONE, 1 A open, 2 B open, 3 no ticket) and the 200-day filter (a filter200()
    dict or 'F-UP' / 'F-MIX' / 'F-DN'). Returns dict(side, word, label, col, line, stance) or None without a phase."""
    if code is None:
        return None
    fs = _fs(f200)
    below = (isinstance(f200, dict) and _fin(f200.get('close')) and _fin(f200.get('sma'))
             and f200['close'] < f200['sma'])
    if code in (1, 2):
        side, word = ('NEUTRAL', 'BEAR RALLY') if fs == 'F-DN' else ('RISK-ON', 'TREND' if code == 2 else 'RECOVERY')
    elif code == 0:
        if fs == 'F-UP':
            side, word = 'RISK-OFF', 'BUYABLE DIP'
        elif fs == 'F-DN':
            side, word = 'RISK-OFF', 'DOWNTREND'
        elif fs == 'F-MIX':
            side, word = 'RISK-OFF', ('LINE LOST' if below else ('LINE NOT RISING' if isinstance(f200, dict) else 'MIXED 200-DAY'))
        else:
            side, word = 'RISK-OFF', 'DIP'
    else:
        side, word = ('RISK-OFF', 'DOWNTREND') if fs == 'F-DN' else ('NEUTRAL', 'NO TICKET')
    line = RISK_LINE[(side, word)]
    if code != 0 and word == 'DOWNTREND':
        line = 'No ticket, below a falling 200-day: the long-term trend is down. Small budget.'
    return dict(side=side, word=word, label=f'{side} · {word}', col=RISK_COL[side], line=line, stance=stance(code, f200))


def risk_label_html(code, f200=None, vix=None) -> str:
    """One line under the stance badge: the label chip, the VIX band as width, the line and the evidence tag."""
    import ui
    r = risk_label(code, f200)
    if r is None:
        return ''
    vb = vol_band(vix)
    width = (f'<span style="font-size:12px;letter-spacing:.08em;color:{MUTED};border:1px solid {DIM};padding:2px 8px;'
             f'border-radius:4px;white-space:nowrap;">VIX {_e(vb)} · width, not direction</span>') if vb else ''
    return (f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 8px 0;">'
            f'<span style="font-weight:700;font-size:13px;letter-spacing:.1em;padding:3px 10px;border-radius:4px;'
            f'white-space:nowrap;color:{r["col"]};border:1px solid {r["col"]};background:{r["col"]}18;">{_e(r["label"])}</span>'
            f'{width}<span style="color:{MUTED};font-size:13px;">{_e(r["line"])}</span>'
            f'{ui.tag("Descriptive")}</div>')


def risk_label_md() -> str:
    """The full translation table for the explainer fold."""
    rows = [('Ticket B open (breakout)', 'RISK-ON · TREND', 'RISK-ON · TREND', 'NEUTRAL · BEAR RALLY', 'PRESS'),
            ('Ticket A open (rebound)', 'RISK-ON · RECOVERY', 'RISK-ON · RECOVERY', 'NEUTRAL · BEAR RALLY', 'PRESS'),
            ('DIP ZONE', 'RISK-OFF · BUYABLE DIP', 'RISK-OFF · LINE LOST (below a rising 200-day) / LINE NOT RISING',
             'RISK-OFF · DOWNTREND', 'NORMAL above a rising 200-day, else EASE'),
            ('No ticket', 'NEUTRAL · NO TICKET', 'NEUTRAL · NO TICKET', 'RISK-OFF · DOWNTREND', 'EASE')]
    t = ('| SPY phase | Above a rising 200-day | Lost, or not rising | Below a falling 200-day | Stance |\n'
         '|---|---|---|---|---|\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows))
    return ('**Risk label.** The same two inputs as the stance (SPY phase and the 200-day filter), in market language: the '
            'phase gives the short-term side, the 200-day the long-term backdrop. It describes the market; the stance '
            'alone sets the size. A buyable dip is risk-off in market terms and the best entry state in the data '
            '(normal size); a bear rally keeps PRESS because the tickets trade it. The VIX band is shown beside the label '
            'as width and never changes it. Descriptive: only the DIP ZONE cells carry measured odds (DF1 / DF1b).\n\n' + t + '\n')


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
    return (f'<div class="tile">{_tag("Tested · 2001–26")}<div class="k">Clock · sessions since the dip</div>'
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
    return (f'<div class="tile">{_tag("Tested · selection-adjusted")}<div class="k">Swing open risk</div>'
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





# ── 8.11.0 (22 Nov 2026): entry timing after a weak Friday (WD1) ─────────────────────────────
def _nyse_holidays(years):
    """NYSE full-day holidays (the federal ones the exchange keeps, plus Good Friday)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from dateutil.easter import easter
    y0, y1 = min(years), max(years)
    fed = USFederalHolidayCalendar().holidays(f'{y0}-01-01', f'{y1}-12-31')
    fed = [d for d in fed if not (d.month == 10 or (d.month == 11 and d.day < 20))]     # no Columbus / Veterans Day
    gf = [pd.Timestamp(easter(y)) - pd.Timedelta(days=2) for y in range(y0, y1 + 1)]
    return set(pd.DatetimeIndex(fed).normalize()) | set(gf)


def _next_session(d):
    d = pd.Timestamp(d).normalize(); hol = _nyse_holidays([d.year, d.year + 1])
    n = d + pd.Timedelta(days=1)
    while n.dayofweek >= 5 or n in hol:
        n += pd.Timedelta(days=1)
    return n


def weak_friday(spy_ohlcv: pd.DataFrame, now=None):
    """8.11.0 (WD1, frozen and run 20 Nov 2026). For the last settled SPY session, if it is the last session of its week:
    dict(date, mv, pct, thr, weak), where mv = (close - prior close) / prior ATR14 (Wilder) and pct = its percentile among
    all daily moves in the history (incl. this one); weak = pct <= 10%. Returns None mid-week, or once the next week's
    session is trading (the rule is about Monday's open)."""
    if spy_ohlcv is None or len(spy_ohlcv) < 30:
        return None
    s = spy_ohlcv.rename(columns=str.title)
    nxt_live = live_bar(spy_ohlcv, now)
    if nxt_live:
        s = s.iloc[:-1]
    h, l, c = (s[k].astype(float) for k in ('High', 'Low', 'Close'))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    mv = ((c - c.shift()) / atr.shift()).dropna()
    if not len(mv):
        return None
    d = pd.Timestamp(mv.index[-1]); d = (d.tz_localize(None) if d.tzinfo else d).normalize()
    nxt = _next_session(d)
    if nxt - pd.Timedelta(days=nxt.dayofweek) <= d - pd.Timedelta(days=d.dayofweek):
        return None                                   # not the week's last session
    if nxt_live:
        return None                                   # the new week's session is already trading
    x = float(mv.iloc[-1])
    if len(mv) >= 250:
        pct, thr = float((mv <= x).mean()), float(mv.quantile(WEAK_FRI_PCT))
        weak = pct <= WEAK_FRI_PCT
    else:
        pct, thr = float('nan'), WEAK_FRI_FALLBACK
        weak = x <= thr
    return dict(date=d, mv=x, pct=pct, thr=thr, weak=bool(weak))


WF_STATS = 'Monday gap ≥ 0.5 ATR down: 18% vs 8% after other Fridays · median gap still slightly up · the session recovers part'


def _timing_tile(wf):
    """Legacy card (card_html): entry timing tile."""
    if wf and wf['weak']:
        body = (f'<div class="lad"><span class="st" style="background:{ORANGE};color:#0a0a0a;">WEAK FRIDAY · {wf["mv"]:+.2f} ATR</span>'
                f'<span class="st">long entries → Monday’s open</span></div><div class="sm">{_e(WF_STATS)}</div>')
    else:
        body = ('<div class="lad"><span class="st">signal close or next open</span></div>'
                '<div class="sm">after a weak Friday: long entries wait for Monday’s open</div>')
    return f'<div class="tile">{_tag("Tested · 1962–2026")}<div class="k">Entry timing</div>{body}</div>'


def timing_html(wf) -> str:
    """8.11.0 detail-row tile: entry timing (WD1) + the no-standing-hedge line (CA1)."""
    import ui
    thr = wf['thr'] if wf else WEAK_FRI_FALLBACK
    if wf and wf['weak']:
        body = (f'<span class="atn-chip" style="background:{ORANGE};color:#0a0a0a;">WEAK FRIDAY · {wf["mv"]:+.2f} ATR · {wf["date"]:%d %b}</span>'
                f'<span class="atn-chip" style="background:{TEAL};color:#0a0a0a;">long entries → Monday’s open</span>'
                f'<div class="atn-sm">{_e(WF_STATS)}</div>')
    else:
        fri = (f'<div class="atn-sm" style="font-size:11px;color:#6e675f">{wf["date"]:%a %d %b}: {wf["mv"]:+.2f} ATR, not a weak Friday</div>'
               if wf else '')
        body = ('<span class="atn-chip">signal close or next open</span>'
                f'<div class="atn-sm">after a weak Friday (move ≤ {thr:+.2f} ATR, bottom 10%): long entries wait for Monday’s open</div>{fri}')
    hedge = ('<div class="atn-sm" style="font-size:11px;color:#6e675f;margin-top:8px">hedge: no standing rule '
             '(the 60-session put was dropped after the account test, 22 Nov)</div>')
    return f'<div class="atn-tile"><div class="atn-tl">Entry timing {ui.tag("Tested · 1962–2026")}</div>{body}{hedge}</div>'


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
    return LOOK_ELSE, 'Tested · no edge'


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
    h += '<div class="grid">' + _exit_tile() + _look_tile(st_, bucket) + _timing_tile(weak_friday(spy_ohlcv)) + '</div>'
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

**VIX.** How wide the moves are, not which way they go. The VIX is compared with its long-run median since 1990 ({VIX_MEDIAN:.2f} on {VIX_MEDIAN_ASOF}), recomputed with every close.
- CALM: below the median.
- WIDE: up to 1.5× the median.
- STRESSED: above that.

A high VIX widens both tails. Because stops are set in ATR, size shrinks on its own; it is not a reason to be out.

{risk_label_md()}
**Swing open risk.** The total risk of open swing trades, measured to their current stops. A trade at break-even uses none. The bar shows the three levels: EASE {c['EASE'] * 100:.1f}%, NORMAL {c['NORMAL'] * 100:.1f}%, PRESS {c['PRESS'] * 100:.1f}%. Raise the levels only when the forward log shows your trades beating a random entry.

**SPY tickets.** Each ticket is one unit sized to risk {CORE_RISK * 100:.1f}% of NLV at its own stop. Account gross stays ≤ 100%, core first, no margin.
These are trades with crash protection, not the engine that beats SPY: out of sample they cut drawdowns but did not beat buy-and-hold.

**Exit.** Initial stop by structure or about 1.5–2 ATR(14). Past +2R, move the stop to break-even, then trail the 21 EMA or structure. No profit cap.
The best 1% of trades made more than the whole book (111% of net P&L).

**Where to look.** In the DIP ZONE itself, beaten-down names (the bottom 40% by 6-month return) beat the prior 6-month leaders: tested on 2001–08 (−0.31R for leaders), same direction in 2009–26. Once SPY has left the zone, there is no RS edge either way.
Use the setups (deep dip, 21 EMA or 50-day pullback, breakout, base) to find clean structures and to frame entry, stop and target. Their return came from the market state: in 2001–2026 none beat a random uptrend entry in the same state. So the state decides whether to trade and how big; the setup decides where the entry, stop and target go. Your own selection is what the forward log measures.

**Entry timing.** Act on a signal at its close or at the next open. One exception: when the week ends with a weak Friday (SPY's move that day in the bottom 10% of all daily moves, measured in ATR), long entries signalled at that close wait for Monday's open.
- In SPY since 1995, Monday then opened 0.5 ATR or more lower on 18% of those weekends, against 8% after other Fridays; on average −0.11 ATR worse. It held in both eras and at the same VIX mix. Weekday nights after a weak day bounce instead.
- The median Monday gap is still slightly positive and Monday's session recovers part of it: the rule avoids a tail, not a typical loss.
- Strong Fridays say nothing about Monday. Tested on the index; for single stocks it applies by extension.

**Hedge.** No standing hedge rule. The ~60-session put about 5% below SPY was dropped after the account test (22 Nov 2026): on the core account it cost return in 2009–26 and deepened the drawdown in both eras. Never roll puts month after month (−0.60 to −0.82 return on premium at real prices). The daily put log keeps collecting real quotes for any later study.

**Footer.** The last follow-through day is commentary only: in C2, waiting for one did not help.

*Not on this card on purpose: the composite MRS score, the short book, RS ranking, margin, deep dips as a place to look, a standing put hedge.*
"""


# ══ 8.4.0 (16 Nov 2026): card layout B (Fabio's pick): header, SPY candles with the phase as bands + rings, A's detail row ══
CLOSE_ET = (16, 5)            # 8.5.3: after 16:05 New York time the day's bar counts as a close


def live_bar(spy_ohlcv: pd.DataFrame, now=None) -> bool:
    """True when the last SPY bar is today's session and the US close has not happened yet (New York time)."""
    if spy_ohlcv is None or not len(spy_ohlcv):
        return False
    try:
        from zoneinfo import ZoneInfo
        now = now or pd.Timestamp.now(tz=ZoneInfo('America/New_York'))
        last = pd.Timestamp(spy_ohlcv.index[-1])
        last = last.tz_convert('America/New_York') if last.tzinfo else last
        return last.date() == now.date() and (now.hour, now.minute) < CLOSE_ET
    except Exception:
        return False


def provisional(spy_state, ph: pd.Series, spy_ohlcv: pd.DataFrame, now=None):
    """8.5.3: while the session is open, the stance at the last close (settled) and the live reading (provisional).
    Returns None after the close, else dict(settled, live, line, price, date)."""
    if ph is None or len(ph) < 2 or not live_bar(spy_ohlcv, now):
        return None
    f_now, f_prev = filter200(spy_ohlcv), filter200(spy_ohlcv, upto=-1)
    settled, live = stance(int(ph.iloc[-2]), f_prev), stance(int(ph.iloc[-1]), f_now)
    s = spy_state or {}
    price = float(spy_ohlcv.rename(columns=str.lower)['close'].iloc[-1])
    line = float(s['rb_level']) if _fin(s.get('rb_level')) else None
    return dict(settled=settled, live=live, line=line, price=price, code=int(ph.iloc[-1]), code_prev=int(ph.iloc[-2]),
                date=pd.Timestamp(spy_ohlcv.index[-1]), f_prev=f_prev)


def _prov_html(pv) -> str:
    if not pv:
        return ''
    now = f'(now {pv["price"]:,.2f})'
    if pv['live'] == pv['settled']:
        txt = f'live bar agrees: {pv["live"]} {now} · provisional until the close'
    elif pv['code'] == 0 and pv['code_prev'] != 0 and pv['line']:
        txt = f'live: {pv["live"]} if SPY closes below {pv["line"]:,.2f} {now} · provisional'
    elif pv['code_prev'] == 0 and pv['code'] != 0 and pv['line']:
        txt = f'live: {pv["live"]} if SPY closes above {pv["line"]:,.2f} {now} · provisional'
    else:
        txt = f'live: {pv["live"]} if this holds at the close {now} · provisional'
    col = STANCE_COL.get(pv['live'], INK)
    return (f'<span style="font-family:Barlow,sans-serif;font-weight:700;font-size:13px;letter-spacing:.06em;padding:3px 10px;'
            f'border:1px dashed {col};border-radius:5px;color:{col};white-space:nowrap">{_e(pv["date"].strftime("%a %d %b") + " · " + txt)}</span>')


def header_html(spy_state, ph: pd.Series, vix, date_txt: str = '', f200=None, prov=None) -> str:
    import ui
    code = int(ph.iloc[-1]) if ph is not None and len(ph) else None
    st_ = stance(code, f200)
    if prov:                                   # 8.5.3: the badge shows the stance at the last close while the session is open
        st_, code = prov['settled'], prov['code_prev']
    if st_ is None:
        return ui.CSS() + '<div class="atn-k">MRS v8 <b>· SPY history not available right now: no stance</b></div>'
    rt, cap, g = BUDGET[st_]
    sub = f'{date_txt} · swing {rt * 100:.3g}% a trade · open risk ≤ {cap * 100:.1f}%' + (f' · swing gross ≤ {g * 100:.0f}%' if g < 1 else '')
    if code == 0 and st_ == 'EASE':
        sub += ' · DIP ZONE without a rising 200-day: small budget'
    return (f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:2px 0 6px 0;">'
            f'<span style="font-family:Barlow,sans-serif;font-weight:800;font-size:30px;letter-spacing:.08em;padding:2px 18px;'
            f'border-radius:5px;color:#0a0a0a;background:{STANCE_COL[st_]};">{st_}</span>'
            f'<div class="atn-k" style="margin:0">MRS v8 <b>· {_e(sub)}</b></div>' + _prov_html(prov) + '</div>'
            + risk_label_html(code, prov.get('f_prev') if prov else f200, vix))   # 8.8.0: same inputs as the badge


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


# 8.7.0 (DF1b, 18 Nov 2026, Descriptive): S&P 500 dip episodes 1963-2026 (first close <= 50-day - 1 ATR), share that fell
# a further 3 / 5 / 7.5 / 10% below the dip close (lows) within 20 sessions, and 10% within 60, by the 200-day state at the dip.
# 'ALL' = every 20th (60th) session, any state. Gap F-UP vs F-DN excludes 0 at every size (year-block bootstrap), both halves.
DF1B = {'F-UP': dict(n=168, d=(0.36, 0.17, 0.08, 0.04), d60=0.12),
        'F-MIX': dict(n=34, d=(0.44, 0.32, 0.12, 0.06), d60=0.21),
        # 8.12.1 (25 Nov 2026): the line-lost part of F-MIX on its own (SPX below a still-rising 200-day; 30 of the 34).
        # The other 4 (above a 200-day not rising: 1963, 1988 x2, 1994) are too few to show alone.
        'F-LOST': dict(n=30, d=(0.47, 0.37, 0.13, 0.07), d60=0.23),
        'F-DN': dict(n=55, d=(0.67, 0.44, 0.29, 0.16), d60=0.45),
        'ALL': dict(n=804, d=(0.39, 0.20, 0.09, 0.05), d60=0.14)}
# 8.15.0 (27 Nov 2026, SV1 Part A: out of sample, 113 S&P 500 dip episodes 2001-26, each scored with the data known then):
# the >= 10% within 60 sessions column ran low in both halves (2001-13 and 2014-26). Forecast vs observed, all dips and by row.
DF1B_60_SINCE2001 = dict(n=113, fc=0.18, obs=0.29, lost=(4, 10), dn_fc=0.40, dn_obs=0.63)
DF1B_ROWS = (('F-UP', 'Above a rising 200-day'), ('F-MIX', 'Mixed: line lost, or not rising'), ('F-DN', 'Below a falling 200-day'),
             ('ALL', 'Any day'))
DF1B_NAMES = dict(DF1B_ROWS, **{'F-LOST': 'Line lost: below a still-rising 200-day',
                                'F-MIX': 'Mixed: 200-day not rising (pooled with line lost; 4 such dips)'})


def df1b_key(f200) -> str:
    """8.12.1: the DF1b row for today's filter state. F-MIX splits by where the close sits: below the 200-day = the
    line-lost row (30 dips); above a 200-day that is not rising = the pooled mixed row (only 4 such dips on their own)."""
    st_ = f200['state'] if isinstance(f200, dict) else f200
    if st_ != 'F-MIX':
        return st_
    above = isinstance(f200, dict) and _fin(f200.get('close')) and _fin(f200.get('sma')) and f200['close'] > f200['sma']
    return 'F-MIX' if above else 'F-LOST'


def drop_odds_html(f200, in_dip: bool = True) -> str:
    """8.7.0: what a dip in today's 200-day state has done next (DF1b), the row of today's state highlighted, plus the close
    needed next session to keep (or turn) the filter ON."""
    import ui
    if not isinstance(f200, dict):
        return ''
    st_ = df1b_key(f200)
    up = st_ == 'F-UP'
    mid = 'F-MIX' if st_ == 'F-MIX' else 'F-LOST'
    hi = TEAL if up else ORANGE
    cells = 'display:grid;grid-template-columns:minmax(200px,1.9fr) repeat(4,minmax(44px,0.55fr)) minmax(70px,0.8fr);gap:3px 14px;align-items:center;font-size:13.5px;max-width:1000px'
    head = ''.join(f'<span style="text-align:right;color:#6e675f">{h}</span>' for h in ('≥ 3%', '≥ 5%', '≥ 7.5%', '≥ 10%', '≥ 10% · 60d'))
    rows = f'<div style="{cells};margin-top:4px"><span style="color:#6e675f">further fall from the dip close · 20 sessions</span>{head}</div>'
    for k in ('F-UP', mid, 'F-DN', 'ALL'):
        v, name = DF1B[k], DF1B_NAMES[k]
        on = k == st_
        col = hi if on else ('#8a847c' if k != 'ALL' else '#6e675f')
        w = '700' if on else '400'
        nums = ''.join(f'<span style="text-align:right;font-family:IBM Plex Mono,monospace;color:{col};font-weight:{w}">{x * 100:.0f}%</span>'
                       for x in (*v['d'], v['d60']))
        rows += (f'<div style="{cells}"><span style="color:{col};font-weight:{w}">{_e(name)}'
                 f'{" ◀ today" if on else ""}<span style="color:#6e675f;font-weight:400"> · {v["n"]}</span></span>{nums}</div>')
    d3 = DF1B[st_]['d'][0] * 100
    lead = ('SPY is in the DIP ZONE in this state' if in_dip else 'If SPY dips into the zone in this state')
    txt = (f'{lead}: the S&P 500 went another 3% lower within 20 sessions in {d3:.0f}% of such dips '
           f'(any day {DF1B["ALL"]["d"][0] * 100:.0f}%).')
    nd = f200.get('need')
    if nd is not None and nd == nd:
        if up:
            txt += f' A close above {nd:,.2f} next session keeps the filter ON; below it, the line-lost row ({DF1B["F-LOST"]["d"][0] * 100:.0f}%).'
        elif nd <= f200['close'] * 1.03:
            txt += f' A close above {nd:,.2f} next session turns it ON.'
        else:
            txt += f' ON needs a close above {nd:,.2f}: the 200-day slope rules it out next session.'
    k6 = DF1B_60_SINCE2001
    note = (f'<div class="atn-sm" style="font-size:11px;margin-top:5px;padding-left:8px;border-left:2px solid {ORANGE};border-radius:0">'
            f'≥ 10% · 60d has run low since 2001: of {k6["n"]} dips, {k6["obs"] * 100:.0f}% fell another 10% within 60 sessions against '
            f'{k6["fc"] * 100:.0f}% forecast (line lost {k6["lost"][0]} of {k6["lost"][1]}; below a falling 200-day '
            f'{k6["dn_obs"] * 100:.0f}% vs {k6["dn_fc"] * 100:.0f}%). Read this column as a floor.</div>')
    return (f'<div style="margin-top:8px"><div class="atn-tl" style="margin:0 0 2px 0">What dips did next, by the 200-day '
            f'{ui.tag("Descriptive · SPX dips 1963–2026")}</div>{rows}{note}'
            f'<div class="atn-sm" style="font-size:11px;margin-top:4px">{_e(txt)} The upside was about the same in every row: '
            f'losing the line mainly adds downside.</div></div>')


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
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:14px">'
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
        rows += (f'<div style="display:grid;grid-template-columns:260px 1fr 250px;align-items:center;gap:12px;'
                 f'font-size:14px;color:{INK if on else DIM}"><span><b style="font-weight:{700 if on else 500}">{st_}</b> '
                 f'<span style="color:{MUTED if on else DIM};font-size:12.5px">{when}</span></span><div>'
                 + ui.bar([(0, cap / 0.04 * 100, col)], margin='6px 0', height=11)
                 + f'</div><span style="font-family:IBM Plex Mono,monospace;text-align:right">{rt * 100:.3g}% · ≤ {cap * 100:.1f}%'
                 + (f' · gross {g * 100:.0f}%' if g < 1 else '') + '</span></div>')
    return (f'<div class="atn-tl" style="margin:10px 0 2px 0">Swing budget by stance <span style="font-size:10px;'
            f'letter-spacing:.06em;color:#6e675f;text-transform:none">a trade · open-risk cap</span></div>' + rows)


def levels_svg(s: dict, W: int = 1000, H: int = 108, f200=None) -> str:
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
    g = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="display:block;max-width:1480px;margin:0 auto">',   # 8.10.2: wider tile, text ~14px at 2560
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


def detail_html(spy_state: dict, ph: pd.Series, put: dict = None, spy_ohlcv: pd.DataFrame = None, f200=None, now=None) -> str:
    """A's detail row: phase + tickets · exit + where to look · entry timing (8.11.0; was hedge), then the footer.
    `put` is kept for compatibility and no longer shown."""
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
    t3 = timing_html(weak_friday(spy_ohlcv, now))                     # 8.11.0: replaces the hedge tile (CA1, WD1)
    f = last_ftd(spy_ohlcv) if spy_ohlcv is not None else None
    foot = [f'sessions since the dip: {n}' if n is not None else 'no DIP ZONE in the history']
    if f is not None:
        foot.append(f'last follow-through day {pd.Timestamp(f):%d %b %Y} (commentary)')
    return ('<div class="atn-row" style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:4px;">' + t1 + t2 + t3 + '</div>'
            + f'<div class="atn-sm" style="font-size:11px;color:#6e675f;margin-top:6px">{_e(" · ".join(foot))}</div>').replace('\n', ' ')


def ribbon_svg(spy_ohlcv: pd.DataFrame, ph: pd.Series, n: int = 120, W: int = 600, H: int = 560) -> str:
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
    # 8.13.0 (26 Nov 2026, Fabio): TradingView hollow candles: colour by the prior close, hollow body when close >= open
    pcl = spy_ohlcv.rename(columns=str.title)['Close'].astype(float).shift().iloc[-n:].values
    for k in range(len(s)):
        prv = pcl[k] if np.isfinite(pcl[k]) else o[k]
        col = TEAL if c[k] >= prv else ORANGE
        y0, y1 = sorted((Y(o[k]), Y(c[k])))
        g.append(f'<line x1="{X(k):.1f}" x2="{X(k):.1f}" y1="{Y(h[k]):.1f}" y2="{y0:.1f}" stroke="{col}" stroke-width="1"/>'
                 f'<line x1="{X(k):.1f}" x2="{X(k):.1f}" y1="{y1:.1f}" y2="{Y(l[k]):.1f}" stroke="{col}" stroke-width="1"/>')
        body = (f'fill="none" stroke="{col}" stroke-width="0.9"' if c[k] >= o[k] else f'fill="{col}"')
        g.append(f'<rect x="{X(k) - bw / 2:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{max(y1 - y0, 0.8):.1f}" {body}/>')
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


def card_v9_html(spy_state, ph: pd.Series, vix, put: dict = None, spy_ohlcv: pd.DataFrame = None, date_txt: str = '',
                 now=None) -> str:
    """8.4.0 card (layout B + A's detail row) as one HTML block."""
    import ui
    if ph is None or not len(ph):
        return ui.CSS() + '<div class="atn-k">MRS v8 <b>· SPY history not available right now: no stance</b></div>'
    f200 = filter200(spy_ohlcv) if spy_ohlcv is not None else None
    prov = provisional(spy_state, ph, spy_ohlcv, now)
    css = ('<style>.v9c .atn-tl{font-size:11.5px}.v9c .atn-sm{font-size:13px !important;line-height:1.5}'      # 8.10.2: use the width
           '.v9c .atn-ring.sm{width:118px;height:118px}.v9c .atn-ring.sm > div{width:92px;height:92px;font-size:24px}'
           '.v9c .atn-chip{font-size:14px;padding:4px 11px}.v9c .atn-row .pill{font-size:14px;padding:4px 11px;min-width:52px}'
           '.v9c .atn-row .sm{font-size:14.5px;line-height:1.7}.v9c .atn-row .ft{font-size:12.5px}.v9c .atn-row .atn-tile{padding:13px 16px}'
           '@media (max-width:1900px){.v9c .v9g{grid-template-columns:0.8fr 1.35fr !important}}'
           '@media (max-width:1100px){.v9c .v9g{grid-template-columns:1fr !important}}</style>')
    return (css + '<div class="v9c" style="background:#121212;border:1px solid #2a2724;border-radius:6px;padding:14px 16px 10px 16px;margin:2px 0 6px 0">'
            + header_html(spy_state, ph, vix, date_txt, f200, prov)
            + '<div class="v9g" style="display:grid;grid-template-columns:0.56fr 1.44fr;gap:10px;align-items:stretch">'
            + f'<div class="atn-tile"><div class="atn-tl">SPY · last 120 sessions with its phase {ui.tag("Tested")}</div>{ribbon_svg(spy_ohlcv, ph)}</div>'
            + f'<div class="atn-tile"><div class="atn-tl">Clock · width · budget {ui.tag("Tested · selection-adjusted")}</div>{rings_html(ph, vix, f200)}'
            + filter_html(f200) + drop_odds_html(f200, in_dip=int(ph.iloc[-1]) == 0)
            + levels_svg(spy_state, f200=f200) + budget_html(ph, f200) + '</div></div>'
            + detail_html(spy_state, ph, put, spy_ohlcv, f200, now) + '</div>').replace('\n', ' ')
