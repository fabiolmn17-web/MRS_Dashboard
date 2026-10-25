"""
rs_labels.py — RS label with a 3-day hold, days in label, RS 1M tag, buy-leaders helpers
=========================================================================================
Shared by app.py (sector table) and StockScanner/industry.py (industry table).

Score (unchanged, tested 25 Oct 2026 in studies/momentum_rs_persistence):
    RS(n)  = p_t / p_(t-n+1)  -  spy_t / spy_(t-n+1)        n = 252 / 126 / 63 / 21 sessions
    Score  = 0.5 * (sign RS 1Y + sign RS 6M + sign RS 3M)   -> -1.5 .. +1.5
    RS 1M is NOT scored: adding it raised label changes to ~49 a year with no gain in
    forward information (t_label_variants.csv). It is shown as an early-warning tag.

Hold (05_label_variants.py, "V0 + 3d hold"):
    The sign state (1Y, 6M, 3M) only changes after the new raw state has held for
    HOLD_DAYS consecutive sessions. Label changes fall from ~28-30 to ~10 a year and
    1-3 day spells from ~60% to ~9%, with the same rank correlation to forward RS.
    Label and 6M->3M trend both come from the held state, so they never disagree.

Days in label: sessions since the held label last changed (">= N" when the price
history is shorter than the spell).
"""
from typing import Optional

import numpy as np
import pandas as pd

HOLD_DAYS = 3
PERIODS = {'RS 1Y': 252, 'RS 6M': 126, 'RS 3M': 63, 'RS 1M': 21}


def composite_label(score: float) -> str:
    if score >= 1.5:  return 'STRUCTURAL LEADER'
    if score >= 1.0:  return 'STRONG OUTPERFORMER'
    if score >= 0.5:  return 'OUTPERFORMING'
    if score <= -1.5: return 'STRUCTURAL LAGGARD'
    if score <= -1.0: return 'STRONG UNDERPERFORMER'
    if score <= -0.5: return 'UNDERPERFORMING'
    return 'NEUTRAL'


def trend_tag(s6: int, s3: int) -> str:
    if s3 > s6:  return 'IMPROVING'
    if s3 < s6:  return 'FADING'
    if s3 > 0:   return 'STABLE+'
    if s3 < 0:   return 'STABLE-'
    return 'NEUTRAL'


def _sgn(v) -> int:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    return 1 if v > 0 else -1 if v < 0 else 0


def rs_history(s: pd.Series, spy: pd.Series) -> pd.DataFrame:
    """Daily RS 1Y/6M/3M/1M on the common trading dates of s and SPY."""
    df = pd.concat([s.rename('p'), spy.rename('spy')], axis=1).dropna()
    out = pd.DataFrame(index=df.index)
    for k, n in PERIODS.items():
        out[k] = df['p'] / df['p'].shift(n - 1) - df['spy'] / df['spy'].shift(n - 1)
    return out


def _hold(states: list, n: int):
    """Held state per day: a new raw state replaces the held one after n consecutive days.

    Returns (held list, pending_state, pending_days) for the last day.
    """
    held, cur, cand, k = [], None, None, 0
    for x in states:
        if cur is None:
            cur = x
        elif x == cur:
            cand, k = None, 0
        else:
            if x == cand:
                k += 1
            else:
                cand, k = x, 1
            if k >= n:
                cur, cand, k = x, None, 0
        held.append(cur)
    return held, cand, k


def label_info(s: pd.Series, spy: pd.Series, hold: int = HOLD_DAYS) -> Optional[dict]:
    """RS values for the last day plus the held label, trend, days in label and 1M tag.

    Needs at least 63 common sessions. With fewer than 252 + a few sessions the
    hold cannot run; the raw state is used (as before) and the history flag says so.
    """
    rh = rs_history(s.dropna(), spy.dropna())
    if len(rh) < 63:
        return None
    last = rh.iloc[-1]
    rs_now = {k: (float(last[k]) if pd.notna(last[k]) else np.nan) for k in PERIODS}

    full = rh[['RS 1Y', 'RS 6M', 'RS 3M']].dropna()
    raw_state = (_sgn(rs_now['RS 1Y']), _sgn(rs_now['RS 6M']), _sgn(rs_now['RS 3M']))
    raw_score = 0.5 * sum(raw_state)

    if len(full) >= hold + 1 and full.index[-1] == rh.index[-1]:
        states = [tuple(int(np.sign(v)) for v in row) for row in full.to_numpy()]
        held, cand, k = _hold(states, hold)
        state = held[-1]
        scores = [0.5 * sum(h) for h in held]
        run = 1
        for v in reversed(scores[:-1]):
            if v == scores[-1]:
                run += 1
            else:
                break
        capped = run == len(scores)
        pend_state = cand if (cand is not None and 0.5 * sum(cand) != scores[-1]) else None
        hist_days = len(scores)
    else:
        state, run, capped, pend_state, k, hist_days = raw_state, np.nan, True, None, 0, 0

    score = 0.5 * sum(state)
    label = composite_label(score)
    trend = trend_tag(state[1], state[2])

    # RS 1M early warning (shown, not scored)
    tag_1m = ''
    r1 = rs_now['RS 1M']
    if not np.isnan(r1):
        if score >= 1.5 and r1 < 0:
            tag_1m = '1M ↓'
        elif 0.5 <= score < 1.5 and r1 > 0:
            tag_1m = '1M ↑'

    return {
        **rs_now,
        'Score': score, 'Label': label, 'Trend': trend,
        'RawScore': raw_score, 'RawLabel': composite_label(raw_score),
        'DaysInLabel': run, 'DaysCapped': bool(capped), 'LabelHistory': hist_days,
        'PendingLabel': composite_label(0.5 * sum(pend_state)) if pend_state is not None else '',
        'PendingDays': int(k) if pend_state is not None else 0,
        'Tag1M': tag_1m,
    }


# ── buy-leaders helpers ─────────────────────────────────────────────────────────
def momentum_light(spy: pd.Series, band: str) -> dict:
    """Momentum ON when SPY is above its close 252 sessions ago and the MRS band is not RISK-OFF.

    Otherwise CAUTION: momentum-crash state (Daniel-Moskowitz). Tested in
    momentum_rs_persistence: leader-minus-laggard industry spread +0.8 vs -1.9 pp at
    1 month and +1.8 vs -4.6 pp at 3 months, SPY 12m up vs down (mostly 2008-09).
    """
    spy = spy.dropna()
    r12 = float(spy.iloc[-1] / spy.iloc[-252] - 1) if len(spy) >= 252 else np.nan
    up = (not np.isnan(r12)) and r12 > 0
    off = band == 'RISK-OFF'
    on = up and not off
    if on:
        why = 'SPY above its level 252 sessions ago and MRS not RISK-OFF'
    else:
        parts = []
        if not up:
            parts.append('SPY below its level 252 sessions ago')
        if off:
            parts.append('MRS in RISK-OFF')
        why = ' and '.join(parts) if parts else 'SPY 12-month return unavailable'
    return {'on': on, 'spy_12m': r12, 'risk_off': off, 'why': why}


def leaders_shortlist(ind: pd.DataFrame) -> pd.DataFrame:
    """Industry ETFs to look at first for longs (sectors are context only).

    A  structural leader (held score +1.5) whose RS 1M is not negative       -> Tested (industry momentum, small)
    B  Outperforming (held +0.5) and IMPROVING (1Y+, 6M-, 3M+)               -> Tested (beat SPY 55% vs 48% at 1 mo)
    Laggards are never listed. Sorted by RS 3M.
    """
    if ind is None or len(ind) == 0 or 'Score' not in ind.columns:
        return pd.DataFrame()
    d = ind.copy()
    d['Score'] = pd.to_numeric(d['Score'], errors='coerce')
    r1 = pd.to_numeric(d.get('RS 1M'), errors='coerce')
    a = (d['Score'] >= 1.5) & ~(r1 < 0)
    b = (d['Score'] == 0.5) & (d['Trend'] == 'IMPROVING')
    out = d[a | b].copy()
    out['Why'] = np.where(a[a | b], 'Structural leader, 1M not turning down', 'Outperforming + improving')
    return out.sort_values('RS 3M', ascending=False).reset_index(drop=True)
