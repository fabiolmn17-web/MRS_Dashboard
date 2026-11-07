"""
bases.py — breakouts and bases from daily bars (nightly scanner + chart overlay), 3 Nov 2026
=============================================================================================
Rule-based, fixed in advance (no tuning, no machine learning). One function, detect(df), used by the nightly
scanner (auto-adjusted prices) and by the dashboard chart (unadjusted prices, so the drawn pivot is the real price).

Breakouts (today's close against the prior sessions)
  21-day high   close > highest high of the 21 sessions BEFORE today
  52-week high  close > highest high of the 252 sessions BEFORE today

Bases (the current base: it ends yesterday, so today can be the breakout day)
  Flat base         >= 25 sessions (5 weeks) within 15% (highest high to lowest low), after a prior advance of
                    >= 20% (base start vs the lowest low of the 65 sessions before it). Pivot = base high.
  Cup with handle   O'Neil-style:
                    - prior uptrend >= 30% (left lip vs the lowest low of the 130 sessions before it)
                    - cup 35-325 sessions (7-65 weeks) from the left lip to the right lip, 12-33% deep
                    - rounded bottom, not a V: >= 10 sessions and >= 15% of the cup with the close in the bottom third,
                      and the low between 15% and 85% of the way from the left lip to the right lip
                    - right lip within 10% below (at most 3% above) the left lip
                    - handle 5-30 sessions after the right lip, <= 12% deep, its low in the upper half of the cup
                    Pivot = right-lip high (the handle high) + $0.10.
  High tight flag   pole: >= +100% in 20-40 sessions; flag: 15-25 sessions after the pole high, 10-25% pullback,
                    no new high. Pivot = pole high.
  One pattern per stock, in this order of preference: cup with handle, high tight flag, flat base.

State (today's close vs the pivot)
  Broke out today       close > pivot today (the base was found ending yesterday)
  Broke out Nd ago      a base ending N+1 sessions ago was cleared N sessions ago (N = 1-4) and the close is still
                        above that pivot
  Near pivot            close at most 5% below the pivot
  In base               deeper than that
Breakout volume = the breakout day's volume / its prior 50-day average (>= 1.4x = confirmed, O'Neil's 40-50%).
Quality (0-3, a sort aid, not a test): volume dry-up (last 10 sessions' volume < 80% of the 50-day average),
tight closes (last 10 closes within 5%), RS rating >= 80 (added by the scanner).

Descriptive: these are definitions, not tested signals. Follow-through after such breakouts in this universe has not
been measured (planned study); the SPY/QQQ pivot study found broken levels retested 84-89% of the time.
"""
from typing import Optional

import numpy as np
import pandas as pd

# 7 Nov 2026: short side. detect_short() runs the same rules on inverted prices (K / price: highs become lows), so a
# flat base / tight area / VCP / cup with handle in the inverted series is the same shape upside down under a support
# pivot low (bear base). Percent thresholds carry over approximately (a 20% decline = a 25% rise in K / price).
# States: In base, Near breakdown (close at most 5% above the pivot low), Broke down today, Broke down Nd ago.
SHORT_FIELDS = ['sb_type', 'sb_state', 'sb_pivot', 'sb_dist_pct', 'sb_depth', 'sb_weeks', 'sb_days_since']

NEAR_PIVOT = 0.05            # near pivot: close at most 5% below
BO_VOL_OK = 1.4              # breakout volume >= 1.4x the 50-day average
RECENT = 4                   # "broke out N sessions ago", N = 1..4

FIELDS = ['hi21', 'bo21', 'd21_pct', 'hi52w', 'bo52', 'd52_pct', 'vol_ratio',
          'base_type', 'base_state', 'base_days_since_bo', 'base_pivot', 'base_dist_pct', 'base_dist_atr',
          'base_depth', 'base_weeks', 'base_start', 'base_end', 'base_low', 'base_low_date', 'base_left',
          'base_left_date', 'base_right_date', 'handle_start', 'handle_low', 'handle_depth', 'base_prior_adv',
          'base_bo_vol', 'base_quality', 'base_dryup', 'base_tight',
          'piv_hi', 'piv_date', 'piv_bo', 'piv_dist_pct', 'tt_pass', 'tt_score', 'vcp_n', 'vcp_contractions', 'vcp_points']


def _atr(h, l, c, n=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    out = np.full(len(tr), np.nan)
    if len(tr) <= n:
        return out
    out[n] = np.nanmean(tr[1:n + 1])
    for i in range(n + 1, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def _local_peaks(h, lo, hi, w=10):
    """Indexes i in [lo, hi] where h[i] is the highest high within +-w sessions."""
    out = []
    for i in range(max(lo, 0), hi + 1):
        a, b = max(0, i - w), min(len(h), i + w + 1)
        if h[i] >= h[a:b].max():
            out.append(i)
    return out


# ── patterns, each evaluated on bars [0 .. e] (e = the base's last session) ──────────────────────────────────────
PIV_L, PIV_R = 10, 10          # swing pivots as in Fabio's TradingView "Pivot Points High Low" (10 left / 10 right)


def pivot_highs(h, e, L=PIV_L, R=PIV_R):
    """Confirmed pivot highs on bars [0 .. e]: h[i] is the highest high of the L bars before and R bars after
    (strictly above the left side, like ta.pivothigh). Needs R bars after i, so i <= e - R."""
    out = []
    for i in range(L, e - R + 1):
        if h[i] > h[i - L:i].max() and h[i] >= h[i + 1:i + R + 1].max():
            out.append(i)
    return out


def pivot_lows(l, e, L=PIV_L, R=PIV_R):
    out = []
    for i in range(L, e - R + 1):
        if l[i] < l[i - L:i].min() and l[i] <= l[i + 1:i + R + 1].min():
            out.append(i)
    return out


def active_pivot(h, c, e, L=PIV_L, R=PIV_R):
    """The most recent confirmed pivot high that no CLOSE has cleared since (intraday probes allowed): the
    resistance a breakout has to clear. Index or None."""
    for i in reversed(pivot_highs(h, e, L, R)):
        if i < e and c[i + 1:e + 1].max() <= h[i]:
            return i
        if i >= e:
            continue
        return None                                   # the latest pivot was already cleared: no active pivot
    return None


def _pivot_base(h, l, c, e):
    """Flat base / tight area anchored on the active pivot (v2, 4 Nov 2026): the base runs from the pivot high to e
    and never closes above it. Flat base: >= 25 sessions, <= 15% deep. Tight area: 10-24 sessions, <= 8% deep.
    Both after a prior advance >= 20% (pivot vs the lowest low of the 65 sessions before it)."""
    p = active_pivot(h, c, e)
    if p is None or p < 65:
        return None
    n = e - p + 1
    pv = h[p]
    low = l[p:e + 1].min()
    depth = (pv - low) / pv
    adv = pv / l[p - 65:p].min() - 1
    if adv < 0.20:
        return None
    if n >= 25 and depth <= 0.15:
        typ = 'Flat base'
    elif 10 <= n < 25 and depth <= 0.08:
        typ = 'Tight area'
    else:
        return None
    return dict(type=typ, start=p, end=e, pivot=float(pv), low=float(low), low_i=p + int(np.argmin(l[p:e + 1])),
                depth=float(depth), prior_adv=float(adv), left=float(pv), left_i=p)


def _swing_highs(h, a, e, w=5):
    """5/5 swing highs inside [a, e] (confirmed: w bars after them inside [.., e])."""
    return [i for i in range(max(a, w), e - w + 1) if h[i] >= h[i - w:i + w + 1].max()]


def _vcp(h, l, c, v, e, ctx):
    """Volatility contraction pattern (Minervini), rule-based, on bars [0 .. e]:
    - trend template (ctx['tt'] at e): price > 150- and 200-day, 150 > 200, 200 rising over 21 sessions, 50 > 150 and
      200, price > 50-day, >= 30% above the 52-week low, within 25% of the 52-week high, RS rating >= 70 (if known)
    - base: from the highest high of the last 325 sessions (no close above it since), 15-325 sessions long
    - contractions: each swing high (5/5) to the lowest low before the next swing high; >= 2 of them, each <= 85% of
      the previous one, the first 10-35%, the last <= 10%
    - volume dry-up: average volume of the last contraction below the 50-day average (skipped without volume)
    Pivot = the high of the last contraction."""
    if not ctx.get('tt', False):
        return None
    a = max(0, e - 325)
    i0 = a + int(np.argmax(h[a:e - 4]))
    if c[i0 + 1:e + 1].max() > h[i0] or not (15 <= e - i0 <= 325):
        return None
    sh = [i0] + [i for i in _swing_highs(h, i0 + 1, e) if h[i] <= h[i0]]
    # keep swing highs separated by a real pullback: walk forward, one contraction per swing high
    con, k = [], 0
    while k < len(sh):
        i = sh[k]
        nxt = next((j for j in sh[k + 1:] if j > i), None)
        stop = (nxt - 1) if nxt is not None else e
        if stop <= i:
            k += 1
            continue
        lo_i = i + 1 + int(np.argmin(l[i + 1:stop + 1]))
        d = (h[i] - l[lo_i]) / h[i]
        con.append((i, lo_i, d))
        # next swing high after the low
        k = next((m for m in range(k + 1, len(sh)) if sh[m] > lo_i), len(sh))
    con = [x for x in con if x[2] >= 0.01]
    if len(con) < 2:
        return None
    ds = [x[2] for x in con]
    if not (0.10 <= ds[0] <= 0.35) or ds[-1] > 0.10 or any(ds[j] > 0.85 * ds[j - 1] for j in range(1, len(ds))):
        return None
    li, lo_i, _ = con[-1]
    if np.isfinite(v[li:e + 1]).any():
        v50 = np.nanmean(v[max(0, e - 49):e + 1])
        if not np.nanmean(v[li:e + 1]) < v50:
            return None
    return dict(type='VCP', start=i0, end=e, pivot=float(h[li]), low=float(l[i0:e + 1].min()),
                low_i=i0 + int(np.argmin(l[i0:e + 1])), depth=float(ds[0]), prior_adv=None, left=float(h[i0]),
                left_i=i0, vcp=[(int(a_), int(b_), float(d_)) for a_, b_, d_ in con])


def _trend_template(h, l, c, e, rs=None):
    """Minervini's trend template at bar e: (all passed, number passed of 8)."""
    if e < 220:
        return False, 0
    cl = c[:e + 1]
    s50, s150, s200 = cl[-50:].mean(), cl[-150:].mean(), cl[-200:].mean()
    s200_21 = c[e - 220:e - 20].mean()
    lo52, hi52 = l[max(0, e - 251):e + 1].min(), h[max(0, e - 251):e + 1].max()
    px = c[e]
    t = [px > s150 and px > s200, s150 > s200, s200 > s200_21, s50 > s150 and s50 > s200, px > s50,
         px >= 1.3 * lo52, px >= 0.75 * hi52, (rs is None or not np.isfinite(rs) or rs >= 70)]
    return all(t), int(sum(t))


def _cup_handle(h, l, c, e):
    best = None
    for iL in _local_peaks(h, e - 325 - 30, e - 35 - 5):
        if iL < 130:
            continue
        left = h[iL]
        prior_low = l[iL - 130:iL].min()
        adv = left / prior_low - 1
        if adv < 0.30:
            continue
        seg_l = l[iL + 1:e + 1]
        if len(seg_l) < 40:
            continue
        iB = iL + 1 + int(np.argmin(seg_l))
        if h[iL:iB + 1].max() > left:                       # the decline starts at the left lip
            continue
        low = l[iB]
        depth = (left - low) / left
        if not (0.12 <= depth <= 0.33):
            continue
        if iB >= e - 5:
            continue
        iR = iB + 1 + int(np.argmax(h[iB + 1:e + 1]))       # right lip = highest high after the cup low
        cup_len = iR - iL
        if not (35 <= cup_len <= 325):
            continue
        right = h[iR]
        if not (left * 0.90 <= right <= left * 1.03):
            continue
        hlen = e - iR                                       # handle = the sessions after the right lip
        if not (5 <= hlen <= 30):
            continue
        hlow = l[iR + 1:e + 1].min()
        hdepth = (right - hlow) / right
        if hdepth > 0.12 or hlow < low + 0.5 * (left - low):
            continue
        if not (0.15 <= (iB - iL) / cup_len <= 0.85):         # the low sits inside the cup, not at one edge
            continue
        third = low + (left - low) / 3                      # rounded bottom: time spent in the bottom third
        n_bot = int((c[iL:iR + 1] <= third).sum())
        if n_bot < 10 or n_bot < 0.15 * cup_len:
            continue
        cand = dict(type='Cup w/ handle', start=iL, end=e, pivot=float(right + 0.10), low=float(low), low_i=iB,
                    left=float(left), left_i=iL, right_i=iR, handle_i=iR + 1, handle_low=float(hlow),
                    handle_depth=float(hdepth), depth=float(depth), prior_adv=float(adv))
        if best is None or left > best['left']:
            best = cand
    return best


def _high_tight_flag(h, l, c, e):
    for fl in range(15, 26):                                # flag length, sessions after the pole high
        iP = e - fl
        if iP < 41:
            break
        peak = h[iP]
        if h[iP + 1:e + 1].max() > peak:
            continue
        pole_lo = l[iP - 40:iP - 19].min() if iP >= 40 else np.nan     # pole start 20-40 sessions before the peak
        if not np.isfinite(pole_lo) or peak / pole_lo - 1 < 1.0:
            continue
        if h[iP - 40:iP].max() > peak:
            continue
        flo = l[iP + 1:e + 1].min()
        pull = (peak - flo) / peak
        if 0.10 <= pull <= 0.25:
            s = iP - 40 + int(np.argmin(l[iP - 40:iP - 19]))
            return dict(type='High tight flag', start=s, end=e, pivot=float(peak), low=float(flo),
                        low_i=iP + 1 + int(np.argmin(l[iP + 1:e + 1])), depth=float(pull),
                        prior_adv=float(peak / pole_lo - 1), left=float(peak), left_i=iP)
    return None


def _base_at(h, l, c, v, e, rs=None):
    tt, _ = _trend_template(h, l, c, e, rs)
    b = _cup_handle(h, l, c, e) or _vcp(h, l, c, v, e, {'tt': tt}) or _high_tight_flag(h, l, c, e) \
        or _pivot_base(h, l, c, e)
    return b


def detect(df: pd.DataFrame, rs_rating: Optional[float] = None) -> dict:
    """All FIELDS for the latest session of df (DatetimeIndex, columns High / Low / Close, optional Volume)."""
    out = {k: None for k in FIELDS}
    if df is None or len(df) < 30 or not {'High', 'Low', 'Close'} <= set(df.columns):
        return out
    d = df.dropna(subset=['High', 'Low', 'Close'])
    h, l, c = d['High'].to_numpy(float), d['Low'].to_numpy(float), d['Close'].to_numpy(float)
    v = d['Volume'].to_numpy(float) if 'Volume' in d.columns else np.full(len(c), np.nan)
    dates = d.index
    n = len(c)
    last = n - 1
    atr = _atr(h, l, c)
    vol50 = lambda i: np.nanmean(v[max(0, i - 50):i]) if i > 0 else np.nan
    vr = lambda i: float(v[i] / vol50(i)) if np.isfinite(v[i]) and vol50(i) and np.isfinite(vol50(i)) else None
    ds = lambda i: str(dates[i].date()) if i is not None and 0 <= i < n else None

    # breakouts vs the prior 21 / 252 sessions
    if n > 21:
        hi21 = float(h[last - 21:last].max())
        out.update(hi21=hi21, bo21=bool(c[last] > hi21), d21_pct=float(c[last] / hi21 - 1))
    if n > 252:
        hi52 = float(h[last - 252:last].max())
        out.update(hi52w=hi52, bo52=bool(c[last] > hi52), d52_pct=float(c[last] / hi52 - 1))
    out['vol_ratio'] = vr(last)
    # v2: the active 10/10 pivot high (as yesterday) and today's close against it; Minervini trend template today
    if n > PIV_L + PIV_R + 2:
        p_ = active_pivot(h, c, last - 1)
        if p_ is not None:
            out.update(piv_hi=round(float(h[p_]), 2), piv_date=ds(p_), piv_bo=bool(c[last] > h[p_]),
                       piv_dist_pct=float(c[last] / h[p_] - 1))
    tt_, tts_ = _trend_template(h, l, c, last, rs_rating)
    out.update(tt_pass=tt_, tt_score=tts_)

    # the current base (ends yesterday); else a base cleared 1..RECENT sessions ago and still held
    base, bo_i = None, None
    b = _base_at(h, l, c, v, last - 1, rs_rating)
    if b:
        base = b
        if c[last] > b['pivot']:
            bo_i = last
    else:
        for k in range(1, RECENT + 1):
            e = last - 1 - k
            if e < 60:
                break
            b = _base_at(h, l, c, v, e, rs_rating)
            if b and c[e + 1] > b['pivot'] and c[last] > b['pivot']:
                base, bo_i = b, e + 1
                break
    if not base:
        return out
    pv = base['pivot']
    dist = c[last] / pv - 1
    if bo_i is not None:
        since = last - bo_i
        state = 'Broke out today' if since == 0 else f'Broke out {since}d ago'
    else:
        since = None
        state = 'Near pivot' if dist >= -NEAR_PIVOT else 'In base'
    e = base['end']
    w10 = slice(max(0, e - 9), e + 1)
    dry = bool(np.isfinite(vol50(e + 1)) and np.nanmean(v[w10]) < 0.8 * vol50(e + 1)) if np.isfinite(v[w10]).any() else None
    tight = bool((c[w10].max() - c[w10].min()) / c[w10].max() <= 0.05)
    q = int(bool(dry)) + int(tight) + int(rs_rating is not None and np.isfinite(rs_rating) and rs_rating >= 80)
    out.update(
        base_type=base['type'], base_state=state, base_days_since_bo=since, base_pivot=round(pv, 2),
        base_dist_pct=float(dist), base_dist_atr=float((c[last] - pv) / atr[last]) if np.isfinite(atr[last]) else None,
        base_depth=base['depth'], base_weeks=round((e - base['start'] + 1) / 5, 1),
        base_start=ds(base['start']), base_end=ds(e), base_low=base['low'], base_low_date=ds(base.get('low_i')),
        base_left=base.get('left'), base_left_date=ds(base.get('left_i')), base_right_date=ds(base.get('right_i')),
        handle_start=ds(base.get('handle_i')), handle_low=base.get('handle_low'), handle_depth=base.get('handle_depth'),
        base_prior_adv=base.get('prior_adv'), base_bo_vol=vr(bo_i) if bo_i is not None else None,
        base_quality=q, base_dryup=dry, base_tight=tight,
        vcp_n=len(base['vcp']) if base.get('vcp') else None,
        vcp_contractions=' · '.join(f"{x[2] * 100:.0f}%" for x in base['vcp']) if base.get('vcp') else None,
        vcp_points=';'.join(f"{ds(a_)},{h[a_]:.2f},{ds(b_)},{l[b_]:.2f}" for a_, b_, _ in base['vcp'])
        if base.get('vcp') else None)
    return out


def detect_short(df: pd.DataFrame, rs_rating: Optional[float] = None) -> dict:
    """Bear bases (7 Nov 2026): the long rules on inverted prices. Returns SHORT_FIELDS; pivot and distance in real
    prices (sb_dist_pct > 0 = close above the pivot low). High tight flags are not mirrored."""
    out = {k: None for k in SHORT_FIELDS}
    if df is None or len(df) < 30 or not {'High', 'Low', 'Close'} <= set(df.columns):
        return out
    d = df.dropna(subset=['High', 'Low', 'Close'])
    d = d[(d['Low'] > 0) & (d['Close'] > 0) & (d['High'] > 0)]
    if len(d) < 30:
        return out
    K = float(d['Close'].median()) ** 2                 # keeps the inverted prices near the real ones
    inv = pd.DataFrame({'High': K / d['Low'], 'Low': K / d['High'], 'Close': K / d['Close']}, index=d.index)
    if 'Volume' in d.columns:
        inv['Volume'] = d['Volume']
    r = detect(inv, None if rs_rating is None or pd.isna(rs_rating) else 100 - float(rs_rating))
    t = r.get('base_type')
    if not t or t == 'High tight flag':
        return out
    st_ = str(r.get('base_state') or '')
    st_ = (st_.replace('Broke out', 'Broke down').replace('Near pivot', 'Near breakdown'))
    piv = r.get('base_pivot')
    real_piv = K / float(piv) if piv else None
    close = float(d['Close'].iloc[-1])
    out.update(sb_type=t, sb_state=st_, sb_pivot=round(real_piv, 2) if real_piv else None,
               sb_dist_pct=(close / real_piv - 1) if real_piv else None, sb_depth=r.get('base_depth'),
               sb_weeks=r.get('base_weeks'), sb_days_since=r.get('base_days_since_bo'))
    return out
