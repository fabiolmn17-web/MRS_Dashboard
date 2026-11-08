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

v3, 8 Nov 2026: IBD context and quality layer (O'Neil, "How To Find & Own America's Greatest Opportunities", 70
annotated weekly charts of hand-picked winners, used as a MODEL BOOK; see MRS_Dashboard/docs/bases_v3_ibd_layer.md).
All of it is Descriptive and untested: the features below describe the IBD template; whether they separate outcomes
in this universe is the planned Base Quality v1 study.
  Market (follow_through(): SPY and QQQ)  correction = close >= 8% below the 52-week high; rally attempt day 1 = first
      up close after a new correction low; follow-through day (FTD) = day >= 4 of the attempt, index >= +1.25% on
      volume above the prior day (the 1.25% and day-4 are IBD's published convention, ASSUMPTIONS: not in the PDF);
      the attempt / uptrend fails if the correction low is undercut.
  New patterns  Double bottom (second low undercuts the first by 0-8%, pivot = middle high + $0.10, >= 7 weeks,
      15-35% deep, up to 50% when SPY fell >= 20% in the window, not three bottoms; "Shakeout+3" tag when the close
      clears 10% above the shakeout low), Base-on-base (flat base / tight area holding >= 97% of the prior base's
      pivot, prior breakout <= 63 sessions before), Ascending base (3 pullbacks of 10-20%, higher lows and highs,
      9-17 weeks), IPO base (25-250 sessions of history, >= 15 sessions below the post-listing high, <= 35% deep).
      Cups may be up to 50% deep when SPY fell >= 20% in the window (was 33% fixed).
  Context (needs mkt = SPY daily OHLCV)  mkt_dd_in_base (SPY max drawdown inside the base), depth_vs_mkt, base_in_corr
      (>= 8%), rs_holds (SPY makes a new low in the last third of the base, the stock stays >= 3% above its earlier
      low), rs_line_hi (RS line = close / SPY at a 52-week high in the 5 sessions to the breakout or today), rs_leads
      (RS line at a 52-week high inside the base while price was still under the pivot), ftd_age, base_bo_ftd
      (breakout within 20 sessions after an FTD).
  Character (weekly bars, W-FRI)  tight_weeks (longest run of weekly closes within 1.5% of the prior week, last 8
      weeks of the base, counted in closes: 3 = "3 weeks tight"), support_wks (volume >= 1.4x the prior 10 weeks,
      close in the upper 60% of the range, low at or below the prior week's low + 1%), distrib_flag (2 of the 3
      biggest-volume weeks are down weeks: all 3 of them,
      and down-week volume > up-week volume; the first week is excluded), ud_vol_50 (up-day / down-day volume, 50 sessions), handle_dryup (handle,
      or the last third of the base: volume < 0.8x the 50 sessions before it), base_bo_light (breakout < 1.4x volume).
  Faults  wedging handle, V / straight up from the bottom, wide and loose, left side > 13 weeks, stage >= 3,
      light-volume breakout, distribution, deeper than 2.5x the market.
  Quality  base_q of base_q_max checks (7 with market data): formed in a correction, RS line at a 52-week high,
      held up while SPY fell (rs_holds or rs_leads), 3+ weeks tight, volume dry-up, accumulation (ud_vol_50 >= 1.2 or
      a support week), depth in line with the market (<= 2.5x SPY's drawdown; <= 20% when SPY fell < 5%).
      base_q_detail is the tooltip text. The old base_quality (0-3) is kept for older views.
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
          'piv_hi', 'piv_date', 'piv_bo', 'piv_dist_pct', 'tt_pass', 'tt_score', 'vcp_n', 'vcp_contractions', 'vcp_points',
          # v3 (8 Nov 2026): IBD context and quality layer
          'mkt_dd_in_base', 'depth_vs_mkt', 'base_in_corr', 'rs_holds', 'rs_line_hi', 'rs_leads', 'ftd_age',
          'base_bo_ftd', 'tight_weeks', 'support_wks', 'distrib_flag', 'ud_vol_50', 'handle_dryup', 'base_bo_light',
          'base_stage', 'base_faults', 'base_q', 'base_q_max', 'base_q_detail',
          'db_l1', 'db_l1_date', 'db_mid', 'db_mid_date', 'db_sk3', 'db_sk3_level']
# market columns written once per scan (constant across rows): follow_through() on SPY and QQQ
MKT_FIELDS = ['mkt_state', 'ftd_date', 'ftd_age', 'mkt_dd', 'qqq_state', 'qqq_ftd_date', 'qqq_ftd_age']

# v3 thresholds (IBD model book; see the docstring)
FTD_CORR = 0.08          # correction: close >= 8% below the 52-week high
FTD_MIN_DAY = 4          # ASSUMPTION (IBD convention, not in the PDF): FTD on day 4 or later of the rally attempt
FTD_MIN_GAIN = 0.0125    # ASSUMPTION (IBD convention, not in the PDF): index up >= 1.25% on higher volume
FTD_STALE = 126          # a confirmed uptrend older than ~6 months with no new correction reads "Uptrend"
FTD_NEAR = 20            # breakout within 20 sessions after an FTD
BEAR_DD = 0.20           # SPY drawdown in the window that allows deeper cups / double bottoms (up to 50%)
TIGHT_WK = 0.015         # weekly close within 1.5% of the prior week
DEEP_VS_MKT = 2.5        # base deeper than 2.5x the market's drawdown = fault


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


def _cup_handle(h, l, c, e, mdd=None):
    """mdd(a, b): SPY max drawdown over bars a..b (v3) or None. Depth up to 50% when SPY fell >= 20% in the cup."""
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
        if not (0.12 <= depth <= 0.50):
            continue
        if depth > 0.33 and (mdd is None or not mdd(iL, e) >= BEAR_DD):
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


# ── v3 (8 Nov 2026): market follow-through day, new IBD patterns, context and quality ────────────────────────────
def follow_through(m: Optional[pd.DataFrame]) -> dict:
    """IBD-style market state on an index (SPY / QQQ daily OHLCV). Returns state (text), ftd_date, ftd_age (sessions
    since the latest FTD), dd (close vs the 52-week high), corr_low, attempt_day and ftd_dates (all FTDs, for
    'breakout within 20 sessions of an FTD'). The 1.25% / day-4 rule is an assumption (see FTD_MIN_*)."""
    out = dict(state=None, ftd_date=None, ftd_age=None, dd=None, corr_low=None, attempt_day=None, ftd_dates=[])
    if m is None or len(m) < 30 or not {'High', 'Low', 'Close'} <= set(m.columns):
        return out
    d = m.dropna(subset=['High', 'Low', 'Close'])
    h, l, c = d['High'].to_numpy(float), d['Low'].to_numpy(float), d['Close'].to_numpy(float)
    v = d['Volume'].to_numpy(float) if 'Volume' in d.columns else np.full(len(c), np.nan)
    hi = pd.Series(h).rolling(252, min_periods=1).max().to_numpy()
    n = len(c)
    state, corr_low, day1, ftd_i, ftd_low, peak = 'Uptrend', np.nan, None, None, np.nan, np.nan
    ftds = []
    for t in range(1, n):
        dd = 1 - c[t] / hi[t]
        if state == 'Uptrend':
            if dd >= FTD_CORR:
                state, corr_low, day1 = 'Correction', l[t], None
            continue
        if state == 'Confirmed':                         # after an FTD: fails on an undercut of the rally low, or a
            peak = max(peak, h[t])                       # fresh 8% decline from the high made since the FTD
            if l[t] < ftd_low or c[t] <= (1 - FTD_CORR) * peak:
                state, corr_low, day1 = 'Correction', min(l[t], ftd_low) if l[t] < ftd_low else l[t], None
            continue
        if l[t] < corr_low:                              # new correction low: any rally attempt fails
            corr_low = l[t]
            day1 = t if c[t] > c[t - 1] else None        # a new-low day that closes up can be day 1
            continue
        if day1 is None:
            if c[t] > c[t - 1]:
                day1 = t
            continue
        k = t - day1 + 1
        if k >= FTD_MIN_DAY and c[t] / c[t - 1] - 1 >= FTD_MIN_GAIN and (not np.isfinite(v[t]) or not np.isfinite(v[t - 1])
                                                                       or v[t] > v[t - 1]):
            state, ftd_i, ftd_low, peak = 'Confirmed', t, corr_low, h[t]
            ftds.append(d.index[t])
    last = n - 1
    dd = 1 - c[last] / hi[last]
    age = None if ftd_i is None else last - ftd_i
    if state == 'Confirmed':
        txt = (f'Confirmed uptrend (FTD {d.index[ftd_i]:%d %b})' if age <= FTD_STALE or dd >= FTD_CORR / 2
               else 'Uptrend')
    elif state == 'Uptrend':
        txt = 'Uptrend'
    elif day1 is not None:
        txt = f'Rally attempt (day {last - day1 + 1})'
    else:
        txt = 'Correction'
    out.update(state=txt, ftd_date=None if ftd_i is None else str(d.index[ftd_i].date()), ftd_age=age, dd=float(dd),
               corr_low=None if not np.isfinite(corr_low) else float(corr_low),
               attempt_day=None if state not in ('Correction',) or day1 is None else last - day1 + 1, ftd_dates=ftds)
    return out


def market_fields(spy: Optional[pd.DataFrame], qqq: Optional[pd.DataFrame] = None) -> dict:
    """MKT_FIELDS for the scan (constant columns) from SPY and QQQ."""
    s, q = follow_through(spy), follow_through(qqq)
    return dict(mkt_state=s['state'], ftd_date=s['ftd_date'], ftd_age=s['ftd_age'], mkt_dd=s['dd'],
                qqq_state=q['state'], qqq_ftd_date=q['ftd_date'], qqq_ftd_age=q['ftd_age'])


def _mdd_fn(mh, ml):
    """mdd(a, b): the market's max drawdown over bars a..b (from the running high inside the window)."""
    if mh is None:
        return None

    def mdd(a, b):
        a, b = max(0, int(a)), int(b)
        if b <= a:
            return 0.0
        hh, ll = mh[a:b + 1], ml[a:b + 1]
        ok = np.isfinite(hh) & np.isfinite(ll)
        if ok.sum() < 2:
            return np.nan
        pk = np.maximum.accumulate(np.where(ok, hh, -np.inf))
        return float(np.nanmax(np.where(ok, (pk - ll) / pk, np.nan)))
    return mdd


def _double_bottom(h, l, c, e, mdd=None):
    """W: left high -> first low L1 -> middle high M -> second low L2 that undercuts L1 by 0-8% (shakeout).
    >= 35 sessions from the left high, 15-35% deep (up to 50% in a >= 20% market decline), prior advance >= 30%,
    no third bottom near L1 before it, M not yet cleared by a close. Pivot = M high + $0.10."""
    lows = pivot_lows(l, e, 10, 5)
    for i1 in reversed(lows):
        if i1 < e - 325 or i1 > e - 15:
            continue
        seg = l[i1 + 1:e + 1]
        i2 = i1 + 1 + int(np.argmin(seg))
        if not (l[i2] < l[i1] and l[i2] >= 0.92 * l[i1]) or i2 - i1 < 10 or e - i2 < 3:
            continue
        iM = i1 + 1 + int(np.argmax(h[i1 + 1:i2]))
        if (h[iM] - l[i1]) / l[i1] < 0.08:
            continue
        a = max(0, i1 - 130)
        if i1 - a < 20:
            continue
        iLf = a + int(np.argmax(h[a:i1]))
        left = h[iLf]
        if h[iM] > left or e - iLf + 1 < 35 or iLf < 30 or i1 - iLf < 10:
            continue
        depth = (left - l[i2]) / left
        lim = 0.50 if (mdd is not None and mdd(iLf, e) >= BEAR_DD) else 0.35
        if not (0.15 <= depth <= lim):
            continue
        adv = left / l[max(0, iLf - 130):iLf].min() - 1
        if adv < 0.30:
            continue
        if c[iM + 1:e + 1].max() > h[iM]:
            continue
        if any((iLf < j < i1 or i1 + 2 < j < i2 - 2) and l[j] <= 1.03 * l[i1] for j in lows):   # three bottoms: faulty
            continue
        return dict(type='Double bottom', start=iLf, end=e, pivot=float(h[iM] + 0.10), low=float(l[i2]), low_i=i2,
                    left=float(left), left_i=iLf, right_i=iM, depth=float(depth), prior_adv=float(adv),
                    db_l1=float(l[i1]), db_l1_i=i1, db_mid=float(h[iM]), db_mid_i=iM)
    return None


def _ascending_base(h, l, c, e):
    """Three pullbacks of 10-20%, each from a higher high to a higher low, spanning 45-85 sessions (9-17 weeks);
    the last high not cleared by a close; prior advance >= 20%. Pivot = the last (highest) swing high."""
    a = max(0, e - 85)
    sh = _swing_highs(h, a, e)
    if len(sh) < 3:
        return None
    i3 = max(sh, key=lambda i: h[i])
    if c[i3 + 1:e + 1].max() > h[i3] if i3 < e else False:
        return None
    seq = [i3]
    for i in reversed([j for j in sh if j < i3]):
        if h[i] < h[seq[-1]]:
            seq.append(i)
        if len(seq) == 3:
            break
    if len(seq) < 3:
        return None
    seq = seq[::-1]
    lows, pulls = [], []
    for k, i in enumerate(seq):
        j = seq[k + 1] if k + 1 < len(seq) else e + 1
        if j - i < 2:
            return None
        lo_i = i + 1 + int(np.argmin(l[i + 1:j]))
        lows.append(l[lo_i]); pulls.append((h[i] - l[lo_i]) / h[i])
    if not all(0.10 <= p <= 0.20 for p in pulls) or not (lows[0] < lows[1] < lows[2]):
        return None
    if not (45 <= e - seq[0] + 1 <= 85) or seq[0] < 65:
        return None
    adv = h[seq[0]] / l[seq[0] - 65:seq[0]].min() - 1
    if adv < 0.20:
        return None
    lo = float(min(lows))
    return dict(type='Ascending base', start=seq[0], end=e, pivot=float(h[i3]), low=lo,
                low_i=seq[0] + int(np.argmin(l[seq[0]:e + 1])), depth=float(max(pulls)), prior_adv=float(adv),
                left=float(h[seq[0]]), left_i=seq[0])


def _ipo_base(h, l, c, e):
    """Recent listing (25-250 sessions of history): the first base below the post-listing high, >= 15 sessions,
    8-35% deep. Pivot = the post-listing high."""
    if not (25 <= e + 1 <= 250):
        return None
    iH = int(np.argmax(h[:e + 1]))
    n = e - iH + 1
    if n < 15:
        return None
    low = l[iH:e + 1].min()
    depth = (h[iH] - low) / h[iH]
    if not (0.08 <= depth <= 0.35):
        return None
    return dict(type='IPO base', start=iH, end=e, pivot=float(h[iH]), low=float(low),
                low_i=iH + int(np.argmin(l[iH:e + 1])), depth=float(depth), prior_adv=None, left=float(h[iH]), left_i=iH)


def _base_on_base(h, l, c, e, b, ph):
    """A flat base / tight area b whose low holds >= 97% of the prior base's pivot, where that prior base (a 10/10
    pivot high that stood >= 25 sessions) was cleared by a close <= 63 sessions before this base started."""
    p = b['start']
    for p0 in reversed([i for i in ph if i < p]):
        if h[p0] >= h[p]:
            continue
        above = np.nonzero(c[p0 + 1:p + 1] > h[p0])[0]
        if not len(above):
            continue
        j = p0 + 1 + int(above[0])
        if j - p0 >= 25 and p - j <= 63 and b['low'] >= 0.97 * h[p0]:
            out = dict(b)
            out.update(type='Base-on-base', prior_pivot=float(h[p0]))
            return out
        break
    return None


def _stage(h, l, c, start, ph, sma200, mkt_reset=None):
    """IBD base stage: 1 + base breakouts (a 10/10 pivot high that stood >= 25 sessions and >= 5% deep, cleared by a
    close; one per 25 sessions) since the last close below the 200-day or the last >= 10% SPY drawdown."""
    below = np.nonzero(np.isfinite(sma200[:start + 1]) & (c[:start + 1] < sma200[:start + 1]))[0]
    reset = int(below[-1]) if len(below) else 0
    if mkt_reset is not None:
        reset = max(reset, mkt_reset)
    js = []
    for i in ph:
        if i <= reset or i >= start:
            continue
        above = np.nonzero(c[i + 1:start] > h[i])[0]
        if not len(above):
            continue
        j = i + 1 + int(above[0])
        if j - i >= 25 and (h[i] - l[i:j].min()) / h[i] >= 0.05:
            js.append(j)
    cnt, lastj = 0, -10 ** 9
    for j in sorted(set(js)):
        if j - lastj >= 25:
            cnt += 1; lastj = j
    return 1 + cnt


def _naive(idx):
    """tz-naive, normalized DatetimeIndex (scanner downloads are naive, chart histories may carry a timezone)."""
    i = pd.DatetimeIndex(idx)
    if i.tz is not None:
        i = i.tz_localize(None)
    return i.normalize()


def _weekly(d):
    """Weekly bars (W-FRI) and the week code of every daily bar."""
    wk = _naive(d.index).to_period('W-FRI')
    codes, uniq = pd.factorize(wk, sort=True)
    g = pd.DataFrame({'h': d['High'].to_numpy(float), 'l': d['Low'].to_numpy(float), 'c': d['Close'].to_numpy(float),
                      'v': d['Volume'].to_numpy(float) if 'Volume' in d.columns else np.nan, 'k': codes})
    w = g.groupby('k').agg(h=('h', 'max'), l=('l', 'min'), c=('c', 'last'), v=('v', 'sum'))
    return w, codes


def _longest_true(x):
    best = run = 0
    for b in x:
        run = run + 1 if b else 0
        best = max(best, run)
    return best


def _context_quality(d, h, l, c, v, base, e, bo_i, last, ctx):
    """v3 context, character, faults and quality for the found base (fields named as in FIELDS)."""
    out = {}
    s0, typ, pv = base['start'], base['type'], base['pivot']
    mh, ml, mc, mdd = ctx.get('mh'), ctx.get('ml'), ctx.get('mc'), ctx.get('mdd')
    has_m = mh is not None
    depth = base.get('depth')
    # ── context ──
    if has_m:
        md = mdd(s0, e)
        out['mkt_dd_in_base'] = md
        out['base_in_corr'] = bool(np.isfinite(md) and md >= FTD_CORR)
        out['depth_vs_mkt'] = float(depth / md) if depth is not None and np.isfinite(md) and md >= 0.02 else None
        n = e - s0 + 1
        if n >= 9:
            k = s0 + (2 * n) // 3
            if np.nanmin(ml[k:e + 1]) < np.nanmin(ml[s0:k]):
                out['rs_holds'] = bool(l[k:e + 1].min() >= 1.03 * l[s0:k].min())
        rsl, rmax = ctx['rsl'], ctx['rmax']
        ref = bo_i if bo_i is not None else last
        w = slice(max(0, ref - 4), ref + 1)
        hi_ok = np.isfinite(rmax[w]) & (rsl[w] >= rmax[w])
        out['rs_line_hi'] = bool(hi_ok.any())
        stop = (bo_i - 1) if bo_i is not None else last
        seg = slice(s0, stop + 1)
        out['rs_leads'] = bool((np.isfinite(rmax[seg]) & (rsl[seg] >= rmax[seg]) & (c[seg] <= pv)).any())
        f = ctx.get('ftd') or {}
        out['ftd_age'] = f.get('ftd_age')
        if bo_i is not None and f.get('ftd_dates'):
            pos = _naive(d.index).searchsorted(_naive(f['ftd_dates']))
            out['base_bo_ftd'] = bool(any(0 <= bo_i - p_ <= FTD_NEAR for p_ in pos if p_ < len(d)))
    # ── character (weekly) ──
    wkly, codes = ctx['weekly']
    Wh, Wl, Wc, Wv = (wkly[k].to_numpy(float) for k in ('h', 'l', 'c', 'v'))
    ws, we = int(codes[s0]), int(codes[e])
    a8 = max(ws, we - 7)
    if we - a8 >= 1:
        ch = np.abs(Wc[a8 + 1:we + 1] / Wc[a8:we] - 1) <= TIGHT_WK
        r = _longest_true(ch)
        out['tight_weeks'] = int(r + 1) if r else 0
    else:
        out['tight_weeks'] = 0
    has_v = np.isfinite(v).any()
    if has_v:
        sup = 0
        for wi in range(max(ws, 10), we + 1):
            pv10 = np.nanmean(Wv[wi - 10:wi])
            rng = Wh[wi] - Wl[wi]
            if pv10 > 0 and Wv[wi] >= 1.4 * pv10 and rng > 0 and Wc[wi] >= Wl[wi] + 0.4 * rng and Wl[wi] <= 1.01 * Wl[wi - 1]:
                sup += 1
        out['support_wks'] = sup
        bw = list(range(max(ws + 1, 1), we + 1))           # weeks after the first one (the break from the high)
        if len(bw) >= 5:
            top3 = sorted(bw, key=lambda wi: -Wv[wi])[:3]
            dn3 = [Wc[wi] < Wc[wi - 1] for wi in top3]
            upv = sum(Wv[wi] for wi in bw if Wc[wi] > Wc[wi - 1]); dnv = sum(Wv[wi] for wi in bw if Wc[wi] < Wc[wi - 1])
            # the biggest-volume weeks are down weeks: all 3, or the biggest one + 2 of 3 with down volume > up volume
            out['distrib_flag'] = bool(all(dn3) and dnv > upv)
        a50 = max(1, e - 49)
        dc = np.diff(c[a50 - 1:e + 1])
        vv = v[a50:e + 1]
        up, dn = np.nansum(vv[dc > 0]), np.nansum(vv[dc < 0])
        out['ud_vol_50'] = float(up / dn) if dn > 0 else None
        hs = base.get('handle_i') if typ == 'Cup w/ handle' else s0 + (2 * (e - s0 + 1)) // 3
        if hs is not None and hs - 50 >= 0 and e >= hs:
            ref50 = np.nanmean(v[hs - 50:hs])
            out['handle_dryup'] = bool(ref50 > 0 and np.nanmean(v[hs:e + 1]) < 0.8 * ref50)
        if bo_i is not None:
            v50 = np.nanmean(v[max(0, bo_i - 50):bo_i])
            out['base_bo_light'] = bool(np.isfinite(v[bo_i]) and v50 > 0 and v[bo_i] / v50 < BO_VOL_OK)
    # ── faults ──
    faults = []
    if typ == 'Cup w/ handle' and base.get('handle_i') is not None:
        hh_ = l[base['handle_i']:e + 1]
        if len(hh_) >= 5:
            m_ = len(hh_) // 2
            slope = np.polyfit(np.arange(len(hh_)), hh_, 1)[0]
            if hh_[m_:].min() > hh_[:m_].min() and slope > 0:
                faults.append('wedging handle')
    if typ in ('Cup w/ handle', 'Double bottom'):
        lo_i, left = base['low_i'], base['left']
        hit = np.nonzero(h[lo_i + 1:e + 1] >= 0.95 * left)[0]
        if len(hit) and hit[0] + 1 <= 15:
            t_ = lo_i + 1 + int(hit[0])
            wk_ = range(int(codes[lo_i]) + 1, int(codes[t_]) + 1)
            if sum(Wc[wi] < Wc[wi - 1] for wi in wk_) < 2:
                faults.append('V: straight up from the low')
        if ((base.get('db_l1_i') or base['low_i']) - base['left_i']) / 5 > 13:     # double bottom: left high -> L1
            faults.append('left side > 13 wks')
    half = ws + (we - ws) // 2
    if we - half >= 2 and ws - 13 >= 0:
        rb = np.nanmedian((Wh[half:we + 1] - Wl[half:we + 1]) / Wc[half:we + 1])
        rp = np.nanmedian((Wh[ws - 13:ws] - Wl[ws - 13:ws]) / Wc[ws - 13:ws])
        if rp > 0 and rb >= 1.5 * rp:
            faults.append('wide and loose')
    stg = _stage(h, l, c, s0, ctx['ph'], ctx['sma200'], ctx.get('mreset'))
    out['base_stage'] = stg
    if stg >= 3:
        faults.append(f'late stage ({stg})')
    if out.get('base_bo_light'):
        faults.append('light-volume breakout')
    if out.get('distrib_flag'):
        faults.append('distribution')
    dvm = out.get('depth_vs_mkt')
    if dvm is not None and dvm > DEEP_VS_MKT and (out.get('mkt_dd_in_base') or 0) >= 0.05:
        faults.append(f'{dvm:.1f}x deeper than the market')
    out['base_faults'] = '; '.join(faults)
    # ── quality: 7 checks (context checks only with market data) ──
    chk = []
    if has_m:
        md = out.get('mkt_dd_in_base')
        chk.append(('formed in a correction', out.get('base_in_corr')))
        chk.append(('RS line at a 52-week high', out.get('rs_line_hi')))
        chk.append(('held up while SPY fell', bool(out.get('rs_holds')) or bool(out.get('rs_leads'))))
    chk.append(('3+ weeks tight', out.get('tight_weeks', 0) >= 3))
    if has_v:
        chk.append(('volume dry-up', out.get('handle_dryup')))
        chk.append(('accumulation', (out.get('ud_vol_50') or 0) >= 1.2 or out.get('support_wks', 0) >= 1))
    if has_m:
        md = out.get('mkt_dd_in_base')
        ok = (depth is not None and np.isfinite(md) and
              ((md >= 0.05 and depth <= DEEP_VS_MKT * md) or (md < 0.05 and depth <= 0.20)))
        chk.append(('depth in line with the market', ok))
    out['base_q'] = int(sum(bool(x) for _, x in chk))
    out['base_q_max'] = len(chk)
    out['base_q_detail'] = ' · '.join(('✓ ' if x else '✗ ') + k for k, x in chk) + \
        ('' if not faults else ' | faults: ' + ', '.join(faults))
    if typ == 'Double bottom':
        lvl = 1.10 * base['low']
        out['db_sk3_level'] = round(float(lvl), 2)
        a3 = max(1, last - 2)
        cross = any(c[t] > lvl >= c[t - 1] for t in range(a3, last + 1))
        out['db_sk3'] = bool(cross and lvl < base['pivot'] and c[last] > lvl)
    return out


def _base_at(h, l, c, v, e, rs=None, ctx=None):
    """One base per stock, in this order (v3): cup with handle, double bottom, VCP, high tight flag, ascending base,
    flat base / tight area (as base-on-base when it sits on the prior base), IPO base."""
    ctx = ctx or {}
    mdd = ctx.get('mdd')
    tt, _ = _trend_template(h, l, c, e, rs)
    b = (_cup_handle(h, l, c, e, mdd) or _double_bottom(h, l, c, e, mdd) or _vcp(h, l, c, v, e, {'tt': tt})
         or _high_tight_flag(h, l, c, e) or _ascending_base(h, l, c, e))
    if b:
        return b
    b = _pivot_base(h, l, c, e)
    if b:
        ph = [i for i in ctx.get('ph', []) if i <= e - PIV_R] if ctx.get('ph') is not None else pivot_highs(h, e)
        return _base_on_base(h, l, c, e, b, ph) or b
    return _ipo_base(h, l, c, e)


def detect(df: pd.DataFrame, rs_rating: Optional[float] = None, mkt: Optional[pd.DataFrame] = None,
           ftd: Optional[dict] = None) -> dict:
    """All FIELDS for the latest session of df (DatetimeIndex, columns High / Low / Close, optional Volume).
    v3: mkt = SPY daily OHLCV (same kind of dates) enables the market context fields; ftd = follow_through(SPY),
    computed once per scan (computed here from mkt when None). Without mkt the context fields stay None."""
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
    # v3 context arrays (once per call)
    ctx = {'ph': pivot_highs(h, last) if n > PIV_L + PIV_R + 2 else []}
    if mkt is not None and len(mkt) and {'High', 'Low', 'Close'} <= set(mkt.columns):
        try:
            mi = mkt[['High', 'Low', 'Close']].copy()
            mi.index = _naive(mi.index)
            mi = mi[~mi.index.duplicated(keep='last')].reindex(_naive(dates)).ffill()
            mh, ml, mc = (mi[k].to_numpy(float) for k in ('High', 'Low', 'Close'))
            if np.isfinite(mc).sum() >= 0.8 * n:
                ctx.update(mh=mh, ml=ml, mc=mc, mdd=_mdd_fn(mh, ml))
                rsl = c / mc
                ctx['rsl'] = rsl
                ctx['rmax'] = pd.Series(rsl).rolling(252, min_periods=60).max().shift(1).to_numpy()
                ddm = 1 - ml / np.fmax.accumulate(np.where(np.isfinite(mh), mh, 0))
                big = np.nonzero(ddm >= 0.10)[0]
                ctx['mreset_all'] = big
                ctx['ftd'] = ftd if ftd is not None else follow_through(mkt)
        except Exception:
            pass

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
    b = _base_at(h, l, c, v, last - 1, rs_rating, ctx)
    if b:
        base = b
        if c[last] > b['pivot']:
            bo_i = last
    else:
        for k in range(1, RECENT + 1):
            e = last - 1 - k
            if e < 60:
                break
            b = _base_at(h, l, c, v, e, rs_rating, ctx)
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
    # v3: IBD context, character, faults and quality
    try:
        if n > 40:
            ctx['sma200'] = pd.Series(c).rolling(200).mean().to_numpy()
            big = ctx.get('mreset_all')
            if big is not None:
                prior = big[big <= base['start']]
                ctx['mreset'] = int(prior[-1]) if len(prior) else None
            ctx['weekly'] = _weekly(d)
            out.update(_context_quality(d, h, l, c, v, base, e, bo_i, last, ctx))
        if base['type'] == 'Double bottom':
            out.update(db_l1=round(base['db_l1'], 2), db_l1_date=ds(base['db_l1_i']), db_mid=round(base['db_mid'], 2),
                       db_mid_date=ds(base['db_mid_i']))
    except Exception:
        pass
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
