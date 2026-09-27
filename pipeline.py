"""
pipeline.py — MRS core scoring engine (web edition)
====================================================
Extracted from run_mrs.py.  No Excel / openpyxl dependencies.
Import this module from update_data.py (daily batch) and app.py (dashboard).
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Cloud-safe browser headers (bypasses 403 on CBOE/Yahoo in GitHub Actions) ─
_BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/csv,application/csv,*/*',
    'Referer': 'https://www.cboe.com/',
}

# ── Constants ──────────────────────────────────────────────────────────────────
PHI_W    = 756   # 3-year rolling window (~756 trading days)
CBOE_URL = ('https://cdn.cboe.com/data/us/options/market_statistics/'
            'daily_puts_calls.csv')

# ── Component Weights (MRS v4.0 — Sept 2026, v8 component re-test) ───────────
# The dashboard is used as a DISPERSION dial: negative = wider two-sided swings
# over the next 5-60 sessions, positive = calmer, narrower market. v4.0 weights
# and state scores are calibrated to that job (research/studies/v8_component_retest):
#   * State score = -ln(large-move rate in that state / all sessions), averaged
#     over 10 and 21 sessions, 2008-2026, rounded to 0.25. Every state's sign is
#     the same in 2008-16 and 2017-26.
#   * VIX is the anchor (~65% of the score's variation) and is scored
#     continuously from its Phi; VIX Phi alone matched or beat the v3.0
#     composite out of sample.
#   * Extension, Momentum, PC, SKEW and Breadth form an equal-weighted context
#     block (~35%); ADL and B20 share one Breadth weight (they were
#     double-counted before). The context block adds information within a VIX
#     regime (clearly 2017-26, weaker 2008-16).
#   * Volume Divergence is shown for information only (weight 0): it marks a
#     calm uptrend, not distribution, and was not distinctive at drawdown peaks.
#   * Out of sample (split halves) v4.0 was never worse than v3.0 or VIX alone;
#     better than v3.0 in 5 of 8 tests, better than VIX alone in 2 of 8.
#   * The overall scale matches v3.0 (same SD and mean over 2008-2026), so the
#     band cut-offs 1.5 / 0.5 / -0.5 / -1.5 are unchanged.
# Previous v3.0 weights: vix 1.34, ext 0.91, mom 1.25, adl 1.16, b20 1.10,
# pc 1.40, skew 1.14, gamma 1.0, vol 1.0.
COMPONENT_WEIGHTS = {
    'vix':   2.22,  # anchor; score = VIX_INTERCEPT - 3 x vix_phi (continuous)
    'ext':   0.91,  # context block, equal weights
    'mom':   0.91,
    'adl':   0.45,  # Breadth = ADL + B20, half weight each
    'b20':   0.45,
    'pc':    0.91,
    'skew':  0.91,
    'gamma': 1.0,   # unchanged — zero-gamma amplifier study pending (needs GEX history)
    'vol':   0.0,   # information only
}
# VIX score = VIX_INTERCEPT - 3 x Phi: +0.91 at Phi 0 ... 0 at Phi ~0.30 ... -2.09 at Phi 1.
# The intercept carries the calibration offset that keeps the v3.0 scale.
VIX_INTERCEPT = 0.906
SCORING_VERSION = 'MRS v4.0 (Sept 2026, v8 dispersion re-test)'

# Score each component sits at in its "normal" state — used to tell which
# components are pushing the dial away from normal (signal-quality text).
NEUTRAL_SCORE = {'ext': 0.25, 'mom': 0.25, 'adl': 0.25, 'b20': 0.25,
                 'pc': 0.25, 'skew': 0.0, 'gamma': 0.0}

HIST_COLS = [
    'date', 'spy', 'spx', 'vix', 'skew', 'pc_ratio',
    'pc_sma10', 'pc_sma20', 'pc_sma50',
    'sma50', 'ext_raw', 'mom_raw',
    'adl_level', 'adl_roc20',
    'b20_pct', 'b50_pct',
    'zero_gamma',
    'volume', 'price_60d_chg', 'vol_60d_chg', 'vol_divergence',
    'vix_phi', 'ext_phi', 'mom_phi', 'skew_phi', 'adl_phi', 'b20_phi',
    'spike_flag', 'compressed', 'trigger_days',
    'vix_score', 'ext_score', 'mom_score',
    'adl_score', 'b20_score', 'pc_score', 'skew_score', 'gamma_score',
    'vol_score',
    'vix_state', 'ext_state', 'mom_state',
    'adl_state', 'b20_state', 'pc_state', 'skew_state', 'gamma_state',
    'vol_state',
    'mrs_score'
]


# ── Rolling Phi ────────────────────────────────────────────────────────────────
def rolling_phi(series: pd.Series, window: int = PHI_W) -> pd.Series:
    """Empirical percentile rank over a rolling look-back window."""
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        if np.isnan(arr[i]):
            continue
        w     = arr[i - window : i]
        valid = ~np.isnan(w)
        if valid.sum() > 0:
            out[i] = np.nansum(w[valid] < arr[i]) / valid.sum()
    return pd.Series(out, index=series.index)


# ── CBOE PC ratio fetch ────────────────────────────────────────────────────────
def fetch_cboe_pc() -> pd.Series:
    try:
        r = requests.get(CBOE_URL, headers=_BROWSER_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c), None)
        eq_col   = next((c for c in df.columns if 'equity' in c or 'total' in c), None)
        if date_col and eq_col:
            df[date_col] = pd.to_datetime(df[date_col])
            s = df.set_index(date_col)[eq_col].dropna().astype(float)
            s.index = s.index.normalize()
            return s
    except Exception as e:
        print(f'  [WARN] CBOE PC fetch failed: {e}')
    return pd.Series(dtype=float)


# ── Scoring functions (MRS v4.0) ───────────────────────────────────────────────
# Positive = calmer / narrower market, negative = wider two-sided swings.
# Ratios in the comments: large-move rate vs all sessions (10D / 21D, 2008-2026).
def score_vix(phi: float):
    """Continuous anchor. State labels keep the old cut-offs for reading."""
    if np.isnan(phi): return 0.0, 'No data'
    s = round(VIX_INTERCEPT - 3.0 * phi, 2)
    if phi < 0.30:    return s, 'Low'      # 0.49x / 0.48x
    if phi < 0.60:    return s, 'Mid'      # 0.80x / 0.75x
    if phi < 0.80:    return s, 'High'     # 1.27x / 1.26x
    return s, 'Stress'                     # 1.89x / 1.98x

def score_extension(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5,  'Compressed'   # 1.65x / 1.72x
    if phi < 0.70:    return  0.25, 'Normal'       # 0.76x / 0.71x
    return 0.5, 'Extended'                         # 0.68x / 0.67x (was -0.5)

def score_momentum(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5,  'Weak'         # 1.58x / 1.67x
    if phi < 0.70:    return  0.25, 'Normal'       # 0.85x / 0.84x
    return 0.5, 'Strong'                           # 0.69x / 0.63x

def score_adl(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5,  'Weak'         # 1.48x / 1.51x
    if phi < 0.70:    return  0.25, 'Normal'       # 0.82x / 0.80x
    return 0.25, 'Strong'                          # 0.78x / 0.79x (was 0)

def score_b20(phi: float, adl_phi: float = np.nan):
    """B20 Low is now scored on its own (v8: it adds short-horizon information
    with or without ADL Weak). adl_phi kept in the signature for compatibility."""
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5,  'Low'          # 1.51x / 1.46x
    if phi < 0.70:    return  0.25, 'Normal'       # 0.83x / 0.81x
    return 0.25, 'High'                            # 0.74x / 0.81x

def score_pc(pc: float, pc_sma10: float):
    """Five-Zone Model (zone cut-offs June 2026, Studies 7 & 8; ~10th / 18th /
    80th / 90th percentile of pc_sma10). v4.0 scores are for dispersion:
    high put/call = wider swings in BOTH directions (the upside tail of Extreme
    HIGH is part of that wide distribution, not a calm signal)."""
    if np.isnan(pc_sma10): return 0.0, 'No data'
    if pc_sma10 < 0.686:   return  0.25, 'Extreme LOW (complacency)'  # 0.84x / 0.74x
    if pc_sma10 < 0.732:   return  0.25, 'Moderate LOW'               # 0.77x / 0.88x
    if pc_sma10 < 0.944:   return  0.25, 'Mid'                        # 0.88x / 0.86x
    if pc_sma10 < 1.003:   return -0.25, 'Moderate HIGH (fear building)'  # 1.42x / 1.49x
    return                        -0.5,  'Extreme HIGH (fear, widest swings)'  # 1.79x / 1.79x (was +1.0)

def score_skew(phi: float, pc: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30 and not np.isnan(pc) and pc > 1.00:
        return -0.5, 'Low+HighPC(DANGER)'          # 1.73x / 1.71x
    if phi > 0.70 and not np.isnan(pc) and pc < 0.70:
        return  0.5, 'High+LowPC(SAFE)'            # 0.64x / 0.61x
    if phi < 0.30: return -0.25, 'Low'             # 1.20x / 1.26x
    if phi <= 0.70: return 0.0, 'Mid'              # 1.02x / 1.07x
    return 0.25, 'High'                            # 0.83x / 0.76x

def score_gamma(spx: float, zero_gamma: float):
    if np.isnan(spx) or np.isnan(zero_gamma) or zero_gamma <= 0:
        return 0.0, 'No data'
    dist = (spx - zero_gamma) / spx
    if dist > 0.0025:  return  0.5, 'Above Gamma'
    if dist > -0.0025: return  0.0, 'Near Gamma'
    return -0.5, 'Below Gamma'


def score_volume_divergence(price_60d_chg: float, vol_60d_chg: float):
    """
    Volume Divergence — information only in v4.0 (weight 0).
    Condition: price_60d_chg > 0 AND vol_60d_chg < -0.10 (volume down >10%).
    v8 re-test: the flag marks a CALM uptrend (large-move rate 0.67x), with no
    extra downside vs other up-trending sessions; it was on at 42-48% of the
    peaks before >=10% drawdowns vs 49% of up-trending sessions, so the earlier
    '58% of drawdown peaks' claim is not distinctive.
    """
    if np.isnan(price_60d_chg) or np.isnan(vol_60d_chg):
        return 0.0, 'No data'
    if price_60d_chg > 0 and vol_60d_chg < -0.10:
        return 0.0, 'Divergence (calm uptrend)'
    return 0.0, 'Normal'


# ── Recovery Signal (MRS v2.0) ────────────────────────────────────────────────
def compute_recovery_signal(last: dict, hist: pd.DataFrame, ref_date) -> dict:
    """
    Evaluate buy-side recovery signals based on component timing analysis.

    Recovery hierarchy (fastest to slowest at bottoms):
    1. PC Ratio — best buy-side, fastest recovery (+1.4 weight)
    2. B20% — second fastest recovery (+1.1 weight)
    3. Other components follow

    Returns dict with:
      - active: bool (True if recovery signal is firing)
      - strength: 'STRONG' | 'MODERATE' | 'WEAK' | 'NONE'
      - components: list of recovering components
      - description: string explanation
    """
    def _val(key, default=np.nan):
        v = last.get(key, default)
        try:
            return float(v) if not pd.isna(v) else default
        except:
            return default

    mrs_score = _val('mrs_score', 0)
    pc_score = _val('pc_score', 0)
    b20_score = _val('b20_score', 0)
    b20_phi = _val('b20_phi')
    adl_phi = _val('adl_phi')
    vix_phi = _val('vix_phi')
    pc_sma10 = _val('pc_sma10')

    recovering = []
    signals = 0

    # PC Ratio in contrarian HIGH zone (fear = buying opportunity).
    # Uses the zone, not the score (v4.0 scores Extreme HIGH negative for dispersion).
    if not np.isnan(pc_sma10) and pc_sma10 >= 1.003:
        recovering.append('PC Ratio (contrarian high)')
        signals += 2  # Weight 1.4 ≈ 2 points

    # B20% showing strength (High state)
    if not np.isnan(b20_phi) and b20_phi >= 0.70:
        recovering.append('B20% (breadth expanding)')
        signals += 1

    # VIX in spike zone (post-spike recovery tends to be strong)
    if not np.isnan(vix_phi) and vix_phi > 0.70:
        recovering.append('VIX spike (contrarian)')
        signals += 1

    # Check for improving breadth trend (5-day)
    if 'date' in hist.columns:
        df5 = hist[hist['date'] <= pd.Timestamp(ref_date)].sort_values('date').tail(6)
        if 'b20_phi' in df5.columns and len(df5) >= 3:
            b20_vals = df5['b20_phi'].dropna()
            if len(b20_vals) >= 3:
                delta = float(b20_vals.iloc[-1]) - float(b20_vals.iloc[0])
                if delta > 0.03:  # Rising >3% in 5 days
                    recovering.append('B20% trend improving')
                    signals += 1

    # Determine strength
    if signals >= 4:
        strength = 'STRONG'
        color = '22c55e'
    elif signals >= 2:
        strength = 'MODERATE'
        color = '86efac'
    elif signals >= 1:
        strength = 'WEAK'
        color = 'facc15'
    else:
        strength = 'NONE'
        color = '6b7280'

    # Only fire recovery signal when MRS is negative (we're in a drawdown)
    active = signals >= 2 and mrs_score < 0

    # Build description
    if active:
        desc = f"Recovery signals firing: {', '.join(recovering)}. " \
               f"PC Ratio leads recovery historically (fastest to turn positive after troughs)."
    elif signals >= 1 and mrs_score >= 0:
        desc = f"MRS positive — no recovery signal needed. Components healthy: {', '.join(recovering) if recovering else 'baseline'}."
    elif signals >= 1:
        desc = f"Early signs: {', '.join(recovering)}. Waiting for confirmation (need 2+ signals)."
    else:
        desc = "No recovery signals. Monitor PC Ratio (first to recover) and B20% (breadth)."

    return {
        'active': active,
        'strength': strength,
        'color': color,
        'components': recovering,
        'signals': signals,
        'description': desc,
    }


# ── Regime helpers ─────────────────────────────────────────────────────────────
def regime_label(mrs: float) -> str:
    if mrs >= 1.5:  return 'RISK-ON'
    if mrs >= 0.5:  return 'MILD RISK-ON'
    if mrs >= -0.5: return 'NEUTRAL'
    if mrs >= -1.5: return 'MILD RISK-OFF'
    return 'RISK-OFF'

def regime_color(mrs: float) -> str:
    """Hex color for regime band (web display)."""
    if mrs >= 1.5:  return '#1a7f37'
    if mrs >= 0.5:  return '#57a66b'
    if mrs >= -0.5: return '#6b7280'
    if mrs >= -1.5: return '#d97706'
    return '#b91c1c'

def compute_regime_duration(hist: pd.DataFrame, ref_date) -> int:
    df = hist[hist['date'] <= pd.Timestamp(ref_date)].sort_values('date')
    if df.empty:
        return 0
    scores = df['mrs_score'].tolist()
    is_neg = float(scores[-1]) < 0
    count  = 0
    for v in reversed(scores):
        try:
            if (float(v) < 0) == is_neg:
                count += 1
            else:
                break
        except Exception:
            break
    return count


def compute_signal_quality(last: dict, hist: pd.DataFrame, ref_date) -> tuple:
    """
    Evaluate the structural quality of the current MRS regime signal.
    Returns (label, description, hex_color).

    Four quality states:
      CONFIRMED     -- Breadth (B20 + ADL) aligned with regime direction.
      UNCONFIRMED   -- Score carried by non-breadth components; breadth is neutral.
      DIVERGENT     -- Breadth actively opposing the composite regime direction.
      FRAGILE       -- One or more components within 0.05 Phi of a scoring threshold.

    UNCONFIRMED + FRAGILE can co-occur.

    Scoring threshold reference (v4.0):
      B20:  Phi < 0.30 -> -0.5  |  otherwise +0.25
      ADL:  Phi < 0.30 -> -0.5  |  otherwise +0.25
      VIX:  continuous (no threshold); state labels at Phi 0.30 / 0.60 / 0.80
    Component scores are compared with their 'normal' score (NEUTRAL_SCORE), so
    a component in its normal state counts as neutral, not as confirming.
    """
    score   = float(last.get('mrs_score', 0) or 0)
    regime  = regime_label(score)
    is_pos  = score > 0
    is_neg  = score < 0
    is_neut = not is_pos and not is_neg

    # ── Component Phi values ───────────────────────────────────────────────────
    def _phi(key):
        v = last.get(key, np.nan)
        try:    return float(v) if not pd.isna(v) else np.nan
        except: return np.nan

    b20_phi  = _phi('b20_phi')
    adl_phi  = _phi('adl_phi')
    vix_phi  = _phi('vix_phi')
    skew_phi = _phi('skew_phi')

    # ── Component scores ───────────────────────────────────────────────────────
    def _sc(key):
        v = last.get(key, 0)
        try:    return float(v) if not pd.isna(v) else 0.0
        except: return 0.0

    # Centered on each component's normal-state score (v4.0)
    b20_sc  = _sc('b20_score')   - NEUTRAL_SCORE['b20']
    adl_sc  = _sc('adl_score')   - NEUTRAL_SCORE['adl']
    pc_sc   = _sc('pc_score')    - NEUTRAL_SCORE['pc']
    skew_sc = _sc('skew_score')  - NEUTRAL_SCORE['skew']
    mom_sc  = _sc('mom_score')   - NEUTRAL_SCORE['mom']
    vix_sc  = _sc('vix_score')
    ext_sc  = _sc('ext_score')   - NEUTRAL_SCORE['ext']
    gam_sc  = _sc('gamma_score') - NEUTRAL_SCORE['gamma']

    # ── Breadth vs. non-breadth attribution ───────────────────────────────────
    breadth_sum = b20_sc + adl_sc
    flow_sum    = pc_sc  + skew_sc

    if is_pos:
        # v4.0: breadth has no 'strong = extra calm' state (High scores like Normal),
        # so a calm reading is confirmed when breadth is NOT weak.
        breadth_state = 'opposing' if breadth_sum < 0 else 'confirming'
        flow_state    = 'opposing' if flow_sum    < 0 else 'confirming'
    elif is_neg:
        breadth_state = 'confirming' if breadth_sum < 0 else ('opposing' if breadth_sum > 0 else 'neutral')
        flow_state    = 'confirming' if flow_sum    < 0 else ('opposing' if flow_sum    > 0 else 'neutral')
    else:
        breadth_state = 'neutral'
        flow_state    = 'neutral'

    # ── Threshold proximity (within 0.05 Phi of a scoring boundary) ──────────
    PROX    = 0.05
    at_risk = []

    if not np.isnan(b20_phi):
        if b20_sc >= 0 and b20_phi < 0.30 + PROX:
            at_risk.append(
                f'B20 Phi={b20_phi:.3f} is {abs(b20_phi - 0.300):.3f} from bearish threshold '
                f'(cross below 0.300 -> score -0.5)'
            )

    if not np.isnan(adl_phi):
        if adl_sc >= 0 and adl_phi < 0.30 + PROX:
            at_risk.append(
                f'ADL Phi={adl_phi:.3f} is {abs(adl_phi - 0.300):.3f} from bearish threshold '
                f'(cross below 0.300 -> score -0.5)'
            )

    # VIX is continuous in v4.0 — no threshold to be fragile around.

    # ── 5-session Phi trend with quantified delta ──────────────────────────────
    df5 = hist[hist['date'] <= pd.Timestamp(ref_date)].sort_values('date').tail(6)

    def _phi_trend(col):
        vals = df5[col].dropna() if col in df5.columns else pd.Series(dtype=float)
        if len(vals) >= 3:
            delta = float(vals.iloc[-1]) - float(vals.iloc[0])
            direction = 'declining' if delta < -0.02 else ('rising' if delta > 0.02 else 'stable')
            return direction, delta
        return 'unknown', 0.0

    b20_trend, b20_delta = _phi_trend('b20_phi')
    adl_trend, adl_delta = _phi_trend('adl_phi')

    def _trend_phrase(name, phi, trend, delta):
        if np.isnan(phi): return None
        if trend == 'rising':
            return f'{name} Phi rising {delta:+.3f} over 5 sessions (now {phi:.3f})'
        if trend == 'declining':
            return f'{name} Phi declining {delta:+.3f} over 5 sessions (now {phi:.3f})'
        return f'{name} Phi stable at {phi:.3f}'

    breadth_trend_parts = [
        s for s in [
            _trend_phrase('B20', b20_phi, b20_trend, b20_delta),
            _trend_phrase('ADL', adl_phi, adl_trend, adl_delta),
        ] if s
    ]
    breadth_trend_str = '; '.join(breadth_trend_parts) if breadth_trend_parts else None

    # ── Score margin ──────────────────────────────────────────────────────────
    if score >= 1.5:
        margin_desc = f'Score {score:+.2f} -- in RISK-ON territory ({score - 1.5:+.2f} above threshold)'
    elif score >= 0.5:
        margin_desc = f'Score {score:+.2f} -- {1.5 - score:.2f} pts from RISK-ON, {score - 0.5:.2f} pts above MILD RISK-ON floor'
    elif score >= -0.5:
        margin_desc = f'Score {score:+.2f} -- Neutral band ({0.5 - score:.2f} pts from MILD RISK-ON, {score + 0.5:.2f} pts from MILD RISK-OFF)'
    elif score >= -1.5:
        margin_desc = f'Score {score:+.2f} -- {-0.5 - score:.2f} pts below Neutral, {score + 1.5:.2f} pts above RISK-OFF floor'
    else:
        margin_desc = f'Score {score:+.2f} -- in RISK-OFF territory ({-1.5 - score:+.2f} below threshold)'

    # ── Drivers (all non-zero components) ─────────────────────────────────────
    driver_parts = []
    for name, sc in [('PC Ratio', pc_sc), ('SKEW', skew_sc), ('Momentum', mom_sc),
                     ('Gamma', gam_sc), ('VIX', vix_sc), ('Extension', ext_sc),
                     ('B20', b20_sc), ('ADL', adl_sc)]:
        if abs(sc) > 1e-9 and name != 'VIX':
            driver_parts.append(f'{name} ({sc:+.2f} vs normal)')
    if not np.isnan(vix_phi):
        driver_parts.append(f'VIX ({vix_sc:+.2f})')
    drivers_str = ', '.join(driver_parts) if driver_parts else 'no components scoring'

    non_breadth = [p for p in driver_parts
                   if not p.startswith('B20') and not p.startswith('ADL')]
    non_breadth_str = ', '.join(non_breadth) if non_breadth else 'positioning/technical components'

    # ── Fragile note ──────────────────────────────────────────────────────────
    fragile_str = (' | FRAGILE: ' + '; '.join(at_risk)) if at_risk else ''

    # ── Gap to breadth confirmation threshold (regime-aware) ──────────────────
    def _gap_str(phi, is_negative_regime):
        if np.isnan(phi): return None
        if is_negative_regime:
            # Negative confirmation: need Phi to DROP below 0.30
            gap = phi - 0.30
            return f'needs -{gap:.3f} Phi to confirm selloff (cross below 0.300)' if gap > 0 else 'already at bearish breadth threshold'
        else:
            # Positive confirmation: need Phi to RISE above 0.70
            gap = 0.70 - phi
            return f'needs +{gap:.3f} Phi to confirm rally (cross above 0.700)' if gap > 0 else 'already at bullish breadth threshold'

    b20_gap = _gap_str(b20_phi, is_neg)
    adl_gap = _gap_str(adl_phi, is_neg)

    # ── Classification ─────────────────────────────────────────────────────────
    if is_neut:
        if breadth_sum < 0 or flow_sum < 0:
            lbl = 'NEUTRAL -- BEARISH LEAN'
            col = 'C55A11'
            desc = (
                f'Score is zero but internal structure leans bearish. '
                f'Active components: {drivers_str}. '
                + (f'Breadth: {breadth_trend_str}. ' if breadth_trend_str else '')
                + f'{margin_desc}.{fragile_str}'
            )
        elif breadth_sum > 0 or flow_sum > 0:
            lbl = 'NEUTRAL -- BULLISH LEAN'
            col = '375623'
            desc = (
                f'Score is zero but internal structure leans bullish. '
                f'Active components: {drivers_str}. '
                + (f'Breadth: {breadth_trend_str}. ' if breadth_trend_str else '')
                + f'{margin_desc}.{fragile_str}'
            )
        else:
            lbl = 'NEUTRAL -- NO EDGE'
            col = '595959'
            desc = f'All components near zero. No structural bias. {margin_desc}.'

    elif breadth_state == 'confirming' and flow_state != 'opposing':
        lbl = 'CONFIRMED'
        col = '375623'
        b20_str = f'B20 Phi={b20_phi:.3f}' if not np.isnan(b20_phi) else 'B20 N/A'
        adl_str = f'ADL Phi={adl_phi:.3f}' if not np.isnan(adl_phi) else 'ADL N/A'
        _where = ('at least one below the 0.300 weak-breadth threshold' if is_neg
                  else 'neither in the weak zone (below 0.300)')
        desc = (
            f'Breadth confirms: {b20_str}, {adl_str} -- {_where}. '
            f'Drivers: {drivers_str}. '
            + (f'Breadth trend: {breadth_trend_str}. ' if breadth_trend_str else '')
            + f'{margin_desc}.{fragile_str}'
        )

    elif breadth_state == 'opposing':
        lbl = 'DIVERGENT'
        col = '7B0000'
        b20_str = f'B20 Phi={b20_phi:.3f}' if not np.isnan(b20_phi) else 'B20 N/A'
        adl_str = f'ADL Phi={adl_phi:.3f}' if not np.isnan(adl_phi) else 'ADL N/A'
        desc = (
            f'Score is {score:+.2f} ({regime}) but breadth is weak: '
            f'{b20_str}, {adl_str}. '
            f'Score is held up by: {non_breadth_str}. '
            + (f'Breadth trend: {breadth_trend_str}. ' if breadth_trend_str else '')
            + f'Inside a calm VIX regime, weak context components have come with wider '
            f'swings than VIX alone suggests (v8 re-test: 2.4-4x in 2017-26, weaker in 2008-16). '
            + f'{margin_desc}.{fragile_str}'
        )

    else:
        # breadth_state == 'neutral' -> UNCONFIRMED
        lbl = 'UNCONFIRMED'
        col = 'ED7D31'

        gap_parts = []
        if b20_gap:
            gap_parts.append(f'B20 Phi={b20_phi:.3f} ({b20_gap})')
        if adl_gap:
            gap_parts.append(f'ADL Phi={adl_phi:.3f} ({adl_gap})')
        gap_sentence = '; '.join(gap_parts) + '.' if gap_parts else ''

        breadth_threshold_note = (
            'Breadth is not weak: the wide reading is carried by VIX and the other components.'
            if is_neg else
            'Breadth is neutral.'
        )
        desc = (
            f'Score {score:+.2f} is driven by {non_breadth_str}. '
            f'Breadth is not scoring: {gap_sentence} '
            + (f'Breadth trend: {breadth_trend_str}. ' if breadth_trend_str else '')
            + breadth_threshold_note + ' '
            + f'{margin_desc}.{fragile_str}'
        )

    return (lbl, desc, col)


# ── History I/O ────────────────────────────────────────────────────────────────
def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    if 'b50_raw' in df.columns and 'b20_pct' not in df.columns:
        df['b20_pct'] = df['b50_raw']
    for col in HIST_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df.sort_values('date').reset_index(drop=True)

def save_history(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


# ── Score DataFrame (SINGLE SOURCE OF TRUTH) ──────────────────────────────────
def score_dataframe(hist: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Compute all derived signals, Phi values, and MRS scores for a history DataFrame.

    This is the SINGLE SOURCE OF TRUTH for all scoring logic.
    Both update_history() and backfill.py call this function.

    Args:
        hist: DataFrame with raw data (spy, spx, vix, skew, pc_ratio, adl_level,
              b20_pct, zero_gamma, volume)
        verbose: Print progress messages

    Returns:
        DataFrame with all derived columns and scores added
    """
    if verbose:
        print('  Computing derived signals...')

    spy  = hist['spy'].astype(float).ffill()
    vix  = hist['vix'].astype(float)
    skew = hist['skew'].astype(float).ffill()   # SKEW lags 1 day -- carry forward
    pc   = hist['pc_ratio'].astype(float)
    adl  = hist['adl_level'].astype(float)
    b20  = hist['b20_pct'].astype(float)

    hist['sma50']    = spy.rolling(50, min_periods=1).mean()
    hist['ext_raw']  = (spy - hist['sma50']) / hist['sma50']
    hist['mom_raw']  = hist['sma50'] - hist['sma50'].shift(5)  # 5-day SMA50 slope
    hist['pc_sma10'] = pc.rolling(10, min_periods=1).mean()
    hist['pc_sma20'] = pc.rolling(20, min_periods=1).mean()
    hist['pc_sma50'] = pc.rolling(50, min_periods=1).mean()
    adl_prev = adl.shift(20)
    hist['adl_roc20'] = np.where(adl_prev.abs() > 1e-9,
                                 (adl - adl_prev) / adl_prev.abs(), np.nan)

    # Volume divergence (MRS v2.0)
    vol = hist['volume'].astype(float) if 'volume' in hist.columns else pd.Series(np.nan, index=hist.index)
    hist['price_60d_chg'] = spy.pct_change(60)
    hist['vol_60d_chg']   = vol.pct_change(60)
    hist['vol_divergence'] = ((hist['price_60d_chg'] > 0) & (hist['vol_60d_chg'] < -0.10)).astype(int)

    # Rolling Phi
    if verbose:
        print('  Computing Phi...')
    hist['vix_phi']  = rolling_phi(vix,  PHI_W)
    hist['ext_phi']  = rolling_phi(hist['ext_raw'].astype(float), PHI_W)
    hist['mom_phi']  = rolling_phi(hist['mom_raw'].astype(float), PHI_W)
    hist['skew_phi'] = rolling_phi(skew, PHI_W)
    hist['adl_phi']  = rolling_phi(hist['adl_roc20'].astype(float), PHI_W)
    hist['b20_phi']  = rolling_phi(b20, PHI_W)

    # VIX flags
    vix_chg = vix.pct_change(fill_method=None)
    hist['spike_flag'] = (vix_chg > 0.30).fillna(False).astype(int)
    vix_phi_s       = hist['vix_phi']
    compressed_flag = (vix_phi_s < 0.30).fillna(False).astype(int)
    crossed         = ((compressed_flag.shift(1) == 1) & (vix_phi_s >= 0.30)).fillna(False).astype(int)
    trig            = np.zeros(len(hist), dtype=float)
    count           = 0
    for i in range(len(hist)):
        if crossed.iloc[i]:                            count = 1
        elif compressed_flag.iloc[i] == 0 and count:  count += 1
        else:                                          count = 0
        trig[i] = count if 0 < count <= 7 else 0
    hist['trigger_days'] = trig
    hist['compressed']   = compressed_flag

    # Score every row (weighted composite — see COMPONENT_WEIGHTS)
    if verbose:
        print(f'  Scoring ({SCORING_VERSION} with component weights)...')
    score_cols = ['vix_score','ext_score','mom_score','adl_score',
                  'b20_score','pc_score','skew_score','gamma_score','vol_score']
    state_cols = ['vix_state','ext_state','mom_state','adl_state',
                  'b20_state','pc_state','skew_state','gamma_state','vol_state']
    res = {c: [] for c in score_cols + state_cols + ['mrs_score']}

    for _, row in hist.iterrows():
        def g(c):
            v = row[c]; return v if not pd.isna(v) else np.nan

        vs, vst  = score_vix(g('vix_phi'))
        es, est  = score_extension(g('ext_phi'))
        ms, mst  = score_momentum(g('mom_phi'))
        as_, ast = score_adl(g('adl_phi'))
        bs, bst  = score_b20(g('b20_phi'), g('adl_phi'))
        ps, pst  = score_pc(g('pc_ratio'), g('pc_sma10'))
        ss, sst  = score_skew(g('skew_phi'), g('pc_ratio'))
        gs, gst  = score_gamma(g('spx'), g('zero_gamma'))
        vols, volst = score_volume_divergence(g('price_60d_chg'), g('vol_60d_chg'))

        # Apply component weights
        weighted_scores = [
            vs   * COMPONENT_WEIGHTS['vix'],
            es   * COMPONENT_WEIGHTS['ext'],
            ms   * COMPONENT_WEIGHTS['mom'],
            as_  * COMPONENT_WEIGHTS['adl'],
            bs   * COMPONENT_WEIGHTS['b20'],
            ps   * COMPONENT_WEIGHTS['pc'],
            ss   * COMPONENT_WEIGHTS['skew'],
            gs   * COMPONENT_WEIGHTS['gamma'],
            vols * COMPONENT_WEIGHTS['vol'],
        ]
        raw_scores = [vs, es, ms, as_, bs, ps, ss, gs, vols]
        states = [vst, est, mst, ast, bst, pst, sst, gst, volst]
        mrs = round(sum(c for c in weighted_scores if not np.isnan(c)), 2)

        for col, val in zip(score_cols, raw_scores):  res[col].append(val)
        for col, val in zip(state_cols, states):      res[col].append(val)
        res['mrs_score'].append(mrs)

    for col in score_cols + state_cols + ['mrs_score']:
        hist[col] = res[col]

    if verbose:
        print(f'  Done. Latest {SCORING_VERSION}: {hist["mrs_score"].iloc[-1]:+.2f} -- {regime_label(hist["mrs_score"].iloc[-1])}')

    return hist


# ── Core update logic ──────────────────────────────────────────────────────────
def update_history(hist: pd.DataFrame, inp_map: dict) -> pd.DataFrame:
    """
    Append new trading days and rescore the full history.

    inp_map: dict of pd.Timestamp -> {adl_level, b20_pct, zero_gamma, pc_ratio, skew}
             Built by auto_fetch.py (web) or MRS_Inputs_v4.xlsx (local).
    """
    today_dt  = date.today()
    last_date = hist['date'].max()

    # ── 1. Fetch market data (individual calls -- avoids GitHub Actions 403) ───
    print('  Fetching SPY / SPX / VIX / SKEW...')
    start_fetch = (last_date - timedelta(days=10)).strftime('%Y-%m-%d')
    _ticker_map = {'spy': 'SPY', 'spx': '^GSPC', 'vix': '^VIX', 'skew': '^SKEW'}
    _frames = {}
    _volume = None
    for field, ticker in _ticker_map.items():
        try:
            h = yf.Ticker(ticker).history(start=start_fetch, auto_adjust=True)
            if not h.empty:
                h.index = pd.to_datetime(h.index).normalize().tz_localize(None)
                _frames[field] = h['Close'].rename(field)
                if field == 'spy' and 'Volume' in h.columns:
                    _volume = h['Volume'].rename('volume')
        except Exception as e:
            print(f'  [WARN] {ticker}: {e}')
    if not _frames:
        print('  [ERROR] No market data fetched -- aborting.')
        return hist
    close = pd.concat(_frames.values(), axis=1)
    if _volume is not None:
        close = pd.concat([close, _volume], axis=1)
    close.index = close.index.normalize()

    # ── 2. Fetch CBOE PC ratio ────────────────────────────────────────────────
    print('  Fetching CBOE PC ratio...')
    pc_series = fetch_cboe_pc()

    # ── 3. Carry-forward helper ───────────────────────────────────────────────
    _empty = dict(adl_level=np.nan, b20_pct=np.nan, zero_gamma=np.nan,
                  pc_ratio=np.nan, skew=np.nan)

    def _get_manual(dt):
        if not inp_map: return _empty
        if dt in inp_map: return inp_map[dt]
        prior = [d for d in inp_map if d < dt]
        return inp_map[max(prior)] if prior else _empty

    # ── 4. Append new rows ────────────────────────────────────────────────────
    new_dates = [d for d in close.index if d > last_date]
    new_rows  = []
    for dt in new_dates:
        row = {col: np.nan for col in HIST_COLS}
        row['date'] = dt
        m = _get_manual(dt)

        if dt in close.index:
            row['spy']    = float(close.loc[dt, 'spy'])    if 'spy'    in close.columns else np.nan
            row['spx']    = float(close.loc[dt, 'spx'])    if 'spx'    in close.columns else np.nan
            row['vix']    = float(close.loc[dt, 'vix'])    if 'vix'    in close.columns else np.nan
            row['volume'] = float(close.loc[dt, 'volume']) if 'volume' in close.columns else np.nan
            yf_skew       = float(close.loc[dt, 'skew'])   if 'skew'   in close.columns else np.nan
            row['skew']   = yf_skew if not np.isnan(yf_skew) else m['skew']

        if dt in pc_series.index:
            row['pc_ratio'] = float(pc_series.loc[dt])
        elif not np.isnan(m['pc_ratio']):
            row['pc_ratio'] = m['pc_ratio']

        row['adl_level']    = m['adl_level']
        row['b20_pct']      = m['b20_pct']
        row['zero_gamma']   = m['zero_gamma']
        row['spike_flag']   = 0
        row['compressed']   = 0
        row['trigger_days'] = 0.0
        new_rows.append(row)

    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
        hist = hist.sort_values('date').reset_index(drop=True)
        print(f'  Appended {len(new_rows)} new row(s).')
    else:
        print('  No new market dates to append.')

    # ── 4b. Retroactive fill of manual inputs ─────────────────────────────────
    # Manual daily inputs (dashboard sidebar → backfill.py) always win: the
    # automated run only fills a blank, it never overwrites a stored value.
    manual_cols = ['adl_level', 'b20_pct', 'zero_gamma']
    hist = hist.set_index('date')
    patched = 0
    for hist_date in hist.index:
        m = _get_manual(hist_date)
        for col in manual_cols:
            val = m.get(col, np.nan)
            if not np.isnan(val):
                old = hist.loc[hist_date, col]
                if pd.isna(old):
                    hist.loc[hist_date, col] = val
                    patched += 1
        if not np.isnan(m['pc_ratio']) and pd.isna(hist.loc[hist_date, 'pc_ratio']):
            hist.loc[hist_date, 'pc_ratio'] = m['pc_ratio']
        if not np.isnan(m['skew']) and pd.isna(hist.loc[hist_date, 'skew']):
            hist.loc[hist_date, 'skew'] = m['skew']
    hist = hist.reset_index()
    if patched:
        print(f'  Retroactive fill: {patched} blank field(s) filled.')

    # ── 5. Score using single source of truth ─────────────────────────────────
    hist = score_dataframe(hist, verbose=True)

    return hist
