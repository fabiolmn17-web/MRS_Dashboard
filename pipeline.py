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

# ── Component Weights (MRS v2.0 - July 2026 calibration) ──────────────────────
# Weights derived from component timing analysis:
# - Sell-side: components that warn earliest before drawdowns
# - Buy-side: components that recover fastest at bottoms
# Validated robust across 1yr, 2yr, 3yr, 5yr Phi windows
COMPONENT_WEIGHTS = {
    'vix':  1.3,   # Strong sell-side (2nd best lead time, most first-to-warn)
    'ext':  1.2,   # Good sell-side (3rd best lead time)
    'mom':  1.0,   # Average timing
    'adl':  1.0,   # Average timing
    'b20':  1.1,   # Good buy-side (2nd fastest recovery)
    'pc':   1.4,   # Best buy-side (fastest recovery, most first-to-recover)
    'skew': 1.3,   # Best sell-side lead time
    'gamma': 1.0,  # No timing data available
    'vol':  1.0,   # Volume divergence (sell-side only)
}

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


# ── Scoring functions ──────────────────────────────────────────────────────────
def score_vix(phi: float):
    if np.isnan(phi): return  0.0, 'No data'
    if phi < 0.30:    return  1.0, 'Low'
    if phi < 0.60:    return  0.0, 'Mid'
    if phi < 0.80:    return -0.5, 'High'
    return -1.5, 'Stress'

def score_extension(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5, 'Compressed'
    if phi < 0.70:    return  0.0, 'Normal'
    return -0.5, 'Extended'

def score_momentum(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return  0.0, 'Normal'
    return 0.5, 'Strong'

def score_adl(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return  0.0, 'Normal'
    return 0.0, 'Strong'

def score_b20(phi: float, adl_phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:
        s = -0.5 if (not np.isnan(adl_phi) and adl_phi < 0.30) else 0.0
        return s, 'Low'
    if phi < 0.70: return 0.0, 'Normal'
    return 0.5, 'High'

def score_pc(pc: float, pc_sma10: float):
    """Five-Zone Model — June 2026 calibration (Studies 7 & 8)."""
    if np.isnan(pc_sma10): return 0.0, 'No data'
    if pc_sma10 < 0.686:   return  0.5, 'Extreme LOW (complacency)'
    if pc_sma10 < 0.732:   return -0.5, 'Moderate LOW (transition)'
    if pc_sma10 < 0.944:   return  0.0, 'Mid'
    if pc_sma10 < 1.003:   return  0.5, 'Moderate HIGH (fear building)'
    return                         1.0, 'Extreme HIGH (contrarian)'

def score_skew(phi: float, pc: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30 and not np.isnan(pc) and pc > 1.00:
        return -2.0, 'Low+HighPC(DANGER)'
    if phi > 0.70 and not np.isnan(pc) and pc < 0.70:
        return  1.5, 'High+LowPC(SAFE)'
    if phi < 0.30: return -1.0, 'Low'
    if phi < 0.70: return  0.0, 'Mid'
    return 0.5, 'High'

def score_gamma(spx: float, zero_gamma: float):
    if np.isnan(spx) or np.isnan(zero_gamma) or zero_gamma <= 0:
        return 0.0, 'No data'
    dist = (spx - zero_gamma) / spx
    if dist > 0.0025:  return  0.5, 'Above Gamma'
    if dist > -0.0025: return  0.0, 'Near Gamma'
    return -0.5, 'Below Gamma'


def score_volume_divergence(price_60d_chg: float, vol_60d_chg: float):
    """
    Volume Divergence (MRS v2.0 - July 2026 calibration).

    Bearish signal: Price rising but volume declining = distribution.
    Condition: price_60d_chg > 0 AND vol_60d_chg < -0.10 (volume down >10%)

    Statistical validation: 58% of drawdown peaks showed this pattern.
    Raw divergence outperformed seasonal normalization (odds ratio 1.24 vs 0.61).
    """
    if np.isnan(price_60d_chg) or np.isnan(vol_60d_chg):
        return 0.0, 'No data'
    if price_60d_chg > 0 and vol_60d_chg < -0.10:
        return -0.5, 'Divergence (bearish)'
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

    # PC Ratio in contrarian HIGH zone (fear = buying opportunity)
    if pc_score >= 0.5:
        recovering.append('PC Ratio (contrarian high)')
        signals += 2  # Weight 1.4 ≈ 2 points

    # B20% showing strength
    if b20_score > 0:
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

    Scoring threshold reference:
      B20:  Phi < 0.30 -> -0.5  |  Phi > 0.70 -> +0.5
      ADL:  Phi < 0.30 -> -1.0  |  Phi > 0.70 -> +0.5
      VIX:  Phi > 0.70 -> -0.5  |  Phi < 0.30 -> +1.0
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

    b20_sc  = _sc('b20_score')
    adl_sc  = _sc('adl_score')
    pc_sc   = _sc('pc_score')
    skew_sc = _sc('skew_score')
    mom_sc  = _sc('mom_score')
    vix_sc  = _sc('vix_score')
    ext_sc  = _sc('ext_score')
    gam_sc  = _sc('gamma_score')

    # ── Breadth vs. non-breadth attribution ───────────────────────────────────
    breadth_sum = b20_sc + adl_sc
    flow_sum    = pc_sc  + skew_sc

    if is_pos:
        breadth_state = 'confirming' if breadth_sum > 0 else ('opposing' if breadth_sum < 0 else 'neutral')
        flow_state    = 'confirming' if flow_sum    > 0 else ('opposing' if flow_sum    < 0 else 'neutral')
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
        elif b20_sc <= 0 and b20_phi > 0.70 - PROX:
            at_risk.append(
                f'B20 Phi={b20_phi:.3f} is {abs(0.700 - b20_phi):.3f} from bullish threshold '
                f'(cross above 0.700 -> score +0.5)'
            )

    if not np.isnan(adl_phi):
        if adl_sc >= 0 and adl_phi < 0.30 + PROX:
            at_risk.append(
                f'ADL Phi={adl_phi:.3f} is {abs(adl_phi - 0.300):.3f} from bearish threshold '
                f'(cross below 0.300 -> score -1.0)'
            )
        elif adl_sc <= 0 and adl_phi > 0.70 - PROX:
            at_risk.append(
                f'ADL Phi={adl_phi:.3f} is {abs(0.700 - adl_phi):.3f} from bullish threshold '
                f'(cross above 0.700 -> score +0.5)'
            )

    if not np.isnan(vix_phi):
        if vix_sc >= 0 and vix_phi > 0.70 - PROX:
            at_risk.append(
                f'VIX Phi={vix_phi:.3f} is {abs(0.700 - vix_phi):.3f} from bearish threshold '
                f'(cross above 0.700 -> score -0.5)'
            )
        elif vix_sc <= 0 and vix_phi < 0.30 + PROX:
            at_risk.append(
                f'VIX Phi={vix_phi:.3f} is {abs(vix_phi - 0.300):.3f} from bullish threshold '
                f'(cross below 0.300 -> score +1.0)'
            )

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
        if sc != 0:
            driver_parts.append(f'{name} ({sc:+.1f})')
    drivers_str = ', '.join(driver_parts) if driver_parts else 'no components scoring'

    non_breadth = [p for p in driver_parts
                   if not p.startswith('B20') and not p.startswith('ADL')]
    non_breadth_str = ', '.join(non_breadth) if non_breadth else 'positioning/technical components'

    # ── Fragile note ──────────────────────────────────────────────────────────
    fragile_str = (' | FRAGILE: ' + '; '.join(at_risk)) if at_risk else ''

    # ── Gap to +0.5 scoring threshold ─────────────────────────────────────────
    def _gap_str(phi, threshold=0.70):
        if np.isnan(phi): return None
        gap = threshold - phi
        return f'needs +{gap:.3f} Phi to score' if gap > 0 else 'already past scoring threshold'

    b20_gap = _gap_str(b20_phi)
    adl_gap = _gap_str(adl_phi)

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
        desc = (
            f'Breadth confirms: {b20_str}, {adl_str} -- both above the 0.700 scoring threshold. '
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
            f'Score is {score:+.2f} ({regime}) but breadth is actively opposing: '
            f'{b20_str}, {adl_str}. '
            f'Score is held up by: {drivers_str}. '
            + (f'Breadth trend: {breadth_trend_str}. ' if breadth_trend_str else '')
            + f'DIVERGENT regimes that do not recover breadth within 5-7 sessions '
            f'historically resolve to the downside. '
            f'Do not add risk until B20 and ADL Phi stabilize. '
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

        desc = (
            f'Score {score:+.2f} is driven by {non_breadth_str}. '
            f'Breadth is not scoring: {gap_sentence} '
            + (f'Breadth trend: {breadth_trend_str}. ' if breadth_trend_str else '')
            + f'Until B20 or ADL Phi crosses 0.700, treat this as a positioning signal only. '
            f'{margin_desc}.{fragile_str}'
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

    # Score every row (MRS v2.0 with weights)
    if verbose:
        print('  Scoring (MRS v2.0 with component weights)...')
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

        # Apply component weights (MRS v2.0)
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
        print(f'  Done. Latest MRS v2.0: {hist["mrs_score"].iloc[-1]:+.2f} -- {regime_label(hist["mrs_score"].iloc[-1])}')

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

    # ── 4b. Retroactive carry-forward patch ───────────────────────────────────
    manual_cols = ['adl_level', 'b20_pct', 'zero_gamma']
    hist = hist.set_index('date')
    patched = 0
    for hist_date in hist.index:
        m = _get_manual(hist_date)
        for col in manual_cols:
            val = m.get(col, np.nan)
            if not np.isnan(val):
                old = hist.loc[hist_date, col]
                if pd.isna(old) or old != val:
                    hist.loc[hist_date, col] = val
                    patched += 1
        if not np.isnan(m['pc_ratio']) and pd.isna(hist.loc[hist_date, 'pc_ratio']):
            hist.loc[hist_date, 'pc_ratio'] = m['pc_ratio']
        if not np.isnan(m['skew']) and pd.isna(hist.loc[hist_date, 'skew']):
            hist.loc[hist_date, 'skew'] = m['skew']
    hist = hist.reset_index()
    if patched:
        print(f'  Retroactive patch: {patched} field(s) corrected.')

    # ── 5. Score using single source of truth ─────────────────────────────────
    hist = score_dataframe(hist, verbose=True)

    return hist
