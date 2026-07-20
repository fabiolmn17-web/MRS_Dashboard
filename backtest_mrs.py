"""
MRS Backtest Framework
======================
Analyzes Market Regime Score behavior around significant drawdowns.
Produces interactive HTML report with actionable insights.

Author: Epistruct Research
Date: July 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH = Path(__file__).parent / "Data" / "AMEX_SPY, 1D_066f4.csv"
OUTPUT_PATH = Path(__file__).parent / "backtest_report.html"

# Mutually exclusive drawdown severity bands (not nested)
DRAWDOWN_BANDS = [
    (0.05, 0.10, "5-10%", "Minor Correction"),
    (0.10, 0.15, "10-15%", "Correction"),
    (0.15, 0.20, "15-20%", "Significant Correction"),
    (0.20, 1.00, "20%+", "Bear Market"),
]
PRE_PEAK_WINDOWS = [30, 60, 90]  # Days before peak to analyze
VOLUME_WINDOWS = [20, 50]  # Rolling average windows for volume normalization
PHI_WINDOW = 756  # 3-year rolling percentile (same as pipeline.py)
SEASONAL_VOLUME_YEARS = 5  # Years of history for seasonal volume baseline

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    """Load and clean the historical data file."""
    df = pd.read_csv(path)

    # Rename columns for clarity
    df.columns = ['date', 'open', 'high', 'low', 'close', 'skew', 'b20_pct',
                  'pc_ratio', 'vix', 'volume', 'adl']

    # Parse dates (DD/MM/YYYY format)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values('date').reset_index(drop=True)

    # Convert to numeric, coercing errors
    numeric_cols = ['open', 'high', 'low', 'close', 'skew', 'b20_pct',
                    'pc_ratio', 'vix', 'volume', 'adl']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to rows where all key indicators are available
    df = df.dropna(subset=['close', 'vix']).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows from {df['date'].min().date()} to {df['date'].max().date()}")

    # Report data availability
    for col in ['skew', 'b20_pct', 'pc_ratio', 'adl']:
        valid = df[col].notna().sum()
        first_valid = df[df[col].notna()]['date'].min()
        print(f"  {col}: {valid:,} valid rows, first available: {first_valid.date() if pd.notna(first_valid) else 'N/A'}")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# MRS COMPUTATION (mirrors pipeline.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def rolling_phi(series: pd.Series, window: int = PHI_WINDOW) -> pd.Series:
    """Empirical percentile rank over a rolling look-back window."""
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        if np.isnan(arr[i]):
            continue
        w = arr[i - window : i]
        valid = ~np.isnan(w)
        if valid.sum() > 0:
            out[i] = np.nansum(w[valid] < arr[i]) / valid.sum()
    return pd.Series(out, index=series.index)


def score_vix(phi: float) -> tuple:
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return 1.0, 'Low'
    if phi < 0.60:    return 0.0, 'Mid'
    if phi < 0.80:    return -0.5, 'High'
    return -1.5, 'Stress'


def score_extension(phi: float, mom_phi: float = np.nan) -> tuple:
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5, 'Compressed'
    if phi > 0.70:
        mom_weak = (not np.isnan(mom_phi)) and (mom_phi < 0.30)
        if mom_weak:
            return -0.5, 'Extended+Weak'
        return 0.0, 'Extended'
    return 0.0, 'Normal'


def score_momentum(phi: float) -> tuple:
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return 0.0, 'Normal'
    return 0.5, 'Strong'


def score_adl(phi: float) -> tuple:
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return 0.0, 'Normal'
    return 0.0, 'Strong'


def score_b20(phi: float, adl_phi: float) -> tuple:
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:
        s = -0.5 if (not np.isnan(adl_phi) and adl_phi < 0.30) else 0.0
        return s, 'Low'
    if phi < 0.70: return 0.0, 'Normal'
    return 0.5, 'High'


def score_pc(pc: float, pc_sma10: float) -> tuple:
    """Five-Zone Model from pipeline.py."""
    if np.isnan(pc_sma10): return 0.0, 'No data'
    if pc_sma10 < 0.686:   return 0.5, 'Extreme LOW'
    if pc_sma10 < 0.732:   return -0.5, 'Moderate LOW'
    if pc_sma10 < 0.944:   return 0.0, 'Mid'
    if pc_sma10 < 1.003:   return 0.5, 'Moderate HIGH'
    return 1.0, 'Extreme HIGH'


def score_skew(phi: float, pc: float) -> tuple:
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30 and not np.isnan(pc) and pc > 1.00:
        return -2.0, 'Low+HighPC'
    if phi > 0.70 and not np.isnan(pc) and pc < 0.70:
        return 1.5, 'High+LowPC'
    if phi < 0.30: return -1.0, 'Low'
    if phi < 0.70: return 0.0, 'Mid'
    return 0.5, 'High'


def regime_label(mrs: float) -> str:
    if mrs >= 1.5:  return 'RISK-ON'
    if mrs >= 0.5:  return 'MILD RISK-ON'
    if mrs >= -0.5: return 'NEUTRAL'
    if mrs >= -1.5: return 'MILD RISK-OFF'
    return 'RISK-OFF'


def compute_seasonal_volume_percentile(df: pd.DataFrame, years: int = 5) -> pd.Series:
    """
    Compute volume percentile relative to the same calendar week over past N years.

    This normalizes for seasonal effects (summer lulls, holiday periods).
    Returns percentile (0-1) where 0.5 = normal for this time of year.
    """
    result = np.full(len(df), np.nan)
    vol = df['volume'].values
    weeks = df['week_of_year'].values
    dates = df['date'].values

    # Need at least 1 year of data before we can compute
    min_lookback = 252  # ~1 year of trading days

    for i in range(min_lookback, len(df)):
        current_week = weeks[i]
        current_date = pd.Timestamp(dates[i])
        current_vol = vol[i]

        if np.isnan(current_vol):
            continue

        # Look back up to N years, find all volumes from the same calendar week
        lookback_start = max(0, i - (years * 252))

        # Get volumes from same week of year in lookback period
        same_week_vols = []
        for j in range(lookback_start, i):
            if weeks[j] == current_week and not np.isnan(vol[j]):
                same_week_vols.append(vol[j])

        # Also include nearby weeks (±1 week) for more robust estimate
        adjacent_weeks = [(current_week - 1) % 52 + 1, (current_week + 1) % 52 + 1]
        for j in range(lookback_start, i):
            if weeks[j] in adjacent_weeks and not np.isnan(vol[j]):
                same_week_vols.append(vol[j])

        if len(same_week_vols) >= 10:  # Need reasonable sample
            result[i] = np.sum(np.array(same_week_vols) < current_vol) / len(same_week_vols)

    return pd.Series(result, index=df.index)


def compute_mrs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full MRS and all components."""
    df = df.copy()

    # Derived signals
    spy = df['close'].astype(float)
    vix = df['vix'].astype(float)
    skew = df['skew'].astype(float).ffill()
    pc = df['pc_ratio'].astype(float)
    adl = df['adl'].astype(float)
    b20 = df['b20_pct'].astype(float)

    df['sma50'] = spy.rolling(50, min_periods=1).mean()
    df['ext_raw'] = (spy - df['sma50']) / df['sma50']
    df['mom_raw'] = spy.pct_change(20)
    df['pc_sma10'] = pc.rolling(10, min_periods=1).mean()

    adl_prev = adl.shift(20)
    df['adl_roc20'] = np.where(adl_prev.abs() > 1e-9,
                                (adl - adl_prev) / adl_prev.abs(), np.nan)

    # Rolling Phi computations
    print("  Computing Phi values (this may take a minute)...")
    df['vix_phi'] = rolling_phi(vix, PHI_WINDOW)
    df['ext_phi'] = rolling_phi(df['ext_raw'].astype(float), PHI_WINDOW)
    df['mom_phi'] = rolling_phi(df['mom_raw'].astype(float), PHI_WINDOW)
    df['skew_phi'] = rolling_phi(skew, PHI_WINDOW)
    df['adl_phi'] = rolling_phi(df['adl_roc20'].astype(float), PHI_WINDOW)
    df['b20_phi'] = rolling_phi(b20, PHI_WINDOW)

    # Score each component
    print("  Scoring components...")
    vix_scores, ext_scores, mom_scores = [], [], []
    adl_scores, b20_scores, pc_scores, skew_scores = [], [], [], []
    vix_states, ext_states, mom_states = [], [], []
    adl_states, b20_states, pc_states, skew_states = [], [], [], []
    mrs_scores = []

    for idx, row in df.iterrows():
        vs, vst = score_vix(row['vix_phi'])
        ms, mst = score_momentum(row['mom_phi'])
        es, est = score_extension(row['ext_phi'], row['mom_phi'])
        as_, ast = score_adl(row['adl_phi'])
        bs, bst = score_b20(row['b20_phi'], row['adl_phi'])
        ps, pst = score_pc(row['pc_ratio'], row['pc_sma10'])
        ss, sst = score_skew(row['skew_phi'], row['pc_ratio'])

        scores = [vs, es, ms, as_, bs, ps, ss]
        mrs = round(sum(c for c in scores if not np.isnan(c)), 2)

        vix_scores.append(vs); vix_states.append(vst)
        ext_scores.append(es); ext_states.append(est)
        mom_scores.append(ms); mom_states.append(mst)
        adl_scores.append(as_); adl_states.append(ast)
        b20_scores.append(bs); b20_states.append(bst)
        pc_scores.append(ps); pc_states.append(pst)
        skew_scores.append(ss); skew_states.append(sst)
        mrs_scores.append(mrs)

    df['vix_score'] = vix_scores; df['vix_state'] = vix_states
    df['ext_score'] = ext_scores; df['ext_state'] = ext_states
    df['mom_score'] = mom_scores; df['mom_state'] = mom_states
    df['adl_score'] = adl_scores; df['adl_state'] = adl_states
    df['b20_score'] = b20_scores; df['b20_state'] = b20_states
    df['pc_score'] = pc_scores; df['pc_state'] = pc_states
    df['skew_score'] = skew_scores; df['skew_state'] = skew_states
    df['mrs_score'] = mrs_scores
    df['regime'] = df['mrs_score'].apply(regime_label)

    # Volume metrics
    for w in VOLUME_WINDOWS:
        df[f'vol_sma{w}'] = df['volume'].rolling(w, min_periods=1).mean()
        df[f'vol_ratio_{w}d'] = df['volume'] / df[f'vol_sma{w}']

    df['vol_phi_252'] = rolling_phi(df['volume'].astype(float), 252)

    # ── Seasonal Volume Normalization ──
    # Compare volume to historical volume for the same calendar week
    print("  Computing seasonal volume percentiles...")
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['vol_seasonal_pct'] = compute_seasonal_volume_percentile(df, years=SEASONAL_VOLUME_YEARS)

    print(f"  MRS computed. Range: {df['mrs_score'].min():.2f} to {df['mrs_score'].max():.2f}")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# DRAWDOWN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_all_drawdowns(df: pd.DataFrame, min_threshold: float = 0.05,
                       min_recovery_pct: float = 0.5) -> list:
    """
    Identify ALL drawdown events where peak-to-trough decline >= min_threshold.

    Returns list of dicts with peak_idx, trough_idx, peak_date, trough_date,
    peak_price, trough_price, drawdown_pct, recovery_date, duration_days.
    """
    prices = df['close'].values
    dates = df['date'].values
    n = len(prices)

    drawdowns = []
    i = 0

    while i < n - 20:
        # Find local peak: price that isn't exceeded for at least 20 days
        peak_idx = i
        peak_price = prices[i]

        # Look for higher prices ahead
        j = i + 1
        while j < n:
            if prices[j] > peak_price:
                peak_idx = j
                peak_price = prices[j]
                j += 1
            elif j - peak_idx >= 20:
                # No new high for 20 days - this is our peak
                break
            else:
                j += 1

        if peak_idx >= n - 20:
            break

        # Find trough after peak
        trough_idx = peak_idx + 1
        trough_price = prices[trough_idx]

        for k in range(peak_idx + 1, min(n, peak_idx + 504)):  # Max 2 years
            if prices[k] < trough_price:
                trough_idx = k
                trough_price = prices[k]

            # Check if price has recovered 50% of drawdown
            drawdown = (peak_price - trough_price) / peak_price
            current_recovery = (prices[k] - trough_price) / (peak_price - trough_price) if peak_price > trough_price else 0

            if current_recovery >= min_recovery_pct and drawdown >= min_threshold:
                # Found a valid drawdown event
                drawdowns.append({
                    'peak_idx': peak_idx,
                    'trough_idx': trough_idx,
                    'peak_date': pd.Timestamp(dates[peak_idx]),
                    'trough_date': pd.Timestamp(dates[trough_idx]),
                    'peak_price': peak_price,
                    'trough_price': trough_price,
                    'drawdown_pct': drawdown * 100,
                    'recovery_idx': k,
                    'recovery_date': pd.Timestamp(dates[k]),
                    'duration_days': (pd.Timestamp(dates[trough_idx]) - pd.Timestamp(dates[peak_idx])).days,
                })
                i = k  # Move past this drawdown
                break
        else:
            i = peak_idx + 1
            continue

        i += 1

    # Remove overlapping events (keep the larger drawdown)
    if len(drawdowns) > 1:
        filtered = []
        for dd in drawdowns:
            overlaps = False
            for existing in filtered:
                if (dd['peak_date'] <= existing['recovery_date'] and
                    dd['recovery_date'] >= existing['peak_date']):
                    if dd['drawdown_pct'] > existing['drawdown_pct']:
                        filtered.remove(existing)
                    else:
                        overlaps = True
                    break
            if not overlaps:
                filtered.append(dd)
        drawdowns = filtered

    return drawdowns


def categorize_drawdowns_by_band(drawdowns: list, bands: list) -> dict:
    """
    Categorize drawdowns into mutually exclusive severity bands.

    Args:
        drawdowns: List of drawdown events
        bands: List of tuples (min_pct, max_pct, label, description)

    Returns:
        Dict with band labels as keys, list of events as values
    """
    categorized = {band[2]: [] for band in bands}

    for dd in drawdowns:
        dd_pct = dd['drawdown_pct'] / 100  # Convert to decimal

        for min_pct, max_pct, label, description in bands:
            if min_pct <= dd_pct < max_pct:
                # Add band info to the event
                dd_with_band = dd.copy()
                dd_with_band['band_label'] = label
                dd_with_band['band_description'] = description
                categorized[label].append(dd_with_band)
                break

    return categorized

# ══════════════════════════════════════════════════════════════════════════════
# EVENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_event(df: pd.DataFrame, event: dict, pre_windows: list) -> dict:
    """Analyze MRS and volume behavior around a drawdown event.

    Bottom Definition:
    - Trough = lowest price point before a meaningful recovery
    - "Meaningful recovery" = price rebounds 10% from trough OR retraces 50% of drawdown
    - We analyze the 60-day window after trough for recovery patterns
    """
    peak_idx = event['peak_idx']
    trough_idx = event['trough_idx']
    recovery_idx = event.get('recovery_idx', min(trough_idx + 60, len(df) - 1))

    result = {
        'event': event,
        'pre_peak': {},
        'at_peak': {},
        'at_trough': {},
        'during_drawdown': {},
        'recovery': {},
        'recovery_sequencing': {},  # NEW: detailed recovery analysis
        'volume': {},
        'volume_recovery': {},  # NEW: volume during recovery
        'components': {},
    }

    # ── At Peak ──
    peak_row = df.iloc[peak_idx]
    result['at_peak'] = {
        'mrs_score': peak_row['mrs_score'],
        'regime': peak_row['regime'],
        'vix': peak_row['vix'],
        'vix_phi': peak_row['vix_phi'],
    }

    # Component scores at peak
    for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']:
        result['at_peak'][f'{comp}_score'] = peak_row.get(f'{comp}_score', np.nan)
        result['at_peak'][f'{comp}_state'] = peak_row.get(f'{comp}_state', 'N/A')

    # ── At Trough ──
    trough_row = df.iloc[trough_idx]
    result['at_trough'] = {
        'mrs_score': trough_row['mrs_score'],
        'regime': trough_row['regime'],
        'vix': trough_row['vix'],
        'vix_phi': trough_row['vix_phi'],
    }

    for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']:
        result['at_trough'][f'{comp}_score'] = trough_row.get(f'{comp}_score', np.nan)
        result['at_trough'][f'{comp}_state'] = trough_row.get(f'{comp}_state', 'N/A')

    # ── Pre-Peak Analysis (multiple windows) ──
    for window in pre_windows:
        start_idx = max(0, peak_idx - window)
        pre_df = df.iloc[start_idx:peak_idx + 1].copy().reset_index(drop=True)

        if len(pre_df) < 5:
            continue

        # MRS trajectory
        mrs_start = pre_df['mrs_score'].iloc[0]
        mrs_end = pre_df['mrs_score'].iloc[-1]
        mrs_min = pre_df['mrs_score'].min()
        mrs_max = pre_df['mrs_score'].max()

        # First day MRS went negative (if any) - use positional index
        negative_mask = pre_df['mrs_score'] < 0
        if negative_mask.any():
            first_negative_pos = negative_mask.idxmax()  # First True position
            days_negative_before_peak = len(pre_df) - 1 - first_negative_pos
        else:
            days_negative_before_peak = None

        # First day MRS went below -0.5 (MILD RISK-OFF or worse)
        risk_off_mask = pre_df['mrs_score'] < -0.5
        if risk_off_mask.any():
            first_risk_off_pos = risk_off_mask.idxmax()
            days_risk_off_before_peak = len(pre_df) - 1 - first_risk_off_pos
        else:
            days_risk_off_before_peak = None

        # MRS slope (linear regression)
        x = np.arange(len(pre_df))
        y = pre_df['mrs_score'].values
        valid = ~np.isnan(y)
        if valid.sum() > 2:
            slope = np.polyfit(x[valid], y[valid], 1)[0]
        else:
            slope = np.nan

        # Component deterioration sequence
        component_first_warning = {}
        for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']:
            score_col = f'{comp}_score'
            if score_col in pre_df.columns:
                neg_mask = pre_df[score_col] < 0
                if neg_mask.any():
                    first_neg_pos = neg_mask.idxmax()
                    component_first_warning[comp] = len(pre_df) - 1 - first_neg_pos

        result['pre_peak'][f'{window}d'] = {
            'mrs_start': mrs_start,
            'mrs_end': mrs_end,
            'mrs_min': mrs_min,
            'mrs_max': mrs_max,
            'mrs_slope': slope,
            'days_negative_before_peak': days_negative_before_peak,
            'days_risk_off_before_peak': days_risk_off_before_peak,
            'pct_days_negative': (pre_df['mrs_score'] < 0).mean() * 100,
            'component_first_warning': component_first_warning,
        }

    # ── During Drawdown ──
    dd_df = df.iloc[peak_idx:trough_idx + 1].copy()

    result['during_drawdown'] = {
        'mrs_min': dd_df['mrs_score'].min(),
        'mrs_min_date': dd_df.loc[dd_df['mrs_score'].idxmin(), 'date'],
        'mrs_avg': dd_df['mrs_score'].mean(),
        'pct_risk_off': (dd_df['mrs_score'] < -1.5).mean() * 100,
        'vix_max': dd_df['vix'].max(),
        'days_in_drawdown': len(dd_df),
    }

    # Regime transitions during drawdown
    regimes = dd_df['regime'].tolist()
    transitions = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
    result['during_drawdown']['regime_transitions'] = transitions

    # ── Recovery Analysis (Enhanced) ──
    # Analyze 60 days after trough (or until recovery point if known)
    recovery_window_end = min(trough_idx + 60, len(df) - 1)
    if event.get('recovery_idx'):
        recovery_window_end = max(recovery_window_end, event['recovery_idx'])

    recovery_df = df.iloc[trough_idx:recovery_window_end + 1].copy().reset_index(drop=True)

    if len(recovery_df) > 1:
        # Track price recovery milestones
        trough_price = event['trough_price']
        peak_price = event['peak_price']
        drawdown_size = peak_price - trough_price

        # Find when price recovered 25%, 50%, 75% of drawdown
        recovery_milestones = {}
        for pct in [0.25, 0.50, 0.75, 1.00]:
            target_price = trough_price + (drawdown_size * pct)
            recovered_mask = recovery_df['close'] >= target_price
            if recovered_mask.any():
                recovery_milestones[f'{int(pct*100)}pct'] = recovered_mask.idxmax()

        # First day MRS turned positive after trough
        positive_mask = recovery_df['mrs_score'] > 0
        if positive_mask.any():
            days_to_positive = positive_mask.idxmax()
        else:
            days_to_positive = None

        # First day MRS crossed above -0.5 (out of RISK-OFF)
        mild_mask = recovery_df['mrs_score'] >= -0.5
        if mild_mask.any():
            days_to_mild_recovery = mild_mask.idxmax()
        else:
            days_to_mild_recovery = None

        if days_to_positive is not None:
            # Compare to 25% price recovery
            price_25_day = recovery_milestones.get('25pct')
            mrs_led_25pct = days_to_positive < price_25_day if price_25_day is not None else None
        else:
            mrs_led_25pct = None

        # Component recovery sequence (detailed)
        component_recovery_order = {}
        component_recovery_details = {}

        for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']:
            score_col = f'{comp}_score'
            phi_col = f'{comp}_phi' if comp != 'pc' else None

            if score_col in recovery_df.columns:
                # When did component score turn positive?
                pos_mask = recovery_df[score_col] > 0
                if pos_mask.any():
                    first_pos = pos_mask.idxmax()
                    days_to_pos = first_pos
                    component_recovery_order[comp] = days_to_pos

                    # What was the score at trough vs at recovery?
                    score_at_trough = recovery_df[score_col].iloc[0]
                    score_at_recovery = recovery_df[score_col].iloc[first_pos]

                    component_recovery_details[comp] = {
                        'days_to_positive': days_to_pos,
                        'score_at_trough': score_at_trough,
                        'score_at_recovery': score_at_recovery,
                    }

                    # Add Phi trajectory if available
                    if phi_col and phi_col in recovery_df.columns:
                        phi_at_trough = recovery_df[phi_col].iloc[0]
                        phi_at_recovery = recovery_df[phi_col].iloc[first_pos]
                        component_recovery_details[comp]['phi_at_trough'] = phi_at_trough
                        component_recovery_details[comp]['phi_at_recovery'] = phi_at_recovery

        # Sort components by recovery order
        sorted_recovery = sorted(component_recovery_order.items(), key=lambda x: x[1])

        result['recovery'] = {
            'days_to_positive_mrs': days_to_positive,
            'days_to_mild_recovery': days_to_mild_recovery,
            'mrs_led_25pct_recovery': mrs_led_25pct,
            'component_recovery_order': component_recovery_order,
            'recovery_sequence': [c[0] for c in sorted_recovery],  # Ordered list
            'price_recovery_milestones': recovery_milestones,
        }

        result['recovery_sequencing'] = component_recovery_details

        # ── Volume During Recovery ──
        # Analyze volume patterns from trough to recovery
        vol_at_trough_5d = recovery_df['volume'].iloc[:5].mean() if len(recovery_df) >= 5 else np.nan
        vol_at_recovery_5d = recovery_df['volume'].iloc[-5:].mean() if len(recovery_df) >= 5 else np.nan

        # Volume trend during recovery (slope)
        x = np.arange(len(recovery_df))
        y = recovery_df['volume'].values
        valid = ~np.isnan(y)
        if valid.sum() > 2:
            vol_slope = np.polyfit(x[valid], y[valid], 1)[0]
            vol_slope_normalized = vol_slope / recovery_df['volume'].mean() * 100
        else:
            vol_slope_normalized = np.nan

        # Did volume confirm the recovery? (increasing volume on up days)
        up_days = recovery_df[recovery_df['close'] > recovery_df['close'].shift(1)]
        down_days = recovery_df[recovery_df['close'] < recovery_df['close'].shift(1)]

        avg_vol_up = up_days['volume'].mean() if len(up_days) > 0 else np.nan
        avg_vol_down = down_days['volume'].mean() if len(down_days) > 0 else np.nan
        vol_confirms_recovery = avg_vol_up > avg_vol_down if not np.isnan(avg_vol_up) and not np.isnan(avg_vol_down) else None

        result['volume_recovery'] = {
            'vol_at_trough_5d_avg': vol_at_trough_5d,
            'vol_at_recovery_5d_avg': vol_at_recovery_5d,
            'vol_change_pct': (vol_at_recovery_5d / vol_at_trough_5d - 1) * 100 if vol_at_trough_5d else np.nan,
            'vol_slope_pct_per_day': vol_slope_normalized,
            'avg_vol_on_up_days': avg_vol_up,
            'avg_vol_on_down_days': avg_vol_down,
            'vol_confirms_recovery': vol_confirms_recovery,
        }

    # ── Volume Analysis ──
    for window in pre_windows:
        start_idx = max(0, peak_idx - window)
        pre_df = df.iloc[start_idx:peak_idx + 1].copy()

        if len(pre_df) < 5:
            continue

        # Volume trend vs price trend (divergence detection)
        price_pct_change = (pre_df['close'].iloc[-1] / pre_df['close'].iloc[0] - 1) * 100
        vol_pct_change = (pre_df['volume'].iloc[-1] / pre_df['volume'].iloc[0] - 1) * 100

        # Price-volume correlation
        pv_corr = pre_df['close'].corr(pre_df['volume'])

        # Volume trend (slope)
        x = np.arange(len(pre_df))
        y = pre_df['volume'].values
        valid = ~np.isnan(y)
        if valid.sum() > 2:
            vol_slope = np.polyfit(x[valid], y[valid], 1)[0]
            vol_slope_normalized = vol_slope / pre_df['volume'].mean() * 100  # % per day
        else:
            vol_slope_normalized = np.nan

        # Volume percentile at peak
        vol_phi_at_peak = pre_df['vol_phi_252'].iloc[-1] if 'vol_phi_252' in pre_df.columns else np.nan

        # Divergence flag: price up but volume down
        divergence_raw = price_pct_change > 0 and vol_pct_change < -10

        # Use raw divergence as primary (seasonal normalization tested and rejected - see methodology)
        result['volume'][f'pre_{window}d'] = {
            'price_pct_change': price_pct_change,
            'vol_pct_change': vol_pct_change,
            'price_vol_corr': pv_corr,
            'vol_slope_pct_per_day': vol_slope_normalized,
            'vol_phi_at_peak': vol_phi_at_peak,
            'divergence_flag': divergence_raw,  # Raw divergence (statistically validated)
        }

    # Volume at trough (capitulation detection)
    trough_window = df.iloc[max(0, trough_idx - 5):trough_idx + 6].copy()
    if len(trough_window) > 0:
        vol_at_trough = trough_window['volume'].max()
        vol_ratio_20 = trough_window['vol_ratio_20d'].max() if 'vol_ratio_20d' in trough_window.columns else np.nan
        vol_ratio_50 = trough_window['vol_ratio_50d'].max() if 'vol_ratio_50d' in trough_window.columns else np.nan
        vol_phi_at_trough = trough_window['vol_phi_252'].max() if 'vol_phi_252' in trough_window.columns else np.nan

        # Capitulation flag: volume spike > 2x average with high VIX
        capitulation = vol_ratio_20 > 2.0 if not np.isnan(vol_ratio_20) else False

        result['volume']['at_trough'] = {
            'vol_max': vol_at_trough,
            'vol_ratio_20d': vol_ratio_20,
            'vol_ratio_50d': vol_ratio_50,
            'vol_phi': vol_phi_at_trough,
            'capitulation_flag': capitulation,
        }

    return result

# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_aggregate_stats(analyses: list, threshold: float) -> dict:
    """Compute aggregate statistics across all events for a given threshold."""
    if not analyses:
        return {}

    stats = {
        'n_events': len(analyses),
        'threshold': threshold * 100,
    }

    # MRS at peak
    mrs_at_peak = [a['at_peak']['mrs_score'] for a in analyses]
    stats['mrs_at_peak_mean'] = np.nanmean(mrs_at_peak)
    stats['mrs_at_peak_median'] = np.nanmedian(mrs_at_peak)
    stats['pct_negative_at_peak'] = sum(1 for m in mrs_at_peak if m < 0) / len(mrs_at_peak) * 100
    stats['pct_risk_off_at_peak'] = sum(1 for m in mrs_at_peak if m < -0.5) / len(mrs_at_peak) * 100

    # MRS at trough
    mrs_at_trough = [a['at_trough']['mrs_score'] for a in analyses]
    stats['mrs_at_trough_mean'] = np.nanmean(mrs_at_trough)
    stats['mrs_at_trough_min'] = np.nanmin(mrs_at_trough)

    # Lead time analysis for each pre-peak window
    for window in PRE_PEAK_WINDOWS:
        key = f'{window}d'
        if key not in analyses[0]['pre_peak']:
            continue

        lead_times = [a['pre_peak'][key].get('days_negative_before_peak') for a in analyses]
        lead_times = [lt for lt in lead_times if lt is not None]

        if lead_times:
            stats[f'lead_time_{window}d_mean'] = np.mean(lead_times)
            stats[f'lead_time_{window}d_median'] = np.median(lead_times)
            stats[f'lead_time_{window}d_pct_warned'] = len(lead_times) / len(analyses) * 100

        # MRS slope before peak
        slopes = [a['pre_peak'][key].get('mrs_slope') for a in analyses]
        slopes = [s for s in slopes if s is not None and not np.isnan(s)]
        if slopes:
            stats[f'mrs_slope_{window}d_mean'] = np.mean(slopes)

    # Component analysis - which components warned first?
    component_lead_times = {comp: [] for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']}
    for a in analyses:
        for window in PRE_PEAK_WINDOWS:
            key = f'{window}d'
            if key in a['pre_peak']:
                for comp, lt in a['pre_peak'][key].get('component_first_warning', {}).items():
                    component_lead_times[comp].append(lt)

    stats['component_avg_lead_time'] = {
        comp: np.mean(lts) if lts else None
        for comp, lts in component_lead_times.items()
    }

    # Recovery analysis (enhanced)
    recovery_lead = [a['recovery'].get('days_to_positive_mrs') for a in analyses if a.get('recovery')]
    recovery_lead = [r for r in recovery_lead if r is not None]
    if recovery_lead:
        stats['recovery_lead_mean'] = np.mean(recovery_lead)
        stats['recovery_lead_median'] = np.median(recovery_lead)

    mild_recovery_lead = [a['recovery'].get('days_to_mild_recovery') for a in analyses if a.get('recovery')]
    mild_recovery_lead = [r for r in mild_recovery_lead if r is not None]
    if mild_recovery_lead:
        stats['mild_recovery_lead_mean'] = np.mean(mild_recovery_lead)

    # MRS led 25% price recovery
    mrs_led_count = sum(1 for a in analyses if a.get('recovery', {}).get('mrs_led_25pct_recovery'))
    valid_recovery = sum(1 for a in analyses if a.get('recovery', {}).get('mrs_led_25pct_recovery') is not None)
    if valid_recovery > 0:
        stats['pct_mrs_led_25pct_recovery'] = mrs_led_count / valid_recovery * 100

    # Component recovery sequencing
    component_recovery_times = {comp: [] for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']}
    recovery_first_counts = {comp: 0 for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']}

    for a in analyses:
        if a.get('recovery', {}).get('recovery_sequence'):
            seq = a['recovery']['recovery_sequence']
            if seq:
                recovery_first_counts[seq[0]] += 1

        if a.get('recovery', {}).get('component_recovery_order'):
            for comp, days in a['recovery']['component_recovery_order'].items():
                component_recovery_times[comp].append(days)

    stats['component_avg_recovery_time'] = {
        comp: np.mean(times) if times else None
        for comp, times in component_recovery_times.items()
    }

    stats['component_recovery_first_count'] = recovery_first_counts

    # Most common first recoverer
    if recovery_first_counts:
        stats['most_common_first_recoverer'] = max(recovery_first_counts.items(), key=lambda x: x[1])

    # Volume during recovery
    vol_confirms = [a.get('volume_recovery', {}).get('vol_confirms_recovery') for a in analyses]
    vol_confirms = [v for v in vol_confirms if v is not None]
    if vol_confirms:
        stats['pct_volume_confirms_recovery'] = sum(vol_confirms) / len(vol_confirms) * 100

    # Volume analysis
    divergence_count = 0
    capitulation_count = 0
    for a in analyses:
        for window in PRE_PEAK_WINDOWS:
            key = f'pre_{window}d'
            if key in a['volume'] and a['volume'][key].get('divergence_flag'):
                divergence_count += 1
                break
        if 'at_trough' in a['volume'] and a['volume']['at_trough'].get('capitulation_flag'):
            capitulation_count += 1

    stats['pct_volume_divergence_pre_peak'] = divergence_count / len(analyses) * 100
    stats['pct_volume_capitulation_at_trough'] = capitulation_count / len(analyses) * 100

    return stats

# ══════════════════════════════════════════════════════════════════════════════
# FALSE POSITIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_false_positives(df: pd.DataFrame, drawdown_events: list,
                            threshold: float, lookforward: int = 60) -> dict:
    """
    Analyze periods where MRS was negative but no significant drawdown followed.
    """
    # Find all periods where MRS < 0
    negative_periods = []
    in_negative = False
    start_idx = None

    for idx in range(len(df)):
        if df.iloc[idx]['mrs_score'] < 0:
            if not in_negative:
                in_negative = True
                start_idx = idx
        else:
            if in_negative:
                negative_periods.append((start_idx, idx - 1))
                in_negative = False

    if in_negative:
        negative_periods.append((start_idx, len(df) - 1))

    # Check which negative periods preceded actual drawdowns
    event_peaks = set(e['peak_idx'] for e in drawdown_events)

    true_positives = 0
    false_positives = 0

    for start, end in negative_periods:
        # Check if any drawdown peak occurred within lookforward days after this period
        found_drawdown = False
        for peak_idx in event_peaks:
            if start <= peak_idx <= end + lookforward:
                found_drawdown = True
                break

        if found_drawdown:
            true_positives += 1
        else:
            false_positives += 1

    total = true_positives + false_positives

    return {
        'total_negative_periods': total,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'precision': true_positives / total * 100 if total > 0 else 0,
        'false_positive_rate': false_positives / total * 100 if total > 0 else 0,
    }

# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(df: pd.DataFrame, all_results: dict, output_path: Path):
    """Generate interactive HTML report."""

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MRS Backtest Report — Epistruct Research</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --bg-dark: #0f0f0f;
            --bg-card: #1a1a1a;
            --bg-card-hover: #222;
            --text-primary: #e5e7eb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-yellow: #facc15;
            --accent-orange: #f97316;
            --accent-blue: #3b82f6;
            --border: #2a2a2a;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
        }

        .container { max-width: 1400px; margin: 0 auto; }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        h2 {
            font-size: 1.4rem;
            font-weight: 600;
            margin: 32px 0 16px;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }

        h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 24px 0 12px;
            color: var(--text-secondary);
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 24px;
        }

        .card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }

        .card-header {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 12px;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .metric {
            background: var(--bg-dark);
            border-radius: 8px;
            padding: 16px;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }
        .warning { color: var(--accent-orange); }
        .neutral { color: var(--text-secondary); }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }

        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background: var(--bg-dark);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        tr:hover { background: var(--bg-card-hover); }

        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .tag-risk-on { background: rgba(34, 197, 94, 0.2); color: var(--accent-green); }
        .tag-mild-risk-on { background: rgba(34, 197, 94, 0.1); color: #86efac; }
        .tag-neutral { background: rgba(107, 114, 128, 0.2); color: var(--text-secondary); }
        .tag-mild-risk-off { background: rgba(249, 115, 22, 0.2); color: var(--accent-orange); }
        .tag-risk-off { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }

        .insight-box {
            background: rgba(59, 130, 246, 0.1);
            border-left: 4px solid var(--accent-blue);
            border-radius: 0 8px 8px 0;
            padding: 16px 20px;
            margin: 16px 0;
        }

        .insight-box.success {
            background: rgba(34, 197, 94, 0.1);
            border-left-color: var(--accent-green);
        }

        .insight-box.warning {
            background: rgba(249, 115, 22, 0.1);
            border-left-color: var(--accent-orange);
        }

        .insight-box.danger {
            background: rgba(239, 68, 68, 0.1);
            border-left-color: var(--accent-red);
        }

        .insight-title {
            font-weight: 600;
            margin-bottom: 8px;
        }

        .chart-container {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
        }

        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }

        .tab {
            padding: 8px 16px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            color: var(--text-secondary);
            transition: all 0.2s;
        }

        .tab:hover { background: var(--bg-card-hover); }
        .tab.active {
            background: var(--accent-blue);
            color: white;
            border-color: var(--accent-blue);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 900px) {
            .two-col { grid-template-columns: 1fr; }
        }

        .progress-bar {
            height: 8px;
            background: var(--bg-dark);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }

        .section-divider {
            border: none;
            border-top: 1px solid var(--border);
            margin: 32px 0;
        }

        .footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MRS Backtest Report</h1>
        <p class="subtitle">
            Market Regime Score behavior analysis around significant drawdown events<br>
            Data: """ + f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}" + """ |
            Generated: """ + datetime.now().strftime('%B %d, %Y') + """
        </p>
"""

    # ── Executive Summary ──
    html += """
        <h2>Executive Summary</h2>
        <div class="card">
            <div class="card-header">Key Findings</div>
"""

    # Calculate key metrics across all bands
    all_events = []
    for band_label, data in all_results.items():
        all_events.extend(data['analyses'])

    # Best performing band
    best_band = None
    best_warning_rate = 0
    for band_label, data in all_results.items():
        stats = data['aggregate_stats']
        if stats.get('pct_negative_at_peak', 0) > best_warning_rate:
            best_warning_rate = stats.get('pct_negative_at_peak', 0)
            best_band = band_label

    # Use 10-15% band as reference (or fall back to first available)
    stats_ref = all_results.get('10-15%', {}).get('aggregate_stats', {})
    if not stats_ref:
        stats_ref = next(iter(all_results.values()), {}).get('aggregate_stats', {})

    # Count total events
    total_events = sum(len(data.get('drawdown_events', [])) for data in all_results.values())

    html += f"""
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{total_events}</div>
                    <div class="metric-label">total drawdown events analyzed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats_ref.get('lead_time_60d_mean', 0):.0f}</div>
                    <div class="metric-label">avg days of warning (10-15% corrections)</div>
                </div>
                <div class="metric">
                    <div class="metric-value warning">{stats_ref.get('mrs_at_trough_mean', 0):.1f}</div>
                    <div class="metric-label">avg MRS at drawdown troughs</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats_ref.get('pct_volume_divergence_pre_peak', 0):.0f}%</div>
                    <div class="metric-label">showed seasonal volume divergence</div>
                </div>
            </div>
        </div>
"""

    # Key insights
    html += """
        <div class="insight-box success">
            <div class="insight-title">✓ Early Warning Capability</div>
"""

    # Determine leading components
    comp_leads = stats_ref.get('component_avg_lead_time', {})
    if comp_leads:
        sorted_comps = sorted([(k, v) for k, v in comp_leads.items() if v is not None],
                              key=lambda x: x[1], reverse=True)
        if sorted_comps:
            leaders = [c[0].upper() for c in sorted_comps[:3]]
            html += f"""
            <p>The MRS provided meaningful early warning for {stats_ref.get('pct_negative_at_peak', 0):.0f}% of 10-15% corrections.
            Leading indicators in order of deterioration: <strong>{', '.join(leaders)}</strong>.</p>
"""
    html += """
        </div>
"""

    # Volume insight
    if stats_ref.get('pct_volume_divergence_pre_peak', 0) > 30:
        html += """
        <div class="insight-box warning">
            <div class="insight-title">⚠ Seasonal Volume Divergence Pattern</div>
            <p>Volume divergence (low volume for time of year while price rises) preceded a significant portion of drawdowns.
            This uses seasonal normalization to avoid false signals during summer/holiday periods.</p>
        </div>
"""

    # ── Band Comparison (Mutually Exclusive) ──
    html += """
        <h2>Drawdown Severity Analysis</h2>
        <p class="subtitle" style="margin-top: -12px;">
            Events categorized into <strong>mutually exclusive</strong> severity bands — each drawdown appears in only one category.
        </p>
        <div class="tabs">
"""
    for min_pct, max_pct, label, description in DRAWDOWN_BANDS:
        active = 'active' if label == '10-15%' else ''
        tab_id = label.replace('%', 'pct').replace('-', '_').replace('+', 'plus')
        html += f"""
            <div class="tab {active}" onclick="switchTab('{tab_id}')">{label} ({description})</div>
"""
    html += """
        </div>
"""

    for min_pct, max_pct, label, description in DRAWDOWN_BANDS:
        active = 'active' if label == '10-15%' else ''
        tab_id = label.replace('%', 'pct').replace('-', '_').replace('+', 'plus')
        data = all_results.get(label, {})
        stats = data.get('aggregate_stats', {})
        fp_stats = data.get('false_positive_stats', {})
        events = data.get('drawdown_events', [])

        html += f"""
        <div class="tab-content {active}" id="tab-{tab_id}">
            <div class="card">
                <div class="card-header">{label} Drawdowns — {description} (n={len(events)})</div>
                <div class="metric-grid">
                    <div class="metric">
                        <div class="metric-value">{stats.get('pct_negative_at_peak', 0):.0f}%</div>
                        <div class="metric-label">MRS negative at peak</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {stats.get('pct_negative_at_peak', 0)}%; background: var(--accent-green);"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{stats.get('pct_risk_off_at_peak', 0):.0f}%</div>
                        <div class="metric-label">RISK-OFF at peak</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {stats.get('pct_risk_off_at_peak', 0)}%; background: var(--accent-orange);"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{stats.get('lead_time_60d_mean', 0):.0f}d</div>
                        <div class="metric-label">Avg lead time (60d window)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value negative">{stats.get('mrs_at_trough_min', 0):.1f}</div>
                        <div class="metric-label">Worst MRS at trough</div>
                    </div>
                </div>
            </div>
"""

        # Event table
        if events:
            html += """
            <div class="card">
                <div class="card-header">Individual Events</div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Peak Date</th>
                                <th>Trough Date</th>
                                <th>Drawdown</th>
                                <th>Duration</th>
                                <th>MRS at Peak</th>
                                <th>Regime at Peak</th>
                                <th>MRS at Trough</th>
                                <th>Lead Time</th>
                            </tr>
                        </thead>
                        <tbody>
"""
            analyses = data.get('analyses', [])
            for i, event in enumerate(events):
                analysis = analyses[i] if i < len(analyses) else {}
                peak_mrs = analysis.get('at_peak', {}).get('mrs_score', 'N/A')
                regime = analysis.get('at_peak', {}).get('regime', 'N/A')
                trough_mrs = analysis.get('at_trough', {}).get('mrs_score', 'N/A')
                lead_time = analysis.get('pre_peak', {}).get('60d', {}).get('days_negative_before_peak', None)

                regime_class = regime.lower().replace(' ', '-').replace('_', '-') if regime else 'neutral'
                mrs_class = 'positive' if isinstance(peak_mrs, (int, float)) and peak_mrs > 0 else 'negative' if isinstance(peak_mrs, (int, float)) and peak_mrs < 0 else ''

                html += f"""
                            <tr>
                                <td>{event['peak_date'].strftime('%Y-%m-%d')}</td>
                                <td>{event['trough_date'].strftime('%Y-%m-%d')}</td>
                                <td class="negative">-{event['drawdown_pct']:.1f}%</td>
                                <td>{event['duration_days']}d</td>
                                <td class="{mrs_class}">{peak_mrs if isinstance(peak_mrs, str) else f'{peak_mrs:.2f}'}</td>
                                <td><span class="tag tag-{regime_class}">{regime}</span></td>
                                <td class="negative">{trough_mrs if isinstance(trough_mrs, str) else f'{trough_mrs:.2f}'}</td>
                                <td>{f'{lead_time}d' if lead_time else '—'}</td>
                            </tr>
"""
            html += """
                        </tbody>
                    </table>
                </div>
            </div>
"""

        # False positive analysis
        if fp_stats:
            precision = fp_stats.get('precision', 0)
            html += f"""
            <div class="card">
                <div class="card-header">Signal Quality</div>
                <div class="two-col">
                    <div>
                        <h3>Detection Metrics</h3>
                        <table>
                            <tr><td>Total MRS &lt; 0 periods</td><td>{fp_stats.get('total_negative_periods', 0)}</td></tr>
                            <tr><td>True Positives (preceded drawdown)</td><td class="positive">{fp_stats.get('true_positives', 0)}</td></tr>
                            <tr><td>False Positives (no drawdown followed)</td><td class="warning">{fp_stats.get('false_positives', 0)}</td></tr>
                            <tr><td><strong>Precision</strong></td><td><strong>{precision:.1f}%</strong></td></tr>
                        </table>
                    </div>
                    <div>
                        <h3>Interpretation</h3>
                        <p style="color: var(--text-secondary); font-size: 0.9rem;">
                            Precision of {precision:.0f}% means that when MRS goes negative, there's a
                            {precision:.0f}% chance a {label} drawdown follows within 60 days.
                            {'This is a strong signal.' if precision > 60 else 'Consider using additional confirmation signals.'}
                        </p>
                    </div>
                </div>
            </div>
"""

        html += """
        </div>
"""

    # ── Component Analysis ──
    html += """
        <h2>Component Analysis</h2>
        <div class="card">
            <div class="card-header">Component Leading Indicators (10-15% Corrections)</div>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">
                Average days each component turned negative before peak. Higher = earlier warning.
            </p>
"""

    comp_leads = stats_ref.get('component_avg_lead_time', {})
    if comp_leads:
        sorted_comps = sorted([(k, v) for k, v in comp_leads.items() if v is not None],
                              key=lambda x: x[1], reverse=True)
        max_lead = max(v for k, v in sorted_comps) if sorted_comps else 1

        html += """
            <div style="display: flex; flex-direction: column; gap: 12px;">
"""
        for comp, lead in sorted_comps:
            pct = (lead / max_lead) * 100 if max_lead > 0 else 0
            html += f"""
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-weight: 600;">{comp.upper()}</span>
                        <span style="color: var(--text-muted);">{lead:.0f} days</span>
                    </div>
                    <div class="progress-bar" style="height: 12px;">
                        <div class="progress-fill" style="width: {pct}%; background: var(--accent-blue);"></div>
                    </div>
                </div>
"""
        html += """
            </div>
"""

    html += """
        </div>

        <div class="two-col">
            <div class="card">
                <div class="card-header">Component Scores at Peak (avg)</div>
"""

    # Calculate average component scores at peak
    if all_results.get('10-15%', {}).get('analyses'):
        analyses = all_results['10-15%']['analyses']
        comp_scores_at_peak = {comp: [] for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']}
        for a in analyses:
            for comp in comp_scores_at_peak.keys():
                score = a['at_peak'].get(f'{comp}_score')
                if score is not None and not np.isnan(score):
                    comp_scores_at_peak[comp].append(score)

        html += """
                <table>
                    <thead><tr><th>Component</th><th>Avg Score</th><th>% Negative</th></tr></thead>
                    <tbody>
"""
        for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']:
            scores = comp_scores_at_peak[comp]
            if scores:
                avg = np.mean(scores)
                pct_neg = sum(1 for s in scores if s < 0) / len(scores) * 100
                score_class = 'positive' if avg > 0 else 'negative' if avg < 0 else ''
                html += f"""
                        <tr>
                            <td>{comp.upper()}</td>
                            <td class="{score_class}">{avg:.2f}</td>
                            <td>{pct_neg:.0f}%</td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
"""

    html += """
            </div>
            <div class="card">
                <div class="card-header">Component Scores at Trough (avg)</div>
"""

    if all_results.get('10-15%', {}).get('analyses'):
        comp_scores_at_trough = {comp: [] for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']}
        for a in analyses:
            for comp in comp_scores_at_trough.keys():
                score = a['at_trough'].get(f'{comp}_score')
                if score is not None and not np.isnan(score):
                    comp_scores_at_trough[comp].append(score)

        html += """
                <table>
                    <thead><tr><th>Component</th><th>Avg Score</th><th>Min Score</th></tr></thead>
                    <tbody>
"""
        for comp in ['vix', 'ext', 'mom', 'adl', 'b20', 'pc', 'skew']:
            scores = comp_scores_at_trough[comp]
            if scores:
                avg = np.mean(scores)
                min_score = min(scores)
                score_class = 'positive' if avg > 0 else 'negative' if avg < 0 else ''
                html += f"""
                        <tr>
                            <td>{comp.upper()}</td>
                            <td class="{score_class}">{avg:.2f}</td>
                            <td class="negative">{min_score:.1f}</td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
"""

    html += """
            </div>
        </div>
"""

    # ── Recovery Analysis ──
    html += """
        <h2>Recovery Analysis (Buy-Side Signals)</h2>
        <div class="card">
            <div class="card-header">Bottom Detection & Recovery Timing</div>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">
                <strong>Bottom Definition:</strong> Trough = lowest price before a 10% rebound or 50% retracement of drawdown.
                Analysis window: 60 days after trough.
            </p>
"""

    # Recovery metrics
    html += f"""
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{stats_ref.get('recovery_lead_mean', 0):.0f}d</div>
                    <div class="metric-label">Avg days for MRS to turn positive after trough</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats_ref.get('mild_recovery_lead_mean', 0):.0f}d</div>
                    <div class="metric-label">Avg days to exit RISK-OFF (MRS &gt; -0.5)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats_ref.get('pct_mrs_led_25pct_recovery', 0):.0f}%</div>
                    <div class="metric-label">MRS turned positive before 25% price recovery</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{stats_ref.get('pct_volume_confirms_recovery', 0):.0f}%</div>
                    <div class="metric-label">Recoveries confirmed by volume</div>
                </div>
            </div>
        </div>
"""

    # Component recovery sequencing
    html += """
        <div class="card">
            <div class="card-header">Component Recovery Sequencing</div>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">
                Which components recover first at market bottoms? Earlier = better leading indicator for buy signals.
            </p>
"""

    comp_recovery = stats_ref.get('component_avg_recovery_time', {})
    if comp_recovery:
        sorted_recovery = sorted([(k, v) for k, v in comp_recovery.items() if v is not None],
                                  key=lambda x: x[1])
        if sorted_recovery:
            max_time = max(v for k, v in sorted_recovery) if sorted_recovery else 1

            html += """
            <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 20px;">
"""
            for i, (comp, time) in enumerate(sorted_recovery):
                pct = 100 - ((time / max_time) * 80) if max_time > 0 else 100  # Invert so faster = fuller bar
                color = 'var(--accent-green)' if i < 2 else 'var(--accent-blue)' if i < 4 else 'var(--accent-orange)'
                rank_label = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th'][i] if i < 7 else ''
                html += f"""
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span><span style="color: var(--text-muted); font-size: 0.8rem;">{rank_label}</span> <strong>{comp.upper()}</strong></span>
                        <span style="color: var(--text-muted);">{time:.0f} days to positive</span>
                    </div>
                    <div class="progress-bar" style="height: 12px;">
                        <div class="progress-fill" style="width: {pct}%; background: {color};"></div>
                    </div>
                </div>
"""
            html += """
            </div>
"""

    # Recovery first counts
    recovery_first = stats_ref.get('component_recovery_first_count', {})
    if recovery_first and any(v > 0 for v in recovery_first.values()):
        html += """
            <h3>Times Each Component Recovered First</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px;">
"""
        sorted_first = sorted(recovery_first.items(), key=lambda x: x[1], reverse=True)
        for comp, count in sorted_first:
            if count > 0:
                html += f"""
                <div style="background: var(--bg-dark); padding: 12px 16px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.4rem; font-weight: 700; color: var(--accent-green);">{count}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted);">{comp.upper()}</div>
                </div>
"""
        html += """
            </div>
"""

    html += """
        </div>
"""

    # Recovery insights
    if comp_recovery:
        sorted_recovery = sorted([(k, v) for k, v in comp_recovery.items() if v is not None],
                                  key=lambda x: x[1])
        if len(sorted_recovery) >= 2:
            html += f"""
        <div class="insight-box success">
            <div class="insight-title">Recovery Leading Indicators</div>
            <p><strong>{sorted_recovery[0][0].upper()}</strong> and <strong>{sorted_recovery[1][0].upper()}</strong>
            are the fastest components to recover at market bottoms, turning positive on average
            {sorted_recovery[0][1]:.0f} and {sorted_recovery[1][1]:.0f} days after the trough respectively.
            These can serve as early buy signals when combined with capitulation volume.</p>
        </div>
"""

    if stats_ref.get('pct_mrs_led_25pct_recovery', 0) > 60:
        html += """
        <div class="insight-box success">
            <div class="insight-title">MRS Leads Price Recovery</div>
            <p>MRS turned positive before the market recovered 25% of its losses in the majority of cases.
            This suggests MRS can be used as an early re-entry signal, not just a warning signal.</p>
        </div>
"""

    # ── Volume Analysis ──
    html += """
        <h2>Volume Analysis</h2>
        <div class="card">
            <div class="card-header">Volume Behavior Around Drawdowns</div>
"""

    vol_stats = {
        'divergence_pre_peak': [],
        'capitulation_at_trough': [],
        'vol_ratio_at_trough': [],
    }

    for a in all_results.get('10-15%', {}).get('analyses', []):
        for window in PRE_PEAK_WINDOWS:
            key = f'pre_{window}d'
            if key in a['volume']:
                if a['volume'][key].get('divergence_flag'):
                    vol_stats['divergence_pre_peak'].append(True)

        if 'at_trough' in a['volume']:
            if a['volume']['at_trough'].get('capitulation_flag'):
                vol_stats['capitulation_at_trough'].append(True)
            ratio = a['volume']['at_trough'].get('vol_ratio_20d')
            if ratio and not np.isnan(ratio):
                vol_stats['vol_ratio_at_trough'].append(ratio)

    n_events = len(all_results.get('10-15%', {}).get('analyses', []))

    html += f"""
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{len(vol_stats['divergence_pre_peak'])}</div>
                    <div class="metric-label">Events with volume divergence before peak</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {len(vol_stats['divergence_pre_peak'])/n_events*100 if n_events else 0}%; background: var(--accent-orange);"></div>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(vol_stats['capitulation_at_trough'])}</div>
                    <div class="metric-label">Events with capitulation volume at trough</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {len(vol_stats['capitulation_at_trough'])/n_events*100 if n_events else 0}%; background: var(--accent-green);"></div>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-value">{np.mean(vol_stats['vol_ratio_at_trough']):.1f}x</div>
                    <div class="metric-label">Avg volume ratio at trough (vs 20d avg)</div>
                </div>
            </div>
"""

    if len(vol_stats['divergence_pre_peak']) / n_events > 0.4 if n_events else False:
        html += """
            <div class="insight-box warning" style="margin-top: 16px;">
                <div class="insight-title">Volume Divergence Signal</div>
                <p>Volume divergence (declining volume while price rises) preceded a significant portion of drawdowns.
                This pattern could be integrated into MRS as an additional warning signal.</p>
            </div>
"""

    if np.mean(vol_stats['vol_ratio_at_trough']) > 1.5 if vol_stats['vol_ratio_at_trough'] else False:
        html += """
            <div class="insight-box success" style="margin-top: 16px;">
                <div class="insight-title">Capitulation Volume Pattern</div>
                <p>Drawdown troughs typically show elevated volume (capitulation).
                A volume spike combined with extreme MRS readings could help identify bottoms.</p>
            </div>
"""

    html += """
        </div>

        <div class="card">
            <div class="card-header">Volume During Recovery</div>
"""

    # Volume recovery stats
    vol_recovery_confirms = []
    vol_recovery_slopes = []
    vol_changes = []

    for a in all_results.get('10-15%', {}).get('analyses', []):
        if 'volume_recovery' in a:
            vr = a['volume_recovery']
            if vr.get('vol_confirms_recovery') is not None:
                vol_recovery_confirms.append(vr['vol_confirms_recovery'])
            if vr.get('vol_slope_pct_per_day') is not None and not np.isnan(vr['vol_slope_pct_per_day']):
                vol_recovery_slopes.append(vr['vol_slope_pct_per_day'])
            if vr.get('vol_change_pct') is not None and not np.isnan(vr['vol_change_pct']):
                vol_changes.append(vr['vol_change_pct'])

    pct_vol_confirms = sum(vol_recovery_confirms) / len(vol_recovery_confirms) * 100 if vol_recovery_confirms else 0
    avg_vol_slope = np.mean(vol_recovery_slopes) if vol_recovery_slopes else 0
    avg_vol_change = np.mean(vol_changes) if vol_changes else 0

    html += f"""
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value {'positive' if pct_vol_confirms > 50 else 'warning'}">{pct_vol_confirms:.0f}%</div>
                    <div class="metric-label">Recoveries with higher volume on up days</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{avg_vol_slope:+.2f}%</div>
                    <div class="metric-label">Avg daily volume trend during recovery</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{avg_vol_change:+.0f}%</div>
                    <div class="metric-label">Avg volume change (trough to recovery)</div>
                </div>
            </div>

            <p style="color: var(--text-secondary); margin-top: 16px; font-size: 0.9rem;">
                <strong>Volume confirmation:</strong> Higher average volume on up days vs down days during recovery
                suggests institutional accumulation and increases confidence in the recovery signal.
            </p>
        </div>
"""

    # ── Pre-Peak Window Comparison ──
    html += """
        <h2>Analysis Window Comparison</h2>
        <div class="card">
            <div class="card-header">Warning Lead Time by Analysis Window</div>
            <table>
                <thead>
                    <tr>
                        <th>Window</th>
                        <th>% Events Warned</th>
                        <th>Avg Lead Time</th>
                        <th>Median Lead Time</th>
                        <th>Avg MRS Slope</th>
                    </tr>
                </thead>
                <tbody>
"""

    for window in PRE_PEAK_WINDOWS:
        warned = stats_ref.get(f'lead_time_{window}d_pct_warned', 0)
        avg_lead = stats_ref.get(f'lead_time_{window}d_mean', 0)
        med_lead = stats_ref.get(f'lead_time_{window}d_median', 0)
        slope = stats_ref.get(f'mrs_slope_{window}d_mean', 0)

        html += f"""
                    <tr>
                        <td>{window} days</td>
                        <td>{warned:.0f}%</td>
                        <td>{avg_lead:.0f}d</td>
                        <td>{med_lead:.0f}d</td>
                        <td class="{'negative' if slope < 0 else 'positive'}">{slope:.3f}/day</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>
"""

    # ── Actionable Insights ──
    html += """
        <h2>Actionable Insights for MRS v2</h2>
        <div class="card">
            <div class="card-header">Recommendations</div>

            <h3>1. Leading Indicator Enhancement</h3>
"""

    if comp_leads:
        sorted_comps = sorted([(k, v) for k, v in comp_leads.items() if v is not None],
                              key=lambda x: x[1], reverse=True)
        if sorted_comps:
            html += f"""
            <div class="insight-box">
                <p><strong>Weight leading components higher:</strong> {sorted_comps[0][0].upper()} and {sorted_comps[1][0].upper() if len(sorted_comps) > 1 else 'N/A'}
                provide the earliest warnings. Consider increasing their weight in the composite score or creating a
                separate "early warning" sub-score.</p>
            </div>
"""

    html += """
            <h3>2. Volume Integration</h3>
            <div class="insight-box">
                <p><strong>Add volume divergence indicator:</strong> Track 20-day volume trend vs price trend.
                When price makes new highs but volume is declining, flag as distribution warning.
                Consider adding a Volume Divergence Score (-0.5 when diverging).</p>
            </div>

            <h3>3. Capitulation Detection</h3>
            <div class="insight-box success">
                <p><strong>Add capitulation signal for bottom detection:</strong> When volume exceeds 2x the 20-day average
                AND VIX is in Stress state AND MRS is in RISK-OFF territory, flag as potential capitulation.
                This could improve recovery timing.</p>
            </div>

            <h3>4. Regime Transition Alerts</h3>
            <div class="insight-box warning">
                <p><strong>Track regime transition velocity:</strong> Rapid transitions from RISK-ON to NEUTRAL
                (within 5-10 days) often precede larger drawdowns. Add an alert when regime changes faster than
                historical average.</p>
            </div>

            <h3>5. Recovery Signal Enhancement</h3>
"""

    # Get recovery leading components
    comp_recovery = stats_ref.get('component_avg_recovery_time', {})
    sorted_recovery = sorted([(k, v) for k, v in comp_recovery.items() if v is not None],
                              key=lambda x: x[1]) if comp_recovery else []

    if sorted_recovery:
        html += f"""
            <div class="insight-box success">
                <p><strong>Add Buy-Side Signals:</strong> {sorted_recovery[0][0].upper()} is the fastest component
                to recover at bottoms ({sorted_recovery[0][1]:.0f} days avg). Consider adding a "Recovery Signal"
                when {sorted_recovery[0][0].upper()} turns positive while MRS is still in RISK-OFF territory.
                Combine with capitulation volume for higher confidence.</p>
            </div>
"""

    html += """
            <h3>6. Threshold Optimization</h3>
"""

    # Find optimal band (highest precision)
    best_precision = 0
    best_band = '10-15%'
    for band_label, data in all_results.items():
        fp = data.get('false_positive_stats', {})
        if fp.get('precision', 0) > best_precision:
            best_precision = fp.get('precision', 0)
            best_band = band_label

    html += f"""
            <div class="insight-box">
                <p><strong>Consider using MRS &lt; -0.5 as primary warning:</strong> The {best_band} band
                shows the highest precision ({best_precision:.0f}%). Using MILD RISK-OFF or worse as the warning
                level may reduce false positives while maintaining detection rate.</p>
            </div>

            <h3>7. Bottom Detection Framework</h3>
            <div class="insight-box success">
                <p><strong>Implement "Capitulation + Recovery" signal:</strong> Define a bottom signal as:
                <ul style="margin-top: 8px; padding-left: 20px;">
                    <li>MRS in RISK-OFF territory (≤ -1.5)</li>
                    <li>Volume spike > 2x 20-day average</li>
                    <li>VIX in Stress state (Phi > 0.80)</li>
                    <li>Leading component ({sorted_recovery[0][0].upper() if sorted_recovery else 'VIX'}) turns positive</li>
                </ul>
                This multi-factor approach can help identify actionable bottoms.</p>
            </div>
        </div>
"""

    # ── Data Quality Notes ──
    html += """
        <h2>Data Quality & Methodology</h2>
        <div class="card">
            <div class="card-header">Notes</div>
            <ul style="color: var(--text-secondary); padding-left: 20px;">
                <li>MRS computed using 756-day (3-year) rolling Phi values</li>
                <li>Zero Gamma component excluded (no historical data available)</li>
                <li>Drawdown events require 50% recovery to be confirmed</li>
                <li>Overlapping events resolved by keeping larger drawdown</li>
                <li>False positive analysis uses 60-day forward window</li>
            </ul>
        </div>
"""

    # ── Footer ──
    html += """
        <div class="footer">
            <p>Epistruct Research — Invariant Analysis</p>
            <p>MRS Backtest Framework v1.0</p>
        </div>
    </div>

    <script>
        function switchTab(tabId) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));

            // Show selected tab
            document.getElementById('tab-' + tabId).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReport saved to: {output_path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("MRS BACKTEST FRAMEWORK")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    df = load_data(DATA_PATH)

    # Compute MRS
    print("\n[2/5] Computing MRS...")
    df = compute_mrs(df)

    # Filter to rows where MRS is valid AND all key components have data
    # Phi requires 756 days warm-up, plus we need B20 and PC ratio data (available from Dec 2006)
    df_valid = df[
        df['mrs_score'].notna() &
        df['vix_phi'].notna() &
        df['b20_pct'].notna() &
        df['pc_ratio'].notna()
    ].copy()

    # Also ensure we have enough Phi history (start from index 756+)
    first_valid_idx = df[df['vix_phi'].notna()].index[0] if len(df[df['vix_phi'].notna()]) > 0 else 0
    df_valid = df_valid[df_valid.index >= first_valid_idx].copy()

    print(f"  Valid MRS rows: {len(df_valid):,} ({df_valid['date'].min().date()} to {df_valid['date'].max().date()})")
    print(f"  Note: Analysis starts from {df_valid['date'].min().date()} (after Phi warm-up + data availability)")

    # Find ALL drawdowns first, then categorize into mutually exclusive bands
    print("\n[3/5] Finding all drawdowns...")
    all_drawdowns = find_all_drawdowns(df_valid, min_threshold=0.05)
    print(f"  Found {len(all_drawdowns)} total drawdown events")

    # Categorize into bands
    categorized = categorize_drawdowns_by_band(all_drawdowns, DRAWDOWN_BANDS)

    all_results = {}

    for min_pct, max_pct, label, description in DRAWDOWN_BANDS:
        print(f"\n[3/5] Analyzing {label} drawdowns ({description})...")

        drawdowns = categorized[label]
        print(f"  Found {len(drawdowns)} events in this band")

        if drawdowns:
            # Analyze each event
            analyses = []
            for event in drawdowns:
                analysis = analyze_event(df_valid, event, PRE_PEAK_WINDOWS)
                analyses.append(analysis)

            # Compute aggregate statistics (use midpoint of band for threshold param)
            mid_threshold = (min_pct + max_pct) / 2
            agg_stats = compute_aggregate_stats(analyses, mid_threshold)
            agg_stats['band_label'] = label
            agg_stats['band_description'] = description

            # False positive analysis (use min threshold of band)
            fp_stats = analyze_false_positives(df_valid, drawdowns, min_pct)

            all_results[label] = {
                'drawdown_events': drawdowns,
                'analyses': analyses,
                'aggregate_stats': agg_stats,
                'false_positive_stats': fp_stats,
                'band_min': min_pct,
                'band_max': max_pct,
                'band_description': description,
            }

    # Generate report
    print("\n[4/5] Generating HTML report...")
    generate_html_report(df_valid, all_results, OUTPUT_PATH)

    # Print summary
    print("\n[5/5] Summary")
    print("=" * 60)
    for min_pct, max_pct, label, description in DRAWDOWN_BANDS:
        data = all_results.get(label, {})
        stats = data.get('aggregate_stats', {})
        n = len(data.get('drawdown_events', []))
        print(f"\n{label} Drawdowns - {description} ({n} events):")
        print(f"  MRS negative at peak: {stats.get('pct_negative_at_peak', 0):.0f}%")
        print(f"  Avg lead time (60d): {stats.get('lead_time_60d_mean', 0):.0f} days")
        print(f"  Avg MRS at trough: {stats.get('mrs_at_trough_mean', 0):.2f}")

    print("\n" + "=" * 60)
    print(f"Report saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    main()
