"""
QQQ MRS Backtest
================
4-Component Market Regime Score for QQQ:
  VIX · Extension · Momentum · ADL (NASDAQ A/D)

Same framework as backtest_mrs.py — adapted for QQQ data format.
Missing vs SPY: B20%, PC Ratio, SKEW, Zero Gamma.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("/sessions/keen-zealous-ramanujan/mnt/uploads/NASDAQ_QQQ, 1D_05c32.csv")
OUTPUT_PATH = Path("/sessions/keen-zealous-ramanujan/mnt/Epistruct resesarch/MRS_WebApp/backtest_qqq_report.html")

DRAWDOWN_BANDS = [
    (0.05, 0.10, "5-10%",  "Minor Correction"),
    (0.10, 0.15, "10-15%", "Correction"),
    (0.15, 0.20, "15-20%", "Significant Correction"),
    (0.20, 1.00, "20%+",   "Bear Market"),
]
PRE_PEAK_WINDOWS  = [30, 60, 90]
VOLUME_WINDOWS    = [20, 50]
PHI_WINDOW        = 756
COMPONENTS        = ['vix', 'ext', 'mom', 'adl']

# ── LOAD ──────────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    df.columns = ['date','open','high','low','close','vix','volume','vol_ma','sma50','adl']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    for col in ['open','high','low','close','vix','volume','vol_ma','sma50','adl']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # ADL = 0 means no data (before NASDAQ A/D available on TV)
    df['adl'] = df['adl'].where(df['adl'] != 0, np.nan)
    df = df.dropna(subset=['close','vix']).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  ADL available from: {df[df['adl'].notna()]['date'].min().date()}")
    return df

# ── ROLLING PHI ───────────────────────────────────────────────────────────────
def rolling_phi(series, window=PHI_WINDOW):
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        if np.isnan(arr[i]): continue
        w = arr[i - window: i]
        valid = ~np.isnan(w)
        if valid.sum() > 0:
            out[i] = np.nansum(w[valid] < arr[i]) / valid.sum()
    return pd.Series(out, index=series.index)

# ── SCORING ───────────────────────────────────────────────────────────────────
def score_vix(phi):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return 1.0, 'Low'
    if phi < 0.60:    return 0.0, 'Mid'
    if phi < 0.80:    return -0.5, 'High'
    return -1.5, 'Stress'

def score_extension(phi, mom_phi=np.nan):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5, 'Compressed'
    if phi > 0.70:
        if not np.isnan(mom_phi) and mom_phi < 0.30:
            return -0.5, 'Extended+Weak'
        return 0.0, 'Extended'
    return 0.0, 'Normal'

def score_momentum(phi):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return 0.0, 'Normal'
    return 0.5, 'Strong'

def score_adl(phi):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return 0.0, 'Normal'
    return 0.0, 'Strong'

def regime_label(mrs):
    if mrs >= 1.5:  return 'RISK-ON'
    if mrs >= 0.5:  return 'MILD RISK-ON'
    if mrs >= -0.5: return 'NEUTRAL'
    if mrs >= -1.5: return 'MILD RISK-OFF'
    return 'RISK-OFF'

# ── COMPUTE MRS ───────────────────────────────────────────────────────────────
def compute_mrs(df):
    df = df.copy()
    spy   = df['close'].astype(float)
    vix   = df['vix'].astype(float)
    adl   = df['adl'].astype(float)
    sma50 = df['sma50'].astype(float)

    df['ext_raw']  = (spy - sma50) / sma50
    df['mom_raw']  = spy.pct_change(20)

    adl_prev = adl.shift(20)
    df['adl_roc20'] = np.where(adl_prev.abs() > 1e-9,
                                (adl - adl_prev) / adl_prev.abs(), np.nan)

    print("  Computing Phi values...")
    df['vix_phi'] = rolling_phi(vix,              PHI_WINDOW)
    df['ext_phi'] = rolling_phi(df['ext_raw'],    PHI_WINDOW)
    df['mom_phi'] = rolling_phi(df['mom_raw'],    PHI_WINDOW)
    df['adl_phi'] = rolling_phi(df['adl_roc20'],  PHI_WINDOW)

    print("  Scoring components...")
    vix_sc, ext_sc, mom_sc, adl_sc = [], [], [], []
    vix_st, ext_st, mom_st, adl_st = [], [], [], []
    mrs_scores = []

    for _, row in df.iterrows():
        vs, vst = score_vix(row['vix_phi'])
        ms, mst = score_momentum(row['mom_phi'])
        es, est = score_extension(row['ext_phi'], row['mom_phi'])
        as_, ast = score_adl(row['adl_phi'])
        mrs = round(sum(c for c in [vs, es, ms, as_] if not np.isnan(c)), 2)
        vix_sc.append(vs); vix_st.append(vst)
        ext_sc.append(es); ext_st.append(est)
        mom_sc.append(ms); mom_st.append(mst)
        adl_sc.append(as_); adl_st.append(ast)
        mrs_scores.append(mrs)

    df['vix_score']=vix_sc; df['vix_state']=vix_st
    df['ext_score']=ext_sc; df['ext_state']=ext_st
    df['mom_score']=mom_sc; df['mom_state']=mom_st
    df['adl_score']=adl_sc; df['adl_state']=adl_st
    df['mrs_score']=mrs_scores
    df['regime']=df['mrs_score'].apply(regime_label)

    for w in VOLUME_WINDOWS:
        df[f'vol_sma{w}']   = df['volume'].rolling(w, min_periods=1).mean()
        df[f'vol_ratio_{w}d'] = df['volume'] / df[f'vol_sma{w}']
    df['vol_phi_252'] = rolling_phi(df['volume'].astype(float), 252)

    print(f"  MRS range: {df['mrs_score'].min():.2f} to {df['mrs_score'].max():.2f}")
    return df

# ── DRAWDOWN DETECTION (unchanged from backtest_mrs.py) ──────────────────────
def find_all_drawdowns(df, min_threshold=0.05, min_recovery_pct=0.5):
    prices = df['close'].values
    dates  = df['date'].values
    n = len(prices)
    drawdowns = []
    i = 0
    while i < n - 20:
        peak_idx = i; peak_price = prices[i]
        j = i + 1
        while j < n:
            if prices[j] > peak_price:
                peak_idx = j; peak_price = prices[j]; j += 1
            elif j - peak_idx >= 20: break
            else: j += 1
        if peak_idx >= n - 20: break
        trough_idx = peak_idx + 1; trough_price = prices[trough_idx]
        for k in range(peak_idx + 1, min(n, peak_idx + 504)):
            if prices[k] < trough_price:
                trough_idx = k; trough_price = prices[k]
            drawdown = (peak_price - trough_price) / peak_price
            recovery = (prices[k] - trough_price) / (peak_price - trough_price) if peak_price > trough_price else 0
            if recovery >= min_recovery_pct and drawdown >= min_threshold:
                drawdowns.append({
                    'peak_idx': peak_idx, 'trough_idx': trough_idx,
                    'peak_date': pd.Timestamp(dates[peak_idx]),
                    'trough_date': pd.Timestamp(dates[trough_idx]),
                    'peak_price': peak_price, 'trough_price': trough_price,
                    'drawdown_pct': drawdown * 100,
                    'recovery_idx': k,
                    'recovery_date': pd.Timestamp(dates[k]),
                    'duration_days': (pd.Timestamp(dates[trough_idx]) - pd.Timestamp(dates[peak_idx])).days,
                })
                i = k; break
        else:
            i = peak_idx + 1; continue
        i += 1
    if len(drawdowns) > 1:
        filtered = []
        for dd in drawdowns:
            overlaps = False
            for ex in filtered:
                if dd['peak_date'] <= ex['recovery_date'] and dd['recovery_date'] >= ex['peak_date']:
                    if dd['drawdown_pct'] > ex['drawdown_pct']: filtered.remove(ex)
                    else: overlaps = True
                    break
            if not overlaps: filtered.append(dd)
        drawdowns = filtered
    return drawdowns

def categorize_drawdowns(drawdowns, bands):
    cat = {b[2]: [] for b in bands}
    for dd in drawdowns:
        p = dd['drawdown_pct'] / 100
        for mn, mx, lbl, desc in bands:
            if mn <= p < mx:
                d = dd.copy(); d['band_label']=lbl; d['band_description']=desc
                cat[lbl].append(d); break
    return cat

# ── EVENT ANALYSIS ────────────────────────────────────────────────────────────
def analyze_event(df, event):
    peak_idx   = event['peak_idx']
    trough_idx = event['trough_idx']
    result     = {'event': event, 'pre_peak': {}, 'at_peak': {}, 'at_trough': {},
                  'during_drawdown': {}, 'recovery': {}, 'volume': {}}

    def row_summary(row):
        d = {'mrs_score': row['mrs_score'], 'regime': row['regime'], 'vix': row['vix']}
        for c in COMPONENTS:
            d[f'{c}_score'] = row.get(f'{c}_score', np.nan)
            d[f'{c}_state'] = row.get(f'{c}_state', 'N/A')
        return d

    result['at_peak']   = row_summary(df.iloc[peak_idx])
    result['at_trough'] = row_summary(df.iloc[trough_idx])

    for window in PRE_PEAK_WINDOWS:
        start = max(0, peak_idx - window)
        pre   = df.iloc[start:peak_idx + 1].copy().reset_index(drop=True)
        if len(pre) < 5: continue
        neg_mask   = pre['mrs_score'] < 0
        ro_mask    = pre['mrs_score'] < -0.5
        x = np.arange(len(pre)); y = pre['mrs_score'].values
        valid = ~np.isnan(y)
        slope = np.polyfit(x[valid], y[valid], 1)[0] if valid.sum() > 2 else np.nan
        comp_warn = {}
        for c in COMPONENTS:
            col = f'{c}_score'
            if col in pre.columns:
                m = pre[col] < 0
                if m.any(): comp_warn[c] = len(pre) - 1 - m.idxmax()
        result['pre_peak'][f'{window}d'] = {
            'mrs_start': pre['mrs_score'].iloc[0],
            'mrs_end':   pre['mrs_score'].iloc[-1],
            'mrs_min':   pre['mrs_score'].min(),
            'mrs_slope': slope,
            'days_negative_before_peak':  len(pre)-1-neg_mask.idxmax() if neg_mask.any() else None,
            'days_risk_off_before_peak':  len(pre)-1-ro_mask.idxmax()  if ro_mask.any()  else None,
            'pct_days_negative': neg_mask.mean() * 100,
            'component_first_warning': comp_warn,
        }

    dd_df = df.iloc[peak_idx:trough_idx + 1]
    result['during_drawdown'] = {
        'mrs_min':        dd_df['mrs_score'].min(),
        'mrs_avg':        dd_df['mrs_score'].mean(),
        'pct_risk_off':   (dd_df['mrs_score'] < -1.5).mean() * 100,
        'vix_max':        dd_df['vix'].max(),
        'days':           len(dd_df),
    }

    rec_end = min(trough_idx + 60, len(df) - 1)
    rec_df  = df.iloc[trough_idx:rec_end + 1].copy().reset_index(drop=True)
    if len(rec_df) > 1:
        pos_mask  = rec_df['mrs_score'] > 0
        mild_mask = rec_df['mrs_score'] >= -0.5
        dp = pos_mask.idxmax() if pos_mask.any() else None
        dm = mild_mask.idxmax() if mild_mask.any() else None
        comp_rec = {}
        for c in COMPONENTS:
            col = f'{c}_score'
            if col in rec_df.columns:
                m = rec_df[col] > 0
                if m.any(): comp_rec[c] = m.idxmax()
        result['recovery'] = {
            'days_to_positive_mrs': dp,
            'days_to_mild_recovery': dm,
            'component_recovery_order': comp_rec,
            'recovery_sequence': sorted(comp_rec, key=comp_rec.get),
        }

    # Volume pre-peak
    for window in PRE_PEAK_WINDOWS:
        start = max(0, peak_idx - window)
        pre   = df.iloc[start:peak_idx + 1]
        if len(pre) < 5: continue
        pc = (pre['close'].iloc[-1] / pre['close'].iloc[0] - 1) * 100
        vc = (pre['volume'].iloc[-1] / pre['volume'].iloc[0] - 1) * 100
        result['volume'][f'pre_{window}d'] = {
            'price_pct_change': pc, 'vol_pct_change': vc,
            'divergence_flag': pc > 0 and vc < -10,
        }

    # Volume at trough
    tw = df.iloc[max(0, trough_idx-5):trough_idx+6]
    r20 = tw['vol_ratio_20d'].max() if 'vol_ratio_20d' in tw.columns else np.nan
    result['volume']['at_trough'] = {
        'vol_ratio_20d': r20,
        'capitulation_flag': r20 > 2.0 if not np.isnan(r20) else False,
    }
    return result

# ── AGGREGATE STATS ───────────────────────────────────────────────────────────
def aggregate_stats(analyses):
    if not analyses: return {}
    stats = {'n_events': len(analyses)}
    at_peak   = [a['at_peak']['mrs_score']   for a in analyses]
    at_trough = [a['at_trough']['mrs_score'] for a in analyses]
    stats['mrs_at_peak_mean']      = np.nanmean(at_peak)
    stats['mrs_at_peak_median']    = np.nanmedian(at_peak)
    stats['pct_negative_at_peak']  = sum(1 for m in at_peak if m < 0) / len(at_peak) * 100
    stats['pct_risk_off_at_peak']  = sum(1 for m in at_peak if m < -0.5) / len(at_peak) * 100
    stats['mrs_at_trough_mean']    = np.nanmean(at_trough)
    stats['mrs_at_trough_min']     = np.nanmin(at_trough)
    for w in PRE_PEAK_WINDOWS:
        key = f'{w}d'
        lts = [a['pre_peak'].get(key, {}).get('days_negative_before_peak') for a in analyses]
        lts = [l for l in lts if l is not None]
        if lts:
            stats[f'lead_time_{w}d_mean']    = np.mean(lts)
            stats[f'lead_time_{w}d_median']  = np.median(lts)
            stats[f'lead_time_{w}d_pct']     = len(lts) / len(analyses) * 100
    comp_leads = {c: [] for c in COMPONENTS}
    for a in analyses:
        for w in PRE_PEAK_WINDOWS:
            key = f'{w}d'
            for c, lt in a['pre_peak'].get(key, {}).get('component_first_warning', {}).items():
                comp_leads[c].append(lt)
    stats['component_avg_lead_time'] = {c: np.mean(v) if v else None for c, v in comp_leads.items()}
    rec_lead = [a['recovery'].get('days_to_positive_mrs') for a in analyses if a.get('recovery')]
    rec_lead = [r for r in rec_lead if r is not None]
    if rec_lead:
        stats['recovery_lead_mean']   = np.mean(rec_lead)
        stats['recovery_lead_median'] = np.median(rec_lead)
    cap = sum(1 for a in analyses if a['volume'].get('at_trough', {}).get('capitulation_flag'))
    stats['pct_capitulation_at_trough'] = cap / len(analyses) * 100
    div = sum(1 for a in analyses
              for w in PRE_PEAK_WINDOWS
              if a['volume'].get(f'pre_{w}d', {}).get('divergence_flag'))
    stats['pct_volume_divergence'] = min(div / len(analyses) * 100, 100)
    return stats

# ── FALSE POSITIVE ANALYSIS ───────────────────────────────────────────────────
def false_positives(df, drawdown_events, lookforward=60):
    neg_periods = []
    in_neg = False; start = None
    for i in range(len(df)):
        if df.iloc[i]['mrs_score'] < 0:
            if not in_neg: in_neg = True; start = i
        else:
            if in_neg: neg_periods.append((start, i-1)); in_neg = False
    if in_neg: neg_periods.append((start, len(df)-1))
    peaks = set(e['peak_idx'] for e in drawdown_events)
    tp = fp = 0
    for s, e in neg_periods:
        if any(s <= p <= e + lookforward for p in peaks): tp += 1
        else: fp += 1
    total = tp + fp
    return {'total': total, 'tp': tp, 'fp': fp,
            'precision': tp/total*100 if total else 0}

# ── PRINT SUMMARY ─────────────────────────────────────────────────────────────
def print_summary(all_results):
    print("\n" + "="*65)
    print("QQQ MRS BACKTEST — SUMMARY")
    print("="*65)
    for mn, mx, lbl, desc in DRAWDOWN_BANDS:
        data  = all_results.get(lbl, {})
        stats = data.get('agg', {})
        events= data.get('drawdown_events', [])
        fp    = data.get('fp', {})
        if not events: print(f"\n{lbl} ({desc}): no events"); continue
        print(f"\n{lbl} ({desc})  n={len(events)}")
        print(f"  MRS negative at peak:  {stats.get('pct_negative_at_peak',0):.0f}%")
        print(f"  MRS risk-off at peak:  {stats.get('pct_risk_off_at_peak',0):.0f}%")
        print(f"  Avg lead time (60d):   {stats.get('lead_time_60d_mean',0):.0f} days")
        print(f"  MRS at trough (avg):   {stats.get('mrs_at_trough_mean',0):.2f}")
        print(f"  Capitulation vol:      {stats.get('pct_capitulation_at_trough',0):.0f}%")
        print(f"  Signal precision:      {fp.get('precision',0):.0f}%  "
              f"(TP={fp.get('tp',0)} FP={fp.get('fp',0)})")
        cl = stats.get('component_avg_lead_time', {})
        cl_sorted = sorted([(k,v) for k,v in cl.items() if v], key=lambda x: x[1], reverse=True)
        print(f"  Component lead times:  {', '.join(f'{k.upper()}={v:.0f}d' for k,v in cl_sorted)}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print("Computing MRS...")
    df = compute_mrs(df)

    print("Detecting drawdowns...")
    all_dds   = find_all_drawdowns(df)
    cat_dds   = categorize_drawdowns(all_dds, DRAWDOWN_BANDS)

    all_results = {}
    for mn, mx, lbl, desc in DRAWDOWN_BANDS:
        events   = cat_dds.get(lbl, [])
        analyses = [analyze_event(df, e) for e in events]
        agg      = aggregate_stats(analyses)
        fp       = false_positives(df, events)
        all_results[lbl] = {
            'drawdown_events': events, 'analyses': analyses,
            'agg': agg, 'fp': fp,
        }
        print(f"  {lbl}: {len(events)} events")

    print_summary(all_results)

    # Save df with MRS scores for further use
    out_csv = Path("/sessions/keen-zealous-ramanujan/mnt/Epistruct resesarch/MRS_WebApp/qqq_mrs_history.csv")
    df[['date','close','vix','volume','adl',
        'ext_phi','mom_phi','vix_phi','adl_phi',
        'ext_score','mom_score','vix_score','adl_score',
        'ext_state','mom_state','vix_state','adl_state',
        'mrs_score','regime']].to_csv(out_csv, index=False)
    print(f"\nMRS history saved -> {out_csv.name}")
    print("Done.")

if __name__ == '__main__':
    main()
