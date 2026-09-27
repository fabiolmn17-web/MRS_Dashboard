"""
risk_dial.py — MRS as a dispersion ("risk dial") signal  (Report v7.0)
======================================================================
Report v7.0 found that the MRS band does not reliably predict the DIRECTION
of the next 5-60 sessions, but it does predict their SPREAD: RISK-OFF is
followed by larger adverse AND larger favorable excursions than RISK-ON
(27 of 35 pre-specified tests survive Benjamini-Hochberg FDR, q<0.05).

This module turns that finding into descriptive lookup tables for the
dashboard, computed directly from mrs_history.csv so they refresh as history
grows. Numbers here are descriptive (all days, overlapping windows); the
formal inference (non-overlapping samples, FDR, bootstrap) is in the report.

Definitions (long position entered at the close of day t, horizon h):
  MAE = worst close-to-close drawdown from entry within t+1..t+h
  MFE = best close-to-close gain from entry within t+1..t+h
  P(adverse) = P(MAE <= -X_h),  P(favorable) = P(MFE >= +X_h)
  X_h = 2% / 3% / 4% / 5% / 6% / 8% / 10% at 5 / 10 / 15 / 20 / 30 / 45 / 60 sessions
Price = SPY close (full history in mrs_history.csv).
"""
import numpy as np
import pandas as pd

import pipeline

BANDS = ['RISK-ON', 'MILD RISK-ON', 'NEUTRAL', 'MILD RISK-OFF', 'RISK-OFF']
HORIZONS = [5, 10, 20, 30, 60]
THRESH = {5: 0.02, 10: 0.03, 15: 0.04, 20: 0.05, 30: 0.06, 45: 0.08, 60: 0.10}
CORE_PHI = ['vix_phi', 'ext_phi', 'mom_phi', 'skew_phi', 'adl_phi', 'b20_phi']

# Plain-language reading per band (Report v7.0 §6). Descriptive guidance for a
# 5-60 session swing book — not a tested trading rule (the sizing backtest is
# an open item).
BAND_GUIDANCE = {
    'RISK-ON': ('Calm, narrow range',
                'Smallest swings in both directions. Drawdowns on new longs are '
                'shallowest here, but so is the upside run. A calm base can break: '
                'volatility tends to rise from here in relative terms.'),
    'MILD RISK-ON': ('Fairly calm',
                     'Swings a little wider than RISK-ON; still a low-dispersion regime.'),
    'NEUTRAL': ('Average dispersion',
                'Close to the all-days baseline. No dispersion edge either way.'),
    'MILD RISK-OFF': ('Widening range',
                      'Favorable excursions already exceed the all-days baseline; '
                      'adverse excursions are close to baseline.'),
    'RISK-OFF': ('Wide, two-sided',
                 'Largest swings both ways: a large adverse move is about 3x as likely '
                 'as in RISK-ON, and a large favorable move about 4x. Keep initial risk '
                 'small and cut quickly if a new long does not work; give a long that is '
                 'working more room, because the upside tail is widest here.'),
}


def study_frame(hist: pd.DataFrame) -> pd.DataFrame:
    """Rows from the first date all six Phi inputs are burned in (2008-05 onward)."""
    h = hist.sort_values('date').reset_index(drop=True).copy()
    have = [c for c in CORE_PHI if c in h.columns]
    first_ok = h.dropna(subset=have).index.min() if have else 0
    if pd.isna(first_ok):
        first_ok = 0
    h = h.loc[first_ok:].reset_index(drop=True)
    h['price'] = h['spy'].astype(float).ffill()
    h = h.dropna(subset=['mrs_score', 'price']).reset_index(drop=True)
    h['band'] = h['mrs_score'].apply(pipeline.regime_label)
    return h


def _excursions(price: np.ndarray, h: int):
    """Vectorised MAE / MFE for every start index (NaN where t+h is not yet known)."""
    n = len(price)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    if n <= h:
        return mae, mfe
    # windows of the next h closes for each start t: shape (n-h, h)
    idx = np.arange(n - h)[:, None] + np.arange(1, h + 1)[None, :]
    path = price[idx] / price[:n - h, None] - 1.0
    mae[:n - h] = path.min(axis=1)
    mfe[:n - h] = path.max(axis=1)
    return mae, mfe


def band_profile(hist: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Per band (and ALL days) × horizon: n, P(adverse), P(favorable), MAE/MFE quantiles."""
    f = study_frame(hist)
    price = f['price'].values
    out = []
    for h in horizons:
        mae, mfe = _excursions(price, h)
        thr = THRESH[h]
        g = pd.DataFrame({'band': f['band'].values, 'mae': mae, 'mfe': mfe}).dropna()
        groups = [('ALL', g)] + [(b, g[g['band'] == b]) for b in BANDS]
        for name, sub in groups:
            if len(sub) == 0:
                continue
            out.append(dict(
                band=name, horizon=h, n=len(sub), thresh=thr,
                p_adverse=(sub['mae'] <= -thr).mean(),
                p_favorable=(sub['mfe'] >= thr).mean(),
                mae_median=sub['mae'].median(),
                mae_p10=sub['mae'].quantile(0.10),
                mfe_median=sub['mfe'].median(),
                mfe_p90=sub['mfe'].quantile(0.90),
            ))
    return pd.DataFrame(out)


def score_context(hist: pd.DataFrame, score: float) -> dict:
    """Percentile of today's score within its own history + deep-state flags."""
    f = study_frame(hist)
    s = f['mrs_score'].astype(float)
    p10, p20, p90 = s.quantile(0.10), s.quantile(0.20), s.quantile(0.90)
    pct = float((s < score).mean() * 100) if len(s) else np.nan
    if score <= p10:
        deep = 'DEEP RISK-OFF'
    elif score >= p90:
        deep = 'DEEP RISK-ON'
    elif score <= p20:
        deep = 'LOWER QUINTILE'
    else:
        deep = ''
    band_share = f['band'].value_counts(normalize=True).reindex(BANDS).fillna(0).to_dict()
    return dict(percentile=pct, p10=p10, p20=p20, p90=p90, deep_state=deep,
                band_share=band_share, start=f['date'].min(), n=len(f))


def pc_stale_sessions(hist: pd.DataFrame) -> int:
    """Consecutive most-recent sessions carrying an identical pc_ratio (stale-input check)."""
    s = hist.sort_values('date')['pc_ratio'].dropna()
    if s.empty:
        return 0
    last = s.iloc[-1]
    run = 0
    for v in reversed(s.values):
        if abs(v - last) < 1e-9:
            run += 1
        else:
            break
    return run
