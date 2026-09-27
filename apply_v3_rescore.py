"""
apply_v3_rescore.py — One-off migration to MRS v3.0 scoring (Report v7.0)
=========================================================================
1. Backs up mrs_history.csv -> mrs_history_pre_v3.csv
2. Corrects raw inputs against the calibration sources Fabio exported from
   TradingView (Data/USI_PC_1D_source.csv, Data/CBOE_SKEW_1D_source.csv):
     - pc_ratio: replaced with the USI:PC daily close wherever it differs by
       more than 0.001 (drift found Apr-Sep 2026: SPY-options-proxy fallback
       values, stale carry-forward, one-day-lagged entries — Report v7.0 §2)
     - skew: replaced where it differs by more than 0.01 (two rows)
     - both: filled from the source where the history has no value (SKEW had
       420 blank rows, mostly 2005-06, which shortened the early Phi window)
3. Rescores the full history with the v3.0 pipeline (new COMPONENT_WEIGHTS and
   score_pc zone scores) via pipeline.score_dataframe — the single source of
   truth used by the daily update as well.
Run once:  python apply_v3_rescore.py
"""
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pipeline

ROOT = Path(__file__).parent
HIST = ROOT / 'mrs_history.csv'
BACKUP = ROOT / 'mrs_history_pre_v3.csv'
PC_SRC = ROOT / 'Data' / 'USI_PC_1D_source.csv'
SKEW_SRC = ROOT / 'Data' / 'CBOE_SKEW_1D_source.csv'


def main():
    hist = pipeline.load_history(HIST)
    if not BACKUP.exists():
        shutil.copy(HIST, BACKUP)
    print(f'Backed up {len(hist)} rows to {BACKUP.name}')
    old_score = hist.set_index('date')['mrs_score'].copy()

    pc = pd.read_csv(PC_SRC, usecols=[0, 4])
    pc.columns = ['date', 'pc_src']
    pc['date'] = pd.to_datetime(pc['date'])
    sk = pd.read_csv(SKEW_SRC)
    sk.columns = ['date', 'skew_src']
    sk['date'] = pd.to_datetime(sk['date'])

    h = hist.merge(pc, on='date', how='left').merge(sk, on='date', how='left')
    pc_fix = h['pc_src'].notna() & (h['pc_ratio'].isna() | ((h['pc_ratio'] - h['pc_src']).abs() > 0.001))
    sk_fix = h['skew_src'].notna() & (h['skew'].isna() | ((h['skew'] - h['skew_src']).abs() > 0.01))
    print(f'pc_ratio corrected on {pc_fix.sum()} rows; skew corrected on {sk_fix.sum()} rows')
    h.loc[pc_fix, 'pc_ratio'] = h.loc[pc_fix, 'pc_src']
    h.loc[sk_fix, 'skew'] = h.loc[sk_fix, 'skew_src']
    h = h.drop(columns=['pc_src', 'skew_src'])

    h = pipeline.score_dataframe(h, verbose=True)
    new_score = h.set_index('date')['mrs_score']
    both = pd.concat([old_score.rename('old'), new_score.rename('new')], axis=1).dropna()
    print(f'Rescored. corr(old, new) = {both["old"].corr(both["new"]):.3f}; '
          f'latest {new_score.index[-1].date()}: {both["old"].iloc[-1]:+.2f} -> {both["new"].iloc[-1]:+.2f}')
    pipeline.save_history(h, HIST)
    print(f'Saved {HIST.name}')


if __name__ == '__main__':
    main()
