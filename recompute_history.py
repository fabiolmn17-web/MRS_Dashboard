"""
recompute_history.py — Recompute mrs_history.csv with corrected Extension scoring
==================================================================================
Run once after updating score_extension() to take mom_phi.

This script:
1. Backs up the old mrs_history.csv
2. Recomputes all scores with the corrected logic
3. Validates that only the expected rows changed
4. Saves the corrected history
"""
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pipeline

HIST_PATH = Path(__file__).parent / 'mrs_history.csv'
BACKUP_PATH = Path(__file__).parent / 'mrs_history_pre_extfix.csv'


def main():
    print(f'\n=== Recompute MRS History — Extension Fix ===\n')
    print(f'Timestamp: {datetime.now().isoformat()}')

    # 1. Load and backup
    if not HIST_PATH.exists():
        print(f'ERROR: {HIST_PATH} not found')
        return

    print(f'Loading {HIST_PATH}...')
    old_hist = pipeline.load_history(HIST_PATH)
    print(f'  Rows: {len(old_hist)}')
    print(f'  Date range: {old_hist["date"].min().date()} to {old_hist["date"].max().date()}')

    # Backup
    print(f'\nBacking up to {BACKUP_PATH}...')
    shutil.copy(HIST_PATH, BACKUP_PATH)
    print('  Done.')

    # 2. Recompute scores
    print('\nRecomputing scores with corrected Extension logic...')
    hist = old_hist.copy()

    score_cols = ['vix_score', 'ext_score', 'mom_score', 'adl_score',
                  'b20_score', 'pc_score', 'skew_score', 'gamma_score']
    state_cols = ['vix_state', 'ext_state', 'mom_state', 'adl_state',
                  'b20_state', 'pc_state', 'skew_state', 'gamma_state']

    res = {c: [] for c in score_cols + state_cols + ['mrs_score']}

    for _, row in hist.iterrows():
        def g(c):
            v = row.get(c, np.nan)
            return np.nan if pd.isna(v) else float(v)

        vs, vst = pipeline.score_vix(g('vix_phi'))
        ms, mst = pipeline.score_momentum(g('mom_phi'))
        es, est = pipeline.score_extension(g('ext_phi'), g('mom_phi'))  # NEW: pass mom_phi
        as_, ast = pipeline.score_adl(g('adl_phi'))
        bs, bst = pipeline.score_b20(g('b20_phi'), g('adl_phi'))
        ps, pst = pipeline.score_pc(g('pc_ratio'), g('pc_sma10'))
        ss, sst = pipeline.score_skew(g('skew_phi'), g('pc_ratio'))
        gs, gst = pipeline.score_gamma(g('spx'), g('zero_gamma'))

        scores = [vs, es, ms, as_, bs, ps, ss, gs]
        states = [vst, est, mst, ast, bst, pst, sst, gst]
        mrs = round(sum(c for c in scores if not np.isnan(c)), 2)

        for col, val in zip(score_cols, scores):
            res[col].append(val)
        for col, val in zip(state_cols, states):
            res[col].append(val)
        res['mrs_score'].append(mrs)

    for col in score_cols + state_cols + ['mrs_score']:
        hist[col] = res[col]

    # 3. Validate changes
    print('\nValidating changes...')

    # Find rows where ext_score changed
    ext_changed = hist[hist['ext_score'] != old_hist['ext_score']].copy()
    ext_changed['old_ext_score'] = old_hist.loc[ext_changed.index, 'ext_score']
    ext_changed['delta'] = ext_changed['ext_score'] - ext_changed['old_ext_score']

    print(f'  Rows with changed ext_score: {len(ext_changed)}')

    if len(ext_changed) > 0:
        # All changed rows should have ext_phi > 0.70 AND mom_phi >= 0.30
        # (these are the Extended + non-Weak rows that now score 0 instead of -0.5)
        expected_changed = (
            (ext_changed['ext_phi'] > 0.70) &
            ((ext_changed['mom_phi'] >= 0.30) | ext_changed['mom_phi'].isna())
        )

        if expected_changed.all():
            print('  ✓ All changed rows have ext_phi > 0.70 and mom_phi >= 0.30 (or NaN)')
        else:
            unexpected = ext_changed[~expected_changed]
            print(f'  ✗ WARNING: {len(unexpected)} rows changed unexpectedly:')
            print(unexpected[['date', 'ext_phi', 'mom_phi', 'old_ext_score', 'ext_score']].head(10))

        # All deltas should be +0.5 (was -0.5, now 0)
        if (ext_changed['delta'] == 0.5).all():
            print('  ✓ All changes are +0.5 as expected')
        else:
            wrong_delta = ext_changed[ext_changed['delta'] != 0.5]
            print(f'  ✗ WARNING: {len(wrong_delta)} rows have unexpected delta:')
            print(wrong_delta[['date', 'old_ext_score', 'ext_score', 'delta']].head(10))

    # Check mrs_score changes
    mrs_changed = hist[hist['mrs_score'] != old_hist['mrs_score']]
    print(f'  Rows with changed mrs_score: {len(mrs_changed)}')

    # 4. Save
    print(f'\nSaving corrected history to {HIST_PATH}...')
    pipeline.save_history(hist, HIST_PATH)

    # Summary
    last = hist.iloc[-1]
    print(f'\n{"="*60}')
    print(f'Recompute complete.')
    print(f'  Latest date: {last["date"].date()}')
    print(f'  Latest MRS:  {last["mrs_score"]:+.2f} ({pipeline.regime_label(last["mrs_score"])})')
    print(f'  Backup at:   {BACKUP_PATH}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
