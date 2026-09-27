"""
apply_v4_rescore.py — One-off migration to MRS v4.0 scoring (v8 dispersion re-test)
====================================================================================
1. Backs up mrs_history.csv -> mrs_history_pre_v4.csv
2. Rescores the full history with the v4.0 pipeline (VIX anchor + context block,
   new state scores, Volume Divergence at weight 0) via pipeline.score_dataframe,
   the single source of truth also used by the daily update.
Raw inputs are not touched (they were corrected in the v3.0 migration).
Run once:  python apply_v4_rescore.py
"""
import shutil
from pathlib import Path
import pandas as pd
import pipeline

ROOT = Path(__file__).parent
HIST = ROOT / 'mrs_history.csv'
BACKUP = ROOT / 'mrs_history_pre_v4.csv'


def main():
    hist = pipeline.load_history(HIST)
    if not BACKUP.exists():
        shutil.copy(HIST, BACKUP)
    print(f'Backed up {len(hist)} rows to {BACKUP.name}')
    old = hist.set_index('date')['mrs_score'].copy()
    old_band = old.apply(pipeline.regime_label)

    h = pipeline.score_dataframe(hist, verbose=True)
    new = h.set_index('date')['mrs_score']
    new_band = new.apply(pipeline.regime_label)
    both = pd.concat([old.rename('old'), new.rename('new')], axis=1).dropna()
    print(f'Rescored. corr(old, new) = {both["old"].corr(both["new"]):.3f}; same band on '
          f'{(old_band == new_band).mean():.0%} of sessions')
    print(f'latest {new.index[-1].date()}: {old.iloc[-1]:+.2f} ({old_band.iloc[-1]}) -> '
          f'{new.iloc[-1]:+.2f} ({new_band.iloc[-1]})')
    pipeline.save_history(h, HIST)
    print(f'Saved {HIST.name}')


if __name__ == '__main__':
    main()
