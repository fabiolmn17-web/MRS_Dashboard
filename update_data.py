"""
update_data.py — Daily batch update (run by GitHub Actions at 5:30 PM ET)
=========================================================================
Usage:
    python update_data.py

Reads mrs_history.csv, auto-fetches all inputs, rescores, and saves.
No Excel, no manual entry required.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import pipeline
import auto_fetch

HIST_PATH = Path(__file__).parent / 'mrs_history.csv'


def main():
    print(f'\n=== MRS Daily Update — {__import__("datetime").date.today()} ===\n')

    if not HIST_PATH.exists():
        print(f'[ERROR] History file not found at {HIST_PATH}')
        print('  Place mrs_history.csv in the same folder as this script.')
        sys.exit(1)

    # 1. Load existing history
    hist = pipeline.load_history(HIST_PATH)
    print(f'  History: {len(hist)} rows | last date: {hist["date"].max().date()}')

    # 1b. Drop any trailing rows with missing SPX close prices (caused by mid-day runs)
    #     so that update_history re-fetches them fresh after market close.
    dropped = 0
    while len(hist) > 0:
        last_spx = hist.iloc[-1].get('spx', np.nan)
        if pd.isna(last_spx):
            drop_date = hist.iloc[-1]['date']
            print(f'  Dropping incomplete row {pd.Timestamp(drop_date).date()} (NaN SPX) — will re-fetch at close')
            hist = hist.iloc[:-1].reset_index(drop=True)
            dropped += 1
        else:
            break
    if dropped:
        print(f'  Dropped {dropped} incomplete row(s). History now ends: {hist["date"].max().date()}')

    # 2. Auto-fetch all manual inputs
    inp_map = auto_fetch.build_inp_map(hist)

    # 3. Update history (append + rescore)
    hist = pipeline.update_history(hist, inp_map)

    # 4. Save
    pipeline.save_history(hist, HIST_PATH)
    print(f'\n  Saved {len(hist)} rows to {HIST_PATH.name}')
    print(f'  Latest MRS: {hist["mrs_score"].iloc[-1]:+.2f} — {pipeline.regime_label(hist["mrs_score"].iloc[-1])}')
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
