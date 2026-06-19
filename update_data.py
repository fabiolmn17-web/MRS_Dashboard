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

    # 1b. Drop today's row if SPX is missing (mid-day or holiday run)
    #     so update_history re-fetches it fresh after market close.
    import datetime
    today_date = pd.Timestamp(datetime.date.today()).normalize()
    if len(hist) > 0:
        last_row  = hist.iloc[-1]
        last_date = pd.Timestamp(last_row['date']).normalize()
        if last_date == today_date and pd.isna(last_row.get('spx', np.nan)):
            print(f'  Dropping today\'s incomplete row {today_date.date()} (NaN SPX) — will re-fetch at close')
            hist = hist.iloc[:-1].reset_index(drop=True)

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
