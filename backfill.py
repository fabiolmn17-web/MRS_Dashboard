"""
backfill.py — Detect and fill gaps in mrs_history.csv
=======================================================
Drops NaN-SPX trailing rows, then lets pipeline.update_history()
re-fetch all missing trading days automatically via yfinance.
"""

import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import pipeline

HIST_PATH = Path(__file__).parent / 'mrs_history.csv'


def main():
    print(f'\n=== MRS Backfill — {datetime.date.today()} ===\n')

    hist  = pipeline.load_history(HIST_PATH)
    today = pd.Timestamp(datetime.date.today()).normalize()

    # ── Find last date with a VALID SPX close ─────────────────────────────────
    spx_valid = hist.dropna(subset=['spx'])
    if len(spx_valid) == 0:
        print('  No valid SPX data in history — cannot backfill.')
        return

    last_valid_date = spx_valid['date'].max()
    last_csv_date   = hist['date'].max()

    print(f'  Last row in CSV      : {last_csv_date.date()}')
    print(f'  Last valid SPX close : {last_valid_date.date()}')
    print(f'  Today                : {today.date()}')

    if last_valid_date >= today - pd.Timedelta(days=1):
        print('  History is up to date — nothing to backfill.')
        return

    # ── Drop all rows after last valid date (NaN holiday/gap rows) ───────────
    hist = hist[hist['date'] <= last_valid_date].reset_index(drop=True)
    print(f'  Trimmed history to {len(hist)} rows (up to {last_valid_date.date()})')

    # ── Carry-forward last known manual inputs ────────────────────────────────
    def last_val(col):
        if col in hist.columns and hist[col].notna().any():
            return float(hist[col].dropna().iloc[-1])
        return np.nan

    manual = {
        'adl_level':  last_val('adl_level'),
        'b20_pct':    last_val('b20_pct'),
        'zero_gamma': last_val('zero_gamma'),
        'pc_ratio':   last_val('pc_ratio'),
        'skew':       np.nan,   # pipeline fetches ^SKEW from yfinance
    }

    print(f'\n  Carry-forward inputs:')
    for k, v in manual.items():
        print(f'    {k}: {v}')

    # ── Build inp_map: anchor at last valid date so pipeline carry-forwards ───
    # pipeline._get_manual() picks the most recent prior entry for any new date
    inp_map = {last_valid_date: manual}

    # ── Run pipeline — it auto-detects and fetches all missing trading days ───
    print(f'\n  Running pipeline.update_history() ...')
    hist = pipeline.update_history(hist, inp_map)

    # ── Save ──────────────────────────────────────────────────────────────────
    pipeline.save_history(hist, HIST_PATH)
    print(f'\n  Saved {len(hist)} rows → {HIST_PATH.name}')
    print(f'  Latest MRS : {hist["mrs_score"].iloc[-1]:+.2f} — '
          f'{pipeline.regime_label(hist["mrs_score"].iloc[-1])}')
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
