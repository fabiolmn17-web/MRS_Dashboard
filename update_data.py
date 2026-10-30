"""
update_data.py — Daily batch update (run by GitHub Actions at 5:30 PM ET)
=========================================================================
Usage:
    python update_data.py
Reads mrs_history.csv, auto-fetches all inputs, rescores, and saves.
No Excel, no manual entry required.
"""
import sys
import datetime
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path

import pipeline
import auto_fetch
import risk_panel

HIST_PATH = Path(__file__).parent / 'mrs_history.csv'

# ── NYSE holiday calendar (uses pandas built-in — no extra packages needed) ───
_CAL      = USFederalHolidayCalendar()
_GF_DATES = {  # Good Friday (NYSE-specific, not a federal holiday)
    pd.Timestamp('2024-03-29'), pd.Timestamp('2025-04-18'),
    pd.Timestamp('2026-04-03'), pd.Timestamp('2027-03-26'),
    pd.Timestamp('2028-04-14'), pd.Timestamp('2029-03-30'),
    pd.Timestamp('2030-04-19'),
}

def is_trading_day(dt: pd.Timestamp) -> bool:
    """Return True if dt is a NYSE trading session."""
    if dt.weekday() >= 5:          # Saturday / Sunday
        return False
    fed_holidays = _CAL.holidays(start=dt, end=dt)
    if len(fed_holidays) > 0:      # Federal holiday (incl. Juneteenth)
        return False
    if dt in _GF_DATES:            # Good Friday
        return False
    return True


def main():
    today_date = pd.Timestamp(datetime.date.today()).normalize()
    print(f'\n=== MRS Daily Update — {today_date.date()} ===\n')

    # ── Guard: skip non-trading days ─────────────────────────────────────────
    if not is_trading_day(today_date):
        print(f'  {today_date.date()} is not a NYSE trading day — nothing to do.')
        print('\n=== Skipped ===')
        sys.exit(0)

    if not HIST_PATH.exists():
        print(f'[ERROR] History file not found at {HIST_PATH}')
        print('  Place mrs_history.csv in the same folder as this script.')
        sys.exit(1)

    # 1. Load existing history
    hist = pipeline.load_history(HIST_PATH)
    print(f'  History: {len(hist)} rows | last date: {hist["date"].max().date()}')

    # 1b. Drop today's row if SPX is missing (mid-day or stale run)
    #     so update_history re-fetches it fresh after market close.
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

    # 3b. v5 forward log (risk panel lines for the last session). Never blocks the update.
    try:
        ind_p = Path(__file__).parent / 'StockScanner' / 'output' / 'industry_rs.csv'
        etfs = pd.read_csv(ind_p)['ETF'].dropna().unique().tolist() if ind_p.exists() else []
        spy, v3, closes = risk_panel.fetch_inputs(etfs)
        snap = risk_panel.build_snapshot(hist, spy, v3, closes, pipeline.regime_label)
        row = risk_panel.log_row(snap)
        mask = pd.to_datetime(hist['date']).dt.normalize() == pd.Timestamp(snap['date']).normalize()
        if mask.any():
            for k, v in row.items():
                if k not in hist.columns:
                    hist[k] = np.nan
                if isinstance(v, str):
                    hist[k] = hist[k].astype(object)
                hist.loc[mask, k] = v
            print(f"  Forward log {pd.Timestamp(snap['date']).date()}: pedal {row['pedal']} | cell {row['cell']} | "
                  f"corr {row['ind_corr63']} | n_eff {row['n_eff5']} | put cost {row['put3m_cost']} | invested {row['vt_invested']}")
        else:
            print(f"  Forward log skipped: SPY session {pd.Timestamp(snap['date']).date()} not in history")
    except Exception as e:
        print(f'  Forward log skipped: {e}')

    # 4. Save
    pipeline.save_history(hist, HIST_PATH)
    print(f'\n  Saved {len(hist)} rows to {HIST_PATH.name}')
    print(f'  Latest MRS: {hist["mrs_score"].iloc[-1]:+.2f} — {pipeline.regime_label(hist["mrs_score"].iloc[-1])}')
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
