"""
backfill.py — Detect and fill gaps in mrs_history.csv
=======================================================
Fetches historical price data for any missing trading days and rescores.
Run via the manual GitHub Actions workflow 'MRS Backfill'.
"""

import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

import pipeline

HIST_PATH = Path(__file__).parent / 'mrs_history.csv'

US_HOLIDAYS_2025_2026 = pd.to_datetime([
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18',
    '2025-05-26','2025-06-19','2025-07-04','2025-09-01',
    '2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03',
    '2026-05-25','2026-06-19','2026-07-03','2026-09-07',
    '2026-11-26','2026-12-25',
])

def get_missing_trading_days(last_date, today):
    bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=today - pd.Timedelta(days=1))
    return [d for d in bdays if d.normalize() not in US_HOLIDAYS_2025_2026]

def get_close(df, day):
    day = pd.Timestamp(day).normalize()
    idx = df.index.normalize()
    matches = df[idx == day]
    if len(matches) == 0:
        return np.nan
    val = matches['Close'].iloc[0]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val)

def main():
    print(f'\n=== MRS Backfill — {datetime.date.today()} ===\n')

    hist = pipeline.load_history(HIST_PATH)
    last_date = hist['date'].max()
    today = pd.Timestamp(datetime.date.today()).normalize()

    print(f'  History ends : {last_date.date()}')
    print(f'  Today        : {today.date()}')

    missing = get_missing_trading_days(last_date, today)

    if not missing:
        print('  No missing trading days — nothing to do.')
        return

    print(f'  Missing days : {[d.date() for d in missing]}')

    # Fetch price history for the full gap window
    fetch_start = (last_date - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    fetch_end   = today.strftime('%Y-%m-%d')

    print(f'\n  Downloading price data {fetch_start} → {fetch_end} ...')
    spx_df  = yf.download('^GSPC', start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    vix_df  = yf.download('^VIX',  start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    skew_df = yf.download('^SKEW', start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)
    spy_df  = yf.download('SPY',   start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)

    # Carry-forward manual inputs (ADL, B20%, PC, Zero Gamma)
    def last_valid(col):
        if col in hist.columns and hist[col].notna().any():
            return float(hist[col].dropna().iloc[-1])
        return np.nan

    cf = {
        'adl_level':  last_valid('adl_level'),
        'b20_pct':    last_valid('b20_pct'),
        'pc_ratio':   last_valid('pc_ratio'),
        'zero_gamma': last_valid('zero_gamma'),
    }

    filled = 0
    for day in missing:
        spx  = get_close(spx_df,  day)
        vix  = get_close(vix_df,  day)
        skew = get_close(skew_df, day)
        spy  = get_close(spy_df,  day)

        if np.isnan(spx):
            print(f'  [{day.date()}] No SPX data — skipping (likely holiday)')
            continue

        inp_map = {
            'spx':        spx,
            'spy':        spy,
            'vix':        vix,
            'skew':       skew,
            **cf
        }

        hist = pipeline.update_history(hist, inp_map)
        mrs  = hist['mrs_score'].iloc[-1]
        print(f'  [{day.date()}]  SPX={spx:,.2f}  VIX={vix:.2f}  SKEW={skew:.1f}  '
              f'MRS={mrs:+.2f} — {pipeline.regime_label(mrs)}')
        filled += 1

    if filled:
        pipeline.save_history(hist, HIST_PATH)
        print(f'\n  Saved {len(hist)} rows → {HIST_PATH.name}')
    else:
        print('\n  Nothing was filled.')

    print('\n=== Done ===')

if __name__ == '__main__':
    main()
