"""
gamma_shadow.py — store the Gamma Live zero gamma sent from the trading PC (shadow column, not scored)
=====================================================================================================
Run by .github/workflows/gamma_shadow.yml (workflow_dispatch), which Gamma Live triggers after each close
(gamma_live/mrs_push.py). Inputs arrive as environment variables:

    GL_DATE        session date, YYYY-MM-DD (New York)
    GL_SPY_CLOSE   SPY close used for the calculation
    GL_ZG40_NEXT   SPY 40-day zero gamma without the expiring contracts (= the next session's map)
    GL_ZG40_CLOSE  SPY 40-day zero gamma as Gamma Live showed it at the close (expiring contracts included)
    GL_ZG3_NEXT    SPY zero gamma of the next 3 expiries
    GL_NET40_BN    SPY 40-day net GEX, $bn per 1%, without the expiring contracts

Upserts one row per date into gamma_shadow.csv. The dashboard converts SPY points to SPX points with that
session's SPX/SPY closes from mrs_history.csv and shows it next to the InsiderFinance value.
"""
import os
import sys
from pathlib import Path

import pandas as pd

PATH = Path(__file__).parent / 'gamma_shadow.csv'
COLS = ['date', 'spy_close', 'zg40_next', 'zg40_close', 'zg3_next', 'net40_next_bn']
ENV = {'spy_close': 'GL_SPY_CLOSE', 'zg40_next': 'GL_ZG40_NEXT', 'zg40_close': 'GL_ZG40_CLOSE',
       'zg3_next': 'GL_ZG3_NEXT', 'net40_next_bn': 'GL_NET40_BN'}


def main():
    day = os.environ.get('GL_DATE', '').strip()
    try:
        pd.Timestamp(day)
    except Exception:
        print(f'GL_DATE "{day}" is not a date - nothing stored.')
        sys.exit(1)
    row = {'date': day}
    for col, var in ENV.items():
        v = os.environ.get(var, '').strip()
        try:
            row[col] = round(float(v), 4) if v else float('nan')
        except ValueError:
            row[col] = float('nan')
    if not (200 < row['zg40_next'] < 2000 or 200 < row['zg40_close'] < 2000):
        print(f'No valid SPY zero gamma in the inputs ({row}) - nothing stored.')
        sys.exit(1)
    old = pd.read_csv(PATH, dtype={'date': str}) if PATH.exists() else pd.DataFrame(columns=COLS)
    out = pd.concat([old[old['date'] != day], pd.DataFrame([row])], ignore_index=True)
    out = out.reindex(columns=COLS).sort_values('date')
    out.to_csv(PATH, index=False)
    print(f'Stored {day}: {row}  ({len(out)} rows)')


if __name__ == '__main__':
    main()
