"""Offline build of joint_long.csv (MRS 8.10.0, 21 Nov 2026): the history behind the Joint view card.
Rows: S&P 500 daily since 1963. Columns:
  date, E (close vs the 50-day in Wilder ATR(14)), F (200-day filter ON / OFF: above a rising 200-day, slope over 20 sessions),
  fall60 (worst further fall within 60 sessions, % of the close; DIP ZONE days only, E <= -1),
  ret (daily return %), dvix (VIX change in points), vb (VIX width at the PRIOR close vs the 1990 median: CALM / WIDE / STRESSED).
Inputs (PC, Studies folder): TradingView SPX daily (real OHLC from 1962) and the VIX daily from 1990, as build_risk_long.py.
Re-run once or twice a year; the card reads the file and is skipped without it."""
import os
import sys

import numpy as np
import pandas as pd

VM = 17.56   # VIX median since 1990 at build time (exposure_guide.VIX_MEDIAN_FIXED)
B = os.path.expanduser('~/mnt/Epistruct resesarch/TRADING DESK/Studies/Index_Edge_Program_2026-11')
spx = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else B + '/MRS_v7/v2/TV_SPX_1D_1871-02-01_to_2026-11-13.csv')
spx.index = pd.to_datetime(spx['time']); spx = spx[spx.index >= '1962-03-23'][['open', 'high', 'low', 'close']]
tv = pd.read_csv(B + '/data/raw/tv/TV_VIX_rates_HYG_VVIX_VIX3M_VIX6M_VX1_1D_1990-01-03_to_2026-11-13.csv')
vix = pd.Series(tv['close'].values, index=pd.to_datetime(tv['time'])).dropna()

c, h, l = spx['close'], spx['high'], spx['low']
tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
E = (c - c.rolling(50).mean()) / atr
m = c.rolling(200).mean()
F = np.where((c > m) & (m.diff(20) > 0), 'ON', 'OFF')
cv, lv = c.values, l.values; n = len(cv)
fall = np.full(n, np.nan)
for i in range(n - 60):
    if E.iloc[i] <= -1:
        fall[i] = (lv[i + 1:i + 61].min() / cv[i] - 1) * 100
v = vix.reindex(spx.index)
vb_prev = v.shift().map(lambda x: '' if x != x else ('CALM' if x < VM else ('WIDE' if x < 1.5 * VM else 'STRESSED')))
out = pd.DataFrame({'date': spx.index.strftime('%Y-%m-%d'), 'E': E.round(3).values, 'F': F,
                    'fall60': np.round(fall, 2), 'ret': (c.pct_change() * 100).round(3).values,
                    'dvix': v.diff().round(2).values, 'vb': vb_prev.values})
keep = (out.date >= '1963-01-01') & (out.fall60.notna() | ((out.vb != '') & out.dvix.notna()))
out = out[keep]
out.to_csv('joint_long.csv', index=False)
print(len(out), out.date.iloc[0], out.date.iloc[-1], 'DIP rows', int(out.fall60.notna().sum()), 'VIX rows', int((out.vb != '').sum()))
