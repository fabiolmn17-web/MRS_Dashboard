"""Offline build of risk_long.csv (MRS 8.6.0): S&P 500 daily since 1990 with the dashboard's own rules.
Columns: date, ph (SPY-phase code from chart_view.spy_phase, 0 = DIP ZONE), vb (VIX band vs 17.56: CALM / WIDE / STRESSED),
f20 (close-to-close return over the next 20 sessions), mae20 (lowest low over the next 20 sessions vs the entry close).
Inputs (PC, Studies folder): TradingView SPX daily (real OHLC from 1962) and the VIX daily from 1990.
Re-run once or twice a year; the Risk today panel reads the file and falls back to SPY since 2008 without it."""
import sys, os, numpy as np, pandas as pd
import chart_view
VM = 17.56
B = os.path.expanduser('~/mnt/Epistruct resesarch/TRADING DESK/Studies/Index_Edge_Program_2026-11')
spx = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else B + '/MRS_v7/v2/TV_SPX_1D_1871-02-01_to_2026-11-13.csv')
spx.index = pd.to_datetime(spx['time']); spx = spx[spx.index >= '1962-03-23'][['open', 'high', 'low', 'close']].rename(columns=str.title)
tv = pd.read_csv(B + '/data/raw/tv/TV_VIX_rates_HYG_VVIX_VIX3M_VIX6M_VX1_1D_1990-01-03_to_2026-11-13.csv')
vix = pd.Series(tv['close'].values, index=pd.to_datetime(tv['time'])).dropna()
ph = chart_view.spy_phase(spx)
c, l = spx['Close'].values, spx['Low'].values; n = len(c)
f20 = np.full(n, np.nan); mae = np.full(n, np.nan)
for i in range(n - 20):
    f20[i] = c[i + 20] / c[i] - 1; mae[i] = l[i + 1:i + 21].min() / c[i] - 1
v = vix.reindex(spx.index).ffill().values
vb = np.where(np.isnan(v), '', np.where(v < VM, 'CALM', np.where(v < 1.5 * VM, 'WIDE', 'STRESSED')))
out = pd.DataFrame({'date': spx.index.strftime('%Y-%m-%d'), 'ph': ph.values.astype(int), 'vb': vb, 'f20': np.round(f20, 5), 'mae20': np.round(mae, 5)})
out = out[(out.date >= '1990-01-02') & out.f20.notna() & (out.vb != '')]
out.to_csv('risk_long.csv', index=False); print(len(out), out.date.iloc[0], out.date.iloc[-1])
