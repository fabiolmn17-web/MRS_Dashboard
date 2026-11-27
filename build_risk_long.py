"""Build risk_long.csv for the Risk today panel: S&P 500 daily since 1990 with the dashboard's own rules.

Columns: date, ph (SPY-phase code from chart_view.spy_phase, 0 = DIP ZONE), vix (VIX close), vb (VIX band vs the 1990
expanding median at the build date; the panel re-classifies from vix with the live median since 8.15.0), f20 (close-to-
close return over the next 20 sessions), mae20 (lowest low over the next 20 sessions vs the entry close), r1..r20 (8.15.0:
close-to-close return after 1..20 sessions, for the fan chart). Only days with a complete 20-session outcome are kept.

Inputs, two modes:
  PC (default): TradingView SPX daily (real OHLC from 1962) and the TradingView VIX file in the Studies folder;
                --ext adds later bars (csv with date, spx_open, spx_high, spx_low, spx_close, vix_close; e.g. IBKR read-only).
  --yahoo:      ^GSPC and ^VIX from Yahoo (GitHub Actions, monthly: workflows_8_15/risk_long_monthly.yml). The new file is
                written only if it reaches at least as far as the current one and keeps at least as many rows.
usage: python build_risk_long.py [--spx F] [--vix F] [--ext F] [--yahoo] [--out risk_long.csv]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import chart_view

B = os.path.expanduser('~/mnt/Epistruct resesarch/TRADING DESK/Studies/Index_Edge_Program_2026-11')
SPX_TV = B + '/MRS_v7/v2/TV_SPX_1D_1871-02-01_to_2026-11-13.csv'
VIX_TV = B + '/data/raw/tv/TV_VIX_rates_HYG_VVIX_VIX3M_VIX6M_VX1_1D_1990-01-03_to_2026-11-13.csv'
H = 20


def _flat(df):
    if isinstance(df.columns, pd.MultiIndex):        # yfinance >= 0.2.48 returns (field, ticker) columns
        df = df.copy(); df.columns = df.columns.get_level_values(0)
    return df


def load_pc(spx_path, vix_path, ext_path=None):
    spx = pd.read_csv(spx_path)
    t = pd.to_datetime(spx['time'])
    spx.index = t.dt.tz_localize(None) if t.dt.tz is not None else t
    spx = spx[spx.index >= '1962-03-23'][['open', 'high', 'low', 'close']].rename(columns=str.title)
    tv = pd.read_csv(vix_path)
    vix = pd.Series(tv['close'].values, index=pd.to_datetime(tv['time'])).dropna()
    if ext_path:
        e = pd.read_csv(ext_path, parse_dates=['date']).set_index('date').sort_index()
        e = e[e.index > spx.index[-1]]
        add = e[['spx_open', 'spx_high', 'spx_low', 'spx_close']].set_axis(['Open', 'High', 'Low', 'Close'], axis=1)
        spx = pd.concat([spx, add])
        ve = e['vix_close'].dropna()
        vix = pd.concat([vix, ve[ve.index > vix.index[-1]]])
    return spx, vix


def load_yahoo():
    import yfinance as yf
    g = _flat(yf.download('^GSPC', start='1962-01-01', auto_adjust=False, progress=False))
    v = _flat(yf.download('^VIX', start='1990-01-01', auto_adjust=False, progress=False))
    spx = g[['Open', 'High', 'Low', 'Close']].dropna()
    spx.index = pd.to_datetime(spx.index).tz_localize(None).normalize()
    vix = v['Close'].dropna(); vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
    return spx, vix


def build(spx, vix):
    ph = chart_view.spy_phase(spx)
    c, l = spx['Close'].values.astype(float), spx['Low'].values.astype(float); n = len(c)
    f20 = np.full(n, np.nan); mae = np.full(n, np.nan); R = np.full((n, H), np.nan)
    for i in range(n - H):
        f20[i] = c[i + H] / c[i] - 1; mae[i] = l[i + 1:i + H + 1].min() / c[i] - 1
        R[i] = c[i + 1:i + H + 1] / c[i] - 1
    v = vix.reindex(spx.index).ffill().values
    med = float(vix[vix.index >= '1990-01-01'].median())
    vb = np.where(np.isnan(v), '', np.where(v < med, 'CALM', np.where(v < 1.5 * med, 'WIDE', 'STRESSED')))
    out = pd.DataFrame({'date': spx.index.strftime('%Y-%m-%d'), 'ph': np.asarray(ph.values, int), 'vix': np.round(v, 2), 'vb': vb,
                        'f20': np.round(f20, 5), 'mae20': np.round(mae, 5)})
    for k in range(H):
        out[f'r{k + 1}'] = np.round(R[:, k], 4)
    out = out[(out.date >= '1990-01-02') & out.f20.notna() & (out.vb != '')]
    return out, med


def should_write(out: pd.DataFrame, old_path: str):
    """Guard for the unattended (Yahoo) rebuild: (ok, reason). Never replace the file with a shorter or older one."""
    if not os.path.exists(old_path):
        return True, 'no current file'
    old = pd.read_csv(old_path, usecols=['date'])
    if out.date.iloc[-1] < old.date.iloc[-1] or len(out) < len(old) - 5:
        return False, (f'new file ends {out.date.iloc[-1]} with {len(out)} rows; current ends {old.date.iloc[-1]} '
                       f'with {len(old)} rows')
    return True, 'ok'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--spx', default=SPX_TV); ap.add_argument('--vix', default=VIX_TV); ap.add_argument('--ext')
    ap.add_argument('--yahoo', action='store_true'); ap.add_argument('--out', default='risk_long.csv')
    a = ap.parse_args()
    spx, vix = load_yahoo() if a.yahoo else load_pc(a.spx, a.vix, a.ext)
    out, med = build(spx, vix)
    if a.yahoo:
        ok, why = should_write(out, a.out)
        if not ok:
            print('not written: ' + why); sys.exit(0)
    out.to_csv(a.out, index=False)
    print(len(out), out.date.iloc[0], out.date.iloc[-1], f'VIX median {med:.2f}')
