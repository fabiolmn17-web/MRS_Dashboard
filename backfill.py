"""
backfill.py — Fill gaps in mrs_history.csv
===========================================
Bypasses pipeline.update_history() which uses yf.download() (403 on GitHub Actions).
Uses individual yf.Ticker().history() calls instead.
Accepts manual overrides from workflow_dispatch env vars (B20_PCT, ADL_TV, ZERO_GAMMA, PC_RATIO).
"""

import datetime
import os
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

import pipeline

HIST_PATH = Path(__file__).parent / 'mrs_history.csv'

# ── Hardcoded overrides (fallback when env vars not set) ──────────────────────
# Format: 'YYYY-MM-DD': {field: value}
MANUAL_OVERRIDES = {
    '2026-06-22': {
        'adl_level':  1_827_690.0,
        'b20_pct':    50.69,
        'zero_gamma': 7_446.59,
        'pc_ratio':   0.756,
    },
}


def get_env_override() -> dict:
    """Read workflow_dispatch inputs from environment variables."""
    _b20 = os.environ.get('B20_PCT', '').strip()
    _adl = os.environ.get('ADL_TV', '').strip()
    _zg  = os.environ.get('ZERO_GAMMA', '').strip()
    _pc  = os.environ.get('PC_RATIO', '').strip()

    override = {}
    if _b20: override['b20_pct']    = float(_b20)
    if _adl: override['adl_level']  = float(_adl) * 1000   # TradingView ×1000 → CSV scale
    if _zg:  override['zero_gamma'] = float(_zg)
    if _pc:  override['pc_ratio']   = float(_pc)
    return override


def fetch_one(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch a single ticker's Close. Individual calls avoid GitHub Actions 403."""
    try:
        h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if h.empty:
            print(f'    [{ticker}] empty response')
            return pd.Series(dtype=float)
        h.index = pd.to_datetime(h.index).normalize().tz_localize(None)
        return h['Close'].rename(ticker)
    except Exception as e:
        print(f'    [{ticker}] ERROR: {e}')
        return pd.Series(dtype=float)


def rescore(hist: pd.DataFrame) -> pd.DataFrame:
    """Replicate pipeline scoring block."""
    spy_s  = hist['spy'].astype(float)
    vix_s  = hist['vix'].astype(float)
    skew_s = hist['skew'].astype(float)
    pc_s   = hist['pc_ratio'].astype(float)
    adl_s  = hist['adl_level'].astype(float)
    b20_s  = hist['b20_pct'].astype(float)

    hist['sma50']    = spy_s.rolling(50, min_periods=1).mean()
    hist['ext_raw']  = (spy_s - hist['sma50']) / hist['sma50']
    hist['mom_raw']  = spy_s.pct_change(20, fill_method=None)
    hist['pc_sma10'] = pc_s.rolling(10, min_periods=1).mean()
    hist['pc_sma20'] = pc_s.rolling(20, min_periods=1).mean()
    hist['pc_sma50'] = pc_s.rolling(50, min_periods=1).mean()
    adl_prev = adl_s.shift(20)
    hist['adl_roc20'] = np.where(adl_prev.abs() > 1e-9,
                                  (adl_s - adl_prev) / adl_prev.abs(), np.nan)

    hist['vix_phi']  = pipeline.rolling_phi(vix_s, pipeline.PHI_W)
    hist['ext_phi']  = pipeline.rolling_phi(hist['ext_raw'].astype(float), pipeline.PHI_W)
    hist['mom_phi']  = pipeline.rolling_phi(hist['mom_raw'].astype(float), pipeline.PHI_W)
    hist['skew_phi'] = pipeline.rolling_phi(skew_s, pipeline.PHI_W)
    hist['adl_phi']  = pipeline.rolling_phi(hist['adl_roc20'].astype(float), pipeline.PHI_W)
    hist['b20_phi']  = pipeline.rolling_phi(b20_s, pipeline.PHI_W)

    vix_chg = vix_s.pct_change(fill_method=None)
    hist['spike_flag'] = (vix_chg > 0.30).fillna(False).astype(int)
    vix_phi_s       = hist['vix_phi']
    compressed_flag = (vix_phi_s < 0.30).fillna(False).astype(int)
    crossed = ((compressed_flag.shift(1) == 1) & (vix_phi_s >= 0.30)).fillna(False).astype(int)
    trig = np.zeros(len(hist), dtype=float)
    count = 0
    for i in range(len(hist)):
        if crossed.iloc[i]:                            count = 1
        elif compressed_flag.iloc[i] == 0 and count:  count += 1
        else:                                          count = 0
        trig[i] = count if 0 < count <= 7 else 0
    hist['trigger_days'] = trig
    hist['compressed']   = compressed_flag

    score_cols = ['vix_score','ext_score','mom_score','adl_score',
                  'b20_score','pc_score','skew_score','gamma_score']
    state_cols = ['vix_state','ext_state','mom_state','adl_state',
                  'b20_state','pc_state','skew_state','gamma_state']
    res = {c: [] for c in score_cols + state_cols + ['mrs_score']}

    for _, row in hist.iterrows():
        def g(c):
            v = row.get(c, np.nan)
            return np.nan if pd.isna(v) else float(v)
        vs, vst  = pipeline.score_vix(g('vix_phi'))
        es, est  = pipeline.score_extension(g('ext_phi'))
        ms, mst  = pipeline.score_momentum(g('mom_phi'))
        as_, ast = pipeline.score_adl(g('adl_phi'))
        bs, bst  = pipeline.score_b20(g('b20_phi'), g('adl_phi'))
        ps, pst  = pipeline.score_pc(g('pc_ratio'), g('pc_sma10'))
        ss, sst  = pipeline.score_skew(g('skew_phi'), g('pc_ratio'))
        gs, gst  = pipeline.score_gamma(g('spx'), g('zero_gamma'))
        scores = [vs, es, ms, as_, bs, ps, ss, gs]
        states = [vst, est, mst, ast, bst, pst, sst, gst]
        mrs = round(sum(c for c in scores if not np.isnan(c)), 2)
        for col, val in zip(score_cols, scores): res[col].append(val)
        for col, val in zip(state_cols, states): res[col].append(val)
        res['mrs_score'].append(mrs)

    for col in score_cols + state_cols + ['mrs_score']:
        hist[col] = res[col]
    return hist


def main():
    print(f'\n=== MRS Backfill — {datetime.date.today()} ===\n')

    # ── Read env var overrides (from workflow_dispatch inputs) ────────────────
    env_override = get_env_override()
    if env_override:
        print(f'  ENV overrides: {env_override}')

    hist = pipeline.load_history(HIST_PATH)

    # ── Last date with a valid SPX close ──────────────────────────────────────
    spx_valid = hist.dropna(subset=['spx'])
    if spx_valid.empty:
        print('  ERROR: no valid SPX rows in history.')
        return
    last_valid = spx_valid['date'].max()
    print(f'  Last valid SPX close : {last_valid.date()}')
    print(f'  Last row in CSV      : {hist["date"].max().date()}')

    # ── Trim trailing NaN / holiday rows ──────────────────────────────────────
    hist = hist[hist['date'] <= last_valid].reset_index(drop=True)

    # ── Missing business days ─────────────────────────────────────────────────
    today     = pd.Timestamp(datetime.date.today()).normalize()
    yesterday = today - pd.Timedelta(days=1)
    missing   = pd.bdate_range(start=last_valid + pd.Timedelta(days=1), end=yesterday)

    if len(missing) == 0:
        print('  Already up to date — nothing to backfill.')
        return

    print(f'  Missing: {[str(d.date()) for d in missing]}')

    # ── Carry-forward base manual inputs ──────────────────────────────────────
    def last_val(col):
        s = hist[col].dropna() if col in hist.columns else pd.Series(dtype=float)
        return float(s.iloc[-1]) if len(s) > 0 else np.nan

    base_manual = {
        'adl_level':  last_val('adl_level'),
        'b20_pct':    last_val('b20_pct'),
        'zero_gamma': last_val('zero_gamma'),
        'pc_ratio':   last_val('pc_ratio'),
    }
    print(f'  Base carry-forward   : {base_manual}')

    # ── CBOE PC ratio ─────────────────────────────────────────────────────────
    pc_series = pipeline.fetch_cboe_pc()

    # ── Fetch market data via individual Ticker calls ─────────────────────────
    fetch_start = (last_valid - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    fetch_end   = (yesterday  + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    print(f'\n  Fetching market data ({fetch_start} → {fetch_end})...')

    spy_s  = fetch_one('SPY',   fetch_start, fetch_end)
    spx_s  = fetch_one('^GSPC', fetch_start, fetch_end)
    vix_s  = fetch_one('^VIX',  fetch_start, fetch_end)
    skew_s = fetch_one('^SKEW', fetch_start, fetch_end)

    print(f'    SPY  rows: {len(spy_s)}  last: {spy_s.index[-1].date() if len(spy_s) else "—"}')
    print(f'    SPX  rows: {len(spx_s)}  last: {spx_s.index[-1].date() if len(spx_s) else "—"}')
    print(f'    VIX  rows: {len(vix_s)}  last: {vix_s.index[-1].date() if len(vix_s) else "—"}')
    print(f'    SKEW rows: {len(skew_s)} last: {skew_s.index[-1].date() if len(skew_s) else "—"}')

    # ── Append one row per missing day ────────────────────────────────────────
    appended = 0
    for dt in missing:
        spx_val = float(spx_s.loc[dt]) if dt in spx_s.index else np.nan
        if np.isnan(spx_val):
            print(f'\n  {dt.date()}: no SPX data — skipping (holiday?)')
            continue

        spy_val  = float(spy_s.loc[dt])  if dt in spy_s.index  else np.nan
        vix_val  = float(vix_s.loc[dt])  if dt in vix_s.index  else np.nan
        skew_val = float(skew_s.loc[dt]) if dt in skew_s.index else np.nan

        # Merge: base → hardcoded override → env override (env wins)
        date_str = dt.strftime('%Y-%m-%d')
        override = {**MANUAL_OVERRIDES.get(date_str, {}), **env_override}
        manual   = {**base_manual, **override}

        pc_val = float(pc_series.loc[dt]) if dt in pc_series.index else manual['pc_ratio']

        print(f'\n  {dt.date()}:')
        print(f'    SPX={spx_val:.2f}  VIX={vix_val:.2f}  SPY={spy_val:.2f}  SKEW={skew_val:.2f}')
        print(f'    ADL={manual["adl_level"]:.0f}  B20={manual["b20_pct"]:.2f}%  '
              f'ZeroG={manual["zero_gamma"]:.2f}  PC={pc_val:.3f}')
        if override:
            print(f'    [OVERRIDE applied: {list(override.keys())}]')

        new_row = {col: np.nan for col in pipeline.HIST_COLS}
        new_row.update({
            'date':         dt,
            'spy':          spy_val,
            'spx':          spx_val,
            'vix':          vix_val,
            'skew':         skew_val,
            'adl_level':    manual['adl_level'],
            'b20_pct':      manual['b20_pct'],
            'zero_gamma':   manual['zero_gamma'],
            'pc_ratio':     pc_val,
            'spike_flag':   0,
            'compressed':   0,
            'trigger_days': 0.0,
        })

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        hist = hist.sort_values('date').reset_index(drop=True)
        appended += 1

    if appended == 0:
