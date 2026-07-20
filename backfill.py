"""
backfill.py — Fill gaps in mrs_history.csv
===========================================
Uses individual yf.Ticker().history() calls (avoids GitHub Actions 403).
Accepts manual overrides from workflow_dispatch env vars.

IMPORTANT: This file does NOT duplicate scoring logic.
All scoring is done by calling pipeline.score_dataframe().
"""
import datetime
import os
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import pipeline

HIST_PATH = Path(__file__).parent / 'mrs_history.csv'


def get_env_override() -> dict:
    """Read workflow_dispatch inputs from environment variables — with validation."""
    _b20 = os.environ.get('B20_PCT', '').strip()
    _adl = os.environ.get('ADL_TV', '').strip()
    _zg  = os.environ.get('ZERO_GAMMA', '').strip()
    _pc  = os.environ.get('PC_RATIO', '').strip()
    override = {}

    if _b20:
        b20 = float(_b20)
        if 0 < b20 <= 100:
            override['b20_pct'] = b20
        else:
            print(f'  WARNING: B20_PCT={b20} is outside [0,100] — ignored.')

    if _adl:
        adl = float(_adl) * 1000   # TradingView ×1000 → CSV scale
        if 100_000 < adl < 20_000_000:
            override['adl_level'] = adl
        else:
            print(f'  WARNING: ADL_TV={_adl} converts to {adl:.0f} which looks wrong — ignored.')

    if _zg:
        zg = float(_zg)
        if 2_000 < zg < 15_000:
            override['zero_gamma'] = zg
        else:
            print(f'  WARNING: ZERO_GAMMA={zg} is outside normal SPX range [2000,15000] — ignored.')

    if _pc:
        pc = float(_pc)
        if 0 < pc <= 3.0:
            override['pc_ratio'] = pc
        else:
            print(f'  WARNING: PC_RATIO={pc} is outside [0,3.0] — ignored (will auto-fetch).')

    return override


def fetch_one(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Fetch a single ticker's OHLCV with retry. Returns DataFrame with Close and Volume."""
    import time as _time
    for attempt in range(1, retries + 1):
        try:
            h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if h.empty:
                print(f'    [{ticker}] empty response (attempt {attempt}/{retries})')
            else:
                h.index = pd.to_datetime(h.index).normalize().tz_localize(None)
                return h[['Close', 'Volume']] if 'Volume' in h.columns else h[['Close']]
        except Exception as e:
            print(f'    [{ticker}] ERROR attempt {attempt}/{retries}: {e}')
        if attempt < retries:
            _time.sleep(3)
    print(f'    [{ticker}] all {retries} attempts failed — returning empty DataFrame')
    return pd.DataFrame()


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
    end_date  = today if env_override else yesterday

    # When env_override is provided, re-process the last business day so
    # corrected manual inputs (B20, ADL, Zero Gamma) overwrite the existing row.
    if env_override and len(hist) > 0:
        last_bday = pd.bdate_range(end=today, periods=1)[0]
        if last_valid == last_bday:
            print(f'  ENV override: dropping {last_bday.date()} to re-process with new inputs...')
            hist       = hist[hist['date'] < last_bday].reset_index(drop=True)
            last_valid = hist['date'].max() if len(hist) > 0 else pd.Timestamp('2000-01-01')

    missing = pd.bdate_range(start=last_valid + pd.Timedelta(days=1), end=end_date)

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
    fetch_end   = (today + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print(f'\n  Fetching market data ({fetch_start} → {fetch_end})...')

    spy_df  = fetch_one('SPY',   fetch_start, fetch_end)
    spx_df  = fetch_one('^GSPC', fetch_start, fetch_end)
    vix_df  = fetch_one('^VIX',  fetch_start, fetch_end)
    skew_df = fetch_one('^SKEW', fetch_start, fetch_end)

    spy_s  = spy_df['Close'] if not spy_df.empty else pd.Series(dtype=float)
    vol_s  = spy_df['Volume'] if not spy_df.empty and 'Volume' in spy_df.columns else pd.Series(dtype=float)
    spx_s  = spx_df['Close'] if not spx_df.empty else pd.Series(dtype=float)
    vix_s  = vix_df['Close'] if not vix_df.empty else pd.Series(dtype=float)
    skew_s = skew_df['Close'] if not skew_df.empty else pd.Series(dtype=float)

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
        vol_val  = float(vol_s.loc[dt])  if dt in vol_s.index  else np.nan

        # Merge: base → env override (env wins)
        manual = {**base_manual, **env_override}

        pc_val = float(pc_series.loc[dt]) if dt in pc_series.index else manual['pc_ratio']

        print(f'\n  {dt.date()}:')
        print(f'    SPX={spx_val:.2f}  VIX={vix_val:.2f}  SPY={spy_val:.2f}  SKEW={skew_val:.2f}')
        vol_str = f'{vol_val/1e6:.1f}M' if not np.isnan(vol_val) else '—'
        print(f'    ADL={manual["adl_level"]:.0f}  B20={manual["b20_pct"]:.2f}%  '
              f'ZeroG={manual["zero_gamma"]:.2f}  PC={pc_val:.3f}  VOL={vol_str}')
        if env_override:
            print(f'    [OVERRIDE applied: {list(env_override.keys())}]')

        new_row = {col: np.nan for col in pipeline.HIST_COLS}
        new_row.update({
            'date':         dt,
            'spy':          spy_val,
            'spx':          spx_val,
            'vix':          vix_val,
            'skew':         skew_val,
            'volume':       vol_val,
            'adl_level':    manual['adl_level'],
            'b20_pct':      manual['b20_pct'],
            'zero_gamma':   manual['zero_gamma'],
            'pc_ratio':     pc_val,
        })
        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        hist = hist.sort_values('date').reset_index(drop=True)
        appended += 1

    if appended == 0:
        print('\n  No rows appended (all missing days were holidays or had no SPX data).')
        return

    # ── Rescore using pipeline (SINGLE SOURCE OF TRUTH) ───────────────────────
    print(f'\n  Rescoring {len(hist)} rows using pipeline.score_dataframe()...')
    hist = pipeline.score_dataframe(hist)

    # ── Save ──────────────────────────────────────────────────────────────────
    pipeline.save_history(hist, HIST_PATH)
    last_row = hist.iloc[-1]
    print(f'\n  ✅ Saved {appended} new row(s) to {HIST_PATH.name}')
    print(f'  Last row: {last_row["date"].date()} | '
          f'SPX={last_row["spx"]:.2f} | VIX={last_row["vix"]:.2f} | '
          f'MRS={last_row["mrs_score"]:+.2f}')


if __name__ == '__main__':
    main()
