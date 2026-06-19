"""
pipeline.py — MRS core scoring engine (web edition)
====================================================
Extracted from run_mrs.py.  No Excel / openpyxl dependencies.
Import this module from update_data.py (daily batch) and app.py (dashboard).
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Cloud-safe browser headers (bypasses 403 on CBOE/Yahoo in GitHub Actions) ─
_BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/csv,application/csv,*/*',
    'Referer': 'https://www.cboe.com/',
}

# ── Constants ──────────────────────────────────────────────────────────────────
PHI_W    = 756   # 3-year rolling window (~756 trading days)
CBOE_URL = ('https://cdn.cboe.com/data/us/options/market_statistics/'
            'daily_puts_calls.csv')

HIST_COLS = [
    'date', 'spy', 'spx', 'vix', 'skew', 'pc_ratio',
    'pc_sma10', 'pc_sma20', 'pc_sma50',
    'sma50', 'ext_raw', 'mom_raw',
    'adl_level', 'adl_roc20',
    'b20_pct', 'b50_pct',
    'zero_gamma',
    'vix_phi', 'ext_phi', 'mom_phi', 'skew_phi', 'adl_phi', 'b20_phi',
    'spike_flag', 'compressed', 'trigger_days',
    'vix_score', 'ext_score', 'mom_score',
    'adl_score', 'b20_score', 'pc_score', 'skew_score', 'gamma_score',
    'vix_state', 'ext_state', 'mom_state',
    'adl_state', 'b20_state', 'pc_state', 'skew_state', 'gamma_state',
    'mrs_score'
]


# ── Rolling Phi ────────────────────────────────────────────────────────────────
def rolling_phi(series: pd.Series, window: int = PHI_W) -> pd.Series:
    """Empirical percentile rank over a rolling look-back window."""
    arr = series.values.astype(float)
    out = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        if np.isnan(arr[i]):
            continue
        w     = arr[i - window : i]
        valid = ~np.isnan(w)
        if valid.sum() > 0:
            out[i] = np.nansum(w[valid] < arr[i]) / valid.sum()
    return pd.Series(out, index=series.index)


# ── CBOE PC ratio fetch ────────────────────────────────────────────────────────
def fetch_cboe_pc() -> pd.Series:
    try:
        r = requests.get(CBOE_URL, headers=_BROWSER_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c), None)
        eq_col   = next((c for c in df.columns if 'equity' in c or 'total' in c), None)
        if date_col and eq_col:
            df[date_col] = pd.to_datetime(df[date_col])
            s = df.set_index(date_col)[eq_col].dropna().astype(float)
            s.index = s.index.normalize()
            return s
    except Exception as e:
        print(f'  [WARN] CBOE PC fetch failed: {e}')
    return pd.Series(dtype=float)


# ── Scoring functions ──────────────────────────────────────────────────────────
def score_vix(phi: float):
    if np.isnan(phi): return  0.0, 'No data'
    if phi < 0.30:    return  1.0, 'Low'
    if phi < 0.60:    return  0.0, 'Mid'
    if phi < 0.80:    return -0.5, 'High'
    return -1.5, 'Stress'

def score_extension(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -0.5, 'Compressed'
    if phi < 0.70:    return  0.0, 'Normal'
    return -0.5, 'Extended'

def score_momentum(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return  0.0, 'Normal'
    return 0.5, 'Strong'

def score_adl(phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:    return -1.0, 'Weak'
    if phi < 0.70:    return  0.0, 'Normal'
    return 0.0, 'Strong'

def score_b20(phi: float, adl_phi: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30:
        s = -0.5 if (not np.isnan(adl_phi) and adl_phi < 0.30) else 0.0
        return s, 'Low'
    if phi < 0.70: return 0.0, 'Normal'
    return 0.5, 'High'

def score_pc(pc: float, pc_sma10: float):
    """Five-Zone Model — June 2026 calibration (Studies 7 & 8)."""
    if np.isnan(pc_sma10): return 0.0, 'No data'
    if pc_sma10 < 0.686:   return  0.5, 'Extreme LOW (complacency)'
    if pc_sma10 < 0.732:   return -0.5, 'Moderate LOW (transition)'
    if pc_sma10 < 0.944:   return  0.0, 'Mid'
    if pc_sma10 < 1.003:   return  0.5, 'Moderate HIGH (fear building)'
    return                         1.0, 'Extreme HIGH (contrarian)'

def score_skew(phi: float, pc: float):
    if np.isnan(phi): return 0.0, 'No data'
    if phi < 0.30 and not np.isnan(pc) and pc > 1.00:
        return -2.0, 'Low+HighPC(DANGER)'
    if phi > 0.70 and not np.isnan(pc) and pc < 0.70:
        return  1.5, 'High+LowPC(SAFE)'
    if phi < 0.30: return -1.0, 'Low'
    if phi < 0.70: return  0.0, 'Mid'
    return 0.5, 'High'

def score_gamma(spx: float, zero_gamma: float):
    if np.isnan(spx) or np.isnan(zero_gamma) or zero_gamma <= 0:
        return 0.0, 'No data'
    dist = (spx - zero_gamma) / spx
    if dist > 0.01:  return  0.5, 'Above Gamma'
    if dist > -0.01: return  0.0, 'Near Gamma'
    return -0.5, 'Below Gamma'


# ── Regime helpers ─────────────────────────────────────────────────────────────
def regime_label(mrs: float) -> str:
    if mrs >= 1.5:  return 'RISK-ON'
    if mrs >= 0.5:  return 'MILD RISK-ON'
    if mrs >= -0.5: return 'NEUTRAL'
    if mrs >= -1.5: return 'MILD RISK-OFF'
    return 'RISK-OFF'

def regime_color(mrs: float) -> str:
    """Hex color for regime band (web display)."""
    if mrs >= 1.5:  return '#1a7f37'
    if mrs >= 0.5:  return '#57a66b'
    if mrs >= -0.5: return '#6b7280'
    if mrs >= -1.5: return '#d97706'
    return '#b91c1c'

def compute_regime_duration(hist: pd.DataFrame, ref_date) -> int:
    df = hist[hist['date'] <= pd.Timestamp(ref_date)].sort_values('date')
    if df.empty:
        return 0
    scores = df['mrs_score'].tolist()
    is_neg = float(scores[-1]) < 0
    count  = 0
    for v in reversed(scores):
        try:
            if (float(v) < 0) == is_neg:
                count += 1
            else:
                break
        except Exception:
            break
    return count

def compute_signal_quality(last: dict, hist: pd.DataFrame, ref_date) -> tuple:
    """Returns (label, description, hex_color). See run_mrs.py for full docs."""
    score   = float(last.get('mrs_score', 0) or 0)
    regime  = regime_label(score)
    is_pos  = score > 0
    is_neg  = score < 0
    is_neut = not is_pos and not is_neg

    def _phi(key):
        v = last.get(key, np.nan)
        try:    return float(v) if not pd.isna(v) else np.nan
        except: return np.nan

    def _sc(key):
        v = last.get(key, 0)
        try:    return float(v) if not pd.isna(v) else 0.0
        except: return 0.0

    b20_phi = _phi('b20_phi');  adl_phi  = _phi('adl_phi')
    vix_phi = _phi('vix_phi');  skew_phi = _phi('skew_phi')
    b20_sc  = _sc('b20_score'); adl_sc   = _sc('adl_score')
    pc_sc   = _sc('pc_score');  skew_sc  = _sc('skew_score')
    mom_sc  = _sc('mom_score'); gam_sc   = _sc('gamma_score')

    breadth_sum = b20_sc + adl_sc
    flow_sum    = pc_sc  + skew_sc

    if is_pos:
        breadth_state = 'confirming' if breadth_sum > 0 else ('opposing' if breadth_sum < 0 else 'neutral')
    elif is_neg:
        breadth_state = 'confirming' if breadth_sum < 0 else ('opposing' if breadth_sum > 0 else 'neutral')
    else:
        breadth_state = 'neutral'

    PROX    = 0.05
    at_risk = []
    if not np.isnan(b20_phi):
        if b20_sc >= 0 and b20_phi < 0.30 + PROX:
            at_risk.append(f'B20 Φ={b20_phi:.3f} near −0.5 threshold')
        elif b20_sc <= 0 and b20_phi > 0.70 - PROX:
            at_risk.append(f'B20 Φ={b20_phi:.3f} near +0.5 threshold')
    if not np.isnan(adl_phi):
        if adl_sc >= 0 and adl_phi < 0.30 + PROX:
            at_risk.append(f'ADL Φ={adl_phi:.3f} near −1.0 threshold')
    if not np.isnan(vix_phi):
        if vix_phi > 0.70 - PROX and _sc('vix_score') >= 0:
            at_risk.append(f'VIX Φ={vix_phi:.3f} near −0.5 threshold')

    df5 = hist[hist['date'] <= pd.Timestamp(ref_date)].sort_values('date').tail(6)

    def _trend(col):
        vals = df5[col].dropna() if col in df5.columns else pd.Series(dtype=float)
        if len(vals) >= 3:
            delta = float(vals.iloc[-1]) - float(vals.iloc[0])
            return 'declining' if delta < -0.02 else ('rising' if delta > 0.02 else 'stable')
        return 'unknown'

    b20_trend = _trend('b20_phi')
    adl_trend = _trend('adl_phi')

    driver_parts = []
    for name, sc in [('PC', pc_sc), ('SKEW', skew_sc), ('Momentum', mom_sc),
                     ('Gamma', gam_sc), ('VIX', _sc('vix_score')),
                     ('B20', b20_sc), ('ADL', adl_sc)]:
        if sc != 0:
            driver_parts.append(f'{name} ({sc:+.1f})')
    drivers_str = ', '.join(driver_parts) if driver_parts else 'all at zero'
    fragile_str = ('⚠ FRAGILE: ' + '; '.join(at_risk)) if at_risk else ''

    if is_neut:
        if breadth_sum < 0 or flow_sum < 0:
            return ('NEUTRAL — BEARISH LEAN', f'Internal structure has bearish bias. Drivers: {drivers_str}. {fragile_str}', 'C55A11')
        elif breadth_sum > 0 or flow_sum > 0:
            return ('NEUTRAL — BULLISH LEAN', f'Internal structure has bullish bias. Drivers: {drivers_str}. {fragile_str}', '375623')
        else:
            return ('NEUTRAL — NO EDGE', 'All components near zero. No directional bias.', '595959')
    elif breadth_state == 'confirming':
        return ('CONFIRMED',
                f'Breadth (B20 Φ={b20_phi:.3f}, ADL Φ={adl_phi:.3f}) aligned with {regime}. Drivers: {drivers_str}. {fragile_str}',
                '375623')
    elif breadth_state == 'opposing':
        trend_note = ''
        if b20_trend == 'declining': trend_note += f' B20 Φ declining.'
        if adl_trend == 'declining': trend_note += f' ADL Φ declining.'
        return ('DIVERGENT',
                f'Mathematically {regime} but breadth opposes signal. B20 Φ={b20_phi:.3f}, ADL Φ={adl_phi:.3f}.{trend_note} Drivers: {drivers_str}. {fragile_str}',
                '7B0000')
    else:
        trend_note = ''
        if b20_trend == 'declining': trend_note += f' B20 Φ declining.'
        if adl_trend == 'declining': trend_note += f' ADL Φ declining.'
        return ('UNCONFIRMED',
                f'Mathematically {regime}, driven by flow/sentiment. Breadth neutral: B20 Φ={b20_phi:.3f}, ADL Φ={adl_phi:.3f}.{trend_note} Drivers: {drivers_str}. {fragile_str}',
                'ED7D31')


# ── History I/O ────────────────────────────────────────────────────────────────
def load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    if 'b50_raw' in df.columns and 'b20_pct' not in df.columns:
        df['b20_pct'] = df['b50_raw']
    for col in HIST_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df.sort_values('date').reset_index(drop=True)

def save_history(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


# ── Core update logic ──────────────────────────────────────────────────────────
def update_history(hist: pd.DataFrame, inp_map: dict) -> pd.DataFrame:
    """
    Append new trading days and rescore the full history.

    inp_map: dict of pd.Timestamp -> {adl_level, b20_pct, zero_gamma, pc_ratio, skew}
             Built by auto_fetch.py (web) or MRS_Inputs_v4.xlsx (local).
    """
    today_dt  = date.today()
    last_date = hist['date'].max()

    # ── 1. Fetch market data (individual calls — avoids GitHub Actions 403) ───
    print('  Fetching SPY / SPX / VIX / SKEW...')
    start_fetch = (last_date - timedelta(days=10)).strftime('%Y-%m-%d')
    _ticker_map = {'spy': 'SPY', 'spx': '^GSPC', 'vix': '^VIX', 'skew': '^SKEW'}
    _frames = {}
    for field, ticker in _ticker_map.items():
        try:
            h = yf.Ticker(ticker).history(start=start_fetch, auto_adjust=True)
            if not h.empty:
                h.index = pd.to_datetime(h.index).normalize().tz_localize(None)
                _frames[field] = h['Close'].rename(field)
        except Exception as e:
            print(f'  [WARN] {ticker}: {e}')
    if not _frames:
        print('  [ERROR] No market data fetched — aborting.')
        return hist
    close = pd.concat(_frames.values(), axis=1)
    close.index = close.index.normalize()

    # ── 2. Fetch CBOE PC ratio ────────────────────────────────────────────────
    print('  Fetching CBOE PC ratio...')
    pc_series = fetch_cboe_pc()

    # ── 3. Carry-forward helper ───────────────────────────────────────────────
    _empty = dict(adl_level=np.nan, b20_pct=np.nan, zero_gamma=np.nan,
                  pc_ratio=np.nan, skew=np.nan)

    def _get_manual(dt):
        if not inp_map: return _empty
        if dt in inp_map: return inp_map[dt]
        prior = [d for d in inp_map if d < dt]
        return inp_map[max(prior)] if prior else _empty

    # ── 4. Append new rows ────────────────────────────────────────────────────
    new_dates = [d for d in close.index if d > last_date]
    new_rows  = []
    for dt in new_dates:
        row = {col: np.nan for col in HIST_COLS}
        row['date'] = dt
        m = _get_manual(dt)

        if dt in close.index:
            row['spy']  = float(close.loc[dt, 'spy'])  if 'spy'  in close.columns else np.nan
            row['spx']  = float(close.loc[dt, 'spx'])  if 'spx'  in close.columns else np.nan
            row['vix']  = float(close.loc[dt, 'vix'])  if 'vix'  in close.columns else np.nan
            yf_skew     = float(close.loc[dt, 'skew']) if 'skew' in close.columns else np.nan
            row['skew'] = yf_skew if not np.isnan(yf_skew) else m['skew']

        if dt in pc_series.index:
            row['pc_ratio'] = float(pc_series.loc[dt])
        elif not np.isnan(m['pc_ratio']):
            row['pc_ratio'] = m['pc_ratio']

        row['adl_level']    = m['adl_level']
        row['b20_pct']      = m['b20_pct']
        row['zero_gamma']   = m['zero_gamma']
        row['spike_flag']   = 0
        row['compressed']   = 0
        row['trigger_days'] = 0.0
        new_rows.append(row)

    if new_rows:
        hist = pd.concat([hist, pd.DataFrame(new_rows)], ignore_index=True)
        hist = hist.sort_values('date').reset_index(drop=True)
        print(f'  Appended {len(new_rows)} new row(s).')
    else:
        print('  No new market dates to append.')

    # ── 4b. Retroactive carry-forward patch ───────────────────────────────────
    manual_cols = ['adl_level', 'b20_pct', 'zero_gamma']
    hist = hist.set_index('date')
    patched = 0
    for hist_date in hist.index:
        m = _get_manual(hist_date)
        for col in manual_cols:
            val = m.get(col, np.nan)
            if not np.isnan(val):
                old = hist.loc[hist_date, col]
                if pd.isna(old) or old != val:
                    hist.loc[hist_date, col] = val
                    patched += 1
        if not np.isnan(m['pc_ratio']) and pd.isna(hist.loc[hist_date, 'pc_ratio']):
            hist.loc[hist_date, 'pc_ratio'] = m['pc_ratio']
        if not np.isnan(m['skew']) and pd.isna(hist.loc[hist_date, 'skew']):
            hist.loc[hist_date, 'skew'] = m['skew']
    hist = hist.reset_index()
    if patched:
        print(f'  Retroactive patch: {patched} field(s) corrected.')

    # ── 5. Derived signals ────────────────────────────────────────────────────
    print('  Computing derived signals...')
    spy  = hist['spy'].astype(float)
    vix  = hist['vix'].astype(float)
    skew = hist['skew'].astype(float)
    pc   = hist['pc_ratio'].astype(float)
    adl  = hist['adl_level'].astype(float)
    b20  = hist['b20_pct'].astype(float)

    hist['sma50']    = spy.rolling(50, min_periods=1).mean()
    hist['ext_raw']  = (spy - hist['sma50']) / hist['sma50']
    hist['mom_raw']  = spy.pct_change(20, fill_method=None)
    hist['pc_sma10'] = pc.rolling(10, min_periods=1).mean()
    hist['pc_sma20'] = pc.rolling(20, min_periods=1).mean()
    hist['pc_sma50'] = pc.rolling(50, min_periods=1).mean()
    adl_prev = adl.shift(20)
    hist['adl_roc20'] = np.where(adl_prev.abs() > 1e-9,
                                 (adl - adl_prev) / adl_prev.abs(), np.nan)

    # ── 6. Rolling Phi ────────────────────────────────────────────────────────
    print('  Computing Phi...')
    hist['vix_phi']  = rolling_phi(vix,  PHI_W)
    hist['ext_phi']  = rolling_phi(hist['ext_raw'].astype(float), PHI_W)
    hist['mom_phi']  = rolling_phi(hist['mom_raw'].astype(float), PHI_W)
    hist['skew_phi'] = rolling_phi(skew, PHI_W)
    hist['adl_phi']  = rolling_phi(hist['adl_roc20'].astype(float), PHI_W)
    hist['b20_phi']  = rolling_phi(b20, PHI_W)

    # ── 7. VIX flags ──────────────────────────────────────────────────────────
    vix_chg = vix.pct_change(fill_method=None)
    hist['spike_flag'] = (vix_chg > 0.30).fillna(False).astype(int)
    vix_phi_s       = hist['vix_phi']
    compressed_flag = (vix_phi_s < 0.30).fillna(False).astype(int)
    crossed         = ((compressed_flag.shift(1) == 1) & (vix_phi_s >= 0.30)).fillna(False).astype(int)
    trig            = np.zeros(len(hist), dtype=float)
    count           = 0
    for i in range(len(hist)):
        if crossed.iloc[i]:                            count = 1
        elif compressed_flag.iloc[i] == 0 and count:  count += 1
        else:                                          count = 0
        trig[i] = count if 0 < count <= 7 else 0
    hist['trigger_days'] = trig
    hist['compressed']   = compressed_flag

    # ── 8. Score every row ────────────────────────────────────────────────────
    print('  Scoring...')
    score_cols = ['vix_score','ext_score','mom_score','adl_score',
                  'b20_score','pc_score','skew_score','gamma_score']
    state_cols = ['vix_state','ext_state','mom_state','adl_state',
                  'b20_state','pc_state','skew_state','gamma_state']
    res = {c: [] for c in score_cols + state_cols + ['mrs_score']}

    for _, row in hist.iterrows():
        def g(c):
            v = row[c]; return v if not pd.isna(v) else np.nan

        vs, vst  = score_vix(g('vix_phi'))
        es, est  = score_extension(g('ext_phi'))
        ms, mst  = score_momentum(g('mom_phi'))
        as_, ast = score_adl(g('adl_phi'))
        bs, bst  = score_b20(g('b20_phi'), g('adl_phi'))
        ps, pst  = score_pc(g('pc_ratio'), g('pc_sma10'))
        ss, sst  = score_skew(g('skew_phi'), g('pc_ratio'))
        gs, gst  = score_gamma(g('spx'), g('zero_gamma'))

        scores = [vs, es, ms, as_, bs, ps, ss, gs]
        states = [vst, est, mst, ast, bst, pst, sst, gst]
        mrs    = round(sum(c for c in scores if not np.isnan(c)), 2)

        for col, val in zip(score_cols, scores):  res[col].append(val)
        for col, val in zip(state_cols, states):  res[col].append(val)
        res['mrs_score'].append(mrs)

    for col in score_cols + state_cols + ['mrs_score']:
        hist[col] = res[col]

    print(f'  Done. Latest MRS: {hist["mrs_score"].iloc[-1]:+.2f} — {regime_label(hist["mrs_score"].iloc[-1])}')
    return hist
