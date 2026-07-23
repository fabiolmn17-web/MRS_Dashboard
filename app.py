"""
app.py — MRS Live Dashboard (Streamlit)
========================================
Password-protected. Reads mrs_history.csv and displays:
  • Current regime score + signal quality
  • 8-component breakdown table
  • VIX lifecycle state layer
  • Zero Gamma position
  • 90-day MRS history chart + SPX close panel + VIX state panel
  • S&P 500 Sector Performance & Relative Strength
  • CAN SLIM Stock Scanner
"""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import requests
import yfinance as yf
from pathlib import Path
from datetime import date
from itertools import groupby
import pipeline

# ── Sector RS — constants & helpers ───────────────────────────────────────────
# yfinance sector strings → our SECTOR_MAP display names
YF_TO_SECTOR = {
    'Technology':             'Technology',
    'Communication Services': 'Comm Services',
    'Consumer Cyclical':      'Consumer Disc',
    'Consumer Defensive':     'Consumer Staples',
    'Energy':                 'Energy',
    'Financial Services':     'Financials',
    'Healthcare':             'Health Care',
    'Industrials':            'Industrials',
    'Basic Materials':        'Materials',
    'Real Estate':            'Real Estate',
    'Utilities':              'Utilities',
}

SECTOR_MAP = {
    'XLK':  'Technology',
    'XLC':  'Comm Services',
    'XLY':  'Consumer Disc',
    'XLP':  'Consumer Staples',
    'XLE':  'Energy',
    'XLF':  'Financials',
    'XLV':  'Health Care',
    'XLI':  'Industrials',
    'XLB':  'Materials',
    'XLRE': 'Real Estate',
    'XLU':  'Utilities',
}

@st.cache_data(ttl=3600)
def load_scanner_results():
    """Load latest scanner results from StockScanner/output/scan_results.csv."""
    try:
        csv_path = Path(__file__).parent / 'StockScanner' / 'output' / 'scan_results.csv'
        if not csv_path.exists():
            return None, None
        df = pd.read_csv(csv_path)
        if df.empty:
            return None, None
        scan_date = df['scan_date'].iloc[0] if 'scan_date' in df.columns else 'unknown'
        return df, scan_date
    except Exception as e:
        print(f'[scanner] load error: {e}')
        return None, None


@st.cache_data(ttl=3600)
def load_sector_data():
    """Fetch ~14 months of daily closes for 11 SPDR ETFs + SPY. Cached 1 hour."""
    tickers = list(SECTOR_MAP.keys()) + ['SPY']
    try:
        raw = yf.download(tickers, period='14mo', auto_adjust=True, progress=False)
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        closes.index = pd.to_datetime(closes.index).normalize().tz_localize(None)
        return closes.dropna(how='all')
    except Exception as e:
        print(f'[sector] fetch error: {e}')
        return None

def _sector_composite_label(score):
    if score >=  1.5: return 'STRUCTURAL LEADER'
    if score >=  1.0: return 'STRONG OUTPERFORMER'
    if score >=  0.5: return 'OUTPERFORMING'
    if score <= -1.5: return 'STRUCTURAL LAGGARD'
    if score <= -1.0: return 'STRONG UNDERPERFORMER'
    if score <= -0.5: return 'UNDERPERFORMING'
    return 'NEUTRAL'

def build_sector_table(closes):
    """Compute absolute returns + RS vs SPY, matching Pine Script formula exactly."""
    if closes is None or 'SPY' not in closes.columns:
        return None
    spy = closes['SPY'].dropna()
    spy_now = spy.iloc[-1]
    current_year = closes.index[-1].year
    prev_year = closes[closes.index.year < current_year]

    rows = []
    for etf, sector in SECTOR_MAP.items():
        if etf not in closes.columns:
            continue
        s = closes[etf].dropna()
        if len(s) < 63:
            continue
        p = s.iloc[-1]
        p_prev = s.iloc[-2] if len(s) >= 2 else p
        daily_chg = (p / p_prev - 1) if p_prev != 0 else 0.0

        ytd_base = prev_year[etf].dropna().iloc[-1] if len(prev_year) > 0 and etf in prev_year else s.iloc[0]

        def ret(n):
            return (p / s.iloc[-n] - 1) if len(s) > n else np.nan
        def rs(n):
            if len(s) <= n or len(spy) <= n: return np.nan
            return (p / s.iloc[-n]) - (spy_now / spy.iloc[-n])

        r_ytd = p / ytd_base - 1
        r_1y, r_6m, r_3m = ret(252), ret(126), ret(63)
        rs_1y, rs_6m, rs_3m = rs(252), rs(126), rs(63)

        sc = lambda v: (0.5 if v > 0 else -0.5 if v < 0 else 0.0) if not (v is None or np.isnan(v)) else 0.0
        composite = sc(rs_1y) + sc(rs_6m) + sc(rs_3m)

        # Momentum trend: sign shift between RS 6M and RS 3M
        def _sgn(v): return 1 if (v is not None and not np.isnan(v) and v > 0) else (-1 if (v is not None and not np.isnan(v) and v < 0) else 0)
        s6, s3 = _sgn(rs_6m), _sgn(rs_3m)
        if   s3 > s6:  trend = 'IMPROVING'
        elif s3 < s6:  trend = 'FADING'
        elif s3 > 0:   trend = 'STABLE+'
        elif s3 < 0:   trend = 'STABLE-'
        else:           trend = 'NEUTRAL'

        rows.append({
            'Sector': sector, 'ETF': etf, 'Close': p, 'DailyChg': daily_chg,
            'YTD': r_ytd, '1Y': r_1y, '6M': r_6m, '3M': r_3m,
            'RS 1Y': rs_1y, 'RS 6M': rs_6m, 'RS 3M': rs_3m,
            'Score': composite, 'Label': _sector_composite_label(composite),
            'Trend': trend,
        })

    df = pd.DataFrame(rows).sort_values('Score', ascending=False).reset_index(drop=True)
    return df

# ── trigger GitHub Actions backfill workflow ───────────────────────────────────
def _trigger_backfill(gh_token: str, b20: float, adl: float,
                      zg: float, pc: float) -> bool:
    url = ('https://api.github.com/repos/fabiolmn17-web/'
           'MRS_Dashboard/actions/workflows/backfill.yml/dispatches')
    headers = {
        'Authorization': f'token {gh_token}',
        'Accept': 'application/vnd.github.v3+json',
    }
    payload = {
        'ref': 'main',
        'inputs': {
            'b20_pct':    str(b20),
            'adl_tv':     str(adl),
            'zero_gamma': str(zg),
            'pc_ratio':   str(pc) if pc > 0 else '',
        },
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        return r.status_code == 204
    except Exception as e:
        print(f'Workflow trigger error: {e}')
        return False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='MRS Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── Password gate ──────────────────────────────────────────────────────────────
def check_password() -> bool:
    if st.session_state.get('authenticated'):
        return True
    st.markdown('## MRS Dashboard')
    pwd = st.text_input('Password', type='password', key='pwd_input')
    if st.button('Enter'):
        expected = st.secrets.get('APP_PASSWORD', '')
        if pwd == expected and expected:
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error('Incorrect password.')
    return False

if not check_password():
    st.stop()

# ── Load data (before sidebar so we can show last date + live defaults) ────────
HIST_PATH = Path(__file__).parent / 'mrs_history.csv'
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/fabiolmn17-web/MRS_Dashboard/main/mrs_history.csv'

@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    """Load history from GitHub raw URL (always fresh), fallback to local file."""
    try:
        df = pd.read_csv(GITHUB_RAW_URL, parse_dates=['date'])
        for col in pipeline.HIST_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f'[load_data] GitHub fetch failed ({e}), falling back to local file')
        return pipeline.load_history(HIST_PATH)

hist = load_data()

def _last_val(col, default=np.nan):
    """Return most recent non-NaN value from hist for a column."""
    if col not in hist.columns:
        return default
    s = hist[col].dropna()
    return float(s.iloc[-1]) if len(s) > 0 else default

# ── Sidebar daily-input form ───────────────────────────────────────────────────
with st.sidebar:
    # Last data date banner
    _csv_last = hist['date'].max()
    _today    = pd.Timestamp(date.today())
    _days_old = len(pd.bdate_range(end=_today, start=_csv_last)) - 1
    if _days_old <= 0:
        st.sidebar.success(f'Data current: {_csv_last.strftime("%b %d %Y")}')
    elif _days_old == 1:
        st.sidebar.warning(f'Data through {_csv_last.strftime("%b %d %Y")} — 1 session behind')
    else:
        st.sidebar.error(f'Data through {_csv_last.strftime("%b %d %Y")} — {_days_old} sessions behind')

    st.markdown('### Daily Inputs')
    st.caption('Enter after market close (4 PM ET). ADL: TradingView value — auto x1000.')

    # Pre-populate: prefer URL params (survive reload) then CSV last value
    _qp = st.query_params
    _def_b20 = float(_qp['b20']) if 'b20' in _qp else _last_val('b20_pct',    50.0)
    _def_adl = float(_qp['adl']) if 'adl' in _qp else _last_val('adl_level',  1_827_000.0) / 1000.0
    _def_zg  = float(_qp['zg'])  if 'zg'  in _qp else _last_val('zero_gamma', 7_400.0)

    with st.form('daily_inputs_form'):
        inp_b20 = st.number_input('B20% (S5TW)',            min_value=0.0,  max_value=100.0,
                                   value=round(_def_b20, 2), step=0.01)
        inp_adl = st.number_input('ADL (TradingView x1000)', value=round(_def_adl, 2), step=0.01)
        inp_zg  = st.number_input('Zero Gamma (SPX level)',  value=round(_def_zg, 2),  step=1.0)
        inp_pc  = st.number_input('PC Ratio (0 = auto)',     value=0.0, step=0.001,
                                   min_value=0.0, max_value=3.0,
                                   help='Leave 0 to auto-fetch. Valid range: 0.3 – 2.0.')
        submitted = st.form_submit_button('Submit & Update')

    if submitted:
        # ── Input validation ───────────────────────────────────────────────────
        _errors = []
        if inp_b20 <= 0:
            _errors.append('B20% must be > 0.')
        if inp_b20 > 100:
            _errors.append('B20% must be <= 100.')
        if inp_adl < 100 or inp_adl > 10_000:
            _errors.append(f'ADL {inp_adl:.0f} looks wrong (expected 500–5000 range).')
        if inp_zg < 2_000 or inp_zg > 15_000:
            _errors.append(f'Zero Gamma {inp_zg:.0f} is outside a valid SPX range.')
        if inp_pc > 3.0:
            _errors.append(f'PC Ratio {inp_pc:.3f} looks like an SPX price, not a ratio. Use 0 for auto-fetch.')

        if _errors:
            for e in _errors:
                st.sidebar.error(e)
        else:
            gh_token = st.secrets.get('GITHUB_TOKEN', '')
            if not gh_token:
                st.sidebar.error('GITHUB_TOKEN not in Streamlit secrets.')
            else:
                # Persist in URL params (survives reload) + session_state (live preview)
                st.query_params['b20'] = str(inp_b20)
                st.query_params['adl'] = str(inp_adl)
                st.query_params['zg']  = str(inp_zg)
                if inp_pc > 0:
                    st.query_params['pc'] = str(inp_pc)
                st.session_state['pending_b20']  = inp_b20
                st.session_state['pending_adl']  = inp_adl * 1000  # ×1000 → CSV scale
                st.session_state['pending_zg']   = inp_zg
                st.session_state['pending_pc']   = inp_pc
                st.session_state['pending_date'] = _today.strftime('%Y-%m-%d')
                ok = _trigger_backfill(gh_token, inp_b20, inp_adl, inp_zg, inp_pc)
                if ok:
                    st.sidebar.success(
                        f'Update triggered for {_today.strftime("%b %d")}.\n'
                        'Wait ~90 sec, then hit Refresh Data.'
                    )
                else:
                    st.sidebar.error('API call failed — check GITHUB_TOKEN.')

    st.sidebar.divider()
    if st.sidebar.button('Refresh Data', use_container_width=True):
        for _k in ['pending_b20','pending_adl','pending_zg','pending_pc','pending_date']:
            st.session_state.pop(_k, None)
        for _qk in ['b20','adl','zg','pc']:
            st.query_params.pop(_qk, None)
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption('Clears cache and reloads CSV from disk.')
complete = hist.dropna(subset=['vix', 'mrs_score'])
last     = complete.iloc[-1].to_dict() if len(complete) else hist.iloc[-1].to_dict()
last_dt  = pd.Timestamp(last['date'])
last_dt_str = last_dt.strftime('%Y-%m-%d')  # For plotly vlines

# Fill price-derived fields from most recent row with valid SPX close + valid ext_phi
if pd.isna(last.get('spx', np.nan)) or pd.isna(last.get('ext_phi', np.nan)):
    price_hist = hist.dropna(subset=['spx', 'ext_phi'])
    if len(price_hist) > 0:
        price_last = price_hist.iloc[-1]
        for col in ['spx', 'spy', 'ext_raw', 'ext_phi', 'ext_score', 'ext_state',
                    'mom_raw', 'mom_phi', 'mom_score', 'mom_state', 'mrs_score']:
            if col in price_last.index:
                last[col] = price_last[col]

# Fill manual inputs from most recent non-NaN row for each
for col in ['b20_pct', 'b20_phi', 'b20_score', 'b20_state',
            'adl_level', 'adl_phi', 'adl_score', 'adl_state',
            'zero_gamma', 'gamma_score', 'gamma_state']:
    if pd.isna(last.get(col, np.nan)) and col in hist.columns:
        valid = hist[hist[col].notna()]
        if len(valid) > 0:
            last[col] = valid.iloc[-1][col]

# SKEW lags one day from yfinance — carry forward when today's is missing
if pd.isna(last.get('skew', np.nan)) and 'skew' in hist.columns:
    valid_skew = hist[hist['skew'].notna()]
    if len(valid_skew) > 0:
        for col in ['skew', 'skew_phi', 'skew_score', 'skew_state']:
            if col in valid_skew.columns:
                last[col] = valid_skew.iloc[-1][col]

# ── Live preview: overlay today's submitted inputs ────────────────────────────
_preview_active = False
if st.session_state.get('pending_date') == date.today().strftime('%Y-%m-%d'):
    last['b20_pct']    = st.session_state['pending_b20']
    last['adl_level']  = st.session_state['pending_adl']
    last['zero_gamma'] = st.session_state['pending_zg']
    _pc_sub = st.session_state['pending_pc']
    if _pc_sub > 0:
        last['pc_ratio'] = _pc_sub
    # Recompute scores for components directly sensitive to manual inputs
    try:
        _gs, _gst = pipeline.score_gamma(float(last.get('spx', np.nan)), float(last['zero_gamma']))
        last['gamma_score'] = _gs
        last['gamma_state'] = _gst
        _ps, _pst = pipeline.score_pc(float(last.get('pc_ratio', np.nan)), float(last.get('pc_sma10', np.nan)))
        last['pc_score'] = _ps
        last['pc_state'] = _pst
        # Recompute weighted mrs_score estimate
        _wts = pipeline.COMPONENT_WEIGHTS
        _sc_map = {
            'vix': 'vix_score', 'ext': 'ext_score', 'mom': 'mom_score',
            'adl': 'adl_score', 'b20': 'b20_score', 'pc': 'pc_score',
            'skew': 'skew_score', 'gamma': 'gamma_score', 'vol': 'vol_score',
        }
        _wsum = sum(
            float(last.get(sc, 0) or 0) * _wts[k]
            for k, sc in _sc_map.items()
            if not np.isnan(float(last.get(sc, 0) or 0))
        )
        last['mrs_score'] = round(_wsum, 2)
    except Exception:
        pass
    _preview_active = True

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #252538;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
  }
  .regime-label {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.04em;
  }
  .score-number {
    font-size: 3.6rem;
    font-weight: 900;
    line-height: 1;
  }
  .quality-chip {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .section-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #d1d5db;
    margin-bottom: 8px;
    margin-top: 20px;
  }
  .hazard-row {
    background: #3a2020;
    border-left: 4px solid #ef4444;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.90rem;
    color: #fca5a5;
  }
  .safe-row {
    background: #1f3520;
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.90rem;
    color: #86efac;
  }
  .neutral-row {
    background: #23283a;
    border-left: 4px solid #8892a4;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.90rem;
    color: #c4cad6;
  }
  div[data-testid="stDataFrame"] { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ───────────────────────────────────────────────────────────
def _f(key, fmt='{:.2f}'):
    v = last.get(key, np.nan)
    try:
        fv = float(v)
        return '—' if np.isnan(fv) else fmt.format(fv)
    except:
        return str(v) if v else '—'

QUALITY_COLORS = {
    'CONFIRMED':              '#22c55e',
    'NEUTRAL — BULLISH LEAN': '#22c55e',
    'NEUTRAL — BEARISH LEAN': '#f97316',
    'NEUTRAL — NO EDGE':      '#9ca3af',
    'UNCONFIRMED':            '#f97316',
    'DIVERGENT':              '#ef4444',
    'FRAGILE':                '#facc15',
}

def quality_color(label: str) -> str:
    for k, v in QUALITY_COLORS.items():
        if k in label:
            return v
    return '#9ca3af'

# ══════════════════════════════════════════════════════════════════════════════
# HEADER ROW
# ══════════════════════════════════════════════════════════════════════════════
mrs    = float(last.get('mrs_score', 0) or 0)
reg    = pipeline.regime_label(mrs)
rcol   = pipeline.regime_color(mrs)
dur    = pipeline.compute_regime_duration(hist, last_dt)
sq_lbl, sq_desc, sq_hex = pipeline.compute_signal_quality(last, hist, last_dt)
sq_col = '#' + sq_hex

col_score, col_regime, col_sq, col_dur = st.columns([1, 2, 2.5, 1])

with col_score:
    st.markdown(f"""
    <div class="metric-card">
      <div class="section-header">MRS Score</div>
      <div class="score-number" style="color:{rcol};">{mrs:+.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_regime:
    st.markdown(f"""
    <div class="metric-card">
      <div class="section-header">Regime — {last_dt.strftime('%b %d %Y')}</div>
      <div class="regime-label" style="color:{rcol};">{reg}</div>
      <div style="font-size:0.78rem;color:#9ca3af;margin-top:4px;">
        {dur} consecutive session{'s' if dur != 1 else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_sq:
    st.markdown(f"""
    <div class="metric-card">
      <div class="section-header">Signal Quality</div>
      <span class="quality-chip" style="background:{sq_col}20;color:{sq_col};border:1px solid {sq_col};">
        {sq_lbl}
      </span>
      <div style="font-size:0.78rem;color:#9ca3af;margin-top:8px;line-height:1.5;">
        {sq_desc}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_dur:
    vix_now = _f('vix', '{:.2f}')
    zg_now  = _f('zero_gamma', '{:,.0f}')
    spx_now = _f('spx', '{:,.0f}')
    st.markdown(f"""
    <div class="metric-card">
      <div class="section-header">Levels</div>
      <div style="font-size:0.82rem;line-height:2;">
        <b>VIX</b> {vix_now}<br>
        <b>SPX</b> {spx_now}<br>
        <b>Zero γ</b> {zg_now}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.4, 1], gap='large')

with left:
    if _preview_active:
        st.markdown(
            '<div style="background:rgba(161,98,7,0.15);border:1px solid #a16207;border-radius:6px;'
            'padding:7px 12px;margin-bottom:8px;font-size:0.78rem;color:#fbbf24;">'
            '📥 Showing today\'s submitted inputs — CSV update pending (~90 sec, then Refresh Data)'
            '</div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="section-header">Component Breakdown</div>', unsafe_allow_html=True)
    COMP_DEF = [
        ('VIX',        'vix_phi',  'vix_score',  'vix_state',   'vix'),
        ('Extension',  'ext_phi',  'ext_score',  'ext_state',   'ext_raw'),
        ('Momentum',   'mom_phi',  'mom_score',  'mom_state',   'mom_raw'),
        ('ADL Trend',  'adl_phi',  'adl_score',  'adl_state',   'adl_level'),
        ('B20%',       'b20_phi',  'b20_score',  'b20_state',   'b20_pct'),
        ('PC Ratio',   None,       'pc_score',   'pc_state',    'pc_sma10'),
        ('SKEW',       'skew_phi', 'skew_score', 'skew_state',  'skew'),
        ('Zero Gamma', None,       'gamma_score','gamma_state', 'zero_gamma'),
        ('Volume',     None,       'vol_score',  'vol_state',   'volume'),
    ]

    # Compute volume vs 20d average for display
    vol_vs_20d = '—'
    if 'volume' in hist.columns:
        vol_series = hist['volume'].astype(float)
        vol_20d_avg = vol_series.rolling(20, min_periods=1).mean()
        if len(vol_series) > 0 and len(vol_20d_avg) > 0:
            last_vol = vol_series.iloc[-1]
            last_avg = vol_20d_avg.iloc[-1]
            if not np.isnan(last_vol) and not np.isnan(last_avg) and last_avg > 0:
                pct_diff = (last_vol / last_avg - 1) * 100
                vol_vs_20d = f'vs 20d: {pct_diff:+.0f}%'

    rows = []
    for name, phi_key, sc_key, st_key, raw_key in COMP_DEF:
        phi_v = last.get(phi_key, np.nan) if phi_key else np.nan
        sc_v  = last.get(sc_key, np.nan)
        st_v  = last.get(st_key, '—') or '—'
        raw_v = last.get(raw_key, np.nan)
        try: phi_f = f'{float(phi_v):.3f}' if not np.isnan(float(phi_v)) else '—'
        except: phi_f = '—'
        try:
            sc_float = float(sc_v)
            sc_f = f'{sc_float:+.1f}' if not np.isnan(sc_float) else '—'
        except:
            sc_f = '—'
        try:
            raw_float = float(raw_v)
            if np.isnan(raw_float):
                raw_f = '—'
            elif name == 'Volume':
                raw_f = f'{raw_float/1e6:.1f}M'  # Format volume in millions
            else:
                raw_f = f'{raw_float:,.2f}'
        except:
            raw_f = '—'
        # Fix state display for NaN
        if st_v == 'nan' or (isinstance(st_v, float) and np.isnan(st_v)):
            st_v = '—'
        # Special handling for Volume state - show vs 20d avg
        if name == 'Volume':
            st_v = vol_vs_20d
        rows.append({'Component': name, 'Raw Value': raw_f,
                     'Phi': phi_f, 'State': str(st_v), 'Score': sc_f})
    df_comp = pd.DataFrame(rows)
    def color_score(val):
        try:
            v = float(val)
            if v > 0: return 'color: #22c55e; font-weight: 700'
            if v < 0: return 'color: #ef4444; font-weight: 700'
            return 'color: #6b7280'
        except:
            return ''
    styled = df_comp.style.map(color_score, subset=['Score'])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#6b7280;margin-top:4px;">
    Phi = percentile rank over rolling 756-session window (3 years).
    Score = discretized contribution to the MRS composite.
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-header">VIX Lifecycle State</div>', unsafe_allow_html=True)
    vix_phi   = last.get('vix_phi', np.nan)
    trig_days = float(last.get('trigger_days', 0) or 0)
    try: vix_phi_f = float(vix_phi)
    except: vix_phi_f = np.nan

    if not np.isnan(vix_phi_f) and vix_phi_f < 0.30:
        vix_row_cls = 'hazard-row'
        vix_txt = (f'🔴 COMPRESSION — VIX Phi = {vix_phi_f:.3f}. Latent fragility state. '
                   f'Exit events historically elevate VIX +8% (5D, d=+0.46) and suppress SPY (d=−0.32).')
    elif not np.isnan(vix_phi_f) and vix_phi_f > 0.70:
        vix_row_cls = 'neutral-row'
        vix_txt = (f'🟡 SPIKE ZONE — VIX Phi = {vix_phi_f:.3f}. Elevated VIX. '
                   f'Post-spike SPY recovery: mean +2.91% vs +0.38% baseline (10D). Left-tail contracted.')
    elif not np.isnan(vix_phi_f):
        vix_row_cls = 'safe-row'
        vix_txt = f'🟢 MID RANGE — VIX Phi = {vix_phi_f:.3f}. Normal expansion phase. No structural signal.'
    else:
        vix_row_cls = 'neutral-row'
        vix_txt = '⚪ VIX Phi: no data'

    st.markdown(f'<div class="{vix_row_cls}">{vix_txt}</div>', unsafe_allow_html=True)

    if trig_days > 0:
        days_left = int(7 - trig_days)
        st.markdown(f'<div class="hazard-row">🔴 COMPRESSION EXIT — Day {int(trig_days)} of 7. '
                    f'Active SPY suppression window (d=−0.32 at 5D, persists to ~21D). '
                    f'{days_left}d remaining in tracking window.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe-row">🟢 No active compression exit event</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:16px;">Zero Gamma Position</div>',
                unsafe_allow_html=True)
    try:
        spx_v = float(last.get('spx', np.nan))
        zg_v  = float(last.get('zero_gamma', np.nan))
        if not np.isnan(spx_v) and not np.isnan(zg_v) and zg_v > 0:
            dist_pct = (spx_v - zg_v) / spx_v * 100
            if dist_pct > 0.25:
                gcls = 'safe-row'
                gtxt = f'🟢 SPX {spx_v:,.0f} is {dist_pct:.1f}% ABOVE zero-gamma ({zg_v:,.0f}). Dealers long gamma — dampening environment.'
            elif dist_pct > -0.25:
                gcls = 'neutral-row'
                gtxt = f'🟡 SPX {spx_v:,.0f} is NEAR zero-gamma ({zg_v:,.0f}, {dist_pct:+.1f}%). Transition zone — regime could flip.'
            else:
                gcls = 'hazard-row'
                gtxt = f'🔴 SPX {spx_v:,.0f} is {abs(dist_pct):.1f}% BELOW zero-gamma ({zg_v:,.0f}). Dealers short gamma — amplifying moves.'
            st.markdown(f'<div class="{gcls}">{gtxt}</div>', unsafe_allow_html=True)
        else:
            zg_only = float(last.get('zero_gamma', np.nan))
            if not np.isnan(zg_only) and zg_only > 0:
                st.markdown(f'<div class="neutral-row">⚪ Zero Gamma level: <b>{zg_only:,.0f}</b> — SPX close pending</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="neutral-row">⚪ Zero Gamma: no data today</div>', unsafe_allow_html=True)
    except:
        try:
            zg_only = float(last.get('zero_gamma', np.nan))
            if not np.isnan(zg_only) and zg_only > 0:
                st.markdown(f'<div class="neutral-row">⚪ Zero Gamma level: <b>{zg_only:,.0f}</b> — SPX close pending</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="neutral-row">⚪ Zero Gamma: no data today</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="neutral-row">⚪ Zero Gamma: no data today</div>', unsafe_allow_html=True)

    # ── Recovery Signal (MRS v2.0) ─────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:16px;">Recovery Signal</div>',
                unsafe_allow_html=True)
    recovery = pipeline.compute_recovery_signal(last, hist, last_dt)
    rec_color = '#' + recovery['color']

    if recovery['active']:
        rec_cls = 'safe-row'
        rec_icon = '🟢'
        rec_title = f"RECOVERY — {recovery['strength']}"
    elif recovery['signals'] >= 1:
        rec_cls = 'neutral-row'
        rec_icon = '🟡'
        rec_title = f"EARLY SIGNS ({recovery['signals']} signal{'s' if recovery['signals'] > 1 else ''})"
    else:
        rec_cls = 'neutral-row'
        rec_icon = '⚪'
        rec_title = 'NO SIGNAL'

    st.markdown(f'''
    <div class="{rec_cls}">
      <span style="font-weight:700;color:{rec_color};">{rec_icon} {rec_title}</span><br>
      <span style="font-size:0.82rem;color:#9ca3af;">{recovery["description"]}</span>
    </div>
    ''', unsafe_allow_html=True)

    if recovery['components']:
        comp_list = ' • '.join(recovery['components'])
        st.markdown(f'''
        <div style="font-size:0.75rem;color:#6b7280;margin-top:4px;padding-left:8px;">
          Components: {comp_list}
        </div>
        ''', unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
hist90  = hist.dropna(subset=['mrs_score']).tail(90).copy()
hist90['date'] = pd.to_datetime(hist90['date'])
x_min   = hist.tail(90)['date'].min()
x_max   = hist.tail(90)['date'].max()
x_range = [x_min, x_max]

VLINE_STYLE = dict(line_dash='dot', line_color='rgba(250,204,21,0.6)', line_width=1.5)
LAYOUT_BASE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    xaxis=dict(showgrid=False, tickformat='%b %d', tickfont_size=11, range=x_range),
)

# ── 1. 90-Day MRS History ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">90-Day MRS History</div>', unsafe_allow_html=True)

fig = go.Figure()
band_defs = [
    (1.5,  5.0,  'rgba(26,127,55,0.12)',  'RISK-ON'),
    (0.5,  1.5,  'rgba(87,166,107,0.10)', 'MILD RISK-ON'),
    (-0.5, 0.5,  'rgba(107,114,128,0.08)','NEUTRAL'),
    (-1.5, -0.5, 'rgba(217,119,6,0.10)',  'MILD RISK-OFF'),
    (-5.0, -1.5, 'rgba(185,28,28,0.12)',  'RISK-OFF'),
]
for y0, y1, fill, label in band_defs:
    fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0)
    fig.add_annotation(
        x=1.01, xref='paper',
        y=(y0 + y1) / 2, yref='y',
        text=label, showarrow=False,
        font=dict(size=9, color='#6b7280'),
        xanchor='left',
    )

fig.add_trace(go.Scatter(
    x=hist90['date'], y=hist90['mrs_score'],
    mode='lines+markers', name='MRS',
    line=dict(color='#60a5fa', width=2.5),
    marker=dict(size=4, color='#60a5fa'),
    hovertemplate='<b>%{x|%b %d}</b><br>MRS: %{y:+.2f}<extra></extra>',
))
fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.25)', line_width=1)
fig.add_shape(type='line', x0=last_dt_str, x1=last_dt_str, y0=-4.5, y1=4.5,
              line=dict(dash='dot', color='rgba(250,204,21,0.6)', width=1.5))
fig.add_annotation(x=last_dt_str, y=4.2, text='Today', showarrow=False,
                   font=dict(size=10, color='#facc15'), xanchor='left')
fig.update_layout(**LAYOUT_BASE,
    margin=dict(l=10, r=130, t=10, b=30), height=300,
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
               tickformat='+.1f', range=[-4.5, 4.5], tickfont_size=11),
)
st.plotly_chart(fig, use_container_width=True)

# ── 2. SPX Close + Zero Gamma line ────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:0; margin-bottom:4px;">SPX Close</div>',
            unsafe_allow_html=True)

hist90_spx       = hist.tail(90).copy()
hist90_spx_valid = hist90_spx.dropna(subset=['spx'])

if len(hist90_spx_valid) > 0:
    spx_min = hist90_spx_valid['spx'].min()
    spx_max = hist90_spx_valid['spx'].max()
    spx_pad = (spx_max - spx_min) * 0.10
    zg_val = np.nan
    try:
        zg_val = float(last.get('zero_gamma', np.nan))
    except:
        pass
    if not np.isnan(zg_val):
        spx_min = min(spx_min, zg_val)
        spx_max = max(spx_max, zg_val)

    fig_spx = go.Figure()
    fig_spx.add_trace(go.Scatter(
        x=hist90_spx['date'],
        y=hist90_spx['spx'],
        mode='lines', name='SPX',
        line=dict(color='#a78bfa', width=2),
        hovertemplate='<b>%{x|%b %d}</b><br>SPX: %{y:,.0f}<extra></extra>',
    ))
    if not np.isnan(zg_val):
        fig_spx.add_hline(
            y=zg_val,
            line_dash='dash',
            line_color='rgba(250,204,21,0.55)',
            line_width=1.2,
            annotation_text=f'Zero γ {zg_val:,.0f}',
            annotation_position='right',
            annotation_font_color='#facc15',
            annotation_font_size=10,
        )
    fig_spx.add_shape(type='line', x0=last_dt_str, x1=last_dt_str, y0=0, y1=1, yref='paper',
                  line=dict(dash='dot', color='rgba(250,204,21,0.6)', width=1.5))
    fig_spx.update_layout(**LAYOUT_BASE,
        margin=dict(l=10, r=130, t=4, b=30), height=180,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   tickformat=',.0f', tickfont_size=11,
                   range=[spx_min - spx_pad, spx_max + spx_pad]),
    )
    st.plotly_chart(fig_spx, use_container_width=True)

# ── 3. VIX State panel ────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:0; margin-bottom:4px;">VIX State</div>',
            unsafe_allow_html=True)

hist90_vix       = hist.tail(90).copy()
hist90_vix_valid = hist90_vix.dropna(subset=['vix'])

if len(hist90_vix_valid) > 0:
    vix_min = hist90_vix_valid['vix'].min()
    vix_max = hist90_vix_valid['vix'].max()
    vix_pad = (vix_max - vix_min) * 0.12

    fig_vix = go.Figure()
    if 'vix_phi' in hist90_vix_valid.columns:
        def _vix_state(phi):
            try:
                p = float(phi)
                if p < 0.30: return 'compression'
                if p > 0.70: return 'spike'
                return 'mid'
            except:
                return 'mid'
        hist90_vix_valid = hist90_vix_valid.copy()
        hist90_vix_valid['vstate'] = hist90_vix_valid['vix_phi'].apply(_vix_state)
        STATE_COLORS = {
            'compression': 'rgba(239,68,68,0.18)',
            'spike':       'rgba(250,204,21,0.14)',
            'mid':         'rgba(34,197,94,0.07)',
        }
        for state, grp in groupby(hist90_vix_valid.itertuples(index=False), key=lambda r: r.vstate):
            rows = list(grp)
            d0 = pd.Timestamp(rows[0].date)
            d1 = pd.Timestamp(rows[-1].date)
            fig_vix.add_vrect(x0=d0, x1=d1, fillcolor=STATE_COLORS[state], line_width=0)

    fig_vix.add_trace(go.Scatter(
        x=hist90_vix['date'],
        y=hist90_vix['vix'],
        mode='lines', name='VIX',
        line=dict(color='#f9a8d4', width=2),
        hovertemplate='<b>%{x|%b %d}</b><br>VIX: %{y:.2f}<extra></extra>',
    ))
    fig_vix.add_hline(y=20, line_dash='dot',
                      line_color='rgba(255,255,255,0.20)', line_width=1,
                      annotation_text='20', annotation_position='right',
                      annotation_font_color='#6b7280', annotation_font_size=10)
    fig_vix.add_shape(type='line', x0=last_dt_str, x1=last_dt_str, y0=0, y1=1, yref='paper',
                  line=dict(dash='dot', color='rgba(250,204,21,0.6)', width=1.5))
    fig_vix.update_layout(**LAYOUT_BASE,
        margin=dict(l=10, r=130, t=4, b=30), height=160,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   tickformat='.0f', tickfont_size=11,
                   range=[max(0, vix_min - vix_pad), vix_max + vix_pad]),
    )
    st.plotly_chart(fig_vix, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT HISTORY CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Component History</div>', unsafe_allow_html=True)

# Shared colour palette (consistent across both charts)
_COMP_COLORS = {
    'VIX':       '#60a5fa',   # blue
    'Extension': '#f97316',   # orange
    'Momentum':  '#4ade80',   # green
    'ADL':       '#c084fc',   # purple
    'B20%':      '#fbbf24',   # yellow
    'SKEW':      '#22d3ee',   # cyan
    'PC Ratio':  '#f472b6',   # pink
    'Gamma':     '#a3e635',   # lime
}

_PHI_MAP = [
    ('VIX',       'vix_phi'),
    ('Extension', 'ext_phi'),
    ('Momentum',  'mom_phi'),
    ('ADL',       'adl_phi'),
    ('B20%',      'b20_phi'),
    ('SKEW',      'skew_phi'),
]

_SCORE_MAP = [
    ('VIX',       'vix_score',   1.3),
    ('Extension', 'ext_score',   1.2),
    ('Momentum',  'mom_score',   1.0),
    ('ADL',       'adl_score',   1.0),
    ('B20%',      'b20_score',   1.1),
    ('PC Ratio',  'pc_score',    1.4),
    ('SKEW',      'skew_score',  1.3),
    ('Gamma',     'gamma_score', 1.0),
]

_hist_phi = hist90.copy()
# Only use rows where at least one phi is non-zero
_phi_cols = [c for _, c in _PHI_MAP if c in _hist_phi.columns]
_hist_phi = _hist_phi[(_hist_phi[_phi_cols] != 0).any(axis=1)]

# ── Chart A: Phi trajectories ─────────────────────────────────────────────────
_fig_phi = go.Figure()

# Threshold bands
_fig_phi.add_hrect(y0=0, y1=0.30, fillcolor='rgba(239,68,68,0.08)',
                   line_width=0, annotation_text='Danger zone',
                   annotation_position='top left',
                   annotation_font=dict(size=10, color='rgba(239,68,68,0.5)'))
_fig_phi.add_hrect(y0=0.70, y1=1.0, fillcolor='rgba(34,197,94,0.08)',
                   line_width=0, annotation_text='Positive zone',
                   annotation_position='bottom left',
                   annotation_font=dict(size=10, color='rgba(34,197,94,0.5)'))
_fig_phi.add_hline(y=0.30, line_dash='dot', line_color='rgba(239,68,68,0.4)', line_width=1)
_fig_phi.add_hline(y=0.70, line_dash='dot', line_color='rgba(34,197,94,0.4)', line_width=1)

for label, col in _PHI_MAP:
    if col not in _hist_phi.columns:
        continue
    _series = _hist_phi[col].replace(0, np.nan)
    _fig_phi.add_trace(go.Scatter(
        x=_hist_phi['date'], y=_series,
        mode='lines', name=label,
        line=dict(color=_COMP_COLORS[label], width=1.8),
        hovertemplate=f'<b>{label}</b>: %{{y:.3f}}<extra></extra>',
    ))

_fig_phi.update_layout(
    **LAYOUT_BASE,
    height=280,
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
               title='Percentile Rank (Φ)', title_font_size=11,
               tickformat='.2f', range=[-0.02, 1.02], tickfont_size=11),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(_fig_phi, use_container_width=True)

# ── Chart B: Weighted score contributions (stacked bar) ───────────────────────
st.markdown(
    '<p style="color:#6b7280;font-size:0.72rem;margin:4px 0 8px 0;">'
    'Score contribution = raw component score × weight. Bars stack to MRS composite.</p>',
    unsafe_allow_html=True,
)

_fig_sc = go.Figure()

for label, col, _w in _SCORE_MAP:
    if col not in _hist_phi.columns:
        continue
    _weighted = (_hist_phi[col] * _w).replace(0, np.nan)
    _fig_sc.add_trace(go.Bar(
        x=_hist_phi['date'], y=_weighted,
        name=label,
        marker_color=_COMP_COLORS.get(label, '#6b7280'),
        hovertemplate=f'<b>{label}</b>: %{{y:+.2f}}<extra></extra>',
    ))

# MRS composite line overlay
_fig_sc.add_trace(go.Scatter(
    x=_hist_phi['date'], y=_hist_phi['mrs_score'],
    mode='lines', name='MRS',
    line=dict(color='#ffffff', width=2, dash='dot'),
    hovertemplate='<b>MRS</b>: %{y:+.2f}<extra></extra>',
))
_fig_sc.add_hline(y=0, line_color='rgba(255,255,255,0.2)', line_width=1)

_fig_sc.update_layout(
    **LAYOUT_BASE,
    height=260,
    barmode='relative',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
               title='Weighted Score', title_font_size=11,
               tickformat='+.1f', tickfont_size=11),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(_fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PC RATIO CONTEXT (collapsible)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander('PC Ratio — Five-Zone Context'):
    pc_sma10 = last.get('pc_sma10', np.nan)
    pc_daily = last.get('pc_ratio', np.nan)
    try:
        pc10 = float(pc_sma10)
        if pc10 < 0.686:
            zone, note = 'Extreme LOW', 'Complacency. Both tails compressed. T+63 TRR+=1.26×, TRR-=0.75×.'
            zcol = '#f97316'
        elif pc10 < 0.732:
            zone, note = 'Moderate LOW — TRANSITION ZONE', 'EXIT from complacency is the danger. T+63 TRR-=1.49× (p=0.007).'
            zcol = '#ef4444'
        elif pc10 < 0.944:
            zone, note = 'Mid', 'No distributional edge. Baseline.'
            zcol = '#6b7280'
        elif pc10 < 1.003:
            zone, note = 'Moderate HIGH', 'Early contrarian signal. Fear building.'
            zcol = '#22c55e'
        else:
            zone, note = 'Extreme HIGH', 'Sustained fear fully priced. T+21 TRR+=1.67× (p<0.0001).'
            zcol = '#22c55e'
        col1, col2, col3 = st.columns(3)
        col1.metric('PC SMA-10', f'{pc10:.3f}')
        col2.metric('Daily PC', f'{float(pc_daily):.3f}' if not np.isnan(float(pc_daily)) else '—')
        col3.metric('Zone', zone)
        st.markdown(f"""
        <div style="background:#1e1e2e;border-left:3px solid {zcol};border-radius:6px;
                    padding:10px 16px;font-size:0.86rem;margin-top:8px;">
        <b style="color:{zcol};">{zone}</b><br>{note}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        | Zone | SMA-10 | Score | T+63 TRR+ | T+63 TRR- |
        |------|--------|-------|-----------|-----------|
        | Extreme LOW | < 0.686 | +0.5 | 1.26× | 0.75× |
        | Moderate LOW ⚠ | 0.686–0.732 | **−0.5** | 0.82× | **1.49×** |
        | Mid | 0.732–0.944 | 0.0 | baseline | baseline |
        | Moderate HIGH | 0.944–1.003 | +0.5 | — | — |
        | Extreme HIGH | > 1.003 | +1.0 | **1.67×** | — |
        """)
    except:
        st.write('PC SMA-10 data not available.')

# ══════════════════════════════════════════════════════════════════════════════
# SECTOR PERFORMANCE & RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-header">S&P 500 Sector Performance &amp; Relative Strength</div>',
            unsafe_allow_html=True)

_sec_closes = load_sector_data()
_sec_df     = build_sector_table(_sec_closes)

if _sec_df is not None and len(_sec_df) > 0:
    def _pct(v, decimals=1):
        if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
        sign = '+' if v > 0 else ''
        return f'{sign}{v * 100:.{decimals}f}%'

    def _cell_color(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return '#6b7280'
        return '#22c55e' if v > 0 else '#ef4444' if v < 0 else '#6b7280'

    def _score_color(v):
        if v >= 1.0:  return '#22c55e'
        if v >= 0.5:  return '#86efac'
        if v <= -1.0: return '#ef4444'
        if v <= -0.5: return '#f97316'
        return '#6b7280'

    TREND_CFG = {
        'IMPROVING': ('↑ IMPROVING',  '#22c55e', 'rgba(34,197,94,0.12)'),
        'FADING':    ('↓ FADING',     '#f97316', 'rgba(249,115,22,0.12)'),
        'STABLE+':  ('→ STABLE',      '#86efac', 'rgba(0,0,0,0)'),
        'STABLE-':  ('→ STABLE',      '#f87171', 'rgba(0,0,0,0)'),
        'NEUTRAL':  ('— NEUTRAL',     '#6b7280', 'rgba(0,0,0,0)'),
    }

    # ── Sort controls ──────────────────────────────────────────────────────────
    SORT_OPTIONS = {
        'Composite Score': 'Score',
        'RS 3M':  'RS 3M',
        'RS 6M':  'RS 6M',
        'RS 1Y':  'RS 1Y',
        '3M Return': '3M',
        '6M Return': '6M',
        '1Y Return': '1Y',
        'YTD Return': 'YTD',
    }
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        sort_label = st.selectbox('Sort by', list(SORT_OPTIONS.keys()), index=0, key='sector_sort', label_visibility='collapsed')
    with sc2:
        sort_asc = st.checkbox('Ascending', value=False, key='sector_asc')
    sort_col = SORT_OPTIONS[sort_label]
    _sec_df  = _sec_df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    hdr_style = 'background:#21262d;color:#9ca3af;font-size:0.72rem;font-weight:600;padding:7px 10px;text-align:right;border-bottom:1px solid #374151;white-space:nowrap;'
    hdr_left  = hdr_style.replace('text-align:right', 'text-align:left')
    row_style = 'background:#161b22;color:#e5e7eb;font-size:0.80rem;padding:6px 10px;text-align:right;border-bottom:1px solid #1f2937;'
    row_left  = row_style.replace('text-align:right', 'text-align:left')
    hdr_ctr   = hdr_style.replace('text-align:right', 'text-align:center')


    html  = '<div style="overflow-x:auto;margin-top:8px;">'
    html += '<table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">'
    html += '<thead><tr>'
    html += '<th style="' + hdr_left  + '">Sector</th>'
    html += '<th style="' + hdr_style + '">ETF</th>'
    html += '<th style="' + hdr_style + '">Close</th>'
    html += '<th style="' + hdr_style + '">YTD</th>'
    html += '<th style="' + hdr_style + '">1Y</th>'
    html += '<th style="' + hdr_style + '">6M</th>'
    html += '<th style="' + hdr_style + '">3M</th>'
    html += '<th style="' + hdr_style + '">RS 1Y</th>'
    html += '<th style="' + hdr_style + '">RS 6M</th>'
    html += '<th style="' + hdr_style + '">RS 3M</th>'
    html += '<th style="' + hdr_ctr   + '">6M&#8594;3M Trend</th>'
    html += '<th style="' + hdr_style + '">Score</th>'
    html += '</tr></thead><tbody>'

    for _, row in _sec_df.iterrows():
        score    = row['Score']
        sc_color = _score_color(score)
        label    = row['Label']
        score_txt = ('+' if score > 0 else '') + f"{score:.1f}  {label}"
        trend_key = row.get('Trend', 'NEUTRAL')
        t_lbl, t_col, t_bg = TREND_CFG.get(trend_key, TREND_CFG['NEUTRAL'])
        trend_cell = (
            'background:' + t_bg + ';color:' + t_col + ';'
            'font-weight:700;font-size:0.75rem;text-align:center;'
            'padding:6px 10px;border-bottom:1px solid #1f2937;white-space:nowrap;'
        )
        html += '<tr>'
        html += '<td style="' + row_left  + '"><b style="color:#e5e7eb;">' + row['Sector'] + '</b></td>'
        html += '<td style="' + row_style + 'color:#9ca3af;">' + row['ETF'] + '</td>'
        _dc = row['DailyChg']
        _dc_sign = '+' if _dc >= 0 else ''
        _dc_color = '#22c55e' if _dc >= 0 else '#f87171'
        html += (
            '<td style="' + row_style + 'white-space:nowrap;">'
            + f"${row['Close']:,.2f}"
            + f'<br><span style="font-size:0.70rem;color:{_dc_color};">{_dc_sign}{_dc*100:.2f}%</span>'
            + '</td>'
        )
        html += '<td style="' + row_style + 'color:' + _cell_color(row['YTD'])  + ';">' + _pct(row['YTD'])   + '</td>'
        html += '<td style="' + row_style + 'color:' + _cell_color(row['1Y'])   + ';">' + _pct(row['1Y'])    + '</td>'
        html += '<td style="' + row_style + 'color:' + _cell_color(row['6M'])   + ';">' + _pct(row['6M'])    + '</td>'
        html += '<td style="' + row_style + 'color:' + _cell_color(row['3M'])   + ';">' + _pct(row['3M'])    + '</td>'
        html += '<td style="' + row_style + 'color:' + _cell_color(row['RS 1Y'])+ ';">' + _pct(row['RS 1Y']) + '</td>'
        html += '<td style="' + row_style + 'color:' + _cell_color(row['RS 6M'])+ ';">' + _pct(row['RS 6M']) + '</td>'
        html += '<td style="' + row_style + 'color:' + _cell_color(row['RS 3M'])+ ';">' + _pct(row['RS 3M']) + '</td>'
        html += '<td style="' + trend_cell + '">' + t_lbl + '</td>'
        html += '<td style="' + row_style + 'color:' + sc_color + ';font-weight:700;white-space:nowrap;">' + score_txt + '</td>'
        html += '</tr>'

    html += '</tbody></table></div>'
    html += (
        '<div style="font-size:0.70rem;color:#6b7280;margin-top:6px;">'
        'RS = additive return differential vs SPY. '
        'Trend = RS sign shift 6M to 3M. '
        'Score = sign-based composite (range -1.5 to +1.5). '
        'Data cached hourly.</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
else:
    st.info('Sector data temporarily unavailable.')

# ══════════════════════════════════════════════════════════════════════════════
# CAN SLIM STOCK SCANNER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-header">CAN SLIM Stock Scanner</div>', unsafe_allow_html=True)

_scan_df, _scan_date = load_scanner_results()

if _scan_df is not None and len(_scan_df) > 0:

    # ── Sector score / trend lookup (join via yfinance → display name mapping) ──
    _sector_score_map = {}
    _sector_trend_map = {}
    if _sec_df is not None and len(_sec_df) > 0:
        _sector_score_map = dict(zip(_sec_df['Sector'], _sec_df['Score']))
        _sector_trend_map = dict(zip(_sec_df['Sector'], _sec_df['Trend']))

    # ── MRS context banner ─────────────────────────────────────────────────────
    _scan_mrs = _scan_df['mrs_score'].iloc[0] if 'mrs_score' in _scan_df.columns else None
    _scan_state = _scan_df['mrs_state'].iloc[0] if 'mrs_state' in _scan_df.columns else None
    if _scan_mrs is not None and not (isinstance(_scan_mrs, float) and np.isnan(_scan_mrs)):
        _mrs_color = '#22c55e' if float(_scan_mrs) > 0 else '#f87171'
        st.markdown(
            f'<div style="font-size:0.78rem;color:#9ca3af;margin-bottom:6px;">'
            f'MRS at scan: <span style="color:{_mrs_color};font-weight:700;">'
            f'{float(_scan_mrs):+.2f} {_scan_state or ""}</span>'
            f'&nbsp;·&nbsp;Scan date: <b style="color:#e5e7eb;">{_scan_date}</b>'
            f'&nbsp;·&nbsp;Universe: Russell 1000'
            f'&nbsp;·&nbsp;Updated nightly at 10 PM UTC'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Filter controls ────────────────────────────────────────────────────────
    _sc_col1, _sc_col2, _sc_col3, _sc_col4 = st.columns([2, 2, 1, 1])
    with _sc_col1:
        _mode_filter = st.selectbox(
            'Filter',
            ['All candidates', 'STRICT only', 'RELAXED only'],
            index=0, key='scanner_mode', label_visibility='collapsed'
        )
    with _sc_col2:
        _sectors = sorted(_scan_df['sector'].dropna().unique().tolist()) if 'sector' in _scan_df.columns else []
        _sector_options = ['All sectors'] + _sectors
        _sector_filter = st.selectbox(
            'Sector',
            _sector_options, index=0, key='scanner_sector', label_visibility='collapsed'
        )
    with _sc_col3:
        _atr_only = st.checkbox('ATR compressed 🔵', value=False, key='scanner_atr')
    with _sc_col4:
        _squeeze_only = st.checkbox('Squeeze ⚡', value=False, key='scanner_squeeze')

    # ── Apply filters ──────────────────────────────────────────────────────────
    _display_df = _scan_df[_scan_df['pass_mode'].notna()].copy()

    if _mode_filter == 'STRICT only':
        _display_df = _display_df[_display_df['pass_mode'] == 'STRICT']
    elif _mode_filter == 'RELAXED only':
        _display_df = _display_df[_display_df['pass_mode'] == 'RELAXED']

    if _sector_filter != 'All sectors':
        _display_df = _display_df[_display_df['sector'] == _sector_filter]

    if _atr_only and 'atr_compressed' in _display_df.columns:
        _display_df = _display_df[_display_df['atr_compressed'] == True]

    if _squeeze_only and 'bb_kc_squeeze' in _display_df.columns:
        _display_df = _display_df[_display_df['bb_kc_squeeze'] == True]
    elif _squeeze_only:
        st.warning('Squeeze data not available — re-run the scanner to update the CSV.', icon='⚡')

    _display_df = _display_df.sort_values('rs_composite', ascending=False).reset_index(drop=True)

    if _display_df.empty:
        st.info('No candidates match the selected filters.')
    else:
        def _isnan(v):
            try: return np.isnan(float(v))
            except: return True

        def _pct_s(v, dec=1):
            if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
            sign = '+' if v > 0 else ''
            return f'{sign}{v * 100:.{dec}f}%'

        def _cc(v):
            if v is None or (isinstance(v, float) and np.isnan(v)): return '#6b7280'
            return '#22c55e' if v > 0 else '#ef4444' if v < 0 else '#6b7280'

        def _sc_clr(v):
            if v >= 1.0:  return '#22c55e'
            if v >= 0.5:  return '#86efac'
            if v <= -1.0: return '#ef4444'
            if v <= -0.5: return '#f97316'
            return '#6b7280'

        # ── Style templates ────────────────────────────────────────────────────
        _hs = 'background:#21262d;color:#9ca3af;font-size:0.72rem;font-weight:600;padding:7px 10px;text-align:right;border-bottom:1px solid #374151;white-space:nowrap;cursor:pointer;user-select:none;'
        _hl = _hs.replace('text-align:right', 'text-align:left')
        _hc = _hs.replace('text-align:right', 'text-align:center')
        _rs = 'background:#161b22;color:#e5e7eb;font-size:0.80rem;padding:6px 10px;text-align:right;border-bottom:1px solid #1f2937;'
        _rl = _rs.replace('text-align:right', 'text-align:left')
        _rc = _rs.replace('text-align:right', 'text-align:center')

        _TREND_CFG_S = {
            'IMPROVING': ('↑ IMP',  '#22c55e'),
            'FADING':    ('↓ FADE', '#f97316'),
            'STABLE+':   ('→ STB',  '#86efac'),
            'STABLE-':   ('→ STB',  '#f87171'),
            'NEUTRAL':   ('— NEU',  '#6b7280'),
        }

        def _th(style, label, col_idx):
            """Sortable header cell."""
            return (f'<th style="{style}" data-col="{col_idx}" onclick="sortTbl(this)">'
                    f'{label}<span class="sarr" id="sarr{col_idx}"></span></th>')

        _scan_html  = '<div style="overflow-x:auto;margin-top:8px;">'
        _scan_html += '<table id="scantbl" style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">'
        _scan_html += '<thead><tr>'
        _scan_html += _th(_hl, 'Ticker',     0)
        _scan_html += _th(_hl, 'Name',       1)
        _scan_html += _th(_hl, 'Sector',     2)
        _scan_html += _th(_hs, 'Sect Score', 3)
        _scan_html += _th(_hc, 'Sect Trend', 4)
        _scan_html += _th(_hs, 'Close',      5)
        _scan_html += _th(_hs, '% ATH',      6)
        _scan_html += _th(_hs, 'RS Score',   7)
        _scan_html += _th(_hs, 'RS 1Y',      8)
        _scan_html += _th(_hs, 'RS 6M',      9)
        _scan_html += _th(_hs, 'RS 3M',      10)
        _scan_html += _th(_hs, 'EPS QoQ',    11)
        _scan_html += _th(_hs, 'Rev QoQ',    12)
        _scan_html += _th(_hs, 'ROE',        13)
        _scan_html += _th(_hc, 'ATR',        14)
        _scan_html += _th(_hc, 'Squeeze',    15)
        _scan_html += _th(_hc, 'Mode',       16)
        _scan_html += '</tr></thead><tbody>'

        for _, row in _display_df.iterrows():
            rs_comp  = row.get('rs_composite', 0) or 0
            sc_color = _sc_clr(float(rs_comp))
            mode     = row.get('pass_mode', '')
            mode_bg  = 'rgba(34,197,94,0.12)'  if mode == 'STRICT'  else 'rgba(251,191,36,0.10)'
            mode_col = '#22c55e'                if mode == 'STRICT'  else '#fbbf24'

            eps  = row.get('eps_qtr_yoy')
            rev  = row.get('rev_qtr_yoy')
            roe  = row.get('roe')
            rs1y = row.get('rs_1y')
            rs6m = row.get('rs_6m')
            rs3m = row.get('rs_3m')

            yf_sector    = str(row.get('sector', ''))
            display_sect = YF_TO_SECTOR.get(yf_sector, yf_sector)
            sect_score   = _sector_score_map.get(display_sect)
            sect_trend   = _sector_trend_map.get(display_sect, 'NEUTRAL')
            if sect_score is not None:
                sect_score_txt = _sector_composite_label(float(sect_score))
                sect_score_col = _sc_clr(float(sect_score))
                sect_score_val = float(sect_score)
            else:
                sect_score_txt = '—'
                sect_score_col = '#6b7280'
                sect_score_val = -999
            t_lbl, t_col = _TREND_CFG_S.get(sect_trend, ('— NEU', '#6b7280'))

            atr_c       = bool(row.get('atr_compressed', False))
            atr_txt     = '🔵' if atr_c else '—'
            atr_col     = '#60a5fa' if atr_c else '#4b5563'
            squeeze     = bool(row.get('bb_kc_squeeze', False))
            squeeze_txt = '⚡' if squeeze else '—'
            squeeze_col = '#fbbf24' if squeeze else '#4b5563'

            ticker  = str(row.get('ticker', ''))
            name    = str(row.get('name', ''))[:28]
            close_v = row.get('close')
            pct_ath = row.get('pct_from_ath')

            close_txt   = f'${float(close_v):,.2f}' if close_v is not None and not _isnan(close_v) else '—'
            close_val   = float(close_v) if close_v is not None and not _isnan(close_v) else -999
            pct_ath_txt = f'{float(pct_ath)*100:+.1f}%' if pct_ath is not None and not _isnan(pct_ath) else '—'
            pct_ath_val = float(pct_ath)*100 if pct_ath is not None and not _isnan(pct_ath) else -999
            pct_ath_col = ('#22c55e' if (pct_ath is not None and not _isnan(pct_ath) and float(pct_ath) > -0.05)
                           else '#f87171' if (pct_ath is not None and not _isnan(pct_ath) and float(pct_ath) < -0.15)
                           else '#fbbf24')

            def _sv(v):
                """Raw sort value — -999 for missing so blanks sort last."""
                if v is None or (isinstance(v, float) and np.isnan(v)): return -999
                return float(v)

            _scan_html += '<tr>'
            _scan_html += f'<td style="{_rl}" data-sort="{ticker}"><b style="color:#e5e7eb;">{ticker}</b></td>'
            _scan_html += f'<td style="{_rl}color:#9ca3af;font-size:0.75rem;" data-sort="{name}">{name}</td>'
            _scan_html += f'<td style="{_rl}color:#9ca3af;font-size:0.75rem;" data-sort="{display_sect}">{display_sect}</td>'
            _scan_html += f'<td style="{_rs}color:{sect_score_col};font-weight:700;font-size:0.72rem;white-space:nowrap;" data-sort="{sect_score_val}">{sect_score_txt}</td>'
            _scan_html += f'<td style="{_rc}color:{t_col};font-size:0.75rem;font-weight:600;" data-sort="{sect_trend}">{t_lbl}</td>'
            _scan_html += f'<td style="{_rs}" data-sort="{close_val}">{close_txt}</td>'
            _scan_html += f'<td style="{_rs}color:{pct_ath_col};font-weight:600;" data-sort="{pct_ath_val}">{pct_ath_txt}</td>'
            _scan_html += f'<td style="{_rs}color:{sc_color};font-weight:700;" data-sort="{float(rs_comp)}">{float(rs_comp):+.1f}</td>'
            _scan_html += f'<td style="{_rs}color:{_cc(rs1y)};" data-sort="{_sv(rs1y)}">{_pct_s(rs1y)}</td>'
            _scan_html += f'<td style="{_rs}color:{_cc(rs6m)};" data-sort="{_sv(rs6m)}">{_pct_s(rs6m)}</td>'
            _scan_html += f'<td style="{_rs}color:{_cc(rs3m)};" data-sort="{_sv(rs3m)}">{_pct_s(rs3m)}</td>'
            _scan_html += f'<td style="{_rs}color:{_cc(eps)};" data-sort="{_sv(eps)}">{_pct_s(eps)}</td>'
            _scan_html += f'<td style="{_rs}color:{_cc(rev)};" data-sort="{_sv(rev)}">{_pct_s(rev)}</td>'
            _scan_html += f'<td style="{_rs}color:{_cc(roe)};" data-sort="{_sv(roe)}">{_pct_s(roe)}</td>'
            _scan_html += f'<td style="{_rc}color:{atr_col};font-size:0.85rem;" data-sort="{1 if atr_c else 0}">{atr_txt}</td>'
            _scan_html += f'<td style="{_rc}color:{squeeze_col};font-size:0.85rem;" data-sort="{1 if squeeze else 0}">{squeeze_txt}</td>'
            _scan_html += f'<td style="background:{mode_bg};color:{mode_col};font-weight:700;font-size:0.75rem;text-align:center;padding:6px 10px;border-bottom:1px solid #1f2937;" data-sort="{mode}">{mode}</td>'
            _scan_html += '</tr>'

        _scan_html += '</tbody></table></div>'
        _scan_html += (
            '<div style="font-size:0.70rem;color:#6b7280;margin-top:6px;">'
            'Sect Score/Trend = sector RS from table above. '
            'RS = stock return differential vs SPY. '
            '% ATH = distance from all-time high. '
            'ATR 🔵 = bottom 35th pctile (tight base). '
            'Squeeze ⚡ = Bollinger Bands inside Keltner Channel (±1 ATR). '
            'STRICT ≥ 25% EPS + Rev. RELAXED ≥ 20% EPS, ≥ 15% Rev. Click any header to sort.'
            '</div>'
        )

        # ── JS sort logic ──────────────────────────────────────────────────────
        _scan_html += '''
<style>
  th:hover { background:#2d333b !important; }
  .sarr { margin-left:4px; font-size:0.65rem; opacity:0.7; }
</style>
<script>
var _sCol = -1, _sAsc = true;
function sortTbl(th) {
  var col = parseInt(th.getAttribute('data-col'));
  if (_sCol === col) { _sAsc = !_sAsc; } else { _sCol = col; _sAsc = false; }

  // Update arrows
  document.querySelectorAll('.sarr').forEach(function(s){ s.textContent = ''; });
  document.getElementById('sarr' + col).textContent = _sAsc ? ' ↑' : ' ↓';

  var tbody = document.querySelector('#scantbl tbody');
  var rows  = Array.from(tbody.querySelectorAll('tr'));
  var isNum = !isNaN(parseFloat(rows[0].cells[col].getAttribute('data-sort')));

  rows.sort(function(a, b) {
    var av = a.cells[col].getAttribute('data-sort');
    var bv = b.cells[col].getAttribute('data-sort');
    if (isNum) {
      av = parseFloat(av); bv = parseFloat(bv);
      // Push -999 (missing) to bottom regardless of direction
      if (av === -999 && bv === -999) return 0;
      if (av === -999) return 1;
      if (bv === -999) return -1;
    }
    if (av < bv) return _sAsc ? -1 : 1;
    if (av > bv) return _sAsc ? 1 : -1;
    return 0;
  });
  rows.forEach(function(r){ tbody.appendChild(r); });
}
</script>
'''
        _row_h  = 34
        _tbl_h  = len(_display_df) * _row_h + 120   # header + footer note
        components.html(_scan_html, height=_tbl_h, scrolling=False)

        st.caption(f'{len(_display_df)} candidates shown · '
                   f'{int((_scan_df["pass_mode"] == "STRICT").sum())} strict · '
                   f'{int((_scan_df["pass_mode"] == "RELAXED").sum())} relaxed in universe')

else:
    st.info(
        'No scanner results yet. The scanner runs nightly at 10 PM UTC (Mon–Fri) via GitHub Actions. '
        'You can also trigger it manually under Actions → CAN SLIM Scanner → Run workflow.'
    )

# ══════════════════════════════════════════════════════════════════════════════
# EPISODIC PIVOT SCANNER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-header">Episodic Pivots</div>', unsafe_allow_html=True)

try:
    from StockScanner.ep_scanner import load_ep_results as _load_ep
    _ep_df, _ep_date = _load_ep()
except Exception:
    _ep_df, _ep_date = None, None

if _ep_df is not None and len(_ep_df) > 0:

    # ── Banner ─────────────────────────────────────────────────────────────────
    _ep_mode = str(_ep_df['scan_mode'].iloc[0]) if 'scan_mode' in _ep_df.columns else 'confirmed'
    _ep_mode_lbl = 'PRE-MARKET' if _ep_mode == 'premarket' else 'AT-OPEN CONFIRMED'
    _ep_mode_col = '#fbbf24' if _ep_mode == 'premarket' else '#22c55e'
    st.markdown(
        f'<div style="font-size:0.78rem;color:#9ca3af;margin-bottom:6px;">'
        f'Mode: <span style="color:{_ep_mode_col};font-weight:700;">{_ep_mode_lbl}</span>'
        f'&nbsp;·&nbsp;Scan date: <b style="color:#e5e7eb;">{_ep_date}</b>'
        f'&nbsp;·&nbsp;Gap ≥10% · Volume ≥2× avg · 20 SMA &gt; 200 SMA · Price &gt; 200 SMA'
        f'&nbsp;·&nbsp;Fires 8:00 AM ET (pre-market) &amp; 9:45 AM ET (confirmed)'
        f'</div>',
        unsafe_allow_html=True,
    )

    def _ep_isnan(v):
        try: return np.isnan(float(v))
        except: return True

    def _ep_pct(v, dec=1):
        if v is None or _ep_isnan(v): return '—'
        sign = '+' if float(v) > 0 else ''
        return f'{sign}{float(v)*100:.{dec}f}%'

    def _ep_cc(v):
        if v is None or _ep_isnan(v): return '#6b7280'
        return '#22c55e' if float(v) > 0 else '#ef4444'

    # Header style (same palette as CAN SLIM table, clickable)
    _eh = 'background:#21262d;color:#9ca3af;font-size:0.72rem;font-weight:600;padding:7px 10px;text-align:right;border-bottom:1px solid #374151;white-space:nowrap;cursor:pointer;user-select:none;'
    _el = _eh.replace('text-align:right', 'text-align:left')
    _ec = _eh.replace('text-align:right', 'text-align:center')
    _er = 'background:#161b22;color:#e5e7eb;font-size:0.80rem;padding:6px 10px;text-align:right;border-bottom:1px solid #1f2937;'
    _erl = _er.replace('text-align:right', 'text-align:left')
    _erc = _er.replace('text-align:right', 'text-align:center')

    def _ep_th(style, label, col_idx):
        return (f'<th style="{style}" data-col="{col_idx}" onclick="sortEP(this)">'
                f'{label}<span class="eparr" id="eparr{col_idx}"></span></th>')

    _ep_html  = '<div style="overflow-x:auto;margin-top:8px;">'
    _ep_html += '<table id="eptbl" style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">'
    _ep_html += '<thead><tr>'
    _ep_html += _ep_th(_el, 'Ticker',     0)
    _ep_html += _ep_th(_el, 'Name',       1)
    _ep_html += _ep_th(_el, 'Sector',     2)
    _ep_html += _ep_th(_eh, 'Sect Score', 3)
    _ep_html += _ep_th(_ec, 'Sect Trend', 4)
    _ep_html += _ep_th(_eh, 'Gap %',      5)
    _ep_html += _ep_th(_eh, 'Open',       6)
    _ep_html += _ep_th(_eh, 'Prev Close', 7)
    _ep_html += _ep_th(_eh, 'Vol Ratio',  8)
    _ep_html += _ep_th(_ec, '20>200 SMA', 9)
    _ep_html += _ep_th(_ec, 'Mode',       10)
    _ep_html += _ep_th(_ec, 'Earnings',  11)
    _ep_html += _ep_th(_el, 'Headline',  12)
    _ep_html += '</tr></thead><tbody>'

    _TREND_CFG_EP = {
        'IMPROVING': ('↑ IMP',  '#22c55e'),
        'FADING':    ('↓ FADE', '#f97316'),
        'STABLE+':   ('→ STB',  '#86efac'),
        'STABLE-':   ('→ STB',  '#f87171'),
        'NEUTRAL':   ('— NEU',  '#6b7280'),
    }

    for _, row in _ep_df.iterrows():
        ticker      = str(row.get('ticker', ''))
        name        = str(row.get('name', ''))[:28]
        yf_sect     = str(row.get('sector', ''))
        disp_sect   = YF_TO_SECTOR.get(yf_sect, yf_sect)
        gap_pct     = row.get('gap_pct')
        today_open  = row.get('today_open')
        prev_close  = row.get('prev_close')
        vol_ratio   = row.get('vol_ratio')
        sma20       = row.get('sma20')
        sma200      = row.get('sma200')
        ep_mode          = str(row.get('scan_mode', 'confirmed'))
        earnings_flag    = bool(row.get('earnings_flag', False))
        days_since_earn  = row.get('days_since_earnings')
        headline         = str(row.get('headline', ''))[:80]
        news_url         = str(row.get('news_url', ''))
        publisher        = str(row.get('publisher', ''))

        # Sector score + trend
        ep_sect_score = _sector_score_map.get(disp_sect)
        ep_sect_trend = _sector_trend_map.get(disp_sect, 'NEUTRAL')
        if ep_sect_score is not None:
            ep_ss_txt = _sector_composite_label(float(ep_sect_score))
            ep_ss_col = _sc_clr(float(ep_sect_score))
            ep_ss_val = float(ep_sect_score)
        else:
            ep_ss_txt = '—'
            ep_ss_col = '#6b7280'
            ep_ss_val = -999
        ep_t_lbl, ep_t_col = _TREND_CFG_EP.get(ep_sect_trend, ('— NEU', '#6b7280'))

        # Gap colour
        gap_val = float(gap_pct) if gap_pct is not None and not _ep_isnan(gap_pct) else 0
        gap_txt = f'+{gap_val*100:.1f}%' if gap_val else '—'
        gap_col = '#22c55e' if gap_val >= 0.20 else '#86efac' if gap_val >= 0.10 else '#6b7280'

        # Open / prev close
        open_txt  = f'${float(today_open):,.2f}' if today_open is not None and not _ep_isnan(today_open) else '—'
        close_txt = f'${float(prev_close):,.2f}' if prev_close is not None and not _ep_isnan(prev_close) else '—'
        open_val  = float(today_open) if today_open is not None and not _ep_isnan(today_open) else -999
        close_val = float(prev_close) if prev_close is not None and not _ep_isnan(prev_close) else -999

        # Vol ratio
        vr_val = float(vol_ratio) if vol_ratio is not None and not _ep_isnan(vol_ratio) else 0
        vr_txt = f'{vr_val:.1f}×' if vr_val else '—'
        vr_col = '#22c55e' if vr_val >= 5 else '#86efac' if vr_val >= 2 else '#f97316'

        # SMA check
        sma_ok  = (sma20 is not None and sma200 is not None
                   and not _ep_isnan(sma20) and not _ep_isnan(sma200)
                   and float(sma20) > float(sma200))
        sma_txt = '✓' if sma_ok else '✗'
        sma_col = '#22c55e' if sma_ok else '#ef4444'

        # Mode badge
        mode_bg  = 'rgba(34,197,94,0.12)'  if ep_mode == 'confirmed' else 'rgba(251,191,36,0.10)'
        mode_col = '#22c55e'               if ep_mode == 'confirmed' else '#fbbf24'
        mode_lbl = 'CONF' if ep_mode == 'confirmed' else 'PRE'

        # Headline link
        if news_url and headline:
            hl_html = f'<a href="{news_url}" target="_blank" style="color:#60a5fa;text-decoration:none;font-size:0.73rem;">{headline}</a>'
            if publisher:
                hl_html += f' <span style="color:#6b7280;font-size:0.68rem;">({publisher})</span>'
        elif headline:
            hl_html = f'<span style="color:#9ca3af;font-size:0.73rem;">{headline}</span>'
        else:
            hl_html = '<span style="color:#4b5563;">—</span>'

        _ep_html += '<tr>'
        _ep_html += f'<td style="{_erl}" data-sort="{ticker}"><b style="color:#e5e7eb;">{ticker}</b></td>'
        _ep_html += f'<td style="{_erl}color:#9ca3af;font-size:0.75rem;" data-sort="{name}">{name}</td>'
        _ep_html += f'<td style="{_erl}color:#9ca3af;font-size:0.75rem;" data-sort="{disp_sect}">{disp_sect}</td>'
        _ep_html += f'<td style="{_er}color:{ep_ss_col};font-weight:700;font-size:0.72rem;white-space:nowrap;" data-sort="{ep_ss_val}">{ep_ss_txt}</td>'
        _ep_html += f'<td style="{_erc}color:{ep_t_col};font-size:0.75rem;font-weight:600;" data-sort="{ep_sect_trend}">{ep_t_lbl}</td>'
        _ep_html += f'<td style="{_er}color:{gap_col};font-weight:700;" data-sort="{gap_val}">{gap_txt}</td>'
        _ep_html += f'<td style="{_er}" data-sort="{open_val}">{open_txt}</td>'
        _ep_html += f'<td style="{_er}" data-sort="{close_val}">{close_txt}</td>'
        _ep_html += f'<td style="{_er}color:{vr_col};font-weight:600;" data-sort="{vr_val}">{vr_txt}</td>'
        _ep_html += f'<td style="{_erc}color:{sma_col};font-weight:700;" data-sort="{1 if sma_ok else 0}">{sma_txt}</td>'
        _ep_html += f'<td style="background:{mode_bg};color:{mode_col};font-weight:700;font-size:0.75rem;text-align:center;padding:6px 10px;border-bottom:1px solid #1f2937;" data-sort="{ep_mode}">{mode_lbl}</td>'

        # Earnings flag cell
        if earnings_flag:
            days_tag = f' +{days_since_earn}d' if days_since_earn is not None else ''
            earn_html = f'<span style="color:#fbbf24;font-weight:700;font-size:0.73rem;">⚡ EPS{days_tag}</span>'
            earn_sort = 1
        else:
            earn_html = '<span style="color:#4b5563;">—</span>'
            earn_sort = 0
        _ep_html += f'<td style="{_erc}" data-sort="{earn_sort}">{earn_html}</td>'

        _ep_html += f'<td style="{_erl}max-width:340px;" data-sort="{headline}">{hl_html}</td>'
        _ep_html += '</tr>'

    _ep_html += '</tbody></table></div>'
    _ep_html += (
        '<div style="font-size:0.70rem;color:#6b7280;margin-top:6px;">'
        'Gap = today open vs prior close. '
        'Vol Ratio = intraday volume ÷ 50-day avg. '
        'Sect Score/Trend from sector table above. '
        'Click any header to sort. '
        'PRE = pre-market signal · CONF = confirmed at open.'
        '</div>'
    )

    # ── JS sort (separate namespace from CAN SLIM table) ──────────────────────
    _ep_html += '''
<style>
  #eptbl th:hover { background:#2d333b !important; }
  .eparr { margin-left:4px; font-size:0.65rem; opacity:0.7; }
</style>
<script>
var _epCol = -1, _epAsc = true;
function sortEP(th) {
  var col = parseInt(th.getAttribute('data-col'));
  if (_epCol === col) { _epAsc = !_epAsc; } else { _epCol = col; _epAsc = false; }
  document.querySelectorAll('.eparr').forEach(function(s){ s.textContent = ''; });
  document.getElementById('eparr' + col).textContent = _epAsc ? ' ↑' : ' ↓';
  var tbody = document.querySelector('#eptbl tbody');
  var rows  = Array.from(tbody.querySelectorAll('tr'));
  var isNum = !isNaN(parseFloat(rows[0].cells[col].getAttribute('data-sort')));
  rows.sort(function(a, b) {
    var av = a.cells[col].getAttribute('data-sort');
    var bv = b.cells[col].getAttribute('data-sort');
    if (isNum) {
      av = parseFloat(av); bv = parseFloat(bv);
      if (av === -999 && bv === -999) return 0;
      if (av === -999) return 1;
      if (bv === -999) return -1;
    }
    if (av < bv) return _epAsc ? -1 : 1;
    if (av > bv) return _epAsc ? 1 : -1;
    return 0;
  });
  rows.forEach(function(r){ tbody.appendChild(r); });
}
</script>
'''
    _ep_row_h = 42   # slightly taller due to headline column
    _ep_h     = len(_ep_df) * _ep_row_h + 130
    components.html(_ep_html, height=_ep_h, scrolling=False)
    st.caption(f'{len(_ep_df)} episodic pivot(s) detected today')

else:
    st.info(
        'No Episodic Pivots detected today — or scan hasn\'t run yet. '
        'The EP scanner fires at 8:00 AM ET (pre-market) and 9:45 AM ET (confirmed) Mon–Fri via GitHub Actions.'
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
last_upd = hist['date'].max()
st.markdown(
    '<div style="text-align:center;font-size:0.72rem;color:#4b5563;margin-top:24px;">'
    'Epistruct &nbsp;|&nbsp;'
    'Data through ' + last_upd.strftime('%B %d, %Y') + ' &nbsp;|&nbsp;'
    'Updates daily at 4:30 PM ET'
    '</div>',
    unsafe_allow_html=True,
)
