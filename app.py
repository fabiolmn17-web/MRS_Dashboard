"""
app.py — MRS Live Dashboard (Streamlit)
========================================
Password-protected. Reads mrs_history.csv and displays:
  • Current regime score (MRS v4.0 scoring, v8 dispersion re-test) + percentile + signal quality
  • Risk Dial: forward dispersion profile of today's band (risk_dial.py)
  • 9-component breakdown table with weights and weighted contributions
  • VIX lifecycle state layer
  • Zero Gamma position
  • 90-day MRS history chart + SPX close panel + VIX state panel
  • S&P 500 Sector Performance & Relative Strength, with industry drill-down
    (industry ETFs + top 10 holdings, sector_view.py / StockScanner/industry.py)
  • CAN SLIM Stock Scanner
"""
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf
from pathlib import Path
from datetime import date
from itertools import groupby
import os
import pipeline
import risk_dial
import sector_view
import rotation_view
import rs_labels
import market_state
import scan_view
import chart_view

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
def load_industry_data():
    """Industry RS + top-10 holdings from the nightly job (StockScanner/run_industry.py)."""
    try:
        base = Path(__file__).parent / 'StockScanner' / 'output'
        ind_p, hol_p = base / 'industry_rs.csv', base / 'industry_holdings.csv'
        if not ind_p.exists():
            return None, None
        ind = pd.read_csv(ind_p)
        hol = pd.read_csv(hol_p) if hol_p.exists() else pd.DataFrame()
        return (ind if len(ind) else None), hol
    except Exception as e:
        print(f'[industry] load error: {e}')
        return None, None


@st.cache_data(ttl=3600)
def load_sector_data():
    """Fetch ~2 years of daily closes for 11 SPDR ETFs + SPY. Cached 1 hour.

    Uses individual yf.Ticker().history() calls per ticker rather than bulk
    yf.download() — the bulk endpoint silently returns stale/incomplete data
    for a subset of tickers on cloud infra (same root cause as the Jun 2026
    mrs_history gap; see pipeline.py / backfill.py for the same fix).
    """
    tickers = list(SECTOR_MAP.keys()) + ['SPY']
    series_map = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period='2y', auto_adjust=True)   # 2y: ~250 sessions of label history for the 3-day hold
            if h is None or h.empty or 'Close' not in h.columns:
                print(f'[sector] {t}: empty history')
                continue
            s = h['Close'].copy()
            s.index = pd.to_datetime(s.index).normalize().tz_localize(None)
            series_map[t] = s
        except Exception as e:
            print(f'[sector] {t}: fetch error: {e}')

    if not series_map:
        return None

    closes = pd.DataFrame(series_map)
    return closes.dropna(how='all')

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

        r_ytd = p / ytd_base - 1
        r_1y, r_6m, r_3m, r_1m = ret(252), ret(126), ret(63), ret(21)

        # RS, score, 3-day held label, 6M->3M trend, days in label, RS 1M tag (rs_labels.py)
        li = rs_labels.label_info(s, spy)
        if li is None:
            continue

        rows.append({
            'Sector': sector, 'ETF': etf, 'Close': p, 'DailyChg': daily_chg,
            'YTD': r_ytd, '1Y': r_1y, '6M': r_6m, '3M': r_3m, '1M': r_1m,
            **li,
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
            'b20_pct':    '' if b20 is None else str(b20),   # '' = keep the daily run's S&P 500 B20%
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
    """Load history from GitHub raw URL (always fresh), fallback to local file.
    Set MRS_LOCAL_DATA=1 to read the local mrs_history.csv (testing a branch)."""
    if os.environ.get('MRS_LOCAL_DATA') == '1':
        return pipeline.load_history(HIST_PATH)
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

# ── Zero gamma shadow: Gamma Live SPY 40-day flip, sent after each close by the Gamma Live PC ──────────────────
GAMMA_SHADOW_PATH = Path(__file__).parent / 'gamma_shadow.csv'
GAMMA_SHADOW_URL = 'https://raw.githubusercontent.com/fabiolmn17-web/MRS_Dashboard/main/gamma_shadow.csv'

@st.cache_data(ttl=60)
def load_gamma_shadow() -> pd.DataFrame:
    """date, spy_close, zg40_next (40-day flip without the expiring contracts = the next session's map), zg40_close
    (as Gamma Live showed it at the close), zg3_next, net40_next_bn. Shadow only: not used in the MRS score."""
    for src in ([] if os.environ.get('MRS_LOCAL_DATA') == '1' else [GAMMA_SHADOW_URL]) + [GAMMA_SHADOW_PATH]:
        try:
            g = pd.read_csv(src, parse_dates=['date'])
            if len(g):
                return g.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame()

def gamma_shadow_table(h: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Per session: InsiderFinance SPX zero gamma (the one scored) vs Gamma Live 40d in SPX points (× SPX/SPY
    of that session), with the gamma state each would give."""
    if g is None or g.empty:
        return pd.DataFrame()
    m = g.merge(h[['date', 'spx', 'spy', 'zero_gamma']], on='date', how='left')
    ratio = (m['spx'] / m['spy']).where(m['spy'] > 0)
    m['gl_spx'] = m['zg40_next'] * ratio
    _st = lambda a, b: (str(pipeline.score_gamma(a, b)[1]).replace(' Gamma', '')
                        if pd.notna(a) and pd.notna(b) else '—')
    m['if_state'] = [_st(a, b) for a, b in zip(m['spx'], m['zero_gamma'])]
    m['gl_state'] = [_st(a, b) for a, b in zip(m['spx'], m['gl_spx'])]
    return m

gshadow = load_gamma_shadow()

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
    st.caption('Enter after market close (4 PM ET). ADL: TradingView value — auto x1000. '
               'Values entered here always take priority; the automated run only fills blanks.')

    # Pre-populate: prefer URL params (survive reload) then CSV last value
    _qp = st.query_params
    _def_b20 = float(_qp['b20']) if 'b20' in _qp else _last_val('b20_pct',    50.0)
    _def_adl = float(_qp['adl']) if 'adl' in _qp else _last_val('adl_level',  1_827_000.0) / 1000.0
    _def_zg  = float(_qp['zg'])  if 'zg'  in _qp else _last_val('zero_gamma', 7_400.0)

    _b20_auto_last = _last_val('b20_auto')
    with st.form('daily_inputs_form'):
        inp_b20_auto = st.checkbox('B20%: automatic (full S&P 500, daily run)', value=True,
                                   help='The daily run computes B20% from all S&P 500 stocks (the scanner universe), the '
                                        'same definition as S5TW. Untick to enter the TradingView S5TW value instead. '
                                        + (f'Last automatic value: {_b20_auto_last:.2f}%.' if _b20_auto_last == _b20_auto_last
                                           else 'No automatic value stored yet (first run after this update).'))
        inp_b20 = st.number_input('B20% (S5TW) — only if unticked', min_value=0.0,  max_value=100.0,
                                   value=round(_def_b20, 2), step=0.01)
        inp_adl = st.number_input('ADL (TradingView x1000)', value=round(_def_adl, 2), step=0.01)
        inp_zg  = st.number_input('Zero Gamma (SPX level)',  value=round(_def_zg, 2),  step=1.0)
        inp_pc  = st.number_input('PC Ratio — USI:PC close', value=0.0, step=0.0001,
                                   min_value=0.0, max_value=3.0, format='%.4f',
                                   help=('Enter the USI:PC daily close from TradingView — the series '
                                         'the Five-Zone cutoffs are calibrated on. Leave 0 to try the '
                                         'CBOE feed; if it is unavailable PC is left blank (scores '
                                         'neutral). Valid range: 0.3 – 2.0.'))
        inp_same_ok = st.checkbox('Same B20% and ADL as the last stored session (intended)', value=False,
                                  help='The form starts from the last stored values. If B20% and ADL are both '
                                       'unchanged, the update is held back unless this box is ticked, so a second '
                                       'click with the pre-filled numbers cannot overwrite the values just entered.')
        submitted = st.form_submit_button('Submit & Update')

    if submitted:
        # ── Input validation ───────────────────────────────────────────────────
        _errors = []
        if not inp_b20_auto and inp_b20 <= 0:
            _errors.append('B20% must be > 0.')
        if not inp_b20_auto and inp_b20 > 100:
            _errors.append('B20% must be <= 100.')
        if inp_adl < 100 or inp_adl > 10_000:
            _errors.append(f'ADL {inp_adl:.0f} looks wrong (expected 500–5000 range).')
        if inp_zg < 2_000 or inp_zg > 15_000:
            _errors.append(f'Zero Gamma {inp_zg:.0f} is outside a valid SPX range.')
        if inp_pc > 3.0:
            _errors.append(f'PC Ratio {inp_pc:.3f} looks like an SPX price, not a ratio. Use 0 for auto-fetch.')
        # Guard against re-submitting the pre-filled (previous) values: on 23 Oct 2026 a second submit with the
        # form defaults overwrote the B20% / ADL / zero gamma entered a minute earlier for 22 Oct.
        _prev_b20, _prev_adl = _last_val('b20_pct'), _last_val('adl_level')
        if inp_b20_auto:                        # B20% is not typed: guard on ADL alone
            if not inp_same_ok and _prev_adl == _prev_adl and abs(inp_adl * 1000 - _prev_adl) < 1e-3:
                _errors.append(f'ADL {inp_adl:.2f} is the same as the last stored session. Enter today\'s value, '
                               'or tick "Same B20% and ADL ..." if this is intended.')
        elif (not inp_same_ok and _prev_b20 == _prev_b20 and _prev_adl == _prev_adl
                and abs(inp_b20 - _prev_b20) < 1e-6 and abs(inp_adl * 1000 - _prev_adl) < 1e-3):
            _errors.append(f'B20% {inp_b20:.2f} and ADL {inp_adl:.2f} are the same as the last stored session. '
                           'Enter today\'s values, or tick "Same B20% and ADL ..." if this is intended.')

        if _errors:
            for e in _errors:
                st.sidebar.error(e)
        else:
            gh_token = st.secrets.get('GITHUB_TOKEN', '')
            if not gh_token:
                st.sidebar.error('GITHUB_TOKEN not in Streamlit secrets.')
            else:
                # Persist in URL params (survives reload) + session_state (live preview)
                if not inp_b20_auto:
                    st.query_params['b20'] = str(inp_b20)
                st.query_params['adl'] = str(inp_adl)
                st.query_params['zg']  = str(inp_zg)
                if inp_pc > 0:
                    st.query_params['pc'] = str(inp_pc)
                st.session_state['pending_b20']  = None if inp_b20_auto else inp_b20
                st.session_state['pending_adl']  = inp_adl * 1000  # ×1000 → CSV scale
                st.session_state['pending_zg']   = inp_zg
                st.session_state['pending_pc']   = inp_pc
                st.session_state['pending_date'] = _today.strftime('%Y-%m-%d')
                ok = _trigger_backfill(gh_token, None if inp_b20_auto else inp_b20, inp_adl, inp_zg, inp_pc)
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

    # ── Zero gamma: InsiderFinance (scored) vs Gamma Live 40d (shadow) ────────────────────────────────────────
    _gst = gamma_shadow_table(hist, gshadow)
    if len(_gst):
        with st.sidebar.expander('Zero gamma: InsiderFinance vs Gamma Live (shadow)'):
            _v = _gst.dropna(subset=['gl_spx']).tail(12)
            _agree = (_v['if_state'] == _v['gl_state']).sum()
            st.caption(f'Gamma Live = SPY 40-day flip without the expiring contracts, × SPX/SPY of the session. '
                       f'Not used in the score. Same gamma state on {_agree} of {len(_v)} sessions shown.')
            st.dataframe(pd.DataFrame({
                'Date': _v['date'].dt.strftime('%d %b'),
                'IF': _v['zero_gamma'].map(lambda x: f'{x:,.0f}' if pd.notna(x) else '—'),
                'GL': _v['gl_spx'].map(lambda x: f'{x:,.0f}'),
                'IF state': _v['if_state'], 'GL state': _v['gl_state'],
            }).iloc[::-1], hide_index=True, use_container_width=True)
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
    if st.session_state.get('pending_b20') is not None:
        last['b20_pct'] = st.session_state['pending_b20']
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
    background: #121212;
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
    color: #a89f95;
    margin-bottom: 8px;
    margin-top: 20px;
  }
  .hazard-row {
    background: #2a1c13;
    border-left: 4px solid #f4a062;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.90rem;
    color: #b9805a;
  }
  .safe-row {
    background: #132322;
    border-left: 4px solid #85cac5;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.90rem;
    color: #649995;
  }
  .neutral-row {
    background: #1a1918;
    border-left: 4px solid #6e675f;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.90rem;
    color: #a89f95;
  }
  div[data-testid="stDataFrame"] { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)
import atn_brand
atn_brand.inject(st, 'Market Regime Score')
st.markdown("""<style>
.metric-card {background:#121212 !important; border:1px solid #2a2724; border-radius:4px !important;}
.score-number {font-family:'IBM Plex Mono', monospace !important; font-weight:500 !important; letter-spacing:-0.02em;}
.regime-label {font-family:Barlow, sans-serif !important; font-weight:600 !important; letter-spacing:0.02em !important;}
.quality-chip {border-radius:2px !important; font-family:Barlow, sans-serif; font-weight:500 !important; letter-spacing:0.08em !important;}
.section-header {color:#e3b48e !important; letter-spacing:0.2em !important; font-weight:500 !important;}
.hazard-row, .safe-row, .neutral-row {border-left:1px solid #2a2724 !important; border:1px solid #2a2724; border-radius:4px !important;}
.hazard-row {color:#f4a062 !important;} .safe-row {color:#85cac5 !important;} .neutral-row {color:#a89f95 !important;}
</style>""", unsafe_allow_html=True)

# ── Helper functions ───────────────────────────────────────────────────────────
def _f(key, fmt='{:.2f}'):
    v = last.get(key, np.nan)
    try:
        fv = float(v)
        return '—' if np.isnan(fv) else fmt.format(fv)
    except:
        return str(v) if v else '—'

QUALITY_COLORS = {
    'CONFIRMED':              '#85cac5',
    'NEUTRAL — BULLISH LEAN': '#85cac5',
    'NEUTRAL — BEARISH LEAN': '#b9805a',
    'NEUTRAL — NO EDGE':      '#a89f95',
    'UNCONFIRMED':            '#b9805a',
    'DIVERGENT':              '#f4a062',
    'FRAGILE':                '#e3b48e',
}

def quality_color(label: str) -> str:
    for k, v in QUALITY_COLORS.items():
        if k in label:
            return v
    return '#a89f95'

@st.cache_data(ttl=600)
def _band_profile(h: pd.DataFrame) -> pd.DataFrame:
    return risk_dial.band_profile(h)

@st.cache_data(ttl=600)
def _score_context(h: pd.DataFrame, score: float) -> dict:
    return risk_dial.score_context(h, score)

def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

# Risk-dial colors (validated for CVD separation and 3:1 contrast on the #0a0a0a surface)
ADV_COLOR = '#f4a062'   # large adverse move
FAV_COLOR = '#85cac5'   # large favorable move

# ══════════════════════════════════════════════════════════════════════════════
# HEADER ROW
# ══════════════════════════════════════════════════════════════════════════════
mrs    = float(last.get('mrs_score', 0) or 0)
reg    = pipeline.regime_label(mrs)
rcol   = pipeline.regime_color(mrs)
dur    = pipeline.compute_regime_duration(hist, last_dt)
sq_lbl, sq_desc, sq_hex = pipeline.compute_signal_quality(last, hist, last_dt)
sq_col = '#' + sq_hex
ctx    = _score_context(hist, mrs)
_deep  = ctx['deep_state']
_deep_html = ''
if _deep in ('DEEP RISK-OFF', 'DEEP RISK-ON'):
    _dc = '#f4a062' if _deep == 'DEEP RISK-OFF' else '#85cac5'
    _deep_html = (f'<span class="quality-chip" style="margin-top:8px;background:{_dc}20;color:{_dc};'
                  f'border:1px solid {_dc};">{_deep}</span>')


# ── Data for the decision strip, the leaders list and the scanner (loaded once, cached) ─────────
_sec_closes = load_sector_data()
_sec_df     = build_sector_table(_sec_closes)
_ind_df, _hold_df = load_industry_data()
_scan_df, _scan_date = load_scanner_results()
_bp = _band_profile(hist)
_mom_light = rs_labels.momentum_light(hist.sort_values('date').set_index('date')['spy'].astype(float), reg)
_leaders = rs_labels.leaders_shortlist(_ind_df)
_added, _dropped = rs_labels.shortlist_changes(_ind_df)

# ── What changed since the last session ──────────────────────────────────────────
import html as _html_mod
_esc = lambda x: _html_mod.escape(str(x))
_changes = []
_h2 = hist.dropna(subset=['mrs_score']).sort_values('date')
if len(_h2) >= 2:
    _prev_s = float(_h2['mrs_score'].iloc[-2])
    _prev_r = pipeline.regime_label(_prev_s)
    _changes.append(f'MRS <b>{_prev_s:+.2f} → {mrs:+.2f}</b>'
                    + (f' (<b>{_prev_r} → {reg}</b>)' if _prev_r != reg else f' ({reg}, session {dur})'))
if len(_added):
    _changes.append('New on the leaders list: <b>' + ', '.join(_esc(x) for x in _added['Industry']) + '</b>')
if len(_dropped):
    _changes.append('Dropped from the list: <b>' + ', '.join(_esc(x) for x in _dropped['Industry']) + '</b>')
if _sec_df is not None and len(_sec_df) and 'PrevScore' in _sec_df.columns:
    for _, _r in _sec_df.iterrows():
        if pd.notna(_r['PrevScore']) and float(_r['PrevScore']) != float(_r['Score']):
            _changes.append(f'{_esc(_r["Sector"])}: {rs_labels.composite_label(float(_r["PrevScore"])).title()} → '
                            f'<b>{_esc(str(_r["Label"]).title())}</b>')
_n_pend = 0
for _d in (_sec_df, _ind_df):
    if _d is not None and 'PendingLabel' in _d.columns:
        _n_pend += int(_d['PendingLabel'].fillna('').astype(str).str.len().gt(0).sum())
if _n_pend:
    _changes.append(f'{_n_pend} label{"s" if _n_pend != 1 else ""} with a change in progress (see "→ x/3" in the tables)')

# ── Freshness of each data block ─────────────────────────────────────────────────
_fresh = [f'MRS {last_dt:%b %d}']
if _sec_closes is not None and len(_sec_closes):
    _fresh.append(f'sectors {_sec_closes.index.max():%b %d}')
if _ind_df is not None and 'as_of' in _ind_df.columns:
    _fresh.append(f'industries {_ind_df["as_of"].iloc[0]} (nightly)')
if _scan_df is not None and _scan_date:
    _fresh.append(f'scanner {_scan_date} (nightly)')

_deep_line = ''
if _deep == 'DEEP RISK-OFF':
    _deep_line = (f'DEEP RISK-OFF (score ≤ P10 = {ctx["p10"]:+.2f}). Historically not a bottom signal: over the next '
                  '10–20 sessions downside ran fatter than usual (10-session drop ≥3% in 29% of episodes vs 11%); the wide '
                  'upside tended to come later. Exploratory, v3.0 scoring, did not survive multiple-testing correction.')
elif _deep == 'DEEP RISK-ON':
    _deep_line = (f'DEEP RISK-ON (score ≥ P90 = {ctx["p90"]:+.2f}). Historically a calm path with shallow drawdowns, '
                  'but no reliable edge in returns. Exploratory, v3.0 scoring.')

def _gl_header() -> dict:
    """Shadow zero gamma (Gamma Live 40d, SPX points) for the header, only when it is for the session shown."""
    try:
        _t = gamma_shadow_table(hist, gshadow)
        _r = _t[_t['date'] == last_dt]
        if len(_r) and pd.notna(_r['gl_spx'].iloc[0]):
            return {'Zero γ GL': f"{_r['gl_spx'].iloc[0]:,.0f} (shadow)"}
    except Exception:
        pass
    return {}

st.markdown(market_state.decision_strip_html(
    mrs, reg, rcol, dur, f"{_ordinal(int(round(ctx['percentile'])))} percentile since {ctx['start'].year}",
    sq_lbl, sq_col, sq_desc, _mom_light, _bp, _leaders,
    {'VIX': _f('vix', '{:.2f}'), 'SPX': _f('spx', '{:,.0f}'), 'Zero γ': _f('zero_gamma', '{:,.0f}'),
     **_gl_header()},
    _changes, _deep_line, 'Data as of: ' + ' · '.join(_fresh)), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHERE TO LOOK: leaders to buy (industries first), rotation, scanner
# ══════════════════════════════════════════════════════════════════════════════
try:
    _ld_asof = str(_ind_df['as_of'].iloc[0]) if _ind_df is not None and 'as_of' in _ind_df.columns else None
    st.markdown(market_state.leaders_html(_leaders, _mom_light, _hold_df, _ld_asof, _dropped), unsafe_allow_html=True)
except Exception as _ld_err:
    print(f'[leaders] {_ld_err}')


# ── Sector rotation at a glance (rotation_view.py) ─────────────────────────────
try:
    _rot_pts = rotation_view.rotation_points(load_sector_data(), SECTOR_MAP)
    if len(_rot_pts):
        _rot_ind, _ = load_industry_data()
        _rot_asof = str(_rot_ind['as_of'].iloc[0]) if _rot_ind is not None and 'as_of' in _rot_ind.columns else None
        st.markdown('<div class="section-header" style="margin-top:6px;">Sector rotation — relative strength vs SPY</div>',
                    unsafe_allow_html=True)
        _rot_h_lbl = st.radio('Map horizon', ['6M vs 3M (standard)', '3M vs 1M (short-term)'], index=0,
                              horizontal=True, key='rot_horizon', label_visibility='collapsed')
        _rot_h = 'short' if _rot_h_lbl.startswith('3M') else 'standard'
        _rot_l, _rot_r = st.columns([1.35, 1], gap='large')
        with _rot_l:
            st.plotly_chart(rotation_view.rotation_figure(_rot_pts, horizon=_rot_h), use_container_width=True,
                            config={'displayModeBar': False})
        with _rot_r:
            st.markdown(rotation_view.rotation_summary_html(_rot_pts, _rot_ind, _rot_asof, horizon=_rot_h),
                        unsafe_allow_html=True)
except Exception as _rot_err:
    print(f'[rotation] {_rot_err}')

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM & TREND SCANNER (scan_view.py; nightly StockScanner job)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Momentum &amp; trend scanner</div>', unsafe_allow_html=True)

# Table (click a row) + daily chart panel on the right. The table is a small two-way component
# (scan_table/index.html) that returns the clicked ticker; the chart is fetched per ticker and cached.
_scan_table = components.declare_component('scan_table', path=str(Path(__file__).parent / 'scan_table'))


@st.cache_data(ttl=1800, show_spinner=False)
def _chart_history(ticker):
    return chart_view.fetch(ticker)


def _scan_chart_panel(tickers):
    t = st.session_state.get('scan_pick')
    if not t or t not in tickers:
        t = tickers[0]
    h = _chart_history(t)
    if h is None or len(h) < 30:
        st.info(f'No price history for {t}.')
        return
    lv = chart_view.levels(h)
    dc = lv['chg']
    dcol = '#85cac5' if dc >= 0 else '#f4a062'
    def _d(v):   # distance from the close, in %
        return '—' if np.isnan(v) else f'{(lv["close"] / v - 1) * 100:+.1f}%'
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:10px;margin:2px 0 4px 0;">'
        f'<span style="font-size:1.25rem;font-weight:700;color:#f3ece2;">{_html_mod.escape(t)}</span>'
        f'<span style="font-family:IBM Plex Mono,monospace;color:#f3ece2;">${lv["close"]:,.2f}</span>'
        f'<span style="font-family:IBM Plex Mono,monospace;color:{dcol};">{dc * 100:+.2f}%</span></div>'
        f'<div style="font-size:0.72rem;color:#a89f95;line-height:1.6;">'
        f'vs SMA 50 <b style="color:#e8dccb;">{_d(lv["sma50"])}</b> · vs SMA 200 <b style="color:#b9805a;">{_d(lv["sma200"])}</b>'
        f' · vs 52W high <b style="color:#e3b48e;">{_d(lv["hi52"])}</b> · vs 20D high <b style="color:#9b958f;">{_d(lv["hi20"])}</b></div>',
        unsafe_allow_html=True)
    st.plotly_chart(chart_view.figure(h, t, height=520), use_container_width=True,
                    config={'displayModeBar': False, 'scrollZoom': True}, key='scan_chart')
    st.markdown('<div style="font-size:0.66rem;color:#9b958f;">Daily, last 12 months. SMA 50 / 200; 52W high = highest '
                'high of the last 252 sessions; 20D high = highest high of the 20 sessions before today. Click a row '
                'in the table to switch.</div>', unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _extmom_html(ticker):
    """The TradingView Ext·Mom table for one ticker: its own 10-year history plus the pooled stock panel."""
    from StockScanner import extmom_table as _xm
    h = _chart_history(ticker)
    try:
        r = _xm.compute(h, _xm.load_pooled())
    except Exception:
        r = None
    return _xm.render_html(r, ticker) if r else ''


@st.cache_data(ttl=1800, show_spinner=False)
def _extmom_lines(ticker):
    """Ext·Mom as compact lines for the ticker lookup (same content as the table)."""
    from StockScanner import extmom_table as _xm
    h = _chart_history(ticker)
    try:
        r = _xm.compute(h, _xm.load_pooled())
    except Exception:
        r = None
    return _xm.render_lines_html(r, ticker) if r else ''


def _scan_extmom(ticker):
    if not ticker:
        return
    html = _extmom_html(ticker)
    if html:
        st.markdown(f'<div style="font-size:0.8rem;color:#f3ece2;font-weight:600;margin:6px 0 2px 0;">Ext·Mom · '
                    f'{_html_mod.escape(ticker)}</div>' + html, unsafe_allow_html=True)


def _scan_table_and_chart_body(table_html, tickers, height):
    show = st.toggle('Chart panel (click a row)', value=True, key='ms_chart')
    if not show:
        _scan_table(html=table_html, height=height, selected=st.session_state.get('scan_pick'),
                    key='scan_table_click', default=None)
        return
    c_tab, c_chart = st.columns([2.6, 1], gap='medium')
    with c_tab:
        picked = _scan_table(html=table_html, height=height, selected=st.session_state.get('scan_pick'),
                             key='scan_table_click', default=None)
        if picked and picked != st.session_state.get('scan_pick'):
            st.session_state['scan_pick'] = picked
    with c_chart:
        _scan_chart_panel(tickers)
    _t = st.session_state.get('scan_pick')
    _scan_extmom(_t if _t in tickers else (tickers[0] if tickers else None))


# st.fragment (Streamlit 1.37+): a click reruns only the table and the chart, not the whole page
_scan_table_and_chart = st.fragment(_scan_table_and_chart_body) if hasattr(st, 'fragment') else _scan_table_and_chart_body

# ── Ticker lookup: how a name looks through the scan (any US ticker) ─────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _lookup_live(ticker):
    """Price-only scan fields for a ticker outside the nightly universe (same functions as the scanner).
    CAN SLIM fundamentals are not fetched here."""
    from StockScanner import momentum as _mom
    from StockScanner.calculations import RelativeStrengthCalculator, PatternDetector
    h = _chart_history(ticker)
    spy = _chart_history('SPY')
    if h is None or spy is None or len(h) < 60:
        return None
    row = {'ticker': ticker, 'close': float(h['Close'].iloc[-1])}
    row.update(_mom.momentum_fields(h, spy))
    rs = RelativeStrengthCalculator().compute(h, spy)
    row.update({k: rs.get(k) for k in ('rs_1y', 'rs_6m', 'rs_3m', 'rs_1m')})
    row.update(PatternDetector().compute(h))
    c = h['Close']
    row['sma200'] = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else np.nan
    row['sma50'] = float(c.rolling(50).mean().iloc[-1])
    try:
        info = yf.Ticker(ticker).info or {}
        row.update({'name': info.get('longName') or info.get('shortName', ticker),
                    'sector': info.get('sector', ''), 'industry': info.get('industry', '')})
    except Exception:
        row.update({'name': ticker, 'sector': '', 'industry': ''})
    row['next_earnings'] = _mom.next_earnings(ticker)
    row['pass_mode'] = None
    return row


def _scan_lookup(sd, sec_label, sec_score, leader_hold):
    c_in, c_note = st.columns([1, 3])
    with c_in:
        q = st.text_input('Look up a ticker', value='', key='ms_lookup', placeholder='e.g. CRM, PNTG, XNCR',
                          help='Shows the stock as the scan sees it. S&P 500 names come from the nightly scan; '
                               'any other US ticker is computed live from prices (no CAN SLIM fundamentals).')
    t = (q or '').strip().upper().lstrip('$')
    if not t:
        return
    hit = sd[sd['ticker'].astype(str).str.upper() == t]
    in_scan = len(hit) > 0
    if in_scan:
        r = hit.iloc[0].to_dict()
        src = f'nightly scan ({_scan_date})'
    else:
        with st.spinner(f'Computing {t} from prices...'):
            r = _lookup_live(t)
        if r is None:
            st.warning(f'No price history found for {t}.')
            return
        src = 'computed live from prices (not in the nightly universe; CAN SLIM not checked)'
    with c_note:
        st.markdown(f'<div style="font-size:0.74rem;color:#a89f95;padding-top:30px;">'
                    f'<b style="color:#f3ece2;">{_html_mod.escape(str(r.get("name") or t))}</b> · {src}</div>',
                    unsafe_allow_html=True)
    components.html(scan_view.build_html(pd.DataFrame([r]), sec_label, sec_score, leader_hold, YF_TO_SECTOR),
                    height=96, scrolling=True)

    def ok(v):
        return str(v).lower() == 'true'
    hs = pd.to_numeric(pd.Series([r.get('held_score')]), errors='coerce').iloc[0]
    checks = [(not pd.isna(hs) and hs >= 1.0, f'held RS score {hs:+.1f} (needs +1 or higher)' if not pd.isna(hs) else 'held RS score n/a'),
              (ok(r.get('above_sma200')), 'close above the 200-day SMA'),
              (ok(r.get('sma50_above_sma200')), '50-day SMA above the 200-day')]
    mark = lambda b: ('<span style="color:#85cac5;font-weight:700;">✓</span>' if b
                      else '<span style="color:#f4a062;font-weight:700;">✗</span>')
    on_list = all(b for b, _ in checks)
    lines = [f'<b style="color:#f3ece2;">Momentum list:</b> '
             + ('<span style="color:#85cac5;">on the list</span>' if on_list else '<span style="color:#f4a062;">not on the list</span>')
             + ' — ' + ' · '.join(f'{mark(b)} {txt}' for b, txt in checks)]
    if on_list and in_scan:
        ml = sd[sd['_mom']].sort_values('rs_3m', ascending=False)['ticker'].astype(str).str.upper().tolist()
        if t in ml:
            lines.append(f'<b style="color:#f3ece2;">Rank:</b> #{ml.index(t) + 1} of {len(ml)} on the momentum list by RS 3M')
    if in_scan:
        pm = r.get('pass_mode')
        if isinstance(pm, str) and pm:
            lines.append(f'<b style="color:#f3ece2;">CAN SLIM:</b> {pm}')
        else:
            why = r.get('tech_fail_reason')
            why = '' if why is None or (isinstance(why, float) and np.isnan(why)) else str(why)
            why = why.replace('_', ' ').replace('sma', 'SMA ').replace('  ', ' ').strip()
            lines.append('<b style="color:#f3ece2;">CAN SLIM:</b> not passing' + (f' ({_html_mod.escape(why)})' if why else ''))
    else:
        lines.append('<b style="color:#f3ece2;">CAN SLIM:</b> not checked (outside the nightly universe)')
    etf = leader_hold.get(t)
    lines.append('<b style="color:#f3ece2;">Leader industry:</b> '
                 + (f'top-10 holding of {etf}' if etf else 'not a top-10 holding of a leader industry ETF'))
    c_txt, c_ch = st.columns([1.15, 1], gap='medium')
    with c_txt:
        st.markdown('<div style="font-size:0.78rem;color:#c9c1b7;line-height:1.9;margin:2px 0 6px 0;">'
                    + '<br>'.join(lines) + _extmom_lines(t) + '</div>', unsafe_allow_html=True)
    h = _chart_history(t)
    if h is not None and len(h) >= 30:
        with c_ch:
            st.plotly_chart(chart_view.figure(h, t, height=360), use_container_width=True,
                            config={'displayModeBar': False, 'scrollZoom': True}, key='lookup_chart')
    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)



if _scan_df is not None and len(_scan_df) > 0:
    _sd = _scan_df.copy()
    if 'ext_zone' in _sd.columns:   # labels before 28 Oct 2026
        _sd['ext_zone'] = _sd['ext_zone'].replace({'Stretched': 'Above', 'Extreme': 'Far above'})
    _has_mom = 'mom_list' in _sd.columns
    _bool = lambda c: _sd[c].astype(str).str.lower().eq('true') if c in _sd.columns else pd.Series(False, index=_sd.index)
    _sd['_mom'] = _bool('mom_list')
    _sd['_atr'] = _bool('atr_compressed')
    _sd['_sq'] = _bool('bb_kc_squeeze')
    _has_sqz = 'sqz_level' in _sd.columns
    _sd['_sqz_lvl'] = _sd['sqz_level'].fillna('').astype(str) if _has_sqz else ''
    _sd['_sqz_fired'] = _bool('sqz_fired')
    _sd['_sqz_start'] = _bool('sqz_started')
    for _c in ('held_score', 'rs_3m', 'ext_atr', 'rs_composite'):
        if _c in _sd.columns:
            _sd[_c] = pd.to_numeric(_sd[_c], errors='coerce')
    if not _has_mom:   # CSV from before the momentum scan: fall back to the CAN SLIM rows
        _sd['held_score'] = _sd.get('rs_composite')
        _sd['held_label'] = _sd.get('rs_label')

    _sec_label = dict(zip(_sec_df['Sector'], _sec_df['Label'])) if _sec_df is not None else {}
    _sec_score = dict(zip(_sec_df['Sector'], _sec_df['Score'])) if _sec_df is not None else {}
    _leader_hold = {}
    if _hold_df is not None and len(_hold_df) and len(_leaders):
        for _etf in _leaders['ETF']:
            for _t in _hold_df.loc[_hold_df['ETF'] == _etf, 'ticker']:
                _leader_hold.setdefault(str(_t), _etf)

    _scan_mrs = _sd['mrs_score'].iloc[0] if 'mrs_score' in _sd.columns else None
    _n_mom = int(_sd['_mom'].sum())
    st.markdown(
        f'<div style="font-size:0.78rem;color:#a89f95;margin-bottom:6px;">'
        f'Scan date <b style="color:#f3ece2;">{_scan_date}</b> · universe {len(_sd)} stocks · '
        f'<b style="color:#f3ece2;">{_n_mom}</b> on the momentum list · '
        f'{int((_sd["pass_mode"] == "STRICT").sum())} CAN SLIM strict · {int((_sd["pass_mode"] == "RELAXED").sum())} relaxed'
        + (f' · MRS at scan {float(_scan_mrs):+.2f}' if _scan_mrs is not None and pd.notna(_scan_mrs) else '')
        + ' · updated nightly 10 PM UTC</div>', unsafe_allow_html=True)
    if not _has_mom:
        st.info('The momentum columns arrive with the next nightly scan. Until then the table shows the CAN SLIM candidates.')

    _scan_lookup(_sd, _sec_label, _sec_score, _leader_hold)

    _LIST_OPTS = ['Momentum list', 'Momentum list that also meets CAN SLIM', 'CAN SLIM strict', 'CAN SLIM relaxed',
                  'All scanned stocks']
    _EXT_OPTS = {'Any extension': None,
                 'Close to the 50-day SMA (−1 to +0.5 ATR)': ['Just below', 'Near'],
                 'Above (+0.5 to +5 ATR)': ['Above'],
                 'Far above (above +5 ATR)': ['Far above'],
                 'Below the 50-day SMA (−1 ATR or lower)': ['Below', 'Deep below']}
    _f1, _f2, _f3 = st.columns([1.3, 1.4, 1.6])
    with _f1:
        _lmode = st.selectbox('List', _LIST_OPTS, index=0 if _has_mom else 2, key='ms_list')
    with _f2:
        _emode = st.selectbox('Extension vs the 50-day SMA', list(_EXT_OPTS.keys()), index=0, key='ms_ext')
    with _f3:
        _mz_sel = st.multiselect('50-day SMA momentum', ['Persistent', 'Building', 'Fading', 'Falling', 'Weak'],
                                 default=[], key='ms_mom', placeholder='Any momentum zone')
    _g1, _g2, _g3, _g4, _g5, _g6 = st.columns([1.3, 1.1, 1.1, 0.9, 1.3, 1.0])
    with _g1:
        _sects = sorted({YF_TO_SECTOR.get(str(s), str(s)) for s in _sd['sector'].dropna() if str(s)})
        _sect_f = st.selectbox('Sector', ['All sectors'] + _sects, index=0, key='ms_sector')
    with _g2:
        _lead_sec = st.checkbox('Leading sectors only', value=False, key='ms_leadsec', help='Sector RS score +1 or higher (held label).')
    with _g3:
        _lead_ind = st.checkbox('In a leader industry', value=False, key='ms_leadind', help='Top-10 holding of an industry ETF on the Leaders-to-buy list.')
    with _g4:
        _atr_f = st.checkbox('ATR compressed', value=False, key='ms_atr')
    with _g5:
        _SQ_OPTS = ['Any', 'In a squeeze (any level)', 'Mid or High compression', 'High compression only',
                    'Squeeze started today', 'Squeeze fired today'] if _has_sqz else ['Any', 'In a squeeze (any level)']
        _sq_f = st.selectbox('Squeeze', _SQ_OPTS, index=0, key='ms_sqz',
                             help='Beardy Squeeze Pro, daily: Bollinger 2.0 inside Keltner 2.0 / 1.5 / 1.0 = Low / Mid / High '
                                  'compression (black / red / orange dots). Fired = out of the squeeze today after a squeeze yesterday.')
    with _g6:
        _no_earn = st.checkbox('Hide earnings ≤ 7 days', value=False, key='ms_earn')

    _v = _sd
    if _lmode == 'Momentum list':
        _v = _v[_v['_mom']]
    elif _lmode == 'Momentum list that also meets CAN SLIM':
        _v = _v[_v['_mom'] & _v['pass_mode'].notna()]
    elif _lmode == 'CAN SLIM strict':
        _v = _v[_v['pass_mode'] == 'STRICT']
    elif _lmode == 'CAN SLIM relaxed':
        _v = _v[_v['pass_mode'] == 'RELAXED']
    if _EXT_OPTS[_emode] is not None and 'ext_zone' in _v.columns:
        _v = _v[_v['ext_zone'].isin(_EXT_OPTS[_emode])]
    if _mz_sel and 'mom_zone' in _v.columns:
        _v = _v[_v['mom_zone'].isin(_mz_sel)]
    _v_sect = _v['sector'].astype(str).map(lambda s: YF_TO_SECTOR.get(s, s))
    if _sect_f != 'All sectors':
        _v = _v[_v_sect == _sect_f]
        _v_sect = _v_sect[_v.index]
    if _lead_sec:
        _v = _v[_v_sect.map(lambda s: (_sec_score.get(s) or -9) >= 1.0)]
    if _lead_ind:
        _v = _v[_v['ticker'].astype(str).isin(_leader_hold.keys())]
    if _atr_f:
        _v = _v[_v['_atr']]
    if _sq_f == 'In a squeeze (any level)':
        _v = _v[_v['_sq']]
    elif _sq_f == 'Mid or High compression':
        _v = _v[_v['_sqz_lvl'].isin(['Mid', 'High'])]
    elif _sq_f == 'High compression only':
        _v = _v[_v['_sqz_lvl'] == 'High']
    elif _sq_f == 'Squeeze started today':
        _v = _v[_v['_sqz_start']]
    elif _sq_f == 'Squeeze fired today':
        _v = _v[_v['_sqz_fired']]
    if _no_earn and 'next_earnings' in _v.columns:
        _dd = (pd.to_datetime(_v['next_earnings'], errors='coerce') - pd.Timestamp.today().normalize()).dt.days
        _v = _v[~_dd.between(0, 7)]
    _v = _v.sort_values(['held_score', 'rs_3m'], ascending=[False, False], na_position='last')

    if _v.empty:
        st.info('No stocks match the selected filters.')
    else:
        _shown = _v.head(300)
        _scan_table_and_chart(scan_view.build_html(_shown, _sec_label, _sec_score, _leader_hold, YF_TO_SECTOR,
                                                   clickable=True),
                              list(_shown['ticker'].astype(str)), min(760, 64 + 32 * len(_shown)))
        st.markdown(
            '<div style="font-size:0.70rem;color:#9b958f;line-height:1.55;">'
            f'{len(_v)} stocks{" (first 300 shown)" if len(_v) > 300 else ""}. Click a header to sort. '
            'Momentum list = held RS score +1 or higher (beats SPY on 1Y / 6M / 3M, label held 3 sessions) AND close above the '
            '200-day SMA AND 50-day above 200-day. CAN SLIM = STRICT ≥ 25% EPS and revenue growth, RELAXED ≥ 20% / 15% '
            '(fundamentals are fetched only for stocks above their 50-day SMA, within 20% of the 52-week high). '
            'Ext vs SMA50 = (close − SMA50) / ATR(14): Deep below ≤ −3, Below ≤ −1, Just below ≤ 0, Near ≤ +0.5, '
            'Above ≤ +5, Far above beyond (Above is a trend state, not a reversal warning: in the 28 Oct study the left tail '
            'did not grow with extension in stocks). Momentum = 50-day SMA slope over 5 sessions: Weak ≤ −0.5%, Falling ≤ 0, '
            'Building ≤ +0.5 (Fading if it came down from above +0.5 in the last 11 sessions), Persistent above. '
            '5-day: dip ≤ −2%, pullback ≤ −3%, sharp ≤ −4%. Earnings in orange = within 7 days. '
            'Squeeze = Beardy Squeeze Pro on daily bars: Bollinger (20, 2.0) inside Keltner (SMA 20 ± k × average true range 20), '
            'High k = 1.0, Mid 1.5, Low 2.0; days = consecutive sessions in a squeeze; Fired ↑/↓ = out of the squeeze today, '
            'by momentum sign; the square is the momentum colour (aqua &gt; 0 rising, blue &gt; 0 falling, red &lt; 0 falling, yellow &lt; 0 rising). '
            + market_state.ev('Descriptive') + ' The momentum rules come from the industry study and the literature; the '
            'Ext·Mom table under the chart shows the selected stock\'s own history (up to 10 years) and a pooled stock panel, '
            'like the TradingView indicator. Where to start the chart work, not a buy signal.'
            '</div>', unsafe_allow_html=True)
else:
    st.info('No scanner results yet. The scanner runs nightly at 10 PM UTC (Mon–Fri) via GitHub Actions.')

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR PERFORMANCE & RELATIVE STRENGTH
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-header">Sectors and industries — relative strength vs SPY</div>',
            unsafe_allow_html=True)


# Debug: show the actual as-of date of the fetched data, and flag any ticker
# whose last date lags behind the most recent one (partial-fetch staleness).
if _sec_closes is not None and len(_sec_closes) > 0:
    _last_dates = {c: _sec_closes[c].dropna().index.max() for c in _sec_closes.columns}
    _max_date   = max(_last_dates.values())
    _stale      = {c: d for c, d in _last_dates.items() if d < _max_date}
    _staleness_msg = f'Sector data as of **{_max_date.strftime("%b %d, %Y")}**'
    if _stale:
        _lag_list = ', '.join(f'{c} ({d.strftime("%b %d")})' for c, d in _stale.items())
        _staleness_msg += f'  \n:warning: Lagging tickers: {_lag_list}'
    st.caption(_staleness_msg)
else:
    st.caption(':warning: Sector data fetch returned nothing.')

if _sec_df is not None and len(_sec_df) > 0:
    def _pct(v, decimals=1):
        if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
        sign = '+' if v > 0 else ''
        return f'{sign}{v * 100:.{decimals}f}%'

    def _cell_color(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return '#9b958f'
        return '#85cac5' if v > 0 else '#f4a062' if v < 0 else '#9b958f'

    def _score_color(v):
        if v >= 1.0:  return '#85cac5'
        if v >= 0.5:  return '#649995'
        if v <= -1.0: return '#f4a062'
        if v <= -0.5: return '#b9805a'
        return '#9b958f'

    TREND_CFG = {
        'IMPROVING': ('↑ IMPROVING',  '#85cac5', 'rgba(133,202,197,0.18)'),
        'FADING':    ('↓ FADING',     '#f4a062', 'rgba(244,160,98,0.18)'),
        'STABLE+':  ('→ STABLE',      '#649995', 'rgba(0,0,0,0)'),
        'STABLE-':  ('→ STABLE',      '#b9805a', 'rgba(0,0,0,0)'),
        'NEUTRAL':  ('— NEUTRAL',     '#9b958f', 'rgba(0,0,0,0)'),
    }

    # ── Sort controls ──────────────────────────────────────────────────────────
    SORT_OPTIONS = {
        'Composite Score': 'Score',
        'Days in label': 'DaysInLabel',
        'RS 1M':  'RS 1M',
        'RS 3M':  'RS 3M',
        'RS 6M':  'RS 6M',
        'RS 1Y':  'RS 1Y',
        '1M Return': '1M',
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

    # ── Sector -> industry -> top-10-holdings drill-down (sector_view.py) ──────
    _ind_df, _hold_df = load_industry_data()
    _ind_as_of = None
    if _ind_df is not None and 'as_of' in _ind_df.columns and len(_ind_df):
        _ind_as_of = str(_ind_df['as_of'].iloc[0])
    _drill_html = sector_view.build_drilldown_html(
        _sec_df, _ind_df, _hold_df, sort_col, sort_asc, ind_as_of=_ind_as_of)
    # Native <details> rows rendered in the page itself (no iframe, so no height limit).
    st.markdown(_drill_html, unsafe_allow_html=True)
else:
    st.info('Sector data temporarily unavailable.')


# ══════════════════════════════════════════════════════════════════════════════
# RISK DIAL (detail, collapsed: its one-line summary is in the decision strip)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander('Risk dial — how far price tends to travel from here (detail)', expanded=False):
    _bp = _band_profile(hist)
    st.markdown('<div class="section-header">Risk Dial — how far price tends to travel from here</div>',
                unsafe_allow_html=True)
    _rd_left, _rd_right = st.columns([1.15, 1], gap='large')

    with _rd_left:
        _g_title, _g_text = risk_dial.BAND_GUIDANCE.get(reg, ('', ''))
        st.markdown(f"""
        <div class="metric-card" style="border:1px solid #2a2724;">
          <div style="font-size:1.05rem;font-weight:800;color:{rcol};">{reg} · {_g_title}</div>
          <div style="font-size:0.84rem;color:#a89f95;margin-top:6px;line-height:1.5;">{_g_text}</div>
        </div>
        """, unsafe_allow_html=True)

        _rows = []
        for _h in risk_dial.HORIZONS:
            _b = _bp[(_bp['band'] == reg) & (_bp['horizon'] == _h)]
            _a = _bp[(_bp['band'] == 'ALL') & (_bp['horizon'] == _h)]
            if _b.empty or _a.empty:
                continue
            _b, _a = _b.iloc[0], _a.iloc[0]
            _x = int(round(_b['thresh'] * 100))
            _rows.append({
                'Sessions': _h,
                'Move size': f'±{_x}%',
                'Large adverse': f"{_b['p_adverse']*100:.0f}%  (all {_a['p_adverse']*100:.0f}%)",
                'Large favorable': f"{_b['p_favorable']*100:.0f}%  (all {_a['p_favorable']*100:.0f}%)",
                'Bad-case drawdown': f"{_b['mae_p10']*100:+.1f}%",
                'Good-case run-up': f"{_b['mfe_p90']*100:+.1f}%",
            })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        _n_band = int(_bp[(_bp['band'] == reg) & (_bp['horizon'] == 5)]['n'].iloc[0]) if len(_rows) else 0
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#9b958f;margin-top:2px;line-height:1.5;">
        Long entered at today's close, SPY, all {reg} sessions since {ctx['start'].strftime('%b %Y')} (n={_n_band}).
        Large adverse / favorable = worst drawdown / best gain from entry reaches the move size within the window.
        Bad-case drawdown = 10th percentile; good-case run-up = 90th percentile. Descriptive — the
        RISK-OFF vs RISK-ON contrast is confirmed after multiple-testing correction (Report v7.0 §4);
        the v4.0 weights were checked out of sample in the v8 study.
        </div>
        """, unsafe_allow_html=True)

    with _rd_right:
        _hsel = st.radio('Horizon (sessions)', risk_dial.HORIZONS, index=2, horizontal=True,
                         key='rd_horizon')
        _sub = _bp[_bp['horizon'] == _hsel].set_index('band')
        _bands_present = [b for b in risk_dial.BANDS if b in _sub.index]
        _labels = [b.replace('MILD ', 'MILD<br>') for b in _bands_present]
        _fig_rd = go.Figure()
        if reg in _bands_present:
            _i = _bands_present.index(reg)
            _fig_rd.add_vrect(x0=_i - 0.42, x1=_i + 0.42, fillcolor='rgba(227,180,142,0.10)',
                              line_width=0)
            _fig_rd.add_annotation(x=_i, y=1.0, yref='paper', text='Today', showarrow=False,
                                   font=dict(size=10, color='#e3b48e'), yanchor='bottom')
        for _col, _name, _clr in [('p_adverse', 'Large adverse move', ADV_COLOR),
                                  ('p_favorable', 'Large favorable move', FAV_COLOR)]:
            _y = (_sub.loc[_bands_present, _col] * 100).values
            _fig_rd.add_trace(go.Scatter(
                x=list(range(len(_bands_present))), y=_y, name=_name, mode='lines+markers',
                line=dict(color=_clr, width=2), marker=dict(size=9, color=_clr, line=dict(color='#0a0a0a', width=2)),
                customdata=_bands_present,
                hovertemplate='<b>%{customdata}</b><br>' + _name + ': %{y:.0f}% of sessions<extra></extra>',
            ))
            if 'ALL' in _sub.index:
                _fig_rd.add_hline(y=float(_sub.loc['ALL', _col]) * 100, line_width=1,
                                  line_color=_clr, opacity=0.35)
        _thr = int(round(risk_dial.THRESH[_hsel] * 100))
        _fig_rd.update_layout(
            template='plotly_dark+atn', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=270, margin=dict(l=10, r=10, t=24, b=10), hovermode='closest',
            legend=dict(orientation='h', yanchor='bottom', y=-0.42, x=0, font=dict(size=11, color='#a89f95')),
            xaxis=dict(tickmode='array', tickvals=list(range(len(_bands_present))), ticktext=_labels,
                       tickfont=dict(size=10, color='#a89f95'), showgrid=False, range=[-0.5, len(_bands_present) - 0.5]),
            yaxis=dict(title=dict(text=f'% of sessions, move ≥ {_thr}%', font=dict(size=10, color='#a89f95')),
                       ticksuffix='%', tickfont=dict(size=10, color='#a89f95'),
                       gridcolor='rgba(255,255,255,0.06)', rangemode='tozero'),
        )
        st.plotly_chart(_fig_rd, use_container_width=True)
        st.markdown('<div style="font-size:0.72rem;color:#9b958f;">Thin lines: same measure over all sessions. '
                    'The dial steps through all five bands — read it as how much room and size a trade '
                    'needs, not which way the market goes.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WHY THE MODEL SAYS THIS (diagnostics, in tabs)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header" style="margin-top:10px;">Why the model says this</div>', unsafe_allow_html=True)
_t_comp, _t_hist, _t_ch, _t_pc = st.tabs(['Score components & states', 'MRS, SPX and VIX history', 'Component history', 'Put/call context'])
with _t_comp:
    # ══════════════════════════════════════════════════════════════════════════════
    # TWO-COLUMN LAYOUT
    # ══════════════════════════════════════════════════════════════════════════════
    left, right = st.columns([1.4, 1], gap='large')

    with left:
        if _preview_active:
            st.markdown(
                '<div style="background:rgba(161,98,7,0.15);border:1px solid #e3b48e;border-radius:6px;'
                'padding:7px 12px;margin-bottom:8px;font-size:0.78rem;color:#e3b48e;">'
                '📥 Showing today\'s submitted inputs — CSV update pending (~90 sec, then Refresh Data)'
                '</div>',
                unsafe_allow_html=True
            )
        st.markdown('<div class="section-header">Component Breakdown</div>', unsafe_allow_html=True)
        COMP_DEF = [
            ('VIX',        'vix_phi',  'vix_score',  'vix_state',   'vix',        'vix'),
            ('Extension',  'ext_phi',  'ext_score',  'ext_state',   'ext_raw',    'ext'),
            ('Momentum',   'mom_phi',  'mom_score',  'mom_state',   'mom_raw',    'mom'),
            ('ADL Trend',  'adl_phi',  'adl_score',  'adl_state',   'adl_level',  'adl'),
            ('B20%',       'b20_phi',  'b20_score',  'b20_state',   'b20_pct',    'b20'),
            ('PC Ratio',   None,       'pc_score',   'pc_state',    'pc_sma10',   'pc'),
            ('SKEW',       'skew_phi', 'skew_score', 'skew_state',  'skew',       'skew'),
            ('Zero Gamma', None,       'gamma_score','gamma_state', 'zero_gamma', 'gamma'),
            ('Volume',     None,       'vol_score',  'vol_state',   'volume',     'vol'),
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
        _contrib_total = 0.0
        for name, phi_key, sc_key, st_key, raw_key, w_key in COMP_DEF:
            phi_v = last.get(phi_key, np.nan) if phi_key else np.nan
            sc_v  = last.get(sc_key, np.nan)
            st_v  = last.get(st_key, '—') or '—'
            raw_v = last.get(raw_key, np.nan)
            try: phi_f = f'{float(phi_v):.3f}' if not np.isnan(float(phi_v)) else '—'
            except: phi_f = '—'
            _w = pipeline.COMPONENT_WEIGHTS[w_key]
            try:
                sc_float = float(sc_v)
                sc_f = f'{sc_float:+.2f}' if not np.isnan(sc_float) else '—'
                ct_f = f'{sc_float * _w:+.2f}' if not np.isnan(sc_float) else '—'
                if not np.isnan(sc_float):
                    _contrib_total += sc_float * _w
            except:
                sc_f = '—'
                ct_f = '—'
            try:
                raw_float = float(raw_v)
                if np.isnan(raw_float):
                    raw_f = '—'
                elif name == 'Volume':
                    raw_f = f'{raw_float/1e6:.1f}M'  # Format volume in millions
                elif name in ('Extension', 'Momentum'):
                    raw_f = f'{raw_float*100:+.2f}%'  # % above SMA50 / 5-day % change of SMA50
                else:
                    raw_f = f'{raw_float:,.2f}'
            except:
                raw_f = '—'
            # Fix state display for NaN
            if st_v == 'nan' or (isinstance(st_v, float) and np.isnan(st_v)):
                st_v = '—'
            # Special handling for Volume state - show vs 20d avg
            if name == 'Volume':
                st_v = f'{st_v} · {vol_vs_20d} · info only'
            rows.append({'Component': name, 'Raw Value': raw_f,
                         'Phi': phi_f, 'State': str(st_v), 'Score': sc_f,
                         'Weight': f'×{_w:.2f}', 'Contribution': ct_f})
        rows.append({'Component': 'MRS composite', 'Raw Value': '', 'Phi': '', 'State': reg,
                     'Score': '', 'Weight': '', 'Contribution': f'{_contrib_total:+.2f}'})
        df_comp = pd.DataFrame(rows)
        def color_score(val):
            try:
                v = float(val)
                if v > 0: return 'color: #85cac5; font-weight: 700'
                if v < 0: return 'color: #f4a062; font-weight: 700'
                return 'color: #9b958f'
            except:
                return ''
        styled = df_comp.style.map(color_score, subset=['Score', 'Contribution'])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown("""
        <div style="font-size:0.72rem;color:#9b958f;margin-top:4px;">
        Phi = percentile rank over rolling 756-session window (3 years).
        Score = component state score (positive = calmer, negative = wider swings) ·
        Weight = MRS v4.0 weight: VIX is the anchor (scored continuously from its Phi and
        carrying the scale offset), Extension / Momentum / PC / SKEW / Breadth are an equal-weighted
        context block, ADL and B20 share the Breadth weight, Volume is information only ·
        Contribution = Score × Weight; contributions sum to the MRS composite (v8 study).
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">VIX Lifecycle State</div>', unsafe_allow_html=True)
        vix_phi   = last.get('vix_phi', np.nan)
        trig_days = float(last.get('trigger_days', 0) or 0)
        try: vix_phi_f = float(vix_phi)
        except: vix_phi_f = np.nan

        if not np.isnan(vix_phi_f) and vix_phi_f < 0.30:
            vix_row_cls = 'safe-row'
            vix_txt = (f'<span style="color:#85cac5">●</span> LOW VIX — VIX Phi = {vix_phi_f:.3f}. Calmest swings historically: large-move '
                       f'rate about half the all-sessions rate (0.49x at 10 sessions, 2008–2026). '
                       f'Calm bases can break — watch the context components.')
        elif not np.isnan(vix_phi_f) and vix_phi_f > 0.70:
            vix_row_cls = 'hazard-row'
            vix_txt = (f'<span style="color:#f4a062">●</span> HIGH VIX — VIX Phi = {vix_phi_f:.3f}. Widest two-sided swings: large-move '
                       f'rate 1.3–2x the all-sessions rate (High / Stress, 2008–2026). Both tails are wide.')
        elif not np.isnan(vix_phi_f):
            vix_row_cls = 'neutral-row'
            vix_txt = f'<span style="color:#e3b48e">●</span> MID RANGE — VIX Phi = {vix_phi_f:.3f}. Swings near to slightly below average.'
        else:
            vix_row_cls = 'neutral-row'
            vix_txt = '<span style="color:#a89f95">●</span> VIX Phi: no data'

        st.markdown(f'<div class="{vix_row_cls}">{vix_txt}</div>', unsafe_allow_html=True)

        if trig_days > 0:
            days_left = int(7 - trig_days)
            st.markdown(f'<div class="neutral-row"><span style="color:#a89f95">●</span> Compression exit — day {int(trig_days)} of 7 '
                        f'({days_left}d left). Reference only: the earlier suppression effect did not '
                        f'replicate in the v8 re-test (0.86x large-move rate, not significant).</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:16px;">Zero Gamma Position</div>',
                    unsafe_allow_html=True)
        try:
            spx_v = float(last.get('spx', np.nan))
            zg_v  = float(last.get('zero_gamma', np.nan))
            if not np.isnan(spx_v) and not np.isnan(zg_v) and zg_v > 0:
                dist_pct = (spx_v - zg_v) / spx_v * 100
                if dist_pct > 0.25:
                    gcls = 'safe-row'
                    gtxt = f'<span style="color:#85cac5">●</span> SPX {spx_v:,.0f} is {dist_pct:.1f}% ABOVE zero-gamma ({zg_v:,.0f}). Dealers long gamma — dampening environment.'
                elif dist_pct > -0.25:
                    gcls = 'neutral-row'
                    gtxt = f'<span style="color:#e3b48e">●</span> SPX {spx_v:,.0f} is NEAR zero-gamma ({zg_v:,.0f}, {dist_pct:+.1f}%). Transition zone — regime could flip.'
                else:
                    gcls = 'hazard-row'
                    gtxt = f'<span style="color:#f4a062">●</span> SPX {spx_v:,.0f} is {abs(dist_pct):.1f}% BELOW zero-gamma ({zg_v:,.0f}). Dealers short gamma — amplifying moves.'
                st.markdown(f'<div class="{gcls}">{gtxt}</div>', unsafe_allow_html=True)
            else:
                zg_only = float(last.get('zero_gamma', np.nan))
                if not np.isnan(zg_only) and zg_only > 0:
                    st.markdown(f'<div class="neutral-row"><span style="color:#a89f95">●</span> Zero Gamma level: <b>{zg_only:,.0f}</b> — SPX close pending</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="neutral-row"><span style="color:#a89f95">●</span> Zero Gamma: no data today</div>', unsafe_allow_html=True)
        except:
            try:
                zg_only = float(last.get('zero_gamma', np.nan))
                if not np.isnan(zg_only) and zg_only > 0:
                    st.markdown(f'<div class="neutral-row"><span style="color:#a89f95">●</span> Zero Gamma level: <b>{zg_only:,.0f}</b> — SPX close pending</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="neutral-row"><span style="color:#a89f95">●</span> Zero Gamma: no data today</div>', unsafe_allow_html=True)
            except:
                st.markdown('<div class="neutral-row"><span style="color:#a89f95">●</span> Zero Gamma: no data today</div>', unsafe_allow_html=True)

        # ── Recovery Signal (MRS v2.0) ─────────────────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:16px;">Recovery Signal</div>',
                    unsafe_allow_html=True)
        recovery = pipeline.compute_recovery_signal(last, hist, last_dt)
        rec_color = '#' + recovery['color']

        if recovery['active']:
            rec_cls = 'safe-row'
            rec_icon = '<span style="color:#85cac5">●</span>'
            rec_title = f"RECOVERY — {recovery['strength']}"
        elif recovery['signals'] >= 1:
            rec_cls = 'neutral-row'
            rec_icon = '<span style="color:#e3b48e">●</span>'
            rec_title = f"EARLY SIGNS ({recovery['signals']} signal{'s' if recovery['signals'] > 1 else ''})"
        else:
            rec_cls = 'neutral-row'
            rec_icon = '<span style="color:#a89f95">●</span>'
            rec_title = 'NO SIGNAL'

        st.markdown(f'''
        <div class="{rec_cls}">
          <span style="font-weight:700;color:{rec_color};">{rec_icon} {rec_title}</span><br>
          <span style="font-size:0.82rem;color:#a89f95;">{recovery["description"]}</span>
        </div>
        ''', unsafe_allow_html=True)

        if recovery['components']:
            comp_list = ' • '.join(recovery['components'])
            st.markdown(f'''
            <div style="font-size:0.75rem;color:#9b958f;margin-top:4px;padding-left:8px;">
              Components: {comp_list}
            </div>
            ''', unsafe_allow_html=True)

with _t_hist:
    # ══════════════════════════════════════════════════════════════════════════════
    # CHARTS
    # ══════════════════════════════════════════════════════════════════════════════
    hist90  = hist.dropna(subset=['mrs_score']).tail(90).copy()
    hist90['date'] = pd.to_datetime(hist90['date'])
    x_min   = hist.tail(90)['date'].min()
    x_max   = hist.tail(90)['date'].max()
    x_range = [x_min, x_max]

    VLINE_STYLE = dict(line_dash='dot', line_color='rgba(227,180,142,0.6)', line_width=1.5)
    LAYOUT_BASE = dict(
        template='plotly_dark+atn',
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
        (-0.5, 0.5,  'rgba(155,149,143,0.08)','NEUTRAL'),
        (-1.5, -0.5, 'rgba(217,119,6,0.10)',  'MILD RISK-OFF'),
        (-5.0, -1.5, 'rgba(185,28,28,0.12)',  'RISK-OFF'),
    ]
    _ylo = min(-4.5, float(hist90['mrs_score'].min()) - 0.5) if len(hist90) else -4.5
    _yhi = max(4.5, float(hist90['mrs_score'].max()) + 0.5) if len(hist90) else 4.5
    band_defs[0] = (1.5, 12.0, band_defs[0][2], band_defs[0][3])
    band_defs[-1] = (-12.0, -1.5, band_defs[-1][2], band_defs[-1][3])
    for y0, y1, fill, label in band_defs:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0)
        fig.add_annotation(
            x=1.01, xref='paper',
            y=(max(y0, _ylo) + min(y1, _yhi)) / 2, yref='y',
            text=label, showarrow=False,
            font=dict(size=9, color='#9b958f'),
            xanchor='left',
        )

    fig.add_trace(go.Scatter(
        x=hist90['date'], y=hist90['mrs_score'],
        mode='lines+markers', name='MRS',
        line=dict(color='#e8dccb', width=2.5),
        marker=dict(size=4, color='#e8dccb'),
        hovertemplate='<b>%{x|%b %d}</b><br>MRS: %{y:+.2f}<extra></extra>',
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.25)', line_width=1)
    fig.add_shape(type='line', x0=last_dt_str, x1=last_dt_str, y0=_ylo, y1=_yhi,
                  line=dict(dash='dot', color='rgba(227,180,142,0.6)', width=1.5))
    fig.add_annotation(x=last_dt_str, y=_yhi - 0.3, text='Today', showarrow=False,
                       font=dict(size=10, color='#e3b48e'), xanchor='left')
    fig.update_layout(**LAYOUT_BASE,
        margin=dict(l=10, r=130, t=10, b=30), height=300,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   tickformat='+.1f', range=[_ylo, _yhi], tickfont_size=11),
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
            line=dict(color='#9b958f', width=2),
            hovertemplate='<b>%{x|%b %d}</b><br>SPX: %{y:,.0f}<extra></extra>',
        ))
        if not np.isnan(zg_val):
            fig_spx.add_hline(
                y=zg_val,
                line_dash='dash',
                line_color='rgba(227,180,142,0.55)',
                line_width=1.2,
                annotation_text=f'Zero γ {zg_val:,.0f}',
                annotation_position='right',
                annotation_font_color='#e3b48e',
                annotation_font_size=10,
            )
        fig_spx.add_shape(type='line', x0=last_dt_str, x1=last_dt_str, y0=0, y1=1, yref='paper',
                      line=dict(dash='dot', color='rgba(227,180,142,0.6)', width=1.5))
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
                'compression': 'rgba(244,160,98,0.18)',
                'spike':       'rgba(227,180,142,0.14)',
                'mid':         'rgba(133,202,197,0.07)',
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
            line=dict(color='#9b958f', width=2),
            hovertemplate='<b>%{x|%b %d}</b><br>VIX: %{y:.2f}<extra></extra>',
        ))
        fig_vix.add_hline(y=20, line_dash='dot',
                          line_color='rgba(255,255,255,0.20)', line_width=1,
                          annotation_text='20', annotation_position='right',
                          annotation_font_color='#9b958f', annotation_font_size=10)
        fig_vix.add_shape(type='line', x0=last_dt_str, x1=last_dt_str, y0=0, y1=1, yref='paper',
                      line=dict(dash='dot', color='rgba(227,180,142,0.6)', width=1.5))
        fig_vix.update_layout(**LAYOUT_BASE,
            margin=dict(l=10, r=130, t=4, b=30), height=160,
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                       tickformat='.0f', tickfont_size=11,
                       range=[max(0, vix_min - vix_pad), vix_max + vix_pad]),
        )
        st.plotly_chart(fig_vix, use_container_width=True)

with _t_ch:
    # ══════════════════════════════════════════════════════════════════════════════
    # COMPONENT HISTORY CHARTS
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">Component History</div>', unsafe_allow_html=True)

    _PHI_CFG = [
        # (label, phi_col, line_color)
        ('VIX',       'vix_phi',  '#e8dccb'),
        ('Extension', 'ext_phi',  '#b9805a'),
        ('Momentum',  'mom_phi',  '#649995'),
        ('ADL',       'adl_phi',  '#9b958f'),
        ('B20%',      'b20_phi',  '#e3b48e'),
        ('SKEW',      'skew_phi', '#e8dccb'),
    ]

    # Prepare data — rows with at least one valid phi
    _hist_phi = hist90.copy()
    _phi_cols = [c for _, c, _ in _PHI_CFG if c in _hist_phi.columns]
    _hist_phi = _hist_phi[(_hist_phi[_phi_cols].fillna(0) != 0).any(axis=1)].copy()
    _hist_phi['date'] = pd.to_datetime(_hist_phi['date'])

    # Normalize SPX to [0, 1] for underlay overlay
    _spy_norm = None
    if 'spx' in _hist_phi.columns:
        _spy_raw = _hist_phi['spx'].astype(float)
        _spy_min, _spy_max = _spy_raw.min(), _spy_raw.max()
        if _spy_max > _spy_min:
            _spy_norm = (_spy_raw - _spy_min) / (_spy_max - _spy_min)

    # ── Chart A: Small multiples (2 × 3 grid) ────────────────────────────────────
    _fig_sm = make_subplots(
        rows=2, cols=3,
        subplot_titles=[lbl for lbl, _, _ in _PHI_CFG],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.14,
        horizontal_spacing=0.04,
    )

    _DANGER_FILL  = 'rgba(244,160,98,0.18)'
    _POSITIVE_FILL = 'rgba(133,202,197,0.18)'
    _DANGER_LINE   = 'rgba(244,160,98,0.45)'
    _POSITIVE_LINE = 'rgba(133,202,197,0.45)'

    for _i, (label, col, color) in enumerate(_PHI_CFG):
        _r, _c = _i // 3 + 1, _i % 3 + 1
        if col not in _hist_phi.columns:
            continue

        _y = _hist_phi[col].replace(0, np.nan)
        _x = _hist_phi['date']

        # SPX underlay (normalized to [0,1])
        if _spy_norm is not None:
            _fig_sm.add_trace(go.Scatter(
                x=_x, y=_spy_norm.values,
                mode='lines',
                line=dict(color='rgba(255,255,255,0.18)', width=1.2),
                showlegend=False,
                hoverinfo='skip',
            ), row=_r, col=_c)

        # Threshold bands — red = zone of wider swings, green = zone of calmer swings.
        # For VIX a HIGH Phi is the wide-swing zone, so its bands are flipped.
        _hi_wide = (label == 'VIX')
        _lo_fill, _hi_fill = ((_POSITIVE_FILL, _DANGER_FILL) if _hi_wide else (_DANGER_FILL, _POSITIVE_FILL))
        _lo_line, _hi_line = ((_POSITIVE_LINE, _DANGER_LINE) if _hi_wide else (_DANGER_LINE, _POSITIVE_LINE))
        _fig_sm.add_hrect(y0=0, y1=0.30, fillcolor=_lo_fill,
                          line_width=0, row=_r, col=_c)
        _fig_sm.add_hrect(y0=0.70, y1=1.0, fillcolor=_hi_fill,
                          line_width=0, row=_r, col=_c)
        _fig_sm.add_hline(y=0.30, line_dash='dot', line_color=_lo_line,
                          line_width=1, row=_r, col=_c)
        _fig_sm.add_hline(y=0.70, line_dash='dot', line_color=_hi_line,
                          line_width=1, row=_r, col=_c)

        # Phi line
        _fig_sm.add_trace(go.Scatter(
            x=_x, y=_y,
            mode='lines',
            line=dict(color=color, width=1.8),
            showlegend=False,
            hovertemplate=f'<b>{label}</b> %{{x|%b %d}}: Φ=%{{y:.3f}}<extra></extra>',
        ), row=_r, col=_c)

        # Current value dot (last non-NaN point)
        _valid_mask = _y.notna()
        if _valid_mask.any():
            _last_idx  = _valid_mask[_valid_mask].index[-1]
            _last_phi  = float(_y.loc[_last_idx])
            _last_date = _x.loc[_last_idx]
            _lo_c, _hi_c = (('#85cac5', '#f4a062') if _hi_wide else ('#f4a062', '#85cac5'))
            _dot_color = (_lo_c if _last_phi < 0.30
                          else _hi_c if _last_phi > 0.70
                          else color)
            _fig_sm.add_trace(go.Scatter(
                x=[_last_date], y=[_last_phi],
                mode='markers',
                marker=dict(color=_dot_color, size=7, line=dict(color='white', width=1)),
                showlegend=False, hoverinfo='skip',
            ), row=_r, col=_c)

    _fig_sm.update_layout(
        template='plotly_dark+atn',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=360,
        showlegend=False,
        margin=dict(l=0, r=0, t=36, b=0),
    )
    # Style all axes uniformly
    _fig_sm.update_yaxes(range=[0, 1], tickformat='.1f', tickfont_size=10, showgrid=False)
    _fig_sm.update_xaxes(tickformat='%b %d', tickfont_size=10, showgrid=False)
    # Style subplot titles
    for ann in _fig_sm.layout.annotations:
        ann.update(font=dict(size=11, color='#a89f95'))

    st.plotly_chart(_fig_sm, use_container_width=True)

    # ── Chart B: State heatmap ────────────────────────────────────────────────────
    st.markdown(
        '<p style="color:#9b958f;font-size:0.72rem;margin:2px 0 6px 0;">'
        'State heatmap (v4.0 scores) — red: component pushing toward wider swings · grey: normal state · '
        'green: pushing toward calmer swings. B20% and ADL score a high reading the same as a normal one (v4.0 re-test), '
        'so their rows are only ever red or grey.</p>',
        unsafe_allow_html=True,
    )

    # Build z matrix: rows=components, cols=dates
    _hm_labels = [lbl for lbl, _, _ in _PHI_CFG]
    _hm_dates  = _hist_phi['date'].tolist()
    _hm_z      = []
    _hm_text   = []

    _HM_SCORE = {'VIX': ('vix_score', None), 'Extension': ('ext_score', 'ext'), 'Momentum': ('mom_score', 'mom'),
                 'ADL': ('adl_score', 'adl'), 'B20%': ('b20_score', 'b20'), 'SKEW': ('skew_score', 'skew')}
    for label, col, _ in _PHI_CFG:
        _row_z    = []
        _row_text = []
        _sc_col, _nk = _HM_SCORE[label]
        _phis = (_hist_phi[col].replace(0, np.nan) if col in _hist_phi.columns
                 else pd.Series([np.nan] * len(_hist_phi), index=_hist_phi.index))
        _scs = (_hist_phi[_sc_col] if _sc_col in _hist_phi.columns
                else pd.Series([np.nan] * len(_hist_phi), index=_hist_phi.index))
        for phi_val, sc_val in zip(_phis, _scs):
            if pd.isna(phi_val) or pd.isna(sc_val):
                _row_z.append(np.nan)
                _row_text.append('—')
                continue
            # VIX is continuous: use its state bands (Low calm, Mid normal, High/Stress wide)
            if _nk is None:
                _d = 1.0 if phi_val < 0.30 else (-1.0 if phi_val >= 0.60 else 0.0)
            else:
                _d = float(sc_val) - pipeline.NEUTRAL_SCORE[_nk]
            _tol = 1e-9
            if _d < -_tol:
                _row_z.append(0.15); _st = 'toward wider swings'
            elif _d > _tol:
                _row_z.append(0.85); _st = 'toward calmer swings'
            else:
                _row_z.append(0.5); _st = 'normal'
            _row_text.append(f'Φ={phi_val:.3f}, score {float(sc_val):+.2f} ({_st})')
        _hm_z.append(_row_z)
        _hm_text.append(_row_text)

    # Custom colorscale: red → dark grey (neutral) → green, with sharp breaks at thresholds
    _hm_colorscale = [
        [0.00, '#74492e'],
        [0.20, '#f4a062'],
        [0.29, '#b9805a'],
        [0.30, '#1a1918'],   # sharp neutral start
        [0.70, '#1a1918'],   # sharp neutral end
        [0.71, '#649995'],
        [0.90, '#85cac5'],
        [1.00, '#3d5d5a'],
    ]

    _fig_hm = go.Figure(go.Heatmap(
        z=_hm_z,
        x=_hm_dates,
        y=_hm_labels,
        text=_hm_text,
        hovertemplate='<b>%{y}</b> %{x|%b %d}: %{text}<extra></extra>',
        colorscale=_hm_colorscale,
        zmin=0, zmax=1,
        showscale=False,
        xgap=1, ygap=2,
    ))

    _fig_hm.update_layout(
        template='plotly_dark+atn',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=180,
        margin=dict(l=0, r=0, t=4, b=0),
        xaxis=dict(tickformat='%b %d', tickfont_size=10, showgrid=False),
        yaxis=dict(tickfont_size=11, showgrid=False, autorange='reversed'),
    )
    st.plotly_chart(_fig_hm, use_container_width=True)

with _t_pc:
    # ══════════════════════════════════════════════════════════════════════════════
    # PC RATIO CONTEXT (collapsible)
    # ══════════════════════════════════════════════════════════════════════════════
    _pc_stale = risk_dial.pc_stale_sessions(hist)
    if _pc_stale >= 3:
        st.markdown(
            f'<div class="neutral-row"><span style="color:#e3b48e">●</span> PC ratio has carried the same value ({float(last.get("pc_ratio", np.nan)):.3f}) '
            f'for {_pc_stale} sessions. Check the input — enter the USI:PC close in the sidebar. '
            'Stale PC values caused most of the Apr–Sep 2026 drift (Report v7.0 §2.4).</div>',
            unsafe_allow_html=True)

    with st.expander('PC Ratio — Five-Zone Context (v4.0 scores)', expanded=_pc_stale >= 3):
        pc_sma10 = last.get('pc_sma10', np.nan)
        pc_daily = last.get('pc_ratio', np.nan)
        try:
            pc10 = float(pc_sma10)
            if pc10 < 0.686:
                zone, note = 'Extreme LOW', ('Complacency zone. Swings a little narrower than average '
                                             '(large-move rate 0.84x / 0.74x at 10 / 21 sessions). Score +0.25.')
                zcol = '#85cac5'
            elif pc10 < 0.732:
                zone, note = 'Moderate LOW', 'Slightly narrower swings than average (0.77x / 0.88x). Score +0.25.'
                zcol = '#649995'
            elif pc10 < 0.944:
                zone, note = 'Mid', 'Normal zone, slightly narrower swings than average (0.88x / 0.86x). Score +0.25.'
                zcol = '#9b958f'
            elif pc10 < 1.003:
                zone, note = 'Moderate HIGH', ('Fear building: wider swings (1.42x / 1.49x). Score −0.25.')
                zcol = '#e3b48e'
            else:
                zone, note = 'Extreme HIGH', ('Heavy put buying: the widest swings of any zone in BOTH directions '
                                              '(1.79x at 10 and 21 sessions). Its strong upside tail is part of that wide '
                                              'distribution, not a calm signal. Score −0.5 (was +1.0 in v3.0).')
                zcol = '#f4a062'
            col1, col2, col3 = st.columns(3)
            col1.metric('PC SMA-10', f'{pc10:.3f}')
            col2.metric('Daily PC', f'{float(pc_daily):.3f}' if not np.isnan(float(pc_daily)) else '—')
            col3.metric('Zone', zone)
            st.markdown(f"""
            <div style="background:#121212;border:1px solid #2a2724;border-radius:6px;
                        padding:10px 16px;font-size:0.86rem;margin-top:8px;">
            <b style="color:{zcol};">{zone}</b><br>{note}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            | Zone | SMA-10 | v4.0 score | Large-move rate 10D | 21D | v3.0 score |
            |------|--------|-----------|---------|---------|------|
            | Extreme LOW | < 0.686 | **+0.25** | 0.84x | 0.74x | +1.0 |
            | Moderate LOW | 0.686–0.732 | **+0.25** | 0.77x | 0.88x | +0.5 |
            | Mid | 0.732–0.944 | +0.25 | 0.88x | 0.86x | 0.0 |
            | Moderate HIGH | 0.944–1.003 | **−0.25** | 1.42x | 1.49x | −1.0 |
            | Extreme HIGH | ≥ 1.003 | **−0.5** | 1.79x | 1.79x | +1.0 |

            Large-move rate = share of sessions followed by a move of ±3% (10D) or ±5% (21D) in either
            direction, relative to all sessions, 2008–2026 (v8 study). Scores are for the dashboard's
            dispersion job: positive = calmer, negative = wider swings. Cutoffs sit at the ~10th / 18th /
            80th / 90th percentile of the 10-day average.
            """)
        except:
            st.write('PC SMA-10 data not available.')


# ── Footer ─────────────────────────────────────────────────────────────────────
last_upd = hist['date'].max()
st.markdown(
    '<div style="text-align:center;font-size:0.72rem;color:#6e675f;margin-top:24px;">'
    'Epistruct &nbsp;|&nbsp;'
    'Data through ' + last_upd.strftime('%B %d, %Y') + ' &nbsp;|&nbsp;'
    + pipeline.SCORING_VERSION + ' &nbsp;|&nbsp;'
    'Updates daily at 4:30 PM ET'
    '</div>',
    unsafe_allow_html=True,
)
