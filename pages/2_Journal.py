"""
pages/2_Journal.py — Trading Journal
=====================================
Single-page trade log backed by GitHub CSVs.
  • Log equity and options trades via web form
  • Close trades — MAE/MFE auto-computed from yfinance for equity
  • History table with date/type/setup filters
  • Analytics: hit rate, avg R, duration (winners vs losers), MAE/MFE box, setup table, regime matrix
  • Setup management: add, retire, restore

Required Streamlit secrets:
  APP_PASSWORD = "..."
  GH_PAT       = "ghp_..."   # Personal Access Token with repo write scope
  GH_REPO      = "fabiolmn17-web/MRS_Dashboard"   # optional — defaults to this
"""

import base64
import io
import time
from datetime import datetime, date
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Trading Journal',
    page_icon='📓',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── Password gate (mirrors app.py) ────────────────────────────────────────────
def _check_password() -> bool:
    if st.session_state.get('authenticated'):
        return True
    st.markdown('## 📓 Trading Journal')
    pwd = st.text_input('Password', type='password', key='j_pwd')
    if st.button('Enter', key='j_enter'):
        expected = st.secrets.get('APP_PASSWORD', '')
        if pwd == expected and expected:
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error('Incorrect password.')
    return False

if not _check_password():
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
_GH_REPO = st.secrets.get('GH_REPO', 'fabiolmn17-web/MRS_Dashboard')
_MRS_RAW = f'https://raw.githubusercontent.com/{_GH_REPO}/main/mrs_history.csv'
_GH_API  = 'https://api.github.com'

EQUITY_CSV  = 'JournalData/trades_equity.csv'
OPTIONS_CSV = 'JournalData/trades_options.csv'
SETUPS_CSV  = 'JournalData/setups.csv'

EQUITY_COLS = [
    'id', 'symbol', 'side', 'setup',
    'open_date', 'open_price', 'qty', 'stop',
    'chart_link', 'notes',
    'close_date', 'close_price',
    'mae_pct', 'mfe_pct',
    'pnl_pct', 'r_multiple', 'duration_days',
    'mrs_score_open', 'mrs_state_open',
    'status', 'created_at',
]
OPTIONS_COLS = [
    'id', 'underlying', 'strategy', 'contract_desc', 'side',
    'open_date', 'premium', 'qty', 'risk_usd', 'setup',
    'chart_link', 'notes',
    'close_date', 'close_premium',
    'pnl_pct', 'r_multiple', 'duration_days',
    'mrs_score_open', 'mrs_state_open',
    'status', 'created_at',
]
SETUP_COLS = ['name', 'description', 'active', 'created_date']

OPT_STRATEGIES = [
    'Long Call', 'Long Put',
    'Bull Put Spread', 'Bear Call Spread',
    'Bull Call Spread', 'Bear Put Spread',
    'Iron Condor', 'Iron Butterfly',
    'Covered Call', 'Cash-Secured Put',
    'Straddle', 'Strangle', 'Custom',
]

# ── PAT check ─────────────────────────────────────────────────────────────────
_GH_PAT = st.secrets.get('GH_PAT', '')
if not _GH_PAT:
    st.warning('⚠️ **GitHub PAT not configured.**')
    st.markdown(
        'Add the following to your Streamlit Cloud secrets to enable trade logging:\n\n'
        '```toml\nGH_PAT = "ghp_your_token_here"  # Personal Access Token — repo write scope\n```'
    )
    st.stop()

# ── GitHub IO ─────────────────────────────────────────────────────────────────
def _gh_headers() -> dict:
    return {'Authorization': f'token {_GH_PAT}', 'Accept': 'application/vnd.github.v3+json'}


def _gh_read(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return (df, sha) or (None, None) if file not found."""
    url = f'{_GH_API}/repos/{_GH_REPO}/contents/{path}'
    try:
        r = requests.get(url, headers=_gh_headers(), timeout=10)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        d = r.json()
        content = base64.b64decode(d['content']).decode('utf-8')
        return pd.read_csv(io.StringIO(content)), d['sha']
    except Exception as e:
        st.error(f'GitHub read error ({path}): {e}')
        return None, None


def _gh_write(path: str, df: pd.DataFrame, msg: str, sha: Optional[str] = None) -> bool:
    """Write df as CSV to GitHub. Returns True on success."""
    url = f'{_GH_API}/repos/{_GH_REPO}/contents/{path}'
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    payload: dict = {'message': msg, 'content': base64.b64encode(csv_bytes).decode('utf-8')}
    if sha:
        payload['sha'] = sha
    try:
        r = requests.put(url, headers=_gh_headers(), json=payload, timeout=15)
        return r.status_code in (200, 201)
    except Exception as e:
        st.error(f'GitHub write error ({path}): {e}')
        return False


def _load(path: str, cols: list) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load CSV from GitHub — session-state cached so subsequent calls within
    a single page run are instant.  Clear cache with _invalidate() after writes.
    """
    cache_key = f'_j_{path}'
    sha_key   = f'_j_sha_{path}'
    if cache_key not in st.session_state:
        df, sha = _gh_read(path)
        if df is None:
            df = pd.DataFrame(columns=cols)
            if _gh_write(path, df, f'init: {path}'):
                _, sha = _gh_read(path)
        else:
            for c in cols:
                if c not in df.columns:
                    df[c] = pd.NA
        st.session_state[cache_key] = df
        st.session_state[sha_key]   = sha
    return st.session_state[cache_key].copy(), st.session_state.get(sha_key)


def _invalidate(*paths: str) -> None:
    """Clear session-state cache for given CSV paths so next _load fetches fresh."""
    for p in paths:
        st.session_state.pop(f'_j_{p}', None)
        st.session_state.pop(f'_j_sha_{p}', None)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _current_mrs() -> Tuple[Optional[float], Optional[str]]:
    try:
        df = pd.read_csv(_MRS_RAW)
        if df.empty:
            return None, None
        last = df.iloc[-1]
        return float(last.get('mrs_score', 0)), str(last.get('mrs_state', ''))
    except Exception:
        return None, None


def _compute_mae_mfe(
    symbol: str, open_date: str, close_date: str,
    side: str, open_price: float,
) -> dict:
    """
    Fetch daily OHLC from yfinance for the hold period and return
    MAE% (adverse) and MFE% (favorable) relative to entry price.
    Both are sign-adjusted: MAE is negative (worst against you),
    MFE is positive (best in your favor).
    """
    import yfinance as yf
    try:
        start = (pd.Timestamp(open_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        end   = (pd.Timestamp(close_date) + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
        raw   = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty or 'High' not in raw.columns:
            return {}
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        mask = (raw.index >= pd.Timestamp(open_date)) & (raw.index <= pd.Timestamp(close_date))
        hold = raw[mask]
        if hold.empty:
            return {}
        if side == 'Long':
            worst = float(hold['Low'].min())
            best  = float(hold['High'].max())
            mae_pct = (worst - open_price) / open_price * 100   # negative
            mfe_pct = (best  - open_price) / open_price * 100   # positive
        else:  # Short
            worst = float(hold['High'].max())
            best  = float(hold['Low'].min())
            mae_pct = (open_price - worst) / open_price * 100   # negative
            mfe_pct = (open_price - best)  / open_price * 100   # positive
        return {'mae_pct': round(mae_pct, 2), 'mfe_pct': round(mfe_pct, 2)}
    except Exception:
        return {}


def _safe_float(val, default: float = np.nan) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _fmt_pct(v) -> str:
    try:
        return f'{float(v):+.1f}%'
    except (ValueError, TypeError):
        return '—'


def _fmt_r(v) -> str:
    try:
        return f'{float(v):.2f}R'
    except (ValueError, TypeError):
        return '—'


def _chart_icon(url) -> str:
    if url and str(url).startswith('http'):
        return f'[📊]({url})'
    return ''


def _days_open(open_date_str) -> int:
    try:
        return (date.today() - pd.Timestamp(open_date_str).date()).days
    except Exception:
        return 0

# ── Page header ───────────────────────────────────────────────────────────────
mrs_score, mrs_state = _current_mrs()
if mrs_score is not None:
    mrs_badge = f'MRS: **{mrs_state}** ({mrs_score:+.2f})'
else:
    mrs_badge = ''

hdr_col, ref_col = st.columns([10, 1])
hdr_col.markdown(f'## 📓 Trading Journal &nbsp;&nbsp;<small style="color:gray">{mrs_badge}</small>',
                 unsafe_allow_html=True)
if ref_col.button('🔄', help='Refresh journal data from GitHub'):
    _invalidate(EQUITY_CSV, OPTIONS_CSV, SETUPS_CSV)
    st.rerun()

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_log, tab_open, tab_hist, tab_analytics, tab_setups = st.tabs([
    '📝 Log Trade', '📂 Open Positions', '📚 History', '📊 Analytics', '⚙️ Setups',
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LOG TRADE
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:

    # Load active setups for dropdowns
    setups_df, _ = _load(SETUPS_CSV, SETUP_COLS)
    if not setups_df.empty and 'active' in setups_df.columns:
        active_setups = setups_df[setups_df['active'].astype(str) == 'True']['name'].tolist()
    else:
        active_setups = []
    if not active_setups:
        active_setups = ['(add setups in ⚙️ Setups tab)']

    eq_ltab, opt_ltab = st.tabs(['📈 Equity', '🎯 Options'])

    # ── Equity log form ───────────────────────────────────────────────────────
    with eq_ltab:
        with st.form('eq_log_form', clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            symbol_eq  = c1.text_input('Symbol', placeholder='AAPL')
            side_eq    = c2.selectbox('Side', ['Long', 'Short'])
            setup_eq   = c3.selectbox('Setup', active_setups)

            c4, c5, c6 = st.columns(3)
            open_date_eq  = c4.date_input('Open Date', value=date.today())
            open_price_eq = c5.number_input('Open Price ($)', min_value=0.01, value=100.0,
                                             step=0.01, format='%.2f')
            qty_eq        = c6.number_input('Qty (shares)', min_value=1, value=100, step=1)

            c7, c8 = st.columns(2)
            stop_eq       = c7.number_input('Stop Price ($)', min_value=0.0, value=0.0,
                                             step=0.01, format='%.2f',
                                             help='Used to compute R-multiple. Leave 0 to skip.')
            chart_link_eq = c8.text_input('Chart Link', placeholder='https://www.tradingview.com/x/...')

            notes_eq = st.text_area('Notes', height=80)
            sub_eq   = st.form_submit_button('📝 Log Equity Trade', type='primary', use_container_width=True)

        if sub_eq:
            sym = symbol_eq.strip().upper()
            if not sym:
                st.error('Symbol is required.')
            else:
                mrs_s, mrs_st = _current_mrs()
                new_row = {
                    'id':            f'EQ{int(time.time() * 1000)}',
                    'symbol':        sym,
                    'side':          side_eq,
                    'setup':         setup_eq,
                    'open_date':     open_date_eq.isoformat(),
                    'open_price':    open_price_eq,
                    'qty':           qty_eq,
                    'stop':          stop_eq if stop_eq > 0 else None,
                    'chart_link':    chart_link_eq.strip(),
                    'notes':         notes_eq.strip(),
                    'close_date':    None, 'close_price':  None,
                    'mae_pct':       None, 'mfe_pct':      None,
                    'pnl_pct':       None, 'r_multiple':   None, 'duration_days': None,
                    'mrs_score_open': mrs_s, 'mrs_state_open': mrs_st,
                    'status':        'Open',
                    'created_at':    datetime.now().isoformat(),
                }
                with st.spinner('Saving…'):
                    df, sha = _load(EQUITY_CSV, EQUITY_COLS)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    ok = _gh_write(EQUITY_CSV, df, f'trade: {sym} {side_eq} open {open_date_eq}', sha)
                if ok:
                    _invalidate(EQUITY_CSV)
                    mrs_tag = f' | MRS: {mrs_st} ({mrs_s:+.2f})' if mrs_s is not None else ''
                    st.success(f'✅ {sym} {side_eq} logged{mrs_tag}')
                else:
                    st.error('❌ Failed to save to GitHub.')

    # ── Options log form ──────────────────────────────────────────────────────
    with opt_ltab:
        with st.form('opt_log_form', clear_on_submit=True):
            o1, o2 = st.columns(2)
            underlying_opt = o1.text_input('Underlying', placeholder='SPY')
            strategy_opt   = o2.selectbox('Strategy', OPT_STRATEGIES)

            contract_opt = st.text_input('Contract Description',
                                          placeholder='SPY 500C exp 2026-03-20')

            o3, o4, o5 = st.columns(3)
            side_opt      = o3.selectbox('Side', ['Long', 'Short'], key='opt_side')
            open_date_opt = o4.date_input('Open Date', value=date.today(), key='opt_odate')
            setup_opt     = o5.selectbox('Setup', active_setups, key='opt_setup')

            o6, o7, o8 = st.columns(3)
            premium_opt  = o6.number_input('Premium / contract ($)', min_value=0.01,
                                            value=1.0, step=0.01, format='%.2f')
            qty_opt      = o7.number_input('Qty (contracts)', min_value=1, value=1, step=1)
            risk_usd_opt = o8.number_input('Max Risk ($)', min_value=0.0, value=0.0,
                                            step=10.0, format='%.0f',
                                            help='Total max loss on the position (for R calc)')

            chart_link_opt = st.text_input('Chart Link', placeholder='https://www.tradingview.com/x/...',
                                            key='opt_chart')
            notes_opt = st.text_area('Notes', height=80, key='opt_notes')
            sub_opt   = st.form_submit_button('📝 Log Options Trade', type='primary', use_container_width=True)

        if sub_opt:
            und = underlying_opt.strip().upper()
            if not und:
                st.error('Underlying is required.')
            else:
                mrs_s, mrs_st = _current_mrs()
                new_row = {
                    'id':             f'OPT{int(time.time() * 1000)}',
                    'underlying':     und,
                    'strategy':       strategy_opt,
                    'contract_desc':  contract_opt.strip(),
                    'side':           side_opt,
                    'open_date':      open_date_opt.isoformat(),
                    'premium':        premium_opt,
                    'qty':            qty_opt,
                    'risk_usd':       risk_usd_opt if risk_usd_opt > 0 else None,
                    'setup':          setup_opt,
                    'chart_link':     chart_link_opt.strip(),
                    'notes':          notes_opt.strip(),
                    'close_date':     None, 'close_premium': None,
                    'pnl_pct':        None, 'r_multiple':    None, 'duration_days': None,
                    'mrs_score_open': mrs_s, 'mrs_state_open': mrs_st,
                    'status':         'Open',
                    'created_at':     datetime.now().isoformat(),
                }
                with st.spinner('Saving…'):
                    df, sha = _load(OPTIONS_CSV, OPTIONS_COLS)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    ok = _gh_write(OPTIONS_CSV, df, f'trade: {und} {strategy_opt} open {open_date_opt}', sha)
                if ok:
                    _invalidate(OPTIONS_CSV)
                    st.success(f'✅ {und} {strategy_opt} logged')
                else:
                    st.error('❌ Failed to save to GitHub.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_open:
    eq_df,  eq_sha  = _load(EQUITY_CSV,  EQUITY_COLS)
    opt_df, opt_sha = _load(OPTIONS_CSV, OPTIONS_COLS)

    open_eq  = eq_df[ eq_df['status'] == 'Open'].copy()  if not eq_df.empty  else pd.DataFrame()
    open_opt = opt_df[opt_df['status'] == 'Open'].copy() if not opt_df.empty else pd.DataFrame()

    # ── Equity open table ─────────────────────────────────────────────────────
    st.markdown('#### 📈 Equity Positions')
    if open_eq.empty:
        st.caption('No open equity trades.')
    else:
        open_eq['Days Open'] = open_eq['open_date'].apply(_days_open)
        show_eq_cols = [c for c in
            ['symbol', 'side', 'setup', 'open_date', 'open_price', 'qty', 'stop', 'Days Open', 'mrs_state_open', 'chart_link']
            if c in open_eq.columns]
        disp = open_eq[show_eq_cols].rename(columns={
            'symbol': 'Symbol', 'side': 'Side', 'setup': 'Setup',
            'open_date': 'Open Date', 'open_price': 'Open $', 'qty': 'Qty',
            'stop': 'Stop $', 'mrs_state_open': 'MRS', 'chart_link': 'Chart',
        })
        if 'Chart' in disp.columns:
            disp['Chart'] = disp['Chart'].apply(_chart_icon)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Options open table ────────────────────────────────────────────────────
    st.markdown('#### 🎯 Options Positions')
    if open_opt.empty:
        st.caption('No open options trades.')
    else:
        open_opt['Days Open'] = open_opt['open_date'].apply(_days_open)
        show_opt_cols = [c for c in
            ['underlying', 'strategy', 'contract_desc', 'side', 'setup',
             'open_date', 'premium', 'qty', 'risk_usd', 'Days Open', 'mrs_state_open', 'chart_link']
            if c in open_opt.columns]
        disp_opt = open_opt[show_opt_cols].rename(columns={
            'underlying': 'Underlying', 'strategy': 'Strategy', 'contract_desc': 'Contract',
            'side': 'Side', 'setup': 'Setup', 'open_date': 'Open Date',
            'premium': 'Premium $', 'qty': 'Qty', 'risk_usd': 'Risk $',
            'mrs_state_open': 'MRS', 'chart_link': 'Chart',
        })
        if 'Chart' in disp_opt.columns:
            disp_opt['Chart'] = disp_opt['Chart'].apply(_chart_icon)
        st.dataframe(disp_opt, use_container_width=True, hide_index=True)

    # ── Close a Trade ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown('#### Close a Trade')
    cl_eq_col, cl_opt_col = st.columns(2)

    # Close equity
    with cl_eq_col:
        st.markdown('**Equity**')
        if open_eq.empty:
            st.caption('No open equity positions.')
        else:
            eq_labels = {
                f"{r['symbol']} {r['side']} @ ${_safe_float(r['open_price']):.2f}  ({r['open_date']})": r['id']
                for _, r in open_eq.iterrows()
            }
            sel_eq_label = st.selectbox('Select trade', list(eq_labels.keys()), key='close_eq_sel')
            sel_eq_id    = eq_labels[sel_eq_label]
            tr_eq        = open_eq[open_eq['id'] == sel_eq_id].iloc[0]

            with st.form('close_eq_form'):
                ce1, ce2 = st.columns(2)
                close_date_eq  = ce1.date_input('Close Date', value=date.today())
                close_price_eq = ce2.number_input(
                    'Close Price ($)',
                    min_value=0.01,
                    value=_safe_float(tr_eq['open_price'], 100.0),
                    step=0.01, format='%.2f',
                )
                sub_close_eq = st.form_submit_button(
                    '⚡ Close & Compute MAE/MFE', type='primary', use_container_width=True,
                )

            if sub_close_eq:
                open_p = _safe_float(tr_eq['open_price'])
                side_v = str(tr_eq['side'])
                pnl_pct = (
                    (close_price_eq - open_p) / open_p * 100 if side_v == 'Long'
                    else (open_p - close_price_eq) / open_p * 100
                )
                stop_v = _safe_float(tr_eq.get('stop'), default=np.nan)
                if not np.isnan(stop_v) and stop_v > 0:
                    risk_ps = abs(open_p - stop_v)
                    pnl_ps  = (close_price_eq - open_p) * (1 if side_v == 'Long' else -1)
                    r_mult  = round(pnl_ps / risk_ps, 2) if risk_ps > 0 else None
                else:
                    r_mult = None
                dur = (close_date_eq - pd.Timestamp(tr_eq['open_date']).date()).days

                with st.spinner('Fetching price history for MAE/MFE…'):
                    mf = _compute_mae_mfe(
                        str(tr_eq['symbol']), str(tr_eq['open_date']),
                        close_date_eq.isoformat(), side_v, open_p,
                    )

                idx = eq_df[eq_df['id'] == sel_eq_id].index[0]
                eq_df.at[idx, 'close_date']    = close_date_eq.isoformat()
                eq_df.at[idx, 'close_price']   = close_price_eq
                eq_df.at[idx, 'mae_pct']       = mf.get('mae_pct')
                eq_df.at[idx, 'mfe_pct']       = mf.get('mfe_pct')
                eq_df.at[idx, 'pnl_pct']       = round(pnl_pct, 2)
                eq_df.at[idx, 'r_multiple']    = r_mult
                eq_df.at[idx, 'duration_days'] = dur
                eq_df.at[idx, 'status']        = 'Closed'

                with st.spinner('Saving…'):
                    ok = _gh_write(EQUITY_CSV, eq_df, f"close: {tr_eq['symbol']} {close_date_eq}", eq_sha)
                if ok:
                    _invalidate(EQUITY_CSV)
                    parts = [f"P&L: {pnl_pct:+.1f}%"]
                    if mf.get('mae_pct') is not None:
                        parts += [f"MAE: {mf['mae_pct']:+.1f}%", f"MFE: {mf['mfe_pct']:+.1f}%"]
                    if r_mult is not None:
                        parts.append(f"R: {r_mult:.2f}")
                    st.success(f"✅ Closed {tr_eq['symbol']} | {' | '.join(parts)}")
                    st.rerun()
                else:
                    st.error('❌ Failed to save.')

    # Close options
    with cl_opt_col:
        st.markdown('**Options**')
        if open_opt.empty:
            st.caption('No open options positions.')
        else:
            opt_labels = {
                f"{r['underlying']} {r['strategy']} ({r['open_date']})": r['id']
                for _, r in open_opt.iterrows()
            }
            sel_opt_label = st.selectbox('Select trade', list(opt_labels.keys()), key='close_opt_sel')
            sel_opt_id    = opt_labels[sel_opt_label]
            tr_opt        = open_opt[open_opt['id'] == sel_opt_id].iloc[0]

            with st.form('close_opt_form'):
                co1, co2 = st.columns(2)
                close_date_opt = co1.date_input('Close Date', value=date.today())
                close_prem_opt = co2.number_input(
                    'Close Premium ($)',
                    min_value=0.0,
                    value=_safe_float(tr_opt['premium'], 1.0),
                    step=0.01, format='%.2f',
                )
                sub_close_opt = st.form_submit_button(
                    '⚡ Close Options Trade', type='primary', use_container_width=True,
                )

            if sub_close_opt:
                open_prem = _safe_float(tr_opt['premium'])
                side_v    = str(tr_opt['side'])
                pnl_pct   = (
                    (close_prem_opt - open_prem) / open_prem * 100 if side_v == 'Long'
                    else (open_prem - close_prem_opt) / open_prem * 100
                )
                dur = (close_date_opt - pd.Timestamp(tr_opt['open_date']).date()).days

                risk_v = _safe_float(tr_opt.get('risk_usd'), default=np.nan)
                if not np.isnan(risk_v) and risk_v > 0:
                    qty_v = _safe_float(tr_opt.get('qty'), default=1.0)
                    pnl_usd = (close_prem_opt - open_prem) * qty_v * 100 * (1 if side_v == 'Long' else -1)
                    r_mult_opt = round(pnl_usd / risk_v, 2)
                else:
                    r_mult_opt = None

                idx = opt_df[opt_df['id'] == sel_opt_id].index[0]
                opt_df.at[idx, 'close_date']    = close_date_opt.isoformat()
                opt_df.at[idx, 'close_premium'] = close_prem_opt
                opt_df.at[idx, 'pnl_pct']       = round(pnl_pct, 2)
                opt_df.at[idx, 'r_multiple']    = r_mult_opt
                opt_df.at[idx, 'duration_days'] = dur
                opt_df.at[idx, 'status']        = 'Closed'

                with st.spinner('Saving…'):
                    ok = _gh_write(OPTIONS_CSV, opt_df, f"close: {tr_opt['underlying']} {close_date_opt}", opt_sha)
                if ok:
                    _invalidate(OPTIONS_CSV)
                    parts = [f"P&L: {pnl_pct:+.1f}%"]
                    if r_mult_opt is not None:
                        parts.append(f"R: {r_mult_opt:.2f}")
                    st.success(f"✅ Closed {tr_opt['underlying']} {tr_opt['strategy']} | {' | '.join(parts)}")
                    st.rerun()
                else:
                    st.error('❌ Failed to save.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    eq_df,  _ = _load(EQUITY_CSV,  EQUITY_COLS)
    opt_df, _ = _load(OPTIONS_CSV, OPTIONS_COLS)

    closed_eq  = eq_df[ eq_df['status'] == 'Closed'].copy()  if not eq_df.empty  else pd.DataFrame()
    closed_opt = opt_df[opt_df['status'] == 'Closed'].copy() if not opt_df.empty else pd.DataFrame()

    # ── Filter bar ────────────────────────────────────────────────────────────
    hf1, hf2, hf3 = st.columns(3)
    h_type    = hf1.selectbox('Type', ['All', 'Equity', 'Options'])
    h_from    = hf2.date_input('From', value=date(date.today().year, 1, 1))
    h_to      = hf3.date_input('To', value=date.today())

    hist_rows = []

    if h_type in ('All', 'Equity') and not closed_eq.empty:
        for _, r in closed_eq.iterrows():
            od = str(r.get('open_date', ''))[:10]
            if od < h_from.isoformat() or od > h_to.isoformat():
                continue
            hist_rows.append({
                'Type':   '📈 Equity',
                'Symbol': r.get('symbol', ''),
                'Side':   r.get('side', ''),
                'Strategy': '—',
                'Setup':  r.get('setup', ''),
                'Open':   od,
                'Close':  str(r.get('close_date', ''))[:10],
                'Days':   r.get('duration_days'),
                'P&L%':   _fmt_pct(r.get('pnl_pct')),
                'R':      _fmt_r(r.get('r_multiple')),
                'MAE%':   _fmt_pct(r.get('mae_pct')),
                'MFE%':   _fmt_pct(r.get('mfe_pct')),
                'MRS':    r.get('mrs_state_open', ''),
                'Chart':  _chart_icon(str(r.get('chart_link', ''))),
            })

    if h_type in ('All', 'Options') and not closed_opt.empty:
        for _, r in closed_opt.iterrows():
            od = str(r.get('open_date', ''))[:10]
            if od < h_from.isoformat() or od > h_to.isoformat():
                continue
            hist_rows.append({
                'Type':     '🎯 Options',
                'Symbol':   r.get('underlying', ''),
                'Side':     r.get('side', ''),
                'Strategy': r.get('strategy', ''),
                'Setup':    r.get('setup', ''),
                'Open':     od,
                'Close':    str(r.get('close_date', ''))[:10],
                'Days':     r.get('duration_days'),
                'P&L%':     _fmt_pct(r.get('pnl_pct')),
                'R':        _fmt_r(r.get('r_multiple')),
                'MAE%':     '—',
                'MFE%':     '—',
                'MRS':      r.get('mrs_state_open', ''),
                'Chart':    _chart_icon(str(r.get('chart_link', ''))),
            })

    if hist_rows:
        # Sort by Open date descending
        hist_df = pd.DataFrame(hist_rows).sort_values('Open', ascending=False)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        st.caption(f'{len(hist_df)} closed trades')
    else:
        st.info('No closed trades in the selected range.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    eq_df,  _ = _load(EQUITY_CSV,  EQUITY_COLS)
    opt_df, _ = _load(OPTIONS_CSV, OPTIONS_COLS)

    closed_eq  = eq_df[ eq_df['status'] == 'Closed'].copy()  if not eq_df.empty  else pd.DataFrame()
    closed_opt = opt_df[opt_df['status'] == 'Closed'].copy() if not opt_df.empty else pd.DataFrame()

    total_closed = len(closed_eq) + len(closed_opt)

    if total_closed < 3:
        st.info(f'Close at least 3 trades to see analytics ({total_closed} closed so far).')
    else:
        # Build unified frame
        parts = []
        if not closed_eq.empty:
            p = closed_eq[['setup', 'side', 'pnl_pct', 'r_multiple', 'duration_days',
                            'mrs_state_open', 'mae_pct', 'mfe_pct']].copy()
            p['type'] = 'Equity'
            parts.append(p)
        if not closed_opt.empty:
            p = closed_opt[['setup', 'side', 'pnl_pct', 'r_multiple', 'duration_days',
                             'mrs_state_open']].copy()
            p['type'] = 'Options'
            p['mae_pct'] = np.nan
            p['mfe_pct'] = np.nan
            parts.append(p)

        all_c = pd.concat(parts, ignore_index=True)
        all_c['pnl_pct']      = pd.to_numeric(all_c['pnl_pct'],      errors='coerce')
        all_c['r_multiple']   = pd.to_numeric(all_c['r_multiple'],   errors='coerce')
        all_c['duration_days']= pd.to_numeric(all_c['duration_days'],errors='coerce')
        all_c['win']          = all_c['pnl_pct'] > 0

        n   = len(all_c)
        hit = all_c['win'].sum()

        r_vals  = all_c['r_multiple'].dropna()
        avg_r   = r_vals.mean()    if len(r_vals) > 0 else np.nan
        avg_dur = all_c['duration_days'].dropna().mean()
        avg_pnl = all_c['pnl_pct'].dropna().mean()

        # ── Summary metrics ───────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Hit Rate',     f'{hit/n*100:.0f}%',            f'{int(hit)}/{n} wins')
        m2.metric('Avg R',        f'{avg_r:.2f}R' if not np.isnan(avg_r) else '—',
                  'equity + options')
        m3.metric('Avg Duration', f'{avg_dur:.0f}d' if not np.isnan(avg_dur) else '—')
        m4.metric('Avg P&L%',     f'{avg_pnl:+.1f}%' if not np.isnan(avg_pnl) else '—')

        st.divider()

        # ── Duration: winners vs losers ───────────────────────────────────────
        dur_data = all_c[all_c['duration_days'].notna()].copy()
        if not dur_data.empty:
            dur_win  = dur_data[dur_data['win'] == True]['duration_days'].mean()
            dur_loss = dur_data[dur_data['win'] == False]['duration_days'].mean()
            st.markdown('**Hold Duration — Winners vs Losers**')
            fig_dur = go.Figure()
            if not np.isnan(dur_win):
                fig_dur.add_trace(go.Bar(
                    name='Winners', x=['Winners'], y=[dur_win],
                    marker_color='#26a69a',
                    text=[f'{dur_win:.0f}d'], textposition='auto',
                ))
            if not np.isnan(dur_loss):
                fig_dur.add_trace(go.Bar(
                    name='Losers', x=['Losers'], y=[dur_loss],
                    marker_color='#ef5350',
                    text=[f'{dur_loss:.0f}d'], textposition='auto',
                ))
            fig_dur.update_layout(
                height=260, showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis_title='Avg days',
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            )
            left, _ = st.columns([2, 3])
            left.plotly_chart(fig_dur, use_container_width=True)

        # ── MAE / MFE (equity) ────────────────────────────────────────────────
        if not closed_eq.empty:
            mae_s = pd.to_numeric(closed_eq['mae_pct'], errors='coerce').dropna()
            mfe_s = pd.to_numeric(closed_eq['mfe_pct'], errors='coerce').dropna()
            if len(mae_s) > 0 or len(mfe_s) > 0:
                st.markdown('**MAE / MFE Distribution (Equity)**')
                fig_mf = go.Figure()
                if len(mae_s) > 0:
                    fig_mf.add_trace(go.Box(
                        y=mae_s, name='MAE %', marker_color='#ef5350', boxmean=True,
                    ))
                if len(mfe_s) > 0:
                    fig_mf.add_trace(go.Box(
                        y=mfe_s, name='MFE %', marker_color='#26a69a', boxmean=True,
                    ))
                fig_mf.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=20, b=0),
                    yaxis_title='% from entry (MAE negative, MFE positive)',
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                left2, _ = st.columns([2, 3])
                left2.plotly_chart(fig_mf, use_container_width=True)

        st.divider()

        # ── Setup analysis ────────────────────────────────────────────────────
        st.markdown('**Setup Analysis**')
        setup_valid = all_c[all_c['setup'].notna() & (all_c['setup'] != '')].copy()
        if not setup_valid.empty:
            grp = (
                setup_valid.groupby('setup')
                .agg(
                    Trades=('pnl_pct', 'count'),
                    Wins=('win', 'sum'),
                )
                .reset_index()
            )
            grp['Hit Rate'] = (grp['Wins'] / grp['Trades'] * 100).map(lambda x: f'{x:.0f}%')

            # Avg P&L per setup
            avg_pnl_grp = setup_valid.groupby('setup')['pnl_pct'].mean().map(_fmt_pct)
            grp['Avg P&L%'] = grp['setup'].map(avg_pnl_grp)

            # Avg R per setup (from equity only)
            if not closed_eq.empty:
                eq_setup = closed_eq.copy()
                eq_setup['r_multiple'] = pd.to_numeric(eq_setup['r_multiple'], errors='coerce')
                avg_r_grp = eq_setup.groupby('setup')['r_multiple'].mean().map(
                    lambda v: f'{v:.2f}R' if not np.isnan(v) else '—'
                )
                grp['Avg R'] = grp['setup'].map(avg_r_grp).fillna('—')
            else:
                grp['Avg R'] = '—'

            # Avg duration per setup
            avg_dur_grp = setup_valid.groupby('setup')['duration_days'].mean().map(
                lambda v: f'{v:.0f}d' if not np.isnan(v) else '—'
            )
            grp['Avg Dur'] = grp['setup'].map(avg_dur_grp)

            grp = grp.rename(columns={'setup': 'Setup'}).drop(columns=['Wins'])
            st.dataframe(grp, use_container_width=True, hide_index=True)

        # ── Regime matrix ─────────────────────────────────────────────────────
        st.markdown('**Regime Matrix** — Hit rate by Setup × MRS State')
        mat = all_c[
            all_c['setup'].notna() & (all_c['setup'] != '') &
            all_c['mrs_state_open'].notna() & (all_c['mrs_state_open'] != '')
        ].copy()
        if len(mat) >= 3:
            pivot = mat.pivot_table(
                index='setup', columns='mrs_state_open', values='win',
                aggfunc=lambda x: f'{x.mean()*100:.0f}% ({len(x)})',
            )
            st.dataframe(pivot, use_container_width=True)
        else:
            st.caption('Needs trades across multiple regimes to populate.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SETUPS
# ══════════════════════════════════════════════════════════════════════════════
with tab_setups:
    setups_df, setups_sha = _load(SETUPS_CSV, SETUP_COLS)

    st.markdown('#### Setups')
    if setups_df.empty:
        st.caption('No setups yet — add your first below.')
    else:
        for _, row in setups_df.iterrows():
            is_active = str(row.get('active', 'True')) == 'True'
            rc1, rc2, rc3, rc4 = st.columns([1, 3, 5, 1])
            rc1.markdown('✅' if is_active else '🚫')
            rc2.markdown(f"**{row['name']}**")
            rc3.caption(str(row.get('description', '')))
            btn_label = 'Retire' if is_active else 'Restore'
            if rc4.button(btn_label, key=f"tog_{row['name']}"):
                idx = setups_df[setups_df['name'] == row['name']].index[0]
                setups_df.at[idx, 'active'] = not is_active
                with st.spinner('Saving…'):
                    ok = _gh_write(
                        SETUPS_CSV, setups_df,
                        f"setup: {'retire' if is_active else 'restore'} {row['name']}",
                        setups_sha,
                    )
                if ok:
                    _invalidate(SETUPS_CSV)
                    st.rerun()

    st.divider()
    st.markdown('#### Add Setup')
    with st.form('add_setup_form', clear_on_submit=True):
        as1, as2 = st.columns([2, 4])
        new_name = as1.text_input('Name', placeholder='Momentum Breakout')
        new_desc = as2.text_input('Description', placeholder='Above SMA50, high RS, volume surge on breakout')
        sub_setup = st.form_submit_button('➕ Add Setup', type='primary')

    if sub_setup:
        name = new_name.strip()
        if not name:
            st.error('Name is required.')
        elif not setups_df.empty and name in setups_df['name'].values:
            st.warning(f'"{name}" already exists.')
        else:
            new_s = {
                'name': name, 'description': new_desc.strip(),
                'active': True, 'created_date': date.today().isoformat(),
            }
            setups_df = pd.concat([setups_df, pd.DataFrame([new_s])], ignore_index=True)
            with st.spinner('Saving…'):
                ok = _gh_write(SETUPS_CSV, setups_df, f'setup: add {name}', setups_sha)
            if ok:
                _invalidate(SETUPS_CSV)
                st.success(f'✅ Setup "{name}" added.')
                st.rerun()
            else:
                st.error('❌ Failed to save.')
