
"""
app.py — MRS Live Dashboard (Streamlit)
========================================
Password-protected. Reads mrs_history.csv and displays:
  • Current regime score + signal quality
  • 8-component breakdown table
  • VIX lifecycle state layer (paper-grounded: compression / mid / spike-zone)
  • Zero Gamma position
  • 90-day MRS history chart
  • Regime duration counter
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import date

import pipeline

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

# ── Load data ──────────────────────────────────────────────────────────────────
HIST_PATH = Path(__file__).parent / 'mrs_history.csv'

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    df = pipeline.load_history(HIST_PATH)
    return df

hist = load_data()
complete = hist.dropna(subset=['vix', 'mrs_score'])
last     = complete.iloc[-1].to_dict() if len(complete) else hist.iloc[-1].to_dict()
last_dt  = pd.Timestamp(last['date'])

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


# ── Helper functions ────────────────────────────────────────────────────────────
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
      <div style="font-size:0.78rem;color:#c4cad6;margin-top:8px;line-height:1.5;">
        {sq_desc[:220]}{'…' if len(sq_desc) > 220 else ''}
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
      <div style="font-size:0.88rem;line-height:2.2;color:#e2e8f0;">
        <b>VIX</b> {vix_now}<br>
        <b>SPX</b> {spx_now}<br>
        <b>Zero γ</b> {zg_now}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT: Components (left) | VIX State + Gamma (right)
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1.4, 1], gap='large')

# ── LEFT: Component breakdown ──────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-header">Component Breakdown</div>', unsafe_allow_html=True)

    COMP_DEF = [
        ('VIX',        'vix_phi',  'vix_score',  'vix_state',   'vix'),
        ('Extension',  'ext_phi',  'ext_score',  'ext_state',   'spy'),
        ('Momentum',   'mom_phi',  'mom_score',  'mom_state',   'spy'),
        ('ADL Trend',  'adl_phi',  'adl_score',  'adl_state',   'adl_level'),
        ('B20%',       'b20_phi',  'b20_score',  'b20_state',   'b20_pct'),
        ('PC Ratio',   None,       'pc_score',   'pc_state',    'pc_sma10'),
        ('SKEW',       'skew_phi', 'skew_score', 'skew_state',  'skew'),
        ('Zero Gamma', None,       'gamma_score','gamma_state', 'zero_gamma'),
    ]

    rows = []
    for name, phi_key, sc_key, st_key, raw_key in COMP_DEF:
        phi_v  = last.get(phi_key, np.nan) if phi_key else np.nan
        sc_v   = last.get(sc_key, 0) or 0
        st_v   = last.get(st_key, '—') or '—'
        raw_v  = last.get(raw_key, np.nan)

        try: phi_f = f'{float(phi_v):.3f}' if not np.isnan(float(phi_v)) else '—'
        except: phi_f = '—'

        try: sc_f = f'{float(sc_v):+.1f}'
        except: sc_f = '0.0'

        try: raw_f = f'{float(raw_v):,.2f}' if not np.isnan(float(raw_v)) else '—'
        except: raw_f = '—'

        rows.append({
            'Component': name,
            'Raw Value': raw_f,
            'Phi': phi_f,
            'State': str(st_v),
            'Score': sc_f,
        })

    df_comp = pd.DataFrame(rows)

    def color_score(val):
        try:
            v = float(val)
            if v > 0:  return 'color: #22c55e; font-weight: 700'
            if v < 0:  return 'color: #ef4444; font-weight: 700'
            return 'color: #9ca3af'
        except:
            return ''

    styled = df_comp.style.map(color_score, subset=['Score'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="font-size:0.72rem;color:#8892a4;margin-top:4px;">
    Phi = percentile rank over rolling 756-session window (3 years).
    Score = discretized contribution to the MRS composite.
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT: VIX Lifecycle State + Zero Gamma ───────────────────────────────────
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
