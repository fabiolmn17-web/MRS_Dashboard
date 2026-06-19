"""
app.py — MRS Live Dashboard (Streamlit)
========================================
Password-protected. Reads mrs_history.csv and displays:
  • Current regime score + signal quality
  • 8-component breakdown table
  • VIX lifecycle state layer (paper-grounded: compression / mid / spike-zone)
  • Zero Gamma position
  • 90-day MRS history chart
  • SPX close price panel
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
    """Simple password gate using Streamlit secrets."""
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

@st.cache_data(ttl=300)   # refresh cache every 5 minutes
def load_data() -> pd.DataFrame:
    df = pipeline.load_history(HIST_PATH)
    return df

hist = load_data()
complete = hist.dropna(subset=['vix', 'mrs_score'])
last     = complete.iloc[-1].to_dict() if len(complete) else hist.iloc[-1].to_dict()
last_dt  = pd.Timestamp(last['date'])

# Fill price-derived fields from most recent row with valid SPX close
if pd.isna(last.get('spx', np.nan)):
    price_hist = hist.dropna(subset=['spx'])
    if len(price_hist) > 0:
        price_last = price_hist.iloc[-1]
        for col in ['spx', 'spy', 'ext_phi', 'ext_score', 'ext_state',
                    'mom_phi', 'mom_score', 'mom_state']:
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

def _phi_bar(phi):
    """Compact visual bar for Phi values."""
    try:
        p = float(phi)
        if np.isnan(p): return '— '
        filled = int(round(p * 10))
        bar    = '█' * filled + '░' * (10 - filled)
        return f'{bar} {p:.3f}'
    except:
        return '—'

QUALITY_COLORS = {
    'CONFIRMED':            '#22c55e',
    'NEUTRAL — BULLISH LEAN': '#22c55e',
    'NEUTRAL — BEARISH LEAN': '#f97316',
    'NEUTRAL — NO EDGE':    '#9ca3af',
    'UNCONFIRMED':          '#f97316',
    'DIVERGENT':            '#ef4444',
    'FRAGILE':              '#facc15',
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
      <div style="font-size:0.82rem;line-height:2;">
        <b>VIX</b> {vix_now}<br>
        <b>SPX</b> {spx_now}<br>
        <b>Zero γ</b> {zg_now}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT: Components (left) | VIX Hazard + Chart (right)
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

        # Score color
        try:
            sv = float(sc_v)
            sc_color = '#22c55e' if sv > 0 else ('#ef4444' if sv < 0 else '#6b7280')
        except:
            sc_color = '#6b7280'

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
            return 'color: #6b7280'
        except:
            return ''

    styled = df_comp.style.map(color_score, subset=['Score'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Phi explanation note
    st.markdown("""
    <div style="font-size:0.72rem;color:#6b7280;margin-top:4px;">
    Phi = percentile rank over rolling 756-session window (3 years).
    Score = discretized contribution to the MRS composite.
    </div>
    """, unsafe_allow_html=True)


# ── RIGHT: VIX Hazard + Gamma ──────────────────────────────────────────────────
with right:

    # VIX Lifecycle State
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

    # Zero Gamma position
    st.markdown('<div class="section-header" style="margin-top:16px;">Zero Gamma Position</div>', unsafe_allow_html=True)
    try:
        spx_v = float(last.get('spx', np.nan))
        zg_v  = float(last.get('zero_gamma', np.nan))
        if not np.isnan(spx_v) and not np.isnan(zg_v) and zg_v > 0:
            dist_pct = (spx_v - zg_v) / spx_v * 100
            if dist_pct > 1:
                gcls = 'safe-row'
                gtxt = f'🟢 SPX {spx_v:,.0f} is {dist_pct:.1f}% ABOVE zero-gamma ({zg_v:,.0f}). Dealers short gamma — dampening environment.'
            elif dist_pct > -1:
                gcls = 'neutral-row'
                gtxt = f'🟡 SPX {spx_v:,.0f} is NEAR zero-gamma ({zg_v:,.0f}, {dist_pct:+.1f}%). Transition zone — regime could flip.'
            else:
                gcls = 'hazard-row'
                gtxt = f'🔴 SPX {spx_v:,.0f} is {abs(dist_pct):.1f}% BELOW zero-gamma ({zg_v:,.0f}). Dealers long gamma — amplifying moves.'
            st.markdown(f'<div class="{gcls}">{gtxt}</div>', unsafe_allow_html=True)
        else:
            try:
                zg_only = float(last.get('zero_gamma', np.nan))
                if not np.isnan(zg_only) and zg_only > 0:
                    st.markdown(f'<div class="neutral-row">⚪ Zero Gamma level: <b>{zg_only:,.0f}</b> — SPX close pending (run after market close)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="neutral-row">⚪ Zero Gamma: no data today</div>', unsafe_allow_html=True)
            except:
                st.markdown('<div class="neutral-row">⚪ Zero Gamma: no data today</div>', unsafe_allow_html=True)
    except:
        try:
            zg_only = float(last.get('zero_gamma', np.nan))
            if not np.isnan(zg_only) and zg_only > 0:
                st.markdown(f'<div class="neutral-row">⚪ Zero Gamma level: <b>{zg_only:,.0f}</b> — SPX close pending (run after market close)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="neutral-row">⚪ Zero Gamma: no data today</div>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="neutral-row">⚪ Zero Gamma: no data today</div>', unsafe_allow_html=True)


st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# 90-DAY MRS HISTORY CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">90-Day MRS History</div>', unsafe_allow_html=True)

hist90 = hist.dropna(subset=['mrs_score']).tail(90).copy()

fig = go.Figure()

# Regime band fills
band_defs = [
    (1.5,  5.0,  'rgba(26,127,55,0.12)',  'RISK-ON'),
    (0.5,  1.5,  'rgba(87,166,107,0.10)', 'MILD RISK-ON'),
    (-0.5, 0.5,  'rgba(107,114,128,0.08)','NEUTRAL'),
    (-1.5, -0.5, 'rgba(217,119,6,0.10)',  'MILD RISK-OFF'),
    (-5.0, -1.5, 'rgba(185,28,28,0.12)',  'RISK-OFF'),
]
for y0, y1, fill, label in band_defs:
    fig.add_hrect(y0=y0, y1=y1, fillcolor=fill, line_width=0,
                  annotation_text=label,
                  annotation_position='right',
                  annotation_font_size=10,
                  annotation_font_color='#6b7280')

# MRS line
fig.add_trace(go.Scatter(
    x=hist90['date'],
    y=hist90['mrs_score'],
    mode='lines+markers',
    name='MRS',
    line=dict(color='#60a5fa', width=2.5),
    marker=dict(size=4, color='#60a5fa'),
    hovertemplate='<b>%{x|%b %d}</b><br>MRS: %{y:+.2f}<extra></extra>',
))

# Zero line
fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.25)', line_width=1)

# Today marker
fig.add_vline(
    x=last_dt,
    line_dash='dot',
    line_color='rgba(250,204,21,0.6)',
    line_width=1.5,
    annotation_text='Today',
    annotation_font_color='#facc15',
    annotation_font_size=10,
)

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=10, r=80, t=10, b=30),
    height=300,
    showlegend=False,
    xaxis=dict(showgrid=False, tickformat='%b %d', tickfont_size=11),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
               tickformat='+.1f', range=[-4.5, 4.5], tickfont_size=11),
)

st.plotly_chart(fig, use_container_width=True)


# ── SPX Close panel ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:0; margin-bottom:4px;">SPX Close</div>', unsafe_allow_html=True)

hist90_spx = hist.tail(90).dropna(subset=['spx']).copy()

if len(hist90_spx) > 0:
    spx_min = hist90_spx['spx'].min()
    spx_max = hist90_spx['spx'].max()
    spx_pad = (spx_max - spx_min) * 0.08

    fig_spx = go.Figure()

    fig_spx.add_trace(go.Scatter(
        x=hist90_spx['date'],
        y=hist90_spx['spx'],
        mode='lines',
        name='SPX',
        line=dict(color='#a78bfa', width=2),
        fill='tonexty',
        fillcolor='rgba(167,139,250,0.08)',
        hovertemplate='<b>%{x|%b %d}</b><br>SPX: %{y:,.0f}<extra></extra>',
    ))

    fig_spx.add_vline(
        x=last_dt,
        line_dash='dot',
        line_color='rgba(250,204,21,0.6)',
        line_width=1.5,
    )

    fig_spx.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=80, t=4, b=30),
        height=180,
        showlegend=False,
        xaxis=dict(showgrid=False, tickformat='%b %d', tickfont_size=11),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.06)',
            tickformat=',.0f',
            tickfont_size=11,
            range=[spx_min - spx_pad, spx_max + spx_pad],
        ),
    )

    st.plotly_chart(fig_spx, use_container_width=True)


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


# ── Footer ────────────────────────────────────────────────────────────────────
last_upd = hist['date'].max()
st.markdown(f"""
<div style="text-align:center;font-size:0.72rem;color:#4b5563;margin-top:24px;">
  Epistruct — Invariant Research &nbsp;|&nbsp;
  Data through {last_upd.strftime('%B %d, %Y')} &nbsp;|&nbsp;
  Updates daily at 4:30 PM ET
</div>
""", unsafe_allow_html=True)
