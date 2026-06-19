# ── 2. SPX Close + Zero Gamma line ───────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:0; margin-bottom:4px;">SPX Close</div>',
            unsafe_allow_html=True)

# Full 90-row base (NaN rows become natural gaps in the line)
hist90_spx      = hist.tail(90).copy()
hist90_spx_valid = hist90_spx.dropna(subset=['spx'])   # for range calc only

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
        y=hist90_spx['spx'],          # NaN = gap, line ends at last close
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

    fig_spx.add_vline(x=last_dt, **VLINE_STYLE)

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

# Full 90-row base (NaN rows become natural gaps in the line)
hist90_vix       = hist.tail(90).copy()
hist90_vix_valid = hist90_vix.dropna(subset=['vix'])   # for range calc only

if len(hist90_vix_valid) > 0:
    vix_min = hist90_vix_valid['vix'].min()
    vix_max = hist90_vix_valid['vix'].max()
    vix_pad = (vix_max - vix_min) * 0.12

    fig_vix = go.Figure()

    # Background shading by VIX phi state
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
        y=hist90_vix['vix'],           # NaN = gap, line ends at last close
        mode='lines', name='VIX',
        line=dict(color='#f9a8d4', width=2),
        hovertemplate='<b>%{x|%b %d}</b><br>VIX: %{y:.2f}<extra></extra>',
    ))

    fig_vix.add_hline(y=20, line_dash='dot',
                      line_color='rgba(255,255,255,0.20)', line_width=1,
                      annotation_text='20', annotation_position='right',
                      annotation_font_color='#6b7280', annotation_font_size=10)

    fig_vix.add_vline(x=last_dt, **VLINE_STYLE)

    fig_vix.update_layout(**LAYOUT_BASE,
        margin=dict(l=10, r=130, t=4, b=30), height=160,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                   tickformat='.0f', tickfont_size=11,
                   range=[max(0, vix_min - vix_pad), vix_max + vix_pad]),
    )
    st.plotly_chart(fig_vix, use_container_width=True)
