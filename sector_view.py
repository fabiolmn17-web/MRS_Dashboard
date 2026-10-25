"""
sector_view.py — Sector -> industry -> top-10-holdings drill-down table
========================================================================
Builds the HTML for the dashboard's sector table. Click a sector to open its
industries (industry ETFs, same RS metrics and score as the sector row); click
an industry to open its top 10 holdings (scanner-style metrics).

Sector rows are computed live in app.py (build_sector_table). Industry and
holdings rows come from the nightly job (StockScanner/run_industry.py).
"""
import html as _html
import math

import pandas as pd

# ── palette (same as the rest of the dashboard) ───────────────────────────────
GREEN, LGREEN, RED, LRED, ORANGE, GREY = '#85cac5', '#649995', '#f4a062', '#b9805a', '#b9805a', '#9b958f'

TREND_CFG = {
    'IMPROVING': ('↑ IMPROVING', '#85cac5', 'rgba(133,202,197,0.18)'),
    'FADING':    ('↓ FADING',    '#f4a062', 'rgba(244,160,98,0.18)'),
    'STABLE+':   ('→ STABLE',    '#649995', 'rgba(0,0,0,0)'),
    'STABLE-':   ('→ STABLE',    '#b9805a', 'rgba(0,0,0,0)'),
    'NEUTRAL':   ('— NEUTRAL',   '#9b958f', 'rgba(0,0,0,0)'),
}


def _nan(v):
    try:
        return v is None or math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _pct(v, dec=1):
    if _nan(v):
        return '—'
    v = float(v)
    return f'{"+" if v > 0 else ""}{v * 100:.{dec}f}%'


def _col(v):
    if _nan(v):
        return GREY
    v = float(v)
    return GREEN if v > 0 else RED if v < 0 else GREY


def _score_col(v):
    if _nan(v):
        return GREY
    v = float(v)
    if v >= 1.0:  return GREEN
    if v >= 0.5:  return LGREEN
    if v <= -1.0: return RED
    if v <= -0.5: return ORANGE
    return GREY


def _e(s):
    return _html.escape(str(s), quote=True)


GRID = 'grid-template-columns:minmax(190px,2.2fr) 0.55fr 0.95fr repeat(9,minmax(56px,0.72fr)) 1.05fr minmax(190px,1.9fr);'

CSS = (
    '<style>'
    '.sv{overflow-x:auto;margin-top:6px;font-size:0.80rem;}'
    '.sv .in{min-width:1290px;}'
    '.sv .g{display:grid;' + GRID + 'align-items:center;}'
    '.sv .g>div{padding:6px 10px;text-align:right;white-space:nowrap;color:#f3ece2;}'
    '.sv .g>div.l{text-align:left;} .sv .g>div.c{text-align:center;}'
    '.sv .hd{background:#2a2724;border-bottom:1px solid #2a2724;}'
    '.sv .hd>div{color:#a89f95 !important;font-size:0.72rem;font-weight:600;padding:7px 10px;}'
    '.sv .sr{background:#121212;border-bottom:1px solid #1a1918;}'
    '.sv details>summary{list-style:none;cursor:pointer;}'
    '.sv details>summary::-webkit-details-marker{display:none;}'
    '.sv details>summary::marker{content:"";}'
    '.sv details.sec>summary:hover{background:#1a1918;}'
    '.sv .ir{background:#0a0a0a;border-bottom:1px solid #1a1918;font-size:0.77rem;}'
    '.sv details.ind>summary:hover{background:#121212;}'
    '.sv .ir>div.l{padding-left:30px;}'
    '.sv .car{display:inline-block;width:16px;color:#a89f95;font-size:0.85rem;}'
    '.sv .car::before{content:"▸";}'
    '.sv details[open]>summary .car::before{content:"▾";}'
    '.sv .sub{font-size:0.70rem;}'
    '.sv .cnt{color:#9b958f;font-size:0.68rem;font-weight:400;margin-left:6px;}'
    '.sv .hp{background:#0a0a0a;padding:4px 10px 10px 30px;border-bottom:1px solid #1a1918;}'
    '.sv table.hold{width:100%;border-collapse:collapse;margin:2px 0 0 0;}'
    '.sv table.hold th{background:#121212;color:#a89f95;font-size:0.68rem;font-weight:600;padding:5px 8px;'
    'text-align:right;border-bottom:1px solid #2a2724;white-space:nowrap;}'
    '.sv table.hold th.l{text-align:left;} .sv table.hold th.c{text-align:center;}'
    '.sv table.hold td{background:#0a0a0a;font-size:0.75rem;padding:5px 8px;border-bottom:1px solid #121212;'
    'text-align:right;white-space:nowrap;color:#f3ece2;}'
    '.sv table.hold td.l{text-align:left;} .sv table.hold td.c{text-align:center;}'
    '.sv table.hold,.sv table.hold th,.sv table.hold td{border-left:none;border-right:none;border-top:none;}'
    '.sv .ctl{font-size:0.72rem;color:#a89f95;margin:2px 0 6px 0;}'
    '.sv .note{font-size:0.70rem;color:#9b958f;margin-top:6px;line-height:1.5;white-space:normal;}'
    '.sv .meta{font-weight:400;font-size:0.68rem;}'
    '.sv .dl{color:#a89f95;font-family:"IBM Plex Mono",monospace;}'
    '.sv .pend{color:#e3b48e;margin-left:4px;}'
    '.chip{display:inline-block;border:1px solid;border-radius:3px;padding:0 4px;margin-left:4px;font-size:0.66rem;'
    'font-weight:700;line-height:1.35;}'
    '</style>'
)


def _metric_cells(r, score_txt=True):
    """The ten numeric/trend/score cells shared by sector and industry rows."""
    out = ''
    if _nan(r.get('Close')):
        out += '<div>—</div>'
    else:
        dc = r.get('DailyChg')
        dcc = GREEN if (not _nan(dc) and float(dc) >= 0) else LRED
        dct = '' if _nan(dc) else f'<br><span class="sub" style="color:{dcc};">{"+" if float(dc) >= 0 else ""}{float(dc) * 100:.2f}%</span>'
        out += f'<div>${float(r["Close"]):,.2f}{dct}</div>'
    for k in ['YTD', '1Y', '6M', '3M', '1M', 'RS 1Y', 'RS 6M', 'RS 3M', 'RS 1M']:
        out += f'<div style="color:{_col(r.get(k))};">{_pct(r.get(k))}</div>'
    t_lbl, t_col, t_bg = TREND_CFG.get(r.get('Trend', 'NEUTRAL'), TREND_CFG['NEUTRAL'])
    bg = '' if t_bg == 'rgba(0,0,0,0)' else f'background:{t_bg};'
    out += f'<div class="c" style="{bg}color:{t_col};font-weight:700;font-size:0.75rem;align-self:stretch;display:flex;align-items:center;justify-content:center;">{t_lbl}</div>'
    sc = r.get('Score')
    if _nan(sc):
        out += '<div style="color:#9b958f;">no data</div>'
    else:
        sc = float(sc)
        lbl = r.get('Label', '')
        out += (f'<div style="color:{_score_col(sc)};font-weight:700;">{"+" if sc > 0 else ""}{sc:.1f}&nbsp; {_e(lbl)}'
                f'{_label_meta(r)}</div>')
    return out


def _txt(v):
    return '' if _nan(v) and not isinstance(v, str) else str(v)


def chip_1m(tag: str) -> str:
    """RS 1M early-warning chip (shown, not scored)."""
    if tag == '1M ↓':
        return ('<span class="chip" style="color:#f4a062;background:rgba(244,160,98,0.18);border-color:#f4a062;"'
                ' title="Structural leader whose RS 1M is negative: the last month is lagging SPY">1M ↓</span>')
    if tag == '1M ↑':
        return ('<span class="chip" style="color:#85cac5;background:rgba(133,202,197,0.18);border-color:#85cac5;"'
                ' title="Outperforming name whose RS 1M is positive: the last month is leading SPY">1M ↑</span>')
    return ''


def _label_meta(r) -> str:
    """Second line of the Score cell: days in label, pending change, RS 1M chip."""
    parts = []
    d = r.get('DaysInLabel')
    if not _nan(d):
        capped = str(r.get('DaysCapped')).lower() == 'true'
        parts.append(f'<span class="dl" title="Sessions since the held label last changed">'
                     f'{"≥" if capped else ""}{int(float(d))}d</span>')
    pl = _txt(r.get('PendingLabel'))
    if pl:
        pdays = r.get('PendingDays')
        k = 0 if _nan(pdays) else int(float(pdays))
        parts.append(f'<span class="pend" title="Raw label differs; it replaces the held label after 3 sessions in a row">'
                     f'→ {_e(pl.title())} {k}/3</span>')
    c = chip_1m(_txt(r.get('Tag1M')))
    if c:
        parts.append(c)
    return ('<br><span class="meta">' + ' '.join(parts) + '</span>') if parts else ''


def _holdings_table(h: pd.DataFrame) -> str:
    if h is None or h.empty:
        return '<div class="note" style="margin:4px 0;">Holdings not available for this ETF.</div>'
    s = '<table class="hold"><thead><tr>'
    for lbl, cls in [('#', 'c'), ('Ticker', 'l'), ('Name', 'l'), ('Weight', ''), ('Close', ''), ('% 52W Hi', ''),
                     ('RS Score', ''), ('RS 1Y', ''), ('RS 6M', ''), ('RS 3M', ''), ('RS 1M', ''), ('ATR', 'c'), ('Squeeze', 'c')]:
        s += f'<th class="{cls}">{lbl}</th>'
    s += '</tr></thead><tbody>'
    for _, r in h.sort_values('rank').iterrows():
        w = r.get('weight')
        w_txt = '—' if _nan(w) else f'{float(w) * 100:.1f}%'
        close = r.get('close')
        if _nan(close):
            close_txt = '—'
        else:
            dc = r.get('daily_chg')
            dcc = GREEN if (not _nan(dc) and float(dc) >= 0) else LRED
            dct = '' if _nan(dc) else f' <span class="sub" style="color:{dcc};">{"+" if float(dc) >= 0 else ""}{float(dc) * 100:.2f}%</span>'
            close_txt = f'${float(close):,.2f}{dct}'
        ath = r.get('pct_from_52w_high')
        if _nan(ath):
            ath_txt, ath_col = '—', GREY
        else:
            ath = float(ath)
            ath_txt = f'{ath * 100:+.1f}%'
            ath_col = GREEN if ath > -0.05 else LRED if ath < -0.15 else '#e3b48e'
        rc = r.get('rs_composite')
        rc_txt = '—' if _nan(rc) else f'{float(rc):+.1f}'
        atr = str(r.get('atr_compressed')).lower() == 'true'
        sq = str(r.get('bb_kc_squeeze')).lower() == 'true'
        s += '<tr>'
        s += f'<td class="c" style="color:#9b958f;">{int(r["rank"])}</td>'
        s += f'<td class="l"><b style="color:#f3ece2;">{_e(r["ticker"])}</b></td>'
        s += f'<td class="l" style="color:#a89f95;">{_e(str(r.get("name", ""))[:30])}</td>'
        s += f'<td style="color:#a89f95;">{w_txt}</td>'
        s += f'<td>{close_txt}</td>'
        s += f'<td style="color:{ath_col};font-weight:600;">{ath_txt}</td>'
        s += f'<td style="color:{_score_col(rc)};font-weight:700;">{rc_txt}</td>'
        for k in ['rs_1y', 'rs_6m', 'rs_3m', 'rs_1m']:
            s += f'<td style="color:{_col(r.get(k))};">{_pct(r.get(k))}</td>'
        s += f'<td class="c" style="color:{"#e8dccb" if atr else "#6e675f"};">{"●" if atr else "—"}</td>'
        s += f'<td class="c" style="color:{"#e3b48e" if sq else "#6e675f"};">{"⚡" if sq else "—"}</td>'
        s += '</tr>'
    s += '</tbody></table>'
    return s


def _sort(df: pd.DataFrame, col: str, asc: bool) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    return df.sort_values(col, ascending=asc, na_position='last')


def build_drilldown_html(sec_df: pd.DataFrame, ind_df, hold_df, sort_col: str, sort_asc: bool,
                         ind_as_of: str = None) -> str:
    """HTML for st.markdown(unsafe_allow_html=True): native <details> rows, no iframe, no JS.

    Single line (no blank lines / indentation) so the markdown parser passes it through.
    """
    have_ind = ind_df is not None and len(ind_df) > 0
    hold_by_etf = {}
    if hold_df is not None and len(hold_df) > 0 and 'ETF' in hold_df.columns:
        hold_by_etf = {etf: g for etf, g in hold_df.groupby('ETF')}

    h = CSS + '<div class="sv"><div class="ctl">'
    h += ('Click a sector to see its industries, then an industry to see its top 10 holdings.' if have_ind
          else 'Industry drill-down not available yet: it is produced by the nightly scanner job.')
    h += '</div><div class="in"><div class="g hd">'
    for lbl, cls in [('Sector / Industry', 'l'), ('ETF', ''), ('Close', ''), ('YTD', ''), ('1Y', ''), ('6M', ''),
                     ('3M', ''), ('1M', ''), ('RS 1Y', ''), ('RS 6M', ''), ('RS 3M', ''), ('RS 1M', ''), ('6M&#8594;3M Trend', 'c'), ('Score', '')]:
        h += f'<div class="{cls}">{lbl}</div>'
    h += '</div>'

    for _, srow in sec_df.iterrows():
        inds = ind_df[ind_df['Sector'] == srow['Sector']] if have_ind else pd.DataFrame()
        inds = _sort(inds, sort_col, sort_asc)
        click = len(inds) > 0
        cnt = (f'<span class="cnt">{len(inds)} industr{"y" if len(inds) == 1 else "ies"}</span>' if click else '')
        cells = (f'<div class="l"><span class="car"></span><b style="color:#f3ece2;">{_e(srow["Sector"])}</b>{cnt}</div>'
                 if click else f'<div class="l"><span style="display:inline-block;width:14px;"></span>'
                               f'<b style="color:#f3ece2;">{_e(srow["Sector"])}</b></div>')
        cells += f'<div style="color:#a89f95;">{_e(srow["ETF"])}</div>' + _metric_cells(srow)
        if not click:
            h += f'<div class="g sr">{cells}</div>'
            continue
        h += f'<details class="sec"><summary class="g sr">{cells}</summary>'
        for _, irow in inds.iterrows():
            icells = (f'<div class="l"><span class="car"></span><span style="color:#a89f95;">{_e(irow["Industry"])}</span></div>'
                      f'<div style="color:#a89f95;">{_e(irow["ETF"])}</div>' + _metric_cells(irow))
            h += (f'<details class="ind"><summary class="g ir">{icells}</summary>'
                  f'<div class="hp">{_holdings_table(hold_by_etf.get(irow["ETF"]))}</div></details>')
        h += '</details>'

    h += '</div>'
    note = ('RS = additive return differential vs SPY. Trend = RS sign shift 6M to 3M. '
            'Score = sign-based composite of RS 1Y / 6M / 3M (range -1.5 to +1.5); RS 1M (21 sessions, same formula) is shown but not scored. '
            'Label hold: the label (and trend) change only after the new reading has held 3 sessions in a row; '
            'Nd = sessions in the current label, "→ X k/3" = a change in progress. '
            '1M ↓ = structural leader with negative RS 1M; 1M ↑ = outperforming name with positive RS 1M (early warning, not scored). '
            'Sector rows: live, cached hourly. ')
    if have_ind:
        note += (f'Industry rows (industry ETFs, same formula) and top 10 holdings: nightly'
                 f'{", as of " + _e(ind_as_of) if ind_as_of else ""}. '
                 'Holdings: weight in the ETF, % 52W Hi = distance from the 52-week closing high, '
                 'RS Score = same composite, ATR <span style="color:#e8dccb">●</span> = bottom 35th pctile, Squeeze &#9889; = Bollinger inside Keltner. '
                 'Some industry ETFs hold non-US listings.')
    h += f'<div class="note">{note}</div></div>'
    return h.replace('\n', ' ')
