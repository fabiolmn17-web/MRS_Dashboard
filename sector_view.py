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
GREEN, LGREEN, RED, LRED, ORANGE, GREY = '#22c55e', '#86efac', '#ef4444', '#f87171', '#f97316', '#6b7280'

TREND_CFG = {
    'IMPROVING': ('↑ IMPROVING', '#22c55e', 'rgba(34,197,94,0.12)'),
    'FADING':    ('↓ FADING',    '#f97316', 'rgba(249,115,22,0.12)'),
    'STABLE+':   ('→ STABLE',    '#86efac', 'rgba(0,0,0,0)'),
    'STABLE-':   ('→ STABLE',    '#f87171', 'rgba(0,0,0,0)'),
    'NEUTRAL':   ('— NEUTRAL',   '#6b7280', 'rgba(0,0,0,0)'),
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


CSS = """
<style>
  body { margin:0; font-family:"Source Sans Pro", -apple-system, "Segoe UI", sans-serif; background:transparent; }
  .ctl { font-size:0.72rem; color:#9ca3af; margin:2px 0 6px 0; }
  .ctl a { color:#60a5fa; cursor:pointer; text-decoration:none; margin-right:12px; }
  .ctl a:hover { text-decoration:underline; }
  table.sec { width:100%; border-collapse:collapse; }
  table.sec th { background:#21262d; color:#9ca3af; font-size:0.72rem; font-weight:600; padding:7px 10px;
                 text-align:right; border-bottom:1px solid #374151; white-space:nowrap; }
  table.sec th.l { text-align:left; } table.sec th.c { text-align:center; }
  table.sec td { color:#e5e7eb; font-size:0.80rem; padding:6px 10px; text-align:right;
                 border-bottom:1px solid #1f2937; white-space:nowrap; }
  table.sec td.l { text-align:left; } table.sec td.c { text-align:center; }
  tr.srow td { background:#161b22; }
  tr.srow.click, tr.irow.click { cursor:pointer; }
  tr.srow.click:hover td { background:#1c2330; }
  tr.irow td { background:#10151c; font-size:0.77rem; }
  tr.irow.click:hover td { background:#18202b; }
  tr.irow td.l.name { padding-left:30px; }
  .car { display:inline-block; width:12px; color:#6b7280; font-size:0.70rem; }
  .sub { font-size:0.70rem; }
  .cnt { color:#6b7280; font-size:0.68rem; font-weight:400; margin-left:6px; }
  tr.hrow > td { background:#0b0f14; padding:4px 10px 10px 30px; }
  table.hold { width:100%; border-collapse:collapse; margin-top:2px; }
  table.hold th { background:#1a1f27; color:#9ca3af; font-size:0.68rem; font-weight:600; padding:5px 8px;
                  text-align:right; border-bottom:1px solid #2b3340; white-space:nowrap; }
  table.hold th.l { text-align:left; } table.hold th.c { text-align:center; }
  table.hold td { background:#0f141a; font-size:0.75rem; padding:5px 8px; border-bottom:1px solid #1a2029; }
  .note { font-size:0.70rem; color:#6b7280; margin-top:6px; line-height:1.5; }
  .hide { display:none; }
</style>
"""

JS = """
<script>
function fit(){ try{ var f=window.frameElement; if(f){ f.style.height=(document.body.scrollHeight+8)+'px'; } }catch(e){} }
function tog(cls, el){
  var rows=document.querySelectorAll('tr.'+cls); var open=el.getAttribute('data-open')==='1';
  rows.forEach(function(r){ if(open){ r.classList.add('hide'); } else { r.classList.remove('hide'); } });
  if(open){ // closing a sector also closes its industries' holdings
    rows.forEach(function(r){ var k=r.getAttribute('data-hcls');
      if(k){ document.querySelectorAll('tr.'+k).forEach(function(h){h.classList.add('hide');});
             r.setAttribute('data-open','0'); var c=r.querySelector('.car'); if(c){c.textContent='▸';} } });
  }
  el.setAttribute('data-open', open?'0':'1');
  var c=el.querySelector('.car'); if(c){ c.textContent= open?'▸':'▾'; }
  fit();
}
function allSec(open){
  document.querySelectorAll('tr.srow.click').forEach(function(r){
    var isOpen=r.getAttribute('data-open')==='1'; if(isOpen!==open){ r.click(); } });
  fit();
}
window.addEventListener('load', fit);
</script>
"""


def _metric_cells(r, score_txt=True):
    """The ten numeric/trend/score cells shared by sector and industry rows."""
    out = ''
    if _nan(r.get('Close')):
        out += '<td>—</td>'
    else:
        dc = r.get('DailyChg')
        dcc = GREEN if (not _nan(dc) and float(dc) >= 0) else LRED
        dct = '' if _nan(dc) else f'<br><span class="sub" style="color:{dcc};">{"+" if float(dc) >= 0 else ""}{float(dc) * 100:.2f}%</span>'
        out += f'<td>${float(r["Close"]):,.2f}{dct}</td>'
    for k in ['YTD', '1Y', '6M', '3M', 'RS 1Y', 'RS 6M', 'RS 3M']:
        out += f'<td style="color:{_col(r.get(k))};">{_pct(r.get(k))}</td>'
    t_lbl, t_col, t_bg = TREND_CFG.get(r.get('Trend', 'NEUTRAL'), TREND_CFG['NEUTRAL'])
    bg = '' if t_bg == 'rgba(0,0,0,0)' else f'background:{t_bg};'
    out += f'<td class="c" style="{bg}color:{t_col};font-weight:700;font-size:0.75rem;">{t_lbl}</td>'
    sc = r.get('Score')
    if _nan(sc):
        out += '<td style="color:#6b7280;">no data</td>'
    else:
        sc = float(sc)
        lbl = r.get('Label', '')
        out += f'<td style="color:{_score_col(sc)};font-weight:700;">{"+" if sc > 0 else ""}{sc:.1f}&nbsp; {_e(lbl)}</td>'
    return out


def _holdings_table(h: pd.DataFrame) -> str:
    if h is None or h.empty:
        return '<div class="note" style="margin:4px 0;">Holdings not available for this ETF.</div>'
    s = '<table class="hold"><thead><tr>'
    for lbl, cls in [('#', 'c'), ('Ticker', 'l'), ('Name', 'l'), ('Weight', ''), ('Close', ''), ('% 52W Hi', ''),
                     ('RS Score', ''), ('RS 1Y', ''), ('RS 6M', ''), ('RS 3M', ''), ('ATR', 'c'), ('Squeeze', 'c')]:
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
            ath_col = GREEN if ath > -0.05 else LRED if ath < -0.15 else '#fbbf24'
        rc = r.get('rs_composite')
        rc_txt = '—' if _nan(rc) else f'{float(rc):+.1f}'
        atr = str(r.get('atr_compressed')).lower() == 'true'
        sq = str(r.get('bb_kc_squeeze')).lower() == 'true'
        s += '<tr>'
        s += f'<td class="c" style="color:#6b7280;">{int(r["rank"])}</td>'
        s += f'<td class="l"><b style="color:#e5e7eb;">{_e(r["ticker"])}</b></td>'
        s += f'<td class="l" style="color:#9ca3af;">{_e(str(r.get("name", ""))[:30])}</td>'
        s += f'<td style="color:#9ca3af;">{w_txt}</td>'
        s += f'<td>{close_txt}</td>'
        s += f'<td style="color:{ath_col};font-weight:600;">{ath_txt}</td>'
        s += f'<td style="color:{_score_col(rc)};font-weight:700;">{rc_txt}</td>'
        for k in ['rs_1y', 'rs_6m', 'rs_3m']:
            s += f'<td style="color:{_col(r.get(k))};">{_pct(r.get(k))}</td>'
        s += f'<td class="c" style="color:{"#60a5fa" if atr else "#4b5563"};">{"🔵" if atr else "—"}</td>'
        s += f'<td class="c" style="color:{"#fbbf24" if sq else "#4b5563"};">{"⚡" if sq else "—"}</td>'
        s += '</tr>'
    s += '</tbody></table>'
    return s


def _sort(df: pd.DataFrame, col: str, asc: bool) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return df
    return df.sort_values(col, ascending=asc, na_position='last')


def build_drilldown_html(sec_df: pd.DataFrame, ind_df, hold_df, sort_col: str, sort_asc: bool,
                         ind_as_of: str = None):
    """Return (html, collapsed_height_px)."""
    have_ind = ind_df is not None and len(ind_df) > 0
    hold_by_etf = {}
    if hold_df is not None and len(hold_df) > 0 and 'ETF' in hold_df.columns:
        hold_by_etf = {etf: g for etf, g in hold_df.groupby('ETF')}

    h = CSS + '<div class="ctl">'
    if have_ind:
        h += '<a onclick="allSec(true)">Expand all</a><a onclick="allSec(false)">Collapse all</a>'
        h += 'Click a sector to see its industries, then an industry to see its top 10 holdings.'
    else:
        h += 'Industry drill-down not available yet: it is produced by the nightly scanner job.'
    h += '</div>'
    h += '<table class="sec"><thead><tr>'
    for lbl, cls in [('Sector / Industry', 'l'), ('ETF', ''), ('Close', ''), ('YTD', ''), ('1Y', ''), ('6M', ''),
                     ('3M', ''), ('RS 1Y', ''), ('RS 6M', ''), ('RS 3M', ''), ('6M&#8594;3M Trend', 'c'), ('Score', '')]:
        h += f'<th class="{cls}">{lbl}</th>'
    h += '</tr></thead><tbody>'

    n_rows = 0
    for si, (_, srow) in enumerate(sec_df.iterrows()):
        sid = f's{si}'
        inds = ind_df[ind_df['Sector'] == srow['Sector']] if have_ind else pd.DataFrame()
        inds = _sort(inds, sort_col, sort_asc)
        click = len(inds) > 0
        cnt = f'<span class="cnt">{len(inds)} industr{"y" if len(inds) == 1 else "ies"}</span>' if click else ''
        h += (f'<tr class="srow{" click" if click else ""}" data-open="0"'
              + (f' onclick="tog(\'ind-{sid}\', this)"' if click else '') + '>')
        h += (f'<td class="l"><span class="car">{"▸" if click else ""}</span>'
              f'<b style="color:#e5e7eb;">{_e(srow["Sector"])}</b>{cnt}</td>')
        h += f'<td style="color:#9ca3af;">{_e(srow["ETF"])}</td>'
        h += _metric_cells(srow)
        h += '</tr>'
        n_rows += 1
        for ii, (_, irow) in enumerate(inds.iterrows()):
            iid = f'{sid}i{ii}'
            hold = hold_by_etf.get(irow['ETF'])
            h += (f'<tr class="irow click ind-{sid} hide" data-open="0" data-hcls="hold-{iid}" '
                  f'onclick="event.stopPropagation(); tog(\'hold-{iid}\', this)">')
            h += (f'<td class="l name"><span class="car">▸</span>'
                  f'<span style="color:#d1d5db;">{_e(irow["Industry"])}</span></td>')
            h += f'<td style="color:#9ca3af;">{_e(irow["ETF"])}</td>'
            h += _metric_cells(irow)
            h += '</tr>'
            h += f'<tr class="hrow hold-{iid} hide"><td colspan="12">{_holdings_table(hold)}</td></tr>'

    h += '</tbody></table>'
    note = ('RS = additive return differential vs SPY. Trend = RS sign shift 6M to 3M. '
            'Score = sign-based composite (range -1.5 to +1.5). Sector rows: live, cached hourly. ')
    if have_ind:
        note += (f'Industry rows (industry ETFs, same formula) and top 10 holdings: nightly'
                 f'{", as of " + _e(ind_as_of) if ind_as_of else ""}. '
                 'Holdings: weight in the ETF, % 52W Hi = distance from the 52-week closing high, RS Score = same composite, '
                 'ATR 🔵 = bottom 35th pctile, Squeeze ⚡ = Bollinger inside Keltner. '
                 'Some industry ETFs hold non-US listings.')
    h += f'<div class="note">{note}</div>'
    h += JS
    collapsed_h = 34 + 36 + n_rows * 44 + 70
    return h, collapsed_h
