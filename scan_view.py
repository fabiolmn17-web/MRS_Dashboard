"""
scan_view.py — HTML table for the Momentum & Trend scanner (rendered with components.html, sortable)
======================================================================================================
Rows come from StockScanner/output/scan_results.csv (nightly). The table is descriptive: the momentum list
uses the dashboard's RS label and trend rules; the Ext-Mom zones are the 50-day SMA study's definitions,
whose odds were measured on SPY and QQQ, not on single stocks.
"""
import html as _html
import math
import re

import pandas as pd

from market_state import EXT_COL, MOM_COL, chip
import heat_score

TEAL, TEAL_M, ORANGE, ORANGE_M, COPPER, DIM = '#85cac5', '#649995', '#f4a062', '#b9805a', '#e3b48e', '#9b958f'


def _e(s):
    return _html.escape(str(s), quote=True)


def _nan(v):
    try:
        return v is None or math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _sv(v):
    return -999 if _nan(v) else float(v)


def _pct(v, dec=1):
    return '—' if _nan(v) else f'{"+" if float(v) > 0 else ""}{float(v) * 100:.{dec}f}%'


def _col(v):
    return DIM if _nan(v) else TEAL if float(v) > 0 else ORANGE if float(v) < 0 else DIM


def _score_col(v):
    if _nan(v):
        return DIM
    v = float(v)
    return TEAL if v >= 1.0 else TEAL_M if v >= 0.5 else ORANGE if v <= -1.0 else ORANGE_M if v <= -0.5 else DIM


def _s(v):
    return '' if v is None or (isinstance(v, float) and math.isnan(v)) or str(v) == 'nan' else str(v)


# Beardy Squeeze Pro: dot colours (orange = High, red = Mid, black = Low, green = none) and momentum colours
SQZ_COL = {'High': '#f4a062', 'Mid': '#e0605a', 'Low': '#b9b2a9'}
SQZ_RANK = {'High': 3, 'Mid': 2, 'Low': 1}
MOM_SQZ_COL = {'aqua': '#4dd0e1', 'blue': '#2962ff', 'red': '#ef5350', 'yellow': '#fdd835'}
MOM_SQZ_TIP = {'aqua': 'momentum > 0 and rising', 'blue': 'momentum > 0 and falling',
               'red': 'momentum < 0 and falling', 'yellow': 'momentum < 0 and rising'}


def squeeze_cell(r):
    """(html, sort key) for the Squeeze column. Falls back to the old True/False flag for older CSVs."""
    lvl = _s(r.get('sqz_level'))
    if not lvl:
        on = str(r.get('bb_kc_squeeze')).lower() == 'true'
        return (f'<span style="color:{COPPER};">⚡</span>' if on else '<span style="color:#6e675f;">—</span>'), int(on)
    mc = _s(r.get('sqz_mom_color'))
    dot = (f' <span title="{_e(MOM_SQZ_TIP.get(mc, ""))}" style="display:inline-block;width:7px;height:7px;'
           f'border-radius:1px;background:{MOM_SQZ_COL.get(mc, "#6e675f")};vertical-align:middle;"></span>') if mc else ''
    fired = str(r.get('sqz_fired')).lower() == 'true'
    if fired:
        up = mc in ('aqua', 'blue')
        return (chip('Fired ' + ('↑' if up else '↓'), TEAL if up else ORANGE, True) + dot), 0.5
    if lvl in SQZ_RANK:
        b = r.get('sqz_bars')
        bars = '' if _nan(b) else f' <span class="sub">{int(float(b))}d</span>'
        return chip(lvl, SQZ_COL[lvl], lvl != 'Low') + bars + dot, SQZ_RANK[lvl] + min(float(b or 0), 99) / 1000
    return '<span style="color:#6e675f;">—</span>' + dot, 0


HEAD = [('Ticker', 'l'), ('Heat', 'c'), ('Sector', 'l'), ('Close', ''), ('% 52W Hi', ''), ('RS label (held)', 'l'),
        ('RS 3M', ''), ('RS 1M', ''), ('RS rtg', ''), ('Ext vs SMA50', ''), ('Momentum', 'c'), ('5-day', ''), ('CAN SLIM', 'c'),
        ('EPS QoQ', ''), ('ATR', 'c'), ('Squeeze', 'c'), ('Base / pivot', 'l'), ('Breakout', 'c'), ('Earnings', ''),
        ('Industry (rank · state)', 'l')]

BASE_ABBR = {'Cup w/ handle': 'Cup·H', 'Flat base': 'Flat', 'High tight flag': 'HTF', 'VCP': 'VCP',
             'Tight area': 'Tight'}


def base_cell(r):
    """(html, sort key) for the Base / pivot column (bases.py fields; '—' for older CSVs)."""
    bt = _s(r.get('base_type'))
    if not bt or bt == 'None':
        return '<span style="color:#6e675f;">—</span>', -1
    st_ = _s(r.get('base_state'))
    pv, dist = r.get('base_pivot'), r.get('base_dist_pct')
    broke = st_.startswith('Broke out')
    col = TEAL if broke else COPPER if st_ == 'Near pivot' else DIM
    q = r.get('base_quality')
    stars = '' if _nan(q) else ' <span class="sub" title="Quality: volume dry-up, tight closes, RS rating ≥ 80">' + \
        '★' * int(float(q)) + '☆' * (3 - int(float(q))) + '</span>'
    if broke:
        bv = r.get('base_bo_vol')
        vtxt = '' if _nan(bv) else (f' <span style="color:{TEAL if float(bv) >= 1.4 else DIM};">'
                                    f'{float(bv):.1f}× vol</span>')
        txt = f'{_e(st_.replace("Broke out ", "out "))}{vtxt}'
    else:
        txt = '' if _nan(dist) else f'{float(dist) * 100:+.1f}%'
    pv_txt = '' if _nan(pv) else f' <span class="sub">to {float(pv):,.2f}</span>'
    w = r.get('base_weeks')
    wk = '' if _nan(w) else f' <span class="sub">{float(w):.0f}w</span>'
    vc = _s(r.get('vcp_contractions'))
    if bt == 'VCP' and vc and vc != 'None':           # e.g. "VCP 3T 18% · 9% · 4%"
        n_ = r.get('vcp_n')
        wk += (f' <span class="sub">{"" if _nan(n_) else f"{int(float(n_))}T "}{_e(vc)}</span>')
    rank = {'Broke out today': 4}.get(st_, 3 if broke else 2 if st_ == 'Near pivot' else 1)
    sk = rank * 1000 + (0 if _nan(dist) else float(dist) * 100)
    dep = r.get('base_depth')
    title = _e(f"{bt}: pivot {pv}" + ('' if _nan(dep) else f", depth {float(dep) * 100:.0f}%")
               + ('' if not _s(r.get('base_start')) else f", from {_s(r.get('base_start'))}"))
    return (f'<span title="{title}">{chip(BASE_ABBR.get(bt, bt), col, broke or st_ == "Near pivot")}{wk} '
            f'<span style="color:{col};">{txt}</span>{pv_txt}{stars}</span>'), sk


def ind_cell(r, etf=''):
    """(html, sort key): industry group from the scanned stocks (industry_groups.py, 4 Nov 2026): RS-rating rank inside
    the group, leader chip, and the group's held MRS state without the stock. Falls back to the ETF holding."""
    g = _s(r.get('ind_group'))
    if not g or g == 'None':
        return (chip(etf, TEAL, True) if etf else '<span style="color:#6e675f;">—</span>'), (1 if etf else 0)
    rk, n = r.get('ind_rank'), r.get('ind_n')
    role = _s(r.get('ind_role'))
    sc = r.get('ind_score')
    lab = _s(r.get('ind_label')).title()
    short = g if len(g) <= 26 else g[:25] + '…'
    rk_txt = '' if _nan(rk) or _nan(n) else f' <span class="sub">{int(float(rk))}/{int(float(n))}</span>'
    if role.startswith('Leader'):
        ccol = TEAL if 'leading' in role else ORANGE if 'lagging' in role else COPPER
        lead = chip('LEADER', ccol, 'leading' in role) + ' '
    else:
        lead = ''
    st_txt = '' if _nan(sc) else f' <span style="color:{_score_col(sc)};">{float(sc):+.1f}</span>'
    title = _e(f"{g} · rank {'' if _nan(rk) else int(float(rk))} of {'' if _nan(n) else int(float(n))} by RS rating · "
               f"industry without this stock: {lab or 'n/a'} ({_s(r.get('ind_trend')).title()})"
               + (f" · {role}" if role else ''))
    rank_k = {'Leader · leading industry': 3, 'Leader': 2, 'Leader · lagging industry': 1}.get(role, 0)
    sk = rank_k * 1000 + (0 if _nan(sc) else float(sc) * 100) - (0 if _nan(rk) else float(rk))
    return f'<span title="{title}">{lead}{_e(short)}{rk_txt}{st_txt}</span>', sk


def breakout_cell(r):
    """(html, sort key): today's close above the prior 21-session / 252-session high, with today's volume ratio."""
    b21 = str(r.get('bo21')).lower() == 'true'
    b52 = str(r.get('bo52')).lower() == 'true'
    bpv = str(r.get('piv_bo')).lower() == 'true'      # v2: close above the active 10/10 pivot high
    if not (b21 or b52 or bpv):
        return '<span style="color:#6e675f;">—</span>', 0
    vr = r.get('vol_ratio')
    vtxt = '' if _nan(vr) else (f' <span class="sub" style="color:{TEAL if float(vr) >= 1.4 else DIM};">'
                                f'{float(vr):.1f}×</span>')
    ph = r.get('piv_hi')
    ptitle = '' if _nan(ph) else f' title="Above the 10/10 pivot high {float(ph):,.2f} ({_e(_s(r.get("piv_date")))})"'
    chips = ((chip('52W', COPPER, True) + ' ' if b52 else '') + (chip('21D', TEAL, True) + ' ' if b21 else '')
             + (f'<span{ptitle}>{chip("PIV", TEAL, True)}</span>' if bpv else ''))
    return chips + vtxt, (2 if b52 else 1 if b21 else 0.5) + (0 if _nan(vr) else min(float(vr), 9) / 10)

CSS = """
<style>
body{margin:0;background:transparent;font-family:Barlow,system-ui,sans-serif;}
.wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;min-width:1410px;}
th{background:#2a2724;color:#a89f95;font-size:0.70rem;font-weight:600;padding:7px 6px;text-align:right;white-space:nowrap;cursor:pointer;user-select:none;position:sticky;top:0;}
th.l,td.l{text-align:left;} th.c,td.c{text-align:center;}
th:hover{color:#e3b48e;}
td{background:#121212;color:#f3ece2;font-size:0.76rem;padding:6px 6px;text-align:right;border-bottom:1px solid #1a1918;white-space:nowrap;}
td.num{font-family:'IBM Plex Mono',monospace;font-size:0.76rem;}
tr:hover td{background:#1a1918;}
tr[data-t]{cursor:pointer;} tr.sel td{background:#2a2420 !important;} tr.sel td:first-child{box-shadow:inset 3px 0 0 #e3b48e;}
.sub{color:#a89f95;font-size:0.68rem;}
.sarr{margin-left:3px;font-size:0.62rem;}
</style>
"""

JS = """
<script>
var _c=-1,_a=false;
function srt(th){var col=+th.dataset.col;if(_c===col){_a=!_a}else{_c=col;_a=false}
document.querySelectorAll('.sarr').forEach(function(s){s.textContent=''});
document.getElementById('sa'+col).textContent=_a?'↑':'↓';
var tb=document.querySelector('#mt tbody');var rs=Array.from(tb.rows);
var num=rs.length&&!isNaN(parseFloat(rs[0].cells[col].dataset.s));
rs.sort(function(x,y){var a=x.cells[col].dataset.s,b=y.cells[col].dataset.s;
if(num){a=parseFloat(a);b=parseFloat(b);if(a===-999&&b===-999)return 0;if(a===-999)return 1;if(b===-999)return -1;}
return a<b?(_a?-1:1):a>b?(_a?1:-1):0});rs.forEach(function(r){tb.appendChild(r)})}
</script>
"""


def _row(r, today, sec_label, sec_score, leader_hold, yf_to_sector, clickable=False) -> str:
    """One table row (<tr>...</tr>) of the scan table."""
    h = ''
    t = _s(r.get('ticker'))
    sect = yf_to_sector.get(_s(r.get('sector')), _s(r.get('sector')))
    ssc = sec_score.get(sect)
    # RS label (held) + days + 1M chip
    hs = r.get('held_score')
    lab = _s(r.get('held_label')).title()
    days = r.get('days_in_label')
    tag = _s(r.get('tag_1m'))
    tag_html = chip(tag, ORANGE if tag == '1M ↓' else TEAL, True) if tag else ''
    pend = _s(r.get('pending_label'))
    pend_html = f' <span style="color:#e3b48e;font-size:0.66rem;">→ {_e(pend.title())}</span>' if pend else ''
    rs_html = (f'<b style="color:{_score_col(hs)};">{"" if _nan(hs) else f"{float(hs):+.1f}"} {_e(lab)}</b>'
               f' <span class="sub">{"" if _nan(days) else str(int(float(days))) + "d"}</span> {tag_html}{pend_html}')
    ext = r.get('ext_atr')
    ez = _s(r.get('ext_zone'))
    ez = {'Stretched': 'Above', 'Extreme': 'Far above'}.get(ez, ez)   # labels before 28 Oct 2026
    mz = _s(r.get('mom_zone'))
    r5 = r.get('ret_5d')
    pb = _s(r.get('pullback'))
    pb_txt = '' if pb in ('', 'None') else f' <span class="sub">{_e(pb.lower())}</span>'
    mode = _s(r.get('pass_mode'))
    cs = chip(mode, TEAL if mode == 'STRICT' else COPPER, True) if mode else '<span style="color:#6e675f;">—</span>'
    eps = r.get('eps_qtr_yoy')
    atr_c = str(r.get('atr_compressed')).lower() == 'true'
    sq_html, sq_sort = squeeze_cell(r)
    ne = pd.to_datetime(_s(r.get('next_earnings')) or None, errors='coerce')
    if pd.isna(ne):
        e_html, e_sort = '<span style="color:#6e675f;">—</span>', 99999
    else:
        dd = (ne - today).days
        soon = 0 <= dd <= 7
        e_html = (f'<span style="color:{ORANGE if soon else "#a89f95"};font-weight:{700 if soon else 400};">'
                  f'{ne:%b %d}{" · " + str(dd) + "d" if 0 <= dd <= 14 else ""}</span>')
        e_sort = dd
    etf = leader_hold.get(t, '')
    p52 = r.get('pct_from_52w_high')
    if _nan(p52):
        p52 = r.get('pct_52w')
    p52c = DIM if _nan(p52) else TEAL if float(p52) > -0.05 else COPPER if float(p52) > -0.15 else ORANGE_M
    h += f'<tr data-t="{_e(t)}">' if clickable else '<tr>'
    h += f'<td class="l" data-s="{_e(t)}" title="{_e(_s(r.get("name")))}"><b>{_e(t)}</b></td>'
    hs_ = r.get('heat_score')                       # heat score 1-100 (heat_score.py, 4 Nov 2026)
    if _nan(hs_):
        h += '<td class="c" data-s="-999"><span style="color:#6e675f;">—</span></td>'
    else:
        hn_, hc_ = heat_score.band(float(hs_))
        h += (f'<td class="c" data-s="{float(hs_)}" title="Heat score {int(float(hs_))}/100 ({hn_}): RS rating, RS label '
              f'and trend tag, industry (group state + leader), sector, 52-week high, trend. Descriptive, untested.">'
              f'<span style="display:inline-block;min-width:30px;border-radius:3px;padding:0 5px;font-family:IBM Plex Mono,'
              f'monospace;font-weight:700;color:#121212;background:{hc_};">{int(float(hs_))}</span></td>')
    h += (f'<td class="l" data-s="{_e(sect)}"><span style="color:{_score_col(ssc)};">{_e(sect)}</span>'
          f'{" <span class=sub>" + _e(sec_label.get(sect, "").title()) + "</span>" if sec_label.get(sect) else ""}</td>')
    cl = r.get('close')
    cl_txt = '—' if _nan(cl) else '${:,.2f}'.format(float(cl))
    h += f'<td class="num" data-s="{_sv(cl)}">{cl_txt}</td>'
    h += f'<td class="num" data-s="{_sv(p52)}" style="color:{p52c};">{_pct(p52)}</td>'
    h += f'<td class="l" data-s="{_sv(hs) * 1000 + (0 if _nan(days) else float(days))}">{rs_html}</td>'
    for k in ('rs_3m', 'rs_1m'):
        h += f'<td class="num" data-s="{_sv(r.get(k))}" style="color:{_col(r.get(k))};">{_pct(r.get(k))}</td>'
    # RS rating (3 Nov 2026): IBD-style percentile 1-99 across the scanned universe; >= 80 = CAN SLIM "L"
    rr = r.get('rs_rating')
    rr_c = DIM if _nan(rr) else TEAL if float(rr) >= 80 else '#e8dccb' if float(rr) >= 50 else COPPER
    h += (f'<td class="num" data-s="{_sv(rr)}" style="color:{rr_c};" title="RS rating: weighted 12-month performance '
          f'(last quarter x2) ranked 1-99 across the scanned universe; 80+ = top 20%">'
          f'{"—" if _nan(rr) else int(float(rr))}</td>')
    h += (f'<td class="num" data-s="{_sv(ext)}">{"—" if _nan(ext) else f"{float(ext):+.2f}"} '
          f'{chip(ez, EXT_COL.get(ez, DIM), ez in ("Near", "Just below"))}</td>')
    h += f'<td class="c" data-s="{_e(mz)}">{chip(mz, MOM_COL.get(mz, DIM), mz in ("Persistent", "Building"))}</td>'
    h += f'<td class="num" data-s="{_sv(r5)}" style="color:{_col(r5)};">{"—" if _nan(r5) else f"{float(r5):+.1f}%"}{pb_txt}</td>'
    h += f'<td class="c" data-s="{ {"STRICT": 2, "RELAXED": 1}.get(mode, 0) }">{cs}</td>'
    low = str(r.get('eps_low_base')).lower() == 'true'
    low_html = (' <span class="sub" title="Growth from a loss or a near-zero base a year ago (or above +300%): not '
                'counted for CAN SLIM" style="color:#e3b48e;">low base</span>') if low else ''
    h += f'<td class="num" data-s="{_sv(eps)}" style="color:{DIM if low else _col(eps)};">{_pct(eps, 0)}{low_html}</td>'
    h += f'<td class="c" data-s="{int(atr_c)}" style="color:{"#e8dccb" if atr_c else "#6e675f"};">{"●" if atr_c else "—"}</td>'
    h += f'<td class="c" data-s="{sq_sort}">{sq_html}</td>'
    b_html, b_sort = base_cell(r)
    h += f'<td class="l" data-s="{b_sort}">{b_html}</td>'
    o_html, o_sort = breakout_cell(r)
    h += f'<td class="c" data-s="{o_sort}">{o_html}</td>'
    h += f'<td class="num" data-s="{e_sort}">{e_html}</td>'
    i_html, i_sort = ind_cell(r, etf)
    h += f'<td class="l" data-s="{i_sort}">{i_html}</td>'
    h += '</tr>'
    return h


def build_html(df: pd.DataFrame, sec_label: dict, sec_score: dict, leader_hold: dict, yf_to_sector: dict,
               clickable: bool = False) -> str:
    """df: filtered scan rows. sec_label/sec_score: display sector -> held label / score.
    leader_hold: ticker -> industry ETF on the buy-leaders list whose top 10 holds it."""
    today = pd.Timestamp.today().normalize()
    h = CSS + '<div class="wrap"><table id="mt"><thead><tr>'
    for i, (lbl, cls) in enumerate(HEAD):
        h += f'<th class="{cls}" data-col="{i}" onclick="srt(this)">{lbl}<span class="sarr" id="sa{i}"></span></th>'
    h += '</tr></thead><tbody>'
    for _, r in df.iterrows():
        h += _row(r, today, sec_label, sec_score, leader_hold, yf_to_sector, clickable)
    h += '</tbody></table></div>' + ('' if clickable else JS)
    return h


VCSS = """
<style>
.vt{width:100%;border-collapse:collapse;font-family:Barlow,system-ui,sans-serif;}
.vt th{background:transparent;color:#a89f95;font-size:0.70rem;font-weight:600;text-align:left;padding:4px 10px 4px 0;
       white-space:nowrap;vertical-align:top;cursor:default;position:static;width:1%;}
.vt td{background:transparent;color:#f3ece2;font-size:0.76rem;padding:4px 0;text-align:left !important;
       border-bottom:1px solid #1a1918;white-space:normal;}
.vt td.num{font-family:'IBM Plex Mono',monospace;}
.vt .sub{color:#a89f95;font-size:0.68rem;}
</style>
"""


def vertical_html(r, sec_label: dict, sec_score: dict, leader_hold: dict, yf_to_sector: dict) -> str:
    """The same cells as one scan-table row, as a label / value list (watchlist side panel, 4 Nov 2026)."""
    row = _row(r, pd.Timestamp.today().normalize(), sec_label, sec_score, leader_hold, yf_to_sector)
    tds = re.findall(r'<td\b.*?</td>', row, flags=re.S)
    pairs = list(zip([lbl for lbl, _ in HEAD], tds))
    half = (len(pairs) + 1) // 2
    h = VCSS + '<div style="display:flex;flex-wrap:wrap;gap:0 18px;">'
    for part in (pairs[:half], pairs[half:]):        # two columns side by side (one column on a narrow screen)
        h += '<table class="vt" style="flex:1 1 240px;width:auto;">'
        for lbl, td in part:
            h += f'<tr><th>{lbl}</th>{td}</tr>'
        h += '</table>'
    return h + '</div>'
