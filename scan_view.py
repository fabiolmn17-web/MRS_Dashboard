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


# Squeeze (Bollinger inside Keltner): dot colours (orange = High, red = Mid, black = Low, green = none) and momentum colours
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
        ('RS 3M', ''), ('RS 1M', ''), ('RS rtg', ''), ('Ext vs SMA50', ''), ('Momentum', 'c'), ('5-day', ''), ('RVOL', ''), ('Growth', 'c'),
        ('EPS QoQ', ''), ('Setup', 'l'), ('Squeeze', 'c'), ('Base / pivot', 'l'), ('Breakout', 'c'), ('Earnings', ''),
        ('Industry (rank · state)', 'l')]

BASE_ABBR = {'Cup w/ handle': 'Cup·H', 'Flat base': 'Flat', 'High tight flag': 'HTF', 'VCP': 'VCP',
             'Tight area': 'Tight', 'Double bottom': 'DB', 'Base-on-base': 'BoB', 'Ascending base': 'Asc',
             'IPO base': 'IPO', 'Cup': 'Cup', '52-week high': '52W hi', '52-week low': '52W lo'}   # v4: Cup, 52W
# bases v3 (8 Nov 2026): short labels for the fault chips (full text in the tooltip)
FAULT_ABBR = [('wedging handle', 'wedge'), ('V: straight up', 'V'), ('left side', 'long left'), ('late stage', 'late'),
              ('light-volume', 'light vol'), ('distribution', 'distrib'), ('wide and loose', 'loose'),
              ('deeper than the market', 'too deep')]


def _truthy(v):
    return str(v).strip().lower() in ('true', '1', '1.0')


def quality_html(r):
    """bases v3: 'Q k/7' chip (teal 5+, copper 3-4, grey 0-2) and orange fault chips; tooltips carry the detail.
    '' for scans before 8 Nov 2026."""
    q, qm = r.get('base_q'), r.get('base_q_max')
    if _nan(q) or _nan(qm):
        return ''
    q, qm = int(float(q)), int(float(qm))
    col = TEAL if q >= 5 else COPPER if q >= 3 else DIM
    detail = _s(r.get('base_q_detail'))
    out = f' <span title="{_e("Base quality checks (descriptive, untested): " + detail)}">{chip(f"Q {q}/{qm}", col, q >= 5)}</span>'
    fl = [f for f in _s(r.get('base_faults')).split('; ') if f and f != 'None']
    for f in fl[:2]:
        ab = next((a for k, a in FAULT_ABBR if k in f), f)
        out += f' <span title="{_e("Fault: " + f)}">{chip(ab, ORANGE, False)}</span>'
    if len(fl) > 2:
        out += f' <span class="sub" title="{_e("; ".join(fl))}">+{len(fl) - 2}</span>'
    if _truthy(r.get('base_bo_ftd')):
        out += (' <span title="Breakout within 20 sessions after a follow-through day on SPY">'
                + chip('FTD', TEAL, True) + '</span>')
    if _truthy(r.get('db_sk3')):
        lv = r.get('db_sk3_level')
        out += (f' <span title="{_e("Shakeout+3: the close cleared 10% above the double-bottom shakeout low " + ("" if _nan(lv) else f"({float(lv):,.2f})") + " in the last 3 sessions")}">'
                + chip('SK+3', TEAL, True) + '</span>')
    return out


def short_base_cell(r):
    """7 Nov 2026: bear base (bases.detect_short): pattern under a pivot LOW; distance = close vs that low."""
    bt = _s(r.get('sb_type'))
    if not bt or bt == 'None':
        return '<span style="color:#6e675f;">—</span>', -1
    st_ = _s(r.get('sb_state'))
    pv, dist = r.get('sb_pivot'), r.get('sb_dist_pct')
    broke = st_.startswith('Broke down')
    col = ORANGE if broke else COPPER if st_ == 'Near breakdown' else DIM
    da = r.get('sb_dist_atr')
    txt = _e(st_.replace('Broke down ', 'down ')) if broke else (
        f'{float(da):+.1f} ATR' if not _nan(da) else '' if _nan(dist) else f'{float(dist) * 100:+.1f}%')   # v4: ATR
    pv_txt = '' if _nan(pv) else f' <span class="sub">to {float(pv):,.2f}</span>'
    w = r.get('sb_weeks')
    wk = '' if _nan(w) else f' <span class="sub">{float(w):.0f}w</span>'
    rank = {'Broke down today': 4}.get(st_, 3 if broke else 2 if st_ == 'Near breakdown' else 1)
    sk = rank * 1000 - (0 if _nan(dist) else float(dist) * 100)
    title = _e(f"Bear {bt.lower()} (mirrored rules): breakdown level {pv}")
    return (f'<span title="{title}">{chip(BASE_ABBR.get(bt, bt) + " ↓", col, broke or st_ == "Near breakdown")}{wk} '
            f'<span style="color:{col};">{txt}</span>{pv_txt}</span>'), sk


def rvol_cell(r) -> str:
    """9 Nov 2026: RVOL (Epistruct RVOL at Time, daily): today's volume / the average of the last 20 sessions (anchor =
    chart timeframe). Tooltip: week-to-date vs the same weekday over 20 weeks (anchor 1W, cumulative)."""
    v, w = r.get('rvol_d'), r.get('rvol_w')
    if _nan(v):
        return '<td class="num" data-s="-999"><span style="color:#6e675f;">—</span></td>'
    v = float(v)
    col, fw, zone = ((ORANGE, 700, 'extreme') if v >= 2 else (COPPER, 700, 'heavy') if v >= 1.5
                     else ('#6e675f', 400, 'quiet') if v < 0.8 else ('#e8dccb', 400, 'normal'))
    tip = (f'RVOL {v:.2f}x ({zone}): today\'s volume vs the average of the last 20 sessions. '
           + ('' if _nan(w) else f'Week to date vs the same weekday over the last 20 weeks: {float(w):.2f}x. ')
           + 'Quiet < 0.8, heavy >= 1.5, extreme >= 2. Same rules as the Epistruct RVOL at Time indicator. Descriptive.')
    return (f'<td class="num" data-s="{v}" title="{_e(tip)}" style="color:{col};font-weight:{fw};">{v:.2f}×</td>')


def setup_cell(r):
    """7 Nov 2026: pullback to a rising 20 / 50-day SMA (long) or rally to a falling one (short), from momentum.py."""
    mode = _s(r.get('_dd_mode'))
    if mode:                                 # 8 Nov 2026: Deep dip / Deep rip triggers and watch list
        sh = r.get('_side') == 'short'
        pre = 'dr' if sh else 'dd'
        col = ORANGE if sh else TEAL
        if mode == 'watch':
            w, xa = _s(r.get(f'{pre}_watch')), r.get(f'{pre}_watch_atr')
            if not w:
                return '<span style="color:#6e675f;">—</span>', 0
            d = '' if _nan(xa) else f' <span class="sub">{float(xa):.1f} ATR</span>'
            return f'<span title="{_e(w + " SMA: no trigger yet")}">{chip(w.replace("Near ", "≈"), col, False)}{d}</span>', 1
        tg = _s(r.get(f'{pre}_trig'))
        if not tg:
            return '<span style="color:#6e675f;">—</span>', 0
        k, sma, days = r.get(f'{pre}_k'), _s(r.get(f'{pre}_sma')), r.get(f'{pre}_days')
        stop, risk, rv = r.get(f'{pre}_stop'), r.get(f'{pre}_risk_pct'), r.get(f'{pre}_rvol')
        ab = (not sh and r.get('ema21_above') in (True, 'True', 1)) or (sh and r.get('ema21_above') in (False, 'False', 0))
        short_t = {'Undercut': 'UC', 'Reclaim': 'RC', 'Overcut': 'OC', 'Lost reclaim': 'LR'}.get(tg, tg)
        lab = f'{short_t} {sma}' + ('' if _nan(k) or int(float(k)) == 0 else f' ·{int(float(k))}')
        dtxt = '' if _nan(days) else ('today' if int(float(days)) == 0 else f'{int(float(days))}d ago')
        sub = ' '.join(x for x in [dtxt, '' if _nan(stop) else f'stop {float(stop):,.2f}',
                                    '' if _nan(risk) else f'({float(risk):.1f}%)',
                                    '' if _nan(rv) else f'vol {float(rv):.1f}×', ('↓21' if sh else '↑21') if ab else ''] if x)
        title = (f'{tg} of the {sma} SMA' + ('' if _nan(k) or int(float(k)) == 0 else f' after {int(float(k))} close(s) beyond')
                 + f', {dtxt}. Entry ref = trigger close {r.get(f"{pre}_entry")}; stop = trigger-day '
                 + ('high + 1 ATR(10)' if sh else 'low − 1 ATR(10)') + '. vol = trigger-day volume ÷ the 10 sessions before.')
        rank = 10 - (0 if _nan(days) else int(float(days)))
        return f'<span title="{_e(title)}">{chip(lab, col, True)} <span class="sub">{_e(sub)}</span></span>', rank
    dp = _s(r.get('_dip'))
    if dp:                                   # 7 Nov 2026 (c): Deep dip / Deep rip scans (old proximity version)
        col = TEAL if r.get('_side') != 'short' else ORANGE
        return f'<span title="{_e(_s(r.get("_dip_t")))}">{chip(dp, col, True)}</span>', 3
    s_ = _s(r.get('pb_setup')) if r.get('_side') != 'short' else _s(r.get('rally_setup'))
    if not s_ or s_ == 'None':
        return '<span style="color:#6e675f;">—</span>', 0
    is50 = '50' in s_
    col = TEAL if r.get('_side') != 'short' else ORANGE
    lab = ('PB ' if s_.startswith('Pullback') else 'Rally ') + ('50d' if is50 else '20d')
    return chip(lab, col, is50), (2 if is50 else 1)


def base_cell(r):
    """(html, sort key) for the Base / pivot column (bases.py fields; '—' for older CSVs)."""
    if r.get('_side') == 'short':
        return short_base_cell(r)
    bt = _s(r.get('base_type'))
    if not bt or bt == 'None':
        return '<span style="color:#6e675f;">—</span>', -1
    st_ = _s(r.get('base_state'))
    pv, dist = r.get('base_pivot'), r.get('base_dist_pct')
    broke = st_.startswith('Broke out')
    col = TEAL if broke else COPPER if st_ == 'Near pivot' else DIM
    q = r.get('base_quality')
    stars = quality_html(r)
    if not stars:                                   # scans before bases v3 (8 Nov 2026): the old 0-3 stars
        stars = '' if _nan(q) else ' <span class="sub" title="Quality: volume dry-up, tight closes, RS rating ≥ 80">' + \
            '★' * int(float(q)) + '☆' * (3 - int(float(q))) + '</span>'
    da = r.get('base_dist_atr')
    extra = ''
    if broke:
        bv = r.get('base_bo_vol')
        if _truthy(r.get('base_bo_light')):        # v3: a light-volume breakout reads grey (not a buy)
            col = DIM
        vtxt = '' if _nan(bv) else (f' <span style="color:{TEAL if float(bv) >= 1.4 else DIM};">'
                                    f'{float(bv):.1f}× vol</span>')
        # v4 (9 Nov 2026): how far above the pivot in ATR (orange beyond 1.5 = extended)
        atxt = '' if _nan(da) else (f' <span style="color:{ORANGE if float(da) > 1.5 else TEAL};">'
                                    f'+{float(da):.1f} ATR</span>')
        txt = f'{_e(st_.replace("Broke out ", "out "))}{vtxt}{atxt}'
    else:
        # v4: distance below the pivot in ATR (near = within 1.5 ATR); % in the tooltip
        txt = (f'{float(da):+.1f} ATR' if not _nan(da) else '' if _nan(dist) else f'{float(dist) * 100:+.1f}%')
    oh, oha = r.get('overhead'), r.get('overhead_atr')
    if (not _nan(oh) and not _nan(oha) and float(oha) <= 2.0       # v4: overhead pivot high close above,
            and (_nan(pv) or float(oh) > 1.005 * float(pv))):     # when it is above the pivot itself
        extra = f' <span class="sub" title="Next overhead pivot high {float(oh):,.2f}, {float(oha):.1f} ATR above the close">↑{float(oh):,.2f}</span>'
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
    cl = r.get('base_bo_cl')
    title = _e(f"{bt}: pivot {pv}" + ('' if _nan(dep) else f", depth {float(dep) * 100:.0f}%")
               + ('' if not _s(r.get('base_start')) else f", from {_s(r.get('base_start'))}")
               + ('' if _nan(dist) else f", close {float(dist) * 100:+.1f}% vs pivot")
               + ('' if not broke or _nan(cl) else f", breakout day closed at {float(cl) * 100:.0f}% of its range")
               + ('' if _nan(oh) else f", next overhead pivot high {float(oh):,.2f}"))
    return (f'<span title="{title}">{chip(BASE_ABBR.get(bt, bt), col, broke or st_ == "Near pivot")}{wk} '
            f'<span style="color:{col};">{txt}</span>{pv_txt}{stars}{extra}</span>'), sk


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
    if tag == '1M ↓':                               # dropped 8 Nov 2026 (audit v2: Null); older scans may carry it
        tag = ''
    tag_html = chip(tag, TEAL, True) if tag else ''
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
              f'and trend tag, industry (group state + leader), sector, 52-week high, trend. Describes leadership, not a return '
              f'forecast: 2009–26, top-quartile heat earned the same as the universe (±0.3 pp a month). Descriptive.">'
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
    # RS rating (3 Nov 2026): percentile 1-99 across the scanned universe; >= 80 = the Growth screen's RS gate
    rr = r.get('rs_rating')
    rr_c = DIM if _nan(rr) else TEAL if float(rr) >= 80 else '#e8dccb' if float(rr) >= 50 else COPPER
    h += (f'<td class="num" data-s="{_sv(rr)}" style="color:{rr_c};" title="RS rating: weighted 12-month performance '
          f'(last quarter x2) ranked 1-99 across the scanned universe; 80+ = top 20%">'
          f'{"—" if _nan(rr) else int(float(rr))}</td>')
    h += (f'<td class="num" data-s="{_sv(ext)}">{"—" if _nan(ext) else f"{float(ext):+.2f}"} '
          f'{chip(ez, EXT_COL.get(ez, DIM), ez in ("Near", "Just below"))}</td>')
    h += f'<td class="c" data-s="{_e(mz)}">{chip(mz, MOM_COL.get(mz, DIM), mz in ("Persistent", "Building"))}</td>'
    h += f'<td class="num" data-s="{_sv(r5)}" style="color:{_col(r5)};">{"—" if _nan(r5) else f"{float(r5):+.1f}%"}{pb_txt}</td>'
    h += rvol_cell(r)                               # 9 Nov 2026: RVOL column (sortable)
    h += f'<td class="c" data-s="{ {"STRICT": 2, "RELAXED": 1}.get(mode, 0) }">{cs}</td>'
    low = str(r.get('eps_low_base')).lower() == 'true'
    low_html = (' <span class="sub" title="Growth from a loss or a near-zero base a year ago (or above +300%): not '
                'counted for the Growth screen" style="color:#e3b48e;">low base</span>') if low else ''
    h += f'<td class="num" data-s="{_sv(eps)}" style="color:{DIM if low else _col(eps)};">{_pct(eps, 0)}{low_html}</td>'
    su_html, su_sort = setup_cell(r)                # 7 Nov 2026: replaces the ATR-compression column
    h += f'<td class="l" data-s="{su_sort}">{su_html}</td>'
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
