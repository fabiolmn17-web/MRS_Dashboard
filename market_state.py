"""
market_state.py — "Market state" banner, momentum light and "Leaders to buy" shortlist
=======================================================================================
Turns the 25 Oct 2026 studies into desk lines. Every line carries an evidence tag:

  Tested       formal test in the Epistruct battery (non-overlapping samples, bootstrap;
               Report v7.0 for the risk dial, studies/momentum_rs_persistence for momentum)
  Descriptive  a historical regularity that did NOT pass multiple-testing correction
               (studies/industry_by_mrs_band) — context, not a rule
  Watch        not tested; shown because it is cheap to watch (RS 1M early warning)

Nothing here is a buy signal on its own. The shortlist says where to look first for longs.
"""
import html as _html
import math

import numpy as np
import pandas as pd

TEAL, TEAL_M, ORANGE, ORANGE_M = '#85cac5', '#649995', '#f4a062', '#b9805a'
COPPER, COPPER_L, INK, MUTED, DIM = '#e3b48e', '#f8d6b4', '#f3ece2', '#a89f95', '#9b958f'

EVIDENCE = {
    'Tested':      (COPPER, 'rgba(227,180,142,0.14)', 'solid',
                    'Formally tested (non-overlapping samples, bootstrap)'),
    'Descriptive': (MUTED, 'rgba(0,0,0,0)', 'solid',
                    'Historical regularity, did not pass multiple-testing correction: context, not a rule'),
    'Watch':       (COPPER_L, 'rgba(0,0,0,0)', 'dashed',
                    'Not tested: an early warning worth watching'),
}

BAND_ACTION = {
    'RISK-ON':       '<b>Small swings both ways.</b> Normal size. Big runs are rare: take profits at your targets.',
    'MILD RISK-ON':  '<b>Fairly calm.</b> Normal size.',
    'NEUTRAL':       '<b>Average swings.</b> Normal size.',
    'MILD RISK-OFF': '<b>Swings widening both ways.</b> Your stop stays where the setup is wrong; if it is hit, you are out.',
    'RISK-OFF':      ('<b>Widest swings, up and down.</b> No extra room for the stop: it sits where the setup is wrong, '
                      'and if it is hit you are out. Time the entry well and the upside is large: let winners run. '
                      '<b>Not a sell signal.</b>'),
}
REGIME_SUB = 'The MRS tells you how wide the next weeks can be, not which way.'


def sq_plain(desc: str) -> str:
    """One plain line from the signal-quality description (the full text stays in the tooltip)."""
    import re as _re
    out = []
    m = _re.search(r'driven by (.*?)\.\s', desc or '')
    if m:
        names = [x.strip() for x in _re.sub(r'\([^)]*\)', '', m.group(1)).split(',') if x.strip()]
        out.append('Driven by ' + ', '.join(names))
    b = _re.search(r'B20 Phi=([0-9.]+)', desc or '')
    if 'Breadth is not scoring' in (desc or '') and b:
        out.append(f'breadth does not confirm (B20 percentile {float(b.group(1)) * 100:.0f})')
    elif 'Breadth' in (desc or '') and 'confirm' in (desc or '').lower() and b:
        out.append(f'breadth percentile {float(b.group(1)) * 100:.0f}')
    if not out:
        first = (desc or '').split('. ')[0]
        return first[:140] + ('…' if len(first) > 140 else '')
    return '; '.join(out) + '.'

COMMODITY = 'SLX, COPX, XME, GDX, LIT, URA'

CSS = (
    '<style>'
    '.ms{background:#121212;border:1px solid #2a2724;border-left:3px solid #e3b48e;border-radius:6px;'
    'padding:12px 16px 10px 16px;margin:4px 0 10px 0;}'
    '.ms .k{font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#e3b48e;font-weight:700;margin-bottom:6px;}'
    '.ms .ln{display:flex;align-items:baseline;gap:10px;padding:5px 0;border-top:1px solid #1a1918;font-size:0.84rem;'
    'color:#f3ece2;line-height:1.45;}'
    '.ms .ln:first-of-type{border-top:none;}'
    '.ms .h{min-width:118px;color:#a89f95;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;}'
    '.ms .t{flex:1;}'
    '.ms .sub{color:#a89f95;font-size:0.76rem;}'
    '@media (max-width:760px){.ms .ln{flex-wrap:wrap;gap:4px 10px;}.ms .h{min-width:100%;}.ms .t .sub{display:none;}'
    '.ds .top .sm[title]{display:none;}}'
    '.ev{display:inline-block;border:1px solid;border-radius:3px;padding:0 5px;font-size:0.64rem;font-weight:700;'
    'letter-spacing:0.04em;white-space:nowrap;line-height:1.5;}'
    '.ml{display:inline-flex;align-items:center;gap:8px;font-weight:800;}'
    '.ml .dot{width:11px;height:11px;border-radius:50%;display:inline-block;}'
    '.ld{margin:6px 0 10px 0;overflow-x:auto;}'
    '.ld .hdr{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px;}'
    '.ld .ttl{font-size:0.95rem;font-weight:800;color:#f3ece2;}'
    '.ld table{width:100%;border-collapse:collapse;font-size:0.78rem;}'
    '.ld th{background:#1a1918;color:#a89f95;font-size:0.68rem;font-weight:600;padding:6px 9px;text-align:right;'
    'border-bottom:1px solid #2a2724;white-space:nowrap;}'
    '.ld th.l,.ld td.l{text-align:left;}'
    '.ld td{background:#0a0a0a;color:#f3ece2;padding:6px 9px;text-align:right;border-bottom:1px solid #1a1918;white-space:nowrap;}'
    '.ld td.num{font-family:"IBM Plex Mono",monospace;}'
    '.ld .note{font-size:0.70rem;color:#9b958f;margin-top:6px;line-height:1.5;}'
    '.ld table,.ld th,.ld td{border-left:none;border-right:none;border-top:none;}'
    '</style>'
)


def _e(s):
    return _html.escape(str(s), quote=True)


def _nan(v):
    try:
        return v is None or math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def ev(kind: str) -> str:
    c, bg, style, tip = EVIDENCE[kind]
    return (f'<span class="ev" title="{_e(tip)}" style="color:{c};border-color:{c};background:{bg};'
            f'border-style:{style};">{kind.upper()}</span>')


def _pct(v, dec=1):
    if _nan(v):
        return '—'
    v = float(v)
    return f'{"+" if v > 0 else ""}{v * 100:.{dec}f}%'


def light_html(light: dict) -> str:
    on = light['on']
    col = TEAL if on else ORANGE
    txt = 'Momentum ON' if on else 'Momentum CAUTION'
    glow = 'rgba(133,202,197,0.35)' if on else 'rgba(244,160,98,0.35)'
    return (f'<span class="ml" style="color:{col};"><span class="dot" style="background:{col};'
            f'box-shadow:0 0 0 3px {glow};"></span>{txt}</span>')


def light_detail(light: dict) -> str:
    r12 = light.get('spy_12m')
    r12_txt = '' if _nan(r12) else f'SPY 12-month {_pct(r12)}. '
    if light['on']:
        return (f'{r12_txt}{light["why"]}: relative-strength leaders have tended to keep leading. '
                'Normal size on leaders.')
    return (f'{r12_txt}{light["why"]}: momentum-crash state, leaders can reverse hard. '
            'Smaller size and faster exits on leaders (not a ban).')


def swing_line(band: str, bp: pd.DataFrame, h: int = 20) -> str:
    """Risk-dial line for today's band at h sessions (Report v7.0)."""
    try:
        b = bp[(bp['band'] == band) & (bp['horizon'] == h)].iloc[0]
        a = bp[(bp['band'] == 'ALL') & (bp['horizon'] == h)].iloc[0]
    except Exception:
        return BAND_ACTION.get(band, '')
    x = int(round(b['thresh'] * 100))
    return (f'{BAND_ACTION.get(band, "")} <span class="sub">Within {h} sessions SPY moved ≥{x}% against a new long in '
            f'{b["p_adverse"] * 100:.0f}% of {band} days (all days {a["p_adverse"] * 100:.0f}%) and ≥{x}% in favour in '
            f'{b["p_favorable"] * 100:.0f}% (all {a["p_favorable"] * 100:.0f}%).</span>')


def rotation_line(band: str) -> str:
    if band == 'RISK-ON':
        return (f'Commodities &amp; metals ({COMMODITY}) have lagged SPY in RISK-ON: beat SPY in 38% of 20-session '
                'windows vs 46% otherwise. A long there needs its own reason.')
    if band == 'RISK-OFF':
        return ('Defensives have given no edge in RISK-OFF (about +0.1 pp vs SPY at 20 sessions). '
                f'Commodities &amp; metals ({COMMODITY}) tended to rebound harder (+0.7 pp, beat SPY 51% vs 46%).')
    return ('Middle bands flip almost daily (median episode 1 session): do not rotate on a band change. '
            'Sector leadership by band is not reliable in any band.')


def banner_html(light: dict, band: str, band_col: str, bp: pd.DataFrame, short: pd.DataFrame) -> str:
    h = CSS + '<div class="ms"><div class="k">Market state · what to do with it</div>'
    h += (f'<div class="ln"><div class="h">Momentum</div><div class="t">{light_html(light)} '
          f'<span class="sub">&nbsp;{_e(light_detail(light))}</span></div>{ev("Tested")}</div>')
    h += (f'<div class="ln"><div class="h">Swing size</div><div class="t"><b style="color:{band_col};">{_e(band)}</b> · '
          f'{swing_line(band, bp)}</div>{ev("Tested")}</div>')
    h += (f'<div class="ln"><div class="h">Rotation</div><div class="t">{rotation_line(band)}</div>'
          f'{ev("Descriptive")}</div>')
    if short is not None and len(short):
        names = ', '.join(f'<b>{_e(r["Industry"])}</b> <span class="sub">({_e(r["ETF"])})</span>'
                          for _, r in short.head(4).iterrows())
        more = f' <span class="sub">+{len(short) - 4} more</span>' if len(short) > 4 else ''
        h += (f'<div class="ln"><div class="h">Leaders</div><div class="t">{len(short)} industr'
              f'{"y" if len(short) == 1 else "ies"} on the buy-leaders list: {names}{more}. '
              f'<span class="sub">See "Leaders to buy" in the sector section.</span></div>{ev("Tested")}</div>')
    else:
        h += ('<div class="ln"><div class="h">Leaders</div><div class="t"><span class="sub">No industry passes the '
              'leaders filter today.</span></div></div>')
    h += '</div>'
    return h.replace('\n', ' ')


def _strong_holdings(hold_df: pd.DataFrame, etf: str, n: int = 3) -> str:
    """Up to n holdings with RS score >= +1 and within 10% of the 52-week closing high, by RS 3M."""
    if hold_df is None or len(hold_df) == 0 or 'ETF' not in hold_df.columns:
        return '—'
    g = hold_df[hold_df['ETF'] == etf].copy()
    if g.empty:
        return '—'
    for c in ('rs_composite', 'pct_from_52w_high', 'rs_3m'):
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g = g[(g['rs_composite'] >= 1.0) & (g['pct_from_52w_high'] >= -0.10)].sort_values('rs_3m', ascending=False)
    if g.empty:
        return '<span style="color:#9b958f;">none near highs</span>'
    return ', '.join(f'<b>{_e(t)}</b>' for t in g['ticker'].head(n))


def _earn_soon(hold_df: pd.DataFrame, etf: str, days: int = 10) -> str:
    """Top-10 holdings of this ETF reporting within `days` calendar days (from the nightly job)."""
    if hold_df is None or len(hold_df) == 0 or 'next_earnings' not in hold_df.columns:
        return '<span style="color:#6e675f;">—</span>'
    g = hold_df[hold_df['ETF'] == etf].copy()
    g['d'] = pd.to_datetime(g['next_earnings'], errors='coerce')
    today = pd.Timestamp.today().normalize()
    g = g[(g['d'] >= today) & (g['d'] <= today + pd.Timedelta(days=days))].sort_values('d')
    if g.empty:
        return '<span style="color:#6e675f;">none in 10 days</span>'
    parts = [f'<b>{_e(t)}</b> <span style="color:#e3b48e;">{d:%a %d}</span>' for t, d in zip(g['ticker'], g['d'])]
    return ', '.join(parts[:3]) + (f' <span style="color:#a89f95;">+{len(parts) - 3}</span>' if len(parts) > 3 else '')


def _how(summary: str, body: str) -> str:
    return (f'<details class="note" style="margin-top:6px;"><summary style="cursor:pointer;color:#a89f95;">{summary}</summary>'
            f'<div style="margin-top:4px;">{body}</div></details>')


def leaders_html(short: pd.DataFrame, light: dict, hold_df: pd.DataFrame = None, as_of: str = None,
                 dropped: pd.DataFrame = None) -> str:
    h = CSS + '<div class="ld"><div class="hdr"><span class="ttl">Leaders to buy — industries first</span>'
    h += light_html(light) + ev('Tested') + '</div>'
    h += (f'<div class="note" style="font-size:0.80rem;color:#e8dccb;margin:-2px 0 4px 0;">{_e(light_detail(light))}</div>'
          '<div class="note" style="font-size:0.80rem;color:#e8dccb;margin:0 0 8px 0;">Where to look first for longs, '
          'not a basket to hold. Industry leaders keep a small edge vs SPY, but holding them as a group did not beat '
          'SPY in our 2008–2026 replay.</div>')
    if short is None or len(short) == 0:
        h += ('<div class="note" style="font-size:0.78rem;">No industry ETF passes the filter today '
              '(structural leader with RS 1M not negative, or outperforming and improving).</div></div>')
        return h.replace('\n', ' ')
    h += '<table><thead><tr>'
    for lbl, cls in [('Industry', 'l'), ('ETF', 'l'), ('Sector', 'l'), ('Label', 'l'), ('Days in label', ''),
                     ('RS 1M', ''), ('RS 3M vs SPY (sort)', 'l'), ('RS 6M', ''), ('RS 1Y', ''), ('Why on the list', 'l'),
                     ('Strong holdings', 'l'), ('Earnings, next 10 days', 'l')]:
        h += f'<th class="{cls}">{lbl}</th>'
    h += '</tr></thead><tbody>'
    for _, r in short.iterrows():
        sc = float(r['Score'])
        lc = TEAL if sc >= 1.0 else TEAL_M
        d = r.get('DaysInLabel')
        capped = str(r.get('DaysCapped')).lower() == 'true'
        d_txt = '—' if _nan(d) else f'{"≥" if capped else ""}{int(float(d))}d'
        pend = r.get('PendingLabel')
        pend = '' if _nan(pend) and not isinstance(pend, str) else str(pend)
        pend = '' if pend == 'nan' else pend
        pend_txt = (f'<br><span style="color:#e3b48e;font-size:0.68rem;font-weight:600;" title="Labels change only after '
                    f'3 sessions in a row at the new value">changing to {_e(pend.title())}: day '
                    f'{int(float(r.get("PendingDays") or 0))} of 3</span>') if pend else ''
        tag = str(r.get('Tag1M') or '')
        tag = '' if tag == 'nan' else tag
        why = _e(r['Why'])
        if tag == '1M ↑':
            why += ' · 1M ↑ ' + ev('Watch')
        new = ' <span class="ev" style="color:#85cac5;border-color:#85cac5;background:rgba(133,202,197,0.18);">NEW</span>' if bool(r.get('New', False)) else ''
        h += '<tr>'
        h += f'<td class="l"><b>{_e(r["Industry"])}</b>{new}</td>'
        h += f'<td class="l" style="color:#a89f95;">{_e(r["ETF"])}</td>'
        h += f'<td class="l" style="color:#a89f95;">{_e(r["Sector"])}</td>'
        h += f'<td class="l" style="color:{lc};font-weight:700;">{"+" if sc > 0 else ""}{sc:.1f} {_e(r["Label"])}{pend_txt}</td>'
        h += f'<td class="num" style="color:#a89f95;">{d_txt}</td>'
        for k in ('RS 1M', 'RS 3M', 'RS 6M', 'RS 1Y'):
            v = r.get(k)
            if k == 'RS 3M':
                h += f'<td class="l">{_rs_bar(v)}</td>'
                continue
            c = DIM if _nan(v) else (TEAL if float(v) > 0 else ORANGE if float(v) < 0 else DIM)
            h += f'<td class="num" style="color:{c};">{_pct(v)}</td>'
        h += f'<td class="l" style="color:#e8dccb;">{why}</td>'
        h += f'<td class="l">{_strong_holdings(hold_df, r["ETF"])}</td>'
        h += f'<td class="l">{_earn_soon(hold_df, r["ETF"])}</td>'
        h += '</tr>'
    h += '</tbody></table>'
    h += _how('How this list is built and what is tested',
              'Industry ETFs, held labels (a label changes only after 3 sessions in a row at the new value). On the list: '
              'structural leader (+1.5) whose RS 1M is not negative, or outperforming (+0.5) and IMPROVING (1Y+, 6M−, 3M+). '
              'Sorted by RS 3M. Sector rows are context only (sectors showed no momentum since 2008). '
              f'{ev("Tested")} industry leaders kept a small edge vs SPY (+0.4 / +0.8 / +1.5 pp at 1 / 3 / 6 months); '
              'within outperforming, IMPROVING beat SPY 55% vs 48% of months for FADING. The edge came from Momentum ON '
              'periods (+0.8 pp vs −1.9 pp at 1 month when SPY was below its 12-month level, mostly 2008–09). '
              f'{ev("Watch")} the RS 1M filter itself is not tested. {ev("Descriptive")} Strong holdings = top-10 '
              'holdings with RS score ≥ +1 within 10% of the 52-week closing high, by RS 3M: where to start the chart '
              'work, not a buy list.' + (f' Industry data as of {_e(as_of)}.' if as_of else ''))
    h += '</div>'
    return h.replace('\n', ' ')


# ── chips for the scanner ────────────────────────────────────────────────────────
EXT_COL = {'Deep below': ORANGE, 'Below': ORANGE_M, 'Just below': TEAL_M, 'Near': TEAL, 'Above': COPPER, 'Far above': COPPER_L,
           'Stretched': COPPER, 'Extreme': COPPER_L}   # old labels kept for older CSVs
MOM_COL = {'Persistent': TEAL, 'Building': TEAL_M, 'Fading': COPPER, 'Falling': ORANGE_M, 'Weak': ORANGE}


def chip(txt: str, col: str, filled: bool = False, title: str = '') -> str:
    if not txt:
        return '<span style="color:#6e675f;">—</span>'
    bg = col + '2e' if filled else 'transparent'
    t = f' title="{_e(title)}"' if title else ''
    return (f'<span{t} style="display:inline-block;border:1px solid {col};color:{col};background:{bg};border-radius:3px;'
            f'padding:0 5px;font-size:0.68rem;font-weight:700;white-space:nowrap;">{_e(txt)}</span>')


# ── decision strip: score, regime, what changed, what to do ───────────────────────
def decision_strip_html(mrs: float, band: str, band_col: str, sessions: int, pct_txt: str, sq_lbl: str, sq_col: str,
                        sq_desc: str, light: dict, bp: pd.DataFrame, short: pd.DataFrame, levels: dict,
                        changes: list, deep_line: str = '', fresh: str = '', show_stats: bool = True) -> str:
    """show_stats=False when the Risk today panel is on: it shows the same odds for today's exact state."""
    h = CSS + (
        '<style>'
        '.ds{background:#121212;border:1px solid #2a2724;border-left:3px solid #e3b48e;border-radius:6px;padding:12px 16px 8px 16px;margin:2px 0 12px 0;}'
        '.ds .top{display:grid;grid-template-columns:auto auto 1fr auto;gap:26px;align-items:start;padding-bottom:10px;border-bottom:1px solid #2a2724;}'
        '.ds .lab{font-size:0.64rem;letter-spacing:0.14em;text-transform:uppercase;color:#a89f95;margin-bottom:3px;}'
        '.ds .big{font-family:"IBM Plex Mono",monospace;font-size:2.1rem;font-weight:600;line-height:1;}'
        '.ds .reg{font-size:1.45rem;font-weight:800;line-height:1.05;}'
        '.ds .sm{font-size:0.74rem;color:#a89f95;}'
        '.ds .lv{font-family:"IBM Plex Mono",monospace;font-size:0.78rem;color:#f3ece2;line-height:1.55;text-align:right;}'
        '.ds .chg{display:flex;flex-wrap:wrap;gap:6px 18px;padding:9px 0 5px 0;font-size:0.82rem;color:#f3ece2;align-items:baseline;}'
        '.ds .chg .it{white-space:nowrap;}'
        '.ds .fresh{font-size:0.68rem;color:#6e675f;margin-top:4px;}'
        '@media (max-width:760px){.ds .top{grid-template-columns:1fr 1fr;}.ds .lv{text-align:left;}}'
        '</style>')
    h += '<div class="ds"><div class="k" style="font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#e3b48e;font-weight:700;margin-bottom:8px;">Today · what to do with it</div>'
    h += '<div class="top">'
    h += f'<div><div class="lab">MRS score</div><div class="big" style="color:{band_col};">{mrs:+.2f}</div><div class="sm">{_e(pct_txt)}</div></div>'
    h += (f'<div><div class="lab">Regime · session {sessions}</div><div class="reg" style="color:{band_col};">{_e(band)}</div>'
          f'<div class="sm" style="color:#e8dccb;margin-top:4px;max-width:250px;">{REGIME_SUB}</div></div>')
    h += (f'<div><div class="lab">Signal quality</div>{chip(sq_lbl, sq_col, True, sq_desc)}'
          f'<div class="sm" style="margin-top:5px;" title="{_e(sq_desc)}">{_e(sq_plain(sq_desc))} '
          f'<span style="color:#6e675f;cursor:help;">ⓘ</span></div></div>')
    h += ('<div class="lv">' + '<br>'.join(f'<span style="color:#a89f95;">{_e(k)}</span> {_e(v)}' for k, v in levels.items()) + '</div>')
    h += '</div>'
    if changes:
        h += ('<div class="chg"><span class="lab" style="margin:0;">Since the last session</span>'
              + ''.join(f'<span class="it">{c}</span>' for c in changes) + '</div>')
    h += '<div class="ms" style="background:transparent;border:none;padding:0;margin:0;">'
    stats = ''
    if show_stats:
        try:
            b = bp[(bp['band'] == band) & (bp['horizon'] == 20)].iloc[0]
            a = bp[(bp['band'] == 'ALL') & (bp['horizon'] == 20)].iloc[0]
            x = int(round(b['thresh'] * 100))
            stats = (f' <span class="sub">Next 20 sessions: ≥{x}% against a new long {b["p_adverse"] * 100:.0f}% of days '
                     f'(all days {a["p_adverse"] * 100:.0f}%), ≥{x}% in favour {b["p_favorable"] * 100:.0f}% '
                     f'(all {a["p_favorable"] * 100:.0f}%).</span>')
        except Exception:
            stats = ''
    h += (f'<div class="ln"><div class="h">What to do</div><div class="t" style="font-size:0.90rem;">'
          f'{BAND_ACTION.get(band, "")}{stats}</div>{ev("Tested")}</div>')
    h += f'<div class="ln"><div class="h">Rotation</div><div class="t">{rotation_line(band)}</div>{ev("Descriptive")}</div>'
    if deep_line:
        h += f'<div class="ln"><div class="h">Deep state</div><div class="t">{deep_line}</div>{ev("Descriptive")}</div>'
    h += '</div>'
    if fresh:
        h += f'<div class="fresh">{fresh}</div>'
    h += '</div>'
    return h.replace('\n', ' ')


# ── v5: Avoid list (laggards) ───────────────────────────────────────────────────
def _weak_holdings(hold_df: pd.DataFrame, etf: str, n: int = 3) -> str:
    """Up to n top-10 holdings with RS score <= -1, weakest RS 3M first."""
    if hold_df is None or len(hold_df) == 0 or 'ETF' not in hold_df.columns:
        return '—'
    g = hold_df[hold_df['ETF'] == etf].copy()
    if g.empty:
        return '—'
    for c in ('rs_composite', 'rs_3m'):
        g[c] = pd.to_numeric(g[c], errors='coerce')
    g = g[g['rs_composite'] <= -1.0].sort_values('rs_3m', ascending=True)
    if g.empty:
        return '<span style="color:#6e675f;">none ≤ −1</span>'
    return ', '.join(f'<b style="color:#f4a062;">{_e(t)}</b>' for t in g['ticker'].head(n))


def _rs_bar(v, lim: float = 0.20) -> str:
    """Diverging bar around zero: orange left (lagging SPY), teal right (leading)."""
    if _nan(v):
        return '—'
    v = float(v)
    w = min(abs(v) / lim, 1.0) * 50
    col = ORANGE if v < 0 else TEAL
    left = 50 - w if v < 0 else 50
    return (f'<span style="display:inline-flex;align-items:center;gap:6px;">'
            f'<span style="position:relative;display:inline-block;width:90px;height:8px;background:#1a1918;border-radius:2px;">'
            f'<span style="position:absolute;left:50%;top:-2px;width:1px;height:12px;background:#6e675f;"></span>'
            f'<span style="position:absolute;left:{left:.1f}%;width:{w:.1f}%;height:8px;background:{col};border-radius:2px;"></span>'
            f'</span><span class="num" style="color:{col};min-width:46px;text-align:right;">{_pct(v)}</span></span>')


def avoid_html(lag: pd.DataFrame, hold_df: pd.DataFrame = None) -> str:
    h = CSS + ('<div class="ld" style="margin-top:14px;"><div class="hdr"><span class="ttl" style="color:#f4a062;">'
               'Avoid: laggards — don\'t buy them for a catch-up</span>' + ev('Tested') + '</div>')
    if lag is None or len(lag) == 0:
        h += '<div class="note" style="font-size:0.78rem;">No industry passes the laggard filter today.</div></div>'
        return h.replace('\n', ' ')
    h += '<table><thead><tr>'
    for lbl, cls in [('Industry', 'l'), ('ETF', 'l'), ('Sector', 'l'), ('Label', 'l'), ('RS 1M', ''),
                     ('RS 3M vs SPY (sort)', 'l'), ('Why on the list', 'l'), ('Weakest holdings', 'l')]:
        h += f'<th class="{cls}">{lbl}</th>'
    h += '</tr></thead><tbody>'
    for _, r in lag.iterrows():
        sc = float(r['Score'])
        lc = ORANGE if sc <= -1.0 else ORANGE_M
        v1 = r.get('RS 1M')
        c1 = DIM if _nan(v1) else (TEAL if float(v1) > 0 else ORANGE if float(v1) < 0 else DIM)
        h += '<tr>'
        h += f'<td class="l"><b>{_e(r["Industry"])}</b></td>'
        h += f'<td class="l" style="color:#a89f95;">{_e(r["ETF"])}</td>'
        h += f'<td class="l" style="color:#a89f95;">{_e(r["Sector"])}</td>'
        h += f'<td class="l" style="color:{lc};font-weight:700;">{sc:+.1f} {_e(r["Label"])}</td>'
        h += f'<td class="num" style="color:{c1};">{_pct(v1)}</td>'
        h += f'<td class="l">{_rs_bar(r.get("RS 3M"))}</td>'
        h += f'<td class="l" style="color:#e8dccb;">{_e(r["Why"])}</td>'
        h += f'<td class="l">{_weak_holdings(hold_df, r["ETF"])}</td>'
        h += '</tr>'
    h += '</tbody></table>'
    h += ('<div class="note" style="font-size:0.80rem;color:#f3ece2;margin-top:6px;"><b>Use: don\'t buy these for a '
          'rebound. Short only with your own setup at a level, with a tight stop.</b></div>')
    h += _how('How this list is built and what is tested',
              'Industry ETFs, held labels. On the list: structural laggard (−1.5) whose RS 1M is not positive, or '
              'underperforming (−0.5) and FADING. Sorted weakest RS 3M first. Weakest holdings = top-10 holdings with RS '
              f'score ≤ −1, by RS 3M. {ev("Tested")} Laggards keep lagging SPY slightly, but shorting them has had no '
              'edge: no better than shorting any industry. Outright shorts lost money on average, most in RISK-OFF '
              '(−2.3% a month; a ≥5% squeeze in 55% of months).')
    h += '</div>'
    return h.replace('\n', ' ')
