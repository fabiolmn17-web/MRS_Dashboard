"""
heat_score.py — Heat score 1-100 for a stock (v0, 4 Nov 2026)
===============================================================
A descriptive composite of the trend and relative-strength states the dashboard already shows. The weights follow the
strength of the evidence we have; the score itself is NOT a tested predictor yet (forward check from Nov 2026).

Component (max points)                          Evidence
------------------------------------------------ ------------------------------------------------------------------
RS rating, cross-sectional (25)                 Literature: cross-sectional stock momentum (Jegadeesh-Titman 12-1).
                                                Not tested in our program on single stocks.
Held RS label vs SPY + trend tag (15)           ETFs (momentum_rs_persistence, 25 Oct 2026; audit v2 8 Nov: Descriptive,
                                                window-sensitive): 3-day hold;
                                                among Outperforming, IMPROVING beat SPY 55% vs FADING 48% (1 mo).
                                                v5.5: the trend tag is now the label's direction (up / down in the
                                                last 21 sessions); the 55 / 48 split was measured on the 3M-vs-6M state.
Industry: group state + leader inside it (20)   Audit v2 (8 Nov 2026): Descriptive. Leaders minus laggards (+0.4 / +0.8 / +1.5 pp at 1/3/6 mo,
                                                TS sign p 0.02). Leader-in-group rank: untested (industry_groups.py).
Sector state (5)                                Sectors showed no reliable momentum (audit v2: near null), so context only.
52-week high proximity (15)                     Literature: George-Hwang (2004) 52-week-high momentum. Untested here.
Trend: 200-day, 50 > 200, SMA50 momentum (20)   Practitioner rules (O'Neil / Minervini trend template). Untested here.
Extension above the SMA50 (0, no penalty)       stretched_label_v0 (28 Oct 2026): in stocks the left tail did NOT grow
                                                with extension, so no penalty.
Market regime (not in the score)                Provisional (audit v2, 8 Nov 2026): in the crash state (SPY 12m down / MRS
                                                RISK-OFF) leaders lose their edge over laggards because laggards rebound.
                                                Shown next to the score as the momentum light, not mixed in.
Audit v2 (8 Nov 2026, studies/_framework/audit/heat_score_backtest): the score describes leadership; top-quartile heat
earned the same as the universe (±0.3 pp a month). No cut-off changes (every neighbour is also ~0).

compute(row, sec_score) -> dict(score, parts=[(name, points, max, text)], missing=[...])
Missing inputs score 0 and are listed; the score is not rescaled (a live lookup without an RS rating reads lower).
"""
import math

import pandas as pd

WEIGHTS = {'rs_rating': 25, 'rs_label': 15, 'industry': 20, 'sector': 5, 'high52': 15, 'trend': 20}
BANDS = [(80, 'Hot', '#85cac5'), (65, 'Warm', '#649995'), (45, 'Neutral', '#9b958f'), (25, 'Cool', '#b9805a'),
         (0, 'Cold', '#f4a062')]


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _true(v):
    return str(v).lower() == 'true'


def band(score):
    if score is None:
        return '', '#6e675f'
    for lo, name, col in BANDS:
        if score >= lo:
            return name, col
    return BANDS[-1][1], BANDS[-1][2]


def compute(r, sec_score=None, yf_to_sector=None, leader_etf=None):
    """r: one scan row (dict / Series). sec_score: display sector -> held sector score.
    leader_etf: ETF on the leaders list holding this stock (fallback when the scan has no industry groups)."""
    g = (lambda k: r.get(k)) if hasattr(r, 'get') else (lambda k: None)
    parts, missing = [], []

    # 1. RS rating (1-99 across the scanned universe)
    rr = _num(g('rs_rating'))
    if rr is None:
        missing.append('RS rating')
        parts.append(('RS rating', 0, WEIGHTS['rs_rating'], 'n/a'))
    else:
        parts.append(('RS rating', round(WEIGHTS['rs_rating'] * rr / 99), WEIGHTS['rs_rating'], f'{rr:.0f}'))

    # 2. held RS label vs SPY + trend tag
    hs = _num(g('held_score'))
    tr = str(g('held_trend') or '').upper()
    if hs is None:
        missing.append('RS label')
        parts.append(('RS label', 0, WEIGHTS['rs_label'], 'n/a'))
    else:
        p = 12 if hs >= 1.5 else 9 if hs >= 1.0 else 6 if hs >= 0.5 else 3 if hs > -0.5 else 0
        p += 3 if tr == 'IMPROVING' else -3 if tr == 'FADING' else 0
        p = max(0, min(WEIGHTS['rs_label'], p))
        parts.append(('RS label', p, WEIGHTS['rs_label'], f'{hs:+.1f} {tr.title()}'.strip()))

    # 3. industry: group state (without the stock) + leader inside the group; ETF fallback for older scans
    if g('ind_group') not in (None, '') and not (isinstance(g('ind_group'), float) and math.isnan(g('ind_group'))):
        isc = _num(g('ind_score'))
        leading = _true(g('ind_leading'))
        st_pts = 10 if leading else (7 if isc is not None and isc >= 1.0 else 4 if isc is not None and isc >= 0.5
                                     else 2 if isc is not None and isc > -0.5 else 0)
        ld_pts = 10 if _true(g('ind_leader')) else 0
        rk, n = _num(g('ind_rank')), _num(g('ind_n'))
        txt = (('leading' if leading else (f'{isc:+.1f}' if isc is not None else 'n/a'))
               + (f' · rank {int(rk)}/{int(n)}' if rk is not None and n else '') + (' · leader' if ld_pts else ''))
        parts.append(('Industry', st_pts + ld_pts, WEIGHTS['industry'], txt))
    elif leader_etf:
        parts.append(('Industry', 10, WEIGHTS['industry'], f'top-10 holding of {leader_etf} (leaders list)'))
    else:
        missing.append('industry groups')
        parts.append(('Industry', 0, WEIGHTS['industry'], 'n/a'))

    # 4. sector held score
    sect = str(g('sector') or '')
    sect = (yf_to_sector or {}).get(sect, sect)
    ss = _num((sec_score or {}).get(sect))
    if ss is None:
        parts.append(('Sector', 0, WEIGHTS['sector'], 'n/a'))
    else:
        parts.append(('Sector', 5 if ss >= 1.0 else 3 if ss >= 0.5 else 1 if ss > -0.5 else 0, WEIGHTS['sector'],
                      f'{sect} {ss:+.1f}'))

    # 5. distance from the 52-week closing high
    p52 = _num(g('pct_from_52w_high'))
    if p52 is None:
        p52 = _num(g('pct_52w'))
    if p52 is None:
        missing.append('52-week high')
        parts.append(('52W high', 0, WEIGHTS['high52'], 'n/a'))
    else:
        parts.append(('52W high', 15 if p52 >= -0.05 else 10 if p52 >= -0.15 else 5 if p52 >= -0.25 else 0,
                      WEIGHTS['high52'], f'{p52 * 100:+.1f}%'))

    # 6. trend: above the 200-day, 50-day above the 200-day, SMA50 momentum zone
    a200, g50 = _true(g('above_sma200')), _true(g('sma50_above_sma200'))
    mz = str(g('mom_zone') or '')
    mp = 6 if mz in ('Persistent', 'Building') else 3 if mz == 'Fading' else 0
    parts.append(('Trend', (7 if a200 else 0) + (7 if g50 else 0) + mp, WEIGHTS['trend'],
                  ' · '.join([('>200' if a200 else '<200'), ('50>200' if g50 else '50<200'), mz or 'n/a'])))

    score = sum(p for _, p, _, _ in parts)
    return {'score': int(max(1, min(100, score))), 'parts': parts, 'missing': missing}


def add_column(df, sec_score=None, yf_to_sector=None, leader_hold=None):
    """Adds heat_score (int) to a scan frame."""
    if df is None or df.empty:
        return df
    lh = leader_hold or {}
    out = df.copy()
    out['heat_score'] = [compute(r, sec_score, yf_to_sector, lh.get(str(r.get('ticker'))))['score']
                         for r in out.to_dict('records')]
    return out


def breakdown_html(res, light_on=None):
    """Small HTML block: the score, its band and the component bars (for the lookup / watchlist side panel)."""
    if not res:
        return ''
    s = res['score']
    name, col = band(s)
    h = (f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;">'
         f'<span style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;font-weight:700;color:{col};">{s}</span>'
         f'<span style="color:{col};font-weight:600;">{name}</span>'
         f'<span style="color:#6e675f;font-size:0.68rem;">heat score / 100 · describes leadership, not a return forecast · descriptive</span></div>')
    if light_on is not None:
        h += (f'<div style="font-size:0.70rem;color:{"#85cac5" if light_on else "#f4a062"};margin-bottom:4px;">'
              f'Momentum light {"ON" if light_on else "CAUTION: leaders lose their edge over laggards (laggards rebound)"}</div>')
    for nm, p, mx, txt in res['parts']:
        w = 0 if not mx else int(round(100 * p / mx))
        h += (f'<div style="display:flex;align-items:center;gap:6px;font-size:0.70rem;line-height:1.5;">'
              f'<span style="width:62px;color:#a89f95;">{nm}</span>'
              f'<span style="flex:0 0 70px;height:5px;background:#2a2724;border-radius:2px;overflow:hidden;">'
              f'<span style="display:block;height:5px;width:{w}%;background:{col};"></span></span>'
              f'<span style="width:38px;font-family:IBM Plex Mono,monospace;color:#f3ece2;">{p}/{mx}</span>'
              f'<span style="color:#9b958f;">{txt}</span></div>')
    if res['missing']:
        h += f'<div style="font-size:0.66rem;color:#6e675f;margin-top:2px;">Not available: {", ".join(res["missing"])}</div>'
    return h
