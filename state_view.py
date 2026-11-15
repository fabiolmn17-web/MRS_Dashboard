"""
state_view.py — the scanner and the leader / laggard lists read through the v8 market state (8.1.0, 15 Nov 2026)
================================================================================================================
One place that turns the v8 state (SPY phase code 0 DIP ZONE / 1 A open / 2 B open / 3 no ticket, and the clock)
into display choices. Colour and chips only; nothing is hidden or locked (Fabio, 15 Nov 2026).

Evidence (survivorship-free unless noted):
  S2 rerun (13 Nov): no setup beats a random uptrend entry in the same SPY state; the 21-day breakout lagged random in
     the rebound (-0.05R, CI -0.10 to -0.00) and was the weakest overall (-0.03R).
  RS1 (15 Nov, setups by state, descriptive): Deep dip +0.27 to +0.30R in DIP / A / B, about 0R with no ticket;
     21 EMA / 50 SMA pullbacks +0.25 to +0.32R in DIP and A.
  Overview / S3 cells: with no ticket open, base and breakout entries lost money while deep dips held up.
  RS1 + S3 rerun: in the DIP ZONE leaders did worst (Q5 +0.19R vs Q1 +0.62R; S3 OFF side Q5-Q1 -0.50R in the dip,
     -0.28R in the recovery); in calm trends no difference (RS1-a failed); on the risk-on side leaders slightly better
     (S3, exploratory).
  C1: the dip window = IN DIP or <= 20 sessions after the last DIP ZONE day.
"""
import html as _html

TEAL, GREY, COPPER, INK, DIM = '#85cac5', '#8a847c', '#e3b48e', '#f3ece2', '#6e675f'
VERDICT_COL = {'works': TEAL, 'no edge': GREY, 'lagged': COPPER}
SCAN_NAME = {'L_dip': 'Deep dip', 'L_pb': 'Pullbacks', 'L_bases': 'Bases', 'L_bo': 'Breakouts', 'L_all': 'Momentum'}

# scan -> {phase code: (verdict, why)}
SCAN_EVIDENCE = {
    'L_dip': {0: ('works', 'Deep dips made +0.27 to +0.30R in the dip zone and with a ticket open (RS1).'),
              1: ('works', 'Deep dips made +0.27 to +0.30R with ticket A open (RS1).'),
              2: ('works', 'Deep dips made +0.27 to +0.30R with ticket B open (RS1).'),
              3: ('no edge', 'With no ticket open, deep dips made about 0R (RS1); they held up better than bases and breakouts.')},
    'L_pb': {0: ('works', 'Pullbacks made +0.25 to +0.32R in the dip zone (RS1).'),
             1: ('works', 'Pullbacks made +0.25 to +0.32R with ticket A open (RS1).'),
             2: ('no edge', 'With ticket B open, pullbacks were no better than a random entry (S2).'),
             3: ('no edge', 'With no ticket open, pullbacks were no better than a random entry (S2).')},
    'L_bases': {0: ('no edge', 'Bases are untested in the dip zone.'),
                1: ('no edge', 'Bases are untested with ticket A open.'),
                2: ('no edge', 'Bases are untested with ticket B open.'),
                3: ('lagged', 'With no ticket open, base entries lost money.')},
    'L_bo': {0: ('lagged', 'Breakouts in the dip zone chase: they lagged a random entry (S2).'),
             1: ('lagged', 'With ticket A open, 21-day breakouts lagged a random entry by 0.05R (S2, survivorship-free).'),
             2: ('no edge', 'With ticket B open, breakouts were about as good as a random entry (S2).'),
             3: ('lagged', 'With no ticket open, breakout entries lost money.')},
}


def crash_window(v8: dict) -> bool:
    """IN DIP or <= 20 sessions after the last DIP ZONE day (C1 clock)."""
    return bool(v8) and v8.get('bucket') in ('IN DIP', 'EARLY')


def scan_verdict(key: str, v8: dict):
    """(verdict, why) for a long scan in today's state, or (None, '') without a state."""
    if not v8 or v8.get('code') is None:
        return None, ''
    if key == 'L_all':
        if crash_window(v8):
            return 'lagged', ('In the dip and the 20 sessions after it, prior leaders did worst (RS1: +0.19R vs +0.62R '
                              'for laggards in the dip zone).')
        return 'no edge', 'Leaders had no fatter right tail in trends (RS1-a failed); momentum describes, it does not select.'
    ev = SCAN_EVIDENCE.get(key, {}).get(int(v8['code']))
    return ev if ev else (None, '')


def rs_mode(v8: dict) -> str:
    """'invert' in the dip window (laggards did better), 'neutral' in a calm trend (no difference), else 'normal'."""
    if not v8 or v8.get('code') is None:
        return 'normal'
    if crash_window(v8):
        return 'invert'
    if v8['code'] in (1, 2) and v8.get('vb') == 'CALM':
        return 'neutral'
    return 'normal'


RS_MODE_TIP = {
    'invert': ('Dip window: leaders did worst and laggards best (RS1 dip zone: +0.19R vs +0.62R). Colours are flipped: '
               'teal = laggard, copper = leader.'),
    'neutral': 'Calm trend: leaders and laggards did the same (RS1-a failed). RS is shown in grey.',
    'normal': 'RS colours as usual (teal = leader). On the risk-on side, leaders did slightly better (S3, exploratory).',
}


def scan_strip_html(v8: dict) -> str:
    """One row of chips: how each long scan did in today's SPY state (hover for the evidence)."""
    if not v8 or v8.get('code') is None:
        return ''
    ph = {0: 'DIP ZONE', 1: 'A open', 2: 'B open', 3: 'no ticket'}[int(v8['code'])]
    h = ('<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:2px 0 6px 0;">'
         f'<span style="font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#a89f95;font-weight:700;">'
         f'In this state · {_html.escape(ph)}{" · dip window" if crash_window(v8) else ""}</span>')
    for k in ('L_dip', 'L_pb', 'L_bases', 'L_bo', 'L_all'):
        vd, why = scan_verdict(k, v8)
        if vd is None:
            continue
        c = VERDICT_COL[vd]
        mark = {'works': '▲', 'no edge': '●', 'lagged': '▼'}[vd]
        h += (f'<span title="{_html.escape(SCAN_NAME[k] + ": " + vd + ". " + why, quote=True)}" style="font-size:0.72rem;'
              f'font-weight:800;padding:2px 8px;border-radius:4px;color:#0a0a0a;background:{c};">{mark} '
              f'{_html.escape(SCAN_NAME[k])}</span>')
    h += (f'<span style="font-size:0.62rem;color:{DIM};">▲ works here · ● no edge · ▼ lagged a random entry · '
          f'hover for the evidence</span></div>')
    return h
