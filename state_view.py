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
  8.5.0 (16 Nov 2026), out of sample 2001-08 on the 2000 panel:
     S2: no setup beat a random entry in any state; deep dips and 50-day pullbacks lagged it in the rebound.
     RS1-b: in the DIP ZONE, leaders lagged the bottom two RS fifths by -0.31R (tested, both sub-eras);
     over the whole dip window (+20 sessions) the gap was noise (-0.09R). So the RS flip is shown in the DIP ZONE only.
"""
import html as _html

TEAL, GREY, COPPER, INK, DIM = '#85cac5', '#8a847c', '#e3b48e', '#f3ece2', '#6e675f'
VERDICT_COL = {'works': TEAL, 'no edge': GREY, 'lagged': COPPER, 'untested': DIM}
VERDICT_WORD = {'works': 'the state paid', 'no edge': 'about flat here', 'lagged': 'lagged', 'untested': 'untested'}
VERDICT_MARK = {'works': '▲', 'no edge': '●', 'lagged': '▼', 'untested': '○'}
SCAN_NAME = {'L_dip': 'Deep dip', 'L_pb': 'Pullbacks', 'L_bases': 'Bases', 'L_bo': 'Breakouts', 'L_all': 'Momentum'}

# 8.5.1 (16 Nov 2026): the verdict says whether the STATE paid for this scan's setups, not whether the pattern beats a
# random entry. The setups frame entry, stop and target; their return came from the state (S2, 2001-26: about the same
# as a random uptrend entry in the same state). 'lagged' = worse than a random entry, or than the alternative.
FRAME = 'Use it to frame entry, stop and target. The return came from the state: about the same as a random uptrend entry here.'
# scan -> {phase code: (verdict, why)}
SCAN_EVIDENCE = {
    'L_dip': {0: ('works', 'DIP ZONE with SPY above a rising 200-day: deep dips made money (+0.27R 2009-26; +0.09R 2001-08).'),
              1: ('works', 'Ticket A open: deep dips made +0.27 to +0.30R in 2009-26; thinner in 2001-08 (random entries +0.06R).'),
              2: ('works', 'Ticket B open: deep dips made +0.27 to +0.30R in 2009-26; random entries +0.28R in 2001-08.'),
              3: ('no edge', 'No ticket open: deep dips made about 0R (RS1); they held up better than bases and breakouts.')},
    'L_pb': {0: ('works', 'DIP ZONE with SPY above a rising 200-day: pullbacks made money (+0.25 to +0.32R 2009-26). 50-day pullbacks lagged a random entry in the 2001-08 rebounds (-0.10R).'),
             1: ('works', 'Ticket A open: pullbacks made +0.25 to +0.32R in 2009-26; thinner in 2001-08 (random entries +0.06R).'),
             2: ('works', 'Ticket B open: random uptrend entries made money in both periods (+0.12 to +0.28R); pullbacks about the same.'),
             3: ('no edge', 'No ticket open: about flat, no better than a random entry (S2).')},
    'L_bases': {0: ('untested', 'Bases are untested in the dip zone.'),
                1: ('untested', 'Bases are untested with ticket A open.'),
                2: ('untested', 'Bases are untested with ticket B open.'),
                3: ('lagged', 'No ticket open: base entries lost money.')},
    'L_bo': {0: ('lagged', 'Breakouts in the dip zone chase: they lagged a random entry (S2).'),
             1: ('lagged', 'Ticket A open: 21-day breakouts lagged a random entry by 0.05R in 2009-26 (about equal in 2001-08).'),
             2: ('works', 'Ticket B open: breakouts made about what a random entry made, and that was positive in both periods.'),
             3: ('lagged', 'No ticket open: breakout entries lost money.')},
}
DIP_OFF = ('no edge', 'DIP ZONE without a rising 200-day: dips kept falling (random entries -0.32R in 2001-08). '
           'Small budget; frame only the cleanest structures.')


def crash_window(v8: dict) -> bool:
    """IN DIP or <= 20 sessions after the last DIP ZONE day (C1 clock)."""
    return bool(v8) and v8.get('bucket') in ('IN DIP', 'EARLY')


def scan_verdict(key: str, v8: dict):
    """(verdict, why) for a long scan in today's state, or (None, '') without a state."""
    if not v8 or v8.get('code') is None:
        return None, ''
    if key == 'L_all':
        if v8.get('bucket') == 'IN DIP':
            return 'lagged', ('In the DIP ZONE, prior leaders lagged the bottom 40% by 6-month return: -0.31R in 2001-08 '
                              '(tested), same direction in 2009-26 (RS1-b).')
        if crash_window(v8):
            return 'no edge', 'After the DIP ZONE, leaders and laggards did about the same (RS1-b out of sample).'
        return 'no edge', 'Leaders had no fatter right tail in trends (RS1-a failed); momentum describes, it does not select.'
    code = int(v8['code'])
    f = v8.get('f200')
    fs = f.get('state') if isinstance(f, dict) else f
    if code == 0 and key in ('L_dip', 'L_pb') and fs != 'F-UP':
        return DIP_OFF
    ev = SCAN_EVIDENCE.get(key, {}).get(code)
    return ev if ev else (None, '')


def rs_mode(v8: dict) -> str:
    """'invert' in the DIP ZONE (laggards did better; 8.5.0: zone only), 'neutral' in a calm trend, else 'normal'."""
    if not v8 or v8.get('code') is None:
        return 'normal'
    if v8.get('bucket') == 'IN DIP':
        return 'invert'
    if v8['code'] in (1, 2) and v8.get('vb') == 'CALM':
        return 'neutral'
    return 'normal'


RS_MODE_TIP = {
    'invert': ('DIP ZONE: leaders lagged the beaten-down names (RS1-b: -0.31R, tested 2001-08; same direction 2009-26). '
               'Colours are flipped: teal = laggard, copper = leader.'),
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
        mark = VERDICT_MARK[vd]
        tip = SCAN_NAME[k] + ': ' + VERDICT_WORD[vd] + '. ' + why + ('' if k == 'L_all' else ' ' + FRAME)
        h += (f'<span title="{_html.escape(tip, quote=True)}" style="font-size:0.72rem;'
              f'font-weight:800;padding:2px 8px;border-radius:4px;color:#0a0a0a;background:{c};">{mark} '
              f'{_html.escape(SCAN_NAME[k])}</span>')
    h += (f'<span style="font-size:0.62rem;color:{DIM};">▲ the state paid · ● about flat · ▼ lagged · ○ untested · '
          f'the setup frames the trade, the state carries it · hover for the evidence</span></div>')
    return h
