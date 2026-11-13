"""
context_card.py — "Market context" card (MRS v6.3, 13 Nov 2026)
================================================================
One card that unifies the three context levels, in this order:
  1. Regime   — MRS band (and the DEEP RISK-OFF flag): how much risk, how active.
  2. SPY phase — one track that merges the rebound window and the allocation tickets:
       DIP ZONE  SPY close at least 1 ATR(14) below its 50-day SMA (the rebound window; Ticket A buys here)
       RECOVERY  Ticket A (rebound) open, SPY back above the dip zone
       TREND     Ticket B (21-day breakout) open, no rebound ticket
       OUT       no ticket open
  3. Setup focus — what has worked in that phase, plus the trade management that held in every setup.
Evidence: Setups x market state study S2 (12 Nov 2026, stocks 2009-26, today's list = survivorship-biased) and the
Deep dip stop-width study S1. The four phases were defined AFTER S2 (exploratory): the card guides, it is not a tested
rule. Numbers: mean R per trade, stop 2 ATR(14), target 2R then trail the 21 EMA, unless stated.
"""
import html as _html

TEAL, TEAL_M, ORANGE, COPPER, INK, MUTED, DIM = '#85cac5', '#649995', '#f4a062', '#e3b48e', '#f3ece2', '#a89f95', '#6e675f'
PHASES = ['DIP ZONE', 'RECOVERY', 'TREND', 'OUT']
CHASE_PHASES = ('DIP ZONE', 'RECOVERY')          # breakouts lag in these phases ("chasing" tag on the scanner)


def _e(s):
    return _html.escape(str(s), quote=True)


def _sg(x):
    return f'{x:+.1f}'.replace('-', '−')


def phase(s: dict):
    """Phase of SPY from allocation.state(). Returns (phase, one-line detail) or (None, '') without data."""
    if not s:
        return None, ''
    E, A, B = s.get('E'), s.get('A', {}), s.get('B', {})
    sma, c = s.get('sma50'), s.get('close')
    if E is not None and E <= -1:
        if A.get('open'):
            tk = 'Ticket A open'
        elif s.get('in_selloff') and not A.get('open'):
            why = ('this sell-off already used its rebound entry' if s.get('ep_taken') else
                   'this sell-off began while the previous rebound trade was open')
            tk = f'Ticket A locked: {why}; it re-arms after a close above the 50-day ({sma:.2f})'
        else:
            tk = 'Ticket A buys at this close'
        return 'DIP ZONE', f'SPY {_sg(E)} ATR from its 50-day (rebound window). {tk}.'
    if A.get('open'):
        return 'RECOVERY', (f'Ticket A (rebound) open since {A["entry_date"]:%d %b}; exit on 2 closes below the 50-day '
                            f'({A["ma"]:.2f}).')
    if B.get('open'):
        return 'TREND', (f'Ticket B (21-day breakout) open since {B["entry_date"]:%d %b}; exit on 2 closes below the '
                         f'21 EMA ({B["ma"]:.2f}).')
    nxt = [f'breakout buy above {s["bo_level"]:.2f} ({_sg((s["bo_level"] / c - 1) * 100)}%)']
    if s.get('in_selloff'):
        nxt.insert(0, f'rebound re-arms after a close above the 50-day {sma:.2f} ({_sg((sma / c - 1) * 100)}%)')
    else:
        nxt.insert(0, f'dip zone below about {s["rb_level"]:.2f} ({_sg((s["rb_level"] / c - 1) * 100)}%)')
    return 'OUT', f'SPY {_sg(E)} ATR from its 50-day, no ticket open. Next: ' + '; '.join(nxt) + '.'


def regime(band: str, deep: str = ''):
    """(level text, colour, short tag) for the MRS band."""
    b = (band or '').upper()
    if deep == 'DEEP RISK-OFF':
        return ('Size down. Payoffs for longs are high here, but swings are wide and every setup did worse than a '
                'random entry (+0.20 to +0.26R against +0.28R). Cut risk, don\'t sell on it.', ORANGE, 'DEEP RISK-OFF')
    if 'OFF' in b:
        return ('Be active. Longs started on the risk-off side paid about twice what they paid in Neutral / Risk-On '
                '(any uptrend stock, 2R target: +0.26 to +0.27R against +0.11 to +0.13R per trade).', TEAL, b)
    return ('Be selective, normal size. Longs paid about half the risk-off result here (+0.11 to +0.13R per trade, '
            '2R target) and big runs were rarer.', COPPER, b or '—')


FOCUS = {
    'DIP ZONE': ('Deep dips and reclaims of the 50-day. Skip breakouts: they lagged (+0.25R against +0.37 to +0.40R '
                 'for Deep dips, 50-day pullbacks and random entries).', TEAL),
    'RECOVERY': ('Deep dips and pullbacks first (+0.18 to +0.24R). Breakouts lag (+0.14R, below a random entry).', TEAL),
    'TREND': ('Deep dips did best (+0.32R against +0.21R for a random entry, the only setup above random). Pullbacks '
              'and breakouts about random (+0.19 to +0.22R).', TEAL),
    'OUT': ('No setup has an edge here (all about +0.14R, at or below a random entry). Only the strongest names, '
            'smaller size.', COPPER),
}
RISK = ('Stop 2 ATR(14), at most 10% below the entry (breakouts: under the 5-day low). Take 2R, then trail the '
        '21 EMA (first close below it).')
PICK = 'Choose the name by leadership (sector, industry, RS); the setup only sets the entry and the stop.'
EVIDENCE = ('S2, 12 Nov 2026: 210K stock trades 2009-26 (today\'s list: survivorship-biased), stop 2 ATR(14), 2R then '
            'trail the 21 EMA, mean R per trade. DIP ZONE: Deep dip +0.37, 50-day pullback +0.40, 21 EMA pullback +0.34, '
            '21-day breakout +0.25, random +0.40. RECOVERY: +0.24 / +0.21 / +0.18 / +0.14, random +0.19. TREND: +0.32 / '
            '+0.22 / +0.21 / +0.19, random +0.21. OUT: +0.14 / +0.15 / +0.14 / +0.14, random +0.18. Regime (2R target, '
            'random uptrend entries): MILD RISK-OFF +0.26, RISK-OFF +0.27, NEUTRAL +0.11, RISK-ON +0.13, DEEP +0.28 '
            '(setups +0.20 to +0.26). Tested in S2: no setup beat random in the same SPY state; breakouts were below '
            'random while the rebound ticket was open (-0.06R). The four phases were drawn after S2: exploratory. '
            'Stops and exits: S1 / S2 (1.5 ATR costs about 0.05R; 2R then trail beat a plain 2R by +0.03 to +0.08R).')

CSS = ('<style>.cx{background:#121212;border:1px solid #2a2724;border-left:3px solid #e3b48e;border-radius:6px;'
       'padding:8px 12px 8px 12px;margin:2px 0 12px 0;}'
       '.cx .k{font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;color:#e3b48e;font-weight:700;margin-bottom:6px;}'
       '.cx .row{display:grid;grid-template-columns:150px 1fr;gap:12px;align-items:start;border-top:1px solid #24211e;padding:6px 0;}'
       '.cx .t{font-size:0.68rem;letter-spacing:0.10em;text-transform:uppercase;color:#a89f95;font-weight:700;padding-top:2px;}'
       '.cx .tx{font-size:0.82rem;color:#f3ece2;line-height:1.42;}'
       '.cx .sub{font-size:0.72rem;color:#a89f95;line-height:1.35;}'
       '.cx .trk{display:flex;gap:4px;margin:0 0 4px 0;flex-wrap:wrap;}'
       '.cx .st{font-size:0.70rem;font-weight:700;letter-spacing:0.06em;padding:2px 8px;border-radius:3px;'
       'border:1px solid #2a2724;color:#6e675f;background:#0d0d0d;}'
       '.cx .ar{color:#3a3632;font-size:0.8rem;padding-top:1px;}'
       '@media (max-width:900px){.cx .row{grid-template-columns:1fr;}}</style>')


def card_html(band: str, deep: str, spy_state: dict) -> str:
    rtxt, rcol, rtag = regime(band, deep)
    ph, pdet = phase(spy_state)
    track = ''
    for i, p in enumerate(PHASES):
        on = p == ph
        col = (TEAL if p != 'OUT' else COPPER) if on else None
        style = f' style="color:#0a0a0a;background:{col};border-color:{col};"' if on else ''
        track += f'<span class="st"{style}>{p}</span>' + ('<span class="ar">→</span>' if i < len(PHASES) - 1 else '')
    ftxt, fcol = FOCUS.get(ph, ('SPY phase not available right now.', MUTED))
    return (CSS + f'<div class="cx" title="{_e(EVIDENCE)}"><div class="k">Market context · regime → SPY phase → setups '
            f'· provisional (hover for the numbers)</div>'
            f'<div class="row"><div class="t">1 · Regime</div><div class="tx"><b style="color:{rcol};">{_e(rtag)}</b> — '
            f'{_e(rtxt)}</div></div>'
            f'<div class="row"><div class="t">2 · SPY phase</div><div><div class="trk">{track}</div>'
            f'<div class="sub">{_e(pdet)}</div></div></div>'
            f'<div class="row"><div class="t">3 · Setups</div><div class="tx"><b style="color:{fcol};">{_e(ph or "—")}:</b> '
            f'{_e(ftxt)}<div class="sub" style="margin-top:2px;">{_e(PICK)}</div></div></div>'
            f'<div class="row"><div class="t">Risk</div><div class="tx">{_e(RISK)}</div></div>'
            f'<div class="sub" style="border-top:1px solid #24211e;padding-top:4px;">The trade stays discretionary: this '
            f'card says when longs have paid and where to look first. Phases and figures are in-sample and exploratory '
            f'(stocks 2009-26, survivorship-biased); logged forward from 13 Nov 2026.</div></div>').replace('\n', ' ')


def scanner_hint(band: str, deep: str, spy_state: dict):
    """(colour, text) for the line above the scan pills, or None."""
    ph, _ = phase(spy_state)
    if ph is None:
        return None
    _, rcol, rtag = regime(band, deep)
    short = {'DIP ZONE': 'Deep dips and 50-day reclaims first; breakouts are chasing.',
             'RECOVERY': 'Deep dips and pullbacks first; breakouts lag.',
             'TREND': 'Deep dips did best; pullbacks and breakouts about random.',
             'OUT': 'No setup edge: only the strongest names, smaller size.'}[ph]
    return rcol, (f'{rtag} · SPY {ph} — {short} Stop 2 ATR(14), max 10%; take 2R, then trail the 21 EMA. '
                  f'Grey setup = stop wider than 10%.')


def chasing(spy_state: dict) -> bool:
    return phase(spy_state)[0] in CHASE_PHASES
