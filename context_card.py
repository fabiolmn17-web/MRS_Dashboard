"""
context_card.py — "Market context" card (MRS v6.3.1, 13 Nov 2026)
===================================================================
One card, one job: WHAT HAS PAID in today's exact state and WHERE TO LOOK. It does not size positions: how much risk
belongs to the pedal (Risk today panel). Three context levels:
  1. Regime side — MRS band: risk-off side (score < -0.5, incl. DEEP) vs neutral / risk-on side.
  2. SPY phase — one track that merges the dip zone (= the pedal's rebound window) and the allocation tickets:
       DIP ZONE  SPY close at least 1 ATR(14) below its 50-day SMA (Ticket A buys the first such close per sell-off)
       RECOVERY  Ticket A (rebound) open, SPY back above the dip zone
       TREND     Ticket B (21-day breakout) open, no rebound ticket
       OUT       no ticket open
     The distance to the 50-day is the pedal's own number when the app passes it in (one number on the page).
  3. The cell (regime side x phase): mean R per trade of the setups and of a random entry in that cell, then the setup
     focus for the phase and the trade management that held in every setup.
Evidence: Setups x market state study S2 (12 Nov 2026, stocks 2009-26, today's list = survivorship-biased), stop
2 ATR(14), 2R then trail the 21 EMA. The phases and cells were drawn AFTER S2: exploratory, not tested rules.
"""
import html as _html

TEAL, TEAL_M, ORANGE, COPPER, INK, MUTED, DIM = '#85cac5', '#649995', '#f4a062', '#e3b48e', '#f3ece2', '#a89f95', '#6e675f'
PHASES = ['DIP ZONE', 'RECOVERY', 'TREND', 'OUT']
CHASE_PHASES = ('DIP ZONE', 'RECOVERY')          # breakouts lag in these phases ("chasing" tag on the scanner)

# S2 cross-tab (exploratory): mean R per trade, stop 2 ATR(14), 2R then trail the 21 EMA.
# dd = Deep dip, p50 = 50-day pullback, bo = 21-day breakout, rnd = random uptrend entry; n_dd = Deep dip trades.
CELL = {
    ('OFF', 'DIP ZONE'): dict(dd=0.38, p50=0.42, bo=0.25, rnd=0.41, n_dd=10091),
    ('ON', 'DIP ZONE'): dict(dd=0.02, p50=-0.01, bo=0.27, rnd=0.27, n_dd=476),
    ('OFF', 'RECOVERY'): dict(dd=0.37, p50=0.38, bo=0.30, rnd=0.39, n_dd=4037),
    ('ON', 'RECOVERY'): dict(dd=0.15, p50=0.09, bo=0.08, rnd=0.12, n_dd=5803),
    ('OFF', 'TREND'): dict(dd=0.44, p50=0.32, bo=0.26, rnd=0.19, n_dd=1370),
    ('ON', 'TREND'): dict(dd=0.26, p50=0.19, bo=0.17, rnd=0.21, n_dd=3319),
    ('OFF', 'OUT'): dict(dd=0.13, p50=0.17, bo=0.10, rnd=0.17, n_dd=2803),
    ('ON', 'OUT'): dict(dd=0.15, p50=0.11, bo=0.18, rnd=0.18, n_dd=1526),
}
BEST = ('OFF', 'DIP ZONE')
SMALL_N = 1000


def _e(s):
    return _html.escape(str(s), quote=True)


def _sg(x, d=1):
    return f'{x:+.{d}f}'.replace('-', '−')


def side(band: str) -> str:
    return 'OFF' if 'OFF' in (band or '').upper() else 'ON'


def phase(s: dict, ext=None):
    """Phase of SPY from allocation.state(); ext = the pedal's distance to the 50-day in ATR (preferred when given,
    so the page shows one number). Returns (phase, one-line detail) or (None, '') without data."""
    if not s:
        return None, ''
    E = ext if ext is not None and ext == ext else s.get('E')
    A, B = s.get('A', {}), s.get('B', {})
    sma, c = s.get('sma50'), s.get('close')
    if E is not None and E <= -1:
        if A.get('open'):
            tk = 'Ticket A open'
        elif s.get('in_selloff'):
            why = ('this sell-off already used its rebound entry' if s.get('ep_taken') else
                   'this sell-off began while the previous rebound trade was open')
            tk = f'Ticket A locked: {why}; it re-arms after a close above the 50-day ({sma:.2f})'
        else:
            tk = 'Ticket A buys at this close'
        return 'DIP ZONE', f'SPY {_sg(E, 2)} ATR from its 50-day (the pedal\'s rebound window). {tk}.'
    if A.get('open'):
        return 'RECOVERY', (f'SPY {_sg(E, 2)} ATR from its 50-day. Ticket A (rebound) open since {A["entry_date"]:%d %b}; '
                            f'exit on 2 closes below the 50-day ({A["ma"]:.2f}).')
    if B.get('open'):
        return 'TREND', (f'SPY {_sg(E, 2)} ATR from its 50-day. Ticket B (21-day breakout) open since '
                         f'{B["entry_date"]:%d %b}; exit on 2 closes below the 21 EMA ({B["ma"]:.2f}).')
    nxt = [f'breakout buy above {s["bo_level"]:.2f} ({_sg((s["bo_level"] / c - 1) * 100)}%)']
    if s.get('in_selloff'):
        nxt.insert(0, f'rebound re-arms after a close above the 50-day {sma:.2f} ({_sg((sma / c - 1) * 100)}%)')
    else:
        nxt.insert(0, f'dip zone below about {s["rb_level"]:.2f} ({_sg((s["rb_level"] / c - 1) * 100)}%)')
    return 'OUT', f'SPY {_sg(E, 2)} ATR from its 50-day, no ticket open. Next: ' + '; '.join(nxt) + '.'


def cell_line(band: str, deep: str, ph: str):
    """(text, colour) for the regime x phase cell."""
    sd = side(band)
    c = CELL.get((sd, ph))
    if c is None:
        return 'SPY phase not available right now.', MUTED
    lo, hi = min(c['dd'], c['p50'], c['bo']), max(c['dd'], c['p50'], c['bo'])
    who = 'Risk-off side' if sd == 'OFF' else 'Neutral / risk-on side'
    txt = (f'{who} + {ph}: longs averaged {_sg(lo, 2)} to {_sg(hi, 2)}R per trade (Deep dip {_sg(c["dd"], 2)}, '
           f'50-day pullback {_sg(c["p50"], 2)}, breakout {_sg(c["bo"], 2)}; random entry {_sg(c["rnd"], 2)}).')
    if c['n_dd'] < SMALL_N:
        txt += f' Few cases on this side ({c["n_dd"]} Deep dips): treat as anecdotal.'
    if (sd, ph) != BEST:
        b = CELL[BEST]
        txt += (f' For scale: the strongest cell is risk-off side + dip zone ({_sg(min(b["dd"], b["p50"]), 2)} to '
                f'{_sg(max(b["dd"], b["p50"]), 2)}R).')
    if deep == 'DEEP RISK-OFF':
        txt += ' In DEEP RISK-OFF every setup did worse than a random entry.'
    col = TEAL if hi >= 0.30 else COPPER
    return txt, col


FOCUS = {
    'DIP ZONE': 'Deep dips and reclaims of the 50-day. Skip breakouts: they lagged in this phase.',
    'RECOVERY': 'Deep dips and pullbacks first; breakouts lag.',
    'TREND': 'Deep dips did best; pullbacks and breakouts about random.',
    'OUT': 'No setup beat a random entry here: only the strongest names.',
}
RISK = ('Stop 2 ATR(14), at most 10% below the entry (breakouts: under the 5-day low). Take 2R, then trail the '
        '21 EMA (first close below it). How much risk: see the pedal below.')
PICK = 'Choose the name by leadership (sector, industry, RS); the setup only sets the entry and the stop.'
EVIDENCE = ('S2, 12 Nov 2026: 210K stock trades 2009-26 (today\'s list: survivorship-biased, which flatters buying in '
            'sell-offs most), stop 2 ATR(14), 2R then trail the 21 EMA, mean R per trade. Cells = MRS side (risk-off: '
            'score < -0.5) x SPY phase; phases and cells drawn after S2: exploratory. Tested in S2: no setup beat a random '
            'uptrend entry in the same SPY allocation state; breakouts were below random while the rebound ticket was '
            'open (-0.06R). Stops / exits: 1.5 ATR cost about 0.05R; 2R then trail beat a plain 2R by +0.03 to +0.08R.')

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
LABEL = {'DIP ZONE': 'DIP ZONE · rebound window'}


def card_html(band: str, deep: str, spy_state: dict, ext=None) -> str:
    ph, pdet = phase(spy_state, ext)
    track = ''
    for i, p in enumerate(PHASES):
        on = p == ph
        col = (TEAL if p != 'OUT' else COPPER) if on else None
        style = f' style="color:#0a0a0a;background:{col};border-color:{col};"' if on else ''
        track += (f'<span class="st"{style}>{LABEL.get(p, p)}</span>'
                  + ('<span class="ar">→</span>' if i < len(PHASES) - 1 else ''))
    ctxt, ccol = cell_line(band, deep, ph) if ph else ('SPY phase not available right now.', MUTED)
    tag = (deep if deep in ('DEEP RISK-OFF', 'DEEP RISK-ON') else (band or '—')).upper()
    ftxt = FOCUS.get(ph, '')
    return (CSS + f'<div class="cx" title="{_e(EVIDENCE)}"><div class="k">Market context · what has paid in this state '
            f'and where to look · provisional (hover for the evidence)</div>'
            f'<div class="row"><div class="t">SPY phase</div><div><div class="trk">{track}</div>'
            f'<div class="sub">{_e(pdet)}</div></div></div>'
            f'<div class="row"><div class="t">This state</div><div class="tx"><b style="color:{ccol};">{_e(tag)} · '
            f'{_e(ph or "—")}</b> — {_e(ctxt)}</div></div>'
            f'<div class="row"><div class="t">Where to look</div><div class="tx">{_e(ftxt)}'
            f'<div class="sub" style="margin-top:2px;">{_e(PICK)}</div></div></div>'
            f'<div class="row"><div class="t">Trade management</div><div class="tx">{_e(RISK)}</div></div>'
            f'<div class="sub" style="border-top:1px solid #24211e;padding-top:4px;">The trade stays discretionary. '
            f'Cells and phases are in-sample and exploratory (stocks 2009-26, survivorship-biased); logged forward from '
            f'13 Nov 2026.</div></div>').replace('\n', ' ')


def scanner_hint(band: str, deep: str, spy_state: dict, ext=None):
    """(colour, text) for the line above the scan pills, or None."""
    ph, _ = phase(spy_state, ext)
    if ph is None:
        return None
    _, ccol = cell_line(band, deep, ph)
    tag = (deep if deep in ('DEEP RISK-OFF', 'DEEP RISK-ON') else (band or '')).upper()
    return ccol, (f'{tag} · SPY {ph} — {FOCUS[ph]} Stop 2 ATR(14), max 10%; take 2R, then '
                  f'trail the 21 EMA. Grey setup = stop wider than 10%.')


def chasing(spy_state: dict, ext=None) -> bool:
    return phase(spy_state, ext)[0] in CHASE_PHASES
