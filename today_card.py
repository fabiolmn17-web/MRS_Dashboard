"""
today_card.py — the Today card (MRS v6.4, 14 Nov 2026)
=======================================================
One card at the top of the page answers what can be done today: the market state, how much to risk, where to look
and how to exit. It replaces the decision strip, the Market context card, the pedal tile and the allocation card on
the main path; their detail sits in the "Market detail" fold. Roadmap: "MRS roadmap: from the Nov 2026 tests to the
dashboard" (Claude Doc), steps 1, 3 and 4; page inventory (Step 0).

Evidence:
  State cells  survivorship-free S3 / S2 (panel B, delisted names included, stocks 2009-26): context_card.CELL.
  Sizing       P2 portfolio simulation (14 Nov 2026): 1% on the risk-off side with SPY in DIP ZONE / RECOVERY / TREND,
               0.7% on the risk-on side with SPY in TREND, 0.5% elsewhere; risk counted to the current stop ("earned
               exposure"); open-risk cap 3.5% of NLV (worst 5 days about -11% vs -16% uncapped); at most 10 positions.
               Provisional: +0.3 points a year over flat sizing at book level.
  Hard stop    Fabio's rule (14 Nov 2026): -15% from the account peak. P2 addendum (exploratory): at P2 risk it fires
               about every 2.5 years and does not cap the loss from the peak; the drawdown is set by size.
  Exit         S1 / S2 / P2: stop 2 ATR(14), at most 10%; hold the initial stop until +2R, then trail the 21 EMA.
"""
import html as _html

import context_card
import market_state

TEAL, TEAL_M, ORANGE, COPPER, INK, MUTED, DIM = '#85cac5', '#649995', '#f4a062', '#e3b48e', '#f3ece2', '#a89f95', '#6e675f'
STATE_RISK = {('OFF', 'DIP ZONE'): 0.010, ('OFF', 'RECOVERY'): 0.010, ('OFF', 'TREND'): 0.010, ('ON', 'TREND'): 0.007}
BASE_RISK = 0.005
OPEN_RISK_CAP, MAX_POS, HARD_STOP = 0.035, 10, -0.15
EDGE = (('OFF', 'DIP ZONE'), ('OFF', 'RECOVERY'), ('OFF', 'TREND'))


def _e(s):
    return _html.escape(str(s), quote=True)


def risk_for(band: str, ph):
    """Risk per new trade (fraction of NLV) for the MRS side x SPY phase cell; BASE_RISK when the phase is unknown."""
    return STATE_RISK.get((context_card.side(band), ph), BASE_RISK)


CSS = ('<style>.td{background:#121212;border:1px solid #2a2724;border-left:3px solid #e3b48e;border-radius:6px;'
       'padding:10px 14px 8px 14px;margin:2px 0 10px 0;}'
       '.td .hd{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 22px;padding-bottom:8px;border-bottom:1px solid #2a2724;}'
       '.td .k{font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;color:#e3b48e;font-weight:700;}'
       '.td .big{font-family:"IBM Plex Mono",monospace;font-size:1.9rem;font-weight:600;line-height:1;}'
       '.td .reg{font-size:1.30rem;font-weight:800;line-height:1.05;}'
       '.td .lv{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:0.74rem;color:#a89f95;text-align:right;}'
       '.td .row{display:grid;grid-template-columns:130px 1fr auto;gap:12px;align-items:start;border-top:1px solid #24211e;padding:6px 0;}'
       '.td .row:first-of-type{border-top:none;}'
       '.td .t{font-size:0.68rem;letter-spacing:0.10em;text-transform:uppercase;color:#a89f95;font-weight:700;padding-top:2px;}'
       '.td .tx{font-size:0.84rem;color:#f3ece2;line-height:1.42;}'
       '.td .sub{font-size:0.72rem;color:#a89f95;line-height:1.35;}'
       '.td .trk{display:flex;gap:4px;margin:0 0 4px 0;flex-wrap:wrap;}'
       '.td .st{font-size:0.70rem;font-weight:700;letter-spacing:0.06em;padding:2px 8px;border-radius:3px;'
       'border:1px solid #2a2724;color:#6e675f;background:#0d0d0d;}'
       '.td .ar{color:#3a3632;font-size:0.8rem;padding-top:1px;}'
       '.td .pill{font-size:0.66rem;font-weight:800;letter-spacing:0.08em;padding:1px 7px;border-radius:3px;color:#0a0a0a;}'
       '.td .al{font-size:0.80rem;color:#f3ece2;border-left:3px solid #f4a062;background:#1d1510;padding:3px 8px;margin:4px 0 0 0;}'
       '.td .ft{font-size:0.68rem;color:#6e675f;margin-top:6px;}'
       '@media (max-width:900px){.td .row{grid-template-columns:1fr;}.td .lv{margin-left:0;text-align:left;}}</style>')


def _row(label, body, tag=''):
    return f'<div class="row"><div class="t">{label}</div><div class="tx">{body}</div><div>{tag}</div></div>'


def card_html(mrs: float, band: str, band_col: str, pct_txt: str, deep: str, deep_cut: float, spy_state: dict,
              ext=None, snap: dict = None, since: str = '', levels: dict = None, fresh: str = '', date_txt: str = '') -> str:
    ph, pdet = context_card.phase(spy_state, ext)
    sd = context_card.side(band)
    edge = (sd, ph) in EDGE
    risk = risk_for(band, ph)
    ev = market_state.ev

    # header: score, band (+ DEEP badge), percentile, levels
    lv = ' · '.join(f'{_e(k)} {_e(v)}' for k, v in (levels or {}).items())
    h = (market_state.CSS + CSS + '<div class="td"><div class="hd">'
         f'<span class="k">Today{(" · " + _e(date_txt)) if date_txt else ""}</span>'
         f'<span class="big" style="color:{band_col};">{mrs:+.2f}</span>'
         f'<span class="reg" style="color:{band_col};">{_e(band)}{market_state._deep_badge(deep, deep_cut)}</span>'
         f'<span class="sub">{_e(pct_txt)} · {"risk-off side" if sd == "OFF" else "neutral / risk-on side"}</span>'
         f'<span class="lv">{lv}</span></div>')

    # SPY phase
    track = ''
    for i, p in enumerate(context_card.PHASES):
        on = p == ph
        col = (TEAL if p != 'OUT' else COPPER) if on else None
        style = f' style="color:#0a0a0a;background:{col};border-color:{col};"' if on else ''
        track += (f'<span class="st"{style}>{context_card.LABEL.get(p, p)}</span>'
                  + ('<span class="ar">→</span>' if i < len(context_card.PHASES) - 1 else ''))
    h += _row('SPY phase', f'<div class="trk">{track}</div><div class="sub">{_e(pdet) if ph else "SPY phase not available right now."}</div>')

    # this state
    if ph:
        ctxt, ccol = context_card.cell_line(band, deep, ph)
        pill = (f'<span class="pill" style="background:{TEAL};">EDGE</span>' if edge else
                f'<span class="pill" style="background:{COPPER};">NO EDGE</span>')
        h += _row('This state', f'{pill} <b style="color:{ccol};">{_e(band.upper())} · {_e(ph)}</b> — {_e(ctxt)}',
                  ev('Tested') if edge else ev('Descriptive'))

    # risk per trade
    rcol = TEAL if risk >= 0.01 else (INK if risk >= 0.007 else MUTED)
    h += _row('Risk per trade',
              f'<b style="color:{rcol};font-size:0.95rem;">{risk * 100:.1f}% of NLV</b> per new trade'
              f'{"" if ph else " (SPY phase unknown: base risk)"}. Open risk at most {OPEN_RISK_CAP * 100:.1f}% of NLV, '
              f'counted to the <i>current</i> stops (a trade at break-even uses none); at most {MAX_POS} positions, '
              f'gross at most 100%.<div class="sub">1% on the risk-off side with SPY in DIP ZONE, RECOVERY or TREND; '
              f'0.7% on the risk-on side with SPY in TREND; 0.5% elsewhere. P2 simulation: the cap held the worst week '
              f'to about −11% (−16% uncapped).</div>', ev('Provisional'))

    # hard stop
    h += _row('Hard stop',
              f'<b>−{abs(HARD_STOP) * 100:.0f}% from the account peak:</b> close, stop new trades, review the last 20 trades '
              f'before restarting.<div class="sub">At this risk the stop fires about every 2.5 years (P2 addendum, '
              f'exploratory) and does not cap the loss from the peak: the size of the drawdown is set by the risk per '
              f'trade and the cap.</div>', '<span class="sub" style="color:#e3b48e;">YOUR RULE</span>')

    # where to look
    if ph:
        nm = ('Risk-off side: weaker names (lower RS, further below the high) have done better than leaders.'
              if sd == 'OFF' else 'Risk-on side: stronger names have done better.')
        h += _row('Where to look', f'{_e(context_card.FOCUS.get(ph, ""))}<div class="sub">{_e(nm)} Name choice adds '
                  f'little (about +0.08R for the best fifth): the state decides most of the outcome.</div>',
                  ev('Descriptive'))

    # exit
    h += _row('Exit', 'Stop 2 ATR(14), at most 10% below the entry (breakouts: under the 5-day low). Hold the initial '
              'stop until +2R, then trail the 21 EMA (first close below it).'
              '<div class="sub">Moving the stop to −0.5R at +1R cost about 0.02R per trade (P2). The edge sits in the '
              '7–8% of trades that run to 3R or more.</div>', ev('Tested'))

    # alerts, only when active
    al = []
    if snap and snap.get('pedal') == 'EASE OFF':
        al.append('<b>EASE OFF:</b> RISK-OFF and SPY below its close of a year ago. Fast drops of 7%+ came in 1 of 3 '
                  'cases (all days 1 in 8): fewer positions, partials sooner.')
    if snap and (snap.get('trend') == 'down' or snap.get('inverted')):
        neff = snap.get('neff')
        al.append('<b>Correlation:</b> industries move together in this state'
                  + (f' (5 positions ≈ {neff:.1f} bets now)' if neff == neff and neff is not None else '')
                  + ': count names that move together as one position.')
    if deep == 'DEEP RISK-OFF':
        al.append('<b>DEEP RISK-OFF:</b> wider both ways, not a bottom call. Every setup did worse than a random entry '
                  'here: size down.')
    if band == 'RISK-OFF' or deep == 'DEEP RISK-OFF':
        al.append('<b>No shorts on laggards:</b> in RISK-OFF they rebounded (shorts lost 2.7% a month).')
    if al:
        h += ''.join(f'<div class="al">{a}</div>' for a in al)

    h += (f'<div class="ft">{since + " · " if since else ""}{fresh}</div>'
          '<div class="ft">Discretion stays with the trade. State cells: survivorship-free stocks 2009–26 (S2 / S3, '
          'in-sample, phases drawn after S2). Sizing and the cap: P2 portfolio simulation. Detail in the Market detail fold.</div>')
    return (h + '</div>').replace('\n', ' ')


def detail_html(changes: list, rotation: str, deep_line: str) -> str:
    """Market detail fold: the full 'since the last session' list, the rotation line and the deep-state text."""
    h = market_state.CSS + CSS + '<div class="td" style="border-left-color:#2a2724;">'
    if changes:
        h += _row('Since last session', '<br>'.join(changes))
    if rotation:
        h += _row('Rotation', rotation)
    if deep_line:
        h += _row('Deep state', deep_line)
    return (h + '</div>').replace('\n', ' ')
