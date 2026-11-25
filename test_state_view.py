"""Unit checks for state_view.py and the v8.1.0 state-aware colours (run: python test_state_view.py)."""
import pandas as pd
import state_view as sv
import scan_view
import rs_labels
import market_state


def test_crash_and_modes():
    assert sv.crash_window({'bucket': 'IN DIP', 'code': 0}) and sv.crash_window({'bucket': 'EARLY', 'code': 1})
    assert not sv.crash_window({'bucket': 'MID', 'code': 2}) and not sv.crash_window({})
    assert sv.rs_mode({'bucket': 'IN DIP', 'code': 0, 'vb': 'WIDE'}) == 'invert'
    assert sv.rs_mode({'bucket': 'EARLY', 'code': 3, 'vb': 'CALM'}) == 'normal'          # 8.5.0: DIP ZONE only
    assert sv.rs_mode({'bucket': 'LATE', 'code': 2, 'vb': 'CALM'}) == 'neutral'
    assert sv.rs_mode({'bucket': 'LATE', 'code': 2, 'vb': 'WIDE'}) == 'normal'
    assert sv.rs_mode({}) == 'normal'


def test_scan_verdicts():
    v = lambda code, b='MID': {'code': code, 'bucket': b}
    assert sv.scan_verdict('L_bo', v(1))[0] == 'lagged' and sv.scan_verdict('L_bo', v(2))[0] == 'works'
    on, off = dict(v(0, 'IN DIP'), f200={'state': 'F-UP'}), dict(v(0, 'IN DIP'), f200={'state': 'F-DN'})
    assert sv.scan_verdict('L_dip', on)[0] == 'works' and sv.scan_verdict('L_dip', off) == sv.DIP_OFF       # 8.5.1
    assert sv.scan_verdict('L_dip', v(0, 'IN DIP'))[0] == 'no edge'          # filter unknown: not paying
    assert sv.scan_verdict('L_pb', dict(v(0, 'IN DIP'), f200='F-UP'))[0] == 'works' and sv.scan_verdict('L_dip', v(3))[0] == 'no edge'
    assert sv.scan_verdict('L_bases', v(1))[0] == 'untested'
    h2 = sv.scan_strip_html(on)
    assert 'frame entry, stop and target' in h2 and 'the state carries it' in h2 and 'no edge' not in h2
    assert sv.scan_verdict('L_bases', v(3))[0] == 'lagged'
    assert sv.scan_verdict('L_all', v(0, 'IN DIP'))[0] == 'lagged' and sv.scan_verdict('L_all', v(1, 'EARLY'))[0] == 'no edge'
    assert sv.scan_verdict('L_all', v(2, 'LATE'))[0] == 'no edge'
    assert sv.scan_verdict('L_dip', {}) == (None, '')
    h = sv.scan_strip_html(v(1, 'EARLY'))
    assert 'dip window' in h and 'Breakouts' in h and '▼' in h
    assert sv.scan_strip_html({}) == ''


def test_rs_colours():
    scan_view.RS_MODE = 'normal'
    assert scan_view._rs_col(1.5) == scan_view.TEAL and scan_view._rr_col(90) == scan_view.TEAL
    scan_view.RS_MODE = 'invert'
    assert scan_view._rs_col(1.5) == scan_view.ORANGE and scan_view._rs_col(-1.5) == scan_view.TEAL
    assert scan_view._rr_col(90) == scan_view.COPPER and scan_view._rr_col(20) == scan_view.TEAL
    scan_view.RS_MODE = 'neutral'
    assert scan_view._rs_col(1.5) == scan_view.RS_GREY == scan_view._rr_col(90)
    scan_view.RS_MODE = 'normal'


def test_momentum_light_crash():
    spy = pd.Series(range(100, 400), dtype=float)
    assert rs_labels.momentum_light(spy, 'RISK-ON', crash=False)['on']
    l = rs_labels.momentum_light(spy, 'RISK-ON', crash=True)
    assert not l['on'] and 'dip window' in l['why']
    assert not rs_labels.momentum_light(spy, 'RISK-OFF')['on']          # old behaviour without a v8 state


def test_avoid_banner():
    lag = pd.DataFrame([dict(Industry='X', ETF='XX', Sector='S', Label='Structural Laggard', Score=-1.5, **{'RS 1M': -0.01},
                             **{'RS 3M': -0.1}, Why='w')])
    a, b = market_state.avoid_html(lag, None, crash=True), market_state.avoid_html(lag, None, crash=False)
    assert 'rebound window' in a and 'opacity:0.55' in a and 'Avoid: laggards' in b
    assert a.count('<div') == a.count('</div>') and b.count('<div') == b.count('</div>')
    e = market_state.avoid_html(pd.DataFrame(), None, crash=True)
    assert e.count('<div') == e.count('</div>')


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('ok', n)


def test_scanner_all_columns_sticky_872():
    import scan_view as sv
    css = sv.CSS
    assert 'th:first-child,td:first-child{position:sticky;left:0;' in css
    assert 'min-width:1410px' in css          # all 21 columns keep the full-width table
    src = open('app.py', encoding='utf-8').read()
    assert "st.toggle('More columns'" not in src and '_cols = None   # 8.7.2' in src


def test_leaders_wording_8121():
    """8.12.1: the leader / laggard texts apply the RS1 result to the DIP ZONE, not the whole dip window."""
    import market_state as ms
    off = dict(on=False, spy_12m=0.10, risk_off=True, why='SPY in the dip window (DIP ZONE or ≤ 20 sessions after)')
    on = dict(on=True, spy_12m=0.10, risk_off=False, why='SPY above its level 252 sessions ago and outside the dip window')
    assert 'In the DIP ZONE itself' in ms.light_detail(off) and 'gap was noise' in ms.light_detail(off)
    h_off, h_on = ms.leaders_html(pd.DataFrame(), off), ms.leaders_html(pd.DataFrame(), on)
    assert 'beaten-down names (bottom 40% by 6-month return), not prior leaders' in h_off
    assert 'not prior leaders' not in h_on and '2008–2026 replay' in h_on
    lag = pd.DataFrame([dict(Industry='X', ETF='XX', Sector='S', Label='Structural Laggard', Score=-1.5, **{'RS 1M': -0.01},
                             **{'RS 3M': -0.1}, Why='w')])
    a = ms.avoid_html(lag, None, crash=True)
    assert 'In the DIP ZONE itself' in a and 'confirmed 2001–08' in a and a.count('<div') == a.count('</div>')
