"""Unit checks for exposure_guide.py (MRS v8) and put_log.py (run: python test_exposure_guide.py)."""
import pandas as pd
import exposure_guide as eg
import put_log as pl


def test_vol_band():
    m = eg.VIX_MEDIAN
    assert eg.vol_band(m - 0.01) == 'CALM' and eg.vol_band(m) == 'WIDE' and eg.vol_band(1.5 * m) == 'STRESSED'
    assert eg.vol_band(None) is None and eg.vol_band(float('nan')) is None


def test_clock_and_stance():
    assert eg.clock(pd.Series([3, 0])) == ('IN DIP', 0)
    assert eg.clock(pd.Series([0] + [3] * 20)) == ('EARLY', 20)
    assert eg.clock(pd.Series([0] + [2] * 21)) == ('MID', 21)
    assert eg.clock(pd.Series([0] + [2] * 61)) == ('LATE', 61)
    assert eg.stance(1) == 'PRESS' and eg.stance(2) == 'PRESS' and eg.stance(3) == 'EASE'
    # 8.5.0: DIP ZONE keeps NORMAL only above a rising 200-day; unknown or other -> EASE
    assert eg.stance(0, 'F-UP') == 'NORMAL' and eg.stance(0, {'state': 'F-UP'}) == 'NORMAL'
    assert eg.stance(0, 'F-DN') == 'EASE' and eg.stance(0, 'F-MIX') == 'EASE' and eg.stance(0) == 'EASE'
    assert eg.stance(1, 'F-DN') == 'PRESS' and eg.stance(3, 'F-UP') == 'EASE'
    assert eg.stance(None) is None


def test_filter200():
    import numpy as np
    idx = pd.bdate_range('2024-01-01', periods=300)
    up = pd.DataFrame({'Close': np.linspace(100, 160, 300)}, index=idx)
    dn = pd.DataFrame({'Close': np.linspace(160, 100, 300)}, index=idx)
    f = eg.filter200(up)
    assert f['state'] == 'F-UP' and f['close'] > f['sma'] and f['slope'] > 0
    assert eg.filter200(dn)['state'] == 'F-DN'
    mix = pd.DataFrame({'close': np.r_[np.linspace(100, 160, 280), np.full(20, 120.0)]}, index=idx)   # below a rising 200d
    assert eg.filter200(mix)['state'] == 'F-MIX'
    assert eg.filter200(up.iloc[:150]) is None and eg.filter200(None) is None
    assert eg.filter200(up, upto=-1)['close'] < f['close']
    assert 'not available' in eg.filter_html(None) and 'ON' in eg.filter_html(f) and 'OFF' in eg.filter_html(eg.filter200(dn))
    assert eg._look('IN DIP')[0] == eg.LOOK_DIP and eg._look('EARLY')[0] == eg.LOOK_ELSE


def test_provisional_live_bar():
    import numpy as np
    from zoneinfo import ZoneInfo
    idx = pd.bdate_range(end='2026-11-16', periods=300)
    c = np.linspace(600, 760, 300)
    spy = pd.DataFrame({'Open': c, 'High': c * 1.003, 'Low': c * 0.997, 'Close': c, 'Volume': 1e6}, index=idx)
    ny = lambda h, m: pd.Timestamp(2026, 11, 16, h, m, tz=ZoneInfo('America/New_York'))
    assert eg.live_bar(spy, ny(12, 0)) and not eg.live_bar(spy, ny(16, 10)) and not eg.live_bar(spy.iloc[:-1], ny(12, 0))
    ph = pd.Series([3] * 299 + [0], index=idx)                    # dropped into the DIP ZONE intraday
    st = dict(rb_level=752.89)
    pv = eg.provisional(st, ph, spy, ny(12, 0))
    assert pv['settled'] == 'EASE' and pv['live'] == 'NORMAL' and pv['line'] == 752.89
    h = eg.card_v9_html(st, ph, 19.3, None, spy, 'Fri 13 Nov 2026', now=ny(12, 0))
    assert 'provisional' in h and 'closes below 752.89' in h and '>EASE<' in h
    assert eg.provisional(st, ph, spy, ny(16, 30)) is None
    h2 = eg.card_v9_html(st, ph, 19.3, None, spy, 'Mon 16 Nov 2026', now=ny(16, 30))
    assert 'provisional' not in h2 and '>NORMAL<' in h2


def test_hedge_window():
    assert eg.hedge_window(pd.Series([3] * 10 + [2])) == ('trend start', 0)
    assert eg.hedge_window(pd.Series([2] * 10 + [3, 3])) == ('trend end', 1)
    assert eg.hedge_window(pd.Series([2] * 20)) is None
    assert eg.hedge_window(pd.Series([3] * 5 + [2] + [2] * 6)) is None          # started 6 sessions ago: window over


def test_unit_size_and_caps():
    assert abs(eg.unit_size(100, 97.5) - 0.60) < 1e-9          # 1.5% / 2.5% stop
    assert eg.unit_size(100, 99.5) == 1.0 and eg.unit_size(100, 101) is None
    assert eg.BUDGET == {'PRESS': (0.0075, 0.04, 1.0), 'NORMAL': (0.0075, 0.03, 1.0), 'EASE': (0.00375, 0.015, 0.6)}


def test_card_renders():
    s = dict(A=dict(open=True, entry=700.0, stop=680.0), B=dict(open=False), rb_level=690.0, bo_level=760.0)
    for ph in ([0, 0], [0] + [1] * 30, [0] + [2] * 80, [2] * 70 + [3]):
        for vix in (14.0, 20.0, 30.0, None):
            h = eg.card_html(s, pd.Series(ph), vix, pl.latest(None), None, 'Fri 13 Nov 2026')
            assert 'MRS v8' in h and 'Swing open risk' in h and 'Hedge' in h, (ph, vix)
    assert 'not available' in eg.card_html({}, None, 15.0)
    assert 'Stance' in eg.explainer_md()


def test_put_log_row():
    q = {'2026-12-18': {675: (2.25, 2.29), 712: (4.84, 4.91), 750: (12.62, 12.74)},
         '2027-02-19': {675: (7.03, 7.11), 710: (11.27, 11.38), 750: (20.74, 20.90)}}
    ch = lambda e: pd.DataFrame([dict(strike=k, bid=b, ask=a, lastPrice=0) for k, (b, a) in q.get(e, {}).items()])
    r = pl.build_row(749.56, '2026-11-13', list(q), ch, 0.0432, 19.29, 20.62)
    assert r['pl_p5_60_k'] == 710 and abs(r['pl_p5_60_mid'] - 11.325) < 1e-9 and abs(r['pl_p5_60_iv'] - 0.1883) < 2e-4
    assert abs(r['pl_p10_20_mod'] / r['pl_p10_20_mid'] - 0.265) < 0.01          # the 13 Nov chain check
    assert r['pl_src'] == 'yahoo ba'
    assert pl.latest(pd.DataFrame([dict(date='2026-11-13', **r)]))['strike'] == 710


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('ok', n)
