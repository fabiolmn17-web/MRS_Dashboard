"""Checks for the v8.2.0 Risk today panel (run: python test_risk_panel_v8.py)."""
import numpy as np
import pandas as pd
import risk_panel as rp


def _d(n=600):
    idx = pd.bdate_range('2010-01-01', periods=n)
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    d = pd.DataFrame({'close': c, 'f20': c.shift(-20) / c - 1, 'mae20': -abs(rng.normal(0.03, 0.03, n))}, index=idx)
    return d


def test_vol_band_and_clock():
    assert rp.vol_band(17.0) == 'CALM' and rp.vol_band(20.0) == 'WIDE' and rp.vol_band(30.0) == 'STRESSED'
    assert rp.clock_of(pd.Series([3, 0, 3, 3])) == ('EARLY', 2)
    assert rp.clock_of(pd.Series([0])) == ('IN DIP', 0)
    assert rp.clock_of(pd.Series([0] + [2] * 61)) == ('LATE', 61)


def test_v8_cell_fallback():
    d = _d()
    ph = pd.Series(np.where(np.arange(len(d)) % 7 == 0, 0, 3), index=d.index)
    vx = pd.Series(15.0, index=d.index)
    c = rp.v8_cell(d, ph, vx, 3, 15.0, use_long=False)
    assert c['scope'] == 'cell' and c['cell']['n'] > 150
    c2 = rp.v8_cell(d, ph, vx, 3, 30.0, use_long=False)     # no STRESSED days -> phase only
    assert c2['scope'] == 'phase'


def test_v8_cell_long_history():
    """8.6.0: with risk_long.csv the cell is the S&P 500 since 1990 (DIP ZONE x WIDE: ~900 days)."""
    if rp.long_hist() is None:
        return
    d = _d()
    c = rp.v8_cell(d, None, None, 0, 22.9)
    assert c['src'] == 'S&P 500 since 1990' and c['scope'] == 'cell' and c['cell']['n'] > 800
    assert 0.15 < c['cell']['up5'] < 0.35 and c['dz']['n'] > c['cell']['n']
    assert rp.C1_CLOCK[0][0] == 'IN DIP' and rp.C1_CLOCK_OFF[3][1] is None


def test_panel_renders():
    d = _d()
    ph = pd.Series(3, index=d.index)
    v8 = rp.v8_cell(d, ph, pd.Series(19.0, index=d.index), 3, 19.0)
    v8['bucket'], v8['n_since'] = 'MID', 30
    snap = dict(ext_atr=-1.4, ep_day=5, ep_start=d.index[-5], v8=v8, rho=0.3, neff=2.3, trend='up', inverted=False)
    h = rp.panel_html(snap)
    for k in ('SPY vs its 50-day', 'This state', 'Clock', 'Effective bets', 'Dip zone: wide both ways'):
        assert k in h, k
    for k in ('Pedal', 'RISK-OFF', 'Insurance', 'volatility guide'):
        assert k not in h, k
    assert h.count('<div') == h.count('</div>')
    snap['v8'] = {}
    assert 'not available' in rp.panel_html(snap)


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('ok', n)
