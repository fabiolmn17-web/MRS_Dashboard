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
    # 8.16.0: smooth VIX weights over every DIP ZONE day since 1990 (no box)
    assert c['src'] == 'S&P 500 since 1990' and c['scope'] == 'kernel' and c['cell']['n'] == c['dz']['n'] > 1500
    assert 0.15 < c['cell']['up5'] < 0.35 and c['kernel']['h'] == 0.10 and c['kernel']['ess_days'] > 300
    assert c['fan'] is None or (c['fan']['n'] == c['cell']['n'] and len(c['fan']['c50']) == 20)
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


def _long(n=800, seed=2):
    """A risk_long-like frame with the 8.15.0 columns (vix, r1..r20)."""
    rng = np.random.default_rng(seed)
    R = np.cumsum(rng.normal(0, 0.01, (n, 20)), axis=1)
    x = pd.DataFrame(R, columns=rp.RCOLS)
    x['ph'] = np.where(np.arange(n) % 3 == 0, 0, 3); x['vix'] = np.where(np.arange(n) % 2 == 0, 30.0, 20.0)
    x['f20'] = x['r20']; x['mae20'] = np.minimum(x[rp.RCOLS].min(axis=1), 0)
    return x


def test_fan_stats_and_near_line_8150():
    x = _long()
    f = rp.fan_stats(x[x.ph == 0], x)
    assert f['n'] == int((x.ph == 0).sum()) and all(len(f[k]) == 20 for k in ('c10', 'c25', 'c50', 'c75', 'c90', 'a10', 'a90'))
    assert all(a <= b <= c for a, b, c in zip(f['c10'], f['c50'], f['c90']))
    assert rp.fan_stats(x[['f20']], x) is None                       # old file without paths -> no fan
    m = rp.VIX_MEDIAN
    assert rp.near_line(1.5 * m * 1.02) == (1.5 * m, 'WIDE') and rp.near_line(1.5 * m * 0.97) == (1.5 * m, 'STRESSED')
    assert rp.near_line(m * 0.98) == (m, 'WIDE') and rp.near_line(m * 1.2) is None and rp.near_line(float('nan')) is None


def test_fan_svg_and_panel_8150():
    x = _long()
    f = rp.fan_stats(x[x.ph == 0], x)
    rec = list(np.linspace(0.06, 0.0, 20))
    h = rp.fan_svg(f, rec, 716.70, {'200-day': 733.37, 'dip line': 744.5, 'far': 1000.0}, start_lbl='30 Oct')
    assert h.count('<svg') == 1 and '</svg>' in h and 'P90' in h and 'P10' in h and '200-day 733.4' in h
    assert 'far' not in h and '30 Oct' in h and '717' in h                 # levels > 15% away are left out; price axis
    d = _d()
    v8 = rp.v8_cell(d, pd.Series(3, index=d.index), pd.Series(19.0, index=d.index), 3, 19.0)
    v8.update(fan=f, spy_now=716.70, recent=rec, recent_start='30 Oct', lv={'200-day': 733.37}, bucket='MID', n_since=30,
              near=dict(line=26.3, vb='WIDE', n=900, up5=0.23, dn5=0.15, fast=0.21))
    snap = dict(ext_atr=-2.6, ep_day=19, ep_start=d.index[-19], v8=v8, rho=0.37, neff=2.0, trend='down', inverted=False)
    p = rp.panel_v9_html(snap)
    for k in ("from today's close", 'anchor 716.70', 'Out of sample (every dip', 'not a forecast', 'On the other side (WIDE)'):
        assert k in p, k
    assert p.count('<div') == p.count('</div>')
    v8['fan'] = None
    assert "from today's close" not in rp.panel_v9_html(snap)            # falls back to the distribution picture



def test_vix_kernel_8160():
    """8.16.0 (VK1): weights, width ladder, weighted quantiles, no jump across a VIX bucket line."""
    w, h, ed, es = rp.vix_weights([20.0, 22.0, 24.4, 26.0], [1, 2, 3, 4], 22.0)
    assert h == 1.0 or es >= rp.KERNEL_MIN_ESS                         # 4 spells only: ladder runs to the end
    w, h, ed, es = rp.vix_weights(np.r_[np.full(50, 22.0), np.full(50, 24.2)], np.arange(100), 22.0)
    assert h == 0.10 and abs(w[50] - np.exp(-0.5 * (np.log(1.1) / 0.1) ** 2)) < 1e-12 and w[0] == 1.0
    assert abs(rp._wq(np.arange(1, 1002), np.ones(1001), [0.5])[0] - 501) < 1e-9
    if rp.long_hist() is None:
        return
    d = _d(); m = rp.VIX_MEDIAN
    lo, hi = rp.v8_cell(d, None, None, 0, 1.499 * m), rp.v8_cell(d, None, None, 0, 1.501 * m)
    for k in ('up5', 'dn5', 'fast'):
        assert abs(lo['cell'][k] - hi['cell'][k]) < 0.01, k              # boxes: 23% -> 35% on 25 Nov
    assert 'near' not in lo and lo['vb'] != hi['vb']
    ext = rp.v8_cell(d, None, None, 0, 4.5 * m)
    assert ext['kernel']['h'] > 0.10 and ext['kernel']['ess_spells'] >= rp.KERNEL_MIN_ESS


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('ok', n)
