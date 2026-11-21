"""8.10.0: Joint view card (joint_view.py)."""
import numpy as np
import pandas as pd

import joint_view as jv


def _long(n=4000, seed=7):
    """Synthetic history: OFF days fall deeper; VIX moves against SPY, wider when STRESSED."""
    r = np.random.default_rng(seed)
    d = pd.DataFrame({'date': pd.bdate_range('1990-01-02', periods=n)})
    d['F'] = np.where(r.random(n) < 0.35, 'ON', 'OFF')
    d['E'] = -1 - r.gamma(1.5, 0.8, n)
    d['fall60'] = np.where(d['F'] == 'ON', -r.gamma(2, 1.6, n), -r.gamma(2, 3.0, n))
    d['vb'] = np.where(r.random(n) < 0.6, 'CALM', np.where(r.random(n) < 0.5, 'WIDE', 'STRESSED'))
    d['ret'] = r.normal(0, np.where(d['vb'] == 'STRESSED', 2.0, 0.8))
    d['dvix'] = -1.2 * d['ret'] + r.normal(0, np.where(d['vb'] == 'STRESSED', 2.0, 0.5))
    return d


def _spy(n=300, seed=3):
    r = np.random.default_rng(seed)
    c = 700 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    c[-15:] = c[-16] * np.linspace(1, 0.93, 15)          # a sell-off into the last close
    idx = pd.bdate_range('2025-09-01', periods=n)
    return pd.DataFrame({'Open': c, 'High': c * 1.006, 'Low': c * 0.994, 'Close': c}, index=idx)


def test_filter_stats_and_near_cell():
    L = _long()
    s = jv.filter_stats(L, E_now=-3.0, f_now='OFF')
    assert s['OFF']['med'] < s['ON']['med']                     # OFF falls deeper
    assert s['in_dip'] and s['near']['n'] == 120 and s['near']['f'] == 'OFF'
    assert s['near']['lo'] <= -3.0 <= s['near']['hi']
    assert not jv.filter_stats(L, E_now=0.4, f_now='ON')['in_dip']


def test_vix_stats_slope_and_percentile():
    L = _long()
    v = jv.vix_stats(L, ret_now=-2.5, dvix_now=8.0, vb_now='STRESSED')
    assert v['STRESSED']['slope'] < 0 and v['CALM']['slope'] < 0
    assert v['STRESSED']['sd'] > v['CALM']['sd']
    t = v['today']
    assert 0 <= t['pct'] <= 1 and t['pct'] >= 0.8 and 'more fear' in t['label']
    low = jv.vix_stats(L, ret_now=-2.5, dvix_now=-3.0, vb_now='STRESSED')['today']
    assert 'less fear' in low['label']


def test_today_inputs():
    spy = _spy()
    hist = pd.DataFrame({'date': spy.index[-3:], 'spy': spy['Close'].values[-3:], 'vix': [18.0, 22.0, 26.5]})
    vb = lambda v: 'CALM' if v < 17.56 else ('WIDE' if v < 26.34 else 'STRESSED')
    t = jv.today_inputs(spy, hist, {'state': 'F-DN'}, vb)
    assert t['E'] < -1 and t['f'] == 'OFF'
    assert abs(t['dvix'] - 4.5) < 1e-9 and t['vb'] == 'WIDE' and t['ret'] < 0
    t2 = jv.today_inputs(None, None, None, None)              # nothing available -> no crash
    assert t2['E'] is None and t2['ret'] is None


def test_figures_and_words():
    L = _long()
    today = dict(date=pd.Timestamp('2026-11-20'), E=-4.9, f='OFF', ret=-2.9, dvix=6.0, vix=28.4, vb='STRESSED')
    fs = jv.filter_stats(L, today['E'], today['f']); vs = jv.vix_stats(L, today['ret'], today['dvix'], today['vb'])
    f1 = jv.figure_filter(L, today, fs); f2 = jv.figure_vix(L, today, vs)
    assert len(f1.data) >= 7 and len(f2.data) >= 7
    for fig in (f1, f2):
        txt = ' '.join(a.text or '' for a in fig.layout.annotations)
        assert 'today' in txt
    lf, lv = jv.filter_lines(today, fs), jv.vix_lines(today, vs)
    assert any('Days like today' in x for x in lf) and any('percentile' in x for x in lv)
    h = jv.lines_html('What the chart says', lf)
    assert 'atn-tile' in h and '<script' not in h
    calm = dict(today, vb='CALM')
    f3 = jv.figure_vix(L, calm, jv.vix_stats(L, -0.5, 0.4, 'CALM'))
    assert 'for comparison' in ' '.join(a.text or '' for a in f3.layout.annotations)


def test_brand_colours_only():
    for col in jv.STATE_COL.values():
        assert col in jv.T.values()
