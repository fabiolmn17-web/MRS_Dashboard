"""Checks for MRS 8.4.0 (16 Nov 2026): shared ATN components, card v9, Risk today v9, vol surface (python test_design_84.py)."""
import base64
import gzip
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import ui
import exposure_guide as eg
import put_log as pl
import risk_panel as rp
import vol_core
import vol_view


def test_tag_kinds():
    h = ui.tag('Tested · in sample')
    assert 'atn-tested' in h and '>Tested<' in h and 'in sample' in h
    assert 'atn-desc' in ui.tag('Descriptive')
    assert 'atn-prov' in ui.tag('Provisional (watch)')
    assert 'atn-model' in ui.tag('Modelled + real quotes') and '+ real' not in ui.tag('Modelled + real quotes')
    assert 'atn-tested' in ui.tag('Per trade · in sample')
    assert ui.bar([(0, 30, '#85cac5')], pin=50).count('atn-pin') == 1
    css = ui.CSS()
    for c in ('#e07a5f', '#ff4f5e', '#7a3a2e'):             # no non-brand reds
        assert c not in css


def _spy(n=300):
    idx = pd.bdate_range('2025-06-02', periods=n)
    rng = np.random.default_rng(3)
    c = 600 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame({'Open': o, 'High': np.maximum(o, c) * 1.004, 'Low': np.minimum(o, c) * 0.996, 'Close': c,
                         'Volume': 1e6}, index=idx)


def test_card_v9_renders():
    spy = _spy()
    for ph_last, vix in ((3, 19.3), (0, 27.0), (2, 14.0), (1, 18.0)):
        ph = pd.Series([3] * 200 + [0] * 20 + [1] * 40 + [ph_last] * 40, index=spy.index)
        h = eg.card_v9_html({}, ph, vix, pl.latest(None), spy, 'Mon 16 Nov 2026')
        assert '<svg' in h and 'atn-ring' in h and 'atn-row' in h
        assert h.count('<div') == h.count('</div>'), (ph_last, h.count('<div'), h.count('</div>'))
        assert 'MRS score' not in h
    assert 'not available' in eg.card_v9_html({}, None, 15.0)
    assert 'The picture.' in eg.explainer_md()


def _d(n=600):
    idx = pd.bdate_range('2010-01-01', periods=n)
    rng = np.random.default_rng(1)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return pd.DataFrame({'close': c, 'f20': c.shift(-20) / c - 1, 'mae20': -abs(rng.normal(0.03, 0.03, n))}, index=idx)


def test_risk_today_v9_renders():
    d = _d()
    ph = pd.Series(3, index=d.index)
    v8 = rp.v8_cell(d, ph, pd.Series(19.0, index=d.index), 3, 19.0)
    v8['bucket'], v8['n_since'] = 'MID', 30
    snap = dict(ext_atr=-1.4, ep_day=5, ep_start=d.index[-5], v8=v8, rho=0.3, neff=2.3, trend='up', inverted=False)
    h = rp.panel_v9_html(snap)
    for k in ('<svg', 'This state', 'Clock', 'Effective bets', 'atn-tag'):
        assert k in h, k
    assert h.count('<div') == h.count('</div>')
    snap['v8'] = {}
    assert 'not available' in rp.panel_v9_html(snap)


def _chain(day='2026-11-13', spot=750.0):
    rows = []
    d0 = pd.Timestamp(day)
    for k_d in (3, 4, 7, 10, 14, 21, 28, 35, 49):
        e = (d0 + pd.Timedelta(days=k_d)).strftime('%Y%m%d')
        t = k_d / 365
        for k in np.arange(spot * 0.9, spot * 1.1, 2.5):
            m = np.log(k / spot)
            iv = 0.15 - 0.6 * m + 2.0 * m * m + (0.03 if k_d == 4 else 0)
            c = vol_core.bs(spot, k, t, iv, 'C'); p = vol_core.bs(spot, k, t, iv, 'P')
            rows.append(dict(expiry=e, strike=round(k, 1), dte=k_d, t=t, c_mid=c, p_mid=p, iv=iv))
    return pd.DataFrame(rows)


def test_vol_core_and_store():
    w = _chain()
    snap = vol_core.snapshot(w, 750.0, '2026-11-13', 'yahoo')
    assert snap['term'] and snap['smiles'] and snap['event']['next']['expiry'] == '20261116'
    assert abs(snap['term'][0]['atm_iv'] - 0.15) < 0.01
    s = json.dumps(snap)                                     # JSON-safe (no NaN)
    assert 'NaN' not in s
    b64 = base64.b64encode(gzip.compress(s.encode())).decode()
    assert len(b64) < 60000
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'v.json'
        for i in range(20):
            sn = dict(snap, date=f'2026-10-{i + 1:02d}')
            vol_core.store(sn, 'yahoo', p)
        assert len(json.loads(p.read_text())) == vol_core.KEEP_DAYS


def test_vol_view():
    a = vol_core.snapshot(_chain(), 750.0, '2026-11-13', 'yahoo')
    b = vol_core.snapshot(_chain('2026-11-12', 748.0), 748.0, '2026-11-12', 'gl')
    gl = {'2026-11-12': b}
    yh = {'2026-11-12': dict(b, src='yahoo'), '2026-11-13': a}
    m = vol_view.merged(gl, yh)
    assert m['2026-11-12']['src'] == 'gl' and m['2026-11-13']['src'] == 'yahoo'
    now, prev = vol_view.latest(m)
    assert now['date'] == '2026-11-13' and prev['date'] == '2026-11-12'
    assert [l for e, l in vol_view.picks(now)][:1] == ['next']
    lines = vol_view.read_lines(now, prev, 19.3)
    assert len(lines) == 4 and 'Read with the card' in lines[-1]
    assert 'Yahoo backup' in vol_view.kicker_html(now, prev)
    fig = vol_view.figure(now, prev)
    assert len(fig.data) >= 4
    assert vol_view.latest({}) == (None, None)


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f()
            print('ok', n)
