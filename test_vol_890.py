"""8.9.0: the vol-surface charts carry their own reading (stage 1)."""
import numpy as np
import pandas as pd

import vol_core
import vol_view


def _chain(day='2026-11-13', spot=750.0):
    """Same synthetic chain as test_design_84 (put skew, a 4-day event bump)."""
    rows, d0 = [], pd.Timestamp(day)
    for k_d in (3, 4, 7, 10, 14, 21, 28, 35, 49):
        e = (d0 + pd.Timedelta(days=k_d)).strftime('%Y%m%d')
        t = k_d / 365
        for k in np.arange(spot * 0.9, spot * 1.1, 2.5):
            m = np.log(k / spot)
            iv = 0.15 - 0.6 * m + 2.0 * m * m + (0.03 if k_d == 4 else 0)
            c = vol_core.bs(spot, k, t, iv, 'C'); p = vol_core.bs(spot, k, t, iv, 'P')
            rows.append(dict(expiry=e, strike=round(k, 1), dte=k_d, t=t, c_mid=c, p_mid=p, iv=iv))
    return pd.DataFrame(rows)


def test_rr25_and_annotations_890():
    a = vol_core.snapshot(_chain(), 750.0, '2026-11-13', 'yahoo')
    b = vol_core.snapshot(_chain('2026-11-12', 748.0), 748.0, '2026-11-12', 'gl')
    e30 = [e for e, l in vol_view.picks(a) if l == '~30d'][0]
    r = vol_view.rr25(a, e30)
    assert r is not None and r['k_P'] < 750.0 < r['k_C']
    assert abs(r['rr'] - (r['iv_C'] - r['iv_P']) * 100) < 1e-9
    assert abs(vol_view.delta(750.0, 750.0, 0.08, 0.2, 'C') - 0.5) < 0.05
    assert vol_view.rr25(a, 'nope') is None
    fig = vol_view.figure(a, b, 19.3)
    txt = ' '.join(str(x.text) for x in fig.layout.annotations)
    for key in ('spot 750.00', 'RR25', '25Δ put', 'front − 30d', 'VIX 19.3'):
        assert key in txt, key
    fig2 = vol_view.figure(a, b)                       # no VIX: no VIX line, no error
    assert 'VIX' not in ' '.join(str(x.text) for x in fig2.layout.annotations)
    lines = vol_view.read_lines(a, b, 19.3)
    assert 'front − 30d' in lines[1]


if __name__ == '__main__':
    test_rr25_and_annotations_890()
    print('ok test_rr25_and_annotations_890')
