"""8.15.0 checks for build_risk_long.py (run: python test_build_risk_long_8150.py). The Yahoo path is checked with a
stand-in for yfinance built from synthetic prices (Yahoo is blocked in the sandboxes; the first monthly run is the live check)."""
import sys, types, tempfile, os
import numpy as np
import pandas as pd
import build_risk_long as br


def _px(n=1500, seed=3):
    idx = pd.bdate_range('1989-01-02', periods=n)
    rng = np.random.default_rng(seed)
    c = 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
    o = c * (1 + rng.normal(0, 0.003, n)); h = np.maximum(o, c) * 1.004; l = np.minimum(o, c) * 0.996
    spx = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c}, index=idx)
    vix = pd.Series(18 + 6 * np.abs(np.sin(np.arange(n) / 40)), index=idx)
    return spx, vix


def test_build_columns_8150():
    spx, vix = _px()
    out, med = br.build(spx, vix)
    assert list(out.columns[:6]) == ['date', 'ph', 'vix', 'vb', 'f20', 'mae20'] and all(f'r{k}' in out for k in range(1, 21))
    assert (abs(out['r20'] - out['f20']) < 1e-4).all()
    assert (out['mae20'] <= out[[f'r{k}' for k in range(1, 21)]].min(axis=1) + 1e-4).all()      # lows sit at or below closes
    assert out.date.iloc[0] >= '1990-01-02' and out['f20'].notna().all() and 10 < med < 30


def test_yahoo_path_and_guard_8150():
    spx, vix = _px()
    def fake_download(tk, **kw):            # yfinance >= 0.2.48 shape: (field, ticker) MultiIndex columns
        df = spx.copy() if tk == '^GSPC' else pd.DataFrame({'Open': vix, 'High': vix, 'Low': vix, 'Close': vix})
        df['Adj Close'] = df['Close']; df['Volume'] = 0
        df.columns = pd.MultiIndex.from_product([df.columns, [tk]])
        return df
    sys.modules['yfinance'] = types.SimpleNamespace(download=fake_download)
    s2, v2 = br.load_yahoo()
    assert list(s2.columns) == ['Open', 'High', 'Low', 'Close'] and len(s2) == len(spx) and len(v2) > 1000
    out, _ = br.build(s2, v2)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'risk_long.csv')
        assert br.should_write(out, f)[0]
        out.to_csv(f, index=False)
        assert br.should_write(out, f)[0]
        assert not br.should_write(out.iloc[:-30], f)[0]          # shorter / older -> not written


if __name__ == '__main__':
    for n, fn in list(globals().items()):
        if n.startswith('test_'):
            fn(); print('ok', n)
