import numpy as np
import pandas as pd
import tv_export as tx
import chart_view as cv


def test_tv_symbol_and_text():
    assert tx.tv_symbol('brk-b') == 'BRK.B' and tx.tv_symbol(np.nan) == '' and tx.tv_symbol(' nvda ') == 'NVDA'
    assert tx.tv_text(['NVDA', 'BRK-B', 'NVDA', None, '']) == 'NVDA,BRK.B'
    assert tx.tv_text(['A', 'B'], 'Deep dip · Technology, Semis') == '###DEEP DIP - TECHNOLOGY SEMIS,A,B'
    assert tx.tv_text([], 'X') == '' and tx.tv_text(None) == ''
    assert tx.file_name('Deep dip / Tech', '2026-11-26') == 'MRS_Deep_dip_Tech_2026-11-26.txt'


def test_hollow_classes():
    o = np.array([10, 11, 12, 12, 13.0])
    c = np.array([11, 10.5, 12.5, 11, 13.0])
    prev = np.array([9.5, 11, 10.5, 12.5, 11])
    m = cv.hollow_classes(o, c, prev)
    # bar 0: up vs prev, close > open -> hollow up; bar 1: down vs prev, close < open -> filled down
    # bar 2: up vs prev, close > open -> hollow up; bar 3: down vs prev, close < open -> filled down
    # bar 4: up vs prev, close == open -> hollow (doji) up
    assert list(m['hollow_up']) == [True, False, True, False, True]
    assert list(m['filled_down']) == [False, True, False, True, False]
    assert not m['hollow_down'].any() and not m['filled_up'].any()
    assert sum(int(m[k].sum()) for k in m) == len(o)


def test_figure_has_hollow_traces():
    idx = pd.bdate_range('2025-01-01', periods=300)
    rng = np.random.default_rng(1)
    cl = 100 + np.cumsum(rng.normal(0, 1, 300))
    op = cl + rng.normal(0, 0.8, 300)
    h = pd.DataFrame({'Open': op, 'High': np.maximum(op, cl) + 0.5, 'Low': np.minimum(op, cl) - 0.5, 'Close': cl,
                      'Volume': 1e6}, index=idx)
    import plotly.io as pio, plotly.graph_objects as go
    if 'atn' not in pio.templates:                    # registered by atn_brand.inject() in the app
        pio.templates['atn'] = go.layout.Template()
    fig = cv.figure(h, 'TEST', height=400, spy=None)
    cs = [t for t in fig.data if t.type == 'candlestick']
    assert 2 <= len(cs) <= 4
    fills = {t.increasing.fillcolor for t in cs} | {t.decreasing.fillcolor for t in cs}
    assert 'rgba(0,0,0,0)' in fills and (cv.UP in fills or cv.DOWN in fills)


def test_chart_price_line_and_strip_8131():
    import plotly.io as pio, plotly.graph_objects as go
    if 'atn' not in pio.templates:
        pio.templates['atn'] = go.layout.Template()
    idx = pd.bdate_range('2025-01-01', periods=300)
    rng = np.random.default_rng(2)
    cl = 100 + np.cumsum(rng.normal(0, 1, 300)); op = cl + rng.normal(0, 0.8, 300)
    h = pd.DataFrame({'Open': op, 'High': np.maximum(op, cl) + 0.5, 'Low': np.minimum(op, cl) - 0.5, 'Close': cl,
                      'Volume': 1e6}, index=idx)
    fig = cv.figure(h, 'TEST', height=400, spy=None)
    lines = [s_ for s_ in fig.layout.shapes if s_.type == 'line' and s_.xref == 'paper']
    assert any(abs(s_.y0 - cl[-1]) < 1e-9 for s_ in lines)
    assert any(a.text == f'{cl[-1]:,.2f}' for a in fig.layout.annotations)
    strip = [s_ for s_ in fig.layout.shapes if s_.type == 'rect' and s_.x0 == ' ']
    assert len(strip) >= 5
    assert not any(getattr(t, 'fill', None) == 'tonexty' for t in fig.data)        # no full-width shading any more
    assert fig.layout.xaxis.showspikes is False and 'p2l' in cv.CHART_JS and 'border-left:1px' in cv.CHART_JS


def test_candle_classes_no_nan_and_show_arrays_8132():
    import plotly.io as pio, plotly.graph_objects as go
    if 'atn' not in pio.templates:
        pio.templates['atn'] = go.layout.Template()
    idx = pd.bdate_range('2025-01-01', periods=300)
    rng = np.random.default_rng(3)
    cl = 100 + np.cumsum(rng.normal(0, 1, 300)); op = cl + rng.normal(0, 0.8, 300)
    h = pd.DataFrame({'Open': op, 'High': np.maximum(op, cl) + 0.5, 'Low': np.minimum(op, cl) - 0.5, 'Close': cl,
                      'Volume': 1e6}, index=idx)
    fig = cv.figure(h, 'TEST', height=400, spy=None)
    cs = [t for t in fig.data if t.type == 'candlestick']
    assert sum(len(t.x) for t in cs) == cv.BARS                        # every bar once, no NaN padding
    assert all(np.isfinite(np.array(t.open, float)).all() for t in cs)
    cats = list(fig.layout.xaxis.categoryarray)
    assert len(cats) == cv.BARS + cv.FUTURE and cats[cv.BARS - 1] == idx[-1].strftime('%Y-%m-%d')
    assert 'mrs_hlines' in cv.CHART_JS and 'altKey' in cv.CHART_JS and 'mrs_tlines' in cv.CHART_JS and 'shiftKey' in cv.CHART_JS
