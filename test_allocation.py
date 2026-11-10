"""Unit checks for allocation.py on synthetic prices (run: python test_allocation.py)."""
import numpy as np, pandas as pd
import allocation as al


def _df(close, spread=0.5):
    c = np.asarray(close, float)
    idx = pd.bdate_range('2020-01-01', periods=len(c))
    return pd.DataFrame({'open': c, 'high': c + spread, 'low': c - spread, 'close': c}, index=idx)


def test_rebound_one_entry_per_selloff():
    up = np.linspace(100, 130, 120)
    dn = np.linspace(130, 112, 15)                     # sharp drop: well below the 50-day
    flat = np.full(10, 112.0)
    d = al.indicators(_df(np.r_[up, dn, flat]))
    e = al.rebound_entries(d)
    assert len(e) == 1, e
    assert d['E'].values[e[0]] <= -1 and d['E'].values[e[0] - 1] > -1


def test_stop_fills_at_open_on_gap():
    up = np.linspace(100, 130, 120)
    d = _df(np.r_[up, [131.0, 120.0]])
    d.iloc[-1, d.columns.get_loc('open')] = 120.0; d.iloc[-1, d.columns.get_loc('low')] = 119.0
    di = al.indicators(d)
    i = len(di) - 2
    j, px, stop, *_ = al._run_trade(di, i, di['ema21'].values)
    assert j == len(di) - 1 and px == 120.0 and stop > 120.0


def test_breakout_exit_two_closes_below_ema():
    c = np.r_[np.linspace(100, 120, 80), [121, 122, 123, 121.0, 118.4, 118.2]]
    raw = _df(c, spread=0.1); raw['low'] = raw['close'] - 2.0      # wide bars -> ATR ~2, stop well below
    raw.iloc[-3:, raw.columns.get_loc('low')] = raw['close'].values[-3:] - 0.2
    d = al.indicators(raw)
    i = 81                                            # close 122 > prior 21-session high
    assert i in al.breakout_entries(d)
    j, px, stop, armed, run = al._run_trade(d, i, d['ema21'].values)
    assert armed and j is not None and run == 2 and px == d['close'].values[j]


def test_one_position_per_ticket_and_reentry_on_exit_day():
    d = al.indicators(_df(np.r_[np.linspace(100, 140, 150), np.linspace(140, 120, 20), np.linspace(120, 150, 60)], spread=0.3))
    tr = al.replay(d, al.breakout_entries(d), d['ema21'].values)
    for a, b in zip(tr, tr[1:]):
        assert b['i'] >= a['j']


def test_state_and_card_render():
    d = _df(np.r_[np.linspace(100, 140, 200), np.linspace(140, 125, 15)], spread=0.4)
    s = al.state(d, 'SPY')
    html = al.card_html({'SPY': s})
    assert 'Allocation rule' in html and ('IN' in html or 'OUT' in html)
    row = al.log_row({'SPY': s})
    assert set(row) == {'alloc_spy_pos', 'alloc_spy_tickets', 'alloc_spy_stopA', 'alloc_spy_stopB'}


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
