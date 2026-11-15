"""Checks for data_check.py (run: python test_data_check.py)."""
from datetime import datetime
import pandas as pd
import data_check as dc


def test_expected_session():
    assert dc.expected_session(datetime(2026, 11, 16, 3, 0)) == pd.Timestamp('2026-11-13')      # Monday morning -> Friday
    assert dc.expected_session(datetime(2026, 11, 13, 17, 0)) == pd.Timestamp('2026-11-13')     # Friday after the close
    assert dc.expected_session(datetime(2026, 11, 27, 9, 0)) == pd.Timestamp('2026-11-25')      # day after Thanksgiving


def test_status_and_strip():
    e = pd.Timestamp('2026-11-13')
    assert dc.status('2026-11-13', e) == 'ok' and dc.status('2026-11-12', e) == 'late' and dc.status('2026-11-10', e) == 'stale'
    assert dc.status(None, e) == 'missing' and dc.status(None, e, pending=True) == 'waiting'
    h = dc.strip_html([('A', '2026-11-13', 't', False), ('B', None, 't', True)], [], e)
    assert 'ALL CURRENT' in h and '·' in h
    h = dc.strip_html([('A', '2026-11-12', 't', False)], [], e); assert '1 LATE' in h
    h = dc.strip_html([('A', None, 't', False)], [('A', 'x')], e); assert '1 ISSUE' in h and 'reload' in h


if __name__ == '__main__':
    for n, f in list(globals().items()):
        if n.startswith('test_'):
            f(); print('ok', n)
