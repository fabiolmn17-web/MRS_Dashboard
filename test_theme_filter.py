import pandas as pd
import theme_filter as tf

YF = {'Technology': 'Technology', 'Healthcare': 'Health Care', 'Energy': 'Energy'}


def _df():
    return pd.DataFrame({
        'ticker': ['NVDA', 'AMD', 'MSFT', 'NTRA', 'XOM', 'ZZZ'],
        'sector': ['Technology', 'Technology', 'Technology', 'Healthcare', 'Energy', None],
        'ind_group': ['Semiconductors', 'Semiconductors', 'Computer Software', 'Biotech', 'Oil & Gas', None],
        'ind_role': ['Leader · leading industry', None, 'Leader · lagging industry', 'Leader', None, None],
        'ind_leading': [True, True, False, False, False, False],
    })


def test_options():
    d = _df()
    assert tf.sector_options(d, YF) == ['Energy', 'Health Care', 'Technology']
    assert tf.industry_options(d, YF) == ['Biotech', 'Computer Software', 'Oil & Gas', 'Semiconductors']
    assert tf.industry_options(d, YF, sectors=['Technology']) == ['Computer Software', 'Semiconductors']
    assert 'Biotech' in tf.industry_options(d, YF, sectors=['Technology'], keep=['Biotech'])


def test_no_filter_returns_all():
    d = _df()
    assert len(tf.apply(d, yf_map=YF)) == len(d)
    assert not tf.is_active()


def test_sector_and_industry_and():
    d = _df()
    assert list(tf.apply(d, sectors=['Technology'], yf_map=YF)['ticker']) == ['NVDA', 'AMD', 'MSFT']
    assert list(tf.apply(d, sectors=['Technology', 'Energy'], yf_map=YF)['ticker']) == ['NVDA', 'AMD', 'MSFT', 'XOM']
    assert list(tf.apply(d, sectors=['Technology'], industries=['Semiconductors'], yf_map=YF)['ticker']) == ['NVDA', 'AMD']
    assert tf.apply(d, sectors=['Energy'], industries=['Semiconductors'], yf_map=YF).empty


def test_leading_sectors():
    d = _df()
    out = tf.apply(d, lead_sec=True, sec_score={'Health Care': 1.5, 'Technology': 0.5, 'Energy': None}, yf_map=YF)
    assert list(out['ticker']) == ['NTRA']


def test_roles():
    d = _df()
    assert list(tf.apply(d, role=tf.ROLE_OPTS[1], yf_map=YF)['ticker']) == ['NVDA', 'MSFT', 'NTRA']
    assert list(tf.apply(d, role=tf.ROLE_OPTS[2], yf_map=YF)['ticker']) == ['NVDA']
    assert list(tf.apply(d, role=tf.ROLE_OPTS[3], yf_map=YF)['ticker']) == ['NVDA', 'AMD']


def test_role_fallback_without_groups():
    d = _df().drop(columns=['ind_role'])
    out = tf.apply(d, role=tf.ROLE_OPTS[1], yf_map=YF, leader_hold={'AMD': 'SMH'})
    assert list(out['ticker']) == ['AMD']


def test_industry_falls_back_to_industry_column():
    d = _df().drop(columns=['ind_group']).assign(industry=['Semis', 'Semis', 'Software', 'Bio', 'Oil', None])
    assert list(tf.apply(d, industries=['Semis'], yf_map=YF)['ticker']) == ['NVDA', 'AMD']


def test_empty_and_summary():
    assert tf.apply(_df().iloc[0:0], sectors=['Technology'], yf_map=YF).empty
    s = tf.summary(['Technology'], ['Semiconductors', 'A', 'B', 'C'], True, tf.ROLE_OPTS[1])
    assert s == 'Technology · Semiconductors, A, B +1 · leading sectors · industry leaders'
    assert tf.is_active(role=tf.ROLE_OPTS[3])
