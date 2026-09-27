"""
test_scoring_v4.py — checks for the MRS v4.0 scoring (v8 dispersion re-test)
Run: python test_scoring_v4.py
"""
import numpy as np
import pandas as pd
import pipeline
import risk_dial


def test_weights():
    w = pipeline.COMPONENT_WEIGHTS
    assert w['vix'] == 2.22
    assert w['ext'] == w['mom'] == w['pc'] == w['skew'] == 0.91
    assert w['adl'] == w['b20'] == 0.45          # Breadth shares one weight
    assert (w['gamma'], w['vol']) == (1.0, 0.0)  # gamma pending study; volume info only
    print('  ✓ weights')


def test_state_scores():
    assert pipeline.score_vix(0.0)[0] == 0.91 and pipeline.score_vix(1.0)[0] == -2.09
    assert pipeline.score_vix(0.5) == (-0.59, 'Mid')
    assert pipeline.score_extension(0.9) == (0.5, 'Extended')
    assert pipeline.score_extension(0.1) == (-0.5, 'Compressed')
    assert pipeline.score_momentum(0.1)[0] == -0.5 and pipeline.score_momentum(0.9)[0] == 0.5
    assert pipeline.score_adl(0.9) == (0.25, 'Strong')
    assert pipeline.score_b20(0.1, 0.9)[0] == -0.5   # B20 Low scored without ADL Weak
    for sma, exp in [(0.60, 0.25), (0.70, 0.25), (0.80, 0.25), (0.97, -0.25), (1.10, -0.5)]:
        assert pipeline.score_pc(np.nan, sma)[0] == exp, sma
    assert pipeline.score_skew(0.1, 1.2)[0] == -0.5 and pipeline.score_skew(0.9, 0.6)[0] == 0.5
    assert pipeline.score_volume_divergence(0.05, -0.2)[0] == 0.0
    for f, args in [(pipeline.score_vix, (np.nan,)), (pipeline.score_pc, (np.nan, np.nan))]:
        assert f(*args)[0] == 0.0
    print('  ✓ state scores')


def test_history_consistent():
    """Stored mrs_score must equal the weighted sum of stored component scores."""
    h = pipeline.load_history(pipeline.Path(__file__).parent / 'mrs_history.csv')
    w = pipeline.COMPONENT_WEIGHTS
    cols = dict(vix='vix_score', ext='ext_score', mom='mom_score', adl='adl_score', b20='b20_score',
                pc='pc_score', skew='skew_score', gamma='gamma_score', vol='vol_score')
    total = sum(h[c].fillna(0) * w[k] for k, c in cols.items()).round(2)
    bad = (total - h['mrs_score']).abs() > 0.011
    assert bad.sum() == 0, f'{bad.sum()} rows where mrs_score != weighted sum'
    print(f'  ✓ history consistent ({len(h)} rows)')


def test_scale_matches_v3():
    """v4.0 keeps the v3.0 scale so the fixed band cut-offs still apply."""
    h = pipeline.load_history(pipeline.Path(__file__).parent / 'mrs_history.csv')
    s = h.loc[h['date'] >= '2008-01-31', 'mrs_score']
    assert abs(s.std() - 2.81) < 0.15 and abs(s.mean() + 0.66) < 0.15, (s.mean(), s.std())
    print(f'  ✓ scale (mean {s.mean():+.2f}, sd {s.std():.2f})')


def test_risk_dial():
    h = pipeline.load_history(pipeline.Path(__file__).parent / 'mrs_history.csv')
    bp = risk_dial.band_profile(h, horizons=[5, 20])
    for hz in [5, 20]:
        sub = bp[bp['horizon'] == hz].set_index('band')
        assert sub.loc['RISK-OFF', 'p_adverse'] > sub.loc['RISK-ON', 'p_adverse']
        assert sub.loc['RISK-OFF', 'p_favorable'] > sub.loc['RISK-ON', 'p_favorable']
    ctx = risk_dial.score_context(h, -10.0)
    assert ctx['deep_state'] == 'DEEP RISK-OFF'
    print('  ✓ risk dial profile + deep-state flag')


if __name__ == '__main__':
    test_weights()
    test_state_scores()
    test_history_consistent()
    test_scale_matches_v3()
    test_risk_dial()
    print('All v4.0 tests passed.')
