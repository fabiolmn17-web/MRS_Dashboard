"""
test_scoring_v3.py — checks for the MRS v3.0 scoring changes (Report v7.0)
Run: python test_scoring_v3.py
"""
import numpy as np
import pandas as pd
import pipeline
import risk_dial


def test_weights():
    w = pipeline.COMPONENT_WEIGHTS
    five = w['vix'] + w['ext'] + w['mom'] + w['adl'] + w['skew']
    assert abs(five - 5.80) < 1e-9, f'rescaled group must sum to 5.80, got {five}'
    assert (w['b20'], w['pc'], w['gamma'], w['vol']) == (1.10, 1.40, 1.0, 1.0)
    print('  ✓ weights')


def test_score_pc_zones():
    cases = [(0.60, 1.0), (0.70, 0.5), (0.80, 0.0), (0.97, -1.0), (1.10, 1.0)]
    for sma, expected in cases:
        score, _ = pipeline.score_pc(np.nan, sma)
        assert score == expected, f'pc_sma10={sma}: expected {expected}, got {score}'
    assert pipeline.score_pc(np.nan, np.nan)[0] == 0.0
    print('  ✓ score_pc zones')


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
    test_score_pc_zones()
    test_history_consistent()
    test_risk_dial()
    print('All v3.0 tests passed.')
