"""
test_scoring.py — Unit tests for MRS scoring functions
=======================================================
Verifies the corrected Extension scoring logic.
Run: python test_scoring.py
"""
import numpy as np
import pipeline


def test_score_extension():
    """Test score_extension with the corrected conditional logic."""
    print('Testing score_extension...')

    # Extended + Strong momentum → 0 (was -0.5: THE FIX)
    score, state = pipeline.score_extension(0.85, 0.80)
    assert score == 0.0, f'Extended + Strong should be 0.0, got {score}'
    assert state == 'Extended', f'State should be Extended, got {state}'
    print(f'  ✓ Extended + Strong: {score} ({state})')

    # Extended + Normal momentum → 0
    score, state = pipeline.score_extension(0.85, 0.50)
    assert score == 0.0, f'Extended + Normal should be 0.0, got {score}'
    print(f'  ✓ Extended + Normal: {score} ({state})')

    # Extended + Weak momentum → -0.5 (Fragile+Extreme)
    score, state = pipeline.score_extension(0.85, 0.20)
    assert score == -0.5, f'Extended + Weak should be -0.5, got {score}'
    assert state == 'Extended+Weak', f'State should be Extended+Weak, got {state}'
    print(f'  ✓ Extended + Weak: {score} ({state})')

    # Compressed → -0.5 (bear continuation, independent of momentum)
    score, state = pipeline.score_extension(0.20, 0.50)
    assert score == -0.5, f'Compressed should be -0.5, got {score}'
    assert state == 'Compressed', f'State should be Compressed, got {state}'
    print(f'  ✓ Compressed: {score} ({state})')

    # Compressed + Weak → still -0.5 (bear continuation)
    score, state = pipeline.score_extension(0.20, 0.20)
    assert score == -0.5, f'Compressed + Weak should be -0.5, got {score}'
    print(f'  ✓ Compressed + Weak: {score} ({state})')

    # Normal → 0
    score, state = pipeline.score_extension(0.50, 0.50)
    assert score == 0.0, f'Normal should be 0.0, got {score}'
    assert state == 'Normal', f'State should be Normal, got {state}'
    print(f'  ✓ Normal: {score} ({state})')

    # Extended with NaN momentum → 0 (default to no penalty when mom unknown)
    score, state = pipeline.score_extension(0.85, np.nan)
    assert score == 0.0, f'Extended + NaN momentum should be 0.0, got {score}'
    print(f'  ✓ Extended + NaN mom: {score} ({state})')

    # NaN extension → 0
    score, state = pipeline.score_extension(np.nan, 0.50)
    assert score == 0.0, f'NaN extension should be 0.0, got {score}'
    print(f'  ✓ NaN extension: {score} ({state})')

    print('  All score_extension tests passed!\n')


def test_score_momentum():
    """Test score_momentum (unchanged from original)."""
    print('Testing score_momentum...')

    # Weak
    score, state = pipeline.score_momentum(0.20)
    assert score == -1.0, f'Weak should be -1.0, got {score}'
    print(f'  ✓ Weak: {score} ({state})')

    # Strong
    score, state = pipeline.score_momentum(0.80)
    assert score == 0.5, f'Strong should be 0.5, got {score}'
    print(f'  ✓ Strong: {score} ({state})')

    # Normal
    score, state = pipeline.score_momentum(0.50)
    assert score == 0.0, f'Normal should be 0.0, got {score}'
    print(f'  ✓ Normal: {score} ({state})')

    # NaN
    score, state = pipeline.score_momentum(np.nan)
    assert score == 0.0, f'NaN should be 0.0, got {score}'
    print(f'  ✓ NaN: {score} ({state})')

    print('  All score_momentum tests passed!\n')


def test_structural_flags():
    """Test the structural flag functions."""
    print('Testing structural flags...')

    # Bear continuation: Compressed + Weak
    assert pipeline.bear_continuation_onset(0.20, 0.20) == True
    print('  ✓ bear_continuation_onset(0.20, 0.20) = True')

    assert pipeline.bear_continuation_onset(0.20, 0.50) == False
    print('  ✓ bear_continuation_onset(0.20, 0.50) = False')

    assert pipeline.bear_continuation_onset(0.50, 0.20) == False
    print('  ✓ bear_continuation_onset(0.50, 0.20) = False')

    # Fragility extended: Extended + Weak
    assert pipeline.fragility_extended(0.80, 0.20) == True
    print('  ✓ fragility_extended(0.80, 0.20) = True')

    assert pipeline.fragility_extended(0.80, 0.50) == False
    print('  ✓ fragility_extended(0.80, 0.50) = False')

    assert pipeline.fragility_extended(0.50, 0.20) == False
    print('  ✓ fragility_extended(0.50, 0.20) = False')

    # NaN handling
    assert pipeline.bear_continuation_onset(np.nan, 0.20) == False
    assert pipeline.fragility_extended(0.80, np.nan) == False
    print('  ✓ NaN inputs return False')

    print('  All structural flag tests passed!\n')


def main():
    print('\n' + '='*60)
    print('MRS Scoring Unit Tests')
    print('='*60 + '\n')

    test_score_extension()
    test_score_momentum()
    test_structural_flags()

    print('='*60)
    print('ALL TESTS PASSED')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
