"""
test_scoring_v3.py — superseded by test_scoring_v4.py (MRS v4.0, Sept 2026).
Kept so old instructions still work; it simply runs the v4.0 checks.
"""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).parent / 'test_scoring_v4.py'), run_name='__main__')
