#!/usr/bin/env python3
"""
run_scanner.py — CLI entry point for GitHub Actions
=====================================================
Usage:
    cd MRS_WebApp/
    python StockScanner/run_scanner.py

Environment variables (optional):
    MRS_SCORE   — current MRS score to stamp (auto-reads mrs_history.csv if absent)
    MRS_STATE   — current MRS state string (e.g. "RISK-OFF")
    UNIVERSE    — universe source: sp500 (default), sp500_nasdaq100

Results are written to StockScanner/output/scan_results.csv.
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd

# repo root = MRS_WebApp/ (parent of StockScanner/)
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from StockScanner.scanner import run_scan  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)


def _read_last_mrs():
    """Auto-read latest MRS score from mrs_history.csv if no env vars supplied."""
    try:
        csv_path = _repo_root / 'mrs_history.csv'
        df = pd.read_csv(csv_path)
        if df.empty:
            return None, None
        last = df.iloc[-1]
        return float(last.get('mrs_score', 0)), str(last.get('mrs_state', ''))
    except Exception as e:
        print(f'  [mrs] Could not read mrs_history.csv: {e}')
        return None, None


def main():
    mrs_score_str = os.environ.get('MRS_SCORE', '').strip()
    mrs_state     = os.environ.get('MRS_STATE', '').strip()
    universe      = os.environ.get('UNIVERSE', 'sp500')

    mrs_score = None
    if mrs_score_str:
        try:
            mrs_score = float(mrs_score_str)
        except ValueError:
            pass

    if mrs_score is None:
        mrs_score, mrs_state_auto = _read_last_mrs()
        if not mrs_state:
            mrs_state = mrs_state_auto

    print(f'MRS context: score={mrs_score}, state={mrs_state}')

    try:
        df = run_scan(
            universe_source=universe,
            mrs_score=mrs_score,
            mrs_state=mrs_state or None,
            verbose=True,
        )

        if df.empty:
            print('\nWARNING: Empty results — check logs above')
            sys.exit(1)

        n_strict  = int((df.get('pass_mode', pd.Series()) == 'STRICT').sum())
        n_relaxed = int((df.get('pass_mode', pd.Series()) == 'RELAXED').sum())
        print(f'\nScan complete: {n_strict} STRICT, {n_relaxed} RELAXED candidates')
        sys.exit(0)

    except Exception as e:
        print(f'\nScan failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
