#!/usr/bin/env python3
"""
run_industry.py — CLI entry point for GitHub Actions (industry drill-down)
==========================================================================
Usage (from the repo root):
    python StockScanner/run_industry.py

Writes StockScanner/output/industry_rs.csv and industry_holdings.csv, which the
dashboard reads for the sector -> industry -> top-10-holdings drill-down.
"""
import sys
import logging
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from StockScanner.industry import run_industry  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')


def main():
    industries, _ = run_industry(verbose=True)
    if industries is None or industries['Score'].notna().sum() == 0:
        print('WARNING: no industry data computed')
        sys.exit(1)


if __name__ == '__main__':
    main()
