#!/usr/bin/env python3
"""
run_ep_scanner.py — CLI entry point for the Episodic Pivot scanner.

Usage:
    cd MRS_WebApp/
    python StockScanner/run_ep_scanner.py

Environment variables:
    EP_MODE   — 'premarket' or 'confirmed' (default: 'confirmed')
    UNIVERSE  — universe source (default: 'russell1000')
"""

import os
import sys
import logging
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from StockScanner.ep_scanner import run_ep_scan   # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)


def main():
    mode     = os.environ.get('EP_MODE', 'confirmed').strip()
    universe = os.environ.get('UNIVERSE', 'russell1000').strip()

    if mode not in ('premarket', 'confirmed'):
        print(f'WARNING: unknown EP_MODE "{mode}", defaulting to "confirmed"')
        mode = 'confirmed'

    print(f'EP Scanner  |  mode={mode}  |  universe={universe}')

    try:
        df = run_ep_scan(
            universe_source=universe,
            mode=mode,
            verbose=True,
        )

        if df.empty:
            print('\nNo EP candidates today — clean market.')
            sys.exit(0)

        print(f'\n{"─"*60}')
        print(f'{len(df)} Episodic Pivot candidate(s):')
        print(f'{"─"*60}')
        for _, row in df.iterrows():
            gap  = row.get('gap_pct', 0) * 100
            vol  = row.get('vol_ratio', 0)
            news = str(row.get('headline', ''))[:70]
            print(f"  {row['ticker']:6s} | +{gap:.1f}% gap | {vol:.1f}x vol | {news}")
        print(f'{"─"*60}')
        sys.exit(0)

    except Exception as e:
        print(f'\nEP scan failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
