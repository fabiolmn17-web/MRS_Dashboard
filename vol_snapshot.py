"""
vol_snapshot.py — store the Gamma Live vol surface snapshot sent from the trading PC (MRS 8.4.0, 16 Nov 2026)
============================================================================================================
Run by .github/workflows/vol_snapshot.yml (workflow_dispatch), which Gamma Live triggers after each close
(gamma_live/mrs_push.py, after the zero-gamma shadow). Inputs arrive as environment variables:

    VOL_DATE   session date, YYYY-MM-DD (New York)
    VOL_B64    the snapshot (vol_core.snapshot format) as gzip-compressed JSON, base64 encoded

Upserts the snapshot into vol_gl.json (the last 15 sessions are kept). The dashboard's Vol surface panel prefers it
over the Yahoo backup (vol_yahoo.json, written by the nightly update).
"""
import base64
import gzip
import json
import os
import sys

import pandas as pd

import vol_core


def decode(b64: str) -> dict:
    return json.loads(gzip.decompress(base64.b64decode(b64)).decode('utf-8'))


def main():
    day = os.environ.get('VOL_DATE', '').strip()
    try:
        pd.Timestamp(day)
    except Exception:
        print(f'VOL_DATE "{day}" is not a date - nothing stored.')
        sys.exit(1)
    try:
        snap = decode(os.environ.get('VOL_B64', '').strip())
    except Exception as err:
        print(f'VOL_B64 could not be decoded ({err}) - nothing stored.')
        sys.exit(1)
    if snap.get('date') != day or not snap.get('term') or not (100 < float(snap.get('spot') or 0) < 3000):
        print(f'Snapshot does not look valid (date {snap.get("date")}, {len(snap.get("term") or [])} expiries) - nothing stored.')
        sys.exit(1)
    snap['src'] = 'gl'
    p = vol_core.store(snap, 'gl')
    print(f'Stored {day}: SPY {snap["spot"]}, {len(snap["term"])} expiries, {len(snap.get("smiles") or {})} smiles -> {p.name}')


if __name__ == '__main__':
    main()
