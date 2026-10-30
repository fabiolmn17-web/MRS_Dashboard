"""
watchlist.py — personal watchlist for the MRS dashboard (v5.1, 30 Oct 2026)
============================================================================
Stored as watchlist.csv on a separate branch ("watchlist") of the dashboard repo, written through the GitHub
contents API with the GITHUB_TOKEN from the Streamlit secrets.

Why a separate branch: the daily, backfill and scanner workflows push to main without pulling first. A watchlist
commit on main while one of them runs would make its push fail. On its own branch the watchlist never touches main,
never triggers a redeploy and needs no workflow change.

Columns: ticker, added (YYYY-MM-DD), note, trigger, stop.
"""
import base64
import io
import math

import pandas as pd
import requests

REPO = 'fabiolmn17-web/MRS_Dashboard'
BRANCH = 'watchlist'
PATH = 'watchlist.csv'
COLS = ['ticker', 'added', 'note', 'trigger', 'stop']
API = 'https://api.github.com'


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def _hdr(token):
    return {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy() if df is not None else empty()
    for c in COLS:
        if c not in d.columns:
            d[c] = None
    d = d[COLS]
    d['ticker'] = d['ticker'].astype(str).str.strip().str.upper().str.lstrip('$')
    d = d[d['ticker'].ne('') & d['ticker'].ne('NAN') & d['ticker'].ne('NONE')]
    d['note'] = d['note'].fillna('').astype(str).replace('nan', '')
    d['added'] = d['added'].fillna('').astype(str)
    for c in ('trigger', 'stop'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.drop_duplicates('ticker', keep='last').reset_index(drop=True)


def load(token: str):
    """-> (DataFrame, sha or None, error or None). A missing branch or file is an empty list, not an error."""
    if not token:
        return empty(), None, 'GITHUB_TOKEN not in the Streamlit secrets: the watchlist is kept for this session only.'
    try:
        r = requests.get(f'{API}/repos/{REPO}/contents/{PATH}', params={'ref': BRANCH}, headers=_hdr(token), timeout=10)
        if r.status_code == 404:
            return empty(), None, None
        if r.status_code != 200:
            return empty(), None, f'Could not read the watchlist (GitHub {r.status_code}).'
        j = r.json()
        txt = base64.b64decode(j.get('content', '')).decode('utf-8')
        df = pd.read_csv(io.StringIO(txt)) if txt.strip() else empty()
        return _clean(df), j.get('sha'), None
    except Exception as e:
        return empty(), None, f'Could not read the watchlist ({e}).'


def _ensure_branch(token: str):
    """Create the watchlist branch from main if it does not exist yet."""
    r = requests.get(f'{API}/repos/{REPO}/git/ref/heads/{BRANCH}', headers=_hdr(token), timeout=10)
    if r.status_code == 200:
        return None
    m = requests.get(f'{API}/repos/{REPO}/git/ref/heads/main', headers=_hdr(token), timeout=10)
    if m.status_code != 200:
        return f'Could not read main (GitHub {m.status_code}).'
    c = requests.post(f'{API}/repos/{REPO}/git/refs', headers=_hdr(token), timeout=10,
                      json={'ref': f'refs/heads/{BRANCH}', 'sha': m.json()['object']['sha']})
    if c.status_code not in (200, 201):
        return f'Could not create the watchlist branch (GitHub {c.status_code}).'
    return None


def save(token: str, df: pd.DataFrame, sha, message: str):
    """Write the CSV. -> (new sha or None, error or None)."""
    if not token:
        return None, 'No GITHUB_TOKEN: kept for this session only.'
    try:
        err = _ensure_branch(token)
        if err:
            return None, err
        body = {'message': message, 'branch': BRANCH,
                'content': base64.b64encode(_clean(df).to_csv(index=False).encode('utf-8')).decode('ascii')}
        if sha:
            body['sha'] = sha
        r = requests.put(f'{API}/repos/{REPO}/contents/{PATH}', headers=_hdr(token), json=body, timeout=15)
        if r.status_code == 409 or (r.status_code == 422 and 'sha' in r.text):
            return None, 'The watchlist changed elsewhere. Reload the page and try again.'
        if r.status_code == 403:
            return None, ('GitHub refused the write (403). The token needs "Contents: Read and write" on '
                          'MRS_Dashboard.')
        if r.status_code not in (200, 201):
            return None, f'Could not save the watchlist (GitHub {r.status_code}).'
        return r.json().get('content', {}).get('sha'), None
    except Exception as e:
        return None, f'Could not save the watchlist ({e}).'


def add(df: pd.DataFrame, ticker: str, today: str) -> pd.DataFrame:
    d = _clean(df)
    t = str(ticker).strip().upper()
    if t in set(d['ticker']):
        return d
    return _clean(pd.concat([d, pd.DataFrame([{'ticker': t, 'added': today, 'note': '', 'trigger': None,
                                                'stop': None}])], ignore_index=True))


def remove(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    d = _clean(df)
    return d[d['ticker'] != str(ticker).strip().upper()].reset_index(drop=True)


def _fin(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def metrics(row, close):
    """Distance from the last close to the trigger, and the risk from trigger to stop, in %."""
    trg, stp = row.get('trigger'), row.get('stop')
    to_trg = (float(trg) / close - 1) if _fin(trg) and _fin(close) and close else math.nan
    risk = (abs(float(trg) - float(stp)) / float(trg)) if _fin(trg) and _fin(stp) and float(trg) else math.nan
    return to_trg, risk


def atr14(h: pd.DataFrame) -> float:
    """Wilder ATR(14) on daily bars (the TradingView default)."""
    if h is None or len(h) < 15:
        return math.nan
    c, hi, lo = h['Close'], h['High'], h['Low']
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])


def live_row(row, close, atr):
    """All computed columns for one entry: To trigger (% and % of ATR), Risk (% and % of ATR), ATR as % of price."""
    trg, stp = row.get('trigger'), row.get('stop')
    ok_c, ok_a = _fin(close) and close, _fin(atr) and atr
    to_trg = (float(trg) / close - 1) if _fin(trg) and ok_c else math.nan
    to_trg_atr = (float(trg) - close) / atr if _fin(trg) and ok_c and ok_a else math.nan
    risk = (abs(float(trg) - float(stp)) / float(trg)) if _fin(trg) and _fin(stp) and float(trg) else math.nan
    risk_atr = abs(float(trg) - float(stp)) / atr if _fin(trg) and _fin(stp) and ok_a else math.nan
    atr_pct = atr / close if ok_a and ok_c else math.nan
    return to_trg, to_trg_atr, risk, risk_atr, atr_pct
