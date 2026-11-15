"""
put_log.py — daily SPY put forward log (MRS v8, 15 Nov 2026)
=============================================================
Why: the Layer 3 hedge rules were built on modelled put prices. The 13 Nov 2026 IBKR check (Studies/.../MRS_v7/chain/
FINDINGS_L3_chain_check_2026-11-13.md) showed the model prices the 60-session 5% OTM put about right (model / market
1.02) but underprices short-dated and 10% OTM puts badly (0.27 to 0.77). This log collects real quotes every session,
free, so the hedge row on the card shows a real price and so the DIP-entry put rule can be checked when the next dip
comes.

What: once a day, from Yahoo's SPY option chain (yfinance, the same source as the rest of the pipeline):
  two expiries: the listed expiry nearest to 20 sessions (29 calendar days) and to 60 sessions (87 calendar days);
  three strikes each: nearest to ATM, 5% OTM and 10% OTM;
  per quote: mid (bid/ask; last price only when bid/ask is empty, flagged), implied vol from the mid (Black-Scholes,
  q = 1.6%, r = 13-week T-bill), and the Layer 0 model price (c = 0.85 x VIX/VIX3M term vol, k = 1.5).
Columns go into mrs_history.csv with the prefix pl_ (like the other forward logs), so the daily workflow needs no
change. Never blocks the update.
"""
import math
import datetime as _dt
import numpy as np
import pandas as pd

Q, C, K_SMILE = 0.016, 0.85, 1.5
TARGETS = {'20': 29, '60': 87}           # sessions -> calendar days
OTMS = {'atm': 0.0, 'p5': 0.05, 'p10': 0.10}


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_put(S, K, T, r, s, q=Q):
    if T <= 0 or s <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def implied_vol(price, S, K, T, r, q=Q):
    """Bisection; None when the price is outside the no-arbitrage range."""
    lo, hi = 0.005, 3.0
    if not (bs_put(S, K, T, r, lo, q) <= price <= bs_put(S, K, T, r, hi, q)):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_put(S, K, T, r, mid, q) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def term_vol(days, vix, vix3m):
    """VIX (30 d) / VIX3M (93 d) interpolated in total variance (Layer 0)."""
    v1, v2 = vix / 100, (vix3m / 100 if vix3m and vix3m == vix3m else vix / 100)
    if days <= 30:
        return v1
    w1, w2 = v1 * v1 * 30, v2 * v2 * 93
    w = w1 + (w2 - w1) * (days - 30) / 63
    return math.sqrt(max(w, 1e-8) / days)


def model_put(S, K, days, r, vix, vix3m):
    sK = max(C * term_vol(days, vix, vix3m) * (1 + K_SMILE * math.log(S / K)), 0.01)
    return bs_put(S, K, days / 365, r, sK)


def pick_expiry(expiries, today, target_days, min_days=10):
    best, bd = None, None
    for e in expiries:
        d = (pd.Timestamp(e) - pd.Timestamp(today)).days
        if d < min_days:
            continue
        if bd is None or abs(d - target_days) < abs(bd - target_days):
            best, bd = e, d
    return best, bd


def quote(puts: pd.DataFrame, K_target: float):
    """Row nearest to K_target: (strike, mid, source)."""
    p = puts.dropna(subset=['strike'])
    if p.empty:
        return None
    r = p.iloc[(p['strike'] - K_target).abs().argsort().iloc[0]]
    b, a, last = float(r.get('bid', 0) or 0), float(r.get('ask', 0) or 0), float(r.get('lastPrice', 0) or 0)
    if b > 0 and a > 0 and a >= b:
        return float(r['strike']), 0.5 * (b + a), 'ba'
    if last > 0:
        return float(r['strike']), last, 'last'
    return None


def build_row(spot, today, expiries, chain_fn, r, vix, vix3m):
    """Pure function (testable): chain_fn(expiry) -> puts DataFrame (yfinance layout)."""
    row = {'pl_spot': round(float(spot), 2)}
    srcs = []
    for tag, tdays in TARGETS.items():
        e, d = pick_expiry(expiries, today, tdays)
        if e is None:
            continue
        row[f'pl_exp{tag}'] = str(e)
        row[f'pl_days{tag}'] = int(d)
        puts = chain_fn(e)
        for nm, otm in OTMS.items():
            qv = quote(puts, spot * (1 - otm))
            if qv is None:
                continue
            K, mid, src = qv
            srcs.append(src)
            iv = implied_vol(mid, spot, K, d / 365, r)
            row[f'pl_{nm}_{tag}_k'] = K
            row[f'pl_{nm}_{tag}_mid'] = round(mid, 3)
            row[f'pl_{nm}_{tag}_iv'] = round(iv, 4) if iv else np.nan
            if vix and vix == vix:
                row[f'pl_{nm}_{tag}_mod'] = round(model_put(spot, K, d, r, vix, vix3m), 3)
    row['pl_src'] = 'yahoo ' + ('ba' if srcs and all(s == 'ba' for s in srcs) else ('mixed' if srcs else 'none'))
    return row


def fetch_row():
    """Live: Yahoo SPY chain at the close. Returns (session date, row) or raises."""
    import yfinance as yf
    t = yf.Ticker('SPY')
    h = t.history(period='5d', auto_adjust=False)
    spot = float(h['Close'].iloc[-1])
    day = pd.Timestamp(h.index[-1]).tz_localize(None).normalize()
    def last_close(sym):
        try:
            x = yf.Ticker(sym).history(period='5d', auto_adjust=False)['Close'].dropna()
            return float(x.iloc[-1])
        except Exception:
            return None
    vix, vix3m, irx = last_close('^VIX'), last_close('^VIX3M'), last_close('^IRX')
    r = (irx / 100) if irx else 0.04
    row = build_row(spot, day, t.options, lambda e: t.option_chain(e).puts, r, vix, vix3m)
    return day, row


# Seed until the first logged session: IBKR quote at the 13 Nov 2026 close (read-only snapshot), SPY 749.56
SEED = {'date': '2026-11-13', 'pl_spot': 749.56, 'pl_days60': 98, 'pl_p5_60_k': 710.0, 'pl_p5_60_mid': 11.325,
        'pl_p5_60_iv': 0.1883, 'pl_p5_60_mod': 11.508, 'pl_src': 'ibkr ba (seed)'}


def latest(hist: pd.DataFrame):
    """Last logged 60-session 5% OTM put for the card: dict (falls back to SEED)."""
    h = hist.dropna(subset=['pl_p5_60_mid']) if (hist is not None and 'pl_p5_60_mid' in hist.columns) else None
    if h is None or h.empty:
        h = pd.DataFrame([SEED])
    r = h.iloc[-1]
    S, mid = float(r['pl_spot']), float(r['pl_p5_60_mid'])
    mod = r.get('pl_p5_60_mod', np.nan)
    return {'date': pd.Timestamp(r['date']), 'days': int(r.get('pl_days60', 0) or 0), 'strike': float(r['pl_p5_60_k']),
            'mid': mid, 'cost_pct': mid / S, 'iv': r.get('pl_p5_60_iv', np.nan),
            'model_ratio': (float(mod) / mid) if mod == mod and mid > 0 else np.nan, 'src': str(r.get('pl_src', ''))}
