"""
extmom_table.py — the Ext·Mom indicator table for one ticker (scanner chart panel and ticker lookup)
====================================================================================================
Same definitions as Epistruct_ExtMom v4.3 (TradingView), SMA 50, ATR(14) Wilder, daily bars:

  Extension  = (close - SMA50) / ATR14   (also shown in std-dev(50) units)
               Deep below <= -3 | Below <= -1 | Just below <= 0 | Near <= +0.5 | Above <= +5 | Far above > +5
  Momentum   = % change of SMA50 over 5 bars
               Weak <= -0.5 | Falling <= 0 | Building <= +0.5 (Fading if above +0.5 in the last 11 bars) | Persistent
  Short-term = 5-bar return: None | Dip <= -2% | Pullback <= -3% | Sharp pullback <= -4%

History, next 21 sessions (closing basis), exactly as the indicator's "This symbol" odds:
  Extension row  : share of past days in the same extension zone that closed back through the SMA within 21 / 63
  Momentum row   : same momentum zone -> slope still > 0 after 21; new 50-bar closing high within 21
  Short-term row : same momentum zone AND 5-day bin -> new 50-bar high; median worst close (further drawdown)
  Swings row     : bad / good case = P10 / P90 of the 21-session return in the same extension zone
                   (in the same momentum x 5-day cell when there is a dip or pullback); typical drawdown = median
                   worst close in the same momentum zone.
Each row also shows the pooled stock-panel reference (StockScanner/data/extmom_pooled_stocks.csv, built by the
Epistruct study studies/stretched_label). Overlapping days, descriptive; cells under 30 days are flagged.

Evidence (study 28 Oct 2026, exploratory): in single stocks a higher extension did NOT come with a fatter left
tail or a weaker median; in ETFs the upside narrowed. Hence "Above" (not "Stretched").
"""
from pathlib import Path

import numpy as np
import pandas as pd

EXT_ZONES = ['Deep below', 'Below', 'Just below', 'Near', 'Above', 'Far above']
EXT_READ = ['Deep below the SMA', 'Below the SMA', 'Just below the SMA', 'Near the SMA', 'Above the SMA',
            'Far above the SMA']
MOM_ZONES = ['Weak', 'Falling', 'Building', 'Persistent', 'Fading']
PB_BINS = ['None', 'Dip', 'Pullback', 'Sharp pullback']
ZONE_ALIAS = {'Stretched': 'Above', 'Extreme': 'Far above'}   # labels used before 28 Oct 2026
MIN_N = 30
POOLED_CSV = Path(__file__).resolve().parent / 'data' / 'extmom_pooled_stocks.csv'


def _atr_wilder(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr = tr.iloc[1:]
    out = pd.Series(np.nan, index=c.index)
    if len(tr) < n:
        return out
    v = tr.values
    a = v[:n].mean()
    vals = [np.nan] * (n - 1) + [a]
    for x in v[n:]:
        a = (a * (n - 1) + x) / n
        vals.append(a)
    out.iloc[1:] = vals
    return out


def states(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day state and forward outcomes. df: daily OHLC with columns High, Low, Close (any case)."""
    d = df.rename(columns=str.title)[['High', 'Low', 'Close']].dropna()
    c, h, l = d['Close'], d['High'], d['Low']
    sma = c.rolling(50).mean()
    atr = _atr_wilder(h, l, c, 14)
    sd = c.rolling(50).std(ddof=0)
    ext = (c - sma) / atr
    exs = (c - sma) / sd
    slope = (sma / sma.shift(5) - 1) * 100
    pk = slope.rolling(11).max()
    r5 = (c / c.shift(5) - 1) * 100
    ez = pd.Series(np.select([ext <= -3, ext <= -1, ext <= 0, ext <= 0.5, ext <= 5], [0, 1, 2, 3, 4], 5),
                   index=c.index).where(ext.notna())
    mz = pd.Series(np.select([slope <= -0.5, slope <= 0, (slope <= 0.5) & (pk > 0.5), slope <= 0.5], [0, 1, 4, 2], 3),
                   index=c.index).where(slope.notna())
    pb = pd.Series(np.select([r5 <= -4, r5 <= -3, r5 <= -2], [3, 2, 1], 0), index=c.index).where(r5.notna())
    s = pd.DataFrame({'close': c, 'sma': sma, 'atr': atr, 'ext': ext, 'ext_sd': exs, 'slope': slope,
                      'steeper': slope > slope.shift(5), 'r5': r5, 'ez': ez, 'mz': mz, 'pb': pb})
    # forward outcomes (Pine: looked back from t+21 / t+63)
    fwd_max = lambda x, n: x[::-1].rolling(n, min_periods=n).max()[::-1].shift(-1)
    fwd_min = lambda x, n: x[::-1].rolling(n, min_periods=n).min()[::-1].shift(-1)
    below = (c <= sma).astype(float).where(sma.notna())
    above = (c >= sma).astype(float).where(sma.notna())
    s['r21'] = (c.shift(-21) / c - 1) * 100
    s['dd21'] = (fwd_min(c, 21) / c - 1) * 100
    s['nh21'] = (fwd_max(c, 21) >= c.rolling(50).max()).astype(float).where(fwd_max(c, 21).notna())
    for n in (21, 63):
        nb, na_ = fwd_max(below, n), fwd_max(above, n)
        s[f'x{n}'] = np.where(ext > 0, nb, na_)
        s[f'x{n}'] = s[f'x{n}'].where(nb.notna())
    s['pos21'] = (slope.shift(-21) > 0).astype(float).where(slope.shift(-21).notna())
    return s


def _q(x, p):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, p)) if len(x) else np.nan


def cells(s: pd.DataFrame, ez: int, mz: int, pb: int) -> dict:
    """The indicator's history cells for the current state, from a states() frame."""
    H = s[s['ez'].notna() & s['mz'].notna() & s['pb'].notna() & s['r21'].notna() & s['pos21'].notna()]
    G = s[s['ez'].notna() & s['x63'].notna()]
    e = H[H['ez'] == ez]
    m = H[H['mz'] == mz]
    st = m[m['pb'] == pb]
    g = G[G['ez'] == ez]
    return {
        'e_n': len(e), 'e_x21': e['x21'].mean() if len(e) else np.nan, 'e_x63': g['x63'].mean() if len(g) else np.nan,
        'e_p10': _q(e['r21'], 10), 'e_p90': _q(e['r21'], 90),
        'm_n': len(m), 'm_pos': m['pos21'].mean() if len(m) else np.nan, 'm_nh': m['nh21'].mean() if len(m) else np.nan,
        'm_mae': float(m['dd21'].median()) if len(m) else np.nan,
        'st_n': len(st), 'st_nh': st['nh21'].mean() if len(st) else np.nan,
        'st_p10': _q(st['r21'], 10), 'st_p90': _q(st['r21'], 90),
        'st_dd': float(st['dd21'].median()) if len(st) else np.nan,
        'first': H.index.min() if len(H) else None, 'last': s.index.max(),
    }


def pooled_table(frames: list) -> pd.DataFrame:
    """Pooled reference cells over many tickers' states() frames (for the study script)."""
    s = pd.concat(frames)
    rows = []
    for ez in range(6):
        for mz in range(5):
            for pb in range(4):
                c = cells(s, ez, mz, pb)
                rows.append({'ez': ez, 'mz': mz, 'pb': pb, **{k: v for k, v in c.items() if k not in ('first', 'last')}})
    return pd.DataFrame(rows)


def load_pooled():
    try:
        return pd.read_csv(POOLED_CSV)
    except Exception:
        return None


def compute(df: pd.DataFrame, pooled: pd.DataFrame = None):
    """Current reading + own-history cells + pooled cells, or None if the history is too short."""
    if df is None or len(df) < 120:
        return None
    s = states(df)
    last = s.iloc[-1]
    if pd.isna(last['ez']) or pd.isna(last['mz']) or pd.isna(last['pb']):
        return None
    ez, mz, pb = int(last['ez']), int(last['mz']), int(last['pb'])
    own = cells(s, ez, mz, pb)
    ref = None
    if pooled is not None and len(pooled):
        r = pooled[(pooled['ez'] == ez) & (pooled['mz'] == mz) & (pooled['pb'] == pb)]
        ref = r.iloc[0].to_dict() if len(r) else None
    return {'ext': float(last['ext']), 'ext_sd': float(last['ext_sd']), 'slope': float(last['slope']),
            'steeper': bool(last['steeper']), 'r5': float(last['r5']), 'ez': ez, 'mz': mz, 'pb': pb,
            'close': float(last['close']), 'sma': float(last['sma']), 'atr': float(last['atr']),
            'own': own, 'ref': ref}


# ── rendering ───────────────────────────────────────────────────────────────────────────────────
EXT_COLORS = ['#f4a062', '#e8b48e', '#a9c9c5', '#85cac5', '#c98d5f', '#e3b48e']
MOM_COLORS = ['#f4a062', '#e8b48e', '#85cac5', '#5fd0b5', '#e5c07b']
TXT, DIM, HEAD = '#e8dccb', '#9b958f', '#f3ece2'


def _p(x):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x * 100:.0f}%'


def _s(x, f='{:+.1f}'):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f.format(x)


def _few(n):
    return f' <span style="color:#f4a062;">[few: n={int(n)}]</span>' if n is not None and n < MIN_N else ''


def render_html(r: dict, ticker: str) -> str:
    if r is None:
        return ''
    o, f = r['own'], r['ref'] or {}
    ez, mz, pb = r['ez'], r['mz'], r['pb']
    yrs = (f"{o['first'].year}–{o['last'].year}" if o.get('first') is not None else '')

    def two(own_txt, ref_txt):
        return (f'<td>{own_txt}</td><td style="color:{DIM};">{ref_txt}</td>')

    word = 'Back above SMA' if r['ext'] < 0 else 'At/below SMA'
    ext_own = f"{word}: {_p(o['e_x21'])} in 21 · {_p(o['e_x63'])} in 63{_few(o['e_n'])}"
    ext_ref = f"{_p(f.get('e_x21'))} · {_p(f.get('e_x63'))}" if f else '—'
    nh_part = (' · new-high odds: see Short-term' if pb > 0 else f" · new 50-bar high: {_p(o['m_nh'])}")
    mom_own = f"Slope still rising: {_p(o['m_pos'])}{nh_part}{_few(o['m_n'])}"
    mom_ref = (f"{_p(f.get('m_pos'))}" + ('' if pb > 0 else f" · {_p(f.get('m_nh'))}")) if f else '—'
    st_own = f"New 50-bar high: {_p(o['st_nh'])} · further drawdown {_s(o['st_dd'])}% (n={o['st_n']}){_few(o['st_n'])}"
    st_ref = f"{_p(f.get('st_nh'))} · {_s(f.get('st_dd'))}% (n={int(f['st_n'])})" if f else '—'
    if pb > 0:
        sw_own = f"Bad case {_s(o['st_p10'])}% · good case {_s(o['st_p90'])}% (trend + pullback){_few(o['st_n'])}"
        sw_ref = f"{_s(f.get('st_p10'))}% · {_s(f.get('st_p90'))}%" if f else '—'
    else:
        sw_own = f"Bad case {_s(o['e_p10'])}% · good case {_s(o['e_p90'])}% · typical drawdown {_s(o['m_mae'])}%"
        sw_ref = f"{_s(f.get('e_p10'))}% · {_s(f.get('e_p90'))}% · {_s(f.get('m_mae'))}%" if f else '—'
    read = (f"{EXT_READ[ez]} · trend {MOM_ZONES[mz].lower()}, {'accelerating' if r['steeper'] else 'decelerating'}"
            + ((' · but a ' + PB_BINS[pb].lower() + ' in 5 bars: expect repair, wider swings') if pb > 0 and mz in (2, 3, 4)
               else (' · plus a ' + PB_BINS[pb].lower() + ' in 5 bars') if pb > 0 else ''))
    zc = lambda txt, col: f'<span style="color:{col};font-weight:600;">{txt}</span>'
    pb_col = '#e5c07b' if pb > 0 else TXT
    rows = [
        ('Extension', f"{r['ext']:+.2f} ATR <span style='color:{DIM};'>({r['ext_sd']:+.2f} σ)</span>",
         zc(EXT_ZONES[ez], EXT_COLORS[ez]), two(ext_own, ext_ref)),
        ('Momentum', f"{r['slope']:+.2f}% / 5 {'↑' if r['steeper'] else '↓'}", zc(MOM_ZONES[mz], MOM_COLORS[mz]),
         two(mom_own, mom_ref)),
        ('Short-term', f"{r['r5']:+.1f}% / 5", zc(PB_BINS[pb], pb_col), two(st_own, st_ref)),
        ('Swings', '—', '21 sessions', two(sw_own, sw_ref)),
    ]
    css = ('<style>.xm{width:100%%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:0.74rem;color:%s;}'
           '.xm th{background:#1d1b19;color:%s;font-weight:600;text-align:left;padding:4px 7px;border-bottom:1px solid #34302c;}'
           '.xm td{padding:4px 7px;border-bottom:1px solid #26231f;vertical-align:top;}'
           '.xm td.v{font-family:IBM Plex Mono,monospace;text-align:right;white-space:nowrap;}'
           '.xm td.z{text-align:center;white-space:nowrap;}</style>') % (TXT, HEAD)
    h = [css, '<table class="xm"><tr><th>SMA50</th><th style="text-align:right;">Value</th><th style="text-align:center;">Zone</th>'
         f'<th>{ticker} history, next 21 bars</th><th style="color:{DIM};">Stock panel</th></tr>']
    for name, val, zone, rest in rows:
        h.append(f'<tr><td>{name}</td><td class="v">{val}</td><td class="z">{zone}</td>{rest}</tr>')
    h.append(f'<tr><td style="color:{HEAD};font-weight:600;">Read</td><td colspan="4">{read}</td></tr></table>')
    ref_src = 'stock panel = 9 large caps 2005–2026 (study 28 Oct; to be rebuilt on the 501-stock universe)'
    h.append(f'<div style="font-size:0.64rem;color:{DIM};line-height:1.5;margin-top:3px;">'
             f'{ticker} history {yrs} (daily, up to 10 years). {ref_src}. Overlapping days, descriptive; '
             f'cells under {MIN_N} days flagged. Zones as the TradingView Ext·Mom v4.3: Above = +0.5 to +5 ATR (a trend state, '
             'not a reversal warning: in the study the left tail did not grow with extension).</div>')
    return ''.join(h)


def render_lines_html(r: dict, ticker: str) -> str:
    """The same content as render_html as compact lines (ticker lookup: sits under the momentum-list checks).
    Own history first; the pooled stock panel in grey brackets."""
    if r is None:
        return ''
    o, f = r['own'], r['ref'] or {}
    ez, mz, pb = r['ez'], r['mz'], r['pb']
    yrs = (f"{o['first'].year}–{o['last'].year}" if o.get('first') is not None else '')
    B = lambda t: f'<b style="color:{HEAD};">{t}</b>'
    zc = lambda txt, col: f'<b style="color:{col};">{txt}</b>'
    ref = lambda t: f' <span style="color:{DIM};">(panel {t})</span>' if f else ''
    word = 'back above SMA' if r['ext'] < 0 else 'at/below SMA'
    lines = [
        f"{B('Extension:')} {r['ext']:+.2f} ATR ({r['ext_sd']:+.2f} σ) · {zc(EXT_ZONES[ez], EXT_COLORS[ez])} — "
        f"{word} {_p(o['e_x21'])} in 21 · {_p(o['e_x63'])} in 63{_few(o['e_n'])}"
        + ref(f"{_p(f.get('e_x21'))} · {_p(f.get('e_x63'))}"),
        f"{B('Momentum:')} {r['slope']:+.2f}% / 5 {'↑' if r['steeper'] else '↓'} · {zc(MOM_ZONES[mz], MOM_COLORS[mz])} — "
        f"slope still rising {_p(o['m_pos'])}" + ('' if pb > 0 else f" · new 50-bar high {_p(o['m_nh'])}") + _few(o['m_n'])
        + ref(f"{_p(f.get('m_pos'))}" + ('' if pb > 0 else f" · {_p(f.get('m_nh'))}")),
        f"{B('Short-term:')} {r['r5']:+.1f}% / 5 · {zc(PB_BINS[pb], '#e5c07b' if pb > 0 else TXT)} — "
        f"new 50-bar high {_p(o['st_nh'])} · further drawdown {_s(o['st_dd'])}% (n={o['st_n']}){_few(o['st_n'])}"
        + ref(f"{_p(f.get('st_nh'))} · {_s(f.get('st_dd'))}%" if f else ''),
    ]
    if pb > 0:
        lines.append(f"{B('Swings, 21 sessions:')} bad case {_s(o['st_p10'])}% · good case {_s(o['st_p90'])}% (trend + pullback)"
                     + _few(o['st_n']) + ref(f"{_s(f.get('st_p10'))}% · {_s(f.get('st_p90'))}%" if f else ''))
    else:
        lines.append(f"{B('Swings, 21 sessions:')} bad case {_s(o['e_p10'])}% · good case {_s(o['e_p90'])}% · typical drawdown "
                     f"{_s(o['m_mae'])}%" + ref(f"{_s(f.get('e_p10'))}% · {_s(f.get('e_p90'))}% · {_s(f.get('m_mae'))}%" if f else ''))
    read = (f"{EXT_READ[ez]} · trend {MOM_ZONES[mz].lower()}, {'accelerating' if r['steeper'] else 'decelerating'}"
            + ((' · but a ' + PB_BINS[pb].lower() + ' in 5 bars: expect repair, wider swings') if pb > 0 and mz in (2, 3, 4)
               else (' · plus a ' + PB_BINS[pb].lower() + ' in 5 bars') if pb > 0 else ''))
    lines.append(f"{B('Read:')} {read}")
    head = (f'<div style="margin-top:10px;"><b style="color:{HEAD};">Ext·Mom (SMA50)</b> '
            f'<span style="color:{DIM};font-size:0.70rem;">{ticker} history {yrs}, next 21 sessions · '
            f'panel = 9 large caps 2005–2026</span></div>')
    foot = (f'<div style="font-size:0.66rem;color:{DIM};line-height:1.5;margin-top:2px;">Overlapping days, descriptive; '
            f'cells under {MIN_N} days flagged. Above = +0.5 to +5 ATR: a trend state, not a reversal warning.</div>')
    return head + '<br>'.join(lines) + foot
