"""Scanner theme filter (8.12.0, 25 Nov 2026).

Narrows any scan to a market theme: one or more sectors, one or more industry
groups, leading sectors only, and the industry role of the stock.
Brings back the sector / industry filters removed on 7 Nov 2026 (commit 1bbada3),
now with several sectors and industry groups at once.
Pure functions on the scan DataFrame, so they can be tested without Streamlit.
"""
import pandas as pd

ROLE_OPTS = ['Any', 'Industry leader', 'Leader in a leading industry', 'In a leading industry (any stock)']


def sector_col(df, yf_map=None):
    """Display sector name per row (yfinance name mapped to the dashboard name)."""
    yf_map = yf_map or {}
    if 'sector' not in df.columns:
        return pd.Series('', index=df.index, dtype=object)

    def _m(s):
        if s is None or (isinstance(s, float) and pd.isna(s)):
            return ''
        s = str(s).strip()
        return '' if s.lower() in ('', 'nan', 'none') else yf_map.get(s, s)
    return df['sector'].map(_m)


def group_col(df):
    """Industry group per row: ind_group (NASDAQ industry, small groups merged), else industry."""
    for c in ('ind_group', 'industry'):
        if c in df.columns and df[c].notna().any():
            g = df[c].fillna('').astype(str).str.strip()
            return g.where(~g.str.lower().isin(['nan', 'none']), '')
    return pd.Series('', index=df.index, dtype=object)


def sector_options(df, yf_map=None):
    return sorted({s for s in sector_col(df, yf_map) if s})


def industry_options(df, yf_map=None, sectors=(), keep=()):
    """Industry groups, narrowed to the chosen sectors; values already chosen are always kept."""
    g = group_col(df)
    if sectors:
        g = g[sector_col(df, yf_map).isin(list(sectors))]
    return sorted({x for x in g if x} | {k for k in keep if k})


def _score(v):
    try:
        v = float(v)
        return v if v == v else -9.0
    except (TypeError, ValueError):
        return -9.0


def apply(df, sectors=(), industries=(), lead_sec=False, role='Any', sec_score=None, yf_map=None,
          leader_hold=None):
    """Rows of df that match every active part of the theme (AND across parts, OR within a list)."""
    if df is None or df.empty:
        return df
    m = pd.Series(True, index=df.index)
    sec = sector_col(df, yf_map)
    if sectors:
        m &= sec.isin(list(sectors))
    if industries:
        m &= group_col(df).isin(list(industries))
    if lead_sec:
        ss = sec_score or {}
        m &= sec.map(lambda s: _score(ss.get(s)) >= 1.0)
    if role and role != 'Any':
        if 'ind_role' in df.columns and df['ind_role'].notna().any():
            r = df['ind_role'].fillna('').astype(str)
            if role == ROLE_OPTS[1]:
                m &= r.str.startswith('Leader')
            elif role == ROLE_OPTS[2]:
                m &= r.eq('Leader · leading industry')
            elif 'ind_leading' in df.columns:
                m &= df['ind_leading'].astype(str).str.strip().str.lower().eq('true')
            else:
                m &= False
        else:                                   # scans without industry groups: leader-ETF holdings
            m &= df['ticker'].astype(str).isin(list((leader_hold or {}).keys())) if 'ticker' in df.columns else False
    return df[m]


def is_active(sectors=(), industries=(), lead_sec=False, role='Any'):
    return bool(sectors) or bool(industries) or bool(lead_sec) or (role not in (None, '', 'Any'))


def summary(sectors=(), industries=(), lead_sec=False, role='Any', max_items=3):
    """Short text of the active theme, e.g. 'Technology · Semiconductors, EDP Services · industry leaders'."""
    def _lst(x):
        x = list(x)
        return ', '.join(x[:max_items]) + (f' +{len(x) - max_items}' if len(x) > max_items else '')
    parts = []
    if sectors:
        parts.append(_lst(sectors))
    if industries:
        parts.append(_lst(industries))
    if lead_sec:
        parts.append('leading sectors')
    if role and role != 'Any':
        parts.append({ROLE_OPTS[1]: 'industry leaders', ROLE_OPTS[2]: 'leaders in leading industries',
                      ROLE_OPTS[3]: 'leading industries'}.get(role, role))
    return ' · '.join(parts)
