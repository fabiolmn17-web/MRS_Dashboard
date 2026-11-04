"""
industry_groups.py — industry groups built from the scanned stocks, and the leaders inside them (v0, 4 Nov 2026)
================================================================================================================
Replaces "top-10 holding of an industry ETF" as the way a stock is tied to an industry.

Group     NASDAQ industry of the US list (data/universe.py), one taxonomy for the whole universe. Groups with
          fewer than MIN_GROUP scanned members are merged into "Other <sector>".
Index     Equal-weight, daily-rebalanced index of the group's members (mean daily return).
State     The MRS rules (rs_labels.label_info) on that index vs SPY: held score / label / trend / days in label.
          Two versions:
            grp_*  the whole group (the industry table)
            ind_*  the group WITHOUT the stock itself (the stock's industry reading: a strong stock does not lift
                   its own industry)
Rank      The stock's RS rating ranked inside its group (1 = best), out of ind_n members with a rating.
Leader    ind_rank <= max(3, ceil(20% of the group))  AND  held RS score >= +1.0
          AND  close above the 200-day  AND  within 25% of the 52-week closing high.
Leading   The industry (without the stock) passes the MRS leaders-list rule: held +1.5 with RS 1M not
          negative, or held +0.5 and Improving (rs_labels._on_list). Lagging = industry held score <= -0.5.
Role      'Leader · leading industry' / 'Leader · lagging industry' / 'Leader' / ''.

Descriptive definitions, not tested: whether the within-industry rank adds information beyond the stock's own
RS rating is the open question (claude/mrs_industry_leaders_v0.md).
"""
import math
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import rs_labels  # noqa: E402

MIN_GROUP = 5
LEAD_PCT = 0.20
LEAD_MIN = 3
FIELDS = ['ind_group', 'ind_n', 'ind_rank', 'ind_leader', 'ind_leading', 'ind_role',
          'ind_score', 'ind_label', 'ind_trend', 'ind_days', 'ind_rs3m', 'ind_rs1m',
          'grp_score', 'grp_label', 'grp_trend', 'grp_days', 'grp_rs1y', 'grp_rs6m', 'grp_rs3m', 'grp_rs1m',
          'grp_leading', 'grp_rank', 'grp_n_groups', 'grp_med_rating']


def _key(ticker: str, row: dict, us_meta: Dict[str, dict]) -> str:
    ind = (us_meta.get(ticker, {}) or {}).get('industry') or row.get('industry') or ''
    return str(ind).strip()


def _leading(score, trend, rs1m) -> bool:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return False
    return bool((score >= 1.5 and not (rs1m is not None and np.isfinite(rs1m) and rs1m < 0))
                or (score == 0.5 and trend == 'IMPROVING'))


def _state(idx: pd.Series, spy: pd.Series) -> Optional[dict]:
    try:
        return rs_labels.label_info(idx, spy)
    except Exception:
        return None


def assign(df: pd.DataFrame, price_data: Dict[str, pd.DataFrame], spy_df: pd.DataFrame,
           us_meta: Optional[Dict[str, dict]] = None, log=print) -> pd.DataFrame:
    """Add FIELDS to the scan results (one row per ticker). Needs columns ticker, sector, industry, rs_rating,
    held_score, close, sma200, pct_from_52w_high."""
    us_meta = us_meta or {}
    out = df.copy()
    for k in FIELDS:
        out[k] = None
    if out.empty or spy_df is None or spy_df.empty:
        return out
    spy = spy_df['Close'].dropna()

    # 1. groups
    keys = {r['ticker']: _key(r['ticker'], r, us_meta) for r in out.to_dict('records')}
    cnt = pd.Series(keys).value_counts()
    sect = out.set_index('ticker')['sector'].fillna('').astype(str)
    grp = {t: (k if k and cnt.get(k, 0) >= MIN_GROUP else f"Other {sect.get(t) or 'stocks'}".strip())
           for t, k in keys.items()}
    cnt2 = pd.Series(grp).value_counts()                # merged groups still under MIN_GROUP: one catch-all group
    grp = {t: (g if cnt2.get(g, 0) >= MIN_GROUP else 'Other (small groups)') for t, g in grp.items()}
    out['ind_group'] = out['ticker'].map(grp)

    # 2. daily returns on SPY's dates
    rets = {}
    for t in out['ticker']:
        d = price_data.get(t)
        if d is None or d.empty or 'Close' not in d:
            continue
        rets[t] = d['Close'].reindex(spy.index).pct_change(fill_method=None)
    if not rets:
        return out
    R = pd.DataFrame(rets)
    G = pd.Series(grp).reindex(R.columns)
    has = R.notna()
    S = R.fillna(0.0).T.groupby(G).sum().T            # date x group: sum of member returns
    N = has.T.groupby(G).sum().T                      # date x group: members with a return that day

    def _index(r: pd.Series) -> pd.Series:
        return (1.0 + r.fillna(0.0)).cumprod()

    # 3. whole-group state + group rank by median RS rating
    gstate = {}
    for g in S.columns:
        r = S[g] / N[g].replace(0, np.nan)
        gstate[g] = _state(_index(r), spy)
    med = out.groupby('ind_group')['rs_rating'].median()
    grank = med.rank(ascending=False, method='min')
    ngroups = int(med.notna().sum())

    # 4. rank inside the group by RS rating
    rr = pd.to_numeric(out['rs_rating'], errors='coerce')
    out['ind_rank'] = rr.groupby(out['ind_group']).rank(ascending=False, method='min')
    out['ind_n'] = rr.groupby(out['ind_group']).transform('count')

    # 5. per stock: leave-one-out industry state, leader, leading
    rows = []
    for i, r in out.iterrows():
        t, g = r['ticker'], r['ind_group']
        st = gstate.get(g)
        upd = {'grp_score': None}
        if st:
            upd.update(grp_score=st['Score'], grp_label=st['Label'], grp_trend=st['Trend'], grp_days=st['DaysInLabel'],
                       grp_rs1y=st['RS 1Y'], grp_rs6m=st['RS 6M'], grp_rs3m=st['RS 3M'], grp_rs1m=st['RS 1M'],
                       grp_leading=_leading(st['Score'], st['Trend'], st['RS 1M']))
        upd.update(grp_rank=None if pd.isna(grank.get(g)) else int(grank.get(g)), grp_n_groups=ngroups,
                   grp_med_rating=None if pd.isna(med.get(g)) else float(med.get(g)))
        # without the stock itself
        if g in S.columns:
            if t in R.columns:
                own = R[t]
                n_ = N[g] - own.notna().astype(int)
                s_ = S[g] - own.fillna(0.0)
            else:
                n_, s_ = N[g], S[g]
            loo = _state(_index(s_ / n_.replace(0, np.nan)), spy) if (n_ > 0).sum() > 63 else None
            if loo:
                upd.update(ind_score=loo['Score'], ind_label=loo['Label'], ind_trend=loo['Trend'],
                           ind_days=loo['DaysInLabel'], ind_rs3m=loo['RS 3M'], ind_rs1m=loo['RS 1M'],
                           ind_leading=_leading(loo['Score'], loo['Trend'], loo['RS 1M']))
        n = r['ind_n']
        rank = r['ind_rank']
        hs = pd.to_numeric(pd.Series([r.get('held_score')]), errors='coerce').iloc[0]
        p52 = pd.to_numeric(pd.Series([r.get('pct_from_52w_high')]), errors='coerce').iloc[0]
        cl = pd.to_numeric(pd.Series([r.get('close')]), errors='coerce').iloc[0]
        s200 = pd.to_numeric(pd.Series([r.get('sma200')]), errors='coerce').iloc[0]
        top = (pd.notna(rank) and pd.notna(n) and n > 0 and rank <= max(LEAD_MIN, math.ceil(LEAD_PCT * n)))
        lead = bool(top and pd.notna(hs) and hs >= 1.0 and pd.notna(cl) and pd.notna(s200) and cl > s200
                    and pd.notna(p52) and p52 >= -0.25)
        upd['ind_leader'] = lead
        lg = bool(upd.get('ind_leading'))
        upd['ind_leading'] = lg
        lag = upd.get('ind_score') is not None and pd.notna(upd.get('ind_score')) and upd['ind_score'] <= -0.5
        upd['ind_role'] = ('Leader · leading industry' if lead and lg else 'Leader · lagging industry' if lead and lag
                           else 'Leader' if lead else '')
        rows.append(upd)
    add = pd.DataFrame(rows, index=out.index)
    for k in add.columns:
        out[k] = add[k]
    out['ind_rank'] = out['ind_rank'].astype('Int64')
    out['ind_n'] = out['ind_n'].astype('Int64')
    log(f"  Industry groups: {out['ind_group'].nunique()} (min {MIN_GROUP} members; smaller ones merged into 'Other "
        f"<sector>') · leaders {int(out['ind_leader'].sum())} · in a leading industry "
        f"{int((out['ind_role'] == 'Leader · leading industry').sum())} · leading groups "
        f"{int(pd.Series([bool(v) for v in out.groupby('ind_group')['grp_leading'].first()]).sum())}")
    return out


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group from the scan results (grp_* columns), with its leaders. Sorted: leading first, then score,
    then RS 3M."""
    if df is None or df.empty or 'ind_group' not in df.columns:
        return pd.DataFrame()
    d = df[df['ind_group'].notna()]
    rows = []
    for g, x in d.groupby('ind_group'):
        f = x.iloc[0]
        ld = x[x['ind_leader'].astype(str).str.lower() == 'true'].sort_values('ind_rank')
        rows.append({'Group': g, 'Sector': x['sector'].mode().iloc[0] if x['sector'].notna().any() else '',
                     'N': len(x), 'Score': f.get('grp_score'), 'Label': f.get('grp_label'), 'Trend': f.get('grp_trend'),
                     'Days': f.get('grp_days'), 'RS 1Y': f.get('grp_rs1y'), 'RS 6M': f.get('grp_rs6m'),
                     'RS 3M': f.get('grp_rs3m'), 'RS 1M': f.get('grp_rs1m'),
                     'Leading': str(f.get('grp_leading')).lower() == 'true',
                     'Rank': f.get('grp_rank'), 'Median RS rating': f.get('grp_med_rating'),
                     'Leaders': ', '.join(ld['ticker'].astype(str).head(8)), 'N leaders': len(ld)})
    t = pd.DataFrame(rows)
    t['_s'] = pd.to_numeric(t['Score'], errors='coerce')
    t['_r'] = pd.to_numeric(t['RS 3M'], errors='coerce')
    return t.sort_values(['Leading', '_s', '_r'], ascending=[False, False, False]).drop(columns=['_s', '_r'])


def tv_watchlist(df: pd.DataFrame) -> str:
    """TradingView watchlist import text: sections of industry leaders (### = section)."""
    if df is None or df.empty or 'ind_role' not in df.columns:
        return ''
    d = df.copy()
    d['_r'] = pd.to_numeric(d['rs_rating'], errors='coerce')
    a = d[d['ind_role'] == 'Leader · leading industry'].sort_values('_r', ascending=False)
    b = d[d['ind_role'].isin(['Leader', 'Leader · lagging industry'])].sort_values('_r', ascending=False)
    parts = []
    if len(a):
        parts.append('###LEADERS IN LEADING INDUSTRIES,' + ','.join(a['ticker'].astype(str)))
    if len(b):
        parts.append('###OTHER INDUSTRY LEADERS,' + ','.join(b['ticker'].astype(str)))
    return ','.join(parts)
