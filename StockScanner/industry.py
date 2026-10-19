"""
industry.py — Industry relative strength inside each S&P 500 sector
=====================================================================
Each sector of the dashboard's sector table is broken into industries, each
represented by an industry ETF. For every industry ETF the job computes the
SAME metrics and score as the sector table in app.py (returns, RS vs SPY over
1Y / 6M / 3M, sign-based composite, 6M->3M trend), and for its top 10 holdings
the same RS metrics the CAN SLIM scanner shows (plus distance from the
52-week high, ATR compression and squeeze).

Outputs (written by run_industry.py, read by app.py):
    StockScanner/output/industry_rs.csv        one row per industry ETF
    StockScanner/output/industry_holdings.csv  top 10 holdings per industry ETF

Data: yfinance price history (batch download with the scanner's staleness
fallback) and yfinance funds_data.top_holdings for the ETF holdings.
"""

import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .calculations import RelativeStrengthCalculator, PatternDetector

logger = logging.getLogger(__name__)

OUTPUT_DIR    = Path(__file__).parent / 'output'
INDUSTRY_CSV  = OUTPUT_DIR / 'industry_rs.csv'
HOLDINGS_CSV  = OUTPUT_DIR / 'industry_holdings.csv'
PRICE_DAYS    = 420      # ~14 months calendar, same as the scanner
BATCH_SIZE    = 100
N_HOLDINGS    = 10
HOLDINGS_DELAY = 0.5     # seconds between funds_data calls

# ── Industry ETFs per sector (display names match SECTOR_MAP in app.py) ──────
# One liquid US-listed ETF per industry. Some sectors have few dedicated
# industry ETFs (Staples, Real Estate, Utilities); a few ETFs hold non-US names
# (CARZ, COPX, LIT, URA, ESPO), so their holdings can include foreign listings.
INDUSTRY_MAP: Dict[str, List[tuple]] = {
    'Technology': [
        ('SMH',  'Semiconductors'),
        ('IGV',  'Software'),
        ('CIBR', 'Cybersecurity'),
        ('SKYY', 'Cloud Computing'),
    ],
    'Comm Services': [
        ('FDN',  'Internet'),
        ('XTL',  'Telecom'),
        ('ESPO', 'Video Games & Esports'),
    ],
    'Consumer Disc': [
        ('XRT',  'Retail'),
        ('IBUY', 'Online Retail'),
        ('ITB',  'Homebuilders'),
        ('PEJ',  'Leisure & Entertainment'),
        ('CARZ', 'Autos'),
    ],
    'Consumer Staples': [
        ('PBJ',  'Food & Beverage'),
    ],
    'Energy': [
        ('XOP',  'Oil & Gas E&P'),
        ('OIH',  'Oil Services'),
        ('CRAK', 'Refiners'),
        ('AMLP', 'Midstream (MLPs)'),
        ('FCG',  'Natural Gas'),
        ('URA',  'Uranium'),
    ],
    'Financials': [
        ('KBE',  'Banks'),
        ('KRE',  'Regional Banks'),
        ('IAI',  'Brokers & Exchanges'),
        ('KIE',  'Insurance'),
        ('IPAY', 'Payments'),
    ],
    'Health Care': [
        ('XBI',  'Biotech'),
        ('XPH',  'Pharma'),
        ('IHI',  'Medical Devices'),
        ('IHF',  'Healthcare Providers'),
    ],
    'Industrials': [
        ('ITA',  'Aerospace & Defense'),
        ('IYT',  'Transportation'),
        ('JETS', 'Airlines'),
        ('PAVE', 'Infrastructure'),
    ],
    'Materials': [
        ('XME',  'Metals & Mining'),
        ('GDX',  'Gold Miners'),
        ('COPX', 'Copper Miners'),
        ('SLX',  'Steel'),
        ('LIT',  'Lithium & Batteries'),
    ],
    'Real Estate': [
        ('REZ',  'Residential REITs'),
        ('INDS', 'Industrial REITs'),
        ('SRVR', 'Data Center & Infrastructure REITs'),
    ],
    'Utilities': [
        ('ICLN', 'Clean Energy'),
        ('NLR',  'Nuclear'),
        ('PHO',  'Water'),
        ('GRID', 'Smart Grid'),
    ],
}

SECTOR_ETF = {
    'Technology': 'XLK', 'Comm Services': 'XLC', 'Consumer Disc': 'XLY',
    'Consumer Staples': 'XLP', 'Energy': 'XLE', 'Financials': 'XLF',
    'Health Care': 'XLV', 'Industrials': 'XLI', 'Materials': 'XLB',
    'Real Estate': 'XLRE', 'Utilities': 'XLU',
}


def composite_label(score: float) -> str:
    """Same labels as _sector_composite_label() in app.py."""
    if score >= 1.5:  return 'STRUCTURAL LEADER'
    if score >= 1.0:  return 'STRONG OUTPERFORMER'
    if score >= 0.5:  return 'OUTPERFORMING'
    if score <= -1.5: return 'STRUCTURAL LAGGARD'
    if score <= -1.0: return 'STRONG UNDERPERFORMER'
    if score <= -0.5: return 'UNDERPERFORMING'
    return 'NEUTRAL'


def rs_row(s: pd.Series, spy: pd.Series) -> Optional[dict]:
    """Returns, RS vs SPY, composite and trend for one close series.

    Identical formula to build_sector_table() in app.py:
      RS(n)     = s_now / s_n_ago - spy_now / spy_n_ago   (n = 252, 126, 63 bars)
      Score     = sum of sign(RS) * 0.5 over the three periods (-1.5 .. +1.5)
      Trend     = sign shift between RS 6M and RS 3M
      YTD base  = last close of the previous calendar year
    """
    s = s.dropna()
    spy = spy.dropna()
    if len(s) < 63 or spy.empty:
        return None
    p = float(s.iloc[-1])
    p_prev = float(s.iloc[-2]) if len(s) >= 2 else p
    spy_now = float(spy.iloc[-1])
    prev_year = s[s.index.year < s.index[-1].year]
    ytd_base = float(prev_year.iloc[-1]) if len(prev_year) else float(s.iloc[0])

    def ret(n):
        return (p / float(s.iloc[-n]) - 1) if len(s) > n else np.nan

    def rs(n):
        if len(s) <= n or len(spy) <= n:
            return np.nan
        return (p / float(s.iloc[-n])) - (spy_now / float(spy.iloc[-n]))

    rs_1y, rs_6m, rs_3m = rs(252), rs(126), rs(63)
    sc = lambda v: 0.0 if (v is None or np.isnan(v)) else (0.5 if v > 0 else -0.5 if v < 0 else 0.0)
    score = sc(rs_1y) + sc(rs_6m) + sc(rs_3m)

    def sgn(v):
        if v is None or np.isnan(v): return 0
        return 1 if v > 0 else -1 if v < 0 else 0
    s6, s3 = sgn(rs_6m), sgn(rs_3m)
    if s3 > s6:   trend = 'IMPROVING'
    elif s3 < s6: trend = 'FADING'
    elif s3 > 0:  trend = 'STABLE+'
    elif s3 < 0:  trend = 'STABLE-'
    else:         trend = 'NEUTRAL'

    return {
        'Close': p, 'DailyChg': (p / p_prev - 1) if p_prev else 0.0,
        'YTD': p / ytd_base - 1, '1Y': ret(252), '6M': ret(126), '3M': ret(63),
        'RS 1Y': rs_1y, 'RS 6M': rs_6m, 'RS 3M': rs_3m,
        'Score': score, 'Label': composite_label(score), 'Trend': trend,
    }


def fetch_top_holdings(etf: str, n: int = N_HOLDINGS) -> pd.DataFrame:
    """Top n holdings of an ETF from yfinance funds_data. Empty frame on failure."""
    import yfinance as yf
    try:
        th = yf.Ticker(etf).funds_data.top_holdings
        if th is None or th.empty:
            return pd.DataFrame()
        th = th.reset_index()
        sym_col = 'Symbol' if 'Symbol' in th.columns else th.columns[0]
        name_col = 'Name' if 'Name' in th.columns else None
        w_col = 'Holding Percent' if 'Holding Percent' in th.columns else None
        out = pd.DataFrame({
            # Yahoo symbols as given (US class shares already use '-', e.g. BRK-B;
            # foreign listings keep their suffix, e.g. 2330.TW)
            'ticker': th[sym_col].astype(str).str.strip(),
            'name': th[name_col].astype(str) if name_col else '',
            'weight': pd.to_numeric(th[w_col], errors='coerce') if w_col else np.nan,
        })
        out = out[out['ticker'].str.len() > 0].head(n).reset_index(drop=True)
        out['rank'] = np.arange(1, len(out) + 1)
        return out
    except Exception as e:
        logger.warning(f'{etf}: holdings fetch error: {e}')
        return pd.DataFrame()


def build_industry_tables(price_data: Dict[str, pd.DataFrame],
                          holdings: Dict[str, pd.DataFrame],
                          as_of: Optional[str] = None):
    """Pure computation (no network): industry rows + holdings rows.

    price_data: ticker -> OHLCV frame (must include 'SPY')
    holdings:   industry ETF -> frame with ticker, name, weight, rank
    """
    as_of = as_of or date.today().isoformat()
    spy_df = price_data.get('SPY')
    if spy_df is None or spy_df.empty:
        raise ValueError('SPY price data missing')
    spy = spy_df['Close']

    ind_rows = []
    for sector, items in INDUSTRY_MAP.items():
        for etf, industry in items:
            df = price_data.get(etf)
            r = rs_row(df['Close'], spy) if df is not None and not df.empty else None
            row = {'Sector': sector, 'SectorETF': SECTOR_ETF[sector], 'Industry': industry, 'ETF': etf}
            if r is None:
                row.update({k: np.nan for k in ['Close', 'DailyChg', 'YTD', '1Y', '6M', '3M',
                                                 'RS 1Y', 'RS 6M', 'RS 3M', 'Score']})
                row.update({'Label': 'NO DATA', 'Trend': 'NEUTRAL'})
            else:
                row.update(r)
            row['as_of'] = as_of
            row['last_bar'] = df.index.max().date().isoformat() if df is not None and not df.empty else ''
            ind_rows.append(row)
    industries = pd.DataFrame(ind_rows)

    rs_calc, pat_calc = RelativeStrengthCalculator(), PatternDetector()
    h_rows = []
    for etf, h in holdings.items():
        if h is None or h.empty:
            continue
        for _, hr in h.iterrows():
            t = hr['ticker']
            df = price_data.get(t)
            row = {'ETF': etf, 'rank': int(hr['rank']), 'ticker': t,
                   'name': hr.get('name', ''), 'weight': hr.get('weight', np.nan)}
            if df is not None and len(df) >= 2:
                c = df['Close'].dropna()
                rs = rs_calc.compute(df, spy_df)
                try:
                    pat = pat_calc.compute(df)
                except Exception:
                    pat = {}
                row.update({
                    'close': float(c.iloc[-1]),
                    'daily_chg': float(c.iloc[-1] / c.iloc[-2] - 1) if c.iloc[-2] else np.nan,
                    # distance from the highest close of the last 252 sessions
                    # (computed here: the scanner's TechnicalCalculator skips it
                    # for stocks that fail its trend filters)
                    'pct_from_52w_high': float(c.iloc[-1] / c.iloc[-252:].max() - 1),
                    'rs_1y': rs['rs_1y'], 'rs_6m': rs['rs_6m'], 'rs_3m': rs['rs_3m'],
                    'rs_composite': rs['rs_composite'], 'rs_label': rs['rs_label'],
                    'atr_compressed': bool(pat.get('atr_compressed', False)),
                    'bb_kc_squeeze': bool(pat.get('bb_kc_squeeze', False)),
                })
            else:
                row.update({'close': np.nan, 'daily_chg': np.nan, 'pct_from_52w_high': np.nan,
                            'rs_1y': np.nan, 'rs_6m': np.nan, 'rs_3m': np.nan,
                            'rs_composite': np.nan, 'rs_label': 'NO DATA',
                            'atr_compressed': False, 'bb_kc_squeeze': False})
            row['as_of'] = as_of
            h_rows.append(row)
    holdings_df = pd.DataFrame(h_rows)
    return industries, holdings_df


def run_industry(verbose: bool = True):
    """Network run: fetch holdings and prices, compute, save both CSVs."""
    from .scanner import _batch_download
    log = print if verbose else logger.info
    t0 = time.time()

    etfs = [etf for items in INDUSTRY_MAP.values() for etf, _ in items]
    log(f'[1/3] Top {N_HOLDINGS} holdings for {len(etfs)} industry ETFs...')
    holdings = {}
    for etf in etfs:
        holdings[etf] = fetch_top_holdings(etf)
        log(f'  {etf}: {len(holdings[etf])} holdings')
        time.sleep(HOLDINGS_DELAY)

    stock_tickers = sorted({t for h in holdings.values() if not h.empty for t in h['ticker']})
    all_tickers = sorted(set(etfs) | set(stock_tickers) | {'SPY'})
    log(f'[2/3] Price history for {len(all_tickers)} tickers...')
    price_data = _batch_download(all_tickers, PRICE_DAYS, BATCH_SIZE, log)
    missing = [t for t in all_tickers if t not in price_data]
    if missing:
        log(f'  No price data for {len(missing)}: {", ".join(missing[:30])}')

    log('[3/3] Computing industry RS and holdings metrics...')
    industries, holdings_df = build_industry_tables(price_data, holdings)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    industries.to_csv(INDUSTRY_CSV, index=False)
    holdings_df.to_csv(HOLDINGS_CSV, index=False)
    log(f'  {industries["Score"].notna().sum()}/{len(industries)} industries with data, '
        f'{len(holdings_df)} holding rows, {time.time() - t0:.0f}s')
    return industries, holdings_df


def load_industry_results():
    """(industries, holdings) DataFrames, or (None, None) if not produced yet."""
    try:
        if not INDUSTRY_CSV.exists():
            return None, None
        ind = pd.read_csv(INDUSTRY_CSV)
        hol = pd.read_csv(HOLDINGS_CSV) if HOLDINGS_CSV.exists() else pd.DataFrame()
        return (ind if not ind.empty else None), hol
    except Exception as e:
        logger.warning(f'industry load error: {e}')
        return None, None
