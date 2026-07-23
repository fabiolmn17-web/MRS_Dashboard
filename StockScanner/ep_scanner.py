"""
ep_scanner.py — Episodic Pivot Scanner
=========================================
Detects stocks that gap ≥10% on ≥2x average volume with uptrend structure.

Filters applied:
  - Gap   : today's open vs previous close ≥ 10%
  - Volume: intraday volume ≥ 2x 50-day average
  - Trend : price > 200 SMA  AND  20 SMA > 200 SMA
  - Liq   : 50-day avg volume ≥ 50,000 shares

Two scan modes:
  - 'premarket' : runs ~8:00 AM ET, uses pre-market price for gap detection
  - 'confirmed' : runs ~9:45 AM ET, uses actual open + real intraday volume

News: yfinance ticker.news (free, no API key needed).
"""

import logging
import time
import warnings
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf

from .data.universe import UniverseManager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'output'
EP_CSV     = OUTPUT_DIR / 'ep_results.csv'

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_GAP       = 0.10     # 10% minimum gap (open vs prev close)
MIN_VOL_RATIO = 1.5      # 1.5x 50-day avg volume
MIN_AVG_VOL   = 500_000  # minimum 500k shares average daily volume
SMA_SHORT     = 20
SMA_LONG      = 200
BATCH_SIZE    = 100
RATE_DELAY    = 0.35     # seconds between per-ticker calls


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_ep_scan(
    universe_source: str = 'russell1000',
    mode: str = 'confirmed',          # 'premarket' or 'confirmed'
    min_gap: float = MIN_GAP,
    min_vol_ratio: float = MIN_VOL_RATIO,
    min_avg_vol: int = MIN_AVG_VOL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full Episodic Pivot scan.

    Returns a DataFrame of EP candidates (also saves to ep_results.csv).
    """
    log = print if verbose else logger.info

    log('═' * 60)
    log(f'Episodic Pivot Scanner — {mode.upper()} mode')
    log(f'  Min gap: {min_gap*100:.0f}%  |  Min vol ratio: {min_vol_ratio:.1f}x  '
        f'|  Min avg vol: {min_avg_vol:,}')
    log('═' * 60)

    # ── 1. Universe ───────────────────────────────────────────────────────────
    log('\n[1/4] Loading universe...')
    um = UniverseManager()
    tickers = um.load_universe(source=universe_source)
    log(f'  {len(tickers)} tickers loaded')

    # ── 2. Batch download daily history ───────────────────────────────────────
    # 300 calendar days covers ~210 trading days → enough for SMA200 + recent open
    log('\n[2/4] Downloading price history (300 calendar days, daily)...')
    price_data = _batch_download_daily(tickers, cal_days=300, log=log)
    log(f'  {len(price_data)} tickers with usable data')

    # ── 3. Fast filter: gap + SMA + avg volume ────────────────────────────────
    log('\n[3/4] Screening for EP setups...')
    candidates = []
    for ticker, df in price_data.items():
        result = _check_ep(df, ticker, min_gap, min_avg_vol)
        if result:
            candidates.append(result)

    log(f'  {len(candidates)} gap/SMA/liquidity candidates')

    if not candidates:
        log('\n  No EP candidates today.')
        return _save_empty()

    # ── 4. Volume confirmation + info + news ──────────────────────────────────
    log(f'\n[4/4] Volume + info + news for {len(candidates)} candidates...')
    results = []

    for c in candidates:
        ticker  = c['ticker']
        avg_vol = c['avg_vol_50d']

        # Intraday volume ratio
        vol_ratio = _get_volume_ratio(ticker, avg_vol, mode)
        c['vol_ratio'] = round(vol_ratio, 2)

        if vol_ratio < min_vol_ratio:
            log(f'  {ticker}: vol ratio {vol_ratio:.1f}x < {min_vol_ratio:.1f}x — skip')
            continue

        c['scan_mode'] = mode

        # Name / sector / industry + earnings flag
        c['earnings_flag']      = False
        c['days_since_earnings'] = None
        try:
            t    = yf.Ticker(ticker)
            info = t.info or {}
            c['name']     = info.get('longName') or info.get('shortName', ticker)
            c['sector']   = info.get('sector', '')
            c['industry'] = info.get('industry', '')

            # Earnings flag: was there an earnings release in the last 2 days?
            today = date.today()
            # earningsTimestamp = Unix timestamp of most recent earnings
            et = info.get('earningsTimestamp') or info.get('mostRecentQuarter')
            if et:
                try:
                    earnings_date = date.fromtimestamp(int(et))
                    days_diff = (today - earnings_date).days
                    c['days_since_earnings'] = days_diff
                    c['earnings_flag'] = 0 <= days_diff <= 2
                except Exception:
                    pass

            # Fallback: check calendar for upcoming/recent earnings
            if not c['earnings_flag']:
                try:
                    cal = t.calendar
                    if cal is not None and not cal.empty:
                        # calendar index has 'Earnings Date' row
                        if 'Earnings Date' in cal.index:
                            ed = cal.loc['Earnings Date']
                            for val in (ed if hasattr(ed, '__iter__') else [ed]):
                                try:
                                    ed_date = pd.Timestamp(val).date()
                                    days_diff = (today - ed_date).days
                                    if 0 <= days_diff <= 2:
                                        c['earnings_flag'] = True
                                        c['days_since_earnings'] = days_diff
                                        break
                                except Exception:
                                    pass
                except Exception:
                    pass

        except Exception:
            c['name']     = ticker
            c['sector']   = ''
            c['industry'] = ''

        # News: top headline
        c['headline']  = ''
        c['news_url']  = ''
        c['publisher'] = ''
        try:
            news = yf.Ticker(ticker).news or []
            if news:
                item = news[0]
                # yfinance ≥0.2.x wraps content in a dict
                content = item.get('content', item)
                if isinstance(content, dict):
                    c['headline']  = content.get('title', item.get('title', ''))
                    url_obj        = content.get('canonicalUrl') or {}
                    c['news_url']  = url_obj.get('url', item.get('link', ''))
                    pub_obj        = content.get('provider') or {}
                    c['publisher'] = pub_obj.get('displayName', item.get('publisher', ''))
                else:
                    c['headline']  = item.get('title', '')
                    c['news_url']  = item.get('link', '')
                    c['publisher'] = item.get('publisher', '')
        except Exception as e:
            logger.debug(f'{ticker}: news error: {e}')

        earn_tag = ' [EARNINGS]' if c['earnings_flag'] else ''
        log(f'  ✓ {ticker}: +{c["gap_pct"]*100:.1f}% gap | {vol_ratio:.1f}x vol{earn_tag} | {c["headline"][:50]}')
        results.append(c)
        time.sleep(RATE_DELAY)

    log(f'\n  ✓ {len(results)} confirmed EP candidates')

    if not results:
        return _save_empty()

    df_out = pd.DataFrame(results)
    df_out['scan_date'] = date.today().isoformat()

    # Sort by gap descending
    df_out = df_out.sort_values('gap_pct', ascending=False).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(EP_CSV, index=False)
    log(f'\n  Results saved → {EP_CSV}')
    log('═' * 60)

    return df_out


def load_ep_results():
    """
    Load the most recent EP scan results.
    Returns (DataFrame, date_str) or (None, None) if no file.
    """
    if not EP_CSV.exists():
        return None, None
    try:
        df = pd.read_csv(EP_CSV)
        if df.empty:
            return df, None
        scan_date = str(df['scan_date'].iloc[0]) if 'scan_date' in df.columns else 'unknown'
        return df, scan_date
    except Exception as e:
        logger.warning(f'Failed to load EP results: {e}')
        return None, None


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _check_ep(
    df: pd.DataFrame,
    ticker: str,
    min_gap: float,
    min_avg_vol: int,
) -> Optional[dict]:
    """
    Apply gap + SMA + liquidity filters.
    Returns a candidate dict or None.

    Daily DataFrame layout (yf.download output):
      - iloc[-1] = today's partial bar (has today's Open)
      - iloc[-2] = yesterday's completed bar (has prev close)
    """
    if df is None or len(df) < SMA_LONG + 5:
        return None

    close = df['Close'].astype(float)
    opens = df['Open'].astype(float)
    vol   = df['Volume'].astype(float)

    # Need at least 2 bars for gap calculation
    if len(df) < 2:
        return None

    # SMAs (calculated on full history up to yesterday to avoid look-ahead)
    sma20  = close.rolling(SMA_SHORT).mean()
    sma200 = close.rolling(SMA_LONG).mean()

    # Use yesterday's values for trend filter (last completed bar)
    curr_sma20  = float(sma20.iloc[-2])
    curr_sma200 = float(sma200.iloc[-2])
    prev_close  = float(close.iloc[-2])

    if pd.isna(curr_sma200) or pd.isna(curr_sma20) or curr_sma200 == 0:
        return None

    # Trend: price > 200 SMA AND 20 SMA > 200 SMA
    if prev_close <= curr_sma200:
        return None
    if curr_sma20 <= curr_sma200:
        return None

    # Liquidity: 50-day average volume (excluding today)
    avg_vol = float(vol.iloc[-51:-1].mean()) if len(vol) > 51 else float(vol.iloc[:-1].mean())
    if pd.isna(avg_vol) or avg_vol < min_avg_vol:
        return None

    # Gap: today's open vs yesterday's close
    today_open = float(opens.iloc[-1])
    if today_open <= 0 or prev_close <= 0:
        return None

    gap_pct = (today_open - prev_close) / prev_close
    if gap_pct < min_gap:
        return None

    return {
        'ticker':          ticker,
        'gap_pct':         round(gap_pct, 4),
        'today_open':      round(today_open, 2),
        'prev_close':      round(prev_close, 2),
        'sma20':           round(curr_sma20, 2),
        'sma200':          round(curr_sma200, 2),
        'avg_vol_50d':     round(avg_vol, 0),
        'pct_above_sma200': round((prev_close / curr_sma200 - 1) * 100, 1),
    }


def _get_volume_ratio(ticker: str, avg_vol: float, mode: str) -> float:
    """
    Fetch current intraday volume and compute ratio vs 50d average.

    premarket mode : sum of pre-market 1m bars (before 09:30 ET)
    confirmed mode : sum of all intraday 1m bars so far today
    """
    try:
        t = yf.Ticker(ticker)

        if mode == 'premarket':
            intra = t.history(period='1d', interval='1m', prepost=True)
            if intra.empty:
                return 0.0
            # Localise to ET and keep only pre-market
            idx = intra.index
            if idx.tzinfo is None:
                idx = idx.tz_localize('UTC')
            idx_et = idx.tz_convert('America/New_York')
            cutoff = pd.Timestamp('09:30', tz='America/New_York').time()
            mask   = idx_et.time < cutoff
            vol    = float(intra['Volume'][mask].sum())
        else:
            intra = t.history(period='1d', interval='1m', prepost=False)
            if intra.empty:
                return 0.0
            vol = float(intra['Volume'].sum())

        if avg_vol <= 0:
            return 0.0
        return vol / avg_vol

    except Exception as e:
        logger.debug(f'{ticker}: volume ratio error: {e}')
        return 0.0


def _batch_download_daily(
    tickers: List[str],
    cal_days: int,
    log,
) -> Dict[str, pd.DataFrame]:
    """Batch-download daily price history for the full universe."""
    end   = datetime.now()
    start = end - timedelta(days=int(cal_days * 1.5))

    results   = {}
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch     = tickers[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log(f'  Batch {batch_num}/{n_batches}: {len(batch)} tickers...')
        try:
            raw = yf.download(
                batch,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by='ticker',
            )
            if raw.empty:
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    elif isinstance(raw.columns, pd.MultiIndex):
                        lvl0 = raw.columns.get_level_values(0)
                        if ticker in lvl0:
                            df = raw[ticker].copy()
                        else:
                            continue
                    else:
                        df = raw.copy()

                    if df.empty or 'Close' not in df.columns:
                        continue

                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
                    df   = df[cols].dropna(subset=['Close'])
                    if not df.empty:
                        results[ticker] = df

                except Exception:
                    pass

        except Exception as e:
            log(f'  Batch {batch_num} download error: {e}')

    return results


def _save_empty() -> pd.DataFrame:
    cols = [
        'ticker', 'name', 'sector', 'industry',
        'gap_pct', 'today_open', 'prev_close',
        'sma20', 'sma200', 'avg_vol_50d', 'pct_above_sma200',
        'vol_ratio', 'scan_mode',
        'earnings_flag', 'days_since_earnings',
        'headline', 'news_url', 'publisher',
        'scan_date',
    ]
    df = pd.DataFrame(columns=cols)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EP_CSV, index=False)
    return df
