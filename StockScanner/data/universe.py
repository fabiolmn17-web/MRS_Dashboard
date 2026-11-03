"""
universe.py — Stock universe management
========================================
Fetches and manages the list of US equities to scan.
Handles filtering of ETFs, preferred shares, warrants, etc.
"""

import logging
import re
from datetime import date
from pathlib import Path
from typing import List, Dict, Optional, Set
import pandas as pd
import requests
from io import StringIO

from ..config import get_config

logger = logging.getLogger(__name__)


# Security type keywords to exclude (case-insensitive matching)
EXCLUDED_KEYWORDS = {
    "etf", "etn", "preferred", "pfd", "warrant", "wt", "unit",
    "closed-end", "fund", "trust", "spac", "acquisition",
    "rights", "reit",  # REITs are optional - can be enabled
}

# Suffix patterns that indicate non-common stock
EXCLUDED_SUFFIXES = {
    ".WS",   # Warrants
    ".U",    # Units
    ".R",    # Rights
    # "-A" / "-B" / "-C" class shares are KEPT (BRK-B, BF-B were wrongly dropped before 3 Nov 2026)
    ".PR",   # Preferred
    "-P",
}



# ── US liquid universe (3 Nov 2026) ──────────────────────────────────────────
# Every NYSE / NASDAQ / AMEX listing from the daily NASDAQ screener copies kept on GitHub (rreichel3/US-Stock-Symbols,
# reachable from GitHub Actions, unlike iShares and api.nasdaq.com), filtered to common stocks and ADRs with a market
# cap of at least $1B and a price of at least $10. The scanner then drops names whose 50-day average dollar volume is
# under $20M (scanner.MIN_DOLLAR_VOLUME), from its own price download. The last good list is saved next to this file
# (universe_us.csv) and used when the download fails; the S&P 500 is the last resort.
US_LIST_URL = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/{ex}/{ex}_full_tickers.json'
SAVED_UNIVERSE = Path(__file__).parent / 'universe_us.csv'
US_MIN_MCAP = 1e9
US_MIN_PRICE = 10.0
US_MIN_NAMES = 900                         # fewer = the download is broken: use the saved list
_NOT_COMMON = re.compile(r'\bnotes?\b|debenture|preferred|\bwarrants?\b|\brights?\b(?! to receive)|perpetual|'
                         r'\bunits?\s*$|acquisition corp|\bETF\b|\bfund\b', re.I)
NASDAQ_TO_YF_SECTOR = {'Technology': 'Technology', 'Finance': 'Financial Services', 'Health Care': 'Healthcare',
                       'Consumer Discretionary': 'Consumer Cyclical', 'Consumer Staples': 'Consumer Defensive',
                       'Industrials': 'Industrials', 'Energy': 'Energy', 'Utilities': 'Utilities',
                       'Real Estate': 'Real Estate', 'Basic Materials': 'Basic Materials',
                       'Telecommunications': 'Communication Services'}


def build_us_liquid(timeout: int = 60) -> pd.DataFrame:
    """Download and filter the US listings. Columns: ticker, name, exchange, sector (yfinance naming, from NASDAQ's
    coarse classification: a fallback only), industry, market_cap, price, country, built."""
    rows = []
    for ex in ('nyse', 'nasdaq', 'amex'):
        r = requests.get(US_LIST_URL.format(ex=ex), timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        for x in r.json():
            x['exchange'] = ex.upper()
            rows.append(x)
    d = pd.DataFrame(rows)
    d['price'] = pd.to_numeric(d['lastsale'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    d['market_cap'] = pd.to_numeric(d['marketCap'], errors='coerce')
    sym = d['symbol'].astype(str).str.strip()
    d = d[~sym.str.contains(r'\^', regex=True)].copy()                 # ^ = preferred series
    d['ticker'] = d['symbol'].astype(str).str.strip().str.replace('/', '-', regex=False)   # BRK/B -> BRK-B
    dv = d['price'] * pd.to_numeric(d['volume'], errors='coerce')
    # market cap missing in the source for a few real stocks (BF-B): keep those if the day traded >= $20M
    big = (d['market_cap'] >= US_MIN_MCAP) | ((d['market_cap'].isna() | (d['market_cap'] <= 0)) & (dv >= 20e6))
    d = d[big & (d['price'] >= US_MIN_PRICE) & (d['ticker'].str.len() <= 6)
          & ~d['name'].astype(str).str.contains(_NOT_COMMON)]
    d = d.drop_duplicates('ticker')
    out = pd.DataFrame({'ticker': d['ticker'], 'name': d['name'].astype(str).str.strip(), 'exchange': d['exchange'],
                        'sector': d['sector'].map(NASDAQ_TO_YF_SECTOR).fillna(''), 'industry': d['industry'].fillna(''),
                        'market_cap': d['market_cap'], 'price': d['price'], 'country': d['country'].fillna(''),
                        'built': date.today().isoformat()})
    return out.sort_values('market_cap', ascending=False).reset_index(drop=True)


class UniverseManager:
    """Manages the stock universe for scanning."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self._tickers: List[str] = []
        self._ticker_info: Dict[str, Dict] = {}
        self.source_note = ''                      # what load_universe actually used (written into the scan)
        self.us_meta: Dict[str, Dict] = {}         # ticker -> name / sector / industry from the US list

    # ── Static fallback ────────────────────────────────────────────────────────
    # Core S&P 500 members — used when all network sources fail.
    # Update this list when significant composition changes occur.
    _SP500_STATIC: List[str] = [
        # Tech
        "AAPL","MSFT","NVDA","AVGO","ORCL","AMD","QCOM","TXN","INTC","CSCO",
        "ADBE","CRM","INTU","AMAT","LRCX","MU","KLAC","SNPS","CDNS","PANW",
        "FTNT","CRWD","ANET","ANSS","MPWR","MCHP","ON","WDC","STX","NTAP",
        "HPQ","HPE","DELL","GDDY","PTC","TYL","JKHY","EPAM","CTSH","ACN",
        "IBM","FSLR","ENPH","SMCI","SWKS","ZBRA","JNPR","GPN","FLT",
        # Financials
        "JPM","BAC","WFC","GS","MS","BLK","C","AXP","V","MA","COF","DFS",
        "USB","PNC","TFC","SCHW","MCO","SPGI","ICE","CME","CBOE","NDAQ",
        "FIS","FISERV","PYPL","AIG","MET","PRU","AFL","ALL","CB","TRV",
        "PGR","HIG","AJG","MMC","AON","L","CINF","GL","BEN","IVZ","TROW",
        "STT","BK","NTRS","RF","CFG","KEY","HBAN","FITB","MTB","ZION","CMA",
        # Healthcare
        "UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","BSX","SYK","MDT",
        "ISRG","EW","VRTX","REGN","BIIB","GILD","AMGN","MRNA","ILMN","BMY",
        "PFE","ZBH","BAX","BDX","IQV","CRL","HOLX","RMD","DXCM","IDXX","MTD",
        "WAT","A","COO","HSIC","ALGN","TFX","STE","PODD","INCY","CTLT",
        "MCK","CVS","CI","ELV","HUM","MOH","HCA","CNC","DVA",
        # Consumer Discretionary
        "AMZN","TSLA","HD","MCD","SBUX","NKE","LOW","TGT","BKNG","ABNB",
        "CMG","DRI","YUM","DPZ","HLT","MAR","H","PHM","DHI","LEN","NVR",
        "TOL","AZO","ORLY","GPC","AN","PAG","KMX","APTV","F","GM","BWA",
        "LEA","TSCO","TJX","ROST","BBY","RL","PVH","TPR","ETSY","EBAY",
        # Consumer Staples
        "WMT","PG","KO","PEP","COST","PM","MO","MDLZ","EL","KHC","CL",
        "GIS","K","CAG","SJM","MKC","HSY","HRL","BG","CPB","TSN","LW",
        "SYY","MCK","CAH","ABC","CASY","ACI","SFM",
        # Energy
        "XOM","CVX","COP","EOG","SLB","MPC","OXY","PXD","VLO","PSX","HAL",
        "HES","DVN","FANG","MRO","APA","BKR","CTRA","EQT","AR","MTDR",
        "NOV","PTEN","CHK","RRC","MGY","SM","NOG","CNX","OVV",
        # Industrials
        "GE","HON","RTX","CAT","DE","LMT","UPS","NOC","BA","GD","EMR",
        "ETN","PH","ROK","ITW","ROP","FAST","GWW","CMI","FDX","TEX","PCAR",
        "OTIS","CARR","TDG","HEI","AXON","CACI","SAIC","LDOS","L3H","KNX",
        "WAB","CSX","UNP","NSC","J","CHRW","XPO","JBHT","LSTR","ODFL",
        "AOS","LII","GNRC","HII","NDSN","WMS",
        # Materials
        "LIN","APD","ECL","PPG","SHW","NEM","GOLD","FCX","DOW","DD","LYB",
        "EMN","CE","ALB","AA","NUE","STLD","RS","CF","MOS","FMC","IFF",
        "AVY","PKG","SEE","IP","WRK","SON","GEF",
        # Real Estate
        "AMT","CCI","PLD","SPG","O","WELL","VTR","DLR","EQIX","EXR","PSA",
        "AVB","EQR","UDR","CPT","MAA","SBA","SBAC","GLPI","VICI","WY",
        "CBRE","JLL","CSGP","ARE","BXP","KIM","REG","FRT","NNN","WPC",
        "STAG","REXR","EGP","FR",
        # Utilities
        "NEE","DUK","SO","D","EXC","AEP","SRE","XEL","ES","ED","FE","CNP",
        "NI","CMS","LNT","PEG","AEE","DTE","WEC","AWK","ETR","PPL","NRG",
        "CEG","VST",
        # Communication Services
        "GOOGL","GOOG","META","NFLX","DIS","CMCSA","VZ","T","TMUS","CHTR",
        "WBD","PARA","FOX","FOXA","OMC","IPG","NWSA","NWS","NYT","MTCH",
        "SNAP","PINS","UBER","LYFT","IAC",
    ]

    _BROWSER_HEADERS: dict = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def _fetch_sp500(self) -> List[str]:
        """
        Fetch S&P 500 constituents.
        Sources tried in order:
          1. GitHub datasets CSV (reliable in CI/CD, no bot blocking)
          2. Wikipedia via requests + browser User-Agent (bypasses 403)
          3. Static fallback list embedded in this file
        """
        # ── Source 1: GitHub datasets ──────────────────────────────────────────
        gh_url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/main/data/constituents.csv"
        )
        try:
            resp = requests.get(gh_url, headers=self._BROWSER_HEADERS, timeout=20)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            tickers = df[col].str.replace(".", "-", regex=False).tolist()
            if len(tickers) > 400:
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from GitHub datasets")
                return tickers
        except Exception as e:
            logger.warning(f"GitHub datasets fetch failed: {e}")

        # ── Source 2: Wikipedia with browser User-Agent ────────────────────────
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            resp = requests.get(wiki_url, headers=self._BROWSER_HEADERS, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            df = tables[0]
            col = "Symbol" if "Symbol" in df.columns else df.columns[0]
            tickers = df[col].str.replace(".", "-", regex=False).tolist()
            if len(tickers) > 400:
                logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
                return tickers
        except Exception as e:
            logger.warning(f"Wikipedia S&P 500 fetch failed: {e}")

        # ── Source 3: Static fallback ──────────────────────────────────────────
        logger.warning(f"Using static S&P 500 list ({len(self._SP500_STATIC)} tickers)")
        return list(self._SP500_STATIC)

    def _fetch_nasdaq100(self) -> List[str]:
        """Fetch Nasdaq 100 — Wikipedia with browser User-Agent."""
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        try:
            resp = requests.get(url, headers=self._BROWSER_HEADERS, timeout=30)
            resp.raise_for_status()
            tables = pd.read_html(StringIO(resp.text))
            for table in tables:
                col = None
                if "Ticker" in table.columns:
                    col = "Ticker"
                elif "Symbol" in table.columns:
                    col = "Symbol"
                if col and len(table) > 80:
                    tickers = table[col].str.replace(".", "-", regex=False).tolist()
                    logger.info(f"Fetched {len(tickers)} Nasdaq 100 tickers from Wikipedia")
                    return tickers
        except Exception as e:
            logger.warning(f"Nasdaq 100 Wikipedia fetch failed: {e}")
        return []

    # Product IDs for iShares ETF holdings CSV downloads
    _ISHARES_URLS = {
        "IWB": "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
        "IWV": "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
        "IWM": "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund",
    }

    def _fetch_russell_from_ishares(self, etf: str = "IWB") -> List[str]:
        """
        Fetch Russell universe from iShares ETF holdings CSV.
        IWB = Russell 1000  (default — ~1000 large/mid caps)
        IWV = Russell 3000
        IWM = Russell 2000

        iShares CSVs have a variable-length metadata header before the
        actual holdings table, so we locate the Ticker column row and
        use on_bad_lines='skip' to tolerate field-count mismatches.
        """
        url = self._ISHARES_URLS.get(etf, self._ISHARES_URLS["IWB"])
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()

            lines = resp.text.splitlines()

            # Find the header row that contains the "Ticker" column
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith("Ticker") or ",Ticker," in line or line.startswith('"Ticker"'):
                    header_idx = i
                    break

            if header_idx is None:
                logger.warning(f"iShares {etf}: could not locate Ticker header row")
                return []

            csv_data = "\n".join(lines[header_idx:])
            df = pd.read_csv(
                StringIO(csv_data),
                on_bad_lines="skip",   # skip metadata trailer lines with wrong field count
                dtype=str,
            )

            ticker_col = next((c for c in df.columns if c.strip() == "Ticker"), None)
            if ticker_col is None:
                logger.warning(f"iShares {etf}: Ticker column not found in parsed CSV")
                return []

            tickers = df[ticker_col].dropna().tolist()
            tickers = [t.strip().replace(".", "-") for t in tickers if isinstance(t, str) and t.strip()]
            # Drop cash, money-market, and blank placeholder rows
            tickers = [t for t in tickers if t and not t.startswith("-") and len(t) <= 5]

            if len(tickers) > 400:
                logger.info(f"Fetched {len(tickers)} tickers from iShares {etf}")
                return tickers
            else:
                logger.warning(f"iShares {etf}: only {len(tickers)} tickers — suspiciously low, discarding")
                return []

        except Exception as e:
            logger.warning(f"Failed to fetch from iShares {etf}: {e}")
            return []

    def _fetch_from_nasdaq_api(self) -> List[str]:
        """Fetch all NASDAQ/NYSE listed stocks from NASDAQ's API."""
        tickers = []
        for exchange in ["nasdaq", "nyse", "amex"]:
            url = f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&exchange={exchange}&download=true"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                rows = data.get("data", {}).get("rows", [])
                exchange_tickers = [
                    row.get("symbol", "").replace(".", "-")
                    for row in rows
                    if row.get("symbol")
                ]
                tickers.extend(exchange_tickers)
                logger.info(f"Fetched {len(exchange_tickers)} tickers from {exchange.upper()}")
            except Exception as e:
                logger.warning(f"Failed to fetch {exchange}: {e}")

        return list(set(tickers))

    def _is_excluded_security(self, ticker: str, name: str = "") -> bool:
        """Check if a security should be excluded based on ticker or name."""
        ticker_upper = ticker.upper()
        name_lower = (name or "").lower()

        for suffix in EXCLUDED_SUFFIXES:
            if ticker_upper.endswith(suffix):
                return True

        for keyword in EXCLUDED_KEYWORDS:
            if keyword in name_lower:
                return True

        if len(ticker) > 5:
            return True

        return False

    def _filter_tickers(self, tickers: List[str]) -> List[str]:
        """Filter out excluded security types."""
        filtered = []
        for ticker in tickers:
            if not self._is_excluded_security(ticker):
                filtered.append(ticker)

        excluded_count = len(tickers) - len(filtered)
        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} excluded securities")

        return filtered

    def _load_us_liquid(self) -> List[str]:
        """Live list (saved for next time), else the saved list. Sets self.source_note and self.us_meta."""
        df = None
        try:
            df = build_us_liquid()
            if len(df) >= US_MIN_NAMES:
                tmp = SAVED_UNIVERSE.with_suffix('.tmp')
                df.to_csv(tmp, index=False)
                tmp.replace(SAVED_UNIVERSE)
                self.source_note = f'US liquid list, live ({len(df)} names)'
            else:
                logger.warning(f'US list download gave only {len(df)} names: using the saved list')
                df = None
        except Exception as e:
            logger.warning(f'US list download failed: {e}')
            df = None
        if df is None and SAVED_UNIVERSE.exists():
            try:
                df = pd.read_csv(SAVED_UNIVERSE)
                self.source_note = f"US liquid list, saved {str(df['built'].iloc[0]) if len(df) else '?'} ({len(df)} names)"
            except Exception as e:
                logger.warning(f'Saved US list unreadable: {e}')
                df = None
        if df is None:
            return []
        self.us_meta = {str(r.ticker): {'name': str(r.name), 'sector': str(r.sector) if pd.notna(r.sector) else '',
                                        'industry': str(r.industry) if pd.notna(r.industry) else ''}
                        for r in df.itertuples()}
        return df['ticker'].astype(str).tolist()

    def load_universe(self, source: Optional[str] = None) -> List[str]:
        """
        Load the stock universe based on configuration.

        Args:
            source: Override config source. Options:
                - "sp500": S&P 500 only
                - "sp500_nasdaq100": S&P 500 + Nasdaq 100
                - "russell1000": Russell 1000
                - "russell3000": Russell 3000
                - "all_us": All NYSE/NASDAQ/AMEX
                - "custom": Use custom ticker file
        """
        source = (source or self.config.get("universe.source", "russell1000")).strip()
        logger.info(f"Loading universe: {source}")

        tickers = []

        self.source_note = source
        if not hasattr(self, 'us_meta'):
            self.us_meta = {}
        if source in ("us_liquid", "russell1000", "russell3000"):
            # 3 Nov 2026: the iShares Russell download failed on GitHub Actions every night, and the fallback
            # quietly gave the S&P 500 only. Russell requests now get the US liquid list (same idea, broader).
            tickers = self._load_us_liquid()
            if not tickers:
                logger.warning("US liquid list unavailable: S&P 500 + Nasdaq 100")
                tickers = list(set(self._fetch_sp500() + self._fetch_nasdaq100()))
                self.source_note = 'S&P 500 + Nasdaq 100 (fallback)'
        elif source == "sp500":
            tickers = self._fetch_sp500()
        elif source == "sp500_nasdaq100":
            tickers = list(set(self._fetch_sp500() + self._fetch_nasdaq100()))
        elif source == "ishares_russell1000":
            tickers = self._fetch_russell_from_ishares("IWB")
            if not tickers:
                logger.warning("iShares fallback failed, using S&P 500 + Nasdaq 100")
                tickers = list(set(self._fetch_sp500() + self._fetch_nasdaq100()))
        elif source == "ishares_russell3000":
            tickers = self._fetch_russell_from_ishares("IWV")
            if not tickers:
                logger.warning("iShares IWV failed, falling back to Russell 1000 (IWB)")
                tickers = self._fetch_russell_from_ishares("IWB")
            if not tickers:
                logger.warning("iShares fallback failed, using S&P 500 + Nasdaq 100")
                tickers = list(set(self._fetch_sp500() + self._fetch_nasdaq100()))
        elif source == "all_us":
            tickers = self._fetch_from_nasdaq_api()
        elif source == "custom":
            custom_file = self.config.get("universe.custom_tickers_file")
            if custom_file and Path(custom_file).exists():
                with open(custom_file, "r") as f:
                    tickers = [line.strip() for line in f if line.strip()]
            else:
                logger.error(f"Custom ticker file not found: {custom_file}")

        self._tickers = self._filter_tickers(tickers)
        logger.info(f"Universe loaded: {len(self._tickers)} tickers")
        return self._tickers

    def get_tickers(self) -> List[str]:
        """Get the current ticker list (loads if empty)."""
        if not self._tickers:
            self.load_universe()
        return self._tickers

    def add_tickers(self, tickers: List[str]) -> None:
        """Add tickers to the universe."""
        new_tickers = self._filter_tickers(tickers)
        self._tickers = list(set(self._tickers + new_tickers))

    def remove_tickers(self, tickers: List[str]) -> None:
        """Remove tickers from the universe."""
        remove_set = set(t.upper() for t in tickers)
        self._tickers = [t for t in self._tickers if t.upper() not in remove_set]

    def save_to_file(self, filepath: Path) -> None:
        """Save current universe to a file."""
        with open(filepath, "w") as f:
            for ticker in sorted(self._tickers):
                f.write(f"{ticker}\n")
        logger.info(f"Saved {len(self._tickers)} tickers to {filepath}")

    def load_from_file(self, filepath: Path) -> List[str]:
        """Load universe from a file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Universe file not found: {filepath}")

        with open(filepath, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]

        self._tickers = self._filter_tickers(tickers)
        return self._tickers


_universe: Optional[UniverseManager] = None


def get_universe() -> UniverseManager:
    """Get or create the global universe manager."""
    global _universe
    if _universe is None:
        _universe = UniverseManager()
    return _universe
