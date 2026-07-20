"""
universe.py — Stock universe management
========================================
Fetches and manages the list of US equities to scan.
Handles filtering of ETFs, preferred shares, warrants, etc.
"""

import logging
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
    "-A",    # Class shares (keep these)
    "-B",
    "-C",
    ".PR",   # Preferred
    "-P",
}


class UniverseManager:
    """Manages the stock universe for scanning."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self._tickers: List[str] = []
        self._ticker_info: Dict[str, Dict] = {}

    def _fetch_sp500(self) -> List[str]:
        """Fetch S&P 500 constituents from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
            logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500: {e}")
            return []

    def _fetch_nasdaq100(self) -> List[str]:
        """Fetch Nasdaq 100 constituents from Wikipedia."""
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        try:
            tables = pd.read_html(url)
            for table in tables:
                if "Ticker" in table.columns or "Symbol" in table.columns:
                    col = "Ticker" if "Ticker" in table.columns else "Symbol"
                    tickers = table[col].str.replace(".", "-", regex=False).tolist()
                    logger.info(f"Fetched {len(tickers)} Nasdaq 100 tickers")
                    return tickers
            return []
        except Exception as e:
            logger.error(f"Failed to fetch Nasdaq 100: {e}")
            return []

    def _fetch_russell_from_ishares(self, etf: str = "IWV") -> List[str]:
        """
        Fetch Russell universe from iShares ETF holdings.
        IWV = Russell 3000
        IWB = Russell 1000
        IWM = Russell 2000
        """
        url = f"https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName={etf}_holdings&dataType=fund"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()

            lines = resp.text.split("\n")
            start_idx = 0
            for i, line in enumerate(lines):
                if "Ticker" in line:
                    start_idx = i
                    break

            csv_data = "\n".join(lines[start_idx:])
            df = pd.read_csv(StringIO(csv_data))

            if "Ticker" in df.columns:
                tickers = df["Ticker"].dropna().tolist()
                tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
                tickers = [t.replace(".", "-") for t in tickers]
                logger.info(f"Fetched {len(tickers)} tickers from iShares {etf}")
                return tickers
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
        source = source or self.config.get("universe.source", "russell3000")
        logger.info(f"Loading universe: {source}")

        tickers = []

        if source == "sp500":
            tickers = self._fetch_sp500()
        elif source == "sp500_nasdaq100":
            tickers = list(set(self._fetch_sp500() + self._fetch_nasdaq100()))
        elif source == "russell1000":
            tickers = self._fetch_russell_from_ishares("IWB")
            if not tickers:
                logger.warning("iShares fallback failed, using S&P 500 + Nasdaq 100")
                tickers = list(set(self._fetch_sp500() + self._fetch_nasdaq100()))
        elif source == "russell3000":
            tickers = self._fetch_russell_from_ishares("IWV")
            if not tickers:
                logger.warning("iShares fallback failed, using NASDAQ API")
                tickers = self._fetch_from_nasdaq_api()
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
