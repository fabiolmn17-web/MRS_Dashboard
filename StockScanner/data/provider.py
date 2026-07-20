"""
provider.py — Data provider abstraction
========================================
Unified interface for fetching price and fundamental data.
Supports yfinance (default), with architecture for FMP/Polygon expansion.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from ..config import get_config, ScannerConfig
from .cache import DataCache

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        days: int = 300,
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex.
        If adjusted=True, returns split/dividend adjusted prices.
        """
        pass

    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch fundamental data for a stock.

        Returns dict with standardized keys:
        - quarterly_eps: List of (date, eps) tuples
        - quarterly_revenue: List of (date, revenue) tuples
        - annual_eps: List of (year, eps) tuples
        - annual_revenue: List of (year, revenue) tuples
        - roe: float
        - ttm_net_income: float
        - market_cap: float
        - sector: str
        - industry: str
        - exchange: str
        - name: str
        """
        pass

    @abstractmethod
    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch basic stock info (name, sector, exchange, etc.)."""
        pass

    @abstractmethod
    def get_batch_prices(
        self,
        tickers: List[str],
        days: int = 300,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch price history for multiple tickers."""
        pass


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance library."""

    def __init__(self, config: Optional[ScannerConfig] = None):
        self.config = config or get_config()
        self.cache = DataCache(ttl_hours=self.config.cache_ttl_hours)
        self._rate_limit_delay = 60 / self.config.get(
            "data_provider.rate_limit_requests_per_minute", 100
        )
        self._retry_attempts = self.config.get("data_provider.retry_attempts", 3)
        self._retry_delay = self.config.get("data_provider.retry_delay_seconds", 2)
        self._last_request_time = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with retry and exponential backoff."""
        last_error = None
        for attempt in range(self._retry_attempts):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self._retry_attempts - 1:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.debug(f"Retry {attempt + 1}/{self._retry_attempts} after {delay}s: {e}")
                    time.sleep(delay)

        logger.warning(f"All {self._retry_attempts} attempts failed: {last_error}")
        return None

    def get_price_history(
        self,
        ticker: str,
        days: int = 300,
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical price data from Yahoo Finance."""
        import yfinance as yf

        cache_key = f"price_{ticker}_{days}_{adjusted}"
        if self.config.cache_enabled:
            cached = self.cache.get_dataframe(cache_key)
            if cached is not None:
                return cached

        def fetch():
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days * 1.5))
            hist = stock.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=adjusted,
            )
            if hist.empty:
                return None

            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in hist.columns:
                    hist[col] = np.nan

            return hist[required_cols].dropna(subset=["Close"])

        df = self._retry_with_backoff(fetch)
        if df is not None and self.config.cache_enabled:
            self.cache.set_dataframe(cache_key, df)

        return df

    def get_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch fundamental data from Yahoo Finance."""
        import yfinance as yf

        cache_key = f"fundamentals_{ticker}"
        if self.config.cache_enabled:
            cached = self.cache.get_json(cache_key)
            if cached is not None:
                return cached

        def fetch():
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            result = {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "market_cap": info.get("marketCap"),
                "quote_type": info.get("quoteType"),
                "roe": info.get("returnOnEquity"),
                "trailing_eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "ttm_net_income": None,
                "quarterly_eps": [],
                "quarterly_revenue": [],
                "annual_eps": [],
                "annual_revenue": [],
                "operating_margin": info.get("operatingMargins"),
                "profit_margin": info.get("profitMargins"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "institutional_ownership": info.get("heldPercentInstitutions"),
                "operating_cash_flow": info.get("operatingCashflow"),
                "data_source": "yfinance",
                "fetch_timestamp": datetime.now().isoformat(),
                "missing_fields": [],
            }

            try:
                quarterly = stock.quarterly_financials
                if quarterly is not None and not quarterly.empty:
                    for col in quarterly.columns:
                        date_str = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
                        eps = self._safe_get_financial(quarterly, col, [
                            "Diluted EPS", "Basic EPS", "Earnings Per Share"
                        ])
                        rev = self._safe_get_financial(quarterly, col, [
                            "Total Revenue", "Revenue", "Net Sales"
                        ])
                        if eps is not None:
                            result["quarterly_eps"].append((date_str, eps))
                        if rev is not None:
                            result["quarterly_revenue"].append((date_str, rev))
            except Exception as e:
                logger.debug(f"{ticker}: quarterly financials error: {e}")
                result["missing_fields"].append("quarterly_financials")

            try:
                annual = stock.financials
                if annual is not None and not annual.empty:
                    for col in annual.columns:
                        year = col.year if hasattr(col, "year") else str(col)[:4]
                        eps = self._safe_get_financial(annual, col, [
                            "Diluted EPS", "Basic EPS", "Earnings Per Share"
                        ])
                        rev = self._safe_get_financial(annual, col, [
                            "Total Revenue", "Revenue", "Net Sales"
                        ])
                        net_income = self._safe_get_financial(annual, col, [
                            "Net Income", "Net Income Common Stockholders"
                        ])
                        if eps is not None:
                            result["annual_eps"].append((year, eps))
                        if rev is not None:
                            result["annual_revenue"].append((year, rev))
                        if net_income is not None and result["ttm_net_income"] is None:
                            result["ttm_net_income"] = net_income
            except Exception as e:
                logger.debug(f"{ticker}: annual financials error: {e}")
                result["missing_fields"].append("annual_financials")

            try:
                income = stock.income_stmt
                if income is not None and not income.empty:
                    latest_col = income.columns[0]
                    if result["ttm_net_income"] is None:
                        result["ttm_net_income"] = self._safe_get_financial(
                            income, latest_col, ["Net Income", "Net Income Common Stockholders"]
                        )
            except Exception:
                pass

            if not result["quarterly_eps"]:
                result["missing_fields"].append("quarterly_eps")
            if not result["quarterly_revenue"]:
                result["missing_fields"].append("quarterly_revenue")
            if not result["annual_eps"]:
                result["missing_fields"].append("annual_eps")
            if result["roe"] is None:
                result["missing_fields"].append("roe")

            return result

        fundamentals = self._retry_with_backoff(fetch)
        if fundamentals is not None and self.config.cache_enabled:
            self.cache.set_json(cache_key, fundamentals)

        return fundamentals

    def _safe_get_financial(
        self,
        df: pd.DataFrame,
        col: Any,
        possible_rows: List[str],
    ) -> Optional[float]:
        """Safely extract a financial value, trying multiple possible row names."""
        for row_name in possible_rows:
            try:
                if row_name in df.index:
                    val = df.loc[row_name, col]
                    if pd.notna(val):
                        return float(val)
            except Exception:
                continue
        return None

    def get_stock_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch basic stock info."""
        import yfinance as yf

        cache_key = f"info_{ticker}"
        if self.config.cache_enabled:
            cached = self.cache.get_json(cache_key)
            if cached is not None:
                return cached

        def fetch():
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            return {
                "ticker": ticker,
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "market_cap": info.get("marketCap"),
                "quote_type": info.get("quoteType"),
            }

        result = self._retry_with_backoff(fetch)
        if result is not None and self.config.cache_enabled:
            self.cache.set_json(cache_key, result)

        return result

    def get_batch_prices(
        self,
        tickers: List[str],
        days: int = 300,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch price history for multiple tickers.

        Uses yfinance's batch download for efficiency, then validates each.
        """
        import yfinance as yf

        results = {}
        missing = []

        for ticker in tickers:
            cache_key = f"price_{ticker}_{days}_True"
            if self.config.cache_enabled:
                cached = self.cache.get_dataframe(cache_key)
                if cached is not None:
                    results[ticker] = cached
                    continue
            missing.append(ticker)

        if not missing:
            return results

        batch_size = 50
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            logger.debug(f"Fetching batch {i // batch_size + 1}: {len(batch)} tickers")

            try:
                self._rate_limit()
                end_date = datetime.now()
                start_date = end_date - timedelta(days=int(days * 1.5))

                data = yf.download(
                    batch,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )

                if data.empty:
                    continue

                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            df = data.copy()
                        else:
                            df = data.xs(ticker, axis=1, level=1) if isinstance(
                                data.columns, pd.MultiIndex
                            ) else data

                        if df.empty or "Close" not in df.columns:
                            continue

                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

                        if not df.empty:
                            results[ticker] = df
                            if self.config.cache_enabled:
                                cache_key = f"price_{ticker}_{days}_True"
                                self.cache.set_dataframe(cache_key, df)

                    except Exception as e:
                        logger.debug(f"Failed to extract {ticker} from batch: {e}")

            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
                for ticker in batch:
                    df = self.get_price_history(ticker, days)
                    if df is not None:
                        results[ticker] = df

        return results


_provider: Optional[DataProvider] = None


def get_provider(provider_name: Optional[str] = None) -> DataProvider:
    """Get or create the data provider instance."""
    global _provider
    config = get_config()
    name = provider_name or config.data_provider

    if _provider is None or (provider_name and provider_name != config.data_provider):
        if name == "yfinance":
            _provider = YFinanceProvider(config)
        else:
            logger.warning(f"Unknown provider '{name}', falling back to yfinance")
            _provider = YFinanceProvider(config)

    return _provider
