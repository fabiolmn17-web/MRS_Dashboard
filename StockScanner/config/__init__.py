"""Configuration module for CAN SLIM Scanner."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import os

CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = CONFIG_DIR / "scanner_config.yaml"


class ScannerConfig:
    """Load and access scanner configuration with environment variable support."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_FILE
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Example: config.get('technical.sma_periods.short') -> 50
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key from environment variable.

        Supported providers:
        - fmp: FMP_API_KEY
        - polygon: POLYGON_API_KEY
        - alpha_vantage: ALPHA_VANTAGE_API_KEY
        """
        env_vars = {
            "fmp": "FMP_API_KEY",
            "polygon": "POLYGON_API_KEY",
            "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
        }
        env_var = env_vars.get(provider.lower())
        if env_var:
            return os.environ.get(env_var)
        return None

    @property
    def data_provider(self) -> str:
        """Primary data provider name."""
        return self.get("data_provider.primary", "yfinance")

    @property
    def cache_enabled(self) -> bool:
        """Whether caching is enabled."""
        return self.get("data_provider.cache_enabled", True)

    @property
    def cache_ttl_hours(self) -> int:
        """Cache time-to-live in hours."""
        return self.get("data_provider.cache_ttl_hours", 24)

    # ── Technical Settings ──────────────────────────────────────────────────────

    @property
    def sma_short(self) -> int:
        return self.get("technical.sma_periods.short", 50)

    @property
    def sma_medium(self) -> int:
        return self.get("technical.sma_periods.medium", 100)

    @property
    def sma_long(self) -> int:
        return self.get("technical.sma_periods.long", 200)

    @property
    def max_distance_from_ath(self) -> float:
        return self.get("technical.max_distance_from_ath", -0.20)

    # ── Liquidity Settings ──────────────────────────────────────────────────────

    @property
    def min_avg_volume(self) -> int:
        return self.get("liquidity.min_avg_volume", 500000)

    @property
    def volume_lookback_days(self) -> int:
        return self.get("liquidity.volume_lookback_days", 50)

    # ── Relative Strength Settings ──────────────────────────────────────────────

    @property
    def benchmark_ticker(self) -> str:
        return self.get("relative_strength.benchmark_ticker", "SPY")

    @property
    def rs_sma_period(self) -> int:
        return self.get("relative_strength.sma_period", 100)

    @property
    def min_rs_ratio(self) -> float:
        return self.get("relative_strength.min_rs_ratio", 2.0)

    @property
    def rs_score_thresholds(self) -> Dict[str, float]:
        return self.get("relative_strength.score_thresholds", {
            "score_3": 3.0,
            "score_2": 2.0,
            "score_1": 0.0,
        })

    # ── Strict CAN SLIM Settings ────────────────────────────────────────────────

    @property
    def strict_quarterly_eps_growth_min(self) -> float:
        return self.get("strict_canslim.quarterly_eps_growth_min", 0.25)

    @property
    def strict_quarterly_revenue_growth_min(self) -> float:
        return self.get("strict_canslim.quarterly_revenue_growth_min", 0.25)

    @property
    def strict_annual_eps_cagr_min(self) -> float:
        return self.get("strict_canslim.annual_eps_cagr_min", 0.25)

    @property
    def strict_roe_min(self) -> float:
        return self.get("strict_canslim.roe_min", 0.17)

    # ── Relaxed CAN SLIM Settings ───────────────────────────────────────────────

    @property
    def relaxed_quarterly_eps_growth_min(self) -> float:
        return self.get("relaxed_canslim.quarterly_eps_growth_min", 0.20)

    @property
    def relaxed_quarterly_revenue_growth_min(self) -> float:
        return self.get("relaxed_canslim.quarterly_revenue_growth_min", 0.15)

    @property
    def relaxed_annual_eps_cagr_min(self) -> float:
        return self.get("relaxed_canslim.annual_eps_cagr_min", 0.15)

    @property
    def relaxed_roe_min(self) -> float:
        return self.get("relaxed_canslim.roe_min", 0.15)

    # ── Turnaround Settings ─────────────────────────────────────────────────────

    @property
    def turnaround_enabled(self) -> bool:
        return self.get("turnaround.enabled", False)

    # ── Pattern Settings ────────────────────────────────────────────────────────

    @property
    def patterns_enabled(self) -> bool:
        return self.get("patterns.enabled", True)

    @property
    def atr_compression_enabled(self) -> bool:
        return self.get("patterns.atr_compression.enabled", True)

    @property
    def higher_lows_enabled(self) -> bool:
        return self.get("patterns.higher_lows.enabled", True)

    # ── Output Settings ─────────────────────────────────────────────────────────

    @property
    def output_directory(self) -> str:
        return self.get("output.directory", "output")

    @property
    def sort_order(self) -> list:
        return self.get("output.sort_order", [])

    def __repr__(self) -> str:
        return f"ScannerConfig(config_path={self.config_path})"


# Global config instance (lazy loaded)
_config: Optional[ScannerConfig] = None


def get_config(config_path: Optional[Path] = None) -> ScannerConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None or config_path is not None:
        _config = ScannerConfig(config_path)
    return _config
