"""
cache.py — Local caching for scanner data
==========================================
Implements file-based caching with TTL support for price and fundamental data.
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Dict
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"


class DataCache:
    """File-based cache with TTL support."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._metadata_file = self.cache_dir / "_metadata.json"
        self._metadata: Dict[str, Any] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, key: str) -> str:
        """Generate a filesystem-safe cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, key: str, extension: str = "pkl") -> Path:
        """Get the file path for a cache entry."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.{extension}"

    def is_valid(self, key: str) -> bool:
        """Check if a cache entry exists and is not expired."""
        cache_key = self._get_cache_key(key)
        if cache_key not in self._metadata:
            return False

        entry = self._metadata[cache_key]
        cached_time = datetime.fromisoformat(entry["timestamp"])
        return datetime.now() - cached_time < self.ttl

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve a cached DataFrame."""
        if not self.is_valid(key):
            return None

        cache_path = self._get_cache_path(key, "pkl")
        if not cache_path.exists():
            return None

        try:
            df = pd.read_pickle(cache_path)
            logger.debug(f"Cache hit for: {key}")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None

    def set_dataframe(self, key: str, df: pd.DataFrame) -> None:
        """Store a DataFrame in cache."""
        cache_path = self._get_cache_path(key, "pkl")
        cache_key = self._get_cache_key(key)

        try:
            df.to_pickle(cache_path)
            self._metadata[cache_key] = {
                "key": key,
                "timestamp": datetime.now().isoformat(),
                "rows": len(df),
                "columns": list(df.columns),
            }
            self._save_metadata()
            logger.debug(f"Cached: {key} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached JSON data."""
        if not self.is_valid(key):
            return None

        cache_path = self._get_cache_path(key, "json")
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            logger.debug(f"Cache hit for: {key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None

    def set_json(self, key: str, data: Dict[str, Any]) -> None:
        """Store JSON data in cache."""
        cache_path = self._get_cache_path(key, "json")
        cache_key = self._get_cache_key(key)

        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self._metadata[cache_key] = {
                "key": key,
                "timestamp": datetime.now().isoformat(),
                "type": "json",
            }
            self._save_metadata()
            logger.debug(f"Cached JSON: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")

    def invalidate(self, key: str) -> None:
        """Remove a specific cache entry."""
        cache_key = self._get_cache_key(key)
        for ext in ["pkl", "json"]:
            cache_path = self._get_cache_path(key, ext)
            if cache_path.exists():
                cache_path.unlink()

        if cache_key in self._metadata:
            del self._metadata[cache_key]
            self._save_metadata()
        logger.debug(f"Invalidated cache: {key}")

    def clear_all(self) -> None:
        """Clear all cached data."""
        for f in self.cache_dir.glob("*"):
            if f.is_file() and f.name != "_metadata.json":
                f.unlink()
        self._metadata = {}
        self._save_metadata()
        logger.info("Cleared all cache")

    def clear_expired(self) -> int:
        """Remove all expired cache entries. Returns count of removed entries."""
        expired_keys = []
        for cache_key, entry in self._metadata.items():
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - cached_time >= self.ttl:
                expired_keys.append(entry.get("key", cache_key))

        for key in expired_keys:
            self.invalidate(key)

        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file()
        )
        valid_count = sum(1 for key in self._metadata if self.is_valid(
            self._metadata[key].get("key", "")
        ))

        return {
            "total_entries": len(self._metadata),
            "valid_entries": valid_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl.total_seconds() / 3600,
        }
