"""
src/cache_manager.py — JSON result cache management for TalentForge.

Provides simple save/load/exists operations for pre-computed demo results,
enabling Demo Mode with zero LLM API calls.
"""

import json
import os
from typing import Any


class CacheManager:
    """
    Manages serialization and retrieval of pre-computed pipeline results.

    All data is stored as pretty-printed JSON for human readability
    and easy inspection by judges.
    """

    def save_results(self, path: str, data: Any) -> None:
        """
        Save data to a JSON file.

        Args:
            path: Absolute file path to write to (including .json extension).
            data: JSON-serializable Python object.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Cache saved: {path} ({os.path.getsize(path):,} bytes)")

    def load_results(self, path: str) -> Any:
        """
        Load and parse a JSON cache file.

        Args:
            path: Absolute file path to read from.

        Returns:
            Parsed Python object (dict, list, etc.).

        Raises:
            FileNotFoundError: If the cache file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cache file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def exists(self, path: str) -> bool:
        """
        Check whether a cache file exists.

        Args:
            path: Absolute file path to check.

        Returns:
            True if the file exists, False otherwise.
        """
        return os.path.exists(path)
