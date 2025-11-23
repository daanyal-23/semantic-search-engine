# src/cache_manager.py
"""
Cache Manager for storing and loading document embeddings.

According to the assignment requirements:
- Cache stores: doc_id, embedding, hash, updated_at
- If a document's hash hasn't changed, reuse the cached embedding
- If the hash changed, regenerate the embedding

We use JSON for simplicity, readability, and portability.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# Cache file location (ignored by Git)
CACHE_PATH = Path("cache/embeddings.json")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


class CacheManager:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self._data: Dict[str, Any] = {}
        self._load()

    # ---------------------------------------------------------
    # INTERNAL LOAD/SAVE
    # ---------------------------------------------------------
    def _load(self):
        """Load cache JSON if it exists."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except json.JSONDecodeError:
                # If cache is corrupted, reset it
                self._data = {}
        else:
            self._data = {}

    def save(self):
        """Write all cached entries back to JSON."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ---------------------------------------------------------
    # BASIC OPERATIONS
    # ---------------------------------------------------------
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return cache entry for doc_id, or None if missing."""
        return self._data.get(doc_id)

    def set(self, doc_id: str, embedding: List[float], hash_str: str):
        """Add or update a cache entry."""
        self._data[doc_id] = {
            "doc_id": doc_id,
            "embedding": embedding,
            "hash": hash_str,
            "updated_at": int(time.time())
        }
        self.save()

    # ---------------------------------------------------------
    # BULK VALIDATION
    # ---------------------------------------------------------
    def bulk_get_changed(self, metas: List[Dict[str, Any]]) -> Dict[str, Optional[Dict]]:
        """
        For each metadata entry:
            If document is cached AND hash matches   → return cached entry
            If missing OR hash changed               → return None
        Returns dict: {doc_id: cache_entry_or_None}
        """
        result = {}
        for meta in metas:
            doc_id = meta["doc_id"]
            cached = self.get(doc_id)
            if cached and cached.get("hash") == meta["hash"]:
                result[doc_id] = cached     # use cached vector
            else:
                result[doc_id] = None       # needs re-embedding
        return result

    # ---------------------------------------------------------
    # CLEAR ENTIRE CACHE (YOU JUST FIXED THIS)
    # ---------------------------------------------------------
    def clear(self):
        """Remove ALL cached embeddings."""
        self._data = {}     # reset actual cache dictionary
        self.save()         # persist as empty file
        print("Cache cleared.")
