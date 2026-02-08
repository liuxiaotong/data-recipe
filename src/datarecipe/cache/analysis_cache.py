"""Analysis caching system for avoiding redundant computations."""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class CacheEntry:
    """Metadata for a cached analysis."""

    dataset_id: str
    cache_key: str  # Hash of dataset state
    created_at: str
    expires_at: str
    dataset_type: str = ""
    sample_count: int = 0
    output_dir: str = ""
    hf_commit: str = ""  # HuggingFace commit hash
    hf_last_modified: str = ""  # Last modified timestamp
    file_count: int = 0
    total_size: int = 0  # bytes

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        try:
            expires = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expires
        except (ValueError, TypeError):
            return True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AnalysisCache:
    """Cache manager for dataset analysis results."""

    def __init__(
        self,
        cache_dir: str | None = None,
        default_ttl_days: int = 7,
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
            default_ttl_days: Default time-to-live for cache entries
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.datarecipe/cache")
        self.default_ttl_days = default_ttl_days
        self.index_path = os.path.join(self.cache_dir, "index.json")
        self.index: dict[str, CacheEntry] = {}
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, encoding="utf-8") as f:
                    data = json.load(f)
                self.index = {k: CacheEntry.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                self.index = {}
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.index = {}

    def _save_index(self):
        """Save cache index to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict() for k, v in self.index.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _get_dataset_key(self, dataset_id: str) -> str:
        """Get normalized key for dataset."""
        return dataset_id.replace("/", "_").replace("\\", "_").lower()

    def _compute_cache_key(
        self,
        dataset_id: str,
        hf_commit: str = None,
        hf_last_modified: str = None,
        sample_size: int = None,
    ) -> str:
        """Compute cache key based on dataset state.

        Args:
            dataset_id: Dataset identifier
            hf_commit: HuggingFace commit hash
            hf_last_modified: Last modified timestamp
            sample_size: Sample size used for analysis

        Returns:
            Hash string representing dataset state
        """
        key_data = f"{dataset_id}:{hf_commit or ''}:{hf_last_modified or ''}:{sample_size or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    def get_hf_metadata(self, dataset_id: str) -> dict[str, str]:
        """Get metadata from HuggingFace for a dataset.

        Args:
            dataset_id: Dataset identifier (e.g., "Anthropic/hh-rlhf")

        Returns:
            Dict with 'commit' and 'last_modified' keys
        """
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            info = api.dataset_info(dataset_id)

            return {
                "commit": info.sha or "",
                "last_modified": info.last_modified.isoformat() if info.last_modified else "",
            }
        except (ImportError, OSError, ValueError):
            return {"commit": "", "last_modified": ""}

    def get(
        self,
        dataset_id: str,
        check_freshness: bool = True,
    ) -> CacheEntry | None:
        """Get cached analysis if available and fresh.

        Args:
            dataset_id: Dataset identifier
            check_freshness: Whether to check if dataset has changed

        Returns:
            CacheEntry if cache hit, None otherwise
        """
        key = self._get_dataset_key(dataset_id)

        if key not in self.index:
            return None

        entry = self.index[key]

        # Check expiration
        if entry.is_expired():
            self.invalidate(dataset_id)
            return None

        # Check if dataset has changed on HuggingFace
        if check_freshness:
            current_meta = self.get_hf_metadata(dataset_id)
            if current_meta.get("commit") and entry.hf_commit:
                if current_meta["commit"] != entry.hf_commit:
                    # Dataset has been updated
                    return None

        # Verify cache files exist
        if entry.output_dir and not os.path.exists(entry.output_dir):
            self.invalidate(dataset_id)
            return None

        return entry

    def put(
        self,
        dataset_id: str,
        output_dir: str,
        dataset_type: str = "",
        sample_count: int = 0,
        ttl_days: int = None,
    ) -> CacheEntry:
        """Store analysis result in cache.

        Args:
            dataset_id: Dataset identifier
            output_dir: Directory containing analysis output
            dataset_type: Type of dataset
            sample_count: Number of samples analyzed
            ttl_days: Time-to-live in days (overrides default)

        Returns:
            Created CacheEntry
        """
        key = self._get_dataset_key(dataset_id)
        ttl = ttl_days or self.default_ttl_days

        # Get HuggingFace metadata
        hf_meta = self.get_hf_metadata(dataset_id)

        # Compute cache key
        cache_key = self._compute_cache_key(
            dataset_id,
            hf_meta.get("commit"),
            hf_meta.get("last_modified"),
            sample_count,
        )

        # Calculate output stats
        file_count = 0
        total_size = 0
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                fpath = os.path.join(output_dir, f)
                if os.path.isfile(fpath):
                    file_count += 1
                    total_size += os.path.getsize(fpath)

        entry = CacheEntry(
            dataset_id=dataset_id,
            cache_key=cache_key,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=ttl)).isoformat(),
            dataset_type=dataset_type,
            sample_count=sample_count,
            output_dir=output_dir,
            hf_commit=hf_meta.get("commit", ""),
            hf_last_modified=hf_meta.get("last_modified", ""),
            file_count=file_count,
            total_size=total_size,
        )

        self.index[key] = entry
        self._save_index()

        return entry

    def invalidate(self, dataset_id: str, delete_files: bool = False):
        """Invalidate cache for a dataset.

        Args:
            dataset_id: Dataset identifier
            delete_files: Whether to delete cached files
        """
        key = self._get_dataset_key(dataset_id)

        if key in self.index:
            entry = self.index[key]

            if delete_files and entry.output_dir and os.path.exists(entry.output_dir):
                shutil.rmtree(entry.output_dir, ignore_errors=True)

            del self.index[key]
            self._save_index()

    def clear_expired(self, delete_files: bool = True) -> int:
        """Clear all expired cache entries.

        Args:
            delete_files: Whether to delete cached files

        Returns:
            Number of entries cleared
        """
        expired_keys = [k for k, v in self.index.items() if v.is_expired()]

        for key in expired_keys:
            entry = self.index[key]
            if delete_files and entry.output_dir and os.path.exists(entry.output_dir):
                shutil.rmtree(entry.output_dir, ignore_errors=True)
            del self.index[key]

        if expired_keys:
            self._save_index()

        return len(expired_keys)

    def clear_all(self, delete_files: bool = True):
        """Clear entire cache.

        Args:
            delete_files: Whether to delete cached files
        """
        if delete_files:
            for entry in self.index.values():
                if entry.output_dir and os.path.exists(entry.output_dir):
                    shutil.rmtree(entry.output_dir, ignore_errors=True)

        self.index = {}
        self._save_index()

    def list_entries(self) -> list[CacheEntry]:
        """List all cache entries.

        Returns:
            List of CacheEntry objects
        """
        return list(self.index.values())

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        entries = list(self.index.values())
        expired = sum(1 for e in entries if e.is_expired())
        total_size = sum(e.total_size for e in entries)

        return {
            "total_entries": len(entries),
            "expired_entries": expired,
            "valid_entries": len(entries) - expired,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "cache_dir": self.cache_dir,
            "datasets": [e.dataset_id for e in entries],
        }

    def copy_to_output(self, dataset_id: str, output_dir: str) -> bool:
        """Copy cached results to specified output directory.

        Args:
            dataset_id: Dataset identifier
            output_dir: Target output directory

        Returns:
            True if successful, False otherwise
        """
        entry = self.get(dataset_id, check_freshness=False)
        if not entry or not entry.output_dir:
            return False

        if not os.path.exists(entry.output_dir):
            return False

        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(entry.output_dir):
            src = os.path.join(entry.output_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)

        return True
