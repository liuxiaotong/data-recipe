"""Unit tests for AnalysisCache system.

Tests CacheEntry dataclass, cache CRUD operations,
TTL/expiration, key normalization, and index persistence.
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from datarecipe.cache.analysis_cache import AnalysisCache, CacheEntry


class TestCacheEntryDataclass(unittest.TestCase):
    """Test CacheEntry dataclass."""

    def test_create_entry(self):
        entry = CacheEntry(
            dataset_id="test/dataset",
            cache_key="abc123",
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=7)).isoformat(),
        )
        self.assertEqual(entry.dataset_id, "test/dataset")
        self.assertEqual(entry.cache_key, "abc123")

    def test_not_expired(self):
        entry = CacheEntry(
            dataset_id="test",
            cache_key="abc",
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(days=7)).isoformat(),
        )
        self.assertFalse(entry.is_expired())

    def test_expired(self):
        entry = CacheEntry(
            dataset_id="test",
            cache_key="abc",
            created_at=(datetime.now() - timedelta(days=14)).isoformat(),
            expires_at=(datetime.now() - timedelta(days=7)).isoformat(),
        )
        self.assertTrue(entry.is_expired())

    def test_invalid_expires_at_is_expired(self):
        entry = CacheEntry(
            dataset_id="test",
            cache_key="abc",
            created_at="",
            expires_at="not-a-date",
        )
        self.assertTrue(entry.is_expired())

    def test_to_dict(self):
        entry = CacheEntry(
            dataset_id="test/ds",
            cache_key="abc123",
            created_at="2025-01-01T00:00:00",
            expires_at="2025-01-08T00:00:00",
            dataset_type="preference",
            sample_count=1000,
        )
        d = entry.to_dict()
        self.assertEqual(d["dataset_id"], "test/ds")
        self.assertEqual(d["cache_key"], "abc123")
        self.assertEqual(d["dataset_type"], "preference")
        self.assertEqual(d["sample_count"], 1000)

    def test_from_dict(self):
        d = {
            "dataset_id": "test/ds",
            "cache_key": "abc123",
            "created_at": "2025-01-01T00:00:00",
            "expires_at": "2025-01-08T00:00:00",
            "dataset_type": "sft",
            "sample_count": 500,
        }
        entry = CacheEntry.from_dict(d)
        self.assertEqual(entry.dataset_id, "test/ds")
        self.assertEqual(entry.dataset_type, "sft")
        self.assertEqual(entry.sample_count, 500)

    def test_from_dict_ignores_unknown_fields(self):
        d = {
            "dataset_id": "test",
            "cache_key": "abc",
            "created_at": "2025-01-01",
            "expires_at": "2025-01-08",
            "unknown_field": "should be ignored",
        }
        entry = CacheEntry.from_dict(d)
        self.assertEqual(entry.dataset_id, "test")
        self.assertFalse(hasattr(entry, "unknown_field"))

    def test_roundtrip_to_dict_from_dict(self):
        original = CacheEntry(
            dataset_id="org/dataset",
            cache_key="hash123",
            created_at="2025-01-01T00:00:00",
            expires_at="2025-01-08T00:00:00",
            dataset_type="evaluation",
            sample_count=2000,
            output_dir="/tmp/cache/test",
            hf_commit="abc",
            file_count=10,
            total_size=1024,
        )
        d = original.to_dict()
        restored = CacheEntry.from_dict(d)
        self.assertEqual(original.dataset_id, restored.dataset_id)
        self.assertEqual(original.cache_key, restored.cache_key)
        self.assertEqual(original.sample_count, restored.sample_count)
        self.assertEqual(original.total_size, restored.total_size)


class TestAnalysisCacheKeyGeneration(unittest.TestCase):
    """Test cache key normalization and hashing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = AnalysisCache(cache_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dataset_key_normalization(self):
        self.assertEqual(self.cache._get_dataset_key("Anthropic/hh-rlhf"), "anthropic_hh-rlhf")
        self.assertEqual(self.cache._get_dataset_key("org\\dataset"), "org_dataset")
        self.assertEqual(self.cache._get_dataset_key("simple"), "simple")

    def test_dataset_key_lowercase(self):
        self.assertEqual(self.cache._get_dataset_key("ORG/Dataset"), "org_dataset")

    def test_cache_key_is_12_chars(self):
        key = self.cache._compute_cache_key("test/ds", "commit1", "2025-01-01", 1000)
        self.assertEqual(len(key), 12)

    def test_cache_key_deterministic(self):
        key1 = self.cache._compute_cache_key("test/ds", "c1", "2025-01-01", 1000)
        key2 = self.cache._compute_cache_key("test/ds", "c1", "2025-01-01", 1000)
        self.assertEqual(key1, key2)

    def test_cache_key_changes_with_commit(self):
        key1 = self.cache._compute_cache_key("test/ds", "commit1", "2025-01-01", 1000)
        key2 = self.cache._compute_cache_key("test/ds", "commit2", "2025-01-01", 1000)
        self.assertNotEqual(key1, key2)

    def test_cache_key_handles_none(self):
        key = self.cache._compute_cache_key("test/ds")
        self.assertEqual(len(key), 12)


class TestAnalysisCacheCRUD(unittest.TestCase):
    """Test cache CRUD operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_dir)

        # Create some files in output dir
        with open(os.path.join(self.output_dir, "test.md"), "w") as f:
            f.write("# Test output")

        self.cache = AnalysisCache(cache_dir=self.cache_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_put_and_get(self, _mock_hf):
        entry = self.cache.put(
            dataset_id="test/dataset",
            output_dir=self.output_dir,
            dataset_type="preference",
            sample_count=1000,
        )
        self.assertIsInstance(entry, CacheEntry)
        self.assertEqual(entry.dataset_id, "test/dataset")
        self.assertEqual(entry.sample_count, 1000)
        self.assertEqual(entry.file_count, 1)  # One file created
        self.assertGreater(entry.total_size, 0)

        # Retrieve
        cached = self.cache.get("test/dataset", check_freshness=False)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.dataset_id, "test/dataset")

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_get_missing_returns_none(self, _mock_hf):
        cached = self.cache.get("nonexistent/dataset", check_freshness=False)
        self.assertIsNone(cached)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_invalidate(self, _mock_hf):
        self.cache.put("test/ds", self.output_dir)
        self.assertIsNotNone(self.cache.get("test/ds", check_freshness=False))

        self.cache.invalidate("test/ds")
        self.assertIsNone(self.cache.get("test/ds", check_freshness=False))

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_invalidate_with_delete_files(self, _mock_hf):
        out_dir = os.path.join(self.tmpdir, "to_delete")
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, "file.txt"), "w") as f:
            f.write("data")

        self.cache.put("test/ds", out_dir)
        self.cache.invalidate("test/ds", delete_files=True)
        self.assertFalse(os.path.exists(out_dir))

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_list_entries(self, _mock_hf):
        self.cache.put("test/ds1", self.output_dir)
        self.cache.put("test/ds2", self.output_dir)
        entries = self.cache.list_entries()
        self.assertEqual(len(entries), 2)
        ids = [e.dataset_id for e in entries]
        self.assertIn("test/ds1", ids)
        self.assertIn("test/ds2", ids)


class TestAnalysisCacheTTL(unittest.TestCase):
    """Test TTL and expiration logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_dir)
        self.cache = AnalysisCache(cache_dir=self.cache_dir, default_ttl_days=7)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_default_ttl_7_days(self, _mock_hf):
        entry = self.cache.put("test/ds", self.output_dir)
        expires = datetime.fromisoformat(entry.expires_at)
        created = datetime.fromisoformat(entry.created_at)
        delta = expires - created
        self.assertAlmostEqual(delta.days, 7, delta=1)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_custom_ttl(self, _mock_hf):
        entry = self.cache.put("test/ds", self.output_dir, ttl_days=30)
        expires = datetime.fromisoformat(entry.expires_at)
        created = datetime.fromisoformat(entry.created_at)
        delta = expires - created
        self.assertAlmostEqual(delta.days, 30, delta=1)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_expired_entry_returns_none(self, _mock_hf):
        # Put entry
        self.cache.put("test/ds", self.output_dir)

        # Manually expire it
        key = self.cache._get_dataset_key("test/ds")
        entry = self.cache.index[key]
        entry.expires_at = (datetime.now() - timedelta(days=1)).isoformat()

        # Get should return None and invalidate
        cached = self.cache.get("test/ds", check_freshness=False)
        self.assertIsNone(cached)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_clear_expired(self, _mock_hf):
        # Add two entries
        self.cache.put("test/ds1", self.output_dir)
        self.cache.put("test/ds2", self.output_dir)

        # Expire one
        key1 = self.cache._get_dataset_key("test/ds1")
        self.cache.index[key1].expires_at = (datetime.now() - timedelta(days=1)).isoformat()

        count = self.cache.clear_expired(delete_files=False)
        self.assertEqual(count, 1)
        self.assertEqual(len(self.cache.list_entries()), 1)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_clear_all(self, _mock_hf):
        self.cache.put("test/ds1", self.output_dir)
        self.cache.put("test/ds2", self.output_dir)

        self.cache.clear_all(delete_files=False)
        self.assertEqual(len(self.cache.list_entries()), 0)


class TestAnalysisCacheFreshness(unittest.TestCase):
    """Test HuggingFace freshness checking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_dir)
        self.cache = AnalysisCache(cache_dir=self.cache_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "abc123", "last_modified": "2025-01-01"})
    def test_stale_commit_returns_none(self, mock_hf):
        self.cache.put("test/ds", self.output_dir)
        # Now mock a different commit for freshness check
        mock_hf.return_value = {"commit": "different_commit", "last_modified": "2025-01-02"}
        cached = self.cache.get("test/ds", check_freshness=True)
        self.assertIsNone(cached)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "abc123", "last_modified": "2025-01-01"})
    def test_same_commit_returns_entry(self, _mock_hf):
        self.cache.put("test/ds", self.output_dir)
        cached = self.cache.get("test/ds", check_freshness=True)
        self.assertIsNotNone(cached)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_no_commit_skips_freshness_check(self, _mock_hf):
        self.cache.put("test/ds", self.output_dir)
        cached = self.cache.get("test/ds", check_freshness=True)
        self.assertIsNotNone(cached)


class TestAnalysisCacheStats(unittest.TestCase):
    """Test get_stats() method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_dir)
        self.cache = AnalysisCache(cache_dir=self.cache_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_stats_empty_cache(self, _mock_hf):
        stats = self.cache.get_stats()
        self.assertEqual(stats["total_entries"], 0)
        self.assertEqual(stats["valid_entries"], 0)
        self.assertEqual(stats["expired_entries"], 0)
        self.assertEqual(stats["cache_dir"], self.cache_dir)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_stats_with_entries(self, _mock_hf):
        self.cache.put("test/ds1", self.output_dir, sample_count=100)
        self.cache.put("test/ds2", self.output_dir, sample_count=200)
        stats = self.cache.get_stats()
        self.assertEqual(stats["total_entries"], 2)
        self.assertEqual(stats["valid_entries"], 2)
        self.assertIn("test/ds1", stats["datasets"])
        self.assertIn("test/ds2", stats["datasets"])


class TestAnalysisCachePersistence(unittest.TestCase):
    """Test index persistence across instances."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_index_persists_across_instances(self, _mock_hf):
        # Instance 1: put entry
        cache1 = AnalysisCache(cache_dir=self.cache_dir)
        cache1.put("test/ds", self.output_dir, dataset_type="sft")

        # Instance 2: read entry
        cache2 = AnalysisCache(cache_dir=self.cache_dir)
        entry = cache2.get("test/ds", check_freshness=False)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.dataset_type, "sft")

    def test_corrupt_index_gracefully_handled(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(os.path.join(self.cache_dir, "index.json"), "w") as f:
            f.write("corrupt json{{{")

        cache = AnalysisCache(cache_dir=self.cache_dir)
        self.assertEqual(len(cache.list_entries()), 0)


class TestAnalysisCacheCopyToOutput(unittest.TestCase):
    """Test copy_to_output() method."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmpdir, "cache")
        self.src_dir = os.path.join(self.tmpdir, "source")
        self.dst_dir = os.path.join(self.tmpdir, "dest")
        os.makedirs(self.src_dir)

        # Create source files
        with open(os.path.join(self.src_dir, "report.md"), "w") as f:
            f.write("# Report")
        with open(os.path.join(self.src_dir, "data.json"), "w") as f:
            json.dump({"key": "value"}, f)

        self.cache = AnalysisCache(cache_dir=self.cache_dir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_copy_to_output_success(self, _mock_hf):
        self.cache.put("test/ds", self.src_dir)
        result = self.cache.copy_to_output("test/ds", self.dst_dir)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.dst_dir, "report.md")))
        self.assertTrue(os.path.exists(os.path.join(self.dst_dir, "data.json")))

    def test_copy_missing_dataset_returns_false(self):
        result = self.cache.copy_to_output("nonexistent/ds", self.dst_dir)
        self.assertFalse(result)

    @patch.object(AnalysisCache, "get_hf_metadata", return_value={"commit": "", "last_modified": ""})
    def test_copy_missing_source_dir_returns_false(self, _mock_hf):
        self.cache.put("test/ds", "/nonexistent/path")
        result = self.cache.copy_to_output("test/ds", self.dst_dir)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
