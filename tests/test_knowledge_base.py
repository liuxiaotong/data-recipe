"""Unit tests for KnowledgeBase, PatternStore, TrendAnalyzer, and DatasetCatalog.

Tests cover initialization, data ingestion, querying, persistence,
edge cases (empty data, missing fields), and report generation.
"""

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from datarecipe.knowledge.dataset_catalog import (
    INDUSTRY_BENCHMARKS,
    KNOWN_DATASETS,
    DatasetCatalog,
    DatasetCategory,
    DatasetInfo,
    IndustryBenchmark,
)
from datarecipe.knowledge.knowledge_base import (
    CostBenchmark,
    KnowledgeBase,
    PatternEntry,
    PatternStore,
    TrendAnalyzer,
)

# ---------------------------------------------------------------------------
# Helper: minimal RecipeSummary stand-in
# ---------------------------------------------------------------------------

@dataclass
class FakeRecipeSummary:
    """Minimal stand-in for RecipeSummary used in ingest_analysis tests."""

    dataset_id: str = "test/dataset"
    dataset_type: str = "preference"
    reproduction_cost: dict = field(default_factory=lambda: {"human": 5000.0, "api": 200.0})
    human_percentage: float = 70.0
    sample_count: int = 1000
    fields: list = field(default_factory=lambda: ["prompt", "chosen", "rejected"])


@dataclass
class FakeRubricsResult:
    """Minimal stand-in for a rubrics analysis result."""

    verb_distribution: dict = field(default_factory=lambda: {"evaluate": 5, "compare": 3})


# ===================================================================
# PatternEntry dataclass
# ===================================================================

class TestPatternEntry(unittest.TestCase):
    """Test PatternEntry dataclass."""

    def test_defaults(self):
        entry = PatternEntry(pattern_type="rubric", pattern_key="evaluate")
        self.assertEqual(entry.pattern_type, "rubric")
        self.assertEqual(entry.pattern_key, "evaluate")
        self.assertEqual(entry.frequency, 0)
        self.assertEqual(entry.datasets, [])
        self.assertEqual(entry.examples, [])
        self.assertEqual(entry.metadata, {})

    def test_custom_values(self):
        entry = PatternEntry(
            pattern_type="schema",
            pattern_key="prompt",
            frequency=3,
            datasets=["ds1", "ds2"],
            examples=["ex1"],
            metadata={"source": "test"},
        )
        self.assertEqual(entry.frequency, 3)
        self.assertEqual(len(entry.datasets), 2)
        self.assertEqual(entry.metadata["source"], "test")


# ===================================================================
# CostBenchmark dataclass
# ===================================================================

class TestCostBenchmark(unittest.TestCase):
    """Test CostBenchmark dataclass."""

    def test_defaults(self):
        cb = CostBenchmark(dataset_type="sft")
        self.assertEqual(cb.dataset_type, "sft")
        self.assertEqual(cb.sample_count, 0)
        self.assertAlmostEqual(cb.avg_human_cost, 0.0)
        self.assertEqual(cb.datasets, [])

    def test_custom_values(self):
        cb = CostBenchmark(
            dataset_type="preference",
            sample_count=5000,
            avg_human_cost=100.0,
            avg_api_cost=20.0,
            avg_total_cost=120.0,
            avg_human_percentage=80.0,
            min_cost=50.0,
            max_cost=200.0,
            datasets=["ds1"],
        )
        self.assertAlmostEqual(cb.avg_total_cost, 120.0)
        self.assertEqual(len(cb.datasets), 1)


# ===================================================================
# PatternStore
# ===================================================================

class TestPatternStore(unittest.TestCase):
    """Test PatternStore init, add, query, and persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, "patterns.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # -- init --

    def test_init_no_file(self):
        store = PatternStore(store_path=self.store_path)
        self.assertEqual(len(store.patterns), 0)

    def test_init_loads_existing_file(self):
        data = {
            "rubric:evaluate": {
                "pattern_type": "rubric",
                "pattern_key": "evaluate",
                "frequency": 2,
                "datasets": ["ds1"],
                "examples": [],
                "metadata": {},
            }
        }
        with open(self.store_path, "w") as f:
            json.dump(data, f)
        store = PatternStore(store_path=self.store_path)
        self.assertEqual(len(store.patterns), 1)
        self.assertEqual(store.patterns["rubric:evaluate"].frequency, 2)

    def test_init_corrupt_file_ignored(self):
        with open(self.store_path, "w") as f:
            f.write("NOT VALID JSON")
        store = PatternStore(store_path=self.store_path)
        self.assertEqual(len(store.patterns), 0)

    # -- add_pattern --

    def test_add_new_pattern(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1", example="ex1", metadata={"x": 1})
        self.assertIn("rubric:evaluate", store.patterns)
        entry = store.patterns["rubric:evaluate"]
        self.assertEqual(entry.frequency, 1)
        self.assertIn("ds1", entry.datasets)
        self.assertIn("ex1", entry.examples)
        self.assertEqual(entry.metadata["x"], 1)

    def test_add_pattern_increments_frequency(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        store.add_pattern("rubric", "evaluate", "ds2")
        self.assertEqual(store.patterns["rubric:evaluate"].frequency, 2)

    def test_add_pattern_deduplicates_datasets(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        store.add_pattern("rubric", "evaluate", "ds1")
        self.assertEqual(store.patterns["rubric:evaluate"].datasets, ["ds1"])
        # frequency still increments though
        self.assertEqual(store.patterns["rubric:evaluate"].frequency, 2)

    def test_add_pattern_limits_examples_to_five(self):
        store = PatternStore(store_path=self.store_path)
        for i in range(10):
            store.add_pattern("rubric", "evaluate", "ds1", example=f"ex{i}")
        self.assertEqual(len(store.patterns["rubric:evaluate"].examples), 5)

    def test_add_pattern_no_example(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        self.assertEqual(store.patterns["rubric:evaluate"].examples, [])

    def test_add_pattern_no_metadata(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        self.assertEqual(store.patterns["rubric:evaluate"].metadata, {})

    def test_add_pattern_metadata_merges(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1", metadata={"a": 1})
        store.add_pattern("rubric", "evaluate", "ds1", metadata={"b": 2})
        self.assertEqual(store.patterns["rubric:evaluate"].metadata, {"a": 1, "b": 2})

    # -- persistence --

    def test_save_and_reload(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("schema", "prompt", "ds1")
        store.add_pattern("schema", "response", "ds2")

        store2 = PatternStore(store_path=self.store_path)
        self.assertEqual(len(store2.patterns), 2)
        self.assertEqual(store2.patterns["schema:prompt"].frequency, 1)

    # -- get_top_patterns --

    def test_get_top_patterns_all(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        store.add_pattern("rubric", "evaluate", "ds2")
        store.add_pattern("schema", "prompt", "ds1")
        top = store.get_top_patterns(limit=10)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0].pattern_key, "evaluate")  # higher frequency

    def test_get_top_patterns_by_type(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        store.add_pattern("schema", "prompt", "ds1")
        top = store.get_top_patterns(pattern_type="schema", limit=10)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].pattern_key, "prompt")

    def test_get_top_patterns_limit(self):
        store = PatternStore(store_path=self.store_path)
        for i in range(10):
            store.add_pattern("rubric", f"verb{i}", "ds1")
        top = store.get_top_patterns(limit=3)
        self.assertEqual(len(top), 3)

    def test_get_top_patterns_empty(self):
        store = PatternStore(store_path=self.store_path)
        self.assertEqual(store.get_top_patterns(), [])

    # -- find_patterns_for_dataset --

    def test_find_patterns_for_dataset(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        store.add_pattern("schema", "prompt", "ds1")
        store.add_pattern("schema", "response", "ds2")
        result = store.find_patterns_for_dataset("ds1")
        self.assertEqual(len(result), 2)

    def test_find_patterns_for_dataset_not_found(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        result = store.find_patterns_for_dataset("nonexistent")
        self.assertEqual(result, [])

    # -- get_pattern_stats --

    def test_get_pattern_stats(self):
        store = PatternStore(store_path=self.store_path)
        store.add_pattern("rubric", "evaluate", "ds1")
        store.add_pattern("rubric", "compare", "ds1")
        store.add_pattern("schema", "prompt", "ds2")
        stats = store.get_pattern_stats()
        self.assertEqual(stats["total_patterns"], 3)
        self.assertEqual(stats["by_type"]["rubric"], 2)
        self.assertEqual(stats["by_type"]["schema"], 1)
        self.assertIsInstance(stats["top_patterns"], list)

    def test_get_pattern_stats_empty(self):
        store = PatternStore(store_path=self.store_path)
        stats = store.get_pattern_stats()
        self.assertEqual(stats["total_patterns"], 0)
        self.assertEqual(stats["top_patterns"], [])


# ===================================================================
# TrendAnalyzer
# ===================================================================

class TestTrendAnalyzer(unittest.TestCase):
    """Test TrendAnalyzer init, recording, benchmarks, and trends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # -- init --

    def test_init_empty(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        self.assertEqual(ta.trends, {})
        self.assertEqual(ta.benchmarks, {})

    def test_init_loads_existing_data(self):
        trends_path = os.path.join(self.tmpdir, "trends.json")
        bench_path = os.path.join(self.tmpdir, "cost_benchmarks.json")
        with open(trends_path, "w") as f:
            json.dump({"2024-01-01": {"datasets_analyzed": 1, "types": {}, "total_cost": 100}}, f)
        with open(bench_path, "w") as f:
            json.dump({
                "sft": {
                    "dataset_type": "sft",
                    "sample_count": 100,
                    "avg_human_cost": 50.0,
                    "avg_api_cost": 10.0,
                    "avg_total_cost": 60.0,
                    "avg_human_percentage": 80.0,
                    "min_cost": 30.0,
                    "max_cost": 90.0,
                    "datasets": ["ds1"],
                }
            }, f)
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        self.assertIn("2024-01-01", ta.trends)
        self.assertIn("sft", ta.benchmarks)
        self.assertEqual(ta.benchmarks["sft"].sample_count, 100)

    def test_init_corrupt_trends_file(self):
        with open(os.path.join(self.tmpdir, "trends.json"), "w") as f:
            f.write("CORRUPT")
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        self.assertEqual(ta.trends, {})

    def test_init_corrupt_benchmarks_file(self):
        with open(os.path.join(self.tmpdir, "cost_benchmarks.json"), "w") as f:
            f.write("CORRUPT")
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        self.assertEqual(ta.benchmarks, {})

    # -- record_analysis --

    def test_record_analysis_first_entry(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis(
            dataset_id="ds1",
            dataset_type="preference",
            human_cost=5000.0,
            api_cost=200.0,
            human_percentage=70.0,
            sample_count=1000,
        )
        bench = ta.benchmarks["preference"]
        self.assertAlmostEqual(bench.avg_human_cost, 5000.0)
        self.assertAlmostEqual(bench.avg_api_cost, 200.0)
        self.assertAlmostEqual(bench.avg_total_cost, 5200.0)
        self.assertAlmostEqual(bench.min_cost, 5200.0)
        self.assertAlmostEqual(bench.max_cost, 5200.0)
        self.assertEqual(bench.sample_count, 1000)
        self.assertIn("ds1", bench.datasets)

    def test_record_analysis_incremental_average(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "sft", 100.0, 50.0, 60.0, 500)
        ta.record_analysis("ds2", "sft", 200.0, 100.0, 80.0, 1000)

        bench = ta.benchmarks["sft"]
        # avg_human_cost: (100 * 1 + 200) / 2 = 150
        self.assertAlmostEqual(bench.avg_human_cost, 150.0)
        # avg_api_cost: (50 * 1 + 100) / 2 = 75
        self.assertAlmostEqual(bench.avg_api_cost, 75.0)
        # total costs: 150 and 300 -> avg = 225
        self.assertAlmostEqual(bench.avg_total_cost, 225.0)
        self.assertAlmostEqual(bench.min_cost, 150.0)
        self.assertAlmostEqual(bench.max_cost, 300.0)
        self.assertEqual(bench.sample_count, 1500)
        self.assertEqual(len(bench.datasets), 2)

    def test_record_analysis_duplicate_dataset_id(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "sft", 100.0, 50.0, 60.0, 500)
        ta.record_analysis("ds1", "sft", 200.0, 100.0, 80.0, 500)
        bench = ta.benchmarks["sft"]
        # dataset should not be added twice
        self.assertEqual(bench.datasets.count("ds1"), 1)

    def test_record_analysis_saves_trend_data(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "preference", 5000.0, 200.0, 70.0, 1000)
        today = datetime.now().strftime("%Y-%m-%d")
        self.assertIn(today, ta.trends)
        self.assertEqual(ta.trends[today]["datasets_analyzed"], 1)
        self.assertAlmostEqual(ta.trends[today]["total_cost"], 5200.0)

    def test_record_analysis_persistence(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "sft", 100.0, 50.0, 60.0, 500)

        ta2 = TrendAnalyzer(data_dir=self.tmpdir)
        self.assertIn("sft", ta2.benchmarks)
        self.assertEqual(ta2.benchmarks["sft"].sample_count, 500)

    # -- get_cost_benchmark --

    def test_get_cost_benchmark_exists(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "sft", 100.0, 50.0, 60.0, 500)
        result = ta.get_cost_benchmark("sft")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.avg_total_cost, 150.0)

    def test_get_cost_benchmark_missing(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        self.assertIsNone(ta.get_cost_benchmark("nonexistent"))

    # -- get_all_benchmarks --

    def test_get_all_benchmarks(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "sft", 100.0, 50.0, 60.0, 500)
        ta.record_analysis("ds2", "preference", 200.0, 100.0, 70.0, 800)
        result = ta.get_all_benchmarks()
        self.assertEqual(len(result), 2)
        self.assertIn("sft", result)
        self.assertIn("preference", result)

    def test_get_all_benchmarks_returns_copy(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        result = ta.get_all_benchmarks()
        result["fake"] = CostBenchmark(dataset_type="fake")
        self.assertNotIn("fake", ta.benchmarks)

    # -- get_trend_summary --

    def test_get_trend_summary_no_data(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        result = ta.get_trend_summary(30)
        self.assertEqual(result["period"], "last 30 days")
        self.assertEqual(result["data"], [])

    def test_get_trend_summary_with_data(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "sft", 100.0, 50.0, 60.0, 500)
        ta.record_analysis("ds2", "preference", 200.0, 100.0, 70.0, 800)
        result = ta.get_trend_summary(30)
        self.assertEqual(result["datasets_analyzed"], 2)
        self.assertAlmostEqual(result["total_cost"], 450.0)
        self.assertIn("type_distribution", result)
        self.assertIn("daily_data", result)

    def test_get_trend_summary_excludes_old_data(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        # Inject old trend data directly
        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        ta.trends[old_date] = {
            "datasets_analyzed": 1,
            "types": {"sft": 1},
            "total_cost": 999.0,
        }
        ta._save()
        result = ta.get_trend_summary(30)
        # old data should not be included
        self.assertEqual(result["data"], [])

    def test_get_trend_summary_avg_cost_per_dataset(self):
        ta = TrendAnalyzer(data_dir=self.tmpdir)
        ta.record_analysis("ds1", "sft", 100.0, 0.0, 60.0, 500)
        result = ta.get_trend_summary(30)
        expected_avg = round(100.0 / 1, 2)
        self.assertEqual(result["avg_cost_per_dataset"], expected_avg)


# ===================================================================
# KnowledgeBase
# ===================================================================

class TestKnowledgeBase(unittest.TestCase):
    """Test KnowledgeBase init, ingest, query, and report generation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # -- init --

    def test_init_creates_directory(self):
        base_dir = os.path.join(self.tmpdir, "kb_subdir")
        kb = KnowledgeBase(base_dir=base_dir)
        self.assertTrue(os.path.isdir(base_dir))
        self.assertIsNotNone(kb.patterns)
        self.assertIsNotNone(kb.trends)

    def test_init_default_creates_sub_stores(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        self.assertIsInstance(kb.patterns, PatternStore)
        self.assertIsInstance(kb.trends, TrendAnalyzer)

    # -- ingest_analysis --

    def test_ingest_analysis_basic(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary()
        kb.ingest_analysis("test/dataset", summary)

        # Should have recorded trend
        bench = kb.trends.get_cost_benchmark("preference")
        self.assertIsNotNone(bench)
        self.assertEqual(bench.sample_count, 1000)

        # Should have recorded schema field patterns
        field_patterns = kb.patterns.find_patterns_for_dataset("test/dataset")
        field_keys = [p.pattern_key for p in field_patterns]
        self.assertIn("prompt", field_keys)
        self.assertIn("chosen", field_keys)
        self.assertIn("rejected", field_keys)

    def test_ingest_analysis_with_rubrics(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary()
        rubrics = FakeRubricsResult()
        kb.ingest_analysis("test/dataset", summary, rubrics_result=rubrics)

        patterns = kb.patterns.find_patterns_for_dataset("test/dataset")
        pattern_keys = [p.pattern_key for p in patterns]
        self.assertIn("evaluate", pattern_keys)
        self.assertIn("compare", pattern_keys)

    def test_ingest_analysis_records_dataset_type_pattern(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary(dataset_type="preference")
        kb.ingest_analysis("test/dataset", summary)

        patterns = kb.patterns.find_patterns_for_dataset("test/dataset")
        type_patterns = [p for p in patterns if p.pattern_type == "dataset_type"]
        self.assertEqual(len(type_patterns), 1)
        self.assertEqual(type_patterns[0].pattern_key, "preference")

    def test_ingest_analysis_no_dataset_type(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary(dataset_type="")
        kb.ingest_analysis("test/dataset", summary)

        patterns = kb.patterns.find_patterns_for_dataset("test/dataset")
        type_patterns = [p for p in patterns if p.pattern_type == "dataset_type"]
        self.assertEqual(len(type_patterns), 0)

    def test_ingest_analysis_empty_fields(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary(fields=[])
        kb.ingest_analysis("test/dataset", summary)

        patterns = kb.patterns.find_patterns_for_dataset("test/dataset")
        schema_patterns = [p for p in patterns if p.pattern_type == "schema_field"]
        self.assertEqual(len(schema_patterns), 0)

    def test_ingest_analysis_no_rubrics(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary()
        kb.ingest_analysis("test/dataset", summary, rubrics_result=None)
        # Should not raise
        patterns = kb.patterns.find_patterns_for_dataset("test/dataset")
        rubric_patterns = [p for p in patterns if p.pattern_type == "rubric_verb"]
        self.assertEqual(len(rubric_patterns), 0)

    def test_ingest_analysis_rubrics_no_verb_distribution(self):
        """Rubrics result without verb_distribution attribute should be skipped."""
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary()
        rubrics = MagicMock(spec=[])  # no verb_distribution attr
        kb.ingest_analysis("test/dataset", summary, rubrics_result=rubrics)
        patterns = kb.patterns.find_patterns_for_dataset("test/dataset")
        rubric_patterns = [p for p in patterns if p.pattern_type == "rubric_verb"]
        self.assertEqual(len(rubric_patterns), 0)

    def test_ingest_analysis_missing_cost_keys(self):
        """reproduction_cost with missing keys should default to 0."""
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary(reproduction_cost={})
        kb.ingest_analysis("test/dataset", summary)
        bench = kb.trends.get_cost_benchmark("preference")
        self.assertIsNotNone(bench)
        self.assertAlmostEqual(bench.avg_human_cost, 0.0)
        self.assertAlmostEqual(bench.avg_api_cost, 0.0)

    def test_ingest_analysis_unknown_type(self):
        """dataset_type=None should record as 'unknown'."""
        kb = KnowledgeBase(base_dir=self.tmpdir)
        summary = FakeRecipeSummary(dataset_type=None)
        kb.ingest_analysis("test/dataset", summary)
        bench = kb.trends.get_cost_benchmark("unknown")
        self.assertIsNotNone(bench)

    # -- get_similar_patterns --

    def test_get_similar_patterns_basic(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        # Ingest two datasets with overlapping fields
        s1 = FakeRecipeSummary(fields=["prompt", "chosen", "rejected"])
        s2 = FakeRecipeSummary(fields=["prompt", "chosen", "response"])
        kb.ingest_analysis("ds1", s1)
        kb.ingest_analysis("ds2", s2)

        similar = kb.get_similar_patterns("ds1")
        self.assertIn("ds2", similar)

    def test_get_similar_patterns_no_patterns(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        result = kb.get_similar_patterns("nonexistent")
        self.assertEqual(result, [])

    def test_get_similar_patterns_excludes_self(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        s1 = FakeRecipeSummary(fields=["prompt", "chosen"])
        kb.ingest_analysis("ds1", s1)
        similar = kb.get_similar_patterns("ds1")
        self.assertNotIn("ds1", similar)

    def test_get_similar_patterns_limit(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        # Ingest many datasets sharing "prompt" field
        for i in range(20):
            s = FakeRecipeSummary(fields=["prompt"])
            kb.ingest_analysis(f"ds{i}", s)
        similar = kb.get_similar_patterns("ds0", limit=5)
        self.assertLessEqual(len(similar), 5)

    # -- get_recommendations --

    def test_get_recommendations_no_data(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        result = kb.get_recommendations("preference")
        self.assertEqual(result["dataset_type"], "preference")
        self.assertIsNone(result["cost_estimate"])
        self.assertEqual(result["common_patterns"], [])
        self.assertEqual(result["suggested_fields"], [])

    def test_get_recommendations_with_data(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        s1 = FakeRecipeSummary(
            dataset_type="preference",
            reproduction_cost={"human": 5000.0, "api": 200.0},
            human_percentage=70.0,
            sample_count=1000,
            fields=["prompt", "chosen", "rejected"],
        )
        kb.ingest_analysis("ds1", s1)
        result = kb.get_recommendations("preference")
        self.assertIsNotNone(result["cost_estimate"])
        self.assertEqual(result["cost_estimate"]["based_on"], 1)
        self.assertAlmostEqual(result["cost_estimate"]["avg_total"], 5200.0)

    def test_get_recommendations_suggested_fields_sorted_by_frequency(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        for i in range(3):
            s = FakeRecipeSummary(fields=["prompt", "chosen"])
            kb.ingest_analysis(f"ds{i}", s)
        # Add one more with "response"
        s_extra = FakeRecipeSummary(fields=["response"])
        kb.ingest_analysis("ds_extra", s_extra)
        result = kb.get_recommendations("preference")
        # "prompt" and "chosen" should appear before "response"
        fields = result["suggested_fields"]
        if "prompt" in fields and "response" in fields:
            self.assertLess(fields.index("prompt"), fields.index("response"))

    # -- find_similar_datasets --

    def test_find_similar_datasets_by_type(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        results = kb.find_similar_datasets(dataset_type="preference")
        # Should return catalog entries for 'preference'
        for r in results:
            self.assertEqual(r.category, DatasetCategory.PREFERENCE)

    def test_find_similar_datasets_by_id(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        results = kb.find_similar_datasets(
            dataset_type="preference",
            dataset_id="Anthropic/hh-rlhf",
        )
        # Should return results (similar datasets from catalog)
        self.assertIsInstance(results, list)

    def test_find_similar_datasets_unknown_type(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        results = kb.find_similar_datasets(dataset_type="nonexistent_category")
        self.assertEqual(results, [])

    def test_find_similar_datasets_limit(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        results = kb.find_similar_datasets(dataset_type="preference", limit=2)
        self.assertLessEqual(len(results), 2)

    # -- export_report --

    def test_export_report_creates_file(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        output_path = os.path.join(self.tmpdir, "report.md")
        result = kb.export_report(output_path=output_path)
        self.assertEqual(result, output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_export_report_default_path(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        result = kb.export_report()
        expected = os.path.join(self.tmpdir, "KNOWLEDGE_REPORT.md")
        self.assertEqual(result, expected)
        self.assertTrue(os.path.exists(expected))

    def test_export_report_content(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        s1 = FakeRecipeSummary(
            dataset_type="preference",
            reproduction_cost={"human": 5000.0, "api": 200.0},
            human_percentage=70.0,
            sample_count=1000,
            fields=["prompt", "chosen"],
        )
        kb.ingest_analysis("ds1", s1)

        output_path = os.path.join(self.tmpdir, "report.md")
        kb.export_report(output_path=output_path)

        with open(output_path, encoding="utf-8") as f:
            content = f.read()

        self.assertIn("DataRecipe", content)
        self.assertIn("preference", content)

    def test_export_report_empty_kb(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        output_path = os.path.join(self.tmpdir, "report.md")
        kb.export_report(output_path=output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("0", content)  # total patterns should be 0


# ===================================================================
# DatasetCatalog
# ===================================================================

class TestDatasetCatalog(unittest.TestCase):
    """Test DatasetCatalog search and comparison methods."""

    def setUp(self):
        self.catalog = DatasetCatalog()

    # -- get_dataset --

    def test_get_dataset_exact_match(self):
        result = self.catalog.get_dataset("Anthropic/hh-rlhf")
        self.assertIsNotNone(result)
        self.assertEqual(result.dataset_id, "Anthropic/hh-rlhf")

    def test_get_dataset_normalized_match(self):
        result = self.catalog.get_dataset("truthful-qa")
        self.assertIsNotNone(result)
        self.assertEqual(result.dataset_id, "truthful_qa")

    def test_get_dataset_not_found(self):
        result = self.catalog.get_dataset("nonexistent/dataset")
        self.assertIsNone(result)

    def test_get_dataset_case_insensitive(self):
        result = self.catalog.get_dataset("ANTHROPIC/HH-RLHF")
        self.assertIsNotNone(result)

    # -- find_similar_datasets --

    def test_find_similar_datasets_by_id(self):
        results = self.catalog.find_similar_datasets(dataset_id="Anthropic/hh-rlhf")
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

    def test_find_similar_datasets_by_category(self):
        results = self.catalog.find_similar_datasets(category="preference")
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r.category, DatasetCategory.PREFERENCE)

    def test_find_similar_datasets_by_tags(self):
        results = self.catalog.find_similar_datasets(tags=["rlhf", "safety"])
        self.assertTrue(len(results) > 0)

    def test_find_similar_datasets_limit(self):
        results = self.catalog.find_similar_datasets(category="preference", limit=2)
        self.assertLessEqual(len(results), 2)

    def test_find_similar_datasets_unknown_id(self):
        results = self.catalog.find_similar_datasets(dataset_id="nonexistent/ds")
        # Should return empty since the dataset has no similar_to
        self.assertEqual(results, [])

    def test_find_similar_datasets_invalid_category(self):
        results = self.catalog.find_similar_datasets(category="invalid_category_xyz")
        self.assertEqual(results, [])

    def test_find_similar_datasets_combined_filters(self):
        results = self.catalog.find_similar_datasets(
            dataset_id="Anthropic/hh-rlhf",
            category="preference",
        )
        self.assertIsInstance(results, list)

    def test_find_similar_datasets_excludes_self(self):
        results = self.catalog.find_similar_datasets(
            dataset_id="Anthropic/hh-rlhf",
            category="preference",
        )
        ids = [r.dataset_id for r in results]
        self.assertNotIn("Anthropic/hh-rlhf", ids)

    # -- find_by_category --

    def test_find_by_category_preference(self):
        results = self.catalog.find_by_category("preference")
        self.assertTrue(len(results) > 0)
        for r in results:
            self.assertEqual(r.category, DatasetCategory.PREFERENCE)

    def test_find_by_category_invalid(self):
        results = self.catalog.find_by_category("invalid_xyz")
        self.assertEqual(results, [])

    def test_find_by_category_sorted_by_citation(self):
        results = self.catalog.find_by_category("preference")
        if len(results) >= 2:
            self.assertGreaterEqual(results[0].citation_count, results[1].citation_count)

    def test_find_by_category_limit(self):
        results = self.catalog.find_by_category("preference", limit=1)
        self.assertEqual(len(results), 1)

    # -- find_by_tags --

    def test_find_by_tags_single(self):
        results = self.catalog.find_by_tags(["rlhf"])
        self.assertTrue(len(results) > 0)
        for r in results:
            tags_lower = {t.lower() for t in r.tags}
            self.assertIn("rlhf", tags_lower)

    def test_find_by_tags_multiple(self):
        results = self.catalog.find_by_tags(["evaluation", "math"])
        self.assertTrue(len(results) > 0)

    def test_find_by_tags_no_match(self):
        results = self.catalog.find_by_tags(["nonexistent_tag_xyz"])
        self.assertEqual(results, [])

    def test_find_by_tags_limit(self):
        results = self.catalog.find_by_tags(["rlhf"], limit=2)
        self.assertLessEqual(len(results), 2)

    def test_find_by_tags_case_insensitive(self):
        results = self.catalog.find_by_tags(["RLHF"])
        self.assertTrue(len(results) > 0)

    # -- get_benchmark --

    def test_get_benchmark_exists(self):
        result = self.catalog.get_benchmark("preference")
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "preference")

    def test_get_benchmark_not_found(self):
        result = self.catalog.get_benchmark("nonexistent")
        self.assertIsNone(result)

    def test_get_benchmark_case_handling(self):
        result = self.catalog.get_benchmark("PREFERENCE")
        # get_benchmark lowercases the input, so uppercase should match
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "preference")

    # -- get_all_benchmarks --

    def test_get_all_benchmarks(self):
        result = self.catalog.get_all_benchmarks()
        self.assertIsInstance(result, dict)
        self.assertIn("preference", result)
        self.assertIn("sft", result)

    def test_get_all_benchmarks_returns_copy(self):
        result = self.catalog.get_all_benchmarks()
        result["fake"] = None
        self.assertNotIn("fake", self.catalog.benchmarks)

    # -- compare_with_benchmark --

    def test_compare_below_average(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=500.0,  # 0.5 per sample, below min of 1.0
            human_percentage=85.0,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["comparison"]["cost_rating"], "below_average")

    def test_compare_average(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=2000.0,  # 2.0 per sample, between 1.0 and 3.0
            human_percentage=85.0,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["comparison"]["cost_rating"], "average")

    def test_compare_above_average(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=5000.0,  # 5.0 per sample, between 3.0 and 10.0
            human_percentage=85.0,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["comparison"]["cost_rating"], "above_average")

    def test_compare_high_cost(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=50000.0,  # 50.0 per sample, above max 10.0
            human_percentage=85.0,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["comparison"]["cost_rating"], "high")

    def test_compare_no_benchmark(self):
        result = self.catalog.compare_with_benchmark(
            category="nonexistent",
            sample_count=1000,
            total_cost=5000.0,
            human_percentage=70.0,
        )
        self.assertFalse(result["available"])
        self.assertIn("reason", result)

    def test_compare_human_typical(self):
        # benchmark avg_human_percentage for preference is 85.0
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=3000.0,
            human_percentage=85.0,  # within 10% of 85
        )
        self.assertEqual(result["comparison"]["human_rating"], "typical")

    def test_compare_human_more_human(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=3000.0,
            human_percentage=98.0,  # 13% above 85 -> more_human
        )
        self.assertEqual(result["comparison"]["human_rating"], "more_human")

    def test_compare_human_more_automated(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=3000.0,
            human_percentage=50.0,  # 35% below 85 -> more_automated
        )
        self.assertEqual(result["comparison"]["human_rating"], "more_automated")

    def test_compare_zero_samples(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=0,
            total_cost=5000.0,
            human_percentage=70.0,
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["your_project"]["cost_per_sample"], 0)

    def test_compare_includes_similar_projects(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=3000.0,
            human_percentage=85.0,
        )
        self.assertIn("similar_projects", result)
        self.assertIsInstance(result["similar_projects"], list)

    def test_compare_cost_vs_avg_string(self):
        result = self.catalog.compare_with_benchmark(
            category="preference",
            sample_count=1000,
            total_cost=3000.0,
            human_percentage=85.0,
        )
        self.assertIn("cost_vs_avg", result["comparison"])
        self.assertTrue(result["comparison"]["cost_vs_avg"].endswith("%"))


# ===================================================================
# DatasetInfo and IndustryBenchmark dataclasses
# ===================================================================

class TestDatasetInfo(unittest.TestCase):
    """Test DatasetInfo dataclass."""

    def test_defaults(self):
        info = DatasetInfo(
            dataset_id="test/ds",
            category=DatasetCategory.SFT,
        )
        self.assertEqual(info.dataset_id, "test/ds")
        self.assertEqual(info.category, DatasetCategory.SFT)
        self.assertEqual(info.description, "")
        self.assertEqual(info.similar_to, [])
        self.assertEqual(info.tags, [])

    def test_known_datasets_populated(self):
        self.assertTrue(len(KNOWN_DATASETS) > 0)
        for ds_id, info in KNOWN_DATASETS.items():
            self.assertEqual(ds_id, info.dataset_id)
            self.assertIsInstance(info.category, DatasetCategory)


class TestIndustryBenchmark(unittest.TestCase):
    """Test IndustryBenchmark dataclass."""

    def test_defaults(self):
        bench = IndustryBenchmark(category="test", description="Test benchmark")
        self.assertEqual(bench.category, "test")
        self.assertEqual(bench.typical_project_size, 1000)
        self.assertAlmostEqual(bench.avg_human_percentage, 70.0)

    def test_known_benchmarks_populated(self):
        self.assertTrue(len(INDUSTRY_BENCHMARKS) > 0)
        for key, bench in INDUSTRY_BENCHMARKS.items():
            self.assertEqual(key, bench.category)
            self.assertTrue(bench.avg_cost_per_sample > 0)


class TestDatasetCategory(unittest.TestCase):
    """Test DatasetCategory enum."""

    def test_values(self):
        self.assertEqual(DatasetCategory.PREFERENCE.value, "preference")
        self.assertEqual(DatasetCategory.EVALUATION.value, "evaluation")
        self.assertEqual(DatasetCategory.SFT.value, "sft")
        self.assertEqual(DatasetCategory.CODE.value, "code")

    def test_from_string(self):
        self.assertEqual(DatasetCategory("preference"), DatasetCategory.PREFERENCE)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            DatasetCategory("nonexistent")


# ===================================================================
# Edge cases and integration
# ===================================================================

class TestKnowledgeBaseIntegration(unittest.TestCase):
    """Integration tests for KnowledgeBase with multiple ingestions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_multiple_ingestions_accumulate(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        for i in range(5):
            s = FakeRecipeSummary(
                dataset_type="sft",
                reproduction_cost={"human": 100.0 * (i + 1), "api": 10.0 * (i + 1)},
                human_percentage=60.0 + i,
                sample_count=100 * (i + 1),
                fields=["prompt", "response"],
            )
            kb.ingest_analysis(f"ds{i}", s)

        bench = kb.trends.get_cost_benchmark("sft")
        self.assertEqual(len(bench.datasets), 5)
        self.assertEqual(bench.sample_count, 100 + 200 + 300 + 400 + 500)

    def test_persistence_across_instances(self):
        kb1 = KnowledgeBase(base_dir=self.tmpdir)
        s = FakeRecipeSummary(fields=["prompt", "response"])
        kb1.ingest_analysis("ds1", s)

        kb2 = KnowledgeBase(base_dir=self.tmpdir)
        patterns = kb2.patterns.find_patterns_for_dataset("ds1")
        self.assertTrue(len(patterns) > 0)

    def test_export_report_after_ingestion(self):
        kb = KnowledgeBase(base_dir=self.tmpdir)
        s = FakeRecipeSummary(
            dataset_type="evaluation",
            reproduction_cost={"human": 2000.0, "api": 500.0},
            human_percentage=80.0,
            sample_count=500,
            fields=["question", "answer", "category"],
        )
        kb.ingest_analysis("eval/ds1", s)

        output_path = os.path.join(self.tmpdir, "report.md")
        kb.export_report(output_path=output_path)

        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("evaluation", content)
        self.assertIn("schema_field", content)

    def test_recommendations_common_patterns(self):
        """Patterns appearing in >50% of datasets should show up as common."""
        kb = KnowledgeBase(base_dir=self.tmpdir)
        # All three share "prompt"; only two share "answer"
        for i in range(3):
            fields = ["prompt"]
            if i < 2:
                fields.append("answer")
            s = FakeRecipeSummary(
                dataset_type="sft",
                reproduction_cost={"human": 100.0, "api": 10.0},
                human_percentage=70.0,
                sample_count=100,
                fields=fields,
            )
            kb.ingest_analysis(f"ds{i}", s)

        result = kb.get_recommendations("sft")
        common_keys = [p["pattern"] for p in result["common_patterns"]]
        # "prompt" appears in all 3 datasets (100% > 50%) -> should be common
        self.assertIn("prompt", common_keys)

    def test_get_similar_patterns_scores_correctly(self):
        """Datasets sharing more patterns should rank higher."""
        kb = KnowledgeBase(base_dir=self.tmpdir)
        s1 = FakeRecipeSummary(fields=["a", "b", "c"])
        s2 = FakeRecipeSummary(fields=["a", "b", "d"])  # 2 overlaps
        s3 = FakeRecipeSummary(fields=["a", "e", "f"])  # 1 overlap
        kb.ingest_analysis("ds1", s1)
        kb.ingest_analysis("ds2", s2)
        kb.ingest_analysis("ds3", s3)

        similar = kb.get_similar_patterns("ds1")
        self.assertEqual(similar[0], "ds2")  # ds2 has higher overlap


if __name__ == "__main__":
    unittest.main()
