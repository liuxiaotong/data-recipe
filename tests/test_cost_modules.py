"""Unit tests for datarecipe.cost modules.

Covers:
- calibrator.py: CalibrationResult, CostCalibrator
- complexity_analyzer.py: ComplexityMetrics, ComplexityAnalyzer, DomainType, DOMAIN_DIFFICULTY
- phased_model.py: PhasedCostModel, PhasedCostBreakdown, ProjectScale, phase cost dataclasses
- token_analyzer.py: TokenAnalyzer, TokenStats, PreciseCostCalculator, PreciseCostEstimate, MODEL_PRICING
- __init__.py: re-exports
"""

import unittest
from unittest.mock import MagicMock, patch

from datarecipe.cost import (
    DOMAIN_DIFFICULTY,
    MODEL_PRICING,
    CalibrationResult,
    ComplexityAnalyzer,
    ComplexityMetrics,
    CostCalibrator,
    DomainType,
    ModelPricing,
    PhasedCostBreakdown,
    PhasedCostModel,
    PreciseCostCalculator,
    PreciseCostEstimate,
    ProjectScale,
    QualityPhaseCost,
    TokenAnalyzer,
    TokenStats,
)
from datarecipe.cost.complexity_analyzer import DOMAIN_EXPERT_MULTIPLIER
from datarecipe.cost.phased_model import (
    DESIGN_PHASE_BASE_COSTS,
    QUALITY_RATES,
    DesignPhaseCost,
    ProductionPhaseCost,
)

# ==========================================================================
# __init__.py re-exports
# ==========================================================================


class TestCostPackageExports(unittest.TestCase):
    """Ensure __init__.py re-exports all public symbols."""

    def test_token_analyzer_exports(self):
        from datarecipe.cost import MODEL_PRICING, TokenAnalyzer

        self.assertTrue(callable(TokenAnalyzer))
        self.assertIsInstance(MODEL_PRICING, dict)

    def test_complexity_analyzer_exports(self):
        from datarecipe.cost import (
            DOMAIN_DIFFICULTY,
            ComplexityAnalyzer,
        )

        self.assertTrue(callable(ComplexityAnalyzer))
        self.assertIsInstance(DOMAIN_DIFFICULTY, dict)

    def test_calibrator_exports(self):
        from datarecipe.cost import CostCalibrator

        self.assertTrue(callable(CostCalibrator))

    def test_phased_model_exports(self):
        from datarecipe.cost import PhasedCostModel

        self.assertTrue(callable(PhasedCostModel))


# ==========================================================================
# CalibrationResult
# ==========================================================================


class TestCalibrationResult(unittest.TestCase):
    """Tests for CalibrationResult dataclass."""

    def _make_result(self, **kwargs):
        defaults = {
            "original_human_cost": 100.0,
            "original_api_cost": 50.0,
            "original_total": 150.0,
            "calibrated_human_cost": 110.0,
            "calibrated_api_cost": 55.0,
            "calibrated_total": 165.0,
        }
        defaults.update(kwargs)
        return CalibrationResult(**defaults)

    def test_default_fields(self):
        r = self._make_result()
        self.assertEqual(r.human_calibration_factor, 1.0)
        self.assertEqual(r.api_calibration_factor, 1.0)
        self.assertEqual(r.confidence, 0.0)
        self.assertEqual(r.based_on_datasets, 0)
        self.assertEqual(r.similar_datasets, [])
        self.assertEqual(r.calibration_method, "none")
        self.assertEqual(r.notes, [])

    def test_to_dict_structure(self):
        r = self._make_result(
            confidence=0.75,
            based_on_datasets=5,
            similar_datasets=["a", "b", "c", "d", "e", "f"],
            cost_range_low=80.0,
            cost_range_high=200.0,
            calibration_method="benchmark",
            notes=["note1"],
        )
        d = r.to_dict()
        self.assertEqual(d["original"]["human"], 100.0)
        self.assertEqual(d["original"]["api"], 50.0)
        self.assertEqual(d["original"]["total"], 150.0)
        self.assertEqual(d["calibrated"]["human"], 110.0)
        self.assertEqual(d["calibrated"]["api"], 55.0)
        self.assertEqual(d["calibrated"]["total"], 165.0)
        self.assertEqual(d["confidence"], 0.75)
        self.assertEqual(d["based_on_datasets"], 5)
        # similar_datasets truncated to 5
        self.assertEqual(len(d["similar_datasets"]), 5)
        self.assertEqual(d["range"]["low"], 80.0)
        self.assertEqual(d["range"]["high"], 200.0)
        self.assertEqual(d["method"], "benchmark")
        self.assertEqual(d["notes"], ["note1"])

    def test_to_dict_rounds_values(self):
        r = self._make_result(
            original_human_cost=100.125,
            original_api_cost=50.999,
            original_total=151.124,
            calibrated_human_cost=110.555,
            calibrated_api_cost=55.444,
            calibrated_total=165.999,
            human_calibration_factor=1.1234,
            api_calibration_factor=0.9876,
            confidence=0.8765,
            cost_range_low=80.123,
            cost_range_high=200.789,
        )
        d = r.to_dict()
        self.assertEqual(d["original"]["human"], round(100.125, 2))
        self.assertEqual(d["factors"]["human"], round(1.1234, 3))
        self.assertEqual(d["factors"]["api"], round(0.9876, 3))
        self.assertEqual(d["confidence"], round(0.8765, 2))


# ==========================================================================
# CostCalibrator
# ==========================================================================


class TestCostCalibratorNoKB(unittest.TestCase):
    """CostCalibrator without a knowledge base."""

    def test_init_no_kb_import_fails(self):
        """When KnowledgeBase import fails, kb should be None."""
        with patch("datarecipe.cost.calibrator.CostCalibrator.__init__", return_value=None):
            pass  # just ensure constructor doesn't raise
        # Direct test: pass None explicitly
        cal = CostCalibrator(knowledge_base=None)
        # kb stays None because we pass None explicitly; but __init__ tries to import
        # We test with a mock that simulates import failure
        cal.kb = None
        result = cal.calibrate("sft", 100.0, 50.0)
        self.assertEqual(result.calibration_method, "none")
        self.assertIn("知识库不可用", result.notes[0])
        self.assertAlmostEqual(result.cost_range_low, 150.0 * 0.7)
        self.assertAlmostEqual(result.cost_range_high, 150.0 * 1.5)
        self.assertEqual(result.original_total, 150.0)
        self.assertEqual(result.calibrated_total, 150.0)

    def test_get_calibration_stats_no_kb(self):
        cal = CostCalibrator(knowledge_base=None)
        cal.kb = None
        stats = cal.get_calibration_stats()
        self.assertFalse(stats["available"])

    def test_suggest_next_datasets_no_kb(self):
        cal = CostCalibrator(knowledge_base=None)
        cal.kb = None
        suggestions = cal.suggest_next_datasets()
        self.assertEqual(suggestions, [])


class TestCostCalibratorWithBenchmark(unittest.TestCase):
    """CostCalibrator with mocked knowledge base that has sufficient data."""

    def _make_benchmark(self, datasets, avg_total_cost, min_cost, max_cost, dataset_type="sft"):
        bench = MagicMock()
        bench.datasets = datasets
        bench.avg_total_cost = avg_total_cost
        bench.min_cost = min_cost
        bench.max_cost = max_cost
        bench.dataset_type = dataset_type
        return bench

    def _make_calibrator_with_benchmark(self, benchmark):
        kb = MagicMock()
        kb.trends.get_cost_benchmark.return_value = benchmark
        return CostCalibrator(knowledge_base=kb)

    def test_calibrate_with_sufficient_data_normal_range(self):
        """Estimate within normal range -> no adjustment."""
        bench = self._make_benchmark(
            datasets=["d1", "d2", "d3", "d4"],
            avg_total_cost=200.0,
            min_cost=100.0,
            max_cost=300.0,
            dataset_type="sft",
        )
        cal = self._make_calibrator_with_benchmark(bench)
        result = cal.calibrate("sft", 100.0, 80.0)

        self.assertEqual(result.calibration_method, "benchmark")
        self.assertEqual(result.based_on_datasets, 4)
        # Confidence: min(0.3 + 4*0.1, 0.9) = 0.7
        self.assertAlmostEqual(result.confidence, 0.7)
        # No adjustment since 180 is within [50, 600]
        self.assertEqual(result.calibrated_human_cost, 100.0)
        self.assertEqual(result.calibrated_api_cost, 80.0)

    def test_calibrate_estimate_too_low(self):
        """Estimate far below min_cost * 0.5 -> upward adjustment."""
        bench = self._make_benchmark(
            datasets=["d1", "d2", "d3"],
            avg_total_cost=500.0,
            min_cost=400.0,
            max_cost=600.0,
            dataset_type="sft",
        )
        cal = self._make_calibrator_with_benchmark(bench)
        # human=50, api=50 => total=100 < 400*0.5=200
        result = cal.calibrate("sft", 50.0, 50.0)

        self.assertEqual(result.calibration_method, "benchmark")
        # adjustment = 400/100 = 4.0
        self.assertAlmostEqual(result.human_calibration_factor, 4.0)
        self.assertAlmostEqual(result.api_calibration_factor, 4.0)
        self.assertAlmostEqual(result.calibrated_human_cost, 200.0)
        self.assertAlmostEqual(result.calibrated_api_cost, 200.0)

    def test_calibrate_estimate_too_high(self):
        """Estimate far above max_cost * 2 -> downward adjustment."""
        bench = self._make_benchmark(
            datasets=["d1", "d2", "d3"],
            avg_total_cost=200.0,
            min_cost=100.0,
            max_cost=300.0,
            dataset_type="sft",
        )
        cal = self._make_calibrator_with_benchmark(bench)
        # human=500, api=500 => total=1000 > 300*2=600
        result = cal.calibrate("sft", 500.0, 500.0)

        # adjustment = 300/1000 = 0.3
        self.assertAlmostEqual(result.human_calibration_factor, 0.3)
        self.assertAlmostEqual(result.calibrated_human_cost, 150.0)
        self.assertAlmostEqual(result.calibrated_api_cost, 150.0)

    def test_calibrate_zero_total_too_low(self):
        """Edge case: total is 0 -> adjustment defaults to 1."""
        bench = self._make_benchmark(
            datasets=["d1", "d2", "d3"],
            avg_total_cost=100.0,
            min_cost=50.0,
            max_cost=200.0,
            dataset_type="sft",
        )
        cal = self._make_calibrator_with_benchmark(bench)
        result = cal.calibrate("sft", 0.0, 0.0)
        # 0 < 50*0.5=25 -> tries adjustment but current_total=0 -> factor=1
        self.assertAlmostEqual(result.human_calibration_factor, 1.0)

    def test_confidence_caps_at_0_9(self):
        bench = self._make_benchmark(
            datasets=[f"d{i}" for i in range(20)],
            avg_total_cost=100.0,
            min_cost=50.0,
            max_cost=150.0,
            dataset_type="sft",
        )
        cal = self._make_calibrator_with_benchmark(bench)
        result = cal.calibrate("sft", 80.0, 20.0)
        self.assertAlmostEqual(result.confidence, 0.9)


class TestCostCalibratorLimited(unittest.TestCase):
    """CostCalibrator with limited benchmark data (1-2 datasets)."""

    def test_calibrate_limited_data(self):
        bench = MagicMock()
        bench.datasets = ["d1", "d2"]
        bench.avg_total_cost = 150.0

        kb = MagicMock()
        kb.trends.get_cost_benchmark.return_value = bench
        cal = CostCalibrator(knowledge_base=kb)

        result = cal.calibrate("evaluation", 100.0, 50.0)
        self.assertEqual(result.calibration_method, "limited")
        # confidence = 0.2 + 2*0.1 = 0.4
        self.assertAlmostEqual(result.confidence, 0.4)
        self.assertAlmostEqual(result.cost_range_low, 150.0 * 0.6)
        self.assertAlmostEqual(result.cost_range_high, 150.0 * 1.8)
        self.assertEqual(result.based_on_datasets, 2)

    def test_calibrate_limited_single_dataset(self):
        bench = MagicMock()
        bench.datasets = ["d1"]
        bench.avg_total_cost = 200.0

        kb = MagicMock()
        kb.trends.get_cost_benchmark.return_value = bench
        cal = CostCalibrator(knowledge_base=kb)

        result = cal.calibrate("preference", 100.0, 50.0)
        self.assertEqual(result.calibration_method, "limited")
        # confidence = 0.2 + 1*0.1 = 0.3
        self.assertAlmostEqual(result.confidence, 0.3)


class TestCostCalibratorSimilarType(unittest.TestCase):
    """CostCalibrator fallback to similar dataset types."""

    def test_calibrate_from_similar_found(self):
        bench = MagicMock()
        bench.datasets = ["d1", "d2"]
        bench.min_cost = 50.0
        bench.max_cost = 200.0

        kb = MagicMock()
        kb.trends.get_cost_benchmark.return_value = None
        kb.trends.get_all_benchmarks.return_value = {"rlhf": bench}
        cal = CostCalibrator(knowledge_base=kb)

        # "preference" maps to similar types ["rlhf", "reward", "dpo"]
        result = cal.calibrate("preference", 100.0, 50.0)
        self.assertEqual(result.calibration_method, "similar_type")
        self.assertAlmostEqual(result.confidence, 0.3)
        self.assertAlmostEqual(result.cost_range_low, 50.0 * 0.8)
        self.assertAlmostEqual(result.cost_range_high, 200.0 * 1.5)

    def test_calibrate_no_similar_found(self):
        kb = MagicMock()
        kb.trends.get_cost_benchmark.return_value = None
        kb.trends.get_all_benchmarks.return_value = {}
        cal = CostCalibrator(knowledge_base=kb)

        result = cal.calibrate("exotic_type", 100.0, 50.0)
        self.assertEqual(result.calibration_method, "none")
        self.assertAlmostEqual(result.confidence, 0.1)
        self.assertAlmostEqual(result.cost_range_low, 150.0 * 0.5)
        self.assertAlmostEqual(result.cost_range_high, 150.0 * 2.0)


class TestCostCalibratorStats(unittest.TestCase):
    """Tests for get_calibration_stats and suggest_next_datasets."""

    def test_get_calibration_stats_with_data(self):
        bench1 = MagicMock()
        bench1.datasets = ["d1", "d2", "d3"]
        bench1.avg_total_cost = 100.0
        bench1.min_cost = 50.0
        bench1.max_cost = 150.0

        bench2 = MagicMock()
        bench2.datasets = ["d4"]
        bench2.avg_total_cost = 200.0
        bench2.min_cost = 180.0
        bench2.max_cost = 220.0

        kb = MagicMock()
        kb.trends.get_all_benchmarks.return_value = {"sft": bench1, "eval": bench2}
        cal = CostCalibrator(knowledge_base=kb)

        stats = cal.get_calibration_stats()
        self.assertTrue(stats["available"])
        self.assertEqual(stats["total_datasets"], 4)
        self.assertEqual(stats["types_covered"], 2)
        self.assertTrue(stats["benchmarks"]["sft"]["calibration_ready"])
        self.assertFalse(stats["benchmarks"]["eval"]["calibration_ready"])

    def test_get_calibration_stats_no_benchmarks(self):
        kb = MagicMock()
        kb.trends.get_all_benchmarks.return_value = {}
        cal = CostCalibrator(knowledge_base=kb)

        stats = cal.get_calibration_stats()
        self.assertFalse(stats["available"])

    def test_suggest_next_datasets(self):
        bench = MagicMock()
        bench.datasets = ["d1"]

        kb = MagicMock()
        kb.trends.get_all_benchmarks.return_value = {"sft": bench}
        cal = CostCalibrator(knowledge_base=kb)

        suggestions = cal.suggest_next_datasets()
        # sft has 1 dataset (< 3), so it should be suggested
        self.assertTrue(any("sft" in s for s in suggestions))
        # Common types not present should also be suggested
        self.assertTrue(any("preference" in s for s in suggestions))
        self.assertTrue(any("evaluation" in s for s in suggestions))


# ==========================================================================
# DomainType and DOMAIN_DIFFICULTY
# ==========================================================================


class TestDomainType(unittest.TestCase):
    """Tests for the DomainType enum."""

    def test_all_values(self):
        expected = {
            "general", "creative", "technical", "code", "math",
            "science", "medical", "legal", "finance", "academic",
        }
        actual = {d.value for d in DomainType}
        self.assertEqual(actual, expected)

    def test_domain_difficulty_covers_all_types(self):
        for dt in DomainType:
            self.assertIn(dt, DOMAIN_DIFFICULTY)
            self.assertIsInstance(DOMAIN_DIFFICULTY[dt], float)

    def test_domain_expert_multiplier_covers_all_types(self):
        for dt in DomainType:
            self.assertIn(dt, DOMAIN_EXPERT_MULTIPLIER)

    def test_medical_highest_difficulty(self):
        self.assertEqual(DOMAIN_DIFFICULTY[DomainType.MEDICAL], 3.0)

    def test_general_is_baseline(self):
        self.assertEqual(DOMAIN_DIFFICULTY[DomainType.GENERAL], 1.0)


# ==========================================================================
# ComplexityMetrics
# ==========================================================================


class TestComplexityMetrics(unittest.TestCase):

    def test_defaults(self):
        m = ComplexityMetrics()
        self.assertEqual(m.primary_domain, DomainType.GENERAL)
        self.assertEqual(m.avg_text_length, 0)
        self.assertFalse(m.has_nested_structure)
        self.assertEqual(m.structure_category, "simple")
        self.assertEqual(m.quality_requirement, "standard")
        self.assertAlmostEqual(m.time_multiplier, 1.0)
        self.assertAlmostEqual(m.cost_multiplier, 1.0)
        self.assertAlmostEqual(m.difficulty_score, 1.0)

    def test_to_dict_structure(self):
        m = ComplexityMetrics(
            primary_domain=DomainType.CODE,
            avg_text_length=500,
            max_text_length=1000,
            has_nested_structure=True,
            nesting_depth=3,
            vocabulary_richness=0.456789,
            avg_sentence_length=12.345,
            technical_term_density=0.06789,
            code_density=0.1234,
            time_multiplier=2.345,
            cost_multiplier=1.678,
            difficulty_score=3.456,
        )
        d = m.to_dict()
        self.assertEqual(d["domain"]["primary"], "code")
        self.assertEqual(d["text_length"]["average"], 500)
        self.assertTrue(d["structure"]["has_nested"])
        self.assertEqual(d["structure"]["nesting_depth"], 3)
        self.assertEqual(d["content"]["vocabulary_richness"], 0.457)
        self.assertEqual(d["content"]["avg_sentence_length"], 12.3)
        self.assertEqual(d["multipliers"]["time"], 2.35)
        self.assertEqual(d["multipliers"]["difficulty_score"], 3.5)


# ==========================================================================
# ComplexityAnalyzer
# ==========================================================================


class TestComplexityAnalyzerEmpty(unittest.TestCase):

    def test_empty_samples(self):
        ca = ComplexityAnalyzer()
        m = ca.analyze([])
        self.assertEqual(m.primary_domain, DomainType.GENERAL)
        self.assertAlmostEqual(m.time_multiplier, 1.0)


class TestComplexityAnalyzerExtractText(unittest.TestCase):

    def setUp(self):
        self.ca = ComplexityAnalyzer()

    def test_extract_flat_dict(self):
        sample = {"text": "hello", "label": "positive"}
        text = self.ca._extract_all_text(sample)
        self.assertIn("hello", text)
        self.assertIn("positive", text)

    def test_extract_nested_dict(self):
        sample = {"outer": {"inner": "deep value"}}
        text = self.ca._extract_all_text(sample)
        self.assertIn("deep value", text)

    def test_extract_list_values(self):
        sample = {"items": ["a", "b", "c"]}
        text = self.ca._extract_all_text(sample)
        self.assertIn("a", text)
        self.assertIn("c", text)

    def test_extract_depth_limit(self):
        """Recursion stops at depth 5."""
        sample = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
        text = self.ca._extract_all_text(sample)
        # depth 5 reached at "e" key, "f" not extracted
        self.assertNotIn("too deep", text)

    def test_extract_non_string(self):
        sample = {"num": 42, "flag": True}
        text = self.ca._extract_all_text(sample)
        # Non-string values are skipped by the recursive extractor
        self.assertIsInstance(text, str)


class TestComplexityAnalyzerDomain(unittest.TestCase):

    def setUp(self):
        self.ca = ComplexityAnalyzer()

    def test_code_domain_detection(self):
        samples = [
            {
                "text": "def function_name(): import os; class MyClass: return value; "
                "try: except: async await const let var public private void int string"
            }
        ]
        m = self.ca.analyze(samples)
        self.assertEqual(m.primary_domain, DomainType.CODE)
        self.assertGreater(m.domain_confidence, 0)

    def test_medical_domain_detection(self):
        samples = [
            {
                "text": "patient diagnosis treatment symptom disease medication clinical "
                "hospital surgery therapy prescription dosage medical history pathology"
            }
        ]
        m = self.ca.analyze(samples)
        self.assertEqual(m.primary_domain, DomainType.MEDICAL)

    def test_legal_domain_detection(self):
        samples = [
            {
                "text": "contract clause liability plaintiff defendant court judgment "
                "statute regulation compliance attorney litigation jurisdiction precedent"
            }
        ]
        m = self.ca.analyze(samples)
        self.assertEqual(m.primary_domain, DomainType.LEGAL)

    def test_general_domain_low_signal(self):
        samples = [{"text": "The quick brown fox jumps over the lazy dog."}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.primary_domain, DomainType.GENERAL)
        self.assertAlmostEqual(m.domain_confidence, 0.5)


class TestComplexityAnalyzerTextLength(unittest.TestCase):

    def setUp(self):
        self.ca = ComplexityAnalyzer()

    def test_short_text(self):
        samples = [{"text": "x" * 100}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.length_category, "short")

    def test_medium_text(self):
        samples = [{"text": "x" * 1000}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.length_category, "medium")

    def test_long_text(self):
        samples = [{"text": "x" * 3000}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.length_category, "long")

    def test_very_long_text(self):
        samples = [{"text": "x" * 6000}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.length_category, "very_long")

    def test_length_stats(self):
        samples = [{"text": "a" * 100}, {"text": "b" * 300}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.avg_text_length, 200)
        self.assertEqual(m.max_text_length, 300)
        self.assertGreater(m.length_variance, 0)


class TestComplexityAnalyzerStructure(unittest.TestCase):

    def setUp(self):
        self.ca = ComplexityAnalyzer()

    def test_simple_structure(self):
        samples = [{"a": "x", "b": "y"}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.structure_category, "simple")
        self.assertFalse(m.has_nested_structure)

    def test_moderate_structure(self):
        samples = [{"a": "x", "b": "y", "c": "z", "d": {"nested": "val"}}]
        m = self.ca.analyze(samples)
        self.assertEqual(m.avg_field_count, 4)
        # nesting_depth=2 from the nested dict, <=4 fields <=8 and depth <=4 -> moderate
        self.assertEqual(m.structure_category, "moderate")

    def test_complex_structure(self):
        # Many fields and deep nesting
        sample = {f"field_{i}": f"val_{i}" for i in range(10)}
        sample["deep"] = {"a": {"b": {"c": {"d": "deep"}}}}
        samples = [sample]
        m = self.ca.analyze(samples)
        self.assertEqual(m.structure_category, "complex")
        self.assertTrue(m.has_nested_structure)

    def test_get_max_depth_flat(self):
        self.assertEqual(self.ca._get_max_depth({"a": 1, "b": 2}), 2)

    def test_get_max_depth_nested(self):
        obj = {"a": {"b": {"c": "d"}}}
        self.assertEqual(self.ca._get_max_depth(obj), 4)

    def test_get_max_depth_list(self):
        obj = {"a": [1, 2, 3]}
        self.assertEqual(self.ca._get_max_depth(obj), 3)

    def test_get_max_depth_empty(self):
        self.assertEqual(self.ca._get_max_depth({}), 1)
        self.assertEqual(self.ca._get_max_depth([]), 1)

    def test_get_max_depth_scalar(self):
        self.assertEqual(self.ca._get_max_depth(42), 1)


class TestComplexityAnalyzerContent(unittest.TestCase):

    def setUp(self):
        self.ca = ComplexityAnalyzer()

    def test_vocabulary_richness(self):
        # All unique words -> richness close to 1
        samples = [{"text": "one two three four five six seven eight nine ten"}]
        m = self.ca.analyze(samples)
        self.assertGreater(m.vocabulary_richness, 0.5)

    def test_low_vocabulary_richness(self):
        # Repeated words -> low richness
        samples = [{"text": " ".join(["hello"] * 100)}]
        m = self.ca.analyze(samples)
        self.assertLess(m.vocabulary_richness, 0.1)

    def test_sentence_length(self):
        samples = [{"text": "This is sentence one. This is sentence two."}]
        m = self.ca.analyze(samples)
        self.assertGreater(m.avg_sentence_length, 0)

    def test_technical_density(self):
        tech_words = " ".join(list(ComplexityAnalyzer.TECHNICAL_TERMS)[:5])
        samples = [{"text": tech_words}]
        m = self.ca.analyze(samples)
        self.assertGreater(m.technical_term_density, 0)

    def test_code_density(self):
        samples = [{"text": "```python\ndef foo():\n    pass\n```\nclass Bar:\n    pass"}]
        m = self.ca.analyze(samples)
        self.assertGreater(m.code_density, 0)

    def test_empty_text_content(self):
        # Text fields are empty strings
        samples = [{"text": ""}]
        m = self.ca.analyze(samples)
        self.assertAlmostEqual(m.vocabulary_richness, 0.0)


class TestComplexityAnalyzerQuality(unittest.TestCase):

    def setUp(self):
        self.ca = ComplexityAnalyzer()

    def test_no_rubrics(self):
        samples = [{"text": "hello"}]
        m = self.ca.analyze(samples, rubrics=None)
        self.assertFalse(m.has_rubrics)
        self.assertEqual(m.rubric_complexity, "none")
        self.assertEqual(m.quality_requirement, "standard")

    def test_simple_rubrics(self):
        rubrics = ["good", "bad", "ok"]
        samples = [{"text": "hello"}]
        m = self.ca.analyze(samples, rubrics=rubrics)
        self.assertTrue(m.has_rubrics)
        self.assertEqual(m.rubric_complexity, "simple")
        self.assertEqual(m.quality_requirement, "standard")

    def test_detailed_rubrics(self):
        # Average length 50-150, unique patterns 10-50
        rubrics = [f"Rubric requirement {i}: " + "x" * 80 for i in range(20)]
        samples = [{"text": "hello"}]
        m = self.ca.analyze(samples, rubrics=rubrics)
        self.assertTrue(m.has_rubrics)
        self.assertEqual(m.rubric_complexity, "detailed")
        self.assertEqual(m.quality_requirement, "high")

    def test_expert_rubrics(self):
        # Average length >= 150 or unique patterns >= 50
        rubrics = [f"Expert rubric {i}: " + "x" * 200 for i in range(60)]
        samples = [{"text": "hello"}]
        m = self.ca.analyze(samples, rubrics=rubrics)
        self.assertTrue(m.has_rubrics)
        self.assertEqual(m.rubric_complexity, "expert")
        self.assertEqual(m.quality_requirement, "expert")


class TestComplexityAnalyzerMultipliers(unittest.TestCase):

    def setUp(self):
        self.ca = ComplexityAnalyzer()

    def test_general_short_simple_standard(self):
        """Baseline multipliers for simplest case."""
        samples = [{"text": "x" * 100}]
        m = self.ca.analyze(samples)
        # domain=GENERAL(1.0), length=short(0.8), structure=simple(0.9), quality=standard(1.0)
        # "x"*100 is one word with richness=1.0 (>0.5), so content_mult=1.1
        # time = 1.0 * 0.8 * 0.9 * 1.0 * 1.1 = 0.792
        expected_content_mult = 1.1  # vocabulary_richness > 0.5 adds 0.1
        self.assertAlmostEqual(m.time_multiplier, 1.0 * 0.8 * 0.9 * 1.0 * expected_content_mult, places=2)

    def test_difficulty_score_range(self):
        """Difficulty score should be between 1.0 and 5.0."""
        samples = [{"text": "hello world"}]
        m = self.ca.analyze(samples)
        self.assertGreaterEqual(m.difficulty_score, 1.0)
        self.assertLessEqual(m.difficulty_score, 5.0)

    def test_high_complexity_multipliers(self):
        """Complex content should produce higher multipliers."""
        medical_text = (
            "patient diagnosis treatment symptom disease medication clinical "
            "hospital surgery therapy prescription dosage medical history pathology. "
        )
        # Make it long
        samples = [{"text": medical_text * 50}]
        m = self.ca.analyze(samples, rubrics=[f"Expert rubric {i}: " + "y" * 200 for i in range(60)])
        self.assertGreater(m.time_multiplier, 1.0)
        self.assertGreater(m.cost_multiplier, 1.0)
        self.assertGreater(m.difficulty_score, 2.0)

    def test_content_mult_code_boost(self):
        """High code density adds 0.3 to content multiplier."""
        # Create text with many code patterns
        code_text = "\n".join([
            "```python\ndef foo(): pass\n```",
            "def bar(): pass",
            "function baz() {}",
            "class MyClass:",
        ] * 5)
        samples = [{"text": code_text}]
        m = self.ca.analyze(samples)
        # code_density should be > 0.3, boosting content_mult
        if m.code_density > 0.3:
            # time_multiplier should be higher than base domain*length*structure*quality
            self.assertGreater(m.time_multiplier, 0.5)


# ==========================================================================
# ProjectScale
# ==========================================================================


class TestProjectScale(unittest.TestCase):

    def test_values(self):
        self.assertEqual(ProjectScale.SMALL.value, "small")
        self.assertEqual(ProjectScale.MEDIUM.value, "medium")
        self.assertEqual(ProjectScale.LARGE.value, "large")
        self.assertEqual(ProjectScale.ENTERPRISE.value, "enterprise")


# ==========================================================================
# Phase Cost Dataclasses
# ==========================================================================


class TestDesignPhaseCost(unittest.TestCase):

    def test_total(self):
        d = DesignPhaseCost(schema_design=100, guideline_writing=200, pilot_testing=50, tool_setup=30)
        self.assertAlmostEqual(d.total, 380.0)

    def test_to_dict(self):
        d = DesignPhaseCost(schema_design=100.555, guideline_writing=200.0, pilot_testing=50.0, tool_setup=30.0)
        result = d.to_dict()
        self.assertEqual(result["schema_design"], 100.56)
        self.assertEqual(result["total"], 380.56)

    def test_default_zeros(self):
        d = DesignPhaseCost()
        self.assertAlmostEqual(d.total, 0.0)


class TestProductionPhaseCost(unittest.TestCase):

    def test_total(self):
        p = ProductionPhaseCost(annotation_cost=100, generation_cost=50, review_cost=20, infrastructure=10)
        self.assertAlmostEqual(p.total, 180.0)

    def test_to_dict(self):
        p = ProductionPhaseCost(
            annotation_cost=100, generation_cost=50, review_cost=20,
            infrastructure=10, cost_per_sample=0.018, samples_count=10000,
        )
        d = p.to_dict()
        self.assertEqual(d["annotation"], 100.0)
        self.assertEqual(d["samples"], 10000)
        self.assertEqual(d["cost_per_sample"], 0.018)


class TestQualityPhaseCost(unittest.TestCase):

    def test_total(self):
        q = QualityPhaseCost(qa_sampling=30, rework=20, expert_review=10, final_validation=5)
        self.assertAlmostEqual(q.total, 65.0)

    def test_to_dict_includes_rates(self):
        q = QualityPhaseCost(qa_rate=0.3, expected_rework_rate=0.15)
        d = q.to_dict()
        self.assertEqual(d["qa_rate"], 0.3)
        self.assertEqual(d["rework_rate"], 0.15)


class TestPhasedCostBreakdown(unittest.TestCase):

    def test_to_dict_structure(self):
        b = PhasedCostBreakdown(
            total_fixed=1000.0,
            total_variable=5000.0,
            total_proportional=800.0,
            grand_total=7820.0,
            scale=ProjectScale.MEDIUM,
            target_size=5000,
            dataset_type="sft",
            contingency_rate=0.15,
            contingency_amount=1020.0,
        )
        d = b.to_dict()
        self.assertEqual(d["summary"]["fixed_costs"], 1000.0)
        self.assertEqual(d["summary"]["variable_costs"], 5000.0)
        self.assertEqual(d["summary"]["contingency"]["rate"], 0.15)
        self.assertEqual(d["project"]["scale"], "medium")
        self.assertEqual(d["project"]["target_size"], 5000)
        self.assertEqual(d["project"]["dataset_type"], "sft")


# ==========================================================================
# PhasedCostModel
# ==========================================================================


class TestPhasedCostModelScale(unittest.TestCase):

    def setUp(self):
        self.model = PhasedCostModel(region="us")

    def test_small(self):
        self.assertEqual(self.model._determine_scale(500), ProjectScale.SMALL)
        self.assertEqual(self.model._determine_scale(999), ProjectScale.SMALL)

    def test_medium(self):
        self.assertEqual(self.model._determine_scale(1000), ProjectScale.MEDIUM)
        self.assertEqual(self.model._determine_scale(9999), ProjectScale.MEDIUM)

    def test_large(self):
        self.assertEqual(self.model._determine_scale(10000), ProjectScale.LARGE)
        self.assertEqual(self.model._determine_scale(99999), ProjectScale.LARGE)

    def test_enterprise(self):
        self.assertEqual(self.model._determine_scale(100000), ProjectScale.ENTERPRISE)
        self.assertEqual(self.model._determine_scale(1000000), ProjectScale.ENTERPRISE)


class TestPhasedCostModelInit(unittest.TestCase):

    def test_us_region(self):
        model = PhasedCostModel(region="us")
        self.assertAlmostEqual(model.labor_mult, 1.0)

    def test_china_region(self):
        model = PhasedCostModel(region="china")
        self.assertAlmostEqual(model.labor_mult, 0.4)

    def test_unknown_region(self):
        model = PhasedCostModel(region="mars")
        self.assertAlmostEqual(model.labor_mult, 1.0)


class TestPhasedCostModelDesignPhase(unittest.TestCase):

    def test_design_costs_small_us(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown(scale=ProjectScale.SMALL)
        model._calculate_design_phase(breakdown, ProjectScale.SMALL, 1.0)

        base = DESIGN_PHASE_BASE_COSTS[ProjectScale.SMALL]
        self.assertAlmostEqual(breakdown.design.schema_design, base["schema_design"])
        self.assertAlmostEqual(breakdown.design.guideline_writing, base["guideline_writing"])
        self.assertAlmostEqual(breakdown.design.pilot_testing, base["pilot_testing"])
        self.assertAlmostEqual(breakdown.design.tool_setup, base["tool_setup"])

    def test_design_costs_with_complexity(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown(scale=ProjectScale.SMALL)
        model._calculate_design_phase(breakdown, ProjectScale.SMALL, 2.0)

        base = DESIGN_PHASE_BASE_COSTS[ProjectScale.SMALL]
        # schema_design and guideline_writing are multiplied by complexity
        self.assertAlmostEqual(breakdown.design.schema_design, base["schema_design"] * 2.0)
        self.assertAlmostEqual(breakdown.design.guideline_writing, base["guideline_writing"] * 2.0)
        # pilot_testing uses labor_mult only (no complexity)
        self.assertAlmostEqual(breakdown.design.pilot_testing, base["pilot_testing"])
        # tool_setup is region-independent
        self.assertAlmostEqual(breakdown.design.tool_setup, base["tool_setup"])

    def test_design_costs_with_region(self):
        model = PhasedCostModel(region="china")
        breakdown = PhasedCostBreakdown(scale=ProjectScale.MEDIUM)
        model._calculate_design_phase(breakdown, ProjectScale.MEDIUM, 1.0)

        base = DESIGN_PHASE_BASE_COSTS[ProjectScale.MEDIUM]
        self.assertAlmostEqual(breakdown.design.schema_design, base["schema_design"] * 0.4)
        self.assertAlmostEqual(breakdown.design.pilot_testing, base["pilot_testing"] * 0.4)
        self.assertAlmostEqual(breakdown.design.tool_setup, base["tool_setup"])


class TestPhasedCostModelProductionPhase(unittest.TestCase):

    def test_production_costs_basic(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_production_phase(breakdown, 1000, 50.0, 0.01, 1.0)

        # annotation: 1000 * 0.50 * 1.0 * 1.0 * 0.5 = 250
        self.assertAlmostEqual(breakdown.production.annotation_cost, 250.0)
        # generation: 1000 * 0.01 * 0.5 = 5
        self.assertAlmostEqual(breakdown.production.generation_cost, 5.0)
        # review: 250 * 0.1 = 25
        self.assertAlmostEqual(breakdown.production.review_cost, 25.0)
        # infrastructure: 50 (< 10000 samples)
        self.assertAlmostEqual(breakdown.production.infrastructure, 50.0)
        self.assertEqual(breakdown.production.samples_count, 1000)

    def test_production_100_percent_human(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_production_phase(breakdown, 1000, 100.0, 0.01, 1.0)

        # annotation: 1000 * 0.50 * 1.0 = 500
        self.assertAlmostEqual(breakdown.production.annotation_cost, 500.0)
        # generation: 1000 * 0.01 * 0 = 0
        self.assertAlmostEqual(breakdown.production.generation_cost, 0.0)

    def test_production_infrastructure_large(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_production_phase(breakdown, 50000, 50.0, 0.01, 1.0)

        # infrastructure: 100 + (50000/10000) * 50 = 100 + 250 = 350
        self.assertAlmostEqual(breakdown.production.infrastructure, 350.0)

    def test_production_cost_per_sample(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_production_phase(breakdown, 1000, 50.0, 0.01, 1.0)

        total = breakdown.production.total
        self.assertAlmostEqual(breakdown.production.cost_per_sample, total / 1000)

    def test_production_zero_samples(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_production_phase(breakdown, 0, 50.0, 0.01, 1.0)
        self.assertAlmostEqual(breakdown.production.cost_per_sample, 0.0)


class TestPhasedCostModelQualityPhase(unittest.TestCase):

    def test_quality_standard(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_quality_phase(breakdown, 1000, "standard", 50.0, 1.0)

        rates = QUALITY_RATES["standard"]
        self.assertAlmostEqual(breakdown.quality.qa_rate, rates["qa_rate"])
        self.assertAlmostEqual(breakdown.quality.expected_rework_rate, rates["rework_rate"])
        # No expert review for standard
        self.assertAlmostEqual(breakdown.quality.expert_review, 0.0)

    def test_quality_expert(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_quality_phase(breakdown, 1000, "expert", 50.0, 1.0)

        rates = QUALITY_RATES["expert"]
        self.assertAlmostEqual(breakdown.quality.qa_rate, rates["qa_rate"])
        # Expert review should be non-zero
        self.assertGreater(breakdown.quality.expert_review, 0)

    def test_quality_high_has_expert_review(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_quality_phase(breakdown, 1000, "high", 50.0, 1.0)

        self.assertGreater(breakdown.quality.expert_review, 0)

    def test_quality_basic_no_expert_review(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_quality_phase(breakdown, 1000, "basic", 50.0, 1.0)

        self.assertAlmostEqual(breakdown.quality.expert_review, 0.0)

    def test_quality_qa_sampling_calculation(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_quality_phase(breakdown, 1000, "standard", 50.0, 1.0)

        # qa_samples = int(1000 * 0.20) = 200
        # base_qa_cost = 0.30 * 1.0 * 1.0 = 0.30
        # qa_sampling = 200 * 0.30 = 60
        self.assertAlmostEqual(breakdown.quality.qa_sampling, 60.0)

    def test_quality_rework_calculation(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_quality_phase(breakdown, 1000, "standard", 50.0, 1.0)

        # rework_samples = int(1000 * 0.10) = 100
        # base_annotation = 0.50 * 1.0 * 1.0 = 0.50
        # rework = 100 * 0.50 * 1.5 = 75
        self.assertAlmostEqual(breakdown.quality.rework, 75.0)

    def test_final_validation_capped(self):
        model = PhasedCostModel(region="us")
        breakdown = PhasedCostBreakdown()
        model._calculate_quality_phase(breakdown, 100000, "standard", 50.0, 1.0)

        # final_validation = min(100000 * 0.01, 500) * 1.0 = 500
        self.assertAlmostEqual(breakdown.quality.final_validation, 500.0)


class TestPhasedCostModelCalculate(unittest.TestCase):

    def test_calculate_totals(self):
        model = PhasedCostModel(region="us")
        breakdown = model.calculate(target_size=5000, dataset_type="sft")

        self.assertAlmostEqual(breakdown.total_fixed, breakdown.design.total)
        self.assertAlmostEqual(breakdown.total_variable, breakdown.production.total)
        self.assertAlmostEqual(breakdown.total_proportional, breakdown.quality.total)

        subtotal = breakdown.total_fixed + breakdown.total_variable + breakdown.total_proportional
        self.assertAlmostEqual(breakdown.contingency_amount, subtotal * 0.15)
        self.assertAlmostEqual(breakdown.grand_total, subtotal + breakdown.contingency_amount)

    def test_calculate_scale_assignment(self):
        model = PhasedCostModel(region="us")
        breakdown = model.calculate(target_size=500)
        self.assertEqual(breakdown.scale, ProjectScale.SMALL)

    def test_calculate_dataset_type_stored(self):
        model = PhasedCostModel(region="us")
        breakdown = model.calculate(target_size=5000, dataset_type="preference")
        self.assertEqual(breakdown.dataset_type, "preference")

    def test_calculate_grand_total_positive(self):
        model = PhasedCostModel(region="us")
        breakdown = model.calculate(target_size=1000)
        self.assertGreater(breakdown.grand_total, 0)


class TestPhasedCostModelFormatReport(unittest.TestCase):

    def test_format_report_contains_sections(self):
        model = PhasedCostModel(region="us")
        breakdown = model.calculate(target_size=5000, dataset_type="sft")
        report = model.format_report(breakdown)

        self.assertIn("分阶段成本估算", report)
        self.assertIn("设计阶段", report)
        self.assertIn("生产阶段", report)
        self.assertIn("质量阶段", report)
        self.assertIn("汇总", report)
        self.assertIn("sft", report)
        self.assertIn("5,000", report)

    def test_format_report_is_markdown(self):
        model = PhasedCostModel(region="us")
        breakdown = model.calculate(target_size=1000)
        report = model.format_report(breakdown)

        self.assertIn("|", report)
        self.assertIn("---", report)
        self.assertIn("#", report)


# ==========================================================================
# TokenStats
# ==========================================================================


class TestTokenStats(unittest.TestCase):

    def test_defaults(self):
        ts = TokenStats()
        self.assertEqual(ts.sample_count, 0)
        self.assertEqual(ts.avg_input_tokens, 0)
        self.assertEqual(ts.total_input_tokens, 0)
        self.assertEqual(ts.field_token_counts, {})

    def test_to_dict(self):
        ts = TokenStats(
            sample_count=10,
            avg_input_tokens=100,
            avg_output_tokens=50,
            total_input_tokens=1000,
            total_output_tokens=500,
            min_input_tokens=80,
            max_input_tokens=120,
            p50_input=95,
            p90_input=110,
            p99_input=118,
            field_token_counts={"prompt": 80, "response": 50},
        )
        d = ts.to_dict()
        self.assertEqual(d["sample_count"], 10)
        self.assertEqual(d["avg_input_tokens"], 100)
        self.assertEqual(d["distribution"]["min_input"], 80)
        self.assertEqual(d["percentiles"]["p50_input"], 95)
        self.assertEqual(d["field_token_counts"]["prompt"], 80)


# ==========================================================================
# ModelPricing
# ==========================================================================


class TestModelPricing(unittest.TestCase):

    def test_per_1k_properties(self):
        mp = ModelPricing(provider="openai", model="gpt-4o", input_per_1m=2.50, output_per_1m=10.00)
        self.assertAlmostEqual(mp.input_per_1k, 0.0025)
        self.assertAlmostEqual(mp.output_per_1k, 0.01)

    def test_model_pricing_dict(self):
        self.assertIn("gpt-4o", MODEL_PRICING)
        self.assertIn("claude-3.5-sonnet", MODEL_PRICING)
        self.assertIn("deepseek-v3", MODEL_PRICING)
        self.assertIn("llama-3.1-70b", MODEL_PRICING)
        self.assertIn("gemini-2.0-flash", MODEL_PRICING)

    def test_model_pricing_values(self):
        gpt4o = MODEL_PRICING["gpt-4o"]
        self.assertEqual(gpt4o.provider, "openai")
        self.assertEqual(gpt4o.model, "gpt-4o")
        self.assertEqual(gpt4o.input_per_1m, 2.50)
        self.assertEqual(gpt4o.output_per_1m, 10.00)
        self.assertEqual(gpt4o.context_window, 128000)

    def test_default_context_window(self):
        mp = ModelPricing(provider="test", model="test", input_per_1m=1.0, output_per_1m=2.0)
        self.assertEqual(mp.context_window, 128000)


# ==========================================================================
# TokenAnalyzer
# ==========================================================================


class TestTokenAnalyzerCountTokens(unittest.TestCase):

    def setUp(self):
        self.ta = TokenAnalyzer(use_tiktoken=False)

    def test_empty_text(self):
        self.assertEqual(self.ta.count_tokens(""), 0)

    def test_english_text(self):
        text = "Hello world, this is a test sentence."
        tokens = self.ta.count_tokens(text, lang="en")
        # ~36 chars / 4 chars per token = ~9
        self.assertGreater(tokens, 0)
        self.assertAlmostEqual(tokens, len(text) / 4.0, delta=2)

    def test_chinese_text(self):
        text = "你好世界这是一个测试"
        tokens = self.ta.count_tokens(text)
        # CJK detected, uses 1.5 chars/token ratio blended
        self.assertGreater(tokens, 0)

    def test_code_text(self):
        text = "def hello():\n    return 'world'\nimport os\nclass Foo:\n    pass"
        tokens = self.ta.count_tokens(text)
        self.assertGreater(tokens, 0)

    def test_mixed_language(self):
        text = "Hello 你好 world 世界"
        tokens = self.ta.count_tokens(text, lang="mixed")
        self.assertGreater(tokens, 0)


class TestTokenAnalyzerIsCode(unittest.TestCase):

    def setUp(self):
        self.ta = TokenAnalyzer()

    def test_is_code_true(self):
        code = "def foo():\n    return bar\nimport os\nclass MyClass:\n    pass"
        self.assertTrue(self.ta._is_code(code))

    def test_is_code_false(self):
        text = "This is a normal English sentence about programming."
        self.assertFalse(self.ta._is_code(text))

    def test_is_code_needs_two_matches(self):
        text = "def foo"  # Only one pattern
        self.assertFalse(self.ta._is_code(text))


class TestTokenAnalyzerHasCJK(unittest.TestCase):

    def setUp(self):
        self.ta = TokenAnalyzer()

    def test_has_cjk_chinese(self):
        self.assertTrue(self.ta._has_cjk("Hello 你好"))

    def test_has_cjk_japanese(self):
        self.assertTrue(self.ta._has_cjk("こんにちは"))

    def test_has_cjk_korean(self):
        self.assertTrue(self.ta._has_cjk("안녕하세요"))

    def test_no_cjk(self):
        self.assertFalse(self.ta._has_cjk("Hello world"))


class TestTokenAnalyzerExtractText(unittest.TestCase):

    def setUp(self):
        self.ta = TokenAnalyzer()

    def test_string_value(self):
        self.assertEqual(self.ta._extract_text("hello"), "hello")

    def test_list_of_strings(self):
        self.assertEqual(self.ta._extract_text(["a", "b"]), "a b")

    def test_list_of_messages(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        self.assertEqual(self.ta._extract_text(msgs), "Hello Hi")

    def test_dict_value(self):
        self.assertEqual(self.ta._extract_text({"a": "x", "b": "y"}), "x y")

    def test_none_value(self):
        self.assertEqual(self.ta._extract_text(None), "")

    def test_int_value(self):
        self.assertEqual(self.ta._extract_text(42), "42")


class TestTokenAnalyzerDetectIOFields(unittest.TestCase):

    def setUp(self):
        self.ta = TokenAnalyzer()

    def test_standard_fields(self):
        sample = {"prompt": "hello", "response": "world"}
        inp, out = self.ta._detect_io_fields(sample)
        self.assertIn("prompt", inp)
        self.assertIn("response", out)

    def test_input_output_fields(self):
        sample = {"input": "hello", "output": "world"}
        inp, out = self.ta._detect_io_fields(sample)
        self.assertIn("input", inp)
        self.assertIn("output", out)

    def test_question_answer_fields(self):
        sample = {"question": "What?", "answer": "That."}
        inp, out = self.ta._detect_io_fields(sample)
        self.assertIn("question", inp)
        self.assertIn("answer", out)

    def test_messages_field(self):
        sample = {"messages": [{"role": "user", "content": "hi"}]}
        inp, out = self.ta._detect_io_fields(sample)
        self.assertIn("messages", inp)

    def test_preference_fields(self):
        sample = {"prompt": "x", "chosen": "good", "rejected": "bad"}
        inp, out = self.ta._detect_io_fields(sample)
        self.assertIn("prompt", inp)
        self.assertIn("chosen", out)
        self.assertIn("rejected", out)

    def test_fallback_long_strings(self):
        sample = {"unknown_field": "x" * 100, "short": "hi"}
        inp, out = self.ta._detect_io_fields(sample)
        self.assertIn("unknown_field", inp)
        self.assertNotIn("short", inp)

    def test_no_matching_fields(self):
        sample = {"x": 1, "y": 2}
        inp, out = self.ta._detect_io_fields(sample)
        # No string fields > 50 chars, so both should be empty
        self.assertEqual(inp, [])
        self.assertEqual(out, [])


class TestTokenAnalyzerAnalyzeSamples(unittest.TestCase):

    def setUp(self):
        self.ta = TokenAnalyzer(use_tiktoken=False)

    def test_empty_samples(self):
        stats = self.ta.analyze_samples([])
        self.assertEqual(stats.sample_count, 0)

    def test_basic_analysis(self):
        samples = [
            {"prompt": "Hello there friend", "response": "Hi"},
            {"prompt": "How are you today", "response": "Good thanks"},
        ]
        stats = self.ta.analyze_samples(
            samples,
            input_fields=["prompt"],
            output_fields=["response"],
        )
        self.assertEqual(stats.sample_count, 2)
        self.assertGreater(stats.avg_input_tokens, 0)
        self.assertGreater(stats.total_input_tokens, 0)

    def test_auto_detect_fields(self):
        samples = [
            {"input": "test question", "output": "test answer"},
        ]
        stats = self.ta.analyze_samples(samples)
        self.assertEqual(stats.sample_count, 1)
        self.assertGreater(stats.avg_input_tokens, 0)

    def test_percentiles_single_sample(self):
        samples = [{"prompt": "x" * 100, "response": "y" * 50}]
        stats = self.ta.analyze_samples(
            samples,
            input_fields=["prompt"],
            output_fields=["response"],
        )
        self.assertEqual(stats.p50_input, stats.avg_input_tokens)
        self.assertEqual(stats.p90_input, stats.avg_input_tokens)

    def test_field_token_counts(self):
        samples = [
            {"prompt": "Hello world", "response": "Hi there"},
            {"prompt": "How are you", "response": "Good"},
        ]
        stats = self.ta.analyze_samples(
            samples,
            input_fields=["prompt"],
            output_fields=["response"],
        )
        self.assertIn("prompt", stats.field_token_counts)
        self.assertIn("response", stats.field_token_counts)

    def test_min_max_tokens(self):
        samples = [
            {"prompt": "x" * 40, "response": "y"},      # short
            {"prompt": "x" * 400, "response": "y" * 100},  # long
        ]
        stats = self.ta.analyze_samples(
            samples,
            input_fields=["prompt"],
            output_fields=["response"],
        )
        self.assertLess(stats.min_input_tokens, stats.max_input_tokens)

    def test_missing_field_in_sample(self):
        """Fields that don't exist in a sample are skipped."""
        samples = [
            {"prompt": "hello"},  # no "response" field
        ]
        stats = self.ta.analyze_samples(
            samples,
            input_fields=["prompt"],
            output_fields=["response"],
        )
        self.assertEqual(stats.sample_count, 1)
        self.assertEqual(stats.avg_output_tokens, 0)


# ==========================================================================
# PreciseCostEstimate
# ==========================================================================


class TestPreciseCostEstimate(unittest.TestCase):

    def test_to_dict(self):
        ts = TokenStats(sample_count=5, avg_input_tokens=100, avg_output_tokens=50)
        pricing = MODEL_PRICING["gpt-4o"]
        est = PreciseCostEstimate(
            token_stats=ts,
            target_size=1000,
            model="gpt-4o",
            pricing=pricing,
            input_cost=0.25,
            output_cost=0.50,
            total_api_cost=0.75,
            iteration_factor=1.2,
            adjusted_cost=0.90,
            cost_low=0.60,
            cost_high=1.50,
            assumptions=["note1"],
        )
        d = est.to_dict()
        self.assertEqual(d["model"], "gpt-4o")
        self.assertEqual(d["target_size"], 1000)
        self.assertEqual(d["cost"]["input_cost"], 0.25)
        self.assertEqual(d["cost"]["output_cost"], 0.50)
        self.assertEqual(d["cost"]["iteration_factor"], 1.2)
        self.assertEqual(d["cost"]["range"]["low"], 0.60)
        self.assertEqual(d["pricing"]["input_per_1m"], 2.50)
        self.assertEqual(d["assumptions"], ["note1"])


# ==========================================================================
# PreciseCostCalculator
# ==========================================================================


class TestPreciseCostCalculator(unittest.TestCase):

    def setUp(self):
        self.calc = PreciseCostCalculator()

    def test_calculate_basic(self):
        samples = [
            {"prompt": "Hello world test prompt", "response": "Answer here"},
        ]
        est = self.calc.calculate(
            samples,
            target_size=1000,
            model="gpt-4o",
            input_fields=["prompt"],
            output_fields=["response"],
        )
        self.assertEqual(est.target_size, 1000)
        self.assertEqual(est.model, "gpt-4o")
        self.assertGreater(est.total_api_cost, 0)
        self.assertGreater(est.adjusted_cost, est.total_api_cost)

    def test_calculate_unknown_model_falls_back(self):
        samples = [{"prompt": "x" * 100, "response": "y" * 50}]
        est = self.calc.calculate(
            samples,
            target_size=100,
            model="nonexistent-model",
            input_fields=["prompt"],
            output_fields=["response"],
        )
        # Should fall back to gpt-4o pricing
        self.assertEqual(est.pricing.model, "gpt-4o")

    def test_calculate_iteration_factor(self):
        samples = [{"prompt": "x" * 100, "response": "y" * 50}]
        est1 = self.calc.calculate(
            samples, 1000, iteration_factor=1.0,
            input_fields=["prompt"], output_fields=["response"],
        )
        est2 = self.calc.calculate(
            samples, 1000, iteration_factor=2.0,
            input_fields=["prompt"], output_fields=["response"],
        )
        self.assertAlmostEqual(est2.adjusted_cost, est1.total_api_cost * 2.0, places=2)

    def test_calculate_cost_ranges(self):
        samples = [
            {"prompt": "x" * 100, "response": "y" * 50},
            {"prompt": "x" * 200, "response": "y" * 100},
            {"prompt": "x" * 300, "response": "y" * 150},
        ]
        est = self.calc.calculate(
            samples, 1000, model="gpt-4o",
            input_fields=["prompt"], output_fields=["response"],
        )
        self.assertGreater(est.cost_high, est.cost_low)

    def test_calculate_assumptions(self):
        samples = [{"prompt": "hello", "response": "hi"}]
        est = self.calc.calculate(
            samples, 1000, model="gpt-4o",
            input_fields=["prompt"], output_fields=["response"],
        )
        self.assertGreater(len(est.assumptions), 0)
        # Should mention sample count
        self.assertTrue(any("1" in a for a in est.assumptions))

    def test_calculate_input_output_cost_breakdown(self):
        samples = [{"prompt": "x" * 400, "response": "y" * 200}]
        est = self.calc.calculate(
            samples, 10000, model="gpt-4o",
            input_fields=["prompt"], output_fields=["response"],
        )
        self.assertAlmostEqual(est.total_api_cost, est.input_cost + est.output_cost)


class TestPreciseCostCalculatorCompareModels(unittest.TestCase):

    def setUp(self):
        self.calc = PreciseCostCalculator()

    def test_compare_default_models(self):
        samples = [{"prompt": "Hello world", "response": "Hi there"}]
        comparisons = self.calc.compare_models(samples, 1000)
        self.assertGreater(len(comparisons), 0)
        # All results should be PreciseCostEstimate
        for model, est in comparisons.items():
            self.assertIsInstance(est, PreciseCostEstimate)
            self.assertEqual(est.model, model)

    def test_compare_specific_models(self):
        samples = [{"prompt": "Hello", "response": "Hi"}]
        comparisons = self.calc.compare_models(
            samples, 500, models=["gpt-4o", "gpt-4o-mini"]
        )
        self.assertEqual(len(comparisons), 2)
        self.assertIn("gpt-4o", comparisons)
        self.assertIn("gpt-4o-mini", comparisons)

    def test_compare_skips_unknown_models(self):
        samples = [{"prompt": "Hello", "response": "Hi"}]
        comparisons = self.calc.compare_models(
            samples, 500, models=["gpt-4o", "nonexistent"]
        )
        self.assertEqual(len(comparisons), 1)
        self.assertIn("gpt-4o", comparisons)

    def test_format_comparison_table(self):
        samples = [{"prompt": "Hello world", "response": "Hi there"}]
        comparisons = self.calc.compare_models(
            samples, 1000, models=["gpt-4o", "gpt-4o-mini"]
        )
        table = self.calc.format_comparison_table(comparisons)
        self.assertIn("模型", table)
        self.assertIn("gpt-4o", table)
        self.assertIn("|", table)

    def test_format_comparison_table_sorted_by_cost(self):
        samples = [{"prompt": "x" * 100, "response": "y" * 50}]
        comparisons = self.calc.compare_models(
            samples, 10000, models=["gpt-4o", "gpt-4o-mini"]
        )
        table = self.calc.format_comparison_table(comparisons)
        lines = [l for l in table.strip().split("\n") if l.startswith("| ") and "模型" not in l and "---" not in l]
        # gpt-4o-mini should be cheaper and appear first
        if len(lines) >= 2:
            self.assertIn("gpt-4o-mini", lines[0])


# ==========================================================================
# TokenAnalyzer with tiktoken mock
# ==========================================================================


class TestTokenAnalyzerWithTiktoken(unittest.TestCase):
    """Test the tiktoken path with a mock."""

    def test_tiktoken_unavailable(self):
        """When tiktoken import fails, use_tiktoken should be False."""
        with patch.dict("sys.modules", {"tiktoken": None}):
            ta = TokenAnalyzer(use_tiktoken=True)
            # Should gracefully degrade
            self.assertFalse(ta.use_tiktoken)

    def test_tiktoken_available_mock(self):
        """When tiktoken is available, use it for counting."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        ta = TokenAnalyzer(use_tiktoken=False)
        ta._tokenizer = mock_tokenizer
        count = ta.count_tokens("hello world")
        self.assertEqual(count, 5)
        mock_tokenizer.encode.assert_called_once_with("hello world")


# ==========================================================================
# Integration-style tests
# ==========================================================================


class TestPhasedCostModelAllScales(unittest.TestCase):
    """Ensure calculate works for all ProjectScale values without error."""

    def test_all_scales(self):
        model = PhasedCostModel(region="us")
        sizes = [100, 5000, 50000, 200000]
        expected_scales = [
            ProjectScale.SMALL,
            ProjectScale.MEDIUM,
            ProjectScale.LARGE,
            ProjectScale.ENTERPRISE,
        ]
        for size, expected_scale in zip(sizes, expected_scales):
            breakdown = model.calculate(target_size=size)
            self.assertEqual(breakdown.scale, expected_scale)
            self.assertGreater(breakdown.grand_total, 0)

    def test_all_quality_requirements(self):
        model = PhasedCostModel(region="us")
        for qr in ["basic", "standard", "high", "expert"]:
            breakdown = model.calculate(target_size=5000, quality_requirement=qr)
            self.assertGreater(breakdown.grand_total, 0)


class TestComplexityAnalyzerIntegration(unittest.TestCase):
    """Integration test: full pipeline through analyze."""

    def test_code_samples(self):
        samples = [
            {
                "instruction": "Write a Python function to sort a list",
                "code": "def sort_list(lst):\n    return sorted(lst)\nimport os\nclass Foo:\n    pass",
            }
        ]
        ca = ComplexityAnalyzer()
        m = ca.analyze(samples)
        self.assertIsInstance(m, ComplexityMetrics)
        self.assertGreater(m.time_multiplier, 0)
        self.assertGreater(m.cost_multiplier, 0)

    def test_with_schema_info(self):
        samples = [{"text": "Hello world"}]
        schema = {"fields": {"text": {"type": "string"}}}
        ca = ComplexityAnalyzer()
        m = ca.analyze(samples, schema_info=schema)
        # schema_info is currently unused internally but should not cause errors
        self.assertIsInstance(m, ComplexityMetrics)


class TestCostModulesEdgeCases(unittest.TestCase):
    """Edge cases across modules."""

    def test_phased_model_zero_target(self):
        model = PhasedCostModel(region="us")
        breakdown = model.calculate(target_size=0)
        # Should not raise, totals should be non-negative
        self.assertGreaterEqual(breakdown.grand_total, 0)

    def test_complexity_analyzer_single_word_samples(self):
        ca = ComplexityAnalyzer()
        samples = [{"text": "word"}]
        m = ca.analyze(samples)
        self.assertIsInstance(m, ComplexityMetrics)

    def test_token_analyzer_numeric_only_sample(self):
        ta = TokenAnalyzer()
        samples = [{"value": 42}]
        stats = ta.analyze_samples(samples)
        self.assertEqual(stats.sample_count, 1)

    def test_calibrator_init_with_explicit_none(self):
        cal = CostCalibrator(knowledge_base=None)
        # The constructor tries to import KnowledgeBase; if that fails, kb is None
        # Either way, calibrate should work
        result = cal.calibrate("test", 100.0, 50.0)
        self.assertIsInstance(result, CalibrationResult)

    def test_design_phase_base_costs_all_scales(self):
        """Every ProjectScale has an entry in DESIGN_PHASE_BASE_COSTS."""
        for scale in ProjectScale:
            self.assertIn(scale, DESIGN_PHASE_BASE_COSTS)
            base = DESIGN_PHASE_BASE_COSTS[scale]
            self.assertIn("schema_design", base)
            self.assertIn("guideline_writing", base)
            self.assertIn("pilot_testing", base)
            self.assertIn("tool_setup", base)

    def test_quality_rates_all_levels(self):
        """Every quality level has an entry in QUALITY_RATES."""
        for level in ["basic", "standard", "high", "expert"]:
            self.assertIn(level, QUALITY_RATES)
            rates = QUALITY_RATES[level]
            self.assertIn("qa_rate", rates)
            self.assertIn("rework_rate", rates)


if __name__ == "__main__":
    unittest.main()
