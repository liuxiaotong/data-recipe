"""Unit tests for IndustryBenchmarkGenerator.

Tests BenchmarkComparison dataclass, generate(), to_markdown(), and to_dict() methods.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from datarecipe.generators.industry_benchmark import (
    BenchmarkComparison,
    IndustryBenchmarkGenerator,
)


# ---------- Stub for DatasetCatalog ----------


@dataclass
class StubBenchmarkResult:
    """Pre-built benchmark result dict for mocking DatasetCatalog.compare_with_benchmark."""

    available: bool = True
    benchmark: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    similar_projects: list = field(default_factory=list)

    def to_dict(self) -> dict:
        result: dict[str, Any] = {"available": self.available}
        if self.available:
            result["benchmark"] = self.benchmark
            result["comparison"] = self.comparison
            result["similar_projects"] = self.similar_projects
        return result


def _make_benchmark_result(
    cost_rating: str = "average",
    human_rating: str = "typical",
    cost_vs_avg: str = "+0%",
) -> dict:
    """Create a realistic benchmark result dict like DatasetCatalog returns."""
    return {
        "available": True,
        "benchmark": {
            "category": "preference",
            "description": "RLHF/DPO preference data",
            "typical_cost_per_sample": {
                "min": 1.0,
                "avg": 3.0,
                "max": 10.0,
            },
            "typical_project": {
                "size": 10000,
                "cost": 30000,
                "duration_days": 60,
            },
            "avg_human_percentage": 85.0,
            "source": "Industry estimate",
        },
        "your_project": {
            "sample_count": 5000,
            "total_cost": 15000,
            "cost_per_sample": 3.0,
            "human_percentage": 70.0,
        },
        "comparison": {
            "cost_rating": cost_rating,
            "cost_explanation": "Cost is within normal range",
            "cost_vs_avg": cost_vs_avg,
            "human_rating": human_rating,
            "human_explanation": "Human ratio is typical",
        },
        "similar_projects": [
            {"name": "Anthropic/hh-rlhf", "size": 170000, "estimated_cost": 500000},
            {"name": "stanfordnlp/SHP", "size": 385000, "estimated_cost": 0},
        ],
    }


def _make_unavailable_benchmark_result() -> dict:
    """Create a result for when no benchmark is available."""
    return {
        "available": False,
        "reason": "No benchmark for category: unknown_type",
    }


# ==================== BenchmarkComparison Dataclass ====================


class TestBenchmarkComparisonDataclass(unittest.TestCase):
    """Test BenchmarkComparison dataclass defaults and construction."""

    def test_default_values(self):
        bc = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=1000,
            total_cost=5000.0,
            cost_per_sample=5.0,
            human_percentage=70.0,
        )
        self.assertFalse(bc.benchmark_available)
        self.assertEqual(bc.benchmark_description, "")
        self.assertEqual(bc.benchmark_cost_range, {})
        self.assertEqual(bc.benchmark_human_percentage, 0.0)
        self.assertEqual(bc.cost_rating, "")
        self.assertEqual(bc.cost_vs_benchmark, "")
        self.assertEqual(bc.cost_explanation, "")
        self.assertEqual(bc.human_rating, "")
        self.assertEqual(bc.human_explanation, "")
        self.assertEqual(bc.similar_projects, [])
        self.assertEqual(bc.recommendations, [])

    def test_full_construction(self):
        bc = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="sft",
            sample_count=5000,
            total_cost=10000.0,
            cost_per_sample=2.0,
            human_percentage=80.0,
            benchmark_available=True,
            benchmark_description="SFT data",
            benchmark_cost_range={"min": 0.5, "avg": 2.0, "max": 8.0},
            benchmark_human_percentage=70.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="Normal range",
            human_rating="more_human",
            human_explanation="Higher human ratio",
            similar_projects=[{"name": "ds1", "size": 100, "estimated_cost": 500}],
            recommendations=["Rec 1"],
        )
        self.assertTrue(bc.benchmark_available)
        self.assertEqual(bc.benchmark_description, "SFT data")
        self.assertEqual(bc.cost_rating, "average")
        self.assertEqual(len(bc.similar_projects), 1)
        self.assertEqual(len(bc.recommendations), 1)

    def test_mutable_defaults_are_independent(self):
        """Ensure default mutable fields are not shared across instances."""
        bc1 = BenchmarkComparison(
            dataset_id="a", dataset_type="sft",
            sample_count=100, total_cost=100, cost_per_sample=1.0,
            human_percentage=50.0,
        )
        bc2 = BenchmarkComparison(
            dataset_id="b", dataset_type="sft",
            sample_count=200, total_cost=200, cost_per_sample=1.0,
            human_percentage=50.0,
        )
        bc1.recommendations.append("Only for bc1")
        self.assertEqual(bc2.recommendations, [])


# ==================== generate() ====================


class TestIndustryBenchmarkGenerate(unittest.TestCase):
    """Test IndustryBenchmarkGenerator.generate() method."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()
        self.base_kwargs = {
            "dataset_id": "test/preference-dataset",
            "dataset_type": "preference",
            "sample_count": 5000,
            "reproduction_cost": {"total": 15000, "human": 10000, "machine": 5000},
            "human_percentage": 70.0,
        }

    def test_generate_returns_benchmark_comparison(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertIsInstance(result, BenchmarkComparison)

    def test_generate_sets_basic_fields(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertEqual(result.dataset_id, "test/preference-dataset")
        self.assertEqual(result.dataset_type, "preference")
        self.assertEqual(result.sample_count, 5000)
        self.assertEqual(result.total_cost, 15000)
        self.assertEqual(result.human_percentage, 70.0)

    def test_generate_computes_cost_per_sample(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertEqual(result.cost_per_sample, 3.0)  # 15000 / 5000

    def test_generate_cost_per_sample_rounded(self):
        kwargs = {**self.base_kwargs, "reproduction_cost": {"total": 10000}, "sample_count": 3}
        result = self.gen.generate(**kwargs)
        # 10000 / 3 = 3333.333... -> should round to 3333.33
        self.assertEqual(result.cost_per_sample, round(10000 / 3, 2))

    def test_generate_zero_sample_count(self):
        kwargs = {**self.base_kwargs, "sample_count": 0}
        result = self.gen.generate(**kwargs)
        self.assertEqual(result.cost_per_sample, 0)

    def test_generate_missing_total_in_cost(self):
        kwargs = {**self.base_kwargs, "reproduction_cost": {"human": 5000}}
        result = self.gen.generate(**kwargs)
        self.assertEqual(result.total_cost, 0)
        self.assertEqual(result.cost_per_sample, 0)

    def test_generate_always_has_recommendations(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertIsInstance(result.recommendations, list)
        self.assertGreater(len(result.recommendations), 0)


class TestIndustryBenchmarkGenerateWithCatalog(unittest.TestCase):
    """Test generate() with mocked DatasetCatalog."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()
        self.base_kwargs = {
            "dataset_id": "test/preference-dataset",
            "dataset_type": "preference",
            "sample_count": 5000,
            "reproduction_cost": {"total": 15000, "human": 10000, "machine": 5000},
            "human_percentage": 70.0,
        }

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_with_benchmark_available(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.return_value = _make_benchmark_result(
            cost_rating="average", human_rating="typical", cost_vs_avg="+0%"
        )

        result = self.gen.generate(**self.base_kwargs)

        self.assertTrue(result.benchmark_available)
        self.assertEqual(result.benchmark_description, "RLHF/DPO preference data")
        self.assertEqual(result.benchmark_cost_range, {"min": 1.0, "avg": 3.0, "max": 10.0})
        self.assertEqual(result.benchmark_human_percentage, 85.0)
        self.assertEqual(result.cost_rating, "average")
        self.assertEqual(result.cost_vs_benchmark, "+0%")
        self.assertEqual(result.human_rating, "typical")

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_with_benchmark_unavailable(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.return_value = _make_unavailable_benchmark_result()

        result = self.gen.generate(**self.base_kwargs)

        self.assertFalse(result.benchmark_available)
        self.assertEqual(result.benchmark_description, "")

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_populates_similar_projects(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.return_value = _make_benchmark_result()

        result = self.gen.generate(**self.base_kwargs)

        self.assertEqual(len(result.similar_projects), 2)
        self.assertEqual(result.similar_projects[0]["name"], "Anthropic/hh-rlhf")

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_import_error_gracefully_handled(self, MockCatalog):
        MockCatalog.side_effect = ImportError("Module not found")

        result = self.gen.generate(**self.base_kwargs)

        self.assertFalse(result.benchmark_available)
        self.assertIsInstance(result, BenchmarkComparison)

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_attribute_error_gracefully_handled(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.side_effect = AttributeError("No such attr")

        result = self.gen.generate(**self.base_kwargs)

        self.assertFalse(result.benchmark_available)

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_key_error_gracefully_handled(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.side_effect = KeyError("missing key")

        result = self.gen.generate(**self.base_kwargs)

        self.assertFalse(result.benchmark_available)

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_type_error_gracefully_handled(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.side_effect = TypeError("bad type")

        result = self.gen.generate(**self.base_kwargs)

        self.assertFalse(result.benchmark_available)


# ==================== _generate_recommendations() ====================


class TestGenerateRecommendations(unittest.TestCase):
    """Test the _generate_recommendations() private method."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()

    def _make_comparison(self, **overrides) -> BenchmarkComparison:
        defaults = {
            "dataset_id": "test/ds",
            "dataset_type": "preference",
            "sample_count": 5000,
            "total_cost": 15000.0,
            "cost_per_sample": 3.0,
            "human_percentage": 70.0,
            "benchmark_available": True,
            "cost_rating": "average",
            "human_rating": "typical",
        }
        defaults.update(overrides)
        return BenchmarkComparison(**defaults)

    def test_no_benchmark_available(self):
        comp = self._make_comparison(benchmark_available=False)
        recs = self.gen._generate_recommendations(comp)
        self.assertEqual(len(recs), 1)
        self.assertIn("暂无", recs[0])

    def test_below_average_cost_recommendations(self):
        comp = self._make_comparison(cost_rating="below_average")
        recs = self.gen._generate_recommendations(comp)
        quality_recs = [r for r in recs if "质量" in r]
        self.assertGreater(len(quality_recs), 0)

    def test_high_cost_recommendations(self):
        comp = self._make_comparison(cost_rating="high")
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("成本高于" in r for r in recs))
        self.assertTrue(any("自动化" in r for r in recs))
        self.assertTrue(any("优化" in r for r in recs))
        self.assertTrue(any("分阶段" in r for r in recs))

    def test_above_average_cost_recommendation(self):
        comp = self._make_comparison(cost_rating="above_average")
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("略高" in r for r in recs))

    def test_more_human_recommendation(self):
        comp = self._make_comparison(human_rating="more_human")
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("AI 辅助" in r for r in recs))

    def test_more_automated_recommendation(self):
        comp = self._make_comparison(human_rating="more_automated")
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("人工抽检" in r for r in recs))

    def test_small_sample_recommendation(self):
        comp = self._make_comparison(sample_count=50)
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("样本量较小" in r for r in recs))

    def test_large_sample_recommendation(self):
        comp = self._make_comparison(sample_count=20000)
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("大规模" in r for r in recs))

    def test_medium_sample_no_scale_recommendation(self):
        comp = self._make_comparison(sample_count=5000)
        recs = self.gen._generate_recommendations(comp)
        scale_recs = [r for r in recs if "样本量较小" in r or "大规模" in r]
        self.assertEqual(len(scale_recs), 0)

    def test_average_cost_typical_human_no_scale_default_recommendation(self):
        """When all params are 'normal', the fallback recommendation fires."""
        comp = self._make_comparison(
            cost_rating="average",
            human_rating="typical",
            sample_count=5000,
        )
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("按计划推进" in r for r in recs))

    def test_below_average_cost_small_sample_combines(self):
        """Multiple recommendation categories combine."""
        comp = self._make_comparison(
            cost_rating="below_average",
            sample_count=50,
        )
        recs = self.gen._generate_recommendations(comp)
        has_quality = any("质量" in r for r in recs)
        has_small = any("样本量较小" in r for r in recs)
        self.assertTrue(has_quality)
        self.assertTrue(has_small)

    def test_high_cost_more_human_large_sample_combines(self):
        """Multiple recommendation categories combine."""
        comp = self._make_comparison(
            cost_rating="high",
            human_rating="more_human",
            sample_count=20000,
        )
        recs = self.gen._generate_recommendations(comp)
        has_cost = any("成本高于" in r for r in recs)
        has_human = any("AI 辅助" in r for r in recs)
        has_scale = any("大规模" in r for r in recs)
        self.assertTrue(has_cost)
        self.assertTrue(has_human)
        self.assertTrue(has_scale)

    def test_boundary_sample_count_100(self):
        """Sample count exactly 100 should NOT trigger small sample recommendation."""
        comp = self._make_comparison(sample_count=100)
        recs = self.gen._generate_recommendations(comp)
        self.assertFalse(any("样本量较小" in r for r in recs))

    def test_boundary_sample_count_10000(self):
        """Sample count exactly 10000 should NOT trigger large sample recommendation."""
        comp = self._make_comparison(sample_count=10000)
        recs = self.gen._generate_recommendations(comp)
        self.assertFalse(any("大规模" in r for r in recs))

    def test_boundary_sample_count_99(self):
        """Sample count 99 triggers small sample recommendation."""
        comp = self._make_comparison(sample_count=99)
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("样本量较小" in r for r in recs))

    def test_boundary_sample_count_10001(self):
        """Sample count 10001 triggers large sample recommendation."""
        comp = self._make_comparison(sample_count=10001)
        recs = self.gen._generate_recommendations(comp)
        self.assertTrue(any("大规模" in r for r in recs))


# ==================== to_markdown() ====================


class TestIndustryBenchmarkMarkdownNoBenchmark(unittest.TestCase):
    """Test to_markdown() when benchmark is not available."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()
        self.comparison = BenchmarkComparison(
            dataset_id="test/unknown-dataset",
            dataset_type="unknown_type",
            sample_count=500,
            total_cost=2500.0,
            cost_per_sample=5.0,
            human_percentage=60.0,
            benchmark_available=False,
            recommendations=["暂无该类型数据集的行业基准数据，建议收集更多同类项目信息"],
        )

    def test_markdown_contains_header(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("# test/unknown-dataset 行业基准对比", md)

    def test_markdown_contains_dataset_type(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("unknown_type", md)

    def test_markdown_contains_project_summary_table(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("项目概况", md)
        self.assertIn("样本数量", md)
        self.assertIn("500", md)
        self.assertIn("$2,500", md)
        self.assertIn("$5.00", md)
        self.assertIn("60%", md)

    def test_markdown_contains_unavailable_warning(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("基准数据不可用", md)
        self.assertIn("暂无 `unknown_type` 类型数据集的行业基准数据", md)

    def test_markdown_contains_recommendations(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("建议", md)
        self.assertIn("暂无该类型数据集的行业基准数据", md)

    def test_markdown_contains_footer(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("基准数据来源于公开信息和行业调研，仅供参考", md)

    def test_markdown_does_not_contain_benchmark_section(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertNotIn("行业基准\n", md.replace("## 行业基准对比", ""))
        self.assertNotIn("成本定位", md)
        self.assertNotIn("对比分析", md)


class TestIndustryBenchmarkMarkdownWithBenchmark(unittest.TestCase):
    """Test to_markdown() when benchmark IS available."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()
        self.comparison = BenchmarkComparison(
            dataset_id="test/pref-dataset",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="RLHF/DPO preference data",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="Cost is within normal range",
            human_rating="typical",
            human_explanation="Human ratio is typical",
            similar_projects=[
                {"name": "Anthropic/hh-rlhf", "size": 170000, "estimated_cost": 500000},
            ],
            recommendations=["项目参数符合行业惯例，可按计划推进"],
        )

    def test_markdown_contains_header(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("# test/pref-dataset 行业基准对比", md)

    def test_markdown_contains_generation_time(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("生成时间:", md)

    def test_markdown_contains_project_summary(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("5,000", md)
        self.assertIn("$15,000", md)
        self.assertIn("$3.00", md)
        self.assertIn("70%", md)

    def test_markdown_contains_benchmark_section(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("## 行业基准", md)
        self.assertIn("RLHF/DPO preference data", md)

    def test_markdown_contains_cost_range(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("$1.00/条", md)
        self.assertIn("$3.00/条", md)
        self.assertIn("$10.00/条", md)

    def test_markdown_contains_benchmark_human_percentage(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("行业平均人工比例", md)
        self.assertIn("85%", md)

    def test_markdown_contains_comparison_analysis(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("对比分析", md)
        self.assertIn("成本定位", md)

    def test_markdown_contains_ascii_bar(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("低 │", md)
        self.assertIn("│ 高", md)

    def test_markdown_contains_cost_rating(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("成本评级", md)
        self.assertIn("Cost is within normal range", md)

    def test_markdown_contains_cost_vs_benchmark(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("与行业平均差异", md)
        self.assertIn("+0%", md)

    def test_markdown_contains_human_explanation(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("人工比例评估", md)
        self.assertIn("Human ratio is typical", md)

    def test_markdown_contains_similar_projects(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("类似项目参考", md)
        self.assertIn("Anthropic/hh-rlhf", md)
        self.assertIn("170,000", md)
        self.assertIn("$500,000", md)

    def test_markdown_contains_recommendations(self):
        md = self.gen.to_markdown(self.comparison)
        self.assertIn("建议", md)
        self.assertIn("项目参数符合行业惯例，可按计划推进", md)


class TestIndustryBenchmarkMarkdownCostRatingIcons(unittest.TestCase):
    """Test that cost rating icons render correctly per rating."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()

    def _make_comparison_with_rating(self, cost_rating: str) -> BenchmarkComparison:
        return BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
            cost_rating=cost_rating,
            cost_vs_benchmark="+0%",
            cost_explanation="explanation",
            human_rating="typical",
            human_explanation="human explanation",
            recommendations=["rec"],
        )

    def test_below_average_green_icon(self):
        comp = self._make_comparison_with_rating("below_average")
        md = self.gen.to_markdown(comp)
        # The line with cost rating should have green icon
        for line in md.split("\n"):
            if "成本评级" in line:
                self.assertIn("\U0001f7e2", line)  # green circle
                break

    def test_average_green_icon(self):
        comp = self._make_comparison_with_rating("average")
        md = self.gen.to_markdown(comp)
        for line in md.split("\n"):
            if "成本评级" in line:
                self.assertIn("\U0001f7e2", line)
                break

    def test_above_average_yellow_icon(self):
        comp = self._make_comparison_with_rating("above_average")
        md = self.gen.to_markdown(comp)
        for line in md.split("\n"):
            if "成本评级" in line:
                self.assertIn("\U0001f7e1", line)  # yellow circle
                break

    def test_high_red_icon(self):
        comp = self._make_comparison_with_rating("high")
        md = self.gen.to_markdown(comp)
        for line in md.split("\n"):
            if "成本评级" in line:
                self.assertIn("\U0001f534", line)  # red circle
                break


class TestIndustryBenchmarkMarkdownEdgeCases(unittest.TestCase):
    """Test to_markdown() edge cases."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()

    def test_no_similar_projects(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="explanation",
            human_rating="typical",
            human_explanation="expl",
            similar_projects=[],
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        self.assertNotIn("类似项目参考", md)

    def test_similar_projects_with_zero_cost_excluded(self):
        """Projects with zero estimated_cost should not be rendered."""
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="explanation",
            human_rating="",
            human_explanation="",
            similar_projects=[
                {"name": "FreeDS", "size": 1000, "estimated_cost": 0},
            ],
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        # The table header should still appear since there's a project entry
        self.assertIn("类似项目参考", md)
        # But "FreeDS" should not appear as a data row since cost is 0
        self.assertNotIn("FreeDS", md)

    def test_no_human_rating_omits_human_section(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="explanation",
            human_rating="",
            human_explanation="",
            similar_projects=[],
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        self.assertNotIn("人工比例评估", md)

    def test_ascii_bar_cost_below_min(self):
        """When cost is below benchmark min, marker should be at position 0."""
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=500.0,
            cost_per_sample=0.1,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
            cost_rating="below_average",
            cost_vs_benchmark="-97%",
            cost_explanation="Very low cost",
            human_rating="typical",
            human_explanation="expl",
            similar_projects=[],
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        # Should still render without error
        self.assertIn("低 │", md)
        self.assertIn("│ 高", md)

    def test_ascii_bar_cost_above_max(self):
        """When cost is above benchmark max, marker should be clamped."""
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=100000.0,
            cost_per_sample=20.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
            cost_rating="high",
            cost_vs_benchmark="+567%",
            cost_explanation="Very high cost",
            human_rating="typical",
            human_explanation="expl",
            similar_projects=[],
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        self.assertIn("低 │", md)
        self.assertIn("│ 高", md)

    def test_ascii_bar_equal_min_max(self):
        """Edge case: min == max in cost range."""
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={"min": 3.0, "avg": 3.0, "max": 3.0},
            benchmark_human_percentage=85.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="explanation",
            human_rating="typical",
            human_explanation="expl",
            similar_projects=[],
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        # Should handle division by zero gracefully (range_width becomes 1)
        self.assertIn("低 │", md)

    def test_ascii_bar_empty_cost_range(self):
        """Edge case: cost_range dict is empty."""
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="Test",
            benchmark_cost_range={},
            benchmark_human_percentage=85.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="explanation",
            human_rating="typical",
            human_explanation="expl",
            similar_projects=[],
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        # Should handle missing keys with defaults
        self.assertIn("$0.00/条", md)

    def test_multiple_recommendations_rendered(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            recommendations=["Rec A", "Rec B", "Rec C"],
        )
        md = self.gen.to_markdown(comp)
        self.assertIn("- Rec A", md)
        self.assertIn("- Rec B", md)
        self.assertIn("- Rec C", md)

    def test_large_sample_count_formatted(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=1000000,
            total_cost=5000000.0,
            cost_per_sample=5.0,
            human_percentage=70.0,
            recommendations=["rec"],
        )
        md = self.gen.to_markdown(comp)
        self.assertIn("1,000,000", md)
        self.assertIn("$5,000,000", md)


# ==================== to_dict() ====================


class TestIndustryBenchmarkToDict(unittest.TestCase):
    """Test to_dict() conversion."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()

    def test_dict_structure_minimal(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="sft",
            sample_count=1000,
            total_cost=2000.0,
            cost_per_sample=2.0,
            human_percentage=50.0,
        )
        d = self.gen.to_dict(comp)
        self.assertIn("dataset_id", d)
        self.assertIn("dataset_type", d)
        self.assertIn("your_project", d)
        self.assertIn("benchmark", d)
        self.assertIn("comparison", d)
        self.assertIn("similar_projects", d)
        self.assertIn("recommendations", d)

    def test_dict_dataset_id_and_type(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="evaluation",
            sample_count=1000,
            total_cost=5000.0,
            cost_per_sample=5.0,
            human_percentage=80.0,
        )
        d = self.gen.to_dict(comp)
        self.assertEqual(d["dataset_id"], "test/ds")
        self.assertEqual(d["dataset_type"], "evaluation")

    def test_dict_your_project_section(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
        )
        d = self.gen.to_dict(comp)
        yp = d["your_project"]
        self.assertEqual(yp["sample_count"], 5000)
        self.assertEqual(yp["total_cost"], 15000.0)
        self.assertEqual(yp["cost_per_sample"], 3.0)
        self.assertEqual(yp["human_percentage"], 70.0)

    def test_dict_benchmark_section(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            benchmark_available=True,
            benchmark_description="RLHF data",
            benchmark_cost_range={"min": 1.0, "avg": 3.0, "max": 10.0},
            benchmark_human_percentage=85.0,
        )
        d = self.gen.to_dict(comp)
        bm = d["benchmark"]
        self.assertTrue(bm["available"])
        self.assertEqual(bm["description"], "RLHF data")
        self.assertEqual(bm["cost_range"], {"min": 1.0, "avg": 3.0, "max": 10.0})
        self.assertEqual(bm["avg_human_percentage"], 85.0)

    def test_dict_benchmark_not_available(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="unknown",
            sample_count=1000,
            total_cost=2000.0,
            cost_per_sample=2.0,
            human_percentage=50.0,
        )
        d = self.gen.to_dict(comp)
        self.assertFalse(d["benchmark"]["available"])

    def test_dict_comparison_section(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            cost_rating="average",
            cost_vs_benchmark="+0%",
            cost_explanation="Normal range",
            human_rating="typical",
            human_explanation="Typical ratio",
        )
        d = self.gen.to_dict(comp)
        cmp = d["comparison"]
        self.assertEqual(cmp["cost_rating"], "average")
        self.assertEqual(cmp["cost_vs_benchmark"], "+0%")
        self.assertEqual(cmp["cost_explanation"], "Normal range")
        self.assertEqual(cmp["human_rating"], "typical")
        self.assertEqual(cmp["human_explanation"], "Typical ratio")

    def test_dict_similar_projects(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            similar_projects=[
                {"name": "ds1", "size": 1000, "estimated_cost": 5000},
                {"name": "ds2", "size": 2000, "estimated_cost": 10000},
            ],
        )
        d = self.gen.to_dict(comp)
        self.assertEqual(len(d["similar_projects"]), 2)
        self.assertEqual(d["similar_projects"][0]["name"], "ds1")

    def test_dict_recommendations(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=5000,
            total_cost=15000.0,
            cost_per_sample=3.0,
            human_percentage=70.0,
            recommendations=["Rec A", "Rec B"],
        )
        d = self.gen.to_dict(comp)
        self.assertEqual(d["recommendations"], ["Rec A", "Rec B"])

    def test_dict_empty_fields(self):
        comp = BenchmarkComparison(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=0,
            total_cost=0.0,
            cost_per_sample=0.0,
            human_percentage=0.0,
        )
        d = self.gen.to_dict(comp)
        self.assertEqual(d["your_project"]["sample_count"], 0)
        self.assertEqual(d["your_project"]["total_cost"], 0.0)
        self.assertEqual(d["similar_projects"], [])
        self.assertEqual(d["recommendations"], [])


# ==================== Integration-style tests ====================


class TestIndustryBenchmarkIntegration(unittest.TestCase):
    """Integration-style tests testing generate -> to_markdown / to_dict pipeline."""

    def setUp(self):
        self.gen = IndustryBenchmarkGenerator()

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_then_to_markdown(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.return_value = _make_benchmark_result(
            cost_rating="above_average",
            human_rating="more_human",
            cost_vs_avg="+50%",
        )

        comparison = self.gen.generate(
            dataset_id="test/integration",
            dataset_type="preference",
            sample_count=5000,
            reproduction_cost={"total": 22500, "human": 15000, "machine": 7500},
            human_percentage=80.0,
        )
        md = self.gen.to_markdown(comparison)

        self.assertIn("# test/integration 行业基准对比", md)
        self.assertIn("5,000", md)
        self.assertIn("$22,500", md)
        self.assertIn("80%", md)
        self.assertIn("行业基准", md)
        self.assertIn("建议", md)

    @patch("datarecipe.knowledge.DatasetCatalog")
    def test_generate_then_to_dict(self, MockCatalog):
        mock_instance = MockCatalog.return_value
        mock_instance.compare_with_benchmark.return_value = _make_benchmark_result()

        comparison = self.gen.generate(
            dataset_id="test/integration",
            dataset_type="preference",
            sample_count=5000,
            reproduction_cost={"total": 15000},
            human_percentage=70.0,
        )
        d = self.gen.to_dict(comparison)

        self.assertEqual(d["dataset_id"], "test/integration")
        self.assertEqual(d["your_project"]["sample_count"], 5000)
        self.assertTrue(d["benchmark"]["available"])
        self.assertGreater(len(d["recommendations"]), 0)

    def test_generate_without_catalog_then_to_markdown(self):
        """Test full pipeline even if DatasetCatalog import fails."""
        comparison = self.gen.generate(
            dataset_id="test/no-catalog",
            dataset_type="unknown_type",
            sample_count=100,
            reproduction_cost={"total": 500},
            human_percentage=50.0,
        )
        md = self.gen.to_markdown(comparison)

        self.assertIn("# test/no-catalog 行业基准对比", md)
        self.assertIn("基准数据不可用", md)
        self.assertIn("建议", md)

    def test_generate_without_catalog_then_to_dict(self):
        comparison = self.gen.generate(
            dataset_id="test/no-catalog",
            dataset_type="unknown_type",
            sample_count=100,
            reproduction_cost={"total": 500},
            human_percentage=50.0,
        )
        d = self.gen.to_dict(comparison)

        self.assertFalse(d["benchmark"]["available"])
        self.assertEqual(d["dataset_id"], "test/no-catalog")


if __name__ == "__main__":
    unittest.main()
