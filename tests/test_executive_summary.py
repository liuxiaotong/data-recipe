"""Tests for executive summary generator.

Covers:
- ValueAssessment generation for various dataset types
- Value score calculation edge cases
- Recommendation logic
- ROI calculation
- Payback scenarios
- Risk assessment
- _find_alternatives with KnowledgeBase fallback
- _get_competitive_advantage
- to_markdown (deep-analyze pipeline)
- spec_to_markdown (analyze-spec pipeline)
- to_dict
"""

import unittest
from dataclasses import dataclass, field

from datarecipe.generators.executive_summary import (
    DATASET_TYPE_CONFIG,
    ExecutiveSummaryGenerator,
    Recommendation,
    ValueAssessment,
)

# ==================== Stub objects ====================


@dataclass
class StubComplexityMetrics:
    primary_domain: object = None
    difficulty_score: float = 2.0
    length_category: str = "medium"


class StubDomain:
    def __init__(self, value: str):
        self.value = value


@dataclass
class StubEnhancedContext:
    generated: bool = True
    tailored_use_cases: list = field(default_factory=list)
    tailored_roi_scenarios: list = field(default_factory=list)
    tailored_risks: list = field(default_factory=list)
    competitive_positioning: str = ""
    dataset_purpose_summary: str = ""


@dataclass
class StubSpecAnalysis:
    project_name: str = "Test Project"
    dataset_type: str = "evaluation"
    estimated_difficulty: str = "medium"
    estimated_human_percentage: float = 80.0
    estimated_domain: str = "general"
    description: str = "A test dataset"
    task_description: str = "Task description here"
    has_images: bool = False
    image_count: int = 0
    forbidden_items: list = field(default_factory=list)
    similar_datasets: list = field(default_factory=list)


# ==================== ValueAssessment Defaults ====================


class TestValueAssessmentDefaults(unittest.TestCase):
    def test_defaults(self):
        va = ValueAssessment()
        self.assertEqual(va.score, 5.0)
        self.assertEqual(va.recommendation, Recommendation.CONDITIONAL)
        self.assertEqual(va.roi_ratio, 1.0)
        self.assertEqual(va.risks, [])
        self.assertEqual(va.alternatives, [])


# ==================== generate() ====================


class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()
        self.base_kwargs = {
            "dataset_id": "test/dataset",
            "dataset_type": "preference",
            "sample_count": 5000,
            "reproduction_cost": {"total": 10000, "human": 8000, "api": 2000},
            "human_percentage": 80.0,
        }

    def test_generate_preference(self):
        assessment = self.gen.generate(**self.base_kwargs)
        self.assertIsInstance(assessment, ValueAssessment)
        self.assertTrue(assessment.primary_use_case)
        self.assertTrue(assessment.secondary_use_cases)
        self.assertTrue(assessment.expected_outcomes)
        self.assertTrue(assessment.risks)

    def test_generate_evaluation(self):
        kwargs = {**self.base_kwargs, "dataset_type": "evaluation"}
        assessment = self.gen.generate(**kwargs)
        self.assertIn("评测", assessment.primary_use_case)

    def test_generate_sft(self):
        kwargs = {**self.base_kwargs, "dataset_type": "sft"}
        assessment = self.gen.generate(**kwargs)
        self.assertIn("微调", assessment.primary_use_case)

    def test_generate_swe_bench(self):
        kwargs = {**self.base_kwargs, "dataset_type": "swe_bench"}
        assessment = self.gen.generate(**kwargs)
        self.assertIn("代码", assessment.primary_use_case)

    def test_generate_unknown_type(self):
        kwargs = {**self.base_kwargs, "dataset_type": "unknown_custom"}
        assessment = self.gen.generate(**kwargs)
        self.assertTrue(assessment.primary_use_case)

    def test_generate_with_enhanced_context_use_cases(self):
        """Lines 188-190: enhanced context overrides use cases."""
        ec = StubEnhancedContext(
            generated=True,
            tailored_use_cases=["Custom use 1", "Custom use 2", "Custom use 3"],
        )
        kwargs = {**self.base_kwargs, "enhanced_context": ec}
        assessment = self.gen.generate(**kwargs)
        self.assertEqual(assessment.primary_use_case, "Custom use 1")
        self.assertEqual(assessment.secondary_use_cases, ["Custom use 2", "Custom use 3"])

    def test_generate_with_enhanced_context_roi(self):
        """Lines 223-224: enhanced context overrides ROI scenarios."""
        ec = StubEnhancedContext(
            generated=True,
            tailored_roi_scenarios=["ROI scenario 1"],
        )
        kwargs = {**self.base_kwargs, "enhanced_context": ec}
        assessment = self.gen.generate(**kwargs)
        self.assertEqual(assessment.payback_scenarios, ["ROI scenario 1"])

    def test_generate_with_enhanced_context_risks(self):
        """Lines 232-233: enhanced context overrides risks."""
        ec = StubEnhancedContext(
            generated=True,
            tailored_risks=[{"level": "high", "description": "Custom risk", "mitigation": "Fix"}],
        )
        kwargs = {**self.base_kwargs, "enhanced_context": ec}
        assessment = self.gen.generate(**kwargs)
        self.assertEqual(len(assessment.risks), 1)
        self.assertEqual(assessment.risks[0]["description"], "Custom risk")

    def test_generate_with_enhanced_context_competitive(self):
        """Lines 243-244: enhanced context overrides competitive positioning."""
        ec = StubEnhancedContext(
            generated=True,
            competitive_positioning="Strong market leader",
        )
        kwargs = {**self.base_kwargs, "enhanced_context": ec}
        assessment = self.gen.generate(**kwargs)
        self.assertEqual(assessment.competitive_advantage, "Strong market leader")

    def test_generate_with_not_generated_enhanced_context(self):
        """Enhanced context with generated=False should fall back to defaults."""
        ec = StubEnhancedContext(
            generated=False,
            tailored_use_cases=["Should not be used"],
        )
        kwargs = {**self.base_kwargs, "enhanced_context": ec}
        assessment = self.gen.generate(**kwargs)
        self.assertNotEqual(assessment.primary_use_case, "Should not be used")


# ==================== _calculate_value_score ====================


class TestCalculateValueScore(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_large_dataset_bonus(self):
        """sample_count >= 10000 gets +1.5."""
        score = self.gen._calculate_value_score(
            dataset_type="preference",
            sample_count=10000,
            reproduction_cost={"total": 1000},
            human_percentage=50.0,
            complexity_metrics=None,
            config=DATASET_TYPE_CONFIG["preference"],
        )
        self.assertGreater(score, 5.0)

    def test_medium_dataset_bonus(self):
        """sample_count >= 1000 gets +1.0."""
        score = self.gen._calculate_value_score(
            dataset_type="sft",
            sample_count=1000,
            reproduction_cost={"total": 1000},
            human_percentage=50.0,
            complexity_metrics=None,
            config=DATASET_TYPE_CONFIG["sft"],
        )
        self.assertGreater(score, 5.0)

    def test_small_dataset_bonus(self):
        """sample_count >= 100 gets +0.5."""
        score = self.gen._calculate_value_score(
            dataset_type="sft",
            sample_count=100,
            reproduction_cost={"total": 100},
            human_percentage=50.0,
            complexity_metrics=None,
            config=DATASET_TYPE_CONFIG["sft"],
        )
        self.assertGreaterEqual(score, 1.0)

    def test_very_small_dataset_penalty(self):
        """sample_count < 50 gets -1.0."""
        score = self.gen._calculate_value_score(
            dataset_type="sft",
            sample_count=30,
            reproduction_cost={"total": 100},
            human_percentage=50.0,
            complexity_metrics=None,
            config=DATASET_TYPE_CONFIG.get("sft", {}),
        )
        # With sft multiplier of 1.3: 5*1.3=6.5, -1.0=5.5, +cost bonus
        self.assertIsInstance(score, float)

    def test_expensive_cost_per_sample_penalty(self):
        """cost_per_sample > 50 gets -1.0."""
        score = self.gen._calculate_value_score(
            dataset_type="sft",
            sample_count=100,
            reproduction_cost={"total": 10000},  # $100/sample
            human_percentage=50.0,
            complexity_metrics=None,
            config=DATASET_TYPE_CONFIG.get("sft", {}),
        )
        self.assertIsInstance(score, float)

    def test_complexity_good_range_bonus(self):
        """difficulty 1.5-3.0 gets +0.5."""
        metrics = StubComplexityMetrics(difficulty_score=2.0)
        score = self.gen._calculate_value_score(
            dataset_type="sft",
            sample_count=1000,
            reproduction_cost={"total": 1000},
            human_percentage=50.0,
            complexity_metrics=metrics,
            config=DATASET_TYPE_CONFIG.get("sft", {}),
        )
        self.assertIsInstance(score, float)

    def test_complexity_high_penalty(self):
        """difficulty > 4.0 gets -0.5."""
        metrics = StubComplexityMetrics(difficulty_score=5.0)
        score = self.gen._calculate_value_score(
            dataset_type="sft",
            sample_count=1000,
            reproduction_cost={"total": 1000},
            human_percentage=50.0,
            complexity_metrics=metrics,
            config=DATASET_TYPE_CONFIG.get("sft", {}),
        )
        self.assertIsInstance(score, float)

    def test_score_clamped_to_range(self):
        """Score is clamped between 1.0 and 10.0."""
        score = self.gen._calculate_value_score(
            dataset_type="preference",
            sample_count=100000,
            reproduction_cost={"total": 1},
            human_percentage=50.0,
            complexity_metrics=StubComplexityMetrics(difficulty_score=2.0),
            config=DATASET_TYPE_CONFIG["preference"],
        )
        self.assertLessEqual(score, 10.0)
        self.assertGreaterEqual(score, 1.0)

    def test_zero_sample_count_no_crash(self):
        """Zero samples should not cause division by zero."""
        score = self.gen._calculate_value_score(
            dataset_type="sft",
            sample_count=0,
            reproduction_cost={"total": 100},
            human_percentage=50.0,
            complexity_metrics=None,
            config={},
        )
        self.assertIsInstance(score, float)


# ==================== _get_recommendation ====================


class TestGetRecommendation(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_highly_recommended(self):
        rec, reason = self.gen._get_recommendation(8.5, {"total": 1000}, "sft")
        self.assertEqual(rec, Recommendation.HIGHLY_RECOMMENDED)

    def test_recommended(self):
        rec, reason = self.gen._get_recommendation(6.5, {"total": 1000}, "sft")
        self.assertEqual(rec, Recommendation.RECOMMENDED)

    def test_conditional_high_cost(self):
        """Lines 321-324: conditional with total_cost > 10000."""
        rec, reason = self.gen._get_recommendation(4.5, {"total": 15000}, "sft")
        self.assertEqual(rec, Recommendation.CONDITIONAL)
        self.assertIn("成本较高", reason)

    def test_conditional_low_cost(self):
        """Lines 325-329: conditional with lower cost."""
        rec, reason = self.gen._get_recommendation(4.5, {"total": 5000}, "sft")
        self.assertEqual(rec, Recommendation.CONDITIONAL)
        self.assertIn("具体需求", reason)

    def test_not_recommended(self):
        rec, reason = self.gen._get_recommendation(3.0, {"total": 1000}, "sft")
        self.assertEqual(rec, Recommendation.NOT_RECOMMENDED)


# ==================== _calculate_roi ====================


class TestCalculateROI(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_roi_zero_cost(self):
        """Line 347: zero cost returns special message."""
        roi, explanation = self.gen._calculate_roi("sft", {"total": 0}, 100, {})
        self.assertEqual(roi, 0)
        self.assertIn("成本为零", explanation)

    def test_roi_preference_type(self):
        roi, explanation = self.gen._calculate_roi(
            "preference", {"total": 1000}, 1000, DATASET_TYPE_CONFIG["preference"]
        )
        self.assertGreater(roi, 0)

    def test_roi_evaluation_type(self):
        roi, explanation = self.gen._calculate_roi(
            "evaluation", {"total": 1000}, 1000, DATASET_TYPE_CONFIG["evaluation"]
        )
        self.assertGreater(roi, 0)

    def test_roi_swe_bench_type(self):
        """Line 359: swe_bench base_value_per_sample = 3.0."""
        roi, explanation = self.gen._calculate_roi(
            "swe_bench", {"total": 1000}, 1000, DATASET_TYPE_CONFIG["swe_bench"]
        )
        self.assertGreater(roi, 0)

    def test_roi_unknown_type(self):
        """Line 362: fallback base_value = 0.5."""
        roi, explanation = self.gen._calculate_roi(
            "other_type", {"total": 1000}, 1000, {}
        )
        self.assertGreater(roi, 0)

    def test_roi_high_value(self):
        """Line 369: roi >= 3.0 message."""
        roi, explanation = self.gen._calculate_roi(
            "swe_bench", {"total": 100}, 1000, DATASET_TYPE_CONFIG["swe_bench"]
        )
        self.assertIn("价值高", explanation)

    def test_roi_near_breakeven(self):
        """Line 373: roi near 1.0."""
        roi, explanation = self.gen._calculate_roi(
            "other_type", {"total": 500}, 1000, {}
        )
        self.assertIsInstance(explanation, str)


# ==================== _generate_payback_scenarios ====================


class TestGeneratePaybackScenarios(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_preference_scenarios(self):
        scenarios = self.gen._generate_payback_scenarios("preference", {"total": 1000})
        self.assertEqual(len(scenarios), 3)
        self.assertIn("对齐", scenarios[0])

    def test_evaluation_scenarios(self):
        scenarios = self.gen._generate_payback_scenarios("evaluation", {"total": 1000})
        self.assertEqual(len(scenarios), 3)

    def test_sft_scenarios(self):
        scenarios = self.gen._generate_payback_scenarios("sft", {"total": 1000})
        self.assertEqual(len(scenarios), 3)

    def test_swe_bench_scenarios(self):
        """Lines 407-412: swe_bench payback scenarios."""
        scenarios = self.gen._generate_payback_scenarios("swe_bench", {"total": 1000})
        self.assertEqual(len(scenarios), 3)

    def test_unknown_type_scenarios(self):
        scenarios = self.gen._generate_payback_scenarios("custom_type", {"total": 1000})
        self.assertEqual(len(scenarios), 3)


# ==================== _assess_risks ====================


class TestAssessRisks(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_high_cost_risk(self):
        risks = self.gen._assess_risks({"total": 60000}, 50.0, None)
        levels = [r["level"] for r in risks]
        self.assertIn("高", levels)

    def test_expert_required_domain(self):
        """Lines 438-443: medical/legal/finance domain triggers expert risk."""
        metrics = StubComplexityMetrics(primary_domain=StubDomain("medical"))
        risks = self.gen._assess_risks({"total": 5000}, 50.0, metrics)
        descriptions = [r["description"] for r in risks]
        self.assertTrue(any("专家" in d for d in descriptions))

    def test_high_human_percentage_risk(self):
        risks = self.gen._assess_risks({"total": 5000}, 90.0, None)
        descriptions = [r["description"] for r in risks]
        self.assertTrue(any("质量" in d for d in descriptions))

    def test_time_intensive_risk(self):
        risks = self.gen._assess_risks({"total": 25000}, 50.0, None)
        descriptions = [r["description"] for r in risks]
        self.assertTrue(any("周期" in d for d in descriptions))

    def test_always_includes_data_freshness(self):
        risks = self.gen._assess_risks({"total": 100}, 50.0, None)
        levels = [r["level"] for r in risks]
        self.assertIn("低", levels)

    def test_domain_without_value_attr(self):
        """Line 441: domain without .value attribute uses str()."""
        metrics = StubComplexityMetrics(primary_domain="medical")
        risks = self.gen._assess_risks({"total": 5000}, 50.0, metrics)
        # Should not crash
        self.assertIsInstance(risks, list)


# ==================== _find_alternatives ====================


class TestFindAlternatives(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_preference_alternatives(self):
        """Lines 467-495: known alternatives for preference type."""
        alts = self.gen._find_alternatives("test/ds", "preference")
        self.assertTrue(len(alts) <= 3)
        for a in alts:
            self.assertIsInstance(a, str)

    def test_excludes_self(self):
        """Should exclude the dataset itself from alternatives."""
        alts = self.gen._find_alternatives("Anthropic/hh-rlhf", "preference")
        self.assertNotIn("Anthropic/hh-rlhf", alts)

    def test_evaluation_alternatives(self):
        alts = self.gen._find_alternatives("test/ds", "evaluation")
        self.assertTrue(len(alts) <= 3)

    def test_sft_alternatives(self):
        alts = self.gen._find_alternatives("test/ds", "sft")
        self.assertTrue(len(alts) <= 3)

    def test_swe_bench_alternatives(self):
        alts = self.gen._find_alternatives("test/ds", "swe_bench")
        self.assertTrue(len(alts) <= 3)

    def test_unknown_type_empty(self):
        alts = self.gen._find_alternatives("test/ds", "custom_type")
        self.assertEqual(alts, [])


# ==================== _get_competitive_advantage ====================


class TestGetCompetitiveAdvantage(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_top_org_advantage(self):
        """Lines 509-511: top org mention."""
        adv = self.gen._get_competitive_advantage("Anthropic/dataset", "preference", 1000)
        self.assertIn("Anthropic", adv)

    def test_large_dataset_advantage(self):
        """Lines 514-515: large dataset."""
        adv = self.gen._get_competitive_advantage("unknown/dataset", "sft", 10000)
        self.assertIn("规模大", adv)

    def test_medium_dataset_advantage(self):
        """Lines 516-517: medium dataset."""
        adv = self.gen._get_competitive_advantage("unknown/dataset", "sft", 1000)
        self.assertIn("规模适中", adv)

    def test_preference_type_advantage(self):
        """Lines 520-521: preference type advantage."""
        adv = self.gen._get_competitive_advantage("unknown/dataset", "preference", 100)
        self.assertIn("RLHF", adv)

    def test_swe_bench_type_advantage(self):
        """Lines 522-523: swe_bench advantage."""
        adv = self.gen._get_competitive_advantage("unknown/dataset", "swe_bench", 100)
        self.assertIn("代码", adv)

    def test_no_advantages_fallback(self):
        """Line 525: fallback when no advantages."""
        adv = self.gen._get_competitive_advantage("unknown", "custom_type", 50)
        self.assertIn("无特殊优势", adv)

    def test_no_org_in_id(self):
        """Dataset ID without '/' should still work."""
        adv = self.gen._get_competitive_advantage("nodash", "sft", 1000)
        self.assertIsInstance(adv, str)


# ==================== to_markdown ====================


class TestToMarkdown(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_to_markdown_all_recommendations(self):
        """Test markdown rendering for each recommendation level."""
        for rec in Recommendation:
            assessment = ValueAssessment(
                score=7.0,
                recommendation=rec,
                recommendation_reason="Test reason",
                primary_use_case="Test use case",
                secondary_use_cases=["Secondary 1"],
                expected_outcomes=["Outcome 1"],
                roi_ratio=2.0,
                roi_explanation="Good ROI",
                payback_scenarios=["Scenario A"],
                risks=[{"level": "中", "description": "Risk", "mitigation": "Fix"}],
                alternatives=["Alt 1"],
                competitive_advantage="Strong",
            )
            md = self.gen.to_markdown(
                assessment,
                dataset_id="test/ds",
                dataset_type="sft",
                reproduction_cost={"total": 1000, "human": 800},
            )
            self.assertIn("test/ds", md)
            self.assertIn("决策建议", md)
            self.assertIn("7.0/10", md)

    def test_to_markdown_zero_total_cost(self):
        """Line 580-583: zero total cost path."""
        assessment = ValueAssessment(
            primary_use_case="Test",
            recommendation=Recommendation.CONDITIONAL,
            risks=[],
            alternatives=[],
        )
        md = self.gen.to_markdown(
            assessment,
            dataset_id="test/ds",
            dataset_type="sft",
            reproduction_cost={"total": 0, "human": 0},
        )
        self.assertIn("$0", md)

    def test_to_markdown_no_alternatives(self):
        """Lines 649-651: no alternatives message."""
        assessment = ValueAssessment(
            primary_use_case="Test",
            recommendation=Recommendation.CONDITIONAL,
            risks=[],
            alternatives=[],
        )
        md = self.gen.to_markdown(
            assessment,
            dataset_id="test/ds",
            dataset_type="sft",
            reproduction_cost={"total": 1000, "human": 800},
        )
        self.assertIn("暂无已知替代方案", md)

    def test_to_markdown_not_recommended_icon(self):
        """Lines 561-562: NOT_RECOMMENDED red icon."""
        assessment = ValueAssessment(
            score=2.0,
            recommendation=Recommendation.NOT_RECOMMENDED,
            recommendation_reason="Low value",
            primary_use_case="Test",
        )
        md = self.gen.to_markdown(
            assessment,
            dataset_id="test/ds",
            dataset_type="sft",
            reproduction_cost={"total": 1000, "human": 800},
        )
        self.assertIn("不推荐", md)

    def test_to_markdown_recommended_icon(self):
        """Lines 555-556: RECOMMENDED icon."""
        assessment = ValueAssessment(
            score=7.0,
            recommendation=Recommendation.RECOMMENDED,
            recommendation_reason="Good value",
            primary_use_case="Test",
        )
        md = self.gen.to_markdown(
            assessment,
            dataset_id="test/ds",
            dataset_type="sft",
            reproduction_cost={"total": 1000, "human": 800},
        )
        self.assertIn("推荐", md)


# ==================== to_dict ====================


class TestToDict(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_to_dict_all_fields(self):
        assessment = ValueAssessment(
            score=7.5,
            recommendation=Recommendation.RECOMMENDED,
            recommendation_reason="Good",
            primary_use_case="Primary",
            secondary_use_cases=["Secondary"],
            expected_outcomes=["Outcome"],
            roi_ratio=2.5,
            roi_explanation="Great ROI",
            payback_scenarios=["S1"],
            risks=[{"level": "low"}],
            alternatives=["Alt1"],
            competitive_advantage="Strong",
        )
        d = self.gen.to_dict(assessment)
        self.assertEqual(d["score"], 7.5)
        self.assertEqual(d["recommendation"], "recommended")
        self.assertEqual(d["roi_ratio"], 2.5)
        self.assertEqual(len(d["risks"]), 1)


# ==================== spec_to_markdown (analyze-spec pipeline) ====================


class TestSpecToMarkdown(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_spec_to_markdown_medium_difficulty(self):
        """Lines 706-712: medium difficulty -> strongly recommended."""
        analysis = StubSpecAnalysis(estimated_difficulty="medium")
        md = self.gen.spec_to_markdown(
            analysis, target_size=1000, region="cn", cost_per_item=5.0
        )
        self.assertIn("强烈推荐", md)
        self.assertIn("7.5/10", md)
        self.assertIn("Test Project", md)

    def test_spec_to_markdown_hard_difficulty(self):
        """Lines 704-706: hard difficulty -> recommended."""
        analysis = StubSpecAnalysis(estimated_difficulty="hard")
        md = self.gen.spec_to_markdown(
            analysis, target_size=1000, region="cn", cost_per_item=5.0
        )
        self.assertIn("推荐", md)
        self.assertIn("6.5/10", md)

    def test_spec_to_markdown_expert_difficulty(self):
        """Lines 701-703: expert difficulty -> conditional."""
        analysis = StubSpecAnalysis(estimated_difficulty="expert")
        md = self.gen.spec_to_markdown(
            analysis, target_size=1000, region="cn", cost_per_item=5.0
        )
        self.assertIn("有条件推荐", md)
        self.assertIn("5.5/10", md)

    def test_spec_to_markdown_with_enhanced_context(self):
        """Lines 752-775: enhanced context sections."""
        ec = StubEnhancedContext(
            generated=True,
            dataset_purpose_summary="Enhanced purpose",
            tailored_use_cases=["Use 1", "Use 2"],
            tailored_roi_scenarios=["ROI 1"],
            competitive_positioning="Strong positioning",
            tailored_risks=[{"level": "中", "description": "Risk", "mitigation": "Fix"}],
        )
        analysis = StubSpecAnalysis()
        md = self.gen.spec_to_markdown(
            analysis, target_size=1000, region="cn", cost_per_item=5.0,
            enhanced_context=ec,
        )
        self.assertIn("Enhanced purpose", md)
        self.assertIn("Use 1", md)
        self.assertIn("ROI 1", md)
        self.assertIn("Strong positioning", md)

    def test_spec_to_markdown_with_ai_forbidden(self):
        """Lines 792-797: AI in forbidden items."""
        analysis = StubSpecAnalysis(
            forbidden_items=["AI生成内容", "模板抄袭"],
            estimated_difficulty="hard",
        )
        md = self.gen.spec_to_markdown(
            analysis, target_size=1000, region="cn", cost_per_item=5.0
        )
        self.assertIn("禁止使用AI", md)

    def test_spec_to_markdown_with_images(self):
        """Lines 803-804: has_images risk."""
        analysis = StubSpecAnalysis(
            has_images=True,
            image_count=50,
            estimated_difficulty="hard",
        )
        md = self.gen.spec_to_markdown(
            analysis, target_size=1000, region="cn", cost_per_item=5.0
        )
        self.assertIn("图片", md)

    def test_spec_to_markdown_similar_datasets(self):
        """Lines 811-817: similar datasets section."""
        analysis = StubSpecAnalysis(similar_datasets=["MMLU", "HellaSwag"])
        md = self.gen.spec_to_markdown(
            analysis, target_size=1000, region="cn", cost_per_item=5.0
        )
        self.assertIn("MMLU", md)
        self.assertIn("HellaSwag", md)


# ==================== spec_to_dict ====================


class TestSpecToDict(unittest.TestCase):
    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_spec_to_dict(self):
        """Lines 825-853: spec_to_dict."""
        analysis = StubSpecAnalysis(estimated_difficulty="hard")
        d = self.gen.spec_to_dict(analysis, target_size=1000, cost_per_item=5.0)
        self.assertEqual(d["project_name"], "Test Project")
        self.assertEqual(d["recommendation"], "推荐")
        self.assertEqual(d["score"], 6.5)
        self.assertEqual(d["total_cost"], 5000.0)

    def test_spec_to_dict_expert(self):
        analysis = StubSpecAnalysis(estimated_difficulty="expert")
        d = self.gen.spec_to_dict(analysis, target_size=500, cost_per_item=10.0)
        self.assertEqual(d["recommendation"], "有条件推荐")
        self.assertEqual(d["score"], 5.5)

    def test_spec_to_dict_medium(self):
        analysis = StubSpecAnalysis(estimated_difficulty="medium")
        d = self.gen.spec_to_dict(analysis, target_size=200, cost_per_item=2.0)
        self.assertEqual(d["recommendation"], "强烈推荐")
        self.assertEqual(d["score"], 7.5)


if __name__ == "__main__":
    unittest.main()
