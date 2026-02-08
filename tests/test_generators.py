"""Unit tests for generator classes.

Tests ExecutiveSummaryGenerator, AnnotationSpecGenerator, and MilestonePlanGenerator.
"""

import unittest
from dataclasses import dataclass, field

from datarecipe.generators.annotation_spec import (
    AnnotationSpec,
    AnnotationSpecGenerator,
    ExampleItem,
    ScoringCriterion,
)
from datarecipe.generators.executive_summary import (
    DATASET_TYPE_CONFIG,
    RISK_TEMPLATES,
    ExecutiveSummaryGenerator,
    Recommendation,
    ValueAssessment,
)
from datarecipe.generators.milestone_plan import (
    AcceptanceCriteria,
    Milestone,
    MilestonePlan,
    MilestonePlanGenerator,
    MilestoneStatus,
    RiskItem,
)

# ---------- Stub objects for optional parameters ----------


@dataclass
class StubComplexityMetrics:
    primary_domain: object = None
    difficulty_score: float = 2.0
    length_category: str = "medium"
    time_multiplier: float = 1.0


class StubDomain:
    """Mimics an Enum-like domain with .value attribute."""

    def __init__(self, value: str):
        self.value = value


@dataclass
class StubEnhancedContext:
    generated: bool = True
    tailored_use_cases: list[str] = field(default_factory=list)
    tailored_roi_scenarios: list[str] = field(default_factory=list)
    tailored_risks: list = field(default_factory=list)
    competitive_positioning: str = ""
    domain_specific_guidelines: str = ""
    quality_pitfalls: list[str] = field(default_factory=list)
    example_analysis: list = field(default_factory=list)
    phase_specific_risks: list = field(default_factory=list)
    team_recommendations: str = ""


@dataclass
class StubLLMAnalysis:
    purpose: str = ""
    production_steps: list[str] = field(default_factory=list)
    quality_criteria: list[str] = field(default_factory=list)


@dataclass
class StubRubricsResult:
    verb_distribution: dict = field(default_factory=dict)
    structured_patterns: list = field(default_factory=list)
    total_rubrics: int = 0
    unique_patterns: int = 0


# ==================== ExecutiveSummaryGenerator ====================


class TestValueAssessmentDataclass(unittest.TestCase):
    """Test ValueAssessment dataclass defaults."""

    def test_default_values(self):
        va = ValueAssessment()
        self.assertEqual(va.score, 5.0)
        self.assertEqual(va.recommendation, Recommendation.CONDITIONAL)
        self.assertEqual(va.recommendation_reason, "")
        self.assertEqual(va.primary_use_case, "")
        self.assertEqual(va.secondary_use_cases, [])
        self.assertEqual(va.risks, [])

    def test_recommendation_enum(self):
        self.assertEqual(Recommendation.HIGHLY_RECOMMENDED.value, "highly_recommended")
        self.assertEqual(Recommendation.NOT_RECOMMENDED.value, "not_recommended")


class TestExecutiveSummaryGenerate(unittest.TestCase):
    """Test ExecutiveSummaryGenerator.generate()."""

    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()
        self.base_kwargs = {
            "dataset_id": "test/dataset",
            "dataset_type": "preference",
            "sample_count": 5000,
            "reproduction_cost": {"total": 10000, "human": 7000, "machine": 3000},
            "human_percentage": 70.0,
        }

    def test_generate_returns_value_assessment(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertIsInstance(result, ValueAssessment)

    def test_generate_preference_type_uses_config(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertEqual(result.primary_use_case, DATASET_TYPE_CONFIG["preference"]["primary_use_case"])
        self.assertGreater(len(result.secondary_use_cases), 0)

    def test_generate_unknown_type_falls_back(self):
        kwargs = {**self.base_kwargs, "dataset_type": "unknown_type"}
        result = self.gen.generate(**kwargs)
        self.assertEqual(result.primary_use_case, "通用数据集，用途待定")

    def test_enhanced_context_overrides_use_cases(self):
        ec = StubEnhancedContext(
            generated=True,
            tailored_use_cases=["Custom use 1", "Custom use 2", "Custom use 3"],
        )
        result = self.gen.generate(**self.base_kwargs, enhanced_context=ec)
        self.assertEqual(result.primary_use_case, "Custom use 1")
        self.assertEqual(result.secondary_use_cases, ["Custom use 2", "Custom use 3"])

    def test_enhanced_context_overrides_risks(self):
        ec = StubEnhancedContext(
            generated=True,
            tailored_risks=[{"level": "高", "description": "Custom risk", "mitigation": "Fix it"}],
        )
        result = self.gen.generate(**self.base_kwargs, enhanced_context=ec)
        self.assertEqual(len(result.risks), 1)
        self.assertEqual(result.risks[0]["description"], "Custom risk")

    def test_enhanced_context_not_generated_ignored(self):
        ec = StubEnhancedContext(
            generated=False,
            tailored_use_cases=["Should not appear"],
        )
        result = self.gen.generate(**self.base_kwargs, enhanced_context=ec)
        self.assertNotEqual(result.primary_use_case, "Should not appear")

    def test_score_in_valid_range(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertGreaterEqual(result.score, 1.0)
        self.assertLessEqual(result.score, 10.0)

    def test_risks_always_include_data_freshness(self):
        result = self.gen.generate(**self.base_kwargs)
        freshness_risks = [r for r in result.risks if r.get("description") == RISK_TEMPLATES["data_freshness"]["description"]]
        self.assertEqual(len(freshness_risks), 1)

    def test_high_cost_triggers_risk(self):
        kwargs = {**self.base_kwargs, "reproduction_cost": {"total": 60000, "human": 50000, "machine": 10000}}
        result = self.gen.generate(**kwargs)
        high_cost_risks = [r for r in result.risks if r == RISK_TEMPLATES["high_cost"]]
        self.assertEqual(len(high_cost_risks), 1)

    def test_high_human_percentage_triggers_quality_risk(self):
        kwargs = {**self.base_kwargs, "human_percentage": 90.0}
        result = self.gen.generate(**kwargs)
        quality_risks = [r for r in result.risks if r == RISK_TEMPLATES["quality_variance"]]
        self.assertEqual(len(quality_risks), 1)

    def test_expert_domain_triggers_risk(self):
        metrics = StubComplexityMetrics(primary_domain=StubDomain("medical"))
        result = self.gen.generate(**self.base_kwargs, complexity_metrics=metrics)
        expert_risks = [r for r in result.risks if r == RISK_TEMPLATES["expert_required"]]
        self.assertEqual(len(expert_risks), 1)


class TestExecutiveSummaryScoring(unittest.TestCase):
    """Test value score calculation logic."""

    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_large_dataset_gets_bonus(self):
        score_large = self.gen._calculate_value_score(
            "preference", 15000, {"total": 10000}, 50.0, None, DATASET_TYPE_CONFIG["preference"]
        )
        score_small = self.gen._calculate_value_score(
            "preference", 30, {"total": 10000}, 50.0, None, DATASET_TYPE_CONFIG["preference"]
        )
        self.assertGreater(score_large, score_small)

    def test_cost_efficient_gets_bonus(self):
        score_cheap = self.gen._calculate_value_score(
            "sft", 10000, {"total": 5000}, 50.0, None, DATASET_TYPE_CONFIG["sft"]
        )
        score_expensive = self.gen._calculate_value_score(
            "sft", 10000, {"total": 1000000}, 50.0, None, DATASET_TYPE_CONFIG["sft"]
        )
        self.assertGreater(score_cheap, score_expensive)

    def test_score_clamped_to_1_10(self):
        # Edge case: very small dataset, very expensive, bad type
        score = self.gen._calculate_value_score(
            "unknown", 10, {"total": 1000000}, 50.0, None, {}
        )
        self.assertGreaterEqual(score, 1.0)
        self.assertLessEqual(score, 10.0)


class TestExecutiveSummaryRecommendation(unittest.TestCase):
    """Test recommendation logic."""

    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_high_score_highly_recommended(self):
        rec, _ = self.gen._get_recommendation(8.5, {"total": 1000}, "preference")
        self.assertEqual(rec, Recommendation.HIGHLY_RECOMMENDED)

    def test_medium_score_recommended(self):
        rec, _ = self.gen._get_recommendation(6.5, {"total": 1000}, "preference")
        self.assertEqual(rec, Recommendation.RECOMMENDED)

    def test_low_score_conditional(self):
        rec, _ = self.gen._get_recommendation(4.5, {"total": 1000}, "preference")
        self.assertEqual(rec, Recommendation.CONDITIONAL)

    def test_very_low_score_not_recommended(self):
        rec, _ = self.gen._get_recommendation(2.0, {"total": 1000}, "preference")
        self.assertEqual(rec, Recommendation.NOT_RECOMMENDED)

    def test_conditional_with_high_cost_mentions_budget(self):
        rec, reason = self.gen._get_recommendation(4.5, {"total": 50000}, "preference")
        self.assertEqual(rec, Recommendation.CONDITIONAL)
        self.assertIn("成本较高", reason)


class TestExecutiveSummaryROI(unittest.TestCase):
    """Test ROI calculation."""

    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()

    def test_zero_cost_returns_zero(self):
        ratio, explanation = self.gen._calculate_roi("preference", {"total": 0}, 1000, {})
        self.assertEqual(ratio, 0)

    def test_preference_type_high_value(self):
        ratio, _ = self.gen._calculate_roi(
            "preference", {"total": 1000}, 10000, DATASET_TYPE_CONFIG["preference"]
        )
        self.assertGreater(ratio, 1.0)

    def test_swe_bench_highest_per_sample_value(self):
        ratio_swe, _ = self.gen._calculate_roi(
            "swe_bench", {"total": 1000}, 1000, DATASET_TYPE_CONFIG.get("swe_bench", {})
        )
        ratio_unknown, _ = self.gen._calculate_roi(
            "unknown", {"total": 1000}, 1000, {}
        )
        self.assertGreater(ratio_swe, ratio_unknown)


class TestExecutiveSummaryMarkdown(unittest.TestCase):
    """Test to_markdown() output."""

    def setUp(self):
        self.gen = ExecutiveSummaryGenerator()
        self.assessment = ValueAssessment(
            score=7.5,
            recommendation=Recommendation.RECOMMENDED,
            recommendation_reason="Good dataset",
            primary_use_case="Training reward models",
            secondary_use_cases=["RLHF", "Alignment"],
            expected_outcomes=["Better model"],
            roi_ratio=2.5,
            roi_explanation="Good ROI",
            payback_scenarios=["Scenario A"],
            risks=[{"level": "中", "description": "Some risk", "mitigation": "Fix it"}],
            alternatives=["alt/dataset"],
            competitive_advantage="Strong org",
        )

    def test_markdown_contains_header(self):
        md = self.gen.to_markdown(
            self.assessment, "test/dataset", "preference", {"total": 1000, "human": 500}
        )
        self.assertIn("# test/dataset 执行摘要", md)

    def test_markdown_contains_recommendation(self):
        md = self.gen.to_markdown(
            self.assessment, "test/dataset", "preference", {"total": 1000, "human": 500}
        )
        self.assertIn("推荐", md)
        self.assertIn("7.5/10", md)

    def test_markdown_contains_roi(self):
        md = self.gen.to_markdown(
            self.assessment, "test/dataset", "preference", {"total": 1000, "human": 500}
        )
        self.assertIn("2.5x", md)

    def test_markdown_contains_risks_table(self):
        md = self.gen.to_markdown(
            self.assessment, "test/dataset", "preference", {"total": 1000, "human": 500}
        )
        self.assertIn("Some risk", md)
        self.assertIn("Fix it", md)

    def test_markdown_contains_alternatives(self):
        md = self.gen.to_markdown(
            self.assessment, "test/dataset", "preference", {"total": 1000, "human": 500}
        )
        self.assertIn("alt/dataset", md)

    def test_markdown_no_alternatives(self):
        self.assessment.alternatives = []
        md = self.gen.to_markdown(
            self.assessment, "test/dataset", "preference", {"total": 1000, "human": 500}
        )
        self.assertIn("暂无已知替代方案", md)


class TestExecutiveSummaryToDict(unittest.TestCase):
    """Test to_dict() output."""

    def test_dict_contains_all_keys(self):
        gen = ExecutiveSummaryGenerator()
        assessment = ValueAssessment(score=7.0, recommendation=Recommendation.RECOMMENDED)
        d = gen.to_dict(assessment)
        self.assertIn("score", d)
        self.assertIn("recommendation", d)
        self.assertEqual(d["recommendation"], "recommended")
        self.assertEqual(d["score"], 7.0)


# ==================== AnnotationSpecGenerator ====================


class TestAnnotationSpecGenerate(unittest.TestCase):
    """Test AnnotationSpecGenerator.generate()."""

    def setUp(self):
        self.gen = AnnotationSpecGenerator()
        self.base_kwargs = {
            "dataset_id": "test/eval-dataset",
            "dataset_type": "evaluation",
            "schema_info": {
                "question": {"type": "str"},
                "answer": {"type": "str"},
                "score": {"type": "int"},
            },
            "sample_items": [
                {"question": "What is 2+2?", "answer": "4", "score": 1},
                {"question": "Explain gravity", "answer": "Force of attraction between masses", "score": 1},
            ],
        }

    def test_generate_returns_annotation_spec(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertIsInstance(result, AnnotationSpec)

    def test_generate_sets_basic_fields(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertEqual(result.dataset_id, "test/eval-dataset")
        self.assertEqual(result.dataset_type, "evaluation")
        self.assertEqual(result.source_samples, 2)

    def test_generate_sets_task_info_from_profile(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertTrue(len(result.task_name) > 0)
        self.assertTrue(len(result.task_description) > 0)
        self.assertTrue(len(result.cognitive_requirements) > 0)

    def test_generate_data_requirements_includes_field_count(self):
        result = self.gen.generate(**self.base_kwargs)
        has_field_count = any("3 个字段" in r for r in result.data_requirements)
        self.assertTrue(has_field_count)

    def test_generate_format_requirements(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertGreater(len(result.format_requirements), 0)
        has_question = any("question" in r for r in result.format_requirements)
        self.assertTrue(has_question)

    def test_generate_quality_constraints(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertGreater(len(result.quality_constraints), 0)
        has_ai_constraint = any("AI" in c for c in result.quality_constraints)
        self.assertTrue(has_ai_constraint)

    def test_generate_default_scoring_dimensions(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertGreater(len(result.scoring_dimensions), 0)

    def test_generate_partial_credit_rules(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertGreater(len(result.partial_credit_rules), 0)

    def test_generate_with_llm_analysis_overrides(self):
        llm = StubLLMAnalysis(
            purpose="Custom purpose description",
            production_steps=["Step1", "Step2", "Step3"],
        )
        result = self.gen.generate(**self.base_kwargs, llm_analysis=llm)
        self.assertEqual(result.task_description, "Custom purpose description")
        self.assertIn("Step1", result.reasoning_chain)

    def test_generate_with_rubrics_result(self):
        rubrics = StubRubricsResult(
            verb_distribution={"分析": 10, "评估": 8, "判断": 5},
        )
        result = self.gen.generate(**self.base_kwargs, rubrics_result=rubrics)
        has_verb = any("分析" in c for c in result.quality_constraints)
        self.assertTrue(has_verb)

    def test_generate_with_complexity_metrics(self):
        metrics = StubComplexityMetrics(
            primary_domain=StubDomain("medical"),
            length_category="long",
        )
        result = self.gen.generate(**self.base_kwargs, complexity_metrics=metrics)
        has_domain = any("medical" in r for r in result.data_requirements)
        self.assertTrue(has_domain)

    def test_generate_empty_samples(self):
        kwargs = {**self.base_kwargs, "sample_items": []}
        result = self.gen.generate(**kwargs)
        self.assertIsInstance(result, AnnotationSpec)
        self.assertEqual(result.examples, [])


class TestAnnotationSpecExampleSelection(unittest.TestCase):
    """Test example quality scoring and selection."""

    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_preference_duplicate_scores_zero(self):
        item = {"chosen": "same text", "rejected": "same text"}
        score = self.gen._score_example_quality(item, "preference")
        self.assertEqual(score, 0)

    def test_preference_good_example_scores_higher(self):
        good = {
            "chosen": "This is a helpful and detailed response about machine learning." * 3,
            "rejected": "IDK.",
        }
        bad = {"chosen": "x", "rejected": "y"}
        score_good = self.gen._score_example_quality(good, "preference")
        score_bad = self.gen._score_example_quality(bad, "preference")
        self.assertGreater(score_good, score_bad)

    def test_select_best_examples_returns_top_n(self):
        items = [
            {"question": f"Q{i}", "answer": f"A{i}" * 20} for i in range(10)
        ]
        selected = self.gen._select_best_examples(items, "evaluation", count=3)
        self.assertLessEqual(len(selected), 3)

    def test_select_best_examples_empty_input(self):
        selected = self.gen._select_best_examples([], "evaluation")
        self.assertEqual(selected, [])


class TestAnnotationSpecMarkdown(unittest.TestCase):
    """Test AnnotationSpec to_markdown() output."""

    def setUp(self):
        self.gen = AnnotationSpecGenerator()
        self.spec = AnnotationSpec(
            dataset_id="test/dataset",
            dataset_type="evaluation",
            task_name="评测基准数据",
            task_description="Test description",
            cognitive_requirements=["Req 1", "Req 2"],
            reasoning_chain="Step1 → Step2 → Step3",
            data_requirements=["Need field A"],
            quality_constraints=["Must be accurate", "禁止使用 AI"],
            difficulty_calibration="Use GPT-4 for calibration",
            format_requirements=["`question`: 字符串类型"],
            examples=[
                ExampleItem(
                    id=1,
                    question_text="What is 2+2?",
                    answer="4",
                    scoring_criteria=[ScoringCriterion(score="1分", description="Correct")],
                ),
            ],
            scoring_dimensions=["准确性", "完整性"],
            scoring_rubrics=[{"dimension": "准确性", "description": "Answer is correct"}],
            partial_credit_rules=["Partial answers get partial credit"],
            generated_at="2025-01-01 00:00",
            source_samples=10,
        )

    def test_markdown_header(self):
        md = self.gen.to_markdown(self.spec)
        self.assertIn("# test/dataset 标注规范", md)

    def test_markdown_task_section(self):
        md = self.gen.to_markdown(self.spec)
        self.assertIn("题目类型描述", md)
        self.assertIn("评测基准数据", md)

    def test_markdown_quality_constraints_bold_important(self):
        md = self.gen.to_markdown(self.spec)
        self.assertIn("**禁止使用 AI**", md)

    def test_markdown_examples_table(self):
        md = self.gen.to_markdown(self.spec)
        self.assertIn("What is 2+2?", md)

    def test_markdown_scoring_section(self):
        md = self.gen.to_markdown(self.spec)
        self.assertIn("评分维度", md)
        self.assertIn("准确性", md)

    def test_markdown_with_enhanced_context(self):
        ec = StubEnhancedContext(
            generated=True,
            domain_specific_guidelines="Important domain-specific note",
            quality_pitfalls=["Common mistake 1"],
            example_analysis=[
                {"sample_index": 0, "strengths": "Good", "weaknesses": "Bad", "annotation_tips": "Tip"},
            ],
        )
        self.spec._enhanced_context = ec
        md = self.gen.to_markdown(self.spec)
        self.assertIn("领域标注指导", md)
        self.assertIn("Important domain-specific note", md)
        self.assertIn("Common mistake 1", md)


class TestAnnotationSpecToDict(unittest.TestCase):
    """Test to_dict() serialization."""

    def test_dict_structure(self):
        gen = AnnotationSpecGenerator()
        spec = AnnotationSpec(
            dataset_id="test/dataset",
            dataset_type="sft",
            task_name="SFT",
            source_samples=5,
        )
        d = gen.to_dict(spec)
        self.assertEqual(d["dataset_id"], "test/dataset")
        self.assertIn("task", d)
        self.assertIn("requirements", d)
        self.assertIn("scoring", d)
        self.assertIn("metadata", d)


# ==================== MilestonePlanGenerator ====================


class TestMilestonePlanGenerate(unittest.TestCase):
    """Test MilestonePlanGenerator.generate()."""

    def setUp(self):
        self.gen = MilestonePlanGenerator()
        self.base_kwargs = {
            "dataset_id": "test/pref-dataset",
            "dataset_type": "preference",
            "target_size": 5000,
            "reproduction_cost": {"total": 20000, "human": 15000, "machine": 5000},
            "human_percentage": 75.0,
        }

    def test_generate_returns_milestone_plan(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertIsInstance(result, MilestonePlan)

    def test_generate_sets_basic_fields(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertEqual(result.dataset_id, "test/pref-dataset")
        self.assertEqual(result.dataset_type, "preference")
        self.assertEqual(result.target_size, 5000)

    def test_generate_preference_has_5_milestones(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertEqual(len(result.milestones), 5)

    def test_generate_evaluation_has_4_milestones(self):
        kwargs = {**self.base_kwargs, "dataset_type": "evaluation"}
        result = self.gen.generate(**kwargs)
        self.assertEqual(len(result.milestones), 4)

    def test_generate_sft_has_4_milestones(self):
        kwargs = {**self.base_kwargs, "dataset_type": "sft"}
        result = self.gen.generate(**kwargs)
        self.assertEqual(len(result.milestones), 4)

    def test_generate_unknown_type_falls_back_to_preference(self):
        kwargs = {**self.base_kwargs, "dataset_type": "unknown_type"}
        result = self.gen.generate(**kwargs)
        self.assertEqual(len(result.milestones), 5)  # Same as preference

    def test_milestones_have_correct_dependencies(self):
        result = self.gen.generate(**self.base_kwargs)
        # M1 has no dependencies
        self.assertEqual(result.milestones[0].dependencies, [])
        # M2 depends on M1
        self.assertIn("M1", result.milestones[1].dependencies)

    def test_effort_percentages_sum_to_100(self):
        result = self.gen.generate(**self.base_kwargs)
        total = sum(m.effort_percentage for m in result.milestones)
        self.assertEqual(total, 100)

    def test_milestones_customized_with_target_size(self):
        result = self.gen.generate(**self.base_kwargs)
        # Check that deliverables contain actual numbers, not template placeholders
        all_deliverables = " ".join(
            d for m in result.milestones for d in m.deliverables
        )
        self.assertNotIn("目标的 5%", all_deliverables)
        self.assertIn("250 条", all_deliverables)  # 5000 * 0.05 = 250

    def test_all_milestones_start_as_not_started(self):
        result = self.gen.generate(**self.base_kwargs)
        for m in result.milestones:
            self.assertEqual(m.status, MilestoneStatus.NOT_STARTED)

    def test_risks_always_present(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertGreater(len(result.risks), 0)

    def test_risks_include_quality_and_schedule(self):
        result = self.gen.generate(**self.base_kwargs)
        categories = [r.category for r in result.risks]
        self.assertIn("quality", categories)
        self.assertIn("schedule", categories)

    def test_complex_domain_adds_resource_risk(self):
        metrics = StubComplexityMetrics(primary_domain=StubDomain("medical"))
        result = self.gen.generate(**self.base_kwargs, complexity_metrics=metrics)
        categories = [r.category for r in result.risks]
        self.assertIn("resource", categories)

    def test_high_difficulty_adds_requirement_risk(self):
        metrics = StubComplexityMetrics(difficulty_score=4.0)
        result = self.gen.generate(**self.base_kwargs, complexity_metrics=metrics)
        categories = [r.category for r in result.risks]
        self.assertIn("technical", categories)

    def test_enhanced_context_overrides_risks(self):
        ec = StubEnhancedContext(
            generated=True,
            phase_specific_risks=[{"phase": "Phase1", "risk": "Custom risk", "mitigation": "Handle it"}],
        )
        result = self.gen.generate(**self.base_kwargs, enhanced_context=ec)
        # risks should be the enhanced ones
        self.assertEqual(len(result.risks), 1)
        self.assertEqual(result.risks[0]["phase"], "Phase1")

    def test_acceptance_criteria_for_preference(self):
        result = self.gen.generate(**self.base_kwargs)
        self.assertGreater(len(result.acceptance_criteria), 0)
        categories = [ac.category for ac in result.acceptance_criteria]
        self.assertIn("一致性", categories)


class TestMilestonePlanTeam(unittest.TestCase):
    """Test team composition calculation."""

    def setUp(self):
        self.gen = MilestonePlanGenerator()

    def test_high_human_large_dataset_many_annotators(self):
        team = self.gen._calculate_team("preference", 15000, 80.0)
        self.assertEqual(team["标注员"], 8)

    def test_low_human_few_annotators(self):
        team = self.gen._calculate_team("preference", 15000, 30.0)
        self.assertEqual(team["标注员"], 2)

    def test_evaluation_has_special_roles(self):
        team = self.gen._calculate_team("evaluation", 1000, 80.0)
        self.assertIn("评测设计师", team)
        self.assertIn("评测工程师", team)

    def test_sft_has_special_roles(self):
        team = self.gen._calculate_team("sft", 1000, 80.0)
        self.assertIn("ML 工程师", team)
        self.assertIn("数据工程师", team)


class TestMilestonePlanDuration(unittest.TestCase):
    """Test duration estimation."""

    def setUp(self):
        self.gen = MilestonePlanGenerator()

    def test_minimum_10_days(self):
        days = self.gen._estimate_duration(10, 10.0, None)
        self.assertGreaterEqual(days, 10)

    def test_larger_dataset_longer_duration(self):
        days_small = self.gen._estimate_duration(100, 50.0, None)
        days_large = self.gen._estimate_duration(10000, 50.0, None)
        self.assertGreater(days_large, days_small)

    def test_complexity_multiplier(self):
        metrics = StubComplexityMetrics(time_multiplier=2.0)
        days_normal = self.gen._estimate_duration(5000, 50.0, None)
        days_complex = self.gen._estimate_duration(5000, 50.0, metrics)
        self.assertGreater(days_complex, days_normal)


class TestMilestonePlanMarkdown(unittest.TestCase):
    """Test to_markdown() output."""

    def setUp(self):
        self.gen = MilestonePlanGenerator()
        self.plan = MilestonePlan(
            dataset_id="test/dataset",
            dataset_type="preference",
            target_size=5000,
            milestones=[
                Milestone(
                    id="M1",
                    name="Phase 1",
                    description="First phase",
                    deliverables=["Doc v1"],
                    acceptance_criteria=["Pass review"],
                    team=["PM", "Expert"],
                    effort_percentage=20,
                ),
                Milestone(
                    id="M2",
                    name="Phase 2",
                    description="Second phase",
                    deliverables=["Data"],
                    acceptance_criteria=["Quality check"],
                    dependencies=["M1"],
                    team=["Annotator"],
                    effort_percentage=80,
                ),
            ],
            risks=[
                RiskItem(
                    id="R1",
                    category="quality",
                    description="Quality issue",
                    probability="medium",
                    impact="high",
                    mitigation="Train more",
                    contingency="Hire more",
                ),
            ],
            acceptance_criteria=[
                AcceptanceCriteria(
                    category="Test",
                    criterion="Test criterion",
                    metric="Kappa",
                    threshold=">= 0.7",
                    verification_method="Manual check",
                ),
            ],
            team_composition={"PM": 1, "Expert": 2},
            estimated_days=30,
        )

    def test_markdown_header(self):
        md = self.gen.to_markdown(self.plan)
        self.assertIn("# test/dataset 里程碑计划", md)
        self.assertIn("5,000 条", md)
        self.assertIn("30 工作日", md)

    def test_markdown_milestones(self):
        md = self.gen.to_markdown(self.plan)
        self.assertIn("M1: Phase 1", md)
        self.assertIn("M2: Phase 2", md)
        self.assertIn("依赖", md)

    def test_markdown_risks(self):
        md = self.gen.to_markdown(self.plan)
        self.assertIn("Quality issue", md)
        self.assertIn("Train more", md)

    def test_markdown_acceptance_criteria_table(self):
        md = self.gen.to_markdown(self.plan)
        self.assertIn("Test criterion", md)
        self.assertIn("Kappa", md)

    def test_markdown_team_table(self):
        md = self.gen.to_markdown(self.plan)
        self.assertIn("PM", md)
        self.assertIn("Expert", md)

    def test_markdown_checklist(self):
        md = self.gen.to_markdown(self.plan)
        self.assertIn("启动检查清单", md)
        self.assertIn("- [ ]", md)

    def test_markdown_dict_risks_format(self):
        """Test that LLM-enhanced dict risks render correctly."""
        self.plan.risks = [
            {"phase": "试点阶段", "risk": "Custom risk", "mitigation": "Handle it"},
        ]
        md = self.gen.to_markdown(self.plan)
        self.assertIn("试点阶段", md)
        self.assertIn("Custom risk", md)


class TestMilestonePlanToDict(unittest.TestCase):
    """Test to_dict() serialization."""

    def test_dict_structure(self):
        gen = MilestonePlanGenerator()
        plan = gen.generate(
            dataset_id="test/dataset",
            dataset_type="preference",
            target_size=1000,
            reproduction_cost={"total": 5000},
            human_percentage=50.0,
        )
        d = gen.to_dict(plan)
        self.assertIn("dataset_id", d)
        self.assertIn("milestones", d)
        self.assertIn("risks", d)
        self.assertIn("acceptance_criteria", d)
        self.assertIsInstance(d["milestones"], list)
        self.assertGreater(len(d["milestones"]), 0)


if __name__ == "__main__":
    unittest.main()
