"""Unit tests for EnhancedGuideGenerator.

Tests EnhancedProductionGuide, WorkloadEstimate dataclasses,
and EnhancedGuideGenerator.generate(), to_markdown(), to_dict() methods.
"""

import unittest
from dataclasses import dataclass, field
from enum import Enum

from datarecipe.generators.enhanced_guide import (
    EnhancedGuideGenerator,
    EnhancedProductionGuide,
    WorkloadEstimate,
)


# ---------- Stub objects for dependencies ----------


class StubStrategyType(Enum):
    """Mimics ContextStrategyType enum."""

    SYNTHETIC = "synthetic"
    MODIFIED = "modified"
    NICHE = "niche"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class StubContextStrategy:
    """Mimics ContextStrategy dataclass."""

    primary_strategy: StubStrategyType = StubStrategyType.SYNTHETIC
    confidence: float = 0.85
    recommendations: list[str] = field(
        default_factory=lambda: ["Use fictional scenarios", "Verify novelty"]
    )


@dataclass
class StubRubricsAnalysisResult:
    """Mimics RubricsAnalysisResult dataclass."""

    patterns: list = field(default_factory=list)
    verb_distribution: dict = field(default_factory=dict)
    category_distribution: dict = field(default_factory=dict)
    top_templates: list[str] = field(default_factory=list)
    structured_templates: list = field(default_factory=list)
    total_rubrics: int = 0
    unique_patterns: int = 0
    avg_rubrics_per_task: float = 0.0


@dataclass
class StubPromptLibrary:
    """Mimics PromptLibrary dataclass."""

    templates: list = field(default_factory=list)
    total_extracted: int = 0
    unique_count: int = 0
    deduplication_ratio: float = 0.0
    category_counts: dict = field(default_factory=dict)
    domain_counts: dict = field(default_factory=dict)


class StubAllocationDecision(Enum):
    """Mimics AllocationDecision enum."""

    HUMAN_ONLY = "human_only"
    MACHINE_ONLY = "machine_only"
    HUMAN_PRIMARY = "human_primary"
    MACHINE_PRIMARY = "machine_primary"


@dataclass
class StubTaskAllocation:
    """Mimics TaskAllocation dataclass."""

    task_id: str = "T1"
    task_name: str = "Context Writing"
    description: str = "Write original context content"
    decision: StubAllocationDecision = StubAllocationDecision.HUMAN_ONLY
    human_percentage: float = 100.0
    machine_percentage: float = 0.0
    human_hours: float = 40.0
    human_cost: float = 1000.0
    machine_cost: float = 0.0
    machine_method: str = ""


@dataclass
class StubHumanMachineAllocation:
    """Mimics HumanMachineAllocation dataclass."""

    tasks: list = field(default_factory=list)
    total_human_hours: float = 0.0
    total_human_cost: float = 0.0
    total_machine_cost: float = 0.0
    total_cost: float = 0.0
    human_only_tasks: list = field(default_factory=list)
    machine_only_tasks: list = field(default_factory=list)
    hybrid_tasks: list = field(default_factory=list)
    human_work_percentage: float = 70.0
    machine_work_percentage: float = 30.0
    estimated_savings_vs_all_human: float = 5000.0
    timeline_reduction_percentage: float = 20.0


# ==================== WorkloadEstimate Dataclass ====================


class TestWorkloadEstimateDataclass(unittest.TestCase):
    """Test WorkloadEstimate dataclass defaults and fields."""

    def test_default_values(self):
        we = WorkloadEstimate(
            phase_name="Test Phase",
            person_days=10.0,
            team_size=3,
            duration_weeks=2.0,
            cost=5000.0,
        )
        self.assertEqual(we.phase_name, "Test Phase")
        self.assertEqual(we.person_days, 10.0)
        self.assertEqual(we.team_size, 3)
        self.assertEqual(we.duration_weeks, 2.0)
        self.assertEqual(we.cost, 5000.0)
        self.assertFalse(we.critical_path)
        self.assertEqual(we.dependencies, [])

    def test_critical_path_flag(self):
        we = WorkloadEstimate(
            phase_name="Critical",
            person_days=50.0,
            team_size=4,
            duration_weeks=6.0,
            cost=10000.0,
            critical_path=True,
        )
        self.assertTrue(we.critical_path)

    def test_dependencies_list(self):
        we = WorkloadEstimate(
            phase_name="Phase B",
            person_days=20.0,
            team_size=2,
            duration_weeks=3.0,
            cost=3000.0,
            dependencies=["Phase A", "Phase C"],
        )
        self.assertEqual(we.dependencies, ["Phase A", "Phase C"])


# ==================== EnhancedProductionGuide Dataclass ====================


class TestEnhancedProductionGuideDataclass(unittest.TestCase):
    """Test EnhancedProductionGuide dataclass defaults and summary()."""

    def test_default_values(self):
        guide = EnhancedProductionGuide(
            dataset_name="test-dataset",
            target_size=5000,
        )
        self.assertEqual(guide.dataset_name, "test-dataset")
        self.assertEqual(guide.target_size, 5000)
        self.assertIsNone(guide.rubrics_analysis)
        self.assertIsNone(guide.prompt_library)
        self.assertIsNone(guide.context_strategy)
        self.assertIsNone(guide.allocation)
        self.assertEqual(guide.workload_estimates, [])
        self.assertEqual(guide.total_person_days, 0.0)
        self.assertEqual(guide.total_weeks, 0.0)
        self.assertEqual(guide.total_cost, 0.0)
        self.assertEqual(guide.recommended_team, {})
        self.assertEqual(guide.quality_checkpoints, [])

    def test_generation_date_auto_set(self):
        guide = EnhancedProductionGuide(
            dataset_name="test",
            target_size=100,
        )
        # Should be a date string like "2025-01-15"
        self.assertRegex(guide.generation_date, r"\d{4}-\d{2}-\d{2}")

    def test_summary_output(self):
        guide = EnhancedProductionGuide(
            dataset_name="my-dataset",
            target_size=10000,
            total_person_days=150,
            total_weeks=8.5,
            total_cost=45000,
        )
        summary = guide.summary()
        self.assertIn("my-dataset", summary)
        self.assertIn("10,000", summary)
        self.assertIn("150 person-days", summary)
        self.assertIn("8.5 weeks", summary)
        self.assertIn("$45,000", summary)

    def test_summary_formatting(self):
        guide = EnhancedProductionGuide(
            dataset_name="ds",
            target_size=500,
            total_person_days=25.7,
            total_weeks=3.2,
            total_cost=12500.50,
        )
        summary = guide.summary()
        self.assertIn("26 person-days", summary)  # .0f rounds
        self.assertIn("3.2 weeks", summary)
        self.assertIn("$12,500", summary)  # .0f with banker's rounding


# ==================== EnhancedGuideGenerator.generate() ====================


class TestEnhancedGuideGenerate(unittest.TestCase):
    """Test EnhancedGuideGenerator.generate() method."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_generate_returns_enhanced_production_guide(self):
        guide = self.gen.generate(dataset_name="test-ds", target_size=1000)
        self.assertIsInstance(guide, EnhancedProductionGuide)

    def test_generate_sets_basic_fields(self):
        guide = self.gen.generate(dataset_name="my-dataset", target_size=5000)
        self.assertEqual(guide.dataset_name, "my-dataset")
        self.assertEqual(guide.target_size, 5000)

    def test_generate_creates_workload_estimates(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        self.assertGreater(len(guide.workload_estimates), 0)
        for we in guide.workload_estimates:
            self.assertIsInstance(we, WorkloadEstimate)

    def test_generate_calculates_totals(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        # total_person_days should be sum of all phase person_days
        expected_total = sum(w.person_days for w in guide.workload_estimates)
        self.assertAlmostEqual(guide.total_person_days, expected_total)

    def test_generate_total_weeks_is_max(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        expected_max = max(w.duration_weeks for w in guide.workload_estimates)
        self.assertAlmostEqual(guide.total_weeks, expected_max)

    def test_generate_total_cost_is_sum(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        expected_cost = sum(w.cost for w in guide.workload_estimates)
        self.assertAlmostEqual(guide.total_cost, expected_cost)

    def test_generate_has_recommended_team(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        self.assertIsInstance(guide.recommended_team, dict)
        self.assertGreater(len(guide.recommended_team), 0)

    def test_generate_has_quality_checkpoints(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        self.assertIsInstance(guide.quality_checkpoints, list)
        self.assertGreater(len(guide.quality_checkpoints), 0)

    def test_generate_with_allocation(self):
        allocation = StubHumanMachineAllocation(
            machine_work_percentage=30.0,
            human_work_percentage=70.0,
        )
        guide = self.gen.generate(
            dataset_name="test", target_size=1000, allocation=allocation
        )
        self.assertIs(guide.allocation, allocation)

    def test_generate_with_rubrics_analysis(self):
        rubrics = StubRubricsAnalysisResult(
            total_rubrics=100,
            unique_patterns=25,
            top_templates=["The response should include relevant details"],
        )
        guide = self.gen.generate(
            dataset_name="test", target_size=1000, rubrics_analysis=rubrics
        )
        self.assertIs(guide.rubrics_analysis, rubrics)

    def test_generate_with_prompt_library(self):
        prompts = StubPromptLibrary(
            unique_count=15,
            category_counts={"system": 5, "task": 10},
        )
        guide = self.gen.generate(
            dataset_name="test", target_size=1000, prompt_library=prompts
        )
        self.assertIs(guide.prompt_library, prompts)

    def test_generate_with_context_strategy(self):
        strategy = StubContextStrategy(
            primary_strategy=StubStrategyType.SYNTHETIC, confidence=0.9
        )
        guide = self.gen.generate(
            dataset_name="test", target_size=1000, context_strategy=strategy
        )
        self.assertIs(guide.context_strategy, strategy)

    def test_generate_with_all_optional_params(self):
        rubrics = StubRubricsAnalysisResult(
            total_rubrics=50, unique_patterns=10, top_templates=["template1"]
        )
        prompts = StubPromptLibrary(unique_count=5, category_counts={"system": 5})
        strategy = StubContextStrategy()
        allocation = StubHumanMachineAllocation()
        guide = self.gen.generate(
            dataset_name="full-test",
            target_size=2000,
            rubrics_analysis=rubrics,
            prompt_library=prompts,
            context_strategy=strategy,
            allocation=allocation,
            region="us",
        )
        self.assertEqual(guide.dataset_name, "full-test")
        self.assertIsNotNone(guide.rubrics_analysis)
        self.assertIsNotNone(guide.prompt_library)
        self.assertIsNotNone(guide.context_strategy)
        self.assertIsNotNone(guide.allocation)


# ==================== _estimate_workload() ====================


class TestEstimateWorkload(unittest.TestCase):
    """Test _estimate_workload() internal method."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_returns_six_phases(self):
        estimates = self.gen._estimate_workload(1000, None, "china")
        self.assertEqual(len(estimates), 6)

    def test_phase_names(self):
        estimates = self.gen._estimate_workload(1000, None, "china")
        names = [e.phase_name for e in estimates]
        self.assertIn("Planning & Setup", names)
        self.assertIn("Context Creation", names)
        self.assertIn("Task Design", names)
        self.assertIn("Rubrics Writing", names)
        self.assertIn("Quality Review", names)
        self.assertIn("Final Validation", names)

    def test_critical_path_phases(self):
        estimates = self.gen._estimate_workload(1000, None, "china")
        critical = {e.phase_name for e in estimates if e.critical_path}
        self.assertIn("Context Creation", critical)
        self.assertIn("Rubrics Writing", critical)
        self.assertEqual(len(critical), 2)

    def test_dependencies_set_correctly(self):
        estimates = self.gen._estimate_workload(1000, None, "china")
        by_name = {e.phase_name: e for e in estimates}
        self.assertEqual(by_name["Planning & Setup"].dependencies, [])
        self.assertIn("Planning & Setup", by_name["Context Creation"].dependencies)
        self.assertIn("Context Creation", by_name["Task Design"].dependencies)
        self.assertIn("Task Design", by_name["Rubrics Writing"].dependencies)
        self.assertIn("Rubrics Writing", by_name["Quality Review"].dependencies)
        self.assertIn("Quality Review", by_name["Final Validation"].dependencies)

    def test_scaling_with_target_size(self):
        est_1k = self.gen._estimate_workload(1000, None, "china")
        est_5k = self.gen._estimate_workload(5000, None, "china")
        total_1k = sum(e.person_days for e in est_1k)
        total_5k = sum(e.person_days for e in est_5k)
        self.assertGreater(total_5k, total_1k)
        # Should be exactly 5x since linear scaling of person_days
        self.assertAlmostEqual(total_5k / total_1k, 5.0, places=1)

    def test_automation_factor_with_allocation(self):
        allocation = StubHumanMachineAllocation(machine_work_percentage=50.0)
        est_no_alloc = self.gen._estimate_workload(1000, None, "china")
        est_alloc = self.gen._estimate_workload(1000, allocation, "china")
        total_no = sum(e.person_days for e in est_no_alloc)
        total_with = sum(e.person_days for e in est_alloc)
        self.assertLess(total_with, total_no)

    def test_automation_factor_zero_machine_no_reduction(self):
        allocation = StubHumanMachineAllocation(machine_work_percentage=0.0)
        est_no = self.gen._estimate_workload(1000, None, "china")
        est_zero = self.gen._estimate_workload(1000, allocation, "china")
        total_no = sum(e.person_days for e in est_no)
        total_zero = sum(e.person_days for e in est_zero)
        self.assertAlmostEqual(total_no, total_zero, places=2)

    def test_region_us_higher_cost(self):
        est_china = self.gen._estimate_workload(1000, None, "china")
        est_us = self.gen._estimate_workload(1000, None, "us")
        cost_china = sum(e.cost for e in est_china)
        cost_us = sum(e.cost for e in est_us)
        self.assertAlmostEqual(cost_us / cost_china, 3.0, places=1)

    def test_region_europe_higher_cost(self):
        est_china = self.gen._estimate_workload(1000, None, "china")
        est_eu = self.gen._estimate_workload(1000, None, "europe")
        cost_china = sum(e.cost for e in est_china)
        cost_eu = sum(e.cost for e in est_eu)
        self.assertAlmostEqual(cost_eu / cost_china, 2.5, places=1)

    def test_region_india_lower_cost(self):
        est_china = self.gen._estimate_workload(1000, None, "china")
        est_india = self.gen._estimate_workload(1000, None, "india")
        cost_china = sum(e.cost for e in est_china)
        cost_india = sum(e.cost for e in est_india)
        self.assertLess(cost_india, cost_china)
        self.assertAlmostEqual(cost_india / cost_china, 0.7, places=1)

    def test_unknown_region_defaults_to_1x(self):
        est_china = self.gen._estimate_workload(1000, None, "china")
        est_unknown = self.gen._estimate_workload(1000, None, "mars")
        cost_china = sum(e.cost for e in est_china)
        cost_unknown = sum(e.cost for e in est_unknown)
        self.assertAlmostEqual(cost_china, cost_unknown, places=2)

    def test_all_estimates_have_positive_values(self):
        estimates = self.gen._estimate_workload(1000, None, "china")
        for e in estimates:
            self.assertGreater(e.person_days, 0)
            self.assertGreater(e.team_size, 0)
            self.assertGreater(e.duration_weeks, 0)
            self.assertGreater(e.cost, 0)


# ==================== _recommend_team() ====================


class TestRecommendTeam(unittest.TestCase):
    """Test _recommend_team() internal method."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_returns_dict_with_expected_roles(self):
        team = self.gen._recommend_team([])
        expected_roles = [
            "domain_experts",
            "task_designers",
            "annotators",
            "qa_reviewers",
            "project_manager",
        ]
        for role in expected_roles:
            self.assertIn(role, team)

    def test_each_role_has_count_role_skills(self):
        team = self.gen._recommend_team([])
        for role_key, info in team.items():
            self.assertIn("count", info)
            self.assertIn("role", info)
            self.assertIn("skills", info)
            self.assertIsInstance(info["count"], int)
            self.assertIsInstance(info["role"], str)
            self.assertIsInstance(info["skills"], list)
            self.assertGreater(info["count"], 0)
            self.assertGreater(len(info["skills"]), 0)

    def test_project_manager_has_one(self):
        team = self.gen._recommend_team([])
        self.assertEqual(team["project_manager"]["count"], 1)


# ==================== _define_quality_checkpoints() ====================


class TestDefineQualityCheckpoints(unittest.TestCase):
    """Test _define_quality_checkpoints() internal method."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_returns_three_checkpoints_by_default(self):
        checkpoints = self.gen._define_quality_checkpoints(None, None)
        self.assertEqual(len(checkpoints), 3)

    def test_checkpoint_phases(self):
        checkpoints = self.gen._define_quality_checkpoints(None, None)
        phases = [cp["phase"] for cp in checkpoints]
        self.assertEqual(phases, ["Context Creation", "Task Design", "Rubrics Writing"])

    def test_each_checkpoint_has_required_keys(self):
        checkpoints = self.gen._define_quality_checkpoints(None, None)
        for cp in checkpoints:
            self.assertIn("phase", cp)
            self.assertIn("checks", cp)
            self.assertIn("threshold", cp)
            self.assertIsInstance(cp["checks"], list)
            self.assertGreater(len(cp["checks"]), 0)

    def test_rubrics_analysis_adds_pattern_check(self):
        rubrics = StubRubricsAnalysisResult(
            top_templates=["The response should include relevant details"],
        )
        checkpoints = self.gen._define_quality_checkpoints(rubrics, None)
        rubrics_checks = checkpoints[2]["checks"]
        pattern_checks = [c for c in rubrics_checks if "discovered patterns" in c.lower()]
        self.assertEqual(len(pattern_checks), 1)
        self.assertIn("The response should include relevant", pattern_checks[0])

    def test_rubrics_analysis_empty_templates_no_extra_check(self):
        rubrics = StubRubricsAnalysisResult(top_templates=[])
        checkpoints = self.gen._define_quality_checkpoints(rubrics, None)
        rubrics_checks = checkpoints[2]["checks"]
        pattern_checks = [c for c in rubrics_checks if "discovered patterns" in c.lower()]
        self.assertEqual(len(pattern_checks), 0)

    def test_context_strategy_synthetic_adds_check(self):
        strategy = StubContextStrategy(primary_strategy=StubStrategyType.SYNTHETIC)
        checkpoints = self.gen._define_quality_checkpoints(None, strategy)
        context_checks = checkpoints[0]["checks"]
        novelty_checks = [c for c in context_checks if "novelty" in c.lower()]
        self.assertEqual(len(novelty_checks), 1)

    def test_context_strategy_modified_adds_check(self):
        strategy = StubContextStrategy(primary_strategy=StubStrategyType.MODIFIED)
        checkpoints = self.gen._define_quality_checkpoints(None, strategy)
        context_checks = checkpoints[0]["checks"]
        mod_checks = [c for c in context_checks if "modification" in c.lower()]
        self.assertEqual(len(mod_checks), 1)

    def test_context_strategy_niche_adds_check(self):
        strategy = StubContextStrategy(primary_strategy=StubStrategyType.NICHE)
        checkpoints = self.gen._define_quality_checkpoints(None, strategy)
        context_checks = checkpoints[0]["checks"]
        niche_checks = [c for c in context_checks if "expert" in c.lower()]
        self.assertEqual(len(niche_checks), 1)

    def test_context_strategy_unknown_adds_generic_check(self):
        strategy = StubContextStrategy(primary_strategy=StubStrategyType.UNKNOWN)
        checkpoints = self.gen._define_quality_checkpoints(None, strategy)
        context_checks = checkpoints[0]["checks"]
        generic_checks = [c for c in context_checks if "quality" in c.lower()]
        self.assertGreater(len(generic_checks), 0)

    def test_both_rubrics_and_strategy(self):
        rubrics = StubRubricsAnalysisResult(
            top_templates=["Template pattern here"],
        )
        strategy = StubContextStrategy(primary_strategy=StubStrategyType.SYNTHETIC)
        checkpoints = self.gen._define_quality_checkpoints(rubrics, strategy)
        # Should have rubrics pattern check in rubrics phase
        rubrics_checks = checkpoints[2]["checks"]
        self.assertTrue(any("discovered patterns" in c.lower() for c in rubrics_checks))
        # Should have novelty check in context phase
        context_checks = checkpoints[0]["checks"]
        self.assertTrue(any("novelty" in c.lower() for c in context_checks))


# ==================== to_markdown() ====================


class TestToMarkdown(unittest.TestCase):
    """Test to_markdown() output formatting."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def _make_basic_guide(self):
        """Create a basic guide for testing."""
        return self.gen.generate(dataset_name="test-dataset", target_size=1000)

    def test_markdown_contains_header(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn("# Production Guide: test-dataset", md)

    def test_markdown_contains_generation_date(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn(f"> Generated: {guide.generation_date}", md)

    def test_markdown_contains_target_size(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn("> Target Size: 1,000 examples", md)

    def test_markdown_contains_executive_summary_table(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn("## Executive Summary", md)
        self.assertIn("| Metric | Value |", md)
        self.assertIn("Total Effort", md)
        self.assertIn("Timeline", md)
        self.assertIn("Estimated Cost", md)

    def test_markdown_contains_workload_breakdown(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn("## Workload Breakdown", md)
        self.assertIn("| Phase | Person-Days | Team Size | Weeks | Cost | Critical |", md)
        for we in guide.workload_estimates:
            self.assertIn(we.phase_name, md)

    def test_markdown_contains_team_structure(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn("## Recommended Team Structure", md)
        self.assertIn("Domain Experts", md)
        self.assertIn("Task Designers", md)
        self.assertIn("Annotators", md)
        self.assertIn("Qa Reviewers", md)
        self.assertIn("Project Manager", md)

    def test_markdown_contains_quality_checkpoints(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn("## Quality Checkpoints", md)
        self.assertIn("- [ ]", md)  # Checkbox format

    def test_markdown_contains_footer(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        self.assertIn("---", md)
        self.assertIn("> Generated by DataRecipe", md)

    def test_markdown_critical_path_marked(self):
        guide = self._make_basic_guide()
        md = self.gen.to_markdown(guide)
        # Context Creation and Rubrics Writing should be marked as critical
        self.assertIn("| Yes |", md)
        self.assertIn("| No |", md)


class TestToMarkdownWithAllocation(unittest.TestCase):
    """Test to_markdown() with human-machine allocation data."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def _make_guide_with_allocation(self):
        human_task = StubTaskAllocation(
            task_id="T1",
            task_name="Context Writing",
            description="Write original context",
            decision=StubAllocationDecision.HUMAN_ONLY,
            human_percentage=100.0,
            human_hours=40.0,
            human_cost=1000.0,
            machine_cost=0.0,
            machine_method="",
        )
        machine_task = StubTaskAllocation(
            task_id="T2",
            task_name="Format Conversion",
            description="Convert formats",
            decision=StubAllocationDecision.MACHINE_ONLY,
            human_percentage=0.0,
            human_hours=0.0,
            human_cost=0.0,
            machine_cost=50.0,
            machine_method="Automated script",
        )
        allocation = StubHumanMachineAllocation(
            tasks=[human_task, machine_task],
            human_only_tasks=[human_task],
            machine_only_tasks=[machine_task],
            human_work_percentage=70.0,
            machine_work_percentage=30.0,
            estimated_savings_vs_all_human=5000.0,
        )
        return self.gen.generate(
            dataset_name="alloc-test",
            target_size=1000,
            allocation=allocation,
        )

    def test_markdown_contains_allocation_section(self):
        guide = self._make_guide_with_allocation()
        md = self.gen.to_markdown(guide)
        self.assertIn("## Human-Machine Allocation", md)

    def test_markdown_contains_allocation_summary(self):
        guide = self._make_guide_with_allocation()
        md = self.gen.to_markdown(guide)
        self.assertIn("Human Work: 70%", md)
        self.assertIn("Machine Work: 30%", md)
        self.assertIn("Estimated Savings: $5,000", md)

    def test_markdown_contains_task_breakdown_table(self):
        guide = self._make_guide_with_allocation()
        md = self.gen.to_markdown(guide)
        self.assertIn("### Task Breakdown", md)
        self.assertIn("| Task | Allocation | Human % | Human Hours | Cost |", md)
        self.assertIn("Context Writing", md)
        self.assertIn("Format Conversion", md)

    def test_markdown_contains_human_must_do(self):
        guide = self._make_guide_with_allocation()
        md = self.gen.to_markdown(guide)
        self.assertIn("### What Humans Must Do", md)
        self.assertIn("**Context Writing**", md)

    def test_markdown_contains_machine_can_do(self):
        guide = self._make_guide_with_allocation()
        md = self.gen.to_markdown(guide)
        self.assertIn("### What Machines Can Do", md)
        self.assertIn("**Format Conversion**", md)
        self.assertIn("Automated script", md)

    def test_markdown_no_allocation_section_without_allocation(self):
        guide = self.gen.generate(dataset_name="no-alloc", target_size=1000)
        md = self.gen.to_markdown(guide)
        self.assertNotIn("## Human-Machine Allocation", md)


class TestToMarkdownWithRubrics(unittest.TestCase):
    """Test to_markdown() with rubrics analysis data."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_markdown_contains_rubrics_patterns(self):
        rubrics = StubRubricsAnalysisResult(
            total_rubrics=100,
            unique_patterns=25,
            verb_distribution={"include": 30, "explain": 20, "analyze": 15},
            top_templates=[
                "The response should include [topic]",
                "The answer must explain [concept]",
            ],
        )
        guide = self.gen.generate(
            dataset_name="rubrics-test",
            target_size=1000,
            rubrics_analysis=rubrics,
        )
        md = self.gen.to_markdown(guide)
        self.assertIn("## Discovered Patterns", md)
        self.assertIn("### Rubrics Patterns", md)
        self.assertIn("Total analyzed: 100", md)
        self.assertIn("Unique patterns: 25", md)
        self.assertIn("**Top Verbs:**", md)
        self.assertIn("`include`", md)
        self.assertIn("`explain`", md)
        self.assertIn("**Top Templates:**", md)

    def test_markdown_verb_percentages(self):
        rubrics = StubRubricsAnalysisResult(
            total_rubrics=100,
            unique_patterns=10,
            verb_distribution={"include": 50, "explain": 30},
            top_templates=[],
        )
        guide = self.gen.generate(
            dataset_name="test", target_size=1000, rubrics_analysis=rubrics
        )
        md = self.gen.to_markdown(guide)
        self.assertIn("50.0%", md)
        self.assertIn("30.0%", md)

    def test_markdown_no_rubrics_section_without_rubrics(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        md = self.gen.to_markdown(guide)
        self.assertNotIn("## Discovered Patterns", md)
        self.assertNotIn("### Rubrics Patterns", md)


class TestToMarkdownWithPromptLibrary(unittest.TestCase):
    """Test to_markdown() with prompt library data."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_markdown_contains_prompt_section(self):
        prompts = StubPromptLibrary(
            unique_count=15,
            category_counts={"system": 5, "task": 8, "format": 2},
        )
        guide = self.gen.generate(
            dataset_name="prompt-test",
            target_size=1000,
            prompt_library=prompts,
        )
        md = self.gen.to_markdown(guide)
        self.assertIn("### System Prompt Templates", md)
        self.assertIn("Extracted: 15 unique templates", md)
        self.assertIn("**By Category:**", md)
        self.assertIn("system: 5", md)
        self.assertIn("task: 8", md)
        self.assertIn("format: 2", md)

    def test_markdown_no_prompt_section_without_prompts(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        md = self.gen.to_markdown(guide)
        self.assertNotIn("### System Prompt Templates", md)


class TestToMarkdownWithContextStrategy(unittest.TestCase):
    """Test to_markdown() with context strategy data."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_markdown_contains_strategy_section(self):
        strategy = StubContextStrategy(
            primary_strategy=StubStrategyType.SYNTHETIC,
            confidence=0.85,
            recommendations=["Create fictional scenarios", "Verify novelty"],
        )
        guide = self.gen.generate(
            dataset_name="strategy-test",
            target_size=1000,
            context_strategy=strategy,
        )
        md = self.gen.to_markdown(guide)
        self.assertIn("### Context Construction Strategy", md)
        self.assertIn("**Primary Strategy:** synthetic", md)
        self.assertIn("**Confidence:** 85%", md)
        self.assertIn("**Recommendations:**", md)
        self.assertIn("- Create fictional scenarios", md)
        self.assertIn("- Verify novelty", md)

    def test_markdown_no_strategy_section_without_strategy(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        md = self.gen.to_markdown(guide)
        self.assertNotIn("### Context Construction Strategy", md)


# ==================== to_dict() ====================


class TestToDict(unittest.TestCase):
    """Test to_dict() output structure and values."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_dict_contains_required_keys(self):
        guide = self.gen.generate(dataset_name="dict-test", target_size=1000)
        d = self.gen.to_dict(guide)
        self.assertIn("dataset_name", d)
        self.assertIn("target_size", d)
        self.assertIn("generation_date", d)
        self.assertIn("summary", d)
        self.assertIn("workload", d)
        self.assertIn("team", d)
        self.assertIn("quality_checkpoints", d)

    def test_dict_basic_values(self):
        guide = self.gen.generate(dataset_name="dict-test", target_size=2000)
        d = self.gen.to_dict(guide)
        self.assertEqual(d["dataset_name"], "dict-test")
        self.assertEqual(d["target_size"], 2000)

    def test_dict_summary_structure(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        d = self.gen.to_dict(guide)
        summary = d["summary"]
        self.assertIn("total_person_days", summary)
        self.assertIn("total_weeks", summary)
        self.assertIn("total_cost", summary)
        self.assertAlmostEqual(summary["total_person_days"], guide.total_person_days)
        self.assertAlmostEqual(summary["total_weeks"], guide.total_weeks)
        self.assertAlmostEqual(summary["total_cost"], guide.total_cost)

    def test_dict_workload_is_list(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        d = self.gen.to_dict(guide)
        self.assertIsInstance(d["workload"], list)
        self.assertEqual(len(d["workload"]), len(guide.workload_estimates))

    def test_dict_workload_item_structure(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        d = self.gen.to_dict(guide)
        for item in d["workload"]:
            self.assertIn("phase", item)
            self.assertIn("person_days", item)
            self.assertIn("team_size", item)
            self.assertIn("weeks", item)
            self.assertIn("cost", item)
            self.assertIn("critical_path", item)

    def test_dict_team_matches_guide(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        d = self.gen.to_dict(guide)
        self.assertEqual(d["team"], guide.recommended_team)

    def test_dict_quality_checkpoints_matches_guide(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        d = self.gen.to_dict(guide)
        self.assertEqual(d["quality_checkpoints"], guide.quality_checkpoints)

    def test_dict_generation_date_string(self):
        guide = self.gen.generate(dataset_name="test", target_size=1000)
        d = self.gen.to_dict(guide)
        self.assertIsInstance(d["generation_date"], str)
        self.assertRegex(d["generation_date"], r"\d{4}-\d{2}-\d{2}")


# ==================== Integration: generate + to_markdown ====================


class TestGenerateAndMarkdownIntegration(unittest.TestCase):
    """Test full flow: generate() -> to_markdown() with various combinations."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_minimal_guide_produces_valid_markdown(self):
        guide = self.gen.generate(dataset_name="minimal", target_size=500)
        md = self.gen.to_markdown(guide)
        # Should be a non-empty string with markdown headers
        self.assertIsInstance(md, str)
        self.assertGreater(len(md), 100)
        self.assertTrue(md.startswith("# Production Guide:"))

    def test_full_guide_produces_valid_markdown(self):
        rubrics = StubRubricsAnalysisResult(
            total_rubrics=200,
            unique_patterns=40,
            verb_distribution={"include": 60, "explain": 40, "analyze": 30},
            top_templates=[
                "The response should include [topic]",
                "The answer must explain [concept]",
                "Analysis should cover [area]",
            ],
        )
        prompts = StubPromptLibrary(
            unique_count=20,
            category_counts={"system": 10, "task": 8, "format": 2},
        )
        strategy = StubContextStrategy(
            primary_strategy=StubStrategyType.MODIFIED,
            confidence=0.75,
            recommendations=["Track source modifications", "Verify attribution"],
        )
        human_task = StubTaskAllocation(
            task_name="Expert Review",
            description="Domain expert review",
            decision=StubAllocationDecision.HUMAN_ONLY,
            human_percentage=100.0,
            human_hours=80.0,
            human_cost=2000.0,
            machine_cost=0.0,
        )
        machine_task = StubTaskAllocation(
            task_name="Data Formatting",
            description="Auto-format data",
            decision=StubAllocationDecision.MACHINE_ONLY,
            human_percentage=0.0,
            human_hours=0.0,
            human_cost=0.0,
            machine_cost=100.0,
            machine_method="Script-based conversion",
        )
        allocation = StubHumanMachineAllocation(
            tasks=[human_task, machine_task],
            human_only_tasks=[human_task],
            machine_only_tasks=[machine_task],
            human_work_percentage=80.0,
            machine_work_percentage=20.0,
            estimated_savings_vs_all_human=8000.0,
        )
        guide = self.gen.generate(
            dataset_name="full-integration",
            target_size=10000,
            rubrics_analysis=rubrics,
            prompt_library=prompts,
            context_strategy=strategy,
            allocation=allocation,
            region="us",
        )
        md = self.gen.to_markdown(guide)

        # Verify all major sections present
        self.assertIn("# Production Guide: full-integration", md)
        self.assertIn("## Executive Summary", md)
        self.assertIn("## Human-Machine Allocation", md)
        self.assertIn("## Discovered Patterns", md)
        self.assertIn("### System Prompt Templates", md)
        self.assertIn("### Context Construction Strategy", md)
        self.assertIn("## Workload Breakdown", md)
        self.assertIn("## Recommended Team Structure", md)
        self.assertIn("## Quality Checkpoints", md)
        self.assertIn("> Generated by DataRecipe", md)

    def test_different_regions_produce_different_costs_in_markdown(self):
        guide_china = self.gen.generate(
            dataset_name="region-test", target_size=1000, region="china"
        )
        guide_us = self.gen.generate(
            dataset_name="region-test", target_size=1000, region="us"
        )
        md_china = self.gen.to_markdown(guide_china)
        md_us = self.gen.to_markdown(guide_us)
        # The US version should show higher costs
        self.assertGreater(guide_us.total_cost, guide_china.total_cost)
        # Both should be valid markdown
        self.assertIn("# Production Guide:", md_china)
        self.assertIn("# Production Guide:", md_us)


# ==================== Integration: generate + to_dict ====================


class TestGenerateAndDictIntegration(unittest.TestCase):
    """Test full flow: generate() -> to_dict()."""

    def setUp(self):
        self.gen = EnhancedGuideGenerator()

    def test_dict_is_json_serializable(self):
        import json

        guide = self.gen.generate(dataset_name="json-test", target_size=1000)
        d = self.gen.to_dict(guide)
        # Should be JSON-serializable without errors
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)
        # Round-trip should work
        parsed = json.loads(json_str)
        self.assertEqual(parsed["dataset_name"], "json-test")

    def test_dict_workload_values_match_guide(self):
        guide = self.gen.generate(dataset_name="test", target_size=2000)
        d = self.gen.to_dict(guide)
        for i, item in enumerate(d["workload"]):
            we = guide.workload_estimates[i]
            self.assertEqual(item["phase"], we.phase_name)
            self.assertAlmostEqual(item["person_days"], we.person_days)
            self.assertEqual(item["team_size"], we.team_size)
            self.assertAlmostEqual(item["weeks"], we.duration_weeks)
            self.assertAlmostEqual(item["cost"], we.cost)
            self.assertEqual(item["critical_path"], we.critical_path)


if __name__ == "__main__":
    unittest.main()
