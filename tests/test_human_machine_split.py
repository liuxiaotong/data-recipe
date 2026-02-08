"""Unit tests for HumanMachineSplitter and related classes.

Tests TaskType, AllocationDecision, TaskAllocation, HumanMachineAllocation,
and HumanMachineSplitter including analyze(), to_dict(), summary(),
to_markdown_table(), and all helper methods.
"""

import unittest

from datarecipe.generators.human_machine_split import (
    AllocationDecision,
    HumanMachineAllocation,
    HumanMachineSplitter,
    TaskAllocation,
    TaskType,
)


# ==================== Enum Tests ====================


class TestTaskTypeEnum(unittest.TestCase):
    """Test TaskType enum values."""

    def test_all_task_types_exist(self):
        expected = [
            "context_creation",
            "task_design",
            "rubrics_writing",
            "data_collection",
            "data_generation",
            "data_filtering",
            "labeling",
            "quality_review",
            "format_conversion",
            "edge_case_handling",
        ]
        actual = [t.value for t in TaskType]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_task_type_count(self):
        self.assertEqual(len(TaskType), 10)


class TestAllocationDecisionEnum(unittest.TestCase):
    """Test AllocationDecision enum values."""

    def test_all_decisions_exist(self):
        expected = ["human_only", "machine_only", "human_primary", "machine_primary"]
        actual = [d.value for d in AllocationDecision]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_decision_count(self):
        self.assertEqual(len(AllocationDecision), 4)


# ==================== TaskAllocation Dataclass Tests ====================


class TestTaskAllocation(unittest.TestCase):
    """Test TaskAllocation dataclass behavior."""

    def test_default_values(self):
        ta = TaskAllocation(
            task_id="T01",
            task_name="Test Task",
            task_type=TaskType.LABELING,
        )
        self.assertEqual(ta.task_id, "T01")
        self.assertEqual(ta.task_name, "Test Task")
        self.assertEqual(ta.task_type, TaskType.LABELING)
        self.assertEqual(ta.description, "")
        self.assertEqual(ta.decision, AllocationDecision.HUMAN_ONLY)
        self.assertEqual(ta.human_percentage, 100.0)
        self.assertEqual(ta.machine_percentage, 0.0)
        self.assertEqual(ta.human_role, "")
        self.assertEqual(ta.human_hours, 0.0)
        self.assertEqual(ta.human_hourly_rate, 25.0)
        self.assertEqual(ta.machine_method, "")
        self.assertEqual(ta.machine_cost, 0.0)
        self.assertEqual(ta.automation_confidence, 0.0)
        self.assertEqual(ta.rationale, "")
        self.assertEqual(ta.risk_factors, [])
        self.assertEqual(ta.quality_impact, "")

    def test_post_init_calculates_human_cost(self):
        """__post_init__ should compute human_cost = human_hours * human_hourly_rate."""
        ta = TaskAllocation(
            task_id="T01",
            task_name="Test",
            task_type=TaskType.LABELING,
            human_hours=10.0,
            human_hourly_rate=30.0,
        )
        self.assertAlmostEqual(ta.human_cost, 300.0)

    def test_post_init_zero_hours(self):
        ta = TaskAllocation(
            task_id="T01",
            task_name="Test",
            task_type=TaskType.LABELING,
            human_hours=0.0,
            human_hourly_rate=50.0,
        )
        self.assertAlmostEqual(ta.human_cost, 0.0)

    def test_post_init_overrides_provided_human_cost(self):
        """Even if human_cost is supplied, __post_init__ recalculates it."""
        ta = TaskAllocation(
            task_id="T01",
            task_name="Test",
            task_type=TaskType.LABELING,
            human_hours=5.0,
            human_hourly_rate=20.0,
            human_cost=9999.0,  # should be overridden
        )
        self.assertAlmostEqual(ta.human_cost, 100.0)

    def test_risk_factors_independent_instances(self):
        """Ensure default risk_factors list is not shared across instances."""
        ta1 = TaskAllocation(task_id="T01", task_name="A", task_type=TaskType.LABELING)
        ta2 = TaskAllocation(task_id="T02", task_name="B", task_type=TaskType.LABELING)
        ta1.risk_factors.append("Risk 1")
        self.assertEqual(len(ta2.risk_factors), 0)


# ==================== HumanMachineAllocation Dataclass Tests ====================


class TestHumanMachineAllocation(unittest.TestCase):
    """Test HumanMachineAllocation dataclass and its methods."""

    def test_default_values(self):
        alloc = HumanMachineAllocation()
        self.assertEqual(alloc.tasks, [])
        self.assertEqual(alloc.total_human_hours, 0.0)
        self.assertEqual(alloc.total_human_cost, 0.0)
        self.assertEqual(alloc.total_machine_cost, 0.0)
        self.assertEqual(alloc.total_cost, 0.0)
        self.assertEqual(alloc.human_only_tasks, [])
        self.assertEqual(alloc.machine_only_tasks, [])
        self.assertEqual(alloc.hybrid_tasks, [])
        self.assertEqual(alloc.human_work_percentage, 0.0)
        self.assertEqual(alloc.machine_work_percentage, 0.0)
        self.assertEqual(alloc.estimated_savings_vs_all_human, 0.0)
        self.assertEqual(alloc.timeline_reduction_percentage, 0.0)

    def test_lists_are_independent_instances(self):
        a1 = HumanMachineAllocation()
        a2 = HumanMachineAllocation()
        t = TaskAllocation(task_id="T01", task_name="X", task_type=TaskType.LABELING)
        a1.tasks.append(t)
        self.assertEqual(len(a2.tasks), 0)


class TestHumanMachineAllocationSummary(unittest.TestCase):
    """Test HumanMachineAllocation.summary() method."""

    def _make_allocation(self, **overrides):
        defaults = {
            "tasks": [],
            "human_only_tasks": [],
            "machine_only_tasks": [],
            "hybrid_tasks": [],
            "total_human_hours": 100.0,
            "total_human_cost": 2500.0,
            "total_machine_cost": 500.0,
            "total_cost": 3000.0,
            "human_work_percentage": 70.0,
            "machine_work_percentage": 30.0,
            "estimated_savings_vs_all_human": 0.0,
            "timeline_reduction_percentage": 0.0,
        }
        defaults.update(overrides)
        return HumanMachineAllocation(**defaults)

    def test_summary_contains_header(self):
        alloc = self._make_allocation()
        s = alloc.summary()
        self.assertIn("HUMAN-MACHINE ALLOCATION SUMMARY", s)

    def test_summary_contains_task_counts(self):
        t1 = TaskAllocation(task_id="T01", task_name="A", task_type=TaskType.LABELING)
        t2 = TaskAllocation(task_id="T02", task_name="B", task_type=TaskType.DATA_GENERATION)
        alloc = self._make_allocation(
            tasks=[t1, t2],
            human_only_tasks=[t1],
            machine_only_tasks=[t2],
        )
        s = alloc.summary()
        self.assertIn("Total Tasks: 2", s)
        self.assertIn("Human Only: 1", s)
        self.assertIn("Machine Only: 1", s)
        self.assertIn("Hybrid: 0", s)

    def test_summary_contains_costs(self):
        alloc = self._make_allocation(
            total_human_cost=5000.0,
            total_machine_cost=1000.0,
            total_cost=6000.0,
            total_human_hours=200.0,
        )
        s = alloc.summary()
        self.assertIn("$5,000", s)
        self.assertIn("$1,000", s)
        self.assertIn("$6,000", s)
        self.assertIn("200 hours", s)

    def test_summary_contains_workload_split(self):
        alloc = self._make_allocation(
            human_work_percentage=65.0,
            machine_work_percentage=35.0,
        )
        s = alloc.summary()
        self.assertIn("Human: 65%", s)
        self.assertIn("Machine: 35%", s)

    def test_summary_includes_savings_when_positive(self):
        alloc = self._make_allocation(
            estimated_savings_vs_all_human=2000.0,
            timeline_reduction_percentage=25.0,
        )
        s = alloc.summary()
        self.assertIn("$2,000", s)
        self.assertIn("25%", s)

    def test_summary_no_savings_when_zero(self):
        alloc = self._make_allocation(
            estimated_savings_vs_all_human=0.0,
        )
        s = alloc.summary()
        self.assertNotIn("Estimated Savings", s)
        self.assertNotIn("Timeline Reduction", s)


class TestHumanMachineAllocationMarkdownTable(unittest.TestCase):
    """Test HumanMachineAllocation.to_markdown_table() method."""

    def test_markdown_table_header(self):
        alloc = HumanMachineAllocation()
        md = alloc.to_markdown_table()
        self.assertIn("| Task |", md)
        self.assertIn("| Type |", md)
        self.assertIn("| Allocation |", md)
        self.assertIn("|------|", md)

    def test_markdown_table_with_tasks(self):
        t = TaskAllocation(
            task_id="T01",
            task_name="Data Labeling",
            task_type=TaskType.LABELING,
            decision=AllocationDecision.HUMAN_PRIMARY,
            human_percentage=70.0,
            human_hours=50.0,
            human_hourly_rate=25.0,
            machine_cost=100.0,
        )
        alloc = HumanMachineAllocation(tasks=[t])
        md = alloc.to_markdown_table()
        self.assertIn("Data Labeling", md)
        self.assertIn("labeling", md)
        self.assertIn("human_primary", md)
        self.assertIn("70%", md)
        self.assertIn("50.0h", md)
        self.assertIn("$1250", md)  # 50 * 25
        self.assertIn("$100", md)

    def test_markdown_table_multiple_tasks(self):
        t1 = TaskAllocation(
            task_id="T01",
            task_name="Task A",
            task_type=TaskType.LABELING,
            decision=AllocationDecision.HUMAN_ONLY,
            human_percentage=95.0,
            human_hours=10.0,
        )
        t2 = TaskAllocation(
            task_id="T02",
            task_name="Task B",
            task_type=TaskType.FORMAT_CONVERSION,
            decision=AllocationDecision.MACHINE_ONLY,
            human_percentage=5.0,
            human_hours=1.0,
            machine_cost=50.0,
        )
        alloc = HumanMachineAllocation(tasks=[t1, t2])
        md = alloc.to_markdown_table()
        lines = md.strip().split("\n")
        # header + separator + 2 data rows
        self.assertEqual(len(lines), 4)
        self.assertIn("Task A", lines[2])
        self.assertIn("Task B", lines[3])

    def test_markdown_table_empty_tasks(self):
        alloc = HumanMachineAllocation()
        md = alloc.to_markdown_table()
        lines = md.strip().split("\n")
        # Only header and separator
        self.assertEqual(len(lines), 2)


# ==================== HumanMachineSplitter Tests ====================


class TestHumanMachineSplitterInit(unittest.TestCase):
    """Test HumanMachineSplitter initialization."""

    def test_default_region_china(self):
        splitter = HumanMachineSplitter()
        self.assertEqual(splitter.region, "china")
        self.assertEqual(splitter.rates, HumanMachineSplitter.HOURLY_RATES["china"])

    def test_region_us(self):
        splitter = HumanMachineSplitter(region="us")
        self.assertEqual(splitter.region, "us")
        self.assertEqual(splitter.rates["general"], 25.0)
        self.assertEqual(splitter.rates["expert"], 75.0)
        self.assertEqual(splitter.rates["professional"], 150.0)

    def test_region_europe(self):
        splitter = HumanMachineSplitter(region="europe")
        self.assertEqual(splitter.rates["general"], 20.0)

    def test_region_india(self):
        splitter = HumanMachineSplitter(region="india")
        self.assertEqual(splitter.rates["general"], 5.0)
        self.assertEqual(splitter.rates["expert"], 15.0)
        self.assertEqual(splitter.rates["professional"], 35.0)

    def test_unknown_region_falls_back_to_us(self):
        splitter = HumanMachineSplitter(region="mars")
        self.assertEqual(splitter.rates, HumanMachineSplitter.HOURLY_RATES["us"])


class TestHumanMachineSplitterAnalyze(unittest.TestCase):
    """Test HumanMachineSplitter.analyze() method."""

    def setUp(self):
        self.splitter = HumanMachineSplitter(region="china")

    def test_returns_human_machine_allocation(self):
        result = self.splitter.analyze(dataset_size=1000)
        self.assertIsInstance(result, HumanMachineAllocation)

    def test_default_task_types_all(self):
        """When task_types is None, all TaskType values should be used."""
        result = self.splitter.analyze(dataset_size=1000)
        self.assertEqual(len(result.tasks), len(TaskType))

    def test_specific_task_types(self):
        types = [TaskType.CONTEXT_CREATION, TaskType.LABELING]
        result = self.splitter.analyze(dataset_size=1000, task_types=types)
        self.assertEqual(len(result.tasks), 2)
        actual_types = [t.task_type for t in result.tasks]
        self.assertEqual(actual_types, types)

    def test_single_task_type(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.FORMAT_CONVERSION],
        )
        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(result.tasks[0].task_type, TaskType.FORMAT_CONVERSION)

    def test_task_ids_sequential(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING, TaskType.QUALITY_REVIEW, TaskType.DATA_GENERATION],
        )
        ids = [t.task_id for t in result.tasks]
        self.assertEqual(ids, ["T01", "T02", "T03"])

    def test_default_context_count(self):
        """context_count defaults to dataset_size // 4."""
        result = self.splitter.analyze(
            dataset_size=4000,
            task_types=[TaskType.CONTEXT_CREATION],
        )
        # For context_creation, total_hours = base_hours * (context_count / 1000)
        # context_count = 4000 // 4 = 1000
        # base_hours = 200
        # total_hours = 200 * (1000 / 1000) = 200
        task = result.tasks[0]
        # human_pct for CONTEXT_CREATION (automation=0.2 -> HUMAN_ONLY -> 95%)
        expected_human_hours = 200.0 * 0.95
        self.assertAlmostEqual(task.human_hours, expected_human_hours, places=1)

    def test_custom_context_count(self):
        result = self.splitter.analyze(
            dataset_size=4000,
            task_types=[TaskType.CONTEXT_CREATION],
            context_count=500,
        )
        task = result.tasks[0]
        # total_hours = 200 * (500 / 1000) = 100
        # human_hours = 100 * 0.95 = 95
        self.assertAlmostEqual(task.human_hours, 95.0, places=1)

    def test_context_count_minimum_1(self):
        """Very small datasets should still have context_count >= 1."""
        result = self.splitter.analyze(
            dataset_size=2,
            task_types=[TaskType.CONTEXT_CREATION],
        )
        # context_count = max(1, 2 // 4) = max(1, 0) = 1
        task = result.tasks[0]
        # total_hours = 200 * (1 / 1000) = 0.2
        # human_hours = 0.2 * 0.95 = 0.19
        self.assertAlmostEqual(task.human_hours, 0.19, places=2)

    def test_custom_rubrics_per_task(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.RUBRICS_WRITING],
            rubrics_per_task=30.0,
        )
        task = result.tasks[0]
        # scale_factor = 1.0, base_hours = 100
        # total_hours = 100 * 1.0 * (30 / 15) = 200
        # RUBRICS_WRITING automation = 0.4 -> HUMAN_PRIMARY -> human_pct = 70%
        # human_hours = 200 * 0.7 = 140
        self.assertAlmostEqual(task.human_hours, 140.0, places=1)

    def test_custom_hours_override(self):
        custom = {TaskType.LABELING: 100.0}
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
            custom_hours=custom,
        )
        task = result.tasks[0]
        # base_hours = 100 (custom), scale_factor = 1.0
        # total_hours = 100 * 1.0 = 100
        # LABELING automation = 0.5 -> HUMAN_PRIMARY -> 70%
        # human_hours = 100 * 0.7 = 70
        self.assertAlmostEqual(task.human_hours, 70.0, places=1)

    def test_categorization_human_only(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.EDGE_CASE_HANDLING],  # automation = 0.1 -> HUMAN_ONLY
        )
        self.assertEqual(len(result.human_only_tasks), 1)
        self.assertEqual(len(result.machine_only_tasks), 0)
        self.assertEqual(len(result.hybrid_tasks), 0)

    def test_categorization_machine_only(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.DATA_GENERATION],  # automation = 0.9 -> MACHINE_ONLY
        )
        self.assertEqual(len(result.human_only_tasks), 0)
        self.assertEqual(len(result.machine_only_tasks), 1)
        self.assertEqual(len(result.hybrid_tasks), 0)

    def test_categorization_machine_primary_is_hybrid(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.DATA_COLLECTION],  # automation = 0.7 -> MACHINE_PRIMARY
        )
        self.assertEqual(len(result.hybrid_tasks), 1)

    def test_categorization_human_primary_is_hybrid(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.RUBRICS_WRITING],  # automation = 0.4 -> HUMAN_PRIMARY
        )
        self.assertEqual(len(result.hybrid_tasks), 1)

    def test_totals_calculated(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING, TaskType.FORMAT_CONVERSION],
        )
        self.assertGreater(result.total_human_hours, 0)
        self.assertGreater(result.total_cost, 0)
        self.assertAlmostEqual(
            result.total_cost,
            result.total_human_cost + result.total_machine_cost,
            places=2,
        )

    def test_work_percentages_sum_to_100(self):
        result = self.splitter.analyze(dataset_size=1000)
        total_pct = result.human_work_percentage + result.machine_work_percentage
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_savings_non_negative(self):
        result = self.splitter.analyze(dataset_size=1000)
        self.assertGreaterEqual(result.estimated_savings_vs_all_human, 0.0)

    def test_large_dataset_higher_costs(self):
        small = self.splitter.analyze(
            dataset_size=100,
            task_types=[TaskType.LABELING],
        )
        large = self.splitter.analyze(
            dataset_size=10000,
            task_types=[TaskType.LABELING],
        )
        self.assertGreater(large.total_cost, small.total_cost)


class TestHumanMachineSplitterAllocationDecisions(unittest.TestCase):
    """Test that the allocation decisions match automation feasibility thresholds."""

    def setUp(self):
        self.splitter = HumanMachineSplitter(region="us")

    def test_format_conversion_is_machine_only(self):
        """FORMAT_CONVERSION has automation=1.0, should be MACHINE_ONLY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.FORMAT_CONVERSION],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.MACHINE_ONLY)
        self.assertAlmostEqual(result.tasks[0].human_percentage, 5.0)
        self.assertAlmostEqual(result.tasks[0].machine_percentage, 95.0)

    def test_data_generation_is_machine_only(self):
        """DATA_GENERATION has automation=0.9, should be MACHINE_ONLY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.DATA_GENERATION],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.MACHINE_ONLY)

    def test_data_collection_is_machine_primary(self):
        """DATA_COLLECTION has automation=0.7, should be MACHINE_PRIMARY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.DATA_COLLECTION],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.MACHINE_PRIMARY)
        self.assertAlmostEqual(result.tasks[0].human_percentage, 20.0)
        self.assertAlmostEqual(result.tasks[0].machine_percentage, 80.0)

    def test_data_filtering_is_machine_primary(self):
        """DATA_FILTERING has automation=0.8, should be MACHINE_PRIMARY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.DATA_FILTERING],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.MACHINE_PRIMARY)

    def test_rubrics_writing_is_human_primary(self):
        """RUBRICS_WRITING has automation=0.4, should be HUMAN_PRIMARY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.RUBRICS_WRITING],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.HUMAN_PRIMARY)
        self.assertAlmostEqual(result.tasks[0].human_percentage, 70.0)
        self.assertAlmostEqual(result.tasks[0].machine_percentage, 30.0)

    def test_labeling_is_human_primary(self):
        """LABELING has automation=0.5, should be HUMAN_PRIMARY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.HUMAN_PRIMARY)

    def test_context_creation_is_human_only(self):
        """CONTEXT_CREATION has automation=0.2, should be HUMAN_ONLY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.CONTEXT_CREATION],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.HUMAN_ONLY)
        self.assertAlmostEqual(result.tasks[0].human_percentage, 95.0)
        self.assertAlmostEqual(result.tasks[0].machine_percentage, 5.0)

    def test_edge_case_handling_is_human_only(self):
        """EDGE_CASE_HANDLING has automation=0.1, should be HUMAN_ONLY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.EDGE_CASE_HANDLING],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.HUMAN_ONLY)

    def test_quality_review_is_human_only(self):
        """QUALITY_REVIEW has automation=0.3, should be HUMAN_ONLY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.QUALITY_REVIEW],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.HUMAN_ONLY)

    def test_task_design_is_human_only(self):
        """TASK_DESIGN has automation=0.3, should be HUMAN_ONLY."""
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.TASK_DESIGN],
        )
        self.assertEqual(result.tasks[0].decision, AllocationDecision.HUMAN_ONLY)


class TestHumanMachineSplitterRegionCosts(unittest.TestCase):
    """Test that different regions produce different costs."""

    def test_china_cheaper_than_us(self):
        china = HumanMachineSplitter(region="china")
        us = HumanMachineSplitter(region="us")
        result_cn = china.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
        )
        result_us = us.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
        )
        self.assertLess(result_cn.total_human_cost, result_us.total_human_cost)

    def test_india_cheapest(self):
        india = HumanMachineSplitter(region="india")
        china = HumanMachineSplitter(region="china")
        result_in = india.analyze(
            dataset_size=1000,
            task_types=[TaskType.CONTEXT_CREATION],
        )
        result_cn = china.analyze(
            dataset_size=1000,
            task_types=[TaskType.CONTEXT_CREATION],
        )
        self.assertLess(result_in.total_human_cost, result_cn.total_human_cost)

    def test_expert_tasks_use_expert_rates(self):
        """CONTEXT_CREATION requires expert level; should use expert rate."""
        splitter = HumanMachineSplitter(region="us")
        result = splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.CONTEXT_CREATION],
        )
        task = result.tasks[0]
        self.assertEqual(task.human_role, "expert")
        self.assertAlmostEqual(task.human_hourly_rate, 75.0)

    def test_professional_tasks_use_professional_rates(self):
        """EDGE_CASE_HANDLING requires professional level."""
        splitter = HumanMachineSplitter(region="us")
        result = splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.EDGE_CASE_HANDLING],
        )
        task = result.tasks[0]
        self.assertEqual(task.human_role, "professional")
        self.assertAlmostEqual(task.human_hourly_rate, 150.0)

    def test_general_tasks_use_general_rates(self):
        """LABELING requires general level."""
        splitter = HumanMachineSplitter(region="china")
        result = splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
        )
        task = result.tasks[0]
        self.assertEqual(task.human_role, "general")
        self.assertAlmostEqual(task.human_hourly_rate, 8.0)


# ==================== Helper Method Tests ====================


class TestEstimateMachineCost(unittest.TestCase):
    """Test _estimate_machine_cost helper."""

    def setUp(self):
        self.splitter = HumanMachineSplitter()

    def test_data_generation_cost(self):
        cost = self.splitter._estimate_machine_cost(
            TaskType.DATA_GENERATION, scale_factor=1.0, machine_pct=100.0
        )
        # base = 50.0, scale=1.0, pct=100%
        self.assertAlmostEqual(cost, 50.0)

    def test_data_filtering_cost(self):
        cost = self.splitter._estimate_machine_cost(
            TaskType.DATA_FILTERING, scale_factor=2.0, machine_pct=80.0
        )
        # base = 10.0, scale=2.0, pct=80%
        self.assertAlmostEqual(cost, 16.0)

    def test_format_conversion_cost(self):
        cost = self.splitter._estimate_machine_cost(
            TaskType.FORMAT_CONVERSION, scale_factor=1.0, machine_pct=95.0
        )
        # base = 5.0, scale=1.0, pct=95%
        self.assertAlmostEqual(cost, 4.75)

    def test_labeling_cost(self):
        cost = self.splitter._estimate_machine_cost(
            TaskType.LABELING, scale_factor=5.0, machine_pct=30.0
        )
        # base = 30.0, scale=5.0, pct=30%
        self.assertAlmostEqual(cost, 45.0)

    def test_unknown_task_type_default_base(self):
        """Task types not in base_costs dict default to 5.0."""
        cost = self.splitter._estimate_machine_cost(
            TaskType.CONTEXT_CREATION, scale_factor=1.0, machine_pct=100.0
        )
        self.assertAlmostEqual(cost, 5.0)

    def test_zero_machine_pct(self):
        cost = self.splitter._estimate_machine_cost(
            TaskType.DATA_GENERATION, scale_factor=1.0, machine_pct=0.0
        )
        self.assertAlmostEqual(cost, 0.0)

    def test_scaling(self):
        cost_1x = self.splitter._estimate_machine_cost(
            TaskType.DATA_GENERATION, scale_factor=1.0, machine_pct=100.0
        )
        cost_10x = self.splitter._estimate_machine_cost(
            TaskType.DATA_GENERATION, scale_factor=10.0, machine_pct=100.0
        )
        self.assertAlmostEqual(cost_10x, cost_1x * 10)


class TestGetTaskDescription(unittest.TestCase):
    """Test _get_task_description helper."""

    def setUp(self):
        self.splitter = HumanMachineSplitter()

    def test_all_task_types_have_descriptions(self):
        for task_type in TaskType:
            desc = self.splitter._get_task_description(task_type)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0, f"No description for {task_type}")

    def test_specific_descriptions(self):
        self.assertIn("context", self.splitter._get_task_description(TaskType.CONTEXT_CREATION).lower())
        self.assertIn("label", self.splitter._get_task_description(TaskType.LABELING).lower())
        self.assertIn("quality", self.splitter._get_task_description(TaskType.QUALITY_REVIEW).lower())
        self.assertIn("format", self.splitter._get_task_description(TaskType.FORMAT_CONVERSION).lower())


class TestGetMachineMethod(unittest.TestCase):
    """Test _get_machine_method helper."""

    def setUp(self):
        self.splitter = HumanMachineSplitter()

    def test_data_generation_method(self):
        method = self.splitter._get_machine_method(TaskType.DATA_GENERATION)
        self.assertIn("LLM", method)

    def test_format_conversion_method(self):
        method = self.splitter._get_machine_method(TaskType.FORMAT_CONVERSION)
        self.assertIn("script", method.lower())

    def test_unknown_returns_na(self):
        """Tasks not in the methods dict return 'N/A'."""
        method = self.splitter._get_machine_method(TaskType.CONTEXT_CREATION)
        self.assertEqual(method, "N/A")

    def test_rubrics_writing_method(self):
        method = self.splitter._get_machine_method(TaskType.RUBRICS_WRITING)
        self.assertIn("template", method.lower())


class TestGetRationale(unittest.TestCase):
    """Test _get_rationale helper."""

    def setUp(self):
        self.splitter = HumanMachineSplitter()

    def test_low_automation_rationale(self):
        rationale = self.splitter._get_rationale(TaskType.CONTEXT_CREATION, 0.2)
        self.assertIn("creativity", rationale.lower())
        self.assertIn("20%", rationale)

    def test_medium_automation_rationale(self):
        rationale = self.splitter._get_rationale(TaskType.LABELING, 0.5)
        self.assertIn("oversight", rationale.lower())
        self.assertIn("50%", rationale)

    def test_high_automation_rationale(self):
        rationale = self.splitter._get_rationale(TaskType.DATA_COLLECTION, 0.7)
        self.assertIn("automatable", rationale.lower())
        self.assertIn("70%", rationale)

    def test_full_automation_rationale(self):
        rationale = self.splitter._get_rationale(TaskType.FORMAT_CONVERSION, 0.95)
        self.assertIn("fully automatable", rationale.lower())

    def test_boundary_0_3(self):
        """Boundary: automation=0.3 should be 'creativity/judgment'."""
        rationale = self.splitter._get_rationale(TaskType.QUALITY_REVIEW, 0.29)
        self.assertIn("creativity", rationale.lower())

    def test_boundary_0_6(self):
        """Boundary: automation=0.6 should be 'oversight'."""
        rationale = self.splitter._get_rationale(TaskType.LABELING, 0.59)
        self.assertIn("oversight", rationale.lower())

    def test_boundary_0_9(self):
        """Boundary: automation=0.89 should be 'mostly automatable'."""
        rationale = self.splitter._get_rationale(TaskType.DATA_FILTERING, 0.89)
        self.assertIn("mostly automatable", rationale.lower())


class TestGetRiskFactors(unittest.TestCase):
    """Test _get_risk_factors helper."""

    def setUp(self):
        self.splitter = HumanMachineSplitter()

    def test_human_only_has_cost_and_scaling_risks(self):
        risks = self.splitter._get_risk_factors(TaskType.LABELING, AllocationDecision.HUMAN_ONLY)
        self.assertIn("Higher cost and longer timeline", risks)
        self.assertIn("Scaling challenges", risks)

    def test_machine_only_context_creation_has_quality_risk(self):
        risks = self.splitter._get_risk_factors(TaskType.CONTEXT_CREATION, AllocationDecision.MACHINE_ONLY)
        self.assertIn("Quality may suffer without human creativity", risks)

    def test_machine_primary_task_design_has_quality_risk(self):
        risks = self.splitter._get_risk_factors(TaskType.TASK_DESIGN, AllocationDecision.MACHINE_PRIMARY)
        self.assertIn("Quality may suffer without human creativity", risks)

    def test_machine_only_quality_review_has_subtle_issues_risk(self):
        risks = self.splitter._get_risk_factors(TaskType.QUALITY_REVIEW, AllocationDecision.MACHINE_ONLY)
        self.assertIn("May miss subtle quality issues", risks)

    def test_machine_primary_quality_review_has_subtle_issues_risk(self):
        risks = self.splitter._get_risk_factors(TaskType.QUALITY_REVIEW, AllocationDecision.MACHINE_PRIMARY)
        self.assertIn("May miss subtle quality issues", risks)

    def test_human_primary_no_machine_risks(self):
        risks = self.splitter._get_risk_factors(TaskType.LABELING, AllocationDecision.HUMAN_PRIMARY)
        self.assertEqual(risks, [])

    def test_machine_only_labeling_no_creativity_risk(self):
        """LABELING is not context_creation/task_design, so no creativity risk."""
        risks = self.splitter._get_risk_factors(TaskType.LABELING, AllocationDecision.MACHINE_ONLY)
        self.assertNotIn("Quality may suffer without human creativity", risks)


# ==================== _calculate_totals Tests ====================


class TestCalculateTotals(unittest.TestCase):
    """Test _calculate_totals method."""

    def setUp(self):
        self.splitter = HumanMachineSplitter(region="china")

    def test_totals_with_single_task(self):
        result = HumanMachineAllocation()
        task = TaskAllocation(
            task_id="T01",
            task_name="Test",
            task_type=TaskType.LABELING,
            human_hours=50.0,
            human_hourly_rate=8.0,
            human_percentage=70.0,
            machine_cost=100.0,
        )
        result.tasks.append(task)
        self.splitter._calculate_totals(result)

        self.assertAlmostEqual(result.total_human_hours, 50.0)
        self.assertAlmostEqual(result.total_human_cost, 400.0)  # 50 * 8
        self.assertAlmostEqual(result.total_machine_cost, 100.0)
        self.assertAlmostEqual(result.total_cost, 500.0)

    def test_totals_with_multiple_tasks(self):
        result = HumanMachineAllocation()
        t1 = TaskAllocation(
            task_id="T01",
            task_name="A",
            task_type=TaskType.LABELING,
            human_hours=20.0,
            human_hourly_rate=10.0,
            human_percentage=70.0,
            machine_cost=50.0,
        )
        t2 = TaskAllocation(
            task_id="T02",
            task_name="B",
            task_type=TaskType.DATA_GENERATION,
            human_hours=5.0,
            human_hourly_rate=10.0,
            human_percentage=5.0,
            machine_cost=200.0,
        )
        result.tasks.extend([t1, t2])
        self.splitter._calculate_totals(result)

        self.assertAlmostEqual(result.total_human_hours, 25.0)
        self.assertAlmostEqual(result.total_human_cost, 250.0)  # 200 + 50
        self.assertAlmostEqual(result.total_machine_cost, 250.0)
        self.assertAlmostEqual(result.total_cost, 500.0)

    def test_work_percentages_calculated(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING, TaskType.DATA_GENERATION],
        )
        # Both percentages should be meaningful and sum to 100
        self.assertGreater(result.human_work_percentage, 0)
        self.assertGreater(result.machine_work_percentage, 0)
        self.assertAlmostEqual(
            result.human_work_percentage + result.machine_work_percentage,
            100.0,
            places=1,
        )

    def test_savings_calculated(self):
        """Estimated savings should be positive when automation reduces cost."""
        result = self.splitter.analyze(
            dataset_size=5000,
            task_types=[TaskType.DATA_GENERATION, TaskType.FORMAT_CONVERSION, TaskType.LABELING],
        )
        # Highly automatable tasks should produce savings
        self.assertGreater(result.estimated_savings_vs_all_human, 0)

    def test_timeline_reduction_percentage(self):
        result = self.splitter.analyze(
            dataset_size=5000,
            task_types=[TaskType.DATA_GENERATION, TaskType.FORMAT_CONVERSION],
        )
        if result.estimated_savings_vs_all_human > 0:
            self.assertGreater(result.timeline_reduction_percentage, 0)

    def test_empty_tasks(self):
        result = HumanMachineAllocation()
        self.splitter._calculate_totals(result)
        self.assertAlmostEqual(result.total_human_hours, 0.0)
        self.assertAlmostEqual(result.total_cost, 0.0)
        self.assertAlmostEqual(result.human_work_percentage, 0.0)
        self.assertAlmostEqual(result.machine_work_percentage, 0.0)


# ==================== to_dict Tests ====================


class TestHumanMachineSplitterToDict(unittest.TestCase):
    """Test HumanMachineSplitter.to_dict() method."""

    def setUp(self):
        self.splitter = HumanMachineSplitter(region="china")

    def test_dict_has_summary_and_tasks(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
        )
        d = self.splitter.to_dict(result)
        self.assertIn("summary", d)
        self.assertIn("tasks", d)

    def test_summary_keys(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
        )
        d = self.splitter.to_dict(result)
        summary = d["summary"]
        expected_keys = [
            "total_tasks",
            "human_only_count",
            "machine_only_count",
            "hybrid_count",
            "total_human_hours",
            "total_human_cost",
            "total_machine_cost",
            "total_cost",
            "human_work_percentage",
            "machine_work_percentage",
            "estimated_savings",
        ]
        for key in expected_keys:
            self.assertIn(key, summary, f"Missing key: {key}")

    def test_summary_values_match_allocation(self):
        result = self.splitter.analyze(
            dataset_size=2000,
            task_types=[TaskType.LABELING, TaskType.DATA_GENERATION],
        )
        d = self.splitter.to_dict(result)
        summary = d["summary"]
        self.assertEqual(summary["total_tasks"], len(result.tasks))
        self.assertEqual(summary["human_only_count"], len(result.human_only_tasks))
        self.assertEqual(summary["machine_only_count"], len(result.machine_only_tasks))
        self.assertEqual(summary["hybrid_count"], len(result.hybrid_tasks))
        self.assertAlmostEqual(summary["total_human_hours"], result.total_human_hours)
        self.assertAlmostEqual(summary["total_cost"], result.total_cost)

    def test_tasks_list_structure(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.LABELING],
        )
        d = self.splitter.to_dict(result)
        self.assertIsInstance(d["tasks"], list)
        self.assertEqual(len(d["tasks"]), 1)
        task = d["tasks"][0]
        expected_keys = [
            "task_id",
            "task_name",
            "task_type",
            "decision",
            "human_percentage",
            "human_hours",
            "human_cost",
            "machine_cost",
            "rationale",
            "risks",
        ]
        for key in expected_keys:
            self.assertIn(key, task, f"Missing key in task: {key}")

    def test_task_values_serialized_correctly(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.DATA_GENERATION],
        )
        d = self.splitter.to_dict(result)
        task = d["tasks"][0]
        self.assertEqual(task["task_type"], "data_generation")
        self.assertEqual(task["decision"], "machine_only")
        self.assertIsInstance(task["risks"], list)

    def test_multiple_tasks_in_dict(self):
        types = [TaskType.CONTEXT_CREATION, TaskType.LABELING, TaskType.FORMAT_CONVERSION]
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=types,
        )
        d = self.splitter.to_dict(result)
        self.assertEqual(len(d["tasks"]), 3)
        task_ids = [t["task_id"] for t in d["tasks"]]
        self.assertEqual(task_ids, ["T01", "T02", "T03"])

    def test_empty_allocation_dict(self):
        result = HumanMachineAllocation()
        d = self.splitter.to_dict(result)
        self.assertEqual(d["summary"]["total_tasks"], 0)
        self.assertEqual(d["tasks"], [])


# ==================== Task Name Tests ====================


class TestTaskNameFormatting(unittest.TestCase):
    """Test that task names are formatted correctly."""

    def setUp(self):
        self.splitter = HumanMachineSplitter()

    def test_task_names_are_title_case(self):
        result = self.splitter.analyze(dataset_size=1000)
        for task in result.tasks:
            # Each word should start with uppercase
            self.assertEqual(task.task_name, task.task_name.title())

    def test_context_creation_name(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.CONTEXT_CREATION],
        )
        self.assertEqual(result.tasks[0].task_name, "Context Creation")

    def test_edge_case_handling_name(self):
        result = self.splitter.analyze(
            dataset_size=1000,
            task_types=[TaskType.EDGE_CASE_HANDLING],
        )
        self.assertEqual(result.tasks[0].task_name, "Edge Case Handling")


# ==================== Integration Tests ====================


class TestHumanMachineSplitterIntegration(unittest.TestCase):
    """Integration tests simulating realistic usage."""

    def test_full_workflow_china(self):
        """Simulate a realistic dataset production analysis for China region."""
        splitter = HumanMachineSplitter(region="china")
        result = splitter.analyze(
            dataset_size=10000,
            task_types=[
                TaskType.CONTEXT_CREATION,
                TaskType.TASK_DESIGN,
                TaskType.RUBRICS_WRITING,
                TaskType.LABELING,
                TaskType.QUALITY_REVIEW,
            ],
            context_count=2500,
            rubrics_per_task=15.0,
        )

        # Should have 5 tasks
        self.assertEqual(len(result.tasks), 5)

        # Check categorization
        self.assertGreater(len(result.human_only_tasks), 0)
        self.assertGreater(len(result.hybrid_tasks), 0)

        # Summary should be non-empty
        summary = result.summary()
        self.assertIn("HUMAN-MACHINE ALLOCATION SUMMARY", summary)

        # Table should contain all tasks
        table = result.to_markdown_table()
        for task in result.tasks:
            self.assertIn(task.task_name, table)

        # to_dict should be serializable
        d = splitter.to_dict(result)
        self.assertEqual(d["summary"]["total_tasks"], 5)

    def test_full_workflow_us(self):
        """Simulate analysis for US region - should be more expensive."""
        splitter_us = HumanMachineSplitter(region="us")
        splitter_cn = HumanMachineSplitter(region="china")

        types = [TaskType.CONTEXT_CREATION, TaskType.LABELING]
        result_us = splitter_us.analyze(dataset_size=1000, task_types=types)
        result_cn = splitter_cn.analyze(dataset_size=1000, task_types=types)

        self.assertGreater(result_us.total_human_cost, result_cn.total_human_cost)
        # Machine costs should be the same regardless of region
        self.assertAlmostEqual(
            result_us.total_machine_cost,
            result_cn.total_machine_cost,
            places=2,
        )

    def test_automation_confidence_values(self):
        """Automation confidence should be set from the AUTOMATION_FEASIBILITY dict."""
        splitter = HumanMachineSplitter()
        result = splitter.analyze(dataset_size=1000)
        for task in result.tasks:
            expected = HumanMachineSplitter.AUTOMATION_FEASIBILITY[task.task_type]
            self.assertAlmostEqual(task.automation_confidence, expected)


if __name__ == "__main__":
    unittest.main()
