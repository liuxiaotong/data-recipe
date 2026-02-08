"""Unit tests for radar integration module.

Tests RadarDataset, RecipeSummary, and RadarIntegration classes.
"""

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from datarecipe.integrations.radar import (
    RadarDataset,
    RadarIntegration,
    RecipeSummary,
)

# ---------- Stub objects for optional parameters ----------


@dataclass
class StubAllocation:
    """Mimics HumanMachineAllocation result."""

    total_human_cost: float = 5000.0
    total_machine_cost: float = 200.0
    total_cost: float = 5200.0
    human_work_percentage: float = 70.0
    machine_work_percentage: float = 30.0


@dataclass
class StubRubricsResult:
    """Mimics RubricsAnalysis result."""

    verb_distribution: dict = field(default_factory=dict)
    unique_patterns: int = 0


@dataclass
class StubPromptLibrary:
    """Mimics PromptLibrary result."""

    unique_count: int = 0


@dataclass
class StubComplexityMetrics:
    """Mimics ComplexityMetrics result."""

    primary_domain: object = None
    difficulty_score: float = 2.0


class StubDomain:
    """Mimics an Enum-like domain with .value attribute."""

    def __init__(self, value: str):
        self.value = value


@dataclass
class StubLLMAnalysis:
    """Mimics LLMDatasetAnalysis result."""

    dataset_type: str = ""
    purpose: str = ""
    estimated_difficulty: str = ""
    similar_datasets: list = field(default_factory=list)


# ==================== RadarDataset ====================


class TestRadarDatasetDefaults(unittest.TestCase):
    """Test RadarDataset dataclass defaults."""

    def test_default_values(self):
        ds = RadarDataset(id="test/dataset")
        self.assertEqual(ds.id, "test/dataset")
        self.assertEqual(ds.category, "")
        self.assertEqual(ds.downloads, 0)
        self.assertEqual(ds.signals, [])
        self.assertEqual(ds.source, "huggingface")
        self.assertEqual(ds.discovered_date, "")
        self.assertEqual(ds.org, "")

    def test_custom_values(self):
        ds = RadarDataset(
            id="Anthropic/hh-rlhf",
            category="preference",
            downloads=50000,
            signals=["trending", "quality"],
            source="huggingface",
            discovered_date="2025-01-01",
            org="Anthropic",
        )
        self.assertEqual(ds.id, "Anthropic/hh-rlhf")
        self.assertEqual(ds.category, "preference")
        self.assertEqual(ds.downloads, 50000)
        self.assertEqual(ds.signals, ["trending", "quality"])
        self.assertEqual(ds.org, "Anthropic")


class TestRadarDatasetFromRadarJson(unittest.TestCase):
    """Test RadarDataset.from_radar_json() class method."""

    def test_full_data(self):
        data = {
            "id": "Anthropic/hh-rlhf",
            "category": "preference",
            "downloads": 50000,
            "signals": ["trending"],
            "source": "github",
            "discovered_date": "2025-01-01",
        }
        ds = RadarDataset.from_radar_json(data)
        self.assertEqual(ds.id, "Anthropic/hh-rlhf")
        self.assertEqual(ds.category, "preference")
        self.assertEqual(ds.downloads, 50000)
        self.assertEqual(ds.signals, ["trending"])
        self.assertEqual(ds.source, "github")
        self.assertEqual(ds.discovered_date, "2025-01-01")
        self.assertEqual(ds.org, "Anthropic")

    def test_extracts_org_from_id(self):
        data = {"id": "OpenAI/some-dataset"}
        ds = RadarDataset.from_radar_json(data)
        self.assertEqual(ds.org, "OpenAI")

    def test_no_slash_in_id_gives_empty_org(self):
        data = {"id": "standalone-dataset"}
        ds = RadarDataset.from_radar_json(data)
        self.assertEqual(ds.org, "")

    def test_empty_data(self):
        ds = RadarDataset.from_radar_json({})
        self.assertEqual(ds.id, "")
        self.assertEqual(ds.category, "")
        self.assertEqual(ds.downloads, 0)
        self.assertEqual(ds.signals, [])
        self.assertEqual(ds.source, "huggingface")
        self.assertEqual(ds.org, "")

    def test_missing_fields_use_defaults(self):
        data = {"id": "test/ds", "downloads": 100}
        ds = RadarDataset.from_radar_json(data)
        self.assertEqual(ds.id, "test/ds")
        self.assertEqual(ds.downloads, 100)
        self.assertEqual(ds.category, "")
        self.assertEqual(ds.signals, [])
        self.assertEqual(ds.source, "huggingface")


# ==================== RecipeSummary ====================


class TestRecipeSummaryDefaults(unittest.TestCase):
    """Test RecipeSummary dataclass defaults."""

    def test_default_values(self):
        summary = RecipeSummary(dataset_id="test/dataset")
        self.assertEqual(summary.dataset_id, "test/dataset")
        self.assertEqual(summary.analysis_date, "")
        self.assertEqual(summary.analysis_version, "1.0")
        self.assertEqual(summary.dataset_type, "")
        self.assertEqual(summary.category, "")
        self.assertEqual(summary.purpose, "")
        self.assertEqual(summary.reproduction_cost, {})
        self.assertEqual(summary.difficulty, "")
        self.assertEqual(summary.human_percentage, 0.0)
        self.assertEqual(summary.machine_percentage, 0.0)
        self.assertEqual(summary.key_patterns, [])
        self.assertEqual(summary.rubric_patterns, 0)
        self.assertEqual(summary.prompt_templates, 0)
        self.assertEqual(summary.fields, [])
        self.assertEqual(summary.sample_count, 0)
        self.assertEqual(summary.similar_datasets, [])
        self.assertEqual(summary.report_path, "")
        self.assertEqual(summary.guide_path, "")


class TestRecipeSummaryToDict(unittest.TestCase):
    """Test RecipeSummary.to_dict()."""

    def test_to_dict_contains_all_fields(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=1000,
        )
        d = summary.to_dict()
        self.assertEqual(d["dataset_id"], "test/ds")
        self.assertEqual(d["dataset_type"], "preference")
        self.assertEqual(d["sample_count"], 1000)
        self.assertIn("analysis_date", d)
        self.assertIn("reproduction_cost", d)
        self.assertIn("key_patterns", d)

    def test_to_dict_with_nested_data(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            reproduction_cost={"human": 100.0, "api": 50.0, "total": 150.0},
            key_patterns=["rubric:analyze", "domain:medical"],
            fields=["question", "answer"],
        )
        d = summary.to_dict()
        self.assertEqual(d["reproduction_cost"]["human"], 100.0)
        self.assertEqual(len(d["key_patterns"]), 2)
        self.assertEqual(d["fields"], ["question", "answer"])


class TestRecipeSummaryToJson(unittest.TestCase):
    """Test RecipeSummary.to_json()."""

    def test_to_json_returns_valid_json(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            dataset_type="sft",
            sample_count=500,
        )
        json_str = summary.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["dataset_id"], "test/ds")
        self.assertEqual(parsed["dataset_type"], "sft")

    def test_to_json_with_indent(self):
        summary = RecipeSummary(dataset_id="test/ds")
        json_str = summary.to_json(indent=4)
        # 4-space indent should have more spaces than 2-space
        self.assertIn("    ", json_str)

    def test_to_json_handles_unicode(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            purpose="RLHF 偏好数据",
        )
        json_str = summary.to_json()
        self.assertIn("偏好数据", json_str)


class TestRecipeSummaryFromDict(unittest.TestCase):
    """Test RecipeSummary.from_dict() class method."""

    def test_round_trip(self):
        original = RecipeSummary(
            dataset_id="test/ds",
            dataset_type="preference",
            category="rlhf",
            purpose="Training reward models",
            reproduction_cost={"human": 100.0, "api": 50.0, "total": 150.0},
            difficulty="medium",
            human_percentage=70.0,
            machine_percentage=30.0,
            key_patterns=["rubric:analyze"],
            rubric_patterns=5,
            prompt_templates=3,
            fields=["question", "answer"],
            sample_count=1000,
            similar_datasets=["other/ds"],
            report_path="/path/to/report",
            guide_path="/path/to/guide",
        )
        d = original.to_dict()
        restored = RecipeSummary.from_dict(d)
        self.assertEqual(restored.dataset_id, original.dataset_id)
        self.assertEqual(restored.dataset_type, original.dataset_type)
        self.assertEqual(restored.reproduction_cost, original.reproduction_cost)
        self.assertEqual(restored.key_patterns, original.key_patterns)
        self.assertEqual(restored.sample_count, original.sample_count)

    def test_from_dict_ignores_unknown_keys(self):
        data = {
            "dataset_id": "test/ds",
            "unknown_field": "should be ignored",
            "another_field": 42,
        }
        summary = RecipeSummary.from_dict(data)
        self.assertEqual(summary.dataset_id, "test/ds")
        self.assertFalse(hasattr(summary, "unknown_field"))

    def test_from_dict_empty_dict(self):
        # dataset_id is required positional arg, so from_dict with empty dict
        # should raise TypeError
        with self.assertRaises(TypeError):
            RecipeSummary.from_dict({})

    def test_from_dict_minimal(self):
        summary = RecipeSummary.from_dict({"dataset_id": "test/ds"})
        self.assertEqual(summary.dataset_id, "test/ds")
        self.assertEqual(summary.dataset_type, "")


# ==================== RadarIntegration ====================


class TestRadarIntegrationInit(unittest.TestCase):
    """Test RadarIntegration initialization."""

    def test_init_empty_datasets(self):
        ri = RadarIntegration()
        self.assertEqual(ri.datasets, [])


class TestRadarIntegrationLoadReport(unittest.TestCase):
    """Test RadarIntegration.load_radar_report()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ri = RadarIntegration()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_report_with_datasets(self):
        report = {
            "datasets": [
                {"id": "Anthropic/hh-rlhf", "category": "preference", "downloads": 50000},
                {"id": "OpenAI/sft-data", "category": "sft", "downloads": 10000},
            ]
        }
        path = os.path.join(self.tmpdir, "intel_report_2025-01-01.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f)

        result = self.ri.load_radar_report(path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "Anthropic/hh-rlhf")
        self.assertEqual(result[0].org, "Anthropic")
        self.assertEqual(result[1].id, "OpenAI/sft-data")
        self.assertEqual(self.ri.datasets, result)

    def test_load_report_empty_datasets(self):
        report = {"datasets": []}
        path = os.path.join(self.tmpdir, "empty_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f)

        result = self.ri.load_radar_report(path)
        self.assertEqual(len(result), 0)

    def test_load_report_no_datasets_key(self):
        report = {"metadata": {"date": "2025-01-01"}}
        path = os.path.join(self.tmpdir, "no_datasets.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f)

        result = self.ri.load_radar_report(path)
        self.assertEqual(len(result), 0)

    def test_load_report_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.ri.load_radar_report("/nonexistent/path/report.json")

    def test_load_report_replaces_previous_datasets(self):
        # First load
        report1 = {"datasets": [{"id": "ds1"}, {"id": "ds2"}]}
        path1 = os.path.join(self.tmpdir, "report1.json")
        with open(path1, "w", encoding="utf-8") as f:
            json.dump(report1, f)
        self.ri.load_radar_report(path1)
        self.assertEqual(len(self.ri.datasets), 2)

        # Second load should replace
        report2 = {"datasets": [{"id": "ds3"}]}
        path2 = os.path.join(self.tmpdir, "report2.json")
        with open(path2, "w", encoding="utf-8") as f:
            json.dump(report2, f)
        self.ri.load_radar_report(path2)
        self.assertEqual(len(self.ri.datasets), 1)
        self.assertEqual(self.ri.datasets[0].id, "ds3")


class TestRadarIntegrationFilterDatasets(unittest.TestCase):
    """Test RadarIntegration.filter_datasets()."""

    def setUp(self):
        self.ri = RadarIntegration()
        self.ri.datasets = [
            RadarDataset(id="Anthropic/hh-rlhf", category="preference", downloads=50000,
                         signals=["trending", "quality"], org="Anthropic"),
            RadarDataset(id="OpenAI/sft-data", category="sft", downloads=10000,
                         signals=["quality"], org="OpenAI"),
            RadarDataset(id="Meta/llama-data", category="sft", downloads=30000,
                         signals=["trending"], org="Meta"),
            RadarDataset(id="Anthropic/eval-set", category="evaluation", downloads=5000,
                         signals=["benchmark"], org="Anthropic"),
            RadarDataset(id="small/ds", category="preference", downloads=100,
                         signals=[], org="small"),
        ]

    def test_no_filters_returns_all_sorted_by_downloads(self):
        result = self.ri.filter_datasets()
        self.assertEqual(len(result), 5)
        # Should be sorted by downloads descending
        self.assertEqual(result[0].id, "Anthropic/hh-rlhf")
        self.assertEqual(result[1].id, "Meta/llama-data")
        self.assertEqual(result[2].id, "OpenAI/sft-data")

    def test_filter_by_orgs(self):
        result = self.ri.filter_datasets(orgs=["Anthropic"])
        self.assertEqual(len(result), 2)
        for ds in result:
            self.assertEqual(ds.org, "Anthropic")

    def test_filter_by_orgs_case_insensitive(self):
        result = self.ri.filter_datasets(orgs=["anthropic"])
        self.assertEqual(len(result), 2)

    def test_filter_by_categories(self):
        result = self.ri.filter_datasets(categories=["sft"])
        self.assertEqual(len(result), 2)
        for ds in result:
            self.assertEqual(ds.category, "sft")

    def test_filter_by_categories_case_insensitive(self):
        result = self.ri.filter_datasets(categories=["SFT"])
        self.assertEqual(len(result), 2)

    def test_filter_by_min_downloads(self):
        result = self.ri.filter_datasets(min_downloads=10000)
        self.assertEqual(len(result), 3)
        for ds in result:
            self.assertGreaterEqual(ds.downloads, 10000)

    def test_filter_by_signals(self):
        result = self.ri.filter_datasets(signals=["trending"])
        self.assertEqual(len(result), 2)
        ids = [ds.id for ds in result]
        self.assertIn("Anthropic/hh-rlhf", ids)
        self.assertIn("Meta/llama-data", ids)

    def test_filter_by_signals_case_insensitive(self):
        result = self.ri.filter_datasets(signals=["TRENDING"])
        self.assertEqual(len(result), 2)

    def test_filter_by_signals_intersection(self):
        # Any dataset that has at least one matching signal
        result = self.ri.filter_datasets(signals=["benchmark"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "Anthropic/eval-set")

    def test_filter_with_limit(self):
        result = self.ri.filter_datasets(limit=2)
        self.assertEqual(len(result), 2)
        # Should be top 2 by downloads
        self.assertEqual(result[0].id, "Anthropic/hh-rlhf")
        self.assertEqual(result[1].id, "Meta/llama-data")

    def test_filter_combined_criteria(self):
        result = self.ri.filter_datasets(
            orgs=["Anthropic"],
            categories=["preference"],
            min_downloads=1000,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "Anthropic/hh-rlhf")

    def test_filter_no_matches(self):
        result = self.ri.filter_datasets(orgs=["NonExistentOrg"])
        self.assertEqual(len(result), 0)

    def test_filter_limit_zero_means_no_limit(self):
        result = self.ri.filter_datasets(limit=0)
        self.assertEqual(len(result), 5)

    def test_filter_does_not_modify_original(self):
        self.ri.filter_datasets(orgs=["Anthropic"])
        self.assertEqual(len(self.ri.datasets), 5)

    def test_filter_min_downloads_zero_no_effect(self):
        result = self.ri.filter_datasets(min_downloads=0)
        self.assertEqual(len(result), 5)


class TestRadarIntegrationGetDatasetIds(unittest.TestCase):
    """Test RadarIntegration.get_dataset_ids()."""

    def setUp(self):
        self.ri = RadarIntegration()
        self.ri.datasets = [
            RadarDataset(id="ds1"),
            RadarDataset(id="ds2"),
            RadarDataset(id="ds3"),
        ]

    def test_get_ids_default(self):
        ids = self.ri.get_dataset_ids()
        self.assertEqual(ids, ["ds1", "ds2", "ds3"])

    def test_get_ids_from_provided_list(self):
        custom = [RadarDataset(id="custom1"), RadarDataset(id="custom2")]
        ids = self.ri.get_dataset_ids(datasets=custom)
        self.assertEqual(ids, ["custom1", "custom2"])

    def test_get_ids_empty_list(self):
        self.ri.datasets = []
        ids = self.ri.get_dataset_ids()
        self.assertEqual(ids, [])

    def test_get_ids_with_empty_provided_list(self):
        ids = self.ri.get_dataset_ids(datasets=[])
        self.assertEqual(ids, [])


# ==================== RadarIntegration.create_summary() ====================


class TestCreateSummaryBasic(unittest.TestCase):
    """Test RadarIntegration.create_summary() basic behavior."""

    def test_minimal_call(self):
        summary = RadarIntegration.create_summary(dataset_id="test/ds")
        self.assertIsInstance(summary, RecipeSummary)
        self.assertEqual(summary.dataset_id, "test/ds")
        self.assertNotEqual(summary.analysis_date, "")

    def test_with_dataset_type(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="preference",
        )
        self.assertEqual(summary.dataset_type, "preference")

    def test_sample_count(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            sample_count=5000,
        )
        self.assertEqual(summary.sample_count, 5000)

    def test_analysis_date_is_set(self):
        summary = RadarIntegration.create_summary(dataset_id="test/ds")
        # Should be in YYYY-MM-DD HH:MM format
        self.assertRegex(summary.analysis_date, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}")


class TestCreateSummaryDefaultPurposeAndCategory(unittest.TestCase):
    """Test auto-filling purpose and category from dataset type."""

    def test_preference_type_defaults(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="preference",
        )
        self.assertIn("RLHF", summary.purpose)
        self.assertEqual(summary.category, "rlhf")

    def test_evaluation_type_defaults(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="evaluation",
        )
        self.assertIn("评测", summary.purpose)
        self.assertEqual(summary.category, "benchmark")

    def test_sft_type_defaults(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="sft",
        )
        self.assertIn("微调", summary.purpose)
        self.assertEqual(summary.category, "instruction")

    def test_swe_bench_type_defaults(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="swe_bench",
        )
        self.assertIn("软件工程", summary.purpose)
        self.assertEqual(summary.category, "code")

    def test_instruction_type_defaults(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="instruction",
        )
        self.assertIn("指令", summary.purpose)
        self.assertEqual(summary.category, "instruction")

    def test_chat_type_defaults(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="chat",
        )
        self.assertIn("对话", summary.purpose)
        self.assertEqual(summary.category, "conversation")

    def test_unknown_type_gets_fallback_purpose(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="unknown",
        )
        self.assertIn("通用", summary.purpose)

    def test_explicit_purpose_not_overridden(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="preference",
            purpose="My custom purpose",
        )
        self.assertEqual(summary.purpose, "My custom purpose")

    def test_explicit_category_not_overridden(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="preference",
            category="my_category",
        )
        self.assertEqual(summary.category, "my_category")

    def test_no_type_no_defaults(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="",
        )
        self.assertEqual(summary.purpose, "")
        self.assertEqual(summary.category, "")


class TestCreateSummaryWithAllocation(unittest.TestCase):
    """Test create_summary() with allocation data."""

    def test_allocation_populates_costs(self):
        alloc = StubAllocation(
            total_human_cost=5000.123,
            total_machine_cost=200.456,
            total_cost=5200.579,
            human_work_percentage=70.35,
            machine_work_percentage=29.65,
        )
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            allocation=alloc,
        )
        self.assertEqual(summary.reproduction_cost["human"], 5000.12)
        self.assertEqual(summary.reproduction_cost["api"], 200.46)
        self.assertEqual(summary.reproduction_cost["total"], 5200.58)
        self.assertEqual(summary.human_percentage, 70.3)
        self.assertEqual(summary.machine_percentage, 29.6)

    def test_no_allocation_empty_costs(self):
        summary = RadarIntegration.create_summary(dataset_id="test/ds")
        self.assertEqual(summary.reproduction_cost, {})
        self.assertEqual(summary.human_percentage, 0.0)
        self.assertEqual(summary.machine_percentage, 0.0)


class TestCreateSummaryWithRubrics(unittest.TestCase):
    """Test create_summary() with rubrics data."""

    def test_rubrics_unique_patterns(self):
        rubrics = StubRubricsResult(unique_patterns=15)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            rubrics_result=rubrics,
        )
        self.assertEqual(summary.rubric_patterns, 15)

    def test_rubrics_verb_distribution_top_5(self):
        rubrics = StubRubricsResult(
            unique_patterns=10,
            verb_distribution={
                "analyze": 20,
                "evaluate": 15,
                "judge": 10,
                "compare": 8,
                "classify": 5,
                "sort": 2,  # Should be excluded from top 5
            },
        )
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            rubrics_result=rubrics,
        )
        self.assertEqual(len(summary.key_patterns), 5)
        self.assertIn("rubric:analyze", summary.key_patterns)
        self.assertIn("rubric:evaluate", summary.key_patterns)
        self.assertNotIn("rubric:sort", summary.key_patterns)

    def test_rubrics_no_verb_distribution(self):
        rubrics = StubRubricsResult(unique_patterns=5)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            rubrics_result=rubrics,
        )
        self.assertEqual(summary.key_patterns, [])


class TestCreateSummaryWithPromptLibrary(unittest.TestCase):
    """Test create_summary() with prompt library data."""

    def test_prompt_templates_count(self):
        prompts = StubPromptLibrary(unique_count=12)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            prompt_library=prompts,
        )
        self.assertEqual(summary.prompt_templates, 12)

    def test_no_prompt_library(self):
        summary = RadarIntegration.create_summary(dataset_id="test/ds")
        self.assertEqual(summary.prompt_templates, 0)


class TestCreateSummaryWithSchemaInfo(unittest.TestCase):
    """Test create_summary() with schema info."""

    def test_schema_fields_populated(self):
        schema = {"question": {"type": "str"}, "answer": {"type": "str"}, "score": {"type": "int"}}
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            schema_info=schema,
        )
        self.assertEqual(set(summary.fields), {"question", "answer", "score"})

    def test_no_schema_info(self):
        summary = RadarIntegration.create_summary(dataset_id="test/ds")
        self.assertEqual(summary.fields, [])


class TestCreateSummaryWithComplexityMetrics(unittest.TestCase):
    """Test create_summary() with complexity metrics."""

    def test_easy_difficulty(self):
        metrics = StubComplexityMetrics(difficulty_score=1.0)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertEqual(summary.difficulty, "easy")

    def test_medium_difficulty(self):
        metrics = StubComplexityMetrics(difficulty_score=2.0)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertEqual(summary.difficulty, "medium")

    def test_hard_difficulty(self):
        metrics = StubComplexityMetrics(difficulty_score=3.0)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertEqual(summary.difficulty, "hard")

    def test_expert_difficulty(self):
        metrics = StubComplexityMetrics(difficulty_score=4.0)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertEqual(summary.difficulty, "expert")

    def test_boundary_easy_medium(self):
        metrics = StubComplexityMetrics(difficulty_score=1.5)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertEqual(summary.difficulty, "easy")

    def test_boundary_medium_hard(self):
        metrics = StubComplexityMetrics(difficulty_score=2.5)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertEqual(summary.difficulty, "medium")

    def test_boundary_hard_expert(self):
        metrics = StubComplexityMetrics(difficulty_score=3.5)
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertEqual(summary.difficulty, "hard")

    def test_domain_added_to_key_patterns(self):
        metrics = StubComplexityMetrics(
            difficulty_score=2.0,
            primary_domain=StubDomain("medical"),
        )
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertIn("domain:medical", summary.key_patterns)

    def test_no_domain_no_key_pattern(self):
        metrics = StubComplexityMetrics(
            difficulty_score=2.0,
            primary_domain=None,
        )
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        domain_patterns = [p for p in summary.key_patterns if p.startswith("domain:")]
        self.assertEqual(len(domain_patterns), 0)

    def test_domain_as_string(self):
        """Test domain that is a plain string (no .value attribute)."""
        metrics = StubComplexityMetrics(
            difficulty_score=2.0,
            primary_domain="coding",
        )
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertIn("domain:coding", summary.key_patterns)


class TestCreateSummaryWithLLMAnalysis(unittest.TestCase):
    """Test create_summary() with LLM analysis enrichment."""

    def test_llm_fills_missing_dataset_type(self):
        llm = StubLLMAnalysis(dataset_type="preference")
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            llm_analysis=llm,
        )
        self.assertEqual(summary.dataset_type, "preference")

    def test_llm_does_not_override_existing_dataset_type(self):
        llm = StubLLMAnalysis(dataset_type="sft")
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="preference",
            llm_analysis=llm,
        )
        self.assertEqual(summary.dataset_type, "preference")

    def test_llm_fills_missing_purpose(self):
        llm = StubLLMAnalysis(purpose="LLM-detected purpose")
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            llm_analysis=llm,
        )
        self.assertEqual(summary.purpose, "LLM-detected purpose")

    def test_llm_overrides_difficulty(self):
        metrics = StubComplexityMetrics(difficulty_score=2.0)
        llm = StubLLMAnalysis(estimated_difficulty="hard because reasons")
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
            llm_analysis=llm,
        )
        # LLM difficulty overrides complexity-based; takes first word
        self.assertEqual(summary.difficulty, "hard")

    def test_llm_similar_datasets(self):
        llm = StubLLMAnalysis(similar_datasets=["ds1", "ds2", "ds3"])
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            llm_analysis=llm,
        )
        self.assertEqual(summary.similar_datasets, ["ds1", "ds2", "ds3"])

    def test_llm_empty_difficulty_no_override(self):
        metrics = StubComplexityMetrics(difficulty_score=2.0)
        llm = StubLLMAnalysis(estimated_difficulty="")
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            complexity_metrics=metrics,
            llm_analysis=llm,
        )
        self.assertEqual(summary.difficulty, "medium")


class TestCreateSummaryOutputPaths(unittest.TestCase):
    """Test create_summary() output path generation."""

    def test_output_dir_sets_paths(self):
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            output_dir="/some/output/dir",
        )
        self.assertEqual(summary.report_path, "/some/output/dir/ANALYSIS_REPORT.md")
        self.assertEqual(summary.guide_path, "/some/output/dir/REPRODUCTION_GUIDE.md")

    def test_no_output_dir_empty_paths(self):
        summary = RadarIntegration.create_summary(dataset_id="test/ds")
        self.assertEqual(summary.report_path, "")
        self.assertEqual(summary.guide_path, "")


class TestCreateSummarySimilarDatasetsFallback(unittest.TestCase):
    """Test create_summary() similar datasets fallback logic."""

    def test_llm_provided_similar_datasets_used(self):
        """When LLM provides similar datasets, they should be used directly."""
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="preference",
            llm_analysis=StubLLMAnalysis(similar_datasets=["known/ds"]),
        )
        # LLM-provided similar datasets should be used
        self.assertEqual(summary.similar_datasets, ["known/ds"])

    def test_no_llm_similar_datasets_uses_knowledge_fallback(self):
        """Without LLM similar datasets, knowledge base fallback is attempted."""
        # The knowledge imports are inside try/except blocks, so even if they
        # fail (ImportError, etc.), the code handles it gracefully
        summary = RadarIntegration.create_summary(
            dataset_id="test/ds",
            dataset_type="preference",
        )
        # Should not crash; similar_datasets is a list (possibly populated by knowledge base)
        self.assertIsInstance(summary.similar_datasets, list)


class TestCreateSummaryCombined(unittest.TestCase):
    """Test create_summary() with all parameters combined."""

    def test_full_summary(self):
        alloc = StubAllocation()
        rubrics = StubRubricsResult(
            unique_patterns=10,
            verb_distribution={"analyze": 20, "evaluate": 15},
        )
        prompts = StubPromptLibrary(unique_count=8)
        schema = {"question": {"type": "str"}, "answer": {"type": "str"}}
        metrics = StubComplexityMetrics(
            difficulty_score=2.5,
            primary_domain=StubDomain("legal"),
        )

        summary = RadarIntegration.create_summary(
            dataset_id="org/full-test",
            dataset_type="evaluation",
            category="benchmark",
            purpose="Custom purpose",
            allocation=alloc,
            rubrics_result=rubrics,
            prompt_library=prompts,
            schema_info=schema,
            sample_count=3000,
            output_dir="/output",
            complexity_metrics=metrics,
        )

        self.assertEqual(summary.dataset_id, "org/full-test")
        self.assertEqual(summary.dataset_type, "evaluation")
        self.assertEqual(summary.category, "benchmark")
        self.assertEqual(summary.purpose, "Custom purpose")
        self.assertEqual(summary.sample_count, 3000)
        self.assertEqual(summary.reproduction_cost["human"], 5000.0)
        self.assertEqual(summary.rubric_patterns, 10)
        self.assertEqual(summary.prompt_templates, 8)
        self.assertEqual(set(summary.fields), {"question", "answer"})
        # medium because 2.5 <= 2.5
        self.assertEqual(summary.difficulty, "medium")
        self.assertIn("rubric:analyze", summary.key_patterns)
        self.assertIn("domain:legal", summary.key_patterns)
        self.assertIn("ANALYSIS_REPORT.md", summary.report_path)


# ==================== RadarIntegration.save_summary() / load_summary() ====================


class TestSaveAndLoadSummary(unittest.TestCase):
    """Test RadarIntegration.save_summary() and load_summary()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_creates_file(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            dataset_type="preference",
            sample_count=1000,
        )
        path = RadarIntegration.save_summary(summary, self.tmpdir)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(os.path.basename(path), "recipe_summary.json")

    def test_save_returns_correct_path(self):
        summary = RecipeSummary(dataset_id="test/ds")
        path = RadarIntegration.save_summary(summary, self.tmpdir)
        expected = os.path.join(self.tmpdir, "recipe_summary.json")
        self.assertEqual(path, expected)

    def test_save_creates_directory(self):
        nested_dir = os.path.join(self.tmpdir, "nested", "dir")
        summary = RecipeSummary(dataset_id="test/ds")
        path = RadarIntegration.save_summary(summary, nested_dir)
        self.assertTrue(os.path.exists(path))

    def test_save_content_is_valid_json(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            reproduction_cost={"human": 100.0, "api": 50.0, "total": 150.0},
        )
        path = RadarIntegration.save_summary(summary, self.tmpdir)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["dataset_id"], "test/ds")
        self.assertEqual(data["reproduction_cost"]["human"], 100.0)

    def test_save_handles_unicode(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            purpose="RLHF 偏好数据",
        )
        path = RadarIntegration.save_summary(summary, self.tmpdir)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("偏好数据", content)

    def test_load_summary(self):
        summary = RecipeSummary(
            dataset_id="test/ds",
            dataset_type="sft",
            sample_count=2000,
            difficulty="medium",
            key_patterns=["rubric:analyze"],
        )
        path = RadarIntegration.save_summary(summary, self.tmpdir)
        loaded = RadarIntegration.load_summary(path)
        self.assertEqual(loaded.dataset_id, "test/ds")
        self.assertEqual(loaded.dataset_type, "sft")
        self.assertEqual(loaded.sample_count, 2000)
        self.assertEqual(loaded.difficulty, "medium")
        self.assertEqual(loaded.key_patterns, ["rubric:analyze"])

    def test_round_trip_full_summary(self):
        original = RecipeSummary(
            dataset_id="org/ds",
            analysis_date="2025-01-01 10:00",
            analysis_version="1.0",
            dataset_type="preference",
            category="rlhf",
            purpose="Training",
            reproduction_cost={"human": 5000.0, "api": 200.0, "total": 5200.0},
            difficulty="hard",
            human_percentage=70.0,
            machine_percentage=30.0,
            key_patterns=["rubric:analyze", "domain:medical"],
            rubric_patterns=15,
            prompt_templates=8,
            fields=["question", "answer"],
            sample_count=10000,
            similar_datasets=["other/ds1", "other/ds2"],
            report_path="/path/to/report.md",
            guide_path="/path/to/guide.md",
        )
        path = RadarIntegration.save_summary(original, self.tmpdir)
        loaded = RadarIntegration.load_summary(path)

        self.assertEqual(loaded.dataset_id, original.dataset_id)
        self.assertEqual(loaded.analysis_date, original.analysis_date)
        self.assertEqual(loaded.dataset_type, original.dataset_type)
        self.assertEqual(loaded.category, original.category)
        self.assertEqual(loaded.reproduction_cost, original.reproduction_cost)
        self.assertEqual(loaded.difficulty, original.difficulty)
        self.assertEqual(loaded.human_percentage, original.human_percentage)
        self.assertEqual(loaded.key_patterns, original.key_patterns)
        self.assertEqual(loaded.rubric_patterns, original.rubric_patterns)
        self.assertEqual(loaded.prompt_templates, original.prompt_templates)
        self.assertEqual(loaded.fields, original.fields)
        self.assertEqual(loaded.sample_count, original.sample_count)
        self.assertEqual(loaded.similar_datasets, original.similar_datasets)

    def test_save_overwrites_existing_file(self):
        summary1 = RecipeSummary(dataset_id="first/ds")
        RadarIntegration.save_summary(summary1, self.tmpdir)

        summary2 = RecipeSummary(dataset_id="second/ds")
        path = RadarIntegration.save_summary(summary2, self.tmpdir)

        loaded = RadarIntegration.load_summary(path)
        self.assertEqual(loaded.dataset_id, "second/ds")

    def test_load_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            RadarIntegration.load_summary("/nonexistent/path/summary.json")


# ==================== RadarIntegration.aggregate_summaries() ====================


class TestAggregateSummaries(unittest.TestCase):
    """Test RadarIntegration.aggregate_summaries()."""

    def test_empty_list_returns_empty_dict(self):
        result = RadarIntegration.aggregate_summaries([])
        self.assertEqual(result, {})

    def test_single_summary(self):
        summaries = [
            RecipeSummary(
                dataset_id="ds1",
                dataset_type="preference",
                difficulty="medium",
                reproduction_cost={"human": 1000.0, "api": 200.0},
                human_percentage=70.0,
            )
        ]
        result = RadarIntegration.aggregate_summaries(summaries)
        self.assertEqual(result["total_datasets"], 1)
        self.assertEqual(result["total_reproduction_cost"]["human"], 1000.0)
        self.assertEqual(result["total_reproduction_cost"]["api"], 200.0)
        self.assertEqual(result["total_reproduction_cost"]["total"], 1200.0)
        self.assertEqual(result["avg_human_percentage"], 70.0)
        self.assertEqual(result["type_distribution"], {"preference": 1})
        self.assertEqual(result["difficulty_distribution"], {"medium": 1})
        self.assertEqual(result["datasets"], ["ds1"])

    def test_multiple_summaries(self):
        summaries = [
            RecipeSummary(
                dataset_id="ds1",
                dataset_type="preference",
                difficulty="easy",
                reproduction_cost={"human": 1000.0, "api": 200.0},
                human_percentage=60.0,
            ),
            RecipeSummary(
                dataset_id="ds2",
                dataset_type="sft",
                difficulty="medium",
                reproduction_cost={"human": 2000.0, "api": 300.0},
                human_percentage=80.0,
            ),
            RecipeSummary(
                dataset_id="ds3",
                dataset_type="preference",
                difficulty="easy",
                reproduction_cost={"human": 500.0, "api": 100.0},
                human_percentage=50.0,
            ),
        ]
        result = RadarIntegration.aggregate_summaries(summaries)
        self.assertEqual(result["total_datasets"], 3)
        self.assertEqual(result["total_reproduction_cost"]["human"], 3500.0)
        self.assertEqual(result["total_reproduction_cost"]["api"], 600.0)
        self.assertEqual(result["total_reproduction_cost"]["total"], 4100.0)
        self.assertAlmostEqual(result["avg_human_percentage"], 63.3, places=1)
        self.assertEqual(result["type_distribution"], {"preference": 2, "sft": 1})
        self.assertEqual(result["difficulty_distribution"], {"easy": 2, "medium": 1})
        self.assertEqual(result["datasets"], ["ds1", "ds2", "ds3"])

    def test_missing_cost_keys_default_to_zero(self):
        summaries = [
            RecipeSummary(
                dataset_id="ds1",
                reproduction_cost={},  # No human or api keys
            ),
        ]
        result = RadarIntegration.aggregate_summaries(summaries)
        self.assertEqual(result["total_reproduction_cost"]["human"], 0)
        self.assertEqual(result["total_reproduction_cost"]["api"], 0)
        self.assertEqual(result["total_reproduction_cost"]["total"], 0)

    def test_empty_type_and_difficulty_excluded(self):
        summaries = [
            RecipeSummary(
                dataset_id="ds1",
                dataset_type="",
                difficulty="",
                reproduction_cost={},
            ),
        ]
        result = RadarIntegration.aggregate_summaries(summaries)
        self.assertEqual(result["type_distribution"], {})
        self.assertEqual(result["difficulty_distribution"], {})

    def test_rounding(self):
        summaries = [
            RecipeSummary(
                dataset_id="ds1",
                reproduction_cost={"human": 1000.119, "api": 200.229},
                human_percentage=33.3,
            ),
            RecipeSummary(
                dataset_id="ds2",
                reproduction_cost={"human": 2000.449, "api": 300.559},
                human_percentage=66.7,
            ),
        ]
        result = RadarIntegration.aggregate_summaries(summaries)
        # Verify rounding to 2 decimal places for costs
        self.assertEqual(result["total_reproduction_cost"]["human"], round(1000.119 + 2000.449, 2))
        self.assertEqual(result["total_reproduction_cost"]["api"], round(200.229 + 300.559, 2))
        # Average human percentage rounded to 1 decimal
        self.assertEqual(result["avg_human_percentage"], 50.0)


# ==================== Class-level constants ====================


class TestRadarIntegrationConstants(unittest.TestCase):
    """Test that DATASET_TYPE_PURPOSES and DATASET_TYPE_CATEGORIES are consistent."""

    def test_purposes_has_expected_keys(self):
        expected = {"preference", "evaluation", "sft", "swe_bench", "instruction", "chat", "unknown"}
        self.assertEqual(set(RadarIntegration.DATASET_TYPE_PURPOSES.keys()), expected)

    def test_categories_has_expected_keys(self):
        expected = {"preference", "evaluation", "sft", "swe_bench", "instruction", "chat"}
        self.assertEqual(set(RadarIntegration.DATASET_TYPE_CATEGORIES.keys()), expected)

    def test_purposes_are_non_empty_strings(self):
        for key, value in RadarIntegration.DATASET_TYPE_PURPOSES.items():
            self.assertIsInstance(value, str, f"Purpose for {key} should be string")
            self.assertTrue(len(value) > 0, f"Purpose for {key} should not be empty")

    def test_categories_are_non_empty_strings(self):
        for key, value in RadarIntegration.DATASET_TYPE_CATEGORIES.items():
            self.assertIsInstance(value, str, f"Category for {key} should be string")
            self.assertTrue(len(value) > 0, f"Category for {key} should not be empty")


if __name__ == "__main__":
    unittest.main()
