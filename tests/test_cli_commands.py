"""CLI integration tests using Click's CliRunner.

Tests cover:
- Main CLI group (--help, --version)
- All command --help outputs
- Helper functions (validate_output_path, recipe_to_markdown, display_recipe)
- Basic invocations with mocked external dependencies

Patch targets follow the rule:
- Top-level imports (e.g. DatasetAnalyzer in analyze.py) -> patch on the cli module
- Lazy imports inside functions (e.g. CostCalculator) -> patch on the source module
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from datarecipe.cli import main, recipe_to_markdown, validate_output_path
from datarecipe.schema import (
    Cost,
    GenerationMethod,
    GenerationType,
    Recipe,
    Reproducibility,
    SourceType,
)

# ==================== Helpers ====================


def _make_recipe(**overrides) -> Recipe:
    """Build a Recipe with sensible defaults; override any field via kwargs."""
    defaults = {
        "name": "test/dataset",
        "version": "1.0",
        "source_type": SourceType.HUGGINGFACE,
        "source_id": "test/dataset",
        "num_examples": 1000,
        "languages": ["en"],
        "license": "MIT",
        "description": "A test dataset",
        "generation_type": GenerationType.MIXED,
        "synthetic_ratio": 0.7,
        "human_ratio": 0.3,
        "teacher_models": ["gpt-4"],
        "generation_methods": [
            GenerationMethod(
                method_type="distillation",
                teacher_model="gpt-4",
                prompt_template_available=True,
            ),
        ],
        "cost": Cost(
            estimated_total_usd=5000.0,
            api_calls_usd=3000.0,
            human_annotation_usd=2000.0,
            compute_usd=0.0,
            confidence="medium",
        ),
        "reproducibility": Reproducibility(
            score=7,
            available=["description", "teacher_model_names", "prompt_templates"],
            missing=["exact_prompts", "filtering_criteria"],
            notes="Reasonably reproducible.",
        ),
    }
    defaults.update(overrides)
    return Recipe(**defaults)


# ==================== Main Group & Version ====================


class TestMainGroup(unittest.TestCase):
    """Test the top-level CLI group."""

    def setUp(self):
        self.runner = CliRunner()

    def test_main_help(self):
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("DataRecipe", result.output)
        self.assertIn("--version", result.output)
        self.assertIn("--help", result.output)

    def test_main_version(self):
        result = self.runner.invoke(main, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("0.3.0", result.output)

    def test_main_no_args_shows_usage(self):
        result = self.runner.invoke(main, [])
        # Click group with no subcommand returns exit_code 0 or 2 depending on
        # invoke_without_command setting. Here it exits 0 but shows usage text.
        self.assertIn("Usage", result.output)


# ==================== Command --help Tests ====================


class TestCommandHelp(unittest.TestCase):
    """Every registered command should respond to --help with exit code 0."""

    def setUp(self):
        self.runner = CliRunner()

    def _check_help(self, cmd_name: str, expect_in_output: str = None):
        result = self.runner.invoke(main, [cmd_name, "--help"])
        self.assertEqual(result.exit_code, 0, f"{cmd_name} --help failed: {result.output}")
        self.assertIn("Usage", result.output)
        if expect_in_output:
            self.assertIn(expect_in_output, result.output)

    # analyze.py commands
    def test_analyze_help(self):
        self._check_help("analyze", "DATASET_ID")

    def test_show_help(self):
        self._check_help("show", "RECIPE_FILE")

    def test_export_help(self):
        self._check_help("export", "OUTPUT_FILE")

    def test_list_sources_help(self):
        self._check_help("list-sources")

    def test_guide_help(self):
        self._check_help("guide", "production guide")

    def test_deep_guide_help(self):
        self._check_help("deep-guide", "URL")

    # tools.py commands
    def test_create_help(self):
        self._check_help("create", "recipe")

    def test_cost_help(self):
        self._check_help("cost", "DATASET_ID")

    def test_quality_help(self):
        self._check_help("quality", "DATASET_ID")

    def test_batch_help(self):
        self._check_help("batch", "DATASET_IDS")

    def test_compare_help(self):
        self._check_help("compare", "DATASET_IDS")

    def test_profile_help(self):
        self._check_help("profile", "DATASET_ID")

    def test_deploy_help(self):
        self._check_help("deploy", "DATASET_ID")

    def test_providers_help(self):
        self._check_help("providers")

    def test_workflow_help(self):
        self._check_help("workflow", "DATASET_ID")

    def test_extract_rubrics_help(self):
        self._check_help("extract-rubrics", "DATASET_ID")

    def test_extract_prompts_help(self):
        self._check_help("extract-prompts", "DATASET_ID")

    def test_detect_strategy_help(self):
        self._check_help("detect-strategy", "DATASET_ID")

    def test_allocate_help(self):
        self._check_help("allocate")

    def test_enhanced_guide_help(self):
        self._check_help("enhanced-guide", "DATASET_ID")

    def test_generate_help(self):
        self._check_help("generate")

    # deep.py command
    def test_deep_analyze_help(self):
        self._check_help("deep-analyze", "DATASET_ID")

    # batch.py commands
    def test_batch_from_radar_help(self):
        self._check_help("batch-from-radar", "RADAR_REPORT")

    def test_integrate_report_help(self):
        self._check_help("integrate-report")

    # infra.py commands
    def test_watch_help(self):
        self._check_help("watch", "WATCH_DIR")

    def test_cache_help(self):
        self._check_help("cache")

    def test_knowledge_help(self):
        self._check_help("knowledge")

    # spec.py command
    def test_analyze_spec_help(self):
        self._check_help("analyze-spec")


# ==================== Helper Functions ====================


class TestValidateOutputPath(unittest.TestCase):
    """Test validate_output_path security utility."""

    def test_normal_path(self):
        result = validate_output_path("/tmp/datarecipe_output")
        self.assertIsInstance(result, Path)
        self.assertTrue(result.is_absolute())

    def test_relative_resolves_to_absolute(self):
        result = validate_output_path("output/test")
        self.assertTrue(result.is_absolute())

    def test_base_dir_within(self):
        result = validate_output_path("/tmp/output/sub", base_dir=Path("/tmp/output"))
        self.assertIsInstance(result, Path)

    def test_base_dir_outside_raises(self):
        with self.assertRaises(ValueError) as ctx:
            validate_output_path("/other/path", base_dir=Path("/tmp/output"))
        self.assertIn("outside allowed directory", str(ctx.exception))

    def test_dangerous_etc_or_var(self):
        # On macOS, /etc -> /private/etc and /var -> /private/var.
        # The blocked patterns are literal prefixes checked against the resolved path.
        # So we test /usr/ which does NOT have a macOS symlink redirect.
        with self.assertRaises(ValueError):
            validate_output_path("/usr/share/something")

    def test_dangerous_usr(self):
        with self.assertRaises(ValueError):
            validate_output_path("/usr/local/share/data")

    def test_dangerous_root(self):
        with self.assertRaises(ValueError):
            validate_output_path("/root/.config")

    def test_home_dir_allowed(self):
        result = validate_output_path("/Users/testuser/output")
        self.assertIsInstance(result, Path)

    def test_tmp_dir_allowed(self):
        result = validate_output_path("/tmp/some/nested/path")
        self.assertIsInstance(result, Path)


class TestRecipeToMarkdown(unittest.TestCase):
    """Test recipe_to_markdown helper."""

    def test_basic_output(self):
        recipe = _make_recipe()
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)
        self.assertIn("MIT", md)
        self.assertIn("gpt-4", md)
        self.assertIn("1,000", md)  # num_examples formatted

    def test_contains_cost_section(self):
        recipe = _make_recipe()
        md = recipe_to_markdown(recipe)
        self.assertIn("$", md)
        self.assertIn("API", md)

    def test_contains_reproducibility_section(self):
        recipe = _make_recipe()
        md = recipe_to_markdown(recipe)
        # The reproducibility section header contains the score
        self.assertIn("7/10", md)

    def test_no_cost(self):
        recipe = _make_recipe(cost=None)
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)

    def test_no_reproducibility(self):
        recipe = _make_recipe(reproducibility=None)
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)

    def test_no_teacher_models(self):
        recipe = _make_recipe(teacher_models=[])
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)

    def test_no_generation_methods(self):
        recipe = _make_recipe(generation_methods=[])
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)

    def test_no_ratios(self):
        recipe = _make_recipe(synthetic_ratio=None, human_ratio=None)
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)

    def test_low_confidence_cost_range(self):
        recipe = _make_recipe(
            cost=Cost(estimated_total_usd=1000.0, confidence="low")
        )
        md = recipe_to_markdown(recipe)
        # Low confidence shows a range: $500 - $1,500
        self.assertIn("500", md)
        self.assertIn("1,500", md)


class TestDisplayRecipe(unittest.TestCase):
    """Test display_recipe helper (console output)."""

    def test_display_recipe_runs_without_error(self):
        from datarecipe.cli._helpers import display_recipe

        recipe = _make_recipe()
        # display_recipe prints to console; just verify it doesn't raise
        try:
            display_recipe(recipe)
        except Exception as e:
            self.fail(f"display_recipe raised {e}")

    def test_display_recipe_no_cost(self):
        from datarecipe.cli._helpers import display_recipe

        recipe = _make_recipe(cost=None)
        try:
            display_recipe(recipe)
        except Exception as e:
            self.fail(f"display_recipe raised {e}")

    def test_display_recipe_unknown_generation(self):
        from datarecipe.cli._helpers import display_recipe

        recipe = _make_recipe(
            generation_type=GenerationType.UNKNOWN,
            synthetic_ratio=None,
            human_ratio=None,
        )
        try:
            display_recipe(recipe)
        except Exception as e:
            self.fail(f"display_recipe raised {e}")


# ==================== analyze.py Command Tests ====================


class TestAnalyzeCommand(unittest.TestCase):
    """Test the `analyze` command with mocked analyzer."""

    def setUp(self):
        self.runner = CliRunner()
        self.recipe = _make_recipe()

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_default_output(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.return_value = self.recipe
        result = self.runner.invoke(main, ["analyze", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        mock_instance.analyze.assert_called_once_with("test/dataset")

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_json_flag(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.return_value = self.recipe
        result = self.runner.invoke(main, ["analyze", "test/dataset", "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('"name"', result.output)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_yaml_flag(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.return_value = self.recipe
        result = self.runner.invoke(main, ["analyze", "test/dataset", "--yaml"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("name:", result.output)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_markdown_flag(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.return_value = self.recipe
        result = self.runner.invoke(main, ["analyze", "test/dataset", "--markdown"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("test/dataset", result.output)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_output_json_file(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.return_value = self.recipe
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "recipe.json")
            result = self.runner.invoke(main, ["analyze", "test/dataset", "-o", outpath])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))
            data = json.loads(Path(outpath).read_text())
            self.assertEqual(data["name"], "test/dataset")

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_output_md_file(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.return_value = self.recipe
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "recipe.md")
            result = self.runner.invoke(main, ["analyze", "test/dataset", "-o", outpath])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))
            content = Path(outpath).read_text()
            self.assertIn("test/dataset", content)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_value_error(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.side_effect = ValueError("Dataset not found")
        result = self.runner.invoke(main, ["analyze", "bad/dataset"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_generic_error(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.side_effect = RuntimeError("Network error")
        result = self.runner.invoke(main, ["analyze", "bad/dataset"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)


class TestListSourcesCommand(unittest.TestCase):
    """Test the `list-sources` command."""

    def setUp(self):
        self.runner = CliRunner()

    def test_list_sources(self):
        result = self.runner.invoke(main, ["list-sources"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("HuggingFace", result.output)
        self.assertIn("GitHub", result.output)
        self.assertIn("Local", result.output)


class TestShowCommand(unittest.TestCase):
    """Test the `show` command."""

    def setUp(self):
        self.runner = CliRunner()

    def test_show_missing_file(self):
        result = self.runner.invoke(main, ["show", "/nonexistent/recipe.yaml"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_show_valid_file(self, MockAnalyzer):
        recipe = _make_recipe()
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze_from_yaml.return_value = recipe
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: test\n")
            tmppath = f.name
        try:
            result = self.runner.invoke(main, ["show", tmppath])
            self.assertEqual(result.exit_code, 0)
            mock_instance.analyze_from_yaml.assert_called_once_with(tmppath)
        finally:
            os.unlink(tmppath)


class TestExportCommand(unittest.TestCase):
    """Test the `export` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_export_success(self, MockAnalyzer):
        recipe = _make_recipe()
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.return_value = recipe
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "recipe.yaml")
            result = self.runner.invoke(main, ["export", "test/dataset", outpath])
            self.assertEqual(result.exit_code, 0)
            mock_instance.export_recipe.assert_called_once()

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_export_analyze_error(self, MockAnalyzer):
        mock_instance = MockAnalyzer.return_value
        mock_instance.analyze.side_effect = ValueError("not found")
        result = self.runner.invoke(main, ["export", "bad/dataset", "/tmp/out.yaml"])
        self.assertNotEqual(result.exit_code, 0)


# ==================== tools.py Command Tests ====================
#
# Tools.py has two import styles:
# - DatasetAnalyzer is top-level -> patch "datarecipe.cli.tools.DatasetAnalyzer"
# - CostCalculator, QualityAnalyzer, etc are lazy inside functions
#   -> patch the source module "datarecipe.cost_calculator.CostCalculator"
#


class TestCostCommand(unittest.TestCase):
    """Test the `cost` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.cost_calculator.CostCalculator")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_cost_default(self, MockAnalyzer, MockCalc):
        recipe = _make_recipe()
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_breakdown = MagicMock()
        mock_breakdown.api_cost.low = 100
        mock_breakdown.api_cost.expected = 200
        mock_breakdown.api_cost.high = 300
        mock_breakdown.human_annotation_cost.low = 50
        mock_breakdown.human_annotation_cost.expected = 100
        mock_breakdown.human_annotation_cost.high = 150
        mock_breakdown.compute_cost.low = 10
        mock_breakdown.compute_cost.expected = 20
        mock_breakdown.compute_cost.high = 30
        mock_breakdown.total.low = 160
        mock_breakdown.total.expected = 320
        mock_breakdown.total.high = 480
        mock_breakdown.assumptions = ["Assumes 500 tokens per example"]
        MockCalc.return_value.estimate_from_recipe.return_value = mock_breakdown

        result = self.runner.invoke(main, ["cost", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Cost", result.output)

    @patch("datarecipe.cost_calculator.CostCalculator")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_cost_json_flag(self, MockAnalyzer, MockCalc):
        recipe = _make_recipe()
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_breakdown = MagicMock()
        mock_breakdown.to_dict.return_value = {"total": 320}
        MockCalc.return_value.estimate_from_recipe.return_value = mock_breakdown

        result = self.runner.invoke(main, ["cost", "test/dataset", "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("total", result.output)


class TestCompareCommand(unittest.TestCase):
    """Test the `compare` command."""

    def setUp(self):
        self.runner = CliRunner()

    def test_compare_requires_two_datasets(self):
        result = self.runner.invoke(main, ["compare", "single/dataset"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.comparator.DatasetComparator")
    def test_compare_success(self, MockComparator):
        mock_report = MagicMock()
        mock_report.to_table.return_value = "| Feature | ds1 | ds2 |\n"
        mock_report.recommendations = ["Use ds1 for X"]
        MockComparator.return_value.compare_by_ids.return_value = mock_report

        result = self.runner.invoke(main, ["compare", "ds1", "ds2"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.comparator.DatasetComparator")
    def test_compare_markdown_format(self, MockComparator):
        mock_report = MagicMock()
        mock_report.to_markdown.return_value = "# Comparison\n"
        MockComparator.return_value.compare_by_ids.return_value = mock_report

        result = self.runner.invoke(main, ["compare", "ds1", "ds2", "--format", "markdown"])
        self.assertEqual(result.exit_code, 0)


class TestAllocateCommand(unittest.TestCase):
    """Test the `allocate` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    def test_allocate_default(self, MockTaskType, MockSplitter):
        # TaskType enum members must be accessible
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.DATA_GENERATION = "DATA_GENERATION"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        mock_result = MagicMock()
        mock_result.summary.return_value = "Summary text"
        mock_result.to_markdown_table.return_value = "| Task | ... |"
        MockSplitter.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(main, ["allocate"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    def test_allocate_json_format(self, MockTaskType, MockSplitter):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.DATA_GENERATION = "DATA_GENERATION"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        mock_result = MagicMock()
        MockSplitter.return_value.analyze.return_value = mock_result
        MockSplitter.return_value.to_dict.return_value = {"tasks": []}

        result = self.runner.invoke(main, ["allocate", "--format", "json"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    def test_allocate_with_size(self, MockTaskType, MockSplitter):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.DATA_GENERATION = "DATA_GENERATION"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        mock_result = MagicMock()
        mock_result.summary.return_value = "Summary"
        mock_result.to_markdown_table.return_value = "| Task |"
        MockSplitter.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(main, ["allocate", "--size", "5000"])
        self.assertEqual(result.exit_code, 0)
        MockSplitter.return_value.analyze.assert_called_once()
        call_kwargs = MockSplitter.return_value.analyze.call_args
        self.assertEqual(call_kwargs[1]["dataset_size"], 5000)


class TestGenerateCommand(unittest.TestCase):
    """Test the `generate` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.generators.PatternGenerator")
    def test_generate_rubrics(self, MockGen):
        mock_result = MagicMock()
        mock_result.summary.return_value = "Generated 10 rubrics"
        mock_result.items = [MagicMock(data_type="rubric", content="Test rubric content here")]
        MockGen.return_value.generate_rubrics.return_value = mock_result

        result = self.runner.invoke(main, ["generate", "--type", "rubrics", "--count", "10"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.generators.PatternGenerator")
    def test_generate_prompts(self, MockGen):
        mock_result = MagicMock()
        mock_result.summary.return_value = "Generated 5 prompts"
        mock_result.items = []
        MockGen.return_value.generate_prompts.return_value = mock_result

        result = self.runner.invoke(main, ["generate", "--type", "prompts", "--count", "5"])
        self.assertEqual(result.exit_code, 0)


# ==================== infra.py Command Tests ====================
#
# infra.py lazy-imports from datarecipe.cache, datarecipe.knowledge,
# datarecipe.triggers.  Patch the source modules.
#


class TestCacheCommand(unittest.TestCase):
    """Test the `cache` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.cache.AnalysisCache")
    def test_cache_default_shows_stats(self, MockCache):
        mock_cache = MockCache.return_value
        mock_cache.get_stats.return_value = {
            "total_entries": 5,
            "valid_entries": 3,
            "expired_entries": 2,
            "total_size_mb": 1.2,
            "cache_dir": "/tmp/cache",
        }

        result = self.runner.invoke(main, ["cache"])
        self.assertEqual(result.exit_code, 0)
        # Default action shows overview with entry count
        self.assertIn("5", result.output)

    @patch("datarecipe.cache.AnalysisCache")
    def test_cache_stats(self, MockCache):
        mock_cache = MockCache.return_value
        mock_cache.get_stats.return_value = {
            "total_entries": 10,
            "valid_entries": 8,
            "expired_entries": 2,
            "total_size_mb": 2.5,
            "cache_dir": "/tmp/cache",
        }

        result = self.runner.invoke(main, ["cache", "--stats"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("10", result.output)

    @patch("datarecipe.cache.AnalysisCache")
    def test_cache_list_empty(self, MockCache):
        mock_cache = MockCache.return_value
        mock_cache.list_entries.return_value = []

        result = self.runner.invoke(main, ["cache", "--list"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.cache.AnalysisCache")
    def test_cache_clear(self, MockCache):
        mock_cache = MockCache.return_value
        result = self.runner.invoke(main, ["cache", "--clear"])
        self.assertEqual(result.exit_code, 0)
        mock_cache.clear_all.assert_called_once_with(delete_files=True)

    @patch("datarecipe.cache.AnalysisCache")
    def test_cache_clear_expired(self, MockCache):
        mock_cache = MockCache.return_value
        mock_cache.clear_expired.return_value = 3
        result = self.runner.invoke(main, ["cache", "--clear-expired"])
        self.assertEqual(result.exit_code, 0)
        mock_cache.clear_expired.assert_called_once_with(delete_files=True)

    @patch("datarecipe.cache.AnalysisCache")
    def test_cache_invalidate(self, MockCache):
        mock_cache = MockCache.return_value
        result = self.runner.invoke(main, ["cache", "--invalidate", "some/dataset"])
        self.assertEqual(result.exit_code, 0)
        mock_cache.invalidate.assert_called_once_with("some/dataset", delete_files=False)


class TestKnowledgeCommand(unittest.TestCase):
    """Test the `knowledge` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_default(self, MockKB):
        mock_kb = MockKB.return_value
        mock_kb.patterns.get_pattern_stats.return_value = {"total_patterns": 0}
        mock_kb.trends.get_all_benchmarks.return_value = {}

        result = self.runner.invoke(main, ["knowledge"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_patterns(self, MockKB):
        mock_kb = MockKB.return_value
        mock_kb.patterns.get_pattern_stats.return_value = {
            "total_patterns": 5,
            "top_patterns": [
                {"key": "distillation", "type": "generation", "frequency": 3},
            ],
        }

        result = self.runner.invoke(main, ["knowledge", "--patterns"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("distillation", result.output)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_patterns_empty(self, MockKB):
        mock_kb = MockKB.return_value
        mock_kb.patterns.get_pattern_stats.return_value = {
            "total_patterns": 0,
            "top_patterns": [],
        }

        result = self.runner.invoke(main, ["knowledge", "--patterns"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_benchmarks_empty(self, MockKB):
        mock_kb = MockKB.return_value
        mock_kb.trends.get_all_benchmarks.return_value = {}

        result = self.runner.invoke(main, ["knowledge", "--benchmarks"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_trends_empty(self, MockKB):
        mock_kb = MockKB.return_value
        mock_kb.trends.get_trend_summary.return_value = {"datasets_analyzed": 0}

        result = self.runner.invoke(main, ["knowledge", "--trends"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_recommend(self, MockKB):
        mock_kb = MockKB.return_value
        mock_kb.get_recommendations.return_value = {
            "cost_estimate": {
                "avg_total": 5000,
                "range": [2000, 8000],
                "avg_human_percentage": 60,
                "based_on": 3,
            },
            "common_patterns": [],
            "suggested_fields": [],
        }

        result = self.runner.invoke(main, ["knowledge", "--recommend", "preference"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("5,000", result.output)


# ==================== deep.py Command Tests ====================
#
# deep.py lazy-imports AnalysisCache from datarecipe.cache and
# DeepAnalyzerCore from datarecipe.core.deep_analyzer.
#


class TestDeepAnalyzeCommand(unittest.TestCase):
    """Test the `deep-analyze` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_success(self, MockCore):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sample_count = 200
        mock_result.dataset_type = "evaluation"
        mock_result.rubric_patterns = 50
        mock_result.prompt_templates = 10
        mock_result.human_percentage = 60.0
        mock_result.files_generated = ["data.json", "ANALYSIS_REPORT.md"]
        mock_result.output_dir = "/tmp/test_output"
        mock_result.warnings = []
        MockCore.return_value.analyze.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["deep-analyze", "test/dataset", "-o", tmpdir, "--no-cache"]
            )
            self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_failure(self, MockCore):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Could not load dataset"
        MockCore.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(main, ["deep-analyze", "bad/dataset", "--no-cache"])
        self.assertEqual(result.exit_code, 0)  # exits normally, prints error

    @patch("datarecipe.cache.AnalysisCache")
    def test_deep_analyze_cache_hit(self, MockCache):
        mock_entry = MagicMock()
        mock_entry.created_at = "2025-01-01T00:00:00"
        mock_entry.dataset_type = "evaluation"
        mock_entry.sample_count = 500
        mock_entry.output_dir = "/tmp/cached"
        MockCache.return_value.get.return_value = mock_entry

        result = self.runner.invoke(main, ["deep-analyze", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("2025-01-01", result.output)


# ==================== Guide Command Tests ====================


class TestGuideCommand(unittest.TestCase):
    """Test the `guide` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.pipeline.pipeline_to_markdown")
    @patch("datarecipe.pipeline.get_pipeline_template")
    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_guide_success(self, MockAnalyzer, MockTemplate, MockToMd):
        recipe = _make_recipe()
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_pipeline = MagicMock()
        mock_pipeline.name = "Mixed Pipeline"
        mock_pipeline.steps = [MagicMock(), MagicMock()]
        mock_pipeline.estimated_total_cost = 5000
        MockTemplate.return_value = mock_pipeline

        MockToMd.return_value = "# Production Guide\nStep 1..."

        result = self.runner.invoke(main, ["guide", "test/dataset"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.pipeline.pipeline_to_markdown")
    @patch("datarecipe.pipeline.get_pipeline_template")
    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_guide_with_output(self, MockAnalyzer, MockTemplate, MockToMd):
        recipe = _make_recipe()
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_pipeline = MagicMock()
        mock_pipeline.name = "Mixed Pipeline"
        mock_pipeline.steps = [MagicMock()]
        mock_pipeline.estimated_total_cost = 3000
        MockTemplate.return_value = mock_pipeline

        MockToMd.return_value = "# Production Guide\nStep 1..."

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "guide.md")
            result = self.runner.invoke(main, ["guide", "test/dataset", "-o", outpath])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))


# ==================== Batch Command Tests ====================


class TestBatchCommand(unittest.TestCase):
    """Test the `batch` command from tools.py."""

    def setUp(self):
        self.runner = CliRunner()

    def test_batch_no_args(self):
        result = self.runner.invoke(main, ["batch"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.batch_analyzer.BatchAnalyzer")
    def test_batch_with_ids(self, MockBatch):
        mock_result = MagicMock()
        mock_result.results = [MagicMock(), MagicMock()]
        mock_result.successful = 2
        mock_result.failed = 0
        mock_result.total_duration_seconds = 5.0
        mock_result.get_failed.return_value = []
        MockBatch.return_value.analyze_batch.return_value = mock_result

        result = self.runner.invoke(main, ["batch", "ds1", "ds2"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Batch Analysis Complete", result.output)


# ==================== Quality Command Tests ====================


class TestQualityCommand(unittest.TestCase):
    """Test the `quality` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.quality_metrics.QualityAnalyzer")
    def test_quality_json_output(self, MockQA):
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"overall_score": 85}
        MockQA.return_value.analyze_from_huggingface.return_value = mock_report

        result = self.runner.invoke(main, ["quality", "test/dataset", "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("overall_score", result.output)

    @patch("datarecipe.quality_metrics.QualityAnalyzer")
    def test_quality_error(self, MockQA):
        MockQA.return_value.analyze_from_huggingface.side_effect = RuntimeError("API timeout")
        result = self.runner.invoke(main, ["quality", "test/dataset"])
        self.assertNotEqual(result.exit_code, 0)


# ==================== Profile Command Tests ====================


class TestProfileCommand(unittest.TestCase):
    """Test the `profile` command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.profiler.profile_to_markdown")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_profile_json_output(self, MockAnalyzer, MockProfiler, MockToMd):
        recipe = _make_recipe()
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_profile = MagicMock()
        mock_profile.to_dict.return_value = {"team_size": 10}
        MockProfiler.return_value.generate_profile.return_value = mock_profile

        result = self.runner.invoke(main, ["profile", "test/dataset", "--json"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("team_size", result.output)

    @patch("datarecipe.profiler.profile_to_markdown")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_profile_markdown_output(self, MockAnalyzer, MockProfiler, MockToMd):
        recipe = _make_recipe()
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_profile = MagicMock()
        MockProfiler.return_value.generate_profile.return_value = mock_profile
        MockToMd.return_value = "# Annotator Profile\n..."

        result = self.runner.invoke(main, ["profile", "test/dataset", "--markdown"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Annotator Profile", result.output)


# ==================== Providers Subgroup Tests ====================


class TestProvidersCommand(unittest.TestCase):
    """Test the `providers` group and its subcommands."""

    def setUp(self):
        self.runner = CliRunner()

    def test_providers_group_help(self):
        result = self.runner.invoke(main, ["providers", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)

    @patch("datarecipe.providers.list_providers")
    def test_providers_list(self, MockListProviders):
        MockListProviders.return_value = [
            {"name": "local", "description": "Local filesystem output"},
            {"name": "judgeguild", "description": "JudgeGuild integration"},
        ]

        result = self.runner.invoke(main, ["providers", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("local", result.output)


# ==================== Watch Command Tests ====================


class TestWatchCommand(unittest.TestCase):
    """Test the `watch` command (once mode only)."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.triggers.RadarWatcher")
    @patch("datarecipe.triggers.TriggerConfig")
    def test_watch_once_no_reports(self, MockConfig, MockWatcher):
        mock_config = MagicMock(
            orgs=[], categories=[], min_downloads=0
        )
        MockConfig.return_value = mock_config

        MockWatcher.return_value.check_once.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, ["watch", tmpdir, "--once"])
            self.assertEqual(result.exit_code, 0)


# ==================== Edge Cases ====================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.runner = CliRunner()

    def test_unknown_command(self):
        result = self.runner.invoke(main, ["nonexistent-command"])
        self.assertNotEqual(result.exit_code, 0)

    def test_analyze_missing_argument(self):
        result = self.runner.invoke(main, ["analyze"])
        self.assertNotEqual(result.exit_code, 0)

    def test_export_missing_output_file(self):
        result = self.runner.invoke(main, ["export", "test/dataset"])
        self.assertNotEqual(result.exit_code, 0)

    def test_compare_single_dataset(self):
        """compare requires at least 2 datasets."""
        result = self.runner.invoke(main, ["compare", "one/dataset"])
        self.assertNotEqual(result.exit_code, 0)


class TestRecipeToMarkdownEdgeCases(unittest.TestCase):
    """Test recipe_to_markdown with various edge cases."""

    def test_empty_languages(self):
        recipe = _make_recipe(languages=[])
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)

    def test_none_language_filtered(self):
        recipe = _make_recipe(languages=[None, "en", None])
        md = recipe_to_markdown(recipe)
        self.assertIn("en", md)

    def test_no_license(self):
        recipe = _make_recipe(license=None)
        md = recipe_to_markdown(recipe)
        self.assertNotIn("MIT", md)

    def test_no_num_examples(self):
        recipe = _make_recipe(num_examples=None)
        md = recipe_to_markdown(recipe)
        self.assertIn("test/dataset", md)

    def test_high_confidence_cost(self):
        recipe = _make_recipe(
            cost=Cost(estimated_total_usd=2000.0, confidence="high")
        )
        md = recipe_to_markdown(recipe)
        self.assertIn("2,000", md)

    def test_multiple_teacher_models(self):
        recipe = _make_recipe(teacher_models=["gpt-4", "claude-3", "gemini"])
        md = recipe_to_markdown(recipe)
        self.assertIn("gpt-4", md)
        self.assertIn("claude-3", md)
        self.assertIn("gemini", md)

    def test_generation_method_with_platform(self):
        recipe = _make_recipe(
            generation_methods=[
                GenerationMethod(
                    method_type="human_annotation",
                    platform="Scale AI",
                ),
            ]
        )
        md = recipe_to_markdown(recipe)
        self.assertIn("Scale AI", md)


if __name__ == "__main__":
    unittest.main()
