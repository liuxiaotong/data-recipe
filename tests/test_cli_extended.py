"""Extended CLI tests to increase coverage for deep.py, spec.py, batch.py, tools.py, analyze.py, infra.py.

Supplements tests/test_cli_commands.py with deeper path coverage:
- deep.py: deep_analyze full success/failure paths, _generate_analysis_report, _generate_reproduction_guide
- spec.py: all 3 modes (API, interactive, from-json)
- batch.py: batch_from_radar with filtering/sorting/incremental, integrate_report
- tools.py: workflow, deploy, extract-rubrics, extract-prompts, detect-strategy, enhanced-guide,
            create, quality (table mode), compare (output), profile (table+export), generate (contexts)
- analyze.py: deep_guide, guide error paths
- infra.py: watch (with results), cache --list with entries, knowledge --report/--benchmarks/--trends
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from datarecipe.cli import main

# ============================================================================
# deep.py: _generate_analysis_report
# ============================================================================


class TestGenerateAnalysisReport(unittest.TestCase):
    """Test the _generate_analysis_report helper in deep.py."""

    def _make_allocation(self):
        alloc = MagicMock()
        alloc.total_cost = 10000
        alloc.total_human_cost = 6000
        alloc.total_machine_cost = 4000
        alloc.human_work_percentage = 60
        alloc.machine_work_percentage = 40
        alloc.estimated_savings_vs_all_human = 5000
        task = MagicMock()
        task.task_name = "Context Creation"
        task.decision.value = "human_primary"
        task.human_percentage = 70
        task.human_hours = 100
        task.human_cost = 3000
        task.machine_cost = 500
        alloc.tasks = [task]
        return alloc

    def _make_rubrics_result(self):
        r = MagicMock()
        r.total_rubrics = 100
        r.unique_patterns = 20
        r.avg_rubrics_per_task = 5.0
        r.verb_distribution = {"include": 30, "explain": 20, "provide": 15}
        r.category_distribution = {"accuracy": 40, "completeness": 30, "format": 20}
        r.structured_templates = [
            {"category": "accuracy", "action": "include", "target": "facts", "condition": None, "frequency": 10}
        ]
        return r

    def _make_prompt_library(self):
        lib = MagicMock()
        lib.total_extracted = 50
        lib.unique_count = 30
        lib.deduplication_ratio = 0.4
        lib.avg_length = 500
        lib.category_counts = {"system": 20, "task": 10}
        lib.domain_counts = {"general": 15, "coding": 10}
        return lib

    def _make_strategy_result(self):
        sr = MagicMock()
        sr.primary_strategy.value = "synthetic"
        sr.confidence = 0.85
        sr.synthetic_score = 0.7
        sr.modified_score = 0.2
        sr.niche_score = 0.1
        sr.synthetic_indicators = ["AI-generated text detected"]
        sr.modified_indicators = ["Paraphrased content"]
        sr.niche_indicators = ["Domain-specific terminology"]
        sr.recommendations = ["Use API-based generation"]
        return sr

    def test_report_with_all_sections(self):
        from datarecipe.cli.deep import _generate_analysis_report

        result = _generate_analysis_report(
            dataset_id="test/dataset",
            sample_count=500,
            actual_size=10000,
            rubrics_result=self._make_rubrics_result(),
            prompt_library=self._make_prompt_library(),
            strategy_result=self._make_strategy_result(),
            allocation=self._make_allocation(),
            region="china",
        )
        self.assertIn("test/dataset", result)
        self.assertIn("10,000", result)
        self.assertIn("include", result)
        self.assertIn("synthetic", result)
        self.assertIn("$10,000", result)

    def test_report_without_rubrics(self):
        from datarecipe.cli.deep import _generate_analysis_report

        result = _generate_analysis_report(
            dataset_id="test/ds",
            sample_count=100,
            actual_size=5000,
            rubrics_result=None,
            prompt_library=self._make_prompt_library(),
            strategy_result=None,
            allocation=self._make_allocation(),
            region="us",
        )
        self.assertIn("test/ds", result)
        # Should still have allocation section
        self.assertIn("人机任务分配", result)

    def test_report_minimal(self):
        from datarecipe.cli.deep import _generate_analysis_report

        result = _generate_analysis_report(
            dataset_id="minimal/ds",
            sample_count=10,
            actual_size=100,
            rubrics_result=None,
            prompt_library=None,
            strategy_result=None,
            allocation=self._make_allocation(),
            region="china",
        )
        self.assertIn("minimal/ds", result)
        self.assertIn("DataRecipe", result)


# ============================================================================
# deep.py: _generate_reproduction_guide
# ============================================================================


class TestGenerateReproductionGuide(unittest.TestCase):
    """Test _generate_reproduction_guide helper in deep.py."""

    def _make_allocation(self):
        alloc = MagicMock()
        alloc.total_human_cost = 5000
        alloc.total_machine_cost = 2000
        alloc.total_cost = 7000
        return alloc

    def _make_rubrics_result(self):
        r = MagicMock()
        r.verb_distribution = {"include": 20, "explain": 10}
        return r

    @patch("datarecipe.analyzers.llm_dataset_analyzer.generate_llm_guide_section", return_value="## LLM Section")
    def test_guide_default(self, mock_llm_section):
        from datarecipe.cli.deep import _generate_reproduction_guide

        result = _generate_reproduction_guide(
            dataset_id="test/ds",
            schema_info={
                "messages": {"type": "list", "nested_type": "dict"},
                "rubrics": {"type": "list", "nested_type": "str"},
                "metadata": {"type": "dict", "nested_type": ["category", "sub"]},
            },
            category_set={"coding", "math"},
            sub_category_set={"algebra", "geometry"},
            system_prompts_by_domain={
                "general": [{"content": "You are a helpful assistant."}],
            },
            rubrics_examples=[
                {
                    "rubrics": ["Include all facts", "Explain reasoning"],
                    "metadata": {"context_category": "coding", "sub_category": "python"},
                }
            ],
            sample_items=[{"messages": [{"role": "user", "content": "Hello"}], "rubrics": ["Test"]}],
            rubrics_result=self._make_rubrics_result(),
            prompt_library=None,
            allocation=self._make_allocation(),
        )
        self.assertIn("test/ds", result)
        self.assertIn("coding", result)
        self.assertIn("algebra", result)
        self.assertIn("$7,000", result)
        self.assertIn("SOP", result)

    @patch("datarecipe.analyzers.llm_dataset_analyzer.generate_llm_guide_section", return_value="")
    def test_guide_preference_dataset(self, mock_llm_section):
        from datarecipe.cli.deep import _generate_reproduction_guide

        result = _generate_reproduction_guide(
            dataset_id="pref/ds",
            schema_info={"chosen": {"type": "str"}, "rejected": {"type": "str"}},
            category_set=set(),
            sub_category_set=set(),
            system_prompts_by_domain={},
            rubrics_examples=[],
            sample_items=[],
            rubrics_result=None,
            prompt_library=None,
            allocation=self._make_allocation(),
            is_preference_dataset=True,
            preference_pairs=[
                {"topic": "safety", "human_query": "Is this safe?", "chosen_response": "Yes", "rejected_response": "No"},
            ],
            preference_topics={"safety": 10, "helpfulness": 5},
            preference_patterns={"chosen_longer": 8, "rejected_longer": 2, "same_length": 0, "chosen_safer": 3},
        )
        self.assertIn("RLHF", result)
        self.assertIn("偏好", result)
        self.assertIn("safety", result)

    @patch("datarecipe.analyzers.llm_dataset_analyzer.generate_llm_guide_section", return_value="")
    def test_guide_swe_dataset(self, mock_llm_section):
        from datarecipe.cli.deep import _generate_reproduction_guide

        result = _generate_reproduction_guide(
            dataset_id="swe/ds",
            schema_info={"repo": {"type": "str"}, "patch": {"type": "str"}},
            category_set=set(),
            sub_category_set=set(),
            system_prompts_by_domain={},
            rubrics_examples=[],
            sample_items=[],
            rubrics_result=None,
            prompt_library=None,
            allocation=self._make_allocation(),
            is_swe_dataset=True,
            swe_stats={
                "languages": {"Python": 50, "Java": 30},
                "repos": {"django/django": 20, "flask/flask": 10},
                "issue_types": {"bug_fix": 30, "feature": 20},
                "issue_categories": {"testing": 15, "database": 10},
                "patch_lines": [10, 20, 30, 40, 50],
                "examples": [
                    {
                        "repo": "django/django",
                        "language": "Python",
                        "problem_statement": "Fix the ORM query...",
                        "requirements": "Add test case...",
                    }
                ],
            },
        )
        self.assertIn("SWE-bench", result)
        self.assertIn("Python", result)
        self.assertIn("django/django", result)

    @patch("datarecipe.analyzers.llm_dataset_analyzer.generate_llm_guide_section", return_value="## LLM Section")
    def test_guide_with_llm_analysis(self, mock_llm_section):
        from datarecipe.cli.deep import _generate_reproduction_guide

        llm_analysis = MagicMock()
        llm_analysis.dataset_type = "evaluation"
        llm_analysis.purpose = "Evaluate language models"

        result = _generate_reproduction_guide(
            dataset_id="llm/ds",
            schema_info={"text": {"type": "str"}},
            category_set=set(),
            sub_category_set=set(),
            system_prompts_by_domain={},
            rubrics_examples=[],
            sample_items=[],
            rubrics_result=None,
            prompt_library=None,
            allocation=self._make_allocation(),
            llm_analysis=llm_analysis,
        )
        self.assertIn("evaluation", result)
        self.assertIn("LLM Section", result)


# ============================================================================
# deep.py: deep_analyze CLI command extended paths
# ============================================================================


class TestDeepAnalyzeExtended(unittest.TestCase):
    """Extended tests for deep_analyze CLI command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_preference_type(self, MockCore):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sample_count = 200
        mock_result.dataset_type = "preference"
        mock_result.rubric_patterns = 0
        mock_result.prompt_templates = 0
        mock_result.human_percentage = 80.0
        mock_result.files_generated = []
        mock_result.output_dir = "/tmp/pref_output"
        mock_result.warnings = []
        MockCore.return_value.analyze.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["deep-analyze", "test/pref-dataset", "-o", tmpdir, "--no-cache"]
            )
            self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_swe_bench_type(self, MockCore):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sample_count = 100
        mock_result.dataset_type = "swe_bench"
        mock_result.rubric_patterns = 0
        mock_result.prompt_templates = 5
        mock_result.human_percentage = 50.0
        mock_result.files_generated = ["data.json"]
        mock_result.output_dir = "/tmp/swe_output"
        mock_result.warnings = ["Some warning"]
        MockCore.return_value.analyze.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake file to test file size display
            os.makedirs(os.path.join(tmpdir, "test_swe-bench"), exist_ok=True)
            fpath = os.path.join(tmpdir, "test_swe-bench", "data.json")
            Path(fpath).write_text('{"test": true}')
            mock_result.output_dir = os.path.join(tmpdir, "test_swe-bench")

            result = self.runner.invoke(
                main, ["deep-analyze", "test/swe-bench", "-o", tmpdir, "--no-cache"]
            )
            self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_with_rubrics_and_prompts(self, MockCore):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sample_count = 300
        mock_result.dataset_type = "evaluation"
        mock_result.rubric_patterns = 25
        mock_result.prompt_templates = 15
        mock_result.human_percentage = 65.0
        mock_result.files_generated = ["ANALYSIS_REPORT.md", "REPRODUCTION_GUIDE.md"]
        mock_result.output_dir = "/tmp/eval_output"
        mock_result.warnings = []
        MockCore.return_value.analyze.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["deep-analyze", "test/eval-dataset", "-o", tmpdir, "--no-cache"]
            )
            self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_exception(self, MockCore):
        MockCore.return_value.analyze.side_effect = RuntimeError("Connection failed")

        result = self.runner.invoke(
            main, ["deep-analyze", "test/bad-dataset", "--no-cache"]
        )
        self.assertEqual(result.exit_code, 0)  # catches exceptions, prints error

    @patch("datarecipe.cache.AnalysisCache")
    def test_deep_analyze_cache_hit_different_dir(self, MockCache):
        """Test cache hit where cached output_dir differs from requested."""
        mock_entry = MagicMock()
        mock_entry.created_at = "2025-06-01T12:00:00"
        mock_entry.dataset_type = "evaluation"
        mock_entry.sample_count = 500
        mock_entry.output_dir = "/tmp/old_cached_dir"
        MockCache.return_value.get.return_value = mock_entry

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["deep-analyze", "test/dataset", "-o", tmpdir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("2025-06-01", result.output)
            MockCache.return_value.copy_to_output.assert_called_once()

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_with_options(self, MockCore):
        """Test deep_analyze with --use-llm, --size, --split, --region options."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.sample_count = 100
        mock_result.dataset_type = "evaluation"
        mock_result.rubric_patterns = 10
        mock_result.prompt_templates = 5
        mock_result.human_percentage = 55.0
        mock_result.files_generated = []
        mock_result.output_dir = "/tmp/test"
        mock_result.warnings = []
        MockCore.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(
            main,
            [
                "deep-analyze", "test/ds",
                "--no-cache",
                "--use-llm",
                "--llm-provider", "openai",
                "--size", "5000",
                "--region", "us",
                "--split", "test",
                "--sample-size", "100",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        MockCore.assert_called_once_with(
            output_dir="./projects",
            region="us",
            use_llm=True,
            llm_provider="openai",
            enhance_mode="auto",
        )

    @patch("datarecipe.core.deep_analyzer.DeepAnalyzerCore")
    def test_deep_analyze_large_file_display(self, MockCore):
        """Test file size formatting for large files (>1KB)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = os.path.join(tmpdir, "test_dataset")
            os.makedirs(outdir, exist_ok=True)
            # Create a file > 1KB
            large_file = os.path.join(outdir, "big_data.json")
            Path(large_file).write_text("x" * 2048)
            # Create a small file
            small_file = os.path.join(outdir, "small.md")
            Path(small_file).write_text("hello")

            mock_result = MagicMock()
            mock_result.success = True
            mock_result.sample_count = 50
            mock_result.dataset_type = "evaluation"
            mock_result.rubric_patterns = 5
            mock_result.prompt_templates = 3
            mock_result.human_percentage = 70.0
            mock_result.files_generated = ["big_data.json", "small.md"]
            mock_result.output_dir = outdir
            mock_result.warnings = []
            MockCore.return_value.analyze.return_value = mock_result

            result = self.runner.invoke(
                main, ["deep-analyze", "test/dataset", "-o", tmpdir, "--no-cache"]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("KB", result.output)
            self.assertIn("B", result.output)


# ============================================================================
# spec.py: analyze-spec command
# ============================================================================


class TestAnalyzeSpecCommand(unittest.TestCase):
    """Test the analyze-spec command in spec.py."""

    def setUp(self):
        self.runner = CliRunner()

    def test_spec_no_args(self):
        """analyze-spec with no file and no --from-json should print error."""
        result = self.runner.invoke(main, ["analyze-spec"])
        self.assertEqual(result.exit_code, 0)  # command handles error gracefully
        self.assertIn("错误", result.output)

    @patch("datarecipe.generators.spec_output.SpecOutputGenerator")
    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_api_mode(self, MockAnalyzer, MockGenerator):
        """Test API mode (default)."""
        mock_analysis = MagicMock()
        mock_analysis.project_name = "TestProject"
        mock_analysis.dataset_type = "evaluation"
        mock_analysis.estimated_difficulty = "medium"
        mock_analysis.estimated_human_percentage = 60
        mock_analysis.estimated_domain = "general"
        mock_analysis.has_images = False
        mock_analysis.image_count = 0
        MockAnalyzer.return_value.analyze.return_value = mock_analysis

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.files_generated = ["test.md"]
        mock_result.output_dir = "/tmp/spec_out"
        MockGenerator.return_value.generate.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Test requirement document")
            tmppath = f.name

        try:
            # Patch LLMEnhancer to avoid real API calls
            with patch("datarecipe.generators.llm_enhancer.LLMEnhancer") as MockEnhancer:
                mock_enhanced = MagicMock()
                mock_enhanced.generated = False
                MockEnhancer.return_value.enhance.return_value = mock_enhanced

                result = self.runner.invoke(
                    main, ["analyze-spec", tmppath, "-o", "/tmp/test_spec"]
                )
                self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.generators.spec_output.SpecOutputGenerator")
    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_api_mode_with_images(self, MockAnalyzer, MockGenerator):
        """Test API mode with images in document."""
        mock_analysis = MagicMock()
        mock_analysis.project_name = "ImageProject"
        mock_analysis.dataset_type = "classification"
        mock_analysis.estimated_difficulty = "hard"
        mock_analysis.estimated_human_percentage = 80
        mock_analysis.estimated_domain = "vision"
        mock_analysis.has_images = True
        mock_analysis.image_count = 5
        MockAnalyzer.return_value.analyze.return_value = mock_analysis

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.files_generated = []
        mock_result.output_dir = "/tmp/spec_img"
        MockGenerator.return_value.generate.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".pdf", mode="wb", delete=False) as f:
            f.write(b"fake pdf content")
            tmppath = f.name

        try:
            with patch("datarecipe.generators.llm_enhancer.LLMEnhancer") as MockEnhancer:
                MockEnhancer.return_value.enhance.side_effect = Exception("no API key")

                result = self.runner.invoke(
                    main, ["analyze-spec", tmppath, "-o", "/tmp/test_spec"]
                )
                self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.generators.spec_output.SpecOutputGenerator")
    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_from_json_mode(self, MockAnalyzer, MockGenerator):
        """Test --from-json mode."""
        mock_analysis = MagicMock()
        mock_analysis.project_name = "JSONProject"
        mock_analysis.dataset_type = "qa"
        mock_analysis.estimated_difficulty = "easy"
        mock_analysis.estimated_human_percentage = 40
        mock_analysis.estimated_domain = "tech"
        MockAnalyzer.return_value.create_analysis_from_json.return_value = mock_analysis

        mock_doc = MagicMock()
        mock_doc.has_images.return_value = False
        MockAnalyzer.return_value.parse_document.return_value = mock_doc

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.files_generated = ["summary.json"]
        mock_result.output_dir = "/tmp/json_out"
        MockGenerator.return_value.generate.return_value = mock_result

        # Create JSON file
        json_data = {"project_name": "JSONProject", "fields": []}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as jf:
            json.dump(json_data, jf)
            json_path = jf.name

        # Create source file
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as sf:
            sf.write("Source document")
            src_path = sf.name

        try:
            with patch("datarecipe.generators.llm_enhancer.LLMEnhancer") as MockEnhancer:
                mock_enhanced = MagicMock()
                mock_enhanced.generated = True
                MockEnhancer.return_value.enhance.return_value = mock_enhanced

                result = self.runner.invoke(
                    main, ["analyze-spec", src_path, "--from-json", json_path]
                )
                self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(json_path)
            os.unlink(src_path)

    @patch("datarecipe.generators.spec_output.SpecOutputGenerator")
    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_from_json_no_source_file(self, MockAnalyzer, MockGenerator):
        """Test --from-json without source file."""
        mock_analysis = MagicMock()
        mock_analysis.project_name = "NoFile"
        mock_analysis.dataset_type = "unknown"
        mock_analysis.estimated_difficulty = "medium"
        mock_analysis.estimated_human_percentage = 50
        mock_analysis.estimated_domain = "general"
        MockAnalyzer.return_value.create_analysis_from_json.return_value = mock_analysis

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.files_generated = []
        mock_result.output_dir = "/tmp/no_file_out"
        MockGenerator.return_value.generate.return_value = mock_result

        json_data = {"project_name": "NoFile"}
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as jf:
            json.dump(json_data, jf)
            json_path = jf.name

        try:
            with patch("datarecipe.generators.llm_enhancer.LLMEnhancer") as MockEnhancer:
                MockEnhancer.return_value.enhance.side_effect = Exception("skip")

                result = self.runner.invoke(
                    main, ["analyze-spec", "--from-json", json_path]
                )
                self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(json_path)

    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_interactive_mode(self, MockAnalyzer):
        """Test interactive mode with stdin JSON input."""
        mock_doc = MagicMock()
        mock_doc.has_images.return_value = False
        MockAnalyzer.return_value.parse_document.return_value = mock_doc
        MockAnalyzer.return_value.get_extraction_prompt.return_value = "Analyze this document"

        mock_analysis = MagicMock()
        mock_analysis.project_name = "InteractiveProject"
        mock_analysis.dataset_type = "qa"
        mock_analysis.estimated_difficulty = "medium"
        mock_analysis.estimated_human_percentage = 50
        mock_analysis.estimated_domain = "general"
        MockAnalyzer.return_value.create_analysis_from_json.return_value = mock_analysis

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Document content")
            tmppath = f.name

        try:
            with patch("datarecipe.generators.spec_output.SpecOutputGenerator") as MockGen:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.files_generated = []
                mock_result.output_dir = "/tmp/interactive_out"
                MockGen.return_value.generate.return_value = mock_result

                with patch("datarecipe.generators.llm_enhancer.LLMEnhancer") as MockEnhancer:
                    MockEnhancer.return_value.enhance.side_effect = Exception("skip")

                    # Provide JSON via stdin
                    json_input = '{"project_name": "Test"}\n\n'
                    result = self.runner.invoke(
                        main,
                        ["analyze-spec", tmppath, "--interactive"],
                        input=json_input,
                    )
                    self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_interactive_empty_stdin(self, MockAnalyzer):
        """Test interactive mode with empty stdin."""
        mock_doc = MagicMock()
        mock_doc.has_images.return_value = False
        MockAnalyzer.return_value.parse_document.return_value = mock_doc
        MockAnalyzer.return_value.get_extraction_prompt.return_value = "Analyze"

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Doc content")
            tmppath = f.name

        try:
            result = self.runner.invoke(
                main, ["analyze-spec", tmppath, "--interactive"],
                input="\n",
            )
            self.assertEqual(result.exit_code, 0)
            # Should print error about no JSON input
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_interactive_json_code_block(self, MockAnalyzer):
        """Test interactive mode with JSON in markdown code block."""
        mock_doc = MagicMock()
        mock_doc.has_images.return_value = True
        mock_doc.images = ["img1.png", "img2.png"]
        MockAnalyzer.return_value.parse_document.return_value = mock_doc
        MockAnalyzer.return_value.get_extraction_prompt.return_value = "Analyze"

        mock_analysis = MagicMock()
        mock_analysis.project_name = "CodeBlockProject"
        mock_analysis.dataset_type = "qa"
        mock_analysis.estimated_difficulty = "easy"
        mock_analysis.estimated_human_percentage = 30
        mock_analysis.estimated_domain = "tech"
        MockAnalyzer.return_value.create_analysis_from_json.return_value = mock_analysis

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Content")
            tmppath = f.name

        try:
            with patch("datarecipe.generators.spec_output.SpecOutputGenerator") as MockGen:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.files_generated = []
                mock_result.output_dir = "/tmp/code_block"
                MockGen.return_value.generate.return_value = mock_result

                with patch("datarecipe.generators.llm_enhancer.LLMEnhancer") as MockEnhancer:
                    MockEnhancer.return_value.enhance.side_effect = Exception("skip")

                    json_input = '```json\n{"project_name": "Test"}\n```\n\n'
                    result = self.runner.invoke(
                        main,
                        ["analyze-spec", tmppath, "--interactive"],
                        input=json_input,
                    )
                    self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.generators.spec_output.SpecOutputGenerator")
    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_generation_failure(self, MockAnalyzer, MockGenerator):
        """Test when output generation fails."""
        mock_analysis = MagicMock()
        mock_analysis.project_name = "FailProject"
        mock_analysis.dataset_type = "unknown"
        mock_analysis.estimated_difficulty = "medium"
        mock_analysis.estimated_human_percentage = 50
        mock_analysis.estimated_domain = "general"
        mock_analysis.has_images = False
        MockAnalyzer.return_value.analyze.return_value = mock_analysis

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Generation failed"
        MockGenerator.return_value.generate.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Content")
            tmppath = f.name

        try:
            with patch("datarecipe.generators.llm_enhancer.LLMEnhancer") as MockEnhancer:
                MockEnhancer.return_value.enhance.side_effect = Exception("skip")

                result = self.runner.invoke(
                    main, ["analyze-spec", tmppath]
                )
                self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_value_error(self, MockAnalyzer):
        """Test ValueError during analysis."""
        MockAnalyzer.return_value.analyze.side_effect = ValueError("Unsupported format")

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Content")
            tmppath = f.name

        try:
            result = self.runner.invoke(main, ["analyze-spec", tmppath])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("错误", result.output)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.analyzers.spec_analyzer.SpecAnalyzer")
    def test_spec_file_not_found(self, MockAnalyzer):
        """Test FileNotFoundError."""
        MockAnalyzer.return_value.analyze.side_effect = FileNotFoundError("missing.pdf")

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Content")
            tmppath = f.name

        try:
            result = self.runner.invoke(main, ["analyze-spec", tmppath])
            self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)


# ============================================================================
# batch.py: batch-from-radar
# ============================================================================


class TestBatchFromRadar(unittest.TestCase):
    """Test batch-from-radar command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.integrations.radar.RadarIntegration")
    def test_batch_from_radar_load_error(self, MockRadar):
        MockRadar.return_value.load_radar_report.side_effect = Exception("File corrupted")

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            tmppath = f.name

        try:
            result = self.runner.invoke(main, ["batch-from-radar", tmppath])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("错误", result.output)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.integrations.radar.RadarIntegration")
    def test_batch_from_radar_no_matching_datasets(self, MockRadar):
        MockRadar.return_value.load_radar_report.return_value = [MagicMock()]
        MockRadar.return_value.filter_datasets.return_value = []

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            tmppath = f.name

        try:
            result = self.runner.invoke(main, ["batch-from-radar", tmppath])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("没有符合条件", result.output)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.integrations.radar.RadarIntegration")
    def test_batch_from_radar_sort_by_name(self, MockRadar):
        ds1 = MagicMock()
        ds1.id = "zzz/dataset"
        ds1.downloads = 100
        ds1.category = "qa"
        ds2 = MagicMock()
        ds2.id = "aaa/dataset"
        ds2.downloads = 50
        ds2.category = "eval"

        MockRadar.return_value.load_radar_report.return_value = [ds1, ds2]
        MockRadar.return_value.filter_datasets.return_value = [ds1, ds2]

        # Make the analysis loop fail quickly with an exception
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            tmppath = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("datasets.load_dataset", side_effect=Exception("skip")):
                    result = self.runner.invoke(
                        main,
                        ["batch-from-radar", tmppath, "-o", tmpdir, "--sort-by", "name", "--limit", "2"],
                    )
                    self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.integrations.radar.RadarIntegration")
    def test_batch_from_radar_sort_by_category(self, MockRadar):
        ds1 = MagicMock()
        ds1.id = "test/ds1"
        ds1.downloads = 100
        ds1.category = "beta"
        ds2 = MagicMock()
        ds2.id = "test/ds2"
        ds2.downloads = 200
        ds2.category = "alpha"

        MockRadar.return_value.load_radar_report.return_value = [ds1, ds2]
        MockRadar.return_value.filter_datasets.return_value = [ds1, ds2]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            tmppath = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("datasets.load_dataset", side_effect=Exception("skip")):
                    result = self.runner.invoke(
                        main,
                        ["batch-from-radar", tmppath, "-o", tmpdir, "--sort-by", "category"],
                    )
                    self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.integrations.radar.RadarIntegration")
    def test_batch_from_radar_incremental_all_done(self, MockRadar):
        ds1 = MagicMock()
        ds1.id = "test/ds1"
        ds1.downloads = 100
        ds1.category = "eval"

        MockRadar.return_value.load_radar_report.return_value = [ds1]
        MockRadar.return_value.filter_datasets.return_value = [ds1]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            tmppath = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create the summary file so incremental mode skips it
                ds_dir = os.path.join(tmpdir, "test_ds1")
                os.makedirs(ds_dir)
                Path(os.path.join(ds_dir, "recipe_summary.json")).write_text("{}")

                result = self.runner.invoke(
                    main,
                    ["batch-from-radar", tmppath, "-o", tmpdir, "--incremental"],
                )
                self.assertEqual(result.exit_code, 0)
                self.assertIn("已分析完成", result.output)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.integrations.radar.RadarIntegration")
    def test_batch_from_radar_with_org_filter(self, MockRadar):
        ds1 = MagicMock()
        ds1.id = "Anthropic/test"
        ds1.downloads = 100
        ds1.category = "eval"

        MockRadar.return_value.load_radar_report.return_value = [ds1]
        MockRadar.return_value.filter_datasets.return_value = [ds1]

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({}, f)
            tmppath = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("datasets.load_dataset", side_effect=Exception("skip")):
                    result = self.runner.invoke(
                        main,
                        ["batch-from-radar", tmppath, "-o", tmpdir, "--orgs", "Anthropic", "--categories", "eval"],
                    )
                    self.assertEqual(result.exit_code, 0)
        finally:
            os.unlink(tmppath)


# ============================================================================
# batch.py: integrate-report
# ============================================================================


class TestIntegrateReport(unittest.TestCase):
    """Test integrate-report command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.reports.IntegratedReportGenerator")
    def test_integrate_report_success(self, MockGen):
        mock_report = MagicMock()
        mock_report.period_start = "2025-01-01"
        mock_report.period_end = "2025-01-07"
        mock_report.total_discovered = 10
        mock_report.total_analyzed = 8
        mock_report.total_reproduction_cost = 50000
        mock_report.insights = ["Insight 1", "Insight 2"]
        MockGen.return_value.generate_weekly_report.return_value = mock_report
        MockGen.return_value.save_report.return_value = {"md": "/tmp/report.md", "json": "/tmp/report.json"}

        result = self.runner.invoke(
            main,
            ["integrate-report", "-o", "/tmp/reports"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("2025-01-01", result.output)

    @patch("datarecipe.reports.IntegratedReportGenerator")
    def test_integrate_report_with_dates(self, MockGen):
        mock_report = MagicMock()
        mock_report.period_start = "2025-06-01"
        mock_report.period_end = "2025-06-07"
        mock_report.total_discovered = 5
        mock_report.total_analyzed = 3
        mock_report.total_reproduction_cost = 10000
        mock_report.insights = []
        MockGen.return_value.generate_weekly_report.return_value = mock_report
        MockGen.return_value.save_report.return_value = {"md": "/tmp/r.md"}

        result = self.runner.invoke(
            main,
            [
                "integrate-report",
                "--start-date", "2025-06-01",
                "--end-date", "2025-06-07",
                "-o", "/tmp/reports",
            ],
        )
        self.assertEqual(result.exit_code, 0)


# ============================================================================
# tools.py: workflow command
# ============================================================================


class TestWorkflowCommand(unittest.TestCase):
    """Test the workflow command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.workflow.WorkflowGenerator")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_workflow_success(self, MockAnalyzer, MockWFGen):

        recipe = MagicMock()
        recipe.name = "test/dataset"
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_wf = MagicMock()
        mock_wf.target_size = 10000
        mock_wf.estimated_total_cost = 5000
        mock_wf.steps = [MagicMock(), MagicMock()]
        mock_wf.export_project.return_value = ["README.md", "config.yaml", "run.sh"]
        MockWFGen.return_value.generate.return_value = mock_wf

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["workflow", "test/dataset", "-o", tmpdir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Workflow generated", result.output)

    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_workflow_analyze_error(self, MockAnalyzer):
        MockAnalyzer.return_value.analyze.side_effect = ValueError("Not found")
        result = self.runner.invoke(
            main, ["workflow", "bad/ds", "-o", "/tmp/wf"]
        )
        self.assertNotEqual(result.exit_code, 0)


# ============================================================================
# tools.py: deploy command
# ============================================================================


class TestDeployCommand(unittest.TestCase):
    """Test the deploy command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.deployer.ProductionDeployer")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_deploy_success(self, MockAnalyzer, MockProfiler, MockDeployer):
        recipe = MagicMock()
        recipe.name = "test/dataset"
        recipe.version = "1.0"
        recipe.source_type = MagicMock()
        recipe.source_id = "test/dataset"
        recipe.num_examples = 1000
        recipe.languages = ["en"]
        recipe.license = "MIT"
        recipe.description = "Test"
        recipe.generation_type = MagicMock()
        recipe.synthetic_ratio = 0.5
        recipe.human_ratio = 0.5
        recipe.generation_methods = []
        recipe.teacher_models = []
        recipe.tags = []
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_profile = MagicMock()
        MockProfiler.return_value.generate_profile.return_value = mock_profile

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.project_handle = None
        mock_result.details = "Deployed successfully"
        MockDeployer.return_value.generate_config.return_value = MagicMock()
        MockDeployer.return_value.deploy.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["deploy", "test/dataset", "-o", tmpdir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Deployment successful", result.output)

    @patch("datarecipe.deployer.ProductionDeployer")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_deploy_failure(self, MockAnalyzer, MockProfiler, MockDeployer):
        recipe = MagicMock()
        recipe.name = "test/dataset"
        recipe.version = "1.0"
        recipe.source_type = MagicMock()
        recipe.source_id = "test"
        recipe.num_examples = 100
        recipe.languages = []
        recipe.license = None
        recipe.description = None
        recipe.generation_type = MagicMock()
        recipe.synthetic_ratio = 0
        recipe.human_ratio = 1
        recipe.generation_methods = []
        recipe.teacher_models = []
        recipe.tags = []
        MockAnalyzer.return_value.analyze.return_value = recipe

        MockProfiler.return_value.generate_profile.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Provider unavailable"
        MockDeployer.return_value.generate_config.return_value = MagicMock()
        MockDeployer.return_value.deploy.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["deploy", "test/ds", "-o", tmpdir]
            )
            self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_deploy_analyze_error(self, MockAnalyzer):
        MockAnalyzer.return_value.analyze.side_effect = Exception("Cannot analyze")
        result = self.runner.invoke(
            main, ["deploy", "bad/ds", "-o", "/tmp/deploy"]
        )
        self.assertNotEqual(result.exit_code, 0)


# ============================================================================
# tools.py: quality (table mode)
# ============================================================================


class TestQualityTableMode(unittest.TestCase):
    """Test quality command table (non-JSON) output."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.quality_metrics.QualityAnalyzer")
    def test_quality_table_output(self, MockQA):
        mock_report = MagicMock()
        mock_report.sample_size = 100
        mock_report.overall_score = 85
        mock_report.diversity.unique_token_ratio = 0.8
        mock_report.diversity.vocabulary_size = 5000
        mock_report.diversity.semantic_diversity = 0.7
        mock_report.consistency.format_consistency = 0.9
        mock_report.consistency.structure_score = 0.85
        mock_report.consistency.field_completeness = 0.95
        mock_report.complexity.avg_length = 200
        mock_report.complexity.avg_tokens = 50
        mock_report.complexity.vocabulary_richness = 0.6
        mock_report.complexity.readability_score = 70
        mock_report.ai_detection = None
        mock_report.recommendations = ["Improve diversity"]
        mock_report.warnings = ["Low sample size"]
        MockQA.return_value.analyze_from_huggingface.return_value = mock_report

        result = self.runner.invoke(main, ["quality", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Quality Report", result.output)
        self.assertIn("Recommendations", result.output)
        self.assertIn("Warnings", result.output)

    @patch("datarecipe.quality_metrics.QualityAnalyzer")
    def test_quality_with_ai_detection(self, MockQA):
        mock_report = MagicMock()
        mock_report.sample_size = 50
        mock_report.overall_score = 60
        mock_report.diversity.unique_token_ratio = 0.5
        mock_report.diversity.vocabulary_size = 2000
        mock_report.diversity.semantic_diversity = 0.4
        mock_report.consistency.format_consistency = 0.7
        mock_report.consistency.structure_score = 0.6
        mock_report.consistency.field_completeness = 0.8
        mock_report.complexity.avg_length = 100
        mock_report.complexity.avg_tokens = 25
        mock_report.complexity.vocabulary_richness = 0.3
        mock_report.complexity.readability_score = 50
        mock_report.ai_detection = MagicMock()
        mock_report.ai_detection.ai_probability = 0.8
        mock_report.ai_detection.confidence = 0.9
        mock_report.ai_detection.indicators = ["repetitive phrases", "uniform length", "formal tone"]
        mock_report.recommendations = []
        mock_report.warnings = []
        MockQA.return_value.analyze_from_huggingface.return_value = mock_report

        result = self.runner.invoke(
            main, ["quality", "test/dataset", "--detect-ai"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("AI Detection", result.output)


# ============================================================================
# tools.py: compare with output file
# ============================================================================


class TestCompareWithOutput(unittest.TestCase):
    """Test compare command with --output option."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.comparator.DatasetComparator")
    def test_compare_table_with_output(self, MockComparator):
        mock_report = MagicMock()
        mock_report.to_table.return_value = "| Feature | ds1 | ds2 |\n"
        mock_report.recommendations = []
        MockComparator.return_value.compare_by_ids.return_value = mock_report

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "compare.txt")
            result = self.runner.invoke(
                main, ["compare", "ds1", "ds2", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datarecipe.comparator.DatasetComparator")
    def test_compare_error(self, MockComparator):
        MockComparator.return_value.compare_by_ids.side_effect = RuntimeError("API Error")
        result = self.runner.invoke(main, ["compare", "ds1", "ds2"])
        self.assertNotEqual(result.exit_code, 0)


# ============================================================================
# tools.py: profile table mode and export
# ============================================================================


class TestProfileTableMode(unittest.TestCase):
    """Test profile command with default table output and export."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.profiler.profile_to_markdown")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_profile_table_output(self, MockAnalyzer, MockProfiler, MockToMd):
        recipe = MagicMock()
        recipe.name = "test/dataset"
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_profile = MagicMock()
        skill = MagicMock()
        skill.name = "Python"
        skill.level = "Advanced"
        skill.required = True
        mock_profile.skill_requirements = [skill]
        mock_profile.experience_level.value = "Senior"
        mock_profile.education_level.value = "Bachelor"
        mock_profile.domain_knowledge = ["NLP", "ML"]
        mock_profile.language_requirements = ["English", "Chinese"]
        mock_profile.hourly_rate_range = {"min": 20, "max": 50}
        mock_profile.estimated_person_days = 30
        mock_profile.estimated_hours_per_example = 0.5
        mock_profile.team_size = 5
        MockProfiler.return_value.generate_profile.return_value = mock_profile

        result = self.runner.invoke(main, ["profile", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Annotator Profile", result.output)
        self.assertIn("Senior", result.output)

    @patch("datarecipe.profiler.profile_to_markdown")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_profile_export_md(self, MockAnalyzer, MockProfiler, MockToMd):
        recipe = MagicMock()
        recipe.name = "test/dataset"
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_profile = MagicMock()
        mock_profile.skill_requirements = []
        mock_profile.experience_level.value = "Mid"
        mock_profile.education_level.value = "Master"
        mock_profile.domain_knowledge = []
        mock_profile.language_requirements = []
        mock_profile.hourly_rate_range = {"min": 15, "max": 45}
        mock_profile.estimated_person_days = 20
        mock_profile.estimated_hours_per_example = 0.3
        mock_profile.team_size = 3
        MockProfiler.return_value.generate_profile.return_value = mock_profile
        MockToMd.return_value = "# Profile\nContent"

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "profile.md")
            result = self.runner.invoke(
                main, ["profile", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datarecipe.profiler.profile_to_markdown")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_profile_export_json(self, MockAnalyzer, MockProfiler, MockToMd):
        recipe = MagicMock()
        recipe.name = "test/dataset"
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_profile = MagicMock()
        mock_profile.to_dict.return_value = {"team_size": 5}
        mock_profile.skill_requirements = []
        mock_profile.experience_level.value = "Mid"
        mock_profile.education_level.value = "Master"
        mock_profile.domain_knowledge = []
        mock_profile.language_requirements = []
        mock_profile.hourly_rate_range = {"min": 15, "max": 45}
        mock_profile.estimated_person_days = 20
        mock_profile.estimated_hours_per_example = 0.3
        mock_profile.team_size = 3
        MockProfiler.return_value.generate_profile.return_value = mock_profile

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "profile.json")
            result = self.runner.invoke(
                main, ["profile", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datarecipe.profiler.profile_to_markdown")
    @patch("datarecipe.profiler.AnnotatorProfiler")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_profile_export_yaml(self, MockAnalyzer, MockProfiler, MockToMd):
        recipe = MagicMock()
        recipe.name = "test/dataset"
        MockAnalyzer.return_value.analyze.return_value = recipe

        mock_profile = MagicMock()
        mock_profile.to_dict.return_value = {"team_size": 3}
        mock_profile.skill_requirements = []
        mock_profile.experience_level.value = "Junior"
        mock_profile.education_level.value = "Bachelor"
        mock_profile.domain_knowledge = []
        mock_profile.language_requirements = []
        mock_profile.hourly_rate_range = {"min": 10, "max": 30}
        mock_profile.estimated_person_days = 10
        mock_profile.estimated_hours_per_example = 0.2
        mock_profile.team_size = 2
        MockProfiler.return_value.generate_profile.return_value = mock_profile

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "profile.yaml")
            result = self.runner.invoke(
                main, ["profile", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_profile_analyze_error(self, MockAnalyzer):
        MockAnalyzer.return_value.analyze.side_effect = Exception("Cannot analyze")
        result = self.runner.invoke(main, ["profile", "bad/ds"])
        self.assertNotEqual(result.exit_code, 0)


# ============================================================================
# tools.py: generate contexts
# ============================================================================


class TestGenerateContexts(unittest.TestCase):
    """Test generate command with contexts type."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.generators.PatternGenerator")
    def test_generate_contexts(self, MockGen):
        mock_result = MagicMock()
        mock_result.summary.return_value = "Generated 5 contexts"
        mock_result.items = [
            MagicMock(data_type="context", content="A long context about AI safety and alignment research"),
        ]
        MockGen.return_value.generate_contexts.return_value = mock_result

        result = self.runner.invoke(
            main, ["generate", "--type", "contexts", "--count", "5"]
        )
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.generators.PatternGenerator")
    def test_generate_with_output(self, MockGen):
        mock_result = MagicMock()
        mock_result.summary.return_value = "Generated 3 rubrics"
        mock_result.items = []
        MockGen.return_value.generate_rubrics.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "output.jsonl")
            result = self.runner.invoke(
                main, ["generate", "--type", "rubrics", "--count", "3", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            MockGen.return_value.export_jsonl.assert_called_once()

    @patch("datarecipe.generators.PatternGenerator")
    def test_generate_many_items(self, MockGen):
        """Test generate with > 5 items to test truncation display."""
        mock_result = MagicMock()
        mock_result.summary.return_value = "Generated 10 prompts"
        items = [MagicMock(data_type="prompt", content=f"Prompt {i} content text") for i in range(10)]
        mock_result.items = items
        MockGen.return_value.generate_prompts.return_value = mock_result

        result = self.runner.invoke(
            main, ["generate", "--type", "prompts", "--count", "10"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("5 more", result.output)


# ============================================================================
# tools.py: extract-rubrics, extract-prompts, detect-strategy
# ============================================================================


class TestExtractRubrics(unittest.TestCase):
    """Test extract-rubrics command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datasets.load_dataset")
    @patch("datarecipe.extractors.RubricsAnalyzer")
    def test_extract_rubrics_success(self, MockAnalyzer, MockLoadDS):
        # Mock dataset with rubrics
        items = [
            {"rubrics": ["Include all facts", "Explain reasoning"]},
            {"rubrics": ["Be concise"]},
        ]
        MockLoadDS.return_value = iter(items)

        mock_result = MagicMock()
        mock_result.summary.return_value = "Found 3 rubrics"
        mock_result.structured_templates = [
            {"category": "accuracy", "action": "include", "target": "facts", "condition": None}
        ]
        MockAnalyzer.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(main, ["extract-rubrics", "test/dataset"])
        self.assertEqual(result.exit_code, 0)

    @patch("datasets.load_dataset")
    def test_extract_rubrics_no_rubrics_found(self, MockLoadDS):
        items = [{"text": "just text"}]
        MockLoadDS.return_value = iter(items)

        result = self.runner.invoke(main, ["extract-rubrics", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No rubrics found", result.output)

    @patch("datasets.load_dataset")
    @patch("datarecipe.extractors.RubricsAnalyzer")
    def test_extract_rubrics_with_output(self, MockAnalyzer, MockLoadDS):
        items = [{"rubrics": ["Test rubric"]}]
        MockLoadDS.return_value = iter(items)

        mock_result = MagicMock()
        mock_result.summary.return_value = "1 rubric"
        mock_result.structured_templates = []
        MockAnalyzer.return_value.analyze.return_value = mock_result
        MockAnalyzer.return_value.to_dict.return_value = {"rubrics": []}
        MockAnalyzer.return_value.to_yaml_templates.return_value = "templates: []"
        MockAnalyzer.return_value.to_markdown_templates.return_value = "# Templates"

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "rubrics.json")
            result = self.runner.invoke(
                main, ["extract-rubrics", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datasets.load_dataset")
    def test_extract_rubrics_exception(self, MockLoadDS):
        MockLoadDS.side_effect = Exception("Dataset not found")

        result = self.runner.invoke(main, ["extract-rubrics", "bad/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)


class TestExtractPrompts(unittest.TestCase):
    """Test extract-prompts command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datasets.load_dataset")
    @patch("datarecipe.extractors.PromptExtractor")
    def test_extract_prompts_success(self, MockExtractor, MockLoadDS):
        items = [
            {"messages": [{"role": "system", "content": "You are helpful"}]},
            {"messages": [{"role": "user", "content": "Hello"}]},
        ]
        MockLoadDS.return_value = iter(items)

        mock_library = MagicMock()
        mock_library.summary.return_value = "Found 2 prompts"
        MockExtractor.return_value.extract.return_value = mock_library

        result = self.runner.invoke(main, ["extract-prompts", "test/dataset"])
        self.assertEqual(result.exit_code, 0)

    @patch("datasets.load_dataset")
    def test_extract_prompts_no_messages(self, MockLoadDS):
        items = [{"text": "no messages here"}]
        MockLoadDS.return_value = iter(items)

        result = self.runner.invoke(main, ["extract-prompts", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No messages found", result.output)

    @patch("datasets.load_dataset")
    @patch("datarecipe.extractors.PromptExtractor")
    def test_extract_prompts_with_output(self, MockExtractor, MockLoadDS):
        items = [{"messages": [{"role": "system", "content": "Test"}]}]
        MockLoadDS.return_value = iter(items)

        mock_library = MagicMock()
        mock_library.summary.return_value = "1 prompt"
        MockExtractor.return_value.extract.return_value = mock_library
        MockExtractor.return_value.to_dict.return_value = {"prompts": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "prompts.json")
            result = self.runner.invoke(
                main, ["extract-prompts", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))


class TestDetectStrategy(unittest.TestCase):
    """Test detect-strategy command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datasets.load_dataset")
    @patch("datarecipe.analyzers.ContextStrategyDetector")
    def test_detect_strategy_success(self, MockDetector, MockLoadDS):
        items = [
            {"context": "Some context text about science"},
            {"context": "Another context about math"},
        ]
        MockLoadDS.return_value = iter(items)

        mock_result = MagicMock()
        mock_result.summary.return_value = "Detected: synthetic"
        MockDetector.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(main, ["detect-strategy", "test/dataset"])
        self.assertEqual(result.exit_code, 0)

    @patch("datasets.load_dataset")
    def test_detect_strategy_no_contexts(self, MockLoadDS):
        items = [{"id": "1"}]
        MockLoadDS.return_value = iter(items)

        result = self.runner.invoke(main, ["detect-strategy", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No contexts found", result.output)

    @patch("datasets.load_dataset")
    @patch("datarecipe.analyzers.ContextStrategyDetector")
    def test_detect_strategy_with_output(self, MockDetector, MockLoadDS):
        items = [{"context": "Text"}]
        MockLoadDS.return_value = iter(items)

        mock_result = MagicMock()
        mock_result.summary.return_value = "Strategy detected"
        MockDetector.return_value.analyze.return_value = mock_result
        MockDetector.return_value.to_dict.return_value = {"strategy": "synthetic"}

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "strategy.json")
            result = self.runner.invoke(
                main, ["detect-strategy", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datasets.load_dataset")
    @patch("datarecipe.analyzers.ContextStrategyDetector")
    def test_detect_strategy_with_messages(self, MockDetector, MockLoadDS):
        """Test extracting context from messages field."""
        items = [
            {"messages": [{"role": "user", "content": "What is AI?"}]},
        ]
        MockLoadDS.return_value = iter(items)

        mock_result = MagicMock()
        mock_result.summary.return_value = "Detected"
        MockDetector.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(main, ["detect-strategy", "test/dataset"])
        self.assertEqual(result.exit_code, 0)


# ============================================================================
# tools.py: enhanced-guide
# ============================================================================


class TestEnhancedGuide(unittest.TestCase):
    """Test enhanced-guide command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.generators.EnhancedGuideGenerator")
    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    @patch("datasets.load_dataset")
    def test_enhanced_guide_success(self, MockLoadDS, MockTaskType, MockSplitter, MockGuideGen):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        items = [
            {"rubrics": ["Test rubric"], "messages": [{"role": "system", "content": "You are helpful"}], "context": "Some text"},
        ]
        MockLoadDS.return_value = iter(items)

        mock_allocation = MagicMock()
        MockSplitter.return_value.analyze.return_value = mock_allocation

        mock_guide = MagicMock()
        MockGuideGen.return_value.generate.return_value = mock_guide
        MockGuideGen.return_value.to_markdown.return_value = "# Enhanced Guide\nContent..."

        result = self.runner.invoke(
            main, ["enhanced-guide", "test/dataset"]
        )
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.generators.EnhancedGuideGenerator")
    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    @patch("datasets.load_dataset")
    def test_enhanced_guide_with_output(self, MockLoadDS, MockTaskType, MockSplitter, MockGuideGen):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        MockLoadDS.side_effect = Exception("Can't load")

        mock_allocation = MagicMock()
        MockSplitter.return_value.analyze.return_value = mock_allocation

        mock_guide = MagicMock()
        MockGuideGen.return_value.generate.return_value = mock_guide
        MockGuideGen.return_value.to_markdown.return_value = "# Guide"

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "guide.md")
            result = self.runner.invoke(
                main, ["enhanced-guide", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datarecipe.generators.EnhancedGuideGenerator")
    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    def test_enhanced_guide_error(self, MockTaskType, MockSplitter, MockGuideGen):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        MockSplitter.return_value.analyze.side_effect = Exception("Splitter broke")

        result = self.runner.invoke(main, ["enhanced-guide", "test/dataset"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)


# ============================================================================
# tools.py: allocate with output and markdown format
# ============================================================================


class TestAllocateExtended(unittest.TestCase):
    """Extended allocate command tests."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    def test_allocate_markdown_format(self, MockTaskType, MockSplitter):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.DATA_GENERATION = "DATA_GENERATION"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        mock_result = MagicMock()
        mock_result.summary.return_value = "Summary"
        mock_result.to_markdown_table.return_value = "| Task | ... |"
        MockSplitter.return_value.analyze.return_value = mock_result

        result = self.runner.invoke(main, ["allocate", "--format", "markdown"])
        self.assertEqual(result.exit_code, 0)

    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    def test_allocate_with_json_output_file(self, MockTaskType, MockSplitter):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.DATA_GENERATION = "DATA_GENERATION"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        mock_result = MagicMock()
        MockSplitter.return_value.analyze.return_value = mock_result
        MockSplitter.return_value.to_dict.return_value = {"tasks": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "allocate.json")
            result = self.runner.invoke(
                main, ["allocate", "--format", "json", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datarecipe.generators.HumanMachineSplitter")
    @patch("datarecipe.generators.TaskType")
    def test_allocate_with_md_output_file(self, MockTaskType, MockSplitter):
        MockTaskType.CONTEXT_CREATION = "CONTEXT_CREATION"
        MockTaskType.TASK_DESIGN = "TASK_DESIGN"
        MockTaskType.RUBRICS_WRITING = "RUBRICS_WRITING"
        MockTaskType.DATA_GENERATION = "DATA_GENERATION"
        MockTaskType.QUALITY_REVIEW = "QUALITY_REVIEW"

        mock_result = MagicMock()
        mock_result.summary.return_value = "Summary text"
        mock_result.to_markdown_table.return_value = "| Task |"
        MockSplitter.return_value.analyze.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "allocate.md")
            result = self.runner.invoke(
                main, ["allocate", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))


# ============================================================================
# tools.py: batch with file and output
# ============================================================================


class TestBatchFromFile(unittest.TestCase):
    """Test batch command from tools.py with -f file option."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.batch_analyzer.BatchAnalyzer")
    def test_batch_from_file(self, MockBatch):
        mock_result = MagicMock()
        mock_result.results = [MagicMock()]
        mock_result.successful = 1
        mock_result.failed = 0
        mock_result.total_duration_seconds = 2.0
        mock_result.get_failed.return_value = []
        MockBatch.return_value.analyze_from_file.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("ds1\nds2\n")
            tmppath = f.name

        try:
            result = self.runner.invoke(main, ["batch", "-f", tmppath])
            self.assertEqual(result.exit_code, 0)
            MockBatch.return_value.analyze_from_file.assert_called_once_with(tmppath)
        finally:
            os.unlink(tmppath)

    @patch("datarecipe.batch_analyzer.BatchAnalyzer")
    def test_batch_with_output(self, MockBatch):
        mock_result = MagicMock()
        mock_result.results = [MagicMock()]
        mock_result.successful = 1
        mock_result.failed = 0
        mock_result.total_duration_seconds = 1.0
        mock_result.get_failed.return_value = []
        MockBatch.return_value.analyze_batch.return_value = mock_result
        MockBatch.return_value.export_results.return_value = ["file1.yaml"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["batch", "ds1", "-o", tmpdir]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Results exported", result.output)

    @patch("datarecipe.batch_analyzer.BatchAnalyzer")
    def test_batch_with_failures(self, MockBatch):
        failed = MagicMock()
        failed.dataset_id = "bad/ds"
        failed.error = "Not found"

        mock_result = MagicMock()
        mock_result.results = [MagicMock(), failed]
        mock_result.successful = 1
        mock_result.failed = 1
        mock_result.total_duration_seconds = 3.0
        mock_result.get_failed.return_value = [failed]
        MockBatch.return_value.analyze_batch.return_value = mock_result

        result = self.runner.invoke(main, ["batch", "ds1", "bad/ds"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Failed", result.output)

    def test_batch_no_ids_no_file(self):
        result = self.runner.invoke(main, ["batch"])
        self.assertNotEqual(result.exit_code, 0)


# ============================================================================
# analyze.py: deep_guide
# ============================================================================


class TestDeepGuideCommand(unittest.TestCase):
    """Test the deep-guide command."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.analyzers.url_analyzer.deep_analysis_to_markdown")
    @patch("datarecipe.analyzers.llm_url_analyzer.LLMAnalyzer")
    def test_deep_guide_without_llm(self, MockLLMAnalyzer, MockToMd):
        mock_result = MagicMock()
        mock_result.name = "TestDataset"
        mock_result.category.value = "evaluation"
        mock_result.domain = "NLP"
        mock_result.methodology = "Active learning"
        mock_result.key_innovations = ["Innovation 1"]
        mock_result.generation_steps = ["Step 1", "Step 2"]
        mock_result.code_available = True
        mock_result.code_url = "https://github.com/test"
        mock_result.data_available = True
        mock_result.data_url = "https://huggingface.co/test"
        mock_result.paper_url = "https://arxiv.org/abs/1234"
        MockLLMAnalyzer.return_value.analyze.return_value = mock_result
        MockToMd.return_value = "# Deep Guide\nContent"

        result = self.runner.invoke(
            main, ["deep-guide", "https://arxiv.org/abs/1234"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("TestDataset", result.output)

    @patch("datarecipe.analyzers.url_analyzer.deep_analysis_to_markdown")
    @patch("datarecipe.analyzers.llm_url_analyzer.LLMAnalyzer")
    def test_deep_guide_with_output(self, MockLLMAnalyzer, MockToMd):
        mock_result = MagicMock()
        mock_result.name = "Test"
        mock_result.category.value = "generation"
        mock_result.domain = None
        mock_result.methodology = None
        mock_result.key_innovations = []
        mock_result.generation_steps = []
        mock_result.code_available = False
        mock_result.code_url = None
        mock_result.data_available = False
        mock_result.data_url = None
        # No paper_url attribute
        del mock_result.paper_url
        MockLLMAnalyzer.return_value.analyze.return_value = mock_result
        MockToMd.return_value = "# Guide"

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "guide.md")
            result = self.runner.invoke(
                main, ["deep-guide", "https://example.com", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(outpath))

    @patch("datarecipe.analyzers.llm_url_analyzer.LLMAnalyzer")
    def test_deep_guide_value_error(self, MockLLMAnalyzer):
        MockLLMAnalyzer.return_value.analyze.side_effect = ValueError("Invalid URL")
        result = self.runner.invoke(main, ["deep-guide", "https://bad.url"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.analyzers.llm_url_analyzer.LLMAnalyzer")
    def test_deep_guide_generic_error(self, MockLLMAnalyzer):
        MockLLMAnalyzer.return_value.analyze.side_effect = RuntimeError("Connection error")
        result = self.runner.invoke(main, ["deep-guide", "https://bad.url"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.analyzers.url_analyzer.deep_analysis_to_markdown")
    @patch("datarecipe.analyzers.llm_url_analyzer.LLMAnalyzer")
    def test_deep_guide_with_llm_flag(self, MockLLMAnalyzer, MockToMd):
        """Test deep-guide with --llm flag."""
        mock_result = MagicMock()
        mock_result.name = "LLMTest"
        mock_result.category.value = "evaluation"
        mock_result.domain = "NLP"
        mock_result.methodology = "LLM-enhanced"
        mock_result.key_innovations = []
        mock_result.generation_steps = ["Step 1"]
        mock_result.code_available = False
        mock_result.code_url = None
        mock_result.data_available = False
        mock_result.data_url = None
        mock_result.paper_url = None
        MockLLMAnalyzer.return_value.analyze.return_value = mock_result
        MockToMd.return_value = "# LLM Guide"

        result = self.runner.invoke(
            main, ["deep-guide", "https://example.com", "--llm"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("LLMTest", result.output)


# ============================================================================
# analyze.py: guide error paths, export error paths
# ============================================================================


class TestGuideExtended(unittest.TestCase):
    """Extended guide command tests."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_guide_value_error(self, MockAnalyzer):
        MockAnalyzer.return_value.analyze.side_effect = ValueError("Not found")
        result = self.runner.invoke(main, ["guide", "bad/ds"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_guide_generic_error(self, MockAnalyzer):
        MockAnalyzer.return_value.analyze.side_effect = RuntimeError("Network fail")
        result = self.runner.invoke(main, ["guide", "bad/ds"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_export_generic_error(self, MockAnalyzer):
        MockAnalyzer.return_value.analyze.side_effect = RuntimeError("Network")
        result = self.runner.invoke(main, ["export", "bad/ds", "/tmp/out.yaml"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.cli.analyze.display_recipe")
    @patch("datarecipe.cli.analyze.DatasetAnalyzer")
    def test_analyze_output_default_yaml(self, MockAnalyzer, MockDisplay):
        """Test analyze with output file that is not .json or .md (defaults to YAML export)."""
        recipe = MagicMock()
        recipe.name = "test/dataset"
        recipe.to_dict.return_value = {"name": "test/dataset"}
        MockAnalyzer.return_value.analyze.return_value = recipe

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = os.path.join(tmpdir, "recipe.yaml")
            result = self.runner.invoke(
                main, ["analyze", "test/dataset", "-o", outpath]
            )
            self.assertEqual(result.exit_code, 0)
            MockAnalyzer.return_value.export_recipe.assert_called_once()


# ============================================================================
# infra.py: watch with results, cache list with entries
# ============================================================================


class TestWatchExtended(unittest.TestCase):
    """Extended watch command tests."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.triggers.RadarWatcher")
    @patch("datarecipe.triggers.TriggerConfig")
    def test_watch_once_with_results(self, MockConfig, MockWatcher):
        mock_config = MagicMock(orgs=[], categories=[], min_downloads=0)
        MockConfig.return_value = mock_config

        MockWatcher.return_value.check_once.return_value = [
            {"report": "intel_report_2025.json", "datasets_analyzed": 5, "datasets_failed": 1}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, ["watch", tmpdir, "--once"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("intel_report_2025", result.output)

    @patch("datarecipe.triggers.RadarWatcher")
    @patch("datarecipe.triggers.TriggerConfig")
    def test_watch_once_with_config_file(self, MockConfig, MockWatcher):
        MockConfig.from_yaml.return_value = MagicMock(
            orgs=["Anthropic"], categories=["eval"], min_downloads=100
        )
        MockWatcher.return_value.check_once.return_value = []

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("orgs: [Anthropic]\n")
            config_path = f.name

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                result = self.runner.invoke(
                    main, ["watch", tmpdir, "--once", "--config", config_path]
                )
                self.assertEqual(result.exit_code, 0)
                MockConfig.from_yaml.assert_called_once_with(config_path)
            finally:
                os.unlink(config_path)

    @patch("datarecipe.triggers.RadarWatcher")
    @patch("datarecipe.triggers.TriggerConfig")
    def test_watch_with_org_and_category_filters(self, MockConfig, MockWatcher):
        mock_config = MagicMock(
            orgs=["OpenAI"], categories=["qa"], min_downloads=500
        )
        MockConfig.return_value = mock_config
        MockWatcher.return_value.check_once.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(
                main, ["watch", tmpdir, "--once", "--orgs", "OpenAI", "--categories", "qa", "--min-downloads", "500"]
            )
            self.assertEqual(result.exit_code, 0)


class TestCacheListExtended(unittest.TestCase):
    """Test cache --list with actual entries."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.cache.AnalysisCache")
    def test_cache_list_with_entries(self, MockCache):
        entry1 = MagicMock()
        entry1.dataset_id = "test/ds1"
        entry1.dataset_type = "evaluation"
        entry1.sample_count = 500
        entry1.created_at = "2025-06-01T12:00:00"
        entry1.is_expired.return_value = False

        entry2 = MagicMock()
        entry2.dataset_id = "test/ds2"
        entry2.dataset_type = None
        entry2.sample_count = 100
        entry2.created_at = "2025-05-01T12:00:00"
        entry2.is_expired.return_value = True

        MockCache.return_value.list_entries.return_value = [entry1, entry2]

        result = self.runner.invoke(main, ["cache", "--list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("test/ds1", result.output)
        self.assertIn("test/ds2", result.output)


class TestKnowledgeExtended(unittest.TestCase):
    """Extended knowledge command tests."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_report(self, MockKB):
        MockKB.return_value.export_report.return_value = "/tmp/kb_report.md"

        result = self.runner.invoke(main, ["knowledge", "--report"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("kb_report.md", result.output)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_benchmarks_with_data(self, MockKB):
        mock_bench = MagicMock()
        mock_bench.avg_total_cost = 5000
        mock_bench.min_cost = 1000
        mock_bench.max_cost = 10000
        mock_bench.avg_human_percentage = 60
        mock_bench.datasets = ["ds1", "ds2"]

        MockKB.return_value.trends.get_all_benchmarks.return_value = {
            "evaluation": mock_bench
        }

        result = self.runner.invoke(main, ["knowledge", "--benchmarks"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("evaluation", result.output)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_trends_with_data(self, MockKB):
        MockKB.return_value.trends.get_trend_summary.return_value = {
            "datasets_analyzed": 10,
            "total_cost": 50000,
            "avg_cost_per_dataset": 5000,
            "type_distribution": {"evaluation": 5, "preference": 3, "swe": 2},
        }

        result = self.runner.invoke(main, ["knowledge", "--trends"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("10", result.output)

    @patch("datarecipe.knowledge.KnowledgeBase")
    def test_knowledge_recommend_with_patterns(self, MockKB):
        MockKB.return_value.get_recommendations.return_value = {
            "cost_estimate": {
                "avg_total": 8000,
                "range": [3000, 12000],
                "avg_human_percentage": 70,
                "based_on": 5,
            },
            "common_patterns": [
                {"pattern": "rubric_based", "type": "evaluation"},
            ],
            "suggested_fields": ["messages", "rubrics", "metadata"],
        }

        result = self.runner.invoke(main, ["knowledge", "--recommend", "evaluation"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("8,000", result.output)
        self.assertIn("rubric_based", result.output)
        self.assertIn("messages", result.output)


# ============================================================================
# tools.py: cost error handling
# ============================================================================


class TestCostExtended(unittest.TestCase):
    """Extended cost command tests."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_cost_analyze_error(self, MockAnalyzer):
        MockAnalyzer.return_value.analyze.side_effect = RuntimeError("Dataset not accessible")
        result = self.runner.invoke(main, ["cost", "bad/dataset"])
        self.assertNotEqual(result.exit_code, 0)

    @patch("datarecipe.cost_calculator.CostCalculator")
    @patch("datarecipe.cli.tools.DatasetAnalyzer")
    def test_cost_with_examples_option(self, MockAnalyzer, MockCalc):
        recipe = MagicMock()
        recipe.num_examples = 5000
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
        mock_breakdown.assumptions = []
        MockCalc.return_value.estimate_from_recipe.return_value = mock_breakdown

        result = self.runner.invoke(
            main, ["cost", "test/dataset", "--examples", "20000"]
        )
        self.assertEqual(result.exit_code, 0)
        # Verify target size was used
        call_args = MockCalc.return_value.estimate_from_recipe.call_args
        self.assertEqual(call_args[0][1], 20000)


if __name__ == "__main__":
    unittest.main()
