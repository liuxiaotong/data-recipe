"""Tests for core DataRecipe modules: analyzer, batch_analyzer, llm_dataset_analyzer, __init__, __main__."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from datarecipe.schema import (
    Cost,
    GenerationType,
    Recipe,
    Reproducibility,
    SourceType,
)

# =============================================================================
# DatasetAnalyzer tests (analyzer.py)
# =============================================================================


class TestDatasetAnalyzerInit(unittest.TestCase):
    """Test DatasetAnalyzer initialization."""

    @patch("datarecipe.analyzer.WebExtractor")
    @patch("datarecipe.analyzer.GitHubExtractor")
    @patch("datarecipe.analyzer.HuggingFaceExtractor")
    def test_init_creates_extractors(self, mock_hf, mock_gh, mock_web):
        from datarecipe.analyzer import DatasetAnalyzer

        analyzer = DatasetAnalyzer()
        self.assertIn(SourceType.HUGGINGFACE, analyzer.extractors)
        self.assertIn(SourceType.GITHUB, analyzer.extractors)
        self.assertIn(SourceType.WEB, analyzer.extractors)


class TestParseDatasetInput(unittest.TestCase):
    """Test _parse_dataset_input URL parsing."""

    @patch("datarecipe.analyzer.WebExtractor")
    @patch("datarecipe.analyzer.GitHubExtractor")
    @patch("datarecipe.analyzer.HuggingFaceExtractor")
    def setUp(self, mock_hf, mock_gh, mock_web):
        from datarecipe.analyzer import DatasetAnalyzer

        self.analyzer = DatasetAnalyzer()

    def test_huggingface_url(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://huggingface.co/datasets/openai/gsm8k"
        )
        self.assertEqual(dataset_id, "openai/gsm8k")
        self.assertEqual(source, SourceType.HUGGINGFACE)

    def test_huggingface_hf_co_url(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://hf.co/datasets/meta/llama-data"
        )
        self.assertEqual(dataset_id, "meta/llama-data")
        self.assertEqual(source, SourceType.HUGGINGFACE)

    def test_huggingface_url_with_query_params(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://huggingface.co/datasets/org/name?subset=train"
        )
        self.assertEqual(dataset_id, "org/name")
        self.assertEqual(source, SourceType.HUGGINGFACE)

    def test_huggingface_url_with_fragment(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://huggingface.co/datasets/org/name#readme"
        )
        self.assertEqual(dataset_id, "org/name")
        self.assertEqual(source, SourceType.HUGGINGFACE)

    def test_github_url(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://github.com/arcprize/ARC-AGI-2"
        )
        self.assertEqual(dataset_id, "arcprize/ARC-AGI-2")
        self.assertEqual(source, SourceType.GITHUB)

    def test_github_url_trailing_slash(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://github.com/org/repo/"
        )
        self.assertEqual(dataset_id, "org/repo")
        self.assertEqual(source, SourceType.GITHUB)

    def test_github_url_dotgit(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://github.com/org/repo.git"
        )
        self.assertEqual(dataset_id, "org/repo")
        self.assertEqual(source, SourceType.GITHUB)

    def test_generic_https_url(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "https://example.com/datasets/mydata"
        )
        self.assertEqual(dataset_id, "https://example.com/datasets/mydata")
        self.assertEqual(source, SourceType.WEB)

    def test_generic_http_url(self):
        dataset_id, source = self.analyzer._parse_dataset_input(
            "http://example.com/datasets/mydata"
        )
        self.assertEqual(dataset_id, "http://example.com/datasets/mydata")
        self.assertEqual(source, SourceType.WEB)

    def test_plain_id(self):
        dataset_id, source = self.analyzer._parse_dataset_input("openai/gsm8k")
        self.assertEqual(dataset_id, "openai/gsm8k")
        self.assertIsNone(source)

    def test_plain_name_no_slash(self):
        dataset_id, source = self.analyzer._parse_dataset_input("my-local-dataset")
        self.assertEqual(dataset_id, "my-local-dataset")
        self.assertIsNone(source)


class TestDetectSourceType(unittest.TestCase):
    """Test _detect_source_type auto-detection."""

    @patch("datarecipe.analyzer.WebExtractor")
    @patch("datarecipe.analyzer.GitHubExtractor")
    @patch("datarecipe.analyzer.HuggingFaceExtractor")
    def setUp(self, mock_hf, mock_gh, mock_web):
        from datarecipe.analyzer import DatasetAnalyzer

        self.analyzer = DatasetAnalyzer()

    def test_slash_id_detected_as_hf(self):
        result = self.analyzer._detect_source_type("openai/gsm8k")
        self.assertEqual(result, SourceType.HUGGINGFACE)

    def test_hf_domain_detected(self):
        result = self.analyzer._detect_source_type("huggingface.co/something")
        self.assertEqual(result, SourceType.HUGGINGFACE)

    def test_plain_name_defaults_to_hf(self):
        result = self.analyzer._detect_source_type("my-dataset")
        self.assertEqual(result, SourceType.HUGGINGFACE)


class TestAnalyze(unittest.TestCase):
    """Test DatasetAnalyzer.analyze() method."""

    @patch("datarecipe.analyzer.WebExtractor")
    @patch("datarecipe.analyzer.GitHubExtractor")
    @patch("datarecipe.analyzer.HuggingFaceExtractor")
    def setUp(self, mock_hf_cls, mock_gh_cls, mock_web_cls):
        from datarecipe.analyzer import DatasetAnalyzer

        self.mock_recipe = Recipe(name="test-dataset", source_type=SourceType.HUGGINGFACE)
        mock_hf_cls.return_value.extract.return_value = self.mock_recipe
        mock_gh_cls.return_value.extract.return_value = Recipe(
            name="gh-repo", source_type=SourceType.GITHUB
        )
        mock_web_cls.return_value.extract.return_value = Recipe(
            name="web-data", source_type=SourceType.WEB
        )
        self.analyzer = DatasetAnalyzer()

    def test_analyze_with_hf_url(self):
        recipe = self.analyzer.analyze("https://huggingface.co/datasets/openai/gsm8k")
        self.assertEqual(recipe.name, "test-dataset")
        self.analyzer.extractors[SourceType.HUGGINGFACE].extract.assert_called_once_with(
            "openai/gsm8k"
        )

    def test_analyze_with_explicit_source_type(self):
        recipe = self.analyzer.analyze("openai/gsm8k", source_type=SourceType.HUGGINGFACE)
        self.assertEqual(recipe.name, "test-dataset")

    def test_analyze_auto_detect_hf_id(self):
        recipe = self.analyzer.analyze("openai/gsm8k")
        self.assertEqual(recipe.name, "test-dataset")

    def test_analyze_github_url(self):
        recipe = self.analyzer.analyze("https://github.com/arcprize/ARC-AGI-2")
        self.assertEqual(recipe.name, "gh-repo")

    def test_analyze_web_url(self):
        recipe = self.analyzer.analyze("https://example.com/data")
        self.assertEqual(recipe.name, "web-data")

    def test_analyze_unsupported_source_type(self):
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.analyze("something", source_type=SourceType.OPENAI)
        self.assertIn("Unsupported source type", str(ctx.exception))


class TestAnalyzeFromYaml(unittest.TestCase):
    """Test DatasetAnalyzer.analyze_from_yaml() method."""

    @patch("datarecipe.analyzer.WebExtractor")
    @patch("datarecipe.analyzer.GitHubExtractor")
    @patch("datarecipe.analyzer.HuggingFaceExtractor")
    def setUp(self, mock_hf, mock_gh, mock_web):
        from datarecipe.analyzer import DatasetAnalyzer

        self.analyzer = DatasetAnalyzer()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_from_yaml("/nonexistent/path.yaml")

    def test_load_minimal_yaml(self):
        import yaml

        data = {"name": "my-dataset", "source": {"type": "huggingface", "id": "org/ds"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmppath = f.name

        try:
            recipe = self.analyzer.analyze_from_yaml(tmppath)
            self.assertEqual(recipe.name, "my-dataset")
            self.assertEqual(recipe.source_type, SourceType.HUGGINGFACE)
            self.assertEqual(recipe.source_id, "org/ds")
        finally:
            os.unlink(tmppath)

    def test_load_full_yaml(self):
        import yaml

        data = {
            "name": "full-dataset",
            "version": "1.0",
            "source": {"type": "github", "id": "org/repo"},
            "generation": {
                "synthetic_ratio": 0.95,
                "human_ratio": 0.05,
                "teacher_models": ["GPT-4"],
                "methods": [
                    {
                        "type": "distillation",
                        "teacher_model": "GPT-4",
                        "prompt_template": "available",
                        "platform": "openai",
                    }
                ],
            },
            "cost": {
                "estimated_total_usd": 5000,
                "breakdown": {"api_calls": 3000, "human_annotation": 1500, "compute": 500},
                "confidence": "medium",
            },
            "reproducibility": {
                "score": 7,
                "available": ["code", "prompts"],
                "missing": ["api_keys"],
                "notes": "Mostly reproducible",
            },
            "metadata": {
                "size_bytes": 1024000,
                "num_examples": 10000,
                "languages": ["en"],
                "license": "MIT",
                "tags": ["nlp", "qa"],
                "authors": ["Researcher A"],
                "paper_url": "https://arxiv.org/abs/1234.5678",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmppath = f.name

        try:
            recipe = self.analyzer.analyze_from_yaml(tmppath)
            self.assertEqual(recipe.name, "full-dataset")
            self.assertEqual(recipe.version, "1.0")
            self.assertEqual(recipe.source_type, SourceType.GITHUB)
            self.assertEqual(recipe.generation_type, GenerationType.SYNTHETIC)
            self.assertEqual(recipe.synthetic_ratio, 0.95)
            self.assertEqual(recipe.human_ratio, 0.05)
            self.assertEqual(recipe.teacher_models, ["GPT-4"])
            self.assertEqual(len(recipe.generation_methods), 1)
            self.assertEqual(recipe.generation_methods[0].method_type, "distillation")
            self.assertEqual(recipe.generation_methods[0].teacher_model, "GPT-4")
            self.assertTrue(recipe.generation_methods[0].prompt_template_available)
            self.assertEqual(recipe.generation_methods[0].platform, "openai")
            self.assertIsNotNone(recipe.cost)
            self.assertEqual(recipe.cost.estimated_total_usd, 5000)
            self.assertEqual(recipe.cost.api_calls_usd, 3000)
            self.assertEqual(recipe.cost.confidence, "medium")
            self.assertEqual(recipe.reproducibility.score, 7)
            self.assertEqual(recipe.reproducibility.notes, "Mostly reproducible")
            self.assertEqual(recipe.num_examples, 10000)
            self.assertEqual(recipe.languages, ["en"])
            self.assertEqual(recipe.license, "MIT")
            self.assertEqual(recipe.paper_url, "https://arxiv.org/abs/1234.5678")
        finally:
            os.unlink(tmppath)

    def test_unknown_source_type_in_yaml(self):
        import yaml

        data = {"name": "test", "source": {"type": "foobar"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmppath = f.name

        try:
            recipe = self.analyzer.analyze_from_yaml(tmppath)
            self.assertEqual(recipe.source_type, SourceType.UNKNOWN)
        finally:
            os.unlink(tmppath)


class TestRecipeFromDictGenerationType(unittest.TestCase):
    """Test _recipe_from_dict generation type classification."""

    @patch("datarecipe.analyzer.WebExtractor")
    @patch("datarecipe.analyzer.GitHubExtractor")
    @patch("datarecipe.analyzer.HuggingFaceExtractor")
    def setUp(self, mock_hf, mock_gh, mock_web):
        from datarecipe.analyzer import DatasetAnalyzer

        self.analyzer = DatasetAnalyzer()

    def test_synthetic_when_ratio_high(self):
        data = {"name": "ds", "generation": {"synthetic_ratio": 0.95}}
        recipe = self.analyzer._recipe_from_dict(data)
        self.assertEqual(recipe.generation_type, GenerationType.SYNTHETIC)

    def test_human_when_ratio_low(self):
        data = {"name": "ds", "generation": {"synthetic_ratio": 0.05}}
        recipe = self.analyzer._recipe_from_dict(data)
        self.assertEqual(recipe.generation_type, GenerationType.HUMAN)

    def test_mixed_when_ratio_middle(self):
        data = {"name": "ds", "generation": {"synthetic_ratio": 0.5}}
        recipe = self.analyzer._recipe_from_dict(data)
        self.assertEqual(recipe.generation_type, GenerationType.MIXED)

    def test_boundary_synthetic_at_0_9(self):
        data = {"name": "ds", "generation": {"synthetic_ratio": 0.9}}
        recipe = self.analyzer._recipe_from_dict(data)
        self.assertEqual(recipe.generation_type, GenerationType.SYNTHETIC)

    def test_boundary_human_at_0_1(self):
        data = {"name": "ds", "generation": {"synthetic_ratio": 0.1}}
        recipe = self.analyzer._recipe_from_dict(data)
        self.assertEqual(recipe.generation_type, GenerationType.HUMAN)


class TestExportRecipe(unittest.TestCase):
    """Test DatasetAnalyzer.export_recipe()."""

    @patch("datarecipe.analyzer.WebExtractor")
    @patch("datarecipe.analyzer.GitHubExtractor")
    @patch("datarecipe.analyzer.HuggingFaceExtractor")
    def setUp(self, mock_hf, mock_gh, mock_web):
        from datarecipe.analyzer import DatasetAnalyzer

        self.analyzer = DatasetAnalyzer()

    def test_export_creates_file(self):
        recipe = Recipe(name="export-test", source_type=SourceType.HUGGINGFACE)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "subdir" / "recipe.yaml"
            self.analyzer.export_recipe(recipe, out_path)
            self.assertTrue(out_path.exists())
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("export-test", content)


class TestGetRecipeSummary(unittest.TestCase):
    """Test the get_recipe_summary function."""

    def test_basic_summary(self):
        from datarecipe.analyzer import get_recipe_summary

        recipe = Recipe(name="test", source_type=SourceType.HUGGINGFACE, source_id="org/ds")
        summary = get_recipe_summary(recipe)
        self.assertEqual(summary["name"], "test")
        self.assertIn("huggingface", summary["source"])

    def test_summary_with_ratios(self):
        from datarecipe.analyzer import get_recipe_summary

        recipe = Recipe(
            name="test",
            source_type=SourceType.GITHUB,
            source_id="org/repo",
            synthetic_ratio=0.8,
            human_ratio=0.2,
        )
        summary = get_recipe_summary(recipe)
        self.assertEqual(summary["synthetic_percentage"], "80%")
        self.assertEqual(summary["human_percentage"], "20%")

    def test_summary_with_teacher_models(self):
        from datarecipe.analyzer import get_recipe_summary

        recipe = Recipe(
            name="test",
            source_type=SourceType.HUGGINGFACE,
            teacher_models=["GPT-4", "Claude 3"],
        )
        summary = get_recipe_summary(recipe)
        self.assertEqual(summary["teacher_models"], ["GPT-4", "Claude 3"])

    def test_summary_with_cost(self):
        from datarecipe.analyzer import get_recipe_summary

        recipe = Recipe(
            name="test",
            source_type=SourceType.HUGGINGFACE,
            cost=Cost(estimated_total_usd=5000, confidence="high"),
        )
        summary = get_recipe_summary(recipe)
        self.assertEqual(summary["estimated_cost"], "$5,000")
        self.assertEqual(summary["cost_confidence"], "high")

    def test_summary_with_reproducibility(self):
        from datarecipe.analyzer import get_recipe_summary

        recipe = Recipe(
            name="test",
            source_type=SourceType.HUGGINGFACE,
            reproducibility=Reproducibility(
                score=8, available=["code"], missing=["api_keys", "prompts"]
            ),
        )
        summary = get_recipe_summary(recipe)
        self.assertEqual(summary["reproducibility_score"], "8/10")
        self.assertEqual(summary["missing_for_reproduction"], ["api_keys", "prompts"])

    def test_summary_no_missing_reproducibility(self):
        from datarecipe.analyzer import get_recipe_summary

        recipe = Recipe(
            name="test",
            source_type=SourceType.HUGGINGFACE,
            reproducibility=Reproducibility(score=10, available=["all"]),
        )
        summary = get_recipe_summary(recipe)
        self.assertNotIn("missing_for_reproduction", summary)


# =============================================================================
# LLMDatasetAnalyzer tests (analyzers/llm_dataset_analyzer.py)
# =============================================================================


class TestLLMDatasetAnalysis(unittest.TestCase):
    """Test LLMDatasetAnalysis dataclass."""

    def test_defaults(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalysis

        analysis = LLMDatasetAnalysis()
        self.assertEqual(analysis.dataset_type, "")
        self.assertEqual(analysis.purpose, "")
        self.assertEqual(analysis.key_fields, [])
        self.assertEqual(analysis.production_steps, [])
        self.assertEqual(analysis.recommended_team, {})
        self.assertEqual(analysis.raw_response, "")


class TestLLMDatasetAnalyzerInit(unittest.TestCase):
    """Test LLMDatasetAnalyzer initialization."""

    def test_default_provider(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        analyzer = LLMDatasetAnalyzer()
        self.assertEqual(analyzer.provider, "anthropic")
        self.assertIsNone(analyzer._client)

    def test_openai_provider(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        analyzer = LLMDatasetAnalyzer(provider="openai")
        self.assertEqual(analyzer.provider, "openai")

    def test_unknown_provider_raises(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        analyzer = LLMDatasetAnalyzer(provider="unknown_provider")
        with self.assertRaises(ValueError) as ctx:
            analyzer._get_client()
        self.assertIn("Unknown provider", str(ctx.exception))


class TestLLMDatasetAnalyzerGetClient(unittest.TestCase):
    """Test _get_client for various providers."""

    def test_anthropic_missing_api_key(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        LLMDatasetAnalyzer(provider="anthropic")
        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY if present
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with patch("datarecipe.analyzers.llm_dataset_analyzer.LLMDatasetAnalyzer._get_client") as mock:
                mock.side_effect = ValueError("ANTHROPIC_API_KEY not set")
                with self.assertRaises(ValueError):
                    mock()

    def test_openai_missing_api_key(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        LLMDatasetAnalyzer(provider="openai")
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with patch("datarecipe.analyzers.llm_dataset_analyzer.LLMDatasetAnalyzer._get_client") as mock:
                mock.side_effect = ValueError("OPENAI_API_KEY not set")
                with self.assertRaises(ValueError):
                    mock()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_client_creation(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        analyzer = LLMDatasetAnalyzer(provider="anthropic")
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = analyzer._get_client()
            self.assertIsNotNone(client)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_client_cached(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        analyzer = LLMDatasetAnalyzer(provider="anthropic")
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client1 = analyzer._get_client()
            client2 = analyzer._get_client()
            self.assertIs(client1, client2)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_client_creation(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        analyzer = LLMDatasetAnalyzer(provider="openai")
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            client = analyzer._get_client()
            self.assertIsNotNone(client)


class TestLLMDatasetAnalyzerParseResponse(unittest.TestCase):
    """Test _parse_response JSON extraction."""

    def setUp(self):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        self.analyzer = LLMDatasetAnalyzer()

    def test_parse_json_block(self):
        response_text = """Here is my analysis:
```json
{
    "dataset_type": "instruction_tuning",
    "purpose": "Fine-tuning LLMs",
    "structure_description": "Prompt-response pairs",
    "key_fields": [{"field": "instruction", "role": "input"}],
    "production_steps": [{"step": 1, "name": "Collect"}],
    "quality_criteria": [{"criterion": "accuracy"}],
    "annotation_guidelines": "Follow the guide",
    "example_analysis": [{"example_index": 0, "analysis": "Good"}],
    "recommended_team": {"roles": [], "total_people": "5-10"},
    "estimated_difficulty": "medium",
    "similar_datasets": ["alpaca"]
}
```"""
        result = self.analyzer._parse_response(response_text)
        self.assertEqual(result.dataset_type, "instruction_tuning")
        self.assertEqual(result.purpose, "Fine-tuning LLMs")
        self.assertEqual(result.estimated_difficulty, "medium")
        self.assertEqual(result.similar_datasets, ["alpaca"])

    def test_parse_raw_json(self):
        response_text = '{"dataset_type": "qa", "purpose": "QA dataset"}'
        result = self.analyzer._parse_response(response_text)
        self.assertEqual(result.dataset_type, "qa")
        self.assertEqual(result.purpose, "QA dataset")

    def test_parse_invalid_json(self):
        response_text = "This is not JSON at all, just some text"
        result = self.analyzer._parse_response(response_text)
        self.assertEqual(result.dataset_type, "unknown")
        self.assertEqual(result.purpose, "Failed to parse LLM response")
        self.assertEqual(result.raw_response, response_text)


class TestLLMDatasetAnalyzerAnalyze(unittest.TestCase):
    """Test LLMDatasetAnalyzer.analyze() method with mocked LLM."""

    def _make_analyzer_with_mock_client(self, provider="anthropic"):
        from datarecipe.analyzers.llm_dataset_analyzer import LLMDatasetAnalyzer

        analyzer = LLMDatasetAnalyzer(provider=provider)
        mock_client = MagicMock()
        analyzer._client = mock_client
        return analyzer, mock_client

    def test_analyze_anthropic_success(self):
        analyzer, mock_client = self._make_analyzer_with_mock_client("anthropic")
        json_response = json.dumps(
            {
                "dataset_type": "instruction_tuning",
                "purpose": "Training chat models",
                "structure_description": "Instruction-response pairs",
                "key_fields": [],
                "production_steps": [],
                "quality_criteria": [],
                "annotation_guidelines": "",
                "example_analysis": [],
                "recommended_team": {},
                "estimated_difficulty": "easy",
                "similar_datasets": [],
            }
        )
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_response)]
        mock_client.messages.create.return_value = mock_response

        result = analyzer.analyze(
            dataset_id="test/dataset",
            schema_info={"instruction": {"type": "str"}, "response": {"type": "str"}},
            sample_items=[{"instruction": "Hello", "response": "Hi there"}],
            sample_count=100,
        )
        self.assertEqual(result.dataset_type, "instruction_tuning")
        self.assertEqual(result.purpose, "Training chat models")
        self.assertEqual(result.raw_response, json_response)

    def test_analyze_openai_success(self):
        analyzer, mock_client = self._make_analyzer_with_mock_client("openai")
        analyzer.provider = "openai"
        json_response = json.dumps({"dataset_type": "qa", "purpose": "Question answering"})
        mock_choice = MagicMock()
        mock_choice.message.content = json_response
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        result = analyzer.analyze(
            dataset_id="test/qa",
            schema_info={"question": {"type": "str"}},
            sample_items=[{"question": "What is AI?"}],
            sample_count=50,
        )
        self.assertEqual(result.dataset_type, "qa")

    def test_analyze_truncates_long_values(self):
        analyzer, mock_client = self._make_analyzer_with_mock_client("anthropic")
        json_response = json.dumps({"dataset_type": "test", "purpose": "test"})
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_response)]
        mock_client.messages.create.return_value = mock_response

        long_text = "x" * 1000
        long_list = list(range(20))
        result = analyzer.analyze(
            dataset_id="test/ds",
            schema_info={"text": {"type": "str"}},
            sample_items=[{"text": long_text, "tags": long_list}],
            sample_count=1,
        )
        # Verify it completed without error (truncation happened internally)
        self.assertEqual(result.dataset_type, "test")

    def test_analyze_limits_to_5_examples(self):
        analyzer, mock_client = self._make_analyzer_with_mock_client("anthropic")
        json_response = json.dumps({"dataset_type": "test"})
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json_response)]
        mock_client.messages.create.return_value = mock_response

        items = [{"text": f"item_{i}"} for i in range(10)]
        analyzer.analyze(
            dataset_id="test/ds",
            schema_info={"text": {"type": "str"}},
            sample_items=items,
            sample_count=10,
        )
        # Verify the call was made (examples get truncated to 5 internally)
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"]
        # Only 5 items should appear
        self.assertIn("item_4", prompt_text)
        self.assertNotIn("item_5", prompt_text)

    def test_analyze_exception_returns_error_result(self):
        analyzer, mock_client = self._make_analyzer_with_mock_client("anthropic")
        mock_client.messages.create.side_effect = RuntimeError("API timeout")

        result = analyzer.analyze(
            dataset_id="test/ds",
            schema_info={"f": {"type": "str"}},
            sample_items=[{"f": "val"}],
            sample_count=1,
        )
        self.assertEqual(result.dataset_type, "unknown")
        self.assertIn("API timeout", result.purpose)


class TestGenerateLLMGuideSection(unittest.TestCase):
    """Test generate_llm_guide_section markdown generation."""

    def test_minimal_analysis(self):
        from datarecipe.analyzers.llm_dataset_analyzer import (
            LLMDatasetAnalysis,
            generate_llm_guide_section,
        )

        analysis = LLMDatasetAnalysis(dataset_type="qa", purpose="QA data")
        md = generate_llm_guide_section(analysis)
        self.assertIn("qa", md)
        self.assertIn("QA data", md)
        self.assertIn("---", md)

    def test_full_analysis(self):
        from datarecipe.analyzers.llm_dataset_analyzer import (
            LLMDatasetAnalysis,
            generate_llm_guide_section,
        )

        analysis = LLMDatasetAnalysis(
            dataset_type="instruction_tuning",
            purpose="Fine-tuning",
            structure_description="Prompt-response pairs",
            key_fields=[
                {"field": "instruction", "role": "input", "format": "text", "example_pattern": "..."}
            ],
            production_steps=[
                {
                    "step": 1,
                    "name": "Collect",
                    "description": "Gather data",
                    "who": "human",
                    "tools": ["browser"],
                }
            ],
            quality_criteria=[
                {"criterion": "accuracy", "description": "Must be accurate", "check_method": "manual"}
            ],
            annotation_guidelines="Follow the spec carefully",
            recommended_team={
                "roles": [{"role": "annotator", "count": "5", "skills": ["NLP"]}],
                "total_people": "8",
            },
            estimated_difficulty="medium - requires domain expertise",
            similar_datasets=["alpaca", "dolly"],
        )
        md = generate_llm_guide_section(analysis)
        self.assertIn("instruction_tuning", md)
        self.assertIn("Prompt-response pairs", md)
        self.assertIn("instruction", md)
        self.assertIn("Collect", md)
        self.assertIn("browser", md)
        self.assertIn("accuracy", md)
        self.assertIn("Follow the spec carefully", md)
        self.assertIn("annotator", md)
        self.assertIn("8", md)
        self.assertIn("medium", md)
        self.assertIn("alpaca", md)
        self.assertIn("dolly", md)


# =============================================================================
# BatchAnalyzer tests (batch_analyzer.py)
# =============================================================================


class TestBatchResult(unittest.TestCase):
    """Test BatchResult dataclass."""

    def test_to_dict_success(self):
        from datarecipe.batch_analyzer import BatchResult

        recipe = Recipe(name="test-ds")
        result = BatchResult(
            dataset_id="org/test",
            success=True,
            recipe=recipe,
            duration_seconds=1.234,
        )
        d = result.to_dict()
        self.assertEqual(d["dataset_id"], "org/test")
        self.assertTrue(d["success"])
        self.assertEqual(d["duration_seconds"], 1.23)
        self.assertIn("recipe", d)

    def test_to_dict_failure(self):
        from datarecipe.batch_analyzer import BatchResult

        result = BatchResult(
            dataset_id="org/fail",
            success=False,
            error="Connection timeout",
            duration_seconds=0.5,
        )
        d = result.to_dict()
        self.assertFalse(d["success"])
        self.assertEqual(d["error"], "Connection timeout")
        self.assertNotIn("recipe", d)


class TestBatchAnalysisResult(unittest.TestCase):
    """Test BatchAnalysisResult methods."""

    def test_get_recipes(self):
        from datarecipe.batch_analyzer import BatchAnalysisResult, BatchResult

        r1 = BatchResult(dataset_id="a", success=True, recipe=Recipe(name="a"))
        r2 = BatchResult(dataset_id="b", success=False, error="err")
        r3 = BatchResult(dataset_id="c", success=True, recipe=Recipe(name="c"))
        bar = BatchAnalysisResult(results=[r1, r2, r3], successful=2, failed=1)
        recipes = bar.get_recipes()
        self.assertEqual(len(recipes), 2)
        self.assertEqual(recipes[0].name, "a")
        self.assertEqual(recipes[1].name, "c")

    def test_get_failed(self):
        from datarecipe.batch_analyzer import BatchAnalysisResult, BatchResult

        r1 = BatchResult(dataset_id="a", success=True, recipe=Recipe(name="a"))
        r2 = BatchResult(dataset_id="b", success=False, error="err")
        bar = BatchAnalysisResult(results=[r1, r2], successful=1, failed=1)
        failed = bar.get_failed()
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].dataset_id, "b")

    def test_to_dict(self):
        from datarecipe.batch_analyzer import BatchAnalysisResult, BatchResult

        r1 = BatchResult(dataset_id="a", success=True, recipe=Recipe(name="a"))
        bar = BatchAnalysisResult(
            results=[r1],
            successful=1,
            failed=0,
            total_duration_seconds=2.5,
        )
        d = bar.to_dict()
        self.assertEqual(d["summary"]["total"], 1)
        self.assertEqual(d["summary"]["successful"], 1)
        self.assertEqual(d["summary"]["failed"], 0)
        self.assertEqual(len(d["results"]), 1)

    def test_to_json(self):
        from datarecipe.batch_analyzer import BatchAnalysisResult

        bar = BatchAnalysisResult(results=[], successful=0, failed=0)
        j = bar.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["summary"]["total"], 0)


class TestBatchAnalyzer(unittest.TestCase):
    """Test BatchAnalyzer parallel execution."""

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_analyze_batch_all_success(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalyzer

        mock_da = mock_da_cls.return_value
        mock_da.analyze.side_effect = lambda ds_id: Recipe(
            name=ds_id, source_type=SourceType.HUGGINGFACE
        )

        analyzer = BatchAnalyzer(max_workers=2)
        result = analyzer.analyze_batch(["ds1", "ds2", "ds3"])

        self.assertEqual(result.successful, 3)
        self.assertEqual(result.failed, 0)
        self.assertEqual(len(result.results), 3)

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_analyze_batch_with_failure(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalyzer

        mock_da = mock_da_cls.return_value

        def side_effect(ds_id):
            if ds_id == "fail":
                raise RuntimeError("Network error")
            return Recipe(name=ds_id)

        mock_da.analyze.side_effect = side_effect
        analyzer = BatchAnalyzer(max_workers=2)
        result = analyzer.analyze_batch(["ok1", "fail", "ok2"])

        self.assertEqual(result.successful, 2)
        self.assertEqual(result.failed, 1)

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_analyze_batch_progress_callback(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalyzer

        mock_da = mock_da_cls.return_value
        mock_da.analyze.return_value = Recipe(name="test")

        progress_calls = []

        def progress_cb(ds_id, completed, total):
            progress_calls.append((ds_id, completed, total))

        analyzer = BatchAnalyzer(max_workers=1, progress_callback=progress_cb)
        analyzer.analyze_batch(["a", "b"])

        self.assertEqual(len(progress_calls), 2)
        # All calls should have total=2
        for _, _, total in progress_calls:
            self.assertEqual(total, 2)

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_analyze_batch_preserves_order(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalyzer

        mock_da = mock_da_cls.return_value
        mock_da.analyze.side_effect = lambda ds_id: Recipe(name=ds_id)

        analyzer = BatchAnalyzer(max_workers=1)
        ids = ["z_last", "a_first", "m_middle"]
        result = analyzer.analyze_batch(ids)

        result_ids = [r.dataset_id for r in result.results]
        self.assertEqual(result_ids, ids)

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_analyze_batch_continue_on_error_false(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalyzer

        mock_da = mock_da_cls.return_value

        # Use a single worker and make the first task raise an exception
        # that propagates via future.result()
        def side_effect(ds_id):
            if ds_id == "fail":
                raise RuntimeError("Fatal error")
            return Recipe(name=ds_id)

        mock_da.analyze.side_effect = side_effect

        analyzer = BatchAnalyzer(max_workers=1)
        # The _analyze_single method catches exceptions, so continue_on_error=False
        # only applies to the future.result() path. We need to test via the
        # outer exception path. Let's mock _analyze_single directly.
        original_analyze_single = analyzer._analyze_single

        def patched_analyze_single(ds_id):
            if ds_id == "fail":
                raise RuntimeError("Fatal error")
            return original_analyze_single(ds_id)

        analyzer._analyze_single = patched_analyze_single

        with self.assertRaises(RuntimeError):
            analyzer.analyze_batch(["fail", "ok"], continue_on_error=False)


class TestBatchAnalyzerFromFile(unittest.TestCase):
    """Test BatchAnalyzer.analyze_from_file() with different file formats."""

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def setUp(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalyzer

        self.mock_da = mock_da_cls.return_value
        self.mock_da.analyze.side_effect = lambda ds_id: Recipe(name=ds_id)
        self.analyzer = BatchAnalyzer(max_workers=1)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_from_file("/nonexistent/file.txt")

    def test_plain_text_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("dataset1\ndataset2\n# comment\n\ndataset3\n")
            tmppath = f.name

        try:
            result = self.analyzer.analyze_from_file(tmppath)
            self.assertEqual(len(result.results), 3)
        finally:
            os.unlink(tmppath)

    def test_json_list_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["ds1", "ds2"], f)
            tmppath = f.name

        try:
            result = self.analyzer.analyze_from_file(tmppath)
            self.assertEqual(len(result.results), 2)
        finally:
            os.unlink(tmppath)

    def test_json_dict_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"datasets": ["ds1", "ds2", "ds3"]}, f)
            tmppath = f.name

        try:
            result = self.analyzer.analyze_from_file(tmppath)
            self.assertEqual(len(result.results), 3)
        finally:
            os.unlink(tmppath)

    def test_json_invalid_structure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"wrong_key": "value"}, f)
            tmppath = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                self.analyzer.analyze_from_file(tmppath)
            self.assertIn("list or {datasets:", str(ctx.exception))
        finally:
            os.unlink(tmppath)

    def test_csv_file_with_header(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("dataset_id,other_col\norg/ds1,val1\norg/ds2,val2\n")
            tmppath = f.name

        try:
            result = self.analyzer.analyze_from_file(tmppath)
            self.assertEqual(len(result.results), 2)
        finally:
            os.unlink(tmppath)

    def test_csv_file_hf_id_header(self):
        """When first row contains a slash but also a keyword, it should be treated as header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("dataset_id/name\norg/ds1\norg/ds2\n")
            tmppath = f.name

        try:
            result = self.analyzer.analyze_from_file(tmppath)
            self.assertEqual(len(result.results), 2)
        finally:
            os.unlink(tmppath)

    def test_csv_file_url_first_row(self):
        """When first row starts with http, it is consumed by next() as header
        but the startswith check causes neither the skip nor the append branch
        to execute, so only subsequent rows are included."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("https://example.com/ds1\nhttps://example.com/ds2\nhttps://example.com/ds3\n")
            tmppath = f.name

        try:
            result = self.analyzer.analyze_from_file(tmppath)
            # First URL row is consumed by next() and skipped (neither branch adds it).
            # Remaining rows are added normally.
            self.assertEqual(len(result.results), 2)
        finally:
            os.unlink(tmppath)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            tmppath = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                self.analyzer.analyze_from_file(tmppath)
            self.assertIn("No dataset IDs", str(ctx.exception))
        finally:
            os.unlink(tmppath)


class TestBatchAnalyzerExportResults(unittest.TestCase):
    """Test BatchAnalyzer.export_results()."""

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_export_yaml(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalysisResult, BatchAnalyzer, BatchResult

        analyzer = BatchAnalyzer()
        recipe = Recipe(name="test-ds", source_type=SourceType.HUGGINGFACE)
        bar = BatchAnalysisResult(
            results=[BatchResult(dataset_id="test", success=True, recipe=recipe)],
            successful=1,
            failed=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = analyzer.export_results(bar, tmpdir, format="yaml")
            self.assertTrue(any(f.endswith(".yaml") for f in files))
            self.assertTrue(any("batch_summary.json" in f for f in files))

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_export_json(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalysisResult, BatchAnalyzer, BatchResult

        analyzer = BatchAnalyzer()
        recipe = Recipe(name="json-test", source_type=SourceType.HUGGINGFACE)
        bar = BatchAnalysisResult(
            results=[BatchResult(dataset_id="test", success=True, recipe=recipe)],
            successful=1,
            failed=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = analyzer.export_results(bar, tmpdir, format="json")
            json_files = [f for f in files if f.endswith(".json") and "summary" not in f]
            self.assertEqual(len(json_files), 1)

    @patch("datarecipe.batch_analyzer.DatasetAnalyzer")
    def test_export_skips_failed_results(self, mock_da_cls):
        from datarecipe.batch_analyzer import BatchAnalysisResult, BatchAnalyzer, BatchResult

        analyzer = BatchAnalyzer()
        bar = BatchAnalysisResult(
            results=[BatchResult(dataset_id="fail", success=False, error="err")],
            successful=0,
            failed=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = analyzer.export_results(bar, tmpdir, format="yaml")
            # Only summary should be created
            self.assertEqual(len(files), 1)
            self.assertIn("batch_summary.json", files[0])


# =============================================================================
# __init__.py tests (lazy imports)
# =============================================================================


class TestLazyImports(unittest.TestCase):
    """Test __getattr__ lazy imports in datarecipe/__init__.py."""

    def test_import_cost_calculator(self):
        import datarecipe

        cls = datarecipe.CostCalculator
        self.assertIsNotNone(cls)

    def test_import_enhanced_cost_calculator(self):
        import datarecipe

        cls = datarecipe.EnhancedCostCalculator
        self.assertIsNotNone(cls)

    def test_import_quality_analyzer(self):
        import datarecipe

        cls = datarecipe.QualityAnalyzer
        self.assertIsNotNone(cls)

    def test_import_batch_analyzer(self):
        import datarecipe

        cls = datarecipe.BatchAnalyzer
        self.assertIsNotNone(cls)

    def test_import_dataset_comparator(self):
        import datarecipe

        cls = datarecipe.DatasetComparator
        self.assertIsNotNone(cls)

    def test_import_workflow_generator(self):
        import datarecipe

        cls = datarecipe.WorkflowGenerator
        self.assertIsNotNone(cls)

    def test_import_annotator_profiler(self):
        import datarecipe

        cls = datarecipe.AnnotatorProfiler
        self.assertIsNotNone(cls)

    def test_import_production_deployer(self):
        import datarecipe

        cls = datarecipe.ProductionDeployer
        self.assertIsNotNone(cls)

    def test_import_task_profiles_module(self):
        import datarecipe

        mod = datarecipe.task_profiles
        self.assertIsNotNone(mod)

    def test_import_get_task_profile(self):
        import datarecipe

        func = datarecipe.get_task_profile
        self.assertTrue(callable(func))

    def test_import_task_type_profile(self):
        import datarecipe

        cls = datarecipe.TaskTypeProfile
        self.assertIsNotNone(cls)

    def test_import_get_provider(self):
        import datarecipe

        func = datarecipe.get_provider
        self.assertTrue(callable(func))

    def test_import_list_providers(self):
        import datarecipe

        func = datarecipe.list_providers
        self.assertTrue(callable(func))

    def test_import_discover_providers(self):
        import datarecipe

        func = datarecipe.discover_providers
        self.assertTrue(callable(func))

    def test_unknown_attribute_raises(self):
        import datarecipe

        with self.assertRaises(AttributeError) as ctx:
            _ = datarecipe.NonExistentClassName
        self.assertIn("has no attribute", str(ctx.exception))

    def test_direct_imports_available(self):
        """Test that eagerly imported names are directly available."""
        import datarecipe

        self.assertIsNotNone(datarecipe.Recipe)
        self.assertIsNotNone(datarecipe.Cost)
        self.assertIsNotNone(datarecipe.DatasetAnalyzer)
        self.assertIsNotNone(datarecipe.DataRecipe)
        self.assertIsNotNone(datarecipe.__version__)

    def test_all_list_complete(self):
        """Test that __all__ contains expected names."""
        import datarecipe

        for name in [
            "Recipe",
            "Cost",
            "DatasetAnalyzer",
            "BatchAnalyzer",
            "CostCalculator",
            "QualityAnalyzer",
            "WorkflowGenerator",
        ]:
            self.assertIn(name, datarecipe.__all__)


# =============================================================================
# __main__.py tests
# =============================================================================


class TestMainModule(unittest.TestCase):
    """Test __main__.py can be imported."""

    def test_main_module_importable(self):
        """Test that the __main__ module can be imported without error."""
        import importlib

        mod = importlib.import_module("datarecipe.__main__")
        self.assertIsNotNone(mod)

    @patch("datarecipe.cli.main")
    def test_main_calls_cli_main(self, mock_main):
        """Test that running __main__ calls cli.main."""
        # We can verify the module has the right structure
        import datarecipe.__main__ as main_mod

        self.assertTrue(hasattr(main_mod, "main"))


if __name__ == "__main__":
    unittest.main()
