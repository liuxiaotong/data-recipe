"""Comprehensive unit tests for HuggingFaceExtractor.

Tests the HuggingFace source extractor: README fetching, teacher model detection,
generation type detection, generation methods building, prompt template detection,
annotation platform detection, cost estimation, reproducibility assessment,
and the full extract() orchestrator with mocked HuggingFace Hub API calls.
"""

import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

from datarecipe.schema import (
    GenerationMethod,
    GenerationType,
    SourceType,
)
from datarecipe.sources.huggingface import (
    HUMAN_KEYWORDS,
    SYNTHETIC_KEYWORDS,
    TEACHER_MODEL_PATTERNS,
    HuggingFaceExtractor,
)

# ==================== Stub Objects ====================


@dataclass
class StubCardData:
    """Minimal stub matching HuggingFace card_data interface."""

    language: object = None


@dataclass
class StubDatasetInfo:
    """Minimal stub matching HuggingFace DatasetInfo interface."""

    description: Optional[str] = None
    tags: Optional[list] = None
    author: Optional[str] = None
    license: Optional[str] = None
    card_data: Optional[StubCardData] = None


# ==================== Initialization Tests ====================


class TestHuggingFaceExtractorInit(unittest.TestCase):
    """Test HuggingFaceExtractor initialization."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def test_init(self, mock_hfapi_cls):
        extractor = HuggingFaceExtractor()
        self.assertIsNotNone(extractor.api)
        mock_hfapi_cls.assert_called_once()


# ==================== Constants Tests ====================


class TestConstants(unittest.TestCase):
    """Test that module-level constants are properly defined."""

    def test_teacher_model_patterns_not_empty(self):
        self.assertGreater(len(TEACHER_MODEL_PATTERNS), 0)

    def test_synthetic_keywords_not_empty(self):
        self.assertGreater(len(SYNTHETIC_KEYWORDS), 0)

    def test_human_keywords_not_empty(self):
        self.assertGreater(len(HUMAN_KEYWORDS), 0)

    def test_teacher_model_patterns_are_tuples(self):
        for item in TEACHER_MODEL_PATTERNS:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


# ==================== _get_readme Tests ====================


class TestGetReadme(unittest.TestCase):
    """Test _get_readme method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    @patch("datarecipe.sources.huggingface.hf_hub_download")
    def test_successful_readme_fetch(self, mock_download):
        import os
        import tempfile

        # Create a temp file with README content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# My Dataset\nThis is a README")
            tmp_path = f.name

        try:
            mock_download.return_value = tmp_path
            result = self.extractor._get_readme("org/dataset")
            self.assertEqual(result, "# My Dataset\nThis is a README")
            mock_download.assert_called_once_with(
                repo_id="org/dataset", filename="README.md", repo_type="dataset"
            )
        finally:
            os.unlink(tmp_path)

    @patch("datarecipe.sources.huggingface.hf_hub_download")
    def test_readme_not_found_returns_none(self, mock_download):
        from huggingface_hub.utils import EntryNotFoundError

        mock_download.side_effect = EntryNotFoundError("Not found")
        result = self.extractor._get_readme("org/dataset")
        self.assertIsNone(result)

    @patch("datarecipe.sources.huggingface.hf_hub_download")
    def test_readme_general_exception_returns_none(self, mock_download):
        mock_download.side_effect = Exception("Network error")
        result = self.extractor._get_readme("org/dataset")
        self.assertIsNone(result)


# ==================== _detect_teacher_models Tests ====================


class TestDetectTeacherModels(unittest.TestCase):
    """Test _detect_teacher_models method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def test_from_description(self):
        info = StubDatasetInfo(description="Generated using GPT-4")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("GPT-4", result)

    def test_from_readme(self):
        info = StubDatasetInfo()
        result = self.extractor._detect_teacher_models(info, "We used Claude 3 for annotation")
        self.assertIn("Claude 3", result)

    def test_from_tags(self):
        info = StubDatasetInfo(tags=["llama-3", "synthetic"])
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("Llama 3", result)

    def test_combined_sources(self):
        info = StubDatasetInfo(
            description="Uses GPT-4",
            tags=["claude-3", "synthetic"],
        )
        result = self.extractor._detect_teacher_models(info, "Also Llama 3")
        self.assertIn("GPT-4", result)
        self.assertIn("Claude 3", result)
        self.assertIn("Llama 3", result)

    def test_no_models_found(self):
        info = StubDatasetInfo(description="A dataset about animals")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertEqual(result, [])

    def test_empty_info(self):
        info = StubDatasetInfo()
        result = self.extractor._detect_teacher_models(info, None)
        self.assertEqual(result, [])

    def test_results_sorted(self):
        info = StubDatasetInfo(description="Uses Qwen, GPT-4, Claude 3, and DeepSeek")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertEqual(result, sorted(result))

    def test_gpt52(self):
        info = StubDatasetInfo(description="GPT-5.2 was used")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("GPT-5.2", result)

    def test_claude45(self):
        info = StubDatasetInfo(description="Claude 4.5 Sonnet")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("Claude 4.5", result)

    def test_mixtral(self):
        info = StubDatasetInfo(description="Mixtral model was used")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("Mixtral", result)

    def test_qwen3(self):
        info = StubDatasetInfo(description="Qwen 3 model")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("Qwen 3", result)

    def test_deepseek_v3(self):
        info = StubDatasetInfo(description="DeepSeek V3 model")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("DeepSeek V3", result)

    def test_gemini2(self):
        info = StubDatasetInfo(description="Gemini 2 model")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("Gemini 2", result)

    def test_case_insensitive(self):
        info = StubDatasetInfo(description="we used gpt-4 and claude 3")
        result = self.extractor._detect_teacher_models(info, None)
        self.assertIn("GPT-4", result)
        self.assertIn("Claude 3", result)


# ==================== _detect_generation_type Tests ====================


class TestDetectGenerationType(unittest.TestCase):
    """Test _detect_generation_type method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def test_synthetic(self):
        info = StubDatasetInfo(description="synthetic generated dataset using distillation")
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)
        self.assertEqual(syn_ratio, 1.0)
        self.assertEqual(hum_ratio, 0.0)

    def test_human(self):
        info = StubDatasetInfo(description="manually annotated by human experts")
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.HUMAN)
        self.assertEqual(syn_ratio, 0.0)
        self.assertEqual(hum_ratio, 1.0)

    def test_mixed(self):
        info = StubDatasetInfo(
            description="synthetic data with human annotation for quality control"
        )
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.MIXED)
        self.assertGreater(syn_ratio, 0)
        self.assertGreater(hum_ratio, 0)
        self.assertAlmostEqual(syn_ratio + hum_ratio, 1.0)

    def test_unknown(self):
        info = StubDatasetInfo(description="A weather dataset")
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.UNKNOWN)
        self.assertIsNone(syn_ratio)
        self.assertIsNone(hum_ratio)

    def test_from_readme(self):
        info = StubDatasetInfo()
        gen_type, _, _ = self.extractor._detect_generation_type(
            info, "This was synthetic generated data"
        )
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)

    def test_from_tags(self):
        info = StubDatasetInfo(tags=["synthetic", "generated"])
        gen_type, _, _ = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)

    def test_empty_info(self):
        info = StubDatasetInfo()
        gen_type, syn, hum = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.UNKNOWN)
        self.assertIsNone(syn)
        self.assertIsNone(hum)

    def test_mturk_as_human(self):
        info = StubDatasetInfo(description="Annotations via mturk workers")
        gen_type, _, _ = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.HUMAN)


# ==================== _build_generation_methods Tests ====================


class TestBuildGenerationMethods(unittest.TestCase):
    """Test _build_generation_methods method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def test_with_teacher_models(self):
        methods = self.extractor._build_generation_methods(
            ["GPT-4", "Claude 3"], GenerationType.SYNTHETIC, None
        )
        distill_methods = [m for m in methods if m.method_type == "distillation"]
        self.assertEqual(len(distill_methods), 2)
        teachers = [m.teacher_model for m in distill_methods]
        self.assertIn("GPT-4", teachers)
        self.assertIn("Claude 3", teachers)

    def test_human_type(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.HUMAN, None
        )
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].method_type, "human_annotation")

    def test_mixed_type(self):
        methods = self.extractor._build_generation_methods(
            ["GPT-4"], GenerationType.MIXED, None
        )
        types = [m.method_type for m in methods]
        self.assertIn("distillation", types)
        self.assertIn("human_annotation", types)

    def test_prompt_template_available(self):
        readme = "Here is the prompt template we used"
        methods = self.extractor._build_generation_methods(
            ["GPT-4"], GenerationType.SYNTHETIC, readme
        )
        distill_methods = [m for m in methods if m.method_type == "distillation"]
        self.assertTrue(distill_methods[0].prompt_template_available)

    def test_prompt_template_not_available(self):
        methods = self.extractor._build_generation_methods(
            ["GPT-4"], GenerationType.SYNTHETIC, None
        )
        distill_methods = [m for m in methods if m.method_type == "distillation"]
        self.assertFalse(distill_methods[0].prompt_template_available)

    def test_annotation_platform_detected(self):
        readme = "We used Scale AI for annotations"
        methods = self.extractor._build_generation_methods(
            [], GenerationType.HUMAN, readme
        )
        annotation_methods = [m for m in methods if m.method_type == "human_annotation"]
        self.assertEqual(annotation_methods[0].platform, "Scale AI")

    def test_empty_methods_for_synthetic_no_models(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.SYNTHETIC, None
        )
        self.assertEqual(len(methods), 0)


# ==================== _has_prompt_template Tests ====================


class TestHasPromptTemplate(unittest.TestCase):
    """Test _has_prompt_template method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def test_none_readme(self):
        self.assertFalse(self.extractor._has_prompt_template(None))

    def test_prompt_keyword(self):
        self.assertTrue(self.extractor._has_prompt_template("The prompt used was..."))

    def test_template_keyword(self):
        self.assertTrue(self.extractor._has_prompt_template("Template for generation"))

    def test_instruction_keyword(self):
        self.assertTrue(self.extractor._has_prompt_template("Instruction format"))

    def test_system_message_keyword(self):
        self.assertTrue(self.extractor._has_prompt_template("The system message was"))

    def test_no_keywords(self):
        self.assertFalse(self.extractor._has_prompt_template("Just a readme about cats"))

    def test_case_insensitive(self):
        self.assertTrue(self.extractor._has_prompt_template("PROMPT TEMPLATE DETAILS"))


# ==================== _detect_annotation_platform Tests ====================


class TestDetectAnnotationPlatform(unittest.TestCase):
    """Test _detect_annotation_platform method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def test_none_readme(self):
        self.assertIsNone(self.extractor._detect_annotation_platform(None))

    def test_scale_ai(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("Used scale ai for annotations"),
            "Scale AI",
        )

    def test_mturk(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("Workers from mturk"),
            "Amazon MTurk",
        )

    def test_mechanical_turk(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("Mechanical turk was used"),
            "Amazon MTurk",
        )

    def test_labelbox(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("Annotations via Labelbox"),
            "Labelbox",
        )

    def test_prolific(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("Recruited from Prolific"),
            "Prolific",
        )

    def test_appen(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("Used Appen platform"),
            "Appen",
        )

    def test_surge(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("Surge AI annotators"),
            "Surge AI",
        )

    def test_no_platform(self):
        self.assertIsNone(
            self.extractor._detect_annotation_platform("Just some readme content")
        )

    def test_case_insensitive(self):
        self.assertEqual(
            self.extractor._detect_annotation_platform("SCALE AI was used"),
            "Scale AI",
        )


# ==================== _estimate_cost Tests ====================


class TestEstimateCost(unittest.TestCase):
    """Test _estimate_cost method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def _make_recipe(self, teacher_models=None, generation_type=GenerationType.UNKNOWN):
        from datarecipe.schema import Recipe

        recipe = Recipe(name="test")
        recipe.teacher_models = teacher_models or []
        recipe.generation_type = generation_type
        return recipe

    def test_with_teacher_models(self):
        recipe = self._make_recipe(teacher_models=["GPT-4"])
        cost = self.extractor._estimate_cost(StubDatasetInfo(), recipe)
        self.assertEqual(cost.api_calls_usd, 10000)
        self.assertEqual(cost.confidence, "low")

    def test_with_human(self):
        recipe = self._make_recipe(generation_type=GenerationType.HUMAN)
        cost = self.extractor._estimate_cost(StubDatasetInfo(), recipe)
        self.assertEqual(cost.human_annotation_usd, 25000)
        self.assertEqual(cost.confidence, "low")

    def test_with_mixed(self):
        recipe = self._make_recipe(
            teacher_models=["GPT-4"], generation_type=GenerationType.MIXED
        )
        cost = self.extractor._estimate_cost(StubDatasetInfo(), recipe)
        self.assertEqual(cost.estimated_total_usd, 35000)

    def test_no_cost_signals(self):
        recipe = self._make_recipe()
        cost = self.extractor._estimate_cost(StubDatasetInfo(), recipe)
        self.assertIsNone(cost.api_calls_usd)
        self.assertIsNone(cost.human_annotation_usd)
        self.assertIsNone(cost.estimated_total_usd)


# ==================== _assess_reproducibility Tests ====================


class TestAssessReproducibility(unittest.TestCase):
    """Test _assess_reproducibility method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def _make_recipe(self, teacher_models=None, generation_methods=None):
        from datarecipe.schema import Recipe

        recipe = Recipe(name="test")
        recipe.teacher_models = teacher_models or []
        recipe.generation_methods = generation_methods or []
        return recipe

    def test_base_score(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        repro = self.extractor._assess_reproducibility(info, None, recipe)
        self.assertEqual(repro.score, 5)  # Base score
        self.assertIn("filtering_criteria", repro.missing)
        self.assertIn("quality_thresholds", repro.missing)

    def test_description_increases_score(self):
        info = StubDatasetInfo(description="A detailed description")
        recipe = self._make_recipe()
        repro = self.extractor._assess_reproducibility(info, None, recipe)
        self.assertEqual(repro.score, 6)  # 5 base + 1 description
        self.assertIn("description", repro.available)

    def test_detailed_readme_increases_score(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        long_readme = "x" * 600
        repro = self.extractor._assess_reproducibility(info, long_readme, recipe)
        self.assertEqual(repro.score, 6)  # 5 base + 1 detailed_documentation
        self.assertIn("detailed_documentation", repro.available)

    def test_short_readme_no_increase(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        repro = self.extractor._assess_reproducibility(info, "Short readme", recipe)
        self.assertEqual(repro.score, 5)
        self.assertNotIn("detailed_documentation", repro.available)

    def test_teacher_models_increase_score(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe(teacher_models=["GPT-4"])
        repro = self.extractor._assess_reproducibility(info, None, recipe)
        self.assertIn("teacher_model_names", repro.available)
        self.assertGreater(repro.score, 5)

    def test_no_teacher_models_adds_missing(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        repro = self.extractor._assess_reproducibility(info, None, recipe)
        self.assertIn("teacher_model_info", repro.missing)

    def test_prompt_templates_increase_score(self):
        info = StubDatasetInfo()
        method = GenerationMethod(
            method_type="distillation",
            teacher_model="GPT-4",
            prompt_template_available=True,
        )
        recipe = self._make_recipe(
            teacher_models=["GPT-4"], generation_methods=[method]
        )
        repro = self.extractor._assess_reproducibility(info, None, recipe)
        self.assertIn("prompt_templates", repro.available)

    def test_no_prompt_templates_adds_missing(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        repro = self.extractor._assess_reproducibility(info, None, recipe)
        self.assertIn("exact_prompts", repro.missing)

    def test_github_reference_increases_score(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        readme = "Code available on github.com/org/repo"
        repro = self.extractor._assess_reproducibility(info, readme, recipe)
        self.assertIn("source_code_reference", repro.available)

    def test_code_reference_increases_score(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        readme = "See the code directory for scripts"
        repro = self.extractor._assess_reproducibility(info, readme, recipe)
        self.assertIn("source_code_reference", repro.available)

    def test_no_code_reference_adds_missing(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        repro = self.extractor._assess_reproducibility(info, "No links here", recipe)
        self.assertIn("generation_scripts", repro.missing)

    def test_score_capped_at_10(self):
        info = StubDatasetInfo(description="description")
        method = GenerationMethod(
            method_type="distillation",
            teacher_model="GPT-4",
            prompt_template_available=True,
        )
        recipe = self._make_recipe(
            teacher_models=["GPT-4"], generation_methods=[method]
        )
        long_readme = "github code " * 200
        repro = self.extractor._assess_reproducibility(info, long_readme, recipe)
        self.assertLessEqual(repro.score, 10)

    def test_score_minimum_1(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        repro = self.extractor._assess_reproducibility(info, None, recipe)
        self.assertGreaterEqual(repro.score, 1)


# ==================== Full extract() Tests ====================


class TestExtract(unittest.TestCase):
    """Test the full extract() orchestrator method."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()
        self.mock_api = self.extractor.api

    def test_dataset_not_found(self):
        from huggingface_hub.utils import RepositoryNotFoundError

        # RepositoryNotFoundError requires a 'response' kwarg, so mock it
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.url = "https://huggingface.co/api/datasets/nonexistent/dataset"
        error = RepositoryNotFoundError("Not found", response=mock_response)
        self.mock_api.dataset_info.side_effect = error

        with self.assertRaises(ValueError) as ctx:
            self.extractor.extract("nonexistent/dataset")
        self.assertIn("Dataset not found", str(ctx.exception))

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_successful_extract(self, mock_get_readme):
        mock_get_readme.return_value = "# Dataset\nGenerated using GPT-4 with prompt templates"

        mock_info = StubDatasetInfo(
            description="A synthetic NLP dataset generated with GPT-4",
            tags=["synthetic", "nlp"],
            author="test_org",
            license="mit",
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("test_org/dataset")

        self.assertEqual(recipe.name, "test_org/dataset")
        self.assertEqual(recipe.source_type, SourceType.HUGGINGFACE)
        self.assertEqual(recipe.source_id, "test_org/dataset")
        self.assertEqual(recipe.description, mock_info.description)
        self.assertEqual(recipe.license, "mit")
        self.assertIn("synthetic", recipe.tags)
        self.assertIn("nlp", recipe.tags)
        self.assertEqual(recipe.authors, ["test_org"])
        self.assertIn("GPT-4", recipe.teacher_models)
        self.assertEqual(recipe.generation_type, GenerationType.SYNTHETIC)
        self.assertIsNotNone(recipe.cost)
        self.assertIsNotNone(recipe.reproducibility)

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_no_readme(self, mock_get_readme):
        mock_get_readme.return_value = None

        mock_info = StubDatasetInfo(
            description="Human annotated dataset",
            tags=["human", "annotated"],
            author="org",
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("org/dataset")

        self.assertEqual(recipe.generation_type, GenerationType.HUMAN)
        self.assertEqual(recipe.authors, ["org"])

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_no_author(self, mock_get_readme):
        mock_get_readme.return_value = None

        mock_info = StubDatasetInfo(
            description="A dataset",
            author=None,
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("dataset")

        self.assertEqual(recipe.authors, [])

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_no_tags(self, mock_get_readme):
        mock_get_readme.return_value = None

        mock_info = StubDatasetInfo(
            description="A dataset",
            tags=None,
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("dataset")

        self.assertEqual(recipe.tags, [])

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_with_card_data_language_list(self, mock_get_readme):
        mock_get_readme.return_value = None

        card = StubCardData(language=["en", "zh"])
        mock_info = StubDatasetInfo(
            description="Multilingual dataset",
            card_data=card,
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("org/multi")

        self.assertEqual(recipe.languages, ["en", "zh"])

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_with_card_data_language_string(self, mock_get_readme):
        mock_get_readme.return_value = None

        card = StubCardData(language="en")
        mock_info = StubDatasetInfo(
            description="English dataset",
            card_data=card,
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("org/english")

        self.assertEqual(recipe.languages, ["en"])

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_with_no_card_data(self, mock_get_readme):
        mock_get_readme.return_value = None

        mock_info = StubDatasetInfo(
            description="No card data",
            card_data=None,
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("org/nocard")

        self.assertEqual(recipe.languages, [])

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_card_data_no_language(self, mock_get_readme):
        mock_get_readme.return_value = None

        # card_data exists but has no language attribute
        card = MagicMock(spec=[])  # Empty spec means no attributes
        mock_info = StubDatasetInfo(
            description="Dataset",
            card_data=card,
        )
        # card_data exists but hasattr(card, 'language') is False
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("org/nolang")
        # Should not crash and languages should remain default
        self.assertEqual(recipe.languages, [])

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_mixed_generation(self, mock_get_readme):
        mock_get_readme.return_value = (
            "This dataset uses synthetic generation via GPT-4 "
            "with human expert annotation and review."
        )

        mock_info = StubDatasetInfo(
            description="Mixed synthetic and human data",
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("org/mixed")

        self.assertEqual(recipe.generation_type, GenerationType.MIXED)
        self.assertIsNotNone(recipe.synthetic_ratio)
        self.assertIsNotNone(recipe.human_ratio)

    @patch.object(HuggingFaceExtractor, "_get_readme")
    def test_extract_with_annotation_platform(self, mock_get_readme):
        mock_get_readme.return_value = "Annotations collected via Scale AI platform"

        mock_info = StubDatasetInfo(
            description="Human annotated dataset",
        )
        self.mock_api.dataset_info.return_value = mock_info

        recipe = self.extractor.extract("org/annotated")

        annotation_methods = [
            m for m in recipe.generation_methods if m.method_type == "human_annotation"
        ]
        if annotation_methods:
            self.assertEqual(annotation_methods[0].platform, "Scale AI")


# ==================== Edge Cases ====================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    @patch("datarecipe.sources.huggingface.HfApi")
    def setUp(self, mock_hfapi_cls):
        self.extractor = HuggingFaceExtractor()

    def test_detect_teacher_models_all_none(self):
        info = StubDatasetInfo()
        result = self.extractor._detect_teacher_models(info, None)
        self.assertEqual(result, [])

    def test_detect_generation_type_all_none(self):
        info = StubDatasetInfo()
        gen_type, syn, hum = self.extractor._detect_generation_type(info, None)
        self.assertEqual(gen_type, GenerationType.UNKNOWN)

    def test_build_methods_empty(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.UNKNOWN, None
        )
        self.assertEqual(methods, [])

    def test_readme_exactly_500_chars(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        readme = "x" * 500
        repro = self.extractor._assess_reproducibility(info, readme, recipe)
        # 500 is not > 500, so no detailed_documentation
        self.assertNotIn("detailed_documentation", repro.available)

    def test_readme_501_chars(self):
        info = StubDatasetInfo()
        recipe = self._make_recipe()
        readme = "x" * 501
        repro = self.extractor._assess_reproducibility(info, readme, recipe)
        self.assertIn("detailed_documentation", repro.available)

    def _make_recipe(self, teacher_models=None, generation_methods=None):
        from datarecipe.schema import Recipe

        recipe = Recipe(name="test")
        recipe.teacher_models = teacher_models or []
        recipe.generation_methods = generation_methods or []
        return recipe


if __name__ == "__main__":
    unittest.main()
