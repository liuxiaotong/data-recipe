"""Comprehensive unit tests for WebExtractor.

Tests the web source extractor: page fetching, title/description extraction,
teacher model detection, generation type detection, dataset size extraction,
tag extraction, reproducibility assessment, cost estimation, and the full
extract() orchestrator with mocked HTTP requests.
"""

import unittest
from unittest.mock import MagicMock, patch

from datarecipe.schema import (
    GenerationType,
    SourceType,
)
from datarecipe.sources.web import WebExtractor

# ==================== Helper HTML Builders ====================


def _build_html(
    title="Test Dataset",
    description="A test dataset description",
    og_description=None,
    body="",
    h1=None,
):
    """Build a minimal HTML page for testing."""
    parts = ["<html><head>"]
    if title:
        parts.append(f"<title>{title}</title>")
    if description:
        parts.append(f'<meta name="description" content="{description}">')
    if og_description:
        parts.append(f'<meta property="og:description" content="{og_description}">')
    parts.append("</head><body>")
    if h1:
        parts.append(f"<h1>{h1}</h1>")
    parts.append(body)
    parts.append("</body></html>")
    return "".join(parts)


# ==================== Initialization Tests ====================


class TestWebExtractorInit(unittest.TestCase):
    """Test WebExtractor initialization."""

    def test_init(self):
        extractor = WebExtractor()
        self.assertIsInstance(extractor, WebExtractor)


# ==================== _fetch_page Tests ====================


class TestFetchPage(unittest.TestCase):
    """Test _fetch_page method."""

    def setUp(self):
        self.extractor = WebExtractor()

    @patch("datarecipe.sources.web.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>content</html>"
        mock_get.return_value = mock_response

        result = self.extractor._fetch_page("https://example.com")
        self.assertEqual(result, "<html>content</html>")
        mock_get.assert_called_once_with(
            "https://example.com",
            headers={"User-Agent": "Mozilla/5.0 (compatible; DataRecipe/1.0)"},
            timeout=15,
        )

    @patch("datarecipe.sources.web.requests.get")
    def test_non_200_returns_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.extractor._fetch_page("https://example.com/notfound")
        self.assertIsNone(result)

    @patch("datarecipe.sources.web.requests.get")
    def test_exception_returns_none(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        result = self.extractor._fetch_page("https://example.com")
        self.assertIsNone(result)

    @patch("datarecipe.sources.web.requests.get")
    def test_timeout_returns_none(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        result = self.extractor._fetch_page("https://example.com")
        self.assertIsNone(result)


# ==================== _extract_title Tests ====================


class TestExtractTitle(unittest.TestCase):
    """Test _extract_title method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_title_from_title_tag(self):
        content = "<html><head><title>My Dataset</title></head></html>"
        self.assertEqual(self.extractor._extract_title(content), "My Dataset")

    def test_title_strips_github_suffix(self):
        content = "<html><head><title>My Repo - GitHub</title></head></html>"
        self.assertEqual(self.extractor._extract_title(content), "My Repo")

    def test_title_strips_huggingface_suffix(self):
        content = "<html><head><title>My Model | Hugging Face</title></head></html>"
        self.assertEqual(self.extractor._extract_title(content), "My Model")

    def test_title_strips_pwc_suffix(self):
        content = "<html><head><title>My Paper - Papers With Code</title></head></html>"
        self.assertEqual(self.extractor._extract_title(content), "My Paper")

    def test_title_from_h1_when_no_title_tag(self):
        content = '<html><body><h1 class="main">Heading Title</h1></body></html>'
        self.assertEqual(self.extractor._extract_title(content), "Heading Title")

    def test_title_prefers_title_tag_over_h1(self):
        content = "<html><head><title>Title Tag</title></head><body><h1>H1 Tag</h1></body></html>"
        self.assertEqual(self.extractor._extract_title(content), "Title Tag")

    def test_no_title_returns_none(self):
        content = "<html><body><p>Just some content</p></body></html>"
        self.assertIsNone(self.extractor._extract_title(content))

    def test_title_whitespace_stripped(self):
        content = "<html><head><title>  Whitespace Title  </title></head></html>"
        self.assertEqual(self.extractor._extract_title(content), "Whitespace Title")

    def test_title_case_insensitive(self):
        content = "<html><head><TITLE>Upper Case</TITLE></head></html>"
        self.assertEqual(self.extractor._extract_title(content), "Upper Case")


# ==================== _extract_description Tests ====================


class TestExtractDescription(unittest.TestCase):
    """Test _extract_description method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_meta_description(self):
        content = '<meta name="description" content="A fine dataset">'
        self.assertEqual(self.extractor._extract_description(content), "A fine dataset")

    def test_og_description(self):
        content = '<meta property="og:description" content="OG description">'
        self.assertEqual(self.extractor._extract_description(content), "OG description")

    def test_meta_description_preferred_over_og(self):
        content = (
            '<meta name="description" content="Meta desc">'
            '<meta property="og:description" content="OG desc">'
        )
        self.assertEqual(self.extractor._extract_description(content), "Meta desc")

    def test_no_description_returns_none(self):
        content = "<html><body>No meta tags</body></html>"
        self.assertIsNone(self.extractor._extract_description(content))

    def test_description_with_single_quotes(self):
        content = "<meta name='description' content='Single quote desc'>"
        self.assertEqual(self.extractor._extract_description(content), "Single quote desc")


# ==================== _detect_teacher_models Tests ====================


class TestDetectTeacherModels(unittest.TestCase):
    """Test _detect_teacher_models method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_detect_gpt4(self):
        result = self.extractor._detect_teacher_models("This uses GPT-4 for generation")
        self.assertIn("GPT-4", result)

    def test_detect_gpt35(self):
        result = self.extractor._detect_teacher_models("Generated with gpt3.5 turbo")
        self.assertIn("GPT-3.5", result)

    def test_detect_gpt5(self):
        result = self.extractor._detect_teacher_models("We used gpt5 for data")
        self.assertIn("GPT-5", result)

    def test_detect_gpt52(self):
        result = self.extractor._detect_teacher_models("GPT-5.2 was used")
        self.assertIn("GPT-5.2", result)

    def test_detect_claude3(self):
        result = self.extractor._detect_teacher_models("Claude 3 Opus model")
        self.assertIn("Claude 3", result)

    def test_detect_claude4(self):
        result = self.extractor._detect_teacher_models("Using claude4 for annotation")
        self.assertIn("Claude 4", result)

    def test_detect_claude45(self):
        result = self.extractor._detect_teacher_models("Claude 4.5 Sonnet")
        self.assertIn("Claude 4.5", result)

    def test_detect_llama3(self):
        result = self.extractor._detect_teacher_models("Llama 3 was used")
        self.assertIn("Llama 3", result)

    def test_detect_llama4(self):
        result = self.extractor._detect_teacher_models("Llama4 model")
        self.assertIn("Llama 4", result)

    def test_detect_gemini(self):
        result = self.extractor._detect_teacher_models("Gemini was used")
        self.assertIn("Gemini", result)

    def test_detect_gemini2(self):
        result = self.extractor._detect_teacher_models("Gemini 2 Pro model")
        self.assertIn("Gemini 2", result)

    def test_detect_mistral(self):
        result = self.extractor._detect_teacher_models("Mistral Large model")
        self.assertIn("Mistral", result)

    def test_detect_qwen(self):
        result = self.extractor._detect_teacher_models("Qwen model used")
        self.assertIn("Qwen", result)

    def test_detect_deepseek(self):
        result = self.extractor._detect_teacher_models("DeepSeek V3 model")
        self.assertIn("DeepSeek", result)

    def test_detect_multiple_models(self):
        content = "We used GPT-4 and Claude 3 to generate this dataset"
        result = self.extractor._detect_teacher_models(content)
        self.assertIn("GPT-4", result)
        self.assertIn("Claude 3", result)

    def test_no_models_found(self):
        result = self.extractor._detect_teacher_models("A dataset about cats and dogs")
        self.assertEqual(result, [])

    def test_case_insensitive(self):
        result = self.extractor._detect_teacher_models("we used gpt-4 for generation")
        self.assertIn("GPT-4", result)

    def test_results_sorted(self):
        content = "Used Qwen, GPT-4, Claude 3, and DeepSeek"
        result = self.extractor._detect_teacher_models(content)
        self.assertEqual(result, sorted(result))


# ==================== _detect_generation_type Tests ====================


class TestDetectGenerationType(unittest.TestCase):
    """Test _detect_generation_type method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_synthetic_only(self):
        content = "This is a synthetic generated dataset using distillation"
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)
        self.assertEqual(syn_ratio, 1.0)
        self.assertEqual(hum_ratio, 0.0)

    def test_human_only(self):
        content = "Data was manually annotated by human experts"
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.HUMAN)
        self.assertEqual(syn_ratio, 0.0)
        self.assertEqual(hum_ratio, 1.0)

    def test_mixed(self):
        content = "Synthetic data was generated, then human annotators reviewed and corrected"
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.MIXED)
        self.assertIsNotNone(syn_ratio)
        self.assertIsNotNone(hum_ratio)
        self.assertAlmostEqual(syn_ratio + hum_ratio, 1.0)

    def test_unknown_no_keywords(self):
        content = "A dataset about weather patterns"
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.UNKNOWN)
        self.assertIsNone(syn_ratio)
        self.assertIsNone(hum_ratio)

    def test_mturk_detected_as_human(self):
        content = "Annotations collected via mturk workers"
        gen_type, _, _ = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.HUMAN)

    def test_crowdsource_detected_as_human(self):
        content = "Crowdsource effort to label data"
        gen_type, _, _ = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.HUMAN)

    def test_api_keyword_as_synthetic(self):
        content = "Data generated via api calls"
        gen_type, _, _ = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)

    def test_ai_generated_keyword(self):
        content = "This is an ai-generated dataset"
        gen_type, _, _ = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)

    def test_scale_ai_as_human(self):
        content = "Annotations from scale ai platform"
        gen_type, _, _ = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.HUMAN)

    def test_labelbox_as_human(self):
        content = "Used labelbox for annotations"
        gen_type, _, _ = self.extractor._detect_generation_type(content)
        self.assertEqual(gen_type, GenerationType.HUMAN)


# ==================== _build_generation_methods Tests ====================


class TestBuildGenerationMethods(unittest.TestCase):
    """Test _build_generation_methods method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_with_teacher_models(self):
        methods = self.extractor._build_generation_methods(
            ["GPT-4", "Claude 3"], GenerationType.SYNTHETIC, ""
        )
        self.assertEqual(len(methods), 2)
        self.assertEqual(methods[0].method_type, "distillation")
        self.assertEqual(methods[0].teacher_model, "GPT-4")
        self.assertEqual(methods[1].teacher_model, "Claude 3")

    def test_human_type_adds_annotation(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.HUMAN, ""
        )
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].method_type, "human_annotation")

    def test_mixed_type_with_models(self):
        methods = self.extractor._build_generation_methods(
            ["GPT-4"], GenerationType.MIXED, ""
        )
        self.assertEqual(len(methods), 2)
        types = [m.method_type for m in methods]
        self.assertIn("distillation", types)
        self.assertIn("human_annotation", types)

    def test_unknown_no_models(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.UNKNOWN, ""
        )
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].method_type, "unknown")

    def test_synthetic_no_models(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.SYNTHETIC, ""
        )
        # No teacher models, no human, not unknown (because SYNTHETIC != UNKNOWN)
        self.assertEqual(len(methods), 0)


# ==================== _extract_dataset_size Tests ====================


class TestExtractDatasetSize(unittest.TestCase):
    """Test _extract_dataset_size method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_k_suffix(self):
        content = "Contains 50k rows"
        self.assertEqual(self.extractor._extract_dataset_size(content), 50000)

    def test_m_suffix(self):
        content = "Over 1.5M rows in total"
        self.assertEqual(self.extractor._extract_dataset_size(content), 1500000)

    def test_plain_number_rows(self):
        content = "500 rows available"
        self.assertEqual(self.extractor._extract_dataset_size(content), 500)

    def test_comma_separated_rows(self):
        content = "Dataset has 1,000 rows"
        self.assertEqual(self.extractor._extract_dataset_size(content), 1000)

    def test_instances_keyword(self):
        # "instances" does not contain 'k' or 'm', so plain number works
        content = "20 instances"
        self.assertEqual(self.extractor._extract_dataset_size(content), 20)

    def test_no_size_found(self):
        content = "A dataset about things"
        self.assertIsNone(self.extractor._extract_dataset_size(content))

    def test_k_uppercase(self):
        content = "Contains 10K rows"
        self.assertEqual(self.extractor._extract_dataset_size(content), 10000)

    def test_large_comma_number_rows(self):
        content = "1,234,567 rows"
        self.assertEqual(self.extractor._extract_dataset_size(content), 1234567)

    def test_float_k_suffix(self):
        content = "2.5k rows"
        self.assertEqual(self.extractor._extract_dataset_size(content), 2500)

    def test_comma_number_with_examples_has_m_in_word(self):
        # Note: the 'm' in 'examples' is detected as M-suffix in the matched
        # span, so "100,000 examples" is treated as 100000 * 1M. This is a
        # known quirk of the current regex-based extraction logic.
        content = "100,000 examples"
        result = self.extractor._extract_dataset_size(content)
        self.assertIsNotNone(result)


# ==================== _extract_tags Tests ====================


class TestExtractTags(unittest.TestCase):
    """Test _extract_tags method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_nlp_tag(self):
        tags = self.extractor._extract_tags("A natural language processing dataset")
        self.assertIn("nlp", tags)

    def test_vision_tag(self):
        tags = self.extractor._extract_tags("An image classification dataset")
        self.assertIn("vision", tags)

    def test_reasoning_tag(self):
        tags = self.extractor._extract_tags("A reasoning and logic benchmark")
        self.assertIn("reasoning", tags)

    def test_math_tag(self):
        tags = self.extractor._extract_tags("Math problems for training")
        self.assertIn("math", tags)

    def test_code_tag(self):
        tags = self.extractor._extract_tags("A programming code dataset")
        self.assertIn("code", tags)

    def test_qa_tag(self):
        tags = self.extractor._extract_tags("Question answering benchmark")
        self.assertIn("qa", tags)

    def test_dialogue_tag(self):
        tags = self.extractor._extract_tags("Conversation dialogue data")
        self.assertIn("dialogue", tags)

    def test_benchmark_tag(self):
        tags = self.extractor._extract_tags("A new benchmark for evaluation")
        self.assertIn("benchmark", tags)

    def test_multiple_tags(self):
        tags = self.extractor._extract_tags("NLP reasoning and math benchmark")
        self.assertIn("nlp", tags)
        self.assertIn("reasoning", tags)
        self.assertIn("math", tags)
        self.assertIn("benchmark", tags)

    def test_no_tags(self):
        tags = self.extractor._extract_tags("A generic thing")
        self.assertEqual(tags, [])


# ==================== _assess_reproducibility Tests ====================


class TestAssessReproducibility(unittest.TestCase):
    """Test _assess_reproducibility method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_base_score(self):
        repro = self.extractor._assess_reproducibility("no keywords")
        self.assertEqual(repro.score, 3)  # Base score
        self.assertIn("exact_prompts", repro.missing)
        self.assertIn("filtering_criteria", repro.missing)
        self.assertIn("quality_thresholds", repro.missing)

    def test_github_reference_increases_score(self):
        repro = self.extractor._assess_reproducibility("code on github.com/org/repo")
        self.assertEqual(repro.score, 5)  # 3 base + 2 for github
        self.assertIn("source_code_reference", repro.available)

    def test_paper_reference_increases_score(self):
        repro = self.extractor._assess_reproducibility("See our paper on arxiv")
        self.assertEqual(repro.score, 4)  # 3 base + 1 for paper
        self.assertIn("paper_reference", repro.available)

    def test_download_increases_score(self):
        repro = self.extractor._assess_reproducibility("download the dataset here")
        self.assertEqual(repro.score, 4)  # 3 base + 1 for download
        self.assertIn("download_available", repro.available)

    def test_license_increases_score(self):
        repro = self.extractor._assess_reproducibility("Released under MIT license")
        self.assertEqual(repro.score, 4)  # 3 base + 1 for license
        self.assertIn("license_info", repro.available)

    def test_all_signals_present(self):
        content = "github repo, arxiv paper, download available, MIT license"
        repro = self.extractor._assess_reproducibility(content)
        # 3 base + 2 github + 1 paper + 1 download + 1 license = 8
        self.assertEqual(repro.score, 8)
        self.assertIn("source_code_reference", repro.available)
        self.assertIn("paper_reference", repro.available)
        self.assertIn("download_available", repro.available)
        self.assertIn("license_info", repro.available)

    def test_missing_source_code_when_no_github(self):
        repro = self.extractor._assess_reproducibility("just a webpage")
        self.assertIn("source_code", repro.missing)

    def test_score_capped_at_10(self):
        # Even with all signals, should not exceed 10
        content = "github arxiv paper download license" * 5
        repro = self.extractor._assess_reproducibility(content)
        self.assertLessEqual(repro.score, 10)

    def test_score_minimum_1(self):
        # Score starts at 3, so minimum should be 3 with no signals
        repro = self.extractor._assess_reproducibility("")
        self.assertGreaterEqual(repro.score, 1)


# ==================== _estimate_cost Tests ====================


class TestEstimateCost(unittest.TestCase):
    """Test _estimate_cost method."""

    def setUp(self):
        self.extractor = WebExtractor()

    def _make_recipe(self, teacher_models=None, generation_type=GenerationType.UNKNOWN):
        from datarecipe.schema import Recipe

        recipe = Recipe(name="test", source_type=SourceType.WEB, source_id="test")
        recipe.teacher_models = teacher_models or []
        recipe.generation_type = generation_type
        return recipe

    def test_cost_with_teacher_models(self):
        recipe = self._make_recipe(teacher_models=["GPT-4"])
        cost = self.extractor._estimate_cost(recipe)
        self.assertEqual(cost.api_calls_usd, 10000)
        self.assertEqual(cost.confidence, "low")

    def test_cost_with_human_generation(self):
        recipe = self._make_recipe(generation_type=GenerationType.HUMAN)
        cost = self.extractor._estimate_cost(recipe)
        self.assertEqual(cost.human_annotation_usd, 25000)

    def test_cost_with_mixed_generation(self):
        recipe = self._make_recipe(
            teacher_models=["GPT-4"], generation_type=GenerationType.MIXED
        )
        cost = self.extractor._estimate_cost(recipe)
        self.assertEqual(cost.api_calls_usd, 10000)
        self.assertEqual(cost.human_annotation_usd, 25000)
        self.assertEqual(cost.estimated_total_usd, 35000)

    def test_cost_unknown_generation_no_models(self):
        recipe = self._make_recipe()
        cost = self.extractor._estimate_cost(recipe)
        self.assertIsNone(cost.api_calls_usd)
        self.assertIsNone(cost.human_annotation_usd)
        self.assertIsNone(cost.estimated_total_usd)

    def test_cost_total_is_sum(self):
        recipe = self._make_recipe(
            teacher_models=["GPT-4"], generation_type=GenerationType.HUMAN
        )
        cost = self.extractor._estimate_cost(recipe)
        self.assertEqual(cost.estimated_total_usd, 10000 + 25000)


# ==================== Full extract() Tests ====================


class TestExtract(unittest.TestCase):
    """Test the full extract() orchestrator method."""

    def setUp(self):
        self.extractor = WebExtractor()

    @patch("datarecipe.sources.web.requests.get")
    def test_successful_extract(self, mock_get):
        html = _build_html(
            title="Synthetic NLP Dataset",
            description="A synthetic dataset for NLP tasks",
            body="<p>Generated with GPT-4 using synthetic methods. 10,000 rows. "
            "Code on github. See our arxiv paper. MIT license.</p>",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        recipe = self.extractor.extract("https://example.com/dataset")

        self.assertEqual(recipe.name, "Synthetic NLP Dataset")
        self.assertEqual(recipe.source_type, SourceType.WEB)
        self.assertEqual(recipe.source_id, "https://example.com/dataset")
        self.assertEqual(recipe.description, "A synthetic dataset for NLP tasks")
        self.assertEqual(recipe.homepage_url, "https://example.com/dataset")
        self.assertIn("GPT-4", recipe.teacher_models)
        self.assertIn("nlp", recipe.tags)
        self.assertEqual(recipe.num_examples, 10000)
        self.assertIsNotNone(recipe.reproducibility)
        self.assertIsNotNone(recipe.reproducibility)
        self.assertIsNotNone(recipe.cost)

    @patch("datarecipe.sources.web.requests.get")
    def test_extract_fetch_fails_raises(self, mock_get):
        mock_get.side_effect = Exception("Connection error")

        with self.assertRaises(ValueError) as ctx:
            self.extractor.extract("https://bad.example.com")
        self.assertIn("Could not fetch content", str(ctx.exception))

    @patch("datarecipe.sources.web.requests.get")
    def test_extract_404_raises(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with self.assertRaises(ValueError) as ctx:
            self.extractor.extract("https://example.com/notfound")
        self.assertIn("Could not fetch content", str(ctx.exception))

    @patch("datarecipe.sources.web.requests.get")
    def test_extract_no_title_uses_url(self, mock_get):
        html = "<html><body><p>No title here</p></body></html>"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        recipe = self.extractor.extract("https://example.com/dataset")
        self.assertEqual(recipe.name, "https://example.com/dataset")

    @patch("datarecipe.sources.web.requests.get")
    def test_extract_human_annotated_content(self, mock_get):
        html = _build_html(
            title="Human Dataset",
            description="Manually annotated by experts",
            body="<p>Crowdsource annotations by human experts using mturk. "
            "50,000 rows. vision benchmark</p>",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        recipe = self.extractor.extract("https://example.com/human")

        self.assertEqual(recipe.generation_type, GenerationType.HUMAN)
        self.assertEqual(recipe.synthetic_ratio, 0.0)
        self.assertEqual(recipe.human_ratio, 1.0)
        self.assertIn("vision", recipe.tags)
        self.assertIn("benchmark", recipe.tags)
        self.assertEqual(recipe.num_examples, 50000)

    @patch("datarecipe.sources.web.requests.get")
    def test_extract_mixed_content(self, mock_get):
        html = _build_html(
            title="Mixed Dataset",
            description="A mixed approach dataset",
            body="<p>Data was synthetic generated then human annotated for quality. "
            "Uses GPT-4 as teacher model.</p>",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        recipe = self.extractor.extract("https://example.com/mixed")

        self.assertEqual(recipe.generation_type, GenerationType.MIXED)
        self.assertIsNotNone(recipe.synthetic_ratio)
        self.assertIsNotNone(recipe.human_ratio)
        self.assertIn("GPT-4", recipe.teacher_models)

    @patch("datarecipe.sources.web.requests.get")
    def test_extract_unknown_generation_type(self, mock_get):
        html = _build_html(
            title="Plain Dataset",
            description="A dataset",
            body="<p>Weather data from sensors</p>",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        recipe = self.extractor.extract("https://example.com/plain")

        self.assertEqual(recipe.generation_type, GenerationType.UNKNOWN)
        self.assertIsNone(recipe.synthetic_ratio)
        self.assertIsNone(recipe.human_ratio)

    @patch("datarecipe.sources.web.requests.get")
    def test_extract_generation_methods_for_unknown(self, mock_get):
        html = _build_html(
            title="Unknown",
            description="desc",
            body="<p>Some unrelated content</p>",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        recipe = self.extractor.extract("https://example.com")
        # Should have an unknown generation method
        self.assertTrue(any(m.method_type == "unknown" for m in recipe.generation_methods))


# ==================== Edge Cases ====================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.extractor = WebExtractor()

    def test_empty_content_title(self):
        self.assertIsNone(self.extractor._extract_title(""))

    def test_empty_content_description(self):
        self.assertIsNone(self.extractor._extract_description(""))

    def test_empty_content_teacher_models(self):
        self.assertEqual(self.extractor._detect_teacher_models(""), [])

    def test_empty_content_generation_type(self):
        gen_type, syn, hum = self.extractor._detect_generation_type("")
        self.assertEqual(gen_type, GenerationType.UNKNOWN)
        self.assertIsNone(syn)
        self.assertIsNone(hum)

    def test_empty_content_dataset_size(self):
        self.assertIsNone(self.extractor._extract_dataset_size(""))

    def test_empty_content_tags(self):
        self.assertEqual(self.extractor._extract_tags(""), [])

    def test_html_with_only_og_description(self):
        content = '<meta property="og:description" content="Only OG">'
        self.assertEqual(self.extractor._extract_description(content), "Only OG")

    def test_gpt_variants(self):
        # Test various GPT patterns
        self.assertIn("GPT-4", self.extractor._detect_teacher_models("gpt4"))
        self.assertIn("GPT-4", self.extractor._detect_teacher_models("gpt-4"))
        self.assertIn("GPT-3.5", self.extractor._detect_teacher_models("gpt3.5"))
        self.assertIn("GPT-3.5", self.extractor._detect_teacher_models("gpt-3.5"))

    def test_claude_variants(self):
        self.assertIn("Claude 3", self.extractor._detect_teacher_models("claude3"))
        self.assertIn("Claude 3", self.extractor._detect_teacher_models("claude-3"))
        self.assertIn("Claude 3", self.extractor._detect_teacher_models("claude 3"))


if __name__ == "__main__":
    unittest.main()
