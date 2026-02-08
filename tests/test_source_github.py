"""Comprehensive unit tests for GitHubExtractor.

Tests the GitHub source extractor: repo info fetching, README fetching,
license extraction, teacher model detection, generation type detection,
generation methods building, reproducibility assessment, and the full
extract() orchestrator with mocked HTTP requests.
"""

import base64
import unittest
from unittest.mock import MagicMock, patch

from datarecipe.schema import (
    GenerationType,
    SourceType,
)
from datarecipe.sources.github import GitHubExtractor

# ==================== Initialization Tests ====================


class TestGitHubExtractorInit(unittest.TestCase):
    """Test GitHubExtractor initialization."""

    def test_init(self):
        extractor = GitHubExtractor()
        self.assertEqual(extractor.api_base, "https://api.github.com")


# ==================== _fetch_repo_info Tests ====================


class TestFetchRepoInfo(unittest.TestCase):
    """Test _fetch_repo_info method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    @patch("datarecipe.sources.github.requests.get")
    def test_successful_fetch(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "my-repo",
            "description": "A cool repo",
            "html_url": "https://github.com/org/my-repo",
            "topics": ["ai", "datasets"],
            "license": {"spdx_id": "MIT", "name": "MIT License"},
        }
        mock_get.return_value = mock_response

        result = self.extractor._fetch_repo_info("org/my-repo")

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "my-repo")
        self.assertEqual(result["description"], "A cool repo")
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/org/my-repo", timeout=10
        )

    @patch("datarecipe.sources.github.requests.get")
    def test_404_returns_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.extractor._fetch_repo_info("nonexistent/repo")
        self.assertIsNone(result)

    @patch("datarecipe.sources.github.requests.get")
    def test_exception_returns_none(self, mock_get):
        mock_get.side_effect = Exception("Connection error")

        result = self.extractor._fetch_repo_info("org/repo")
        self.assertIsNone(result)

    @patch("datarecipe.sources.github.requests.get")
    def test_timeout_returns_none(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("Timeout")

        result = self.extractor._fetch_repo_info("org/repo")
        self.assertIsNone(result)


# ==================== _fetch_readme Tests ====================


class TestFetchReadme(unittest.TestCase):
    """Test _fetch_readme method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    @patch("datarecipe.sources.github.requests.get")
    def test_successful_readme_fetch(self, mock_get):
        readme_content = "# My Project\nThis is a great project."
        encoded = base64.b64encode(readme_content.encode()).decode()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": encoded}
        mock_get.return_value = mock_response

        result = self.extractor._fetch_readme("org/repo")

        self.assertEqual(result, readme_content)
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/org/repo/readme", timeout=10
        )

    @patch("datarecipe.sources.github.requests.get")
    def test_readme_not_found_returns_none(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = self.extractor._fetch_readme("org/repo")
        self.assertIsNone(result)

    @patch("datarecipe.sources.github.requests.get")
    def test_readme_exception_returns_none(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        result = self.extractor._fetch_readme("org/repo")
        self.assertIsNone(result)

    @patch("datarecipe.sources.github.requests.get")
    def test_readme_empty_content(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": ""}
        mock_get.return_value = mock_response

        result = self.extractor._fetch_readme("org/repo")
        self.assertEqual(result, "")


# ==================== _extract_license Tests ====================


class TestExtractLicense(unittest.TestCase):
    """Test _extract_license method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    def test_spdx_id(self):
        repo_info = {"license": {"spdx_id": "MIT", "name": "MIT License"}}
        self.assertEqual(self.extractor._extract_license(repo_info), "MIT")

    def test_name_fallback(self):
        repo_info = {"license": {"spdx_id": None, "name": "Custom License"}}
        self.assertEqual(self.extractor._extract_license(repo_info), "Custom License")

    def test_no_license(self):
        repo_info = {"license": None}
        self.assertIsNone(self.extractor._extract_license(repo_info))

    def test_empty_license(self):
        repo_info = {}
        self.assertIsNone(self.extractor._extract_license(repo_info))

    def test_license_with_both_none(self):
        repo_info = {"license": {"spdx_id": None, "name": None}}
        self.assertIsNone(self.extractor._extract_license(repo_info))

    def test_apache_license(self):
        repo_info = {"license": {"spdx_id": "Apache-2.0", "name": "Apache License 2.0"}}
        self.assertEqual(self.extractor._extract_license(repo_info), "Apache-2.0")


# ==================== _detect_teacher_models Tests ====================


class TestDetectTeacherModels(unittest.TestCase):
    """Test _detect_teacher_models method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    def test_detect_gpt4(self):
        result = self.extractor._detect_teacher_models("Generated using GPT-4")
        self.assertIn("GPT-4", result)

    def test_detect_gpt52(self):
        result = self.extractor._detect_teacher_models("GPT-5.2 model")
        self.assertIn("GPT-5.2", result)

    def test_detect_gpt5(self):
        result = self.extractor._detect_teacher_models("Uses gpt5")
        self.assertIn("GPT-5", result)

    def test_detect_claude3(self):
        result = self.extractor._detect_teacher_models("Claude 3 model")
        self.assertIn("Claude 3", result)

    def test_detect_claude4(self):
        result = self.extractor._detect_teacher_models("claude4 was used")
        self.assertIn("Claude 4", result)

    def test_detect_claude45(self):
        result = self.extractor._detect_teacher_models("claude 4.5 sonnet")
        self.assertIn("Claude 4.5", result)

    def test_detect_llama3(self):
        result = self.extractor._detect_teacher_models("Uses Llama 3")
        self.assertIn("Llama 3", result)

    def test_detect_llama4(self):
        result = self.extractor._detect_teacher_models("Llama4 model")
        self.assertIn("Llama 4", result)

    def test_detect_gemini(self):
        result = self.extractor._detect_teacher_models("Uses Gemini model")
        self.assertIn("Gemini", result)

    def test_detect_multiple(self):
        content = "Uses GPT-4 and Claude 3 and Llama 3"
        result = self.extractor._detect_teacher_models(content)
        self.assertIn("GPT-4", result)
        self.assertIn("Claude 3", result)
        self.assertIn("Llama 3", result)

    def test_no_models(self):
        result = self.extractor._detect_teacher_models("A dataset about flowers")
        self.assertEqual(result, [])

    def test_results_sorted(self):
        content = "Uses Llama 3, GPT-4, Claude 3"
        result = self.extractor._detect_teacher_models(content)
        self.assertEqual(result, sorted(result))

    def test_case_insensitive(self):
        result = self.extractor._detect_teacher_models("gpt-4 model")
        self.assertIn("GPT-4", result)


# ==================== _detect_generation_type Tests ====================


class TestDetectGenerationType(unittest.TestCase):
    """Test _detect_generation_type method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    def test_synthetic(self):
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(
            "Synthetic generated data using distillation"
        )
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)
        self.assertEqual(syn_ratio, 1.0)
        self.assertEqual(hum_ratio, 0.0)

    def test_human(self):
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(
            "Manually annotated by human experts"
        )
        self.assertEqual(gen_type, GenerationType.HUMAN)
        self.assertEqual(syn_ratio, 0.0)
        self.assertEqual(hum_ratio, 1.0)

    def test_mixed(self):
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(
            "Synthetic data with human annotation review"
        )
        self.assertEqual(gen_type, GenerationType.MIXED)
        self.assertIsNotNone(syn_ratio)
        self.assertIsNotNone(hum_ratio)
        self.assertAlmostEqual(syn_ratio + hum_ratio, 1.0)

    def test_unknown(self):
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(
            "Weather data from sensors"
        )
        self.assertEqual(gen_type, GenerationType.UNKNOWN)
        self.assertIsNone(syn_ratio)
        self.assertIsNone(hum_ratio)

    def test_api_keyword_synthetic(self):
        gen_type, _, _ = self.extractor._detect_generation_type(
            "Data generated via api calls"
        )
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)

    def test_llm_generated_keyword(self):
        gen_type, _, _ = self.extractor._detect_generation_type(
            "llm-generated responses"
        )
        self.assertEqual(gen_type, GenerationType.SYNTHETIC)

    def test_crowdsource_keyword(self):
        gen_type, _, _ = self.extractor._detect_generation_type(
            "Crowdsource annotation effort"
        )
        self.assertEqual(gen_type, GenerationType.HUMAN)

    def test_expert_keyword(self):
        gen_type, _, _ = self.extractor._detect_generation_type(
            "Expert annotators labeled the data"
        )
        self.assertEqual(gen_type, GenerationType.HUMAN)


# ==================== _build_generation_methods Tests ====================


class TestBuildGenerationMethods(unittest.TestCase):
    """Test _build_generation_methods method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    def test_with_teacher_models(self):
        methods = self.extractor._build_generation_methods(
            ["GPT-4", "Claude 3"], GenerationType.SYNTHETIC, ""
        )
        distill_methods = [m for m in methods if m.method_type == "distillation"]
        self.assertEqual(len(distill_methods), 2)
        self.assertEqual(distill_methods[0].teacher_model, "GPT-4")
        self.assertEqual(distill_methods[1].teacher_model, "Claude 3")

    def test_human_type(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.HUMAN, ""
        )
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].method_type, "human_annotation")

    def test_mixed_type_with_models(self):
        methods = self.extractor._build_generation_methods(
            ["GPT-4"], GenerationType.MIXED, ""
        )
        types = [m.method_type for m in methods]
        self.assertIn("distillation", types)
        self.assertIn("human_annotation", types)

    def test_no_models_synthetic(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.SYNTHETIC, ""
        )
        self.assertEqual(len(methods), 0)

    def test_unknown_no_models(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.UNKNOWN, ""
        )
        self.assertEqual(len(methods), 0)


# ==================== _assess_reproducibility Tests ====================


class TestAssessReproducibility(unittest.TestCase):
    """Test _assess_reproducibility method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    def test_base_score_with_source_code(self):
        repo_info = {}
        repro = self.extractor._assess_reproducibility(repo_info, None)
        # 5 base + 1 source_code = 6
        self.assertEqual(repro.score, 6)
        self.assertIn("source_code", repro.available)

    def test_description_increases_score(self):
        repo_info = {"description": "A great dataset"}
        repro = self.extractor._assess_reproducibility(repo_info, None)
        # 5 base + 1 source_code + 1 description = 7
        self.assertEqual(repro.score, 7)
        self.assertIn("description", repro.available)

    def test_detailed_readme_increases_score(self):
        repo_info = {}
        long_readme = "x" * 600
        repro = self.extractor._assess_reproducibility(repo_info, long_readme)
        # 5 base + 1 source_code + 1 detailed_documentation = 7
        self.assertEqual(repro.score, 7)
        self.assertIn("detailed_documentation", repro.available)

    def test_short_readme_no_increase(self):
        repo_info = {}
        repro = self.extractor._assess_reproducibility(repo_info, "Short readme")
        # 5 base + 1 source_code = 6
        self.assertEqual(repro.score, 6)
        self.assertNotIn("detailed_documentation", repro.available)

    def test_license_increases_score(self):
        repo_info = {"license": {"spdx_id": "MIT"}}
        repro = self.extractor._assess_reproducibility(repo_info, None)
        # 5 base + 1 source_code + 1 license_info = 7
        self.assertEqual(repro.score, 7)
        self.assertIn("license_info", repro.available)

    def test_all_signals(self):
        repo_info = {
            "description": "A dataset",
            "license": {"spdx_id": "MIT"},
        }
        long_readme = "x" * 600
        repro = self.extractor._assess_reproducibility(repo_info, long_readme)
        # 5 base + 1 description + 1 detailed_doc + 1 license + 1 source_code = 9
        self.assertEqual(repro.score, 9)

    def test_missing_items_always_present(self):
        repo_info = {}
        repro = self.extractor._assess_reproducibility(repo_info, None)
        self.assertIn("exact_prompts", repro.missing)
        self.assertIn("filtering_criteria", repro.missing)

    def test_score_capped_at_10(self):
        repo_info = {
            "description": "A dataset",
            "license": {"spdx_id": "MIT"},
        }
        long_readme = "x" * 600
        repro = self.extractor._assess_reproducibility(repo_info, long_readme)
        self.assertLessEqual(repro.score, 10)

    def test_none_readme(self):
        repo_info = {}
        repro = self.extractor._assess_reproducibility(repo_info, None)
        self.assertNotIn("detailed_documentation", repro.available)

    def test_readme_exactly_500(self):
        repo_info = {}
        repro = self.extractor._assess_reproducibility(repo_info, "x" * 500)
        self.assertNotIn("detailed_documentation", repro.available)

    def test_readme_501(self):
        repo_info = {}
        repro = self.extractor._assess_reproducibility(repo_info, "x" * 501)
        self.assertIn("detailed_documentation", repro.available)


# ==================== Full extract() Tests ====================


class TestExtract(unittest.TestCase):
    """Test the full extract() orchestrator method."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_repo_not_found_raises(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = None

        with self.assertRaises(ValueError) as ctx:
            self.extractor.extract("nonexistent/repo")
        self.assertIn("Repository not found", str(ctx.exception))

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_successful_extract_with_readme(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "my-dataset",
            "description": "A great dataset project",
            "html_url": "https://github.com/org/my-dataset",
            "topics": ["ai", "datasets", "nlp"],
            "license": {"spdx_id": "Apache-2.0", "name": "Apache License 2.0"},
        }
        mock_fetch_readme.return_value = (
            "# My Dataset\n\nThis dataset was generated using GPT-4 "
            "with synthetic methods and distillation. " + "x" * 500
        )

        recipe = self.extractor.extract("org/my-dataset")

        self.assertEqual(recipe.name, "my-dataset")
        self.assertEqual(recipe.source_type, SourceType.GITHUB)
        self.assertEqual(recipe.source_id, "org/my-dataset")
        self.assertEqual(recipe.description, "A great dataset project")
        self.assertEqual(recipe.license, "Apache-2.0")
        self.assertEqual(recipe.homepage_url, "https://github.com/org/my-dataset")
        self.assertEqual(recipe.tags, ["ai", "datasets", "nlp"])
        self.assertIn("GPT-4", recipe.teacher_models)
        self.assertEqual(recipe.generation_type, GenerationType.SYNTHETIC)
        self.assertEqual(recipe.synthetic_ratio, 1.0)
        self.assertEqual(recipe.human_ratio, 0.0)
        self.assertIsNotNone(recipe.reproducibility)
        self.assertIsNotNone(recipe.cost)
        self.assertEqual(recipe.cost.estimated_total_usd, 25000)
        self.assertEqual(recipe.cost.confidence, "low")

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_without_readme(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "simple-repo",
            "description": "Simple dataset",
            "html_url": "https://github.com/org/simple-repo",
            "topics": [],
        }
        mock_fetch_readme.return_value = None

        recipe = self.extractor.extract("org/simple-repo")

        self.assertEqual(recipe.name, "simple-repo")
        self.assertEqual(recipe.teacher_models, [])
        self.assertEqual(recipe.generation_type, GenerationType.UNKNOWN)
        self.assertIsNone(recipe.synthetic_ratio)
        self.assertIsNone(recipe.human_ratio)
        self.assertEqual(recipe.generation_methods, [])

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_human_annotated(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "human-dataset",
            "description": "Human annotated",
            "html_url": "https://github.com/org/human-dataset",
            "topics": ["annotation"],
            "license": {"spdx_id": "MIT", "name": "MIT License"},
        }
        mock_fetch_readme.return_value = (
            "This dataset was manually annotated by human experts."
        )

        recipe = self.extractor.extract("org/human-dataset")

        self.assertEqual(recipe.generation_type, GenerationType.HUMAN)
        self.assertEqual(recipe.synthetic_ratio, 0.0)
        self.assertEqual(recipe.human_ratio, 1.0)
        annotation_methods = [
            m for m in recipe.generation_methods if m.method_type == "human_annotation"
        ]
        self.assertEqual(len(annotation_methods), 1)

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_mixed_generation(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "mixed-dataset",
            "description": "Mixed dataset",
            "html_url": "https://github.com/org/mixed-dataset",
            "topics": [],
        }
        mock_fetch_readme.return_value = (
            "Synthetic data was generated using api calls, "
            "then human annotators reviewed and corrected it."
        )

        recipe = self.extractor.extract("org/mixed-dataset")

        self.assertEqual(recipe.generation_type, GenerationType.MIXED)
        self.assertIsNotNone(recipe.synthetic_ratio)
        self.assertIsNotNone(recipe.human_ratio)
        self.assertAlmostEqual(recipe.synthetic_ratio + recipe.human_ratio, 1.0)

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_no_license(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "no-license",
            "description": "No license",
            "html_url": "https://github.com/org/no-license",
            "topics": [],
        }
        mock_fetch_readme.return_value = None

        recipe = self.extractor.extract("org/no-license")
        self.assertIsNone(recipe.license)

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_no_description(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "no-desc",
            "topics": [],
        }
        mock_fetch_readme.return_value = None

        recipe = self.extractor.extract("org/no-desc")
        self.assertIsNone(recipe.description)

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_no_topics(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "no-topics",
            "description": "A repo",
        }
        mock_fetch_readme.return_value = None

        recipe = self.extractor.extract("org/no-topics")
        self.assertEqual(recipe.tags, [])

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_uses_repo_id_as_name_fallback(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "description": "A repo",
            "topics": [],
        }
        mock_fetch_readme.return_value = None

        recipe = self.extractor.extract("org/fallback-name")
        self.assertEqual(recipe.name, "org/fallback-name")

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_extract_multiple_teacher_models(self, mock_fetch_info, mock_fetch_readme):
        mock_fetch_info.return_value = {
            "name": "multi-model",
            "description": "Multi model dataset",
            "topics": [],
        }
        mock_fetch_readme.return_value = (
            "Generated using GPT-4 and Claude 3 and Llama 3. "
            "Synthetic data only."
        )

        recipe = self.extractor.extract("org/multi-model")

        self.assertIn("GPT-4", recipe.teacher_models)
        self.assertIn("Claude 3", recipe.teacher_models)
        self.assertIn("Llama 3", recipe.teacher_models)
        distill_methods = [
            m for m in recipe.generation_methods if m.method_type == "distillation"
        ]
        self.assertEqual(len(distill_methods), 3)


# ==================== Edge Cases ====================


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        self.extractor = GitHubExtractor()

    def test_empty_readme_teacher_models(self):
        result = self.extractor._detect_teacher_models("")
        self.assertEqual(result, [])

    def test_empty_readme_generation_type(self):
        gen_type, syn, hum = self.extractor._detect_generation_type("")
        self.assertEqual(gen_type, GenerationType.UNKNOWN)
        self.assertIsNone(syn)
        self.assertIsNone(hum)

    def test_build_methods_unknown_empty(self):
        methods = self.extractor._build_generation_methods(
            [], GenerationType.UNKNOWN, ""
        )
        self.assertEqual(methods, [])

    def test_extract_license_empty_object(self):
        self.assertIsNone(self.extractor._extract_license({"license": {}}))

    def test_reproducibility_no_description_no_license(self):
        repo_info = {}
        repro = self.extractor._assess_reproducibility(repo_info, None)
        self.assertNotIn("description", repro.available)
        self.assertNotIn("license_info", repro.available)
        self.assertIn("source_code", repro.available)

    @patch.object(GitHubExtractor, "_fetch_readme")
    @patch.object(GitHubExtractor, "_fetch_repo_info")
    def test_cost_always_placeholder(self, mock_fetch_info, mock_fetch_readme):
        """Cost should always be the placeholder value regardless of content."""
        mock_fetch_info.return_value = {
            "name": "repo",
            "description": "desc",
            "topics": [],
        }
        mock_fetch_readme.return_value = None

        recipe = self.extractor.extract("org/repo")
        self.assertEqual(recipe.cost.estimated_total_usd, 25000)
        self.assertEqual(recipe.cost.confidence, "low")

    def test_gpt_variants(self):
        self.assertIn("GPT-4", self.extractor._detect_teacher_models("gpt4"))
        self.assertIn("GPT-4", self.extractor._detect_teacher_models("gpt-4"))
        self.assertIn("GPT-5", self.extractor._detect_teacher_models("gpt5"))
        self.assertIn("GPT-5", self.extractor._detect_teacher_models("gpt-5"))

    def test_claude_variants(self):
        self.assertIn("Claude 3", self.extractor._detect_teacher_models("claude3"))
        self.assertIn("Claude 3", self.extractor._detect_teacher_models("claude-3"))
        self.assertIn("Claude 3", self.extractor._detect_teacher_models("claude 3"))

    def test_mixed_ratios_sum_to_one(self):
        gen_type, syn_ratio, hum_ratio = self.extractor._detect_generation_type(
            "synthetic generated annotated human expert manual"
        )
        if gen_type == GenerationType.MIXED:
            self.assertAlmostEqual(syn_ratio + hum_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
