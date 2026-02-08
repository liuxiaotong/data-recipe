"""Comprehensive unit tests for the providers package.

Tests:
- providers/__init__.py: discover_providers, get_provider, list_providers, ProviderNotFoundError
- providers/local.py: LocalFilesProvider (validate_config, match_annotators, create_project,
  submit, get_status, cancel, all private _generate_* helpers)
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from datarecipe.providers import (
    ProviderNotFoundError,
    discover_providers,
    get_provider,
    list_providers,
)
from datarecipe.providers.local import LocalFilesProvider
from datarecipe.schema import (
    AcceptanceCriterion,
    AnnotatorProfile,
    DataRecipe,
    DeploymentResult,
    EnhancedCost,
    Milestone,
    ProductionConfig,
    ProjectHandle,
    ProjectStatus,
    QualityRule,
    ReviewWorkflow,
    SourceType,
    ValidationResult,
)

# =============================================================================
# Helper factories
# =============================================================================


def _make_recipe(**overrides) -> DataRecipe:
    """Create a minimal DataRecipe for testing."""
    defaults = {
        "name": "test-dataset",
        "source_type": SourceType.HUGGINGFACE,
        "description": "A test dataset for unit tests",
        "num_examples": 1000,
    }
    defaults.update(overrides)
    return DataRecipe(**defaults)


def _make_production_config(**overrides) -> ProductionConfig:
    """Create a ProductionConfig with standard test data."""
    defaults = {
        "annotation_guide": "# Annotation Guide\n\nFollow these rules.",
        "quality_rules": [
            QualityRule(
                rule_id="QR001",
                name="Non-empty",
                description="Fields must not be empty",
                check_type="format",
                severity="error",
                auto_check=True,
            ),
            QualityRule(
                rule_id="QR002",
                name="Length check",
                description="Text must be within length limits",
                check_type="content",
                severity="warning",
                auto_check=False,
            ),
        ],
        "acceptance_criteria": [
            AcceptanceCriterion(
                criterion_id="AC001",
                name="Accuracy",
                description="Annotation accuracy rate",
                threshold=0.95,
                metric_type="accuracy",
                priority="required",
            ),
            AcceptanceCriterion(
                criterion_id="AC002",
                name="Completeness",
                description="Task completion rate",
                threshold=0.98,
                metric_type="completeness",
                priority="required",
            ),
        ],
        "milestones": [
            Milestone(
                name="Phase 1",
                description="Preparation",
                deliverables=["Data prepared", "Team trained"],
                estimated_days=5,
            ),
            Milestone(
                name="Phase 2",
                description="Execution",
                deliverables=["Data annotated"],
                estimated_days=15,
                dependencies=["Phase 1"],
            ),
        ],
        "review_workflow": ReviewWorkflow.DOUBLE,
        "review_sample_rate": 0.2,
        "estimated_timeline_days": 20,
    }
    defaults.update(overrides)
    return ProductionConfig(**defaults)


def _make_project_handle(**overrides) -> ProjectHandle:
    """Create a minimal ProjectHandle."""
    defaults = {
        "project_id": "local_20250101_120000",
        "provider": "local",
        "created_at": "2025-01-01T12:00:00",
        "status": "created",
    }
    defaults.update(overrides)
    return ProjectHandle(**defaults)


# =============================================================================
# Tests for providers/__init__.py
# =============================================================================


class TestProviderNotFoundError(unittest.TestCase):
    """Test ProviderNotFoundError exception."""

    def test_is_exception(self):
        self.assertTrue(issubclass(ProviderNotFoundError, Exception))

    def test_message(self):
        err = ProviderNotFoundError("test message")
        self.assertEqual(str(err), "test message")

    def test_raise_and_catch(self):
        with self.assertRaises(ProviderNotFoundError):
            raise ProviderNotFoundError("not found")


class TestDiscoverProviders(unittest.TestCase):
    """Test discover_providers function."""

    def test_always_includes_local(self):
        providers = discover_providers()
        self.assertIn("local", providers)
        self.assertEqual(providers["local"], LocalFilesProvider)

    @patch("importlib.metadata.entry_points")
    def test_loads_entry_points(self, mock_eps):
        """Simulates finding a custom provider via entry_points."""
        mock_ep = MagicMock()
        mock_ep.name = "custom"
        mock_ep.load.return_value = type("CustomProvider", (), {})
        mock_eps.return_value = [mock_ep]

        providers = discover_providers()
        self.assertIn("custom", providers)
        self.assertIn("local", providers)

    @patch("importlib.metadata.entry_points")
    def test_failing_entry_point_is_skipped(self, mock_eps):
        """A broken entry_point should be logged and skipped."""
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("no such module")
        mock_eps.return_value = [mock_ep]

        providers = discover_providers()
        self.assertNotIn("broken", providers)
        self.assertIn("local", providers)

    @patch("importlib.metadata.entry_points")
    def test_entry_points_exception_handled(self, mock_eps):
        """If entry_points() itself raises, we still get local provider."""
        mock_eps.side_effect = Exception("metadata error")

        providers = discover_providers()
        self.assertIn("local", providers)
        self.assertEqual(len(providers), 1)

    @patch("importlib.metadata.entry_points")
    def test_local_override_prevented(self, mock_eps):
        """If an entry_point provides 'local', it overrides the fallback."""
        mock_ep = MagicMock()
        mock_ep.name = "local"
        custom_class = type("CustomLocal", (), {})
        mock_ep.load.return_value = custom_class
        mock_eps.return_value = [mock_ep]

        providers = discover_providers()
        # The entry_point local was loaded, so it should be the custom one
        self.assertEqual(providers["local"], custom_class)

    @patch("importlib.metadata.entry_points")
    def test_multiple_providers(self, mock_eps):
        """Multiple entry_points should all be loaded."""
        ep1 = MagicMock()
        ep1.name = "provider_a"
        ep1.load.return_value = type("ProviderA", (), {})

        ep2 = MagicMock()
        ep2.name = "provider_b"
        ep2.load.return_value = type("ProviderB", (), {})

        mock_eps.return_value = [ep1, ep2]

        providers = discover_providers()
        self.assertIn("provider_a", providers)
        self.assertIn("provider_b", providers)
        self.assertIn("local", providers)


class TestGetProvider(unittest.TestCase):
    """Test get_provider function."""

    def test_get_local_provider(self):
        provider = get_provider("local")
        self.assertIsInstance(provider, LocalFilesProvider)

    def test_get_nonexistent_provider_raises(self):
        with self.assertRaises(ProviderNotFoundError) as ctx:
            get_provider("nonexistent_xyz")
        self.assertIn("nonexistent_xyz", str(ctx.exception))
        self.assertIn("Available providers", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))

    def test_get_provider_returns_instance(self):
        """get_provider should return an instance, not a class."""
        provider = get_provider("local")
        self.assertNotEqual(provider, LocalFilesProvider)
        self.assertIsInstance(provider, LocalFilesProvider)

    @patch("importlib.metadata.entry_points")
    def test_get_custom_provider(self, mock_eps):
        """Can retrieve a custom provider loaded from entry_points."""

        class FakeProvider:
            @property
            def name(self):
                return "fake"

            @property
            def description(self):
                return "Fake provider"

        mock_ep = MagicMock()
        mock_ep.name = "fake"
        mock_ep.load.return_value = FakeProvider
        mock_eps.return_value = [mock_ep]

        provider = get_provider("fake")
        self.assertIsInstance(provider, FakeProvider)


class TestListProviders(unittest.TestCase):
    """Test list_providers function."""

    def test_list_includes_local(self):
        result = list_providers()
        names = [p["name"] for p in result]
        self.assertIn("local", names)

    def test_list_provider_has_description(self):
        result = list_providers()
        local_info = [p for p in result if p["name"] == "local"][0]
        self.assertIn("description", local_info)
        self.assertIsInstance(local_info["description"], str)
        self.assertGreater(len(local_info["description"]), 0)

    def test_list_returns_list_of_dicts(self):
        result = list_providers()
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn("name", item)
            self.assertIn("description", item)

    @patch("importlib.metadata.entry_points")
    def test_list_broken_provider_shows_error(self, mock_eps):
        """A provider that fails to instantiate should still appear with an error."""

        class FailingProvider:
            def __init__(self):
                raise RuntimeError("init failed")

        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.return_value = FailingProvider
        mock_eps.return_value = [mock_ep]

        result = list_providers()
        names = [p["name"] for p in result]
        self.assertIn("broken", names)
        broken_info = [p for p in result if p["name"] == "broken"][0]
        self.assertIn("Error loading", broken_info["description"])


# =============================================================================
# Tests for providers/local.py — LocalFilesProvider
# =============================================================================


class TestLocalFilesProviderProperties(unittest.TestCase):
    """Test name and description properties."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_name(self):
        self.assertEqual(self.provider.name, "local")

    def test_description(self):
        desc = self.provider.description
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 0)


class TestValidateConfig(unittest.TestCase):
    """Test validate_config method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_valid_full_config(self):
        config = _make_production_config()
        result = self.provider.validate_config(config)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

    def test_empty_annotation_guide_warns(self):
        config = _make_production_config(annotation_guide="")
        result = self.provider.validate_config(config)
        self.assertTrue(result.valid)  # warnings don't make it invalid
        self.assertIn("标注指南为空", result.warnings)

    def test_no_quality_rules_warns(self):
        config = _make_production_config(quality_rules=[])
        result = self.provider.validate_config(config)
        self.assertTrue(result.valid)
        self.assertIn("未定义质检规则", result.warnings)

    def test_no_acceptance_criteria_warns(self):
        config = _make_production_config(acceptance_criteria=[])
        result = self.provider.validate_config(config)
        self.assertTrue(result.valid)
        self.assertIn("未定义验收标准", result.warnings)

    def test_all_empty_produces_all_warnings(self):
        config = ProductionConfig()
        result = self.provider.validate_config(config)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.warnings), 3)

    def test_errors_list_always_empty(self):
        """Current implementation never produces errors."""
        config = ProductionConfig()
        result = self.provider.validate_config(config)
        self.assertEqual(result.errors, [])


class TestMatchAnnotators(unittest.TestCase):
    """Test match_annotators method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_returns_empty_list(self):
        profile = AnnotatorProfile()
        result = self.provider.match_annotators(profile)
        self.assertEqual(result, [])

    def test_returns_empty_with_custom_limit(self):
        profile = AnnotatorProfile()
        result = self.provider.match_annotators(profile, limit=100)
        self.assertEqual(result, [])


class TestSubmit(unittest.TestCase):
    """Test submit method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_always_succeeds(self):
        handle = _make_project_handle()
        result = self.provider.submit(handle)
        self.assertIsInstance(result, DeploymentResult)
        self.assertTrue(result.success)
        self.assertEqual(result.project_handle, handle)

    def test_details_have_message(self):
        handle = _make_project_handle()
        result = self.provider.submit(handle)
        self.assertIn("message", result.details)


class TestGetStatus(unittest.TestCase):
    """Test get_status method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_returns_completed(self):
        handle = _make_project_handle()
        result = self.provider.get_status(handle)
        self.assertIsInstance(result, ProjectStatus)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.progress, 100.0)

    def test_counts_are_zero(self):
        handle = _make_project_handle()
        result = self.provider.get_status(handle)
        self.assertEqual(result.completed_count, 0)
        self.assertEqual(result.total_count, 0)


class TestCancel(unittest.TestCase):
    """Test cancel method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_always_returns_true(self):
        handle = _make_project_handle()
        self.assertTrue(self.provider.cancel(handle))


# =============================================================================
# create_project — integration test with tmp_path
# =============================================================================


class TestCreateProjectMinimal(unittest.TestCase):
    """Test create_project with minimal recipe (no config)."""

    def test_creates_project_handle(self):
        """Minimal create_project should succeed and return a ProjectHandle."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            handle = provider.create_project(recipe, output_dir=tmpdir)

            self.assertIsInstance(handle, ProjectHandle)
            self.assertEqual(handle.provider, "local")
            self.assertTrue(handle.project_id.startswith("local_"))
            self.assertEqual(handle.status, "created")
            self.assertIn("output_dir", handle.metadata)
            self.assertIn("files_created", handle.metadata)

    def test_creates_recipe_yaml(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            provider.create_project(recipe, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            recipe_path = deploy_dir / "recipe.yaml"
            self.assertTrue(recipe_path.exists())
            content = recipe_path.read_text(encoding="utf-8")
            self.assertIn("test-dataset", content)

    def test_creates_readme(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            provider.create_project(recipe, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            readme_path = deploy_dir / "README.md"
            self.assertTrue(readme_path.exists())
            content = readme_path.read_text(encoding="utf-8")
            self.assertIn("test-dataset", content)
            self.assertIn("投产项目", content)

    def test_creates_scripts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            provider.create_project(recipe, output_dir=tmpdir)

            scripts_dir = Path(tmpdir) / "10_生产部署" / "scripts"
            self.assertTrue(scripts_dir.exists())
            self.assertTrue((scripts_dir / "01_prepare_data.py").exists())
            self.assertTrue((scripts_dir / "02_generate.py").exists())
            self.assertTrue((scripts_dir / "03_validate.py").exists())

    def test_no_annotator_profile_file_without_profile(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            provider.create_project(recipe, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            self.assertFalse((deploy_dir / "annotator_profile.yaml").exists())

    def test_no_cost_estimate_without_enhanced_cost(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            provider.create_project(recipe, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            self.assertFalse((deploy_dir / "cost_estimate.yaml").exists())


class TestCreateProjectWithProfile(unittest.TestCase):
    """Test create_project with annotator_profile."""

    def test_creates_annotator_profile_yaml(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe(annotator_profile=AnnotatorProfile())
            provider.create_project(recipe, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            profile_path = deploy_dir / "annotator_profile.yaml"
            self.assertTrue(profile_path.exists())
            content = profile_path.read_text(encoding="utf-8")
            self.assertIn("skill_requirements", content)


class TestCreateProjectWithEnhancedCost(unittest.TestCase):
    """Test create_project with enhanced_cost."""

    def test_creates_cost_estimate_yaml(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            enhanced_cost = EnhancedCost(
                api_cost=100.0,
                human_cost=500.0,
                total_cost=600.0,
                total_range={"low": 400.0, "high": 800.0},
            )
            recipe = _make_recipe(enhanced_cost=enhanced_cost)
            provider.create_project(recipe, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            cost_path = deploy_dir / "cost_estimate.yaml"
            self.assertTrue(cost_path.exists())
            content = cost_path.read_text(encoding="utf-8")
            self.assertIn("api_cost", content)


class TestCreateProjectWithFullConfig(unittest.TestCase):
    """Test create_project with a full ProductionConfig."""

    def _run(self):
        """Helper that runs create_project and returns (tmpdir_path, handle)."""
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        provider = LocalFilesProvider()
        enhanced_cost = EnhancedCost(
            api_cost=100.0,
            human_cost=500.0,
            total_cost=600.0,
            total_range={"low": 400.0, "high": 800.0},
        )
        recipe = _make_recipe(
            annotator_profile=AnnotatorProfile(),
            enhanced_cost=enhanced_cost,
            teacher_models=["gpt-4o"],
            synthetic_ratio=0.5,
        )
        config = _make_production_config()
        handle = provider.create_project(recipe, config=config, output_dir=self._tmpdir)
        return Path(self._tmpdir), handle

    def tearDown(self):
        import shutil

        if hasattr(self, "_tmpdir"):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_annotation_guide_created(self):
        root, _ = self._run()
        guide_path = root / "10_生产部署" / "annotation_guide.md"
        self.assertTrue(guide_path.exists())
        content = guide_path.read_text(encoding="utf-8")
        self.assertIn("Annotation Guide", content)

    def test_quality_rules_yaml_created(self):
        root, _ = self._run()
        rules_path = root / "10_生产部署" / "quality_rules.yaml"
        self.assertTrue(rules_path.exists())

    def test_quality_rules_md_created(self):
        root, _ = self._run()
        rules_md = root / "10_生产部署" / "quality_rules.md"
        self.assertTrue(rules_md.exists())
        content = rules_md.read_text(encoding="utf-8")
        self.assertIn("质检规则", content)

    def test_acceptance_criteria_yaml_created(self):
        root, _ = self._run()
        criteria_path = root / "10_生产部署" / "acceptance_criteria.yaml"
        self.assertTrue(criteria_path.exists())

    def test_acceptance_criteria_md_created(self):
        root, _ = self._run()
        criteria_md = root / "10_生产部署" / "acceptance_criteria.md"
        self.assertTrue(criteria_md.exists())
        content = criteria_md.read_text(encoding="utf-8")
        self.assertIn("验收标准", content)

    def test_timeline_md_created(self):
        root, _ = self._run()
        timeline_path = root / "10_生产部署" / "timeline.md"
        self.assertTrue(timeline_path.exists())
        content = timeline_path.read_text(encoding="utf-8")
        self.assertIn("项目时间线", content)

    def test_files_created_list_complete(self):
        root, handle = self._run()
        files = handle.metadata["files_created"]
        # At minimum: recipe.yaml, annotator_profile.yaml, cost_estimate.yaml,
        # annotation_guide.md, quality_rules.yaml, quality_rules.md,
        # acceptance_criteria.yaml, acceptance_criteria.md, timeline.md,
        # README.md, 3 scripts = 13
        self.assertGreaterEqual(len(files), 13)

    def test_root_readme_created(self):
        root, _ = self._run()
        readme = root / "README.md"
        self.assertTrue(readme.exists())
        content = readme.read_text(encoding="utf-8")
        self.assertIn("项目产出", content)

    def test_project_manifest_created(self):
        root, _ = self._run()
        manifest = root / ".project_manifest.json"
        self.assertTrue(manifest.exists())


class TestCreateProjectNoConfig(unittest.TestCase):
    """Test that config-dependent files are not created when config is None."""

    def test_no_guide_without_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            provider.create_project(recipe, config=None, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            self.assertFalse((deploy_dir / "annotation_guide.md").exists())
            self.assertFalse((deploy_dir / "quality_rules.yaml").exists())
            self.assertFalse((deploy_dir / "acceptance_criteria.yaml").exists())
            self.assertFalse((deploy_dir / "timeline.md").exists())


class TestCreateProjectEmptyConfig(unittest.TestCase):
    """Test with config provided but all optional fields empty."""

    def test_no_optional_files_with_empty_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            config = ProductionConfig()  # all defaults empty
            provider.create_project(recipe, config=config, output_dir=tmpdir)

            deploy_dir = Path(tmpdir) / "10_生产部署"
            # annotation_guide is "" which is falsy, so no file
            self.assertFalse((deploy_dir / "annotation_guide.md").exists())
            self.assertFalse((deploy_dir / "quality_rules.yaml").exists())
            self.assertFalse((deploy_dir / "acceptance_criteria.yaml").exists())
            self.assertFalse((deploy_dir / "timeline.md").exists())
            # But README and recipe.yaml and scripts always exist
            self.assertTrue((deploy_dir / "README.md").exists())
            self.assertTrue((deploy_dir / "recipe.yaml").exists())


# =============================================================================
# Private helper methods
# =============================================================================


class TestGenerateReadme(unittest.TestCase):
    """Test _generate_readme private method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_contains_recipe_name(self):
        recipe = _make_recipe(name="my-cool-dataset")
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("my-cool-dataset", md)
        self.assertIn("投产项目", md)

    def test_contains_description(self):
        recipe = _make_recipe(description="My description text")
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("My description text", md)

    def test_contains_source_type(self):
        recipe = _make_recipe()
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("huggingface", md)

    def test_contains_num_examples(self):
        recipe = _make_recipe(num_examples=5000)
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("5,000", md)

    def test_contains_synthetic_ratio(self):
        recipe = _make_recipe(synthetic_ratio=0.7)
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("70%", md)

    def test_contains_teacher_models(self):
        recipe = _make_recipe(teacher_models=["gpt-4o", "claude-3.5"])
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("gpt-4o", md)
        self.assertIn("claude-3.5", md)

    def test_contains_cost_section_with_enhanced_cost(self):
        enhanced_cost = EnhancedCost(
            api_cost=100.0,
            human_cost=500.0,
            total_cost=600.0,
            total_range={"low": 400.0, "high": 800.0},
        )
        recipe = _make_recipe(enhanced_cost=enhanced_cost)
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("成本估算", md)
        self.assertIn("$100.00", md)
        self.assertIn("$500.00", md)
        self.assertIn("$600.00", md)

    def test_no_cost_section_without_enhanced_cost(self):
        recipe = _make_recipe()
        md = self.provider._generate_readme(recipe, None)
        # The "## 成本估算" section header should not be present,
        # though "成本估算" may appear in the static project structure listing
        self.assertNotIn("## 成本估算", md)

    def test_milestones_in_readme_with_config(self):
        recipe = _make_recipe()
        config = _make_production_config()
        md = self.provider._generate_readme(recipe, config)
        self.assertIn("里程碑", md)
        self.assertIn("Phase 1", md)
        self.assertIn("Phase 2", md)

    def test_no_milestones_without_config(self):
        recipe = _make_recipe()
        md = self.provider._generate_readme(recipe, None)
        self.assertNotIn("里程碑", md)

    def test_contains_project_structure(self):
        recipe = _make_recipe()
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("项目结构", md)
        self.assertIn("recipe.yaml", md)

    def test_contains_quick_start(self):
        recipe = _make_recipe()
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("快速开始", md)

    def test_contains_datarecipe_footer(self):
        recipe = _make_recipe()
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("DataRecipe", md)

    def test_no_description(self):
        """Recipe with no description should still generate valid README."""
        recipe = _make_recipe(description=None)
        md = self.provider._generate_readme(recipe, None)
        self.assertIn("test-dataset", md)

    def test_no_num_examples(self):
        recipe = _make_recipe(num_examples=None)
        md = self.provider._generate_readme(recipe, None)
        self.assertNotIn("目标数量", md)


class TestGenerateTimelineMd(unittest.TestCase):
    """Test _generate_timeline_md private method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def test_header(self):
        config = _make_production_config()
        md = self.provider._generate_timeline_md(config)
        self.assertIn("# 项目时间线", md)

    def test_contains_total_days(self):
        config = _make_production_config(estimated_timeline_days=30)
        md = self.provider._generate_timeline_md(config)
        self.assertIn("30 天", md)

    def test_contains_review_info(self):
        config = _make_production_config(
            review_workflow=ReviewWorkflow.DOUBLE,
            review_sample_rate=0.2,
        )
        md = self.provider._generate_timeline_md(config)
        self.assertIn("double", md)
        self.assertIn("20%", md)

    def test_milestones_listed(self):
        config = _make_production_config()
        md = self.provider._generate_timeline_md(config)
        self.assertIn("M1: Phase 1", md)
        self.assertIn("M2: Phase 2", md)

    def test_cumulative_days(self):
        config = _make_production_config()
        md = self.provider._generate_timeline_md(config)
        # Phase 1: 5 days, cumulative = 5
        self.assertIn("5", md)
        # Phase 2: 15 days, cumulative = 20
        self.assertIn("20", md)

    def test_dependencies_shown(self):
        config = _make_production_config()
        md = self.provider._generate_timeline_md(config)
        self.assertIn("Phase 1", md)

    def test_deliverables_as_checkboxes(self):
        config = _make_production_config()
        md = self.provider._generate_timeline_md(config)
        self.assertIn("- [ ]", md)

    def test_gantt_chart(self):
        config = _make_production_config()
        md = self.provider._generate_timeline_md(config)
        self.assertIn("甘特图", md)
        self.assertIn("总计", md)

    def test_no_dependencies_section(self):
        """A milestone without dependencies should not show dependency line."""
        config = _make_production_config(
            milestones=[
                Milestone(
                    name="Solo Phase",
                    description="No deps",
                    deliverables=["Something"],
                    estimated_days=5,
                    dependencies=[],
                ),
            ]
        )
        md = self.provider._generate_timeline_md(config)
        self.assertNotIn("**依赖**:", md)


class TestGenerateQualityRulesMd(unittest.TestCase):
    """Test _generate_quality_rules_md private method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def _make_rules(self):
        return [
            QualityRule(
                rule_id="QR001",
                name="Non-empty",
                description="Fields must not be empty",
                check_type="format",
                severity="error",
                auto_check=True,
            ),
            QualityRule(
                rule_id="QR002",
                name="Content quality",
                description="Content must be relevant",
                check_type="content",
                severity="warning",
                auto_check=False,
            ),
            QualityRule(
                rule_id="QR003",
                name="Consistency",
                description="Data must be consistent",
                check_type="consistency",
                severity="info",
                auto_check=True,
            ),
        ]

    def test_header(self):
        md = self.provider._generate_quality_rules_md(self._make_rules())
        self.assertIn("# 质检规则说明", md)

    def test_severity_legend(self):
        md = self.provider._generate_quality_rules_md(self._make_rules())
        self.assertIn("error", md)
        self.assertIn("warning", md)
        self.assertIn("info", md)

    def test_check_type_sections(self):
        md = self.provider._generate_quality_rules_md(self._make_rules())
        self.assertIn("格式检查规则", md)
        self.assertIn("内容检查规则", md)
        self.assertIn("一致性检查规则", md)

    def test_rule_details(self):
        md = self.provider._generate_quality_rules_md(self._make_rules())
        self.assertIn("QR001", md)
        self.assertIn("Non-empty", md)
        self.assertIn("Fields must not be empty", md)

    def test_auto_check_labels(self):
        md = self.provider._generate_quality_rules_md(self._make_rules())
        self.assertIn("自动检查", md)
        self.assertIn("人工检查", md)

    def test_usage_guide(self):
        md = self.provider._generate_quality_rules_md(self._make_rules())
        self.assertIn("质检流程", md)
        self.assertIn("常见问题", md)

    def test_footer(self):
        md = self.provider._generate_quality_rules_md(self._make_rules())
        self.assertIn("DataRecipe", md)

    def test_dict_based_rules(self):
        """Rules passed as dicts (not QualityRule objects) should also work."""
        rules = [
            {
                "id": "QR_DICT",
                "name": "Dict Rule",
                "description": "A dict-based rule",
                "type": "format",
                "severity": "warning",
                "auto_check": False,
            }
        ]
        md = self.provider._generate_quality_rules_md(rules)
        self.assertIn("QR_DICT", md)
        self.assertIn("Dict Rule", md)

    def test_empty_rules_list(self):
        md = self.provider._generate_quality_rules_md([])
        self.assertIn("# 质检规则说明", md)

    def test_unknown_check_type(self):
        """Rules with unknown check_type should use raw type name as heading."""
        rules = [
            QualityRule(
                rule_id="QR_CUSTOM",
                name="Custom",
                description="Custom type",
                check_type="custom_type",
                severity="info",
                auto_check=True,
            )
        ]
        md = self.provider._generate_quality_rules_md(rules)
        self.assertIn("custom_type", md)

    def test_unknown_severity_uses_default_icon(self):
        """Unknown severity should not crash."""
        rules = [
            QualityRule(
                rule_id="QR_UNK",
                name="Unknown Sev",
                description="Unknown severity level",
                check_type="format",
                severity="critical",
                auto_check=True,
            )
        ]
        md = self.provider._generate_quality_rules_md(rules)
        self.assertIn("QR_UNK", md)


class TestGenerateAcceptanceCriteriaMd(unittest.TestCase):
    """Test _generate_acceptance_criteria_md private method."""

    def setUp(self):
        self.provider = LocalFilesProvider()

    def _make_criteria(self):
        return [
            AcceptanceCriterion(
                criterion_id="AC001",
                name="Accuracy",
                description="Annotation accuracy rate",
                threshold=0.95,
                metric_type="accuracy",
                priority="required",
            ),
            AcceptanceCriterion(
                criterion_id="AC002",
                name="Completeness",
                description="Task completion rate",
                threshold=0.98,
                metric_type="completeness",
                priority="required",
            ),
            AcceptanceCriterion(
                criterion_id="AC003",
                name="Agreement",
                description="Inter-annotator agreement",
                threshold=0.8,
                metric_type="agreement",
                priority="recommended",
            ),
        ]

    def test_header(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("# 验收标准说明", md)

    def test_summary_table(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("验收指标总览", md)
        self.assertIn("Accuracy", md)
        self.assertIn("95%", md)

    def test_priority_labels(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("必须", md)
        self.assertIn("建议", md)

    def test_detailed_sections(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("指标详细说明", md)
        self.assertIn("准确率", md)
        self.assertIn("完成率", md)
        self.assertIn("一致性", md)

    def test_metric_explanations(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("计算方式", md)
        self.assertIn("实践建议", md)

    def test_verification_flow(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("验收流程", md)
        self.assertIn("自检提交", md)

    def test_handling_table(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("不达标处理", md)

    def test_footer(self):
        md = self.provider._generate_acceptance_criteria_md(self._make_criteria())
        self.assertIn("DataRecipe", md)

    def test_dict_based_criteria(self):
        """Criteria passed as dicts should also work."""
        criteria = [
            {
                "name": "Dict Criterion",
                "description": "A dict-based criterion",
                "threshold": 0.9,
                "type": "format",
                "priority": "required",
            }
        ]
        md = self.provider._generate_acceptance_criteria_md(criteria)
        self.assertIn("Dict Criterion", md)
        self.assertIn("90%", md)

    def test_threshold_greater_than_one(self):
        """Threshold > 1 should be displayed as-is, not as percentage."""
        criteria = [
            AcceptanceCriterion(
                criterion_id="AC_HIGH",
                name="High threshold",
                description="Raw number threshold",
                threshold=50,
                metric_type="timeliness",
                priority="required",
            )
        ]
        md = self.provider._generate_acceptance_criteria_md(criteria)
        self.assertIn("50", md)
        # Should NOT have "5000%" or similar percentage conversion
        self.assertNotIn("5000%", md)

    def test_unknown_metric_type(self):
        """Unknown metric_type should use fallback title."""
        criteria = [
            AcceptanceCriterion(
                criterion_id="AC_UNK",
                name="Unknown Metric",
                description="Some unknown metric",
                threshold=0.85,
                metric_type="unknown_metric",
                priority="required",
            )
        ]
        md = self.provider._generate_acceptance_criteria_md(criteria)
        self.assertIn("Unknown Metric", md)

    def test_empty_criteria(self):
        md = self.provider._generate_acceptance_criteria_md([])
        self.assertIn("# 验收标准说明", md)

    def test_all_known_metric_types(self):
        """Test that all known metric types produce their titled sections."""
        known_types = ["completeness", "accuracy", "agreement", "format", "timeliness"]
        criteria = [
            AcceptanceCriterion(
                criterion_id=f"AC_{mt}",
                name=f"Test {mt}",
                description=f"Desc for {mt}",
                threshold=0.9,
                metric_type=mt,
                priority="required",
            )
            for mt in known_types
        ]
        md = self.provider._generate_acceptance_criteria_md(criteria)
        self.assertIn("完成率", md)
        self.assertIn("准确率", md)
        self.assertIn("一致性", md)
        self.assertIn("格式合规", md)
        self.assertIn("时效性", md)


class TestGenerateScripts(unittest.TestCase):
    """Test _generate_scripts private method."""

    def test_generates_three_scripts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir()

            provider = LocalFilesProvider()
            recipe = _make_recipe()
            files = provider._generate_scripts(recipe, scripts_dir)

            self.assertEqual(len(files), 3)
            for f in files:
                self.assertTrue(Path(f).exists())

    def test_prepare_script_uses_num_examples(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir()

            provider = LocalFilesProvider()
            recipe = _make_recipe(num_examples=42000)
            provider._generate_scripts(recipe, scripts_dir)

            content = (scripts_dir / "01_prepare_data.py").read_text(encoding="utf-8")
            self.assertIn("42000", content)

    def test_generate_script_uses_teacher_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir()

            provider = LocalFilesProvider()
            recipe = _make_recipe(teacher_models=["claude-3.5-sonnet"])
            provider._generate_scripts(recipe, scripts_dir)

            content = (scripts_dir / "02_generate.py").read_text(encoding="utf-8")
            self.assertIn("claude-3.5-sonnet", content)

    def test_generate_script_default_model(self):
        """When no teacher model is specified, defaults to gpt-4o."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir()

            provider = LocalFilesProvider()
            recipe = _make_recipe(teacher_models=[])
            provider._generate_scripts(recipe, scripts_dir)

            content = (scripts_dir / "02_generate.py").read_text(encoding="utf-8")
            self.assertIn("gpt-4o", content)

    def test_validate_script_is_valid_python(self):
        """Ensure generated scripts are syntactically valid Python."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir()

            provider = LocalFilesProvider()
            recipe = _make_recipe()
            provider._generate_scripts(recipe, scripts_dir)

            for script_name in ["01_prepare_data.py", "02_generate.py", "03_validate.py"]:
                content = (scripts_dir / script_name).read_text(encoding="utf-8")
                # compile() raises SyntaxError if not valid Python
                compile(content, script_name, "exec")


# =============================================================================
# Edge cases and integration
# =============================================================================


class TestCreateProjectDirectoryCreation(unittest.TestCase):
    """Test that create_project creates directories as needed."""

    def test_creates_nested_output_dir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "deep" / "nested" / "dir"
            provider = LocalFilesProvider()
            recipe = _make_recipe()
            handle = provider.create_project(recipe, output_dir=str(nested))
            self.assertTrue(nested.exists())
            self.assertEqual(handle.metadata["output_dir"], str(nested))


class TestDeploymentProviderProtocol(unittest.TestCase):
    """Test that LocalFilesProvider satisfies the DeploymentProvider protocol."""

    def test_has_name_property(self):
        provider = LocalFilesProvider()
        self.assertEqual(provider.name, "local")

    def test_has_description_property(self):
        provider = LocalFilesProvider()
        self.assertIsInstance(provider.description, str)

    def test_has_validate_config(self):
        self.assertTrue(hasattr(LocalFilesProvider, "validate_config"))
        self.assertTrue(callable(LocalFilesProvider.validate_config))

    def test_has_match_annotators(self):
        self.assertTrue(hasattr(LocalFilesProvider, "match_annotators"))

    def test_has_create_project(self):
        self.assertTrue(hasattr(LocalFilesProvider, "create_project"))

    def test_has_submit(self):
        self.assertTrue(hasattr(LocalFilesProvider, "submit"))

    def test_has_get_status(self):
        self.assertTrue(hasattr(LocalFilesProvider, "get_status"))

    def test_has_cancel(self):
        self.assertTrue(hasattr(LocalFilesProvider, "cancel"))


if __name__ == "__main__":
    unittest.main()
