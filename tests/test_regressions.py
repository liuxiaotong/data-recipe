"""Regression tests for recent fixes."""

import unittest
from datetime import datetime, timezone

from datarecipe.profiler import AnnotatorProfiler
from datarecipe.schema import (
    Recipe,
    SourceType,
    DataRecipe,
    ProductionConfig,
    AnnotatorProfile,
    ProjectHandle,
    ValidationResult,
    DeploymentResult,
)
from datarecipe.sources.github import GitHubExtractor
from datarecipe.deployer import ProductionDeployer


class StubProvider:
    """Minimal deployment provider used for testing."""

    name = "stub"
    description = "stub"

    def __init__(self):
        self.submit_called = False

    def validate_config(self, config: ProductionConfig) -> ValidationResult:  # type: ignore[override]
        return ValidationResult(valid=True)

    def match_annotators(self, profile: AnnotatorProfile, limit: int = 10):  # type: ignore[override]
        return []

    def create_project(self, recipe: DataRecipe, config: ProductionConfig):  # type: ignore[override]
        return ProjectHandle(
            project_id="stub",
            provider="stub",
            created_at=datetime.now(timezone.utc).isoformat(),
            status="created",
        )

    def submit(self, project: ProjectHandle) -> DeploymentResult:  # type: ignore[override]
        self.submit_called = True
        return DeploymentResult(success=True, project_handle=project)

    def get_status(self, project: ProjectHandle):  # type: ignore[override]
        raise NotImplementedError

    def cancel(self, project: ProjectHandle):  # type: ignore[override]
        return True


class RegressionTests(unittest.TestCase):
    def test_profiler_respects_zero_human_ratio(self):
        recipe = Recipe(name="demo", num_examples=1000, human_ratio=0.0)
        profile = AnnotatorProfiler().generate_profile(recipe)

        self.assertEqual(profile.estimated_person_days, 0.0)

    def test_github_extractor_marks_source_type(self):
        extractor = GitHubExtractor()

        def fake_repo_info(_repo_id):
            return {
                "name": "demo",
                "description": "desc",
                "topics": [],
                "html_url": "https://github.com/demo/demo",
            }

        extractor._fetch_repo_info = fake_repo_info  # type: ignore[method-assign]
        extractor._fetch_readme = lambda _repo_id: None  # type: ignore[method-assign]

        recipe = extractor.extract("demo/demo")
        self.assertEqual(recipe.source_type, SourceType.GITHUB)

    def test_deployer_dry_run_skips_submit(self):
        recipe = DataRecipe(name="demo", source_type=SourceType.HUGGINGFACE)
        config = ProductionConfig(annotation_guide="test guide")
        profile = AnnotatorProfile()

        deployer = ProductionDeployer()
        stub = StubProvider()
        deployer._providers = {"stub": stub}

        result = deployer.deploy(
            recipe,
            output="/tmp/demo",
            provider="stub",
            config=config,
            profile=profile,
            submit=False,
        )

        self.assertTrue(result.success)
        self.assertFalse(stub.submit_called)


if __name__ == "__main__":
    unittest.main()
