"""Unit tests for deployer.py — ProductionDeployer.

Covers:
- ProductionDeployer.__init__: provider loading (success + failure)
- list_providers(): returns name/description dicts
- get_provider(): found and not-found cases
- generate_config(): full config generation with all sub-components
- deploy(): local provider, platform provider (submit/no-submit), provider not found,
            validation failure, create_project exception, auto-generate profile/config
- _generate_annotation_guide(): synthetic, human, mixed types; with/without profile;
            languages, num_examples, synthetic_ratio branches
- _generate_quality_rules(): all generation types (synthetic, human, mixed);
            single-language vs multi-language
- _generate_acceptance_criteria(): verifies 5 criteria with correct IDs and thresholds
- _determine_review_workflow(): high human ratio, expert tags, large dataset, default
- _generate_milestones(): milestone count, names, dependencies, estimated_days
- _estimate_timeline(): with profile person_days, small/large datasets, clamping
"""

from unittest.mock import MagicMock, patch

import pytest

from datarecipe.providers import ProviderNotFoundError
from datarecipe.schema import (
    AcceptanceCriterion,
    AnnotatorProfile,
    DataRecipe,
    DeploymentResult,
    GenerationType,
    Milestone,
    ProductionConfig,
    ProjectHandle,
    QualityRule,
    ReviewWorkflow,
    ValidationResult,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_data_recipe(**kwargs) -> DataRecipe:
    """Create a DataRecipe with sensible defaults."""
    defaults = {
        "name": "test-dataset",
        "description": "A test dataset",
        "languages": ["en"],
        "tags": [],
        "num_examples": 5000,
        "generation_type": GenerationType.SYNTHETIC,
        "synthetic_ratio": 0.8,
        "human_ratio": 0.2,
    }
    defaults.update(kwargs)
    return DataRecipe(**defaults)


def _make_profile(**kwargs) -> AnnotatorProfile:
    """Create an AnnotatorProfile with sensible defaults."""
    return AnnotatorProfile(**kwargs)


def _make_production_config(**kwargs) -> ProductionConfig:
    """Create a ProductionConfig with sensible defaults."""
    return ProductionConfig(**kwargs)


def _make_project_handle(**kwargs) -> ProjectHandle:
    """Create a ProjectHandle with sensible defaults."""
    defaults = {
        "project_id": "test_123",
        "provider": "local",
        "created_at": "2025-01-01T00:00:00",
        "status": "created",
        "metadata": {"files_created": ["file1.yaml", "file2.md"]},
    }
    defaults.update(kwargs)
    return ProjectHandle(**defaults)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_discover_providers():
    """Patch discover_providers to return a mock local provider."""
    mock_provider_class = MagicMock()
    mock_provider_instance = MagicMock()
    mock_provider_instance.description = "Mock local provider"
    mock_provider_class.return_value = mock_provider_instance

    with patch("datarecipe.deployer.discover_providers") as mock_discover:
        mock_discover.return_value = {"local": mock_provider_class}
        yield mock_discover, mock_provider_instance


@pytest.fixture
def deployer(mock_discover_providers):
    """Create a ProductionDeployer with mocked providers."""
    from datarecipe.deployer import ProductionDeployer

    return ProductionDeployer()


# =============================================================================
# __init__ and _load_providers
# =============================================================================


class TestInit:
    """Tests for ProductionDeployer initialization."""

    def test_providers_loaded(self, deployer, mock_discover_providers):
        """Providers should be loaded during init."""
        _, mock_instance = mock_discover_providers
        assert "local" in deployer._providers
        assert deployer._providers["local"] is mock_instance

    def test_provider_instantiation_failure_logged(self):
        """If a provider class fails to instantiate, it should be skipped with a warning."""
        bad_class = MagicMock(side_effect=ImportError("bad provider"))
        good_class = MagicMock()
        good_instance = MagicMock()
        good_instance.description = "Good provider"
        good_class.return_value = good_instance

        with patch("datarecipe.deployer.discover_providers") as mock_discover:
            mock_discover.return_value = {
                "bad": bad_class,
                "good": good_class,
            }
            from datarecipe.deployer import ProductionDeployer

            d = ProductionDeployer()
            assert "bad" not in d._providers
            assert "good" in d._providers

    def test_provider_instantiation_type_error(self):
        """TypeError during instantiation should be caught."""
        bad_class = MagicMock(side_effect=TypeError("bad args"))

        with patch("datarecipe.deployer.discover_providers") as mock_discover:
            mock_discover.return_value = {"broken": bad_class}
            from datarecipe.deployer import ProductionDeployer

            d = ProductionDeployer()
            assert "broken" not in d._providers

    def test_provider_instantiation_value_error(self):
        """ValueError during instantiation should be caught."""
        bad_class = MagicMock(side_effect=ValueError("invalid config"))

        with patch("datarecipe.deployer.discover_providers") as mock_discover:
            mock_discover.return_value = {"broken": bad_class}
            from datarecipe.deployer import ProductionDeployer

            d = ProductionDeployer()
            assert "broken" not in d._providers

    def test_provider_instantiation_os_error(self):
        """OSError during instantiation should be caught."""
        bad_class = MagicMock(side_effect=OSError("file not found"))

        with patch("datarecipe.deployer.discover_providers") as mock_discover:
            mock_discover.return_value = {"broken": bad_class}
            from datarecipe.deployer import ProductionDeployer

            d = ProductionDeployer()
            assert "broken" not in d._providers


# =============================================================================
# list_providers
# =============================================================================


class TestListProviders:
    """Tests for list_providers()."""

    def test_returns_list_of_dicts(self, deployer, mock_discover_providers):
        _, mock_instance = mock_discover_providers
        result = deployer.list_providers()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "local"
        assert result[0]["description"] == "Mock local provider"

    def test_empty_providers(self):
        """Empty providers should return empty list."""
        with patch("datarecipe.deployer.discover_providers") as mock_discover:
            mock_discover.return_value = {}
            from datarecipe.deployer import ProductionDeployer

            d = ProductionDeployer()
            assert d.list_providers() == []

    def test_multiple_providers(self):
        """Multiple providers should all appear."""
        p1_class = MagicMock()
        p1 = MagicMock()
        p1.description = "Provider 1"
        p1_class.return_value = p1

        p2_class = MagicMock()
        p2 = MagicMock()
        p2.description = "Provider 2"
        p2_class.return_value = p2

        with patch("datarecipe.deployer.discover_providers") as mock_discover:
            mock_discover.return_value = {"p1": p1_class, "p2": p2_class}
            from datarecipe.deployer import ProductionDeployer

            d = ProductionDeployer()
            result = d.list_providers()
            names = {r["name"] for r in result}
            assert names == {"p1", "p2"}


# =============================================================================
# get_provider
# =============================================================================


class TestGetProvider:
    """Tests for get_provider()."""

    def test_existing_provider(self, deployer, mock_discover_providers):
        _, mock_instance = mock_discover_providers
        p = deployer.get_provider("local")
        assert p is mock_instance

    def test_missing_provider_raises(self, deployer):
        with pytest.raises(ProviderNotFoundError, match="nonexistent"):
            deployer.get_provider("nonexistent")

    def test_error_message_lists_available(self, deployer):
        with pytest.raises(ProviderNotFoundError) as exc_info:
            deployer.get_provider("missing")
        assert "local" in str(exc_info.value)


# =============================================================================
# _generate_annotation_guide
# =============================================================================


class TestGenerateAnnotationGuide:
    """Tests for _generate_annotation_guide()."""

    def test_synthetic_type_content(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.SYNTHETIC)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "标注指南" in guide
        assert recipe.name in guide
        assert "合成数据验证" in guide

    def test_human_type_content(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.HUMAN)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "人工标注要求" in guide

    def test_mixed_type_content(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.MIXED)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "混合数据处理" in guide

    def test_description_included(self, deployer):
        recipe = _make_data_recipe(description="Custom description here")
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "Custom description here" in guide

    def test_no_description(self, deployer):
        recipe = _make_data_recipe(description="")
        guide = deployer._generate_annotation_guide(recipe, None)
        # Should still have the header but no description block
        assert "标注指南" in guide

    def test_num_examples_displayed(self, deployer):
        recipe = _make_data_recipe(num_examples=10000)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "10,000" in guide

    def test_no_num_examples(self, deployer):
        recipe = _make_data_recipe(num_examples=None)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "目标数量" not in guide

    def test_synthetic_ratio_displayed(self, deployer):
        recipe = _make_data_recipe(synthetic_ratio=0.7)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "70%" in guide

    def test_no_synthetic_ratio(self, deployer):
        recipe = _make_data_recipe(synthetic_ratio=None)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "合成比例" not in guide

    def test_languages_displayed(self, deployer):
        recipe = _make_data_recipe(languages=["en", "zh"])
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "en" in guide
        assert "zh" in guide

    def test_no_languages(self, deployer):
        recipe = _make_data_recipe(languages=[])
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "语言" not in guide

    def test_empty_string_languages_filtered(self, deployer):
        recipe = _make_data_recipe(languages=["", "en", ""])
        guide = deployer._generate_annotation_guide(recipe, None)
        # Only 'en' should appear
        assert "en" in guide

    def test_all_empty_languages_no_display(self, deployer):
        recipe = _make_data_recipe(languages=["", ""])
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "语言" not in guide

    def test_guide_sections_present(self, deployer):
        recipe = _make_data_recipe()
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "## 1. 任务概述" in guide
        assert "## 2. 标注目标" in guide
        assert "## 3. 标注标准" in guide
        assert "## 4. 示例" in guide
        assert "## 5. 特殊情况处理" in guide
        assert "## 6. 常见问题" in guide
        assert "## 7. 联系与反馈" in guide

    def test_guide_footer(self, deployer):
        recipe = _make_data_recipe()
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "DataRecipe 自动生成" in guide


# =============================================================================
# _generate_quality_rules
# =============================================================================


class TestGenerateQualityRules:
    """Tests for _generate_quality_rules()."""

    def test_common_rules_always_present(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.HUMAN)
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR001" in rule_ids  # non-empty check
        assert "QR002" in rule_ids  # length check
        assert "QR003" in rule_ids  # duplicate check
        assert "QR004" in rule_ids  # format check

    def test_synthetic_type_adds_extra_rules(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.SYNTHETIC)
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR005" in rule_ids  # factuality check
        assert "QR006" in rule_ids  # AI trace check

    def test_mixed_type_adds_extra_rules(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.MIXED)
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR005" in rule_ids
        assert "QR006" in rule_ids

    def test_human_type_no_extra_rules(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.HUMAN)
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR005" not in rule_ids
        assert "QR006" not in rule_ids

    def test_multi_language_adds_consistency_rule(self, deployer):
        recipe = _make_data_recipe(languages=["en", "zh"])
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR007" in rule_ids

    def test_single_language_no_consistency_rule(self, deployer):
        recipe = _make_data_recipe(languages=["en"])
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR007" not in rule_ids

    def test_no_languages_no_consistency_rule(self, deployer):
        recipe = _make_data_recipe(languages=[])
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR007" not in rule_ids

    def test_rules_are_quality_rule_instances(self, deployer):
        recipe = _make_data_recipe()
        rules = deployer._generate_quality_rules(recipe)
        for rule in rules:
            assert isinstance(rule, QualityRule)

    def test_rule_properties(self, deployer):
        recipe = _make_data_recipe(generation_type=GenerationType.HUMAN)
        rules = deployer._generate_quality_rules(recipe)
        qr001 = next(r for r in rules if r.rule_id == "QR001")
        assert qr001.name == "非空检查"
        assert qr001.check_type == "format"
        assert qr001.severity == "error"
        assert qr001.auto_check is True


# =============================================================================
# _generate_acceptance_criteria
# =============================================================================


class TestGenerateAcceptanceCriteria:
    """Tests for _generate_acceptance_criteria()."""

    def test_returns_five_criteria(self, deployer):
        recipe = _make_data_recipe()
        criteria = deployer._generate_acceptance_criteria(recipe)
        assert len(criteria) == 5

    def test_criteria_are_instances(self, deployer):
        recipe = _make_data_recipe()
        criteria = deployer._generate_acceptance_criteria(recipe)
        for c in criteria:
            assert isinstance(c, AcceptanceCriterion)

    def test_criteria_ids(self, deployer):
        recipe = _make_data_recipe()
        criteria = deployer._generate_acceptance_criteria(recipe)
        ids = [c.criterion_id for c in criteria]
        assert ids == ["AC001", "AC002", "AC003", "AC004", "AC005"]

    def test_criteria_thresholds(self, deployer):
        recipe = _make_data_recipe()
        criteria = deployer._generate_acceptance_criteria(recipe)
        thresholds = {c.criterion_id: c.threshold for c in criteria}
        assert thresholds["AC001"] == 0.98  # completeness
        assert thresholds["AC002"] == 0.95  # accuracy
        assert thresholds["AC003"] == 0.7   # agreement
        assert thresholds["AC004"] == 1.0   # format
        assert thresholds["AC005"] == 0.9   # timeliness

    def test_required_vs_recommended(self, deployer):
        recipe = _make_data_recipe()
        criteria = deployer._generate_acceptance_criteria(recipe)
        priorities = {c.criterion_id: c.priority for c in criteria}
        assert priorities["AC001"] == "required"
        assert priorities["AC005"] == "recommended"

    def test_metric_types(self, deployer):
        recipe = _make_data_recipe()
        criteria = deployer._generate_acceptance_criteria(recipe)
        types = {c.criterion_id: c.metric_type for c in criteria}
        assert types["AC001"] == "completeness"
        assert types["AC002"] == "accuracy"
        assert types["AC003"] == "agreement"
        assert types["AC004"] == "format"
        assert types["AC005"] == "timeliness"


# =============================================================================
# _determine_review_workflow
# =============================================================================


class TestDetermineReviewWorkflow:
    """Tests for _determine_review_workflow()."""

    def test_high_human_ratio_returns_double(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.6)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.DOUBLE

    def test_exact_boundary_human_ratio_0_5_returns_single(self, deployer):
        """human_ratio=0.5 is not > 0.5, so should not be DOUBLE."""
        recipe = _make_data_recipe(human_ratio=0.5, tags=[], num_examples=1000)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.SINGLE

    def test_human_ratio_just_above_0_5(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.51, tags=[])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.DOUBLE

    def test_medical_tag_returns_expert(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=["medical"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.EXPERT

    def test_legal_tag_returns_expert(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=["legal"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.EXPERT

    def test_financial_tag_returns_expert(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=["financial"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.EXPERT

    def test_chinese_medical_tag_returns_expert(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=["医疗"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.EXPERT

    def test_chinese_legal_tag_returns_expert(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=["法律"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.EXPERT

    def test_chinese_financial_tag_returns_expert(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=["金融"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.EXPERT

    def test_large_dataset_returns_double(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=[], num_examples=60000)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.DOUBLE

    def test_boundary_50000_not_double(self, deployer):
        """Exactly 50000 is not > 50000, so should not trigger DOUBLE."""
        recipe = _make_data_recipe(human_ratio=0.2, tags=[], num_examples=50000)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.SINGLE

    def test_default_single(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=[], num_examples=1000)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.SINGLE

    def test_none_human_ratio(self, deployer):
        recipe = _make_data_recipe(human_ratio=None, tags=[], num_examples=1000)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.SINGLE

    def test_none_num_examples(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=[], num_examples=None)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.SINGLE

    def test_none_tags(self, deployer):
        recipe = _make_data_recipe(human_ratio=0.2, tags=None, num_examples=1000)
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.SINGLE

    def test_human_ratio_priority_over_tags(self, deployer):
        """High human_ratio check happens before tags check."""
        recipe = _make_data_recipe(human_ratio=0.8, tags=["medical"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.DOUBLE

    def test_expert_tag_case_insensitive(self, deployer):
        """Tags are lowercased, so mixed-case should still match."""
        recipe = _make_data_recipe(human_ratio=0.2, tags=["Medical"])
        assert deployer._determine_review_workflow(recipe) == ReviewWorkflow.EXPERT


# =============================================================================
# _estimate_timeline
# =============================================================================


class TestEstimateTimeline:
    """Tests for _estimate_timeline()."""

    def test_default_30_days(self, deployer):
        recipe = _make_data_recipe(num_examples=5000)
        days = deployer._estimate_timeline(recipe, None)
        assert days == 30

    def test_with_profile_person_days(self, deployer):
        recipe = _make_data_recipe(num_examples=5000)
        profile = _make_profile(
            estimated_person_days=80.0,
            team_size=10,
        )
        # base_days = int(80 / max(1, 10 * 0.8)) + 10 = int(80/8) + 10 = 10 + 10 = 20
        # num_examples=5000 < 100000 and >= 1000, no adjustment
        # max(14, min(180, 20)) = 20
        days = deployer._estimate_timeline(recipe, profile)
        assert days == 20

    def test_small_dataset_minimum_14(self, deployer):
        recipe = _make_data_recipe(num_examples=500)
        days = deployer._estimate_timeline(recipe, None)
        # base_days = 30, num_examples < 1000 -> base_days = max(14, 30) = 30
        # max(14, min(180, 30)) = 30
        assert days == 30

    def test_small_dataset_with_small_profile(self, deployer):
        recipe = _make_data_recipe(num_examples=500)
        profile = _make_profile(estimated_person_days=10.0, team_size=5)
        # base_days = int(10 / max(1, 5*0.8)) + 10 = int(10/4) + 10 = 2 + 10 = 12
        # num_examples=500 < 1000 -> base_days = max(14, 12) = 14
        # max(14, min(180, 14)) = 14
        days = deployer._estimate_timeline(recipe, profile)
        assert days == 14

    def test_large_dataset_60_days(self, deployer):
        recipe = _make_data_recipe(num_examples=200000)
        days = deployer._estimate_timeline(recipe, None)
        # base_days = 30, num_examples > 100000 -> base_days = max(60, 30) = 60
        # max(14, min(180, 60)) = 60
        assert days == 60

    def test_clamp_max_180(self, deployer):
        recipe = _make_data_recipe(num_examples=200000)
        profile = _make_profile(estimated_person_days=5000.0, team_size=2)
        # base_days = int(5000 / max(1, 2*0.8)) + 10 = int(5000/1.6) + 10 = 3125 + 10 = 3135
        # num_examples > 100000 -> base_days = max(60, 3135) = 3135
        # max(14, min(180, 3135)) = 180
        days = deployer._estimate_timeline(recipe, profile)
        assert days == 180

    def test_clamp_min_14(self, deployer):
        recipe = _make_data_recipe(num_examples=5000)
        profile = _make_profile(estimated_person_days=5.0, team_size=50)
        # base_days = int(5 / max(1, 50*0.8)) + 10 = int(5/40) + 10 = 0 + 10 = 10
        # num_examples neither < 1000 nor > 100000 -> no adjustment
        # max(14, min(180, 10)) = 14
        days = deployer._estimate_timeline(recipe, profile)
        assert days == 14

    def test_no_num_examples(self, deployer):
        recipe = _make_data_recipe(num_examples=None)
        days = deployer._estimate_timeline(recipe, None)
        # base_days=30, no num_examples adjustments
        # max(14, min(180, 30)) = 30
        assert days == 30

    def test_team_size_zero_defaults_to_10(self, deployer):
        """team_size=0 is falsy, so `profile.team_size or 10` gives 10."""
        recipe = _make_data_recipe(num_examples=5000)
        profile = _make_profile(estimated_person_days=80.0, team_size=0)
        # team_size=0 -> "0 or 10" = 10
        # base_days = int(80 / max(1, 10*0.8)) + 10 = int(80/8) + 10 = 10 + 10 = 20
        # max(14, min(180, 20)) = 20
        days = deployer._estimate_timeline(recipe, profile)
        assert days == 20

    def test_profile_no_person_days(self, deployer):
        recipe = _make_data_recipe(num_examples=5000)
        profile = _make_profile(estimated_person_days=0.0)
        # estimated_person_days is 0.0, which is falsy
        # Falls back to base_days=30
        days = deployer._estimate_timeline(recipe, profile)
        assert days == 30


# =============================================================================
# _generate_milestones
# =============================================================================


class TestGenerateMilestones:
    """Tests for _generate_milestones()."""

    def test_returns_five_milestones(self, deployer):
        recipe = _make_data_recipe()
        milestones = deployer._generate_milestones(recipe, None)
        assert len(milestones) == 5

    def test_milestone_names(self, deployer):
        recipe = _make_data_recipe()
        milestones = deployer._generate_milestones(recipe, None)
        names = [m.name for m in milestones]
        assert names == ["项目启动", "试标注", "正式标注", "质检验收", "交付归档"]

    def test_milestones_are_instances(self, deployer):
        recipe = _make_data_recipe()
        milestones = deployer._generate_milestones(recipe, None)
        for m in milestones:
            assert isinstance(m, Milestone)

    def test_milestone_dependencies(self, deployer):
        recipe = _make_data_recipe()
        milestones = deployer._generate_milestones(recipe, None)
        assert milestones[0].dependencies == []  # project start has no deps
        assert milestones[1].dependencies == ["项目启动"]
        assert milestones[2].dependencies == ["试标注"]
        assert milestones[3].dependencies == ["正式标注"]
        assert milestones[4].dependencies == ["质检验收"]

    def test_milestone_days_positive(self, deployer):
        recipe = _make_data_recipe()
        milestones = deployer._generate_milestones(recipe, None)
        for m in milestones:
            assert m.estimated_days > 0

    def test_milestone_deliverables_populated(self, deployer):
        recipe = _make_data_recipe()
        milestones = deployer._generate_milestones(recipe, None)
        for m in milestones:
            assert len(m.deliverables) > 0

    def test_milestones_with_profile(self, deployer):
        recipe = _make_data_recipe(num_examples=200000)
        profile = _make_profile(estimated_person_days=500.0, team_size=10)
        milestones = deployer._generate_milestones(recipe, profile)
        # Should still have 5 milestones, with larger days
        assert len(milestones) == 5
        total_days = sum(m.estimated_days for m in milestones)
        assert total_days > 0


# =============================================================================
# generate_config
# =============================================================================


class TestGenerateConfig:
    """Tests for generate_config()."""

    def test_returns_production_config(self, deployer):
        recipe = _make_data_recipe()
        config = deployer.generate_config(recipe)
        assert isinstance(config, ProductionConfig)

    def test_annotation_guide_populated(self, deployer):
        recipe = _make_data_recipe()
        config = deployer.generate_config(recipe)
        assert len(config.annotation_guide) > 0
        assert "标注指南" in config.annotation_guide

    def test_quality_rules_populated(self, deployer):
        recipe = _make_data_recipe()
        config = deployer.generate_config(recipe)
        assert len(config.quality_rules) >= 4

    def test_acceptance_criteria_populated(self, deployer):
        recipe = _make_data_recipe()
        config = deployer.generate_config(recipe)
        assert len(config.acceptance_criteria) == 5

    def test_review_workflow_set(self, deployer):
        recipe = _make_data_recipe()
        config = deployer.generate_config(recipe)
        assert isinstance(config.review_workflow, ReviewWorkflow)

    def test_milestones_populated(self, deployer):
        recipe = _make_data_recipe()
        config = deployer.generate_config(recipe)
        assert len(config.milestones) == 5

    def test_estimated_timeline_set(self, deployer):
        recipe = _make_data_recipe()
        config = deployer.generate_config(recipe)
        assert config.estimated_timeline_days >= 14

    def test_with_profile(self, deployer):
        recipe = _make_data_recipe()
        profile = _make_profile(estimated_person_days=80.0, team_size=10)
        config = deployer.generate_config(recipe, profile)
        assert isinstance(config, ProductionConfig)


# =============================================================================
# deploy
# =============================================================================


class TestDeploy:
    """Tests for deploy()."""

    def test_deploy_local_success(self, deployer, mock_discover_providers):
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle()
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.return_value = handle

        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config(annotation_guide="test guide")

        result = deployer.deploy(
            recipe=recipe,
            output="/tmp/test_output",
            provider="local",
            config=config,
            profile=profile,
        )

        assert result.success is True
        assert result.project_handle is handle
        assert result.details["output_path"] == "/tmp/test_output"
        mock_provider.create_project.assert_called_once_with(
            recipe, config, output_dir="/tmp/test_output"
        )

    def test_deploy_platform_with_submit(self, deployer, mock_discover_providers):
        """Platform provider with submit=True should call p.submit()."""
        _, mock_provider = mock_discover_providers
        # Add a second provider as "platform"
        mock_platform_class = MagicMock()
        mock_platform = MagicMock()
        mock_platform.description = "Platform provider"
        mock_platform_class.return_value = mock_platform
        deployer._providers["platform"] = mock_platform

        handle = _make_project_handle(provider="platform")
        mock_platform.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_platform.create_project.return_value = handle
        submit_result = DeploymentResult(success=True, project_handle=handle)
        mock_platform.submit.return_value = submit_result

        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config()

        result = deployer.deploy(
            recipe=recipe,
            output="my-project",
            provider="platform",
            config=config,
            profile=profile,
            submit=True,
        )

        assert result.success is True
        mock_platform.submit.assert_called_once_with(handle)

    def test_deploy_platform_no_submit(self, deployer, mock_discover_providers):
        """Platform provider with submit=False should not call p.submit()."""
        _, mock_provider = mock_discover_providers
        mock_platform = MagicMock()
        mock_platform.description = "Platform provider"
        deployer._providers["platform"] = mock_platform

        handle = _make_project_handle(provider="platform")
        mock_platform.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=["Some warning"]
        )
        mock_platform.create_project.return_value = handle

        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config()

        result = deployer.deploy(
            recipe=recipe,
            output="my-project",
            provider="platform",
            config=config,
            profile=profile,
            submit=False,
        )

        assert result.success is True
        assert "未自动提交" in result.details["message"]
        assert result.details["warnings"] == ["Some warning"]
        mock_platform.submit.assert_not_called()

    def test_deploy_provider_not_found(self, deployer):
        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config()

        result = deployer.deploy(
            recipe=recipe,
            output="/tmp/output",
            provider="nonexistent",
            config=config,
            profile=profile,
        )

        assert result.success is False
        assert "nonexistent" in result.error

    def test_deploy_validation_failure(self, deployer, mock_discover_providers):
        _, mock_provider = mock_discover_providers
        mock_provider.validate_config.return_value = ValidationResult(
            valid=False,
            errors=["Missing required field"],
            warnings=["Some warning"],
        )

        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config()

        result = deployer.deploy(
            recipe=recipe,
            output="/tmp/output",
            provider="local",
            config=config,
            profile=profile,
        )

        assert result.success is False
        assert "配置验证失败" in result.error
        assert "warnings" in result.details

    def test_deploy_create_project_exception(self, deployer, mock_discover_providers):
        _, mock_provider = mock_discover_providers
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.side_effect = RuntimeError("Disk full")

        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config()

        result = deployer.deploy(
            recipe=recipe,
            output="/tmp/output",
            provider="local",
            config=config,
            profile=profile,
        )

        assert result.success is False
        assert "Disk full" in result.error

    def test_deploy_auto_generates_profile(self, deployer, mock_discover_providers):
        """When profile=None, deploy should auto-generate one using AnnotatorProfiler."""
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle()
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.return_value = handle

        mock_profile = _make_profile()
        mock_profiler_instance = MagicMock()
        mock_profiler_instance.generate_profile.return_value = mock_profile

        recipe = _make_data_recipe()
        config = _make_production_config()

        with patch("datarecipe.profiler.AnnotatorProfiler") as MockProfiler:
            MockProfiler.return_value = mock_profiler_instance

            result = deployer.deploy(
                recipe=recipe,
                output="/tmp/output",
                provider="local",
                config=config,
                profile=None,
            )

            assert result.success is True
            mock_profiler_instance.generate_profile.assert_called_once()
            assert recipe.annotator_profile is mock_profile

    def test_deploy_auto_generates_config(self, deployer, mock_discover_providers):
        """When config=None, deploy should auto-generate one."""
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle()
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.return_value = handle

        recipe = _make_data_recipe()
        profile = _make_profile()

        result = deployer.deploy(
            recipe=recipe,
            output="/tmp/output",
            provider="local",
            config=None,
            profile=profile,
        )

        assert result.success is True
        assert recipe.production_config is not None
        assert isinstance(recipe.production_config, ProductionConfig)

    def test_deploy_auto_generates_both(self, deployer, mock_discover_providers):
        """When both profile and config are None, both should be auto-generated."""
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle()
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.return_value = handle

        mock_profile = _make_profile()
        mock_profiler_instance = MagicMock()
        mock_profiler_instance.generate_profile.return_value = mock_profile

        recipe = _make_data_recipe()

        with patch("datarecipe.profiler.AnnotatorProfiler") as MockProfiler:
            MockProfiler.return_value = mock_profiler_instance

            result = deployer.deploy(
                recipe=recipe,
                output="/tmp/output",
                provider="local",
                config=None,
                profile=None,
            )

            assert result.success is True
            assert recipe.annotator_profile is mock_profile
            assert recipe.production_config is not None

    def test_deploy_local_includes_files_created(self, deployer, mock_discover_providers):
        """Local deploy result should include files_created from handle metadata."""
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle(
            metadata={"files_created": ["recipe.yaml", "guide.md"]}
        )
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.return_value = handle

        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config()

        result = deployer.deploy(
            recipe=recipe,
            output="/tmp/output",
            provider="local",
            config=config,
            profile=profile,
        )

        assert result.details["files_created"] == ["recipe.yaml", "guide.md"]

    def test_deploy_local_includes_warnings(self, deployer, mock_discover_providers):
        """Local deploy result should include validation warnings."""
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle()
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=["标注指南为空"]
        )
        mock_provider.create_project.return_value = handle

        recipe = _make_data_recipe()
        profile = _make_profile()
        config = _make_production_config()

        result = deployer.deploy(
            recipe=recipe,
            output="/tmp/output",
            provider="local",
            config=config,
            profile=profile,
        )

        assert result.details["warnings"] == ["标注指南为空"]


# =============================================================================
# Edge cases / integration
# =============================================================================


class TestEdgeCases:
    """Edge case tests for various methods."""

    def test_unknown_generation_type(self, deployer):
        """UNKNOWN generation type should go to the 'else' branch (mixed)."""
        recipe = _make_data_recipe(generation_type=GenerationType.UNKNOWN)
        guide = deployer._generate_annotation_guide(recipe, None)
        assert "混合数据处理" in guide

    def test_quality_rules_unknown_generation_type(self, deployer):
        """UNKNOWN generation type should not add synthetic-specific rules."""
        recipe = _make_data_recipe(generation_type=GenerationType.UNKNOWN)
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR005" not in rule_ids

    def test_synthetic_with_multi_language(self, deployer):
        """Synthetic type with multiple languages should include both extra rules and QR007."""
        recipe = _make_data_recipe(
            generation_type=GenerationType.SYNTHETIC,
            languages=["en", "zh", "ja"],
        )
        rules = deployer._generate_quality_rules(recipe)
        rule_ids = [r.rule_id for r in rules]
        assert "QR005" in rule_ids  # factuality
        assert "QR006" in rule_ids  # AI trace
        assert "QR007" in rule_ids  # language consistency

    def test_generate_config_full_integration(self, deployer):
        """Full integration test for generate_config with various recipe settings."""
        recipe = _make_data_recipe(
            name="integration-test",
            description="Full integration test",
            generation_type=GenerationType.MIXED,
            languages=["en", "zh"],
            num_examples=100000,
            human_ratio=0.6,
            synthetic_ratio=0.4,
            tags=["medical", "nlp"],
        )
        profile = _make_profile(estimated_person_days=200.0, team_size=15)
        config = deployer.generate_config(recipe, profile)

        # annotation guide should be for mixed type
        assert "混合数据处理" in config.annotation_guide

        # quality rules should include synthetic extras + language consistency
        rule_ids = [r.rule_id for r in config.quality_rules]
        assert "QR005" in rule_ids
        assert "QR007" in rule_ids

        # review workflow: human_ratio=0.6 > 0.5 -> DOUBLE
        assert config.review_workflow == ReviewWorkflow.DOUBLE

        # 5 milestones
        assert len(config.milestones) == 5

        # acceptance: 5 criteria
        assert len(config.acceptance_criteria) == 5

    def test_deploy_sets_annotator_profile_on_recipe(
        self, deployer, mock_discover_providers
    ):
        """When auto-generating profile, it should be set on the recipe object."""
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle()
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.return_value = handle

        mock_profile = _make_profile()
        mock_profiler = MagicMock()
        mock_profiler.generate_profile.return_value = mock_profile

        recipe = _make_data_recipe()
        config = _make_production_config()

        with patch("datarecipe.profiler.AnnotatorProfiler") as MockProfiler:
            MockProfiler.return_value = mock_profiler
            deployer.deploy(
                recipe=recipe,
                output="/tmp/output",
                provider="local",
                config=config,
                profile=None,
            )

        assert recipe.annotator_profile is mock_profile

    def test_deploy_sets_production_config_on_recipe(
        self, deployer, mock_discover_providers
    ):
        """When auto-generating config, it should be set on the recipe object."""
        _, mock_provider = mock_discover_providers
        handle = _make_project_handle()
        mock_provider.validate_config.return_value = ValidationResult(
            valid=True, errors=[], warnings=[]
        )
        mock_provider.create_project.return_value = handle

        recipe = _make_data_recipe()
        profile = _make_profile()

        deployer.deploy(
            recipe=recipe,
            output="/tmp/output",
            provider="local",
            config=None,
            profile=profile,
        )

        assert recipe.production_config is not None
        assert isinstance(recipe.production_config, ProductionConfig)

    def test_milestone_minimum_days(self, deployer):
        """Even with very small timeline, milestones should have minimum days."""
        recipe = _make_data_recipe(num_examples=5000)
        profile = _make_profile(estimated_person_days=5.0, team_size=50)
        milestones = deployer._generate_milestones(recipe, profile)
        # total_days = 14 (clamped minimum)
        # M1: max(2, int(14*0.1)) = max(2, 1) = 2
        assert milestones[0].estimated_days >= 2
        # M2: max(3, int(14*0.15)) = max(3, 2) = 3
        assert milestones[1].estimated_days >= 3
        # M3: max(10, int(14*0.5)) = max(10, 7) = 10
        assert milestones[2].estimated_days >= 10
        # M4: max(3, int(14*0.15)) = max(3, 2) = 3
        assert milestones[3].estimated_days >= 3
        # M5: max(2, int(14*0.1)) = max(2, 1) = 2
        assert milestones[4].estimated_days >= 2
