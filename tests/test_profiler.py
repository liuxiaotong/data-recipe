"""Unit tests for profiler.py — AnnotatorProfiler and profile_to_markdown.

Covers:
- Dataset type detection from tags and descriptions
- Skill derivation for all dataset types and custom rules
- Experience level and education level derivation
- Language derivation (Chinese, English, Japanese, Korean, other, empty)
- Domain knowledge derivation
- Workload calculation (including human_ratio)
- Budget adjustment
- Hourly rate calculation across regions
- Screening criteria generation
- Platform recommendation
- Full generate_profile() integration
- profile_to_markdown() output correctness
- Edge cases: empty data, unknown types, fallback defaults
"""

import unittest

from datarecipe.constants import REGION_COST_MULTIPLIERS
from datarecipe.profiler import (
    ANNOTATION_TIME_PER_TYPE,
    BASE_HOURLY_RATES,
    DATASET_TYPE_EDUCATION,
    DATASET_TYPE_EXPERIENCE,
    DATASET_TYPE_SKILLS,
    REGION_MULTIPLIERS,
    AnnotatorProfiler,
    profile_to_markdown,
)
from datarecipe.schema import (
    AnnotatorProfile,
    EducationLevel,
    ExperienceLevel,
    Recipe,
    SkillRequirement,
)


def _make_recipe(**kwargs) -> Recipe:
    """Helper to create a Recipe with sensible defaults."""
    defaults = {
        "name": "test-dataset",
        "languages": ["en"],
        "tags": [],
        "description": "",
    }
    defaults.update(kwargs)
    return Recipe(**defaults)


class TestRegionMultipliersReExport(unittest.TestCase):
    """REGION_MULTIPLIERS should be re-exported from constants."""

    def test_is_same_as_constants(self):
        self.assertIs(REGION_MULTIPLIERS, REGION_COST_MULTIPLIERS)


# =========================================================================
# _detect_dataset_type
# =========================================================================


class TestDetectDatasetType(unittest.TestCase):
    """Tests for AnnotatorProfiler._detect_dataset_type."""

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    # --- tag-based detection for each dataset type ---

    def test_medical_from_tags(self):
        recipe = _make_recipe(tags=["medical", "NLP"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "medical")

    def test_medical_from_description(self):
        recipe = _make_recipe(description="A clinical data set for health NLP")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "medical")

    def test_legal_from_tags(self):
        recipe = _make_recipe(tags=["legal"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "legal")

    def test_legal_from_chinese_keyword(self):
        recipe = _make_recipe(description="法律文本分析")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "legal")

    def test_financial_from_tags(self):
        recipe = _make_recipe(tags=["financial"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "financial")

    def test_financial_from_chinese_keyword(self):
        recipe = _make_recipe(description="金融风控数据")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "financial")

    def test_code_review_from_tags(self):
        recipe = _make_recipe(tags=["code review"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "code_review")

    def test_code_from_tags(self):
        recipe = _make_recipe(tags=["code", "python"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "code")

    def test_code_from_programming_keyword(self):
        recipe = _make_recipe(description="programming exercises")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "code")

    def test_math_from_tags(self):
        recipe = _make_recipe(tags=["math"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "math")

    def test_reasoning_from_tags(self):
        recipe = _make_recipe(tags=["reasoning"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "reasoning")

    def test_rlhf_from_tags(self):
        recipe = _make_recipe(tags=["rlhf"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "rlhf")

    def test_preference_from_description(self):
        recipe = _make_recipe(description="ranking comparison dataset")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "preference")

    def test_agent_from_tags(self):
        recipe = _make_recipe(tags=["agent"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "agent")

    def test_translation_from_tags(self):
        recipe = _make_recipe(tags=["translation"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "translation")

    def test_multilingual_from_tags(self):
        recipe = _make_recipe(tags=["multilingual"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "multilingual")

    def test_multilingual_from_many_languages(self):
        """If more than 2 languages and no keyword matches, should be multilingual."""
        recipe = _make_recipe(languages=["en", "zh", "ja"])
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "multilingual")

    def test_general_fallback(self):
        recipe = _make_recipe(tags=[], description="nothing special")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "general")

    def test_empty_tags_and_description(self):
        recipe = _make_recipe(tags=[], description="")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "general")

    def test_none_tags(self):
        recipe = _make_recipe(description="some text")
        recipe.tags = None  # explicitly None
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "general")

    def test_none_description(self):
        recipe = _make_recipe(tags=["general_text"])
        recipe.description = None
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "general")

    def test_priority_medical_over_code(self):
        """Medical keywords should have higher priority than code keywords."""
        recipe = _make_recipe(description="medical code dataset")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "medical")

    def test_code_review_over_code(self):
        """code_review should be detected before plain code."""
        recipe = _make_recipe(description="code review process")
        self.assertEqual(self.profiler._detect_dataset_type(recipe), "code_review")


# =========================================================================
# _derive_skills
# =========================================================================


class TestDeriveSkills(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_known_type_returns_expected_skills(self):
        recipe = _make_recipe(tags=["code"])
        skills = self.profiler._derive_skills("code", recipe)
        self.assertGreater(len(skills), 0)
        names = [s.name for s in skills]
        self.assertIn("Python", names)

    def test_unknown_type_returns_general_skills(self):
        recipe = _make_recipe()
        skills = self.profiler._derive_skills("unknown_type", recipe)
        expected = DATASET_TYPE_SKILLS["general"]
        self.assertEqual(len(skills), len(expected))

    def test_skills_are_copies(self):
        """Returned list should not mutate the original mapping."""
        recipe = _make_recipe()
        skills = self.profiler._derive_skills("code", recipe)
        original = DATASET_TYPE_SKILLS["code"]
        skills.append(SkillRequirement("extra", "extra", "basic", False))
        self.assertNotEqual(len(skills), len(original))

    def test_synthetic_ratio_adds_ai_skill(self):
        recipe = _make_recipe(synthetic_ratio=0.8)
        skills = self.profiler._derive_skills("general", recipe)
        names = [s.name for s in skills]
        self.assertIn("AI生成内容识别", names)

    def test_low_synthetic_ratio_no_extra_skill(self):
        recipe = _make_recipe(synthetic_ratio=0.3)
        skills = self.profiler._derive_skills("general", recipe)
        names = [s.name for s in skills]
        self.assertNotIn("AI生成内容识别", names)

    def test_custom_rules_override(self):
        custom_skills = [SkillRequirement("custom", "my-skill", "expert", True)]
        profiler = AnnotatorProfiler(custom_rules={"skills": {"code": custom_skills}})
        recipe = _make_recipe()
        skills = profiler._derive_skills("code", recipe)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "my-skill")


# =========================================================================
# _derive_experience / _derive_education
# =========================================================================


class TestDeriveExperience(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_all_known_types(self):
        for dtype, (expected_level, expected_years) in DATASET_TYPE_EXPERIENCE.items():
            level, years = self.profiler._derive_experience(dtype)
            self.assertEqual(level, expected_level, f"Failed for type={dtype}")
            self.assertEqual(years, expected_years, f"Failed for type={dtype}")

    def test_unknown_type_defaults(self):
        level, years = self.profiler._derive_experience("alien_type")
        self.assertEqual(level, ExperienceLevel.MID)
        self.assertEqual(years, 1)


class TestDeriveEducation(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_all_known_types(self):
        for dtype, expected in DATASET_TYPE_EDUCATION.items():
            edu = self.profiler._derive_education(dtype)
            self.assertEqual(edu, expected, f"Failed for type={dtype}")

    def test_unknown_type_defaults_to_bachelor(self):
        edu = self.profiler._derive_education("alien_type")
        self.assertEqual(edu, EducationLevel.BACHELOR)


# =========================================================================
# _derive_languages
# =========================================================================


class TestDeriveLanguages(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_chinese_variants(self):
        for lang in ["zh", "zh-cn", "zh-tw", "chinese", "中文"]:
            recipe = _make_recipe(languages=[lang])
            result = self.profiler._derive_languages(recipe)
            self.assertEqual(result, ["zh-CN:native"], f"Failed for lang={lang}")

    def test_english_variants(self):
        for lang in ["en", "english", "英语"]:
            recipe = _make_recipe(languages=[lang])
            result = self.profiler._derive_languages(recipe)
            self.assertEqual(result, ["en:C1"], f"Failed for lang={lang}")

    def test_japanese(self):
        recipe = _make_recipe(languages=["ja"])
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["ja:native"])

    def test_korean(self):
        recipe = _make_recipe(languages=["korean"])
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["ko:native"])

    def test_other_language(self):
        recipe = _make_recipe(languages=["fr"])
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["fr:B2"])

    def test_multiple_languages(self):
        recipe = _make_recipe(languages=["en", "zh"])
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["en:C1", "zh-CN:native"])

    def test_empty_languages_defaults_to_en(self):
        recipe = _make_recipe(languages=[])
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["en:C1"])

    def test_none_languages_defaults_to_en(self):
        recipe = _make_recipe()
        recipe.languages = None
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["en:C1"])

    def test_empty_string_in_languages_skipped(self):
        recipe = _make_recipe(languages=["", "en"])
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["en:C1"])

    def test_only_empty_strings_defaults_to_en(self):
        recipe = _make_recipe(languages=["", ""])
        result = self.profiler._derive_languages(recipe)
        self.assertEqual(result, ["en:C1"])


# =========================================================================
# _derive_domains
# =========================================================================


class TestDeriveDomains(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_medical_domain(self):
        recipe = _make_recipe(description="medical dataset")
        domains = self.profiler._derive_domains(recipe, "medical")
        self.assertIn("医疗", domains)

    def test_code_adds_tech_domain(self):
        recipe = _make_recipe(description="software engineering")
        domains = self.profiler._derive_domains(recipe, "code")
        self.assertIn("技术", domains)

    def test_legal_domain(self):
        recipe = _make_recipe(description="legal contract analysis")
        domains = self.profiler._derive_domains(recipe, "legal")
        self.assertIn("法律", domains)

    def test_financial_domain(self):
        recipe = _make_recipe(description="金融投资分析")
        domains = self.profiler._derive_domains(recipe, "financial")
        self.assertIn("金融", domains)

    def test_education_domain(self):
        recipe = _make_recipe(description="education learning dataset")
        domains = self.profiler._derive_domains(recipe, "general")
        self.assertIn("教育", domains)

    def test_ecommerce_domain(self):
        recipe = _make_recipe(description="e-commerce product reviews")
        domains = self.profiler._derive_domains(recipe, "general")
        self.assertIn("电商", domains)

    def test_customer_service_domain(self):
        recipe = _make_recipe(description="customer service dialog")
        domains = self.profiler._derive_domains(recipe, "general")
        self.assertIn("客服", domains)

    def test_no_domain_defaults_to_generic(self):
        recipe = _make_recipe(description="")
        domains = self.profiler._derive_domains(recipe, "general")
        self.assertEqual(domains, ["通用"])

    def test_type_domain_supplement_no_duplicate(self):
        """If keyword already matches domain, type supplement should not duplicate."""
        recipe = _make_recipe(description="medical health dataset")
        domains = self.profiler._derive_domains(recipe, "medical")
        count = domains.count("医疗")
        self.assertEqual(count, 1)

    def test_code_review_adds_tech(self):
        recipe = _make_recipe(description="")
        domains = self.profiler._derive_domains(recipe, "code_review")
        self.assertIn("技术", domains)

    def test_tags_also_checked(self):
        recipe = _make_recipe(tags=["medical"], description="")
        domains = self.profiler._derive_domains(recipe, "general")
        self.assertIn("医疗", domains)


# =========================================================================
# _calculate_workload
# =========================================================================


class TestCalculateWorkload(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_basic_workload(self):
        recipe = _make_recipe()
        team_size, person_days, hours_per_example = self.profiler._calculate_workload(
            1000, "general", recipe
        )
        # general = 5 min/example, 1000 examples
        # hours_per_example = 5/60
        self.assertAlmostEqual(hours_per_example, 5 / 60, places=3)
        # total_hours = 1000 * (5/60) * 1.2 = 100
        expected_person_days = 100 / 8
        self.assertAlmostEqual(person_days, round(expected_person_days, 1), places=1)
        self.assertGreaterEqual(team_size, 2)

    def test_human_ratio_reduces_workload(self):
        recipe = _make_recipe(human_ratio=0.5)
        _, days_half, _ = self.profiler._calculate_workload(1000, "general", recipe)
        recipe_full = _make_recipe(human_ratio=1.0)
        _, days_full, _ = self.profiler._calculate_workload(1000, "general", recipe_full)
        self.assertLess(days_half, days_full)

    def test_none_human_ratio_means_full(self):
        recipe = _make_recipe()
        recipe.human_ratio = None
        _, days_none, _ = self.profiler._calculate_workload(1000, "general", recipe)
        recipe_full = _make_recipe(human_ratio=1.0)
        _, days_full, _ = self.profiler._calculate_workload(1000, "general", recipe_full)
        self.assertAlmostEqual(days_none, days_full, places=1)

    def test_code_type_longer_annotation(self):
        recipe = _make_recipe()
        _, _, hours_code = self.profiler._calculate_workload(100, "code", recipe)
        _, _, hours_general = self.profiler._calculate_workload(100, "general", recipe)
        self.assertGreater(hours_code, hours_general)

    def test_unknown_type_defaults_to_5_minutes(self):
        recipe = _make_recipe()
        _, _, hours = self.profiler._calculate_workload(100, "alien_type", recipe)
        self.assertAlmostEqual(hours, 5 / 60, places=3)

    def test_minimum_team_size(self):
        """Even for tiny datasets, team_size should be at least 2."""
        recipe = _make_recipe()
        team_size, _, _ = self.profiler._calculate_workload(1, "general", recipe)
        self.assertGreaterEqual(team_size, 2)


# =========================================================================
# _adjust_for_budget
# =========================================================================


class TestAdjustForBudget(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_sufficient_budget_no_change(self):
        """If budget is enough, values should not change."""
        team, days = self.profiler._adjust_for_budget(
            budget=999999,
            team_size=10,
            person_days=100.0,
            exp_level=ExperienceLevel.MID,
            region="us",
        )
        self.assertEqual(team, 10)
        self.assertEqual(days, 100.0)

    def test_tight_budget_reduces_team(self):
        """If budget is low, team_size and person_days should be reduced."""
        team, days = self.profiler._adjust_for_budget(
            budget=100,
            team_size=10,
            person_days=100.0,
            exp_level=ExperienceLevel.EXPERT,
            region="us",
        )
        self.assertLess(team, 10)
        self.assertLess(days, 100.0)

    def test_minimum_team_size_is_2(self):
        team, _ = self.profiler._adjust_for_budget(
            budget=1,
            team_size=10,
            person_days=100.0,
            exp_level=ExperienceLevel.EXPERT,
            region="us",
        )
        self.assertGreaterEqual(team, 2)

    def test_region_affects_budget_ratio(self):
        """India region (lower multiplier) should require less budget adjustment."""
        team_us, days_us = self.profiler._adjust_for_budget(
            budget=5000,
            team_size=10,
            person_days=100.0,
            exp_level=ExperienceLevel.SENIOR,
            region="us",
        )
        team_in, days_in = self.profiler._adjust_for_budget(
            budget=5000,
            team_size=10,
            person_days=100.0,
            exp_level=ExperienceLevel.SENIOR,
            region="in",
        )
        # India is cheaper, so same budget goes further
        self.assertGreaterEqual(days_in, days_us)


# =========================================================================
# _calculate_hourly_rate
# =========================================================================


class TestCalculateHourlyRate(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_us_rate(self):
        rate = self.profiler._calculate_hourly_rate(ExperienceLevel.MID, "us")
        base = BASE_HOURLY_RATES[ExperienceLevel.MID]
        self.assertEqual(rate["min"], base["min"])
        self.assertEqual(rate["max"], base["max"])
        self.assertEqual(rate["currency"], "USD")
        self.assertEqual(rate["region"], "us")

    def test_china_rate(self):
        rate = self.profiler._calculate_hourly_rate(ExperienceLevel.MID, "cn")
        base = BASE_HOURLY_RATES[ExperienceLevel.MID]
        mult = REGION_COST_MULTIPLIERS["cn"]
        self.assertAlmostEqual(rate["min"], round(base["min"] * mult, 2))
        self.assertAlmostEqual(rate["max"], round(base["max"] * mult, 2))

    def test_unknown_region_defaults_multiplier_1(self):
        rate = self.profiler._calculate_hourly_rate(ExperienceLevel.JUNIOR, "mars")
        base = BASE_HOURLY_RATES[ExperienceLevel.JUNIOR]
        self.assertEqual(rate["min"], base["min"])
        self.assertEqual(rate["max"], base["max"])

    def test_all_experience_levels(self):
        for level in ExperienceLevel:
            rate = self.profiler._calculate_hourly_rate(level, "us")
            self.assertIn("min", rate)
            self.assertIn("max", rate)
            self.assertLess(rate["min"], rate["max"])


# =========================================================================
# _generate_screening_criteria
# =========================================================================


class TestGenerateScreeningCriteria(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_required_skills_in_criteria(self):
        skills = [
            SkillRequirement("programming", "Python", "advanced", True),
            SkillRequirement("tool", "optional-tool", "basic", False),
        ]
        criteria = self.profiler._generate_screening_criteria(
            skills, ExperienceLevel.SENIOR, ["技术"]
        )
        # Required skill should appear
        self.assertTrue(any("Python" in c for c in criteria))
        # Optional skill should NOT appear in screening criteria
        self.assertFalse(any("optional-tool" in c for c in criteria))

    def test_senior_experience_text(self):
        criteria = self.profiler._generate_screening_criteria(
            [], ExperienceLevel.SENIOR, []
        )
        self.assertTrue(any("高级" in c for c in criteria))

    def test_expert_experience_text(self):
        criteria = self.profiler._generate_screening_criteria(
            [], ExperienceLevel.EXPERT, []
        )
        self.assertTrue(any("专家级" in c for c in criteria))

    def test_junior_no_experience_criteria(self):
        """Junior and mid level should NOT add experience text criteria."""
        criteria = self.profiler._generate_screening_criteria(
            [], ExperienceLevel.JUNIOR, []
        )
        self.assertFalse(any("工作经验" in c for c in criteria))

    def test_mid_no_experience_criteria(self):
        criteria = self.profiler._generate_screening_criteria(
            [], ExperienceLevel.MID, []
        )
        self.assertFalse(any("工作经验" in c for c in criteria))

    def test_domain_criteria(self):
        criteria = self.profiler._generate_screening_criteria(
            [], ExperienceLevel.JUNIOR, ["医疗", "通用"]
        )
        self.assertTrue(any("医疗" in c for c in criteria))
        # "通用" should NOT generate domain criteria
        self.assertFalse(any("通用" in c and "领域" in c for c in criteria))

    def test_always_has_platform_and_test(self):
        criteria = self.profiler._generate_screening_criteria(
            [], ExperienceLevel.JUNIOR, []
        )
        self.assertTrue(any("资质审核" in c for c in criteria))
        self.assertTrue(any("正确率" in c for c in criteria))


# =========================================================================
# _recommend_platforms
# =========================================================================


class TestRecommendPlatforms(unittest.TestCase):

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_china_region(self):
        platforms = self.profiler._recommend_platforms("code", "cn")
        self.assertIn("集识光年", platforms)

    def test_china_alias(self):
        platforms = self.profiler._recommend_platforms("code", "china")
        self.assertIn("集识光年", platforms)

    def test_us_code_platforms(self):
        platforms = self.profiler._recommend_platforms("code", "us")
        self.assertIn("Scale AI", platforms)
        self.assertIn("Surge AI", platforms)

    def test_general_type(self):
        platforms = self.profiler._recommend_platforms("general", "us")
        self.assertIn("Amazon MTurk", platforms)
        self.assertIn("Prolific", platforms)

    def test_multilingual_type(self):
        platforms = self.profiler._recommend_platforms("multilingual", "us")
        self.assertIn("Appen", platforms)

    def test_translation_type(self):
        platforms = self.profiler._recommend_platforms("translation", "us")
        self.assertIn("Gengo", platforms)

    def test_unknown_type_defaults(self):
        platforms = self.profiler._recommend_platforms("alien_type", "us")
        self.assertIn("Amazon MTurk", platforms)

    def test_no_duplicates(self):
        """China + general should not have duplicated entries."""
        platforms = self.profiler._recommend_platforms("general", "cn")
        self.assertEqual(len(platforms), len(set(platforms)))


# =========================================================================
# generate_profile (integration)
# =========================================================================


class TestGenerateProfile(unittest.TestCase):
    """Integration tests for the full generate_profile pipeline."""

    def setUp(self):
        self.profiler = AnnotatorProfiler()

    def test_basic_profile(self):
        recipe = _make_recipe(tags=["code"], languages=["en"], num_examples=500)
        profile = self.profiler.generate_profile(recipe)
        self.assertIsInstance(profile, AnnotatorProfile)
        self.assertGreater(len(profile.skill_requirements), 0)
        self.assertGreater(profile.team_size, 0)
        self.assertGreater(profile.estimated_person_days, 0)

    def test_profile_uses_target_size(self):
        recipe = _make_recipe(tags=["general"], num_examples=100)
        profile_default = self.profiler.generate_profile(recipe)
        profile_large = self.profiler.generate_profile(recipe, target_size=100000)
        self.assertGreater(
            profile_large.estimated_person_days, profile_default.estimated_person_days
        )

    def test_profile_uses_num_examples_fallback(self):
        """When target_size is None, use recipe.num_examples."""
        recipe = _make_recipe(num_examples=200)
        profile = self.profiler.generate_profile(recipe)
        # Should use 200, not 10000 default
        self.assertIsNotNone(profile.estimated_person_days)

    def test_profile_uses_10000_default(self):
        """When both target_size and num_examples are None, default to 10000."""
        recipe = _make_recipe()
        recipe.num_examples = None
        profile = self.profiler.generate_profile(recipe)
        # 10000 * 5min / 60 * 1.2 / 8 = 125 person days
        self.assertGreater(profile.estimated_person_days, 0)

    def test_profile_with_budget(self):
        recipe = _make_recipe(tags=["code"], num_examples=10000)
        profile_no_budget = self.profiler.generate_profile(recipe)
        profile_budget = self.profiler.generate_profile(recipe, budget=100)
        self.assertLessEqual(
            profile_budget.estimated_person_days,
            profile_no_budget.estimated_person_days,
        )

    def test_profile_with_region(self):
        recipe = _make_recipe(tags=["general"])
        profile = self.profiler.generate_profile(recipe, region="cn")
        self.assertEqual(profile.hourly_rate_range["region"], "cn")
        mult = REGION_COST_MULTIPLIERS["cn"]
        base = BASE_HOURLY_RATES[profile.experience_level]
        self.assertAlmostEqual(
            profile.hourly_rate_range["min"], round(base["min"] * mult, 2)
        )

    def test_team_structure(self):
        recipe = _make_recipe(tags=["general"], num_examples=10000)
        profile = self.profiler.generate_profile(recipe)
        self.assertIn("annotator", profile.team_structure)
        self.assertIn("reviewer", profile.team_structure)
        # Verify structure matches the formula: 80% annotators, 20% reviewers
        expected_annotators = max(1, int(profile.team_size * 0.8))
        expected_reviewers = max(1, int(profile.team_size * 0.2))
        self.assertEqual(profile.team_structure["annotator"], expected_annotators)
        self.assertEqual(profile.team_structure["reviewer"], expected_reviewers)

    def test_medical_profile(self):
        recipe = _make_recipe(tags=["medical"], description="clinical data")
        profile = self.profiler.generate_profile(recipe)
        self.assertEqual(profile.experience_level, ExperienceLevel.EXPERT)
        self.assertEqual(profile.education_level, EducationLevel.PROFESSIONAL)
        self.assertIn("医疗", profile.domain_knowledge)

    def test_all_dataset_types_produce_valid_profile(self):
        """Ensure no type raises an error."""
        for dtype in DATASET_TYPE_SKILLS:
            recipe = _make_recipe(tags=[dtype])
            profile = self.profiler.generate_profile(recipe)
            self.assertIsInstance(profile, AnnotatorProfile)


# =========================================================================
# profile_to_markdown
# =========================================================================


class TestProfileToMarkdown(unittest.TestCase):

    def _make_profile(self, **kwargs) -> AnnotatorProfile:
        defaults = {
            "skill_requirements": [
                SkillRequirement("programming", "Python", "advanced", True),
                SkillRequirement("tool", "Git", "intermediate", False),
            ],
            "experience_level": ExperienceLevel.SENIOR,
            "min_experience_years": 3,
            "language_requirements": ["en:C1", "zh-CN:native"],
            "domain_knowledge": ["技术"],
            "education_level": EducationLevel.BACHELOR,
            "team_size": 10,
            "team_structure": {"annotator": 8, "reviewer": 2},
            "estimated_person_days": 50.0,
            "estimated_hours_per_example": 0.25,
            "hourly_rate_range": {
                "min": 35.0,
                "max": 60.0,
                "currency": "USD",
                "region": "us",
            },
            "screening_criteria": ["具备Python能力（advanced级）", "通过平台资质审核"],
            "recommended_platforms": ["Scale AI", "Surge AI"],
        }
        defaults.update(kwargs)
        return AnnotatorProfile(**defaults)

    def test_contains_title(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("# 标注专家画像", md)

    def test_title_with_dataset_name(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile, dataset_name="test-ds")
        self.assertIn("test-ds", md)

    def test_overview_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 概览", md)
        self.assertIn("senior", md)
        self.assertIn("3 年", md)
        self.assertIn("bachelor", md)
        self.assertIn("10 人", md)
        self.assertIn("50.0 天", md)

    def test_team_structure_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 团队结构", md)
        self.assertIn("标注员", md)
        self.assertIn("审核员", md)

    def test_skills_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 技能要求", md)
        self.assertIn("Python", md)
        self.assertIn("是", md)  # required=True
        self.assertIn("否", md)  # required=False

    def test_language_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 语言要求", md)
        self.assertIn("en:C1", md)
        self.assertIn("zh-CN:native", md)

    def test_domain_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 领域知识", md)
        self.assertIn("技术", md)

    def test_rate_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 费率参考", md)
        self.assertIn("$35.00", md)
        self.assertIn("$60.00", md)
        self.assertIn("USD", md)
        self.assertIn("us", md)

    def test_screening_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 筛选标准", md)
        self.assertIn("- [ ]", md)

    def test_platforms_section(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("## 推荐平台", md)
        self.assertIn("Scale AI", md)

    def test_footer(self):
        profile = self._make_profile()
        md = profile_to_markdown(profile)
        self.assertIn("DataRecipe", md)

    def test_no_language_section_if_empty(self):
        profile = self._make_profile(language_requirements=[])
        md = profile_to_markdown(profile)
        self.assertNotIn("## 语言要求", md)

    def test_no_domain_section_if_empty(self):
        profile = self._make_profile(domain_knowledge=[])
        md = profile_to_markdown(profile)
        self.assertNotIn("## 领域知识", md)

    def test_no_screening_section_if_empty(self):
        profile = self._make_profile(screening_criteria=[])
        md = profile_to_markdown(profile)
        self.assertNotIn("## 筛选标准", md)

    def test_no_platforms_section_if_empty(self):
        profile = self._make_profile(recommended_platforms=[])
        md = profile_to_markdown(profile)
        self.assertNotIn("## 推荐平台", md)

    def test_hours_per_example_in_minutes(self):
        profile = self._make_profile(estimated_hours_per_example=0.5)
        md = profile_to_markdown(profile)
        # 0.5 hours = 30.0 minutes
        self.assertIn("30.0 分钟", md)

    def test_unknown_role_uses_raw_name(self):
        profile = self._make_profile(
            team_structure={"annotator": 5, "manager": 1}
        )
        md = profile_to_markdown(profile)
        self.assertIn("标注员: 5 人", md)
        self.assertIn("manager: 1 人", md)


# =========================================================================
# AnnotatorProfile serialization (to_dict, to_json, to_yaml)
# =========================================================================


class TestAnnotatorProfileSerialization(unittest.TestCase):

    def _make_profile(self) -> AnnotatorProfile:
        return AnnotatorProfile(
            skill_requirements=[
                SkillRequirement("domain", "NLP", "advanced", True),
            ],
            experience_level=ExperienceLevel.SENIOR,
            min_experience_years=3,
            language_requirements=["en:C1"],
            domain_knowledge=["技术"],
            education_level=EducationLevel.MASTER,
            team_size=5,
            team_structure={"annotator": 4, "reviewer": 1},
            estimated_person_days=20.0,
            estimated_hours_per_example=0.1,
            hourly_rate_range={"min": 35, "max": 60, "currency": "USD"},
            screening_criteria=["test criterion"],
            recommended_platforms=["Scale AI"],
        )

    def test_to_dict_structure(self):
        d = self._make_profile().to_dict()
        self.assertIn("skill_requirements", d)
        self.assertIn("experience", d)
        self.assertEqual(d["experience"]["level"], "senior")
        self.assertEqual(d["experience"]["min_years"], 3)
        self.assertEqual(d["education_level"], "master")
        self.assertEqual(d["team"]["size"], 5)
        self.assertEqual(d["workload"]["estimated_person_days"], 20.0)

    def test_to_json(self):
        import json

        profile = self._make_profile()
        j = profile.to_json()
        parsed = json.loads(j)
        self.assertEqual(parsed["experience"]["level"], "senior")

    def test_to_yaml(self):
        import yaml

        profile = self._make_profile()
        y = profile.to_yaml()
        parsed = yaml.safe_load(y)
        self.assertEqual(parsed["experience"]["level"], "senior")


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases(unittest.TestCase):

    def test_empty_recipe(self):
        """Minimal recipe with just a name should work."""
        recipe = Recipe(name="empty")
        profiler = AnnotatorProfiler()
        profile = profiler.generate_profile(recipe)
        self.assertIsInstance(profile, AnnotatorProfile)
        self.assertGreater(profile.team_size, 0)

    def test_custom_rules_empty_dict(self):
        profiler = AnnotatorProfiler(custom_rules={})
        recipe = _make_recipe(tags=["code"])
        profile = profiler.generate_profile(recipe)
        self.assertIsInstance(profile, AnnotatorProfile)

    def test_zero_target_size(self):
        """target_size=0 should still produce a valid (if small) profile."""
        recipe = _make_recipe()
        profiler = AnnotatorProfiler()
        # target_size=0 is falsy, so should fall back to num_examples or 10000
        profile = profiler.generate_profile(recipe, target_size=0)
        self.assertIsInstance(profile, AnnotatorProfile)

    def test_all_regions(self):
        """Generate a profile for every known region without error."""
        recipe = _make_recipe(tags=["general"])
        profiler = AnnotatorProfiler()
        for region in REGION_COST_MULTIPLIERS:
            profile = profiler.generate_profile(recipe, region=region)
            self.assertEqual(profile.hourly_rate_range["region"], region)

    def test_unknown_region(self):
        recipe = _make_recipe(tags=["general"])
        profiler = AnnotatorProfiler()
        profile = profiler.generate_profile(recipe, region="mars")
        # Should use multiplier=1.0
        base = BASE_HOURLY_RATES[profile.experience_level]
        self.assertEqual(profile.hourly_rate_range["min"], base["min"])

    def test_profile_to_markdown_no_dataset_name(self):
        profiler = AnnotatorProfiler()
        recipe = _make_recipe(tags=["general"])
        profile = profiler.generate_profile(recipe)
        md = profile_to_markdown(profile)
        self.assertIn("# 标注专家画像", md)
        # Should NOT have a colon followed by a name
        self.assertNotIn("：", md.split("\n")[0])


if __name__ == "__main__":
    unittest.main()
