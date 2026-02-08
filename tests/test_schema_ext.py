"""Comprehensive unit tests for datarecipe.schema module.

Covers all dataclasses, enums, to_dict() methods, from_recipe(),
default values, edge cases, and the DeploymentProvider protocol.
"""

import json
import unittest
from dataclasses import fields

from datarecipe.schema import (
    AcceptanceCriterion,
    AnnotatorMatch,
    AnnotatorProfile,
    Cost,
    DataRecipe,
    DeploymentProvider,
    DeploymentResult,
    EducationLevel,
    EnhancedCost,
    ExperienceLevel,
    GenerationMethod,
    GenerationType,
    Milestone,
    ProductionConfig,
    ProjectHandle,
    ProjectStatus,
    QualityRule,
    Recipe,
    Reproducibility,
    ReviewWorkflow,
    SkillRequirement,
    SourceType,
    ValidationResult,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestGenerationType(unittest.TestCase):
    """Test GenerationType enum values."""

    def test_all_values(self):
        self.assertEqual(GenerationType.SYNTHETIC.value, "synthetic")
        self.assertEqual(GenerationType.HUMAN.value, "human")
        self.assertEqual(GenerationType.MIXED.value, "mixed")
        self.assertEqual(GenerationType.UNKNOWN.value, "unknown")

    def test_member_count(self):
        self.assertEqual(len(GenerationType), 4)


class TestSourceType(unittest.TestCase):
    """Test SourceType enum values."""

    def test_all_values(self):
        self.assertEqual(SourceType.HUGGINGFACE.value, "huggingface")
        self.assertEqual(SourceType.GITHUB.value, "github")
        self.assertEqual(SourceType.OPENAI.value, "openai")
        self.assertEqual(SourceType.LOCAL.value, "local")
        self.assertEqual(SourceType.WEB.value, "web")
        self.assertEqual(SourceType.UNKNOWN.value, "unknown")

    def test_member_count(self):
        self.assertEqual(len(SourceType), 6)


class TestExperienceLevel(unittest.TestCase):
    """Test ExperienceLevel enum values."""

    def test_all_values(self):
        self.assertEqual(ExperienceLevel.JUNIOR.value, "junior")
        self.assertEqual(ExperienceLevel.MID.value, "mid")
        self.assertEqual(ExperienceLevel.SENIOR.value, "senior")
        self.assertEqual(ExperienceLevel.EXPERT.value, "expert")


class TestEducationLevel(unittest.TestCase):
    """Test EducationLevel enum values."""

    def test_all_values(self):
        self.assertEqual(EducationLevel.HIGH_SCHOOL.value, "high_school")
        self.assertEqual(EducationLevel.BACHELOR.value, "bachelor")
        self.assertEqual(EducationLevel.MASTER.value, "master")
        self.assertEqual(EducationLevel.PHD.value, "phd")
        self.assertEqual(EducationLevel.PROFESSIONAL.value, "professional")


class TestReviewWorkflow(unittest.TestCase):
    """Test ReviewWorkflow enum values."""

    def test_all_values(self):
        self.assertEqual(ReviewWorkflow.SINGLE.value, "single")
        self.assertEqual(ReviewWorkflow.DOUBLE.value, "double")
        self.assertEqual(ReviewWorkflow.EXPERT.value, "expert")


# =============================================================================
# Cost Tests
# =============================================================================


class TestCost(unittest.TestCase):
    """Test Cost dataclass."""

    def test_defaults(self):
        cost = Cost()
        self.assertIsNone(cost.estimated_total_usd)
        self.assertIsNone(cost.api_calls_usd)
        self.assertIsNone(cost.human_annotation_usd)
        self.assertIsNone(cost.compute_usd)
        self.assertEqual(cost.confidence, "low")
        self.assertIsNone(cost.low_estimate_usd)
        self.assertIsNone(cost.high_estimate_usd)
        self.assertEqual(cost.assumptions, [])
        self.assertIsNone(cost.tokens_estimated)

    def test_to_dict_minimal(self):
        cost = Cost()
        d = cost.to_dict()
        self.assertIsNone(d["estimated_total_usd"])
        self.assertEqual(d["confidence"], "low")
        self.assertEqual(d["breakdown"]["api_calls"], None)
        self.assertEqual(d["breakdown"]["human_annotation"], None)
        self.assertEqual(d["breakdown"]["compute"], None)
        # Optional fields should not be present when None/empty
        self.assertNotIn("low_estimate_usd", d)
        self.assertNotIn("high_estimate_usd", d)
        self.assertNotIn("assumptions", d)
        self.assertNotIn("tokens_estimated", d)

    def test_to_dict_full(self):
        cost = Cost(
            estimated_total_usd=5000.0,
            api_calls_usd=2000.0,
            human_annotation_usd=2500.0,
            compute_usd=500.0,
            confidence="high",
            low_estimate_usd=4000.0,
            high_estimate_usd=6000.0,
            assumptions=["GPT-4 pricing", "10 annotators"],
            tokens_estimated=1000000,
        )
        d = cost.to_dict()
        self.assertEqual(d["estimated_total_usd"], 5000.0)
        self.assertEqual(d["breakdown"]["api_calls"], 2000.0)
        self.assertEqual(d["breakdown"]["human_annotation"], 2500.0)
        self.assertEqual(d["breakdown"]["compute"], 500.0)
        self.assertEqual(d["confidence"], "high")
        self.assertEqual(d["low_estimate_usd"], 4000.0)
        self.assertEqual(d["high_estimate_usd"], 6000.0)
        self.assertEqual(d["assumptions"], ["GPT-4 pricing", "10 annotators"])
        self.assertEqual(d["tokens_estimated"], 1000000)

    def test_to_dict_partial_optional_fields(self):
        """Only low_estimate set, high_estimate not set."""
        cost = Cost(low_estimate_usd=100.0)
        d = cost.to_dict()
        self.assertIn("low_estimate_usd", d)
        self.assertNotIn("high_estimate_usd", d)

    def test_to_dict_assumptions_empty_not_included(self):
        cost = Cost(assumptions=[])
        d = cost.to_dict()
        self.assertNotIn("assumptions", d)

    def test_to_dict_assumptions_nonempty_included(self):
        cost = Cost(assumptions=["assumption1"])
        d = cost.to_dict()
        self.assertIn("assumptions", d)
        self.assertEqual(d["assumptions"], ["assumption1"])

    def test_to_dict_tokens_estimated_zero_included(self):
        """tokens_estimated=0 is not None, so it should be included."""
        cost = Cost(tokens_estimated=0)
        d = cost.to_dict()
        self.assertIn("tokens_estimated", d)
        self.assertEqual(d["tokens_estimated"], 0)


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestReproducibility(unittest.TestCase):
    """Test Reproducibility dataclass."""

    def test_defaults(self):
        r = Reproducibility(score=5)
        self.assertEqual(r.score, 5)
        self.assertEqual(r.available, [])
        self.assertEqual(r.missing, [])
        self.assertIsNone(r.notes)

    def test_to_dict(self):
        r = Reproducibility(
            score=8,
            available=["code", "data"],
            missing=["environment"],
            notes="Mostly reproducible",
        )
        d = r.to_dict()
        self.assertEqual(d["score"], 8)
        self.assertEqual(d["available"], ["code", "data"])
        self.assertEqual(d["missing"], ["environment"])
        self.assertEqual(d["notes"], "Mostly reproducible")

    def test_to_dict_notes_none(self):
        r = Reproducibility(score=3)
        d = r.to_dict()
        self.assertIsNone(d["notes"])


# =============================================================================
# GenerationMethod Tests
# =============================================================================


class TestGenerationMethod(unittest.TestCase):
    """Test GenerationMethod dataclass."""

    def test_defaults(self):
        gm = GenerationMethod(method_type="distillation")
        self.assertEqual(gm.method_type, "distillation")
        self.assertIsNone(gm.teacher_model)
        self.assertFalse(gm.prompt_template_available)
        self.assertIsNone(gm.platform)
        self.assertEqual(gm.details, {})

    def test_to_dict_minimal(self):
        gm = GenerationMethod(method_type="web_scrape")
        d = gm.to_dict()
        self.assertEqual(d, {"type": "web_scrape"})

    def test_to_dict_with_teacher_model(self):
        gm = GenerationMethod(method_type="distillation", teacher_model="GPT-4")
        d = gm.to_dict()
        self.assertEqual(d["type"], "distillation")
        self.assertEqual(d["teacher_model"], "GPT-4")

    def test_to_dict_with_prompt_template(self):
        gm = GenerationMethod(method_type="synthetic", prompt_template_available=True)
        d = gm.to_dict()
        self.assertEqual(d["prompt_template"], "available")

    def test_to_dict_prompt_template_false_not_included(self):
        gm = GenerationMethod(method_type="synthetic", prompt_template_available=False)
        d = gm.to_dict()
        self.assertNotIn("prompt_template", d)

    def test_to_dict_with_platform(self):
        gm = GenerationMethod(method_type="human_annotation", platform="Surge AI")
        d = gm.to_dict()
        self.assertEqual(d["platform"], "Surge AI")

    def test_to_dict_platform_none_not_included(self):
        gm = GenerationMethod(method_type="human_annotation")
        d = gm.to_dict()
        self.assertNotIn("platform", d)

    def test_to_dict_with_details(self):
        gm = GenerationMethod(
            method_type="distillation",
            details={"temperature": 0.7, "top_p": 0.9},
        )
        d = gm.to_dict()
        self.assertEqual(d["type"], "distillation")
        self.assertEqual(d["temperature"], 0.7)
        self.assertEqual(d["top_p"], 0.9)

    def test_to_dict_details_empty_not_merged(self):
        gm = GenerationMethod(method_type="test", details={})
        d = gm.to_dict()
        self.assertEqual(d, {"type": "test"})

    def test_to_dict_full(self):
        gm = GenerationMethod(
            method_type="distillation",
            teacher_model="GPT-4",
            prompt_template_available=True,
            platform="OpenAI",
            details={"batch_size": 32},
        )
        d = gm.to_dict()
        self.assertEqual(d["type"], "distillation")
        self.assertEqual(d["teacher_model"], "GPT-4")
        self.assertEqual(d["prompt_template"], "available")
        self.assertEqual(d["platform"], "OpenAI")
        self.assertEqual(d["batch_size"], 32)


# =============================================================================
# SkillRequirement Tests
# =============================================================================


class TestSkillRequirement(unittest.TestCase):
    """Test SkillRequirement dataclass."""

    def test_defaults(self):
        sr = SkillRequirement(skill_type="programming", name="Python", level="advanced")
        self.assertTrue(sr.required)
        self.assertIsNone(sr.details)

    def test_to_dict(self):
        sr = SkillRequirement(
            skill_type="language",
            name="English",
            level="native",
            required=False,
            details="IELTS 8.0+",
        )
        d = sr.to_dict()
        self.assertEqual(d["type"], "language")
        self.assertEqual(d["name"], "English")
        self.assertEqual(d["level"], "native")
        self.assertFalse(d["required"])
        self.assertEqual(d["details"], "IELTS 8.0+")

    def test_to_dict_details_none(self):
        sr = SkillRequirement(skill_type="tool", name="Excel", level="basic")
        d = sr.to_dict()
        self.assertIsNone(d["details"])


# =============================================================================
# AnnotatorProfile Tests
# =============================================================================


class TestAnnotatorProfile(unittest.TestCase):
    """Test AnnotatorProfile dataclass."""

    def test_defaults(self):
        ap = AnnotatorProfile()
        self.assertEqual(ap.skill_requirements, [])
        self.assertEqual(ap.experience_level, ExperienceLevel.MID)
        self.assertEqual(ap.min_experience_years, 1)
        self.assertEqual(ap.language_requirements, [])
        self.assertEqual(ap.domain_knowledge, [])
        self.assertEqual(ap.education_level, EducationLevel.BACHELOR)
        self.assertEqual(ap.team_size, 10)
        self.assertEqual(ap.team_structure, {"annotator": 8, "reviewer": 2})
        self.assertEqual(ap.estimated_person_days, 0.0)
        self.assertEqual(ap.estimated_hours_per_example, 0.0)
        self.assertEqual(ap.hourly_rate_range, {"min": 15, "max": 45, "currency": "USD"})
        self.assertEqual(ap.screening_criteria, [])
        self.assertEqual(ap.recommended_platforms, [])

    def test_to_dict_defaults(self):
        ap = AnnotatorProfile()
        d = ap.to_dict()
        self.assertEqual(d["skill_requirements"], [])
        self.assertEqual(d["experience"]["level"], "mid")
        self.assertEqual(d["experience"]["min_years"], 1)
        self.assertEqual(d["language_requirements"], [])
        self.assertEqual(d["domain_knowledge"], [])
        self.assertEqual(d["education_level"], "bachelor")
        self.assertEqual(d["team"]["size"], 10)
        self.assertEqual(d["team"]["structure"], {"annotator": 8, "reviewer": 2})
        self.assertEqual(d["workload"]["estimated_person_days"], 0.0)
        self.assertEqual(d["workload"]["hours_per_example"], 0.0)
        self.assertEqual(d["hourly_rate_range"]["currency"], "USD")
        self.assertEqual(d["screening_criteria"], [])
        self.assertEqual(d["recommended_platforms"], [])

    def test_to_dict_with_skills(self):
        skills = [
            SkillRequirement(skill_type="programming", name="Python", level="advanced"),
            SkillRequirement(skill_type="domain", name="NLP", level="intermediate", required=False),
        ]
        ap = AnnotatorProfile(skill_requirements=skills)
        d = ap.to_dict()
        self.assertEqual(len(d["skill_requirements"]), 2)
        self.assertEqual(d["skill_requirements"][0]["name"], "Python")
        self.assertEqual(d["skill_requirements"][1]["name"], "NLP")
        self.assertFalse(d["skill_requirements"][1]["required"])

    def test_to_dict_with_custom_values(self):
        ap = AnnotatorProfile(
            experience_level=ExperienceLevel.EXPERT,
            min_experience_years=5,
            language_requirements=["zh-CN:native", "en:C1"],
            domain_knowledge=["medical", "legal"],
            education_level=EducationLevel.PHD,
            team_size=20,
            team_structure={"annotator": 15, "reviewer": 3, "lead": 2},
            estimated_person_days=100.0,
            estimated_hours_per_example=0.5,
            hourly_rate_range={"min": 30, "max": 80, "currency": "EUR"},
            screening_criteria=["Pass domain test", "3+ years experience"],
            recommended_platforms=["Scale AI", "Surge AI"],
        )
        d = ap.to_dict()
        self.assertEqual(d["experience"]["level"], "expert")
        self.assertEqual(d["experience"]["min_years"], 5)
        self.assertEqual(d["language_requirements"], ["zh-CN:native", "en:C1"])
        self.assertEqual(d["domain_knowledge"], ["medical", "legal"])
        self.assertEqual(d["education_level"], "phd")
        self.assertEqual(d["team"]["size"], 20)
        self.assertEqual(d["team"]["structure"]["lead"], 2)
        self.assertEqual(d["workload"]["estimated_person_days"], 100.0)
        self.assertEqual(d["workload"]["hours_per_example"], 0.5)
        self.assertEqual(d["hourly_rate_range"]["currency"], "EUR")
        self.assertEqual(len(d["screening_criteria"]), 2)
        self.assertEqual(len(d["recommended_platforms"]), 2)

    def test_to_yaml(self):
        ap = AnnotatorProfile()
        yaml_str = ap.to_yaml()
        self.assertIsInstance(yaml_str, str)
        self.assertIn("experience", yaml_str)
        self.assertIn("education_level", yaml_str)

    def test_to_json(self):
        ap = AnnotatorProfile(
            language_requirements=["zh-CN:native"],
            domain_knowledge=["medical"],
        )
        json_str = ap.to_json()
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["language_requirements"], ["zh-CN:native"])
        self.assertEqual(parsed["domain_knowledge"], ["medical"])

    def test_to_json_unicode(self):
        """Ensure ensure_ascii=False works for CJK characters."""
        ap = AnnotatorProfile(domain_knowledge=["医疗", "金融"])
        json_str = ap.to_json()
        # Should contain actual CJK characters, not escaped
        self.assertIn("医疗", json_str)
        self.assertIn("金融", json_str)

    def test_default_factory_independence(self):
        """Ensure default factory lists are independent across instances."""
        ap1 = AnnotatorProfile()
        ap2 = AnnotatorProfile()
        ap1.skill_requirements.append(
            SkillRequirement(skill_type="test", name="test", level="basic")
        )
        self.assertEqual(len(ap2.skill_requirements), 0)

    def test_default_factory_dict_independence(self):
        """Ensure default factory dicts are independent across instances."""
        ap1 = AnnotatorProfile()
        ap2 = AnnotatorProfile()
        ap1.team_structure["new_role"] = 5
        self.assertNotIn("new_role", ap2.team_structure)


# =============================================================================
# QualityRule Tests
# =============================================================================


class TestQualityRule(unittest.TestCase):
    """Test QualityRule dataclass."""

    def test_defaults(self):
        qr = QualityRule(
            rule_id="R1",
            name="Format check",
            description="Check JSON format",
            check_type="format",
            severity="error",
        )
        self.assertTrue(qr.auto_check)
        self.assertIsNone(qr.check_code)

    def test_to_dict(self):
        qr = QualityRule(
            rule_id="R2",
            name="Length check",
            description="Response must be 100+ chars",
            check_type="content",
            severity="warning",
            auto_check=False,
            check_code="len(text) >= 100",
        )
        d = qr.to_dict()
        self.assertEqual(d["id"], "R2")
        self.assertEqual(d["name"], "Length check")
        self.assertEqual(d["description"], "Response must be 100+ chars")
        self.assertEqual(d["type"], "content")
        self.assertEqual(d["severity"], "warning")
        self.assertFalse(d["auto_check"])
        # check_code is NOT included in to_dict
        self.assertNotIn("check_code", d)


# =============================================================================
# AcceptanceCriterion Tests
# =============================================================================


class TestAcceptanceCriterion(unittest.TestCase):
    """Test AcceptanceCriterion dataclass."""

    def test_defaults(self):
        ac = AcceptanceCriterion(
            criterion_id="AC1",
            name="Accuracy",
            description="Must achieve 95% accuracy",
            threshold=0.95,
            metric_type="accuracy",
        )
        self.assertEqual(ac.priority, "required")

    def test_to_dict(self):
        ac = AcceptanceCriterion(
            criterion_id="AC2",
            name="Agreement",
            description="Inter-annotator agreement",
            threshold=0.8,
            metric_type="agreement",
            priority="recommended",
        )
        d = ac.to_dict()
        self.assertEqual(d["id"], "AC2")
        self.assertEqual(d["name"], "Agreement")
        self.assertEqual(d["description"], "Inter-annotator agreement")
        self.assertEqual(d["threshold"], 0.8)
        self.assertEqual(d["type"], "agreement")
        self.assertEqual(d["priority"], "recommended")


# =============================================================================
# Milestone Tests
# =============================================================================


class TestMilestone(unittest.TestCase):
    """Test Milestone dataclass (schema version, not generator version)."""

    def test_defaults(self):
        m = Milestone(
            name="Phase 1",
            description="Setup phase",
            deliverables=["Doc v1"],
            estimated_days=10,
        )
        self.assertEqual(m.dependencies, [])

    def test_to_dict(self):
        m = Milestone(
            name="Phase 2",
            description="Production phase",
            deliverables=["Dataset v1", "Quality report"],
            estimated_days=20,
            dependencies=["Phase 1"],
        )
        d = m.to_dict()
        self.assertEqual(d["name"], "Phase 2")
        self.assertEqual(d["description"], "Production phase")
        self.assertEqual(d["deliverables"], ["Dataset v1", "Quality report"])
        self.assertEqual(d["estimated_days"], 20)
        self.assertEqual(d["dependencies"], ["Phase 1"])

    def test_to_dict_no_dependencies(self):
        m = Milestone(
            name="Phase 1",
            description="Init",
            deliverables=["Setup"],
            estimated_days=5,
        )
        d = m.to_dict()
        self.assertEqual(d["dependencies"], [])


# =============================================================================
# ProductionConfig Tests
# =============================================================================


class TestProductionConfig(unittest.TestCase):
    """Test ProductionConfig dataclass."""

    def test_defaults(self):
        pc = ProductionConfig()
        self.assertEqual(pc.annotation_guide, "")
        self.assertIsNone(pc.annotation_guide_url)
        self.assertEqual(pc.quality_rules, [])
        self.assertEqual(pc.acceptance_criteria, [])
        self.assertEqual(pc.review_workflow, ReviewWorkflow.DOUBLE)
        self.assertEqual(pc.review_sample_rate, 0.1)
        self.assertEqual(pc.estimated_timeline_days, 30)
        self.assertEqual(pc.milestones, [])
        self.assertEqual(pc.labeling_tool_config, {})

    def test_to_dict_defaults(self):
        pc = ProductionConfig()
        d = pc.to_dict()
        self.assertEqual(d["annotation_guide"], "")
        self.assertIsNone(d["annotation_guide_url"])
        self.assertEqual(d["quality_rules"], [])
        self.assertEqual(d["acceptance_criteria"], [])
        self.assertEqual(d["review"]["workflow"], "double")
        self.assertEqual(d["review"]["sample_rate"], 0.1)
        self.assertEqual(d["timeline"]["estimated_days"], 30)
        self.assertEqual(d["timeline"]["milestones"], [])

    def test_to_dict_annotation_guide_truncation(self):
        """Annotation guide longer than 500 chars should be truncated."""
        long_guide = "A" * 600
        pc = ProductionConfig(annotation_guide=long_guide)
        d = pc.to_dict()
        # Should be first 500 chars + "..."
        self.assertEqual(len(d["annotation_guide"]), 503)
        self.assertTrue(d["annotation_guide"].endswith("..."))
        self.assertTrue(d["annotation_guide"].startswith("A" * 500))

    def test_to_dict_annotation_guide_exact_500(self):
        """Annotation guide of exactly 500 chars should NOT be truncated."""
        guide = "B" * 500
        pc = ProductionConfig(annotation_guide=guide)
        d = pc.to_dict()
        self.assertEqual(d["annotation_guide"], guide)
        self.assertFalse(d["annotation_guide"].endswith("..."))

    def test_to_dict_annotation_guide_501(self):
        """Annotation guide of 501 chars should be truncated."""
        guide = "C" * 501
        pc = ProductionConfig(annotation_guide=guide)
        d = pc.to_dict()
        self.assertEqual(len(d["annotation_guide"]), 503)
        self.assertTrue(d["annotation_guide"].endswith("..."))

    def test_to_dict_with_quality_rules(self):
        rules = [
            QualityRule(
                rule_id="R1",
                name="Format",
                description="Check format",
                check_type="format",
                severity="error",
            ),
        ]
        pc = ProductionConfig(quality_rules=rules)
        d = pc.to_dict()
        self.assertEqual(len(d["quality_rules"]), 1)
        self.assertEqual(d["quality_rules"][0]["id"], "R1")

    def test_to_dict_with_acceptance_criteria(self):
        criteria = [
            AcceptanceCriterion(
                criterion_id="AC1",
                name="Accuracy",
                description="High accuracy",
                threshold=0.95,
                metric_type="accuracy",
            ),
        ]
        pc = ProductionConfig(acceptance_criteria=criteria)
        d = pc.to_dict()
        self.assertEqual(len(d["acceptance_criteria"]), 1)
        self.assertEqual(d["acceptance_criteria"][0]["id"], "AC1")

    def test_to_dict_with_milestones(self):
        milestones = [
            Milestone(
                name="Setup",
                description="Initial setup",
                deliverables=["Environment ready"],
                estimated_days=5,
            ),
            Milestone(
                name="Production",
                description="Main production",
                deliverables=["Dataset v1"],
                estimated_days=20,
                dependencies=["Setup"],
            ),
        ]
        pc = ProductionConfig(milestones=milestones)
        d = pc.to_dict()
        self.assertEqual(len(d["timeline"]["milestones"]), 2)
        self.assertEqual(d["timeline"]["milestones"][0]["name"], "Setup")
        self.assertEqual(d["timeline"]["milestones"][1]["dependencies"], ["Setup"])

    def test_to_dict_review_workflows(self):
        for workflow in ReviewWorkflow:
            pc = ProductionConfig(review_workflow=workflow)
            d = pc.to_dict()
            self.assertEqual(d["review"]["workflow"], workflow.value)


# =============================================================================
# EnhancedCost Tests
# =============================================================================


class TestEnhancedCost(unittest.TestCase):
    """Test EnhancedCost dataclass."""

    def test_defaults(self):
        ec = EnhancedCost()
        self.assertEqual(ec.api_cost, 0.0)
        self.assertEqual(ec.compute_cost, 0.0)
        self.assertEqual(ec.human_cost, 0.0)
        self.assertEqual(ec.human_cost_breakdown, {
            "annotation": 0.0,
            "review": 0.0,
            "expert_consultation": 0.0,
            "project_management": 0.0,
        })
        self.assertEqual(ec.region, "us")
        self.assertEqual(ec.region_multiplier, 1.0)
        self.assertEqual(ec.total_cost, 0.0)
        self.assertEqual(ec.total_range, {"low": 0.0, "high": 0.0})
        self.assertEqual(ec.confidence, "medium")
        self.assertEqual(ec.assumptions, [])
        self.assertIsNone(ec.estimated_dataset_value)
        self.assertIsNone(ec.roi_ratio)

    def test_to_dict(self):
        ec = EnhancedCost(
            api_cost=500.0,
            compute_cost=200.0,
            human_cost=3000.0,
            human_cost_breakdown={
                "annotation": 2000.0,
                "review": 500.0,
                "expert_consultation": 300.0,
                "project_management": 200.0,
            },
            region="cn",
            region_multiplier=0.5,
            total_cost=3700.0,
            total_range={"low": 3000.0, "high": 4500.0},
            confidence="high",
            assumptions=["8-hour workday", "10 annotators"],
        )
        d = ec.to_dict()
        self.assertEqual(d["api_cost"], 500.0)
        self.assertEqual(d["compute_cost"], 200.0)
        self.assertEqual(d["human_cost"], 3000.0)
        self.assertEqual(d["human_cost_breakdown"]["annotation"], 2000.0)
        self.assertEqual(d["region"], "cn")
        self.assertEqual(d["region_multiplier"], 0.5)
        self.assertEqual(d["total_cost"], 3700.0)
        self.assertEqual(d["total_range"]["low"], 3000.0)
        self.assertEqual(d["total_range"]["high"], 4500.0)
        self.assertEqual(d["confidence"], "high")
        self.assertEqual(d["assumptions"], ["8-hour workday", "10 annotators"])

    def test_to_dict_excludes_roi_fields(self):
        """ROI fields (estimated_dataset_value, roi_ratio) are NOT in to_dict."""
        ec = EnhancedCost(estimated_dataset_value=10000.0, roi_ratio=2.7)
        d = ec.to_dict()
        self.assertNotIn("estimated_dataset_value", d)
        self.assertNotIn("roi_ratio", d)

    def test_default_factory_independence(self):
        ec1 = EnhancedCost()
        ec2 = EnhancedCost()
        ec1.assumptions.append("test")
        self.assertEqual(len(ec2.assumptions), 0)

    def test_default_factory_dict_independence(self):
        ec1 = EnhancedCost()
        ec2 = EnhancedCost()
        ec1.human_cost_breakdown["new_item"] = 100.0
        self.assertNotIn("new_item", ec2.human_cost_breakdown)


# =============================================================================
# Recipe Tests
# =============================================================================


class TestRecipe(unittest.TestCase):
    """Test Recipe dataclass."""

    def test_defaults(self):
        r = Recipe(name="test-dataset")
        self.assertEqual(r.name, "test-dataset")
        self.assertIsNone(r.version)
        self.assertEqual(r.source_type, SourceType.UNKNOWN)
        self.assertIsNone(r.source_id)
        self.assertIsNone(r.size)
        self.assertIsNone(r.num_examples)
        self.assertEqual(r.languages, [])
        self.assertIsNone(r.license)
        self.assertIsNone(r.description)
        self.assertEqual(r.generation_type, GenerationType.UNKNOWN)
        self.assertIsNone(r.synthetic_ratio)
        self.assertIsNone(r.human_ratio)
        self.assertEqual(r.generation_methods, [])
        self.assertEqual(r.teacher_models, [])
        self.assertIsNone(r.cost)
        self.assertIsNone(r.reproducibility)
        self.assertEqual(r.tags, [])
        self.assertIsNone(r.created_date)
        self.assertEqual(r.authors, [])
        self.assertIsNone(r.paper_url)
        self.assertIsNone(r.homepage_url)
        self.assertIsNone(r.quality_metrics)

    def test_to_dict_minimal(self):
        r = Recipe(name="test-dataset")
        d = r.to_dict()
        self.assertEqual(d["name"], "test-dataset")
        self.assertEqual(d["source"]["type"], "unknown")
        self.assertIsNone(d["source"]["id"])
        # No version, no generation, no cost, no reproducibility, no metadata
        self.assertNotIn("version", d)
        self.assertNotIn("generation", d)
        self.assertNotIn("cost", d)
        self.assertNotIn("reproducibility", d)
        self.assertNotIn("metadata", d)

    def test_to_dict_with_version(self):
        r = Recipe(name="test", version="1.0.0")
        d = r.to_dict()
        self.assertEqual(d["version"], "1.0.0")

    def test_to_dict_version_none_not_included(self):
        r = Recipe(name="test")
        d = r.to_dict()
        self.assertNotIn("version", d)

    def test_to_dict_with_generation(self):
        methods = [
            GenerationMethod(method_type="distillation", teacher_model="GPT-4"),
        ]
        r = Recipe(
            name="test",
            synthetic_ratio=0.7,
            human_ratio=0.3,
            generation_methods=methods,
            teacher_models=["GPT-4", "Claude"],
        )
        d = r.to_dict()
        self.assertIn("generation", d)
        self.assertEqual(d["generation"]["synthetic_ratio"], 0.7)
        self.assertEqual(d["generation"]["human_ratio"], 0.3)
        self.assertEqual(len(d["generation"]["methods"]), 1)
        self.assertEqual(d["generation"]["teacher_models"], ["GPT-4", "Claude"])

    def test_to_dict_generation_empty_not_included(self):
        """No generation fields set means no 'generation' key."""
        r = Recipe(name="test")
        d = r.to_dict()
        self.assertNotIn("generation", d)

    def test_to_dict_generation_partial(self):
        """Only synthetic_ratio set, other generation fields unset."""
        r = Recipe(name="test", synthetic_ratio=1.0)
        d = r.to_dict()
        self.assertIn("generation", d)
        self.assertEqual(d["generation"]["synthetic_ratio"], 1.0)
        self.assertNotIn("human_ratio", d["generation"])
        self.assertNotIn("methods", d["generation"])
        self.assertNotIn("teacher_models", d["generation"])

    def test_to_dict_with_cost(self):
        cost = Cost(estimated_total_usd=1000.0, confidence="high")
        r = Recipe(name="test", cost=cost)
        d = r.to_dict()
        self.assertIn("cost", d)
        self.assertEqual(d["cost"]["estimated_total_usd"], 1000.0)

    def test_to_dict_with_reproducibility(self):
        repro = Reproducibility(score=7, available=["code"])
        r = Recipe(name="test", reproducibility=repro)
        d = r.to_dict()
        self.assertIn("reproducibility", d)
        self.assertEqual(d["reproducibility"]["score"], 7)

    def test_to_dict_with_metadata(self):
        r = Recipe(
            name="test",
            size=1000000,
            num_examples=5000,
            languages=["en", "zh"],
            license="MIT",
            tags=["nlp", "chat"],
            authors=["Alice", "Bob"],
            paper_url="https://arxiv.org/abs/1234",
        )
        d = r.to_dict()
        self.assertIn("metadata", d)
        self.assertEqual(d["metadata"]["size_bytes"], 1000000)
        self.assertEqual(d["metadata"]["num_examples"], 5000)
        self.assertEqual(d["metadata"]["languages"], ["en", "zh"])
        self.assertEqual(d["metadata"]["license"], "MIT")
        self.assertEqual(d["metadata"]["tags"], ["nlp", "chat"])
        self.assertEqual(d["metadata"]["authors"], ["Alice", "Bob"])
        self.assertEqual(d["metadata"]["paper_url"], "https://arxiv.org/abs/1234")

    def test_to_dict_metadata_empty_not_included(self):
        r = Recipe(name="test")
        d = r.to_dict()
        self.assertNotIn("metadata", d)

    def test_to_dict_metadata_partial(self):
        """Only size set, other metadata fields unset."""
        r = Recipe(name="test", size=100)
        d = r.to_dict()
        self.assertIn("metadata", d)
        self.assertEqual(d["metadata"]["size_bytes"], 100)
        self.assertNotIn("num_examples", d["metadata"])
        self.assertNotIn("languages", d["metadata"])

    def test_to_dict_homepage_url_not_in_metadata(self):
        """homepage_url is NOT included in metadata (not checked in to_dict)."""
        r = Recipe(name="test", homepage_url="https://example.com")
        d = r.to_dict()
        # homepage_url is not added to metadata in the current implementation
        if "metadata" in d:
            self.assertNotIn("homepage_url", d["metadata"])

    def test_to_yaml(self):
        r = Recipe(name="test-yaml", version="1.0", source_type=SourceType.HUGGINGFACE)
        yaml_str = r.to_yaml()
        self.assertIsInstance(yaml_str, str)
        self.assertIn("test-yaml", yaml_str)
        self.assertIn("huggingface", yaml_str)

    def test_to_yaml_unicode(self):
        r = Recipe(name="test", languages=["zh", "en"], tags=["中文数据"])
        yaml_str = r.to_yaml()
        # allow_unicode=True should keep CJK characters
        self.assertIn("中文数据", yaml_str)

    def test_source_types(self):
        """Test that all source types work in to_dict."""
        for st in SourceType:
            r = Recipe(name="test", source_type=st)
            d = r.to_dict()
            self.assertEqual(d["source"]["type"], st.value)

    def test_to_dict_full(self):
        """Full recipe with all fields populated."""
        cost = Cost(estimated_total_usd=5000.0)
        repro = Reproducibility(score=8, available=["code", "data"])
        methods = [GenerationMethod(method_type="distillation", teacher_model="GPT-4")]
        r = Recipe(
            name="full-recipe",
            version="2.0",
            source_type=SourceType.HUGGINGFACE,
            source_id="org/dataset",
            size=500000,
            num_examples=10000,
            languages=["en"],
            license="Apache-2.0",
            description="A test dataset",
            generation_type=GenerationType.MIXED,
            synthetic_ratio=0.6,
            human_ratio=0.4,
            generation_methods=methods,
            teacher_models=["GPT-4"],
            cost=cost,
            reproducibility=repro,
            tags=["test"],
            created_date="2025-01-01",
            authors=["Test Author"],
            paper_url="https://arxiv.org/abs/0001",
            quality_metrics={"accuracy": 0.95},
        )
        d = r.to_dict()
        self.assertEqual(d["name"], "full-recipe")
        self.assertEqual(d["version"], "2.0")
        self.assertEqual(d["source"]["type"], "huggingface")
        self.assertEqual(d["source"]["id"], "org/dataset")
        self.assertIn("generation", d)
        self.assertIn("cost", d)
        self.assertIn("reproducibility", d)
        self.assertIn("metadata", d)


# =============================================================================
# DataRecipe Tests
# =============================================================================


class TestDataRecipe(unittest.TestCase):
    """Test DataRecipe class (extends Recipe)."""

    def test_inherits_recipe(self):
        self.assertTrue(issubclass(DataRecipe, Recipe))

    def test_defaults(self):
        dr = DataRecipe(name="test")
        self.assertIsNone(dr.annotator_profile)
        self.assertIsNone(dr.production_config)
        self.assertIsNone(dr.enhanced_cost)
        # Inherited defaults
        self.assertEqual(dr.source_type, SourceType.UNKNOWN)
        self.assertEqual(dr.generation_type, GenerationType.UNKNOWN)

    def test_to_dict_minimal(self):
        """DataRecipe with no V2 fields should produce same output as Recipe."""
        dr = DataRecipe(name="test")
        d = dr.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertNotIn("enhanced_cost", d)
        self.assertNotIn("annotator_profile", d)
        self.assertNotIn("production_config", d)

    def test_to_dict_with_enhanced_cost(self):
        ec = EnhancedCost(api_cost=100.0, total_cost=500.0)
        dr = DataRecipe(name="test", enhanced_cost=ec)
        d = dr.to_dict()
        self.assertIn("enhanced_cost", d)
        self.assertEqual(d["enhanced_cost"]["api_cost"], 100.0)
        self.assertEqual(d["enhanced_cost"]["total_cost"], 500.0)

    def test_to_dict_with_annotator_profile(self):
        ap = AnnotatorProfile(
            experience_level=ExperienceLevel.SENIOR,
            team_size=5,
        )
        dr = DataRecipe(name="test", annotator_profile=ap)
        d = dr.to_dict()
        self.assertIn("annotator_profile", d)
        self.assertEqual(d["annotator_profile"]["experience"]["level"], "senior")
        self.assertEqual(d["annotator_profile"]["team"]["size"], 5)

    def test_to_dict_with_production_config(self):
        pc = ProductionConfig(
            annotation_guide="Test guide",
            review_workflow=ReviewWorkflow.EXPERT,
        )
        dr = DataRecipe(name="test", production_config=pc)
        d = dr.to_dict()
        self.assertIn("production_config", d)
        self.assertEqual(d["production_config"]["annotation_guide"], "Test guide")
        self.assertEqual(d["production_config"]["review"]["workflow"], "expert")

    def test_to_dict_inherits_recipe_fields(self):
        """DataRecipe to_dict includes Recipe base fields."""
        dr = DataRecipe(
            name="test",
            version="1.0",
            source_type=SourceType.GITHUB,
            source_id="org/repo",
            size=1000,
            num_examples=100,
            languages=["en"],
            license="MIT",
            synthetic_ratio=0.5,
            tags=["test"],
            authors=["Author"],
        )
        d = dr.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["version"], "1.0")
        self.assertEqual(d["source"]["type"], "github")
        self.assertIn("generation", d)
        self.assertIn("metadata", d)

    def test_to_dict_full_v2(self):
        """DataRecipe with all V2 fields populated."""
        dr = DataRecipe(
            name="full-v2",
            enhanced_cost=EnhancedCost(total_cost=10000.0),
            annotator_profile=AnnotatorProfile(team_size=20),
            production_config=ProductionConfig(estimated_timeline_days=60),
        )
        d = dr.to_dict()
        self.assertIn("enhanced_cost", d)
        self.assertIn("annotator_profile", d)
        self.assertIn("production_config", d)
        self.assertEqual(d["enhanced_cost"]["total_cost"], 10000.0)
        self.assertEqual(d["annotator_profile"]["team"]["size"], 20)
        self.assertEqual(d["production_config"]["timeline"]["estimated_days"], 60)


class TestDataRecipeFromRecipe(unittest.TestCase):
    """Test DataRecipe.from_recipe() classmethod."""

    def test_basic_conversion(self):
        recipe = Recipe(name="original", version="1.0")
        dr = DataRecipe.from_recipe(recipe)
        self.assertIsInstance(dr, DataRecipe)
        self.assertEqual(dr.name, "original")
        self.assertEqual(dr.version, "1.0")

    def test_all_fields_copied(self):
        cost = Cost(estimated_total_usd=1000.0)
        repro = Reproducibility(score=7)
        methods = [GenerationMethod(method_type="distillation")]
        recipe = Recipe(
            name="full",
            version="2.0",
            source_type=SourceType.HUGGINGFACE,
            source_id="org/dataset",
            size=500000,
            num_examples=10000,
            languages=["en", "zh"],
            license="Apache-2.0",
            description="Test description",
            generation_type=GenerationType.MIXED,
            synthetic_ratio=0.6,
            human_ratio=0.4,
            generation_methods=methods,
            teacher_models=["GPT-4"],
            cost=cost,
            reproducibility=repro,
            quality_metrics={"accuracy": 0.95},
            tags=["test", "nlp"],
            created_date="2025-01-01",
            authors=["Alice"],
            paper_url="https://arxiv.org/abs/1234",
            homepage_url="https://example.com",
        )
        dr = DataRecipe.from_recipe(recipe)

        self.assertEqual(dr.name, "full")
        self.assertEqual(dr.version, "2.0")
        self.assertEqual(dr.source_type, SourceType.HUGGINGFACE)
        self.assertEqual(dr.source_id, "org/dataset")
        self.assertEqual(dr.size, 500000)
        self.assertEqual(dr.num_examples, 10000)
        self.assertEqual(dr.languages, ["en", "zh"])
        self.assertEqual(dr.license, "Apache-2.0")
        self.assertEqual(dr.description, "Test description")
        self.assertEqual(dr.generation_type, GenerationType.MIXED)
        self.assertEqual(dr.synthetic_ratio, 0.6)
        self.assertEqual(dr.human_ratio, 0.4)
        self.assertEqual(len(dr.generation_methods), 1)
        self.assertEqual(dr.teacher_models, ["GPT-4"])
        self.assertIs(dr.cost, cost)
        self.assertIs(dr.reproducibility, repro)
        self.assertEqual(dr.quality_metrics, {"accuracy": 0.95})
        self.assertEqual(dr.tags, ["test", "nlp"])
        self.assertEqual(dr.created_date, "2025-01-01")
        self.assertEqual(dr.authors, ["Alice"])
        self.assertEqual(dr.paper_url, "https://arxiv.org/abs/1234")
        self.assertEqual(dr.homepage_url, "https://example.com")

    def test_v2_fields_default_none(self):
        recipe = Recipe(name="test")
        dr = DataRecipe.from_recipe(recipe)
        self.assertIsNone(dr.annotator_profile)
        self.assertIsNone(dr.production_config)
        self.assertIsNone(dr.enhanced_cost)

    def test_lists_are_copied(self):
        """Ensure lists are copies, not references to the original."""
        recipe = Recipe(
            name="test",
            languages=["en"],
            generation_methods=[GenerationMethod(method_type="test")],
            teacher_models=["model1"],
            tags=["tag1"],
            authors=["author1"],
        )
        dr = DataRecipe.from_recipe(recipe)

        # Mutate the DataRecipe lists
        dr.languages.append("zh")
        dr.generation_methods.append(GenerationMethod(method_type="test2"))
        dr.teacher_models.append("model2")
        dr.tags.append("tag2")
        dr.authors.append("author2")

        # Original should be unchanged
        self.assertEqual(recipe.languages, ["en"])
        self.assertEqual(len(recipe.generation_methods), 1)
        self.assertEqual(recipe.teacher_models, ["model1"])
        self.assertEqual(recipe.tags, ["tag1"])
        self.assertEqual(recipe.authors, ["author1"])

    def test_from_recipe_with_empty_lists(self):
        """Recipe with empty lists should produce empty lists in DataRecipe."""
        recipe = Recipe(name="empty")
        dr = DataRecipe.from_recipe(recipe)
        self.assertEqual(dr.languages, [])
        self.assertEqual(dr.generation_methods, [])
        self.assertEqual(dr.teacher_models, [])
        self.assertEqual(dr.tags, [])
        self.assertEqual(dr.authors, [])

    def test_from_recipe_preserves_none_fields(self):
        recipe = Recipe(name="minimal")
        dr = DataRecipe.from_recipe(recipe)
        self.assertIsNone(dr.version)
        self.assertIsNone(dr.source_id)
        self.assertIsNone(dr.size)
        self.assertIsNone(dr.num_examples)
        self.assertIsNone(dr.license)
        self.assertIsNone(dr.description)
        self.assertIsNone(dr.synthetic_ratio)
        self.assertIsNone(dr.human_ratio)
        self.assertIsNone(dr.cost)
        self.assertIsNone(dr.reproducibility)
        self.assertIsNone(dr.quality_metrics)
        self.assertIsNone(dr.created_date)
        self.assertIsNone(dr.paper_url)
        self.assertIsNone(dr.homepage_url)


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult dataclass."""

    def test_valid(self):
        vr = ValidationResult(valid=True)
        self.assertTrue(vr.valid)
        self.assertEqual(vr.errors, [])
        self.assertEqual(vr.warnings, [])

    def test_invalid_with_errors(self):
        vr = ValidationResult(
            valid=False,
            errors=["Missing field", "Invalid format"],
            warnings=["Deprecated field"],
        )
        self.assertFalse(vr.valid)
        self.assertEqual(len(vr.errors), 2)
        self.assertEqual(len(vr.warnings), 1)


# =============================================================================
# AnnotatorMatch Tests
# =============================================================================


class TestAnnotatorMatch(unittest.TestCase):
    """Test AnnotatorMatch dataclass."""

    def test_defaults(self):
        am = AnnotatorMatch(
            annotator_id="ann-001",
            name="Alice",
            match_score=0.85,
        )
        self.assertEqual(am.annotator_id, "ann-001")
        self.assertEqual(am.name, "Alice")
        self.assertEqual(am.match_score, 0.85)
        self.assertEqual(am.skills_matched, [])
        self.assertEqual(am.skills_missing, [])
        self.assertEqual(am.hourly_rate, 0.0)
        self.assertEqual(am.availability, "unknown")

    def test_full(self):
        am = AnnotatorMatch(
            annotator_id="ann-002",
            name="Bob",
            match_score=0.95,
            skills_matched=["Python", "NLP"],
            skills_missing=["Medical"],
            hourly_rate=35.0,
            availability="available",
        )
        self.assertEqual(am.skills_matched, ["Python", "NLP"])
        self.assertEqual(am.skills_missing, ["Medical"])
        self.assertEqual(am.hourly_rate, 35.0)
        self.assertEqual(am.availability, "available")


# =============================================================================
# ProjectHandle Tests
# =============================================================================


class TestProjectHandle(unittest.TestCase):
    """Test ProjectHandle dataclass."""

    def test_defaults(self):
        ph = ProjectHandle(
            project_id="proj-001",
            provider="scale",
            created_at="2025-01-01T00:00:00Z",
            status="created",
        )
        self.assertIsNone(ph.url)
        self.assertEqual(ph.metadata, {})

    def test_full(self):
        ph = ProjectHandle(
            project_id="proj-002",
            provider="surge",
            created_at="2025-06-01T12:00:00Z",
            status="active",
            url="https://platform.surge.ai/project/002",
            metadata={"batch_size": 100},
        )
        self.assertEqual(ph.url, "https://platform.surge.ai/project/002")
        self.assertEqual(ph.metadata["batch_size"], 100)


# =============================================================================
# DeploymentResult Tests
# =============================================================================


class TestDeploymentResult(unittest.TestCase):
    """Test DeploymentResult dataclass."""

    def test_success(self):
        ph = ProjectHandle(
            project_id="p1", provider="test", created_at="2025-01-01", status="ok"
        )
        dr = DeploymentResult(success=True, project_handle=ph)
        self.assertTrue(dr.success)
        self.assertIsNotNone(dr.project_handle)
        self.assertIsNone(dr.error)
        self.assertEqual(dr.details, {})

    def test_failure(self):
        dr = DeploymentResult(
            success=False,
            error="Authentication failed",
            details={"status_code": 401},
        )
        self.assertFalse(dr.success)
        self.assertIsNone(dr.project_handle)
        self.assertEqual(dr.error, "Authentication failed")
        self.assertEqual(dr.details["status_code"], 401)


# =============================================================================
# ProjectStatus Tests
# =============================================================================


class TestProjectStatus(unittest.TestCase):
    """Test ProjectStatus dataclass."""

    def test_defaults(self):
        ps = ProjectStatus(status="pending")
        self.assertEqual(ps.status, "pending")
        self.assertEqual(ps.progress, 0.0)
        self.assertEqual(ps.completed_count, 0)
        self.assertEqual(ps.total_count, 0)
        self.assertIsNone(ps.quality_score)
        self.assertIsNone(ps.estimated_completion)

    def test_in_progress(self):
        ps = ProjectStatus(
            status="in_progress",
            progress=45.0,
            completed_count=450,
            total_count=1000,
            quality_score=0.92,
            estimated_completion="2025-03-15",
        )
        self.assertEqual(ps.progress, 45.0)
        self.assertEqual(ps.completed_count, 450)
        self.assertEqual(ps.total_count, 1000)
        self.assertEqual(ps.quality_score, 0.92)
        self.assertEqual(ps.estimated_completion, "2025-03-15")


# =============================================================================
# DeploymentProvider Protocol Tests
# =============================================================================


class TestDeploymentProviderProtocol(unittest.TestCase):
    """Test DeploymentProvider runtime_checkable protocol."""

    def test_protocol_is_runtime_checkable(self):
        """DeploymentProvider is decorated with @runtime_checkable."""

        class ValidProvider:
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "test provider"

            def validate_config(self, config):
                return ValidationResult(valid=True)

            def match_annotators(self, profile, limit=10):
                return []

            def create_project(self, recipe, config=None):
                return ProjectHandle(
                    project_id="p1", provider="test",
                    created_at="2025-01-01", status="created",
                )

            def submit(self, project):
                return DeploymentResult(success=True)

            def get_status(self, project):
                return ProjectStatus(status="pending")

            def cancel(self, project):
                return True

        provider = ValidProvider()
        self.assertIsInstance(provider, DeploymentProvider)

    def test_non_provider_not_instance(self):
        """A class missing protocol methods should not pass isinstance check."""

        class NotAProvider:
            pass

        obj = NotAProvider()
        self.assertNotIsInstance(obj, DeploymentProvider)


# =============================================================================
# Edge Case and Integration Tests
# =============================================================================


class TestRecipeListFieldIndependence(unittest.TestCase):
    """Ensure default factory lists are independent across Recipe instances."""

    def test_languages_independence(self):
        r1 = Recipe(name="r1")
        r2 = Recipe(name="r2")
        r1.languages.append("en")
        self.assertEqual(r2.languages, [])

    def test_tags_independence(self):
        r1 = Recipe(name="r1")
        r2 = Recipe(name="r2")
        r1.tags.append("test")
        self.assertEqual(r2.tags, [])

    def test_authors_independence(self):
        r1 = Recipe(name="r1")
        r2 = Recipe(name="r2")
        r1.authors.append("Alice")
        self.assertEqual(r2.authors, [])

    def test_generation_methods_independence(self):
        r1 = Recipe(name="r1")
        r2 = Recipe(name="r2")
        r1.generation_methods.append(GenerationMethod(method_type="test"))
        self.assertEqual(r2.generation_methods, [])

    def test_teacher_models_independence(self):
        r1 = Recipe(name="r1")
        r2 = Recipe(name="r2")
        r1.teacher_models.append("GPT-4")
        self.assertEqual(r2.teacher_models, [])


class TestDataRecipeInheritance(unittest.TestCase):
    """Test that DataRecipe properly inherits from Recipe."""

    def test_isinstance(self):
        dr = DataRecipe(name="test")
        self.assertIsInstance(dr, Recipe)
        self.assertIsInstance(dr, DataRecipe)

    def test_recipe_methods_available(self):
        dr = DataRecipe(name="test")
        # to_yaml should be inherited from Recipe
        yaml_str = dr.to_yaml()
        self.assertIn("test", yaml_str)

    def test_datarecipe_to_dict_is_superset_of_recipe(self):
        """DataRecipe.to_dict() should contain all Recipe.to_dict() keys plus optional V2 keys."""
        r = Recipe(name="test", version="1.0", size=100)
        dr = DataRecipe(
            name="test", version="1.0", size=100,
            enhanced_cost=EnhancedCost(total_cost=500.0),
        )
        r_dict = r.to_dict()
        dr_dict = dr.to_dict()
        # All Recipe keys should be in DataRecipe dict
        for key in r_dict:
            self.assertIn(key, dr_dict)
        # DataRecipe has extra key
        self.assertIn("enhanced_cost", dr_dict)


class TestCostToDict_ZeroValues(unittest.TestCase):
    """Test Cost.to_dict with zero values (not None)."""

    def test_zero_estimated_total(self):
        cost = Cost(estimated_total_usd=0.0)
        d = cost.to_dict()
        self.assertEqual(d["estimated_total_usd"], 0.0)

    def test_zero_low_estimate(self):
        cost = Cost(low_estimate_usd=0.0)
        d = cost.to_dict()
        # 0.0 is not None, so should be included
        self.assertIn("low_estimate_usd", d)
        self.assertEqual(d["low_estimate_usd"], 0.0)


class TestRecipeSizeZero(unittest.TestCase):
    """Test Recipe.to_dict when size is 0 (falsy but not None)."""

    def test_size_zero_not_in_metadata(self):
        """size=0 is falsy, so it won't appear in metadata."""
        r = Recipe(name="test", size=0)
        d = r.to_dict()
        # In the implementation, `if self.size:` is False for 0
        # This tests the actual behavior
        if "metadata" in d:
            self.assertNotIn("size_bytes", d["metadata"])

    def test_num_examples_zero_not_in_metadata(self):
        """num_examples=0 is falsy, so it won't appear in metadata."""
        r = Recipe(name="test", num_examples=0)
        d = r.to_dict()
        if "metadata" in d:
            self.assertNotIn("num_examples", d["metadata"])


class TestProductionConfigComplexScenario(unittest.TestCase):
    """Test ProductionConfig with multiple nested objects."""

    def test_full_config(self):
        rules = [
            QualityRule(
                rule_id="R1", name="Format", description="JSON format",
                check_type="format", severity="error",
            ),
            QualityRule(
                rule_id="R2", name="Length", description="Min length",
                check_type="content", severity="warning", auto_check=False,
            ),
        ]
        criteria = [
            AcceptanceCriterion(
                criterion_id="AC1", name="Accuracy", description="High accuracy",
                threshold=0.95, metric_type="accuracy", priority="required",
            ),
            AcceptanceCriterion(
                criterion_id="AC2", name="Speed", description="Fast completion",
                threshold=0.8, metric_type="completeness", priority="optional",
            ),
        ]
        milestones = [
            Milestone(
                name="Setup", description="Initial setup",
                deliverables=["Env ready"], estimated_days=5,
            ),
            Milestone(
                name="Production", description="Main production",
                deliverables=["Dataset v1", "Report"],
                estimated_days=20, dependencies=["Setup"],
            ),
        ]
        pc = ProductionConfig(
            annotation_guide="# Guide\nDetailed instructions here.",
            annotation_guide_url="https://docs.example.com/guide",
            quality_rules=rules,
            acceptance_criteria=criteria,
            review_workflow=ReviewWorkflow.EXPERT,
            review_sample_rate=0.2,
            estimated_timeline_days=45,
            milestones=milestones,
            labeling_tool_config={"tool": "label-studio", "version": "1.8"},
        )
        d = pc.to_dict()

        self.assertEqual(len(d["quality_rules"]), 2)
        self.assertEqual(len(d["acceptance_criteria"]), 2)
        self.assertEqual(d["review"]["workflow"], "expert")
        self.assertEqual(d["review"]["sample_rate"], 0.2)
        self.assertEqual(d["timeline"]["estimated_days"], 45)
        self.assertEqual(len(d["timeline"]["milestones"]), 2)
        self.assertEqual(d["timeline"]["milestones"][1]["dependencies"], ["Setup"])
        # annotation_guide_url should be present
        self.assertEqual(d["annotation_guide_url"], "https://docs.example.com/guide")
        # labeling_tool_config is NOT in to_dict output
        self.assertNotIn("labeling_tool_config", d)


if __name__ == "__main__":
    unittest.main()
