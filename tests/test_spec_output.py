"""Comprehensive unit tests for SpecOutputGenerator (spec_output.py).

Covers all inline generator methods, helper functions, and the main
generate() orchestrator. Uses tmp directories so no real files leak.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from datarecipe.analyzers.spec_analyzer import (
    FieldDefinition,
    SpecificationAnalysis,
    ValidationStrategy,
)
from datarecipe.generators.spec_output import SpecOutputGenerator, SpecOutputResult


# --------------- helpers ---------------


def _make_analysis(**overrides) -> SpecificationAnalysis:
    """Create a SpecificationAnalysis with sensible defaults."""
    defaults = dict(
        project_name="TestProject",
        dataset_type="evaluation",
        description="A test dataset for evaluation.",
        task_type="reasoning",
        task_description="Solve complex reasoning problems.",
        cognitive_requirements=["逻辑推理", "空间想象"],
        reasoning_chain=["理解题意", "分析条件", "推导结果"],
        data_requirements=["高质量原创题目"],
        quality_constraints=["答案必须唯一", "步骤完整"],
        forbidden_items=["禁止使用 AI 生成内容"],
        difficulty_criteria="expert-level problems",
        fields=[
            {"name": "question", "type": "string", "required": True, "description": "题目文字"},
            {"name": "answer", "type": "string", "required": True, "description": "标准答案"},
            {"name": "explanation", "type": "text", "required": False, "description": "解析"},
        ],
        field_requirements={"question": "不少于20字"},
        estimated_difficulty="hard",
        estimated_domain="数学",
        estimated_human_percentage=95.0,
        has_images=False,
        image_count=0,
        scoring_rubric=[{"name": "正确性", "weight": 1.0}],
        examples=[],
    )
    defaults.update(overrides)
    return SpecificationAnalysis(**defaults)


def _make_analysis_with_images(**overrides) -> SpecificationAnalysis:
    """Analysis with images enabled."""
    return _make_analysis(has_images=True, image_count=5, **overrides)


def _make_analysis_with_difficulty_validation(**overrides) -> SpecificationAnalysis:
    """Analysis with difficulty validation configured."""
    return _make_analysis(
        difficulty_validation={
            "model": "doubao1.8",
            "settings": "高思考深度",
            "test_count": 3,
            "max_correct": 1,
            "requires_record": True,
        },
        **overrides,
    )


def _make_analysis_minimal(**overrides) -> SpecificationAnalysis:
    """Minimal analysis with no fields, no images, etc."""
    return _make_analysis(
        fields=[],
        cognitive_requirements=[],
        reasoning_chain=[],
        forbidden_items=[],
        quality_constraints=[],
        scoring_rubric=[],
        has_images=False,
        **overrides,
    )


# ==================== SpecOutputResult ====================


class TestSpecOutputResult(unittest.TestCase):
    """Test the SpecOutputResult dataclass."""

    def test_defaults(self):
        r = SpecOutputResult()
        self.assertTrue(r.success)
        self.assertEqual(r.error, "")
        self.assertEqual(r.output_dir, "")
        self.assertEqual(r.files_generated, [])

    def test_custom_values(self):
        r = SpecOutputResult(success=False, error="boom", output_dir="/tmp/x", files_generated=["a.md"])
        self.assertFalse(r.success)
        self.assertEqual(r.error, "boom")


# ==================== _estimate_cost_per_item ====================


class TestEstimateCostPerItem(unittest.TestCase):
    """Test the private cost estimation helper."""

    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_easy_china(self):
        a = _make_analysis(estimated_difficulty="easy", forbidden_items=[], has_images=False, reasoning_chain=["a"])
        cost = self.gen._estimate_cost_per_item(a, "china")
        # easy base=5, china=*0.6, no images, no forbidden, reasoning<=3
        self.assertAlmostEqual(cost, 5 * 0.6)

    def test_hard_us(self):
        a = _make_analysis(estimated_difficulty="hard", forbidden_items=["no AI"], has_images=True, reasoning_chain=["a", "b", "c", "d"])
        cost = self.gen._estimate_cost_per_item(a, "us")
        # hard=20, images*1.5, reasoning>3*1.3, forbidden*1.2, us*1.0
        expected = 20 * 1.5 * 1.3 * 1.2
        self.assertAlmostEqual(cost, expected)

    def test_expert_china_with_everything(self):
        a = _make_analysis(
            estimated_difficulty="expert",
            forbidden_items=["no AI"],
            has_images=True,
            reasoning_chain=["a", "b", "c", "d", "e"],
        )
        cost = self.gen._estimate_cost_per_item(a, "china")
        expected = 40 * 1.5 * 1.3 * 1.2 * 0.6
        self.assertAlmostEqual(cost, expected)

    def test_unknown_difficulty(self):
        a = _make_analysis(estimated_difficulty="unknown", forbidden_items=[], has_images=False, reasoning_chain=[])
        cost = self.gen._estimate_cost_per_item(a, "us")
        # default base=15
        self.assertAlmostEqual(cost, 15)

    def test_medium_no_region(self):
        a = _make_analysis(estimated_difficulty="medium", forbidden_items=[], has_images=False, reasoning_chain=[])
        cost = self.gen._estimate_cost_per_item(a, "eu")
        # No region adjustment for 'eu', multiplier stays 1.0
        self.assertAlmostEqual(cost, 10)


# ==================== _ensure_fields ====================


class TestEnsureFields(unittest.TestCase):
    """Test the _ensure_fields helper."""

    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_existing_fields_not_overwritten(self):
        a = _make_analysis()
        original_fields = list(a.fields)
        self.gen._ensure_fields(a)
        self.assertEqual(a.fields, original_fields)

    def test_empty_fields_populated_from_profile(self):
        a = _make_analysis(fields=[], dataset_type="evaluation")
        self.gen._ensure_fields(a)
        # evaluation profile should have default_fields
        self.assertTrue(len(a.fields) > 0)

    def test_cognitive_requirements_populated(self):
        a = _make_analysis(fields=[], cognitive_requirements=[], dataset_type="evaluation")
        self.gen._ensure_fields(a)
        # profile may provide cognitive_requirements
        # Just ensure no crash - may or may not have values depending on profile

    def test_reasoning_chain_split_from_profile(self):
        a = _make_analysis(
            fields=[], reasoning_chain=[], cognitive_requirements=[], dataset_type="evaluation"
        )
        self.gen._ensure_fields(a)
        # reasoning_chain should be populated as list of stripped strings
        if a.reasoning_chain:
            for item in a.reasoning_chain:
                self.assertEqual(item, item.strip())

    def test_cached_field_definitions_invalidated(self):
        a = _make_analysis(fields=[])
        # Force cache creation
        _ = a.field_definitions
        self.gen._ensure_fields(a)
        # After ensure, cache should be invalidated (None)
        if hasattr(a, "_cached_field_definitions"):
            self.assertIsNone(a._cached_field_definitions)


# ==================== _build_difficulty_explanation ====================


class TestBuildDifficultyExplanation(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_with_all_factors(self):
        a = _make_analysis(has_images=True)
        explanation = self.gen._build_difficulty_explanation(a)
        self.assertIn("认知能力", explanation)
        self.assertIn("推理链", explanation)
        self.assertIn("视觉理解", explanation)
        self.assertIn("hard", explanation)

    def test_with_no_factors(self):
        a = _make_analysis(cognitive_requirements=[], reasoning_chain=[], has_images=False)
        explanation = self.gen._build_difficulty_explanation(a)
        self.assertIn("hard", explanation)
        self.assertIn("综合评估", explanation)

    def test_only_cognitive(self):
        a = _make_analysis(reasoning_chain=[], has_images=False)
        explanation = self.gen._build_difficulty_explanation(a)
        self.assertIn("认知能力", explanation)
        self.assertNotIn("推理链", explanation)


# ==================== _generate_value_from_field ====================


class TestGenerateValueFromField(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_string_template(self):
        fd = FieldDefinition(name="question", type="string", description="题目")
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertIn("请填写", val)
        self.assertIn("题目", val)

    def test_string_sample(self):
        fd = FieldDefinition(name="question", type="string", description="题目")
        val = self.gen._generate_value_from_field(fd, mode="sample")
        self.assertIn("题目", val)

    def test_number_returns_zero(self):
        fd = FieldDefinition(name="score", type="number")
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertEqual(val, 0)

    def test_integer_returns_zero(self):
        fd = FieldDefinition(name="count", type="integer")
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertEqual(val, 0)

    def test_boolean_returns_true(self):
        fd = FieldDefinition(name="flag", type="boolean")
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertTrue(val)

    def test_array_no_items_returns_empty_list(self):
        fd = FieldDefinition(name="tags", type="array")
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertEqual(val, [])

    def test_object_no_properties_returns_empty_dict(self):
        fd = FieldDefinition(name="metadata", type="object")
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertEqual(val, {})

    def test_enum_template_picks_first(self):
        fd = FieldDefinition(name="level", type="string", enum=["easy", "medium", "hard"])
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertEqual(val, "easy")

    def test_enum_sample_varies_by_index(self):
        fd = FieldDefinition(name="level", type="string", enum=["easy", "medium", "hard"])
        val0 = self.gen._generate_value_from_field(fd, mode="sample", context={"sample_index": 0})
        val1 = self.gen._generate_value_from_field(fd, mode="sample", context={"sample_index": 1})
        val2 = self.gen._generate_value_from_field(fd, mode="sample", context={"sample_index": 2})
        self.assertEqual(val0, "easy")
        self.assertEqual(val1, "medium")
        self.assertEqual(val2, "hard")

    def test_object_with_properties_recurse(self):
        fd = FieldDefinition(
            name="answer",
            type="object",
            properties=[
                FieldDefinition(name="value", type="string", description="val"),
                FieldDefinition(name="score", type="number"),
            ],
        )
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertIsInstance(val, dict)
        self.assertIn("value", val)
        self.assertIn("score", val)
        self.assertEqual(val["score"], 0)

    def test_array_with_items_wraps_one(self):
        fd = FieldDefinition(
            name="steps",
            type="array",
            items=FieldDefinition(name="step", type="string", description="步骤"),
        )
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertIsInstance(val, list)
        self.assertEqual(len(val), 1)
        self.assertIn("请填写", val[0])

    def test_unknown_type_fallback(self):
        fd = FieldDefinition(name="custom", type="weird_type", description="something")
        val = self.gen._generate_value_from_field(fd, mode="template")
        # _map_type maps unknown types to "string", so we get a string placeholder
        self.assertIn("请填写", val)


# ==================== _template_placeholder / _sample_placeholder ====================


class TestPlaceholders(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_template_placeholder(self):
        fd = FieldDefinition(name="q", type="string", description="question")
        val = self.gen._template_placeholder(fd)
        self.assertIn("请填写", val)

    def test_sample_placeholder(self):
        fd = FieldDefinition(name="q", type="string", description="question")
        val = self.gen._sample_placeholder(fd, {"sample_index": 0})
        self.assertIn("question", val)


# ==================== _build_schema_from_fields ====================


class TestBuildSchemaFromFields(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_basic_schema(self):
        a = _make_analysis()
        schema = self.gen._build_schema_from_fields(a)
        self.assertEqual(schema["type"], "object")
        self.assertIn("$schema", schema)
        self.assertIn("id", schema["properties"])
        self.assertIn("question", schema["properties"])
        self.assertIn("id", schema["required"])
        self.assertIn("question", schema["required"])
        # explanation is not required
        self.assertNotIn("explanation", schema["required"])
        self.assertIn("metadata", schema["properties"])
        self.assertIn("x-field-definitions", schema)

    def test_id_field_not_duplicated(self):
        a = _make_analysis(fields=[
            {"name": "id", "type": "string", "required": True, "description": "唯一ID"},
            {"name": "q", "type": "string", "required": True, "description": "question"},
        ])
        schema = self.gen._build_schema_from_fields(a)
        # id should appear once in required
        self.assertEqual(schema["required"].count("id"), 1)


class TestBuildGenericSchema(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_generic_schema_structure(self):
        a = _make_analysis_minimal()
        schema = self.gen._build_generic_schema(a)
        self.assertIn("question", schema["properties"])
        self.assertIn("answer", schema["properties"])
        self.assertIn("explanation", schema["properties"])
        self.assertIn("scoring_rubric", schema["properties"])
        self.assertIn("metadata", schema["properties"])
        self.assertIn("id", schema["required"])


# ==================== _extract_task_types ====================


class TestExtractTaskTypes(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_default_when_no_info(self):
        a = _make_analysis(fields=[], examples=[])
        result = self.gen._extract_task_types(a)
        self.assertEqual(result, ["default"])

    def test_from_field_with_slash_description(self):
        a = _make_analysis(
            fields=[
                {"name": "task_type", "type": "string", "description": "类型：understanding/editing/generation"},
            ],
            examples=[],
        )
        result = self.gen._extract_task_types(a)
        self.assertEqual(result, ["understanding", "editing", "generation"])

    def test_from_examples(self):
        a = _make_analysis(
            fields=[],
            examples=[
                {"task_type": "coding", "question": "q1"},
                {"task_type": "math", "question": "q2"},
                {"task_type": "coding", "question": "q3"},  # duplicate
            ],
        )
        result = self.gen._extract_task_types(a)
        self.assertEqual(result, ["coding", "math"])

    def test_from_field_simple_slash(self):
        a = _make_analysis(
            fields=[
                {"name": "task_type", "type": "string", "description": "read/write"},
            ],
            examples=[],
        )
        result = self.gen._extract_task_types(a)
        self.assertEqual(result, ["read", "write"])


# ==================== _analyze_automation_feasibility ====================


class TestAnalyzeAutomationFeasibility(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_no_blockers_high_automation(self):
        a = _make_analysis(
            forbidden_items=[], cognitive_requirements=["分析"],
            estimated_difficulty="medium",
        )
        result = self.gen._analyze_automation_feasibility(a)
        self.assertEqual(result["overall_rate"], 80)
        self.assertTrue(result["default"]["can_automate"])

    def test_forbidden_ai_blocks(self):
        a = _make_analysis(forbidden_items=["禁止使用 AI 生成"])
        result = self.gen._analyze_automation_feasibility(a)
        self.assertEqual(result["overall_rate"], 10)
        self.assertFalse(result["default"]["can_automate"])
        self.assertTrue(any(b["type"] == "forbidden_ai" for b in result["blockers"]))

    def test_creativity_required(self):
        a = _make_analysis(
            forbidden_items=[],
            cognitive_requirements=["创意设计", "原创内容"],
            estimated_difficulty="medium",
        )
        result = self.gen._analyze_automation_feasibility(a)
        self.assertEqual(result["overall_rate"], 30)
        self.assertTrue(any(b["type"] == "creativity_required" for b in result["blockers"]))

    def test_expert_knowledge_required(self):
        a = _make_analysis(
            forbidden_items=[],
            cognitive_requirements=["专业领域知识"],
            estimated_difficulty="expert",
        )
        result = self.gen._analyze_automation_feasibility(a)
        self.assertTrue(any(b["type"] == "expert_knowledge" for b in result["blockers"]))

    def test_difficulty_validation_blocker(self):
        a = _make_analysis_with_difficulty_validation(forbidden_items=[], cognitive_requirements=["分析"])
        result = self.gen._analyze_automation_feasibility(a)
        self.assertTrue(any(b["type"] == "difficulty_validation" for b in result["blockers"]))


# ==================== _generate_sample_svg ====================


class TestGenerateSampleSvg(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_editing_type(self):
        svg = self.gen._generate_sample_svg("editing", 0)
        self.assertIn("<circle", svg)
        self.assertIn("red", svg)

    def test_generation_type(self):
        svg = self.gen._generate_sample_svg("generation", 0)
        self.assertIn("待生成", svg)

    def test_default_type(self):
        svg = self.gen._generate_sample_svg("understanding", 1)
        self.assertIn("<rect", svg)
        self.assertIn("<circle", svg)

    def test_color_rotation(self):
        svg0 = self.gen._generate_sample_svg("edit", 0)
        svg1 = self.gen._generate_sample_svg("edit", 1)
        self.assertIn("red", svg0)
        self.assertIn("blue", svg1)


# ==================== _generate_sample_instruction ====================


class TestGenerateSampleInstruction(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_from_examples(self):
        a = _make_analysis(examples=[
            {"task_type": "math", "question": "What is 2+2?"},
        ])
        result = self.gen._generate_sample_instruction(a, "math", 0)
        self.assertEqual(result, "What is 2+2?")

    def test_understanding_type(self):
        a = _make_analysis(examples=[])
        result = self.gen._generate_sample_instruction(a, "understanding", 0)
        self.assertIn("SVG", result)

    def test_editing_type(self):
        a = _make_analysis(examples=[])
        result = self.gen._generate_sample_instruction(a, "editing", 0)
        self.assertIn("颜色", result)

    def test_generation_type(self):
        a = _make_analysis(examples=[])
        result = self.gen._generate_sample_instruction(a, "generation", 0)
        self.assertIn("创建", result)

    def test_default_type(self):
        a = _make_analysis(examples=[])
        result = self.gen._generate_sample_instruction(a, "unknown_task", 5)
        self.assertIn("执行任务", result)


# ==================== _generate_production_notes ====================


class TestGenerateProductionNotes(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_high_automation(self):
        auto = {"overall_rate": 80, "blockers": []}
        a = _make_analysis()
        notes = self.gen._generate_production_notes(a, auto)
        self.assertIn("批量自动化", notes["recommendation"])

    def test_medium_automation(self):
        auto = {"overall_rate": 50, "blockers": []}
        a = _make_analysis()
        notes = self.gen._generate_production_notes(a, auto)
        self.assertIn("半自动化", notes["recommendation"])

    def test_low_automation(self):
        auto = {"overall_rate": 30, "blockers": []}
        a = _make_analysis()
        notes = self.gen._generate_production_notes(a, auto)
        self.assertIn("人工为主", notes["recommendation"])

    def test_manual_only(self):
        auto = {"overall_rate": 5, "blockers": [{"description": "No AI allowed"}]}
        a = _make_analysis()
        notes = self.gen._generate_production_notes(a, auto)
        self.assertIn("全人工", notes["recommendation"])
        self.assertIn("No AI allowed", notes["key_blockers"])


# ==================== _get_optimization_suggestions ====================


class TestGetOptimizationSuggestions(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_low_automation_gets_template_suggestion(self):
        auto = {"overall_rate": 20}
        a = _make_analysis()
        suggestions = self.gen._get_optimization_suggestions(a, auto)
        areas = [s["area"] for s in suggestions]
        self.assertIn("模板化", areas)

    def test_difficulty_validation_suggestion(self):
        auto = {"overall_rate": 80}
        a = _make_analysis_with_difficulty_validation()
        suggestions = self.gen._get_optimization_suggestions(a, auto)
        areas = [s["area"] for s in suggestions]
        self.assertIn("难度验证", areas)

    def test_many_fields_suggestion(self):
        fields = [{"name": f"f{i}", "type": "string", "required": True, "description": f"field {i}"} for i in range(6)]
        auto = {"overall_rate": 80}
        a = _make_analysis(fields=fields)
        suggestions = self.gen._get_optimization_suggestions(a, auto)
        areas = [s["area"] for s in suggestions]
        self.assertIn("字段简化", areas)

    def test_always_has_qa_suggestion(self):
        auto = {"overall_rate": 80}
        a = _make_analysis()
        suggestions = self.gen._get_optimization_suggestions(a, auto)
        areas = [s["area"] for s in suggestions]
        self.assertIn("质检抽样", areas)


# ==================== _generate_single_sample ====================


class TestGenerateSingleSample(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_basic_sample(self):
        a = _make_analysis()
        auto_info = {"can_automate": False, "automation_rate": 10, "manual_steps": [
            {"step": "content_creation", "reason": "manual"},
        ]}
        sample = self.gen._generate_single_sample(a, "default", 1, auto_info)
        self.assertEqual(sample["id"], "SAMPLE_001")
        self.assertEqual(sample["task_type"], "default")
        self.assertIn("data", sample)
        self.assertIn("think_process", sample)
        self.assertIn("automation_status", sample)
        self.assertFalse(sample["automation_status"]["can_fully_automate"])

    def test_fully_automated_sample(self):
        a = _make_analysis()
        auto_info = {"can_automate": True, "automation_rate": 80}
        sample = self.gen._generate_single_sample(a, "default", 2, auto_info)
        self.assertTrue(sample["automation_status"]["can_fully_automate"])
        self.assertTrue(len(sample["automation_status"]["automated_steps"]) >= 2)

    def test_sample_with_task_type_field(self):
        a = _make_analysis(fields=[
            {"name": "task_type", "type": "string", "description": "type"},
            {"name": "question", "type": "string", "description": "q"},
        ])
        auto_info = {"can_automate": False, "automation_rate": 10, "manual_steps": []}
        sample = self.gen._generate_single_sample(a, "math", 1, auto_info)
        self.assertEqual(sample["data"]["task_type"], "math")

    def test_sample_with_image_field(self):
        a = _make_analysis(fields=[
            {"name": "image", "type": "image", "description": "题目图片"},
        ])
        auto_info = {"can_automate": False, "automation_rate": 0, "manual_steps": []}
        sample = self.gen._generate_single_sample(a, "default", 1, auto_info)
        self.assertIn("图片占位符", sample["data"]["image"])

    def test_sample_with_svg_field(self):
        a = _make_analysis(fields=[
            {"name": "svg_code", "type": "string", "description": "SVG代码"},
        ])
        auto_info = {"can_automate": False, "automation_rate": 0, "manual_steps": []}
        sample = self.gen._generate_single_sample(a, "default", 1, auto_info)
        self.assertIn("svg", sample["data"]["svg_code"].lower())


# ==================== File-writing generator methods (individual) ====================


class TestGenerateCostBreakdown(unittest.TestCase):
    """Test _generate_cost_breakdown writes correct files."""

    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        # Create subdirs
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        for key in ["cost"]:
            os.makedirs(os.path.join(self.tmpdir, self.subdirs[key]), exist_ok=True)

    def test_generates_md_and_json(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_cost_breakdown(a, self.tmpdir, self.subdirs, 100, "china", result)
        # Should generate 2 files
        self.assertEqual(len(result.files_generated), 2)
        # Check md
        md_path = os.path.join(self.tmpdir, self.subdirs["cost"], "COST_BREAKDOWN.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("成本明细", content)
        self.assertIn("TestProject", content)
        # Check json
        json_path = os.path.join(self.tmpdir, self.subdirs["cost"], "cost_breakdown.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["target_size"], 100)
        self.assertIn("grand_total", data)

    def test_hard_difficulty_higher_design_cost(self):
        a = _make_analysis(estimated_difficulty="hard")
        result = SpecOutputResult()
        self.gen._generate_cost_breakdown(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["cost"], "cost_breakdown.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["design_cost"], 2000)

    def test_easy_difficulty_lower_design_cost(self):
        a = _make_analysis(estimated_difficulty="easy")
        result = SpecOutputResult()
        self.gen._generate_cost_breakdown(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["cost"], "cost_breakdown.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["design_cost"], 1200)

    def test_with_images(self):
        a = _make_analysis_with_images()
        result = SpecOutputResult()
        self.gen._generate_cost_breakdown(a, self.tmpdir, self.subdirs, 100, "china", result)
        md_path = os.path.join(self.tmpdir, self.subdirs["cost"], "COST_BREAKDOWN.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("图片制作", content)


class TestGenerateIndustryBenchmark(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["project"]), exist_ok=True)

    def test_generates_md_file(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_industry_benchmark(a, self.tmpdir, self.subdirs, 100, "china", result)
        self.assertTrue(len(result.files_generated) >= 1)
        md_path = os.path.join(self.tmpdir, self.subdirs["project"], "INDUSTRY_BENCHMARK.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("行业基准", content)
        self.assertIn("evaluation", content)

    def test_cost_rating_below_average(self):
        # evaluation benchmark avg=15. Cost with easy difficulty in china = 5*0.6=3 < 15
        a = _make_analysis(estimated_difficulty="easy", forbidden_items=[], has_images=False, reasoning_chain=[])
        result = SpecOutputResult()
        self.gen._generate_industry_benchmark(a, self.tmpdir, self.subdirs, 100, "china", result)
        md_path = os.path.join(self.tmpdir, self.subdirs["project"], "INDUSTRY_BENCHMARK.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("成本低于行业平均", content)

    def test_multimodal_type(self):
        a = _make_analysis(dataset_type="multimodal", estimated_difficulty="easy", forbidden_items=[], has_images=False, reasoning_chain=[])
        result = SpecOutputResult()
        self.gen._generate_industry_benchmark(a, self.tmpdir, self.subdirs, 100, "china", result)
        md_path = os.path.join(self.tmpdir, self.subdirs["project"], "INDUSTRY_BENCHMARK.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("multimodal", content)


class TestGenerateRawAnalysis(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["data"]), exist_ok=True)

    def test_generates_json(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_raw_analysis(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["data"], "spec_analysis.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["project_name"], "TestProject")


class TestGenerateTrainingGuide(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["annotation"]), exist_ok=True)

    def test_basic_guide(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("标注员培训手册", content)
        self.assertIn("逻辑推理", content)  # cognitive requirement
        self.assertIn("推理链", content)
        self.assertIn("question", content)
        self.assertIn("必须遵守的规则", content)
        self.assertIn("质量标准", content)

    def test_with_images(self):
        a = _make_analysis_with_images()
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("图片", content)

    def test_with_examples(self):
        a = _make_analysis(examples=[
            {"question": "What is 2+2?", "answer": "4", "scoring_rubric": "exact match", "analysis": "simple addition"},
        ])
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("What is 2+2?", content)
        self.assertIn("simple addition", content)

    def test_no_examples_generates_template(self):
        a = _make_analysis(examples=[])
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("示例模板", content)

    def test_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("难度验证", content)
        self.assertIn("3 次测试", content)

    def test_with_scoring_rubric(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("评分标准错误", content)

    def test_no_fields_generates_generic_checklist(self):
        a = _make_analysis_minimal()
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("所有必填项已填写", content)

    def test_image_field_error(self):
        a = _make_analysis(fields=[
            {"name": "img", "type": "image", "required": True, "description": "图片"},
        ])
        result = SpecOutputResult()
        self.gen._generate_training_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "TRAINING_GUIDE.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("AI 生成图片", content)


class TestGenerateQaChecklist(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["annotation"]), exist_ok=True)

    def test_basic_checklist(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_qa_checklist(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "QA_CHECKLIST.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("质量检查清单", content)
        self.assertIn("题目检查", content)
        self.assertIn("答案检查", content)

    def test_with_images(self):
        a = _make_analysis_with_images()
        result = SpecOutputResult()
        self.gen._generate_qa_checklist(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "QA_CHECKLIST.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("图片检查", content)
        self.assertIn("原创性", content)

    def test_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_qa_checklist(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "QA_CHECKLIST.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("难度验证检查", content)
        self.assertIn("难度验证通过率", content)

    def test_with_quality_gates(self):
        a = _make_analysis(quality_gates=[
            {"name": "accuracy_gate", "metric": "accuracy", "operator": ">=", "threshold": 0.95, "severity": "blocker"},
        ])
        result = SpecOutputResult()
        self.gen._generate_qa_checklist(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "QA_CHECKLIST.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("质量门禁", content)
        self.assertIn("accuracy_gate", content)

    def test_with_field_constraints(self):
        a = _make_analysis(
            field_constraints=[
                {
                    "field_name": "question",
                    "constraint_type": "length",
                    "rule": "不少于20字",
                    "severity": "error",
                    "auto_checkable": True,
                },
            ]
        )
        result = SpecOutputResult()
        self.gen._generate_qa_checklist(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["annotation"], "QA_CHECKLIST.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("结构化字段约束", content)
        self.assertIn("不少于20字", content)


class TestGenerateDifficultyValidation(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["guide"]), exist_ok=True)

    def test_no_validation_no_file(self):
        a = _make_analysis()  # no difficulty_validation
        result = SpecOutputResult()
        self.gen._generate_difficulty_validation(a, self.tmpdir, self.subdirs, result)
        self.assertEqual(len(result.files_generated), 0)

    def test_with_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_difficulty_validation(a, self.tmpdir, self.subdirs, result)
        self.assertTrue(len(result.files_generated) >= 1)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DIFFICULTY_VALIDATION.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("难度验证流程", content)
        self.assertIn("doubao1.8", content)
        self.assertIn("3", content)  # test_count

    def test_with_images(self):
        a = _make_analysis_with_difficulty_validation(has_images=True, image_count=3)
        result = SpecOutputResult()
        self.gen._generate_difficulty_validation(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DIFFICULTY_VALIDATION.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("上传图片", content)

    def test_batch_table(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_difficulty_validation(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DIFFICULTY_VALIDATION.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("批量记录表格", content)
        self.assertIn("001", content)


class TestGenerateValidationGuide(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["guide"]), exist_ok=True)

    def test_no_strategies_no_files(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_validation_guide(a, self.tmpdir, self.subdirs, result)
        self.assertEqual(len(result.files_generated), 0)

    def test_model_test_delegates_to_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_validation_guide(a, self.tmpdir, self.subdirs, result)
        # Should generate DIFFICULTY_VALIDATION.md
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DIFFICULTY_VALIDATION.md")
        self.assertTrue(os.path.exists(md_path))

    def test_human_review_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "human_review", "enabled": True, "config": {"sample_rate": 0.3}, "description": "人工抽检"},
        ])
        result = SpecOutputResult()
        self.gen._generate_validation_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "VALIDATION_HUMAN_REVIEW.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("human_review", content)
        self.assertIn("审核员", content)

    def test_format_check_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "format_check", "enabled": True, "config": {}, "description": ""},
        ])
        result = SpecOutputResult()
        self.gen._generate_validation_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "VALIDATION_FORMAT_CHECK.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("DATA_SCHEMA.json", content)

    def test_cross_validation_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "cross_validation", "enabled": True, "config": {}, "description": ""},
        ])
        result = SpecOutputResult()
        self.gen._generate_validation_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "VALIDATION_CROSS_VALIDATION.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("Cohen's Kappa", content)

    def test_auto_scoring_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "auto_scoring", "enabled": True, "config": {}, "description": ""},
        ])
        result = SpecOutputResult()
        self.gen._generate_validation_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "VALIDATION_AUTO_SCORING.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("评分函数", content)

    def test_strategy_with_config(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "human_review", "enabled": True, "config": {"sample_rate": 0.5, "reviewer_count": 3}, "description": "detailed"},
        ])
        result = SpecOutputResult()
        self.gen._generate_validation_guide(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "VALIDATION_HUMAN_REVIEW.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("配置参数", content)
        self.assertIn("sample_rate", content)


class TestGenerateDataTemplate(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["templates"]), exist_ok=True)

    def test_basic_template(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_data_template(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["templates"], "data_template.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("id", data)
        self.assertEqual(data["id"], "EXAMPLE_001")
        self.assertIn("question", data)
        self.assertIn("metadata", data)

    def test_with_images(self):
        a = _make_analysis_with_images()
        result = SpecOutputResult()
        self.gen._generate_data_template(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["templates"], "data_template.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("image", data)

    def test_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_data_template(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["templates"], "data_template.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("model_test", data)
        self.assertEqual(data["model_test"]["model"], "doubao1.8")
        self.assertEqual(len(data["model_test"]["results"]), 3)

    def test_no_fields_uses_profile(self):
        a = _make_analysis(fields=[])
        result = SpecOutputResult()
        self.gen._generate_data_template(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["templates"], "data_template.json")
        self.assertTrue(os.path.exists(json_path))


class TestGenerateProductionSop(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["guide"]), exist_ok=True)

    def test_basic_sop(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_production_sop(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "PRODUCTION_SOP.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("生产标准操作流程", content)
        self.assertIn("准备阶段", content)
        self.assertIn("内容创作", content)
        self.assertIn("提交", content)

    def test_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_production_sop(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "PRODUCTION_SOP.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("难度验证", content)
        self.assertIn("doubao1.8", content)
        # Phase numbering should shift
        self.assertIn("阶段4", content)  # quality check phase number (arabic numeral)

    def test_with_images(self):
        a = _make_analysis_with_images(forbidden_items=["禁止使用 AI 生成图片"])
        result = SpecOutputResult()
        self.gen._generate_production_sop(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["guide"], "PRODUCTION_SOP.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("图片制作", content)


class TestGenerateDataSchema(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["guide"]), exist_ok=True)

    def test_with_fields(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_data_schema(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DATA_SCHEMA.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["type"], "object")
        self.assertIn("question", data["properties"])

    def test_without_fields(self):
        a = _make_analysis(fields=[])
        result = SpecOutputResult()
        self.gen._generate_data_schema(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DATA_SCHEMA.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        # Should use generic schema
        self.assertIn("question", data["properties"])
        self.assertIn("answer", data["properties"])

    def test_with_images(self):
        a = _make_analysis_with_images()
        result = SpecOutputResult()
        self.gen._generate_data_schema(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DATA_SCHEMA.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("image", data["properties"])
        self.assertIn("image", data["required"])

    def test_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_data_schema(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["guide"], "DATA_SCHEMA.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("model_test", data["properties"])
        self.assertIn("model_test", data["required"])


# ==================== AI Agent layer ====================


class TestGenerateAiAgentContext(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["ai_agent"]), exist_ok=True)

    def test_basic_context(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_ai_agent_context(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "agent_context.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["project"]["name"], "TestProject")
        self.assertIsNone(data["validation"])
        self.assertIn("file_references", data)
        self.assertIn("quick_actions", data)

    def test_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_ai_agent_context(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "agent_context.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIsNotNone(data["validation"])
        self.assertTrue(data["validation"]["enabled"])
        self.assertIn("difficulty_validation", data["file_references"])


class TestGenerateAiWorkflowState(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["ai_agent"]), exist_ok=True)

    def test_basic_state(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_ai_workflow_state(a, self.tmpdir, self.subdirs, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "workflow_state.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["current_phase"], "ready_for_review")
        self.assertIn("phases", data)
        self.assertIn("analysis", data["phases"])
        self.assertIn("next_actions", data)
        self.assertTrue(len(data["next_actions"]) >= 1)


class TestGenerateAiReasoningTraces(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["ai_agent"]), exist_ok=True)

    def test_basic_traces(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_ai_reasoning_traces(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "reasoning_traces.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("reasoning", data)
        self.assertIn("difficulty", data["reasoning"])
        self.assertIn("cost", data["reasoning"])
        self.assertIn("human_percentage", data["reasoning"])

    def test_difficulty_chain_with_reasoning(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_ai_reasoning_traces(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "reasoning_traces.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        chain = data["reasoning"]["difficulty"]["chain"]
        self.assertTrue(len(chain) >= 1)
        # Should have cognitive, reasoning chain steps
        steps = [c["step"] for c in chain]
        self.assertTrue(any("推理链" in s for s in steps))
        self.assertTrue(any("认知" in s for s in steps))

    def test_with_images_adds_step(self):
        a = _make_analysis_with_images()
        result = SpecOutputResult()
        self.gen._generate_ai_reasoning_traces(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "reasoning_traces.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        chain = data["reasoning"]["difficulty"]["chain"]
        steps = [c["step"] for c in chain]
        self.assertTrue(any("多模态" in s for s in steps))

    def test_human_percentage_with_ai_restriction(self):
        a = _make_analysis(forbidden_items=["禁止使用 AI 生成"])
        result = SpecOutputResult()
        self.gen._generate_ai_reasoning_traces(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "reasoning_traces.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        hp = data["reasoning"]["human_percentage"]
        self.assertEqual(hp["confidence"], 0.95)
        self.assertIn("AI", hp["chain"][0]["evidence"])

    def test_cost_chain_region_china(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_ai_reasoning_traces(a, self.tmpdir, self.subdirs, 100, "china", result)
        json_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "reasoning_traces.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        cost_chain = data["reasoning"]["cost"]["chain"]
        steps = [c["step"] for c in cost_chain]
        self.assertTrue(any("区域" in s for s in steps))


class TestGenerateAiPipeline(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["ai_agent"]), exist_ok=True)

    def test_basic_pipeline(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_ai_pipeline(a, self.tmpdir, self.subdirs, result)
        yaml_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "pipeline.yaml")
        self.assertTrue(os.path.exists(yaml_path))
        with open(yaml_path, "r") as f:
            content = f.read()
        self.assertIn("数据生产流水线", content)
        self.assertIn("setup", content)
        self.assertIn("pilot", content)
        self.assertIn("production", content)
        self.assertIn("final_qa", content)

    def test_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = SpecOutputResult()
        self.gen._generate_ai_pipeline(a, self.tmpdir, self.subdirs, result)
        yaml_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "pipeline.yaml")
        with open(yaml_path, "r") as f:
            content = f.read()
        self.assertIn("validation_model", content)
        self.assertIn("run_model_test", content)

    def test_with_human_review_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "human_review", "enabled": True, "config": {"sample_rate": 0.2}, "description": "人工审核"},
        ])
        result = SpecOutputResult()
        self.gen._generate_ai_pipeline(a, self.tmpdir, self.subdirs, result)
        yaml_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "pipeline.yaml")
        with open(yaml_path, "r") as f:
            content = f.read()
        self.assertIn("human_review", content)
        self.assertIn("sample_rate", content)

    def test_with_format_check_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "format_check", "enabled": True, "config": {}, "description": ""},
        ])
        result = SpecOutputResult()
        self.gen._generate_ai_pipeline(a, self.tmpdir, self.subdirs, result)
        yaml_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "pipeline.yaml")
        with open(yaml_path, "r") as f:
            content = f.read()
        self.assertIn("format_check", content)

    def test_with_cross_validation_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "cross_validation", "enabled": True, "config": {"min_annotators": 3, "min_kappa": 0.8}, "description": ""},
        ])
        result = SpecOutputResult()
        self.gen._generate_ai_pipeline(a, self.tmpdir, self.subdirs, result)
        yaml_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "pipeline.yaml")
        with open(yaml_path, "r") as f:
            content = f.read()
        self.assertIn("cross_validation", content)
        self.assertIn("min_kappa", content)

    def test_with_auto_scoring_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "auto_scoring", "enabled": True, "config": {"threshold": 0.7}, "description": ""},
        ])
        result = SpecOutputResult()
        self.gen._generate_ai_pipeline(a, self.tmpdir, self.subdirs, result)
        yaml_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "pipeline.yaml")
        with open(yaml_path, "r") as f:
            content = f.read()
        self.assertIn("auto_scoring", content)
        self.assertIn("threshold", content)

    def test_with_custom_strategy(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "custom_check", "enabled": True, "config": {}, "description": "Custom validation"},
        ])
        result = SpecOutputResult()
        self.gen._generate_ai_pipeline(a, self.tmpdir, self.subdirs, result)
        yaml_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "pipeline.yaml")
        with open(yaml_path, "r") as f:
            content = f.read()
        self.assertIn("custom_check", content)
        self.assertIn("Custom validation", content)


class TestGenerateAiReadme(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["ai_agent"]), exist_ok=True)

    def test_basic_readme(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_ai_readme(a, self.tmpdir, self.subdirs, result)
        md_path = os.path.join(self.tmpdir, self.subdirs["ai_agent"], "README.md")
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("AI Agent 入口", content)
        self.assertIn("agent_context.json", content)
        self.assertIn("workflow_state.json", content)
        self.assertIn("reasoning_traces.json", content)
        self.assertIn("pipeline.yaml", content)


# ==================== Samples generation ====================


class TestGenerateThinkPoSamples(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["samples"]), exist_ok=True)

    def test_basic_samples(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_think_po_samples(a, self.tmpdir, self.subdirs, 10, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["samples"], "samples.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertIn("samples", data)
        self.assertIn("automation_summary", data)
        self.assertIn("production_notes", data)
        self.assertTrue(len(data["samples"]) > 0)

    def test_sample_guide_generated(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_think_po_samples(a, self.tmpdir, self.subdirs, 5, result)
        guide_path = os.path.join(self.tmpdir, self.subdirs["samples"], "SAMPLE_GUIDE.md")
        self.assertTrue(os.path.exists(guide_path))
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertIn("样例数据指南", content)
        self.assertIn("自动化评估", content)

    def test_capped_at_target_size(self):
        a = _make_analysis()
        result = SpecOutputResult()
        self.gen._generate_think_po_samples(a, self.tmpdir, self.subdirs, 3, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["samples"], "samples.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertTrue(len(data["samples"]) <= 3)

    def test_multiple_task_types(self):
        a = _make_analysis(
            fields=[
                {"name": "task_type", "type": "string", "description": "类型：read/write"},
                {"name": "question", "type": "string", "description": "q"},
            ],
            examples=[],
        )
        result = SpecOutputResult()
        self.gen._generate_think_po_samples(a, self.tmpdir, self.subdirs, 20, result)
        json_path = os.path.join(self.tmpdir, self.subdirs["samples"], "samples.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        task_types = {s["task_type"] for s in data["samples"]}
        self.assertIn("read", task_types)
        self.assertIn("write", task_types)


# ==================== _generate_samples_guide ====================


class TestGenerateSamplesGuide(unittest.TestCase):
    def setUp(self):
        self.gen = SpecOutputGenerator()
        self.tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        self.subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(self.tmpdir, self.subdirs["samples"]), exist_ok=True)

    def _make_samples_doc(self, automation_rate=10, blockers=None, samples=None):
        if samples is None:
            samples = [
                {
                    "id": "S001",
                    "task_type": "default",
                    "data": {"question": "test"},
                    "think_process": {
                        "step_1_parse": "Parse input",
                        "step_2_exec": "Execute",
                    },
                    "automation_status": {
                        "can_fully_automate": False,
                        "automation_rate": automation_rate,
                        "automated_steps": [],
                        "manual_steps": [
                            {"step": "content_creation", "reason": "manual", "effort": "高"},
                        ],
                    },
                },
            ]
        return {
            "_meta": {"total_samples": len(samples), "target_size": 100},
            "automation_summary": {
                "overall_automation_rate": automation_rate,
                "fully_automated_tasks": [],
                "partially_automated_tasks": [],
                "manual_tasks": ["default"],
                "automation_blockers": blockers or [],
            },
            "samples": samples,
            "production_notes": {
                "recommendation": "全人工生产",
                "suggested_workflow": "人工创作",
                "key_blockers": [],
                "optimization_suggestions": [
                    {"area": "模板化", "suggestion": "Use templates", "impact": "10% improvement"},
                ],
            },
        }

    def test_basic_guide(self):
        a = _make_analysis()
        samples_doc = self._make_samples_doc()
        result = SpecOutputResult()
        self.gen._generate_samples_guide(a, self.tmpdir, self.subdirs, samples_doc, result)
        guide_path = os.path.join(self.tmpdir, self.subdirs["samples"], "SAMPLE_GUIDE.md")
        self.assertTrue(os.path.exists(guide_path))
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertIn("样例数据指南", content)

    def test_high_automation_status(self):
        a = _make_analysis()
        samples_doc = self._make_samples_doc(automation_rate=80)
        result = SpecOutputResult()
        self.gen._generate_samples_guide(a, self.tmpdir, self.subdirs, samples_doc, result)
        guide_path = os.path.join(self.tmpdir, self.subdirs["samples"], "SAMPLE_GUIDE.md")
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertIn("高度自动化", content)

    def test_medium_automation_status(self):
        a = _make_analysis()
        samples_doc = self._make_samples_doc(automation_rate=50)
        result = SpecOutputResult()
        self.gen._generate_samples_guide(a, self.tmpdir, self.subdirs, samples_doc, result)
        guide_path = os.path.join(self.tmpdir, self.subdirs["samples"], "SAMPLE_GUIDE.md")
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertIn("半自动化", content)

    def test_low_automation_status(self):
        a = _make_analysis()
        samples_doc = self._make_samples_doc(automation_rate=30)
        result = SpecOutputResult()
        self.gen._generate_samples_guide(a, self.tmpdir, self.subdirs, samples_doc, result)
        guide_path = os.path.join(self.tmpdir, self.subdirs["samples"], "SAMPLE_GUIDE.md")
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertIn("低自动化", content)

    def test_with_blockers(self):
        a = _make_analysis()
        samples_doc = self._make_samples_doc(blockers=[
            {"type": "forbidden_ai", "description": "No AI allowed", "impact": "Manual only"},
        ])
        result = SpecOutputResult()
        self.gen._generate_samples_guide(a, self.tmpdir, self.subdirs, samples_doc, result)
        guide_path = os.path.join(self.tmpdir, self.subdirs["samples"], "SAMPLE_GUIDE.md")
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertIn("自动化阻塞因素", content)
        self.assertIn("No AI allowed", content)

    def test_no_manual_steps(self):
        sample = {
            "id": "S001",
            "task_type": "default",
            "data": {"question": "test"},
            "think_process": {"step_1_p": "parse"},
            "automation_status": {
                "can_fully_automate": True,
                "automation_rate": 90,
                "automated_steps": [{"step": "all", "method": "auto"}],
                "manual_steps": [],
            },
        }
        a = _make_analysis()
        samples_doc = self._make_samples_doc(automation_rate=90, samples=[sample])
        result = SpecOutputResult()
        self.gen._generate_samples_guide(a, self.tmpdir, self.subdirs, samples_doc, result)
        guide_path = os.path.join(self.tmpdir, self.subdirs["samples"], "SAMPLE_GUIDE.md")
        with open(guide_path, "r") as f:
            content = f.read()
        self.assertIn("无需人工参与", content)


# ==================== Full orchestrator: generate() ====================


class TestGenerateOrchestrator(unittest.TestCase):
    """Test the main generate() method end-to-end using a temp directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.gen = SpecOutputGenerator(output_dir=self.tmpdir)

    def test_full_generation_basic(self):
        a = _make_analysis()
        result = self.gen.generate(a, target_size=10, region="china")
        self.assertTrue(result.success, f"generate() failed: {result.error}")
        self.assertTrue(len(result.files_generated) > 10)
        self.assertIn("README.md", result.files_generated)

    def test_full_generation_with_images(self):
        a = _make_analysis_with_images()
        result = self.gen.generate(a, target_size=5, region="china")
        self.assertTrue(result.success, f"generate() failed: {result.error}")

    def test_full_generation_with_difficulty_validation(self):
        a = _make_analysis_with_difficulty_validation()
        result = self.gen.generate(a, target_size=5, region="china")
        self.assertTrue(result.success, f"generate() failed: {result.error}")

    def test_full_generation_minimal(self):
        a = _make_analysis_minimal()
        result = self.gen.generate(a, target_size=5, region="us")
        self.assertTrue(result.success, f"generate() failed: {result.error}")

    def test_full_generation_with_validation_strategies(self):
        a = _make_analysis(validation_strategies=[
            {"strategy_type": "human_review", "enabled": True, "config": {"sample_rate": 0.2}, "description": "人工审核"},
            {"strategy_type": "format_check", "enabled": True, "config": {}, "description": "格式检查"},
        ])
        result = self.gen.generate(a, target_size=5, region="china")
        self.assertTrue(result.success, f"generate() failed: {result.error}")

    def test_output_dir_created(self):
        a = _make_analysis(project_name="My Project")
        result = self.gen.generate(a, target_size=5, region="china")
        self.assertTrue(result.success)
        self.assertTrue(os.path.isdir(result.output_dir))

    def test_error_handling(self):
        """Test that generate() catches exceptions and returns error result."""
        a = _make_analysis()
        # Use a read-only directory to trigger an error
        gen = SpecOutputGenerator(output_dir="/nonexistent/path/that/should/not/exist")
        result = gen.generate(a, target_size=5, region="china")
        self.assertFalse(result.success)
        self.assertTrue(len(result.error) > 0)

    def test_default_project_name(self):
        a = _make_analysis(project_name="")
        result = self.gen.generate(a, target_size=5, region="china")
        self.assertTrue(result.success, f"generate() failed: {result.error}")

    def test_with_enhanced_context(self):
        """Test that enhanced_context parameter is accepted and passed through."""
        # Use None to test that the parameter is accepted without side effects;
        # A real EnhancedContext would be needed for deeper integration tests.
        a = _make_analysis()
        result = self.gen.generate(a, target_size=5, region="china", enhanced_context=None)
        self.assertTrue(result.success, f"generate() failed: {result.error}")

    def test_with_examples(self):
        a = _make_analysis(examples=[
            {"question": "Q1", "answer": "A1", "scoring_rubric": "exact", "analysis": "simple"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ])
        result = self.gen.generate(a, target_size=5, region="china")
        self.assertTrue(result.success, f"generate() failed: {result.error}")


# ==================== Edge cases ====================


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_empty_enum_template(self):
        # Empty list is falsy, so it skips the enum block and falls through to string
        fd = FieldDefinition(name="x", type="string", enum=[])
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertIn("请填写", val)

    def test_enum_sample_no_context(self):
        fd = FieldDefinition(name="x", type="string", enum=["a", "b"])
        val = self.gen._generate_value_from_field(fd, mode="sample")
        self.assertEqual(val, "a")

    def test_deeply_nested_fields(self):
        fd = FieldDefinition(
            name="root",
            type="object",
            properties=[
                FieldDefinition(
                    name="child",
                    type="object",
                    properties=[
                        FieldDefinition(name="leaf", type="string", description="deep leaf"),
                    ],
                ),
            ],
        )
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertIsInstance(val, dict)
        self.assertIn("child", val)
        self.assertIsInstance(val["child"], dict)
        self.assertIn("leaf", val["child"])

    def test_field_definition_no_description(self):
        fd = FieldDefinition(name="x", type="string")
        val = self.gen._generate_value_from_field(fd, mode="template")
        self.assertIn("x", val)  # uses name as fallback for desc

    def test_forbidden_items_non_ai(self):
        """Forbidden items that don't contain 'AI' - should still affect cost."""
        a = _make_analysis(forbidden_items=["no copying"])
        cost = self.gen._estimate_cost_per_item(a, "china")
        a2 = _make_analysis(forbidden_items=[])
        cost2 = self.gen._estimate_cost_per_item(a2, "china")
        self.assertGreater(cost, cost2)

    def test_qa_checklist_with_gate_id_fallback(self):
        """Quality gates with gate_id instead of name."""
        tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(tmpdir, subdirs["annotation"]), exist_ok=True)

        a = _make_analysis(quality_gates=[
            {"gate_id": "gate_1", "metric": "acc", "operator": ">=", "threshold": 0.9},
        ])
        result = SpecOutputResult()
        self.gen._generate_qa_checklist(a, tmpdir, subdirs, result)
        md_path = os.path.join(tmpdir, subdirs["annotation"], "QA_CHECKLIST.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("gate_1", content)

    def test_cost_high_above_benchmark(self):
        """Cost above max benchmark should show red rating."""
        tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(tmpdir, subdirs["project"]), exist_ok=True)

        # expert difficulty + images + long reasoning + forbidden + US = very expensive
        a = _make_analysis(
            estimated_difficulty="expert",
            has_images=True,
            image_count=10,
            reasoning_chain=["a", "b", "c", "d", "e"],
            forbidden_items=["no AI"],
        )
        result = SpecOutputResult()
        self.gen._generate_industry_benchmark(a, tmpdir, subdirs, 100, "us", result)
        md_path = os.path.join(tmpdir, subdirs["project"], "INDUSTRY_BENCHMARK.md")
        with open(md_path, "r") as f:
            content = f.read()
        self.assertIn("成本高于行业基准", content)


class TestRemainingCoveragePaths(unittest.TestCase):
    """Cover the remaining missed lines for near-100% coverage."""

    def setUp(self):
        self.gen = SpecOutputGenerator()

    def test_data_template_skips_id_field(self):
        """Line 1293: field named 'id' should be skipped in template."""
        tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(tmpdir, subdirs["templates"]), exist_ok=True)

        a = _make_analysis(fields=[
            {"name": "id", "type": "string", "required": True, "description": "unique ID"},
            {"name": "question", "type": "string", "required": True, "description": "q"},
        ])
        result = SpecOutputResult()
        self.gen._generate_data_template(a, tmpdir, subdirs, result)
        json_path = os.path.join(tmpdir, subdirs["templates"], "data_template.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        # id should be the static EXAMPLE_001, not a placeholder
        self.assertEqual(data["id"], "EXAMPLE_001")

    def test_reasoning_traces_forbidden_without_ai(self):
        """Line 2042: forbidden_items that don't contain 'AI'."""
        tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(tmpdir, subdirs["ai_agent"]), exist_ok=True)

        a = _make_analysis(forbidden_items=["禁止抄袭", "禁止复制"])
        result = SpecOutputResult()
        self.gen._generate_ai_reasoning_traces(a, tmpdir, subdirs, 100, "china", result)
        json_path = os.path.join(tmpdir, subdirs["ai_agent"], "reasoning_traces.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        hp = data["reasoning"]["human_percentage"]
        # Should hit the else branch (non-AI forbidden items)
        chain = hp["chain"]
        self.assertTrue(len(chain) >= 1)
        self.assertIn("内容限制", chain[0]["step"])

    def test_samples_break_on_max(self):
        """Line 2472: samples loop breaks when hitting max_samples."""
        tmpdir = tempfile.mkdtemp()
        from datarecipe.core.project_layout import OUTPUT_SUBDIRS
        subdirs = OUTPUT_SUBDIRS
        os.makedirs(os.path.join(tmpdir, subdirs["samples"]), exist_ok=True)

        # Multiple task types with small target_size to trigger the break
        a = _make_analysis(
            fields=[
                {"name": "task_type", "type": "string", "description": "类型：a/b/c/d/e"},
                {"name": "q", "type": "string", "description": "question"},
            ],
            examples=[],
        )
        result = SpecOutputResult()
        self.gen._generate_think_po_samples(a, tmpdir, subdirs, 3, result)
        json_path = os.path.join(tmpdir, subdirs["samples"], "samples.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertTrue(len(data["samples"]) <= 3)

    def test_single_sample_no_field_definitions(self):
        """Lines 2654-2655: _generate_single_sample with empty field_definitions."""
        a = _make_analysis(fields=[])
        # Ensure field_definitions is empty
        self.assertEqual(len(a.field_definitions), 0)
        auto_info = {"can_automate": False, "automation_rate": 0, "manual_steps": []}
        sample = self.gen._generate_single_sample(a, "default", 1, auto_info)
        self.assertIn("data", sample)
        self.assertIsInstance(sample["data"], dict)


if __name__ == "__main__":
    unittest.main()
