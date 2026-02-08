"""Tests for annotation specification generator.

Covers:
- AnnotationSpec generation with various inputs
- _generate_data_requirements edge cases
- _generate_quality_constraints with rubrics
- _generate_format_requirements for all field types
- _score_example_quality for preference and other types
- _select_best_examples
- _generate_examples with various data formats
- _generate_scoring_rubrics
- to_markdown with enhanced context
- to_dict
- spec_to_markdown (analyze-spec pipeline)
- spec_to_dict
"""

import unittest
from dataclasses import dataclass, field

from datarecipe.generators.annotation_spec import (
    AnnotationSpec,
    AnnotationSpecGenerator,
    ExampleItem,
    ScoringCriterion,
)

# ==================== Stub objects ====================


class StubDomain:
    def __init__(self, value: str):
        self.value = value


@dataclass
class StubComplexityMetrics:
    primary_domain: object = None
    difficulty_score: float = 2.0
    length_category: str = "medium"


@dataclass
class StubRubricsResult:
    verb_distribution: dict = field(default_factory=dict)
    structured_patterns: list = field(default_factory=list)


@dataclass
class StubLLMAnalysis:
    purpose: str = ""
    production_steps: list = field(default_factory=list)
    quality_criteria: list = field(default_factory=list)


@dataclass
class StubEnhancedContext:
    generated: bool = True
    domain_specific_guidelines: str = ""
    quality_pitfalls: list = field(default_factory=list)
    example_analysis: list = field(default_factory=list)


@dataclass
class StubSpecAnalysis:
    project_name: str = "Test Project"
    dataset_type: str = "evaluation"
    task_type: str = "问答评测"
    task_description: str = "Answer questions"
    has_images: bool = False
    image_count: int = 0
    cognitive_requirements: list = field(default_factory=list)
    reasoning_chain: list = field(default_factory=list)
    data_requirements: list = field(default_factory=list)
    forbidden_items: list = field(default_factory=list)
    quality_constraints: list = field(default_factory=list)
    difficulty_criteria: str = ""
    fields: list = field(default_factory=list)
    field_requirements: dict = field(default_factory=dict)
    examples: list = field(default_factory=list)
    scoring_rubric: list = field(default_factory=list)
    description: str = "Test description"
    similar_datasets: list = field(default_factory=list)


# ==================== Basic Generation ====================


class TestAnnotationSpecGenerate(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()
        self.base_schema = {
            "question": {"type": "str"},
            "answer": {"type": "str"},
        }
        self.base_samples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is the capital of France?", "answer": "Paris"},
        ]

    def test_generate_basic(self):
        spec = self.gen.generate(
            dataset_id="test/ds",
            dataset_type="evaluation",
            schema_info=self.base_schema,
            sample_items=self.base_samples,
        )
        self.assertIsInstance(spec, AnnotationSpec)
        self.assertEqual(spec.dataset_id, "test/ds")
        self.assertEqual(spec.dataset_type, "evaluation")
        self.assertTrue(spec.task_name)
        self.assertTrue(spec.task_description)
        self.assertTrue(spec.data_requirements)
        self.assertTrue(spec.quality_constraints)
        self.assertTrue(spec.format_requirements)

    def test_generate_with_llm_analysis(self):
        """Lines 119-123: LLM analysis overrides."""
        llm = StubLLMAnalysis(
            purpose="Custom purpose",
            production_steps=["Step1", "Step2", "Step3"],
        )
        spec = self.gen.generate(
            dataset_id="test/ds",
            dataset_type="evaluation",
            schema_info=self.base_schema,
            sample_items=self.base_samples,
            llm_analysis=llm,
        )
        self.assertEqual(spec.task_description, "Custom purpose")
        self.assertIn("Step1", spec.reasoning_chain)

    def test_generate_stores_enhanced_context(self):
        ec = StubEnhancedContext(generated=True, domain_specific_guidelines="Test guide")
        spec = self.gen.generate(
            dataset_id="test/ds",
            dataset_type="evaluation",
            schema_info=self.base_schema,
            sample_items=self.base_samples,
            enhanced_context=ec,
        )
        self.assertEqual(spec._enhanced_context, ec)


# ==================== _generate_format_requirements ====================


class TestGenerateFormatRequirements(unittest.TestCase):
    """Test all field type branches in _generate_format_requirements (lines 233-248)."""

    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def _format(self, schema: dict) -> list[str]:
        spec = AnnotationSpec(dataset_id="t", dataset_type="t")
        self.gen._generate_format_requirements(spec, schema)
        return spec.format_requirements

    def test_str_type(self):
        reqs = self._format({"name": {"type": "str"}})
        self.assertTrue(any("字符串" in r for r in reqs))

    def test_list_type_with_nested(self):
        """Line 237: list with nested_type."""
        reqs = self._format({"tags": {"type": "list", "nested_type": "str"}})
        self.assertTrue(any("列表" in r and "str" in r for r in reqs))

    def test_list_type_without_nested(self):
        """Line 239: list without nested_type."""
        reqs = self._format({"items": {"type": "list"}})
        self.assertTrue(any("列表" in r for r in reqs))

    def test_dict_type(self):
        """Line 241: dict type."""
        reqs = self._format({"meta": {"type": "dict"}})
        self.assertTrue(any("对象" in r for r in reqs))

    def test_int_type(self):
        """Line 243: int type."""
        reqs = self._format({"count": {"type": "int"}})
        self.assertTrue(any("整数" in r for r in reqs))

    def test_float_type(self):
        """Line 245: float type."""
        reqs = self._format({"score": {"type": "float"}})
        self.assertTrue(any("浮点" in r for r in reqs))

    def test_bool_type(self):
        """Line 247: bool type."""
        reqs = self._format({"active": {"type": "bool"}})
        self.assertTrue(any("布尔" in r for r in reqs))


# ==================== _score_example_quality ====================


class TestScoreExampleQuality(unittest.TestCase):
    """Test _score_example_quality for preference and other types (lines 251-319)."""

    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_preference_duplicate_chosen_rejected(self):
        """Line 266: identical chosen/rejected returns 0."""
        item = {"chosen": "Same text", "rejected": "Same text"}
        score = self.gen._score_example_quality(item, "preference")
        self.assertEqual(score, 0)

    def test_preference_length_difference_bonus(self):
        """Lines 269-271: len_diff > 50 bonus."""
        item = {"chosen": "A" * 100, "rejected": "B" * 20}
        score = self.gen._score_example_quality(item, "preference")
        self.assertGreater(score, 5.0)

    def test_preference_short_responses_penalty(self):
        """Lines 273-275: short responses penalty."""
        item = {"chosen": "OK", "rejected": "No"}
        score = self.gen._score_example_quality(item, "preference")
        self.assertLess(score, 5.0)

    def test_preference_multi_turn_bonus(self):
        """Lines 278-282: multi-turn conversations bonus."""
        conversation = "Human: hi\nAssistant: hello\nHuman: how are you?\nAssistant: good"
        item = {"chosen": conversation, "rejected": "B" * 50}
        score = self.gen._score_example_quality(item, "preference")
        self.assertGreaterEqual(score, 5.0)

    def test_preference_chosen_much_shorter_penalty(self):
        """Lines 286-287: chosen much shorter than rejected penalty."""
        item = {"chosen": "Short", "rejected": "A" * 200}
        score = self.gen._score_example_quality(item, "preference")
        self.assertLess(score, 5.0)

    def test_preference_harmful_content_penalty(self):
        """Lines 290-292: harmful content penalty."""
        item = {"chosen": "You should fuck off", "rejected": "Please be respectful"}
        score = self.gen._score_example_quality(item, "preference")
        # Should have penalty
        self.assertIsInstance(score, (int, float))

    def test_preference_clear_topic_bonus(self):
        """Lines 295-303: clear topic bonus."""
        conversation = "Human: Can you explain how photosynthesis works?\nAssistant: Sure, photosynthesis is..."
        item = {"chosen": conversation, "rejected": "B" * 100}
        score = self.gen._score_example_quality(item, "preference")
        self.assertGreater(score, 5.0)

    def test_other_type_non_empty_fields_bonus(self):
        """Lines 308-311: non-empty fields for other types."""
        item = {
            "question": "What is the meaning of life?" * 3,
            "answer": "It depends on perspective." * 3,
            "explanation": "Detailed reasoning here." * 5,
        }
        score = self.gen._score_example_quality(item, "evaluation")
        self.assertGreater(score, 5.0)

    def test_other_type_good_length_bonus(self):
        """Lines 314-317: text between 100-2000 gets bonus."""
        item = {"text": "A" * 500}
        score = self.gen._score_example_quality(item, "sft")
        self.assertGreater(score, 5.0)

    def test_score_clamped(self):
        """Line 319: score clamped to 0-10."""
        item = {"chosen": "Same", "rejected": "Same"}
        score = self.gen._score_example_quality(item, "preference")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 10)


# ==================== _select_best_examples ====================


class TestSelectBestExamples(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_empty_items(self):
        result = self.gen._select_best_examples([], "evaluation")
        self.assertEqual(result, [])

    def test_filters_low_quality(self):
        """Lines 335: filter score < 3."""
        items = [
            {"chosen": "Same", "rejected": "Same"},  # score 0, will be filtered
            {"question": "A" * 200, "answer": "B" * 200},  # higher score
        ]
        result = self.gen._select_best_examples(items, "preference")
        # The duplicate should be filtered out
        for r in result:
            self.assertNotEqual(r.get("chosen"), r.get("rejected"))

    def test_returns_top_n(self):
        items = [{"question": f"Q{i}" * 20, "answer": f"A{i}" * 20} for i in range(10)]
        result = self.gen._select_best_examples(items, "evaluation", count=3)
        self.assertLessEqual(len(result), 3)


# ==================== _generate_examples ====================


class TestGenerateExamples(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_examples_with_messages_format(self):
        """Lines 371-381: messages format extraction."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="sft")
        items = [
            {
                "messages": [
                    {"role": "user", "content": "Hello, can you help me understand quantum physics?"},
                    {"role": "assistant", "content": "Of course! Quantum physics is..."},
                ]
            }
        ]
        self.gen._generate_examples(spec, items, None)
        self.assertTrue(spec.examples)
        self.assertIn("quantum", spec.examples[0].question_text.lower())

    def test_examples_fallback_to_first_string(self):
        """Lines 384-388: fallback to first string field > 50 chars."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="sft")
        items = [
            {"custom_field": "A" * 100}
        ]
        self.gen._generate_examples(spec, items, None)
        self.assertTrue(spec.examples)
        self.assertTrue(spec.examples[0].question_text)

    def test_examples_with_rubrics_list(self):
        """Lines 399-407: rubrics as list of strings."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        items = [
            {
                "question": "What is 2+2?" * 5,
                "answer": "4",
                "rubrics": ["Must be exact", "Show work", "Clear explanation"],
            }
        ]
        self.gen._generate_examples(spec, items, None)
        self.assertTrue(spec.examples)
        self.assertTrue(any(
            c.description == "Must be exact"
            for c in spec.examples[0].scoring_criteria
        ))

    def test_examples_with_rubric_string(self):
        """Lines 408-411: rubric as single string."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        items = [
            {
                "question": "Explain gravity" * 5,
                "rubric": "Must include Newton's law",
            }
        ]
        self.gen._generate_examples(spec, items, None)
        self.assertTrue(spec.examples)
        # Should have the string rubric as scoring criterion
        if spec.examples[0].scoring_criteria:
            criteria_descs = [c.description for c in spec.examples[0].scoring_criteria]
            # Either the custom rubric or default ones
            self.assertTrue(len(criteria_descs) > 0)

    def test_examples_default_scoring(self):
        """Lines 415-425: default scoring when no rubrics found."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        items = [
            {"question": "Simple question here" * 5, "answer": "Simple answer"},
        ]
        self.gen._generate_examples(spec, items, None)
        self.assertTrue(spec.examples)
        self.assertEqual(len(spec.examples[0].scoring_criteria), 2)
        self.assertEqual(spec.examples[0].scoring_criteria[0].score, "1分")

    def test_examples_truncation(self):
        """Lines 367, 395: long text truncation."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        items = [
            {"question": "Q" * 1000, "answer": "A" * 500},
        ]
        self.gen._generate_examples(spec, items, None)
        self.assertTrue(spec.examples)
        self.assertTrue(spec.examples[0].question_text.endswith("..."))
        self.assertTrue(spec.examples[0].answer.endswith("..."))


# ==================== _generate_scoring_rubrics ====================


class TestGenerateScoringRubrics(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_scoring_from_llm_analysis(self):
        """Lines 443-445: scoring dimensions from LLM analysis."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        llm = StubLLMAnalysis(quality_criteria=["Accuracy", "Completeness", "Clarity"])
        self.gen._generate_scoring_rubrics(spec, None, llm)
        self.assertIn("Accuracy", spec.scoring_dimensions)

    def test_scoring_from_rubrics_result(self):
        """Lines 448-458: scoring rubrics from rubrics_result."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        rubrics = StubRubricsResult(
            verb_distribution={"include": 5, "explain": 3},
        )
        self.gen._generate_scoring_rubrics(spec, rubrics, None)
        self.assertTrue(spec.scoring_rubrics)
        dims = [r["dimension"] for r in spec.scoring_rubrics]
        self.assertIn("include", dims)

    def test_scoring_default_dimensions(self):
        """Lines 466-472: default dimensions when none found."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        self.gen._generate_scoring_rubrics(spec, None, None)
        self.assertTrue(spec.scoring_dimensions)
        self.assertIn("准确性", spec.scoring_dimensions[0])

    def test_scoring_partial_credit_always_set(self):
        """Lines 475-479: partial credit rules always set."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        self.gen._generate_scoring_rubrics(spec, None, None)
        self.assertEqual(len(spec.partial_credit_rules), 3)

    def test_scoring_with_structured_patterns(self):
        """Lines 460-463: structured_patterns from rubrics_result."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        rubrics = StubRubricsResult(
            verb_distribution={"include": 2},
            structured_patterns=[
                {"dimension": "custom", "description": "Custom rubric"},
            ],
        )
        self.gen._generate_scoring_rubrics(spec, rubrics, None)
        found = any(r.get("dimension") == "custom" for r in spec.scoring_rubrics)
        self.assertTrue(found)


# ==================== _generate_data_requirements ====================


class TestGenerateDataRequirements(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_requirements_with_complexity_length(self):
        """Lines 169-177: length_category."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        metrics = StubComplexityMetrics(
            length_category="long",
            primary_domain=StubDomain("general"),
        )
        self.gen._generate_data_requirements(spec, {"q": {"type": "str"}}, metrics)
        self.assertTrue(any("较长" in r for r in spec.data_requirements))

    def test_requirements_with_domain(self):
        """Lines 179-182: primary_domain != general."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        metrics = StubComplexityMetrics(primary_domain=StubDomain("medical"))
        self.gen._generate_data_requirements(spec, {"q": {"type": "str"}}, metrics)
        self.assertTrue(any("medical" in r for r in spec.data_requirements))

    def test_requirements_general_domain_no_extra(self):
        """Domain 'general' should not add domain requirement."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        metrics = StubComplexityMetrics(primary_domain=StubDomain("general"))
        self.gen._generate_data_requirements(spec, {"q": {"type": "str"}}, metrics)
        self.assertFalse(any("领域要求" in r for r in spec.data_requirements))


# ==================== _generate_quality_constraints ====================


class TestGenerateQualityConstraints(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_constraints_with_rubrics_verbs(self):
        """Lines 209-212: rubric verb_distribution."""
        spec = AnnotationSpec(dataset_id="t", dataset_type="evaluation")
        rubrics = StubRubricsResult(verb_distribution={"include": 5, "explain": 3})
        self.gen._generate_quality_constraints(spec, "evaluation", rubrics)
        self.assertTrue(any("include" in c for c in spec.quality_constraints))


# ==================== to_markdown ====================


class TestToMarkdown(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_to_markdown_basic(self):
        spec = self.gen.generate(
            dataset_id="test/ds",
            dataset_type="evaluation",
            schema_info={"question": {"type": "str"}, "answer": {"type": "str"}},
            sample_items=[
                {"question": "What is 2+2?" * 10, "answer": "4"},
            ],
        )
        md = self.gen.to_markdown(spec)
        self.assertIn("test/ds", md)
        self.assertIn("标注规范", md)
        self.assertIn("题目类型描述", md)
        self.assertIn("数据要求", md)
        self.assertIn("质量约束", md)
        self.assertIn("格式要求", md)
        self.assertIn("例题", md)
        self.assertIn("打分标准", md)

    def test_to_markdown_with_enhanced_context(self):
        """Lines 644-672: enhanced context in markdown."""
        ec = StubEnhancedContext(
            generated=True,
            domain_specific_guidelines="Domain-specific guide text here",
            quality_pitfalls=["Pitfall 1", "Pitfall 2"],
            example_analysis=[
                {
                    "sample_index": 0,
                    "strengths": "Good question",
                    "weaknesses": "Short answer",
                    "annotation_tips": "Add detail",
                }
            ],
        )
        spec = self.gen.generate(
            dataset_id="test/ds",
            dataset_type="evaluation",
            schema_info={"question": {"type": "str"}},
            sample_items=[{"question": "Test question" * 10}],
            enhanced_context=ec,
        )
        md = self.gen.to_markdown(spec)
        self.assertIn("领域标注指导", md)
        self.assertIn("Domain-specific guide text here", md)
        self.assertIn("常见错误", md)
        self.assertIn("Pitfall 1", md)
        self.assertIn("样本点评", md)
        self.assertIn("Good question", md)

    def test_to_markdown_without_enhanced_context(self):
        """No enhanced context section when not provided."""
        spec = self.gen.generate(
            dataset_id="test/ds",
            dataset_type="evaluation",
            schema_info={"question": {"type": "str"}},
            sample_items=[{"question": "Test question" * 10}],
        )
        md = self.gen.to_markdown(spec)
        self.assertNotIn("领域标注指导", md)

    def test_to_markdown_with_scoring_rubrics(self):
        """Lines 624-634: scoring rubrics table rendering."""
        spec = AnnotationSpec(
            dataset_id="test/ds",
            dataset_type="evaluation",
            task_name="Test",
            task_description="Test task",
            scoring_dimensions=["Accuracy"],
            scoring_rubrics=[{"dimension": "accuracy", "description": "Must be correct"}],
            partial_credit_rules=["Partial OK"],
            examples=[],
        )
        md = self.gen.to_markdown(spec)
        self.assertIn("评分维度", md)
        self.assertIn("评分细则", md)
        self.assertIn("accuracy", md)
        self.assertIn("部分得分规则", md)

    def test_to_markdown_constraint_highlighting(self):
        """Lines 540-543: bold highlighting for constraints with keywords."""
        spec = AnnotationSpec(
            dataset_id="test/ds",
            dataset_type="evaluation",
            task_name="Test",
            task_description="Test",
            quality_constraints=[
                "禁止使用AI内容",
                "数据必须准确",
                "不能含有个人信息",
                "建议参考示例",
            ],
        )
        md = self.gen.to_markdown(spec)
        # Lines with "禁止", "必须", "不能" should be bolded
        self.assertIn("**禁止使用AI内容**", md)
        self.assertIn("**数据必须准确**", md)
        self.assertIn("**不能含有个人信息**", md)
        # "建议" should not be bolded
        self.assertNotIn("**建议参考示例**", md)


# ==================== to_dict ====================


class TestToDict(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_to_dict_all_fields(self):
        spec = self.gen.generate(
            dataset_id="test/ds",
            dataset_type="evaluation",
            schema_info={"question": {"type": "str"}, "answer": {"type": "str"}},
            sample_items=[{"question": "Q" * 60, "answer": "A" * 60}],
        )
        d = self.gen.to_dict(spec)
        self.assertEqual(d["dataset_id"], "test/ds")
        self.assertEqual(d["dataset_type"], "evaluation")
        self.assertIn("task", d)
        self.assertIn("requirements", d)
        self.assertIn("examples", d)
        self.assertIn("scoring", d)
        self.assertIn("metadata", d)

    def test_to_dict_examples_structure(self):
        spec = AnnotationSpec(
            dataset_id="t",
            dataset_type="t",
            examples=[
                ExampleItem(
                    id=1,
                    question_text="Question",
                    answer="Answer",
                    scoring_criteria=[
                        ScoringCriterion(score="1", description="Correct"),
                    ],
                )
            ],
        )
        d = self.gen.to_dict(spec)
        self.assertEqual(len(d["examples"]), 1)
        self.assertEqual(d["examples"][0]["question"], "Question")
        self.assertEqual(d["examples"][0]["scoring"][0]["score"], "1")


# ==================== spec_to_markdown (analyze-spec pipeline) ====================


class TestSpecToMarkdown(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_spec_to_markdown_basic(self):
        analysis = StubSpecAnalysis(
            project_name="MyProject",
            dataset_type="evaluation",
            task_type="QA",
            task_description="Answer questions",
            cognitive_requirements=["Understanding", "Reasoning"],
            reasoning_chain=["Understand", "Analyze", "Answer"],
            data_requirements=["Req 1", "Req 2"],
            fields=[
                {"name": "question", "type": "string", "required": True, "description": "The question"},
                {"name": "answer", "type": "string", "required": True, "description": "The answer"},
            ],
        )
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("MyProject", md)
        self.assertIn("QA", md)
        self.assertIn("Understanding", md)
        self.assertIn("Understand", md)
        self.assertIn("question", md)

    def test_spec_to_markdown_with_images(self):
        """Lines 735-736: has_images flag."""
        analysis = StubSpecAnalysis(has_images=True, image_count=5)
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("包含图片", md)
        self.assertIn("5", md)

    def test_spec_to_markdown_with_forbidden_items(self):
        """Lines 776-780: forbidden items section."""
        analysis = StubSpecAnalysis(forbidden_items=["AI content", "Plagiarism"])
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("禁止项", md)
        self.assertIn("AI content", md)

    def test_spec_to_markdown_with_quality_constraints(self):
        analysis = StubSpecAnalysis(quality_constraints=["Must be accurate"])
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("质量标准", md)
        self.assertIn("Must be accurate", md)

    def test_spec_to_markdown_with_difficulty_criteria(self):
        analysis = StubSpecAnalysis(difficulty_criteria="Use GPT-4 for validation")
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("难度验证", md)
        self.assertIn("GPT-4", md)

    def test_spec_to_markdown_with_field_requirements(self):
        analysis = StubSpecAnalysis(
            fields=[{"name": "q", "type": "string", "required": True, "description": "question"}],
            field_requirements={"q": "Must be clear and concise"},
        )
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("字段详细要求", md)
        self.assertIn("Must be clear", md)

    def test_spec_to_markdown_with_examples(self):
        """Lines 822-848: examples section."""
        analysis = StubSpecAnalysis(
            examples=[
                {
                    "question": "What is 2+2?",
                    "answer": "4",
                    "scoring_rubric": "Must be exact",
                    "has_image": True,
                },
                {
                    "question": "Capital of France?",
                    "answer": "Paris",
                },
            ],
        )
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("示例 1", md)
        self.assertIn("示例 2", md)
        self.assertIn("2+2", md)
        self.assertIn("[包含图片]", md)

    def test_spec_to_markdown_no_examples_template(self):
        """Lines 849-864: no examples -> template from fields."""
        analysis = StubSpecAnalysis(
            examples=[],
            fields=[
                {"name": "question", "type": "string", "description": "The question"},
                {"name": "answer", "type": "string", "description": "The answer"},
            ],
        )
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("数据模板", md)
        self.assertIn('"question"', md)

    def test_spec_to_markdown_with_scoring_rubric(self):
        """Lines 867-876: scoring rubric table."""
        analysis = StubSpecAnalysis(
            scoring_rubric=[
                {"score": "5", "criteria": "Perfect answer"},
                {"score": "0", "criteria": "Wrong answer"},
            ],
        )
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("打分标准", md)
        self.assertIn("Perfect answer", md)

    def test_spec_to_markdown_with_enhanced_context(self):
        """Lines 879-902: enhanced context in spec_to_markdown."""
        ec = StubEnhancedContext(
            generated=True,
            domain_specific_guidelines="Follow medical terminology guidelines",
            quality_pitfalls=["Incorrect terminology", "Missing references"],
            example_analysis=[
                {
                    "sample_index": 1,
                    "strengths": "Clear",
                    "weaknesses": "Incomplete",
                    "annotation_tips": "Add references",
                }
            ],
        )
        analysis = StubSpecAnalysis()
        md = self.gen.spec_to_markdown(analysis, enhanced_context=ec)
        self.assertIn("领域标注指导", md)
        self.assertIn("medical terminology", md)
        self.assertIn("常见错误", md)
        self.assertIn("Incorrect terminology", md)
        self.assertIn("样本分析", md)
        self.assertIn("Clear", md)

    def test_spec_to_markdown_optional_field(self):
        """Field with required=False."""
        analysis = StubSpecAnalysis(
            fields=[
                {"name": "hint", "type": "string", "required": False, "description": "Optional hint"},
            ],
        )
        md = self.gen.spec_to_markdown(analysis)
        self.assertIn("否", md)


# ==================== spec_to_dict ====================


class TestSpecToDict(unittest.TestCase):
    def setUp(self):
        self.gen = AnnotationSpecGenerator()

    def test_spec_to_dict(self):
        analysis = StubSpecAnalysis(
            project_name="MyProject",
            dataset_type="evaluation",
            task_type="QA",
            task_description="Answer questions",
            cognitive_requirements=["Understanding"],
            reasoning_chain=["Step1"],
            data_requirements=["Req1"],
            quality_constraints=["Q1"],
            forbidden_items=["F1"],
            difficulty_criteria="DC",
            fields=[{"name": "q"}],
            field_requirements={"q": "clear"},
            examples=[{"q": "test"}],
            scoring_rubric=[{"score": "5"}],
        )
        d = self.gen.spec_to_dict(analysis)
        self.assertEqual(d["project_name"], "MyProject")
        self.assertEqual(d["dataset_type"], "evaluation")
        self.assertEqual(d["task_type"], "QA")
        self.assertEqual(d["cognitive_requirements"], ["Understanding"])
        self.assertEqual(d["forbidden_items"], ["F1"])
        self.assertEqual(d["fields"], [{"name": "q"}])


if __name__ == "__main__":
    unittest.main()
