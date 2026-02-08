"""Comprehensive unit tests for SpecAnalyzer and related dataclasses.

Covers:
- FieldDefinition: to_dict, from_dict, to_json_schema (all branches)
- FieldConstraint: to_dict, from_dict
- ValidationStrategy: to_dict, from_dict, from_difficulty_validation
- SpecificationAnalysis: properties, computed fields, to_dict
- SpecAnalyzer: __init__, parse_document, get_extraction_prompt,
  create_analysis_from_json, analyze, _extract_with_llm,
  _call_anthropic, _call_openai, _parse_json_response
- Error handling: invalid JSON, API failures, missing fields
"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from datarecipe.analyzers.spec_analyzer import (
    FieldConstraint,
    FieldDefinition,
    SpecAnalyzer,
    SpecificationAnalysis,
    ValidationStrategy,
    _map_type,
)
from datarecipe.parsers import ParsedDocument

# --------------- helpers ---------------


def _make_parsed_doc(
    text: str = "Sample specification document content.",
    images: list[dict] | None = None,
    file_path: str = "/tmp/test_spec.md",
    file_type: str = "text",
) -> ParsedDocument:
    """Create a ParsedDocument for testing."""
    return ParsedDocument(
        file_path=file_path,
        file_type=file_type,
        text_content=text,
        images=images or [],
        pages=1,
    )


def _make_extracted_json(**overrides) -> dict[str, Any]:
    """Create a minimal extracted JSON dict with sensible defaults."""
    defaults: dict[str, Any] = {
        "project_name": "TestProject",
        "dataset_type": "evaluation",
        "description": "A test dataset.",
        "task_type": "reasoning",
        "task_description": "Solve reasoning problems.",
        "cognitive_requirements": ["logic"],
        "reasoning_chain": ["understand", "analyze", "solve"],
        "data_requirements": ["high quality"],
        "quality_constraints": ["unique answers"],
        "forbidden_items": ["no AI"],
        "difficulty_criteria": "expert level",
        "fields": [
            {"name": "question", "type": "string", "required": True, "description": "The question"},
            {"name": "answer", "type": "string", "required": True, "description": "The answer"},
        ],
        "field_requirements": {"question": "at least 20 chars"},
        "field_constraints": [
            {
                "field_name": "question",
                "constraint_type": "format",
                "rule": "Must be a complete sentence",
                "severity": "error",
                "auto_checkable": True,
            }
        ],
        "validation_strategies": [],
        "quality_gates": [],
        "examples": [{"question": "What is 1+1?", "answer": "2"}],
        "scoring_rubric": [{"score": "1", "criteria": "correct"}],
        "estimated_difficulty": "hard",
        "estimated_domain": "math",
        "estimated_human_percentage": 90,
        "similar_datasets": ["GSM8K"],
        "difficulty_validation": None,
    }
    defaults.update(overrides)
    return defaults


# ==================== _map_type ====================


class TestMapType(unittest.TestCase):
    """Test the type mapping utility function."""

    def test_standard_types(self):
        self.assertEqual(_map_type("string"), "string")
        self.assertEqual(_map_type("text"), "string")
        self.assertEqual(_map_type("code"), "string")
        self.assertEqual(_map_type("image"), "string")
        self.assertEqual(_map_type("number"), "number")
        self.assertEqual(_map_type("float"), "number")
        self.assertEqual(_map_type("double"), "number")
        self.assertEqual(_map_type("integer"), "integer")
        self.assertEqual(_map_type("int"), "integer")
        self.assertEqual(_map_type("boolean"), "boolean")
        self.assertEqual(_map_type("bool"), "boolean")
        self.assertEqual(_map_type("array"), "array")
        self.assertEqual(_map_type("list"), "array")
        self.assertEqual(_map_type("object"), "object")
        self.assertEqual(_map_type("dict"), "object")
        self.assertEqual(_map_type("map"), "object")

    def test_case_insensitive(self):
        self.assertEqual(_map_type("STRING"), "string")
        self.assertEqual(_map_type("Integer"), "integer")

    def test_whitespace_stripped(self):
        self.assertEqual(_map_type("  string  "), "string")

    def test_unknown_falls_back_to_string(self):
        self.assertEqual(_map_type("unknown"), "string")
        self.assertEqual(_map_type("custom_type"), "string")


# ==================== FieldDefinition ====================


class TestFieldDefinitionToDict(unittest.TestCase):
    """Test FieldDefinition.to_dict()."""

    def test_minimal(self):
        fd = FieldDefinition(name="q", type="string")
        d = fd.to_dict()
        self.assertEqual(d, {"name": "q", "type": "string"})

    def test_all_fields(self):
        fd = FieldDefinition(
            name="tags",
            type="array",
            description="Tag list",
            required=True,
            items=FieldDefinition(name="tag", type="string"),
            enum=None,
            min_length=1,
            max_length=100,
        )
        d = fd.to_dict()
        self.assertEqual(d["name"], "tags")
        self.assertTrue(d["required"])
        self.assertEqual(d["description"], "Tag list")
        self.assertIn("items", d)
        self.assertEqual(d["items"]["type"], "string")
        self.assertEqual(d["min_length"], 1)
        self.assertEqual(d["max_length"], 100)

    def test_with_properties(self):
        fd = FieldDefinition(
            name="obj",
            type="object",
            properties=[
                FieldDefinition(name="a", type="string"),
                FieldDefinition(name="b", type="number"),
            ],
        )
        d = fd.to_dict()
        self.assertEqual(len(d["properties"]), 2)
        self.assertEqual(d["properties"][0]["name"], "a")

    def test_with_enum(self):
        fd = FieldDefinition(name="level", type="string", enum=["A", "B", "C"])
        d = fd.to_dict()
        self.assertEqual(d["enum"], ["A", "B", "C"])

    def test_with_any_of(self):
        """Cover line 82: any_of serialization."""
        fd = FieldDefinition(
            name="value",
            type="string",
            any_of=[
                FieldDefinition(name="str_val", type="string"),
                FieldDefinition(name="num_val", type="number"),
            ],
        )
        d = fd.to_dict()
        self.assertIn("any_of", d)
        self.assertEqual(len(d["any_of"]), 2)
        self.assertEqual(d["any_of"][0]["type"], "string")
        self.assertEqual(d["any_of"][1]["type"], "number")

    def test_with_constraints(self):
        fd = FieldDefinition(
            name="x", type="number",
            minimum=0.0, maximum=100.0, pattern=r"\d+",
        )
        d = fd.to_dict()
        self.assertEqual(d["minimum"], 0.0)
        self.assertEqual(d["maximum"], 100.0)
        self.assertEqual(d["pattern"], r"\d+")


class TestFieldDefinitionFromDict(unittest.TestCase):
    """Test FieldDefinition.from_dict()."""

    def test_minimal(self):
        fd = FieldDefinition.from_dict({"name": "q"})
        self.assertEqual(fd.name, "q")
        self.assertEqual(fd.type, "string")
        self.assertFalse(fd.required)

    def test_required_truthy_string(self):
        fd = FieldDefinition.from_dict({"name": "q", "required": "yes"})
        self.assertTrue(fd.required)

    def test_required_false_string(self):
        fd = FieldDefinition.from_dict({"name": "q", "required": "no"})
        self.assertFalse(fd.required)

    def test_with_nested_items(self):
        fd = FieldDefinition.from_dict({
            "name": "tags",
            "type": "array",
            "items": {"name": "tag", "type": "string"},
        })
        self.assertIsNotNone(fd.items)
        self.assertEqual(fd.items.name, "tag")

    def test_with_properties(self):
        fd = FieldDefinition.from_dict({
            "name": "obj",
            "type": "object",
            "properties": [
                {"name": "a", "type": "string"},
                {"name": "b", "type": "integer"},
            ],
        })
        self.assertIsNotNone(fd.properties)
        self.assertEqual(len(fd.properties), 2)

    def test_with_any_of(self):
        fd = FieldDefinition.from_dict({
            "name": "x",
            "any_of": [
                {"name": "str", "type": "string"},
                {"name": "num", "type": "number"},
            ],
        })
        self.assertIsNotNone(fd.any_of)
        self.assertEqual(len(fd.any_of), 2)

    def test_camel_case_constraints(self):
        """Backwards compatibility with camelCase constraint names."""
        fd = FieldDefinition.from_dict({
            "name": "x", "type": "string",
            "minLength": 5, "maxLength": 50,
        })
        self.assertEqual(fd.min_length, 5)
        self.assertEqual(fd.max_length, 50)


class TestFieldDefinitionToJsonSchema(unittest.TestCase):
    """Test FieldDefinition.to_json_schema()."""

    def test_basic_string(self):
        fd = FieldDefinition(name="q", type="string", description="question")
        schema = fd.to_json_schema()
        self.assertEqual(schema["type"], "string")
        self.assertEqual(schema["description"], "question")

    def test_string_with_constraints(self):
        """Cover line 135: string minLength/maxLength."""
        fd = FieldDefinition(name="q", type="string", min_length=5, max_length=200)
        schema = fd.to_json_schema()
        self.assertEqual(schema["minLength"], 5)
        self.assertEqual(schema["maxLength"], 200)

    def test_number_with_constraints(self):
        """Cover line 142: numeric minimum/maximum."""
        fd = FieldDefinition(name="score", type="number", minimum=0.0, maximum=100.0)
        schema = fd.to_json_schema()
        self.assertEqual(schema["type"], "number")
        self.assertEqual(schema["minimum"], 0.0)
        self.assertEqual(schema["maximum"], 100.0)

    def test_integer_with_constraints(self):
        fd = FieldDefinition(name="count", type="integer", minimum=1, maximum=10)
        schema = fd.to_json_schema()
        self.assertEqual(schema["type"], "integer")
        self.assertEqual(schema["minimum"], 1)
        self.assertEqual(schema["maximum"], 10)

    def test_enum(self):
        fd = FieldDefinition(name="x", type="string", enum=["a", "b", "c"])
        schema = fd.to_json_schema()
        self.assertEqual(schema["enum"], ["a", "b", "c"])

    def test_pattern(self):
        fd = FieldDefinition(name="x", type="string", pattern=r"^\d{4}$")
        schema = fd.to_json_schema()
        self.assertEqual(schema["pattern"], r"^\d{4}$")

    def test_array_with_items(self):
        fd = FieldDefinition(
            name="tags", type="array",
            items=FieldDefinition(name="tag", type="string"),
        )
        schema = fd.to_json_schema()
        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["items"]["type"], "string")

    def test_object_with_properties(self):
        fd = FieldDefinition(
            name="meta", type="object",
            properties=[
                FieldDefinition(name="key", type="string", required=True),
                FieldDefinition(name="val", type="string", required=False),
            ],
        )
        schema = fd.to_json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("key", schema["properties"])
        self.assertEqual(schema["required"], ["key"])

    def test_any_of_removes_type(self):
        fd = FieldDefinition(
            name="x", type="string",
            any_of=[
                FieldDefinition(name="s", type="string"),
                FieldDefinition(name="n", type="number"),
            ],
        )
        schema = fd.to_json_schema()
        self.assertNotIn("type", schema)
        self.assertIn("anyOf", schema)
        self.assertEqual(len(schema["anyOf"]), 2)

    def test_no_constraints_on_wrong_type(self):
        """String constraints should not appear on number type."""
        fd = FieldDefinition(name="x", type="number", min_length=5)
        schema = fd.to_json_schema()
        self.assertNotIn("minLength", schema)


# ==================== FieldConstraint ====================


class TestFieldConstraint(unittest.TestCase):
    def test_to_dict(self):
        fc = FieldConstraint(
            field_name="q", constraint_type="format",
            rule="must be sentence", severity="warning", auto_checkable=True,
        )
        d = fc.to_dict()
        self.assertEqual(d["field_name"], "q")
        self.assertEqual(d["constraint_type"], "format")
        self.assertTrue(d["auto_checkable"])

    def test_from_dict(self):
        fc = FieldConstraint.from_dict({
            "field_name": "x", "constraint_type": "range",
            "rule": ">0", "severity": "info", "auto_checkable": True,
        })
        self.assertEqual(fc.field_name, "x")
        self.assertEqual(fc.constraint_type, "range")
        self.assertEqual(fc.severity, "info")

    def test_from_dict_defaults(self):
        fc = FieldConstraint.from_dict({})
        self.assertEqual(fc.field_name, "")
        self.assertEqual(fc.constraint_type, "general")
        self.assertEqual(fc.severity, "error")
        self.assertFalse(fc.auto_checkable)


# ==================== ValidationStrategy ====================


class TestValidationStrategy(unittest.TestCase):
    def test_to_dict(self):
        vs = ValidationStrategy(
            strategy_type="model_test", enabled=True,
            config={"model": "gpt-4"}, description="Test with GPT-4",
        )
        d = vs.to_dict()
        self.assertEqual(d["strategy_type"], "model_test")
        self.assertTrue(d["enabled"])
        self.assertEqual(d["config"]["model"], "gpt-4")

    def test_from_dict(self):
        vs = ValidationStrategy.from_dict({
            "strategy_type": "human_review", "enabled": False,
            "config": {"rate": 0.3}, "description": "Manual review",
        })
        self.assertEqual(vs.strategy_type, "human_review")
        self.assertFalse(vs.enabled)
        self.assertEqual(vs.config["rate"], 0.3)

    def test_from_dict_defaults(self):
        vs = ValidationStrategy.from_dict({})
        self.assertEqual(vs.strategy_type, "")
        self.assertTrue(vs.enabled)
        self.assertEqual(vs.config, {})

    def test_from_difficulty_validation(self):
        diff = {
            "model": "doubao1.8",
            "settings": "high depth",
            "test_count": 3,
            "max_correct": 1,
            "pass_criteria": "3 tests max 1 correct",
            "requires_record": True,
        }
        vs = ValidationStrategy.from_difficulty_validation(diff)
        self.assertEqual(vs.strategy_type, "model_test")
        self.assertTrue(vs.enabled)
        self.assertEqual(vs.config["model"], "doubao1.8")
        self.assertEqual(vs.config["test_count"], 3)
        self.assertIn("doubao1.8", vs.description)

    def test_from_difficulty_validation_defaults(self):
        vs = ValidationStrategy.from_difficulty_validation({})
        self.assertEqual(vs.strategy_type, "model_test")
        self.assertEqual(vs.config["test_count"], 3)


# ==================== SpecificationAnalysis ====================


class TestSpecificationAnalysis(unittest.TestCase):
    def test_field_definitions_cached(self):
        sa = SpecificationAnalysis(
            fields=[
                {"name": "q", "type": "string", "required": True, "description": "question"},
            ]
        )
        defs = sa.field_definitions
        self.assertEqual(len(defs), 1)
        self.assertEqual(defs[0].name, "q")
        # Access again - should hit the cache
        defs2 = sa.field_definitions
        self.assertIs(defs, defs2)

    def test_field_definitions_empty(self):
        sa = SpecificationAnalysis(fields=[])
        defs = sa.field_definitions
        self.assertEqual(len(defs), 0)

    def test_parsed_constraints_new_format(self):
        sa = SpecificationAnalysis(
            field_constraints=[
                {"field_name": "q", "constraint_type": "format", "rule": "sentence"},
            ],
        )
        constraints = sa.parsed_constraints
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].field_name, "q")

    def test_parsed_constraints_legacy_field_requirements(self):
        sa = SpecificationAnalysis(
            field_requirements={"question": "at least 20 chars"},
        )
        constraints = sa.parsed_constraints
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].field_name, "question")
        self.assertEqual(constraints[0].rule, "at least 20 chars")

    def test_parsed_constraints_legacy_quality_constraints(self):
        sa = SpecificationAnalysis(
            quality_constraints=["unique answers"],
        )
        constraints = sa.parsed_constraints
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].field_name, "_global")
        self.assertEqual(constraints[0].rule, "unique answers")

    def test_parsed_constraints_dedup(self):
        """New-format constraint should prevent duplicate from legacy."""
        sa = SpecificationAnalysis(
            field_constraints=[
                {"field_name": "q", "constraint_type": "general", "rule": "at least 20 chars"},
            ],
            field_requirements={"q": "at least 20 chars"},
        )
        constraints = sa.parsed_constraints
        # Should only have 1 constraint, not 2
        q_constraints = [c for c in constraints if c.field_name == "q"]
        self.assertEqual(len(q_constraints), 1)

    def test_constraints_for_field(self):
        sa = SpecificationAnalysis(
            field_constraints=[
                {"field_name": "q", "constraint_type": "format", "rule": "rule1"},
                {"field_name": "a", "constraint_type": "format", "rule": "rule2"},
            ],
            quality_constraints=["global rule"],
        )
        q_constraints = sa.constraints_for_field("q")
        # Should include q-specific + _global
        field_names = [c.field_name for c in q_constraints]
        self.assertIn("q", field_names)
        self.assertIn("_global", field_names)
        self.assertNotIn("a", field_names)

    def test_parsed_validation_strategies_new_format(self):
        sa = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "human_review", "enabled": True, "config": {}, "description": ""},
            ],
        )
        strategies = sa.parsed_validation_strategies
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].strategy_type, "human_review")

    def test_parsed_validation_strategies_legacy_difficulty(self):
        sa = SpecificationAnalysis(
            difficulty_validation={"model": "gpt-4", "test_count": 3, "max_correct": 1},
        )
        strategies = sa.parsed_validation_strategies
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].strategy_type, "model_test")

    def test_parsed_validation_strategies_no_dup_model_test(self):
        """If new-format already has model_test, legacy should not add another."""
        sa = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "model_test", "enabled": True, "config": {}, "description": ""},
            ],
            difficulty_validation={"model": "gpt-4"},
        )
        strategies = sa.parsed_validation_strategies
        model_tests = [s for s in strategies if s.strategy_type == "model_test"]
        self.assertEqual(len(model_tests), 1)

    def test_get_strategy(self):
        sa = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "human_review", "enabled": True, "config": {}, "description": ""},
                {"strategy_type": "format_check", "enabled": False, "config": {}, "description": ""},
            ],
        )
        self.assertIsNotNone(sa.get_strategy("human_review"))
        # disabled strategy should return None
        self.assertIsNone(sa.get_strategy("format_check"))
        self.assertIsNone(sa.get_strategy("nonexistent"))

    def test_has_strategy(self):
        sa = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "human_review", "enabled": True, "config": {}, "description": ""},
            ],
        )
        self.assertTrue(sa.has_strategy("human_review"))
        self.assertFalse(sa.has_strategy("format_check"))

    def test_to_dict(self):
        sa = SpecificationAnalysis(
            project_name="Test",
            dataset_type="evaluation",
            fields=[{"name": "q", "type": "string", "required": True, "description": "q"}],
        )
        d = sa.to_dict()
        self.assertEqual(d["project_name"], "Test")
        self.assertIn("field_definitions", d)
        self.assertNotIn("raw_text", d)

    def test_to_dict_no_fields(self):
        sa = SpecificationAnalysis(project_name="Test", fields=[])
        d = sa.to_dict()
        self.assertNotIn("field_definitions", d)

    def test_has_difficulty_validation_legacy(self):
        sa = SpecificationAnalysis(difficulty_validation={"model": "gpt-4"})
        self.assertTrue(sa.has_difficulty_validation())

    def test_has_difficulty_validation_new_strategy(self):
        sa = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "model_test", "enabled": True, "config": {}, "description": ""},
            ],
        )
        self.assertTrue(sa.has_difficulty_validation())

    def test_has_difficulty_validation_none(self):
        sa = SpecificationAnalysis()
        self.assertFalse(sa.has_difficulty_validation())


# ==================== SpecAnalyzer.__init__ ====================


class TestSpecAnalyzerInit(unittest.TestCase):
    """Test SpecAnalyzer initialization."""

    def test_default_provider(self):
        analyzer = SpecAnalyzer()
        self.assertEqual(analyzer.provider, "anthropic")
        self.assertIsNone(analyzer._last_doc)

    def test_custom_provider(self):
        analyzer = SpecAnalyzer(provider="openai")
        self.assertEqual(analyzer.provider, "openai")

    def test_parser_created(self):
        analyzer = SpecAnalyzer()
        self.assertIsNotNone(analyzer.parser)


# ==================== SpecAnalyzer.parse_document ====================


class TestParseDocument(unittest.TestCase):
    """Test parse_document delegates to DocumentParser."""

    @patch.object(SpecAnalyzer, "__init__", lambda self, **kw: None)
    def test_delegates_to_parser(self):
        analyzer = SpecAnalyzer()
        analyzer.provider = "anthropic"
        analyzer._last_doc = None
        mock_parser = MagicMock()
        fake_doc = _make_parsed_doc(text="hello world")
        mock_parser.parse.return_value = fake_doc
        analyzer.parser = mock_parser

        result = analyzer.parse_document("/tmp/test.md")

        mock_parser.parse.assert_called_once_with("/tmp/test.md")
        self.assertEqual(result.text_content, "hello world")
        self.assertIs(analyzer._last_doc, fake_doc)


# ==================== SpecAnalyzer.get_extraction_prompt ====================


class TestGetExtractionPrompt(unittest.TestCase):
    """Test get_extraction_prompt builds LLM prompt."""

    def setUp(self):
        self.analyzer = SpecAnalyzer()

    def test_with_explicit_doc(self):
        doc = _make_parsed_doc(text="My specification content")
        prompt = self.analyzer.get_extraction_prompt(doc)
        self.assertIn("My specification content", prompt)
        self.assertIn("JSON", prompt)

    def test_uses_last_doc_if_none(self):
        doc = _make_parsed_doc(text="Last doc content")
        self.analyzer._last_doc = doc
        prompt = self.analyzer.get_extraction_prompt()
        self.assertIn("Last doc content", prompt)

    def test_raises_when_no_doc(self):
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.get_extraction_prompt()
        self.assertIn("No document parsed", str(ctx.exception))

    def test_truncates_long_content(self):
        doc = _make_parsed_doc(text="x" * 20000)
        prompt = self.analyzer.get_extraction_prompt(doc)
        # The prompt uses doc.text_content[:15000]
        self.assertLessEqual(len(prompt), len(SpecAnalyzer.EXTRACTION_PROMPT) + 15000 + 100)


# ==================== SpecAnalyzer._parse_json_response ====================


class TestParseJsonResponse(unittest.TestCase):
    """Test JSON parsing from LLM response text."""

    def setUp(self):
        self.analyzer = SpecAnalyzer()

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"project_name": "Test"}\n```\nDone.'
        result = self.analyzer._parse_json_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["project_name"], "Test")

    def test_raw_json_object(self):
        text = 'The response is: {"project_name": "RawTest", "type": "eval"}'
        result = self.analyzer._parse_json_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["project_name"], "RawTest")

    def test_no_json_found(self):
        text = "This is just plain text without any JSON."
        result = self.analyzer._parse_json_response(text)
        self.assertIsNone(result)

    def test_malformed_json(self):
        text = '```json\n{"project_name": "broken",}\n```'
        result = self.analyzer._parse_json_response(text)
        self.assertIsNone(result)

    def test_json_with_nested_braces(self):
        text = '{"project_name": "Test", "config": {"key": "value"}}'
        result = self.analyzer._parse_json_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["config"]["key"], "value")

    def test_json_code_block_preferred_over_raw(self):
        """If both code block and raw JSON exist, code block should be used."""
        text = (
            '{"stray": true}\n'
            '```json\n{"project_name": "CodeBlock"}\n```'
        )
        result = self.analyzer._parse_json_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["project_name"], "CodeBlock")

    def test_empty_string(self):
        result = self.analyzer._parse_json_response("")
        self.assertIsNone(result)


# ==================== SpecAnalyzer.create_analysis_from_json ====================


class TestCreateAnalysisFromJson(unittest.TestCase):
    """Test building SpecificationAnalysis from extracted JSON."""

    def setUp(self):
        self.analyzer = SpecAnalyzer()

    def test_full_json(self):
        doc = _make_parsed_doc(text="full content", images=[{"data": "abc", "type": "image/png"}])
        extracted = _make_extracted_json()
        analysis = self.analyzer.create_analysis_from_json(extracted, doc)

        self.assertEqual(analysis.project_name, "TestProject")
        self.assertEqual(analysis.dataset_type, "evaluation")
        self.assertEqual(analysis.task_type, "reasoning")
        self.assertEqual(analysis.estimated_difficulty, "hard")
        self.assertEqual(analysis.estimated_domain, "math")
        self.assertEqual(analysis.estimated_human_percentage, 90)
        self.assertEqual(analysis.raw_text, "full content")
        self.assertTrue(analysis.has_images)
        self.assertEqual(analysis.image_count, 1)
        self.assertEqual(len(analysis.fields), 2)
        self.assertEqual(len(analysis.examples), 1)
        self.assertEqual(len(analysis.scoring_rubric), 1)
        self.assertEqual(analysis.similar_datasets, ["GSM8K"])

    def test_with_difficulty_validation_enabled(self):
        extracted = _make_extracted_json(
            difficulty_validation={
                "enabled": True,
                "model": "doubao1.8",
                "settings": "high",
                "test_count": 5,
                "max_correct": 2,
                "pass_criteria": "5 tests max 2 correct",
                "requires_record": True,
            },
        )
        analysis = self.analyzer.create_analysis_from_json(extracted)
        self.assertIsNotNone(analysis.difficulty_validation)
        self.assertEqual(analysis.difficulty_validation["model"], "doubao1.8")
        self.assertEqual(analysis.difficulty_validation["test_count"], 5)

    def test_with_difficulty_validation_disabled(self):
        extracted = _make_extracted_json(
            difficulty_validation={"enabled": False, "model": "gpt-4"},
        )
        analysis = self.analyzer.create_analysis_from_json(extracted)
        self.assertIsNone(analysis.difficulty_validation)

    def test_with_difficulty_validation_no_model(self):
        extracted = _make_extracted_json(
            difficulty_validation={"enabled": True, "model": ""},
        )
        analysis = self.analyzer.create_analysis_from_json(extracted)
        self.assertIsNone(analysis.difficulty_validation)

    def test_with_difficulty_validation_null(self):
        extracted = _make_extracted_json(difficulty_validation=None)
        analysis = self.analyzer.create_analysis_from_json(extracted)
        self.assertIsNone(analysis.difficulty_validation)

    def test_uses_last_doc_if_none(self):
        doc = _make_parsed_doc(text="stored doc")
        self.analyzer._last_doc = doc
        extracted = _make_extracted_json()
        analysis = self.analyzer.create_analysis_from_json(extracted)
        self.assertEqual(analysis.raw_text, "stored doc")

    def test_no_doc_at_all(self):
        """When no doc is provided or stored, raw_text should be empty."""
        extracted = _make_extracted_json()
        analysis = self.analyzer.create_analysis_from_json(extracted)
        self.assertEqual(analysis.raw_text, "")
        self.assertFalse(analysis.has_images)
        self.assertEqual(analysis.image_count, 0)

    def test_empty_extracted_dict(self):
        analysis = self.analyzer.create_analysis_from_json({})
        self.assertEqual(analysis.project_name, "")
        self.assertEqual(analysis.estimated_difficulty, "hard")  # default
        self.assertEqual(analysis.estimated_human_percentage, 95)  # default
        self.assertEqual(analysis.fields, [])

    def test_field_constraints_preserved(self):
        extracted = _make_extracted_json()
        analysis = self.analyzer.create_analysis_from_json(extracted)
        self.assertEqual(len(analysis.field_constraints), 1)
        self.assertEqual(analysis.field_constraints[0]["field_name"], "question")


# ==================== SpecAnalyzer._extract_with_llm ====================


class TestExtractWithLlm(unittest.TestCase):
    """Test _extract_with_llm routing to providers."""

    def test_anthropic_provider(self):
        analyzer = SpecAnalyzer(provider="anthropic")
        doc = _make_parsed_doc()
        with patch.object(analyzer, "_call_anthropic", return_value={"key": "val"}) as mock_call:
            result = analyzer._extract_with_llm(doc)
            mock_call.assert_called_once()
            self.assertEqual(result, {"key": "val"})

    def test_openai_provider(self):
        analyzer = SpecAnalyzer(provider="openai")
        doc = _make_parsed_doc()
        with patch.object(analyzer, "_call_openai", return_value={"key": "val"}) as mock_call:
            result = analyzer._extract_with_llm(doc)
            mock_call.assert_called_once()
            self.assertEqual(result, {"key": "val"})

    def test_unknown_provider(self):
        analyzer = SpecAnalyzer(provider="unknown_llm")
        doc = _make_parsed_doc()
        with self.assertRaises(ValueError) as ctx:
            analyzer._extract_with_llm(doc)
        self.assertIn("Unknown provider", str(ctx.exception))

    def test_passes_images(self):
        analyzer = SpecAnalyzer(provider="anthropic")
        images = [{"data": "base64data", "type": "image/png"}]
        doc = _make_parsed_doc(images=images)
        with patch.object(analyzer, "_call_anthropic") as mock_call:
            analyzer._extract_with_llm(doc)
            args = mock_call.call_args
            # Second argument should be the images list
            self.assertEqual(args[0][1], images)


# ==================== SpecAnalyzer._call_anthropic ====================


class TestCallAnthropic(unittest.TestCase):
    """Test Anthropic API call with mocked client."""

    def setUp(self):
        self.analyzer = SpecAnalyzer(provider="anthropic")

    @patch("datarecipe.analyzers.spec_analyzer.anthropic", create=True)
    def test_successful_call(self, mock_anthropic_module):
        """Test successful Anthropic API call."""
        # We need to patch the import inside the method
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"project_name": "Test"}\n```')]
        mock_client.messages.create.return_value = mock_response

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            mock_anthropic_module.Anthropic.return_value = mock_client
            mock_anthropic_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
            result = self.analyzer._call_anthropic("test prompt", [])

        self.assertIsNotNone(result)
        self.assertEqual(result["project_name"], "Test")

    @patch.dict("sys.modules", {"anthropic": None})
    def test_import_error(self):
        """When anthropic is not installed, ImportError should be raised."""
        # Force the module to not be importable
        import sys
        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None
        try:
            with self.assertRaises(ImportError) as ctx:
                self.analyzer._call_anthropic("test prompt", [])
            self.assertIn("anthropic", str(ctx.exception).lower())
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                sys.modules.pop("anthropic", None)

    def test_authentication_error(self):
        """Test handling of AuthenticationError."""
        mock_module = MagicMock()
        mock_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_module.AuthenticationError("bad key")
        mock_module.Anthropic.return_value = mock_client

        import sys
        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_module
        try:
            result = self.analyzer._call_anthropic("test prompt", [])
            self.assertIsNone(result)
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                sys.modules.pop("anthropic", None)

    def test_generic_exception(self):
        """Test handling of generic exceptions."""
        mock_module = MagicMock()
        mock_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("network error")
        mock_module.Anthropic.return_value = mock_client

        import sys
        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_module
        try:
            result = self.analyzer._call_anthropic("test prompt", [])
            self.assertIsNone(result)
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                sys.modules.pop("anthropic", None)

    def test_with_images(self):
        """Test that images are included in the API call."""
        mock_module = MagicMock()
        mock_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"project_name": "ImgTest"}')]
        mock_client.messages.create.return_value = mock_response
        mock_module.Anthropic.return_value = mock_client

        images = [
            {"data": "base64img1", "type": "image/png"},
            {"data": "base64img2", "type": "image/jpeg"},
        ]

        import sys
        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_module
        try:
            result = self.analyzer._call_anthropic("test prompt", images)
            self.assertIsNotNone(result)
            # Verify the messages content includes images
            call_args = mock_client.messages.create.call_args
            messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[1].get("messages")
            content = messages[0]["content"]
            # First 2 items should be images, last should be text
            self.assertEqual(content[0]["type"], "image")
            self.assertEqual(content[1]["type"], "image")
            self.assertEqual(content[2]["type"], "text")
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                sys.modules.pop("anthropic", None)

    def test_images_limited_to_five(self):
        """At most 5 images should be sent."""
        mock_module = MagicMock()
        mock_module.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"project_name": "Test"}')]
        mock_client.messages.create.return_value = mock_response
        mock_module.Anthropic.return_value = mock_client

        images = [{"data": f"img{i}", "type": "image/png"} for i in range(10)]

        import sys
        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mock_module
        try:
            self.analyzer._call_anthropic("test prompt", images)
            call_args = mock_client.messages.create.call_args
            messages = call_args[1]["messages"]
            content = messages[0]["content"]
            image_items = [c for c in content if c["type"] == "image"]
            self.assertEqual(len(image_items), 5)
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                sys.modules.pop("anthropic", None)


# ==================== SpecAnalyzer._call_openai ====================


class TestCallOpenai(unittest.TestCase):
    """Test OpenAI API call with mocked client."""

    def setUp(self):
        self.analyzer = SpecAnalyzer(provider="openai")

    def test_successful_call(self):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '```json\n{"project_name": "OpenAITest"}\n```'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_module.OpenAI.return_value = mock_client

        import sys
        orig = sys.modules.get("openai")
        sys.modules["openai"] = mock_module
        try:
            result = self.analyzer._call_openai("test prompt", [])
            self.assertIsNotNone(result)
            self.assertEqual(result["project_name"], "OpenAITest")
        finally:
            if orig is not None:
                sys.modules["openai"] = orig
            else:
                sys.modules.pop("openai", None)

    def test_import_error(self):
        import sys
        orig = sys.modules.get("openai")
        sys.modules["openai"] = None
        try:
            with self.assertRaises(ImportError) as ctx:
                self.analyzer._call_openai("test prompt", [])
            self.assertIn("openai", str(ctx.exception).lower())
        finally:
            if orig is not None:
                sys.modules["openai"] = orig
            else:
                sys.modules.pop("openai", None)

    def test_generic_exception(self):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        mock_module.OpenAI.return_value = mock_client

        import sys
        orig = sys.modules.get("openai")
        sys.modules["openai"] = mock_module
        try:
            result = self.analyzer._call_openai("test prompt", [])
            self.assertIsNone(result)
        finally:
            if orig is not None:
                sys.modules["openai"] = orig
            else:
                sys.modules.pop("openai", None)

    def test_with_images(self):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"project_name": "ImgTest"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_module.OpenAI.return_value = mock_client

        images = [{"data": "abc123", "type": "image/png"}]

        import sys
        orig = sys.modules.get("openai")
        sys.modules["openai"] = mock_module
        try:
            result = self.analyzer._call_openai("test prompt", images)
            self.assertIsNotNone(result)
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            content = messages[0]["content"]
            image_items = [c for c in content if c["type"] == "image_url"]
            self.assertEqual(len(image_items), 1)
            self.assertIn("data:image/png;base64,abc123", image_items[0]["image_url"]["url"])
        finally:
            if orig is not None:
                sys.modules["openai"] = orig
            else:
                sys.modules.pop("openai", None)

    def test_images_limited_to_five(self):
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"project_name": "Test"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_module.OpenAI.return_value = mock_client

        images = [{"data": f"img{i}", "type": "image/png"} for i in range(10)]

        import sys
        orig = sys.modules.get("openai")
        sys.modules["openai"] = mock_module
        try:
            self.analyzer._call_openai("test prompt", images)
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            content = messages[0]["content"]
            image_items = [c for c in content if c["type"] == "image_url"]
            self.assertEqual(len(image_items), 5)
        finally:
            if orig is not None:
                sys.modules["openai"] = orig
            else:
                sys.modules.pop("openai", None)


# ==================== SpecAnalyzer.analyze (full pipeline) ====================


class TestAnalyzePipeline(unittest.TestCase):
    """Test the full analyze() pipeline."""

    def test_successful_analyze(self):
        analyzer = SpecAnalyzer(provider="anthropic")
        doc = _make_parsed_doc(text="Test specification document")
        extracted = _make_extracted_json()

        with patch.object(analyzer, "parse_document", return_value=doc) as mock_parse, \
             patch.object(analyzer, "_extract_with_llm", return_value=extracted) as mock_extract:
            result = analyzer.analyze("/tmp/test.md")

            mock_parse.assert_called_once_with("/tmp/test.md")
            mock_extract.assert_called_once_with(doc)
            self.assertEqual(result.project_name, "TestProject")
            self.assertEqual(result.raw_text, "Test specification document")

    def test_analyze_llm_fails(self):
        """When LLM extraction fails, a minimal analysis should be returned."""
        analyzer = SpecAnalyzer()
        doc = _make_parsed_doc(
            text="Document content",
            images=[{"data": "abc", "type": "image/png"}],
        )

        with patch.object(analyzer, "parse_document", return_value=doc), \
             patch.object(analyzer, "_extract_with_llm", return_value=None):
            result = analyzer.analyze("/tmp/test.md")

            self.assertEqual(result.raw_text, "Document content")
            self.assertTrue(result.has_images)
            self.assertEqual(result.image_count, 1)
            # Should be default/empty for extracted fields
            self.assertEqual(result.project_name, "")
            self.assertEqual(result.fields, [])

    def test_analyze_with_empty_extracted(self):
        """When LLM returns empty dict, still creates analysis."""
        analyzer = SpecAnalyzer()
        doc = _make_parsed_doc(text="content")

        with patch.object(analyzer, "parse_document", return_value=doc), \
             patch.object(analyzer, "_extract_with_llm", return_value={}):
            result = analyzer.analyze("/tmp/test.md")
            # Empty dict is truthy, so create_analysis_from_json is called
            self.assertEqual(result.raw_text, "content")
            self.assertEqual(result.project_name, "")


# ==================== Round-trip serialization ====================


class TestFieldDefinitionRoundTrip(unittest.TestCase):
    """Test that to_dict -> from_dict produces equivalent objects."""

    def test_simple_round_trip(self):
        fd = FieldDefinition(
            name="q", type="string", description="question", required=True,
            min_length=5, max_length=500,
        )
        d = fd.to_dict()
        fd2 = FieldDefinition.from_dict(d)
        self.assertEqual(fd.name, fd2.name)
        self.assertEqual(fd.type, fd2.type)
        self.assertEqual(fd.required, fd2.required)
        self.assertEqual(fd.min_length, fd2.min_length)
        self.assertEqual(fd.max_length, fd2.max_length)

    def test_complex_round_trip(self):
        fd = FieldDefinition(
            name="turns", type="array",
            items=FieldDefinition(
                name="turn", type="object",
                properties=[
                    FieldDefinition(name="role", type="string", enum=["user", "assistant"]),
                    FieldDefinition(name="content", type="string", required=True),
                ],
            ),
        )
        d = fd.to_dict()
        fd2 = FieldDefinition.from_dict(d)
        self.assertEqual(fd2.items.name, "turn")
        self.assertEqual(len(fd2.items.properties), 2)
        self.assertEqual(fd2.items.properties[0].enum, ["user", "assistant"])


# ==================== Edge cases ====================


class TestEdgeCases(unittest.TestCase):
    """Miscellaneous edge cases."""

    def test_parse_json_response_only_curly_braces(self):
        """Text with curly braces but not valid JSON."""
        analyzer = SpecAnalyzer()
        result = analyzer._parse_json_response("function() { return 1; }")
        self.assertIsNone(result)

    def test_create_analysis_difficulty_validation_non_dict(self):
        """difficulty_validation is a non-dict truthy value."""
        analyzer = SpecAnalyzer()
        extracted = _make_extracted_json(difficulty_validation="some string")
        analysis = analyzer.create_analysis_from_json(extracted)
        # Since it's not a dict, the branch checking isinstance should skip
        self.assertIsNone(analysis.difficulty_validation)

    def test_field_definition_items_non_dict_ignored(self):
        """Items as a non-dict should be ignored."""
        fd = FieldDefinition.from_dict({"name": "x", "items": "not_a_dict"})
        self.assertIsNone(fd.items)

    def test_field_definition_properties_non_list_ignored(self):
        fd = FieldDefinition.from_dict({"name": "x", "properties": "not_a_list"})
        self.assertIsNone(fd.properties)

    def test_field_definition_any_of_non_list_ignored(self):
        fd = FieldDefinition.from_dict({"name": "x", "any_of": "not_a_list"})
        self.assertIsNone(fd.any_of)

    def test_spec_analyzer_extraction_prompt_format(self):
        """The EXTRACTION_PROMPT template should be valid with .format()."""
        SpecAnalyzer()
        prompt = SpecAnalyzer.EXTRACTION_PROMPT.format(document_content="test content")
        self.assertIn("test content", prompt)
        self.assertIn("project_name", prompt)

    def test_object_json_schema_no_required_fields(self):
        """Object with properties but none required should not have 'required' key."""
        fd = FieldDefinition(
            name="meta", type="object",
            properties=[
                FieldDefinition(name="a", type="string", required=False),
                FieldDefinition(name="b", type="string", required=False),
            ],
        )
        schema = fd.to_json_schema()
        self.assertNotIn("required", schema)

    def test_array_without_items_schema(self):
        """Array type without items should not have 'items' in schema."""
        fd = FieldDefinition(name="tags", type="array")
        schema = fd.to_json_schema()
        self.assertEqual(schema["type"], "array")
        self.assertNotIn("items", schema)

    def test_number_no_constraints(self):
        """Number type without constraints should not have min/max."""
        fd = FieldDefinition(name="x", type="number")
        schema = fd.to_json_schema()
        self.assertEqual(schema["type"], "number")
        self.assertNotIn("minimum", schema)
        self.assertNotIn("maximum", schema)


if __name__ == "__main__":
    unittest.main()
