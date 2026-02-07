"""Tests for FieldDefinition, FieldConstraint, and ValidationStrategy (Upgrades 1, 3, 6)."""

from datarecipe.analyzers.spec_analyzer import (
    FieldConstraint,
    FieldDefinition,
    SpecificationAnalysis,
    ValidationStrategy,
    _map_type,
)

# ---- FieldDefinition ----


class TestFieldDefinition:
    """Tests for FieldDefinition dataclass."""

    def test_from_dict_simple_string(self):
        fd = FieldDefinition.from_dict(
            {"name": "question", "type": "string", "description": "题目"}
        )
        assert fd.name == "question"
        assert fd.type == "string"
        assert fd.description == "题目"
        assert fd.required is False

    def test_from_dict_required_bool(self):
        fd = FieldDefinition.from_dict({"name": "x", "type": "string", "required": True})
        assert fd.required is True

    def test_from_dict_required_string_true(self):
        fd = FieldDefinition.from_dict({"name": "x", "type": "string", "required": "true"})
        assert fd.required is True

    def test_from_dict_required_string_false(self):
        fd = FieldDefinition.from_dict({"name": "x", "type": "string", "required": "false"})
        assert fd.required is False

    def test_from_dict_with_enum(self):
        fd = FieldDefinition.from_dict(
            {"name": "role", "type": "string", "enum": ["user", "assistant"]}
        )
        assert fd.enum == ["user", "assistant"]

    def test_from_dict_with_constraints(self):
        fd = FieldDefinition.from_dict(
            {
                "name": "text",
                "type": "string",
                "min_length": 10,
                "max_length": 500,
                "pattern": "^[A-Z]",
            }
        )
        assert fd.min_length == 10
        assert fd.max_length == 500
        assert fd.pattern == "^[A-Z]"

    def test_from_dict_with_camelCase_constraints(self):
        """Backwards compat: accept minLength/maxLength."""
        fd = FieldDefinition.from_dict(
            {
                "name": "text",
                "type": "string",
                "minLength": 10,
                "maxLength": 500,
            }
        )
        assert fd.min_length == 10
        assert fd.max_length == 500

    def test_from_dict_nested_array(self):
        fd = FieldDefinition.from_dict(
            {
                "name": "messages",
                "type": "array",
                "items": {
                    "name": "msg",
                    "type": "object",
                    "properties": [
                        {"name": "role", "type": "string", "enum": ["user", "assistant"]},
                        {"name": "content", "type": "string"},
                    ],
                },
            }
        )
        assert fd.type == "array"
        assert fd.items is not None
        assert fd.items.type == "object"
        assert len(fd.items.properties) == 2
        assert fd.items.properties[0].enum == ["user", "assistant"]

    def test_from_dict_any_of(self):
        fd = FieldDefinition.from_dict(
            {
                "name": "answer",
                "type": "string",
                "any_of": [
                    {"name": "a", "type": "string"},
                    {"name": "a", "type": "integer"},
                ],
            }
        )
        assert fd.any_of is not None
        assert len(fd.any_of) == 2

    def test_round_trip(self):
        """from_dict → to_dict round trip preserves data."""
        original = {
            "name": "conversations",
            "type": "array",
            "required": True,
            "description": "对话历史",
            "items": {
                "name": "turn",
                "type": "object",
                "properties": [
                    {"name": "role", "type": "string", "enum": ["user", "assistant"]},
                    {"name": "content", "type": "string", "min_length": 1},
                ],
            },
        }
        fd = FieldDefinition.from_dict(original)
        restored = fd.to_dict()
        fd2 = FieldDefinition.from_dict(restored)
        assert fd2.name == fd.name
        assert fd2.items.properties[0].enum == ["user", "assistant"]
        assert fd2.items.properties[1].min_length == 1

    def test_to_json_schema_string(self):
        fd = FieldDefinition(name="q", type="string", description="题目", min_length=10)
        schema = fd.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "题目"
        assert schema["minLength"] == 10

    def test_to_json_schema_enum(self):
        fd = FieldDefinition(name="role", type="string", enum=["user", "assistant"])
        schema = fd.to_json_schema()
        assert schema["enum"] == ["user", "assistant"]

    def test_to_json_schema_nested_object(self):
        fd = FieldDefinition(
            name="answer",
            type="object",
            properties=[
                FieldDefinition(name="value", type="string", required=True),
                FieldDefinition(name="score", type="number", minimum=0, maximum=1),
            ],
        )
        schema = fd.to_json_schema()
        assert schema["type"] == "object"
        assert "value" in schema["properties"]
        assert schema["required"] == ["value"]
        assert schema["properties"]["score"]["minimum"] == 0

    def test_to_json_schema_array(self):
        fd = FieldDefinition(
            name="tags",
            type="array",
            items=FieldDefinition(name="tag", type="string"),
        )
        schema = fd.to_json_schema()
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_to_json_schema_any_of(self):
        fd = FieldDefinition(
            name="result",
            type="string",
            any_of=[
                FieldDefinition(name="r", type="string"),
                FieldDefinition(name="r", type="integer"),
            ],
        )
        schema = fd.to_json_schema()
        assert "anyOf" in schema
        assert "type" not in schema
        assert len(schema["anyOf"]) == 2

    def test_old_flat_format_compat(self):
        """Old format {"name": "x", "type": "string"} works fine."""
        fd = FieldDefinition.from_dict({"name": "x", "type": "string"})
        assert fd.name == "x"
        assert fd.type == "string"
        assert fd.items is None
        assert fd.properties is None
        schema = fd.to_json_schema()
        assert schema == {"type": "string"}


class TestMapType:
    def test_known_types(self):
        assert _map_type("string") == "string"
        assert _map_type("text") == "string"
        assert _map_type("integer") == "integer"
        assert _map_type("int") == "integer"
        assert _map_type("boolean") == "boolean"
        assert _map_type("array") == "array"
        assert _map_type("list") == "array"
        assert _map_type("object") == "object"
        assert _map_type("dict") == "object"
        assert _map_type("number") == "number"
        assert _map_type("float") == "number"

    def test_unknown_defaults_to_string(self):
        assert _map_type("unknown_type") == "string"

    def test_case_insensitive(self):
        assert _map_type("String") == "string"
        assert _map_type("BOOLEAN") == "boolean"


# ---- FieldConstraint ----


class TestFieldConstraint:
    def test_from_dict(self):
        fc = FieldConstraint.from_dict(
            {
                "field_name": "answer",
                "constraint_type": "format",
                "rule": "必须是有效JSON",
                "severity": "error",
                "auto_checkable": True,
            }
        )
        assert fc.field_name == "answer"
        assert fc.auto_checkable is True

    def test_round_trip(self):
        fc = FieldConstraint(
            field_name="q",
            constraint_type="range",
            rule="len > 10",
            severity="warning",
            auto_checkable=True,
        )
        d = fc.to_dict()
        fc2 = FieldConstraint.from_dict(d)
        assert fc2.field_name == fc.field_name
        assert fc2.auto_checkable is True


class TestSpecificationAnalysisParsedConstraints:
    def test_merges_new_and_legacy(self):
        analysis = SpecificationAnalysis(
            field_constraints=[
                {
                    "field_name": "q",
                    "constraint_type": "format",
                    "rule": "new rule",
                    "severity": "error",
                }
            ],
            field_requirements={"q": "legacy requirement"},
            quality_constraints=["global constraint"],
        )
        constraints = analysis.parsed_constraints
        # Should have 3: new field_constraint, legacy field_requirement, legacy quality_constraint
        assert len(constraints) == 3
        field_names = [c.field_name for c in constraints]
        assert "q" in field_names
        assert "_global" in field_names

    def test_constraints_for_field(self):
        analysis = SpecificationAnalysis(
            field_constraints=[
                {"field_name": "q", "constraint_type": "format", "rule": "r1"},
                {"field_name": "a", "constraint_type": "content", "rule": "r2"},
            ],
            quality_constraints=["global rule"],
        )
        q_constraints = analysis.constraints_for_field("q")
        # Should get: q's constraint + _global constraint
        assert len(q_constraints) == 2


# ---- ValidationStrategy ----


class TestValidationStrategy:
    def test_from_dict(self):
        vs = ValidationStrategy.from_dict(
            {
                "strategy_type": "human_review",
                "enabled": True,
                "config": {"sample_rate": 0.2},
                "description": "人工审核",
            }
        )
        assert vs.strategy_type == "human_review"
        assert vs.config["sample_rate"] == 0.2

    def test_from_difficulty_validation(self):
        diff_val = {
            "model": "doubao1.8",
            "settings": "高思考深度",
            "test_count": 3,
            "max_correct": 1,
        }
        vs = ValidationStrategy.from_difficulty_validation(diff_val)
        assert vs.strategy_type == "model_test"
        assert vs.config["model"] == "doubao1.8"
        assert vs.config["test_count"] == 3

    def test_round_trip(self):
        vs = ValidationStrategy(
            strategy_type="format_check",
            enabled=True,
            config={"strict": True},
            description="格式校验",
        )
        d = vs.to_dict()
        vs2 = ValidationStrategy.from_dict(d)
        assert vs2.strategy_type == vs.strategy_type
        assert vs2.config == vs.config


class TestSpecificationAnalysisValidationStrategies:
    def test_parsed_validation_strategies_from_new_format(self):
        analysis = SpecificationAnalysis(
            validation_strategies=[
                {
                    "strategy_type": "human_review",
                    "enabled": True,
                    "config": {},
                    "description": "审核",
                },
                {
                    "strategy_type": "format_check",
                    "enabled": True,
                    "config": {},
                    "description": "格式",
                },
            ],
        )
        strategies = analysis.parsed_validation_strategies
        assert len(strategies) == 2
        assert strategies[0].strategy_type == "human_review"

    def test_parsed_validation_strategies_from_legacy(self):
        analysis = SpecificationAnalysis(
            difficulty_validation={"model": "gpt-4", "test_count": 3, "max_correct": 1},
        )
        strategies = analysis.parsed_validation_strategies
        assert len(strategies) == 1
        assert strategies[0].strategy_type == "model_test"

    def test_no_duplicate_model_test(self):
        analysis = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "model_test", "enabled": True, "config": {"model": "gpt-4"}},
            ],
            difficulty_validation={"model": "gpt-4", "test_count": 3, "max_correct": 1},
        )
        strategies = analysis.parsed_validation_strategies
        # Should not duplicate model_test
        assert sum(1 for s in strategies if s.strategy_type == "model_test") == 1

    def test_has_strategy(self):
        analysis = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "human_review", "enabled": True, "config": {}},
            ],
        )
        assert analysis.has_strategy("human_review") is True
        assert analysis.has_strategy("model_test") is False

    def test_get_strategy(self):
        analysis = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "format_check", "enabled": True, "config": {"strict": True}},
            ],
        )
        s = analysis.get_strategy("format_check")
        assert s is not None
        assert s.config["strict"] is True

    def test_has_difficulty_validation_from_strategy(self):
        """has_difficulty_validation() should also check new validation_strategies."""
        analysis = SpecificationAnalysis(
            validation_strategies=[
                {"strategy_type": "model_test", "enabled": True, "config": {"model": "gpt-4"}},
            ],
        )
        assert analysis.has_difficulty_validation() is True


class TestFieldDefinitionsProperty:
    def test_field_definitions_from_fields(self):
        analysis = SpecificationAnalysis(
            fields=[
                {"name": "q", "type": "string", "required": True},
                {"name": "a", "type": "string"},
            ]
        )
        fds = analysis.field_definitions
        assert len(fds) == 2
        assert fds[0].name == "q"
        assert fds[0].required is True

    def test_field_definitions_nested(self):
        analysis = SpecificationAnalysis(
            fields=[
                {
                    "name": "messages",
                    "type": "array",
                    "items": {
                        "name": "turn",
                        "type": "object",
                        "properties": [
                            {"name": "role", "type": "string"},
                            {"name": "content", "type": "string"},
                        ],
                    },
                }
            ]
        )
        fds = analysis.field_definitions
        assert len(fds) == 1
        assert fds[0].items is not None
        assert len(fds[0].items.properties) == 2

    def test_field_definitions_empty(self):
        analysis = SpecificationAnalysis()
        assert analysis.field_definitions == []


class TestToDict:
    def test_to_dict_includes_new_fields(self):
        analysis = SpecificationAnalysis(
            fields=[{"name": "q", "type": "string"}],
            field_constraints=[{"field_name": "q", "rule": "r"}],
            validation_strategies=[{"strategy_type": "human_review", "enabled": True}],
            quality_gates=[
                {"gate_id": "g1", "metric": "overall_score", "operator": ">=", "threshold": 60}
            ],
        )
        d = analysis.to_dict()
        assert "field_constraints" in d
        assert "validation_strategies" in d
        assert "quality_gates" in d
        assert "field_definitions" in d
        assert len(d["field_definitions"]) == 1
