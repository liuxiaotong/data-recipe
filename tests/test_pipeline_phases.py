"""Tests for Composable Pipeline (Upgrade 5)."""

from datarecipe.pipeline import (
    _PHASE_REGISTRY,
    DEFAULT_PHASE_SEQUENCE,
    PhaseDefinition,
    _evaluate_condition,
    assemble_pipeline,
    get_phase,
    list_phases,
    register_phase,
)


class TestPhaseDefinition:
    def test_to_dict(self):
        phase = PhaseDefinition(
            phase_id="test",
            name="Test Phase",
            description="A test",
            default_steps=[{"action": "do_thing"}],
            depends_on=["setup"],
            condition="has_difficulty_validation",
            assignee="human",
        )
        d = phase.to_dict()
        assert d["phase_id"] == "test"
        assert d["name"] == "Test Phase"
        assert d["depends_on"] == ["setup"]
        assert d["condition"] == "has_difficulty_validation"
        assert d["assignee"] == "human"
        assert len(d["default_steps"]) == 1

    def test_defaults(self):
        phase = PhaseDefinition(phase_id="p", name="P", description="D")
        assert phase.default_steps == []
        assert phase.depends_on == []
        assert phase.condition is None
        assert phase.assignee == "human"


class TestPhaseRegistry:
    def test_builtin_phases_registered(self):
        phases = list_phases()
        ids = [p.phase_id for p in phases]
        assert "setup" in ids
        assert "pilot" in ids
        assert "model_test" in ids
        assert "human_review" in ids
        assert "production" in ids
        assert "final_qa" in ids

    def test_get_phase(self):
        phase = get_phase("setup")
        assert phase is not None
        assert phase.phase_id == "setup"
        assert phase.name == "环境准备"

    def test_get_phase_missing(self):
        assert get_phase("nonexistent_phase_xyz") is None

    def test_register_custom_phase(self):
        custom = PhaseDefinition(
            phase_id="custom_test_phase",
            name="Custom",
            description="For testing",
        )
        register_phase(custom)
        assert get_phase("custom_test_phase") is not None
        assert get_phase("custom_test_phase").name == "Custom"
        # Clean up
        del _PHASE_REGISTRY["custom_test_phase"]

    def test_builtin_phase_details(self):
        """Verify key properties of built-in phases."""
        setup = get_phase("setup")
        assert len(setup.default_steps) > 0
        assert setup.depends_on == []

        pilot = get_phase("pilot")
        assert "setup" in pilot.depends_on

        model_test = get_phase("model_test")
        assert model_test.condition == "has_difficulty_validation"
        assert "pilot" in model_test.depends_on

        human_review = get_phase("human_review")
        assert human_review.condition == "has_strategy:human_review"

        production = get_phase("production")
        assert "pilot" in production.depends_on

        final_qa = get_phase("final_qa")
        assert "production" in final_qa.depends_on


class TestDefaultPhaseSequence:
    def test_sequence_content(self):
        assert DEFAULT_PHASE_SEQUENCE == [
            "setup",
            "pilot",
            "model_test",
            "human_review",
            "production",
            "final_qa",
        ]


class _MockAnalysis:
    """Mock SpecificationAnalysis for condition evaluation."""

    def __init__(self, has_diff_val=False, strategies=None):
        self._has_diff_val = has_diff_val
        self._strategies = strategies or []

    def has_difficulty_validation(self):
        return self._has_diff_val

    def has_strategy(self, strategy_type):
        return strategy_type in self._strategies


class TestEvaluateCondition:
    def test_has_difficulty_validation_true(self):
        analysis = _MockAnalysis(has_diff_val=True)
        assert _evaluate_condition("has_difficulty_validation", analysis) is True

    def test_has_difficulty_validation_false(self):
        analysis = _MockAnalysis(has_diff_val=False)
        assert _evaluate_condition("has_difficulty_validation", analysis) is False

    def test_has_strategy_true(self):
        analysis = _MockAnalysis(strategies=["human_review"])
        assert _evaluate_condition("has_strategy:human_review", analysis) is True

    def test_has_strategy_false(self):
        analysis = _MockAnalysis(strategies=[])
        assert _evaluate_condition("has_strategy:human_review", analysis) is False

    def test_unknown_condition_defaults_true(self):
        analysis = _MockAnalysis()
        assert _evaluate_condition("some_unknown_condition", analysis) is True


class TestAssemblePipeline:
    def test_default_sequence_no_analysis(self):
        """Without analysis, all phases (even conditional ones) should be included."""
        phases = assemble_pipeline()
        ids = [p.phase_id for p in phases]
        assert "setup" in ids
        assert "pilot" in ids
        assert "production" in ids
        assert "final_qa" in ids
        # Conditional phases included since no analysis to evaluate conditions
        assert "model_test" in ids
        assert "human_review" in ids

    def test_filters_by_condition(self):
        """Conditional phases should be filtered when analysis says no."""
        analysis = _MockAnalysis(has_diff_val=False, strategies=[])
        phases = assemble_pipeline(analysis=analysis)
        ids = [p.phase_id for p in phases]
        assert "model_test" not in ids
        assert "human_review" not in ids
        assert "setup" in ids
        assert "pilot" in ids
        assert "production" in ids
        assert "final_qa" in ids

    def test_includes_conditional_when_met(self):
        """Conditional phases should be included when conditions are met."""
        analysis = _MockAnalysis(has_diff_val=True, strategies=["human_review"])
        phases = assemble_pipeline(analysis=analysis)
        ids = [p.phase_id for p in phases]
        assert "model_test" in ids
        assert "human_review" in ids

    def test_custom_phase_ids(self):
        """Passing a subset of phase_ids."""
        phases = assemble_pipeline(phase_ids=["setup", "production", "final_qa"])
        ids = [p.phase_id for p in phases]
        assert ids == ["setup", "production", "final_qa"]

    def test_missing_phase_id_skipped(self):
        """Unknown phase IDs are simply skipped."""
        phases = assemble_pipeline(phase_ids=["setup", "nonexistent", "final_qa"])
        ids = [p.phase_id for p in phases]
        assert ids == ["setup", "final_qa"]

    def test_depends_on_resolved(self):
        """depends_on should only reference present phases."""
        # Exclude pilot; model_test depends on pilot, so that dep should be removed
        analysis = _MockAnalysis(has_diff_val=True)
        phases = assemble_pipeline(
            phase_ids=["setup", "model_test", "production", "final_qa"],
            analysis=analysis,
        )
        ids = [p.phase_id for p in phases]
        assert "pilot" not in ids

        # model_test originally depends_on=["pilot"], which is not present
        model_test_phase = next(p for p in phases if p.phase_id == "model_test")
        assert "pilot" not in model_test_phase.depends_on

    def test_production_depends_on_last_validation(self):
        """Production should depend on the last validation phase before it."""
        analysis = _MockAnalysis(has_diff_val=True, strategies=["human_review"])
        phases = assemble_pipeline(analysis=analysis)

        production = next(p for p in phases if p.phase_id == "production")
        # human_review comes after model_test in sequence, so it should be the last
        assert "human_review" in production.depends_on

    def test_production_depends_on_pilot_when_no_validation(self):
        """When no validation phases, production depends on pilot."""
        analysis = _MockAnalysis(has_diff_val=False, strategies=[])
        phases = assemble_pipeline(analysis=analysis)

        production = next(p for p in phases if p.phase_id == "production")
        assert "pilot" in production.depends_on

    def test_does_not_mutate_registry(self):
        """assemble_pipeline should not mutate the registered phase definitions."""
        original_deps = list(get_phase("production").depends_on)
        analysis = _MockAnalysis(has_diff_val=True, strategies=["human_review"])
        assemble_pipeline(analysis=analysis)
        assert get_phase("production").depends_on == original_deps
