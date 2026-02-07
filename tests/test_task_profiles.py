"""Tests for TaskTypeProfile (Upgrade 7)."""

import pytest
from datarecipe.task_profiles import (
    TaskTypeProfile,
    get_task_profile,
    list_task_profiles,
    register_task_profile,
    _PROFILE_REGISTRY,
)


class TestTaskTypeProfile:
    def test_builtin_profiles_registered(self):
        profiles = list_task_profiles()
        ids = [p.type_id for p in profiles]
        assert "preference" in ids
        assert "evaluation" in ids
        assert "sft" in ids
        assert "swe_bench" in ids
        assert "unknown" in ids

    def test_get_known_profile(self):
        p = get_task_profile("preference")
        assert p.type_id == "preference"
        assert p.name == "偏好对比数据"
        assert len(p.cognitive_requirements) > 0
        assert len(p.default_quality_constraints) > 0
        assert len(p.default_fields) > 0

    def test_get_unknown_falls_back(self):
        p = get_task_profile("nonexistent_type_xyz")
        assert p.type_id == "unknown"
        assert p.name == "通用数据标注"

    def test_evaluation_profile(self):
        p = get_task_profile("evaluation")
        assert p.cost_multiplier == 1.5
        assert p.default_human_percentage == 95.0
        assert p.preferred_pipeline == "benchmark"
        assert len(p.default_scoring_dimensions) > 0

    def test_sft_profile(self):
        p = get_task_profile("sft")
        assert p.cost_multiplier == 0.8
        assert p.default_human_percentage == 60.0

    def test_swe_bench_profile(self):
        p = get_task_profile("swe_bench")
        assert p.cost_multiplier == 2.0
        assert p.preferred_pipeline == "programmatic"

    def test_register_custom_profile(self):
        custom = TaskTypeProfile(
            type_id="custom_test_profile",
            name="Test Profile",
            description="For testing",
            cognitive_requirements=["test"],
        )
        register_task_profile(custom)
        assert get_task_profile("custom_test_profile").name == "Test Profile"
        # Clean up
        del _PROFILE_REGISTRY["custom_test_profile"]

    def test_to_dict(self):
        p = get_task_profile("preference")
        d = p.to_dict()
        assert d["type_id"] == "preference"
        assert isinstance(d["default_fields"], list)
        assert isinstance(d["default_quality_constraints"], list)
        assert isinstance(d["cost_multiplier"], float)

    def test_default_fields_have_structure(self):
        """Ensure default_fields have at least name and type."""
        for p in list_task_profiles():
            for f in p.default_fields:
                assert "name" in f, f"Missing 'name' in field of {p.type_id}"
                assert "type" in f, f"Missing 'type' in field of {p.type_id}"

    def test_evaluation_has_nested_fields(self):
        """evaluation profile should have nested object fields."""
        p = get_task_profile("evaluation")
        answer_field = None
        for f in p.default_fields:
            if f.get("name") == "answer":
                answer_field = f
                break
        assert answer_field is not None
        assert answer_field["type"] == "object"
        assert "properties" in answer_field
