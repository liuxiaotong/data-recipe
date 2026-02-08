"""Comprehensive tests for datarecipe.pipeline module.

Covers:
- PipelineStepType enum
- PipelineStep dataclass
- ProductionPipeline dataclass
- PIPELINE_TEMPLATES dictionary
- get_pipeline_template() function (lines 808-842)
- pipeline_to_markdown() function (lines 847-945)
- Composable pipeline edge cases (line 1171)
"""

import pytest

from datarecipe.pipeline import (
    PIPELINE_TEMPLATES,
    PipelineStep,
    PipelineStepType,
    ProductionPipeline,
    assemble_pipeline,
    get_pipeline_template,
    pipeline_to_markdown,
)

# =============================================================================
# PipelineStepType enum
# =============================================================================


class TestPipelineStepType:
    """Test PipelineStepType enum values."""

    def test_all_expected_values_exist(self):
        expected = [
            "data_collection",
            "seed_data",
            "prompt_design",
            "llm_generation",
            "human_annotation",
            "quality_filter",
            "deduplication",
            "format_conversion",
            "validation",
            "post_processing",
        ]
        actual = [e.value for e in PipelineStepType]
        assert sorted(actual) == sorted(expected)

    def test_enum_count(self):
        assert len(PipelineStepType) == 10

    def test_enum_access_by_name(self):
        assert PipelineStepType.DATA_COLLECTION.value == "data_collection"
        assert PipelineStepType.SEED_DATA.value == "seed_data"
        assert PipelineStepType.PROMPT_DESIGN.value == "prompt_design"
        assert PipelineStepType.LLM_GENERATION.value == "llm_generation"
        assert PipelineStepType.HUMAN_ANNOTATION.value == "human_annotation"
        assert PipelineStepType.QUALITY_FILTER.value == "quality_filter"
        assert PipelineStepType.DEDUPLICATION.value == "deduplication"
        assert PipelineStepType.FORMAT_CONVERSION.value == "format_conversion"
        assert PipelineStepType.VALIDATION.value == "validation"
        assert PipelineStepType.POST_PROCESSING.value == "post_processing"


# =============================================================================
# PipelineStep dataclass
# =============================================================================


class TestPipelineStep:
    """Test PipelineStep dataclass."""

    def test_required_fields(self):
        step = PipelineStep(
            step_number=1,
            step_type=PipelineStepType.DATA_COLLECTION,
            name="Test Step",
            description="A test step",
        )
        assert step.step_number == 1
        assert step.step_type == PipelineStepType.DATA_COLLECTION
        assert step.name == "Test Step"
        assert step.description == "A test step"

    def test_default_values(self):
        step = PipelineStep(
            step_number=1,
            step_type=PipelineStepType.SEED_DATA,
            name="Step",
            description="Desc",
        )
        assert step.inputs == []
        assert step.outputs == []
        assert step.tools == []
        assert step.estimated_cost is None
        assert step.estimated_time is None
        assert step.code_snippet is None
        assert step.tips == []

    def test_all_fields(self):
        step = PipelineStep(
            step_number=3,
            step_type=PipelineStepType.LLM_GENERATION,
            name="Generate",
            description="Generate data",
            inputs=["seeds.jsonl"],
            outputs=["raw.jsonl"],
            tools=["OpenAI API"],
            estimated_cost=0.05,
            estimated_time="2 hours",
            code_snippet="print('hello')",
            tips=["Tip 1", "Tip 2"],
        )
        assert step.inputs == ["seeds.jsonl"]
        assert step.outputs == ["raw.jsonl"]
        assert step.tools == ["OpenAI API"]
        assert step.estimated_cost == 0.05
        assert step.estimated_time == "2 hours"
        assert step.code_snippet == "print('hello')"
        assert step.tips == ["Tip 1", "Tip 2"]

    def test_default_lists_are_independent(self):
        """Verify default factory creates separate list instances."""
        step1 = PipelineStep(
            step_number=1, step_type=PipelineStepType.SEED_DATA, name="S1", description="D1"
        )
        step2 = PipelineStep(
            step_number=2, step_type=PipelineStepType.SEED_DATA, name="S2", description="D2"
        )
        step1.inputs.append("a.txt")
        assert step2.inputs == []


# =============================================================================
# ProductionPipeline dataclass
# =============================================================================


class TestProductionPipeline:
    """Test ProductionPipeline dataclass."""

    def test_required_fields(self):
        pipeline = ProductionPipeline(name="Test", description="A test pipeline")
        assert pipeline.name == "Test"
        assert pipeline.description == "A test pipeline"

    def test_default_values(self):
        pipeline = ProductionPipeline(name="Test", description="Desc")
        assert pipeline.target_size is None
        assert pipeline.estimated_total_cost is None
        assert pipeline.estimated_total_time is None
        assert pipeline.prerequisites == []
        assert pipeline.steps == []
        assert pipeline.quality_criteria == []
        assert pipeline.common_pitfalls == []

    def test_full_construction(self):
        step = PipelineStep(
            step_number=1,
            step_type=PipelineStepType.SEED_DATA,
            name="Step1",
            description="First step",
        )
        pipeline = ProductionPipeline(
            name="Full Pipeline",
            description="All fields set",
            target_size=10000,
            estimated_total_cost=5000.0,
            estimated_total_time="2 weeks",
            prerequisites=["API key", "Storage"],
            steps=[step],
            quality_criteria=["Accuracy > 90%"],
            common_pitfalls=["Data leakage"],
        )
        assert pipeline.target_size == 10000
        assert pipeline.estimated_total_cost == 5000.0
        assert pipeline.estimated_total_time == "2 weeks"
        assert len(pipeline.prerequisites) == 2
        assert len(pipeline.steps) == 1
        assert len(pipeline.quality_criteria) == 1
        assert len(pipeline.common_pitfalls) == 1


# =============================================================================
# PIPELINE_TEMPLATES
# =============================================================================


class TestPipelineTemplates:
    """Test pre-defined pipeline templates."""

    def test_all_expected_templates_exist(self):
        expected_keys = {"distillation", "human_annotation", "hybrid", "programmatic", "simulation", "benchmark"}
        assert set(PIPELINE_TEMPLATES.keys()) == expected_keys

    def test_all_templates_are_production_pipelines(self):
        for key, template in PIPELINE_TEMPLATES.items():
            assert isinstance(template, ProductionPipeline), f"{key} is not a ProductionPipeline"

    def test_all_templates_have_steps(self):
        for key, template in PIPELINE_TEMPLATES.items():
            assert len(template.steps) > 0, f"{key} has no steps"

    def test_all_templates_have_quality_criteria(self):
        for key, template in PIPELINE_TEMPLATES.items():
            assert len(template.quality_criteria) > 0, f"{key} has no quality criteria"

    def test_all_templates_have_common_pitfalls(self):
        for key, template in PIPELINE_TEMPLATES.items():
            assert len(template.common_pitfalls) > 0, f"{key} has no common pitfalls"

    def test_distillation_template_details(self):
        t = PIPELINE_TEMPLATES["distillation"]
        assert len(t.steps) == 6
        assert len(t.prerequisites) == 4
        step_types = [s.step_type for s in t.steps]
        assert PipelineStepType.SEED_DATA in step_types
        assert PipelineStepType.LLM_GENERATION in step_types
        assert PipelineStepType.QUALITY_FILTER in step_types

    def test_human_annotation_template_details(self):
        t = PIPELINE_TEMPLATES["human_annotation"]
        assert len(t.steps) == 5
        step_types = [s.step_type for s in t.steps]
        assert PipelineStepType.HUMAN_ANNOTATION in step_types

    def test_hybrid_template_details(self):
        t = PIPELINE_TEMPLATES["hybrid"]
        assert len(t.steps) == 5
        step_types = [s.step_type for s in t.steps]
        assert PipelineStepType.LLM_GENERATION in step_types
        assert PipelineStepType.HUMAN_ANNOTATION in step_types

    def test_programmatic_template_has_six_steps(self):
        t = PIPELINE_TEMPLATES["programmatic"]
        assert len(t.steps) == 6

    def test_simulation_template_has_six_steps(self):
        t = PIPELINE_TEMPLATES["simulation"]
        assert len(t.steps) == 6

    def test_benchmark_template_has_five_steps(self):
        t = PIPELINE_TEMPLATES["benchmark"]
        assert len(t.steps) == 5

    def test_step_numbers_are_sequential(self):
        """All templates should have steps numbered 1, 2, 3, ..."""
        for key, template in PIPELINE_TEMPLATES.items():
            numbers = [s.step_number for s in template.steps]
            assert numbers == list(range(1, len(template.steps) + 1)), (
                f"{key} has non-sequential step numbers: {numbers}"
            )


# =============================================================================
# get_pipeline_template() — lines 808-842
# =============================================================================


class TestGetPipelineTemplate:
    """Test get_pipeline_template() function with all routing branches."""

    # --- Category-based routing (lines 808-819) ---

    def test_category_programmatic(self):
        result = get_pipeline_template("anything", category="programmatic")
        assert result is PIPELINE_TEMPLATES["programmatic"]

    def test_category_programmatic_generation(self):
        result = get_pipeline_template("anything", category="programmatic_generation")
        assert result is PIPELINE_TEMPLATES["programmatic"]

    def test_category_simulation(self):
        result = get_pipeline_template("anything", category="simulation")
        assert result is PIPELINE_TEMPLATES["simulation"]

    def test_category_simulator(self):
        result = get_pipeline_template("anything", category="simulator")
        assert result is PIPELINE_TEMPLATES["simulation"]

    def test_category_benchmark(self):
        result = get_pipeline_template("anything", category="benchmark")
        assert result is PIPELINE_TEMPLATES["benchmark"]

    def test_category_evaluation(self):
        result = get_pipeline_template("anything", category="evaluation")
        assert result is PIPELINE_TEMPLATES["benchmark"]

    def test_category_llm_distillation(self):
        result = get_pipeline_template("anything", category="llm_distillation")
        assert result is PIPELINE_TEMPLATES["distillation"]

    def test_category_distillation(self):
        result = get_pipeline_template("anything", category="distillation")
        assert result is PIPELINE_TEMPLATES["distillation"]

    def test_category_human_annotation(self):
        result = get_pipeline_template("anything", category="human_annotation")
        assert result is PIPELINE_TEMPLATES["human_annotation"]

    def test_category_human(self):
        result = get_pipeline_template("anything", category="human")
        assert result is PIPELINE_TEMPLATES["human_annotation"]

    def test_category_case_insensitive(self):
        """Category matching should be case-insensitive."""
        result = get_pipeline_template("anything", category="Programmatic")
        assert result is PIPELINE_TEMPLATES["programmatic"]

    def test_category_case_insensitive_mixed(self):
        result = get_pipeline_template("anything", category="BENCHMARK")
        assert result is PIPELINE_TEMPLATES["benchmark"]

    def test_category_takes_priority_over_synthetic_ratio(self):
        """Category should override synthetic_ratio."""
        result = get_pipeline_template("synthetic", synthetic_ratio=1.0, category="benchmark")
        assert result is PIPELINE_TEMPLATES["benchmark"]

    def test_category_takes_priority_over_generation_type(self):
        """Category should override generation_type."""
        result = get_pipeline_template("human", category="programmatic")
        assert result is PIPELINE_TEMPLATES["programmatic"]

    def test_unknown_category_falls_through_to_synthetic_ratio(self):
        """Unknown category should fall through to synthetic_ratio logic."""
        result = get_pipeline_template("whatever", synthetic_ratio=0.95, category="unknown_category")
        assert result is PIPELINE_TEMPLATES["distillation"]

    # --- Synthetic ratio fallback (lines 822-828) ---

    def test_synthetic_ratio_high_returns_distillation(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.95)
        assert result is PIPELINE_TEMPLATES["distillation"]

    def test_synthetic_ratio_exactly_0_9_returns_distillation(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.9)
        assert result is PIPELINE_TEMPLATES["distillation"]

    def test_synthetic_ratio_low_returns_human_annotation(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.05)
        assert result is PIPELINE_TEMPLATES["human_annotation"]

    def test_synthetic_ratio_exactly_0_1_returns_human_annotation(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.1)
        assert result is PIPELINE_TEMPLATES["human_annotation"]

    def test_synthetic_ratio_middle_returns_hybrid(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.5)
        assert result is PIPELINE_TEMPLATES["hybrid"]

    def test_synthetic_ratio_just_above_0_1_returns_hybrid(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.11)
        assert result is PIPELINE_TEMPLATES["hybrid"]

    def test_synthetic_ratio_just_below_0_9_returns_hybrid(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.89)
        assert result is PIPELINE_TEMPLATES["hybrid"]

    def test_synthetic_ratio_zero_returns_human_annotation(self):
        result = get_pipeline_template("anything", synthetic_ratio=0.0)
        assert result is PIPELINE_TEMPLATES["human_annotation"]

    def test_synthetic_ratio_one_returns_distillation(self):
        result = get_pipeline_template("anything", synthetic_ratio=1.0)
        assert result is PIPELINE_TEMPLATES["distillation"]

    # --- Generation type fallback (lines 831-842) ---

    def test_generation_type_synthetic(self):
        result = get_pipeline_template("synthetic")
        assert result is PIPELINE_TEMPLATES["distillation"]

    def test_generation_type_distillation(self):
        result = get_pipeline_template("distillation")
        assert result is PIPELINE_TEMPLATES["distillation"]

    def test_generation_type_human(self):
        result = get_pipeline_template("human")
        assert result is PIPELINE_TEMPLATES["human_annotation"]

    def test_generation_type_human_annotation(self):
        result = get_pipeline_template("human_annotation")
        assert result is PIPELINE_TEMPLATES["human_annotation"]

    def test_generation_type_programmatic(self):
        result = get_pipeline_template("programmatic")
        assert result is PIPELINE_TEMPLATES["programmatic"]

    def test_generation_type_simulation(self):
        result = get_pipeline_template("simulation")
        assert result is PIPELINE_TEMPLATES["simulation"]

    def test_generation_type_benchmark(self):
        result = get_pipeline_template("benchmark")
        assert result is PIPELINE_TEMPLATES["benchmark"]

    def test_generation_type_unknown_returns_hybrid(self):
        """Any unrecognized generation type falls back to hybrid."""
        result = get_pipeline_template("something_unknown")
        assert result is PIPELINE_TEMPLATES["hybrid"]

    def test_generation_type_empty_string_returns_hybrid(self):
        result = get_pipeline_template("")
        assert result is PIPELINE_TEMPLATES["hybrid"]

    def test_no_category_no_ratio_uses_generation_type(self):
        result = get_pipeline_template("benchmark", synthetic_ratio=None, category=None)
        assert result is PIPELINE_TEMPLATES["benchmark"]


# =============================================================================
# pipeline_to_markdown() — lines 847-945
# =============================================================================


class TestPipelineToMarkdown:
    """Test pipeline_to_markdown() function."""

    @pytest.fixture()
    def simple_pipeline(self):
        """A minimal pipeline for testing markdown output."""
        return ProductionPipeline(
            name="Test Pipeline",
            description="A test pipeline for unit tests",
            prerequisites=["Python 3.10+", "API key"],
            steps=[
                PipelineStep(
                    step_number=1,
                    step_type=PipelineStepType.SEED_DATA,
                    name="Prepare Seeds",
                    description="Collect seed data",
                    inputs=["raw_data.csv"],
                    outputs=["seeds.jsonl"],
                    tips=["Quality matters", "Diversity helps"],
                ),
                PipelineStep(
                    step_number=2,
                    step_type=PipelineStepType.LLM_GENERATION,
                    name="Generate Data",
                    description="Use LLM to generate",
                    inputs=["seeds.jsonl"],
                    outputs=["generated.jsonl"],
                    tools=["OpenAI API", "vLLM"],
                    estimated_cost=0.02,
                    code_snippet='print("hello")',
                ),
            ],
            quality_criteria=["Accuracy > 90%", "No duplicates"],
            common_pitfalls=["Bad seeds", "Wrong temperature"],
        )

    @pytest.fixture()
    def full_pipeline(self):
        """A pipeline with all optional fields populated."""
        return ProductionPipeline(
            name="Full Pipeline",
            description="Pipeline with all fields",
            target_size=10000,
            estimated_total_cost=5000.0,
            estimated_total_time="3 weeks",
            prerequisites=["Prereq 1"],
            steps=[
                PipelineStep(
                    step_number=1,
                    step_type=PipelineStepType.DATA_COLLECTION,
                    name="Collect",
                    description="Collect data",
                    inputs=["source"],
                    outputs=["data.jsonl"],
                    tools=["scrapy"],
                    estimated_cost=0.01,
                    code_snippet="import scrapy",
                    tips=["Be polite"],
                ),
            ],
            quality_criteria=["Complete"],
            common_pitfalls=["Timeout"],
        )

    def test_title_with_dataset_name(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline, dataset_name="my-dataset")
        assert md.startswith("# 数据生产指南：my-dataset")

    def test_title_without_dataset_name(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "# 数据生产指南：Test Pipeline" in md

    def test_overview_section(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "## 概述" in md
        assert "> A test pipeline for unit tests" in md

    def test_estimated_total_cost(self, full_pipeline):
        md = pipeline_to_markdown(full_pipeline)
        assert "**预估总成本**: $5,000" in md

    def test_estimated_total_time(self, full_pipeline):
        md = pipeline_to_markdown(full_pipeline)
        assert "**预估时间**: 3 weeks" in md

    def test_target_size(self, full_pipeline):
        md = pipeline_to_markdown(full_pipeline)
        assert "**目标数据量**: 10,000 条" in md

    def test_no_cost_time_target_when_none(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "预估总成本" not in md
        assert "预估时间" not in md
        assert "目标数据量" not in md

    def test_prerequisites_section(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "## 前置准备" in md
        assert "- [ ] Python 3.10+" in md
        assert "- [ ] API key" in md

    def test_flow_diagram(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "### 流程图" in md
        assert "[1. Prepare Seeds]" in md
        assert "[2. Generate Data]" in md
        # Steps connected by arrow
        assert "[1. Prepare Seeds] → [2. Generate Data]" in md

    def test_detailed_steps_section(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "### 详细步骤" in md
        assert "#### 步骤 1: Prepare Seeds" in md
        assert "#### 步骤 2: Generate Data" in md

    def test_step_description(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "**描述**: Collect seed data" in md
        assert "**描述**: Use LLM to generate" in md

    def test_step_inputs_outputs(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "**输入**: raw_data.csv" in md
        assert "**输出**: seeds.jsonl" in md

    def test_step_tools(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "**工具**: OpenAI API, vLLM" in md

    def test_step_cost(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "**成本**: $0.02 per item" in md

    def test_step_code_snippet(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "**代码示例**:" in md
        assert "```python" in md
        assert 'print("hello")' in md
        assert "```" in md

    def test_step_tips(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "**提示**:" in md
        assert "- Quality matters" in md
        assert "- Diversity helps" in md

    def test_step_separator(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "---" in md

    def test_quality_criteria_section(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "## 质量标准" in md
        assert "- [ ] Accuracy > 90%" in md
        assert "- [ ] No duplicates" in md

    def test_common_pitfalls_section(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "## 常见问题与避坑指南" in md
        assert "1. " in md
        assert "Bad seeds" in md
        assert "2. " in md
        assert "Wrong temperature" in md

    def test_footer(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert "> 由 DataRecipe 生成 — 数据生产指南" in md

    def test_output_is_string(self, simple_pipeline):
        md = pipeline_to_markdown(simple_pipeline)
        assert isinstance(md, str)

    def test_no_tools_field_when_empty(self):
        """Steps without tools should not have a tools line."""
        pipeline = ProductionPipeline(
            name="No Tools",
            description="Desc",
            steps=[
                PipelineStep(
                    step_number=1,
                    step_type=PipelineStepType.SEED_DATA,
                    name="Step",
                    description="Desc",
                    # no tools, no cost, no code_snippet, no tips
                ),
            ],
            quality_criteria=["OK"],
            common_pitfalls=["None"],
        )
        md = pipeline_to_markdown(pipeline)
        assert "**工具**" not in md
        assert "**成本**" not in md
        assert "**代码示例**" not in md
        assert "**提示**:" not in md

    def test_no_inputs_outputs_when_empty(self):
        """Steps without inputs/outputs should not have those lines."""
        pipeline = ProductionPipeline(
            name="Empty IO",
            description="Desc",
            steps=[
                PipelineStep(
                    step_number=1,
                    step_type=PipelineStepType.VALIDATION,
                    name="Validate",
                    description="Check",
                ),
            ],
            quality_criteria=["Done"],
            common_pitfalls=["Skip"],
        )
        md = pipeline_to_markdown(pipeline)
        assert "**输入**" not in md
        assert "**输出**" not in md

    def test_empty_pipeline_no_steps(self):
        """A pipeline with no steps should still generate valid markdown."""
        pipeline = ProductionPipeline(
            name="Empty",
            description="No steps",
            prerequisites=[],
            steps=[],
            quality_criteria=[],
            common_pitfalls=[],
        )
        md = pipeline_to_markdown(pipeline)
        assert "# 数据生产指南：Empty" in md
        assert "## 概述" in md
        assert "## 生产流程" in md
        assert "> 由 DataRecipe 生成" in md


class TestPipelineToMarkdownWithTemplates:
    """Test pipeline_to_markdown with actual PIPELINE_TEMPLATES."""

    @pytest.mark.parametrize("template_key", list(PIPELINE_TEMPLATES.keys()))
    def test_template_renders_to_valid_markdown(self, template_key):
        """Each template should render without errors."""
        template = PIPELINE_TEMPLATES[template_key]
        md = pipeline_to_markdown(template, dataset_name=f"test-{template_key}")
        assert isinstance(md, str)
        assert len(md) > 100
        assert f"# 数据生产指南：test-{template_key}" in md
        assert "## 概述" in md
        assert "## 前置准备" in md
        assert "## 生产流程" in md
        assert "## 质量标准" in md
        assert "## 常见问题与避坑指南" in md

    def test_distillation_template_markdown_has_code_snippets(self):
        """Distillation template has code snippets that should render."""
        md = pipeline_to_markdown(PIPELINE_TEMPLATES["distillation"])
        assert "```python" in md

    def test_programmatic_template_markdown_has_code_snippets(self):
        md = pipeline_to_markdown(PIPELINE_TEMPLATES["programmatic"])
        assert "```python" in md

    def test_simulation_template_markdown_has_code_snippets(self):
        md = pipeline_to_markdown(PIPELINE_TEMPLATES["simulation"])
        assert "```python" in md


# =============================================================================
# Integration: get_pipeline_template + pipeline_to_markdown
# =============================================================================


class TestGetTemplateAndRender:
    """Integration tests: get a template and render it to markdown."""

    def test_distillation_roundtrip(self):
        template = get_pipeline_template("synthetic")
        md = pipeline_to_markdown(template, dataset_name="synth-data")
        assert "synth-data" in md
        # Description should appear in the overview section
        assert "通过大型语言模型生成高质量训练数据" in md

    def test_human_annotation_roundtrip(self):
        template = get_pipeline_template("human")
        md = pipeline_to_markdown(template, dataset_name="human-data")
        assert "human-data" in md
        assert "通过众包或专家标注创建高质量数据" in md

    def test_hybrid_roundtrip(self):
        template = get_pipeline_template("anything_else")
        md = pipeline_to_markdown(template, dataset_name="hybrid-data")
        assert "hybrid-data" in md
        assert "结合 LLM 生成和人工验证/修正" in md

    def test_programmatic_by_category_roundtrip(self):
        template = get_pipeline_template("ignored", category="programmatic")
        md = pipeline_to_markdown(template)
        assert "程序化" in md

    def test_benchmark_by_category_roundtrip(self):
        template = get_pipeline_template("ignored", category="evaluation")
        md = pipeline_to_markdown(template)
        assert "评估基准" in md

    def test_simulation_by_category_roundtrip(self):
        template = get_pipeline_template("ignored", category="simulation")
        md = pipeline_to_markdown(template)
        assert "模拟器" in md


# =============================================================================
# Edge case: assemble_pipeline production depends_on pilot fallback (line 1171)
# =============================================================================


class TestAssemblePipelineProductionDependency:
    """Test the specific edge case at line 1171.

    When production phase is assembled and there are no validation phases
    between pilot and production, but pilot IS present, production should
    depend on pilot.
    """

    def test_production_falls_back_to_pilot_with_only_pilot_and_production(self):
        """Only setup + pilot + production + final_qa, no validation phases."""

        class _NoConditionsAnalysis:
            def has_difficulty_validation(self):
                return False

            def has_strategy(self, t):
                return False

        analysis = _NoConditionsAnalysis()
        phases = assemble_pipeline(
            phase_ids=["setup", "pilot", "production", "final_qa"],
            analysis=analysis,
        )
        ids = [p.phase_id for p in phases]
        assert "production" in ids
        production = next(p for p in phases if p.phase_id == "production")
        assert "pilot" in production.depends_on

    def test_production_no_pilot_no_validation(self):
        """If neither pilot nor validation phases are present, production
        should have its depends_on list resolved (removing absent references).
        """

        class _NoConditionsAnalysis:
            def has_difficulty_validation(self):
                return False

            def has_strategy(self, t):
                return False

        analysis = _NoConditionsAnalysis()
        phases = assemble_pipeline(
            phase_ids=["setup", "production", "final_qa"],
            analysis=analysis,
        )
        production = next(p for p in phases if p.phase_id == "production")
        # pilot not in present_ids, so the elif branch at line 1171 is not reached
        # but the validation_phases list is empty because only "setup" precedes production
        # Since setup is excluded from validation_phases, and pilot not present,
        # depends_on should still be resolved (pilot removed since not present)
        assert "pilot" not in production.depends_on

    def test_production_depends_on_pilot_when_pilot_ordered_after_production(self):
        """Hit line 1171: pilot is in present_ids but not yet in resolved
        when production is processed (because pilot appears after production
        in the phase_ids ordering). validation_phases is empty so the elif
        branch fires.
        """
        # By putting pilot AFTER production, when we process production,
        # resolved only has setup. validation_phases will be empty (setup is excluded).
        # But pilot IS in present_ids, so line 1171 is reached.
        phases = assemble_pipeline(
            phase_ids=["setup", "production", "pilot", "final_qa"],
        )
        production = next(p for p in phases if p.phase_id == "production")
        assert production.depends_on == ["pilot"]
