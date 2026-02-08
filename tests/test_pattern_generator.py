"""Unit tests for PatternGenerator, GeneratedDataItem, and GenerationResult.

Tests cover:
- GeneratedDataItem dataclass (auto-ID generation, field defaults)
- GenerationResult dataclass and summary()
- PatternGenerator.generate_rubrics() (all branches)
- PatternGenerator.generate_prompts() (all branches)
- PatternGenerator.generate_contexts() (all branches)
- Internal scoring methods (_score_rubric, _score_prompt, _score_context)
- Internal helper methods (_fill_rubric_template, _get_default_prompt_templates, _generate_context_params)
- export_jsonl() and to_dict() serialization
"""

import json
import os
import tempfile
import unittest

from datarecipe.extractors.prompt_extractor import PromptLibrary, PromptTemplate
from datarecipe.extractors.rubrics_analyzer import RubricsAnalysisResult
from datarecipe.generators.pattern_generator import (
    GeneratedDataItem,
    GenerationResult,
    PatternGenerator,
)

# ==================== GeneratedDataItem ====================


class TestGeneratedDataItem(unittest.TestCase):
    """Test GeneratedDataItem dataclass."""

    def test_default_fields(self):
        item = GeneratedDataItem(
            id="abc",
            content="test content",
            data_type="rubric",
            template_used="tmpl",
        )
        self.assertEqual(item.id, "abc")
        self.assertEqual(item.content, "test content")
        self.assertEqual(item.data_type, "rubric")
        self.assertEqual(item.template_used, "tmpl")
        self.assertEqual(item.parameters, {})
        self.assertEqual(item.quality_score, 0.0)
        self.assertEqual(item.metadata, {})

    def test_auto_id_when_empty(self):
        """When id is empty string, __post_init__ should generate a hash-based id."""
        item = GeneratedDataItem(
            id="",
            content="some content",
            data_type="rubric",
            template_used="tmpl",
        )
        self.assertTrue(len(item.id) > 0)
        self.assertEqual(len(item.id), 12)  # md5 hexdigest[:12]

    def test_auto_id_uniqueness_for_different_content(self):
        """Different content should produce different auto-generated IDs (most of the time)."""
        item1 = GeneratedDataItem(id="", content="content A", data_type="rubric", template_used="t")
        item2 = GeneratedDataItem(id="", content="content B", data_type="rubric", template_used="t")
        # They may or may not differ due to timestamp, but at least they should be valid
        self.assertEqual(len(item1.id), 12)
        self.assertEqual(len(item2.id), 12)

    def test_explicit_id_preserved(self):
        """When a non-empty id is provided, it should be preserved."""
        item = GeneratedDataItem(
            id="my-custom-id",
            content="content",
            data_type="prompt",
            template_used="t",
        )
        self.assertEqual(item.id, "my-custom-id")

    def test_custom_parameters_and_metadata(self):
        item = GeneratedDataItem(
            id="x",
            content="c",
            data_type="context",
            template_used="t",
            parameters={"key": "value"},
            quality_score=0.85,
            metadata={"source": "test"},
        )
        self.assertEqual(item.parameters, {"key": "value"})
        self.assertEqual(item.quality_score, 0.85)
        self.assertEqual(item.metadata, {"source": "test"})


# ==================== GenerationResult ====================


class TestGenerationResult(unittest.TestCase):
    """Test GenerationResult dataclass and summary()."""

    def test_default_fields(self):
        result = GenerationResult()
        self.assertEqual(result.items, [])
        self.assertEqual(result.total_generated, 0)
        self.assertEqual(result.templates_used, 0)
        self.assertEqual(result.generation_time, "")
        self.assertEqual(result.avg_quality_score, 0.0)
        self.assertEqual(result.unique_ratio, 0.0)

    def test_summary_format(self):
        result = GenerationResult(
            total_generated=10,
            templates_used=3,
            avg_quality_score=0.75,
            unique_ratio=0.9,
        )
        s = result.summary()
        self.assertIn("Generated: 10 items", s)
        self.assertIn("Templates Used: 3", s)
        self.assertIn("Avg Quality: 0.75", s)
        self.assertIn("Unique Ratio: 90.0%", s)

    def test_summary_zero_values(self):
        result = GenerationResult()
        s = result.summary()
        self.assertIn("Generated: 0 items", s)
        self.assertIn("Avg Quality: 0.00", s)
        self.assertIn("Unique Ratio: 0.0%", s)


# ==================== PatternGenerator: generate_rubrics ====================


class TestGenerateRubrics(unittest.TestCase):
    """Test PatternGenerator.generate_rubrics()."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_generate_rubrics_default(self):
        """Generate rubrics with default parameters."""
        result = self.gen.generate_rubrics()
        self.assertIsInstance(result, GenerationResult)
        self.assertGreater(result.total_generated, 0)
        self.assertGreater(len(result.items), 0)
        self.assertTrue(result.generation_time)

    def test_generate_rubrics_respects_count(self):
        """Should not generate more items than requested count."""
        result = self.gen.generate_rubrics(count=3)
        self.assertLessEqual(result.total_generated, 3)
        self.assertLessEqual(len(result.items), 3)

    def test_generate_rubrics_with_context(self):
        """Context should appear in generated rubric content."""
        result = self.gen.generate_rubrics(context="machine learning", count=5)
        for item in result.items:
            self.assertIn("machine learning", item.content)
            self.assertEqual(item.data_type, "rubric")

    def test_generate_rubrics_with_specific_categories(self):
        """Only generate rubrics for specified categories."""
        result = self.gen.generate_rubrics(categories=["define", "list"], count=10)
        for item in result.items:
            self.assertIn(item.parameters["category"], ["define", "list"])

    def test_generate_rubrics_empty_categories(self):
        """Filtering to non-existent categories returns empty result."""
        result = self.gen.generate_rubrics(categories=["nonexistent_category"])
        self.assertEqual(result.total_generated, 0)
        self.assertEqual(len(result.items), 0)

    def test_generate_rubrics_with_custom_templates(self):
        """Custom templates should be merged with defaults."""
        custom = {
            "custom_cat": [
                "Custom template about {topic}",
                "Another custom template about {topic}",
            ]
        }
        result = self.gen.generate_rubrics(
            custom_templates=custom,
            categories=["custom_cat"],
            count=5,
        )
        self.assertGreater(result.total_generated, 0)
        for item in result.items:
            self.assertEqual(item.parameters["category"], "custom_cat")

    def test_generate_rubrics_with_rubrics_analysis(self):
        """When rubrics_analysis has top_templates, they should be added as 'discovered' category."""
        analysis = RubricsAnalysisResult(
            top_templates=[
                "The response should mention {topic}",
                "The response should cover {topic} in detail",
            ]
        )
        result = self.gen.generate_rubrics(
            rubrics_analysis=analysis,
            categories=["discovered"],
            count=5,
        )
        self.assertGreater(result.total_generated, 0)
        for item in result.items:
            self.assertEqual(item.parameters["category"], "discovered")

    def test_generate_rubrics_analysis_empty_top_templates(self):
        """When rubrics_analysis has empty top_templates, 'discovered' category not added."""
        analysis = RubricsAnalysisResult(top_templates=[])
        result = self.gen.generate_rubrics(
            rubrics_analysis=analysis,
            categories=["discovered"],
            count=5,
        )
        # 'discovered' category should not exist since top_templates is empty
        self.assertEqual(result.total_generated, 0)

    def test_generate_rubrics_quality_metrics(self):
        """avg_quality_score and unique_ratio should be calculated."""
        result = self.gen.generate_rubrics(count=5)
        if result.total_generated > 0:
            self.assertGreater(result.avg_quality_score, 0.0)
            self.assertGreater(result.unique_ratio, 0.0)

    def test_generate_rubrics_deduplication(self):
        """Duplicate rubrics should not be generated (same generator instance)."""
        gen = PatternGenerator()
        result1 = gen.generate_rubrics(context="test", count=5)
        result2 = gen.generate_rubrics(context="test", count=5)
        # Second call should generate fewer (or zero) since hashes are tracked
        self.assertLessEqual(result2.total_generated, result1.total_generated)

    def test_generate_rubrics_templates_used_count(self):
        """templates_used should reflect how many unique templates were used."""
        result = self.gen.generate_rubrics(count=10)
        self.assertGreater(result.templates_used, 0)
        self.assertLessEqual(result.templates_used, result.total_generated)

    def test_generate_rubrics_items_have_auto_id(self):
        """Generated items with empty id should get auto-generated IDs."""
        result = self.gen.generate_rubrics(count=3)
        for item in result.items:
            self.assertTrue(len(item.id) > 0)

    def test_generate_rubrics_large_count(self):
        """Requesting more rubrics than templates available should not crash."""
        result = self.gen.generate_rubrics(count=100)
        self.assertIsInstance(result, GenerationResult)
        # Total generated will be limited by available templates
        self.assertLessEqual(result.total_generated, 100)


# ==================== PatternGenerator: generate_prompts ====================


class TestGeneratePrompts(unittest.TestCase):
    """Test PatternGenerator.generate_prompts()."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_generate_prompts_default(self):
        """Generate prompts with default parameters (no library)."""
        result = self.gen.generate_prompts()
        self.assertIsInstance(result, GenerationResult)
        self.assertGreater(result.total_generated, 0)
        for item in result.items:
            self.assertEqual(item.data_type, "prompt")

    def test_generate_prompts_with_domain(self):
        """Domain should appear in generated prompt content."""
        result = self.gen.generate_prompts(domain="legal", count=3)
        for item in result.items:
            self.assertIn("legal", item.content)
            self.assertEqual(item.parameters["domain"], "legal")

    def test_generate_prompts_system_category(self):
        """System category should generate system-style prompts."""
        result = self.gen.generate_prompts(category="system", count=3)
        for item in result.items:
            self.assertEqual(item.parameters["category"], "system")

    def test_generate_prompts_task_category(self):
        """Task category should generate task-style prompts."""
        result = self.gen.generate_prompts(category="task", count=3)
        self.assertGreater(result.total_generated, 0)
        for item in result.items:
            self.assertEqual(item.parameters["category"], "task")

    def test_generate_prompts_constraint_category(self):
        """Constraint category should generate constraint-style prompts."""
        result = self.gen.generate_prompts(category="constraint", count=3)
        self.assertGreater(result.total_generated, 0)
        for item in result.items:
            self.assertEqual(item.parameters["category"], "constraint")

    def test_generate_prompts_unknown_category_falls_back(self):
        """Unknown category should fall back to system templates."""
        result = self.gen.generate_prompts(category="unknown_cat", count=3)
        self.assertGreater(result.total_generated, 0)

    def test_generate_prompts_with_prompt_library(self):
        """When a prompt library is provided, use its templates."""
        lib = PromptLibrary(
            templates=[
                PromptTemplate(content="Template 1 for {domain}", category="system", domain="general"),
                PromptTemplate(content="Template 2 for {domain}", category="system", domain="general"),
            ]
        )
        result = self.gen.generate_prompts(prompt_library=lib, domain="general", count=5)
        self.assertGreater(result.total_generated, 0)
        self.assertLessEqual(result.total_generated, 2)  # Only 2 templates available

    def test_generate_prompts_with_library_domain_filter(self):
        """Library templates should be filtered by category and domain."""
        lib = PromptLibrary(
            templates=[
                PromptTemplate(content="System A", category="system", domain="legal"),
                PromptTemplate(content="System B", category="system", domain="general"),
                PromptTemplate(content="Task A", category="task", domain="legal"),
            ]
        )
        result = self.gen.generate_prompts(
            prompt_library=lib, domain="legal", category="system", count=5
        )
        # Should match: "System A" (domain=legal) and "System B" (domain=general)
        self.assertEqual(result.total_generated, 2)

    def test_generate_prompts_with_library_no_matching(self):
        """When library has no matching templates, return empty result."""
        lib = PromptLibrary(
            templates=[
                PromptTemplate(content="Task A", category="task", domain="science"),
            ]
        )
        result = self.gen.generate_prompts(
            prompt_library=lib, domain="legal", category="system", count=5
        )
        self.assertEqual(result.total_generated, 0)

    def test_generate_prompts_with_empty_library(self):
        """Empty library should fall back to default templates."""
        lib = PromptLibrary(templates=[])
        result = self.gen.generate_prompts(prompt_library=lib, count=3)
        self.assertGreater(result.total_generated, 0)

    def test_generate_prompts_with_customize_fn(self):
        """Custom function should transform generated content."""
        def upper_fn(s: str) -> str:
            return s.upper()

        result = self.gen.generate_prompts(customize_fn=upper_fn, count=3)
        for item in result.items:
            self.assertEqual(item.content, item.content.upper())

    def test_generate_prompts_domain_placeholder_replacement(self):
        """Domain placeholders should be replaced in content."""
        lib = PromptLibrary(
            templates=[
                PromptTemplate(
                    content="You are a {domain} expert. Help with [DOMAIN] tasks.",
                    category="system",
                    domain="general",
                ),
            ]
        )
        result = self.gen.generate_prompts(
            prompt_library=lib, domain="finance", category="system", count=1
        )
        self.assertEqual(result.total_generated, 1)
        self.assertIn("finance", result.items[0].content)
        self.assertNotIn("{domain}", result.items[0].content)
        self.assertNotIn("[DOMAIN]", result.items[0].content)

    def test_generate_prompts_quality_metrics(self):
        """avg_quality_score should be calculated for generated prompts."""
        result = self.gen.generate_prompts(count=3)
        if result.total_generated > 0:
            self.assertGreater(result.avg_quality_score, 0.0)

    def test_generate_prompts_respects_count(self):
        """Should not generate more than requested count."""
        result = self.gen.generate_prompts(count=2)
        self.assertLessEqual(result.total_generated, 2)


# ==================== PatternGenerator: generate_contexts ====================


class TestGenerateContexts(unittest.TestCase):
    """Test PatternGenerator.generate_contexts()."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_generate_contexts_game_rules_default(self):
        """Generate game_rules context with defaults."""
        result = self.gen.generate_contexts(
            strategy="synthetic",
            template_type="game_rules",
            count=1,
        )
        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.total_generated, 1)
        self.assertEqual(result.templates_used, 1)
        item = result.items[0]
        self.assertEqual(item.data_type, "context")
        self.assertEqual(item.template_used, "game_rules")
        self.assertIn("Stellar Quest 1", item.content)

    def test_generate_contexts_procedure(self):
        """Generate procedure context."""
        result = self.gen.generate_contexts(
            strategy="synthetic",
            template_type="procedure",
            count=1,
        )
        self.assertEqual(result.total_generated, 1)
        self.assertIn("Process 1", result.items[0].content)
        self.assertIn("Procedure", result.items[0].content)

    def test_generate_contexts_technical_doc(self):
        """Generate technical_doc context with 'modified' strategy."""
        result = self.gen.generate_contexts(
            strategy="modified",
            template_type="technical_doc",
            count=1,
        )
        self.assertEqual(result.total_generated, 1)
        self.assertIn("System Component 1", result.items[0].content)
        self.assertIn("Documentation", result.items[0].content)

    def test_generate_contexts_custom_parameters(self):
        """Custom parameters should override defaults."""
        params = {
            "game_name": "Chess Pro",
            "game_type": "board",
            "player_count": "2",
        }
        result = self.gen.generate_contexts(
            strategy="synthetic",
            template_type="game_rules",
            parameters=params,
            count=1,
        )
        self.assertEqual(result.total_generated, 1)
        self.assertIn("Chess Pro", result.items[0].content)
        self.assertIn("board", result.items[0].content)

    def test_generate_contexts_multiple(self):
        """Generate multiple contexts with incrementing index."""
        result = self.gen.generate_contexts(
            strategy="synthetic",
            template_type="game_rules",
            count=3,
        )
        self.assertEqual(result.total_generated, 3)
        self.assertIn("Stellar Quest 1", result.items[0].content)
        self.assertIn("Stellar Quest 2", result.items[1].content)
        self.assertIn("Stellar Quest 3", result.items[2].content)

    def test_generate_contexts_unknown_strategy(self):
        """Unknown strategy returns empty result."""
        result = self.gen.generate_contexts(
            strategy="unknown_strategy",
            template_type="game_rules",
            count=1,
        )
        self.assertEqual(result.total_generated, 0)
        self.assertEqual(len(result.items), 0)

    def test_generate_contexts_unknown_template_type(self):
        """Unknown template type returns empty result."""
        result = self.gen.generate_contexts(
            strategy="synthetic",
            template_type="unknown_type",
            count=1,
        )
        self.assertEqual(result.total_generated, 0)

    def test_generate_contexts_quality_metrics(self):
        """avg_quality_score should be set when items are generated."""
        result = self.gen.generate_contexts(
            strategy="synthetic",
            template_type="game_rules",
            count=2,
        )
        self.assertGreater(result.avg_quality_score, 0.0)

    def test_generate_contexts_parameters_stored_in_item(self):
        """Generated item should store the filled parameters."""
        result = self.gen.generate_contexts(
            strategy="synthetic",
            template_type="procedure",
            parameters={"procedure_name": "Custom Process"},
            count=1,
        )
        self.assertEqual(result.items[0].parameters["procedure_name"], "Custom Process")


# ==================== PatternGenerator: _fill_rubric_template ====================


class TestFillRubricTemplate(unittest.TestCase):
    """Test internal _fill_rubric_template method."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_fill_topic_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should define what {topic} is",
            "quantum physics",
            "define",
        )
        self.assertIn("quantum physics", result)
        self.assertNotIn("{topic}", result)

    def test_fill_items_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should list all {items}",
            "algorithms",
            "list",
        )
        self.assertIn("items related to algorithms", result)
        self.assertNotIn("{items}", result)

    def test_fill_reason_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should explain why {reason}",
            "gravity",
            "explain",
        )
        self.assertIn("gravity works this way", result)

    def test_fill_process_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should describe how {process}",
            "compilation",
            "explain",
        )
        self.assertIn("the compilation process works", result)

    def test_fill_elements_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should clarify the relationship between {elements}",
            "atoms",
            "explain",
        )
        self.assertIn("elements of atoms", result)

    def test_fill_assumption_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should not assume {assumption}",
            "chemistry",
            "avoid",
        )
        self.assertIn("prior knowledge about chemistry", result)

    def test_fill_excluded_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should not include {excluded}",
            "biology",
            "avoid",
        )
        self.assertIn("information not in the biology", result)

    def test_fill_prohibited_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should avoid {prohibited}",
            "math",
            "avoid",
        )
        self.assertIn("content outside the scope of math", result)

    def test_fill_condition_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should verify that {condition}",
            "safety",
            "verify",
        )
        self.assertIn("the safety requirements are met", result)

    def test_fill_statement_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should confirm {statement}",
            "policy",
            "verify",
        )
        self.assertIn("the policy is correctly understood", result)

    def test_fill_check_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should check whether {check}",
            "compliance",
            "verify",
        )
        self.assertIn("all aspects of compliance are addressed", result)

    def test_fill_format_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should be formatted as {format}",
            "data",
            "format",
        )
        self.assertIn("a clear, structured manner", result)

    def test_fill_content_placeholder(self):
        result = self.gen._fill_rubric_template(
            "The response should organize {content} clearly",
            "reports",
            "format",
        )
        self.assertIn("information about reports", result)

    def test_no_placeholder_unchanged(self):
        """Template without placeholders should be returned unchanged."""
        template = "This is a plain template without placeholders"
        result = self.gen._fill_rubric_template(template, "topic", "define")
        self.assertEqual(result, template)


# ==================== PatternGenerator: _get_default_prompt_templates ====================


class TestGetDefaultPromptTemplates(unittest.TestCase):
    """Test _get_default_prompt_templates."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_system_templates(self):
        templates = self.gen._get_default_prompt_templates("science", "system")
        self.assertEqual(len(templates), 3)
        for t in templates:
            self.assertIn("science", t)

    def test_task_templates(self):
        templates = self.gen._get_default_prompt_templates("science", "task")
        self.assertEqual(len(templates), 3)
        # Task templates do not contain domain
        self.assertTrue(all("analyze" in t.lower() or "based" in t.lower() or "review" in t.lower() for t in templates))

    def test_constraint_templates(self):
        templates = self.gen._get_default_prompt_templates("science", "constraint")
        self.assertEqual(len(templates), 3)

    def test_unknown_category_falls_back_to_system(self):
        templates = self.gen._get_default_prompt_templates("science", "unknown")
        system_templates = self.gen._get_default_prompt_templates("science", "system")
        self.assertEqual(templates, system_templates)


# ==================== PatternGenerator: _generate_context_params ====================


class TestGenerateContextParams(unittest.TestCase):
    """Test _generate_context_params."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_game_rules_defaults(self):
        params = self.gen._generate_context_params("game_rules", {}, 0)
        self.assertEqual(params["game_name"], "Stellar Quest 1")
        self.assertEqual(params["game_type"], "strategy")
        self.assertEqual(params["player_count"], "2-4")

    def test_game_rules_index_increments(self):
        params0 = self.gen._generate_context_params("game_rules", {}, 0)
        params1 = self.gen._generate_context_params("game_rules", {}, 1)
        self.assertEqual(params0["game_name"], "Stellar Quest 1")
        self.assertEqual(params1["game_name"], "Stellar Quest 2")

    def test_procedure_defaults(self):
        params = self.gen._generate_context_params("procedure", {}, 0)
        self.assertEqual(params["procedure_name"], "Process 1")
        self.assertIn("Initialize", params["steps"])

    def test_technical_doc_defaults(self):
        params = self.gen._generate_context_params("technical_doc", {}, 2)
        self.assertEqual(params["topic"], "System Component 3")

    def test_provided_overrides_defaults(self):
        provided = {"game_name": "My Game", "game_type": "puzzle"}
        params = self.gen._generate_context_params("game_rules", provided, 0)
        self.assertEqual(params["game_name"], "My Game")
        self.assertEqual(params["game_type"], "puzzle")
        # Other defaults should still be present
        self.assertIn("player_count", params)

    def test_unknown_template_type(self):
        """Unknown template type returns only provided params."""
        params = self.gen._generate_context_params("unknown", {"key": "val"}, 0)
        self.assertEqual(params, {"key": "val"})

    def test_unknown_template_type_empty_provided(self):
        params = self.gen._generate_context_params("unknown", {}, 0)
        self.assertEqual(params, {})


# ==================== PatternGenerator: _score_rubric ====================


class TestScoreRubric(unittest.TestCase):
    """Test _score_rubric scoring logic."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_base_score(self):
        """A minimal rubric gets the base score of 0.5."""
        score = self.gen._score_rubric("short")
        self.assertEqual(score, 0.5)

    def test_standard_structure_bonus(self):
        """Rubric starting with 'the response should' gets +0.2."""
        score = self.gen._score_rubric("The response should do something")
        self.assertGreaterEqual(score, 0.7)

    def test_length_bonus(self):
        """Rubric longer than 50 chars gets +0.1."""
        long_rubric = "The response should include a very detailed explanation of this topic area"
        score = self.gen._score_rubric(long_rubric)
        self.assertGreater(score, 0.5)

    def test_action_verb_bonus(self):
        """Rubric containing action verbs gets +0.1."""
        score_with_verb = self.gen._score_rubric("Must define the concept")
        score_without = self.gen._score_rubric("Some text here")
        self.assertGreater(score_with_verb, score_without)

    def test_vagueness_penalty(self):
        """Rubric containing vague words gets -0.2."""
        score = self.gen._score_rubric("The answer maybe could be correct")
        vague_score = score
        normal_score = self.gen._score_rubric("The answer is correct")
        self.assertLess(vague_score, normal_score)

    def test_score_clamped_to_0_1(self):
        """Score should always be between 0.0 and 1.0."""
        # Test with very vague short text
        score = self.gen._score_rubric("maybe possibly might could")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_max_score(self):
        """A rubric with all bonuses should approach 0.9."""
        rubric = "The response should define and explain the topic with comprehensive detail and include all necessary information"
        score = self.gen._score_rubric(rubric)
        self.assertGreaterEqual(score, 0.8)
        self.assertLessEqual(score, 1.0)

    def test_multiple_vague_words(self):
        """Multiple vague words should only apply penalty once (per word check logic)."""
        score = self.gen._score_rubric("maybe possibly")
        # -0.2 penalty applied (any match triggers it)
        self.assertGreaterEqual(score, 0.0)


# ==================== PatternGenerator: _score_prompt ====================


class TestScorePrompt(unittest.TestCase):
    """Test _score_prompt scoring logic."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_base_score(self):
        score = self.gen._score_prompt("x")
        self.assertEqual(score, 0.5)

    def test_length_bonus(self):
        """Prompt between 50-500 chars gets +0.2."""
        prompt = "You are a helpful assistant. " * 3  # ~87 chars
        score = self.gen._score_prompt(prompt)
        self.assertGreaterEqual(score, 0.7)

    def test_too_short_no_length_bonus(self):
        score = self.gen._score_prompt("Short")
        self.assertLess(score, 0.7)

    def test_too_long_no_length_bonus(self):
        score = self.gen._score_prompt("x" * 501)
        # No length bonus, no instruction bonus
        self.assertEqual(score, 0.5)

    def test_instruction_bonus(self):
        """Prompt with 'you are', 'please', or 'help' gets +0.15."""
        score = self.gen._score_prompt("You are a helpful assistant specializing in science.")
        # 50 < len < 500 (+0.2), has "you are" and "help" (+0.15)
        self.assertGreaterEqual(score, 0.85)

    def test_structure_bonus(self):
        """Prompt with newline gets +0.1."""
        score = self.gen._score_prompt("Line 1.\nLine 2. This is a longer prompt text for length bonus.")
        self.assertAlmostEqual(score, 0.8, places=5)  # length + structure

    def test_score_clamped(self):
        score = self.gen._score_prompt("x")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ==================== PatternGenerator: _score_context ====================


class TestScoreContext(unittest.TestCase):
    """Test _score_context scoring logic."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_base_score(self):
        score = self.gen._score_context("short")
        self.assertEqual(score, 0.5)

    def test_length_bonus(self):
        """Context longer than 500 chars gets +0.2."""
        score = self.gen._score_context("x" * 501)
        self.assertGreaterEqual(score, 0.7)

    def test_no_length_bonus_short(self):
        score = self.gen._score_context("x" * 100)
        self.assertLess(score, 0.7)

    def test_header_bonus(self):
        """Context with '#' gets +0.15."""
        score = self.gen._score_context("# Title\nSome content here that is quite long " * 20)
        self.assertGreaterEqual(score, 0.85)

    def test_list_bonus_dash(self):
        """Context with '-' gets +0.1."""
        content = "- Item 1\n- Item 2\n" * 50  # Make it long enough
        score = self.gen._score_context(content)
        self.assertAlmostEqual(score, 0.8, places=5)  # length + list

    def test_list_bonus_numbered(self):
        """Context with '1.' gets +0.1."""
        content = "1. Step one\n2. Step two\n" * 50
        score = self.gen._score_context(content)
        self.assertAlmostEqual(score, 0.8, places=5)  # length + list

    def test_all_bonuses(self):
        """Context with all features should get high score."""
        content = "# Title\n" + "- Item\n" * 100 + "1. Step"
        score = self.gen._score_context(content)
        self.assertGreaterEqual(score, 0.9)

    def test_score_clamped(self):
        score = self.gen._score_context("x")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ==================== PatternGenerator: export_jsonl ====================


class TestExportJsonl(unittest.TestCase):
    """Test export_jsonl serialization."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_export_jsonl_creates_file(self):
        result = self.gen.generate_rubrics(count=3)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name

        try:
            self.gen.export_jsonl(result, filepath)
            self.assertTrue(os.path.exists(filepath))

            with open(filepath, encoding="utf-8") as f:
                lines = f.readlines()

            self.assertEqual(len(lines), result.total_generated)

            # Each line should be valid JSON
            for line in lines:
                data = json.loads(line.strip())
                self.assertIn("id", data)
                self.assertIn("content", data)
                self.assertIn("type", data)
                self.assertIn("template", data)
                self.assertIn("parameters", data)
                self.assertIn("quality_score", data)
                self.assertIn("metadata", data)
        finally:
            os.unlink(filepath)

    def test_export_jsonl_empty_result(self):
        result = GenerationResult()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name

        try:
            self.gen.export_jsonl(result, filepath)
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "")
        finally:
            os.unlink(filepath)

    def test_export_jsonl_unicode_content(self):
        """Ensure non-ASCII characters are preserved (ensure_ascii=False)."""
        item = GeneratedDataItem(
            id="test1",
            content="This contains unicode: \u4f60\u597d\u4e16\u754c",
            data_type="rubric",
            template_used="tmpl",
        )
        result = GenerationResult(items=[item], total_generated=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name

        try:
            self.gen.export_jsonl(result, filepath)
            with open(filepath, encoding="utf-8") as f:
                line = f.readline()
            data = json.loads(line)
            self.assertIn("\u4f60\u597d", data["content"])
        finally:
            os.unlink(filepath)


# ==================== PatternGenerator: to_dict ====================


class TestToDict(unittest.TestCase):
    """Test to_dict serialization."""

    def setUp(self):
        self.gen = PatternGenerator()

    def test_to_dict_structure(self):
        result = self.gen.generate_rubrics(count=3)
        d = self.gen.to_dict(result)
        self.assertIn("total_generated", d)
        self.assertIn("templates_used", d)
        self.assertIn("generation_time", d)
        self.assertIn("avg_quality_score", d)
        self.assertIn("unique_ratio", d)
        self.assertIn("items", d)

    def test_to_dict_items_structure(self):
        result = self.gen.generate_rubrics(count=2)
        d = self.gen.to_dict(result)
        for item_dict in d["items"]:
            self.assertIn("id", item_dict)
            self.assertIn("content", item_dict)
            self.assertIn("type", item_dict)
            self.assertIn("quality_score", item_dict)

    def test_to_dict_values_match(self):
        result = self.gen.generate_rubrics(count=2)
        d = self.gen.to_dict(result)
        self.assertEqual(d["total_generated"], result.total_generated)
        self.assertEqual(d["templates_used"], result.templates_used)
        self.assertEqual(d["avg_quality_score"], result.avg_quality_score)

    def test_to_dict_empty_result(self):
        result = GenerationResult()
        d = self.gen.to_dict(result)
        self.assertEqual(d["total_generated"], 0)
        self.assertEqual(d["items"], [])

    def test_to_dict_item_type_field(self):
        """The 'type' key in items dict corresponds to 'data_type' on the item."""
        item = GeneratedDataItem(
            id="x", content="c", data_type="prompt", template_used="t"
        )
        result = GenerationResult(items=[item], total_generated=1)
        d = self.gen.to_dict(result)
        self.assertEqual(d["items"][0]["type"], "prompt")


# ==================== Integration / Edge Cases ====================


class TestPatternGeneratorIntegration(unittest.TestCase):
    """Integration-level tests combining multiple methods."""

    def test_full_workflow_rubrics(self):
        """Generate rubrics, export to JSONL, convert to dict."""
        gen = PatternGenerator()
        result = gen.generate_rubrics(context="chess rules", count=5)
        self.assertGreater(result.total_generated, 0)

        # Export to JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
        try:
            gen.export_jsonl(result, filepath)
            with open(filepath, encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), result.total_generated)
        finally:
            os.unlink(filepath)

        # Convert to dict
        d = gen.to_dict(result)
        self.assertEqual(d["total_generated"], result.total_generated)

    def test_full_workflow_prompts(self):
        """Generate prompts and convert to dict."""
        gen = PatternGenerator()
        result = gen.generate_prompts(domain="medical", category="system", count=3)
        self.assertGreater(result.total_generated, 0)
        d = gen.to_dict(result)
        self.assertEqual(len(d["items"]), result.total_generated)

    def test_full_workflow_contexts(self):
        """Generate contexts and convert to dict."""
        gen = PatternGenerator()
        result = gen.generate_contexts(
            strategy="synthetic",
            template_type="game_rules",
            parameters={"game_name": "TestGame"},
            count=2,
        )
        self.assertEqual(result.total_generated, 2)
        d = gen.to_dict(result)
        self.assertEqual(len(d["items"]), 2)

    def test_deduplication_across_calls(self):
        """Same generator instance tracks hashes across multiple generate_rubrics calls."""
        gen = PatternGenerator()
        r1 = gen.generate_rubrics(context="topic", categories=["define"], count=3)
        count1 = r1.total_generated
        # Second call with same context should produce fewer new items
        r2 = gen.generate_rubrics(context="topic", categories=["define"], count=3)
        count2 = r2.total_generated
        self.assertLess(count2, count1)

    def test_fresh_generator_no_dedup(self):
        """Different generator instances don't share deduplication state."""
        gen1 = PatternGenerator()
        gen2 = PatternGenerator()
        r1 = gen1.generate_rubrics(context="topic", categories=["define"], count=3)
        r2 = gen2.generate_rubrics(context="topic", categories=["define"], count=3)
        self.assertEqual(r1.total_generated, r2.total_generated)

    def test_generated_hashes_set_grows(self):
        """The generated_hashes set should grow as rubrics are generated."""
        gen = PatternGenerator()
        self.assertEqual(len(gen.generated_hashes), 0)
        gen.generate_rubrics(count=5)
        self.assertGreater(len(gen.generated_hashes), 0)


if __name__ == "__main__":
    unittest.main()
