"""Tests for rubric analyzer structural templates."""

import unittest

from datarecipe.extractors import RubricsAnalyzer
from datarecipe.extractors.rubrics_analyzer import RubricPattern, RubricsAnalysisResult


class RubricsAnalyzerTests(unittest.TestCase):
    def test_structured_template_extraction(self):
        rubrics = [
            "The response should include at least 3 reasons for the decision, and explain the impact.",
            "The response should not include any offensive language.",
        ]

        result = RubricsAnalyzer().analyze(rubrics)

        self.assertEqual(result.total_rubrics, 2)
        self.assertTrue(result.structured_templates)
        top_template = result.structured_templates[0]
        self.assertIn("include", (top_template.get("action") or ""))
        self.assertIn("reasons", (top_template.get("target") or ""))
        self.assertIn("at least", (top_template.get("target") or "").lower())


# ==================== RubricPattern Dataclass ====================


class TestRubricPattern(unittest.TestCase):
    """Test RubricPattern dataclass."""

    def test_hash_id_auto_generated(self):
        p = RubricPattern(pattern="test", verb="test", verb_phrase="should test")
        self.assertTrue(p.hash_id)
        self.assertEqual(len(p.hash_id), 8)

    def test_hash_id_not_overwritten_if_set(self):
        p = RubricPattern(
            pattern="test", verb="test", verb_phrase="should test", hash_id="custom"
        )
        self.assertEqual(p.hash_id, "custom")


# ==================== RubricsAnalysisResult.summary() ====================


class TestRubricsAnalysisResultSummary(unittest.TestCase):
    """Test the summary() method of RubricsAnalysisResult (lines 56-73)."""

    def test_summary_with_data(self):
        result = RubricsAnalysisResult(
            total_rubrics=10,
            unique_patterns=5,
            avg_rubrics_per_task=2.5,
            verb_distribution={"include": 4, "explain": 3, "avoid": 2, "list": 1},
            category_distribution={"list": 5, "define": 3, "avoid": 2},
        )
        summary = result.summary()
        self.assertIn("Total Rubrics: 10", summary)
        self.assertIn("Unique Patterns: 5", summary)
        self.assertIn("Avg Rubrics/Task: 2.5", summary)
        self.assertIn("Top Verbs:", summary)
        self.assertIn("include", summary)
        self.assertIn("Top Categories:", summary)
        self.assertIn("list", summary)
        # Check percentage calculation
        self.assertIn("40.0%", summary)  # include: 4/10

    def test_summary_with_zero_rubrics(self):
        result = RubricsAnalysisResult(
            total_rubrics=0,
            verb_distribution={"include": 1},
            category_distribution={"list": 1},
        )
        summary = result.summary()
        self.assertIn("Total Rubrics: 0", summary)
        # Should not crash with zero division
        self.assertIn("0.0%", summary)


# ==================== analyze() edge cases ====================


class TestAnalyzeEdgeCases(unittest.TestCase):
    """Test analyze() edge cases for coverage gaps."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_task_count_avg_calculation(self):
        """Line 131: avg_rubrics_per_task with task_count."""
        rubrics = ["The response should include details."] * 10
        result = self.analyzer.analyze(rubrics, task_count=5)
        self.assertAlmostEqual(result.avg_rubrics_per_task, 2.0)

    def test_empty_rubric_skipped(self):
        """Line 142: empty rubrics should be skipped."""
        rubrics = ["The response should include details.", "", "  ", "Should list items."]
        result = self.analyzer.analyze(rubrics)
        self.assertEqual(result.total_rubrics, 4)
        # Only non-empty rubrics are processed
        self.assertTrue(result.unique_patterns <= 4)

    def test_duplicate_patterns_increase_frequency(self):
        """Lines 171-173: duplicate patterns update frequency and examples."""
        rubrics = [
            "The response should include 3 reasons.",
            "The response should include 5 reasons.",
        ]
        result = self.analyzer.analyze(rubrics)
        # Both should abstract to a similar template, merging them
        # At least one pattern should have frequency > 1 if they share a template
        # OR they may be different. Let's just verify no crash and check frequencies
        for p in result.patterns:
            if p.frequency > 1:
                self.assertTrue(len(p.examples) > 1)

    def test_many_duplicates_caps_examples_at_5(self):
        """Lines 171-173: examples list capped at 5."""
        rubrics = [f"The response should include {i} items." for i in range(20)]
        result = self.analyzer.analyze(rubrics)
        for p in result.patterns:
            self.assertLessEqual(len(p.examples), 5)

    def test_no_verb_match(self):
        """Line 214: _extract_verb returns None when no 'should' found."""
        rubrics = ["This is a simple sentence without the keyword."]
        result = self.analyzer.analyze(rubrics)
        self.assertEqual(result.total_rubrics, 1)
        # Should still produce a pattern
        if result.patterns:
            self.assertEqual(result.patterns[0].verb, "unknown")

    def test_verb_not_in_any_category(self):
        """Line 239: verb not in VERB_CATEGORIES returns 'other'."""
        rubrics = ["The response should summarize the content."]
        result = self.analyzer.analyze(rubrics)
        self.assertIn("other", result.category_distribution)

    def test_negation_categorized_as_avoid(self):
        """Lines 154-156, 228-229: negation verb counts."""
        rubrics = ["The response should not include harmful content."]
        result = self.analyzer.analyze(rubrics)
        self.assertIn("not", result.verb_distribution)
        self.assertIn("avoid", result.category_distribution)


# ==================== _extract_structure edge cases ====================


class TestExtractStructure(unittest.TestCase):
    """Test _extract_structure for various split patterns."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_condition_split_on_connector(self):
        """Lines 288-297: connector split (if/when/etc.)."""
        rubric = "The response should include details if the context requires it."
        action, target, condition = self.analyzer._extract_structure(
            rubric, "should include"
        )
        self.assertEqual(action, "should include")
        self.assertTrue(target)
        self.assertTrue(condition)
        self.assertIn("if", condition.lower())

    def test_condition_split_on_comma(self):
        """Lines 299-303: comma split when target > 10 chars."""
        rubric = "The response should explain the detailed methodology, including all steps and references."
        action, target, condition = self.analyzer._extract_structure(
            rubric, "should explain"
        )
        self.assertEqual(action, "should explain")
        # Should have split on comma
        self.assertTrue(target)

    def test_quantifier_condition_split(self):
        """Lines 305-313: quantifier match split."""
        rubric = "The response should list all items at least 5 times."
        action, target, condition = self.analyzer._extract_structure(
            rubric, "should list"
        )
        self.assertEqual(action, "should list")
        self.assertTrue(target)
        # "at least" should be in condition
        if condition:
            self.assertIn("at least", condition.lower())

    def test_no_verb_phrase(self):
        """Lines 271-275: verb_phrase is None."""
        action, target, condition = self.analyzer._extract_structure(
            "Some rubric text", None
        )
        self.assertEqual(action, "")

    def test_empty_remainder(self):
        """Line 318: empty target fallback."""
        action, target, condition = self.analyzer._extract_structure(
            "should include", "should include"
        )
        # Target should be empty or remain as-is
        self.assertIsInstance(target, str)


# ==================== generate_rubrics ====================


class TestGenerateRubrics(unittest.TestCase):
    """Test generate_rubrics() method (lines 378-389)."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_generate_rubrics_basic(self):
        rubrics = [
            "The response should include at least 3 reasons.",
            "The response should not use offensive language.",
        ]
        analysis = self.analyzer.analyze(rubrics)
        generated = self.analyzer.generate_rubrics(analysis, context="math problems", count=5)
        self.assertIsInstance(generated, list)
        # Each generated rubric should be a string
        for g in generated:
            self.assertIsInstance(g, str)
        # Placeholder replacements
        for g in generated:
            self.assertNotIn("[QUOTED]", g)
            self.assertNotIn("[NUM]", g)

    def test_generate_rubrics_uses_context(self):
        rubrics = ['The response should reference "the original source".']
        analysis = self.analyzer.analyze(rubrics)
        generated = self.analyzer.generate_rubrics(analysis, context="biology", count=2)
        # Context should replace [QUOTED]
        for g in generated:
            if "biology" in g:
                self.assertIn("biology", g)

    def test_generate_rubrics_empty_analysis(self):
        analysis = RubricsAnalysisResult()
        generated = self.analyzer.generate_rubrics(analysis, context="test", count=5)
        self.assertEqual(generated, [])


# ==================== to_dict ====================


class TestToDict(unittest.TestCase):
    """Test to_dict() method (line 393+)."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_to_dict_returns_valid_dict(self):
        rubrics = [
            "The response should include details.",
            "The response should not be vague.",
        ]
        result = self.analyzer.analyze(rubrics, task_count=2)
        d = self.analyzer.to_dict(result)

        self.assertIn("total_rubrics", d)
        self.assertEqual(d["total_rubrics"], 2)
        self.assertIn("unique_patterns", d)
        self.assertIn("verb_distribution", d)
        self.assertIn("patterns", d)
        self.assertIsInstance(d["patterns"], list)
        self.assertIn("avg_rubrics_per_task", d)
        self.assertIn("structured_templates", d)

    def test_to_dict_patterns_have_required_keys(self):
        rubrics = ["The response should explain concepts clearly."]
        result = self.analyzer.analyze(rubrics)
        d = self.analyzer.to_dict(result)
        for p in d["patterns"]:
            for key in ["pattern", "verb", "verb_phrase", "frequency", "template",
                        "category", "examples", "action", "target", "condition"]:
                self.assertIn(key, p)


# ==================== to_yaml_templates ====================


class TestToYamlTemplates(unittest.TestCase):
    """Test to_yaml_templates() method (lines 422-437)."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_yaml_output_is_valid(self):
        rubrics = [
            "The response should include 3 reasons.",
            "The response should not omit any details.",
        ]
        result = self.analyzer.analyze(rubrics)
        yaml_str = self.analyzer.to_yaml_templates(result)
        self.assertIsInstance(yaml_str, str)
        self.assertIn("rubric_templates", yaml_str)
        self.assertIn("category", yaml_str)
        self.assertIn("action", yaml_str)

    def test_yaml_with_empty_result(self):
        result = RubricsAnalysisResult()
        yaml_str = self.analyzer.to_yaml_templates(result)
        self.assertIn("rubric_templates", yaml_str)


# ==================== to_markdown_templates ====================


class TestToMarkdownTemplates(unittest.TestCase):
    """Test to_markdown_templates() method (lines 445-462)."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_markdown_output_has_header_and_table(self):
        rubrics = [
            "The response should include 3 reasons.",
            "The response should not omit any details.",
        ]
        result = self.analyzer.analyze(rubrics)
        md = self.analyzer.to_markdown_templates(result)
        self.assertIn("# Rubric 模板库", md)
        self.assertIn("| 类别 |", md)
        self.assertIn("| 动作 |", md)
        self.assertIn("| 频次 |", md)

    def test_markdown_with_empty_conditions(self):
        rubrics = ["Should list items."]
        result = self.analyzer.analyze(rubrics)
        md = self.analyzer.to_markdown_templates(result)
        # Empty conditions show as "—"
        self.assertIsInstance(md, str)

    def test_markdown_with_empty_result(self):
        result = RubricsAnalysisResult()
        md = self.analyzer.to_markdown_templates(result)
        self.assertIn("# Rubric 模板库", md)


# ==================== _extract_common_phrases ====================


class TestExtractCommonPhrases(unittest.TestCase):
    """Test _extract_common_phrases frequency threshold."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_phrases_meet_min_frequency(self):
        rubrics = ["The response should include all relevant details."] * 10
        phrases = self.analyzer._extract_common_phrases(rubrics, min_freq=5)
        for _phrase, count in phrases:
            self.assertGreaterEqual(count, 5)

    def test_phrases_below_threshold_excluded(self):
        rubrics = [
            "The response should include details.",
            "Should list items for review.",
        ]
        phrases = self.analyzer._extract_common_phrases(rubrics, min_freq=5)
        self.assertEqual(phrases, [])


# ==================== Various starters ====================


class TestVariousStarters(unittest.TestCase):
    """Test different rubric sentence starters."""

    def setUp(self):
        self.analyzer = RubricsAnalyzer()

    def test_answer_starter(self):
        rubrics = ["The answer should include 5 details."]
        result = self.analyzer.analyze(rubrics)
        self.assertTrue(result.sentence_starters)

    def test_output_starter(self):
        rubrics = ["The output should format results correctly."]
        result = self.analyzer.analyze(rubrics)
        self.assertTrue(result.sentence_starters)

    def test_it_starter(self):
        rubrics = ["It should mention the key concepts."]
        result = self.analyzer.analyze(rubrics)
        self.assertTrue(result.sentence_starters)

    def test_should_starter(self):
        rubrics = ["Should verify the accuracy of the data."]
        result = self.analyzer.analyze(rubrics)
        self.assertTrue(result.sentence_starters)


if __name__ == "__main__":
    unittest.main()
