"""Tests for rubric analyzer structural templates."""

import unittest

from datarecipe.extractors import RubricsAnalyzer


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


if __name__ == "__main__":
    unittest.main()
