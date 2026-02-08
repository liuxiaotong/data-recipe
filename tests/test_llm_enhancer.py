"""Unit tests for LLM Enhancement Layer.

Tests LLMEnhancer class: prompt building, response parsing,
mode resolution, and the get_prompt/enhance_from_response pattern.
"""

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from datarecipe.generators.llm_enhancer import (
    EnhancedContext,
    LLMEnhancer,
)

# ---------- Stub objects ----------


@dataclass
class StubComplexityMetrics:
    primary_domain: object = None
    difficulty_score: float = 2.0


class StubDomain:
    def __init__(self, value: str):
        self.value = value


@dataclass
class StubAllocation:
    human_work_percentage: float = 60.0
    total_cost: float = 15000.0


@dataclass
class StubRubricsResult:
    total_rubrics: int = 50
    unique_patterns: int = 12


@dataclass
class StubLLMAnalysis:
    purpose: str = "Training reward models"


# ==================== EnhancedContext Dataclass ====================


class TestEnhancedContextDefaults(unittest.TestCase):
    """Test EnhancedContext dataclass defaults."""

    def test_defaults_empty(self):
        ctx = EnhancedContext()
        self.assertEqual(ctx.dataset_purpose_summary, "")
        self.assertEqual(ctx.key_methodology_insights, [])
        self.assertFalse(ctx.generated)
        self.assertEqual(ctx.raw_response, "")

    def test_generated_flag(self):
        ctx = EnhancedContext(generated=True)
        self.assertTrue(ctx.generated)


# ==================== Prompt Building ====================


class TestBuildPrompt(unittest.TestCase):
    """Test _build_prompt() method."""

    def setUp(self):
        self.enhancer = LLMEnhancer(mode="interactive")

    def test_basic_prompt_contains_dataset_info(self):
        prompt = self.enhancer._build_prompt(
            dataset_id="test/dataset",
            dataset_type="preference",
            sample_count=5000,
        )
        self.assertIn("test/dataset", prompt)
        self.assertIn("preference", prompt)
        self.assertIn("5000", prompt)

    def test_prompt_contains_schema_info(self):
        schema = {
            "question": {"type": "str"},
            "answer": {"type": "str", "nested_type": "list"},
        }
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            schema_info=schema,
        )
        self.assertIn("question", prompt)
        self.assertIn("answer", prompt)

    def test_prompt_truncates_long_values(self):
        long_text = "x" * 500
        sample_items = [{"text": long_text}]
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            sample_items=sample_items,
        )
        self.assertIn("...", prompt)

    def test_prompt_truncates_long_lists(self):
        sample_items = [{"tags": ["a", "b", "c", "d", "e"]}]
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            sample_items=sample_items,
        )
        # Should contain truncated list marker
        self.assertIn("...", prompt)

    def test_prompt_limits_to_3_samples(self):
        items = [{"q": f"question_{i}"} for i in range(10)]
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            sample_items=items,
        )
        # Only first 3 should appear
        self.assertIn("question_0", prompt)
        self.assertIn("question_2", prompt)
        self.assertNotIn("question_3", prompt)

    def test_prompt_with_complexity_metrics(self):
        metrics = StubComplexityMetrics(
            primary_domain=StubDomain("medical"),
            difficulty_score=7.5,
        )
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            complexity_metrics=metrics,
        )
        self.assertIn("medical", prompt)
        self.assertIn("7.5/10", prompt)

    def test_prompt_with_allocation(self):
        alloc = StubAllocation(human_work_percentage=65.0, total_cost=20000.0)
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            allocation=alloc,
        )
        self.assertIn("65.0", prompt)
        self.assertIn("20000.0", prompt)

    def test_prompt_with_rubrics_result(self):
        rubrics = StubRubricsResult(total_rubrics=50, unique_patterns=12)
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            rubrics_result=rubrics,
        )
        self.assertIn("50", prompt)
        self.assertIn("12", prompt)

    def test_prompt_with_llm_analysis(self):
        analysis = StubLLMAnalysis(purpose="Training reward models")
        prompt = self.enhancer._build_prompt(
            dataset_id="test/ds",
            llm_analysis=analysis,
        )
        self.assertIn("Training reward models", prompt)

    def test_prompt_no_extra_analysis(self):
        prompt = self.enhancer._build_prompt(dataset_id="test/ds")
        self.assertIn("无额外分析", prompt)


class TestGetPrompt(unittest.TestCase):
    """Test get_prompt() public API."""

    def test_get_prompt_returns_string(self):
        enhancer = LLMEnhancer(mode="interactive")
        prompt = enhancer.get_prompt(dataset_id="test/ds", dataset_type="sft")
        self.assertIsInstance(prompt, str)
        self.assertIn("test/ds", prompt)
        self.assertIn("sft", prompt)


# ==================== Response Parsing ====================


class TestParseResponse(unittest.TestCase):
    """Test _parse_response() with various input formats."""

    def setUp(self):
        self.enhancer = LLMEnhancer(mode="interactive")
        self.valid_json = json.dumps({
            "dataset_purpose_summary": "Test purpose",
            "key_methodology_insights": ["Insight 1", "Insight 2"],
            "reproduction_strategy": "Strategy here",
            "domain_specific_tips": ["Tip 1"],
            "tailored_use_cases": ["Use case 1"],
            "tailored_roi_scenarios": ["ROI 1"],
            "tailored_risks": [{"level": "中", "description": "Risk", "mitigation": "Fix"}],
            "competitive_positioning": "Strong position",
            "domain_specific_guidelines": "Guidelines",
            "quality_pitfalls": ["Pitfall 1"],
            "example_analysis": [],
            "phase_specific_risks": [],
            "team_recommendations": "Team rec",
            "realistic_sample_seeds": [],
        })

    def test_parse_json_in_markdown_fence(self):
        response = f"Here's the analysis:\n```json\n{self.valid_json}\n```\nThat's it."
        ctx = self.enhancer._parse_response(response)
        self.assertTrue(ctx.generated)
        self.assertEqual(ctx.dataset_purpose_summary, "Test purpose")
        self.assertEqual(len(ctx.key_methodology_insights), 2)

    def test_parse_raw_json(self):
        ctx = self.enhancer._parse_response(self.valid_json)
        self.assertTrue(ctx.generated)
        self.assertEqual(ctx.dataset_purpose_summary, "Test purpose")

    def test_parse_json_with_surrounding_text(self):
        response = f"Some text before\n{self.valid_json}\nSome text after"
        ctx = self.enhancer._parse_response(response)
        self.assertTrue(ctx.generated)

    def test_parse_invalid_json_returns_not_generated(self):
        ctx = self.enhancer._parse_response("This is not JSON at all")
        self.assertFalse(ctx.generated)
        self.assertIn("This is not JSON at all", ctx.raw_response)

    def test_parse_empty_string(self):
        ctx = self.enhancer._parse_response("")
        self.assertFalse(ctx.generated)

    def test_parse_partial_json(self):
        ctx = self.enhancer._parse_response('{"dataset_purpose_summary": "incomplete')
        self.assertFalse(ctx.generated)

    def test_parse_json_missing_fields_uses_defaults(self):
        minimal = json.dumps({"dataset_purpose_summary": "Only this field"})
        ctx = self.enhancer._parse_response(minimal)
        self.assertTrue(ctx.generated)
        self.assertEqual(ctx.dataset_purpose_summary, "Only this field")
        self.assertEqual(ctx.key_methodology_insights, [])

    def test_parse_preserves_all_fields(self):
        ctx = self.enhancer._parse_response(self.valid_json)
        self.assertEqual(ctx.competitive_positioning, "Strong position")
        self.assertEqual(ctx.domain_specific_guidelines, "Guidelines")
        self.assertEqual(ctx.team_recommendations, "Team rec")
        self.assertEqual(len(ctx.tailored_risks), 1)


class TestEnhanceFromResponse(unittest.TestCase):
    """Test enhance_from_response() public API."""

    def test_enhance_from_response_calls_parse(self):
        enhancer = LLMEnhancer(mode="interactive")
        data = json.dumps({"dataset_purpose_summary": "Test"})
        response = f"```json\n{data}\n```"
        ctx = enhancer.enhance_from_response(response)
        self.assertTrue(ctx.generated)
        self.assertEqual(ctx.dataset_purpose_summary, "Test")


# ==================== JSON File Loading ====================


class TestEnhanceFromJson(unittest.TestCase):
    """Test enhance_from_json() with file I/O."""

    def setUp(self):
        self.enhancer = LLMEnhancer(mode="from-json")

    def test_load_valid_json_file(self):
        data = {
            "dataset_purpose_summary": "From file",
            "key_methodology_insights": ["Insight"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            ctx = self.enhancer.enhance_from_json(tmp_path)
            self.assertTrue(ctx.generated)
            self.assertEqual(ctx.dataset_purpose_summary, "From file")
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent_file(self):
        ctx = self.enhancer.enhance_from_json("/nonexistent/path.json")
        self.assertFalse(ctx.generated)
        self.assertIn("Failed to load", ctx.raw_response)

    def test_load_invalid_json_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json content")
            tmp_path = f.name

        try:
            ctx = self.enhancer.enhance_from_json(tmp_path)
            self.assertFalse(ctx.generated)
        finally:
            os.unlink(tmp_path)

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            tmp_path = f.name

        try:
            ctx = self.enhancer.enhance_from_json(tmp_path)
            self.assertFalse(ctx.generated)
        finally:
            os.unlink(tmp_path)


# ==================== Mode Resolution ====================


class TestModeResolution(unittest.TestCase):
    """Test _resolve_mode() logic."""

    def test_explicit_mode_returned_directly(self):
        enhancer = LLMEnhancer(mode="interactive")
        self.assertEqual(enhancer._resolve_mode(), "interactive")

        enhancer = LLMEnhancer(mode="api")
        self.assertEqual(enhancer._resolve_mode(), "api")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_auto_with_anthropic_key_uses_api(self):
        enhancer = LLMEnhancer(mode="auto", provider="anthropic")
        self.assertEqual(enhancer._resolve_mode(), "api")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_auto_with_openai_key_uses_api(self):
        enhancer = LLMEnhancer(mode="auto", provider="openai")
        self.assertEqual(enhancer._resolve_mode(), "api")

    @patch.dict(os.environ, {}, clear=True)
    @patch("sys.stdin")
    def test_auto_no_key_piped_stdin_uses_interactive(self, mock_stdin):
        mock_stdin.isatty.return_value = False
        # Remove potential API keys
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            enhancer = LLMEnhancer(mode="auto", provider="anthropic")
            self.assertEqual(enhancer._resolve_mode(), "interactive")

    @patch.dict(os.environ, {}, clear=True)
    @patch("sys.stdin")
    def test_auto_no_key_tty_returns_none(self, mock_stdin):
        mock_stdin.isatty.return_value = True
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            enhancer = LLMEnhancer(mode="auto", provider="anthropic")
            self.assertEqual(enhancer._resolve_mode(), "none")


class TestEnhanceNoMode(unittest.TestCase):
    """Test enhance() when no mode is available."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("sys.stdin")
    def test_enhance_returns_not_generated_when_no_mode(self, mock_stdin):
        mock_stdin.isatty.return_value = True
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            enhancer = LLMEnhancer(mode="auto", provider="anthropic")
            ctx = enhancer.enhance(dataset_id="test/ds")
            self.assertFalse(ctx.generated)
            self.assertIn("No LLM mode", ctx.raw_response)


# ==================== Dict-to-Context Conversion ====================


class TestDictToContext(unittest.TestCase):
    """Test _dict_to_context() conversion."""

    def setUp(self):
        self.enhancer = LLMEnhancer()

    def test_full_dict(self):
        data = {
            "dataset_purpose_summary": "Purpose",
            "key_methodology_insights": ["I1", "I2"],
            "reproduction_strategy": "Strategy",
            "domain_specific_tips": ["T1"],
            "tailored_use_cases": ["U1"],
            "tailored_roi_scenarios": ["R1"],
            "tailored_risks": [{"level": "高"}],
            "competitive_positioning": "CP",
            "domain_specific_guidelines": "DG",
            "quality_pitfalls": ["QP1"],
            "example_analysis": [{"sample_index": 0}],
            "phase_specific_risks": [{"phase": "P1"}],
            "team_recommendations": "TR",
            "realistic_sample_seeds": [{"instruction": "I"}],
        }
        ctx = self.enhancer._dict_to_context(data)
        self.assertTrue(ctx.generated)
        self.assertEqual(ctx.dataset_purpose_summary, "Purpose")
        self.assertEqual(len(ctx.tailored_risks), 1)

    def test_empty_dict_uses_defaults(self):
        ctx = self.enhancer._dict_to_context({})
        self.assertTrue(ctx.generated)  # generated defaults to True in _dict_to_context
        self.assertEqual(ctx.dataset_purpose_summary, "")
        self.assertEqual(ctx.key_methodology_insights, [])

    def test_provider_set_from_enhancer(self):
        enhancer = LLMEnhancer(provider="openai")
        ctx = enhancer._dict_to_context({})
        self.assertEqual(ctx.llm_provider, "openai")

    def test_provider_overridden_by_data(self):
        ctx = self.enhancer._dict_to_context({"llm_provider": "custom"})
        self.assertEqual(ctx.llm_provider, "custom")


# ==================== Integration: get_prompt + enhance_from_response ====================


class TestGetPromptEnhanceFromResponsePattern(unittest.TestCase):
    """Test the complete get_prompt() → LLM → enhance_from_response() pattern."""

    def test_full_pattern(self):
        enhancer = LLMEnhancer(mode="interactive")

        # Step 1: Get prompt
        prompt = enhancer.get_prompt(
            dataset_id="Anthropic/hh-rlhf",
            dataset_type="preference",
            schema_info={"chosen": {"type": "str"}, "rejected": {"type": "str"}},
            sample_items=[{"chosen": "Good answer", "rejected": "Bad answer"}],
            sample_count=170000,
            domain="NLP",
            human_percentage=90.0,
            total_cost=50000.0,
        )
        self.assertIsInstance(prompt, str)
        self.assertIn("Anthropic/hh-rlhf", prompt)

        # Step 2: Simulate LLM response
        llm_response = """
Here's my analysis:

```json
{
  "dataset_purpose_summary": "This dataset contains human preference comparisons for RLHF training",
  "key_methodology_insights": ["Crowdsourced preferences", "Multi-turn conversations"],
  "reproduction_strategy": "Collect diverse prompts, generate response pairs, have annotators compare",
  "domain_specific_tips": ["Ensure diverse prompt categories"],
  "tailored_use_cases": ["Reward model training", "DPO fine-tuning"],
  "tailored_roi_scenarios": ["2x ROI through alignment improvement"],
  "tailored_risks": [{"level": "中", "description": "Annotator bias", "mitigation": "Training"}],
  "competitive_positioning": "Well-established benchmark dataset",
  "domain_specific_guidelines": "Focus on helpfulness and harmlessness",
  "quality_pitfalls": ["Inconsistent labeling"],
  "example_analysis": [{"sample_index": 0, "strengths": "Clear", "weaknesses": "Short"}],
  "phase_specific_risks": [{"phase": "Pilot", "risk": "Low agreement", "mitigation": "Calibration"}],
  "team_recommendations": "5 experienced annotators with NLP background",
  "realistic_sample_seeds": [{"instruction": "Compare these responses", "difficulty": "medium"}]
}
```
"""

        # Step 3: Parse response
        ctx = enhancer.enhance_from_response(llm_response)
        self.assertTrue(ctx.generated)
        self.assertIn("RLHF", ctx.dataset_purpose_summary)
        self.assertEqual(len(ctx.key_methodology_insights), 2)
        self.assertEqual(len(ctx.tailored_use_cases), 2)
        self.assertEqual(ctx.team_recommendations, "5 experienced annotators with NLP background")


if __name__ == "__main__":
    unittest.main()
