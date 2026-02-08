"""Unit tests for cost_calculator.py — CostCalculator, EnhancedCostCalculator, and data classes.

Covers:
- CostEstimate: __str__ formatting, equal and ranged values
- CostBreakdown: to_dict() serialization
- CostCalculator.calculate(): all parameter combinations, input validation
- CostCalculator._calculate_api_cost(): known/unknown models, retries
- CostCalculator._calculate_human_cost(): all annotation types, zero/missing
- CostCalculator._calculate_compute_cost(): all compute types, zero/missing
- CostCalculator._match_model(): all fuzzy matching branches
- CostCalculator._estimate_tokens(): all generation method types, fallback
- CostCalculator._infer_annotation_type(): all method types, fallback by ratio
- CostCalculator._estimate_compute(): training, embedding, large dataset, small dataset
- CostCalculator.estimate_from_recipe(): full integration with Recipe
- CostCalculator.get_model_pricing() / list_models()
- CostCalculator.format_cost_report(): report formatting
- EnhancedCostBreakdown: to_dict() serialization
- EnhancedCostCalculator.calculate_labor_cost(): regions, experience levels, annotation types
- EnhancedCostCalculator.calculate_enhanced_cost(): full integration
- EnhancedCostCalculator.calculate_roi(): positive, negative, zero, missing comparables
- EnhancedCostCalculator.format_enhanced_report(): report formatting
- Module-level constants: LABOR_COST_RATES, ANNOTATION_TIME_ESTIMATES
"""

import unittest

from datarecipe.constants import REGION_COST_MULTIPLIERS
from datarecipe.cost_calculator import (
    ANNOTATION_TIME_ESTIMATES,
    LABOR_COST_RATES,
    CostBreakdown,
    CostCalculator,
    CostEstimate,
    EnhancedCostBreakdown,
    EnhancedCostCalculator,
    TokenPricing,
)
from datarecipe.schema import GenerationMethod, Recipe


# =============================================================================
# Helpers
# =============================================================================


def _make_recipe(**kwargs) -> Recipe:
    """Helper to create a Recipe with sensible defaults."""
    defaults = {
        "name": "test-dataset",
        "languages": ["en"],
        "tags": [],
        "description": "",
    }
    defaults.update(kwargs)
    return Recipe(**defaults)


def _make_calculator() -> CostCalculator:
    return CostCalculator()


def _make_enhanced_calculator() -> EnhancedCostCalculator:
    return EnhancedCostCalculator()


# =============================================================================
# CostEstimate tests
# =============================================================================


class TestCostEstimate(unittest.TestCase):
    """Tests for the CostEstimate dataclass."""

    def test_str_equal_low_high(self):
        """When low == high, should show single value."""
        est = CostEstimate(low=10.0, high=10.0, expected=10.0)
        self.assertEqual(str(est), "$10.00")

    def test_str_range(self):
        """When low != high, should show range."""
        est = CostEstimate(low=5.50, high=15.75, expected=10.0)
        self.assertEqual(str(est), "$5.50 - $15.75")

    def test_str_zero(self):
        """Zero cost should display properly."""
        est = CostEstimate(low=0, high=0, expected=0)
        self.assertEqual(str(est), "$0.00")

    def test_str_large_values(self):
        """Large values should include commas."""
        est = CostEstimate(low=1000.0, high=5000.0, expected=3000.0)
        self.assertEqual(str(est), "$1,000.00 - $5,000.00")

    def test_default_unit(self):
        est = CostEstimate(low=0, high=0, expected=0)
        self.assertEqual(est.unit, "USD")


# =============================================================================
# CostBreakdown tests
# =============================================================================


class TestCostBreakdown(unittest.TestCase):
    """Tests for the CostBreakdown dataclass."""

    def test_to_dict_structure(self):
        bd = CostBreakdown(
            api_cost=CostEstimate(1.0, 2.0, 1.5),
            human_annotation_cost=CostEstimate(3.0, 6.0, 4.5),
            compute_cost=CostEstimate(0.5, 1.0, 0.75),
            total=CostEstimate(4.5, 9.0, 6.75),
            assumptions=["assumption1"],
            details={"key": "value"},
        )
        d = bd.to_dict()

        self.assertEqual(d["api_cost"]["low"], 1.0)
        self.assertEqual(d["api_cost"]["high"], 2.0)
        self.assertEqual(d["api_cost"]["expected"], 1.5)

        self.assertEqual(d["human_annotation_cost"]["low"], 3.0)
        self.assertEqual(d["compute_cost"]["expected"], 0.75)
        self.assertEqual(d["total"]["expected"], 6.75)

        self.assertEqual(d["assumptions"], ["assumption1"])
        self.assertEqual(d["details"], {"key": "value"})

    def test_to_dict_empty_defaults(self):
        bd = CostBreakdown(
            api_cost=CostEstimate(0, 0, 0),
            human_annotation_cost=CostEstimate(0, 0, 0),
            compute_cost=CostEstimate(0, 0, 0),
            total=CostEstimate(0, 0, 0),
        )
        d = bd.to_dict()
        self.assertEqual(d["assumptions"], [])
        self.assertEqual(d["details"], {})


# =============================================================================
# EnhancedCostBreakdown tests
# =============================================================================


class TestEnhancedCostBreakdown(unittest.TestCase):
    """Tests for the EnhancedCostBreakdown dataclass."""

    def test_to_dict_structure(self):
        bd = EnhancedCostBreakdown(
            api_cost=CostEstimate(1.0, 2.0, 1.5),
            labor_cost=CostEstimate(10.0, 20.0, 15.0),
            compute_cost=CostEstimate(0.5, 1.0, 0.75),
            total=CostEstimate(11.5, 23.0, 17.25),
            labor_breakdown={"annotation": 8.0, "review": 5.0, "project_management": 2.0},
            region="cn",
            region_multiplier=0.4,
            assumptions=["test assumption"],
            details={"num_examples": 1000},
        )
        d = bd.to_dict()

        self.assertEqual(d["api_cost"]["low"], 1.0)
        self.assertEqual(d["labor_cost"]["expected"], 15.0)
        self.assertEqual(d["labor_cost"]["breakdown"]["annotation"], 8.0)
        self.assertEqual(d["region"], "cn")
        self.assertEqual(d["region_multiplier"], 0.4)
        self.assertEqual(d["assumptions"], ["test assumption"])
        self.assertEqual(d["details"]["num_examples"], 1000)

    def test_to_dict_defaults(self):
        bd = EnhancedCostBreakdown(
            api_cost=CostEstimate(0, 0, 0),
            labor_cost=CostEstimate(0, 0, 0),
            compute_cost=CostEstimate(0, 0, 0),
            total=CostEstimate(0, 0, 0),
        )
        d = bd.to_dict()
        self.assertEqual(d["region"], "us")
        self.assertEqual(d["region_multiplier"], 1.0)
        self.assertEqual(d["labor_cost"]["breakdown"], {})


# =============================================================================
# CostCalculator — input validation
# =============================================================================


class TestCalculateValidation(unittest.TestCase):
    """Tests for input validation in CostCalculator.calculate()."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_negative_num_examples(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=-1)

    def test_zero_num_examples(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=0)

    def test_negative_input_tokens(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, avg_input_tokens=-10)

    def test_zero_input_tokens(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, avg_input_tokens=0)

    def test_negative_output_tokens(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, avg_output_tokens=-5)

    def test_zero_output_tokens(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, avg_output_tokens=0)

    def test_annotation_ratio_too_high(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, human_annotation_ratio=1.5)

    def test_annotation_ratio_negative(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, human_annotation_ratio=-0.1)

    def test_negative_compute_hours(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, compute_hours=-1.0)

    def test_retries_below_one(self):
        with self.assertRaises(ValueError):
            self.calc.calculate(num_examples=100, retries=0.5)


# =============================================================================
# CostCalculator.calculate() — happy path
# =============================================================================


class TestCalculateHappyPath(unittest.TestCase):
    """Tests for CostCalculator.calculate() with valid inputs."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_api_only_basic(self):
        """Simple API-only cost calculation."""
        result = self.calc.calculate(num_examples=1000, model="gpt-4o")
        self.assertIsInstance(result, CostBreakdown)
        self.assertGreater(result.api_cost.expected, 0)
        self.assertEqual(result.human_annotation_cost.expected, 0)
        self.assertEqual(result.compute_cost.expected, 0)
        self.assertEqual(result.total.expected, result.api_cost.expected)

    def test_api_cost_range(self):
        """API cost should have low < expected < high."""
        result = self.calc.calculate(num_examples=1000, model="gpt-4o")
        self.assertLess(result.api_cost.low, result.api_cost.expected)
        self.assertLess(result.api_cost.expected, result.api_cost.high)

    def test_with_human_annotation(self):
        """Cost with human annotation included."""
        result = self.calc.calculate(
            num_examples=1000,
            human_annotation_type="text_classification",
            human_annotation_ratio=0.5,
        )
        self.assertGreater(result.human_annotation_cost.expected, 0)
        expected_total = (
            result.api_cost.expected
            + result.human_annotation_cost.expected
            + result.compute_cost.expected
        )
        self.assertAlmostEqual(result.total.expected, expected_total, places=5)

    def test_with_compute(self):
        """Cost with compute resources included."""
        result = self.calc.calculate(
            num_examples=1000,
            compute_type="gpu_a100",
            compute_hours=10.0,
        )
        self.assertGreater(result.compute_cost.expected, 0)

    def test_with_all_components(self):
        """Cost with all three components."""
        result = self.calc.calculate(
            num_examples=5000,
            model="claude-3-sonnet",
            avg_input_tokens=600,
            avg_output_tokens=300,
            human_annotation_type="expert_annotation",
            human_annotation_ratio=0.3,
            compute_type="gpu_h100",
            compute_hours=5.0,
            retries=1.5,
        )
        self.assertGreater(result.api_cost.expected, 0)
        self.assertGreater(result.human_annotation_cost.expected, 0)
        self.assertGreater(result.compute_cost.expected, 0)

        total = (
            result.api_cost.expected
            + result.human_annotation_cost.expected
            + result.compute_cost.expected
        )
        self.assertAlmostEqual(result.total.expected, total, places=5)

    def test_assumptions_populated(self):
        """Assumptions list should be populated."""
        result = self.calc.calculate(
            num_examples=100,
            model="gpt-4o",
            human_annotation_type="simple_label",
            human_annotation_ratio=0.1,
            compute_type="cpu_standard",
            compute_hours=1.0,
        )
        # Should contain model, tokens, retry, annotation, compute assumptions
        self.assertGreaterEqual(len(result.assumptions), 4)

    def test_details_populated(self):
        """Details dict should have expected keys."""
        result = self.calc.calculate(num_examples=500, avg_input_tokens=300, avg_output_tokens=100)
        self.assertEqual(result.details["num_examples"], 500)
        self.assertEqual(result.details["avg_input_tokens"], 300)
        self.assertEqual(result.details["avg_output_tokens"], 100)
        self.assertEqual(result.details["total_tokens"], 500 * 400)

    def test_boundary_annotation_ratio_zero(self):
        """annotation_ratio=0.0 should yield zero human cost even with type set."""
        result = self.calc.calculate(
            num_examples=100,
            human_annotation_type="expert_annotation",
            human_annotation_ratio=0.0,
        )
        self.assertEqual(result.human_annotation_cost.expected, 0)

    def test_boundary_annotation_ratio_one(self):
        """annotation_ratio=1.0 should annotate all examples."""
        result = self.calc.calculate(
            num_examples=100,
            human_annotation_type="simple_label",
            human_annotation_ratio=1.0,
        )
        self.assertGreater(result.human_annotation_cost.expected, 0)


# =============================================================================
# CostCalculator._calculate_api_cost()
# =============================================================================


class TestCalculateApiCost(unittest.TestCase):
    """Tests for _calculate_api_cost() internal method."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_known_model_gpt4o(self):
        """Known model should use its pricing."""
        cost = self.calc._calculate_api_cost(1000, "gpt-4o", 500, 200, 1.0)
        pricing = self.calc.LLM_PRICING["gpt-4o"]
        expected_input = (1000 * 500 / 1000) * pricing.input_price_per_1k
        expected_output = (1000 * 200 / 1000) * pricing.output_price_per_1k
        expected = expected_input + expected_output
        self.assertAlmostEqual(cost.expected, expected, places=5)

    def test_unknown_model_defaults_to_gpt4o(self):
        """Unknown model should fall back to gpt-4o pricing."""
        cost_unknown = self.calc._calculate_api_cost(1000, "unknown-model", 500, 200, 1.0)
        cost_gpt4o = self.calc._calculate_api_cost(1000, "gpt-4o", 500, 200, 1.0)
        self.assertAlmostEqual(cost_unknown.expected, cost_gpt4o.expected, places=5)

    def test_retries_factor(self):
        """Retry factor should multiply token counts."""
        cost_no_retry = self.calc._calculate_api_cost(1000, "gpt-4o", 500, 200, 1.0)
        cost_with_retry = self.calc._calculate_api_cost(1000, "gpt-4o", 500, 200, 2.0)
        self.assertAlmostEqual(cost_with_retry.expected, cost_no_retry.expected * 2.0, places=5)

    def test_low_high_range(self):
        """Low should be 0.8x expected, high should be 1.5x expected."""
        cost = self.calc._calculate_api_cost(1000, "gpt-4o", 500, 200, 1.2)
        self.assertAlmostEqual(cost.low, cost.expected * 0.8, places=5)
        self.assertAlmostEqual(cost.high, cost.expected * 1.5, places=5)

    def test_all_known_models(self):
        """All models in LLM_PRICING should produce non-negative costs."""
        for model_name in self.calc.LLM_PRICING:
            cost = self.calc._calculate_api_cost(100, model_name, 100, 100, 1.0)
            self.assertGreaterEqual(cost.expected, 0, f"Failed for model {model_name}")


# =============================================================================
# CostCalculator._calculate_human_cost()
# =============================================================================


class TestCalculateHumanCost(unittest.TestCase):
    """Tests for _calculate_human_cost() internal method."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_no_annotation_type(self):
        """None annotation type should return zero cost."""
        cost = self.calc._calculate_human_cost(1000, None, 0.5)
        self.assertEqual(cost.expected, 0)

    def test_zero_ratio(self):
        """Zero ratio should return zero cost."""
        cost = self.calc._calculate_human_cost(1000, "text_classification", 0.0)
        self.assertEqual(cost.expected, 0)

    def test_known_annotation_type(self):
        """Known annotation type should use its cost."""
        cost = self.calc._calculate_human_cost(1000, "simple_label", 0.5)
        expected = 1000 * 0.5 * 0.05  # 500 * $0.05
        self.assertAlmostEqual(cost.expected, expected, places=5)

    def test_unknown_annotation_type_defaults(self):
        """Unknown annotation type should default to text_classification cost."""
        cost = self.calc._calculate_human_cost(1000, "unknown_type", 1.0)
        expected = 1000 * 1.0 * 0.10  # text_classification default
        self.assertAlmostEqual(cost.expected, expected, places=5)

    def test_low_high_range(self):
        """Low should be 0.7x, high should be 1.5x."""
        cost = self.calc._calculate_human_cost(1000, "text_generation", 1.0)
        self.assertAlmostEqual(cost.low, cost.expected * 0.7, places=5)
        self.assertAlmostEqual(cost.high, cost.expected * 1.5, places=5)

    def test_expert_annotation_cost(self):
        """Expert annotation should use $5.00 per annotation."""
        cost = self.calc._calculate_human_cost(100, "expert_annotation", 1.0)
        expected = 100 * 5.00
        self.assertAlmostEqual(cost.expected, expected, places=5)

    def test_preference_ranking_cost(self):
        cost = self.calc._calculate_human_cost(200, "preference_ranking", 0.5)
        expected = 200 * 0.5 * 0.30
        self.assertAlmostEqual(cost.expected, expected, places=5)


# =============================================================================
# CostCalculator._calculate_compute_cost()
# =============================================================================


class TestCalculateComputeCost(unittest.TestCase):
    """Tests for _calculate_compute_cost() internal method."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_no_compute_type(self):
        cost = self.calc._calculate_compute_cost(None, 10.0)
        self.assertEqual(cost.expected, 0)

    def test_zero_hours(self):
        cost = self.calc._calculate_compute_cost("gpu_a100", 0.0)
        self.assertEqual(cost.expected, 0)

    def test_known_compute_type(self):
        cost = self.calc._calculate_compute_cost("gpu_a100", 10.0)
        expected = 10.0 * 3.00
        self.assertAlmostEqual(cost.expected, expected, places=5)

    def test_unknown_compute_type_defaults(self):
        """Unknown compute type should default to cpu_standard."""
        cost = self.calc._calculate_compute_cost("unknown_gpu", 10.0)
        expected = 10.0 * 0.05  # cpu_standard
        self.assertAlmostEqual(cost.expected, expected, places=5)

    def test_low_high_range(self):
        cost = self.calc._calculate_compute_cost("gpu_h100", 5.0)
        self.assertAlmostEqual(cost.low, cost.expected * 0.8, places=5)
        self.assertAlmostEqual(cost.high, cost.expected * 1.3, places=5)

    def test_all_compute_types(self):
        for ct, rate in self.calc.COMPUTE_COSTS.items():
            cost = self.calc._calculate_compute_cost(ct, 1.0)
            self.assertAlmostEqual(cost.expected, rate, places=5, msg=f"Failed for {ct}")


# =============================================================================
# CostCalculator._match_model()
# =============================================================================


class TestMatchModel(unittest.TestCase):
    """Tests for _match_model() fuzzy matching."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_direct_match(self):
        self.assertEqual(self.calc._match_model("gpt-4o"), "gpt-4o")
        self.assertEqual(self.calc._match_model("claude-3-haiku"), "claude-3-haiku")

    def test_case_insensitive(self):
        self.assertEqual(self.calc._match_model("GPT-4o"), "gpt-4o")
        self.assertEqual(self.calc._match_model("Claude-3-Opus"), "claude-3-opus")

    def test_gpt4_mini(self):
        self.assertEqual(self.calc._match_model("gpt-4o-mini-2024"), "gpt-4o-mini")
        self.assertEqual(self.calc._match_model("GPT-4-mini"), "gpt-4o-mini")

    def test_gpt4_general(self):
        self.assertEqual(self.calc._match_model("gpt-4-turbo-2024"), "gpt-4o")

    def test_gpt35(self):
        self.assertEqual(self.calc._match_model("gpt-3.5-turbo-16k"), "gpt-3.5-turbo")
        self.assertEqual(self.calc._match_model("gpt-35-turbo"), "gpt-3.5-turbo")

    def test_claude_opus(self):
        self.assertEqual(self.calc._match_model("claude-3-opus-20240229"), "claude-3-opus")

    def test_claude_haiku(self):
        self.assertEqual(self.calc._match_model("claude-3-haiku-20240307"), "claude-3-haiku")

    def test_claude_generic(self):
        """Generic claude name should map to claude-3-sonnet."""
        self.assertEqual(self.calc._match_model("claude-2"), "claude-3-sonnet")

    def test_gemini_15(self):
        self.assertEqual(self.calc._match_model("gemini-1.5-pro-latest"), "gemini-1.5-pro")

    def test_gemini_generic(self):
        self.assertEqual(self.calc._match_model("gemini-pro-vision"), "gemini-pro")

    def test_llama_70b(self):
        self.assertEqual(self.calc._match_model("llama-3.1-70b-instruct"), "llama-3-70b")

    def test_llama_generic(self):
        """Generic llama should map to llama-3-8b."""
        self.assertEqual(self.calc._match_model("llama-3-8b-chat"), "llama-3-8b")

    def test_mixtral(self):
        self.assertEqual(self.calc._match_model("mixtral-8x22b"), "mixtral-8x7b")

    def test_completely_unknown(self):
        """Completely unknown model should default to gpt-4o."""
        self.assertEqual(self.calc._match_model("some-random-model"), "gpt-4o")


# =============================================================================
# CostCalculator._estimate_tokens()
# =============================================================================


class TestEstimateTokens(unittest.TestCase):
    """Tests for _estimate_tokens() based on recipe generation methods."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_distillation(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="distillation")]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (600, 300))

    def test_instruction(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="instruction_following")]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (400, 200))

    def test_conversation(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="conversation_generation")]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (800, 400))

    def test_chat(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="multi_chat")]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (800, 400))

    def test_code(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="code_generation")]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (500, 500))

    def test_summarization(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="summarization")]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (1000, 200))

    def test_summary_variant(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="text_summary")]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (1000, 200))

    def test_fallback_high_synthetic(self):
        """High synthetic ratio should default to (500, 200)."""
        recipe = _make_recipe(synthetic_ratio=0.8, generation_methods=[])
        self.assertEqual(self.calc._estimate_tokens(recipe), (500, 200))

    def test_fallback_low_synthetic(self):
        """Low synthetic ratio should default to (300, 150)."""
        recipe = _make_recipe(synthetic_ratio=0.3, generation_methods=[])
        self.assertEqual(self.calc._estimate_tokens(recipe), (300, 150))

    def test_fallback_no_synthetic_ratio(self):
        """None synthetic ratio should default to (300, 150)."""
        recipe = _make_recipe(synthetic_ratio=None, generation_methods=[])
        self.assertEqual(self.calc._estimate_tokens(recipe), (300, 150))

    def test_first_method_wins(self):
        """Should match on the first generation method."""
        recipe = _make_recipe(
            generation_methods=[
                GenerationMethod(method_type="code_generation"),
                GenerationMethod(method_type="distillation"),
            ]
        )
        self.assertEqual(self.calc._estimate_tokens(recipe), (500, 500))


# =============================================================================
# CostCalculator._infer_annotation_type()
# =============================================================================


class TestInferAnnotationType(unittest.TestCase):
    """Tests for _infer_annotation_type()."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_no_human_ratio(self):
        recipe = _make_recipe(human_ratio=None)
        self.assertIsNone(self.calc._infer_annotation_type(recipe))

    def test_zero_human_ratio(self):
        recipe = _make_recipe(human_ratio=0.0)
        self.assertIsNone(self.calc._infer_annotation_type(recipe))

    def test_preference(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="preference_ranking")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "preference_ranking")

    def test_rlhf(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="rlhf_collection")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "preference_ranking")

    def test_expert(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="expert_review")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "expert_annotation")

    def test_verification(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="human_verification")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "quality_check")

    def test_check(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="quality_check")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "quality_check")

    def test_generation(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="text_generation")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "text_generation")

    def test_writing(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="creative_writing")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "text_generation")

    def test_classification(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="text_classification")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "text_classification")

    def test_label(self):
        recipe = _make_recipe(
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="multi_label")],
        )
        self.assertEqual(self.calc._infer_annotation_type(recipe), "text_classification")

    def test_fallback_high_human_ratio(self):
        """High human_ratio without method hint should default to text_generation."""
        recipe = _make_recipe(human_ratio=0.6, generation_methods=[])
        self.assertEqual(self.calc._infer_annotation_type(recipe), "text_generation")

    def test_fallback_low_human_ratio(self):
        """Low human_ratio without method hint should default to quality_check."""
        recipe = _make_recipe(human_ratio=0.3, generation_methods=[])
        self.assertEqual(self.calc._infer_annotation_type(recipe), "quality_check")


# =============================================================================
# CostCalculator._estimate_compute()
# =============================================================================


class TestEstimateCompute(unittest.TestCase):
    """Tests for _estimate_compute()."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_training_method(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="finetune_model")]
        )
        compute_type, hours = self.calc._estimate_compute(recipe, 10000)
        self.assertEqual(compute_type, "gpu_a100")
        self.assertAlmostEqual(hours, 4.0, places=2)

    def test_training_scaled(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="train_classifier")]
        )
        compute_type, hours = self.calc._estimate_compute(recipe, 50000)
        self.assertEqual(compute_type, "gpu_a100")
        self.assertAlmostEqual(hours, 20.0, places=2)

    def test_embedding_method(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="semantic_search")]
        )
        compute_type, hours = self.calc._estimate_compute(recipe, 100000)
        self.assertEqual(compute_type, "gpu_t4")
        self.assertAlmostEqual(hours, 1.0, places=2)

    def test_embedding_method_variant(self):
        recipe = _make_recipe(
            generation_methods=[GenerationMethod(method_type="embedding_generation")]
        )
        compute_type, hours = self.calc._estimate_compute(recipe, 200000)
        self.assertEqual(compute_type, "gpu_t4")
        self.assertAlmostEqual(hours, 2.0, places=2)

    def test_large_dataset_no_special_method(self):
        """Large datasets (>100k) should get cpu_high."""
        recipe = _make_recipe(generation_methods=[])
        compute_type, hours = self.calc._estimate_compute(recipe, 200000)
        self.assertEqual(compute_type, "cpu_high")
        self.assertAlmostEqual(hours, 1.0, places=2)

    def test_small_dataset_no_special_method(self):
        """Small datasets should need no compute."""
        recipe = _make_recipe(generation_methods=[])
        compute_type, hours = self.calc._estimate_compute(recipe, 1000)
        self.assertIsNone(compute_type)
        self.assertEqual(hours, 0.0)

    def test_boundary_100k(self):
        """Exactly 100k should NOT trigger cpu_high (need >100k)."""
        recipe = _make_recipe(generation_methods=[])
        compute_type, hours = self.calc._estimate_compute(recipe, 100000)
        self.assertIsNone(compute_type)
        self.assertEqual(hours, 0.0)


# =============================================================================
# CostCalculator.estimate_from_recipe()
# =============================================================================


class TestEstimateFromRecipe(unittest.TestCase):
    """Tests for estimate_from_recipe() integration."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_basic_recipe(self):
        recipe = _make_recipe(num_examples=5000)
        result = self.calc.estimate_from_recipe(recipe)
        self.assertIsInstance(result, CostBreakdown)
        self.assertEqual(result.details["num_examples"], 5000)

    def test_default_num_examples(self):
        """Without num_examples, should default to 10000."""
        recipe = _make_recipe(num_examples=None)
        result = self.calc.estimate_from_recipe(recipe)
        self.assertEqual(result.details["num_examples"], 10000)

    def test_target_size_override(self):
        recipe = _make_recipe(num_examples=5000)
        result = self.calc.estimate_from_recipe(recipe, target_size=2000)
        self.assertEqual(result.details["num_examples"], 2000)

    def test_model_auto_detection(self):
        """Should auto-detect model from teacher_models."""
        recipe = _make_recipe(teacher_models=["claude-3-opus-20240229"])
        result = self.calc.estimate_from_recipe(recipe)
        # Should have inferred model assumption
        has_inferred = any("Inferred model" in a for a in result.assumptions)
        self.assertTrue(has_inferred)

    def test_model_explicit(self):
        """Explicit model should override auto-detection."""
        recipe = _make_recipe(teacher_models=["gpt-4"])
        result = self.calc.estimate_from_recipe(recipe, model="claude-3-haiku")
        self.assertEqual(result.details["model"], "claude-3-haiku")

    def test_no_teacher_models_defaults_gpt4o(self):
        """Without teacher_models, should default to gpt-4o."""
        recipe = _make_recipe(teacher_models=[])
        result = self.calc.estimate_from_recipe(recipe)
        has_inferred = any("gpt-4o" in a for a in result.assumptions)
        self.assertTrue(has_inferred)

    def test_with_human_ratio(self):
        """Recipe with human_ratio should include human annotation cost."""
        recipe = _make_recipe(
            num_examples=1000,
            human_ratio=0.5,
            generation_methods=[GenerationMethod(method_type="text_generation")],
        )
        result = self.calc.estimate_from_recipe(recipe)
        self.assertGreater(result.human_annotation_cost.expected, 0)

    def test_with_compute_intensive_method(self):
        """Recipe with training method should include compute cost."""
        recipe = _make_recipe(
            num_examples=10000,
            generation_methods=[GenerationMethod(method_type="finetune_model")],
        )
        result = self.calc.estimate_from_recipe(recipe)
        self.assertGreater(result.compute_cost.expected, 0)

    def test_assumptions_include_target_size(self):
        recipe = _make_recipe(num_examples=3000)
        result = self.calc.estimate_from_recipe(recipe)
        has_target = any("3,000" in a for a in result.assumptions)
        self.assertTrue(has_target)


# =============================================================================
# CostCalculator.get_model_pricing() / list_models()
# =============================================================================


class TestModelPricingAndListing(unittest.TestCase):
    """Tests for get_model_pricing() and list_models()."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_get_known_model(self):
        pricing = self.calc.get_model_pricing("gpt-4o")
        self.assertIsInstance(pricing, TokenPricing)
        self.assertEqual(pricing.provider, "openai")
        self.assertEqual(pricing.model, "gpt-4o")
        self.assertGreater(pricing.input_price_per_1k, 0)

    def test_get_unknown_model(self):
        pricing = self.calc.get_model_pricing("nonexistent-model")
        self.assertIsNone(pricing)

    def test_list_models(self):
        models = self.calc.list_models()
        self.assertIsInstance(models, list)
        self.assertIn("gpt-4o", models)
        self.assertIn("claude-3-sonnet", models)
        self.assertIn("gemini-pro", models)
        self.assertEqual(len(models), len(self.calc.LLM_PRICING))


# =============================================================================
# CostCalculator.format_cost_report()
# =============================================================================


class TestFormatCostReport(unittest.TestCase):
    """Tests for format_cost_report()."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_report_structure(self):
        breakdown = self.calc.calculate(num_examples=1000)
        report = self.calc.format_cost_report(breakdown)

        self.assertIn("COST ESTIMATION REPORT", report)
        self.assertIn("BREAKDOWN:", report)
        self.assertIn("API Costs:", report)
        self.assertIn("Human Annotation:", report)
        self.assertIn("Compute:", report)
        self.assertIn("TOTAL:", report)
        self.assertIn("ASSUMPTIONS:", report)
        self.assertIn("DETAILS:", report)

    def test_report_with_large_values(self):
        """Large integer values should be comma-formatted."""
        breakdown = CostBreakdown(
            api_cost=CostEstimate(100, 200, 150),
            human_annotation_cost=CostEstimate(0, 0, 0),
            compute_cost=CostEstimate(0, 0, 0),
            total=CostEstimate(100, 200, 150),
            details={"total_tokens": 5000000, "model": "gpt-4o"},
        )
        report = self.calc.format_cost_report(breakdown)
        self.assertIn("5,000,000", report)
        self.assertIn("model: gpt-4o", report)

    def test_report_with_empty_details(self):
        """Report should handle empty details gracefully."""
        breakdown = CostBreakdown(
            api_cost=CostEstimate(0, 0, 0),
            human_annotation_cost=CostEstimate(0, 0, 0),
            compute_cost=CostEstimate(0, 0, 0),
            total=CostEstimate(0, 0, 0),
            details={},
        )
        report = self.calc.format_cost_report(breakdown)
        # Should NOT have DETAILS section content (but no crash)
        self.assertNotIn("DETAILS:\n  ", report)


# =============================================================================
# Module-level constants
# =============================================================================


class TestModuleConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_labor_cost_rates_structure(self):
        for level in ("junior", "mid", "senior", "expert"):
            self.assertIn(level, LABOR_COST_RATES)
            rates = LABOR_COST_RATES[level]
            self.assertIn("base", rates)
            self.assertIn("annotation", rates)
            self.assertIn("review", rates)

    def test_annotation_time_estimates_structure(self):
        expected_types = [
            "simple_label", "text_classification", "preference_ranking",
            "text_generation", "complex_annotation", "expert_annotation",
            "code_review", "quality_check",
        ]
        for t in expected_types:
            self.assertIn(t, ANNOTATION_TIME_ESTIMATES)
            self.assertGreater(ANNOTATION_TIME_ESTIMATES[t], 0)

    def test_labor_rates_ordering(self):
        """More experienced annotators should cost more."""
        levels = ["junior", "mid", "senior", "expert"]
        for key in ("base", "annotation", "review"):
            values = [LABOR_COST_RATES[l][key] for l in levels]
            self.assertEqual(values, sorted(values), f"Rates for {key} should increase")


# =============================================================================
# EnhancedCostCalculator.calculate_labor_cost()
# =============================================================================


class TestCalculateLaborCost(unittest.TestCase):
    """Tests for EnhancedCostCalculator.calculate_labor_cost()."""

    def setUp(self):
        self.calc = _make_enhanced_calculator()

    def test_basic_labor_cost(self):
        result = self.calc.calculate_labor_cost(
            num_examples=1000,
            annotation_type="text_classification",
            experience_level="mid",
            region="us",
        )
        self.assertIn("annotation_cost", result)
        self.assertIn("review_cost", result)
        self.assertIn("pm_cost", result)
        self.assertIn("total", result)
        self.assertIn("details", result)
        self.assertGreater(result["total"], 0)

    def test_annotation_cost_calculation(self):
        """Verify annotation cost formula."""
        result = self.calc.calculate_labor_cost(
            num_examples=60,  # 60 items * 2 min = 120 min = 2 hours
            annotation_type="text_classification",
            experience_level="mid",
            region="us",
            review_rate=0.0,
        )
        # 60 items * 2 min = 120 min = 2 hours
        # 2 hours * $20/hr * 1.0 (us) = $40
        expected_annotation = 40.0
        self.assertAlmostEqual(result["annotation_cost"], expected_annotation, places=2)

    def test_review_cost_included(self):
        """Review items should be calculated as ratio * 1.5x time."""
        result = self.calc.calculate_labor_cost(
            num_examples=100,
            annotation_type="simple_label",
            experience_level="mid",
            region="us",
            review_rate=0.5,
        )
        self.assertGreater(result["review_cost"], 0)

    def test_pm_cost_is_10_percent(self):
        """PM cost should be 10% of annotation + review."""
        result = self.calc.calculate_labor_cost(
            num_examples=100,
            annotation_type="text_classification",
            experience_level="mid",
            region="us",
        )
        expected_pm = round((result["annotation_cost"] + result["review_cost"]) * 0.1, 2)
        self.assertAlmostEqual(result["pm_cost"], expected_pm, places=2)

    def test_total_is_sum(self):
        result = self.calc.calculate_labor_cost(
            num_examples=500,
            annotation_type="expert_annotation",
            experience_level="senior",
            region="eu",
        )
        expected_total = result["annotation_cost"] + result["review_cost"] + result["pm_cost"]
        self.assertAlmostEqual(result["total"], expected_total, places=1)

    def test_region_multiplier_cn(self):
        """China region should have lower costs."""
        result_us = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="text_classification", region="us"
        )
        result_cn = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="text_classification", region="cn"
        )
        self.assertGreater(result_us["total"], result_cn["total"])

    def test_region_multiplier_india(self):
        result_us = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="text_classification", region="us"
        )
        result_in = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="text_classification", region="in"
        )
        self.assertGreater(result_us["total"], result_in["total"])

    def test_unknown_region_defaults_to_1(self):
        """Unknown region should use multiplier 1.0."""
        result_us = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="text_classification", region="us"
        )
        result_unknown = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="text_classification", region="unknown_region"
        )
        self.assertAlmostEqual(result_us["total"], result_unknown["total"], places=2)

    def test_experience_levels_affect_cost(self):
        """Higher experience should cost more."""
        costs = {}
        for level in ("junior", "mid", "senior", "expert"):
            result = self.calc.calculate_labor_cost(
                num_examples=100,
                annotation_type="text_classification",
                experience_level=level,
                region="us",
            )
            costs[level] = result["total"]
        self.assertLess(costs["junior"], costs["mid"])
        self.assertLess(costs["mid"], costs["senior"])
        self.assertLess(costs["senior"], costs["expert"])

    def test_unknown_experience_defaults_to_mid(self):
        result_mid = self.calc.calculate_labor_cost(
            num_examples=100,
            annotation_type="text_classification",
            experience_level="mid",
        )
        result_unknown = self.calc.calculate_labor_cost(
            num_examples=100,
            annotation_type="text_classification",
            experience_level="unknown_level",
        )
        self.assertAlmostEqual(result_mid["total"], result_unknown["total"], places=2)

    def test_unknown_annotation_type_defaults(self):
        """Unknown annotation type should default to text_classification time."""
        result_known = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="text_classification"
        )
        result_unknown = self.calc.calculate_labor_cost(
            num_examples=100, annotation_type="totally_unknown"
        )
        self.assertAlmostEqual(result_known["total"], result_unknown["total"], places=2)

    def test_details_structure(self):
        result = self.calc.calculate_labor_cost(
            num_examples=100,
            annotation_type="text_classification",
            experience_level="senior",
            region="eu",
        )
        details = result["details"]
        self.assertIn("annotation_hours", details)
        self.assertIn("review_hours", details)
        self.assertIn("effective_hourly_rate", details)
        self.assertEqual(details["region"], "eu")
        self.assertEqual(details["experience_level"], "senior")
        self.assertAlmostEqual(
            details["region_multiplier"],
            REGION_COST_MULTIPLIERS["eu"],
            places=2,
        )

    def test_zero_review_rate(self):
        result = self.calc.calculate_labor_cost(
            num_examples=100,
            annotation_type="text_classification",
            review_rate=0.0,
        )
        self.assertEqual(result["review_cost"], 0.0)


# =============================================================================
# EnhancedCostCalculator.calculate_enhanced_cost()
# =============================================================================


class TestCalculateEnhancedCost(unittest.TestCase):
    """Tests for calculate_enhanced_cost() integration."""

    def setUp(self):
        self.calc = _make_enhanced_calculator()

    def test_basic_no_labor(self):
        """Recipe with no human ratio should have zero labor cost."""
        recipe = _make_recipe(num_examples=5000, human_ratio=None)
        result = self.calc.calculate_enhanced_cost(recipe)
        self.assertIsInstance(result, EnhancedCostBreakdown)
        self.assertEqual(result.labor_cost.expected, 0)
        self.assertEqual(result.labor_breakdown, {})

    def test_with_labor(self):
        """Recipe with human ratio should include labor costs."""
        recipe = _make_recipe(
            num_examples=1000,
            human_ratio=0.5,
            generation_methods=[GenerationMethod(method_type="text_generation")],
        )
        result = self.calc.calculate_enhanced_cost(recipe, region="us")
        self.assertGreater(result.labor_cost.expected, 0)
        self.assertIn("annotation", result.labor_breakdown)
        self.assertIn("review", result.labor_breakdown)
        self.assertIn("project_management", result.labor_breakdown)

    def test_region_applied(self):
        recipe = _make_recipe(
            num_examples=1000,
            human_ratio=0.5,
            generation_methods=[GenerationMethod(method_type="text_generation")],
        )
        result_us = self.calc.calculate_enhanced_cost(recipe, region="us")
        result_cn = self.calc.calculate_enhanced_cost(recipe, region="cn")
        self.assertGreater(result_us.labor_cost.expected, result_cn.labor_cost.expected)

    def test_include_labor_false(self):
        """include_labor=False should disable labor even with human_ratio."""
        recipe = _make_recipe(num_examples=1000, human_ratio=0.5)
        result = self.calc.calculate_enhanced_cost(recipe, include_labor=False)
        self.assertEqual(result.labor_cost.expected, 0)

    def test_target_size_override(self):
        recipe = _make_recipe(num_examples=5000)
        result = self.calc.calculate_enhanced_cost(recipe, target_size=2000)
        self.assertEqual(result.details["num_examples"], 2000)

    def test_default_num_examples(self):
        recipe = _make_recipe(num_examples=None)
        result = self.calc.calculate_enhanced_cost(recipe)
        self.assertEqual(result.details["num_examples"], 10000)

    def test_total_is_sum(self):
        recipe = _make_recipe(
            num_examples=1000,
            human_ratio=0.3,
            generation_methods=[GenerationMethod(method_type="finetune_model")],
        )
        result = self.calc.calculate_enhanced_cost(recipe)
        expected_total = (
            result.api_cost.expected
            + result.compute_cost.expected
            + result.labor_cost.expected
        )
        self.assertAlmostEqual(result.total.expected, expected_total, places=2)

    def test_labor_cost_range(self):
        """Labor cost low=0.7x, high=1.5x expected."""
        recipe = _make_recipe(
            num_examples=1000,
            human_ratio=0.5,
            generation_methods=[GenerationMethod(method_type="text_generation")],
        )
        result = self.calc.calculate_enhanced_cost(recipe)
        self.assertAlmostEqual(
            result.labor_cost.low, result.labor_cost.expected * 0.7, places=2
        )
        self.assertAlmostEqual(
            result.labor_cost.high, result.labor_cost.expected * 1.5, places=2
        )

    def test_region_multiplier_stored(self):
        recipe = _make_recipe(num_examples=1000)
        result = self.calc.calculate_enhanced_cost(recipe, region="eu")
        self.assertEqual(result.region, "eu")
        self.assertAlmostEqual(
            result.region_multiplier, REGION_COST_MULTIPLIERS["eu"], places=2
        )

    def test_assumptions_populated(self):
        recipe = _make_recipe(
            num_examples=1000,
            human_ratio=0.5,
            generation_methods=[GenerationMethod(method_type="text_generation")],
        )
        result = self.calc.calculate_enhanced_cost(recipe, region="cn")
        # Should mention target size, region, annotation type
        combined = " ".join(result.assumptions)
        self.assertIn("1,000", combined)
        self.assertIn("cn", combined)

    def test_details_structure(self):
        recipe = _make_recipe(
            num_examples=1000,
            synthetic_ratio=0.7,
            human_ratio=0.3,
        )
        result = self.calc.calculate_enhanced_cost(recipe, model="claude-3-haiku")
        self.assertEqual(result.details["num_examples"], 1000)
        self.assertEqual(result.details["model"], "claude-3-haiku")
        self.assertEqual(result.details["synthetic_ratio"], 0.7)
        self.assertEqual(result.details["human_ratio"], 0.3)

    def test_model_auto_detected_in_details(self):
        recipe = _make_recipe(num_examples=1000)
        result = self.calc.calculate_enhanced_cost(recipe)
        self.assertEqual(result.details["model"], "auto-detected")


# =============================================================================
# EnhancedCostCalculator.calculate_roi()
# =============================================================================


class TestCalculateROI(unittest.TestCase):
    """Tests for calculate_roi()."""

    def setUp(self):
        self.calc = _make_enhanced_calculator()

    def test_positive_roi(self):
        """Comparable price > production cost should yield positive ROI."""
        result = self.calc.calculate_roi(
            production_cost=1000.0,
            comparable_dataset_price=2000.0,
        )
        self.assertEqual(result["production_cost"], 1000.0)
        self.assertEqual(result["comparable_price"], 2000.0)
        self.assertAlmostEqual(result["roi_ratio"], 1.0, places=5)
        self.assertIsNotNone(result["break_even_uses"])

    def test_high_roi_recommendation(self):
        """ROI > 0.5 should recommend building."""
        result = self.calc.calculate_roi(
            production_cost=100.0,
            comparable_dataset_price=200.0,
        )
        self.assertIn("强烈建议自建", result["recommendation"])

    def test_moderate_roi_recommendation(self):
        """0 < ROI <= 0.5 should recommend building with caveats."""
        result = self.calc.calculate_roi(
            production_cost=100.0,
            comparable_dataset_price=130.0,
        )
        self.assertIn("建议自建", result["recommendation"])
        self.assertNotIn("强烈", result["recommendation"])

    def test_negative_roi_recommendation(self):
        """Negative ROI should recommend buying."""
        result = self.calc.calculate_roi(
            production_cost=200.0,
            comparable_dataset_price=100.0,
        )
        self.assertIn("建议购买", result["recommendation"])

    def test_no_comparable_price(self):
        """Without comparable price, ROI should be None."""
        result = self.calc.calculate_roi(production_cost=1000.0)
        self.assertIsNone(result["roi_ratio"])
        self.assertIsNone(result["break_even_uses"])
        self.assertEqual(result["recommendation"], "")

    def test_zero_comparable_price(self):
        """Zero comparable price should not compute ROI."""
        result = self.calc.calculate_roi(
            production_cost=1000.0,
            comparable_dataset_price=0.0,
        )
        self.assertIsNone(result["roi_ratio"])

    def test_break_even_uses(self):
        """Break-even uses = production_cost / comparable_price."""
        result = self.calc.calculate_roi(
            production_cost=1000.0,
            comparable_dataset_price=500.0,
        )
        self.assertAlmostEqual(result["break_even_uses"], 2.0, places=5)

    def test_usage_scenarios_param_accepted(self):
        """usage_scenarios parameter should be accepted without error."""
        result = self.calc.calculate_roi(
            production_cost=1000.0,
            comparable_dataset_price=2000.0,
            usage_scenarios=5,
        )
        self.assertIsNotNone(result["roi_ratio"])


# =============================================================================
# EnhancedCostCalculator.format_enhanced_report()
# =============================================================================


class TestFormatEnhancedReport(unittest.TestCase):
    """Tests for format_enhanced_report()."""

    def setUp(self):
        self.calc = _make_enhanced_calculator()

    def test_report_structure_basic(self):
        """Report should contain all sections."""
        bd = EnhancedCostBreakdown(
            api_cost=CostEstimate(10, 20, 15),
            labor_cost=CostEstimate(0, 0, 0),
            compute_cost=CostEstimate(0, 0, 0),
            total=CostEstimate(10, 20, 15),
            region="us",
            region_multiplier=1.0,
            assumptions=["test assumption"],
        )
        report = self.calc.format_enhanced_report(bd)
        self.assertIn("增强成本估算报告", report)
        self.assertIn("成本分解", report)
        self.assertIn("API 成本", report)
        self.assertIn("人力成本", report)
        self.assertIn("算力成本", report)
        self.assertIn("总计", report)
        self.assertIn("地区信息", report)
        self.assertIn("假设", report)
        self.assertIn("test assumption", report)

    def test_report_with_labor_breakdown(self):
        """Report should show labor breakdown when present."""
        bd = EnhancedCostBreakdown(
            api_cost=CostEstimate(10, 20, 15),
            labor_cost=CostEstimate(50, 100, 75),
            compute_cost=CostEstimate(5, 10, 7),
            total=CostEstimate(65, 130, 97),
            labor_breakdown={"annotation": 40.0, "review": 25.0, "project_management": 10.0},
            region="cn",
            region_multiplier=0.4,
        )
        report = self.calc.format_enhanced_report(bd)
        self.assertIn("人力成本细分", report)
        self.assertIn("标注", report)
        self.assertIn("审核", report)
        self.assertIn("项目管理", report)
        self.assertIn("cn", report)
        self.assertIn("0.4", report)

    def test_report_no_labor_breakdown(self):
        """Report without labor breakdown should skip that section."""
        bd = EnhancedCostBreakdown(
            api_cost=CostEstimate(10, 20, 15),
            labor_cost=CostEstimate(0, 0, 0),
            compute_cost=CostEstimate(0, 0, 0),
            total=CostEstimate(10, 20, 15),
            labor_breakdown={},
        )
        report = self.calc.format_enhanced_report(bd)
        self.assertNotIn("人力成本细分", report)


# =============================================================================
# Integration: EnhancedCostCalculator inherits CostCalculator
# =============================================================================


class TestEnhancedInheritance(unittest.TestCase):
    """Verify EnhancedCostCalculator properly inherits CostCalculator."""

    def test_is_subclass(self):
        self.assertTrue(issubclass(EnhancedCostCalculator, CostCalculator))

    def test_can_use_base_calculate(self):
        calc = _make_enhanced_calculator()
        result = calc.calculate(num_examples=100)
        self.assertIsInstance(result, CostBreakdown)

    def test_can_use_base_estimate_from_recipe(self):
        calc = _make_enhanced_calculator()
        recipe = _make_recipe(num_examples=500)
        result = calc.estimate_from_recipe(recipe)
        self.assertIsInstance(result, CostBreakdown)

    def test_can_use_list_models(self):
        calc = _make_enhanced_calculator()
        models = calc.list_models()
        self.assertIn("gpt-4o", models)


# =============================================================================
# Edge cases / numerical correctness
# =============================================================================


class TestNumericalCorrectness(unittest.TestCase):
    """Verify specific numerical calculations are correct."""

    def setUp(self):
        self.calc = _make_calculator()

    def test_gpt4o_1k_examples_exact(self):
        """Verify exact cost for 1000 examples with gpt-4o at default settings."""
        result = self.calc.calculate(
            num_examples=1000,
            model="gpt-4o",
            avg_input_tokens=500,
            avg_output_tokens=200,
            retries=1.0,
        )
        pricing = self.calc.LLM_PRICING["gpt-4o"]
        input_cost = (1000 * 500 / 1000) * pricing.input_price_per_1k
        output_cost = (1000 * 200 / 1000) * pricing.output_price_per_1k
        expected = input_cost + output_cost
        self.assertAlmostEqual(result.api_cost.expected, expected, places=5)

    def test_gpt35_cheapest_option(self):
        """GPT-3.5-turbo should be cheaper than GPT-4o."""
        cost_35 = self.calc.calculate(num_examples=1000, model="gpt-3.5-turbo", retries=1.0)
        cost_4o = self.calc.calculate(num_examples=1000, model="gpt-4o", retries=1.0)
        self.assertLess(cost_35.api_cost.expected, cost_4o.api_cost.expected)

    def test_scaling_linearity(self):
        """Doubling examples should double API cost."""
        cost_1k = self.calc.calculate(num_examples=1000, retries=1.0)
        cost_2k = self.calc.calculate(num_examples=2000, retries=1.0)
        self.assertAlmostEqual(
            cost_2k.api_cost.expected, cost_1k.api_cost.expected * 2, places=5
        )


if __name__ == "__main__":
    unittest.main()
