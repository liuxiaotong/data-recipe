"""Cost calculator for dataset production estimation."""

from dataclasses import dataclass, field
from typing import Optional

from datarecipe.schema import Recipe


@dataclass
class TokenPricing:
    """Pricing information for a specific model."""

    provider: str  # openai, anthropic, google, etc.
    model: str  # gpt-4o, claude-3-sonnet, etc.
    input_price_per_1k: float  # USD per 1K input tokens
    output_price_per_1k: float  # USD per 1K output tokens


@dataclass
class CostEstimate:
    """A single cost estimate with range."""

    low: float
    high: float
    expected: float
    unit: str = "USD"

    def __str__(self) -> str:
        if self.low == self.high:
            return f"${self.expected:,.2f}"
        return f"${self.low:,.2f} - ${self.high:,.2f}"


@dataclass
class CostBreakdown:
    """Complete cost breakdown for dataset production."""

    api_cost: CostEstimate
    human_annotation_cost: CostEstimate
    compute_cost: CostEstimate
    total: CostEstimate
    assumptions: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "api_cost": {
                "low": self.api_cost.low,
                "high": self.api_cost.high,
                "expected": self.api_cost.expected,
            },
            "human_annotation_cost": {
                "low": self.human_annotation_cost.low,
                "high": self.human_annotation_cost.high,
                "expected": self.human_annotation_cost.expected,
            },
            "compute_cost": {
                "low": self.compute_cost.low,
                "high": self.compute_cost.high,
                "expected": self.compute_cost.expected,
            },
            "total": {
                "low": self.total.low,
                "high": self.total.high,
                "expected": self.total.expected,
            },
            "assumptions": self.assumptions,
            "details": self.details,
        }


class CostCalculator:
    """Calculator for estimating dataset production costs."""

    # Current LLM pricing (as of 2024, update as needed)
    LLM_PRICING = {
        # OpenAI models
        "gpt-4o": TokenPricing("openai", "gpt-4o", 0.005, 0.015),
        "gpt-4o-mini": TokenPricing("openai", "gpt-4o-mini", 0.00015, 0.0006),
        "gpt-4-turbo": TokenPricing("openai", "gpt-4-turbo", 0.01, 0.03),
        "gpt-4": TokenPricing("openai", "gpt-4", 0.03, 0.06),
        "gpt-3.5-turbo": TokenPricing("openai", "gpt-3.5-turbo", 0.0005, 0.0015),
        # Anthropic models
        "claude-3-opus": TokenPricing("anthropic", "claude-3-opus", 0.015, 0.075),
        "claude-3-sonnet": TokenPricing("anthropic", "claude-3-sonnet", 0.003, 0.015),
        "claude-3-haiku": TokenPricing("anthropic", "claude-3-haiku", 0.00025, 0.00125),
        "claude-3.5-sonnet": TokenPricing("anthropic", "claude-3.5-sonnet", 0.003, 0.015),
        # Google models
        "gemini-pro": TokenPricing("google", "gemini-pro", 0.0005, 0.0015),
        "gemini-1.5-pro": TokenPricing("google", "gemini-1.5-pro", 0.00125, 0.005),
        # Open source (typical API costs)
        "llama-3-70b": TokenPricing("together", "llama-3-70b", 0.0009, 0.0009),
        "llama-3-8b": TokenPricing("together", "llama-3-8b", 0.0002, 0.0002),
        "mixtral-8x7b": TokenPricing("together", "mixtral-8x7b", 0.0006, 0.0006),
    }

    # Human annotation costs (USD per annotation)
    ANNOTATION_COSTS = {
        "simple_label": 0.05,  # Simple yes/no or category
        "text_classification": 0.10,  # Multi-label classification
        "text_generation": 0.50,  # Write short text
        "complex_annotation": 1.00,  # Detailed annotation
        "expert_annotation": 5.00,  # Domain expert required
        "preference_ranking": 0.30,  # Compare and rank options
        "quality_check": 0.15,  # Verify existing annotation
    }

    # Compute costs (USD per hour)
    COMPUTE_COSTS = {
        "cpu_standard": 0.05,  # Standard CPU instance
        "cpu_high": 0.20,  # High-memory CPU
        "gpu_t4": 0.50,  # NVIDIA T4
        "gpu_a10": 1.00,  # NVIDIA A10
        "gpu_a100": 3.00,  # NVIDIA A100
        "gpu_h100": 5.00,  # NVIDIA H100
    }

    def __init__(self):
        """Initialize the cost calculator."""
        pass

    def calculate(
        self,
        num_examples: int,
        model: str = "gpt-4o",
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 200,
        human_annotation_type: Optional[str] = None,
        human_annotation_ratio: float = 0.0,
        compute_type: Optional[str] = None,
        compute_hours: float = 0.0,
        retries: float = 1.2,  # Average retry factor
    ) -> CostBreakdown:
        """Calculate cost breakdown for dataset production.

        Args:
            num_examples: Number of examples to generate
            model: LLM model to use
            avg_input_tokens: Average input tokens per example
            avg_output_tokens: Average output tokens per example
            human_annotation_type: Type of human annotation (if any)
            human_annotation_ratio: Ratio of examples needing annotation
            compute_type: Type of compute resources
            compute_hours: Total compute hours needed
            retries: Average retry factor for API calls

        Returns:
            CostBreakdown with detailed cost estimates
        """
        assumptions = []

        # API costs
        api_cost = self._calculate_api_cost(
            num_examples, model, avg_input_tokens, avg_output_tokens, retries
        )
        if model in self.LLM_PRICING:
            assumptions.append(f"Using {model} pricing")
        assumptions.append(f"Average {avg_input_tokens} input + {avg_output_tokens} output tokens")
        assumptions.append(f"Retry factor: {retries}x")

        # Human annotation costs
        human_cost = self._calculate_human_cost(
            num_examples, human_annotation_type, human_annotation_ratio
        )
        if human_annotation_type:
            assumptions.append(
                f"Human annotation: {human_annotation_type} for {human_annotation_ratio*100:.0f}%"
            )

        # Compute costs
        compute_cost = self._calculate_compute_cost(compute_type, compute_hours)
        if compute_type:
            assumptions.append(f"Compute: {compute_type} for {compute_hours:.1f} hours")

        # Total
        total_low = api_cost.low + human_cost.low + compute_cost.low
        total_high = api_cost.high + human_cost.high + compute_cost.high
        total_expected = api_cost.expected + human_cost.expected + compute_cost.expected

        return CostBreakdown(
            api_cost=api_cost,
            human_annotation_cost=human_cost,
            compute_cost=compute_cost,
            total=CostEstimate(total_low, total_high, total_expected),
            assumptions=assumptions,
            details={
                "num_examples": num_examples,
                "model": model,
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "total_tokens": num_examples * (avg_input_tokens + avg_output_tokens),
            },
        )

    def estimate_from_recipe(
        self,
        recipe: Recipe,
        target_size: Optional[int] = None,
        model: Optional[str] = None,
    ) -> CostBreakdown:
        """Estimate production cost based on a dataset recipe.

        Args:
            recipe: The recipe to estimate costs for
            target_size: Target number of examples (defaults to recipe's num_examples)
            model: LLM model to use (auto-detected from recipe if not provided)

        Returns:
            CostBreakdown with detailed cost estimates
        """
        # Determine number of examples
        num_examples = target_size or recipe.num_examples or 10000
        assumptions = [f"Target size: {num_examples:,} examples"]

        # Determine model
        if model is None:
            if recipe.teacher_models:
                # Try to match to known models
                model = self._match_model(recipe.teacher_models[0])
            else:
                model = "gpt-4o"  # Default
            assumptions.append(f"Inferred model: {model}")

        # Estimate token counts based on generation type
        avg_input_tokens, avg_output_tokens = self._estimate_tokens(recipe)

        # Determine human annotation needs
        human_ratio = recipe.human_ratio or 0.0
        human_type = self._infer_annotation_type(recipe)

        # Calculate compute needs
        compute_type, compute_hours = self._estimate_compute(recipe, num_examples)

        # Build cost breakdown
        cost = self.calculate(
            num_examples=num_examples,
            model=model,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            human_annotation_type=human_type,
            human_annotation_ratio=human_ratio,
            compute_type=compute_type,
            compute_hours=compute_hours,
        )

        # Add recipe-specific assumptions
        cost.assumptions = assumptions + cost.assumptions

        return cost

    def _calculate_api_cost(
        self,
        num_examples: int,
        model: str,
        avg_input_tokens: int,
        avg_output_tokens: int,
        retries: float,
    ) -> CostEstimate:
        """Calculate API costs."""
        pricing = self.LLM_PRICING.get(model)

        if pricing is None:
            # Default to GPT-4o pricing if unknown
            pricing = self.LLM_PRICING["gpt-4o"]

        total_input_tokens = num_examples * avg_input_tokens * retries
        total_output_tokens = num_examples * avg_output_tokens * retries

        input_cost = (total_input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (total_output_tokens / 1000) * pricing.output_price_per_1k

        expected = input_cost + output_cost
        low = expected * 0.8  # Optimistic
        high = expected * 1.5  # Pessimistic (more retries, longer outputs)

        return CostEstimate(low, high, expected)

    def _calculate_human_cost(
        self,
        num_examples: int,
        annotation_type: Optional[str],
        annotation_ratio: float,
    ) -> CostEstimate:
        """Calculate human annotation costs."""
        if annotation_type is None or annotation_ratio == 0:
            return CostEstimate(0, 0, 0)

        cost_per_annotation = self.ANNOTATION_COSTS.get(
            annotation_type, self.ANNOTATION_COSTS["text_classification"]
        )

        num_annotations = num_examples * annotation_ratio
        expected = num_annotations * cost_per_annotation
        low = expected * 0.7  # Efficient annotators
        high = expected * 1.5  # Slower/quality issues

        return CostEstimate(low, high, expected)

    def _calculate_compute_cost(
        self,
        compute_type: Optional[str],
        compute_hours: float,
    ) -> CostEstimate:
        """Calculate compute costs."""
        if compute_type is None or compute_hours == 0:
            return CostEstimate(0, 0, 0)

        cost_per_hour = self.COMPUTE_COSTS.get(
            compute_type, self.COMPUTE_COSTS["cpu_standard"]
        )

        expected = compute_hours * cost_per_hour
        low = expected * 0.8
        high = expected * 1.3

        return CostEstimate(low, high, expected)

    def _match_model(self, model_name: str) -> str:
        """Match a model name to a known pricing model."""
        model_lower = model_name.lower()

        # Direct matches
        if model_lower in self.LLM_PRICING:
            return model_lower

        # Fuzzy matching
        if "gpt-4" in model_lower and "mini" in model_lower:
            return "gpt-4o-mini"
        if "gpt-4" in model_lower:
            return "gpt-4o"
        if "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            return "gpt-3.5-turbo"
        if "claude-3" in model_lower and "opus" in model_lower:
            return "claude-3-opus"
        if "claude-3" in model_lower and "haiku" in model_lower:
            return "claude-3-haiku"
        if "claude" in model_lower:
            return "claude-3-sonnet"
        if "gemini" in model_lower and "1.5" in model_lower:
            return "gemini-1.5-pro"
        if "gemini" in model_lower:
            return "gemini-pro"
        if "llama" in model_lower and "70b" in model_lower:
            return "llama-3-70b"
        if "llama" in model_lower:
            return "llama-3-8b"
        if "mixtral" in model_lower:
            return "mixtral-8x7b"

        # Default
        return "gpt-4o"

    def _estimate_tokens(self, recipe: Recipe) -> tuple[int, int]:
        """Estimate average input/output tokens based on recipe."""
        # Check generation methods for hints
        for method in recipe.generation_methods:
            method_type = method.method_type.lower()

            if "distillation" in method_type:
                return (600, 300)  # Longer context for distillation
            if "instruction" in method_type:
                return (400, 200)  # Instruction following
            if "conversation" in method_type or "chat" in method_type:
                return (800, 400)  # Multi-turn conversations
            if "code" in method_type:
                return (500, 500)  # Code generation
            if "summarization" in method_type or "summary" in method_type:
                return (1000, 200)  # Long input, short output

        # Default based on synthetic ratio
        if recipe.synthetic_ratio and recipe.synthetic_ratio > 0.5:
            return (500, 200)  # Synthetic generation
        else:
            return (300, 150)  # Mixed or human-heavy

    def _infer_annotation_type(self, recipe: Recipe) -> Optional[str]:
        """Infer annotation type from recipe."""
        if recipe.human_ratio is None or recipe.human_ratio == 0:
            return None

        for method in recipe.generation_methods:
            method_type = method.method_type.lower()

            if "preference" in method_type or "rlhf" in method_type:
                return "preference_ranking"
            if "expert" in method_type:
                return "expert_annotation"
            if "verification" in method_type or "check" in method_type:
                return "quality_check"
            if "generation" in method_type or "writing" in method_type:
                return "text_generation"
            if "classification" in method_type or "label" in method_type:
                return "text_classification"

        # Default based on complexity
        if recipe.human_ratio > 0.5:
            return "text_generation"
        else:
            return "quality_check"

    def _estimate_compute(
        self, recipe: Recipe, num_examples: int
    ) -> tuple[Optional[str], float]:
        """Estimate compute requirements."""
        # Check for compute-intensive methods
        for method in recipe.generation_methods:
            method_type = method.method_type.lower()

            if "train" in method_type or "finetune" in method_type:
                # Training requires GPU
                hours = (num_examples / 10000) * 4  # ~4 hours per 10k
                return ("gpu_a100", hours)

            if "embedding" in method_type or "semantic" in method_type:
                # Embedding computation
                hours = (num_examples / 100000) * 1  # ~1 hour per 100k
                return ("gpu_t4", hours)

        # Light processing only
        if num_examples > 100000:
            hours = (num_examples / 100000) * 0.5
            return ("cpu_high", hours)

        return (None, 0.0)

    def get_model_pricing(self, model: str) -> Optional[TokenPricing]:
        """Get pricing information for a specific model."""
        return self.LLM_PRICING.get(model)

    def list_models(self) -> list[str]:
        """List all supported models."""
        return list(self.LLM_PRICING.keys())

    def format_cost_report(self, breakdown: CostBreakdown) -> str:
        """Format a cost breakdown as a readable report."""
        lines = []
        lines.append("=" * 50)
        lines.append("COST ESTIMATION REPORT")
        lines.append("=" * 50)
        lines.append("")

        lines.append("BREAKDOWN:")
        lines.append(f"  API Costs:        {breakdown.api_cost}")
        lines.append(f"  Human Annotation: {breakdown.human_annotation_cost}")
        lines.append(f"  Compute:          {breakdown.compute_cost}")
        lines.append("-" * 50)
        lines.append(f"  TOTAL:            {breakdown.total}")
        lines.append("")

        lines.append("ASSUMPTIONS:")
        for assumption in breakdown.assumptions:
            lines.append(f"  - {assumption}")
        lines.append("")

        if breakdown.details:
            lines.append("DETAILS:")
            for key, value in breakdown.details.items():
                if isinstance(value, int) and value > 1000:
                    lines.append(f"  {key}: {value:,}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)
