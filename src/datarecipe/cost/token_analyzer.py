"""Token analysis and precise API cost calculation.

Analyzes actual dataset samples to calculate precise token counts
and API costs based on real data rather than fixed estimates.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenStats:
    """Token statistics for a dataset."""

    # Sample info
    sample_count: int = 0

    # Token counts
    avg_input_tokens: int = 0
    avg_output_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Distribution
    min_input_tokens: int = 0
    max_input_tokens: int = 0
    min_output_tokens: int = 0
    max_output_tokens: int = 0

    # By field
    field_token_counts: Dict[str, int] = field(default_factory=dict)

    # Percentiles
    p50_input: int = 0
    p90_input: int = 0
    p99_input: int = 0
    p50_output: int = 0
    p90_output: int = 0
    p99_output: int = 0

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "distribution": {
                "min_input": self.min_input_tokens,
                "max_input": self.max_input_tokens,
                "min_output": self.min_output_tokens,
                "max_output": self.max_output_tokens,
            },
            "percentiles": {
                "p50_input": self.p50_input,
                "p90_input": self.p90_input,
                "p99_input": self.p99_input,
                "p50_output": self.p50_output,
                "p90_output": self.p90_output,
                "p99_output": self.p99_output,
            },
            "field_token_counts": self.field_token_counts,
        }


@dataclass
class ModelPricing:
    """Pricing for a specific model (per 1M tokens)."""

    provider: str
    model: str
    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    context_window: int = 128000

    @property
    def input_per_1k(self) -> float:
        return self.input_per_1m / 1000

    @property
    def output_per_1k(self) -> float:
        return self.output_per_1m / 1000


# Updated pricing as of 2024-2025
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": ModelPricing("openai", "gpt-4o", 2.50, 10.00, 128000),
    "gpt-4o-mini": ModelPricing("openai", "gpt-4o-mini", 0.15, 0.60, 128000),
    "gpt-4-turbo": ModelPricing("openai", "gpt-4-turbo", 10.00, 30.00, 128000),
    "gpt-4": ModelPricing("openai", "gpt-4", 30.00, 60.00, 8192),
    "gpt-3.5-turbo": ModelPricing("openai", "gpt-3.5-turbo", 0.50, 1.50, 16385),
    "o1": ModelPricing("openai", "o1", 15.00, 60.00, 200000),
    "o1-mini": ModelPricing("openai", "o1-mini", 3.00, 12.00, 128000),

    # Anthropic
    "claude-3-opus": ModelPricing("anthropic", "claude-3-opus", 15.00, 75.00, 200000),
    "claude-3.5-sonnet": ModelPricing("anthropic", "claude-3.5-sonnet", 3.00, 15.00, 200000),
    "claude-3-sonnet": ModelPricing("anthropic", "claude-3-sonnet", 3.00, 15.00, 200000),
    "claude-3-haiku": ModelPricing("anthropic", "claude-3-haiku", 0.25, 1.25, 200000),
    "claude-3.5-haiku": ModelPricing("anthropic", "claude-3.5-haiku", 0.80, 4.00, 200000),

    # Google
    "gemini-1.5-pro": ModelPricing("google", "gemini-1.5-pro", 1.25, 5.00, 2000000),
    "gemini-1.5-flash": ModelPricing("google", "gemini-1.5-flash", 0.075, 0.30, 1000000),
    "gemini-2.0-flash": ModelPricing("google", "gemini-2.0-flash", 0.10, 0.40, 1000000),

    # DeepSeek
    "deepseek-v3": ModelPricing("deepseek", "deepseek-v3", 0.27, 1.10, 64000),
    "deepseek-r1": ModelPricing("deepseek", "deepseek-r1", 0.55, 2.19, 64000),
    "deepseek-chat": ModelPricing("deepseek", "deepseek-chat", 0.14, 0.28, 64000),

    # Open Source (via Together/Fireworks)
    "llama-3.1-405b": ModelPricing("together", "llama-3.1-405b", 3.50, 3.50, 128000),
    "llama-3.1-70b": ModelPricing("together", "llama-3.1-70b", 0.88, 0.88, 128000),
    "llama-3.1-8b": ModelPricing("together", "llama-3.1-8b", 0.18, 0.18, 128000),
    "qwen-2.5-72b": ModelPricing("together", "qwen-2.5-72b", 0.90, 0.90, 128000),
    "mixtral-8x22b": ModelPricing("together", "mixtral-8x22b", 1.20, 1.20, 65536),
}


class TokenAnalyzer:
    """Analyzes dataset samples to calculate token statistics."""

    # Approximate tokens per character for different languages
    CHARS_PER_TOKEN = {
        "en": 4.0,      # English: ~4 chars per token
        "zh": 1.5,      # Chinese: ~1.5 chars per token
        "ja": 1.5,      # Japanese
        "ko": 2.0,      # Korean
        "mixed": 3.0,   # Mixed content
        "code": 3.5,    # Code
    }

    def __init__(self, use_tiktoken: bool = False):
        """Initialize the analyzer.

        Args:
            use_tiktoken: Whether to use tiktoken for accurate OpenAI token counts.
                         Falls back to estimation if tiktoken not available.
        """
        self.use_tiktoken = use_tiktoken
        self._tokenizer = None

        if use_tiktoken:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self.use_tiktoken = False

    def count_tokens(self, text: str, lang: str = "mixed") -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for
            lang: Language hint for estimation

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        if self._tokenizer:
            return len(self._tokenizer.encode(text))

        # Estimation based on character count and language
        chars_per_token = self.CHARS_PER_TOKEN.get(lang, 3.0)

        # Detect language/content type
        if self._is_code(text):
            chars_per_token = self.CHARS_PER_TOKEN["code"]
        elif self._has_cjk(text):
            # Calculate CJK ratio
            cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
            cjk_ratio = cjk_chars / len(text) if text else 0
            chars_per_token = 1.5 * cjk_ratio + 4.0 * (1 - cjk_ratio)

        return int(len(text) / chars_per_token)

    def _is_code(self, text: str) -> bool:
        """Detect if text is likely code."""
        code_patterns = [
            r'def\s+\w+\s*\(',
            r'function\s+\w+\s*\(',
            r'class\s+\w+',
            r'import\s+\w+',
            r'from\s+\w+\s+import',
            r'\{\s*\n',
            r'if\s*\(.+\)\s*\{',
            r'return\s+',
        ]
        matches = sum(1 for p in code_patterns if re.search(p, text))
        return matches >= 2

    def _has_cjk(self, text: str) -> bool:
        """Check if text contains CJK characters."""
        return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

    def analyze_samples(
        self,
        samples: List[Dict[str, Any]],
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> TokenStats:
        """Analyze samples to calculate token statistics.

        Args:
            samples: List of sample dictionaries
            input_fields: Fields to count as input (auto-detect if None)
            output_fields: Fields to count as output (auto-detect if None)

        Returns:
            TokenStats with detailed token statistics
        """
        if not samples:
            return TokenStats()

        # Auto-detect fields if not specified
        if input_fields is None or output_fields is None:
            detected_input, detected_output = self._detect_io_fields(samples[0])
            input_fields = input_fields or detected_input
            output_fields = output_fields or detected_output

        input_counts = []
        output_counts = []
        field_totals: Dict[str, int] = {}

        for sample in samples:
            # Count input tokens
            input_tokens = 0
            for field in input_fields:
                if field in sample:
                    text = self._extract_text(sample[field])
                    tokens = self.count_tokens(text)
                    input_tokens += tokens
                    field_totals[field] = field_totals.get(field, 0) + tokens
            input_counts.append(input_tokens)

            # Count output tokens
            output_tokens = 0
            for field in output_fields:
                if field in sample:
                    text = self._extract_text(sample[field])
                    tokens = self.count_tokens(text)
                    output_tokens += tokens
                    field_totals[field] = field_totals.get(field, 0) + tokens
            output_counts.append(output_tokens)

        # Calculate statistics
        input_counts.sort()
        output_counts.sort()
        n = len(input_counts)

        stats = TokenStats(
            sample_count=n,
            avg_input_tokens=int(sum(input_counts) / n) if n > 0 else 0,
            avg_output_tokens=int(sum(output_counts) / n) if n > 0 else 0,
            total_input_tokens=sum(input_counts),
            total_output_tokens=sum(output_counts),
            min_input_tokens=min(input_counts) if input_counts else 0,
            max_input_tokens=max(input_counts) if input_counts else 0,
            min_output_tokens=min(output_counts) if output_counts else 0,
            max_output_tokens=max(output_counts) if output_counts else 0,
            field_token_counts={k: v // n for k, v in field_totals.items()} if n > 0 else {},
        )

        # Percentiles
        if n > 0:
            stats.p50_input = input_counts[int(n * 0.5)]
            stats.p90_input = input_counts[int(n * 0.9)]
            stats.p99_input = input_counts[min(int(n * 0.99), n - 1)]
            stats.p50_output = output_counts[int(n * 0.5)]
            stats.p90_output = output_counts[int(n * 0.9)]
            stats.p99_output = output_counts[min(int(n * 0.99), n - 1)]

        return stats

    def _detect_io_fields(self, sample: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Auto-detect input and output fields from sample structure."""
        input_fields = []
        output_fields = []

        # Common patterns
        input_patterns = ["input", "prompt", "question", "query", "context", "instruction", "problem"]
        output_patterns = ["output", "response", "answer", "completion", "solution", "target"]

        for field in sample.keys():
            field_lower = field.lower()

            # Check for messages field (chat format)
            if field == "messages" and isinstance(sample[field], list):
                # Messages format: separate user (input) and assistant (output)
                input_fields.append("messages")  # Will be handled specially
                continue

            # Check for preference format
            if field in ["chosen", "rejected"]:
                output_fields.append(field)
                continue

            # Pattern matching
            if any(p in field_lower for p in input_patterns):
                input_fields.append(field)
            elif any(p in field_lower for p in output_patterns):
                output_fields.append(field)

        # Fallback: if no fields detected, use all string fields
        if not input_fields and not output_fields:
            for field, value in sample.items():
                if isinstance(value, str) and len(value) > 50:
                    input_fields.append(field)

        return input_fields, output_fields

    def _extract_text(self, value: Any) -> str:
        """Extract text from a field value."""
        if isinstance(value, str):
            return value

        if isinstance(value, list):
            # Handle messages format
            texts = []
            for item in value:
                if isinstance(item, dict):
                    content = item.get("content", "")
                    if isinstance(content, str):
                        texts.append(content)
                elif isinstance(item, str):
                    texts.append(item)
            return " ".join(texts)

        if isinstance(value, dict):
            # Concatenate all string values
            texts = []
            for v in value.values():
                if isinstance(v, str):
                    texts.append(v)
            return " ".join(texts)

        return str(value) if value else ""


@dataclass
class PreciseCostEstimate:
    """Precise cost estimate based on actual token analysis."""

    # Token stats
    token_stats: TokenStats

    # Target scale
    target_size: int

    # Model info
    model: str
    pricing: ModelPricing

    # Cost breakdown
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_api_cost: float = 0.0

    # With iterations/retries
    iteration_factor: float = 1.0
    adjusted_cost: float = 0.0

    # Ranges
    cost_low: float = 0.0   # Optimistic (p50 tokens)
    cost_high: float = 0.0  # Pessimistic (p99 tokens)

    # Assumptions
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "target_size": self.target_size,
            "token_stats": self.token_stats.to_dict(),
            "cost": {
                "input_cost": round(self.input_cost, 2),
                "output_cost": round(self.output_cost, 2),
                "total_api_cost": round(self.total_api_cost, 2),
                "iteration_factor": self.iteration_factor,
                "adjusted_cost": round(self.adjusted_cost, 2),
                "range": {
                    "low": round(self.cost_low, 2),
                    "expected": round(self.adjusted_cost, 2),
                    "high": round(self.cost_high, 2),
                },
            },
            "pricing": {
                "input_per_1m": self.pricing.input_per_1m,
                "output_per_1m": self.pricing.output_per_1m,
            },
            "assumptions": self.assumptions,
        }


class PreciseCostCalculator:
    """Calculate precise API costs based on actual token analysis."""

    def __init__(self):
        self.token_analyzer = TokenAnalyzer()

    def calculate(
        self,
        samples: List[Dict[str, Any]],
        target_size: int,
        model: str = "gpt-4o",
        iteration_factor: float = 1.2,
        input_fields: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> PreciseCostEstimate:
        """Calculate precise API cost based on sample analysis.

        Args:
            samples: Sample data to analyze
            target_size: Target dataset size
            model: LLM model for pricing
            iteration_factor: Factor for retries/iterations (1.0 = no retries)
            input_fields: Fields to count as input
            output_fields: Fields to count as output

        Returns:
            PreciseCostEstimate with detailed breakdown
        """
        # Analyze tokens
        token_stats = self.token_analyzer.analyze_samples(
            samples, input_fields, output_fields
        )

        # Get pricing
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])

        # Calculate base cost
        total_input = token_stats.avg_input_tokens * target_size
        total_output = token_stats.avg_output_tokens * target_size

        input_cost = (total_input / 1_000_000) * pricing.input_per_1m
        output_cost = (total_output / 1_000_000) * pricing.output_per_1m
        total_api_cost = input_cost + output_cost

        # Apply iteration factor
        adjusted_cost = total_api_cost * iteration_factor

        # Calculate ranges using percentiles
        cost_low = (
            (token_stats.p50_input * target_size / 1_000_000) * pricing.input_per_1m +
            (token_stats.p50_output * target_size / 1_000_000) * pricing.output_per_1m
        )

        cost_high = (
            (token_stats.p99_input * target_size / 1_000_000) * pricing.input_per_1m +
            (token_stats.p99_output * target_size / 1_000_000) * pricing.output_per_1m
        ) * iteration_factor * 1.5  # Extra buffer for edge cases

        # Build assumptions
        assumptions = [
            f"基于 {token_stats.sample_count} 个样本分析",
            f"平均 input: {token_stats.avg_input_tokens} tokens",
            f"平均 output: {token_stats.avg_output_tokens} tokens",
            f"目标规模: {target_size:,} 条",
            f"模型: {model} (${pricing.input_per_1m}/M in, ${pricing.output_per_1m}/M out)",
            f"迭代系数: {iteration_factor}x",
        ]

        return PreciseCostEstimate(
            token_stats=token_stats,
            target_size=target_size,
            model=model,
            pricing=pricing,
            input_cost=input_cost,
            output_cost=output_cost,
            total_api_cost=total_api_cost,
            iteration_factor=iteration_factor,
            adjusted_cost=adjusted_cost,
            cost_low=cost_low,
            cost_high=cost_high,
            assumptions=assumptions,
        )

    def compare_models(
        self,
        samples: List[Dict[str, Any]],
        target_size: int,
        models: Optional[List[str]] = None,
    ) -> Dict[str, PreciseCostEstimate]:
        """Compare costs across multiple models.

        Args:
            samples: Sample data
            target_size: Target size
            models: Models to compare (default: popular models)

        Returns:
            Dict mapping model name to cost estimate
        """
        if models is None:
            models = [
                "gpt-4o", "gpt-4o-mini",
                "claude-3.5-sonnet", "claude-3-haiku",
                "gemini-1.5-pro", "gemini-1.5-flash",
                "deepseek-v3",
                "llama-3.1-70b",
            ]

        results = {}
        for model in models:
            if model in MODEL_PRICING:
                results[model] = self.calculate(samples, target_size, model)

        return results

    def format_comparison_table(
        self, comparisons: Dict[str, PreciseCostEstimate]
    ) -> str:
        """Format model comparison as markdown table."""
        lines = [
            "| 模型 | 预期成本 | 范围 | Input $/M | Output $/M |",
            "|------|----------|------|-----------|------------|",
        ]

        # Sort by cost
        sorted_items = sorted(
            comparisons.items(),
            key=lambda x: x[1].adjusted_cost
        )

        for model, estimate in sorted_items:
            lines.append(
                f"| {model} | ${estimate.adjusted_cost:,.2f} | "
                f"${estimate.cost_low:,.2f}-${estimate.cost_high:,.2f} | "
                f"${estimate.pricing.input_per_1m:.2f} | "
                f"${estimate.pricing.output_per_1m:.2f} |"
            )

        return "\n".join(lines)
