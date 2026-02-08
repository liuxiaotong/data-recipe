"""
Context Construction Strategy Detector

Identifies how dataset contexts were constructed:
- Synthetic/Fictional: Created from scratch
- Modified: Based on existing data with modifications
- Niche: From specialized/rare sources
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class ContextStrategyType(Enum):
    """Types of context construction strategies."""

    SYNTHETIC = "synthetic"  # 100% created/fictional
    MODIFIED = "modified"  # Based on existing data
    NICHE = "niche"  # From specialized sources
    HYBRID = "hybrid"  # Mixed strategies
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class ContextStrategy:
    """Analysis result for context construction strategy."""

    primary_strategy: ContextStrategyType
    confidence: float = 0.0  # 0.0-1.0

    # Strategy-specific scores
    synthetic_score: float = 0.0  # How synthetic/fictional
    modified_score: float = 0.0  # How much based on real data
    niche_score: float = 0.0  # How specialized/rare

    # Evidence
    synthetic_indicators: list[str] = field(default_factory=list)
    modified_indicators: list[str] = field(default_factory=list)
    niche_indicators: list[str] = field(default_factory=list)

    # Additional analysis
    data_sources: list[str] = field(default_factory=list)
    modification_types: list[str] = field(default_factory=list)
    domain_specificity: float = 0.0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Primary Strategy: {self.primary_strategy.value}",
            f"Confidence: {self.confidence:.1%}",
            "",
            "Strategy Scores:",
            f"  - Synthetic: {self.synthetic_score:.1%}",
            f"  - Modified: {self.modified_score:.1%}",
            f"  - Niche: {self.niche_score:.1%}",
        ]

        if self.synthetic_indicators:
            lines.append("")
            lines.append("Synthetic Indicators:")
            for ind in self.synthetic_indicators[:5]:
                lines.append(f"  - {ind}")

        if self.modified_indicators:
            lines.append("")
            lines.append("Modification Indicators:")
            for ind in self.modified_indicators[:5]:
                lines.append(f"  - {ind}")

        if self.niche_indicators:
            lines.append("")
            lines.append("Niche/Specialized Indicators:")
            for ind in self.niche_indicators[:5]:
                lines.append(f"  - {ind}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class ContextStrategyDetector:
    """
    Detects how dataset contexts were constructed.

    Example usage:
        detector = ContextStrategyDetector()
        strategy = detector.analyze(contexts, metadata)
        print(strategy.summary())
    """

    # Indicators for synthetic/fictional content
    SYNTHETIC_INDICATORS = {
        "fictional_markers": [
            r"fictional",
            r"imaginary",
            r"hypothetical",
            r"made-up",
            r"invented",
            r"fantasy",
            r"assume that",
            r"suppose that",
            r"in this scenario",
        ],
        "generation_markers": [
            r"generated",
            r"synthesized",
            r"created by",
            r"produced by",
            r"gpt-\d",
            r"claude",
            r"llama",
            r"model-generated",
        ],
        "placeholder_markers": [
            r"\[.*?\]",  # [placeholder]
            r"\{.*?\}",  # {placeholder}
            r"<.*?>",  # <placeholder>
            r"xxx+",  # xxx placeholder
            r"example\.com",
            r"john doe",
            r"jane doe",
        ],
    }

    # Indicators for modified/augmented content
    MODIFIED_INDICATORS = {
        "augmentation_markers": [
            r"augmented",
            r"paraphrased",
            r"modified",
            r"adapted from",
            r"based on",
            r"derived from",
            r"transformed",
        ],
        "source_references": [
            r"original source",
            r"source:",
            r"from:",
            r"reference:",
            r"adapted from",
            r"wikipedia",
            r"arxiv",
        ],
        "modification_types": [
            r"back-?translation",
            r"synonym replacement",
            r"word swap",
            r"sentence shuffle",
        ],
    }

    # Indicators for niche/specialized content
    NICHE_INDICATORS = {
        "domain_specific": [
            r"proprietary",
            r"internal",
            r"confidential",
            r"specialized",
            r"domain expert",
            r"professional",
        ],
        "rare_sources": [
            r"unpublished",
            r"private dataset",
            r"collected from",
            r"manually gathered",
            r"annotated by experts",
        ],
        "technical_depth": [
            r"technical specification",
            r"api documentation",
            r"code repository",
            r"research paper",
            r"clinical trial",
            r"legal document",
        ],
    }

    def __init__(self):
        # Compile all patterns
        self.synthetic_patterns = self._compile_patterns(self.SYNTHETIC_INDICATORS)
        self.modified_patterns = self._compile_patterns(self.MODIFIED_INDICATORS)
        self.niche_patterns = self._compile_patterns(self.NICHE_INDICATORS)

    def _compile_patterns(
        self, indicator_dict: dict[str, list[str]]
    ) -> dict[str, list[re.Pattern]]:
        """Compile regex patterns."""
        return {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in indicator_dict.items()
        }

    def analyze(self, contexts: list[str], metadata: dict | None = None) -> ContextStrategy:
        """
        Analyze contexts to detect construction strategy.

        Args:
            contexts: List of context strings from the dataset
            metadata: Optional metadata about the dataset

        Returns:
            ContextStrategy with analysis results
        """
        result = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN)

        if not contexts:
            return result

        # Analyze each context
        synthetic_matches = []
        modified_matches = []
        niche_matches = []

        for context in contexts:
            context_lower = context.lower()

            # Check synthetic indicators
            for category, patterns in self.synthetic_patterns.items():
                for pattern in patterns:
                    if pattern.search(context_lower):
                        synthetic_matches.append(f"{category}: {pattern.pattern}")

            # Check modified indicators
            for category, patterns in self.modified_patterns.items():
                for pattern in patterns:
                    if pattern.search(context_lower):
                        modified_matches.append(f"{category}: {pattern.pattern}")

            # Check niche indicators
            for category, patterns in self.niche_patterns.items():
                for pattern in patterns:
                    if pattern.search(context_lower):
                        niche_matches.append(f"{category}: {pattern.pattern}")

        # Calculate scores based on match density
        total_contexts = len(contexts)
        result.synthetic_score = min(1.0, len(synthetic_matches) / (total_contexts * 2))
        result.modified_score = min(1.0, len(modified_matches) / (total_contexts * 2))
        result.niche_score = min(1.0, len(niche_matches) / (total_contexts * 2))

        # Normalize scores
        total_score = result.synthetic_score + result.modified_score + result.niche_score
        if total_score > 0:
            result.synthetic_score /= total_score
            result.modified_score /= total_score
            result.niche_score /= total_score

        # Store unique indicators
        result.synthetic_indicators = list(set(synthetic_matches))[:10]
        result.modified_indicators = list(set(modified_matches))[:10]
        result.niche_indicators = list(set(niche_matches))[:10]

        # Determine primary strategy
        scores = {
            ContextStrategyType.SYNTHETIC: result.synthetic_score,
            ContextStrategyType.MODIFIED: result.modified_score,
            ContextStrategyType.NICHE: result.niche_score,
        }

        max_score = max(scores.values())
        if max_score < 0.2:
            result.primary_strategy = ContextStrategyType.UNKNOWN
            result.confidence = 0.3
        else:
            result.primary_strategy = max(scores, key=scores.get)
            result.confidence = max_score

            # Check for hybrid
            second_score = sorted(scores.values())[-2]
            if second_score > 0.3 and max_score - second_score < 0.2:
                result.primary_strategy = ContextStrategyType.HYBRID
                result.confidence = (max_score + second_score) / 2

        # Analyze metadata if available
        if metadata:
            self._analyze_metadata(result, metadata)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        # Additional analysis
        result.domain_specificity = self._calculate_domain_specificity(contexts)

        return result

    def _analyze_metadata(self, result: ContextStrategy, metadata: dict) -> None:
        """Enhance analysis with metadata."""
        # Check for data source information
        source = metadata.get("source", "")
        if source:
            result.data_sources.append(source)

        # Check for generation method
        method = metadata.get("generation_method", "")
        if "synthetic" in method.lower() or "generated" in method.lower():
            result.synthetic_score = max(result.synthetic_score, 0.5)
            result.synthetic_indicators.append(f"metadata: {method}")

        # Check for original source
        if "original_source" in metadata:
            result.modified_score = max(result.modified_score, 0.3)
            result.modified_indicators.append(f"has original source: {metadata['original_source']}")

    def _calculate_domain_specificity(self, contexts: list[str]) -> float:
        """Calculate how domain-specific the contexts are."""
        if not contexts:
            return 0.0

        # Technical/specialized term indicators
        tech_patterns = [
            r"\b[A-Z]{2,}\b",  # Acronyms
            r"\b\w+(?:tion|sion|ment|ity)\b",  # Abstract nouns
            r"\b(?:cf\.|e\.g\.|i\.e\.|et al\.)",  # Academic markers
            r"\b\d+(?:\.\d+)+\b",  # Version numbers
            r"\b[a-z]+_[a-z]+\b",  # snake_case (technical)
        ]

        total_matches = 0
        total_words = 0

        for context in contexts[:100]:  # Sample first 100
            words = context.split()
            total_words += len(words)
            for pattern in tech_patterns:
                total_matches += len(re.findall(pattern, context))

        if total_words == 0:
            return 0.0

        return min(1.0, total_matches / (total_words * 0.1))

    def _generate_recommendations(self, result: ContextStrategy) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if result.primary_strategy == ContextStrategyType.SYNTHETIC:
            recommendations.extend(
                [
                    "Use LLM (GPT-4, Claude) for context generation",
                    "Create detailed fictional scenarios with internal consistency",
                    "Design rule systems or game mechanics for structured content",
                    "Ensure generated content is novel (not in training data)",
                ]
            )

        elif result.primary_strategy == ContextStrategyType.MODIFIED:
            recommendations.extend(
                [
                    "Start with existing documents as base",
                    "Apply paraphrasing and augmentation techniques",
                    "Modify key details while preserving structure",
                    "Track modifications for reproducibility",
                ]
            )

        elif result.primary_strategy == ContextStrategyType.NICHE:
            recommendations.extend(
                [
                    "Identify domain experts for content creation",
                    "Source from specialized/professional documents",
                    "Consider licensing for proprietary content",
                    "Ensure domain accuracy through expert review",
                ]
            )

        elif result.primary_strategy == ContextStrategyType.HYBRID:
            recommendations.extend(
                [
                    "Combine multiple strategies based on category",
                    "Use synthetic for rule-based content",
                    "Use modification for factual content",
                    "Use niche sources for domain expertise",
                ]
            )

        else:
            recommendations.extend(
                [
                    "Gather more samples for accurate detection",
                    "Review dataset documentation for methodology",
                    "Consider manual inspection of sample contexts",
                ]
            )

        return recommendations

    def to_dict(self, result: ContextStrategy) -> dict:
        """Convert result to dictionary for JSON export."""
        return {
            "primary_strategy": result.primary_strategy.value,
            "confidence": result.confidence,
            "scores": {
                "synthetic": result.synthetic_score,
                "modified": result.modified_score,
                "niche": result.niche_score,
            },
            "indicators": {
                "synthetic": result.synthetic_indicators,
                "modified": result.modified_indicators,
                "niche": result.niche_indicators,
            },
            "data_sources": result.data_sources,
            "domain_specificity": result.domain_specificity,
            "recommendations": result.recommendations,
        }
