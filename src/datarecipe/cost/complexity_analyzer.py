"""Content complexity analysis for dynamic cost adjustment.

Analyzes dataset content to determine complexity factors that
affect human annotation time and quality requirements.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DomainType(Enum):
    """Domain categories with difficulty levels."""

    GENERAL = "general"  # General conversation, Q&A
    CREATIVE = "creative"  # Creative writing, stories
    TECHNICAL = "technical"  # Technical documentation
    CODE = "code"  # Programming, software
    MATH = "math"  # Mathematics, calculations
    SCIENCE = "science"  # Scientific content
    MEDICAL = "medical"  # Medical, healthcare
    LEGAL = "legal"  # Legal documents
    FINANCE = "finance"  # Financial, trading
    ACADEMIC = "academic"  # Research, papers


# Domain difficulty multipliers for human annotation time
DOMAIN_DIFFICULTY = {
    DomainType.GENERAL: 1.0,
    DomainType.CREATIVE: 1.2,
    DomainType.TECHNICAL: 1.5,
    DomainType.CODE: 2.0,
    DomainType.MATH: 2.5,
    DomainType.SCIENCE: 1.8,
    DomainType.MEDICAL: 3.0,
    DomainType.LEGAL: 2.5,
    DomainType.FINANCE: 2.0,
    DomainType.ACADEMIC: 1.8,
}

# Expert hourly rate multipliers by domain
DOMAIN_EXPERT_MULTIPLIER = {
    DomainType.GENERAL: 1.0,
    DomainType.CREATIVE: 1.2,
    DomainType.TECHNICAL: 1.5,
    DomainType.CODE: 2.0,
    DomainType.MATH: 2.0,
    DomainType.SCIENCE: 2.0,
    DomainType.MEDICAL: 3.5,
    DomainType.LEGAL: 3.0,
    DomainType.FINANCE: 2.5,
    DomainType.ACADEMIC: 1.8,
}


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics for a dataset."""

    # Domain analysis
    primary_domain: DomainType = DomainType.GENERAL
    domain_scores: dict[str, float] = field(default_factory=dict)
    domain_confidence: float = 0.0

    # Text length metrics
    avg_text_length: int = 0
    max_text_length: int = 0
    length_variance: float = 0.0
    length_category: str = "medium"  # short, medium, long, very_long

    # Structure complexity
    avg_field_count: int = 0
    has_nested_structure: bool = False
    nesting_depth: int = 0
    structure_category: str = "simple"  # simple, moderate, complex

    # Content complexity
    vocabulary_richness: float = 0.0  # Type-token ratio
    avg_sentence_length: float = 0.0
    technical_term_density: float = 0.0
    code_density: float = 0.0

    # Quality indicators
    has_rubrics: bool = False
    rubric_complexity: str = "none"  # none, simple, detailed, expert
    quality_requirement: str = "standard"  # basic, standard, high, expert

    # Final multipliers
    time_multiplier: float = 1.0
    cost_multiplier: float = 1.0
    difficulty_score: float = 1.0  # 1-5 scale

    def to_dict(self) -> dict:
        return {
            "domain": {
                "primary": self.primary_domain.value,
                "scores": self.domain_scores,
                "confidence": round(self.domain_confidence, 2),
            },
            "text_length": {
                "average": self.avg_text_length,
                "maximum": self.max_text_length,
                "variance": round(self.length_variance, 2),
                "category": self.length_category,
            },
            "structure": {
                "avg_fields": self.avg_field_count,
                "has_nested": self.has_nested_structure,
                "nesting_depth": self.nesting_depth,
                "category": self.structure_category,
            },
            "content": {
                "vocabulary_richness": round(self.vocabulary_richness, 3),
                "avg_sentence_length": round(self.avg_sentence_length, 1),
                "technical_density": round(self.technical_term_density, 3),
                "code_density": round(self.code_density, 3),
            },
            "quality": {
                "has_rubrics": self.has_rubrics,
                "rubric_complexity": self.rubric_complexity,
                "requirement": self.quality_requirement,
            },
            "multipliers": {
                "time": round(self.time_multiplier, 2),
                "cost": round(self.cost_multiplier, 2),
                "difficulty_score": round(self.difficulty_score, 1),
            },
        }


class ComplexityAnalyzer:
    """Analyzes dataset complexity for cost adjustment."""

    # Domain detection keywords
    DOMAIN_KEYWORDS = {
        DomainType.CODE: [
            "function",
            "class",
            "def ",
            "import ",
            "return",
            "if ",
            "for ",
            "while",
            "try:",
            "except",
            "async",
            "await",
            "const ",
            "let ",
            "var ",
            "public",
            "private",
            "void",
            "int ",
            "string",
            "```python",
            "```java",
            "```javascript",
            "```cpp",
        ],
        DomainType.MATH: [
            "equation",
            "formula",
            "calculate",
            "derivative",
            "integral",
            "theorem",
            "proof",
            "∫",
            "∑",
            "√",
            "π",
            "∞",
            "≤",
            "≥",
            "matrix",
            "vector",
            "polynomial",
            "logarithm",
            "exponential",
        ],
        DomainType.MEDICAL: [
            "patient",
            "diagnosis",
            "treatment",
            "symptom",
            "disease",
            "medication",
            "clinical",
            "hospital",
            "surgery",
            "therapy",
            "prescription",
            "dosage",
            "medical history",
            "pathology",
        ],
        DomainType.LEGAL: [
            "contract",
            "clause",
            "liability",
            "plaintiff",
            "defendant",
            "court",
            "judgment",
            "statute",
            "regulation",
            "compliance",
            "attorney",
            "litigation",
            "jurisdiction",
            "precedent",
        ],
        DomainType.FINANCE: [
            "investment",
            "portfolio",
            "stock",
            "bond",
            "dividend",
            "market",
            "trading",
            "asset",
            "liability",
            "equity",
            "revenue",
            "profit",
            "loss",
            "balance sheet",
            "cash flow",
        ],
        DomainType.SCIENCE: [
            "hypothesis",
            "experiment",
            "observation",
            "theory",
            "research",
            "study",
            "analysis",
            "data",
            "methodology",
            "conclusion",
            "findings",
            "sample",
            "variable",
            "control",
        ],
        DomainType.ACADEMIC: [
            "abstract",
            "introduction",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
            "citation",
            "literature review",
            "hypothesis",
            "findings",
        ],
        DomainType.TECHNICAL: [
            "configuration",
            "installation",
            "setup",
            "documentation",
            "api",
            "endpoint",
            "request",
            "response",
            "server",
            "client",
            "database",
            "query",
            "protocol",
            "specification",
        ],
        DomainType.CREATIVE: [
            "story",
            "character",
            "plot",
            "narrative",
            "dialogue",
            "scene",
            "chapter",
            "protagonist",
            "setting",
            "theme",
            "poem",
            "verse",
            "metaphor",
            "imagery",
        ],
    }

    # Technical terms for density calculation
    TECHNICAL_TERMS = {
            "algorithm",
            "parameter",
            "configuration",
            "implementation",
            "optimization",
            "architecture",
            "framework",
            "interface",
            "protocol",
            "specification",
            "methodology",
            "infrastructure",
            "deployment",
            "scalability",
            "performance",
            "latency",
        }

    def analyze(
        self,
        samples: list[dict[str, Any]],
        schema_info: Optional[dict[str, Any]] = None,
        rubrics: Optional[list[str]] = None,
    ) -> ComplexityMetrics:
        """Analyze dataset complexity.

        Args:
            samples: Sample data items
            schema_info: Schema information from analysis
            rubrics: Extracted rubrics if available

        Returns:
            ComplexityMetrics with all analysis results
        """
        metrics = ComplexityMetrics()

        if not samples:
            return metrics

        # Collect all text content
        all_text = []
        text_lengths = []
        field_counts = []

        for sample in samples:
            text = self._extract_all_text(sample)
            all_text.append(text)
            text_lengths.append(len(text))
            field_counts.append(len(sample))

        combined_text = " ".join(all_text)

        # 1. Domain analysis
        self._analyze_domain(metrics, combined_text)

        # 2. Text length analysis
        self._analyze_text_length(metrics, text_lengths)

        # 3. Structure complexity
        self._analyze_structure(metrics, samples, schema_info)

        # 4. Content complexity
        self._analyze_content(metrics, combined_text, all_text)

        # 5. Quality requirements
        self._analyze_quality(metrics, rubrics)

        # 6. Calculate final multipliers
        self._calculate_multipliers(metrics)

        return metrics

    def _extract_all_text(self, sample: dict[str, Any]) -> str:
        """Extract all text content from a sample."""
        texts = []

        def extract_recursive(obj, depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
            if isinstance(obj, str):
                texts.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item, depth + 1)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value, depth + 1)

        extract_recursive(sample)
        return " ".join(texts)

    def _analyze_domain(self, metrics: ComplexityMetrics, text: str) -> None:
        """Detect primary domain from text content."""
        text_lower = text.lower()
        scores = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            # Normalize by keyword count
            scores[domain.value] = score / len(keywords) if keywords else 0

        if scores:
            # Find primary domain
            max_domain = max(scores, key=scores.get)
            max_score = scores[max_domain]

            metrics.domain_scores = {k: round(v, 3) for k, v in scores.items() if v > 0}

            if max_score > 0.1:  # Minimum threshold
                metrics.primary_domain = DomainType(max_domain)
                metrics.domain_confidence = min(max_score * 2, 1.0)
            else:
                metrics.primary_domain = DomainType.GENERAL
                metrics.domain_confidence = 0.5

    def _analyze_text_length(self, metrics: ComplexityMetrics, lengths: list[int]) -> None:
        """Analyze text length distribution."""
        if not lengths:
            return

        avg_len = sum(lengths) / len(lengths)
        max_len = max(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

        metrics.avg_text_length = int(avg_len)
        metrics.max_text_length = max_len
        metrics.length_variance = variance

        # Categorize
        if avg_len < 500:
            metrics.length_category = "short"
        elif avg_len < 2000:
            metrics.length_category = "medium"
        elif avg_len < 5000:
            metrics.length_category = "long"
        else:
            metrics.length_category = "very_long"

    def _analyze_structure(
        self,
        metrics: ComplexityMetrics,
        samples: list[dict[str, Any]],
        schema_info: Optional[dict[str, Any]],
    ) -> None:
        """Analyze structure complexity."""
        if not samples:
            return

        # Average field count
        field_counts = [len(s) for s in samples]
        metrics.avg_field_count = int(sum(field_counts) / len(field_counts))

        # Check for nested structures
        max_depth = 0
        for sample in samples[:10]:  # Check first 10
            depth = self._get_max_depth(sample)
            max_depth = max(max_depth, depth)

        metrics.nesting_depth = max_depth
        metrics.has_nested_structure = max_depth > 2

        # Categorize
        if metrics.avg_field_count <= 3 and max_depth <= 2:
            metrics.structure_category = "simple"
        elif metrics.avg_field_count <= 8 and max_depth <= 4:
            metrics.structure_category = "moderate"
        else:
            metrics.structure_category = "complex"

    def _get_max_depth(self, obj: Any, current_depth: int = 1) -> int:
        """Get maximum nesting depth of an object."""
        if current_depth > 10:
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_max_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_max_depth(item, current_depth + 1) for item in obj[:5])
        return current_depth

    def _analyze_content(
        self,
        metrics: ComplexityMetrics,
        combined_text: str,
        all_texts: list[str],
    ) -> None:
        """Analyze content complexity indicators."""
        if not combined_text:
            return

        # Vocabulary richness (type-token ratio)
        words = re.findall(r"\b\w+\b", combined_text.lower())
        if words:
            unique_words = set(words)
            metrics.vocabulary_richness = len(unique_words) / len(words)

        # Average sentence length
        sentences = re.split(r"[.!?]+", combined_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            word_counts = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
            metrics.avg_sentence_length = sum(word_counts) / len(word_counts)

        # Technical term density
        if words:
            tech_count = sum(1 for w in words if w in self.TECHNICAL_TERMS)
            metrics.technical_term_density = tech_count / len(words)

        # Code density
        code_patterns = [
            r"```[\s\S]*?```",
            r"def\s+\w+\s*\(",
            r"function\s+\w+\s*\(",
            r"class\s+\w+",
            r'\{\s*\n\s*"',
        ]
        code_matches = sum(len(re.findall(p, combined_text)) for p in code_patterns)
        metrics.code_density = min(code_matches / max(len(all_texts), 1), 1.0)

    def _analyze_quality(
        self,
        metrics: ComplexityMetrics,
        rubrics: Optional[list[str]],
    ) -> None:
        """Analyze quality requirements from rubrics."""
        if not rubrics:
            metrics.has_rubrics = False
            metrics.rubric_complexity = "none"
            metrics.quality_requirement = "standard"
            return

        metrics.has_rubrics = True

        # Analyze rubric complexity
        avg_rubric_len = sum(len(r) for r in rubrics) / len(rubrics)
        unique_patterns = len(set(rubrics))

        if avg_rubric_len < 50 and unique_patterns < 10:
            metrics.rubric_complexity = "simple"
            metrics.quality_requirement = "standard"
        elif avg_rubric_len < 150 and unique_patterns < 50:
            metrics.rubric_complexity = "detailed"
            metrics.quality_requirement = "high"
        else:
            metrics.rubric_complexity = "expert"
            metrics.quality_requirement = "expert"

    def _calculate_multipliers(self, metrics: ComplexityMetrics) -> None:
        """Calculate final time and cost multipliers."""
        # Base domain multiplier
        domain_mult = DOMAIN_DIFFICULTY.get(metrics.primary_domain, 1.0)
        expert_mult = DOMAIN_EXPERT_MULTIPLIER.get(metrics.primary_domain, 1.0)

        # Length multiplier
        length_mult = {
            "short": 0.8,
            "medium": 1.0,
            "long": 1.5,
            "very_long": 2.0,
        }.get(metrics.length_category, 1.0)

        # Structure multiplier
        structure_mult = {
            "simple": 0.9,
            "moderate": 1.0,
            "complex": 1.3,
        }.get(metrics.structure_category, 1.0)

        # Quality multiplier
        quality_mult = {
            "basic": 0.8,
            "standard": 1.0,
            "high": 1.3,
            "expert": 1.8,
        }.get(metrics.quality_requirement, 1.0)

        # Content complexity adjustment
        content_mult = 1.0
        if metrics.code_density > 0.3:
            content_mult += 0.3
        if metrics.technical_term_density > 0.05:
            content_mult += 0.2
        if metrics.vocabulary_richness > 0.5:
            content_mult += 0.1

        # Final multipliers
        metrics.time_multiplier = (
            domain_mult * length_mult * structure_mult * quality_mult * content_mult
        )
        metrics.cost_multiplier = expert_mult * length_mult * quality_mult * content_mult

        # Difficulty score (1-5 scale)
        raw_difficulty = (domain_mult + length_mult + structure_mult + quality_mult) / 4
        metrics.difficulty_score = min(max(raw_difficulty * 2, 1.0), 5.0)
