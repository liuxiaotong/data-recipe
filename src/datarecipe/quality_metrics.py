"""Quality metrics analysis for datasets."""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Any

from datarecipe.schema import Recipe


@dataclass
class DiversityMetrics:
    """Diversity metrics for a dataset."""

    unique_token_ratio: float  # Unique tokens / total tokens
    vocabulary_size: int  # Number of unique tokens
    semantic_diversity: float  # Semantic similarity variance (0-1)
    ngram_diversity: dict = field(default_factory=dict)  # n-gram diversity scores

    def to_dict(self) -> dict:
        return {
            "unique_token_ratio": round(self.unique_token_ratio, 4),
            "vocabulary_size": self.vocabulary_size,
            "semantic_diversity": round(self.semantic_diversity, 4),
            "ngram_diversity": self.ngram_diversity,
        }


@dataclass
class ConsistencyMetrics:
    """Consistency metrics for a dataset."""

    format_consistency: float  # How consistent is the format (0-1)
    structure_score: float  # Structural consistency (0-1)
    field_completeness: float  # Ratio of non-empty fields (0-1)
    length_variance: float  # Normalized length variance

    def to_dict(self) -> dict:
        return {
            "format_consistency": round(self.format_consistency, 4),
            "structure_score": round(self.structure_score, 4),
            "field_completeness": round(self.field_completeness, 4),
            "length_variance": round(self.length_variance, 4),
        }


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a dataset."""

    avg_length: float  # Average text length in characters
    avg_tokens: float  # Average token count
    vocabulary_richness: float  # Type-token ratio
    avg_sentence_length: float  # Average words per sentence
    readability_score: float  # Flesch-Kincaid or similar (0-100)

    def to_dict(self) -> dict:
        return {
            "avg_length": round(self.avg_length, 2),
            "avg_tokens": round(self.avg_tokens, 2),
            "vocabulary_richness": round(self.vocabulary_richness, 4),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "readability_score": round(self.readability_score, 2),
        }


@dataclass
class AIDetectionMetrics:
    """AI content detection metrics."""

    ai_probability: float  # Probability of AI-generated (0-1)
    confidence: float  # Detection confidence (0-1)
    indicators: list[str] = field(default_factory=list)  # Detection indicators

    def to_dict(self) -> dict:
        return {
            "ai_probability": round(self.ai_probability, 4),
            "confidence": round(self.confidence, 4),
            "indicators": self.indicators,
        }


@dataclass
class QualityGateRule:
    """A single quality gate rule (pass/fail)."""

    gate_id: str
    name: str
    metric: str  # dot-path into QualityReport, e.g. "overall_score", "diversity.unique_token_ratio"
    operator: str  # >=, <=, >, <, ==, !=
    threshold: float
    severity: str = "blocker"  # blocker, warning

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "severity": self.severity,
        }


@dataclass
class GateResult:
    """Result of evaluating a single gate."""

    gate: QualityGateRule
    actual_value: float
    passed: bool
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate.gate_id,
            "name": self.gate.name,
            "metric": self.gate.metric,
            "threshold": self.gate.threshold,
            "operator": self.gate.operator,
            "actual_value": round(self.actual_value, 4),
            "passed": self.passed,
            "severity": self.gate.severity,
            "message": self.message,
        }


@dataclass
class QualityGateReport:
    """Aggregated gate evaluation report."""

    passed: bool  # overall pass (no blocker failures)
    results: list["GateResult"] = field(default_factory=list)
    blocking_failures: list["GateResult"] = field(default_factory=list)
    warnings: list["GateResult"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
            "blocking_failures": [r.to_dict() for r in self.blocking_failures],
            "warnings": [r.to_dict() for r in self.warnings],
        }


# Default quality gates
DEFAULT_QUALITY_GATES: list[QualityGateRule] = [
    QualityGateRule("min_overall_score", "最低综合分", "overall_score", ">=", 60, "blocker"),
    QualityGateRule("min_diversity", "最低多样性", "diversity.unique_token_ratio", ">=", 0.05, "blocker"),
    QualityGateRule("min_consistency", "最低格式一致性", "consistency.format_consistency", ">=", 0.5, "blocker"),
    QualityGateRule("max_ai_probability", "AI 内容上限", "ai_detection.ai_probability", "<=", 0.5, "warning"),
    QualityGateRule("min_completeness", "最低字段完整性", "consistency.field_completeness", ">=", 0.8, "warning"),
]


@dataclass
class QualityReport:
    """Complete quality analysis report."""

    diversity: DiversityMetrics
    consistency: ConsistencyMetrics
    complexity: ComplexityMetrics
    ai_detection: Optional[AIDetectionMetrics] = None
    overall_score: float = 0.0  # 0-100
    recommendations: list[str] = field(default_factory=list)
    sample_size: int = 0
    warnings: list[str] = field(default_factory=list)
    gate_report: Optional[QualityGateReport] = None

    def to_dict(self) -> dict:
        result = {
            "diversity": self.diversity.to_dict(),
            "consistency": self.consistency.to_dict(),
            "complexity": self.complexity.to_dict(),
            "overall_score": round(self.overall_score, 2),
            "recommendations": self.recommendations,
            "sample_size": self.sample_size,
            "warnings": self.warnings,
        }
        if self.ai_detection:
            result["ai_detection"] = self.ai_detection.to_dict()
        if self.gate_report:
            result["gate_report"] = self.gate_report.to_dict()
        return result


class QualityAnalyzer:
    """Analyzer for dataset quality metrics."""

    def __init__(self, use_embeddings: bool = False):
        """Initialize the quality analyzer.

        Args:
            use_embeddings: Whether to use sentence embeddings for semantic analysis
        """
        self.use_embeddings = use_embeddings
        self._embedder = None

    def analyze_sample(
        self,
        data: list[dict],
        text_field: str = "text",
        detect_ai: bool = False,
    ) -> QualityReport:
        """Analyze quality metrics from a data sample.

        Args:
            data: List of data examples (dicts)
            text_field: Field name containing the text to analyze
            detect_ai: Whether to run AI detection

        Returns:
            QualityReport with analysis results
        """
        if not data:
            return QualityReport(
                diversity=DiversityMetrics(0, 0, 0),
                consistency=ConsistencyMetrics(0, 0, 0, 0),
                complexity=ComplexityMetrics(0, 0, 0, 0, 0),
                overall_score=0,
                recommendations=["No data to analyze"],
                sample_size=0,
            )

        # Extract text content
        texts = self._extract_texts(data, text_field)

        if not texts:
            return QualityReport(
                diversity=DiversityMetrics(0, 0, 0),
                consistency=ConsistencyMetrics(0, 0, 0, 0),
                complexity=ComplexityMetrics(0, 0, 0, 0, 0),
                overall_score=0,
                recommendations=[f"No text found in field '{text_field}'"],
                sample_size=len(data),
                warnings=[f"Could not extract text from field '{text_field}'"],
            )

        # Calculate metrics
        diversity = self._calculate_diversity(texts)
        consistency = self._calculate_consistency(data, texts)
        complexity = self._calculate_complexity(texts)

        # AI detection (if requested)
        ai_detection = None
        if detect_ai:
            ai_detection = self._detect_ai_content(texts)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            diversity, consistency, complexity, ai_detection
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            diversity, consistency, complexity, ai_detection
        )

        # Generate warnings
        warnings = self._generate_warnings(data, texts, text_field)

        return QualityReport(
            diversity=diversity,
            consistency=consistency,
            complexity=complexity,
            ai_detection=ai_detection,
            overall_score=overall_score,
            recommendations=recommendations,
            sample_size=len(data),
            warnings=warnings,
        )

    def analyze_from_huggingface(
        self,
        dataset_id: str,
        text_field: str = "text",
        sample_size: int = 1000,
        split: str = "train",
        detect_ai: bool = False,
    ) -> QualityReport:
        """Analyze quality metrics from a HuggingFace dataset.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "Anthropic/hh-rlhf")
            text_field: Field name containing the text
            sample_size: Number of examples to sample
            split: Dataset split to use
            detect_ai: Whether to run AI detection

        Returns:
            QualityReport with analysis results
        """
        try:
            from datasets import load_dataset
        except ImportError:
            return QualityReport(
                diversity=DiversityMetrics(0, 0, 0),
                consistency=ConsistencyMetrics(0, 0, 0, 0),
                complexity=ComplexityMetrics(0, 0, 0, 0, 0),
                overall_score=0,
                recommendations=["Install 'datasets' package: pip install datasets"],
                sample_size=0,
                warnings=["datasets package not installed"],
            )

        try:
            # Load dataset
            dataset = load_dataset(dataset_id, split=split, streaming=True)

            # Sample data
            data = []
            for i, example in enumerate(dataset):
                if i >= sample_size:
                    break
                data.append(example)

            # Auto-detect text field if not found
            if data and text_field not in data[0]:
                text_field = self._detect_text_field(data[0])

            return self.analyze_sample(data, text_field, detect_ai)

        except Exception as e:
            return QualityReport(
                diversity=DiversityMetrics(0, 0, 0),
                consistency=ConsistencyMetrics(0, 0, 0, 0),
                complexity=ComplexityMetrics(0, 0, 0, 0, 0),
                overall_score=0,
                recommendations=[f"Failed to load dataset: {str(e)}"],
                sample_size=0,
                warnings=[f"Dataset loading error: {str(e)}"],
            )

    def _extract_texts(self, data: list[dict], text_field: str) -> list[str]:
        """Extract text content from data."""
        texts = []
        for item in data:
            text = self._get_nested_field(item, text_field)
            if text is not None:
                if isinstance(text, str):
                    texts.append(text)
                elif isinstance(text, list):
                    # Handle list of texts (e.g., conversation turns)
                    for t in text:
                        if isinstance(t, str):
                            texts.append(t)
                        elif isinstance(t, dict):
                            # Try common content fields
                            for key in ["content", "text", "message", "value"]:
                                if key in t:
                                    texts.append(str(t[key]))
                                    break
        return texts

    def _get_nested_field(self, item: dict, field: str) -> Any:
        """Get a potentially nested field from a dict."""
        if "." in field:
            parts = field.split(".")
            value = item
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        return item.get(field)

    def _detect_text_field(self, example: dict) -> str:
        """Auto-detect the text field in a dataset example."""
        # Priority order for common text fields
        candidates = [
            "text",
            "content",
            "message",
            "input",
            "output",
            "question",
            "answer",
            "prompt",
            "response",
            "instruction",
            "chosen",
            "rejected",
        ]

        for candidate in candidates:
            if candidate in example:
                value = example[candidate]
                if isinstance(value, str) and len(value) > 10:
                    return candidate

        # Check nested fields
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 50:
                return key
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], str):
                    return key
                if isinstance(value[0], dict):
                    for subkey in ["content", "text", "message"]:
                        if subkey in value[0]:
                            return f"{key}.{subkey}"

        return "text"  # Default fallback

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Simple word tokenization
        return re.findall(r"\b\w+\b", text.lower())

    def _calculate_diversity(self, texts: list[str]) -> DiversityMetrics:
        """Calculate diversity metrics."""
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))

        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))

        if total_tokens == 0:
            return DiversityMetrics(0, 0, 0)

        unique_ratio = unique_tokens / total_tokens

        # Calculate n-gram diversity
        ngram_diversity = {}
        for n in [2, 3]:
            ngrams = []
            for text in texts:
                tokens = self._tokenize(text)
                for i in range(len(tokens) - n + 1):
                    ngrams.append(tuple(tokens[i : i + n]))
            if ngrams:
                ngram_diversity[f"{n}-gram"] = len(set(ngrams)) / len(ngrams)

        # Semantic diversity (simplified without embeddings)
        semantic_diversity = self._calculate_semantic_diversity(texts)

        return DiversityMetrics(
            unique_token_ratio=unique_ratio,
            vocabulary_size=unique_tokens,
            semantic_diversity=semantic_diversity,
            ngram_diversity=ngram_diversity,
        )

    def _calculate_semantic_diversity(self, texts: list[str]) -> float:
        """Calculate semantic diversity score."""
        if not self.use_embeddings:
            # Fallback: use length-normalized Jaccard diversity
            if len(texts) < 2:
                return 0.0

            similarities = []
            sample = texts[: min(100, len(texts))]  # Sample for efficiency

            for i in range(len(sample)):
                for j in range(i + 1, min(i + 10, len(sample))):
                    tokens_i = set(self._tokenize(sample[i]))
                    tokens_j = set(self._tokenize(sample[j]))
                    if tokens_i or tokens_j:
                        jaccard = len(tokens_i & tokens_j) / len(tokens_i | tokens_j)
                        similarities.append(jaccard)

            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                return 1 - avg_similarity  # Diversity is inverse of similarity

            return 0.5

        # Use embeddings if available
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            sample = texts[: min(500, len(texts))]
            embeddings = self._embedder.encode(sample)

            # Calculate pairwise cosine similarities
            import numpy as np

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / (norms + 1e-10)

            # Sample pairs for efficiency
            n_samples = min(1000, len(sample) * (len(sample) - 1) // 2)
            similarities = []

            for _ in range(n_samples):
                i, j = np.random.choice(len(sample), 2, replace=False)
                sim = np.dot(normalized[i], normalized[j])
                similarities.append(sim)

            avg_similarity = np.mean(similarities)
            return float(1 - avg_similarity)

        except ImportError:
            return 0.5  # Default if embeddings not available

    def _calculate_consistency(
        self, data: list[dict], texts: list[str]
    ) -> ConsistencyMetrics:
        """Calculate consistency metrics."""
        # Format consistency: check if examples have same structure
        if not data:
            return ConsistencyMetrics(0, 0, 0, 0)

        # Check field presence across examples
        all_keys = set()
        key_counts = Counter()

        for item in data:
            keys = self._flatten_keys(item)
            all_keys.update(keys)
            for key in keys:
                key_counts[key] += 1

        # Format consistency: ratio of fields present in all examples
        n = len(data)
        consistent_fields = sum(1 for count in key_counts.values() if count == n)
        format_consistency = consistent_fields / len(all_keys) if all_keys else 1.0

        # Structure score: normalized entropy of field patterns
        patterns = Counter()
        for item in data:
            pattern = tuple(sorted(self._flatten_keys(item)))
            patterns[pattern] += 1

        n_patterns = len(patterns)
        structure_score = 1.0 / (1 + math.log(n_patterns + 1))

        # Field completeness
        total_possible = len(data) * len(all_keys)
        total_present = sum(key_counts.values())
        field_completeness = total_present / total_possible if total_possible > 0 else 1.0

        # Length variance
        if texts:
            lengths = [len(t) for t in texts]
            mean_length = sum(lengths) / len(lengths)
            variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
            std_dev = math.sqrt(variance)
            # Normalize by mean (coefficient of variation)
            length_variance = std_dev / mean_length if mean_length > 0 else 0
        else:
            length_variance = 0

        return ConsistencyMetrics(
            format_consistency=format_consistency,
            structure_score=structure_score,
            field_completeness=field_completeness,
            length_variance=length_variance,
        )

    def _flatten_keys(self, d: dict, prefix: str = "") -> set[str]:
        """Flatten nested dict keys."""
        keys = set()
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            keys.add(key)
            if isinstance(v, dict):
                keys.update(self._flatten_keys(v, key))
        return keys

    def _calculate_complexity(self, texts: list[str]) -> ComplexityMetrics:
        """Calculate complexity metrics."""
        if not texts:
            return ComplexityMetrics(0, 0, 0, 0, 0)

        lengths = [len(t) for t in texts]
        avg_length = sum(lengths) / len(lengths)

        token_counts = [len(self._tokenize(t)) for t in texts]
        avg_tokens = sum(token_counts) / len(token_counts)

        # Vocabulary richness (type-token ratio)
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))
        vocabulary_richness = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0

        # Average sentence length
        sentence_lengths = []
        for text in texts:
            sentences = re.split(r"[.!?]+", text)
            for sent in sentences:
                words = self._tokenize(sent)
                if words:
                    sentence_lengths.append(len(words))

        avg_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        )

        # Simplified readability score (based on sentence and word length)
        # Approximation of Flesch Reading Ease
        if avg_sentence_length > 0 and avg_tokens > 0:
            avg_syllables = avg_length / (avg_tokens * 3)  # Rough estimate
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
            readability = max(0, min(100, readability))
        else:
            readability = 50

        return ComplexityMetrics(
            avg_length=avg_length,
            avg_tokens=avg_tokens,
            vocabulary_richness=vocabulary_richness,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability,
        )

    def _detect_ai_content(self, texts: list[str]) -> AIDetectionMetrics:
        """Detect potential AI-generated content using heuristics."""
        indicators = []
        scores = []

        # Sample texts for analysis
        sample = texts[: min(100, len(texts))]

        for text in sample:
            text_score = 0
            text_indicators = []

            # Check for common AI patterns
            ai_patterns = [
                (r"\bAs an AI\b", "Self-reference as AI"),
                (r"\bI don't have personal\b", "Personal experience disclaimer"),
                (r"\bI cannot provide\b", "Capability disclaimer"),
                (r"\bIt's important to note\b", "Hedging phrase"),
                (r"\bIn summary\b", "Summary phrase"),
                (r"\bFurthermore\b", "Formal transition"),
                (r"\bMoreover\b", "Formal transition"),
                (r"\bNevertheless\b", "Formal transition"),
                (r"\bHowever, it's worth\b", "Hedging phrase"),
            ]

            for pattern, indicator in ai_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    text_score += 0.1
                    if indicator not in text_indicators:
                        text_indicators.append(indicator)

            # Check for uniform sentence structure
            sentences = re.split(r"[.!?]+", text)
            if len(sentences) >= 3:
                lengths = [len(s.split()) for s in sentences if s.strip()]
                if lengths:
                    variance = sum((l - sum(lengths) / len(lengths)) ** 2 for l in lengths)
                    variance /= len(lengths)
                    if variance < 5:  # Very uniform
                        text_score += 0.1
                        text_indicators.append("Uniform sentence length")

            # Check for excessive politeness/formality
            polite_patterns = [
                r"\bplease\b",
                r"\bthank you\b",
                r"\bI hope this helps\b",
                r"\bfeel free to\b",
            ]
            polite_count = sum(
                1 for p in polite_patterns if re.search(p, text, re.IGNORECASE)
            )
            if polite_count >= 2:
                text_score += 0.1
                text_indicators.append("Excessive politeness markers")

            scores.append(min(1.0, text_score))
            indicators.extend(text_indicators)

        avg_score = sum(scores) / len(scores) if scores else 0
        indicator_counts = Counter(indicators)
        top_indicators = [ind for ind, _ in indicator_counts.most_common(5)]

        return AIDetectionMetrics(
            ai_probability=avg_score,
            confidence=0.6 if avg_score > 0.2 else 0.4,  # Heuristic confidence
            indicators=top_indicators,
        )

    def _calculate_overall_score(
        self,
        diversity: DiversityMetrics,
        consistency: ConsistencyMetrics,
        complexity: ComplexityMetrics,
        ai_detection: Optional[AIDetectionMetrics],
    ) -> float:
        """Calculate overall quality score (0-100)."""
        score = 0

        # Diversity (30 points max)
        diversity_score = 0
        diversity_score += min(10, diversity.unique_token_ratio * 50)  # Up to 10
        diversity_score += min(10, diversity.semantic_diversity * 15)  # Up to 10
        ngram_avg = sum(diversity.ngram_diversity.values()) / max(
            1, len(diversity.ngram_diversity)
        )
        diversity_score += min(10, ngram_avg * 15)  # Up to 10
        score += diversity_score

        # Consistency (30 points max)
        consistency_score = 0
        consistency_score += consistency.format_consistency * 15  # Up to 15
        consistency_score += consistency.structure_score * 10  # Up to 10
        consistency_score += consistency.field_completeness * 5  # Up to 5
        score += consistency_score

        # Complexity (25 points max)
        complexity_score = 0
        # Reward moderate complexity
        if 50 < complexity.avg_tokens < 500:
            complexity_score += 10
        elif complexity.avg_tokens > 0:
            complexity_score += 5
        complexity_score += min(10, complexity.vocabulary_richness * 30)
        complexity_score += min(5, (100 - abs(complexity.readability_score - 60)) / 20)
        score += complexity_score

        # AI detection penalty (up to 15 points)
        if ai_detection:
            # High AI probability reduces score
            ai_penalty = ai_detection.ai_probability * 15
            score += 15 - ai_penalty
        else:
            score += 15  # No detection = no penalty

        return min(100, max(0, score))

    def _generate_recommendations(
        self,
        diversity: DiversityMetrics,
        consistency: ConsistencyMetrics,
        complexity: ComplexityMetrics,
        ai_detection: Optional[AIDetectionMetrics],
    ) -> list[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        # Diversity recommendations
        if diversity.unique_token_ratio < 0.1:
            recommendations.append("Low vocabulary diversity - consider adding more varied content")
        if diversity.semantic_diversity < 0.3:
            recommendations.append("Low semantic diversity - examples may be too similar")

        # Consistency recommendations
        if consistency.format_consistency < 0.7:
            recommendations.append(
                "Inconsistent data format - consider standardizing field structure"
            )
        if consistency.length_variance > 2.0:
            recommendations.append("High length variance - consider normalizing text lengths")

        # Complexity recommendations
        if complexity.avg_tokens < 20:
            recommendations.append("Very short texts - may lack sufficient context")
        if complexity.avg_tokens > 1000:
            recommendations.append(
                "Very long texts - consider chunking or summarizing"
            )
        if complexity.readability_score < 30:
            recommendations.append(
                "Low readability - texts may be too complex or technical"
            )

        # AI detection recommendations
        if ai_detection and ai_detection.ai_probability > 0.5:
            recommendations.append(
                "High AI-generated content probability - may affect model diversity"
            )
            if ai_detection.indicators:
                recommendations.append(
                    f"Common AI indicators: {', '.join(ai_detection.indicators[:3])}"
                )

        if not recommendations:
            recommendations.append("Dataset quality looks good!")

        return recommendations

    def _generate_warnings(
        self, data: list[dict], texts: list[str], text_field: str
    ) -> list[str]:
        """Generate warnings about potential issues."""
        warnings = []

        if len(texts) < len(data):
            missing = len(data) - len(texts)
            warnings.append(f"{missing} examples missing text in field '{text_field}'")

        if len(texts) < 100:
            warnings.append("Small sample size may not be representative")

        # Check for empty or very short texts
        short_count = sum(1 for t in texts if len(t) < 10)
        if short_count > len(texts) * 0.1:
            warnings.append(f"{short_count} texts are very short (<10 chars)")

        return warnings

    # --- Quality Gate evaluation (Upgrade 4) ---

    def evaluate_gates(
        self,
        report: QualityReport,
        gates: Optional[list[QualityGateRule]] = None,
    ) -> QualityGateReport:
        """Evaluate quality gates against a report.

        Args:
            report: QualityReport to evaluate
            gates: list of QualityGateRule (defaults to DEFAULT_QUALITY_GATES)

        Returns:
            QualityGateReport with pass/fail results
        """
        if gates is None:
            gates = DEFAULT_QUALITY_GATES

        results: list[GateResult] = []
        blocking: list[GateResult] = []
        warns: list[GateResult] = []

        for gate in gates:
            actual = self._extract_metric(report, gate.metric)
            if actual is None:
                # Metric not available (e.g. ai_detection not run) — skip
                results.append(GateResult(
                    gate=gate, actual_value=0.0, passed=True,
                    message=f"Metric '{gate.metric}' not available, skipped",
                ))
                continue

            passed = self._compare(actual, gate.operator, gate.threshold)
            msg = (
                f"{gate.name}: {actual:.4f} {gate.operator} {gate.threshold} → "
                f"{'PASS' if passed else 'FAIL'}"
            )
            gr = GateResult(gate=gate, actual_value=actual, passed=passed, message=msg)
            results.append(gr)

            if not passed:
                if gate.severity == "blocker":
                    blocking.append(gr)
                else:
                    warns.append(gr)

        gate_report = QualityGateReport(
            passed=len(blocking) == 0,
            results=results,
            blocking_failures=blocking,
            warnings=warns,
        )
        report.gate_report = gate_report
        return gate_report

    def _extract_metric(self, report: QualityReport, metric_path: str) -> Optional[float]:
        """Extract a metric value from a QualityReport by dot-path."""
        parts = metric_path.split(".")
        obj: Any = report
        for part in parts:
            if obj is None:
                return None
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        if isinstance(obj, (int, float)):
            return float(obj)
        return None

    @staticmethod
    def _compare(actual: float, operator: str, threshold: float) -> bool:
        """Compare actual value against threshold using operator."""
        ops = {
            ">=": actual >= threshold,
            "<=": actual <= threshold,
            ">": actual > threshold,
            "<": actual < threshold,
            "==": actual == threshold,
            "!=": actual != threshold,
        }
        return ops.get(operator, False)
