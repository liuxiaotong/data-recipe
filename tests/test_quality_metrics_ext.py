"""Extended unit tests for quality_metrics.py.

Covers QualityAnalyzer methods, dataclass serialization, and edge cases
not covered by test_quality_gates.py.
"""

import unittest
from unittest.mock import MagicMock, patch

from datarecipe.quality_metrics import (
    AIDetectionMetrics,
    ComplexityMetrics,
    ConsistencyMetrics,
    DiversityMetrics,
    GateResult,
    QualityAnalyzer,
    QualityGateReport,
    QualityGateRule,
    QualityReport,
)


# ==================== Dataclass to_dict Tests ====================


class TestDiversityMetricsToDict(unittest.TestCase):
    """Test DiversityMetrics.to_dict() serialization."""

    def test_to_dict_basic(self):
        m = DiversityMetrics(
            unique_token_ratio=0.12345,
            vocabulary_size=500,
            semantic_diversity=0.67891,
            ngram_diversity={"2-gram": 0.8, "3-gram": 0.9},
        )
        d = m.to_dict()
        self.assertEqual(d["unique_token_ratio"], 0.1235)
        self.assertEqual(d["vocabulary_size"], 500)
        self.assertEqual(d["semantic_diversity"], 0.6789)
        self.assertEqual(d["ngram_diversity"], {"2-gram": 0.8, "3-gram": 0.9})

    def test_to_dict_zero_values(self):
        m = DiversityMetrics(0, 0, 0)
        d = m.to_dict()
        self.assertEqual(d["unique_token_ratio"], 0.0)
        self.assertEqual(d["vocabulary_size"], 0)
        self.assertEqual(d["semantic_diversity"], 0.0)
        self.assertEqual(d["ngram_diversity"], {})


class TestConsistencyMetricsToDict(unittest.TestCase):
    """Test ConsistencyMetrics.to_dict() serialization."""

    def test_to_dict_rounds_values(self):
        m = ConsistencyMetrics(
            format_consistency=0.123456,
            structure_score=0.987654,
            field_completeness=0.555555,
            length_variance=1.111111,
        )
        d = m.to_dict()
        self.assertEqual(d["format_consistency"], 0.1235)
        self.assertEqual(d["structure_score"], 0.9877)
        self.assertEqual(d["field_completeness"], 0.5556)
        self.assertEqual(d["length_variance"], 1.1111)


class TestComplexityMetricsToDict(unittest.TestCase):
    """Test ComplexityMetrics.to_dict() serialization."""

    def test_to_dict_rounds_values(self):
        m = ComplexityMetrics(
            avg_length=123.456,
            avg_tokens=45.678,
            vocabulary_richness=0.12345,
            avg_sentence_length=12.345,
            readability_score=67.891,
        )
        d = m.to_dict()
        self.assertEqual(d["avg_length"], 123.46)
        self.assertEqual(d["avg_tokens"], 45.68)
        self.assertEqual(d["vocabulary_richness"], 0.1235)
        self.assertEqual(d["avg_sentence_length"], 12.35)
        self.assertEqual(d["readability_score"], 67.89)


class TestAIDetectionMetricsToDict(unittest.TestCase):
    """Test AIDetectionMetrics.to_dict() serialization."""

    def test_to_dict_basic(self):
        m = AIDetectionMetrics(
            ai_probability=0.12345,
            confidence=0.98765,
            indicators=["Self-reference as AI", "Hedging phrase"],
        )
        d = m.to_dict()
        self.assertEqual(d["ai_probability"], 0.1235)
        self.assertEqual(d["confidence"], 0.9877)
        self.assertEqual(d["indicators"], ["Self-reference as AI", "Hedging phrase"])

    def test_to_dict_empty_indicators(self):
        m = AIDetectionMetrics(ai_probability=0.0, confidence=0.0)
        d = m.to_dict()
        self.assertEqual(d["indicators"], [])


class TestQualityReportToDict(unittest.TestCase):
    """Test QualityReport.to_dict() serialization."""

    def _make_report(self, ai_detection=None, gate_report=None):
        return QualityReport(
            diversity=DiversityMetrics(0.15, 1000, 0.6),
            consistency=ConsistencyMetrics(0.9, 0.8, 0.95, 0.3),
            complexity=ComplexityMetrics(200.0, 50.0, 0.3, 15.0, 60.0),
            ai_detection=ai_detection,
            overall_score=75.5,
            recommendations=["Looks good"],
            sample_size=100,
            warnings=["Small sample"],
            gate_report=gate_report,
        )

    def test_to_dict_without_optional_fields(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertIn("diversity", d)
        self.assertIn("consistency", d)
        self.assertIn("complexity", d)
        self.assertEqual(d["overall_score"], 75.5)
        self.assertEqual(d["recommendations"], ["Looks good"])
        self.assertEqual(d["sample_size"], 100)
        self.assertEqual(d["warnings"], ["Small sample"])
        self.assertNotIn("ai_detection", d)
        self.assertNotIn("gate_report", d)

    def test_to_dict_with_ai_detection(self):
        ai = AIDetectionMetrics(0.3, 0.6, ["Hedging phrase"])
        report = self._make_report(ai_detection=ai)
        d = report.to_dict()
        self.assertIn("ai_detection", d)
        self.assertEqual(d["ai_detection"]["ai_probability"], 0.3)

    def test_to_dict_with_gate_report(self):
        gate = QualityGateRule("g1", "Test", "overall_score", ">=", 60, "blocker")
        gr = GateResult(gate=gate, actual_value=75.0, passed=True, message="ok")
        gate_report = QualityGateReport(passed=True, results=[gr])
        report = self._make_report(gate_report=gate_report)
        d = report.to_dict()
        self.assertIn("gate_report", d)
        self.assertTrue(d["gate_report"]["passed"])


# ==================== QualityAnalyzer: _tokenize ====================


class TestTokenize(unittest.TestCase):
    """Test _tokenize method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_basic_tokenization(self):
        tokens = self.analyzer._tokenize("Hello World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_punctuation_removed(self):
        tokens = self.analyzer._tokenize("Hello, world! How are you?")
        self.assertEqual(tokens, ["hello", "world", "how", "are", "you"])

    def test_empty_string(self):
        tokens = self.analyzer._tokenize("")
        self.assertEqual(tokens, [])

    def test_numbers_included(self):
        tokens = self.analyzer._tokenize("There are 42 items")
        self.assertIn("42", tokens)

    def test_case_insensitive(self):
        tokens = self.analyzer._tokenize("ABC Def GHI")
        self.assertEqual(tokens, ["abc", "def", "ghi"])


# ==================== QualityAnalyzer: _get_nested_field ====================


class TestGetNestedField(unittest.TestCase):
    """Test _get_nested_field method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_simple_field(self):
        item = {"text": "hello", "id": 1}
        self.assertEqual(self.analyzer._get_nested_field(item, "text"), "hello")

    def test_nested_field(self):
        item = {"meta": {"content": "nested value"}}
        self.assertEqual(
            self.analyzer._get_nested_field(item, "meta.content"), "nested value"
        )

    def test_deeply_nested_field(self):
        item = {"a": {"b": {"c": "deep"}}}
        self.assertEqual(self.analyzer._get_nested_field(item, "a.b.c"), "deep")

    def test_missing_field_returns_none(self):
        item = {"text": "hello"}
        self.assertIsNone(self.analyzer._get_nested_field(item, "missing"))

    def test_missing_nested_field_returns_none(self):
        item = {"meta": {"content": "val"}}
        self.assertIsNone(self.analyzer._get_nested_field(item, "meta.missing"))

    def test_missing_intermediate_returns_none(self):
        item = {"meta": "not_a_dict"}
        self.assertIsNone(self.analyzer._get_nested_field(item, "meta.content"))

    def test_no_dot_returns_get(self):
        item = {"x": 42}
        self.assertEqual(self.analyzer._get_nested_field(item, "x"), 42)
        self.assertIsNone(self.analyzer._get_nested_field(item, "y"))


# ==================== QualityAnalyzer: _extract_texts ====================


class TestExtractTexts(unittest.TestCase):
    """Test _extract_texts method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_simple_text_field(self):
        data = [
            {"text": "Hello"},
            {"text": "World"},
        ]
        texts = self.analyzer._extract_texts(data, "text")
        self.assertEqual(texts, ["Hello", "World"])

    def test_missing_field_skipped(self):
        data = [
            {"text": "Hello"},
            {"other": "World"},
        ]
        texts = self.analyzer._extract_texts(data, "text")
        self.assertEqual(texts, ["Hello"])

    def test_list_of_strings(self):
        data = [{"text": ["line1", "line2", "line3"]}]
        texts = self.analyzer._extract_texts(data, "text")
        self.assertEqual(texts, ["line1", "line2", "line3"])

    def test_list_of_dicts_with_content(self):
        data = [
            {
                "messages": [
                    {"content": "Hello", "role": "user"},
                    {"content": "Hi there", "role": "assistant"},
                ]
            }
        ]
        texts = self.analyzer._extract_texts(data, "messages")
        self.assertEqual(texts, ["Hello", "Hi there"])

    def test_list_of_dicts_with_text_key(self):
        data = [{"turns": [{"text": "Turn 1"}, {"text": "Turn 2"}]}]
        texts = self.analyzer._extract_texts(data, "turns")
        self.assertEqual(texts, ["Turn 1", "Turn 2"])

    def test_list_of_dicts_with_message_key(self):
        data = [{"items": [{"message": "Msg 1"}]}]
        texts = self.analyzer._extract_texts(data, "items")
        self.assertEqual(texts, ["Msg 1"])

    def test_list_of_dicts_with_value_key(self):
        data = [{"items": [{"value": "Val 1"}]}]
        texts = self.analyzer._extract_texts(data, "items")
        self.assertEqual(texts, ["Val 1"])

    def test_list_of_dicts_no_recognized_key(self):
        data = [{"items": [{"unknown_key": "ignored"}]}]
        texts = self.analyzer._extract_texts(data, "items")
        self.assertEqual(texts, [])

    def test_none_value_skipped(self):
        data = [{"text": None}]
        texts = self.analyzer._extract_texts(data, "text")
        self.assertEqual(texts, [])

    def test_nested_field_extraction(self):
        data = [{"meta": {"content": "nested text"}}]
        texts = self.analyzer._extract_texts(data, "meta.content")
        self.assertEqual(texts, ["nested text"])

    def test_mixed_types_in_list(self):
        data = [{"text": ["string_val", 42, {"content": "dict_val"}]}]
        texts = self.analyzer._extract_texts(data, "text")
        self.assertEqual(texts, ["string_val", "dict_val"])


# ==================== QualityAnalyzer: _detect_text_field ====================


class TestDetectTextField(unittest.TestCase):
    """Test _detect_text_field method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_detects_text_field(self):
        example = {"text": "This is a long enough text for detection", "id": 1}
        self.assertEqual(self.analyzer._detect_text_field(example), "text")

    def test_detects_content_field(self):
        example = {"content": "This is content long enough for detection", "id": 1}
        self.assertEqual(self.analyzer._detect_text_field(example), "content")

    def test_detects_instruction_field(self):
        example = {"instruction": "This instruction is long enough for detection", "id": 1}
        self.assertEqual(self.analyzer._detect_text_field(example), "instruction")

    def test_fallback_to_long_string_field(self):
        example = {
            "custom_field": "x" * 51,
            "short": "hi",
        }
        self.assertEqual(self.analyzer._detect_text_field(example), "custom_field")

    def test_fallback_to_list_of_strings(self):
        example = {"turns": ["Turn 1", "Turn 2"], "id": 1}
        self.assertEqual(self.analyzer._detect_text_field(example), "turns")

    def test_fallback_to_list_of_dicts_with_content(self):
        example = {
            "conversation": [{"content": "Hello"}, {"content": "Hi"}],
            "id": 1,
        }
        self.assertEqual(
            self.analyzer._detect_text_field(example), "conversation.content"
        )

    def test_default_fallback(self):
        example = {"id": 1, "num": 42}
        self.assertEqual(self.analyzer._detect_text_field(example), "text")

    def test_priority_order(self):
        """text is preferred over content, content over message, etc."""
        example = {
            "content": "This is content long enough for detection",
            "message": "This is a message long enough for detection",
        }
        self.assertEqual(self.analyzer._detect_text_field(example), "content")

    def test_short_candidate_skipped(self):
        """Fields < 10 chars are skipped even if in candidates list."""
        example = {"text": "short", "prompt": "This is a longer prompt for detection"}
        self.assertEqual(self.analyzer._detect_text_field(example), "prompt")


# ==================== QualityAnalyzer: _flatten_keys ====================


class TestFlattenKeys(unittest.TestCase):
    """Test _flatten_keys method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_flat_dict(self):
        keys = self.analyzer._flatten_keys({"a": 1, "b": 2})
        self.assertEqual(keys, {"a", "b"})

    def test_nested_dict(self):
        keys = self.analyzer._flatten_keys({"a": {"b": 1, "c": 2}})
        self.assertEqual(keys, {"a", "a.b", "a.c"})

    def test_deeply_nested(self):
        keys = self.analyzer._flatten_keys({"a": {"b": {"c": 1}}})
        self.assertEqual(keys, {"a", "a.b", "a.b.c"})

    def test_empty_dict(self):
        keys = self.analyzer._flatten_keys({})
        self.assertEqual(keys, set())

    def test_mixed_values(self):
        keys = self.analyzer._flatten_keys({"a": 1, "b": {"c": 2}, "d": "text"})
        self.assertEqual(keys, {"a", "b", "b.c", "d"})


# ==================== QualityAnalyzer: _calculate_diversity ====================


class TestCalculateDiversity(unittest.TestCase):
    """Test _calculate_diversity method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_empty_texts(self):
        result = self.analyzer._calculate_diversity([])
        self.assertEqual(result.unique_token_ratio, 0)
        self.assertEqual(result.vocabulary_size, 0)
        self.assertEqual(result.semantic_diversity, 0)

    def test_single_text(self):
        result = self.analyzer._calculate_diversity(["Hello world test"])
        self.assertGreater(result.unique_token_ratio, 0)
        self.assertGreater(result.vocabulary_size, 0)

    def test_all_unique_high_ratio(self):
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        result = self.analyzer._calculate_diversity(texts)
        self.assertEqual(result.unique_token_ratio, 1.0)

    def test_repetitive_low_ratio(self):
        texts = ["the the the the"] * 10
        result = self.analyzer._calculate_diversity(texts)
        self.assertAlmostEqual(result.unique_token_ratio, 1.0 / 40.0)
        self.assertEqual(result.vocabulary_size, 1)

    def test_ngram_diversity_computed(self):
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast red cat leaps across the sleepy hound",
        ]
        result = self.analyzer._calculate_diversity(texts)
        self.assertIn("2-gram", result.ngram_diversity)
        self.assertIn("3-gram", result.ngram_diversity)
        self.assertGreater(result.ngram_diversity["2-gram"], 0)
        self.assertGreater(result.ngram_diversity["3-gram"], 0)

    def test_texts_with_no_tokens(self):
        """Texts that produce no tokens (only punctuation)."""
        result = self.analyzer._calculate_diversity(["!!!", "???"])
        self.assertEqual(result.unique_token_ratio, 0)
        self.assertEqual(result.vocabulary_size, 0)


# ==================== QualityAnalyzer: _calculate_semantic_diversity ====================


class TestCalculateSemanticDiversity(unittest.TestCase):
    """Test _calculate_semantic_diversity method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer(use_embeddings=False)

    def test_single_text_returns_zero(self):
        result = self.analyzer._calculate_semantic_diversity(["Only one text"])
        self.assertEqual(result, 0.0)

    def test_empty_returns_zero(self):
        result = self.analyzer._calculate_semantic_diversity([])
        self.assertEqual(result, 0.0)

    def test_identical_texts_low_diversity(self):
        texts = ["the same text here"] * 20
        result = self.analyzer._calculate_semantic_diversity(texts)
        # Identical texts should have jaccard similarity = 1, diversity = 0
        self.assertAlmostEqual(result, 0.0, places=2)

    def test_diverse_texts_higher_diversity(self):
        texts = [
            "Machine learning algorithms for classification",
            "Cooking recipes for Italian pasta dishes",
            "Space exploration and planetary science",
            "Financial markets and stock trading strategies",
            "Ancient Roman history and architecture",
        ]
        result = self.analyzer._calculate_semantic_diversity(texts)
        self.assertGreater(result, 0.5)

    def test_two_different_texts(self):
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        result = self.analyzer._calculate_semantic_diversity(texts)
        # Completely different tokens -> jaccard = 0 -> diversity = 1.0
        self.assertAlmostEqual(result, 1.0, places=2)

    def test_samples_at_most_100(self):
        """Method should sample at most 100 texts."""
        texts = [f"text number {i} with unique content {i * 7}" for i in range(200)]
        result = self.analyzer._calculate_semantic_diversity(texts)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 1.0)

    def test_texts_with_empty_tokens(self):
        """Texts that produce empty token sets should be handled."""
        texts = ["!!!", "???", "hello world"]
        result = self.analyzer._calculate_semantic_diversity(texts)
        # Should not crash; result should be a valid float
        self.assertIsInstance(result, float)


# ==================== QualityAnalyzer: _calculate_consistency ====================


class TestCalculateConsistency(unittest.TestCase):
    """Test _calculate_consistency method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_empty_data(self):
        result = self.analyzer._calculate_consistency([], [])
        self.assertEqual(result.format_consistency, 0)
        self.assertEqual(result.structure_score, 0)
        self.assertEqual(result.field_completeness, 0)
        self.assertEqual(result.length_variance, 0)

    def test_uniform_data(self):
        data = [
            {"text": "Hello", "id": 1},
            {"text": "World", "id": 2},
            {"text": "Test", "id": 3},
        ]
        texts = ["Hello", "World", "Test"]
        result = self.analyzer._calculate_consistency(data, texts)
        self.assertEqual(result.format_consistency, 1.0)
        self.assertEqual(result.field_completeness, 1.0)

    def test_inconsistent_fields(self):
        data = [
            {"text": "a", "id": 1},
            {"text": "b", "extra": "x"},
            {"text": "c"},
        ]
        texts = ["a", "b", "c"]
        result = self.analyzer._calculate_consistency(data, texts)
        # "text" is in all 3, "id" in 1, "extra" in 1 -> 1/3 consistent
        self.assertLess(result.format_consistency, 1.0)

    def test_length_variance_zero_for_equal_lengths(self):
        data = [{"t": "x"}]
        texts = ["hello", "hello", "hello"]
        result = self.analyzer._calculate_consistency(data, texts)
        self.assertAlmostEqual(result.length_variance, 0.0)

    def test_length_variance_nonzero_for_varied_lengths(self):
        data = [{"t": "x"}]
        texts = ["a", "abcdefghij" * 10]
        result = self.analyzer._calculate_consistency(data, texts)
        self.assertGreater(result.length_variance, 0)

    def test_no_texts_zero_variance(self):
        data = [{"id": 1}]
        result = self.analyzer._calculate_consistency(data, [])
        self.assertEqual(result.length_variance, 0)

    def test_nested_dict_keys_flattened(self):
        data = [
            {"a": {"b": 1}, "c": 2},
            {"a": {"b": 2}, "c": 3},
        ]
        result = self.analyzer._calculate_consistency(data, ["x", "y"])
        # Keys: a, a.b, c -> all present in both -> consistency = 1.0
        self.assertEqual(result.format_consistency, 1.0)

    def test_structure_score_decreases_with_more_patterns(self):
        data_uniform = [{"a": 1, "b": 2}] * 5
        data_diverse = [
            {"a": 1, "b": 2},
            {"a": 1},
            {"b": 2},
            {"c": 3},
            {"a": 1, "b": 2, "c": 3},
        ]
        result_uniform = self.analyzer._calculate_consistency(data_uniform, ["x"] * 5)
        result_diverse = self.analyzer._calculate_consistency(data_diverse, ["x"] * 5)
        self.assertGreater(result_uniform.structure_score, result_diverse.structure_score)


# ==================== QualityAnalyzer: _calculate_complexity ====================


class TestCalculateComplexity(unittest.TestCase):
    """Test _calculate_complexity method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_empty_texts(self):
        result = self.analyzer._calculate_complexity([])
        self.assertEqual(result.avg_length, 0)
        self.assertEqual(result.avg_tokens, 0)
        self.assertEqual(result.vocabulary_richness, 0)
        self.assertEqual(result.avg_sentence_length, 0)
        self.assertEqual(result.readability_score, 0)

    def test_basic_text(self):
        texts = ["The quick brown fox jumps over the lazy dog."]
        result = self.analyzer._calculate_complexity(texts)
        self.assertGreater(result.avg_length, 0)
        self.assertGreater(result.avg_tokens, 0)
        self.assertGreater(result.vocabulary_richness, 0)
        self.assertGreater(result.avg_sentence_length, 0)

    def test_multiple_sentences(self):
        texts = ["First sentence. Second sentence. Third sentence."]
        result = self.analyzer._calculate_complexity(texts)
        self.assertGreater(result.avg_sentence_length, 0)

    def test_readability_in_range(self):
        texts = [
            "This is a simple sentence. It is easy to read. The words are short."
        ]
        result = self.analyzer._calculate_complexity(texts)
        self.assertGreaterEqual(result.readability_score, 0)
        self.assertLessEqual(result.readability_score, 100)

    def test_vocabulary_richness_all_unique(self):
        texts = ["alpha beta gamma delta epsilon"]
        result = self.analyzer._calculate_complexity(texts)
        self.assertEqual(result.vocabulary_richness, 1.0)

    def test_vocabulary_richness_all_same(self):
        texts = ["the the the the the"]
        result = self.analyzer._calculate_complexity(texts)
        self.assertAlmostEqual(result.vocabulary_richness, 0.2)

    def test_no_tokens_readability_fallback(self):
        """Text with no word tokens should get readability of 50."""
        texts = ["!!! ???"]
        result = self.analyzer._calculate_complexity(texts)
        self.assertEqual(result.readability_score, 50)


# ==================== QualityAnalyzer: _detect_ai_content ====================


class TestDetectAIContent(unittest.TestCase):
    """Test _detect_ai_content method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_normal_text_low_probability(self):
        texts = [
            "The cat sat on the mat. It was a sunny day.",
            "Python is a programming language.",
        ]
        result = self.analyzer._detect_ai_content(texts)
        self.assertIsInstance(result, AIDetectionMetrics)
        self.assertLessEqual(result.ai_probability, 0.3)

    def test_ai_pattern_detected(self):
        texts = [
            "As an AI language model, I don't have personal opinions. "
            "It's important to note that I cannot provide medical advice. "
            "Furthermore, moreover, nevertheless, these concepts are complex."
        ]
        result = self.analyzer._detect_ai_content(texts)
        self.assertGreater(result.ai_probability, 0)
        self.assertGreater(len(result.indicators), 0)

    def test_self_reference_as_ai(self):
        texts = ["As an AI, I can help you with this task."]
        result = self.analyzer._detect_ai_content(texts)
        self.assertIn("Self-reference as AI", result.indicators)

    def test_hedging_phrases(self):
        texts = ["It's important to note that this is significant. However, it's worth mentioning."]
        result = self.analyzer._detect_ai_content(texts)
        indicators = result.indicators
        self.assertTrue(
            any("Hedging" in ind for ind in indicators),
            f"Expected hedging indicator, got {indicators}",
        )

    def test_formal_transitions(self):
        texts = [
            "Furthermore, this is relevant. Moreover, we should note. Nevertheless, it remains true."
        ]
        result = self.analyzer._detect_ai_content(texts)
        self.assertIn("Formal transition", result.indicators)

    def test_uniform_sentence_length(self):
        # Create text with very uniform sentence lengths
        texts = [
            "This is five words. That is five words. Here is five words. "
            "Same is five words. Last is five words."
        ]
        result = self.analyzer._detect_ai_content(texts)
        # Should detect uniform sentence length
        self.assertIn("Uniform sentence length", result.indicators)

    def test_excessive_politeness(self):
        texts = [
            "Thank you for asking. Please feel free to reach out. "
            "I hope this helps with your question."
        ]
        result = self.analyzer._detect_ai_content(texts)
        self.assertIn("Excessive politeness markers", result.indicators)

    def test_confidence_threshold(self):
        # Low AI probability -> low confidence
        texts = ["Just a normal text without any AI patterns."]
        result = self.analyzer._detect_ai_content(texts)
        self.assertEqual(result.confidence, 0.4)

    def test_high_ai_probability_higher_confidence(self):
        texts = [
            "As an AI, I don't have personal experience. "
            "I cannot provide medical advice. "
            "It's important to note these limitations. "
            "Furthermore, moreover, please feel free to ask. Thank you."
        ]
        result = self.analyzer._detect_ai_content(texts)
        if result.ai_probability > 0.2:
            self.assertEqual(result.confidence, 0.6)

    def test_empty_after_sampling(self):
        """Even with empty texts, should not crash."""
        texts = [""]
        result = self.analyzer._detect_ai_content(texts)
        self.assertIsInstance(result, AIDetectionMetrics)

    def test_top_5_indicators_only(self):
        """Should only return top 5 indicators."""
        # Create a text with many different indicators
        texts = [
            "As an AI, I don't have personal opinions. "
            "I cannot provide advice. It's important to note this. "
            "Furthermore, moreover, nevertheless. "
            "Please feel free to ask. Thank you very much. I hope this helps."
        ] * 20
        result = self.analyzer._detect_ai_content(texts)
        self.assertLessEqual(len(result.indicators), 5)


# ==================== QualityAnalyzer: _calculate_overall_score ====================


class TestCalculateOverallScore(unittest.TestCase):
    """Test _calculate_overall_score method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def _make_metrics(
        self,
        unique_token_ratio=0.2,
        semantic_diversity=0.7,
        ngram_diversity=None,
        format_consistency=0.9,
        structure_score=0.8,
        field_completeness=0.95,
        avg_tokens=100.0,
        vocabulary_richness=0.3,
        readability_score=60.0,
    ):
        if ngram_diversity is None:
            ngram_diversity = {"2-gram": 0.7, "3-gram": 0.8}
        diversity = DiversityMetrics(unique_token_ratio, 1000, semantic_diversity, ngram_diversity)
        consistency = ConsistencyMetrics(format_consistency, structure_score, field_completeness, 0.3)
        complexity = ComplexityMetrics(200.0, avg_tokens, vocabulary_richness, 15.0, readability_score)
        return diversity, consistency, complexity

    def test_score_in_range(self):
        d, cons, comp = self._make_metrics()
        score = self.analyzer._calculate_overall_score(d, cons, comp, None)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_no_ai_detection_gets_full_bonus(self):
        d, cons, comp = self._make_metrics()
        score_no_ai = self.analyzer._calculate_overall_score(d, cons, comp, None)
        ai = AIDetectionMetrics(0.5, 0.6, [])
        score_with_ai = self.analyzer._calculate_overall_score(d, cons, comp, ai)
        self.assertGreater(score_no_ai, score_with_ai)

    def test_high_ai_probability_lowers_score(self):
        d, cons, comp = self._make_metrics()
        ai_low = AIDetectionMetrics(0.1, 0.4, [])
        ai_high = AIDetectionMetrics(0.9, 0.6, [])
        score_low = self.analyzer._calculate_overall_score(d, cons, comp, ai_low)
        score_high = self.analyzer._calculate_overall_score(d, cons, comp, ai_high)
        self.assertGreater(score_low, score_high)

    def test_moderate_tokens_get_bonus(self):
        d, cons, comp_mod = self._make_metrics(avg_tokens=100)
        d2, cons2, comp_low = self._make_metrics(avg_tokens=10)
        score_mod = self.analyzer._calculate_overall_score(d, cons, comp_mod, None)
        score_low = self.analyzer._calculate_overall_score(d2, cons2, comp_low, None)
        self.assertGreater(score_mod, score_low)

    def test_zero_tokens_get_some_score(self):
        d, cons, comp = self._make_metrics(avg_tokens=0)
        score = self.analyzer._calculate_overall_score(d, cons, comp, None)
        # avg_tokens = 0 -> complexity_score gets 0 for token part
        self.assertGreaterEqual(score, 0)

    def test_score_clamped_to_0_100(self):
        # Create metrics that would push score beyond limits
        d, cons, comp = self._make_metrics(
            unique_token_ratio=1.0,
            semantic_diversity=1.0,
            format_consistency=1.0,
            structure_score=1.0,
            field_completeness=1.0,
            avg_tokens=200,
            vocabulary_richness=1.0,
            readability_score=60.0,
        )
        score = self.analyzer._calculate_overall_score(d, cons, comp, None)
        self.assertLessEqual(score, 100)
        self.assertGreaterEqual(score, 0)

    def test_empty_ngram_diversity(self):
        d, cons, comp = self._make_metrics(ngram_diversity={})
        score = self.analyzer._calculate_overall_score(d, cons, comp, None)
        self.assertGreaterEqual(score, 0)


# ==================== QualityAnalyzer: _generate_recommendations ====================


class TestGenerateRecommendations(unittest.TestCase):
    """Test _generate_recommendations method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def _make_metrics(self, **overrides):
        defaults = {
            "unique_token_ratio": 0.2,
            "semantic_diversity": 0.7,
            "format_consistency": 0.9,
            "length_variance": 0.5,
            "avg_tokens": 100,
            "readability_score": 60,
        }
        defaults.update(overrides)
        diversity = DiversityMetrics(defaults["unique_token_ratio"], 1000, defaults["semantic_diversity"])
        consistency = ConsistencyMetrics(defaults["format_consistency"], 0.8, 0.95, defaults["length_variance"])
        complexity = ComplexityMetrics(200.0, defaults["avg_tokens"], 0.3, 15.0, defaults["readability_score"])
        return diversity, consistency, complexity

    def test_good_metrics_no_issues(self):
        d, cons, comp = self._make_metrics()
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertEqual(recs, ["Dataset quality looks good!"])

    def test_low_vocabulary_diversity(self):
        d, cons, comp = self._make_metrics(unique_token_ratio=0.05)
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertTrue(any("vocabulary diversity" in r for r in recs))

    def test_low_semantic_diversity(self):
        d, cons, comp = self._make_metrics(semantic_diversity=0.1)
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertTrue(any("semantic diversity" in r.lower() for r in recs))

    def test_inconsistent_format(self):
        d, cons, comp = self._make_metrics(format_consistency=0.5)
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertTrue(any("format" in r.lower() for r in recs))

    def test_high_length_variance(self):
        d, cons, comp = self._make_metrics(length_variance=3.0)
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertTrue(any("length variance" in r.lower() for r in recs))

    def test_very_short_texts(self):
        d, cons, comp = self._make_metrics(avg_tokens=10)
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertTrue(any("short" in r.lower() for r in recs))

    def test_very_long_texts(self):
        d, cons, comp = self._make_metrics(avg_tokens=1500)
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertTrue(any("long" in r.lower() for r in recs))

    def test_low_readability(self):
        d, cons, comp = self._make_metrics(readability_score=20)
        recs = self.analyzer._generate_recommendations(d, cons, comp, None)
        self.assertTrue(any("readability" in r.lower() for r in recs))

    def test_high_ai_probability(self):
        d, cons, comp = self._make_metrics()
        ai = AIDetectionMetrics(0.7, 0.6, ["Self-reference as AI", "Hedging phrase"])
        recs = self.analyzer._generate_recommendations(d, cons, comp, ai)
        self.assertTrue(any("ai" in r.lower() for r in recs))
        self.assertTrue(any("indicator" in r.lower() for r in recs))

    def test_high_ai_without_indicators(self):
        d, cons, comp = self._make_metrics()
        ai = AIDetectionMetrics(0.7, 0.6, [])
        recs = self.analyzer._generate_recommendations(d, cons, comp, ai)
        ai_recs = [r for r in recs if "ai" in r.lower()]
        self.assertGreater(len(ai_recs), 0)
        # No indicator recommendation since indicators list is empty
        indicator_recs = [r for r in recs if "indicator" in r.lower()]
        self.assertEqual(len(indicator_recs), 0)

    def test_low_ai_probability_no_recommendation(self):
        d, cons, comp = self._make_metrics()
        ai = AIDetectionMetrics(0.3, 0.4, [])
        recs = self.analyzer._generate_recommendations(d, cons, comp, ai)
        ai_recs = [r for r in recs if "ai-generated" in r.lower()]
        self.assertEqual(len(ai_recs), 0)


# ==================== QualityAnalyzer: _generate_warnings ====================


class TestGenerateWarnings(unittest.TestCase):
    """Test _generate_warnings method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_no_warnings_for_good_data(self):
        data = [{"text": f"Text {i}"} for i in range(200)]
        texts = [f"Text {i} with enough content" for i in range(200)]
        warnings = self.analyzer._generate_warnings(data, texts, "text")
        self.assertEqual(warnings, [])

    def test_missing_texts_warning(self):
        data = [{"text": "a"}, {"other": "b"}, {"text": "c"}]
        texts = ["a", "c"]
        warnings = self.analyzer._generate_warnings(data, texts, "text")
        self.assertTrue(any("1 examples missing" in w for w in warnings))

    def test_small_sample_warning(self):
        data = [{"text": f"Text {i}" * 5} for i in range(50)]
        texts = [f"Text {i}" * 5 for i in range(50)]
        warnings = self.analyzer._generate_warnings(data, texts, "text")
        self.assertTrue(any("Small sample" in w for w in warnings))

    def test_short_texts_warning(self):
        data = [{"text": "x"} for _ in range(200)]
        texts = ["x"] * 200  # All very short
        warnings = self.analyzer._generate_warnings(data, texts, "text")
        self.assertTrue(any("very short" in w for w in warnings))

    def test_short_texts_threshold(self):
        """Only warn if > 10% of texts are short."""
        long_texts = ["This is a sufficiently long text for testing"] * 95
        short_texts = ["hi"] * 5
        all_texts = long_texts + short_texts
        data = [{"text": t} for t in all_texts]
        warnings = self.analyzer._generate_warnings(data, all_texts, "text")
        # 5/100 = 5% < 10% -> no warning about short texts
        short_warnings = [w for w in warnings if "very short" in w]
        self.assertEqual(len(short_warnings), 0)

    def test_exact_match_no_missing(self):
        data = [{"text": "a"}]
        texts = ["a"]
        warnings = self.analyzer._generate_warnings(data, texts, "text")
        missing_warnings = [w for w in warnings if "missing" in w]
        self.assertEqual(len(missing_warnings), 0)


# ==================== QualityAnalyzer: analyze_sample (integration) ====================


class TestAnalyzeSample(unittest.TestCase):
    """Test analyze_sample method end-to-end."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def test_empty_data(self):
        report = self.analyzer.analyze_sample([])
        self.assertEqual(report.overall_score, 0)
        self.assertEqual(report.sample_size, 0)
        self.assertIn("No data to analyze", report.recommendations)

    def test_no_texts_found(self):
        data = [{"id": 1}, {"id": 2}]
        report = self.analyzer.analyze_sample(data, text_field="text")
        self.assertEqual(report.overall_score, 0)
        self.assertEqual(report.sample_size, 2)
        self.assertTrue(any("No text found" in r for r in report.recommendations))
        self.assertTrue(any("Could not extract" in w for w in report.warnings))

    def test_basic_analysis(self):
        data = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "A fast red car drives down the highway."},
            {"text": "The beautiful garden has many colorful flowers."},
        ]
        report = self.analyzer.analyze_sample(data, text_field="text")
        self.assertIsInstance(report, QualityReport)
        self.assertEqual(report.sample_size, 3)
        self.assertGreater(report.overall_score, 0)
        self.assertIsNotNone(report.diversity)
        self.assertIsNotNone(report.consistency)
        self.assertIsNotNone(report.complexity)
        self.assertIsNone(report.ai_detection)

    def test_with_ai_detection(self):
        data = [
            {"text": "Normal text about programming."},
            {"text": "As an AI, I cannot provide medical advice. It's important to note."},
        ]
        report = self.analyzer.analyze_sample(data, text_field="text", detect_ai=True)
        self.assertIsNotNone(report.ai_detection)
        self.assertIsInstance(report.ai_detection, AIDetectionMetrics)

    def test_custom_text_field(self):
        data = [
            {"content": "Hello world this is a test."},
            {"content": "Another piece of content here."},
        ]
        report = self.analyzer.analyze_sample(data, text_field="content")
        self.assertEqual(report.sample_size, 2)
        self.assertGreater(report.overall_score, 0)

    def test_nested_text_field(self):
        data = [
            {"meta": {"text": "Nested text content here."}},
            {"meta": {"text": "Another nested text value."}},
        ]
        report = self.analyzer.analyze_sample(data, text_field="meta.text")
        self.assertEqual(report.sample_size, 2)
        self.assertGreater(report.overall_score, 0)

    def test_partial_data_with_warnings(self):
        data = [
            {"text": "Has text content here."},
            {"no_text": "Missing text field."},
            {"text": "Another text entry."},
        ]
        report = self.analyzer.analyze_sample(data, text_field="text")
        self.assertEqual(report.sample_size, 3)
        # Should warn about missing texts
        self.assertTrue(any("missing" in w for w in report.warnings))

    def test_conversation_data(self):
        data = [
            {
                "messages": [
                    {"content": "Hello, how are you?"},
                    {"content": "I am fine, thank you!"},
                ]
            },
            {
                "messages": [
                    {"content": "What is the weather like?"},
                    {"content": "It is sunny today."},
                ]
            },
        ]
        report = self.analyzer.analyze_sample(data, text_field="messages")
        self.assertEqual(report.sample_size, 2)
        self.assertGreater(report.overall_score, 0)

    def test_report_has_recommendations(self):
        data = [{"text": f"Sample text number {i}" * 10} for i in range(5)]
        report = self.analyzer.analyze_sample(data)
        self.assertIsInstance(report.recommendations, list)
        self.assertGreater(len(report.recommendations), 0)


# ==================== QualityAnalyzer: analyze_from_huggingface ====================


class TestAnalyzeFromHuggingFace(unittest.TestCase):
    """Test analyze_from_huggingface method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    @patch("datarecipe.quality_metrics.QualityAnalyzer.analyze_sample")
    def test_import_error_returns_empty_report(self, mock_analyze):
        """When datasets package is not installed, should return error report."""
        with patch.dict("sys.modules", {"datasets": None}):
            # Force ImportError by patching the import
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "datasets":
                    raise ImportError("No module named 'datasets'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                report = self.analyzer.analyze_from_huggingface("test/dataset")
                self.assertEqual(report.overall_score, 0)
                self.assertTrue(
                    any("datasets" in r for r in report.recommendations)
                )
                self.assertTrue(
                    any("datasets" in w for w in report.warnings)
                )

    def test_exception_returns_error_report(self):
        """When dataset loading fails, should return error report."""
        # Create a mock module with a load_dataset that raises
        mock_datasets_mod = MagicMock()
        mock_datasets_mod.load_dataset.side_effect = Exception("Connection error")

        with patch.dict("sys.modules", {"datasets": mock_datasets_mod}):
            report = self.analyzer.analyze_from_huggingface("nonexistent/dataset")
            self.assertEqual(report.overall_score, 0)
            self.assertTrue(
                any("Failed to load" in r or "Connection error" in r for r in report.recommendations)
            )
            self.assertTrue(
                any("Connection error" in w for w in report.warnings)
            )


# ==================== QualityAnalyzer: _extract_metric ====================


class TestExtractMetric(unittest.TestCase):
    """Test _extract_metric method."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def _make_report(self):
        return QualityReport(
            diversity=DiversityMetrics(0.15, 1000, 0.6),
            consistency=ConsistencyMetrics(0.9, 0.8, 0.95, 0.3),
            complexity=ComplexityMetrics(200.0, 50.0, 0.3, 15.0, 60.0),
            overall_score=75.0,
            sample_size=100,
        )

    def test_extract_top_level_metric(self):
        report = self._make_report()
        val = self.analyzer._extract_metric(report, "overall_score")
        self.assertEqual(val, 75.0)

    def test_extract_nested_metric(self):
        report = self._make_report()
        val = self.analyzer._extract_metric(report, "diversity.unique_token_ratio")
        self.assertAlmostEqual(val, 0.15)

    def test_extract_nonexistent_metric(self):
        report = self._make_report()
        val = self.analyzer._extract_metric(report, "nonexistent")
        self.assertIsNone(val)

    def test_extract_none_sub_object(self):
        report = self._make_report()
        report.ai_detection = None
        val = self.analyzer._extract_metric(report, "ai_detection.ai_probability")
        self.assertIsNone(val)

    def test_extract_from_ai_detection(self):
        report = self._make_report()
        report.ai_detection = AIDetectionMetrics(0.3, 0.6, [])
        val = self.analyzer._extract_metric(report, "ai_detection.ai_probability")
        self.assertAlmostEqual(val, 0.3)

    def test_non_numeric_returns_none(self):
        report = self._make_report()
        val = self.analyzer._extract_metric(report, "recommendations")
        self.assertIsNone(val)

    def test_extract_int_metric(self):
        report = self._make_report()
        val = self.analyzer._extract_metric(report, "sample_size")
        self.assertEqual(val, 100.0)
        self.assertIsInstance(val, float)

    def test_extract_deeply_nested_nonexistent(self):
        report = self._make_report()
        val = self.analyzer._extract_metric(report, "diversity.nonexistent.deep")
        self.assertIsNone(val)


# ==================== QualityAnalyzer: _compare ====================


class TestCompare(unittest.TestCase):
    """Test _compare static method."""

    def test_greater_equal(self):
        self.assertTrue(QualityAnalyzer._compare(5.0, ">=", 5.0))
        self.assertTrue(QualityAnalyzer._compare(6.0, ">=", 5.0))
        self.assertFalse(QualityAnalyzer._compare(4.0, ">=", 5.0))

    def test_less_equal(self):
        self.assertTrue(QualityAnalyzer._compare(5.0, "<=", 5.0))
        self.assertTrue(QualityAnalyzer._compare(4.0, "<=", 5.0))
        self.assertFalse(QualityAnalyzer._compare(6.0, "<=", 5.0))

    def test_greater(self):
        self.assertTrue(QualityAnalyzer._compare(6.0, ">", 5.0))
        self.assertFalse(QualityAnalyzer._compare(5.0, ">", 5.0))

    def test_less(self):
        self.assertTrue(QualityAnalyzer._compare(4.0, "<", 5.0))
        self.assertFalse(QualityAnalyzer._compare(5.0, "<", 5.0))

    def test_equal(self):
        self.assertTrue(QualityAnalyzer._compare(5.0, "==", 5.0))
        self.assertFalse(QualityAnalyzer._compare(4.0, "==", 5.0))

    def test_not_equal(self):
        self.assertTrue(QualityAnalyzer._compare(4.0, "!=", 5.0))
        self.assertFalse(QualityAnalyzer._compare(5.0, "!=", 5.0))

    def test_unknown_operator_returns_false(self):
        self.assertFalse(QualityAnalyzer._compare(5.0, "??", 5.0))


# ==================== QualityAnalyzer: evaluate_gates ====================


class TestEvaluateGatesExtended(unittest.TestCase):
    """Extended tests for evaluate_gates beyond test_quality_gates.py."""

    def setUp(self):
        self.analyzer = QualityAnalyzer()

    def _make_report(self, **kwargs):
        defaults = {
            "overall_score": 75.0,
            "unique_token_ratio": 0.15,
            "format_consistency": 0.9,
            "field_completeness": 0.95,
        }
        defaults.update(kwargs)
        return QualityReport(
            diversity=DiversityMetrics(defaults["unique_token_ratio"], 1000, 0.6),
            consistency=ConsistencyMetrics(defaults["format_consistency"], 0.8, defaults["field_completeness"], 0.3),
            complexity=ComplexityMetrics(200.0, 50.0, 0.3, 15.0, 60.0),
            overall_score=defaults["overall_score"],
            sample_size=100,
        )

    def test_multiple_blockers(self):
        report = self._make_report(overall_score=30, unique_token_ratio=0.01, format_consistency=0.1)
        gate_report = self.analyzer.evaluate_gates(report)
        self.assertFalse(gate_report.passed)
        self.assertGreater(len(gate_report.blocking_failures), 1)

    def test_gate_message_format(self):
        gate = QualityGateRule("test", "Test Gate", "overall_score", ">=", 60, "blocker")
        report = self._make_report(overall_score=50)
        gate_report = self.analyzer.evaluate_gates(report, gates=[gate])
        result = gate_report.results[0]
        self.assertIn("FAIL", result.message)
        self.assertIn("Test Gate", result.message)

    def test_pass_message(self):
        gate = QualityGateRule("test", "Test Gate", "overall_score", ">=", 60, "blocker")
        report = self._make_report(overall_score=80)
        gate_report = self.analyzer.evaluate_gates(report, gates=[gate])
        result = gate_report.results[0]
        self.assertIn("PASS", result.message)

    def test_empty_gates_list(self):
        report = self._make_report()
        gate_report = self.analyzer.evaluate_gates(report, gates=[])
        self.assertTrue(gate_report.passed)
        self.assertEqual(len(gate_report.results), 0)

    def test_gate_report_to_dict(self):
        report = self._make_report()
        gate_report = self.analyzer.evaluate_gates(report)
        d = gate_report.to_dict()
        self.assertIn("passed", d)
        self.assertIn("results", d)
        self.assertIn("blocking_failures", d)
        self.assertIn("warnings", d)

    def test_warning_severity_not_blocking(self):
        """Warning severity failures should not cause overall gate failure."""
        gate = QualityGateRule("warn", "Warning Gate", "overall_score", ">=", 90, "warning")
        report = self._make_report(overall_score=80)
        gate_report = self.analyzer.evaluate_gates(report, gates=[gate])
        self.assertTrue(gate_report.passed)
        self.assertEqual(len(gate_report.warnings), 1)
        self.assertEqual(len(gate_report.blocking_failures), 0)


# ==================== QualityAnalyzer: constructor ====================


class TestQualityAnalyzerInit(unittest.TestCase):
    """Test QualityAnalyzer initialization."""

    def test_default_no_embeddings(self):
        analyzer = QualityAnalyzer()
        self.assertFalse(analyzer.use_embeddings)
        self.assertIsNone(analyzer._embedder)

    def test_with_embeddings_flag(self):
        analyzer = QualityAnalyzer(use_embeddings=True)
        self.assertTrue(analyzer.use_embeddings)
        self.assertIsNone(analyzer._embedder)


# ==================== GateResult.to_dict ====================


class TestGateResultToDict(unittest.TestCase):
    """Test GateResult.to_dict() serialization."""

    def test_to_dict_pass(self):
        gate = QualityGateRule("g1", "Test", "overall_score", ">=", 60, "blocker")
        gr = GateResult(gate=gate, actual_value=75.1234, passed=True, message="ok")
        d = gr.to_dict()
        self.assertEqual(d["gate_id"], "g1")
        self.assertEqual(d["name"], "Test")
        self.assertEqual(d["metric"], "overall_score")
        self.assertEqual(d["threshold"], 60)
        self.assertEqual(d["operator"], ">=")
        self.assertEqual(d["actual_value"], 75.1234)
        self.assertTrue(d["passed"])
        self.assertEqual(d["severity"], "blocker")
        self.assertEqual(d["message"], "ok")

    def test_to_dict_fail(self):
        gate = QualityGateRule("g2", "Low Score", "overall_score", ">=", 80, "warning")
        gr = GateResult(gate=gate, actual_value=50.0, passed=False, message="fail")
        d = gr.to_dict()
        self.assertFalse(d["passed"])
        self.assertEqual(d["severity"], "warning")


# ==================== Integration: Full pipeline ====================


class TestFullPipeline(unittest.TestCase):
    """Integration test for the full analysis and gate evaluation pipeline."""

    def test_analyze_and_evaluate_gates(self):
        analyzer = QualityAnalyzer()
        data = [
            {"text": "The quick brown fox jumps over the lazy dog. This is a standard test sentence."},
            {"text": "Machine learning is a subset of artificial intelligence that learns from data."},
            {"text": "Python programming language is widely used for data science and web development."},
            {"text": "Natural language processing enables computers to understand human language."},
            {"text": "Deep learning models have achieved remarkable results in computer vision tasks."},
        ]
        report = analyzer.analyze_sample(data, text_field="text", detect_ai=False)
        gate_report = analyzer.evaluate_gates(report)

        self.assertIsInstance(gate_report, QualityGateReport)
        self.assertIsNotNone(report.gate_report)
        self.assertEqual(report.gate_report, gate_report)

        # Verify full serialization works
        report_dict = report.to_dict()
        self.assertIn("gate_report", report_dict)
        self.assertIn("diversity", report_dict)
        self.assertIn("consistency", report_dict)
        self.assertIn("complexity", report_dict)

    def test_analyze_with_ai_detection_and_gates(self):
        analyzer = QualityAnalyzer()
        data = [
            {"text": "As an AI, I don't have personal opinions. It's important to note."},
            {"text": "Furthermore, moreover, I cannot provide medical advice."},
            {"text": "Normal text about everyday life and activities."},
        ]
        report = analyzer.analyze_sample(data, text_field="text", detect_ai=True)
        self.assertIsNotNone(report.ai_detection)

        gate_report = analyzer.evaluate_gates(report)
        self.assertIsInstance(gate_report, QualityGateReport)

        # AI gate should be evaluated since ai_detection is present
        ai_gates = [r for r in gate_report.results if "ai" in r.gate.metric]
        self.assertGreater(len(ai_gates), 0)


if __name__ == "__main__":
    unittest.main()
