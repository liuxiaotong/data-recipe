"""Unit tests for context_strategy.py - context construction strategy detection."""

import unittest

from datarecipe.analyzers.context_strategy import (
    ContextStrategy,
    ContextStrategyDetector,
    ContextStrategyType,
)


class TestContextStrategyType(unittest.TestCase):
    """Tests for the ContextStrategyType enum."""

    def test_enum_values(self):
        self.assertEqual(ContextStrategyType.SYNTHETIC.value, "synthetic")
        self.assertEqual(ContextStrategyType.MODIFIED.value, "modified")
        self.assertEqual(ContextStrategyType.NICHE.value, "niche")
        self.assertEqual(ContextStrategyType.HYBRID.value, "hybrid")
        self.assertEqual(ContextStrategyType.UNKNOWN.value, "unknown")

    def test_enum_members(self):
        self.assertEqual(len(ContextStrategyType), 5)


class TestContextStrategy(unittest.TestCase):
    """Tests for the ContextStrategy dataclass."""

    def test_defaults(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN)
        self.assertEqual(cs.confidence, 0.0)
        self.assertEqual(cs.synthetic_score, 0.0)
        self.assertEqual(cs.modified_score, 0.0)
        self.assertEqual(cs.niche_score, 0.0)
        self.assertEqual(cs.synthetic_indicators, [])
        self.assertEqual(cs.modified_indicators, [])
        self.assertEqual(cs.niche_indicators, [])
        self.assertEqual(cs.data_sources, [])
        self.assertEqual(cs.modification_types, [])
        self.assertEqual(cs.domain_specificity, 0.0)
        self.assertEqual(cs.recommendations, [])

    def test_summary_basic(self):
        cs = ContextStrategy(
            primary_strategy=ContextStrategyType.SYNTHETIC,
            confidence=0.8,
            synthetic_score=0.7,
            modified_score=0.2,
            niche_score=0.1,
        )
        summary = cs.summary()
        self.assertIn("Primary Strategy: synthetic", summary)
        self.assertIn("Confidence: 80.0%", summary)
        self.assertIn("Synthetic: 70.0%", summary)
        self.assertIn("Modified: 20.0%", summary)
        self.assertIn("Niche: 10.0%", summary)

    def test_summary_with_indicators(self):
        cs = ContextStrategy(
            primary_strategy=ContextStrategyType.SYNTHETIC,
            synthetic_indicators=["fictional_markers: fictional"],
            modified_indicators=["source_references: wikipedia"],
            niche_indicators=["domain_specific: proprietary"],
        )
        summary = cs.summary()
        self.assertIn("Synthetic Indicators:", summary)
        self.assertIn("fictional_markers: fictional", summary)
        self.assertIn("Modification Indicators:", summary)
        self.assertIn("wikipedia", summary)
        self.assertIn("Niche/Specialized Indicators:", summary)
        self.assertIn("proprietary", summary)

    def test_summary_with_recommendations(self):
        cs = ContextStrategy(
            primary_strategy=ContextStrategyType.UNKNOWN,
            recommendations=["Do X", "Do Y"],
        )
        summary = cs.summary()
        self.assertIn("Recommendations:", summary)
        self.assertIn("Do X", summary)
        self.assertIn("Do Y", summary)

    def test_summary_indicators_capped_at_5(self):
        cs = ContextStrategy(
            primary_strategy=ContextStrategyType.SYNTHETIC,
            synthetic_indicators=[f"ind_{i}" for i in range(10)],
        )
        summary = cs.summary()
        # Only first 5 should appear
        self.assertIn("ind_0", summary)
        self.assertIn("ind_4", summary)
        self.assertNotIn("ind_5", summary)

    def test_summary_no_indicators(self):
        cs = ContextStrategy(
            primary_strategy=ContextStrategyType.UNKNOWN,
        )
        summary = cs.summary()
        self.assertNotIn("Synthetic Indicators:", summary)
        self.assertNotIn("Modification Indicators:", summary)
        self.assertNotIn("Niche/Specialized Indicators:", summary)
        self.assertNotIn("Recommendations:", summary)


# ===========================================================================
# ContextStrategyDetector tests
# ===========================================================================

class TestContextStrategyDetectorInit(unittest.TestCase):
    """Tests for detector initialization."""

    def test_patterns_compiled(self):
        detector = ContextStrategyDetector()
        self.assertIsInstance(detector.synthetic_patterns, dict)
        self.assertIsInstance(detector.modified_patterns, dict)
        self.assertIsInstance(detector.niche_patterns, dict)

    def test_pattern_categories_match_class_indicators(self):
        detector = ContextStrategyDetector()
        self.assertEqual(
            set(detector.synthetic_patterns.keys()),
            set(ContextStrategyDetector.SYNTHETIC_INDICATORS.keys()),
        )
        self.assertEqual(
            set(detector.modified_patterns.keys()),
            set(ContextStrategyDetector.MODIFIED_INDICATORS.keys()),
        )
        self.assertEqual(
            set(detector.niche_patterns.keys()),
            set(ContextStrategyDetector.NICHE_INDICATORS.keys()),
        )


class TestContextStrategyDetectorAnalyze(unittest.TestCase):
    """Tests for the main analyze() method."""

    def setUp(self):
        self.detector = ContextStrategyDetector()

    # -- empty / trivial input ----------------------------------------------

    def test_empty_contexts(self):
        result = self.detector.analyze([])
        self.assertEqual(result.primary_strategy, ContextStrategyType.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)

    def test_single_empty_string(self):
        result = self.detector.analyze([""])
        self.assertEqual(result.primary_strategy, ContextStrategyType.UNKNOWN)

    # -- synthetic detection ------------------------------------------------

    def test_detects_synthetic_fictional(self):
        contexts = [
            "This is a fictional scenario about a magical kingdom.",
            "In this hypothetical situation, assume that gravity is reversed.",
            "The imaginary company Acme Corp was invented for this example.",
        ]
        result = self.detector.analyze(contexts)
        self.assertIn(
            result.primary_strategy,
            [ContextStrategyType.SYNTHETIC, ContextStrategyType.HYBRID],
        )
        self.assertTrue(result.synthetic_score > 0)
        self.assertTrue(len(result.synthetic_indicators) > 0)

    def test_detects_synthetic_generation_markers(self):
        contexts = [
            "This text was generated by GPT-4.",
            "Content synthesized using Claude model.",
            "Model-generated response for training data.",
        ]
        result = self.detector.analyze(contexts)
        self.assertTrue(result.synthetic_score > 0)

    def test_detects_synthetic_placeholders(self):
        contexts = [
            "Contact John Doe at john@example.com for details.",
            "Visit [company website] for more information.",
            "The xxxxx field should be filled in.",
        ]
        result = self.detector.analyze(contexts)
        self.assertTrue(result.synthetic_score > 0)

    # -- modified detection -------------------------------------------------

    def test_detects_modified(self):
        contexts = [
            "This passage was paraphrased from the original source.",
            "Adapted from Wikipedia article on machine learning.",
            "The text was augmented using back-translation techniques.",
        ]
        result = self.detector.analyze(contexts)
        self.assertTrue(result.modified_score > 0)
        self.assertTrue(len(result.modified_indicators) > 0)

    def test_detects_modification_types(self):
        contexts = [
            "Applied synonym replacement to create variations.",
            "Sentence shuffle was used for augmentation.",
        ]
        result = self.detector.analyze(contexts)
        self.assertTrue(result.modified_score > 0)

    # -- niche detection ----------------------------------------------------

    def test_detects_niche(self):
        contexts = [
            "This proprietary dataset contains specialized domain expertise.",
            "Data manually gathered from clinical trial records.",
            "Annotated by experts in legal document analysis.",
        ]
        result = self.detector.analyze(contexts)
        self.assertTrue(result.niche_score > 0)
        self.assertTrue(len(result.niche_indicators) > 0)

    def test_detects_niche_technical(self):
        contexts = [
            "Extracted from technical specification documents.",
            "Based on api documentation from research papers.",
        ]
        result = self.detector.analyze(contexts)
        self.assertTrue(result.niche_score > 0)

    # -- hybrid detection ---------------------------------------------------

    def test_detects_hybrid(self):
        """When both synthetic and modified signals are strong and close, strategy is HYBRID."""
        contexts = [
            "This fictional scenario was adapted from Wikipedia.",
            "A hypothetical case based on existing research papers.",
            "Imaginary company derived from real-world examples.",
            "Generated content augmented with source references.",
        ]
        result = self.detector.analyze(contexts)
        # Should either be HYBRID or the dominant one
        self.assertIn(
            result.primary_strategy,
            [ContextStrategyType.HYBRID, ContextStrategyType.SYNTHETIC, ContextStrategyType.MODIFIED],
        )

    # -- unknown detection --------------------------------------------------

    def test_unknown_when_no_indicators(self):
        contexts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a simple sentence.",
        ]
        result = self.detector.analyze(contexts)
        self.assertEqual(result.primary_strategy, ContextStrategyType.UNKNOWN)

    # -- confidence ---------------------------------------------------------

    def test_unknown_low_confidence(self):
        contexts = ["Plain text with no indicators."]
        result = self.detector.analyze(contexts)
        self.assertEqual(result.primary_strategy, ContextStrategyType.UNKNOWN)
        self.assertAlmostEqual(result.confidence, 0.3)

    def test_strong_signal_higher_confidence(self):
        contexts = [
            "Fictional scenario imaginary invented hypothetical fantasy assume that.",
        ] * 5
        result = self.detector.analyze(contexts)
        self.assertTrue(result.confidence > 0.3)

    # -- score normalization ------------------------------------------------

    def test_scores_normalized(self):
        """Scores should sum to roughly 1.0 when any indicators are found."""
        contexts = [
            "Fictional generated content adapted from source.",
        ]
        result = self.detector.analyze(contexts)
        total = result.synthetic_score + result.modified_score + result.niche_score
        if total > 0:
            self.assertAlmostEqual(total, 1.0, places=5)

    # -- metadata analysis --------------------------------------------------

    def test_metadata_source(self):
        contexts = ["Some text."]
        metadata = {"source": "custom_crawler"}
        result = self.detector.analyze(contexts, metadata)
        self.assertIn("custom_crawler", result.data_sources)

    def test_metadata_generation_method_synthetic(self):
        contexts = ["Some text."]
        metadata = {"generation_method": "synthetic generation via GPT-4"}
        result = self.detector.analyze(contexts, metadata)
        self.assertTrue(result.synthetic_score >= 0.5)
        # Should have a metadata indicator
        has_metadata_indicator = any("metadata:" in ind for ind in result.synthetic_indicators)
        self.assertTrue(has_metadata_indicator)

    def test_metadata_generation_method_generated(self):
        contexts = ["Some text."]
        metadata = {"generation_method": "auto-generated pipeline"}
        result = self.detector.analyze(contexts, metadata)
        self.assertTrue(result.synthetic_score >= 0.5)

    def test_metadata_original_source(self):
        contexts = ["Some text."]
        metadata = {"original_source": "arxiv:2301.12345"}
        result = self.detector.analyze(contexts, metadata)
        self.assertTrue(result.modified_score >= 0.3)
        has_source_indicator = any("original source" in ind for ind in result.modified_indicators)
        self.assertTrue(has_source_indicator)

    def test_metadata_empty(self):
        contexts = ["Some text."]
        result = self.detector.analyze(contexts, metadata={})
        self.assertEqual(result.data_sources, [])

    def test_metadata_none(self):
        contexts = ["Some text."]
        result = self.detector.analyze(contexts, metadata=None)
        self.assertEqual(result.data_sources, [])

    # -- recommendations ----------------------------------------------------

    def test_recommendations_synthetic(self):
        contexts = [
            "Fictional scenario generated by GPT-4 model-generated hypothetical.",
        ] * 5
        result = self.detector.analyze(contexts)
        if result.primary_strategy == ContextStrategyType.SYNTHETIC:
            self.assertTrue(any("LLM" in r for r in result.recommendations))

    def test_recommendations_unknown(self):
        contexts = ["Plain text."]
        result = self.detector.analyze(contexts)
        self.assertEqual(result.primary_strategy, ContextStrategyType.UNKNOWN)
        self.assertTrue(any("Gather more samples" in r for r in result.recommendations))


class TestContextStrategyDetectorRecommendations(unittest.TestCase):
    """Tests for _generate_recommendations directly."""

    def setUp(self):
        self.detector = ContextStrategyDetector()

    def test_synthetic_recommendations(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.SYNTHETIC)
        recs = self.detector._generate_recommendations(cs)
        self.assertTrue(len(recs) > 0)
        self.assertTrue(any("LLM" in r for r in recs))

    def test_modified_recommendations(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.MODIFIED)
        recs = self.detector._generate_recommendations(cs)
        self.assertTrue(len(recs) > 0)
        self.assertTrue(any("paraphrasing" in r.lower() for r in recs))

    def test_niche_recommendations(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.NICHE)
        recs = self.detector._generate_recommendations(cs)
        self.assertTrue(len(recs) > 0)
        self.assertTrue(any("domain" in r.lower() for r in recs))

    def test_hybrid_recommendations(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.HYBRID)
        recs = self.detector._generate_recommendations(cs)
        self.assertTrue(len(recs) > 0)
        self.assertTrue(any("multiple strategies" in r.lower() for r in recs))

    def test_unknown_recommendations(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN)
        recs = self.detector._generate_recommendations(cs)
        self.assertTrue(len(recs) > 0)
        self.assertTrue(any("Gather more samples" in r for r in recs))


class TestContextStrategyDetectorDomainSpecificity(unittest.TestCase):
    """Tests for _calculate_domain_specificity."""

    def setUp(self):
        self.detector = ContextStrategyDetector()

    def test_empty_contexts(self):
        result = self.detector._calculate_domain_specificity([])
        self.assertAlmostEqual(result, 0.0)

    def test_plain_text_low_specificity(self):
        contexts = [
            "The cat sat on the mat.",
            "Dogs are good pets.",
        ]
        result = self.detector._calculate_domain_specificity(contexts)
        # Should be relatively low
        self.assertTrue(result < 0.5)

    def test_technical_text_higher_specificity(self):
        contexts = [
            "The API v2.3.1 endpoint requires authentication via OAuth2. "
            "See RFC 7519 for JWT specification. The function_name parameter "
            "uses snake_case convention. Cf. previous documentation.",
        ]
        result = self.detector._calculate_domain_specificity(contexts)
        self.assertTrue(result > 0)

    def test_many_acronyms_increases_specificity(self):
        contexts = [
            "The NLP API uses ML models via GPU processing with CUDA and TF framework. "
            "CPU usage is monitored by the SRE team following SLA requirements. "
            "The ETL pipeline processes CSV and JSON data via REST endpoints.",
        ]
        result = self.detector._calculate_domain_specificity(contexts)
        self.assertTrue(result > 0)

    def test_caps_to_one(self):
        """Domain specificity should be capped at 1.0."""
        # Generate context with tons of technical terms
        contexts = [
            " ".join(["API", "NLP", "ML", "GPU", "CPU", "SRE", "cf.", "e.g.", "i.e.", "et al."] * 20)
        ]
        result = self.detector._calculate_domain_specificity(contexts)
        self.assertLessEqual(result, 1.0)

    def test_empty_string_context(self):
        contexts = [""]
        result = self.detector._calculate_domain_specificity(contexts)
        self.assertAlmostEqual(result, 0.0)

    def test_samples_first_100(self):
        """Should only process first 100 contexts."""
        contexts = ["simple text"] * 200
        # Should not raise; just processes first 100
        result = self.detector._calculate_domain_specificity(contexts)
        self.assertIsInstance(result, float)


class TestContextStrategyDetectorToDict(unittest.TestCase):
    """Tests for to_dict conversion."""

    def setUp(self):
        self.detector = ContextStrategyDetector()

    def test_to_dict_structure(self):
        cs = ContextStrategy(
            primary_strategy=ContextStrategyType.SYNTHETIC,
            confidence=0.85,
            synthetic_score=0.7,
            modified_score=0.2,
            niche_score=0.1,
            synthetic_indicators=["ind1"],
            modified_indicators=["ind2"],
            niche_indicators=["ind3"],
            data_sources=["src1"],
            domain_specificity=0.5,
            recommendations=["rec1"],
        )
        d = self.detector.to_dict(cs)
        self.assertEqual(d["primary_strategy"], "synthetic")
        self.assertAlmostEqual(d["confidence"], 0.85)
        self.assertEqual(d["scores"]["synthetic"], 0.7)
        self.assertEqual(d["scores"]["modified"], 0.2)
        self.assertEqual(d["scores"]["niche"], 0.1)
        self.assertEqual(d["indicators"]["synthetic"], ["ind1"])
        self.assertEqual(d["indicators"]["modified"], ["ind2"])
        self.assertEqual(d["indicators"]["niche"], ["ind3"])
        self.assertEqual(d["data_sources"], ["src1"])
        self.assertAlmostEqual(d["domain_specificity"], 0.5)
        self.assertEqual(d["recommendations"], ["rec1"])

    def test_to_dict_unknown(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN)
        d = self.detector.to_dict(cs)
        self.assertEqual(d["primary_strategy"], "unknown")

    def test_to_dict_empty_lists(self):
        cs = ContextStrategy(primary_strategy=ContextStrategyType.NICHE)
        d = self.detector.to_dict(cs)
        self.assertEqual(d["indicators"]["synthetic"], [])
        self.assertEqual(d["data_sources"], [])
        self.assertEqual(d["recommendations"], [])


class TestContextStrategyDetectorAnalyzeMetadata(unittest.TestCase):
    """Tests for _analyze_metadata helper."""

    def setUp(self):
        self.detector = ContextStrategyDetector()

    def test_source_added(self):
        result = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN)
        self.detector._analyze_metadata(result, {"source": "web_crawl"})
        self.assertIn("web_crawl", result.data_sources)

    def test_empty_source_not_added(self):
        result = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN)
        self.detector._analyze_metadata(result, {"source": ""})
        self.assertEqual(result.data_sources, [])

    def test_generation_method_with_synthetic(self):
        result = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN, synthetic_score=0.1)
        self.detector._analyze_metadata(result, {"generation_method": "Synthetic pipeline"})
        self.assertTrue(result.synthetic_score >= 0.5)

    def test_generation_method_with_generated(self):
        result = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN, synthetic_score=0.1)
        self.detector._analyze_metadata(result, {"generation_method": "Auto-generated"})
        self.assertTrue(result.synthetic_score >= 0.5)

    def test_original_source_present(self):
        result = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN, modified_score=0.1)
        self.detector._analyze_metadata(result, {"original_source": "dataset_v1"})
        self.assertTrue(result.modified_score >= 0.3)
        has_indicator = any("original source" in ind for ind in result.modified_indicators)
        self.assertTrue(has_indicator)

    def test_no_relevant_keys(self):
        result = ContextStrategy(primary_strategy=ContextStrategyType.UNKNOWN)
        self.detector._analyze_metadata(result, {"irrelevant_key": "value"})
        self.assertEqual(result.data_sources, [])
        self.assertEqual(result.synthetic_indicators, [])
        self.assertEqual(result.modified_indicators, [])


class TestContextStrategyDetectorCompilePatterns(unittest.TestCase):
    """Tests for _compile_patterns."""

    def setUp(self):
        self.detector = ContextStrategyDetector()

    def test_compiles_to_regex(self):
        import re
        patterns = self.detector._compile_patterns({"cat": [r"hello", r"world"]})
        self.assertEqual(len(patterns["cat"]), 2)
        for p in patterns["cat"]:
            self.assertIsInstance(p, re.Pattern)

    def test_case_insensitive(self):
        import re
        patterns = self.detector._compile_patterns({"cat": [r"Test"]})
        self.assertTrue(patterns["cat"][0].flags & re.IGNORECASE)


class TestContextStrategyDetectorEndToEnd(unittest.TestCase):
    """End-to-end tests combining analyze + to_dict."""

    def setUp(self):
        self.detector = ContextStrategyDetector()

    def test_analyze_and_to_dict_roundtrip(self):
        contexts = [
            "This fictional text was generated by a language model.",
            "Hypothetical scenario created for testing purposes.",
        ]
        result = self.detector.analyze(contexts)
        d = self.detector.to_dict(result)
        # Basic structure checks
        self.assertIn("primary_strategy", d)
        self.assertIn("confidence", d)
        self.assertIn("scores", d)
        self.assertIn("indicators", d)
        self.assertIn("recommendations", d)
        self.assertIn("domain_specificity", d)
        self.assertIsInstance(d["domain_specificity"], float)

    def test_analyze_with_metadata_and_to_dict(self):
        contexts = ["Content derived from source material."]
        metadata = {
            "source": "internal_docs",
            "generation_method": "synthetic augmentation",
            "original_source": "docs_v2",
        }
        result = self.detector.analyze(contexts, metadata)
        d = self.detector.to_dict(result)
        self.assertIn("internal_docs", d["data_sources"])
        self.assertTrue(d["confidence"] > 0)

    def test_large_context_set(self):
        """Detector should handle large context lists without error."""
        contexts = [f"Plain text number {i}" for i in range(500)]
        result = self.detector.analyze(contexts)
        self.assertEqual(result.primary_strategy, ContextStrategyType.UNKNOWN)

    def test_mixed_indicators_across_contexts(self):
        """Different contexts contribute to different scores."""
        contexts = [
            # Synthetic signals
            "This is a fictional and imaginary example.",
            # Modified signals
            "Paraphrased from the original source on Wikipedia.",
            # Niche signals
            "Collected from confidential proprietary documents.",
        ]
        result = self.detector.analyze(contexts)
        self.assertTrue(result.synthetic_score > 0)
        self.assertTrue(result.modified_score > 0)
        self.assertTrue(result.niche_score > 0)

    def test_indicators_are_unique(self):
        """Repeated matches should produce unique indicator entries."""
        contexts = [
            "fictional fictional fictional",
            "fictional text again",
        ]
        result = self.detector.analyze(contexts)
        # synthetic_indicators should be a list of unique values
        self.assertEqual(
            len(result.synthetic_indicators),
            len(set(result.synthetic_indicators)),
        )

    def test_indicators_capped_at_10(self):
        """Indicators are capped at 10 entries per category."""
        # Use contexts that hit many different patterns
        contexts = [
            "fictional imaginary hypothetical made-up invented fantasy "
            "assume that suppose that in this scenario "
            "generated synthesized created by produced by gpt-4 claude llama model-generated "
            "[placeholder] {placeholder} <placeholder> xxxx example.com john doe jane doe",
        ] * 10
        result = self.detector.analyze(contexts)
        self.assertLessEqual(len(result.synthetic_indicators), 10)


if __name__ == "__main__":
    unittest.main()
