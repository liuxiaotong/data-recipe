"""Unit tests for PromptExtractor and related dataclasses.

Tests PromptTemplate, PromptLibrary, and PromptExtractor including:
- extract() method with various message formats
- extract_from_conversations() method
- to_dict() serialization
- export_templates() in json, markdown, and unsupported formats
- Internal methods: categorization, variable extraction, domain detection,
  deduplication, and statistics calculation
- Edge cases: empty inputs, non-dict messages, missing fields, etc.
"""

import hashlib
import json
import unittest

from datarecipe.extractors.prompt_extractor import (
    PromptExtractor,
    PromptLibrary,
    PromptTemplate,
)

# ==================== PromptTemplate Dataclass ====================


class TestPromptTemplateDataclass(unittest.TestCase):
    """Test PromptTemplate dataclass defaults and __post_init__."""

    def test_default_values(self):
        t = PromptTemplate(content="Hello world", category="system")
        self.assertEqual(t.content, "Hello world")
        self.assertEqual(t.category, "system")
        self.assertEqual(t.domain, "general")
        self.assertEqual(t.frequency, 1)
        self.assertEqual(t.variables, [])

    def test_post_init_hash_id(self):
        content = "You are a helpful assistant."
        t = PromptTemplate(content=content, category="system")
        expected_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        self.assertEqual(t.hash_id, expected_hash)

    def test_post_init_char_count(self):
        content = "Hello world"
        t = PromptTemplate(content=content, category="system")
        self.assertEqual(t.char_count, len(content))

    def test_post_init_word_count(self):
        content = "Hello world foo bar"
        t = PromptTemplate(content=content, category="system")
        self.assertEqual(t.word_count, 4)

    def test_explicit_hash_id_not_overwritten(self):
        t = PromptTemplate(content="test", category="system", hash_id="custom123")
        self.assertEqual(t.hash_id, "custom123")

    def test_explicit_char_count_not_overwritten(self):
        t = PromptTemplate(content="test", category="system", char_count=999)
        self.assertEqual(t.char_count, 999)

    def test_explicit_word_count_not_overwritten(self):
        t = PromptTemplate(content="test", category="system", word_count=999)
        self.assertEqual(t.word_count, 999)

    def test_empty_content(self):
        t = PromptTemplate(content="", category="system")
        self.assertEqual(t.char_count, 0)
        self.assertEqual(t.word_count, 0)
        # hash_id is still set because empty string has a hash
        self.assertTrue(len(t.hash_id) > 0)


# ==================== PromptLibrary Dataclass ====================


class TestPromptLibraryDataclass(unittest.TestCase):
    """Test PromptLibrary dataclass defaults and methods."""

    def test_default_values(self):
        lib = PromptLibrary()
        self.assertEqual(lib.templates, [])
        self.assertEqual(lib.total_extracted, 0)
        self.assertEqual(lib.unique_count, 0)
        self.assertEqual(lib.deduplication_ratio, 0.0)
        self.assertEqual(lib.category_counts, {})
        self.assertEqual(lib.domain_counts, {})
        self.assertEqual(lib.avg_length, 0.0)
        self.assertEqual(lib.max_length, 0)
        self.assertEqual(lib.min_length, 0)

    def test_summary_basic(self):
        lib = PromptLibrary(
            total_extracted=10,
            unique_count=5,
            deduplication_ratio=0.5,
            avg_length=100.0,
            category_counts={"system": 3, "task": 2},
        )
        summary = lib.summary()
        self.assertIn("Total Extracted: 10", summary)
        self.assertIn("Unique Templates: 5", summary)
        self.assertIn("50.0%", summary)
        self.assertIn("100 chars", summary)
        self.assertIn("system: 3", summary)
        self.assertIn("task: 2", summary)

    def test_summary_with_domains(self):
        lib = PromptLibrary(
            total_extracted=5,
            unique_count=5,
            deduplication_ratio=0.0,
            avg_length=50.0,
            category_counts={"system": 5},
            domain_counts={"technical": 3, "medical": 2},
        )
        summary = lib.summary()
        self.assertIn("By Domain:", summary)
        self.assertIn("technical: 3", summary)
        self.assertIn("medical: 2", summary)

    def test_summary_without_domains(self):
        lib = PromptLibrary(
            total_extracted=1,
            unique_count=1,
            deduplication_ratio=0.0,
            avg_length=10.0,
            category_counts={"system": 1},
            domain_counts={},
        )
        summary = lib.summary()
        self.assertNotIn("By Domain:", summary)

    def test_summary_domain_limits_to_5(self):
        lib = PromptLibrary(
            total_extracted=10,
            unique_count=10,
            deduplication_ratio=0.0,
            avg_length=10.0,
            category_counts={"system": 10},
            domain_counts={f"domain{i}": i for i in range(1, 8)},
        )
        summary = lib.summary()
        # Should only show top 5 domains
        lines_with_domain = [
            line
            for line in summary.split("\n")
            if line.strip().startswith("- domain")
        ]
        self.assertLessEqual(len(lines_with_domain), 5)

    def test_get_by_category(self):
        t1 = PromptTemplate(content="sys1", category="system")
        t2 = PromptTemplate(content="task1", category="task")
        t3 = PromptTemplate(content="sys2", category="system")
        lib = PromptLibrary(templates=[t1, t2, t3])

        system_templates = lib.get_by_category("system")
        self.assertEqual(len(system_templates), 2)
        self.assertTrue(all(t.category == "system" for t in system_templates))

    def test_get_by_category_empty(self):
        lib = PromptLibrary(templates=[])
        result = lib.get_by_category("system")
        self.assertEqual(result, [])

    def test_get_by_category_no_match(self):
        t1 = PromptTemplate(content="test", category="system")
        lib = PromptLibrary(templates=[t1])
        result = lib.get_by_category("task")
        self.assertEqual(result, [])

    def test_categories_sorted_by_count_descending(self):
        lib = PromptLibrary(
            total_extracted=10,
            unique_count=10,
            deduplication_ratio=0.0,
            avg_length=10.0,
            category_counts={"task": 2, "system": 5, "format": 1},
        )
        summary = lib.summary()
        # system (5) should appear before task (2) before format (1)
        system_pos = summary.index("system: 5")
        task_pos = summary.index("task: 2")
        format_pos = summary.index("format: 1")
        self.assertLess(system_pos, task_pos)
        self.assertLess(task_pos, format_pos)


# ==================== PromptExtractor Initialization ====================


class TestPromptExtractorInit(unittest.TestCase):
    """Test PromptExtractor initialization."""

    def test_default_init(self):
        ext = PromptExtractor()
        self.assertEqual(ext.similarity_threshold, 0.85)
        self.assertEqual(ext.max_unique, 1000)

    def test_custom_init(self):
        ext = PromptExtractor(similarity_threshold=0.9, max_unique=500)
        self.assertEqual(ext.similarity_threshold, 0.9)
        self.assertEqual(ext.max_unique, 500)

    def test_category_regexes_compiled(self):
        ext = PromptExtractor()
        self.assertIn("system", ext.category_regexes)
        self.assertIn("task", ext.category_regexes)
        self.assertIn("constraint", ext.category_regexes)
        self.assertIn("format", ext.category_regexes)
        self.assertIn("example", ext.category_regexes)

    def test_variable_regexes_compiled(self):
        ext = PromptExtractor()
        self.assertEqual(len(ext.variable_regexes), 5)


# ==================== PromptExtractor.extract() ====================


class TestPromptExtractorExtract(unittest.TestCase):
    """Test the main extract() method."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_extract_system_message(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 1)
        self.assertEqual(lib.unique_count, 1)
        self.assertEqual(lib.templates[0].category, "system")
        self.assertIn("You are a helpful assistant.", lib.templates[0].content)

    def test_extract_user_message_with_prompt(self):
        messages = [
            {
                "role": "user",
                "content": "You are a code reviewer. Please review this function.",
            },
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 1)
        self.assertEqual(lib.unique_count, 1)

    def test_extract_user_message_without_prompt_indicators(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 0)
        self.assertEqual(lib.unique_count, 0)

    def test_extract_assistant_messages_ignored(self):
        messages = [
            {
                "role": "assistant",
                "content": "You are right. Please continue with your analysis.",
            },
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 0)

    def test_extract_empty_messages(self):
        lib = self.ext.extract([])
        self.assertEqual(lib.total_extracted, 0)
        self.assertEqual(lib.unique_count, 0)
        self.assertEqual(lib.templates, [])

    def test_extract_non_dict_messages_skipped(self):
        messages = ["not a dict", 42, None, True]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 0)

    def test_extract_missing_content_skipped(self):
        messages = [
            {"role": "system"},
            {"role": "system", "content": ""},
            {"role": "system", "content": None},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 0)

    def test_extract_content_not_string_skipped(self):
        messages = [
            {"role": "system", "content": 123},
            {"role": "system", "content": ["list", "of", "strings"]},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 0)

    def test_extract_missing_role_skipped(self):
        messages = [
            {"content": "You are a helpful assistant."},
        ]
        lib = self.ext.extract(messages)
        # role is empty string, not "system" or matching user prompt-like
        self.assertEqual(lib.total_extracted, 0)

    def test_extract_multiple_system_messages(self):
        messages = [
            {"role": "system", "content": "You are assistant A."},
            {"role": "system", "content": "You are assistant B."},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 2)
        self.assertEqual(lib.unique_count, 2)

    def test_extract_duplicate_system_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 3)
        self.assertEqual(lib.unique_count, 1)
        self.assertEqual(lib.templates[0].frequency, 3)

    def test_extract_without_deduplication(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        lib = self.ext.extract(messages, deduplicate=False)
        self.assertEqual(lib.total_extracted, 2)
        self.assertEqual(lib.unique_count, 2)

    def test_extract_deduplication_ratio(self):
        messages = [
            {"role": "system", "content": "You are assistant A."},
            {"role": "system", "content": "You are assistant A."},
            {"role": "system", "content": "You are assistant B."},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 3)
        self.assertEqual(lib.unique_count, 2)
        # dedup ratio = 1 - (2/3) = 0.333...
        self.assertAlmostEqual(lib.deduplication_ratio, 1 / 3, places=5)

    def test_extract_strips_content(self):
        messages = [
            {"role": "system", "content": "  You are a helpful assistant.  "},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(
            lib.templates[0].content, "You are a helpful assistant."
        )

    def test_extract_mixed_messages(self):
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine, thank you!"},
            {
                "role": "user",
                "content": "Please analyze the following code for bugs.",
            },
        ]
        lib = self.ext.extract(messages)
        # 1 system + 1 user with "please" indicator
        self.assertEqual(lib.total_extracted, 2)


# ==================== PromptExtractor.extract_from_conversations() ====================


class TestPromptExtractorExtractFromConversations(unittest.TestCase):
    """Test extract_from_conversations() method."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_extract_from_single_conversation(self):
        convs = [
            [
                {"role": "system", "content": "You are a teacher."},
                {"role": "user", "content": "What is math?"},
            ],
        ]
        lib = self.ext.extract_from_conversations(convs)
        self.assertEqual(lib.total_extracted, 1)

    def test_extract_from_multiple_conversations(self):
        convs = [
            [
                {"role": "system", "content": "You are a teacher."},
            ],
            [
                {"role": "system", "content": "You are a doctor."},
            ],
        ]
        lib = self.ext.extract_from_conversations(convs)
        self.assertEqual(lib.total_extracted, 2)

    def test_extract_from_conversations_deduplicates_across(self):
        convs = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
        ]
        lib = self.ext.extract_from_conversations(convs)
        self.assertEqual(lib.total_extracted, 2)
        self.assertEqual(lib.unique_count, 1)

    def test_extract_from_conversations_empty(self):
        lib = self.ext.extract_from_conversations([])
        self.assertEqual(lib.total_extracted, 0)

    def test_extract_from_conversations_non_list_items_skipped(self):
        convs = [
            "not a list",
            None,
            [{"role": "system", "content": "Valid system prompt."}],
        ]
        lib = self.ext.extract_from_conversations(convs)
        self.assertEqual(lib.total_extracted, 1)

    def test_extract_from_conversations_without_dedup(self):
        convs = [
            [{"role": "system", "content": "Same prompt."}],
            [{"role": "system", "content": "Same prompt."}],
        ]
        lib = self.ext.extract_from_conversations(convs, deduplicate=False)
        self.assertEqual(lib.unique_count, 2)


# ==================== Internal: _looks_like_prompt() ====================


class TestLooksLikePrompt(unittest.TestCase):
    """Test _looks_like_prompt() detection."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_you_are_detected(self):
        self.assertTrue(self.ext._looks_like_prompt("You are an expert analyst."))

    def test_your_task_detected(self):
        self.assertTrue(self.ext._looks_like_prompt("Your task is to summarize."))

    def test_please_detected(self):
        self.assertTrue(self.ext._looks_like_prompt("Please help me with this."))

    def test_instructions_detected(self):
        self.assertTrue(
            self.ext._looks_like_prompt("Instructions: follow these steps.")
        )

    def test_follow_these_detected(self):
        self.assertTrue(self.ext._looks_like_prompt("Follow these guidelines."))

    def test_given_the_detected(self):
        self.assertTrue(
            self.ext._looks_like_prompt("Given the following context, answer.")
        )

    def test_plain_question_not_detected(self):
        self.assertFalse(self.ext._looks_like_prompt("What is 2+2?"))

    def test_short_greeting_not_detected(self):
        self.assertFalse(self.ext._looks_like_prompt("Hello!"))

    def test_case_insensitive(self):
        self.assertTrue(self.ext._looks_like_prompt("YOU ARE an expert."))
        self.assertTrue(self.ext._looks_like_prompt("PLEASE analyze this."))


# ==================== Internal: _categorize() ====================


class TestCategorize(unittest.TestCase):
    """Test _categorize() method."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_system_category(self):
        content = "You are a helpful AI assistant. Your role is to help users."
        self.assertEqual(self.ext._categorize(content), "system")

    def test_task_category(self):
        content = "Please solve the problem and answer the question."
        self.assertEqual(self.ext._categorize(content), "task")

    def test_constraint_category(self):
        content = "Do not make up information. You must not hallucinate. Never lie."
        self.assertEqual(self.ext._categorize(content), "constraint")

    def test_format_category(self):
        content = "Format your response in JSON. Use the following format."
        self.assertEqual(self.ext._categorize(content), "format")

    def test_example_category(self):
        content = "Example: here is an example of good output. For example, you might say."
        self.assertEqual(self.ext._categorize(content), "example")

    def test_other_category_no_match(self):
        content = "Random text with no prompt indicators whatsoever."
        self.assertEqual(self.ext._categorize(content), "other")

    def test_highest_score_wins(self):
        # System patterns: "you are" + "your role" + "act as" = 3
        # vs constraint: "never" = 1
        content = "You are a coder. Your role is critical. Act as a mentor. Never stop."
        result = self.ext._categorize(content)
        self.assertEqual(result, "system")


# ==================== Internal: _extract_variables() ====================


class TestExtractVariables(unittest.TestCase):
    """Test _extract_variables() method."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_curly_brace_variables(self):
        content = "Hello {name}, your order {order_id} is ready."
        variables = self.ext._extract_variables(content)
        self.assertIn("name", variables)
        self.assertIn("order_id", variables)

    def test_square_bracket_variables(self):
        content = "The [topic] is important for [audience]."
        variables = self.ext._extract_variables(content)
        self.assertIn("topic", variables)
        self.assertIn("audience", variables)

    def test_angle_bracket_variables(self):
        content = "Insert <input> here and provide <output>."
        variables = self.ext._extract_variables(content)
        self.assertIn("input", variables)
        self.assertIn("output", variables)

    def test_dollar_brace_variables(self):
        content = "Use ${HOME} and ${USER} for paths."
        variables = self.ext._extract_variables(content)
        self.assertIn("HOME", variables)
        self.assertIn("USER", variables)

    def test_underscore_variables(self):
        content = "Fill in __blank1__ and __blank2__."
        variables = self.ext._extract_variables(content)
        self.assertIn("blank1", variables)
        self.assertIn("blank2", variables)

    def test_no_variables(self):
        content = "No variables in this text."
        variables = self.ext._extract_variables(content)
        self.assertEqual(variables, [])

    def test_mixed_variable_types(self):
        content = "Hello {name}, use <tool> on __item__."
        variables = self.ext._extract_variables(content)
        self.assertIn("name", variables)
        self.assertIn("tool", variables)
        self.assertIn("item", variables)

    def test_duplicate_variables_deduplicated(self):
        content = "Use {name} and {name} again."
        variables = self.ext._extract_variables(content)
        self.assertEqual(variables.count("name"), 1)


# ==================== Internal: _detect_domain() ====================


class TestDetectDomain(unittest.TestCase):
    """Test _detect_domain() method."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_legal_domain(self):
        content = "Analyze the legal implications of this contract for the attorney."
        self.assertEqual(self.ext._detect_domain(content), "legal")

    def test_medical_domain(self):
        content = "The patient diagnosis requires a doctor review of medical records."
        self.assertEqual(self.ext._detect_domain(content), "medical")

    def test_technical_domain(self):
        content = "Debug this code. The programming api has a software bug."
        self.assertEqual(self.ext._detect_domain(content), "technical")

    def test_education_domain(self):
        content = "Teach the student about education and explain concepts."
        self.assertEqual(self.ext._detect_domain(content), "education")

    def test_creative_domain(self):
        content = "Write a creative fiction story with a poem."
        self.assertEqual(self.ext._detect_domain(content), "creative")

    def test_business_domain(self):
        content = "Analyze the market for this product. Sales and customer data."
        self.assertEqual(self.ext._detect_domain(content), "business")

    def test_scientific_domain(self):
        content = "Design an experiment to test the hypothesis with data analysis."
        self.assertEqual(self.ext._detect_domain(content), "scientific")

    def test_general_domain_no_keywords(self):
        content = "Hello world."
        self.assertEqual(self.ext._detect_domain(content), "general")

    def test_highest_score_domain_wins(self):
        # medical: medical + patient + doctor = 3
        # scientific: data = 1
        content = "The medical patient sees the doctor for data review."
        self.assertEqual(self.ext._detect_domain(content), "medical")


# ==================== Internal: _deduplicate() ====================


class TestDeduplicate(unittest.TestCase):
    """Test _deduplicate() method."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_empty_list(self):
        result = self.ext._deduplicate([])
        self.assertEqual(result, [])

    def test_no_duplicates(self):
        templates = [
            PromptTemplate(content="Template A", category="system"),
            PromptTemplate(content="Template B", category="system"),
        ]
        result = self.ext._deduplicate(templates)
        self.assertEqual(len(result), 2)

    def test_exact_duplicates_merged(self):
        templates = [
            PromptTemplate(content="Same content", category="system"),
            PromptTemplate(content="Same content", category="system"),
            PromptTemplate(content="Same content", category="system"),
        ]
        result = self.ext._deduplicate(templates)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].frequency, 3)

    def test_sorted_by_frequency_descending(self):
        templates = [
            PromptTemplate(content="Rare", category="system"),
            PromptTemplate(content="Common", category="system"),
            PromptTemplate(content="Common", category="system"),
            PromptTemplate(content="Common", category="system"),
        ]
        result = self.ext._deduplicate(templates)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "Common")
        self.assertEqual(result[0].frequency, 3)
        self.assertEqual(result[1].content, "Rare")
        self.assertEqual(result[1].frequency, 1)

    def test_max_unique_limit(self):
        ext = PromptExtractor(max_unique=3)
        templates = [
            PromptTemplate(content=f"Template {i}", category="system")
            for i in range(10)
        ]
        result = ext._deduplicate(templates)
        self.assertEqual(len(result), 3)


# ==================== Internal: _calculate_stats() ====================


class TestCalculateStats(unittest.TestCase):
    """Test _calculate_stats() method."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_empty_library(self):
        lib = PromptLibrary()
        self.ext._calculate_stats(lib)
        # Should not crash; values remain defaults
        self.assertEqual(lib.avg_length, 0.0)
        self.assertEqual(lib.max_length, 0)
        self.assertEqual(lib.min_length, 0)

    def test_category_counts_calculated(self):
        lib = PromptLibrary(
            templates=[
                PromptTemplate(content="a", category="system"),
                PromptTemplate(content="b", category="system"),
                PromptTemplate(content="c", category="task"),
            ]
        )
        self.ext._calculate_stats(lib)
        self.assertEqual(lib.category_counts["system"], 2)
        self.assertEqual(lib.category_counts["task"], 1)

    def test_domain_counts_calculated(self):
        lib = PromptLibrary(
            templates=[
                PromptTemplate(content="a", category="system", domain="technical"),
                PromptTemplate(content="b", category="system", domain="technical"),
                PromptTemplate(content="c", category="system", domain="medical"),
            ]
        )
        self.ext._calculate_stats(lib)
        self.assertEqual(lib.domain_counts["technical"], 2)
        self.assertEqual(lib.domain_counts["medical"], 1)

    def test_length_stats_calculated(self):
        lib = PromptLibrary(
            templates=[
                PromptTemplate(content="short", category="system"),
                PromptTemplate(content="medium length text", category="system"),
                PromptTemplate(
                    content="a very long text that has many characters in it",
                    category="system",
                ),
            ]
        )
        self.ext._calculate_stats(lib)
        lengths = [t.char_count for t in lib.templates]
        self.assertEqual(lib.max_length, max(lengths))
        self.assertEqual(lib.min_length, min(lengths))
        self.assertAlmostEqual(
            lib.avg_length, sum(lengths) / len(lengths), places=5
        )


# ==================== PromptExtractor.to_dict() ====================


class TestToDict(unittest.TestCase):
    """Test to_dict() serialization."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_basic_structure(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        lib = self.ext.extract(messages)
        d = self.ext.to_dict(lib)

        self.assertIn("total_extracted", d)
        self.assertIn("unique_count", d)
        self.assertIn("deduplication_ratio", d)
        self.assertIn("avg_length", d)
        self.assertIn("category_counts", d)
        self.assertIn("domain_counts", d)
        self.assertIn("templates", d)

    def test_template_fields(self):
        messages = [
            {"role": "system", "content": "You are a {role} assistant."},
        ]
        lib = self.ext.extract(messages)
        d = self.ext.to_dict(lib)

        template = d["templates"][0]
        self.assertIn("hash_id", template)
        self.assertIn("content", template)
        self.assertIn("category", template)
        self.assertIn("domain", template)
        self.assertIn("frequency", template)
        self.assertIn("char_count", template)
        self.assertIn("word_count", template)
        self.assertIn("variables", template)
        self.assertIn("role", template["variables"])

    def test_empty_library(self):
        lib = PromptLibrary()
        d = self.ext.to_dict(lib)
        self.assertEqual(d["total_extracted"], 0)
        self.assertEqual(d["unique_count"], 0)
        self.assertEqual(d["templates"], [])

    def test_serializable_to_json(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        lib = self.ext.extract(messages)
        d = self.ext.to_dict(lib)
        # Should be JSON serializable without errors
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)


# ==================== PromptExtractor.export_templates() ====================


class TestExportTemplates(unittest.TestCase):
    """Test export_templates() in various formats."""

    def setUp(self):
        self.ext = PromptExtractor()
        messages = [
            {"role": "system", "content": "You are a coding expert. Debug this software code."},
            {"role": "system", "content": "You are a medical advisor for patient health."},
        ]
        self.lib = self.ext.extract(messages)

    def test_export_json(self):
        result = self.ext.export_templates(self.lib, format="json")
        parsed = json.loads(result)
        self.assertIn("total_extracted", parsed)
        self.assertIn("templates", parsed)
        self.assertEqual(len(parsed["templates"]), 2)

    def test_export_json_is_indented(self):
        result = self.ext.export_templates(self.lib, format="json")
        # Indented JSON has newlines
        self.assertIn("\n", result)

    def test_export_markdown(self):
        result = self.ext.export_templates(self.lib, format="markdown")
        self.assertIn("# Prompt Template Library", result)
        self.assertIn("unique templates", result)

    def test_export_markdown_contains_categories(self):
        result = self.ext.export_templates(self.lib, format="markdown")
        self.assertIn("## System", result)

    def test_export_markdown_contains_template_content(self):
        result = self.ext.export_templates(self.lib, format="markdown")
        self.assertIn("You are a coding expert", result)

    def test_export_markdown_contains_domain_info(self):
        result = self.ext.export_templates(self.lib, format="markdown")
        self.assertIn("Domain:", result)

    def test_export_markdown_contains_frequency_info(self):
        result = self.ext.export_templates(self.lib, format="markdown")
        self.assertIn("Frequency:", result)

    def test_export_markdown_truncates_long_content(self):
        messages = [
            {"role": "system", "content": "A" * 600},
        ]
        lib = self.ext.extract(messages)
        result = self.ext.export_templates(lib, format="markdown")
        self.assertIn("...", result)

    def test_export_markdown_no_truncation_short_content(self):
        messages = [
            {"role": "system", "content": "Short content."},
        ]
        lib = self.ext.extract(messages)
        result = self.ext.export_templates(lib, format="markdown")
        self.assertNotIn("...", result)

    def test_export_unsupported_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.ext.export_templates(self.lib, format="csv")
        self.assertIn("Unsupported format", str(ctx.exception))
        self.assertIn("csv", str(ctx.exception))


# ==================== Integration Tests ====================


class TestPromptExtractorIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_full_pipeline(self):
        """Test complete extraction pipeline with realistic messages."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional code reviewer. "
                    "Your role is to analyze code for bugs and security issues. "
                    "Do not suggest style changes. Always focus on correctness. "
                    "Format your response in JSON with {severity} and {description} fields."
                ),
            },
            {
                "role": "user",
                "content": "Please review the following Python function for bugs.",
            },
            {"role": "assistant", "content": "I'll review the code now."},
            {
                "role": "user",
                "content": "Given the analysis, please suggest improvements.",
            },
        ]
        lib = self.ext.extract(messages)

        # System message + 2 user messages with prompt indicators
        self.assertEqual(lib.total_extracted, 3)

        # Verify the system prompt has correct attributes
        system_templates = lib.get_by_category("system")
        self.assertGreater(len(system_templates), 0)
        sys_template = system_templates[0]
        self.assertIn("severity", sys_template.variables)
        self.assertIn("description", sys_template.variables)

        # Verify stats are populated
        self.assertGreater(lib.avg_length, 0)
        self.assertGreater(lib.max_length, 0)
        self.assertGreater(lib.min_length, 0)

        # Verify to_dict and export work
        d = self.ext.to_dict(lib)
        self.assertIsInstance(d, dict)

        json_export = self.ext.export_templates(lib, format="json")
        self.assertIsInstance(json.loads(json_export), dict)

        md_export = self.ext.export_templates(lib, format="markdown")
        self.assertIn("# Prompt Template Library", md_export)

    def test_large_scale_deduplication(self):
        """Test with many duplicate messages."""
        base_prompts = [
            "You are a helpful assistant.",
            "You are a coding expert.",
            "You are a writing tutor.",
        ]
        messages = []
        for prompt in base_prompts:
            for _ in range(100):
                messages.append({"role": "system", "content": prompt})

        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 300)
        self.assertEqual(lib.unique_count, 3)
        self.assertAlmostEqual(lib.deduplication_ratio, 1 - 3 / 300, places=5)

        # Most frequent should have frequency 100
        self.assertEqual(lib.templates[0].frequency, 100)

    def test_conversation_level_extraction(self):
        """Test extract_from_conversations with multiple conversations."""
        conversations = [
            [
                {"role": "system", "content": "You are a legal advisor."},
                {"role": "user", "content": "What are the contract terms?"},
            ],
            [
                {"role": "system", "content": "You are a medical doctor."},
                {
                    "role": "user",
                    "content": "Please diagnose the patient symptoms.",
                },
            ],
            [
                {"role": "system", "content": "You are a legal advisor."},
                {"role": "user", "content": "Review the court decision."},
            ],
        ]
        lib = self.ext.extract_from_conversations(conversations)

        # 3 system + 1 user with "please"
        self.assertEqual(lib.total_extracted, 4)

        # Check domains
        self.assertIn("legal", lib.domain_counts)
        self.assertIn("medical", lib.domain_counts)


class TestPromptExtractorEdgeCases(unittest.TestCase):
    """Test edge cases and unusual inputs."""

    def setUp(self):
        self.ext = PromptExtractor()

    def test_very_long_content(self):
        content = "You are an assistant. " * 10000
        messages = [{"role": "system", "content": content}]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 1)
        self.assertGreater(lib.templates[0].char_count, 0)

    def test_unicode_content(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Please respond in Japanese. "},
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 1)

    def test_multiline_content(self):
        content = "You are a helpful assistant.\n\nPlease follow these rules:\n1. Be concise\n2. Be accurate"
        messages = [{"role": "system", "content": content}]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 1)
        self.assertIn("\n", lib.templates[0].content)

    def test_only_whitespace_content(self):
        messages = [{"role": "system", "content": "   \n\t  "}]
        lib = self.ext.extract(messages)
        # Non-empty string passes the check, but content is whitespace
        # It will be extracted since isinstance check passes and content is truthy
        # after strip it becomes empty
        self.assertEqual(lib.total_extracted, 1)

    def test_special_characters_in_content(self):
        messages = [
            {
                "role": "system",
                "content": 'You are <b>bold</b> & "quoted" \'single\' @#$%^&*()',
            },
        ]
        lib = self.ext.extract(messages)
        self.assertEqual(lib.total_extracted, 1)

    def test_max_unique_zero(self):
        ext = PromptExtractor(max_unique=0)
        messages = [
            {"role": "system", "content": "Template A"},
        ]
        lib = ext.extract(messages)
        # After deduplication with max_unique=0, no templates are kept
        self.assertEqual(lib.unique_count, 0)

    def test_export_markdown_empty_library(self):
        lib = PromptLibrary()
        result = self.ext.export_templates(lib, format="markdown")
        self.assertIn("# Prompt Template Library", result)
        self.assertIn("0 unique templates", result)

    def test_export_json_empty_library(self):
        lib = PromptLibrary()
        result = self.ext.export_templates(lib, format="json")
        parsed = json.loads(result)
        self.assertEqual(parsed["templates"], [])

    def test_summary_on_extracted_library(self):
        messages = [
            {"role": "system", "content": "You are a coding assistant for software development."},
            {"role": "system", "content": "You are a medical advisor for patient health."},
        ]
        lib = self.ext.extract(messages)
        summary = lib.summary()
        self.assertIn("Total Extracted:", summary)
        self.assertIn("Unique Templates:", summary)
        self.assertIn("By Category:", summary)


if __name__ == "__main__":
    unittest.main()
