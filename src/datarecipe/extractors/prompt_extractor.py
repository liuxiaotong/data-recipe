"""
System Prompt Template Extractor

Extracts, deduplicates, and categorizes prompt templates from datasets.
"""

import re
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
# from difflib import SequenceMatcher  # Kept for potential fuzzy deduplication


@dataclass
class PromptTemplate:
    """A single prompt template extracted from the dataset."""

    content: str                    # Full prompt content
    category: str                   # system/task/example/constraint/format
    domain: str = "general"         # Domain where used
    frequency: int = 1              # Usage count
    char_count: int = 0             # Character length
    word_count: int = 0             # Word count
    variables: list[str] = field(default_factory=list)  # Detected variables
    hash_id: str = ""               # Unique identifier

    def __post_init__(self):
        if not self.hash_id:
            self.hash_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
        if not self.char_count:
            self.char_count = len(self.content)
        if not self.word_count:
            self.word_count = len(self.content.split())


@dataclass
class PromptLibrary:
    """Collection of extracted prompt templates."""

    templates: list[PromptTemplate] = field(default_factory=list)
    total_extracted: int = 0
    unique_count: int = 0
    deduplication_ratio: float = 0.0

    # Statistics by category
    category_counts: dict[str, int] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)

    # Metadata
    avg_length: float = 0.0
    max_length: int = 0
    min_length: int = 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total Extracted: {self.total_extracted}",
            f"Unique Templates: {self.unique_count}",
            f"Deduplication Ratio: {self.deduplication_ratio:.1%}",
            f"Avg Length: {self.avg_length:.0f} chars",
            "",
            "By Category:",
        ]
        for cat, count in sorted(self.category_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {cat}: {count}")

        if self.domain_counts:
            lines.append("")
            lines.append("By Domain:")
            for domain, count in sorted(self.domain_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {domain}: {count}")

        return "\n".join(lines)

    def get_by_category(self, category: str) -> list[PromptTemplate]:
        """Get templates by category."""
        return [t for t in self.templates if t.category == category]


class PromptExtractor:
    """
    Extracts and analyzes prompt templates from datasets.

    Example usage:
        extractor = PromptExtractor()
        library = extractor.extract(messages_list)
        print(library.summary())
    """

    # Prompt category patterns
    CATEGORY_PATTERNS = {
        "system": [
            r"you are",
            r"your role",
            r"as an? \w+ assistant",
            r"act as",
            r"your task is",
        ],
        "task": [
            r"please \w+",
            r"your goal is",
            r"complete the following",
            r"answer the question",
            r"solve the problem",
        ],
        "constraint": [
            r"do not",
            r"must not",
            r"never",
            r"always",
            r"make sure",
            r"ensure that",
        ],
        "format": [
            r"format your",
            r"output format",
            r"respond in",
            r"use the following format",
            r"return.*json",
        ],
        "example": [
            r"example:",
            r"for example",
            r"here is an example",
            r"sample:",
        ],
    }

    # Variable detection patterns
    VARIABLE_PATTERNS = [
        r"\{(\w+)\}",           # {variable}
        r"\[(\w+)\]",           # [variable]
        r"<(\w+)>",             # <variable>
        r"\$\{(\w+)\}",         # ${variable}
        r"__(\w+)__",           # __variable__
    ]

    def __init__(self, similarity_threshold: float = 0.85, max_unique: int = 1000):
        """
        Initialize the extractor.

        Args:
            similarity_threshold: Threshold for considering prompts as duplicates
            max_unique: Maximum number of unique templates to keep (for performance)
        """
        self.similarity_threshold = similarity_threshold
        self.max_unique = max_unique

        # Compile category patterns
        self.category_regexes = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.CATEGORY_PATTERNS.items()
        }

        # Compile variable patterns
        self.variable_regexes = [
            re.compile(p) for p in self.VARIABLE_PATTERNS
        ]

    def extract(
        self,
        messages: list[dict],
        deduplicate: bool = True
    ) -> PromptLibrary:
        """
        Extract prompt templates from a list of message dictionaries.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            deduplicate: Whether to remove similar prompts

        Returns:
            PromptLibrary with extracted templates
        """
        library = PromptLibrary()
        raw_templates: list[PromptTemplate] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            if not content or not isinstance(content, str):
                continue

            # Extract system prompts
            if role == "system":
                template = self._create_template(content, "system")
                raw_templates.append(template)

            # Also check user messages for embedded prompts
            elif role == "user":
                # Look for prompt-like content in user messages
                if self._looks_like_prompt(content):
                    category = self._categorize(content)
                    template = self._create_template(content, category)
                    raw_templates.append(template)

        library.total_extracted = len(raw_templates)

        # Deduplicate if requested
        if deduplicate and raw_templates:
            unique_templates = self._deduplicate(raw_templates)
        else:
            unique_templates = raw_templates

        library.templates = unique_templates
        library.unique_count = len(unique_templates)

        if library.total_extracted > 0:
            library.deduplication_ratio = 1 - (
                library.unique_count / library.total_extracted
            )

        # Calculate statistics
        self._calculate_stats(library)

        return library

    def extract_from_conversations(
        self,
        conversations: list[list[dict]],
        deduplicate: bool = True
    ) -> PromptLibrary:
        """
        Extract from multiple conversations.

        Args:
            conversations: List of conversations, each being a list of messages
            deduplicate: Whether to remove similar prompts

        Returns:
            PromptLibrary with extracted templates
        """
        all_messages = []
        for conv in conversations:
            if isinstance(conv, list):
                all_messages.extend(conv)

        return self.extract(all_messages, deduplicate)

    def _create_template(self, content: str, category: str) -> PromptTemplate:
        """Create a PromptTemplate from content."""
        variables = self._extract_variables(content)
        domain = self._detect_domain(content)

        return PromptTemplate(
            content=content.strip(),
            category=category,
            domain=domain,
            variables=variables,
        )

    def _looks_like_prompt(self, content: str) -> bool:
        """Check if content looks like a prompt/instruction."""
        content_lower = content.lower()

        # Check for common prompt indicators
        indicators = [
            "you are",
            "your task",
            "please",
            "instructions:",
            "follow these",
            "given the",
        ]

        return any(ind in content_lower for ind in indicators)

    def _categorize(self, content: str) -> str:
        """Categorize a prompt based on its content."""
        content_lower = content.lower()

        scores = {}
        for category, regexes in self.category_regexes.items():
            score = sum(
                1 for regex in regexes if regex.search(content_lower)
            )
            scores[category] = score

        if not scores or max(scores.values()) == 0:
            return "other"

        return max(scores, key=scores.get)

    def _extract_variables(self, content: str) -> list[str]:
        """Extract variable placeholders from content."""
        variables = []
        for regex in self.variable_regexes:
            matches = regex.findall(content)
            variables.extend(matches)
        return list(set(variables))

    def _detect_domain(self, content: str) -> str:
        """Detect the domain of a prompt."""
        content_lower = content.lower()

        domain_keywords = {
            "legal": ["law", "legal", "court", "attorney", "contract"],
            "medical": ["medical", "health", "doctor", "patient", "diagnosis"],
            "technical": ["code", "programming", "software", "api", "debug"],
            "education": ["teach", "learn", "student", "explain", "education"],
            "creative": ["write", "story", "creative", "fiction", "poem"],
            "business": ["business", "market", "sales", "customer", "product"],
            "scientific": ["research", "experiment", "hypothesis", "data", "analysis"],
        }

        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return "general"

        return max(scores, key=scores.get)

    def _deduplicate(
        self,
        templates: list[PromptTemplate]
    ) -> list[PromptTemplate]:
        """Remove duplicate templates using hash-based deduplication.

        This is O(n) complexity - fast and efficient for large datasets.
        Uses content hash for exact match deduplication.
        """
        if not templates:
            return []

        # Hash-based exact deduplication (O(n))
        hash_groups: dict[str, list[PromptTemplate]] = {}
        for template in templates:
            h = template.hash_id
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append(template)

        # Get one representative per unique hash
        unique = []
        for group in hash_groups.values():
            # Prefer longer templates (more complete)
            group.sort(key=lambda t: -t.char_count)
            rep = group[0]
            rep.frequency = len(group)
            unique.append(rep)

        # Sort by frequency (most common first)
        unique.sort(key=lambda t: -t.frequency)

        # Limit to max_unique for performance
        if len(unique) > self.max_unique:
            unique = unique[:self.max_unique]

        return unique

    # NOTE: _calculate_similarity is kept for potential future use in fuzzy deduplication.
    # Currently we use hash-based exact matching for O(n) performance.
    # Uncomment and integrate if similarity-based deduplication is needed.
    #
    # def _calculate_similarity(self, a: str, b: str) -> float:
    #     """Calculate similarity between two strings using SequenceMatcher."""
    #     a = a.lower().strip()
    #     b = b.lower().strip()
    #     return SequenceMatcher(None, a, b).ratio()

    def _calculate_stats(self, library: PromptLibrary) -> None:
        """Calculate statistics for the library."""
        if not library.templates:
            return

        # Category counts
        category_counter = Counter(t.category for t in library.templates)
        library.category_counts = dict(category_counter)

        # Domain counts
        domain_counter = Counter(t.domain for t in library.templates)
        library.domain_counts = dict(domain_counter)

        # Length stats
        lengths = [t.char_count for t in library.templates]
        library.avg_length = sum(lengths) / len(lengths)
        library.max_length = max(lengths)
        library.min_length = min(lengths)

    def to_dict(self, library: PromptLibrary) -> dict:
        """Convert library to dictionary for JSON export."""
        return {
            "total_extracted": library.total_extracted,
            "unique_count": library.unique_count,
            "deduplication_ratio": library.deduplication_ratio,
            "avg_length": library.avg_length,
            "category_counts": library.category_counts,
            "domain_counts": library.domain_counts,
            "templates": [
                {
                    "hash_id": t.hash_id,
                    "content": t.content,
                    "category": t.category,
                    "domain": t.domain,
                    "frequency": t.frequency,
                    "char_count": t.char_count,
                    "word_count": t.word_count,
                    "variables": t.variables,
                }
                for t in library.templates
            ],
        }

    def export_templates(
        self,
        library: PromptLibrary,
        format: str = "json"
    ) -> str:
        """
        Export templates in specified format.

        Args:
            library: The prompt library to export
            format: Output format ('json', 'yaml', 'markdown')

        Returns:
            Formatted string
        """
        import json

        if format == "json":
            return json.dumps(self.to_dict(library), indent=2, ensure_ascii=False)

        elif format == "markdown":
            lines = [
                "# Prompt Template Library",
                "",
                f"Total: {library.unique_count} unique templates",
                "",
            ]

            for category in sorted(library.category_counts.keys()):
                lines.append(f"## {category.title()}")
                lines.append("")
                templates = library.get_by_category(category)
                for t in templates[:10]:  # Limit to 10 per category
                    lines.append(f"### Template {t.hash_id}")
                    lines.append(f"- Domain: {t.domain}")
                    lines.append(f"- Frequency: {t.frequency}")
                    lines.append("```")
                    lines.append(t.content[:500])  # Truncate long content
                    if len(t.content) > 500:
                        lines.append("...")
                    lines.append("```")
                    lines.append("")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported format: {format}")
