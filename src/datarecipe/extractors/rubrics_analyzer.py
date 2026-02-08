"""
Rubrics/Evaluation Criteria Pattern Analyzer

Extracts patterns like "The response should [verb] [object] [condition]"
from dataset evaluation criteria.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class RubricPattern:
    """A single rubric pattern extracted from the dataset."""

    pattern: str  # Original pattern text
    verb: str  # Core verb (e.g., "include", "not", "explain")
    verb_phrase: str  # Full verb phrase (e.g., "should include")
    frequency: int = 1  # How many times this pattern appears
    examples: list[str] = field(default_factory=list)  # Original examples
    template: str = ""  # Abstracted template
    category: str = "general"  # Category (define, list, explain, etc.)
    action: str = ""  # Canonical action phrase (e.g., "should include")
    target: str = ""  # Target object / focus of the rubric
    condition: str = ""  # Additional conditions/constraints
    hash_id: str = ""  # Unique identifier

    def __post_init__(self):
        if not self.hash_id:
            self.hash_id = hashlib.md5(self.pattern.encode()).hexdigest()[:8]


@dataclass
class RubricsAnalysisResult:
    """Complete analysis result for rubrics patterns."""

    patterns: list[RubricPattern] = field(default_factory=list)
    verb_distribution: dict[str, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)
    top_templates: list[str] = field(default_factory=list)
    structured_templates: list[dict] = field(default_factory=list)

    total_rubrics: int = 0
    unique_patterns: int = 0
    avg_rubrics_per_task: float = 0.0

    # Pattern statistics
    sentence_starters: dict[str, int] = field(default_factory=dict)
    common_phrases: list[tuple[str, int]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total Rubrics: {self.total_rubrics}",
            f"Unique Patterns: {self.unique_patterns}",
            f"Avg Rubrics/Task: {self.avg_rubrics_per_task:.1f}",
            "",
            "Top Verbs:",
        ]
        for verb, count in sorted(self.verb_distribution.items(), key=lambda x: -x[1])[:10]:
            pct = count / self.total_rubrics * 100 if self.total_rubrics else 0
            lines.append(f"  - {verb}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("Top Categories:")
        for cat, count in sorted(self.category_distribution.items(), key=lambda x: -x[1])[:5]:
            pct = count / self.total_rubrics * 100 if self.total_rubrics else 0
            lines.append(f"  - {cat}: {count} ({pct:.1f}%)")

        return "\n".join(lines)


class RubricsAnalyzer:
    """
    Analyzes rubrics/evaluation criteria to extract patterns.

    Example usage:
        analyzer = RubricsAnalyzer()
        result = analyzer.analyze(rubrics_list)
        print(result.summary())
    """

    # Common rubric sentence starters
    STARTERS = [
        r"The response should",
        r"The answer should",
        r"The output should",
        r"The model should",
        r"It should",
        r"Should",
    ]

    # Verb categories for classification
    VERB_CATEGORIES = {
        "define": ["define", "explain", "describe", "clarify"],
        "list": ["list", "enumerate", "include", "mention", "provide"],
        "avoid": ["not", "avoid", "refrain", "never"],
        "state": ["state", "indicate", "specify", "note"],
        "verify": ["verify", "confirm", "check", "ensure"],
        "format": ["format", "structure", "organize", "present"],
        "compare": ["compare", "contrast", "differentiate", "distinguish"],
        "analyze": ["analyze", "evaluate", "assess", "examine"],
    }

    def __init__(self):
        # Build starter regex pattern
        self.starter_pattern = re.compile(r"(" + "|".join(self.STARTERS) + r")\s+", re.IGNORECASE)
        # Build verb extraction pattern
        self.verb_pattern = re.compile(r"should\s+(not\s+)?(\w+)", re.IGNORECASE)

    def analyze(
        self, rubrics: list[str], task_count: int | None = None
    ) -> RubricsAnalysisResult:
        """
        Analyze a list of rubrics to extract patterns.

        Args:
            rubrics: List of rubric strings
            task_count: Number of tasks (for calculating avg rubrics/task)

        Returns:
            RubricsAnalysisResult with extracted patterns and statistics
        """
        result = RubricsAnalysisResult()
        result.total_rubrics = len(rubrics)

        if task_count:
            result.avg_rubrics_per_task = len(rubrics) / task_count

        # Extract patterns
        verb_counter = Counter()
        category_counter = Counter()
        starter_counter = Counter()
        pattern_map: dict[str, RubricPattern] = {}

        for rubric in rubrics:
            rubric = rubric.strip()
            if not rubric:
                continue

            # Extract sentence starter
            starter_match = self.starter_pattern.match(rubric)
            if starter_match:
                starter = starter_match.group(1)
                starter_counter[starter] += 1

            # Extract verb
            verb, verb_phrase, is_negation = self._extract_verb(rubric)
            if verb:
                # Handle negation specially
                if is_negation:
                    verb_counter["not"] += 1
                    verb_counter[f"not {verb}"] += 1
                else:
                    verb_counter[verb] += 1

            # Categorize
            category = self._categorize(verb, is_negation)
            category_counter[category] += 1

            # Create or update pattern
            template = self._abstract_template(rubric)
            pattern_key = template.lower()

            action, target, condition = self._extract_structure(rubric, verb_phrase)

            if pattern_key in pattern_map:
                pattern_map[pattern_key].frequency += 1
                if len(pattern_map[pattern_key].examples) < 5:
                    pattern_map[pattern_key].examples.append(rubric)
            else:
                pattern_map[pattern_key] = RubricPattern(
                    pattern=rubric,
                    verb=verb or "unknown",
                    verb_phrase=verb_phrase or "",
                    template=template,
                    category=category,
                    action=action,
                    target=target,
                    condition=condition,
                    examples=[rubric],
                )

        # Compile results
        result.patterns = sorted(pattern_map.values(), key=lambda p: -p.frequency)
        result.unique_patterns = len(pattern_map)
        result.verb_distribution = dict(verb_counter)
        result.category_distribution = dict(category_counter)
        result.sentence_starters = dict(starter_counter)

        # Extract top templates
        result.top_templates = [p.template for p in result.patterns[:20]]

        # Extract common phrases
        result.common_phrases = self._extract_common_phrases(rubrics)

        # Structured template summaries
        result.structured_templates = self._build_structured_templates(result.patterns)

        return result

    def _extract_verb(self, rubric: str) -> tuple[str | None, str | None, bool]:
        """
        Extract the main verb from a rubric.

        Returns:
            (verb, verb_phrase, is_negation)
        """
        match = self.verb_pattern.search(rubric)
        if not match:
            return None, None, False

        negation = match.group(1)
        verb = match.group(2).lower()

        if negation:
            verb_phrase = f"should not {verb}"
            return verb, verb_phrase, True
        else:
            verb_phrase = f"should {verb}"
            return verb, verb_phrase, False

    def _categorize(self, verb: str | None, is_negation: bool) -> str:
        """Categorize a rubric based on its verb."""
        if is_negation:
            return "avoid"

        if not verb:
            return "other"

        verb_lower = verb.lower()
        for category, verbs in self.VERB_CATEGORIES.items():
            if verb_lower in verbs:
                return category

        return "other"

    def _abstract_template(self, rubric: str) -> str:
        """
        Abstract a rubric into a template by replacing specifics with placeholders.

        Example:
            "The response should include all 5 game rules"
            -> "The response should include [QUANTITY] [NOUN]"
        """
        template = rubric

        # Replace quoted strings
        template = re.sub(r'"[^"]*"', "[QUOTED]", template)
        template = re.sub(r"'[^']*'", "[QUOTED]", template)

        # Replace numbers
        template = re.sub(r"\b\d+\b", "[NUM]", template)

        # Replace specific lists
        template = re.sub(
            r"\b(all|each|every|any)\s+\d*\s*\w+s?\b", r"\1 [ITEMS]", template, flags=re.IGNORECASE
        )

        return template

    def _extract_structure(
        self,
        rubric: str,
        verb_phrase: str | None,
    ) -> tuple[str, str, str]:
        """Extract action/target/condition pieces from the rubric."""
        action = (verb_phrase or "").strip()
        text = rubric.strip()

        remainder = text
        if action:
            lower = text.lower()
            idx = lower.find(action.lower())
            if idx != -1:
                remainder = text[idx + len(action) :]

        remainder = remainder.strip(" .:;-\n")

        target = remainder
        condition = ""

        if remainder:
            # Split on connectors such as "if", "when", "so that", etc.
            connector_split = re.split(
                r"\b(if|when|unless|provided that|so that|because|to ensure|while)\b",
                remainder,
                maxsplit=1,
                flags=re.IGNORECASE,
            )

            if len(connector_split) > 1:
                target = connector_split[0].strip(" ,.;:()[]")
                condition = " ".join(connector_split[1:]).strip(" ,.;:()[]")
            else:
                # Try splitting on comma
                comma_split = remainder.split(",", 1)
                if len(comma_split) > 1 and len(comma_split[0]) > 10:
                    target = comma_split[0].strip(" ,.;:()[]")
                    condition = comma_split[1].strip(" ,.;:()[]")
                else:
                    quantifier_match = re.search(
                        r"\b(at least|at most|no more than|less than|greater than|minimum|maximum|must|should)\b",
                        remainder,
                        flags=re.IGNORECASE,
                    )
                    if quantifier_match and quantifier_match.start() > 0:
                        q_index = quantifier_match.start()
                        target = remainder[:q_index].strip(" ,.;:()[]")
                        condition = remainder[q_index:].strip(" ,.;:()[]")
                    else:
                        target = remainder.strip(" ,.;:()[]")

        if not target:
            target = remainder.strip(" ,.;:()[]")

        return action, target, condition

    def _build_structured_templates(
        self,
        patterns: list[RubricPattern],
        limit: int = 20,
    ) -> list[dict]:
        """Build structured template summaries for downstream consumption."""
        summaries = []
        for pattern in patterns[:limit]:
            summaries.append(
                {
                    "id": pattern.hash_id,
                    "category": pattern.category,
                    "action": pattern.action or pattern.verb_phrase,
                    "target": pattern.target,
                    "condition": pattern.condition,
                    "template": pattern.template,
                    "frequency": pattern.frequency,
                    "examples": pattern.examples[:2],
                }
            )
        return summaries

    def _extract_common_phrases(
        self, rubrics: list[str], min_freq: int = 5, max_phrases: int = 20
    ) -> list[tuple[str, int]]:
        """Extract commonly occurring phrases."""
        phrase_counter = Counter()

        # Extract 3-5 word phrases
        for rubric in rubrics:
            words = rubric.lower().split()
            for n in range(3, 6):
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i : i + n])
                    if phrase.startswith(("the response", "should")):
                        phrase_counter[phrase] += 1

        # Filter and return top phrases
        common = [(phrase, count) for phrase, count in phrase_counter.items() if count >= min_freq]
        common.sort(key=lambda x: -x[1])
        return common[:max_phrases]

    def generate_rubrics(
        self, analysis: RubricsAnalysisResult, context: str, count: int = 10
    ) -> list[str]:
        """
        Generate new rubrics based on discovered patterns.

        Args:
            analysis: Previous analysis result
            context: Context/topic for the new rubrics
            count: Number of rubrics to generate

        Returns:
            List of generated rubric strings
        """
        generated = []
        templates = analysis.top_templates[:count]

        for template in templates:
            # Simple template filling - replace placeholders
            rubric = template
            rubric = rubric.replace("[NUM]", "all")
            rubric = rubric.replace("[QUOTED]", f'"{context}"')
            rubric = rubric.replace("[ITEMS]", "relevant items")
            generated.append(rubric)

        return generated

    def to_dict(self, result: RubricsAnalysisResult) -> dict:
        """Convert analysis result to dictionary for JSON export."""
        return {
            "total_rubrics": result.total_rubrics,
            "unique_patterns": result.unique_patterns,
            "avg_rubrics_per_task": result.avg_rubrics_per_task,
            "verb_distribution": result.verb_distribution,
            "category_distribution": result.category_distribution,
            "sentence_starters": result.sentence_starters,
            "top_templates": result.top_templates,
            "structured_templates": result.structured_templates,
            "common_phrases": result.common_phrases,
            "patterns": [
                {
                    "pattern": p.pattern,
                    "verb": p.verb,
                    "verb_phrase": p.verb_phrase,
                    "frequency": p.frequency,
                    "template": p.template,
                    "category": p.category,
                    "examples": p.examples[:3],
                    "action": p.action,
                    "target": p.target,
                    "condition": p.condition,
                }
                for p in result.patterns[:50]
            ],
        }

    def to_yaml_templates(self, result: RubricsAnalysisResult) -> str:
        """Export structured templates as YAML."""
        import yaml

        templates = [
            {
                "id": entry.get("id"),
                "category": entry.get("category"),
                "action": entry.get("action"),
                "target": entry.get("target"),
                "condition": entry.get("condition"),
                "frequency": entry.get("frequency"),
                "examples": entry.get("examples", []),
            }
            for entry in result.structured_templates
        ]

        return yaml.dump(
            {"rubric_templates": templates},
            allow_unicode=True,
            sort_keys=False,
        )

    def to_markdown_templates(self, result: RubricsAnalysisResult) -> str:
        """Render structured templates as Markdown for documentation."""
        lines = []
        lines.append("# Rubric 模板库")
        lines.append("")
        lines.append("| 类别 | 动作 | 目标 | 条件 | 频次 | 示例 |")
        lines.append("|------|------|------|------|------|------|")

        for entry in result.structured_templates:
            action = entry.get("action") or "—"
            target = entry.get("target") or "—"
            condition = entry.get("condition") or "—"
            freq = entry.get("frequency", 0)
            examples = entry.get("examples", [])
            example_text = "<br/>".join(examples)
            lines.append(
                f"| {entry.get('category', 'general')} | {action} | {target} | {condition} | {freq} | {example_text} |"
            )

        return "\n".join(lines)
