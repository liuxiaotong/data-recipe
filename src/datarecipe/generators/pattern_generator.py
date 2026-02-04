"""
Pattern-Based Data Generator

Generates new data based on discovered patterns from reverse engineering.
"""

import re
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime

from ..extractors.rubrics_analyzer import RubricsAnalysisResult, RubricPattern
from ..extractors.prompt_extractor import PromptLibrary, PromptTemplate


@dataclass
class GeneratedDataItem:
    """A single generated data item."""

    id: str
    content: str
    data_type: str              # "rubric", "prompt", "context", etc.
    template_used: str
    parameters: dict = field(default_factory=dict)
    quality_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.md5(
                f"{self.content}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]


@dataclass
class GenerationResult:
    """Result of a generation run."""

    items: list[GeneratedDataItem] = field(default_factory=list)
    total_generated: int = 0
    templates_used: int = 0
    generation_time: str = ""

    # Quality metrics
    avg_quality_score: float = 0.0
    unique_ratio: float = 0.0

    def summary(self) -> str:
        """Generate summary."""
        return (
            f"Generated: {self.total_generated} items\n"
            f"Templates Used: {self.templates_used}\n"
            f"Avg Quality: {self.avg_quality_score:.2f}\n"
            f"Unique Ratio: {self.unique_ratio:.1%}"
        )


class PatternGenerator:
    """
    Generates new data based on discovered patterns.

    Example usage:
        generator = PatternGenerator()

        # Generate rubrics
        result = generator.generate_rubrics(
            rubrics_analysis=analysis_result,
            context="game rules",
            count=20
        )

        # Generate prompts
        result = generator.generate_prompts(
            prompt_library=library,
            domain="legal",
            count=10
        )
    """

    # Rubric verb templates
    RUBRIC_TEMPLATES = {
        "define": [
            "The response should define what {topic} is",
            "The response should explain the meaning of {topic}",
            "The response should clarify the definition of {topic}",
        ],
        "list": [
            "The response should list all {items}",
            "The response should enumerate the {items}",
            "The response should include all relevant {items}",
        ],
        "explain": [
            "The response should explain why {reason}",
            "The response should describe how {process}",
            "The response should clarify the relationship between {elements}",
        ],
        "avoid": [
            "The response should not assume {assumption}",
            "The response should not include {excluded}",
            "The response should avoid {prohibited}",
        ],
        "verify": [
            "The response should verify that {condition}",
            "The response should confirm {statement}",
            "The response should check whether {check}",
        ],
        "format": [
            "The response should be formatted as {format}",
            "The response should present {content} in {format} format",
            "The response should organize {content} clearly",
        ],
    }

    # Context templates by strategy
    CONTEXT_TEMPLATES = {
        "synthetic": {
            "game_rules": """
# {game_name} Rules

## Overview
{game_name} is a {game_type} game for {player_count} players.

## Core Mechanics
{mechanics}

## Victory Conditions
{victory_conditions}

## Special Rules
{special_rules}
""",
            "procedure": """
# {procedure_name} Procedure

## Purpose
{purpose}

## Prerequisites
{prerequisites}

## Steps
{steps}

## Exceptions
{exceptions}
""",
        },
        "modified": {
            "technical_doc": """
# {topic} Documentation

## Overview
{overview}

## Key Concepts
{concepts}

## Implementation Details
{details}

## Common Issues
{issues}
""",
        },
    }

    def __init__(self):
        self.generated_hashes: set[str] = set()

    def generate_rubrics(
        self,
        rubrics_analysis: Optional[RubricsAnalysisResult] = None,
        context: str = "the topic",
        count: int = 10,
        categories: Optional[list[str]] = None,
        custom_templates: Optional[dict[str, list[str]]] = None,
    ) -> GenerationResult:
        """
        Generate rubrics based on discovered patterns.

        Args:
            rubrics_analysis: Analysis result to base generation on
            context: Context/topic for the rubrics
            count: Number of rubrics to generate
            categories: Specific categories to generate (default: all)
            custom_templates: Custom templates to use

        Returns:
            GenerationResult with generated rubrics
        """
        result = GenerationResult(generation_time=datetime.now().isoformat())

        # Merge templates
        templates = dict(self.RUBRIC_TEMPLATES)
        if custom_templates:
            templates.update(custom_templates)

        # Use patterns from analysis if available
        if rubrics_analysis and rubrics_analysis.top_templates:
            templates["discovered"] = rubrics_analysis.top_templates[:10]

        # Select categories
        if categories:
            templates = {k: v for k, v in templates.items() if k in categories}

        if not templates:
            return result

        # Generate rubrics
        generated = []
        templates_used = set()

        # Distribute count across categories
        cats = list(templates.keys())
        per_category = max(1, count // len(cats))

        for category, category_templates in templates.items():
            for template in category_templates[:per_category]:
                # Fill template
                rubric = self._fill_rubric_template(template, context, category)

                # Check uniqueness
                content_hash = hashlib.md5(rubric.lower().encode()).hexdigest()
                if content_hash in self.generated_hashes:
                    continue
                self.generated_hashes.add(content_hash)

                item = GeneratedDataItem(
                    id="",
                    content=rubric,
                    data_type="rubric",
                    template_used=template,
                    parameters={"context": context, "category": category},
                    quality_score=self._score_rubric(rubric),
                )
                generated.append(item)
                templates_used.add(template)

                if len(generated) >= count:
                    break

            if len(generated) >= count:
                break

        result.items = generated
        result.total_generated = len(generated)
        result.templates_used = len(templates_used)

        # Calculate metrics
        if generated:
            result.avg_quality_score = sum(
                i.quality_score for i in generated
            ) / len(generated)
            result.unique_ratio = len(generated) / max(count, len(generated))

        return result

    def generate_prompts(
        self,
        prompt_library: Optional[PromptLibrary] = None,
        domain: str = "general",
        category: str = "system",
        count: int = 5,
        customize_fn: Optional[Callable[[str], str]] = None,
    ) -> GenerationResult:
        """
        Generate prompts based on extracted templates.

        Args:
            prompt_library: Library to base generation on
            domain: Target domain
            category: Prompt category
            count: Number of prompts to generate
            customize_fn: Function to customize generated prompts

        Returns:
            GenerationResult with generated prompts
        """
        result = GenerationResult(generation_time=datetime.now().isoformat())

        if not prompt_library or not prompt_library.templates:
            # Use default templates
            default_templates = self._get_default_prompt_templates(domain, category)
            source_templates = [
                PromptTemplate(content=t, category=category, domain=domain)
                for t in default_templates
            ]
        else:
            # Filter by category and domain
            source_templates = [
                t for t in prompt_library.templates
                if t.category == category and (t.domain == domain or t.domain == "general")
            ]

        if not source_templates:
            return result

        generated = []
        templates_used = set()

        for template in source_templates[:count]:
            content = template.content

            # Customize if function provided
            if customize_fn:
                content = customize_fn(content)

            # Replace domain-specific placeholders
            content = content.replace("{domain}", domain)
            content = content.replace("[DOMAIN]", domain)

            item = GeneratedDataItem(
                id="",
                content=content,
                data_type="prompt",
                template_used=template.content[:50],
                parameters={"domain": domain, "category": category},
                quality_score=self._score_prompt(content),
            )
            generated.append(item)
            templates_used.add(template.hash_id)

        result.items = generated
        result.total_generated = len(generated)
        result.templates_used = len(templates_used)

        if generated:
            result.avg_quality_score = sum(
                i.quality_score for i in generated
            ) / len(generated)

        return result

    def generate_contexts(
        self,
        strategy: str = "synthetic",
        template_type: str = "game_rules",
        parameters: Optional[dict] = None,
        count: int = 1,
    ) -> GenerationResult:
        """
        Generate context documents based on templates.

        Args:
            strategy: "synthetic" or "modified"
            template_type: Type of context to generate
            parameters: Parameters to fill in template
            count: Number of contexts to generate

        Returns:
            GenerationResult with generated contexts
        """
        result = GenerationResult(generation_time=datetime.now().isoformat())

        strategy_templates = self.CONTEXT_TEMPLATES.get(strategy, {})
        template = strategy_templates.get(template_type, "")

        if not template:
            return result

        params = parameters or {}
        generated = []

        for i in range(count):
            # Generate unique parameters if not all provided
            filled_params = self._generate_context_params(template_type, params, i)

            # Fill template
            content = template
            for key, value in filled_params.items():
                content = content.replace(f"{{{key}}}", str(value))

            item = GeneratedDataItem(
                id="",
                content=content,
                data_type="context",
                template_used=template_type,
                parameters=filled_params,
                quality_score=self._score_context(content),
            )
            generated.append(item)

        result.items = generated
        result.total_generated = len(generated)
        result.templates_used = 1

        if generated:
            result.avg_quality_score = sum(
                i.quality_score for i in generated
            ) / len(generated)

        return result

    def _fill_rubric_template(
        self,
        template: str,
        context: str,
        category: str
    ) -> str:
        """Fill a rubric template with context."""
        result = template

        # Generic replacements
        replacements = {
            "{topic}": context,
            "{items}": f"items related to {context}",
            "{reason}": f"{context} works this way",
            "{process}": f"the {context} process works",
            "{elements}": f"elements of {context}",
            "{assumption}": f"prior knowledge about {context}",
            "{excluded}": f"information not in the {context}",
            "{prohibited}": f"content outside the scope of {context}",
            "{condition}": f"the {context} requirements are met",
            "{statement}": f"the {context} is correctly understood",
            "{check}": f"all aspects of {context} are addressed",
            "{format}": "a clear, structured manner",
            "{content}": f"information about {context}",
        }

        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _get_default_prompt_templates(
        self,
        domain: str,
        category: str
    ) -> list[str]:
        """Get default prompt templates."""
        templates = {
            "system": [
                f"You are a helpful assistant specializing in {domain}.",
                f"You are an expert {domain} assistant. Provide accurate and helpful responses.",
                f"As a {domain} specialist, help users with their questions.",
            ],
            "task": [
                "Please analyze the following and provide your assessment.",
                "Based on the context provided, answer the question below.",
                "Review the information and give a detailed response.",
            ],
            "constraint": [
                "Do not make assumptions beyond what is stated.",
                "Only use information provided in the context.",
                "Be precise and avoid speculation.",
            ],
        }
        return templates.get(category, templates["system"])

    def _generate_context_params(
        self,
        template_type: str,
        provided: dict,
        index: int
    ) -> dict:
        """Generate parameters for context template."""
        defaults = {
            "game_rules": {
                "game_name": f"Stellar Quest {index + 1}",
                "game_type": "strategy",
                "player_count": "2-4",
                "mechanics": "- Resource collection\n- Territory control\n- Card drafting",
                "victory_conditions": "First player to reach 100 points wins.",
                "special_rules": "- Wild cards can substitute any resource\n- Bonus points for completing sets",
            },
            "procedure": {
                "procedure_name": f"Process {index + 1}",
                "purpose": "To ensure consistent and quality outcomes.",
                "prerequisites": "- Required training completed\n- Access permissions granted",
                "steps": "1. Initialize\n2. Execute\n3. Verify\n4. Document",
                "exceptions": "If step 2 fails, retry up to 3 times.",
            },
            "technical_doc": {
                "topic": f"System Component {index + 1}",
                "overview": "This component handles core functionality.",
                "concepts": "- Concept A\n- Concept B\n- Concept C",
                "details": "Implementation follows standard patterns.",
                "issues": "- Issue 1: Solution\n- Issue 2: Workaround",
            },
        }

        base = defaults.get(template_type, {})
        base.update(provided)
        return base

    def _score_rubric(self, rubric: str) -> float:
        """Score rubric quality (0-1)."""
        score = 0.5  # Base score

        # Check for standard structure
        if rubric.lower().startswith("the response should"):
            score += 0.2

        # Check for specificity
        if len(rubric) > 50:
            score += 0.1

        # Check for action verb
        action_verbs = ["define", "list", "explain", "include", "verify", "not"]
        if any(v in rubric.lower() for v in action_verbs):
            score += 0.1

        # Penalty for vagueness
        vague_words = ["maybe", "possibly", "might", "could"]
        if any(v in rubric.lower() for v in vague_words):
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _score_prompt(self, prompt: str) -> float:
        """Score prompt quality (0-1)."""
        score = 0.5

        # Length check
        if 50 < len(prompt) < 500:
            score += 0.2

        # Has clear instruction
        if any(w in prompt.lower() for w in ["you are", "please", "help"]):
            score += 0.15

        # Has structure
        if "\n" in prompt:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_context(self, context: str) -> float:
        """Score context quality (0-1)."""
        score = 0.5

        # Length check
        if len(context) > 500:
            score += 0.2

        # Has headers
        if "#" in context:
            score += 0.15

        # Has lists
        if "-" in context or "1." in context:
            score += 0.1

        return max(0.0, min(1.0, score))

    def export_jsonl(self, result: GenerationResult, filepath: str) -> None:
        """Export generated items to JSONL file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for item in result.items:
                line = json.dumps({
                    "id": item.id,
                    "content": item.content,
                    "type": item.data_type,
                    "template": item.template_used,
                    "parameters": item.parameters,
                    "quality_score": item.quality_score,
                    "metadata": item.metadata,
                }, ensure_ascii=False)
                f.write(line + "\n")

    def to_dict(self, result: GenerationResult) -> dict:
        """Convert result to dictionary."""
        return {
            "total_generated": result.total_generated,
            "templates_used": result.templates_used,
            "generation_time": result.generation_time,
            "avg_quality_score": result.avg_quality_score,
            "unique_ratio": result.unique_ratio,
            "items": [
                {
                    "id": i.id,
                    "content": i.content,
                    "type": i.data_type,
                    "quality_score": i.quality_score,
                }
                for i in result.items
            ],
        }
