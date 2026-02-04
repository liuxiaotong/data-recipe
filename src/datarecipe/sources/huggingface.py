"""Extract recipe information from HuggingFace datasets."""

import re
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

from datarecipe.schema import (
    Recipe,
    Cost,
    Reproducibility,
    GenerationMethod,
    GenerationType,
    SourceType,
)


# Known teacher models and their patterns
TEACHER_MODEL_PATTERNS = [
    (r"gpt-?5\.?2", "GPT-5.2"),
    (r"gpt-?5", "GPT-5"),
    (r"gpt-?4", "GPT-4"),
    (r"gpt-?3\.?5", "GPT-3.5"),
    (r"claude[-\s]?4\.?5", "Claude 4.5"),
    (r"claude[-\s]?4", "Claude 4"),
    (r"claude[-\s]?3", "Claude 3"),
    (r"claude", "Claude"),
    (r"llama[-\s]?4", "Llama 4"),
    (r"llama[-\s]?3", "Llama 3"),
    (r"gemini[-\s]?2", "Gemini 2"),
    (r"gemini", "Gemini"),
    (r"mistral", "Mistral"),
    (r"mixtral", "Mixtral"),
    (r"qwen[-\s]?3", "Qwen 3"),
    (r"qwen", "Qwen"),
    (r"deepseek[-\s]?v3", "DeepSeek V3"),
    (r"deepseek", "DeepSeek"),
]

# Keywords indicating synthetic data
SYNTHETIC_KEYWORDS = [
    "synthetic",
    "generated",
    "distill",
    "teacher",
    "api",
    "llm-generated",
    "machine-generated",
    "ai-generated",
]

# Keywords indicating human annotation
HUMAN_KEYWORDS = [
    "human",
    "annotated",
    "crowdsource",
    "manual",
    "expert",
    "mturk",
    "mechanical turk",
    "scale ai",
    "labelbox",
]


class HuggingFaceExtractor:
    """Extract dataset recipe information from HuggingFace Hub."""

    def __init__(self):
        """Initialize the extractor."""
        self.api = HfApi()

    def extract(self, dataset_id: str) -> Recipe:
        """Extract recipe information from a HuggingFace dataset.

        Args:
            dataset_id: The dataset ID (e.g., 'allenai/Sera-4.6-Lite-T2')

        Returns:
            Recipe object with extracted information
        """
        try:
            info = self.api.dataset_info(dataset_id)
        except RepositoryNotFoundError:
            raise ValueError(f"Dataset not found: {dataset_id}")

        # Basic info
        recipe = Recipe(
            name=dataset_id,
            source_type=SourceType.HUGGINGFACE,
            source_id=dataset_id,
            description=info.description,
            license=info.license if hasattr(info, "license") else None,
            tags=list(info.tags) if info.tags else [],
            authors=[info.author] if info.author else [],
        )

        # Try to get README content for deeper analysis
        readme_content = self._get_readme(dataset_id)

        # Detect teacher models
        recipe.teacher_models = self._detect_teacher_models(info, readme_content)

        # Detect generation type
        recipe.generation_type, recipe.synthetic_ratio, recipe.human_ratio = (
            self._detect_generation_type(info, readme_content)
        )

        # Build generation methods
        recipe.generation_methods = self._build_generation_methods(
            recipe.teacher_models, recipe.generation_type, readme_content
        )

        # Estimate cost
        recipe.cost = self._estimate_cost(info, recipe)

        # Assess reproducibility
        recipe.reproducibility = self._assess_reproducibility(info, readme_content, recipe)

        # Extract additional metadata
        if hasattr(info, "card_data") and info.card_data:
            card = info.card_data
            if hasattr(card, "language"):
                recipe.languages = (
                    card.language if isinstance(card.language, list) else [card.language]
                )

        return recipe

    def _get_readme(self, dataset_id: str) -> Optional[str]:
        """Try to fetch and return README content."""
        try:
            readme_path = hf_hub_download(
                repo_id=dataset_id, filename="README.md", repo_type="dataset"
            )
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
        except (EntryNotFoundError, Exception):
            return None

    def _detect_teacher_models(self, info, readme_content: Optional[str]) -> list[str]:
        """Detect which teacher models were used."""
        found_models = set()

        # Combine all text sources efficiently using list + join
        text_parts = []
        if info.description:
            text_parts.append(info.description.lower())
        if readme_content:
            text_parts.append(readme_content.lower())
        if info.tags:
            text_parts.append(" ".join(info.tags).lower())
        text_to_search = " ".join(text_parts)

        # Search for known models
        for pattern, model_name in TEACHER_MODEL_PATTERNS:
            if re.search(pattern, text_to_search, re.IGNORECASE):
                found_models.add(model_name)

        return sorted(list(found_models))

    def _detect_generation_type(
        self, info, readme_content: Optional[str]
    ) -> tuple[GenerationType, Optional[float], Optional[float]]:
        """Detect whether data is synthetic, human, or mixed."""
        text_to_search = ""

        if info.description:
            text_to_search += info.description.lower() + " "
        if readme_content:
            text_to_search += readme_content.lower() + " "
        if info.tags:
            text_to_search += " ".join(info.tags).lower() + " "

        synthetic_score = sum(1 for kw in SYNTHETIC_KEYWORDS if kw in text_to_search)
        human_score = sum(1 for kw in HUMAN_KEYWORDS if kw in text_to_search)

        total = synthetic_score + human_score
        if total == 0:
            return GenerationType.UNKNOWN, None, None

        if synthetic_score > 0 and human_score > 0:
            # Mixed - estimate ratios
            synthetic_ratio = synthetic_score / total
            return GenerationType.MIXED, round(synthetic_ratio, 2), round(1 - synthetic_ratio, 2)
        elif synthetic_score > human_score:
            return GenerationType.SYNTHETIC, 1.0, 0.0
        else:
            return GenerationType.HUMAN, 0.0, 1.0

    def _build_generation_methods(
        self,
        teacher_models: list[str],
        generation_type: GenerationType,
        readme_content: Optional[str],
    ) -> list[GenerationMethod]:
        """Build list of generation methods based on detected info."""
        methods = []

        if teacher_models:
            for model in teacher_models:
                methods.append(
                    GenerationMethod(
                        method_type="distillation",
                        teacher_model=model,
                        prompt_template_available=self._has_prompt_template(readme_content),
                    )
                )

        if generation_type == GenerationType.HUMAN or generation_type == GenerationType.MIXED:
            methods.append(
                GenerationMethod(
                    method_type="human_annotation",
                    platform=self._detect_annotation_platform(readme_content),
                )
            )

        return methods

    def _has_prompt_template(self, readme_content: Optional[str]) -> bool:
        """Check if prompt templates are documented."""
        if not readme_content:
            return False
        keywords = ["prompt", "template", "instruction", "system message"]
        return any(kw in readme_content.lower() for kw in keywords)

    def _detect_annotation_platform(self, readme_content: Optional[str]) -> Optional[str]:
        """Detect annotation platform if mentioned."""
        if not readme_content:
            return None

        platforms = {
            "scale ai": "Scale AI",
            "mturk": "Amazon MTurk",
            "mechanical turk": "Amazon MTurk",
            "labelbox": "Labelbox",
            "prolific": "Prolific",
            "appen": "Appen",
            "surge": "Surge AI",
        }

        content_lower = readme_content.lower()
        for keyword, platform in platforms.items():
            if keyword in content_lower:
                return platform
        return None

    def _estimate_cost(self, info, recipe: Recipe) -> Cost:
        """Estimate the cost of creating this dataset."""
        # This is a rough estimation based on heuristics
        cost = Cost()

        # Try to estimate based on dataset size
        # Note: This is very rough and should be improved with actual data

        if recipe.teacher_models:
            # Assume some API usage
            # Rough estimate: $0.01 per 1K tokens, assume 500 tokens per example
            # This is just a placeholder estimation
            cost.api_calls_usd = 10000  # Placeholder
            cost.confidence = "low"

        if recipe.generation_type in [GenerationType.HUMAN, GenerationType.MIXED]:
            # Human annotation is expensive
            cost.human_annotation_usd = 25000  # Placeholder
            cost.confidence = "low"

        if cost.api_calls_usd or cost.human_annotation_usd:
            cost.estimated_total_usd = (cost.api_calls_usd or 0) + (cost.human_annotation_usd or 0)

        return cost

    def _assess_reproducibility(
        self, info, readme_content: Optional[str], recipe: Recipe
    ) -> Reproducibility:
        """Assess how reproducible this dataset is."""
        available = []
        missing = []
        score = 5  # Start at middle

        # Check what's available
        if info.description:
            available.append("description")
            score += 1

        if readme_content and len(readme_content) > 500:
            available.append("detailed_documentation")
            score += 1

        if recipe.teacher_models:
            available.append("teacher_model_names")
            score += 1
        else:
            missing.append("teacher_model_info")

        if any(m.prompt_template_available for m in recipe.generation_methods):
            available.append("prompt_templates")
            score += 1
        else:
            missing.append("exact_prompts")

        # Check for code/scripts
        if readme_content and ("github" in readme_content.lower() or "code" in readme_content.lower()):
            available.append("source_code_reference")
            score += 1
        else:
            missing.append("generation_scripts")

        # Common missing items
        missing.append("filtering_criteria")
        missing.append("quality_thresholds")

        # Cap score
        score = min(10, max(1, score))

        return Reproducibility(score=score, available=available, missing=missing)
