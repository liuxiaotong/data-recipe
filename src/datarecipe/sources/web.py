"""Extract recipe information from web URLs."""

import re
from typing import Optional

import requests

from datarecipe.schema import (
    Cost,
    GenerationMethod,
    GenerationType,
    Recipe,
    Reproducibility,
    SourceType,
)


class WebExtractor:
    """Extract dataset recipe information from arbitrary web URLs."""

    def __init__(self):
        """Initialize the extractor."""
        pass

    def extract(self, url: str) -> Recipe:
        """Extract recipe information from a web URL.

        Args:
            url: The URL of the dataset page

        Returns:
            Recipe object with extracted information
        """
        # Fetch page content
        content = self._fetch_page(url)
        if not content:
            raise ValueError(f"Could not fetch content from: {url}")

        # Extract title
        title = self._extract_title(content) or url

        # Build recipe
        recipe = Recipe(
            name=title,
            source_type=SourceType.WEB,
            source_id=url,
            description=self._extract_description(content),
            homepage_url=url,
        )

        # Analyze content
        recipe.teacher_models = self._detect_teacher_models(content)
        recipe.generation_type, recipe.synthetic_ratio, recipe.human_ratio = (
            self._detect_generation_type(content)
        )
        recipe.generation_methods = self._build_generation_methods(
            recipe.teacher_models, recipe.generation_type, content
        )

        # Extract additional info
        recipe.num_examples = self._extract_dataset_size(content)
        recipe.tags = self._extract_tags(content)

        # Assess reproducibility
        recipe.reproducibility = self._assess_reproducibility(content)

        # Estimate cost
        recipe.cost = self._estimate_cost(recipe)

        return recipe

    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch and parse web page content."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; DataRecipe/1.0)"}
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.text
        except Exception:
            pass
        return None

    def _extract_title(self, content: str) -> Optional[str]:
        """Extract page title."""
        # Try <title> tag
        match = re.search(r"<title>([^<]+)</title>", content, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean up common suffixes
            for suffix in [" - GitHub", " | Hugging Face", " - Papers With Code"]:
                if title.endswith(suffix):
                    title = title[: -len(suffix)]
            return title

        # Try <h1> tag
        match = re.search(r"<h1[^>]*>([^<]+)</h1>", content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    def _extract_description(self, content: str) -> Optional[str]:
        """Extract page description."""
        # Try meta description
        match = re.search(
            r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\']',
            content,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Try og:description
        match = re.search(
            r'<meta\s+property=["\']og:description["\']\s+content=["\']([^"\']+)["\']',
            content,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        return None

    def _detect_teacher_models(self, content: str) -> list[str]:
        """Detect teacher models from content."""
        patterns = [
            (r"gpt-?5\.?2", "GPT-5.2"),
            (r"gpt-?5", "GPT-5"),
            (r"gpt-?4", "GPT-4"),
            (r"gpt-?3\.?5", "GPT-3.5"),
            (r"claude[-\s]?4\.?5", "Claude 4.5"),
            (r"claude[-\s]?4", "Claude 4"),
            (r"claude[-\s]?3", "Claude 3"),
            (r"llama[-\s]?4", "Llama 4"),
            (r"llama[-\s]?3", "Llama 3"),
            (r"gemini[-\s]?2", "Gemini 2"),
            (r"gemini", "Gemini"),
            (r"mistral", "Mistral"),
            (r"qwen", "Qwen"),
            (r"deepseek", "DeepSeek"),
        ]

        found = set()
        content_lower = content.lower()
        for pattern, name in patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                found.add(name)
        return sorted(found)

    def _detect_generation_type(self, content: str) -> tuple:
        """Detect generation type from content."""
        content_lower = content.lower()

        synthetic_keywords = [
            "synthetic",
            "generated",
            "distill",
            "teacher",
            "api",
            "llm-generated",
            "machine-generated",
            "ai-generated",
        ]
        human_keywords = [
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

        synthetic_score = sum(1 for kw in synthetic_keywords if kw in content_lower)
        human_score = sum(1 for kw in human_keywords if kw in content_lower)

        total = synthetic_score + human_score
        if total == 0:
            return GenerationType.UNKNOWN, None, None

        if synthetic_score > 0 and human_score > 0:
            ratio = synthetic_score / total
            return GenerationType.MIXED, round(ratio, 2), round(1 - ratio, 2)
        elif synthetic_score > human_score:
            return GenerationType.SYNTHETIC, 1.0, 0.0
        else:
            return GenerationType.HUMAN, 0.0, 1.0

    def _build_generation_methods(self, teacher_models, generation_type, content) -> list:
        """Build generation methods list."""
        methods = []

        if teacher_models:
            for model in teacher_models:
                methods.append(
                    GenerationMethod(
                        method_type="distillation",
                        teacher_model=model,
                    )
                )

        if generation_type in [GenerationType.HUMAN, GenerationType.MIXED]:
            methods.append(
                GenerationMethod(
                    method_type="human_annotation",
                )
            )

        if not methods and generation_type == GenerationType.UNKNOWN:
            methods.append(
                GenerationMethod(
                    method_type="unknown",
                )
            )

        return methods

    def _extract_dataset_size(self, content: str) -> Optional[int]:
        """Try to extract dataset size from content."""
        # Look for common patterns like "100,000 examples" or "100k samples"
        patterns = [
            r"(\d{1,3}(?:,\d{3})+)\s*(?:examples?|samples?|rows?|instances?)",
            r"(\d+(?:\.\d+)?)\s*[kKmM]\s*(?:examples?|samples?|rows?|instances?)",
            r"(\d+)\s*(?:examples?|samples?|rows?|instances?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                num_str = match.group(1)
                # Handle k/m suffixes
                if "k" in content[match.start() : match.end()].lower():
                    return int(float(num_str.replace(",", "")) * 1000)
                elif "m" in content[match.start() : match.end()].lower():
                    return int(float(num_str.replace(",", "")) * 1000000)
                else:
                    return int(num_str.replace(",", ""))

        return None

    def _extract_tags(self, content: str) -> list[str]:
        """Extract relevant tags from content."""
        tags = []

        tag_keywords = {
            "nlp": ["nlp", "natural language", "text"],
            "vision": ["vision", "image", "visual"],
            "reasoning": ["reasoning", "logic"],
            "math": ["math", "arithmetic", "calculation"],
            "code": ["code", "programming", "coding"],
            "qa": ["question answering", "qa", "q&a"],
            "dialogue": ["dialogue", "conversation", "chat"],
            "benchmark": ["benchmark", "evaluation", "eval"],
        }

        content_lower = content.lower()
        for tag, keywords in tag_keywords.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)

        return tags

    def _assess_reproducibility(self, content: str) -> Reproducibility:
        """Assess reproducibility from content."""
        available = []
        missing = []
        score = 3  # Base score for having a web page

        content_lower = content.lower()

        if "github" in content_lower:
            available.append("source_code_reference")
            score += 2
        else:
            missing.append("source_code")

        if "paper" in content_lower or "arxiv" in content_lower:
            available.append("paper_reference")
            score += 1

        if "download" in content_lower:
            available.append("download_available")
            score += 1

        if "license" in content_lower:
            available.append("license_info")
            score += 1

        missing.append("exact_prompts")
        missing.append("filtering_criteria")
        missing.append("quality_thresholds")

        return Reproducibility(
            score=min(10, max(1, score)),
            available=available,
            missing=missing,
        )

    def _estimate_cost(self, recipe: Recipe) -> Cost:
        """Estimate cost based on available information."""
        cost = Cost(confidence="low")

        if recipe.teacher_models:
            cost.api_calls_usd = 10000

        if recipe.generation_type in [GenerationType.HUMAN, GenerationType.MIXED]:
            cost.human_annotation_usd = 25000

        if cost.api_calls_usd or cost.human_annotation_usd:
            cost.estimated_total_usd = (cost.api_calls_usd or 0) + (cost.human_annotation_usd or 0)

        return cost
