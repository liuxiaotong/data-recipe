"""Extract recipe information from GitHub repositories."""

import re
from typing import Optional

import requests

from datarecipe.schema import (
    Recipe,
    Cost,
    Reproducibility,
    GenerationMethod,
    GenerationType,
    SourceType,
)


class GitHubExtractor:
    """Extract dataset recipe information from GitHub repositories."""

    def __init__(self):
        """Initialize the extractor."""
        self.api_base = "https://api.github.com"

    def extract(self, repo_id: str) -> Recipe:
        """Extract recipe information from a GitHub repository.

        Args:
            repo_id: The repository ID (e.g., 'arcprize/ARC-AGI-2')

        Returns:
            Recipe object with extracted information
        """
        # Fetch repo info
        repo_info = self._fetch_repo_info(repo_id)
        if not repo_info:
            raise ValueError(f"Repository not found: {repo_id}")

        # Fetch README
        readme_content = self._fetch_readme(repo_id)

        # Build recipe
        recipe = Recipe(
            name=repo_info.get("name", repo_id),
            source_type=SourceType.HUGGINGFACE,  # Using as generic, will show as github
            source_id=repo_id,
            description=repo_info.get("description"),
            license=self._extract_license(repo_info),
            tags=repo_info.get("topics", []),
            homepage_url=repo_info.get("html_url"),
        )

        # Analyze README for generation info
        if readme_content:
            recipe.teacher_models = self._detect_teacher_models(readme_content)
            recipe.generation_type, recipe.synthetic_ratio, recipe.human_ratio = (
                self._detect_generation_type(readme_content)
            )
            recipe.generation_methods = self._build_generation_methods(
                recipe.teacher_models, recipe.generation_type, readme_content
            )

        # Assess reproducibility
        recipe.reproducibility = self._assess_reproducibility(repo_info, readme_content)

        # Estimate cost (placeholder)
        recipe.cost = Cost(
            estimated_total_usd=25000,
            confidence="low",
        )

        return recipe

    def _fetch_repo_info(self, repo_id: str) -> Optional[dict]:
        """Fetch repository information from GitHub API."""
        try:
            url = f"{self.api_base}/repos/{repo_id}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def _fetch_readme(self, repo_id: str) -> Optional[str]:
        """Fetch README content from repository."""
        try:
            url = f"{self.api_base}/repos/{repo_id}/readme"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Decode base64 content
                import base64
                content = base64.b64decode(data.get("content", "")).decode("utf-8")
                return content
        except Exception:
            pass
        return None

    def _extract_license(self, repo_info: dict) -> Optional[str]:
        """Extract license from repo info."""
        license_info = repo_info.get("license")
        if license_info:
            return license_info.get("spdx_id") or license_info.get("name")
        return None

    def _detect_teacher_models(self, readme_content: str) -> list[str]:
        """Detect teacher models from README."""
        patterns = [
            (r"gpt-?5\.?2", "GPT-5.2"),
            (r"gpt-?5", "GPT-5"),
            (r"gpt-?4", "GPT-4"),
            (r"claude[-\s]?4\.?5", "Claude 4.5"),
            (r"claude[-\s]?4", "Claude 4"),
            (r"claude[-\s]?3", "Claude 3"),
            (r"llama[-\s]?4", "Llama 4"),
            (r"llama[-\s]?3", "Llama 3"),
            (r"gemini", "Gemini"),
        ]

        found = set()
        content_lower = readme_content.lower()
        for pattern, name in patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                found.add(name)
        return sorted(list(found))

    def _detect_generation_type(self, readme_content: str) -> tuple:
        """Detect generation type from README."""
        content_lower = readme_content.lower()

        synthetic_keywords = ["synthetic", "generated", "distill", "api", "llm-generated"]
        human_keywords = ["human", "annotated", "crowdsource", "manual", "expert"]

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

    def _build_generation_methods(self, teacher_models, generation_type, readme_content) -> list:
        """Build generation methods list."""
        methods = []

        if teacher_models:
            for model in teacher_models:
                methods.append(GenerationMethod(
                    method_type="distillation",
                    teacher_model=model,
                ))

        if generation_type in [GenerationType.HUMAN, GenerationType.MIXED]:
            methods.append(GenerationMethod(
                method_type="human_annotation",
            ))

        return methods

    def _assess_reproducibility(self, repo_info: dict, readme_content: Optional[str]) -> Reproducibility:
        """Assess reproducibility of the dataset."""
        available = []
        missing = []
        score = 5

        if repo_info.get("description"):
            available.append("description")
            score += 1

        if readme_content and len(readme_content) > 500:
            available.append("detailed_documentation")
            score += 1

        if repo_info.get("license"):
            available.append("license_info")
            score += 1

        # GitHub repos usually have code
        available.append("source_code")
        score += 1

        missing.append("exact_prompts")
        missing.append("filtering_criteria")

        return Reproducibility(
            score=min(10, score),
            available=available,
            missing=missing,
        )
