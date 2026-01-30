"""Main analysis logic for dataset recipes."""

from pathlib import Path
from typing import Optional, Union

import yaml

from datarecipe.schema import Recipe, SourceType
from datarecipe.sources.huggingface import HuggingFaceExtractor


class DatasetAnalyzer:
    """Main analyzer for extracting dataset recipes."""

    def __init__(self):
        """Initialize the analyzer with source extractors."""
        self.extractors = {
            SourceType.HUGGINGFACE: HuggingFaceExtractor(),
        }

    def analyze(
        self,
        dataset_id: str,
        source_type: Optional[SourceType] = None,
    ) -> Recipe:
        """Analyze a dataset and extract its recipe.

        Args:
            dataset_id: The dataset identifier
            source_type: The source type (auto-detected if not provided)

        Returns:
            Recipe object with extracted information
        """
        # Auto-detect source type if not provided
        if source_type is None:
            source_type = self._detect_source_type(dataset_id)

        if source_type not in self.extractors:
            raise ValueError(f"Unsupported source type: {source_type}")

        extractor = self.extractors[source_type]
        return extractor.extract(dataset_id)

    def analyze_from_yaml(self, yaml_path: Union[str, Path]) -> Recipe:
        """Load a recipe from a YAML file.

        Args:
            yaml_path: Path to the YAML file

        Returns:
            Recipe object loaded from the file
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Recipe file not found: {yaml_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._recipe_from_dict(data)

    def export_recipe(self, recipe: Recipe, output_path: Union[str, Path]) -> None:
        """Export a recipe to a YAML file.

        Args:
            recipe: The recipe to export
            output_path: Path to write the YAML file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(recipe.to_yaml())

    def _detect_source_type(self, dataset_id: str) -> SourceType:
        """Auto-detect the source type from the dataset ID."""
        # Check if it looks like a HuggingFace dataset ID (org/name format)
        if "/" in dataset_id and not dataset_id.startswith("http"):
            return SourceType.HUGGINGFACE

        # Check for URLs
        if "huggingface.co" in dataset_id:
            return SourceType.HUGGINGFACE

        # Default to HuggingFace for now
        return SourceType.HUGGINGFACE

    def _recipe_from_dict(self, data: dict) -> Recipe:
        """Convert a dictionary to a Recipe object."""
        from datarecipe.schema import (
            Cost,
            Reproducibility,
            GenerationMethod,
            GenerationType,
        )

        recipe = Recipe(
            name=data.get("name", "unknown"),
            version=data.get("version"),
        )

        # Source info
        if "source" in data:
            source = data["source"]
            source_type_str = source.get("type", "unknown")
            try:
                recipe.source_type = SourceType(source_type_str)
            except ValueError:
                recipe.source_type = SourceType.UNKNOWN
            recipe.source_id = source.get("id")

        # Generation info
        if "generation" in data:
            gen = data["generation"]
            recipe.synthetic_ratio = gen.get("synthetic_ratio")
            recipe.human_ratio = gen.get("human_ratio")
            recipe.teacher_models = gen.get("teacher_models", [])

            if recipe.synthetic_ratio is not None:
                if recipe.synthetic_ratio >= 0.9:
                    recipe.generation_type = GenerationType.SYNTHETIC
                elif recipe.synthetic_ratio <= 0.1:
                    recipe.generation_type = GenerationType.HUMAN
                else:
                    recipe.generation_type = GenerationType.MIXED

            if "methods" in gen:
                for method_data in gen["methods"]:
                    method = GenerationMethod(
                        method_type=method_data.get("type", "unknown"),
                        teacher_model=method_data.get("teacher_model"),
                        prompt_template_available=method_data.get("prompt_template") == "available",
                        platform=method_data.get("platform"),
                    )
                    recipe.generation_methods.append(method)

        # Cost info
        if "cost" in data:
            cost_data = data["cost"]
            breakdown = cost_data.get("breakdown", {})
            recipe.cost = Cost(
                estimated_total_usd=cost_data.get("estimated_total_usd"),
                api_calls_usd=breakdown.get("api_calls"),
                human_annotation_usd=breakdown.get("human_annotation"),
                compute_usd=breakdown.get("compute"),
                confidence=cost_data.get("confidence", "low"),
            )

        # Reproducibility info
        if "reproducibility" in data:
            repro_data = data["reproducibility"]
            recipe.reproducibility = Reproducibility(
                score=repro_data.get("score", 5),
                available=repro_data.get("available", []),
                missing=repro_data.get("missing", []),
                notes=repro_data.get("notes"),
            )

        # Metadata
        if "metadata" in data:
            meta = data["metadata"]
            recipe.size = meta.get("size_bytes")
            recipe.num_examples = meta.get("num_examples")
            recipe.languages = meta.get("languages", [])
            recipe.license = meta.get("license")
            recipe.tags = meta.get("tags", [])
            recipe.authors = meta.get("authors", [])
            recipe.paper_url = meta.get("paper_url")

        return recipe


def get_recipe_summary(recipe: Recipe) -> dict:
    """Get a summary of the recipe for display.

    Args:
        recipe: The recipe to summarize

    Returns:
        Dictionary with summary information
    """
    summary = {
        "name": recipe.name,
        "source": f"{recipe.source_type.value}: {recipe.source_id}",
    }

    # Generation summary
    if recipe.synthetic_ratio is not None:
        summary["synthetic_percentage"] = f"{recipe.synthetic_ratio * 100:.0f}%"
    if recipe.human_ratio is not None:
        summary["human_percentage"] = f"{recipe.human_ratio * 100:.0f}%"

    if recipe.teacher_models:
        summary["teacher_models"] = recipe.teacher_models

    # Cost summary
    if recipe.cost and recipe.cost.estimated_total_usd:
        summary["estimated_cost"] = f"${recipe.cost.estimated_total_usd:,.0f}"
        summary["cost_confidence"] = recipe.cost.confidence

    # Reproducibility summary
    if recipe.reproducibility:
        summary["reproducibility_score"] = f"{recipe.reproducibility.score}/10"
        if recipe.reproducibility.missing:
            summary["missing_for_reproduction"] = recipe.reproducibility.missing

    return summary
