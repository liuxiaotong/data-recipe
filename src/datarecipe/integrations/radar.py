"""Integration with ai-dataset-radar project."""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RadarDataset:
    """Dataset information from ai-dataset-radar."""

    id: str  # e.g., "Anthropic/hh-rlhf"
    category: str = ""  # e.g., "preference", "sft", "synthetic"
    downloads: int = 0
    signals: list[str] = field(default_factory=list)
    # Additional metadata from radar
    source: str = "huggingface"  # huggingface, github, etc.
    discovered_date: str = ""
    org: str = ""

    @classmethod
    def from_radar_json(cls, data: dict) -> "RadarDataset":
        """Create from radar JSON format."""
        dataset_id = data.get("id", "")
        org = dataset_id.split("/")[0] if "/" in dataset_id else ""

        return cls(
            id=dataset_id,
            category=data.get("category", ""),
            downloads=data.get("downloads", 0),
            signals=data.get("signals", []),
            source=data.get("source", "huggingface"),
            discovered_date=data.get("discovered_date", ""),
            org=org,
        )


@dataclass
class RecipeSummary:
    """Standardized output format for recipe analysis results.

    This format is designed to be consumed by ai-dataset-radar
    for indexing and trend analysis.
    """

    # Identification
    dataset_id: str
    analysis_date: str = ""
    analysis_version: str = "1.0"

    # Dataset classification
    dataset_type: str = ""  # instruction_tuning, preference, swe_bench, etc.
    category: str = ""  # From radar or detected
    purpose: str = ""

    # Cost estimation
    reproduction_cost: dict[str, float] = field(default_factory=dict)
    # {"human": 5000, "api": 200, "total": 5200}

    # Complexity assessment
    difficulty: str = ""  # easy, medium, hard
    human_percentage: float = 0.0
    machine_percentage: float = 0.0

    # Key patterns discovered
    key_patterns: list[str] = field(default_factory=list)
    rubric_patterns: int = 0
    prompt_templates: int = 0

    # Schema info
    fields: list[str] = field(default_factory=list)
    sample_count: int = 0

    # Similar datasets for reference
    similar_datasets: list[str] = field(default_factory=list)

    # File paths for detailed reports
    report_path: str = ""
    guide_path: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RecipeSummary":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RadarIntegration:
    """Handle integration between data-recipe and ai-dataset-radar."""

    def __init__(self):
        self.datasets: list[RadarDataset] = []

    def load_radar_report(self, report_path: str) -> list[RadarDataset]:
        """Load datasets from a radar report JSON file.

        Args:
            report_path: Path to intel_report_YYYY-MM-DD.json

        Returns:
            List of RadarDataset objects
        """
        path = Path(report_path)
        if not path.exists():
            raise FileNotFoundError(f"Radar report not found: {report_path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.datasets = []

        # Parse datasets array
        for ds_data in data.get("datasets", []):
            dataset = RadarDataset.from_radar_json(ds_data)
            self.datasets.append(dataset)

        return self.datasets

    def filter_datasets(
        self,
        orgs: list[str] | None = None,
        categories: list[str] | None = None,
        min_downloads: int = 0,
        signals: list[str] | None = None,
        limit: int = 0,
    ) -> list[RadarDataset]:
        """Filter datasets based on criteria.

        Args:
            orgs: Filter by organization names
            categories: Filter by category types
            min_downloads: Minimum download count
            signals: Required signals/tags
            limit: Maximum number of results (0 = no limit)

        Returns:
            Filtered list of datasets
        """
        result = self.datasets.copy()

        if orgs:
            orgs_lower = [o.lower() for o in orgs]
            result = [d for d in result if d.org.lower() in orgs_lower]

        if categories:
            cats_lower = [c.lower() for c in categories]
            result = [d for d in result if d.category.lower() in cats_lower]

        if min_downloads > 0:
            result = [d for d in result if d.downloads >= min_downloads]

        if signals:
            signals_lower = {s.lower() for s in signals}
            result = [
                d for d in result if signals_lower.intersection({s.lower() for s in d.signals})
            ]

        # Sort by downloads (descending)
        result.sort(key=lambda x: x.downloads, reverse=True)

        if limit > 0:
            result = result[:limit]

        return result

    def get_dataset_ids(self, datasets: list[RadarDataset] | None = None) -> list[str]:
        """Get list of dataset IDs.

        Args:
            datasets: Optional list of datasets (uses self.datasets if None)

        Returns:
            List of dataset ID strings
        """
        ds_list = datasets if datasets is not None else self.datasets
        return [d.id for d in ds_list]

    # Default purpose descriptions by dataset type
    DATASET_TYPE_PURPOSES = {
        "preference": "RLHF 偏好数据，用于训练奖励模型或直接偏好优化 (DPO)",
        "evaluation": "模型能力评测数据，用于评估 AI 模型在特定任务上的表现",
        "sft": "监督微调数据，用于训练模型遵循指令",
        "swe_bench": "软件工程评测数据，用于评估代码生成和问题修复能力",
        "instruction": "指令遵循数据，用于提升模型指令理解能力",
        "chat": "对话数据，用于训练对话式 AI",
        "unknown": "通用数据集",
    }

    # Default categories by dataset type
    DATASET_TYPE_CATEGORIES = {
        "preference": "rlhf",
        "evaluation": "benchmark",
        "sft": "instruction",
        "swe_bench": "code",
        "instruction": "instruction",
        "chat": "conversation",
    }

    @staticmethod
    def create_summary(
        dataset_id: str,
        dataset_type: str = "",
        category: str = "",
        purpose: str = "",
        allocation: Any = None,
        rubrics_result: Any = None,
        prompt_library: Any = None,
        schema_info: dict | None = None,
        sample_count: int = 0,
        llm_analysis: Any = None,
        output_dir: str = "",
        complexity_metrics: Any = None,
    ) -> RecipeSummary:
        """Create a standardized RecipeSummary from analysis results.

        Args:
            dataset_id: Dataset identifier
            dataset_type: Detected dataset type
            category: Category from radar or detected
            purpose: Dataset purpose description
            allocation: HumanMachineAllocation result
            rubrics_result: RubricsAnalysis result
            prompt_library: PromptLibrary result
            schema_info: Schema information dict
            sample_count: Number of samples analyzed
            llm_analysis: LLMDatasetAnalysis result
            output_dir: Output directory path
            complexity_metrics: ComplexityMetrics result

        Returns:
            RecipeSummary object
        """
        # Fill in default purpose and category based on dataset type
        if not purpose and dataset_type:
            purpose = RadarIntegration.DATASET_TYPE_PURPOSES.get(
                dataset_type, RadarIntegration.DATASET_TYPE_PURPOSES.get("unknown", "")
            )
        if not category and dataset_type:
            category = RadarIntegration.DATASET_TYPE_CATEGORIES.get(dataset_type, "")

        summary = RecipeSummary(
            dataset_id=dataset_id,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            dataset_type=dataset_type,
            category=category,
            purpose=purpose,
            sample_count=sample_count,
        )

        # Cost and allocation info
        if allocation:
            summary.reproduction_cost = {
                "human": round(allocation.total_human_cost, 2),
                "api": round(allocation.total_machine_cost, 2),
                "total": round(allocation.total_cost, 2),
            }
            summary.human_percentage = round(allocation.human_work_percentage, 1)
            summary.machine_percentage = round(allocation.machine_work_percentage, 1)

        # Rubrics patterns
        if rubrics_result:
            summary.rubric_patterns = rubrics_result.unique_patterns
            # Extract top patterns as key patterns
            if hasattr(rubrics_result, "verb_distribution"):
                top_verbs = sorted(
                    rubrics_result.verb_distribution.items(), key=lambda x: x[1], reverse=True
                )[:5]
                summary.key_patterns.extend([f"rubric:{v[0]}" for v in top_verbs])

        # Prompt templates
        if prompt_library:
            summary.prompt_templates = prompt_library.unique_count

        # Schema fields
        if schema_info:
            summary.fields = list(schema_info.keys())

        # Complexity-based difficulty assessment
        if complexity_metrics:
            # Calculate difficulty based on complexity metrics
            difficulty_score = getattr(complexity_metrics, "difficulty_score", 1.0)
            if difficulty_score <= 1.5:
                summary.difficulty = "easy"
            elif difficulty_score <= 2.5:
                summary.difficulty = "medium"
            elif difficulty_score <= 3.5:
                summary.difficulty = "hard"
            else:
                summary.difficulty = "expert"

            # Add domain as key pattern
            domain = getattr(complexity_metrics, "primary_domain", None)
            if domain:
                domain_value = domain.value if hasattr(domain, "value") else str(domain)
                summary.key_patterns.append(f"domain:{domain_value}")

        # LLM analysis enrichment (overrides complexity-based values if available)
        if llm_analysis:
            if not summary.dataset_type:
                summary.dataset_type = llm_analysis.dataset_type
            if not summary.purpose:
                summary.purpose = llm_analysis.purpose
            if llm_analysis.estimated_difficulty:
                summary.difficulty = llm_analysis.estimated_difficulty.split()[
                    0
                ]  # "medium，xxx" -> "medium"
            if llm_analysis.similar_datasets:
                summary.similar_datasets = llm_analysis.similar_datasets

        # Find similar datasets from catalog and knowledge base
        if not summary.similar_datasets and dataset_type:
            try:
                from datarecipe.knowledge import DatasetCatalog

                catalog = DatasetCatalog()
                similar = catalog.find_similar_datasets(
                    dataset_id=dataset_id,
                    category=dataset_type,
                    limit=5,
                )
                summary.similar_datasets = [
                    s.dataset_id for s in similar if s.dataset_id.lower() != dataset_id.lower()
                ][:3]
            except Exception:
                pass

            # Fallback to knowledge base patterns if catalog didn't find enough
            if len(summary.similar_datasets) < 3:
                try:
                    from datarecipe.knowledge import KnowledgeBase

                    kb = KnowledgeBase()
                    similar = kb.find_similar_datasets(dataset_type, dataset_id=dataset_id, limit=5)
                    for s in similar:
                        if (
                            s.dataset_id.lower() != dataset_id.lower()
                            and s.dataset_id not in summary.similar_datasets
                        ):
                            summary.similar_datasets.append(s.dataset_id)
                            if len(summary.similar_datasets) >= 3:
                                break
                except Exception:
                    pass

        # Output paths
        if output_dir:
            summary.report_path = os.path.join(output_dir, "ANALYSIS_REPORT.md")
            summary.guide_path = os.path.join(output_dir, "REPRODUCTION_GUIDE.md")

        return summary

    @staticmethod
    def save_summary(summary: RecipeSummary, output_dir: str) -> str:
        """Save RecipeSummary to JSON file.

        Args:
            summary: RecipeSummary object
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "recipe_summary.json")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary.to_json())

        return output_path

    @staticmethod
    def load_summary(path: str) -> RecipeSummary:
        """Load RecipeSummary from JSON file.

        Args:
            path: Path to recipe_summary.json

        Returns:
            RecipeSummary object
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return RecipeSummary.from_dict(data)

    @staticmethod
    def aggregate_summaries(summaries: list[RecipeSummary]) -> dict:
        """Aggregate multiple summaries for trend analysis.

        Args:
            summaries: List of RecipeSummary objects

        Returns:
            Aggregated statistics dict
        """
        if not summaries:
            return {}

        total_human_cost = sum(s.reproduction_cost.get("human", 0) for s in summaries)
        total_api_cost = sum(s.reproduction_cost.get("api", 0) for s in summaries)

        type_counts = {}
        difficulty_counts = {}
        for s in summaries:
            if s.dataset_type:
                type_counts[s.dataset_type] = type_counts.get(s.dataset_type, 0) + 1
            if s.difficulty:
                difficulty_counts[s.difficulty] = difficulty_counts.get(s.difficulty, 0) + 1

        return {
            "total_datasets": len(summaries),
            "total_reproduction_cost": {
                "human": round(total_human_cost, 2),
                "api": round(total_api_cost, 2),
                "total": round(total_human_cost + total_api_cost, 2),
            },
            "avg_human_percentage": round(
                sum(s.human_percentage for s in summaries) / len(summaries), 1
            ),
            "type_distribution": type_counts,
            "difficulty_distribution": difficulty_counts,
            "datasets": [s.dataset_id for s in summaries],
        }
