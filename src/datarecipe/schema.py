"""Data classes for representing dataset recipes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GenerationType(Enum):
    """Type of data generation."""

    SYNTHETIC = "synthetic"
    HUMAN = "human"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SourceType(Enum):
    """Type of data source."""

    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    OPENAI = "openai"
    LOCAL = "local"
    WEB = "web"
    UNKNOWN = "unknown"


@dataclass
class Cost:
    """Estimated cost breakdown for dataset creation."""

    estimated_total_usd: Optional[float] = None
    api_calls_usd: Optional[float] = None
    human_annotation_usd: Optional[float] = None
    compute_usd: Optional[float] = None
    confidence: str = "low"  # low, medium, high
    # New fields for detailed estimation
    low_estimate_usd: Optional[float] = None
    high_estimate_usd: Optional[float] = None
    assumptions: list[str] = field(default_factory=list)
    tokens_estimated: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "estimated_total_usd": self.estimated_total_usd,
            "breakdown": {
                "api_calls": self.api_calls_usd,
                "human_annotation": self.human_annotation_usd,
                "compute": self.compute_usd,
            },
            "confidence": self.confidence,
        }
        if self.low_estimate_usd is not None:
            result["low_estimate_usd"] = self.low_estimate_usd
        if self.high_estimate_usd is not None:
            result["high_estimate_usd"] = self.high_estimate_usd
        if self.assumptions:
            result["assumptions"] = self.assumptions
        if self.tokens_estimated is not None:
            result["tokens_estimated"] = self.tokens_estimated
        return result


@dataclass
class Reproducibility:
    """Reproducibility assessment for a dataset."""

    score: int  # 1-10
    available: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "available": self.available,
            "missing": self.missing,
            "notes": self.notes,
        }


@dataclass
class GenerationMethod:
    """Details about how data was generated."""

    method_type: str  # distillation, human_annotation, web_scrape, etc.
    teacher_model: Optional[str] = None
    prompt_template_available: bool = False
    platform: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {"type": self.method_type}
        if self.teacher_model:
            result["teacher_model"] = self.teacher_model
        if self.prompt_template_available:
            result["prompt_template"] = "available"
        if self.platform:
            result["platform"] = self.platform
        if self.details:
            result.update(self.details)
        return result


@dataclass
class Recipe:
    """Complete recipe for a dataset."""

    name: str
    version: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    source_id: Optional[str] = None

    # Dataset characteristics
    size: Optional[int] = None
    num_examples: Optional[int] = None
    languages: list[str] = field(default_factory=list)
    license: Optional[str] = None
    description: Optional[str] = None

    # Generation details
    generation_type: GenerationType = GenerationType.UNKNOWN
    synthetic_ratio: Optional[float] = None
    human_ratio: Optional[float] = None
    generation_methods: list[GenerationMethod] = field(default_factory=list)
    teacher_models: list[str] = field(default_factory=list)

    # Assessment
    cost: Optional[Cost] = None
    reproducibility: Optional[Reproducibility] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    created_date: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    paper_url: Optional[str] = None
    homepage_url: Optional[str] = None

    # Quality metrics (populated by quality analyzer)
    quality_metrics: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML export."""
        result = {
            "name": self.name,
            "source": {
                "type": self.source_type.value,
                "id": self.source_id,
            },
        }

        if self.version:
            result["version"] = self.version

        # Generation info
        generation = {}
        if self.synthetic_ratio is not None:
            generation["synthetic_ratio"] = self.synthetic_ratio
        if self.human_ratio is not None:
            generation["human_ratio"] = self.human_ratio
        if self.generation_methods:
            generation["methods"] = [m.to_dict() for m in self.generation_methods]
        if self.teacher_models:
            generation["teacher_models"] = self.teacher_models
        if generation:
            result["generation"] = generation

        # Cost
        if self.cost:
            result["cost"] = self.cost.to_dict()

        # Reproducibility
        if self.reproducibility:
            result["reproducibility"] = self.reproducibility.to_dict()

        # Metadata
        metadata = {}
        if self.size:
            metadata["size_bytes"] = self.size
        if self.num_examples:
            metadata["num_examples"] = self.num_examples
        if self.languages:
            metadata["languages"] = self.languages
        if self.license:
            metadata["license"] = self.license
        if self.tags:
            metadata["tags"] = self.tags
        if self.authors:
            metadata["authors"] = self.authors
        if self.paper_url:
            metadata["paper_url"] = self.paper_url
        if metadata:
            result["metadata"] = metadata

        return result

    def to_yaml(self) -> str:
        """Export recipe as YAML string."""
        import yaml

        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
