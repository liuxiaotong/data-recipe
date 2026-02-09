"""Dataset comparison and reporting."""

import math
from dataclasses import dataclass, field

from datarecipe.analyzer import DatasetAnalyzer
from datarecipe.cost_calculator import CostBreakdown, CostCalculator
from datarecipe.quality_metrics import QualityAnalyzer, QualityReport
from datarecipe.schema import GenerationType, Recipe, SourceType


# ==================== Similarity dataclasses ====================


@dataclass
class SimilarityWeights:
    """Configurable weights for similarity scoring dimensions."""

    schema_overlap: float = 0.20
    size_ratio: float = 0.15
    generation_type: float = 0.15
    quality_profile: float = 0.20
    tag_overlap: float = 0.15
    cost_ratio: float = 0.15

    def validate(self) -> None:
        """Raise ValueError if weights do not sum to 1.0."""
        total = (
            self.schema_overlap
            + self.size_ratio
            + self.generation_type
            + self.quality_profile
            + self.tag_overlap
            + self.cost_ratio
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


@dataclass
class SimilarityBreakdown:
    """Detailed breakdown of similarity sub-scores."""

    schema_overlap: float
    size_ratio: float
    generation_type: float
    quality_profile: float
    tag_overlap: float
    cost_ratio: float


@dataclass
class SimilarityResult:
    """Pairwise similarity result between two datasets."""

    dataset_a: str
    dataset_b: str
    overall_score: float
    breakdown: SimilarityBreakdown
    weights: SimilarityWeights

    def to_dict(self) -> dict:
        return {
            "dataset_a": self.dataset_a,
            "dataset_b": self.dataset_b,
            "overall_score": round(self.overall_score, 4),
            "breakdown": {
                "schema_overlap": round(self.breakdown.schema_overlap, 4),
                "size_ratio": round(self.breakdown.size_ratio, 4),
                "generation_type": round(self.breakdown.generation_type, 4),
                "quality_profile": round(self.breakdown.quality_profile, 4),
                "tag_overlap": round(self.breakdown.tag_overlap, 4),
                "cost_ratio": round(self.breakdown.cost_ratio, 4),
            },
        }


# ==================== Field diff dataclasses ====================


@dataclass
class FieldDiff:
    """Per-field comparison between two datasets."""

    field_name: str
    value_a: str
    value_b: str
    delta: str
    indicator: str  # "=", "~", "!=", "+", "-"

    def to_dict(self) -> dict:
        return {
            "field": self.field_name,
            "a": self.value_a,
            "b": self.value_b,
            "delta": self.delta,
            "indicator": self.indicator,
        }


# ==================== Schema comparison dataclasses ====================


@dataclass
class SchemaComparison:
    """Comparison of data schemas between two datasets."""

    dataset_a: str
    dataset_b: str
    common_fields: list[str]
    only_in_a: list[str]
    only_in_b: list[str]
    type_mismatches: dict[str, tuple[str, str]]
    jaccard_similarity: float

    def to_dict(self) -> dict:
        return {
            "dataset_a": self.dataset_a,
            "dataset_b": self.dataset_b,
            "common_fields": self.common_fields,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "type_mismatches": {
                k: {"a": v[0], "b": v[1]} for k, v in self.type_mismatches.items()
            },
            "jaccard_similarity": round(self.jaccard_similarity, 4),
        }


# ==================== Core dataclasses ====================


@dataclass
class DatasetMetrics:
    """Collected metrics for a single dataset."""

    dataset_id: str
    recipe: Recipe
    cost: CostBreakdown | None = None
    quality: QualityReport | None = None
    data_schema: dict[str, str] | None = None


@dataclass
class ComparisonReport:
    """Report comparing multiple datasets."""

    datasets: list[str]
    metrics: list[DatasetMetrics] = field(default_factory=list)
    cost_comparison: dict = field(default_factory=dict)
    quality_comparison: dict = field(default_factory=dict)
    strengths: dict[str, list[str]] = field(default_factory=dict)
    weaknesses: dict[str, list[str]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    best_for: dict[str, str] = field(default_factory=dict)
    similarity_results: list[SimilarityResult] = field(default_factory=list)
    field_diffs: list[list[FieldDiff]] = field(default_factory=list)
    schema_comparisons: list[SchemaComparison] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = []

        # Title
        lines.append("# Dataset Comparison Report")
        lines.append("")

        # Overview table
        lines.append("## Overview")
        lines.append("")
        lines.append("| Dataset | Size | Cost (Est.) | Quality | Reproducibility |")
        lines.append("|---------|------|-------------|---------|-----------------|")

        for m in self.metrics:
            size = f"{m.recipe.num_examples:,}" if m.recipe.num_examples else "N/A"
            cost = f"${m.cost.total.expected:,.0f}" if m.cost else "N/A"
            quality = f"{m.quality.overall_score:.0f}/100" if m.quality else "N/A"
            repro = f"{m.recipe.reproducibility.score}/10" if m.recipe.reproducibility else "N/A"
            lines.append(f"| {m.dataset_id} | {size} | {cost} | {quality} | {repro} |")

        lines.append("")

        # Cost comparison
        if self.cost_comparison:
            lines.append("## Cost Comparison")
            lines.append("")
            lines.append("| Dataset | API Cost | Human Cost | Compute | Total |")
            lines.append("|---------|----------|------------|---------|-------|")

            for dataset_id, costs in self.cost_comparison.items():
                api = f"${costs.get('api', 0):,.0f}"
                human = f"${costs.get('human', 0):,.0f}"
                compute = f"${costs.get('compute', 0):,.0f}"
                total = f"${costs.get('total', 0):,.0f}"
                lines.append(f"| {dataset_id} | {api} | {human} | {compute} | {total} |")

            lines.append("")

        # Quality comparison
        if self.quality_comparison:
            lines.append("## Quality Comparison")
            lines.append("")
            lines.append("| Dataset | Diversity | Consistency | Complexity | Overall |")
            lines.append("|---------|-----------|-------------|------------|---------|")

            for dataset_id, scores in self.quality_comparison.items():
                diversity = f"{scores.get('diversity', 0):.2f}"
                consistency = f"{scores.get('consistency', 0):.2f}"
                complexity = f"{scores.get('complexity', 0):.2f}"
                overall = f"{scores.get('overall', 0):.0f}"
                lines.append(
                    f"| {dataset_id} | {diversity} | {consistency} | {complexity} | {overall} |"
                )

            lines.append("")

        # Similarity matrix
        if self.similarity_results:
            lines.append("## Similarity Matrix")
            lines.append("")
            lines.append(
                "| Pair | Score | Schema | Size | Gen Type | Quality | Tags | Cost |"
            )
            lines.append(
                "|------|-------|--------|------|----------|---------|------|------|"
            )
            for sr in self.similarity_results:
                b = sr.breakdown
                lines.append(
                    f"| {sr.dataset_a} vs {sr.dataset_b} "
                    f"| {sr.overall_score:.2f} "
                    f"| {b.schema_overlap:.2f} "
                    f"| {b.size_ratio:.2f} "
                    f"| {b.generation_type:.2f} "
                    f"| {b.quality_profile:.2f} "
                    f"| {b.tag_overlap:.2f} "
                    f"| {b.cost_ratio:.2f} |"
                )
            lines.append("")

        # Field-level diffs
        for i, diffs in enumerate(self.field_diffs):
            if not diffs:
                continue
            sr = self.similarity_results[i] if i < len(self.similarity_results) else None
            if sr:
                title = f"## Field Comparison: {sr.dataset_a} vs {sr.dataset_b}"
            else:
                title = f"## Field Comparison #{i + 1}"
            lines.append(title)
            lines.append("")
            lines.append("| Field | A | B | Delta |")
            lines.append("|-------|---|---|-------|")
            for d in diffs:
                lines.append(f"| {d.field_name} | {d.value_a} | {d.value_b} | {d.delta} |")
            lines.append("")

        # Schema comparisons
        for sc in self.schema_comparisons:
            lines.append(f"## Schema Comparison: {sc.dataset_a} vs {sc.dataset_b}")
            lines.append("")
            lines.append(f"- **Jaccard similarity:** {sc.jaccard_similarity:.2f}")
            if sc.common_fields:
                lines.append(
                    f"- **Common fields ({len(sc.common_fields)}):** "
                    f"{', '.join(sc.common_fields)}"
                )
            if sc.only_in_a:
                lines.append(
                    f"- **Only in {sc.dataset_a} ({len(sc.only_in_a)}):** "
                    f"{', '.join(sc.only_in_a)}"
                )
            if sc.only_in_b:
                lines.append(
                    f"- **Only in {sc.dataset_b} ({len(sc.only_in_b)}):** "
                    f"{', '.join(sc.only_in_b)}"
                )
            if sc.type_mismatches:
                lines.append("- **Type mismatches:**")
                for fld, (ta, tb) in sc.type_mismatches.items():
                    lines.append(f"  - {fld}: {ta} vs {tb}")
            lines.append("")

        # Strengths and weaknesses
        lines.append("## Analysis")
        lines.append("")

        for dataset_id in self.datasets:
            lines.append(f"### {dataset_id}")
            lines.append("")

            if dataset_id in self.strengths and self.strengths[dataset_id]:
                lines.append("**Strengths:**")
                for s in self.strengths[dataset_id]:
                    lines.append(f"- {s}")
                lines.append("")

            if dataset_id in self.weaknesses and self.weaknesses[dataset_id]:
                lines.append("**Weaknesses:**")
                for w in self.weaknesses[dataset_id]:
                    lines.append(f"- {w}")
                lines.append("")

        # Best for
        if self.best_for:
            lines.append("## Best For")
            lines.append("")
            lines.append("| Use Case | Recommended Dataset |")
            lines.append("|----------|---------------------|")
            for use_case, dataset in self.best_for.items():
                lines.append(f"| {use_case} | {dataset} |")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("---")
        lines.append("> Generated by DataRecipe")

        return "\n".join(lines)

    def to_table(self) -> str:
        """Generate ASCII table comparison."""
        # Calculate column widths
        cols = ["Metric"] + self.datasets
        widths = [max(14, len(c)) for c in cols]

        # Header
        lines = []
        header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths, strict=False)) + " |"
        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        # Data rows
        rows_data = [
            (
                "Size",
                [
                    f"{m.recipe.num_examples:,}" if m.recipe.num_examples else "N/A"
                    for m in self.metrics
                ],
            ),
            (
                "Est. Cost",
                [f"${m.cost.total.expected:,.0f}" if m.cost else "N/A" for m in self.metrics],
            ),
            (
                "Quality",
                [
                    f"{m.quality.overall_score:.0f}/100" if m.quality else "N/A"
                    for m in self.metrics
                ],
            ),
            (
                "Reproducibility",
                [
                    f"{m.recipe.reproducibility.score}/10" if m.recipe.reproducibility else "N/A"
                    for m in self.metrics
                ],
            ),
            (
                "Synthetic %",
                [
                    f"{m.recipe.synthetic_ratio * 100:.0f}%" if m.recipe.synthetic_ratio else "N/A"
                    for m in self.metrics
                ],
            ),
        ]

        for row_name, values in rows_data:
            row = "| " + row_name.ljust(widths[0])
            for i, val in enumerate(values):
                row += " | " + str(val).ljust(widths[i + 1])
            row += " |"
            lines.append(row)

        lines.append(separator)

        # Similarity scores
        if self.similarity_results:
            lines.append("")
            lines.append("Similarity:")
            for sr in self.similarity_results:
                lines.append(f"  {sr.dataset_a} vs {sr.dataset_b} = {sr.overall_score:.2f}")

        # Best for section
        if self.best_for:
            lines.append("")
            lines.append("Recommendations:")
            for use_case, dataset in self.best_for.items():
                lines.append(f"  {use_case}: {dataset}")

        return "\n".join(lines)


# ==================== Comparator ====================


class DatasetComparator:
    """Compare multiple datasets."""

    def __init__(
        self,
        include_cost: bool = True,
        include_quality: bool = False,
        include_similarity: bool = False,
        include_schema: bool = False,
        quality_sample_size: int = 500,
    ):
        """Initialize the comparator.

        Args:
            include_cost: Whether to include cost estimation
            include_quality: Whether to include quality analysis
            include_similarity: Whether to compute pairwise similarity scores
            include_schema: Whether to compare data schemas
            quality_sample_size: Sample size for quality analysis
        """
        self.include_cost = include_cost
        self.include_quality = include_quality
        self.include_similarity = include_similarity
        self.include_schema = include_schema
        self.quality_sample_size = quality_sample_size

        self.analyzer = DatasetAnalyzer()
        self.cost_calculator = CostCalculator()
        self.quality_analyzer = QualityAnalyzer()

    def compare(self, recipes: list[Recipe]) -> ComparisonReport:
        """Compare multiple dataset recipes.

        Args:
            recipes: List of Recipe objects to compare

        Returns:
            ComparisonReport with comparison data
        """
        metrics = []

        for recipe in recipes:
            dataset_id = recipe.source_id or recipe.name

            # Cost estimation
            cost = None
            if self.include_cost:
                try:
                    cost = self.cost_calculator.estimate_from_recipe(recipe)
                except (ValueError, TypeError, KeyError, AttributeError):
                    pass

            # Quality analysis (requires data loading)
            quality = None
            if self.include_quality and recipe.source_id:
                try:
                    quality = self.quality_analyzer.analyze_from_huggingface(
                        recipe.source_id,
                        sample_size=self.quality_sample_size,
                    )
                except (ImportError, OSError, ValueError, AttributeError):
                    pass

            # Schema inference
            data_schema = None
            if self.include_schema or self.include_similarity:
                data_schema = self._infer_data_schema(recipe)

            metrics.append(
                DatasetMetrics(
                    dataset_id=dataset_id,
                    recipe=recipe,
                    cost=cost,
                    quality=quality,
                    data_schema=data_schema,
                )
            )

        return self._build_report(metrics)

    def compare_by_ids(self, dataset_ids: list[str]) -> ComparisonReport:
        """Compare datasets by their IDs.

        Args:
            dataset_ids: List of dataset IDs to compare

        Returns:
            ComparisonReport with comparison data
        """
        recipes = []
        for dataset_id in dataset_ids:
            try:
                recipe = self.analyzer.analyze(dataset_id)
                recipes.append(recipe)
            except (OSError, ValueError, KeyError, AttributeError):
                # Create a minimal recipe for failed analyses
                recipe = Recipe(
                    name=dataset_id,
                    source_id=dataset_id,
                )
                recipes.append(recipe)

        return self.compare(recipes)

    # ==================== Similarity scoring ====================

    def compute_similarity(
        self,
        metric_a: DatasetMetrics,
        metric_b: DatasetMetrics,
        weights: SimilarityWeights | None = None,
    ) -> SimilarityResult:
        """Compute pairwise similarity between two datasets.

        Args:
            metric_a: Metrics for dataset A
            metric_b: Metrics for dataset B
            weights: Optional custom weights (defaults to equal-ish distribution)

        Returns:
            SimilarityResult with overall score and breakdown
        """
        if weights is None:
            weights = SimilarityWeights()

        breakdown = SimilarityBreakdown(
            schema_overlap=self._schema_jaccard(metric_a, metric_b),
            size_ratio=self._size_similarity(metric_a, metric_b),
            generation_type=self._generation_type_similarity(metric_a, metric_b),
            quality_profile=self._quality_similarity(metric_a, metric_b),
            tag_overlap=self._tag_jaccard(metric_a, metric_b),
            cost_ratio=self._cost_similarity(metric_a, metric_b),
        )

        overall = (
            weights.schema_overlap * breakdown.schema_overlap
            + weights.size_ratio * breakdown.size_ratio
            + weights.generation_type * breakdown.generation_type
            + weights.quality_profile * breakdown.quality_profile
            + weights.tag_overlap * breakdown.tag_overlap
            + weights.cost_ratio * breakdown.cost_ratio
        )

        return SimilarityResult(
            dataset_a=metric_a.dataset_id,
            dataset_b=metric_b.dataset_id,
            overall_score=overall,
            breakdown=breakdown,
            weights=weights,
        )

    def _schema_jaccard(self, a: DatasetMetrics, b: DatasetMetrics) -> float:
        if not a.data_schema or not b.data_schema:
            return 0.5
        fields_a = set(a.data_schema.keys())
        fields_b = set(b.data_schema.keys())
        union = fields_a | fields_b
        if not union:
            return 1.0
        return len(fields_a & fields_b) / len(union)

    def _size_similarity(self, a: DatasetMetrics, b: DatasetMetrics) -> float:
        n1 = a.recipe.num_examples
        n2 = b.recipe.num_examples
        if not n1 or not n2:
            return 0.5
        if n1 == 0 or n2 == 0:
            return 0.0
        ratio = abs(math.log(n1 / n2))
        return max(0.0, 1.0 - ratio / math.log(100))

    def _generation_type_similarity(self, a: DatasetMetrics, b: DatasetMetrics) -> float:
        gt_a = a.recipe.generation_type
        gt_b = b.recipe.generation_type
        if gt_a == gt_b:
            return 1.0
        if gt_a == GenerationType.UNKNOWN or gt_b == GenerationType.UNKNOWN:
            return 0.5
        return 0.0

    def _quality_similarity(self, a: DatasetMetrics, b: DatasetMetrics) -> float:
        if not a.quality or not b.quality:
            return 0.5
        dims_a = [
            a.quality.diversity.unique_token_ratio,
            a.quality.consistency.format_consistency,
            a.quality.complexity.vocabulary_richness,
            a.quality.overall_score / 100.0,
        ]
        dims_b = [
            b.quality.diversity.unique_token_ratio,
            b.quality.consistency.format_consistency,
            b.quality.complexity.vocabulary_richness,
            b.quality.overall_score / 100.0,
        ]
        dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(dims_a, dims_b)))
        max_dist = math.sqrt(len(dims_a))
        return max(0.0, 1.0 - dist / max_dist)

    def _tag_jaccard(self, a: DatasetMetrics, b: DatasetMetrics) -> float:
        tags_a = set(a.recipe.tags or [])
        tags_b = set(b.recipe.tags or [])
        union = tags_a | tags_b
        if not union:
            return 1.0
        return len(tags_a & tags_b) / len(union)

    def _cost_similarity(self, a: DatasetMetrics, b: DatasetMetrics) -> float:
        c1 = a.cost.total.expected if a.cost else None
        c2 = b.cost.total.expected if b.cost else None
        if c1 is None or c2 is None:
            return 0.5
        if c1 <= 0 or c2 <= 0:
            return 1.0 if c1 == c2 else 0.0
        ratio = abs(math.log(c1 / c2))
        return max(0.0, 1.0 - ratio / math.log(100))

    # ==================== Field-level diff ====================

    def compute_field_diff(
        self,
        metric_a: DatasetMetrics,
        metric_b: DatasetMetrics,
    ) -> list[FieldDiff]:
        """Compare Recipe-level fields between two datasets.

        Returns a list of FieldDiff, one per compared field.
        """
        diffs = []
        ra = metric_a.recipe
        rb = metric_b.recipe

        # num_examples
        diffs.append(self._diff_numeric(
            "num_examples", ra.num_examples, rb.num_examples, fmt_fn=lambda v: f"{v:,}",
        ))

        # size (bytes)
        diffs.append(self._diff_numeric(
            "size", ra.size, rb.size,
            fmt_fn=lambda v: f"{v / (1024 * 1024):.1f} MB",
        ))

        # generation_type
        ga = ra.generation_type.value if ra.generation_type else "N/A"
        gb = rb.generation_type.value if rb.generation_type else "N/A"
        diffs.append(FieldDiff(
            field_name="generation_type",
            value_a=ga, value_b=gb,
            delta="match" if ga == gb else "different",
            indicator="=" if ga == gb else "!=",
        ))

        # synthetic_ratio
        diffs.append(self._diff_ratio("synthetic_ratio", ra.synthetic_ratio, rb.synthetic_ratio))

        # human_ratio
        diffs.append(self._diff_ratio("human_ratio", ra.human_ratio, rb.human_ratio))

        # languages
        diffs.append(self._diff_list("languages", ra.languages, rb.languages))

        # tags
        diffs.append(self._diff_list("tags", ra.tags, rb.tags))

        # license
        la = ra.license or "N/A"
        lb = rb.license or "N/A"
        diffs.append(FieldDiff(
            field_name="license",
            value_a=la, value_b=lb,
            delta="match" if la == lb else "different",
            indicator="=" if la == lb else "!=",
        ))

        # teacher_models
        diffs.append(self._diff_list("teacher_models", ra.teacher_models, rb.teacher_models))

        # reproducibility score
        sa = ra.reproducibility.score if ra.reproducibility else None
        sb = rb.reproducibility.score if rb.reproducibility else None
        diffs.append(self._diff_int("reproducibility", sa, sb))

        # quality overall_score
        qa = metric_a.quality.overall_score if metric_a.quality else None
        qb = metric_b.quality.overall_score if metric_b.quality else None
        diffs.append(self._diff_int("quality_score", qa, qb))

        # cost total
        ca = metric_a.cost.total.expected if metric_a.cost else None
        cb = metric_b.cost.total.expected if metric_b.cost else None
        diffs.append(self._diff_numeric(
            "cost_total", ca, cb, fmt_fn=lambda v: f"${v:,.0f}",
        ))

        return diffs

    def _diff_numeric(
        self, name: str, a: float | int | None, b: float | int | None,
        fmt_fn=None,
    ) -> FieldDiff:
        if fmt_fn is None:
            fmt_fn = str
        va = fmt_fn(a) if a is not None else "N/A"
        vb = fmt_fn(b) if b is not None else "N/A"
        if a is None or b is None:
            return FieldDiff(name, va, vb, "N/A", "~")
        if a == b:
            return FieldDiff(name, va, vb, "match", "=")
        if a == 0:
            return FieldDiff(name, va, vb, "N/A", "~")
        pct = (b - a) / abs(a) * 100
        delta = f"{pct:+.0f}%"
        if abs(pct) <= 10:
            indicator = "~"
        elif pct > 0:
            indicator = "+"
        else:
            indicator = "-"
        return FieldDiff(name, va, vb, delta, indicator)

    def _diff_ratio(self, name: str, a: float | None, b: float | None) -> FieldDiff:
        va = f"{a:.2f}" if a is not None else "N/A"
        vb = f"{b:.2f}" if b is not None else "N/A"
        if a is None or b is None:
            return FieldDiff(name, va, vb, "N/A", "~")
        if a == b:
            return FieldDiff(name, va, vb, "match", "=")
        diff = b - a
        delta = f"{diff:+.2f}"
        if abs(diff) <= 0.05:
            indicator = "~"
        elif diff > 0:
            indicator = "+"
        else:
            indicator = "-"
        return FieldDiff(name, va, vb, delta, indicator)

    def _diff_int(self, name: str, a: int | float | None, b: int | float | None) -> FieldDiff:
        va = str(a) if a is not None else "N/A"
        vb = str(b) if b is not None else "N/A"
        if a is None or b is None:
            return FieldDiff(name, va, vb, "N/A", "~")
        if a == b:
            return FieldDiff(name, va, vb, "match", "=")
        diff = b - a
        delta = f"{diff:+g}"
        if abs(diff) <= 1:
            indicator = "~"
        elif diff > 0:
            indicator = "+"
        else:
            indicator = "-"
        return FieldDiff(name, va, vb, delta, indicator)

    def _diff_list(self, name: str, a: list | None, b: list | None) -> FieldDiff:
        sa = set(a or [])
        sb = set(b or [])
        va = ", ".join(sorted(sa)) if sa else "N/A"
        vb = ", ".join(sorted(sb)) if sb else "N/A"
        union = sa | sb
        if not union:
            return FieldDiff(name, va, vb, "match", "=")
        overlap = len(sa & sb) / len(union) * 100
        if overlap == 100:
            return FieldDiff(name, va, vb, "match", "=")
        delta = f"{overlap:.0f}% overlap"
        indicator = "~" if overlap >= 50 else "!="
        return FieldDiff(name, va, vb, delta, indicator)

    # ==================== Schema comparison ====================

    def compare_schemas(
        self,
        metric_a: DatasetMetrics,
        metric_b: DatasetMetrics,
    ) -> SchemaComparison:
        """Compare data schemas between two datasets."""
        schema_a = metric_a.data_schema or {}
        schema_b = metric_b.data_schema or {}

        fields_a = set(schema_a.keys())
        fields_b = set(schema_b.keys())
        common = sorted(fields_a & fields_b)
        only_a = sorted(fields_a - fields_b)
        only_b = sorted(fields_b - fields_a)

        type_mismatches = {}
        for f in common:
            if schema_a[f] != schema_b[f]:
                type_mismatches[f] = (schema_a[f], schema_b[f])

        union = fields_a | fields_b
        jaccard = len(fields_a & fields_b) / len(union) if union else 1.0

        return SchemaComparison(
            dataset_a=metric_a.dataset_id,
            dataset_b=metric_b.dataset_id,
            common_fields=common,
            only_in_a=only_a,
            only_in_b=only_b,
            type_mismatches=type_mismatches,
            jaccard_similarity=jaccard,
        )

    # ==================== Schema inference ====================

    def _infer_data_schema(self, recipe: Recipe) -> dict[str, str] | None:
        """Try to infer data schema from the dataset source."""
        try:
            if recipe.source_type == SourceType.LOCAL and recipe.source_id:
                from pathlib import Path

                from datarecipe.sources.local import LocalFileExtractor

                p = Path(recipe.source_id)
                if p.exists() and p.is_file():
                    extractor = LocalFileExtractor()
                    from datarecipe.sources.local import detect_format

                    fmt = detect_format(p)
                    samples = extractor._load_samples(p, fmt, max_samples=5)
                    if samples:
                        return extractor._infer_schema(samples)

            if recipe.source_type == SourceType.HUGGINGFACE and recipe.source_id:
                from datasets import load_dataset

                ds = load_dataset(recipe.source_id, split="train", streaming=True)
                sample = next(iter(ds))
                return {k: type(v).__name__ for k, v in sample.items()}
        except Exception:
            pass
        return None

    # ==================== Report building ====================

    def _build_report(self, metrics: list[DatasetMetrics]) -> ComparisonReport:
        """Build comparison report from metrics."""
        datasets = [m.dataset_id for m in metrics]

        # Cost comparison
        cost_comparison = {}
        for m in metrics:
            if m.cost:
                cost_comparison[m.dataset_id] = {
                    "api": m.cost.api_cost.expected,
                    "human": m.cost.human_annotation_cost.expected,
                    "compute": m.cost.compute_cost.expected,
                    "total": m.cost.total.expected,
                }

        # Quality comparison
        quality_comparison = {}
        for m in metrics:
            if m.quality:
                quality_comparison[m.dataset_id] = {
                    "diversity": m.quality.diversity.unique_token_ratio,
                    "consistency": m.quality.consistency.format_consistency,
                    "complexity": m.quality.complexity.vocabulary_richness,
                    "overall": m.quality.overall_score,
                }

        # Analyze strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(metrics)

        # Determine best for use cases
        best_for = self._determine_best_for(metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)

        # Pairwise similarity, field diffs, schema comparisons
        similarity_results = []
        field_diffs = []
        schema_comparisons = []

        if self.include_similarity and len(metrics) >= 2:
            for i in range(len(metrics)):
                for j in range(i + 1, len(metrics)):
                    similarity_results.append(
                        self.compute_similarity(metrics[i], metrics[j])
                    )
                    field_diffs.append(
                        self.compute_field_diff(metrics[i], metrics[j])
                    )

        if self.include_schema and len(metrics) >= 2:
            for i in range(len(metrics)):
                for j in range(i + 1, len(metrics)):
                    schema_comparisons.append(
                        self.compare_schemas(metrics[i], metrics[j])
                    )

        return ComparisonReport(
            datasets=datasets,
            metrics=metrics,
            cost_comparison=cost_comparison,
            quality_comparison=quality_comparison,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            best_for=best_for,
            similarity_results=similarity_results,
            field_diffs=field_diffs,
            schema_comparisons=schema_comparisons,
        )

    def _analyze_strengths_weaknesses(
        self, metrics: list[DatasetMetrics]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Analyze strengths and weaknesses of each dataset."""
        strengths = {}
        weaknesses = {}

        # Find min/max values for comparison
        costs = [(m.dataset_id, m.cost.total.expected) for m in metrics if m.cost]
        sizes = [(m.dataset_id, m.recipe.num_examples) for m in metrics if m.recipe.num_examples]
        repros = [
            (m.dataset_id, m.recipe.reproducibility.score)
            for m in metrics
            if m.recipe.reproducibility
        ]
        qualities = [(m.dataset_id, m.quality.overall_score) for m in metrics if m.quality]

        min_cost = min(costs, key=lambda x: x[1]) if costs else None
        max_size = max(sizes, key=lambda x: x[1]) if sizes else None
        max_repro = max(repros, key=lambda x: x[1]) if repros else None
        max_quality = max(qualities, key=lambda x: x[1]) if qualities else None

        for m in metrics:
            s = []
            w = []

            # Cost analysis
            if m.cost:
                if min_cost and m.dataset_id == min_cost[0]:
                    s.append("Lowest estimated production cost")
                elif costs:
                    avg_cost = sum(c[1] for c in costs) / len(costs)
                    if m.cost.total.expected > avg_cost * 1.5:
                        w.append("Higher than average production cost")

            # Size analysis
            if m.recipe.num_examples:
                if max_size and m.dataset_id == max_size[0]:
                    s.append("Largest dataset size")
                if m.recipe.num_examples < 1000:
                    w.append("Small dataset size")

            # Reproducibility analysis
            if m.recipe.reproducibility:
                score = m.recipe.reproducibility.score
                if max_repro and m.dataset_id == max_repro[0]:
                    s.append("Highest reproducibility score")
                if score >= 8:
                    s.append("Excellent documentation and reproducibility")
                elif score <= 4:
                    w.append("Limited reproducibility information")

            # Quality analysis
            if m.quality:
                if max_quality and m.dataset_id == max_quality[0]:
                    s.append("Highest quality score")
                if m.quality.overall_score >= 80:
                    s.append("High overall quality")
                elif m.quality.overall_score < 50:
                    w.append("Below average quality metrics")

                if m.quality.diversity.semantic_diversity > 0.6:
                    s.append("High content diversity")
                elif m.quality.diversity.semantic_diversity < 0.3:
                    w.append("Low content diversity")

            # Teacher model analysis
            if m.recipe.teacher_models:
                from datarecipe.constants import PREMIUM_TEACHER_MODELS

                if any(pm in str(m.recipe.teacher_models).lower() for pm in PREMIUM_TEACHER_MODELS):
                    s.append("Generated with premium AI models")

            # Synthetic ratio analysis
            if m.recipe.synthetic_ratio is not None:
                if m.recipe.synthetic_ratio > 0.9:
                    s.append("Highly scalable (mostly synthetic)")
                    w.append("May have synthetic data artifacts")
                elif m.recipe.synthetic_ratio < 0.1:
                    s.append("High-quality human annotations")
                    w.append("Difficult to scale")

            strengths[m.dataset_id] = s
            weaknesses[m.dataset_id] = w

        return strengths, weaknesses

    def _determine_best_for(self, metrics: list[DatasetMetrics]) -> dict[str, str]:
        """Determine which dataset is best for each use case."""
        best_for = {}

        if not metrics:
            return best_for

        # Best value (quality per cost)
        value_scores = []
        for m in metrics:
            if m.cost and m.quality and m.cost.total.expected > 0:
                value = m.quality.overall_score / (m.cost.total.expected / 1000)
                value_scores.append((m.dataset_id, value))

        if value_scores:
            best_value = max(value_scores, key=lambda x: x[1])
            best_for["Best value (quality/cost)"] = best_value[0]

        # Largest scale
        sizes = [(m.dataset_id, m.recipe.num_examples) for m in metrics if m.recipe.num_examples]
        if sizes:
            best_for["Largest scale"] = max(sizes, key=lambda x: x[1])[0]

        # Most reproducible
        repros = [
            (m.dataset_id, m.recipe.reproducibility.score)
            for m in metrics
            if m.recipe.reproducibility
        ]
        if repros:
            best_for["Most reproducible"] = max(repros, key=lambda x: x[1])[0]

        # Highest quality
        qualities = [(m.dataset_id, m.quality.overall_score) for m in metrics if m.quality]
        if qualities:
            best_for["Highest quality"] = max(qualities, key=lambda x: x[1])[0]

        # Lowest cost
        costs = [(m.dataset_id, m.cost.total.expected) for m in metrics if m.cost]
        if costs:
            best_for["Lowest cost"] = min(costs, key=lambda x: x[1])[0]

        # Quick iteration (small but quality)
        quick_candidates = [
            (m.dataset_id, m.quality.overall_score if m.quality else 0)
            for m in metrics
            if m.recipe.num_examples and m.recipe.num_examples < 10000
        ]
        if quick_candidates:
            best_for["Quick prototyping"] = max(quick_candidates, key=lambda x: x[1])[0]

        return best_for

    def _generate_recommendations(self, metrics: list[DatasetMetrics]) -> list[str]:
        """Generate overall recommendations."""
        recommendations = []

        if len(metrics) < 2:
            return ["Add more datasets for meaningful comparison"]

        # Find standouts
        has_cost = any(m.cost for m in metrics)
        has_quality = any(m.quality for m in metrics)

        if has_cost:
            costs = [(m.dataset_id, m.cost.total.expected) for m in metrics if m.cost]
            if costs:
                cheapest = min(costs, key=lambda x: x[1])
                expensive = max(costs, key=lambda x: x[1])
                if expensive[1] > cheapest[1] * 3:
                    recommendations.append(
                        f"Consider {cheapest[0]} for budget-constrained projects "
                        f"(${cheapest[1]:,.0f} vs ${expensive[1]:,.0f})"
                    )

        if has_quality:
            qualities = [(m.dataset_id, m.quality.overall_score) for m in metrics if m.quality]
            if qualities:
                best_q = max(qualities, key=lambda x: x[1])
                if best_q[1] >= 80:
                    recommendations.append(
                        f"{best_q[0]} has the highest quality score ({best_q[1]:.0f}/100)"
                    )

        # Reproducibility recommendations
        for m in metrics:
            if m.recipe.reproducibility and m.recipe.reproducibility.score >= 8:
                recommendations.append(
                    f"{m.dataset_id} is highly reproducible - ideal for research replication"
                )
                break

        # Diversity recommendations
        diverse = [
            m.dataset_id
            for m in metrics
            if m.quality and m.quality.diversity.semantic_diversity > 0.6
        ]
        if diverse:
            recommendations.append(f"For diverse training data, consider: {', '.join(diverse)}")

        if not recommendations:
            recommendations.append(
                "All datasets appear comparable - choose based on specific requirements"
            )

        return recommendations
