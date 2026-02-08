"""Dataset comparison and reporting."""

from dataclasses import dataclass, field
from typing import Optional

from datarecipe.analyzer import DatasetAnalyzer
from datarecipe.cost_calculator import CostBreakdown, CostCalculator
from datarecipe.quality_metrics import QualityAnalyzer, QualityReport
from datarecipe.schema import Recipe


@dataclass
class DatasetMetrics:
    """Collected metrics for a single dataset."""

    dataset_id: str
    recipe: Recipe
    cost: Optional[CostBreakdown] = None
    quality: Optional[QualityReport] = None


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
        header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
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

        # Best for section
        if self.best_for:
            lines.append("")
            lines.append("Recommendations:")
            for use_case, dataset in self.best_for.items():
                lines.append(f"  {use_case}: {dataset}")

        return "\n".join(lines)


class DatasetComparator:
    """Compare multiple datasets."""

    def __init__(
        self,
        include_cost: bool = True,
        include_quality: bool = False,
        quality_sample_size: int = 500,
    ):
        """Initialize the comparator.

        Args:
            include_cost: Whether to include cost estimation
            include_quality: Whether to include quality analysis
            quality_sample_size: Sample size for quality analysis
        """
        self.include_cost = include_cost
        self.include_quality = include_quality
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

            metrics.append(
                DatasetMetrics(
                    dataset_id=dataset_id,
                    recipe=recipe,
                    cost=cost,
                    quality=quality,
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

        return ComparisonReport(
            datasets=datasets,
            metrics=metrics,
            cost_comparison=cost_comparison,
            quality_comparison=quality_comparison,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            best_for=best_for,
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
                premium_models = ["gpt-4", "claude-3-opus", "gemini-ultra"]
                if any(pm in str(m.recipe.teacher_models).lower() for pm in premium_models):
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
