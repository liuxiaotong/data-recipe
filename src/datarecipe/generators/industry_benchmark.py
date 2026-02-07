"""Industry benchmark generator for cost comparison.

Generates comparison reports showing how a project compares
to industry standards and similar well-known datasets.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class BenchmarkComparison:
    """Benchmark comparison result."""

    dataset_id: str
    dataset_type: str
    sample_count: int
    total_cost: float
    cost_per_sample: float
    human_percentage: float

    # Industry benchmark
    benchmark_available: bool = False
    benchmark_description: str = ""
    benchmark_cost_range: Dict[str, float] = field(default_factory=dict)  # min, avg, max
    benchmark_human_percentage: float = 0.0

    # Comparison results
    cost_rating: str = ""  # below_average, average, above_average, high
    cost_vs_benchmark: str = ""  # e.g., "+15%" or "-20%"
    cost_explanation: str = ""

    human_rating: str = ""
    human_explanation: str = ""

    # Similar projects
    similar_projects: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class IndustryBenchmarkGenerator:
    """Generate industry benchmark comparison reports."""

    def __init__(self):
        pass

    def generate(
        self,
        dataset_id: str,
        dataset_type: str,
        sample_count: int,
        reproduction_cost: Dict[str, float],
        human_percentage: float,
    ) -> BenchmarkComparison:
        """Generate benchmark comparison.

        Args:
            dataset_id: Dataset identifier
            dataset_type: Type of dataset
            sample_count: Number of samples
            reproduction_cost: Cost breakdown dict
            human_percentage: Human work percentage

        Returns:
            BenchmarkComparison object
        """
        total_cost = reproduction_cost.get("total", 0)
        cost_per_sample = total_cost / sample_count if sample_count > 0 else 0

        comparison = BenchmarkComparison(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            sample_count=sample_count,
            total_cost=total_cost,
            cost_per_sample=round(cost_per_sample, 2),
            human_percentage=human_percentage,
        )

        # Get benchmark data from catalog
        try:
            from datarecipe.knowledge import DatasetCatalog

            catalog = DatasetCatalog()
            benchmark_result = catalog.compare_with_benchmark(
                category=dataset_type,
                sample_count=sample_count,
                total_cost=total_cost,
                human_percentage=human_percentage,
            )

            if benchmark_result.get("available"):
                comparison.benchmark_available = True
                benchmark = benchmark_result.get("benchmark", {})
                comp = benchmark_result.get("comparison", {})

                comparison.benchmark_description = benchmark.get("description", "")
                comparison.benchmark_cost_range = benchmark.get("typical_cost_per_sample", {})
                comparison.benchmark_human_percentage = benchmark.get("avg_human_percentage", 0)

                comparison.cost_rating = comp.get("cost_rating", "")
                comparison.cost_vs_benchmark = comp.get("cost_vs_avg", "")
                comparison.cost_explanation = comp.get("cost_explanation", "")

                comparison.human_rating = comp.get("human_rating", "")
                comparison.human_explanation = comp.get("human_explanation", "")

                comparison.similar_projects = benchmark_result.get("similar_projects", [])

        except Exception:
            pass

        # Generate recommendations
        comparison.recommendations = self._generate_recommendations(comparison)

        return comparison

    def _generate_recommendations(self, comparison: BenchmarkComparison) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []

        if not comparison.benchmark_available:
            recommendations.append("暂无该类型数据集的行业基准数据，建议收集更多同类项目信息")
            return recommendations

        # Cost-based recommendations
        if comparison.cost_rating == "below_average":
            recommendations.append("成本低于行业平均，建议确保质量标准不被降低")
            recommendations.append("考虑增加质量抽检比例以保证数据可用性")
        elif comparison.cost_rating == "high":
            recommendations.append("成本高于行业基准，建议评估以下优化方向：")
            recommendations.append("  - 提高自动化程度，减少人工重复劳动")
            recommendations.append("  - 优化标注流程，提升人效")
            recommendations.append("  - 考虑分阶段实施，优先完成核心子集")
        elif comparison.cost_rating == "above_average":
            recommendations.append("成本略高于平均，但在合理范围内")

        # Human percentage recommendations
        if comparison.human_rating == "more_human":
            recommendations.append("人工比例较高，可探索引入 AI 辅助标注降低成本")
        elif comparison.human_rating == "more_automated":
            recommendations.append("自动化程度高，建议增加人工抽检确保质量")

        # Scale recommendations
        if comparison.sample_count < 100:
            recommendations.append("样本量较小，适合作为试点项目或验证性研究")
        elif comparison.sample_count > 10000:
            recommendations.append("大规模项目，建议分批次滚动交付以控制风险")

        if not recommendations:
            recommendations.append("项目参数符合行业惯例，可按计划推进")

        return recommendations

    def to_markdown(self, comparison: BenchmarkComparison) -> str:
        """Generate benchmark comparison markdown report."""
        lines = []

        # Header
        lines.append(f"# {comparison.dataset_id} 行业基准对比")
        lines.append("")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 数据集类型: {comparison.dataset_type}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Your project summary
        lines.append("## 项目概况")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| 样本数量 | {comparison.sample_count:,} |")
        lines.append(f"| 总成本 | ${comparison.total_cost:,.0f} |")
        lines.append(f"| 单条成本 | ${comparison.cost_per_sample:.2f} |")
        lines.append(f"| 人工占比 | {comparison.human_percentage:.0f}% |")
        lines.append("")

        if not comparison.benchmark_available:
            lines.append("## ⚠️ 基准数据不可用")
            lines.append("")
            lines.append(f"暂无 `{comparison.dataset_type}` 类型数据集的行业基准数据。")
            lines.append("")
        else:
            # Industry benchmark
            lines.append("## 行业基准")
            lines.append("")
            lines.append(f"**数据类型**: {comparison.benchmark_description}")
            lines.append("")

            cost_range = comparison.benchmark_cost_range
            lines.append("### 单条成本基准")
            lines.append("")
            lines.append("```")
            lines.append(f"最低: ${cost_range.get('min', 0):.2f}/条")
            lines.append(f"平均: ${cost_range.get('avg', 0):.2f}/条")
            lines.append(f"最高: ${cost_range.get('max', 0):.2f}/条")
            lines.append("```")
            lines.append("")

            lines.append(f"**行业平均人工比例**: {comparison.benchmark_human_percentage:.0f}%")
            lines.append("")

            # Comparison visualization
            lines.append("---")
            lines.append("")
            lines.append("## 对比分析")
            lines.append("")

            # Cost comparison bar
            min_cost = cost_range.get("min", 0)
            avg_cost = cost_range.get("avg", 1)
            max_cost = cost_range.get("max", avg_cost * 2)

            lines.append("### 成本定位")
            lines.append("")
            lines.append("```")

            # Create a simple ASCII bar chart
            range_width = max_cost - min_cost if max_cost > min_cost else 1
            your_pos = (comparison.cost_per_sample - min_cost) / range_width

            # Clamp position to 0-1
            your_pos = max(0, min(1, your_pos))

            bar_width = 40
            your_marker = int(your_pos * bar_width)

            bar = list("─" * bar_width)
            if 0 <= your_marker < bar_width:
                bar[your_marker] = "◆"

            lines.append(f"低 │{''.join(bar)}│ 高")
            lines.append(
                f"   ${min_cost:.1f}         ◆ 您的项目: ${comparison.cost_per_sample:.2f}         ${max_cost:.1f}"
            )
            lines.append("```")
            lines.append("")

            # Rating indicators
            if comparison.cost_rating == "below_average":
                cost_icon = "🟢"
            elif comparison.cost_rating == "average":
                cost_icon = "🟢"
            elif comparison.cost_rating == "above_average":
                cost_icon = "🟡"
            else:
                cost_icon = "🔴"

            lines.append(f"**成本评级**: {cost_icon} {comparison.cost_explanation}")
            lines.append("")
            lines.append(f"**与行业平均差异**: {comparison.cost_vs_benchmark}")
            lines.append("")

            if comparison.human_rating:
                lines.append(f"**人工比例评估**: {comparison.human_explanation}")
                lines.append("")

            # Similar projects
            if comparison.similar_projects:
                lines.append("---")
                lines.append("")
                lines.append("## 类似项目参考")
                lines.append("")
                lines.append("| 数据集 | 规模 | 估计成本 |")
                lines.append("|--------|------|----------|")
                for proj in comparison.similar_projects:
                    name = proj.get("name", "Unknown")
                    size = proj.get("size", 0)
                    cost = proj.get("estimated_cost", 0)
                    if cost > 0:
                        lines.append(f"| {name} | {size:,} | ${cost:,.0f} |")
                lines.append("")

        # Recommendations
        lines.append("---")
        lines.append("")
        lines.append("## 建议")
        lines.append("")
        for rec in comparison.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("> 基准数据来源于公开信息和行业调研，仅供参考")

        return "\n".join(lines)

    def to_dict(self, comparison: BenchmarkComparison) -> dict:
        """Convert comparison to dictionary."""
        return {
            "dataset_id": comparison.dataset_id,
            "dataset_type": comparison.dataset_type,
            "your_project": {
                "sample_count": comparison.sample_count,
                "total_cost": comparison.total_cost,
                "cost_per_sample": comparison.cost_per_sample,
                "human_percentage": comparison.human_percentage,
            },
            "benchmark": {
                "available": comparison.benchmark_available,
                "description": comparison.benchmark_description,
                "cost_range": comparison.benchmark_cost_range,
                "avg_human_percentage": comparison.benchmark_human_percentage,
            },
            "comparison": {
                "cost_rating": comparison.cost_rating,
                "cost_vs_benchmark": comparison.cost_vs_benchmark,
                "cost_explanation": comparison.cost_explanation,
                "human_rating": comparison.human_rating,
                "human_explanation": comparison.human_explanation,
            },
            "similar_projects": comparison.similar_projects,
            "recommendations": comparison.recommendations,
        }
